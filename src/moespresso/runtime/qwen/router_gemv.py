"""Exact mixed-storage GEMV for the Ornith decode router.

The forty Ornith router matrices are stored as F32 in the package, but every
loaded value lies exactly on the BF16 lattice.  The guarded decode route keeps
the original F32 ``nn.Linear`` for prefill and every fallback, materializes a
BF16 shadow at load, and reads that shadow while preserving the stock F32 lane
assignment, accumulation order, reduction tree, and output dtype.

The Metal loop and launch geometry derive from Apple MLX v0.31.2 commit
``68cf2fddd8de5edd8ab3d926391772b2e2cedad8`` under MIT, specifically
``mlx/backend/metal/kernels/gemv.metal::GEMVKernel::run`` and the specialization
selected by ``mlx/backend/metal/matmul.cpp::gemv_axbpy``.  MoEspresso fixes the
upstream non-transposed F32 specialization to the 256-by-2048 router decode
shape and changes only the matrix load to BF16 followed by explicit F32
conversion.  See ``THIRD-PARTY-NOTICES`` and ``LICENSE-MIT``.

The route is on by default. Set
``MOESPRESSO_QWEN_ROUTER_BF16_F32_GEMV=0`` to keep every router on its original
F32 linear. Installation is transactional: all forty weights must pass an
exact F32-to-BF16-to-F32 bit comparison before any layer is wrapped.
"""

from __future__ import annotations

import os
from functools import cache as memoize
from importlib.metadata import PackageNotFoundError, version

import mlx.core as mx
import mlx.nn as nn


_QWEN_ROUTER_BF16_F32_GEMV = os.environ.get("MOESPRESSO_QWEN_ROUTER_BF16_F32_GEMV", "1") == "1"
_CERTIFIED_MLX_VERSION = "0.31.2"
_CERTIFIED_MLX_LM_VERSION = "0.31.3"
_MODEL_TYPE = "qwen3_5_moe"
_LAYERS = 40
_IN_FEATURES = 2048
_OUT_FEATURES = 256
_SHADOW_BYTES = _LAYERS * _IN_FEATURES * _OUT_FEATURES * 2

try:
    _MLX_VERSION = version("mlx")
except PackageNotFoundError:
    _MLX_VERSION = None
try:
    _MLX_LM_VERSION = version("mlx-lm")
except PackageNotFoundError:
    _MLX_LM_VERSION = None


# Copyright © 2023-2024 Apple Inc.
# Modification notice: this is MLX v0.31.2's
# gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0 arithmetic tree, fixed to
# the Ornith router geometry. The matrix buffer is BF16; every matrix value and
# vector operand is explicitly promoted to F32 before the unchanged multiply,
# serial lane accumulation, and shuffle-down reduction.
_SOURCE = r"""
    const int simd_gid = int(simdgroup_index_in_threadgroup);
    const int simd_lid = int(thread_index_in_simdgroup);

    thread float result[4] = {0.0f};
    thread float inter[4];
    thread float v_coeff[4];

    int bn = simd_lid * 4;
    const int out_row = int(threadgroup_position_in_grid.x) * 16 + simd_gid * 4;
    auto mat = weight + out_row * 2048;

    for (int i = 0; i < 16; ++i) {
        #pragma clang loop unroll(full)
        for (int tn = 0; tn < 4; ++tn) {
            v_coeff[tn] = float(inputs[bn + tn]);
        }

        int mat_offset = 0;
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < 4; ++tm) {
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < 4; ++tn) {
                inter[tn] = float(mat[mat_offset + bn + tn]);
            }
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < 4; ++tn) {
                result[tm] += inter[tn] * v_coeff[tn];
            }
            mat_offset += 2048;
        }
        bn += 128;
    }

    #pragma clang loop unroll(full)
    for (int tm = 0; tm < 4; ++tm) {
        #pragma clang loop unroll(full)
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
        }
    }

    if (simd_lid == 0) {
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < 4; ++tm) {
            output[out_row + tm] = result[tm];
        }
    }
"""


def _kernel_available() -> bool:
    return bool(
        mx.metal.is_available()
        and getattr(getattr(mx, "fast", None), "metal_kernel", None) is not None
    )


def _versions_compatible() -> bool:
    return _MLX_VERSION == _CERTIFIED_MLX_VERSION and _MLX_LM_VERSION == _CERTIFIED_MLX_LM_VERSION


def router_bf16_f32_resident_bytes(config: dict) -> int:
    """Return the persistent capacity reservation for an eligible config."""

    if (
        not isinstance(config, dict)
        or not _QWEN_ROUTER_BF16_F32_GEMV
        or not _versions_compatible()
        or not _kernel_available()
        or config.get("model_type") != _MODEL_TYPE
    ):
        return 0
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        return 0
    geometry = (
        ("num_hidden_layers", _LAYERS),
        ("hidden_size", _IN_FEATURES),
        ("num_experts", _OUT_FEATURES),
        ("num_experts_per_tok", 8),
    )
    try:
        if any(int(text_config.get(name, -1)) != expected for name, expected in geometry):
            return 0
    except (TypeError, ValueError):
        return 0
    return _SHADOW_BYTES


@memoize
def _router_kernel():
    if not _kernel_available():
        raise RuntimeError("the mixed-storage router GEMV requires Metal")
    return mx.fast.metal_kernel(
        name="moespresso_qwen_router_bf16_f32_gemv_256x2048",
        input_names=["weight", "inputs"],
        output_names=["output"],
        source=_SOURCE,
    )


def _mixed_router_gemv(weight_bf16: mx.array, inputs_f32: mx.array) -> mx.array:
    (output,) = _router_kernel()(
        inputs=[weight_bf16, mx.contiguous(inputs_f32)],
        output_shapes=[(1, 1, _OUT_FEATURES)],
        output_dtypes=[mx.float32],
        grid=(16 * 32, 1, 4),
        threadgroup=(32, 1, 4),
    )
    return output


class BF16F32RouterLinear(nn.Module):
    """Wrap one router with a decode-only BF16-read/F32-compute GEMV."""

    def __init__(self, inner: nn.Linear, weight_bf16: mx.array):
        super().__init__()
        self.inner = inner
        self.weight_bf16 = weight_bf16
        self._validated_weight_id = id(inner.weight)
        self.validated_layers = 1
        self.kernel_calls = 0
        self.fallback_disabled = 0
        self.fallback_training = 0
        self.fallback_input_shape = 0
        self.fallback_input_dtype = 0
        self.fallback_weight_contract = 0
        self.fallback_kernel = 0

    @property
    def weight(self) -> mx.array:
        """Expose the original F32 weight to generic linear consumers."""

        return self.inner.weight

    @property
    def bias(self) -> mx.array | None:
        """Expose the original bias contract to generic linear consumers."""

        return getattr(self.inner, "bias", None)

    def _eligibility_failure(self, inputs: mx.array) -> str | None:
        if not _QWEN_ROUTER_BF16_F32_GEMV:
            return "fallback_disabled"
        if self.training or self.inner.training:
            return "fallback_training"
        if tuple(inputs.shape) != (1, 1, _IN_FEATURES):
            return "fallback_input_shape"
        if inputs.dtype != mx.float32:
            return "fallback_input_dtype"
        if (
            tuple(self.weight_bf16.shape) != (_OUT_FEATURES, _IN_FEATURES)
            or self.weight_bf16.dtype != mx.bfloat16
            or id(self.inner.weight) != self._validated_weight_id
            or tuple(self.inner.weight.shape) != (_OUT_FEATURES, _IN_FEATURES)
            or self.inner.weight.dtype != mx.float32
        ):
            return "fallback_weight_contract"
        if not _kernel_available() or not _versions_compatible():
            return "fallback_kernel"
        return None

    def __call__(self, inputs: mx.array) -> mx.array:
        """Use the exact decode kernel or delegate to the original F32 linear."""

        failure = self._eligibility_failure(inputs)
        if failure is not None:
            setattr(self, failure, int(getattr(self, failure)) + 1)
            return self.inner(inputs)
        self.kernel_calls += 1
        return _mixed_router_gemv(self.weight_bf16, inputs)


def _model_layers(model) -> list:
    for path in (
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("layers",),
    ):
        value = model
        for attribute in path:
            value = getattr(value, attribute, None)
            if value is None:
                break
        if value is not None:
            return list(value)
    return []


def _router_slots(model) -> list[tuple[object, object]]:
    slots = []
    for layer in _model_layers(model):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        if gate is not None:
            slots.append((mlp, gate))
    return slots


def _original_router_contract(slots: list[tuple[object, object]]) -> bool:
    if len(slots) != _LAYERS:
        return False
    for mlp, gate in slots:
        if type(gate) is not nn.Linear:
            return False
        if (
            "bias" in gate
            or getattr(gate, "sharding_group", None) is not None
            or tuple(gate.weight.shape) != (_OUT_FEATURES, _IN_FEATURES)
            or gate.weight.dtype != mx.float32
            or getattr(mlp, "top_k", None) != 8
            or getattr(mlp, "norm_topk_prob", None) is not True
        ):
            return False
    return True


def _bf16_shadows_exact(gates: list[nn.Linear]) -> list[mx.array] | None:
    shadows = []
    for gate in gates:
        shadow = mx.contiguous(gate.weight.astype(mx.bfloat16))
        recovered = shadow.astype(mx.float32)
        mx.eval(shadow, recovered)
        mismatches = mx.sum(mx.view(gate.weight, mx.uint32) != mx.view(recovered, mx.uint32))
        if int(mismatches.item()) != 0:
            return None
        del recovered, mismatches
        shadows.append(shadow)
    return shadows


def install_router_bf16_f32_gemv(model) -> int:
    """Install all forty exact router wrappers, or leave the model untouched."""

    if (
        not _QWEN_ROUTER_BF16_F32_GEMV
        or getattr(model, "model_type", None) != _MODEL_TYPE
        or not _versions_compatible()
        or not _kernel_available()
    ):
        return 0

    slots = _router_slots(model)
    wrapped = [isinstance(gate, BF16F32RouterLinear) for _, gate in slots]
    if wrapped and all(wrapped) and len(wrapped) == _LAYERS:
        return 0
    if any(wrapped):
        raise RuntimeError("partial mixed-storage router installation")
    if not _original_router_contract(slots):
        return 0

    gates = [gate for _, gate in slots]
    shadows = _bf16_shadows_exact(gates)
    if shadows is None:
        return 0
    wrappers = [
        BF16F32RouterLinear(gate, shadow) for gate, shadow in zip(gates, shadows, strict=True)
    ]
    for wrapper in wrappers:
        wrapper.eval()

    installed = 0
    try:
        for (mlp, _gate), wrapper in zip(slots, wrappers, strict=True):
            mlp.gate = wrapper
            installed += 1
    except Exception:
        for mlp, gate in slots[:installed]:
            mlp.gate = gate
        raise
    return installed


def router_bf16_f32_stats(model) -> dict[str, int]:
    """Aggregate engagement and fallback counters over installed routers."""

    keys = (
        "validated_layers",
        "kernel_calls",
        "fallback_disabled",
        "fallback_training",
        "fallback_input_shape",
        "fallback_input_dtype",
        "fallback_weight_contract",
        "fallback_kernel",
    )
    stats = {"wrapped_layers": 0, **{key: 0 for key in keys}}
    for _mlp, gate in _router_slots(model):
        if not isinstance(gate, BF16F32RouterLinear):
            continue
        stats["wrapped_layers"] += 1
        for key in keys:
            stats[key] += int(getattr(gate, key))
    return stats
