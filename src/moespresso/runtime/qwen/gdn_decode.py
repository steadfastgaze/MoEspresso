"""Exact decode fusion for the Qwen-family gated-delta convolution state.

The pinned Qwen 3.5 gated-delta block concatenates a three-row recurrent
convolution state with the projected QKV row, copies the newest three rows back
to the cache, and applies a four-tap depthwise convolution.  The eligible
decode route emits the shifted state and convolution output in one Metal
dispatch.  It preserves the stock serial accumulation order and leaves all
other projections, normalization, recurrence, and cache advancement on the
pinned MLX LM path.

The route is on by default after passing real-package bitwise logits and cache
checks plus a served speed A/B. Set
``MOESPRESSO_QWEN_GDN_CONV_STATE_FUSED=0`` to disable it. Every call that does
not match the served Ornith decode contract delegates to the untouched inner
module.

The eligible decode path also folds the fixed Q/K scale factors into the
RMSNorm weights. Set ``MOESPRESSO_QWEN_GDN_RMS_SCALE_FUSED=0`` to restore the
separate scalar multiplications.
"""

from __future__ import annotations

import os
from functools import cache as memoize
from importlib.metadata import PackageNotFoundError, version

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update


_QWEN_GDN_CONV_STATE_FUSED = (
    os.environ.get("MOESPRESSO_QWEN_GDN_CONV_STATE_FUSED", "1") == "1"
)
_QWEN_GDN_RMS_SCALE_FUSED = (
    os.environ.get("MOESPRESSO_QWEN_GDN_RMS_SCALE_FUSED", "1") == "1"
)
_CERTIFIED_MLX_LM_VERSION = "0.31.3"
try:
    _MLX_LM_VERSION = version("mlx-lm")
except PackageNotFoundError:
    _MLX_LM_VERSION = None

_HIDDEN_SIZE = 2048
_CONV_DIM = 8192
_STATE_ROWS = 3
_KERNEL_SIZE = 4
_NUM_K_HEADS = 16
_NUM_V_HEADS = 32
_HEAD_K_DIM = 128
_HEAD_V_DIM = 128
_KEY_DIM = _NUM_K_HEADS * _HEAD_K_DIM
_VALUE_DIM = _NUM_V_HEADS * _HEAD_V_DIM


# This loop follows the serial accumulation contract of MLX's depthwise
# convolution kernel.  Keeping one scalar thread per channel also makes the
# state shift independent across channels.
_FUSED_SOURCE = r"""
    uint channel = thread_position_in_grid.x;
    if (channel >= C) {
        return;
    }

    float qkv_value = static_cast<float>(qkv[channel]);
    float acc = 0.0f;
    for (int tap = 0; tap < K; ++tap) {
        float value = tap < STATE_ROWS
            ? static_cast<float>(state[tap * C + channel])
            : qkv_value;
        acc += value * static_cast<float>(weight[channel * K + tap]);
    }
    conv_out[channel] = acc;

    new_state[channel] = state[C + channel];
    new_state[C + channel] = state[2 * C + channel];
    new_state[2 * C + channel] = qkv_value;
"""


def gdn_conv_state_fusion_enabled() -> bool:
    """Return whether this process enables the guarded decode fusion."""

    return _QWEN_GDN_CONV_STATE_FUSED


def gdn_rms_scale_fusion_enabled() -> bool:
    """Return whether RMSNorm weights absorb the Q/K decode scales."""

    return _QWEN_GDN_RMS_SCALE_FUSED


def _kernel_available() -> bool:
    return bool(
        mx.metal.is_available()
        and getattr(getattr(mx, "fast", None), "metal_kernel", None) is not None
    )


def _mlx_lm_compatible() -> bool:
    return _MLX_LM_VERSION == _CERTIFIED_MLX_LM_VERSION


@memoize
def _fused_kernel():
    if not _kernel_available():
        raise RuntimeError("the gated-delta convolution fusion requires Metal")
    return mx.fast.metal_kernel(
        name="moespresso_qwen_gdn_conv_state_fused",
        input_names=["state", "qkv", "weight"],
        output_names=["conv_out", "new_state"],
        source=_FUSED_SOURCE,
    )


def _fused_conv_state(
    state: mx.array,
    qkv: mx.array,
    weight: mx.array,
) -> tuple[mx.array, mx.array]:
    return tuple(
        _fused_kernel()(
            inputs=[state, qkv, weight],
            template=[
                ("C", _CONV_DIM),
                ("K", _KERNEL_SIZE),
                ("STATE_ROWS", _STATE_ROWS),
            ],
            output_shapes=[
                (1, 1, _CONV_DIM),
                (1, _STATE_ROWS, _CONV_DIM),
            ],
            output_dtypes=[mx.float32, mx.float32],
            grid=(_CONV_DIM, 1, 1),
            threadgroup=(256, 1, 1),
        )
    )


@memoize
def _rms_scale_weights() -> tuple[mx.array, mx.array]:
    inv_scale = _HEAD_K_DIM**-0.5
    return (
        mx.full((_HEAD_K_DIM,), inv_scale**2, dtype=mx.float32),
        mx.full((_HEAD_K_DIM,), inv_scale, dtype=mx.float32),
    )


class FusedDecodeGatedDeltaNet(nn.Module):
    """Wrap one gated-delta block with a fail-closed decode-only fusion."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.fused_calls = 0
        self.rms_scale_fused_calls = 0
        self.fallback_disabled = 0
        self.fallback_training = 0
        self.fallback_input = 0
        self.fallback_mask = 0
        self.fallback_cache = 0
        self.fallback_geometry = 0
        self.fallback_dtype = 0
        self.fallback_kernel = 0
        self.fallback_qkv = 0

    def _fallback(self, counter: str, inputs, mask, cache):
        setattr(self, counter, int(getattr(self, counter)) + 1)
        return self.inner(inputs, mask=mask, cache=cache)

    def _eligibility_failure(self, inputs, mask, cache) -> str | None:
        inner = self.inner
        if not _QWEN_GDN_CONV_STATE_FUSED:
            return "fallback_disabled"
        if inner.training:
            return "fallback_training"
        if tuple(inputs.shape) != (1, 1, _HIDDEN_SIZE):
            return "fallback_input"
        if mask is not None:
            return "fallback_mask"
        if (
            type(cache) is not ArraysCache
            or len(cache.cache) != 2
            or cache.lengths is not None
            or cache.left_padding is not None
            or cache[0] is None
        ):
            return "fallback_cache"
        if (
            getattr(inner, "sharding_group", None) is not None
            or getattr(inner, "hidden_size", None) != _HIDDEN_SIZE
            or getattr(inner, "conv_dim", None) != _CONV_DIM
            or getattr(inner, "conv_kernel_size", None) != _KERNEL_SIZE
            or getattr(inner, "num_k_heads", None) != _NUM_K_HEADS
            or getattr(inner, "num_v_heads", None) != _NUM_V_HEADS
            or getattr(inner, "head_k_dim", None) != _HEAD_K_DIM
            or getattr(inner, "head_v_dim", None) != _HEAD_V_DIM
            or getattr(inner, "key_dim", None) != _KEY_DIM
            or getattr(inner, "value_dim", None) != _VALUE_DIM
        ):
            return "fallback_geometry"

        conv = getattr(inner, "conv1d", None)
        weight = getattr(conv, "weight", None)
        if (
            conv is None
            or weight is None
            or tuple(weight.shape) != (_CONV_DIM, _KERNEL_SIZE, 1)
            or getattr(conv, "groups", None) != _CONV_DIM
            or getattr(conv, "stride", None) != 1
            or getattr(conv, "padding", None) != 0
            or getattr(conv, "dilation", None) != 1
            or "bias" in conv
        ):
            return "fallback_geometry"
        if (
            inputs.dtype != mx.float32
            or tuple(cache[0].shape) != (1, _STATE_ROWS, _CONV_DIM)
            or cache[0].dtype != mx.float32
            or weight.dtype != mx.float32
        ):
            return "fallback_dtype"
        if not _kernel_available():
            return "fallback_kernel"
        return None

    def __call__(self, inputs, mask=None, cache=None):
        """Run the fused served decode contract or the untouched stock block."""

        failure = self._eligibility_failure(inputs, mask, cache)
        if failure is not None:
            return self._fallback(failure, inputs, mask, cache)

        inner = self.inner
        B, S, _ = inputs.shape
        qkv = inner.in_proj_qkv(inputs)
        if tuple(qkv.shape) != (1, 1, _CONV_DIM) or qkv.dtype != mx.bfloat16:
            return self._fallback("fallback_qkv", inputs, mask, cache)

        z = inner.in_proj_z(inputs).reshape(
            B, S, inner.num_v_heads, inner.head_v_dim
        )
        b = inner.in_proj_b(inputs)
        a = inner.in_proj_a(inputs)

        conv_out, new_conv_state = _fused_conv_state(
            cache[0], qkv, inner.conv1d.weight
        )
        cache[0] = new_conv_state
        conv_out = nn.silu(conv_out)

        q, k, v = [
            tensor.reshape(B, S, heads, dimension)
            for tensor, heads, dimension in zip(
                mx.split(conv_out, [inner.key_dim, 2 * inner.key_dim], -1),
                [inner.num_k_heads, inner.num_k_heads, inner.num_v_heads],
                [inner.head_k_dim, inner.head_k_dim, inner.head_v_dim],
            )
        ]

        state = cache[1]
        inv_scale = k.shape[-1] ** -0.5
        if _QWEN_GDN_RMS_SCALE_FUSED:
            q_weight, k_weight = _rms_scale_weights()
            q = mx.fast.rms_norm(q, q_weight, 1e-6)
            k = mx.fast.rms_norm(k, k_weight, 1e-6)
            self.rms_scale_fused_calls += 1
        else:
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            inner.A_log,
            inner.dt_bias,
            state,
            mask,
            use_kernel=not inner.training,
        )

        cache[1] = state
        cache.advance(S)
        out = inner.norm(out, z)
        out = inner.out_proj(out.reshape(B, S, -1))
        self.fused_calls += 1
        return out


def _iter_gated_delta_layers(model):
    for path in (
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("layers",),
    ):
        obj = model
        for attribute in path:
            obj = getattr(obj, attribute, None)
            if obj is None:
                break
        if obj is not None:
            layers = obj
            break
    else:
        return

    for layer in layers:
        if not getattr(layer, "is_linear", False):
            continue
        if getattr(layer, "linear_attn", None) is None:
            continue
        yield layer


def install_fused_gdn_decode(model) -> int:
    """Install the guarded fusion on every gated-delta layer once."""

    if (
        not _QWEN_GDN_CONV_STATE_FUSED
        or not _mlx_lm_compatible()
        or not _kernel_available()
    ):
        return 0
    installed = 0
    for layer in _iter_gated_delta_layers(model):
        if isinstance(layer.linear_attn, FusedDecodeGatedDeltaNet):
            continue
        wrapped = FusedDecodeGatedDeltaNet(layer.linear_attn)
        wrapped.eval()
        layer.linear_attn = wrapped
        installed += 1
    return installed


def fused_gdn_decode_stats(model) -> dict[str, int]:
    """Aggregate engagement and fallback counters across wrapped layers."""

    keys = (
        "fused_calls",
        "rms_scale_fused_calls",
        "fallback_disabled",
        "fallback_training",
        "fallback_input",
        "fallback_mask",
        "fallback_cache",
        "fallback_geometry",
        "fallback_dtype",
        "fallback_kernel",
        "fallback_qkv",
    )
    stats = {"wrapped_layers": 0, **{key: 0 for key in keys}}
    for layer in _iter_gated_delta_layers(model):
        module = layer.linear_attn
        if not isinstance(module, FusedDecodeGatedDeltaNet):
            continue
        stats["wrapped_layers"] += 1
        for key in keys:
            stats[key] += int(getattr(module, key))
    return stats
