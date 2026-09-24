"""MLX implementations of released Qwen4-Exp residual and norm primitives."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn


_HC_CALL_COUNTS = {
    "fused_projection_reads": 0,
    "generic_reads": 0,
    "native_norm_reads": 0,
    "native_folded_writes": 0,
}


def qwen4_hc_call_counts() -> dict[str, int]:
    """Return gated-residual read engagement counters."""
    return dict(_HC_CALL_COUNTS)


def _released_kquant_linear(module: Any, codec: str) -> bool:
    return (
        getattr(module, "mode", None) == "kquant"
        and getattr(module, "kquant_type", None) == codec
        and "bias" not in module
        and isinstance(getattr(module, "weight", None), mx.array)
        and isinstance(getattr(module, "scales", None), mx.array)
    )


def _qwen4_hc_native_projection_contract(
    module: "Qwen4GatedResidual",
    values: mx.array,
) -> bool:
    injection = module.block_inject_weight
    return (
        values.dtype == mx.bfloat16
        and values.size // values.shape[-1] == 1
        and module.hidden_size == 2560
        and module.branch_count == 4
        and module.lowrank_size == 320
        and module.expanded_size == 10240
        and injection is not None
        and _released_kquant_linear(module.input_mix_weight_down, "q6_k")
        and _released_kquant_linear(module.input_mix_weight_up, "q8_0")
        and _released_kquant_linear(injection, "q6_k")
    )


def _qwen4_hc_native_projection_read(
    module: "Qwen4GatedResidual",
    normalized: mx.array,
) -> tuple[mx.array, mx.array]:
    try:
        from mlx_kquant import qwen4_hc_epilogue, qwen4_hc_front
    except ImportError as exc:
        raise RuntimeError("mlx-kquant lacks the Qwen gated-residual projection kernels") from exc
    injection = module.block_inject_weight
    if injection is None:
        raise RuntimeError("native gated-residual projection read requires injection")
    lowrank, injection_weights = qwen4_hc_front(
        normalized,
        module.input_mix_weight_down.weight,
        module.input_mix_weight_down.scales,
        injection.weight,
        injection.scales,
    )
    mixed = qwen4_hc_epilogue(
        lowrank,
        module.input_mix_weight_up.weight,
        module.input_mix_weight_up.scales,
        normalized,
    )
    return mixed, injection_weights


def _qwen4_hc_native_read_contract(
    module: "Qwen4GatedResidual",
    values: mx.array,
    pending_output: mx.array | None,
    pending_injection: mx.array | None,
) -> bool:
    if not _qwen4_hc_native_symbols_available():
        return False
    if (pending_output is None) != (pending_injection is None):
        return False
    norm_weight = module.hc_norm.weight
    if (
        not _qwen4_hc_native_projection_contract(module, values)
        or not isinstance(norm_weight, mx.array)
        or norm_weight.dtype != mx.bfloat16
        or norm_weight.size != module.expanded_size
    ):
        return False
    if pending_output is None or pending_injection is None:
        return True
    return (
        pending_output.dtype == mx.bfloat16
        and pending_output.size == module.hidden_size
        and pending_injection.dtype == mx.bfloat16
        and pending_injection.size == module.branch_count
    )


@lru_cache(maxsize=1)
def _qwen4_hc_native_symbols_available() -> bool:
    try:
        import mlx_kquant as kq
    except ImportError:
        return False
    return all(
        hasattr(kq, name) for name in ("qwen4_hc_norm", "qwen4_hc_front", "qwen4_hc_epilogue")
    )


def _qwen4_hc_native_read(
    module: "Qwen4GatedResidual",
    values: mx.array,
    pending_output: mx.array | None,
    pending_injection: mx.array | None,
) -> tuple[mx.array, mx.array, mx.array]:
    try:
        from mlx_kquant import qwen4_hc_epilogue, qwen4_hc_front, qwen4_hc_norm
    except ImportError as exc:
        raise RuntimeError("mlx-kquant lacks the Qwen gated-residual kernels") from exc
    injection = module.block_inject_weight
    if injection is None:
        raise RuntimeError("native gated-residual read requires injection")
    normalized, updated = qwen4_hc_norm(
        values,
        module.hc_norm.weight,
        pending_output,
        pending_injection,
        eps=float(module.hc_norm.eps),
    )
    lowrank, injection_weights = qwen4_hc_front(
        normalized,
        module.input_mix_weight_down.weight,
        module.input_mix_weight_down.scales,
        injection.weight,
        injection.scales,
    )
    mixed = qwen4_hc_epilogue(
        lowrank,
        module.input_mix_weight_up.weight,
        module.input_mix_weight_up.scales,
        normalized,
    )
    return mixed, updated, injection_weights


def qwen4_projection_compute_dtype(module: object) -> mx.Dtype:
    """Return the activation dtype independently of packed weight storage."""
    if getattr(module, "mode", None) in {"iqk_dense", "kquant"}:
        return mx.bfloat16
    weight = getattr(module, "weight", None)
    if not isinstance(weight, mx.array):
        raise ValueError("Qwen4 projection has no array weight")
    return weight.dtype


class Qwen4RMSNorm(nn.Module):
    """Zero-centered RMSNorm with optional independent feature groups."""

    def __init__(self, dimensions: int, *, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if group_size is not None and (group_size <= 0 or dimensions % group_size):
            raise ValueError("group_size must divide dimensions")
        self.dimensions = dimensions
        self.group_size = group_size
        self.eps = eps
        self.weight = mx.zeros((dimensions,))

    def __call__(self, values: mx.array) -> mx.array:
        if values.shape[-1] != self.dimensions:
            raise ValueError("input width does not match RMSNorm dimensions")
        source_dtype = values.dtype
        source = values.astype(mx.float32)
        if self.group_size is not None:
            source = source.reshape(*source.shape[:-1], -1, self.group_size)
        normalized = source * mx.rsqrt(
            mx.mean(mx.square(source), axis=-1, keepdims=True) + self.eps
        )
        if self.group_size is not None:
            normalized = normalized.reshape(values.shape)
        return (normalized * (1 + self.weight.astype(mx.float32))).astype(source_dtype)


class Qwen4SigmoidRMSNormGated(nn.Module):
    """GDN output normalization followed by the released sigmoid gate."""

    def __init__(self, dimensions: int, *, eps: float = 1e-6):
        super().__init__()
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.eps = eps
        self.weight = mx.ones((dimensions,))

    def __call__(self, values: mx.array, gate: mx.array) -> mx.array:
        if values.shape != gate.shape or values.shape[-1] != self.dimensions:
            raise ValueError("values and gate must match the configured dimensions")
        source_dtype = values.dtype
        source = values.astype(mx.float32)
        normalized = source * mx.rsqrt(
            mx.mean(mx.square(source), axis=-1, keepdims=True) + self.eps
        )
        weighted = self.weight * normalized.astype(source_dtype)
        return (weighted * mx.sigmoid(gate.astype(mx.float32))).astype(source_dtype)


class Qwen4GatedResidual(nn.Module):
    """Read and optional write controls for the four-branch residual stream."""

    def __init__(
        self,
        hidden_size: int,
        branch_count: int,
        lowrank_size: int,
        *,
        combine: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_size <= 0 or branch_count <= 0 or lowrank_size <= 0:
            raise ValueError("residual geometry must be positive")
        self.hidden_size = hidden_size
        self.branch_count = branch_count
        self.lowrank_size = lowrank_size
        self.expanded_size = hidden_size * branch_count
        self.hc_norm = Qwen4RMSNorm(self.expanded_size, group_size=hidden_size, eps=eps)
        self.input_mix_weight_down = nn.Linear(self.expanded_size, lowrank_size, bias=False)
        self.input_mix_weight_up = nn.Linear(lowrank_size, self.expanded_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(self.expanded_size, branch_count, bias=False) if combine else None
        )

    def _read(
        self,
        hyper_input: mx.array,
        *,
        pending_output: mx.array | None = None,
        pending_injection: mx.array | None = None,
    ):
        if hyper_input.shape[-1] != self.expanded_size:
            raise ValueError("input width does not match expanded residual geometry")
        if (pending_output is None) != (pending_injection is None):
            raise ValueError("pending output and injection must be supplied together")
        if _qwen4_hc_native_read_contract(
            self,
            hyper_input,
            pending_output,
            pending_injection,
        ):
            result = _qwen4_hc_native_read(
                self,
                hyper_input,
                pending_output,
                pending_injection,
            )
            _HC_CALL_COUNTS["fused_projection_reads"] += 1
            _HC_CALL_COUNTS["native_norm_reads"] += 1
            if pending_output is not None:
                _HC_CALL_COUNTS["native_folded_writes"] += 1
            return result
        updated = (
            hyper_input
            if pending_output is None or pending_injection is None
            else gated_residual_write(hyper_input, pending_output, pending_injection)
        )
        normalized = self.hc_norm(updated)
        if _qwen4_hc_native_projection_contract(
            self, updated
        ):
            mixed, injection = _qwen4_hc_native_projection_read(self, normalized)
            _HC_CALL_COUNTS["fused_projection_reads"] += 1
            return mixed, updated, injection
        _HC_CALL_COUNTS["generic_reads"] += 1
        lowrank = nn.silu(self.input_mix_weight_down(normalized) / self.branch_count)
        mix = mx.sigmoid(self.input_mix_weight_up(lowrank)).reshape(
            *hyper_input.shape[:-1], self.branch_count, self.hidden_size
        )
        streams = normalized.reshape(*hyper_input.shape[:-1], self.branch_count, self.hidden_size)
        mixed = mx.mean(mix * streams, axis=-2)
        if self.block_inject_weight is None:
            return mixed
        injection = 2 * mx.sigmoid(self.block_inject_weight(normalized) / self.branch_count)
        return mixed, updated, injection

    def __call__(self, hyper_input: mx.array):
        return self._read(hyper_input)

    def read_with_pending(
        self,
        hyper_input: mx.array,
        block_output: mx.array,
        injection_weights: mx.array,
    ):
        """Apply a pending residual write while preparing the next read."""
        return self._read(
            hyper_input,
            pending_output=block_output,
            pending_injection=injection_weights,
        )


def gated_residual_write(
    hyper_input: mx.array,
    block_output: mx.array,
    injection_weights: mx.array,
) -> mx.array:
    """Inject one sublayer result into each residual branch."""
    if hyper_input.shape[:-1] != block_output.shape[:-1]:
        raise ValueError("block output batch dimensions must match the residual")
    if block_output.shape[-1] <= 0 or hyper_input.shape[-1] % block_output.shape[-1]:
        raise ValueError("block output width must divide the expanded residual width")
    branch_count = hyper_input.shape[-1] // block_output.shape[-1]
    if injection_weights.shape != (*block_output.shape[:-1], branch_count):
        raise ValueError("injection weights do not match residual geometry")
    update = block_output[..., None, :] * injection_weights[..., :, None]
    return hyper_input + update.reshape(hyper_input.shape)


__all__ = [
    "Qwen4GatedResidual",
    "Qwen4RMSNorm",
    "Qwen4SigmoidRMSNormGated",
    "gated_residual_write",
    "qwen4_hc_call_counts",
    "qwen4_projection_compute_dtype",
]
