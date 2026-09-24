"""Resident Qwen MTP projections using the existing packed IQ2_K kernels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_iqk.format import component_dtypes, component_shapes
from mlx_iqk.nn import IqkSwitchLinear

from moespresso.package.qwen4.mtp_format import MTP_CODEC, mtp_iq2_projection_plan
from moespresso.runtime.qwen4.moe import qwen4_topk_from_logits


class Qwen4MTPIQ2SwitchLinear(nn.Module):
    """An immutable resident expert projection with bounded packed execution.

    Calls always use packed GEMV, including prompt rows. They never materialize
    the complete expert stack as floating-point weights. Reduction slices are
    combined in float32 before returning the activation dtype.
    """

    def __init__(self, num_experts: int, out_features: int, in_features: int):
        super().__init__()
        self.plan = mtp_iq2_projection_plan((num_experts, out_features, in_features))
        self.num_experts = num_experts
        self.out_features = out_features
        self.in_features = in_features
        self.projections = [
            IqkSwitchLinear(MTP_CODEC, num_experts, self.plan.stored_out_features, part.stored_width)
            for part in self.plan.slices
        ]
        self._loaded = False
        self.freeze()

    def load_streams(self, slices: Sequence[Mapping[str, mx.array]]) -> None:
        """Validate all streams before replacing any projection storage."""
        if self._loaded:
            raise ValueError("MTP projection streams are immutable after loading")
        if len(slices) != len(self.plan.slices):
            raise ValueError("MTP stream slice count disagrees with projection plan")
        dtypes = component_dtypes(MTP_CODEC)
        for part, streams in zip(self.plan.slices, slices, strict=True):
            shapes = component_shapes(
                MTP_CODEC, self.num_experts, self.plan.stored_out_features, part.stored_width,
            )
            if set(streams) != set(shapes):
                raise ValueError("MTP projection has missing or unexpected streams")
            for name, shape in shapes.items():
                value = streams[name]
                if (
                    not isinstance(value, mx.array)
                    or tuple(value.shape) != shape
                    or value.dtype != getattr(mx, dtypes[name].name)
                ):
                    raise ValueError(f"MTP stream {name} has invalid shape or dtype")
        for module, streams in zip(self.projections, slices, strict=True):
            module.load_streams(dict(streams))
        self._loaded = True

    def __call__(self, values: mx.array, indices: mx.array) -> mx.array:
        """Project shared token rows or one activation row per selected expert.

        ``indices`` contains validated router IDs. Its shape is either the
        activation prefix or that prefix plus a selected-expert dimension.
        """
        if not self._loaded:
            raise ValueError("MTP projection streams have not been loaded")
        if values.ndim < 2 or values.shape[-1] != self.in_features:
            raise ValueError("MTP activation shape disagrees with projection plan")
        if values.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError("MTP activations must have a floating-point dtype")
        if indices.dtype != mx.uint32 or indices.size == 0:
            raise ValueError("MTP expert IDs must be nonempty uint32 values")
        if values.shape[:-1] not in (indices.shape, indices.shape[:-1]):
            raise ValueError("MTP expert IDs do not align with activation rows")
        result = None
        for part, module in zip(self.plan.slices, self.projections, strict=True):
            operand = values[..., part.begin:part.end]
            padding = part.stored_width - (part.end - part.begin)
            if padding:
                operand = mx.pad(operand, [(0, 0)] * (operand.ndim - 1) + [(0, padding)])
            projected = module.gemv(operand, indices).squeeze(-2)[..., :self.out_features]
            projected = projected.astype(mx.float32)
            result = projected if result is None else result + projected
        return result.astype(values.dtype)


class Qwen4MTPIQ2Linear(Qwen4MTPIQ2SwitchLinear):
    """A dense MTP projection represented as one resident IQ2_K expert."""

    mode = "iqk_dense"

    def __init__(self, out_features: int, in_features: int):
        super().__init__(1, out_features, in_features)

    def __call__(self, values: mx.array) -> mx.array:
        ids = mx.zeros(values.shape[:-1], dtype=mx.uint32)
        return super().__call__(values, ids)


class Qwen4MTPIQ2Experts(nn.Module):
    """Resident IQ2 MTP experts at the existing sparse-MoE executor boundary."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.gate_up_proj = Qwen4MTPIQ2SwitchLinear(
            num_experts, 2 * intermediate_size, hidden_size,
        )
        self.down_proj = Qwen4MTPIQ2SwitchLinear(num_experts, hidden_size, intermediate_size)

    def __call__(self, values: mx.array, indices: mx.array) -> mx.array:
        gate_up = self.gate_up_proj(values, indices)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up, indices)


class Qwen4MTPIQ2Router(Qwen4MTPIQ2Linear):
    """Packed draft routing with the unchanged Qwen top-k selection rule."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        if not 0 < top_k <= num_experts:
            raise ValueError("MTP top_k must fit the expert count")
        super().__init__(num_experts, hidden_size)
        self.top_k = top_k

    def __call__(self, values: mx.array):
        return qwen4_topk_from_logits(super().__call__(values), self.top_k)
