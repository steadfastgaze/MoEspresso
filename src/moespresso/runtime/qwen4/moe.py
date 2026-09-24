"""Correctness-first Qwen4-Exp sparse MoE implementation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Protocol

import mlx.core as mx
import mlx.nn as nn


_RELEASED_ROUTER_HIDDEN_SIZE = 2_560
_RELEASED_ROUTER_EXPERTS = 512
_RELEASED_ROUTER_TOP_K = 10


def _validated_retained_source_ids(
    source_ids: tuple[int, ...] | None,
    *,
    num_experts: int,
    top_k: int,
) -> tuple[int, ...] | None:
    if source_ids is None:
        return None
    if (
        len(source_ids) < top_k
        or len(source_ids) > num_experts
        or source_ids != tuple(sorted(set(source_ids)))
        or any(source_id < 0 or source_id >= num_experts for source_id in source_ids)
    ):
        raise ValueError("retained source expert ids must be sorted, unique, in range, and fit top_k")
    return source_ids


@lru_cache(maxsize=64)
def _retained_routing_arrays(
    num_experts: int,
    source_ids: tuple[int, ...],
) -> tuple[mx.array, mx.array]:
    retained = set(source_ids)
    mask = mx.array(
        [source_id in retained for source_id in range(num_experts)],
        dtype=mx.bool_,
    )
    compact = [-1] * num_experts
    for compact_id, source_id in enumerate(source_ids):
        compact[source_id] = compact_id
    return mask, mx.array(compact, dtype=mx.int32)


@lru_cache(maxsize=1)
def _resolve_qwen4_fused_exact_router_symbol() -> Callable[[mx.array], Any] | None:
    if not mx.metal.is_available():
        return None
    try:
        import mlx_kquant as kq
    except ImportError:
        return None
    symbol = getattr(kq, "qwen4_router_topk_fused_exact", None)
    return symbol if callable(symbol) else None


@dataclass(frozen=True)
class Qwen4RouterOutput:
    """Router logits, normalized top-k scores, and selected expert ids."""

    logits: mx.array
    scores: mx.array
    indices: mx.array


class Qwen4RoutedExpertExecutor(Protocol):
    """Compute one output row for every selected token-expert pair."""

    hidden_size: int
    intermediate_size: int
    num_experts: int

    def __call__(self, hidden_states: mx.array, indices: mx.array) -> mx.array:
        """Return ``[..., top_k, hidden_size]`` routed expert outputs."""
        ...


class Qwen4TopKRouter(nn.Module):
    """FP32-softmax router matching the released Qwen4 top-k contract."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        normalize_topk: bool = True,
        retained_source_ids: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_experts <= 0 or not 0 < top_k <= num_experts:
            raise ValueError("router geometry must be positive and top_k must fit")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_topk = normalize_topk
        self.retained_source_ids = _validated_retained_source_ids(
            retained_source_ids,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.weight = mx.zeros((num_experts, hidden_size))
        object.__setattr__(
            self,
            "_fused_exact_symbol",
            _resolve_qwen4_fused_exact_router_symbol(),
        )
        self.fused_exact_calls = 0
        self.fused_exact_rows = 0
        self.fused_exact_ineligible_calls = 0
        self.fused_exact_unavailable_calls = 0
        self.retained_mask_calls = 0
        self.retained_mask_rows = 0

    def retained_routing_stats(self) -> dict[str, int]:
        """Return source-space masking engagement counters."""

        return {
            "retained_experts": (
                self.num_experts
                if self.retained_source_ids is None
                else len(self.retained_source_ids)
            ),
            "retained_mask_calls": self.retained_mask_calls,
            "retained_mask_rows": self.retained_mask_rows,
            "source_experts": self.num_experts,
        }

    def _selection_logits(self, logits: mx.array) -> mx.array:
        source_ids = self.retained_source_ids
        if source_ids is None:
            return logits
        mask, _compact = _retained_routing_arrays(self.num_experts, source_ids)
        rows = int(logits.size // logits.shape[-1])
        self.retained_mask_calls += 1
        self.retained_mask_rows += rows
        return mx.where(mask, logits, mx.array(float("-inf"), dtype=logits.dtype))

    def fused_exact_stats(self) -> dict[str, int]:
        """Return monotonic engagement counters for the guarded router path."""

        return {
            "fused_exact_calls": self.fused_exact_calls,
            "fused_exact_rows": self.fused_exact_rows,
            "fused_exact_ineligible_calls": self.fused_exact_ineligible_calls,
            "fused_exact_unavailable_calls": self.fused_exact_unavailable_calls,
        }

    def _fused_exact_eligible(
        self,
        hidden_states: mx.array,
        logits: mx.array,
    ) -> bool:
        weight = self.weight
        return bool(
            self.hidden_size == _RELEASED_ROUTER_HIDDEN_SIZE
            and self.num_experts == _RELEASED_ROUTER_EXPERTS
            and self.top_k == _RELEASED_ROUTER_TOP_K
            and self.normalize_topk
            and hidden_states.ndim >= 2
            and hidden_states.dtype == mx.bfloat16
            and hidden_states.size // hidden_states.shape[-1] == 1
            and isinstance(weight, mx.array)
            and weight.shape == (_RELEASED_ROUTER_EXPERTS, _RELEASED_ROUTER_HIDDEN_SIZE)
            and weight.dtype == mx.bfloat16
            and logits.shape == (*hidden_states.shape[:-1], _RELEASED_ROUTER_EXPERTS)
            and logits.dtype == mx.bfloat16
        )

    def _fused_exact_output(
        self,
        selection_logits: mx.array,
        symbol: Callable[[mx.array], Any],
        *,
        raw_logits: mx.array,
    ) -> Qwen4RouterOutput:
        token_shape = raw_logits.shape[:-1]
        rows = selection_logits.size // selection_logits.shape[-1]
        result = symbol(selection_logits.reshape(rows, _RELEASED_ROUTER_EXPERTS))
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError("exact Qwen router kernel returned an invalid result")
        indices, scores = result
        expected_shape = (rows, _RELEASED_ROUTER_TOP_K)
        if (
            not isinstance(indices, mx.array)
            or indices.shape != expected_shape
            or indices.dtype != mx.uint32
            or not isinstance(scores, mx.array)
            or scores.shape != expected_shape
            or scores.dtype != mx.bfloat16
        ):
            raise RuntimeError("exact Qwen router kernel violated its output contract")
        self.fused_exact_calls += 1
        self.fused_exact_rows += rows
        return Qwen4RouterOutput(
            logits=raw_logits,
            scores=scores.reshape(*token_shape, _RELEASED_ROUTER_TOP_K),
            indices=indices.reshape(*token_shape, _RELEASED_ROUTER_TOP_K),
        )

    def __call__(self, hidden_states: mx.array, *, cache_routing: bool = False) -> Qwen4RouterOutput:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("router input width does not match hidden_size")
        provider = getattr(self, "_cache_routing_provider", None) if cache_routing else None
        maps = None
        if provider is not None:
            if self.training or hidden_states.dtype != mx.bfloat16 or hidden_states.size // self.hidden_size != 1:
                raise ValueError("cache routing requires one BF16 inference row")
            maps = provider.snapshot()
        logits = hidden_states @ self.weight.T
        if maps is not None:
            indices, scores = provider.select(logits, maps)
            return Qwen4RouterOutput(logits=logits, scores=scores, indices=indices)
        selection_logits = self._selection_logits(logits)
        decode_candidate = (
            hidden_states.ndim >= 2 and hidden_states.size // hidden_states.shape[-1] == 1
        )
        if decode_candidate:
            if self._fused_exact_eligible(hidden_states, selection_logits):
                symbol = self._fused_exact_symbol
                if symbol is not None:
                    return self._fused_exact_output(
                        selection_logits,
                        symbol,
                        raw_logits=logits,
                    )
                self.fused_exact_unavailable_calls += 1
            else:
                self.fused_exact_ineligible_calls += 1
        return qwen4_topk_from_logits(
            selection_logits, self.top_k, normalize_topk=self.normalize_topk, raw_logits=logits,
        )


def qwen4_topk_from_logits(
    logits: mx.array,
    top_k: int,
    *,
    normalize_topk: bool = True,
    raw_logits: mx.array | None = None,
) -> Qwen4RouterOutput:
    """Apply the Qwen selection rule independently of the router weight encoding."""
    if not 0 < top_k <= logits.shape[-1]:
        raise ValueError("top_k must fit the router logits")
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    candidates = mx.argpartition(probabilities, kth=-top_k, axis=-1)[..., -top_k:]
    candidate_scores = mx.take_along_axis(probabilities, candidates, axis=-1)
    rank = mx.argsort(-candidate_scores, axis=-1)
    indices = mx.take_along_axis(candidates, rank, axis=-1)
    scores = mx.take_along_axis(candidate_scores, rank, axis=-1)
    if normalize_topk:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    original = logits if raw_logits is None else raw_logits
    return Qwen4RouterOutput(logits=original, scores=scores.astype(original.dtype), indices=indices)


class Qwen4ExpertStack(nn.Module):
    """Released stacked ``[gate | up]`` and down expert projections."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0 or num_experts <= 0:
            raise ValueError("expert geometry must be positive")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.gate_up_proj = mx.zeros((num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = mx.zeros((num_experts, hidden_size, intermediate_size))

    def __call__(self, hidden_states: mx.array, indices: mx.array) -> mx.array:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("expert input width does not match hidden_size")
        if indices.shape != (*hidden_states.shape[:-1], indices.shape[-1]):
            raise ValueError("expert indices must match the token dimensions")
        expanded = mx.expand_dims(hidden_states, (-2, -3))
        combined = mx.gather_mm(
            expanded,
            self.gate_up_proj.swapaxes(-1, -2),
            rhs_indices=indices,
        )
        gate, up = mx.split(combined, 2, axis=-1)
        activated = nn.silu(gate) * up
        output = mx.gather_mm(
            activated,
            self.down_proj.swapaxes(-1, -2),
            rhs_indices=indices,
        )
        return output.squeeze(-2)


def expert_major_weighted_sum(
    expert_outputs: mx.array,
    scores: mx.array,
    indices: mx.array,
) -> mx.array:
    """Accumulate routed rows in ascending expert-id order."""
    if expert_outputs.shape[:-1] != scores.shape or scores.shape != indices.shape:
        raise ValueError("expert outputs, scores, and indices have incompatible shapes")
    order = mx.argsort(indices, axis=-1)
    sorted_scores = mx.take_along_axis(scores, order, axis=-1)
    output_order = mx.broadcast_to(order[..., None], expert_outputs.shape)
    sorted_outputs = mx.take_along_axis(expert_outputs, output_order, axis=-2)
    result = mx.zeros_like(sorted_outputs[..., 0, :])
    for position in range(scores.shape[-1]):
        result = result + sorted_outputs[..., position, :] * sorted_scores[..., position, None]
    return result


_compiled_expert_major_weighted_sum = mx.compile(expert_major_weighted_sum)


def _token_rows(hidden_states: mx.array) -> int:
    rows = 1
    for dimension in hidden_states.shape[:-1]:
        rows *= int(dimension)
    return rows


class Qwen4MLP(nn.Module):
    """Bias-free SiLU-gated dense MLP used by the shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen4SparseMoEBlock(nn.Module):
    """Qwen4 routed experts plus the independently gated shared expert."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        normalize_topk: bool = True,
        expert_executor: Qwen4RoutedExpertExecutor | None = None,
        retained_source_ids: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.retained_source_ids = _validated_retained_source_ids(
            retained_source_ids,
            num_experts=num_experts,
            top_k=top_k,
        )
        physical_experts = (
            num_experts
            if self.retained_source_ids is None
            else len(self.retained_source_ids)
        )
        self.physical_experts = physical_experts
        self.gate = Qwen4TopKRouter(
            hidden_size,
            num_experts,
            top_k,
            normalize_topk=normalize_topk,
            retained_source_ids=self.retained_source_ids,
        )
        if expert_executor is None:
            expert_executor = Qwen4ExpertStack(
                hidden_size,
                intermediate_size,
                physical_experts,
            )
        executor_geometry = (
            getattr(expert_executor, "hidden_size", None),
            getattr(expert_executor, "intermediate_size", None),
            getattr(expert_executor, "num_experts", None),
        )
        if executor_geometry != (hidden_size, intermediate_size, physical_experts):
            raise ValueError("routed expert executor geometry does not match the MoE block")
        # Keep the released parameter name for the resident implementation. A
        # streamed or quantized executor occupies the same semantic boundary.
        self.experts = expert_executor
        self.shared_expert = Qwen4MLP(hidden_size, shared_intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        self.compiled_routed_sum_calls = 0
        self.compiled_routed_sum_slot_elements = 0
        self.compiled_routed_sum_output_elements = 0
        self.weighted_decode_calls = 0
        self.weighted_decode_output_elements = 0

    def __call__(self, hidden_states: mx.array, *, cache_routing: bool = False) -> mx.array:
        router = (self.gate(hidden_states, cache_routing=True) if cache_routing
                  else self.gate(hidden_states))
        if self.retained_source_ids is None:
            expert_indices = router.indices
        else:
            _mask, compact = _retained_routing_arrays(
                self.num_experts,
                self.retained_source_ids,
            )
            expert_indices = compact[router.indices].astype(mx.uint32)
        if getattr(self.experts, "_moespresso_pooled_decode_session", None) is not None:
            return self._pooled_forward(hidden_states, expert_indices, router.scores)
        try_weighted = getattr(
            self.experts,
            "try_full_resident_weighted_decode",
            None,
        )
        routed = (
            try_weighted(hidden_states, expert_indices, router.scores)
            if callable(try_weighted)
            else None
        )
        if routed is not None:
            expected_routed_shape = (*hidden_states.shape[:-1], self.hidden_size)
            if routed.shape != expected_routed_shape:
                raise ValueError("weighted routed expert executor returned an invalid shape")
            if routed.dtype != hidden_states.dtype:
                raise ValueError("weighted routed expert executor returned an invalid dtype")
            self.weighted_decode_calls += 1
            self.weighted_decode_output_elements += int(routed.size)
        else:
            try_decode = getattr(self.experts, "try_full_resident_decode", None)
            expert_outputs = (
                try_decode(hidden_states, expert_indices) if callable(try_decode) else None
            )
            if expert_outputs is None:
                expert_outputs = self.experts(hidden_states, expert_indices)
            expected_shape = (*hidden_states.shape[:-1], self.top_k, self.hidden_size)
            if expert_outputs.shape != expected_shape:
                raise ValueError("routed expert executor returned an invalid output shape")
            if expert_outputs.dtype != hidden_states.dtype:
                raise ValueError("routed expert executor returned an invalid output dtype")
            if _token_rows(hidden_states) == 1:
                routed = _compiled_expert_major_weighted_sum(
                    expert_outputs,
                    router.scores,
                    expert_indices,
                )
                self.compiled_routed_sum_calls += 1
                self.compiled_routed_sum_slot_elements += int(router.scores.size)
                self.compiled_routed_sum_output_elements += int(routed.size)
            else:
                routed = expert_major_weighted_sum(
                    expert_outputs,
                    router.scores,
                    expert_indices,
                )
        shared = mx.sigmoid(self.shared_expert_gate(hidden_states)) * self.shared_expert(
            hidden_states
        )
        return routed + shared

    def _pooled_forward(self, hidden_states, expert_indices, scores):
        """Supply Qwen math to the shared pooled execution schedule."""
        from moespresso.runtime.pooled_moe import run_pooled_moe

        def reduce(outputs, weights, indices):
            expected = (*hidden_states.shape[:-1], self.top_k, self.hidden_size)
            if outputs.shape != expected or outputs.dtype != hidden_states.dtype:
                raise ValueError("routed expert executor returned an invalid output contract")
            if _token_rows(hidden_states) == 1:
                result = _compiled_expert_major_weighted_sum(outputs, weights, indices)
                self.compiled_routed_sum_calls += 1
                self.compiled_routed_sum_slot_elements += int(weights.size)
                self.compiled_routed_sum_output_elements += int(result.size)
                return result
            return expert_major_weighted_sum(outputs, weights, indices)

        def resident(value, indices, weights):
            weighted = getattr(self.experts, "try_resident_weighted_decode", None)
            output = weighted(value, indices, weights) if callable(weighted) else None
            if output is not None:
                if output.shape != value.shape or output.dtype != value.dtype:
                    raise ValueError("weighted routed executor returned an invalid output contract")
                self.weighted_decode_calls += 1
                self.weighted_decode_output_elements += int(output.size)
                return output
            output = self.experts.try_full_resident_decode(value, indices)
            return None if output is None else reduce(output, weights, indices)

        def pipelined(value, indices, weights, *, event_gate):
            output = self.experts.build_pipelined(value, indices, event_gate=event_gate)
            return reduce(output, weights, indices)

        return run_pooled_moe(
            self.experts,
            hidden_states,
            expert_indices,
            scores,
            shared=lambda value: (
                mx.sigmoid(self.shared_expert_gate(value)) * self.shared_expert(value)
            ),
            reduce=reduce,
            resident=resident,
            pipelined=pipelined,
            training=self.training,
            last=getattr(self, "pipeline_is_last", False),
        )
