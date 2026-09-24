"""Correctness-first MLX primitives for released Qwen Sparse Attention."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from moespresso.runtime.qwen4.model import (
    Qwen4MixerOutput,
    _Qwen4TrustedMaskCertificate,
    _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    _QWEN4_BATCHED_UNDO_CAPABILITY,
    _QWEN4_SERIAL_LANE_CAPABILITY,
)
from moespresso.runtime.qwen4.primitives import (
    Qwen4RMSNorm,
    qwen4_projection_compute_dtype,
)

if TYPE_CHECKING:
    from moespresso.runtime.qwen4.mtp_qsa_projections import Qwen4MTPQSAProjectionRow
from moespresso.runtime.qwen4.qsa_native_selector import (
    native_qsa_selected_token_indices,
    native_qsa_selector_eligible,
)


QWEN38_QSA_ROPE_DIM = 64
QWEN38_QSA_ROPE_BASE = 10_000_000.0
QWEN38_QSA_MROPE_SECTION = (11, 11, 10)
QWEN38_QSA_COMPRESS_RATIO = 4
QWEN38_QSA_TOKEN_BUDGET = 2048
_QSA_STATE_SCHEMA = "qwen4_qsa_state_v1"
_QSA_PREFIX_SUM_MIN_QUERIES = 64
_QSA_TRUSTED_ALL_VALID_CAPABILITY = object()
_QSA_DEFER_PENDING_FINITE_CAPABILITY = object()


def _qsa_project_rope_operation():
    """Return the optional native projection symbol without making it mandatory."""

    try:
        import mlx_kquant
    except ImportError:
        return None
    operation = getattr(mlx_kquant, "qwen4_qsa_project_rope_q6", None)
    return operation if callable(operation) else None


def _released_q6_projection(module: Any, shape: tuple[int, int]) -> bool:
    """Recognize one exact released Q6_K direct projection wire."""

    try:
        has_bias = "bias" in module
    except (AttributeError, TypeError):
        return False
    return bool(
        getattr(module, "mode", None) == "kquant"
        and getattr(module, "kquant_type", None) == "q6_k"
        and not has_bias
        and isinstance(getattr(module, "weight", None), mx.array)
        and module.weight.dtype == mx.uint8
        and tuple(module.weight.shape) == shape
        and isinstance(getattr(module, "scales", None), mx.array)
        and module.scales.dtype == mx.uint8
        and module.scales.size == 1
    )


def _qsa_project_rope_contract(
    module: Any,
    hidden_states: mx.array,
    position_ids: mx.array,
    factors: "_Qwen4PartialRoPEFactors | None",
) -> bool:
    """Check the complete released one-token native projection contract."""

    indexer = getattr(module, "indexer", None)
    index_q_norm = getattr(indexer, "q_layernorm", None)
    index_k_norm = getattr(indexer, "k_layernorm", None)
    query_norm = getattr(module, "q_norm", None)
    key_norm = getattr(module, "k_norm", None)
    norms = (index_q_norm, index_k_norm, query_norm, key_norm)
    if factors is None or any(norm is None for norm in norms):
        return False
    eps_values = tuple(getattr(norm, "eps", None) for norm in norms)
    if not all(isinstance(value, (float, int)) for value in eps_values):
        return False
    if len({float(value) for value in eps_values}) != 1:
        return False
    return bool(
        tuple(hidden_states.shape) == (1, 1, 2560)
        and hidden_states.dtype == mx.bfloat16
        and getattr(module, "hidden_size", None) == 2560
        and getattr(module, "num_query_heads", None) == 24
        and getattr(module, "num_kv_heads", None) == 2
        and getattr(module, "head_dim", None) == 256
        and getattr(module, "index_query_heads", None) == 4
        and getattr(module, "index_kv_heads", None) == 1
        and getattr(module, "index_head_dim", None) == 128
        and getattr(module, "rotary_dim", None) == 64
        and getattr(module, "rope_base", None) == QWEN38_QSA_ROPE_BASE
        and getattr(module, "mrope_section", None) == QWEN38_QSA_MROPE_SECTION
        and _released_q6_projection(indexer.index_qk_proj, (640, 2100))
        and _released_q6_projection(module.q_proj, (12288, 2100))
        and _released_q6_projection(module.k_proj, (512, 2100))
        and _released_q6_projection(module.v_proj, (512, 2100))
        and all(
            isinstance(getattr(norm, "weight", None), mx.array)
            and norm.weight.dtype == mx.bfloat16
            and tuple(norm.weight.shape) == shape
            for norm, shape in zip(
                norms,
                ((128,), (128,), (256,), (256,)),
                strict=True,
            )
        )
        and factors.position_ids is position_ids
        and factors.batch_size == 1
        and factors.token_count == 1
        and factors.rotary_dim == 64
        and factors.base == QWEN38_QSA_ROPE_BASE
        and factors.mrope_section == QWEN38_QSA_MROPE_SECTION
        and factors.dtype == mx.bfloat16
        and factors.cosine.dtype == mx.bfloat16
        and factors.sine.dtype == mx.bfloat16
        and tuple(factors.cosine.shape) == (1, 1, 1, 64)
        and tuple(factors.sine.shape) == (1, 1, 1, 64)
        and position_ids.dtype in (mx.int32, mx.int64)
        and tuple(position_ids.shape) == (3, 1, 1)
    )


@dataclass(frozen=True)
class QSAVisiblePrefixLayout:
    """Fixed-shape causal visibility shared by QSA index and attention paths."""

    physical_ids: mx.array
    valid: mx.array
    visible_counts: mx.array
    query_valid: mx.array
    group_ids: mx.array
    group_valid: mx.array
    context_length: int
    compress_ratio: int


@dataclass(frozen=True)
class Qwen4QSAIndexPreparation:
    """Compressed index keys and an opaque backend update for one segment."""

    compressed_keys: mx.array
    update: Any


@dataclass(frozen=True)
class _Qwen4PartialRoPEFactors:
    """Request-local trigonometric factors for one semantic position tensor."""

    position_ids: mx.array
    cosine: mx.array
    sine: mx.array
    batch_size: int
    token_count: int
    rotary_dim: int
    base: float
    mrope_section: tuple[int, int, int]
    dtype: mx.Dtype


def qsa_causal_prefix_layout(
    visible_mask: mx.array,
    query_token_indices: mx.array,
    *,
    compress_ratio: int = QWEN38_QSA_COMPRESS_RATIO,
) -> QSAVisiblePrefixLayout:
    """Pack a 2-D padding mask into per-query causal visible prefixes.

    This represents ordinary causal inference: each query sees the visible
    physical rows at or before its cache position. It deliberately does not
    accept an arbitrary four-dimensional attention mask whose visible set can
    change non-monotonically between queries.
    """
    if visible_mask.ndim != 2 or not all(visible_mask.shape):
        raise ValueError("visible_mask must have shape [batch, context]")
    if visible_mask.dtype != mx.bool_:
        raise ValueError("visible_mask must be boolean")
    if query_token_indices.ndim != 2 or query_token_indices.shape[0] != visible_mask.shape[0]:
        raise ValueError("query_token_indices must have shape [batch, queries]")
    if query_token_indices.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
        raise ValueError("query_token_indices must contain integers")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")

    batch_size, context_length = visible_mask.shape
    physical = mx.arange(context_length, dtype=mx.int32)
    sentinel = mx.array(context_length, dtype=mx.int32)
    packed = mx.sort(
        mx.where(visible_mask, physical[None], sentinel),
        axis=-1,
    )
    valid = packed < context_length
    physical_ids = mx.where(valid, packed, -1)

    query_valid = (query_token_indices >= 0) & (query_token_indices < context_length)
    if query_token_indices.shape[1] < _QSA_PREFIX_SUM_MIN_QUERIES:
        visible_counts = mx.sum(
            visible_mask[:, None, :] & (physical[None, None, :] <= query_token_indices[..., None]),
            axis=-1,
        ).astype(mx.int32)
    else:
        safe_queries = mx.clip(query_token_indices, 0, context_length - 1).astype(mx.int32)
        visible_prefix = mx.cumsum(visible_mask.astype(mx.int32), axis=-1)
        visible_counts = mx.take_along_axis(visible_prefix, safe_queries, axis=-1)
    visible_counts = mx.where(query_valid, visible_counts, 0)

    group_count = context_length // compress_ratio
    grouped_width = group_count * compress_ratio
    group_ids = physical_ids[:, :grouped_width].reshape(
        batch_size,
        group_count,
        compress_ratio,
    )
    group_valid = mx.all(valid[:, :grouped_width].reshape(group_ids.shape), axis=-1)
    return QSAVisiblePrefixLayout(
        physical_ids=physical_ids,
        valid=valid,
        visible_counts=visible_counts,
        query_valid=query_valid,
        group_ids=group_ids,
        group_valid=group_valid,
        context_length=context_length,
        compress_ratio=compress_ratio,
    )


def qwen4_partial_rope(
    values: mx.array,
    position_ids: mx.array,
    *,
    rotary_dim: int = QWEN38_QSA_ROPE_DIM,
    base: float = QWEN38_QSA_ROPE_BASE,
    mrope_section: tuple[int, int, int] = QWEN38_QSA_MROPE_SECTION,
) -> mx.array:
    """Apply released split-half partial RoPE at explicit token positions.

    ``values`` has shape ``[batch, tokens, heads, head_dim]``. Text positions
    may be ``[batch, tokens]``. Multimodal positions use
    ``[3, batch, tokens]`` and the released interleaved MRoPE frequency map.
    The explicit multiply-add path is intentional: fused MLX RoPE does not
    reproduce the released BF16 lattice.
    """
    if values.ndim != 4 or not all(values.shape):
        raise ValueError("values must have shape [batch, tokens, heads, head_dim]")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > values.shape[-1]:
        raise ValueError("rotary_dim must be positive, even, and fit the head width")
    if len(mrope_section) != 3 or sum(mrope_section) != rotary_dim // 2:
        raise ValueError("mrope_section must contain three parts covering rotary_dim / 2")
    if not math.isfinite(base) or base <= 0:
        raise ValueError("base must be finite and positive")

    factors = _qwen4_partial_rope_factors(
        position_ids,
        batch_size=values.shape[0],
        token_count=values.shape[1],
        rotary_dim=rotary_dim,
        base=base,
        mrope_section=mrope_section,
        dtype=values.dtype,
    )
    return _apply_qwen4_partial_rope_factors(
        values,
        position_ids,
        factors,
        rotary_dim=rotary_dim,
        base=base,
        mrope_section=mrope_section,
    )


def _qwen4_partial_rope_factors(
    position_ids: mx.array,
    *,
    batch_size: int,
    token_count: int,
    rotary_dim: int,
    base: float,
    mrope_section: tuple[int, int, int],
    dtype: mx.Dtype,
) -> _Qwen4PartialRoPEFactors:
    """Build the released partial-MRoPE factors for one position tensor."""

    positions = position_ids
    if positions.ndim == 2:
        positions = mx.broadcast_to(positions[None], (3, *positions.shape))
    if positions.ndim != 3 or positions.shape != (3, batch_size, token_count):
        raise ValueError("position_ids must have shape [batch, tokens] or [3, batch, tokens]")

    frequencies = mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim
    inverse = mx.power(mx.array(base, dtype=mx.float32), -frequencies)
    axis_frequencies = positions.astype(mx.float32)[..., None] * inverse
    axis_map = _mrope_axis_map(mrope_section)
    mixed_frequencies = mx.stack(
        [axis_frequencies[axis, ..., index] for index, axis in enumerate(axis_map)],
        axis=-1,
    )
    angles = mx.concatenate([mixed_frequencies, mixed_frequencies], axis=-1)
    return _Qwen4PartialRoPEFactors(
        position_ids=position_ids,
        cosine=mx.cos(angles).astype(dtype)[:, :, None, :],
        sine=mx.sin(angles).astype(dtype)[:, :, None, :],
        batch_size=batch_size,
        token_count=token_count,
        rotary_dim=rotary_dim,
        base=float(base),
        mrope_section=tuple(mrope_section),
        dtype=dtype,
    )


def _apply_qwen4_partial_rope_factors(
    values: mx.array,
    position_ids: mx.array,
    factors: _Qwen4PartialRoPEFactors,
    *,
    rotary_dim: int,
    base: float,
    mrope_section: tuple[int, int, int],
) -> mx.array:
    """Apply request-local factors while preserving the released BF16 lattice."""

    expected_shape = (values.shape[0], values.shape[1], 1, rotary_dim)
    if (
        not isinstance(factors, _Qwen4PartialRoPEFactors)
        or factors.position_ids is not position_ids
        or factors.batch_size != values.shape[0]
        or factors.token_count != values.shape[1]
        or factors.rotary_dim != rotary_dim
        or factors.base != float(base)
        or factors.mrope_section != tuple(mrope_section)
        or factors.dtype != values.dtype
        or factors.cosine.shape != expected_shape
        or factors.sine.shape != expected_shape
        or factors.cosine.dtype != values.dtype
        or factors.sine.dtype != values.dtype
    ):
        raise ValueError("partial RoPE factors do not match values and semantic positions")

    half = rotary_dim // 2
    rotary = values[..., :rotary_dim]
    rotated_half = mx.concatenate([-rotary[..., half:], rotary[..., :half]], axis=-1)
    rotated = rotary * factors.cosine + rotated_half * factors.sine
    return mx.concatenate([rotated, values[..., rotary_dim:]], axis=-1)


def _mrope_axis_map(section: tuple[int, int, int]) -> tuple[int, ...]:
    """Return the released time-height-width interleaving for RoPE frequencies."""
    total = sum(section)
    axes = [0] * total
    for axis in (1, 2):
        for index in range(axis, section[axis] * 3, 3):
            axes[index] = axis
    return tuple(axes)


def qsa_index_scores(index_queries: mx.array, compressed_keys: mx.array) -> mx.array:
    """Score compressed keys with the released ReLU-summed MQA indexer."""
    if index_queries.ndim != 4 or not all(index_queries.shape):
        raise ValueError("index_queries must have shape [batch, queries, heads, width]")
    if compressed_keys.ndim != 3:
        raise ValueError("compressed_keys must have shape [batch, groups, width]")
    if (
        compressed_keys.shape[0] != index_queries.shape[0]
        or compressed_keys.shape[-1] != index_queries.shape[-1]
    ):
        raise ValueError("index query and compressed-key geometry must match")
    raw = (
        index_queries.astype(mx.float32)
        @ compressed_keys.astype(mx.float32).swapaxes(-1, -2)[:, None]
    )
    return mx.sum(mx.maximum(raw, 0), axis=-2) / math.sqrt(index_queries.shape[-1])


def qsa_compress_index_keys(
    raw_index_keys: mx.array,
    position_ids: mx.array,
    key_norm,
    layout: QSAVisiblePrefixLayout,
    *,
    rotary_dim: int = QWEN38_QSA_ROPE_DIM,
    base: float = QWEN38_QSA_ROPE_BASE,
    mrope_section: tuple[int, int, int] = QWEN38_QSA_MROPE_SECTION,
) -> mx.array:
    """Pool complete visible-key groups, then normalize and rotate their starts.

    Group boundaries follow visible rank, not physical token id. The layout is
    shared with selection so padding and physical holes cannot drift between
    the two stages.
    """
    if raw_index_keys.ndim != 3 or not all(raw_index_keys.shape):
        raise ValueError("raw_index_keys must have shape [batch, tokens, width]")
    positions = position_ids
    if positions.ndim == 2:
        positions = mx.broadcast_to(positions[None], (3, *positions.shape))
    if positions.ndim != 3 or positions.shape != (
        3,
        raw_index_keys.shape[0],
        raw_index_keys.shape[1],
    ):
        raise ValueError("position_ids must match raw_index_keys")
    if (
        layout.context_length != raw_index_keys.shape[1]
        or layout.physical_ids.shape != raw_index_keys.shape[:2]
        or layout.group_ids.shape[0] != raw_index_keys.shape[0]
    ):
        raise ValueError("visibility layout does not match the raw index-key cache")

    complete_groups = layout.group_ids.shape[1]
    if not complete_groups:
        return mx.zeros(
            (raw_index_keys.shape[0], 0, raw_index_keys.shape[-1]),
            dtype=raw_index_keys.dtype,
        )
    physical = layout.group_ids
    safe = mx.where(layout.group_valid[..., None], physical, 0)
    batch_offsets = mx.arange(raw_index_keys.shape[0])[:, None] * raw_index_keys.shape[1]
    flat_indices = (safe + batch_offsets[:, :, None]).reshape(-1)
    grouped = mx.take(
        raw_index_keys.reshape(-1, raw_index_keys.shape[-1]), flat_indices, axis=0
    ).reshape(
        raw_index_keys.shape[0],
        complete_groups,
        layout.compress_ratio,
        raw_index_keys.shape[-1],
    )
    pooled = mx.mean(grouped.astype(mx.float32), axis=2).astype(raw_index_keys.dtype)
    normalized = key_norm(pooled)
    group_starts = safe[..., 0]
    position_offsets = mx.arange(raw_index_keys.shape[0])[None, :, None] * raw_index_keys.shape[1]
    flat_position_indices = (group_starts[None] + position_offsets).reshape(-1)
    token_major_positions = positions.reshape(3, -1)
    group_positions = mx.take(token_major_positions, flat_position_indices, axis=1).reshape(
        3,
        raw_index_keys.shape[0],
        complete_groups,
    )
    rotated = qwen4_partial_rope(
        normalized[:, :, None, :],
        group_positions,
        rotary_dim=rotary_dim,
        base=base,
        mrope_section=mrope_section,
    )[:, :, 0, :]
    return mx.where(layout.group_valid[..., None], rotated, 0)


def qsa_selected_token_indices(
    scores: mx.array,
    layout: QSAVisiblePrefixLayout,
    *,
    token_budget: int = QWEN38_QSA_TOKEN_BUDGET,
    compress_ratio: int = QWEN38_QSA_COMPRESS_RATIO,
) -> mx.array:
    """Select fixed-width physical token ids in ascending context order.

    The released implementation turns top-k groups into a set mask over the
    full cache. Its attention reduction therefore visits selected rows in
    original context order, not score order. This function restores that order
    without constructing a context-sized mask. Invalid output lanes carry
    ``-1``.
    """
    if scores.ndim != 3:
        raise ValueError("scores must have shape [batch, queries, groups]")
    if token_budget <= 0 or compress_ratio <= 0 or token_budget % compress_ratio:
        raise ValueError("token_budget must be positive and divisible by compress_ratio")

    batch_size, query_count, group_count = scores.shape
    if layout.compress_ratio != compress_ratio:
        raise ValueError("visibility layout uses another compression ratio")
    if layout.physical_ids.shape[0] != batch_size or layout.visible_counts.shape != (
        batch_size,
        query_count,
    ):
        raise ValueError("visibility layout does not match score batch and query axes")
    visible_width = layout.physical_ids.shape[1]
    if not visible_width:
        raise ValueError("visible_token_indices must contain at least one lane")
    if group_count != layout.group_ids.shape[1]:
        raise ValueError("score groups must match the fixed visible-token layout")
    block_budget = token_budget // compress_ratio
    bounded_counts = layout.visible_counts
    visible_groups = mx.minimum(
        bounded_counts // compress_ratio,
        group_count,
    )
    if group_count:
        group_ids = mx.arange(group_count)[None, None, :]
        visible_scores = mx.where(group_ids < visible_groups[..., None], scores, -mx.inf)
        selected_width = min(block_budget, group_count)
        selected_groups = mx.argpartition(
            -visible_scores,
            kth=selected_width - 1,
            axis=-1,
        )[..., :selected_width]
        selected_valid = selected_groups < visible_groups[..., None]
        expanded_ranks = (
            selected_groups[..., None] * compress_ratio
            + mx.arange(compress_ratio)[None, None, None, :]
        ).reshape(batch_size, query_count, -1)
        expanded_valid = mx.repeat(selected_valid, compress_ratio, axis=-1)
        expanded_ranks = mx.where(
            expanded_valid,
            expanded_ranks,
            0,
        )
        batch_offsets = mx.arange(batch_size)[:, None, None] * visible_width
        expanded = mx.take(
            layout.physical_ids.reshape(-1),
            (expanded_ranks + batch_offsets).reshape(-1),
            axis=0,
        ).reshape(batch_size, query_count, -1)
        expanded = mx.where(
            mx.repeat(selected_valid, compress_ratio, axis=-1),
            expanded,
            -1,
        )
    else:
        expanded = mx.full((batch_size, query_count, 0), -1, dtype=mx.int32)
    if expanded.shape[-1] < token_budget:
        expanded = mx.pad(
            expanded,
            [(0, 0), (0, 0), (0, token_budget - expanded.shape[-1])],
            constant_values=-1,
        )

    tail_start = (bounded_counts // compress_ratio) * compress_ratio
    tail_ranks = tail_start[..., None] + mx.arange(compress_ratio - 1)
    tail_valid = tail_ranks < bounded_counts[..., None]
    safe_tail_ranks = mx.where(tail_valid, tail_ranks, 0)
    tail_offsets = mx.arange(batch_size)[:, None, None] * visible_width
    tail = mx.take(
        layout.physical_ids.reshape(-1),
        (safe_tail_ranks + tail_offsets).reshape(-1),
        axis=0,
    ).reshape(batch_size, query_count, -1)
    tail = mx.where(tail_valid, tail, -1)
    selected = mx.concatenate([expanded, tail], axis=-1).astype(mx.int32)

    sentinel = mx.array(2**31 - 1, dtype=mx.int32)
    selected = mx.sort(mx.where(selected >= 0, selected, sentinel), axis=-1)
    return mx.where(selected == sentinel, -1, selected)


def qsa_normalize_selected_rows(
    selected_indices: mx.array,
    token_count: int,
) -> tuple[mx.array, mx.array]:
    """Restore ascending physical-row set semantics for a QSA selection."""
    if selected_indices.ndim != 3:
        raise ValueError("selected_indices must have shape [batch, queries, width]")
    if selected_indices.dtype != mx.int32:
        raise ValueError("selected_indices must contain int32 values")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError("token_count must be a nonnegative integer")
    sentinel = mx.array(token_count, dtype=selected_indices.dtype)
    normalized = mx.sort(
        mx.where(
            (selected_indices >= 0) & (selected_indices < token_count),
            selected_indices,
            sentinel,
        ),
        axis=-1,
    )
    valid = normalized < token_count
    lane_ids = mx.arange(normalized.shape[-1])
    duplicate = (lane_ids[None, None, :] > 0) & (
        normalized == mx.roll(normalized, shift=1, axis=-1)
    )
    valid = valid & ~duplicate
    return normalized, valid


def qsa_gather_selected_rows(
    keys: mx.array,
    values: mx.array,
    selected_indices: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Gather the selected K/V set in ascending physical context order."""
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("keys and values must match [batch, kv_heads, tokens, width]")
    if selected_indices.ndim != 3 or selected_indices.shape[0] != keys.shape[0]:
        raise ValueError("selected_indices must have shape [batch, queries, width]")
    token_count = keys.shape[-2]
    normalized, valid = qsa_normalize_selected_rows(selected_indices, token_count)
    safe = mx.where(valid, normalized, 0)
    batch_size, query_count, selected_width = safe.shape
    kv_heads, head_width = keys.shape[1], keys.shape[-1]

    batch_offsets = mx.arange(batch_size)[:, None, None] * token_count
    flat_indices = (safe + batch_offsets).reshape(-1)
    token_major_keys = keys.transpose(0, 2, 1, 3).reshape(-1, kv_heads, head_width)
    token_major_values = values.transpose(0, 2, 1, 3).reshape(-1, kv_heads, head_width)
    gathered_keys = mx.take(token_major_keys, flat_indices, axis=0).reshape(
        batch_size,
        query_count,
        selected_width,
        kv_heads,
        head_width,
    )
    gathered_values = mx.take(token_major_values, flat_indices, axis=0).reshape(
        batch_size,
        query_count,
        selected_width,
        kv_heads,
        head_width,
    )
    gathered_keys = gathered_keys.transpose(0, 1, 3, 2, 4)
    gathered_values = gathered_values.transpose(0, 1, 3, 2, 4)
    return gathered_keys, gathered_values, valid


def qsa_selected_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    selected_indices: mx.array,
    *,
    scale: float | None = None,
) -> mx.array:
    """Apply FP32-softmax GQA over the selector's physical-row set.

    The low-level gather restores set semantics, but visibility provenance
    comes from ``qsa_selected_token_indices`` and its shared causal layout.
    """
    if queries.ndim != 4:
        raise ValueError("queries must have shape [batch, queries, query_heads, width]")
    if queries.shape[0] != keys.shape[0] or queries.shape[-1] != keys.shape[-1]:
        raise ValueError("query and KV geometry must match")
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        raise ValueError("query, key, and value tensors must share one dtype")
    if queries.shape[-2] % keys.shape[1]:
        raise ValueError("query heads must form equal groups over KV heads")
    gathered_keys, gathered_values, valid = qsa_gather_selected_rows(
        keys,
        values,
        selected_indices,
    )
    return qsa_attention_from_selected_rows(
        queries,
        gathered_keys,
        gathered_values,
        valid,
        scale=scale,
    )


def qsa_attention_from_selected_rows(
    queries: mx.array,
    gathered_keys: mx.array,
    gathered_values: mx.array,
    valid: mx.array,
    *,
    scale: float | None = None,
) -> mx.array:
    """Apply released QSA arithmetic to an already gathered physical-row set."""
    if queries.ndim != 4:
        raise ValueError("queries must have shape [batch, queries, query_heads, width]")
    if gathered_keys.ndim != 5 or gathered_values.shape != gathered_keys.shape:
        raise ValueError(
            "selected keys and values must match [batch, queries, kv_heads, rows, width]"
        )
    batch_size, query_count, query_heads, head_width = queries.shape
    if gathered_keys.shape[:2] != (batch_size, query_count):
        raise ValueError("selected rows do not match the query batch and token axes")
    if gathered_keys.shape[-1] != head_width:
        raise ValueError("selected rows do not match the query head width")
    if gathered_keys.dtype != queries.dtype or gathered_values.dtype != queries.dtype:
        raise ValueError("queries and selected rows must share one dtype")
    if (
        valid.shape
        != (
            batch_size,
            query_count,
            gathered_keys.shape[-2],
        )
        or valid.dtype != mx.bool_
    ):
        raise ValueError("selected-row validity has incompatible geometry or dtype")
    kv_heads = gathered_keys.shape[2]
    if not kv_heads or query_heads % kv_heads:
        raise ValueError("query heads must form equal groups over selected K/V heads")
    group_size = query_heads // kv_heads
    grouped_queries = queries.reshape(
        batch_size,
        query_count,
        kv_heads,
        group_size,
        head_width,
    )
    attention_scale = head_width**-0.5 if scale is None else float(scale)
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("scale must be finite and positive")
    logits = (grouped_queries @ gathered_keys.swapaxes(-1, -2)) * attention_scale
    logits = mx.where(valid[:, :, None, None, :], logits, -mx.inf)
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True).astype(
        queries.dtype
    )
    probabilities = mx.where(
        mx.any(valid, axis=-1)[:, :, None, None, None],
        probabilities,
        0,
    )
    output = probabilities @ gathered_values
    return output.reshape(batch_size, query_count, query_heads, head_width)


@dataclass(frozen=True)
class Qwen4QSAState:
    """One QSA layer's immutable continuation state at a public frontier."""

    keys: mx.array
    values: mx.array
    raw_index_keys: mx.array
    position_ids: mx.array
    offset: int
    schema: str = _QSA_STATE_SCHEMA

    def __post_init__(self) -> None:
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("QSA state offset must be a nonnegative integer")


class Qwen4QSAIndexer(nn.Module):
    """Released QSA index projection and its two zero-centered norms."""

    def __init__(
        self,
        hidden_size: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        *,
        eps: float,
    ) -> None:
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.index_qk_proj = nn.Linear(
            hidden_size,
            (query_heads + kv_heads) * head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4RMSNorm(head_dim, eps=eps)
        self.k_layernorm = Qwen4RMSNorm(head_dim, eps=eps)


class Qwen4SparseAttention(nn.Module):
    """Checkpoint-shaped projections and norms for released Qwen Sparse Attention."""

    def __init__(
        self,
        hidden_size: int = 2560,
        num_query_heads: int = 24,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        index_query_heads: int = 4,
        index_kv_heads: int = 1,
        index_head_dim: int = 128,
        token_budget: int = QWEN38_QSA_TOKEN_BUDGET,
        compress_ratio: int = QWEN38_QSA_COMPRESS_RATIO,
        *,
        rotary_dim: int = QWEN38_QSA_ROPE_DIM,
        rope_base: float = QWEN38_QSA_ROPE_BASE,
        mrope_section: tuple[int, int, int] = QWEN38_QSA_MROPE_SECTION,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        geometry = {
            "hidden_size": hidden_size,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "index_query_heads": index_query_heads,
            "index_kv_heads": index_kv_heads,
            "index_head_dim": index_head_dim,
            "token_budget": token_budget,
            "compress_ratio": compress_ratio,
            "rotary_dim": rotary_dim,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in geometry.values()
        ):
            raise ValueError(f"QSA geometry must contain positive integers: {geometry}")
        if num_query_heads % num_kv_heads:
            raise ValueError("query heads must form equal groups over KV heads")
        if index_kv_heads != 1:
            raise ValueError("released Qwen4 QSA requires one index KV head")
        if token_budget % compress_ratio:
            raise ValueError("token_budget must be divisible by compress_ratio")
        if rotary_dim % 2 or rotary_dim > min(head_dim, index_head_dim):
            raise ValueError("rotary_dim must be even and fit both attention head widths")
        if len(mrope_section) != 3 or sum(mrope_section) != rotary_dim // 2:
            raise ValueError("mrope_section must contain three parts covering rotary_dim / 2")
        if not math.isfinite(rope_base) or rope_base <= 0:
            raise ValueError("rope_base must be finite and positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")

        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.index_query_heads = index_query_heads
        self.index_kv_heads = index_kv_heads
        self.index_head_dim = index_head_dim
        self.token_budget = token_budget
        self.compress_ratio = compress_ratio
        self.rotary_dim = rotary_dim
        self.rope_base = float(rope_base)
        self.mrope_section = tuple(mrope_section)

        self.q_proj = nn.Linear(hidden_size, num_query_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_query_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen4RMSNorm(head_dim, eps=eps)
        self.k_norm = Qwen4RMSNorm(head_dim, eps=eps)
        self.indexer = Qwen4QSAIndexer(
            hidden_size,
            index_query_heads,
            index_kv_heads,
            index_head_dim,
            eps=eps,
        )


class Qwen4QSAStateBackend(Protocol):
    """Cache representation consumed by the shared QSA execution path."""

    def fork_state(self, state: Any) -> Any: ...

    def snapshot_state(self, state: Any) -> Any: ...

    def frontier(self, state: Any) -> int: ...

    def validate_state_structure(
        self,
        state: Any,
        *,
        expected_frontier: int,
        batch_size: int,
        position_dtype: mx.Dtype,
        projection_dtype: mx.Dtype,
    ) -> None: ...

    def validate_position_identity(
        self,
        state: Any,
        position_history: mx.array,
    ) -> None: ...

    def validate_step_inputs(
        self,
        *,
        hidden_states: mx.array,
        valid_tokens: mx.array,
        visible_history: mx.array,
        projection_dtype: mx.Dtype,
    ) -> None: ...

    def safe_chunk_tokens(self, state: Any, requested: int) -> int: ...

    def prepare_index(
        self,
        state: Any,
        *,
        raw_index_keys: mx.array,
        position_ids: mx.array,
        layout: QSAVisiblePrefixLayout,
        key_norm: Any,
        rotary_dim: int,
        rope_base: float,
        mrope_section: tuple[int, int, int],
    ) -> Qwen4QSAIndexPreparation: ...

    def gather_selected_rows(
        self,
        state: Any,
        pending_keys: mx.array,
        pending_values: mx.array,
        selected_indices: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]: ...

    def append(
        self,
        state: Any,
        *,
        keys: mx.array,
        values: mx.array,
        valid_tokens: mx.array,
        index_preparation: Qwen4QSAIndexPreparation,
    ) -> Any: ...


class Qwen4BF16QSAStateBackend:
    """Full-BF16 QSA state backend used when no other policy is selected."""

    def __init__(self, module: Qwen4SparseAttention) -> None:
        self.module = module

    def fork_state(self, state: Qwen4QSAState | None) -> Qwen4QSAState | None:
        return self._copy_state(state)

    def snapshot_state(self, state: Qwen4QSAState | None) -> Qwen4QSAState | None:
        return self._copy_state(state)

    def frontier(self, state: Qwen4QSAState | None) -> int:
        if state is None:
            return 0
        if not isinstance(state, Qwen4QSAState):
            raise ValueError("QSA state is incompatible with the BF16 backend")
        if isinstance(state.offset, bool) or not isinstance(state.offset, int) or state.offset < 0:
            raise ValueError("QSA state offset must be a nonnegative integer")
        return state.offset

    def validate_state_structure(
        self,
        state: Qwen4QSAState | None,
        *,
        expected_frontier: int,
        batch_size: int,
        position_dtype: mx.Dtype,
        projection_dtype: mx.Dtype,
    ) -> None:
        if expected_frontier == 0:
            if state is not None:
                raise ValueError("QSA state must be empty at frontier zero")
            return
        if not isinstance(state, Qwen4QSAState):
            raise ValueError("QSA state is missing at a live frontier")
        if state.schema != _QSA_STATE_SCHEMA:
            raise ValueError("QSA state schema is incompatible")
        if self.frontier(state) != expected_frontier:
            raise ValueError("QSA state is off the public frontier")

        module = self.module
        if state.keys.shape != (
            batch_size,
            module.num_kv_heads,
            expected_frontier,
            module.head_dim,
        ):
            raise ValueError("QSA key state has incompatible geometry")
        if state.values.shape != state.keys.shape:
            raise ValueError("QSA value state has incompatible geometry")
        if state.raw_index_keys.shape != (
            batch_size,
            expected_frontier,
            module.index_head_dim,
        ):
            raise ValueError("QSA raw index-key state has incompatible geometry")
        if state.position_ids.shape != (3, batch_size, expected_frontier):
            raise ValueError("QSA cached positions have incompatible geometry")
        if (
            state.keys.dtype != projection_dtype
            or state.values.dtype != projection_dtype
            or state.raw_index_keys.dtype != projection_dtype
        ):
            raise ValueError("QSA cached tensors have incompatible dtype")
        if state.position_ids.dtype != position_dtype:
            raise ValueError("QSA cached positions have incompatible dtype")

    def validate_position_identity(
        self,
        state: Qwen4QSAState | None,
        position_history: mx.array,
    ) -> None:
        if state is None:
            if position_history.shape[-1]:
                raise ValueError("QSA position history is live without cache state")
            return
        mx.eval(state.position_ids, position_history)
        if not np.array_equal(np.asarray(state.position_ids), np.asarray(position_history)):
            raise ValueError("QSA cached positions do not match the public history")

    def validate_step_inputs(
        self,
        *,
        hidden_states: mx.array,
        valid_tokens: mx.array,
        visible_history: mx.array,
        projection_dtype: mx.Dtype,
    ) -> None:
        del hidden_states, valid_tokens, visible_history, projection_dtype

    def safe_chunk_tokens(self, state: Qwen4QSAState | None, requested: int) -> int:
        del state
        return requested

    def prepare_index(
        self,
        state: Qwen4QSAState | None,
        *,
        raw_index_keys: mx.array,
        position_ids: mx.array,
        layout: QSAVisiblePrefixLayout,
        key_norm: Any,
        rotary_dim: int,
        rope_base: float,
        mrope_section: tuple[int, int, int],
    ) -> Qwen4QSAIndexPreparation:
        full_index_keys = (
            raw_index_keys
            if state is None
            else mx.concatenate([state.raw_index_keys, raw_index_keys], axis=1)
        )
        full_positions = (
            position_ids
            if state is None
            else mx.concatenate([state.position_ids, position_ids], axis=2)
        )
        compressed = qsa_compress_index_keys(
            full_index_keys,
            full_positions,
            key_norm,
            layout,
            rotary_dim=rotary_dim,
            base=rope_base,
            mrope_section=mrope_section,
        )
        return Qwen4QSAIndexPreparation(
            compressed_keys=compressed,
            update=(full_index_keys, full_positions),
        )

    def gather_selected_rows(
        self,
        state: Qwen4QSAState | None,
        pending_keys: mx.array,
        pending_values: mx.array,
        selected_indices: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        full_keys = (
            pending_keys if state is None else mx.concatenate([state.keys, pending_keys], axis=2)
        )
        full_values = (
            pending_values
            if state is None
            else mx.concatenate([state.values, pending_values], axis=2)
        )
        return qsa_gather_selected_rows(full_keys, full_values, selected_indices)

    def append(
        self,
        state: Qwen4QSAState | None,
        *,
        keys: mx.array,
        values: mx.array,
        valid_tokens: mx.array,
        index_preparation: Qwen4QSAIndexPreparation,
    ) -> Qwen4QSAState:
        del valid_tokens
        full_index_keys, full_positions = index_preparation.update
        return Qwen4QSAState(
            keys=keys if state is None else mx.concatenate([state.keys, keys], axis=2),
            values=values if state is None else mx.concatenate([state.values, values], axis=2),
            raw_index_keys=full_index_keys,
            position_ids=full_positions,
            offset=self.frontier(state) + keys.shape[2],
        )

    @staticmethod
    def _copy_state(state: Qwen4QSAState | None) -> Qwen4QSAState | None:
        if state is None:
            return None
        return Qwen4QSAState(
            keys=state.keys + mx.zeros_like(state.keys),
            values=state.values + mx.zeros_like(state.values),
            raw_index_keys=state.raw_index_keys + mx.zeros_like(state.raw_index_keys),
            position_ids=state.position_ids + mx.zeros_like(state.position_ids),
            offset=state.offset,
            schema=state.schema,
        )


class Qwen4QSAAdapter(nn.Module):
    """Bind released QSA projections and an opaque state backend to the shell."""

    def __init__(
        self,
        module: Qwen4SparseAttention,
        *,
        state_backend: Qwen4QSAStateBackend | None = None,
        max_query_tokens: int | None = None,
    ) -> None:
        super().__init__()
        if max_query_tokens is not None and (
            isinstance(max_query_tokens, bool)
            or not isinstance(max_query_tokens, int)
            or max_query_tokens <= 0
        ):
            raise ValueError("max_query_tokens must be a positive integer or None")
        self.module = module
        self.state_backend = (
            Qwen4BF16QSAStateBackend(module) if state_backend is None else state_backend
        )
        self.max_query_tokens = max_query_tokens
        self.trusted_mask_certificate_calls = 0
        self.trusted_all_valid_certificate_calls = 0
        self.shared_rope_factor_calls = 0
        self.shared_rope_applications = 0
        self.native_selector_calls = 0
        self.native_selector_groups = 0
        self.native_selector_ineligible_calls = 0
        self.native_select_gather_calls = 0
        self.native_select_gather_groups = 0
        self.native_select_gather_ineligible_calls = 0
        self.native_select_gather_unavailable_calls = 0
        self.native_project_rope_calls = 0
        self.native_project_rope_ineligible_calls = 0
        self.native_project_rope_unavailable_calls = 0
        self._trusted_mask_issuer: object | None = None
        self._trusted_mask_scope: object | None = None

    @property
    def supports_trusted_mask_certificate(self) -> bool:
        """Whether the backend accepts the released all-valid certificate."""

        return bool(
            getattr(
                self.state_backend,
                "supports_trusted_mask_certificate",
                False,
            )
        )

    @property
    def supports_batched_append_finite(self) -> bool:
        """Whether trusted one-token appends expose a lazy finite predicate."""

        return bool(
            getattr(
                self.state_backend,
                "supports_batched_append_finite",
                False,
            )
        )

    @property
    def supports_dependency_bound_undo(self) -> bool:
        """Whether append inputs can carry lazy rollback-copy dependencies."""

        return bool(getattr(self.state_backend, "supports_dependency_bound_undo", False))

    @property
    def supports_serial_irrevocable_append(self) -> bool:
        """Whether the backend supports private no-rollback one-row append."""

        return bool(
            getattr(
                self.state_backend,
                "supports_serial_irrevocable_append",
                False,
            )
        )

    def trusted_mask_stats(self) -> dict[str, int]:
        """Return monotonic counters for the request-local trusted seam."""

        return {
            "trusted_mask_certificate_calls": self.trusted_mask_certificate_calls,
            "trusted_all_valid_certificate_calls": (self.trusted_all_valid_certificate_calls),
        }

    @property
    def shared_rope_signature(self) -> tuple[int, float, tuple[int, int, int], mx.Dtype]:
        """Return the current-token RoPE geometry accepted by this adapter."""

        module = self.module
        return (
            module.rotary_dim,
            module.rope_base,
            module.mrope_section,
            self._projection_dtype(),
        )

    def shared_rope_stats(self) -> dict[str, int]:
        """Return monotonic counters for request-local factor reuse."""

        return {
            "shared_rope_factor_calls": self.shared_rope_factor_calls,
            "shared_rope_applications": self.shared_rope_applications,
        }

    def native_selector_stats(self) -> dict[str, int]:
        """Return monotonic counters for the trusted decode selector."""

        return {
            "native_selector_calls": self.native_selector_calls,
            "native_selector_groups": self.native_selector_groups,
            "native_selector_ineligible_calls": self.native_selector_ineligible_calls,
            "native_select_gather_calls": self.native_select_gather_calls,
            "native_select_gather_groups": self.native_select_gather_groups,
            "native_select_gather_ineligible_calls": self.native_select_gather_ineligible_calls,
            "native_select_gather_unavailable_calls": self.native_select_gather_unavailable_calls,
            "native_project_rope_calls": self.native_project_rope_calls,
            "native_project_rope_ineligible_calls": self.native_project_rope_ineligible_calls,
            "native_project_rope_unavailable_calls": self.native_project_rope_unavailable_calls,
        }

    def _prepare_shared_rope_factors(
        self,
        position_ids: mx.array,
    ) -> _Qwen4PartialRoPEFactors:
        """Build one current-token factor set for compatible QSA adapters."""

        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("shared QSA RoPE positions must have three semantic planes")
        if position_ids.shape[-1] != 1:
            raise ValueError("shared QSA RoPE factors are decode-only")
        if position_ids.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("shared QSA RoPE positions must contain integers")
        module = self.module
        return _qwen4_partial_rope_factors(
            position_ids,
            batch_size=position_ids.shape[1],
            token_count=1,
            rotary_dim=module.rotary_dim,
            base=module.rope_base,
            mrope_section=module.mrope_section,
            dtype=self._projection_dtype(),
        )

    def _prepare_mtp_rope_factors(
        self,
        position_ids: mx.array,
    ) -> _Qwen4PartialRoPEFactors:
        """Build one factor set for a two-row MTP projection preparation."""

        if position_ids.ndim != 3 or tuple(position_ids.shape) != (3, 1, 2):
            raise ValueError("MTP QSA RoPE positions must contain two decode rows")
        if position_ids.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("MTP QSA RoPE positions must contain integers")
        module = self.module
        return _qwen4_partial_rope_factors(
            position_ids,
            batch_size=1,
            token_count=2,
            rotary_dim=module.rotary_dim,
            base=module.rope_base,
            mrope_section=module.mrope_section,
            dtype=self._projection_dtype(),
        )

    def _bind_trusted_mask_issuer(self, issuer: object, scope: object) -> None:
        """Bind one shell-issued certificate scope to this adapter."""

        if issuer is None or scope is None:
            raise ValueError("QSA trusted mask issuer and scope must not be None")
        if self._trusted_mask_issuer is not None and self._trusted_mask_issuer is not issuer:
            raise ValueError("QSA adapter is already bound to another model shell")
        if self._trusted_mask_scope is not None:
            raise ValueError("QSA adapter already has an active trusted mask scope")
        self._trusted_mask_issuer = issuer
        self._trusted_mask_scope = scope

    def _clear_trusted_mask_scope(self, issuer: object, scope: object) -> None:
        """Invalidate one completed or failed request-local certificate."""

        if self._trusted_mask_issuer is not issuer or self._trusted_mask_scope is not scope:
            raise ValueError("QSA trusted mask scope does not match the active request")
        self._trusted_mask_scope = None

    def _prepare_trusted_undo(
        self,
        state: Any,
        *,
        new_tokens: int,
        capability: object,
    ) -> Any:
        """Prepare one request-local mutable rollback reservation."""

        if capability is not _QWEN4_BATCHED_UNDO_CAPABILITY:
            raise ValueError("QSA batched undo capability is invalid")
        prepare = getattr(self.state_backend, "_prepare_trusted_undo", None)
        if not callable(prepare):
            raise TypeError("QSA backend does not implement trusted undo preparation")
        return prepare(
            state,
            new_tokens=new_tokens,
            capability=capability,
        )

    def _trusted_undo_arrays(
        self,
        reservation: Any,
        *,
        capability: object,
    ) -> tuple[mx.array, ...]:
        """Return the arrays that freeze one rollback reservation."""

        if capability is not _QWEN4_BATCHED_UNDO_CAPABILITY:
            raise ValueError("QSA batched undo capability is invalid")
        get_arrays = getattr(self.state_backend, "_trusted_undo_arrays", None)
        if not callable(get_arrays):
            raise TypeError("QSA backend does not expose trusted undo arrays")
        return get_arrays(reservation, capability=capability)

    def _mark_trusted_undo_evaluated(
        self,
        reservation: Any,
        *,
        capability: object,
    ) -> None:
        """Mark one shell-evaluated rollback reservation as consumable."""

        if capability is not _QWEN4_BATCHED_UNDO_CAPABILITY:
            raise ValueError("QSA batched undo capability is invalid")
        mark = getattr(self.state_backend, "_mark_trusted_undo_evaluated", None)
        if not callable(mark):
            raise TypeError("QSA backend does not mark trusted undo reservations")
        mark(reservation, capability=capability)

    def fork_state(self, state: Any) -> Any:
        return self.state_backend.fork_state(state)

    def snapshot_state(self, state: Any) -> Any:
        return self.state_backend.snapshot_state(state)

    def restore_state(self, state: Any) -> None:
        """Restore a process-local mutable backend to a selected frontier."""
        restore = getattr(self.state_backend, "restore_state", None)
        if callable(restore):
            restore(state)

    def abandon_serial_state(self, state: Any) -> None:
        """Permanently invalidate request-private mutable storage."""

        abandon = getattr(self.state_backend, "abandon_serial_state", None)
        if not callable(abandon):
            raise TypeError("QSA backend does not implement serial abandonment")
        abandon(state, capability=_QWEN4_SERIAL_LANE_CAPABILITY)

    def commit_state(self, state: Any) -> Any:
        """Publish a mutable backend frontier and release its rollback journal."""
        self.preflight_commit_state(state)
        return self.commit_state_preflighted(state)

    def preflight_commit_state(self, state: Any) -> None:
        """Validate an active mutable view before any layer publishes."""
        preflight = getattr(self.state_backend, "preflight_commit_state", None)
        if callable(preflight):
            preflight(state)

    def commit_state_preflighted(self, state: Any) -> Any:
        """Finalize a mutable view after model-wide commit preflight."""
        finalize = getattr(self.state_backend, "commit_state_preflighted", None)
        return state if not callable(finalize) else finalize(state)

    def validate_state(
        self,
        state: Any,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        """Strictly validate externally supplied or restored QSA state."""
        self.validate_state_strict(
            state,
            expected_frontier=expected_frontier,
            position_history=position_history,
        )

    def validate_state_strict(
        self,
        state: Any,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        """Validate structure and exact semantic-position identity."""
        self._validate_state_structure(
            state,
            expected_frontier=expected_frontier,
            position_history=position_history,
        )
        if expected_frontier == 0:
            return
        assert state is not None
        self.state_backend.validate_position_identity(state, position_history)

    def validate_state_trusted(
        self,
        state: Any,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        """Validate state derived from the shell's committed in-process lineage."""
        self._validate_state_structure(
            state,
            expected_frontier=expected_frontier,
            position_history=position_history,
        )

    def _validate_state_structure(
        self,
        state: Any,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        if (
            isinstance(expected_frontier, bool)
            or not isinstance(expected_frontier, int)
            or expected_frontier < 0
        ):
            raise ValueError("QSA frontier must be a nonnegative integer")
        if (
            position_history.ndim != 3
            or position_history.shape[0] != 3
            or position_history.shape[-1] != expected_frontier
        ):
            raise ValueError("QSA position history does not share the public frontier")
        if position_history.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("QSA position history must contain integers")
        self.state_backend.validate_state_structure(
            state,
            expected_frontier=expected_frontier,
            batch_size=position_history.shape[1],
            position_dtype=position_history.dtype,
            projection_dtype=self._projection_dtype(),
        )

    def _validate_step_state_structure(
        self,
        state: Any,
        *,
        expected_frontier: int,
        batch_size: int,
        position_dtype: mx.Dtype,
    ) -> None:
        self.state_backend.validate_state_structure(
            state,
            expected_frontier=expected_frontier,
            batch_size=batch_size,
            position_dtype=position_dtype,
            projection_dtype=self._projection_dtype(),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state: Any,
        prior_position_history: mx.array | None = None,
    ) -> Qwen4MixerOutput:
        """Advance state after strict validation of an external input state."""
        if state is not None and prior_position_history is None:
            raise ValueError("live external QSA state requires prior_position_history")
        return self._step(
            hidden_states,
            valid_tokens=valid_tokens,
            visible_history=visible_history,
            position_ids=position_ids,
            state=state,
            prior_position_history=prior_position_history,
            trusted_state=False,
            mask_certificate=None,
            shared_rope_factors=None,
            prepared_projections=None,
        )

    def step_trusted(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state: Any,
        prepared_projections: Qwen4MTPQSAProjectionRow | None = None,
    ) -> Qwen4MixerOutput:
        """Advance state owned by the shell's committed in-process lineage."""
        return self._step(
            hidden_states,
            valid_tokens=valid_tokens,
            visible_history=visible_history,
            position_ids=position_ids,
            state=state,
            prior_position_history=None,
            trusted_state=True,
            mask_certificate=None,
            shared_rope_factors=None,
            prepared_projections=prepared_projections,
        )

    def _step_trusted_certified(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state: Any,
        mask_certificate: _Qwen4TrustedMaskCertificate,
        prepared_projections: Qwen4MTPQSAProjectionRow | None = None,
    ) -> Qwen4MixerOutput:
        """Advance trusted state using a shell-bound mask certificate."""

        return self._step(
            hidden_states,
            valid_tokens=valid_tokens,
            visible_history=visible_history,
            position_ids=position_ids,
            state=state,
            prior_position_history=None,
            trusted_state=True,
            mask_certificate=mask_certificate,
            shared_rope_factors=None,
            prepared_projections=prepared_projections,
        )

    def _step_trusted_prepared(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state: Any,
        mask_certificate: _Qwen4TrustedMaskCertificate | None,
        shared_rope_factors: _Qwen4PartialRoPEFactors,
        prepared_projections: Qwen4MTPQSAProjectionRow | None = None,
    ) -> Qwen4MixerOutput:
        """Advance trusted state with request-local QSA preparation."""

        return self._step(
            hidden_states,
            valid_tokens=valid_tokens,
            visible_history=visible_history,
            position_ids=position_ids,
            state=state,
            prior_position_history=None,
            trusted_state=True,
            mask_certificate=mask_certificate,
            shared_rope_factors=shared_rope_factors,
            prepared_projections=prepared_projections,
        )

    def _step(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state: Any,
        prior_position_history: mx.array | None,
        trusted_state: bool,
        mask_certificate: _Qwen4TrustedMaskCertificate | None,
        shared_rope_factors: _Qwen4PartialRoPEFactors | None,
        prepared_projections: Qwen4MTPQSAProjectionRow | None,
    ) -> Qwen4MixerOutput:
        module = self.module
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != module.hidden_size:
            raise ValueError("QSA hidden states have incompatible geometry")
        batch_size, token_count, _ = hidden_states.shape
        projection_dtype = self._projection_dtype()
        if hidden_states.dtype != projection_dtype:
            raise ValueError("QSA hidden states must match the projection dtype")
        if valid_tokens.shape != (batch_size, token_count) or valid_tokens.dtype != mx.bool_:
            raise ValueError("QSA valid-token mask has incompatible geometry")
        current_frontier = self.state_backend.frontier(state)
        next_frontier = current_frontier + token_count
        if (
            visible_history.shape != (batch_size, next_frontier)
            or visible_history.dtype != mx.bool_
        ):
            raise ValueError("QSA visible history does not reach the proposed frontier")
        if position_ids.shape != (3, batch_size, token_count) or position_ids.dtype not in (
            mx.int32,
            mx.int64,
            mx.uint32,
            mx.uint64,
        ):
            raise ValueError("QSA position ids have incompatible geometry or dtype")
        visible_suffix = visible_history[:, current_frontier:next_frontier]
        masks_certified = False
        prepared_undo = None
        serial_append = False
        if mask_certificate is None:
            if not bool(mx.array_equal(valid_tokens, visible_suffix).item()):
                raise ValueError("QSA visible history does not contain the current validity mask")
        else:
            serial_append = mask_certificate.serial_capability is _QWEN4_SERIAL_LANE_CAPABILITY
            transactional_append = bool(mask_certificate.prepared_undos) and all(
                undo is not None for undo in mask_certificate.prepared_undos
            )
            if (
                self._trusted_mask_issuer is None
                or mask_certificate.issuer is not self._trusted_mask_issuer
                or self._trusted_mask_scope is None
                or mask_certificate.scope is not self._trusted_mask_scope
                or len(mask_certificate.allowed_mixers) != len(mask_certificate.source_states)
                or (
                    mask_certificate.prepared_undos
                    and len(mask_certificate.allowed_mixers) != len(mask_certificate.prepared_undos)
                )
                or (mask_certificate.serial_capability is not None and not serial_append)
                or (serial_append and bool(mask_certificate.prepared_undos))
                or (
                    mask_certificate.append_finite_batch is not None
                    and (
                        not (serial_append or transactional_append)
                        or len(mask_certificate.append_finite_batch.allowed_mixers)
                        != len(mask_certificate.allowed_mixers)
                        or any(
                            batch_mixer is not mixer or batch_source is not source
                            for batch_mixer, batch_source, mixer, source in zip(
                                mask_certificate.append_finite_batch.allowed_mixers,
                                mask_certificate.append_finite_batch.source_states,
                                mask_certificate.allowed_mixers,
                                mask_certificate.source_states,
                                strict=True,
                            )
                        )
                    )
                )
                or (serial_append and mask_certificate.append_finite_batch is None)
                or not any(
                    mixer is self and source is state
                    for mixer, source in zip(
                        mask_certificate.allowed_mixers,
                        mask_certificate.source_states,
                    )
                )
                or mask_certificate.valid_tokens is not valid_tokens
                or mask_certificate.visible_history is not visible_history
                or mask_certificate.current_frontier != current_frontier
                or mask_certificate.next_frontier != next_frontier
            ):
                raise ValueError("QSA trusted mask certificate is unissued or stale")
            masks_certified = True
            certificate_undos = mask_certificate.prepared_undos or (
                (None,) * len(mask_certificate.allowed_mixers)
            )
            for mixer, source, reservation in zip(
                mask_certificate.allowed_mixers,
                mask_certificate.source_states,
                certificate_undos,
                strict=True,
            ):
                if mixer is self and source is state:
                    prepared_undo = reservation
                    break
            self.trusted_mask_certificate_calls += 1
            self.trusted_all_valid_certificate_calls += 1
        if masks_certified:
            validate_step_inputs = getattr(
                self.state_backend,
                "_validate_step_inputs_certified",
                None,
            )
            if not callable(validate_step_inputs):
                raise TypeError("QSA backend does not implement certified mask validation")
            validate_step_inputs(
                hidden_states=hidden_states,
                valid_tokens=valid_tokens,
                visible_history=visible_history,
                projection_dtype=projection_dtype,
                capability=_QSA_TRUSTED_ALL_VALID_CAPABILITY,
            )
        else:
            self.state_backend.validate_step_inputs(
                hidden_states=hidden_states,
                valid_tokens=valid_tokens,
                visible_history=visible_history,
                projection_dtype=projection_dtype,
            )
        if trusted_state:
            self._validate_step_state_structure(
                state,
                expected_frontier=current_frontier,
                batch_size=batch_size,
                position_dtype=position_ids.dtype,
            )
        else:
            prior_positions = (
                prior_position_history
                if prior_position_history is not None
                else mx.zeros((3, batch_size, 0), dtype=position_ids.dtype)
            )
            self.validate_state_strict(
                state,
                expected_frontier=current_frontier,
                position_history=prior_positions,
            )
        if shared_rope_factors is not None:
            if not trusted_state or token_count != 1:
                raise ValueError("shared QSA RoPE factors require trusted one-token decode")
            self.shared_rope_factor_calls += 1
        if prepared_projections is not None:
            if not trusted_state or token_count != 1:
                raise ValueError("prepared MTP QSA projections require trusted one-token decode")
            prepared_projections.validate(self, hidden_states)

        def rotate(values: mx.array) -> mx.array:
            if shared_rope_factors is None:
                return qwen4_partial_rope(
                    values,
                    position_ids,
                    rotary_dim=module.rotary_dim,
                    base=module.rope_base,
                    mrope_section=module.mrope_section,
                )
            self.shared_rope_applications += 1
            return _apply_qwen4_partial_rope_factors(
                values,
                position_ids,
                shared_rope_factors,
                rotary_dim=module.rotary_dim,
                base=module.rope_base,
                mrope_section=module.mrope_section,
            )

        native_projection = None
        native_candidate = bool(
            masks_certified
            and trusted_state
            and token_count == 1
            and shared_rope_factors is not None
            and prepared_projections is None
        )
        native_eligible = bool(
            native_candidate
            and _qsa_project_rope_contract(
                module,
                hidden_states,
                position_ids,
                shared_rope_factors,
            )
        )
        if native_eligible:
            operation = _qsa_project_rope_operation()
            if operation is None:
                self.native_project_rope_unavailable_calls += 1
            else:
                native_projection = operation(
                    hidden_states,
                    module.indexer.index_qk_proj.weight,
                    module.indexer.index_qk_proj.scales,
                    module.q_proj.weight,
                    module.q_proj.scales,
                    module.k_proj.weight,
                    module.k_proj.scales,
                    module.v_proj.weight,
                    module.v_proj.scales,
                    module.indexer.q_layernorm.weight,
                    module.q_norm.weight,
                    module.k_norm.weight,
                    shared_rope_factors.cosine,
                    shared_rope_factors.sine,
                    position_ids,
                    eps=float(module.q_norm.eps),
                )
                expected = (
                    ((1, 1, 4, 128), mx.bfloat16),
                    ((1, 1, 128), mx.bfloat16),
                    ((1, 1, 24, 256), mx.bfloat16),
                    ((1, 1, 6144), mx.bfloat16),
                    ((1, 2, 1, 256), mx.bfloat16),
                    ((1, 2, 1, 256), mx.bfloat16),
                )
                if (
                    not isinstance(native_projection, (tuple, list))
                    or len(native_projection) != len(expected)
                    or any(
                        not isinstance(value, mx.array)
                        or tuple(value.shape) != shape
                        or value.dtype != dtype
                        for value, (shape, dtype) in zip(
                            native_projection,
                            expected,
                            strict=True,
                        )
                    )
                ):
                    raise ValueError("native QSA projection returned incompatible outputs")
                self.native_project_rope_calls += 1
                self.shared_rope_applications += 3
        elif native_candidate:
            self.native_project_rope_ineligible_calls += 1

        if native_projection is not None:
            index_queries, raw_index_keys, queries, gate, keys, values = native_projection
        elif prepared_projections is not None:
            index_queries = prepared_projections.index_queries
            raw_index_keys = prepared_projections.raw_index_keys
            queries = prepared_projections.queries
            gate = prepared_projections.gate
            keys = prepared_projections.keys
            values = prepared_projections.values
            if shared_rope_factors is not None:
                self.shared_rope_applications += 3
        else:
            index_projection = module.indexer.index_qk_proj(hidden_states)
            index_query_width = module.index_query_heads * module.index_head_dim
            index_queries, raw_index_keys = mx.split(
                index_projection,
                [index_query_width],
                axis=-1,
            )
            index_queries = index_queries.reshape(
                batch_size,
                token_count,
                module.index_query_heads,
                module.index_head_dim,
            )
            raw_index_keys = raw_index_keys.reshape(
                batch_size,
                token_count,
                module.index_kv_heads,
                module.index_head_dim,
            )[:, :, 0, :]
            index_queries = module.indexer.q_layernorm(index_queries)
            index_queries = rotate(index_queries)

            query_gate = module.q_proj(hidden_states).reshape(
                batch_size,
                token_count,
                module.num_query_heads,
                module.head_dim * 2,
            )
            queries, gate = mx.split(query_gate, [module.head_dim], axis=-1)
            gate = gate.reshape(batch_size, token_count, -1)
            queries = module.q_norm(queries)
            queries = rotate(queries)

            keys = module.k_proj(hidden_states).reshape(
                batch_size,
                token_count,
                module.num_kv_heads,
                module.head_dim,
            )
            keys = module.k_norm(keys)
            keys = rotate(keys).transpose(0, 2, 1, 3)
            values = (
                module.v_proj(hidden_states)
                .reshape(
                    batch_size,
                    token_count,
                    module.num_kv_heads,
                    module.head_dim,
                )
                .transpose(0, 2, 1, 3)
            )
        if any(
            value.dtype != projection_dtype
            for value in (index_queries, raw_index_keys, queries, gate, keys, values)
        ):
            raise ValueError("QSA projections produced an incompatible cache dtype")
        bind_undo = getattr(self.state_backend, "_bind_trusted_undo_dependencies", None)
        if prepared_undo is not None and callable(bind_undo):
            raw_index_keys, keys, values, position_ids = bind_undo(
                prepared_undo,
                (raw_index_keys, keys, values, position_ids),
                capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
            )

        working_state = state
        attention_parts = []
        cursor = 0
        while cursor < token_count:
            width = self.state_backend.safe_chunk_tokens(
                working_state,
                token_count - cursor,
            )
            if (
                isinstance(width, bool)
                or not isinstance(width, int)
                or not 0 < width <= token_count - cursor
            ):
                raise ValueError("QSA state backend returned an invalid chunk width")
            end = cursor + width
            segment_frontier = self.state_backend.frontier(working_state)
            segment_next_frontier = segment_frontier + width
            segment_positions = position_ids[:, :, cursor:end]
            segment_index_keys = raw_index_keys[:, cursor:end]
            segment_keys = keys[:, :, cursor:end]
            segment_values = values[:, :, cursor:end]
            physical_query_rows = mx.broadcast_to(
                mx.arange(segment_frontier, segment_next_frontier, dtype=mx.int32)[None],
                (batch_size, width),
            )
            layout = qsa_causal_prefix_layout(
                visible_history[:, :segment_next_frontier],
                physical_query_rows,
                compress_ratio=module.compress_ratio,
            )
            if masks_certified:
                prepare_index = getattr(
                    self.state_backend,
                    "_prepare_index_certified",
                    None,
                )
                if not callable(prepare_index):
                    raise TypeError("QSA backend does not implement certified index preparation")
                index_preparation = prepare_index(
                    working_state,
                    raw_index_keys=segment_index_keys,
                    position_ids=segment_positions,
                    layout=layout,
                    key_norm=module.indexer.k_layernorm,
                    rotary_dim=module.rotary_dim,
                    rope_base=module.rope_base,
                    mrope_section=module.mrope_section,
                    capability=_QSA_TRUSTED_ALL_VALID_CAPABILITY,
                )
            else:
                index_preparation = self.state_backend.prepare_index(
                    working_state,
                    raw_index_keys=segment_index_keys,
                    position_ids=segment_positions,
                    layout=layout,
                    key_norm=module.indexer.k_layernorm,
                    rotary_dim=module.rotary_dim,
                    rope_base=module.rope_base,
                    mrope_section=module.mrope_section,
                )
            short_selected = None
            if width == 1:
                select_short = getattr(
                    self.state_backend,
                    "select_all_valid_short_context",
                    None,
                )
                if callable(select_short):
                    short_selected = select_short(
                        working_state,
                        pending_tokens=width,
                        token_budget=module.token_budget,
                    )
                    if short_selected is not None and (
                        not isinstance(short_selected, mx.array)
                        or short_selected.shape
                        != (
                            batch_size,
                            1,
                            module.token_budget + module.compress_ratio - 1,
                        )
                        or short_selected.dtype != mx.int32
                    ):
                        raise ValueError("QSA short-context selection has incompatible geometry")
            prepared_rows = None
            gather_prepared = None
            selected_lane_bound = width * (module.token_budget + module.compress_ratio - 1)
            if (
                working_state is not None
                and width > 1
                and segment_next_frontier <= selected_lane_bound
            ):
                prepare_prefill = getattr(
                    self.state_backend,
                    "prepare_prefill_selected_rows",
                    None,
                )
                gather_prepared = getattr(
                    self.state_backend,
                    "gather_prepared_selected_rows",
                    None,
                )
                if callable(prepare_prefill) != callable(gather_prepared):
                    raise TypeError("QSA backend has an incomplete prepared-gather seam")
                if callable(prepare_prefill):
                    prepared_rows = prepare_prefill(
                        working_state,
                        segment_keys,
                        segment_values,
                    )

            def gather_rows(
                selected_indices: mx.array,
            ) -> tuple[mx.array, mx.array, mx.array]:
                if prepared_rows is None:
                    if masks_certified and token_count == 1:
                        gather_deferred = getattr(
                            self.state_backend,
                            "_gather_selected_rows_deferred_finite",
                            None,
                        )
                        if callable(gather_deferred):
                            return gather_deferred(
                                working_state,
                                segment_keys,
                                segment_values,
                                selected_indices,
                                capability=_QSA_DEFER_PENDING_FINITE_CAPABILITY,
                            )
                    return self.state_backend.gather_selected_rows(
                        working_state,
                        segment_keys,
                        segment_values,
                        selected_indices,
                    )
                assert callable(gather_prepared)
                return gather_prepared(prepared_rows, selected_indices)

            def attend_rows(
                query_values: mx.array,
                selected_indices: mx.array,
            ) -> mx.array:
                gathered_keys, gathered_values, selected_valid = gather_rows(selected_indices)
                return qsa_attention_from_selected_rows(
                    query_values,
                    gathered_keys,
                    gathered_values,
                    selected_valid,
                    scale=module.head_dim**-0.5,
                )

            def native_selector_is_eligible(
                scores: mx.array,
                query_layout: QSAVisiblePrefixLayout,
            ) -> bool:
                return bool(
                    masks_certified
                    and token_count == 1
                    and width == 1
                    and native_qsa_selector_eligible(
                        scores,
                        visible_count=query_layout.context_length,
                        token_budget=module.token_budget,
                        compress_ratio=module.compress_ratio,
                    )
                )

            def select_rows_from_scores(
                scores: mx.array,
                query_layout: QSAVisiblePrefixLayout,
            ) -> mx.array:
                native_eligible = native_selector_is_eligible(scores, query_layout)
                if native_eligible:
                    self.native_selector_calls += 1
                    self.native_selector_groups += int(scores.shape[-1])
                    return native_qsa_selected_token_indices(
                        scores,
                        visible_count=query_layout.context_length,
                        token_budget=module.token_budget,
                        compress_ratio=module.compress_ratio,
                    )
                elif masks_certified and token_count == 1 and width == 1:
                    self.native_selector_ineligible_calls += 1
                return qsa_selected_token_indices(
                    scores,
                    query_layout,
                    token_budget=module.token_budget,
                    compress_ratio=module.compress_ratio,
                )

            def select_and_attend(
                query_values: mx.array,
                index_query_values: mx.array,
                query_layout: QSAVisiblePrefixLayout,
            ) -> mx.array:
                if short_selected is not None:
                    return attend_rows(query_values, short_selected)
                scores = qsa_index_scores(
                    index_query_values,
                    index_preparation.compressed_keys,
                )
                native_eligible = native_selector_is_eligible(scores, query_layout)
                select_gather = getattr(
                    self.state_backend,
                    "_select_and_gather_rows_deferred_finite",
                    None,
                )
                composite_eligible = bool(
                    native_eligible
                    and callable(select_gather)
                )
                if composite_eligible:
                    result = select_gather(
                        working_state,
                        segment_keys,
                        segment_values,
                        scores,
                        visible_count=query_layout.context_length,
                        capability=_QSA_DEFER_PENDING_FINITE_CAPABILITY,
                    )
                    if result is not None:
                        selected, selected_valid, gathered_keys, gathered_values = result
                        self.native_selector_calls += 1
                        self.native_selector_groups += int(scores.shape[-1])
                        self.native_select_gather_calls += 1
                        self.native_select_gather_groups += int(scores.shape[-1])
                        return qsa_attention_from_selected_rows(
                            query_values,
                            gathered_keys,
                            gathered_values,
                            selected_valid,
                            scale=module.head_dim**-0.5,
                        )
                    self.native_select_gather_unavailable_calls += 1
                elif native_eligible:
                    self.native_select_gather_ineligible_calls += 1
                selected = select_rows_from_scores(scores, query_layout)
                return attend_rows(query_values, selected)

            if self.max_query_tokens is None:
                attention_parts.append(
                    select_and_attend(
                        queries[:, cursor:end],
                        index_queries[:, cursor:end],
                        layout,
                    )
                )
            else:
                segment_attention = []
                query_cursor = 0
                while query_cursor < width:
                    query_end = min(width, query_cursor + self.max_query_tokens)
                    query_layout = (
                        layout
                        if short_selected is not None
                        else _qsa_query_layout_slice(
                            layout,
                            query_cursor,
                            query_end,
                        )
                    )
                    segment_attention.append(
                        select_and_attend(
                            queries[:, cursor + query_cursor : cursor + query_end],
                            index_queries[:, cursor + query_cursor : cursor + query_end],
                            query_layout,
                        )
                    )
                    query_cursor = query_end
                attention_parts.append(
                    segment_attention[0]
                    if len(segment_attention) == 1
                    else mx.concatenate(segment_attention, axis=1)
                )
            append_state = self.state_backend.append
            append_kwargs = {}
            append_finite_batch = (
                None if mask_certificate is None else mask_certificate.append_finite_batch
            )
            if append_finite_batch is not None and prepared_undo is None and not serial_append:
                raise ValueError("QSA append finite batch requires a prepared undo")
            if serial_append:
                if (
                    not masks_certified
                    or append_finite_batch is None
                    or token_count != 1
                    or cursor != 0
                    or end != 1
                ):
                    raise ValueError("QSA serial append is incompatible with this step")
                append_state = getattr(
                    self.state_backend,
                    "_append_irreversible_deferred_finite",
                    None,
                )
                if not callable(append_state):
                    raise TypeError("QSA backend does not implement serial append")
                append_kwargs = {"capability": _QWEN4_SERIAL_LANE_CAPABILITY}
            elif prepared_undo is not None:
                if not masks_certified or token_count != 1 or cursor != 0 or end != 1:
                    raise ValueError("QSA prepared undo is incompatible with this step")
                append_method = (
                    "_append_with_prepared_undo"
                    if append_finite_batch is None
                    else "_append_with_prepared_undo_deferred_finite"
                )
                append_state = getattr(self.state_backend, append_method, None)
                if not callable(append_state):
                    raise TypeError("QSA backend does not consume prepared undo")
                append_kwargs = {
                    "undo_reservation": prepared_undo,
                    "capability": (
                        _QWEN4_BATCHED_UNDO_CAPABILITY
                        if append_finite_batch is None
                        else _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY
                    ),
                }
            append_result = append_state(
                working_state,
                keys=segment_keys,
                values=segment_values,
                valid_tokens=valid_tokens[:, cursor:end],
                index_preparation=index_preparation,
                **append_kwargs,
            )
            if append_finite_batch is None:
                working_state = append_result
            else:
                working_state, append_finite_predicate = append_result
                append_finite_batch.register(
                    mixer=self,
                    source_state=state,
                    result_state=working_state,
                    predicate=append_finite_predicate,
                    capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
                )
            self._validate_step_state_structure(
                working_state,
                expected_frontier=segment_next_frontier,
                batch_size=batch_size,
                position_dtype=position_ids.dtype,
            )
            cursor = end

        attention = (
            attention_parts[0]
            if len(attention_parts) == 1
            else mx.concatenate(
                attention_parts,
                axis=1,
            )
        )
        attention = attention.reshape(batch_size, token_count, -1)
        gated = attention * mx.sigmoid(gate)
        output = module.o_proj(gated)
        next_state = working_state
        assert next_state is not None
        self._validate_step_state_structure(
            next_state,
            expected_frontier=next_frontier,
            batch_size=batch_size,
            position_dtype=position_ids.dtype,
        )
        return Qwen4MixerOutput(output=output, state=next_state, frontier=next_frontier)

    def _projection_dtype(self):
        module = self.module
        dtype = qwen4_projection_compute_dtype(module.q_proj)
        cache_projections = (
            module.k_proj,
            module.v_proj,
            module.indexer.index_qk_proj,
        )
        if any(
            qwen4_projection_compute_dtype(projection) != dtype for projection in cache_projections
        ):
            raise ValueError("QSA projection modules have incompatible dtype")
        return dtype


def _qsa_query_layout_slice(
    layout: QSAVisiblePrefixLayout,
    start: int,
    end: int,
) -> QSAVisiblePrefixLayout:
    """Slice only the query-dependent fields of a fixed QSA prefix layout."""
    query_count = layout.visible_counts.shape[1]
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start < end <= query_count
    ):
        raise ValueError("QSA query-layout slice is outside the query axis")
    return QSAVisiblePrefixLayout(
        physical_ids=layout.physical_ids,
        valid=layout.valid,
        visible_counts=layout.visible_counts[:, start:end],
        query_valid=layout.query_valid[:, start:end],
        group_ids=layout.group_ids,
        group_valid=layout.group_valid,
        context_length=layout.context_length,
        compress_ratio=layout.compress_ratio,
    )
