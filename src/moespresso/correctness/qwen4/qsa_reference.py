"""Storage-independent NumPy reference for Qwen Sparse Attention."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


QWEN38_QSA_QUERY_HEADS = 24
QWEN38_QSA_KV_HEADS = 2
QWEN38_QSA_HEAD_DIM = 256
QWEN38_QSA_INDEX_HEADS = 4
QWEN38_QSA_INDEX_HEAD_DIM = 128
QWEN38_QSA_ROTARY_DIM = 64
QWEN38_QSA_COMPRESS_RATIO = 4
QWEN38_QSA_TOKEN_BUDGET = 2048


@dataclass(frozen=True)
class QSAReferenceResult:
    """Intermediate and final values from one QSA decode row."""

    scores: np.ndarray
    visible_groups: int
    selected_groups: np.ndarray
    logical_indices: np.ndarray
    output: np.ndarray


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def zero_centered_rms_norm(
    values: np.ndarray,
    weight: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply the released Qwen4-Exp ``(1 + weight)`` RMSNorm contract."""
    source = np.asarray(values)
    scale = np.asarray(weight, dtype=np.float32)
    if source.ndim < 1 or not source.shape[-1]:
        raise ValueError("values must have a nonempty final dimension")
    if scale.shape != (source.shape[-1],):
        raise ValueError("weight must match the final values dimension")
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eps must be finite and positive")
    working = source.astype(np.float32, copy=False)
    variance = np.mean(working * working, axis=-1, keepdims=True, dtype=np.float32)
    normalized = working / np.sqrt(variance + np.float32(epsilon))
    return (normalized * (np.float32(1) + scale)).astype(source.dtype, copy=False)


def apply_partial_rope(
    values: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """Rotate the leading dimensions and preserve the remaining features."""
    source = np.asarray(values)
    cosine = np.asarray(cos, dtype=source.dtype)
    sine = np.asarray(sin, dtype=source.dtype)
    if source.ndim < 1 or not source.shape[-1]:
        raise ValueError("values must have a nonempty final dimension")
    if cosine.shape != sine.shape or cosine.ndim < 1:
        raise ValueError("cos and sin must have matching shapes")
    rotary_dim = cosine.shape[-1]
    if not rotary_dim or rotary_dim % 2 or rotary_dim > source.shape[-1]:
        raise ValueError("rotary dimension must be positive, even, and fit values")
    try:
        broadcast_shape = np.broadcast_shapes(source.shape[:-1], cosine.shape[:-1])
    except ValueError as error:
        raise ValueError("cos and sin do not broadcast over values") from error
    if broadcast_shape != source.shape[:-1]:
        raise ValueError("cos and sin must not expand the values shape")
    rope = source[..., :rotary_dim]
    half = rotary_dim // 2
    rotated_half = np.concatenate((-rope[..., half:], rope[..., :half]), axis=-1)
    rotated = rope * cosine + rotated_half * sine
    return np.concatenate((rotated, source[..., rotary_dim:]), axis=-1)


def prepare_index_queries(
    raw_queries: np.ndarray,
    norm_weight: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize and rotate released QSA index queries."""
    normalized = zero_centered_rms_norm(raw_queries, norm_weight, eps=eps)
    return apply_partial_rope(normalized, cos, sin)


def compress_index_keys(
    raw_keys: np.ndarray,
    norm_weight: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    compress_ratio: int = QWEN38_QSA_COMPRESS_RATIO,
    eps: float = 1e-6,
    visible_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Pool complete raw-key groups, then normalize and rotate block rows.

    The FP32 mean is cast back to the raw-key dtype before normalization, as in
    the released Transformers reference. ``cos`` and ``sin`` carry block-start
    positions and cover only the leading rotary dimensions.
    """
    ratio = _positive_integer("compress_ratio", compress_ratio)
    source = np.asarray(raw_keys)
    if source.ndim != 2 or not all(source.shape):
        raise ValueError("raw_keys must have shape [tokens, head_dim]")
    if visible_indices is not None:
        visible = _ordered_visible_indices(visible_indices, context_length=source.shape[0])
        source = source[visible]
    complete_groups = source.shape[0] // ratio
    pooled = source[: complete_groups * ratio].reshape(
        complete_groups, ratio, source.shape[1]
    )
    pooled = pooled.astype(np.float32).mean(axis=1, dtype=np.float32)
    pooled = pooled.astype(source.dtype)
    normalized = zero_centered_rms_norm(pooled, norm_weight, eps=eps)
    return apply_partial_rope(normalized, cos, sin)


def sigmoid_output_gate(output: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Apply the released QSA sigmoid gate after sparse attention."""
    values = np.asarray(output)
    logits = np.asarray(gate, dtype=np.float32)
    if logits.shape == (values.size,):
        logits = logits.reshape(values.shape)
    if logits.shape != values.shape:
        raise ValueError("gate must match output or its flattened shape")
    multiplier = np.float32(1) / (np.float32(1) + np.exp(-logits))
    return (values.astype(np.float32) * multiplier).astype(values.dtype)


def split_query_gate_projection(
    projected: np.ndarray,
    *,
    query_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the released per-head ``[query, gate]`` projection layout."""
    heads = _positive_integer("query_heads", query_heads)
    width = _positive_integer("head_dim", head_dim)
    values = np.asarray(projected)
    if values.ndim < 1 or values.shape[-1] != heads * width * 2:
        raise ValueError("projected width does not match query and gate geometry")
    paired = values.reshape(*values.shape[:-1], heads, width * 2)
    return paired[..., :width], paired[..., width:]


def project_qsa_output(
    head_output: np.ndarray,
    gate: np.ndarray,
    output_weight: np.ndarray,
) -> np.ndarray:
    """Flatten heads, apply the sigmoid gate, then apply the output matrix."""
    values = np.asarray(head_output)
    if values.ndim < 2:
        raise ValueError("head_output must end in [query_heads, head_dim]")
    flattened = values.reshape(*values.shape[:-2], -1)
    gated = sigmoid_output_gate(flattened, gate)
    weight = np.asarray(output_weight)
    if weight.ndim != 2 or weight.shape[1] != gated.shape[-1]:
        raise ValueError("output_weight must have shape [output_dim, query_width]")
    return gated.astype(np.float32) @ weight.astype(np.float32).T


def _ordered_visible_indices(
    visible_indices: np.ndarray,
    *,
    context_length: int | None = None,
) -> np.ndarray:
    values = np.asarray(visible_indices)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("visible_indices must be a one-dimensional integer array")
    ordered = values.astype(np.int64, copy=False)
    if np.any(ordered < 0) or (ordered.size > 1 and np.any(ordered[1:] <= ordered[:-1])):
        raise ValueError("visible_indices must be nonnegative and strictly increasing")
    if context_length is not None:
        if isinstance(context_length, bool) or not isinstance(context_length, int):
            raise TypeError("context_length must be an int")
        if context_length < 0:
            raise ValueError("context_length must be nonnegative")
        if ordered.size and ordered[-1] >= context_length:
            raise ValueError("visible_indices exceed the context")
    return ordered


def visible_group_rows(
    visible_indices: np.ndarray,
    *,
    compress_ratio: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition ordered visible token ids into complete groups and a tail."""
    ratio = _positive_integer("compress_ratio", compress_ratio)
    visible = _ordered_visible_indices(visible_indices)
    complete = visible.size // ratio
    groups = visible[: complete * ratio].reshape(complete, ratio)
    return groups, visible[complete * ratio :]


def visible_complete_groups(
    query_position: int,
    context_length: int,
    compress_ratio: int,
) -> int:
    """Return the number of complete compression groups visible to a query."""
    ratio = _positive_integer("compress_ratio", compress_ratio)
    if isinstance(query_position, bool) or not isinstance(query_position, int):
        raise TypeError("query_position must be an int")
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise TypeError("context_length must be an int")
    if query_position < 0:
        raise ValueError("query_position must be nonnegative")
    if context_length < 0:
        raise ValueError("context_length must be nonnegative")
    return max(0, min((query_position + 1) // ratio, context_length // ratio))


def score_compressed_groups(
    index_query: np.ndarray,
    compressed_keys: np.ndarray,
    *,
    score_divisor: float | None = None,
) -> np.ndarray:
    """Score compressed groups as a sum of positive query-key dot products.

    ``index_query`` has shape ``[index_heads, head_dim]``. ``compressed_keys``
    has shape ``[groups, head_dim]`` and is shared by every index head.
    """
    query = np.asarray(index_query, dtype=np.float32)
    keys = np.asarray(compressed_keys, dtype=np.float32)
    if query.ndim != 2 or not all(query.shape):
        raise ValueError("index_query must have shape [index_heads, head_dim]")
    if keys.ndim != 2 or keys.shape[1] != query.shape[1]:
        raise ValueError("compressed_keys must have shape [groups, head_dim]")
    divisor = math.sqrt(query.shape[1]) if score_divisor is None else float(score_divisor)
    if not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("score_divisor must be finite and positive")
    dots = query @ keys.T
    return np.maximum(dots, np.float32(0)).sum(axis=0, dtype=np.float32) / divisor


def stable_topk_groups(
    scores: np.ndarray,
    *,
    visible_groups: int,
    block_topk: int,
) -> np.ndarray:
    """Select visible groups with a deterministic MoEspresso tie policy.

    The result always has ``block_topk`` entries. Unused entries are ``-1``.
    The released Transformers reference uses ``torch.topk``, whose tied-index
    ordering is unspecified. Lower ids win ties here so repeated oracle runs
    are reproducible; parity cases must avoid ties at the selection boundary.
    """
    count = _positive_integer("block_topk", block_topk)
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if isinstance(visible_groups, bool) or not isinstance(visible_groups, int):
        raise TypeError("visible_groups must be an int")
    if visible_groups < 0 or visible_groups > values.size:
        raise ValueError("visible_groups is outside the score vector")
    if not np.all(np.isfinite(values[:visible_groups])):
        raise ValueError("visible scores must be finite")

    selected = np.full(count, -1, dtype=np.int32)
    selected_count = min(visible_groups, count)
    if selected_count:
        order = np.argsort(-values[:visible_groups], kind="stable")
        selected[:selected_count] = order[:selected_count].astype(np.int32)
    return selected


def expand_group_indices(
    selected_groups: np.ndarray,
    *,
    query_position: int,
    context_length: int,
    compress_ratio: int,
    token_topk: int,
) -> np.ndarray:
    """Expand groups for a contiguous zero-based text prefix."""
    if isinstance(query_position, bool) or not isinstance(query_position, int):
        raise TypeError("query_position must be an int")
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise TypeError("context_length must be an int")
    if query_position < 0:
        raise ValueError("query_position must be nonnegative")
    if context_length < 0:
        raise ValueError("context_length must be nonnegative")
    visible_length = min(query_position + 1, context_length)
    return expand_visible_group_indices(
        selected_groups,
        visible_indices=np.arange(visible_length, dtype=np.int64),
        compress_ratio=compress_ratio,
        token_topk=token_topk,
    )


def expand_visible_group_indices(
    selected_groups: np.ndarray,
    *,
    visible_indices: np.ndarray,
    compress_ratio: int,
    token_topk: int,
) -> np.ndarray:
    """Expand selected groups of actual visible ids and append their tail."""
    ratio = _positive_integer("compress_ratio", compress_ratio)
    requested_tokens = _positive_integer("token_topk", token_topk)
    if requested_tokens % ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    block_topk = requested_tokens // ratio
    groups = np.asarray(selected_groups)
    if groups.ndim != 1 or groups.size != block_topk:
        raise ValueError("selected_groups has an invalid shape")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("selected_groups must contain integers")

    block_rows, tail = visible_group_rows(visible_indices, compress_ratio=ratio)
    visible = block_rows.shape[0]
    expanded_groups = min(visible, block_topk)
    active_groups = groups[:expanded_groups].astype(np.int64, copy=False)
    if np.any(active_groups < 0) or np.any(active_groups >= visible):
        raise ValueError("active selected groups must be visible")

    output = np.full(requested_tokens + ratio - 1, -1, dtype=np.int32)
    column = 0
    for group in active_groups:
        for logical_token in block_rows[int(group)]:
            output[column] = logical_token
            column += 1

    for logical_token in tail:
        output[column] = logical_token
        column += 1
    return output


def selected_index_mask(
    logical_indices: np.ndarray,
    *,
    context_length: int,
    visible_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Convert selected ids to the set-valued mask used by official QSA."""
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise TypeError("context_length must be an int")
    if context_length < 0:
        raise ValueError("context_length must be nonnegative")
    indices = np.asarray(logical_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("logical_indices must be a one-dimensional integer array")
    mask = np.zeros(context_length, dtype=np.bool_)
    valid = indices[(indices >= 0) & (indices < context_length)].astype(np.int64, copy=False)
    mask[valid] = True
    if visible_indices is not None:
        visible = _ordered_visible_indices(visible_indices, context_length=context_length)
        visibility = np.zeros(context_length, dtype=np.bool_)
        visibility[visible] = True
        mask &= visibility
    return mask


def sparse_grouped_query_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logical_indices: np.ndarray,
    *,
    softmax_scale: float | None = None,
    visible_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Apply grouped-query attention over selected full-resolution KV rows."""
    q = np.asarray(query, dtype=np.float32)
    k = np.asarray(keys, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    indices = np.asarray(logical_indices)
    if q.ndim != 2 or not all(q.shape):
        raise ValueError("query must have shape [query_heads, head_dim]")
    if k.ndim != 3 or v.shape != k.shape:
        raise ValueError("keys and values must have matching [tokens, kv_heads, head_dim]")
    if k.shape[2] != q.shape[1] or not k.shape[1]:
        raise ValueError("query and KV head dimensions must match")
    if q.shape[0] % k.shape[1]:
        raise ValueError("query heads must form equal groups over KV heads")
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("logical_indices must be a one-dimensional integer array")
    scale = q.shape[1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")

    mask = selected_index_mask(
        indices,
        context_length=k.shape[0],
        visible_indices=visible_indices,
    )
    valid_indices = np.flatnonzero(mask)
    output = np.zeros_like(q, dtype=np.float32)
    if not valid_indices.size:
        return output

    group_size = q.shape[0] // k.shape[1]
    for kv_head in range(k.shape[1]):
        first_head = kv_head * group_size
        last_head = first_head + group_size
        selected_keys = k[valid_indices, kv_head]
        selected_values = v[valid_indices, kv_head]
        logits = (q[first_head:last_head] @ selected_keys.T) * scale
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits).astype(np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        output[first_head:last_head] = probabilities @ selected_values
    return output


def qsa_reference_decode(
    *,
    index_query: np.ndarray,
    compressed_keys: np.ndarray,
    attention_query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    query_position: int,
    context_length: int,
    token_topk: int,
    compress_ratio: int = 4,
    score_divisor: float | None = None,
    softmax_scale: float | None = None,
    visible_indices: np.ndarray | None = None,
) -> QSAReferenceResult:
    """Run the release-independent QSA selection and attention semantics."""
    ratio = _positive_integer("compress_ratio", compress_ratio)
    requested_tokens = _positive_integer("token_topk", token_topk)
    if requested_tokens % ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    scores = score_compressed_groups(
        index_query,
        compressed_keys,
        score_divisor=score_divisor,
    )
    if visible_indices is None:
        visible_tokens = np.arange(min(query_position + 1, context_length), dtype=np.int64)
    else:
        visible_tokens = _ordered_visible_indices(visible_indices, context_length=context_length)
    visible = visible_tokens.size // ratio
    if visible > scores.size:
        raise ValueError("compressed keys do not cover every visible group")
    selected = stable_topk_groups(
        scores,
        visible_groups=visible,
        block_topk=requested_tokens // ratio,
    )
    logical_indices = expand_visible_group_indices(
        selected,
        visible_indices=visible_tokens,
        compress_ratio=ratio,
        token_topk=requested_tokens,
    )
    output = sparse_grouped_query_attention(
        attention_query,
        keys,
        values,
        logical_indices,
        softmax_scale=softmax_scale,
        visible_indices=visible_tokens,
    )
    return QSAReferenceResult(
        scores=scores,
        visible_groups=visible,
        selected_groups=selected,
        logical_indices=logical_indices,
        output=output,
    )
