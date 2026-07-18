"""Pure DeepSeek-V4 attention primitives used by synthetic correctness tests.

These helpers intentionally avoid MLX and model weights. They pin the arithmetic
contracts that the DS4 runtime graph must later preserve: layer-specific RoPE,
partial-tail rotation, sink-aware attention, compressed-pool visibility, and the
lightning indexer score scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from moespresso.inventory.architecture_profile import DEEPSEEK_V4_FLASH_COMPRESS_RATIOS

DS4_HEAD_DIM = 512
DS4_ROPE_DIM = 64
DS4_SWA_ROPE_BASE = 10000.0
DS4_COMPRESS_ROPE_BASE = 160000.0
DS4_YARN = {
    "factor": 16.0,
    "original_seq_len": 65536.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
}


@dataclass(frozen=True)
class DeepseekV4RopePolicy:
    layer_index: int
    compress_ratio: int
    rope_base: float
    yarn: Mapping[str, float] | None

    @property
    def uses_yarn(self) -> bool:
        return self.yarn is not None


def deepseek_v4_rope_policy(
    layer_index: int,
    compress_ratios: Sequence[int] = DEEPSEEK_V4_FLASH_COMPRESS_RATIOS,
) -> DeepseekV4RopePolicy:
    """Return the DS4 RoPE policy for a real layer or the MTP slot.

    `compress_ratios` has 43 model-layer entries plus index 43 for the MTP block.
    A nonzero compression ratio selects the compressed base and YaRN. Ratio zero
    selects regular SWA RoPE with no YaRN.
    """
    if layer_index < 0 or layer_index >= len(compress_ratios):
        raise ValueError(f"layer_index {layer_index} outside compress_ratios")
    compress_ratio = int(compress_ratios[layer_index])
    if compress_ratio == 0:
        return DeepseekV4RopePolicy(
            layer_index=layer_index,
            compress_ratio=compress_ratio,
            rope_base=DS4_SWA_ROPE_BASE,
            yarn=None,
        )
    return DeepseekV4RopePolicy(
        layer_index=layer_index,
        compress_ratio=compress_ratio,
        rope_base=DS4_COMPRESS_ROPE_BASE,
        yarn=DS4_YARN,
    )


def rope_inverse_frequencies(
    dim: int,
    base: float,
    yarn: Mapping[str, float] | None = None,
) -> np.ndarray:
    """RoPE inverse frequencies, including DS4's pure-frequency YaRN variant."""
    if dim <= 0 or dim % 2:
        raise ValueError("RoPE dim must be a positive even integer")
    idx = np.arange(0, dim, 2, dtype=np.float64)
    freqs = 1.0 / (float(base) ** (idx / dim))
    if not yarn:
        return freqs

    factor = float(yarn["factor"])
    original_seq_len = float(yarn["original_seq_len"])
    beta_fast = float(yarn.get("beta_fast", 32.0))
    beta_slow = float(yarn.get("beta_slow", 1.0))
    if factor <= 1.0 or original_seq_len <= 0:
        return freqs

    def correction_dim(n: float) -> float:
        return dim * math.log(original_seq_len / (n * 2.0 * math.pi)) / (
            2.0 * math.log(base)
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = (np.arange(dim // 2, dtype=np.float64) - low) / (high - low)
    smooth = 1.0 - np.clip(ramp, 0.0, 1.0)
    return freqs / factor * (1.0 - smooth) + freqs * smooth


def rope_angles(
    positions: Sequence[int] | np.ndarray,
    *,
    dim: int = DS4_ROPE_DIM,
    base: float,
    yarn: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return position x inverse-frequency RoPE angles."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    return np.outer(pos, rope_inverse_frequencies(dim, base, yarn))


def apply_rope_tail(
    x: np.ndarray,
    positions: Sequence[int] | np.ndarray,
    *,
    base: float,
    yarn: Mapping[str, float] | None = None,
    rope_dim: int = DS4_ROPE_DIM,
    inverse: bool = False,
) -> np.ndarray:
    """Apply DS4 partial RoPE to only the final `rope_dim` channels.

    The sequence axis is the second-to-last axis, matching the model graph shape
    convention. Leading channels are copied unchanged.
    """
    arr = np.asarray(x)
    if arr.ndim < 2:
        raise ValueError("x must have at least sequence and feature dimensions")
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError("rope_dim must be a positive even integer")
    if arr.shape[-1] < rope_dim:
        raise ValueError("rope_dim cannot exceed the feature dimension")
    if arr.shape[-2] != len(np.asarray(positions).reshape(-1)):
        raise ValueError("positions length must match x.shape[-2]")

    out = arr.copy()
    tail = out[..., -rope_dim:]
    pairs = tail.astype(np.float64, copy=False).reshape(*tail.shape[:-1], rope_dim // 2, 2)
    angles = rope_angles(positions, dim=rope_dim, base=base, yarn=yarn)
    cos = np.cos(angles)
    sin = np.sin(angles)
    if inverse:
        sin = -sin
    broadcast_shape = (1,) * (pairs[..., 0].ndim - 2) + cos.shape
    cos = cos.reshape(broadcast_shape)
    sin = sin.reshape(broadcast_shape)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    rotated = np.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), axis=-1)
    out[..., -rope_dim:] = rotated.reshape(tail.shape).astype(out.dtype, copy=False)
    return out


def compressed_visibility(
    query_positions: Sequence[int] | np.ndarray,
    pool_indices: Sequence[int] | np.ndarray,
    ratio: int,
) -> np.ndarray:
    """True-for-valid compressed-pool visibility.

    Pool row `k` covers raw positions `[k * ratio, (k + 1) * ratio)`, so query
    position `q` may see it only when `(k + 1) * ratio <= q + 1`.
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    q = np.asarray(query_positions, dtype=np.int64).reshape(-1)
    k = np.asarray(pool_indices, dtype=np.int64).reshape(-1)
    return ((k[None, :] + 1) * int(ratio)) <= (q[:, None] + 1)


def visible_compressed_count(
    query_positions: Sequence[int] | np.ndarray,
    ratio: int,
) -> np.ndarray:
    """Number of compressed rows visible to each absolute query position."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    q = np.asarray(query_positions, dtype=np.int64).reshape(-1)
    return (q + 1) // int(ratio)


def compressed_rope_positions(
    pool_indices: Sequence[int] | np.ndarray,
    ratio: int,
    *,
    pool_base: int = 0,
) -> np.ndarray:
    """Absolute RoPE positions for compressed-pool rows."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    k = np.asarray(pool_indices, dtype=np.int64).reshape(-1)
    return int(pool_base) + k * int(ratio)


def sink_softmax_attention(
    scores: np.ndarray,
    values: np.ndarray,
    sinks: np.ndarray,
) -> np.ndarray:
    """Softmax attention with DS4's virtual zero-value sink in the denominator."""
    score_arr = np.asarray(scores, dtype=np.float64)
    value_arr = np.asarray(values, dtype=np.float64)
    sink_arr = np.asarray(sinks, dtype=np.float64)
    if score_arr.ndim < 1:
        raise ValueError("scores must have at least one dimension")
    if value_arr.ndim < 2:
        raise ValueError("values must have key and value dimensions")
    if value_arr.shape[-2] != score_arr.shape[-1]:
        raise ValueError("values key dimension must match scores")
    value_arr = np.broadcast_to(value_arr, score_arr.shape + (value_arr.shape[-1],))
    sink_arr = np.broadcast_to(sink_arr, score_arr.shape[:-1])

    max_scores = np.max(score_arr, axis=-1)
    m = np.maximum(max_scores, sink_arr)
    weights = np.exp(score_arr - m[..., None])
    sink_weight = np.exp(sink_arr - m)
    den = np.sum(weights, axis=-1) + sink_weight
    return np.sum(weights[..., None] * value_arr, axis=-2) / den[..., None]


def indexer_weight_scale(index_head_dim: int = 128, n_heads: int = 64) -> float:
    """The DS4 indexer scale: head_dim**-0.5 times n_heads**-0.5."""
    if index_head_dim <= 0 or n_heads <= 0:
        raise ValueError("index_head_dim and n_heads must be positive")
    return (index_head_dim ** -0.5) * (n_heads ** -0.5)


def indexer_weighted_scores(
    raw_scores: np.ndarray,
    head_weights: np.ndarray,
    *,
    index_head_dim: int = 128,
    n_heads: int = 64,
) -> np.ndarray:
    """Apply the DS4 indexer ReLU, per-head weights, sum, and scale."""
    raw = np.asarray(raw_scores, dtype=np.float64)
    weights = np.asarray(head_weights, dtype=np.float64)
    if raw.ndim != 3:
        raise ValueError("raw_scores must have shape (queries, heads, pool_rows)")
    if weights.shape != raw.shape[:2]:
        raise ValueError("head_weights must have shape (queries, heads)")
    relu_scores = np.maximum(raw, 0.0)
    return np.sum(relu_scores * weights[..., None], axis=1) * indexer_weight_scale(
        index_head_dim=index_head_dim,
        n_heads=n_heads,
    )


def causal_indexer_scores(
    raw_scores: np.ndarray,
    head_weights: np.ndarray,
    query_positions: Sequence[int] | np.ndarray,
    pool_indices: Sequence[int] | np.ndarray,
    ratio: int,
    *,
    index_head_dim: int = 128,
    n_heads: int = 64,
) -> np.ndarray:
    """Indexer scores with future compressed blocks masked to `-inf`."""
    weighted = indexer_weighted_scores(
        raw_scores,
        head_weights,
        index_head_dim=index_head_dim,
        n_heads=n_heads,
    )
    mask = compressed_visibility(query_positions, pool_indices, ratio)
    if weighted.shape != mask.shape:
        raise ValueError("score shape does not match query_positions x pool_indices")
    return np.where(mask, weighted, -np.inf)


def causal_indexer_topk(
    raw_scores: np.ndarray,
    head_weights: np.ndarray,
    query_positions: Sequence[int] | np.ndarray,
    pool_indices: Sequence[int] | np.ndarray,
    ratio: int,
    *,
    topk: int = 512,
    index_head_dim: int = 128,
    n_heads: int = 64,
) -> np.ndarray:
    """Top-k compressed-pool indices after DS4 causal masking.

    The returned width is `min(topk, len(pool_indices))`; rows with fewer visible
    blocks are padded with `-1`, matching the sparse-attention skip sentinel.
    """
    if topk <= 0:
        raise ValueError("topk must be positive")
    pool = np.asarray(pool_indices, dtype=np.int64).reshape(-1)
    scores = causal_indexer_scores(
        raw_scores,
        head_weights,
        query_positions,
        pool,
        ratio,
        index_head_dim=index_head_dim,
        n_heads=n_heads,
    )
    width = min(int(topk), pool.shape[0])
    out = np.full((scores.shape[0], width), -1, dtype=np.int64)
    for i, row in enumerate(scores):
        valid = np.flatnonzero(np.isfinite(row))
        if valid.size == 0 or width == 0:
            continue
        order = valid[np.argsort(-row[valid], kind="stable")[:width]]
        out[i, : order.size] = pool[order]
    return out
