"""Small TurboQuant stored-array reference.

This module does not import `jang_tools` at runtime. It decodes the stored MJTQ
TQ arrays directly and provides a small correctness reference for sampled rows.

The `_beta_pdf` and `compute_codebook` functions derive from JANG v2.5.29
commit `e0c5a81fb34a63f1547030902044a4b99d3f2345` under Apache-2.0.
Modification notice: MoEspresso uses a local NumPy integration compatibility
helper, returns an immutable tuple, and exposes the functions through this
stored-array reference API. See `THIRD-PARTY-NOTICES` and
`LICENSE-APACHE-2.0`.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


def unpack_tq_indices(packed: np.ndarray, bits: int, in_features: int) -> np.ndarray:
    """Unpack uint32-packed TQ codebook indices, low bits first within each word."""
    if bits <= 0 or bits > 8:
        raise ValueError(f"unsupported TQ bits: {bits}")
    packed = np.asarray(packed, dtype=np.uint32)
    if packed.ndim != 2:
        raise ValueError(f"packed rows must be 2D, got {packed.shape}")
    vals_per_word = 32 // bits
    mask = (1 << bits) - 1
    parts = [((packed >> (i * bits)) & mask).astype(np.uint8) for i in range(vals_per_word)]
    flat = np.stack(parts, axis=-1).reshape(packed.shape[0], -1)
    if flat.shape[1] < in_features:
        raise ValueError(
            f"packed rows hold {flat.shape[1]} values, need {in_features}")
    return flat[:, :in_features]


def generate_random_signs(dim: int, seed: int = 0) -> np.ndarray:
    """Deterministic +/-1 signs for the randomized Hadamard rotation."""
    return np.random.default_rng(seed).choice([-1.0, 1.0], size=dim).astype(np.float32)


def _pow2_blocks(dim: int) -> list[int]:
    blocks = []
    left = dim
    while left > 0:
        block = 1 << (left.bit_length() - 1)
        blocks.append(block)
        left -= block
    return blocks


def _hadamard(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"Hadamard block width must be a power of two, got {d}")
    y = np.asarray(x, dtype=np.float32)
    h = 1
    while h < d:
        prefix = y.shape[:-1]
        y = y.reshape(*prefix, d // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = np.concatenate([a + b, a - b], axis=-1).reshape(*prefix, d)
        h *= 2
    return y * (1.0 / math.sqrt(d))


def hadamard_rotate(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply H * diag(signs), blockwise for non-power-of-two widths."""
    x = np.asarray(x, dtype=np.float32)
    signs = np.asarray(signs, dtype=np.float32)
    parts = []
    off = 0
    for width in _pow2_blocks(x.shape[-1]):
        parts.append(_hadamard(x[..., off:off + width] * signs[off:off + width]))
        off += width
    return np.concatenate(parts, axis=-1)


def hadamard_inverse(y: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply diag(signs) * H, the inverse of `hadamard_rotate`."""
    y = np.asarray(y, dtype=np.float32)
    signs = np.asarray(signs, dtype=np.float32)
    parts = []
    off = 0
    for width in _pow2_blocks(y.shape[-1]):
        parts.append(_hadamard(y[..., off:off + width]) * signs[off:off + width])
        off += width
    return np.concatenate(parts, axis=-1)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(y, x))


def _beta_pdf(x: np.ndarray, dim: int) -> np.ndarray:
    if dim <= 2:
        return np.ones_like(x, dtype=np.float64) * 0.5
    log_const = (
        math.lgamma(dim / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma((dim - 1) / 2.0)
    )
    safe = np.maximum(1.0 - x ** 2, 1e-30)
    return np.exp(log_const + ((dim - 3) / 2.0) * np.log(safe))


@lru_cache(maxsize=64)
def compute_codebook(dim: int, bits: int, n_iter: int = 200) -> tuple[float, ...]:
    """Lloyd-Max scalar codebook for TQ's rotated unit-vector coordinates."""
    if bits < 1:
        return (0.0,)
    n_codes = 1 << bits
    grid = np.linspace(-1.0, 1.0, 10000)
    pdf = _beta_pdf(grid, dim)
    total = _trapezoid(pdf, grid)
    if total > 0:
        pdf = pdf / total
    support = 3.0 / math.sqrt(max(dim, 1))
    centroids = np.linspace(-support, support, n_codes)
    for _ in range(n_iter):
        bounds = np.concatenate(([-1.0], (centroids[:-1] + centroids[1:]) / 2.0, [1.0]))
        new = np.zeros(n_codes)
        for i in range(n_codes):
            lo, hi = bounds[i], bounds[i + 1]
            mask = (grid >= lo) & (grid < hi)
            if i == n_codes - 1:
                mask = (grid >= lo) & (grid <= hi)
            if mask.sum() <= 1:
                new[i] = centroids[i]
                continue
            mass = _trapezoid(pdf[mask], grid[mask])
            moment = _trapezoid(grid[mask] * pdf[mask], grid[mask])
            new[i] = moment / max(mass, 1e-10)
        if np.allclose(centroids, new, atol=1e-10):
            break
        centroids = new
    return tuple(sorted(float(x) for x in centroids))


def tq_decode_rows(
    packed_rows: np.ndarray, norms: np.ndarray, bits: int, in_features: int, seed: int = 42,
) -> np.ndarray:
    """Decode sampled TQ rows from stored arrays to float32 `[rows, in_features]`."""
    idx = unpack_tq_indices(packed_rows, bits, in_features)
    codebook = np.array(compute_codebook(in_features, bits), dtype=np.float32)
    rotated = codebook[idx.astype(np.int64)] * np.asarray(norms, dtype=np.float32)[:, None]
    signs = generate_random_signs(in_features, seed)
    return hadamard_inverse(rotated, signs).astype(np.float32)
