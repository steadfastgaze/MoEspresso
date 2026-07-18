"""Activation-weighted reconstruction quality: pure functions.

No mlx, no model: given an original weight matrix and its quantized-then-
dequantized reconstruction, plus a per-input-channel importance vector h, score
how well the reconstruction preserves what the weight does.

  q = 1 - Σ_j h_j‖W[:,j]-Ŵ[:,j]‖² / (Σ_j h_j‖W[:,j]‖²)

Uniform h reduces it to normalized MSE; a NaN/Inf or blown-up reconstruction
floors at Q_FLOOR. Robust to row-subsampling (the sampling factor cancels in the
error/energy ratio).
"""

from __future__ import annotations

import numpy as np

Q_FLOOR = -1.0  # worst-case quality sentinel (NaN/Inf or catastrophic error)


def cosine(weight_f32: np.ndarray, recon_f32: np.ndarray) -> float:
    """Cosine similarity of two matrices flattened to vectors (legacy metric)."""
    a = weight_f32.reshape(-1)
    b = recon_f32.reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na * nb > 0 else 0.0


def activation_weighted_quality(
    weight_f32: np.ndarray, recon_f32: np.ndarray, importance: np.ndarray,
) -> float:
    """Activation-weighted reconstruction quality, floored at Q_FLOOR.

    importance h has one entry per input channel (column of `weight_f32`).
    Zero/empty importance falls back to uniform weighting (normalized MSE) rather
    than reporting a tensor as catastrophic just because it lacks imatrix data.
    """
    h = np.asarray(importance, dtype=np.float64)
    diff = weight_f32 - recon_f32
    col_err = np.square(diff).sum(axis=0)            # length = in_features
    col_energy = np.square(weight_f32).sum(axis=0)
    if h.shape[0] != col_err.shape[0]:
        raise ValueError(
            f"importance length {h.shape[0]} != in_features {col_err.shape[0]}"
        )

    denom = float(h @ col_energy)
    if denom <= 0.0:  # no per-channel signal -> uniform fallback (NMSE)
        energy = float(col_energy.sum())
        if energy <= 0.0:
            return 1.0  # zero weight reconstructs perfectly
        q = 1.0 - float(col_err.sum()) / energy
    else:
        q = 1.0 - float(h @ col_err) / denom

    if not np.isfinite(q):
        return Q_FLOOR
    return max(Q_FLOOR, q)
