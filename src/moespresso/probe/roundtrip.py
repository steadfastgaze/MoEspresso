"""Quantize -> dequantize round-trips for formats MoEspresso handles.

  - affine_roundtrip:  mlx affine quant (non-expert weights).
  - mx_float_roundtrip: mlx MX float quant (dense mxfp4/mxfp8 candidates).
  - tq_reconstruct:    TurboQuant (stacked MoE experts), via jang_tools.

Both return a float32 reconstruction the caller scores with probe.quality. The
heavy deps (mlx, jang_tools) are imported lazily and degrade to a clear error if
absent, so the rest of the package imports without requiring them.
"""

from __future__ import annotations

import numpy as np

from moespresso.probe.quality import activation_weighted_quality, cosine


def _mx():
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError("mlx is required for round-trip measurement") from e
    return mx


def _flush(mx) -> None:
    mx.eval()
    mx.clear_cache()


def affine_roundtrip(weight_f32: np.ndarray, bits: int, group_size: int = 64) -> np.ndarray:
    """mlx affine quantize->dequantize; float32 reconstruction, same shape."""
    mx = _mx()
    w = mx.array(weight_f32)
    qw, scales, biases = mx.quantize(w, group_size=group_size, bits=bits, mode="affine")
    w_hat = mx.dequantize(qw, scales, biases, group_size=group_size, bits=bits)
    mx.eval(w_hat)
    out = np.array(w_hat, dtype=np.float32)
    del w, qw, scales, biases, w_hat
    _flush(mx)
    return out


def affine_quality(
    weight_f32: np.ndarray, bits: int, group_size: int, importance: np.ndarray,
) -> tuple[float, float]:
    """(cosine, activation_weighted_quality) of one affine round-trip."""
    recon = affine_roundtrip(weight_f32, bits, group_size)
    return cosine(weight_f32, recon), activation_weighted_quality(weight_f32, recon, importance)


def mx_float_roundtrip(weight_f32: np.ndarray, mode: str) -> np.ndarray:
    """MLX MX float quantize->dequantize; float32 reconstruction, same shape."""
    if mode not in {"mxfp4", "mxfp8"}:
        raise ValueError(f"unsupported MX float mode {mode!r}")
    mx = _mx()
    w = mx.array(weight_f32)
    qw, scales = mx.quantize(w, mode=mode)
    w_hat = mx.dequantize(qw, scales, mode=mode).astype(mx.float32)
    mx.eval(w_hat)
    out = np.array(w_hat, dtype=np.float32)
    del w, qw, scales, w_hat
    _flush(mx)
    return out


def mx_float_quality(
    weight_f32: np.ndarray, mode: str, importance: np.ndarray,
) -> tuple[float, float]:
    """(cosine, activation_weighted_quality) of one MX float round-trip."""
    recon = mx_float_roundtrip(weight_f32, mode)
    return cosine(weight_f32, recon), activation_weighted_quality(weight_f32, recon, importance)


def _tq():
    try:
        from jang_tools.turboquant.codebook import compute_codebook
        from jang_tools.turboquant.linear import tq_quantize_weight
        from jang_tools.turboquant.pipeline import unpack_bits
        from jang_tools.turboquant.rotation import generate_random_signs, hadamard_inverse
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError("jang_tools is required for TQ round-trip") from e
    return (compute_codebook, tq_quantize_weight, unpack_bits,
            generate_random_signs, hadamard_inverse)


def tq_reconstruct(weight_f32: np.ndarray, bits: int, seed: int = 42) -> np.ndarray:
    """TurboQuant quantize->dequantize a (small) weight sample; float32 recon.

    Full round-trip: quantize -> unpack bits -> codebook lookup -> scale by row
    norm -> inverse Hadamard. Expects an already-subsampled array.
    """
    mx = _mx()
    compute_codebook, tq_quantize_weight, unpack_bits, gen_signs, had_inv = _tq()
    sample = weight_f32.astype(np.float32)
    in_features = sample.shape[1]

    result = tq_quantize_weight(sample, bits=bits, seed=seed)
    _flush(mx)
    packed_np, norms_np = result["packed"], result["norms"]

    cb = mx.array(np.array(compute_codebook(in_features, bits), dtype=np.float32))
    signs = gen_signs(in_features, seed)
    mx.eval(signs)

    rows = []
    for r in range(sample.shape[0]):
        idx = unpack_bits(mx.array(packed_np[r]), bits, in_features)
        row = mx.take(cb, idx.astype(mx.uint32)) * float(norms_np[r])
        mx.eval(row)
        rows.append(row)
    w_dequant = had_inv(mx.stack(rows), signs)
    mx.eval(w_dequant)
    out = np.array(w_dequant, dtype=np.float32)
    del cb, signs, rows, w_dequant
    _flush(mx)
    return out


def tq_quality(
    weight_f32: np.ndarray, bits: int, importance: np.ndarray, seed: int = 42,
) -> tuple[float, float]:
    """(cosine, activation_weighted_quality) of one TQ round-trip."""
    recon = tq_reconstruct(weight_f32, bits, seed)
    return cosine(weight_f32, recon), activation_weighted_quality(weight_f32, recon, importance)
