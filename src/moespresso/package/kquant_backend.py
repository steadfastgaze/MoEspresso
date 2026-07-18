"""Lazy mlx-kquant encode/decode bridge.

The package recipe path is pure; this module is the compute edge. It imports
`mlx` and `mlx_kquant` only when an explicitly invoked K-quant path calls it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from moespresso.package.deepseek_v4.recipe import DS4KQuantDenseTarget, DS4KQuantExpertTarget
from moespresso.package.kquant_format import IMATRIX_STEERED_CODECS, encode_stream_for_codec
from moespresso.package.kquant_recipe import (
    validate_kquant_target_fit,
)
from moespresso.package.qwen.recipe import QwenKQuantDenseTarget, QwenKQuantExpertTarget


class KQuantBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class KQuantRuntime:
    mx: Any
    kq: Any


@dataclass(frozen=True)
class KQuantEncodedWeight:
    codec: str
    weight: np.ndarray
    scales: np.ndarray


def is_kquant_module(module: object) -> bool:
    """Duck-typed K-quant module detection; never use isinstance here."""
    return getattr(module, "mode", None) == "kquant"


def _load_kquant_runtime() -> KQuantRuntime:
    try:
        import mlx.core as mx
        import mlx_kquant as kq
    except ImportError as exc:  # pragma: no cover - exercised by tests via monkeypatch
        raise KQuantBackendError(
            "mlx-kquant is required for the K-quant package path. Reinstall MoEspresso "
            "so `mlx_kquant` and ABI-compatible `mlx` are available."
        ) from exc
    return KQuantRuntime(mx=mx, kq=kq)


def check_kquant_backend_available() -> None:
    """Fail early when an explicit K-quant encode/load path lacks mlx-kquant."""
    _load_kquant_runtime()


def _stream(runtime: KQuantRuntime, stream: str | None):
    if stream is None:
        return None
    if stream not in {"cpu", "gpu"}:
        raise KQuantBackendError(f"unsupported mlx-kquant stream {stream!r}")
    return getattr(runtime.mx, stream)


def _mx_dtype(runtime: KQuantRuntime, dtype) -> Any:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return runtime.mx.float32
    if dtype == np.dtype(np.float16):
        return runtime.mx.float16
    raise KQuantBackendError(f"unsupported K-quant decode dtype {dtype}")


def _imatrix_arg(
    runtime: KQuantRuntime,
    target: (
        DS4KQuantExpertTarget
        | DS4KQuantDenseTarget
        | QwenKQuantExpertTarget
        | QwenKQuantDenseTarget
    ),
    imatrix_vectors: dict[str, np.ndarray],
):
    vec = imatrix_vectors.get(target.imatrix_key)
    if vec is None or target.codec not in IMATRIX_STEERED_CODECS:
        return None
    return runtime.mx.array(np.asarray(vec, dtype=np.float32))


def encode_kquant_weight(
    weight: np.ndarray,
    target: (
        DS4KQuantExpertTarget
        | DS4KQuantDenseTarget
        | QwenKQuantExpertTarget
        | QwenKQuantDenseTarget
    ),
    imatrix_vectors: dict[str, np.ndarray],
    *,
    stream: str | None = None,
    runtime: KQuantRuntime | None = None,
) -> KQuantEncodedWeight:
    """Encode one float weight matrix into mlx-kquant wire bytes.

    `weight` is logical `[out_features, in_features]`. Fit validation runs before
    importing or touching the backend so bad orientation/imatrix evidence fails
    cheaply.

    `stream=None` (the default) auto-selects the encode stream per codec: the
    GPU stream for codecs with a Metal encoder (the K-quant/legacy families),
    the CPU stream for the `iq*` codecs that have no GPU encoder. The GPU encode
    is bit-identical to the CPU encode and ~30x faster, so it is the right
    default; pass an explicit "cpu"/"gpu" only to override.
    """
    weight_np = np.asarray(weight, dtype=np.float32)
    validate_kquant_target_fit(target, weight_np.shape, imatrix_vectors)
    runtime = runtime or _load_kquant_runtime()

    resolved_stream = stream if stream is not None else encode_stream_for_codec(target.codec)
    w = runtime.mx.array(weight_np)
    imatrix = _imatrix_arg(runtime, target, imatrix_vectors)
    wq, scales = runtime.kq.quantize(
        w,
        target.codec,
        imatrix=imatrix,
        stream=_stream(runtime, resolved_stream),
    )
    runtime.mx.eval(wq, scales)
    return KQuantEncodedWeight(
        codec=target.codec,
        weight=np.ascontiguousarray(np.asarray(wq, dtype=np.uint8)),
        scales=np.ascontiguousarray(np.asarray(scales, dtype=np.uint8)),
    )


def decode_kquant_weight(
    encoded: KQuantEncodedWeight,
    *,
    dtype=np.float32,
    stream: str | None = "cpu",
    runtime: KQuantRuntime | None = None,
) -> np.ndarray:
    """Decode K-quant wire bytes through mlx-kquant for spot checks."""
    runtime = runtime or _load_kquant_runtime()
    out = runtime.kq.dequantize(
        runtime.mx.array(encoded.weight),
        runtime.mx.array(encoded.scales),
        encoded.codec,
        dtype=_mx_dtype(runtime, dtype),
        stream=_stream(runtime, stream),
    )
    runtime.mx.eval(out)
    return np.asarray(out, dtype=dtype)


def kquant_roundtrip_relative_error(
    weight: np.ndarray,
    target: (
        DS4KQuantExpertTarget
        | DS4KQuantDenseTarget
        | QwenKQuantExpertTarget
        | QwenKQuantDenseTarget
    ),
    imatrix_vectors: dict[str, np.ndarray],
    *,
    stream: str | None = "cpu",
    runtime: KQuantRuntime | None = None,
) -> tuple[KQuantEncodedWeight, float]:
    """Encode, decode, and report relative Frobenius error for a spot check."""
    runtime = runtime or _load_kquant_runtime()
    encoded = encode_kquant_weight(
        weight,
        target,
        imatrix_vectors,
        stream=stream,
        runtime=runtime,
    )
    decoded = decode_kquant_weight(encoded, stream=stream, runtime=runtime)
    weight_np = np.asarray(weight, dtype=np.float32)
    err = float(
        np.linalg.norm(decoded.astype(np.float32) - weight_np)
        / (np.linalg.norm(weight_np) + 1e-6)
    )
    return encoded, err
