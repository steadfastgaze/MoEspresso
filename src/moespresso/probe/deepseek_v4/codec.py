"""DeepSeek-V4 source storage codecs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moespresso.inventory.safetensors_header import TensorHeader

FP4_E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

_STORAGE_DTYPE = {
    "I8": np.int8,
    "F8_E4M3": np.uint8,
    "F8_E8M0": np.uint8,
}


def _storage_dtype(header: TensorHeader):
    dtype = _STORAGE_DTYPE.get(header.dtype)
    if dtype is None:
        raise ValueError(f"unsupported DS4 storage dtype {header.dtype!r}")
    return dtype


def _read_storage_range(
    model_dir: Path,
    header: TensorHeader,
    byte_start: int,
    nbytes: int,
) -> bytes:
    with open(Path(model_dir) / header.shard, "rb") as f:
        f.seek(header.header_size + header.begin + byte_start)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(f"short read for {header.name}: {len(raw)} of {nbytes} bytes")
    return raw


def load_storage_tensor(model_dir: Path, header: TensorHeader) -> np.ndarray:
    """Read a DS4 source tensor preserving its storage codes."""
    dtype = _storage_dtype(header)
    nbytes = int(np.prod(header.shape)) * np.dtype(dtype).itemsize
    raw = _read_storage_range(model_dir, header, 0, nbytes)
    return np.frombuffer(raw, dtype=dtype).reshape(header.shape)


def load_storage_rows(
    model_dir: Path,
    header: TensorHeader,
    rows: np.ndarray,
) -> np.ndarray:
    """Read selected rows of a 2D DS4 source tensor preserving storage codes."""
    if len(header.shape) != 2:
        raise ValueError(f"{header.name} is not 2D: {header.shape}")
    dtype = _storage_dtype(header)
    n_rows, cols = header.shape
    row_bytes = cols * np.dtype(dtype).itemsize
    out = []
    for row in np.asarray(rows, dtype=np.int64).tolist():
        if row < 0 or row >= n_rows:
            raise IndexError(f"row {row} outside {header.name} with {n_rows} rows")
        raw = _read_storage_range(model_dir, header, row * row_bytes, row_bytes)
        out.append(np.frombuffer(raw, dtype=dtype))
    shape = (len(out), cols)
    return np.stack(out, axis=0) if out else np.empty(shape, dtype=dtype)


def ue8m0_to_float32(codes: np.ndarray) -> np.ndarray:
    """Decode UE8M0 scale bytes to float32."""
    u = np.asarray(codes, dtype=np.uint8)
    if np.any(u == 0xFF):
        raise ValueError("UE8M0 code 0xFF is reserved")
    return np.exp2(u.astype(np.int16) - 127).astype(np.float32)


def fp8_e4m3_to_float32(codes: np.ndarray) -> np.ndarray:
    """Decode finite E4M3FN bytes using the same layout as MLX/PyTorch."""
    u = np.asarray(codes, dtype=np.uint8)
    if np.any((u == 0x7F) | (u == 0xFF)):
        raise ValueError("E4M3FN codes 0x7F/0xFF are NaN/reserved")
    sign = (u & 0x80) != 0
    abs_codes = (u & 0x7F).astype(np.uint16)
    half_bits = (abs_codes << 7).astype(np.uint16)
    vals = half_bits.view(np.float16).astype(np.float32) * 256.0
    return np.where(sign, -vals, vals).astype(np.float32)


def dequant_fp4_e2m1_ue8m0(
    packed: np.ndarray,
    scale: np.ndarray,
    *,
    fp4_block: int = 32,
    out_dtype=np.float16,
) -> np.ndarray:
    """Dequantize DS4 packed FP4 expert weights with UE8M0 scales."""
    packed_u8 = np.asarray(packed).view(np.uint8)
    if packed_u8.ndim != 2:
        raise ValueError(f"expected 2D packed FP4 tensor, got {packed_u8.ndim}D")
    out_dim, packed_in = packed_u8.shape
    in_dim = packed_in * 2
    if in_dim % fp4_block != 0:
        raise ValueError(f"in_dim {in_dim} is not divisible by fp4_block {fp4_block}")
    expected_scale = (out_dim, in_dim // fp4_block)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(f"scale shape {tuple(scale.shape)} != {expected_scale}")

    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    vals = np.stack([FP4_E2M1_TABLE[low], FP4_E2M1_TABLE[high]], axis=-1).reshape(
        out_dim, in_dim
    )
    scale_expanded = np.repeat(ue8m0_to_float32(scale), fp4_block, axis=1)
    return (vals * scale_expanded).astype(out_dtype)


def dequant_fp8_e4m3_ue8m0(
    weight: np.ndarray,
    scale: np.ndarray,
    *,
    fp8_block: tuple[int, int] = (128, 128),
    out_dtype=np.float16,
) -> np.ndarray:
    """Dequantize DS4 FP8 dense weights with UE8M0 block scales."""
    w = np.asarray(weight, dtype=np.uint8)
    if w.ndim != 2:
        raise ValueError(f"expected 2D FP8 tensor, got {w.ndim}D")
    out_dim, in_dim = w.shape
    b0, b1 = fp8_block
    if out_dim % b0 or in_dim % b1:
        raise ValueError(f"shape {(out_dim, in_dim)} is not divisible by {fp8_block}")
    expected_scale = (out_dim // b0, in_dim // b1)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(f"scale shape {tuple(scale.shape)} != {expected_scale}")

    vals = fp8_e4m3_to_float32(w)
    scale_expanded = np.repeat(np.repeat(ue8m0_to_float32(scale), b0, axis=0), b1, axis=1)
    return (vals * scale_expanded).astype(out_dtype)


def _validate_fp8_block_shapes(
    weight_header: TensorHeader,
    scale: np.ndarray,
    fp8_block: tuple[int, int],
) -> tuple[int, int, int, int]:
    if len(weight_header.shape) != 2:
        raise ValueError(f"{weight_header.name} is not 2D: {weight_header.shape}")
    out_dim, in_dim = weight_header.shape
    b0, b1 = fp8_block
    if out_dim % b0 or in_dim % b1:
        raise ValueError(f"shape {(out_dim, in_dim)} is not divisible by {fp8_block}")
    expected_scale = (out_dim // b0, in_dim // b1)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(f"scale shape {tuple(scale.shape)} != {expected_scale}")
    return out_dim, in_dim, b0, b1


def _scale_rows_for_weight_rows(
    scale: np.ndarray,
    rows: np.ndarray,
    *,
    b0: int,
    b1: int,
    in_dim: int,
) -> np.ndarray:
    scale_f32 = ue8m0_to_float32(scale)
    row_scales = scale_f32[np.asarray(rows, dtype=np.int64) // b0]
    return np.repeat(row_scales, b1, axis=1)[:, :in_dim]


def load_dequantized_fp8_rows(
    model_dir: Path,
    weight_header: TensorHeader,
    scale_header: TensorHeader,
    rows: np.ndarray,
    *,
    fp8_block: tuple[int, int] = (128, 128),
    out_dtype=np.float16,
) -> np.ndarray:
    """Read and dequantize selected rows of one DS4 FP8 source tensor."""
    scale = load_storage_tensor(model_dir, scale_header)
    _out_dim, in_dim, b0, b1 = _validate_fp8_block_shapes(
        weight_header, scale, fp8_block)
    rows = np.asarray(rows, dtype=np.int64)
    vals = fp8_e4m3_to_float32(load_storage_rows(model_dir, weight_header, rows))
    scale_expanded = _scale_rows_for_weight_rows(
        scale, rows, b0=b0, b1=b1, in_dim=in_dim)
    return (vals * scale_expanded).astype(out_dtype)


def iter_dequantized_fp8_row_chunks(
    model_dir: Path,
    weight_header: TensorHeader,
    scale_header: TensorHeader,
    max_chunk_bytes: int,
    *,
    fp8_block: tuple[int, int] = (128, 128),
    out_dtype=np.float32,
):
    """Yield decoded row bands for one DS4 FP8 source tensor."""
    scale = load_storage_tensor(model_dir, scale_header)
    out_dim, in_dim, b0, b1 = _validate_fp8_block_shapes(
        weight_header, scale, fp8_block)
    rows_per_chunk = max(1, max_chunk_bytes // (in_dim * np.dtype(out_dtype).itemsize))
    dtype = _storage_dtype(weight_header)
    row_bytes = in_dim * np.dtype(dtype).itemsize
    for start in range(0, out_dim, rows_per_chunk):
        n = min(rows_per_chunk, out_dim - start)
        raw = _read_storage_range(model_dir, weight_header, start * row_bytes, n * row_bytes)
        codes = np.frombuffer(raw, dtype=dtype).reshape(n, in_dim)
        rows = np.arange(start, start + n, dtype=np.int64)
        vals = fp8_e4m3_to_float32(codes)
        scale_expanded = _scale_rows_for_weight_rows(
            scale, rows, b0=b0, b1=b1, in_dim=in_dim)
        yield start, (vals * scale_expanded).astype(out_dtype)


def load_dequantized_fp4(
    model_dir: Path,
    weight_header: TensorHeader,
    scale_header: TensorHeader,
    *,
    fp4_block: int = 32,
    out_dtype=np.float16,
) -> np.ndarray:
    """Read and dequantize one packed DS4 FP4 source tensor."""
    return dequant_fp4_e2m1_ue8m0(
        load_storage_tensor(model_dir, weight_header),
        load_storage_tensor(model_dir, scale_header),
        fp4_block=fp4_block,
        out_dtype=out_dtype,
    )


def load_dequantized_fp8(
    model_dir: Path,
    weight_header: TensorHeader,
    scale_header: TensorHeader,
    *,
    fp8_block: tuple[int, int] = (128, 128),
    out_dtype=np.float16,
) -> np.ndarray:
    """Read and dequantize one DS4 FP8 source tensor."""
    return dequant_fp8_e4m3_ue8m0(
        load_storage_tensor(model_dir, weight_header),
        load_storage_tensor(model_dir, scale_header),
        fp8_block=fp8_block,
        out_dtype=out_dtype,
    )


def fits_float16(values: np.ndarray) -> bool:
    """Return whether all finite values fit in float16 range."""
    arr = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        return False
    return bool(np.max(np.abs(arr), initial=0.0) <= np.finfo(np.float16).max)
