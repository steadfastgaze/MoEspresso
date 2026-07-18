"""Stream weight bytes from safetensors shards, one tensor (or expert slice)
at a time, never the whole model.

The probe must run in a few GB on a 35B model, so we never materialize a full
tensor: we seek to a tensor's byte range, optionally read only a random subset of
rows (for 2D affine weights) or a random subset of experts (for stacked-3D MoE
tensors), and convert just that sample to float32. BF16 is kept as uint16 until
the last moment to avoid a 2x bloat.

Built on MoEspresso's header reader. No mlx here: this is the I/O edge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moespresso.inventory.safetensors_header import (
    TensorHeader,
    read_headers_with_offsets,
)

# numpy view dtype per safetensors dtype tag (BF16 has no native numpy dtype, so
# we carry it as uint16 and shift into float32 by hand).
_VIEW_DTYPE = {
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "FLOAT": np.float32,
    "U8": np.uint8,
    "U32": np.uint32,
    "I32": np.int32,
    "I64": np.int64,
}


def scan_offsets(model_dir: Path) -> dict[str, TensorHeader]:
    """Map every tensor name -> its TensorHeader (with byte offsets), header-only."""
    catalog: dict[str, TensorHeader] = {}
    for sf in sorted(Path(model_dir).glob("*.safetensors")):
        for h in read_headers_with_offsets(sf):
            catalog[h.name] = h
    return catalog


def _bytes_to_float32(raw: bytes, dtype_tag: str, shape: tuple[int, ...]) -> np.ndarray:
    """Decode raw tensor bytes to a float32 array of `shape`."""
    if dtype_tag == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return np.frombuffer(u32.tobytes(), dtype=np.float32).reshape(shape)
    view = _VIEW_DTYPE.get(dtype_tag)
    if view is None:
        raise ValueError(f"unsupported dtype {dtype_tag}")
    return np.frombuffer(raw, dtype=view).astype(np.float32).reshape(shape)


def _bytes_to_raw(raw: bytes, dtype_tag: str, shape: tuple[int, ...]) -> np.ndarray:
    """Decode raw tensor bytes preserving the safetensors storage dtype."""
    view = _VIEW_DTYPE.get(dtype_tag)
    if view is None:
        raise ValueError(f"unsupported dtype {dtype_tag}")
    return np.frombuffer(raw, dtype=view).reshape(shape)


def _read_range(model_dir: Path, h: TensorHeader, byte_start: int, nbytes: int) -> bytes:
    """Read `nbytes` from shard `h.shard` at data offset `byte_start` (within tensor)."""
    with open(Path(model_dir) / h.shard, "rb") as f:
        f.seek(h.header_size + h.begin + byte_start)
        return f.read(nbytes)


def load_2d_sample(
    model_dir: Path, h: TensorHeader, sample_rows: int, seed: int,
) -> np.ndarray:
    """Load up to `sample_rows` random rows of a 2D tensor as float32.

    Row-subsampling preserves per-column structure, which is exactly what the
    activation-weighted quality needs (the importance vector is per-column).
    Rows are contiguous on disk, so we read the whole tensor's bytes then index,
    cheap relative to the model, and 2D non-expert weights are small.
    """
    if len(h.shape) != 2:
        raise ValueError(f"{h.name} is not 2D: {h.shape}")
    rows, cols = h.shape
    elem = np.dtype(_VIEW_DTYPE[h.dtype]).itemsize
    raw = _read_range(model_dir, h, 0, rows * cols * elem)
    full = _bytes_to_float32(raw, h.dtype, (rows, cols))
    if rows <= sample_rows:
        return full
    idx = np.random.default_rng(seed).choice(rows, sample_rows, replace=False)
    return full[idx]


def sample_indices(total: int, requested: int, seed: int) -> np.ndarray:
    """Deterministic sorted sample indices. Empty only when `total` is zero."""
    n = min(max(0, requested), total)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == total:
        return np.arange(total, dtype=np.int64)
    return np.array(sorted(np.random.default_rng(seed).choice(total, n, replace=False).tolist()),
                    dtype=np.int64)


def load_2d_rows_raw(model_dir: Path, h: TensorHeader, rows: np.ndarray) -> np.ndarray:
    """Load selected rows of a 2D tensor, preserving storage dtype."""
    if len(h.shape) != 2:
        raise ValueError(f"{h.name} is not 2D: {h.shape}")
    n_rows, cols = h.shape
    elem = np.dtype(_VIEW_DTYPE[h.dtype]).itemsize
    row_bytes = cols * elem
    out = []
    for row in rows.tolist():
        if row < 0 or row >= n_rows:
            raise IndexError(f"row {row} outside {h.name} with {n_rows} rows")
        raw = _read_range(model_dir, h, row * row_bytes, row_bytes)
        out.append(_bytes_to_raw(raw, h.dtype, (1, cols))[0])
    return np.stack(out, axis=0) if out else np.empty((0, cols), dtype=_VIEW_DTYPE[h.dtype])


def load_2d_rows(model_dir: Path, h: TensorHeader, rows: np.ndarray) -> np.ndarray:
    """Load selected rows of a 2D tensor as float32."""
    raw = load_2d_rows_raw(model_dir, h, rows)
    if h.dtype == "BF16":
        return _bytes_to_float32(raw.tobytes(), h.dtype, raw.shape)
    return raw.astype(np.float32)


def load_expert_sample(
    model_dir: Path, h: TensorHeader, n_experts: int, seed: int,
) -> np.ndarray:
    """Load `n_experts` random expert slices of a stacked-3D tensor [E, rows, cols].

    Seeks directly to each chosen expert's byte offset, limiting peak memory to
    the sampled experts. Returns them concatenated along rows:
    shape [n_sampled*rows, cols], float32.
    """
    if len(h.shape) != 3:
        raise ValueError(f"{h.name} is not 3D: {h.shape}")
    total, rows, cols = h.shape
    elem = np.dtype(_VIEW_DTYPE[h.dtype]).itemsize
    expert_bytes = rows * cols * elem
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(total, min(n_experts, total), replace=False).tolist())
    slices = []
    for e in chosen:
        raw = _read_range(model_dir, h, e * expert_bytes, expert_bytes)
        slices.append(_bytes_to_float32(raw, h.dtype, (rows, cols)))
    return np.concatenate(slices, axis=0)


def load_3d_rows_raw(
    model_dir: Path, h: TensorHeader, expert_indices: np.ndarray, row_indices: np.ndarray,
) -> np.ndarray:
    """Load selected [expert,row,:] slices of a 3D tensor, preserving dtype."""
    if len(h.shape) != 3:
        raise ValueError(f"{h.name} is not 3D: {h.shape}")
    n_experts, rows, cols = h.shape
    elem = np.dtype(_VIEW_DTYPE[h.dtype]).itemsize
    row_bytes = cols * elem
    out = []
    for expert in expert_indices.tolist():
        if expert < 0 or expert >= n_experts:
            raise IndexError(f"expert {expert} outside {h.name} with {n_experts} experts")
        expert_rows = []
        for row in row_indices.tolist():
            if row < 0 or row >= rows:
                raise IndexError(f"row {row} outside {h.name} with {rows} rows")
            offset = ((expert * rows) + row) * row_bytes
            raw = _read_range(model_dir, h, offset, row_bytes)
            expert_rows.append(_bytes_to_raw(raw, h.dtype, (1, cols))[0])
        out.append(np.stack(expert_rows, axis=0))
    shape = (len(expert_indices), len(row_indices), cols)
    return np.stack(out, axis=0) if out else np.empty(shape, dtype=_VIEW_DTYPE[h.dtype])


def load_3d_rows(
    model_dir: Path, h: TensorHeader, expert_indices: np.ndarray, row_indices: np.ndarray,
) -> np.ndarray:
    """Load selected [expert,row,:] slices of a 3D tensor as float32."""
    raw = load_3d_rows_raw(model_dir, h, expert_indices, row_indices)
    if h.dtype == "BF16":
        return _bytes_to_float32(raw.tobytes(), h.dtype, raw.shape)
    return raw.astype(np.float32)


def load_full(model_dir: Path, h: TensorHeader) -> np.ndarray:
    """Read a whole tensor of any rank as float32. For small structural tensors
    (norms, SSM state) carried as passthrough. They're tiny, no streaming needed."""
    elem = np.dtype(_VIEW_DTYPE[h.dtype]).itemsize
    nbytes = int(np.prod(h.shape)) * elem
    raw = _read_range(model_dir, h, 0, nbytes)
    return _bytes_to_float32(raw, h.dtype, tuple(h.shape))


def load_full_raw(model_dir: Path, h: TensorHeader) -> np.ndarray:
    """Read a whole tensor preserving storage dtype."""
    elem = np.dtype(_VIEW_DTYPE[h.dtype]).itemsize
    nbytes = int(np.prod(h.shape)) * elem
    raw = _read_range(model_dir, h, 0, nbytes)
    return _bytes_to_raw(raw, h.dtype, tuple(h.shape))


def iter_row_chunks(
    model_dir: Path, h: TensorHeader, max_chunk_bytes: int,
):
    """Yield (start_row, f32_chunk) bands of a 2D tensor, never the whole tensor.

    Reads at most ~max_chunk_bytes of float32 at a time via byte-range seeks, so
    a vocab-sized embed/lm_head converts in a bounded footprint. Mirrors the
    proven convert_moe row-chunked path (`_CHUNK_MAX_BYTES`).
    """
    if len(h.shape) != 2:
        raise ValueError(f"{h.name} is not 2D: {h.shape}")
    rows, cols = h.shape
    elem = np.dtype(_VIEW_DTYPE[h.dtype]).itemsize
    f32_per_row = cols * 4
    rows_per_chunk = max(1, max_chunk_bytes // f32_per_row)
    row_bytes = cols * elem
    for start in range(0, rows, rows_per_chunk):
        n = min(rows_per_chunk, rows - start)
        raw = _read_range(model_dir, h, start * row_bytes, n * row_bytes)
        yield start, _bytes_to_float32(raw, h.dtype, (n, cols))


def iter_experts(model_dir: Path, h: TensorHeader, max_experts: int | None = None):
    """Yield one expert `[rows, cols]` f32 at a time from a stacked-3D tensor.

    Per-expert byte-range seeks limit the peak footprint to one expert.
    `max_experts` caps how many (None = all). Mirrors convert_moe's per-expert
    streaming loop.
    """
    if len(h.shape) != 3:
        raise ValueError(f"{h.name} is not 3D: {h.shape}")
    total, rows, cols = h.shape
    elem = np.dtype(_VIEW_DTYPE[h.dtype]).itemsize
    expert_bytes = rows * cols * elem
    n = total if max_experts is None else min(max_experts, total)
    for e in range(n):
        raw = _read_range(model_dir, h, e * expert_bytes, expert_bytes)
        yield _bytes_to_float32(raw, h.dtype, (rows, cols))


def split_fused_gate_up(sample: np.ndarray, n_sampled: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a concatenated fused gate_up expert sample into (gate, up) halves.

    The on-disk fused tensor stacks gate over up within each expert's rows. The
    sample is `n_sampled` experts concatenated, each [rows, cols] with rows = the
    fused height; gate is the top half, up the bottom half, per expert.
    """
    total_rows = sample.shape[0]
    rows = total_rows // n_sampled
    mid = rows // 2
    gate = np.concatenate(
        [sample[i * rows: i * rows + mid] for i in range(n_sampled)], axis=0)
    up = np.concatenate(
        [sample[i * rows + mid: (i + 1) * rows] for i in range(n_sampled)], axis=0)
    return gate, up
