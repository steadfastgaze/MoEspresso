"""Calibration evidence for the probe: imatrix providers.

Calibration is a required input: the probe weights reconstruction error by how
much each input channel actually drives activations on real data. A calibration
provider returns two things:

  - vectors: {gguf_key -> per-input-channel importance h = in_sum2 / count}, the
    diagonal-Hessian activation energy the activation-weighted quality uses;
  - identity: the calibration-dataset identity (name, source, size_bytes, sha256,
    key_count, sampling), so probe_evidence is tied to exactly which calibration
    produced it.

GGUF imatrix files and llama.cpp legacy `.dat` files are supported. Other
providers (e.g. a forward-pass calibration) can implement the same
`(vectors, identity)` shape later. GGUF files are memory-mapped, and legacy
files are read entry by entry, so a large imatrix never lands in RAM whole.
"""

from __future__ import annotations

import hashlib
import mmap
import re
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from moespresso.probe.gguf_parse import GGUFBufferParser

_GGUF_MAGIC = 0x46554747
_DS4_LEGACY_EXPERT_WIDTHS = {
    "gate": 4096,
    "up": 4096,
    "down": 2048,
}
_DS4_LEGACY_EXPERT_KEY = re.compile(
    r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$"
)


def read_imatrix_vectors(path: str | Path) -> dict[str, np.ndarray]:
    """Per-tensor per-input-channel importance vector h = in_sum2 / count (float32).

    Dense tensors store 1D statistics (in_sum2 [in], counts [1]). Stacked
    expert tensors store 2D ones (in_sum2 [n_experts, in] and a count per
    expert), and the per-channel importance is the corpus aggregate
    h_j = sum_e in_sum2[e, j] / sum_e counts[e]. A naive reader that took
    dimensions[0] floats = only expert 0's row over only expert 0's count
    would yield a 1-of-256 sample of the statistic the file already
    holds in full.

    A tensor with total count <= 0 maps to an all-zero vector. Keyed by GGUF
    base name (e.g. "blk.3.ffn_down.weight").
    """
    if _is_gguf(path):
        return _read_gguf_imatrix_vectors(path)
    if not _looks_like_legacy_imatrix(path):
        raise ValueError(
            f"unsupported imatrix format for {Path(path)}: expected GGUF or "
            f"llama.cpp legacy .dat")
    return _read_legacy_imatrix_vectors(path)


def _read_gguf_imatrix_vectors(path: str | Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for base, in_sum2, counts in _iter_channel_sums(path):
        total = float(counts.sum())
        h = in_sum2.sum(axis=0) if in_sum2.ndim == 2 else in_sum2
        if total > 0:
            out[base] = (h / total).astype(np.float32)
        else:
            out[base] = np.zeros(h.shape[0], dtype=np.float32)
    return out


def _read_i32_file(f, path: Path) -> int:
    raw = f.read(4)
    if len(raw) != 4:
        raise ValueError(f"truncated legacy imatrix {path}")
    return struct.unpack("<i", raw)[0]


def _legacy_entry_vector(name: str, values: np.ndarray) -> np.ndarray:
    m = _DS4_LEGACY_EXPERT_KEY.match(name)
    if m is None:
        return values.astype(np.float32, copy=False)
    width = _DS4_LEGACY_EXPERT_WIDTHS[m.group(1)]
    if values.size <= width or values.size % width != 0:
        return values.astype(np.float32, copy=False)
    # ds4 writes one flat [expert, input] entry. The probe needs one per-input
    # importance vector for the logical expert unit, so collapse experts.
    return values.reshape(values.size // width, width).mean(axis=0).astype(np.float32)


def _read_legacy_imatrix_vectors(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        n_entries = _read_i32_file(f, path)
        if n_entries < 1:
            raise ValueError(f"legacy imatrix {path} has no entries")
        for _ in range(n_entries):
            name_len = _read_i32_file(f, path)
            if name_len < 0:
                raise ValueError(f"legacy imatrix {path} has negative name length")
            raw_name = f.read(name_len)
            if len(raw_name) != name_len:
                raise ValueError(f"truncated legacy imatrix name in {path}")
            name = raw_name.decode("utf-8")
            ncall = _read_i32_file(f, path)
            nval = _read_i32_file(f, path)
            if nval < 1:
                raise ValueError(f"legacy imatrix entry {name!r} has no values")
            raw_values = f.read(nval * 4)
            if len(raw_values) != nval * 4:
                raise ValueError(f"truncated legacy imatrix values for {name!r}")
            values = np.frombuffer(raw_values, dtype=np.float32).copy()
            if ncall > 0:
                values /= float(ncall)
            out[name] = _legacy_entry_vector(name, values)
    return out


def _is_gguf(path: str | Path) -> bool:
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(4)
    return len(raw) == 4 and struct.unpack("<I", raw)[0] == _GGUF_MAGIC


def _looks_like_legacy_imatrix(path: str | Path) -> bool:
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(8)
    if len(raw) < 8:
        return False
    n_entries, first_name_len = struct.unpack("<ii", raw)
    return 1 <= n_entries <= 1_000_000 and 1 <= first_name_len <= 4096


def _imatrix_kind(path: str | Path) -> str:
    if _is_gguf(path):
        return "gguf_imatrix"
    if _looks_like_legacy_imatrix(path):
        return "legacy_imatrix"
    return "imatrix_file"


def calibration_identity(path: str | Path, vectors: dict[str, np.ndarray]) -> dict:
    """The spec's calibration-dataset identity for an imatrix file.

    `sampling` records how the importance was derived (here: per-channel mean of
    in_sum2/count over the calibration set, the diagonal Hessian). The sha256 +
    size pin exactly which file produced the probe evidence.
    """
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {
        "kind": _imatrix_kind(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
        "key_count": len(vectors),
        "sampling": "per_channel_in_sum2_over_counts",  # h_j = E[x_j^2]
    }


def imatrix_calibration(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load an imatrix calibration file as (vectors, identity).

    The provider contract the probe consumes. `vectors` feeds activation-weighted
    quality; `identity` is recorded verbatim in probe_evidence so the evidence is
    tied to its calibration source.
    """
    vectors = read_imatrix_vectors(path)
    return vectors, calibration_identity(path, vectors)


# ---------------------------------------------------------------------------
# GGUF imatrix reading
# ---------------------------------------------------------------------------

def _iter_channel_sums(path: str | Path) -> Iterator[tuple[str, np.ndarray, float]]:
    """Yield (base, in_sum2 [1D or [n_exp, in]], counts array) per tensor.

    Each in_sum2 is a fresh float32 array copied out of the memory map (only the
    needed bytes fault in), so it stays valid after iteration and leaves no
    exported pointer that would block mm.close().
    """
    path = Path(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            parser, data_offset = _parse_header(mm, path)
            for base, sum2_off, sum2_shape, counts_off, counts_n in _pairs(
                    parser, data_offset):
                n = 1
                for d in sum2_shape:
                    n *= d
                in_sum2 = np.frombuffer(
                    mm[sum2_off : sum2_off + 4 * n], dtype=np.float32)
                if len(sum2_shape) == 2:
                    # GGUF dims are innermost-first: [in, n_experts] on disk
                    # is logically [n_experts, in], row-major
                    in_sum2 = in_sum2.reshape(sum2_shape[1], sum2_shape[0])
                counts = np.frombuffer(
                    mm[counts_off : counts_off + 4 * counts_n],
                    dtype=np.float32)
                yield base, in_sum2.copy(), counts.copy()
        finally:
            mm.close()


def _parse_header(mm: mmap.mmap, path: Path) -> tuple[GGUFBufferParser, int]:
    """Parse only the GGUF header from the mapped file; return (parser, data_offset).

    Feeds the parser in chunks so only header bytes are copied into memory, never
    the (much larger) tensor-data section that follows.
    """
    parser = GGUFBufferParser()
    pos, size, chunk = 0, len(mm), 1 << 20
    while not parser.is_complete():
        if pos >= size:
            parser.try_parse()
            break
        end = min(pos + chunk, size)
        parser.feed(mm[pos:end])
        parser.try_parse()
        pos = end
    if parser.header is None:
        raise ValueError(f"Failed to parse GGUF header from {path}")
    # GGUF aligns the tensor-data section to 32 bytes after the header.
    data_offset = ((parser.total_consumed() + 31) // 32) * 32
    return parser, data_offset


def _pairs(
    parser: GGUFBufferParser, data_offset: int
) -> Iterator[tuple[str, int, int, int]]:
    """Match <base>.in_sum2 with <base>.counts; yield offsets + shapes.

    in_sum2 must precede its counts in the file (as imatrix writers emit them).
    """
    pending: dict[str, tuple[int, tuple]] = {}  # base -> (sum2_off, dims)
    for ti in parser.tensor_infos:
        if ti.name.endswith(".in_sum2"):
            base = ti.name[: -len(".in_sum2")]
            pending[base] = (data_offset + ti.offset, tuple(ti.dimensions))
        elif ti.name.endswith(".counts"):
            base = ti.name[: -len(".counts")]
            if base in pending:
                sum2_off, dims = pending.pop(base)
                counts_n = 1
                for d in ti.dimensions:
                    counts_n *= d
                yield (base, sum2_off, dims,
                       data_offset + ti.offset, counts_n)


# ---------------------------------------------------------------------------
# Routed-expert usage counts (residency: cold-start hotlist source)
# ---------------------------------------------------------------------------

_EXPERT_COUNTS_KEY = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight\.counts$")


def imatrix_expert_counts(path: str | Path) -> dict[int, np.ndarray]:
    """Per-layer routed-expert usage counters from a GGUF imatrix.

    llama.cpp's imatrix records, for stacked expert tensors, a per-expert
    `.counts` vector (how often each expert was routed during calibration):
    real routing evidence over the whole calibration corpus. A single probe
    prompt cannot supply this statistic. gate/up/down counts are identical (one router), so gate is read.
    Returns {gguf_block_index: float32[num_experts]}; empty for a dense
    imatrix. The caller must verify block indices align with the package's
    routed layers before using these (package/hotlist.py does, fail-closed).
    """
    path = Path(path)
    if not _is_gguf(path):
        return {}
    out: dict[int, np.ndarray] = {}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            parser, data_offset = _parse_header(mm, path)
            for ti in parser.tensor_infos:
                m = _EXPERT_COUNTS_KEY.match(ti.name)
                if m is None:
                    continue
                n = 1
                for d in ti.dimensions:
                    n *= int(d)
                start = data_offset + ti.offset
                out[int(m.group(1))] = np.frombuffer(
                    mm[start:start + 4 * n], dtype=np.float32).copy().reshape(-1)
        finally:
            mm.close()
    return out
