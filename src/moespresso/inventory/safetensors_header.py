"""Read safetensors headers only, never the weight bytes.

safetensors layout: 8-byte little-endian header length, then a JSON object
mapping tensor name -> {dtype, shape, data_offsets}, plus an optional
"__metadata__" key. We only need name/shape/dtype, so this stays tiny and
dependency-free (no mlx, no torch).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorHeader:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str          # file name (not full path) the tensor lives in
    # Byte offsets for streaming the weight later (the probe needs these; the
    # inventory does not, so they default to 0 and stay absent from header-only
    # construction in tests).
    header_size: int = 0  # 8 + JSON-header length: where tensor data begins
    begin: int = 0        # this tensor's start byte, relative to data region
    end: int = 0          # this tensor's end byte, relative to data region


def read_header(path: Path) -> dict[str, dict]:
    """Parse one safetensors file's JSON header (no weight bytes read)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def read_shard_metadata(path: Path) -> dict[str, str]:
    """One shard's `__metadata__` map (string -> string; {} when absent).

    The bundle format carries its per-layer expert geometry here (see
    package/bundle.py), so the expert index stays header-only.
    """
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header.get("__metadata__")
    return dict(meta) if isinstance(meta, dict) else {}


def read_headers_with_offsets(path: Path) -> list[TensorHeader]:
    """Headers for one shard, carrying byte offsets for streaming reads.

    `header_size` = 8 + JSON length (the start of the data region); `begin`/`end`
    are the tensor's data_offsets within that region. Absolute byte position of a
    tensor's first byte is `header_size + begin`.
    """
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        raw = json.loads(f.read(n))
    raw.pop("__metadata__", None)
    header_size = 8 + n
    out: list[TensorHeader] = []
    for name, meta in raw.items():
        begin, end = meta["data_offsets"]
        out.append(TensorHeader(
            name=name, shape=tuple(meta["shape"]), dtype=meta["dtype"],
            shard=path.name, header_size=header_size, begin=begin, end=end,
        ))
    return out


def _shard_files(model_dir: Path) -> list[Path]:
    """All safetensors shards, via the index if present else by glob."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
        names = sorted(set(weight_map.values()))
        return [model_dir / n for n in names]
    return sorted(model_dir.glob("*.safetensors"))


def scan_headers(model_dir: Path) -> list[TensorHeader]:
    """Every tensor across all shards as TensorHeader records (headers only)."""
    out: list[TensorHeader] = []
    for shard in _shard_files(model_dir):
        if not shard.exists():
            continue
        for name, meta in read_header(shard).items():
            out.append(TensorHeader(
                name=name,
                shape=tuple(meta["shape"]),
                dtype=meta["dtype"],
                shard=shard.name,
            ))
    return out
