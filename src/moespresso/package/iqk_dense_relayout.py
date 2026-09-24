"""Dense IQ_K package rows on the direct decode-kernel layout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from pathlib import Path

import numpy as np

from moespresso.package.iqk_cpu import iqk_format
from moespresso.package.iqk_format import (
    IQK_DENSE_MEMBERS,
    IQKFormatError,
    iqk_dense_geometry,
)


iqk = iqk_format()
DENSE_RELAYOUT_MEMBERS = tuple(
    member for member in IQK_DENSE_MEMBERS if member in iqk.DENSE_MEMBERS
)
DENSE_RELAYOUT_IMPLEMENTATION_SCHEMA = "moespresso_iqk_dense_relayout_v1"


@dataclass(frozen=True)
class DenseRelayoutStreamSpan:
    """One direct row stream's byte span."""

    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dense_relayout_implementation_identity() -> dict[str, object]:
    """Return a content identity independent of routed conversion artifacts."""
    try:
        package_version = version("mlx-iqk")
    except PackageNotFoundError as exc:
        raise IQKFormatError("mlx-iqk is not installed") from exc
    body = {
        "schema": DENSE_RELAYOUT_IMPLEMENTATION_SCHEMA,
        "mlx_iqk_version": package_version,
        "modules": {
            "mlx_iqk.format": _sha256_file(Path(inspect.getfile(iqk)).resolve()),
            "moespresso.package.iqk_dense_relayout": _sha256_file(Path(__file__).resolve()),
        },
        "members": list(DENSE_RELAYOUT_MEMBERS),
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return {**body, "identity_sha256": hashlib.sha256(encoded).hexdigest()}


def _check_member(codec: str) -> str:
    if codec not in DENSE_RELAYOUT_MEMBERS:
        raise IQKFormatError(
            f"IQ_K member {codec!r} has no dense relayout; dense members: "
            f"{list(DENSE_RELAYOUT_MEMBERS)}"
        )
    iqk_dense_geometry(codec)
    return codec


def dense_relayout_stream_spans(
    codec: str,
    in_features: int,
) -> tuple[DenseRelayoutStreamSpan, ...]:
    """Return the byte spans stored inside one direct packed row."""
    _check_member(codec)
    shapes = iqk.dense_component_shapes(codec, 1, in_features)
    dtypes = iqk.dense_component_dtypes(codec)
    spans = []
    offset = 0
    for name, shape in shapes.items():
        trailing = tuple(int(value) for value in shape[1:])
        count = 1
        for dimension in trailing:
            count *= dimension
        nbytes = count * dtypes[name].itemsize
        spans.append(
            DenseRelayoutStreamSpan(
                name=name,
                dtype=dtypes[name],
                shape=trailing,
                offset=offset,
                nbytes=nbytes,
            )
        )
        offset += nbytes
    expected = iqk_dense_geometry(codec).bytes_per_row(in_features)
    if offset != expected:
        raise IQKFormatError(f"{codec} dense relayout row is {offset} bytes, expected {expected}")
    return tuple(spans)


def pack_dense_rows(
    codec: str,
    wire_rows: np.ndarray,
    in_features: int,
) -> np.ndarray:
    """Transform quantizer wire rows onto the direct-kernel layout."""
    spans = dense_relayout_stream_spans(codec, in_features)
    row_bytes = sum(span.nbytes for span in spans)
    wire = np.ascontiguousarray(wire_rows, dtype=np.uint8)
    if wire.ndim != 2 or wire.shape[1] != row_bytes:
        raise IQKFormatError(
            f"{codec} wire rows must be [rows, {row_bytes}], got {list(wire.shape)}"
        )
    rows = int(wire.shape[0])
    streams = iqk.pack(codec, wire, in_features)
    packed = np.empty_like(wire)
    for span in spans:
        values = np.ascontiguousarray(streams[span.name]).reshape(rows, -1).view(np.uint8)
        packed[:, span.offset : span.offset + span.nbytes] = values
    return packed


def split_dense_streams(
    codec: str,
    packed_rows: np.ndarray,
    in_features: int,
) -> dict[str, np.ndarray]:
    """Split direct packed rows into the dependency's named streams."""
    spans = dense_relayout_stream_spans(codec, in_features)
    row_bytes = sum(span.nbytes for span in spans)
    packed = np.asarray(packed_rows)
    if packed.dtype != np.uint8 or packed.shape[-1] != row_bytes:
        raise IQKFormatError(
            f"{codec} dense rows must be uint8 ending in {row_bytes}, got "
            f"{packed.dtype} {list(packed.shape)}"
        )
    leading = tuple(int(value) for value in packed.shape[:-1])
    streams = {}
    for span in spans:
        values = np.ascontiguousarray(packed[..., span.offset : span.offset + span.nbytes])
        streams[span.name] = values.view(span.dtype).reshape(
            *leading,
            *span.shape,
        )
    return streams


def unpack_dense_rows(
    codec: str,
    packed_rows: np.ndarray,
    in_features: int,
) -> np.ndarray:
    """Invert :func:`pack_dense_rows` exactly."""
    return iqk.unpack(
        codec,
        split_dense_streams(codec, packed_rows, in_features),
        in_features,
    )


__all__ = [
    "DENSE_RELAYOUT_IMPLEMENTATION_SCHEMA",
    "DENSE_RELAYOUT_MEMBERS",
    "DenseRelayoutStreamSpan",
    "dense_relayout_implementation_identity",
    "dense_relayout_stream_spans",
    "pack_dense_rows",
    "split_dense_streams",
    "unpack_dense_rows",
]
