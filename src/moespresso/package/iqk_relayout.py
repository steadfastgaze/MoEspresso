"""The IQ_K relayout as a bundle payload: one definition for both sides.

An IQ_K bundle stores one opaque `blocks` component per projection, shaped
`[out_features, bytes_per_row]` uint8. The `ik_wire` layout fills a row with
the quantizer's own super-block stream. The `iqk_relayout` layout fills the
same row, at the same byte count, with that row's relayout streams laid
end to end in the format's own stream order.

The byte count is what makes the two layouts interchangeable inside one
bundle component: `mlx_iqk` spends the member's exact budget in either
placement (`IQ2_KS` is 2.1875 bits per weight plus 16 bits per row, `IQ2_K`
is 2.375 bits per weight, `IQ1_S_R4` is 1.5 bits per weight plus 16 bits
per row), so a relayout row is the same width as the wire row it replaces
and nothing about the bundle's shapes, offsets, or row stride moves. Only
the bytes inside the component change, and the projection's `layout` field
says which placement they are on.

Keeping every stream of a row inside that row, rather than grouping a
stream across the rows of an expert, keeps the declared component shape
literally true: element `[r]` of `blocks` is still row `r`'s payload, so a
whole-row read stays a whole-row read. For a member whose ik wire
interleaves rows (`iq1_s_r4` stores four-row groups), that statement holds
on the relayout side only: a wire row is addressable in whole groups, which
:func:`wire_group_rows` exposes so a consumer can slice wire at group
boundaries, and the relayout is what restores per-row addressability.

The relayout build step and the serving installer both read this module, so
the placement cannot drift between the bytes on disk and the arrays the
kernels receive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mlx_iqk import format as iqk

from moespresso.package.iqk_format import IQKFormatError, iqk_geometry

# IQ_K members with a relayout definition and decode kernels. The package
# format's geometry table is wider; the members below are the ones a package
# can be rearranged onto and served from.
RELAYOUT_MEMBERS = tuple(iqk.MEMBERS)


def wire_group_rows(codec: str) -> int:
    """Rows one addressable ik-wire unit of this member covers.

    1 for the per-row wires; 4 for `iq1_s_r4`, whose quantizer interleaves
    four rows into one group so wire bytes are meaningful only at group
    boundaries. A consumer slicing wire rows for a reference decode must
    slice whole groups.
    """
    check_relayout_member(codec)
    return int(iqk.WIRE_GROUP_ROWS[codec])


@dataclass(frozen=True)
class RelayoutStreamSpan:
    """One relayout stream's byte span inside a single expert row."""

    name: str
    dtype: np.dtype
    shape: tuple[int, ...]      # per-row trailing shape ( () for a row scalar )
    offset: int
    nbytes: int


def check_relayout_member(codec: str) -> str:
    if codec not in RELAYOUT_MEMBERS:
        raise IQKFormatError(
            f"IQ_K member {codec!r} has no relayout; relayout members: "
            f"{list(RELAYOUT_MEMBERS)}")
    return codec


def relayout_stream_spans(codec: str, in_features: int) -> tuple[RelayoutStreamSpan, ...]:
    """Byte spans of every relayout stream inside one expert row.

    Derived from the kernel repository's own component shapes, so a stream
    added or resized there moves this definition with it.
    """
    check_relayout_member(codec)
    shapes = iqk.component_shapes(codec, 1, 1, in_features)
    dtypes = iqk.component_dtypes(codec)
    spans: list[RelayoutStreamSpan] = []
    offset = 0
    for name, shape in shapes.items():
        trailing = tuple(int(d) for d in shape[2:])
        count = 1
        for dim in trailing:
            count *= dim
        nbytes = count * dtypes[name].itemsize
        spans.append(RelayoutStreamSpan(name, dtypes[name], trailing, offset, nbytes))
        offset += nbytes
    return tuple(spans)


def relayout_row_bytes(codec: str, in_features: int) -> int:
    """Bytes one relayout row occupies. Equal to the wire row it replaces."""
    spans = relayout_stream_spans(codec, in_features)
    total = sum(span.nbytes for span in spans)
    wire = iqk_geometry(codec).bytes_per_row(in_features)
    if total != wire:
        raise IQKFormatError(
            f"{codec} relayout row is {total} B against {wire} B of wire at "
            f"in_features {in_features}; the two layouts must share a row width")
    return total


def pack_rows(codec: str, wire_rows: np.ndarray, in_features: int) -> np.ndarray:
    """ik wire rows `[rows, row_bytes]` -> relayout rows of the same shape."""
    check_relayout_member(codec)
    row_bytes = relayout_row_bytes(codec, in_features)
    wire_rows = np.ascontiguousarray(wire_rows, dtype=np.uint8)
    if wire_rows.ndim != 2 or wire_rows.shape[1] != row_bytes:
        raise IQKFormatError(
            f"{codec} wire rows must be [rows, {row_bytes}], got "
            f"{list(wire_rows.shape)}")
    rows = wire_rows.shape[0]
    streams = iqk.pack(codec, wire_rows, in_features)
    out = np.empty((rows, row_bytes), dtype=np.uint8)
    for span in relayout_stream_spans(codec, in_features):
        flat = np.ascontiguousarray(streams[span.name]).reshape(rows, -1).view(np.uint8)
        out[:, span.offset:span.offset + span.nbytes] = flat
    return out


def split_streams(codec: str, blocks: np.ndarray, in_features: int) -> dict[str, np.ndarray]:
    """Relayout rows `[..., row_bytes]` -> the format's streams.

    Leading axes pass through, so a whole stacked projection
    `[experts, out_features, row_bytes]` returns the stacked stream shapes
    the switch module loads.
    """
    check_relayout_member(codec)
    row_bytes = relayout_row_bytes(codec, in_features)
    if blocks.dtype != np.uint8 or blocks.shape[-1] != row_bytes:
        raise IQKFormatError(
            f"{codec} relayout rows must be uint8 ending in {row_bytes}, got "
            f"{blocks.dtype} {list(blocks.shape)}")
    lead = tuple(int(d) for d in blocks.shape[:-1])
    out: dict[str, np.ndarray] = {}
    for span in relayout_stream_spans(codec, in_features):
        part = np.ascontiguousarray(blocks[..., span.offset:span.offset + span.nbytes])
        out[span.name] = part.view(span.dtype).reshape(*lead, *span.shape)
    return out


def unpack_rows(codec: str, rows: np.ndarray, in_features: int) -> np.ndarray:
    """Relayout rows -> the ik wire rows they were packed from."""
    streams = split_streams(codec, rows, in_features)
    return iqk.unpack(codec, streams, in_features)


def decode_rows(codec: str, rows: np.ndarray, in_features: int) -> np.ndarray:
    """Reference float32 dequantization of relayout rows."""
    return iqk.decode(codec, split_streams(codec, rows, in_features), in_features)


__all__ = [
    "RELAYOUT_MEMBERS",
    "RelayoutStreamSpan",
    "check_relayout_member",
    "decode_rows",
    "pack_rows",
    "relayout_row_bytes",
    "relayout_stream_spans",
    "split_streams",
    "unpack_rows",
    "wire_group_rows",
]
