"""IQ_K codec geometry and wire layouts, shared by writers and readers.

The IQ_K members are ik's own block codecs. A quantized row is an optional
row-wide scale (the row meta) followed by `in_features / 256` fixed-size
blocks, so a row's byte count is `row_meta_bytes + n_blocks * bytes_per_block`
and the effective rate depends on the row width. The block struct sizes below
are the library's own `static_assert`ed sizes and the row meta sizes are its
`row_meta_size` type traits; nothing here re-derives them from bit counts.

Two wire layouts exist. `ik_wire` is the byte layout the CPU quantizer emits,
row-major within an expert, which is what the conversion artifacts hold.
`iqk_relayout` is reserved for the decode kernels' own layout: a package
records which one its bundles carry, so a package can be rebuilt into the
relayout from the same encoded bytes without re-running the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

# The as-converted layout: ik's row-major block stream, one row after another
# inside an expert. Readers that dequantize with the CPU reference consume it
# directly.
IQK_LAYOUT_IK_WIRE = "ik_wire"
# The decode kernels' layout. Reserved: a package declares it only after a
# build step has actually rearranged the bytes.
IQK_LAYOUT_IQK_RELAYOUT = "iqk_relayout"
IQK_LAYOUTS = (IQK_LAYOUT_IK_WIRE, IQK_LAYOUT_IQK_RELAYOUT)
# The spelling packages carried before the name was corrected to `iqk`. It is
# a value inside shard metadata and manifests, so packages already written
# hold it and rewriting them would mean shipping their bytes again. Readers
# accept it through `normalize_iqk_layout`; nothing writes it, and a test
# pins that.
IQK_LAYOUT_LEGACY_RELAYOUT = "ikq_relayout"


@dataclass(frozen=True)
class IQKCodecGeometry:
    """One IQ_K member's struct facts."""

    ggml_type: int
    bits: int
    weights_per_block: int
    bytes_per_block: int
    row_meta_bytes: int

    def blocks_per_row(self, in_features: int) -> int:
        if in_features <= 0 or in_features % self.weights_per_block:
            raise ValueError(
                f"in_features {in_features} is not a positive multiple of "
                f"{self.weights_per_block}")
        return in_features // self.weights_per_block

    def bytes_per_row(self, in_features: int) -> int:
        return self.row_meta_bytes + self.blocks_per_row(in_features) * self.bytes_per_block

    def in_features_for_row_bytes(self, bytes_per_row: int) -> int:
        """Invert `bytes_per_row`; raises when the byte count is not a row."""
        payload = int(bytes_per_row) - self.row_meta_bytes
        if payload <= 0 or payload % self.bytes_per_block:
            raise ValueError(
                f"bytes_per_row {bytes_per_row} is not {self.row_meta_bytes} B of row "
                f"meta plus whole {self.bytes_per_block} B blocks")
        return (payload // self.bytes_per_block) * self.weights_per_block

    def bpw(self, in_features: int) -> float:
        """Exact bits per weight at this row width, row meta included."""
        return self.bytes_per_row(in_features) * 8.0 / in_features


# `bits` is the member's nominal band, not its rate: two members can share a
# band and differ in rate, so the codec name is what identifies a member.
#
# `iq1_s_r4` is a four-row-group codec: the wire is groups of four rows, each
# group led by four f16 row scales, so the library accounts the prefix as 2
# bytes of row meta per row and a tensor must hold a multiple of four rows.
# Byte offsets are meaningful only at group boundaries, and the reference
# decode consumes a whole group per call, so any consumer that slices wire
# expands a row to its whole group first.
#
# Registration here is a build-side fact and does not imply a serving path.
# Which members serve is the kernel repository's own declaration, which the
# relayout and installer read; the relayout and serving paths fail closed on
# members it does not carry. `iq2_ks`, `iq2_k`, and `iq1_s_r4` serve today;
# `iq3_k`, `iq4_ks`, `iq4_k`, `iq5_k`, and `iq6_k` are registered for
# conversion artifacts and comparators only.
IQK_GEOMETRY = {
    "iq1_s_r4": IQKCodecGeometry(219, 1, 32, 6, 2),
    "iq2_ks": IQKCodecGeometry(145, 2, 256, 70, 2),
    "iq2_k": IQKCodecGeometry(137, 2, 256, 76, 0),
    "iq3_k": IQKCodecGeometry(138, 3, 256, 110, 0),
    "iq4_ks": IQKCodecGeometry(144, 4, 256, 136, 4),
    "iq4_k": IQKCodecGeometry(139, 4, 256, 144, 0),
    "iq5_k": IQKCodecGeometry(140, 5, 256, 176, 0),
    "iq6_k": IQKCodecGeometry(141, 6, 256, 212, 0),
}

# IQ_K members a dense tensor may declare. The dense side of a DS4 package
# holds the attention projections, the shared expert, and the head; the 4-6
# bit members are the dense band this project serves, and the 1-3 bit routed
# members refuse on dense rows so an allocation typo cannot build a package
# whose dense error class no gate has scored.
IQK_DENSE_MEMBERS = ("iq4_ks", "iq4_k", "iq5_k", "iq6_k")

class IQKFormatError(ValueError):
    """Unknown IQ_K member, unknown layout, or a byte count that is not a row."""


def iqk_geometry(codec: str) -> IQKCodecGeometry:
    geometry = IQK_GEOMETRY.get(codec)
    if geometry is None:
        raise IQKFormatError(
            f"unknown IQ_K codec {codec!r}; known: {sorted(IQK_GEOMETRY)}")
    return geometry


def validate_iqk_layout(layout: str) -> str:
    if layout not in IQK_LAYOUTS:
        raise IQKFormatError(
            f"unknown IQ_K wire layout {layout!r}; known: {list(IQK_LAYOUTS)}")
    return layout


def normalize_iqk_layout(layout: object) -> object:
    """Map a recorded layout value onto its current spelling.

    Readers pass a package's declared layout through here, so a package
    written before the name was corrected serves without its bytes being
    rewritten. Derived artifacts also use it to emit the current spelling
    while preserving existing shard bytes. Anything else is returned unchanged,
    so each caller keeps its own refusal and its own message. New package plans
    go through `validate_iqk_layout`, which knows only the current spellings.
    """
    if layout == IQK_LAYOUT_LEGACY_RELAYOUT:
        return IQK_LAYOUT_IQK_RELAYOUT
    return layout


def iqk_dense_geometry(codec: str) -> IQKCodecGeometry:
    """Geometry for a dense IQ_K member; refuses routed-only members."""
    if codec not in IQK_DENSE_MEMBERS:
        raise IQKFormatError(
            f"IQ_K member {codec!r} is not a dense member; dense tensors "
            f"support {list(IQK_DENSE_MEMBERS)}")
    return IQK_GEOMETRY[codec]
