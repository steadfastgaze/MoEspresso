"""Dense IQ_K rows use a relayout contract separate from routed artifacts."""

from __future__ import annotations

import numpy as np
import pytest

from moespresso.package.iqk_dense_relayout import (
    DENSE_RELAYOUT_MEMBERS,
    dense_relayout_implementation_identity,
    pack_dense_rows,
    split_dense_streams,
    unpack_dense_rows,
)
from moespresso.package.iqk_format import iqk_dense_geometry


@pytest.mark.parametrize("member", DENSE_RELAYOUT_MEMBERS)
def test_dense_rows_round_trip_every_wire_byte(member: str) -> None:
    rng = np.random.default_rng(29)
    row_bytes = iqk_dense_geometry(member).bytes_per_row(512)
    wire = rng.integers(0, 256, size=(7, row_bytes), dtype=np.uint8)

    packed = pack_dense_rows(member, wire, 512)
    streams = split_dense_streams(member, packed, 512)
    restored = unpack_dense_rows(member, packed, 512)

    assert packed.shape == wire.shape
    assert set(streams)
    assert np.array_equal(restored, wire)


def test_dense_relayout_identity_does_not_claim_routed_members() -> None:
    identity = dense_relayout_implementation_identity()

    assert identity["schema"] == "moespresso_iqk_dense_relayout_v1"
    assert identity["members"] == list(DENSE_RELAYOUT_MEMBERS)
    assert set(identity["members"]).isdisjoint({"iq2_k", "iq2_ks"})
    assert len(identity["identity_sha256"]) == 64
