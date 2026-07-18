"""Package cold-start expert hotlist from imatrix counts.

Pins: (1) the GGUF expert-counts reader handles per-expert count vectors;
(2) the payload builder ranks per layer and is schema-compatible with
load_expert_hotlist; (3) the fail-closed alignment contract: any layer-set
mismatch between imatrix block indices and package routed layers emits
nothing (a wrong hotlist would silently seed the wrong layers' experts).
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from moespresso.package.hotlist import (
    HOTLIST_NAME,
    HotlistAlignmentError,
    build_package_expert_hotlist,
    write_package_expert_hotlist,
    write_package_expert_hotlist_from_payload,
)
from moespresso.probe.calibration import imatrix_expert_counts

_GGUF_MAGIC = 0x46554747
_F32 = 0


def _gguf_string(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def _write_gguf(path, tensors):
    """tensors: list of (name, float32 ndarray). Minimal GGUF v3 writer that
    preserves each tensor's dimensions (the imatrix expert counts are 2D)."""
    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, len(tensors), 0)
    infos = bytearray()
    offset = 0
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, np.float32)
        infos += _gguf_string(name.encode())
        infos += struct.pack("<I", arr.ndim)
        for d in arr.shape:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", _F32)
        infos += struct.pack("<Q", offset)
        offset += arr.nbytes
    head = header + bytes(infos)
    pad = (-len(head)) % 32
    with open(path, "wb") as f:
        f.write(head)
        f.write(b"\x00" * pad)
        for _name, arr in tensors:
            f.write(np.ascontiguousarray(arr, np.float32).tobytes())


def _imatrix_with_expert_counts(path, layer_counts, extra=()):
    tensors = list(extra)
    for layer, counts in layer_counts.items():
        tensors.append((f"blk.{layer}.ffn_gate_exps.weight.counts",
                        np.asarray(counts, np.float32).reshape(1, -1)))
    _write_gguf(path, tensors)


def test_expert_counts_reader_handles_2d_counts_and_skips_others(tmp_path):
    p = tmp_path / "im.gguf"
    _imatrix_with_expert_counts(
        p,
        {0: [5, 1, 9, 3], 7: [2, 2, 2, 2]},
        extra=[("blk.0.attn_q.weight.in_sum2", np.ones(8, np.float32)),
               ("blk.0.attn_q.weight.counts", np.ones(1, np.float32)),
               ("blk.0.ffn_up_exps.weight.counts",
                np.asarray([[9, 9, 9, 9]], np.float32))],
    )
    counts = imatrix_expert_counts(p)
    assert sorted(counts) == [0, 7]  # gate only; attn/up keys ignored
    np.testing.assert_array_equal(counts[0], [5, 1, 9, 3])


def test_build_payload_ranks_and_matches_loader_schema():
    payload = build_package_expert_hotlist(
        {0: np.array([5.0, 1.0, 9.0, 0.0]), 1: np.array([1.0, 2.0, 3.0, 4.0])},
        layers_indexed=(0, 1),
        num_experts=4,
        source={"imatrix_sha256": "abc"},
    )
    assert payload["kind"] == "expert_hotlist"  # load_expert_hotlist's gate
    assert payload["source"]["kind"] == "gguf_imatrix_counts"
    assert payload["source"]["imatrix_sha256"] == "abc"
    # ranked, zero-count experts omitted
    assert list(payload["layers"]["0"]) == ["2", "0", "1"]
    assert payload["layers"]["0"]["2"] == 9


def test_build_payload_slices_counts_for_smoke_packages():
    payload = build_package_expert_hotlist(
        {0: np.array([1.0, 9.0, 5.0, 100.0])},  # full-model counters
        layers_indexed=(0,),
        num_experts=2,  # smoke package kept only the first 2 experts
        source=None,
    )
    assert list(payload["layers"]["0"]) == ["1", "0"]  # expert 3 ignored


def test_alignment_mismatch_fails_closed():
    with pytest.raises(HotlistAlignmentError, match="refusing"):
        build_package_expert_hotlist(
            {0: np.ones(4), 1: np.ones(4)},
            layers_indexed=(0, 2),  # package routed layers differ
            num_experts=4,
        )
    with pytest.raises(HotlistAlignmentError, match="counters"):
        build_package_expert_hotlist(
            {0: np.ones(2)},
            layers_indexed=(0,),
            num_experts=4,  # more experts than counters: indexing suspicion
        )


def test_write_package_hotlist_end_to_end_on_bundle_package(tmp_path):
    from conftest import write_bundle_package

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    write_bundle_package(pkg, n_layers=2, n_exp=4)
    im = tmp_path / "im.gguf"
    _imatrix_with_expert_counts(im, {0: [5, 1, 9, 3], 1: [1, 2, 3, 4]})

    n = write_package_expert_hotlist(pkg, im, imatrix_identity={"sha256": "s"})
    assert n == 2
    payload = json.loads((pkg / HOTLIST_NAME).read_text())
    assert payload["kind"] == "expert_hotlist"
    assert payload["source"]["imatrix_sha256"] == "s"
    assert list(payload["layers"]["0"]) == ["2", "0", "3", "1"]


def test_write_payload_revalidates_alignment_and_keeps_source(tmp_path):
    from conftest import write_bundle_package

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    write_bundle_package(pkg, n_layers=2, n_exp=4)
    payload = {
        "version": 1,
        "kind": "expert_hotlist",
        "source": {"kind": "gguf_imatrix_counts", "imatrix_name": "im.gguf"},
        "layers": {"0": {"2": 9, "0": 5, "3": 3, "1": 1},
                   "1": {"3": 4, "2": 3, "1": 2, "0": 1}},
    }
    assert write_package_expert_hotlist_from_payload(pkg, payload) == 2
    out = json.loads((pkg / HOTLIST_NAME).read_text())
    assert out["kind"] == "expert_hotlist"
    assert out["source"]["imatrix_name"] == "im.gguf"
    assert list(out["layers"]["0"]) == ["2", "0", "3", "1"]
    assert out["layers"]["0"]["2"] == 9

    # A payload from a different layer layout refuses to write.
    with pytest.raises(HotlistAlignmentError, match="refusing"):
        write_package_expert_hotlist_from_payload(
            pkg, dict(payload, layers={"0": {"1": 1}}))


def test_write_payload_slices_for_smoke_and_skips_empty_or_dense(tmp_path):
    from conftest import write_bundle_package

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    write_bundle_package(pkg, n_layers=1, n_exp=2)
    # Full-model ranking against a smoke package that kept the first 2
    # experts: entries beyond the kept set are ignored.
    payload = {"kind": "expert_hotlist",
               "layers": {"0": {"3": 100, "1": 9, "0": 1}}}
    assert write_package_expert_hotlist_from_payload(pkg, payload) == 1
    out = json.loads((pkg / HOTLIST_NAME).read_text())
    assert list(out["layers"]["0"]) == ["1", "0"]

    assert write_package_expert_hotlist_from_payload(pkg, {"layers": {}}) == 0
    dense = tmp_path / "dense"
    dense.mkdir()
    assert write_package_expert_hotlist_from_payload(dense, payload) == 0
    assert not (dense / HOTLIST_NAME).exists()


def test_write_package_hotlist_skips_dense_imatrix_and_dense_package(tmp_path):
    from conftest import write_bundle_package

    # dense imatrix (no expert counts): nothing written
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    write_bundle_package(pkg, n_layers=1, n_exp=4)
    im = tmp_path / "im.gguf"
    _write_gguf(im, [("blk.0.attn_q.weight.in_sum2", np.ones(8, np.float32)),
                     ("blk.0.attn_q.weight.counts", np.ones(1, np.float32))])
    assert write_package_expert_hotlist(pkg, im) == 0
    assert not (pkg / HOTLIST_NAME).exists()

    # dense package (no routed experts): nothing written
    dense = tmp_path / "dense"
    dense.mkdir()
    im2 = tmp_path / "im2.gguf"
    _imatrix_with_expert_counts(im2, {0: [1, 2, 3, 4]})
    assert write_package_expert_hotlist(dense, im2) == 0
