"""The IQ_K package builder: allocation reading, converted artifacts, dense side.

Allocation reading is exercised on a synthetic candidate file so the suite runs
from a source release without depending on campaign artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from moespresso.package.deepseek_v4.iqk_package import (
    IQKPackageError,
    IQKRoutedArtifacts,
    allocation_member_counts,
    build_dense_allocations,
    dense_format_counts,
    read_iqk_allocation,
)
from moespresso.package.iqk_format import iqk_geometry

# --------------------------------------------------------------------------
# Candidate allocations


def _synthetic_candidates(tmp_path: Path) -> Path:
    mixed = {
        str(layer): {
            "gate": "iq2_k" if layer == 1 else "iq2_ks",
            "up": "iq2_ks",
            "down": "iq3_k" if layer == 0 else "iq2_ks",
        }
        for layer in range(3)
    }
    floor = {
        str(layer): {p: "iq2_ks" for p in ("gate", "up", "down")}
        for layer in range(3)
    }
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"candidates": [
        {"name": "M1 mixed", "allocation_map": {"per_layer": mixed}},
        {"name": "M2 mixed-secondary", "allocation_map": {"per_layer": mixed}},
        {"name": "A1 floor", "allocation_map": {"per_layer": floor}},
    ]}))
    return path


def test_reads_a_mixed_candidate_by_name_and_by_prefix(tmp_path):
    path = _synthetic_candidates(tmp_path)

    mixed, record = read_iqk_allocation(path, "M1 mixed")
    floor, _floor_record = read_iqk_allocation(path, "A1")

    assert record["name"] == "M1 mixed"
    assert allocation_member_counts(mixed) == {"iq2_k": 1, "iq2_ks": 7, "iq3_k": 1}
    assert allocation_member_counts(floor) == {"iq2_ks": 9}


def test_unknown_ambiguous_and_malformed_candidates_fail_closed(tmp_path):
    path = _synthetic_candidates(tmp_path)

    with pytest.raises(IQKPackageError, match="no candidate named"):
        read_iqk_allocation(path, "Z9")
    with pytest.raises(IQKPackageError, match="ambiguous"):
        read_iqk_allocation(path, "M")

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"candidates": [{
        "name": "X", "allocation_map": {"per_layer": {"0": {
            "gate": "iq2_ks", "up": "iq2_ks", "down": "q2_k"}}},
    }]}))
    with pytest.raises(IQKPackageError, match="not an IQ_K member"):
        read_iqk_allocation(bad, "X")

    short = tmp_path / "short.json"
    short.write_text(json.dumps({"candidates": [{
        "name": "X", "allocation_map": {"per_layer": {"0": {"gate": "iq2_ks"}}},
    }]}))
    with pytest.raises(IQKPackageError, match="not gate/up/down"):
        read_iqk_allocation(short, "X")


# --------------------------------------------------------------------------
# The converted routed stack


def _write_cell(root: Path, layer: int, projection: str, codec: str,
                out_features: int, in_features: int, n_experts: int) -> Path:
    row = iqk_geometry(codec).bytes_per_row(in_features)
    rng = np.random.default_rng(layer * 31 + hash(projection) % 97)
    payload = rng.integers(
        0, 256, size=(n_experts, out_features, row), dtype=np.uint8)
    path = root / f"layer{layer:02d}_{projection}.{codec}"
    path.write_bytes(payload.tobytes())
    return path


def _tiny_stack(tmp_path, members, *, out_features=4, in_features=256, n_experts=2):
    root = tmp_path / "tensors"
    root.mkdir()
    written = {}
    for layer, cells in members.items():
        for projection, codec in cells.items():
            written[(layer, projection)] = _write_cell(
                root, layer, projection, codec, out_features, in_features, n_experts)
    shapes = {
        layer: {p: (out_features, in_features) for p in ("gate", "up", "down")}
        for layer in members
    }
    return root, shapes, written


def test_routed_artifacts_read_experts_at_their_own_offsets(tmp_path):
    members = {0: {"gate": "iq2_ks", "up": "iq2_k", "down": "iq2_ks"}}
    root, shapes, written = _tiny_stack(tmp_path, members)

    artifacts = IQKRoutedArtifacts(root, members, shapes, 2)
    try:
        assert artifacts.identity()["member_counts"] == {"iq2_k": 1, "iq2_ks": 2}
        for (layer, projection), path in written.items():
            raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            cell = artifacts.cells[(layer, projection)]
            per_expert = cell["bytes_per_expert"]
            for expert in (0, 1):
                got = artifacts.expert_blocks(layer, expert, projection)
                want = raw[expert * per_expert:(expert + 1) * per_expert].reshape(
                    cell["out_features"], cell["bytes_per_row"])
                assert np.array_equal(got, want)
        with pytest.raises(IQKPackageError, match="outside"):
            artifacts.expert_blocks(0, 2, "gate")
    finally:
        artifacts.close()


def test_routed_artifacts_reject_a_cell_written_at_another_member(tmp_path):
    members = {0: {"gate": "iq2_ks", "up": "iq2_ks", "down": "iq2_ks"}}
    root, shapes, _written = _tiny_stack(tmp_path, members)
    # Same cell, same name, bytes of the wider member.
    (root / "layer00_gate.iq2_ks").write_bytes(
        b"\x00" * (2 * 4 * iqk_geometry("iq2_k").bytes_per_row(256)))

    with pytest.raises(IQKPackageError, match="size"):
        IQKRoutedArtifacts(root, members, shapes, 2)


def test_routed_artifacts_require_the_conversion_digests(tmp_path):
    members = {0: {"gate": "iq2_ks", "up": "iq2_ks", "down": "iq2_ks"}}
    root, shapes, written = _tiny_stack(tmp_path, members)
    recorded = {
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in written.values()
        ]
    }
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(recorded))

    artifacts = IQKRoutedArtifacts(root, members, shapes, 2)
    try:
        report = artifacts.verify_digests(inventory)
        assert report["files_checked"] == 3

        recorded["files"][0]["sha256"] = "0" * 64
        inventory.write_text(json.dumps(recorded))
        with pytest.raises(IQKPackageError, match="sha256"):
            artifacts.verify_digests(inventory)
    finally:
        artifacts.close()


# --------------------------------------------------------------------------
# The dense side


def _dense_inventory():
    return {
        "tensors": [
            {"source_name": "embed.weight", "kind": "affine", "role": "embed_tokens",
             "layer_index": None, "shape": [129280, 4096], "dtype": "BF16",
             "gguf_keys": []},
            {"source_name": "head.weight", "kind": "affine", "role": "lm_head",
             "layer_index": None, "shape": [129280, 4096], "dtype": "BF16",
             "gguf_keys": ["output.weight"]},
            {"source_name": "layers.0.attn.wq_a.weight", "kind": "affine",
             "role": "attn.wq_a", "layer_index": 0, "shape": [1024, 4096],
             "dtype": "F8_E4M3", "gguf_keys": ["blk.0.attn_q_a.weight"]},
            {"source_name": "layers.0.attn.indexer.wq_b.weight", "kind": "affine",
             "role": "attn.indexer.wq_b", "layer_index": 0, "shape": [8192, 1024],
             "dtype": "F8_E4M3", "gguf_keys": []},
            {"source_name": "layers.0.attn.indexer.wq_b.scale", "kind": "codec_scale",
             "role": "attn.indexer.wq_b.scale", "layer_index": 0},
            {"source_name": "layers.0.ffn.gate.weight", "kind": "affine",
             "role": "moe.router_gate", "layer_index": 0, "shape": [256, 4096],
             "dtype": "BF16", "gguf_keys": []},
        ]
    }


def test_dense_side_follows_the_gguf_key_split():
    allocation = build_dense_allocations(_dense_inventory())

    by_name = {a["source_name"]: a for a in allocation}
    assert "layers.0.ffn.gate.weight" not in by_name  # router gate stays passthrough
    assert by_name["embed.weight"]["format"] == "affine"
    assert by_name["embed.weight"]["bits"] == 8
    assert by_name["head.weight"]["format"] == "kquant"
    assert by_name["head.weight"]["kquant_codec"] == "q8_0"
    assert by_name["head.weight"]["module_weight_key"] == "lm_head.weight"
    assert by_name["layers.0.attn.wq_a.weight"]["kquant_codec"] == "q8_0"
    assert by_name["layers.0.attn.indexer.wq_b.weight"]["format"] == "mxfp8"
    assert dense_format_counts(allocation) == {
        "affine": 1, "kquant:q8_0": 2, "mxfp8": 1}


def test_dense_side_rejects_an_unknown_codec():
    with pytest.raises(IQKPackageError, match="unknown dense codec"):
        build_dense_allocations(_dense_inventory(), codec="q9_0")


def test_dense_rows_stay_unsteered_through_the_writer_target():
    # The writer rebuilds each dense target from the allocation row
    # (`dense_target_from_allocation`), which defaults `requires_imatrix`
    # to true. The IQ_K dense side is deliberately unsteered, so the row
    # must carry the flag or an imatrix-steered codec such as q6_k fails
    # at encode with an empty imatrix vector set.
    from moespresso.package.deepseek_v4.recipe import dense_target_from_allocation
    from moespresso.package.kquant_recipe import validate_kquant_target_fit

    allocation = build_dense_allocations(_dense_inventory(), codec="q6_k")
    kquant_rows = [a for a in allocation if a["format"] == "kquant"]
    assert kquant_rows
    for row in kquant_rows:
        assert row["kquant_codec"] == "q6_k"
        target = dense_target_from_allocation(row)
        assert target.requires_imatrix is False
        # The write-time fit check must accept the unsteered encode.
        shape = next(
            t["shape"] for t in _dense_inventory()["tensors"]
            if t["source_name"] == row["source_name"])
        validate_kquant_target_fit(target, shape, {})
