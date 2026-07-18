"""Streaming weight loaders: verified against a hand-built safetensors file.

Synthesizes a tiny .safetensors (a 2D F32 tensor + a stacked-3D F32 tensor),
then confirms the loaders seek to the right bytes, subsample correctly, slice the
right experts, and split a fused gate_up sample into the right halves. No mlx.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from moespresso.inventory.safetensors_header import read_headers_with_offsets
from moespresso.probe.weight_io import (
    load_2d_sample,
    load_expert_sample,
    scan_offsets,
    split_fused_gate_up,
)


def _write_safetensors(path, tensors):
    """tensors: {name: np.ndarray(float32)} -> a valid F32 safetensors file."""
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape),
                        "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def test_load_2d_full_when_small(tmp_path):
    w = np.arange(12, dtype=np.float32).reshape(4, 3)
    sf = tmp_path / "model-00001.safetensors"
    _write_safetensors(sf, {"w": w})
    h = {x.name: x for x in read_headers_with_offsets(sf)}["w"]
    out = load_2d_sample(tmp_path, h, sample_rows=100, seed=0)
    np.testing.assert_array_equal(out, w)


def test_load_2d_subsamples_rows(tmp_path):
    w = np.arange(40, dtype=np.float32).reshape(8, 5)
    sf = tmp_path / "model-00001.safetensors"
    _write_safetensors(sf, {"w": w})
    h = {x.name: x for x in read_headers_with_offsets(sf)}["w"]
    out = load_2d_sample(tmp_path, h, sample_rows=3, seed=7)
    assert out.shape == (3, 5)
    # every sampled row must be an actual row of w
    for row in out:
        assert any(np.array_equal(row, wr) for wr in w)


def test_expert_sample_reads_correct_slices(tmp_path):
    # 5 experts, each a distinct [2,3] block so we can identify which were read.
    experts = np.stack([np.full((2, 3), float(e)) for e in range(5)]).astype(np.float32)
    sf = tmp_path / "model-00001.safetensors"
    _write_safetensors(sf, {"e": experts})
    h = {x.name: x for x in read_headers_with_offsets(sf)}["e"]
    out = load_expert_sample(tmp_path, h, n_experts=2, seed=1)
    assert out.shape == (4, 3)  # 2 experts * 2 rows
    vals = sorted({float(v) for v in np.unique(out)})
    assert len(vals) == 2  # exactly two distinct experts, each block constant
    for block_start in (0, 2):
        block = out[block_start:block_start + 2]
        assert np.all(block == block[0, 0])


def test_expert_sample_caps_at_total(tmp_path):
    experts = np.stack([np.full((2, 3), float(e)) for e in range(3)]).astype(np.float32)
    sf = tmp_path / "model-00001.safetensors"
    _write_safetensors(sf, {"e": experts})
    h = {x.name: x for x in read_headers_with_offsets(sf)}["e"]
    out = load_expert_sample(tmp_path, h, n_experts=10, seed=1)
    assert out.shape == (6, 3)  # all 3 experts


def test_split_fused_gate_up():
    # 2 experts, fused height 4 (gate=top 2 rows, up=bottom 2), cols=3.
    e0 = np.array([[1, 1, 1], [1, 1, 1], [9, 9, 9], [9, 9, 9]], np.float32)
    e1 = np.array([[2, 2, 2], [2, 2, 2], [8, 8, 8], [8, 8, 8]], np.float32)
    sample = np.concatenate([e0, e1], axis=0)
    gate, up = split_fused_gate_up(sample, n_sampled=2)
    assert gate.shape == (4, 3) and up.shape == (4, 3)
    assert np.all(gate[:2] == 1) and np.all(gate[2:] == 2)
    assert np.all(up[:2] == 9) and np.all(up[2:] == 8)


def test_scan_offsets_finds_all(tmp_path):
    sf = tmp_path / "model-00001.safetensors"
    _write_safetensors(sf, {"a": np.zeros((2, 2), np.float32),
                            "b": np.ones((3, 3), np.float32)})
    cat = scan_offsets(tmp_path)
    assert set(cat) == {"a", "b"}
    assert cat["b"].shape == (3, 3)
