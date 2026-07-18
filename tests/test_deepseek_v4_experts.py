from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from moespresso.inventory.build import build_inventory
from moespresso.probe.deepseek_v4.experts import (
    DecodedExpertGroup,
    DeepseekV4ExpertAdapterError,
)


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, (dtype, arr) in tensors.items():
        a = np.ascontiguousarray(arr)
        data = a.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(a.shape),
            "data_offsets": [off, off + len(data)],
        }
        blob += data
        off += len(data)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _expert_tensors(include_scale=True):
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": (
            "I8",
            np.array([[0x21, 0xA7]], dtype=np.uint8).view(np.int8),
        ),
        "layers.0.ffn.experts.0.w3.weight": (
            "I8",
            np.array([[0x43, 0x65]], dtype=np.uint8).view(np.int8),
        ),
        "layers.0.ffn.experts.0.w2.weight": (
            "I8",
            np.array([[0x21], [0xA7]], dtype=np.uint8).view(np.int8),
        ),
        "layers.0.ffn.experts.1.w1.weight": (
            "I8",
            np.array([[0x10, 0x32]], dtype=np.uint8).view(np.int8),
        ),
        "layers.0.ffn.experts.1.w3.weight": (
            "I8",
            np.array([[0x10, 0x32]], dtype=np.uint8).view(np.int8),
        ),
        "layers.0.ffn.experts.1.w2.weight": (
            "I8",
            np.array([[0x10], [0x32]], dtype=np.uint8).view(np.int8),
        ),
    }
    if include_scale:
        for expert in (0, 1):
            tensors[f"layers.0.ffn.experts.{expert}.w1.scale"] = (
                "F8_E8M0",
                np.array([[127, 128]], dtype=np.uint8),
            )
            tensors[f"layers.0.ffn.experts.{expert}.w3.scale"] = (
                "F8_E8M0",
                np.array([[127, 127]], dtype=np.uint8),
            )
            tensors[f"layers.0.ffn.experts.{expert}.w2.scale"] = (
                "F8_E8M0",
                np.array([[127], [128]], dtype=np.uint8),
            )
    return tensors


def test_decoded_expert_group_exposes_logical_projections(tmp_path):
    _write_safetensors(tmp_path / "model-00001.safetensors", _expert_tensors())
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")

    group = DecodedExpertGroup.from_inventory(inv, tmp_path, fp4_block=2)

    assert group.layers() == [0]
    assert group.experts(0) == [0, 1]
    assert group.projections(0) == ["gate", "up", "down"]
    assert group.logical_shape(layer=0, expert_index=0, projection="gate") == (1, 4)
    assert group.logical_shape(layer=0, expert_index=0, projection="down") == (2, 2)


def test_decoded_expert_group_decodes_separate_w1_w3_w2_sources(tmp_path):
    _write_safetensors(tmp_path / "model-00001.safetensors", _expert_tensors())
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, tmp_path, fp4_block=2)

    gate = group.decode(layer=0, expert_index=0, projection="gate", out_dtype=np.float32)
    up = group.decode(layer=0, expert_index=0, projection="up", out_dtype=np.float32)
    down = group.decode(layer=0, expert_index=0, projection="down", out_dtype=np.float32)

    np.testing.assert_allclose(gate, np.array([[0.5, 1.0, 12.0, -2.0]], dtype=np.float32))
    np.testing.assert_allclose(up, np.array([[1.5, 2.0, 3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(down, np.array([[0.5, 1.0], [12.0, -2.0]], dtype=np.float32))


def test_decoded_expert_group_iterates_requested_projection(tmp_path):
    _write_safetensors(tmp_path / "model-00001.safetensors", _expert_tensors())
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, tmp_path, fp4_block=2)

    rows = list(group.iter_projection(layer=0, projection="gate", expert_indices=[1],
                                      out_dtype=np.float32))

    assert rows[0][0] == 1
    np.testing.assert_allclose(rows[0][1], np.array([[0.0, 0.5, 2.0, 3.0]], np.float32))


def test_decoded_expert_group_fails_closed_on_missing_scale(tmp_path):
    _write_safetensors(
        tmp_path / "model-00001.safetensors",
        _expert_tensors(include_scale=False),
    )
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")

    with pytest.raises(DeepseekV4ExpertAdapterError, match="missing scale"):
        DecodedExpertGroup.from_inventory(inv, tmp_path, fp4_block=2)


def test_decoded_expert_group_fails_closed_on_missing_projection_weight(tmp_path):
    tensors = _expert_tensors()
    tensors.pop("layers.0.ffn.experts.1.w2.weight")
    _write_safetensors(tmp_path / "model-00001.safetensors", tensors)
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")

    with pytest.raises(DeepseekV4ExpertAdapterError, match="missing weight"):
        DecodedExpertGroup.from_inventory(inv, tmp_path, fp4_block=2)
