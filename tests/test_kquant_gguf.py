from __future__ import annotations

import struct

import numpy as np
import pytest

from moespresso.package.kquant_gguf import (
    GGUFKQuantError,
    load_gguf_kquant_expert_weight,
)
from moespresso.package.deepseek_v4.recipe import DS4KQuantExpertTarget
from moespresso.package.kquant_format import KQUANT_GEOMETRY
from moespresso.probe.gguf_parse import GGUF_MAGIC


def _gguf_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _target(*, codec: str = "iq2_xxs") -> DS4KQuantExpertTarget:
    return DS4KQuantExpertTarget(
        layer_index=0,
        projection="gate",
        codec=codec,
        gguf_tensor="blk.0.ffn_gate_exps.weight",
        imatrix_key="blk.0.ffn_gate_exps.weight",
        source_weight_template="layers.0.ffn.experts.{expert}.w1.weight",
        source_scale_template="layers.0.ffn.experts.{expert}.w1.scale",
        module_path="model.layers.0.mlp.switch_mlp.gate_proj",
        module_weight_key="model.layers.0.mlp.switch_mlp.gate_proj.weight",
    )


def _write_payload_gguf(path, *, type_id=16, dims=(512, 2, 3), payload=None):
    header = struct.pack("<IIQQ", GGUF_MAGIC, 3, 1, 0)
    info = bytearray()
    info += _gguf_string("blk.0.ffn_gate_exps.weight")
    info += struct.pack("<I", len(dims))
    for dim in dims:
        info += struct.pack("<Q", dim)
    info += struct.pack("<I", type_id)
    info += struct.pack("<Q", 0)
    metadata = header + bytes(info)
    pad = b"\0" * ((((len(metadata) + 31) // 32) * 32) - len(metadata))
    if payload is None:
        codec = "iq2_xxs"
        geometry = KQUANT_GEOMETRY[codec]
        in_features, out_features, experts = dims
        packed_cols = in_features // geometry.weights_per_block * geometry.bytes_per_block
        payload = np.arange(experts * out_features * packed_cols, dtype=np.uint8)
    path.write_bytes(metadata + pad + np.asarray(payload, dtype=np.uint8).tobytes())


def test_load_gguf_kquant_expert_weight_reads_exact_stacked_expert_row(tmp_path):
    path = tmp_path / "recipe.gguf"
    _write_payload_gguf(path)

    encoded = load_gguf_kquant_expert_weight(path, _target(), expert_index=1)

    assert encoded.codec == "iq2_xxs"
    assert encoded.scales.tolist() == [0]
    assert encoded.weight.shape == (2, 132)
    expected = np.arange(3 * 2 * 132, dtype=np.uint8).reshape(3, 2, 132)[1]
    np.testing.assert_array_equal(encoded.weight, expected)


def test_load_gguf_kquant_expert_weight_fails_closed_on_codec_mismatch(tmp_path):
    path = tmp_path / "recipe.gguf"
    _write_payload_gguf(path)

    with pytest.raises(GGUFKQuantError, match="expected q2_k"):
        load_gguf_kquant_expert_weight(path, _target(codec="q2_k"), expert_index=0)


def test_load_gguf_kquant_expert_weight_fails_closed_on_expert_range(tmp_path):
    path = tmp_path / "recipe.gguf"
    _write_payload_gguf(path)

    with pytest.raises(GGUFKQuantError, match="expert 3 out of range"):
        load_gguf_kquant_expert_weight(path, _target(), expert_index=3)
