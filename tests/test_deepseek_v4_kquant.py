from __future__ import annotations

import numpy as np
import pytest

from moespresso.package.bundle import component_array
from moespresso.package.kquant_backend import KQuantEncodedWeight
from moespresso.package.deepseek_v4.kquant import (
    DS4KQuantEncodeError,
    encode_ds4_kquant_layer_bundle,
    load_ds4_kquant_imatrix_vectors,
)
from moespresso.package.deepseek_v4.recipe import DS4KQuantExpertTarget
from moespresso.package.kquant_format import KQUANT_GEOMETRY


def _target(layer: int, projection: str, codec: str):
    gguf_projection = {"gate": "gate", "up": "up", "down": "down"}[projection]
    source_projection = {"gate": "w1", "up": "w3", "down": "w2"}[projection]
    return DS4KQuantExpertTarget(
        layer_index=layer,
        projection=projection,
        codec=codec,
        gguf_tensor=f"blk.{layer}.ffn_{gguf_projection}_exps.weight",
        imatrix_key=f"blk.{layer}.ffn_{gguf_projection}_exps.weight",
        source_weight_template=(
            f"layers.{layer}.ffn.experts.{{expert}}.{source_projection}.weight"
        ),
        source_scale_template=(
            f"layers.{layer}.ffn.experts.{{expert}}.{source_projection}.scale"
        ),
        module_path=f"model.layers.{layer}.mlp.switch_mlp.{projection}_proj",
        module_weight_key=(
            f"model.layers.{layer}.mlp.switch_mlp.{projection}_proj.weight"
        ),
    )


def _targets(layer=7):
    return [
        _target(layer, "gate", "iq2_xxs"),
        _target(layer, "up", "iq2_xxs"),
        _target(layer, "down", "q2_k"),
    ]


def _imatrix(layer=7, width=256):
    return {
        f"blk.{layer}.ffn_gate_exps.weight": np.ones(width, dtype=np.float32),
        f"blk.{layer}.ffn_up_exps.weight": np.ones(width, dtype=np.float32) * 2,
        f"blk.{layer}.ffn_down_exps.weight": np.ones(width, dtype=np.float32) * 3,
    }


class FakeExpertGroup:
    def __init__(self, experts=(0, 1, 2), *, width=256):
        self._experts = list(experts)
        self.width = width
        self.decode_calls = []

    def experts(self, layer):
        assert layer == 7
        return list(self._experts)

    def decode(self, *, layer, expert_index, projection, out_dtype):
        self.decode_calls.append((layer, expert_index, projection, out_dtype))
        rows = {"gate": 3, "up": 3, "down": 4}[projection]
        value = 100 * expert_index + {"gate": 1, "up": 2, "down": 3}[projection]
        return np.full((rows, self.width), value, dtype=out_dtype)


def test_encode_ds4_kquant_layer_bundle_decodes_encodes_and_bundles():
    group = FakeExpertGroup()
    calls = []

    def fake_encoder(weight, target, imatrix_vectors):
        calls.append((weight.copy(), target, imatrix_vectors[target.imatrix_key].copy()))
        geometry = KQUANT_GEOMETRY[target.codec]
        bytes_per_row = weight.shape[1] // geometry.weights_per_block * geometry.bytes_per_block
        return KQuantEncodedWeight(
            codec=target.codec,
            weight=np.full((weight.shape[0], bytes_per_row), len(calls), dtype=np.uint8),
            scales=np.zeros((1,), dtype=np.uint8),
        )

    bundle, geometry = encode_ds4_kquant_layer_bundle(
        group,
        _targets(),
        _imatrix(),
        layer=7,
        max_experts=2,
        encoder=fake_encoder,
    )

    assert bundle.shape == (2, geometry["row_bytes"])
    assert [call[:3] for call in group.decode_calls] == [
        (7, 0, "gate"),
        (7, 1, "gate"),
        (7, 0, "up"),
        (7, 1, "up"),
        (7, 0, "down"),
        (7, 1, "down"),
    ]
    assert all(call[3] is np.float32 for call in group.decode_calls)
    assert [call[1].projection for call in calls] == [
        "gate",
        "gate",
        "up",
        "up",
        "down",
        "down",
    ]
    np.testing.assert_array_equal(calls[0][2], np.ones(256, dtype=np.float32))
    np.testing.assert_array_equal(calls[-1][2], np.ones(256, dtype=np.float32) * 3)
    gate = geometry["projections"]["gate_proj"]
    down = geometry["projections"]["down_proj"]
    assert gate["kquant_codec"] == "iq2_xxs"
    assert down["kquant_codec"] == "q2_k"
    down_weight = component_array(bundle, down["weight"])
    assert down_weight.shape == (2, 4, 84)
    assert set(down_weight.reshape(-1).tolist()) == {5, 6}


def test_encode_ds4_kquant_layer_bundle_rejects_missing_target():
    with pytest.raises(DS4KQuantEncodeError, match="missing K-quant target"):
        encode_ds4_kquant_layer_bundle(
            FakeExpertGroup(),
            _targets()[:2],
            _imatrix(),
            layer=7,
            encoder=lambda *_args: pytest.fail("encoder should not run"),
        )


def test_encode_ds4_kquant_layer_bundle_rejects_empty_expert_set():
    with pytest.raises(DS4KQuantEncodeError, match="no DS4 experts"):
        encode_ds4_kquant_layer_bundle(
            FakeExpertGroup(experts=[]),
            _targets(),
            _imatrix(),
            layer=7,
            encoder=lambda *_args: pytest.fail("encoder should not run"),
        )


def test_encode_ds4_kquant_layer_bundle_validates_imatrix_before_encoding():
    calls = []

    def fake_encoder(*args):
        calls.append(args)
        raise AssertionError("encoder should not run after fit failure")

    with pytest.raises(DS4KQuantEncodeError, match="check tensor orientation"):
        encode_ds4_kquant_layer_bundle(
            FakeExpertGroup(),
            _targets(),
            _imatrix(width=128),
            layer=7,
            encoder=fake_encoder,
        )
    assert calls == []


def test_load_ds4_kquant_imatrix_vectors_uses_existing_reader(monkeypatch):
    from moespresso.package.deepseek_v4 import kquant

    expected = {"blk.7.ffn_down_exps.weight": np.ones(256, dtype=np.float32)}

    def fake_reader(path):
        assert path == "/tmp/imatrix.dat"
        return expected

    monkeypatch.setattr(kquant, "read_imatrix_vectors", fake_reader)

    assert load_ds4_kquant_imatrix_vectors("/tmp/imatrix.dat") is expected
