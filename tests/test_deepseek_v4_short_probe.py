from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from moespresso.correctness.deepseek_v4 import short_probe as probe


class _Switch:
    activation = object()


def _model():
    layers = [
        SimpleNamespace(mlp=SimpleNamespace(switch_mlp=_Switch())),
        SimpleNamespace(mlp=SimpleNamespace(switch_mlp=_Switch())),
        SimpleNamespace(mlp=SimpleNamespace(switch_mlp=_Switch())),
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def test_ffn_routed_default_arm_is_noop():
    model = _model()
    original = model.model.layers[1].mlp.switch_mlp

    probe._patch_ffn_routed_arm(model, "default", Path("/unused.gguf"))

    assert model.model.layers[1].mlp.switch_mlp is original


def test_ffn_routed_gguf_layer1_replaces_only_layer1(monkeypatch):
    model = _model()
    original_layer0 = model.model.layers[0].mlp.switch_mlp
    original_layer2 = model.model.layers[2].mlp.switch_mlp
    calls = []

    def fake_switch(gguf_path, layer, activation):
        calls.append((gguf_path, layer, activation))
        return SimpleNamespace(mode="diagnostic_gguf_routed")

    monkeypatch.setattr(probe, "_make_gguf_routed_switch", fake_switch)
    gguf = Path("/tmp/reference.gguf")

    probe._patch_ffn_routed_arm(model, "ds4_gguf_layer1", gguf)

    assert model.model.layers[0].mlp.switch_mlp is original_layer0
    assert model.model.layers[2].mlp.switch_mlp is original_layer2
    assert model.model.layers[1].mlp.switch_mlp.mode == "diagnostic_gguf_routed"
    assert calls == [(gguf, 1, _Switch.activation)]


def test_ffn_routed_unknown_arm_fails_closed():
    with pytest.raises(ValueError, match="unknown FFN routed arm"):
        probe._patch_ffn_routed_arm(_model(), "source_everything", Path("/x.gguf"))


def test_gguf_kquant_storage_shape_uses_expert_out_packed_order():
    tensor = SimpleNamespace(
        name="blk.1.ffn_gate_exps.weight",
        n_dimensions=3,
        dimensions=[4096, 2048, 256],
    )

    assert probe._gguf_kquant_storage_shape(tensor, "iq2_xxs") == (
        256,
        2048,
        1056,
    )
