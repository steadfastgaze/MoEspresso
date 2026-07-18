"""Resident mixed-bit SwitchGLU safety.

This is the non-streaming baseline: MoEspresso packages may assign different
TQ bit-widths to routed gate/up projections. JANG's class-level fused SwitchGLU
path has one bit-width parameter for both, so the resident runtime must detect
mixed gate/up layers from package metadata and replace only those layers with a
MoEspresso-owned forward.
"""

from __future__ import annotations

import json
import struct

import pytest

from moespresso.runtime.build import (
    MixedBitSwitchGLUError,
    _mixed_gate_up_layers_from_headers,
    _wrap_mixed_bit_switchglus,
    build_model,
)

pytest.importorskip("mlx.core")

from moespresso.runtime.owned_switchglu import OwnedSwitchGLU  # noqa: E402


class _Obj:
    pass


def _write_safetensors(path, tensors):
    """Write a tiny safetensors shard: name -> (dtype, shape, raw bytes)."""
    header, blob, off = {}, bytearray(), 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [off, off + len(raw)],
        }
        blob += raw
        off += len(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _stacked(n_exp=4, out=8, cols=2):
    return "U32", (n_exp, out, cols), bytes(n_exp * out * cols * 4)


def _norms(n_exp=4, out=8):
    return "F16", (n_exp, out), bytes(n_exp * out * 2)


def _tiny_expert_package(tmp_path, layer_bits):
    from conftest import write_bundle_package

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    # one builder call per layer so each layer can carry its own bits map
    for layer, bits_by_proj in sorted(layer_bits.items()):
        write_bundle_package(
            pkg, layers=(layer,),
            specs={proj: (8, 2, bits_by_proj.get(proj, 2))
                   for proj in ("gate_proj", "up_proj", "down_proj")},
            shard_name=f"model-{layer + 1:05d}-of-{len(layer_bits):05d}"
                       ".safetensors")
    return pkg


def _dummy_proj(bits=2, *, expose_bits=True):
    obj = _Obj()
    if expose_bits:
        obj.bits = bits
    return obj


def _switch(gate_bits=2, up_bits=2, *, expose_bits=True):
    sw = _Obj()
    sw.gate_proj = _dummy_proj(gate_bits, expose_bits=expose_bits)
    sw.up_proj = _dummy_proj(up_bits, expose_bits=expose_bits)
    sw.down_proj = _dummy_proj(1, expose_bits=expose_bits)
    sw.activation = _Obj()
    return sw


def _model(layer_specs):
    model = _Obj()
    model.language_model = _Obj()
    model.language_model.model = _Obj()
    model.language_model.model.layers = []
    for spec in layer_specs:
        layer = _Obj()
        layer.mlp = _Obj()
        layer.mlp.switch_mlp = _switch(**spec)
        model.language_model.model.layers.append(layer)
    return model


def _moe_manifest():
    return {
        "architecture": {"family": "qwen3_5_moe"},
        "required_ops": ["affine_dequant", "tq_dequant", "fp16_passthrough"],
        "tensors": [],
    }


def test_mixed_gate_up_layers_are_detected_from_headers(tmp_path):
    pkg = _tiny_expert_package(tmp_path, {
        0: {"gate_proj": 2, "up_proj": 2, "down_proj": 1},
        1: {"gate_proj": 2, "up_proj": 4, "down_proj": 1},
    })

    assert _mixed_gate_up_layers_from_headers(pkg) == {1}


def test_build_wrapper_replaces_only_mixed_gate_up_switchglus():
    model = _model([
        {"gate_bits": 2, "up_bits": 2},
        {"gate_bits": 2, "up_bits": 4},
    ])

    assert _wrap_mixed_bit_switchglus(model) == 1
    assert not isinstance(model.language_model.model.layers[0].mlp.switch_mlp,
                          OwnedSwitchGLU)
    assert isinstance(model.language_model.model.layers[1].mlp.switch_mlp,
                      OwnedSwitchGLU)


def test_build_wrapper_fails_closed_when_required_mixed_layer_lacks_bits():
    model = _model([{"gate_bits": 2, "up_bits": 4, "expose_bits": False}])

    with pytest.raises(MixedBitSwitchGLUError, match="do not expose .bits"):
        _wrap_mixed_bit_switchglus(model, required_mixed_layers={0})


def test_build_wrapper_fails_closed_when_loaded_bits_contradict_headers():
    model = _model([{"gate_bits": 2, "up_bits": 2}])

    with pytest.raises(MixedBitSwitchGLUError, match="loaded gate/up bits are both 2"):
        _wrap_mixed_bit_switchglus(model, required_mixed_layers={0})


def test_build_model_wraps_layers_declared_mixed_by_package_headers(tmp_path):
    pkg = _tiny_expert_package(tmp_path, {
        0: {"gate_proj": 2, "up_proj": 4, "down_proj": 1},
    })
    model = _model([{"gate_bits": 2, "up_bits": 4}])

    def fake_jangtq(package_dir):
        assert package_dir == pkg
        return model, "TOK"

    served, tokenizer = build_model(
        _moe_manifest(), pkg, load_jangtq_fn=fake_jangtq)

    assert served is model
    assert tokenizer == "TOK"
    assert isinstance(model.language_model.model.layers[0].mlp.switch_mlp,
                      OwnedSwitchGLU)
