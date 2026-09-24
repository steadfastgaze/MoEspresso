"""Manifest-selected loaders, packed-weight installation and warning filters."""

from __future__ import annotations

import json
import logging
import struct
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from moespresso.package.bundle import (
    KQUANT_CODEC,
    assemble_layer_bundle,
    encode_bundle_metadata,
)
from moespresso.runtime.expert_index import build_expert_index
from moespresso.runtime.build import (
    _DropDeepSeekV4RopeWarning,
    _DropQwen4ConfigTypeWarning,
    _silence_known_transformers_warnings,
    UnsupportedRuntimeAdapter,
    _decoder_layers,
    _install_routed_experts_from_bundles,
    _load_qwen_kquant_model,
    _runtime_adapter_kind,
    _wrap_mixed_bit_switchglus,
    build_model,
)


def test_deepseek_v4_rope_warning_filter_is_exact():
    warning_filter = _DropDeepSeekV4RopeWarning()
    prefix = "Unrecognized keys in `rope_parameters` for 'rope_type'='default': "
    hidden = logging.makeLogRecord({"msg": prefix + "{'attention_factor'}"})
    visible = logging.makeLogRecord({"msg": prefix + "{'another_field'}"})

    assert not warning_filter.filter(hidden)
    assert warning_filter.filter(visible)


def test_qwen4_config_type_warning_filter_is_exact():
    warning_filter = _DropQwen4ConfigTypeWarning()
    hidden = logging.makeLogRecord({
        "msg": "You are using a model of type `qwen4_exp` to instantiate a model "
        "of type ``. This may be expected if you are loading a checkpoint that shares "
        "a subset of the architecture (e.g., loading a `sam2_video` checkpoint into "
        "`Sam2Model`), but is otherwise not supported and can yield errors. Please "
        "verify that the checkpoint is compatible with the model you are instantiating.",
    })
    visible = logging.makeLogRecord({
        "msg": "You are using a model of type `qwen4_exp_text` to instantiate a model "
        "of type ``. This is not supported for all configurations of models and can yield errors.",
    })

    assert not warning_filter.filter(hidden)
    assert warning_filter.filter(visible)


def test_qwen4_config_type_warning_filter_is_registered_on_its_source_logger():
    logger = logging.getLogger("transformers.configuration_utils")
    original = list(logger.filters)
    try:
        _silence_known_transformers_warnings()
        assert sum(isinstance(item, _DropQwen4ConfigTypeWarning) for item in logger.filters) == 1
    finally:
        logger.filters[:] = original


def test_decoder_layers_finds_qwen_style_path():
    layers = [object()]
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))

    assert _decoder_layers(model) is layers


def test_decoder_layers_finds_deepseek_style_path():
    layers = [object()]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))

    assert _decoder_layers(model) is layers


def test_decoder_layers_fails_closed_when_absent():
    assert _decoder_layers(SimpleNamespace()) is None


def test_wrap_mixed_bit_switchglu_uses_deepseek_style_path():
    gate = SimpleNamespace(bits=4)
    up = SimpleNamespace(bits=6)
    down = SimpleNamespace(bits=4)
    switch = SimpleNamespace(
        gate_proj=gate,
        up_proj=up,
        down_proj=down,
        activation=object(),
    )
    layer = SimpleNamespace(mlp=SimpleNamespace(switch_mlp=switch))
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    assert _wrap_mixed_bit_switchglus(model, required_mixed_layers={0}) == 1
    wrapped = layer.mlp.switch_mlp
    assert wrapped.gate_proj is gate
    assert wrapped.up_proj is up
    assert wrapped.down_proj is down


def _manifest(family, required_ops):
    return {
        "architecture": {"family": family},
        "required_ops": required_ops,
        "tensors": [],
    }


def test_runtime_adapter_selects_regular_jang_for_dense_affine():
    man = _manifest("qwen3_5_dense", ["affine_dequant", "fp16_passthrough"])
    assert _runtime_adapter_kind(man) == "regular_jang_v2"


def test_runtime_adapter_accepts_dense_f32_passthrough():
    man = _manifest("qwen3_5_dense", ["affine_dequant", "f32_passthrough"])
    assert _runtime_adapter_kind(man) == "regular_jang_v2"


def test_runtime_adapter_selects_qwen_kquant_for_gguf_recipe_packages():
    man = _manifest("qwen3_5_moe", ["kquant_dequant", "f32_passthrough"])
    assert _runtime_adapter_kind(man) == "qwen_kquant_moe"


def test_runtime_adapter_selects_dsv4_for_deepseek_manifest():
    man = _manifest(
        "deepseek_v4_flash",
        ["affine_dequant", "iqk_dequant", "fp16_passthrough", "raw_dtype_passthrough"],
    )
    assert _runtime_adapter_kind(man) == "mjtq_dsv4"


def test_runtime_adapter_selects_qwen4_composite_runtime():
    man = _manifest(
        "qwen4_exp",
        ["iqk_dequant", "kquant_dequant", "raw_dtype_passthrough"],
    )
    assert _runtime_adapter_kind(man) == "qwen4_iqk_moe"


def test_runtime_adapter_rejects_unknown_qwen4_ops():
    man = _manifest(
        "qwen4_exp",
        ["iqk_dequant", "kquant_dequant", "future_dequant"],
    )
    with pytest.raises(UnsupportedRuntimeAdapter, match="Qwen4 runtime ops"):
        _runtime_adapter_kind(man)


def test_runtime_adapter_never_sweeps_deepseek_unknown_ops_into_qwen_moe():
    man = _manifest(
        "deepseek_v4_flash",
        [
            "affine_dequant",
            "iqk_dequant",
            "fp16_passthrough",
            "raw_dtype_passthrough",
            "deepseek_v4_composite_cache",
        ],
    )
    with pytest.raises(UnsupportedRuntimeAdapter, match="DeepSeek V4 runtime ops"):
        _runtime_adapter_kind(man)


def test_runtime_adapter_accepts_deepseek_mxfp4():
    man = _manifest(
        "deepseek_v4_flash",
        [
            "affine_dequant",
            "iqk_dequant",
            "mxfp4_dequant",
            "fp16_passthrough",
            "raw_dtype_passthrough",
        ],
    )
    assert _runtime_adapter_kind(man) == "mjtq_dsv4"


def test_runtime_adapter_accepts_deepseek_dense_mxfp8():
    man = _manifest(
        "deepseek_v4_flash",
        [
            "mxfp8_dequant",
            "fp16_passthrough",
            "raw_dtype_passthrough",
        ],
    )
    assert _runtime_adapter_kind(man) == "mjtq_dsv4"


def test_runtime_adapter_accepts_deepseek_kquant():
    man = _manifest(
        "deepseek_v4_flash",
        [
            "affine_dequant",
            "kquant_dequant",
            "fp16_passthrough",
            "raw_dtype_passthrough",
        ],
    )
    assert _runtime_adapter_kind(man) == "mjtq_dsv4"


def test_runtime_adapter_accepts_deepseek_f32_passthrough():
    man = _manifest(
        "deepseek_v4_flash",
        [
            "affine_dequant",
            "f32_passthrough",
            "raw_dtype_passthrough",
        ],
    )
    assert _runtime_adapter_kind(man) == "mjtq_dsv4"


def test_runtime_adapter_fails_closed_for_unknown_dense_ops():
    man = _manifest("qwen3_5_dense", ["affine_dequant", "iqk_dequant"])
    with pytest.raises(UnsupportedRuntimeAdapter, match="unsupported runtime adapter"):
        _runtime_adapter_kind(man)


def test_runtime_adapter_fails_closed_for_unknown_family_affine_only():
    man = _manifest("some_future_dense", ["affine_dequant", "fp16_passthrough"])
    with pytest.raises(UnsupportedRuntimeAdapter, match="unsupported runtime adapter"):
        _runtime_adapter_kind(man)


def test_runtime_adapter_fails_closed_for_unknown_tq_companion_op():
    man = _manifest(
        "qwen3_5_moe",
        ["affine_dequant", "iqk_dequant", "future_dequant"],
    )
    with pytest.raises(UnsupportedRuntimeAdapter, match="unsupported runtime adapter"):
        _runtime_adapter_kind(man)


def test_build_model_uses_regular_jang_loader_for_dense_affine(tmp_path):
    calls = []

    def fake_regular(package_dir):
        calls.append(("regular", package_dir))
        return ("DENSE", "TOK")

    man = _manifest("qwen3_5_dense", ["affine_dequant", "fp16_passthrough"])
    assert build_model(man, tmp_path, load_jang_fn=fake_regular) == ("DENSE", "TOK")
    assert calls == [("regular", tmp_path)]


def test_build_model_uses_qwen_kquant_loader_for_kquant_moe(tmp_path):
    calls = []
    # A settable object lets the qwen3_5_moe build path install the
    # served prefill chunk on the model (the promoted 4096 default writes the
    # step attribute), so the fake model must accept attribute writes.
    fake_model = SimpleNamespace()

    def fake_qwen_kquant(manifest, package_dir):
        calls.append((manifest["architecture"]["family"], package_dir))
        return fake_model, "TOK"

    man = _manifest("qwen3_5_moe", ["kquant_dequant", "f32_passthrough"])
    built_model, built_tok = build_model(
        man,
        tmp_path,
        load_qwen_kquant_fn=fake_qwen_kquant,
    )
    assert built_model is fake_model
    assert built_tok == "TOK"
    # The promoted default set the served prefill chunk with the long-prompt
    # gate; short prompts stay on the stock chunk through the min gate.
    assert fake_model._moespresso_prefill_step_size == 4096
    assert fake_model._moespresso_prefill_step_size_min_prompt_tokens == 4097
    assert calls == [("qwen3_5_moe", tmp_path)]


def test_qwen_kquant_loader_swaps_modules_before_loading_weights(tmp_path):
    events = []
    model = SimpleNamespace()
    manifest = _manifest("qwen3_5_moe", ["kquant_dequant", "f32_passthrough"])

    def fake_load_config(package_dir):
        events.append(("load_config", package_dir))
        return {
            "model_type": "qwen3_5_moe",
            "quantization": {"bits": 4, "group_size": 128},
        }

    def fake_load_model(package_dir, *, lazy, strict, model_config):
        events.append((
            "load_model",
            package_dir,
            lazy,
            strict,
            model_config.get("quantization"),
            model_config.get("quantization_config"),
        ))
        return model, model_config

    def fake_install(model_arg, manifest_arg):
        events.append(("install_kquant", model_arg is model, manifest_arg is manifest))

    def fake_load_non_routed(model_arg, package_dir):
        events.append(("load_non_routed", model_arg is model, package_dir))

    def fake_load_tokenizer(package_dir, *, eos_token_ids):
        events.append(("load_tokenizer", package_dir, eos_token_ids))
        return "TOK"

    assert _load_qwen_kquant_model(
        manifest,
        tmp_path,
        load_config_fn=fake_load_config,
        load_model_fn=fake_load_model,
        load_tokenizer_fn=fake_load_tokenizer,
        install_kquant_modules_fn=fake_install,
        load_non_routed_fn=fake_load_non_routed,
    ) == (model, "TOK")

    assert events == [
        ("load_config", tmp_path),
        ("load_model", tmp_path, True, False, None, None),
        ("install_kquant", True, True),
        ("load_non_routed", True, tmp_path),
        ("load_tokenizer", tmp_path, None),
    ]


def test_qwen_kquant_loader_retains_all_declared_stop_ids(tmp_path):
    model = SimpleNamespace()
    seen = {}

    def fake_load_tokenizer(package_dir, *, eos_token_ids):
        seen["package_dir"] = package_dir
        seen["eos_token_ids"] = eos_token_ids
        return "TOK"

    built_model, tokenizer = _load_qwen_kquant_model(
        _manifest("qwen3_5_moe", ["kquant_dequant", "f32_passthrough"]),
        tmp_path,
        load_config_fn=lambda _path: {
            "model_type": "qwen3_5_moe",
            "text_config": {"eos_token_id": 17},
            "eos_token_id": [23],
        },
        load_model_fn=lambda _path, **kwargs: (model, kwargs["model_config"]),
        load_tokenizer_fn=fake_load_tokenizer,
        install_kquant_modules_fn=lambda _model, _manifest: None,
        load_non_routed_fn=lambda _model, _path: None,
    )

    assert built_model is model
    assert tokenizer == "TOK"
    assert seen["package_dir"] == tmp_path
    assert seen["eos_token_ids"] == {17, 23}
    assert 29 not in seen["eos_token_ids"]


def test_build_model_uses_dsv4_loader_for_deepseek_manifest(tmp_path):
    calls = []

    def fake_dsv4(manifest, package_dir, **kwargs):
        calls.append((manifest["architecture"]["family"], package_dir, kwargs))
        return "DS4", "TOK"

    man = _manifest(
        "deepseek_v4_flash",
        ["affine_dequant", "iqk_dequant", "fp16_passthrough", "raw_dtype_passthrough"],
    )
    assert build_model(man, tmp_path, load_dsv4_fn=fake_dsv4) == ("DS4", "TOK")
    assert calls == [(
        "deepseek_v4_flash",
        tmp_path,
        {},
    )]


def test_build_model_uses_qwen4_loader_with_served_context(tmp_path):
    calls = []

    def fake_qwen4(manifest, package_dir, **kwargs):
        calls.append((manifest["architecture"]["family"], package_dir, kwargs))
        return "QWEN4", "TOK"

    man = _manifest(
        "qwen4_exp",
        ["iqk_dequant", "kquant_dequant", "raw_dtype_passthrough"],
    )
    assert build_model(
        man,
        tmp_path,
        load_qwen4_fn=fake_qwen4,
        context_limit=65536,
        context_limit_explicit=True,
    ) == ("QWEN4", "TOK")
    assert calls == [(
        "qwen4_exp",
        tmp_path,
        {"max_context_tokens": 65536, "cache_routing": "auto"},
    )]
    build_model(
        man, tmp_path, load_qwen4_fn=fake_qwen4, cache_routing="prefer-resident",
        cache_routing_factor=3, cache_routing_protected_routes=0,
    )
    assert calls[-1][2] == {
        "cache_routing": "prefer-resident", "cache_routing_factor": 3,
        "cache_routing_protected_routes": 0,
    }


@pytest.mark.parametrize("options", [
    {"cache_routing_factor": 2},
    {"cache_routing": "prefer-resident", "cache_routing_factor": 4},
    {"cache_routing": "prefer-resident", "cache_routing_protected_routes": 0},
])
def test_other_adapters_reject_routing_controls_before_loading(tmp_path, options):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsupported routing options reached the loader")

    manifest = _manifest(
        "deepseek_v4_flash", ["affine_dequant", "iqk_dequant", "fp16_passthrough", "raw_dtype_passthrough"],
    )
    with pytest.raises((ValueError, UnsupportedRuntimeAdapter), match="routing"):
        build_model(manifest, tmp_path, load_dsv4_fn=forbidden, **options)


def test_build_model_uses_default_dsv4_loader_for_deepseek(tmp_path, monkeypatch):
    import moespresso.runtime.deepseek_v4.model as dsv4_model

    calls = []

    def fake_dsv4(manifest, package_dir, **kwargs):
        calls.append((manifest["architecture"]["family"], package_dir, kwargs))
        return "DS4", "TOK"

    monkeypatch.setattr(dsv4_model, "load_deepseek_v4_package_model", fake_dsv4)

    man = _manifest(
        "deepseek_v4_flash",
        ["affine_dequant", "iqk_dequant", "fp16_passthrough", "raw_dtype_passthrough"],
    )
    assert build_model(
        man,
        tmp_path,
        context_limit=16384,
        context_limit_explicit=True,
    ) == ("DS4", "TOK")
    assert calls == [(
        "deepseek_v4_flash",
        tmp_path,
        {"context_limit": 16384, "context_limit_explicit": True},
    )]


def _write_safetensors(path, tensors, metadata=None):
    header, blob, offset = {}, bytearray(), 0
    if metadata:
        header["__metadata__"] = dict(metadata)
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blob += raw
        offset += len(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def test_install_routed_experts_from_bundles_installs_resident_kquant_modules(
    tmp_path,
    monkeypatch,
):
    import mlx.core as mx

    rng = np.random.default_rng(123)
    components = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        components[(projection, "weight")] = rng.integers(
            0, 256, (2, 3, 84), dtype=np.uint8)
        components[(projection, "scales")] = np.zeros((2, 1), dtype=np.uint8)
    bundle, geometry = assemble_layer_bundle(
        components,
        {projection: 2 for projection in ("gate_proj", "up_proj", "down_proj")},
        codecs={
            projection: KQUANT_CODEC
            for projection in ("gate_proj", "up_proj", "down_proj")
        },
        kquant_codecs={
            projection: "q2_k"
            for projection in ("gate_proj", "up_proj", "down_proj")
        },
    )
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors",
        {
            "language_model.model.layers.0.mlp.switch_mlp.experts.tq_bundle": (
                "U8",
                bundle.shape,
                bundle.tobytes(),
            )
        },
        metadata={"expert_bundles": encode_bundle_metadata({0: geometry})},
    )

    class FakeKQuantSwitchLinear:
        def __init__(self, num_experts, output_dims, input_dims, bias, codec):
            self.num_experts = num_experts
            self.output_dims = output_dims
            self.input_dims = input_dims
            self.bias = bias
            self.kquant_type = codec

    nn_mod = types.ModuleType("mlx_kquant.nn")
    nn_mod.KQuantSwitchLinear = FakeKQuantSwitchLinear
    monkeypatch.setitem(sys.modules, "mlx_kquant", types.ModuleType("mlx_kquant"))
    monkeypatch.setitem(sys.modules, "mlx_kquant.nn", nn_mod)

    switch = SimpleNamespace()
    layer = SimpleNamespace(mlp=SimpleNamespace(switch_mlp=switch))
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    index = build_expert_index(pkg)

    assert _install_routed_experts_from_bundles(model, pkg, index) == 1

    gate = switch.gate_proj
    assert isinstance(gate, FakeKQuantSwitchLinear)
    assert gate.num_experts == 2
    assert gate.output_dims == 3
    assert gate.input_dims == 256
    assert gate.bias is False
    assert gate.kquant_type == "q2_k"
    mx.eval(gate.weight, gate.scales)
    np.testing.assert_array_equal(
        np.asarray(gate.weight),
        components[("gate_proj", "weight")],
    )
    assert tuple(gate.scales.shape) == (1,)
