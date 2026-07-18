from __future__ import annotations

import builtins

import pytest

from moespresso.runtime.kquant_install import (
    KQuantInstallError,
    install_manifest_kquant_modules,
    kquant_weight_codec_map_from_manifest,
)


class FakeModel:
    pass


def _manifest(*tensors):
    return {"tensors": list(tensors)}


def _kquant_tensor(key="model.layers.0.mlp.switch_mlp.down_proj.weight", codec="q2_k"):
    return {
        "source_name": "layers.0.ffn.experts",
        "format": "kquant",
        "format_params": {"kquant_codec": codec},
        "module_weight_key": key,
    }


def test_kquant_weight_codec_map_uses_explicit_manifest_keys():
    manifest = _manifest(
        _kquant_tensor("model.layers.0.mlp.switch_mlp.gate_proj.weight", "iq2_xxs"),
        _kquant_tensor("model.layers.0.mlp.switch_mlp.down_proj.weight", "q2_k"),
        {"source_name": "dense.weight", "format": "affine"},
    )

    assert kquant_weight_codec_map_from_manifest(manifest) == {
        "model.layers.0.mlp.switch_mlp.gate_proj.weight": "iq2_xxs",
        "model.layers.0.mlp.switch_mlp.down_proj.weight": "q2_k",
    }


def test_kquant_weight_codec_map_skips_expert_bundle_entries():
    tensor = _kquant_tensor()
    tensor["kind"] = "expert"

    assert kquant_weight_codec_map_from_manifest(_manifest(tensor)) == {}


def test_kquant_weight_codec_map_rejects_missing_module_key():
    tensor = _kquant_tensor()
    del tensor["module_weight_key"]

    with pytest.raises(KQuantInstallError, match="module_weight_key"):
        kquant_weight_codec_map_from_manifest(_manifest(tensor))


def test_kquant_weight_codec_map_rejects_unknown_codec():
    with pytest.raises(KQuantInstallError, match="unknown K-quant codec"):
        kquant_weight_codec_map_from_manifest(
            _manifest(_kquant_tensor(codec="q2_not_real"))
        )


def test_kquant_weight_codec_map_rejects_conflicting_duplicate_key():
    key = "model.layers.0.mlp.switch_mlp.down_proj.weight"

    with pytest.raises(KQuantInstallError, match="conflicting"):
        kquant_weight_codec_map_from_manifest(
            _manifest(
                _kquant_tensor(key, "q2_k"),
                _kquant_tensor(key, "iq2_xxs"),
            )
        )


def test_install_manifest_kquant_modules_calls_injected_installer():
    calls = []
    model = FakeModel()

    def fake_installer(model_arg, codec_map):
        calls.append((model_arg, codec_map))
        return 2

    installed = install_manifest_kquant_modules(
        model,
        _manifest(_kquant_tensor()),
        installer=fake_installer,
    )

    assert installed == 2
    assert calls == [(model, {"model.layers.0.mlp.switch_mlp.down_proj.weight": "q2_k"})]
    assert model._moespresso_kquant_modules_installed == 2
    assert model._moespresso_kquant_module_weight_keys == (
        "model.layers.0.mlp.switch_mlp.down_proj.weight",
    )


def test_install_manifest_without_kquant_is_noop_and_does_not_import(monkeypatch):
    def fail_import(name, *args, **kwargs):
        if name.startswith("mlx_kquant"):
            raise AssertionError("no K-quant tensors must not import mlx_kquant")
        return real_import(name, *args, **kwargs)

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fail_import)

    assert install_manifest_kquant_modules(FakeModel(), _manifest()) == 0


def test_install_manifest_reports_missing_backend(monkeypatch):
    def fail_import(name, *args, **kwargs):
        if name.startswith("mlx_kquant"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fail_import)

    with pytest.raises(KQuantInstallError, match="Reinstall MoEspresso"):
        install_manifest_kquant_modules(FakeModel(), _manifest(_kquant_tensor()))
