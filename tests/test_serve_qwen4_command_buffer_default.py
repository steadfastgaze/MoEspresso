from __future__ import annotations

import os

import pytest

import moespresso.runtime.serve as serve


_DEFAULTS = {
    "MLX_MAX_OPS_PER_BUFFER": "50",
    "MLX_MAX_MB_PER_BUFFER": "200",
}


def _manifest(family: str = "qwen4_exp") -> dict:
    return {"architecture": {"family": family}}


@pytest.fixture(autouse=True)
def _clean_command_buffer_environment(monkeypatch):
    for name in _DEFAULTS:
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(serve, "_mlx_core_already_imported", lambda: False)
    monkeypatch.setattr(serve, "_installed_mlx_version", lambda: "0.31.2")


@pytest.mark.parametrize("family", ["qwen4_exp", "qwen4_exp_text"])
def test_qwen4_command_buffer_defaults_apply_to_supported_families(family, capsys):
    got = serve.default_qwen4_mlx_command_buffer_limits(_manifest(family))

    assert got == _DEFAULTS
    assert {name: os.environ[name] for name in _DEFAULTS} == _DEFAULTS
    output = capsys.readouterr().out
    assert "MLX_MAX_OPS_PER_BUFFER=50" in output
    assert "MLX_MAX_MB_PER_BUFFER=200" in output


@pytest.mark.parametrize(
    ("explicit_name", "explicit_value", "defaulted_name"),
    [
        ("MLX_MAX_OPS_PER_BUFFER", "77", "MLX_MAX_MB_PER_BUFFER"),
        ("MLX_MAX_MB_PER_BUFFER", "333", "MLX_MAX_OPS_PER_BUFFER"),
    ],
)
def test_qwen4_command_buffer_defaults_preserve_each_override(
    monkeypatch, capsys, explicit_name, explicit_value, defaulted_name,
):
    monkeypatch.setenv(explicit_name, explicit_value)

    got = serve.default_qwen4_mlx_command_buffer_limits(_manifest())

    assert got == {defaulted_name: _DEFAULTS[defaulted_name]}
    assert os.environ[explicit_name] == explicit_value
    assert os.environ[defaulted_name] == _DEFAULTS[defaulted_name]
    output = capsys.readouterr().out
    assert f"{explicit_name}={explicit_value}" in output
    assert f"{defaulted_name}={_DEFAULTS[defaulted_name]}" in output


def test_qwen4_command_buffer_defaults_leave_complete_override_unchanged(
    monkeypatch, capsys,
):
    monkeypatch.setenv("MLX_MAX_OPS_PER_BUFFER", "71")
    monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "444")

    got = serve.default_qwen4_mlx_command_buffer_limits(_manifest())

    assert got == {}
    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "71"
    assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "444"
    assert capsys.readouterr().out == ""


def test_qwen4_command_buffer_defaults_are_idempotent(capsys):
    assert serve.default_qwen4_mlx_command_buffer_limits(_manifest()) == _DEFAULTS
    capsys.readouterr()

    assert serve.default_qwen4_mlx_command_buffer_limits(_manifest()) == {}
    assert {name: os.environ[name] for name in _DEFAULTS} == _DEFAULTS
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("family", ["qwen3_5_moe", "deepseek_v4_flash", "unknown"])
def test_qwen4_command_buffer_defaults_ignore_other_families(family, capsys):
    assert serve.default_qwen4_mlx_command_buffer_limits(_manifest(family)) == {}
    assert all(name not in os.environ for name in _DEFAULTS)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("reason", ["version", "late_import"])
def test_qwen4_command_buffer_defaults_skip_unmeasured_or_late_process(
    monkeypatch, capsys, reason,
):
    if reason == "version":
        monkeypatch.setattr(serve, "_installed_mlx_version", lambda: "0.32.0")
    else:
        monkeypatch.setattr(serve, "_mlx_core_already_imported", lambda: True)

    assert serve.default_qwen4_mlx_command_buffer_limits(_manifest()) == {}
    assert all(name not in os.environ for name in _DEFAULTS)
    output = capsys.readouterr().out
    assert "not applied" in output
    if reason == "version":
        assert "0.31.2" in output
    else:
        assert "already imported" in output
        assert "before startup" in output


def test_load_sets_qwen4_command_buffer_defaults_before_build(monkeypatch, tmp_path):
    class BuildReached(RuntimeError):
        pass

    def build(manifest, package_dir, **kwargs):
        assert manifest == _manifest()
        assert package_dir == tmp_path
        assert {name: os.environ[name] for name in _DEFAULTS} == _DEFAULTS
        raise BuildReached

    with pytest.raises(BuildReached):
        serve.load_served_model(tmp_path, manifest=_manifest(), build_fn=build)
