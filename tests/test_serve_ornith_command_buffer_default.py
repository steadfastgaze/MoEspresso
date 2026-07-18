from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import moespresso.runtime.serve as serve
from moespresso.runtime.serve import default_ornith_mlx_command_buffer_limit


def _ornith_manifest() -> dict:
    return {
        "artifact_id": (
            "pkg:aff416b9eeecfe9d18dd31798bb3e3ee91a0ff634297a5236f74e17a6c9c0ce0"
        ),
        "architecture": {
            "family": "qwen3_5_moe",
            "smoke_max_experts": None,
        },
        "provenance": {
            "package_plan": {
                "producer_reference": (
                    "deepreinforce-ai_Ornith-1.0-35B-Q4_K_M.gguf"
                )
            }
        },
    }


@pytest.fixture(autouse=True)
def _mlx_not_imported(monkeypatch):
    monkeypatch.setattr(serve, "_mlx_core_already_imported", lambda: False)
    monkeypatch.setattr(serve, "_installed_mlx_version", lambda: "0.31.2")


def test_ornith_m3_large_memory_defaults_command_buffer_limit(monkeypatch, capsys):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)

    got = default_ornith_mlx_command_buffer_limit(
        _ornith_manifest(),
        generation=3,
        total_memory_bytes=128 * (1 << 30),
    )

    assert got == "288"
    assert "MLX_MAX_MB_PER_BUFFER=288" in capsys.readouterr().out


def test_ornith_command_buffer_default_preserves_explicit_override(monkeypatch):
    monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "50")

    got = default_ornith_mlx_command_buffer_limit(
        _ornith_manifest(),
        generation=3,
        total_memory_bytes=128 * (1 << 30),
    )

    assert got is None
    assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "50"


def test_ornith_command_buffer_default_keeps_small_memory_policy(monkeypatch):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)

    got = default_ornith_mlx_command_buffer_limit(
        _ornith_manifest(),
        generation=3,
        total_memory_bytes=32 * (1 << 30),
    )

    assert got is None
    assert "MLX_MAX_MB_PER_BUFFER" not in os.environ


def test_ornith_command_buffer_default_is_hardware_and_package_specific(
    monkeypatch,
):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    deepseek = {
        "architecture": {"family": "deepseek_v4"},
        "provenance": {
            "package_plan": {"producer_reference": "DeepSeek-V4-Flash"}
        },
    }

    assert (
        default_ornith_mlx_command_buffer_limit(
            _ornith_manifest(),
            generation=4,
            total_memory_bytes=128 * (1 << 30),
        )
        is None
    )
    assert (
        default_ornith_mlx_command_buffer_limit(
            deepseek,
            generation=3,
            total_memory_bytes=128 * (1 << 30),
        )
        is None
    )
    assert "MLX_MAX_MB_PER_BUFFER" not in os.environ


def test_ornith_command_buffer_default_rejects_other_manifest_and_smoke(
    monkeypatch,
):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    rebuilt = _ornith_manifest()
    rebuilt["artifact_id"] = "pkg:rebuilt"
    smoke = _ornith_manifest()
    smoke["architecture"]["smoke_max_experts"] = 8

    for manifest in (rebuilt, smoke):
        assert (
            default_ornith_mlx_command_buffer_limit(
                manifest,
                generation=3,
                total_memory_bytes=128 * (1 << 30),
            )
            is None
        )
    assert "MLX_MAX_MB_PER_BUFFER" not in os.environ


def test_ornith_command_buffer_default_requires_measured_mlx(monkeypatch, capsys):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    monkeypatch.setattr(serve, "_installed_mlx_version", lambda: "0.32.0")

    got = default_ornith_mlx_command_buffer_limit(
        _ornith_manifest(),
        generation=3,
        total_memory_bytes=128 * (1 << 30),
    )

    assert got is None
    assert "MLX_MAX_MB_PER_BUFFER" not in os.environ
    assert "is not the measured version 0.31.2" in capsys.readouterr().out


def test_ornith_command_buffer_default_warns_after_mlx_import(monkeypatch, capsys):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    monkeypatch.setattr(serve, "_mlx_core_already_imported", lambda: True)

    got = default_ornith_mlx_command_buffer_limit(
        _ornith_manifest(),
        generation=3,
        total_memory_bytes=128 * (1 << 30),
    )

    assert got is None
    assert "MLX_MAX_MB_PER_BUFFER" not in os.environ
    assert "was imported before package load" in capsys.readouterr().out


def test_load_sets_ornith_limit_before_runtime_build(monkeypatch, tmp_path):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    monkeypatch.setattr(serve, "_apple_silicon_generation", lambda: 3)
    import psutil

    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=128 * (1 << 30)),
    )

    def build(manifest, package_dir):
        assert manifest is not None
        assert package_dir == tmp_path
        assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "288"
        return object(), object()

    serve.load_served_model(tmp_path, manifest=_ornith_manifest(), build_fn=build)
