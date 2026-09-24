from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import moespresso.runtime.serve as serve
from moespresso.runtime.serve import default_ornith_mlx_command_buffer_limit

# The predicate reads the served shape, never the artifact id, so these are
# labels rather than real content hashes.
_KQUANT_ARTIFACT_ID = "pkg:kquant"


def _ornith_manifest(
    *,
    artifact_id: str = _KQUANT_ARTIFACT_ID,
    required_ops: tuple[str, ...] = ("f32_passthrough", "kquant_dequant"),
) -> dict:
    return {
        "artifact_id": artifact_id,
        "required_ops": list(required_ops),
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


def test_ornith_m3_large_memory_defaults_command_buffer_limit(
    monkeypatch, capsys
):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)

    got = default_ornith_mlx_command_buffer_limit(
        _ornith_manifest(),
        generation=3,
        total_memory_bytes=128 * (1 << 30),
    )

    assert got == "288"
    assert "MLX_MAX_MB_PER_BUFFER=288" in capsys.readouterr().out


def test_ornith_command_buffer_default_preserves_explicit_override(
    monkeypatch,
):
    monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "50")

    got = default_ornith_mlx_command_buffer_limit(
        _ornith_manifest(),
        generation=3,
        total_memory_bytes=128 * (1 << 30),
    )

    assert got is None
    assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "50"


def test_ornith_command_buffer_default_keeps_small_memory_policy(
    monkeypatch,
):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)

    got = default_ornith_mlx_command_buffer_limit(
        _ornith_manifest(),
        generation=3,
        total_memory_bytes=32 * (1 << 30),
    )

    assert got is None
    assert "MLX_MAX_MB_PER_BUFFER" not in os.environ


def test_ornith_command_buffer_default_is_hardware_and_family_specific(
    monkeypatch,
):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    deepseek = {
        "architecture": {"family": "deepseek_v4_flash"},
        "required_ops": ["fp16_passthrough", "kquant_dequant"],
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


def test_ornith_command_buffer_default_follows_the_served_shape(
    monkeypatch,
):
    """A rebuild keeps the tuning because the served shape decides."""
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    rebuilt = _ornith_manifest()
    rebuilt["artifact_id"] = "pkg:rebuilt"

    assert (
        default_ornith_mlx_command_buffer_limit(
            rebuilt,
            generation=3,
            total_memory_bytes=128 * (1 << 30),
        )
        == "288"
    )


def test_ornith_command_buffer_default_rejects_smoke_and_other_runtimes(
    monkeypatch,
):
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
    smoke = _ornith_manifest()
    smoke["architecture"]["smoke_max_experts"] = 8
    unsupported_codec = _ornith_manifest(
        artifact_id="pkg:unsupported-codec",
        required_ops=("affine_dequant", "fp16_passthrough", "unknown_dequant"),
    )
    unsupported = _ornith_manifest(artifact_id="pkg:unsupported", required_ops=())

    for manifest in (smoke, unsupported_codec, unsupported):
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


def test_command_buffer_predicate_answers_without_importing_mlx(monkeypatch):
    """The predicate reads the adapter kind, and that seam must stay MLX-free:
    the limit only takes effect if it is set before MLX is imported."""
    import sys

    monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
    assert serve._ornith_command_buffer_package(_ornith_manifest())
    assert "mlx.core" not in sys.modules


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
