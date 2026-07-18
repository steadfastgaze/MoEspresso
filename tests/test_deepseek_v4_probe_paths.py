from __future__ import annotations

from pathlib import Path

import pytest

from moespresso.correctness.deepseek_v4 import (
    attention_output_capture,
    attention_output_replay,
    attention_projection_capture,
    ffn_capture,
    gguf_routed_replay,
    rope_capture,
    router_diff,
    short_probe,
    stage_capture,
)


def test_probe_path_environment_helpers_expand_user(monkeypatch):
    monkeypatch.delenv(short_probe.SOURCE_ENV, raising=False)
    assert short_probe._path_from_env(short_probe.SOURCE_ENV) is None

    monkeypatch.setenv(short_probe.SOURCE_ENV, "~/models/ds4")
    assert short_probe._path_from_env(short_probe.SOURCE_ENV) == Path(
        "~/models/ds4"
    ).expanduser()

    monkeypatch.setenv(gguf_routed_replay.GGUF_RECIPE_ENV, "~/models/ds4.gguf")
    assert gguf_routed_replay._path_from_env(
        gguf_routed_replay.GGUF_RECIPE_ENV
    ) == Path("~/models/ds4.gguf").expanduser()


@pytest.mark.parametrize(
    ("arms", "message"),
    [
        ({"embedding_arm": "source_prompt_rows"}, "--source-dir"),
        ({"attention_projection_arm": "source_layer0_all"}, "--source-dir"),
        ({"attention_projection_arm": "source_layers0_3_all"}, "--source-dir"),
        ({"attention_projection_arm": "ds4_gguf_layer0_all"}, "--gguf-path"),
        ({"ffn_routed_arm": "ds4_gguf_layer1"}, "--gguf-path"),
    ],
)
def test_probe_path_validation_fails_closed_for_selected_arm(arms, message):
    with pytest.raises(ValueError, match=message):
        short_probe._validate_probe_input_paths(
            source_dir=None,
            gguf_path=None,
            **arms,
        )


def test_probe_path_validation_allows_default_arms_without_external_sources():
    short_probe._validate_probe_input_paths(source_dir=None, gguf_path=None)


def test_short_probe_default_arms_do_not_require_external_sources(monkeypatch):
    seen = {}

    def fake_run_probe(package_dir, **kwargs):
        seen.update({"package_dir": package_dir, **kwargs})
        return {"ok": True}

    monkeypatch.setattr(short_probe, "_DEFAULT_SOURCE", None)
    monkeypatch.setattr(short_probe, "_DEFAULT_GGUF", None)
    monkeypatch.setattr(short_probe, "run_probe", fake_run_probe)

    assert short_probe.main(["--package", "/package", "--cache-arm", "required_cache"]) == 0
    assert seen["source_dir"] is None
    assert seen["gguf_path"] is None


@pytest.mark.parametrize(
    ("arm_args", "message"),
    [
        (["--embedding-arm", "source_prompt_rows"], short_probe.SOURCE_ENV),
        (["--attention-projection-arm", "source_layer0_all"], short_probe.SOURCE_ENV),
        (
            ["--attention-projection-arm", "ds4_gguf_layer0_all"],
            short_probe.GGUF_RECIPE_ENV,
        ),
        (["--ffn-routed-arm", "ds4_gguf_layer1"], short_probe.GGUF_RECIPE_ENV),
    ],
)
def test_short_probe_cli_names_missing_required_environment(
    monkeypatch,
    capsys,
    arm_args,
    message,
):
    monkeypatch.setattr(short_probe, "_DEFAULT_SOURCE", None)
    monkeypatch.setattr(short_probe, "_DEFAULT_GGUF", None)

    with pytest.raises(SystemExit, match="2"):
        short_probe.main([
            "--package",
            "/package",
            "--cache-arm",
            "required_cache",
            *arm_args,
        ])

    assert message in capsys.readouterr().err


def test_gguf_routed_replay_requires_cli_or_environment(monkeypatch, capsys):
    monkeypatch.setattr(gguf_routed_replay, "DEFAULT_GGUF", None)

    with pytest.raises(SystemExit, match="2"):
        gguf_routed_replay.main([
            "--dump-prefix",
            "/dump",
            "--final-row",
            "0",
        ])

    assert gguf_routed_replay.GGUF_RECIPE_ENV in capsys.readouterr().err


def test_gguf_routed_replay_accepts_explicit_path(monkeypatch):
    seen = {}

    def fake_replay(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(gguf_routed_replay, "DEFAULT_GGUF", None)
    monkeypatch.setattr(gguf_routed_replay, "replay_gguf_routed", fake_replay)

    assert gguf_routed_replay.main([
        "--gguf",
        "/models/ds4.gguf",
        "--dump-prefix",
        "/dump",
        "--final-row",
        "0",
    ]) == 0
    assert seen["gguf_path"] == Path("/models/ds4.gguf")


def test_attention_output_replay_requires_cli_or_environment(monkeypatch, capsys):
    monkeypatch.setattr(attention_output_replay, "DEFAULT_GGUF", None)

    with pytest.raises(SystemExit, match="2"):
        attention_output_replay.main([
            "--package",
            "/package",
            "--dump-prefix",
            "/dump",
            "--final-row",
            "0",
        ])

    assert gguf_routed_replay.GGUF_RECIPE_ENV in capsys.readouterr().err


_CAPTURE_MODULES = (
    rope_capture,
    stage_capture,
    attention_output_capture,
    attention_projection_capture,
    ffn_capture,
)


@pytest.mark.parametrize("module", (router_diff, *_CAPTURE_MODULES))
def test_capture_clis_require_source_for_source_embedding(module, monkeypatch, capsys):
    monkeypatch.setattr(module, "_DEFAULT_SOURCE", None)
    if hasattr(module, "_DEFAULT_GGUF"):
        monkeypatch.setattr(module, "_DEFAULT_GGUF", None)

    with pytest.raises(SystemExit, match="2"):
        module.main([
            "--package",
            "/package",
            "--prompt-file",
            "/prompt",
            "--dump-prefix",
            "/dump",
            "--final-row",
            "0",
            "--embedding-arm",
            "source_prompt_rows",
        ])

    assert short_probe.SOURCE_ENV in capsys.readouterr().err


@pytest.mark.parametrize("module", _CAPTURE_MODULES)
def test_capture_clis_require_gguf_for_gguf_projection(module, monkeypatch, capsys):
    monkeypatch.setattr(module, "_DEFAULT_SOURCE", None)
    monkeypatch.setattr(module, "_DEFAULT_GGUF", None)

    with pytest.raises(SystemExit, match="2"):
        module.main([
            "--package",
            "/package",
            "--prompt-file",
            "/prompt",
            "--dump-prefix",
            "/dump",
            "--final-row",
            "0",
            "--attention-projection-arm",
            "ds4_gguf_layer0_all",
        ])

    assert short_probe.GGUF_RECIPE_ENV in capsys.readouterr().err
