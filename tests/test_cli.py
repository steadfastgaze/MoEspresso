from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from moespresso.core.artifact import compute_artifact_id
from moespresso.runtime.drafter_cli import (
    ExternalDrafterError,
    detect_external_drafter,
    verify_external_dspark,
)


DS4_CONFIG = {
    "hidden_size": 4096,
    "vocab_size": 129280,
    "num_hidden_layers": 43,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "n_routed_experts": 256,
    "moe_intermediate_size": 2048,
    "num_experts_per_tok": 6,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
}


def _recognized_manifest(root: Path, name: str, kind: str) -> None:
    root.mkdir(exist_ok=True)
    (root / name).write_text(json.dumps({"artifact_kind": kind}))


def _verified_dspark(root: Path) -> dict:
    root.mkdir(exist_ok=True)
    shard = root / "model-dspark.safetensors"
    shard.write_bytes(b"draft")
    payload = {
        "artifact_kind": "deepseek_v4_dspark_sidecar",
        "schema_version": {"major": 1, "minor": 0},
        "producer": {"tool": "test", "version": "0"},
        "subject": {"family": "deepseek_v4_flash_dspark"},
        "status": "valid",
        "source_config": dict(DS4_CONFIG),
        "dspark": {
            "n_mtp_layers": 3,
            "block_size": DS4_CONFIG["dspark_block_size"],
            "noise_token_id": DS4_CONFIG["dspark_noise_token_id"],
            "target_layer_ids": DS4_CONFIG["dspark_target_layer_ids"],
            "markov_rank": DS4_CONFIG["dspark_markov_rank"],
        },
        "provenance": {
            "source_snapshot": "DeepSeek-V4-Flash-0731",
            "file_sha256": {
                shard.name: hashlib.sha256(shard.read_bytes()).hexdigest(),
            },
        },
        "tensors": {
            "draft.weight": {"format": "passthrough", "file": shard.name}
        },
    }
    payload["artifact_id"] = compute_artifact_id(payload)
    (root / "dspark_sidecar.json").write_text(json.dumps(payload))
    return payload


def _package_manifest() -> dict:
    return {
        "subject": {"source_root": "DeepSeek-V4-Flash-0731"},
        "architecture": {
            "family": "deepseek_v4_flash",
            "config": dict(DS4_CONFIG),
        },
    }


def test_top_level_cli_help_and_version(capsys):
    from moespresso import __version__
    from moespresso.cli import main

    assert main([]) == 0
    help_text = capsys.readouterr().out
    assert "usage: moespresso" in help_text
    assert "serve" in help_text and "generate" in help_text and "verify" in help_text
    assert "speed" in help_text
    assert "completions-api-timing" in help_text

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"MoEspresso {__version__}"


def test_spaced_cli_delegates_to_the_existing_parser(monkeypatch):
    import moespresso.serve_supervisor as supervisor
    from moespresso.cli import main

    seen = {}

    def fake_main(argv, *, prog):
        seen.update(argv=argv, prog=prog)
        return 7

    monkeypatch.setattr(supervisor, "main", fake_main)
    assert main(["serve", "pkg", "--port", "9000"]) == 7
    assert seen == {
        "argv": ["pkg", "--port", "9000"],
        "prog": "moespresso serve",
    }


def test_external_drafter_root_detection_accepts_only_dspark(tmp_path):
    dspark = tmp_path / "dspark"
    _recognized_manifest(
        dspark, "dspark_sidecar.json", "deepseek_v4_dspark_sidecar"
    )
    detected = detect_external_drafter(dspark)
    assert detected.family == "dspark"
    assert detected.root == dspark

    dflash = tmp_path / "dflash"
    _recognized_manifest(
        dflash, "dflash_sidecar.json", "deepseek_v4_dflash_sidecar"
    )
    with pytest.raises(ExternalDrafterError, match="recognized.*not supported"):
        detect_external_drafter(dflash)

    _recognized_manifest(dspark, "mtp_sidecar.json", "deepseek_v4_mtp_sidecar")
    with pytest.raises(ExternalDrafterError, match="multiple drafter manifests"):
        detect_external_drafter(dspark)


def test_external_dspark_verification_checks_bytes_geometry_and_source(tmp_path):
    sidecar = tmp_path / "sidecar"
    expected = _verified_dspark(sidecar)
    detected = detect_external_drafter(sidecar)

    assert verify_external_dspark(_package_manifest(), detected) == expected

    mismatched = _package_manifest()
    mismatched["architecture"]["config"]["hidden_size"] = 2048
    with pytest.raises(ExternalDrafterError, match="hidden_size"):
        verify_external_dspark(mismatched, detected)

    (sidecar / "model-dspark.safetensors").write_bytes(b"tampered")
    with pytest.raises(ExternalDrafterError, match="hash mismatch"):
        verify_external_dspark(_package_manifest(), detected)


def test_http_cli_threads_external_dspark_to_serve(tmp_path, monkeypatch):
    import moespresso.runtime.http as http

    sidecar = tmp_path / "sidecar"
    _recognized_manifest(
        sidecar, "dspark_sidecar.json", "deepseek_v4_dspark_sidecar"
    )
    seen = {}

    def fake_serve(package_dir, **kwargs):
        seen.update(package_dir=package_dir, **kwargs)
        return 0

    monkeypatch.setattr(http, "serve", fake_serve)
    assert http.main(["pkg", "--drafter", str(sidecar)]) == 0
    assert seen["external_drafter"] == sidecar


def test_http_cli_does_not_expose_a_tool_dialect_override(capsys):
    import moespresso.runtime.http as http

    with pytest.raises(SystemExit) as exc:
        http.main(["--help"])
    assert exc.value.code == 0
    assert "--tool-dialect" not in capsys.readouterr().out


def test_verify_cli_checks_external_dspark_without_loading(
    tmp_path, monkeypatch, capsys
):
    import moespresso.runtime.serve as serve

    package = tmp_path / "package"
    package.mkdir()
    manifest = {
        **_package_manifest(),
        "tensors": [],
        "files": [],
    }
    (package / "package_manifest.json").write_text(json.dumps(manifest))
    sidecar = tmp_path / "sidecar"
    payload = _verified_dspark(sidecar)

    monkeypatch.setattr(serve, "verify_package", lambda _manifest, _root: [])
    monkeypatch.setattr(
        serve, "verify_generated_sidecars", lambda _manifest, _root: []
    )
    assert serve.verify_main(
        [str(package), "--drafter", str(sidecar)], prog="moespresso verify"
    ) == 0
    output = capsys.readouterr().out
    assert "OK: external DSpark sidecar" in output
    assert payload["artifact_id"] in output


def test_load_served_model_external_dspark_overrides_environment(
    tmp_path, monkeypatch
):
    import moespresso.runtime.deepseek_v4.spec_serve as spec_serve
    from moespresso.runtime.serve import load_served_model

    seen = {}

    def fake_resolve(model, manifest, *, package_dir, env_value):
        seen.update(
            model=model,
            manifest=manifest,
            package_dir=package_dir,
            env_value=env_value,
        )
        return None, "dspark"

    monkeypatch.setattr(spec_serve, "resolve_env_drafter", fake_resolve)
    manifest = {
        "artifact_id": "pkg:test",
        "architecture": {"family": "deepseek_v4_flash"},
        "tensors": [],
        "files": [],
    }
    sidecar = tmp_path / "sidecar"
    load_served_model(
        tmp_path,
        manifest=manifest,
        build_fn=lambda _manifest, _path: ("MODEL", "TOKENIZER"),
        drafter=sidecar,
    )
    assert seen["env_value"] == f"dspark:{sidecar}"
