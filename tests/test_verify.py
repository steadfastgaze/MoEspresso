"""Package verifier: pure, no mlx/jang. The 'never trust, verify' gate.

Builds a real package_manifest (via decide on tiny probe_evidence) plus a real
on-disk shard, then pins: a faithful package verifies clean, and every corruption
mode (truncated/edited file -> sha256, missing file, missing declared key) is
caught as a blocking validation.
"""

from __future__ import annotations

import json
import struct

import pytest

from moespresso.core.artifact import Validation, compute_artifact_id, make_artifact, write_artifact
from moespresso.optimize.allocate import AFFINE_BITS, EXPERT_BITS
from moespresso.optimize.decide import decide
from moespresso.package.manifest import build_package_manifest, file_identity, located_key
from moespresso.package.plan import package_plan_from_decision
from moespresso.package.sidecars import build_sidecars
from moespresso.runtime.verify import (
    PackageVerificationError,
    expected_keys,
    verify_generated_sidecars,
    verify_package,
)

SUBJECT = {"source_root": "toy", "source_format": "hf_safetensors"}
PRODUCER = {"tool": "test", "version": "0"}
ARCH = {
    "model_type": "qwen3_moe",
    "text_config": {"num_hidden_layers": 1, "num_experts": 8, "layer_types": ["full_attention"]},
}
SHARD = "model-00001-of-00001.safetensors"


def _affine_unit(name, role, layer_index=0):
    q = {f"{b}_{gs}": 0.99 for b in AFFINE_BITS for gs in (128, 64, 32)}
    return {
        "source_name": name,
        "kind": "affine",
        "role": role,
        "layer_index": layer_index,
        "shape": [64, 128],
        "importance": 1.0,
        "imatrix_mapped": True,
        "quality": q,
    }


def _expert_unit(name, layer, projection):
    q = {str(b): 0.9 + 0.02 * b for b in EXPERT_BITS}
    return {
        "source_name": name,
        "kind": "expert",
        "role": f"moe.expert.{projection}",
        "layer_index": layer,
        "projection": projection,
        "n_experts": 8,
        "sampled": 2,
        "shape": [64, 128],
        "importance": 1.0,
        "imatrix_mapped": True,
        "quality": q,
    }


def _decision():
    units = [
        _affine_unit("model.language_model.layers.0.self_attn.q_proj.weight", "attn.q_proj"),
        _expert_unit("model.language_model.layers.0.mlp.experts.gate", 0, "gate"),
    ]
    ev = make_artifact("probe_evidence", SUBJECT, PRODUCER, status="valid", units=units)
    return decide(ev, target_quality=0.5)


def _write_real_shard(tmp_path, decision):
    """Write a safetensors shard whose keys exactly match what the manifest expects."""
    tensors = {}
    for a in decision["allocation"]:
        if a["kind"] == "affine":
            tensors[f"{a['source_name']}.weight"] = b"\x00" * 16
            tensors[f"{a['source_name']}.scales"] = b"\x00" * 8
            tensors[f"{a['source_name']}.biases"] = b"\x00" * 8
        elif a["kind"] == "expert":
            # bundle format: one per-layer bundle tensor for all projections
            tensors[f"{a['source_name']}.tq_bundle"] = b"\x00" * 16
    header, blob, off = {}, bytearray(), 0
    for k, b in tensors.items():
        header[k] = {"dtype": "U8", "shape": [len(b)], "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hjson = json.dumps(header).encode()
    path = tmp_path / SHARD
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)
    return path


def _manifest(tmp_path):
    dec = _decision()
    plan, _summary = package_plan_from_decision(dec)
    located = {
        located_key(a): {"shard": SHARD, "key_prefix": a["source_name"]} for a in plan["allocation"]
    }
    path = _write_real_shard(tmp_path, plan)
    man = build_package_manifest(plan, ARCH, located, [file_identity(path)])
    return man, path


def _restamp(manifest):
    manifest = dict(manifest)
    manifest["artifact_id"] = compute_artifact_id(manifest)
    return manifest


def _write_sidecars(tmp_path, manifest, *, seed=42):
    config, jang_config = build_sidecars(manifest, seed=seed)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "jang_config.json").write_text(json.dumps(jang_config))


def test_faithful_package_verifies_clean(tmp_path):
    man, _ = _manifest(tmp_path)
    assert verify_package(man, tmp_path) == []


def test_release_verifier_accepts_historical_producer_version(tmp_path):
    """A new engine release must not invalidate an existing content-addressed package."""
    man, _ = _manifest(tmp_path)
    man = dict(man)
    man["producer"] = {**man["producer"], "version": "0.0.1"}
    man = _restamp(man)

    assert verify_package(man, tmp_path) == []


def test_expected_keys_per_format():
    assert expected_keys({"key_prefix": "x", "format": "tq"}) == ["x.tq_bundle"]
    assert expected_keys({"key_prefix": "mx", "format": "mxfp4"}) == ["mx.tq_bundle"]
    assert expected_keys({"key_prefix": "dmx", "format": "mxfp4", "kind": "affine"}) == [
        "dmx.weight",
        "dmx.scales",
    ]
    assert expected_keys({"key_prefix": "m8", "format": "mxfp8", "kind": "affine"}) == [
        "m8.weight",
        "m8.scales",
    ]
    assert expected_keys({"key_prefix": "y", "format": "affine"}) == [
        "y.weight",
        "y.scales",
        "y.biases",
    ]
    assert expected_keys({"key_prefix": "z", "format": "fp16"}) == ["z"]
    assert expected_keys({"key_prefix": "f32", "format": "f32_passthrough"}) == ["f32"]
    assert expected_keys({"key_prefix": "raw", "format": "raw_dtype_passthrough"}) == ["raw"]


def test_expected_keys_refuses_unknown_format():
    with pytest.raises(PackageVerificationError, match="unsupported tensor format"):
        expected_keys({"key_prefix": "x", "format": "nvfp4"})


def test_truncated_file_caught(tmp_path):
    man, path = _manifest(tmp_path)
    path.write_bytes(path.read_bytes()[:-4])  # truncate
    issues = verify_package(man, tmp_path)
    codes = {v.code for v in issues}
    assert "runtime.size_mismatch" in codes or "runtime.sha256_mismatch" in codes
    assert all(v.blocking for v in issues)


def test_edited_file_same_size_caught_by_sha256(tmp_path):
    man, path = _manifest(tmp_path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF  # flip a byte, same length
    path.write_bytes(bytes(data))
    issues = verify_package(man, tmp_path)
    assert any(v.code == "runtime.sha256_mismatch" and v.blocking for v in issues)


def test_missing_file_caught(tmp_path):
    man, path = _manifest(tmp_path)
    path.unlink()
    issues = verify_package(man, tmp_path)
    assert any(v.code == "runtime.missing_file" and v.blocking for v in issues)


def test_missing_declared_key_caught(tmp_path):
    """Manifest declares a tensor whose keys aren't actually in the shard."""
    man, path = _manifest(tmp_path)
    # Append a phantom tensor entry to the manifest pointing at the real shard,
    # then re-stamp file identity so only the key check fails (not sha256).
    man = dict(man)
    man["tensors"] = man["tensors"] + [
        {
            "source_name": "ghost",
            "role": "attn.k_proj",
            "kind": "affine",
            "shard": SHARD,
            "key_prefix": "ghost",
            "format": "affine",
            "format_params": {"bits": 4, "group_size": 64},
        }
    ]
    issues = verify_package(man, tmp_path)
    assert any(v.code == "runtime.missing_tensor_key" and v.blocking for v in issues)
    # sha256 still matches (we didn't touch the file)
    assert not any(v.code == "runtime.sha256_mismatch" for v in issues)


def test_unknown_tensor_format_caught_before_key_guessing(tmp_path):
    """Future codecs must add explicit key rules instead of falling back to raw prefix."""
    man, _ = _manifest(tmp_path)
    man = dict(man)
    man["tensors"] = man["tensors"] + [
        {
            "source_name": "future",
            "role": "moe.expert.gate",
            "kind": "expert",
            "shard": SHARD,
            "key_prefix": "future",
            "format": "nvfp4",
            "format_params": {"bits": 4, "group_size": 32},
        }
    ]
    issues = verify_package(man, tmp_path)
    assert any(v.code == "runtime.unsupported_tensor_format" and v.blocking for v in issues)


def test_manifest_content_status_and_embedded_blockers_are_checked(tmp_path):
    man, _ = _manifest(tmp_path)
    embedded = Validation(
        "error",
        "package.provenance_failed",
        "recorded package validation failed",
        phase="package",
        blocking=True,
    ).as_dict()
    man = _restamp({**man, "status": "invalid", "validation": [embedded]})

    issues = verify_package(man, tmp_path)
    codes = {issue.code for issue in issues}
    assert "runtime.manifest_not_valid" in codes
    assert "package.provenance_failed" in codes

    tampered = {**man, "status": "valid"}  # deliberately do not restamp
    assert any(
        issue.code == "runtime.manifest_id_mismatch" for issue in verify_package(tampered, tmp_path)
    )


def test_manifest_base_contract_is_checked_by_direct_verifier(tmp_path):
    man, _ = _manifest(tmp_path)
    man = dict(man)
    del man["producer"]
    man = _restamp(man)
    issues = verify_package(man, tmp_path)
    assert any(issue.code == "artifact.missing_key" and issue.blocking for issue in issues)


def test_tokenizer_and_agentic_profile_identities_are_hashed(tmp_path):
    man, _ = _manifest(tmp_path)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text('{"version":"1"}')
    profile_path = tmp_path / "agentic_profile.json"
    profile_path.write_text('{"schema_version":1}')
    man = _restamp(
        {
            **man,
            "tokenizer": {
                "files": [file_identity(tokenizer_path)],
                "rendering_id": "test",
                "has_tokenizer": True,
            },
            "agentic_profile": {
                **file_identity(profile_path),
                "family": "qwen3_5_moe",
            },
        }
    )
    assert verify_package(man, tmp_path) == []

    tokenizer_path.write_text('{"version":"2"}')  # same byte length
    issues = verify_package(man, tmp_path)
    assert any(
        issue.code == "runtime.sha256_mismatch" and issue.path == "/tokenizer.json"
        for issue in issues
    )

    profile_path.write_text('{"schema_version":2}')  # same byte length
    issues = verify_package(man, tmp_path)
    assert any(
        issue.code == "runtime.sha256_mismatch" and issue.path == "/agentic_profile.json"
        for issue in issues
    )


def test_unsafe_declared_identity_path_is_rejected(tmp_path):
    man, _ = _manifest(tmp_path)
    unsafe = {
        **man["files"][0],
        "path": "../outside.safetensors",
    }
    man = _restamp({**man, "files": [unsafe]})
    issues = verify_package(man, tmp_path)
    assert any(issue.code == "runtime.unsafe_declared_path" for issue in issues)


def test_generated_sidecars_match_manifest_semantics(tmp_path):
    man, _ = _manifest(tmp_path)
    _write_sidecars(tmp_path, man)
    assert verify_generated_sidecars(man, tmp_path) == []

    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "tampered"
    config_path.write_text(json.dumps(config))
    issues = verify_generated_sidecars(man, tmp_path)
    assert any(
        issue.code == "runtime.sidecar_semantic_mismatch" and issue.path == "/config.json"
        for issue in issues
    )

    (tmp_path / "jang_config.json").unlink()
    issues = verify_generated_sidecars(man, tmp_path)
    assert any(
        issue.code == "runtime.missing_sidecar" and issue.path == "/jang_config.json"
        for issue in issues
    )


def test_tq_manifest_seed_is_authoritative_for_sidecars(tmp_path):
    man, _ = _manifest(tmp_path)
    _write_sidecars(tmp_path, man, seed=7)
    issues = verify_generated_sidecars(man, tmp_path)
    assert any(issue.code == "runtime.sidecar_seed_mismatch" for issue in issues)


def test_kquant_sidecars_may_carry_a_consistent_nondefault_seed(tmp_path):
    man, _ = _manifest(tmp_path)
    tensors = []
    for tensor in man["tensors"]:
        if tensor["format"] == "tq":
            tensor = {
                **tensor,
                "format": "kquant",
                "format_params": {
                    "kquant_codec": "iq2_xxs",
                    "bits": 2.0625,
                    "group_size": 256,
                    "bytes_per_block": 66,
                    "weights_per_block": 256,
                },
            }
        tensors.append(tensor)
    man = _restamp({**man, "tensors": tensors})
    _write_sidecars(tmp_path, man, seed=7)
    assert verify_generated_sidecars(man, tmp_path) == []

    jang_path = tmp_path / "jang_config.json"
    jang = json.loads(jang_path.read_text())
    jang["mxtq_seed"] = 8
    jang_path.write_text(json.dumps(jang))
    issues = verify_generated_sidecars(man, tmp_path)
    assert any(issue.code == "runtime.sidecar_seed_mismatch" for issue in issues)


def test_verify_cli_includes_generated_sidecar_check(tmp_path, capsys):
    from moespresso.runtime.serve import verify_main

    man, _ = _manifest(tmp_path)
    write_artifact(tmp_path / "package_manifest.json", man)
    _write_sidecars(tmp_path, man)
    assert verify_main([str(tmp_path)]) == 0
    assert "OK:" in capsys.readouterr().out

    (tmp_path / "config.json").write_text("{}")
    assert verify_main([str(tmp_path)]) == 2
    assert "runtime.sidecar_semantic_mismatch" in capsys.readouterr().out

    _write_sidecars(tmp_path, man)
    tampered = {**man, "status": "invalid"}  # artifact_id intentionally stale
    (tmp_path / "package_manifest.json").write_text(json.dumps(tampered))
    assert verify_main([str(tmp_path)]) == 2
    output = capsys.readouterr().out
    assert "runtime.manifest_id_mismatch" in output
    assert "runtime.manifest_not_valid" in output
