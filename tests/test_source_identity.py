"""Immutable Hugging Face snapshot identities used by package builders."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from moespresso.core.artifact import artifact_producer, make_artifact, write_artifact
from moespresso.package.qwen4.iqk_package import (
    CALIBRATED_IQK_POLICY,
    FALLBACK_POLICY,
    Qwen4IQKPackageError,
    read_qwen4_iqk_allocation,
)
from moespresso.inventory.qwen4.source_identity import (
    QWEN4_HF_SNAPSHOT_SOURCE_SCHEMA,
    QWEN4_SHARD_COUNT,
    SourceIdentityError,
    qwen4_hf_snapshot_source_identity,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> tuple[Path, dict]:
    revision = "d" * 40
    root = tmp_path / "models--Qwen--Synthetic"
    snapshot = root / "snapshots" / revision
    blobs = root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"qwen4_exp"}')
    weight_map = {}
    for index in range(1, QWEN4_SHARD_COUNT + 1):
        shard = f"model-{index:05d}-of-{QWEN4_SHARD_COUNT:05d}.safetensors"
        blob_name = hashlib.sha256(f"blob-{index}".encode()).hexdigest()
        payload = bytes([index % 251]) * (index + 3)
        (blobs / blob_name).write_bytes(payload)
        (snapshot / shard).symlink_to(Path("../../blobs") / blob_name)
        weight_map[f"tensor.{index:03d}"] = shard
    index_path = snapshot / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": weight_map}, sort_keys=True))
    teacher = {
        "model_id": "Qwen/Qwen3.8-Flash-Next",
        "revision": revision,
        "config_sha256": _sha256(snapshot / "config.json"),
        "index_sha256": _sha256(index_path),
    }
    return snapshot, teacher


def _rewrite_index(snapshot: Path, payload: dict, teacher: dict) -> None:
    path = snapshot / "model.safetensors.index.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    teacher["index_sha256"] = _sha256(path)


def _surface_decision(path: Path, source_identity: dict) -> Path:
    rows = []
    for layer in range(48):
        for projection_index, projection in enumerate(("gate", "up", "down")):
            codec = "q8_0" if layer < 2 else "iq2_k"
            logical = [640, 2560] if projection != "down" else [2560, 640]
            stored = [2560, 768] if projection == "down" and codec == "iq2_k" else logical
            suffix = "gate_up_proj" if projection != "down" else "down_proj"
            rows.append(
                {
                    "kind": "expert",
                    "layer_index": layer,
                    "projection": projection,
                    "source_name": (
                        f"model.language_model.layers.{layer}.mlp.experts.{suffix}"
                    ),
                    "format": "kquant" if codec == "q8_0" else "iqk",
                    "codec": codec,
                    **({"layout": "iqk_relayout"} if codec != "q8_0" else {}),
                    "logical_shape": logical,
                    "stored_shape": stored,
                    "zero_padding": stored[1] - logical[1],
                    "calibration_policy": (
                        FALLBACK_POLICY if layer < 2 else CALIBRATED_IQK_POLICY
                    ),
                    "surface_cell_identity": f"{layer * 3 + projection_index:064x}",
                    "surface_run_contract_identity": "4" * 64,
                }
            )
    decision = make_artifact(
        "optimizer_decision",
        {
            "source_root": source_identity["teacher_source_identity"]["model_id"],
            "source_format": "hf_safetensors",
        },
        artifact_producer("test.qwen4_surface"),
        required_features=["calibration"],
        status="valid",
        source_probe_id="probe:" + "1" * 64,
        source_identity=source_identity,
        allocation=rows,
    )
    write_artifact(path, decision)
    return path


def _expert_inventory() -> dict:
    tensors = []
    for layer in range(48):
        tensors.extend(
            [
                {
                    "source_name": (
                        f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
                    ),
                    "kind": "expert",
                    "layer_index": layer,
                    "projection": "gate_up",
                    "shape": [512, 1280, 2560],
                },
                {
                    "source_name": (
                        f"model.language_model.layers.{layer}.mlp.experts.down_proj"
                    ),
                    "kind": "expert",
                    "layer_index": layer,
                    "projection": "down",
                    "shape": [512, 2560, 640],
                },
            ]
        )
    return {"tensors": tensors}


def test_qwen4_snapshot_identity_is_deterministic_and_canonically_ordered(
    tmp_path: Path,
) -> None:
    snapshot, teacher = _source(tmp_path)

    first = qwen4_hf_snapshot_source_identity(
        snapshot,
        teacher_source_identity=teacher,
    )
    second = qwen4_hf_snapshot_source_identity(
        snapshot,
        teacher_source_identity=teacher,
    )

    assert first == second
    assert first["schema"] == QWEN4_HF_SNAPSHOT_SOURCE_SCHEMA
    assert first["teacher_source_identity"] == teacher
    assert len(first["shards"]) == QWEN4_SHARD_COUNT
    assert [record["name"] for record in first["shards"]] == sorted(
        record["name"] for record in first["shards"]
    )
    manifest_hash = hashlib.sha256(
        json.dumps(first["shards"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert first["shard_manifest_sha256"] == manifest_hash
    without_identity = dict(first)
    without_identity.pop("snapshot_identity_sha256")
    assert first["snapshot_identity_sha256"] == hashlib.sha256(
        json.dumps(without_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_shared_snapshot_identity_is_surface_decision_builder_contract(
    tmp_path: Path,
) -> None:
    snapshot, teacher = _source(tmp_path)
    source_identity = qwen4_hf_snapshot_source_identity(
        snapshot,
        teacher_source_identity=teacher,
    )
    decision = _surface_decision(tmp_path / "decision.json", source_identity)

    cells, payload = read_qwen4_iqk_allocation(
        decision,
        inventory=_expert_inventory(),
        source_identity=source_identity,
    )

    assert len(cells) == 144
    assert payload["source_identity"] == source_identity

    changed = deepcopy(source_identity)
    changed["shards"][0]["size_bytes"] += 1
    with pytest.raises(Qwen4IQKPackageError, match="source identity drifted"):
        read_qwen4_iqk_allocation(
            decision,
            inventory=_expert_inventory(),
            source_identity=changed,
        )


def test_qwen4_snapshot_identity_refuses_missing_shard_symlink(tmp_path: Path) -> None:
    snapshot, teacher = _source(tmp_path)
    (snapshot / "model-00007-of-00131.safetensors").unlink()

    with pytest.raises(SourceIdentityError, match="shard paths do not match"):
        qwen4_hf_snapshot_source_identity(
            snapshot,
            teacher_source_identity=teacher,
        )


def test_qwen4_snapshot_identity_refuses_non_hash_blob_target(tmp_path: Path) -> None:
    snapshot, teacher = _source(tmp_path)
    path = snapshot / "model-00007-of-00131.safetensors"
    path.unlink()
    bad = snapshot.parent.parent / "blobs/not-a-content-hash"
    bad.write_bytes(b"bad")
    path.symlink_to(Path("../../blobs") / bad.name)

    with pytest.raises(SourceIdentityError, match="64-character"):
        qwen4_hf_snapshot_source_identity(
            snapshot,
            teacher_source_identity=teacher,
        )


def test_qwen4_snapshot_identity_refuses_empty_blob(tmp_path: Path) -> None:
    snapshot, teacher = _source(tmp_path)
    path = snapshot / "model-00007-of-00131.safetensors"
    path.resolve().write_bytes(b"")

    with pytest.raises(SourceIdentityError, match="blob is empty"):
        qwen4_hf_snapshot_source_identity(
            snapshot,
            teacher_source_identity=teacher,
        )


def test_qwen4_snapshot_identity_refuses_retarget_and_size_drift(
    tmp_path: Path,
) -> None:
    snapshot, teacher = _source(tmp_path)
    expected = qwen4_hf_snapshot_source_identity(
        snapshot,
        teacher_source_identity=teacher,
    )
    path = snapshot / "model-00007-of-00131.safetensors"
    path.unlink()
    replacement_name = "e" * 64
    replacement = snapshot.parent.parent / "blobs" / replacement_name
    replacement.write_bytes(b"replacement")
    path.symlink_to(Path("../../blobs") / replacement_name)

    with pytest.raises(SourceIdentityError, match="source identity drifted"):
        qwen4_hf_snapshot_source_identity(
            snapshot,
            teacher_source_identity=teacher,
            expected_source_identity=expected,
        )

    path.unlink()
    original_record = expected["shards"][6]
    original = snapshot.parent.parent / "blobs" / original_record["hf_blob_sha256"]
    path.symlink_to(Path("../../blobs") / original.name)
    original.write_bytes(original.read_bytes() + b"changed-size")
    with pytest.raises(SourceIdentityError, match="source identity drifted"):
        qwen4_hf_snapshot_source_identity(
            snapshot,
            teacher_source_identity=teacher,
            expected_source_identity=expected,
        )


@pytest.mark.parametrize(
    "field", ["model_id", "config_sha256", "index_sha256", "revision"]
)
def test_qwen4_snapshot_identity_refuses_teacher_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    snapshot, teacher = _source(tmp_path)
    bad = deepcopy(teacher)
    if field == "model_id":
        bad[field] = "Qwen/Other"
    else:
        bad[field] = ("c" * 40) if field == "revision" else ("c" * 64)

    with pytest.raises(SourceIdentityError):
        qwen4_hf_snapshot_source_identity(
            snapshot,
            teacher_source_identity=bad,
        )


@pytest.mark.parametrize("mutation", ["count", "name", "extra_path"])
def test_qwen4_snapshot_identity_refuses_noncanonical_shard_surface(
    tmp_path: Path,
    mutation: str,
) -> None:
    snapshot, teacher = _source(tmp_path)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    if mutation == "count":
        index["weight_map"].pop("tensor.131")
        _rewrite_index(snapshot, index, teacher)
    elif mutation == "name":
        index["weight_map"]["tensor.131"] = "weights-last.safetensors"
        _rewrite_index(snapshot, index, teacher)
    else:
        extra = snapshot / "model-00132-of-00132.safetensors"
        extra.symlink_to(
            (snapshot / "model-00131-of-00131.safetensors").readlink()
        )

    with pytest.raises(SourceIdentityError):
        qwen4_hf_snapshot_source_identity(
            snapshot,
            teacher_source_identity=teacher,
        )
