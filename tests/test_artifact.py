"""The artifact base contract: these tests ARE the contract's specification.

Test names describe behavior; the functions under test are pure and trivial to
test in isolation.
"""

from __future__ import annotations

import json

import pytest

from moespresso.core.artifact import (
    ArtifactError,
    Validation,
    canonical_json,
    compute_artifact_id,
    make_artifact,
    read_artifact,
    validate_base,
    write_artifact,
)

PRODUCER = {"tool": "test", "version": "0.0.1"}
SUBJECT = {"source_root": "toy", "source_format": "hf_safetensors"}


def _artifact(**extra):
    return make_artifact("source_inventory", SUBJECT, PRODUCER, **extra)


def test_make_artifact_fills_base_keys_and_id():
    a = _artifact()
    assert a["artifact_kind"] == "source_inventory"
    assert a["schema_version"] == {"major": 1, "minor": 0}
    assert a["status"] == "draft"
    assert a["artifact_id"].startswith("inv:")
    assert a["inputs"] == [] and a["validation"] == []
    assert a["required_features"] == []      # base field, empty by default
    assert "created_at" not in a             # not stamped until write time


def test_artifact_id_is_content_addressed_and_excludes_itself():
    a = _artifact(extra_field=1)
    # id must equal the hash recomputed from the payload
    assert a["artifact_id"] == compute_artifact_id(a)
    # changing content changes the id; same content -> same id (deterministic)
    b = _artifact(extra_field=1)
    c = _artifact(extra_field=2)
    assert a["artifact_id"] == b["artifact_id"]
    assert a["artifact_id"] != c["artifact_id"]


def test_canonical_json_is_sorted_and_excludes_id():
    a = _artifact(zeta=1, alpha=2)
    cj = canonical_json(a)
    assert "artifact_id" not in cj
    # keys are sorted -> 'alpha' appears before 'zeta'
    assert cj.index('"alpha"') < cj.index('"zeta"')
    assert json.loads(cj)["alpha"] == 2


def test_unknown_kind_fails_closed():
    with pytest.raises(ArtifactError):
        make_artifact("not_a_kind", SUBJECT, PRODUCER)


def test_unknown_major_version_fails_closed():
    a = _artifact()
    a["schema_version"] = {"major": 999, "minor": 0}
    with pytest.raises(ArtifactError):
        validate_base(a)


def test_bad_status_is_reported():
    a = _artifact()
    a["status"] = "weird"
    issues = validate_base(a)
    assert any(v.code == "artifact.bad_status" for v in issues)


def test_nan_inf_forbidden():
    with pytest.raises(ArtifactError):
        make_artifact("source_inventory", SUBJECT, PRODUCER, metric=float("nan"))
    with pytest.raises(ArtifactError):
        compute_artifact_id({"artifact_kind": "probe_evidence", "x": float("inf")})


def test_write_then_read_roundtrips_and_verifies(tmp_path):
    a = _artifact(payload_field={"b": 2, "a": 1})
    path = tmp_path / "inv.json"
    written_id = write_artifact(path, a)
    assert written_id == a["artifact_id"]
    back = read_artifact(path)
    assert back["artifact_id"] == a["artifact_id"]
    assert back["payload_field"] == {"b": 2, "a": 1}


def test_created_at_stamped_at_write_but_excluded_from_hash(tmp_path):
    a = _artifact(value=7)
    path = tmp_path / "inv.json"
    written_id = write_artifact(path, a, created_at="2026-06-01T00:00:00Z")
    back = read_artifact(path)
    # persisted artifact carries created_at (spec required) ...
    assert back["created_at"] == "2026-06-01T00:00:00Z"
    # ... but it did NOT change the content id (excluded from the hash)
    assert written_id == a["artifact_id"] == back["artifact_id"]


def test_created_at_does_not_perturb_id_regardless_of_value(tmp_path):
    a = _artifact(value=7)
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    write_artifact(p1, a, created_at="2026-01-01T00:00:00Z")
    write_artifact(p2, a, created_at="2099-12-31T23:59:59Z")
    # same content, different timestamps -> identical ids on read-back
    assert read_artifact(p1)["artifact_id"] == read_artifact(p2)["artifact_id"]


def test_required_features_default_empty_and_roundtrip():
    a = _artifact(required_features=["calibration"])
    assert a["required_features"] == ["calibration"]
    assert validate_base(a) == []  # known feature -> no issue


def test_unknown_required_feature_fails_closed():
    # fail-closed at construction: make_artifact raises on an unknown feature.
    with pytest.raises(ArtifactError):
        _artifact(required_features=["teleportation"])
    # and validate_base flags it as a blocking entry (e.g. on a read-back payload).
    payload = _artifact(value=1)
    payload["required_features"] = ["teleportation"]
    issues = validate_base(payload)
    assert any(v.code == "artifact.unknown_required_feature" and v.blocking
               for v in issues)


def test_read_rejects_tampered_artifact(tmp_path):
    a = _artifact(value=10)
    path = tmp_path / "inv.json"
    write_artifact(path, a)
    # tamper with the stored file without fixing the id
    obj = json.loads(path.read_text())
    obj["value"] = 11
    path.write_text(json.dumps(obj))
    with pytest.raises(ArtifactError):
        read_artifact(path)


def test_validation_entries_are_serialized():
    v = Validation("error", "tensor.shape_mismatch", "bad shape",
                   path="/tensors/0/shape", phase="inventory", blocking=True,
                   expected=[4, 4], actual=[4, 3])
    a = _artifact(validation=[v])
    entry = a["validation"][0]
    assert entry["code"] == "tensor.shape_mismatch"
    assert entry["expected"] == [4, 4] and entry["actual"] == [4, 3]
    assert entry["blocking"] is True


# --- correctness-ladder artifact kinds ---

def test_correctness_ladder_kinds_are_registered_and_tagged():
    # architecture_profile + correctness_evidence are registered artifact kinds with
    # their own id tags, validated by the same base contract as every other artifact.
    prof = make_artifact("architecture_profile", SUBJECT, PRODUCER, family="qwen3_5_moe")
    assert prof["artifact_kind"] == "architecture_profile"
    assert prof["artifact_id"].startswith("arch:")
    assert validate_base(prof) == []

    ev = make_artifact("correctness_evidence", SUBJECT, PRODUCER, status="valid", rung="L0")
    assert ev["artifact_id"].startswith("correct:")
    assert validate_base(ev) == []
    assert compute_artifact_id(ev) == ev["artifact_id"]   # content-addressed, deterministic
