"""Qwen4 teacher-capture calibration provider contracts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from moespresso.probe.qwen4.calibration import (
    QWEN4_TEACHER_DENSE_SCHEMA,
    QWEN4_TEACHER_DENSE_TARGET_CONTRACT,
    QWEN4_TEACHER_CALIBRATION_SCHEMA,
    Qwen4TeacherCalibrationError,
    _dense_target_widths,
    _target_set_sha256,
    qwen4_teacher_calibration,
    qwen4_teacher_dense_calibration,
    qwen4_teacher_embedding_counts,
    qwen4_teacher_expert_counts,
)
from moespresso.inventory.qwen4.roles import tensor_role
from moespresso.inventory.qwen4.static import expected_qwen38_flash_next_text_tensors


_LAYERS = 48
_EXPERTS = 512
_HIDDEN = 2560
_INTERMEDIATE = 640
_VALID_STATS_TOKENS = 512


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _layer_arrays(layer: int, *, score2: bool = False) -> dict[str, np.ndarray]:
    counts = np.full(_EXPERTS, 10, dtype=np.uint64)
    gate = np.full((_EXPERTS, _HIDDEN), 10.0 * (layer + 1), dtype=np.float32)
    down = np.full((_EXPERTS, _INTERMEDIATE), 10.0 * (layer + 2), dtype=np.float32)
    arrays = {
        "gate_up_in_sum2": gate,
        "down_in_sum2": down,
        "gate_up_count": counts,
        "down_count": counts.copy(),
        "route_weight_sum": np.full(_EXPERTS, 1.0, dtype=np.float64),
        "route_weight_sum2": np.full(_EXPERTS, 0.1, dtype=np.float64),
    }
    if score2:
        arrays["gate_up_score2_in_sum2"] = gate * 0.1
        arrays["down_score2_in_sum2"] = down * 0.1
    return arrays


def _layer_record(root: Path, path: Path, layer: int) -> dict:
    return {
        "layer_index": layer,
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


@pytest.fixture(scope="module")
def teacher_capture(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("qwen4-teacher-calibration")
    layer_dir = root / "layers"
    layer_dir.mkdir()
    records = []
    for layer in range(_LAYERS):
        path = layer_dir / f"layer{layer:02d}.npz"
        np.savez_compressed(path, **_layer_arrays(layer, score2=layer == 0))
        records.append(_layer_record(root, path, layer))
    manifest = {
        "schema": QWEN4_TEACHER_CALIBRATION_SCHEMA,
        "source_identity": {
            "model_id": "Qwen/Qwen3.8-Flash-Next",
            "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
            "config_sha256": "1" * 64,
            "index_sha256": "2" * 64,
        },
        "capture_identity": {
            "name": "synthetic-teacher-capture",
            "corpus_sha256": "3" * 64,
            "rendered_text_sha256": "4" * 64,
            "token_ids_sha256": "5" * 64,
            "tokenizer_sha256": "6" * 64,
            "renderer_sha256": "7" * 64,
        },
        "geometry": {
            "num_layers": _LAYERS,
            "num_experts": _EXPERTS,
            "top_k": 10,
            "hidden_size": _HIDDEN,
            "intermediate_size": _INTERMEDIATE,
        },
        "valid_stats_tokens": _VALID_STATS_TOKENS,
        "layers": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


@pytest.fixture(scope="module")
def dense_teacher_capture(teacher_capture: Path) -> Path:
    root = teacher_capture
    targets = _dense_target_widths()
    capture_execution_identity = "8" * 64
    stats_identity = _canonical_sha256(
        {
            "schema": "moespresso-qwen4-teacher-stats-v1",
            "capture_identity": capture_execution_identity,
            "target_contract": QWEN4_TEACHER_DENSE_TARGET_CONTRACT,
        }
    )
    arrays: dict[str, np.ndarray] = {
        "meta_schema": np.asarray("moespresso-qwen4-teacher-stats-v1"),
        "meta_capture_identity": np.asarray(capture_execution_identity),
        "meta_target_contract": np.asarray(QWEN4_TEACHER_DENSE_TARGET_CONTRACT),
        "meta_identity": np.asarray(stats_identity),
        "meta_target_count": np.asarray(len(targets), dtype="<u8"),
    }
    digest = hashlib.sha256(stats_identity.encode("ascii"))
    for index, target in enumerate(sorted(targets)):
        prefix = f"target_{index:04d}"
        width = targets[target]
        sum2 = np.full(width, _VALID_STATS_TOKENS * (index + 1), dtype="<f8")
        weighted = np.zeros(width, dtype="<f8")
        count = np.asarray(_VALID_STATS_TOKENS, dtype="<u8")
        zero_float = np.asarray(0, dtype="<f8")
        zero_count = np.asarray(0, dtype="<u8")
        arrays[f"{prefix}_name"] = np.asarray(target)
        arrays[f"{prefix}_sum2"] = sum2
        arrays[f"{prefix}_weighted_sum2"] = weighted
        arrays[f"{prefix}_count"] = count
        arrays[f"{prefix}_score_sum"] = zero_float
        arrays[f"{prefix}_score2_sum"] = zero_float.copy()
        arrays[f"{prefix}_scored_count"] = zero_count
        digest.update(target.encode("ascii"))
        digest.update(sum2.tobytes())
        digest.update(weighted.tobytes())
        digest.update(np.asarray([count], dtype="<u8").tobytes())
        digest.update(np.asarray([zero_float], dtype="<f8").tobytes())
        digest.update(np.asarray([zero_float], dtype="<f8").tobytes())
        digest.update(np.asarray([zero_count], dtype="<u8").tobytes())
    content_sha256 = digest.hexdigest()
    arrays["meta_content_sha256"] = np.asarray(content_sha256)
    dense_path = root / "dense-stats.npz"
    np.savez_compressed(dense_path, **arrays)

    embedding_path = root / "embedding-counts.npz"
    np.savez_compressed(
        embedding_path,
        token_ids=np.asarray([1, 2], dtype="<i8"),
        counts=np.asarray([256, 256], dtype="<u8"),
    )
    manifest = _manifest(root)
    source_hash = _canonical_sha256(manifest["source_identity"])
    manifest["dense_stats"] = {
        "schema": QWEN4_TEACHER_DENSE_SCHEMA,
        "path": dense_path.relative_to(root).as_posix(),
        "size_bytes": dense_path.stat().st_size,
        "sha256": _sha256(dense_path),
        "content_sha256": content_sha256,
        "target_contract": QWEN4_TEACHER_DENSE_TARGET_CONTRACT,
        "capture_execution_identity": capture_execution_identity,
        "source_identity_sha256": source_hash,
        "target_set_sha256": _target_set_sha256(targets),
        "target_count": len(targets),
        "embedding_counts": {
            "tensor": "model.language_model.embed_tokens.weight",
            "path": embedding_path.relative_to(root).as_posix(),
            "size_bytes": embedding_path.stat().st_size,
            "sha256": _sha256(embedding_path),
        },
    }
    return _write_manifest_variant(root, "dense-manifest", manifest)


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def _write_manifest_variant(root: Path, name: str, manifest: dict) -> Path:
    path = root / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _write_layer_variant(
    root: Path,
    name: str,
    arrays: dict[str, np.ndarray],
) -> Path:
    path = root / "layers" / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    manifest = _manifest(root)
    manifest["layers"][0] = _layer_record(root, path, 0)
    return _write_manifest_variant(root, name, manifest)


def _write_dense_variant(root: Path, name: str, mutate) -> Path:
    manifest = json.loads((root / "dense-manifest.json").read_text())
    source_path = root / manifest["dense_stats"]["path"]
    with np.load(source_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    mutate(arrays)
    path = root / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    manifest["dense_stats"].update(
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    )
    return _write_manifest_variant(root, name, manifest)


def test_teacher_capture_provides_exact_logical_vectors_and_identity(teacher_capture) -> None:
    vectors, identity = qwen4_teacher_calibration(teacher_capture)

    assert len(vectors) == _LAYERS * 3
    assert vectors["blk.0.ffn_gate_exps.weight"].shape == (_HIDDEN,)
    assert vectors["blk.0.ffn_down_exps.weight"].shape == (_INTERMEDIATE,)
    np.testing.assert_array_equal(
        vectors["blk.17.ffn_gate_exps.weight"],
        vectors["blk.17.ffn_up_exps.weight"],
    )
    np.testing.assert_allclose(vectors["blk.17.ffn_gate_exps.weight"], 18.0)
    np.testing.assert_allclose(vectors["blk.17.ffn_down_exps.weight"], 19.0)
    assert identity["kind"] == "qwen4_teacher_capture"
    assert identity["name"] == "synthetic-teacher-capture"
    assert identity["key_count"] == 144
    assert identity["source"]["revision"] == "de4b8e4d43b917e7706784d8bb445c9af86a3540"
    assert identity["geometry"]["num_experts"] == _EXPERTS
    assert len(identity["sha256"]) == 64


def test_teacher_capture_exposes_exact_per_layer_counts(teacher_capture) -> None:
    counts = qwen4_teacher_expert_counts(teacher_capture / "manifest.json")

    assert set(counts) == set(range(_LAYERS))
    assert counts[0].dtype == np.uint64
    np.testing.assert_array_equal(counts[47], np.full(_EXPERTS, 10, dtype=np.uint64))


def test_dense_target_inventory_is_the_exact_nonembedding_affine_role_set() -> None:
    expected = {
        name
        for name in expected_qwen38_flash_next_text_tensors()
        if name != "model.language_model.embed_tokens.weight"
        and tensor_role(name) is not None
        and tensor_role(name)["kind"] == "affine"
    }

    assert len(expected) == 773
    assert set(_dense_target_widths()) == expected


def test_dense_capture_provides_exact_affine_vectors_and_embedding_counts(
    dense_teacher_capture,
) -> None:
    vectors, identity = qwen4_teacher_dense_calibration(dense_teacher_capture)
    embedding_counts = qwen4_teacher_embedding_counts(dense_teacher_capture)
    ordered = sorted(_dense_target_widths())

    assert len(vectors) == 773
    assert vectors[ordered[0]].shape == (_dense_target_widths()[ordered[0]],)
    np.testing.assert_array_equal(vectors[ordered[17]], 18.0)
    assert identity["kind"] == "qwen4_teacher_dense_capture"
    assert identity["key_count"] == 773
    assert identity["source"]["revision"] == "de4b8e4d43b917e7706784d8bb445c9af86a3540"
    assert embedding_counts == {1: 256, 2: 256}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_count", 772, "target_count must be 773"),
        ("target_set_sha256", "0" * 64, "target set does not match"),
        ("source_identity_sha256", "0" * 64, "source identity does not match"),
        ("content_sha256", "0" * 64, "content digest mismatch"),
    ],
)
def test_dense_capture_rejects_identity_and_target_drift(
    dense_teacher_capture, field, value, message
) -> None:
    manifest = json.loads(dense_teacher_capture.read_text())
    manifest["dense_stats"][field] = value
    path = _write_manifest_variant(
        dense_teacher_capture.parent,
        f"bad-dense-{field}",
        manifest,
    )

    with pytest.raises(Qwen4TeacherCalibrationError, match=message):
        qwen4_teacher_dense_calibration(path)


def test_expert_provider_accepts_dense_manifest_extension(dense_teacher_capture) -> None:
    vectors, identity = qwen4_teacher_calibration(dense_teacher_capture)

    assert len(vectors) == _LAYERS * 3
    assert identity["kind"] == "qwen4_teacher_capture"


def test_dense_capture_rejects_affine_input_width_drift(dense_teacher_capture) -> None:
    def mutate(arrays) -> None:
        arrays["target_0000_sum2"] = arrays["target_0000_sum2"][:-1]

    path = _write_dense_variant(
        dense_teacher_capture.parent,
        "bad-dense-width",
        mutate,
    )

    with pytest.raises(Qwen4TeacherCalibrationError, match="sum2 has.*expected float64"):
        qwen4_teacher_dense_calibration(path)


def test_dense_capture_rejects_affine_observation_count_drift(
    dense_teacher_capture,
) -> None:
    def mutate(arrays) -> None:
        arrays["target_0000_count"] = np.asarray(511, dtype="<u8")

    path = _write_dense_variant(
        dense_teacher_capture.parent,
        "bad-dense-count",
        mutate,
    )

    with pytest.raises(Qwen4TeacherCalibrationError, match="count is 511, expected 512"):
        qwen4_teacher_dense_calibration(path)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("source_identity", "revision"), "short", "full lowercase 40-character"),
        (("capture_identity", "corpus_sha256"), "A" * 64, "lowercase SHA-256"),
        (("geometry", "num_layers"), 47, "num_layers is 47, expected 48"),
        (("geometry", "hidden_size"), 2559, "hidden_size is 2559, expected 2560"),
    ],
)
def test_teacher_capture_rejects_identity_and_geometry_drift(
    teacher_capture, field_path, value, message
) -> None:
    manifest = _manifest(teacher_capture)
    manifest[field_path[0]][field_path[1]] = value
    path = _write_manifest_variant(teacher_capture, f"bad-{field_path[1]}", manifest)

    with pytest.raises(Qwen4TeacherCalibrationError, match=message):
        qwen4_teacher_calibration(path)


def test_teacher_capture_rejects_layer_hash_mismatch(teacher_capture) -> None:
    manifest = _manifest(teacher_capture)
    manifest["layers"][0]["sha256"] = "0" * 64
    path = _write_manifest_variant(teacher_capture, "bad-layer-hash", manifest)

    with pytest.raises(Qwen4TeacherCalibrationError, match="layer 0 sha256"):
        qwen4_teacher_calibration(path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("gate-width", r"gate_up_in_sum2 has shape .* expected \(512, 2560\)"),
        ("down-width", r"down_in_sum2 has shape .* expected \(512, 640\)"),
        ("expert-count", r"gate_up_count has shape \(511,\), expected \(512,\)"),
        ("nonfinite", "gate_up_in_sum2 contains non-finite"),
        ("negative", "down_in_sum2 contains negative"),
        ("count-mismatch", "gate_up_count does not equal down_count"),
        ("count-total", "route count sum is 5121, expected 5120"),
        ("route-sum", "route_weight_sum exceeds route counts"),
        ("route-sum2", "route_weight_sum2 exceeds route_weight_sum"),
        ("score2", "gate_up_score2_in_sum2 exceeds gate_up_in_sum2"),
    ],
)
def test_teacher_capture_rejects_invalid_layer_statistics(
    teacher_capture, case, message
) -> None:
    arrays = _layer_arrays(0, score2=True)
    if case == "gate-width":
        arrays["gate_up_in_sum2"] = arrays["gate_up_in_sum2"][:, :-1]
    elif case == "down-width":
        arrays["down_in_sum2"] = arrays["down_in_sum2"][:, :-1]
    elif case == "expert-count":
        arrays["gate_up_count"] = arrays["gate_up_count"][:-1]
    elif case == "nonfinite":
        arrays["gate_up_in_sum2"][0, 0] = np.nan
    elif case == "negative":
        arrays["down_in_sum2"][0, 0] = -1.0
    elif case == "count-mismatch":
        arrays["down_count"][0] += 1
    elif case == "count-total":
        arrays["gate_up_count"][0] += 1
        arrays["down_count"][0] += 1
    elif case == "route-sum":
        arrays["route_weight_sum"][0] = 11.0
    elif case == "route-sum2":
        arrays["route_weight_sum2"][0] = 2.0
    elif case == "score2":
        arrays["gate_up_score2_in_sum2"][0, 0] = 11.0
    else:
        raise AssertionError(case)
    path = _write_layer_variant(teacher_capture, f"bad-{case}", arrays)

    with pytest.raises(Qwen4TeacherCalibrationError, match=message):
        qwen4_teacher_calibration(path)


def test_teacher_capture_requires_both_optional_score2_matrices(teacher_capture) -> None:
    arrays = _layer_arrays(0, score2=True)
    del arrays["down_score2_in_sum2"]
    path = _write_layer_variant(teacher_capture, "one-score2-matrix", arrays)

    with pytest.raises(Qwen4TeacherCalibrationError, match="both score2-weighted matrices"):
        qwen4_teacher_calibration(path)


def test_teacher_capture_rejects_noncanonical_layer_path(teacher_capture) -> None:
    manifest = deepcopy(_manifest(teacher_capture))
    manifest["layers"][0]["path"] = "../layer00.npz"
    path = _write_manifest_variant(teacher_capture, "escaped-layer-path", manifest)

    with pytest.raises(Qwen4TeacherCalibrationError, match="safe canonical relative path"):
        qwen4_teacher_calibration(path)
