"""Calibration provider for the released Qwen4 layer-major teacher capture."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from moespresso.inventory.qwen4.roles import module_weight_key, tensor_role
from moespresso.inventory.qwen4.static import (
    expected_qwen38_flash_next_header_specs,
    expected_qwen38_flash_next_text_tensors,
)


QWEN4_TEACHER_CALIBRATION_SCHEMA = "qwen4_teacher_calibration_v1"
QWEN4_TEACHER_MANIFEST_NAME = "manifest.json"
QWEN4_TEACHER_DENSE_SCHEMA = "qwen4_teacher_dense_calibration_v1"
QWEN4_TEACHER_DENSE_TARGET_CONTRACT = "qwen4-dense-affine-inputs-v1"

_MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
_LAYER_COUNT = 48
_EXPERT_COUNT = 512
_TOP_K = 10
_HIDDEN_SIZE = 2560
_INTERMEDIATE_SIZE = 640
_VOCAB_SIZE = 248320

_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "source_identity",
        "capture_identity",
        "geometry",
        "valid_stats_tokens",
        "layers",
    }
)
_DENSE_MANIFEST_FIELD = "dense_stats"
_SOURCE_IDENTITY_FIELDS = frozenset(
    {"model_id", "revision", "config_sha256", "index_sha256"}
)
_CAPTURE_IDENTITY_FIELDS = frozenset(
    {
        "name",
        "corpus_sha256",
        "rendered_text_sha256",
        "token_ids_sha256",
        "tokenizer_sha256",
        "renderer_sha256",
    }
)
_GEOMETRY = {
    "num_layers": _LAYER_COUNT,
    "num_experts": _EXPERT_COUNT,
    "top_k": _TOP_K,
    "hidden_size": _HIDDEN_SIZE,
    "intermediate_size": _INTERMEDIATE_SIZE,
}
_LAYER_FIELDS = frozenset({"layer_index", "path", "size_bytes", "sha256"})
_REQUIRED_ARRAYS = frozenset(
    {
        "gate_up_in_sum2",
        "down_in_sum2",
        "gate_up_count",
        "down_count",
        "route_weight_sum",
        "route_weight_sum2",
    }
)
_OPTIONAL_ARRAYS = frozenset(
    {"gate_up_score2_in_sum2", "down_score2_in_sum2"}
)
_DENSE_RECORD_FIELDS = frozenset(
    {
        "schema",
        "path",
        "size_bytes",
        "sha256",
        "content_sha256",
        "target_contract",
        "capture_execution_identity",
        "source_identity_sha256",
        "target_set_sha256",
        "target_count",
        "embedding_counts",
    }
)
_EMBEDDING_RECORD_FIELDS = frozenset(
    {"tensor", "path", "size_bytes", "sha256"}
)
_DENSE_STATS_NPZ_SCHEMA = "moespresso-qwen4-teacher-stats-v1"


class Qwen4TeacherCalibrationError(ValueError):
    """The Qwen4 teacher capture is missing, malformed, or identity-incompatible."""


def _fail(message: str) -> Qwen4TeacherCalibrationError:
    return Qwen4TeacherCalibrationError(message)


def _exact_fields(value: object, expected: frozenset[str], *, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise _fail(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        raise _fail(
            f"{field} fields do not match the schema: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{field} must be an integer >= {minimum}")
    return value


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _revision(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail("source_identity.revision must be a full lowercase 40-character revision")
    return value


def _manifest_path(capture_path: str | Path) -> Path:
    path = Path(capture_path)
    if path.is_dir():
        path = path / QWEN4_TEACHER_MANIFEST_NAME
    if not path.is_file():
        raise _fail(f"Qwen4 teacher capture manifest is missing: {path}")
    return path


def _read_manifest(capture_path: str | Path) -> tuple[Path, dict]:
    path = _manifest_path(capture_path)
    try:
        with open(path, encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _fail(f"could not read Qwen4 teacher capture manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise _fail("manifest must be an object")
    actual_fields = set(payload)
    allowed_fields = _MANIFEST_FIELDS | {_DENSE_MANIFEST_FIELD}
    if not _MANIFEST_FIELDS <= actual_fields or not actual_fields <= allowed_fields:
        raise _fail(
            "manifest fields do not match the schema: "
            f"missing={sorted(_MANIFEST_FIELDS - actual_fields)} "
            f"extra={sorted(actual_fields - allowed_fields)}"
        )
    if payload.get("schema") != QWEN4_TEACHER_CALIBRATION_SCHEMA:
        raise _fail(
            f"manifest schema {payload.get('schema')!r} is not "
            f"{QWEN4_TEACHER_CALIBRATION_SCHEMA!r}"
        )
    return path, payload


def _validate_identities(manifest: Mapping) -> tuple[dict, dict]:
    source = dict(
        _exact_fields(
            manifest.get("source_identity"),
            _SOURCE_IDENTITY_FIELDS,
            field="source_identity",
        )
    )
    if source.get("model_id") != _MODEL_ID:
        raise _fail(f"source_identity.model_id must be {_MODEL_ID!r}")
    source["revision"] = _revision(source.get("revision"))
    for field in ("config_sha256", "index_sha256"):
        source[field] = _sha256(source.get(field), field=f"source_identity.{field}")

    capture = dict(
        _exact_fields(
            manifest.get("capture_identity"),
            _CAPTURE_IDENTITY_FIELDS,
            field="capture_identity",
        )
    )
    if not isinstance(capture.get("name"), str) or not capture["name"]:
        raise _fail("capture_identity.name must be a non-empty string")
    for field in sorted(_CAPTURE_IDENTITY_FIELDS - {"name"}):
        capture[field] = _sha256(capture.get(field), field=f"capture_identity.{field}")
    return source, capture


def _validate_geometry(manifest: Mapping) -> int:
    geometry = _exact_fields(
        manifest.get("geometry"), frozenset(_GEOMETRY), field="geometry"
    )
    for field, expected in _GEOMETRY.items():
        actual = _integer(geometry.get(field), field=f"geometry.{field}", minimum=1)
        if actual != expected:
            raise _fail(f"geometry.{field} is {actual}, expected {expected}")
    return _integer(
        manifest.get("valid_stats_tokens"), field="valid_stats_tokens", minimum=1
    )


def _declared_layer_path(root: Path, value: object, *, layer: int) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise _fail(f"layers[{layer}].path must be a non-empty string")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise _fail(f"layers[{layer}].path is not a safe canonical relative path")
    try:
        resolved_root = root.resolve()
        resolved = (resolved_root / value).resolve()
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _fail(f"layers[{layer}].path escapes the capture root") from exc
    return value, resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _fail(f"could not hash Qwen4 teacher capture file {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _fail(f"could not canonicalize dense calibration identity: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _dense_target_widths() -> dict[str, int]:
    header_specs = expected_qwen38_flash_next_header_specs()
    targets = {
        name: int(header_specs[name][1][-1])
        for name in expected_qwen38_flash_next_text_tensors()
        if name != "model.language_model.embed_tokens.weight"
        and (resolved := tensor_role(name)) is not None
        and resolved["kind"] == "affine"
    }
    if len(targets) != 773:
        raise AssertionError(f"released Qwen4 dense target inventory has {len(targets)} entries")
    if any(module_weight_key(name) is None for name in targets):
        raise AssertionError("released Qwen4 dense target has no resolved module key")
    return targets


def _target_set_sha256(targets: Mapping[str, int]) -> str:
    return _canonical_sha256(
        [
            {"input_width": targets[name], "source_tensor": name}
            for name in sorted(targets)
        ]
    )


def _declared_artifact_path(root: Path, value: object, *, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise _fail(f"{field} must be a non-empty string")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise _fail(f"{field} is not a safe canonical relative path")
    try:
        resolved_root = root.resolve()
        resolved = (resolved_root / value).resolve()
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _fail(f"{field} escapes the capture root") from exc
    return value, resolved


def _artifact_record(
    root: Path,
    record: Mapping,
    *,
    field: str,
) -> tuple[Path, int]:
    _relative, path = _declared_artifact_path(root, record.get("path"), field=f"{field}.path")
    size = _integer(record.get("size_bytes"), field=f"{field}.size_bytes", minimum=1)
    expected_hash = _sha256(record.get("sha256"), field=f"{field}.sha256")
    if not path.is_file():
        raise _fail(f"{field} file is missing: {path}")
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise _fail(f"could not stat {field} file {path}: {exc}") from exc
    if actual_size != size:
        raise _fail(f"{field} size is {actual_size}, expected {size}")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise _fail(f"{field} sha256 is {actual_hash}, expected {expected_hash}")
    return path, size


def _layer_records(manifest_path: Path, manifest: Mapping) -> list[tuple[Path, int]]:
    raw_layers = manifest.get("layers")
    if not isinstance(raw_layers, list) or len(raw_layers) != _LAYER_COUNT:
        raise _fail(f"layers must contain exactly {_LAYER_COUNT} records")
    records: list[tuple[Path, int]] = []
    seen_paths: set[str] = set()
    for position, raw in enumerate(raw_layers):
        record = _exact_fields(raw, _LAYER_FIELDS, field=f"layers[{position}]")
        layer = _integer(record.get("layer_index"), field=f"layers[{position}].layer_index")
        if layer != position:
            raise _fail(f"layers[{position}].layer_index is {layer}, expected {position}")
        relative, path = _declared_layer_path(
            manifest_path.parent, record.get("path"), layer=layer
        )
        if relative in seen_paths:
            raise _fail(f"duplicate layer capture path {relative!r}")
        seen_paths.add(relative)
        size = _integer(record.get("size_bytes"), field=f"layers[{layer}].size_bytes", minimum=1)
        expected_hash = _sha256(record.get("sha256"), field=f"layers[{layer}].sha256")
        if not path.is_file():
            raise _fail(f"layer {layer} capture file is missing: {path}")
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise _fail(f"could not stat layer {layer} capture file {path}: {exc}") from exc
        if actual_size != size:
            raise _fail(f"layer {layer} size is {actual_size}, expected {size}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise _fail(
                f"layer {layer} sha256 is {actual_hash}, expected {expected_hash}"
            )
        records.append((path, size))
    return records


def _numeric_array(data, name: str, shape: tuple[int, ...], *, layer: int) -> np.ndarray:
    try:
        value = np.asarray(data[name])
    except (KeyError, ValueError, TypeError) as exc:
        raise _fail(f"layer {layer} could not read array {name!r}: {exc}") from exc
    if value.shape != shape:
        raise _fail(f"layer {layer} {name} has shape {value.shape}, expected {shape}")
    if value.dtype.kind not in "iuf":
        raise _fail(f"layer {layer} {name} must have a real numeric dtype")
    value = value.astype(np.float64, copy=False)
    if not np.all(np.isfinite(value)):
        raise _fail(f"layer {layer} {name} contains non-finite values")
    if np.any(value < 0):
        raise _fail(f"layer {layer} {name} contains negative values")
    return value


def _count_array(data, name: str, *, layer: int) -> np.ndarray:
    value = _numeric_array(data, name, (_EXPERT_COUNT,), layer=layer)
    if np.any(value != np.floor(value)):
        raise _fail(f"layer {layer} {name} must contain integer-valued counts")
    if np.any(value > np.iinfo(np.uint64).max):
        raise _fail(f"layer {layer} {name} exceeds uint64")
    return value.astype(np.uint64)


def _load_layer(path: Path, layer: int, valid_stats_tokens: int) -> tuple[np.ndarray, ...]:
    try:
        with np.load(path, allow_pickle=False) as data:
            names = set(data.files)
            missing = _REQUIRED_ARRAYS - names
            extra = names - _REQUIRED_ARRAYS - _OPTIONAL_ARRAYS
            if missing or extra:
                raise _fail(
                    f"layer {layer} arrays do not match the schema: "
                    f"missing={sorted(missing)} extra={sorted(extra)}"
                )
            optional = names & _OPTIONAL_ARRAYS
            if optional and optional != _OPTIONAL_ARRAYS:
                raise _fail(f"layer {layer} must provide both score2-weighted matrices")

            gate_sum2 = _numeric_array(
                data,
                "gate_up_in_sum2",
                (_EXPERT_COUNT, _HIDDEN_SIZE),
                layer=layer,
            )
            down_sum2 = _numeric_array(
                data,
                "down_in_sum2",
                (_EXPERT_COUNT, _INTERMEDIATE_SIZE),
                layer=layer,
            )
            gate_count = _count_array(data, "gate_up_count", layer=layer)
            down_count = _count_array(data, "down_count", layer=layer)
            route_sum = _numeric_array(
                data, "route_weight_sum", (_EXPERT_COUNT,), layer=layer
            )
            route_sum2 = _numeric_array(
                data, "route_weight_sum2", (_EXPERT_COUNT,), layer=layer
            )

            if not np.array_equal(gate_count, down_count):
                raise _fail(f"layer {layer} gate_up_count does not equal down_count")
            expected_total = valid_stats_tokens * _TOP_K
            actual_total = sum(int(value) for value in gate_count)
            if actual_total != expected_total:
                raise _fail(
                    f"layer {layer} route count sum is {actual_total}, "
                    f"expected {expected_total}"
                )
            count_float = gate_count.astype(np.float64)
            if np.any(route_sum > count_float):
                raise _fail(f"layer {layer} route_weight_sum exceeds route counts")
            if np.any(route_sum2 > route_sum):
                raise _fail(f"layer {layer} route_weight_sum2 exceeds route_weight_sum")

            if optional:
                gate_score2 = _numeric_array(
                    data,
                    "gate_up_score2_in_sum2",
                    (_EXPERT_COUNT, _HIDDEN_SIZE),
                    layer=layer,
                )
                down_score2 = _numeric_array(
                    data,
                    "down_score2_in_sum2",
                    (_EXPERT_COUNT, _INTERMEDIATE_SIZE),
                    layer=layer,
                )
                if np.any(gate_score2 > gate_sum2):
                    raise _fail(
                        f"layer {layer} gate_up_score2_in_sum2 exceeds gate_up_in_sum2"
                    )
                if np.any(down_score2 > down_sum2):
                    raise _fail(
                        f"layer {layer} down_score2_in_sum2 exceeds down_in_sum2"
                    )
    except Qwen4TeacherCalibrationError:
        raise
    except (OSError, ValueError) as exc:
        raise _fail(f"could not read layer {layer} capture file {path}: {exc}") from exc
    return gate_sum2, down_sum2, gate_count


def _capture_identity(manifest: Mapping, source: dict, capture: dict, size_bytes: int) -> dict:
    try:
        canonical = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _fail(f"manifest is not canonical finite JSON: {exc}") from exc
    return {
        "kind": "qwen4_teacher_capture",
        "name": capture["name"],
        "size_bytes": size_bytes,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "key_count": _LAYER_COUNT * 3,
        "sampling": "per_expert_channel_in_sum2_over_route_active_counts",
        "source": source,
        "capture": capture,
        "valid_stats_tokens": manifest["valid_stats_tokens"],
        "geometry": dict(_GEOMETRY),
    }


def _load_capture(capture_path: str | Path) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray], dict]:
    manifest_path, manifest = _read_manifest(capture_path)
    source, capture = _validate_identities(manifest)
    valid_stats_tokens = _validate_geometry(manifest)
    records = _layer_records(manifest_path, manifest)

    vectors: dict[str, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    for layer, (path, _size) in enumerate(records):
        gate_sum2, down_sum2, gate_count = _load_layer(
            path, layer, valid_stats_tokens
        )
        total = float(sum(int(value) for value in gate_count))
        gate = (gate_sum2.sum(axis=0, dtype=np.float64) / total).astype(np.float32)
        down = (down_sum2.sum(axis=0, dtype=np.float64) / total).astype(np.float32)
        vectors[f"blk.{layer}.ffn_gate_exps.weight"] = gate
        vectors[f"blk.{layer}.ffn_up_exps.weight"] = gate.copy()
        vectors[f"blk.{layer}.ffn_down_exps.weight"] = down
        counts[layer] = gate_count.copy()

    size_bytes = manifest_path.stat().st_size + sum(size for _path, size in records)
    identity = _capture_identity(manifest, source, capture, size_bytes)
    return vectors, counts, identity


def _npz_scalar(data, name: str, *, field: str) -> object:
    try:
        value = np.asarray(data[name])
    except (KeyError, ValueError, TypeError) as exc:
        raise _fail(f"{field} is missing: {exc}") from exc
    if value.shape != ():
        raise _fail(f"{field} must be a scalar")
    return value.item()


def _load_dense_stats(
    path: Path,
    *,
    capture_execution_identity: str,
    valid_stats_tokens: int,
    expected_content_sha256: str,
) -> dict[str, np.ndarray]:
    targets = _dense_target_widths()
    ordered = sorted(targets)
    metadata_names = {
        "meta_schema",
        "meta_capture_identity",
        "meta_target_contract",
        "meta_identity",
        "meta_content_sha256",
        "meta_target_count",
    }
    expected_names = set(metadata_names)
    for index in range(len(ordered)):
        prefix = f"target_{index:04d}"
        expected_names.update(
            {
                f"{prefix}_name",
                f"{prefix}_sum2",
                f"{prefix}_weighted_sum2",
                f"{prefix}_count",
                f"{prefix}_score_sum",
                f"{prefix}_score2_sum",
                f"{prefix}_scored_count",
            }
        )
    vectors: dict[str, np.ndarray] = {}
    digest = hashlib.sha256()
    try:
        with np.load(path, allow_pickle=False) as data:
            actual_names = set(data.files)
            if actual_names != expected_names:
                raise _fail(
                    "dense statistics arrays do not match the schema: "
                    f"missing={sorted(expected_names - actual_names)} "
                    f"extra={sorted(actual_names - expected_names)}"
                )
            if _npz_scalar(data, "meta_schema", field="dense_stats.meta_schema") != _DENSE_STATS_NPZ_SCHEMA:
                raise _fail("dense statistics payload has an unsupported schema")
            if _npz_scalar(
                data,
                "meta_capture_identity",
                field="dense_stats.meta_capture_identity",
            ) != capture_execution_identity:
                raise _fail("dense statistics capture identity mismatch")
            if _npz_scalar(
                data,
                "meta_target_contract",
                field="dense_stats.meta_target_contract",
            ) != QWEN4_TEACHER_DENSE_TARGET_CONTRACT:
                raise _fail("dense statistics target contract mismatch")
            identity = _canonical_sha256(
                {
                    "schema": _DENSE_STATS_NPZ_SCHEMA,
                    "capture_identity": capture_execution_identity,
                    "target_contract": QWEN4_TEACHER_DENSE_TARGET_CONTRACT,
                }
            )
            if _npz_scalar(data, "meta_identity", field="dense_stats.meta_identity") != identity:
                raise _fail("dense statistics identity mismatch")
            target_count = np.asarray(data["meta_target_count"])
            if target_count.shape != () or target_count.dtype != np.dtype("uint64"):
                raise _fail("dense statistics target count must be a uint64 scalar")
            if int(target_count) != len(ordered):
                raise _fail(
                    f"dense statistics target count is {int(target_count)}, expected {len(ordered)}"
                )

            digest.update(identity.encode("ascii"))
            for index, expected_target in enumerate(ordered):
                prefix = f"target_{index:04d}"
                target_array = np.asarray(data[f"{prefix}_name"])
                if target_array.shape != () or target_array.dtype.kind != "U":
                    raise _fail(f"dense target {index} name must be a Unicode scalar")
                target = str(target_array.item())
                if target != expected_target:
                    raise _fail(
                        f"dense target {index} is {target!r}, expected {expected_target!r}"
                    )
                width = targets[target]
                sum2 = np.asarray(data[f"{prefix}_sum2"])
                weighted = np.asarray(data[f"{prefix}_weighted_sum2"])
                if sum2.dtype != np.dtype("float64") or sum2.shape != (width,):
                    raise _fail(
                        f"dense target {target} sum2 has {sum2.dtype} {sum2.shape}, "
                        f"expected float64 ({width},)"
                    )
                if weighted.dtype != np.dtype("float64") or weighted.shape != (width,):
                    raise _fail(f"dense target {target} weighted_sum2 must match sum2")
                if not np.all(np.isfinite(sum2)) or np.any(sum2 < 0):
                    raise _fail(f"dense target {target} sum2 contains invalid values")
                if np.any(weighted != 0):
                    raise _fail(f"dense target {target} unexpectedly has weighted statistics")

                count = np.asarray(data[f"{prefix}_count"])
                scored_count = np.asarray(data[f"{prefix}_scored_count"])
                if count.shape != () or count.dtype != np.dtype("uint64"):
                    raise _fail(f"dense target {target} count must be a uint64 scalar")
                if scored_count.shape != () or scored_count.dtype != np.dtype("uint64"):
                    raise _fail(f"dense target {target} scored_count must be a uint64 scalar")
                if int(count) != valid_stats_tokens:
                    raise _fail(
                        f"dense target {target} count is {int(count)}, "
                        f"expected {valid_stats_tokens}"
                    )
                if int(scored_count) != 0:
                    raise _fail(f"dense target {target} unexpectedly has scored observations")

                score_sum = np.asarray(data[f"{prefix}_score_sum"])
                score2_sum = np.asarray(data[f"{prefix}_score2_sum"])
                if score_sum.shape != () or score_sum.dtype != np.dtype("float64"):
                    raise _fail(f"dense target {target} score_sum must be a float64 scalar")
                if score2_sum.shape != () or score2_sum.dtype != np.dtype("float64"):
                    raise _fail(f"dense target {target} score2_sum must be a float64 scalar")
                if float(score_sum) != 0 or float(score2_sum) != 0:
                    raise _fail(f"dense target {target} unexpectedly has score sums")

                digest.update(target.encode("ascii"))
                digest.update(np.ascontiguousarray(sum2, dtype="<f8").tobytes())
                digest.update(np.ascontiguousarray(weighted, dtype="<f8").tobytes())
                digest.update(np.asarray([count], dtype="<u8").tobytes())
                digest.update(np.asarray([score_sum], dtype="<f8").tobytes())
                digest.update(np.asarray([score2_sum], dtype="<f8").tobytes())
                digest.update(np.asarray([scored_count], dtype="<u8").tobytes())
                vectors[target] = (sum2 / float(valid_stats_tokens)).astype(np.float32)

            embedded_content = _npz_scalar(
                data,
                "meta_content_sha256",
                field="dense_stats.meta_content_sha256",
            )
    except Qwen4TeacherCalibrationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise _fail(f"could not read dense statistics {path}: {exc}") from exc
    actual_content = digest.hexdigest()
    if embedded_content != actual_content or expected_content_sha256 != actual_content:
        raise _fail("dense statistics content digest mismatch")
    return vectors


def _load_embedding_counts(
    path: Path,
    *,
    valid_stats_tokens: int,
) -> dict[int, int]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"token_ids", "counts"}:
                raise _fail("embedding counts arrays do not match the schema")
            token_ids = np.asarray(data["token_ids"])
            counts = np.asarray(data["counts"])
    except Qwen4TeacherCalibrationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise _fail(f"could not read embedding counts {path}: {exc}") from exc
    if token_ids.dtype != np.dtype("int64") or token_ids.ndim != 1:
        raise _fail("embedding token_ids must be a one-dimensional int64 array")
    if counts.dtype != np.dtype("uint64") or counts.shape != token_ids.shape:
        raise _fail("embedding counts must be a uint64 vector matching token_ids")
    if token_ids.size == 0 or np.any(token_ids < 0) or np.any(token_ids >= _VOCAB_SIZE):
        raise _fail("embedding token_ids contain values outside the released vocabulary")
    if not np.array_equal(token_ids, np.unique(token_ids)):
        raise _fail("embedding token_ids must be sorted and unique")
    if np.any(counts == 0) or sum(int(value) for value in counts) != valid_stats_tokens:
        raise _fail("embedding counts do not cover the exact valid stats-token population")
    return {int(token): int(count) for token, count in zip(token_ids, counts, strict=True)}


def _load_dense_capture(
    capture_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[int, int], dict]:
    manifest_path, manifest = _read_manifest(capture_path)
    source, capture = _validate_identities(manifest)
    valid_stats_tokens = _validate_geometry(manifest)
    record = _exact_fields(
        manifest.get(_DENSE_MANIFEST_FIELD),
        _DENSE_RECORD_FIELDS,
        field=_DENSE_MANIFEST_FIELD,
    )
    if record.get("schema") != QWEN4_TEACHER_DENSE_SCHEMA:
        raise _fail("dense_stats has an unsupported schema")
    if record.get("target_contract") != QWEN4_TEACHER_DENSE_TARGET_CONTRACT:
        raise _fail("dense_stats target contract is incompatible")
    capture_execution_identity = _sha256(
        record.get("capture_execution_identity"),
        field="dense_stats.capture_execution_identity",
    )
    source_hash = _canonical_sha256(source)
    if record.get("source_identity_sha256") != source_hash:
        raise _fail("dense_stats source identity does not match the teacher manifest")
    targets = _dense_target_widths()
    if _integer(record.get("target_count"), field="dense_stats.target_count", minimum=1) != len(targets):
        raise _fail(f"dense_stats target_count must be {len(targets)}")
    if record.get("target_set_sha256") != _target_set_sha256(targets):
        raise _fail("dense_stats target set does not match released Qwen4")
    expected_content = _sha256(
        record.get("content_sha256"),
        field="dense_stats.content_sha256",
    )
    dense_path, dense_size = _artifact_record(
        manifest_path.parent,
        record,
        field="dense_stats",
    )
    vectors = _load_dense_stats(
        dense_path,
        capture_execution_identity=capture_execution_identity,
        valid_stats_tokens=valid_stats_tokens,
        expected_content_sha256=expected_content,
    )

    embedding_record = _exact_fields(
        record.get("embedding_counts"),
        _EMBEDDING_RECORD_FIELDS,
        field="dense_stats.embedding_counts",
    )
    if embedding_record.get("tensor") != "model.language_model.embed_tokens.weight":
        raise _fail("dense_stats embedding tensor does not match released Qwen4")
    embedding_path, embedding_size = _artifact_record(
        manifest_path.parent,
        embedding_record,
        field="dense_stats.embedding_counts",
    )
    embedding_counts = _load_embedding_counts(
        embedding_path,
        valid_stats_tokens=valid_stats_tokens,
    )
    identity = {
        "kind": "qwen4_teacher_dense_capture",
        "name": capture["name"],
        "size_bytes": manifest_path.stat().st_size + dense_size + embedding_size,
        "sha256": _sha256_file(dense_path),
        "content_sha256": expected_content,
        "target_set_sha256": _target_set_sha256(targets),
        "key_count": len(targets),
        "sampling": "per_affine_input_channel_sum2_over_valid_stats_tokens",
        "source": source,
        "capture": capture,
        "valid_stats_tokens": valid_stats_tokens,
    }
    return vectors, embedding_counts, identity


def qwen4_teacher_calibration(
    capture_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict]:
    """Return logical expert importance vectors and the validated capture identity."""
    vectors, _counts, identity = _load_capture(capture_path)
    return vectors, identity


def qwen4_teacher_expert_counts(capture_path: str | Path) -> dict[int, np.ndarray]:
    """Return exact per-layer uint64 routed-expert counts from the teacher capture."""
    _vectors, counts, _identity = _load_capture(capture_path)
    return counts


def qwen4_teacher_dense_calibration(
    capture_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict]:
    """Return exact released affine-input moments and their capture identity."""
    vectors, _embedding_counts, identity = _load_dense_capture(capture_path)
    return vectors, identity


def qwen4_teacher_embedding_counts(capture_path: str | Path) -> dict[int, int]:
    """Return exact input-token row counts from a dense teacher capture."""
    _vectors, counts, _identity = _load_dense_capture(capture_path)
    return counts
