"""Convert released Qwen4 routed experts to calibrated IQ_K cell files.

The CPU quantizer first emits its row-major wire, then the shared relayout
packs those bytes into the headerless ``iqk_relayout`` rows consumed directly
by ``qwen4.iqk_package`` and the decode kernels.  The converter reads one
expert from one layer at a time, derives that expert's steering vector from
the teacher's ordinary route-active moments, and publishes each completed
cell through an atomic rename.  The final inventory is portable and binds the
source checkpoint, allocation, teacher, and surface identities.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import time
from threading import Lock
from typing import Protocol

import numpy as np

from moespresso.inventory.safetensors_header import TensorHeader
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    iqk_geometry,
)
from moespresso.package.iqk_cpu import iqk_codec
from moespresso.package.iqk_relayout import (
    pack_rows,
    relayout_implementation_identity,
)
from moespresso.package.qwen4.iqk_package import (
    CONVERSION_INVENTORY_SCHEMA,
    IQK_MEMBERS,
    PROJECTIONS,
    Qwen4IQKPackageError,
    ZERO_COUNT_MEAN_POLICY,
    _decision_zero_count_policy,
    _source_parts,
    read_qwen4_iqk_allocation,
)
from moespresso.probe.qwen4.calibration import qwen4_teacher_calibration
from moespresso.probe.weight_io import scan_offsets


RUN_CONTRACT_SCHEMA = "qwen4_iqk_conversion_run_v1"
CELL_STATE_SCHEMA = "qwen4_iqk_conversion_cell_state_v1"
LAYER_STATE_SCHEMA = "qwen4_iqk_conversion_layer_state_v1"
INVENTORY_NAME = "inventory.json"
STATE_DIR_NAME = ".qwen4-iqk-convert"
RUN_CONTRACT_NAME = "run-contract.json"
MAX_WORKERS = 8
DEFAULT_WORKERS = 8
DEFAULT_BENCHMARK_EXPERTS = 8
_CPU_CODEC_LOCK = Lock()
_CPU_CODEC_READY = False

_LAYER_COUNT = 48
_EXPERT_COUNT = 512
_HIDDEN_SIZE = 2560
_INTERMEDIATE_SIZE = 640
_PADDED_INTERMEDIATE_SIZE = 768
_GATE_UP_SHAPE = (_EXPERT_COUNT, 2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE)
_DOWN_SHAPE = (_EXPERT_COUNT, _HIDDEN_SIZE, _INTERMEDIATE_SIZE)
_TEACHER_LAYER_ARRAYS = frozenset(
    {
        "gate_up_in_sum2",
        "down_in_sum2",
        "gate_up_count",
        "down_count",
        "route_weight_sum",
        "route_weight_sum2",
    }
)


class Qwen4IQKConversionError(ValueError):
    """The source, calibration, decision, or conversion state is invalid."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Qwen4IQKConversionError(f"could not canonicalize conversion identity: {exc}") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1 << 22), b""):
                digest.update(block)
    except OSError as exc:
        raise Qwen4IQKConversionError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Qwen4IQKConversionError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Qwen4IQKConversionError(f"{field} must be an integer >= {minimum}")
    return value


def _safe_relative(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Qwen4IQKConversionError(f"{field} must be a nonempty relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise Qwen4IQKConversionError(f"{field} is not a safe canonical relative path")
    try:
        resolved_root = root.resolve()
        resolved = (resolved_root / value).resolve()
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise Qwen4IQKConversionError(f"{field} escapes its declared root") from exc
    return resolved


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    created = False
    try:
        with open(temporary, "xb") as destination:
            created = True
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise Qwen4IQKConversionError(f"stale atomic JSON temporary exists: {temporary}") from exc
    finally:
        if created and temporary.exists():
            temporary.unlink()


def _read_json(path: Path, *, field: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Qwen4IQKConversionError(f"could not read {field} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise Qwen4IQKConversionError(f"{field} must be an object")
    return dict(payload)


def _package_version() -> str:
    try:
        return version("mlx-iqk")
    except PackageNotFoundError as exc:
        raise Qwen4IQKConversionError("mlx-iqk is not installed") from exc


@dataclass(frozen=True)
class IQKCellSpec:
    """One headerless converted cell and its exact output geometry."""

    layer_index: int
    projection: str
    codec: str
    logical_shape: tuple[int, int]
    stored_shape: tuple[int, int]
    zero_padding: int
    source_name: str
    surface_cell_identity: str
    surface_run_contract_identity: str
    row_bytes: int
    bytes_per_expert: int
    size_bytes: int
    zero_count_experts: tuple[int, ...] = ()
    zero_count_fallback_policy: str | None = None

    @property
    def name(self) -> str:
        return f"layer{self.layer_index:02d}_{self.projection}.{self.codec}"

    def portable_record(self) -> dict:
        record = {
            "name": self.name,
            "layer_index": self.layer_index,
            "projection": self.projection,
            "codec": self.codec,
            "layout": IQK_LAYOUT_IQK_RELAYOUT,
            "source_name": self.source_name,
            "logical_shape": list(self.logical_shape),
            "stored_shape": list(self.stored_shape),
            "zero_padding": self.zero_padding,
            "row_bytes": self.row_bytes,
            "bytes_per_expert": self.bytes_per_expert,
            "size_bytes": self.size_bytes,
            "surface_cell_identity": self.surface_cell_identity,
            "surface_run_contract_identity": self.surface_run_contract_identity,
        }
        if self.zero_count_experts:
            record["zero_count_experts"] = list(self.zero_count_experts)
            record["zero_count_fallback_policy"] = self.zero_count_fallback_policy
        return record


@dataclass(frozen=True)
class TeacherLayerMoments:
    """Ordinary route-active second moments for one routed layer."""

    layer_index: int
    gate_up_sum2: np.ndarray
    down_sum2: np.ndarray
    counts: np.ndarray
    artifact_sha256: str
    zero_count_experts: tuple[int, ...] = ()
    zero_count_fallback_policy: str | None = None
    zero_count_fallbacks: Mapping[str, np.ndarray] | None = None

    def _active_mean(self, projection: str) -> np.ndarray:
        active = self.counts > 0
        if not np.any(active):
            raise Qwen4IQKConversionError(
                f"layer {self.layer_index} has no route-active calibration rows"
            )
        source = self.gate_up_sum2 if projection in {"gate", "up"} else self.down_sum2
        normalized = (
            np.asarray(source[active], dtype=np.float64)
            / np.asarray(self.counts[active], dtype=np.float64)[:, None]
        )
        vector = np.mean(normalized, axis=0, dtype=np.float64).astype(np.float32)
        if not np.all(np.isfinite(vector)) or np.any(vector < 0) or not np.any(vector > 0):
            raise Qwen4IQKConversionError(
                f"layer {self.layer_index} {projection} has an unusable "
                "active-row mean steering vector"
            )
        return vector

    def steering(self, projection: str, expert: int) -> np.ndarray:
        count = int(self.counts[expert])
        if count <= 0:
            if (
                self.zero_count_fallback_policy != ZERO_COUNT_MEAN_POLICY
                or expert not in self.zero_count_experts
            ):
                raise Qwen4IQKConversionError(
                    f"layer {self.layer_index} expert {expert} has no route-active "
                    "calibration observation"
                )
            if self.zero_count_fallbacks is None:
                raise Qwen4IQKConversionError(
                    f"layer {self.layer_index} zero-count steering was not precomputed"
                )
            return self.zero_count_fallbacks[projection]
        source = self.gate_up_sum2 if projection in {"gate", "up"} else self.down_sum2
        vector = (np.asarray(source[expert], dtype=np.float64) / count).astype(np.float32)
        if not np.all(np.isfinite(vector)) or np.any(vector < 0) or not np.any(vector > 0):
            raise Qwen4IQKConversionError(
                f"layer {self.layer_index} expert {expert} {projection} has an "
                "unusable ordinary sum2/count steering vector"
            )
        return vector

    def precompute_zero_count_fallbacks(self) -> TeacherLayerMoments:
        """Validate and cache every declared fallback vector."""
        actual = tuple(int(expert) for expert in np.flatnonzero(self.counts == 0))
        if actual != self.zero_count_experts:
            raise Qwen4IQKConversionError(
                f"layer {self.layer_index} zero counts for experts {list(actual)} "
                f"do not match the declared set {list(self.zero_count_experts)}"
            )
        if not actual:
            if self.zero_count_fallback_policy is not None:
                raise Qwen4IQKConversionError(
                    f"layer {self.layer_index} declares an unused zero-count policy"
                )
            return self
        if self.zero_count_fallback_policy != ZERO_COUNT_MEAN_POLICY:
            raise Qwen4IQKConversionError(
                f"layer {self.layer_index} zero-count fallback policy drifted"
            )
        gate_up = self._active_mean("gate")
        down = self._active_mean("down")
        gate_up.setflags(write=False)
        down.setflags(write=False)
        return replace(
            self,
            zero_count_fallbacks={"gate": gate_up, "up": gate_up, "down": down},
        )


class ExpertLayerSource(Protocol):
    """One layer's bounded source reader."""

    def read(self, expert: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def close(self) -> None: ...


class IQKEncoder(Protocol):
    """CPU encoder seam used by conversion and synthetic tests."""

    def __call__(
        self,
        member: str,
        weights: np.ndarray,
        steering: np.ndarray,
    ) -> np.ndarray: ...


class IQKRelayout(Protocol):
    """Byte-exact conversion from the CPU quantizer wire to kernel rows."""

    def __call__(
        self,
        member: str,
        wire_rows: np.ndarray,
        in_features: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class PreparedConversion:
    """Validated released inputs and portable conversion contract."""

    model_dir: Path
    teacher_manifest: Path
    cells: tuple[IQKCellSpec, ...]
    source_headers: Mapping[str, TensorHeader]
    source_identity: dict
    allocation_decision_id: str
    teacher_identity: dict
    surface_identity: dict
    zero_count_policy: dict | None
    run_contract: dict
    output_bytes: int
    source_read_bytes: int

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({cell.layer_index for cell in self.cells}))

    def inventory_base(self) -> dict:
        inventory = {
            "schema": CONVERSION_INVENTORY_SCHEMA,
            "source_identity": self.source_identity,
            "allocation_decision_id": self.allocation_decision_id,
            "teacher_identity": self.teacher_identity,
            "surface_identity": self.surface_identity,
            "conversion_run_contract": self.run_contract,
        }
        if self.zero_count_policy is not None:
            inventory["zero_count_policy"] = self.zero_count_policy
        return inventory


def _teacher_manifest_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise Qwen4IQKConversionError(f"teacher manifest is missing: {path}")
    return path.resolve()


def _validate_named_identity(value: object, *, field: str) -> dict:
    if not isinstance(value, Mapping):
        raise Qwen4IQKConversionError(f"{field} must be an object")
    payload = dict(value)
    identity = _digest(payload.get("identity_sha256"), field=f"{field}.identity_sha256")
    body = {key: item for key, item in payload.items() if key != "identity_sha256"}
    actual = _canonical_sha256(body)
    if actual != identity:
        raise Qwen4IQKConversionError(f"{field} identity is {identity}, expected {actual}")
    return payload


def _validate_teacher_binding(
    teacher_manifest: Path,
    decision: Mapping[str, object],
) -> dict:
    teacher = _validate_named_identity(
        decision.get("teacher_identity"), field="decision.teacher_identity"
    )
    _vectors, actual_probe = qwen4_teacher_calibration(teacher_manifest)
    if teacher.get("probe_calibration_identity") != actual_probe:
        raise Qwen4IQKConversionError(
            "decision teacher identity does not match the active calibration capture"
        )
    if teacher.get("manifest_sha256") != _sha256_file(teacher_manifest):
        raise Qwen4IQKConversionError("decision teacher manifest digest drifted")
    evidence = teacher_manifest.parent / "evidence.json"
    if not evidence.is_file() or teacher.get("evidence_sha256") != _sha256_file(evidence):
        raise Qwen4IQKConversionError("decision teacher evidence digest drifted")
    manifest = _read_json(teacher_manifest, field="teacher manifest")
    if teacher.get("layers") != manifest.get("layers"):
        raise Qwen4IQKConversionError("decision teacher layer identities drifted")
    if teacher.get("capture_identity") != manifest.get("capture_identity"):
        raise Qwen4IQKConversionError("decision teacher capture identity drifted")
    if teacher.get("valid_stats_tokens") != manifest.get("valid_stats_tokens"):
        raise Qwen4IQKConversionError("decision teacher token count drifted")
    if teacher.get("geometry") != manifest.get("geometry"):
        raise Qwen4IQKConversionError("decision teacher geometry drifted")
    return teacher


def _validate_surface_binding(
    decision: Mapping[str, object],
    cells: Mapping[tuple[int, str], Mapping[str, object]],
) -> dict:
    surface = decision.get("surface_identity")
    if not isinstance(surface, Mapping):
        raise Qwen4IQKConversionError("decision has no surface identity")
    surface = dict(surface)
    expected_fields = {
        "schema",
        "content_sha256",
        "run_contract_identity",
        "implementation_identity",
        "cell_manifest_sha256",
    }
    if set(surface) != expected_fields:
        raise Qwen4IQKConversionError(
            "decision surface identity fields do not match the conversion contract"
        )
    for field in expected_fields - {"schema"}:
        _digest(surface[field], field=f"decision.surface_identity.{field}")
    if not isinstance(surface["schema"], str) or not surface["schema"]:
        raise Qwen4IQKConversionError("decision surface schema is missing")
    run_identity = surface["run_contract_identity"]
    if {str(cell["surface_run_contract_identity"]) for cell in cells.values()} != {run_identity}:
        raise Qwen4IQKConversionError("allocation cells do not bind the surface run")
    manifest = [
        {
            "layer_index": int(cell["layer_index"]),
            "projection": str(cell["projection"]),
            "surface_cell_identity": str(cell["surface_cell_identity"]),
        }
        for _key, cell in sorted(
            cells.items(), key=lambda item: (item[0][0], PROJECTIONS.index(item[0][1]))
        )
    ]
    if surface["cell_manifest_sha256"] != _canonical_sha256(manifest):
        raise Qwen4IQKConversionError("decision surface cell manifest drifted")
    return surface


def _cell_specs(
    cells: Mapping[tuple[int, str], Mapping[str, object]],
    *,
    zero_count_policy: Mapping[str, object] | None,
) -> tuple[IQKCellSpec, ...]:
    out = []
    for (layer, projection), cell in sorted(
        cells.items(), key=lambda item: (item[0][0], PROJECTIONS.index(item[0][1]))
    ):
        if cell["format"] != "iqk":
            continue
        codec = str(cell["codec"])
        if codec not in IQK_MEMBERS:
            raise Qwen4IQKConversionError(
                f"layer {layer} {projection}: unsupported IQ_K member {codec!r}"
            )
        logical_shape = tuple(int(value) for value in cell["logical_shape"])
        stored_shape = tuple(int(value) for value in cell["stored_shape"])
        row_bytes = iqk_geometry(codec).bytes_per_row(stored_shape[1])
        bytes_per_expert = stored_shape[0] * row_bytes
        out.append(
            IQKCellSpec(
                layer_index=int(layer),
                projection=str(projection),
                codec=codec,
                logical_shape=logical_shape,
                stored_shape=stored_shape,
                zero_padding=int(cell["zero_padding"]),
                source_name=str(cell["source_name"]),
                surface_cell_identity=str(cell["surface_cell_identity"]),
                surface_run_contract_identity=str(cell["surface_run_contract_identity"]),
                row_bytes=row_bytes,
                bytes_per_expert=bytes_per_expert,
                size_bytes=_EXPERT_COUNT * bytes_per_expert,
                zero_count_experts=tuple(
                    int(expert) for expert in cell.get("zero_count_experts", [])
                ),
                zero_count_fallback_policy=(
                    str(cell["zero_count_fallback_policy"])
                    if "zero_count_fallback_policy" in cell
                    else None
                ),
            )
        )
    conservative_expected = {
        (layer, projection) for layer in range(2, _LAYER_COUNT) for projection in PROJECTIONS
    }
    specialized_expected = {
        (layer, projection) for layer in range(_LAYER_COUNT) for projection in PROJECTIONS
    }
    actual = frozenset((cell.layer_index, cell.projection) for cell in out)
    if actual not in {frozenset(conservative_expected), frozenset(specialized_expected)}:
        raise Qwen4IQKConversionError(
            "converted IQ_K cells must be exactly layers 2 through 47 or the "
            "explicit all-48-layer specialization"
        )
    if (actual == frozenset(specialized_expected)) != (zero_count_policy is not None):
        raise Qwen4IQKConversionError(
            "all-48-layer conversion coverage requires the explicit zero-count policy"
        )
    return tuple(out)


def _validate_source_headers(
    headers: Mapping[str, TensorHeader],
    cells: tuple[IQKCellSpec, ...],
) -> None:
    for layer in sorted({cell.layer_index for cell in cells}):
        gate_name = next(
            cell.source_name
            for cell in cells
            if cell.layer_index == layer and cell.projection == "gate"
        )
        down_name = next(
            cell.source_name
            for cell in cells
            if cell.layer_index == layer and cell.projection == "down"
        )
        for name, shape in ((gate_name, _GATE_UP_SHAPE), (down_name, _DOWN_SHAPE)):
            header = headers.get(name)
            if header is None:
                raise Qwen4IQKConversionError(f"source tensor is missing: {name}")
            if header.dtype != "BF16" or tuple(header.shape) != shape:
                raise Qwen4IQKConversionError(
                    f"{name}: source is {header.dtype}{header.shape}, expected BF16{shape}"
                )
            if header.end - header.begin != int(np.prod(shape, dtype=np.int64)) * 2:
                raise Qwen4IQKConversionError(f"{name}: source byte span drifted")


def prepare_qwen4_iqk_conversion(
    model_dir: str | Path,
    *,
    allocation_path: str | Path,
    teacher_capture: str | Path,
    workers: int = DEFAULT_WORKERS,
) -> PreparedConversion:
    """Validate every identity and return the exact conversion work plan."""
    workers = _validate_workers(workers)
    model_dir = Path(model_dir).resolve()
    teacher_manifest = _teacher_manifest_path(teacher_capture)
    try:
        parts = _source_parts(model_dir, teacher_manifest, require_dense=False)
        cells, decision = read_qwen4_iqk_allocation(
            allocation_path,
            inventory=parts["inventory"],
            source_identity=parts["source_identity"],
        )
    except (Qwen4IQKPackageError, OSError, ValueError) as exc:
        raise Qwen4IQKConversionError(str(exc)) from exc
    teacher_identity = _validate_teacher_binding(teacher_manifest, decision)
    if teacher_identity.get("source_identity_sha256") != _canonical_sha256(
        parts["teacher_source_identity"]
    ):
        raise Qwen4IQKConversionError("teacher source identity does not match the snapshot")
    surface_identity = _validate_surface_binding(decision, cells)
    try:
        zero_count_policy = _decision_zero_count_policy(decision)
    except Qwen4IQKPackageError as exc:
        raise Qwen4IQKConversionError(str(exc)) from exc
    specs = _cell_specs(cells, zero_count_policy=zero_count_policy)
    headers = scan_offsets(model_dir)
    _validate_source_headers(headers, specs)
    cell_manifest = [cell.portable_record() for cell in specs]
    run_body = {
        "schema": RUN_CONTRACT_SCHEMA,
        "source_identity_sha256": parts["source_identity"]["snapshot_identity_sha256"],
        "allocation_decision_id": decision["artifact_id"],
        "teacher_identity_sha256": teacher_identity["identity_sha256"],
        "surface_content_sha256": surface_identity["content_sha256"],
        "surface_run_contract_identity": surface_identity["run_contract_identity"],
        "encoder": {
            "package": "mlx-iqk",
            "version": _package_version(),
            "entrypoint": "mlx_iqk.codec.quantize",
            "execution": "cpu",
            "source_layout": "ik_wire",
            "published_layout": IQK_LAYOUT_IQK_RELAYOUT,
        },
        "relayout_implementation": relayout_implementation_identity(),
        "num_experts": _EXPERT_COUNT,
        "worker_policy": {"requested": workers, "maximum": MAX_WORKERS},
        "steering": "ordinary_route_active_per_expert_sum2_div_count",
        **({"zero_count_policy": zero_count_policy} if zero_count_policy is not None else {}),
        "cells": cell_manifest,
    }
    run_contract = {
        **run_body,
        "identity_sha256": _canonical_sha256(run_body),
    }
    layers = len({cell.layer_index for cell in specs})
    source_read_bytes = (
        layers
        * _EXPERT_COUNT
        * (2 * _INTERMEDIATE_SIZE * _HIDDEN_SIZE * 2 + _HIDDEN_SIZE * _INTERMEDIATE_SIZE * 2)
    )
    return PreparedConversion(
        model_dir=model_dir,
        teacher_manifest=teacher_manifest,
        cells=specs,
        source_headers=headers,
        source_identity=dict(parts["source_identity"]),
        allocation_decision_id=str(decision["artifact_id"]),
        teacher_identity=teacher_identity,
        surface_identity=surface_identity,
        zero_count_policy=zero_count_policy,
        run_contract=run_contract,
        output_bytes=sum(cell.size_bytes for cell in specs),
        source_read_bytes=source_read_bytes,
    )


def _validate_workers(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise Qwen4IQKConversionError("workers must be an integer")
    if not 1 <= workers <= MAX_WORKERS:
        raise Qwen4IQKConversionError(f"workers must be in [1, {MAX_WORKERS}]")
    return workers


class TeacherMomentProvider:
    """Stream exact per-expert ordinary moments from a validated teacher."""

    def __init__(self, prepared: PreparedConversion):
        self.root = prepared.teacher_manifest.parent
        self.records = {
            int(record["layer_index"]): dict(record)
            for record in prepared.teacher_identity["layers"]
        }
        self.valid_stats_tokens = int(prepared.teacher_identity["valid_stats_tokens"])
        self.zero_count_policy = prepared.zero_count_policy
        self.expected_zero_experts = {
            layer: tuple(
                int(pair[1])
                for pair in (
                    prepared.zero_count_policy.get("zero_count_pairs", [])
                    if prepared.zero_count_policy is not None
                    else []
                )
                if int(pair[0]) == layer
            )
            for layer in range(_LAYER_COUNT)
        }

    def __call__(self, layer: int) -> TeacherLayerMoments:
        record = self.records.get(int(layer))
        if record is None:
            raise Qwen4IQKConversionError(f"teacher has no layer {layer}")
        path = _safe_relative(
            self.root,
            record.get("path"),
            field=f"teacher.layers[{layer}].path",
        )
        expected_size = _integer(
            record.get("size_bytes"),
            field=f"teacher.layers[{layer}].size_bytes",
            minimum=1,
        )
        expected_hash = _digest(record.get("sha256"), field=f"teacher.layers[{layer}].sha256")
        if not path.is_file() or path.stat().st_size != expected_size:
            raise Qwen4IQKConversionError(f"teacher layer {layer} file drifted")
        if _sha256_file(path) != expected_hash:
            raise Qwen4IQKConversionError(f"teacher layer {layer} digest drifted")
        try:
            with np.load(path, allow_pickle=False) as data:
                if not _TEACHER_LAYER_ARRAYS <= set(data.files):
                    raise Qwen4IQKConversionError(
                        f"teacher layer {layer} lacks ordinary expert moments"
                    )
                gate = np.asarray(data["gate_up_in_sum2"], dtype=np.float64)
                down = np.asarray(data["down_in_sum2"], dtype=np.float64)
                counts = np.asarray(data["gate_up_count"])
                down_counts = np.asarray(data["down_count"])
        except Qwen4IQKConversionError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise Qwen4IQKConversionError(f"could not load teacher layer {layer}: {exc}") from exc
        if gate.shape != (_EXPERT_COUNT, _HIDDEN_SIZE):
            raise Qwen4IQKConversionError(f"teacher layer {layer} gate moments drifted")
        if down.shape != (_EXPERT_COUNT, _INTERMEDIATE_SIZE):
            raise Qwen4IQKConversionError(f"teacher layer {layer} down moments drifted")
        if counts.shape != (_EXPERT_COUNT,) or not np.array_equal(counts, down_counts):
            raise Qwen4IQKConversionError(f"teacher layer {layer} counts drifted")
        if counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise Qwen4IQKConversionError(f"teacher layer {layer} counts are invalid")
        counts = counts.astype(np.uint64, copy=False)
        if sum(int(value) for value in counts) != self.valid_stats_tokens * 10:
            raise Qwen4IQKConversionError(f"teacher layer {layer} route count total drifted")
        zeros = tuple(int(expert) for expert in np.flatnonzero(counts == 0))
        expected_zeros = (
            self.expected_zero_experts[layer] if self.zero_count_policy is not None else ()
        )
        if zeros != expected_zeros:
            raise Qwen4IQKConversionError(
                f"layer {layer} zero-count experts {list(zeros)} do not match "
                f"the declared set {list(expected_zeros)}"
            )
        for name, values in (("gate", gate), ("down", down)):
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise Qwen4IQKConversionError(f"teacher layer {layer} {name} moments are invalid")
        return TeacherLayerMoments(
            layer,
            gate,
            down,
            counts,
            expected_hash,
            zero_count_experts=expected_zeros,
            zero_count_fallback_policy=(ZERO_COUNT_MEAN_POLICY if expected_zeros else None),
        )


def _bf16_bytes_to_float32(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape, dtype=np.int64)) * 2
    if len(raw) != expected:
        raise Qwen4IQKConversionError(
            f"BF16 source read returned {len(raw)} bytes, expected {expected}"
        )
    codes = np.frombuffer(raw, dtype="<u2")
    widened = codes.astype("<u4")
    widened <<= 16
    return widened.view("<f4").reshape(shape)


class ReleasedExpertLayerSource:
    """Read one released BF16 expert pair through positional file reads."""

    def __init__(
        self,
        model_dir: Path,
        gate_up: TensorHeader,
        down: TensorHeader,
    ):
        self.model_dir = model_dir
        self.gate_up = gate_up
        self.down = down
        self._descriptors: dict[str, int] = {}
        try:
            for shard in sorted({gate_up.shard, down.shard}):
                self._descriptors[shard] = os.open(self.model_dir / shard, os.O_RDONLY)
        except OSError:
            self.close()
            raise

    def _fd(self, header: TensorHeader) -> int:
        try:
            return self._descriptors[header.shard]
        except KeyError as exc:
            raise Qwen4IQKConversionError(
                f"source descriptor was not opened before workers: {header.shard}"
            ) from exc

    def _read(self, header: TensorHeader, expert: int, shape: tuple[int, int]) -> np.ndarray:
        byte_count = int(np.prod(shape, dtype=np.int64)) * 2
        offset = header.header_size + header.begin + expert * byte_count
        raw = os.pread(self._fd(header), byte_count, offset)
        if len(raw) != byte_count:
            raise Qwen4IQKConversionError(
                f"{header.name}: short expert {expert} read of {len(raw)} bytes"
            )
        return _bf16_bytes_to_float32(raw, shape)

    def read(self, expert: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 0 <= expert < _EXPERT_COUNT:
            raise Qwen4IQKConversionError(f"expert {expert} is outside [0, {_EXPERT_COUNT})")
        gate_up = self._read(self.gate_up, expert, (2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE))
        down = self._read(self.down, expert, (_HIDDEN_SIZE, _INTERMEDIATE_SIZE))
        return (
            gate_up[:_INTERMEDIATE_SIZE],
            gate_up[_INTERMEDIATE_SIZE:],
            down,
        )

    def close(self) -> None:
        for descriptor in self._descriptors.values():
            os.close(descriptor)
        self._descriptors.clear()


def _released_source_factory(
    prepared: PreparedConversion,
) -> Callable[[int], ReleasedExpertLayerSource]:
    by_key = {(cell.layer_index, cell.projection): cell for cell in prepared.cells}

    def factory(layer: int) -> ReleasedExpertLayerSource:
        gate_name = by_key[(layer, "gate")].source_name
        down_name = by_key[(layer, "down")].source_name
        return ReleasedExpertLayerSource(
            prepared.model_dir,
            prepared.source_headers[gate_name],
            prepared.source_headers[down_name],
        )

    return factory


def cpu_iqk_encoder(
    member: str,
    weights: np.ndarray,
    steering: np.ndarray,
) -> np.ndarray:
    """Encode through the pinned CPU codec without creating MLX arrays."""
    global _CPU_CODEC_READY
    codec = iqk_codec()
    if not _CPU_CODEC_READY:
        with _CPU_CODEC_LOCK:
            if not _CPU_CODEC_READY:
                codec.load()
                _CPU_CODEC_READY = True
    return codec.quantize(member, weights, steering)


def _stored_inputs(
    spec: IQKCellSpec,
    weights: np.ndarray,
    steering: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if tuple(weights.shape) != spec.logical_shape:
        raise Qwen4IQKConversionError(
            f"{spec.name}: source shape {weights.shape} != {spec.logical_shape}"
        )
    if steering.shape != (spec.logical_shape[1],):
        raise Qwen4IQKConversionError(f"{spec.name}: steering shape {steering.shape} is invalid")
    if spec.zero_padding:
        stored_weights = np.pad(weights, ((0, 0), (0, spec.zero_padding)))
        stored_steering = np.pad(steering, (0, spec.zero_padding))
        if np.count_nonzero(stored_weights[:, -spec.zero_padding :]):
            raise AssertionError("Qwen4 down weight padding is not exact zero")
        if np.count_nonzero(stored_steering[-spec.zero_padding :]):
            raise AssertionError("Qwen4 down steering padding is not exact zero")
    else:
        stored_weights = weights
        stored_steering = steering
    if tuple(stored_weights.shape) != spec.stored_shape:
        raise Qwen4IQKConversionError(
            f"{spec.name}: stored shape {stored_weights.shape} != {spec.stored_shape}"
        )
    return (
        np.ascontiguousarray(stored_weights, dtype=np.float32),
        np.ascontiguousarray(stored_steering, dtype=np.float32),
    )


def _encode_expert(
    expert: int,
    specs: tuple[IQKCellSpec, ...],
    source: ExpertLayerSource,
    moments: TeacherLayerMoments,
    encoder: IQKEncoder,
    relayout: IQKRelayout,
) -> tuple[int, dict[str, np.ndarray]]:
    gate, up, down = source.read(expert)
    matrices = {"gate": gate, "up": up, "down": down}
    wires = {}
    for spec in specs:
        steering = moments.steering(spec.projection, expert)
        weights, stored_steering = _stored_inputs(spec, matrices[spec.projection], steering)
        wire = np.asarray(encoder(spec.codec, weights, stored_steering))
        expected_shape = (
            spec.stored_shape[0],
            spec.row_bytes,
        )
        if wire.dtype != np.uint8 or wire.shape != expected_shape:
            raise Qwen4IQKConversionError(
                f"{spec.name}: encoder returned {wire.dtype}{wire.shape}, expected "
                f"uint8[{spec.stored_shape[0]}, {spec.row_bytes}]"
            )
        blocks = np.asarray(relayout(spec.codec, wire, spec.stored_shape[1]))
        if blocks.dtype != np.uint8 or blocks.shape != expected_shape:
            raise Qwen4IQKConversionError(
                f"{spec.name}: relayout returned {blocks.dtype}{blocks.shape}, "
                f"expected uint8[{spec.stored_shape[0]}, {spec.row_bytes}]"
            )
        wires[spec.projection] = np.ascontiguousarray(blocks)
    return expert, wires


def _bounded_ordered_jobs(
    experts: range,
    *,
    workers: int,
    submit: Callable[[int], tuple[int, dict[str, np.ndarray]]],
):
    if workers == 1:
        for expert in experts:
            yield submit(expert)
        return
    pending: deque[Future] = deque()
    iterator = iter(experts)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qwen4-iqk-convert") as pool:
        for _ in range(min(workers * 2, len(experts))):
            pending.append(pool.submit(submit, next(iterator)))
        for expert in iterator:
            yield pending.popleft().result()
            pending.append(pool.submit(submit, expert))
        while pending:
            yield pending.popleft().result()


def _state_paths(out_dir: Path, spec: IQKCellSpec) -> tuple[Path, Path, Path]:
    state = out_dir / STATE_DIR_NAME
    temporary = state / "staging" / f"{spec.name}.partial"
    descriptor = state / "cells" / f"layer{spec.layer_index:02d}-{spec.projection}.json"
    final = out_dir / spec.name
    return temporary, descriptor, final


def _validate_file_record(spec: IQKCellSpec, record: Mapping[str, object]) -> dict:
    expected = spec.portable_record()
    for field, value in expected.items():
        if record.get(field) != value:
            raise Qwen4IQKConversionError(f"{spec.name}: conversion record {field} drifted")
    digest = _digest(record.get("sha256"), field=f"{spec.name}.sha256")
    return {**expected, "sha256": digest}


def _validate_completed_path(path: Path, record: Mapping[str, object]) -> None:
    if path.is_symlink() or not path.is_file():
        raise Qwen4IQKConversionError(f"converted artifact is not regular: {path}")
    expected_size = int(record["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise Qwen4IQKConversionError(f"{path.name}: size {actual_size} != {expected_size}")
    actual_hash = _sha256_file(path)
    if actual_hash != record["sha256"]:
        raise Qwen4IQKConversionError(f"{path.name}: sha256 {actual_hash} != {record['sha256']}")


def _recover_cell(
    out_dir: Path,
    spec: IQKCellSpec,
    run_identity: str,
) -> dict | None:
    temporary, descriptor_path, final = _state_paths(out_dir, spec)
    if final.exists() and not descriptor_path.exists():
        raise Qwen4IQKConversionError(f"{final}: published file has no conversion descriptor")
    if not descriptor_path.exists():
        return None
    descriptor = _read_json(descriptor_path, field="cell state")
    if descriptor.get("schema") != CELL_STATE_SCHEMA:
        raise Qwen4IQKConversionError(f"{descriptor_path}: state schema drifted")
    if descriptor.get("run_contract_identity") != run_identity:
        raise Qwen4IQKConversionError(f"{descriptor_path}: run identity drifted")
    record = descriptor.get("file")
    if not isinstance(record, Mapping):
        raise Qwen4IQKConversionError(f"{descriptor_path}: file record is missing")
    record = _validate_file_record(spec, record)
    if final.exists():
        if temporary.exists():
            raise Qwen4IQKConversionError(f"{spec.name}: both published and staged payloads exist")
        _validate_completed_path(final, record)
        return record
    if not temporary.exists():
        raise Qwen4IQKConversionError(
            f"{spec.name}: prepared state has neither staged nor published payload"
        )
    _validate_completed_path(temporary, record)
    os.replace(temporary, final)
    _fsync_directory(out_dir)
    return record


def _publish_cell(
    out_dir: Path,
    spec: IQKCellSpec,
    run_identity: str,
    digest: str,
) -> dict:
    temporary, descriptor_path, final = _state_paths(out_dir, spec)
    record = {**spec.portable_record(), "sha256": digest}
    _validate_completed_path(temporary, record)
    _atomic_json(
        descriptor_path,
        {
            "schema": CELL_STATE_SCHEMA,
            "run_contract_identity": run_identity,
            "file": record,
        },
    )
    if final.exists():
        raise Qwen4IQKConversionError(
            f"refusing to replace pre-existing converted artifact {final}"
        )
    os.replace(temporary, final)
    _fsync_directory(out_dir)
    return record


def _initialize_state(out_dir: Path, run_contract: Mapping[str, object]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.is_symlink() or not out_dir.is_dir():
        raise Qwen4IQKConversionError(f"conversion output is not a directory: {out_dir}")
    state = out_dir / STATE_DIR_NAME
    contract_path = state / RUN_CONTRACT_NAME
    run_identity = _digest(
        run_contract.get("identity_sha256"), field="run_contract.identity_sha256"
    )
    body = {key: value for key, value in run_contract.items() if key != "identity_sha256"}
    if _canonical_sha256(body) != run_identity:
        raise Qwen4IQKConversionError("conversion run contract identity does not verify")
    if contract_path.exists():
        if _read_json(contract_path, field="run contract") != dict(run_contract):
            raise Qwen4IQKConversionError("conversion output belongs to another run contract")
        return run_identity
    if state.exists():
        if state.is_symlink() or not state.is_dir():
            raise Qwen4IQKConversionError("conversion state exists without a valid run contract")
        if any(state.iterdir()):
            raise Qwen4IQKConversionError("conversion state is nonempty but has no run contract")
    existing = [path for path in out_dir.iterdir() if path.name != STATE_DIR_NAME]
    if existing:
        raise Qwen4IQKConversionError("conversion output is nonempty but has no run contract")
    (state / "staging").mkdir(parents=True, exist_ok=True)
    (state / "cells").mkdir(parents=True, exist_ok=True)
    (state / "layers").mkdir(parents=True, exist_ok=True)
    _atomic_json(contract_path, dict(run_contract))
    return run_identity


_REUSE_RUN_CONTRACT_FIELDS = (
    "schema",
    "source_identity_sha256",
    "teacher_identity_sha256",
    "encoder",
    "relayout_implementation",
    "num_experts",
    "steering",
    "zero_count_policy",
)
_REUSE_CELL_PROVENANCE_FIELDS = frozenset(
    {"surface_cell_identity", "surface_run_contract_identity"}
)


def _same_optional_field(
    left: Mapping[str, object],
    right: Mapping[str, object],
    field: str,
) -> bool:
    return (field in left) == (field in right) and left.get(field) == right.get(field)


def _remaining_storage_report(
    out_dir: Path,
    cells: Sequence[IQKCellSpec],
) -> dict:
    probe = out_dir if out_dir.exists() else out_dir.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    available = int(shutil.disk_usage(probe).free)
    remaining = sum(cell.size_bytes for cell in cells)
    largest_layer = max(
        (
            sum(cell.size_bytes for cell in cells if cell.layer_index == layer)
            for layer in {cell.layer_index for cell in cells}
        ),
        default=0,
    )
    reserve = 1 << 30
    required = remaining + largest_layer + reserve
    return {
        "available_bytes": available,
        "required_bytes": required,
        "remaining_output_bytes": remaining,
        "largest_atomic_layer_bytes": largest_layer,
        "reserve_bytes": reserve,
        "sufficient": available >= required,
    }


def seed_qwen4_iqk_conversion_cache(
    out_dir: str | Path,
    *,
    cells: tuple[IQKCellSpec, ...],
    run_contract: Mapping[str, object],
    inventory_base: Mapping[str, object],
    reuse_from: Sequence[str | Path],
) -> dict:
    """Hardlink byte-identical cells from compatible completed conversions."""
    if isinstance(reuse_from, (str, bytes, os.PathLike)):
        raise Qwen4IQKConversionError("reuse_from must be a sequence of directories")
    target_by_name = {cell.name: cell for cell in cells}
    if len(target_by_name) != len(cells):
        raise Qwen4IQKConversionError("target conversion cell names are not unique")
    target_contract = inventory_base.get("conversion_run_contract")
    if not isinstance(target_contract, Mapping):
        raise Qwen4IQKConversionError("target conversion inventory has no run contract")
    if dict(target_contract) != dict(run_contract):
        raise Qwen4IQKConversionError("target conversion inventory and run contract differ")
    expected_inventory_fields = (
        "schema",
        "source_identity",
        "teacher_identity",
        "zero_count_policy",
    )
    target_surface = inventory_base.get("surface_identity")
    if not isinstance(target_surface, Mapping):
        raise Qwen4IQKConversionError("target conversion inventory has no surface identity")
    target_surface_content = _digest(
        target_surface.get("content_sha256"),
        field="target surface content_sha256",
    )
    if target_surface_content != target_contract.get("surface_content_sha256"):
        raise Qwen4IQKConversionError("target surface and conversion run content identities differ")
    if target_surface.get("run_contract_identity") != target_contract.get(
        "surface_run_contract_identity"
    ):
        raise Qwen4IQKConversionError("target surface and conversion run identities differ")
    out_dir = Path(out_dir)
    resolved_out = out_dir.resolve()
    donor_records: dict[str, tuple[Path, dict]] = {}
    donor_inventories = []
    for raw_donor in reuse_from:
        donor = Path(raw_donor)
        if donor.resolve() == resolved_out:
            raise Qwen4IQKConversionError(
                "conversion cache cannot reuse from its own output directory"
            )
        if donor.is_symlink() or not donor.is_dir():
            raise Qwen4IQKConversionError(
                f"conversion cache donor is not a regular directory: {donor}"
            )
        inventory_path = donor / INVENTORY_NAME
        if inventory_path.is_symlink() or not inventory_path.is_file():
            raise Qwen4IQKConversionError(
                f"conversion cache donor has no regular inventory: {inventory_path}"
            )
        inventory = _read_json(inventory_path, field="reuse conversion inventory")
        for field in expected_inventory_fields:
            if not _same_optional_field(inventory, inventory_base, field):
                raise Qwen4IQKConversionError(
                    f"conversion cache donor {field} is incompatible: {donor}"
                )
        donor_contract = inventory.get("conversion_run_contract")
        if not isinstance(donor_contract, Mapping):
            raise Qwen4IQKConversionError(f"conversion cache donor has no run contract: {donor}")
        donor_surface = inventory.get("surface_identity")
        if not isinstance(donor_surface, Mapping):
            raise Qwen4IQKConversionError(
                f"conversion cache donor has no surface identity: {donor}"
            )
        donor_surface_content = _digest(
            donor_surface.get("content_sha256"),
            field="donor surface content_sha256",
        )
        if donor_surface_content != donor_contract.get("surface_content_sha256"):
            raise Qwen4IQKConversionError(
                f"conversion cache donor surface content contract differs: {donor}"
            )
        if donor_surface.get("run_contract_identity") != donor_contract.get(
            "surface_run_contract_identity"
        ):
            raise Qwen4IQKConversionError(
                f"conversion cache donor surface run contract differs: {donor}"
            )
        # Allocation surfaces choose a codec but do not enter its encoder.
        # Byte reuse is instead bound to the source, teacher, steering,
        # encoder, relayout, zero-count policy, and portable cell geometry.
        for field in ("schema",):
            if field in donor_surface or field in target_surface:
                if not _same_optional_field(donor_surface, target_surface, field):
                    raise Qwen4IQKConversionError(
                        f"conversion cache donor surface {field} is incompatible: {donor}"
                    )
        for field in _REUSE_RUN_CONTRACT_FIELDS:
            if not _same_optional_field(donor_contract, target_contract, field):
                raise Qwen4IQKConversionError(
                    f"conversion cache donor run contract {field} is incompatible: {donor}"
                )
        records = inventory.get("files")
        if not isinstance(records, list):
            raise Qwen4IQKConversionError(
                f"conversion cache donor file inventory is missing: {donor}"
            )
        names = [record.get("name") for record in records if isinstance(record, Mapping)]
        if len(names) != len(records) or len(set(names)) != len(names):
            raise Qwen4IQKConversionError(f"conversion cache donor file names are invalid: {donor}")
        matched = 0
        for record in records:
            name = record["name"]
            spec = target_by_name.get(name)
            if spec is None:
                continue
            portable = spec.portable_record()
            if any(
                record.get(field) != value
                for field, value in portable.items()
                if field not in _REUSE_CELL_PROVENANCE_FIELDS
            ):
                continue
            donor_run_identity = _digest(
                record.get("surface_run_contract_identity"),
                field=f"{name}.surface_run_contract_identity",
            )
            if donor_run_identity != donor_contract.get("surface_run_contract_identity"):
                raise Qwen4IQKConversionError(
                    f"conversion cache donor cell surface run differs: {name}"
                )
            _digest(
                record.get("surface_cell_identity"),
                field=f"{name}.surface_cell_identity",
            )
            digest = _digest(record.get("sha256"), field=f"{name}.sha256")
            validated = {**portable, "sha256": digest}
            source_path = donor / name
            _validate_completed_path(source_path, record)
            previous = donor_records.get(name)
            if previous is not None and previous[1]["sha256"] != digest:
                raise Qwen4IQKConversionError(f"conversion cache donors disagree on {name}")
            donor_records[name] = (source_path, validated)
            matched += 1
        donor_inventories.append(
            {
                "path": str(inventory_path.resolve()),
                "sha256": _sha256_file(inventory_path),
                "matched_cells": matched,
            }
        )

    run_identity = _initialize_state(out_dir, run_contract)
    reused = []
    existing = []
    for spec in cells:
        recovered = _recover_cell(out_dir, spec, run_identity)
        if recovered is not None:
            existing.append(spec.name)
            continue
        donor_record = donor_records.get(spec.name)
        if donor_record is None:
            continue
        source_path, record = donor_record
        temporary, _descriptor, _final = _state_paths(out_dir, spec)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, temporary)
            _publish_cell(out_dir, spec, run_identity, record["sha256"])
        except OSError as exc:
            raise Qwen4IQKConversionError(
                f"could not hardlink reusable conversion cell {spec.name}: {exc}"
            ) from exc
        finally:
            if temporary.exists():
                temporary.unlink()
        reused.append(spec.name)

    present = set(existing) | set(reused)
    missing_cells = tuple(cell for cell in cells if cell.name not in present)
    return {
        "schema": "qwen4_iqk_conversion_cache_seed_v1",
        "status": "seeded",
        "run_contract_identity": run_identity,
        "donor_inventories": donor_inventories,
        "existing_cells": existing,
        "reused_cells": reused,
        "reused_bytes": sum(target_by_name[name].size_bytes for name in reused),
        "missing_cells": [cell.name for cell in missing_cells],
        "storage": _remaining_storage_report(out_dir, missing_cells),
    }


def _prepare_layer_moments(
    specs: tuple[IQKCellSpec, ...],
    moments: TeacherLayerMoments,
    *,
    num_experts: int,
) -> TeacherLayerMoments:
    layer = specs[0].layer_index
    if moments.layer_index != layer or moments.counts.shape != (num_experts,):
        raise Qwen4IQKConversionError(f"layer {layer} moment geometry drifted")
    prepared = moments.precompute_zero_count_fallbacks()
    if prepared.zero_count_experts and any(
        spec.zero_count_experts != prepared.zero_count_experts
        or spec.zero_count_fallback_policy != ZERO_COUNT_MEAN_POLICY
        for spec in specs
    ):
        raise Qwen4IQKConversionError(f"layer {layer} zero-count fallback contract drifted")
    if not prepared.zero_count_experts and any(
        spec.zero_count_experts or spec.zero_count_fallback_policy is not None for spec in specs
    ):
        raise Qwen4IQKConversionError(f"layer {layer} declares unused zero-count cell fields")
    return prepared


def _write_layer(
    out_dir: Path,
    specs: tuple[IQKCellSpec, ...],
    *,
    num_experts: int,
    source: ExpertLayerSource,
    moments: TeacherLayerMoments,
    encoder: IQKEncoder,
    relayout: IQKRelayout,
    workers: int,
    run_identity: str,
) -> tuple[list[dict], dict]:
    layer = specs[0].layer_index
    layer_state_path = out_dir / STATE_DIR_NAME / "layers" / f"layer{layer:02d}.json"
    if layer_state_path.exists():
        layer_state = _read_json(layer_state_path, field="layer state")
        if (
            layer_state.get("schema") != LAYER_STATE_SCHEMA
            or layer_state.get("run_contract_identity") != run_identity
            or layer_state.get("layer_index") != layer
            or layer_state.get("status") != "complete"
        ):
            raise Qwen4IQKConversionError(f"layer {layer} completed state drifted")
        recorded = layer_state.get("files")
        if not isinstance(recorded, list) or len(recorded) != len(specs):
            raise Qwen4IQKConversionError(f"layer {layer} completed file coverage drifted")
        by_projection = {row.get("projection"): row for row in recorded if isinstance(row, Mapping)}
        if set(by_projection) != set(PROJECTIONS):
            raise Qwen4IQKConversionError(f"layer {layer} completed projection coverage drifted")
        completed_records = []
        for spec in specs:
            expected = _validate_file_record(spec, by_projection[spec.projection])
            resumed = _recover_cell(out_dir, spec, run_identity)
            if resumed != expected:
                raise Qwen4IQKConversionError(
                    f"layer {layer} {spec.projection} completed state drifted"
                )
            completed_records.append(resumed)
        return completed_records, layer_state
    completed = {}
    for spec in specs:
        record = _recover_cell(out_dir, spec, run_identity)
        if record is not None:
            completed[spec.projection] = record
    missing = tuple(spec for spec in specs if spec.projection not in completed)
    if missing:
        handles = {}
        digests = {}
        try:
            for spec in missing:
                temporary, _descriptor, _final = _state_paths(out_dir, spec)
                temporary.parent.mkdir(parents=True, exist_ok=True)
                handles[spec.projection] = open(temporary, "wb", buffering=1 << 20)
                digests[spec.projection] = hashlib.sha256()
            started = time.perf_counter()

            def submit(expert: int):
                return _encode_expert(expert, missing, source, moments, encoder, relayout)

            next_expert = 0
            for expert, wires in _bounded_ordered_jobs(
                range(num_experts), workers=workers, submit=submit
            ):
                if expert != next_expert:
                    raise AssertionError("bounded IQ_K workers changed expert order")
                next_expert += 1
                for spec in missing:
                    payload = wires[spec.projection].tobytes(order="C")
                    handles[spec.projection].write(payload)
                    digests[spec.projection].update(payload)
            elapsed = time.perf_counter() - started
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            handles.clear()
            _fsync_directory(out_dir / STATE_DIR_NAME / "staging")
            for spec in missing:
                completed[spec.projection] = _publish_cell(
                    out_dir,
                    spec,
                    run_identity,
                    digests[spec.projection].hexdigest(),
                )
        finally:
            for handle in handles.values():
                handle.close()
    else:
        elapsed = 0.0
    records = [completed[projection] for projection in PROJECTIONS]
    layer_state = {
        "schema": LAYER_STATE_SCHEMA,
        "run_contract_identity": run_identity,
        "layer_index": layer,
        "status": "complete",
        "elapsed_seconds": elapsed,
        "files": records,
    }
    _atomic_json(layer_state_path, layer_state)
    return records, layer_state


def convert_qwen4_iqk_artifacts(
    out_dir: str | Path,
    *,
    cells: tuple[IQKCellSpec, ...],
    run_contract: Mapping[str, object],
    inventory_base: Mapping[str, object],
    moment_provider: Callable[[int], TeacherLayerMoments],
    source_factory: Callable[[int], ExpertLayerSource],
    encoder: IQKEncoder,
    relayout: IQKRelayout = pack_rows,
    workers: int,
    num_experts: int = _EXPERT_COUNT,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run or exactly resume a bounded conversion and publish its inventory."""
    workers = _validate_workers(workers)
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 1:
        raise Qwen4IQKConversionError("num_experts must be a positive integer")
    by_layer: dict[int, tuple[IQKCellSpec, ...]] = {}
    grouped: dict[int, list[IQKCellSpec]] = {}
    for cell in cells:
        grouped.setdefault(cell.layer_index, []).append(cell)
    for layer in sorted(grouped):
        layer_cells = tuple(
            sorted(grouped[layer], key=lambda cell: PROJECTIONS.index(cell.projection))
        )
        if tuple(cell.projection for cell in layer_cells) != PROJECTIONS:
            raise Qwen4IQKConversionError(f"layer {layer} does not cover gate/up/down")
        by_layer[layer] = layer_cells
    out_dir = Path(out_dir)
    run_identity = _initialize_state(out_dir, run_contract)
    inventory_path = out_dir / INVENTORY_NAME
    if inventory_path.exists():
        payload = _read_json(inventory_path, field="conversion inventory")
        expected_base = dict(inventory_base)
        for field, value in expected_base.items():
            if payload.get(field) != value:
                raise Qwen4IQKConversionError(f"existing conversion inventory {field} drifted")
        records = payload.get("files")
        if not isinstance(records, list) or len(records) != len(cells):
            raise Qwen4IQKConversionError("existing conversion inventory coverage drifted")
        by_name = {record.get("name"): record for record in records if isinstance(record, Mapping)}
        if set(by_name) != {cell.name for cell in cells}:
            raise Qwen4IQKConversionError("existing conversion inventory names drifted")
        for cell in cells:
            record = _validate_file_record(cell, by_name[cell.name])
            _validate_completed_path(out_dir / cell.name, record)
        return payload

    records = []
    layer_reports = []
    started = time.perf_counter()
    for layer in sorted(by_layer):
        layer_cells = by_layer[layer]
        moments = _prepare_layer_moments(
            layer_cells,
            moment_provider(layer),
            num_experts=num_experts,
        )
        source = source_factory(layer)
        try:
            layer_records, report = _write_layer(
                out_dir,
                layer_cells,
                num_experts=num_experts,
                source=source,
                moments=moments,
                encoder=encoder,
                relayout=relayout,
                workers=workers,
                run_identity=run_identity,
            )
        finally:
            source.close()
        del moments
        records.extend(layer_records)
        layer_reports.append(report)
        if progress is not None:
            progress(
                {
                    "layer": layer,
                    "completed_cells": len(records),
                    "total_cells": len(cells),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    records.sort(key=lambda row: (int(row["layer_index"]), PROJECTIONS.index(row["projection"])))
    payload = {
        **dict(inventory_base),
        "files": records,
        "summary": {
            "status": "complete",
            "cells": len(records),
            "layers": len(by_layer),
            "num_experts": num_experts,
            "member_counts": dict(sorted(Counter(row["codec"] for row in records).items())),
            "output_bytes": sum(int(row["size_bytes"]) for row in records),
            "elapsed_seconds": time.perf_counter() - started,
            "layer_state_count": len(layer_reports),
            **(
                {
                    "zero_count_fallback_pairs": [
                        [cell.layer_index, expert]
                        for cell in cells
                        if cell.projection == "gate"
                        for expert in cell.zero_count_experts
                    ],
                    "zero_count_fallback_policy": ZERO_COUNT_MEAN_POLICY,
                }
                if any(cell.zero_count_experts for cell in cells)
                else {}
            ),
        },
    }
    _atomic_json(inventory_path, payload)
    return payload


def _benchmark_prepared(
    prepared: PreparedConversion,
    *,
    workers: int,
    experts: int,
    encoder: IQKEncoder,
    relayout: IQKRelayout,
) -> dict:
    if isinstance(experts, bool) or not 1 <= experts <= _EXPERT_COUNT:
        raise Qwen4IQKConversionError(f"benchmark_experts must be in [1, {_EXPERT_COUNT}]")
    by_layer: dict[int, tuple[IQKCellSpec, ...]] = {}
    for layer in prepared.layers:
        by_layer[layer] = tuple(cell for cell in prepared.cells if cell.layer_index == layer)
    signatures: dict[tuple[str, str, str], list[int]] = {}
    for layer, cells in by_layer.items():
        signature = tuple(cell.codec for cell in cells)
        signatures.setdefault(signature, []).append(layer)
    moments_provider = TeacherMomentProvider(prepared)
    source_factory = _released_source_factory(prepared)
    reports = []
    estimated_seconds = 0.0
    for signature, layers in sorted(signatures.items()):
        layer = layers[0]
        moments = _prepare_layer_moments(
            by_layer[layer],
            moments_provider(layer),
            num_experts=_EXPERT_COUNT,
        )
        source = source_factory(layer)
        try:
            started = time.perf_counter()

            def submit(expert: int):
                return _encode_expert(expert, by_layer[layer], source, moments, encoder, relayout)

            completed = 0
            for _expert, _wires in _bounded_ordered_jobs(
                range(experts), workers=workers, submit=submit
            ):
                completed += 1
            elapsed = time.perf_counter() - started
        finally:
            source.close()
        if completed != experts or elapsed <= 0:
            raise Qwen4IQKConversionError("conversion benchmark did not complete")
        projected = elapsed * (_EXPERT_COUNT / experts) * len(layers)
        estimated_seconds += projected
        reports.append(
            {
                "codec_signature": list(signature),
                "representative_layer": layer,
                "layers": layers,
                "experts": experts,
                "workers": workers,
                "elapsed_seconds": elapsed,
                "expert_jobs_per_second": experts / elapsed,
                "projected_seconds": projected,
            }
        )
    return {
        "method": "released representative layer per distinct codec signature",
        "scope": "CPU encode and source reads; excludes final whole-file hashing and thermal drift",
        "metal_arrays_created": False,
        "signatures": reports,
        "estimated_full_seconds": estimated_seconds,
        "estimated_full_hours": estimated_seconds / 3600.0,
    }


def preflight_qwen4_iqk_conversion(
    prepared: PreparedConversion,
    *,
    out_dir: str | Path,
    workers: int,
    benchmark_experts: int = DEFAULT_BENCHMARK_EXPERTS,
    encoder: IQKEncoder = cpu_iqk_encoder,
    relayout: IQKRelayout = pack_rows,
) -> dict:
    """Return exact bytes and a bounded released throughput projection."""
    workers = _validate_workers(workers)
    benchmark = _benchmark_prepared(
        prepared,
        workers=workers,
        experts=benchmark_experts,
        encoder=encoder,
        relayout=relayout,
    )
    out_dir = Path(out_dir)
    probe = out_dir if out_dir.exists() else out_dir.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    available = int(shutil.disk_usage(probe).free)
    largest_layer = max(
        sum(cell.size_bytes for cell in prepared.cells if cell.layer_index == layer)
        for layer in prepared.layers
    )
    required = prepared.output_bytes + largest_layer + (1 << 30)
    return {
        "status": "preflight",
        "source_identity_sha256": prepared.source_identity["snapshot_identity_sha256"],
        "allocation_decision_id": prepared.allocation_decision_id,
        "teacher_identity_sha256": prepared.teacher_identity["identity_sha256"],
        "surface_content_sha256": prepared.surface_identity["content_sha256"],
        "conversion_run_contract_identity": prepared.run_contract["identity_sha256"],
        "cells": len(prepared.cells),
        "layers": len(prepared.layers),
        "num_experts": _EXPERT_COUNT,
        "member_counts": dict(sorted(Counter(cell.codec for cell in prepared.cells).items())),
        "output_bytes_exact": prepared.output_bytes,
        "output_gib": prepared.output_bytes / (1 << 30),
        "source_read_bytes_exact": prepared.source_read_bytes,
        "source_read_gib": prepared.source_read_bytes / (1 << 30),
        "workers": workers,
        "benchmark": benchmark,
        "storage": {
            "available_bytes": available,
            "required_bytes": required,
            "largest_atomic_layer_bytes": largest_layer,
            "reserve_bytes": 1 << 30,
            "sufficient": available >= required,
        },
        "not_run": [
            "full IQ_K conversion",
            "package assembly",
            "runtime correctness and quality gates",
        ],
    }


def build_qwen4_iqk_converted_artifacts(
    prepared: PreparedConversion,
    out_dir: str | Path,
    *,
    workers: int = DEFAULT_WORKERS,
    encoder: IQKEncoder = cpu_iqk_encoder,
    relayout: IQKRelayout = pack_rows,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Convert the complete released eligible routed stack or resume it."""
    return convert_qwen4_iqk_artifacts(
        out_dir,
        cells=prepared.cells,
        run_contract=prepared.run_contract,
        inventory_base=prepared.inventory_base(),
        moment_provider=TeacherMomentProvider(prepared),
        source_factory=_released_source_factory(prepared),
        encoder=encoder,
        relayout=relayout,
        workers=workers,
        progress=progress,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert released Qwen4 routed experts to calibrated IQ_K cells"
    )
    parser.add_argument("model_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--teacher-capture", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--benchmark-experts", type=int, default=DEFAULT_BENCHMARK_EXPERTS)
    parser.add_argument(
        "--reuse-from",
        action="append",
        default=[],
        help="hardlink compatible completed cells from a prior conversion cache",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write the full converted artifact set after preflight",
    )
    args = parser.parse_args(argv)
    try:
        prepared = prepare_qwen4_iqk_conversion(
            args.model_dir,
            allocation_path=args.allocation,
            teacher_capture=args.teacher_capture,
            workers=args.workers,
        )
        preflight = preflight_qwen4_iqk_conversion(
            prepared,
            out_dir=args.out_dir,
            workers=args.workers,
            benchmark_experts=args.benchmark_experts,
        )
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        if not args.execute:
            return 0
        reuse = None
        if args.reuse_from:
            reuse = seed_qwen4_iqk_conversion_cache(
                args.out_dir,
                cells=prepared.cells,
                run_contract=prepared.run_contract,
                inventory_base=prepared.inventory_base(),
                reuse_from=args.reuse_from,
            )
            print(json.dumps(reuse, indent=2, sort_keys=True), flush=True)
        storage = preflight["storage"] if reuse is None else reuse["storage"]
        if not storage["sufficient"]:
            raise Qwen4IQKConversionError("conversion output volume has insufficient free space")

        def progress(record: dict) -> None:
            print(json.dumps(record, sort_keys=True), flush=True)

        inventory = build_qwen4_iqk_converted_artifacts(
            prepared,
            args.out_dir,
            workers=args.workers,
            progress=progress,
        )
        print(json.dumps(inventory["summary"], indent=2, sort_keys=True), flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
