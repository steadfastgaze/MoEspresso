"""Build the released Qwen4 text package from calibrated IQ_K expert cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import shutil

import numpy as np

from moespresso.core.artifact import ArtifactError, read_artifact, write_artifact
from moespresso.inventory.architecture_profile import family_of
from moespresso.inventory.build import build_inventory_from_headers
from moespresso.inventory.qwen4.static import (
    validate_qwen38_flash_next_headers,
    validate_qwen38_flash_next_static,
)
from moespresso.inventory.safetensors_header import scan_headers
from moespresso.package.constants import MANIFEST_NAME
from moespresso.package.source import INVENTORY_NAME, _layer_types, _read_config
from moespresso.package.iqk_artifacts import IQKConvertedArtifacts
from moespresso.package.iqk_format import (
    IQK_DENSE_MEMBERS,
    IQK_GEOMETRY,
    IQK_LAYOUT_IK_WIRE,
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
    iqk_dense_geometry,
)
from moespresso.package.iqk_dense_relayout import (
    dense_relayout_implementation_identity,
    pack_dense_rows,
    unpack_dense_rows,
)
from moespresso.package.iqk_relayout import relayout_implementation_identity
from moespresso.package.iqk_recipe import (
    build_iqk_expert_allocations,
    build_iqk_package_plan,
)
from moespresso.package.kquant_backend import check_kquant_backend_available
from moespresso.package.kquant_cache import KQuantEncodeCache
from moespresso.package.kquant_format import (
    IMATRIX_STEERED_CODECS,
    KQUANT_GEOMETRY,
)
from moespresso.package.kquant_recipe import validate_kquant_target_fit
from moespresso.package.manifest import file_identity
from moespresso.package.plan import parse_force_overrides
from moespresso.package.qwen.recipe import (
    QwenKQuantDenseTarget,
    QwenKQuantExpertTarget,
    build_expert_kquant_allocations,
)
from moespresso.package.qwen4.ple_provider import (
    inspect_qwen4_ple_reuse,
    inspect_qwen4_ple_source,
    write_qwen4_ple_provider,
)
from moespresso.package.qwen4.iqk_reap import (
    EXPERT_SELECTION_NAME,
    SELECTION_FEATURE,
    selection_manifest_block,
    validate_expert_selection,
)
from moespresso.package.tokenizer import copy_tokenizer_into_package
from moespresso.inventory.qwen4.source_identity import (
    qwen4_hf_snapshot_source_identity,
)
from moespresso.package.write import write_package
from moespresso.probe.qwen4.calibration import (
    qwen4_teacher_dense_calibration,
    qwen4_teacher_expert_counts,
)
from moespresso.runtime.qwen4.ple_contract import (
    derive_qwen4_ple_provider_contract,
)


PACKAGE_PLAN_NAME = "package_plan.json"
IQK_REPORT_NAME = "qwen4_iqk_package_report.json"
CONVERSION_INVENTORY_SCHEMA = "qwen4_iqk_converted_artifacts_v1"
QWEN4_FAMILY = "qwen4_exp"
PROJECTIONS = ("gate", "up", "down")
IQK_MEMBERS = frozenset({"iq1_s_r4", "iq2_k", "iq2_ks", "iq3_k"})
PADDED_DOWN_MEMBERS = frozenset({"iq2_k", "iq2_ks", "iq3_k"})
FALLBACK_CODEC = "q8_0"
FALLBACK_LAYERS = frozenset({0, 1})
FALLBACK_POLICY = "unobserved_cell_q8_0"
CALIBRATED_IQK_POLICY = "per_expert_route_active"
ZERO_COUNT_SPECIALIZATION_MODE = "all_iq2_k_active_mean_zero"
ZERO_COUNT_EARLY_IQ3_MODE = "early_iq3_k_active_mean_zero"
ZERO_COUNT_JOINT_MODE = "joint_iqk_active_mean_zero"
ZERO_COUNT_MEAN_POLICY = "same_layer_projection_mean_normalized_route_active_v1"
EARLY_IQ3_ZERO_COUNT_PAIRS = (
    (0, 181),
    (0, 193),
    (0, 236),
    (0, 244),
    (0, 271),
    (0, 413),
    (0, 424),
    (0, 477),
    (1, 116),
)
_ROUTER_GATE_ROLE = "moe.router_gate"

_LAYER_COUNT = 48
_EXPERT_COUNT = 512
_HIDDEN_SIZE = 2560
_INTERMEDIATE_SIZE = 640
_PADDED_INTERMEDIATE_SIZE = 768
DIRECT_CODEC_CHOICES = frozenset({"q4_k", "q5_k", "q6_k", *IQK_DENSE_MEMBERS})
_EXPECTED_DIRECT_BYTES = {
    "q4_k": 3_393_291_520,
    "q5_k": 3_874_545_920,
    "q6_k": 4_385_878_720,
    "iq4_ks": 3_277_928_384,
    "iq4_k": 3_393_291_520,
    "iq5_k": 3_874_545_920,
    "iq6_k": 4_415_957_120,
}
_EXPECTED_PLE_BYTES = 102_400_491_520


class Qwen4IQKPackageError(ValueError):
    """The Qwen4 source, allocation, calibration or encoded cells are invalid."""


def _read_expert_selection(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    try:
        selection = read_artifact(path)
        validate_expert_selection(selection)
    except (ArtifactError, OSError, ValueError) as exc:
        raise Qwen4IQKPackageError(f"could not read Qwen expert selection: {exc}") from exc
    return selection


def zero_count_mean_policy_contract(
    zero_count_pairs: list[list[int]] | tuple[tuple[int, int], ...],
) -> dict:
    """Return a zero-count steering specialization contract."""
    return {
        "schema": "qwen4_iqk_zero_count_policy_v1",
        "name": ZERO_COUNT_MEAN_POLICY,
        "zero_count_pairs": [list(pair) for pair in zero_count_pairs],
        "normalization": "per_expert_sum2_div_route_active_count",
        "reduction": "unweighted_arithmetic_mean_over_positive_count_experts",
        "projection_sources": {
            "gate": "gate_up_in_sum2",
            "up": "gate_up_in_sum2",
            "down": "down_in_sum2",
        },
    }


def _decision_zero_count_policy(decision: Mapping[str, object]) -> dict | None:
    raw = decision.get("zero_count_policy")
    mode = decision.get("mode")
    specialized = mode in {
        ZERO_COUNT_SPECIALIZATION_MODE,
        ZERO_COUNT_EARLY_IQ3_MODE,
        ZERO_COUNT_JOINT_MODE,
    }
    if raw is None:
        if specialized:
            raise Qwen4IQKPackageError(
                "IQ_K zero-count specialization has no declared policy"
            )
        return None
    if not specialized:
        raise Qwen4IQKPackageError(
            "zero-count specialization policy requires its explicit decision mode"
        )
    if not isinstance(raw, Mapping):
        raise Qwen4IQKPackageError("zero-count specialization policy drifted")
    raw = dict(raw)
    pairs = raw.get("zero_count_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise Qwen4IQKPackageError(
            "zero-count specialization must declare a nonempty pair set"
        )
    normalized = []
    for index, pair in enumerate(pairs):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pair)
            or not 0 <= pair[0] < _LAYER_COUNT
            or not 0 <= pair[1] < _EXPERT_COUNT
        ):
            raise Qwen4IQKPackageError(
                f"zero-count specialization pair {index} is invalid"
            )
        normalized.append((pair[0], pair[1]))
    if normalized != sorted(set(normalized)):
        raise Qwen4IQKPackageError(
            "zero-count specialization pairs must be sorted and unique"
        )
    if raw != zero_count_mean_policy_contract(tuple(normalized)):
        raise Qwen4IQKPackageError("zero-count specialization policy drifted")
    if mode == ZERO_COUNT_JOINT_MODE:
        _decision_expert_selection_id(decision, required=True)
    elif mode == ZERO_COUNT_EARLY_IQ3_MODE:
        if tuple(normalized) != EARLY_IQ3_ZERO_COUNT_PAIRS:
            raise Qwen4IQKPackageError(
                "early-IQ3_K specialization zero-count pairs drifted"
            )
        if _decision_expert_selection_id(decision) is not None:
            raise Qwen4IQKPackageError(
                "early-IQ3_K specialization must not bind an expert selection"
            )
    return raw


def _decision_expert_selection_id(
    decision: Mapping[str, object],
    *,
    required: bool = False,
) -> str | None:
    value = decision.get("expert_selection_artifact_id")
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or not value.startswith("select:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise Qwen4IQKPackageError(
            "joint IQ_K allocation has no valid expert selection artifact id"
        )
    return value


def _expected_zero_experts(policy: Mapping[str, object], layer: int) -> list[int]:
    return [
        int(pair[1])
        for pair in policy["zero_count_pairs"]
        if int(pair[0]) == layer
    ]


def _validate_decision_expert_selection(
    decision: Mapping[str, object],
    expert_selection: Mapping[str, object] | None,
    retained: Mapping[int, tuple[int, ...]] | None,
    zero_count_policy: Mapping[str, object] | None,
) -> None:
    if decision.get("mode") == ZERO_COUNT_EARLY_IQ3_MODE and (
        expert_selection is not None or retained is not None
    ):
        raise Qwen4IQKPackageError(
            "early-IQ3_K specialization must retain all 512 experts"
        )
    required_selection_id = _decision_expert_selection_id(decision)
    if required_selection_id is not None:
        if expert_selection is None:
            raise Qwen4IQKPackageError(
                "joint IQ_K allocation requires its bound expert selection"
            )
        if expert_selection.get("artifact_id") != required_selection_id:
            raise Qwen4IQKPackageError(
                "joint IQ_K allocation binds another expert selection"
            )
    if retained is None or zero_count_policy is None:
        return
    leaked = [
        (int(layer), int(expert))
        for layer, expert in zero_count_policy["zero_count_pairs"]
        if int(expert) in retained[int(layer)]
    ]
    if leaked:
        raise Qwen4IQKPackageError(
            f"compact selection retains unobserved calibration experts {leaked}"
        )


def _blocking_messages(artifact: Mapping[str, object]) -> str:
    blocking = [
        item
        for item in artifact.get("validation", [])
        if isinstance(item, Mapping) and item.get("blocking")
    ]
    return "; ".join(f"{item.get('code')}: {item.get('message')}" for item in blocking[:6])


def _source_name(layer: int, projection: str) -> str:
    suffix = "gate_up_proj" if projection in {"gate", "up"} else "down_proj"
    return f"model.language_model.layers.{layer}.mlp.experts.{suffix}"


def _logical_shape(projection: str) -> tuple[int, int]:
    if projection in {"gate", "up"}:
        return _INTERMEDIATE_SIZE, _HIDDEN_SIZE
    return _HIDDEN_SIZE, _INTERMEDIATE_SIZE


def _stored_shape(projection: str, codec: str) -> tuple[int, int]:
    logical = _logical_shape(projection)
    if projection == "down" and codec in PADDED_DOWN_MEMBERS:
        return logical[0], _PADDED_INTERMEDIATE_SIZE
    return logical


def _module_path(layer: int, projection: str) -> str:
    return f"layers.{layer}.mlp.experts.{projection}_proj"


def _shape(value: object, *, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise Qwen4IQKPackageError(f"{field} must be [out_features, in_features]")
    return int(value[0]), int(value[1])


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Qwen4IQKPackageError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _expert_entries(inventory: Mapping[str, object]) -> dict[tuple[int, str], dict]:
    entries: dict[tuple[int, str], dict] = {}
    for raw in inventory.get("tensors", []):
        if not isinstance(raw, Mapping) or raw.get("kind") != "expert":
            continue
        layer = raw.get("layer_index")
        source_projection = raw.get("projection")
        if layer is None or source_projection not in {"gate_up", "down"}:
            raise Qwen4IQKPackageError("Qwen4 inventory has an invalid expert entry")
        key = int(layer), str(source_projection)
        if key in entries:
            raise Qwen4IQKPackageError(f"Qwen4 inventory duplicates layer {key[0]} {key[1]}")
        entries[key] = dict(raw)
    expected = {
        (layer, projection) for layer in range(_LAYER_COUNT) for projection in ("gate_up", "down")
    }
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise Qwen4IQKPackageError(
            f"Qwen4 expert inventory does not cover the released text stack "
            f"(missing={missing[:6]} extra={extra[:6]})"
        )
    for (layer, projection), entry in entries.items():
        expected_shape = (
            (_EXPERT_COUNT, 2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE)
            if projection == "gate_up"
            else (_EXPERT_COUNT, _HIDDEN_SIZE, _INTERMEDIATE_SIZE)
        )
        if tuple(entry.get("shape", ())) != expected_shape:
            raise Qwen4IQKPackageError(
                f"{entry.get('source_name')}: shape {entry.get('shape')} != {list(expected_shape)}"
            )
        if entry.get("source_name") != _source_name(
            layer, "gate" if projection == "gate_up" else "down"
        ):
            raise Qwen4IQKPackageError("Qwen4 expert inventory source identity drifted")
    return entries


def read_qwen4_iqk_allocation(
    path: str | Path,
    *,
    inventory: Mapping[str, object],
    source_identity: Mapping[str, object],
) -> tuple[dict[tuple[int, str], dict], dict]:
    """Read and validate one exact 144-cell optimizer decision."""
    try:
        decision = read_artifact(path)
    except (ArtifactError, OSError, json.JSONDecodeError) as exc:
        raise Qwen4IQKPackageError(f"could not read IQ_K allocation: {exc}") from exc
    if decision.get("artifact_kind") != "optimizer_decision":
        raise Qwen4IQKPackageError("IQ_K allocation must be an optimizer_decision")
    if decision.get("status") != "valid":
        raise Qwen4IQKPackageError("IQ_K allocation decision is not valid")
    if "calibration" not in decision.get("required_features", []):
        raise Qwen4IQKPackageError("IQ_K allocation does not require calibration")
    if not isinstance(decision.get("source_probe_id"), str) or not decision["source_probe_id"]:
        raise Qwen4IQKPackageError("IQ_K allocation has no source_probe_id")
    if decision.get("source_identity") != dict(source_identity):
        raise Qwen4IQKPackageError("IQ_K allocation source identity drifted")
    teacher_source = source_identity.get("teacher_source_identity")
    if not isinstance(teacher_source, Mapping):
        raise Qwen4IQKPackageError("IQ_K source identity has no teacher identity")
    subject = decision.get("subject")
    if (
        not isinstance(subject, Mapping)
        or subject.get("source_root") != teacher_source.get("model_id")
        or subject.get("source_format") != "hf_safetensors"
    ):
        raise Qwen4IQKPackageError("IQ_K allocation subject does not identify the source")
    zero_count_policy = _decision_zero_count_policy(decision)

    expert_entries = _expert_entries(inventory)
    rows = decision.get("allocation")
    if not isinstance(rows, list):
        raise Qwen4IQKPackageError("IQ_K allocation must contain an allocation array")
    cells: dict[tuple[int, str], dict] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise Qwen4IQKPackageError(f"allocation cell {index} is not an object")
        if raw.get("kind") != "expert":
            raise Qwen4IQKPackageError(f"allocation cell {index} must be an expert")
        layer = raw.get("layer_index")
        projection = raw.get("projection")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 0 <= layer < _LAYER_COUNT
            or projection not in PROJECTIONS
        ):
            raise Qwen4IQKPackageError(f"allocation cell {index} has invalid layer/projection")
        key = int(layer), str(projection)
        if key in cells:
            raise Qwen4IQKPackageError(f"IQ_K allocation duplicates layer {layer} {projection}")
        codec = raw.get("codec")
        if codec not in IQK_MEMBERS | {FALLBACK_CODEC}:
            raise Qwen4IQKPackageError(
                f"layer {layer} {projection}: unsupported expert codec {codec!r}"
            )
        expected_format = "iqk" if codec in IQK_MEMBERS else "kquant"
        if raw.get("format", expected_format) != expected_format:
            raise Qwen4IQKPackageError(
                f"layer {layer} {projection}: codec {codec} requires format {expected_format}"
            )
        if expected_format == "iqk":
            if raw.get("layout") != IQK_LAYOUT_IQK_RELAYOUT:
                raise Qwen4IQKPackageError(
                    f"layer {layer} {projection}: IQ_K layout must be {IQK_LAYOUT_IQK_RELAYOUT}"
                )
        elif raw.get("layout") is not None:
            raise Qwen4IQKPackageError(
                f"layer {layer} {projection}: K-quant cell must not declare IQ_K layout"
            )
        source_projection = "gate_up" if projection in {"gate", "up"} else "down"
        source_entry = expert_entries[(layer, source_projection)]
        if raw.get("source_name") != source_entry["source_name"]:
            raise Qwen4IQKPackageError(
                f"layer {layer} {projection}: source tensor does not match inventory"
            )
        logical = _logical_shape(projection)
        stored = _stored_shape(projection, str(codec))
        if _shape(raw.get("logical_shape"), field=f"cell {index}.logical_shape") != logical:
            raise Qwen4IQKPackageError(
                f"layer {layer} {projection}: logical shape must be {list(logical)}"
            )
        if _shape(raw.get("stored_shape"), field=f"cell {index}.stored_shape") != stored:
            raise Qwen4IQKPackageError(
                f"layer {layer} {projection}: stored shape must be {list(stored)}"
            )
        padding = stored[1] - logical[1]
        if raw.get("zero_padding") != padding:
            raise Qwen4IQKPackageError(
                f"layer {layer} {projection}: zero_padding must be {padding}"
            )
        policy = raw.get("calibration_policy")
        if zero_count_policy is None:
            if layer in FALLBACK_LAYERS:
                if codec != FALLBACK_CODEC or policy != FALLBACK_POLICY:
                    raise Qwen4IQKPackageError(
                        f"layer {layer} {projection}: calibration holes require the "
                        "explicit whole-cell Q8_0 fallback"
                    )
                if "zero_count_experts" in raw or "zero_count_fallback_policy" in raw:
                    raise Qwen4IQKPackageError(
                        f"layer {layer} {projection}: conservative fallback must not "
                        "declare an IQ_K zero-count specialization"
                    )
            elif codec in IQK_MEMBERS and policy != CALIBRATED_IQK_POLICY:
                raise Qwen4IQKPackageError(
                    f"layer {layer} {projection}: IQ_K needs per-expert route-active calibration"
                )
            elif not isinstance(policy, str) or not policy:
                raise Qwen4IQKPackageError(
                    f"layer {layer} {projection}: calibration policy is missing"
                )
            if layer not in FALLBACK_LAYERS and (
                "zero_count_experts" in raw or "zero_count_fallback_policy" in raw
            ):
                raise Qwen4IQKPackageError(
                    f"layer {layer} {projection}: unexpected zero-count specialization fields"
                )
        else:
            expected_zero_experts = _expected_zero_experts(zero_count_policy, layer)
            mode = decision.get("mode")
            if mode == ZERO_COUNT_SPECIALIZATION_MODE:
                codec_allowed = codec == "iq2_k"
            elif mode == ZERO_COUNT_EARLY_IQ3_MODE:
                codec_allowed = codec == ("iq3_k" if layer < 2 else "iq2_k")
            else:
                codec_allowed = codec in IQK_MEMBERS
            if not codec_allowed or policy != CALIBRATED_IQK_POLICY:
                specialization = (
                    "early-IQ3_K " if mode == ZERO_COUNT_EARLY_IQ3_MODE else ""
                )
                raise Qwen4IQKPackageError(
                    f"layer {layer} {projection}: {specialization}IQ_K "
                    "zero-count policy drifted"
                )
            if expected_zero_experts and (
                raw.get("zero_count_experts") != expected_zero_experts
                or raw.get("zero_count_fallback_policy") != ZERO_COUNT_MEAN_POLICY
            ):
                raise Qwen4IQKPackageError(
                    f"layer {layer} {projection}: all-IQ2_K zero-count policy drifted"
                )
            if not expected_zero_experts and (
                "zero_count_experts" in raw or "zero_count_fallback_policy" in raw
            ):
                raise Qwen4IQKPackageError(
                    f"layer {layer} {projection}: unexpected zero-count specialization fields"
                )
        cell_identity = _digest(
            raw.get("surface_cell_identity"),
            field=f"cell {index}.surface_cell_identity",
        )
        run_identity = _digest(
            raw.get("surface_run_contract_identity"),
            field=f"cell {index}.surface_run_contract_identity",
        )
        cells[key] = {
            **dict(raw),
            "layer_index": layer,
            "projection": projection,
            "codec": codec,
            "format": expected_format,
            "source_projection": source_projection,
            "logical_shape": list(logical),
            "stored_shape": list(stored),
            "zero_padding": padding,
            "surface_cell_identity": cell_identity,
            "surface_run_contract_identity": run_identity,
        }

    expected_cells = {
        (layer, projection) for layer in range(_LAYER_COUNT) for projection in PROJECTIONS
    }
    if set(cells) != expected_cells:
        missing = sorted(expected_cells - set(cells))
        extra = sorted(set(cells) - expected_cells)
        raise Qwen4IQKPackageError(
            f"IQ_K allocation must cover exactly 144 cells "
            f"(missing={missing[:6]} extra={extra[:6]})"
        )
    run_identities = {str(cell["surface_run_contract_identity"]) for cell in cells.values()}
    if len(run_identities) != 1:
        raise Qwen4IQKPackageError("IQ_K allocation mixes surface run identities")
    cell_identities = [str(cell["surface_cell_identity"]) for cell in cells.values()]
    if len(cell_identities) != len(set(cell_identities)):
        raise Qwen4IQKPackageError("IQ_K allocation duplicates surface cell identities")
    surface = decision.get("surface_identity")
    if surface is not None:
        if not isinstance(surface, Mapping):
            raise Qwen4IQKPackageError(
                "IQ_K allocation surface identity is not an object"
            )
        expected_surface_fields = {
            "schema",
            "content_sha256",
            "run_contract_identity",
            "implementation_identity",
            "cell_manifest_sha256",
        }
        if set(surface) != expected_surface_fields:
            raise Qwen4IQKPackageError(
                "IQ_K allocation surface identity fields do not match the package contract"
            )
        if not isinstance(surface["schema"], str) or not surface["schema"]:
            raise Qwen4IQKPackageError("IQ_K allocation surface schema is missing")
        for field in expected_surface_fields - {"schema"}:
            _digest(surface[field], field=f"surface_identity.{field}")
        if run_identities != {surface["run_contract_identity"]}:
            raise Qwen4IQKPackageError(
                "IQ_K allocation cells do not bind the surface run"
            )
        manifest = [
            {
                "layer_index": int(cell["layer_index"]),
                "projection": str(cell["projection"]),
                "surface_cell_identity": str(cell["surface_cell_identity"]),
            }
            for _key, cell in sorted(
                cells.items(),
                key=lambda item: (item[0][0], PROJECTIONS.index(item[0][1])),
            )
        ]
        encoded_manifest = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        manifest_sha256 = hashlib.sha256(encoded_manifest).hexdigest()
        if surface["cell_manifest_sha256"] != manifest_sha256:
            raise Qwen4IQKPackageError(
                "IQ_K allocation surface cell manifest drifted"
            )
    for layer in range(_LAYER_COUNT):
        formats = {cells[(layer, projection)]["format"] for projection in PROJECTIONS}
        if len(formats) != 1:
            raise Qwen4IQKPackageError(
                f"layer {layer} mixes expert codec families {sorted(formats)}"
            )
    if zero_count_policy is not None:
        codecs = {str(cell["codec"]) for cell in cells.values()}
        if decision.get("mode") == ZERO_COUNT_SPECIALIZATION_MODE:
            if codecs != {"iq2_k"}:
                raise Qwen4IQKPackageError(
                    "all-IQ2_K zero-count specialization must allocate IQ2_K "
                    "to all 144 cells"
                )
        elif decision.get("mode") == ZERO_COUNT_EARLY_IQ3_MODE:
            mismatched = [
                (layer, projection)
                for (layer, projection), cell in cells.items()
                if cell["codec"] != ("iq3_k" if layer < 2 else "iq2_k")
            ]
            if mismatched:
                raise Qwen4IQKPackageError(
                    "early-IQ3_K zero-count specialization must allocate IQ3_K "
                    "to the six layer 0/1 cells and IQ2_K to the other 138 cells"
                )
        elif not codecs or not codecs <= IQK_MEMBERS:
            raise Qwen4IQKPackageError(
                "joint IQ_K zero-count specialization must use IQ_K for all 144 cells"
            )
    _decision_expert_selection_id(decision)
    return cells, decision


_EXPERT_OUTPUT_LAYOUTS = (
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
)


def _expert_allocations(
    cells: Mapping[tuple[int, str], Mapping[str, object]],
    *,
    output_layout: str = IQK_LAYOUT_IQK_RELAYOUT,
) -> list[dict]:
    if output_layout not in _EXPERT_OUTPUT_LAYOUTS:
        raise Qwen4IQKPackageError(
            f"unsupported Qwen4 expert output layout {output_layout!r}; "
            f"expected one of {list(_EXPERT_OUTPUT_LAYOUTS)}"
        )
    iqk_members: dict[int, dict[str, str]] = {}
    for layer in range(_LAYER_COUNT):
        layer_cells = [cells[(layer, projection)] for projection in PROJECTIONS]
        if all(cell["format"] == "iqk" for cell in layer_cells):
            iqk_members[layer] = {
                projection: str(cells[(layer, projection)]["codec"]) for projection in PROJECTIONS
            }

    def iqk_fields(layer: int, projection: str) -> dict:
        cell = cells[(layer, projection)]
        module = _module_path(layer, projection)
        fields = {
            "source_name": cell["source_name"],
            "source_projection": cell["source_projection"],
            "module_path": module,
            "module_weight_key": f"{module}.weight",
            "logical_shape": list(cell["logical_shape"]),
            "stored_shape": list(cell["stored_shape"]),
            "zero_padding": int(cell["zero_padding"]),
            "calibration_policy": cell["calibration_policy"],
        }
        if "zero_count_experts" in cell:
            fields["zero_count_experts"] = list(cell["zero_count_experts"])
            fields["zero_count_fallback_policy"] = cell[
                "zero_count_fallback_policy"
            ]
        fields["surface_cell_identity"] = cell["surface_cell_identity"]
        fields["surface_run_contract_identity"] = cell["surface_run_contract_identity"]
        return fields

    allocations = build_iqk_expert_allocations(
        iqk_members,
        target_fields=iqk_fields,
        layout=output_layout,
    )

    q8_targets = []
    for layer in range(_LAYER_COUNT):
        for projection in PROJECTIONS:
            cell = cells[(layer, projection)]
            if cell["format"] != "kquant":
                continue
            module = _module_path(layer, projection)
            calibration_key = f"qwen4.layers.{layer}.moe.expert.{projection}"
            q8_targets.append(
                QwenKQuantExpertTarget(
                    layer_index=layer,
                    projection=projection,
                    codec=FALLBACK_CODEC,
                    gguf_tensor=calibration_key,
                    imatrix_key=calibration_key,
                    source_name=str(cell["source_name"]),
                    source_projection=str(cell["source_projection"]),
                    module_path=module,
                    module_weight_key=f"{module}.weight",
                )
            )
    q8_allocations = build_expert_kquant_allocations(q8_targets) if q8_targets else []
    for allocation in q8_allocations:
        cell = cells[(int(allocation["layer_index"]), str(allocation["projection"]))]
        allocation.update(
            {
                "logical_shape": list(cell["logical_shape"]),
                "stored_shape": list(cell["stored_shape"]),
                "zero_padding": int(cell["zero_padding"]),
                "calibration_policy": cell["calibration_policy"],
            }
        )
        allocation["surface_cell_identity"] = cell["surface_cell_identity"]
        allocation["surface_run_contract_identity"] = cell["surface_run_contract_identity"]
    allocations.extend(q8_allocations)
    return sorted(
        allocations,
        key=lambda row: (int(row["layer_index"]), PROJECTIONS.index(row["projection"])),
    )


def build_qwen4_direct_allocations(
    inventory: Mapping[str, object],
    dense_vectors: Mapping[str, np.ndarray],
    *,
    direct_codec: str = "q6_k",
) -> tuple[list[dict], list[dict], dict]:
    """Build the deterministic direct policy with lossless router gates.

    K-quant and IQ_K dense members share the same role boundary. The
    embedding and widths outside the 256-value IQ_K block grid stay q8_0;
    the remaining affine tensors take the selected direct member. IQ_K rows
    are emitted directly on the serving relayout, so the package never
    labels the quantizer's wire as kernel-ready bytes.
    """
    if direct_codec not in DIRECT_CODEC_CHOICES:
        raise Qwen4IQKPackageError(
            f"direct_codec must be one of {sorted(DIRECT_CODEC_CHOICES)}, got {direct_codec!r}"
        )
    affine_entries = [
        dict(entry)
        for entry in inventory.get("tensors", [])
        if isinstance(entry, Mapping) and entry.get("kind") == "affine"
    ]
    router_entries = [entry for entry in affine_entries if entry.get("role") == _ROUTER_GATE_ROLE]
    quantized_entries = [
        entry for entry in affine_entries if entry.get("role") != _ROUTER_GATE_ROLE
    ]
    passthrough_entries = [
        dict(entry)
        for entry in inventory.get("tensors", [])
        if isinstance(entry, Mapping) and entry.get("kind") == "passthrough"
    ]
    non_embedding = {
        str(entry["source_name"]) for entry in affine_entries if entry.get("role") != "embed_tokens"
    }
    if set(dense_vectors) != non_embedding:
        raise Qwen4IQKPackageError(
            "dense calibration target set does not match the source inventory"
        )

    allocations: list[dict] = []
    counts: dict[str, int] = {}
    quantized_bytes = 0
    kquant_weight_bytes = 0
    for entry in quantized_entries:
        shape = tuple(int(value) for value in entry.get("shape", ()))
        if len(shape) != 2:
            raise Qwen4IQKPackageError(
                f"{entry.get('source_name')}: affine tensor must be two-dimensional"
            )
        rows, in_features = shape
        codec = "q8_0" if entry.get("role") == "embed_tokens" or in_features % 256 else direct_codec
        module_path = entry.get("module_path")
        module_weight_key = entry.get("module_weight_key")
        if not isinstance(module_path, str) or not isinstance(module_weight_key, str):
            raise Qwen4IQKPackageError(f"{entry['source_name']}: inventory has no module ownership")
        if codec in IQK_DENSE_MEMBERS:
            geometry = iqk_dense_geometry(codec)
            geometry.blocks_per_row(in_features)
            allocations.append(
                {
                    "source_name": str(entry["source_name"]),
                    "kind": "affine",
                    "role": str(entry["role"]),
                    "layer_index": (
                        None if entry.get("layer_index") is None else int(entry["layer_index"])
                    ),
                    "bits": geometry.bits,
                    "format": "iqk",
                    "codec": codec,
                    "iqk_codec": codec,
                    "layout": IQK_LAYOUT_IQK_RELAYOUT,
                    "gguf_tensor": str(entry["source_name"]),
                    "imatrix_key": str(entry["source_name"]),
                    "module_path": module_path,
                    "module_weight_key": module_weight_key,
                }
            )
            quantized_bytes += rows * geometry.bytes_per_row(in_features)
        else:
            geometry = KQUANT_GEOMETRY[codec]
            if in_features % geometry.weights_per_block:
                raise Qwen4IQKPackageError(
                    f"{entry['source_name']}: width {in_features} does not fit {codec}"
                )
            target = QwenKQuantDenseTarget(
                source_name=str(entry["source_name"]),
                role=str(entry["role"]),
                layer_index=(
                    None if entry.get("layer_index") is None else int(entry["layer_index"])
                ),
                codec=codec,
                gguf_tensor=str(entry["source_name"]),
                imatrix_key=str(entry["source_name"]),
                module_path=module_path,
                module_weight_key=module_weight_key,
                requires_imatrix=codec in IMATRIX_STEERED_CODECS,
            )
            validate_kquant_target_fit(target, shape, dict(dense_vectors))
            allocations.append(
                {
                    "source_name": target.source_name,
                    "kind": "affine",
                    "role": target.role,
                    "layer_index": target.layer_index,
                    "bits": geometry.bits,
                    "group_size": geometry.group_size,
                    "format": "kquant",
                    "codec": codec,
                    "kquant_codec": codec,
                    "gguf_tensor": target.gguf_tensor,
                    "imatrix_key": target.imatrix_key,
                    "module_path": target.module_path,
                    "module_weight_key": target.module_weight_key,
                    "requires_imatrix": target.requires_imatrix,
                }
            )
            encoded_bytes = (
                rows * (in_features // geometry.weights_per_block) * geometry.bytes_per_block
            )
            quantized_bytes += encoded_bytes
            kquant_weight_bytes += encoded_bytes
        counts[codec] = counts.get(codec, 0) + 1

    structural_bytes = 0
    passthrough = []
    for entry in [*passthrough_entries, *router_entries]:
        if entry.get("dtype") != "BF16":
            raise Qwen4IQKPackageError(
                f"{entry.get('source_name')}: passthrough tensor must remain BF16"
            )
        shape = tuple(int(value) for value in entry.get("shape", ()))
        elements = 1
        for value in shape:
            elements *= value
        structural_bytes += elements * 2
        passthrough.append(
            {
                **entry,
                "format": "raw_dtype_passthrough",
            }
        )
    counts["bf16"] = len(passthrough)
    counts = dict(sorted(counts.items()))
    direct_bytes = quantized_bytes + structural_bytes
    expected_counts = dict(sorted({"bf16": 341, direct_codec: 580, "q8_0": 146}.items()))
    if counts != expected_counts or direct_bytes != _EXPECTED_DIRECT_BYTES[direct_codec]:
        raise Qwen4IQKPackageError(
            f"released direct policy accounting drifted: counts={counts} bytes={direct_bytes}"
        )
    return (
        sorted(allocations, key=lambda row: row["source_name"]),
        sorted(passthrough, key=lambda row: row["source_name"]),
        {
            "policy_codec": direct_codec,
            "format_counts": counts,
            "quantized_weight_bytes": quantized_bytes,
            "kquant_weight_bytes": kquant_weight_bytes,
            "structural_bf16_bytes": structural_bytes,
            "weight_bytes": direct_bytes,
            "kquant_placeholder_bytes": sum(
                allocation["format"] == "kquant" for allocation in allocations
            ),
        },
    )


def encode_qwen4_dense_iqk(
    matrix: np.ndarray,
    target,
    imatrix: np.ndarray,
) -> np.ndarray:
    """Encode one direct IQ_K tensor onto the serving relayout.

    The linked codec owns the quantization objective. This adapter only
    performs the byte-preserving wire transform already used by the Qwen4
    routed converter and checks every stored byte through the inverse.
    """
    from mlx_iqk import codec as iqk_codec

    wire = iqk_codec.quantize(target.codec, matrix, imatrix)
    packed = pack_dense_rows(target.codec, wire, int(matrix.shape[1]))
    restored = unpack_dense_rows(target.codec, packed, int(matrix.shape[1]))
    if not np.array_equal(restored, wire):
        raise Qwen4IQKPackageError(f"{target.source_name}: dense IQ_K relayout does not round trip")
    return packed


def price_qwen4_expert_allocation(
    cells: Mapping[tuple[int, str], Mapping[str, object]],
    *,
    num_experts: int | Mapping[int, int] = _EXPERT_COUNT,
) -> dict:
    """Return exact encoded weight and auxiliary bytes for all 144 cells."""
    if isinstance(num_experts, Mapping):
        if any(
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            for layer, count in num_experts.items()
        ):
            raise Qwen4IQKPackageError(
                "per-layer expert counts must map integer layers to integer counts"
            )
        per_layer = dict(num_experts)
        if set(per_layer) != set(range(_LAYER_COUNT)):
            raise Qwen4IQKPackageError("per-layer expert counts must cover layers 0 through 47")
    else:
        if isinstance(num_experts, bool) or not 0 < num_experts <= _EXPERT_COUNT:
            raise Qwen4IQKPackageError(f"num_experts must be in [1, {_EXPERT_COUNT}]")
        per_layer = {layer: int(num_experts) for layer in range(_LAYER_COUNT)}
    if any(not 0 < count <= _EXPERT_COUNT for count in per_layer.values()):
        raise Qwen4IQKPackageError(f"num_experts must be in [1, {_EXPERT_COUNT}]")
    counts: dict[str, int] = {}
    weight_bytes = 0
    auxiliary_bytes = 0
    by_cell = []
    for layer, projection in sorted(cells):
        cell = cells[(layer, projection)]
        layer_experts = per_layer[layer]
        codec = str(cell["codec"])
        out_features, in_features = tuple(int(value) for value in cell["stored_shape"])
        if codec in IQK_MEMBERS:
            row_bytes = IQK_GEOMETRY[codec].bytes_per_row(in_features)
            aux = 0
        else:
            geometry = KQUANT_GEOMETRY[codec]
            if in_features % geometry.weights_per_block:
                raise Qwen4IQKPackageError(
                    f"layer {layer} {projection}: width {in_features} does not fit {codec}"
                )
            row_bytes = (in_features // geometry.weights_per_block) * geometry.bytes_per_block
            aux = layer_experts
        cell_bytes = layer_experts * out_features * row_bytes
        counts[codec] = counts.get(codec, 0) + 1
        weight_bytes += cell_bytes
        auxiliary_bytes += aux
        by_cell.append(
            {
                "layer_index": layer,
                "projection": projection,
                "codec": codec,
                "logical_shape": list(cell["logical_shape"]),
                "stored_shape": list(cell["stored_shape"]),
                "zero_padding": int(cell["zero_padding"]),
                "row_bytes": row_bytes,
                "weight_bytes": cell_bytes,
                "auxiliary_bytes": aux,
            }
        )
    return {
        "cell_count": len(by_cell),
        "num_experts": (
            next(iter(set(per_layer.values())))
            if len(set(per_layer.values())) == 1
            else None
        ),
        "per_layer_num_experts": {
            str(layer): count for layer, count in sorted(per_layer.items())
        },
        "codec_counts": dict(sorted(counts.items())),
        "weight_bytes": weight_bytes,
        "bundle_auxiliary_bytes": auxiliary_bytes,
        "stored_payload_bytes": weight_bytes + auxiliary_bytes,
        "cells": by_cell,
    }


def _storage_guard(
    *,
    pricing: Mapping[str, object],
    expert_price: Mapping[str, object],
    direct_allocations: list[dict],
    passthrough: list[dict],
    inventory: Mapping[str, object],
    ple_contract,
    shard_size_gb: float,
    include_encode_cache: bool,
) -> dict:
    """Conservative same-volume free-space guard from exact payload sizes."""
    if not np.isfinite(shard_size_gb) or shard_size_gb <= 0:
        raise Qwen4IQKPackageError("Qwen4 package shards require a positive size cap")
    shard_cap = int(shard_size_gb * (1024**3))
    by_name = {
        str(entry["source_name"]): entry
        for entry in inventory.get("tensors", [])
        if isinstance(entry, Mapping) and entry.get("source_name") is not None
    }
    largest_direct_group = 0
    for allocation in direct_allocations:
        entry = by_name[allocation["source_name"]]
        rows, in_features = (int(value) for value in entry["shape"])
        if allocation["format"] == "iqk":
            geometry = iqk_dense_geometry(allocation["iqk_codec"])
            payload = rows * geometry.bytes_per_row(in_features)
        else:
            geometry = KQUANT_GEOMETRY[allocation["kquant_codec"]]
            payload = (
                rows * (in_features // geometry.weights_per_block) * geometry.bytes_per_block + 1
            )
        largest_direct_group = max(largest_direct_group, payload)
    for entry in passthrough:
        elements = 1
        for value in entry["shape"]:
            elements *= int(value)
        largest_direct_group = max(largest_direct_group, elements * 2)

    layer_payloads: dict[int, int] = {}
    q8_cache_payload = 0
    for cell in expert_price["cells"]:
        layer = int(cell["layer_index"])
        payload = int(cell["weight_bytes"]) + int(cell["auxiliary_bytes"])
        layer_payloads[layer] = layer_payloads.get(layer, 0) + payload
        if cell["codec"] == FALLBACK_CODEC:
            q8_cache_payload += payload
    largest_expert_layer = max(layer_payloads.values())
    ple_shard_bytes = int(ple_contract.rows_per_shard * ple_contract.row_bytes)
    largest_atomic_payload = max(
        shard_cap,
        largest_direct_group,
        largest_expert_layer,
        ple_shard_bytes,
    )
    output_payload = int(pricing["neural_plus_ple_payload_bytes"])
    dense_cache_payload = int(pricing["direct"]["kquant_weight_bytes"]) + int(
        pricing["direct"]["kquant_placeholder_bytes"]
    )
    cache_payload = dense_cache_payload + q8_cache_payload if include_encode_cache else 0
    metadata_reserve = 1 << 30
    required = output_payload + largest_atomic_payload + cache_payload + metadata_reserve
    return {
        "modeled_output_payload_bytes_exact": output_payload,
        "modeled_output_scope": (
            "neural tensors, K-quant placeholders and raw PLE rows; "
            "safetensors headers, tokenizer and generated JSON are reserved separately"
        ),
        "largest_atomic_temp_payload_bytes": largest_atomic_payload,
        "largest_atomic_components": {
            "configured_model_shard_cap": shard_cap,
            "largest_direct_tensor_group": largest_direct_group,
            "largest_expert_layer_bundle": largest_expert_layer,
            "ple_row_file": ple_shard_bytes,
        },
        "optional_encode_cache_payload_bytes_exact": (dense_cache_payload + q8_cache_payload),
        "encode_cache_included": bool(include_encode_cache),
        "metadata_and_filesystem_reserve_bytes": metadata_reserve,
        "required_free_space_bytes": required,
    }


def _storage_volume_status(path: str | Path, required_bytes: int) -> dict:
    candidate = Path(path)
    probe = candidate if candidate.exists() else candidate.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    return {
        "available_free_space_bytes": free,
        "required_free_space_bytes": int(required_bytes),
        "sufficient": free >= int(required_bytes),
    }


def _storage_guard_with_ple_reuse(
    storage_guard: Mapping[str, object],
    *,
    ple_bytes: int,
) -> dict:
    """Price new blocks when final PLE files reuse existing same-volume inodes."""
    guard = dict(storage_guard)
    components = dict(guard["largest_atomic_components"])
    physical_output = int(guard["modeled_output_payload_bytes_exact"]) - ple_bytes
    if physical_output < 0:
        raise Qwen4IQKPackageError("PLE reuse exceeds modeled package payload")
    largest_atomic = max(
        int(components["configured_model_shard_cap"]),
        int(components["largest_direct_tensor_group"]),
        int(components["largest_expert_layer_bundle"]),
    )
    cache = (
        int(guard["optional_encode_cache_payload_bytes_exact"])
        if guard["encode_cache_included"]
        else 0
    )
    reserve = int(guard["metadata_and_filesystem_reserve_bytes"])
    guard.update(
        {
            "physical_new_output_payload_bytes_exact": physical_output,
            "ple_reuse_payload_bytes_exact": ple_bytes,
            "largest_atomic_temp_payload_bytes": largest_atomic,
            "required_free_space_bytes": physical_output
            + largest_atomic
            + cache
            + reserve,
        }
    )
    return guard


def _source_parts(
    model_dir: Path,
    teacher_capture: str | Path,
    *,
    dense_teacher_capture: str | Path | None = None,
    require_dense: bool = True,
) -> dict:
    config = _read_config(model_dir)
    if family_of(config) != QWEN4_FAMILY:
        raise Qwen4IQKPackageError("source is not the released Qwen4-Exp family")
    index_path = model_dir / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen4IQKPackageError("could not read source safetensors index") from exc
    static_issues = validate_qwen38_flash_next_static(config, index)
    blocking = [issue for issue in static_issues if issue.blocking]
    if blocking:
        raise Qwen4IQKPackageError(
            "source static contract failed: "
            + "; ".join(f"{issue.code}: {issue.message}" for issue in blocking[:6])
        )
    headers = scan_headers(model_dir)
    expected_shards = set(index["weight_map"].values())
    complete_shards = {shard for shard in expected_shards if (model_dir / shard).is_file()}
    header_issues = validate_qwen38_flash_next_headers(
        headers,
        index=index,
        complete_shards=complete_shards,
        require_complete=True,
    )
    header_blocking = [issue for issue in header_issues if issue.blocking]
    if header_blocking:
        raise Qwen4IQKPackageError(
            "source header contract failed: "
            + "; ".join(f"{issue.code}: {issue.message}" for issue in header_blocking[:6])
        )
    inventory = build_inventory_from_headers(
        headers,
        {
            "source_root": model_dir.name,
            "source_format": "hf_safetensors",
        },
        layer_types=_layer_types(config),
        family=QWEN4_FAMILY,
    )
    if inventory.get("status") != "valid":
        raise Qwen4IQKPackageError(f"source inventory failed: {_blocking_messages(inventory)}")
    leaked = [
        entry.get("source_name")
        for entry in inventory.get("tensors", [])
        if str(entry.get("source_name", "")).startswith(("model.visual.", "mtp."))
    ]
    if leaked:
        raise Qwen4IQKPackageError("text inventory contains vision or MTP tensors")

    teacher_path = Path(teacher_capture)
    teacher_manifest_path = (
        teacher_path / "manifest.json" if teacher_path.is_dir() else teacher_path
    )
    try:
        teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen4IQKPackageError("could not read teacher manifest") from exc
    raw_teacher_source = teacher_manifest.get("source_identity")
    if not isinstance(raw_teacher_source, Mapping):
        raise Qwen4IQKPackageError("teacher manifest has no source identity")
    teacher_source_identity = dict(raw_teacher_source)
    dense_vectors: dict[str, np.ndarray] = {}
    dense_identity = None
    if require_dense:
        dense_capture = (
            teacher_capture if dense_teacher_capture is None else dense_teacher_capture
        )
        dense_vectors, dense_identity = qwen4_teacher_dense_calibration(dense_capture)
        if dense_identity.get("source") != teacher_source_identity:
            raise Qwen4IQKPackageError(
                "dense and routed teacher captures bind different source models"
            )
    try:
        source_identity = qwen4_hf_snapshot_source_identity(
            model_dir,
            teacher_source_identity=teacher_source_identity,
        )
    except ValueError as exc:
        raise Qwen4IQKPackageError(str(exc)) from exc
    expert_counts = qwen4_teacher_expert_counts(teacher_capture)
    if set(expert_counts) != set(range(_LAYER_COUNT)):
        raise Qwen4IQKPackageError("teacher expert counts do not cover 48 layers")
    return {
        "config": config,
        "index": index,
        "inventory": inventory,
        "dense_vectors": dense_vectors,
        "dense_identity": dense_identity,
        "expert_counts": expert_counts,
        "teacher_source_identity": teacher_source_identity,
        "source_identity": source_identity,
    }


def _ple_contract(config: Mapping[str, object]):
    raw_text = config.get("text_config")
    text = dict(raw_text) if isinstance(raw_text, Mapping) else raw_text
    if isinstance(text, dict):
        text.setdefault("seed", 1234)
    architecture = {
        "family": QWEN4_FAMILY,
        "text_model_type": text.get("model_type") if isinstance(text, Mapping) else None,
        "config": text,
    }
    contract = derive_qwen4_ple_provider_contract(architecture)
    if contract is None:
        raise Qwen4IQKPackageError("released source has no PLE provider contract")
    ple_bytes = contract.padded_rows * contract.row_bytes
    if ple_bytes != _EXPECTED_PLE_BYTES:
        raise Qwen4IQKPackageError(f"released PLE byte accounting drifted: {ple_bytes}")
    return contract


def _build_plan(
    parts: dict,
    cells: dict,
    decision: dict,
    *,
    artifacts_identity=None,
    optimized_kernels_expected: bool = False,
    force_format=(),
    allow_unmatched_force: bool = False,
    dry_run: bool = False,
    shard_size_gb: float = 4.0,
    include_encode_cache: bool = False,
    max_experts: int | None = None,
    direct_codec: str = "q6_k",
    expert_output_layout: str = IQK_LAYOUT_IQK_RELAYOUT,
    expert_selection: Mapping[str, object] | None = None,
) -> tuple[dict, dict]:
    zero_count_policy = _decision_zero_count_policy(decision)
    experts = _expert_allocations(cells, output_layout=expert_output_layout)
    dense, passthrough, direct_price = build_qwen4_direct_allocations(
        parts["inventory"],
        parts["dense_vectors"],
        direct_codec=direct_codec,
    )
    retained = (
        None
        if expert_selection is None
        else validate_expert_selection(expert_selection)
    )
    if retained is not None and max_experts is not None:
        raise Qwen4IQKPackageError(
            "compact expert selection cannot be combined with max_experts"
        )
    _validate_decision_expert_selection(
        decision,
        expert_selection,
        retained,
        zero_count_policy,
    )
    expert_counts = (
        _EXPERT_COUNT
        if retained is None
        else {layer: len(ids) for layer, ids in retained.items()}
    )
    expert_price = price_qwen4_expert_allocation(cells, num_experts=expert_counts)
    ple = _ple_contract(parts["config"])
    pricing = {
        "direct": direct_price,
        "experts": {key: value for key, value in expert_price.items() if key != "cells"},
        "ple_bytes": ple.padded_rows * ple.row_bytes,
        "neural_weight_bytes": direct_price["weight_bytes"] + expert_price["weight_bytes"],
        "neural_payload_bytes": (
            direct_price["weight_bytes"]
            + direct_price["kquant_placeholder_bytes"]
            + expert_price["stored_payload_bytes"]
        ),
    }
    pricing["neural_plus_ple_payload_bytes"] = (
        pricing["neural_payload_bytes"] + pricing["ple_bytes"]
    )
    storage_guard = _storage_guard(
        pricing=pricing,
        expert_price=expert_price,
        direct_allocations=dense,
        passthrough=passthrough,
        inventory=parts["inventory"],
        ple_contract=ple,
        shard_size_gb=shard_size_gb,
        include_encode_cache=include_encode_cache,
    )
    teacher_source = parts["source_identity"]["teacher_source_identity"]
    subject = {
        "source_root": teacher_source["model_id"],
        "source_format": "hf_safetensors",
        "revision": teacher_source["revision"],
        "snapshot_identity_sha256": parts["source_identity"]["snapshot_identity_sha256"],
    }
    selection_block = (
        None
        if expert_selection is None
        else selection_manifest_block(expert_selection)
    )
    plan = build_iqk_package_plan(
        subject,
        experts,
        artifacts_identity=artifacts_identity,
        allocation_source="optimizer_decision",
        imatrix_identity=parts["dense_identity"],
        extra_allocation=dense,
        source_decision_id=decision.get("artifact_id"),
        source_probe_id=decision.get("source_probe_id"),
        additional_source_constraints={
            "family": QWEN4_FAMILY,
            "source_identity": parts["source_identity"],
            "text_boundary": {
                "text_layers": _LAYER_COUNT,
                "vision": "excluded",
                "mtp": "excluded",
                "ple": "raw_bf16_provider",
            },
            "expert_policy": {
                "fallback_layers": (
                    [] if zero_count_policy is not None else sorted(FALLBACK_LAYERS)
                ),
                "fallback_codec": (
                    None if zero_count_policy is not None else FALLBACK_CODEC
                ),
                "iqk_members": sorted(IQK_MEMBERS),
                "surface_run_contract_identity": cells[(0, "gate")][
                    "surface_run_contract_identity"
                ],
                **(
                    {"zero_count_policy": zero_count_policy}
                    if zero_count_policy is not None
                    else {}
                ),
            },
            "direct_policy": {
                "codec": direct_codec,
                "embedding_codec": "q8_0",
                "router_codec": "bf16",
                "calibration": "dense_affine_input_second_moments",
                **(
                    {"iqk_dense_relayout": (dense_relayout_implementation_identity())}
                    if direct_codec in IQK_DENSE_MEMBERS
                    else {}
                ),
            },
            "pricing": pricing,
            "storage_guard": storage_guard,
            **(
                {"expert_selection": selection_block}
                if selection_block is not None
                else {}
            ),
        },
        additional_achieved={
            "qwen4_expert_codec_counts": expert_price["codec_counts"],
            "qwen4_direct_format_counts": direct_price["format_counts"],
            "qwen4_weight_bytes": {
                "direct": direct_price["weight_bytes"],
                "experts": expert_price["weight_bytes"],
                "ple": ple.padded_rows * ple.row_bytes,
            },
            **(
                {
                    "qwen4_expert_selection": {
                        "retained_expert_rows": sum(
                            row["num_experts"]
                            for row in selection_block["layers"].values()
                        ),
                        "per_layer_num_experts": {
                            layer: row["num_experts"]
                            for layer, row in selection_block["layers"].items()
                        },
                    }
                }
                if selection_block is not None
                else {}
            ),
        },
        optimized_kernels_expected=optimized_kernels_expected,
        force_overrides=parse_force_overrides(force_format),
        allow_unmatched_force=allow_unmatched_force,
        dry_run=dry_run,
        inputs=(
            None
            if expert_selection is None
            else [str(expert_selection["artifact_id"])]
        ),
        additional_required_features=(
            None if expert_selection is None else [SELECTION_FEATURE]
        ),
        objective="qwen4_iqk_expert_allocation",
    )
    if plan.get("status") != "valid":
        raise Qwen4IQKPackageError(f"package plan failed: {_blocking_messages(plan)}")
    return plan, {
        "expert_allocations": experts,
        "dense_allocations": dense,
        "passthrough": passthrough,
        "direct_price": direct_price,
        "expert_price": expert_price,
        "ple_contract": ple,
        "pricing": pricing,
        "storage_guard": storage_guard,
        "expert_selection": expert_selection,
        "retained_source_ids": retained,
    }


def preflight_qwen4_iqk_package(
    model_dir: str | Path,
    *,
    allocation_path: str | Path,
    teacher_capture: str | Path,
    dense_teacher_capture: str | Path | None = None,
    shard_size_gb: float = 4.0,
    include_encode_cache: bool = False,
    optimized_kernels_expected: bool = False,
    direct_codec: str = "q6_k",
    expert_output_layout: str = IQK_LAYOUT_IQK_RELAYOUT,
    max_experts: int | None = None,
    expert_selection_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    reuse_ple_from: str | Path | None = None,
) -> dict:
    """Validate source, calibration and allocation and return exact byte prices."""
    model_dir = Path(model_dir)
    parts = _source_parts(
        model_dir,
        teacher_capture,
        dense_teacher_capture=dense_teacher_capture,
    )
    cells, decision = read_qwen4_iqk_allocation(
        allocation_path,
        inventory=parts["inventory"],
        source_identity=parts["source_identity"],
    )
    expert_selection = _read_expert_selection(expert_selection_path)
    plan, assembled = _build_plan(
        parts,
        cells,
        decision,
        optimized_kernels_expected=optimized_kernels_expected,
        shard_size_gb=shard_size_gb,
        include_encode_cache=include_encode_cache,
        max_experts=max_experts,
        direct_codec=direct_codec,
        expert_output_layout=expert_output_layout,
        expert_selection=expert_selection,
    )
    inspect_qwen4_ple_source(model_dir, expected=assembled["ple_contract"])
    storage_guard = assembled["storage_guard"]
    if reuse_ple_from is not None:
        if output_dir is None:
            raise Qwen4IQKPackageError(
                "PLE reuse preflight requires the package output directory"
            )
        inspect_qwen4_ple_reuse(
            reuse_ple_from,
            expected=assembled["ple_contract"],
            target_dir=output_dir,
        )
        storage_guard = _storage_guard_with_ple_reuse(
            storage_guard,
            ple_bytes=int(assembled["pricing"]["ple_bytes"]),
        )
    report = {
        "status": "preflight",
        "validated": [
            "released source metadata and tensor headers",
            "teacher calibration identity",
            "144-cell allocation",
            "direct policy fit",
            "raw PLE source geometry",
            "payload and storage accounting",
        ],
        "not_validated": [
            "converted IQ_K artifact files and digests",
            "package write",
            "runtime load and generation",
        ],
        "source_identity": parts["source_identity"],
        "allocation_decision_id": decision["artifact_id"],
        "package_plan_id": plan["artifact_id"],
        "optimized_kernels_expected": optimized_kernels_expected,
        "direct_codec": direct_codec,
        "expert_output_layout": expert_output_layout,
        "expert_codec_counts": assembled["expert_price"]["codec_counts"],
        "direct_format_counts": assembled["direct_price"]["format_counts"],
        "pricing": assembled["pricing"],
        "storage_guard": storage_guard,
        "ple_reuse": reuse_ple_from is not None,
        "expert_selection_id": (
            None if expert_selection is None else expert_selection["artifact_id"]
        ),
    }
    if output_dir is not None:
        report["storage_volume"] = _storage_volume_status(
            output_dir,
            storage_guard["required_free_space_bytes"],
        )
    return report


def _validate_conversion_zero_count_records(
    records: object,
    *,
    expected_ownership: Mapping[str, tuple[int, str]],
    zero_count_policy: Mapping[str, object] | None,
    field: str,
) -> None:
    if not isinstance(records, list):
        raise Qwen4IQKPackageError(f"{field} must be an array")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise Qwen4IQKPackageError(f"{field}[{index}] must be an object")
        name = record.get("name")
        ownership = expected_ownership.get(str(name))
        if ownership is None:
            raise Qwen4IQKPackageError(
                f"{field}[{index}] has no canonical artifact ownership"
            )
        layer = record.get("layer_index")
        projection = record.get("projection")
        if (layer, projection) != ownership:
            raise Qwen4IQKPackageError(
                f"{field}[{index}] layer/projection ownership drifted"
            )
        canonical_layer = ownership[0]
        expected = (
            _expected_zero_experts(zero_count_policy, canonical_layer)
            if zero_count_policy is not None
            else []
        )
        if expected:
            if (
                record.get("zero_count_experts") != expected
                or record.get("zero_count_fallback_policy")
                != ZERO_COUNT_MEAN_POLICY
            ):
                raise Qwen4IQKPackageError(
                    f"{field}[{index}] zero-count fallback identity drifted"
                )
        elif (
            "zero_count_experts" in record
            or "zero_count_fallback_policy" in record
        ):
            raise Qwen4IQKPackageError(
                f"{field}[{index}] has unexpected zero-count fallback fields"
            )


def _conversion_identity(
    path: str | Path,
    *,
    artifacts: IQKConvertedArtifacts,
    source_identity: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict:
    inventory_path = Path(path)
    try:
        payload = json.loads(inventory_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen4IQKPackageError("could not read converted-artifact inventory") from exc
    if payload.get("schema") != CONVERSION_INVENTORY_SCHEMA:
        raise Qwen4IQKPackageError("converted-artifact inventory schema is unsupported")
    if payload.get("source_identity") != dict(source_identity):
        raise Qwen4IQKPackageError("converted-artifact source identity drifted")
    if payload.get("allocation_decision_id") != decision.get("artifact_id"):
        raise Qwen4IQKPackageError("converted artifacts bind another allocation decision")
    if payload.get("teacher_identity") != decision.get("teacher_identity"):
        raise Qwen4IQKPackageError("converted-artifact teacher identity drifted")
    if payload.get("surface_identity") != decision.get("surface_identity"):
        raise Qwen4IQKPackageError("converted-artifact surface identity drifted")
    zero_count_policy = _decision_zero_count_policy(decision)
    if ("zero_count_policy" in payload) != (zero_count_policy is not None) or payload.get(
        "zero_count_policy"
    ) != zero_count_policy:
        raise Qwen4IQKPackageError("converted-artifact zero-count policy drifted")
    run_contract = payload.get("conversion_run_contract")
    if not isinstance(run_contract, Mapping):
        raise Qwen4IQKPackageError("converted-artifact run contract is missing")
    run_identity = _digest(
        run_contract.get("identity_sha256"),
        field="conversion_run_contract.identity_sha256",
    )
    run_body = {key: value for key, value in run_contract.items() if key != "identity_sha256"}
    canonical = json.dumps(
        run_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if run_identity != hashlib.sha256(canonical.encode()).hexdigest():
        raise Qwen4IQKPackageError("converted-artifact run identity does not verify")
    if run_contract.get("allocation_decision_id") != decision.get("artifact_id"):
        raise Qwen4IQKPackageError("converted-artifact run binds another decision")
    if ("zero_count_policy" in run_contract) != (
        zero_count_policy is not None
    ) or run_contract.get("zero_count_policy") != zero_count_policy:
        raise Qwen4IQKPackageError("converted-artifact run zero-count policy drifted")
    if run_contract.get("source_identity_sha256") != source_identity.get(
        "snapshot_identity_sha256"
    ):
        raise Qwen4IQKPackageError("converted-artifact run source identity drifted")
    teacher = decision.get("teacher_identity")
    surface = decision.get("surface_identity")
    if not isinstance(teacher, Mapping) or run_contract.get(
        "teacher_identity_sha256"
    ) != teacher.get("identity_sha256"):
        raise Qwen4IQKPackageError("converted-artifact run teacher identity drifted")
    if (
        not isinstance(surface, Mapping)
        or run_contract.get("surface_content_sha256") != surface.get("content_sha256")
        or run_contract.get("surface_run_contract_identity") != surface.get("run_contract_identity")
    ):
        raise Qwen4IQKPackageError("converted-artifact run surface identity drifted")
    if run_contract.get("relayout_implementation") != relayout_implementation_identity():
        raise Qwen4IQKPackageError("converted-artifact relayout implementation drifted")
    encoder = run_contract.get("encoder")
    if (
        not isinstance(encoder, Mapping)
        or encoder.get("source_layout") != IQK_LAYOUT_IK_WIRE
        or encoder.get("published_layout") != IQK_LAYOUT_IQK_RELAYOUT
    ):
        raise Qwen4IQKPackageError("converted artifacts are not package-ready relayout")
    if artifacts.layout != IQK_LAYOUT_IQK_RELAYOUT:
        raise Qwen4IQKPackageError("converted artifact reader layout is not relayout")
    entries = payload.get("files")
    if not isinstance(entries, list):
        raise Qwen4IQKPackageError("converted-artifact inventory has no files array")
    expected_ownership = {
        cell["path"].name: (layer, projection)
        for (layer, projection), cell in artifacts.cells.items()
    }
    allocation = decision.get("allocation")
    if isinstance(allocation, list):
        decision_ownership = {}
        for index, cell in enumerate(allocation):
            if not isinstance(cell, Mapping) or cell.get("codec") not in IQK_MEMBERS:
                continue
            layer = cell.get("layer_index")
            projection = cell.get("projection")
            codec = cell.get("codec")
            if (
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or projection not in PROJECTIONS
                or not isinstance(codec, str)
            ):
                raise Qwen4IQKPackageError(
                    f"decision allocation cell {index} has invalid IQ_K ownership"
                )
            name = f"layer{layer:02d}_{projection}.{codec}"
            if name in decision_ownership:
                raise Qwen4IQKPackageError(
                    f"decision allocation duplicates converted artifact {name}"
                )
            decision_ownership[name] = (layer, str(projection))
        if decision_ownership != expected_ownership:
            raise Qwen4IQKPackageError(
                "converted artifacts do not match decision IQ_K ownership"
            )
    _validate_conversion_zero_count_records(
        entries,
        expected_ownership=expected_ownership,
        zero_count_policy=zero_count_policy,
        field="converted-artifact files",
    )
    names = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, Mapping) else None
        if not isinstance(name, str) or Path(name).name != name:
            raise Qwen4IQKPackageError("converted-artifact file name is not canonical")
        if entry.get("layout") != IQK_LAYOUT_IQK_RELAYOUT:
            raise Qwen4IQKPackageError(
                "converted-artifact file is not on the package-ready relayout"
            )
        names.append(name)
    expected = sorted(cell["path"].name for cell in artifacts.cells.values())
    if sorted(names) != expected or len(names) != len(set(names)):
        raise Qwen4IQKPackageError("converted-artifact inventory does not match IQ_K cells")
    by_name = {entry["name"]: entry for entry in entries}
    for cell in artifacts.cells.values():
        entry = by_name[cell["path"].name]
        if (
            entry.get("codec") != cell["codec"]
            or entry.get("stored_shape") != [cell["out_features"], cell["in_features"]]
            or entry.get("row_bytes") != cell["bytes_per_row"]
            or entry.get("bytes_per_expert") != cell["bytes_per_expert"]
        ):
            raise Qwen4IQKPackageError(
                f"converted-artifact geometry drifted for {cell['path'].name}"
            )
    run_cells = run_contract.get("cells")
    _validate_conversion_zero_count_records(
        run_cells,
        expected_ownership=expected_ownership,
        zero_count_policy=zero_count_policy,
        field="converted-artifact run cells",
    )
    expected_run_cells = [
        {key: value for key, value in by_name[name].items() if key != "sha256"} for name in expected
    ]
    expected_run_cells.sort(
        key=lambda row: (
            int(row["layer_index"]),
            PROJECTIONS.index(row["projection"]),
        )
    )
    if run_cells != expected_run_cells:
        raise Qwen4IQKPackageError("converted-artifact run cell contract drifted")
    digest_report = artifacts.verify_digests(inventory_path)
    artifact_identity = artifacts.identity()
    return {
        **artifact_identity,
        "schema": CONVERSION_INVENTORY_SCHEMA,
        "inventory_sha256": digest_report["inventory_sha256"],
        "digests": {
            "files_checked": digest_report["files_checked"],
            "bytes_checked": digest_report["bytes_checked"],
            "inventory_sha256": digest_report["inventory_sha256"],
        },
    }


def _write_hotlist(
    out_dir: Path,
    counts: Mapping[int, np.ndarray],
    retained_source_ids: Mapping[int, tuple[int, ...]] | None = None,
) -> int:
    from moespresso.package.hotlist import build_package_expert_hotlist
    from moespresso.runtime.expert_index import build_expert_index

    index = build_expert_index(out_dir)
    if retained_source_ids is None:
        sliced_counts = {
            int(layer): np.asarray(values)[: index.num_experts]
            for layer, values in counts.items()
        }
        package_expert_counts: int | dict[int, int] = index.num_experts
    else:
        if set(retained_source_ids) != set(counts):
            raise Qwen4IQKPackageError(
                "expert selection and teacher counts cover different layers"
            )
        sliced_counts = {
            int(layer): np.asarray(values)[np.asarray(retained_source_ids[layer])]
            for layer, values in counts.items()
        }
        package_expert_counts = {
            int(layer): index.num_experts_for_layer(layer)
            for layer in index.layers_indexed()
        }
    payload = build_package_expert_hotlist(
        sliced_counts,
        layers_indexed=index.layers_indexed(),
        num_experts=package_expert_counts,
        source={
            "kind": "route_active_calibration_counts",
            "use": "cold-start expert prewarm ranking only",
        },
    )
    (out_dir / "expert_hotlist.json").write_text(json.dumps(payload))
    return len(payload["layers"])


def build_qwen4_iqk_package(
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    allocation_path: str | Path,
    teacher_capture: str | Path,
    dense_teacher_capture: str | Path | None = None,
    routed_artifacts_dir: str | Path,
    routed_inventory_path: str | Path,
    seed: int = 42,
    shard_size_gb: float = 4.0,
    chunk_bytes: int | None = None,
    max_experts: int | None = None,
    kquant_encoder=None,
    iqk_dense_encoder=None,
    kquant_cache_dir: str | Path | None = None,
    optimized_kernels_expected: bool = False,
    direct_codec: str = "q6_k",
    expert_output_layout: str = IQK_LAYOUT_IQK_RELAYOUT,
    expert_selection_path: str | Path | None = None,
    reuse_ple_from: str | Path | None = None,
    force_format: list[str] | tuple[str, ...] | None = None,
    allow_unmatched_force: bool = False,
    force_format_dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Build the text-only package without loading the complete source model."""
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    log = (lambda message: print(message, flush=True)) if verbose else (lambda _message: None)
    parts = _source_parts(
        model_dir,
        teacher_capture,
        dense_teacher_capture=dense_teacher_capture,
    )
    cells, decision = read_qwen4_iqk_allocation(
        allocation_path,
        inventory=parts["inventory"],
        source_identity=parts["source_identity"],
    )
    expert_selection = _read_expert_selection(expert_selection_path)
    initial_plan, initial = _build_plan(
        parts,
        cells,
        decision,
        optimized_kernels_expected=optimized_kernels_expected,
        force_format=force_format or (),
        allow_unmatched_force=allow_unmatched_force,
        dry_run=force_format_dry_run,
        shard_size_gb=shard_size_gb,
        include_encode_cache=kquant_cache_dir is not None,
        max_experts=max_experts,
        direct_codec=direct_codec,
        expert_output_layout=expert_output_layout,
        expert_selection=expert_selection,
    )
    inspect_qwen4_ple_source(model_dir, expected=initial["ple_contract"])
    effective_storage_guard = initial["storage_guard"]
    if reuse_ple_from is not None:
        inspect_qwen4_ple_reuse(
            reuse_ple_from,
            expected=initial["ple_contract"],
            target_dir=out_dir,
        )
        effective_storage_guard = _storage_guard_with_ple_reuse(
            effective_storage_guard,
            ple_bytes=int(initial["pricing"]["ple_bytes"]),
        )
    initial_storage_volume = _storage_volume_status(
        out_dir,
        effective_storage_guard["required_free_space_bytes"],
    )
    if force_format_dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_artifact(out_dir / INVENTORY_NAME, parts["inventory"])
        write_artifact(out_dir / PACKAGE_PLAN_NAME, initial_plan)
        return initial_plan
    if not initial_storage_volume["sufficient"]:
        raise Qwen4IQKPackageError(
            "insufficient output free space: "
            f"{initial_storage_volume['available_free_space_bytes']} available, "
            f"{initial_storage_volume['required_free_space_bytes']} required"
        )

    iqk_members = {
        layer: {projection: str(cells[(layer, projection)]["codec"]) for projection in PROJECTIONS}
        for layer in range(_LAYER_COUNT)
        if all(cells[(layer, projection)]["format"] == "iqk" for projection in PROJECTIONS)
    }
    stored_shapes = {
        layer: {
            projection: tuple(cells[(layer, projection)]["stored_shape"])
            for projection in PROJECTIONS
        }
        for layer in iqk_members
    }
    try:
        artifacts = IQKConvertedArtifacts(
            routed_artifacts_dir,
            iqk_members,
            stored_shapes,
            _EXPERT_COUNT,
            layout=IQK_LAYOUT_IQK_RELAYOUT,
        )
    except ValueError as exc:
        raise Qwen4IQKPackageError(str(exc)) from exc
    try:
        artifacts_identity = _conversion_identity(
            routed_inventory_path,
            artifacts=artifacts,
            source_identity=parts["source_identity"],
            decision=decision,
        )
        package_plan, assembled = _build_plan(
            parts,
            cells,
            decision,
            artifacts_identity=artifacts_identity,
            optimized_kernels_expected=optimized_kernels_expected,
            force_format=force_format or (),
            allow_unmatched_force=allow_unmatched_force,
            shard_size_gb=shard_size_gb,
            include_encode_cache=kquant_cache_dir is not None,
            max_experts=max_experts,
            direct_codec=direct_codec,
            expert_output_layout=expert_output_layout,
            expert_selection=expert_selection,
        )
        storage_volume = initial_storage_volume
        if kquant_encoder is None:
            check_kquant_backend_available()
        has_direct_iqk = any(
            allocation.get("format") == "iqk" and allocation.get("kind") == "affine"
            for allocation in assembled["dense_allocations"]
        )
        if has_direct_iqk and iqk_dense_encoder is None:
            iqk_dense_encoder = encode_qwen4_dense_iqk
        try:
            qwen4_hf_snapshot_source_identity(
                model_dir,
                teacher_source_identity=parts["teacher_source_identity"],
                expected_source_identity=parts["source_identity"],
            )
        except ValueError as exc:
            raise Qwen4IQKPackageError(str(exc)) from exc

        out_dir.mkdir(parents=True, exist_ok=True)
        write_artifact(out_dir / INVENTORY_NAME, parts["inventory"])
        write_artifact(out_dir / PACKAGE_PLAN_NAME, package_plan)
        selection_files = []
        compact_layout = None
        if expert_selection is not None:
            selection_path = out_dir / EXPERT_SELECTION_NAME
            write_artifact(selection_path, expert_selection)
            selection_files.append(file_identity(selection_path))
            compact_layout = {
                "stacked": False,
                "bundled": True,
                "fused_gate_up": True,
                "shard_per_layer": False,
                "per_layer_experts": selection_manifest_block(expert_selection),
            }
        tokenizer = copy_tokenizer_into_package(
            model_dir,
            out_dir,
            family=QWEN4_FAMILY,
        )
        ple_provider, ple_files = write_qwen4_ple_provider(
            model_dir,
            out_dir,
            expected=assembled["ple_contract"],
            reuse_from=reuse_ple_from,
        )
        cache = KQuantEncodeCache(kquant_cache_dir) if kquant_cache_dir is not None else None
        write_kwargs = {
            "shard_size_gb": shard_size_gb,
            "passthrough": assembled["passthrough"],
            "tokenizer": tokenizer,
            "max_experts": max_experts,
            "kquant_imatrix_vectors": parts["dense_vectors"],
            "kquant_encoder": kquant_encoder,
            "iqk_dense_encoder": iqk_dense_encoder,
            "kquant_cache": cache,
            "kquant_cache_context": {
                "recipe_kind": "qwen4_calibrated_iqk_text",
                "source_decision_id": decision["artifact_id"],
            },
            "iqk_expert_loader": artifacts.expert_blocks,
            "iqk_expert_source_layout": IQK_LAYOUT_IQK_RELAYOUT,
            "expert_source_ids": assembled["retained_source_ids"],
            "expert_layout": compact_layout,
            "additional_files": [*ple_files, *selection_files],
            "ple_provider": ple_provider,
        }
        if chunk_bytes is not None:
            write_kwargs["chunk_bytes"] = chunk_bytes
        log("writing text weights and package-owned PLE rows")
        manifest = write_package(
            package_plan,
            model_dir,
            parts["config"],
            out_dir,
            **write_kwargs,
        )
        if manifest.get("status") != "valid":
            raise Qwen4IQKPackageError(f"package manifest failed: {_blocking_messages(manifest)}")
        write_artifact(out_dir / MANIFEST_NAME, manifest)

        from moespresso.package.sidecars import build_sidecars

        config_json, jang_config = build_sidecars(manifest, seed=seed)
        (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))
        (out_dir / "jang_config.json").write_text(json.dumps(jang_config, indent=2))
        hotlist_layers = _write_hotlist(
            out_dir,
            parts["expert_counts"],
            assembled["retained_source_ids"],
        )
        report = {
            "status": "valid",
            "source_identity": parts["source_identity"],
            "allocation_decision_id": decision["artifact_id"],
            "package_plan_id": package_plan["artifact_id"],
            "package_manifest_id": manifest["artifact_id"],
            "expert_codec_counts": assembled["expert_price"]["codec_counts"],
            "expert_layout": expert_output_layout,
            "direct_format_counts": assembled["direct_price"]["format_counts"],
            "direct_codec": direct_codec,
            "pricing": assembled["pricing"],
            "storage_guard": effective_storage_guard,
            "storage_volume": storage_volume,
            "ple_reuse": reuse_ple_from is not None,
            "expert_hotlist_layers": hotlist_layers,
            "kquant_cache": None if cache is None else cache.summary(),
            "expert_selection_id": (
                None if expert_selection is None else expert_selection["artifact_id"]
            ),
        }
        (out_dir / IQK_REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True))
        return manifest
    finally:
        artifacts.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the released Qwen4 text package from calibrated IQ_K cells"
    )
    parser.add_argument("model_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--teacher-capture", required=True)
    parser.add_argument("--dense-teacher-capture")
    parser.add_argument("--routed-artifacts")
    parser.add_argument("--routed-inventory")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size-gb", type=float, default=4.0)
    parser.add_argument("--chunk-bytes", type=int)
    parser.add_argument("--max-experts-per-layer", type=int)
    parser.add_argument("--expert-selection")
    parser.add_argument(
        "--reuse-ple-from",
        help="hardlink manifest-verified PLE rows from a package on the same volume",
    )
    parser.add_argument("--kquant-cache-dir")
    parser.add_argument("--optimized-kernels-expected", action="store_true")
    parser.add_argument(
        "--direct-codec",
        choices=sorted(DIRECT_CODEC_CHOICES),
        default="q6_k",
    )
    parser.add_argument(
        "--expert-output-layout",
        choices=_EXPERT_OUTPUT_LAYOUTS,
        default=IQK_LAYOUT_IQK_RELAYOUT,
        help="routed IQ_K bundle layout; converted artifacts remain iqk_relayout",
    )
    parser.add_argument("--force-format", action="append", default=[])
    parser.add_argument("--allow-unmatched-force", action="store_true")
    parser.add_argument("--force-format-dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.max_experts_per_layer is not None and args.max_experts_per_layer < 1:
        parser.error("--max-experts-per-layer must be >= 1")
    if not args.preflight_only and (args.routed_artifacts is None or args.routed_inventory is None):
        parser.error("a build requires --routed-artifacts and --routed-inventory")
    try:
        if args.preflight_only:
            result = preflight_qwen4_iqk_package(
                args.model_dir,
                allocation_path=args.allocation,
                teacher_capture=args.teacher_capture,
                dense_teacher_capture=args.dense_teacher_capture,
                shard_size_gb=args.shard_size_gb,
                include_encode_cache=args.kquant_cache_dir is not None,
                optimized_kernels_expected=args.optimized_kernels_expected,
                direct_codec=args.direct_codec,
                expert_output_layout=args.expert_output_layout,
                max_experts=args.max_experts_per_layer,
                expert_selection_path=args.expert_selection,
                output_dir=args.out_dir,
                reuse_ple_from=args.reuse_ple_from,
            )
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            return 0
        build_qwen4_iqk_package(
            args.model_dir,
            args.out_dir,
            allocation_path=args.allocation,
            teacher_capture=args.teacher_capture,
            dense_teacher_capture=args.dense_teacher_capture,
            routed_artifacts_dir=args.routed_artifacts,
            routed_inventory_path=args.routed_inventory,
            seed=args.seed,
            shard_size_gb=args.shard_size_gb,
            chunk_bytes=args.chunk_bytes,
            max_experts=args.max_experts_per_layer,
            kquant_cache_dir=args.kquant_cache_dir,
            optimized_kernels_expected=args.optimized_kernels_expected,
            direct_codec=args.direct_codec,
            expert_output_layout=args.expert_output_layout,
            expert_selection_path=args.expert_selection,
            reuse_ple_from=args.reuse_ple_from,
            force_format=args.force_format,
            allow_unmatched_force=args.allow_unmatched_force,
            force_format_dry_run=args.force_format_dry_run,
            verbose=args.verbose,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
