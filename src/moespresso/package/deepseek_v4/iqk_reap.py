"""Build a physically compact DeepSeek-V4 IQ_K package from an expert selection.

The transform is deliberately narrower than conversion. It reads an already
relayouted IQ_K package, copies the selected expert rows without decoding or
re-encoding them, and gathers the learned router's weight and bias rows in the
same order. The first three hash-routed layers remain in their original
256-expert identity space. Learned-router layers may retain different counts.

The output is a new package directory. All work happens in a sibling staging
directory and becomes visible only after the rewritten artifacts and package
verify. Source files are never opened for writing. Old build, relayout, and
drafter reports are not copied; the compact package receives one report that
describes this transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from moespresso.core.artifact import (
    artifact_producer,
    make_artifact,
    read_artifact,
    write_artifact,
)
from moespresso.core.paths import UnsafeArtifactPathError, resolve_artifact_file
from moespresso.package.bundle import (
    IQK_CODEC,
    METADATA_KEY,
    PROJECTIONS as BUNDLE_PROJECTIONS,
    decode_bundle_metadata,
    encode_bundle_metadata,
)
from moespresso.package.constants import MANIFEST_NAME
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    normalize_iqk_layout,
)
from moespresso.package.manifest import file_identity

PACKAGE_PLAN_NAME = "package_plan.json"
EXPERT_SELECTION_NAME = "expert_selection.json"
REAP_REPORT_NAME = "iqk_reap_report.json"

SELECTION_KIND = "deepseek_v4_expert_selection"
SELECTION_FEATURE = "deepseek_v4_per_layer_experts"
HASH_LAYERS = (0, 1, 2)
_PRODUCER = artifact_producer("moespresso.package.deepseek_v4.iqk_reap")
_ROUTER_ROLES = frozenset({"moe.router_gate", "moe.router_bias"})
_OPTIONAL_SOURCE_FURNITURE = ("source_inventory.json",)
_COPY_CHUNK = 8 << 20
_PLAN_PROJECTIONS = ("gate", "up", "down")
_BUNDLE_TO_PLAN_PROJECTION = {
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
}
_SOURCE_NAMES = frozenset({"base", "promotion"})
_PLAN_FORMAT_KEYS = ("bits", "codec", "format", "iqk_codec", "layout")
_EXPERT_IDENTITY_KEYS = (
    "source_name",
    "role",
    "kind",
    "layer_index",
    "projection",
    "module_path",
    "module_weight_key",
)


class IQKReapError(ValueError):
    """The package or selection cannot produce a safe compact IQ_K package."""


SourceVerifier = Callable[[Path, dict], None]
OutputVerifier = Callable[[Path, dict], None]
RowVerifier = Callable[[Path, Path, Sequence[tuple[int, int, int]]], dict]


def build_expert_selection(
    *,
    source_package_manifest_id: str,
    layers: Mapping[int | str, Sequence[int]],
    source_num_experts: int = 256,
    top_k: int = 6,
    subject: dict | None = None,
) -> dict:
    """Create the content-addressed per-layer expert-selection artifact.

    Validation is shared with the repacker, so an artifact produced here is
    accepted by :func:`compact_iqk_package` without a second interpretation of
    its expert-id map.
    """
    normalized: dict[str, dict] = {}
    try:
        ordered = sorted(layers.items(), key=lambda item: int(item[0]))
    except (TypeError, ValueError) as exc:
        raise IQKReapError("selection layer keys must be integers") from exc
    for raw_layer, ids in ordered:
        if isinstance(raw_layer, bool):
            raise IQKReapError("selection layer keys must not be booleans")
        layer = str(int(raw_layer))
        if layer in normalized:
            raise IQKReapError(f"selection layer {layer} is duplicated")
        normalized[layer] = {
            "num_experts": len(ids),
            "source_expert_ids": list(ids),
        }
    payload = make_artifact(
        SELECTION_KIND,
        subject or {"source_package_manifest_id": source_package_manifest_id},
        _PRODUCER,
        inputs=[source_package_manifest_id],
        required_features=[SELECTION_FEATURE],
        status="valid",
        source_package_manifest_id=source_package_manifest_id,
        source_num_experts=source_num_experts,
        top_k=top_k,
        layers=normalized,
    )
    validate_expert_selection(payload)
    return payload


def validate_expert_selection(
    selection: dict,
    *,
    expected_layers: Sequence[int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Validate and return ``layer -> ordered original expert ids``.

    Score-routed IDs are strictly increasing so their order is stable in the
    compact bundle and in the gathered router tensors. Hash layers have the
    stronger identity requirement because their token-to-expert table still
    emits original IDs.
    """
    if selection.get("artifact_kind") != SELECTION_KIND:
        raise IQKReapError(
            f"selection artifact kind is {selection.get('artifact_kind')!r}, "
            f"not {SELECTION_KIND!r}")
    if selection.get("status") != "valid":
        raise IQKReapError(
            f"selection status is {selection.get('status')!r}, not 'valid'")
    if not _digest_artifact_id(selection.get("artifact_id"), "select"):
        raise IQKReapError("selection has no valid artifact_id")
    features = selection.get("required_features")
    if not isinstance(features, list) or SELECTION_FEATURE not in features:
        raise IQKReapError(
            f"selection does not require feature {SELECTION_FEATURE!r}")
    source_count = selection.get("source_num_experts")
    top_k = selection.get("top_k")
    if isinstance(source_count, bool) or not isinstance(source_count, int):
        raise IQKReapError("selection source_num_experts must be an integer")
    if source_count != 256:
        raise IQKReapError(
            f"DeepSeek-V4 selection source_num_experts is {source_count}, not 256")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != 6:
        raise IQKReapError("DeepSeek-V4 selection top_k must be 6")
    if not _digest_artifact_id(selection.get("source_package_manifest_id"), "pkg"):
        raise IQKReapError("selection has no valid source_package_manifest_id")
    raw_layers = selection.get("layers")
    if not isinstance(raw_layers, dict) or not raw_layers:
        raise IQKReapError("selection carries no layer map")

    normalized: dict[int, tuple[int, ...]] = {}
    for raw_layer, record in raw_layers.items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as exc:
            raise IQKReapError(f"invalid selection layer key {raw_layer!r}") from exc
        if str(layer) != str(raw_layer):
            raise IQKReapError(
                f"selection layer key {raw_layer!r} is not canonical decimal")
        if layer < 0:
            raise IQKReapError(f"selection layer {layer} is negative")
        if layer in normalized:
            raise IQKReapError(f"duplicate selection layer {layer}")
        if not isinstance(record, dict):
            raise IQKReapError(f"selection layer {layer} must be an object")
        raw_ids = record.get("source_expert_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise IQKReapError(f"selection layer {layer} has no source_expert_ids")
        ids: list[int] = []
        for expert in raw_ids:
            if isinstance(expert, bool) or not isinstance(expert, int):
                raise IQKReapError(
                    f"selection layer {layer} expert id {expert!r} is not an integer")
            ids.append(expert)
        if ids != sorted(set(ids)):
            raise IQKReapError(
                f"selection layer {layer} expert ids are not strictly increasing")
        if ids[0] < 0 or ids[-1] >= source_count:
            raise IQKReapError(
                f"selection layer {layer} expert ids leave [0, {source_count})")
        declared = record.get("num_experts")
        if declared != len(ids):
            raise IQKReapError(
                f"selection layer {layer} num_experts {declared!r} != {len(ids)} ids")
        if layer in HASH_LAYERS:
            if ids != list(range(source_count)):
                raise IQKReapError(
                    f"hash-routed layer {layer} must retain the full identity map")
        elif len(ids) < top_k:
            raise IQKReapError(
                f"score-routed layer {layer} retains {len(ids)} experts, below top_k "
                f"{top_k}")
        normalized[layer] = tuple(ids)

    if set(normalized) != set(range(len(normalized))):
        raise IQKReapError(
            f"selection must cover contiguous layers 0 through {len(normalized) - 1}")

    if expected_layers is not None:
        want = set(int(layer) for layer in expected_layers)
        got = set(normalized)
        if got != want:
            raise IQKReapError(
                f"selection layers {_layer_span(got)} != package layers {_layer_span(want)}")
    for layer in HASH_LAYERS:
        if layer not in normalized:
            raise IQKReapError(f"selection omits hash-routed layer {layer}")
    return dict(sorted(normalized.items()))


def selection_manifest_block(selection: dict) -> dict:
    """The selection contract embedded into a package plan and manifest."""
    validate_expert_selection(selection)
    return {
        "source_selection_artifact_id": selection["artifact_id"],
        "source_package_manifest_id": selection["source_package_manifest_id"],
        "source_num_experts": selection["source_num_experts"],
        "top_k": selection["top_k"],
        "layers": json.loads(json.dumps(selection["layers"])),
    }


def validate_projection_source_map(
    source_map: Mapping[int | str, Mapping[str, str]],
    *,
    expected_layers: Sequence[int],
) -> dict[int, dict[str, str]]:
    """Validate an explicit base/promotion choice for every routed cell."""
    if not isinstance(source_map, Mapping):
        raise IQKReapError("projection source map must be an object")
    normalized: dict[int, dict[str, str]] = {}
    for raw_layer, raw_cells in source_map.items():
        if isinstance(raw_layer, bool):
            raise IQKReapError("projection source-map layers must not be booleans")
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as exc:
            raise IQKReapError(
                f"invalid projection source-map layer {raw_layer!r}") from exc
        if isinstance(raw_layer, str) and str(layer) != raw_layer:
            raise IQKReapError(
                f"projection source-map layer {raw_layer!r} is not canonical decimal")
        if layer in normalized:
            raise IQKReapError(f"projection source-map layer {layer} is duplicated")
        if not isinstance(raw_cells, Mapping):
            raise IQKReapError(
                f"projection source-map layer {layer} must be an object")
        cells = {str(projection): str(source) for projection, source in raw_cells.items()}
        if tuple(sorted(cells)) != tuple(sorted(_PLAN_PROJECTIONS)):
            raise IQKReapError(
                f"projection source-map layer {layer} covers {sorted(cells)}, "
                f"not {list(_PLAN_PROJECTIONS)}")
        for projection, source in cells.items():
            if source not in _SOURCE_NAMES:
                raise IQKReapError(
                    f"projection source-map layer {layer} {projection} names "
                    f"{source!r}; expected 'base' or 'promotion'")
        normalized[layer] = {
            projection: cells[projection] for projection in _PLAN_PROJECTIONS
        }

    want = set(int(layer) for layer in expected_layers)
    got = set(normalized)
    if got != want:
        raise IQKReapError(
            f"projection source-map layers {_layer_span(got)} != package layers "
            f"{_layer_span(want)}")
    return dict(sorted(normalized.items()))


def projection_source_manifest_block(
    source_map: Mapping[int, Mapping[str, str]],
    *,
    base_manifest_id: str,
    promotion_manifest_id: str,
) -> dict:
    """Portable provenance for a two-source routed projection repack."""
    normalized = validate_projection_source_map(
        source_map, expected_layers=sorted(int(layer) for layer in source_map))
    return {
        "base_package_manifest_id": base_manifest_id,
        "promotion_package_manifest_id": promotion_manifest_id,
        "layers": {
            str(layer): dict(cells) for layer, cells in normalized.items()
        },
    }


def rewrite_package_plan_for_selection(
    plan: dict,
    selection: dict,
    *,
    projection_source_block: dict | None = None,
    promotion_plan: dict | None = None,
    routed_bytes: int | None = None,
) -> dict:
    """Clone a package plan and bind it to a physical expert selection."""
    block = selection_manifest_block(selection)
    out = json.loads(json.dumps(plan))
    promotion_records = (
        _expert_records_by_cell(promotion_plan, "promotion package plan")
        if promotion_plan is not None else {}
    )
    source_layers = (
        (projection_source_block or {}).get("layers") or {}
    )
    for allocation in out.get("allocation", []):
        if allocation.get("kind") != "expert" or allocation.get("format") != IQK_CODEC:
            continue
        layer, projection = _expert_record_cell(allocation, "base package plan")
        if source_layers.get(str(layer), {}).get(projection) == "promotion":
            promoted = promotion_records.get((layer, projection))
            if promoted is None:
                raise IQKReapError(
                    f"promotion package plan has no expert cell {layer} {projection}")
            for key in _PLAN_FORMAT_KEYS:
                if key not in promoted:
                    raise IQKReapError(
                        f"promotion package plan expert {layer} {projection} has no {key}")
                allocation[key] = json.loads(json.dumps(promoted[key]))
        layout = normalize_iqk_layout(allocation.get("layout"))
        if layout != IQK_LAYOUT_IQK_RELAYOUT:
            raise IQKReapError(
                f"plan expert {allocation.get('source_name')} is not on the "
                "IQ_K relayout")
        allocation["layout"] = IQK_LAYOUT_IQK_RELAYOUT
    out["expert_selection"] = block
    out["producer_kind"] = "iqk_compact_expert_repack"
    out["producer_reference"] = selection["artifact_id"]
    out["required_features"] = _with_value(
        out.get("required_features"), SELECTION_FEATURE)
    out["inputs"] = _with_value(out.get("inputs"), selection["artifact_id"])
    out.setdefault("source_constraints", {})["expert_selection"] = block
    for container_name in ("source_constraints", "achieved"):
        container = out.get(container_name)
        if not isinstance(container, dict):
            continue
        iqk_artifacts = container.get("iqk_artifacts")
        if isinstance(iqk_artifacts, dict):
            iqk_artifacts.pop("root", None)
    if projection_source_block is not None:
        out["expert_projection_sources"] = json.loads(
            json.dumps(projection_source_block))
        out["source_constraints"]["expert_projection_sources"] = json.loads(
            json.dumps(projection_source_block))
        out["inputs"] = _with_value(
            out["inputs"], projection_source_block["promotion_package_manifest_id"])
    achieved = out.setdefault("achieved", {})
    achieved["expert_selection"] = {
        "source_num_experts": block["source_num_experts"],
        "per_layer_num_experts": {
            layer: record["num_experts"] for layer, record in block["layers"].items()
        },
        "retained_expert_rows": sum(
            int(record["num_experts"]) for record in block["layers"].values()),
    }
    if routed_bytes is not None:
        achieved["expert_selection"]["routed_bytes"] = int(routed_bytes)
    if projection_source_block is not None:
        codec_counts: dict[str, int] = {}
        for allocation in out.get("allocation", []):
            if allocation.get("kind") == "expert" and allocation.get("format") == IQK_CODEC:
                codec = allocation.get("iqk_codec")
                if not isinstance(codec, str):
                    raise IQKReapError(
                        f"output plan expert {allocation.get('source_name')} has no member")
                codec_counts[codec] = codec_counts.get(codec, 0) + 1
        achieved["expert_codec_counts"] = dict(sorted(codec_counts.items()))
    out.pop("artifact_id", None)
    out.pop("created_at", None)
    return out


def rewrite_package_manifest_for_selection(
    manifest: dict,
    selection: dict,
    *,
    plan_id: str,
    files: Sequence[dict],
    projection_source_block: dict | None = None,
    promotion_manifest: dict | None = None,
) -> dict:
    """Clone a manifest and declare the compact per-layer expert geometry."""
    block = selection_manifest_block(selection)
    out = json.loads(json.dumps(manifest))
    if (out.get("architecture") or {}).get("family") != "deepseek_v4_flash":
        raise IQKReapError("source manifest is not a DeepSeek-V4 package")
    layout = out.setdefault("expert_layout", {})
    layout["per_layer_experts"] = block
    promotion_records = (
        _expert_records_by_cell(promotion_manifest, "promotion package manifest")
        if promotion_manifest is not None else {}
    )
    source_layers = (
        (projection_source_block or {}).get("layers") or {}
    )
    for tensor in out.get("tensors", []):
        if tensor.get("kind") != "expert" or tensor.get("format") != IQK_CODEC:
            continue
        layer, projection = _expert_record_cell(tensor, "base package manifest")
        if source_layers.get(str(layer), {}).get(projection) == "promotion":
            promoted = promotion_records.get((layer, projection))
            if promoted is None:
                raise IQKReapError(
                    f"promotion package manifest has no expert cell {layer} {projection}")
            tensor["format"] = promoted.get("format")
            tensor["format_params"] = json.loads(json.dumps(
                promoted.get("format_params") or {}))
        params = tensor.setdefault("format_params", {})
        wire_layout = normalize_iqk_layout(params.get("layout"))
        if wire_layout != IQK_LAYOUT_IQK_RELAYOUT:
            raise IQKReapError(
                f"manifest expert {tensor.get('source_name')} is not on the "
                "IQ_K relayout")
        params["layout"] = IQK_LAYOUT_IQK_RELAYOUT
    out["files"] = sorted(
        (dict(identity) for identity in files), key=lambda identity: identity["path"])
    out["required_features"] = _with_value(
        out.get("required_features"), SELECTION_FEATURE)
    out["inputs"] = _with_value(out.get("inputs"), selection["artifact_id"])
    out["inputs"] = _with_value(out["inputs"], plan_id)
    provenance = out.setdefault("provenance", {})
    provenance["source_package_manifest_id"] = manifest["artifact_id"]
    provenance["source_expert_selection_id"] = selection["artifact_id"]
    provenance["source_plan_id"] = plan_id
    if projection_source_block is not None:
        provenance["expert_projection_sources"] = json.loads(
            json.dumps(projection_source_block))
        out["inputs"] = _with_value(
            out["inputs"], projection_source_block["promotion_package_manifest_id"])
    package_plan = provenance.setdefault("package_plan", {})
    package_plan["producer_kind"] = "iqk_compact_expert_repack"
    package_plan["producer_reference"] = selection["artifact_id"]
    # A drafter can be attached again after compaction. Its previous declaration
    # is bound to the source package identity and must not survive this rewrite.
    out.pop("drafter", None)
    out.pop("artifact_id", None)
    out.pop("created_at", None)
    return out


def _expert_record_cell(record: dict, where: str) -> tuple[int, str]:
    layer = record.get("layer_index")
    projection = record.get("projection")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise IQKReapError(
            f"{where} expert {record.get('source_name')} has no integer layer")
    if projection not in _PLAN_PROJECTIONS:
        raise IQKReapError(
            f"{where} expert {record.get('source_name')} has projection "
            f"{projection!r}")
    return layer, projection


def _expert_records_by_cell(payload: dict | None, where: str) -> dict[tuple[int, str], dict]:
    if not isinstance(payload, dict):
        raise IQKReapError(f"{where} is not an artifact object")
    field = "allocation" if payload.get("artifact_kind") == "package_plan" else "tensors"
    records: dict[tuple[int, str], dict] = {}
    for record in payload.get(field, []):
        if record.get("kind") != "expert":
            continue
        if record.get("format") != IQK_CODEC:
            raise IQKReapError(
                f"{where} expert {record.get('source_name')} is not IQ_K")
        cell = _expert_record_cell(record, where)
        if cell in records:
            raise IQKReapError(
                f"{where} carries duplicate expert cell {cell[0]} {cell[1]}")
        records[cell] = record
    if not records:
        raise IQKReapError(f"{where} carries no IQ_K expert cells")
    return records


def rewrite_expert_hotlist(hotlist: dict, selection: dict) -> dict:
    """Translate original expert IDs in a hotlist into compact per-layer IDs."""
    retained = validate_expert_selection(selection)
    raw_layers = hotlist.get("layers")
    if not isinstance(raw_layers, dict):
        raise IQKReapError("expert hotlist carries no layer map")
    if set(raw_layers) != {str(layer) for layer in retained}:
        raise IQKReapError("expert hotlist layers do not match the selection")
    layers: dict[str, dict[str, int]] = {}
    for layer, ids in retained.items():
        ranked = raw_layers[str(layer)]
        if not isinstance(ranked, dict):
            raise IQKReapError(f"expert hotlist layer {layer} is not an object")
        original_to_compact = {original: compact for compact, original in enumerate(ids)}
        translated: dict[str, int] = {}
        for raw_expert, raw_count in ranked.items():
            try:
                original = int(raw_expert)
                count = int(raw_count)
            except (TypeError, ValueError) as exc:
                raise IQKReapError(
                    f"invalid hotlist entry layer {layer}: {raw_expert!r}={raw_count!r}") \
                    from exc
            if str(original) != str(raw_expert) or not 0 <= original < 256 or count < 0:
                raise IQKReapError(
                    f"invalid hotlist entry layer {layer}: {raw_expert!r}={raw_count!r}")
            compact = original_to_compact.get(original)
            if compact is not None:
                translated[str(compact)] = count
        layers[str(layer)] = translated
    source = dict(hotlist.get("source") or {})
    source.update({
        "expert_id_space": "compact_per_layer",
        "source_expert_id_space": "original",
        "source_selection_artifact_id": selection["artifact_id"],
    })
    return {
        "version": hotlist.get("version", 1),
        "kind": "expert_hotlist",
        "source": source,
        "layers": layers,
    }


def compact_iqk_package(
    package_dir: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    *,
    promotion_package_dir: str | Path | None = None,
    projection_sources: Mapping[int | str, Mapping[str, str]] | None = None,
    source_verifier: SourceVerifier | None = None,
    output_verifier: OutputVerifier | None = None,
    row_verifier: RowVerifier | None = None,
    identity_fn: Callable[[Path], dict] = file_identity,
    copy_fn: Callable[[Path, Path], object] = shutil.copy2,
    include_drafter: bool = True,
    verbose: bool = False,
) -> dict:
    """Write a new physically compact package and return its transform report.

    The optional promotion package supplies complete projection cells at a
    higher-rate IQ_K member. ``projection_sources`` must then name ``base`` or
    ``promotion`` for every layer's gate, up, and down cell. The resulting
    bundle still declares one member per projection, so the existing pooled
    runtime and its single routed graph seam remain unchanged.

    Package and byte-comparison gates are injectable so synthetic tests can add
    deliberate faults without duplicating the public verification surface.
    Passing ``None`` selects each strict production verifier; it does not
    disable a gate. The base package's optional drafter is copied and rebound by
    default; callers can omit it without changing the compact trunk.
    """
    source = Path(package_dir)
    promotion = (
        Path(promotion_package_dir) if promotion_package_dir is not None else None)
    selection_file = Path(selection_path)
    output = Path(output_dir)
    log = (lambda message: print(message, flush=True)) if verbose else (lambda message: None)

    if not source.is_dir():
        raise IQKReapError(f"source package directory does not exist: {source}")
    if (promotion is None) != (projection_sources is None):
        raise IQKReapError(
            "promotion_package_dir and projection_sources must be provided together")
    if promotion is not None and not promotion.is_dir():
        raise IQKReapError(
            f"promotion package directory does not exist: {promotion}")
    if output.exists():
        raise IQKReapError(f"output must not exist: {output}")
    try:
        source_resolved = source.resolve(strict=True)
        output_resolved = output.resolve(strict=False)
        output_resolved.relative_to(source_resolved)
    except ValueError:
        pass
    else:
        raise IQKReapError("output must not be inside the source package")
    if promotion is not None:
        try:
            promotion_resolved = promotion.resolve(strict=True)
            output_resolved.relative_to(promotion_resolved)
        except ValueError:
            pass
        else:
            raise IQKReapError("output must not be inside the promotion package")
    if not output.parent.is_dir():
        raise IQKReapError(f"output parent does not exist: {output.parent}")

    manifest = read_artifact(source / MANIFEST_NAME)
    plan = read_artifact(source / PACKAGE_PLAN_NAME)
    promotion_manifest = (
        read_artifact(promotion / MANIFEST_NAME) if promotion is not None else None)
    promotion_plan = (
        read_artifact(promotion / PACKAGE_PLAN_NAME) if promotion is not None else None)
    selection = read_artifact(selection_file)
    if selection["source_package_manifest_id"] != manifest.get("artifact_id"):
        raise IQKReapError(
            "selection source package does not match the package manifest: "
            f"{selection['source_package_manifest_id']} != {manifest.get('artifact_id')}")
    if (manifest.get("architecture") or {}).get("family") != "deepseek_v4_flash":
        raise IQKReapError(f"{source} is not a DeepSeek-V4 package")
    config = (manifest.get("architecture") or {}).get("config") or {}
    source_count = _declared_expert_count(config)
    if source_count != selection.get("source_num_experts"):
        raise IQKReapError(
            f"package declares {source_count} experts, selection declares "
            f"{selection.get('source_num_experts')}")
    package_top_k = config.get("num_experts_per_tok")
    if (
        isinstance(package_top_k, bool)
        or not isinstance(package_top_k, int)
        or package_top_k != selection.get("top_k")
    ):
        raise IQKReapError(
            f"package top-k {package_top_k!r} != selection "
            f"top_k {selection.get('top_k')!r}")

    declared_names = _declared_shard_names(manifest)
    layout = _scan_iqk_bundles(source, declared_names, source_count)
    retained = validate_expert_selection(selection, expected_layers=layout)
    normalized_sources = None
    projection_source_block = None
    promotion_layout = None
    if promotion is not None:
        assert promotion_manifest is not None
        assert promotion_plan is not None
        promotion_names = _declared_shard_names(promotion_manifest)
        promotion_layout = _scan_iqk_bundles(
            promotion, promotion_names, source_count)
        normalized_sources = validate_projection_source_map(
            projection_sources or {}, expected_layers=layout)
        _validate_promotion_package(
            manifest,
            plan,
            declared_names,
            layout,
            promotion_manifest,
            promotion_plan,
            promotion_names,
            promotion_layout,
            normalized_sources,
        )
        projection_source_block = projection_source_manifest_block(
            normalized_sources,
            base_manifest_id=manifest["artifact_id"],
            promotion_manifest_id=promotion_manifest["artifact_id"],
        )
    transforms = _router_transforms(manifest, retained, source_count)
    undeclared_router_shards = sorted(set(transforms) - set(declared_names))
    if undeclared_router_shards:
        raise IQKReapError(
            f"router tensors reference undeclared shards: {undeclared_router_shards}")
    source_gate = source_verifier or _verify_package
    output_gate = output_verifier or _verify_package
    row_gate = row_verifier or _compare_byte_spans
    source_gate(source, manifest)
    if promotion is not None:
        assert promotion_manifest is not None
        source_gate(promotion, promotion_manifest)
    log(
        f"[1/5] source {manifest['artifact_id']} verified; "
        f"{len(layout)} routed layers selected"
        + (
            f"; promotion {promotion_manifest['artifact_id']} verified"
            if promotion_manifest is not None else ""))

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.reap-", dir=output.parent))
    try:
        bundle_reports: list[dict] = []
        router_reports: list[dict] = []
        bundle_shards = {record["shard"]: layer for layer, record in layout.items()}
        for name in declared_names:
            src = _resolved(source, name)
            dst = _resolved(stage, name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if name in bundle_shards:
                layer = bundle_shards[name]
                if promotion is None:
                    bundle_reports.append(
                        _compact_bundle_shard(
                            src,
                            dst,
                            layer,
                            retained[layer],
                            source_count,
                            row_verifier=row_gate,
                        ))
                else:
                    assert promotion_layout is not None
                    assert normalized_sources is not None
                    promotion_record = promotion_layout[layer]
                    bundle_reports.append(
                        _compact_mixed_bundle_shard(
                            base=src,
                            promotion=_resolved(
                                promotion, promotion_record["shard"]),
                            output=dst,
                            layer=layer,
                            retained=retained[layer],
                            source_count=source_count,
                            base_record=layout[layer],
                            promotion_record=promotion_record,
                            projection_sources=normalized_sources[layer],
                            row_verifier=row_gate,
                        ))
            elif name in transforms:
                router_reports.append(_rewrite_safetensors_rows(
                    src,
                    dst,
                    transforms[name],
                    source_count,
                    row_verifier=row_gate,
                ))
            else:
                copy_fn(src, dst)
        routed_bytes = sum(report["output_payload_bytes"] for report in bundle_reports)
        log(
            f"[2/5] copied {sum(len(ids) for ids in retained.values())} retained "
            f"expert rows ({routed_bytes} bytes)")

        write_artifact(
            stage / EXPERT_SELECTION_NAME,
            selection,
            created_at=selection.get("created_at"),
        )
        plan_payload = rewrite_package_plan_for_selection(
            plan,
            selection,
            projection_source_block=projection_source_block,
            promotion_plan=promotion_plan,
            routed_bytes=routed_bytes,
        )
        plan_id = write_artifact(
            stage / PACKAGE_PLAN_NAME,
            plan_payload,
            created_at=plan.get("created_at"),
        )
        shard_identities = [identity_fn(stage / name) for name in declared_names]
        package_identities = [
            *shard_identities,
            identity_fn(stage / EXPERT_SELECTION_NAME),
        ]
        trunk_manifest_payload = rewrite_package_manifest_for_selection(
            manifest,
            selection,
            plan_id=plan_id,
            files=package_identities,
            projection_source_block=projection_source_block,
            promotion_manifest=promotion_manifest,
        )
        compact_trunk_manifest_id = write_artifact(
            stage / MANIFEST_NAME,
            trunk_manifest_payload,
            created_at=manifest.get("created_at"),
        )
        compact_trunk_manifest = read_artifact(stage / MANIFEST_NAME)
        drafter = None
        if include_drafter:
            drafter = _copy_rebound_drafter(
                source,
                stage,
                manifest,
                compact_trunk_manifest_id=compact_trunk_manifest_id,
                identity_fn=identity_fn,
                copy_fn=copy_fn,
            )
        final_manifest = json.loads(json.dumps(compact_trunk_manifest))
        if drafter is not None:
            final_manifest["drafter"] = drafter
        final_manifest.setdefault("provenance", {})[
            "compact_trunk_manifest_id"] = compact_trunk_manifest_id
        final_manifest.pop("artifact_id", None)
        final_manifest.pop("created_at", None)
        manifest_id = write_artifact(
            stage / MANIFEST_NAME,
            final_manifest,
            created_at=manifest.get("created_at"),
        )
        log(
            f"[3/5] plan {plan_id}; compact trunk {compact_trunk_manifest_id}; "
            f"manifest {manifest_id}")

        _copy_declared_furniture(source, stage, manifest, copy_fn)
        hotlist_path = _resolved(source, "expert_hotlist.json")
        if hotlist_path.is_file():
            hotlist = rewrite_expert_hotlist(json.loads(hotlist_path.read_text()), selection)
            (stage / "expert_hotlist.json").write_text(
                json.dumps(hotlist, indent=2, sort_keys=True))
        _rewrite_compat_sidecars(source, stage, read_artifact(stage / MANIFEST_NAME))

        report = {
            "schema": "moespresso-ds4-iqk-reap-report-v1",
            "source_package": source.name,
            "output_package": output.name,
            "source_package_manifest_id": manifest["artifact_id"],
            "source_package_plan_id": plan["artifact_id"],
            "promotion_package": promotion.name if promotion is not None else None,
            "promotion_package_manifest_id": (
                promotion_manifest["artifact_id"]
                if promotion_manifest is not None else None
            ),
            "expert_projection_sources": projection_source_block,
            "source_selection_artifact_id": selection["artifact_id"],
            "package_manifest_id": manifest_id,
            "compact_trunk_manifest_id": compact_trunk_manifest_id,
            "package_plan_id": plan_id,
            "source_num_experts": source_count,
            "top_k": selection["top_k"],
            "layers": {
                str(layer): {
                    "num_experts": len(ids),
                    "source_expert_ids_sha256": _ids_sha256(ids),
                }
                for layer, ids in retained.items()
            },
            "routed_bytes": routed_bytes,
            "bundle_shards": bundle_reports,
            "router_shards": router_reports,
            "byte_comparison": {
                "bundle_bytes": sum(
                    row["byte_comparison"]["bytes"] for row in bundle_reports),
                "router_bytes": sum(
                    row["byte_comparison"]["bytes"] for row in router_reports),
                "scope": "every retained bundle row and gathered router row",
            },
            "drafter": (
                {
                    "family": drafter["family"],
                    "sidecar_artifact_id": drafter["sidecar_artifact_id"],
                    "source_package_manifest_id": compact_trunk_manifest_id,
                }
                if drafter is not None else None
            ),
        }
        (stage / REAP_REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True))
        output_gate(stage, read_artifact(stage / MANIFEST_NAME))
        log("[4/5] compact output verified")
        os.replace(stage, output)
        log(f"[5/5] published {output}")
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _scan_iqk_bundles(
    package_dir: Path,
    declared_names: Sequence[str],
    source_count: int,
) -> dict[int, dict]:
    layers: dict[int, dict] = {}
    for name in declared_names:
        path = _resolved(package_dir, name)
        header, data_start = _read_safetensors_header(path)
        metadata = header.get("__metadata__") or {}
        if METADATA_KEY not in metadata:
            continue
        decoded = decode_bundle_metadata(metadata[METADATA_KEY])
        if len(decoded) != 1:
            raise IQKReapError(
                f"{name}: compact repack requires one routed layer per bundle shard")
        (layer, geometry), = decoded.items()
        if layer in layers:
            raise IQKReapError(f"routed layer {layer} appears in more than one shard")
        keys = [key for key in header if key != "__metadata__"]
        if len(keys) != 1:
            raise IQKReapError(f"{name}: bundle shard carries {len(keys)} tensors")
        key = keys[0]
        row_bytes = int(geometry.get("row_bytes", 0))
        if int(geometry.get("num_experts", 0)) != source_count or row_bytes <= 0:
            raise IQKReapError(
                f"{name}: bundle geometry is not {source_count} experts with a "
                "positive row stride")
        entry = header[key]
        if entry.get("dtype") != "U8" or entry.get("shape") != [source_count, row_bytes]:
            raise IQKReapError(
                f"{name}: bundle tensor is not U8[{source_count}, {row_bytes}]")
        if entry.get("data_offsets") != [0, source_count * row_bytes]:
            raise IQKReapError(f"{name}: bundle tensor offsets do not tile its data area")
        projections = geometry.get("projections") or {}
        if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
            raise IQKReapError(f"{name}: bundle does not cover gate/up/down projections")
        for projection, params in projections.items():
            if params.get("codec") != IQK_CODEC:
                raise IQKReapError(f"{name}: {projection} is not IQ_K")
            if params.get("layout") != IQK_LAYOUT_IQK_RELAYOUT:
                raise IQKReapError(
                    f"{name}: {projection} layout {params.get('layout')!r} is not "
                    f"{IQK_LAYOUT_IQK_RELAYOUT!r}")
        layers[int(layer)] = {
            "shard": name,
            "key": key,
            "geometry": geometry,
            "data_start": data_start,
        }
    if not layers:
        raise IQKReapError("package has no relayouted IQ_K expert bundles")
    return dict(sorted(layers.items()))


def _validate_promotion_package(
    base_manifest: dict,
    base_plan: dict,
    base_names: Sequence[str],
    base_layout: Mapping[int, dict],
    promotion_manifest: dict,
    promotion_plan: dict,
    promotion_names: Sequence[str],
    promotion_layout: Mapping[int, dict],
    projection_sources: Mapping[int, Mapping[str, str]],
) -> None:
    """Require two packages to differ only in compatible expert cell bytes."""
    manifest_fields = (
        "subject",
        "architecture",
        "tokenizer",
        "agentic_profile",
        "package_format",
        "package_format_version",
        "required_ops",
        "optimized_kernels_expected",
    )
    for field in manifest_fields:
        if base_manifest.get(field) != promotion_manifest.get(field):
            raise IQKReapError(
                f"promotion package {field} does not match the base package")
    if list(base_names) != list(promotion_names):
        raise IQKReapError(
            "promotion package shard names do not match the base package")
    if set(base_layout) != set(promotion_layout):
        raise IQKReapError(
            "promotion package routed layers do not match the base package")

    base_bundle_names = {record["shard"] for record in base_layout.values()}
    promotion_bundle_names = {
        record["shard"] for record in promotion_layout.values()
    }
    if base_bundle_names != promotion_bundle_names:
        raise IQKReapError(
            "promotion package expert bundle shards do not match the base package")
    base_files = _manifest_file_identities(base_manifest)
    promotion_files = _manifest_file_identities(promotion_manifest)
    for name in sorted(set(base_names) - base_bundle_names):
        if base_files.get(name) != promotion_files.get(name):
            raise IQKReapError(
                f"promotion package non-expert shard {name!r} differs from the base")

    base_manifest_cells = _expert_records_by_cell(
        base_manifest, "base package manifest")
    promotion_manifest_cells = _expert_records_by_cell(
        promotion_manifest, "promotion package manifest")
    base_plan_cells = _expert_records_by_cell(base_plan, "base package plan")
    promotion_plan_cells = _expert_records_by_cell(
        promotion_plan, "promotion package plan")
    cell_sets = (
        set(base_manifest_cells),
        set(promotion_manifest_cells),
        set(base_plan_cells),
        set(promotion_plan_cells),
    )
    if any(cells != cell_sets[0] for cells in cell_sets[1:]):
        raise IQKReapError(
            "base and promotion package expert-cell sets do not match")

    base_nonexpert = [
        record for record in base_manifest.get("tensors", [])
        if record.get("kind") != "expert"
    ]
    promotion_nonexpert = [
        record for record in promotion_manifest.get("tensors", [])
        if record.get("kind") != "expert"
    ]
    if base_nonexpert != promotion_nonexpert:
        raise IQKReapError(
            "promotion package non-expert tensor declarations differ from the base")
    base_nonexpert_plan = [
        record for record in base_plan.get("allocation", [])
        if record.get("kind") != "expert"
    ]
    promotion_nonexpert_plan = [
        record for record in promotion_plan.get("allocation", [])
        if record.get("kind") != "expert"
    ]
    if base_nonexpert_plan != promotion_nonexpert_plan:
        raise IQKReapError(
            "promotion package non-expert plan allocations differ from the base")

    for layer in sorted(base_layout):
        base_record = base_layout[layer]
        promotion_record = promotion_layout[layer]
        if (
            base_record["shard"] != promotion_record["shard"]
            or base_record["key"] != promotion_record["key"]
        ):
            raise IQKReapError(
                f"promotion package layer {layer} bundle location differs from the base")
        for bundle_projection in BUNDLE_PROJECTIONS:
            projection = _BUNDLE_TO_PLAN_PROJECTION[bundle_projection]
            cell = (layer, projection)
            _validate_expert_record_identity(
                base_manifest_cells[cell],
                promotion_manifest_cells[cell],
                where=f"manifest expert cell {layer} {projection}",
            )
            _validate_expert_record_identity(
                base_plan_cells[cell],
                promotion_plan_cells[cell],
                where=f"plan expert cell {layer} {projection}",
            )
            base_geo = base_record["geometry"]["projections"][bundle_projection]
            promotion_geo = promotion_record["geometry"]["projections"][
                bundle_projection]
            _validate_cell_metadata(
                base_geo,
                base_manifest_cells[cell],
                base_plan_cells[cell],
                where=f"base layer {layer} {projection}",
            )
            _validate_cell_metadata(
                promotion_geo,
                promotion_manifest_cells[cell],
                promotion_plan_cells[cell],
                where=f"promotion layer {layer} {projection}",
            )
            if projection_sources[layer][projection] != "promotion":
                continue
            base_blocks = base_geo["blocks"]
            promotion_blocks = promotion_geo["blocks"]
            if (
                base_geo.get("in_features") != promotion_geo.get("in_features")
                or base_blocks.get("shape", [None])[0]
                != promotion_blocks.get("shape", [None])[0]
            ):
                raise IQKReapError(
                    f"promotion layer {layer} {projection} shape differs from the base")
            if int(promotion_blocks.get("nbytes", 0)) <= int(
                base_blocks.get("nbytes", 0)
            ):
                raise IQKReapError(
                    f"promotion layer {layer} {projection} is not higher-rate than "
                    "the base cell")


def _manifest_file_identities(manifest: dict) -> dict[str, dict]:
    identities: dict[str, dict] = {}
    for identity in manifest.get("files", []):
        name = identity.get("path") if isinstance(identity, dict) else None
        if not isinstance(name, str) or not name:
            raise IQKReapError("package manifest file identity has no path")
        if name in identities:
            raise IQKReapError(f"package manifest declares file {name!r} twice")
        identities[name] = dict(identity)
    return identities


def _validate_expert_record_identity(base: dict, promotion: dict, *, where: str) -> None:
    for key in _EXPERT_IDENTITY_KEYS:
        if base.get(key) != promotion.get(key):
            raise IQKReapError(f"{where} differs at {key}")
    for key in ("shard", "key_prefix"):
        if key in base or key in promotion:
            if base.get(key) != promotion.get(key):
                raise IQKReapError(f"{where} differs at {key}")


def _validate_cell_metadata(
    geometry: dict,
    manifest_record: dict,
    plan_record: dict,
    *,
    where: str,
) -> None:
    member = geometry.get("iqk_codec")
    if (
        geometry.get("codec") != IQK_CODEC
        or not isinstance(member, str)
        or normalize_iqk_layout(geometry.get("layout"))
        != IQK_LAYOUT_IQK_RELAYOUT
    ):
        raise IQKReapError(f"{where} is not a serving IQ_K relayout cell")
    params = manifest_record.get("format_params") or {}
    if (
        manifest_record.get("format") != IQK_CODEC
        or params.get("iqk_codec") != member
        or normalize_iqk_layout(params.get("layout"))
        != IQK_LAYOUT_IQK_RELAYOUT
    ):
        raise IQKReapError(f"{where} manifest member disagrees with its bundle")
    if (
        plan_record.get("format") != IQK_CODEC
        or plan_record.get("iqk_codec") != member
        or normalize_iqk_layout(plan_record.get("layout"))
        != IQK_LAYOUT_IQK_RELAYOUT
    ):
        raise IQKReapError(f"{where} plan member disagrees with its bundle")


def _router_transforms(
    manifest: dict,
    retained: Mapping[int, Sequence[int]],
    source_count: int,
) -> dict[str, dict[str, tuple[int, ...]]]:
    by_layer_role: dict[tuple[int, str], dict] = {}
    for tensor in manifest.get("tensors", []):
        role = tensor.get("role")
        if role not in _ROUTER_ROLES:
            continue
        layer = tensor.get("layer_index")
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise IQKReapError(f"router tensor {tensor.get('source_name')} has no layer")
        key = (layer, role)
        if key in by_layer_role:
            raise IQKReapError(f"duplicate manifest router tensor for layer {layer} {role}")
        by_layer_role[key] = tensor

    transforms: dict[str, dict[str, tuple[int, ...]]] = {}
    identity = tuple(range(source_count))
    for layer, ids in retained.items():
        if layer in HASH_LAYERS:
            continue
        for role in sorted(_ROUTER_ROLES):
            tensor = by_layer_role.get((layer, role))
            if tensor is None:
                raise IQKReapError(f"score-routed layer {layer} has no {role} tensor")
            if tuple(ids) == identity:
                continue
            shard = tensor.get("shard")
            key = tensor.get("key_prefix")
            if not isinstance(shard, str) or not isinstance(key, str):
                raise IQKReapError(f"router tensor layer {layer} {role} has no location")
            existing = transforms.setdefault(shard, {}).get(key)
            if existing is not None and existing != tuple(ids):
                raise IQKReapError(f"conflicting row selections for {shard}:{key}")
            transforms[shard][key] = tuple(ids)
    return transforms


def _compact_bundle_shard(
    source: Path,
    output: Path,
    layer: int,
    retained: Sequence[int],
    source_count: int,
    *,
    row_verifier: RowVerifier,
) -> dict:
    header, data_start = _read_safetensors_header(source)
    metadata = dict(header.get("__metadata__") or {})
    layers = decode_bundle_metadata(metadata[METADATA_KEY])
    geometry = json.loads(json.dumps(layers[layer]))
    row_bytes = int(geometry["row_bytes"])
    geometry["num_experts"] = len(retained)
    metadata[METADATA_KEY] = encode_bundle_metadata({layer: geometry})
    key = next(key for key in header if key != "__metadata__")
    out_header = {
        "__metadata__": metadata,
        key: {
            "dtype": "U8",
            "shape": [len(retained), row_bytes],
            "data_offsets": [0, len(retained) * row_bytes],
        },
    }
    with open(source, "rb") as source_handle, open(output, "wb") as output_handle:
        _write_safetensors_header(output_handle, out_header)
        for expert in retained:
            _copy_at(
                source_handle,
                output_handle,
                data_start + expert * row_bytes,
                row_bytes,
            )
        output_handle.flush()
        os.fsync(output_handle.fileno())
    output_data_start = _safetensors_data_start(output)
    spans = [
        (
            data_start + original * row_bytes,
            output_data_start + compact * row_bytes,
            row_bytes,
        )
        for compact, original in enumerate(retained)
    ]
    byte_comparison = row_verifier(source, output, spans)
    return {
        "shard": output.name,
        "layer": layer,
        "source_experts": source_count,
        "output_experts": len(retained),
        "row_bytes": row_bytes,
        "output_payload_bytes": len(retained) * row_bytes,
        "byte_comparison": byte_comparison,
    }


def _compact_mixed_bundle_shard(
    *,
    base: Path,
    promotion: Path,
    output: Path,
    layer: int,
    retained: Sequence[int],
    source_count: int,
    base_record: dict,
    promotion_record: dict,
    projection_sources: Mapping[str, str],
    row_verifier: RowVerifier,
) -> dict:
    """Copy selected projection spans from two homogeneous source bundles."""
    base_header, base_data_start = _read_safetensors_header(base)
    promotion_header, promotion_data_start = _read_safetensors_header(promotion)
    base_key = base_record["key"]
    promotion_key = promotion_record["key"]
    if base_key not in base_header or promotion_key not in promotion_header:
        raise IQKReapError(f"layer {layer}: a source bundle tensor is missing")

    base_geometry = base_record["geometry"]
    promotion_geometry = promotion_record["geometry"]
    geometry = json.loads(json.dumps(base_geometry))
    geometry["num_experts"] = len(retained)
    components: list[dict] = []
    output_offset = 0
    for bundle_projection in BUNDLE_PROJECTIONS:
        projection = _BUNDLE_TO_PLAN_PROJECTION[bundle_projection]
        source_name = projection_sources[projection]
        source_record = (
            promotion_record if source_name == "promotion" else base_record)
        source_geometry = source_record["geometry"]
        params = json.loads(json.dumps(
            source_geometry["projections"][bundle_projection]))
        blocks = params["blocks"]
        source_offset = int(blocks["offset"])
        nbytes = int(blocks["nbytes"])
        if nbytes <= 0:
            raise IQKReapError(
                f"layer {layer} {projection}: source component is empty")
        blocks["offset"] = output_offset
        geometry["projections"][bundle_projection] = params
        components.append({
            "projection": projection,
            "source": source_name,
            "source_offset": source_offset,
            "output_offset": output_offset,
            "nbytes": nbytes,
        })
        output_offset += nbytes
    geometry["row_bytes"] = output_offset

    metadata = dict(base_header.get("__metadata__") or {})
    metadata[METADATA_KEY] = encode_bundle_metadata({layer: geometry})
    out_header = {
        "__metadata__": metadata,
        base_key: {
            "dtype": "U8",
            "shape": [len(retained), output_offset],
            "data_offsets": [0, len(retained) * output_offset],
        },
    }
    base_row_bytes = int(base_geometry["row_bytes"])
    promotion_row_bytes = int(promotion_geometry["row_bytes"])
    with (
        open(base, "rb") as base_handle,
        open(promotion, "rb") as promotion_handle,
        open(output, "wb") as output_handle,
    ):
        handles = {"base": base_handle, "promotion": promotion_handle}
        data_starts = {
            "base": base_data_start,
            "promotion": promotion_data_start,
        }
        row_bytes = {"base": base_row_bytes, "promotion": promotion_row_bytes}
        _write_safetensors_header(output_handle, out_header)
        for expert in retained:
            for component in components:
                source_name = component["source"]
                _copy_at(
                    handles[source_name],
                    output_handle,
                    data_starts[source_name]
                    + expert * row_bytes[source_name]
                    + component["source_offset"],
                    component["nbytes"],
                )
        output_handle.flush()
        os.fsync(output_handle.fileno())

    output_data_start = _safetensors_data_start(output)
    source_paths = {"base": base, "promotion": promotion}
    data_starts = {"base": base_data_start, "promotion": promotion_data_start}
    row_bytes = {"base": base_row_bytes, "promotion": promotion_row_bytes}
    comparisons: dict[str, dict] = {}
    for source_name in ("base", "promotion"):
        spans = []
        for compact, expert in enumerate(retained):
            for component in components:
                if component["source"] != source_name:
                    continue
                spans.append((
                    data_starts[source_name]
                    + expert * row_bytes[source_name]
                    + component["source_offset"],
                    output_data_start
                    + compact * output_offset
                    + component["output_offset"],
                    component["nbytes"],
                ))
        if spans:
            comparisons[source_name] = row_verifier(
                source_paths[source_name], output, spans)
    compared_bytes = sum(result["bytes"] for result in comparisons.values())
    if compared_bytes != len(retained) * output_offset:
        raise IQKReapError(
            f"layer {layer}: compared {compared_bytes} output bytes, expected "
            f"{len(retained) * output_offset}")
    return {
        "shard": output.name,
        "layer": layer,
        "source_experts": source_count,
        "output_experts": len(retained),
        "row_bytes": output_offset,
        "output_payload_bytes": len(retained) * output_offset,
        "projection_sources": dict(projection_sources),
        "byte_comparison": {
            "spans": sum(result["spans"] for result in comparisons.values()),
            "bytes": compared_bytes,
            "result": "identical",
            "sources": comparisons,
        },
    }


def _rewrite_safetensors_rows(
    source: Path,
    output: Path,
    selections: Mapping[str, Sequence[int]],
    source_count: int,
    *,
    row_verifier: RowVerifier,
) -> dict:
    header, data_start = _read_safetensors_header(source)
    tensor_keys = sorted(key for key in header if key != "__metadata__")
    unknown = sorted(set(selections) - set(tensor_keys))
    if unknown:
        raise IQKReapError(f"{source.name}: router tensor(s) absent: {unknown}")

    out_header: dict[str, dict] = {}
    if "__metadata__" in header:
        out_header["__metadata__"] = dict(header["__metadata__"])
    output_sizes: dict[str, int] = {}
    cursor = 0
    for key in tensor_keys:
        entry = header[key]
        start, end = _tensor_offsets(source, key, entry)
        nbytes = end - start
        out_entry = json.loads(json.dumps(entry))
        if key in selections:
            shape = out_entry.get("shape")
            if not isinstance(shape, list) or not shape or shape[0] != source_count:
                raise IQKReapError(
                    f"{source.name}:{key} is not axis-0 expert geometry "
                    f"[{source_count}, ...]")
            if nbytes % source_count:
                raise IQKReapError(
                    f"{source.name}:{key} byte size is not divisible by {source_count}")
            expected = _shape_nbytes(shape, out_entry.get("dtype"))
            if expected != nbytes:
                raise IQKReapError(
                    f"{source.name}:{key} shape describes {expected} bytes, file has {nbytes}")
            nbytes = (nbytes // source_count) * len(selections[key])
            out_entry["shape"][0] = len(selections[key])
        out_entry["data_offsets"] = [cursor, cursor + nbytes]
        output_sizes[key] = nbytes
        cursor += nbytes
        out_header[key] = out_entry

    with open(source, "rb") as source_handle, open(output, "wb") as output_handle:
        _write_safetensors_header(output_handle, out_header)
        for key in tensor_keys:
            entry = header[key]
            start, end = _tensor_offsets(source, key, entry)
            if key not in selections:
                _copy_at(source_handle, output_handle, data_start + start, end - start)
                continue
            row_bytes = (end - start) // source_count
            for expert in selections[key]:
                _copy_at(
                    source_handle,
                    output_handle,
                    data_start + start + expert * row_bytes,
                    row_bytes,
                )
        output_handle.flush()
        os.fsync(output_handle.fileno())
    expected_size = _safetensors_data_start(output) + sum(output_sizes.values())
    if output.stat().st_size != expected_size:
        raise IQKReapError(
            f"{output.name}: wrote {output.stat().st_size} bytes, expected {expected_size}")
    output_header, output_data_start = _read_safetensors_header(output)
    spans: list[tuple[int, int, int]] = []
    for key in sorted(selections):
        source_start, source_end = _tensor_offsets(source, key, header[key])
        output_start, _output_end = _tensor_offsets(output, key, output_header[key])
        row_bytes = (source_end - source_start) // source_count
        for compact, original in enumerate(selections[key]):
            spans.append((
                data_start + source_start + original * row_bytes,
                output_data_start + output_start + compact * row_bytes,
                row_bytes,
            ))
    return {
        "shard": output.name,
        "tensors": sorted(selections),
        "byte_comparison": row_verifier(source, output, spans),
    }


def _copy_rebound_drafter(
    source: Path,
    output: Path,
    source_manifest: dict,
    *,
    compact_trunk_manifest_id: str,
    identity_fn: Callable[[Path], dict],
    copy_fn: Callable[[Path, Path], object],
) -> dict | None:
    """Copy a bundled DSpark sidecar and bind it to the compact trunk identity."""
    declared = source_manifest.get("drafter")
    if declared is None:
        return None
    if not isinstance(declared, dict) or declared.get("family") != "dspark":
        raise IQKReapError("compact repack supports only a bundled DSpark drafter")

    from moespresso.package.deepseek_v4.dspark_bundle import (
        SIDECAR_MANIFEST_NAME,
        drafter_component,
        read_valid_sidecar_manifest,
    )

    if declared.get("manifest_path") != SIDECAR_MANIFEST_NAME:
        raise IQKReapError("source DSpark component has an unexpected manifest path")
    sidecar_manifest = read_valid_sidecar_manifest(source)
    if declared.get("sidecar_artifact_id") != sidecar_manifest.get("artifact_id"):
        raise IQKReapError(
            "source DSpark component and sidecar manifest artifact ids disagree")
    hashed_names = sorted((sidecar_manifest.get("provenance") or {}).get("file_sha256", {}))
    sidecar_names = [SIDECAR_MANIFEST_NAME, *hashed_names]
    declared_names: list[str] = []
    for identity in declared.get("files", []):
        name = identity.get("path") if isinstance(identity, dict) else None
        if not isinstance(name, str):
            raise IQKReapError("source DSpark component has an invalid file identity")
        declared_names.append(name)
    declared_names.sort()
    if declared_names != sorted(sidecar_names):
        raise IQKReapError(
            "source DSpark component file list does not match its sidecar manifest")
    for name in sidecar_names:
        src = _resolved(source, name)
        dst = _resolved(output, name)
        if dst.exists():
            raise IQKReapError(f"DSpark file collides with compact package file {name!r}")
        copy_fn(src, dst)
    identities = [identity_fn(output / name) for name in sidecar_names]
    component = drafter_component(
        sidecar_manifest,
        identities,
        source_package_manifest_id=compact_trunk_manifest_id,
    )
    original_sidecar_id = (
        (declared.get("provenance") or {}).get("source_sidecar_artifact_id")
        or sidecar_manifest["artifact_id"]
    )
    component["provenance"]["source_sidecar_artifact_id"] = original_sidecar_id
    return component


def _copy_declared_furniture(
    source: Path,
    output: Path,
    manifest: dict,
    copy_fn: Callable[[Path, Path], object],
) -> None:
    identities: list[dict] = []
    tokenizer = manifest.get("tokenizer") or {}
    identities.extend(tokenizer.get("files") or [])
    agentic = manifest.get("agentic_profile")
    if isinstance(agentic, dict):
        identities.append(agentic)
    for identity in identities:
        name = identity.get("path")
        if not isinstance(name, str):
            raise IQKReapError("manifest furniture identity has no path")
        src = _resolved(source, name)
        dst = _resolved(output, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            copy_fn(src, dst)
    for name in _OPTIONAL_SOURCE_FURNITURE:
        src = _resolved(source, name)
        if src.is_file():
            copy_fn(src, output / name)


def _rewrite_compat_sidecars(source: Path, output: Path, manifest: dict) -> None:
    config_path = _resolved(source, "config.json")
    jang_path = _resolved(source, "jang_config.json")
    if not (config_path.is_file() and jang_path.is_file()):
        return
    from moespresso.package.sidecars import build_sidecars

    seed = 42
    try:
        raw_seed = json.loads(jang_path.read_text()).get("mxtq_seed")
        if isinstance(raw_seed, int):
            seed = raw_seed
    except (OSError, ValueError):
        pass
    config, jang = build_sidecars(manifest, seed=seed)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "jang_config.json").write_text(json.dumps(jang, indent=2))


def _verify_package(package_dir: Path, manifest: dict) -> None:
    from moespresso.runtime.verify import verify_generated_sidecars, verify_package

    issues = verify_package(manifest, package_dir)
    if (package_dir / "config.json").is_file() or (package_dir / "jang_config.json").is_file():
        issues.extend(verify_generated_sidecars(manifest, package_dir))
    blocking = [issue for issue in issues if issue.blocking]
    if blocking:
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in blocking[:8])
        raise IQKReapError(f"package verification failed: {detail}")


def _declared_shard_names(manifest: dict) -> list[str]:
    names: list[str] = []
    for identity in manifest.get("files", []):
        name = identity.get("path") if isinstance(identity, dict) else None
        if not isinstance(name, str) or not name:
            raise IQKReapError("manifest shard identity has no path")
        if not name.endswith(".safetensors"):
            continue
        if name in names:
            raise IQKReapError(f"manifest declares shard {name!r} twice")
        names.append(name)
    if not names:
        raise IQKReapError("manifest declares no package shards")
    return sorted(names)


def _read_safetensors_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise IQKReapError(f"{path}: too short for a safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if length <= 1 or length > path.stat().st_size - 8:
            raise IQKReapError(f"{path}: invalid safetensors header length {length}")
        blob = handle.read(length)
        if len(blob) != length:
            raise IQKReapError(f"{path}: short safetensors header read")
    try:
        header = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IQKReapError(f"{path}: invalid safetensors header JSON") from exc
    if not isinstance(header, dict):
        raise IQKReapError(f"{path}: safetensors header is not an object")
    return header, 8 + length


def _write_safetensors_header(handle, header: dict) -> None:
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    handle.write(struct.pack("<Q", len(blob)))
    handle.write(blob)


def _safetensors_data_start(path: Path) -> int:
    with open(path, "rb") as handle:
        raw = handle.read(8)
    return 8 + struct.unpack("<Q", raw)[0]


def _tensor_offsets(path: Path, key: str, entry: dict) -> tuple[int, int]:
    offsets = entry.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        raise IQKReapError(f"{path.name}:{key} has invalid data offsets {offsets!r}")
    return offsets[0], offsets[1]


def _copy_at(source, output, offset: int, nbytes: int) -> None:
    source.seek(offset)
    remaining = nbytes
    while remaining:
        chunk = source.read(min(_COPY_CHUNK, remaining))
        if not chunk:
            raise IQKReapError(
                f"short read at offset {offset + nbytes - remaining} of {nbytes} bytes")
        written = output.write(chunk)
        if written != len(chunk):
            raise IQKReapError(f"short write of {written} out of {len(chunk)} bytes")
        remaining -= len(chunk)


def _compare_byte_spans(
    source: Path,
    output: Path,
    spans: Sequence[tuple[int, int, int]],
) -> dict:
    """Compare every selected source byte with its written compact byte."""
    digest = hashlib.sha256()
    compared = 0
    with open(source, "rb") as source_handle, open(output, "rb") as output_handle:
        for span_index, (source_offset, output_offset, nbytes) in enumerate(spans):
            if nbytes <= 0:
                raise IQKReapError(f"byte-comparison span {span_index} is empty")
            source_handle.seek(source_offset)
            output_handle.seek(output_offset)
            remaining = nbytes
            digest.update(struct.pack("<Q", nbytes))
            while remaining:
                count = min(_COPY_CHUNK, remaining)
                source_chunk = source_handle.read(count)
                output_chunk = output_handle.read(count)
                if len(source_chunk) != count or len(output_chunk) != count:
                    raise IQKReapError(
                        f"short byte-comparison read in span {span_index}")
                if source_chunk != output_chunk:
                    mismatch = next(
                        index for index, pair in enumerate(zip(source_chunk, output_chunk))
                        if pair[0] != pair[1]
                    )
                    raise IQKReapError(
                        "selected expert bytes changed after write: "
                        f"span {span_index}, byte {nbytes - remaining + mismatch}")
                digest.update(source_chunk)
                compared += count
                remaining -= count
    return {
        "spans": len(spans),
        "bytes": compared,
        "sha256": digest.hexdigest(),
        "result": "identical",
    }


def _shape_nbytes(shape: Sequence[int], dtype: object) -> int:
    item_sizes = {
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
        "I8": 1,
        "I16": 2,
        "I32": 4,
        "I64": 8,
        "U8": 1,
        "U16": 2,
        "U32": 4,
        "U64": 8,
        "BOOL": 1,
    }
    if dtype not in item_sizes:
        raise IQKReapError(f"unsupported gathered tensor dtype {dtype!r}")
    count = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise IQKReapError(f"invalid tensor shape {shape!r}")
        count *= dimension
    return count * item_sizes[dtype]


def _declared_expert_count(config: dict) -> int:
    for key in ("num_experts", "num_local_experts", "n_routed_experts"):
        value = config.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _with_value(values: object, value: str) -> list:
    out = list(values) if isinstance(values, list) else []
    if value not in out:
        out.append(value)
    return out


def _ids_sha256(ids: Sequence[int]) -> str:
    payload = json.dumps(list(ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _digest_artifact_id(value: object, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(f"{prefix}:"):
        return False
    digest = value[len(prefix) + 1:]
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _layer_span(layers: set[int]) -> str:
    if not layers:
        return "[]"
    return f"[{min(layers)}..{max(layers)}] (n={len(layers)})"


def _resolved(root: Path, name: str) -> Path:
    try:
        return resolve_artifact_file(root, name)
    except UnsafeArtifactPathError as exc:
        raise IQKReapError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", help="relayouted DeepSeek-V4 IQ_K package")
    parser.add_argument("selection", help="deepseek_v4_expert_selection artifact")
    parser.add_argument("--output", required=True, help="new compact package directory")
    parser.add_argument(
        "--promotion-package",
        help="second compatible relayouted package supplying higher-rate cells",
    )
    parser.add_argument(
        "--projection-source-map",
        help="JSON whose layers map every gate/up/down cell to base or promotion",
    )
    parser.add_argument(
        "--omit-drafter",
        action="store_true",
        help="Do not copy the source package's optional drafter component",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        projection_sources = None
        if args.projection_source_map is not None:
            source_payload = json.loads(Path(args.projection_source_map).read_text())
            if not isinstance(source_payload, dict):
                raise IQKReapError("projection source-map JSON must be an object")
            projection_sources = source_payload.get("layers", source_payload)
        report = compact_iqk_package(
            args.package_dir,
            args.selection,
            args.output,
            promotion_package_dir=args.promotion_package,
            projection_sources=projection_sources,
            include_drafter=not args.omit_drafter,
            verbose=args.verbose,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", flush=True)
        return 2
    print(json.dumps({
        "output_package": report["output_package"],
        "package_manifest_id": report["package_manifest_id"],
        "source_selection_artifact_id": report["source_selection_artifact_id"],
        "routed_bytes": report["routed_bytes"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
