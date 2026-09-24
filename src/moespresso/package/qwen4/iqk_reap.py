"""Qwen compact-expert selection artifact and manifest contract."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from moespresso.core.artifact import artifact_producer, make_artifact


SELECTION_KIND = "qwen4_expert_selection"
SELECTION_FEATURE = "qwen4_per_layer_experts"
EXPERT_SELECTION_NAME = "expert_selection.json"
SOURCE_NUM_EXPERTS = 512
ROUTER_TOP_K = 10
NUM_LAYERS = 48
_PRODUCER = artifact_producer("moespresso.package.qwen4.iqk_reap")


class Qwen4IQKReapError(ValueError):
    """The Qwen expert selection cannot produce a compact package."""


def _digest_artifact_id(value: object, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(f"{prefix}:")
        and len(value) == len(prefix) + 65
        and all(character in "0123456789abcdef" for character in value[len(prefix) + 1 :])
    )


def build_expert_selection(
    *,
    source_package_manifest_id: str,
    layers: Mapping[int | str, Sequence[int]],
    subject: dict | None = None,
) -> dict:
    """Create one content-addressed per-layer expert selection."""

    normalized: dict[str, dict] = {}
    try:
        ordered = sorted(layers.items(), key=lambda item: int(item[0]))
    except (TypeError, ValueError) as exc:
        raise Qwen4IQKReapError("selection layer keys must be integers") from exc
    for raw_layer, ids in ordered:
        if isinstance(raw_layer, bool):
            raise Qwen4IQKReapError("selection layer keys must not be booleans")
        layer = str(int(raw_layer))
        if layer in normalized:
            raise Qwen4IQKReapError(f"selection layer {layer} is duplicated")
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
        source_num_experts=SOURCE_NUM_EXPERTS,
        top_k=ROUTER_TOP_K,
        layers=normalized,
    )
    validate_expert_selection(payload)
    return payload


def validate_expert_selection(selection: Mapping[str, object]) -> dict[int, tuple[int, ...]]:
    """Validate and return the compact-row to source-id map by layer."""

    if selection.get("artifact_kind") != SELECTION_KIND:
        raise Qwen4IQKReapError(
            f"selection artifact kind is {selection.get('artifact_kind')!r}, "
            f"not {SELECTION_KIND!r}"
        )
    if selection.get("status") != "valid":
        raise Qwen4IQKReapError("selection status must be 'valid'")
    if not _digest_artifact_id(selection.get("artifact_id"), "select"):
        raise Qwen4IQKReapError("selection has no valid artifact_id")
    features = selection.get("required_features")
    if not isinstance(features, list) or SELECTION_FEATURE not in features:
        raise Qwen4IQKReapError(
            f"selection does not require feature {SELECTION_FEATURE!r}"
        )
    if selection.get("source_num_experts") != SOURCE_NUM_EXPERTS:
        raise Qwen4IQKReapError("Qwen selection source_num_experts must be 512")
    if selection.get("top_k") != ROUTER_TOP_K:
        raise Qwen4IQKReapError("Qwen selection top_k must be 10")
    if not _digest_artifact_id(selection.get("source_package_manifest_id"), "pkg"):
        raise Qwen4IQKReapError("selection has no valid source_package_manifest_id")
    raw_layers = selection.get("layers")
    if not isinstance(raw_layers, Mapping) or len(raw_layers) != NUM_LAYERS:
        raise Qwen4IQKReapError("selection must contain exactly 48 layers")

    normalized: dict[int, tuple[int, ...]] = {}
    for raw_layer, record in raw_layers.items():
        if (
            not isinstance(raw_layer, str)
            or not raw_layer.isdecimal()
            or str(int(raw_layer)) != raw_layer
        ):
            raise Qwen4IQKReapError(
                f"selection layer key {raw_layer!r} is not canonical decimal"
            )
        layer = int(raw_layer)
        if not isinstance(record, Mapping):
            raise Qwen4IQKReapError(f"selection layer {layer} must be an object")
        raw_ids = record.get("source_expert_ids")
        if not isinstance(raw_ids, list):
            raise Qwen4IQKReapError(
                f"selection layer {layer} source_expert_ids must be a list"
            )
        ids = []
        for expert in raw_ids:
            if isinstance(expert, bool) or not isinstance(expert, int):
                raise Qwen4IQKReapError(
                    f"selection layer {layer} expert id {expert!r} is not an integer"
                )
            ids.append(expert)
        if ids != sorted(set(ids)):
            raise Qwen4IQKReapError(
                f"selection layer {layer} expert ids are not strictly increasing"
            )
        if not ids or ids[0] < 0 or ids[-1] >= SOURCE_NUM_EXPERTS:
            raise Qwen4IQKReapError(
                f"selection layer {layer} expert ids leave [0, {SOURCE_NUM_EXPERTS})"
            )
        if record.get("num_experts") != len(ids):
            raise Qwen4IQKReapError(
                f"selection layer {layer} num_experts does not match its source ids"
            )
        if len(ids) < ROUTER_TOP_K:
            raise Qwen4IQKReapError(
                f"selection layer {layer} retains fewer experts than top_k"
            )
        normalized[layer] = tuple(ids)
    if set(normalized) != set(range(NUM_LAYERS)):
        raise Qwen4IQKReapError("selection must cover exactly layers 0 through 47")
    return dict(sorted(normalized.items()))


def selection_manifest_block(selection: Mapping[str, object]) -> dict:
    """Return the selection fields embedded into a package manifest."""

    validate_expert_selection(selection)
    return {
        "source_selection_artifact_id": selection["artifact_id"],
        "source_package_manifest_id": selection["source_package_manifest_id"],
        "source_num_experts": selection["source_num_experts"],
        "top_k": selection["top_k"],
        "layers": json.loads(json.dumps(selection["layers"])),
    }
