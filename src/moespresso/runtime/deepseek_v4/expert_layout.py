"""Pure contract for compact per-layer DeepSeek-V4 expert layouts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


PER_LAYER_EXPERTS_FEATURE = "deepseek_v4_per_layer_experts"
EXPERT_SELECTION_FILENAME = "expert_selection.json"
SOURCE_NUM_EXPERTS = 256
ROUTER_TOP_K = 6
NUM_HASH_LAYERS = 3

_SELECTION_ID = re.compile(r"select:[0-9a-f]{64}\Z")
_PACKAGE_ID = re.compile(r"pkg:[0-9a-f]{64}\Z")


class DeepseekV4ExpertLayoutError(ValueError):
    """The manifest's compact expert-layout contract is invalid."""


@dataclass(frozen=True)
class DeepseekV4LayerExpertLayout:
    """One routed layer's compact-to-source expert mapping."""

    layer: int
    num_experts: int
    source_expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class DeepseekV4ExpertLayout:
    """The selection artifact fields embedded in a package manifest."""

    source_selection_artifact_id: str
    source_package_manifest_id: str
    source_num_experts: int
    top_k: int
    layers: Mapping[int, DeepseekV4LayerExpertLayout]

    def selection_payload(self) -> dict[str, Any]:
        """Return the selection fields in their persisted JSON shape."""
        return {
            "source_package_manifest_id": self.source_package_manifest_id,
            "source_num_experts": self.source_num_experts,
            "top_k": self.top_k,
            "layers": {
                str(layer): {
                    "num_experts": row.num_experts,
                    "source_expert_ids": list(row.source_expert_ids),
                }
                for layer, row in sorted(self.layers.items())
            },
        }


def _integer(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DeepseekV4ExpertLayoutError(f"{path} must be an integer")
    return value


def _artifact_id(value: object, *, pattern: re.Pattern[str], path: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise DeepseekV4ExpertLayoutError(f"{path} has an invalid artifact id")
    return value


def _parse_layers(
    value: object,
    *,
    source_num_experts: int,
    top_k: int,
) -> dict[int, DeepseekV4LayerExpertLayout]:
    path = "/expert_layout/per_layer_experts/layers"
    if not isinstance(value, Mapping) or not value:
        raise DeepseekV4ExpertLayoutError(f"{path} must be a non-empty object")

    layers: dict[int, DeepseekV4LayerExpertLayout] = {}
    for raw_layer, raw_row in value.items():
        if (
            not isinstance(raw_layer, str)
            or not raw_layer.isdecimal()
            or str(int(raw_layer)) != raw_layer
        ):
            raise DeepseekV4ExpertLayoutError(
                f"{path} keys must be canonical non-negative decimal strings"
            )
        layer = int(raw_layer)
        row_path = f"{path}/{raw_layer}"
        if not isinstance(raw_row, Mapping):
            raise DeepseekV4ExpertLayoutError(f"{row_path} must be an object")
        num_experts = _integer(
            raw_row.get("num_experts"), path=f"{row_path}/num_experts")
        if not top_k <= num_experts <= source_num_experts:
            raise DeepseekV4ExpertLayoutError(
                f"{row_path}/num_experts must be in "
                f"[{top_k}, {source_num_experts}]"
            )
        raw_ids = raw_row.get("source_expert_ids")
        if not isinstance(raw_ids, list):
            raise DeepseekV4ExpertLayoutError(
                f"{row_path}/source_expert_ids must be a list"
            )
        source_ids = tuple(
            _integer(item, path=f"{row_path}/source_expert_ids/{index}")
            for index, item in enumerate(raw_ids)
        )
        if len(source_ids) != num_experts:
            raise DeepseekV4ExpertLayoutError(
                f"{row_path} declares {num_experts} experts but carries "
                f"{len(source_ids)} source ids"
            )
        if any(
            source_id < 0 or source_id >= source_num_experts
            for source_id in source_ids
        ):
            raise DeepseekV4ExpertLayoutError(
                f"{row_path}/source_expert_ids contains an out-of-range id"
            )
        if any(left >= right for left, right in zip(source_ids, source_ids[1:])):
            raise DeepseekV4ExpertLayoutError(
                f"{row_path}/source_expert_ids must be strictly increasing"
            )
        layers[layer] = DeepseekV4LayerExpertLayout(
            layer=layer,
            num_experts=num_experts,
            source_expert_ids=source_ids,
        )

    expected_layers = set(range(len(layers)))
    if set(layers) != expected_layers:
        raise DeepseekV4ExpertLayoutError(
            f"{path} must cover contiguous layers 0 through {len(layers) - 1}"
        )
    full_source_ids = tuple(range(source_num_experts))
    for layer in range(NUM_HASH_LAYERS):
        row = layers.get(layer)
        if row is None:
            raise DeepseekV4ExpertLayoutError(
                f"{path} must include hash layer {layer}"
            )
        if (
            row.num_experts != source_num_experts
            or row.source_expert_ids != full_source_ids
        ):
            raise DeepseekV4ExpertLayoutError(
                f"{path}/{layer} is a hash layer and must retain source ids "
                f"0 through {source_num_experts - 1}"
            )
    return layers


def parse_deepseek_v4_expert_layout(
    manifest: Mapping[str, Any],
) -> DeepseekV4ExpertLayout | None:
    """Parse the optional compact layout, failing closed on a partial contract.

    Legacy packages do not carry the required feature or the embedded block and
    return ``None``. A package that declares either side must carry both, and
    the selection artifact id must also appear in the manifest's artifact input
    list. File identity and selection-artifact content are checked by the
    separate package verifier, not the serve load path.
    """
    required_features = manifest.get("required_features", [])
    feature_declared = (
        isinstance(required_features, list)
        and PER_LAYER_EXPERTS_FEATURE in required_features
    )
    expert_layout = manifest.get("expert_layout")
    block = (
        expert_layout.get("per_layer_experts")
        if isinstance(expert_layout, Mapping)
        else None
    )
    if block is None and not feature_declared:
        return None
    if not feature_declared:
        raise DeepseekV4ExpertLayoutError(
            "/required_features must declare "
            f"{PER_LAYER_EXPERTS_FEATURE!r} for a compact expert layout"
        )
    if not isinstance(block, Mapping):
        raise DeepseekV4ExpertLayoutError(
            "/expert_layout/per_layer_experts must be an object"
        )

    selection_id = _artifact_id(
        block.get("source_selection_artifact_id"),
        pattern=_SELECTION_ID,
        path="/expert_layout/per_layer_experts/source_selection_artifact_id",
    )
    source_package_id = _artifact_id(
        block.get("source_package_manifest_id"),
        pattern=_PACKAGE_ID,
        path="/expert_layout/per_layer_experts/source_package_manifest_id",
    )
    source_num_experts = _integer(
        block.get("source_num_experts"),
        path="/expert_layout/per_layer_experts/source_num_experts",
    )
    if source_num_experts != SOURCE_NUM_EXPERTS:
        raise DeepseekV4ExpertLayoutError(
            "/expert_layout/per_layer_experts/source_num_experts must be 256"
        )
    top_k = _integer(
        block.get("top_k"), path="/expert_layout/per_layer_experts/top_k")
    if top_k != ROUTER_TOP_K:
        raise DeepseekV4ExpertLayoutError(
            "/expert_layout/per_layer_experts/top_k must be 6"
        )
    layers = _parse_layers(
        block.get("layers"),
        source_num_experts=source_num_experts,
        top_k=top_k,
    )

    architecture = manifest.get("architecture")
    if (
        not isinstance(architecture, Mapping)
        or architecture.get("family") != "deepseek_v4_flash"
    ):
        raise DeepseekV4ExpertLayoutError(
            "/architecture/family must be 'deepseek_v4_flash' for a compact "
            "expert layout"
        )
    architecture_config = (
        architecture.get("config") if isinstance(architecture, Mapping) else None
    )
    manifest_experts = (
        architecture_config.get("n_routed_experts")
        if isinstance(architecture_config, Mapping)
        else None
    )
    if manifest_experts != source_num_experts:
        raise DeepseekV4ExpertLayoutError(
            "/architecture/config/n_routed_experts must remain 256 for a "
            "compact per-layer expert layout"
        )

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or inputs.count(selection_id) != 1:
        raise DeepseekV4ExpertLayoutError(
            f"/inputs must contain exactly one reference to {selection_id}"
        )
    return DeepseekV4ExpertLayout(
        source_selection_artifact_id=selection_id,
        source_package_manifest_id=source_package_id,
        source_num_experts=source_num_experts,
        top_k=top_k,
        layers=layers,
    )


def validate_expert_index_counts(
    layout: DeepseekV4ExpertLayout,
    index: Any,
) -> None:
    """Require bundle-header layer ids and counts to equal the manifest."""
    try:
        indexed_layers = tuple(int(layer) for layer in index.layers_indexed())
    except (AttributeError, TypeError, ValueError) as exc:
        raise DeepseekV4ExpertLayoutError(
            "routed expert index does not expose valid layer ids"
        ) from exc
    manifest_layers = tuple(sorted(layout.layers))
    if tuple(sorted(indexed_layers)) != manifest_layers:
        raise DeepseekV4ExpertLayoutError(
            "routed expert index layers do not match the compact manifest: "
            f"manifest={list(manifest_layers)}, index={list(sorted(indexed_layers))}"
        )
    for layer in manifest_layers:
        try:
            actual = index.num_experts_for_layer(layer)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise DeepseekV4ExpertLayoutError(
                f"routed expert index has no valid count for layer {layer}"
            ) from exc
        expected = layout.layers[layer].num_experts
        if actual != expected:
            raise DeepseekV4ExpertLayoutError(
                f"routed expert count mismatch at layer {layer}: "
                f"manifest={expected}, index={actual}"
            )
