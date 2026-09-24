"""Pure contract for compact per-layer Qwen expert layouts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


PER_LAYER_EXPERTS_FEATURE = "qwen4_per_layer_experts"
EXPERT_SELECTION_FILENAME = "expert_selection.json"
SOURCE_NUM_EXPERTS = 512
ROUTER_TOP_K = 10
NUM_LAYERS = 48

_SELECTION_ID = re.compile(r"select:[0-9a-f]{64}\Z")
_PACKAGE_ID = re.compile(r"pkg:[0-9a-f]{64}\Z")


class Qwen4ExpertLayoutError(ValueError):
    """The manifest's compact expert-layout contract is invalid."""


@dataclass(frozen=True)
class Qwen4LayerExpertLayout:
    """One routed layer's compact-to-source expert mapping."""

    layer: int
    num_experts: int
    source_expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class Qwen4ExpertLayout:
    """The selection artifact fields embedded in a package manifest."""

    source_selection_artifact_id: str
    source_package_manifest_id: str
    source_num_experts: int
    top_k: int
    layers: Mapping[int, Qwen4LayerExpertLayout]

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
        raise Qwen4ExpertLayoutError(f"{path} must be an integer")
    return value


def _artifact_id(value: object, *, pattern: re.Pattern[str], path: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise Qwen4ExpertLayoutError(f"{path} has an invalid artifact id")
    return value


def _parse_layers(value: object) -> dict[int, Qwen4LayerExpertLayout]:
    path = "/expert_layout/per_layer_experts/layers"
    if not isinstance(value, Mapping):
        raise Qwen4ExpertLayoutError(f"{path} must be an object")
    if len(value) != NUM_LAYERS:
        raise Qwen4ExpertLayoutError(f"{path} must contain exactly {NUM_LAYERS} layers")

    layers: dict[int, Qwen4LayerExpertLayout] = {}
    for raw_layer, raw_row in value.items():
        if (
            not isinstance(raw_layer, str)
            or not raw_layer.isdecimal()
            or str(int(raw_layer)) != raw_layer
        ):
            raise Qwen4ExpertLayoutError(
                f"{path} keys must be canonical non-negative decimal strings"
            )
        layer = int(raw_layer)
        row_path = f"{path}/{raw_layer}"
        if not isinstance(raw_row, Mapping):
            raise Qwen4ExpertLayoutError(f"{row_path} must be an object")
        num_experts = _integer(
            raw_row.get("num_experts"),
            path=f"{row_path}/num_experts",
        )
        if not ROUTER_TOP_K <= num_experts <= SOURCE_NUM_EXPERTS:
            raise Qwen4ExpertLayoutError(
                f"{row_path}/num_experts must be in [{ROUTER_TOP_K}, {SOURCE_NUM_EXPERTS}]"
            )
        raw_ids = raw_row.get("source_expert_ids")
        if not isinstance(raw_ids, list):
            raise Qwen4ExpertLayoutError(f"{row_path}/source_expert_ids must be a list")
        source_ids = tuple(
            _integer(item, path=f"{row_path}/source_expert_ids/{index}")
            for index, item in enumerate(raw_ids)
        )
        if len(source_ids) != num_experts:
            raise Qwen4ExpertLayoutError(
                f"{row_path} declares {num_experts} experts but carries "
                f"{len(source_ids)} source ids"
            )
        if any(source_id < 0 or source_id >= SOURCE_NUM_EXPERTS for source_id in source_ids):
            raise Qwen4ExpertLayoutError(
                f"{row_path}/source_expert_ids contains an out-of-range id"
            )
        if any(left >= right for left, right in zip(source_ids, source_ids[1:])):
            raise Qwen4ExpertLayoutError(
                f"{row_path}/source_expert_ids must be strictly increasing"
            )
        layers[layer] = Qwen4LayerExpertLayout(
            layer=layer,
            num_experts=num_experts,
            source_expert_ids=source_ids,
        )

    if set(layers) != set(range(NUM_LAYERS)):
        raise Qwen4ExpertLayoutError(f"{path} must cover exactly layers 0 through 47")
    return layers


def parse_qwen4_expert_layout(manifest: Mapping[str, Any]) -> Qwen4ExpertLayout | None:
    """Parse the optional compact layout and reject a partial declaration."""

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
        raise Qwen4ExpertLayoutError(
            f"/required_features must declare {PER_LAYER_EXPERTS_FEATURE!r}"
        )
    if not isinstance(block, Mapping):
        raise Qwen4ExpertLayoutError(
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
        raise Qwen4ExpertLayoutError(
            "/expert_layout/per_layer_experts/source_num_experts must be 512"
        )
    top_k = _integer(
        block.get("top_k"),
        path="/expert_layout/per_layer_experts/top_k",
    )
    if top_k != ROUTER_TOP_K:
        raise Qwen4ExpertLayoutError(
            "/expert_layout/per_layer_experts/top_k must be 10"
        )
    layers = _parse_layers(block.get("layers"))

    architecture = manifest.get("architecture")
    config = architecture.get("config") if isinstance(architecture, Mapping) else None
    if (
        not isinstance(architecture, Mapping)
        or architecture.get("family") != "qwen4_exp"
        or not isinstance(config, Mapping)
    ):
        raise Qwen4ExpertLayoutError(
            "/architecture must declare the qwen4_exp family and config"
        )
    if config.get("num_experts") != SOURCE_NUM_EXPERTS:
        raise Qwen4ExpertLayoutError(
            "/architecture/config/num_experts must remain 512"
        )
    if config.get("num_experts_per_tok") != ROUTER_TOP_K:
        raise Qwen4ExpertLayoutError(
            "/architecture/config/num_experts_per_tok must remain 10"
        )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or inputs.count(selection_id) != 1:
        raise Qwen4ExpertLayoutError(
            f"/inputs must contain exactly one reference to {selection_id}"
        )
    return Qwen4ExpertLayout(
        source_selection_artifact_id=selection_id,
        source_package_manifest_id=source_package_id,
        source_num_experts=source_num_experts,
        top_k=top_k,
        layers=layers,
    )


def validate_expert_index_counts(layout: Qwen4ExpertLayout, index: Any) -> None:
    """Require bundle-header layer ids and counts to equal the manifest."""

    try:
        indexed_layers = tuple(int(layer) for layer in index.layers_indexed())
    except (AttributeError, TypeError, ValueError) as exc:
        raise Qwen4ExpertLayoutError(
            "routed expert index does not expose valid layer ids"
        ) from exc
    expected_layers = tuple(range(NUM_LAYERS))
    if tuple(sorted(indexed_layers)) != expected_layers:
        raise Qwen4ExpertLayoutError(
            "routed expert index layers do not match the compact manifest"
        )
    for layer in expected_layers:
        try:
            actual = index.num_experts_for_layer(layer)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise Qwen4ExpertLayoutError(
                f"routed expert index has no valid count for layer {layer}"
            ) from exc
        expected = layout.layers[layer].num_experts
        if actual != expected:
            raise Qwen4ExpertLayoutError(
                f"routed expert index count mismatch at layer {layer}: "
                f"manifest={expected}, index={actual}"
            )
