from __future__ import annotations

from copy import deepcopy

import pytest

from moespresso.package.qwen4.iqk_reap import (
    Qwen4IQKReapError,
    build_expert_selection,
    selection_manifest_block,
    validate_expert_selection,
)


_SOURCE_PACKAGE_ID = "pkg:" + "a" * 64


def _layers(count: int = 448) -> dict[int, tuple[int, ...]]:
    return {layer: tuple(range(count)) for layer in range(48)}


def test_qwen_selection_is_content_addressed_and_round_trips_manifest_block() -> None:
    selection = build_expert_selection(
        source_package_manifest_id=_SOURCE_PACKAGE_ID,
        layers=_layers(),
    )
    retained = validate_expert_selection(selection)
    block = selection_manifest_block(selection)

    assert selection["artifact_id"].startswith("select:")
    assert retained == _layers()
    assert block["source_selection_artifact_id"] == selection["artifact_id"]
    assert block["source_package_manifest_id"] == _SOURCE_PACKAGE_ID
    assert block["source_num_experts"] == 512
    assert block["top_k"] == 10
    assert block["layers"]["0"]["source_expert_ids"] == list(range(448))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda selection: selection.update(artifact_kind="optimizer_decision"),
            "artifact kind",
        ),
        (
            lambda selection: selection["required_features"].clear(),
            "does not require feature",
        ),
        (
            lambda selection: selection["layers"].pop("47"),
            "exactly 48 layers",
        ),
        (
            lambda selection: selection["layers"]["3"].update(
                source_expert_ids=[0] * 448,
            ),
            "strictly increasing",
        ),
        (
            lambda selection: selection["layers"]["3"].update(
                num_experts=9,
                source_expert_ids=list(range(9)),
            ),
            "fewer experts than top_k",
        ),
    ],
)
def test_qwen_selection_rejects_partial_or_invalid_maps(mutate, message) -> None:
    selection = build_expert_selection(
        source_package_manifest_id=_SOURCE_PACKAGE_ID,
        layers=_layers(),
    )
    broken = deepcopy(selection)
    mutate(broken)

    with pytest.raises(Qwen4IQKReapError, match=message):
        validate_expert_selection(broken)


def test_qwen_selection_builder_rejects_duplicate_canonical_layer_keys() -> None:
    layers = _layers()
    layers["0"] = layers[0]

    with pytest.raises(Qwen4IQKReapError, match="duplicated"):
        build_expert_selection(
            source_package_manifest_id=_SOURCE_PACKAGE_ID,
            layers=layers,
        )
