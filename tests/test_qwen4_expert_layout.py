from __future__ import annotations

import copy

import pytest

from moespresso.core.artifact import write_artifact
from moespresso.package.qwen4.iqk_reap import (
    build_expert_selection,
    selection_manifest_block,
)
import moespresso.runtime.verify as runtime_verify
from moespresso.runtime.qwen4.expert_layout import (
    PER_LAYER_EXPERTS_FEATURE,
    Qwen4ExpertLayoutError,
    parse_qwen4_expert_layout,
    validate_expert_index_counts,
)


_SOURCE_PACKAGE_ID = "pkg:" + "a" * 64
_SELECTION_ID = "select:" + "b" * 64


def _layers(count: int = 448) -> dict[str, dict]:
    ids = list(range(count))
    return {
        str(layer): {
            "num_experts": count,
            "source_expert_ids": ids,
        }
        for layer in range(48)
    }


def _manifest(*, layers=None) -> dict:
    return {
        "architecture": {
            "family": "qwen4_exp",
            "config": {
                "num_experts": 512,
                "num_experts_per_tok": 10,
            },
        },
        "required_features": [PER_LAYER_EXPERTS_FEATURE],
        "inputs": [_SELECTION_ID],
        "expert_layout": {
            "per_layer_experts": {
                "source_selection_artifact_id": _SELECTION_ID,
                "source_package_manifest_id": _SOURCE_PACKAGE_ID,
                "source_num_experts": 512,
                "top_k": 10,
                "layers": layers or _layers(),
            }
        },
    }


class _Index:
    def __init__(self, counts):
        self.counts = dict(counts)

    def layers_indexed(self):
        return tuple(sorted(self.counts))

    def num_experts_for_layer(self, layer):
        return self.counts[layer]


def test_parse_qwen4_compact_layout_and_validate_index_counts() -> None:
    layers = _layers()
    layers["3"] = {
        "num_experts": 10,
        "source_expert_ids": [2, 7, 18, 39, 90, 155, 203, 250, 400, 511],
    }
    layout = parse_qwen4_expert_layout(_manifest(layers=layers))

    assert layout is not None
    assert layout.source_num_experts == 512
    assert layout.top_k == 10
    assert layout.layers[3].source_expert_ids == (
        2,
        7,
        18,
        39,
        90,
        155,
        203,
        250,
        400,
        511,
    )
    counts = {layer: 448 for layer in range(48)}
    counts[3] = 10
    validate_expert_index_counts(layout, _Index(counts))

    counts[3] = 11
    with pytest.raises(Qwen4ExpertLayoutError, match="mismatch at layer 3"):
        validate_expert_index_counts(layout, _Index(counts))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest["required_features"].clear(),
            "required_features",
        ),
        (
            lambda manifest: manifest["inputs"].clear(),
            "inputs must contain exactly one",
        ),
        (
            lambda manifest: manifest["expert_layout"]["per_layer_experts"][
                "layers"
            ]["3"].update(
                source_expert_ids=[0] * 448,
            ),
            "strictly increasing",
        ),
        (
            lambda manifest: manifest["expert_layout"]["per_layer_experts"][
                "layers"
            ].pop("47"),
            "exactly 48 layers",
        ),
        (
            lambda manifest: manifest["architecture"]["config"].update(
                num_experts=448,
            ),
            "must remain 512",
        ),
    ],
)
def test_qwen4_compact_layout_rejects_partial_or_inconsistent_contract(
    mutate,
    message,
) -> None:
    manifest = copy.deepcopy(_manifest())
    mutate(manifest)

    with pytest.raises(Qwen4ExpertLayoutError, match=message):
        parse_qwen4_expert_layout(manifest)


def test_qwen4_legacy_manifest_has_no_compact_layout() -> None:
    assert (
        parse_qwen4_expert_layout(
            {
                "architecture": {"family": "qwen4_exp"},
                "expert_layout": {"bundled": True},
            }
        )
        is None
    )


def test_qwen4_verifier_authenticates_selection_payload_and_bundle_counts(
    tmp_path,
    monkeypatch,
) -> None:
    selection = build_expert_selection(
        source_package_manifest_id=_SOURCE_PACKAGE_ID,
        layers={layer: tuple(range(448)) for layer in range(48)},
    )
    write_artifact(tmp_path / "expert_selection.json", selection)
    manifest = _manifest(layers=selection["layers"])
    manifest["inputs"] = [selection["artifact_id"]]
    manifest["expert_layout"]["per_layer_experts"] = selection_manifest_block(
        selection
    )
    manifest["files"] = [{"path": "expert_selection.json"}]
    monkeypatch.setattr(
        runtime_verify,
        "build_expert_index",
        lambda _path: _Index({layer: 448 for layer in range(48)}),
    )

    assert runtime_verify._verify_qwen4_expert_layout(manifest, tmp_path) == []

    broken = copy.deepcopy(manifest)
    broken["expert_layout"]["per_layer_experts"]["layers"]["0"][
        "source_expert_ids"
    ][-1] = 448
    issues = runtime_verify._verify_qwen4_expert_layout(broken, tmp_path)
    assert any(
        issue.code == "runtime.qwen4_expert_selection_payload_mismatch"
        for issue in issues
    )
