from __future__ import annotations

import numpy as np
import pytest

from moespresso.package.bundle import component_array, decode_bundle_metadata, encode_bundle_metadata
from moespresso.package.kquant_backend import KQuantEncodedWeight
from moespresso.package.kquant_bundle import (
    KQuantBundleError,
    assemble_kquant_encoded_layer_bundle,
    stack_kquant_projection_components,
)


def _encoded(codec: str, *, experts=2, rows=3, bytes_per_row=84, seed=0):
    rng = np.random.default_rng(seed)
    return [
        KQuantEncodedWeight(
            codec=codec,
            weight=rng.integers(0, 256, (rows, bytes_per_row), dtype=np.uint8),
            scales=np.zeros((1,), dtype=np.uint8),
        )
        for _idx in range(experts)
    ]


def test_stack_kquant_projection_components_preserves_wire_bytes():
    encoded = _encoded("q2_k", seed=1)

    codec, stacked = stack_kquant_projection_components(encoded, projection="down_proj")

    assert codec == "q2_k"
    assert stacked["weight"].shape == (2, 3, 84)
    assert stacked["scales"].shape == (2, 1)
    np.testing.assert_array_equal(stacked["weight"][0], encoded[0].weight)
    np.testing.assert_array_equal(stacked["scales"][1], encoded[1].scales)


def test_assemble_kquant_encoded_layer_bundle_round_trips_metadata_and_components():
    encoded = {
        "gate": _encoded("iq2_xxs", bytes_per_row=66, seed=1),
        "up": _encoded("iq2_xxs", bytes_per_row=66, seed=2),
        "down": _encoded("q2_k", bytes_per_row=84, seed=3),
    }

    bundle, geometry = assemble_kquant_encoded_layer_bundle(encoded)
    decoded = decode_bundle_metadata(encode_bundle_metadata({0: geometry}))[0]

    assert bundle.shape == (2, geometry["row_bytes"])
    assert decoded["projections"]["gate_proj"]["kquant_codec"] == "iq2_xxs"
    assert decoded["projections"]["down_proj"]["kquant_codec"] == "q2_k"
    down_weight = component_array(bundle, decoded["projections"]["down_proj"]["weight"])
    down_scales = component_array(bundle, decoded["projections"]["down_proj"]["scales"])
    np.testing.assert_array_equal(down_weight, np.stack([x.weight for x in encoded["down"]]))
    np.testing.assert_array_equal(down_scales, np.zeros((2, 1), dtype=np.uint8))


def test_assemble_kquant_encoded_layer_bundle_rejects_missing_projection():
    with pytest.raises(KQuantBundleError, match="missing"):
        assemble_kquant_encoded_layer_bundle({
            "gate": _encoded("q2_k"),
            "up": _encoded("q2_k"),
        })


def test_assemble_kquant_encoded_layer_bundle_rejects_unknown_projection():
    with pytest.raises(KQuantBundleError, match="unknown"):
        assemble_kquant_encoded_layer_bundle({
            "gate_proj": _encoded("q2_k"),
            "up": _encoded("q2_k"),
            "down": _encoded("q2_k"),
        })


def test_stack_kquant_projection_components_rejects_mixed_codec():
    encoded = _encoded("q2_k")
    encoded[1] = KQuantEncodedWeight(
        codec="iq2_xxs",
        weight=encoded[1].weight,
        scales=encoded[1].scales,
    )

    with pytest.raises(KQuantBundleError, match="codec"):
        stack_kquant_projection_components(encoded, projection="down_proj")


def test_stack_kquant_projection_components_rejects_bad_placeholder_shape():
    encoded = _encoded("q2_k")
    encoded[0] = KQuantEncodedWeight(
        codec="q2_k",
        weight=encoded[0].weight,
        scales=np.zeros((2,), dtype=np.uint8),
    )

    with pytest.raises(KQuantBundleError, match="uint8\\[1\\]"):
        stack_kquant_projection_components(encoded, projection="down_proj")
