"""Per-expert bundle assembly + metadata schema (the streaming format).

The bundle module is the single source of truth for the within-row layout:
writer, expert index, correctness ladder, and fixtures all go through it.
These tests pin (1) byte-exact round-trips component -> bundle -> component,
(2) the exact-tiling validation that makes writer/reader drift fail loud,
(3) the dtype/shape contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from moespresso.package.bundle import (
    KQUANT_CODEC,
    ROW_ORDER,
    BundleFormatError,
    assemble_layer_bundle,
    component_array,
    decode_bundle_metadata,
    ds4_source_to_mxfp4_components,
    encode_bundle_metadata,
    row_order_for_codecs,
)


def _components(n_exp=4, out=8, cols=2, down_out=6, down_cols=3, seed=7):
    """Deterministic stacked components with distinct values everywhere."""
    rng = np.random.default_rng(seed)
    comps = {
        ("gate_proj", "packed"): rng.integers(0, 2**32, (n_exp, out, cols), dtype=np.uint32),
        ("gate_proj", "norms"): rng.standard_normal((n_exp, out)).astype(np.float16),
        ("up_proj", "packed"): rng.integers(0, 2**32, (n_exp, out, cols), dtype=np.uint32),
        ("up_proj", "norms"): rng.standard_normal((n_exp, out)).astype(np.float16),
        ("down_proj", "packed"): rng.integers(0, 2**32, (n_exp, down_out, down_cols),
                                              dtype=np.uint32),
        ("down_proj", "norms"): rng.standard_normal((n_exp, down_out)).astype(np.float16),
    }
    bits = {"gate_proj": 2, "up_proj": 2, "down_proj": 3}
    return comps, bits


def test_assemble_round_trips_every_component_byte_exact():
    comps, bits = _components()
    bundle, geo = assemble_layer_bundle(comps, bits)

    assert bundle.dtype == np.uint8
    assert bundle.shape == (4, geo["row_bytes"])
    for (proj, comp) in ROW_ORDER:
        got = component_array(bundle, geo["projections"][proj][comp])
        np.testing.assert_array_equal(got, comps[(proj, comp)])


def test_row_order_tiling_is_exact_and_gap_free():
    comps, bits = _components()
    _, geo = assemble_layer_bundle(comps, bits)
    offset = 0
    for proj, comp in ROW_ORDER:
        c = geo["projections"][proj][comp]
        assert c["offset"] == offset
        offset += c["nbytes"]
    assert offset == geo["row_bytes"]
    assert geo["projections"]["gate_proj"]["packed"]["offset"] == 0


def test_metadata_round_trip_through_json():
    comps, bits = _components()
    _, geo = assemble_layer_bundle(comps, bits)
    text = encode_bundle_metadata({0: geo, 7: geo})
    layers = decode_bundle_metadata(text)
    assert sorted(layers) == [0, 7]
    assert layers[7]["row_bytes"] == geo["row_bytes"]
    assert layers[0]["projections"]["down_proj"]["bits"] == 3


def test_assemble_rejects_missing_component_and_bad_dtype():
    comps, bits = _components()
    broken = dict(comps)
    del broken[("up_proj", "norms")]
    with pytest.raises(BundleFormatError, match="missing"):
        assemble_layer_bundle(broken, bits)

    wrong = dict(comps)
    wrong[("gate_proj", "norms")] = comps[("gate_proj", "norms")].astype(np.float32)
    with pytest.raises(BundleFormatError, match="float16"):
        assemble_layer_bundle(wrong, bits)


def test_assemble_rejects_inconsistent_num_experts_and_bits():
    comps, bits = _components()
    bad = dict(comps)
    bad[("down_proj", "norms")] = np.zeros((5, 6), dtype=np.float16)
    with pytest.raises(BundleFormatError, match="num_experts"):
        assemble_layer_bundle(bad, bits)

    with pytest.raises(BundleFormatError, match="bits"):
        assemble_layer_bundle(comps, {**bits, "up_proj": 0})


def test_decode_rejects_version_gaps_and_tampered_offsets():
    comps, bits = _components()
    _, geo = assemble_layer_bundle(comps, bits)

    with pytest.raises(BundleFormatError, match="version"):
        decode_bundle_metadata('{"version": 99, "layers": {}}')

    import json
    tampered = json.loads(encode_bundle_metadata({0: geo}))
    tampered["layers"]["0"]["projections"]["up_proj"]["packed"]["offset"] += 1
    with pytest.raises(BundleFormatError, match="tiling"):
        decode_bundle_metadata(json.dumps(tampered))

    shrunk = json.loads(encode_bundle_metadata({0: geo}))
    shrunk["layers"]["0"]["row_bytes"] += 4
    with pytest.raises(BundleFormatError, match="no padding"):
        decode_bundle_metadata(json.dumps(shrunk))


def test_component_array_rejects_out_of_range_slice():
    comps, bits = _components()
    bundle, geo = assemble_layer_bundle(comps, bits)
    c = dict(geo["projections"]["down_proj"]["norms"])
    c["offset"] = geo["row_bytes"]  # push past the end
    with pytest.raises(BundleFormatError, match="exceeds"):
        component_array(bundle, c)


def test_mxfp4_projection_uses_packed_and_scales_without_norms():
    rng = np.random.default_rng(11)
    comps = {}
    bits = {"gate_proj": 4, "up_proj": 4, "down_proj": 4}
    codecs = {"gate_proj": "mxfp4", "up_proj": "mxfp4", "down_proj": "mxfp4"}
    for proj, rows in (("gate_proj", 5), ("up_proj", 5), ("down_proj", 4)):
        comps[(proj, "packed")] = rng.integers(
            0,
            2**32,
            (3, rows, 8),
            dtype=np.uint32,
        )
        comps[(proj, "scales")] = rng.integers(
            0,
            256,
            (3, rows, 2),
            dtype=np.uint8,
        )

    bundle, geo = assemble_layer_bundle(comps, bits, codecs=codecs)

    assert row_order_for_codecs(codecs) == (
        ("gate_proj", "packed"), ("gate_proj", "scales"),
        ("up_proj", "packed"), ("up_proj", "scales"),
        ("down_proj", "packed"), ("down_proj", "scales"),
    )
    assert bundle.shape == (3, geo["row_bytes"])
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert geo["projections"][proj]["codec"] == "mxfp4"
        assert "norms" not in geo["projections"][proj]
        for comp in ("packed", "scales"):
            got = component_array(bundle, geo["projections"][proj][comp])
            np.testing.assert_array_equal(got, comps[(proj, comp)])


def test_mixed_tq_and_mxfp4_bundle_metadata_round_trips():
    comps, bits = _components()
    codecs = {"gate_proj": "mxfp4", "up_proj": "tq", "down_proj": "tq"}
    bits = {**bits, "gate_proj": 4}
    del comps[("gate_proj", "norms")]
    comps[("gate_proj", "packed")] = np.zeros((4, 8, 8), dtype=np.uint32)
    comps[("gate_proj", "scales")] = np.zeros((4, 8, 2), dtype=np.uint8)

    _, geo = assemble_layer_bundle(comps, bits, codecs=codecs)
    decoded = decode_bundle_metadata(encode_bundle_metadata({0: geo}))[0]

    assert decoded["projections"]["gate_proj"]["codec"] == "mxfp4"
    assert decoded["projections"]["up_proj"]["codec"] == "tq"
    assert "scales" in decoded["projections"]["gate_proj"]
    assert "norms" in decoded["projections"]["up_proj"]


def test_kquant_projection_uses_wire_weight_and_placeholder_scales():
    rng = np.random.default_rng(19)
    comps, bits = _components()
    codecs = {"gate_proj": KQUANT_CODEC, "up_proj": "tq", "down_proj": "tq"}
    kquant_codecs = {"gate_proj": "q2_k"}
    bits = {**bits, "gate_proj": 2}
    del comps[("gate_proj", "packed")]
    del comps[("gate_proj", "norms")]
    comps[("gate_proj", "weight")] = rng.integers(
        0,
        256,
        (4, 8, 84),
        dtype=np.uint8,
    )
    comps[("gate_proj", "scales")] = np.zeros((4, 1), dtype=np.uint8)

    bundle, geo = assemble_layer_bundle(
        comps,
        bits,
        codecs=codecs,
        kquant_codecs=kquant_codecs,
    )
    decoded = decode_bundle_metadata(encode_bundle_metadata({0: geo}))[0]

    assert row_order_for_codecs(codecs)[:2] == (
        ("gate_proj", "weight"),
        ("gate_proj", "scales"),
    )
    gate = decoded["projections"]["gate_proj"]
    assert gate["codec"] == KQUANT_CODEC
    assert gate["kquant_codec"] == "q2_k"
    assert gate["bytes_per_block"] == 84
    assert gate["weights_per_block"] == 256
    for comp in ("weight", "scales"):
        got = component_array(bundle, gate[comp])
        np.testing.assert_array_equal(got, comps[("gate_proj", comp)])


def test_kquant_projection_rejects_unknown_codec_and_bad_placeholder_shape():
    comps, bits = _components()
    codecs = {"gate_proj": KQUANT_CODEC, "up_proj": "tq", "down_proj": "tq"}
    bits = {**bits, "gate_proj": 2}
    del comps[("gate_proj", "packed")]
    del comps[("gate_proj", "norms")]
    comps[("gate_proj", "weight")] = np.zeros((4, 8, 84), dtype=np.uint8)
    comps[("gate_proj", "scales")] = np.zeros((4, 1), dtype=np.uint8)

    with pytest.raises(BundleFormatError, match="unknown kquant codec"):
        assemble_layer_bundle(
            comps,
            bits,
            codecs=codecs,
            kquant_codecs={"gate_proj": "q2_not_real"},
        )

    bad = dict(comps)
    bad[("gate_proj", "scales")] = np.zeros((4, 2), dtype=np.uint8)
    with pytest.raises(BundleFormatError, match="placeholder"):
        assemble_layer_bundle(
            bad,
            bits,
            codecs=codecs,
            kquant_codecs={"gate_proj": "q2_k"},
        )


def test_ds4_source_bytes_repack_to_live_mxfp4_components():
    logical_codes = np.stack(
        [
            np.arange(64, dtype=np.uint8) % 16,
            15 - (np.arange(64, dtype=np.uint8) % 16),
        ],
        axis=0,
    )
    packed_i8 = (
        logical_codes[:, 0::2] | (logical_codes[:, 1::2] << 4)
    ).astype(np.uint8).view(np.int8)
    scales = np.array([[126, 127], [128, 129]], dtype=np.uint8)

    comps = ds4_source_to_mxfp4_components(packed_i8, scales)
    expected_packed = (
        np.ascontiguousarray(packed_i8)
        .view(np.uint8)
        .reshape(2, 8, 4)
        .copy()
        .view(np.uint32)
        .reshape(2, 8)
    )

    np.testing.assert_array_equal(comps["packed"], expected_packed)
    np.testing.assert_array_equal(comps["scales"], scales)
