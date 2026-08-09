"""IQ_K expert codec registration: struct facts, bundle format, manifest.

The struct facts are pinned against the Phase-4 sweep's own rate table and
against the byte sizes the conversion wrote, so a table edit that disagrees
with the bytes on disk fails here rather than at serve. Mixed allocations are
exercised throughout: nothing assumes a package-wide member.
"""

from __future__ import annotations

import numpy as np
import pytest

from moespresso.package.bundle import (
    BundleFormatError,
    assemble_layer_bundle,
    component_array,
    decode_bundle_metadata,
    encode_bundle_metadata,
)
from moespresso.package.deepseek_v4.recipe import (
    build_ds4_iqk_expert_allocations,
    build_ds4_iqk_plan,
)
from moespresso.package.iqk_format import (
    IQK_GEOMETRY,
    IQK_LAYOUT_IK_WIRE,
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_LEGACY_RELAYOUT,
    IQKFormatError,
    iqk_geometry,
    normalize_iqk_layout,
    validate_iqk_layout,
)
from moespresso.package.manifest import build_package_manifest, file_identity, located_key
from moespresso.package.plan import make_package_plan

# DeepSeek-V4-Flash routed geometry: gate/up are [2048, 4096], down is
# [4096, 2048], 256 experts per cell.
DS4_SHAPES = {"gate": (2048, 4096), "up": (2048, 4096), "down": (4096, 2048)}
DS4_NUM_EXPERTS = 256

# The rate a DeepSeek-V4 routed cell occupies at each member, per row width:
# (bpw at 4096, bpw at 2048, gate/up cell bytes, down cell bytes). These are
# the figures the allocation sweep water-filled against and the byte sizes the
# conversion wrote, restated here so the table is pinned without reading a
# campaign artifact.
DS4_CELL_RATES = {
    "iq1_s_r4": (1.50390625, 1.5078125, 403_701_760, 404_750_336),
    "iq2_ks": (2.19140625, 2.1953125, 588_251_136, 589_299_712),
    "iq2_k": (2.375, 2.375, 637_534_208, 637_534_208),
    "iq3_k": (3.4375, 3.4375, 922_746_880, 922_746_880),
    "iq4_ks": (4.2578125, 4.265625, 1_142_947_840, 1_145_044_992),
    "iq4_k": (4.5, 4.5, 1_207_959_552, 1_207_959_552),
    "iq5_k": (5.5, 5.5, 1_476_395_008, 1_476_395_008),
    "iq6_k": (6.625, 6.625, 1_778_384_896, 1_778_384_896),
}


# --------------------------------------------------------------------------
# Struct facts


@pytest.mark.parametrize("codec", sorted(DS4_CELL_RATES))
def test_iqk_geometry_reproduces_the_declared_cell_rates(codec):
    bpw_4096, bpw_2048, gate_up_bytes, down_bytes = DS4_CELL_RATES[codec]
    geometry = iqk_geometry(codec)

    assert geometry.bpw(4096) == bpw_4096
    assert geometry.bpw(2048) == bpw_2048
    for role in ("gate", "up", "down"):
        out_features, in_features = DS4_SHAPES[role]
        cell_bytes = DS4_NUM_EXPERTS * out_features * geometry.bytes_per_row(in_features)
        assert cell_bytes == (down_bytes if role == "down" else gate_up_bytes), role


def test_iqk_geometry_covers_the_whole_member_table():
    assert sorted(IQK_GEOMETRY) == sorted(DS4_CELL_RATES)


def test_iqk_geometry_inverts_the_row_width_and_fails_closed():
    ks = iqk_geometry("iq2_ks")
    assert ks.bytes_per_row(4096) == 1122
    assert ks.in_features_for_row_bytes(1122) == 4096
    with pytest.raises(ValueError):
        ks.in_features_for_row_bytes(1123)
    with pytest.raises(ValueError):
        ks.bytes_per_row(4000)


def test_iqk_format_rejects_unknown_members_and_layouts():
    with pytest.raises(IQKFormatError):
        iqk_geometry("iq2_xxs")
    with pytest.raises(IQKFormatError):
        validate_iqk_layout("mlx_wire")
    assert validate_iqk_layout(IQK_LAYOUT_IQK_RELAYOUT) == IQK_LAYOUT_IQK_RELAYOUT


def test_the_pre_rename_layout_reads_but_never_writes():
    """The legacy spelling is readable and unwritable, which is the contract.

    Readers normalize it so packages written before the name was corrected
    serve unchanged. Every writer goes through `validate_iqk_layout`, which
    does not know the value, so nothing can put it into a new package.
    """
    assert normalize_iqk_layout(IQK_LAYOUT_LEGACY_RELAYOUT) == IQK_LAYOUT_IQK_RELAYOUT
    assert normalize_iqk_layout(IQK_LAYOUT_IK_WIRE) == IQK_LAYOUT_IK_WIRE
    assert normalize_iqk_layout("mlx_wire") == "mlx_wire"

    with pytest.raises(IQKFormatError):
        validate_iqk_layout(IQK_LAYOUT_LEGACY_RELAYOUT)
    members = {"gate": "iq2_ks", "up": "iq2_ks", "down": "iq2_ks"}
    comps, bits, codecs, iqk_codecs = _iqk_components(members)
    with pytest.raises(BundleFormatError, match="unknown IQ_K wire layout"):
        assemble_layer_bundle(
            comps,
            bits,
            codecs=codecs,
            iqk_codecs=iqk_codecs,
            iqk_layout=IQK_LAYOUT_LEGACY_RELAYOUT,
        )

    allocations = build_ds4_iqk_expert_allocations({0: members})
    allocations[0]["layout"] = IQK_LAYOUT_LEGACY_RELAYOUT
    with pytest.raises(IQKFormatError, match="unknown IQ_K wire layout"):
        build_ds4_iqk_plan({"source_root": "toy"}, allocations)

    allocations = build_ds4_iqk_expert_allocations({0: members})
    allocations[0].pop("format")
    allocations[0]["codec"] = "iqk"
    allocations[0]["layout"] = IQK_LAYOUT_LEGACY_RELAYOUT
    with pytest.raises(IQKFormatError, match="unknown IQ_K wire layout"):
        build_ds4_iqk_plan({"source_root": "toy"}, allocations)


# --------------------------------------------------------------------------
# Bundle registration


def _iqk_components(members, out_features=8, in_features=256, n_experts=3):
    comps, bits, codecs, iqk_codecs = {}, {}, {}, {}
    for projection, codec in members.items():
        geometry = iqk_geometry(codec)
        row = geometry.bytes_per_row(in_features)
        proj_key = f"{projection}_proj"
        rng = np.random.default_rng(abs(hash(proj_key)) % (2**31))
        comps[(proj_key, "blocks")] = rng.integers(
            0, 256, size=(n_experts, out_features, row), dtype=np.uint8)
        bits[proj_key] = geometry.bits
        codecs[proj_key] = "iqk"
        iqk_codecs[proj_key] = codec
    return comps, bits, codecs, iqk_codecs


def test_mixed_member_layer_bundle_round_trips_through_the_metadata():
    members = {"gate": "iq2_ks", "up": "iq2_k", "down": "iq3_k"}
    comps, bits, codecs, iqk_codecs = _iqk_components(members)

    bundle, geometry = assemble_layer_bundle(
        comps, bits, codecs=codecs, iqk_codecs=iqk_codecs)

    assert bundle.dtype == np.uint8 and bundle.shape[0] == 3
    decoded = decode_bundle_metadata(encode_bundle_metadata({7: geometry}))[7]
    for projection, codec in members.items():
        proj = decoded["projections"][f"{projection}_proj"]
        assert proj["codec"] == "iqk"
        assert proj["iqk_codec"] == codec
        assert proj["layout"] == IQK_LAYOUT_IK_WIRE
        assert proj["ggml_type"] == IQK_GEOMETRY[codec].ggml_type
        assert proj["in_features"] == 256
        assert proj["bytes_per_row"] == IQK_GEOMETRY[codec].bytes_per_row(256)
        recovered = component_array(bundle, proj["blocks"])
        assert np.array_equal(recovered, comps[(f"{projection}_proj", "blocks")])


def test_bundle_records_the_relayout_when_the_build_step_declares_it():
    members = {"gate": "iq2_ks", "up": "iq2_ks", "down": "iq2_ks"}
    comps, bits, codecs, iqk_codecs = _iqk_components(members)

    _bundle, geometry = assemble_layer_bundle(
        comps, bits, codecs=codecs, iqk_codecs=iqk_codecs,
        iqk_layout=IQK_LAYOUT_IQK_RELAYOUT)

    decoded = decode_bundle_metadata(encode_bundle_metadata({0: geometry}))[0]
    assert {
        decoded["projections"][p]["layout"] for p in decoded["projections"]
    } == {IQK_LAYOUT_IQK_RELAYOUT}


def test_bundle_rejects_an_unknown_member_and_an_unknown_layout():
    members = {"gate": "iq2_ks", "up": "iq2_ks", "down": "iq2_ks"}
    comps, bits, codecs, iqk_codecs = _iqk_components(members)

    with pytest.raises(BundleFormatError, match="unknown IQ_K codec"):
        assemble_layer_bundle(
            comps, bits, codecs=codecs,
            iqk_codecs={**iqk_codecs, "gate_proj": "iq2_xxs"})
    with pytest.raises(BundleFormatError, match="unknown IQ_K wire layout"):
        assemble_layer_bundle(
            comps, bits, codecs=codecs, iqk_codecs=iqk_codecs, iqk_layout="nope")


def test_bundle_rejects_a_row_width_that_is_not_whole_blocks():
    members = {"gate": "iq2_ks", "up": "iq2_ks", "down": "iq2_ks"}
    comps, bits, codecs, iqk_codecs = _iqk_components(members)
    good = comps[("gate_proj", "blocks")]
    comps[("gate_proj", "blocks")] = np.ascontiguousarray(good[:, :, :-1])

    with pytest.raises(BundleFormatError, match="row meta plus whole"):
        assemble_layer_bundle(comps, bits, codecs=codecs, iqk_codecs=iqk_codecs)


def test_bundle_metadata_rejects_a_member_swapped_after_the_write():
    members = {"gate": "iq2_ks", "up": "iq2_ks", "down": "iq2_ks"}
    comps, bits, codecs, iqk_codecs = _iqk_components(members)
    _bundle, geometry = assemble_layer_bundle(
        comps, bits, codecs=codecs, iqk_codecs=iqk_codecs)
    geometry["projections"]["gate_proj"]["iqk_codec"] = "iq2_k"

    with pytest.raises(BundleFormatError, match="bad IQ_K params"):
        decode_bundle_metadata(encode_bundle_metadata({0: geometry}))


# --------------------------------------------------------------------------
# Manifest registration


DS4_ARCH = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 2,
    "num_nextn_predict_layers": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "sliding_window": 128,
    "index_topk": 512,
    "compress_rope_theta": 160000,
    "compress_ratios": [0, 0],
    "vocab_size": 129280,
}


def _iqk_manifest(tmp_path, members, *, layout=IQK_LAYOUT_IK_WIRE, mutate=None):
    allocation = build_ds4_iqk_expert_allocations(members, layout=layout)
    for alloc in allocation:
        if mutate is not None:
            mutate(alloc)
    plan, _summary = make_package_plan(
        {"source_root": "toy", "source_format": "hf_safetensors"},
        allocation,
        producer_kind="iqk_converted_artifacts",
        required_features=["calibration"],
    )
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"shard")
    located = {
        located_key(a): {
            "shard": shard.name,
            "key_prefix": f"layers.{a['layer_index']}.ffn.experts",
        }
        for a in plan["allocation"]
    }
    return build_package_manifest(
        plan, DS4_ARCH, located, [file_identity(shard)])


def test_manifest_records_every_cell_member_and_the_wire_layout(tmp_path):
    members = {
        0: {"gate": "iq2_ks", "up": "iq2_k", "down": "iq2_ks"},
        1: {"gate": "iq2_k", "up": "iq2_ks", "down": "iq3_k"},
    }

    manifest = _iqk_manifest(tmp_path, members)

    assert manifest["status"] == "valid"
    assert "iqk_dequant" in manifest["required_ops"]
    assert manifest["expert_layout"]["bundled"] is True
    by_cell = {
        (t["layer_index"], t["projection"]): t
        for t in manifest["tensors"] if t["kind"] == "expert"
    }
    assert len(by_cell) == 6
    for layer, cells in members.items():
        for projection, codec in cells.items():
            entry = by_cell[(layer, projection)]
            params = entry["format_params"]
            assert entry["format"] == "iqk"
            assert params["iqk_codec"] == codec
            assert params["layout"] == IQK_LAYOUT_IK_WIRE
            assert params["bits"] == IQK_GEOMETRY[codec].bits
            assert params["ggml_type"] == IQK_GEOMETRY[codec].ggml_type
            assert params["bytes_per_block"] == IQK_GEOMETRY[codec].bytes_per_block
            assert params["row_meta_bytes"] == IQK_GEOMETRY[codec].row_meta_bytes
            assert entry["module_weight_key"].endswith(f"{projection}_proj.weight")


def test_manifest_blocks_an_unknown_member(tmp_path):
    members = {0: {"gate": "iq2_ks", "up": "iq2_ks", "down": "iq2_ks"}}

    def _swap(alloc):
        if alloc["projection"] == "up":
            alloc["iqk_codec"] = "iq2_xxs"

    manifest = _iqk_manifest(tmp_path, members, mutate=_swap)

    assert manifest["status"] == "invalid"
    codes = {v["code"] for v in manifest["validation"] if v["blocking"]}
    assert "package.unsupported_iqk_codec" in codes


def test_manifest_blocks_an_unknown_wire_layout(tmp_path):
    members = {0: {"gate": "iq2_ks", "up": "iq2_ks", "down": "iq2_ks"}}

    def _swap(alloc):
        alloc["layout"] = "mlx_wire"

    manifest = _iqk_manifest(tmp_path, members, mutate=_swap)

    assert manifest["status"] == "invalid"
    codes = {v["code"] for v in manifest["validation"] if v["blocking"]}
    assert "package.unsupported_iqk_layout" in codes


def test_iqk_allocation_rows_reject_an_incomplete_layer():
    with pytest.raises(Exception, match="gate, up and down"):
        build_ds4_iqk_expert_allocations({0: {"gate": "iq2_ks", "up": "iq2_ks"}})
