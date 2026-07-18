"""Expert byte-offset index for SSD streaming (bundle format).

Pure + model-free: builds (layer, expert[, projection, component]) -> byte
range from safetensors headers + shard metadata, so the miss-loader can pread
one bundle row per missed expert. Tested against a synthetic bundle-format
shard so it never needs the 35B package.

Pre-bundle-format stacked packages must fail loudly with a re-convert message
(no compatibility path), never a silent miss.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from moespresso.package.bundle import (
    KQUANT_CODEC,
    ROW_ORDER,
    MXFP4_CODEC,
    assemble_layer_bundle,
    encode_bundle_metadata,
)
from moespresso.runtime.expert_index import (
    ExpertByteRange,
    StackedLayoutError,
    build_expert_index,
)

def _write_safetensors(path, tensors, metadata=None):
    """tensors: name -> (dtype_str, shape, bytes). Writes a minimal safetensors."""
    header, blob, off = {}, bytearray(), 0
    if metadata:
        header["__metadata__"] = dict(metadata)
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + len(raw)]}
        blob += raw
        off += len(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _layer_components(n_exp, out, cols, seed):
    rng = np.random.default_rng(seed)
    comps = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        comps[(proj, "packed")] = rng.integers(
            0, 2**32, (n_exp, out, cols), dtype=np.uint32)
        comps[(proj, "norms")] = rng.standard_normal((n_exp, out)).astype(np.float16)
    return comps


def _mxfp4_layer_components(n_exp, out, packed_words, seed):
    rng = np.random.default_rng(seed)
    scale_cols = packed_words // 4
    comps = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        comps[(proj, "packed")] = rng.integers(
            0, 2**32, (n_exp, out, packed_words), dtype=np.uint32)
        comps[(proj, "scales")] = rng.integers(
            0, 256, (n_exp, out, scale_cols), dtype=np.uint8)
    return comps


def _mxfp4_codecs():
    return {p: MXFP4_CODEC for p in ("gate_proj", "up_proj", "down_proj")}


def _kquant_layer_components(n_exp, out, row_bytes, seed):
    rng = np.random.default_rng(seed)
    comps = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        comps[(proj, "weight")] = rng.integers(
            0, 256, (n_exp, out, row_bytes), dtype=np.uint8)
        comps[(proj, "scales")] = np.zeros((n_exp, 1), dtype=np.uint8)
    return comps


def _kquant_codecs():
    return {p: KQUANT_CODEC for p in ("gate_proj", "up_proj", "down_proj")}


def _tiny_package(tmp_path, *, n_layers=2, n_exp=4, out=8, cols=2, bits=2):
    """A minimal bundle-format package: 1 shard, N layers of switch_mlp bundles."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    tensors, layer_geo = {}, {}
    for layer in range(n_layers):
        comps = _layer_components(n_exp, out, cols, seed=layer)
        bundle, geo = assemble_layer_bundle(
            comps, {p: bits for p in ("gate_proj", "up_proj", "down_proj")})
        base = f"language_model.model.layers.{layer}.mlp.switch_mlp"
        tensors[f"{base}.experts.tq_bundle"] = (
            "U8", bundle.shape, bundle.tobytes())
        layer_geo[layer] = geo
    # a non-expert tensor to prove it's ignored
    tensors["language_model.model.embed_tokens.weight"] = ("F16", (10, 8), bytes(160))
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors", tensors,
        metadata={"format": "mjtq",
                  "expert_bundles": encode_bundle_metadata(layer_geo)})
    return pkg, n_exp, out, cols


def test_index_lists_every_layer_expert_projection(tmp_path):
    pkg, n_exp, out, cols = _tiny_package(tmp_path, n_layers=2, n_exp=4)
    idx = build_expert_index(pkg)

    assert idx.num_layers == 2
    assert idx.num_experts == 4
    assert idx.num_layers_indexed() == 2
    assert idx.layers_indexed() == (0, 1)
    assert idx.num_expert_slots() == 2 * 4


def test_index_accepts_deepseek_root_bundle_key(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    comps = _layer_components(2, 8, 2, seed=0)
    bundle, geo = assemble_layer_bundle(
        comps, {p: 4 for p in ("gate_proj", "up_proj", "down_proj")})
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors",
        {"layers.0.ffn.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes())},
        metadata={"expert_bundles": encode_bundle_metadata({0: geo})},
    )

    idx = build_expert_index(pkg)

    assert idx.layers_indexed() == (0,)
    assert idx.num_experts == 2
    assert idx.bits(layer=0, projection="gate_proj") == 4


def test_index_refuses_separate_mxfp4_bundle_suffix_even_with_tq_bundle(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    tq_comps = _layer_components(2, 8, 2, seed=0)
    tq_bundle, tq_geo = assemble_layer_bundle(
        tq_comps, {p: 4 for p in ("gate_proj", "up_proj", "down_proj")})
    mxfp4_comps = _mxfp4_layer_components(2, 8, 4, seed=1)
    mxfp4_bundle, _mxfp4_geo = assemble_layer_bundle(
        mxfp4_comps,
        {p: 4 for p in ("gate_proj", "up_proj", "down_proj")},
        codecs=_mxfp4_codecs(),
    )
    base = "layers.0.ffn"
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors",
        {
            f"{base}.experts.tq_bundle": ("U8", tq_bundle.shape, tq_bundle.tobytes()),
            f"{base}.experts.mxfp4_bundle": (
                "U8", mxfp4_bundle.shape, mxfp4_bundle.tobytes()),
        },
        metadata={
            "expert_bundles": encode_bundle_metadata({0: tq_geo}),
        },
    )

    with pytest.raises(ValueError, match="unsupported routed-expert bundle suffix"):
        build_expert_index(pkg)


def test_index_exposes_live_mxfp4_bundle_geometry(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    comps = _mxfp4_layer_components(2, 8, 4, seed=0)
    codecs = _mxfp4_codecs()
    bundle, geo = assemble_layer_bundle(
        comps,
        {p: 4 for p in ("gate_proj", "up_proj", "down_proj")},
        codecs=codecs,
    )
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors",
        {"layers.0.ffn.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes())},
        metadata={"expert_bundles": encode_bundle_metadata({0: geo})},
    )

    idx = build_expert_index(pkg)

    assert idx.codec(layer=0, projection="gate_proj") == MXFP4_CODEC
    assert idx.components_for_projection(layer=0, projection="gate_proj") == (
        "packed",
        "scales",
    )
    assert idx.bits(layer=0, projection="gate_proj") == 4
    gate = idx.geometry(layer=0, projection="gate_proj")
    assert gate.codec == MXFP4_CODEC
    assert gate.packed_dtype == "U32"
    assert gate.scales_dtype == "U8"
    assert gate.norms_dtype is None
    assert idx.locate(
        layer=0,
        expert=1,
        projection="gate_proj",
        component="scales",
    ).nbytes == 8 * 1


def test_index_exposes_kquant_bundle_geometry(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    comps = _kquant_layer_components(2, 8, 84, seed=0)
    codecs = _kquant_codecs()
    bundle, geo = assemble_layer_bundle(
        comps,
        {p: 2 for p in ("gate_proj", "up_proj", "down_proj")},
        codecs=codecs,
        kquant_codecs={p: "q2_k" for p in ("gate_proj", "up_proj", "down_proj")},
    )
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors",
        {"layers.0.ffn.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes())},
        metadata={"expert_bundles": encode_bundle_metadata({0: geo})},
    )

    idx = build_expert_index(pkg)

    assert idx.has_projection(layer=0, projection="gate_proj")
    assert idx.codec(layer=0, projection="gate_proj") == KQUANT_CODEC
    assert idx.components_for_projection(layer=0, projection="gate_proj") == (
        "weight",
        "scales",
    )
    gate = idx.geometry(layer=0, projection="gate_proj")
    assert gate.codec == KQUANT_CODEC
    assert gate.kquant_codec == "q2_k"
    assert gate.packed_dtype == "U8"
    assert gate.packed_cols == 84
    assert gate.bits == 2
    assert gate.group_size == 256
    assert gate.bytes_per_block == 84
    assert gate.weights_per_block == 256
    assert gate.scales_dtype == "U8"
    assert gate.norms_dtype is None
    weight = idx.locate(
        layer=0,
        expert=1,
        projection="gate_proj",
        component="weight",
    )
    scales = idx.locate(
        layer=0,
        expert=1,
        projection="gate_proj",
        component="scales",
    )
    assert weight.shape == (8, 84)
    assert weight.dtype == "U8"
    assert weight.nbytes == 8 * 84
    assert scales.shape == (1,)
    assert scales.nbytes == 1


def test_component_ranges_stride_by_full_row(tmp_path):
    pkg, n_exp, out, cols = _tiny_package(tmp_path, n_layers=2, n_exp=4, out=8, cols=2)
    idx = build_expert_index(pkg)

    packed_bytes = out * cols * 4  # U32
    row_bytes = idx.row_bytes(layer=0)
    e0 = idx.locate(layer=0, expert=0, projection="gate_proj", component="packed")
    e1 = idx.locate(layer=0, expert=1, projection="gate_proj", component="packed")
    assert isinstance(e0, ExpertByteRange)
    assert e0.nbytes == packed_bytes
    # consecutive experts stride by the complete bundle-row size
    assert e1.offset == e0.offset + row_bytes
    assert e0.shard == "model-00001-of-00001.safetensors"

    n0 = idx.locate(layer=0, expert=0, projection="down_proj", component="norms")
    n1 = idx.locate(layer=0, expert=1, projection="down_proj", component="norms")
    assert n0.nbytes == out * 2  # F16
    assert n1.offset == n0.offset + row_bytes


def test_locate_row_is_one_contiguous_range_covering_all_components(tmp_path):
    pkg, n_exp, out, cols = _tiny_package(tmp_path, n_layers=1, n_exp=4)
    idx = build_expert_index(pkg)

    row = idx.locate_row(layer=0, expert=2)
    comps = idx.row_components(layer=0)
    assert row.nbytes == idx.row_bytes(layer=0)
    assert row.dtype == "U8"
    # every component's absolute range == row start + its within-row offset,
    # and together they tile the row exactly in ROW_ORDER
    offset = 0
    for proj, comp in ROW_ORDER:
        c = comps[(proj, comp)]
        absolute = idx.locate(layer=0, expert=2, projection=proj, component=comp)
        assert absolute.offset == row.offset + c["offset"]
        assert absolute.nbytes == c["nbytes"]
        assert c["offset"] == offset
        offset += c["nbytes"]
    assert offset == row.nbytes


def test_different_projections_have_different_offsets(tmp_path):
    pkg, *_ = _tiny_package(tmp_path, n_layers=1, n_exp=4)
    idx = build_expert_index(pkg)
    g = idx.locate(layer=0, expert=2, projection="gate_proj", component="packed")
    u = idx.locate(layer=0, expert=2, projection="up_proj", component="packed")
    d = idx.locate(layer=0, expert=2, projection="down_proj", component="packed")
    assert len({g.offset, u.offset, d.offset}) == 3


def test_index_validates_offsets_in_range_and_total(tmp_path):
    pkg, *_ = _tiny_package(tmp_path, n_layers=2, n_exp=4)
    idx = build_expert_index(pkg)
    assert idx.validate() == []


def test_index_exposes_projection_geometry_and_bits(tmp_path):
    pkg, *_ = _tiny_package(tmp_path, n_layers=1, n_exp=4, out=8, cols=2)
    idx = build_expert_index(pkg)

    geo = idx.geometry(layer=0, projection="gate_proj")
    assert geo.out_features == 8
    assert geo.packed_cols == 2
    assert geo.bits == 2
    assert geo.packed_dtype == "U32"
    assert geo.norms_dtype == "F16"
    assert idx.bits(layer=0, projection="gate_proj") == 2


def test_index_rejects_inconsistent_num_experts_across_layers(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    tensors, layer_geo = {}, {}
    for layer, n_exp in ((0, 4), (1, 5)):
        comps = _layer_components(n_exp, 8, 2, seed=layer)
        bundle, geo = assemble_layer_bundle(
            comps, {p: 2 for p in ("gate_proj", "up_proj", "down_proj")})
        base = f"language_model.model.layers.{layer}.mlp.switch_mlp"
        tensors[f"{base}.experts.tq_bundle"] = ("U8", bundle.shape, bundle.tobytes())
        layer_geo[layer] = geo
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", tensors,
                       metadata={"expert_bundles": encode_bundle_metadata(layer_geo)})

    with pytest.raises(ValueError, match="inconsistent num_experts"):
        build_expert_index(pkg)


def test_index_rejects_header_metadata_shape_mismatch(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    comps = _layer_components(4, 8, 2, seed=0)
    bundle, geo = assemble_layer_bundle(
        comps, {p: 2 for p in ("gate_proj", "up_proj", "down_proj")})
    base = "language_model.model.layers.0.mlp.switch_mlp"
    # lie about the tensor shape: one expert short
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        f"{base}.experts.tq_bundle": (
            "U8", (3, bundle.shape[1]), bundle[:3].tobytes()),
    }, metadata={"expert_bundles": encode_bundle_metadata({0: geo})})

    with pytest.raises(ValueError, match="does not match"):
        build_expert_index(pkg)


def test_index_rejects_bundle_without_metadata(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    comps = _layer_components(4, 8, 2, seed=0)
    bundle, _geo = assemble_layer_bundle(
        comps, {p: 2 for p in ("gate_proj", "up_proj", "down_proj")})
    base = "language_model.model.layers.0.mlp.switch_mlp"
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        f"{base}.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes()),
    })
    with pytest.raises(ValueError, match="metadata"):
        build_expert_index(pkg)


def test_stacked_layout_fails_loud_with_reconvert_message(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    base = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        f"{base}.tq_packed": ("U32", (4, 8, 2), bytes(4 * 8 * 2 * 4)),
        f"{base}.tq_norms": ("F16", (4, 8), bytes(4 * 8 * 2)),
        f"{base}.tq_bits": ("U8", (1,), bytes([2])),
    })
    with pytest.raises(StackedLayoutError, match="re-convert"):
        build_expert_index(pkg)


def test_locate_rejects_out_of_range_expert(tmp_path):
    pkg, n_exp, *_ = _tiny_package(tmp_path, n_layers=1, n_exp=4)
    idx = build_expert_index(pkg)
    with pytest.raises((KeyError, IndexError, ValueError)):
        idx.locate(layer=0, expert=99, projection="gate_proj", component="packed")
