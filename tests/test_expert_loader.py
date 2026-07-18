"""Direct expert byte-range loader correctness (bundle format).

The streaming runtime loads one expert's packed/norms bytes from the package
shard via pread (using the bundle index) directly into mx.arrays, without
faulting the whole bundle tensor and without a Python `bytes` payload. These
tests prove the loaded component is bit-identical to the arrays the bundle was
assembled from, i.e. the loader reads the right bytes with the right
shape/dtype through the metadata geometry.

Invariants: no dequant; TQ stays packed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from conftest import write_bundle_package  # noqa: E402

from moespresso.runtime.expert_index import build_expert_index  # noqa: E402
from moespresso.runtime.expert_loader import load_expert  # noqa: E402


def _package(tmp_path, n_exp=6, out=8, cols=4):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    comps = write_bundle_package(pkg, n_layers=1, n_exp=n_exp, out=out,
                                 cols=cols, seed=3)[0]
    return pkg, comps[("gate_proj", "packed")], comps[("gate_proj", "norms")]


def test_loaded_packed_matches_assembled_component(tmp_path):
    pkg, packed, _ = _package(tmp_path)
    idx = build_expert_index(pkg)
    for e in (0, 2, 5):
        got = load_expert(pkg, idx, layer=0, expert=e, projection="gate_proj",
                          component="packed")
        assert np.array_equal(np.array(got), packed[e])


def test_loaded_norms_match_assembled_component(tmp_path):
    pkg, _, norms = _package(tmp_path)
    idx = build_expert_index(pkg)
    for e in (1, 4):
        got = load_expert(pkg, idx, layer=0, expert=e, projection="gate_proj",
                          component="norms")
        assert np.array_equal(np.array(got), norms[e])


def test_loader_reads_every_projection_through_row_geometry(tmp_path):
    """All six components of one expert come back exact from the single row."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    comps = write_bundle_package(pkg, n_layers=1, n_exp=4, out=6, cols=3,
                                 seed=7)[0]
    idx = build_expert_index(pkg)
    e = 2
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for comp in ("packed", "norms"):
            got = load_expert(pkg, idx, layer=0, expert=e, projection=proj,
                              component=comp)
            assert np.array_equal(np.array(got), comps[(proj, comp)][e]), (
                f"{proj}.{comp} mismatch")


def test_loader_uses_index_shard_per_layer(tmp_path):
    """Layer bundles may live in different shards; the index owns shard selection."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    c0 = write_bundle_package(pkg, n_layers=1, n_exp=4, out=6, cols=3, seed=11,
                              shard_name="model-00001-of-00002.safetensors")[0]
    # second shard: a different layer, written as its own one-layer package file
    from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata
    from conftest import make_layer_components, write_safetensors_raw
    c1 = make_layer_components(4, out=6, cols=3, seed=13)
    bundle, geo = assemble_layer_bundle(
        c1, {p: 2 for p in ("gate_proj", "up_proj", "down_proj")})
    write_safetensors_raw(
        pkg / "model-00002-of-00002.safetensors",
        {"language_model.model.layers.1.mlp.switch_mlp.experts.tq_bundle":
            ("U8", bundle.shape, bundle.tobytes())},
        metadata={"expert_bundles": encode_bundle_metadata({1: geo})})

    idx = build_expert_index(pkg)
    got0 = load_expert(pkg, idx, layer=0, expert=2, projection="gate_proj",
                       component="packed")
    got1 = load_expert(pkg, idx, layer=1, expert=2, projection="gate_proj",
                       component="packed")
    assert np.array_equal(np.array(got0), c0[("gate_proj", "packed")][2])
    assert np.array_equal(np.array(got1), c1[("gate_proj", "packed")][2])


def test_loader_reads_kquant_weight_component(tmp_path):
    from moespresso.package.bundle import KQUANT_CODEC, assemble_layer_bundle, encode_bundle_metadata
    from conftest import write_safetensors_raw

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    rng = np.random.default_rng(17)
    comps = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        comps[(proj, "weight")] = rng.integers(
            0,
            256,
            (3, 5, 84),
            dtype=np.uint8,
        )
        comps[(proj, "scales")] = np.zeros((3, 1), dtype=np.uint8)
    codecs = {p: KQUANT_CODEC for p in ("gate_proj", "up_proj", "down_proj")}
    bundle, geo = assemble_layer_bundle(
        comps,
        {p: 2 for p in ("gate_proj", "up_proj", "down_proj")},
        codecs=codecs,
        kquant_codecs={p: "q2_k" for p in ("gate_proj", "up_proj", "down_proj")},
    )
    write_safetensors_raw(
        pkg / "model-00001-of-00001.safetensors",
        {"layers.0.ffn.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes())},
        metadata={"expert_bundles": encode_bundle_metadata({0: geo})},
    )

    idx = build_expert_index(pkg)
    got = load_expert(
        pkg,
        idx,
        layer=0,
        expert=2,
        projection="gate_proj",
        component="weight",
    )

    assert np.array_equal(np.array(got), comps[("gate_proj", "weight")][2])
