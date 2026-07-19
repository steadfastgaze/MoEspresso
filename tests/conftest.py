"""Shared test fixture helpers (importable as `from conftest import ...`).

The bundle format gives every runtime test the same package-building need:
a minimal bundle-format shard (one `...switch_mlp.experts.tq_bundle` per layer
+ geometry metadata). One builder here keeps the offset math in
moespresso.package.bundle, so fixtures can never drift from the writer.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from moespresso.package.bundle import (
    assemble_layer_bundle,
    encode_bundle_metadata,
)


@pytest.fixture(autouse=True)
def _disk_kv_off_for_tests(monkeypatch):
    """Serving defaults the disk KV store on under the user cache directory.

    The suite must never write checkpoints into the real user cache, so
    every test runs with the kill switch set. A test that exercises the
    store sets its own environment (a test-level monkeypatch overrides
    this fixture) or passes an explicit env dict to the resolver.
    """
    monkeypatch.setenv("MOESPRESSO_DISK_KV", "off")

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def write_safetensors_raw(path, tensors, metadata=None):
    """tensors: name -> (dtype_str, shape, raw_bytes). Minimal safetensors writer."""
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


def make_layer_components(n_exp, out=8, cols=2, *, specs=None, seed=0):
    """Random stacked components for one layer, keyed (projection, component).

    `specs` ({projection: (out, cols)}) overrides the uniform out/cols: real
    layers are asymmetric (gate/up [moe_inter, ...], down [hidden, ...]).
    """
    if specs is None:
        specs = {p: (out, cols) for p in PROJECTIONS}
    rng = np.random.default_rng(seed)
    comps = {}
    for proj in PROJECTIONS:
        p_out, p_cols = specs[proj]
        comps[(proj, "packed")] = rng.integers(
            0, 2**32, (n_exp, p_out, p_cols), dtype=np.uint32)
        comps[(proj, "norms")] = (
            rng.standard_normal((n_exp, p_out)) * 0.1).astype(np.float16)
    return comps


def write_bundle_package(pkg_dir, *, n_layers=1, layers=None, n_exp=4, out=8,
                         cols=2, bits=2, specs=None, seed=0, extra_tensors=None,
                         shard_name="model-00001-of-00001.safetensors"):
    """Write a minimal bundle-format package into `pkg_dir` (must exist).

    `specs` ({projection: (out, cols, bits)}) overrides the uniform
    out/cols/bits; `layers` (iterable of layer indexes) overrides
    range(n_layers). Returns {layer: components} with the exact numpy arrays
    that went into each bundle, for byte-level assertions in loader/pool tests.
    """
    if specs is None:
        specs = {p: (out, cols, bits) for p in PROJECTIONS}
    shape_specs = {p: (o, c) for p, (o, c, _b) in specs.items()}
    bits_map = {p: b for p, (_o, _c, b) in specs.items()}
    if layers is None:
        layers = range(n_layers)
    tensors, layer_geo, by_layer = {}, {}, {}
    for layer in layers:
        comps = make_layer_components(n_exp, specs=shape_specs, seed=seed + layer)
        bundle, geo = assemble_layer_bundle(comps, bits_map)
        base = f"language_model.model.layers.{layer}.mlp.switch_mlp"
        tensors[f"{base}.experts.tq_bundle"] = ("U8", bundle.shape, bundle.tobytes())
        layer_geo[layer] = geo
        by_layer[layer] = comps
    tensors.update(extra_tensors or {})
    write_safetensors_raw(
        pkg_dir / shard_name, tensors,
        metadata={"format": "mjtq",
                  "expert_bundles": encode_bundle_metadata(layer_geo)})
    return by_layer
