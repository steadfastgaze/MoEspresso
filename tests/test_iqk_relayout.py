"""The IQ_K relayout: the bundle row definition and the package rebuild step.

The kernel repository proves the transform itself over the whole value space
of every member. What is checked here is the layer above it: that a relayout
row is the same width as the wire row it replaces, that the streams come
back out of a bundle row in the shapes the switch module loads, and that the
rebuild step moves a built package's artifacts and shards together and
refuses every state it cannot honestly rewrite.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from conftest import write_safetensors_raw

from moespresso.core.artifact import make_artifact, read_artifact, write_artifact
from moespresso.package import iqk_relayout as rl
from moespresso.package.bundle import (
    assemble_layer_bundle,
    encode_bundle_metadata,
)
from moespresso.package.deepseek_v4.iqk_relayout import (
    IQKRelayoutError,
    relayout_package,
)
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IK_WIRE,
    IQK_LAYOUT_IQK_RELAYOUT,
    iqk_geometry,
)
from moespresso.package.manifest import PACKAGE_FORMAT, file_identity

MEMBERS = ("iq2_ks", "iq2_k", "iq1_s_r4")
E, OUT, IN = 2, 16, 256
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _wire(codec: str, rows: int, in_features: int, seed: int) -> np.ndarray:
    """Random wire bytes with a finite fp16 scale in every scale slot."""
    rng = np.random.default_rng(seed)
    row_bytes = iqk_geometry(codec).bytes_per_row(in_features)
    out = rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
    if codec == "iq1_s_r4":
        # The scales sit as a 4 x fp16 prefix on each four-row wire group.
        assert rows % 4 == 0
        groups = out.reshape(rows // 4, 4 * row_bytes)
        values = (rng.standard_normal(rows).astype(np.float32)
                  * 0.01).astype(np.float16)
        groups[:, :8] = values.view(np.uint8).reshape(rows // 4, 8)
        return out
    nblocks = in_features // 256
    scale_slots = ([0] if codec == "iq2_ks"
                   else [b * 76 for b in range(nblocks)])
    values = (rng.standard_normal(len(scale_slots) * rows).astype(np.float32)
              * 0.01).astype(np.float16)
    raw = values.view(np.uint8).reshape(rows, len(scale_slots), 2)
    for i, slot in enumerate(scale_slots):
        out[:, slot:slot + 2] = raw[:, i]
    return out


# --------------------------------------------------------------------------
# The bundle row definition


@pytest.mark.parametrize("codec", MEMBERS)
@pytest.mark.parametrize("in_features", (256, 2048, 4096))
def test_a_relayout_row_is_the_width_of_the_wire_row_it_replaces(codec, in_features):
    """The invariant that lets both layouts live in one bundle component."""
    assert (rl.relayout_row_bytes(codec, in_features)
            == iqk_geometry(codec).bytes_per_row(in_features))


@pytest.mark.parametrize("codec", MEMBERS)
def test_pack_rows_round_trips_every_stored_byte(codec):
    wire = _wire(codec, 8, IN, seed=3)
    rows = rl.pack_rows(codec, wire, IN)
    assert rows.shape == wire.shape
    assert not np.array_equal(rows, wire)
    assert np.array_equal(rl.unpack_rows(codec, rows, IN), wire)


@pytest.mark.parametrize("codec", MEMBERS)
def test_split_streams_returns_the_switch_modules_shapes(codec):
    from mlx_iqk import format as fmt

    wire = _wire(codec, E * OUT, IN, seed=5)
    rows = rl.pack_rows(codec, wire, IN).reshape(E, OUT, -1)
    streams = rl.split_streams(codec, rows, IN)
    want = fmt.component_shapes(codec, E, OUT, IN)
    assert {k: tuple(v.shape) for k, v in streams.items()} == {
        k: tuple(v) for k, v in want.items()}
    assert {k: v.dtype for k, v in streams.items()} == fmt.component_dtypes(codec)


@pytest.mark.parametrize("codec", MEMBERS)
def test_relayout_rows_decode_to_the_wire_rows_bit_for_bit(codec):
    from mlx_iqk import codec as iqk_codec

    wire = _wire(codec, 8, IN, seed=7)
    rows = rl.pack_rows(codec, wire, IN)
    want = iqk_codec.dequantize(codec, wire, IN).astype(np.float16)
    got = rl.decode_rows(codec, rows, IN).astype(np.float16)
    assert np.array_equal(want.view(np.uint16), got.view(np.uint16))


def test_relayout_refuses_a_member_with_no_relayout():
    with pytest.raises(ValueError):
        rl.relayout_row_bytes("iq3_k", 2048)


def test_split_streams_refuses_a_row_of_the_wrong_width():
    rows = np.zeros((4, 7), dtype=np.uint8)
    with pytest.raises(ValueError):
        rl.split_streams("iq2_ks", rows, IN)


# --------------------------------------------------------------------------
# A tiny package on the streamed one-bundle-per-shard form


def _layer_members(layer: int) -> dict[str, str]:
    """Layer 1 mixes members inside one layer, as the flagship package does."""
    if layer == 1:
        return {"gate_proj": "iq2_ks", "up_proj": "iq2_k",
                "down_proj": "iq1_s_r4"}
    return {p: "iq2_ks" for p in PROJECTIONS}


def _write_bundle_shard(path, layer, members, seed):
    """One layer bundle, written the way the streamed package writer writes it."""
    comps = {}
    for i, projection in enumerate(PROJECTIONS):
        codec = members[projection]
        wire = _wire(codec, E * OUT, IN, seed=seed + 10 * i)
        comps[(projection, "blocks")] = wire.reshape(E, OUT, -1)
    bundle, geometry = assemble_layer_bundle(
        comps,
        bits={p: iqk_geometry(members[p]).bits for p in PROJECTIONS},
        codecs={p: "iqk" for p in PROJECTIONS},
        iqk_codecs=members,
        iqk_layout=IQK_LAYOUT_IK_WIRE,
    )
    key = f"layers.{layer}.ffn.experts.tq_bundle"
    header = {
        "__metadata__": {
            "format": PACKAGE_FORMAT,
            "expert_bundles": encode_bundle_metadata({layer: geometry}),
        },
        key: {"dtype": "U8", "shape": list(bundle.shape),
              "data_offsets": [0, bundle.nbytes]},
    }
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(bundle.tobytes())
    return comps


def _tiny_package(tmp_path, layers=(0, 1)):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    sources = {}
    names = []
    for layer in layers:
        name = f"model-{layer + 1:05d}-of-{len(layers):05d}.safetensors"
        sources[layer] = _write_bundle_shard(
            pkg / name, layer, _layer_members(layer), seed=100 * (layer + 1))
        names.append(name)

    producer = {"name": "test", "version": "0"}
    subject = {"model_id": "synthetic/ds4"}
    allocation = []
    tensors = []
    for layer in layers:
        members = _layer_members(layer)
        for projection in PROJECTIONS:
            codec = members[projection]
            short = projection.removesuffix("_proj")
            allocation.append({
                "source_name": f"layers.{layer}.ffn.experts.{short}",
                "kind": "expert", "role": f"moe.expert.{short}",
                "layer_index": layer, "projection": short,
                "format": "iqk", "codec": codec, "iqk_codec": codec,
                "bits": iqk_geometry(codec).bits,
                "layout": IQK_LAYOUT_IK_WIRE,
            })
            tensors.append({
                "source_name": f"layers.{layer}.ffn.experts.{short}",
                "role": f"moe.expert.{short}", "kind": "expert",
                "layer_index": layer, "projection": short,
                "shard": f"model-{layer + 1:05d}-of-{len(layers):05d}.safetensors",
                "key_prefix": f"layers.{layer}.ffn.experts",
                "format": "iqk",
                "format_params": {"iqk_codec": codec, "layout": IQK_LAYOUT_IK_WIRE},
            })
    plan = make_artifact("package_plan", subject, producer, status="valid",
                         allocation=allocation)
    write_artifact(pkg / "package_plan.json", plan)
    manifest = make_artifact(
        "package_manifest", subject, producer, status="valid",
        architecture={"family": "deepseek_v4_flash"},
        tensors=tensors,
        required_ops=["iqk_dequant"],
        files=[file_identity(pkg / name) for name in names],
        provenance={"source_plan_id": plan["artifact_id"]},
    )
    write_artifact(pkg / "package_manifest.json", manifest)
    return pkg, sources


def test_relayout_moves_shards_and_artifacts_together(tmp_path):
    from mlx_iqk import codec as iqk_codec
    from moespresso.runtime.expert_index import build_expert_index

    pkg, sources = _tiny_package(tmp_path)
    before_manifest = read_artifact(pkg / "package_manifest.json")
    before_plan = read_artifact(pkg / "package_plan.json")

    report = relayout_package(pkg, workers=1, sample_rows=4)

    after_manifest = read_artifact(pkg / "package_manifest.json")
    after_plan = read_artifact(pkg / "package_plan.json")
    assert after_manifest["artifact_id"] != before_manifest["artifact_id"]
    assert after_plan["artifact_id"] != before_plan["artifact_id"]
    assert after_manifest["provenance"]["source_plan_id"] == after_plan["artifact_id"]
    assert report["manifest_id"]["after"] == after_manifest["artifact_id"]
    assert all(t["format_params"]["layout"] == IQK_LAYOUT_IQK_RELAYOUT
               for t in after_manifest["tensors"])
    assert all(a["layout"] == IQK_LAYOUT_IQK_RELAYOUT
               for a in after_plan["allocation"])
    assert report["gates"]["rows_round_tripped"] == 2 * 3 * E * OUT

    # The declared digests are the digests on disk, and the stored bytes still
    # decode to the weights the wire bytes carried.
    for entry in after_manifest["files"]:
        assert file_identity(pkg / entry["path"]) == entry

    index = build_expert_index(pkg)
    for layer in (0, 1):
        members = _layer_members(layer)
        for projection in PROJECTIONS:
            codec = members[projection]
            geometry = index.geometry(layer=layer, projection=projection)
            assert geometry.layout == IQK_LAYOUT_IQK_RELAYOUT
            assert geometry.iqk_codec == codec
            for expert in range(E):
                got = index.locate(layer=layer, expert=expert,
                                   projection=projection, component="blocks")
                raw = np.fromfile(pkg / got.shard, dtype=np.uint8,
                                  count=got.nbytes, offset=got.offset)
                rows = raw.reshape(got.shape)
                wire = sources[layer][(projection, "blocks")][expert]
                assert np.array_equal(rl.unpack_rows(codec, rows, IN), wire)
                want = iqk_codec.dequantize(codec, wire, IN).astype(np.float16)
                have = rl.decode_rows(codec, rows, IN).astype(np.float16)
                assert np.array_equal(want.view(np.uint16), have.view(np.uint16))


def test_relayout_writes_a_copy_and_leaves_the_source_alone(tmp_path):
    pkg, _sources = _tiny_package(tmp_path, layers=(0,))
    before = [file_identity(p) for p in sorted(pkg.glob("model-*.safetensors"))]
    out = tmp_path / "relayout"

    relayout_package(pkg, output_dir=out, workers=1, sample_rows=2)

    assert [file_identity(p) for p in sorted(pkg.glob("model-*.safetensors"))] == before
    assert read_artifact(pkg / "package_manifest.json")["tensors"][0][
        "format_params"]["layout"] == IQK_LAYOUT_IK_WIRE
    assert read_artifact(out / "package_manifest.json")["tensors"][0][
        "format_params"]["layout"] == IQK_LAYOUT_IQK_RELAYOUT


def test_relayout_refuses_a_package_already_on_the_relayout(tmp_path):
    pkg, _sources = _tiny_package(tmp_path, layers=(0,))
    relayout_package(pkg, workers=1, sample_rows=2)
    with pytest.raises(IQKRelayoutError):
        relayout_package(pkg, workers=1, sample_rows=2)


def test_relayout_refuses_a_shard_that_is_not_one_bundle(tmp_path):
    from moespresso.package.deepseek_v4.iqk_relayout import relayout_shard

    pkg, _sources = _tiny_package(tmp_path, layers=(0,))
    shard = next(pkg.glob("model-*.safetensors"))
    with open(shard, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
        payload = f.read()
    key = next(k for k in header if k != "__metadata__")
    write_safetensors_raw(
        shard, {key: ("U8", header[key]["shape"], payload),
                "spare": ("U8", [1], b"\x00")},
        metadata=header["__metadata__"])
    with pytest.raises(IQKRelayoutError):
        relayout_shard(shard, shard, sample_rows=1)


def test_a_failed_rewrite_leaves_no_partial_shard(tmp_path, monkeypatch):
    from moespresso.package.deepseek_v4 import iqk_relayout as tool

    pkg, _sources = _tiny_package(tmp_path, layers=(0,))
    shard = next(pkg.glob("model-*.safetensors"))
    before = file_identity(shard)

    def _boom(*args, **kwargs):
        raise IQKRelayoutError("synthetic failure mid-rewrite")

    monkeypatch.setattr(tool, "pack_rows", _boom)
    with pytest.raises(IQKRelayoutError):
        relayout_package(pkg, workers=1, sample_rows=1)
    assert file_identity(shard) == before
    assert not list(pkg.glob("*.relayout.tmp"))
    assert read_artifact(pkg / "package_manifest.json")["tensors"][0][
        "format_params"]["layout"] == IQK_LAYOUT_IK_WIRE
