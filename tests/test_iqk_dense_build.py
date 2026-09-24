"""Shared build side of dense-tensor IQ_K support.

Covers the format registry's dense member set, the dense allocation rows,
the manifest's fail-closed dense IQ_K validation, and the writer's dense
IQ_K arm. Everything runs on synthetic tensors; no codec is exercised (the
encode is an injected callable, exactly the seam the real build uses for
the linked ik quantizer).
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from moespresso.core.artifact import make_artifact
from moespresso.package.iqk_format import (
    IQK_DENSE_MEMBERS,
    IQK_GEOMETRY,
    IQKFormatError,
    iqk_dense_geometry,
)
from moespresso.package.iqk_recipe import iqk_dense_target_from_allocation
from moespresso.package.manifest import build_package_manifest

SUBJECT = {"source_root": "toy", "source_format": "hf_safetensors"}
PRODUCER = {"tool": "test", "version": "0"}


# --------------------------------------------------------------------------
# Geometry


def test_dense_member_geometry_is_pinned_to_the_library_structs():
    # Byte counts are ik's own static_asserted block sizes; the rates below
    # are struct arithmetic at a 4096-wide row, checked against bytes.
    cases = {
        # member: (ggml_type, bytes_per_block, row_meta, bpw at 4096)
        "iq4_ks": (144, 136, 4, 4.2578125),
        "iq4_k": (139, 144, 0, 4.5),
        "iq5_k": (140, 176, 0, 5.5),
        "iq6_k": (141, 212, 0, 6.625),
    }
    for member, (ggml_type, bpb, meta, bpw) in cases.items():
        geometry = IQK_GEOMETRY[member]
        assert geometry.ggml_type == ggml_type
        assert geometry.weights_per_block == 256
        assert geometry.bytes_per_block == bpb
        assert geometry.row_meta_bytes == meta
        assert geometry.bytes_per_row(4096) == meta + 16 * bpb
        assert geometry.bpw(4096) == bpw
        # The inverse must round-trip: a stored row width recomputes the
        # logical width, which is what catches a swapped member on disk.
        assert geometry.in_features_for_row_bytes(
            geometry.bytes_per_row(4096)) == 4096


def test_dense_member_allow_list_refuses_routed_members():
    assert set(IQK_DENSE_MEMBERS) == {"iq4_ks", "iq4_k", "iq5_k", "iq6_k"}
    for member in ("iq1_s_r4", "iq2_ks", "iq2_k", "iq3_k"):
        with pytest.raises(IQKFormatError, match="not a dense member"):
            iqk_dense_geometry(member)
    with pytest.raises(IQKFormatError, match="not a dense member"):
        iqk_dense_geometry("q8_0")


# --------------------------------------------------------------------------
# Manifest validation


ARCH = {
    "model_type": "qwen3_moe",
    "text_config": {
        "num_hidden_layers": 2, "hidden_size": 2048, "num_experts": 256,
        "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        "layer_types": ["linear_attention", "full_attention"],
        "vocab_size": 1000,
    },
}


def _dense_iqk_plan(**overrides):
    row = {
        "source_name": "layers.0.attn.wq_a.weight",
        "kind": "affine",
        "role": "attn.wq_a",
        "layer_index": 0,
        "bits": 6,
        "format": "iqk",
        "codec": "iq6_k",
        "iqk_codec": "iq6_k",
        "layout": "ik_wire",
        "gguf_tensor": "blk.0.attn_q_a.weight",
        "imatrix_key": "blk.0.attn_q_a.weight",
        "module_path": "model.layers.0.self_attn.wq_a",
        "module_weight_key": "model.layers.0.self_attn.wq_a.weight",
    }
    row.update(overrides)
    return make_artifact(
        "package_plan", SUBJECT, PRODUCER, status="valid", allocation=[row])


def test_dense_iqk_target_rebuilds_from_a_shared_allocation_row():
    row = _dense_iqk_plan()["allocation"][0]
    target = iqk_dense_target_from_allocation(row)
    assert target.codec == "iq6_k"
    assert target.layout == "ik_wire"
    assert target.imatrix_key == row["gguf_tensor"]

    broken = dict(row)
    broken["iqk_codec"] = "iq2_k"
    broken["codec"] = "iq2_k"
    with pytest.raises(ValueError, match="dense member"):
        iqk_dense_target_from_allocation(broken)


def _located():
    return {"layers.0.attn.wq_a.weight": {
        "shard": "model-00001-of-00001.safetensors",
        "key_prefix": "layers.0.attn.wq_a",
    }}


def _files():
    return [{"path": "model-00001-of-00001.safetensors",
             "size_bytes": 1, "sha256": "0" * 64}]


def test_manifest_records_a_dense_iqk_entry():
    man = build_package_manifest(_dense_iqk_plan(), ARCH, _located(), _files())
    assert man["status"] == "valid"
    entry = next(t for t in man["tensors"]
                 if t["source_name"] == "layers.0.attn.wq_a.weight")
    assert entry["format"] == "iqk"
    assert entry["format_params"]["iqk_codec"] == "iq6_k"
    assert entry["format_params"]["layout"] == "ik_wire"
    assert entry["format_params"]["bytes_per_block"] == 212
    assert entry["format_params"]["row_meta_bytes"] == 0
    assert entry["module_weight_key"] == "model.layers.0.self_attn.wq_a.weight"
    assert "iqk_dequant" in man["required_ops"]


def test_manifest_refuses_a_routed_member_on_a_dense_tensor():
    plan = _dense_iqk_plan(codec="iq2_ks", iqk_codec="iq2_ks", bits=2)
    man = build_package_manifest(plan, ARCH, _located(), _files())
    assert man["status"] == "invalid"
    codes = {v["code"] for v in man["validation"] if v["blocking"]}
    assert "package.unsupported_dense_iqk_member" in codes


def test_manifest_refuses_a_dense_iqk_entry_without_a_module_key():
    plan = _dense_iqk_plan(module_weight_key=None)
    man = build_package_manifest(plan, ARCH, _located(), _files())
    assert man["status"] == "invalid"
    codes = {v["code"] for v in man["validation"] if v["blocking"]}
    assert "package.missing_iqk_module_weight_key" in codes


def test_manifest_refuses_an_unknown_dense_iqk_layout():
    plan = _dense_iqk_plan(layout="row_major")
    man = build_package_manifest(plan, ARCH, _located(), _files())
    assert man["status"] == "invalid"
    codes = {v["code"] for v in man["validation"] if v["blocking"]}
    assert "package.unsupported_iqk_layout" in codes


# --------------------------------------------------------------------------
# The writer arm


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, (dtype, arr) in tensors.items():
        a = np.ascontiguousarray(arr)
        data = a.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(a.shape),
            "data_offsets": [off, off + len(data)],
        }
        blob += data
        off += len(data)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _dense_source(tmp_path, *, out_features=16, in_features=512):
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(7)
    dense = rng.standard_normal((out_features, in_features)).astype(np.float32)
    _write_safetensors(src / "model-00001.safetensors", {
        "layers.0.attn.wq_a.weight": ("F32", dense),
    })
    return src, dense


def _stub_encoder(calls):
    def encode(matrix, target, imatrix):
        geometry = iqk_dense_geometry(target.codec)
        calls.append({
            "codec": target.codec,
            "shape": tuple(int(v) for v in np.asarray(matrix).shape),
            "imatrix_len": int(np.asarray(imatrix).shape[-1]),
        })
        rows = int(matrix.shape[0])
        bpr = geometry.bytes_per_row(int(matrix.shape[1]))
        return np.full((rows, bpr), 0xA5, dtype=np.uint8)
    return encode


def test_write_package_encodes_a_dense_iqk_tensor(tmp_path):
    pytest.importorskip("mlx.core")
    from moespresso.package.write import write_package
    from safetensors.numpy import load_file

    src, dense = _dense_source(tmp_path)
    out = tmp_path / "out"
    plan = _dense_iqk_plan()
    calls = []
    vectors = {"blk.0.attn_q_a.weight": np.ones(512, dtype=np.float32)}

    man = write_package(
        plan, src, ARCH, out,
        iqk_dense_encoder=_stub_encoder(calls),
        kquant_imatrix_vectors=vectors,
    )

    assert man["status"] == "valid"
    assert calls == [{
        "codec": "iq6_k", "shape": (16, 512), "imatrix_len": 512}]
    entry = next(t for t in man["tensors"]
                 if t["source_name"] == "layers.0.attn.wq_a.weight")
    assert entry["format"] == "iqk"
    arrays = {}
    for file in man["files"]:
        arrays.update(load_file(str(out / file["path"])))
    wire = arrays["layers.0.attn.wq_a.weight"]
    assert wire.dtype == np.uint8
    assert wire.shape == (16, IQK_GEOMETRY["iq6_k"].bytes_per_row(512))


def test_write_package_refuses_a_dense_iqk_tensor_without_an_encoder(tmp_path):
    pytest.importorskip("mlx.core")
    from moespresso.package.write import write_package

    src, _dense = _dense_source(tmp_path)
    vectors = {"blk.0.attn_q_a.weight": np.ones(512, dtype=np.float32)}
    with pytest.raises(ValueError, match="iqk_dense_encoder"):
        write_package(
            _dense_iqk_plan(), src, ARCH, tmp_path / "out",
            kquant_imatrix_vectors=vectors,
        )


def test_write_package_refuses_a_dense_iqk_tensor_without_an_imatrix(tmp_path):
    pytest.importorskip("mlx.core")
    from moespresso.package.write import write_package

    src, _dense = _dense_source(tmp_path)
    with pytest.raises(ValueError, match="missing imatrix vector"):
        write_package(
            _dense_iqk_plan(), src, ARCH, tmp_path / "out",
            iqk_dense_encoder=_stub_encoder([]),
        )


def test_write_package_refuses_a_wrong_sized_dense_iqk_wire(tmp_path):
    pytest.importorskip("mlx.core")
    from moespresso.package.write import write_package

    src, _dense = _dense_source(tmp_path)
    vectors = {"blk.0.attn_q_a.weight": np.ones(512, dtype=np.float32)}

    def short_encoder(matrix, target, imatrix):
        return np.zeros((int(matrix.shape[0]), 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="returned shape"):
        write_package(
            _dense_iqk_plan(), src, ARCH, tmp_path / "out",
            iqk_dense_encoder=short_encoder,
            kquant_imatrix_vectors=vectors,
        )
