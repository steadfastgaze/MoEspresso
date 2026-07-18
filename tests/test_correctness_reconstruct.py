"""L1 tensor reconstruction evidence.

The tests are model-free fixtures: no full model load, no serve. They validate that L1 reads
source/package tensor slices, reconstructs package formats, and emits correctness evidence
with honest reference provenance.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from moespresso.core.artifact import validate_base
from moespresso.correctness.reconstruct import l1_tensor_reconstruction
from moespresso.inventory.architecture_profile import (
    deepseek_v4_flash_profile,
    qwen3_5_moe_profile,
)
from moespresso.inventory.build import build_inventory


def _st_meta(value, off):
    dtype = None
    arr = value
    if isinstance(value, tuple):
        dtype, arr = value
    a = np.ascontiguousarray(arr)
    meta = {
        "dtype": dtype or _dtype_tag(a),
        "shape": list(a.shape),
        "data_offsets": [off, off + a.nbytes],
    }
    return meta, a.tobytes()


def _dtype_tag(arr):
    if arr.dtype == np.float32:
        return "F32"
    if arr.dtype == np.float16:
        return "F16"
    if arr.dtype == np.uint32:
        return "U32"
    if arr.dtype == np.uint8:
        return "U8"
    if arr.dtype == np.int8:
        return "I8"
    if arr.dtype == np.int64:
        return "I64"
    raise AssertionError(f"test helper does not support dtype {arr.dtype}")


def _write_safetensors(path, tensors, metadata=None):
    header, blob, off = {}, bytearray(), 0
    if metadata:
        header["__metadata__"] = dict(metadata)
    for name, arr in tensors.items():
        meta, data = _st_meta(arr, off)
        header[name] = meta
        blob += data
        off += len(data)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _manifest_entry(source_name, *, fmt, key_prefix=None, role="attn.q_proj",
                    kind="affine", shard="model-00001-of-00001.safetensors",
                    params=None, projection=None, layer_index=None):
    out = {"source_name": source_name, "role": role, "kind": kind, "shard": shard,
           "key_prefix": key_prefix or source_name, "format": fmt,
           "format_params": params or {}}
    if projection is not None:
        out["projection"] = projection
    if layer_index is not None:
        out["layer_index"] = layer_index
    return out


def test_l1_structural_passthrough_reconstructs_source_storage(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    norm_name = "model.language_model.layers.0.input_layernorm.weight"
    conv_name = "model.language_model.layers.0.linear_attn.conv1d.weight"
    norm = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    conv = np.arange(8, dtype=np.float32).reshape(2, 1, 4)
    _write_safetensors(src / "model-00001.safetensors", {norm_name: norm, conv_name: conv})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        norm_name: norm.astype(np.float16),
        conv_name: conv.astype(np.float16),
    })
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test", "tensors": [
               {"source_name": norm_name, "role": "norm.input_layernorm",
                "kind": "passthrough", "status": "required"},
               {"source_name": conv_name, "role": "ssm.conv1d",
                "kind": "passthrough", "status": "required"},
           ]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(norm_name, fmt="fp16", role="norm.input_layernorm",
                        kind="passthrough"),
        _manifest_entry(conv_name, fmt="fp16", role="ssm.conv1d", kind="passthrough"),
    ]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg)

    assert validate_base(ev) == []
    assert ev["status"] == "valid"
    assert ev["summary"]["sampled_by_format"]["fp16"] == 2
    assert any(r["component"] == "passthrough" and r["kind"] == "independent"
               for r in ev["reference_provenance"])


def test_l1_structural_passthrough_blocks_wrong_stored_values(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "model.language_model.layers.0.input_layernorm.weight"
    _write_safetensors(src / "model-00001.safetensors",
                       {name: np.array([0.0, 1.0], dtype=np.float32)})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors",
                       {name: np.array([9.0, 9.0], dtype=np.float16)})
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "norm.input_layernorm",
                        "kind": "passthrough", "status": "required"}]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(name, fmt="fp16", role="norm.input_layernorm", kind="passthrough")]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg)

    assert ev["status"] == "invalid"
    assert any(v["code"] == "correctness.reconstruction_error" and v["blocking"]
               for v in ev["validation"])


def test_l1_deepseek_v4_router_gate_accepts_reduced_expert_smoke_slice(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "layers.0.ffn.gate.weight"
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    _write_safetensors(src / "model-00001.safetensors", {name: source})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        name: source[:1].astype(np.float16),
    })
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "moe.router_gate",
                        "kind": "passthrough", "status": "required"}]}
    man = {"artifact_id": "pkg:test",
           "architecture": {"family": "deepseek_v4_flash", "smoke_max_experts": 1},
           "tensors": [
               _manifest_entry(name, fmt="fp16", role="moe.router_gate",
                               kind="passthrough")
           ]}

    ev = l1_tensor_reconstruction(deepseek_v4_flash_profile(), inv, man, src, pkg)

    assert ev["status"] == "valid"
    assert ev["summary"]["sampled_by_format"]["fp16"] == 1


def test_l1_router_gate_truncation_blocks_without_smoke_declaration(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "model.language_model.layers.0.mlp.gate.weight"
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    _write_safetensors(src / "model-00001.safetensors", {name: source})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        name: source[:1].astype(np.float16),
    })
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "moe.router_gate",
                        "kind": "passthrough", "status": "required"}]}
    man = {"artifact_id": "pkg:test",
           "architecture": {"family": "qwen3_5_moe", "smoke_max_experts": None},
           "tensors": [
               _manifest_entry(name, fmt="fp16", role="moe.router_gate",
                               kind="passthrough")
           ]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg)

    assert ev["status"] == "invalid"
    assert any(v["code"] == "correctness.shape_mismatch" and v["blocking"]
               for v in ev["validation"])


def test_l1_router_gate_smoke_slice_still_checks_values(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "layers.0.ffn.gate.weight"
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    wrong = source[:1].copy()
    wrong[0, 0] += 10.0
    _write_safetensors(src / "model-00001.safetensors", {name: source})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        name: wrong.astype(np.float16),
    })
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "moe.router_gate",
                        "kind": "passthrough", "status": "required"}]}
    man = {"artifact_id": "pkg:test",
           "architecture": {"family": "deepseek_v4_flash", "smoke_max_experts": 1},
           "tensors": [
               _manifest_entry(name, fmt="fp16", role="moe.router_gate",
                               kind="passthrough")
           ]}

    ev = l1_tensor_reconstruction(deepseek_v4_flash_profile(), inv, man, src, pkg)

    assert ev["status"] == "invalid"
    assert any(v["code"] == "correctness.reconstruction_error" and v["blocking"]
               for v in ev["validation"])


def test_l1_router_gate_smoke_slice_requires_declared_row_count(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "layers.0.ffn.gate.weight"
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    _write_safetensors(src / "model-00001.safetensors", {name: source})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        name: source[:1].astype(np.float16),
    })
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "moe.router_gate",
                        "kind": "passthrough", "status": "required"}]}
    man = {"artifact_id": "pkg:test",
           "architecture": {"family": "deepseek_v4_flash", "smoke_max_experts": 2},
           "tensors": [
               _manifest_entry(name, fmt="fp16", role="moe.router_gate",
                               kind="passthrough")
           ]}

    ev = l1_tensor_reconstruction(deepseek_v4_flash_profile(), inv, man, src, pkg)

    assert ev["status"] == "invalid"
    assert any(v["code"] == "correctness.shape_mismatch" and v["blocking"]
               for v in ev["validation"])


def test_l1_raw_dtype_passthrough_requires_storage_identity(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "layers.0.ffn.gate.tid2eid"
    table = np.array([[1, 2, 3, 4, 5, 6], [250, 251, 252, 253, 254, 255]], dtype=np.int64)
    _write_safetensors(src / "model-00001.safetensors", {name: table})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {name: table})
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "moe.router_tid2eid",
                        "kind": "passthrough", "status": "required",
                        "format": "raw_dtype_passthrough"}]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(name, fmt="raw_dtype_passthrough",
                        role="moe.router_tid2eid", kind="passthrough")]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg)

    assert ev["status"] == "valid"
    assert ev["summary"]["sampled_by_format"]["raw_dtype_passthrough"] == 1
    assert ev["metrics"][0]["format"] == "raw_dtype_passthrough"


def test_l1_raw_dtype_passthrough_blocks_storage_change(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "layers.2.attn.compressor.ape"
    _write_safetensors(src / "model-00001.safetensors",
                       {name: np.array([[1.0, 2.0]], dtype=np.float32)})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors",
                       {name: np.array([[1.0, 3.0]], dtype=np.float32)})
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "attn.compressor.ape",
                        "kind": "passthrough", "status": "required",
                        "format": "raw_dtype_passthrough"}]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(name, fmt="raw_dtype_passthrough",
                        role="attn.compressor.ape", kind="passthrough")]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg)

    assert ev["status"] == "invalid"
    assert any(v["code"] == "correctness.raw_passthrough_mismatch"
               for v in ev["validation"])


def test_l1_f32_passthrough_requires_float32_storage(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "model.language_model.layers.0.linear_attn.in_proj_a.weight"
    source = np.array([[1.25, -2.5], [3.0, 4.5]], dtype=np.float16)
    _write_safetensors(src / "model-00001.safetensors", {name: source})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors",
                       {name: source.astype(np.float32)})
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "attn.linear_in_proj_a",
                        "kind": "passthrough", "status": "required",
                        "format": "f32_passthrough"}]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(name, fmt="f32_passthrough",
                        role="attn.linear_in_proj_a", kind="passthrough")]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg)

    assert ev["status"] == "valid"
    assert ev["summary"]["sampled_by_format"]["f32_passthrough"] == 1
    assert ev["metrics"][0]["format"] == "f32_passthrough"


def test_l1_f32_passthrough_blocks_float16_storage(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "model.language_model.layers.0.linear_attn.in_proj_b.weight"
    source = np.array([[1.25, -2.5], [3.0, 4.5]], dtype=np.float16)
    _write_safetensors(src / "model-00001.safetensors", {name: source})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {name: source})
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "attn.linear_in_proj_b",
                        "kind": "passthrough", "status": "required",
                        "format": "f32_passthrough"}]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(name, fmt="f32_passthrough",
                        role="attn.linear_in_proj_b", kind="passthrough")]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg)

    assert ev["status"] == "invalid"
    assert any(v["code"] == "correctness.dtype_mismatch" and v["blocking"]
               for v in ev["validation"])


def test_l1_affine_reconstruction_checks_sidecars_and_uses_external_codec(tmp_path):
    mx = pytest.importorskip("mlx.core")
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "model.language_model.layers.0.self_attn.q_proj.weight"
    weight = np.arange(256, dtype=np.float32).reshape(8, 32) / 32.0
    qw, scales, biases = mx.quantize(mx.array(weight), bits=4, group_size=32, mode="affine")
    mx.eval(qw, scales, biases)
    _write_safetensors(src / "model-00001.safetensors", {name: weight})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        f"{name}.weight": np.asarray(qw),
        f"{name}.scales": np.asarray(scales, dtype=np.float16),
        f"{name}.biases": np.asarray(biases, dtype=np.float16),
    })
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "attn.q_proj", "kind": "affine",
                        "status": "required"}]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(name, fmt="affine", params={"bits": 4, "group_size": 32})]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg,
                                  sample_policy={"affine_tensors": 1, "rows_per_tensor": 4})

    assert ev["status"] == "valid"
    assert any(r["component"] == "affine" and r["kind"] == "external_codec"
               for r in ev["reference_provenance"])
    assert ev["summary"]["sampled_by_format"]["affine"] == 1


def test_l1_deepseek_v4_affine_reconstruction_decodes_fp8_source_rows(tmp_path):
    mx = pytest.importorskip("mlx.core")
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "layers.0.attn.wq_a.weight"
    weight_codes = np.full((128, 128), 0x38, dtype=np.uint8)
    scales_src = np.array([[127]], dtype=np.uint8)
    source_value = np.ones((128, 128), dtype=np.float32)
    qw, scales, biases = mx.quantize(
        mx.array(source_value),
        bits=4,
        group_size=128,
        mode="affine",
    )
    mx.eval(qw, scales, biases)
    _write_safetensors(src / "model-00001.safetensors", {
        name: ("F8_E4M3", weight_codes),
        "layers.0.attn.wq_a.scale": ("F8_E8M0", scales_src),
    })
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        "layers.0.attn.wq_a.weight": np.asarray(qw),
        "layers.0.attn.wq_a.scales": np.asarray(scales, dtype=np.float16),
        "layers.0.attn.wq_a.biases": np.asarray(biases, dtype=np.float16),
    })
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "attn.wq_a", "kind": "affine",
                        "status": "required"}]}
    man = {"artifact_id": "pkg:test",
           "architecture": {"family": "deepseek_v4_flash"},
           "tensors": [
               _manifest_entry(
                   name,
                   fmt="affine",
                   key_prefix="layers.0.attn.wq_a",
                   role="attn.wq_a",
                   params={"bits": 4, "group_size": 128},
               )
           ]}

    ev = l1_tensor_reconstruction(
        deepseek_v4_flash_profile(),
        inv,
        man,
        src,
        pkg,
        sample_policy={"affine_tensors": 1, "rows_per_tensor": 4},
    )

    assert ev["status"] == "valid"
    assert ev["summary"]["sampled_by_format"]["affine"] == 1


def test_l1_dense_mxfp8_reconstruction_uses_mx_float_dequant(tmp_path):
    mx = pytest.importorskip("mlx.core")
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "layers.0.attn.wq_b.weight"
    weight = np.ones((64, 64), dtype=np.float32)
    qw, scales = mx.quantize(mx.array(weight), bits=8, group_size=32, mode="mxfp8")
    mx.eval(qw, scales)
    _write_safetensors(src / "model-00001.safetensors", {name: weight})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        f"{name}.weight": np.asarray(qw),
        f"{name}.scales": np.asarray(scales, dtype=np.uint8),
    })
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "attn.wq_b", "kind": "affine",
                        "status": "required"}]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(
            name,
            fmt="mxfp8",
            params={"bits": 8, "group_size": 32},
        )
    ]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg,
                                  sample_policy={"affine_tensors": 1, "rows_per_tensor": 4})

    assert ev["status"] == "valid"
    assert ev["summary"]["sampled_by_format"]["mxfp8"] == 1


def test_l1_affine_reconstruction_blocks_missing_sidecar(tmp_path):
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "model.language_model.layers.0.self_attn.q_proj.weight"
    _write_safetensors(src / "model-00001.safetensors",
                       {name: np.zeros((4, 8), dtype=np.float32)})
    _write_safetensors(pkg / "model-00001-of-00001.safetensors",
                       {f"{name}.weight": np.zeros((4, 1), dtype=np.uint32)})
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "attn.q_proj", "kind": "affine",
                        "status": "required"}]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(name, fmt="affine", params={"bits": 4, "group_size": 8})]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg)

    assert ev["status"] == "invalid"
    assert any(v["code"] == "correctness.missing_package_tensor" and v["blocking"]
               for v in ev["validation"])


_BUNDLE_PREFIX = "language_model.model.layers.0.mlp.switch_mlp.experts"


def _tq_layer_fixture(src, pkg, *, swap_gate_up=False):
    """One bundle-format layer (1 expert): fused gate_up source + down source.

    Returns the (inv, man) pair for L1. `swap_gate_up=True` writes up's payload
    into gate's bundle slot and vice versa (the classic silent-corruption mode
    the ladder must catch).
    """
    import jang_tools.turboquant.linear as tq
    from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata

    name_gu = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    name_dn = "model.language_model.layers.0.mlp.experts.down_proj"
    gate = np.arange(64, dtype=np.float32).reshape(4, 16) / 20.0
    up = (100 + np.arange(64, dtype=np.float32)).reshape(4, 16) / 20.0
    down = (200 + np.arange(64, dtype=np.float32)).reshape(4, 16) / 20.0
    _write_safetensors(src / "model-00001.safetensors", {
        name_gu: np.stack([np.concatenate([gate, up], axis=0)]),
        name_dn: down[None, ...],
    })

    q = {p: tq.tq_quantize_weight(w, bits=2, seed=42)
         for p, w in (("gate", gate), ("up", up), ("down", down))}
    if swap_gate_up:
        q["gate"], q["up"] = q["up"], q["gate"]
    comps = {}
    for p in ("gate", "up", "down"):
        comps[(f"{p}_proj", "packed")] = np.asarray(q[p]["packed"])[None, ...]
        comps[(f"{p}_proj", "norms")] = np.asarray(q[p]["norms"])[None, ...]
    bundle, geo = assemble_layer_bundle(
        comps, {"gate_proj": 2, "up_proj": 2, "down_proj": 2})
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors",
        {f"{_BUNDLE_PREFIX}.tq_bundle": bundle},
        metadata={"expert_bundles": encode_bundle_metadata({0: geo})})

    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name_gu, "role": "moe.expert.gate",
                        "kind": "expert", "projection": "gate_up",
                        "status": "required"},
                       {"source_name": name_dn, "role": "moe.expert.down",
                        "kind": "expert", "projection": "down",
                        "status": "required"}]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(name_gu, fmt="tq", key_prefix=_BUNDLE_PREFIX,
                        role="moe.expert.gate", kind="expert",
                        params={"bits": 2, "seed": 42}, projection="gate",
                        layer_index=0),
        _manifest_entry(name_gu, fmt="tq", key_prefix=_BUNDLE_PREFIX,
                        role="moe.expert.up", kind="expert",
                        params={"bits": 2, "seed": 42}, projection="up",
                        layer_index=0),
        _manifest_entry(name_dn, fmt="tq", key_prefix=_BUNDLE_PREFIX,
                        role="moe.expert.down", kind="expert",
                        params={"bits": 2, "seed": 42}, projection="down",
                        layer_index=0),
    ]}
    return inv, man


def test_l1_tq_reconstruction_checks_bits_and_gate_up_mapping(tmp_path):
    pytest.importorskip("jang_tools.turboquant.linear")
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    inv, man = _tq_layer_fixture(src, pkg)

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg,
                                  sample_policy={"tq_experts": 1, "rows_per_expert": 4})

    assert ev["status"] == "valid"
    assert any(r["component"] == "tq" and r["kind"] == "independent"
               for r in ev["reference_provenance"])
    assert ev["summary"]["sampled_by_format"]["tq"] == 3


def test_l1_tq_reconstruction_blocks_swapped_gate_up_payloads(tmp_path):
    pytest.importorskip("jang_tools.turboquant.linear")
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    inv, man = _tq_layer_fixture(src, pkg, swap_gate_up=True)

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg,
                                  sample_policy={"tq_experts": 1, "rows_per_expert": 4})

    assert ev["status"] == "invalid"
    assert any(v["code"] == "correctness.reconstruction_error" and v["blocking"]
               for v in ev["validation"])


def test_l1_tq_reconstruction_blocks_wrong_tq_bits(tmp_path):
    from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata
    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    name = "model.language_model.layers.0.mlp.experts.down_proj"
    _write_safetensors(src / "model-00001.safetensors",
                       {name: np.zeros((1, 4, 16), dtype=np.float32)})
    comps = {}
    for p in ("gate_proj", "up_proj", "down_proj"):
        comps[(p, "packed")] = np.zeros((1, 4, 1), dtype=np.uint32)
        comps[(p, "norms")] = np.ones((1, 4), dtype=np.float16)
    # the bundle metadata declares bits=4; the manifest will say 2
    bundle, geo = assemble_layer_bundle(
        comps, {"gate_proj": 4, "up_proj": 4, "down_proj": 4})
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors",
        {f"{_BUNDLE_PREFIX}.tq_bundle": bundle},
        metadata={"expert_bundles": encode_bundle_metadata({0: geo})})
    inv = {"subject": {"source_root": "toy", "source_format": "hf_safetensors"},
           "artifact_id": "inv:test",
           "tensors": [{"source_name": name, "role": "moe.expert.down",
                        "kind": "expert", "projection": "down_proj",
                        "status": "required"}]}
    man = {"artifact_id": "pkg:test", "tensors": [
        _manifest_entry(name, fmt="tq", key_prefix=_BUNDLE_PREFIX,
                        role="moe.expert.down", kind="expert",
                        params={"bits": 2, "seed": 42}, projection="down",
                        layer_index=0)]}

    ev = l1_tensor_reconstruction(qwen3_5_moe_profile(), inv, man, src, pkg)

    assert ev["status"] == "invalid"
    assert any(v["code"] == "correctness.tq_bits_mismatch" and v["blocking"]
               for v in ev["validation"])


def test_l1_deepseek_v4_tq_reconstruction_uses_logical_expert_adapter(tmp_path):
    pytest.importorskip("jang_tools.turboquant.linear")
    from jang_tools.turboquant.linear import tq_quantize_weight
    from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata
    from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup

    src = tmp_path / "src"
    pkg = tmp_path / "pkg"
    src.mkdir()
    pkg.mkdir()
    packed_gate = np.arange(16, dtype=np.uint8).reshape(1, 16).view(np.int8)
    packed_down = np.arange(32, dtype=np.uint8).reshape(2, 16).view(np.int8)
    tensors = {}
    for expert in (0, 1):
        tensors[f"layers.0.ffn.experts.{expert}.w1.weight"] = packed_gate
        tensors[f"layers.0.ffn.experts.{expert}.w3.weight"] = packed_gate
        tensors[f"layers.0.ffn.experts.{expert}.w2.weight"] = packed_down
        tensors[f"layers.0.ffn.experts.{expert}.w1.scale"] = (
            "F8_E8M0",
            np.array([[127]], dtype=np.uint8),
        )
        tensors[f"layers.0.ffn.experts.{expert}.w3.scale"] = (
            "F8_E8M0",
            np.array([[127]], dtype=np.uint8),
        )
        tensors[f"layers.0.ffn.experts.{expert}.w2.scale"] = (
            "F8_E8M0",
            np.array([[127], [128]], dtype=np.uint8),
        )
    _write_safetensors(src / "model-00001.safetensors", tensors)
    inv = build_inventory(src, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, src, fp4_block=32)

    comps = {}
    for projection in ("gate", "up", "down"):
        packed_parts, norms_parts = [], []
        for expert in group.experts(0):
            decoded = group.decode(
                layer=0,
                expert_index=expert,
                projection=projection,
                out_dtype=np.float32,
            )
            q = tq_quantize_weight(decoded, bits=2, seed=42)
            packed_parts.append(np.asarray(q["packed"]))
            norms_parts.append(np.asarray(q["norms"]))
        comps[(f"{projection}_proj", "packed")] = np.stack(packed_parts, axis=0)
        comps[(f"{projection}_proj", "norms")] = np.stack(norms_parts, axis=0)
    bundle, geo = assemble_layer_bundle(
        comps,
        {"gate_proj": 2, "up_proj": 2, "down_proj": 2},
    )
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors",
        {"layers.0.ffn.experts.tq_bundle": bundle},
        metadata={"expert_bundles": encode_bundle_metadata({0: geo})},
    )
    man = {"artifact_id": "pkg:test",
           "architecture": {"family": "deepseek_v4_flash"},
           "tensors": [
               _manifest_entry("layers.0.ffn.experts.gate", fmt="tq",
                               key_prefix="layers.0.ffn.experts",
                               role="moe.expert.gate", kind="expert",
                               params={"bits": 2, "seed": 42}, projection="gate",
                               layer_index=0),
               _manifest_entry("layers.0.ffn.experts.up", fmt="tq",
                               key_prefix="layers.0.ffn.experts",
                               role="moe.expert.up", kind="expert",
                               params={"bits": 2, "seed": 42}, projection="up",
                               layer_index=0),
               _manifest_entry("layers.0.ffn.experts.down", fmt="tq",
                               key_prefix="layers.0.ffn.experts",
                               role="moe.expert.down", kind="expert",
                               params={"bits": 2, "seed": 42}, projection="down",
                               layer_index=0),
           ]}

    ev = l1_tensor_reconstruction(
        deepseek_v4_flash_profile(),
        inv,
        man,
        src,
        pkg,
        sample_policy={"tq_tensors": 3, "tq_experts": 2, "rows_per_expert": 1},
    )

    assert ev["status"] == "valid"
    assert ev["summary"]["sampled_by_format"]["tq"] == 3
