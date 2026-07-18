"""Package write path: real mlx + jang end-to-end on a synthetic model.

Requires the runtime dependencies. Builds a tiny safetensors model, runs
inventory -> probe -> decide -> write_package, then reloads the written shard and
confirms its keys/contents match what the manifest declares (the engine-never-
guesses contract holds against real bytes).
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant")

from moespresso.inventory.build import build_inventory  # noqa: E402
from moespresso.inventory.architecture_profile import qwen3_5_moe_profile  # noqa: E402
from moespresso.core.artifact import make_artifact  # noqa: E402
from moespresso.package.bundle import (  # noqa: E402
    METADATA_KEY,
    component_array,
    decode_bundle_metadata,
)
from moespresso.package.kquant_backend import KQuantEncodedWeight  # noqa: E402
from moespresso.package.kquant_format import KQUANT_GEOMETRY  # noqa: E402
from moespresso.package.qwen.recipe import (  # noqa: E402
    build_expert_kquant_allocations as build_qwen_expert_kquant_allocations,
    build_expert_kquant_targets as build_qwen_expert_kquant_targets,
)
from moespresso.optimize.decide import decide  # noqa: E402
from moespresso.package.plan import package_plan_from_decision  # noqa: E402
from moespresso.package.write import write_package  # noqa: E402
from moespresso.correctness.reconstruct import l1_tensor_reconstruction  # noqa: E402
from moespresso.probe.build import build_probe_evidence  # noqa: E402


def _plan(decision: dict) -> dict:
    plan, _summary = package_plan_from_decision(decision)
    return plan


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr)
        b = a.tobytes()
        header[name] = {"dtype": _dtype_tag(a), "shape": list(a.shape),
                        "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _dtype_tag(arr):
    if arr.dtype == np.float32:
        return "F32"
    if arr.dtype == np.float16:
        return "F16"
    if arr.dtype == np.int64:
        return "I64"
    raise AssertionError(f"test helper does not support dtype {arr.dtype}")


ARCH = {"model_type": "qwen3_moe",
        "text_config": {"num_hidden_layers": 1, "hidden_size": 128, "num_experts": 8,
                        "num_experts_per_tok": 2, "moe_intermediate_size": 128,
                        "layer_types": ["full_attention"], "vocab_size": 256}}


def _tiny_model(tmp_path):
    # All three projections: the bundle is one tensor per layer, so a layer
    # is only writable when gate+up+down are all present (like every real MoE).
    rng = np.random.default_rng(0)
    _write_safetensors(tmp_path / "model-00001.safetensors", {
        "model.language_model.layers.0.self_attn.q_proj.weight":
            rng.standard_normal((128, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.gate_up_proj":
            rng.standard_normal((8, 256, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.down_proj":
            rng.standard_normal((8, 128, 128)).astype(np.float32),
    })
    (tmp_path / "config.json").write_text(json.dumps(ARCH))


def test_write_package_end_to_end(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _tiny_model(src)
    out = tmp_path / "out"

    inv = build_inventory(src, layer_types=["full_attention"])
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    dec = decide(ev, target_quality=0.5)
    man = write_package(_plan(dec), src, ARCH, out)

    assert man["artifact_kind"] == "package_manifest"
    assert man["status"] == "valid"
    assert man["provenance"]["source_decision_id"] == dec["artifact_id"]

    # the written shard exists and its sha256 matches the manifest's recorded one
    shard = out / "model-00001-of-00001.safetensors"
    assert shard.exists()
    import hashlib
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    assert man["files"][0]["sha256"] == digest

    # every key the manifest declares for a tensor is actually present on disk
    with open(shard, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        on_disk = set(json.loads(f.read(hlen))) - {"__metadata__"}

    for t in man["tensors"]:
        if t["format"] == "tq":
            # all three projections of a layer share one bundle tensor
            assert f"{t['key_prefix']}.tq_bundle" in on_disk
        elif t["format"] == "affine":
            for suf in ("weight", "scales", "biases"):
                assert f"{t['key_prefix']}.{suf}" in on_disk
        else:  # fp16
            assert t["key_prefix"] in on_disk

    l1 = l1_tensor_reconstruction(
        qwen3_5_moe_profile(), inv, man, src, out,
        sample_policy={"affine_tensors": 1, "rows_per_tensor": 16,
                       "tq_tensors": 2, "tq_experts": 2, "rows_per_expert": 16},
    )
    assert l1["status"] == "valid"
    assert l1["summary"]["sampled_by_format"]["affine"] == 1
    assert l1["summary"]["sampled_by_format"]["tq"] == 2


def test_passthrough_structural_tensors_round_trip(tmp_path):
    # Norms + SSM conv1d are carried verbatim (fp16). conv1d must stay in source form
    # [out, 1, k] (shape[-1] != 1). mlx_lm's qwen3_5 sanitize gates two things on that
    # exact condition (qwen3_5.py:309-330): it transposes conv1d to [out,k,1] and adds
    # +1.0 to the RMSNorm weights. The norms are stored unshifted (source convention,
    # ~0.0), so the +1.0 must fire for them to become ~1.0 at runtime. Storing conv1d
    # pre-transposed [out,k,1] suppresses the shift, so norms load ~1.0 too low and the
    # model emits garbage. The bundle stores conv1d [out,1,k] for exactly this reason.
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    norm = rng.standard_normal((128,)).astype(np.float32)
    conv_raw = rng.standard_normal((64, 1, 4)).astype(np.float32)  # source [out,1,k]
    _write_safetensors(src / "model-00001.safetensors", {
        "model.language_model.layers.0.self_attn.q_proj.weight":
            rng.standard_normal((128, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.gate_up_proj":
            rng.standard_normal((8, 256, 128)).astype(np.float32),
        "model.language_model.layers.0.input_layernorm.weight": norm,
        "model.language_model.layers.0.linear_attn.conv1d.weight": conv_raw,
    })
    (src / "config.json").write_text(json.dumps(ARCH))
    out = tmp_path / "out"

    inv = build_inventory(src, layer_types=["full_attention"])
    passthrough = [e for e in inv["tensors"] if e["kind"] == "passthrough"]
    assert {e["source_name"].rsplit(".", 1)[-1] for e in passthrough} >= {"weight"}
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    dec = decide(ev, target_quality=0.5)
    man = write_package(_plan(dec), src, ARCH, out, passthrough=passthrough)

    by_name = {t["source_name"]: t for t in man["tensors"]}
    nm_norm = "model.language_model.layers.0.input_layernorm.weight"
    nm_conv = "model.language_model.layers.0.linear_attn.conv1d.weight"
    assert by_name[nm_norm]["kind"] == "passthrough" and by_name[nm_norm]["format"] == "fp16"
    assert nm_conv in by_name

    from safetensors.numpy import load_file
    arrays = load_file(str(out / man["files"][0]["path"]))
    # norm copied verbatim; the runtime sanitizer adds +1.0 during load
    np.testing.assert_allclose(arrays[nm_norm].astype(np.float32), norm, atol=1e-2)
    # conv1d stored verbatim in source form [out, 1, k] (shape[-1] == k != 1) so
    # mlx_lm sanitize fires the transpose + the coupled norm +1.0 shift.
    assert arrays[nm_conv].shape == (64, 1, 4)
    assert arrays[nm_conv].shape[-1] != 1
    np.testing.assert_allclose(arrays[nm_conv].astype(np.float32), conv_raw, atol=1e-2)


def test_raw_dtype_passthrough_tensors_round_trip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    raw_f32 = np.array([[1.25, -2.5], [3.0, 4.5]], dtype=np.float32)
    raw_i64 = np.array([[0, 1, 2, 3, 4, 5], [250, 251, 252, 253, 254, 255]], dtype=np.int64)
    _write_safetensors(src / "model-00001.safetensors", {
        "model.language_model.layers.0.self_attn.q_proj.weight":
            rng.standard_normal((128, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.gate_up_proj":
            rng.standard_normal((8, 256, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.down_proj":
            rng.standard_normal((8, 128, 128)).astype(np.float32),
        "layers.2.attn.compressor.ape": raw_f32,
        "layers.0.ffn.gate.tid2eid": raw_i64,
    })
    (src / "config.json").write_text(json.dumps(ARCH))
    out = tmp_path / "out"

    inv = build_inventory(src, layer_types=["full_attention"])
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    dec = decide(ev, target_quality=0.5)
    passthrough = [
        {"source_name": "layers.2.attn.compressor.ape", "role": "attn.compressor.ape",
         "kind": "passthrough", "layer_index": 2, "format": "raw_dtype_passthrough"},
        {"source_name": "layers.0.ffn.gate.tid2eid", "role": "moe.router_tid2eid",
         "kind": "passthrough", "layer_index": 0, "format": "raw_dtype_passthrough"},
    ]
    man = write_package(_plan(dec), src, ARCH, out, passthrough=passthrough)

    by_name = {t["source_name"]: t for t in man["tensors"]}
    assert by_name["layers.2.attn.compressor.ape"]["format"] == "raw_dtype_passthrough"
    assert by_name["layers.0.ffn.gate.tid2eid"]["format"] == "raw_dtype_passthrough"
    assert "raw_dtype_passthrough" in man["required_ops"]

    from safetensors.numpy import load_file
    arrays = load_file(str(out / man["files"][0]["path"]))
    assert arrays["layers.2.attn.compressor.ape"].dtype == np.float32
    assert arrays["layers.0.ffn.gate.tid2eid"].dtype == np.int64
    np.testing.assert_array_equal(arrays["layers.2.attn.compressor.ape"], raw_f32)
    np.testing.assert_array_equal(arrays["layers.0.ffn.gate.tid2eid"], raw_i64)


def test_f32_passthrough_tensors_are_promoted_to_float32(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    f16_source = rng.standard_normal((32, 128)).astype(np.float16)
    _write_safetensors(src / "model-00001.safetensors", {
        "model.language_model.layers.0.self_attn.q_proj.weight":
            rng.standard_normal((128, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.gate_up_proj":
            rng.standard_normal((8, 256, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.down_proj":
            rng.standard_normal((8, 128, 128)).astype(np.float32),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": f16_source,
    })
    (src / "config.json").write_text(json.dumps(ARCH))
    out = tmp_path / "out"

    inv = build_inventory(src, layer_types=["linear_attention"])
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    dec = decide(ev, target_quality=0.5)
    name = "model.language_model.layers.0.linear_attn.in_proj_a.weight"
    passthrough = [{
        "source_name": name,
        "role": "attn.linear_in_proj_a",
        "kind": "passthrough",
        "layer_index": 0,
        "format": "f32_passthrough",
    }]
    man = write_package(_plan(dec), src, ARCH, out, passthrough=passthrough)

    entry = {t["source_name"]: t for t in man["tensors"]}[name]
    assert entry["format"] == "f32_passthrough"
    assert "f32_passthrough" in man["required_ops"]

    from safetensors.numpy import load_file
    arrays = load_file(str(out / man["files"][0]["path"]))
    assert arrays[name].dtype == np.float32
    np.testing.assert_array_equal(arrays[name], f16_source.astype(np.float32))


def test_write_package_writes_qwen_kquant_expert_bundle(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    gate_up = np.arange(2 * 4 * 8, dtype=np.float32).reshape(2, 4, 8)
    down = (1000 + np.arange(2 * 3 * 8, dtype=np.float32)).reshape(2, 3, 8)
    _write_safetensors(src / "model-00001.safetensors", {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.language_model.layers.0.mlp.experts.down_proj": down,
    })
    (src / "config.json").write_text(json.dumps(ARCH))
    inventory = {
        "tensors": [
            {
                "source_name": "model.language_model.layers.0.mlp.experts.gate_up_proj",
                "role": "moe.expert.gate_up",
                "kind": "expert",
                "layer_index": 0,
                "projection": "gate_up",
                "shape": [2, 4, 8],
                "gguf_keys": [
                    "blk.0.ffn_gate_exps.weight",
                    "blk.0.ffn_up_exps.weight",
                ],
                "status": "required",
            },
            {
                "source_name": "model.language_model.layers.0.mlp.experts.down_proj",
                "role": "moe.expert.down",
                "kind": "expert",
                "layer_index": 0,
                "projection": "down",
                "shape": [2, 3, 8],
                "gguf_keys": ["blk.0.ffn_down_exps.weight"],
                "status": "required",
            },
        ],
    }
    targets = build_qwen_expert_kquant_targets({
        "blk.0.ffn_gate_exps.weight": "iq2_xxs",
        "blk.0.ffn_up_exps.weight": "iq2_xxs",
        "blk.0.ffn_down_exps.weight": "iq2_s",
    }, inventory, required_layers=[0])
    allocation = build_qwen_expert_kquant_allocations(targets)
    decision = make_artifact(
        "optimizer_decision",
        {"source_root": "toy", "source_format": "hf_safetensors"},
        {"tool": "test", "version": "0"},
        status="valid",
        allocation=allocation,
        constraints={"objective": "gguf_recipe_kquant_allocation"},
    )
    seen: dict[str, list[np.ndarray]] = {"gate": [], "up": [], "down": []}

    def fake_encoder(weight, target, _imatrix_vectors):
        seen[target.projection].append(np.array(weight, copy=True))
        geometry = KQUANT_GEOMETRY[target.codec]
        value = {"gate": 11, "up": 22, "down": 33}[target.projection]
        return KQuantEncodedWeight(
            codec=target.codec,
            weight=np.full(
                (weight.shape[0], geometry.bytes_per_block),
                value,
                dtype=np.uint8,
            ),
            scales=np.zeros((1,), dtype=np.uint8),
        )

    man = write_package(
        _plan(decision),
        src,
        ARCH,
        tmp_path / "out",
        kquant_encoder=fake_encoder,
        kquant_imatrix_vectors={},
    )

    np.testing.assert_array_equal(seen["gate"][0], gate_up[0, :2])
    np.testing.assert_array_equal(seen["up"][0], gate_up[0, 2:])
    np.testing.assert_array_equal(seen["down"][0], down[0])
    assert man["status"] == "valid"
    assert "kquant_dequant" in man["required_ops"]

    from safetensors import safe_open
    shard = tmp_path / "out" / man["files"][0]["path"]
    bundle_key = "language_model.model.layers.0.mlp.switch_mlp.experts.tq_bundle"
    with safe_open(shard, framework="np") as f:
        bundle = np.asarray(f.get_tensor(bundle_key))
        metadata = f.metadata()
    geometry = decode_bundle_metadata(metadata[METADATA_KEY])[0]
    assert geometry["projections"]["gate_proj"]["kquant_codec"] == "iq2_xxs"
    assert geometry["projections"]["down_proj"]["kquant_codec"] == "iq2_s"
    gate_wire = component_array(bundle, geometry["projections"]["gate_proj"]["weight"])
    up_wire = component_array(bundle, geometry["projections"]["up_proj"]["weight"])
    down_wire = component_array(bundle, geometry["projections"]["down_proj"]["weight"])
    assert np.all(gate_wire == 11)
    assert np.all(up_wire == 22)
    assert np.all(down_wire == 33)


def test_written_package_has_expected_tensor_kinds(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _tiny_model(src)
    inv = build_inventory(src, layer_types=["full_attention"])
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    dec = decide(ev, target_quality=0.5)
    man = write_package(_plan(dec), src, ARCH, tmp_path / "out")

    kinds = {t["format"] for t in man["tensors"]}
    assert "tq" in kinds and "affine" in kinds  # experts TQ, q_proj affine
    # fused gate_up -> gate + up entries; down separate; one bundle key for all
    tq_entries = [t for t in man["tensors"] if t["format"] == "tq"]
    assert {t.get("projection") for t in tq_entries} == {"gate", "up", "down"}
    assert len({t["key_prefix"] for t in tq_entries}) == 1


def test_write_package_writes_dense_mxfp8_without_biases(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(9)
    _write_safetensors(src / "model-00001.safetensors", {
        "layers.0.attn.wq_a.weight":
            rng.standard_normal((32, 64)).astype(np.float32),
    })
    (src / "config.json").write_text(json.dumps(ARCH))
    dec = make_artifact(
        "optimizer_decision",
        {"source_root": str(src), "source_format": "hf_safetensors"},
        {"tool": "test", "version": "0"},
        status="valid",
        allocation=[{
            "source_name": "layers.0.attn.wq_a.weight",
            "kind": "affine",
            "role": "attn.wq_a",
            "bits": 8,
            "group_size": 32,
            "format": "mxfp8",
            "source_codec": "fp8_e4m3_ue8m0",
            "lossless": True,
        }],
        achieved={},
        rejected=[],
        feasibility="feasible",
        objective="test",
    )

    man = write_package(_plan(dec), src, ARCH, tmp_path / "out", chunk_bytes=1024)

    entry = man["tensors"][0]
    assert entry["format"] == "mxfp8"
    assert entry["format_params"]["group_size"] == 32
    from safetensors.numpy import load_file
    arrays = load_file(str(tmp_path / "out" / man["files"][0]["path"]))
    assert "layers.0.attn.wq_a.weight" in arrays
    assert "layers.0.attn.wq_a.scales" in arrays
    assert "layers.0.attn.wq_a.biases" not in arrays


def test_shard_writer_output_is_byte_deterministic(tmp_path):
    """Identical tensors + metadata must produce byte-identical shard files.

    The library serializer keeps __metadata__ in per-instance hash order, so
    shard hashes were not reproducible across builds. The deterministic
    writer pins both properties this test asserts: A/A byte identity, and
    layout equality with the library serializer (same parsed header, same
    data segment) so existing packages keep their data layout.
    """
    from safetensors.numpy import load_file, save_file

    from moespresso.package.bundle import encode_bundle_metadata
    from moespresso.package.write import _ShardWriter

    rng = np.random.default_rng(7)
    tensors = {
        "m.layers.0.bundle": rng.integers(0, 255, (4, 33)).astype(np.uint8),
        "a.norm": rng.standard_normal(16).astype(np.float32),
        "z.norm": rng.standard_normal(16).astype(np.float32),
        "m.scales": rng.standard_normal(8).astype(np.float16),
        "m.packed": rng.integers(0, 2**31, (4, 4)).astype(np.uint32),
        "m.ids": np.arange(6, dtype=np.int64),
        "m.mask": np.array([True, False, True]),
    }
    geometry = {0: {"num_experts": 4, "row_bytes": 33, "projections": {}}}

    def write_with_shard_writer(out_dir):
        out_dir.mkdir()
        writer = _ShardWriter(out_dir, cap_bytes=0)
        writer.add_group(dict(tensors), bundle_geo=(0, geometry[0]))
        rename = writer.finalize()
        (name,) = rename.values()
        return out_dir / name

    first = write_with_shard_writer(tmp_path / "a")
    second = write_with_shard_writer(tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()

    loaded = load_file(str(first))
    assert set(loaded) == set(tensors)
    for key, arr in tensors.items():
        assert np.array_equal(loaded[key], arr)

    library = tmp_path / "library.safetensors"
    save_file(tensors, str(library), metadata={
        "format": "mjtq",
        METADATA_KEY: encode_bundle_metadata(geometry),
    })

    def parse(path):
        raw = path.read_bytes()
        (n,) = struct.unpack("<Q", raw[:8])
        return json.loads(raw[8:8 + n]), raw[8 + n:]

    ours_header, ours_data = parse(first)
    lib_header, lib_data = parse(library)
    assert ours_header == lib_header
    assert ours_data == lib_data
