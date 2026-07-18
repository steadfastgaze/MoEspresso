from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant")

from moespresso.inventory.architecture_profile import DEEPSEEK_V4_FLASH_COMPRESS_RATIOS  # noqa: E402
from moespresso.inventory.build import build_inventory  # noqa: E402
from moespresso.package.bundle import component_array  # noqa: E402
from moespresso.optimize.decide import decide  # noqa: E402
from moespresso.package.kquant_backend import KQuantEncodedWeight  # noqa: E402
import moespresso.package.deepseek_v4.write as ds4_write_mod  # noqa: E402
from moespresso.package.deepseek_v4.recipe import (  # noqa: E402
    build_ds4_expert_kquant_targets,
    build_ds4_kquant_plan,
)
from moespresso.package.kquant_format import KQUANT_GEOMETRY  # noqa: E402
from moespresso.package.plan import package_plan_from_decision  # noqa: E402
import moespresso.package.write as write_mod  # noqa: E402
from moespresso.package.write import write_package  # noqa: E402
from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup  # noqa: E402
from moespresso.probe.deepseek_v4.probe import build_deepseek_v4_probe_evidence  # noqa: E402
from moespresso.runtime.expert_index import build_expert_index  # noqa: E402
from moespresso.runtime.verify import verify_package  # noqa: E402


DS4_ARCH = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "sliding_window": 128,
    "index_topk": 512,
    "compress_rope_theta": 160000,
    "compress_ratios": list(DEEPSEEK_V4_FLASH_COMPRESS_RATIOS),
    "vocab_size": 129280,
}


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


def _ds4_tiny_source(path):
    tensors = {
        "layers.0.attn.wq_a.weight": (
            "F8_E4M3",
            np.full((128, 128), 0x38, dtype=np.uint8),
        ),
        "layers.0.attn.wq_a.scale": ("F8_E8M0", np.array([[127]], dtype=np.uint8)),
        "layers.0.attn.attn_sink": ("F32", np.arange(64, dtype=np.float32)),
    }
    packed = np.arange(16, dtype=np.uint8).reshape(1, 16).view(np.int8)
    packed_down = np.arange(32, dtype=np.uint8).reshape(2, 16).view(np.int8)
    for expert in (0, 1):
        tensors[f"layers.0.ffn.experts.{expert}.w1.weight"] = ("I8", packed)
        tensors[f"layers.0.ffn.experts.{expert}.w3.weight"] = ("I8", packed)
        tensors[f"layers.0.ffn.experts.{expert}.w2.weight"] = ("I8", packed_down)
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
    _write_safetensors(path / "model-00001.safetensors", tensors)
    (path / "config.json").write_text(json.dumps(DS4_ARCH))


def _load_package_arrays(out, manifest):
    from safetensors.numpy import load_file

    arrays = {}
    for file in manifest["files"]:
        arrays.update(load_file(str(out / file["path"])))
    return arrays


def test_write_package_writes_synthetic_deepseek_v4_mixed_formats(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _ds4_tiny_source(src)
    out = tmp_path / "out"

    inv = build_inventory(src, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, src, fp4_block=32)
    dense = np.random.default_rng(1).standard_normal((128, 128)).astype(np.float32)
    ev = build_deepseek_v4_probe_evidence(
        inv["subject"],
        expert_group=group,
        affine_samples=[{
            "source_name": "layers.0.attn.wq_a.weight",
            "role": "attn.wq_a",
            "layer_index": 0,
            "shape": [128, 128],
            "sample": dense,
        }],
        expert_sample=2,
        sample_rows=32,
        seed=3,
        source_inventory_id=inv["artifact_id"],
    )
    dec, _summary = package_plan_from_decision(
        decide(ev, target_quality=0.5, allow_unhealthy=True))
    passthrough = [e for e in inv["tensors"] if e["kind"] == "passthrough"]

    man = write_package(
        dec,
        src,
        DS4_ARCH,
        out,
        passthrough=passthrough,
        deepseek_v4_expert_group=group,
    )

    assert man["status"] == "valid"
    assert man["architecture"]["family"] == "deepseek_v4_flash"
    assert man["architecture"]["prompt_renderer"] == "deepseek_v4_dsv4"
    assert {"mxfp4_dequant", "affine_dequant", "raw_dtype_passthrough"}.issubset(
        set(man["required_ops"])
    )

    arrays = _load_package_arrays(out, man)
    assert "layers.0.ffn.experts.tq_bundle" in arrays
    assert "layers.0.attn.wq_a.weight" in arrays
    assert "layers.0.attn.wq_a.scales" in arrays
    assert "layers.0.attn.wq_a.biases" in arrays
    assert arrays["layers.0.attn.attn_sink"].dtype == np.float32
    np.testing.assert_array_equal(
        arrays["layers.0.attn.attn_sink"],
        np.arange(64, dtype=np.float32),
    )

    mxfp4_entries = [t for t in man["tensors"] if t["format"] == "mxfp4"]
    assert {t["projection"] for t in mxfp4_entries} == {"gate", "up", "down"}
    assert {t["key_prefix"] for t in mxfp4_entries} == {"layers.0.ffn.experts"}


def test_write_package_streams_deepseek_v4_bundle_one_expert_row_at_a_time(
    tmp_path,
    monkeypatch,
):
    src = tmp_path / "src"
    src.mkdir()
    _ds4_tiny_source(src)
    out = tmp_path / "out"

    seen_expert_axes = []
    real_assemble = ds4_write_mod.assemble_layer_bundle

    def assemble_spy(components, bits, codecs=None):
        expert_axes = {arr.shape[0] for arr in components.values()}
        seen_expert_axes.append(expert_axes)
        assert expert_axes == {1}
        return real_assemble(components, bits, codecs=codecs)

    monkeypatch.setattr(ds4_write_mod, "assemble_layer_bundle", assemble_spy)

    inv = build_inventory(src, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, src, fp4_block=32)
    dense = np.random.default_rng(1).standard_normal((128, 128)).astype(np.float32)
    ev = build_deepseek_v4_probe_evidence(
        inv["subject"],
        expert_group=group,
        affine_samples=[{
            "source_name": "layers.0.attn.wq_a.weight",
            "role": "attn.wq_a",
            "layer_index": 0,
            "shape": [128, 128],
            "sample": dense,
        }],
        expert_sample=2,
        sample_rows=32,
        seed=3,
        source_inventory_id=inv["artifact_id"],
    )
    dec, _summary = package_plan_from_decision(
        decide(ev, target_quality=0.5, allow_unhealthy=True))

    man = write_package(
        dec,
        src,
        DS4_ARCH,
        out,
        passthrough=[e for e in inv["tensors"] if e["kind"] == "passthrough"],
        deepseek_v4_expert_group=group,
    )

    assert seen_expert_axes == [{1}, {1}]
    arrays = _load_package_arrays(out, man)
    assert arrays["layers.0.ffn.experts.tq_bundle"].shape[0] == 2


def test_write_package_writes_deepseek_v4_kquant_expert_bundle(
    tmp_path,
):
    src = tmp_path / "src"
    src.mkdir()
    _ds4_tiny_source(src)
    out = tmp_path / "out"

    inv = build_inventory(src, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, src, fp4_block=32)
    targets = build_ds4_expert_kquant_targets({
        "blk.0.ffn_gate_exps.weight": "iq2_xxs",
        "blk.0.ffn_up_exps.weight": "iq2_xxs",
        "blk.0.ffn_down_exps.weight": "q2_k",
    }, required_layers=[0])
    dec = build_ds4_kquant_plan(
        inv["subject"],
        targets,
        extra_allocation=[{
            "source_name": "layers.0.attn.wq_a.weight",
            "kind": "affine",
            "role": "attn.wq_a",
            "layer_index": 0,
            "bits": 8,
            "group_size": 32,
            "format": "kquant",
            "codec": "q8_0",
            "kquant_codec": "q8_0",
            "gguf_tensor": "blk.0.attn_q_a.weight",
            "imatrix_key": "blk.0.attn_q_a.weight",
            "module_path": "model.layers.0.self_attn.wq_a",
            "module_weight_key": "model.layers.0.self_attn.wq_a.weight",
        }],
    )
    imatrix = {
        target.imatrix_key: np.ones(32, dtype=np.float32)
        for target in targets
    }
    calls = []

    def fake_encoder(weight, target, imatrix_vectors):
        calls.append((weight.copy(), target, imatrix_vectors[target.imatrix_key].copy()))
        geometry = KQUANT_GEOMETRY[target.codec]
        blocks = (
            weight.shape[1] + geometry.weights_per_block - 1
        ) // geometry.weights_per_block
        return KQuantEncodedWeight(
            codec=target.codec,
            weight=np.full(
                (weight.shape[0], blocks * geometry.bytes_per_block),
                len(calls),
                dtype=np.uint8,
            ),
            scales=np.zeros((1,), dtype=np.uint8),
        )

    man = write_package(
        dec,
        src,
        DS4_ARCH,
        out,
        passthrough=[e for e in inv["tensors"] if e["kind"] == "passthrough"],
        deepseek_v4_expert_group=group,
        kquant_imatrix_vectors={
            **imatrix,
            "blk.0.attn_q_a.weight": np.ones(128, dtype=np.float32),
        },
        kquant_encoder=fake_encoder,
    )

    assert man["status"] == "valid"
    assert "kquant_dequant" in man["required_ops"]
    assert len(calls) == 7
    assert calls[0][1].source_name == "layers.0.attn.wq_a.weight"
    assert calls[0][1].codec == "q8_0"
    assert calls[0][0].shape == (128, 128)
    assert [call[1].projection for call in calls[1:]] == [
        "gate",
        "up",
        "down",
        "gate",
        "up",
        "down",
    ]
    arrays = _load_package_arrays(out, man)
    assert "layers.0.attn.wq_a.weight" in arrays
    assert "layers.0.attn.wq_a.scales" in arrays
    assert "layers.0.attn.wq_a.biases" not in arrays
    assert "layers.0.ffn.experts.tq_bundle" in arrays
    assert arrays["layers.0.ffn.experts.tq_bundle"].shape[0] == 2
    idx = build_expert_index(out)
    assert idx.codec(layer=0, projection="gate_proj") == "kquant"
    assert idx.geometry(layer=0, projection="gate_proj").kquant_codec == "iq2_xxs"
    assert idx.geometry(layer=0, projection="down_proj").kquant_codec == "q2_k"
    entries = [t for t in man["tensors"] if t.get("format") == "kquant"]
    expert_entries = [t for t in entries if t["kind"] == "expert"]
    dense_entries = [t for t in entries if t["kind"] == "affine"]
    assert {t["projection"] for t in expert_entries} == {"gate", "up", "down"}
    assert all(t["module_weight_key"].endswith("_proj.weight") for t in expert_entries)
    assert dense_entries[0]["module_weight_key"] == (
        "model.layers.0.self_attn.wq_a.weight"
    )


def test_write_package_can_source_deepseek_v4_kquant_expert_bundle_from_loader(
    tmp_path,
):
    src = tmp_path / "src"
    src.mkdir()
    _ds4_tiny_source(src)
    out = tmp_path / "out"

    inv = build_inventory(src, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, src, fp4_block=32)
    targets = build_ds4_expert_kquant_targets({
        "blk.0.ffn_gate_exps.weight": "iq2_xxs",
        "blk.0.ffn_up_exps.weight": "iq2_xxs",
        "blk.0.ffn_down_exps.weight": "q2_k",
    }, required_layers=[0])
    dec = build_ds4_kquant_plan(inv["subject"], targets)
    calls = []

    def fake_loader(target, expert_index):
        calls.append((target.projection, expert_index))
        rows = {"gate": 2, "up": 3, "down": 4}[target.projection]
        bytes_per_row = KQUANT_GEOMETRY[target.codec].bytes_per_block
        value = 10 * int(expert_index) + {"gate": 1, "up": 2, "down": 3}[target.projection]
        return KQuantEncodedWeight(
            codec=target.codec,
            weight=np.full((rows, bytes_per_row), value, dtype=np.uint8),
            scales=np.zeros((1,), dtype=np.uint8),
        )

    man = write_package(
        dec,
        src,
        DS4_ARCH,
        out,
        passthrough=[e for e in inv["tensors"] if e["kind"] == "passthrough"],
        deepseek_v4_expert_group=group,
        kquant_expert_loader=fake_loader,
    )

    assert man["status"] == "valid"
    assert calls == [
        ("gate", 0), ("up", 0), ("down", 0),
        ("gate", 1), ("up", 1), ("down", 1),
    ]
    arrays = _load_package_arrays(out, man)
    bundle = arrays["layers.0.ffn.experts.tq_bundle"]
    idx = build_expert_index(out)
    assert idx.codec(layer=0, projection="gate_proj") == "kquant"
    gate_weight = component_array(
        bundle,
        idx.row_components(layer=0)[("gate_proj", "weight")],
    )
    down_weight = component_array(
        bundle,
        idx.row_components(layer=0)[("down_proj", "weight")],
    )
    np.testing.assert_array_equal(gate_weight[1], np.full((2, 66), 11, dtype=np.uint8))
    np.testing.assert_array_equal(down_weight[0], np.full((4, 84), 3, dtype=np.uint8))


def test_write_package_rejects_kquant_expert_without_imatrix_vectors(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _ds4_tiny_source(src)
    out = tmp_path / "out"

    inv = build_inventory(src, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, src, fp4_block=32)
    targets = build_ds4_expert_kquant_targets({
        "blk.0.ffn_gate_exps.weight": "iq2_xxs",
        "blk.0.ffn_up_exps.weight": "iq2_xxs",
        "blk.0.ffn_down_exps.weight": "q2_k",
    }, required_layers=[0])
    dec = build_ds4_kquant_plan(inv["subject"], targets)

    with pytest.raises(ValueError, match="requires imatrix vectors"):
        write_package(
            dec,
            src,
            DS4_ARCH,
            out,
            passthrough=[e for e in inv["tensors"] if e["kind"] == "passthrough"],
            deepseek_v4_expert_group=group,
            kquant_encoder=lambda *_args: pytest.fail("encoder should not run"),
        )


def test_write_package_streams_deepseek_v4_fp8_affine_output(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    _ds4_tiny_source(src)
    out = tmp_path / "out"

    inv = build_inventory(src, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, src, fp4_block=32)
    dense = np.random.default_rng(1).standard_normal((128, 128)).astype(np.float32)
    ev = build_deepseek_v4_probe_evidence(
        inv["subject"],
        expert_group=group,
        affine_samples=[{
            "source_name": "layers.0.attn.wq_a.weight",
            "role": "attn.wq_a",
            "layer_index": 0,
            "shape": [128, 128],
            "sample": dense,
        }],
        expert_sample=2,
        sample_rows=32,
        seed=3,
        source_inventory_id=inv["artifact_id"],
    )
    dec, _summary = package_plan_from_decision(
        decide(ev, target_quality=0.5, allow_unhealthy=True))
    monkeypatch.setattr(write_mod, "_STREAMED_AFFINE_OUTPUT_THRESHOLD_BYTES", 1)

    def _forbidden(*args, **kwargs):
        raise AssertionError("DS4 FP8 affine output used resident concatenate path")

    monkeypatch.setattr(write_mod, "_quantize_affine_streamed_chunks", _forbidden)

    man = write_package(
        dec,
        src,
        DS4_ARCH,
        out,
        passthrough=[e for e in inv["tensors"] if e["kind"] == "passthrough"],
        deepseek_v4_expert_group=group,
        chunk_bytes=64,
    )

    assert not verify_package(man, out)
    assert not list(out.glob("*.tmp"))
    assert not list(out.glob(".*.tmp"))
    arrays = _load_package_arrays(out, man)
    assert "layers.0.attn.wq_a.weight" in arrays
    assert "layers.0.attn.wq_a.scales" in arrays
    assert "layers.0.attn.wq_a.biases" in arrays
