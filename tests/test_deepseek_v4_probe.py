from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant")

from moespresso.core.artifact import validate_base  # noqa: E402
from moespresso.inventory.build import build_inventory  # noqa: E402
from moespresso.probe.build import build_probe_evidence  # noqa: E402
from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup  # noqa: E402
from moespresso.probe.deepseek_v4.probe import (  # noqa: E402
    build_deepseek_v4_probe_evidence,
    iter_deepseek_v4_affine_samples,
    probe_deepseek_v4_affine_sample,
)


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


def _expert_tensors():
    tensors = {}
    packed = np.arange(16, dtype=np.uint8).reshape(1, 16).view(np.int8)
    packed_down = np.arange(32, dtype=np.uint8).reshape(2, 16).view(np.int8)
    for expert in (0, 1):
        tensors[f"layers.0.ffn.experts.{expert}.w1.weight"] = (
            "I8",
            packed,
        )
        tensors[f"layers.0.ffn.experts.{expert}.w3.weight"] = (
            "I8",
            packed,
        )
        tensors[f"layers.0.ffn.experts.{expert}.w2.weight"] = (
            "I8",
            packed_down,
        )
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
    return tensors


def test_deepseek_v4_probe_builds_logical_expert_units_from_decoded_fp4(tmp_path):
    _write_safetensors(tmp_path / "model-00001.safetensors", _expert_tensors())
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, tmp_path, fp4_block=32)
    vectors = {
        "blk.0.ffn_gate_exps.weight": np.ones(32, dtype=np.float32),
        "blk.0.ffn_up_exps.weight": np.ones(32, dtype=np.float32) * 2.0,
        "blk.0.ffn_down_exps.weight": np.ones(32, dtype=np.float32) * 3.0,
    }

    ev = build_deepseek_v4_probe_evidence(
        inv["subject"],
        expert_group=group,
        calibration=(vectors, {"kind": "gguf_imatrix", "sha256": "calib"}),
        expert_sample=2,
        sample_rows=4,
        seed=7,
        source_inventory_id=inv["artifact_id"],
    )

    assert validate_base(ev) == []
    assert ev["required_features"] == ["calibration"]
    assert ev["source_inventory_id"] == inv["artifact_id"]
    assert ev["coverage"]["expert_mapped"] == "3/3"
    assert {u["projection"] for u in ev["units"]} == {"gate", "up", "down"}
    by_projection = {u["projection"]: u for u in ev["units"]}
    assert by_projection["gate"]["source_name"] == "layers.0.ffn.experts.gate"
    assert by_projection["gate"]["shape"] == [1, 32]
    assert by_projection["gate"]["n_experts"] == 2
    assert by_projection["gate"]["sampled"] == 2
    assert set(by_projection["gate"]["quality"]) == {"1", "2", "4"}
    assert by_projection["gate"]["source_codec"] == "fp4_e2m1_ue8m0"
    assert by_projection["gate"]["lossless_codecs"] == ["mxfp4"]
    assert all(-1.0 <= q <= 1.0 for q in by_projection["gate"]["quality"].values())
    assert by_projection["down"]["shape"] == [2, 32]


def test_deepseek_v4_probe_warns_when_calibration_maps_only_some_targets(tmp_path):
    _write_safetensors(tmp_path / "model-00001.safetensors", _expert_tensors())
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, tmp_path, fp4_block=32)
    vectors = {"blk.0.ffn_gate_exps.weight": np.ones(32, dtype=np.float32)}

    ev = build_deepseek_v4_probe_evidence(
        inv["subject"],
        expert_group=group,
        calibration=(vectors, {"kind": "gguf_imatrix", "sha256": "calib"}),
        expert_sample=2,
        sample_rows=4,
        seed=7,
        source_inventory_id=inv["artifact_id"],
    )

    assert ev["coverage"]["expert_mapped"] == "1/3"
    assert any(v["code"] == "probe.partial_imatrix" for v in ev["validation"])


def test_deepseek_v4_probe_builds_affine_units_from_decoded_samples():
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((96, 64)).astype(np.float32)
    unit = probe_deepseek_v4_affine_sample(
        source_name="layers.2.attn.indexer.wq_b.weight",
        role="attn.indexer.wq_b",
        layer_index=2,
        sample=sample,
        shape=(8192, 1024),
        gguf_key=None,
        sample_rows=32,
    )

    assert unit["kind"] == "affine"
    assert unit["role"] == "attn.indexer.wq_b"
    assert unit["shape"] == [8192, 1024]
    assert unit["imatrix_mapped"] is False
    assert set(unit["quality"]) >= {"2_32", "4_32", "8_32"}
    assert set(unit["dense_codec_quality"]) == {"mxfp4_4_32", "mxfp8_8_32"}
    assert all(-1.0 <= q <= 1.0 for q in unit["quality"].values())
    assert all(-1.0 <= q <= 1.0 for q in unit["dense_codec_quality"].values())


def test_deepseek_v4_affine_sampler_decodes_fp8_rows_from_inventory(tmp_path):
    _write_safetensors(tmp_path / "model-00001.safetensors", {
        "layers.0.attn.wq_a.weight": (
            "F8_E4M3",
            np.array(
                [
                    [0x38, 0x40, 0x30, 0x00],
                    [0xB8, 0x30, 0x38, 0x40],
                    [0x40, 0x38, 0xB8, 0x30],
                    [0x30, 0x00, 0x40, 0x38],
                ],
                dtype=np.uint8,
            ),
        ),
        "layers.0.attn.wq_a.scale": (
            "F8_E8M0",
            np.array([[127, 128], [129, 130]], dtype=np.uint8),
        ),
    })
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")

    samples = list(iter_deepseek_v4_affine_samples(
        inv,
        tmp_path,
        sample_rows=2,
        seed=0,
        fp8_block=(2, 2),
    ))

    assert len(samples) == 1
    assert samples[0]["source_name"] == "layers.0.attn.wq_a.weight"
    assert samples[0]["sample"].shape == (2, 4)
    assert samples[0]["sample"].dtype == np.float32
    assert samples[0]["source_codec"] == "fp8_e4m3_ue8m0"
    assert samples[0]["lossless_codecs"] == ["mxfp8"]


def test_build_probe_evidence_dispatches_deepseek_v4_to_real_source_adapter(tmp_path):
    tensors = _expert_tensors()
    tensors.update({
        "layers.0.attn.wq_a.weight": (
            "F8_E4M3",
            np.full((128, 128), 0x38, dtype=np.uint8),
        ),
        "layers.0.attn.wq_a.scale": (
            "F8_E8M0",
            np.array([[127]], dtype=np.uint8),
        ),
    })
    _write_safetensors(tmp_path / "model-00001.safetensors", tensors)
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")

    ev = build_probe_evidence(
        inv,
        tmp_path,
        expert_sample=1,
        sample_rows=2,
        seed=7,
    )

    assert ev["config"]["family"] == "deepseek_v4_flash"
    assert ev["source_inventory_id"] == inv["artifact_id"]
    assert {u["kind"] for u in ev["units"]} == {"expert", "affine"}
    assert any(u["source_name"] == "layers.0.attn.wq_a.weight" for u in ev["units"])
    assert any(u["source_name"] == "layers.0.ffn.experts.gate" for u in ev["units"])


def test_deepseek_v4_affine_sampler_fails_closed_without_fp8_scale(tmp_path):
    _write_safetensors(tmp_path / "model-00001.safetensors", {
        "layers.0.attn.wq_a.weight": (
            "F8_E4M3",
            np.full((128, 128), 0x38, dtype=np.uint8),
        ),
    })
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")

    with pytest.raises(ValueError, match="missing FP8 scale tensor"):
        build_probe_evidence(inv, tmp_path, sample_rows=2)


def test_deepseek_v4_probe_uniform_fallback_cannot_masquerade_as_calibrated(tmp_path):
    _write_safetensors(tmp_path / "model-00001.safetensors", _expert_tensors())
    inv = build_inventory(tmp_path, family="deepseek_v4_flash")
    group = DecodedExpertGroup.from_inventory(inv, tmp_path, fp4_block=32)

    ev = build_deepseek_v4_probe_evidence(
        inv["subject"],
        expert_group=group,
        expert_sample=1,
        sample_rows=4,
        seed=7,
    )

    assert ev["calibration"] == {"kind": "uniform"}
    assert ev["required_features"] == []
    assert ev["coverage"]["expert_mapped"] == "0/3"
    assert any(v["code"] == "probe.no_imatrix" for v in ev["validation"])
