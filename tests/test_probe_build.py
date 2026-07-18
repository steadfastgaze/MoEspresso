"""probe_evidence builder: end-to-end on a synthetic 2-tensor model.

Builds a real inventory over a hand-written safetensors file, runs the probe
(real mlx affine + jang TQ round-trips), and checks the emitted artifact: one
affine unit + a fused expert split into gate/up, populated quality tables, sane
coverage, and a valid base contract. Skips if mlx is absent.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant")

from moespresso.core.artifact import validate_base  # noqa: E402
from moespresso.inventory.build import build_inventory  # noqa: E402
from moespresso.probe.build import build_probe_evidence  # noqa: E402


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape),
                        "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _tiny_model(tmp_path):
    rng = np.random.default_rng(0)
    # one full-attn q_proj (affine), one fused gate_up expert stack (TQ).
    aff = rng.standard_normal((128, 128)).astype(np.float32)
    experts = rng.standard_normal((8, 256, 128)).astype(np.float32)  # 8 experts, fused h=256
    _write_safetensors(tmp_path / "model-00001.safetensors", {
        "model.language_model.layers.0.self_attn.q_proj.weight": aff,
        "model.language_model.layers.0.mlp.experts.gate_up_proj": experts,
    })
    (tmp_path / "config.json").write_text(json.dumps({"layer_types": ["full_attention"]}))


def test_probe_evidence_end_to_end(tmp_path):
    _tiny_model(tmp_path)
    inv = build_inventory(tmp_path, layer_types=["full_attention"])
    assert inv["counts"]["expert"] == 1 and inv["counts"]["affine"] == 1

    ev = build_probe_evidence(inv, tmp_path, expert_sample=2, sample_rows=64)
    assert validate_base(ev) == []  # base contract holds, no blocking errors
    assert ev["artifact_kind"] == "probe_evidence"
    assert ev["source_inventory_id"] == inv["artifact_id"]
    # No calibration passed, so the explicit uniform escape hatch applies.
    assert ev["calibration"] == {"kind": "uniform"}
    assert ev["required_features"] == []

    by_kind = {}
    for u in ev["units"]:
        by_kind.setdefault(u["kind"], []).append(u)
    # fused gate_up -> two expert units (gate, up); one affine unit.
    assert len(by_kind["affine"]) == 1
    assert {u["projection"] for u in by_kind["expert"]} == {"gate", "up"}

    aff = by_kind["affine"][0]
    assert set(aff["quality"]) >= {"8_64", "4_64"}
    assert all(-1.0 <= v <= 1.0 for v in aff["quality"].values())
    assert aff["quality"]["8_64"] > aff["quality"]["2_64"] - 1e-6  # more bits not worse

    exp = by_kind["expert"][0]
    assert set(exp["quality"]) == {"1", "2", "4"}
    assert exp["n_experts"] == 8 and exp["sampled"] == 2


def test_expert_shape_is_true_geometry_not_sampled_rows(tmp_path):
    # A unit's `shape` describes the tensor (true [out_features, in_features]) so the
    # size estimate is right; `sampled` describes the probe. The fused expert stack is
    # (8 experts, out=256, in=128); split gate/up -> true out_features 128 each. Probe
    # with sample_rows=64 < 128: if `shape` leaks the sampled row count, the optimizer
    # under-counts expert bytes ~2x and the size budget collapses (leaking sampled rows
    # into shape strands the allocation at the bit floor). shape[0] is the true
    # out_features, independent of sampling.
    _tiny_model(tmp_path)
    inv = build_inventory(tmp_path, layer_types=["full_attention"])
    ev = build_probe_evidence(inv, tmp_path, expert_sample=2, sample_rows=64)

    experts = [u for u in ev["units"] if u["kind"] == "expert"]
    assert experts, "expected gate/up expert units"
    for u in experts:
        out_features, in_features = u["shape"]
        assert out_features == 128, (  # true per-projection output geometry
            f"{u['projection']} shape[0]={out_features} is the sampled-row count, "
            f"not the true out_features (128)")
        assert in_features == 128
        assert u["sampled"] == 2  # sampling lives here, separate from shape


def test_expert_size_estimate_is_invariant_to_sampling(tmp_path):
    # Size-truth characterization: the optimizer's expert byte estimate must reflect
    # the true tensor, so it cannot depend on how many rows the probe sampled. Probe
    # the same model at two sample_rows and assert the decision's expert_size_gb is
    # identical. When the estimate reads sampled rows it scales with the sample; with
    # the shape carrying true out_features it is invariant.
    from moespresso.optimize.decide import decide

    _tiny_model(tmp_path)
    inv = build_inventory(tmp_path, layer_types=["full_attention"])

    ev_a = build_probe_evidence(inv, tmp_path, expert_sample=2, sample_rows=64)
    ev_b = build_probe_evidence(inv, tmp_path, expert_sample=2, sample_rows=32)
    size_a = decide(ev_a, target_size_gb=1.0)["achieved"]["expert_size_gb"]
    size_b = decide(ev_b, target_size_gb=1.0)["achieved"]["expert_size_gb"]
    assert size_a == pytest.approx(size_b, rel=1e-9), (
        f"expert_size_gb depends on sample_rows ({size_a} vs {size_b}); the size "
        f"estimate is reading sampled rows instead of true out_features")


def test_probe_uniform_fallback_without_calibration(tmp_path):
    _tiny_model(tmp_path)
    inv = build_inventory(tmp_path, layer_types=["full_attention"])
    ev = build_probe_evidence(inv, tmp_path, expert_sample=2, sample_rows=64)
    # no calibration -> everything falls back to uniform, flagged in validation,
    # stamped uniform, and declares no required feature (cannot pass as calibrated).
    assert all(not u["imatrix_mapped"] for u in ev["units"])
    assert any(v["code"] == "probe.no_imatrix" for v in ev["validation"])
    assert ev["calibration"] == {"kind": "uniform"}
    assert ev["required_features"] == []


def test_probe_with_calibration_records_identity_and_feature(tmp_path):
    # Build a tiny GGUF imatrix whose key matches the q_proj's resolved gguf_key,
    # then confirm the evidence records the calibration identity + requires it.
    from moespresso.probe.calibration import calibration_identity

    _tiny_model(tmp_path)
    inv = build_inventory(tmp_path, layer_types=["full_attention"])
    q = next(t for t in inv["tensors"] if t["role"] == "attn.q_proj")
    gguf_key = q["gguf_keys"][0]
    in_features = q["shape"][1]
    vectors = {gguf_key: np.abs(np.random.default_rng(1).standard_normal(in_features)).astype(np.float32)}
    identity = calibration_identity(__file__, vectors)  # any real file for hash/size

    ev = build_probe_evidence(inv, tmp_path, (vectors, identity),
                              expert_sample=2, sample_rows=64)
    assert ev["calibration"]["sha256"] == identity["sha256"]
    assert ev["calibration"]["key_count"] == 1
    assert ev["required_features"] == ["calibration"]
    # The q_proj unit picked up its mapped imatrix.
    q_unit = next(u for u in ev["units"]
                  if u["source_name"].endswith("self_attn.q_proj.weight"))
    assert q_unit["imatrix_mapped"] is True


def test_probe_warns_when_calibration_maps_only_some_targets(tmp_path):
    from moespresso.probe.calibration import calibration_identity

    _tiny_model(tmp_path)
    inv = build_inventory(tmp_path, layer_types=["full_attention"])
    vectors = {
        "blk.0.ffn_gate_exps.weight": np.ones(128, dtype=np.float32),
    }
    identity = calibration_identity(__file__, vectors)

    ev = build_probe_evidence(
        inv,
        tmp_path,
        (vectors, identity),
        expert_sample=2,
        sample_rows=64,
    )

    assert ev["coverage"]["expert_mapped"] == "1/2"
    assert any(v["code"] == "probe.partial_imatrix" for v in ev["validation"])
