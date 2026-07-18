"""End-to-end convert orchestrator: the full streamed pipeline on a tiny model.

Runs convert() (inventory->probe->optimize->package) and checks it writes the
package + all four artifacts, the manifest verifies + loads, and the package it
produces is exactly what moespresso-serve's gate accepts. Requires the runtime stack.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant")

from moespresso.package.convert import (  # noqa: E402
    DECISION_NAME,
    INVENTORY_NAME,
    PACKAGE_PLAN_NAME,
    PROBE_NAME,
    REPORT_NAME,
    convert,
)
from moespresso.package.constants import MANIFEST_NAME  # noqa: E402
from moespresso.optimize.affine_elasticity import (  # noqa: E402
    QWEN35_MOE_AFFINE_ROLE_PROFILE_V1_NAME,
)
from moespresso.inventory.architecture_profile import (  # noqa: E402
    DEEPSEEK_V4_FLASH_COMPRESS_RATIOS,
)
from moespresso.runtime.verify import verify_package  # noqa: E402

ARCH = {"model_type": "qwen3_moe",
        "text_config": {"num_hidden_layers": 1, "hidden_size": 128, "num_experts": 8,
                        "num_experts_per_tok": 2, "moe_intermediate_size": 128,
                        "layer_types": ["full_attention"], "vocab_size": 256}}
DS4_ARCH = {
    "model_type": "deepseek_v4",
    "hidden_size": 128,
    "num_hidden_layers": 1,
    "n_routed_experts": 2,
    "num_experts_per_tok": 1,
    "num_nextn_predict_layers": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "sliding_window": 128,
    "index_topk": 512,
    "compress_rope_theta": 160000,
    "compress_ratios": list(DEEPSEEK_V4_FLASH_COMPRESS_RATIOS),
    "vocab_size": 256,
}
MANY_AFFINE_ARCH = {
    "model_type": "synthetic_dense",
    "num_hidden_layers": 17,
    "hidden_size": 128,
    "layer_types": ["full_attention"] * 17,
    "vocab_size": 128,
}


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


def _write_typed_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, (dtype, arr) in tensors.items():
        a = np.ascontiguousarray(arr)
        b = a.tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape),
                        "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _write_safetensors_index(path, tensors, total_size=1_000_000_000):
    (path / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": total_size},
            "weight_map": {
                name: "model-00001.safetensors"
                for name in sorted(tensors)
            },
        }),
        encoding="utf-8",
    )


def _write_legacy_imatrix(path, entries):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, ncall, values in entries:
            data = np.asarray(values, np.float32)
            encoded = name.encode()
            f.write(struct.pack("<i", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<i", ncall))
            f.write(struct.pack("<i", data.size))
            f.write(data.tobytes())


def _tiny_model(d):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    _write_safetensors(d / "model-00001.safetensors", {
        "model.language_model.layers.0.self_attn.q_proj.weight":
            rng.standard_normal((128, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.gate.weight":
            rng.standard_normal((8, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight":
            rng.standard_normal((1, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight":
            rng.standard_normal((128, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight":
            rng.standard_normal((128, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight":
            rng.standard_normal((128, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.gate_up_proj":
            rng.standard_normal((8, 256, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.down_proj":
            rng.standard_normal((8, 128, 128)).astype(np.float32),
    })
    (d / "config.json").write_text(json.dumps(ARCH))


def _ds4_unknown_model(d):
    d.mkdir()
    _write_safetensors(d / "model-00001.safetensors", {
        "layers.0.unmapped.weight": np.ones((2, 2), dtype=np.float32),
    })
    (d / "config.json").write_text(json.dumps(DS4_ARCH))


def _ds4_expert_tensors():
    tensors = {}
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
    return tensors


def _ds4_tiny_model(d):
    d.mkdir()
    tensors = {
        "layers.0.attn.wq_a.weight": (
            "F8_E4M3",
            np.full((128, 128), 0x38, dtype=np.uint8),
        ),
        "layers.0.attn.wq_a.scale": (
            "F8_E8M0",
            np.array([[127]], dtype=np.uint8),
        ),
        "layers.0.ffn.gate.weight": (
            "F32",
            np.arange(256, dtype=np.float32).reshape(2, 128),
        ),
        "layers.0.attn.attn_sink": ("F32", np.arange(64, dtype=np.float32)),
    }
    tensors.update(_ds4_expert_tensors())
    _write_typed_safetensors(d / "model-00001.safetensors", tensors)
    _write_safetensors_index(d, tensors)
    (d / "config.json").write_text(json.dumps(DS4_ARCH))


def _many_affine_model(d):
    d.mkdir()
    rng = np.random.default_rng(1)
    tensors = {
        f"model.language_model.layers.{i}.self_attn.q_proj.weight":
            rng.standard_normal((64, 128)).astype(np.float32)
        for i in range(17)
    }
    _write_safetensors(d / "model-00001.safetensors", tensors)
    (d / "config.json").write_text(json.dumps(MANY_AFFINE_ARCH))


def test_convert_writes_package_and_all_artifacts(tmp_path):
    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "pkg"

    manifest = convert(src, out, allow_uniform=True, target_quality=0.5, shard_size_gb=0.0)

    # all conversion/package artifacts are written next to the package
    for name in (INVENTORY_NAME, PROBE_NAME, DECISION_NAME, PACKAGE_PLAN_NAME, MANIFEST_NAME):
        assert (out / name).exists(), f"missing {name}"
    assert manifest["artifact_kind"] == "package_manifest"

    # the produced package verifies against its own manifest, and the shard exists
    assert [v for v in verify_package(manifest, out) if v.blocking] == []
    for f in manifest["files"]:
        assert (out / f["path"]).exists()


def test_convert_report_links_phase_artifacts(tmp_path):
    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "pkg"

    manifest = convert(src, out, allow_uniform=True, target_quality=0.5, shard_size_gb=0.0)

    inventory = json.loads((out / INVENTORY_NAME).read_text())
    probe = json.loads((out / PROBE_NAME).read_text())
    decision = json.loads((out / DECISION_NAME).read_text())
    package_plan = json.loads((out / PACKAGE_PLAN_NAME).read_text())
    report = json.loads((out / REPORT_NAME).read_text())

    assert report["inventory_id"] == inventory["artifact_id"]
    assert report["probe_id"] == probe["artifact_id"]
    assert report["decision_id"] == decision["artifact_id"]
    assert report["package_plan_id"] == package_plan["artifact_id"]
    assert report["manifest_id"] == manifest["artifact_id"]
    assert report["required_features"] == {
        "probe": [],
        "decision": [],
        "package_plan": [],
        "manifest": [],
    }
    assert report["memory_rss"]["samples"] > 0


def test_convert_reads_layer_types_from_config(tmp_path):
    src = tmp_path / "src"
    _tiny_model(src)
    manifest = convert(src, tmp_path / "pkg", allow_uniform=True,
                       target_quality=0.5, shard_size_gb=0.0)
    # full_attention layer 0 -> q_proj resolves as an affine attn tensor in the package
    roles = {t["role"] for t in manifest["tensors"]}
    assert "attn.q_proj" in roles


def test_convert_passes_affine_role_weights_to_decision(tmp_path):
    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "pkg"

    convert(
        src,
        out,
        allow_uniform=True,
        target_quality=0.5,
        shard_size_gb=0.0,
        allow_unhealthy=True,
        affine_role_weights={"attn.q_proj": 2.0},
        affine_role_bit_weights={"attn.q_proj": {4: 3.0}},
        affine_role_min_bits={"attn.q_proj": 4},
    )

    decision = json.loads((out / DECISION_NAME).read_text())
    assert (
        decision["constraints"]["affine_role_profile_name"]
        == QWEN35_MOE_AFFINE_ROLE_PROFILE_V1_NAME
    )
    assert decision["constraints"]["affine_role_weights"] == {"attn.q_proj": 2.0}
    assert decision["constraints"]["affine_role_bit_weights"] == {
        "attn.q_proj": {"4": 3.0}
    }
    assert decision["constraints"]["affine_role_min_bits"] == {"attn.q_proj": 4}
    assert "role-adjusted affine risk" in decision["objective"]


def test_convert_applies_moe_affine_profile_by_default(tmp_path):
    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "pkg"

    manifest = convert(src, out, allow_uniform=True, target_quality=0.5,
                       shard_size_gb=0.0, allow_unhealthy=True)

    decision = json.loads((out / DECISION_NAME).read_text())
    by_name = {a["source_name"]: a for a in decision["allocation"]}

    assert manifest["architecture"]["family"] == "qwen3_moe"
    assert (
        decision["constraints"]["affine_role_profile_name"]
        == QWEN35_MOE_AFFINE_ROLE_PROFILE_V1_NAME
    )
    assert "moe.shared_expert.down_proj" in decision["constraints"]["affine_role_weights"]
    assert "moe.router_gate" not in decision["constraints"]["affine_role_weights"]
    assert (
        by_name["model.language_model.layers.0.mlp.gate.weight"]["kind"]
        == "fp16_passthrough"
    )
    assert (
        by_name["model.language_model.layers.0.mlp.shared_expert_gate.weight"]["kind"]
        == "fp16_passthrough"
    )
    assert (
        by_name["model.language_model.layers.0.mlp.experts.gate_up_proj"]["kind"]
        == "expert"
    )
    assert (
        by_name["model.language_model.layers.0.mlp.shared_expert.down_proj.weight"]["kind"]
        == "affine"
    )
    assert (
        by_name["model.language_model.layers.0.mlp.shared_expert.down_proj.weight"]["bits"]
        >= 4
    )


def test_convert_multishard_via_tiny_cap(tmp_path):
    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "pkg"
    manifest = convert(src, out, allow_uniform=True,
                       target_quality=0.5, shard_size_gb=1e-6)
    assert len(manifest["files"]) >= 2  # tiny cap forced a split
    assert [v for v in verify_package(manifest, out) if v.blocking] == []


def test_convert_served_package_loads_and_verifies(tmp_path):
    """The package convert() produces loads cleanly, and verify_package flags a tamper."""
    from moespresso.runtime.serve import load_served_model
    from moespresso.runtime.verify import verify_package

    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "pkg"
    convert(src, out, allow_uniform=True, target_quality=0.5, shard_size_gb=0.0)

    called = []
    # fake backend (real build needs a real model); backend gets (manifest, dir).
    model, tok, man = load_served_model(
        out, build_fn=lambda m, p: called.append(p) or ("M", "T"))
    assert called == [out] and man["artifact_kind"] == "package_manifest"

    # verify (the convert-output gate) catches a tampered shard.
    shard = out / man["files"][0]["path"]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF
    shard.write_bytes(bytes(data))
    assert any(v.blocking for v in verify_package(man, out))


def test_convert_requires_calibration_by_default(tmp_path):
    # mjtq declares it needs calibration: no imatrix and no explicit in-process
    # override -> refuse, before any work. (Fails fast, no partial package.)
    src = tmp_path / "src"
    _tiny_model(src)
    with pytest.raises(ValueError, match="calibration"):
        convert(src, tmp_path / "pkg", target_quality=0.5, shard_size_gb=0.0)


def test_convert_records_inventory_imatrix_key_coverage(tmp_path):
    src = tmp_path / "src"
    _tiny_model(src)
    imatrix = tmp_path / "calibration.dat"
    _write_legacy_imatrix(
        imatrix,
        [("blk.0.attn_q.weight", 1, np.ones(128, dtype=np.float32))],
    )
    out = tmp_path / "pkg"

    manifest = convert(
        src,
        out,
        imatrix_path=imatrix,
        target_quality=0.5,
        shard_size_gb=0.0,
    )

    inventory = json.loads((out / INVENTORY_NAME).read_text())
    report = json.loads((out / REPORT_NAME).read_text())
    coverage = inventory["imatrix_coverage"]
    assert coverage["resolved_keys"] > 0
    assert coverage["present_in_imatrix"] == 1
    assert coverage["absent"] > 0
    assert any(v["code"] == "imatrix.key_absent" for v in inventory["validation"])
    assert manifest["required_features"] == ["calibration"]
    assert report["required_features"] == {
        "probe": ["calibration"],
        "decision": ["calibration"],
        "package_plan": ["calibration"],
        "manifest": ["calibration"],
    }


def test_convert_stops_on_invalid_deepseek_v4_inventory(tmp_path):
    src = tmp_path / "ds4-src"
    _ds4_unknown_model(src)
    out = tmp_path / "pkg"

    with pytest.raises(RuntimeError, match="source inventory failed"):
        convert(src, out, allow_uniform=True, target_quality=0.5, shard_size_gb=0.0)

    inventory = json.loads((out / INVENTORY_NAME).read_text())
    assert inventory["status"] == "invalid"
    assert any(v["code"] == "inventory.unknown_tensors" for v in inventory["validation"])
    assert not (out / PROBE_NAME).exists()


def test_convert_deepseek_v4_synthetic_routes_public_orchestrator_and_size_gate(tmp_path):
    src = tmp_path / "ds4-src"
    _ds4_tiny_model(src)
    out = tmp_path / "pkg"

    manifest = convert(
        src,
        out,
        allow_uniform=True,
        target_quality=0.5,
        shard_size_gb=0.0,
        expert_sample=1,
        sample_rows=2,
        max_experts=1,
        allow_unhealthy=True,
        chunk_bytes=64,
    )

    report = json.loads((out / REPORT_NAME).read_text())
    inventory = json.loads((out / INVENTORY_NAME).read_text())

    assert manifest["architecture"]["family"] == "deepseek_v4_flash"
    assert manifest["architecture"]["smoke_max_experts"] == 1
    assert manifest["architecture"]["config"]["n_routed_experts"] == 1
    assert manifest["architecture"]["config"]["num_experts_per_tok"] == 1
    assert inventory["family"] == "deepseek_v4_flash"
    assert any(
        t["format"] == "mxfp4" and t["role"].startswith("moe.expert.")
        for t in manifest["tensors"]
    )
    assert any(t["format"] == "affine" for t in manifest["tensors"])
    assert any(
        t["format"] == "fp16" and t["role"] == "moe.router_gate"
        for t in manifest["tensors"]
    )
    assert report["package_size_contract"]["family"] == "deepseek_v4_flash"
    assert report["package_size_contract"]["package_le_source"] is True
    assert report["package_size_contract"]["package_size_bytes"] > 0
    assert (out / "correctness" / "L1_evidence.json").exists()


def test_convert_invalid_inventory_message_respects_write_intermediate_false(tmp_path):
    src = tmp_path / "ds4-src"
    _ds4_unknown_model(src)
    out = tmp_path / "pkg"

    with pytest.raises(RuntimeError, match="write_intermediate=False") as exc:
        convert(
            src,
            out,
            allow_uniform=True,
            target_quality=0.5,
            shard_size_gb=0.0,
            write_intermediate=False,
        )

    assert "Inventory written" not in str(exc.value)
    assert not (out / INVENTORY_NAME).exists()


def test_convert_optimizer_infeasible_message_respects_write_intermediate_false(tmp_path):
    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "pkg"

    with pytest.raises(RuntimeError, match="write_intermediate=False") as exc:
        convert(
            src,
            out,
            allow_uniform=True,
            target_quality=0.5,
            tau=2.0,
            shard_size_gb=0.0,
            write_intermediate=False,
        )

    assert "Decision written" not in str(exc.value)
    assert not (out / DECISION_NAME).exists()
    assert not (out / PACKAGE_PLAN_NAME).exists()


def test_convert_health_failure_message_respects_write_intermediate_false(tmp_path):
    src = tmp_path / "src"
    _many_affine_model(src)
    out = tmp_path / "pkg"

    with pytest.raises(RuntimeError, match="write_intermediate=False") as exc:
        convert(
            src,
            out,
            allow_uniform=True,
            target_quality=0.0,
            shard_size_gb=0.0,
            write_intermediate=False,
        )

    assert "Decision written" not in str(exc.value)
    assert not (out / DECISION_NAME).exists()
    assert not (out / PACKAGE_PLAN_NAME).exists()
    assert not (out / MANIFEST_NAME).exists()


def test_cli_refuses_without_imatrix(tmp_path):
    # The CLI never produces an uncalibrated mjtq package: no --imatrix -> error
    # exit, driven by the format's declared features (not a hardcoded CLI rule).
    from moespresso.package.convert import main
    src = tmp_path / "src"
    _tiny_model(src)
    with pytest.raises(SystemExit):  # argparse parser.error
        main([str(src), str(tmp_path / "pkg"), "--target-quality", "0.5"])


def test_cli_max_experts_per_layer_must_be_positive(tmp_path, capsys):
    from moespresso.package.convert import main
    src = tmp_path / "src"
    _tiny_model(src)
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit):
            main([
                str(src), str(tmp_path / f"pkg-{bad}"),
                "--target-quality", "0.5",
                "--max-experts-per-layer", bad,
            ])
        assert "positive integer" in capsys.readouterr().err


def test_cli_force_tq4_lossless_threads_to_convert(tmp_path, monkeypatch):
    from moespresso.package import convert as convert_mod

    src = tmp_path / "src"
    _tiny_model(src)
    captured = {}

    def fake_convert(model_dir, out_dir, **kwargs):
        captured.update(kwargs)
        return {"files": [], "tensors": [], "artifact_id": "a" * 32}

    monkeypatch.setattr(convert_mod, "convert", fake_convert)

    rc = convert_mod.main([
        str(src),
        str(tmp_path / "pkg"),
        "--target-quality",
        "0.5",
        "--imatrix",
        str(tmp_path / "imatrix.gguf"),
        "--force-tq4-lossless",
    ])

    assert rc == 0
    assert captured["force_tq4_lossless"] is True


def test_cli_force_dense_lossless_mx_threads_to_convert(tmp_path, monkeypatch):
    from moespresso.package import convert as convert_mod

    src = tmp_path / "src"
    _tiny_model(src)
    captured = {}

    def fake_convert(model_dir, out_dir, **kwargs):
        captured.update(kwargs)
        return {"files": [], "tensors": [], "artifact_id": "a" * 32}

    monkeypatch.setattr(convert_mod, "convert", fake_convert)

    rc = convert_mod.main([
        str(src),
        str(tmp_path / "pkg"),
        "--target-quality",
        "0.5",
        "--imatrix",
        str(tmp_path / "imatrix.gguf"),
        "--force-dense-lossless-mx",
    ])

    assert rc == 0
    assert captured["force_dense_lossless_mx"] is True


def test_cli_min_routed_expert_bits_threads_to_convert(tmp_path, monkeypatch):
    from moespresso.package import convert as convert_mod

    src = tmp_path / "src"
    _tiny_model(src)
    captured = {}

    def fake_convert(model_dir, out_dir, **kwargs):
        captured.update(kwargs)
        return {"files": [], "tensors": [], "artifact_id": "a" * 32}

    monkeypatch.setattr(convert_mod, "convert", fake_convert)

    rc = convert_mod.main([
        str(src),
        str(tmp_path / "pkg"),
        "--target-quality",
        "0.5",
        "--imatrix",
        str(tmp_path / "imatrix.gguf"),
        "--min-routed-expert-bits",
        "2",
    ])

    assert rc == 0
    assert captured["min_routed_expert_bits"] == 2


def test_convert_expert_allocation_ratio_writes_budget_split(tmp_path):
    """--expert-allocation-ratio threads through to the optimizer as a
    budget_split = {experts: ratio, affine: 1 - ratio} and is recorded in the
    decision's constraints."""
    src = tmp_path / "src"
    _tiny_model(src)
    out = tmp_path / "pkg"

    convert(
        src, out, allow_uniform=True,
        target_size_gb=0.001,
        expert_allocation_ratio=0.80,
        shard_size_gb=0.0, allow_unhealthy=True)

    decision = json.loads((out / DECISION_NAME).read_text())
    split = decision["constraints"]["budget_split"]
    assert split["experts"] == 0.80
    assert split["affine"] == pytest.approx(0.20)


def test_convert_expert_allocation_ratio_requires_target_size(tmp_path):
    src = tmp_path / "src"
    _tiny_model(src)
    with pytest.raises(ValueError, match="requires target_size_gb"):
        convert(src, tmp_path / "pkg", allow_uniform=True,
                target_quality=0.5, expert_allocation_ratio=0.80,
                shard_size_gb=0.0, allow_unhealthy=True)


def test_convert_expert_allocation_ratio_range_validation(tmp_path):
    """Ratio must be in (0, 1]; out-of-range raises."""
    src = tmp_path / "src"
    _tiny_model(src)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="ratio must be in"):
            convert(src, tmp_path / "pkg", allow_uniform=True,
                    target_size_gb=0.001, expert_allocation_ratio=bad,
                    shard_size_gb=0.0, allow_unhealthy=True)


def test_cli_expert_allocation_ratio_mutex(tmp_path):
    """--expert-allocation-ratio is incompatible with --target-quality and
    --tau, and requires --target-size-gb."""
    from moespresso.package.convert import main
    src = tmp_path / "src"
    _tiny_model(src)
    base = [str(src), str(tmp_path / "pkg"), "--expert-allocation-ratio", "0.8"]
    # missing --target-size-gb -> error
    with pytest.raises(SystemExit):
        main(base + ["--target-quality", "0.5"])
    # --tau incompatible
    with pytest.raises(SystemExit):
        main(base + ["--target-size-gb", "2.0", "--tau", "0.95"])
    # --target-quality incompatible
    with pytest.raises(SystemExit):
        main(base + ["--target-size-gb", "2.0", "--target-quality", "0.99"])
