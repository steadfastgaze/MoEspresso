"""Opt-in real-source smoke for local DeepSeek-V4-Flash artifacts.

Default pytest stays model-free. Enable explicitly:

    MOESPRESSO_RUN_DEEPSEEK_V4_REAL_SMOKE=1 uv run --locked python -m pytest \
        tests/test_deepseek_v4_real_smoke.py -s

Required:
    MOESPRESSO_DEEPSEEK_V4_SOURCE=<HF safetensors checkpoint directory>

Optional:
    MOESPRESSO_DEEPSEEK_V4_IMATRIX=<imatrix calibration file>
    MOESPRESSO_DEEPSEEK_V4_OUT=<artifact directory>
    MOESPRESSO_DEEPSEEK_V4_SAMPLE_ROWS=8
    MOESPRESSO_DEEPSEEK_V4_EXPERT_SAMPLE=1
    MOESPRESSO_DEEPSEEK_V4_TARGET_GB=120.0
    MOESPRESSO_DEEPSEEK_V4_MAX_RSS_GB=4.0

Package smoke:
    MOESPRESSO_RUN_DEEPSEEK_V4_PACKAGE_SMOKE=1 uv run --locked python -m pytest \
        tests/test_deepseek_v4_real_smoke.py -s

Optional package knobs:
    MOESPRESSO_DEEPSEEK_V4_PACKAGE_OUT=<package directory>
    MOESPRESSO_DEEPSEEK_V4_PACKAGE_MAX_RSS_GB=6.0
    MOESPRESSO_DEEPSEEK_V4_MAX_EXPERTS=1
    MOESPRESSO_DEEPSEEK_V4_CHUNK_BYTES=16777216

FP4 nibble oracle:
    MOESPRESSO_RUN_DEEPSEEK_V4_FP4_ORACLE=1 uv run --locked python -m pytest \
        tests/test_deepseek_v4_real_smoke.py -s

The probe smoke does not package or execute the model. The package smoke writes a
reduced real package (`max_experts=1` by default) and verifies it, but still does
not execute the model.
"""

from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path

import numpy as np
import pytest


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _require_source() -> Path:
    source = _env_path("MOESPRESSO_DEEPSEEK_V4_SOURCE")
    if source is None:
        pytest.skip("set MOESPRESSO_DEEPSEEK_V4_SOURCE=<checkpoint directory>")
    if not source.exists():
        pytest.skip(f"source model not found: {source}")
    return source


def _optional_imatrix() -> Path | None:
    imatrix = _env_path("MOESPRESSO_DEEPSEEK_V4_IMATRIX")
    if imatrix is not None and not imatrix.exists():
        pytest.skip(f"imatrix not found: {imatrix}")
    return imatrix


def _released_converter_uses_low_then_high_fp4(convert_text: str) -> bool:
    return bool(
        re.search(r"\blow\s*=\s*x\s*&\s*0x0F", convert_text)
        and re.search(r"\bhigh\s*=\s*\(x\s*>>\s*4\)\s*&\s*0x0F", convert_text)
        and re.search(
            r"torch\.stack\(\s*\[\s*FP4_TABLE\[low\.long\(\)\]\s*,\s*"
            r"FP4_TABLE\[high\.long\(\)\]",
            convert_text,
            re.DOTALL,
        )
    )


def _first_fp4_expert_pair(source: Path) -> tuple[str, str, str]:
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip("source index not found")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    for name in sorted(weight_map):
        if not (
            name.startswith("layers.")
            and ".ffn.experts." in name
            and name.endswith(".w1.weight")
        ):
            continue
        scale_name = name.removesuffix(".weight") + ".scale"
        shard = weight_map[name]
        if weight_map.get(scale_name) == shard:
            return name, scale_name, shard
    pytest.skip("no routed FP4 expert tensor pair found")


@pytest.mark.real_model
def test_deepseek_v4_real_fp4_nibble_order_matches_released_converter():
    if os.environ.get("MOESPRESSO_RUN_DEEPSEEK_V4_FP4_ORACLE") != "1":
        pytest.skip(
            "set MOESPRESSO_RUN_DEEPSEEK_V4_FP4_ORACLE=1 to run DS4 FP4 oracle"
        )

    source = _require_source()
    convert_py = source / "inference" / "convert.py"
    if not convert_py.exists():
        pytest.skip("released converter not found under source inference directory")
    assert _released_converter_uses_low_then_high_fp4(
        convert_py.read_text(encoding="utf-8")
    )

    from moespresso.inventory.safetensors_header import read_headers_with_offsets
    from moespresso.probe.deepseek_v4.codec import (
        FP4_E2M1_TABLE,
        dequant_fp4_e2m1_ue8m0,
        load_storage_rows,
        ue8m0_to_float32,
    )

    weight_name, scale_name, shard = _first_fp4_expert_pair(source)
    headers = {h.name: h for h in read_headers_with_offsets(source / shard)}
    weight_header = headers[weight_name]
    scale_header = headers[scale_name]
    assert weight_header.dtype == "I8"
    assert scale_header.dtype == "F8_E8M0"

    rows = np.arange(min(4, weight_header.shape[0]), dtype=np.int64)
    packed = load_storage_rows(source, weight_header, rows)
    scales = load_storage_rows(source, scale_header, rows)
    packed_u8 = packed.view(np.uint8)
    assert np.any((packed_u8 & 0x0F) != ((packed_u8 >> 4) & 0x0F))

    decoded = dequant_fp4_e2m1_ue8m0(
        packed,
        scales,
        fp4_block=32,
        out_dtype=np.float32,
    )
    scale_expanded = np.repeat(ue8m0_to_float32(scales), 32, axis=1)
    high_first = np.stack(
        [
            FP4_E2M1_TABLE[(packed_u8 >> 4) & 0x0F],
            FP4_E2M1_TABLE[packed_u8 & 0x0F],
        ],
        axis=-1,
    ).reshape(decoded.shape)
    high_first = (high_first * scale_expanded).astype(np.float32)

    first_scale = ue8m0_to_float32(scales[:1, :1])[0, 0]
    first_byte = int(packed_u8[0, 0])
    expected_first_pair = np.array(
        [
            FP4_E2M1_TABLE[first_byte & 0x0F] * first_scale,
            FP4_E2M1_TABLE[(first_byte >> 4) & 0x0F] * first_scale,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(decoded[0, :2], expected_first_pair)
    assert float(np.max(np.abs(decoded - high_first))) > 0.0


@pytest.mark.real_model
def test_deepseek_v4_real_source_probe_optimizer_smoke(tmp_path):
    if os.environ.get("MOESPRESSO_RUN_DEEPSEEK_V4_REAL_SMOKE") != "1":
        pytest.skip("set MOESPRESSO_RUN_DEEPSEEK_V4_REAL_SMOKE=1 to run DS4 real smoke")

    pytest.importorskip("mlx.core")
    pytest.importorskip("jang_tools.turboquant")

    source = _require_source()

    out = Path(
        os.environ.get("MOESPRESSO_DEEPSEEK_V4_OUT", str(tmp_path / "deepseek-v4-real-smoke"))
    ).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    sample_rows = int(os.environ.get("MOESPRESSO_DEEPSEEK_V4_SAMPLE_ROWS", "8"))
    expert_sample = int(os.environ.get("MOESPRESSO_DEEPSEEK_V4_EXPERT_SAMPLE", "1"))
    target_size_gb = float(os.environ.get("MOESPRESSO_DEEPSEEK_V4_TARGET_GB", "120.0"))
    max_rss_gb = float(os.environ.get("MOESPRESSO_DEEPSEEK_V4_MAX_RSS_GB", "4.0"))
    seed = int(os.environ.get("MOESPRESSO_DEEPSEEK_V4_SEED", "11"))

    from moespresso.core.artifact import write_artifact
    from moespresso.inventory.architecture_profile import family_of
    from moespresso.inventory.build import build_inventory
    from moespresso.optimize.affine_elasticity import affine_role_profile_for_family
    from moespresso.optimize.decide import decide
    from moespresso.probe.build import build_probe_evidence
    from moespresso.package.convert import (
        DECISION_NAME,
        INVENTORY_NAME,
        PROBE_NAME,
        _rss_watch,
        rss_summary,
    )

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    family = family_of(config)
    assert family == "deepseek_v4_flash"
    role_profile = affine_role_profile_for_family(family)

    imatrix = _optional_imatrix()
    calibration = None
    if imatrix is not None:
        from moespresso.probe.calibration import imatrix_calibration

        calibration = imatrix_calibration(imatrix)
    imatrix_keys = set(calibration[0]) if calibration is not None else None

    with _rss_watch(interval=1.0) as rss_samples:
        inventory = build_inventory(
            source,
            layer_types=config.get("text_config", config).get("layer_types"),
            imatrix_keys=imatrix_keys,
            family=family,
        )
        write_artifact(out / INVENTORY_NAME, inventory)

        evidence = build_probe_evidence(
            inventory,
            source,
            calibration,
            expert_sample=expert_sample,
            sample_rows=sample_rows,
            seed=seed,
        )
        write_artifact(out / PROBE_NAME, evidence)

        decision = decide(
            evidence,
            target_size_gb=target_size_gb,
            affine_role_profile_name=role_profile["name"] if role_profile else None,
            affine_role_weights=role_profile["affine_role_weights"] if role_profile else None,
            affine_role_bit_weights=(
                role_profile["affine_role_bit_weights"] if role_profile else None
            ),
            affine_role_min_bits=role_profile["affine_role_min_bits"] if role_profile else None,
        )
        write_artifact(out / DECISION_NAME, decision)

    mem = rss_summary(rss_samples)
    smoke = {
        "kind": "deepseek_v4_real_probe_optimizer_smoke",
        "family": family,
        "source": str(source),
        "imatrix": str(imatrix) if imatrix else None,
        "out": str(out),
        "inventory_id": inventory["artifact_id"],
        "probe_id": evidence["artifact_id"],
        "decision_id": decision["artifact_id"],
        "inventory_counts": inventory["counts"],
        "probe_units": len(evidence["units"]),
        "probe_coverage": evidence["coverage"],
        "calibration_kind": evidence.get("calibration", {}).get("kind"),
        "target_size_gb": target_size_gb,
        "achieved": decision.get("achieved"),
        "memory_rss": mem,
        "sample_rows": sample_rows,
        "expert_sample": expert_sample,
        "platform": platform.platform(),
    }
    (out / "deepseek_v4_real_smoke_evidence.json").write_text(json.dumps(smoke, indent=2))

    assert inventory["status"] == "valid"
    assert inventory["counts"]["unknown"] == 0
    assert inventory["counts"]["expert_source"] > 0
    assert inventory["counts"]["affine"] > 0
    assert inventory["counts"]["passthrough"] > 0

    assert evidence["status"] == "valid"
    assert any(u["kind"] == "expert" for u in evidence["units"])
    assert any(u["kind"] == "affine" for u in evidence["units"])
    if calibration is None:
        assert evidence["required_features"] == []
        assert evidence["calibration"]["kind"] == "uniform"
    else:
        assert evidence["required_features"] == ["calibration"]

    assert decision["status"] == "valid"
    assert decision["feasibility"] == "feasible"
    assert decision["achieved"]["size_gb"] > 0
    assert decision["achieved"]["expert_size_gb"] > 0
    assert decision["achieved"]["tensor_size_gb"] > 0

    assert mem is not None
    assert mem["peak_gb"] <= max_rss_gb


@pytest.mark.real_model
def test_deepseek_v4_real_partial_package_smoke(tmp_path):
    if os.environ.get("MOESPRESSO_RUN_DEEPSEEK_V4_PACKAGE_SMOKE") != "1":
        pytest.skip(
            "set MOESPRESSO_RUN_DEEPSEEK_V4_PACKAGE_SMOKE=1 to run DS4 package smoke"
        )

    pytest.importorskip("mlx.core")
    pytest.importorskip("jang_tools.turboquant")

    source = _require_source()
    imatrix = _optional_imatrix()
    out = Path(
        os.environ.get(
            "MOESPRESSO_DEEPSEEK_V4_PACKAGE_OUT",
            str(tmp_path / "deepseek-v4-package-smoke"),
        )
    ).expanduser()
    target_size_gb = float(os.environ.get("MOESPRESSO_DEEPSEEK_V4_TARGET_GB", "120.0"))
    sample_rows = int(os.environ.get("MOESPRESSO_DEEPSEEK_V4_SAMPLE_ROWS", "8"))
    expert_sample = int(os.environ.get("MOESPRESSO_DEEPSEEK_V4_EXPERT_SAMPLE", "1"))
    max_experts = int(os.environ.get("MOESPRESSO_DEEPSEEK_V4_MAX_EXPERTS", "1"))
    chunk_bytes = int(os.environ.get("MOESPRESSO_DEEPSEEK_V4_CHUNK_BYTES", str(16 << 20)))
    shard_size_gb = float(os.environ.get("MOESPRESSO_DEEPSEEK_V4_SHARD_GB", "2.0"))
    max_rss_gb = float(os.environ.get("MOESPRESSO_DEEPSEEK_V4_PACKAGE_MAX_RSS_GB", "6.0"))

    from moespresso.package.constants import MANIFEST_NAME
    from moespresso.package.convert import REPORT_NAME, convert
    from moespresso.runtime.verify import verify_package

    manifest = convert(
        source,
        out,
        imatrix_path=imatrix,
        allow_uniform=imatrix is None,
        target_size_gb=target_size_gb,
        expert_sample=expert_sample,
        sample_rows=sample_rows,
        max_experts=max_experts,
        shard_size_gb=shard_size_gb,
        chunk_bytes=chunk_bytes,
        verbose=True,
    )
    blocking = [v for v in verify_package(manifest, out) if v.blocking]
    report = json.loads((out / REPORT_NAME).read_text(encoding="utf-8"))
    expected_features = ["calibration"] if imatrix is not None else []

    smoke = {
        "kind": "deepseek_v4_real_partial_package_smoke",
        "family": manifest.get("architecture", {}).get("family"),
        "source": str(source),
        "imatrix": str(imatrix) if imatrix else None,
        "out": str(out),
        "inventory_id": report.get("inventory_id"),
        "probe_id": report.get("probe_id"),
        "decision_id": report.get("decision_id"),
        "manifest_id": manifest.get("artifact_id"),
        "files": len(manifest.get("files", [])),
        "tensors": len(manifest.get("tensors", [])),
        "package_bytes": sum(int(f["size_bytes"]) for f in manifest.get("files", [])),
        "verification_blocking": len(blocking),
        "memory_rss": report.get("memory_rss"),
        "package_size_contract": report.get("package_size_contract"),
        "manifest_required_features": manifest.get("required_features", []),
        "report_required_features": report.get("required_features"),
        "target_size_gb": target_size_gb,
        "sample_rows": sample_rows,
        "expert_sample": expert_sample,
        "max_experts": max_experts,
        "platform": platform.platform(),
    }
    (out / "deepseek_v4_package_smoke_evidence.json").write_text(json.dumps(smoke, indent=2))

    assert (out / MANIFEST_NAME).exists()
    assert manifest["status"] == "valid"
    assert manifest["architecture"]["family"] == "deepseek_v4_flash"
    assert len(manifest["files"]) > 0
    assert len(manifest["tensors"]) > 0
    assert not blocking
    assert report["manifest_id"] == manifest["artifact_id"]
    assert report["inventory_id"]
    assert report["probe_id"]
    assert report["decision_id"]
    assert manifest["required_features"] == expected_features
    assert report["required_features"] == {
        "probe": expected_features,
        "decision": expected_features,
        "package_plan": expected_features,
        "manifest": expected_features,
    }
    assert report["package_size_contract"]["package_le_source"] is True
    assert report["memory_rss"]["peak_gb"] <= max_rss_gb
