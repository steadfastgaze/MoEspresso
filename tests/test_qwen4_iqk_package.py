"""Qwen4 IQ_K package allocation, byte pricing and bundle geometry."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from moespresso.package.qwen4 import iqk_package as qwen4_iqk_package
from moespresso.core.artifact import artifact_producer, make_artifact, write_artifact
from moespresso.inventory.qwen4 import roles as qwen4_roles
from moespresso.inventory.qwen4.static import expected_qwen38_flash_next_header_specs
from moespresso.package.bundle import (
    BundleFormatError,
    decode_bundle_metadata,
    encode_bundle_metadata,
)
from moespresso.package.bundle import METADATA_KEY
from moespresso.inventory.safetensors_header import read_shard_metadata
from moespresso.package.iqk_artifacts import IQKConvertedArtifacts
from moespresso.package.iqk_recipe import (
    build_iqk_expert_allocations,
    build_iqk_package_plan,
)
from moespresso.package.manifest import build_package_manifest, located_key
from moespresso.package.write import (
    _write_iqk_layer_bundle_streamed,
    validate_additional_file_identities,
    write_package,
)
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
    iqk_geometry,
)
from moespresso.package.iqk_dense_relayout import unpack_dense_rows
from moespresso.package.iqk_relayout import (
    relayout_implementation_identity,
    unpack_stream_major,
)
from moespresso.package.iqk_write import iqk_bundle_row
from moespresso.package.iqk_write import annotate_expert_input_geometry
from moespresso.package.kquant_backend import KQuantEncodedWeight
from moespresso.package.kquant_bundle import assemble_kquant_encoded_layer_bundle
from moespresso.package.qwen4.iqk_package import (
    CALIBRATED_IQK_POLICY,
    CONVERSION_INVENTORY_SCHEMA,
    EARLY_IQ3_ZERO_COUNT_PAIRS,
    FALLBACK_POLICY,
    Qwen4IQKPackageError,
    ZERO_COUNT_EARLY_IQ3_MODE,
    ZERO_COUNT_JOINT_MODE,
    ZERO_COUNT_MEAN_POLICY,
    ZERO_COUNT_SPECIALIZATION_MODE,
    _expert_allocations,
    _storage_guard,
    _storage_guard_with_ple_reuse,
    _conversion_identity,
    _validate_decision_expert_selection,
    build_qwen4_direct_allocations,
    encode_qwen4_dense_iqk,
    price_qwen4_expert_allocation,
    read_qwen4_iqk_allocation,
    zero_count_mean_policy_contract,
)
from moespresso.package.qwen4.iqk_reap import (
    build_expert_selection,
    validate_expert_selection,
)

from conftest import write_safetensors_raw


_TEACHER_SOURCE_IDENTITY = {
    "model_id": "Qwen/Qwen3.8-Flash-Next",
    "revision": "d" * 40,
    "config_sha256": "a" * 64,
    "index_sha256": "b" * 64,
}
_SOURCE_SHARDS = [
    {
        "name": f"model-{index:05d}-of-00131.safetensors",
        "hf_blob_sha256": f"{index:064x}",
        "size_bytes": index,
    }
    for index in range(1, 132)
]
_SHARD_MANIFEST_SHA256 = hashlib.sha256(
    json.dumps(_SOURCE_SHARDS, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
_SOURCE_IDENTITY_WITHOUT_HASH = {
    "schema": "qwen4_hf_snapshot_source_v1",
    "teacher_source_identity": _TEACHER_SOURCE_IDENTITY,
    "shards": _SOURCE_SHARDS,
    "shard_manifest_sha256": _SHARD_MANIFEST_SHA256,
}
_SOURCE_IDENTITY = {
    **_SOURCE_IDENTITY_WITHOUT_HASH,
    "snapshot_identity_sha256": hashlib.sha256(
        json.dumps(
            _SOURCE_IDENTITY_WITHOUT_HASH,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest(),
}
_EXPECTED_ZERO_COUNT_PAIRS = (
    (0, 181),
    (0, 193),
    (0, 236),
    (0, 244),
    (0, 271),
    (0, 413),
    (0, 424),
    (0, 477),
    (1, 116),
)
_JOINT_ZERO_COUNT_PAIRS = (
    (0, 181),
    (0, 193),
    (0, 244),
    (0, 271),
    (0, 413),
    (0, 424),
    (1, 116),
)


def test_cli_preflight_forwards_optimized_kernel_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_preflight(_model_dir: str, **kwargs: object) -> dict:
        captured.update(kwargs)
        return {"status": "preflight"}

    monkeypatch.setattr(qwen4_iqk_package, "preflight_qwen4_iqk_package", fake_preflight)
    rc = qwen4_iqk_package.main(
        [
            str(tmp_path / "model"),
            str(tmp_path / "package"),
            "--allocation",
            str(tmp_path / "allocation.json"),
            "--teacher-capture",
            str(tmp_path / "teacher"),
            "--dense-teacher-capture",
            str(tmp_path / "dense-teacher"),
            "--preflight-only",
            "--optimized-kernels-expected",
            "--direct-codec",
            "q5_k",
            "--expert-output-layout",
            IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
            "--reuse-ple-from",
            str(tmp_path / "donor-package"),
        ]
    )

    assert rc == 0
    assert captured["optimized_kernels_expected"] is True
    assert captured["direct_codec"] == "q5_k"
    assert captured["expert_output_layout"] == IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1
    assert captured["dense_teacher_capture"] == str(tmp_path / "dense-teacher")
    assert captured["reuse_ple_from"] == str(tmp_path / "donor-package")
    assert json.loads(capsys.readouterr().out)["status"] == "preflight"


def test_preflight_builds_the_promoted_package_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts = {"inventory": {}, "source_identity": {}}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        qwen4_iqk_package, "_source_parts", lambda *_args, **_kwargs: parts
    )
    monkeypatch.setattr(
        qwen4_iqk_package,
        "read_qwen4_iqk_allocation",
        lambda *_args, **_kwargs: ({}, {"artifact_id": "decision"}),
    )

    def fake_build_plan(*_args: object, **kwargs: object) -> tuple[dict, dict]:
        captured.update(kwargs)
        return (
            {"artifact_id": "promoted-plan"},
            {
                "ple_contract": object(),
                "expert_price": {"codec_counts": {}},
                "direct_price": {"format_counts": {}},
                "pricing": {},
                "storage_guard": {},
            },
        )

    monkeypatch.setattr(qwen4_iqk_package, "_build_plan", fake_build_plan)
    monkeypatch.setattr(
        qwen4_iqk_package, "inspect_qwen4_ple_source", lambda *_args, **_kwargs: None
    )

    report = qwen4_iqk_package.preflight_qwen4_iqk_package(
        "model",
        allocation_path="allocation.json",
        teacher_capture="teacher",
        optimized_kernels_expected=True,
        direct_codec="q5_k",
    )

    assert captured["optimized_kernels_expected"] is True
    assert captured["direct_codec"] == "q5_k"
    assert report["package_plan_id"] == "promoted-plan"
    assert report["optimized_kernels_expected"] is True
    assert report["expert_output_layout"] == IQK_LAYOUT_IQK_RELAYOUT


def _expert_source(layer: int, projection: str) -> str:
    suffix = "gate_up_proj" if projection in {"gate", "up"} else "down_proj"
    return f"model.language_model.layers.{layer}.mlp.experts.{suffix}"


def _expert_inventory() -> dict:
    tensors = []
    for layer in range(48):
        tensors.extend(
            [
                {
                    "source_name": _expert_source(layer, "gate"),
                    "kind": "expert",
                    "role": "moe.expert.gate_up",
                    "layer_index": layer,
                    "projection": "gate_up",
                    "shape": [512, 1280, 2560],
                },
                {
                    "source_name": _expert_source(layer, "down"),
                    "kind": "expert",
                    "role": "moe.expert.down",
                    "layer_index": layer,
                    "projection": "down",
                    "shape": [512, 2560, 640],
                },
            ]
        )
    return {"tensors": tensors}


def _cells() -> list[dict]:
    rows = []
    for layer in range(48):
        for projection in ("gate", "up", "down"):
            if layer < 2:
                codec = "q8_0"
                policy = FALLBACK_POLICY
            else:
                codec = "iq2_ks" if (layer, projection) == (2, "gate") else "iq2_k"
                policy = CALIBRATED_IQK_POLICY
            logical = [640, 2560] if projection in {"gate", "up"} else [2560, 640]
            stored = (
                [2560, 768] if projection == "down" and codec in {"iq2_k", "iq2_ks"} else logical
            )
            rows.append(
                {
                    "kind": "expert",
                    "layer_index": layer,
                    "projection": projection,
                    "source_name": _expert_source(layer, projection),
                    "format": "iqk" if codec.startswith("iq2_") else "kquant",
                    "codec": codec,
                    **({"layout": IQK_LAYOUT_IQK_RELAYOUT} if codec.startswith("iq2_") else {}),
                    "logical_shape": logical,
                    "stored_shape": stored,
                    "zero_padding": stored[1] - logical[1],
                    "calibration_policy": policy,
                    "surface_cell_identity": (
                        f"{layer * 3 + ('gate', 'up', 'down').index(projection):064x}"
                    ),
                    "surface_run_contract_identity": "4" * 64,
                }
            )
    return rows


def _decision(
    path: Path,
    rows: list[dict] | None = None,
    *,
    source=None,
    extra: dict | None = None,
) -> Path:
    payload = make_artifact(
        "optimizer_decision",
        {"source_root": "Qwen/Qwen3.8-Flash-Next", "source_format": "hf_safetensors"},
        artifact_producer("test.qwen4_iqk"),
        required_features=["calibration"],
        status="valid",
        source_probe_id="probe:" + "1" * 64,
        source_identity=deepcopy(_SOURCE_IDENTITY if source is None else source),
        allocation=deepcopy(_cells() if rows is None else rows),
        achieved={},
        constraints={},
        **deepcopy(extra or {}),
    )
    write_artifact(path, payload)
    return path


def _all_iq2_k_cells() -> list[dict]:
    zeros_by_layer = {
        layer: [
            expert
            for pair_layer, expert in _EXPECTED_ZERO_COUNT_PAIRS
            if pair_layer == layer
        ]
        for layer in range(48)
    }
    rows = _cells()
    for row in rows:
        layer = int(row["layer_index"])
        projection = str(row["projection"])
        row.update(
            {
                "format": "iqk",
                "codec": "iq2_k",
                "layout": IQK_LAYOUT_IQK_RELAYOUT,
                "calibration_policy": CALIBRATED_IQK_POLICY,
                "stored_shape": (
                    [2560, 768] if projection == "down" else [640, 2560]
                ),
                "zero_padding": 128 if projection == "down" else 0,
            }
        )
        if zeros_by_layer[layer]:
            row["zero_count_experts"] = zeros_by_layer[layer]
            row["zero_count_fallback_policy"] = ZERO_COUNT_MEAN_POLICY
    return rows


def _early_iq3_k_cells() -> list[dict]:
    rows = _all_iq2_k_cells()
    for row in rows:
        if int(row["layer_index"]) < 2:
            row["codec"] = "iq3_k"
    return rows


def _joint_iqk_cells() -> list[dict]:
    zeros_by_layer = {
        layer: [
            expert
            for pair_layer, expert in _JOINT_ZERO_COUNT_PAIRS
            if pair_layer == layer
        ]
        for layer in range(48)
    }
    rows = _cells()
    for row in rows:
        layer = int(row["layer_index"])
        projection = str(row["projection"])
        codec = (
            "iq1_s_r4"
            if (layer + ("gate", "up", "down").index(projection)) % 3 == 0
            else "iq2_ks"
            if layer % 2 == 0
            else "iq2_k"
        )
        stored = (
            [2560, 768]
            if projection == "down" and codec in {"iq2_k", "iq2_ks"}
            else row["logical_shape"]
        )
        row.update(
            {
                "format": "iqk",
                "codec": codec,
                "layout": IQK_LAYOUT_IQK_RELAYOUT,
                "calibration_policy": CALIBRATED_IQK_POLICY,
                "stored_shape": stored,
                "zero_padding": stored[1] - row["logical_shape"][1],
            }
        )
        if zeros_by_layer[layer]:
            row["zero_count_experts"] = zeros_by_layer[layer]
            row["zero_count_fallback_policy"] = ZERO_COUNT_MEAN_POLICY
    return rows


def _released_inventory_and_vectors() -> tuple[dict, dict[str, np.ndarray]]:
    tensors = []
    vectors = {}
    for name, (dtype, shape) in expected_qwen38_flash_next_header_specs().items():
        resolved = qwen4_roles.tensor_role(name)
        if resolved is None or resolved["kind"] in {"provider", "unknown"}:
            continue
        entry = {
            "source_name": name,
            "kind": resolved["kind"],
            "role": resolved["role"],
            "layer_index": resolved.get("layer_index"),
            "shape": list(shape),
            "dtype": dtype,
        }
        module_path = qwen4_roles.module_path(name)
        module_weight_key = qwen4_roles.module_weight_key(name)
        if module_path is not None:
            entry["module_path"] = module_path
        if module_weight_key is not None:
            entry["module_weight_key"] = module_weight_key
        tensors.append(entry)
        if resolved["kind"] == "affine" and resolved["role"] != "embed_tokens":
            vectors[name] = np.ones(shape[-1], dtype=np.float32)
    return {"tensors": tensors}, vectors


def test_allocation_is_exactly_144_cells_with_explicit_q8_fallback(tmp_path: Path) -> None:
    path = _decision(tmp_path / "decision.json")

    cells, decision = read_qwen4_iqk_allocation(
        path,
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )

    assert len(cells) == 144
    assert decision["artifact_kind"] == "optimizer_decision"
    assert {
        cells[(layer, projection)]["codec"]
        for layer in (0, 1)
        for projection in ("gate", "up", "down")
    } == {"q8_0"}
    assert cells[(0, "down")]["stored_shape"] == [2560, 640]
    assert cells[(2, "down")]["stored_shape"] == [2560, 768]
    assert cells[(2, "gate")]["codec"] == "iq2_ks"
    assert cells[(2, "gate")]["layout"] == IQK_LAYOUT_IQK_RELAYOUT


def test_allocation_surface_identity_binds_cells_to_top_level_run(
    tmp_path: Path,
) -> None:
    rows = _cells()
    manifest = [
        {
            "layer_index": row["layer_index"],
            "projection": row["projection"],
            "surface_cell_identity": row["surface_cell_identity"],
        }
        for row in rows
    ]
    surface = {
        "schema": "qwen4_test_surface_v1",
        "content_sha256": "5" * 64,
        "run_contract_identity": "4" * 64,
        "implementation_identity": "6" * 64,
        "cell_manifest_sha256": hashlib.sha256(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }
    path = _decision(
        tmp_path / "decision.json",
        rows,
        extra={"surface_identity": surface},
    )

    cells, _decision_payload = read_qwen4_iqk_allocation(
        path,
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    assert len(cells) == 144

    drifted = deepcopy(surface)
    drifted["run_contract_identity"] = "7" * 64
    path = _decision(
        tmp_path / "drifted.json",
        rows,
        extra={"surface_identity": drifted},
    )
    with pytest.raises(
        Qwen4IQKPackageError,
        match="cells do not bind the surface run",
    ):
        read_qwen4_iqk_allocation(
            path,
            inventory=_expert_inventory(),
            source_identity=_SOURCE_IDENTITY,
        )


def test_allocation_accepts_native_width_iq1_down_cell(tmp_path: Path) -> None:
    rows = _cells()
    down = next(
        row
        for row in rows
        if row["layer_index"] == 2 and row["projection"] == "down"
    )
    down.update(
        {
            "codec": "iq1_s_r4",
            "format": "iqk",
            "layout": IQK_LAYOUT_IQK_RELAYOUT,
            "stored_shape": [2560, 640],
            "zero_padding": 0,
        }
    )
    path = _decision(tmp_path / "decision.json", rows)

    cells, _decision_payload = read_qwen4_iqk_allocation(
        path,
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    cell = cells[(2, "down")]

    assert cell["codec"] == "iq1_s_r4"
    assert cell["stored_shape"] == [2560, 640]
    assert cell["zero_padding"] == 0
    price = price_qwen4_expert_allocation(cells)
    by_key = {
        (row["layer_index"], row["projection"]): row
        for row in price["cells"]
    }
    assert by_key[(2, "down")]["row_bytes"] == iqk_geometry(
        "iq1_s_r4"
    ).bytes_per_row(640)


def test_allocation_accepts_padded_iq3_down_cell(tmp_path: Path) -> None:
    rows = _cells()
    down = next(
        row
        for row in rows
        if row["layer_index"] == 2 and row["projection"] == "down"
    )
    down.update(
        {
            "codec": "iq3_k",
            "format": "iqk",
            "layout": IQK_LAYOUT_IQK_RELAYOUT,
            "stored_shape": [2560, 768],
            "zero_padding": 128,
        }
    )
    path = _decision(tmp_path / "decision.json", rows)

    cells, _decision_payload = read_qwen4_iqk_allocation(
        path,
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    cell = cells[(2, "down")]
    assert cell["codec"] == "iq3_k"
    assert cell["stored_shape"] == [2560, 768]
    assert cell["zero_padding"] == 128

    price = price_qwen4_expert_allocation(cells)
    priced = next(
        row
        for row in price["cells"]
        if row["layer_index"] == 2 and row["projection"] == "down"
    )
    assert priced["codec"] == "iq3_k"
    assert priced["row_bytes"] == iqk_geometry("iq3_k").bytes_per_row(768)
    assert priced["zero_padding"] == 128

    allocations = {
        row["projection"]: row
        for row in _expert_allocations(cells)
        if row["layer_index"] == 2
    }

    def loader(_layer: int, _expert: int, projection: str) -> np.ndarray:
        allocation = allocations[projection]
        out_features, in_features = allocation["stored_shape"]
        width = iqk_geometry(allocation["iqk_codec"]).bytes_per_row(in_features)
        return np.zeros((out_features, width), dtype=np.uint8)

    _row, geometry = iqk_bundle_row(2, 0, allocations, expert_loader=loader)
    decoded = decode_bundle_metadata(encode_bundle_metadata({2: geometry}))[2]
    down_geometry = decoded["projections"]["down_proj"]
    assert down_geometry["iqk_codec"] == "iq3_k"
    assert down_geometry["bits"] == 3
    assert down_geometry["bytes_per_block"] == 110
    assert down_geometry["row_meta_bytes"] == 0
    assert down_geometry["in_features"] == 768
    assert down_geometry["logical_in_features"] == 640
    assert down_geometry["stored_in_features"] == 768
    assert down_geometry["zero_padding"] == 128


def test_joint_iqk_allocation_binds_a_selection_that_excludes_holes(
    tmp_path: Path,
) -> None:
    excluded = {layer: set() for layer in range(48)}
    for layer, expert in _JOINT_ZERO_COUNT_PAIRS:
        excluded[layer].add(expert)
    selected = {
        layer: tuple(
            expert
            for expert in range(512)
            if expert not in excluded[layer]
        )[:448]
        for layer in range(48)
    }
    selection = build_expert_selection(
        source_package_manifest_id="pkg:" + "a" * 64,
        layers=selected,
    )
    path = _decision(
        tmp_path / "joint.json",
        _joint_iqk_cells(),
        extra={
            "mode": ZERO_COUNT_JOINT_MODE,
            "zero_count_policy": zero_count_mean_policy_contract(
                _JOINT_ZERO_COUNT_PAIRS
            ),
            "expert_selection_artifact_id": selection["artifact_id"],
        },
    )
    cells, decision = read_qwen4_iqk_allocation(
        path,
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    retained = validate_expert_selection(selection)
    _validate_decision_expert_selection(
        decision,
        selection,
        retained,
        decision["zero_count_policy"],
    )

    assert {cell["format"] for cell in cells.values()} == {"iqk"}
    assert {cell["codec"] for cell in cells.values()} == {
        "iq1_s_r4",
        "iq2_k",
        "iq2_ks",
    }

    leaked_selection = build_expert_selection(
        source_package_manifest_id="pkg:" + "a" * 64,
        layers={
            **selected,
            0: tuple(sorted((*selected[0][:-1], 181))),
        },
    )
    leaked_decision = {
        **decision,
        "expert_selection_artifact_id": leaked_selection["artifact_id"],
    }
    with pytest.raises(Qwen4IQKPackageError, match="retains unobserved"):
        _validate_decision_expert_selection(
            leaked_decision,
            leaked_selection,
            validate_expert_selection(leaked_selection),
            decision["zero_count_policy"],
        )


def test_joint_iqk_allocation_refuses_a_missing_selection_binding(
    tmp_path: Path,
) -> None:
    path = _decision(
        tmp_path / "joint.json",
        _joint_iqk_cells(),
        extra={
            "mode": ZERO_COUNT_JOINT_MODE,
            "zero_count_policy": zero_count_mean_policy_contract(
                _JOINT_ZERO_COUNT_PAIRS
            ),
        },
    )
    with pytest.raises(Qwen4IQKPackageError, match="selection artifact id"):
        read_qwen4_iqk_allocation(
            path,
            inventory=_expert_inventory(),
            source_identity=_SOURCE_IDENTITY,
        )


def test_all_iq2_k_zero_count_specialization_is_explicit_and_exact(
    tmp_path: Path,
) -> None:
    path = _decision(
        tmp_path / "decision.json",
        _all_iq2_k_cells(),
        extra={
            "mode": ZERO_COUNT_SPECIALIZATION_MODE,
            "zero_count_policy": zero_count_mean_policy_contract(
                _EXPECTED_ZERO_COUNT_PAIRS
            ),
        },
    )

    cells, decision = read_qwen4_iqk_allocation(
        path,
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )

    assert {cell["codec"] for cell in cells.values()} == {"iq2_k"}
    assert decision["zero_count_policy"] == zero_count_mean_policy_contract(
        _EXPECTED_ZERO_COUNT_PAIRS
    )
    assert cells[(0, "gate")]["zero_count_experts"] == [
        181,
        193,
        236,
        244,
        271,
        413,
        424,
        477,
    ]
    assert cells[(1, "down")]["zero_count_experts"] == [116]
    assert "zero_count_experts" not in cells[(2, "gate")]


@pytest.mark.parametrize(
    "mutation",
    ("old_mode", "pair_drift", "policy_drift", "mixed_codec"),
)
def test_all_iq2_k_zero_count_specialization_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    rows = _all_iq2_k_cells()
    extra = {
        "mode": ZERO_COUNT_SPECIALIZATION_MODE,
        "zero_count_policy": zero_count_mean_policy_contract(
            _EXPECTED_ZERO_COUNT_PAIRS
        ),
    }
    if mutation == "old_mode":
        extra["mode"] = "all_iq2_k"
    elif mutation == "pair_drift":
        rows[0]["zero_count_experts"] = rows[0]["zero_count_experts"][:-1]
    elif mutation == "policy_drift":
        extra["zero_count_policy"]["reduction"] = "weighted_mean"
    else:
        rows[-1]["codec"] = "iq2_ks"
    path = _decision(tmp_path / f"{mutation}.json", rows, extra=extra)

    with pytest.raises(Qwen4IQKPackageError, match="zero-count|all-IQ2_K"):
        read_qwen4_iqk_allocation(
            path,
            inventory=_expert_inventory(),
            source_identity=_SOURCE_IDENTITY,
        )


def test_early_iq3_k_zero_count_specialization_is_exact(tmp_path: Path) -> None:
    path = _decision(
        tmp_path / "decision.json",
        _early_iq3_k_cells(),
        extra={
            "mode": ZERO_COUNT_EARLY_IQ3_MODE,
            "zero_count_policy": zero_count_mean_policy_contract(
                EARLY_IQ3_ZERO_COUNT_PAIRS
            ),
        },
    )

    cells, decision = read_qwen4_iqk_allocation(
        path,
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    price = price_qwen4_expert_allocation(cells)

    assert decision["mode"] == "early_iq3_k_active_mean_zero"
    assert decision["zero_count_policy"]["zero_count_pairs"] == [
        list(pair) for pair in EARLY_IQ3_ZERO_COUNT_PAIRS
    ]
    assert price["codec_counts"] == {"iq2_k": 138, "iq3_k": 6}
    assert price["stored_payload_bytes"] == 38_965_084_160
    assert cells[(0, "down")]["stored_shape"] == [2560, 768]
    assert cells[(0, "down")]["zero_padding"] == 128
    assert cells[(1, "up")]["codec"] == "iq3_k"
    assert cells[(2, "gate")]["codec"] == "iq2_k"


def test_early_iq3_k_zero_count_specialization_refuses_external_selection() -> None:
    selection = build_expert_selection(
        source_package_manifest_id="pkg:" + "a" * 64,
        layers={layer: tuple(range(448)) for layer in range(48)},
    )

    with pytest.raises(Qwen4IQKPackageError, match="retain all 512 experts"):
        _validate_decision_expert_selection(
            {"mode": ZERO_COUNT_EARLY_IQ3_MODE},
            selection,
            validate_expert_selection(selection),
            zero_count_mean_policy_contract(EARLY_IQ3_ZERO_COUNT_PAIRS),
        )


@pytest.mark.parametrize(
    "mutation",
    ("pair_drift", "selection", "early_codec", "late_codec", "down_geometry"),
)
def test_early_iq3_k_zero_count_specialization_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    rows = _early_iq3_k_cells()
    extra = {
        "mode": ZERO_COUNT_EARLY_IQ3_MODE,
        "zero_count_policy": zero_count_mean_policy_contract(
            EARLY_IQ3_ZERO_COUNT_PAIRS
        ),
    }
    if mutation == "pair_drift":
        extra["zero_count_policy"] = zero_count_mean_policy_contract(
            EARLY_IQ3_ZERO_COUNT_PAIRS[:-1]
        )
    elif mutation == "selection":
        extra["expert_selection_artifact_id"] = "select:" + "a" * 64
    elif mutation == "early_codec":
        rows[0]["codec"] = "iq2_k"
    elif mutation == "late_codec":
        next(row for row in rows if row["layer_index"] == 2)["codec"] = "iq3_k"
    else:
        down = next(
            row
            for row in rows
            if row["layer_index"] == 0 and row["projection"] == "down"
        )
        down["stored_shape"] = [2560, 640]
        down["zero_padding"] = 0
    path = _decision(tmp_path / f"{mutation}.json", rows, extra=extra)

    with pytest.raises(
        Qwen4IQKPackageError,
        match="early-IQ3_K|stored shape",
    ):
        read_qwen4_iqk_allocation(
            path,
            inventory=_expert_inventory(),
            source_identity=_SOURCE_IDENTITY,
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "fallback", "padding", "codec", "layout", "source"],
)
def test_allocation_failures_are_closed(tmp_path: Path, mutation: str) -> None:
    rows = _cells()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "fallback":
        rows[0]["codec"] = "iq2_k"
        rows[0]["format"] = "iqk"
    elif mutation == "padding":
        next(row for row in rows if row["layer_index"] == 2 and row["projection"] == "down")[
            "zero_padding"
        ] = 0
    elif mutation == "codec":
        rows[-1]["codec"] = "iq3_ks"
    elif mutation == "layout":
        rows[-1]["layout"] = "ik_wire"
    else:
        rows[-1]["source_name"] = "model.visual.blocks.0.weight"
    path = _decision(tmp_path / f"{mutation}.json", rows)

    with pytest.raises(Qwen4IQKPackageError):
        read_qwen4_iqk_allocation(
            path,
            inventory=_expert_inventory(),
            source_identity=_SOURCE_IDENTITY,
        )


def test_allocation_rejects_source_identity_drift(tmp_path: Path) -> None:
    drifted = deepcopy(_SOURCE_IDENTITY)
    drifted["shards"][0]["size_bytes"] += 1
    path = _decision(tmp_path / "decision.json", source=drifted)

    with pytest.raises(Qwen4IQKPackageError, match="source identity"):
        read_qwen4_iqk_allocation(
            path,
            inventory=_expert_inventory(),
            source_identity=_SOURCE_IDENTITY,
        )


def test_direct_policy_has_released_counts_and_exact_weight_bytes() -> None:
    inventory, vectors = _released_inventory_and_vectors()

    dense, structural, price = build_qwen4_direct_allocations(inventory, vectors)

    assert len(dense) == 726
    assert len(structural) == 341
    assert price["format_counts"] == {"bf16": 341, "q6_k": 580, "q8_0": 146}
    assert price["weight_bytes"] == 4_385_878_720
    assert price["quantized_weight_bytes"] == 4_254_936_000
    assert price["structural_bf16_bytes"] == 130_942_720
    by_name = {row["source_name"]: row for row in dense}
    assert by_name["model.language_model.embed_tokens.weight"]["kquant_codec"] == "q8_0"
    assert by_name["lm_head.weight"]["kquant_codec"] == "q6_k"
    assert all(row["format"] == "raw_dtype_passthrough" for row in structural)
    routers = [row for row in structural if row["role"] == "moe.router_gate"]
    assert len(routers) == 48
    assert {row["layer_index"] for row in routers} == set(range(48))
    assert all(row["module_weight_key"].endswith(".mlp.gate.weight") for row in routers)


@pytest.mark.parametrize(
    ("codec", "direct_bytes", "quantized_bytes"),
    [
        ("q5_k", 3_874_545_920, 3_743_603_200),
        ("q4_k", 3_393_291_520, 3_262_348_800),
        ("iq4_ks", 3_277_928_384, 3_146_985_664),
        ("iq4_k", 3_393_291_520, 3_262_348_800),
        ("iq5_k", 3_874_545_920, 3_743_603_200),
        ("iq6_k", 4_415_957_120, 4_285_014_400),
    ],
)
def test_direct_policy_supports_calibrated_lower_codec_variants(
    codec: str,
    direct_bytes: int,
    quantized_bytes: int,
) -> None:
    inventory, vectors = _released_inventory_and_vectors()

    dense, structural, price = build_qwen4_direct_allocations(
        inventory,
        vectors,
        direct_codec=codec,
    )

    assert len(dense) == 726
    assert len(structural) == 341
    assert price["policy_codec"] == codec
    assert price["format_counts"] == {"bf16": 341, codec: 580, "q8_0": 146}
    assert price["weight_bytes"] == direct_bytes
    assert price["quantized_weight_bytes"] == quantized_bytes
    by_name = {row["source_name"]: row for row in dense}
    assert by_name["model.language_model.embed_tokens.weight"]["kquant_codec"] == "q8_0"
    assert by_name["model.language_model.embed_tokens.weight"]["requires_imatrix"] is False
    head = by_name["lm_head.weight"]
    if codec.startswith("iq"):
        assert head["format"] == "iqk"
        assert head["iqk_codec"] == codec
        assert head["layout"] == IQK_LAYOUT_IQK_RELAYOUT
        assert "requires_imatrix" not in head
        assert price["kquant_placeholder_bytes"] == 146
        assert price["kquant_weight_bytes"] == 1_096_704_000
    else:
        assert head["kquant_codec"] == codec
        assert head["requires_imatrix"] is True
        assert price["kquant_placeholder_bytes"] == 726
        assert price["kquant_weight_bytes"] == quantized_bytes


def test_direct_policy_refuses_unknown_codec() -> None:
    inventory, vectors = _released_inventory_and_vectors()

    with pytest.raises(Qwen4IQKPackageError, match="direct_codec"):
        build_qwen4_direct_allocations(
            inventory,
            vectors,
            direct_codec="q9_k",
        )


def test_qwen4_dense_iqk_encoder_publishes_exact_relayout_bytes() -> None:
    from mlx_iqk import codec as iqk_codec

    rng = np.random.default_rng(91)
    matrix = rng.standard_normal((4, 512)).astype(np.float32) * 0.02
    importance = np.square(rng.standard_normal(512)).astype(np.float32) + 0.01
    target = SimpleNamespace(codec="iq5_k", source_name="dense.weight")

    packed = encode_qwen4_dense_iqk(matrix, target, importance)
    wire = unpack_dense_rows("iq5_k", packed, 512)

    assert np.array_equal(wire, iqk_codec.quantize("iq5_k", matrix, importance))


def test_expert_price_preserves_native_q8_down_and_padded_iqk_down(tmp_path: Path) -> None:
    cells, _decision_payload = read_qwen4_iqk_allocation(
        _decision(tmp_path / "decision.json"),
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )

    price = price_qwen4_expert_allocation(cells)
    by_key = {(row["layer_index"], row["projection"]): row for row in price["cells"]}

    assert price["cell_count"] == 144
    assert price["codec_counts"] == {"iq2_k": 137, "iq2_ks": 1, "q8_0": 6}
    assert by_key[(0, "down")]["row_bytes"] == 640 // 32 * 34
    assert by_key[(0, "down")]["zero_padding"] == 0
    assert by_key[(2, "down")]["row_bytes"] == iqk_geometry("iq2_k").bytes_per_row(768)
    assert by_key[(2, "down")]["zero_padding"] == 128

    allocations = {
        (row["layer_index"], row["projection"]): row for row in _expert_allocations(cells)
    }
    assert allocations[(0, "down")]["format"] == "kquant"
    assert allocations[(0, "down")]["stored_shape"] == [2560, 640]
    assert allocations[(2, "down")]["format"] == "iqk"
    assert allocations[(2, "down")]["layout"] == IQK_LAYOUT_IQK_RELAYOUT
    assert allocations[(2, "down")]["stored_shape"] == [2560, 768]


def test_expert_price_uses_physical_count_for_each_compact_layer(tmp_path: Path) -> None:
    cells, _decision_payload = read_qwen4_iqk_allocation(
        _decision(tmp_path / "decision.json"),
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    full = price_qwen4_expert_allocation(cells)
    counts = {layer: 448 for layer in range(48)}
    counts[0] = 320
    compact = price_qwen4_expert_allocation(cells, num_experts=counts)
    full_cells = {
        (row["layer_index"], row["projection"]): row
        for row in full["cells"]
    }
    compact_cells = {
        (row["layer_index"], row["projection"]): row
        for row in compact["cells"]
    }

    assert compact["num_experts"] is None
    assert compact["per_layer_num_experts"]["0"] == 320
    assert compact["per_layer_num_experts"]["2"] == 448
    assert compact_cells[(0, "gate")]["weight_bytes"] == (
        full_cells[(0, "gate")]["weight_bytes"] * 320 // 512
    )
    assert compact_cells[(2, "down")]["weight_bytes"] == (
        full_cells[(2, "down")]["weight_bytes"] * 448 // 512
    )


def test_shared_iqk_bundle_records_logical_and_stored_widths(tmp_path: Path) -> None:
    cells, _decision_payload = read_qwen4_iqk_allocation(
        _decision(tmp_path / "decision.json"),
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    rows = _expert_allocations(cells)
    allocations = {row["projection"]: row for row in rows if row["layer_index"] == 2}

    def loader(_layer: int, _expert: int, projection: str) -> np.ndarray:
        allocation = allocations[projection]
        out_features, in_features = allocation["stored_shape"]
        width = iqk_geometry(allocation["iqk_codec"]).bytes_per_row(in_features)
        return np.zeros((out_features, width), dtype=np.uint8)

    row, geometry = iqk_bundle_row(2, 0, allocations, expert_loader=loader)
    decoded = decode_bundle_metadata(encode_bundle_metadata({2: geometry}))[2]

    assert row.shape == (geometry["row_bytes"],)
    gate = decoded["projections"]["gate_proj"]
    down = decoded["projections"]["down_proj"]
    assert gate["iqk_codec"] == "iq2_ks"
    assert gate["logical_in_features"] == gate["stored_in_features"] == 2560
    assert down["logical_in_features"] == 640
    assert down["stored_in_features"] == 768
    assert down["zero_padding"] == 128


def test_shared_iqk_writer_transforms_relayout_artifacts_to_stream_major(
    tmp_path: Path,
) -> None:
    cells, _decision_payload = read_qwen4_iqk_allocation(
        _decision(tmp_path / "decision.json"),
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    rows = _expert_allocations(
        cells,
        output_layout=IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
    )
    allocations = {row["projection"]: row for row in rows if row["layer_index"] == 2}
    sources: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(991)
    for projection, allocation in allocations.items():
        out_features, in_features = allocation["stored_shape"]
        width = iqk_geometry(allocation["iqk_codec"]).bytes_per_row(in_features)
        sources[projection] = rng.integers(
            0,
            256,
            size=(out_features, width),
            dtype=np.uint8,
        )

    def loader(_layer: int, _expert: int, projection: str) -> np.ndarray:
        return sources[projection]

    row, geometry = iqk_bundle_row(
        2,
        0,
        allocations,
        expert_loader=loader,
        source_layout=IQK_LAYOUT_IQK_RELAYOUT,
    )
    decoded = decode_bundle_metadata(encode_bundle_metadata({2: geometry}))[2]
    for projection in ("gate", "up", "down"):
        key = f"{projection}_proj"
        params = decoded["projections"][key]
        component = params["blocks"]
        stored = row[
            component["offset"] : component["offset"] + component["nbytes"]
        ].reshape(component["shape"])
        assert params["layout"] == IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1
        assert params["streams"]
        assert np.array_equal(
            unpack_stream_major(
                params["iqk_codec"],
                stored,
                params["in_features"],
            ),
            sources[projection],
        )
    broken = deepcopy(geometry)
    broken["projections"]["gate_proj"]["streams"][0]["offset"] = 1
    with pytest.raises(BundleFormatError, match="offset"):
        decode_bundle_metadata(encode_bundle_metadata({2: broken}))


def test_manifest_preserves_future_iq2_ks_mix_and_padded_shape(tmp_path: Path) -> None:
    cells, _decision_payload = read_qwen4_iqk_allocation(
        _decision(tmp_path / "decision.json"),
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    allocations = _expert_allocations(cells)
    plan = build_iqk_package_plan(
        {"source_root": "synthetic", "source_format": "hf_safetensors"},
        allocations,
    )
    located = {
        located_key(row): {"shard": "model.safetensors", "key_prefix": "experts"}
        for row in allocations
    }
    manifest = build_package_manifest(
        plan,
        {
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 48,
                "hidden_size": 2560,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "layer_types": ["linear_attention"] * 48,
            },
        },
        located,
        [{"path": "model.safetensors", "size_bytes": 1, "sha256": "0" * 64}],
    )

    assert manifest["status"] == "valid"
    gate = next(
        row
        for row in manifest["tensors"]
        if row.get("layer_index") == 2 and row.get("projection") == "gate"
    )
    down = next(
        row
        for row in manifest["tensors"]
        if row.get("layer_index") == 2 and row.get("projection") == "down"
    )
    assert gate["format_params"]["iqk_codec"] == "iq2_ks"
    assert gate["format_params"]["logical_shape"] == [640, 2560]
    assert down["format_params"]["stored_shape"] == [2560, 768]
    assert down["format_params"]["zero_padding"] == 128
    assert gate["format_params"]["layout"] == IQK_LAYOUT_IQK_RELAYOUT
    assert down["format_params"]["layout"] == IQK_LAYOUT_IQK_RELAYOUT


def test_manifest_preserves_zero_count_expert_policy(tmp_path: Path) -> None:
    path = _decision(
        tmp_path / "specialized.json",
        _all_iq2_k_cells(),
        extra={
            "mode": ZERO_COUNT_SPECIALIZATION_MODE,
            "zero_count_policy": zero_count_mean_policy_contract(
                _EXPECTED_ZERO_COUNT_PAIRS
            ),
        },
    )
    cells, _decision_payload = read_qwen4_iqk_allocation(
        path,
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    allocations = _expert_allocations(cells)
    plan = build_iqk_package_plan(
        {"source_root": "synthetic", "source_format": "hf_safetensors"},
        allocations,
    )
    located = {
        located_key(row): {"shard": "model.safetensors", "key_prefix": "experts"}
        for row in allocations
    }
    manifest = build_package_manifest(
        plan,
        {
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 48,
                "hidden_size": 2560,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "layer_types": ["linear_attention"] * 48,
            },
        },
        located,
        [{"path": "model.safetensors", "size_bytes": 1, "sha256": "0" * 64}],
    )

    gate = next(
        row
        for row in manifest["tensors"]
        if row.get("layer_index") == 0 and row.get("projection") == "gate"
    )
    assert gate["format"] == "iqk"
    assert gate["format_params"]["zero_count_experts"] == [
        181,
        193,
        236,
        244,
        271,
        413,
        424,
        477,
    ]
    assert (
        gate["format_params"]["zero_count_fallback_policy"]
        == ZERO_COUNT_MEAN_POLICY
    )


def test_storage_guard_prices_output_cache_and_atomic_temp(tmp_path: Path) -> None:
    inventory, vectors = _released_inventory_and_vectors()
    dense, structural, direct = build_qwen4_direct_allocations(inventory, vectors)
    cells, _decision_payload = read_qwen4_iqk_allocation(
        _decision(tmp_path / "decision.json"),
        inventory=_expert_inventory(),
        source_identity=_SOURCE_IDENTITY,
    )
    experts = price_qwen4_expert_allocation(cells)
    ple = SimpleNamespace(rows_per_shard=2_500_012, row_bytes=320)
    pricing = {
        "direct": direct,
        "neural_plus_ple_payload_bytes": (
            direct["weight_bytes"]
            + direct["kquant_placeholder_bytes"]
            + experts["stored_payload_bytes"]
            + 102_400_491_520
        ),
    }

    guard = _storage_guard(
        pricing=pricing,
        expert_price=experts,
        direct_allocations=dense,
        passthrough=structural,
        inventory=inventory,
        ple_contract=ple,
        shard_size_gb=4.0,
        include_encode_cache=True,
    )

    assert guard["modeled_output_payload_bytes_exact"] == pricing["neural_plus_ple_payload_bytes"]
    assert guard["largest_atomic_temp_payload_bytes"] == 4 * 1024**3
    assert guard["optional_encode_cache_payload_bytes_exact"] == 9_602_677_398
    assert guard["required_free_space_bytes"] == (
        guard["modeled_output_payload_bytes_exact"]
        + guard["largest_atomic_temp_payload_bytes"]
        + guard["optional_encode_cache_payload_bytes_exact"]
        + guard["metadata_and_filesystem_reserve_bytes"]
    )

    reused = _storage_guard_with_ple_reuse(
        guard,
        ple_bytes=102_400_491_520,
    )
    assert reused["modeled_output_payload_bytes_exact"] == guard[
        "modeled_output_payload_bytes_exact"
    ]
    assert reused["physical_new_output_payload_bytes_exact"] == (
        guard["modeled_output_payload_bytes_exact"] - 102_400_491_520
    )
    assert reused["ple_reuse_payload_bytes_exact"] == 102_400_491_520
    assert reused["required_free_space_bytes"] == (
        reused["physical_new_output_payload_bytes_exact"]
        + reused["largest_atomic_temp_payload_bytes"]
        + reused["optional_encode_cache_payload_bytes_exact"]
        + reused["metadata_and_filesystem_reserve_bytes"]
    )


def test_conversion_inventory_binds_exact_cells_source_and_decision(tmp_path: Path) -> None:
    members = {2: {projection: "iq2_k" for projection in ("gate", "up", "down")}}
    shapes = {2: {projection: (2, 256) for projection in ("gate", "up", "down")}}
    root = tmp_path / "cells"
    root.mkdir()
    records = []
    row_bytes = iqk_geometry("iq2_k").bytes_per_row(256)
    for projection in ("gate", "up", "down"):
        path = root / f"layer02_{projection}.iq2_k"
        path.write_bytes(bytes(2 * 2 * row_bytes))
        records.append(
            {
                "name": path.name,
                "layer_index": 2,
                "projection": projection,
                "codec": "iq2_k",
                "layout": IQK_LAYOUT_IQK_RELAYOUT,
                "source_name": f"source.{projection}",
                "logical_shape": [2, 256],
                "stored_shape": [2, 256],
                "zero_padding": 0,
                "row_bytes": row_bytes,
                "bytes_per_expert": 2 * row_bytes,
                "size_bytes": path.stat().st_size,
                "surface_cell_identity": (f"{('gate', 'up', 'down').index(projection) + 1:064x}"),
                "surface_run_contract_identity": "4" * 64,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    artifacts = IQKConvertedArtifacts(
        root,
        members,
        shapes,
        2,
        layout=IQK_LAYOUT_IQK_RELAYOUT,
    )
    decision_id = "dec:" + "1" * 64
    teacher_identity = {"identity_sha256": "2" * 64}
    surface_identity = {
        "content_sha256": "3" * 64,
        "run_contract_identity": "4" * 64,
    }
    decision = {
        "artifact_id": decision_id,
        "teacher_identity": teacher_identity,
        "surface_identity": surface_identity,
    }
    run_body = {
        "allocation_decision_id": decision_id,
        "source_identity_sha256": _SOURCE_IDENTITY["snapshot_identity_sha256"],
        "teacher_identity_sha256": teacher_identity["identity_sha256"],
        "surface_content_sha256": surface_identity["content_sha256"],
        "surface_run_contract_identity": surface_identity["run_contract_identity"],
        "encoder": {
            "source_layout": "ik_wire",
            "published_layout": IQK_LAYOUT_IQK_RELAYOUT,
        },
        "relayout_implementation": relayout_implementation_identity(),
        "cells": [
            {key: value for key, value in record.items() if key != "sha256"} for record in records
        ],
    }
    run_contract = {
        **run_body,
        "identity_sha256": hashlib.sha256(
            json.dumps(run_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    inventory = tmp_path / "inventory.json"
    payload = {
        "schema": CONVERSION_INVENTORY_SCHEMA,
        "source_identity": _SOURCE_IDENTITY,
        "allocation_decision_id": decision_id,
        "teacher_identity": teacher_identity,
        "surface_identity": surface_identity,
        "conversion_run_contract": run_contract,
        "files": records,
    }
    inventory.write_text(json.dumps(payload))
    try:
        identity = _conversion_identity(
            inventory,
            artifacts=artifacts,
            source_identity=_SOURCE_IDENTITY,
            decision=decision,
        )
        assert identity["cells"] == 3
        assert "root" not in identity
        assert "inventory" not in identity["digests"]
        assert len(identity["cell_geometry"]) == 3
        assert len(identity["inventory_sha256"]) == 64
        assert identity["digests"]["inventory_sha256"] == identity["inventory_sha256"]
        assert str(inventory) not in json.dumps(identity)
        assert identity["digests"]["files_checked"] == 3

        specialized_policy = zero_count_mean_policy_contract(((2, 0),))
        specialized_records = deepcopy(records)
        for record in specialized_records:
            record["zero_count_experts"] = [0]
            record["zero_count_fallback_policy"] = ZERO_COUNT_MEAN_POLICY
        specialized_run_body = {
            **{key: value for key, value in run_body.items() if key != "cells"},
            "zero_count_policy": specialized_policy,
            "cells": [
                {key: value for key, value in record.items() if key != "sha256"}
                for record in specialized_records
            ],
        }
        specialized_run = {
            **specialized_run_body,
            "identity_sha256": hashlib.sha256(
                json.dumps(
                    specialized_run_body,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        specialized_payload = {
            **payload,
            "zero_count_policy": specialized_policy,
            "conversion_run_contract": specialized_run,
            "files": specialized_records,
        }
        specialized_decision = {
            **decision,
            "mode": ZERO_COUNT_SPECIALIZATION_MODE,
            "zero_count_policy": specialized_policy,
            "allocation": [
                {
                    "layer_index": record["layer_index"],
                    "projection": record["projection"],
                    "codec": record["codec"],
                }
                for record in specialized_records
            ],
        }
        inventory.write_text(json.dumps(specialized_payload))
        specialized_identity = _conversion_identity(
            inventory,
            artifacts=artifacts,
            source_identity=_SOURCE_IDENTITY,
            decision=specialized_decision,
        )
        assert specialized_identity["cells"] == 3

        file_drift = deepcopy(specialized_payload)
        file_drift["files"][0]["zero_count_experts"] = [1]
        inventory.write_text(json.dumps(file_drift))
        with pytest.raises(Qwen4IQKPackageError, match="files.*identity drifted"):
            _conversion_identity(
                inventory,
                artifacts=artifacts,
                source_identity=_SOURCE_IDENTITY,
                decision=specialized_decision,
            )

        file_ownership_drift = deepcopy(specialized_payload)
        file_ownership_drift["files"][0]["layer_index"] = 3
        file_ownership_drift["files"][0].pop("zero_count_experts")
        file_ownership_drift["files"][0].pop("zero_count_fallback_policy")
        inventory.write_text(json.dumps(file_ownership_drift))
        with pytest.raises(Qwen4IQKPackageError, match="ownership drifted"):
            _conversion_identity(
                inventory,
                artifacts=artifacts,
                source_identity=_SOURCE_IDENTITY,
                decision=specialized_decision,
            )

        run_cell_drift = deepcopy(specialized_payload)
        run_cell_drift["conversion_run_contract"]["cells"][0][
            "zero_count_fallback_policy"
        ] = "uniform"
        changed_run_body = {
            key: value
            for key, value in run_cell_drift["conversion_run_contract"].items()
            if key != "identity_sha256"
        }
        run_cell_drift["conversion_run_contract"]["identity_sha256"] = hashlib.sha256(
            json.dumps(changed_run_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        inventory.write_text(json.dumps(run_cell_drift))
        with pytest.raises(Qwen4IQKPackageError, match="run cells.*identity drifted"):
            _conversion_identity(
                inventory,
                artifacts=artifacts,
                source_identity=_SOURCE_IDENTITY,
                decision=specialized_decision,
            )

        run_ownership_drift = deepcopy(specialized_payload)
        run_record = run_ownership_drift["conversion_run_contract"]["cells"][0]
        run_record["layer_index"] = 3
        run_record.pop("zero_count_experts")
        run_record.pop("zero_count_fallback_policy")
        changed_run_body = {
            key: value
            for key, value in run_ownership_drift["conversion_run_contract"].items()
            if key != "identity_sha256"
        }
        run_ownership_drift["conversion_run_contract"]["identity_sha256"] = (
            hashlib.sha256(
                json.dumps(
                    changed_run_body,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        inventory.write_text(json.dumps(run_ownership_drift))
        with pytest.raises(Qwen4IQKPackageError, match="ownership drifted"):
            _conversion_identity(
                inventory,
                artifacts=artifacts,
                source_identity=_SOURCE_IDENTITY,
                decision=specialized_decision,
            )

        bad = deepcopy(payload)
        bad["files"].append(deepcopy(bad["files"][0]))
        inventory.write_text(json.dumps(bad))
        with pytest.raises(Qwen4IQKPackageError, match="does not match"):
            _conversion_identity(
                inventory,
                artifacts=artifacts,
                source_identity=_SOURCE_IDENTITY,
                decision=decision,
            )

        wrong_layout = deepcopy(payload)
        wrong_layout["files"][0]["layout"] = "ik_wire"
        inventory.write_text(json.dumps(wrong_layout))
        with pytest.raises(Qwen4IQKPackageError, match="package-ready relayout"):
            _conversion_identity(
                inventory,
                artifacts=artifacts,
                source_identity=_SOURCE_IDENTITY,
                decision=decision,
            )

        wrong_implementation = deepcopy(payload)
        wrong_implementation["conversion_run_contract"]["relayout_implementation"][
            "identity_sha256"
        ] = "0" * 64
        changed_body = {
            key: value
            for key, value in wrong_implementation["conversion_run_contract"].items()
            if key != "identity_sha256"
        }
        wrong_implementation["conversion_run_contract"]["identity_sha256"] = hashlib.sha256(
            json.dumps(changed_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        inventory.write_text(json.dumps(wrong_implementation))
        with pytest.raises(Qwen4IQKPackageError, match="implementation drifted"):
            _conversion_identity(
                inventory,
                artifacts=artifacts,
                source_identity=_SOURCE_IDENTITY,
                decision=decision,
            )

        unexpected_zero_policy = deepcopy(payload)
        unexpected_zero_policy["zero_count_policy"] = zero_count_mean_policy_contract(
            _EXPECTED_ZERO_COUNT_PAIRS
        )
        inventory.write_text(json.dumps(unexpected_zero_policy))
        with pytest.raises(Qwen4IQKPackageError, match="zero-count policy drifted"):
            _conversion_identity(
                inventory,
                artifacts=artifacts,
                source_identity=_SOURCE_IDENTITY,
                decision=decision,
            )

        drifted = deepcopy(payload)
        drifted["source_identity"]["shards"][0]["size_bytes"] += 1
        inventory.write_text(json.dumps(drifted))
        with pytest.raises(Qwen4IQKPackageError, match="source identity"):
            _conversion_identity(
                inventory,
                artifacts=artifacts,
                source_identity=_SOURCE_IDENTITY,
                decision=decision,
            )
    finally:
        artifacts.close()


@pytest.mark.parametrize("compact", [False, True])
def test_real_writer_streams_qwen_iqk_rows_and_binds_ple_file(
    tmp_path: Path,
    compact: bool,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    gate_up_name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    down_name = "model.language_model.layers.0.mlp.experts.down_proj"
    write_safetensors_raw(
        source / "model-00001.safetensors",
        {
            gate_up_name: ("F32", (2, 4, 256), bytes(2 * 4 * 256 * 4)),
            down_name: ("F32", (2, 4, 256), bytes(2 * 4 * 256 * 4)),
        },
    )
    members = {0: {projection: "iq2_k" for projection in ("gate", "up", "down")}}

    def target_fields(_layer: int, projection: str) -> dict:
        logical = [2, 256] if projection in {"gate", "up"} else [4, 256]
        return {
            "source_name": gate_up_name if projection in {"gate", "up"} else down_name,
            "source_projection": "gate_up" if projection in {"gate", "up"} else "down",
            "module_path": f"layers.0.mlp.experts.{projection}_proj",
            "module_weight_key": f"layers.0.mlp.experts.{projection}_proj.weight",
            "logical_shape": logical,
            "stored_shape": logical,
            "zero_padding": 0,
            "calibration_policy": CALIBRATED_IQK_POLICY,
        }

    allocations = build_iqk_expert_allocations(members, target_fields=target_fields)
    selection_id = "select:" + "b" * 64
    plan = build_iqk_package_plan(
        {"source_root": "synthetic", "source_format": "hf_safetensors"},
        allocations,
        inputs=[selection_id] if compact else None,
        additional_required_features=["qwen4_per_layer_experts"] if compact else None,
    )
    output = tmp_path / "package"
    ple_path = output / "ple/rows-000-of-001.bf16"
    ple_path.parent.mkdir(parents=True)
    ple_path.write_bytes(b"\x00\x00\x00\x00")
    ple_file = {
        "path": "ple/rows-000-of-001.bf16",
        "size_bytes": 4,
        "sha256": hashlib.sha256(ple_path.read_bytes()).hexdigest(),
    }

    loader_calls = []

    def loader(_layer: int, expert: int, projection: str) -> np.ndarray:
        loader_calls.append((_layer, expert, projection))
        out_features = 2 if projection in {"gate", "up"} else 4
        row_bytes = iqk_geometry("iq2_k").bytes_per_row(256)
        return np.full((out_features, row_bytes), expert + 1, dtype=np.uint8)

    selection_block = {
        "source_selection_artifact_id": selection_id,
        "source_package_manifest_id": "pkg:" + "a" * 64,
        "source_num_experts": 2,
        "top_k": 1,
        "layers": {
            "0": {
                "num_experts": 1,
                "source_expert_ids": [1],
            }
        },
    }
    manifest = write_package(
        plan,
        source,
        {
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 1,
                "hidden_size": 256,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 2,
                "layer_types": ["linear_attention"],
            },
        },
        output,
        iqk_expert_loader=loader,
        expert_source_ids={0: (1,)} if compact else None,
        expert_layout=(
            {
                "stacked": False,
                "bundled": True,
                "fused_gate_up": True,
                "shard_per_layer": False,
                "per_layer_experts": selection_block,
            }
            if compact
            else None
        ),
        additional_files=[ple_file],
        ple_provider={
            "schema": "qwen4_ple_provider_v1",
            "shards": [{"index": 0, "path": ple_file["path"]}],
        },
    )

    assert manifest["status"] == "valid"
    assert manifest["ple_provider"]["shards"][0]["path"] == ple_file["path"]
    shard = next(
        output / record["path"]
        for record in manifest["files"]
        if record["path"].startswith("model-")
    )
    geometry = decode_bundle_metadata(read_shard_metadata(shard)[METADATA_KEY])[0]
    assert geometry["num_experts"] == (1 if compact else 2)
    assert geometry["projections"]["gate_proj"]["logical_in_features"] == 256
    assert geometry["projections"]["down_proj"]["stored_in_features"] == 256
    expected_experts = [1] if compact else [0, 1]
    assert loader_calls == [
        (0, expert, projection)
        for expert in expected_experts
        for projection in ("gate", "up", "down")
    ]
    if compact:
        assert manifest["inputs"] == [selection_id]
        assert "qwen4_per_layer_experts" in manifest["required_features"]
        assert manifest["architecture"]["config"]["num_experts"] == 2
        assert manifest["expert_layout"]["per_layer_experts"] == selection_block
    else:
        assert manifest["inputs"] == []


def test_iqk_writer_gathers_compact_experts_in_source_id_order() -> None:
    members = {0: {projection: "iq2_k" for projection in ("gate", "up", "down")}}

    def target_fields(_layer: int, projection: str) -> dict:
        logical = [2, 256] if projection in {"gate", "up"} else [4, 256]
        return {
            "source_name": (
                "layers.0.experts.gate_up"
                if projection in {"gate", "up"}
                else "layers.0.experts.down"
            ),
            "source_projection": "gate_up" if projection in {"gate", "up"} else "down",
            "module_path": f"layers.0.mlp.experts.{projection}_proj",
            "module_weight_key": f"layers.0.mlp.experts.{projection}_proj.weight",
            "logical_shape": logical,
            "stored_shape": logical,
            "zero_padding": 0,
            "calibration_policy": CALIBRATED_IQK_POLICY,
        }

    allocations = build_iqk_expert_allocations(members, target_fields=target_fields)
    by_projection = {row["projection"]: row for row in allocations}
    calls = []

    def loader(layer: int, expert: int, projection: str) -> np.ndarray:
        calls.append((layer, expert, projection))
        out_features = 2 if projection in {"gate", "up"} else 4
        row_bytes = iqk_geometry("iq2_k").bytes_per_row(256)
        value = expert * 10 + {"gate": 1, "up": 2, "down": 3}[projection]
        return np.full((out_features, row_bytes), value, dtype=np.uint8)

    class _Writer:
        def add_streamed_bundle(self, name, layer, geometry, rows):
            self.name = name
            self.layer = layer
            self.geometry = geometry
            self.rows = [np.array(row, copy=True) for row in rows]
            return "model-00001-of-00001.safetensors"

    writer = _Writer()
    result = _write_iqk_layer_bundle_streamed(
        writer,
        0,
        by_projection,
        4,
        max_experts=None,
        iqk_expert_loader=loader,
        expert_source_ids=(1, 3),
    )

    assert result is not None
    assert writer.geometry["num_experts"] == 2
    assert len(writer.rows) == 2
    assert calls == [
        (0, 1, "gate"),
        (0, 1, "up"),
        (0, 1, "down"),
        (0, 3, "gate"),
        (0, 3, "up"),
        (0, 3, "down"),
    ]


def test_q8_bundle_metadata_records_native_logical_width() -> None:
    def encoded(rows: int) -> list[KQuantEncodedWeight]:
        return [
            KQuantEncodedWeight(
                codec="q8_0",
                weight=np.zeros((rows, 256 // 32 * 34), dtype=np.uint8),
                scales=np.zeros((1,), dtype=np.uint8),
            )
            for _expert in range(2)
        ]

    bundle, geometry = assemble_kquant_encoded_layer_bundle(
        {"gate": encoded(2), "up": encoded(2), "down": encoded(4)}
    )
    allocations = {
        projection: {
            "logical_shape": [2, 256] if projection != "down" else [4, 256],
            "stored_shape": [2, 256] if projection != "down" else [4, 256],
            "zero_padding": 0,
        }
        for projection in ("gate", "up", "down")
    }
    annotate_expert_input_geometry(geometry, allocations)
    decoded = decode_bundle_metadata(encode_bundle_metadata({0: geometry}))[0]

    assert bundle.shape == (2, geometry["row_bytes"])
    assert decoded["projections"]["down_proj"]["logical_in_features"] == 256
    assert decoded["projections"]["down_proj"]["stored_in_features"] == 256
    assert decoded["projections"]["down_proj"]["zero_padding"] == 0


def test_additional_file_identity_accepts_nested_ple_path(tmp_path: Path) -> None:
    package = tmp_path / "package"
    path = package / "ple/rows-000-of-001.bf16"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"ple")
    record = {
        "path": "ple/rows-000-of-001.bf16",
        "size_bytes": 3,
        "sha256": hashlib.sha256(b"ple").hexdigest(),
    }

    assert validate_additional_file_identities(package, [record]) == [record]


@pytest.mark.parametrize("failure", ["path", "size", "digest", "duplicate", "symlink"])
def test_additional_file_identity_refuses_invalid_boundary(
    tmp_path: Path,
    failure: str,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    path = package / "payload.bin"
    path.write_bytes(b"payload")
    record = {
        "path": "payload.bin",
        "size_bytes": 7,
        "sha256": hashlib.sha256(b"payload").hexdigest(),
    }
    records = [record]
    if failure == "path":
        record["path"] = "../payload.bin"
    elif failure == "size":
        record["size_bytes"] = 8
    elif failure == "digest":
        record["sha256"] = "0" * 64
    elif failure == "duplicate":
        records.append(deepcopy(record))
    else:
        target = package / "target.bin"
        target.write_bytes(b"payload")
        path.unlink()
        path.symlink_to(target.name)

    with pytest.raises(ValueError):
        validate_additional_file_identities(package, records)
