"""Focused contract tests for the Qwen routed IQ_K stream-layout rewriter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from moespresso.core.artifact import artifact_producer, make_artifact, read_artifact, write_artifact
from moespresso.package.bundle import (
    BUNDLE_KEY_SUFFIX,
    IQK_CODEC,
    KQUANT_CODEC,
    METADATA_KEY,
    PROJECTIONS as BUNDLE_PROJECTIONS,
    assemble_layer_bundle,
    decode_bundle_metadata,
    encode_bundle_metadata,
)
from moespresso.package.constants import MANIFEST_NAME
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
    iqk_geometry,
)
from moespresso.package.iqk_relayout import unpack_stream_major
from moespresso.package.kquant_format import KQUANT_GEOMETRY
from moespresso.package.manifest import file_identity
from moespresso.package.qwen4.iqk_package import IQK_REPORT_NAME, PACKAGE_PLAN_NAME
from moespresso.package.qwen4.iqk_stream_layout import (
    FILESYSTEM_RESERVE_BYTES,
    Qwen4IQKStreamLayoutError,
    _read_safetensors_header,
    rewrite_qwen4_iqk_stream_layout,
)
import moespresso.package.qwen4.iqk_stream_layout as stream_layout

from conftest import write_safetensors_raw


QWEN_PROJECTIONS = ("gate", "up", "down")


def _expert_source(layer: int, projection: str) -> str:
    suffix = "gate_up_proj" if projection in {"gate", "up"} else "down_proj"
    return f"model.language_model.layers.{layer}.mlp.experts.{suffix}"


def _bundle_key(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{BUNDLE_KEY_SUFFIX}"


def _bundle_bytes(path: Path) -> tuple[dict, bytes]:
    header, data_start = _read_safetensors_header(path)
    names = [name for name in header if name != "__metadata__"]
    assert len(names) == 1
    return header, path.read_bytes()[data_start:]


def _write_source_package(
    root: Path,
    *,
    extra_tensor: bool = False,
    q8_layers: tuple[int, ...] = (),
    q8_bundle_codec: str = "q8_0",
    plan_codec_override: str | None = None,
    swap_mixed_projection_shards: bool = False,
) -> dict[int, bytes]:
    root.mkdir()
    original_bundles: dict[int, bytes] = {}
    expert_files: list[dict] = []
    manifest_tensors: list[dict] = []
    allocations: list[dict] = []
    row_bytes = iqk_geometry("iq2_k").bytes_per_row(256)

    for layer in range(48):
        components = {}
        is_q8 = layer in q8_layers
        if is_q8:
            kgeometry = KQUANT_GEOMETRY[q8_bundle_codec]
            for projection_index, projection in enumerate(BUNDLE_PROJECTIONS):
                width = kgeometry.bytes_per_block
                values = np.arange(2 * 2 * width, dtype=np.uint8).reshape(2, 2, width)
                components[(projection, "weight")] = values + layer + projection_index
                components[(projection, "scales")] = np.zeros((2, 1), dtype=np.uint8)
            bundle, geometry = assemble_layer_bundle(
                components,
                bits={projection: kgeometry.bits for projection in BUNDLE_PROJECTIONS},
                codecs={projection: KQUANT_CODEC for projection in BUNDLE_PROJECTIONS},
                kquant_codecs={projection: q8_bundle_codec for projection in BUNDLE_PROJECTIONS},
            )
        else:
            for projection_index, projection in enumerate(BUNDLE_PROJECTIONS):
                values = np.arange(2 * 2 * row_bytes, dtype=np.uint8).reshape(2, 2, row_bytes)
                components[(projection, "blocks")] = values + layer + projection_index
            bundle, geometry = assemble_layer_bundle(
                components,
                bits={projection: 2 for projection in BUNDLE_PROJECTIONS},
                codecs={projection: IQK_CODEC for projection in BUNDLE_PROJECTIONS},
                iqk_codecs={projection: "iq2_k" for projection in BUNDLE_PROJECTIONS},
                iqk_layout=IQK_LAYOUT_IQK_RELAYOUT,
            )
        name = f"model-expert-{layer:05d}.safetensors"
        key = _bundle_key(layer)
        tensors = {key: ("U8", bundle.shape, bundle.tobytes())}
        if extra_tensor and layer == 0:
            tensors["unexpected"] = ("U8", [1], b"x")
        write_safetensors_raw(
            root / name,
            tensors,
            metadata={
                "format": "mjtq",
                "source_marker": "preserve",
                METADATA_KEY: encode_bundle_metadata({layer: geometry}),
            },
        )
        original_bundles[layer] = bundle.tobytes()
        expert_files.append(file_identity(root / name))
        for projection in QWEN_PROJECTIONS:
            tensor_format = KQUANT_CODEC if is_q8 else IQK_CODEC
            format_params = (
                {"kquant_codec": "q8_0"}
                if is_q8
                else {"iqk_codec": "iq2_k", "layout": IQK_LAYOUT_IQK_RELAYOUT}
            )
            manifest_tensors.append(
                {
                    "source_name": _expert_source(layer, projection),
                    "kind": "expert",
                    "format": tensor_format,
                    "layer_index": layer,
                    "projection": projection,
                    "shard": name,
                    "key_prefix": key,
                    "format_params": format_params,
                }
            )
            allocation = {
                "source_name": _expert_source(layer, projection),
                "kind": "expert",
                "format": tensor_format,
                "layer_index": layer,
                "projection": projection,
            }
            if is_q8:
                allocation["kquant_codec"] = (
                    plan_codec_override
                    if plan_codec_override is not None and layer == q8_layers[0]
                    else "q8_0"
                )
            else:
                allocation["iqk_codec"] = "iq2_k"
                allocation["layout"] = IQK_LAYOUT_IQK_RELAYOUT
            allocations.append(allocation)

    if swap_mixed_projection_shards:
        gate_zero = next(
            tensor
            for tensor in manifest_tensors
            if tensor["kind"] == "expert"
            and tensor["layer_index"] == 0
            and tensor["projection"] == "gate"
        )
        gate_two = next(
            tensor
            for tensor in manifest_tensors
            if tensor["kind"] == "expert"
            and tensor["layer_index"] == 2
            and tensor["projection"] == "gate"
        )
        gate_zero["shard"], gate_two["shard"] = gate_two["shard"], gate_zero["shard"]

    dense_name = "model-dense.safetensors"
    write_safetensors_raw(
        root / dense_name,
        {"dense.weight": ("U8", [1], b"d")},
        metadata={"format": "mjtq", "unrelated": "preserve"},
    )
    ple_name = "ple/rows-000-of-128.bf16"
    (root / "ple").mkdir()
    (root / ple_name).write_bytes(b"p" * 32)
    ple_identity = file_identity(root / ple_name)
    ple_identity["path"] = ple_name
    manifest_tensors.append(
        {
            "source_name": "model.language_model.layers.0.dense.weight",
            "kind": "affine",
            "format": IQK_CODEC,
            "shard": dense_name,
            "key_prefix": "dense",
            "format_params": {"iqk_codec": "iq6_k", "layout": IQK_LAYOUT_IQK_RELAYOUT},
        }
    )

    plan = make_artifact(
        "package_plan",
        {"model": "synthetic-qwen4"},
        artifact_producer("test.qwen4.stream-layout"),
        status="valid",
        allocation=allocations,
    )
    plan_id = write_artifact(root / PACKAGE_PLAN_NAME, plan, created_at="test")
    manifest = make_artifact(
        "package_manifest",
        {"model": "synthetic-qwen4"},
        artifact_producer("test.qwen4.stream-layout"),
        status="valid",
        architecture={"family": "qwen4_exp", "config": {"model_type": "qwen4_exp_text"}},
        tensors=manifest_tensors,
        files=[*expert_files, file_identity(root / dense_name), ple_identity],
        provenance={"source_plan_id": plan_id},
    )
    manifest_id = write_artifact(root / MANIFEST_NAME, manifest, created_at="test")
    (root / "config.json").write_text("{}")
    (root / "jang_config.json").write_text('{"mxtq_seed": 7}')
    (root / IQK_REPORT_NAME).write_text(
        json.dumps(
            {
                "expert_layout": IQK_LAYOUT_IQK_RELAYOUT,
                "package_plan_id": plan_id,
                "package_manifest_id": manifest_id,
            }
        )
    )
    return original_bundles


def _assert_iqk_inverse(
    source_bundles: dict[int, bytes],
    destination: Path,
    *,
    layers: range,
) -> None:
    for layer in layers:
        header, transformed = _bundle_bytes(destination / f"model-expert-{layer:05d}.safetensors")
        assert header["__metadata__"]["source_marker"] == "preserve"
        geometry = decode_bundle_metadata(header["__metadata__"][METADATA_KEY])[layer]
        restored = bytearray(transformed)
        for expert in range(geometry["num_experts"]):
            row_start = expert * geometry["row_bytes"]
            for projection in BUNDLE_PROJECTIONS:
                params = geometry["projections"][projection]
                assert params["layout"] == IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1
                assert params["streams"]
                blocks = params["blocks"]
                start = row_start + blocks["offset"]
                end = start + blocks["nbytes"]
                out_features, bytes_per_row = blocks["shape"]
                packed = (
                    np.frombuffer(transformed[start:end], dtype=np.uint8)
                    .copy()
                    .reshape(out_features, bytes_per_row)
                )
                restored[start:end] = unpack_stream_major(
                    params["iqk_codec"], packed, params["in_features"]
                ).tobytes()
        assert bytes(restored) == source_bundles[layer]


def test_rewrite_is_lossless_and_keeps_dense_iqk_on_relayout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    original = _write_source_package(source)
    destination = tmp_path / "stream-major"

    report = rewrite_qwen4_iqk_stream_layout(source, destination)

    assert report["expert_layout_mode"] == "all_iqk"
    assert report["expert_shards"] == 48
    assert report["expert_shards_rewritten"] == 48
    assert report["expert_shards_unchanged"] == 0
    assert report["expert_cells_rewritten"] == 144
    assert report["expert_cells_unchanged"] == 0
    assert report["allocations_moved"] == 144
    assert report["manifest_tensors_moved"] == 144
    assert report["copy"]["hardlinked_unchanged_declared_files"] == 2
    assert report["copy"]["hardlinked_unchanged_expert_shards"] == 0
    assert report["storage_guard"]["changed_expert_shards"] == 48
    assert report["storage_guard"]["sufficient"] is True
    assert (
        report["storage_guard"]["required_free_space_bytes"]
        == report["storage_guard"]["changed_expert_payload_bytes"]
        + report["storage_guard"]["largest_atomic_temp_payload_bytes"]
        + FILESYSTEM_RESERVE_BYTES
    )
    assert (source / "model-dense.safetensors").stat().st_ino == (
        destination / "model-dense.safetensors"
    ).stat().st_ino
    assert (source / "config.json").stat().st_ino != (destination / "config.json").stat().st_ino
    assert (source / "ple/rows-000-of-128.bf16").stat().st_ino == (
        destination / "ple/rows-000-of-128.bf16"
    ).stat().st_ino

    plan = read_artifact(destination / PACKAGE_PLAN_NAME)
    manifest = read_artifact(destination / MANIFEST_NAME)
    assert plan["artifact_id"] != read_artifact(source / PACKAGE_PLAN_NAME)["artifact_id"]
    assert manifest["artifact_id"] != read_artifact(source / MANIFEST_NAME)["artifact_id"]
    assert manifest["provenance"]["source_plan_id"] == plan["artifact_id"]
    assert {
        tensor["format_params"]["layout"]
        for tensor in manifest["tensors"]
        if tensor["kind"] == "expert"
    } == {IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1}
    assert {
        tensor["format_params"]["layout"]
        for tensor in manifest["tensors"]
        if tensor["kind"] != "expert" and tensor["format"] == IQK_CODEC
    } == {IQK_LAYOUT_IQK_RELAYOUT}

    _assert_iqk_inverse(original, destination, layers=range(48))

    package_report = json.loads((destination / IQK_REPORT_NAME).read_text())
    assert package_report["expert_layout"] == IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1
    assert package_report["package_plan_id"] == plan["artifact_id"]
    assert package_report["package_manifest_id"] == manifest["artifact_id"]


def test_rewrite_exact_mixed_q8_early_iq2_lattice(tmp_path: Path) -> None:
    source = tmp_path / "source"
    original = _write_source_package(source, q8_layers=(0, 1))
    source_bytes = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    destination = tmp_path / "stream-major"

    report = rewrite_qwen4_iqk_stream_layout(source, destination)

    assert report["expert_layout_mode"] == "q8_early_iq2_k"
    assert report["expert_shards"] == 48
    assert report["expert_shards_rewritten"] == 46
    assert report["expert_shards_unchanged"] == 2
    assert report["expert_cells"] == 144
    assert report["expert_cells_rewritten"] == 138
    assert report["expert_cells_unchanged"] == 6
    assert report["allocations_moved"] == 138
    assert report["manifest_tensors_moved"] == 138
    assert report["rows_inverse_checked"] == 46 * 2 * len(BUNDLE_PROJECTIONS) * 2
    assert report["storage_guard"]["changed_expert_shards"] == 46
    changed_sizes = [
        (source / f"model-expert-{layer:05d}.safetensors").stat().st_size for layer in range(2, 48)
    ]
    assert report["storage_guard"]["changed_expert_payload_bytes"] == sum(changed_sizes)
    assert report["storage_guard"]["largest_atomic_temp_payload_bytes"] == max(changed_sizes)
    assert report["storage_guard"]["required_free_space_bytes"] == (
        sum(changed_sizes) + max(changed_sizes) + FILESYSTEM_RESERVE_BYTES
    )
    assert report["copy"]["hardlinked_unchanged_declared_files"] == 4
    assert report["copy"]["hardlinked_unchanged_expert_shards"] == 2

    for layer in (0, 1):
        source_shard = source / f"model-expert-{layer:05d}.safetensors"
        output_shard = destination / source_shard.name
        assert source_shard.stat().st_ino == output_shard.stat().st_ino
        assert source_shard.read_bytes() == output_shard.read_bytes()
        assert _bundle_bytes(output_shard)[1] == original[layer]
    _assert_iqk_inverse(original, destination, layers=range(2, 48))

    source_plan = read_artifact(source / PACKAGE_PLAN_NAME)
    output_plan = read_artifact(destination / PACKAGE_PLAN_NAME)
    source_manifest = read_artifact(source / MANIFEST_NAME)
    output_manifest = read_artifact(destination / MANIFEST_NAME)
    assert [row for row in source_plan["allocation"] if row["layer_index"] < 2] == [
        row for row in output_plan["allocation"] if row["layer_index"] < 2
    ]
    assert [
        row
        for row in source_manifest["tensors"]
        if row.get("kind") == "expert" and row["layer_index"] < 2
    ] == [
        row
        for row in output_manifest["tensors"]
        if row.get("kind") == "expert" and row["layer_index"] < 2
    ]
    assert {
        row["format_params"]["layout"]
        for row in output_manifest["tensors"]
        if row.get("kind") == "expert" and row["layer_index"] >= 2
    } == {IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1}
    assert source_bytes == {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"q8_layers": (1, 2)}, "six early Q8_0 cells"),
        (
            {"q8_layers": (0, 1), "plan_codec_override": "q6_k"},
            "plan and manifest expert cells or codecs disagree",
        ),
        (
            {"q8_layers": (0, 1), "q8_bundle_codec": "q2_k"},
            "is not an unchanged Q8_0 projection",
        ),
        (
            {"q8_layers": (0, 1), "swap_mixed_projection_shards": True},
            "not one homogeneous isolated layer",
        ),
    ],
)
def test_rewrite_rejects_other_mixed_lattices_and_q8_contract_drift(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    source = tmp_path / "source"
    _write_source_package(source, **kwargs)
    destination = tmp_path / "stream-major"

    with pytest.raises(Qwen4IQKStreamLayoutError, match=message):
        rewrite_qwen4_iqk_stream_layout(source, destination)

    assert not destination.exists()


def test_rewrite_rejects_tampered_unchanged_q8_file_before_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source_package(source, q8_layers=(0, 1))
    q8_shard = source / "model-expert-00000.safetensors"
    payload = bytearray(q8_shard.read_bytes())
    payload[-1] ^= 1
    q8_shard.write_bytes(payload)
    destination = tmp_path / "stream-major"

    with pytest.raises(Qwen4IQKStreamLayoutError, match="declared identity"):
        rewrite_qwen4_iqk_stream_layout(source, destination)

    assert not destination.exists()


def test_rewrite_refuses_in_place_nonempty_and_extra_bundle_tensor(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source_package(source)
    with pytest.raises(Qwen4IQKStreamLayoutError, match="in-place"):
        rewrite_qwen4_iqk_stream_layout(source, source)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "already-here").write_text("x")
    with pytest.raises(Qwen4IQKStreamLayoutError, match="empty"):
        rewrite_qwen4_iqk_stream_layout(source, occupied)

    malformed = tmp_path / "malformed"
    _write_source_package(malformed, extra_tensor=True)
    malformed_output = tmp_path / "malformed-output"
    with pytest.raises(Qwen4IQKStreamLayoutError, match="one bundle tensor"):
        rewrite_qwen4_iqk_stream_layout(malformed, malformed_output)
    assert not malformed_output.exists()


def test_rewrite_refuses_insufficient_space_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_source_package(source)
    destination = tmp_path / "stream-major"
    monkeypatch.setattr(
        stream_layout.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1, used=1, free=1),
    )

    with pytest.raises(Qwen4IQKStreamLayoutError, match="insufficient output free space"):
        rewrite_qwen4_iqk_stream_layout(source, destination)
    assert not destination.exists()
