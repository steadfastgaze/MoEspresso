import hashlib
import json
import shutil
import struct
from pathlib import Path

import pytest

from moespresso.core.artifact import (
    compute_artifact_id,
    make_artifact,
    read_artifact,
    write_artifact,
)
from moespresso.package.bundle import METADATA_KEY, encode_bundle_metadata
from moespresso.package.deepseek_v4.iqk_reap import (
    EXPERT_SELECTION_NAME,
    HASH_LAYERS,
    IQKReapError,
    REAP_REPORT_NAME,
    _compare_byte_spans,
    _read_safetensors_header,
    build_expert_selection,
    compact_iqk_package,
    rewrite_expert_hotlist,
    validate_expert_selection,
    validate_projection_source_map,
)
from moespresso.package.iqk_format import IQK_LAYOUT_IQK_RELAYOUT
from moespresso.package.manifest import file_identity


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]],
                       *, metadata: dict | None = None) -> None:
    header: dict[str, dict] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    cursor = 0
    for key, (dtype, shape, payload) in sorted(tensors.items()):
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for key in sorted(tensors):
            handle.write(tensors[key][2])


def _tensor_bytes(path: Path, key: str) -> bytes:
    header, data_start = _read_safetensors_header(path)
    start, end = header[key]["data_offsets"]
    with open(path, "rb") as handle:
        handle.seek(data_start + start)
        return handle.read(end - start)


def _iqk_geometry(
    layer: int,
    experts: int = 256,
    *,
    member: str = "iq2_k",
) -> dict:
    del layer
    formats = {
        "iq1_s_r4": {
            "bits": 1,
            "ggml_type": 219,
            "weights_per_block": 32,
            "bytes_per_block": 6,
            "row_meta_bytes": 2,
            "bytes_per_row": 50,
        },
        "iq2_k": {
            "bits": 2,
            "ggml_type": 137,
            "weights_per_block": 256,
            "bytes_per_block": 76,
            "row_meta_bytes": 0,
            "bytes_per_row": 76,
        },
    }
    fmt = formats[member]
    projections = {}
    offset = 0
    for projection in ("gate_proj", "up_proj", "down_proj"):
        projections[projection] = {
            "codec": "iqk",
            "bits": fmt["bits"],
            "iqk_codec": member,
            "layout": IQK_LAYOUT_IQK_RELAYOUT,
            "ggml_type": fmt["ggml_type"],
            "weights_per_block": fmt["weights_per_block"],
            "bytes_per_block": fmt["bytes_per_block"],
            "row_meta_bytes": fmt["row_meta_bytes"],
            "bytes_per_row": fmt["bytes_per_row"],
            "in_features": 256,
            "blocks": {
                "offset": offset,
                "nbytes": fmt["bytes_per_row"],
                "shape": [1, fmt["bytes_per_row"]],
                "dtype": "U8",
            },
        }
        offset += fmt["bytes_per_row"]
    return {"num_experts": experts, "row_bytes": offset, "projections": projections}


def _bundle_payload(
    layer: int,
    *,
    experts: int = 256,
    member: str = "iq2_k",
) -> bytes:
    row_bytes = _iqk_geometry(layer, experts, member=member)["row_bytes"]
    return b"".join(
        bytes((layer * 17 + expert * 3 + column) % 256 for column in range(row_bytes))
        for expert in range(experts)
    )


def _write_bundle(
    path: Path,
    layer: int,
    *,
    member: str = "iq2_k",
) -> tuple[str, bytes]:
    key = f"layers.{layer}.ffn.experts.tq_bundle"
    geometry = _iqk_geometry(layer, member=member)
    payload = _bundle_payload(layer, member=member)
    _write_safetensors(
        path,
        {key: ("U8", [256, geometry["row_bytes"]], payload)},
        metadata={
            "format": "mjtq",
            METADATA_KEY: encode_bundle_metadata({layer: geometry}),
        },
    )
    return key, payload


def _package(tmp_path: Path) -> tuple[Path, Path, dict[int, list[int]]]:
    package = tmp_path / "source"
    package.mkdir()
    bundle_files = []
    tensors = []
    plan_allocations = []
    for layer in range(4):
        name = f"model-{layer + 1:05d}-of-00005.safetensors"
        key, _ = _write_bundle(package / name, layer)
        bundle_files.append(name)
        for projection in ("gate", "up", "down"):
            source_name = f"layers.{layer}.ffn.experts.{projection}"
            tensors.append({
                "source_name": source_name,
                "role": f"moe.expert.{projection}",
                "kind": "expert",
                "layer_index": layer,
                "projection": projection,
                "shard": name,
                "key_prefix": key.removesuffix(".tq_bundle"),
                "format": "iqk",
                "format_params": {
                    "iqk_codec": "iq2_k",
                    "layout": IQK_LAYOUT_IQK_RELAYOUT,
                },
            })
            plan_allocations.append({
                "source_name": source_name,
                "role": f"moe.expert.{projection}",
                "kind": "expert",
                "layer_index": layer,
                "projection": projection,
                "bits": 2,
                "format": "iqk",
                "codec": "iq2_k",
                "iqk_codec": "iq2_k",
                "layout": IQK_LAYOUT_IQK_RELAYOUT,
            })

    regular_name = "model-00005-of-00005.safetensors"
    weight = b"".join(struct.pack("<HH", expert, 1000 + expert) for expert in range(256))
    bias = b"".join(struct.pack("<f", expert + 0.25) for expert in range(256))
    furniture = b"fixed"
    _write_safetensors(package / regular_name, {
        "layers.3.ffn.gate.bias": ("F32", [256], bias),
        "layers.3.ffn.gate.weight": ("F16", [256, 2], weight),
        "unchanged": ("U8", [len(furniture)], furniture),
    }, metadata={"format": "mjtq"})
    tensors.extend([
        {
            "source_name": "layers.3.ffn.gate.weight",
            "role": "moe.router_gate",
            "kind": "passthrough",
            "layer_index": 3,
            "shard": regular_name,
            "key_prefix": "layers.3.ffn.gate.weight",
            "format": "fp16",
            "format_params": {},
        },
        {
            "source_name": "layers.3.ffn.gate.bias",
            "role": "moe.router_bias",
            "kind": "passthrough",
            "layer_index": 3,
            "shard": regular_name,
            "key_prefix": "layers.3.ffn.gate.bias",
            "format": "raw_dtype_passthrough",
            "format_params": {},
        },
    ])
    all_files = bundle_files + [regular_name]
    draft_name = "model-dspark-00001.safetensors"
    draft_payload = b"draft expert bytes"
    _write_safetensors(
        package / draft_name,
        {"draft.weight": ("U8", [len(draft_payload)], draft_payload)},
    )
    draft_file_payload = (package / draft_name).read_bytes()
    sidecar = {
        "artifact_kind": "deepseek_v4_dspark_sidecar",
        "schema_version": {"major": 1, "minor": 0},
        "producer": {"tool": "test", "version": "1"},
        "subject": {"family": "deepseek_v4_flash_dspark"},
        "status": "valid",
        "provenance": {
            "file_sha256": {
                draft_name: hashlib.sha256(draft_file_payload).hexdigest(),
            },
            "iqk_staging": {"member": "iq2_k"},
        },
        "tensors": {"blocks.0.x": {"format": "passthrough", "file": draft_name}},
    }
    sidecar["artifact_id"] = compute_artifact_id(sidecar)
    (package / "dspark_sidecar.json").write_text(json.dumps(sidecar))
    sidecar_identities = [
        file_identity(package / "dspark_sidecar.json"),
        file_identity(package / draft_name),
    ]
    plan = make_artifact(
        "package_plan",
        {"source_root": "synthetic", "source_format": "hf_safetensors"},
        {"tool": "test", "version": "1"},
        required_features=["calibration"],
        status="valid",
        allocation=plan_allocations,
        achieved={"expert_format_counts": {"iqk": 12}},
        source_constraints={
            "source_num_experts": 256,
            "iqk_artifacts": {
                "root": "/private/build/iqk/tensors",
                "digests": {"inventory": "records/iqk_inventory.json"},
            },
        },
        optimized_kernels_expected=True,
    )
    write_artifact(package / "package_plan.json", plan)
    manifest = make_artifact(
        "package_manifest",
        {"source_root": "synthetic", "source_format": "hf_safetensors"},
        {"tool": "test", "version": "1"},
        required_features=["calibration"],
        status="valid",
        architecture={
            "family": "deepseek_v4_flash",
            "config": {
                "num_hidden_layers": 4,
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
                "num_hash_layers": 3,
            },
        },
        tensors=tensors,
        required_ops=["iqk_dequant", "fp16_passthrough", "raw_dtype_passthrough"],
        files=[file_identity(package / name) for name in all_files],
        tokenizer={"files": [], "has_tokenizer": False, "rendering_id": None},
        optimized_kernels_expected=True,
        provenance={"source_plan_id": plan["artifact_id"], "package_plan": {}},
        package_format="mjtq",
        package_format_version=1,
        expert_layout={"bundled": True, "key_suffixes": ["tq_bundle"]},
        drafter={
            "family": "dspark",
            "optional": True,
            "manifest_path": "dspark_sidecar.json",
            "sidecar_artifact_id": sidecar["artifact_id"],
            "experts_format": "iq2_k",
            "files": sidecar_identities,
            "provenance": {
                "sidecar_kind": "deepseek_v4_dspark_sidecar",
                "source_package_manifest_id": "pkg:" + "0" * 64,
            },
        },
    )
    write_artifact(package / "package_manifest.json", manifest)
    (package / "iqk_package_report.json").write_text("stale")
    (package / "iqk_relayout_report.json").write_text("stale")
    (package / "dspark_bundle_report.json").write_text("stale")
    (package / "source_inventory.json").write_text('{"source": true}')
    (package / "expert_hotlist.json").write_text(json.dumps({
        "version": 1,
        "kind": "expert_hotlist",
        "source": {"kind": "synthetic"},
        "layers": {
            str(layer): {str(expert): 1000 - expert for expert in range(256)}
            for layer in range(4)
        },
    }))

    retained = {
        **{layer: list(range(256)) for layer in HASH_LAYERS},
        3: [1, 7, 9, 11, 13, 17, 200, 255],
    }
    selection = build_expert_selection(
        source_package_manifest_id=manifest["artifact_id"],
        layers=retained,
        top_k=6,
    )
    selection_path = tmp_path / "selection.json"
    write_artifact(selection_path, selection)
    return package, selection_path, retained


def _clone_package_with_member(source: Path, output: Path, member: str) -> dict:
    shutil.copytree(source, output)
    for layer in range(4):
        name = f"model-{layer + 1:05d}-of-00005.safetensors"
        _write_bundle(output / name, layer, member=member)

    geometry = _iqk_geometry(0, member=member)["projections"]["gate_proj"]
    plan = read_artifact(output / "package_plan.json")
    for allocation in plan["allocation"]:
        if allocation.get("kind") != "expert":
            continue
        allocation.update({
            "bits": geometry["bits"],
            "codec": member,
            "format": "iqk",
            "iqk_codec": member,
            "layout": IQK_LAYOUT_IQK_RELAYOUT,
        })
    plan["achieved"]["expert_codec_counts"] = {member: 12}
    plan.pop("artifact_id")
    plan.pop("created_at", None)
    plan_id = write_artifact(output / "package_plan.json", plan)

    manifest = read_artifact(output / "package_manifest.json")
    for tensor in manifest["tensors"]:
        if tensor.get("kind") != "expert":
            continue
        tensor["format_params"] = {
            "bits": geometry["bits"],
            "bytes_per_block": geometry["bytes_per_block"],
            "ggml_type": geometry["ggml_type"],
            "iqk_codec": member,
            "layout": IQK_LAYOUT_IQK_RELAYOUT,
            "row_meta_bytes": geometry["row_meta_bytes"],
            "weights_per_block": geometry["weights_per_block"],
        }
    manifest["files"] = [
        file_identity(output / identity["path"])
        for identity in manifest["files"]
    ]
    manifest["provenance"]["source_plan_id"] = plan_id
    manifest.pop("artifact_id")
    manifest.pop("created_at", None)
    write_artifact(output / "package_manifest.json", manifest)
    return read_artifact(output / "package_manifest.json")


def _projection_sources(*, promoted: set[tuple[int, str]]) -> dict[str, dict[str, str]]:
    return {
        str(layer): {
            projection: (
                "promotion" if (layer, projection) in promoted else "base")
            for projection in ("gate", "up", "down")
        }
        for layer in range(4)
    }


def test_selection_requires_full_hash_layers_and_sorted_score_ids():
    layers = {layer: list(range(256)) for layer in range(4)}
    with pytest.raises(IQKReapError, match="top_k must be 6"):
        build_expert_selection(
            source_package_manifest_id="pkg:" + "a" * 64,
            layers=layers,
            top_k=5,
        )
    selection = build_expert_selection(
        source_package_manifest_id="pkg:" + "a" * 64,
        layers=layers,
        top_k=6,
    )
    assert validate_expert_selection(selection)[3] == tuple(range(256))

    broken = json.loads(json.dumps(selection))
    broken["layers"]["0"]["source_expert_ids"][-1] = 254
    with pytest.raises(IQKReapError, match="strictly increasing|full identity"):
        validate_expert_selection(broken)

    broken = json.loads(json.dumps(selection))
    broken["layers"]["3"] = {"num_experts": 3, "source_expert_ids": [9, 3, 4]}
    with pytest.raises(IQKReapError, match="strictly increasing"):
        validate_expert_selection(broken)

    broken["layers"]["3"] = {"num_experts": 5, "source_expert_ids": [0, 1, 2, 3, 4]}
    with pytest.raises(IQKReapError, match="below top_k"):
        validate_expert_selection(broken)


def test_rewrite_hotlist_maps_original_ids_to_compact_ids():
    selection = build_expert_selection(
        source_package_manifest_id="pkg:" + "a" * 64,
        layers={
            0: list(range(256)),
            1: list(range(256)),
            2: list(range(256)),
            3: [1, 7, 9, 11, 13, 17, 200, 255],
        },
        top_k=6,
    )
    hotlist = {
        "kind": "expert_hotlist",
        "version": 1,
        "layers": {
            "0": {"4": 3},
            "1": {"5": 4},
            "2": {"6": 5},
            "3": {
                "200": 90,
                "4": 80,
                "1": 70,
                "255": 60,
                "9": 50,
                "17": 40,
            },
        },
    }
    out = rewrite_expert_hotlist(hotlist, selection)
    assert out["layers"]["3"] == {
        "6": 90,
        "0": 70,
        "7": 60,
        "2": 50,
        "5": 40,
    }
    assert out["source"]["source_selection_artifact_id"] == selection["artifact_id"]


def test_projection_source_map_requires_every_cell_and_layer():
    with pytest.raises(IQKReapError, match="covers.*not"):
        validate_projection_source_map(
            {"0": {"gate": "base", "up": "promotion"}},
            expected_layers=[0],
        )
    with pytest.raises(IQKReapError, match="layers.*package layers"):
        validate_projection_source_map(
            {"0": {"gate": "base", "up": "base", "down": "base"}},
            expected_layers=[0, 1],
        )


def test_compact_package_preserves_selected_bundle_rows_and_gathers_router(tmp_path):
    package, selection_path, retained = _package(tmp_path)
    output = tmp_path / "compact"
    calls = []

    def source_gate(path, manifest):
        calls.append(("source", path, manifest["artifact_id"]))

    report = compact_iqk_package(
        package,
        selection_path,
        output,
        source_verifier=source_gate,
    )
    assert [call[0] for call in calls] == ["source"]
    assert report["layers"]["3"]["num_experts"] == 8
    assert report["source_package"] == package.name
    assert report["output_package"] == output.name
    assert str(tmp_path) not in (output / REAP_REPORT_NAME).read_text()

    source_bundle = package / "model-00004-of-00005.safetensors"
    output_bundle = output / "model-00004-of-00005.safetensors"
    source_rows = _tensor_bytes(source_bundle, "layers.3.ffn.experts.tq_bundle")
    output_rows = _tensor_bytes(output_bundle, "layers.3.ffn.experts.tq_bundle")
    want_rows = b"".join(
        source_rows[expert * 228:(expert + 1) * 228] for expert in retained[3]
    )
    assert output_rows == want_rows
    bundle_header, _ = _read_safetensors_header(output_bundle)
    assert bundle_header["layers.3.ffn.experts.tq_bundle"]["shape"] == [8, 228]
    metadata = json.loads(bundle_header["__metadata__"][METADATA_KEY])
    assert metadata["layers"]["3"]["num_experts"] == 8

    source_regular = package / "model-00005-of-00005.safetensors"
    output_regular = output / "model-00005-of-00005.safetensors"
    weight = _tensor_bytes(source_regular, "layers.3.ffn.gate.weight")
    bias = _tensor_bytes(source_regular, "layers.3.ffn.gate.bias")
    assert _tensor_bytes(output_regular, "layers.3.ffn.gate.weight") == b"".join(
        weight[expert * 4:(expert + 1) * 4] for expert in retained[3]
    )
    assert _tensor_bytes(output_regular, "layers.3.ffn.gate.bias") == b"".join(
        bias[expert * 4:(expert + 1) * 4] for expert in retained[3]
    )
    assert _tensor_bytes(output_regular, "unchanged") == b"fixed"
    regular_header, _ = _read_safetensors_header(output_regular)
    assert regular_header["layers.3.ffn.gate.weight"]["shape"] == [8, 2]
    assert regular_header["layers.3.ffn.gate.bias"]["shape"] == [8]

    selection = read_artifact(output / EXPERT_SELECTION_NAME)
    manifest = read_artifact(output / "package_manifest.json")
    plan = read_artifact(output / "package_plan.json")
    block = manifest["expert_layout"]["per_layer_experts"]
    assert block["source_selection_artifact_id"] == selection["artifact_id"]
    assert block["source_package_manifest_id"] == selection["source_package_manifest_id"]
    assert block["source_num_experts"] == 256
    assert block["top_k"] == 6
    assert block["layers"] == selection["layers"]
    assert manifest["architecture"]["config"]["n_routed_experts"] == 256
    source_sidecar = json.loads((package / "dspark_sidecar.json").read_text())
    assert manifest["drafter"]["sidecar_artifact_id"] == source_sidecar["artifact_id"]
    assert manifest["drafter"]["provenance"]["source_package_manifest_id"] == (
        report["compact_trunk_manifest_id"]
    )
    assert (output / "dspark_sidecar.json").read_bytes() == (
        package / "dspark_sidecar.json").read_bytes()
    assert (output / "model-dspark-00001.safetensors").read_bytes() == (
        package / "model-dspark-00001.safetensors").read_bytes()
    assert plan["expert_selection"] == block
    assert "root" not in plan["source_constraints"]["iqk_artifacts"]
    assert plan["source_constraints"]["iqk_artifacts"]["digests"] == {
        "inventory": "records/iqk_inventory.json"
    }
    assert "deepseek_v4_per_layer_experts" in manifest["required_features"]
    assert [entry["path"] for entry in manifest["files"]].count(
        EXPERT_SELECTION_NAME) == 1
    assert (output / REAP_REPORT_NAME).is_file()
    assert (output / "source_inventory.json").is_file()
    assert not (output / "iqk_package_report.json").exists()
    assert not (output / "iqk_relayout_report.json").exists()
    assert not (output / "dspark_bundle_report.json").exists()
    assert (output / "model-dspark-00001.safetensors").is_file()
    hotlist = json.loads((output / "expert_hotlist.json").read_text())
    assert set(hotlist["layers"]["3"]) == {str(index) for index in range(8)}


def test_compact_package_splices_projection_cells_from_promotion_package(tmp_path):
    promotion, _selection_path, retained = _package(tmp_path)
    base = tmp_path / "base-iq1"
    base_manifest = _clone_package_with_member(promotion, base, "iq1_s_r4")
    selection = build_expert_selection(
        source_package_manifest_id=base_manifest["artifact_id"],
        layers=retained,
        top_k=6,
    )
    selection_path = tmp_path / "iq1-selection.json"
    write_artifact(selection_path, selection)
    sources = _projection_sources(promoted={(3, "gate"), (3, "down")})
    output = tmp_path / "mixed-compact"
    verified = []

    report = compact_iqk_package(
        base,
        selection_path,
        output,
        promotion_package_dir=promotion,
        projection_sources=sources,
        source_verifier=lambda path, manifest: verified.append(
            (path.name, manifest["artifact_id"])),
        include_drafter=False,
    )

    assert [name for name, _artifact_id in verified] == [base.name, promotion.name]
    source_key = "layers.3.ffn.experts.tq_bundle"
    base_rows = _tensor_bytes(base / "model-00004-of-00005.safetensors", source_key)
    promotion_rows = _tensor_bytes(
        promotion / "model-00004-of-00005.safetensors", source_key)
    output_bundle = output / "model-00004-of-00005.safetensors"
    output_rows = _tensor_bytes(output_bundle, source_key)
    expected = bytearray()
    for expert in retained[3]:
        expected.extend(promotion_rows[expert * 228:expert * 228 + 76])
        expected.extend(base_rows[expert * 150 + 50:expert * 150 + 100])
        expected.extend(promotion_rows[expert * 228 + 152:(expert + 1) * 228])
    assert output_rows == bytes(expected)

    header, _ = _read_safetensors_header(output_bundle)
    assert header[source_key]["shape"] == [8, 202]
    geometry = json.loads(header["__metadata__"][METADATA_KEY])["layers"]["3"]
    assert geometry["row_bytes"] == 202
    assert geometry["projections"]["gate_proj"]["iqk_codec"] == "iq2_k"
    assert geometry["projections"]["gate_proj"]["blocks"]["offset"] == 0
    assert geometry["projections"]["up_proj"]["iqk_codec"] == "iq1_s_r4"
    assert geometry["projections"]["up_proj"]["blocks"]["offset"] == 76
    assert geometry["projections"]["down_proj"]["iqk_codec"] == "iq2_k"
    assert geometry["projections"]["down_proj"]["blocks"]["offset"] == 126

    manifest = read_artifact(output / "package_manifest.json")
    plan = read_artifact(output / "package_plan.json")
    manifest_members = {
        (tensor["layer_index"], tensor["projection"]):
            tensor["format_params"]["iqk_codec"]
        for tensor in manifest["tensors"] if tensor.get("kind") == "expert"
    }
    plan_members = {
        (allocation["layer_index"], allocation["projection"]):
            allocation["iqk_codec"]
        for allocation in plan["allocation"] if allocation.get("kind") == "expert"
    }
    assert manifest_members[(3, "gate")] == "iq2_k"
    assert manifest_members[(3, "up")] == "iq1_s_r4"
    assert manifest_members[(3, "down")] == "iq2_k"
    assert plan_members == manifest_members
    assert plan["achieved"]["expert_codec_counts"] == {
        "iq1_s_r4": 10,
        "iq2_k": 2,
    }
    source_block = manifest["provenance"]["expert_projection_sources"]
    assert source_block == report["expert_projection_sources"]
    assert source_block["base_package_manifest_id"] == base_manifest["artifact_id"]
    assert source_block["layers"]["3"] == {
        "gate": "promotion",
        "up": "base",
        "down": "promotion",
    }
    layer_report = next(
        row for row in report["bundle_shards"] if row["layer"] == 3)
    assert layer_report["byte_comparison"]["bytes"] == 8 * 202
    assert set(layer_report["byte_comparison"]["sources"]) == {
        "base",
        "promotion",
    }


def test_promotion_package_rejects_a_different_nonexpert_shard(tmp_path):
    promotion, _selection_path, retained = _package(tmp_path)
    base = tmp_path / "base-iq1"
    base_manifest = _clone_package_with_member(promotion, base, "iq1_s_r4")
    selection = build_expert_selection(
        source_package_manifest_id=base_manifest["artifact_id"],
        layers=retained,
        top_k=6,
    )
    selection_path = tmp_path / "iq1-selection.json"
    write_artifact(selection_path, selection)

    promotion_manifest = read_artifact(promotion / "package_manifest.json")
    regular = next(
        identity for identity in promotion_manifest["files"]
        if identity["path"] == "model-00005-of-00005.safetensors")
    regular["sha256"] = "f" * 64
    promotion_manifest.pop("artifact_id")
    promotion_manifest.pop("created_at", None)
    write_artifact(promotion / "package_manifest.json", promotion_manifest)

    with pytest.raises(IQKReapError, match="non-expert shard.*differs"):
        compact_iqk_package(
            base,
            selection_path,
            tmp_path / "mixed-compact",
            promotion_package_dir=promotion,
            projection_sources=_projection_sources(promoted={(3, "gate")}),
            source_verifier=lambda _path, _manifest: None,
            output_verifier=lambda _path, _manifest: None,
            include_drafter=False,
        )


def test_compact_package_can_omit_optional_drafter(tmp_path):
    package, selection_path, _ = _package(tmp_path)
    output = tmp_path / "compact-without-drafter"

    report = compact_iqk_package(
        package,
        selection_path,
        output,
        include_drafter=False,
    )

    manifest = read_artifact(output / "package_manifest.json")
    assert report["drafter"] is None
    assert "drafter" not in manifest
    assert not (output / "dspark_sidecar.json").exists()
    assert not (output / "model-dspark-00001.safetensors").exists()


def test_failure_does_not_publish_output_or_modify_source(tmp_path):
    package, selection_path, _ = _package(tmp_path)
    output = tmp_path / "compact"
    manifest_before = (package / "package_manifest.json").read_bytes()
    shard_before = (package / "model-00004-of-00005.safetensors").read_bytes()

    def fail_output(_path, _manifest):
        raise IQKReapError("injected output failure")

    with pytest.raises(IQKReapError, match="injected output failure"):
        compact_iqk_package(
            package,
            selection_path,
            output,
            source_verifier=lambda _path, _manifest: None,
            output_verifier=fail_output,
        )
    assert not output.exists()
    assert (package / "package_manifest.json").read_bytes() == manifest_before
    assert (package / "model-00004-of-00005.safetensors").read_bytes() == shard_before
    assert not list(tmp_path.glob(".compact.reap-*"))


def test_all_row_comparison_catches_an_injected_output_corruption(tmp_path):
    package, selection_path, _ = _package(tmp_path)
    output = tmp_path / "compact"
    corrupted = False

    def corrupt_then_compare(source, written, spans):
        nonlocal corrupted
        if not corrupted:
            with open(written, "r+b") as handle:
                handle.seek(spans[0][1])
                original = handle.read(1)
                handle.seek(spans[0][1])
                handle.write(bytes([original[0] ^ 0xFF]))
            corrupted = True
        return _compare_byte_spans(source, written, spans)

    with pytest.raises(IQKReapError, match="selected expert bytes changed"):
        compact_iqk_package(
            package,
            selection_path,
            output,
            source_verifier=lambda _path, _manifest: None,
            output_verifier=lambda _path, _manifest: None,
            row_verifier=corrupt_then_compare,
        )
    assert corrupted
    assert not output.exists()
    assert not list(tmp_path.glob(".compact.reap-*"))


def test_existing_even_empty_output_is_rejected(tmp_path):
    package, selection_path, _ = _package(tmp_path)
    output = tmp_path / "compact"
    output.mkdir()
    with pytest.raises(IQKReapError, match="output must not exist"):
        compact_iqk_package(package, selection_path, output)
