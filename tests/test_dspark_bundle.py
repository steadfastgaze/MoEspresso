"""DSpark drafter bundling into a DeepSeek-V4 package.

Builds a tiny real DS4-family package plus a fabricated valid sidecar, then
pins the bundle contract: hard-linked files, the declared optional drafter
component with its identities and provenance, verification coverage of the
drafter bytes, the separability of the component (deleting its files leaves
a verify-clean drafter-off package), and every refusal mode.
"""

from __future__ import annotations

import hashlib
import json
import struct

import pytest

from moespresso.core.artifact import (
    compute_artifact_id,
    make_artifact,
    read_artifact,
    write_artifact,
)
from moespresso.optimize.allocate import AFFINE_BITS
from moespresso.optimize.decide import decide
from moespresso.package.deepseek_v4.dspark_bundle import (
    BUNDLE_REPORT_NAME,
    DSparkBundleError,
    SIDECAR_KIND,
    SIDECAR_MANIFEST_NAME,
    SIDECAR_SCHEMA_MAJOR,
    bundle_dspark_drafter,
    read_valid_sidecar_manifest,
)
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_LEGACY_RELAYOUT,
)
from moespresso.package.manifest import (
    build_package_manifest,
    file_identity,
    located_key,
)
from moespresso.package.plan import package_plan_from_decision
from moespresso.runtime.verify import verify_package

SUBJECT = {"source_root": "toy", "source_format": "hf_safetensors"}
PRODUCER = {"tool": "test", "version": "0"}
DS4_ARCH = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "sliding_window": 128,
    "index_topk": 512,
    "vocab_size": 129280,
}
SHARD = "model-00001-of-00001.safetensors"
DRAFT_SHARD = "model-dspark-00001-of-00001.safetensors"


def _affine_unit(name, role):
    q = {f"{b}_{gs}": 0.99 for b in AFFINE_BITS for gs in (128, 64, 32)}
    return {
        "source_name": name, "kind": "affine", "role": role,
        "layer_index": 0, "shape": [64, 128], "importance": 1.0,
        "imatrix_mapped": True, "quality": q,
    }


def _tiny_ds4_package(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    ev = make_artifact(
        "probe_evidence", SUBJECT, PRODUCER, status="valid",
        units=[_affine_unit(
            "model.layers.0.self_attn.q_proj.weight", "attn.q_proj")])
    plan, _ = package_plan_from_decision(decide(ev, target_quality=0.5))
    located = {
        located_key(a): {"shard": SHARD, "key_prefix": a["source_name"]}
        for a in plan["allocation"]
    }
    tensors = {"layers.0.ffn.experts.tq_bundle": b"\x00" * 8}
    for a in plan["allocation"]:
        tensors[f"{a['source_name']}.weight"] = b"\x00" * 16
        tensors[f"{a['source_name']}.scales"] = b"\x00" * 8
        tensors[f"{a['source_name']}.biases"] = b"\x00" * 8
    header, blob, off = {}, bytearray(), 0
    for key, data in tensors.items():
        header[key] = {"dtype": "U8", "shape": [len(data)],
                       "data_offsets": [off, off + len(data)]}
        blob += data
        off += len(data)
    hjson = json.dumps(header).encode()
    with open(package / SHARD, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)
    manifest = build_package_manifest(
        plan, DS4_ARCH, located, [file_identity(package / SHARD)])
    assert manifest["architecture"]["family"] == "deepseek_v4_flash"
    from moespresso.core.artifact import write_artifact

    write_artifact(package / "package_manifest.json", manifest)
    return package, read_artifact(package / "package_manifest.json")


def _tiny_sidecar(tmp_path):
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    shard = sidecar / DRAFT_SHARD
    shard.write_bytes(b"\x07" * 64)
    payload = {
        "artifact_kind": SIDECAR_KIND,
        "schema_version": {"major": SIDECAR_SCHEMA_MAJOR, "minor": 0},
        "producer": PRODUCER,
        "subject": {"family": "deepseek_v4_flash_dspark"},
        "status": "valid",
        "dspark": {"block_size": 5, "n_mtp_layers": 3},
        "provenance": {
            "file_sha256": {
                DRAFT_SHARD: hashlib.sha256(shard.read_bytes()).hexdigest(),
            },
            "iqk_staging": {"member": "iq2_k"},
        },
        "tensors": {"blocks.0.x": {"format": "passthrough",
                                   "file": DRAFT_SHARD}},
    }
    payload["artifact_id"] = compute_artifact_id(payload)
    (sidecar / SIDECAR_MANIFEST_NAME).write_text(json.dumps(payload))
    return sidecar, payload


def test_restated_sidecar_constants_match_the_builder_module():
    from moespresso.package.deepseek_v4 import dspark_sidecar

    assert SIDECAR_MANIFEST_NAME == dspark_sidecar.SIDECAR_MANIFEST_NAME
    assert SIDECAR_KIND == dspark_sidecar.SIDECAR_KIND
    assert SIDECAR_SCHEMA_MAJOR == dspark_sidecar.SIDECAR_SCHEMA_MAJOR


def test_bundle_links_files_and_declares_the_component(tmp_path):
    package, manifest = _tiny_ds4_package(tmp_path)
    sidecar, payload = _tiny_sidecar(tmp_path)
    out = tmp_path / "bundled"

    report = bundle_dspark_drafter(package, sidecar, out)

    bundled = read_artifact(out / "package_manifest.json")
    component = bundled["drafter"]
    assert component["family"] == "dspark"
    assert component["optional"] is True
    assert component["manifest_path"] == SIDECAR_MANIFEST_NAME
    assert component["sidecar_artifact_id"] == payload["artifact_id"]
    assert component["experts_format"] == "iq2_k"
    assert component["provenance"]["source_package_manifest_id"] == (
        manifest["artifact_id"])
    assert [f["path"] for f in component["files"]] == sorted(
        [SIDECAR_MANIFEST_NAME, DRAFT_SHARD])
    assert bundled["artifact_id"] != manifest["artifact_id"]
    assert report["manifest_id"] == {
        "before": manifest["artifact_id"], "after": bundled["artifact_id"]}

    # Hard links: the model shard and the drafter shard share their source
    # inodes; the rewritten manifest does not.
    assert (out / SHARD).stat().st_ino == (package / SHARD).stat().st_ino
    assert (out / DRAFT_SHARD).stat().st_ino == (
        sidecar / DRAFT_SHARD).stat().st_ino
    assert (out / "package_manifest.json").stat().st_ino != (
        package / "package_manifest.json").stat().st_ino

    assert not any(v.blocking for v in verify_package(bundled, out))
    assert (out / BUNDLE_REPORT_NAME).is_file()
    assert report["bytes"]["total_with_drafter"] == (
        report["bytes"]["package_shards"] + report["bytes"]["drafter_files"])


def test_bundle_canonicalizes_a_legacy_sidecar_without_touching_its_shard(tmp_path):
    package, package_payload = _tiny_ds4_package(tmp_path)
    package_payload["tensors"].append({
        "source_name": "layers.0.ffn.experts.gate",
        "kind": "expert",
        "format": "iqk",
        "format_params": {
            "iqk_codec": "iq2_ks",
            "layout": IQK_LAYOUT_LEGACY_RELAYOUT,
        },
        "shard": SHARD,
        "key_prefix": "layers.0.ffn.experts",
    })
    package_payload["required_ops"].append("iqk_dequant")
    package_payload.pop("artifact_id")
    source_package_manifest_path = package / "package_manifest.json"
    write_artifact(source_package_manifest_path, package_payload)
    source_package_manifest = read_artifact(source_package_manifest_path)
    source_package_manifest_bytes = source_package_manifest_path.read_bytes()

    sidecar, payload = _tiny_sidecar(tmp_path)
    row = payload["tensors"]["blocks.0.x"]
    row["format"] = "iqk"
    row["layout"] = IQK_LAYOUT_LEGACY_RELAYOUT
    payload["artifact_id"] = compute_artifact_id(payload)
    source_manifest_path = sidecar / SIDECAR_MANIFEST_NAME
    source_manifest_path.write_text(json.dumps(payload))
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest_inode = source_manifest_path.stat().st_ino
    source_shard_identity = file_identity(sidecar / DRAFT_SHARD)

    out = tmp_path / "bundled"
    report = bundle_dspark_drafter(package, sidecar, out)

    bundled = read_artifact(out / "package_manifest.json")
    output_sidecar = read_valid_sidecar_manifest(out)
    component = bundled["drafter"]
    bundled_iqk = next(
        tensor for tensor in bundled["tensors"]
        if tensor.get("format") == "iqk")
    manifest_identity = next(
        entry for entry in component["files"]
        if entry["path"] == SIDECAR_MANIFEST_NAME)

    assert bundled_iqk["format_params"]["layout"] == IQK_LAYOUT_IQK_RELAYOUT
    assert read_artifact(source_package_manifest_path)["tensors"][-1][
        "format_params"]["layout"] == IQK_LAYOUT_LEGACY_RELAYOUT
    assert source_package_manifest_path.read_bytes() == source_package_manifest_bytes
    assert component["provenance"]["source_package_manifest_id"] == (
        source_package_manifest["artifact_id"])
    assert output_sidecar["tensors"]["blocks.0.x"]["layout"] == (
        IQK_LAYOUT_IQK_RELAYOUT)
    assert output_sidecar["artifact_id"] != payload["artifact_id"]
    assert component["sidecar_artifact_id"] == output_sidecar["artifact_id"]
    assert component["provenance"]["source_sidecar_artifact_id"] == (
        payload["artifact_id"])
    assert manifest_identity == file_identity(out / SIDECAR_MANIFEST_NAME)
    assert source_manifest_path.read_bytes() == source_manifest_bytes
    assert source_manifest_path.stat().st_ino == source_manifest_inode
    assert (out / SIDECAR_MANIFEST_NAME).stat().st_ino != source_manifest_inode
    assert file_identity(out / DRAFT_SHARD) == source_shard_identity
    assert (out / DRAFT_SHARD).stat().st_ino == (sidecar / DRAFT_SHARD).stat().st_ino
    assert report["placements"]["by_file"][SIDECAR_MANIFEST_NAME] == "rewrite"
    assert report["placements"]["rewritten"] == 1
    assert not any(v.blocking for v in verify_package(bundled, out))


def test_bundle_is_separable_into_a_clean_drafter_off_package(tmp_path):
    package, _ = _tiny_ds4_package(tmp_path)
    sidecar, _ = _tiny_sidecar(tmp_path)
    out = tmp_path / "bundled"
    bundle_dspark_drafter(package, sidecar, out)
    bundled = read_artifact(out / "package_manifest.json")

    (out / SIDECAR_MANIFEST_NAME).unlink()
    (out / DRAFT_SHARD).unlink()

    issues = verify_package(bundled, out)
    assert not any(v.blocking for v in issues)
    assert [v.code for v in issues] == ["runtime.drafter_component_absent"]


def test_bundle_refuses_a_package_with_a_drafter(tmp_path):
    package, _ = _tiny_ds4_package(tmp_path)
    sidecar, _ = _tiny_sidecar(tmp_path)
    first = tmp_path / "bundled"
    bundle_dspark_drafter(package, sidecar, first)
    with pytest.raises(DSparkBundleError, match="already declares"):
        bundle_dspark_drafter(first, sidecar, tmp_path / "again")


def test_bundle_refuses_non_empty_output_and_in_place(tmp_path):
    package, _ = _tiny_ds4_package(tmp_path)
    sidecar, _ = _tiny_sidecar(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stray").write_text("x")
    with pytest.raises(DSparkBundleError, match="not empty"):
        bundle_dspark_drafter(package, sidecar, occupied)
    with pytest.raises(DSparkBundleError, match="new directory"):
        bundle_dspark_drafter(package, sidecar, package)


def test_bundle_refuses_a_tampered_sidecar_shard(tmp_path):
    package, _ = _tiny_ds4_package(tmp_path)
    sidecar, _ = _tiny_sidecar(tmp_path)
    (sidecar / DRAFT_SHARD).write_bytes(b"\x08" * 64)
    with pytest.raises(DSparkBundleError, match="hash mismatch"):
        bundle_dspark_drafter(package, sidecar, tmp_path / "bundled")


def test_bundle_refuses_a_name_collision(tmp_path):
    package, _ = _tiny_ds4_package(tmp_path)
    sidecar, _ = _tiny_sidecar(tmp_path)
    (package / DRAFT_SHARD).write_bytes(b"\x09")
    with pytest.raises(DSparkBundleError, match="collide"):
        bundle_dspark_drafter(package, sidecar, tmp_path / "bundled")


def test_read_valid_sidecar_manifest_checks_the_contract(tmp_path):
    sidecar, payload = _tiny_sidecar(tmp_path)
    assert read_valid_sidecar_manifest(sidecar)["artifact_id"] == (
        payload["artifact_id"])

    wrong_kind = dict(payload)
    wrong_kind["artifact_kind"] = "banana"
    (sidecar / SIDECAR_MANIFEST_NAME).write_text(json.dumps(wrong_kind))
    with pytest.raises(DSparkBundleError, match="artifact kind"):
        read_valid_sidecar_manifest(sidecar)

    tampered = dict(payload)
    tampered["status"] = "draft"  # content changed, stored id kept
    (sidecar / SIDECAR_MANIFEST_NAME).write_text(json.dumps(tampered))
    with pytest.raises(DSparkBundleError, match="hash mismatch"):
        read_valid_sidecar_manifest(sidecar)
