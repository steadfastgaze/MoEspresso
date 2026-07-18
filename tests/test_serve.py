"""Manifest-gated serve adapter: the gate is pure, tested with a fake backend.

The guarantee is "verify against the manifest before building a model". This is
tested with an injected fake build_fn (no mlx/jang needed here): a faithful
package passes the gate and the backend is called; a tampered or incomplete
package is refused before the backend is ever touched.

The real jang/mlx_lm build + generate is exercised through the CLI; that is not
unit-tested here (it needs a real model), but the gate that protects it is.
"""

from __future__ import annotations

import json
import struct

from moespresso.core.artifact import make_artifact, write_artifact
from moespresso.optimize.allocate import AFFINE_BITS, EXPERT_BITS
from moespresso.optimize.decide import decide
from moespresso.package.manifest import build_package_manifest, file_identity, located_key
from moespresso.package.plan import package_plan_from_decision
from moespresso.runtime.serve import (
    MANIFEST_NAME,
    build_manifest_runtime,
    load_served_model,
)

SUBJECT = {"source_root": "toy", "source_format": "hf_safetensors"}
PRODUCER = {"tool": "test", "version": "0"}
ARCH = {"model_type": "qwen3_moe",
        "text_config": {"num_hidden_layers": 1, "num_experts": 8,
                        "layer_types": ["full_attention"]}}
SHARD = "model-00001-of-00001.safetensors"


def _affine_unit(name, role, layer_index=0):
    q = {f"{b}_{gs}": 0.99 for b in AFFINE_BITS for gs in (128, 64, 32)}
    return {"source_name": name, "kind": "affine", "role": role,
            "layer_index": layer_index, "shape": [64, 128], "importance": 1.0,
            "imatrix_mapped": True, "quality": q}


def _expert_unit(name, layer, projection):
    q = {str(b): 0.9 + 0.02 * b for b in EXPERT_BITS}
    return {"source_name": name, "kind": "expert", "role": f"moe.expert.{projection}",
            "layer_index": layer, "projection": projection, "n_experts": 8,
            "sampled": 2, "shape": [64, 128], "importance": 1.0,
            "imatrix_mapped": True, "quality": q}


def _decision():
    units = [
        _affine_unit("model.language_model.layers.0.self_attn.q_proj.weight", "attn.q_proj"),
        _expert_unit("model.language_model.layers.0.mlp.experts.gate_up_proj", 0, "gate"),
        _expert_unit("model.language_model.layers.0.mlp.experts.gate_up_proj", 0, "up"),
    ]
    ev = make_artifact("probe_evidence", SUBJECT, PRODUCER, status="valid", units=units)
    plan, _summary = package_plan_from_decision(decide(ev, target_quality=0.5))
    return plan


def _write_shard_matching(tmp_path, decision):
    """Write a shard whose keys match exactly what the manifest will declare."""
    tensors = {}
    for a in decision["allocation"]:
        if a["kind"] == "affine":
            base = a["source_name"]
            tensors[f"{base}.weight"] = b"\x00" * 16
            tensors[f"{base}.scales"] = b"\x00" * 8
            tensors[f"{base}.biases"] = b"\x00" * 8
        elif a["kind"] == "expert":
            base = f"{a['source_name']}.{a['projection']}"
            tensors[f"{base}.tq_packed"] = b"\x00" * 16
            tensors[f"{base}.tq_norms"] = b"\x00" * 8
            tensors[f"{base}.tq_bits"] = b"\x00" * 1
    header, blob, off = {}, bytearray(), 0
    for k, b in tensors.items():
        header[k] = {"dtype": "U8", "shape": [len(b)], "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hjson = json.dumps(header).encode()
    path = tmp_path / SHARD
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)
    return path


def _packaged(tmp_path):
    """Write a faithful package (shard + manifest.json) and return its manifest."""
    dec = _decision()
    located = {}
    for a in dec["allocation"]:
        prefix = (f"{a['source_name']}.{a['projection']}"
                  if a["kind"] == "expert" else a["source_name"])
        located[located_key(a)] = {"shard": SHARD, "key_prefix": prefix}
    path = _write_shard_matching(tmp_path, dec)
    man = build_package_manifest(dec, ARCH, located, [file_identity(path)])
    write_artifact(tmp_path / MANIFEST_NAME, man)
    return man


def test_faithful_package_builds_and_calls_backend(tmp_path):
    _packaged(tmp_path)
    calls = []

    def fake_build(manifest, pkg_dir):
        calls.append((manifest["artifact_kind"], pkg_dir))
        return ("MODEL", "TOK")

    model, tok, man = load_served_model(tmp_path, build_fn=fake_build)
    assert model == "MODEL" and tok == "TOK"
    assert man["artifact_kind"] == "package_manifest"
    assert calls == [("package_manifest", tmp_path)]  # backend got (manifest, dir)


def test_fast_path_does_not_verify_by_default(tmp_path):
    # The hot path must not hash 16GB on every launch. A tampered shard is served
    # by default (the engine trusts the declared contract); integrity is opt-in.
    _packaged(tmp_path)
    shard = tmp_path / SHARD
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF
    shard.write_bytes(bytes(data))
    model, _, _ = load_served_model(tmp_path, build_fn=lambda m, p: ("M", "T"))
    assert model == "M"  # built without a sha256 gate


def test_verify_package_flags_tampered_shard(tmp_path):
    # moespresso-verify / verify_package owns the convert-output integrity gate.
    # The serve hot path omits it. A flipped byte -> sha256 mismatch -> blocking issue.
    from moespresso.runtime.verify import verify_package
    man = _packaged(tmp_path)
    shard = tmp_path / SHARD
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF
    shard.write_bytes(bytes(data))
    blocking = [v for v in verify_package(man, tmp_path) if v.blocking]
    assert any(v.code == "runtime.sha256_mismatch" for v in blocking)


def test_verify_package_flags_missing_shard(tmp_path):
    from moespresso.runtime.verify import verify_package
    man = _packaged(tmp_path)
    (tmp_path / SHARD).unlink()
    blocking = [v for v in verify_package(man, tmp_path) if v.blocking]
    assert any(v.code == "runtime.missing_file" for v in blocking)


def test_reads_manifest_from_dir_by_default(tmp_path):
    man = _packaged(tmp_path)
    # no manifest= passed -> it must read package_manifest.json (hash-verified)
    _, _, loaded_man = load_served_model(tmp_path, build_fn=lambda m, p: ("M", "T"))
    assert loaded_man["artifact_id"] == man["artifact_id"]


def test_manifest_runtime_uses_ssd_streaming_for_moe_tq(tmp_path):
    calls = []
    manifest = {
        "architecture": {"family": "qwen3_5_moe"},
        "required_ops": ["affine_dequant", "tq_dequant"],
    }

    def streaming_builder(package_dir):
        calls.append(("streaming", package_dir))
        return "M", "T", 3

    assert build_manifest_runtime(
        manifest,
        tmp_path,
        resident_builder=lambda _m, _p: ("BAD", "BAD"),
        streaming_builder=streaming_builder,
    ) == ("M", "T")
    assert calls == [("streaming", tmp_path)]


def test_manifest_runtime_uses_ssd_streaming_for_qwen_kquant_moe(tmp_path):
    calls = []
    manifest = {
        "architecture": {"family": "qwen3_5_moe"},
        "required_ops": ["f32_passthrough", "kquant_dequant"],
    }

    def streaming_builder(package_dir):
        calls.append(("streaming", package_dir))
        return "M", "T", 40

    assert build_manifest_runtime(
        manifest,
        tmp_path,
        resident_builder=lambda _m, _p: ("BAD", "BAD"),
        streaming_builder=streaming_builder,
    ) == ("M", "T")
    assert calls == [("streaming", tmp_path)]


def test_manifest_runtime_keeps_dense_on_resident_path(tmp_path):
    calls = []
    manifest = {
        "architecture": {"family": "qwen3_5_dense"},
        "required_ops": ["affine_dequant"],
    }

    def resident_builder(manifest_arg, package_dir):
        calls.append((manifest_arg["architecture"]["family"], package_dir))
        return "DENSE", "TOK"

    assert build_manifest_runtime(
        manifest,
        tmp_path,
        resident_builder=resident_builder,
        streaming_builder=lambda _p: ("BAD", "BAD", 0),
    ) == ("DENSE", "TOK")
    assert calls == [("qwen3_5_dense", tmp_path)]


def test_manifest_runtime_keeps_deepseek_v4_off_qwen_streaming_path(tmp_path):
    calls = []
    manifest = {
        "architecture": {"family": "deepseek_v4_flash"},
        "required_ops": [
            "affine_dequant",
            "kquant_dequant",
            "tq_dequant",
            "fp16_passthrough",
            "raw_dtype_passthrough",
        ],
    }

    def resident_builder(manifest_arg, package_dir):
        calls.append((manifest_arg["architecture"]["family"], package_dir))
        return "DS4", "TOK"

    assert build_manifest_runtime(
        manifest,
        tmp_path,
        resident_builder=resident_builder,
        streaming_builder=lambda _p: ("BAD", "BAD", 0),
    ) == ("DS4", "TOK")
    assert calls == [("deepseek_v4_flash", tmp_path)]


def test_serve_module_imports_without_loading_runtime_dependencies():
    # The gate is pure; importing the module must not require mlx/jang/mlx_lm.
    import importlib
    import moespresso.runtime.serve as s
    importlib.reload(s)
    assert hasattr(s, "load_served_model") and hasattr(s, "generate_once")


def test_load_served_model_prints_runtime_truth_line(tmp_path, capsys):
    """The user must never guess which runtime they got: one honest
    line at load states package/capacity/hotlist/decode-path."""
    from moespresso.runtime.serve import load_served_model

    manifest = {"artifact_id": "pkg:abcdef1234567890aa", "subject": {}}

    class _M:
        _moespresso_ssd_streaming_capacity = 85
        _moespresso_ssd_hotlist = {"source": "package", "seeded": 4080}

    load_served_model(tmp_path, manifest=manifest,
                      build_fn=lambda m, p: (_M(), object()))
    out = capsys.readouterr().out
    assert "runtime=ssd-streaming" in out
    assert "capacity=85" in out and "hotlist=package" in out
    assert "lookahead=off" in out


def test_load_served_model_missing_dir_is_a_clear_error(tmp_path):
    import pytest

    from moespresso.runtime.serve import PackageNotFoundError, load_served_model

    with pytest.raises(PackageNotFoundError, match="not found"):
        load_served_model(tmp_path / "no-such-package")
    # a dir that exists but is not a package names the missing manifest
    (tmp_path / "stuff").mkdir()
    with pytest.raises(PackageNotFoundError, match="package_manifest.json"):
        load_served_model(tmp_path / "stuff")
