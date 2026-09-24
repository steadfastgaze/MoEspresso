"""Qwen GGUF K-quant package builder orchestration tests."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from moespresso.core.artifact import make_artifact
from moespresso.package.bundle import (
    METADATA_KEY,
    component_array,
    decode_bundle_metadata,
)
from moespresso.package.kquant_backend import KQuantEncodedWeight
from moespresso.package.kquant_format import KQUANT_GEOMETRY
from moespresso.package.kquant_recipe import KQuantRecipeError
from moespresso.package.qwen import kquant_package as qkp
from moespresso.probe.gguf_parse import GGUF_MAGIC


def _config():
    return {
        "model_type": "qwen3_moe",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 1,
            "hidden_size": 256,
            "num_experts": 2,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 256,
            "layer_types": ["full_attention"],
            "vocab_size": 512,
        },
    }


def _inventory():
    return {
        "artifact_kind": "source_inventory",
        "subject": {
            "source_root": "synthetic",
            "source_format": "hf_safetensors",
        },
        "tensors": [
            {
                "source_name": "model.language_model.embed_tokens.weight",
                "role": "embed_tokens",
                "kind": "affine",
                "layer_index": None,
                "shape": [512, 256],
                "status": "required",
            },
            {
                "source_name": "model.language_model.layers.0.mlp.experts.gate_up_proj",
                "role": "moe.expert.gate_up",
                "kind": "expert",
                "layer_index": 0,
                "projection": "gate_up",
                "shape": [2, 512, 256],
                "status": "required",
            },
            {
                "source_name": "model.language_model.layers.0.mlp.experts.down_proj",
                "role": "moe.expert.down",
                "kind": "expert",
                "layer_index": 0,
                "projection": "down",
                "shape": [2, 256, 256],
                "status": "required",
            },
            {
                "source_name": "model.language_model.norm.weight",
                "role": "final_norm",
                "kind": "affine",
                "layer_index": None,
                "shape": [256],
                "status": "required",
            },
        ],
    }


def _recipe():
    return {
        "token_embd.weight": "q6_k",
        "blk.0.ffn_gate_exps.weight": "iq2_xxs",
        "blk.0.ffn_up_exps.weight": "iq2_xxs",
        "blk.0.ffn_down_exps.weight": "iq2_s",
    }


def _tensor_types():
    return {
        "token_embd.weight": "Q6_K",
        "blk.0.ffn_gate_exps.weight": "IQ2_XXS",
        "blk.0.ffn_up_exps.weight": "IQ2_XXS",
        "blk.0.ffn_down_exps.weight": "IQ2_S",
        "output_norm.weight": "F32",
    }


def _imatrix():
    return {
        "token_embd.weight": np.ones(256, dtype=np.float32),
        "blk.0.ffn_gate_exps.weight": np.ones(256, dtype=np.float32),
        "blk.0.ffn_up_exps.weight": np.ones(256, dtype=np.float32),
        "blk.0.ffn_down_exps.weight": np.ones(256, dtype=np.float32),
    }


def _patch_recipe_inputs(monkeypatch, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(_config()))
    gguf = tmp_path / "recipe.gguf"
    gguf.write_bytes(b"fake")
    imatrix = tmp_path / "imatrix.dat"
    imatrix.write_bytes(b"fake")

    monkeypatch.setattr(qkp, "build_inventory", lambda *_args, **_kwargs: _inventory())
    monkeypatch.setattr(
        qkp,
        "imatrix_calibration",
        lambda _path: (_imatrix(), {"name": "imatrix.dat", "sha256": "abc", "key_count": 3}),
    )
    monkeypatch.setattr(qkp, "read_gguf_kquant_recipe", lambda _path: _recipe())
    monkeypatch.setattr(qkp, "read_gguf_tensor_types", lambda _path: _tensor_types())
    return src, gguf, imatrix


def test_preflight_qwen_kquant_package_maps_recipe_and_f32(tmp_path, monkeypatch):
    src, gguf, imatrix = _patch_recipe_inputs(monkeypatch, tmp_path)

    report = qkp.preflight_qwen_kquant_package(
        src,
        gguf_recipe_path=gguf,
        imatrix_path=imatrix,
    )

    assert report["status"] == "valid"
    assert report["recipe"]["dense_targets"] == 1
    assert report["recipe"]["expert_targets"] == 3
    assert report["recipe"]["dense_codec_counts"] == {"q6_k": 1}
    assert report["recipe"]["expert_codec_counts"] == {
        "iq2_s": 1,
        "iq2_xxs": 2,
    }
    assert report["recipe"]["f32_passthrough"] == 1
    assert report["fit"]["status"] == "valid"


def test_build_qwen_kquant_package_writes_manifest_sidecars_and_report(
    tmp_path,
    monkeypatch,
):
    src, gguf, imatrix = _patch_recipe_inputs(monkeypatch, tmp_path)
    out = tmp_path / "out"
    captured = {}

    def fake_copy_tokenizer(model_dir, out_dir, *, family):
        out_dir.mkdir(parents=True, exist_ok=True)
        captured["tokenizer_family"] = family
        return {"files": [], "has_tokenizer": False}

    def fake_write_package(package_plan, model_dir, config, out_dir, **kwargs):
        captured["package_plan"] = package_plan
        captured["model_dir"] = model_dir
        captured["config"] = config
        captured["passthrough"] = kwargs["passthrough"]
        captured["imatrix_keys"] = sorted(kwargs["kquant_imatrix_vectors"])
        captured["cache_context"] = kwargs["kquant_cache_context"]
        return make_artifact(
            "package_manifest",
            {"source_root": str(model_dir), "source_format": "hf_safetensors"},
            {"tool": "test", "version": "0"},
            status="valid",
            architecture={
                "family": "qwen3_5_moe",
                "config": config["text_config"],
            },
            tensors=[],
            files=[],
            required_ops=["kquant_dequant", "f32_passthrough"],
            provenance={},
        )

    monkeypatch.setattr(
        "moespresso.package.tokenizer.copy_tokenizer_into_package",
        fake_copy_tokenizer,
    )
    monkeypatch.setattr(qkp, "write_package", fake_write_package)

    manifest = qkp.build_qwen_kquant_package(
        src,
        out,
        gguf_recipe_path=gguf,
        imatrix_path=imatrix,
        kquant_encoder=lambda *_args: None,
        kquant_cache_dir=tmp_path / "cache",
    )

    assert manifest["status"] == "valid"
    assert captured["tokenizer_family"] == "qwen3_5_moe"
    assert captured["package_plan"]["artifact_kind"] == "package_plan"
    assert captured["package_plan"]["subject"] == {
        "source_root": "synthetic",
        "source_format": "hf_safetensors",
    }
    assert str(src) not in json.dumps(captured["package_plan"]["subject"])
    assert captured["package_plan"]["achieved"]["dense_codec_counts"] == {"q6_k": 1}
    assert captured["package_plan"]["achieved"]["expert_codec_counts"] == {
        "iq2_s": 1,
        "iq2_xxs": 2,
    }
    assert [row["format"] for row in captured["passthrough"]] == ["f32_passthrough"]
    assert captured["imatrix_keys"] == [
        "blk.0.ffn_down_exps.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "token_embd.weight",
    ]
    assert captured["cache_context"]["recipe_kind"] == "qwen_gguf_recipe"
    assert (out / qkp.PACKAGE_PLAN_NAME).exists()
    assert (out / "package_manifest.json").exists()
    assert (out / "config.json").exists()
    assert (out / "jang_config.json").exists()
    report = json.loads((out / qkp.KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["recipe"]["f32_passthrough"] == 1
    assert report["kquant_cache"]["enabled"] is True
    assert "path" not in report["kquant_cache"]
    assert str(tmp_path / "cache") not in (out / qkp.KQUANT_RECIPE_REPORT_NAME).read_text()


def test_build_qwen_kquant_package_registers_the_agentic_profile(
    tmp_path,
    monkeypatch,
):
    # The builder writes the agentic profile sidecar beside the vendored
    # template and registers its identity in the manifest.
    from moespresso.package.agentic_profile import (
        AGENTIC_PROFILE_NAME,
        profile_for_family,
    )

    src, gguf, imatrix = _patch_recipe_inputs(monkeypatch, tmp_path)
    out = tmp_path / "out"

    def fake_copy_tokenizer(model_dir, out_dir, *, family):
        out_dir.mkdir(parents=True, exist_ok=True)
        return {"files": [], "has_tokenizer": False}

    def fake_write_package(package_plan, model_dir, config, out_dir, **kwargs):
        # Mirror the real writer: the agentic_profile block enters the
        # manifest payload before the content hash is taken.
        return make_artifact(
            "package_manifest",
            {"source_root": str(model_dir), "source_format": "hf_safetensors"},
            {"tool": "test", "version": "0"},
            status="valid",
            architecture={
                "family": "qwen3_5_moe",
                "config": config["text_config"],
            },
            tensors=[],
            files=[],
            required_ops=["kquant_dequant", "f32_passthrough"],
            provenance={},
            agentic_profile=kwargs["agentic_profile"],
        )

    monkeypatch.setattr(
        "moespresso.package.tokenizer.copy_tokenizer_into_package",
        fake_copy_tokenizer,
    )
    monkeypatch.setattr(qkp, "write_package", fake_write_package)

    qkp.build_qwen_kquant_package(
        src,
        out,
        gguf_recipe_path=gguf,
        imatrix_path=imatrix,
        kquant_encoder=lambda *_args: None,
    )

    profile_path = out / AGENTIC_PROFILE_NAME
    assert profile_path.is_file()
    assert json.loads(profile_path.read_text()) == profile_for_family(
        "qwen3_5_moe")
    manifest = json.loads((out / "package_manifest.json").read_text())
    block = manifest["agentic_profile"]
    assert block["path"] == AGENTIC_PROFILE_NAME
    assert block["family"] == "qwen3_5_moe"
    assert block["size_bytes"] == profile_path.stat().st_size


def test_build_qwen_kquant_package_emits_hotlist_and_survives_misalignment(
    tmp_path,
    monkeypatch,
    capsys,
):
    src, gguf, imatrix = _patch_recipe_inputs(monkeypatch, tmp_path)

    def fake_copy_tokenizer(model_dir, out_dir, *, family):
        out_dir.mkdir(parents=True, exist_ok=True)
        return {"files": [], "has_tokenizer": False}

    def fake_write_package(package_plan, model_dir, config, out_dir, **kwargs):
        return make_artifact(
            "package_manifest",
            {"source_root": str(model_dir), "source_format": "hf_safetensors"},
            {"tool": "test", "version": "0"},
            status="valid",
            architecture={
                "family": "qwen3_5_moe",
                "config": config["text_config"],
            },
            tensors=[],
            files=[],
            required_ops=["kquant_dequant", "f32_passthrough"],
            provenance={},
        )

    monkeypatch.setattr(
        "moespresso.package.tokenizer.copy_tokenizer_into_package",
        fake_copy_tokenizer,
    )
    monkeypatch.setattr(qkp, "write_package", fake_write_package)

    import moespresso.package.hotlist as hl

    seen = {}

    def fake_hotlist(out_dir, imatrix_path, imatrix_identity=None):
        seen["args"] = (out_dir, imatrix_path, imatrix_identity)
        return 3

    monkeypatch.setattr(hl, "write_package_expert_hotlist", fake_hotlist)
    out = tmp_path / "out"
    qkp.build_qwen_kquant_package(
        src,
        out,
        gguf_recipe_path=gguf,
        imatrix_path=imatrix,
        kquant_encoder=lambda *_args: None,
    )
    assert seen["args"] == (
        out, imatrix, {"name": "imatrix.dat", "sha256": "abc", "key_count": 3})
    report = json.loads((out / qkp.KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["expert_hotlist_layers"] == 3

    # An alignment failure skips the artifact loudly and the build completes.
    def raising(*_args, **_kwargs):
        raise hl.HotlistAlignmentError("layer sets differ")

    monkeypatch.setattr(hl, "write_package_expert_hotlist", raising)
    out2 = tmp_path / "out2"
    qkp.build_qwen_kquant_package(
        src,
        out2,
        gguf_recipe_path=gguf,
        imatrix_path=imatrix,
        kquant_encoder=lambda *_args: None,
    )
    assert "[hotlist] SKIPPED" in capsys.readouterr().out
    report2 = json.loads((out2 / qkp.KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report2["expert_hotlist_layers"] == 0


def test_qwen_kquant_force_dry_run_reports_matched_tensors(tmp_path, monkeypatch):
    src, gguf, imatrix = _patch_recipe_inputs(monkeypatch, tmp_path)
    out = tmp_path / "out"

    plan = qkp.build_qwen_kquant_package(
        src,
        out,
        gguf_recipe_path=gguf,
        imatrix_path=imatrix,
        force_format=["*ffn_gate_exps.weight=kquant:q4_k"],
        force_format_dry_run=True,
    )

    gate = next(
        row for row in plan["allocation"]
        if row.get("gguf_tensor") == "blk.0.ffn_gate_exps.weight"
    )
    assert gate["format"] == "kquant"
    assert "forced_format" not in gate
    assert plan["force_override_preview"]["matched"][0]["before"] == "kquant:iq2_xxs"
    assert plan["force_override_preview"]["matched"][0]["after"] == "kquant:q4_k"
    report = json.loads((out / qkp.KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["matched"][0]["gguf_tensor"] == "blk.0.ffn_gate_exps.weight"
    assert report["matched"][0]["before"] == "kquant:iq2_xxs"
    assert report["matched"][0]["after"] == "kquant:q4_k"


def test_qwen_kquant_source_identity_overrides_inventory_subject(tmp_path, monkeypatch):
    src, gguf, imatrix = _patch_recipe_inputs(monkeypatch, tmp_path)
    identity = (
        "deepreinforce-ai/Ornith-1.0-35B@"
        "5df2ed3f675c7beaa490328cc70bb573b65fb660"
    )

    plan = qkp.build_qwen_kquant_package(
        src,
        tmp_path / "out",
        gguf_recipe_path=gguf,
        imatrix_path=imatrix,
        source_identity=identity,
        force_format_dry_run=True,
    )

    assert plan["subject"] == {
        "source_root": identity,
        "source_format": "hf_safetensors",
    }


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        data = a.tobytes()
        header[name] = {
            "dtype": "F32",
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


def _gguf_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_payload_gguf(path, tensors):
    header = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), 0)
    infos = bytearray()
    offset = 0
    payload = bytearray()
    for name, type_id, dims, data in tensors:
        infos += _gguf_string(name)
        infos += struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<I", type_id)
        infos += struct.pack("<Q", offset)
        raw = np.asarray(data, dtype=np.uint8).tobytes()
        payload += raw
        offset += len(raw)
    metadata = header + bytes(infos)
    pad = b"\0" * ((((len(metadata) + 31) // 32) * 32) - len(metadata))
    path.write_bytes(metadata + pad + bytes(payload))


def _byte_copy_recipe():
    return {
        "token_embd.weight": "q6_k",
        "blk.0.ffn_gate_exps.weight": "iq2_xxs",
        "blk.0.ffn_up_exps.weight": "iq2_xxs",
        "blk.0.ffn_down_exps.weight": "q2_k",
    }


def _byte_copy_tensor_types():
    return {
        "token_embd.weight": "Q6_K",
        "blk.0.ffn_gate_exps.weight": "IQ2_XXS",
        "blk.0.ffn_up_exps.weight": "IQ2_XXS",
        "blk.0.ffn_down_exps.weight": "Q2_K",
        "output_norm.weight": "F32",
    }


def _write_byte_copy_payload_gguf(path):
    # Expert payload values are distinct per tensor and per expert so the test
    # can prove which GGUF row landed in which bundle slot. GGUF dims are
    # [in_features, out_features, experts]; iq2_xxs packs 256 weights into 66
    # bytes and q2_k packs 256 weights into 84 bytes.
    _write_payload_gguf(path, [
        ("blk.0.ffn_gate_exps.weight", 16, [256, 256, 2], np.concatenate([
            np.full(256 * 66, 21, dtype=np.uint8),
            np.full(256 * 66, 31, dtype=np.uint8),
        ])),
        ("blk.0.ffn_up_exps.weight", 16, [256, 256, 2], np.concatenate([
            np.full(256 * 66, 22, dtype=np.uint8),
            np.full(256 * 66, 32, dtype=np.uint8),
        ])),
        ("blk.0.ffn_down_exps.weight", 10, [256, 256, 2], np.concatenate([
            np.full(256 * 84, 23, dtype=np.uint8),
            np.full(256 * 84, 33, dtype=np.uint8),
        ])),
    ])


def _byte_copy_setup(tmp_path, monkeypatch):
    """Real source tensors and payload GGUF; recipe/imatrix parsing mocked."""
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    _write_safetensors(src / "model-00001.safetensors", {
        "model.language_model.embed_tokens.weight":
            rng.standard_normal((512, 256)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.gate_up_proj":
            rng.standard_normal((2, 512, 256)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.down_proj":
            rng.standard_normal((2, 256, 256)).astype(np.float32),
        "model.language_model.norm.weight": np.ones(256, dtype=np.float32),
    })
    (src / "config.json").write_text(json.dumps(_config()))
    gguf = tmp_path / "recipe.gguf"
    _write_byte_copy_payload_gguf(gguf)
    imatrix = tmp_path / "imatrix.dat"
    imatrix.write_bytes(b"fake")

    monkeypatch.setattr(qkp, "build_inventory", lambda *_args, **_kwargs: _inventory())
    monkeypatch.setattr(
        qkp,
        "imatrix_calibration",
        lambda _path: (_imatrix(), {"name": "imatrix.dat", "sha256": "abc", "key_count": 4}),
    )
    monkeypatch.setattr(qkp, "read_gguf_kquant_recipe", lambda _path: _byte_copy_recipe())
    monkeypatch.setattr(qkp, "read_gguf_tensor_types", lambda _path: _byte_copy_tensor_types())
    return src, gguf, imatrix, tmp_path / "out"


def _fake_wire_encoder(calls):
    def encode(weight, target, _imatrix_vectors):
        calls.append(target.codec)
        geometry = KQUANT_GEOMETRY[target.codec]
        blocks = (
            weight.shape[1] + geometry.weights_per_block - 1
        ) // geometry.weights_per_block
        return KQuantEncodedWeight(
            codec=target.codec,
            weight=np.full(
                (weight.shape[0], blocks * geometry.bytes_per_block),
                99,
                dtype=np.uint8,
            ),
            scales=np.zeros((1,), dtype=np.uint8),
        )
    return encode


def _load_layer_bundle(out_dir, manifest):
    from safetensors import safe_open

    bundle_key = "language_model.model.layers.0.mlp.switch_mlp.experts.tq_bundle"
    for file in manifest["files"]:
        with safe_open(str(out_dir / file["path"]), framework="np") as f:
            if bundle_key not in f.keys():
                continue
            bundle = np.asarray(f.get_tensor(bundle_key))
            geometry = decode_bundle_metadata(f.metadata()[METADATA_KEY])[0]
            return bundle, geometry
    raise AssertionError(f"no shard contains {bundle_key}")


def test_build_qwen_kquant_package_can_copy_gguf_expert_bytes(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    src, gguf, imatrix, out = _byte_copy_setup(tmp_path, monkeypatch)
    calls = []

    manifest = qkp.build_qwen_kquant_package(
        src,
        out,
        gguf_recipe_path=gguf,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=_fake_wire_encoder(calls),
        copy_gguf_expert_bytes=True,
    )

    assert manifest["status"] == "valid"
    assert calls == ["q6_k"]
    report = json.loads((out / qkp.KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["recipe"]["expert_byte_source"]["mode"] == "gguf_bytes"
    assert report["recipe"]["expert_byte_source"]["name"] == gguf.name

    bundle, geometry = _load_layer_bundle(out, manifest)
    assert geometry["projections"]["gate_proj"]["kquant_codec"] == "iq2_xxs"
    assert geometry["projections"]["down_proj"]["kquant_codec"] == "q2_k"
    gate = component_array(bundle, geometry["projections"]["gate_proj"]["weight"])
    up = component_array(bundle, geometry["projections"]["up_proj"]["weight"])
    down = component_array(bundle, geometry["projections"]["down_proj"]["weight"])
    np.testing.assert_array_equal(gate[0], np.full((256, 66), 21, dtype=np.uint8))
    np.testing.assert_array_equal(gate[1], np.full((256, 66), 31, dtype=np.uint8))
    np.testing.assert_array_equal(up[0], np.full((256, 66), 22, dtype=np.uint8))
    np.testing.assert_array_equal(up[1], np.full((256, 66), 32, dtype=np.uint8))
    np.testing.assert_array_equal(down[0], np.full((256, 84), 23, dtype=np.uint8))
    np.testing.assert_array_equal(down[1], np.full((256, 84), 33, dtype=np.uint8))


def test_build_qwen_kquant_package_reuses_one_gguf_expert_reader(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    src, gguf, imatrix, out = _byte_copy_setup(tmp_path, monkeypatch)
    init_paths = []

    class FakeReader:
        def __init__(self, path):
            init_paths.append(path)

        def load_expert_weight(self, target, *, expert_index):
            geometry = KQUANT_GEOMETRY[target.codec]
            return KQuantEncodedWeight(
                codec=target.codec,
                weight=np.full((4, geometry.bytes_per_block), 7, dtype=np.uint8),
                scales=np.zeros((1,), dtype=np.uint8),
            )

    monkeypatch.setattr(qkp, "GGUFKQuantExpertReader", FakeReader)

    qkp.build_qwen_kquant_package(
        src,
        out,
        gguf_recipe_path=gguf,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=_fake_wire_encoder([]),
        copy_gguf_expert_bytes=True,
    )

    assert init_paths == [gguf]


def test_build_qwen_kquant_package_rejects_remote_gguf_byte_copy(tmp_path):
    with pytest.raises(KQuantRecipeError, match="local GGUF file"):
        qkp.build_qwen_kquant_package(
            tmp_path / "src",
            tmp_path / "out",
            gguf_recipe_path="https://example.com/recipe.gguf",
            imatrix_path=tmp_path / "imatrix.dat",
            kquant_encoder=lambda *_args: pytest.fail("encoder should not run"),
            copy_gguf_expert_bytes=True,
        )


def test_main_rejects_remote_gguf_byte_copy(tmp_path, capsys):
    rc = qkp.main([
        str(tmp_path / "src"),
        str(tmp_path / "out"),
        "--gguf-recipe", "https://example.com/recipe.gguf",
        "--imatrix", str(tmp_path / "imatrix.dat"),
        "--copy-gguf-expert-bytes",
    ])

    assert rc == 2
    assert "local GGUF file" in capsys.readouterr().out


def test_main_forwards_source_identity(tmp_path, monkeypatch):
    captured = {}

    def fake_build(model_dir, out_dir, **kwargs):
        captured["model_dir"] = model_dir
        captured["out_dir"] = out_dir
        captured["source_identity"] = kwargs["source_identity"]
        return {
            "artifact_kind": "package_manifest",
            "artifact_id": "pkg:test",
            "files": [],
            "tensors": [],
        }

    monkeypatch.setattr(qkp, "build_qwen_kquant_package", fake_build)
    identity = (
        "deepreinforce-ai/Ornith-1.0-35B@"
        "5df2ed3f675c7beaa490328cc70bb573b65fb660"
    )

    rc = qkp.main([
        str(tmp_path / "src"),
        str(tmp_path / "out"),
        "--gguf-recipe", str(tmp_path / "recipe.gguf"),
        "--imatrix", str(tmp_path / "imatrix.dat"),
        "--source-identity", identity,
    ])

    assert rc == 0
    assert captured["source_identity"] == identity
