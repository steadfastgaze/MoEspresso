"""package_manifest builder: pure, no mlx/jang. The 'engine never guesses' core.

Builds a real package_plan (via decide on a tiny probe_evidence) plus
synthetic written-file identities, then pins: every allocation becomes a tensor
entry with the right format, architecture facts are copied faithfully, file
identities are recorded, provenance chains to the plan, the manifest is a
deterministic content-addressed artifact, and a missing written file fails closed.
"""

from __future__ import annotations

import hashlib

from moespresso.core.artifact import compute_artifact_id, make_artifact, validate_base
from moespresso.inventory.architecture_profile import DEEPSEEK_V4_FLASH_COMPRESS_RATIOS
from moespresso.optimize.allocate import AFFINE_BITS, EXPERT_BITS
from moespresso.optimize.decide import decide
from moespresso.package.manifest import build_package_manifest, file_identity, located_key
from moespresso.package.plan import package_plan_from_decision

SUBJECT = {"source_root": "toy", "source_format": "hf_safetensors"}
PRODUCER = {"tool": "test", "version": "0"}

ARCH = {
    "model_type": "qwen3_moe",
    "text_config": {
        "num_hidden_layers": 2, "hidden_size": 2048, "num_experts": 256,
        "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        "layer_types": ["linear_attention", "full_attention"], "vocab_size": 1000,
    },
}

DENSE_ARCH = {
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "vision_config": {"model_type": "qwen3_5"},
    "mtp_num_hidden_layers": 1,
    "text_config": {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "tie_word_embeddings": True,
        "layer_types": [
            "linear_attention", "linear_attention",
            "linear_attention", "full_attention",
        ],
        "vocab_size": 1000,
    },
}

DS4_ARCH = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "sliding_window": 128,
    "index_topk": 512,
    "compress_rope_theta": 160000,
    "compress_ratios": list(DEEPSEEK_V4_FLASH_COMPRESS_RATIOS),
    "vocab_size": 129280,
}


def _affine_unit(name, role, q=0.99, layer_index=0, rows=64, cols=128):
    quality = {f"{b}_{gs}": q for b in AFFINE_BITS for gs in (128, 64, 32)}
    return {"source_name": name, "kind": "affine", "role": role,
            "layer_index": layer_index, "shape": [rows, cols], "importance": 1.0,
            "imatrix_mapped": True, "quality": quality}


def _expert_unit(name, layer, projection, rows=64, cols=128):
    quality = {str(b): 0.9 + 0.02 * b for b in EXPERT_BITS}
    return {"source_name": name, "kind": "expert", "role": f"moe.expert.{projection}",
            "layer_index": layer, "projection": projection, "n_experts": 8,
            "sampled": 2, "shape": [rows, cols], "importance": 1.0,
            "imatrix_mapped": True, "quality": quality}


def _evidence(units):
    return make_artifact("probe_evidence", SUBJECT, PRODUCER, status="valid", units=units)


def _plan(decision: dict) -> dict:
    plan, _summary = package_plan_from_decision(decision)
    return plan


def _decision():
    # A fused gate_up source produces two expert sub-projections that share a
    # source_name: exactly the case that needs projection-qualified locations.
    units = [
        _affine_unit("model.language_model.layers.1.self_attn.q_proj.weight", "attn.q_proj", layer_index=1),
        _affine_unit("model.language_model.layers.0.mlp.gate.weight", "moe.router_gate", q=0.4, layer_index=0),
        _expert_unit("model.language_model.layers.0.mlp.experts.gate_up_proj", 0, "gate"),
        _expert_unit("model.language_model.layers.0.mlp.experts.gate_up_proj", 0, "up"),
    ]
    return _plan(decide(_evidence(units), target_quality=0.5))


def _mxfp4_decision():
    dec = _decision()
    for alloc in dec["allocation"]:
        if alloc["kind"] == "expert":
            alloc["bits"] = 4
            alloc["codec"] = "mxfp4"
            alloc["format"] = "mxfp4"
            alloc["source_codec"] = "fp4_e2m1_ue8m0"
            alloc["lossless"] = True
    return dec


def _kquant_plan(codec="q2_k"):
    dec = _decision()
    for alloc in dec["allocation"]:
        if alloc["kind"] == "expert":
            alloc["bits"] = 2
            alloc["codec"] = codec
            alloc["format"] = "kquant"
            alloc["kquant_codec"] = codec
            alloc["imatrix_key"] = f"blk.{alloc['layer_index']}.ffn_{alloc['projection']}_exps.weight"
            alloc["module_path"] = (
                f"model.layers.{alloc['layer_index']}.mlp.switch_mlp."
                f"{alloc['projection']}_proj"
            )
            alloc["module_weight_key"] = f"{alloc['module_path']}.weight"
    return dec


def _dense_kquant_plan(codec="q8_0"):
    dec = _dense_decision()
    for alloc in dec["allocation"]:
        if alloc["kind"] == "affine" and alloc["source_name"].endswith("q_proj.weight"):
            alloc["format"] = "kquant"
            alloc["codec"] = codec
            alloc["kquant_codec"] = codec
            alloc["gguf_tensor"] = "blk.3.attn_q_a.weight"
            alloc["imatrix_key"] = "blk.3.attn_q_a.weight"
            alloc["module_path"] = "model.layers.3.self_attn.q_proj"
            alloc["module_weight_key"] = f"{alloc['module_path']}.weight"
    return dec


def _calibrated_decision():
    units = [
        _affine_unit(
            "model.language_model.layers.1.self_attn.q_proj.weight",
            "attn.q_proj",
            layer_index=1,
        ),
        _expert_unit("model.language_model.layers.0.mlp.experts.gate_up_proj", 0, "gate"),
        _expert_unit("model.language_model.layers.0.mlp.experts.gate_up_proj", 0, "up"),
    ]
    ev = make_artifact(
        "probe_evidence",
        SUBJECT,
        PRODUCER,
        status="valid",
        required_features=["calibration"],
        units=units,
    )
    return _plan(decide(ev, target_quality=0.5))


def _dense_decision():
    units = [
        _affine_unit("model.language_model.layers.0.mlp.gate_proj.weight", "ffn.gate_proj"),
        _affine_unit("model.language_model.layers.0.mlp.up_proj.weight", "ffn.up_proj"),
        _affine_unit("model.language_model.layers.0.linear_attn.in_proj_a.weight",
                     "ssm.in_proj_a"),
        _affine_unit("model.language_model.layers.3.self_attn.q_proj.weight",
                     "attn.q_proj", layer_index=3),
    ]
    return _plan(decide(_evidence(units), target_quality=0.5))


def _dense_mxfp8_decision():
    dec = _dense_decision()
    for alloc in dec["allocation"]:
        if alloc["kind"] == "affine" and alloc["source_name"].endswith("q_proj.weight"):
            alloc["format"] = "mxfp8"
            alloc["bits"] = 8
            alloc["group_size"] = 32
            alloc["source_codec"] = "fp8_e4m3_ue8m0"
            alloc["lossless"] = True
    return dec


def _located(decision):
    """Synthetic on-disk locations, keyed exactly as the writer/manifest expect."""
    out = {}
    for a in decision["allocation"]:
        prefix = (f"{a['source_name']}.{a['projection']}"
                  if a["kind"] == "expert" else a["source_name"])
        out[located_key(a)] = {"shard": "model-00001-of-00001.safetensors",
                               "key_prefix": prefix}
    return out


def _files(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"not-real-weights-but-a-real-file")
    return [file_identity(shard)]


def test_manifest_is_a_valid_artifact(tmp_path):
    dec = _decision()
    man = build_package_manifest(dec, ARCH, _located(dec), _files(tmp_path))
    assert validate_base(man) == []
    assert man["artifact_kind"] == "package_manifest"
    assert man["status"] == "valid"


def _entry_key(t):
    """Expert entries share a source_name (fused gate_up) -> qualify by projection."""
    return f"{t['source_name']}::{t['projection']}" if t["format"] == "tq" else t["source_name"]


def test_every_allocation_becomes_a_tensor_entry_with_right_format(tmp_path):
    dec = _decision()
    man = build_package_manifest(dec, ARCH, _located(dec), _files(tmp_path))
    by_key = {_entry_key(t): t for t in man["tensors"]}
    # one entry per allocation (gate and up are distinct entries despite the
    # shared source_name).
    assert len(by_key) == len(dec["allocation"]) == len(man["tensors"])
    fmt = {_entry_key(t): t["format"] for t in man["tensors"]}
    assert fmt["model.language_model.layers.1.self_attn.q_proj.weight"] == "affine"
    assert fmt["model.language_model.layers.0.mlp.gate.weight"] == "fp16"
    assert fmt["model.language_model.layers.0.mlp.experts.gate_up_proj::gate"] == "tq"
    assert fmt["model.language_model.layers.0.mlp.experts.gate_up_proj::up"] == "tq"
    # tq entries carry version + seed; affine carries group_size.
    g = by_key["model.language_model.layers.0.mlp.experts.gate_up_proj::gate"]
    assert g["format_params"]["tq_version"] == 1 and "seed" in g["format_params"]
    q = by_key["model.language_model.layers.1.self_attn.q_proj.weight"]
    assert "group_size" in q["format_params"]


def test_passthrough_tensors_become_fp16_entries(tmp_path):
    # Structural tensors from the inventory are recorded as
    # fp16 passthrough so the runtime builds the graph without source files.
    dec = _decision()
    nm = "model.language_model.layers.0.input_layernorm.weight"
    passthrough = [{"source_name": nm, "role": "norm.input_layernorm",
                    "kind": "passthrough", "layer_index": 0}]
    pt_located = {nm: {"shard": "model-00001-of-00001.safetensors", "key_prefix": nm}}
    man = build_package_manifest(dec, ARCH, _located(dec), _files(tmp_path),
                                 passthrough=passthrough, passthrough_located=pt_located)
    by_name = {t["source_name"]: t for t in man["tensors"]}
    assert nm in by_name
    entry = by_name[nm]
    assert entry["kind"] == "passthrough" and entry["format"] == "fp16"
    assert entry["key_prefix"] == nm
    # fp16 passthrough -> the fp16_passthrough op is required.
    assert "fp16_passthrough" in man["required_ops"]
    assert man["status"] == "valid"


def test_raw_dtype_passthrough_tensors_keep_declared_format(tmp_path):
    dec = _decision()
    nm = "layers.0.ffn.gate.tid2eid"
    passthrough = [{"source_name": nm, "role": "moe.router_tid2eid",
                    "kind": "passthrough", "layer_index": 0,
                    "format": "raw_dtype_passthrough"}]
    pt_located = {nm: {"shard": "model-00001-of-00001.safetensors", "key_prefix": nm}}
    man = build_package_manifest(dec, ARCH, _located(dec), _files(tmp_path),
                                 passthrough=passthrough, passthrough_located=pt_located)
    entry = {t["source_name"]: t for t in man["tensors"]}[nm]
    assert entry["format"] == "raw_dtype_passthrough"
    assert "raw_dtype_passthrough" in man["required_ops"]
    assert man["status"] == "valid"


def test_f32_passthrough_tensors_keep_declared_format(tmp_path):
    dec = _decision()
    nm = "model.language_model.layers.0.linear_attn.in_proj_a.weight"
    passthrough = [{"source_name": nm, "role": "attn.linear_in_proj_a",
                    "kind": "passthrough", "layer_index": 0,
                    "format": "f32_passthrough"}]
    pt_located = {nm: {"shard": "model-00001-of-00001.safetensors", "key_prefix": nm}}
    man = build_package_manifest(dec, ARCH, _located(dec), _files(tmp_path),
                                 passthrough=passthrough, passthrough_located=pt_located)
    entry = {t["source_name"]: t for t in man["tensors"]}[nm]
    assert entry["format"] == "f32_passthrough"
    assert "f32_passthrough" in man["required_ops"]
    assert man["status"] == "valid"


def test_control_tensor_routed_to_fp16_passthrough_fails_closed(tmp_path):
    dec = _decision()
    nm = "layers.2.attn.compressor.ape"
    passthrough = [{"source_name": nm, "role": "attn.compressor.ape",
                    "kind": "passthrough", "layer_index": 2,
                    "format": "fp16"}]
    pt_located = {nm: {"shard": "model-00001-of-00001.safetensors", "key_prefix": nm}}
    man = build_package_manifest(dec, ARCH, _located(dec), _files(tmp_path),
                                 passthrough=passthrough, passthrough_located=pt_located)
    assert man["status"] == "invalid"
    assert any(v["code"] == "package.control_tensor_downcast" for v in man["validation"])


def test_passthrough_without_location_fails_closed(tmp_path):
    dec = _decision()
    nm = "model.language_model.layers.0.input_layernorm.weight"
    passthrough = [{"source_name": nm, "role": "norm.input_layernorm",
                    "kind": "passthrough", "layer_index": 0}]
    # no passthrough_located entry -> the manifest must refuse (fail closed).
    man = build_package_manifest(dec, ARCH, _located(dec), _files(tmp_path),
                                 passthrough=passthrough, passthrough_located={})
    assert man["status"] == "invalid"
    assert any(v["code"] == "package.unwritten_tensor" for v in man["validation"])


def test_manifest_declares_our_format_identity(tmp_path):
    # The package self-identifies as the strict format (mjtq = MoEspresso Jang TurboQuant),
    # not upstream "jangtq". jang is only the compression codec.
    dec = _decision()
    man = build_package_manifest(dec, ARCH, _located(dec), _files(tmp_path))
    assert man["package_format"] == "mjtq"
    assert man["package_format_version"] == 1


def test_architecture_facts_copied(tmp_path):
    dec = _decision()
    man = build_package_manifest(dec, ARCH, _located(dec), _files(tmp_path))
    a = man["architecture"]
    assert a["num_hidden_layers"] == 2 and a["num_experts"] == 256
    assert a["family"] == "qwen3_moe"
    assert a["source_nesting"] == "model.language_model."
    assert a["layer_types"] == ["linear_attention", "full_attention"]


def test_dense_qwen_manifest_declares_dense_family_without_expert_layout(tmp_path):
    dec = _dense_decision()
    man = build_package_manifest(dec, DENSE_ARCH, _located(dec), _files(tmp_path))
    a = man["architecture"]
    assert a["family"] == "qwen3_5_dense"
    assert a["wrapper_model_type"] == "qwen3_5"
    assert a["text_model_type"] == "qwen3_5_text"
    assert a["source_nesting"] == "model.language_model."
    assert "tq_dequant" not in man["required_ops"]
    assert "expert_layout" not in man


def test_deepseek_v4_manifest_declares_dsv4_architecture_facts(tmp_path):
    dec = _decision()
    man = build_package_manifest(dec, DS4_ARCH, _located(dec), _files(tmp_path))
    a = man["architecture"]
    assert a["family"] == "deepseek_v4_flash"
    assert a["source_nesting"] == ""
    assert a["excludes"] == ["mtp"]
    assert a["compress_ratios"] == list(DEEPSEEK_V4_FLASH_COMPRESS_RATIOS)
    assert len(a["compress_ratios"]) == 44
    assert a["compress_ratios"][42] == 4
    assert a["compress_ratios"][43] == 0
    assert a["layer_kinds"][42] == "csa"
    assert a["cache_policy"] == {"kind": "deepseek_v4_composite", "generic_kv_bits": False}
    assert a["prompt_renderer"] == "deepseek_v4_dsv4"
    assert a["expert_source_layout"] == "separate_w1_w3_w2"


def test_deepseek_v4_smoke_manifest_clamps_n_routed_experts(tmp_path):
    dec = _decision()
    arch = {
        **DS4_ARCH,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
    }

    man = build_package_manifest(
        dec,
        arch,
        _located(dec),
        _files(tmp_path),
        max_experts=1,
    )
    a = man["architecture"]

    assert a["family"] == "deepseek_v4_flash"
    assert a["smoke_max_experts"] == 1
    assert a["config"]["n_routed_experts"] == 1
    assert a["config"]["num_experts_per_tok"] == 1


def test_required_ops_declared(tmp_path):
    dec = _decision()
    man = build_package_manifest(dec, ARCH, _located(dec), _files(tmp_path))
    assert set(man["required_ops"]) == {"tq_dequant", "affine_dequant", "fp16_passthrough"}


def test_mxfp4_expert_manifest_declares_explicit_codec(tmp_path):
    dec = _mxfp4_decision()
    man = build_package_manifest(dec, DS4_ARCH, _located(dec), _files(tmp_path))

    expert_entries = [t for t in man["tensors"] if t["kind"] == "expert"]

    assert "mxfp4_dequant" in man["required_ops"]
    assert {t["format"] for t in expert_entries} == {"mxfp4"}
    for entry in expert_entries:
        assert entry["format_params"] == {
            "bits": 4,
            "group_size": 32,
            "scale_dtype": "ue8m0",
            "source_codec": "fp4_e2m1_ue8m0",
            "lossless": True,
        }


def test_kquant_expert_manifest_declares_explicit_codec(tmp_path):
    dec = _kquant_plan("q2_k")
    man = build_package_manifest(dec, DS4_ARCH, _located(dec), _files(tmp_path))

    expert_entries = [t for t in man["tensors"] if t["kind"] == "expert"]

    assert man["status"] == "valid"
    assert "kquant_dequant" in man["required_ops"]
    assert {t["format"] for t in expert_entries} == {"kquant"}
    for entry in expert_entries:
        assert entry["module_path"] == (
            f"model.layers.{entry['layer_index']}.mlp.switch_mlp."
            f"{entry['projection']}_proj"
        )
        assert entry["module_weight_key"] == f"{entry['module_path']}.weight"
        assert entry["format_params"] == {
            "kquant_codec": "q2_k",
            "bits": 2,
            "group_size": 256,
            "bytes_per_block": 84,
            "weights_per_block": 256,
            "imatrix_key": f"blk.{entry['layer_index']}.ffn_{entry['projection']}_exps.weight",
        }


def test_kquant_dense_manifest_declares_explicit_codec(tmp_path):
    dec = _dense_kquant_plan("q8_0")
    man = build_package_manifest(dec, DS4_ARCH, _located(dec), _files(tmp_path))

    dense_entries = [
        t for t in man["tensors"]
        if t["kind"] == "affine" and t["format"] == "kquant"
    ]

    assert man["status"] == "valid"
    assert "kquant_dequant" in man["required_ops"]
    assert len(dense_entries) == 1
    entry = dense_entries[0]
    assert entry["module_path"] == "model.layers.3.self_attn.q_proj"
    assert entry["module_weight_key"] == "model.layers.3.self_attn.q_proj.weight"
    assert entry["format_params"] == {
        "kquant_codec": "q8_0",
        "bits": 8,
        "group_size": 32,
        "bytes_per_block": 34,
        "weights_per_block": 32,
        "imatrix_key": "blk.3.attn_q_a.weight",
    }


def test_manifest_rejects_unknown_kquant_codec(tmp_path):
    dec = _kquant_plan("q2_not_real")

    man = build_package_manifest(dec, DS4_ARCH, _located(dec), _files(tmp_path))

    assert man["status"] == "invalid"
    assert any(
        v["code"] == "package.unsupported_kquant_codec"
        and v["actual"] == "q2_not_real"
        for v in man["validation"]
    )


def test_manifest_rejects_kquant_without_module_weight_key(tmp_path):
    dec = _kquant_plan("q2_k")
    for alloc in dec["allocation"]:
        if alloc["kind"] == "expert":
            del alloc["module_weight_key"]

    man = build_package_manifest(dec, DS4_ARCH, _located(dec), _files(tmp_path))

    assert man["status"] == "invalid"
    assert any(
        v["code"] == "package.missing_kquant_module_weight_key"
        for v in man["validation"]
    )


def test_manifest_rejects_dense_kquant_without_module_weight_key(tmp_path):
    dec = _dense_kquant_plan("q8_0")
    for alloc in dec["allocation"]:
        if alloc.get("format") == "kquant":
            del alloc["module_weight_key"]

    man = build_package_manifest(dec, DS4_ARCH, _located(dec), _files(tmp_path))

    assert man["status"] == "invalid"
    assert any(
        v["code"] == "package.missing_kquant_module_weight_key"
        for v in man["validation"]
    )


def test_dense_mxfp8_manifest_declares_explicit_codec_without_expert_layout(tmp_path):
    dec = _dense_mxfp8_decision()
    man = build_package_manifest(dec, DENSE_ARCH, _located(dec), _files(tmp_path))

    entry = next(t for t in man["tensors"] if t["format"] == "mxfp8")
    assert entry["kind"] == "affine"
    assert entry["format_params"] == {
        "bits": 8,
        "group_size": 32,
        "scale_dtype": "ue8m0",
        "source_codec": "fp8_e4m3_ue8m0",
        "lossless": True,
    }
    assert "mxfp8_dequant" in man["required_ops"]
    assert "expert_layout" not in man


def test_manifest_rejects_mxfp8_for_routed_experts(tmp_path):
    dec = _mxfp4_decision()
    for alloc in dec["allocation"]:
        if alloc["kind"] == "expert":
            alloc["format"] = "mxfp8"
            alloc["codec"] = "mxfp8"
            alloc["bits"] = 8

    man = build_package_manifest(dec, DS4_ARCH, _located(dec), _files(tmp_path))

    assert man["status"] == "invalid"
    assert any(
        v["code"] == "package.unsupported_expert_format"
        and v["actual"] == "mxfp8"
        for v in man["validation"]
    )


def test_file_identity_records_sha256(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"abc")
    fid = file_identity(shard)
    assert fid["size_bytes"] == 3
    assert fid["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert fid["path"] == "model-00001-of-00001.safetensors"


def test_provenance_chains_to_package_plan(tmp_path):
    plan = _decision()
    man = build_package_manifest(plan, ARCH, _located(plan), _files(tmp_path))
    assert man["provenance"]["source_plan_id"] == plan["artifact_id"]
    assert man["provenance"]["source_decision_id"] == plan["source_decision_id"]
    assert man["provenance"]["source_probe_id"] == plan["source_probe_id"]


def test_manifest_preserves_decision_required_features(tmp_path):
    plan = _calibrated_decision()
    man = build_package_manifest(plan, ARCH, _located(plan), _files(tmp_path))

    assert validate_base(man) == []
    assert plan["required_features"] == ["calibration"]
    assert man["required_features"] == ["calibration"]
    assert man["provenance"]["source_plan_id"] == plan["artifact_id"]


def test_manifest_is_deterministic(tmp_path):
    dec = _decision()
    loc, files = _located(dec), _files(tmp_path)
    a = build_package_manifest(dec, ARCH, loc, files)
    b = build_package_manifest(dec, ARCH, loc, files)
    assert a["artifact_id"] == b["artifact_id"] == compute_artifact_id(a)


def test_changed_file_changes_id(tmp_path):
    dec = _decision()
    loc = _located(dec)
    a = build_package_manifest(dec, ARCH, loc, _files(tmp_path))
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"different-content-entirely")
    b = build_package_manifest(dec, ARCH, loc, [file_identity(shard)])
    assert a["artifact_id"] != b["artifact_id"]


def test_unwritten_tensor_fails_closed(tmp_path):
    dec = _decision()
    loc = _located(dec)
    loc.pop(dec["allocation"][0]["source_name"])  # drop one location
    man = build_package_manifest(dec, ARCH, loc, _files(tmp_path))
    assert man["status"] == "invalid"
    assert any(v["code"] == "package.unwritten_tensor" for v in man["validation"])


def test_missing_shard_fails_closed(tmp_path):
    dec = _decision()
    loc = _located(dec)
    # point one tensor at a shard that isn't in the files list
    first = dec["allocation"][0]["source_name"]
    loc[first] = {"shard": "ghost.safetensors", "key_prefix": first}
    man = build_package_manifest(dec, ARCH, loc, _files(tmp_path))
    assert man["status"] == "invalid"
    assert any(v["code"] == "package.missing_shard" for v in man["validation"])
