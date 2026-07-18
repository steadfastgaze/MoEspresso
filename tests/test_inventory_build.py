"""source_inventory builder: classifies tensors into roles, records coverage.

Synthetic TensorHeaders (no model load). Mirrors the real Qwen MoE layout:
stacked experts, SSM + full-attn layers, shared experts, embed/lm_head, norms.
"""

from __future__ import annotations

from moespresso.core.artifact import compute_artifact_id
from moespresso.inventory.build import build_inventory_from_headers
from moespresso.inventory.safetensors_header import TensorHeader

SUBJECT = {"source_root": "toy", "source_format": "hf_safetensors"}
LT = ["linear_attention", "full_attention"]  # layer 0 SSM, layer 1 full-attn


def _h(name, shape, dtype="BF16", shard="model-00001.safetensors"):
    return TensorHeader(name=name, shape=tuple(shape), dtype=dtype, shard=shard)


def _headers():
    p = "model.language_model.layers"
    return [
        _h("model.language_model.embed_tokens.weight", [1000, 64]),
        _h("lm_head.weight", [1000, 64]),
        _h(f"{p}.0.linear_attn.in_proj_qkv.weight", [64, 64]),
        _h(f"{p}.0.linear_attn.out_proj.weight", [64, 64]),
        _h(f"{p}.1.self_attn.q_proj.weight", [64, 64]),
        _h(f"{p}.1.self_attn.o_proj.weight", [64, 64]),
        _h(f"{p}.0.mlp.gate.weight", [8, 64]),
        _h(f"{p}.0.mlp.shared_expert.down_proj.weight", [64, 64]),
        _h(f"{p}.0.mlp.experts.gate_up_proj.weight", [8, 128, 64]),  # stacked 3D
        _h(f"{p}.0.mlp.experts.down_proj.weight", [8, 64, 64]),      # stacked 3D
        _h(f"{p}.0.input_layernorm.weight", [64]),                   # passthrough
        _h(f"{p}.0.linear_attn.A_log", [8]),                         # passthrough (ssm)
        _h(f"{p}.0.self_attn.q_proj.bias", [64]),                    # skipped (bias)
    ]


def test_inventory_classifies_experts_and_non_experts():
    inv = build_inventory_from_headers(_headers(), SUBJECT, LT)
    assert inv["artifact_kind"] == "source_inventory"
    assert inv["counts"]["expert"] == 2          # gate_up + down (stacked)
    # affine: embed, lm_head, 2 ssm, 2 self_attn, router gate, shared down = 8
    assert inv["counts"]["affine"] == 8
    # structural passthrough: input_layernorm + linear_attn.A_log
    assert inv["counts"]["passthrough"] == 2
    assert inv["counts"]["unknown"] == 0
    # the layernorm/ssm-state are carried (passthrough), the bias is dropped
    names = {e["source_name"] for e in inv["tensors"]}
    assert any("input_layernorm" in n for n in names)
    assert not any(n.endswith(".bias") for n in names)


def test_inventory_records_roles_and_keys():
    inv = build_inventory_from_headers(_headers(), SUBJECT, LT)
    by_name = {e["source_name"]: e for e in inv["tensors"]}
    qkv = by_name["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]
    assert qkv["role"] == "ssm.in_proj_qkv"
    assert qkv["gguf_keys"] == ["blk.0.attn_qkv.weight"]
    gate_up = by_name["model.language_model.layers.0.mlp.experts.gate_up_proj.weight"]
    assert gate_up["kind"] == "expert" and gate_up["projection"] == "gate_up"
    assert gate_up["gguf_keys"] == [
        "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"]
    assert by_name["lm_head.weight"]["role"] == "lm_head"
    assert by_name["lm_head.weight"]["gguf_keys"] == []


def test_inventory_is_a_valid_content_addressed_artifact():
    inv = build_inventory_from_headers(_headers(), SUBJECT, LT)
    assert inv["status"] == "valid"
    assert inv["artifact_id"] == compute_artifact_id(inv)


def test_inventory_is_deterministic():
    a = build_inventory_from_headers(_headers(), SUBJECT, LT)
    b = build_inventory_from_headers(list(reversed(_headers())), SUBJECT, LT)
    assert a["artifact_id"] == b["artifact_id"]


def test_coverage_flags_absent_keys():
    keys = {
        "blk.0.attn_qkv.weight", "blk.1.attn_q.weight", "blk.1.attn_output.weight",
        "blk.0.ffn_gate_inp.weight", "blk.0.ffn_down_shexp.weight",
        "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    }  # intentionally omit blk.0.ssm_out.weight
    inv = build_inventory_from_headers(_headers(), SUBJECT, LT, imatrix_keys=keys)
    cov = inv["imatrix_coverage"]
    assert cov["absent"] == 1
    assert any(v["code"] == "imatrix.key_absent" for v in inv["validation"])


# --- dense second-pressure fixture (spec: "Avoiding Single-Family Overfit") ---
# A dense transformer has no experts and no router: its whole MLP backbone is
# per-layer gate/up/down. This guards against the inventory silently assuming MoE.

def _dense_headers(n_layers=3):
    p = "model.language_model.layers"
    hs = [
        _h("model.language_model.embed_tokens.weight", [1000, 64]),
        _h("lm_head.weight", [1000, 64]),
    ]
    for i in range(n_layers):
        for suf in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight", "mlp.up_proj.weight",
                    "mlp.down_proj.weight"):
            hs.append(_h(f"{p}.{i}.{suf}", [64, 64]))
        hs.append(_h(f"{p}.{i}.input_layernorm.weight", [64]))  # skipped
    return hs


DENSE_LT = ["full_attention"] * 3


def test_dense_model_has_no_experts_and_no_unknowns():
    inv = build_inventory_from_headers(_dense_headers(), SUBJECT, DENSE_LT)
    assert inv["counts"]["expert"] == 0
    assert inv["counts"]["unknown"] == 0          # the FFN backbone is fully classified
    # 3 layers x (4 attn + 3 ffn) + embed + lm_head = 23 affine, 0 expert.
    assert inv["counts"]["affine"] == 23
    assert inv["status"] == "valid"


def test_dense_ffn_backbone_is_allocatable():
    inv = build_inventory_from_headers(_dense_headers(), SUBJECT, DENSE_LT)
    ffn = [e for e in inv["tensors"] if e["role"].startswith("ffn.")]
    assert len(ffn) == 9                          # 3 projections x 3 layers
    assert all(e["status"] == "required" for e in ffn)
    assert all(e["gguf_keys"] for e in ffn)       # every FFN tensor has an imatrix key


def test_dense_imatrix_coverage_is_complete():
    keys = set()
    for i in range(3):
        for fam in ("attn_q", "attn_k", "attn_v", "attn_output",
                    "ffn_gate", "ffn_up", "ffn_down"):
            keys.add(f"blk.{i}.{fam}.weight")
    inv = build_inventory_from_headers(_dense_headers(), SUBJECT, DENSE_LT, imatrix_keys=keys)
    cov = inv["imatrix_coverage"]
    assert cov["absent"] == 0
    assert cov["present_in_imatrix"] == cov["resolved_keys"]


def test_dense_qwen35_headers_exclude_vision_mtp_and_keep_ssm_passthrough():
    p = "model.language_model.layers"
    hs = _dense_headers(n_layers=4) + [
        _h(f"{p}.0.linear_attn.in_proj_qkv.weight", [6144, 1024]),
        _h(f"{p}.0.linear_attn.in_proj_z.weight", [2048, 1024]),
        _h(f"{p}.0.linear_attn.in_proj_a.weight", [16, 1024]),
        _h(f"{p}.0.linear_attn.in_proj_b.weight", [16, 1024]),
        _h(f"{p}.0.linear_attn.out_proj.weight", [1024, 2048]),
        _h(f"{p}.0.linear_attn.A_log", [16], dtype="F32"),
        _h(f"{p}.0.linear_attn.dt_bias", [16]),
        _h(f"{p}.0.linear_attn.conv1d.weight", [6144, 1, 4]),
        _h(f"{p}.0.linear_attn.norm.weight", [128], dtype="F32"),
        _h("model.visual.blocks.0.attn.qkv.weight", [2304, 768]),
        _h("model.visual.blocks.0.norm1.weight", [768]),
        _h("mtp.layers.0.mlp.gate_proj.weight", [3584, 1024]),
        _h("mtp.layers.0.input_layernorm.weight", [1024]),
    ]

    inv = build_inventory_from_headers(hs, SUBJECT, DENSE_LT + ["full_attention"])
    names = {e["source_name"] for e in inv["tensors"]}

    assert inv["counts"]["expert"] == 0
    assert inv["counts"]["unknown"] == 0
    assert not any(n.startswith("model.visual.") for n in names)
    assert not any(n.startswith("mtp.") for n in names)
    roles = {e["source_name"]: e["role"] for e in inv["tensors"]}
    assert roles[f"{p}.0.linear_attn.in_proj_a.weight"] == "ssm.in_proj_a"
    assert roles[f"{p}.0.linear_attn.in_proj_b.weight"] == "ssm.in_proj_b"
    assert roles[f"{p}.0.linear_attn.conv1d.weight"] == "ssm.conv1d"


def test_dense_qwen_tied_embeddings_do_not_require_lm_head_header():
    hs = [h for h in _dense_headers(n_layers=2) if h.name != "lm_head.weight"]
    inv = build_inventory_from_headers(hs, SUBJECT, ["full_attention"] * 2)
    names = {e["source_name"] for e in inv["tensors"]}
    assert "lm_head.weight" not in names
    assert inv["counts"]["unknown"] == 0
    assert inv["counts"]["expert"] == 0


def test_unknown_2d_weight_is_flagged_not_skipped():
    hs = _headers() + [_h("model.language_model.layers.0.weird.proj.weight", [64, 64])]
    inv = build_inventory_from_headers(hs, SUBJECT, LT)
    assert inv["counts"]["unknown"] == 1
    weird = [e for e in inv["tensors"] if "weird" in e["source_name"]][0]
    assert weird["status"] == "unknown" and weird["role"] == "unknown"
    assert any(v["code"] == "inventory.unknown_tensors" for v in inv["validation"])


def _ds4_headers():
    return [
        _h("embed.weight", [129280, 4096], dtype="BF16"),
        _h("head.weight", [129280, 4096], dtype="BF16"),
        _h("layers.2.attn.wq_a.weight", [1024, 4096], dtype="F8_E4M3"),
        _h("layers.2.attn.wq_a.scale", [8, 32], dtype="F8_E8M0"),
        _h("layers.2.attn.wq_b.weight", [4096, 1024], dtype="F8_E4M3"),
        _h("layers.2.attn.wkv.weight", [1024, 4096], dtype="F8_E4M3"),
        _h("layers.2.attn.wo_a.weight", [1024, 4096], dtype="F8_E4M3"),
        _h("layers.2.attn.wo_b.weight", [4096, 1024], dtype="F8_E4M3"),
        _h("layers.2.attn.compressor.wkv.weight", [1024, 4096], dtype="BF16"),
        _h("layers.2.attn.indexer.wq_b.weight", [8192, 1024], dtype="F8_E4M3"),
        _h("layers.2.attn.indexer.wq_b.scale", [64, 8], dtype="F8_E8M0"),
        _h("layers.2.attn_norm.weight", [4096], dtype="BF16"),
        _h("layers.2.ffn_norm.weight", [4096], dtype="BF16"),
        _h("layers.2.attn.q_norm.weight", [1024], dtype="BF16"),
        _h("layers.2.attn.kv_norm.weight", [512], dtype="BF16"),
        _h("layers.2.attn.attn_sink", [64], dtype="F32"),
        _h("layers.2.attn.compressor.ape", [4, 1024], dtype="F32"),
        _h("layers.2.ffn.gate.weight", [256, 4096], dtype="BF16"),
        _h("layers.2.ffn.gate.bias", [256], dtype="F32"),
        _h("layers.0.ffn.gate.tid2eid", [129280, 6], dtype="I64"),
        _h("layers.2.ffn.shared_experts.w1.weight", [2048, 4096], dtype="F8_E4M3"),
        _h("layers.2.ffn.shared_experts.w3.weight", [2048, 4096], dtype="F8_E4M3"),
        _h("layers.2.ffn.shared_experts.w2.weight", [4096, 2048], dtype="F8_E4M3"),
        _h("layers.2.ffn.experts.0.w1.weight", [2048, 2048], dtype="I8"),
        _h("layers.2.ffn.experts.0.w1.scale", [2048, 128], dtype="F8_E8M0"),
        _h("layers.2.hc_attn_fn", [24, 16384], dtype="F32"),
        _h("hc_head_fn", [4, 16384], dtype="F32"),
        _h("mtp.0.e_proj.weight", [4096, 4096], dtype="F8_E4M3"),
    ]


def test_deepseek_v4_inventory_classifies_short_names_without_qwen_skips():
    inv = build_inventory_from_headers(
        _ds4_headers(),
        SUBJECT,
        layer_types=None,
        family="deepseek_v4_flash",
    )
    by_name = {e["source_name"]: e for e in inv["tensors"]}
    assert inv["status"] == "valid"
    assert inv["counts"]["affine"] == 13
    assert inv["counts"]["expert_source"] == 1
    assert inv["counts"]["codec_scale"] == 3
    assert inv["counts"]["passthrough"] == 10
    assert inv["counts"]["unknown"] == 0
    assert by_name["layers.2.attn.wq_a.scale"]["kind"] == "codec_scale"
    assert by_name["layers.2.ffn.gate.bias"]["format"] == "raw_dtype_passthrough"
    assert by_name["layers.0.ffn.gate.tid2eid"]["format"] == "raw_dtype_passthrough"
    assert by_name["layers.2.attn_norm.weight"]["role"] == "norm.attn_norm"
    assert by_name["layers.2.ffn_norm.weight"]["role"] == "norm.ffn_norm"
    assert by_name["layers.2.attn.q_norm.weight"]["format"] == "fp16"
    assert by_name["layers.2.ffn.experts.0.w1.weight"]["projection"] == "gate"
    assert by_name["layers.2.ffn.experts.0.w1.weight"]["gguf_keys"] == [
        "blk.2.ffn_gate_exps.weight"
    ]
    assert by_name["layers.2.attn.wq_a.weight"]["gguf_keys"] == [
        "blk.2.attn_q_a.weight"
    ]
    assert by_name["layers.2.attn.wq_b.weight"]["gguf_keys"] == [
        "blk.2.attn_q_b.weight"
    ]
    assert by_name["layers.2.attn.wkv.weight"]["gguf_keys"] == [
        "blk.2.attn_kv.weight"
    ]
    assert by_name["layers.2.attn.wo_a.weight"]["gguf_keys"] == [
        "blk.2.attn_output_a.weight"
    ]
    assert by_name["layers.2.attn.wo_b.weight"]["gguf_keys"] == [
        "blk.2.attn_output_b.weight"
    ]
    assert by_name["layers.2.ffn.shared_experts.w1.weight"]["gguf_keys"] == [
        "blk.2.ffn_gate_shexp.weight"
    ]
    assert by_name["layers.2.ffn.shared_experts.w3.weight"]["gguf_keys"] == [
        "blk.2.ffn_up_shexp.weight"
    ]
    assert by_name["layers.2.ffn.shared_experts.w2.weight"]["gguf_keys"] == [
        "blk.2.ffn_down_shexp.weight"
    ]
    assert "mtp.0.e_proj.weight" not in by_name


def test_deepseek_v4_inventory_unknown_tensor_is_blocking():
    headers = _ds4_headers() + [
        _h("layers.2.attn.unexpected.weight", [16, 16], dtype="BF16"),
    ]
    inv = build_inventory_from_headers(
        headers,
        SUBJECT,
        layer_types=None,
        family="deepseek_v4_flash",
    )
    assert inv["status"] == "invalid"
    assert inv["counts"]["unknown"] == 1
    assert any(v["code"] == "inventory.unknown_tensors" and v["blocking"]
               for v in inv["validation"])


def test_deepseek_v4_inventory_top_level_unknown_tensor_is_blocking():
    headers = _ds4_headers() + [
        _h("hc_head_unexpected", [4, 16384], dtype="F32"),
    ]
    inv = build_inventory_from_headers(
        headers,
        SUBJECT,
        layer_types=None,
        family="deepseek_v4_flash",
    )

    by_name = {e["source_name"]: e for e in inv["tensors"]}
    assert inv["status"] == "invalid"
    assert inv["counts"]["unknown"] == 1
    assert by_name["hc_head_unexpected"]["status"] == "unknown"
    assert by_name["hc_head_unexpected"]["layer_index"] is None
    assert any(v["code"] == "inventory.unknown_tensors" and v["blocking"]
               for v in inv["validation"])


def test_deepseek_v4_inventory_records_expert_imatrix_coverage():
    keys = {
        "output.weight",
        "blk.2.attn_q_a.weight",
        "blk.2.attn_q_b.weight",
        "blk.2.attn_kv.weight",
        "blk.2.attn_output_a.weight",
        "blk.2.attn_output_b.weight",
        "blk.2.ffn_gate_shexp.weight",
        "blk.2.ffn_up_shexp.weight",
        "blk.2.ffn_down_shexp.weight",
        "blk.2.ffn_gate_exps.weight",
    }
    inv = build_inventory_from_headers(
        _ds4_headers(),
        SUBJECT,
        layer_types=None,
        imatrix_keys=keys,
        family="deepseek_v4_flash",
    )
    assert inv["status"] == "valid"
    assert inv["imatrix_coverage"] == {
        "resolved_keys": 10,
        "present_in_imatrix": 10,
        "absent": 0,
    }


def test_deepseek_v4_inventory_warns_on_missing_expert_imatrix_key():
    inv = build_inventory_from_headers(
        _ds4_headers(),
        SUBJECT,
        layer_types=None,
        imatrix_keys=set(),
        family="deepseek_v4_flash",
    )
    assert inv["status"] == "valid"
    assert inv["imatrix_coverage"]["absent"] == 10
    assert any(v["code"] == "imatrix.key_absent" and not v["blocking"]
               for v in inv["validation"])


def test_qwen_inventory_behavior_unchanged_without_deepseek_family():
    headers = _ds4_headers()
    inv = build_inventory_from_headers(headers, SUBJECT, layer_types=None)
    assert inv["counts"]["expert_source"] == 0
    assert inv["counts"]["codec_scale"] == 0
    assert inv["counts"]["unknown"] > 0
