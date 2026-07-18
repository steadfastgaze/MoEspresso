"""Role + imatrix-key resolver: the single mapping, tested against real shapes.

Covers real Qwen3.6-35B-A3B name shapes: the full-attn layer set {3,7,...,39}
resolves to keys that exist; embed/lm_head are keyless (not bugs); stacked
gate_up maps to two keys.
"""

from __future__ import annotations

from moespresso.inventory.deepseek_v4.roles import (
    gguf_key as deepseek_v4_gguf_key,
    module_path as deepseek_v4_module_path,
    tensor_role as deepseek_v4_tensor_role,
)
from moespresso.inventory.roles import (
    expert_gguf_keys,
    expert_role,
    non_expert_gguf_key,
    non_expert_role,
    parse_stacked_expert,
    structural_role,
    tensor_layer,
)

LT = ["linear_attention"] * 40
for i in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39):
    LT[i] = "full_attention"


def _n(layer, suffix):
    return f"model.language_model.layers.{layer}.{suffix}"


def test_layer_extraction_handles_double_nesting():
    assert tensor_layer(_n(7, "self_attn.q_proj.weight")) == 7
    assert tensor_layer("lm_head.weight") is None


def test_full_attention_roles_and_keys():
    n = _n(3, "self_attn.q_proj.weight")
    assert non_expert_role(n, LT) == "attn.q_proj"
    assert non_expert_gguf_key(n, LT) == "blk.3.attn_q.weight"


def test_linear_attention_roles_and_keys():
    pairs = {
        "linear_attn.in_proj_qkv.weight": ("ssm.in_proj_qkv", "blk.0.attn_qkv.weight"),
        "linear_attn.in_proj_z.weight": ("ssm.in_proj_z", "blk.0.attn_gate.weight"),
        "linear_attn.in_proj_a.weight": ("ssm.in_proj_a", "blk.0.ssm_alpha.weight"),
        "linear_attn.in_proj_b.weight": ("ssm.in_proj_b", "blk.0.ssm_beta.weight"),
        "linear_attn.out_proj.weight": ("ssm.out_proj", "blk.0.ssm_out.weight"),
    }
    for suf, (role, key) in pairs.items():
        n = _n(0, suf)
        assert non_expert_role(n, LT) == role
        assert non_expert_gguf_key(n, LT) == key


def test_shared_expert_and_router_gate():
    assert non_expert_gguf_key(_n(5, "mlp.gate.weight"), LT) == "blk.5.ffn_gate_inp.weight"
    assert non_expert_role(_n(5, "mlp.shared_expert.down_proj.weight"), LT) == "moe.shared_expert.down_proj"
    assert non_expert_gguf_key(_n(5, "mlp.shared_expert.down_proj.weight"), LT) == "blk.5.ffn_down_shexp.weight"


def test_embed_and_lm_head_have_role_but_no_key():
    assert non_expert_role("model.language_model.embed_tokens.weight", LT) == "embed_tokens"
    assert non_expert_gguf_key("model.language_model.embed_tokens.weight", LT) is None
    assert non_expert_role("lm_head.weight", LT) == "lm_head"
    assert non_expert_gguf_key("lm_head.weight", LT) is None


def test_dense_ffn_roles_and_keys():
    # A dense transformer's per-layer MLP: no router, no experts, 2D weights.
    dense = {
        "mlp.gate_proj.weight": ("ffn.gate_proj", "blk.4.ffn_gate.weight"),
        "mlp.up_proj.weight": ("ffn.up_proj", "blk.4.ffn_up.weight"),
        "mlp.down_proj.weight": ("ffn.down_proj", "blk.4.ffn_down.weight"),
    }
    for suf, (role, key) in dense.items():
        n = _n(4, suf)
        assert non_expert_role(n, LT) == role
        assert non_expert_gguf_key(n, LT) == key


def test_dense_ffn_does_not_collide_with_moe_router_or_experts():
    # 'mlp.gate.weight' (router) stays the router; 'mlp.gate_proj.weight' is dense FFN.
    assert non_expert_role(_n(4, "mlp.gate.weight"), LT) == "moe.router_gate"
    assert non_expert_role(_n(4, "mlp.gate_proj.weight"), LT) == "ffn.gate_proj"
    # dense down_proj is not parsed as a stacked expert (no '.experts.').
    assert parse_stacked_expert(_n(4, "mlp.down_proj.weight")) is None


def test_structural_role_carries_text_norms_and_ssm_state():
    # Norms + SSM state the graph needs but that aren't quantized -> passthrough.
    assert structural_role(_n(3, "input_layernorm.weight")) == "norm.input_layernorm"
    assert structural_role(_n(3, "post_attention_layernorm.weight")) == "norm.post_attention_layernorm"
    assert structural_role(_n(3, "self_attn.q_norm.weight")) == "norm.attn.q_norm"
    assert structural_role(_n(0, "linear_attn.norm.weight")) == "norm.ssm.norm"
    assert structural_role(_n(0, "linear_attn.A_log")) == "ssm.A_log"
    assert structural_role(_n(0, "linear_attn.dt_bias")) == "ssm.dt_bias"
    assert structural_role(_n(0, "linear_attn.conv1d.weight")) == "ssm.conv1d"
    assert structural_role("model.language_model.norm.weight") == "norm.final"


def test_structural_role_excludes_mtp_vision_and_quantized():
    # mtp/vision must not be carried (text-only scope); quantized weights aren't
    # structural; an attn bias is neither.
    assert structural_role("mtp.layers.0.input_layernorm.weight") is None
    assert structural_role("model.visual.blocks.0.norm2.weight") is None
    assert structural_role(_n(3, "self_attn.q_proj.weight")) is None
    assert structural_role(_n(3, "self_attn.q_proj.bias")) is None


def test_stacked_expert_parse_and_keys():
    assert parse_stacked_expert(_n(0, "mlp.experts.gate_up_proj.weight")) == (0, "gate_up")
    assert parse_stacked_expert(_n(0, "mlp.experts.down_proj.weight")) == (0, "down")
    assert expert_gguf_keys(0, "gate_up") == [
        "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"]
    assert expert_gguf_keys(12, "down") == ["blk.12.ffn_down_exps.weight"]
    assert expert_role("gate") == "moe.expert.gate"


def test_full_coverage_on_real_layout():
    """Every real non-expert name resolves to an existing key except embed/lm_head."""
    full_suf = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight"]
    lin_suf = ["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
               "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
               "linear_attn.out_proj.weight"]
    moe_suf = ["mlp.gate.weight", "mlp.shared_expert_gate.weight",
               "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
               "mlp.shared_expert.down_proj.weight"]
    real_keys = set()
    for layer in range(40):
        attn = (["attn_q", "attn_k", "attn_v", "attn_output"] if LT[layer] == "full_attention"
                else ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out"])
        for fam in attn + ["ffn_gate_inp", "ffn_gate_inp_shexp",
                           "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"]:
            real_keys.add(f"blk.{layer}.{fam}.weight")

    names = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    for layer in range(40):
        sufs = moe_suf + (full_suf if LT[layer] == "full_attention" else lin_suf)
        names += [_n(layer, s) for s in sufs]

    mapped = keyless = 0
    for name in names:
        assert non_expert_role(name, LT) is not None, name
        key = non_expert_gguf_key(name, LT)
        if key is None:
            keyless += 1
        else:
            assert key in real_keys, f"{name} -> {key} not a real key"
            mapped += 1
    assert keyless == 2
    assert mapped == len(names) - 2


def _ds4(name):
    return deepseek_v4_tensor_role(name)


def test_deepseek_v4_short_names_map_to_typed_roles():
    assert _ds4("embed.weight")["role"] == "embed_tokens"
    assert _ds4("head.weight")["role"] == "lm_head"
    assert _ds4("layers.2.attn.wq_a.weight")["role"] == "attn.wq_a"
    assert _ds4("layers.2.attn.wq_a.scale")["role"] == "attn.wq_a.scale"
    assert _ds4("layers.2.attn.compressor.wkv.weight")["role"] == "attn.compressor.wkv"
    assert _ds4("layers.2.attn.indexer.wq_b.weight")["role"] == "attn.indexer.wq_b"
    assert _ds4("layers.2.attn.indexer.weights_proj.weight")["role"] == (
        "attn.indexer.weights_proj"
    )
    assert _ds4("layers.2.ffn.gate.weight")["role"] == "moe.router_gate"
    assert _ds4("layers.2.ffn.gate.bias")["role"] == "moe.router_bias"
    assert _ds4("layers.0.ffn.gate.tid2eid")["role"] == "moe.router_tid2eid"


def test_deepseek_v4_short_names_map_to_real_gguf_recipe_keys():
    assert deepseek_v4_gguf_key("layers.2.attn.wq_a.weight") == (
        "blk.2.attn_q_a.weight"
    )
    assert deepseek_v4_gguf_key("layers.2.attn.wq_b.weight") == (
        "blk.2.attn_q_b.weight"
    )
    assert deepseek_v4_gguf_key("layers.2.attn.wkv.weight") == (
        "blk.2.attn_kv.weight"
    )
    assert deepseek_v4_gguf_key("layers.2.attn.wo_a.weight") == (
        "blk.2.attn_output_a.weight"
    )
    assert deepseek_v4_gguf_key("layers.2.attn.wo_b.weight") == (
        "blk.2.attn_output_b.weight"
    )
    assert deepseek_v4_gguf_key("layers.2.ffn.shared_experts.w1.weight") == (
        "blk.2.ffn_gate_shexp.weight"
    )
    assert deepseek_v4_gguf_key("layers.2.ffn.shared_experts.w3.weight") == (
        "blk.2.ffn_up_shexp.weight"
    )
    assert deepseek_v4_gguf_key("layers.2.ffn.shared_experts.w2.weight") == (
        "blk.2.ffn_down_shexp.weight"
    )
    assert deepseek_v4_gguf_key("head.weight") == "output.weight"
    assert deepseek_v4_gguf_key("embed.weight") is None


def test_deepseek_v4_short_names_map_to_jang_module_paths():
    assert deepseek_v4_module_path("layers.2.attn.wq_a.weight") == (
        "model.layers.2.self_attn.wq_a"
    )
    assert deepseek_v4_module_path("layers.2.ffn.shared_experts.w1.weight") == (
        "model.layers.2.mlp.shared_experts.gate_proj"
    )
    assert deepseek_v4_module_path("layers.2.ffn.shared_experts.w3.weight") == (
        "model.layers.2.mlp.shared_experts.up_proj"
    )
    assert deepseek_v4_module_path("layers.2.ffn.shared_experts.w2.weight") == (
        "model.layers.2.mlp.shared_experts.down_proj"
    )
    assert deepseek_v4_module_path("head.weight") == "lm_head"


def test_deepseek_v4_expert_sources_keep_separate_projection_layout():
    w1 = _ds4("layers.7.ffn.experts.123.w1.weight")
    w3 = _ds4("layers.7.ffn.experts.123.w3.weight")
    w2 = _ds4("layers.7.ffn.experts.123.w2.weight")
    assert w1["kind"] == "expert_source" and w1["projection"] == "gate"
    assert w3["kind"] == "expert_source" and w3["projection"] == "up"
    assert w2["kind"] == "expert_source" and w2["projection"] == "down"
    assert w1["expert_index"] == 123
    scale = _ds4("layers.7.ffn.experts.123.w1.scale")
    assert scale["kind"] == "codec_scale"
    assert scale["projection"] == "gate"


def test_deepseek_v4_controls_are_raw_passthrough_and_mtp_is_excluded():
    assert _ds4("layers.2.attn.attn_sink") == {
        "kind": "passthrough",
        "role": "attn.attn_sink",
        "layer_index": 2,
        "format": "raw_dtype_passthrough",
    }
    assert _ds4("layers.2.attn.compressor.ape")["format"] == "raw_dtype_passthrough"
    assert _ds4("layers.2.hc_attn_fn")["role"] == "hc.control"
    assert _ds4("hc_head_fn")["role"] == "hc.control"
    assert _ds4("layers.2.attn.q_norm.weight")["format"] == "fp16"
    assert _ds4("mtp.0.e_proj.weight") is None
