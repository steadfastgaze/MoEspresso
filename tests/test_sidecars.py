"""jang-compatible sidecar generation: pure, no mlx/jang.

These tests pin the invariant: tensor_map / config.quantization are affine-only,
and TQ experts only appear in routed_expert_bit_plan. TQ (switch_mlp) groups must
not leak into `tensor_map`, or the loader would mutate the TQ kernel modules'
bits/group_size -> corrupted dequant -> collapsed output.
"""

from __future__ import annotations

from moespresso.package.sidecars import build_sidecars

ARCH = {
    "family": "qwen3_5_moe",
    "config": {"model_type": "qwen3_5_moe_text", "num_experts": 8,
               "hidden_size": 64, "moe_intermediate_size": 32},
}
DENSE_ARCH = {
    "family": "qwen3_5_dense",
    "wrapper_model_type": "qwen3_5",
    "text_model_type": "qwen3_5_text",
    "config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "tie_word_embeddings": False,
    },
}
DS4_ARCH = {
    "family": "deepseek_v4_flash",
    "config": {
        "model_type": "deepseek_v4",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
    },
}


def _tensor(source_name, fmt, *, bits=4, group_size=128, layer=None, proj=None,
            key_prefix=None, role=None):
    t = {"source_name": source_name, "format": fmt, "role": role,
         "key_prefix": key_prefix or source_name, "format_params": {}}
    if fmt == "affine":
        t["format_params"] = {"bits": bits, "group_size": group_size}
    elif fmt in {"mxfp4", "mxfp8"}:
        t["format_params"] = {"bits": bits, "group_size": group_size}
    elif fmt == "tq":
        t["format_params"] = {"bits": bits, "seed": 42, "tq_version": 1}
        t["layer_index"], t["projection"] = layer, proj
    elif fmt == "kquant":
        t["format_params"] = {
            "kquant_codec": "q8_0",
            "bits": bits,
            "group_size": group_size,
        }
        t["kind"] = "affine"
        t["module_weight_key"] = f"{key_prefix or source_name}.weight"
    return t


def _manifest(tensors):
    return {"architecture": ARCH, "tensors": tensors}


def _dense_manifest(tensors):
    return {"architecture": DENSE_ARCH, "tensors": tensors}


def _deepseek_manifest(tensors):
    return {"architecture": DS4_ARCH, "tensors": tensors}


def _mixed_tensors():
    return [
        _tensor("lm_head.weight", "affine", bits=6),
        _tensor("model.language_model.layers.0.self_attn.q_proj.weight", "affine", bits=3),
        _tensor("model.language_model.layers.0.mlp.gate.weight", "fp16",
                role="moe.router_gate"),
        _tensor("model.language_model.layers.0.input_layernorm.weight", "fp16"),
        _tensor("model.language_model.layers.0.mlp.experts.gate_up_proj", "tq",
                bits=4, layer=0, proj="gate",
                key_prefix="language_model.model.layers.0.mlp.switch_mlp.gate_proj"),
        _tensor("model.language_model.layers.0.mlp.experts.down_proj", "tq",
                bits=2, layer=0, proj="down",
                key_prefix="language_model.model.layers.0.mlp.switch_mlp.down_proj"),
    ]


def test_tensor_map_is_affine_only_never_tq():
    # The regression: a TQ/switch_mlp key in tensor_map mutates the kernel module.
    _, jc = build_sidecars(_manifest(_mixed_tensors()))
    tm = jc["quantization"]["tensor_map"]
    assert all("switch_mlp" not in k for k in tm), f"TQ leaked into tensor_map: {tm}"
    assert all("experts" not in k for k in tm)
    # the affine attn proj is present (sanitized, no .weight suffix).
    assert "language_model.model.layers.0.self_attn.q_proj" in tm


def test_config_quantization_is_affine_only_never_tq():
    cfg, _ = build_sidecars(_manifest(_mixed_tensors()))
    modules = [k for k, v in cfg["quantization"].items() if isinstance(v, dict)]
    assert all("switch_mlp" not in k and "experts" not in k for k in modules)


def test_experts_only_in_routed_bit_plan():
    _, jc = build_sidecars(_manifest(_mixed_tensors()))
    rlb = jc["routed_expert_bit_plan"]["routed_layer_bits"]
    assert rlb == {"0": {"gate": 4, "down": 2}}  # per-layer per-proj expert bits


def test_fp16_passthrough_is_not_quantized():
    # router gate + norms are fp16 -> no quantization entry anywhere.
    cfg, jc = build_sidecars(_manifest(_mixed_tensors()))
    keys = set(cfg["quantization"]) | set(jc["quantization"]["tensor_map"])
    assert not any("mlp.gate" in k and "switch_mlp" not in k for k in keys)
    assert not any("layernorm" in k for k in keys)


def test_mxtq_bits_banner_is_not_authoritative():
    # config mxtq_bits is a single int; jang per-role dict is the 0 sentinel.
    cfg, jc = build_sidecars(_manifest(_mixed_tensors()))
    assert isinstance(cfg["mxtq_bits"], int)
    assert set(jc["mxtq_bits"].values()) == {0}


def test_dense_sidecars_use_regular_jang_v2_not_moe_profile():
    cfg, jc = build_sidecars(_dense_manifest([
        _tensor("model.language_model.layers.0.self_attn.q_proj.weight",
                "affine", bits=4, group_size=64),
        _tensor("model.language_model.layers.0.mlp.gate_proj.weight",
                "affine", bits=6, group_size=128),
        _tensor("model.language_model.layers.0.input_layernorm.weight", "fp16"),
    ]))

    assert cfg["model_type"] == "qwen3_5"
    assert cfg["text_config"]["model_type"] == "qwen3_5_text"
    assert cfg["quantization"]["bits"] == 4
    assert cfg["quantization"]["group_size"] == 64
    assert cfg["quantization"]["language_model.model.layers.0.mlp.gate_proj"] == {
        "bits": 6,
        "group_size": 128,
    }

    assert jc["format"] == "jang"
    assert jc["format_version"] == "2.0"
    assert jc["quantization"]["method"] == "affine"
    assert jc["quantization"]["bits"] == 4
    assert jc["quantization"]["group_size"] == 64
    assert "MOESPRESSO_MOE" not in str(jc)
    assert "routed_expert_bit_plan" not in jc
    assert "mxtq_bits" not in jc


def test_deepseek_sidecar_uses_jang_model_type():
    cfg, jc = build_sidecars(_deepseek_manifest([
        _tensor("embed.weight", "affine", bits=6),
        _tensor("head.weight", "affine", bits=6),
        _tensor("layers.0.attn.wq_a.weight", "affine", bits=4),
        _tensor("layers.0.ffn.shared_experts.w1.weight", "affine", bits=5),
        _tensor("layers.0.ffn.experts", "tq", bits=4, layer=0, proj="gate"),
    ]))

    assert cfg["model_type"] == "deepseek_v4"
    assert cfg["architectures"] == ["DeepseekV4ForCausalLM"]
    assert cfg["hidden_size"] == 64
    assert "text_config" not in cfg
    assert cfg["quantization"]["model.embed"] == {"bits": 6, "group_size": 128}
    assert cfg["quantization"]["lm_head"] == {"bits": 6, "group_size": 128}
    assert cfg["quantization"]["model.layers.0.self_attn.wq_a"] == {
        "bits": 4,
        "group_size": 128,
    }
    assert cfg["quantization"]["model.layers.0.mlp.shared_experts.gate_proj"] == {
        "bits": 5,
        "group_size": 128,
    }
    assert jc["quantization"]["tensor_map"]["model.layers.0.self_attn.wq_a"] == {
        "bits": 4,
        "group_size": 128,
    }
    assert jc["routed_expert_bit_plan"]["routed_layer_bits"] == {"0": {"gate": 4}}


def test_deepseek_sidecar_preserves_dense_mx_mode():
    cfg, jc = build_sidecars(_deepseek_manifest([
        _tensor("layers.0.attn.wq_a.weight", "mxfp8", bits=8, group_size=32),
    ]))

    assert cfg["quantization"]["model.layers.0.self_attn.wq_a"] == {
        "bits": 8,
        "group_size": 32,
        "mode": "mxfp8",
    }
    assert jc["quantization"]["tensor_map"]["model.layers.0.self_attn.wq_a"] == {
        "bits": 8,
        "group_size": 32,
        "mode": "mxfp8",
    }
    assert jc["quantization"]["mode_default"] == "mxfp8"


def test_deepseek_sidecar_keeps_dense_kquant_out_of_affine_quantization():
    cfg, jc = build_sidecars(_deepseek_manifest([
        _tensor("layers.0.attn.wq_a.weight", "kquant", bits=8, group_size=32,
                key_prefix="model.layers.0.self_attn.wq_a"),
        _tensor("layers.0.attn.wq_b.weight", "affine", bits=6, group_size=64),
    ]))

    assert "model.layers.0.self_attn.wq_a" not in cfg["quantization"]
    assert "model.layers.0.self_attn.wq_a" not in jc["quantization"]["tensor_map"]
    assert cfg["quantization"]["model.layers.0.self_attn.wq_b"] == {
        "bits": 6,
        "group_size": 64,
    }
    assert jc["quantization"]["tensor_map"]["model.layers.0.self_attn.wq_b"] == {
        "bits": 6,
        "group_size": 64,
    }
