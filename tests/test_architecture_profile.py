"""architecture_profile: the model-family correctness contract.

These tests pin the contract a package must satisfy: the qwen3_5_moe profile
declares text-only scope, vision/mtp exclusions, TQ experts + mixed-affine
non-experts, the fused gate_up split, and (critically) the conv1d/norm-shift
transform contract whose violation makes the artifact incoherent. The schema
stays generic; a minimal synthetic profile proves family-neutrality.
"""

from __future__ import annotations

from moespresso.core.artifact import validate_base
from moespresso.inventory.architecture_profile import (
    DEEPSEEK_V4_FLASH_COMPRESS_RATIOS,
    deepseek_v4_flash_profile,
    family_of,
    profile_for,
    qwen3_5_dense_profile,
    qwen3_5_moe_profile,
    synthetic_profile,
)


def test_qwen_profile_is_a_valid_architecture_profile_artifact():
    p = qwen3_5_moe_profile()
    assert p["artifact_kind"] == "architecture_profile"
    assert p["artifact_id"].startswith("arch:")
    assert validate_base(p) == []
    assert p["family"] == "qwen3_5_moe"


def test_qwen_profile_declares_text_only_with_vision_mtp_excluded():
    p = qwen3_5_moe_profile()
    assert p["modality"] == "text"
    # Must match the actual Qwen vision prefix model.visual.* (not just "vision"), in lockstep
    # with the inventory SSOT (inventory/roles.py excludes "visual"/"vision"/"mtp.").
    assert "visual" in p["excluded_namespaces"]
    assert "vision" in p["excluded_namespaces"]
    assert "mtp" in p["excluded_namespaces"]


def test_qwen_profile_declares_expert_and_nonexpert_quant_contract():
    p = qwen3_5_moe_profile()
    roles = p["role_quant"]
    # routed experts -> TQ; non-expert projections (incl. SSM in_proj) -> affine;
    # routing gates -> fp16 passthrough.
    assert roles["moe.expert"] == "tq"
    assert roles["ssm.in_proj_a"] == "affine" and roles["ssm.in_proj_b"] == "affine"
    assert roles["attn.q_proj"] == "affine"
    assert roles["moe.router_gate"] == "fp16"


def test_qwen_profile_declares_conv1d_norm_shift_transform_contract():
    # The coupled-transform bug class: conv1d must be stored source [out,1,k] so the
    # runtime sanitizer fires both the conv transpose and the +1.0 RMSNorm shift.
    # Storing conv1d pre-transposed suppresses the coupled shift, so norms load ~1.0
    # too low and the model emits garbage. The profile must declare this coupling so
    # L1 can derive the expected (shifted) norm values.
    p = qwen3_5_moe_profile()
    transforms = {t["name"]: t for t in p["transforms"]}
    conv = transforms["conv1d_layout"]
    assert conv["store_shape"] == "source"            # preserve source [out,1,k]
    assert conv["sanitizer_trigger"]["last_dim_not"] == 1   # structured, machine-checkable
    norm = transforms["rmsnorm_shift"]
    assert norm["delta"] == 1.0                        # runtime norm = source + 1.0
    assert norm["coupled_to"] == "conv1d_layout"       # the coupling behind the bug class
    # storage vs runtime are separate: the trap to respect is rejecting a correctly
    # stored package whose norm is unshifted on disk:
    assert norm["norm_storage"] == "unshifted"         # package stores source norm as-is
    assert norm["runtime_relation"] == "source_plus_delta"
    assert norm["required"] is True


def test_qwen_profile_declares_fused_gate_up_split():
    p = qwen3_5_moe_profile()
    transforms = {t["name"]: t for t in p["transforms"]}
    assert transforms["expert_fused_gate_up"]["splits_into"] == ["gate", "up"]


def test_qwen_profile_requires_l0_l1_rungs():
    p = qwen3_5_moe_profile()
    assert "L0" in p["required_rungs"] and "L1" in p["required_rungs"]


def test_synthetic_profile_is_valid_and_generic():
    # Framework-shape proof: a minimal family with no experts, no SSM: the schema
    # must accept its generic contract without assuming Qwen.
    p = synthetic_profile()
    assert validate_base(p) == []
    assert p["artifact_kind"] == "architecture_profile"
    assert p["family"] == "synthetic_dense"
    assert p["role_quant"]  # has some declared roles


def test_deepseek_v4_flash_profile_declares_length_44_compression_contract():
    p = deepseek_v4_flash_profile()
    assert validate_base(p) == []
    assert p["family"] == "deepseek_v4_flash"
    assert p["modality"] == "text"
    assert p["compress_ratios"] == DEEPSEEK_V4_FLASH_COMPRESS_RATIOS
    assert len(p["compress_ratios"]) == 44
    assert len(p["layer_kinds"]) == 43
    assert p["compress_ratios"][42] == 4
    assert p["layer_kinds"][42] == "csa"
    assert p["compress_ratios"][43] == 0
    assert p["mtp_compress_ratio"] == 0


def test_deepseek_v4_flash_profile_declares_quant_ownership_and_cache_policy():
    p = deepseek_v4_flash_profile()
    roles = p["role_quant"]
    assert roles["moe.expert"] == "tq"
    assert roles["attn.wq_a"] == "affine"
    assert roles["attn.compressor.wkv"] == "affine"
    assert roles["moe.shared_expert.down_proj"] == "affine"
    assert roles["moe.router_gate"] == "fp16"
    assert roles["moe.router_bias"] == "raw_dtype_passthrough"
    assert roles["moe.router_tid2eid"] == "raw_dtype_passthrough"
    assert roles["hc.control"] == "raw_dtype_passthrough"
    assert p["cache_policy"] == {"kind": "deepseek_v4_composite", "generic_kv_bits": False}


def test_family_resolution_recognizes_deepseek_v4_without_touching_qwen():
    config = {"model_type": "deepseek_v4", "num_hidden_layers": 43}
    assert family_of(config) == "deepseek_v4_flash"
    assert profile_for(config)["family"] == "deepseek_v4_flash"

    qwen = {"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_moe_text"}}
    assert family_of(qwen) == "qwen3_5_moe"


def _dense_qwen35_config():
    return {
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
        },
    }


def test_qwen35_dense_profile_is_valid_and_text_only():
    p = qwen3_5_dense_profile()
    assert validate_base(p) == []
    assert p["artifact_kind"] == "architecture_profile"
    assert p["family"] == "qwen3_5_dense"
    assert p["modality"] == "text"
    assert "visual" in p["excluded_namespaces"]
    assert "vision" in p["excluded_namespaces"]
    assert "mtp" in p["excluded_namespaces"]


def test_qwen35_dense_profile_is_affine_hybrid_without_expert_contracts():
    p = qwen3_5_dense_profile()
    roles = p["role_quant"]
    assert roles["ffn.gate_proj"] == "affine"
    assert roles["ffn.up_proj"] == "affine"
    assert roles["ffn.down_proj"] == "affine"
    assert roles["ssm.in_proj_a"] == "affine"
    assert roles["ssm.in_proj_b"] == "affine"
    assert "moe.expert" not in roles
    assert p["router"]["top_k_present"] is False
    assert set(p["layer_kinds"]) == {"linear_attention", "full_attention"}
    transforms = {t["name"]: t for t in p["transforms"]}
    assert "conv1d_layout" in transforms
    assert "rmsnorm_shift" in transforms
    assert "expert_fused_gate_up" not in transforms


def test_family_resolution_distinguishes_dense_qwen35_from_moe_and_qwen3():
    dense = _dense_qwen35_config()
    assert family_of(dense) == "qwen3_5_dense"
    assert profile_for(dense)["family"] == "qwen3_5_dense"

    moe = {"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_moe_text"}}
    assert family_of(moe) == "qwen3_5_moe"

    plain_qwen3 = {"model_type": "qwen3", "num_hidden_layers": 4}
    assert family_of(plain_qwen3) is None
