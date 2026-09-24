from __future__ import annotations

from copy import deepcopy

from moespresso.inventory.qwen4.static import (
    EXPECTED_MTP_TENSOR_COUNT,
    EXPECTED_NGRAM_SHARDS,
    EXPECTED_PLE_PROVIDER_TENSOR_COUNT,
    EXPECTED_SHARD_COUNT,
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TEXT_TENSOR_COUNT,
    EXPECTED_TOTAL_SIZE,
    EXPECTED_VISION_TENSOR_COUNT,
    expected_qwen38_flash_next_header_specs,
    expected_qwen38_flash_next_ple_provider_tensors,
    expected_qwen38_flash_next_text_tensors,
    validate_qwen38_flash_next_headers,
    validate_qwen38_flash_next_static,
)
from moespresso.inventory.safetensors_header import TensorHeader


def _config() -> dict:
    text = {
        "model_type": "qwen4_exp_text",
        "dtype": "bfloat16",
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "max_position_embeddings": 262144,
        "hc_count": 4,
        "hc_lowrank": 320,
        "head_dim": 256,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.25,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "output_gate_type": "sigmoid",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "ple_layer_ids": [2],
        "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20_000_000,
        "make_ngram_vocab_size_divisible_by": 128,
        "split_ngram_parts": 128,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "vocab_size": 248320,
        "bos_token_id": 248044,
        "eos_token_id": 248044,
        "tie_word_embeddings": False,
        "layer_types": [
            kind
            for _ in range(12)
            for kind in (
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            )
        ],
        "mtp": {
            "hybrid": True,
            "layer_types": ["full_attention"],
            "num_hidden_layers": 1,
            "rope_theta": 10_000_000,
        },
    }
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": text,
    }


def _index() -> dict:
    names = set(expected_qwen38_flash_next_text_tensors())
    names.update(expected_qwen38_flash_next_ple_provider_tensors())
    mtp_names = {
        "mtp.fc_embedding.weight",
        "mtp.fc_hidden.weight",
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        "mtp.layers.0.mlp.experts.gate_up_proj",
    }
    mtp_names.update(
        f"mtp.synthetic.{index}" for index in range(EXPECTED_MTP_TENSOR_COUNT - len(mtp_names))
    )
    vision_names = {"model.visual.patch_embed.proj.weight"}
    vision_names.update(
        f"model.visual.synthetic.{index}"
        for index in range(EXPECTED_VISION_TENSOR_COUNT - len(vision_names))
    )
    names.update(mtp_names)
    names.update(vision_names)
    assert len(expected_qwen38_flash_next_text_tensors()) == EXPECTED_TEXT_TENSOR_COUNT
    assert (
        len(expected_qwen38_flash_next_ple_provider_tensors()) == EXPECTED_PLE_PROVIDER_TENSOR_COUNT
    )
    assert EXPECTED_NGRAM_SHARDS == 128
    assert len(names) == EXPECTED_TENSOR_COUNT
    ordered = sorted(names)
    weight_map = {
        name: f"model-{index % EXPECTED_SHARD_COUNT + 1:05d}-of-00131.safetensors"
        for index, name in enumerate(ordered)
    }
    return {"metadata": {"total_size": EXPECTED_TOTAL_SIZE}, "weight_map": weight_map}


def test_static_release_contract_accepts_exact_metadata_surface() -> None:
    assert validate_qwen38_flash_next_static(_config(), _index()) == []


def test_static_release_contract_resolves_omitted_seed_to_class_default() -> None:
    config = _config()
    assert "seed" not in config["text_config"]
    assert validate_qwen38_flash_next_static(config, _index()) == []


def test_static_release_contract_rejects_schedule_drift() -> None:
    config = _config()
    config["text_config"]["layer_types"][3] = "linear_attention"

    issues = validate_qwen38_flash_next_static(config, _index())

    assert any(issue.path == "/config/text_config/layer_types" for issue in issues)


def test_static_release_contract_rejects_missing_ple_buffer() -> None:
    index = deepcopy(_index())
    del index["weight_map"]["model.language_model.layers.1.ple.ple_embedding.layer_multipliers"]

    issues = validate_qwen38_flash_next_static(_config(), index)

    assert any(
        issue.code == "qwen4.missing_tensor" and "layer_multipliers" in issue.path
        for issue in issues
    )


def test_static_release_contract_rejects_missing_moe_family_tensor() -> None:
    index = deepcopy(_index())
    del index["weight_map"]["model.language_model.layers.17.mlp.shared_expert_gate.weight"]

    issues = validate_qwen38_flash_next_static(_config(), index)

    assert any(
        issue.code == "qwen4.missing_tensor" and "shared_expert_gate" in issue.path
        for issue in issues
    )


def test_static_release_contract_rejects_unknown_text_tensor_substitution() -> None:
    index = deepcopy(_index())
    del index["weight_map"]["model.language_model.layers.0.mlp.gate.weight"]
    index["weight_map"]["model.language_model.layers.0.mlp.unexpected.weight"] = (
        "model-00001-of-00131.safetensors"
    )

    issues = validate_qwen38_flash_next_static(_config(), index)

    assert any(
        issue.code == "qwen4.missing_tensor" and "mlp.gate.weight" in issue.path for issue in issues
    )
    assert any(
        issue.code == "qwen4.unknown_tensor" and "unexpected.weight" in issue.path
        for issue in issues
    )


def test_static_release_contract_rejects_qsa_geometry_drift() -> None:
    config = _config()
    config["text_config"]["indexer_compress_ratio"] = 8

    issues = validate_qwen38_flash_next_static(config, _index())

    assert any(issue.path == "/config/text_config/indexer_compress_ratio" for issue in issues)


def _header(name: str, dtype: str, shape: tuple[int, ...]) -> TensorHeader:
    return TensorHeader(name=name, dtype=dtype, shape=shape, shard="model-00001.safetensors")


def test_header_contract_covers_every_served_source_tensor() -> None:
    specs = expected_qwen38_flash_next_header_specs()

    assert len(specs) == EXPECTED_TEXT_TENSOR_COUNT + EXPECTED_PLE_PROVIDER_TENSOR_COUNT
    assert specs["model.language_model.layers.0.linear_attn.A_log"] == ("BF16", (48,))
    assert specs["model.language_model.layers.3.self_attn.q_proj.weight"] == (
        "BF16",
        (12288, 2560),
    )
    assert specs["model.language_model.layers.47.mlp.experts.gate_up_proj"] == (
        "BF16",
        (512, 1280, 2560),
    )
    assert specs[
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight"
    ] == ("BF16", (2_500_012, 160))


def test_header_contract_accepts_progressive_served_and_omitted_headers() -> None:
    specs = expected_qwen38_flash_next_header_specs()
    names = (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
    )
    headers = [_header(name, *specs[name]) for name in names]
    headers.append(_header("model.visual.patch_embed.proj.weight", "BF16", (1,)))
    headers.append(_header("mtp.fc_hidden.weight", "BF16", (1,)))

    assert validate_qwen38_flash_next_headers(headers, require_complete=False) == []


def test_header_contract_rejects_wrong_dtype_and_shape() -> None:
    name = "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"
    headers = [_header(name, "F16", (2560, 640))]

    issues = validate_qwen38_flash_next_headers(headers, require_complete=False)

    assert {issue.code for issue in issues} == {
        "qwen4.dtype_mismatch",
        "qwen4.shape_mismatch",
    }


def test_header_contract_fails_closed_on_missing_complete_tensor() -> None:
    name = "model.language_model.embed_tokens.weight"
    specs = expected_qwen38_flash_next_header_specs()

    issues = validate_qwen38_flash_next_headers([_header(name, *specs[name])])

    assert any(
        issue.code == "qwen4.missing_tensor" and issue.path == "/tensors/lm_head.weight"
        for issue in issues
    )


def test_header_contract_requires_exact_membership_in_completed_shard() -> None:
    specs = expected_qwen38_flash_next_header_specs()
    first = "model.language_model.layers.0.linear_attn.A_log"
    second = "model.language_model.layers.0.linear_attn.dt_bias"
    shard = "model-00001-of-00131.safetensors"
    index = {"weight_map": {first: shard, second: shard}}
    header = TensorHeader(name=first, dtype=specs[first][0], shape=specs[first][1], shard=shard)

    issues = validate_qwen38_flash_next_headers(
        [header],
        index=index,
        complete_shards={shard},
        require_complete=False,
    )

    assert any(
        issue.code == "qwen4.missing_shard_tensor" and second in issue.path for issue in issues
    )


def test_header_contract_rejects_tensor_in_wrong_completed_shard() -> None:
    specs = expected_qwen38_flash_next_header_specs()
    name = "model.language_model.layers.0.linear_attn.A_log"
    indexed_shard = "model-00001-of-00131.safetensors"
    actual_shard = "model-00002-of-00131.safetensors"
    index = {"weight_map": {name: indexed_shard}}
    header = TensorHeader(
        name=name,
        dtype=specs[name][0],
        shape=specs[name][1],
        shard=actual_shard,
    )

    issues = validate_qwen38_flash_next_headers(
        [header],
        index=index,
        complete_shards={actual_shard},
        require_complete=False,
    )

    assert any(issue.code == "qwen4.misplaced_tensor" for issue in issues)
    assert any(issue.code == "qwen4.unknown_shard" for issue in issues)


def test_header_contract_keeps_absent_shard_incomplete_until_final_gate() -> None:
    name = "model.language_model.layers.0.linear_attn.A_log"
    shard = "model-00001-of-00131.safetensors"
    index = {"weight_map": {name: shard}}

    progressive = validate_qwen38_flash_next_headers(
        [],
        index=index,
        complete_shards=set(),
        require_complete=False,
    )
    final = validate_qwen38_flash_next_headers(
        [],
        index=index,
        complete_shards=set(),
        require_complete=True,
    )

    assert progressive == []
    assert any(issue.code == "qwen4.missing_shard" for issue in final)
    assert any(
        issue.code == "qwen4.missing_tensor" and issue.path == f"/tensors/{name}" for issue in final
    )
