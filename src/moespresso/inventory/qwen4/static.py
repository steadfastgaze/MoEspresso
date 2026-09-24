"""Static checks for the released Qwen3.8-Flash-Next checkpoint."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from moespresso.core.artifact import Validation
from moespresso.inventory.safetensors_header import TensorHeader


EXPECTED_TOTAL_SIZE = 359_999_963_128
EXPECTED_SHARD_COUNT = 131
EXPECTED_TENSOR_COUNT = 1_658
EXPECTED_NGRAM_SHARDS = 128
EXPECTED_TEXT_TENSOR_COUNT = 1_163
EXPECTED_PLE_PROVIDER_TENSOR_COUNT = 131
EXPECTED_VISION_TENSOR_COUNT = 333
EXPECTED_MTP_TENSOR_COUNT = 31

_EXPECTED_LAYER_TYPES = [
    kind
    for _ in range(12)
    for kind in ("linear_attention", "linear_attention", "linear_attention", "full_attention")
]

_TOP_LEVEL_FACTS = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "model_type": "qwen4_exp",
}

_TEXT_FACTS = {
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
}

_MTP_FACTS = {
    "hybrid": True,
    "layer_types": ["full_attention"],
    "num_hidden_layers": 1,
    "rope_theta": 10_000_000,
}

_REQUIRED_OMITTED_TENSORS = {
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "model.visual.patch_embed.proj.weight",
}

_PLE_PROVIDER_METADATA = {
    "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
}

_GLOBAL_TEXT_TENSORS = {
    "model.language_model.embed_tokens.weight",
    "model.language_model.hyper_connection_mixer.hc_norm.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    "lm_head.weight",
}

_GDN_TENSOR_SUFFIXES = {
    "A_log",
    "conv1d.weight",
    "dt_bias",
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "norm.weight",
    "out_proj.weight",
}

_QSA_TENSOR_SUFFIXES = {
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "q_norm.weight",
    "k_norm.weight",
    "indexer.index_qk_proj.weight",
    "indexer.q_layernorm.weight",
    "indexer.k_layernorm.weight",
}

_HYPER_CONNECTION_SUFFIXES = {
    "hc_norm.weight",
    "input_mix_weight_down.weight",
    "input_mix_weight_up.weight",
    "block_inject_weight.weight",
}

_MOE_TENSOR_SUFFIXES = {
    "experts.gate_up_proj",
    "experts.down_proj",
    "gate.weight",
    "shared_expert.gate_proj.weight",
    "shared_expert.up_proj.weight",
    "shared_expert.down_proj.weight",
    "shared_expert_gate.weight",
}

_PLE_GRAPH_TENSORS = {
    "model.language_model.layers.1.ple.key_proj.weight",
    "model.language_model.layers.1.ple.value_proj.weight",
    "model.language_model.layers.1.ple.conv1d.weight",
    "model.language_model.layers.1.ple.norm_key.weight",
    "model.language_model.layers.1.ple.norm_query.weight",
    "model.language_model.layers.1.ple.norm_conv.weight",
}


def expected_qwen38_flash_next_text_tensors() -> frozenset[str]:
    """Return the exact released text compute-graph tensor identities."""
    names = set(_GLOBAL_TEXT_TENSORS)
    names.update(_PLE_GRAPH_TENSORS)
    for layer in range(48):
        layer_prefix = f"model.language_model.layers.{layer}"
        for seam in ("attn_hyper_connection", "mlp_hyper_connection"):
            names.update(f"{layer_prefix}.{seam}.{suffix}" for suffix in _HYPER_CONNECTION_SUFFIXES)
        names.update(f"{layer_prefix}.mlp.{suffix}" for suffix in _MOE_TENSOR_SUFFIXES)
        if layer % 4 == 3:
            names.update(f"{layer_prefix}.self_attn.{suffix}" for suffix in _QSA_TENSOR_SUFFIXES)
        else:
            names.update(f"{layer_prefix}.linear_attn.{suffix}" for suffix in _GDN_TENSOR_SUFFIXES)
    if len(names) != EXPECTED_TEXT_TENSOR_COUNT:
        raise AssertionError("Qwen4 text tensor contract has an internal count mismatch")
    return frozenset(names)


def expected_qwen38_flash_next_ple_provider_tensors() -> frozenset[str]:
    """Return the exact PLE lookup metadata and physical table tensors."""
    names = set(_PLE_PROVIDER_METADATA)
    names.update(
        f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        for shard in range(EXPECTED_NGRAM_SHARDS)
    )
    return frozenset(names)


def expected_qwen38_flash_next_header_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    """Return dtype and shape contracts for every served source tensor."""
    specs: dict[str, tuple[str, tuple[int, ...]]] = {
        "model.language_model.embed_tokens.weight": ("BF16", (248320, 2560)),
        "model.language_model.hyper_connection_mixer.hc_norm.weight": ("BF16", (10240,)),
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": (
            "BF16",
            (320, 10240),
        ),
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": (
            "BF16",
            (10240, 320),
        ),
        "lm_head.weight": ("BF16", (248320, 2560)),
        "model.language_model.layers.1.ple.key_proj.weight": ("BF16", (10240, 2560)),
        "model.language_model.layers.1.ple.value_proj.weight": ("BF16", (2560, 2560)),
        "model.language_model.layers.1.ple.conv1d.weight": ("BF16", (10240, 1, 4)),
        "model.language_model.layers.1.ple.norm_key.weight": ("BF16", (10240,)),
        "model.language_model.layers.1.ple.norm_query.weight": ("BF16", (10240,)),
        "model.language_model.layers.1.ple.norm_conv.weight": ("BF16", (10240,)),
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": ("I64", (3,)),
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes": (
            "I64",
            (16,),
        ),
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets": ("I64", (16,)),
    }
    for shard in range(EXPECTED_NGRAM_SHARDS):
        specs[
            f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        ] = ("BF16", (2_500_012, 160))

    hyper_connection_specs = {
        "hc_norm.weight": (10240,),
        "input_mix_weight_down.weight": (320, 10240),
        "input_mix_weight_up.weight": (10240, 320),
        "block_inject_weight.weight": (4, 10240),
    }
    moe_specs = {
        "experts.gate_up_proj": (512, 1280, 2560),
        "experts.down_proj": (512, 2560, 640),
        "gate.weight": (512, 2560),
        "shared_expert.gate_proj.weight": (640, 2560),
        "shared_expert.up_proj.weight": (640, 2560),
        "shared_expert.down_proj.weight": (2560, 640),
        "shared_expert_gate.weight": (1, 2560),
    }
    gdn_specs = {
        "A_log": (48,),
        "conv1d.weight": (10240, 1, 4),
        "dt_bias": (48,),
        "in_proj_qkv.weight": (10240, 2560),
        "in_proj_z.weight": (6144, 2560),
        "in_proj_a.weight": (48, 2560),
        "in_proj_b.weight": (48, 2560),
        "norm.weight": (128,),
        "out_proj.weight": (2560, 6144),
    }
    qsa_specs = {
        "q_proj.weight": (12288, 2560),
        "k_proj.weight": (512, 2560),
        "v_proj.weight": (512, 2560),
        "o_proj.weight": (2560, 6144),
        "q_norm.weight": (256,),
        "k_norm.weight": (256,),
        "indexer.index_qk_proj.weight": (640, 2560),
        "indexer.q_layernorm.weight": (128,),
        "indexer.k_layernorm.weight": (128,),
    }
    for layer in range(48):
        layer_prefix = f"model.language_model.layers.{layer}"
        for seam in ("attn_hyper_connection", "mlp_hyper_connection"):
            for suffix, shape in hyper_connection_specs.items():
                specs[f"{layer_prefix}.{seam}.{suffix}"] = ("BF16", shape)
        for suffix, shape in moe_specs.items():
            specs[f"{layer_prefix}.mlp.{suffix}"] = ("BF16", shape)
        mixer = "self_attn" if layer % 4 == 3 else "linear_attn"
        mixer_specs = qsa_specs if mixer == "self_attn" else gdn_specs
        for suffix, shape in mixer_specs.items():
            specs[f"{layer_prefix}.{mixer}.{suffix}"] = ("BF16", shape)

    expected_names = (
        expected_qwen38_flash_next_text_tensors()
        | expected_qwen38_flash_next_ple_provider_tensors()
    )
    if specs.keys() != expected_names:
        raise AssertionError("Qwen4 header contract does not match the served tensor identities")
    return specs


def _error(code: str, message: str, *, path: str, expected=None, actual=None) -> Validation:
    return Validation(
        "error",
        code,
        message,
        path=path,
        phase="qwen4_static",
        blocking=True,
        expected=expected,
        actual=actual,
    )


def _check_equal(out: list[Validation], path: str, actual, expected) -> None:
    if actual != expected:
        out.append(
            _error(
                "qwen4.static_mismatch",
                f"{path} is {actual!r}, expected {expected!r}",
                path=path,
                expected=expected,
                actual=actual,
            )
        )


def _count_suffix(names: set[str], suffix: str) -> int:
    return sum(name.endswith(suffix) for name in names)


def validate_qwen38_flash_next_static(config: dict, index: dict) -> list[Validation]:
    """Validate released config and index facts without reading tensor payloads."""
    out: list[Validation] = []
    for key, expected in _TOP_LEVEL_FACTS.items():
        _check_equal(out, f"/config/{key}", config.get(key), expected)

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return [
            _error(
                "qwen4.missing_text_config",
                "released config has no text_config object",
                path="/config/text_config",
                expected="object",
                actual=type(text_config).__name__,
            )
        ]
    for key, expected in _TEXT_FACTS.items():
        _check_equal(out, f"/config/text_config/{key}", text_config.get(key), expected)
    _check_equal(
        out,
        "/config/text_config/layer_types",
        text_config.get("layer_types"),
        _EXPECTED_LAYER_TYPES,
    )
    _check_equal(out, "/config/text_config/seed", text_config.get("seed", 1234), 1234)
    _check_equal(
        out,
        "/config/text_config/norm_topk_prob",
        text_config.get("norm_topk_prob", True),
        True,
    )

    mtp = text_config.get("mtp")
    if not isinstance(mtp, dict):
        out.append(
            _error(
                "qwen4.missing_mtp_config",
                "released text config has no mtp object",
                path="/config/text_config/mtp",
                expected="object",
                actual=type(mtp).__name__,
            )
        )
    else:
        for key, expected in _MTP_FACTS.items():
            _check_equal(out, f"/config/text_config/mtp/{key}", mtp.get(key), expected)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        out.append(
            _error(
                "qwen4.missing_weight_map",
                "released safetensors index has no weight_map object",
                path="/index/weight_map",
                expected="object",
                actual=type(weight_map).__name__,
            )
        )
        return out
    names = set(weight_map)
    expected_text = set(expected_qwen38_flash_next_text_tensors())
    expected_provider = set(expected_qwen38_flash_next_ple_provider_tensors())
    vision_names = {name for name in names if name.startswith("model.visual.")}
    mtp_names = {name for name in names if name.startswith("mtp.")}
    text_names = names & expected_text
    provider_names = names & expected_provider
    unknown_names = names - text_names - provider_names - vision_names - mtp_names
    _check_equal(out, "/index/tensor_count", len(names), EXPECTED_TENSOR_COUNT)
    _check_equal(out, "/index/shard_count", len(set(weight_map.values())), EXPECTED_SHARD_COUNT)
    _check_equal(
        out,
        "/index/metadata/total_size",
        index.get("metadata", {}).get("total_size"),
        EXPECTED_TOTAL_SIZE,
    )
    _check_equal(
        out, "/index/gdn_layer_count", _count_suffix(names, ".linear_attn.in_proj_qkv.weight"), 36
    )
    _check_equal(
        out,
        "/index/qsa_layer_count",
        _count_suffix(names, ".self_attn.indexer.index_qk_proj.weight") - 1,
        12,
    )
    _check_equal(
        out, "/index/expert_layer_count", _count_suffix(names, ".mlp.experts.gate_up_proj") - 1, 48
    )
    _check_equal(out, "/index/text_tensor_count", len(text_names), EXPECTED_TEXT_TENSOR_COUNT)
    _check_equal(
        out,
        "/index/ple_provider_tensor_count",
        len(provider_names),
        EXPECTED_PLE_PROVIDER_TENSOR_COUNT,
    )
    _check_equal(out, "/index/vision_tensor_count", len(vision_names), EXPECTED_VISION_TENSOR_COUNT)
    _check_equal(out, "/index/mtp_tensor_count", len(mtp_names), EXPECTED_MTP_TENSOR_COUNT)

    required = expected_text | expected_provider | _REQUIRED_OMITTED_TENSORS
    for name in sorted(required - names):
        out.append(
            _error(
                "qwen4.missing_tensor",
                f"{name} is missing from the safetensors index",
                path=f"/index/weight_map/{name}",
                expected="tensor entry",
                actual=None,
            )
        )
    for name in sorted(unknown_names):
        out.append(
            _error(
                "qwen4.unknown_tensor",
                f"{name} is outside the released text, PLE, vision, and MTP partitions",
                path=f"/index/weight_map/{name}",
                expected="known Qwen4 tensor identity",
                actual=name,
            )
        )
    return out


def validate_qwen38_flash_next_headers(
    headers: Sequence[TensorHeader],
    *,
    index: dict | None = None,
    complete_shards: Collection[str] = (),
    require_complete: bool = True,
) -> list[Validation]:
    """Validate served tensor headers without reading tensor payload bytes."""
    out: list[Validation] = []
    specs = expected_qwen38_flash_next_header_specs()
    by_name: dict[str, TensorHeader] = {}
    for header in headers:
        if header.name in by_name:
            out.append(
                _error(
                    "qwen4.duplicate_tensor",
                    f"{header.name} occurs in more than one safetensors header",
                    path=f"/tensors/{header.name}",
                    expected="one tensor header",
                    actual="duplicate",
                )
            )
            continue
        by_name[header.name] = header
        expected = specs.get(header.name)
        if expected is None:
            if header.name.startswith("model.visual.") or header.name.startswith("mtp."):
                continue
            out.append(
                _error(
                    "qwen4.unknown_tensor",
                    f"{header.name} is outside the released tensor partition",
                    path=f"/tensors/{header.name}",
                    expected="known Qwen4 tensor identity",
                    actual=header.name,
                )
            )
            continue
        expected_dtype, expected_shape = expected
        if header.dtype != expected_dtype:
            out.append(
                _error(
                    "qwen4.dtype_mismatch",
                    f"{header.name} dtype is {header.dtype}, expected {expected_dtype}",
                    path=f"/tensors/{header.name}/dtype",
                    expected=expected_dtype,
                    actual=header.dtype,
                )
            )
        if header.shape != expected_shape:
            out.append(
                _error(
                    "qwen4.shape_mismatch",
                    f"{header.name} shape is {list(header.shape)}, expected {list(expected_shape)}",
                    path=f"/tensors/{header.name}/shape",
                    expected=list(expected_shape),
                    actual=list(header.shape),
                )
            )

    if index is not None:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            out.append(
                _error(
                    "qwen4.missing_weight_map",
                    "released safetensors index has no weight_map object",
                    path="/index/weight_map",
                    expected="object",
                    actual=type(weight_map).__name__,
                )
            )
            return out
        indexed_shards = set(weight_map.values())
        completed = set(complete_shards)
        for shard in sorted(completed - indexed_shards):
            out.append(
                _error(
                    "qwen4.unknown_shard",
                    f"{shard} is not declared by the safetensors index",
                    path=f"/shards/{shard}",
                    expected="indexed shard",
                    actual=shard,
                )
            )
        for header in headers:
            indexed_shard = weight_map.get(header.name)
            if indexed_shard is not None and header.shard != indexed_shard:
                out.append(
                    _error(
                        "qwen4.misplaced_tensor",
                        f"{header.name} occurs in {header.shard}, expected {indexed_shard}",
                        path=f"/tensors/{header.name}/shard",
                        expected=indexed_shard,
                        actual=header.shard,
                    )
                )
        names_by_shard: dict[str, set[str]] = {}
        for header in headers:
            names_by_shard.setdefault(header.shard, set()).add(header.name)
        indexed_names_by_shard: dict[str, set[str]] = {}
        for name, shard in weight_map.items():
            indexed_names_by_shard.setdefault(shard, set()).add(name)
        for shard in sorted(completed & indexed_shards):
            actual_names = names_by_shard.get(shard, set())
            expected_names = indexed_names_by_shard[shard]
            for name in sorted(expected_names - actual_names):
                out.append(
                    _error(
                        "qwen4.missing_shard_tensor",
                        f"{name} is missing from completed shard {shard}",
                        path=f"/shards/{shard}/{name}",
                        expected="tensor header",
                        actual=None,
                    )
                )
            for name in sorted(actual_names - expected_names):
                out.append(
                    _error(
                        "qwen4.unindexed_shard_tensor",
                        f"{name} is not assigned to completed shard {shard}",
                        path=f"/shards/{shard}/{name}",
                        expected="index membership",
                        actual=shard,
                    )
                )
        if require_complete:
            for shard in sorted(indexed_shards - completed):
                out.append(
                    _error(
                        "qwen4.missing_shard",
                        f"indexed shard {shard} is not complete",
                        path=f"/shards/{shard}",
                        expected="completed shard",
                        actual=None,
                    )
                )

    if require_complete:
        for name in sorted(specs.keys() - by_name.keys()):
            expected_dtype, expected_shape = specs[name]
            out.append(
                _error(
                    "qwen4.missing_tensor",
                    f"{name} is missing from safetensors headers",
                    path=f"/tensors/{name}",
                    expected={"dtype": expected_dtype, "shape": list(expected_shape)},
                    actual=None,
                )
            )
    return out
