"""Generate jang-compatible sidecars (config.json, jang_config.json) from a manifest.

mjtq reuses jang's proven loader (load_jangtq_model + tensor_map override) to serve.
That loader reads `config.json` (mlx_lm config + a per-module `quantization` block so
affine non-experts build as QuantizedLinear) and `jang_config.json` (seed, per-role
bits, per-layer routed-expert bit plan, and a per-tensor `tensor_map`). These are not
source archaeology: they are a compatibility view generated from the package_manifest
(the manifest stays the source of truth). Schema matched to a proven working bundle
(Qwen3.6-35B-JANGTQ-pertensor-q95).

A pure function of the manifest. No mlx, no weight bytes, no numpy compute, just
JSON shaping. The package builders handle the file writes.
"""

from __future__ import annotations

from moespresso.inventory.deepseek_v4.roles import module_path as deepseek_v4_module_path

# Internal role -> jang's mxtq_bits role bucket. Used only to summarize per-role
# bits in jang_config (informational for the loader's banner); the authoritative
# per-tensor precision is the tensor_map / config.quantization entries.
_ROLE_BUCKET = {
    "attn.q_proj": "attention", "attn.k_proj": "attention",
    "attn.v_proj": "attention", "attn.o_proj": "attention",
    "ssm.in_proj_qkv": "linear_attention", "ssm.in_proj_z": "linear_attention",
    "ssm.in_proj_a": "linear_attention", "ssm.in_proj_b": "linear_attention",
    "ssm.out_proj": "linear_attention",
    "moe.shared_expert.gate_proj": "shared_expert",
    "moe.shared_expert.up_proj": "shared_expert",
    "moe.shared_expert.down_proj": "shared_expert",
    "embed_tokens": "embed_tokens", "lm_head": "lm_head",
}


def _module_path(source_name: str) -> str:
    """mjtq source name -> the model's sanitized module path (drop .weight)."""
    name = source_name[:-len(".weight")] if source_name.endswith(".weight") else source_name
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    if name.startswith("lm_head"):
        return "language_model.lm_head" + name[len("lm_head"):]
    return name


def _deepseek_v4_module_path(source_name: str) -> str:
    """DS4 source name -> JANG DS4 graph module path (drop tensor suffix)."""
    return deepseek_v4_module_path(source_name)


def _default_dense_quant(affine_alloc: dict[str, dict]) -> tuple[str, int, int]:
    pairs = [
        (str(v.get("mode", "affine")), int(v["bits"]), int(v["group_size"]))
        for v in affine_alloc.values()
    ]
    if not pairs:
        return "affine", 4, 128
    return max(sorted(set(pairs)), key=pairs.count)


def _dense_affine_alloc(tensors: list[dict], *, family: str | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in tensors:
        if t["format"] not in {"affine", "mxfp4", "mxfp8"}:
            continue
        path = (
            _deepseek_v4_module_path(t["source_name"])
            if family == "deepseek_v4_flash"
            else _module_path(t["source_name"])
        )
        alloc = {
            "bits": int(t["format_params"]["bits"]),
            "group_size": int(t["format_params"]["group_size"]),
        }
        if t["format"] != "affine":
            alloc["mode"] = t["format"]
        out[path] = alloc
    return out


def _build_dense_affine_sidecars(manifest: dict) -> tuple[dict, dict]:
    arch = manifest["architecture"]
    text_config = dict(arch["config"])
    wrapper_model_type = arch.get("wrapper_model_type") or "qwen3_5"
    affine_alloc = _dense_affine_alloc(manifest["tensors"], family=arch.get("family"))
    default_mode, default_bits, default_gs = _default_dense_quant(affine_alloc)

    config_quant = {"group_size": default_gs, "bits": default_bits}
    if default_mode != "affine":
        config_quant["mode"] = default_mode
    for path, alloc in sorted(affine_alloc.items()):
        if (
            alloc["bits"] != default_bits
            or alloc["group_size"] != default_gs
            or alloc.get("mode", "affine") != default_mode
        ):
            config_quant[path] = alloc

    bit_widths = sorted({v["bits"] for v in affine_alloc.values()}) or [default_bits]
    actual_bits = (
        sum(v["bits"] for v in affine_alloc.values()) / len(affine_alloc)
        if affine_alloc else float(default_bits)
    )
    config_json = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": wrapper_model_type,
        "text_config": text_config,
        "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
        "language_model_only": True,
        "quantization": config_quant,
    }
    jang_config = {
        "format": "jang",
        "format_version": "2.0",
        "source_model": {"name": "moespresso"},
        "quantization": {
            "method": "affine",
            "mode": default_mode,
            "bits": default_bits,
            "group_size": default_gs,
            "block_size": default_gs,
            "bit_widths_used": bit_widths,
            "actual_bits": actual_bits,
            "per_tensor": dict(sorted(affine_alloc.items())),
        },
    }
    return config_json, jang_config


def build_sidecars(manifest: dict, *, seed: int = 42) -> tuple[dict, dict]:
    """(config.json dict, jang_config.json dict) from the package_manifest.

    Affine tensors -> per-module {bits, group_size} in config.quantization and
    jang tensor_map. TQ experts -> routed_expert_bit_plan.routed_layer_bits and a
    tensor_map entry per group. fp16 passthrough (router/norms/ssm) is unquantized,
    so it gets no quantization entry (the loader keeps it fp16).
    """
    arch = manifest["architecture"]
    if arch.get("family") == "qwen3_5_dense":
        return _build_dense_affine_sidecars(manifest)

    text_config = dict(arch["config"])
    tensors = manifest["tensors"]

    # Affine/MX per-tensor allocation: module path -> {bits, group_size}. This is
    # the only thing tensor_map / config.quantization carry: JANG consumes it to
    # pin affine QuantizedLinear precision. TQ experts and dense K-quant modules
    # are installed by their own manifest-driven paths, so they must not leak into
    # this map and be mutated as affine quantized linears.
    affine_alloc: dict[str, dict] = {}    # sanitized module path -> quantized-linear config
    routed_layer_bits: dict[str, dict] = {}  # str(layer) -> {gate,up,down: bits}

    for t in tensors:
        fmt = t["format"]
        if fmt in {"affine", "mxfp4", "mxfp8"}:
            path = (
                _deepseek_v4_module_path(t["source_name"])
                if arch["family"] == "deepseek_v4_flash"
                else _module_path(t["source_name"])
            )
            alloc = {"bits": int(t["format_params"]["bits"]),
                     "group_size": int(t["format_params"]["group_size"])}
            if fmt != "affine":
                alloc["mode"] = fmt
            affine_alloc[path] = alloc
        elif fmt == "tq":
            layer = str(t.get("layer_index"))
            proj = t.get("projection")  # gate / up / down
            routed_layer_bits.setdefault(layer, {})[proj] = int(t["format_params"]["bits"])
        # fp16 passthrough: not quantized -> no entry.

    # Defaults = the mode (most common) of the affine allocation, like convert_moe.
    all_bits = [v["bits"] for v in affine_alloc.values()]
    all_gs = [v["group_size"] for v in affine_alloc.values()]
    all_modes = [v.get("mode", "affine") for v in affine_alloc.values()]
    default_bits = max(set(all_bits), key=all_bits.count) if all_bits else 4
    default_gs = max(set(all_gs), key=all_gs.count) if all_gs else 128
    default_mode = max(set(all_modes), key=all_modes.count) if all_modes else "affine"
    expert_bits = [b for lb in routed_layer_bits.values() for b in lb.values()]
    default_expert_bits = (max(set(expert_bits), key=expert_bits.count)
                           if expert_bits else 2)

    # config.quantization: top-level default + only the modules that differ from it
    # (mlx_lm's class_predicate applies these per-module overrides).
    config_quant = {"group_size": default_gs, "bits": default_bits}
    if default_mode != "affine":
        config_quant["mode"] = default_mode
    for path, alloc in sorted(affine_alloc.items()):
        if (
            arch["family"] == "deepseek_v4_flash"
            or alloc["bits"] != default_bits
            or alloc["group_size"] != default_gs
            or alloc.get("mode", "affine") != default_mode
        ):
            config_quant[path] = alloc

    is_deepseek_v4 = arch["family"] == "deepseek_v4_flash"
    model_type = "deepseek_v4" if is_deepseek_v4 else arch["family"]
    default_architectures = (
        ["DeepseekV4ForCausalLM"] if is_deepseek_v4 else ["Qwen3_5_MoeForCausalLM"])
    if is_deepseek_v4:
        config_json = {
            **text_config,
            "architectures": text_config.get("architectures", default_architectures),
            "model_type": model_type,
            "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
            "weight_format": "mxtq",
            "mxtq_seed": seed,
            "mxtq_bits": default_expert_bits,
            "quantization": config_quant,
        }
    else:
        config_json = {
            "architectures": text_config.get("architectures", default_architectures),
            "model_type": model_type,
            "text_config": text_config,
            "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
            "language_model_only": True,
            "weight_format": "mxtq",
            "mxtq_seed": seed,
            # config.json mxtq_bits is a single int (convert_moe shape). Not authoritative
            # for compute: the loader reads per-tensor bits from on-disk tq_bits +
            # tensor_map; this field is only a banner. Kept as the default expert bits.
            "mxtq_bits": default_expert_bits,
            "quantization": config_quant,
        }
    jang_config = {
        "version": 2,
        "weight_format": "mxtq",
        "profile": "MOESPRESSO_MOE",
        "mxtq_seed": seed,
        # Per-role summary, verified to be the loader banner only: jang's _hydrate
        # never reads it (bits come from on-disk tq_bits); the affine pin reads
        # tensor_map. Use zero as an explicitly non-authoritative sentinel because
        # the real per-tensor bits live in on-disk tq_bits.
        "mxtq_bits": {
            "attention": 0, "linear_attention": 0, "shared_expert": 0,
            "embed_tokens": 0, "lm_head": 0, "routed_expert": 0,
        },
        "routed_expert_bit_plan": {"routed_layer_bits": routed_layer_bits},
        "quantization": {
            "method": "affine+mxtq",
            "mode_default": default_mode,
            "group_size": default_gs,
            "bits_default": default_bits,
            "per_tensor": dict(sorted(affine_alloc.items())),
            "tensor_map": dict(sorted(affine_alloc.items())),  # affine-only
        },
    }
    return config_json, jang_config
