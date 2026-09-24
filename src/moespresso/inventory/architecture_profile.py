"""architecture_profile: the model-family correctness contract.

A profile declares what a package of a given family must satisfy, so the correctness
ladder (L0-L4) can check the package against the contract rather than against a single
serve. The profile is generic: roles, quant ownership, source->runtime transforms,
layer kinds, exclusions, and the minimum ladder rungs are all declared per family. New
families add a profile; the schema does not change.

The qwen3_5_moe profile below comes directly from the model's facts. It does not
infer them from a working artifact. In particular it declares the conv1d/norm-shift transform contract whose violation produces
garbage output: conv1d.weight stored in source shape [out,1,k] is the predicate mlx_lm's
qwen3_5 sanitize uses to fire both the conv transpose and the +1.0 RMSNorm shift; storing
it pre-transposed [out,k,1] suppresses the shift -> norms ~1.0 too low -> garbage. L1
reconstruction derives the expected (shifted) norm from this declaration.

Pure data + builders; no model, no I/O. Emitted as a standalone `architecture_profile`
artifact (not wired into convert/serve/verify yet; the gate comes later).
"""

from __future__ import annotations

from moespresso.core.artifact import artifact_producer, make_artifact

PRODUCER = artifact_producer("moespresso.architecture_profile")


def _profile(subject: dict, **fields) -> dict:
    """An architecture_profile artifact (generic; fields are the family's contract)."""
    return make_artifact("architecture_profile", subject, PRODUCER, status="valid", **fields)


def qwen3_5_moe_profile() -> dict:
    """The contract a qwen3_5_moe (Qwen3.5/3.6-A3B) text package must satisfy.

    Declares: text-only scope + vision/mtp exclusions; per-role quant ownership (routed
    experts, affine non-experts incl. SSM in_proj, fp16 routing gates); the source->
    runtime transforms (conv1d layout, the coupled RMSNorm +1.0 shift, fused gate_up
    split, the model.language_model.* -> language_model.model.* rename); layer kinds;
    and the minimum ladder rungs required before a full package is allowed.
    """
    return _profile(
        {"family": "qwen3_5_moe", "modality": "text"},
        family="qwen3_5_moe",
        modality="text",
        excluded_namespaces={
            # Real Qwen vision tensors are model.visual.*, the namespace token must match
            # that prefix (not just "vision"), and stay in lockstep with the inventory SSOT
            # (inventory/roles.py excludes "visual"/"vision"/"mtp.").
            "visual": "text-only package; vision tower (model.visual.*) not served",
            "vision": "text-only package; any vision-tower namespace not served",
            "mtp": "multi-token-prediction draft layers; not part of the base text forward "
                   "(and mtp presence is itself a sanitize norm-shift trigger, excluding it "
                   "makes the conv1d-shape predicate the sole shift trigger)",
        },
        # Per typed-role quant ownership (the inventory's role vocabulary). This is the
        # contract: routed experts use their declared codecs, dense projections
        # have affine ownership, and discrete-routing gates stay fp16.
        role_quant={
            "moe.expert": "routed_expert",
            "moe.router_gate": "fp16",
            "moe.shared_expert_gate": "fp16",
            "moe.shared_expert.gate_proj": "affine",
            "moe.shared_expert.up_proj": "affine",
            "moe.shared_expert.down_proj": "affine",
            "attn.q_proj": "affine", "attn.k_proj": "affine",
            "attn.v_proj": "affine", "attn.o_proj": "affine",
            "ssm.in_proj_qkv": "affine", "ssm.in_proj_z": "affine",
            "ssm.in_proj_a": "affine", "ssm.in_proj_b": "affine",
            "ssm.out_proj": "affine",
            "embed_tokens": "affine", "lm_head": "affine",
        },
        # Structural tensors carried verbatim (fp16); the graph needs them; they are not
        # quant choices. conv1d's stored shape is load-bearing (see transforms).
        structural_passthrough=[
            "norm.input_layernorm", "norm.post_attention_layernorm", "norm.final",
            "norm.attn.q_norm", "norm.attn.k_norm", "norm.ssm.norm",
            "ssm.A_log", "ssm.dt_bias", "ssm.conv1d",
        ],
        transforms=[
            {"name": "key_rename",
             "from": "model.language_model.*", "to": "language_model.model.*",
             "applied_by": "runtime sanitize (mlx_lm qwen3_5_moe.Model.sanitize)"},
            {"name": "expert_fused_gate_up",
             "source": "mlp.experts.gate_up_proj", "splits_into": ["gate", "up"],
             "note": "fused [E, 2*moe, hid] -> switch_mlp.gate_proj + up_proj"},
            {"name": "conv1d_layout",
             "applies_to_suffix": "linear_attn.conv1d.weight",
             "store_shape": "source",                # [out, 1, k] (shape[-1] = k != 1)
             "runtime_shape": "moveaxis(2,1) -> [out, k, 1]",
             # Structured trigger so L1 checks mechanically (not by parsing prose): the
             # stored conv1d weight's last dim must be != 1 for the sanitizer to fire.
             "sanitizer_trigger": {"tensor_suffix": "linear_attn.conv1d.weight",
                                   "last_dim_not": 1},
             "note": "must store source [out,1,k]; that shape is the sanitize predicate"},
            {"name": "rmsnorm_shift",
             # Storage vs runtime are separate relations (the trap to respect):
             # norms are stored unshifted; the runtime-effective norm is source + delta.
             # L1 must check storage==source and the presence of the trigger. Requiring
             # the stored norm to equal source+delta would reject a correct
             # unshifted-storage package).
             "norm_storage": "unshifted",            # package stores source norm as-is
             "runtime_relation": "source_plus_delta",
             "delta": 1.0,                            # runtime weight = source weight + 1.0
             "applies_to": ["norm.input_layernorm", "norm.post_attention_layernorm",
                            "norm.final", "norm.attn.q_norm", "norm.attn.k_norm"],
             "coupled_to": "conv1d_layout",
             "required": True,                        # this family requires the runtime shift
             "note": "sanitize adds +1.0 to RMSNorm weights iff conv1d_layout's trigger "
                     "holds; norms stored unshifted. L1: storage==source + trigger present; "
                     "if shift required but trigger absent on disk -> block (suppressed "
                     "shift loads norms ~1.0 too low -> garbage)"},
        ],
        layer_kinds=["full_attention", "linear_attention"],   # hybrid; interleaved
        router={"top_k_present": True, "experts_per_layer_stacked": True},
        tokenizer={"required": True, "thinking_mode_default": True,
                   "note": "tokenizer is a package contract; chat-template renders <think>"},
        required_rungs=["L0", "L1"],   # current scope; higher rungs added as built
    )


def qwen3_5_dense_profile() -> dict:
    """The contract a dense qwen3_5 text package must satisfy.

    Same Qwen3.5 hybrid text stack as the MoE profile (wrapped source namespace,
    linear/full attention schedule, conv1d layout + RMSNorm shift sanitizer
    contract), but with dense MLP gate/up/down projections and no router,
    experts, switch_mlp, or fused expert gate_up split.
    """
    return _profile(
        {"family": "qwen3_5_dense", "modality": "text"},
        family="qwen3_5_dense",
        modality="text",
        excluded_namespaces={
            "visual": "text-only package; vision tower (model.visual.*) not served",
            "vision": "text-only package; any vision-tower namespace not served",
            "mtp": "multi-token-prediction draft layers; not part of the base text forward "
                   "(and mtp presence is itself a sanitize norm-shift trigger, excluding it "
                   "makes the conv1d-shape predicate the sole shift trigger)",
        },
        role_quant={
            "ffn.gate_proj": "affine",
            "ffn.up_proj": "affine",
            "ffn.down_proj": "affine",
            "attn.q_proj": "affine", "attn.k_proj": "affine",
            "attn.v_proj": "affine", "attn.o_proj": "affine",
            "ssm.in_proj_qkv": "affine", "ssm.in_proj_z": "affine",
            "ssm.in_proj_a": "affine", "ssm.in_proj_b": "affine",
            "ssm.out_proj": "affine",
            "embed_tokens": "affine", "lm_head": "affine",
        },
        structural_passthrough=[
            "norm.input_layernorm", "norm.post_attention_layernorm", "norm.final",
            "norm.attn.q_norm", "norm.attn.k_norm", "norm.ssm.norm",
            "ssm.A_log", "ssm.dt_bias", "ssm.conv1d",
        ],
        transforms=[
            {"name": "key_rename",
             "from": "model.language_model.*", "to": "language_model.model.*",
             "applied_by": "runtime sanitize (mlx_lm qwen3_5.Model.sanitize)"},
            {"name": "conv1d_layout",
             "applies_to_suffix": "linear_attn.conv1d.weight",
             "store_shape": "source",
             "runtime_shape": "moveaxis(2,1) -> [out, k, 1]",
             "sanitizer_trigger": {"tensor_suffix": "linear_attn.conv1d.weight",
                                   "last_dim_not": 1},
             "note": "must store source [out,1,k]; that shape is the sanitize predicate"},
            {"name": "rmsnorm_shift",
             "norm_storage": "unshifted",
             "runtime_relation": "source_plus_delta",
             "delta": 1.0,
             "applies_to": ["norm.input_layernorm", "norm.post_attention_layernorm",
                            "norm.final", "norm.attn.q_norm", "norm.attn.k_norm"],
             "coupled_to": "conv1d_layout",
             "required": True,
             "note": "sanitize adds +1.0 to RMSNorm weights iff conv1d_layout's trigger "
                     "holds; norms stored unshifted."},
        ],
        layer_kinds=["full_attention", "linear_attention"],
        router={"top_k_present": False, "experts_per_layer_stacked": False},
        tokenizer={"required": True, "thinking_mode_default": True,
                   "note": "tokenizer is a package contract; chat-template renders <think>"},
        required_rungs=["L0", "L1"],
    )


def qwen4_exp_profile() -> dict:
    """The released Qwen3.8-Flash-Next text-source contract.

    Quant ownership is intentionally unassigned. The profile registers the
    released family and its exact text/provider boundary without allowing the
    Qwen3.5 conversion path or its coupled RMSNorm transform to be inferred.
    """
    layer_kinds = [
        kind
        for _ in range(12)
        for kind in (
            "linear_attention", "linear_attention", "linear_attention",
            "qwen_sparse_attention",
        )
    ]
    return _profile(
        {"family": "qwen4_exp", "modality": "text"},
        family="qwen4_exp",
        modality="text",
        source_partition={
            "text_graph_tensors": 1163,
            "ple_provider_tensors": 131,
            "excluded_vision_tensors": 333,
            "excluded_mtp_tensors": 31,
        },
        excluded_namespaces={
            "model.visual": "vision tower tensors are not part of the text runtime",
            "mtp": "multi-token-prediction layers are postponed",
        },
        role_quant={},
        quantization={"status": "unassigned", "reference_storage": "source_bf16"},
        structural_passthrough=[
            "gr.final.hc_norm", "gr.attention.hc_norm", "gr.mlp.hc_norm",
            "gdn.A_log", "gdn.conv1d", "gdn.dt_bias", "gdn.norm",
            "qsa.q_norm", "qsa.k_norm", "qsa.indexer.q_layernorm",
            "qsa.indexer.k_layernorm", "ple.conv1d", "ple.norm_key",
            "ple.norm_query", "ple.norm_conv",
        ],
        transforms=[
            {"name": "expert_fused_gate_up", "source": "mlp.experts.gate_up_proj",
             "note": "the released source stores routed gate and up projections together"},
            {"name": "ple_provider_boundary", "source": "layers.1.ple.ple_embedding.*",
             "note": "three metadata tensors and 128 tables remain provider-owned"},
        ],
        layer_kinds=layer_kinds,
        router={"top_k_present": True, "experts_per_layer_stacked": True,
                "experts_per_layer": 512, "experts_per_token": 10},
        ple={"layer_index": 1, "ngram_size": 3, "physical_table_shards": 128},
        tokenizer={"required": True},
        cache_policy={"kind": "qwen4_hybrid_reference", "gdn_state": "fixed",
                      "qsa_kv": "growing"},
    )


DEEPSEEK_V4_FLASH_COMPRESS_RATIOS = (
    [0, 0] + [v for _ in range(20) for v in (4, 128)] + [4, 0]
)


def _deepseek_v4_layer_kinds() -> list[str]:
    out = []
    for ratio in DEEPSEEK_V4_FLASH_COMPRESS_RATIOS[:43]:
        if ratio == 0:
            out.append("swa")
        elif ratio == 4:
            out.append("csa")
        elif ratio == 128:
            out.append("hca")
        else:
            raise ValueError(f"unsupported DeepSeek V4 compress_ratio {ratio!r}")
    return out


def deepseek_v4_flash_profile() -> dict:
    """The contract a DeepSeek-V4-Flash text package must satisfy."""
    layer_kinds = _deepseek_v4_layer_kinds()
    rope_base_by_layer = [
        10000 if ratio == 0 else 160000
        for ratio in DEEPSEEK_V4_FLASH_COMPRESS_RATIOS[:43]
    ]
    yarn_by_layer = [
        ratio != 0
        for ratio in DEEPSEEK_V4_FLASH_COMPRESS_RATIOS[:43]
    ]
    return _profile(
        {"family": "deepseek_v4_flash", "modality": "text"},
        family="deepseek_v4_flash",
        modality="text",
        excluded_namespaces={
            "mtp": "multi-token-prediction draft layers; base text packages exclude them",
        },
        role_quant={
            "moe.expert": "routed_expert",
            "attn.wq_a": "affine", "attn.wq_b": "affine",
            "attn.wkv": "affine", "attn.wo_a": "affine", "attn.wo_b": "affine",
            "attn.indexer.wq_b": "affine",
            "attn.indexer.weights_proj": "affine",
            "attn.compressor.wkv": "affine", "attn.compressor.wgate": "affine",
            "attn.indexer.compressor.wkv": "affine",
            "attn.indexer.compressor.wgate": "affine",
            "moe.shared_expert.gate_proj": "affine",
            "moe.shared_expert.up_proj": "affine",
            "moe.shared_expert.down_proj": "affine",
            "embed_tokens": "affine", "lm_head": "affine",
            "moe.router_gate": "fp16",
            "norm.attn_norm": "fp16", "norm.ffn_norm": "fp16",
            "norm.attn.q_norm": "fp16", "norm.attn.kv_norm": "fp16",
            "norm.input_layernorm": "fp16", "norm.post_attention_layernorm": "fp16",
            "norm.attn.compressor": "fp16", "norm.attn.indexer.compressor": "fp16",
            "norm.final": "fp16",
            "attn.attn_sink": "raw_dtype_passthrough",
            "attn.compressor.ape": "raw_dtype_passthrough",
            "attn.indexer.compressor.ape": "raw_dtype_passthrough",
            "moe.router_bias": "raw_dtype_passthrough",
            "moe.router_tid2eid": "raw_dtype_passthrough",
            "hc.control": "raw_dtype_passthrough",
        },
        structural_passthrough=[
            "norm.attn_norm", "norm.ffn_norm",
            "norm.input_layernorm", "norm.post_attention_layernorm", "norm.final",
            "norm.attn.q_norm", "norm.attn.kv_norm", "norm.attn.compressor",
            "norm.attn.indexer.compressor", "attn.attn_sink", "attn.compressor.ape",
            "attn.indexer.compressor.ape", "moe.router_bias", "moe.router_tid2eid",
            "hc.control",
        ],
        transforms=[
            {"name": "expert_separate_w1_w3_w2",
             "source_layout": "separate_w1_w3_w2",
             "maps_to": {"w1": "gate", "w3": "up", "w2": "down"}},
            {"name": "source_decode",
             "fp4_experts": "decode_to_working_tensor_before_probe",
             "fp8_dense": "decode_to_working_tensor_before_probe"},
        ],
        layer_kinds=layer_kinds,
        compress_ratios=DEEPSEEK_V4_FLASH_COMPRESS_RATIOS,
        mtp_compress_ratio=DEEPSEEK_V4_FLASH_COMPRESS_RATIOS[43],
        rope_base_by_layer=rope_base_by_layer,
        yarn_by_layer=yarn_by_layer,
        yarn={"factor": 16, "original_seq_len": 65536, "beta_fast": 32, "beta_slow": 1},
        attention={
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "sliding_window": 128,
            "index_topk": 512,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "attn_sink": True,
        },
        hyper_connections={"hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6},
        router={
            "top_k_present": True,
            "hash_layers": 3,
            "experts_per_token": 6,
            "score_func": "sqrtsoftplus",
            "route_scale": 1.5,
            "expert_source_layout": "separate_w1_w3_w2",
        },
        tokenizer={
            "required": True,
            "renderer": "deepseek_v4_dsv4",
            "thinking_mode_default": False,
        },
        cache_policy={"kind": "deepseek_v4_composite", "generic_kv_bits": False},
        required_rungs=["L0", "L1", "synthetic_attention"],
    )


def synthetic_profile() -> dict:
    """A minimal dense family (no experts, no SSM): proves the schema is generic."""
    return _profile(
        {"family": "synthetic_dense", "modality": "text"},
        family="synthetic_dense",
        modality="text",
        excluded_namespaces={},
        role_quant={
            "attn.q_proj": "affine", "attn.o_proj": "affine",
            "ffn.gate_proj": "affine", "ffn.up_proj": "affine", "ffn.down_proj": "affine",
            "embed_tokens": "affine", "lm_head": "affine",
        },
        structural_passthrough=["norm.input_layernorm", "norm.final"],
        transforms=[],
        layer_kinds=["full_attention"],
        router={"top_k_present": False},
        tokenizer={"required": True},
        required_rungs=["L0", "L1"],
    )


# Family registry: builder per declared family, plus the source-config model_type
# tokens that map to each family. Kept tiny and explicit (no auto-discovery): a new
# family registers itself here once its profile is written from the real model's facts.
# The convert gate is
# format-agnostic: it resolves a profile through this map and skips (loud warning) when
# the model's family has no profile yet, rather than blocking an unprofiled family.
PROFILES = {
    "deepseek_v4_flash": deepseek_v4_flash_profile,
    "qwen3_5_dense": qwen3_5_dense_profile,
    "qwen3_5_moe": qwen3_5_moe_profile,
    "qwen4_exp": qwen4_exp_profile,
    "synthetic_dense": synthetic_profile,
}

# model_type strings seen in the wild (source config, sidecar, mlx_lm) -> our family id.
# These genuinely differ: HF source says "qwen3_moe", our sidecar "qwen3_5_moe_text".
_FAMILY_ALIASES = {
    "deepseek_v4": "deepseek_v4_flash",
    "deepseek_v4_flash": "deepseek_v4_flash",
    "qwen3_moe": "qwen3_5_moe",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen3_5_moe_text": "qwen3_5_moe",
    "qwen3_5_dense": "qwen3_5_dense",
    "qwen4_exp": "qwen4_exp",
    "qwen4_exp_text": "qwen4_exp",
    "synthetic_dense": "synthetic_dense",
}


def _has_experts(config: dict) -> bool:
    """Whether config declares a routed-expert/MoE text stack."""
    fields = ("num_experts", "num_local_experts", "n_routed_experts")
    return any(int((config or {}).get(k) or 0) > 0 for k in fields)


def family_of(config: dict) -> str | None:
    """Resolve a source/sidecar config's model_type to a known family id, or None."""
    cfg = config or {}
    text = cfg.get("text_config") or {}
    mt = cfg.get("model_type")
    text_mt = text.get("model_type")

    if mt == "deepseek_v4" or text_mt == "deepseek_v4":
        return "deepseek_v4_flash"

    if mt in {"qwen4_exp", "qwen4_exp_text"}:
        return "qwen4_exp"
    if text_mt in {"qwen4_exp", "qwen4_exp_text"}:
        return "qwen4_exp"

    # MoE tokens are explicit and win before the dense qwen3_5 wrapper branch.
    if mt in {"qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"}:
        return "qwen3_5_moe"
    if text_mt in {"qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"}:
        return "qwen3_5_moe"

    # Dense Qwen3.5 can be wrapped (top-level qwen3_5 + text_config qwen3_5_text)
    # or unwrapped in MLX-style configs. Resolve only when no expert fields are
    # declared; do not treat an expert-bearing qwen3_5 config as dense by alias.
    qwen35_dense_token = mt in {"qwen3_5", "qwen3_5_text"} or text_mt == "qwen3_5_text"
    dense_config = text if text else cfg
    if qwen35_dense_token and not _has_experts(dense_config):
        return "qwen3_5_dense"

    return _FAMILY_ALIASES.get(mt) or _FAMILY_ALIASES.get(text_mt)


def profile_for(config: dict) -> dict | None:
    """The architecture_profile for a model's config, or None if its family is unknown.

    Generic by construction: maps model_type -> family -> declared profile. Returns None
    (not a guess) when no profile is registered, so callers can skip rather than apply the
    wrong contract."""
    fam = family_of(config)
    builder = PROFILES.get(fam) if fam else None
    return builder() if builder else None
