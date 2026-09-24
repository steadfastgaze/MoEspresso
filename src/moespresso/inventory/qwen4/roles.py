"""Typed tensor roles for the released Qwen3.8-Flash-Next checkpoint.

The released family has a different graph and storage boundary from Qwen3.5.
This resolver therefore names every served tensor family explicitly instead of
passing Qwen4 names through the shared Qwen heuristics.
"""

from __future__ import annotations

import re


_LAYER_COUNT = 48
_QSA_PERIOD = 4

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_PLE_TABLE_RE = re.compile(r"^ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")

_EXCLUDED_PREFIXES = ("model.visual.", "mtp.")

_GLOBAL_ROLES = {
    "model.language_model.embed_tokens.weight": ("affine", "embed_tokens"),
    "model.language_model.hyper_connection_mixer.hc_norm.weight": (
        "passthrough",
        "gr.final.hc_norm",
    ),
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": (
        "affine",
        "gr.final.input_mix_weight_down",
    ),
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": (
        "affine",
        "gr.final.input_mix_weight_up",
    ),
    "lm_head.weight": ("affine", "lm_head"),
}

_HYPER_CONNECTION_ROLES = {
    "hc_norm.weight": ("passthrough", "hc_norm"),
    "input_mix_weight_down.weight": ("affine", "input_mix_weight_down"),
    "input_mix_weight_up.weight": ("affine", "input_mix_weight_up"),
    "block_inject_weight.weight": ("affine", "block_inject_weight"),
}

_MOE_ROLES = {
    "experts.gate_up_proj": ("expert", "moe.expert.gate_up", "gate_up"),
    "experts.down_proj": ("expert", "moe.expert.down", "down"),
    "gate.weight": ("affine", "moe.router_gate", None),
    "shared_expert.gate_proj.weight": (
        "affine",
        "moe.shared_expert.gate_proj",
        None,
    ),
    "shared_expert.up_proj.weight": ("affine", "moe.shared_expert.up_proj", None),
    "shared_expert.down_proj.weight": (
        "affine",
        "moe.shared_expert.down_proj",
        None,
    ),
    "shared_expert_gate.weight": ("affine", "moe.shared_expert_gate", None),
}

_GDN_ROLES = {
    "A_log": ("passthrough", "gdn.A_log"),
    "conv1d.weight": ("passthrough", "gdn.conv1d"),
    "dt_bias": ("passthrough", "gdn.dt_bias"),
    "in_proj_qkv.weight": ("affine", "gdn.in_proj_qkv"),
    "in_proj_z.weight": ("affine", "gdn.in_proj_z"),
    "in_proj_a.weight": ("affine", "gdn.in_proj_a"),
    "in_proj_b.weight": ("affine", "gdn.in_proj_b"),
    "norm.weight": ("passthrough", "gdn.norm"),
    "out_proj.weight": ("affine", "gdn.out_proj"),
}

_QSA_ROLES = {
    "q_proj.weight": ("affine", "qsa.q_proj"),
    "k_proj.weight": ("affine", "qsa.k_proj"),
    "v_proj.weight": ("affine", "qsa.v_proj"),
    "o_proj.weight": ("affine", "qsa.o_proj"),
    "q_norm.weight": ("passthrough", "qsa.q_norm"),
    "k_norm.weight": ("passthrough", "qsa.k_norm"),
    "indexer.index_qk_proj.weight": ("affine", "qsa.indexer.index_qk_proj"),
    "indexer.q_layernorm.weight": ("passthrough", "qsa.indexer.q_layernorm"),
    "indexer.k_layernorm.weight": ("passthrough", "qsa.indexer.k_layernorm"),
}

_PLE_GRAPH_ROLES = {
    "key_proj.weight": ("affine", "ple.key_proj"),
    "value_proj.weight": ("affine", "ple.value_proj"),
    "conv1d.weight": ("passthrough", "ple.conv1d"),
    "norm_key.weight": ("passthrough", "ple.norm_key"),
    "norm_query.weight": ("passthrough", "ple.norm_query"),
    "norm_conv.weight": ("passthrough", "ple.norm_conv"),
}

_PLE_PROVIDER_METADATA_ROLES = {
    "ple_embedding.layer_multipliers": "ple.provider.layer_multipliers",
    "ple_embedding.ngram_heads_vocab_sizes": ("ple.provider.ngram_heads_vocab_sizes"),
    "ple_embedding.ngram_heads_offsets": "ple.provider.ngram_heads_offsets",
}


def _layer_module_path(layer: int, role: str) -> str:
    prefix = f"layers.{layer}"
    if role.startswith("gr.attention."):
        return f"{prefix}.attention_residual.{role.removeprefix('gr.attention.')}"
    if role.startswith("gr.mlp."):
        return f"{prefix}.mlp_residual.{role.removeprefix('gr.mlp.')}"
    if role.startswith("gdn."):
        return f"{prefix}.mixer.module.{role.removeprefix('gdn.')}"
    if role.startswith("qsa."):
        return f"{prefix}.mixer.module.{role.removeprefix('qsa.')}"
    if role.startswith("moe.expert."):
        return f"{prefix}.mlp.experts"
    if role == "moe.router_gate":
        return f"{prefix}.mlp.gate"
    if role.startswith("moe."):
        return f"{prefix}.mlp.{role.removeprefix('moe.')}"
    if role.startswith("ple.") and not role.startswith("ple.provider."):
        return f"{prefix}.ple.{role.removeprefix('ple.')}"
    raise ValueError(f"Qwen4 role {role!r} has no runtime module path")


def module_path(name: str) -> str | None:
    """Return the injected Qwen4 shell module that owns one released tensor."""
    resolved = tensor_role(name)
    if resolved is None or resolved["kind"] in {"provider", "unknown"}:
        return None
    role = resolved["role"]
    if name == "model.language_model.embed_tokens.weight":
        return "embedding"
    if name == "lm_head.weight":
        return "lm_head"
    if role.startswith("gr.final."):
        return f"final_residual.{role.removeprefix('gr.final.')}"
    layer = resolved.get("layer_index")
    if layer is None:
        raise ValueError(f"Qwen4 tensor {name!r} has no runtime layer")
    return _layer_module_path(int(layer), role)


def module_weight_key(name: str) -> str | None:
    """Return the exact shell parameter key hydrated from one released tensor."""
    path = module_path(name)
    if path is None:
        return None
    resolved = tensor_role(name)
    if resolved is None or resolved["kind"] == "expert":
        return None
    if name.endswith(".A_log"):
        return f"{path.rsplit('.', 1)[0]}.A_log"
    if name.endswith(".dt_bias"):
        return f"{path.rsplit('.', 1)[0]}.dt_bias"
    return f"{path}.weight"


def _record(kind: str, role: str, *, layer_index: int | None = None, **fields) -> dict:
    out = {"kind": kind, "role": role, "layer_index": layer_index}
    if kind in {"passthrough", "provider"}:
        out["format"] = "raw_dtype_passthrough"
    out.update(fields)
    return out


def _unknown(layer_index: int | None = None) -> dict:
    return {"kind": "unknown", "role": "unknown", "layer_index": layer_index}


def tensor_layer(name: str) -> int | None:
    """Return the released text-layer index encoded in ``name``, if present."""
    match = _LAYER_RE.match(name)
    return int(match.group(1)) if match else None


def tensor_role(name: str) -> dict | None:
    """Resolve one released Qwen4 tensor, or omit an explicit non-text namespace.

    Unknown tensors in the served namespace return an ``unknown`` role so the
    inventory builder can reject source drift. Only vision and MTP are omitted.
    """
    if name.startswith(_EXCLUDED_PREFIXES):
        return None

    global_role = _GLOBAL_ROLES.get(name)
    if global_role is not None:
        return _record(global_role[0], global_role[1])

    match = _LAYER_RE.match(name)
    if match is None:
        return _unknown()
    layer = int(match.group(1))
    suffix = match.group(2)
    if not 0 <= layer < _LAYER_COUNT:
        return _unknown(layer)

    for seam, role_prefix in (
        ("attn_hyper_connection.", "gr.attention"),
        ("mlp_hyper_connection.", "gr.mlp"),
    ):
        if suffix.startswith(seam):
            resolved = _HYPER_CONNECTION_ROLES.get(suffix.removeprefix(seam))
            if resolved is None:
                return _unknown(layer)
            return _record(resolved[0], f"{role_prefix}.{resolved[1]}", layer_index=layer)

    if suffix.startswith("mlp."):
        resolved = _MOE_ROLES.get(suffix.removeprefix("mlp."))
        if resolved is None:
            return _unknown(layer)
        kind, role, projection = resolved
        fields = {"projection": projection} if projection is not None else {}
        return _record(kind, role, layer_index=layer, **fields)

    if suffix.startswith("linear_attn."):
        if layer % _QSA_PERIOD == _QSA_PERIOD - 1:
            return _unknown(layer)
        resolved = _GDN_ROLES.get(suffix.removeprefix("linear_attn."))
        if resolved is None:
            return _unknown(layer)
        return _record(resolved[0], resolved[1], layer_index=layer)

    if suffix.startswith("self_attn."):
        if layer % _QSA_PERIOD != _QSA_PERIOD - 1:
            return _unknown(layer)
        resolved = _QSA_ROLES.get(suffix.removeprefix("self_attn."))
        if resolved is None:
            return _unknown(layer)
        return _record(resolved[0], resolved[1], layer_index=layer)

    if suffix.startswith("ple."):
        if layer != 1:
            return _unknown(layer)
        ple_suffix = suffix.removeprefix("ple.")
        resolved = _PLE_GRAPH_ROLES.get(ple_suffix)
        if resolved is not None:
            return _record(resolved[0], resolved[1], layer_index=layer)
        provider_role = _PLE_PROVIDER_METADATA_ROLES.get(ple_suffix)
        if provider_role is not None:
            return _record(
                "provider",
                provider_role,
                layer_index=layer,
                component="ple",
                provider_tensor_kind="metadata",
            )
        table_match = _PLE_TABLE_RE.match(ple_suffix)
        if table_match is not None:
            shard_index = int(table_match.group(1))
            if not 0 <= shard_index < 128:
                return _unknown(layer)
            return _record(
                "provider",
                "ple.provider.embedding_table",
                layer_index=layer,
                component="ple",
                provider_tensor_kind="embedding_table",
                provider_shard_index=shard_index,
            )
        return _unknown(layer)

    return _unknown(layer)
