"""Shared Qwen-style source-to-role and role-to-imatrix-key resolver.

One place owns tensor-name mapping. A tensor name maps to an internal role
(stable vocabulary the rest of MoEspresso uses) and, for quantizable weights, to
its GGUF imatrix key. Returning None for a key means "no imatrix entry expected"
(embed/lm_head), distinct from a mapping bug; the inventory counts coverage so a
silent 0-mapped can't pass.

Validated on the real Qwen3.6-35B-A3B (120/120 experts + 390/392 non-experts).
Scope: Qwen-style HF layouts (full-attn + linear-attn/SSM layers, stacked
experts, shared experts) plus shared stacked-expert helpers. Model-specific
families with different naming contracts live under model inventory packages.
"""

from __future__ import annotations

import re

# --- internal role vocabulary (small, grows with families) ---
ROLE_ATTN = {
    "self_attn.q_proj.weight": "attn.q_proj",
    "self_attn.k_proj.weight": "attn.k_proj",
    "self_attn.v_proj.weight": "attn.v_proj",
    "self_attn.o_proj.weight": "attn.o_proj",
}
ROLE_SSM = {
    "linear_attn.in_proj_qkv.weight": "ssm.in_proj_qkv",
    "linear_attn.in_proj_z.weight": "ssm.in_proj_z",
    "linear_attn.in_proj_a.weight": "ssm.in_proj_a",
    "linear_attn.in_proj_b.weight": "ssm.in_proj_b",
    "linear_attn.out_proj.weight": "ssm.out_proj",
}
ROLE_FFN_SHARED = {
    "mlp.gate.weight": "moe.router_gate",
    "mlp.shared_expert_gate.weight": "moe.shared_expert_gate",
    "mlp.shared_expert.gate_proj.weight": "moe.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj.weight": "moe.shared_expert.up_proj",
    "mlp.shared_expert.down_proj.weight": "moe.shared_expert.down_proj",
}
# Dense per-layer FFN (no router, no experts). Distinct from the MoE router
# `mlp.gate.weight` ('gate' vs 'gate_proj' don't collide) and from the stacked
# `mlp.experts.*` path. A dense model's whole MLP backbone lives here.
ROLE_FFN_DENSE = {
    "mlp.gate_proj.weight": "ffn.gate_proj",
    "mlp.up_proj.weight": "ffn.up_proj",
    "mlp.down_proj.weight": "ffn.down_proj",
}
ROLE_GLOBAL = {
    "model.language_model.embed_tokens.weight": "embed_tokens",
    "lm_head.weight": "lm_head",
}

# Structural tensors the text graph needs but that are NOT quantized: norms and
# the linear-attn (SSM) state params. They are copied into the package verbatim
# (passthrough) so the runtime builds the model without reading source files. The
# optimizer never sees these: they flow inventory -> writer -> manifest directly.
# Keyed by the suffix after `...layers.N.` (per-layer) plus the final `model.norm`.
_STRUCTURAL_PERLAYER = {
    "input_layernorm.weight": "norm.input_layernorm",
    "post_attention_layernorm.weight": "norm.post_attention_layernorm",
    "self_attn.q_norm.weight": "norm.attn.q_norm",
    "self_attn.k_norm.weight": "norm.attn.k_norm",
    "linear_attn.norm.weight": "norm.ssm.norm",
    "linear_attn.A_log": "ssm.A_log",
    "linear_attn.dt_bias": "ssm.dt_bias",
    "linear_attn.conv1d.weight": "ssm.conv1d",
}
_STRUCTURAL_GLOBAL = {"model.language_model.norm.weight": "norm.final"}

# --- GGUF imatrix key families (verified present in the real 510-key set) ---
_GGUF_FFN_SHARED = {
    "mlp.gate.weight": "ffn_gate_inp",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp",
}
# Dense FFN imatrix keys: the standard llama.cpp ffn block names.
_GGUF_FFN_DENSE = {
    "mlp.gate_proj.weight": "ffn_gate",
    "mlp.up_proj.weight": "ffn_up",
    "mlp.down_proj.weight": "ffn_down",
}
_GGUF_FULL_ATTN = {
    "self_attn.q_proj.weight": "attn_q",
    "self_attn.k_proj.weight": "attn_k",
    "self_attn.v_proj.weight": "attn_v",
    "self_attn.o_proj.weight": "attn_output",
}
_GGUF_LINEAR_ATTN = {
    "linear_attn.in_proj_qkv.weight": "attn_qkv",
    "linear_attn.in_proj_z.weight": "attn_gate",
    "linear_attn.in_proj_a.weight": "ssm_alpha",
    "linear_attn.in_proj_b.weight": "ssm_beta",
    "linear_attn.out_proj.weight": "ssm_out",
}

_NO_KEY_SUFFIXES = frozenset({"embed_tokens.weight", "lm_head.weight"})
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_STACKED_EXPERT_RE = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.([A-Za-z_]+?)(?:\.weight)?$")
def tensor_layer(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _suffix(name: str, layer: int) -> str:
    for marker in (f".layers.{layer}.", f"layers.{layer}."):
        if marker in name:
            return name[name.index(marker) + len(marker):]
    raise ValueError(f"layer marker for layer {layer} missing in {name!r}")


def non_expert_role(name: str, layer_types: list[str] | None) -> str | None:
    """Internal role for a non-expert tensor, or None if unrecognized."""
    if name in ROLE_GLOBAL:
        return ROLE_GLOBAL[name]
    if any(name.endswith(s) for s in _NO_KEY_SUFFIXES):
        return ROLE_GLOBAL.get(name)
    layer = tensor_layer(name)
    if layer is None:
        return None
    suf = _suffix(name, layer)
    if suf in ROLE_FFN_SHARED:
        return ROLE_FFN_SHARED[suf]
    if suf in ROLE_FFN_DENSE:
        return ROLE_FFN_DENSE[suf]
    return ROLE_ATTN.get(suf) or ROLE_SSM.get(suf)


def non_expert_gguf_key(name: str, layer_types: list[str] | None) -> str | None:
    """GGUF imatrix key for a non-expert tensor, or None if none is expected."""
    if any(name.endswith(s) for s in _NO_KEY_SUFFIXES):
        return None
    layer = tensor_layer(name)
    if layer is None:
        return None
    suf = _suffix(name, layer)
    if suf in _GGUF_FFN_SHARED:
        return f"blk.{layer}.{_GGUF_FFN_SHARED[suf]}.weight"
    if suf in _GGUF_FFN_DENSE:
        return f"blk.{layer}.{_GGUF_FFN_DENSE[suf]}.weight"

    is_full = None
    if layer_types is not None and layer < len(layer_types):
        is_full = layer_types[layer] == "full_attention"
    if is_full is True:
        fam = _GGUF_FULL_ATTN.get(suf)
    elif is_full is False:
        fam = _GGUF_LINEAR_ATTN.get(suf)
    else:  # unknown: suffix sets are disjoint, try both
        fam = _GGUF_FULL_ATTN.get(suf) or _GGUF_LINEAR_ATTN.get(suf)
    return f"blk.{layer}.{fam}.weight" if fam else None


def parse_stacked_expert(name: str) -> tuple[int, str] | None:
    """(layer, short_proj) for a stacked 3D expert tensor, else None.

    short_proj is 'gate_up' or 'down' (the on-disk projection, '_proj' stripped).
    """
    m = _STACKED_EXPERT_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2).replace("_proj", "")


def expert_gguf_keys(layer: int, projection: str) -> list[str]:
    """GGUF imatrix key(s) for a stacked MoE expert group.

    Fused 'gate_up' maps to two keys (gate + up); gate/up/down each to one.
    """
    proj = projection.replace("_proj", "")
    if proj in ("gate_up", "gateup"):
        return [f"blk.{layer}.ffn_gate_exps.weight", f"blk.{layer}.ffn_up_exps.weight"]
    if proj in ("gate", "up", "down"):
        return [f"blk.{layer}.ffn_{proj}_exps.weight"]
    return []


def expert_role(projection: str) -> str:
    return f"moe.expert.{projection.replace('_proj', '')}"


def switch_mlp_key(source_name: str, projection: str) -> str:
    """On-disk key for a stacked-expert sub-projection in jang's switch_mlp layout.

    The TQ kernel (TurboQuantSwitchLinear) consumes pre-stacked tensors keyed by the
    sanitized module path `{layer_prefix}switch_mlp.{gate,up,down}_proj`. jang's
    loader installs the module at exactly this path (no rename), so it must match the
    model's param path. `source_name` is the fused source
    (`model.language_model.layers.N.mlp.experts.gate_up_proj`); we rewrite
    `...mlp.experts.<x>` -> `...mlp.switch_mlp.<proj>_proj` and apply mlx_lm's
    sanitize rename (`model.language_model.` -> `language_model.model.`).
    """
    marker = ".experts."
    head = source_name[: source_name.index(marker)] if marker in source_name else source_name
    if head.startswith("model.language_model."):
        head = "language_model.model." + head[len("model.language_model."):]
    proj = projection if projection.endswith("_proj") else f"{projection}_proj"
    return f"{head}.switch_mlp.{proj}"


def structural_role(name: str) -> str | None:
    """Internal role for a non-quantized text structural tensor, or None.

    Norms + linear-attn (SSM) state under the language model. Returns None for
    anything not in the text graph (mtp.*, vision, rope inv-freq, biases on the
    quantized projections) so those stay dropped. The single place that decides
    'this un-quantized tensor must travel into the package as passthrough'.
    """
    if name in _STRUCTURAL_GLOBAL:
        return _STRUCTURAL_GLOBAL[name]
    if name.startswith("mtp.") or "visual" in name or "vision" in name:
        return None
    layer = tensor_layer(name)
    if layer is None:
        return None
    suf = _suffix(name, layer)
    return _STRUCTURAL_PERLAYER.get(suf)


def switch_mlp_bundle_prefix(source_name: str) -> str:
    """On-disk key prefix for a layer's per-expert bundle.

    The bundle tensor is `<prefix>.tq_bundle` with
    prefix = `{layer_prefix}switch_mlp.experts`, one per layer, replacing the
    three per-projection stacked tensors. Derived from the same sanitized head
    as switch_mlp_key so the two conventions can never drift apart.
    """
    head = switch_mlp_key(source_name, "gate")  # ".../switch_mlp.gate_proj"
    return head.rsplit(".", 1)[0] + ".experts"
