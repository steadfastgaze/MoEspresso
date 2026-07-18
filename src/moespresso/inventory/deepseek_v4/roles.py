"""DeepSeek-V4 source-name resolver.

This module owns DS4-specific tensor-name rules. The shared `inventory.roles`
module stays focused on the generic/Qwen-style HF layout and common helpers.
"""

from __future__ import annotations

import re

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_DS4_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
_DS4_HC_RE = re.compile(r"^(?:layers\.\d+\.hc_(?:attn|ffn)|hc_head)_(?:fn|base|scale)$")

_DS4_WEIGHT_ROLES = {
    "attn.wq_a.weight": "attn.wq_a",
    "attn.wq_b.weight": "attn.wq_b",
    "attn.wkv.weight": "attn.wkv",
    "attn.wo_a.weight": "attn.wo_a",
    "attn.wo_b.weight": "attn.wo_b",
    "attn.indexer.wq_b.weight": "attn.indexer.wq_b",
    "attn.indexer.weights_proj.weight": "attn.indexer.weights_proj",
    "attn.compressor.wkv.weight": "attn.compressor.wkv",
    "attn.compressor.wgate.weight": "attn.compressor.wgate",
    "attn.indexer.compressor.wkv.weight": "attn.indexer.compressor.wkv",
    "attn.indexer.compressor.wgate.weight": "attn.indexer.compressor.wgate",
    "ffn.shared_experts.w1.weight": "moe.shared_expert.gate_proj",
    "ffn.shared_experts.w3.weight": "moe.shared_expert.up_proj",
    "ffn.shared_experts.w2.weight": "moe.shared_expert.down_proj",
    "ffn.gate.weight": "moe.router_gate",
}
_DS4_GGUF_WEIGHT_KEYS = {
    "attn.wq_a.weight": "attn_q_a",
    "attn.wq_b.weight": "attn_q_b",
    "attn.wkv.weight": "attn_kv",
    "attn.wo_a.weight": "attn_output_a",
    "attn.wo_b.weight": "attn_output_b",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp",
    "ffn.shared_experts.w3.weight": "ffn_up_shexp",
    "ffn.shared_experts.w2.weight": "ffn_down_shexp",
}
_DS4_NORM_ROLES = {
    "attn_norm.weight": "norm.attn_norm",
    "ffn_norm.weight": "norm.ffn_norm",
    "input_layernorm.weight": "norm.input_layernorm",
    "post_attention_layernorm.weight": "norm.post_attention_layernorm",
    "attn.q_norm.weight": "norm.attn.q_norm",
    "attn.kv_norm.weight": "norm.attn.kv_norm",
    "attn.compressor.norm.weight": "norm.attn.compressor",
    "attn.indexer.compressor.norm.weight": "norm.attn.indexer.compressor",
}
_DS4_RAW_ROLES = {
    "attn.attn_sink": "attn.attn_sink",
    "attn.compressor.ape": "attn.compressor.ape",
    "attn.indexer.compressor.ape": "attn.indexer.compressor.ape",
    "ffn.gate.bias": "moe.router_bias",
    "ffn.gate.tid2eid": "moe.router_tid2eid",
}
_DS4_GLOBAL_ROLES = {
    "embed.weight": ("affine", "embed_tokens"),
    "head.weight": ("affine", "lm_head"),
    "norm.weight": ("passthrough", "norm.final"),
}
_DS4_GLOBAL_GGUF_KEYS = {
    "head.weight": "output.weight",
}
_DS4_EXPERT_PROJECTIONS = {"w1": "gate", "w3": "up", "w2": "down"}


def tensor_layer(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _suffix(name: str, layer: int) -> str:
    for marker in (f".layers.{layer}.", f"layers.{layer}."):
        if marker in name:
            return name[name.index(marker) + len(marker):]
    raise ValueError(f"layer marker for layer {layer} missing in {name!r}")


def tensor_role(name: str) -> dict | None:
    """Typed DS4 role record, or None when the tensor is explicitly excluded."""
    if name.startswith("mtp."):
        return None
    if _DS4_HC_RE.match(name):
        return {"kind": "passthrough", "role": "hc.control", "format": "raw_dtype_passthrough"}
    global_role = _DS4_GLOBAL_ROLES.get(name)
    if global_role is not None:
        kind, role = global_role
        fmt = "fp16" if kind == "passthrough" else None
        return {"kind": kind, "role": role, "format": fmt}

    expert = _DS4_EXPERT_RE.match(name)
    if expert:
        layer, expert_index, source_proj, suffix = expert.groups()
        projection = _DS4_EXPERT_PROJECTIONS[source_proj]
        kind = "expert_source" if suffix == "weight" else "codec_scale"
        return {
            "kind": kind,
            "role": f"moe.expert.{projection}",
            "layer_index": int(layer),
            "expert_index": int(expert_index),
            "projection": projection,
            "source_projection": source_proj,
        }

    layer = tensor_layer(name)
    if layer is None:
        return {"kind": "unknown", "role": "unknown"}
    suf = _suffix(name, layer)
    if suf.endswith(".scale"):
        weight_suffix = f"{suf[:-len('.scale')]}.weight"
        role = _DS4_WEIGHT_ROLES.get(weight_suffix)
        if role is not None:
            return {"kind": "codec_scale", "role": f"{role}.scale", "layer_index": layer}
    role = _DS4_WEIGHT_ROLES.get(suf)
    if role is not None:
        return {"kind": "affine", "role": role, "layer_index": layer}
    role = _DS4_NORM_ROLES.get(suf)
    if role is not None:
        return {"kind": "passthrough", "role": role, "layer_index": layer, "format": "fp16"}
    role = _DS4_RAW_ROLES.get(suf)
    if role is not None:
        return {
            "kind": "passthrough",
            "role": role,
            "layer_index": layer,
            "format": "raw_dtype_passthrough",
        }
    return {
        "kind": "unknown",
        "role": "unknown",
        "layer_index": layer,
    }


def gguf_key(name: str) -> str | None:
    """GGUF recipe/imatrix key for a DS4 non-expert source tensor."""
    global_key = _DS4_GLOBAL_GGUF_KEYS.get(name)
    if global_key is not None:
        return global_key
    layer = tensor_layer(name)
    if layer is None:
        return None
    suf = _suffix(name, layer)
    stem = _DS4_GGUF_WEIGHT_KEYS.get(suf)
    if stem is None:
        return None
    return f"blk.{layer}.{stem}.weight"


def module_path(source_name: str) -> str:
    """DS4 source tensor name -> JANG DS4 graph module path without suffix."""
    name = source_name
    for suffix in (".weight", ".scales", ".biases"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    if name == "embed":
        return "model.embed"
    if name == "head":
        return "lm_head"
    if name == "norm":
        return "model.norm"
    if name.startswith("hc_head_"):
        return f"model.{name}"

    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "layers":
        return f"model.{name}"

    layer, rest = parts[1], ".".join(parts[2:])
    prefix = f"model.layers.{layer}"
    if rest == "attn_norm":
        return f"{prefix}.input_layernorm"
    if rest == "ffn_norm":
        return f"{prefix}.post_attention_layernorm"
    if rest.startswith("hc_"):
        return f"{prefix}.{rest}"
    if rest.startswith("attn."):
        return f"{prefix}.self_attn.{rest[len('attn.'):]}"
    if rest.startswith("ffn.gate."):
        return f"{prefix}.mlp.gate.{rest[len('ffn.gate.'):]}"
    for source, target in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
        head = f"ffn.shared_experts.{source}"
        if rest == head:
            return f"{prefix}.mlp.shared_experts.{target}"
    if rest.startswith("ffn."):
        return f"{prefix}.mlp.{rest[len('ffn.'):]}"
    return f"{prefix}.{rest}"
