"""Load an optional IQ2 MTP sidecar without modifying the target model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from moespresso.core.paths import resolve_artifact_file
from moespresso.package.qwen4.mtp_format import read_mtp_sidecar_manifest
from moespresso.runtime.qwen4.iqk_dense import _walk_module_path
from moespresso.runtime.qwen4.model import Qwen4DecoderLayer
from moespresso.runtime.qwen4.moe import Qwen4SparseMoEBlock
from moespresso.runtime.qwen4.mtp import Qwen4MTPHead
from moespresso.runtime.qwen4.mtp_iq2 import (
    Qwen4MTPIQ2Experts,
    Qwen4MTPIQ2Linear,
    Qwen4MTPIQ2Router,
    Qwen4MTPIQ2SwitchLinear,
)
from moespresso.runtime.qwen4.primitives import Qwen4GatedResidual, Qwen4RMSNorm
from moespresso.runtime.qwen4.qsa import Qwen4QSAAdapter, Qwen4SparseAttention


@dataclass(frozen=True)
class LoadedQwen4MTP:
    head: Qwen4MTPHead
    embedding: Any
    artifact_id: str
    target_cache_identity: str
    payload_bytes: int
    vocab_size: int


def _build_head(graph: dict, lm_head) -> Qwen4MTPHead:
    hidden, branches, lowrank = graph["hidden_size"], graph["hc_count"], graph["hc_lowrank"]
    eps = graph["rms_norm_eps"]
    attention = Qwen4SparseAttention(
        hidden_size=hidden, num_query_heads=graph["num_attention_heads"],
        num_kv_heads=graph["num_key_value_heads"], head_dim=graph["head_dim"],
        index_query_heads=graph["indexer_n_heads"], index_kv_heads=graph["indexer_kv_heads"],
        index_head_dim=graph["indexer_head_dim"], token_budget=graph["indexer_budget"],
        compress_ratio=graph["indexer_compress_ratio"], rotary_dim=graph["rotary_dim"],
        rope_base=graph["rope_base"], mrope_section=tuple(graph["mrope_section"]), eps=eps,
    )
    mlp = Qwen4SparseMoEBlock(
        hidden, graph["moe_intermediate_size"], graph["shared_expert_intermediate_size"],
        graph["num_experts"], graph["num_experts_per_tok"],
        expert_executor=Qwen4MTPIQ2Experts(hidden, graph["moe_intermediate_size"], graph["num_experts"]),
    )
    mlp.gate = Qwen4MTPIQ2Router(hidden, graph["num_experts"], graph["num_experts_per_tok"])
    layer = Qwen4DecoderLayer(
        mixer_kind="qsa", mixer=Qwen4QSAAdapter(attention),
        attention_residual=Qwen4GatedResidual(hidden, branches, lowrank, eps=eps),
        mlp_residual=Qwen4GatedResidual(hidden, branches, lowrank, eps=eps), mlp=mlp,
    )
    return Qwen4MTPHead(
        hidden, branches, layer=layer, eps=eps,
        fc_embedding=nn.Linear(hidden, hidden, bias=False),
        fc_hidden=nn.Linear(hidden, hidden, bias=False),
        final_residual=Qwen4GatedResidual(hidden, branches, lowrank, combine=False, eps=eps),
        lm_head=lm_head,
    )


def _tensor_ports(head):
    ports = {}
    for path, module in head.named_modules():
        if path == "lm_head" or path.startswith("lm_head."):
            continue
        if isinstance(module, (nn.Linear, Qwen4RMSNorm, Qwen4MTPIQ2Router)):
            ports[(path, None)] = module
        elif isinstance(module, Qwen4MTPIQ2Experts):
            ports[(path, "gate_up")] = module.gate_up_proj
            ports[(path, "down")] = module.down_proj
    return ports


def load_qwen4_mtp_sidecar(sidecar_dir: Path, target_manifest: dict, *, target_model) -> LoadedQwen4MTP:
    """Hydrate manifest-owned ports; leave payload hashing to the separate verifier."""
    target_identity = getattr(target_model, "cache_identity", "")
    if target_identity.partition("|")[0] != target_manifest.get("artifact_id"):
        raise ValueError("MTP target model does not belong to the supplied target manifest")
    manifest = read_mtp_sidecar_manifest(sidecar_dir, target_manifest)
    head = _build_head(manifest["graph"], target_model.lm_head)
    ports = _tensor_ports(head)
    records = manifest["tensors"]
    destinations = [(record["module_path"], record.get("projection")) for record in records]
    if len(set(destinations)) != len(destinations) or set(destinations) != set(ports):
        raise ValueError("MTP sidecar does not cover the draft graph exactly")
    for record, destination in zip(records, destinations, strict=True):
        module = ports[destination]
        if isinstance(module, Qwen4RMSNorm):
            shape, kind = (module.dimensions,), "passthrough"
        elif isinstance(module, Qwen4MTPIQ2SwitchLinear):
            shape = (module.out_features, module.in_features)
            kind = "matrix"
            if destination[1] is not None:
                shape, kind = (module.num_experts, *shape), "expert"
        else:
            shape, kind = tuple(module.weight.shape), "matrix"
        if tuple(record["logical_shape"]) != shape or record["kind"] != kind:
            raise ValueError(f"MTP tensor does not match its graph port: {destination}")

    loaded_files = {}
    payload_bytes = 0
    for record, destination in zip(records, destinations, strict=True):
        module = ports[destination]
        arrays = []
        for payload in record["payloads"]:
            file = payload["file"]
            if file not in loaded_files:
                loaded_files[file] = mx.load(str(resolve_artifact_file(sidecar_dir, file)))
            value = loaded_files[file][payload["key"]]
            arrays.append(value)
            payload_bytes += value.nbytes
        if isinstance(module, Qwen4RMSNorm):
            if len(arrays) != 1 or arrays[0].shape != (module.dimensions,) or arrays[0].dtype != mx.bfloat16:
                raise ValueError("MTP norm payload is invalid")
            module.weight = arrays[0]
            continue
        if not isinstance(module, Qwen4MTPIQ2SwitchLinear):
            module = Qwen4MTPIQ2Linear(*module.weight.shape)
            parent, leaf = _walk_module_path(head, destination[0])
            setattr(parent, leaf, module)
        slices = [{} for _ in module.plan.slices]
        for payload, value in zip(record["payloads"], arrays, strict=True):
            index = payload.get("slice")
            component = payload.get("component")
            if type(index) is not int or not 0 <= index < len(slices) or component in slices[index]:
                raise ValueError("MTP projection has an invalid or duplicate stream slice")
            slices[index][component] = value
        module.load_streams(slices)
    if payload_bytes != manifest["byte_estimate"]["iq2_k_payload_with_input_padding"]:
        raise ValueError("MTP loaded payload bytes disagree with its allocation")
    mx.eval([value for arrays in loaded_files.values() for value in arrays.values()])
    head.eval()
    return LoadedQwen4MTP(
        head=head, embedding=target_model.embedding, artifact_id=manifest["artifact_id"],
        target_cache_identity=target_identity, payload_bytes=payload_bytes,
        vocab_size=manifest["graph"]["vocab_size"],
    )
