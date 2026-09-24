"""Manifest-driven construction of the released Qwen4 text graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace

import mlx.nn as nn

from moespresso.runtime.qwen4.gdn import Qwen4GDNAdapter, Qwen4GatedDeltaNet
from moespresso.runtime.qwen4.expert_layout import (
    Qwen4ExpertLayoutError,
    parse_qwen4_expert_layout,
)
from moespresso.runtime.qwen4.model import Qwen4DecoderLayer, Qwen4TextModelShell
from moespresso.runtime.qwen4.moe import Qwen4RoutedExpertExecutor, Qwen4SparseMoEBlock
from moespresso.runtime.qwen4.ple import (
    Qwen4NGramHasher,
    Qwen4PLEEmbeddingProvider,
    Qwen4PLELayer,
)
from moespresso.runtime.qwen4.ple_contract import (
    Qwen4PLEProviderError,
    derive_qwen4_ple_provider_contract,
)
from moespresso.runtime.qwen4.primitives import Qwen4GatedResidual
from moespresso.runtime.qwen4.qsa import Qwen4QSAAdapter, Qwen4SparseAttention


class Qwen4RuntimeBuildError(RuntimeError):
    """Raised when a package cannot construct the supported Qwen4 graph."""


Qwen4ExpertFactory = Callable[
    [int, int, int, int],
    Qwen4RoutedExpertExecutor,
]
Qwen4QSAStateBackendFactory = Callable[[Qwen4SparseAttention], object]

_RELEASED_LAYER_COUNT = 48
_RELEASED_LAYER_TYPES = tuple(
    "full_attention" if layer % 4 == 3 else "linear_attention"
    for layer in range(_RELEASED_LAYER_COUNT)
)


def _fail(message: str) -> Qwen4RuntimeBuildError:
    return Qwen4RuntimeBuildError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail(f"{field} must be an object")
    return value


def _integer(
    config: Mapping[str, object],
    field: str,
    *,
    expected: int | None = None,
) -> int:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail(f"architecture.config.{field} must be a positive integer")
    if expected is not None and value != expected:
        raise _fail(
            f"architecture.config.{field}={value} is unsupported; expected {expected}"
        )
    return value


def _number(
    config: Mapping[str, object],
    field: str,
    *,
    expected: float,
) -> float:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"architecture.config.{field} must be numeric")
    if float(value) != expected:
        raise _fail(
            f"architecture.config.{field}={value} is unsupported; expected {expected}"
        )
    return float(value)


def _released_config(manifest: Mapping[str, object]) -> Mapping[str, object]:
    architecture = _mapping(manifest.get("architecture"), field="architecture")
    config = _mapping(architecture.get("config"), field="architecture.config")
    families = {
        architecture.get("family"),
        architecture.get("text_model_type"),
        config.get("model_type"),
    }
    if not families.intersection({"qwen4_exp", "qwen4_exp_text"}):
        raise _fail("architecture does not declare the Qwen4-Exp text family")
    if config.get("model_type") != "qwen4_exp_text":
        raise _fail("architecture.config.model_type must be qwen4_exp_text")
    if config.get("dtype") != "bfloat16":
        raise _fail("the released Qwen4 graph requires bfloat16 source semantics")
    if config.get("hidden_act") != "silu":
        raise _fail("the released Qwen4 graph requires the SiLU activation")
    if config.get("output_gate_type") != "sigmoid":
        raise _fail("the released Qwen4 GDN graph requires a sigmoid output gate")
    if config.get("tie_word_embeddings") is not False:
        raise _fail("the released Qwen4 embedding and LM head must remain untied")

    checks = {
        "vocab_size": 248320,
        "hidden_size": 2560,
        "num_hidden_layers": _RELEASED_LAYER_COUNT,
        "hc_count": 4,
        "hc_lowrank": 320,
        "head_dim": 256,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4,
        "eos_token_id": 248044,
    }
    for field, expected in checks.items():
        _integer(config, field, expected=expected)
    _number(config, "rms_norm_eps", expected=1e-6)
    _number(config, "partial_rotary_factor", expected=0.25)

    layer_types = config.get("layer_types")
    if (
        isinstance(layer_types, (str, bytes))
        or not isinstance(layer_types, Sequence)
        or tuple(layer_types) != _RELEASED_LAYER_TYPES
    ):
        raise _fail("architecture.config.layer_types does not match the released pattern")
    if config.get("ple_layer_ids") != [2]:
        raise _fail("the released Qwen4 graph requires PLE on one-based layer 2")
    if config.get("norm_topk_prob", True) is not True:
        raise _fail("the released Qwen4 router requires normalized top-k probabilities")

    rope = _mapping(
        config.get("rope_parameters"),
        field="architecture.config.rope_parameters",
    )
    if (
        rope.get("rope_type") != "default"
        or rope.get("rope_theta") != 10_000_000
        or rope.get("mrope_interleaved") is not True
        or rope.get("mrope_section") != [11, 11, 10]
    ):
        raise _fail("architecture.config.rope_parameters is unsupported")
    return config


def build_qwen4_graph_from_manifest(
    manifest: Mapping[str, object],
    *,
    expert_factory: Qwen4ExpertFactory,
    ple_provider: Qwen4PLEEmbeddingProvider,
    qsa_state_backend_factory: Qwen4QSAStateBackendFactory | None = None,
    qsa_max_query_tokens: int | None = None,
) -> Qwen4TextModelShell:
    """Construct the released text graph without reading source checkpoint files.

    Routed experts and PLE rows remain provider-owned. The builder creates only
    the graph modules and their parameter destinations; package hydration is a
    separate fail-closed phase.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    if not callable(expert_factory):
        raise TypeError("expert_factory must be callable")
    if qsa_state_backend_factory is not None and not callable(
        qsa_state_backend_factory
    ):
        raise TypeError("qsa_state_backend_factory must be callable or None")
    if not hasattr(ple_provider, "lookup"):
        raise TypeError("ple_provider must implement lookup(row_ids)")
    config = _released_config(manifest)
    try:
        expert_layout = parse_qwen4_expert_layout(manifest)
    except Qwen4ExpertLayoutError as exc:
        raise _fail(f"invalid Qwen4 compact expert layout: {exc}") from exc
    architecture = _mapping(manifest.get("architecture"), field="architecture")
    artifact_id = manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise _fail("package artifact_id must be a non-empty string")

    try:
        ple_contract = derive_qwen4_ple_provider_contract(architecture)
    except Qwen4PLEProviderError as exc:
        raise _fail(f"invalid Qwen4 PLE architecture: {exc}") from exc
    if ple_contract is None or ple_contract.layer_index != 1:
        raise _fail("the released Qwen4 PLE contract is missing")

    hidden_size = int(config["hidden_size"])
    branch_count = int(config["hc_count"])
    lowrank_size = int(config["hc_lowrank"])
    eps = float(config["rms_norm_eps"])
    expert_width = int(config["moe_intermediate_size"])
    expert_count = int(config["num_experts"])
    shared_width = int(config["shared_expert_intermediate_size"])
    top_k = int(config["num_experts_per_tok"])

    hasher = Qwen4NGramHasher(
        eos_token_id=int(config["eos_token_id"]),
        ngram_size=ple_contract.ngram_size,
        heads_per_ngram=ple_contract.heads_per_ngram,
        multipliers=ple_contract.multipliers,
        table_sizes=ple_contract.table_sizes,
        table_offsets=ple_contract.table_offsets,
    )
    ple = Qwen4PLELayer(
        hasher,
        ple_provider,
        row_width=ple_contract.row_width,
        hidden_size=hidden_size,
        branch_count=branch_count,
        conv_kernel_size=int(config["ple_conv_kernel_size"]),
        conv_dilation=int(config["ngram_size"]),
        eps=eps,
    )

    layers = []
    for layer_index, layer_type in enumerate(config["layer_types"]):
        if layer_type == "linear_attention":
            module = Qwen4GatedDeltaNet(
                SimpleNamespace(
                    hidden_size=hidden_size,
                    linear_num_value_heads=int(config["linear_num_value_heads"]),
                    linear_num_key_heads=int(config["linear_num_key_heads"]),
                    linear_key_head_dim=int(config["linear_key_head_dim"]),
                    linear_value_head_dim=int(config["linear_value_head_dim"]),
                    linear_conv_kernel_dim=int(config["linear_conv_kernel_dim"]),
                    rms_norm_eps=eps,
                )
            )
            mixer_kind = "gdn"
            mixer = Qwen4GDNAdapter(module)
        else:
            rope = config["rope_parameters"]
            assert isinstance(rope, Mapping)
            module = Qwen4SparseAttention(
                hidden_size=hidden_size,
                num_query_heads=int(config["num_attention_heads"]),
                num_kv_heads=int(config["num_key_value_heads"]),
                head_dim=int(config["head_dim"]),
                index_query_heads=int(config["indexer_n_heads"]),
                index_kv_heads=int(config["indexer_kv_heads"]),
                index_head_dim=int(config["indexer_head_dim"]),
                token_budget=int(config["indexer_budget"]),
                compress_ratio=int(config["indexer_compress_ratio"]),
                rotary_dim=int(config["head_dim"] * config["partial_rotary_factor"]),
                rope_base=float(rope["rope_theta"]),
                mrope_section=tuple(int(value) for value in rope["mrope_section"]),
                eps=eps,
            )
            mixer_kind = "qsa"
            state_backend = (
                None
                if qsa_state_backend_factory is None
                else qsa_state_backend_factory(module)
            )
            mixer = Qwen4QSAAdapter(
                module,
                state_backend=state_backend,
                max_query_tokens=qsa_max_query_tokens,
            )

        retained_source_ids = (
            None
            if expert_layout is None
            else expert_layout.layers[layer_index].source_expert_ids
        )
        physical_experts = (
            expert_count
            if retained_source_ids is None
            else len(retained_source_ids)
        )
        expert_executor = expert_factory(
            layer_index,
            hidden_size,
            expert_width,
            physical_experts,
        )
        layers.append(
            Qwen4DecoderLayer(
                mixer_kind=mixer_kind,
                attention_residual=Qwen4GatedResidual(
                    hidden_size,
                    branch_count,
                    lowrank_size,
                    eps=eps,
                ),
                mixer=mixer,
                mlp_residual=Qwen4GatedResidual(
                    hidden_size,
                    branch_count,
                    lowrank_size,
                    eps=eps,
                ),
                mlp=Qwen4SparseMoEBlock(
                    hidden_size,
                    expert_width,
                    shared_width,
                    expert_count,
                    top_k,
                    normalize_topk=True,
                    expert_executor=expert_executor,
                    retained_source_ids=retained_source_ids,
                ),
                ple=ple if layer_index == ple_contract.layer_index else None,
            )
        )

    return Qwen4TextModelShell(
        cache_identity=artifact_id,
        embedding=nn.Embedding(int(config["vocab_size"]), hidden_size),
        layers=tuple(layers),
        final_residual=Qwen4GatedResidual(
            hidden_size,
            branch_count,
            lowrank_size,
            combine=False,
            eps=eps,
        ),
        lm_head=nn.Linear(hidden_size, int(config["vocab_size"]), bias=False),
        hidden_size=hidden_size,
        branch_count=branch_count,
    )
