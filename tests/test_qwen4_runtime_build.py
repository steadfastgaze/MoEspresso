from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from moespresso.inventory.qwen4.roles import module_weight_key, tensor_role
from moespresso.inventory.qwen4.static import expected_qwen38_flash_next_text_tensors
from moespresso.runtime.qwen4.build import (
    Qwen4RuntimeBuildError,
    build_qwen4_graph_from_manifest,
)


def _manifest() -> dict:
    layer_types = [
        "full_attention" if layer % 4 == 3 else "linear_attention"
        for layer in range(48)
    ]
    return {
        "artifact_id": "pkg:qwen38-flash-next-test",
        "architecture": {
            "family": "qwen4_exp",
            "text_model_type": "qwen4_exp_text",
            "config": {
                "model_type": "qwen4_exp_text",
                "dtype": "bfloat16",
                "vocab_size": 248320,
                "hidden_size": 2560,
                "num_hidden_layers": 48,
                "hidden_act": "silu",
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
                "norm_topk_prob": True,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "ngram_size": 3,
                "heads_per_ngram": 8,
                "ple_embed_dim": 2560,
                "ple_conv_kernel_size": 4,
                "ple_layer_ids": [2],
                "ngram_vocab_size_base": 20000000,
                "make_ngram_vocab_size_divisible_by": 128,
                "split_ngram_parts": 128,
                "seed": 1234,
                "eos_token_id": 248044,
                "rms_norm_eps": 1e-6,
                "output_gate_type": "sigmoid",
                "partial_rotary_factor": 0.25,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000000,
                    "mrope_interleaved": True,
                    "mrope_section": [11, 11, 10],
                },
                "tie_word_embeddings": False,
                "layer_types": layer_types,
            },
        },
    }


def _compact_manifest(count: int = 448) -> dict:
    manifest = _manifest()
    selection_id = "select:" + "b" * 64
    source_ids = list(range(count))
    manifest["required_features"] = ["qwen4_per_layer_experts"]
    manifest["inputs"] = [selection_id]
    manifest["expert_layout"] = {
        "per_layer_experts": {
            "source_selection_artifact_id": selection_id,
            "source_package_manifest_id": "pkg:" + "a" * 64,
            "source_num_experts": 512,
            "top_k": 10,
            "layers": {
                str(layer): {
                    "num_experts": count,
                    "source_expert_ids": source_ids,
                }
                for layer in range(48)
            },
        }
    }
    return manifest


class _Provider:
    def lookup(self, row_ids: mx.array) -> mx.array:
        return mx.zeros((*row_ids.shape, 160), dtype=mx.bfloat16)


class _Experts(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

    def __call__(self, hidden_states: mx.array, indices: mx.array) -> mx.array:
        raise AssertionError("graph-construction tests do not execute experts")


def _build():
    calls = []

    def factory(layer: int, hidden: int, width: int, experts: int):
        calls.append((layer, hidden, width, experts))
        return _Experts(hidden, width, experts)

    model = build_qwen4_graph_from_manifest(
        _manifest(),
        expert_factory=factory,
        ple_provider=_Provider(),
    )
    return model, calls


def test_released_manifest_builds_the_exact_provider_backed_text_graph() -> None:
    model, calls = _build()

    assert model.cache_identity == "pkg:qwen38-flash-next-test"
    assert (model.hidden_size, model.branch_count, model.expanded_size) == (2560, 4, 10240)
    assert len(model.layers) == 48
    assert [layer.mixer_kind for layer in model.layers] == [
        "qsa" if layer % 4 == 3 else "gdn" for layer in range(48)
    ]
    assert [index for index, layer in enumerate(model.layers) if layer.ple is not None] == [1]
    assert calls == [(layer, 2560, 640, 512) for layer in range(48)]
    assert model.embedding.weight is not model.lm_head.weight
    assert model.final_residual.block_inject_weight is None


def test_compact_manifest_keeps_source_router_and_builds_physical_expert_counts() -> None:
    calls = []

    def factory(layer: int, hidden: int, width: int, experts: int):
        calls.append((layer, hidden, width, experts))
        return _Experts(hidden, width, experts)

    model = build_qwen4_graph_from_manifest(
        _compact_manifest(),
        expert_factory=factory,
        ple_provider=_Provider(),
    )

    assert calls == [(layer, 2560, 640, 448) for layer in range(48)]
    assert {layer.mlp.gate.num_experts for layer in model.layers} == {512}
    assert {layer.mlp.physical_experts for layer in model.layers} == {448}
    assert all(
        layer.mlp.retained_source_ids == tuple(range(448))
        for layer in model.layers
    )


def test_released_graph_owns_every_direct_inventory_weight_once() -> None:
    model, _calls = _build()
    actual = {name for name, _array in tree_flatten(model.parameters())}
    expected = set()
    for name in expected_qwen38_flash_next_text_tensors():
        role = tensor_role(name)
        assert role is not None
        if role["kind"] == "expert":
            continue
        key = module_weight_key(name)
        assert key is not None
        expected.add(key)

    assert len(expected) == 1067
    assert actual == expected


def test_released_graph_uses_mlx_grouped_convolution_layouts() -> None:
    model, _calls = _build()

    assert model.layers[0].mixer.module.conv1d.weight.shape == (10240, 4, 1)
    assert model.layers[1].ple is not None
    assert model.layers[1].ple.conv1d.weight.shape == (10240, 4, 1)
    assert model.layers[1].ple.conv_dilation == 3
    assert model.layers[1].ple.short_conv_state_len == 9


def test_released_graph_injects_one_qsa_state_backend_per_attention_layer() -> None:
    backends = []

    def backend_factory(module):
        backend = SimpleNamespace(module=module)
        backends.append(backend)
        return backend

    model = build_qwen4_graph_from_manifest(
        _manifest(),
        expert_factory=lambda _layer, hidden, width, experts: _Experts(
            hidden,
            width,
            experts,
        ),
        ple_provider=_Provider(),
        qsa_state_backend_factory=backend_factory,
        qsa_max_query_tokens=64,
    )

    qsa_mixers = [layer.mixer for layer in model.layers if layer.mixer_kind == "qsa"]
    assert len(qsa_mixers) == len(backends) == 12
    assert [mixer.state_backend for mixer in qsa_mixers] == backends
    assert {mixer.max_query_tokens for mixer in qsa_mixers} == {64}


def test_released_graph_refuses_architecture_drift_before_provider_construction() -> None:
    manifest = deepcopy(_manifest())
    manifest["architecture"]["config"]["layer_types"][7] = "linear_attention"
    calls = []

    with pytest.raises(Qwen4RuntimeBuildError, match="layer_types"):
        build_qwen4_graph_from_manifest(
            manifest,
            expert_factory=lambda *args: calls.append(args),
            ple_provider=_Provider(),
        )

    assert calls == []


def test_released_graph_requires_explicit_expert_and_ple_providers() -> None:
    with pytest.raises(TypeError, match="expert_factory"):
        build_qwen4_graph_from_manifest(
            _manifest(),
            expert_factory=None,  # type: ignore[arg-type]
            ple_provider=_Provider(),
        )
    with pytest.raises(TypeError, match="ple_provider"):
        build_qwen4_graph_from_manifest(
            _manifest(),
            expert_factory=lambda layer, hidden, width, experts: _Experts(
                hidden, width, experts
            ),
            ple_provider=object(),  # type: ignore[arg-type]
        )
