"""Exactness, transactional installation, and fallback tests for router GEMV."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from moespresso.runtime.qwen import router_gemv as rg  # noqa: E402
from moespresso.runtime.qwen.router_gemv import (  # noqa: E402
    BF16F32RouterLinear,
    install_router_bf16_f32_gemv,
    router_bf16_f32_stats,
)


_METAL_AVAILABLE = bool(getattr(mx, "metal", None) is not None and mx.metal.is_available())


def _bitwise_equal(left, right):
    mx.eval(left, right)
    return bool(mx.array_equal(mx.view(left, mx.uint32), mx.view(right, mx.uint32)))


def _lattice_weight(seed=101):
    weight = mx.random.normal(
        (rg._OUT_FEATURES, rg._IN_FEATURES),
        key=mx.random.key(seed),
    )
    return weight.astype(mx.bfloat16).astype(mx.float32)


def _linear(weight=None, *, bias=False):
    gate = nn.Linear(rg._IN_FEATURES, rg._OUT_FEATURES, bias=bias)
    if weight is None:
        weight = mx.zeros((rg._OUT_FEATURES, rg._IN_FEATURES), dtype=mx.float32)
    gate.weight = weight
    gate.eval()
    return gate


def _model(*, n_layers=rg._LAYERS):
    shared_weight = mx.zeros((rg._OUT_FEATURES, rg._IN_FEATURES), dtype=mx.float32)
    layers = []
    for _ in range(n_layers):
        mlp = SimpleNamespace(
            gate=_linear(shared_weight),
            top_k=8,
            norm_topk_prob=True,
        )
        layers.append(SimpleNamespace(mlp=mlp))
    return SimpleNamespace(model_type=rg._MODEL_TYPE, layers=layers)


def _enable(monkeypatch):
    monkeypatch.setattr(rg, "_QWEN_ROUTER_BF16_F32_GEMV", True)
    monkeypatch.setattr(rg, "_versions_compatible", lambda: True)
    monkeypatch.setattr(rg, "_kernel_available", lambda: True)


@pytest.mark.skipif(not _METAL_AVAILABLE, reason="router GEMV requires Metal")
def test_mixed_gemv_is_bit_identical_to_stock_router_and_top8():
    weight_f32 = _lattice_weight()
    weight_bf16 = weight_f32.astype(mx.bfloat16)
    inputs = mx.random.normal((1, 1, rg._IN_FEATURES), key=mx.random.key(202)).astype(mx.float32)

    stock = inputs @ weight_f32.T
    candidate = rg._mixed_router_gemv(weight_bf16, inputs)
    assert _bitwise_equal(candidate, stock)

    stock_probs = mx.softmax(stock, axis=-1, precise=True)
    candidate_probs = mx.softmax(candidate, axis=-1, precise=True)
    stock_ids = mx.argpartition(stock_probs, kth=-8, axis=-1)[..., -8:]
    candidate_ids = mx.argpartition(candidate_probs, kth=-8, axis=-1)[..., -8:]
    stock_scores = mx.take_along_axis(stock_probs, stock_ids, axis=-1)
    candidate_scores = mx.take_along_axis(candidate_probs, candidate_ids, axis=-1)
    stock_scores = stock_scores / stock_scores.sum(axis=-1, keepdims=True)
    candidate_scores = candidate_scores / candidate_scores.sum(axis=-1, keepdims=True)
    mx.eval(stock_ids, candidate_ids, stock_scores, candidate_scores)
    assert bool(mx.array_equal(stock_ids, candidate_ids))
    assert _bitwise_equal(stock_scores, candidate_scores)


@pytest.mark.parametrize(
    ("inputs", "counter"),
    [
        (
            mx.zeros((1, 2, rg._IN_FEATURES), dtype=mx.float32),
            "fallback_input_shape",
        ),
        (
            mx.zeros((1, 1, rg._IN_FEATURES), dtype=mx.bfloat16),
            "fallback_input_dtype",
        ),
    ],
)
def test_off_contract_calls_delegate_bit_identically(inputs, counter):
    inner = _linear(_lattice_weight(303))
    wrapped = BF16F32RouterLinear(inner, inner.weight.astype(mx.bfloat16))
    wrapped.eval()
    expected = inner(inputs)
    got = wrapped(inputs)
    assert _bitwise_equal(got, expected)
    assert getattr(wrapped, counter) == 1
    assert wrapped.kernel_calls == 0


def test_installer_is_transactional_and_idempotent(monkeypatch):
    _enable(monkeypatch)
    model = _model()
    originals = [layer.mlp.gate for layer in model.layers]

    assert install_router_bf16_f32_gemv(model) == rg._LAYERS
    wrappers = [layer.mlp.gate for layer in model.layers]
    assert all(isinstance(gate, BF16F32RouterLinear) for gate in wrappers)
    assert [gate.inner for gate in wrappers] == originals
    assert all(gate.weight_bf16.dtype == mx.bfloat16 for gate in wrappers)
    assert install_router_bf16_f32_gemv(model) == 0

    stats = router_bf16_f32_stats(model)
    assert stats["wrapped_layers"] == rg._LAYERS
    assert stats["validated_layers"] == rg._LAYERS
    assert stats["kernel_calls"] == 0


def test_capacity_reservation_requires_exact_ornith_geometry(monkeypatch):
    _enable(monkeypatch)
    config = {
        "model_type": rg._MODEL_TYPE,
        "text_config": {
            "num_hidden_layers": rg._LAYERS,
            "hidden_size": rg._IN_FEATURES,
            "num_experts": rg._OUT_FEATURES,
            "num_experts_per_tok": 8,
        },
    }

    assert rg.router_bf16_f32_resident_bytes(config) == rg._SHADOW_BYTES
    config["text_config"]["num_hidden_layers"] -= 1
    assert rg.router_bf16_f32_resident_bytes(config) == 0
    config["text_config"]["num_hidden_layers"] += 1
    monkeypatch.setattr(rg, "_QWEN_ROUTER_BF16_F32_GEMV", False)
    assert rg.router_bf16_f32_resident_bytes(config) == 0


def test_one_non_lattice_weight_prevents_every_install(monkeypatch):
    _enable(monkeypatch)
    model = _model()
    original_gates = [layer.mlp.gate for layer in model.layers]
    model.layers[17].mlp.gate.weight = mx.full(
        (rg._OUT_FEATURES, rg._IN_FEATURES), 0.1, dtype=mx.float32
    )

    assert install_router_bf16_f32_gemv(model) == 0
    assert [layer.mlp.gate for layer in model.layers] == original_gates
    assert router_bf16_f32_stats(model)["wrapped_layers"] == 0


def test_installer_fails_closed_on_shape_bias_layer_count_and_switches(monkeypatch):
    _enable(monkeypatch)

    short = _model(n_layers=rg._LAYERS - 1)
    assert install_router_bf16_f32_gemv(short) == 0

    biased = _model()
    biased.layers[0].mlp.gate = _linear(bias=True)
    assert install_router_bf16_f32_gemv(biased) == 0

    wrong_topk = _model()
    wrong_topk.layers[0].mlp.top_k = 4
    assert install_router_bf16_f32_gemv(wrong_topk) == 0

    wrong_family = _model()
    wrong_family.model_type = "other"
    assert install_router_bf16_f32_gemv(wrong_family) == 0

    monkeypatch.setattr(rg, "_QWEN_ROUTER_BF16_F32_GEMV", False)
    disabled = _model()
    assert install_router_bf16_f32_gemv(disabled) == 0


def test_partial_installation_raises(monkeypatch):
    _enable(monkeypatch)
    model = _model()
    inner = model.layers[0].mlp.gate
    model.layers[0].mlp.gate = BF16F32RouterLinear(inner, inner.weight.astype(mx.bfloat16))
    with pytest.raises(RuntimeError, match="partial"):
        install_router_bf16_f32_gemv(model)


def test_wrapper_exposes_linear_contract_and_rejects_replaced_weight(monkeypatch):
    _enable(monkeypatch)
    inner = _linear(_lattice_weight(406))
    wrapped = BF16F32RouterLinear(inner, inner.weight.astype(mx.bfloat16))
    wrapped.eval()
    assert wrapped.weight is inner.weight
    assert wrapped.bias is None

    inner.weight = _lattice_weight(407)
    inputs = mx.zeros((1, 1, rg._IN_FEATURES), dtype=mx.float32)
    expected = inner(inputs)
    monkeypatch.setattr(
        rg,
        "_mixed_router_gemv",
        lambda *_args: pytest.fail("stale BF16 shadow was used"),
    )
    got = wrapped(inputs)
    assert _bitwise_equal(got, expected)
    assert wrapped.fallback_weight_contract == 1
    assert wrapped.kernel_calls == 0


@pytest.mark.skipif(not _METAL_AVAILABLE, reason="router GEMV requires Metal")
def test_eligible_wrapper_counts_kernel_calls(monkeypatch):
    _enable(monkeypatch)
    inner = _linear(_lattice_weight(404))
    wrapped = BF16F32RouterLinear(inner, inner.weight.astype(mx.bfloat16))
    wrapped.eval()
    inputs = mx.random.normal((1, 1, rg._IN_FEATURES), key=mx.random.key(405)).astype(mx.float32)
    expected = inner(inputs)
    got = wrapped(inputs)
    assert _bitwise_equal(got, expected)
    assert wrapped.kernel_calls == 1
    assert wrapped.fallback_input_shape == 0
    assert wrapped.fallback_input_dtype == 0
