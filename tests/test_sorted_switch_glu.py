"""Sorted K-quant routed MoE seam for the resident qwen path.

These tests stay synthetic: they prove the install seam swaps the stock
SwitchGLU for the sorted route, that the route engages on prefill shapes and
falls closed on decode / off-contract shapes, that the kill switch restores the
stock seam, and that the sorted kernels are called with the combined-stack
layout and the right sorted-ids / gate-out / swiglu-limit arguments. The real
The campaign quality ladder owns K-quant numerical acceptance.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

from moespresso.runtime.qwen import sorted_switch_glu as ssg  # noqa: E402
from moespresso.runtime.qwen.sorted_switch_glu import (  # noqa: E402
    SortedKQuantSwitchGLU,
    _combine_gate_up_stack,
    install_sorted_kquant_switchglus,
)


def _kquant_switch_linear(n_experts, out_features, bytes_per_row, codec):
    """A minimal K-quant switch stack: duck-types the mlx_kquant module."""
    mod = nn.Module()
    mod.mode = "kquant"
    mod.kquant_type = codec
    mod.bits = 4
    mod.group_size = 32
    mod.weight = mx.array(
        np.random.default_rng(len(codec) + out_features).integers(
            0, 256, (n_experts, out_features, bytes_per_row), dtype=np.uint8
        )
    )
    mod.scales = mx.zeros((1,), dtype=mx.uint8)
    mx.eval(mod.weight, mod.scales)
    return mod


class _Model(nn.Module):
    """A qwen-shaped model whose switch_mlp carries resident K-quant stacks."""

    def __init__(
        self,
        *,
        hidden=256,
        gate_out=128,
        n_experts=6,
        n_layers=1,
        gate_codec="q4_k",
        up_codec="q4_k",
        down_codec="q6_k",
        gate_bpr=72,
        down_bpr=104,
    ):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = []
        for _ in range(n_layers):
            layer = nn.Module()
            layer.mlp = nn.Module()
            sw = SwitchGLU(hidden, gate_out, n_experts)
            sw.gate_proj = _kquant_switch_linear(n_experts, gate_out, gate_bpr, gate_codec)
            sw.up_proj = _kquant_switch_linear(n_experts, gate_out, gate_bpr, up_codec)
            sw.down_proj = _kquant_switch_linear(n_experts, hidden, down_bpr, down_codec)
            layer.mlp.switch_mlp = sw
            self.language_model.model.layers.append(layer)


def _switch(model, layer=0):
    return model.language_model.model.layers[layer].mlp.switch_mlp


def _fake_mlx_kquant(monkeypatch, calls, *, with_swiglu=True):
    def fake_gather_qmm_sorted(x, w, scales, codec, sorted_ids, transpose=True):
        calls.append(("sorted", tuple(w.shape), codec, int(sorted_ids.shape[0])))
        return mx.zeros((x.shape[0], w.shape[1]), dtype=x.dtype)

    def fake_gather_qmm_sorted_swiglu(
        x, w, scales, codec, sorted_ids, gate_out, swiglu_limit, transpose=True
    ):
        calls.append(("swiglu", tuple(w.shape), codec, int(gate_out), float(swiglu_limit)))
        return mx.zeros((x.shape[0], gate_out), dtype=x.dtype)

    def fake_gather_qmm(x, w, scales, codec, **kwargs):
        calls.append(("gather", tuple(w.shape), codec, kwargs.get("sorted_indices")))
        rhs = kwargs["rhs_indices"]
        return mx.zeros((*x.shape[:-1], rhs.shape[-1], 1, w.shape[1]), dtype=x.dtype)

    ns = types.SimpleNamespace(
        gather_qmm_sorted=fake_gather_qmm_sorted,
        gather_qmm=fake_gather_qmm,
    )
    if with_swiglu:
        ns.gather_qmm_sorted_swiglu = fake_gather_qmm_sorted_swiglu
    monkeypatch.setitem(sys.modules, "mlx_kquant", ns)
    return ns


# ---- combine layout ---------------------------------------------------------


def test_combine_gate_up_stack_places_gate_rows_first():
    n_experts, gate_out, bpr = 4, 128, 72
    gate = _kquant_switch_linear(n_experts, gate_out, bpr, "q4_k")
    up = _kquant_switch_linear(n_experts, gate_out, bpr, "q4_k")
    combined = _combine_gate_up_stack(gate, up)
    assert tuple(combined.shape) == (n_experts, 2 * gate_out, bpr)
    assert bool(mx.array_equal(combined[:, :gate_out], gate.weight).item())
    assert bool(mx.array_equal(combined[:, gate_out:], up.weight).item())


def test_combine_gate_up_stack_rejects_codec_mismatch():
    gate = _kquant_switch_linear(4, 128, 72, "q4_k")
    up = _kquant_switch_linear(4, 128, 72, "q5_k")
    with pytest.raises(ssg.SortedKQuantSwitchGLUError):
        _combine_gate_up_stack(gate, up)


# ---- install seam -----------------------------------------------------------


def test_install_swaps_switch_mlp_and_builds_combined_stack(monkeypatch):
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", True)
    model = _Model(n_layers=2)
    installed = install_sorted_kquant_switchglus(model)
    assert installed == 2
    for layer in range(2):
        sw = _switch(model, layer)
        assert isinstance(sw, SortedKQuantSwitchGLU)
        # combined gate/up stack holds 2 * gate_out rows.
        assert int(sw.gate_up.weight.shape[1]) == 2 * sw.gate_out_features
        assert sw.gate_up_type == "q4_k"
        assert sw.down_type == "q6_k"


def test_install_is_noop_when_kill_switch_off(monkeypatch):
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", False)
    model = _Model()
    installed = install_sorted_kquant_switchglus(model)
    assert installed == 0
    assert isinstance(_switch(model), SwitchGLU)


def test_install_leaves_non_kquant_layers_on_stock_seam(monkeypatch):
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", True)
    model = _Model()
    # Demote gate to a non-K-quant module: the layer must stay on the stock seam.
    _switch(model).gate_proj.mode = "affine"
    installed = install_sorted_kquant_switchglus(model)
    assert installed == 0
    assert isinstance(_switch(model), SwitchGLU)


def test_install_leaves_mismatched_gate_up_codec_on_stock_seam(monkeypatch):
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", True)
    model = _Model(up_codec="q5_k")
    installed = install_sorted_kquant_switchglus(model)
    assert installed == 0
    assert isinstance(_switch(model), SwitchGLU)


# ---- engagement and fallback ------------------------------------------------


def _prefill_inputs(sw, *, tokens):
    top_k = 8
    x = mx.array(
        np.random.default_rng(1).standard_normal((tokens, sw.in_features)), dtype=mx.float32
    )
    inds = mx.array(
        np.random.default_rng(2).integers(0, sw.num_experts, (tokens, top_k)), dtype=mx.uint32
    )
    mx.eval(x, inds)
    return x, inds


def test_prefill_engages_fused_sorted_route(monkeypatch):
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", True)
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED_SWIGLU", True)
    monkeypatch.setattr(ssg, "_SORTED_PREFILL_MIN_ROWS", 4096)
    model = _Model()
    install_sorted_kquant_switchglus(model)
    sw = _switch(model)
    sw.eval()
    calls = []
    _fake_mlx_kquant(monkeypatch, calls, with_swiglu=True)

    x, inds = _prefill_inputs(sw, tokens=700)  # 700*8 = 5600 >= 4096
    mx.eval(sw(x, inds))

    assert sw.sorted_prefill_calls == 1
    assert sw.fused_swiglu_calls == 1
    assert sw.fallback_calls == 0
    assert sw.prefill_calls == 1
    # gate/up combined GEMM (fused swiglu) then the down GEMM (plain sorted).
    kinds = [c[0] for c in calls]
    assert kinds == ["swiglu", "sorted"]
    swiglu_call = calls[0]
    assert swiglu_call[1][1] == 2 * sw.gate_out_features  # combined rows
    assert swiglu_call[2] == "q4_k"
    assert swiglu_call[3] == sw.gate_out_features  # gate_out arg
    assert swiglu_call[4] == 0.0  # unclamped swiglu
    assert calls[1][2] == "q6_k"  # down codec
    # both GEMMs see the same number of sorted pairs.
    assert swiglu_call  # sanity
    assert calls[1][3] == 700 * 8


def test_prefill_engages_unfused_sorted_when_no_fused_kernel(monkeypatch):
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", True)
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED_SWIGLU", True)
    monkeypatch.setattr(ssg, "_SORTED_PREFILL_MIN_ROWS", 4096)
    model = _Model()
    install_sorted_kquant_switchglus(model)
    sw = _switch(model)
    sw.eval()
    calls = []
    _fake_mlx_kquant(monkeypatch, calls, with_swiglu=False)

    x, inds = _prefill_inputs(sw, tokens=700)
    mx.eval(sw(x, inds))

    assert sw.sorted_prefill_calls == 1
    assert sw.fused_swiglu_calls == 0
    # unfused: combined gate/up sorted GEMM, then down sorted GEMM.
    kinds = [c[0] for c in calls]
    assert kinds == ["sorted", "sorted"]
    assert calls[0][1][1] == 2 * sw.gate_out_features


def test_swiglu_kill_switch_forces_unfused_route(monkeypatch):
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", True)
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED_SWIGLU", False)
    monkeypatch.setattr(ssg, "_SORTED_PREFILL_MIN_ROWS", 4096)
    model = _Model()
    install_sorted_kquant_switchglus(model)
    sw = _switch(model)
    sw.eval()
    calls = []
    _fake_mlx_kquant(monkeypatch, calls, with_swiglu=True)  # fused kernel present

    x, inds = _prefill_inputs(sw, tokens=700)
    mx.eval(sw(x, inds))

    assert sw.fused_swiglu_calls == 0
    assert [c[0] for c in calls] == ["sorted", "sorted"]


def test_decode_falls_back_to_unsorted_gather(monkeypatch):
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", True)
    monkeypatch.setattr(ssg, "_SORTED_PREFILL_MIN_ROWS", 4096)
    model = _Model()
    install_sorted_kquant_switchglus(model)
    sw = _switch(model)
    sw.eval()
    calls = []
    _fake_mlx_kquant(monkeypatch, calls, with_swiglu=True)

    x = mx.array(np.random.default_rng(3).standard_normal((1, sw.in_features)), dtype=mx.float32)
    inds = mx.array([[0, 1, 2, 3, 4, 5, 0, 1]], dtype=mx.uint32)
    mx.eval(x, inds)
    mx.eval(sw(x, inds))

    assert sw.sorted_prefill_calls == 0
    assert sw.fallback_calls == 1
    assert sw.decode_calls == 1
    # unsorted gather over combined gate/up then down, sorted_indices False.
    kinds = [c[0] for c in calls]
    assert kinds == ["gather", "gather"]
    assert calls[0][3] is False


def test_short_prefill_below_threshold_falls_back(monkeypatch):
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", True)
    monkeypatch.setattr(ssg, "_SORTED_PREFILL_MIN_ROWS", 4096)
    model = _Model()
    install_sorted_kquant_switchglus(model)
    sw = _switch(model)
    sw.eval()
    calls = []
    _fake_mlx_kquant(monkeypatch, calls, with_swiglu=True)

    x, inds = _prefill_inputs(sw, tokens=100)  # 100*8 = 800 < 4096
    mx.eval(sw(x, inds))

    assert sw.sorted_prefill_calls == 0
    assert sw.fallback_calls == 1
    assert [c[0] for c in calls] == ["gather", "gather"]


def test_training_mode_does_not_block_sorted_route(monkeypatch):
    # The served mlx_lm model reports training=True without ever running a
    # backward pass, and the stock SwitchGLU sorts at this scale regardless of
    # training. The sorted route must engage in either mode so it reaches the
    # whole served path.
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", True)
    monkeypatch.setattr(ssg, "_SORTED_PREFILL_MIN_ROWS", 4096)
    model = _Model()
    install_sorted_kquant_switchglus(model)
    sw = _switch(model)
    sw.train()
    calls = []
    _fake_mlx_kquant(monkeypatch, calls, with_swiglu=True)

    x, inds = _prefill_inputs(sw, tokens=700)
    mx.eval(sw(x, inds))

    assert sw.sorted_prefill_calls == 1
    assert sw.fallback_calls == 0


def test_missing_sorted_kernel_fails_closed(monkeypatch):
    monkeypatch.setattr(ssg, "_QWEN_MOE_SORTED", True)
    monkeypatch.setattr(ssg, "_SORTED_PREFILL_MIN_ROWS", 4096)
    model = _Model()
    install_sorted_kquant_switchglus(model)
    sw = _switch(model)
    sw.eval()
    calls = []
    # no gather_qmm_sorted in the module -> eligibility must fail closed.
    ns = types.SimpleNamespace(
        gather_qmm=lambda x, w, s, c, **kw: (
            calls.append(("gather",))
            or mx.zeros((*x.shape[:-1], kw["rhs_indices"].shape[-1], 1, w.shape[1]), dtype=x.dtype)
        ),
    )
    monkeypatch.setitem(sys.modules, "mlx_kquant", ns)

    x, inds = _prefill_inputs(sw, tokens=700)
    mx.eval(sw(x, inds))

    assert sw.sorted_prefill_calls == 0
    assert sw.fallback_calls == 1
    assert [c[0] for c in calls] == ["gather", "gather"]
