"""Intermediate GDN checkpoints for two-row MTP verification."""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_kquant.nn import KQuantLinear
from mlx_lm.models.cache import ArraysCache

from moespresso.runtime.qwen4.gdn import (
    Qwen4GDNAdapter, Qwen4GDNState, Qwen4GatedDeltaNet, qwen4_gdn_prerouter_step,
)
from moespresso.runtime.qwen4.model import Qwen4DecoderLayer
from moespresso.runtime.qwen4.mtp_gdn import Qwen4MTPGDNPair
from moespresso.runtime.qwen4.primitives import Qwen4GatedResidual, qwen4_hc_call_counts
from test_qwen4_model_shell import _model_with_real_gdn


def _cache(state=None):
    cache = ArraysCache(size=2)
    if state is not None:
        cache.state = [mx.array(value) for value in state]
    return cache


@pytest.mark.parametrize("history", [0, 1, 5])
@pytest.mark.parametrize("first_valid", [False, True])
def test_gdn_pair_prefix_matches_serial_and_rejection_continuation(history, first_valid):
    module = _model_with_real_gdn().layers[0].mixer.module
    rng = np.random.default_rng(93)
    rows = mx.array(rng.normal(size=(1, history + 3, 2)).astype(np.float32))
    base = _cache()
    if history:
        mx.eval(module(rows[:, :history], cache=base))
    serial, batched = _cache(base.state) if history else _cache(), _cache(base.state) if history else _cache()
    inputs = rows[:, history:history + 2]
    valid = mx.array([[first_valid, True]])
    first = module(inputs[:, :1], mask=valid[:, :1], cache=serial)
    prefix = [mx.array(value) for value in serial.state]
    second = module(inputs[:, 1:], mask=valid[:, 1:], cache=serial)
    checkpoints = []
    joined = module(inputs, mask=valid, cache=batched, prefix_states=checkpoints)
    mx.eval(first, second, joined, *prefix, *serial.state, *batched.state, *checkpoints[0])
    np.testing.assert_allclose(np.asarray(joined), np.asarray(mx.concatenate([first, second], axis=1)), rtol=2e-5, atol=2e-6)
    for actual, wanted in zip((*checkpoints[0], *batched.state), (*prefix, *serial.state), strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(wanted), rtol=2e-5, atol=2e-6)
    resumed, expected = _cache(checkpoints[0]), _cache(prefix)
    got = module(rows[:, -1:], cache=resumed)
    want = module(rows[:, -1:], cache=expected)
    mx.eval(got, want, *resumed.state, *expected.state)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-5, atol=2e-6)
    for actual, wanted in zip(resumed.state, expected.state, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(wanted), rtol=2e-5, atol=2e-6)


def test_gdn_prefix_capture_refuses_unsupported_rows_and_reused_capture():
    module = _model_with_real_gdn().layers[0].mixer.module
    with pytest.raises(ValueError, match="two rows"):
        module(mx.zeros((1, 1, 2)), cache=_cache(), prefix_states=[])
    with pytest.raises(ValueError, match="empty checkpoint"):
        module(mx.zeros((1, 2, 2)), cache=_cache(), prefix_states=[(None, None)])


def _make_quant_layer():
    module = Qwen4GatedDeltaNet(SimpleNamespace(
        hidden_size=2560, linear_num_value_heads=48, linear_num_key_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    ))
    for name, output, width in (("in_proj_qkv", 10240, 2560), ("in_proj_z", 6144, 2560),
                                ("in_proj_b", 48, 2560), ("in_proj_a", 48, 2560),
                                ("out_proj", 2560, 6144)):
        setattr(module, name, KQuantLinear(width, output, False, "q6_k"))
    rng = np.random.default_rng(44)
    module.conv1d.weight = mx.array(rng.normal(scale=0.2, size=(10240, 4, 1)).astype(np.float32)).astype(mx.bfloat16)
    module.A_log = mx.zeros((48,), dtype=mx.bfloat16)
    module.dt_bias = mx.zeros((48,), dtype=mx.bfloat16)
    module.norm.weight = mx.ones((128,), dtype=mx.bfloat16)

    def residual():
        value = Qwen4GatedResidual(2560, 4, 320, eps=1e-6)
        value.hc_norm.weight = mx.zeros((10240,), dtype=mx.bfloat16)
        value.input_mix_weight_down = KQuantLinear(10240, 320, False, "q6_k")
        value.block_inject_weight = KQuantLinear(10240, 4, False, "q6_k")
        value.input_mix_weight_up = KQuantLinear(320, 10240, False, "q8_0")
        return value

    layer = Qwen4DecoderLayer(mixer_kind="gdn", mixer=Qwen4GDNAdapter(module),
                             attention_residual=residual(), mlp_residual=residual(), mlp=nn.Identity())
    layer.eval()
    mx.eval(layer.parameters())
    return layer


@pytest.fixture(scope="module")
def quant_layer():
    return _make_quant_layer()


def _inputs():
    rng = np.random.default_rng(48)
    hidden = mx.array(rng.normal(scale=0.2, size=(1, 2, 10240)).astype(np.float32)).astype(mx.bfloat16)
    state = Qwen4GDNState(
        mx.array(rng.normal(scale=0.02, size=(1, 3, 10240)).astype(np.float32)).astype(mx.bfloat16),
        mx.array(rng.normal(scale=0.002, size=(1, 48, 128, 128)).astype(np.float32)), 16,
    )
    return hidden, state


@pytest.mark.parametrize("pending", [False, True])
def test_compiled_mtp_gdn_preserves_native_rows_and_both_checkpoints(quant_layer, pending):
    hidden, state = _inputs()
    output = mx.ones((1, 2, 2560), dtype=mx.bfloat16) * 0.02 if pending else None
    injection = mx.full((1, 2, 4), 0.25, dtype=mx.bfloat16) if pending else None
    first = qwen4_gdn_prerouter_step(
        quant_layer, hidden[:, :1], state=state, certified_all_valid=True,
        pending_output=None if output is None else output[:, :1],
        pending_injection=None if injection is None else injection[:, :1],
    )
    second = qwen4_gdn_prerouter_step(
        quant_layer, hidden[:, 1:], state=first.state, certified_all_valid=True,
        pending_output=None if output is None else output[:, 1:],
        pending_injection=None if injection is None else injection[:, 1:],
    )
    before = qwen4_hc_call_counts()
    pair = Qwen4MTPGDNPair(quant_layer)
    result = pair(hidden, state=state, pending_output=output, pending_injection=injection)
    mx.eval(result.mlp_hidden, result.first_state.recurrent_state, result.final_state.recurrent_state)
    after = qwen4_hc_call_counts()
    for name in ("mlp_hidden", "residual", "injection"):
        expected = mx.concatenate([getattr(first, name), getattr(second, name)], axis=1)
        actual = getattr(result, name)
        np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32)))
    for actual, expected in ((result.first_state, first.state), (result.final_state, second.state)):
        assert actual.offset == expected.offset
        for name in ("conv_state", "recurrent_state"):
            np.testing.assert_array_equal(np.asarray(getattr(actual, name).astype(mx.float32)),
                                          np.asarray(getattr(expected, name).astype(mx.float32)))
    assert after["native_norm_reads"] - before["native_norm_reads"] == 4
    assert after["fused_projection_reads"] - before["fused_projection_reads"] == 4
    assert after["generic_reads"] == before["generic_reads"]
    resumed = pair(hidden, state=result.first_state, pending_output=output, pending_injection=injection)
    assert resumed.first_state.offset == state.offset + 2
    assert resumed.final_state.offset == state.offset + 3


def test_mtp_gdn_frontier_is_not_a_compilation_shape(quant_layer, monkeypatch):
    hidden, state = _inputs()
    original = Qwen4MTPGDNPair._compute
    traces = []

    def observe(self, *args):
        traces.append(True)
        return original(self, *args)

    monkeypatch.setattr(Qwen4MTPGDNPair, "_compute", observe)
    pair = Qwen4MTPGDNPair(quant_layer)
    for offset in (16, 257, 8192):
        result = pair(hidden, state=replace(state, offset=offset))
        mx.eval(result.mlp_hidden, result.first_state.recurrent_state, result.final_state.recurrent_state)
        assert result.final_state.offset == offset + 2
    assert len(traces) == 1


def test_mtp_gdn_refuses_invalid_frontier_and_partial_pending_inputs(quant_layer):
    hidden, state = _inputs()
    pair = Qwen4MTPGDNPair(quant_layer)
    for offset in (0, -1, True):
        with pytest.raises(ValueError, match="live state"):
            pair(hidden, state=replace(state, offset=offset))
    with pytest.raises(ValueError, match="together"):
        pair(hidden, state=state, pending_output=mx.zeros((1, 2, 2560), dtype=mx.bfloat16))


def test_native_pair_nonzero_projections_preserve_prefix_and_rejection_continuation():
    layer = _make_quant_layer()
    module = layer.mixer.module
    rng = np.random.default_rng(173)
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        projection = getattr(module, name)
        packed = rng.integers(0, 256, tuple(projection.weight.shape), dtype=np.uint8)
        blocks = packed.reshape(packed.shape[0], -1, 210)
        blocks[..., 192:208] = 1
        blocks[..., 208:210] = np.asarray([0.001], dtype=np.float16).view(np.uint8)
        projection.weight = mx.array(packed)
    mx.eval(layer.parameters())
    hidden, state = _inputs()
    pair = Qwen4MTPGDNPair(layer)
    actual = pair(hidden, state=state)
    first = qwen4_gdn_prerouter_step(
        layer, hidden[:, :1], state=state, certified_all_valid=True,
        pending_output=None, pending_injection=None,
    )
    second = qwen4_gdn_prerouter_step(
        layer, hidden[:, 1:], state=first.state, certified_all_valid=True,
        pending_output=None, pending_injection=None,
    )
    assert float(mx.max(mx.abs(actual.first_state.recurrent_state)).item()) > 0.01
    for name in ("mlp_hidden", "residual", "injection"):
        expected = mx.concatenate([getattr(first, name), getattr(second, name)], axis=1)
        value = getattr(actual, name).astype(mx.float32)
        assert bool(mx.all(mx.isfinite(value)))
        np.testing.assert_allclose(np.asarray(value), np.asarray(expected.astype(mx.float32)),
                                   rtol=0.02, atol=0.002)
    for got, expected in ((actual.first_state, first.state), (actual.final_state, second.state)):
        for name in ("conv_state", "recurrent_state"):
            value = getattr(got, name).astype(mx.float32)
            assert bool(mx.all(mx.isfinite(value)))
            np.testing.assert_allclose(np.asarray(value), np.asarray(getattr(expected, name).astype(mx.float32)),
                                       rtol=0.02, atol=0.002)
    continuation = hidden[:, ::-1]
    resumed = pair(continuation, state=actual.first_state)
    reference = pair(continuation, state=first.state)
    np.testing.assert_allclose(np.asarray(resumed.mlp_hidden.astype(mx.float32)),
                               np.asarray(reference.mlp_hidden.astype(mx.float32)), rtol=0.02, atol=0.002)
    np.testing.assert_allclose(np.asarray(resumed.final_state.recurrent_state),
                               np.asarray(reference.final_state.recurrent_state), rtol=0.02, atol=0.002)
