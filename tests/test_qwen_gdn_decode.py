"""Exactness, engagement, and fallback tests for gated-delta decode fusion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

if not mx.metal.is_available():
    pytest.skip("gated-delta decode fusion requires Metal", allow_module_level=True)

from mlx_lm.models.cache import ArraysCache  # noqa: E402
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs  # noqa: E402

from moespresso.runtime.qwen import gdn_decode as gd  # noqa: E402
from moespresso.runtime.qwen.gdn_decode import (  # noqa: E402
    FusedDecodeGatedDeltaNet,
    fused_gdn_decode_stats,
    install_fused_gdn_decode,
)


class _FixedProjection(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __call__(self, inputs):
        return mx.broadcast_to(self.value, (*inputs.shape[:-1], self.value.shape[-1]))


class _SliceProjection(nn.Module):
    def __call__(self, inputs):
        return inputs[..., : gd._HIDDEN_SIZE]


class _EchoInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __call__(self, inputs, mask=None, cache=None):
        del mask, cache
        self.calls += 1
        return inputs


def _constant_case(state_rows, qkv_value, taps):
    state = mx.broadcast_to(
        mx.array(state_rows, dtype=mx.float32).reshape(1, gd._STATE_ROWS, 1),
        (1, gd._STATE_ROWS, gd._CONV_DIM),
    )
    qkv = mx.full((1, 1, gd._CONV_DIM), qkv_value, dtype=mx.bfloat16)
    weight = mx.broadcast_to(
        mx.array(taps, dtype=mx.float32).reshape(1, gd._KERNEL_SIZE, 1),
        (gd._CONV_DIM, gd._KERNEL_SIZE, 1),
    )
    return state, qkv, weight


def _reference_conv_state(state, qkv, weight):
    conv = nn.Conv1d(
        in_channels=gd._CONV_DIM,
        out_channels=gd._CONV_DIM,
        kernel_size=gd._KERNEL_SIZE,
        groups=gd._CONV_DIM,
        bias=False,
    )
    conv.weight = weight
    conv_input = mx.concatenate([state, qkv], axis=1)
    new_state = mx.contiguous(conv_input[:, -gd._STATE_ROWS :, :])
    return conv(conv_input), new_state


def _bitwise_equal(left, right):
    return bool(mx.array_equal(mx.view(left, mx.uint32), mx.view(right, mx.uint32)))


@pytest.mark.parametrize(
    ("state_rows", "qkv_value", "taps"),
    [
        ((0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0, 0.0)),
        ((1.0, 2.0, 4.0), 8.0, (1.0, 16.0, 256.0, 4096.0)),
        (
            (2.0**20, -(2.0**20), 2.0**-20),
            -(2.0**-20),
            (1.0, 1.0, 1.0, 1.0),
        ),
        ((-0.0, 0.0, -0.0), -0.0, (1.0, -1.0, 0.0, -0.0)),
        (
            (2.0**-149, -(2.0**-149), 2.0**-126),
            2.0**-133,
            (1.0, -1.0, 2.0**-149, 2.0**-126),
        ),
    ],
)
def test_fused_conv_state_is_bit_identical_on_edge_cases(
    state_rows,
    qkv_value,
    taps,
):
    state, qkv, weight = _constant_case(state_rows, qkv_value, taps)
    reference_conv, reference_state = _reference_conv_state(state, qkv, weight)
    fused_conv, fused_state = gd._fused_conv_state(state, qkv, weight)
    mx.eval(reference_conv, reference_state, fused_conv, fused_state)
    assert _bitwise_equal(fused_conv, reference_conv)
    assert _bitwise_equal(fused_state, reference_state)


def _fixed_projection(width, dtype, key):
    value = 0.01 * mx.random.normal((width,), key=key)
    return _FixedProjection(value.astype(dtype))


def _served_inner(seed=501):
    args = TextModelArgs(
        hidden_size=gd._HIDDEN_SIZE,
        linear_num_value_heads=gd._NUM_V_HEADS,
        linear_num_key_heads=gd._NUM_K_HEADS,
        linear_key_head_dim=gd._HEAD_K_DIM,
        linear_value_head_dim=gd._HEAD_V_DIM,
        linear_conv_kernel_dim=gd._KERNEL_SIZE,
    )
    inner = GatedDeltaNet(args)
    keys = mx.random.split(mx.random.key(seed), 7)
    inner.in_proj_qkv = _fixed_projection(gd._CONV_DIM, mx.bfloat16, keys[0])
    inner.in_proj_z = _fixed_projection(gd._VALUE_DIM, mx.bfloat16, keys[1])
    inner.in_proj_b = _fixed_projection(gd._NUM_V_HEADS, mx.float32, keys[2])
    inner.in_proj_a = _fixed_projection(gd._NUM_V_HEADS, mx.float32, keys[3])
    inner.conv1d.weight = (
        0.01
        * mx.random.normal(
            (gd._CONV_DIM, gd._KERNEL_SIZE, 1),
            key=keys[4],
        )
    ).astype(mx.float32)
    inner.A_log = mx.zeros((gd._NUM_V_HEADS,), dtype=mx.float32)
    inner.dt_bias = mx.zeros((gd._NUM_V_HEADS,), dtype=mx.float32)
    inner.out_proj = _SliceProjection()
    inner.eval()
    mx.eval(inner.parameters())
    return inner, keys[5], keys[6]


def _decode_cache(conv_state, recurrent_state):
    result = ArraysCache(2)
    result[0] = conv_state
    result[1] = recurrent_state
    return result


@pytest.mark.parametrize("rms_scale_fused", [False, True])
def test_full_fused_decode_matches_stock_output_and_both_cache_members(
    monkeypatch,
    rms_scale_fused,
):
    monkeypatch.setattr(gd, "_QWEN_GDN_CONV_STATE_FUSED", True)
    monkeypatch.setattr(gd, "_QWEN_GDN_RMS_SCALE_FUSED", rms_scale_fused)
    inner, input_key, state_key = _served_inner()
    wrapped = FusedDecodeGatedDeltaNet(inner)
    wrapped.eval()

    inputs = mx.random.normal((1, 1, gd._HIDDEN_SIZE), key=input_key).astype(
        mx.float32
    )
    conv_state = mx.random.normal(
        (1, gd._STATE_ROWS, gd._CONV_DIM), key=state_key
    ).astype(mx.float32)
    recurrent_state = mx.zeros(
        (1, gd._NUM_V_HEADS, gd._HEAD_V_DIM, gd._HEAD_K_DIM),
        dtype=mx.float32,
    )
    stock_cache = _decode_cache(conv_state, recurrent_state)
    fused_cache = _decode_cache(conv_state, recurrent_state)

    stock_output = inner(inputs, cache=stock_cache)
    fused_output = wrapped(inputs, cache=fused_cache)
    mx.eval(
        stock_output,
        fused_output,
        stock_cache[0],
        stock_cache[1],
        fused_cache[0],
        fused_cache[1],
    )

    assert wrapped.fused_calls == 1
    assert wrapped.rms_scale_fused_calls == int(rms_scale_fused)
    assert _bitwise_equal(fused_output, stock_output)
    assert _bitwise_equal(fused_cache[0], stock_cache[0])
    assert _bitwise_equal(fused_cache[1], stock_cache[1])


def test_consecutive_decode_reuses_lazy_cache_bit_identically(monkeypatch):
    monkeypatch.setattr(gd, "_QWEN_GDN_CONV_STATE_FUSED", True)
    monkeypatch.setattr(gd, "_QWEN_GDN_RMS_SCALE_FUSED", False)
    inner, input_key, state_key = _served_inner(seed=601)
    wrapped = FusedDecodeGatedDeltaNet(inner)
    wrapped.eval()

    conv_state = mx.random.normal(
        (1, gd._STATE_ROWS, gd._CONV_DIM), key=state_key
    ).astype(mx.float32)
    recurrent_state = mx.zeros(
        (1, gd._NUM_V_HEADS, gd._HEAD_V_DIM, gd._HEAD_K_DIM),
        dtype=mx.float32,
    )
    stock_cache = _decode_cache(conv_state, recurrent_state)
    fused_cache = _decode_cache(conv_state, recurrent_state)
    cache_snapshots = []

    for key in mx.random.split(input_key, 3):
        inputs = mx.random.normal((1, 1, gd._HIDDEN_SIZE), key=key).astype(
            mx.float32
        )
        stock_output = inner(inputs, cache=stock_cache)
        fused_output = wrapped(inputs, cache=fused_cache)
        # Serving evaluates the model output before the next token. Leave both
        # cache siblings lazy here and reuse them on the next iteration.
        mx.eval(stock_output, fused_output)
        assert _bitwise_equal(fused_output, stock_output)
        cache_snapshots.append(
            (stock_cache[0], stock_cache[1], fused_cache[0], fused_cache[1])
        )

    mx.eval(*(array for snapshot in cache_snapshots for array in snapshot))
    for stock_conv, stock_recurrent, fused_conv, fused_recurrent in cache_snapshots:
        assert _bitwise_equal(fused_conv, stock_conv)
        assert _bitwise_equal(fused_recurrent, stock_recurrent)
    assert wrapped.fused_calls == 3


def test_prefill_and_masked_calls_delegate_to_inner(monkeypatch):
    monkeypatch.setattr(gd, "_QWEN_GDN_CONV_STATE_FUSED", True)
    inner = _EchoInner()
    inner.eval()
    wrapped = FusedDecodeGatedDeltaNet(inner)
    wrapped.eval()

    prefill = mx.zeros((1, 2, gd._HIDDEN_SIZE), dtype=mx.float32)
    decode = mx.zeros((1, 1, gd._HIDDEN_SIZE), dtype=mx.float32)
    assert wrapped(prefill) is prefill
    assert wrapped(decode, mask=mx.ones((1, 1), dtype=mx.bool_)) is decode
    assert wrapped.fused_calls == 0
    assert wrapped.fallback_input == 1
    assert wrapped.fallback_mask == 1
    assert inner.calls == 2


def test_lengths_cache_delegates_without_advancing_in_wrapper(monkeypatch):
    monkeypatch.setattr(gd, "_QWEN_GDN_CONV_STATE_FUSED", True)
    inner = _EchoInner()
    inner.eval()
    wrapped = FusedDecodeGatedDeltaNet(inner)
    wrapped.eval()
    cache = ArraysCache(2)
    cache.lengths = mx.array([1])
    inputs = mx.zeros((1, 1, gd._HIDDEN_SIZE), dtype=mx.float32)
    assert wrapped(inputs, cache=cache) is inputs
    assert wrapped.fallback_cache == 1
    assert inner.calls == 1


def _model_with_linear_layers(modules):
    layers = [SimpleNamespace(is_linear=True, linear_attn=module) for module in modules]
    layers.append(SimpleNamespace(is_linear=False))
    return SimpleNamespace(layers=layers)


def test_install_is_guarded_and_idempotent(monkeypatch):
    model = _model_with_linear_layers([_EchoInner(), _EchoInner()])
    monkeypatch.setattr(gd, "_QWEN_GDN_CONV_STATE_FUSED", False)
    assert install_fused_gdn_decode(model) == 0
    assert isinstance(model.layers[0].linear_attn, _EchoInner)

    monkeypatch.setattr(gd, "_QWEN_GDN_CONV_STATE_FUSED", True)
    monkeypatch.setattr(gd, "_kernel_available", lambda: True)
    assert install_fused_gdn_decode(model) == 2
    assert isinstance(model.layers[0].linear_attn, FusedDecodeGatedDeltaNet)
    assert isinstance(model.layers[1].linear_attn, FusedDecodeGatedDeltaNet)
    assert install_fused_gdn_decode(model) == 0


def test_install_rejects_uncertified_mlx_lm(monkeypatch):
    model = _model_with_linear_layers([_EchoInner()])
    monkeypatch.setattr(gd, "_QWEN_GDN_CONV_STATE_FUSED", True)
    monkeypatch.setattr(gd, "_MLX_LM_VERSION", "99.0.0")
    monkeypatch.setattr(gd, "_kernel_available", lambda: True)
    assert install_fused_gdn_decode(model) == 0
    assert isinstance(model.layers[0].linear_attn, _EchoInner)


def test_stats_aggregate_all_wrapped_layers(monkeypatch):
    monkeypatch.setattr(gd, "_QWEN_GDN_CONV_STATE_FUSED", True)
    monkeypatch.setattr(gd, "_kernel_available", lambda: True)
    model = _model_with_linear_layers([_EchoInner(), _EchoInner()])
    install_fused_gdn_decode(model)
    model.layers[0].linear_attn.fused_calls = 3
    model.layers[1].linear_attn.fused_calls = 4
    model.layers[0].linear_attn.rms_scale_fused_calls = 2
    model.layers[1].linear_attn.rms_scale_fused_calls = 3
    model.layers[0].linear_attn.fallback_cache = 2
    model.layers[1].linear_attn.fallback_cache = 5
    stats = fused_gdn_decode_stats(model)
    assert stats["wrapped_layers"] == 2
    assert stats["fused_calls"] == 7
    assert stats["rms_scale_fused_calls"] == 5
    assert stats["fallback_cache"] == 7
