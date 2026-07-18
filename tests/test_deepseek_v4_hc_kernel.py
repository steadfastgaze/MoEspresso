"""Bit-parity and dispatch tests for the fused DS4 mHC kernels."""

import numpy as np
import pytest

from moespresso.runtime.deepseek_v4 import hc_kernel
from moespresso.runtime.deepseek_v4.model import (
    _patch_deepseek_v4_hc_fused,
    _patch_deepseek_v4_hc_post_float32,
)


def _require_metal():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    return mx


def _jang_model():
    return pytest.importorskip("jang_tools.dsv4.mlx_model")


def _assert_bit_equal(mx, expected, actual, label):
    assert expected.shape == actual.shape, label
    assert expected.dtype == actual.dtype, label
    itype = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[expected.dtype.size]
    exp = np.asarray(mx.view(expected.reshape(-1), itype))
    got = np.asarray(mx.view(actual.reshape(-1), itype))
    mismatches = int((exp != got).sum())
    assert mismatches == 0, f"{label}: {mismatches}/{exp.size} bit mismatches"


def _pre_inputs(mx, rows, hidden, dtype):
    mx.random.seed(rows + hidden)
    x = (mx.random.normal((1, rows, 4, hidden)) * 3.0).astype(dtype)
    zero_mask = mx.random.uniform(shape=x.shape) < 0.002
    x = mx.where(zero_mask, mx.array(-0.0, dtype=dtype), x)
    scale = mx.array([1.3, 0.7, 0.9], dtype=mx.float32)
    base = (mx.random.normal((24,)) * 0.5).astype(mx.float32)
    mixes = (mx.random.normal((1, rows, 24)) * 2.0).astype(mx.float32)
    return x, mixes, scale, base


def _composed_pre(mx, jm, x, mixes, scale, base, iters, eps):
    x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
    pre, post, comb = jm.hc_split_sinkhorn(mixes, scale, base, 4, iters, eps)
    y = mx.sum(pre[..., None] * mx.reshape(x_flat, x.shape), axis=2)
    return y, post, comb


def _composed_post_f32(mx, x, residual, post, comb):
    return post[..., None] * x[..., None, :].astype(mx.float32) + mx.matmul(
        mx.swapaxes(comb, -1, -2).astype(mx.float32),
        residual.astype(mx.float32),
    )


@pytest.mark.parametrize("rows,hidden", [(1, 64), (1, 4096), (5, 64), (193, 256), (61, 4096)])
def test_hc_split_weighted_sum_bit_identical(rows, hidden):
    mx = _require_metal()
    jm = _jang_model()
    iters, eps = 20, 1e-6
    x, mixes, scale, base = _pre_inputs(mx, rows, hidden, mx.float32)
    x_flat = mx.flatten(x, start_axis=2)

    y_ref, post_ref, comb_ref = _composed_pre(
        mx, jm, x, mixes, scale, base, iters, eps)
    y, post, comb = hc_kernel.hc_split_weighted_sum(
        mixes, x_flat, scale, base, iters=iters, eps=eps)
    mx.eval(y_ref, post_ref, comb_ref, y, post, comb)

    _assert_bit_equal(mx, post_ref, post, "post")
    _assert_bit_equal(mx, comb_ref, comb, "comb")
    _assert_bit_equal(mx, y_ref, y, "y")


@pytest.mark.parametrize("rows", [1, 17])
def test_hc_split_weighted_sum_other_iteration_counts(rows):
    mx = _require_metal()
    jm = _jang_model()
    x, mixes, scale, base = _pre_inputs(mx, rows, 128, mx.float32)
    x_flat = mx.flatten(x, start_axis=2)
    for iters in (1, 3):
        y_ref, post_ref, comb_ref = _composed_pre(
            mx, jm, x, mixes, scale, base, iters, 1e-6)
        y, post, comb = hc_kernel.hc_split_weighted_sum(
            mixes, x_flat, scale, base, iters=iters, eps=1e-6)
        mx.eval(y_ref, post_ref, comb_ref, y, post, comb)
        _assert_bit_equal(mx, comb_ref, comb, f"comb iters={iters}")
        _assert_bit_equal(mx, y_ref, y, f"y iters={iters}")


def _composed_pre_full(mx, jm, x, fn, scale, base, iters, eps, rms_eps):
    """The full composed pre stage from the raw operands."""
    x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
    rsqrt = mx.rsqrt(
        mx.mean(x_flat.square(), axis=-1, keepdims=True) + rms_eps)
    mixes = (x_flat @ fn.T) * rsqrt
    pre, post, comb = jm.hc_split_sinkhorn(mixes, scale, base, 4, iters, eps)
    y = mx.sum(pre[..., None] * mx.reshape(x_flat, x.shape), axis=2)
    return y, post, comb


def _tail_inputs(mx, hidden, seed, tiny=False):
    mx.random.seed(seed)
    scale_factor = 0.001 if tiny else 3.0
    x = (mx.random.normal((1, 1, 4, hidden)) * scale_factor).astype(mx.float32)
    zero_mask = mx.random.uniform(shape=x.shape) < 0.002
    x = mx.where(zero_mask, mx.array(-0.0, dtype=mx.float32), x)
    fn = (mx.random.normal((24, 4 * hidden)) * 0.02).astype(mx.float32)
    scale = mx.array([1.3, 0.7, 0.9], dtype=mx.float32)
    base = (mx.random.normal((24,)) * 0.5).astype(mx.float32)
    return x, fn, scale, base


@pytest.mark.parametrize("hidden", [1024, 4096])
@pytest.mark.parametrize("iters", [1, 3, 20])
def test_hc_split_weighted_sum_tail_bit_identical(hidden, iters):
    mx = _require_metal()
    jm = _jang_model()
    eps, rms_eps = 1e-6, 1e-6
    for seed in range(8):
        x, fn, scale, base = _tail_inputs(
            mx, hidden, 100 * hidden + seed, tiny=seed % 3 == 0)
        y_ref, post_ref, comb_ref = _composed_pre_full(
            mx, jm, x, fn, scale, base, iters, eps, rms_eps)
        x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
        mixes_raw = x_flat @ fn.T
        y, post, comb = hc_kernel.hc_split_weighted_sum_tail(
            mixes_raw, x_flat, scale, base,
            iters=iters, eps=eps, rms_eps=rms_eps)
        mx.eval(y_ref, post_ref, comb_ref, y, post, comb)
        _assert_bit_equal(mx, post_ref, post, f"tail post seed={seed}")
        _assert_bit_equal(mx, comb_ref, comb, f"tail comb seed={seed}")
        _assert_bit_equal(mx, y_ref, y, f"tail y seed={seed}")


def test_hc_tail_eligibility_and_gate_delegation(monkeypatch):
    mx = _require_metal()
    hidden = 1024
    x = mx.zeros((1, 1, 4, hidden), dtype=mx.float32)
    fn = mx.zeros((24, 4 * hidden), dtype=mx.float32)
    scale = mx.zeros((3,), dtype=mx.float32)
    base = mx.zeros((24,), dtype=mx.float32)

    def tail_ok(x_arg, fn_arg):
        return hc_kernel.hc_split_weighted_sum_tail_eligible(
            x_arg, fn_arg, scale, base, hc_mult=4, iters=20)

    assert tail_ok(x, fn)
    # Multi-row shapes stay on the composed rsqrt tail.
    assert not tail_ok(
        mx.zeros((1, 2, 4, hidden), dtype=mx.float32), fn)
    # Widths below the 1024-thread reduce regime stay composed.
    assert not tail_ok(
        mx.zeros((1, 1, 4, 512), dtype=mx.float32),
        mx.zeros((24, 4 * 512), dtype=mx.float32))

    # The tail kill switch disables only the tail.
    monkeypatch.setenv("MOESPRESSO_DSV4_HC_DECODE_TAIL", "0")
    assert not hc_kernel.hc_decode_tail_enabled()
    assert not tail_ok(x, fn)
    assert hc_kernel.hc_split_weighted_sum_eligible(
        x, fn, scale, base, hc_mult=4, iters=20)

    # The decode gate closes the tail regardless of the tail flag.
    monkeypatch.delenv("MOESPRESSO_DSV4_HC_DECODE_TAIL")
    monkeypatch.setenv("MOESPRESSO_DSV4_HC_DECODE_FUSED", "0")
    assert not hc_kernel.hc_decode_tail_enabled()
    assert not tail_ok(x, fn)


def test_hc_tail_rejects_multi_row_and_bad_widths():
    mx = _require_metal()
    x_flat = mx.zeros((1, 1, 4 * 1024), dtype=mx.float32)
    mixes_raw = mx.zeros((1, 1, 24), dtype=mx.float32)
    scale = mx.zeros((3,), dtype=mx.float32)
    base = mx.zeros((24,), dtype=mx.float32)
    with pytest.raises(ValueError):
        hc_kernel.hc_split_weighted_sum_tail(
            mx.zeros((1, 2, 24), dtype=mx.float32),
            mx.zeros((1, 2, 4 * 1024), dtype=mx.float32),
            scale, base, iters=20, eps=1e-6, rms_eps=1e-6)
    with pytest.raises(ValueError):
        hc_kernel.hc_split_weighted_sum_tail(
            mixes_raw, mx.zeros((1, 1, 4 * 512), dtype=mx.float32),
            scale, base, iters=20, eps=1e-6, rms_eps=1e-6)
    with pytest.raises(ValueError):
        hc_kernel.hc_split_weighted_sum_tail(
            mixes_raw, x_flat.astype(mx.bfloat16),
            scale, base, iters=20, eps=1e-6, rms_eps=1e-6)


@pytest.mark.parametrize("rows,hidden", [(1, 64), (1, 4096), (7, 64), (129, 4096)])
def test_hc_post_recombine_bit_identical_float32(rows, hidden):
    mx = _require_metal()
    mx.random.seed(rows)
    x = (mx.random.normal((1, rows, hidden)) * 2.0).astype(mx.float32)
    residual = (mx.random.normal((1, rows, 4, hidden)) * 3.0).astype(mx.float32)
    post = (2.0 * mx.random.uniform(shape=(1, rows, 4))).astype(mx.float32)
    comb = mx.random.uniform(shape=(1, rows, 4, 4)).astype(mx.float32)

    out_ref = _composed_post_f32(mx, x, residual, post, comb)
    out = hc_kernel.hc_post_recombine(x, residual, post, comb)
    mx.eval(out_ref, out)

    assert out.dtype == mx.float32
    _assert_bit_equal(mx, out_ref, out, "post recombine")


@pytest.mark.parametrize("rows,hidden", [(1, 4096), (33, 256)])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_hc_post_recombine_bit_identical_half_inputs(dtype_name, rows, hidden):
    mx = _require_metal()
    dtype = getattr(mx, dtype_name)
    mx.random.seed(11)
    x = (mx.random.normal((1, rows, hidden)) * 2.0).astype(dtype)
    residual = (mx.random.normal((1, rows, 4, hidden)) * 3.0).astype(dtype)
    zero_mask = mx.random.uniform(shape=residual.shape) < 0.002
    residual = mx.where(zero_mask, mx.array(-0.0, dtype=dtype), residual)
    post = (2.0 * mx.random.uniform(shape=(1, rows, 4))).astype(mx.float32)
    comb = mx.random.uniform(shape=(1, rows, 4, 4)).astype(mx.float32)

    out_ref = _composed_post_f32(mx, x, residual, post, comb)
    out = hc_kernel.hc_post_recombine(x, residual, post, comb)
    mx.eval(out_ref, out)

    assert out.dtype == mx.float32
    _assert_bit_equal(mx, out_ref, out, f"post recombine {dtype_name}")


def test_eligibility_accepts_decode_and_rejects_bad_shapes():
    mx = _require_metal()
    hidden = 64
    x = mx.zeros((1, 8, 4, hidden), dtype=mx.float32)
    fn = mx.zeros((24, 4 * hidden), dtype=mx.float32)
    scale = mx.zeros((3,), dtype=mx.float32)
    base = mx.zeros((24,), dtype=mx.float32)

    assert hc_kernel.hc_split_weighted_sum_eligible(
        x, fn, scale, base, hc_mult=4, iters=20)
    # Single-row decode chunks engage under the decode gate.
    assert hc_kernel.hc_split_weighted_sum_eligible(
        x[:, :1], fn, scale, base, hc_mult=4, iters=20)
    # Non-DS4 hc widths stay composed.
    assert not hc_kernel.hc_split_weighted_sum_eligible(
        x, fn, scale, base, hc_mult=2, iters=20)
    # A non-float32 mixer weight stays composed.
    assert not hc_kernel.hc_split_weighted_sum_eligible(
        x, fn.astype(mx.bfloat16), scale, base, hc_mult=4, iters=20)
    # Hidden sizes that break float4 alignment stay composed.
    assert not hc_kernel.hc_split_weighted_sum_eligible(
        mx.zeros((1, 8, 4, 66), dtype=mx.float32),
        mx.zeros((24, 4 * 66), dtype=mx.float32),
        scale, base, hc_mult=4, iters=20)

    xp = mx.zeros((1, 8, hidden), dtype=mx.float32)
    residual = mx.zeros((1, 8, 4, hidden), dtype=mx.float32)
    post = mx.zeros((1, 8, 4), dtype=mx.float32)
    comb = mx.zeros((1, 8, 4, 4), dtype=mx.float32)
    assert hc_kernel.hc_post_recombine_eligible(xp, residual, post, comb)
    assert hc_kernel.hc_post_recombine_eligible(
        xp[:, :1], residual[:, :1], post[:, :1], comb[:, :1])
    assert not hc_kernel.hc_post_recombine_eligible(
        xp, residual, post.astype(mx.bfloat16), comb)


def test_eligibility_gates_by_phase(monkeypatch):
    mx = _require_metal()
    hidden = 64
    x = mx.zeros((1, 8, 4, hidden), dtype=mx.float32)
    fn = mx.zeros((24, 4 * hidden), dtype=mx.float32)
    scale = mx.zeros((3,), dtype=mx.float32)
    base = mx.zeros((24,), dtype=mx.float32)
    xp = mx.zeros((1, 8, hidden), dtype=mx.float32)
    residual = mx.zeros((1, 8, 4, hidden), dtype=mx.float32)
    post = mx.zeros((1, 8, 4), dtype=mx.float32)
    comb = mx.zeros((1, 8, 4, 4), dtype=mx.float32)

    def pre_ok(rows):
        return hc_kernel.hc_split_weighted_sum_eligible(
            x[:, :rows], fn, scale, base, hc_mult=4, iters=20)

    def post_ok(rows):
        return hc_kernel.hc_post_recombine_eligible(
            xp[:, :rows], residual[:, :rows], post[:, :rows], comb[:, :rows])

    # Prefill gate off: multi-row falls back, single-row still engages.
    monkeypatch.setenv("MOESPRESSO_DSV4_HC_PREFILL_FUSED", "0")
    assert not hc_kernel.hc_prefill_fused_enabled()
    assert hc_kernel.hc_decode_fused_enabled()
    assert not pre_ok(8)
    assert pre_ok(1)
    assert not post_ok(8)
    assert post_ok(1)

    # Decode gate off: single-row falls back, multi-row still engages.
    monkeypatch.delenv("MOESPRESSO_DSV4_HC_PREFILL_FUSED")
    monkeypatch.setenv("MOESPRESSO_DSV4_HC_DECODE_FUSED", "0")
    assert hc_kernel.hc_prefill_fused_enabled()
    assert not hc_kernel.hc_decode_fused_enabled()
    assert pre_ok(8)
    assert not pre_ok(1)
    assert post_ok(8)
    assert not post_ok(1)

    # Both gates off: nothing engages.
    monkeypatch.setenv("MOESPRESSO_DSV4_HC_PREFILL_FUSED", "0")
    assert not hc_kernel.hc_fused_enabled()
    assert not pre_ok(8)
    assert not pre_ok(1)
    assert not post_ok(8)
    assert not post_ok(1)


class _HcLayer:
    """Layer stub carrying the real jang mHC stage methods."""

    def __init__(self, jm, hidden):
        self.args = jm.ModelArgs(hidden_size=hidden)
        self._hc_pre = jm.DeepseekV4DecoderLayer._hc_pre.__get__(self)
        self._hc_post = jm.DeepseekV4DecoderLayer._hc_post.__get__(self)


class _LayerContainer:
    def __init__(self, layers):
        self.layers = layers


class _WrappedModel:
    def __init__(self, layers):
        self.model = _LayerContainer(layers)


def _patched_pair(jm, hidden):
    fused = _WrappedModel([_HcLayer(jm, hidden)])
    composed = _WrappedModel([_HcLayer(jm, hidden)])
    assert _patch_deepseek_v4_hc_post_float32(fused) == 1
    assert _patch_deepseek_v4_hc_post_float32(composed) == 1
    assert _patch_deepseek_v4_hc_fused(fused) == 1
    return fused.model.layers[0], composed.model.layers[0]


def test_patched_layer_stages_bit_identical_and_counted():
    mx = _require_metal()
    jm = _jang_model()
    hidden = 128
    fused_layer, composed_layer = _patched_pair(jm, hidden)

    mx.random.seed(3)
    x = (mx.random.normal((1, 37, 4, hidden)) * 3.0).astype(mx.float32)
    fn = (mx.random.normal((24, 4 * hidden)) * 0.02).astype(mx.float32)
    scale = mx.array([1.3, 0.7, 0.9], dtype=mx.float32)
    base = (mx.random.normal((24,)) * 0.5).astype(mx.float32)

    y_f, post_f, comb_f = fused_layer._hc_pre(x, fn, scale, base)
    y_c, post_c, comb_c = composed_layer._hc_pre(x, fn, scale, base)
    mx.eval(y_f, post_f, comb_f, y_c, post_c, comb_c)
    _assert_bit_equal(mx, y_c, y_f, "patched pre y")
    _assert_bit_equal(mx, post_c, post_f, "patched pre post")
    _assert_bit_equal(mx, comb_c, comb_f, "patched pre comb")
    assert fused_layer._moespresso_dsv4_hc_fused_pre_calls == 1
    assert fused_layer._moespresso_dsv4_hc_fused_pre_decode_calls == 0
    assert fused_layer._moespresso_dsv4_hc_fused_pre_fallback_calls == 0

    block_out = (mx.random.normal((1, 37, hidden)) * 2.0).astype(mx.float32)
    out_f = fused_layer._hc_post(block_out, x, post_f, comb_f)
    out_c = composed_layer._hc_post(block_out, x, post_c, comb_c)
    mx.eval(out_f, out_c)
    assert out_f.dtype == mx.float32
    _assert_bit_equal(mx, out_c, out_f, "patched post")
    assert fused_layer._moespresso_dsv4_hc_fused_post_calls == 1
    assert fused_layer._moespresso_dsv4_hc_fused_post_decode_calls == 0
    assert fused_layer._moespresso_dsv4_hc_fused_post_fallback_calls == 0


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_patched_layer_fuses_decode_shape(dtype_name):
    mx = _require_metal()
    jm = _jang_model()
    dtype = getattr(mx, dtype_name)
    hidden = 4096
    fused_layer, composed_layer = _patched_pair(jm, hidden)

    mx.random.seed(4)
    x = (mx.random.normal((1, 1, 4, hidden)) * 3.0).astype(dtype)
    zero_mask = mx.random.uniform(shape=x.shape) < 0.002
    x = mx.where(zero_mask, mx.array(-0.0, dtype=dtype), x)
    fn = (mx.random.normal((24, 4 * hidden)) * 0.02).astype(mx.float32)
    scale = mx.array([1.3, 0.7, 0.9], dtype=mx.float32)
    base = (mx.random.normal((24,)) * 0.5).astype(mx.float32)

    y_f, post_f, comb_f = fused_layer._hc_pre(x, fn, scale, base)
    y_c, post_c, comb_c = composed_layer._hc_pre(x, fn, scale, base)
    mx.eval(y_f, post_f, comb_f, y_c, post_c, comb_c)
    assert y_f.dtype == dtype
    _assert_bit_equal(mx, post_c, post_f, "decode pre post")
    _assert_bit_equal(mx, comb_c, comb_f, "decode pre comb")
    _assert_bit_equal(mx, y_c, y_f, "decode pre y")
    assert fused_layer._moespresso_dsv4_hc_fused_pre_calls == 1
    assert fused_layer._moespresso_dsv4_hc_fused_pre_decode_calls == 1
    assert fused_layer._moespresso_dsv4_hc_fused_pre_tail_decode_calls == 1
    assert fused_layer._moespresso_dsv4_hc_fused_pre_fallback_calls == 0

    block_out = (mx.random.normal((1, 1, hidden)) * 2.0).astype(dtype)
    out_f = fused_layer._hc_post(block_out, x, post_f, comb_f)
    out_c = composed_layer._hc_post(block_out, x, post_c, comb_c)
    mx.eval(out_f, out_c)
    assert out_f.dtype == mx.float32
    _assert_bit_equal(mx, out_c, out_f, "decode post")
    assert fused_layer._moespresso_dsv4_hc_fused_post_calls == 1
    assert fused_layer._moespresso_dsv4_hc_fused_post_decode_calls == 1
    assert fused_layer._moespresso_dsv4_hc_fused_post_fallback_calls == 0


def test_patched_layer_tail_kill_switch_keeps_fused_split(monkeypatch):
    mx = _require_metal()
    jm = _jang_model()
    hidden = 4096
    fused_layer, composed_layer = _patched_pair(jm, hidden)
    monkeypatch.setenv("MOESPRESSO_DSV4_HC_DECODE_TAIL", "0")

    x, fn, scale, base = _tail_inputs(mx, hidden, 21)
    y_f, post_f, comb_f = fused_layer._hc_pre(x, fn, scale, base)
    y_c, post_c, comb_c = composed_layer._hc_pre(x, fn, scale, base)
    mx.eval(y_f, post_f, comb_f, y_c, post_c, comb_c)
    _assert_bit_equal(mx, y_c, y_f, "tail-off pre y")
    _assert_bit_equal(mx, post_c, post_f, "tail-off pre post")
    _assert_bit_equal(mx, comb_c, comb_f, "tail-off pre comb")
    assert fused_layer._moespresso_dsv4_hc_fused_pre_calls == 1
    assert fused_layer._moespresso_dsv4_hc_fused_pre_decode_calls == 1
    assert fused_layer._moespresso_dsv4_hc_fused_pre_tail_decode_calls == 0
    assert fused_layer._moespresso_dsv4_hc_fused_pre_fallback_calls == 0


def test_patched_layer_decode_kill_switch_falls_back(monkeypatch):
    mx = _require_metal()
    jm = _jang_model()
    hidden = 128
    fused_layer, composed_layer = _patched_pair(jm, hidden)
    monkeypatch.setenv("MOESPRESSO_DSV4_HC_DECODE_FUSED", "0")

    mx.random.seed(4)
    x = (mx.random.normal((1, 1, 4, hidden)) * 3.0).astype(mx.float32)
    fn = (mx.random.normal((24, 4 * hidden)) * 0.02).astype(mx.float32)
    scale = mx.array([1.3, 0.7, 0.9], dtype=mx.float32)
    base = (mx.random.normal((24,)) * 0.5).astype(mx.float32)

    y_f, post_f, comb_f = fused_layer._hc_pre(x, fn, scale, base)
    y_c, post_c, comb_c = composed_layer._hc_pre(x, fn, scale, base)
    mx.eval(y_f, post_f, comb_f, y_c, post_c, comb_c)
    _assert_bit_equal(mx, y_c, y_f, "decode pre y")
    assert fused_layer._moespresso_dsv4_hc_fused_pre_calls == 0
    assert fused_layer._moespresso_dsv4_hc_fused_pre_fallback_calls == 1

    block_out = (mx.random.normal((1, 1, hidden)) * 2.0).astype(mx.float32)
    out_f = fused_layer._hc_post(block_out, x, post_f, comb_f)
    out_c = composed_layer._hc_post(block_out, x, post_c, comb_c)
    mx.eval(out_f, out_c)
    _assert_bit_equal(mx, out_c, out_f, "decode post")
    assert fused_layer._moespresso_dsv4_hc_fused_post_calls == 0
    assert fused_layer._moespresso_dsv4_hc_fused_post_fallback_calls == 1

    # Multi-row prefill shapes still engage under their own gate.
    xw = (mx.random.normal((1, 6, 4, hidden)) * 3.0).astype(mx.float32)
    y_w, post_w, comb_w = fused_layer._hc_pre(xw, fn, scale, base)
    mx.eval(y_w, post_w, comb_w)
    assert fused_layer._moespresso_dsv4_hc_fused_pre_calls == 1
    assert fused_layer._moespresso_dsv4_hc_fused_pre_decode_calls == 0


def test_patch_disabled_by_kill_switches(monkeypatch):
    _require_metal()
    jm = _jang_model()
    monkeypatch.setenv("MOESPRESSO_DSV4_HC_PREFILL_FUSED", "0")
    monkeypatch.setenv("MOESPRESSO_DSV4_HC_DECODE_FUSED", "0")
    model = _WrappedModel([_HcLayer(jm, 64)])
    _patch_deepseek_v4_hc_post_float32(model)
    assert _patch_deepseek_v4_hc_fused(model) == 0
    assert model._moespresso_dsv4_hc_fused_layers == 0
    assert not getattr(
        model.model.layers[0], "_moespresso_dsv4_hc_fused", False)


def test_patch_installs_with_only_decode_gate(monkeypatch):
    _require_metal()
    jm = _jang_model()
    monkeypatch.setenv("MOESPRESSO_DSV4_HC_PREFILL_FUSED", "0")
    model = _WrappedModel([_HcLayer(jm, 64)])
    _patch_deepseek_v4_hc_post_float32(model)
    assert _patch_deepseek_v4_hc_fused(model) == 1
    assert model._moespresso_dsv4_hc_fused_layers == 1


def test_patch_requires_float32_post_contract():
    _require_metal()
    jm = _jang_model()
    model = _WrappedModel([_HcLayer(jm, 64)])
    # Without the float32 recombine patch the fused post contract does not
    # match the layer, so the wrap must refuse.
    assert _patch_deepseek_v4_hc_fused(model) == 0
    assert model._moespresso_dsv4_hc_fused_layers == 0
