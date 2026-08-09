"""Numeric coverage for the DS4 Q8 hC-post fusions on real q8_0 wire.

Commissioned by the adversarial verification pass (errata E4). The rest of
the suite drives these fusions through a stubbed kernel, which proves the
call contract and the routing but never the arithmetic: correctness rested
on served digest rails, which `make test` does not run. These tests call the
real `mlx_kquant` fused kernels on a real q8_0 wire and compare them against
the composition each one replaces.

The comparison uses the same quantized matmul the stock route uses, so the
only thing under test is the fused epilogue rather than the quantized
matmul's own error.

Both comparisons are exact. The fused epilogue reduces in the same order
and the same precision as the composition it replaces, so any drift is a
real change of arithmetic rather than rounding noise: a twelve-seed sweep
of these two cells found zero differing elements out of 3,072 attention
and 196,608 shared-FFN outputs. An exactness failure here is a signal to
read the kernel, not to widen the bound.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
kq = pytest.importorskip("mlx_kquant")

HC = 4
# The attention fusion accepts a reduced geometry; the shared-FFN kernel
# declares the production one and refuses anything else, so its cells run at
# the served widths.
HIDDEN = 64
INPUT = 256
FFN_HIDDEN = 4096
FFN_INPUT = 2048


def _hc_post_reference(x, residual, post, comb):
    """The stock `_hc_post` body from `runtime/deepseek_v4/model.py`.

    DS4's expand kernels read the split-combine matrix transposed relative
    to the row-major Sinkhorn buffer, which is why `comb` is swapped here.
    """
    return post[..., None] * x[..., None, :].astype(mx.float32) + mx.matmul(
        mx.swapaxes(comb, -1, -2).astype(mx.float32),
        residual.astype(mx.float32),
    )


def _wire(rng, out_features, in_features):
    weights = (rng.standard_normal((out_features, in_features)) * 0.05)
    return kq.quantize(mx.array(weights.astype(np.float32)), "q8_0")


def _hc_operands(rng, hidden=HIDDEN):
    residual = mx.array(
        (rng.standard_normal((HC, hidden)) * 0.3).astype(np.float32))
    post = mx.array((rng.standard_normal(HC) * 0.5).astype(np.float32))
    comb = mx.array((rng.standard_normal((HC, HC)) * 0.4).astype(np.float32))
    return residual, post, comb


def test_attention_hc_post_fusion_matches_the_composed_route():
    """`qmv` then hC-post, fused, equals the two run separately."""
    rng = np.random.default_rng(11)
    wire, scales = _wire(rng, HIDDEN, INPUT)
    x = mx.array(
        (rng.standard_normal((1, 1, INPUT)) * 0.5).astype(np.float32))
    residual, post, comb = _hc_operands(rng)

    fused = kq.quantized_matmul_qmv_hc_post(
        x, wire, scales, residual, post, comb, "q8_0")

    projected = kq.quantized_matmul(x, wire, scales, "q8_0", transpose=True)
    composed = _hc_post_reference(
        projected.reshape(HIDDEN), residual, post, comb)

    mx.eval(fused, composed)
    assert fused.shape == composed.shape
    np.testing.assert_array_equal(
        np.asarray(fused, dtype=np.float32),
        np.asarray(composed, dtype=np.float32))


def test_ffn_hc_post_fusion_matches_the_composed_route():
    """Shared-expert down `qmv`, the routed add, and hC-post as one epilogue."""
    rng = np.random.default_rng(23)
    wire, scales = _wire(rng, FFN_HIDDEN, FFN_INPUT)
    x = mx.array(
        (rng.standard_normal((1, 1, FFN_INPUT)) * 0.5).astype(np.float32)
    ).astype(mx.bfloat16)
    routed = mx.array(
        (rng.standard_normal(FFN_HIDDEN) * 0.2).astype(np.float32)
    ).astype(mx.float16)
    residual, post, comb = _hc_operands(rng, FFN_HIDDEN)

    fused = kq.quantized_matmul_qmv_add_hc_post(
        x, wire, scales, routed, residual, post, comb, "q8_0")

    projected = kq.quantized_matmul(x, wire, scales, "q8_0", transpose=True)
    summed = projected.reshape(FFN_HIDDEN).astype(mx.float32) + routed.astype(
        mx.float32)
    composed = _hc_post_reference(summed, residual, post, comb)

    mx.eval(fused, composed)
    assert fused.shape == composed.shape
    np.testing.assert_array_equal(
        np.asarray(fused, dtype=np.float32),
        np.asarray(composed, dtype=np.float32))


def test_ffn_fusion_actually_consumes_the_routed_row():
    """A fused epilogue that ignored `routed` would still match on zeros.

    Two runs differing only in `routed` must differ in the output, so the
    test above cannot pass through an epilogue that drops the add.
    """
    rng = np.random.default_rng(31)
    wire, scales = _wire(rng, FFN_HIDDEN, FFN_INPUT)
    x = mx.array(
        (rng.standard_normal((1, 1, FFN_INPUT)) * 0.5).astype(np.float32)
    ).astype(mx.bfloat16)
    residual, post, comb = _hc_operands(rng, FFN_HIDDEN)
    zero = mx.zeros((FFN_HIDDEN,), dtype=mx.float16)
    nonzero = mx.array(
        np.full(FFN_HIDDEN, 0.5, dtype=np.float32)).astype(mx.float16)

    with_zero = kq.quantized_matmul_qmv_add_hc_post(
        x, wire, scales, zero, residual, post, comb, "q8_0")
    with_value = kq.quantized_matmul_qmv_add_hc_post(
        x, wire, scales, nonzero, residual, post, comb, "q8_0")

    mx.eval(with_zero, with_value)
    delta = float(mx.max(mx.abs(with_value - with_zero)).item())
    assert delta > 1e-3, "the fused epilogue ignored the routed row"
