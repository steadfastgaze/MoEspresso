from __future__ import annotations

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from moespresso.runtime.deepseek_v4 import native_moe as native_ds4_moe  # noqa: E402


def _load_or_skip():
    mod = native_ds4_moe.load_ds4_moe()
    if mod is None:
        if os.environ.get("MOESPRESSO_REQUIRE_NATIVE_DS4_MOE") == "1":
            pytest.fail("native DS4 routed-MoE module is required but unavailable")
        pytest.skip("native DS4 routed-MoE module is not built")
    return mod


def test_native_ds4_moe_weighted_sum6_matches_mlx_reference_float16():
    _load_or_skip()
    rng = np.random.default_rng(123)
    rows = mx.array(
        rng.standard_normal((2, 6, 32)).astype(np.float16),
        dtype=mx.float16,
    )
    scores = mx.array(
        np.array(
            [
                [0.31, 0.22, 0.17, 0.13, 0.09, 0.08],
                [0.08, 0.09, 0.13, 0.17, 0.22, 0.31],
            ],
            dtype=np.float16,
        ),
        dtype=mx.float16,
    )

    got = native_ds4_moe.weighted_sum6(rows, scores)
    expected = mx.sum(rows.astype(mx.float32) * scores[..., None].astype(mx.float32),
                      axis=1)
    mx.eval(got, expected)

    assert got.shape == (2, 32)
    assert got.dtype == mx.float32
    np.testing.assert_allclose(
        np.array(got),
        np.array(expected),
        rtol=1e-6,
        atol=1e-6,
    )


def test_native_ds4_moe_weighted_sum6_broadcasts_score_row():
    _load_or_skip()
    rng = np.random.default_rng(456)
    rows = mx.array(
        rng.standard_normal((3, 6, 16)).astype(np.float32),
        dtype=mx.float32,
    )
    scores = mx.array(
        np.array([[0.2, 0.18, 0.17, 0.16, 0.15, 0.14]], dtype=np.float32),
        dtype=mx.float32,
    )

    got = native_ds4_moe.weighted_sum6(rows, scores)
    expected = mx.sum(rows * scores[..., None], axis=1)
    mx.eval(got, expected)

    assert got.shape == (3, 16)
    np.testing.assert_allclose(
        np.array(got),
        np.array(expected),
        rtol=1e-6,
        atol=1e-6,
    )
