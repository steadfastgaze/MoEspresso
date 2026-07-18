"""Activation-weighted quality: the proxy core.

Pins the behavior the optimizer relies on: uniform importance == NMSE, dead
channels are free, the metric is linear in per-channel importance, NaN/Inf floors
to the worst score, and row-subsampling preserves the ratio.
"""

from __future__ import annotations

import numpy as np

from moespresso.probe.quality import Q_FLOOR, activation_weighted_quality, cosine


def _nmse(W, Wh):
    return 1.0 - float(np.square(W - Wh).sum()) / float(np.square(W).sum())


def test_uniform_importance_equals_nmse():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 7)).astype(np.float32)
    Wh = (W + 0.05 * rng.standard_normal((16, 7))).astype(np.float32)
    q = activation_weighted_quality(W, Wh, np.ones(7, np.float32))
    np.testing.assert_allclose(q, _nmse(W, Wh), rtol=1e-6)


def test_dead_channel_is_free():
    W = np.ones((4, 3), np.float32)
    Wh = W.copy()
    Wh[:, 0] = 0.0
    q = activation_weighted_quality(W, Wh, np.array([0.0, 1.0, 1.0], np.float32))
    np.testing.assert_allclose(q, 1.0, atol=1e-9)


def test_linear_in_importance():
    W = np.ones((4, 3), np.float32)
    Wh = W.copy()
    Wh[:, 0] += 1.0  # error confined to column 0: col_err=[4,0,0], energy=[4,4,4]
    q1 = activation_weighted_quality(W, Wh, np.array([1.0, 1.0, 1.0], np.float32))
    q2 = activation_weighted_quality(W, Wh, np.array([2.0, 1.0, 1.0], np.float32))
    np.testing.assert_allclose(q1, 1.0 - 4.0 / 12.0)   # 0.6667
    np.testing.assert_allclose(q2, 1.0 - 8.0 / 16.0)   # 0.5


def test_nan_reconstruction_floors():
    W = np.ones((4, 3), np.float32)
    Wh = W.copy()
    Wh[0, 0] = np.nan
    assert activation_weighted_quality(W, Wh, np.ones(3, np.float32)) == Q_FLOOR


def test_catastrophic_error_clamped_to_floor():
    W = np.ones((4, 3), np.float32)
    Wh = 5.0 * W
    assert activation_weighted_quality(W, Wh, np.ones(3, np.float32)) == Q_FLOOR


def test_row_subsample_preserves_ratio():
    # Wh = 0.9*W -> exact ratio (0.1)^2 per row, so any subset yields the same q.
    rng = np.random.default_rng(1)
    W = rng.standard_normal((1000, 6)).astype(np.float32)
    Wh = (0.9 * W).astype(np.float32)
    h = np.abs(rng.standard_normal(6)).astype(np.float32) + 0.1
    full = activation_weighted_quality(W, Wh, h)
    sub = activation_weighted_quality(W[:200], Wh[:200], h)
    np.testing.assert_allclose(full, sub, rtol=1e-4)
    np.testing.assert_allclose(full, 1.0 - 0.01, rtol=1e-4)


def test_zero_importance_falls_back_to_nmse():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((8, 4)).astype(np.float32)
    Wh = (W + 0.1 * rng.standard_normal((8, 4))).astype(np.float32)
    q = activation_weighted_quality(W, Wh, np.zeros(4, np.float32))
    np.testing.assert_allclose(q, _nmse(W, Wh), rtol=1e-6)


def test_cosine_of_identical_is_one():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((8, 8)).astype(np.float32)
    np.testing.assert_allclose(cosine(W, W), 1.0, rtol=1e-6)
    assert cosine(np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32)) == 0.0
