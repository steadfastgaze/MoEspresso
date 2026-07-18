"""Quantization round-trips (mlx affine + jang TQ) on small arrays.

Skips cleanly if a heavy dep is missing. Pins the contracts the probe relies on:
shape/finiteness preserved, more bits never worse, uniform-h quality == cosine-ish
NMSE behavior, and a near-lossless round-trip scores near 1.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from moespresso.probe.roundtrip import (  # noqa: E402
    affine_quality,
    affine_roundtrip,
    mx_float_quality,
    mx_float_roundtrip,
)


def test_affine_roundtrip_shape_and_finite():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    recon = affine_roundtrip(w, bits=4, group_size=32)
    assert recon.shape == w.shape
    assert np.all(np.isfinite(recon))


def test_affine_more_bits_not_worse():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    h = np.ones(64, np.float32)
    _, q2 = affine_quality(w, 2, 32, h)
    _, q8 = affine_quality(w, 8, 32, h)
    assert q8 >= q2 - 1e-6


def test_affine_high_bits_near_lossless():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    h = np.ones(64, np.float32)
    _, q8 = affine_quality(w, 8, 32, h)
    assert q8 > 0.99


def test_mx_float_roundtrip_shape_and_finite():
    rng = np.random.default_rng(4)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    recon = mx_float_roundtrip(w, "mxfp8")
    assert recon.shape == w.shape
    assert np.all(np.isfinite(recon))


def test_mx_float_quality_is_measured_separately_from_affine():
    rng = np.random.default_rng(5)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    h = np.ones(64, np.float32)
    _, q = mx_float_quality(w, "mxfp8", h)
    assert -1.0 <= q <= 1.0


def test_tq_roundtrip_if_available():
    pytest.importorskip("jang_tools.turboquant")
    from moespresso.probe.roundtrip import tq_quality

    rng = np.random.default_rng(3)
    w = rng.standard_normal((16, 64)).astype(np.float32)
    h = np.ones(64, np.float32)
    cos4, q4 = tq_quality(w, 4, h)
    cos1, q1 = tq_quality(w, 1, h)
    assert -1.0 <= q4 <= 1.0 and -1.0 <= q1 <= 1.0
    assert cos4 >= cos1 - 1e-6  # more bits not worse
