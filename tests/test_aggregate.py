"""Proxy aggregation: fidelity, per-layer minima, CVaR tail.

Includes the worked example: a high mean F can sit above a catastrophic tail.
"""

from __future__ import annotations

import numpy as np
import pytest

from moespresso.optimize.aggregate import cvar, fidelity, layer_minima, worst_layer_tail


def test_fidelity_weighted_mean():
    assert fidelity([(3.0, 0.5), (1.0, 1.0)]) == pytest.approx(0.625)


def test_fidelity_uniform_weights_is_plain_mean():
    assert fidelity([(1.0, 0.2), (1.0, 0.8)]) == pytest.approx(0.5)


def test_fidelity_zero_importance_falls_back_to_mean():
    assert fidelity([(0.0, 0.4), (0.0, 0.6)]) == pytest.approx(0.5)


def test_layer_minima_takes_worst_per_layer():
    units = [(0, 0.9), (0, 0.7), (0, 0.95), (1, 0.99), (1, 0.5)]
    assert layer_minima(units) == {0: 0.7, 1: 0.5}


def test_cvar_is_mean_of_worst_k():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # N=10
    assert cvar(vals, 0.2) == pytest.approx(0.15)  # mean(0.1, 0.2)


def test_cvar_alpha_one_is_mean():
    assert cvar([0.2, 0.4, 0.9], 1.0) == pytest.approx(0.5)


def test_cvar_rounds_up_fraction():
    vals = [0.80, 0.81, 0.82] + [0.99] * 45  # N=48, alpha=0.05 -> ceil(2.4)=3
    assert cvar(vals, 0.05) == pytest.approx((0.80 + 0.81 + 0.82) / 3)


def test_cvar_single_value():
    assert cvar([0.42], 0.05) == pytest.approx(0.42)


def test_cvar_rejects_bad_alpha():
    with pytest.raises(ValueError):
        cvar([0.5], 0.0)
    with pytest.raises(ValueError):
        cvar([0.5], 1.5)


def test_worked_example_high_mean_hides_catastrophic_tail():
    rng = np.random.default_rng(0)
    units_layer, units_imp = [], []
    for layer in range(48):
        worst_q = 0.80 if layer < 3 else 0.99
        units_layer.append((layer, worst_q))
        units_imp.append((1.0, worst_q))
        for _ in range(7):
            q = 0.995
            units_layer.append((layer, q))
            units_imp.append((float(rng.uniform(0.5, 1.5)), q))
    f = fidelity(units_imp)
    tail = worst_layer_tail(units_layer, alpha=0.05)
    assert f > 0.95
    assert tail == pytest.approx(0.80, abs=1e-9)
    assert tail < 0.95
