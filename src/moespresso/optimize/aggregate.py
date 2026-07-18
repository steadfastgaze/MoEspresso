"""Aggregate per-unit quality into the proxy scalars.

Pure, format-neutral functions. They take plain (importance, quality) and
(layer, quality) pairs (never weights, never a model) so the optimizer can score
an allocation cheaply from the probe's precomputed q-tables.

  - fidelity F: importance-weighted mean quality (the calibrated target).
  - worst-layer tail T_alpha: CVaR over per-layer minima (the hard constraint).
    A layer's health is its weakest unit; the tail averages the worst few layers
    so a single catastrophic layer cannot hide behind a good mean.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def fidelity(units: Iterable[tuple[float, float]]) -> float:
    """Importance-weighted mean quality. `units` = (importance, quality) pairs.

    Falls back to the unweighted mean when total importance is non-positive, so a
    fully-unmapped set still gets a meaningful score rather than 0/0.
    """
    items = list(units)
    if not items:
        return 1.0
    total_w = sum(w for w, _ in items)
    if total_w > 0:
        return sum(w * q for w, q in items) / total_w
    return sum(q for _, q in items) / len(items)


def layer_minima(units: Iterable[tuple[int, float]]) -> dict[int, float]:
    """Worst (min) quality per layer. `units` = (layer, quality) pairs."""
    out: dict[int, float] = {}
    for layer, q in units:
        if layer not in out or q < out[layer]:
            out[layer] = q
    return out


def cvar(values: Sequence[float], alpha: float) -> float:
    """Mean of the lowest ceil(alpha*N) values (CVaR / expected shortfall).

    alpha in (0, 1]; alpha=1 is the plain mean, alpha->0 approaches the minimum.
    Empty input returns 1.0 (nothing to penalize).
    """
    if not values:
        return 1.0
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    k = max(1, math.ceil(alpha * len(values)))
    worst = sorted(values)[:k]
    return sum(worst) / len(worst)


def worst_layer_tail(units: Iterable[tuple[int, float]], alpha: float = 0.05) -> float:
    """CVaR over per-layer minima: the tail statistic T_alpha."""
    return cvar(list(layer_minima(units).values()), alpha)
