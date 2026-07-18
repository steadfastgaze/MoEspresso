"""Router-locality analysis for SSD-streamed MoE.

Pure + import-light (no mlx). Given a trace of per-layer selected expert ids per
decode step, compute the activation histogram, a per-layer hotlist (the cache
seed), and the simulated LRU hit rate for a given per-layer cache size.

The hit rate is the decisive number for whether expert streaming is usable. A
live profiler captures the trace from a real forward pass and feeds it here; the
analysis itself never runs a model, so it is fully unit-tested.

Cache model: experts are cached per (layer, expert). Expert 5 in layer 3 is a
different weight than expert 5 in layer 7 (matching the kernel's expert*stride
addressing), so the LRU sim keeps an independent per-layer cache.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class LocalityStats:
    counts: dict[int, dict[int, int]]   # layer -> {expert_id: activation_count}
    total_activations: int
    num_layers: int
    num_experts: int
    top_k: int


def summarize_trace(trace, *, num_layers: int, num_experts: int,
                    top_k: int) -> LocalityStats:
    """Per-(layer, expert) activation counts over the whole trace."""
    counts: dict[int, dict[int, int]] = {layer: {} for layer in range(num_layers)}
    total = 0
    for step in trace:
        for layer, experts in step.items():
            bucket = counts.setdefault(layer, {})
            for e in experts:
                bucket[e] = bucket.get(e, 0) + 1
                total += 1
    return LocalityStats(counts=counts, total_activations=total,
                         num_layers=num_layers, num_experts=num_experts,
                         top_k=top_k)


def hotlist_from_counts(counts: dict[int, dict[int, int]],
                        *, per_layer: int) -> dict[int, list[int]]:
    """Top-`per_layer` experts by activation count, per layer (the cache seed)."""
    hot: dict[int, list[int]] = {}
    for layer, bucket in counts.items():
        ranked = sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))
        hot[layer] = [e for e, _ in ranked[:per_layer]]
    return hot


def simulate_lru_hit_rate(trace, *, num_layers: int, cache_per_layer: int,
                          seed: dict[int, list[int]] | None = None) -> float:
    """Simulated LRU hit rate over the trace for a per-layer cache of given size.

    Each layer has an independent LRU of `cache_per_layer` experts. For each
    selected expert: hit if resident (and refresh recency), else miss + insert
    (evicting the LRU victim). `seed` pre-warms each layer's cache (the hotlist).
    Returns hits / (hits + misses) across all layers and steps.
    """
    if cache_per_layer <= 0:
        return 0.0

    caches: dict[int, OrderedDict] = {}
    for layer in range(num_layers):
        od: OrderedDict = OrderedDict()
        if seed and layer in seed:
            for e in seed[layer][:cache_per_layer]:
                od[e] = True
        caches[layer] = od

    hits = misses = 0
    for step in trace:
        for layer, experts in step.items():
            cache = caches.setdefault(layer, OrderedDict())
            for e in experts:
                if e in cache:
                    cache.move_to_end(e)
                    hits += 1
                else:
                    misses += 1
                    cache[e] = True
                    if cache_per_layer > 0 and len(cache) > cache_per_layer:
                        cache.popitem(last=False)
    total = hits + misses
    return hits / total if total else 0.0


def coverage_curve(trace, *, num_layers: int,
                   cache_sizes: list[int],
                   seed: dict[int, list[int]] | None = None) -> dict[int, float]:
    """{cache_per_layer: hit_rate} for several cache sizes: the decision curve."""
    return {n: simulate_lru_hit_rate(trace, num_layers=num_layers,
                                     cache_per_layer=n, seed=seed)
            for n in cache_sizes}
