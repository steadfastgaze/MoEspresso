"""Router-locality analysis for SSD streaming.

Pure + model-free: given a trace of per-(layer) selected expert ids per decode
step, compute the per-layer activation histogram, a hotlist, and the simulated
LRU hit rate for various cache sizes. The hit rate is the number that decides
whether expert streaming is usable: high locality -> cheap misses; uniform
routing -> slow.

These functions take a trace (list of {layer: [expert_ids]}) so they are tested
without running a model; the profiler probe feeds them a real trace.
"""

from __future__ import annotations

from moespresso.runtime.expert_locality import (
    LocalityStats,
    hotlist_from_counts,
    simulate_lru_hit_rate,
    summarize_trace,
)


def _trace(steps):
    """steps: list of dict(layer -> list[expert_id])."""
    return steps


def test_summarize_counts_activations_per_layer_expert():
    trace = _trace([
        {0: [1, 2], 1: [5, 6]},
        {0: [1, 3], 1: [5, 7]},
        {0: [1, 2], 1: [5, 6]},
    ])
    stats = summarize_trace(trace, num_layers=2, num_experts=8, top_k=2)
    assert isinstance(stats, LocalityStats)
    # layer 0: expert 1 fired 3x, expert 2 twice, expert 3 once
    assert stats.counts[0][1] == 3
    assert stats.counts[0][2] == 2
    assert stats.counts[0][3] == 1
    # layer 1: expert 5 fired 3x
    assert stats.counts[1][5] == 3
    assert stats.total_activations == 3 * 2 * 2  # 3 steps * 2 layers * 2 experts


def test_hotlist_picks_most_frequent_per_layer():
    counts = {0: {1: 10, 2: 5, 3: 1}, 1: {5: 8, 6: 8, 7: 2}}
    hot = hotlist_from_counts(counts, per_layer=2)
    assert hot[0] == [1, 2]               # top-2 by frequency, frequency order
    assert set(hot[1]) == {5, 6}          # ties allowed, top-2


def test_lru_hit_rate_perfect_locality_is_high():
    # Same experts every step -> after warmup the cache hits ~always.
    trace = _trace([{0: [1, 2, 3, 4]}] * 20)
    # cache big enough to hold all 4 -> near-100% after first step
    hr = simulate_lru_hit_rate(trace, num_layers=1, cache_per_layer=4)
    assert hr > 0.9


def test_lru_hit_rate_uniform_is_low_when_cache_small():
    # Cycle through 16 distinct experts, cache only holds 2 -> mostly misses.
    steps = [{0: [(i * 2) % 16, (i * 2 + 1) % 16]} for i in range(40)]
    hr = simulate_lru_hit_rate(_trace(steps), num_layers=1, cache_per_layer=2)
    assert hr < 0.4


def test_lru_hit_rate_zero_capacity_never_hits():
    trace = _trace([{0: [1, 2]}, {0: [1, 2]}, {0: [1, 2]}])
    hr = simulate_lru_hit_rate(trace, num_layers=1, cache_per_layer=0)
    assert hr == 0.0


def test_lru_hit_rate_monotonic_in_cache_size():
    steps = [{0: sorted({(i * 3) % 32, (i * 5) % 32, (i * 7) % 32, i % 32})}
             for i in range(60)]
    small = simulate_lru_hit_rate(_trace(steps), num_layers=1, cache_per_layer=4)
    big = simulate_lru_hit_rate(_trace(steps), num_layers=1, cache_per_layer=24)
    assert big >= small


def test_hit_rate_is_per_layer_independent():
    # Layer 0 perfectly local, layer 1 thrashing; global rate is the blend.
    steps = []
    for i in range(20):
        steps.append({0: [1, 2], 1: [(i * 2) % 30, (i * 2 + 1) % 30]})
    hr = simulate_lru_hit_rate(_trace(steps), num_layers=2, cache_per_layer=2)
    # layer 0 ~all hits, layer 1 ~all misses -> blend roughly in the middle
    assert 0.3 < hr < 0.7
