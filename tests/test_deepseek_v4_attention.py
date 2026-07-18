from __future__ import annotations

import math

import numpy as np

from moespresso.runtime.deepseek_v4.attention import (
    DS4_COMPRESS_ROPE_BASE,
    DS4_ROPE_DIM,
    DS4_SWA_ROPE_BASE,
    apply_rope_tail,
    causal_indexer_scores,
    causal_indexer_topk,
    compressed_rope_positions,
    compressed_visibility,
    deepseek_v4_rope_policy,
    indexer_weight_scale,
    indexer_weighted_scores,
    rope_angles,
    rope_inverse_frequencies,
    sink_softmax_attention,
    visible_compressed_count,
)


def test_rope_policy_uses_compressed_yarn_only_for_compressed_layers():
    l0 = deepseek_v4_rope_policy(0)
    l1 = deepseek_v4_rope_policy(1)
    l2 = deepseek_v4_rope_policy(2)
    l42 = deepseek_v4_rope_policy(42)
    mtp = deepseek_v4_rope_policy(43)

    assert l0.compress_ratio == 0
    assert l0.rope_base == DS4_SWA_ROPE_BASE
    assert not l0.uses_yarn
    assert l1.rope_base == DS4_SWA_ROPE_BASE
    assert not l1.uses_yarn

    assert l2.compress_ratio == 4
    assert l2.rope_base == DS4_COMPRESS_ROPE_BASE
    assert l2.uses_yarn
    assert l2.yarn == {
        "factor": 16.0,
        "original_seq_len": 65536.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }

    assert l42.compress_ratio == 4
    assert l42.uses_yarn
    assert mtp.compress_ratio == 0
    assert mtp.rope_base == DS4_SWA_ROPE_BASE
    assert not mtp.uses_yarn


def test_yarn_inverse_frequency_values_are_pinned_at_high_positions():
    policy = deepseek_v4_rope_policy(2)
    freqs = rope_inverse_frequencies(DS4_ROPE_DIM, policy.rope_base, policy.yarn)

    np.testing.assert_allclose(
        freqs[[0, 8, 16, 24, 31]],
        np.array(
            [
                1.0,
                5.0e-2,
                2.265625e-3,
                1.953125e-5,
                5.68052903691e-7,
            ],
            dtype=np.float64,
        ),
        rtol=0,
        atol=1e-14,
    )

    positions = np.array(
        [65535, 65536, 65537, 131071, 131072, 1048572, 1048573, 1048574, 1048575],
        dtype=np.int64,
    )
    angles = rope_angles(positions, dim=DS4_ROPE_DIM, base=policy.rope_base, yarn=policy.yarn)
    np.testing.assert_allclose(
        angles[:, [0, 8, 16, 24, 31]],
        np.array(
            [
                [65535.0, 3276.75, 148.477734375, 1.27998046875, 0.0372273470],
                [65536.0, 3276.8, 148.48, 1.28, 0.0372279151],
                [65537.0, 3276.85, 148.482265625, 1.28001953125, 0.0372284831],
                [131071.0, 6553.55, 296.957734375, 2.55998046875, 0.0744552621],
                [131072.0, 6553.6, 296.96, 2.56, 0.0744558302],
                [1048572.0, 52428.6, 2375.6709375, 20.479921875, 0.5956443693],
                [1048573.0, 52428.65, 2375.673203125, 20.4799414062, 0.5956449374],
                [1048574.0, 52428.7, 2375.67546875, 20.4799609375, 0.5956455054],
                [1048575.0, 52428.75, 2375.677734375, 20.47998046875, 0.595646073],
            ],
            dtype=np.float64,
        ),
        rtol=0,
        atol=1e-9,
    )

    swa = rope_inverse_frequencies(DS4_ROPE_DIM, DS4_SWA_ROPE_BASE, None)
    assert math.isclose(swa[16], 1.0e-2)
    assert not np.allclose(freqs, swa)
    swa_angles = rope_angles(
        [131071, 131072, 1048575],
        dim=DS4_ROPE_DIM,
        base=DS4_SWA_ROPE_BASE,
        yarn=None,
    )
    np.testing.assert_allclose(
        swa_angles[:, [0, 8, 16, 24, 31]],
        np.array(
            [
                [131071.0, 13107.1, 1310.71, 131.071, 17.4785987635],
                [131072.0, 13107.2, 1310.72, 131.072, 17.4787321157],
                [1048575.0, 104857.5, 10485.75, 1048.575, 139.829723573],
            ],
            dtype=np.float64,
        ),
        rtol=0,
        atol=1e-9,
    )


def test_partial_rope_rotates_only_tail_and_inverse_cancels():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 512)).astype(np.float32)
    positions = np.array([5, 65536], dtype=np.int64)
    policy = deepseek_v4_rope_policy(2)

    rotated = apply_rope_tail(x, positions, base=policy.rope_base, yarn=policy.yarn)

    np.testing.assert_array_equal(rotated[:, :448], x[:, :448])
    assert not np.allclose(rotated[:, -64:], x[:, -64:])

    restored = apply_rope_tail(
        rotated,
        positions,
        base=policy.rope_base,
        yarn=policy.yarn,
        inverse=True,
    )
    np.testing.assert_allclose(restored, x, rtol=2e-6, atol=2e-6)


def test_sink_softmax_attention_adds_virtual_zero_value_denominator():
    scores = np.array([[0.25]], dtype=np.float64)
    values = np.array([[[8.0, -4.0]]], dtype=np.float64)
    sinks = np.array([1.25], dtype=np.float64)

    out = sink_softmax_attention(scores, values, sinks)
    m = 1.25
    den = math.exp(0.25 - m) + math.exp(1.25 - m)
    expected = values[:, 0, :] * (math.exp(0.25 - m) / den)
    np.testing.assert_allclose(out, expected)

    no_sink = sink_softmax_attention(scores, values, np.array([-np.inf]))
    zero_sink_workaround = sink_softmax_attention(scores, values, np.array([0.0]))
    assert np.allclose(no_sink, values[:, 0, :])
    assert not np.allclose(zero_sink_workaround, out)


def test_compressed_visibility_frontier_is_true_for_valid_at_boundaries():
    pool = np.arange(34, dtype=np.int64)
    vis4 = compressed_visibility([2, 3, 4, 127, 128], pool, ratio=4)

    assert vis4.dtype == np.bool_
    assert not vis4[0, 0]
    assert vis4[1, 0]
    assert vis4[2, 0]
    assert vis4[3, 31]
    assert not vis4[3, 32]
    assert vis4[4, 31]
    assert not vis4[4, 32]

    vis128 = compressed_visibility([126, 127, 128], np.arange(2), ratio=128)
    assert not vis128[0, 0]
    assert vis128[1, 0]
    assert vis128[2, 0]
    assert not vis128[2, 1]

    np.testing.assert_array_equal(
        visible_compressed_count([3, 4, 65535, 65536, 1048575], ratio=4),
        np.array([1, 1, 16384, 16384, 262144], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        compressed_rope_positions([0, 1, 8191], ratio=128),
        np.array([0, 128, 1048448], dtype=np.int64),
    )


def test_indexer_scale_and_topk_preserve_causality():
    scale = indexer_weight_scale(index_head_dim=128, n_heads=64)
    assert math.isclose(scale, (128 ** -0.5) * (64 ** -0.5))
    assert not math.isclose(scale, 64 ** -0.5)

    raw = np.ones((1, 64, 1), dtype=np.float64)
    weights = np.ones((1, 64), dtype=np.float64)
    weighted = indexer_weighted_scores(raw, weights, index_head_dim=128, n_heads=64)
    np.testing.assert_allclose(weighted, np.array([[64.0 * scale]]))

    pool = np.array([0, 1, 2, 3], dtype=np.int64)
    raw_scores = np.array(
        [
            [[1.0, 20.0, 1000.0, 1000.0], [2.0, 10.0, 1000.0, 1000.0]],
            [[1.0, 9.0, 1000.0, 1000.0], [2.0, 8.0, 1000.0, 1000.0]],
        ],
        dtype=np.float64,
    )
    head_weights = np.ones((2, 2), dtype=np.float64)
    masked = causal_indexer_scores(
        raw_scores,
        head_weights,
        query_positions=np.array([3, 7], dtype=np.int64),
        pool_indices=pool,
        ratio=4,
        index_head_dim=128,
        n_heads=2,
    )
    assert np.isfinite(masked[0, 0])
    assert np.isneginf(masked[0, 1])
    assert np.isfinite(masked[1, 1])
    assert np.isneginf(masked[1, 2])

    topk = causal_indexer_topk(
        raw_scores,
        head_weights,
        query_positions=np.array([3, 7], dtype=np.int64),
        pool_indices=pool,
        ratio=4,
        topk=3,
        index_head_dim=128,
        n_heads=2,
    )
    np.testing.assert_array_equal(
        topk,
        np.array(
            [
                [0, -1, -1],
                [1, 0, -1],
            ],
            dtype=np.int64,
        ),
    )
