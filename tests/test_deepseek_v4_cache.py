from __future__ import annotations

import math

import pytest

from moespresso.runtime.deepseek_v4.cache import (
    DEEPSEEK_V4_CACHE_KIND,
    DeepseekV4Cache,
    DeepseekV4DecodeCompressionState,
    DeepseekV4PoolRow,
)


def test_deepseek_cache_tracks_raw_ring_and_compressed_pools():
    cache = DeepseekV4Cache().append_tokens(10)

    l0 = cache.layer(0)
    assert l0.compress_ratio == 0
    assert l0.raw_positions == tuple(range(10))
    assert l0.compressed_rows == ()
    assert l0.indexer_rows == ()

    l2 = cache.layer(2)
    assert l2.compress_ratio == 4
    assert l2.raw_positions == tuple(range(10))
    assert l2.compressed_rows == (
        DeepseekV4PoolRow(row_index=0, raw_start=0, raw_end=4),
        DeepseekV4PoolRow(row_index=1, raw_start=4, raw_end=8),
    )
    assert l2.indexer_rows == l2.compressed_rows
    assert l2.partial_positions == (8, 9)
    assert l2.indexer_partial_positions == (8, 9)

    l3 = cache.layer(3)
    assert l3.compress_ratio == 128
    assert l3.compressed_rows == ()
    assert l3.indexer_rows == ()
    assert l3.partial_positions == tuple(range(10))


def test_deepseek_cache_raw_ring_wraps_while_ratio_128_pool_advances():
    cache = DeepseekV4Cache().append_tokens(130)

    l0 = cache.layer(0)
    assert len(l0.raw_positions) == 128
    assert l0.raw_positions[0] == 2
    assert l0.raw_positions[-1] == 129

    hca = cache.layer(3)
    assert hca.compressed_rows == (DeepseekV4PoolRow(row_index=0, raw_start=0, raw_end=128),)
    assert hca.partial_positions == (128, 129)
    assert hca.indexer_rows == ()


def test_deepseek_cache_trim_drops_stale_compressed_indexer_rows_and_rebuilds_partials():
    cache = DeepseekV4Cache().append_tokens(16)
    assert [r.raw_end for r in cache.layer(2).compressed_rows] == [4, 8, 12, 16]

    cache.trim_to_length(10)

    l2 = cache.layer(2)
    assert cache.token_count == 10
    assert [r.raw_end for r in l2.compressed_rows] == [4, 8]
    assert l2.indexer_rows == l2.compressed_rows
    assert l2.partial_positions == (8, 9)
    assert l2.indexer_partial_positions == (8, 9)
    assert all(r.raw_end <= cache.token_count for r in l2.compressed_rows)


def test_deepseek_cache_fork_is_independent_and_payload_roundtrips():
    cache = DeepseekV4Cache().append_tokens(9)
    fork = cache.fork()
    fork.append_tokens(4)

    assert cache.token_count == 9
    assert fork.token_count == 13
    assert cache.layer(2).partial_positions == (8,)
    assert fork.layer(2).partial_positions == (12,)

    payload = fork.to_payload()
    assert payload["kind"] == DEEPSEEK_V4_CACHE_KIND
    restored = DeepseekV4Cache.from_payload(payload)
    assert restored.to_payload() == payload


def test_deepseek_cache_rejects_invalid_trim_or_payload_kind():
    cache = DeepseekV4Cache().append_tokens(2)
    with pytest.raises(ValueError, match="prefix_len"):
        cache.trim_to_length(3)

    payload = cache.to_payload()
    payload["kind"] = "mlx_prompt_cache"
    with pytest.raises(ValueError, match="DeepSeek V4"):
        DeepseekV4Cache.from_payload(payload)


def test_decode_compression_overlap_carries_previous_window_before_shift():
    state = DeepseekV4DecodeCompressionState(compress_ratio=4)
    assert state.position_slots == (None,) * 8
    assert state.score_state == (-math.inf,) * 8

    steps = [state.append_token(pos) for pos in range(4)]

    assert [step.write_slot for step in steps] == [4, 5, 6, 7]
    assert [step.ape_slot for step in steps] == [0, 1, 2, 3]
    assert [step.should_compress for step in steps] == [False, False, False, True]
    assert steps[-1].compression_positions == (None, None, None, None, 0, 1, 2, 3)
    assert state.position_slots == (0, 1, 2, 3, None, None, None, None)
    assert state.score_state == (
        0.0,
        0.0,
        0.0,
        0.0,
        -math.inf,
        -math.inf,
        -math.inf,
        -math.inf,
    )

    steps = [state.append_token(pos) for pos in range(4, 8)]

    assert [step.write_slot for step in steps] == [4, 5, 6, 7]
    assert steps[-1].should_compress is True
    assert steps[-1].compression_positions == tuple(range(8))
    assert state.position_slots == (4, 5, 6, 7, None, None, None, None)


def test_decode_compression_without_overlap_uses_one_window_and_clears():
    state = DeepseekV4DecodeCompressionState(compress_ratio=128)

    for pos in range(127):
        step = state.append_token(pos)
        assert step.write_slot == pos
        assert step.ape_slot == pos
        assert step.should_compress is False

    step = state.append_token(127)

    assert step.write_slot == 127
    assert step.ape_slot == 127
    assert step.should_compress is True
    assert step.compression_positions == tuple(range(128))
    assert state.position_slots == (None,) * 128
    assert state.score_state == (-math.inf,) * 128

    step = state.append_token(128)
    assert step.write_slot == 0
    assert step.ape_slot == 0
    assert state.position_slots[0] == 128


def test_decode_compression_state_rejects_non_incremental_positions():
    state = DeepseekV4DecodeCompressionState(compress_ratio=4)
    state.append_token(0)
    with pytest.raises(ValueError, match="advance one token"):
        state.append_token(2)
