from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from moespresso.correctness.qwen4.kvarn_reference import reconstruct_qsa_kv_rows
from moespresso.runtime.qwen4.kvarn_cache import (
    QWEN38_KVARN_EXACT_SINK,
    QWEN38_KVARN_EXACT_SUFFIX,
    QWEN38_QSA_TILE_TOKENS,
    Qwen4KVarNQSAState,
    advance_qsa_kvarn_state,
    fork_qsa_kvarn_state,
    gather_qsa_kvarn_selected_rows,
    gather_qsa_kvarn_selected_rows_with_pending,
    prepare_qsa_kvarn_index,
    qsa_kvarn_partition,
    qsa_kvarn_safe_chunk_tokens,
    validate_qsa_kvarn_state,
)
from moespresso.runtime.qwen4.kvarn_encode import encode_qsa_kvarn_tile_mlx
from moespresso.runtime.qwen4.primitives import Qwen4RMSNorm
from moespresso.runtime.qwen4.qsa import qsa_attention_from_selected_rows
from moespresso.runtime.qwen4.qsa import (
    qsa_causal_prefix_layout,
    qsa_compress_index_keys,
    qsa_index_scores,
    qsa_selected_token_indices,
)


FIRST_SEAL = QWEN38_KVARN_EXACT_SINK + QWEN38_KVARN_EXACT_SUFFIX + QWEN38_QSA_TILE_TOKENS
SECOND_SEAL = FIRST_SEAL + QWEN38_QSA_TILE_TOKENS


def _history(token_count: int, seed: int = 601):
    rng = np.random.default_rng(seed)
    keys = mx.array(rng.normal(size=(1, 2, token_count, 256)).astype(np.float32)).astype(
        mx.bfloat16
    )
    values = mx.array(rng.normal(size=(1, 2, token_count, 256)).astype(np.float32)).astype(
        mx.bfloat16
    )
    index = mx.array(rng.normal(size=(1, token_count, 128)).astype(np.float32)).astype(mx.bfloat16)
    physical = np.arange(token_count, dtype=np.int32)
    positions = mx.array(
        np.stack([physical, physical + 10_000, physical + 20_000], axis=0)[:, None]
    )
    valid = mx.ones((1, token_count), dtype=mx.bool_)
    return keys, values, index, positions, valid


def _advance_chunks(
    history: tuple[mx.array, mx.array, mx.array, mx.array, mx.array],
    widths: list[int],
) -> Qwen4KVarNQSAState:
    keys, values, index, positions, valid = history
    key_norm = Qwen4RMSNorm(128)
    max_index_groups = max(1, keys.shape[2] // 4)
    cursor = 0
    state = None
    for width in widths:
        end = cursor + width
        _, index_update = prepare_qsa_kvarn_index(
            state,
            index[:, cursor:end],
            positions[:, :, cursor:end],
            key_norm,
            max_index_groups=max_index_groups,
            rotary_dim=64,
            rope_base=10_000_000.0,
            mrope_section=(11, 11, 10),
        )
        state = advance_qsa_kvarn_state(
            state,
            keys[..., cursor:end, :],
            values[..., cursor:end, :],
            valid[:, cursor:end],
            index_update=index_update,
        )
        mx.eval(
            state.packed_records,
            state.exact_sink_keys,
            state.exact_sink_values,
            state.exact_tail_keys,
            state.exact_tail_values,
            state.compressed_index_keys,
            state.compressed_index_positions,
            state.raw_index_tail,
            state.raw_index_tail_positions,
        )
        cursor = end
    assert cursor == keys.shape[2]
    assert state is not None
    return state


def _zero_tile_encoder(keys: mx.array, values: mx.array) -> mx.array:
    del keys, values
    return mx.zeros((2, 35_072), dtype=mx.uint8)


def _index_error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    nonzero = np.abs(expected) > 0
    mre = np.max(np.abs(delta[nonzero] / expected[nonzero])) if np.any(nonzero) else 0.0
    denominator = float(np.sqrt(np.mean(np.square(expected.astype(np.float64)))))
    relative_rms = float(np.sqrt(np.mean(np.square(delta)))) / denominator if denominator else 0.0
    return {
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "max_relative": float(mre),
        "relative_rms": relative_rms,
    }


def _advance_index_partition(
    raw_index_keys: mx.array,
    positions: mx.array,
    widths: list[int],
    key_norm: Qwen4RMSNorm,
) -> Qwen4KVarNQSAState:
    token_count = raw_index_keys.shape[1]
    max_index_groups = max(1, token_count // 4)
    keys = mx.zeros((1, 2, token_count, 256), dtype=mx.bfloat16)
    valid = mx.ones((1, token_count), dtype=mx.bool_)
    state = None
    cursor = 0
    for requested in widths:
        remaining = requested
        while remaining:
            width = min(remaining, qsa_kvarn_safe_chunk_tokens(state, remaining))
            end = cursor + width
            _, index_update = prepare_qsa_kvarn_index(
                state,
                raw_index_keys[:, cursor:end],
                positions[:, :, cursor:end],
                key_norm,
                max_index_groups=max_index_groups,
                rotary_dim=64,
                rope_base=10_000_000.0,
                mrope_section=(11, 11, 10),
            )
            state = advance_qsa_kvarn_state(
                state,
                keys[..., cursor:end, :],
                keys[..., cursor:end, :],
                valid[:, cursor:end],
                index_update=index_update,
                encode_tile=_zero_tile_encoder,
            )
            cursor = end
            remaining -= width
    assert cursor == token_count
    assert state is not None
    return state


@pytest.mark.parametrize(
    ("frontier", "sink_end", "body_end", "records", "tail_tokens"),
    [
        (0, 0, 0, 0, 0),
        (127, 127, 127, 0, 0),
        (128, 128, 128, 0, 0),
        (129, 128, 128, 0, 1),
        (1_023, 128, 128, 0, 895),
        (1_024, 128, 128, 0, 896),
        (1_151, 128, 128, 0, 1_023),
        (1_152, 128, 128, 0, 1_024),
        (2_048, 128, 128, 0, 1_920),
        (8_192, 128, 128, 0, 8_064),
        (8_319, 128, 128, 0, 8_191),
        (8_320, 128, 128, 0, 8_192),
        (8_447, 128, 128, 0, 8_319),
        (8_448, 128, 256, 1, 8_192),
        (8_575, 128, 256, 1, 8_319),
        (8_576, 128, 384, 2, 8_192),
        (9_216, 128, 1_024, 7, 8_192),
    ],
)
def test_partition_preserves_exact_boundaries_and_complete_tiles(
    frontier: int,
    sink_end: int,
    body_end: int,
    records: int,
    tail_tokens: int,
) -> None:
    assert qsa_kvarn_partition(frontier) == (sink_end, body_end, body_end)
    assert (body_end - sink_end) // 128 == records
    assert frontier - body_end == tail_tokens


def test_state_is_chunk_invariant_at_every_sealing_boundary() -> None:
    history = _history(SECOND_SEAL)
    direct = _advance_chunks(history, [FIRST_SEAL, QWEN38_QSA_TILE_TOKENS])
    coarse = _advance_chunks(
        history,
        [
            QWEN38_KVARN_EXACT_SINK,
            QWEN38_KVARN_EXACT_SUFFIX,
            QWEN38_QSA_TILE_TOKENS,
            QWEN38_QSA_TILE_TOKENS,
        ],
    )
    boundary_heavy = _advance_chunks(
        history,
        [127, 1, QWEN38_KVARN_EXACT_SUFFIX - 1, 1, 127, 1, 128],
    )

    for candidate in (coarse, boundary_heavy):
        assert candidate.frontier == direct.frontier == SECOND_SEAL
        assert candidate.body_frontier == direct.body_frontier == 384
        assert np.array_equal(
            np.asarray(candidate.packed_records),
            np.asarray(direct.packed_records),
        )
        for name in (
            "exact_sink_keys",
            "exact_sink_values",
            "exact_tail_keys",
            "exact_tail_values",
            "compressed_index_keys",
            "compressed_index_positions",
            "raw_index_tail",
            "raw_index_tail_positions",
        ):
            left = getattr(candidate, name)
            right = getattr(direct, name)
            if left.dtype == mx.bfloat16:
                left = left.astype(mx.float32)
                right = right.astype(mx.float32)
            assert np.array_equal(np.asarray(left), np.asarray(right))

    keys, values, *_ = history
    expected_first = encode_qsa_kvarn_tile_mlx(
        keys[0, :, 128:256].transpose(1, 0, 2),
        values[0, :, 128:256].transpose(1, 0, 2),
    )
    mx.eval(expected_first)
    assert np.array_equal(
        np.asarray(direct.packed_records[0]),
        np.asarray(expected_first),
    )
    assert np.array_equal(
        np.asarray(direct.exact_sink_keys.astype(mx.float32)),
        np.asarray(keys[..., :128, :].astype(mx.float32)),
    )
    assert np.array_equal(
        np.asarray(direct.exact_tail_keys.astype(mx.float32)),
        np.asarray(keys[..., 384:, :].astype(mx.float32)),
    )


def test_incremental_index_is_exact_at_4097_tokens_under_arbitrary_chunking() -> None:
    token_count = 4_097
    rng = np.random.default_rng(611)
    raw = mx.array(rng.normal(size=(1, token_count, 128)).astype(np.float32)).astype(mx.bfloat16)
    physical = np.arange(token_count, dtype=np.int32)
    positions = mx.array(
        np.stack(
            [physical * 3 + 7, physical * 5 + 11, physical * 7 + 13],
            axis=0,
        )[:, None]
    )
    key_norm = Qwen4RMSNorm(128)
    key_norm.weight = mx.array(rng.normal(0.0, 0.02, size=(128,)).astype(np.float32)).astype(
        mx.bfloat16
    )

    random_widths = []
    remaining = token_count
    while remaining:
        width = min(remaining, int(rng.integers(1, 212)))
        random_widths.append(width)
        remaining -= width
    coarse = _advance_index_partition(raw, positions, [token_count], key_norm)
    arbitrary = _advance_index_partition(raw, positions, random_widths, key_norm)

    query_rows = mx.array([[0, 3, 2_048, 4_096]], dtype=mx.int32)
    layout = qsa_causal_prefix_layout(
        mx.ones((1, token_count), dtype=mx.bool_),
        query_rows,
        compress_ratio=4,
    )
    reference = qsa_compress_index_keys(
        raw,
        positions,
        key_norm,
        layout,
        rotary_dim=64,
        base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    coarse_keys = coarse.compressed_index_keys[:, : coarse.index_group_count]
    arbitrary_keys = arbitrary.compressed_index_keys[:, : arbitrary.index_group_count]
    index_queries = mx.array(
        rng.normal(size=(1, query_rows.shape[1], 4, 128)).astype(np.float32)
    ).astype(mx.bfloat16)
    reference_scores = qsa_index_scores(index_queries, reference)
    coarse_scores = qsa_index_scores(index_queries, coarse_keys)
    arbitrary_scores = qsa_index_scores(index_queries, arbitrary_keys)
    reference_ids = qsa_selected_token_indices(reference_scores, layout)
    coarse_ids = qsa_selected_token_indices(coarse_scores, layout)
    arbitrary_ids = qsa_selected_token_indices(arbitrary_scores, layout)
    mx.eval(
        reference,
        coarse_keys,
        arbitrary_keys,
        reference_scores,
        coarse_scores,
        arbitrary_scores,
        reference_ids,
        coarse_ids,
        arbitrary_ids,
    )

    reference_np = np.asarray(reference.astype(mx.float32))
    coarse_np = np.asarray(coarse_keys.astype(mx.float32))
    arbitrary_np = np.asarray(arbitrary_keys.astype(mx.float32))
    reference_scores_np = np.asarray(reference_scores)
    coarse_scores_np = np.asarray(coarse_scores)
    arbitrary_scores_np = np.asarray(arbitrary_scores)
    assert _index_error_metrics(coarse_np, reference_np) == {
        "max_abs": 0.0,
        "max_relative": 0.0,
        "relative_rms": 0.0,
    }
    assert _index_error_metrics(arbitrary_np, reference_np) == {
        "max_abs": 0.0,
        "max_relative": 0.0,
        "relative_rms": 0.0,
    }
    assert _index_error_metrics(coarse_scores_np, reference_scores_np) == {
        "max_abs": 0.0,
        "max_relative": 0.0,
        "relative_rms": 0.0,
    }
    assert _index_error_metrics(arbitrary_scores_np, reference_scores_np) == {
        "max_abs": 0.0,
        "max_relative": 0.0,
        "relative_rms": 0.0,
    }
    assert np.array_equal(np.asarray(coarse_ids), np.asarray(reference_ids))
    assert np.array_equal(np.asarray(arbitrary_ids), np.asarray(reference_ids))
    assert coarse.index_stats == {
        "capacity_groups": 1_024,
        "sealed_groups": 1_024,
        "retained_raw_rows": 1,
    }
    assert arbitrary.index_stats == coarse.index_stats
    assert not hasattr(coarse, "raw_index_keys")
    assert not hasattr(coarse, "position_ids")
    assert np.array_equal(
        np.asarray(coarse.raw_index_tail.astype(mx.float32)),
        np.asarray(raw[:, -1:].astype(mx.float32)),
    )
    assert np.array_equal(
        np.asarray(coarse.raw_index_tail_positions), np.asarray(positions[:, :, -1:])
    )


def test_index_group_is_prepared_before_selection_and_published_with_append() -> None:
    history = _history(4, seed=612)
    keys, values, index, positions, valid = history
    norm = Qwen4RMSNorm(128)
    state = _advance_index_partition(index[:, :3], positions[:, :, :3], [3], norm)
    assert state.index_group_count == 0
    assert state.raw_index_tail.shape[1] == 3
    before = np.asarray(state.compressed_index_keys[:, :1].astype(mx.float32)).copy()

    prepared_keys, update = prepare_qsa_kvarn_index(
        state,
        index[:, 3:4],
        positions[:, :, 3:4],
        norm,
        max_index_groups=1,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    mx.eval(prepared_keys, update.new_group_keys, state.compressed_index_keys)
    assert prepared_keys.shape == (1, 1, 128)
    assert update.new_group_keys.shape == (1, 1, 128)
    assert state.index_group_count == 0
    assert np.array_equal(
        np.asarray(state.compressed_index_keys[:, :1].astype(mx.float32)),
        before,
    )

    published = advance_qsa_kvarn_state(
        state,
        keys[..., 3:4, :],
        values[..., 3:4, :],
        valid[:, 3:4],
        index_update=update,
        encode_tile=_zero_tile_encoder,
    )
    mx.eval(published.compressed_index_keys)
    assert published.index_group_count == 1
    assert published.raw_index_tail.shape[1] == 0
    assert np.array_equal(
        np.asarray(published.compressed_index_keys[:, :1].astype(mx.float32)),
        np.asarray(prepared_keys.astype(mx.float32)),
    )


def test_incremental_index_refuses_capacity_exhaustion_before_publication() -> None:
    keys, values, index, positions, valid = _history(8, seed=613)
    norm = Qwen4RMSNorm(128)
    _, first_update = prepare_qsa_kvarn_index(
        None,
        index[:, :4],
        positions[:, :, :4],
        norm,
        max_index_groups=1,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    state = advance_qsa_kvarn_state(
        None,
        keys[..., :4, :],
        values[..., :4, :],
        valid[:, :4],
        index_update=first_update,
        encode_tile=_zero_tile_encoder,
    )
    before = np.asarray(state.compressed_index_keys[:, :1].astype(mx.float32)).copy()
    with pytest.raises(ValueError, match="capacity 1 group.*exhausted"):
        prepare_qsa_kvarn_index(
            state,
            index[:, 4:8],
            positions[:, :, 4:8],
            norm,
            max_index_groups=1,
            rotary_dim=64,
            rope_base=10_000_000.0,
            mrope_section=(11, 11, 10),
        )
    assert state.index_group_count == 1
    assert np.array_equal(
        np.asarray(state.compressed_index_keys[:, :1].astype(mx.float32)),
        before,
    )


def test_update_cannot_cross_an_interior_tile_seal() -> None:
    history = _history(FIRST_SEAL + 1)
    keys, values, index, positions, valid = history
    norm = Qwen4RMSNorm(128)
    _, initial_index = prepare_qsa_kvarn_index(
        None,
        index[:, : FIRST_SEAL - 1],
        positions[:, :, : FIRST_SEAL - 1],
        norm,
        max_index_groups=(FIRST_SEAL + 1) // 4,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    state = advance_qsa_kvarn_state(
        None,
        keys[..., : FIRST_SEAL - 1, :],
        values[..., : FIRST_SEAL - 1, :],
        valid[:, : FIRST_SEAL - 1],
        index_update=initial_index,
    )
    mx.eval(state.packed_records, state.exact_tail_keys)
    assert qsa_kvarn_safe_chunk_tokens(state, 2) == 1
    _, crossing_index = prepare_qsa_kvarn_index(
        state,
        index[:, FIRST_SEAL - 1 : FIRST_SEAL + 1],
        positions[:, :, FIRST_SEAL - 1 : FIRST_SEAL + 1],
        norm,
        max_index_groups=(FIRST_SEAL + 1) // 4,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    with pytest.raises(ValueError, match="split after 1 token"):
        advance_qsa_kvarn_state(
            state,
            keys[..., FIRST_SEAL - 1 : FIRST_SEAL + 1, :],
            values[..., FIRST_SEAL - 1 : FIRST_SEAL + 1, :],
            valid[:, FIRST_SEAL - 1 : FIRST_SEAL + 1],
            index_update=crossing_index,
        )

    _, sealing_index = prepare_qsa_kvarn_index(
        state,
        index[:, FIRST_SEAL - 1 : FIRST_SEAL],
        positions[:, :, FIRST_SEAL - 1 : FIRST_SEAL],
        norm,
        max_index_groups=(FIRST_SEAL + 1) // 4,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    sealed = advance_qsa_kvarn_state(
        state,
        keys[..., FIRST_SEAL - 1 : FIRST_SEAL, :],
        values[..., FIRST_SEAL - 1 : FIRST_SEAL, :],
        valid[:, FIRST_SEAL - 1 : FIRST_SEAL],
        index_update=sealing_index,
    )
    assert sealed.frontier == FIRST_SEAL
    assert sealed.packed_records.shape[0] == 1
    assert state.frontier == FIRST_SEAL - 1
    assert state.packed_records.shape[0] == 0


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_composite_gather_normalizes_once_across_sink_body_and_tail() -> None:
    history = _history(SECOND_SEAL, seed=602)
    keys, values, *_ = history
    state = _advance_chunks(history, [FIRST_SEAL, QWEN38_QSA_TILE_TOKENS])
    selected = mx.array(
        [[[SECOND_SEAL - 1, 0, 129, 127, 128, 128, -1, 384, 255, 383, 99_999]]],
        dtype=mx.int32,
    )
    gathered_k, gathered_v, valid = gather_qsa_kvarn_selected_rows(state, selected)
    mx.eval(gathered_k, gathered_v, valid)

    canonical = np.array([0, 127, 128, 128, 129, 255, 383, 384, SECOND_SEAL - 1, -1, -1])
    expected_valid = np.array([True, True, True, False, True, True, True, True, True, False, False])
    expected_k = np.zeros((canonical.size, 2, 256), dtype=np.float32)
    expected_v = np.zeros_like(expected_k)
    source_k = np.asarray(keys.astype(mx.float32))[0].transpose(1, 0, 2)
    source_v = np.asarray(values.astype(mx.float32))[0].transpose(1, 0, 2)
    packed = np.asarray(state.packed_records)
    for lane, physical in enumerate(canonical):
        if not expected_valid[lane]:
            continue
        if physical < 128 or physical >= state.body_frontier:
            expected_k[lane] = source_k[physical]
            expected_v[lane] = source_v[physical]
        else:
            tile = (physical - 128) // 128
            offset = np.array([(physical - 128) % 128], dtype=np.int32)
            decoded_k, decoded_v = reconstruct_qsa_kv_rows(packed[tile], offset)
            expected_k[lane] = np.asarray(
                mx.array(decoded_k[0]).astype(mx.bfloat16).astype(mx.float32)
            )
            expected_v[lane] = np.asarray(
                mx.array(decoded_v[0]).astype(mx.bfloat16).astype(mx.float32)
            )

    got_k = np.asarray(gathered_k.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    got_v = np.asarray(gathered_v.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    assert np.array_equal(np.asarray(valid)[0, 0], expected_valid)
    assert np.max(np.abs(got_k - expected_k)) <= 0.0078125
    assert np.max(np.abs(got_v - expected_v)) <= 0.0078125

    rng = np.random.default_rng(603)
    queries = mx.array(rng.normal(size=(1, 1, 24, 256)).astype(np.float32)).astype(mx.bfloat16)
    reference_k = mx.array(expected_k).astype(mx.bfloat16).transpose(1, 0, 2)[None, None]
    reference_v = mx.array(expected_v).astype(mx.bfloat16).transpose(1, 0, 2)[None, None]
    reference_valid = mx.array(expected_valid[None, None])
    output = qsa_attention_from_selected_rows(queries, gathered_k, gathered_v, valid)
    reference = qsa_attention_from_selected_rows(
        queries,
        reference_k,
        reference_v,
        reference_valid,
    )
    mx.eval(output, reference)
    assert (
        np.max(
            np.abs(np.asarray(output.astype(mx.float32)) - np.asarray(reference.astype(mx.float32)))
        )
        <= 0.015625
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_pending_gather_preserves_lane_identity_across_prior_and_exact_rows() -> None:
    history = _history(1_411, seed=607)
    keys, values, index, positions, visible = history
    state = _advance_chunks(
        (
            keys[..., :1_408, :],
            values[..., :1_408, :],
            index[:, :1_408],
            positions[:, :, :1_408],
            visible[:, :1_408],
        ),
        [1_280, 128],
    )
    pending_keys = keys[..., 1_408:, :]
    pending_values = values[..., 1_408:, :]
    selected = mx.array(
        [[[1_410, 128, 1_408, 0, 1_409, 128, 383, -1, 9_999]]],
        dtype=mx.int32,
    )
    gathered_k, gathered_v, valid = gather_qsa_kvarn_selected_rows_with_pending(
        state,
        pending_keys,
        pending_values,
        selected,
    )
    mx.eval(gathered_k, gathered_v, valid)

    canonical = np.array([0, 128, 128, 383, 1_408, 1_409, 1_410, -1, -1])
    expected_valid = np.array([True, True, False, True, True, True, True, False, False])
    prior_selected = mx.array([[[0, 128, 383]]], dtype=mx.int32)
    prior_k, prior_v, _ = gather_qsa_kvarn_selected_rows(state, prior_selected)
    mx.eval(prior_k, prior_v)
    expected_k = np.zeros((canonical.size, 2, 256), dtype=np.float32)
    expected_v = np.zeros_like(expected_k)
    prior_expected_k = np.asarray(prior_k.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    prior_expected_v = np.asarray(prior_v.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    expected_k[[0, 1, 3]] = prior_expected_k
    expected_v[[0, 1, 3]] = prior_expected_v
    pending_k = np.asarray(pending_keys.astype(mx.float32))[0].transpose(1, 0, 2)
    pending_v = np.asarray(pending_values.astype(mx.float32))[0].transpose(1, 0, 2)
    expected_k[4:7] = pending_k
    expected_v[4:7] = pending_v

    got_k = np.asarray(gathered_k.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    got_v = np.asarray(gathered_v.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    assert np.array_equal(np.asarray(valid)[0, 0], expected_valid)
    assert np.array_equal(got_k, expected_k)
    assert np.array_equal(got_v, expected_v)


def test_pending_gather_without_prior_state_keeps_rows_exact() -> None:
    keys, values, *_ = _history(3, seed=608)
    selected = mx.array([[[2, 0, 1, 2, -1]]], dtype=mx.int32)
    gathered_k, gathered_v, valid = gather_qsa_kvarn_selected_rows_with_pending(
        None,
        keys,
        values,
        selected,
    )
    mx.eval(gathered_k, gathered_v, valid)

    expected_k = np.asarray(keys.astype(mx.float32))[0].transpose(1, 0, 2)
    expected_v = np.asarray(values.astype(mx.float32))[0].transpose(1, 0, 2)
    got_k = np.asarray(gathered_k.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    got_v = np.asarray(gathered_v.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    assert np.array_equal(np.asarray(valid)[0, 0], [True, True, True, False, False])
    assert np.array_equal(got_k[:3], expected_k)
    assert np.array_equal(got_v[:3], expected_v)
    assert np.count_nonzero(got_k[3:]) == 0
    assert np.count_nonzero(got_v[3:]) == 0


def test_fork_copies_mutable_histories_and_shares_immutable_records() -> None:
    state = _advance_chunks(_history(1_280, seed=604), [1_280])
    forked = fork_qsa_kvarn_state(state)
    mx.eval(forked.exact_sink_keys, state.exact_sink_keys)

    assert forked.packed_records is state.packed_records
    assert forked.exact_sink_keys is not state.exact_sink_keys
    assert forked.compressed_index_keys is not state.compressed_index_keys
    forked.exact_sink_keys[0, 0, 0, 0] = 99
    forked.compressed_index_keys[0, 0, 0] = 99
    mx.eval(
        forked.exact_sink_keys,
        forked.compressed_index_keys,
        state.exact_sink_keys,
        state.compressed_index_keys,
    )
    assert float(state.exact_sink_keys[0, 0, 0, 0].item()) != 99
    assert float(state.compressed_index_keys[0, 0, 0].item()) != 99
    assert state.frontier == 1_280


def test_state_validation_and_updates_fail_closed() -> None:
    history = _history(1, seed=605)
    state = _advance_chunks(history, [1])
    expected_nbytes = sum(
        int(getattr(state, name).nbytes)
        for name in (
            "packed_records",
            "exact_sink_keys",
            "exact_sink_values",
            "exact_tail_keys",
            "exact_tail_values",
            "compressed_index_keys",
            "compressed_index_positions",
            "raw_index_tail",
            "raw_index_tail_positions",
        )
    )
    assert state.nbytes == expected_nbytes

    with pytest.raises(ValueError, match="schema"):
        validate_qsa_kvarn_state(replace(state, schema="unsupported"))
    with pytest.raises(ValueError, match="body frontier"):
        validate_qsa_kvarn_state(replace(state, body_frontier=True))
    with pytest.raises(ValueError, match="shape"):
        gather_qsa_kvarn_selected_rows(
            state,
            mx.zeros((2, 1, 1), dtype=mx.int32),
        )
    with pytest.raises(ValueError, match="shape"):
        gather_qsa_kvarn_selected_rows(
            state,
            mx.array(0, dtype=mx.int32),
        )

    keys, values, index, positions, valid = _history(2, seed=606)
    norm = Qwen4RMSNorm(128)
    _, one_index = prepare_qsa_kvarn_index(
        None,
        index[:, :1],
        positions[:, :, :1],
        norm,
        max_index_groups=1,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    with pytest.raises(ValueError, match="unpad"):
        advance_qsa_kvarn_state(
            None,
            keys[..., :1, :],
            values[..., :1, :],
            mx.zeros((1, 1), dtype=mx.bool_),
            index_update=one_index,
        )
    with pytest.raises(ValueError, match="BF16"):
        advance_qsa_kvarn_state(
            None,
            keys[..., :1, :].astype(mx.float16),
            values[..., :1, :].astype(mx.float16),
            valid[:, :1],
            index_update=one_index,
        )
    with pytest.raises(ValueError, match="position dtype changed"):
        prepare_qsa_kvarn_index(
            state,
            index[:, 1:2],
            positions[:, :, 1:2].astype(mx.int64),
            norm,
            max_index_groups=1,
            rotary_dim=64,
            rope_base=10_000_000.0,
            mrope_section=(11, 11, 10),
        )


@pytest.mark.parametrize("frontier", [-1, True, 1.5])
def test_partition_rejects_invalid_frontiers(frontier) -> None:
    with pytest.raises(ValueError, match="frontier"):
        qsa_kvarn_partition(frontier)
