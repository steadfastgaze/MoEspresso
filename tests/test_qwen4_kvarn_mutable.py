from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from moespresso.runtime.qwen4.kvarn_cache import (
    QWEN38_KVARN_EXACT_SINK,
    QWEN38_KVARN_EXACT_SUFFIX,
    QWEN38_QSA_TILE_TOKENS,
    Qwen4KVarNQSAState,
    advance_qsa_kvarn_state,
    gather_qsa_kvarn_selected_rows_with_pending,
    prepare_qsa_kvarn_index,
    qsa_kvarn_safe_chunk_tokens,
)
from moespresso.runtime.qwen4.kvarn_mutable import (
    QWEN38_KVARN_EXACT_TAIL_CAPACITY,
    Qwen4MutableKVarNStorage,
    _QWEN4_SERIAL_IRREVOCABLE_APPEND_CAPABILITY,
)
from moespresso.runtime.qwen4.primitives import Qwen4RMSNorm
from moespresso.runtime.qwen4.qsa import (
    _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    qsa_attention_from_selected_rows,
    qsa_gather_selected_rows,
    qsa_normalize_selected_rows,
)


FIRST_SEAL = QWEN38_KVARN_EXACT_SINK + QWEN38_KVARN_EXACT_SUFFIX + QWEN38_QSA_TILE_TOKENS
SECOND_SEAL = FIRST_SEAL + QWEN38_QSA_TILE_TOKENS


def _history(token_count: int, seed: int = 811):
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
        np.stack(
            [physical * 3 + 1, physical * 5 + 2, physical * 7 + 3],
            axis=0,
        )[:, None]
    )
    valid = mx.ones((1, token_count), dtype=mx.bool_)
    return keys, values, index, positions, valid


def _zero_tile_encoder(keys: mx.array, values: mx.array) -> mx.array:
    del keys, values
    return mx.zeros((2, 35_072), dtype=mx.uint8)


def _append_mutable(
    storage: Qwen4MutableKVarNStorage,
    history,
    start: int,
    end: int,
    norm: Qwen4RMSNorm,
):
    keys, values, index, positions, valid = history
    _, update = storage.prepare_index(
        index[:, start:end],
        positions[:, :, start:end],
        norm,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    return storage.append(
        keys[..., start:end, :],
        values[..., start:end, :],
        valid[:, start:end],
        index_update=update,
    )


def _storage_bytes(storage: Qwen4MutableKVarNStorage) -> tuple[bytes, ...]:
    arrays = tuple(mx.contiguous(array) for array in storage._physical_arrays())
    mx.eval(*arrays)
    return tuple(b"" if array.size == 0 else bytes(memoryview(array).cast("B")) for array in arrays)


def _functional_state(
    history,
    requested_widths: list[int],
    norm: Qwen4RMSNorm,
) -> Qwen4KVarNQSAState:
    keys, values, index, positions, valid = history
    max_groups = max(1, keys.shape[2] // 4)
    state = None
    cursor = 0
    for requested in requested_widths:
        remaining = requested
        while remaining:
            width = min(remaining, qsa_kvarn_safe_chunk_tokens(state, remaining))
            end = cursor + width
            _, update = prepare_qsa_kvarn_index(
                state,
                index[:, cursor:end],
                positions[:, :, cursor:end],
                norm,
                max_index_groups=max_groups,
                rotary_dim=64,
                rope_base=10_000_000.0,
                mrope_section=(11, 11, 10),
            )
            state = advance_qsa_kvarn_state(
                state,
                keys[..., cursor:end, :],
                values[..., cursor:end, :],
                valid[:, cursor:end],
                index_update=update,
            )
            cursor = end
            remaining -= width
    assert cursor == sum(requested_widths)
    assert state is not None
    return state


def test_mutable_storage_preallocates_fixed_regions_and_reports_bytes() -> None:
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=4_097,
        encode_tile=_zero_tile_encoder,
    )
    array_ids = tuple(id(array) for array in storage._physical_arrays())
    allocated = sum(int(array.nbytes) for array in storage._physical_arrays())
    assert storage.exact_sink_keys.shape == (1, 2, 128, 256)
    assert storage.exact_tail_keys.shape == (
        1,
        2,
        QWEN38_KVARN_EXACT_TAIL_CAPACITY,
        256,
    )
    assert storage.packed_records.shape == (0, 2, 35_072)
    assert storage.compressed_index_keys.shape == (1, 1_024, 128)
    assert storage.allocated_nbytes == allocated
    assert storage.physical_nbytes == allocated
    assert storage.logical_nbytes == 0

    history = _history(17)
    _append_mutable(storage, history, 0, 17, Qwen4RMSNorm(128))
    assert tuple(id(array) for array in storage._physical_arrays()) == array_ids
    assert storage.logical_nbytes > 0
    assert storage.physical_nbytes > storage.allocated_nbytes
    assert storage.stats["indexed_kv_write_calls"] == 1
    assert storage.stats["sink_rows_written"] == 17
    assert storage.stats["tail_rows_written"] == 0
    committed = storage.commit(storage.checkpoint())
    assert committed.frontier == 17
    assert storage.journal_nbytes == 0
    assert storage.physical_nbytes == storage.allocated_nbytes


def test_pending_rows_remain_exact_until_the_sealing_append_is_published() -> None:
    history = _history(FIRST_SEAL, seed=812)
    keys, values, *_ = history
    norm = Qwen4RMSNorm(128)
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=FIRST_SEAL,
        encode_tile=_zero_tile_encoder,
    )
    _append_mutable(storage, history, 0, FIRST_SEAL - 1, norm)
    storage.commit()
    selected = mx.array(
        [[[FIRST_SEAL - 1, 128, FIRST_SEAL - 1, -1, 99_999]]],
        dtype=mx.int32,
    )
    gathered_k, gathered_v, valid = storage.gather_selected_rows_with_pending(
        keys[..., FIRST_SEAL - 1 : FIRST_SEAL, :],
        values[..., FIRST_SEAL - 1 : FIRST_SEAL, :],
        selected,
    )
    mx.eval(gathered_k, gathered_v, valid)
    assert storage.record_count == 0
    got_k = np.asarray(gathered_k.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    got_v = np.asarray(gathered_v.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    pending_k = np.asarray(keys[..., FIRST_SEAL - 1 : FIRST_SEAL, :].astype(mx.float32))[
        0
    ].transpose(1, 0, 2)
    pending_v = np.asarray(values[..., FIRST_SEAL - 1 : FIRST_SEAL, :].astype(mx.float32))[
        0
    ].transpose(1, 0, 2)
    assert np.array_equal(np.asarray(valid)[0, 0], [True, True, False, False, False])
    assert np.array_equal(got_k[1], pending_k[0])
    assert np.array_equal(got_v[1], pending_v[0])

    _append_mutable(storage, history, FIRST_SEAL - 1, FIRST_SEAL, norm)
    assert storage.record_count == 1
    assert storage.body_frontier == 256
    assert storage.tail_start == 256
    assert storage.tail_count == QWEN38_KVARN_EXACT_SUFFIX
    assert storage.stats["tile_records_sealed"] == 1


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_fused_mutable_decode_gather_is_exact_and_guarded() -> None:
    frontier = SECOND_SEAL
    history = _history(frontier + 1, seed=813)
    keys, values, *_ = history
    storage = Qwen4MutableKVarNStorage(max_context_tokens=frontier + 1)
    _append_mutable(storage, history, 0, frontier, Qwen4RMSNorm(128))
    storage.commit()
    selected = mx.array(
        [
            [
                [
                    0,
                    127,
                    128,
                    255,
                    256,
                    FIRST_SEAL - 1,
                    FIRST_SEAL,
                    frontier - 1,
                    frontier,
                    -1,
                    99_999,
                ]
            ]
        ],
        dtype=mx.int32,
    )
    pending_keys = keys[..., frontier : frontier + 1, :]
    pending_values = values[..., frontier : frontier + 1, :]

    normalized, valid = qsa_normalize_selected_rows(selected, frontier + 1)
    incumbent_keys, incumbent_values = storage._gather_normalized_rows_with_pending(
        pending_keys,
        pending_values,
        normalized,
        valid,
    )
    incumbent = incumbent_keys, incumbent_values, valid
    mx.eval(*incumbent)
    assert storage.stats["fused_mutable_gather_calls"] == 0

    candidate = storage._gather_selected_rows_with_pending_deferred_finite(
        pending_keys,
        pending_values,
        selected,
    )
    mx.eval(*candidate)
    assert storage.stats["fused_mutable_gather_calls"] == 1
    assert storage.stats["fused_mutable_gather_lanes"] == selected.size
    for expected, actual in zip(incumbent, candidate, strict=True):
        if expected.dtype == mx.bfloat16:
            expected = expected.astype(mx.float32)
            actual = actual.astype(mx.float32)
        assert np.array_equal(np.asarray(expected), np.asarray(actual))


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    ("frontier", "pending_count", "selected_rows"),
    [
        (
            1_279,
            2,
            [0, 127, 128, 1_278, 1_279, 1_280, 1_280, -1, 9_999],
        ),
        (
            1_408,
            3,
            [0, 127, 128, 383, 384, 1_407, 1_408, 1_410, 128, -1, 9_999],
        ),
        (
            4_096,
            129,
            [
                0,
                127,
                128,
                255,
                3_071,
                3_072,
                3_199,
                3_200,
                4_095,
                4_096,
                4_223,
                4_224,
                4_224,
                -1,
                9_999,
            ],
        ),
    ],
    ids=("first-rollover", "packed-tile", "wrapped-ring"),
)
def test_materialized_prefill_rows_match_selected_gather_and_attention_exactly(
    frontier: int,
    pending_count: int,
    selected_rows: list[int],
) -> None:
    history = _history(frontier + pending_count, seed=821 + frontier)
    keys, values, *_ = history
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=frontier + pending_count,
        encode_tile=_zero_tile_encoder,
    )
    _append_mutable(storage, history, 0, frontier, Qwen4RMSNorm(128))
    storage.commit()
    before = storage.checkpoint()
    selected = mx.array(
        [[selected_rows, list(reversed(selected_rows))]],
        dtype=mx.int32,
    )
    pending_keys = keys[..., frontier : frontier + pending_count, :]
    pending_values = values[..., frontier : frontier + pending_count, :]

    incumbent_k, incumbent_v, incumbent_valid = storage.gather_selected_rows_with_pending(
        pending_keys,
        pending_values,
        selected,
    )
    materialized_k, materialized_v = storage.materialize_rows_with_pending(
        pending_keys,
        pending_values,
    )
    prepared_k, prepared_v, prepared_valid = qsa_gather_selected_rows(
        materialized_k,
        materialized_v,
        selected,
    )
    prepared_mask = prepared_valid[:, :, None, :, None]
    prepared_k = mx.where(prepared_mask, prepared_k, 0)
    prepared_v = mx.where(prepared_mask, prepared_v, 0)
    rng = np.random.default_rng(901 + frontier)
    queries = mx.array(rng.normal(size=(1, 2, 24, 256)).astype(np.float32)).astype(mx.bfloat16)
    incumbent_attention = qsa_attention_from_selected_rows(
        queries,
        incumbent_k,
        incumbent_v,
        incumbent_valid,
    )
    prepared_attention = qsa_attention_from_selected_rows(
        queries,
        prepared_k,
        prepared_v,
        prepared_valid,
    )
    mx.eval(
        incumbent_k,
        incumbent_v,
        incumbent_valid,
        materialized_k,
        materialized_v,
        prepared_k,
        prepared_v,
        prepared_valid,
        incumbent_attention,
        prepared_attention,
    )

    assert storage.checkpoint() == before
    assert materialized_k.shape == (1, 2, frontier + pending_count, 256)
    assert materialized_v.shape == materialized_k.shape
    assert np.array_equal(np.asarray(prepared_valid), np.asarray(incumbent_valid))
    assert np.array_equal(
        np.asarray(prepared_k.astype(mx.float32)),
        np.asarray(incumbent_k.astype(mx.float32)),
    )
    assert np.array_equal(
        np.asarray(prepared_v.astype(mx.float32)),
        np.asarray(incumbent_v.astype(mx.float32)),
    )
    assert np.array_equal(
        np.asarray(prepared_attention.astype(mx.float32)),
        np.asarray(incumbent_attention.astype(mx.float32)),
    )
    assert storage.stats["prefill_materialize_calls"] == 1
    assert storage.stats["prefill_materialize_lanes"] == frontier + pending_count
    assert storage.stats["gather_calls"] == 1
    assert storage.stats["pending_gather_calls"] == 1
    assert storage.stats["gather_lanes"] == selected.size


def test_fork_discard_restores_wrapped_tail_without_whole_state_copy() -> None:
    final_frontier = SECOND_SEAL + 92
    original = _history(final_frontier, seed=813)
    alternative = _history(final_frontier, seed=814)
    norm = Qwen4RMSNorm(128)
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=final_frontier,
        encode_tile=_zero_tile_encoder,
    )
    _append_mutable(storage, original, 0, FIRST_SEAL - 1, norm)
    storage.commit()
    branch = storage.fork()
    selected = mx.array(
        [[[128, 255, FIRST_SEAL - 2, 0, 128, -1]]],
        dtype=mx.int32,
    )
    before_k, before_v, before_valid = storage.gather_selected_rows(selected)
    mx.eval(before_k, before_v, before_valid)
    before_k_np = np.asarray(before_k.astype(mx.float32)).copy()
    before_v_np = np.asarray(before_v.astype(mx.float32)).copy()

    _append_mutable(storage, original, FIRST_SEAL - 1, SECOND_SEAL, norm)
    discarded_future = storage.checkpoint()
    assert storage.record_count == 2
    assert storage.tail_count == QWEN38_KVARN_EXACT_SUFFIX
    with pytest.raises(ValueError, match="mutation cursor"):
        storage.restore(
            replace(
                discarded_future,
                mutation_cursor=0,
                mutation_anchor=0,
            )
        )
    assert storage.frontier == SECOND_SEAL
    storage.discard(branch)
    assert storage.frontier == FIRST_SEAL - 1
    assert storage.record_count == 0
    after_k, after_v, after_valid = storage.gather_selected_rows(selected)
    mx.eval(after_k, after_v, after_valid)
    assert np.array_equal(np.asarray(after_valid), np.asarray(before_valid))
    assert np.array_equal(np.asarray(after_k.astype(mx.float32)), before_k_np)
    assert np.array_equal(np.asarray(after_v.astype(mx.float32)), before_v_np)

    _append_mutable(storage, alternative, FIRST_SEAL - 1, FIRST_SEAL + 20, norm)
    with pytest.raises(ValueError, match="discarded branch"):
        storage.restore(discarded_future)
    active = storage.commit(storage.checkpoint())
    with pytest.raises(ValueError, match="expired lineage"):
        storage.restore(branch)
    assert active.frontier == FIRST_SEAL + 20
    assert storage.journal_nbytes == 0


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_cache_matches_functional_oracle_through_4097_tokens() -> None:
    history = _history(4_097, seed=815)
    keys, values, index, positions, valid = history
    norm = Qwen4RMSNorm(128)
    rng = np.random.default_rng(816)
    norm.weight = mx.array(rng.normal(0.0, 0.02, size=(128,)).astype(np.float32)).astype(
        mx.bfloat16
    )

    functional = _functional_state(history, [17, 2_983, 1_096], norm)
    mutable = Qwen4MutableKVarNStorage(max_context_tokens=4_097)
    cursor = 0
    for width in (17, 2_983, 1_096):
        _append_mutable(mutable, history, cursor, cursor + width, norm)
        mutable.commit()
        cursor += width
    assert cursor == mutable.frontier == functional.frontier == 4_096
    mx.eval(
        mutable.packed_records,
        functional.packed_records,
        mutable.compressed_index_keys,
        functional.compressed_index_keys,
    )
    assert mutable.record_count == functional.packed_records.shape[0] == 0
    assert mutable.tail_count == functional.exact_tail_keys.shape[2] == 3_968
    assert np.array_equal(
        np.asarray(mutable.packed_records[: mutable.record_count]),
        np.asarray(functional.packed_records),
    )
    assert np.array_equal(
        np.asarray(
            mutable.compressed_index_keys[:, : mutable.index_group_count].astype(mx.float32)
        ),
        np.asarray(
            functional.compressed_index_keys[:, : functional.index_group_count].astype(mx.float32)
        ),
    )
    assert np.array_equal(
        np.asarray(mutable.compressed_index_positions[:, :, : mutable.index_group_count]),
        np.asarray(functional.compressed_index_positions[:, :, : functional.index_group_count]),
    )

    selected = mx.array(
        [
            [
                [
                    4_096,
                    0,
                    128,
                    127,
                    2_047,
                    3_071,
                    3_072,
                    4_095,
                    128,
                    3_200,
                    -1,
                    9_999,
                ]
            ]
        ],
        dtype=mx.int32,
    )
    functional_k, functional_v, functional_valid = gather_qsa_kvarn_selected_rows_with_pending(
        functional,
        keys[..., 4_096:4_097, :],
        values[..., 4_096:4_097, :],
        selected,
    )
    mutable_k, mutable_v, mutable_valid = mutable.gather_selected_rows_with_pending(
        keys[..., 4_096:4_097, :],
        values[..., 4_096:4_097, :],
        selected,
    )
    queries = mx.array(rng.normal(size=(1, 1, 24, 256)).astype(np.float32)).astype(mx.bfloat16)
    functional_attention = qsa_attention_from_selected_rows(
        queries,
        functional_k,
        functional_v,
        functional_valid,
    )
    mutable_attention = qsa_attention_from_selected_rows(
        queries,
        mutable_k,
        mutable_v,
        mutable_valid,
    )
    mx.eval(
        functional_k,
        functional_v,
        functional_valid,
        mutable_k,
        mutable_v,
        mutable_valid,
        functional_attention,
        mutable_attention,
    )
    assert np.array_equal(np.asarray(mutable_valid), np.asarray(functional_valid))
    assert np.array_equal(
        np.asarray(mutable_k.astype(mx.float32)),
        np.asarray(functional_k.astype(mx.float32)),
    )
    assert np.array_equal(
        np.asarray(mutable_v.astype(mx.float32)),
        np.asarray(functional_v.astype(mx.float32)),
    )
    assert np.array_equal(
        np.asarray(mutable_attention.astype(mx.float32)),
        np.asarray(functional_attention.astype(mx.float32)),
    )

    _, functional_update = prepare_qsa_kvarn_index(
        functional,
        index[:, 4_096:4_097],
        positions[:, :, 4_096:4_097],
        norm,
        max_index_groups=1_024,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    functional = advance_qsa_kvarn_state(
        functional,
        keys[..., 4_096:4_097, :],
        values[..., 4_096:4_097, :],
        valid[:, 4_096:4_097],
        index_update=functional_update,
    )
    _append_mutable(mutable, history, 4_096, 4_097, norm)
    mutable.commit()
    assert mutable.frontier == functional.frontier == 4_097
    assert mutable.tail_count == functional.exact_tail_keys.shape[2] == 3_969
    assert mutable.stats["tile_records_sealed"] == 0


def test_mutable_storage_fails_closed_on_capacity_dtype_geometry_and_padding() -> None:
    with pytest.raises(ValueError, match="max_context_tokens"):
        Qwen4MutableKVarNStorage(max_context_tokens=True)
    with pytest.raises(ValueError, match="integer dtype"):
        Qwen4MutableKVarNStorage(
            max_context_tokens=8,
            position_dtype=mx.float32,
        )

    history = _history(9, seed=817)
    keys, values, index, positions, valid = history
    norm = Qwen4RMSNorm(128)
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=8,
        encode_tile=_zero_tile_encoder,
    )
    _, update = storage.prepare_index(
        index[:, :1],
        positions[:, :, :1],
        norm,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    with pytest.raises(ValueError, match="BF16"):
        storage.append(
            keys[..., :1, :].astype(mx.float16),
            values[..., :1, :].astype(mx.float16),
            valid[:, :1],
            index_update=update,
        )
    with pytest.raises(ValueError, match="unpadded"):
        storage.append(
            keys[..., :1, :],
            values[..., :1, :],
            mx.zeros((1, 1), dtype=mx.bool_),
            index_update=update,
        )
    with pytest.raises(ValueError, match="match keys"):
        storage.append(
            keys[..., :1, :],
            values[..., :1, :1],
            valid[:, :1],
            index_update=update,
        )
    with pytest.raises(ValueError, match="position_ids"):
        storage.prepare_index(
            index[:, :1],
            positions[:, :, :1].astype(mx.int64),
            norm,
            rotary_dim=64,
            rope_base=10_000_000.0,
            mrope_section=(11, 11, 10),
        )
    with pytest.raises(ValueError, match="shape"):
        storage.gather_selected_rows(mx.zeros((2, 1, 1), dtype=mx.int32))
    with pytest.raises(ValueError, match="int32"):
        storage.gather_selected_rows(mx.zeros((1, 1, 1), dtype=mx.int64))

    _append_mutable(storage, history, 0, 8, norm)
    with pytest.raises(ValueError, match="capacity 8"):
        _append_mutable(storage, history, 8, 9, norm)
    foreign = Qwen4MutableKVarNStorage(
        max_context_tokens=8,
        encode_tile=_zero_tile_encoder,
    )
    with pytest.raises(ValueError, match="another storage"):
        foreign.restore(storage.checkpoint())
    with pytest.raises(ValueError, match="schema"):
        storage.restore(replace(storage.checkpoint(), schema="unsupported"))
    with pytest.raises(ValueError, match="geometry"):
        storage.restore(replace(storage.checkpoint(), tail_count=7))

    _, foreign_update = foreign.prepare_index(
        index[:, :1],
        positions[:, :, :1],
        norm,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    empty = Qwen4MutableKVarNStorage(
        max_context_tokens=8,
        encode_tile=_zero_tile_encoder,
    )
    with pytest.raises(ValueError, match="another storage"):
        empty.append(
            keys[..., :1, :],
            values[..., :1, :],
            valid[:, :1],
            index_update=foreign_update,
        )


def test_mutable_append_rejects_nonfinite_input_before_any_physical_write() -> None:
    history = _history(1, seed=819)
    keys, values, index, positions, valid = history
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=16,
        encode_tile=_zero_tile_encoder,
    )
    _, update = storage.prepare_index(
        index,
        positions,
        Qwen4RMSNorm(128),
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    before_bytes = _storage_bytes(storage)
    before_view = storage.checkpoint()
    before_journal_nbytes = storage.journal_nbytes

    bad_keys = mx.full(keys.shape, np.nan, dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="finite"):
        storage.append(
            bad_keys,
            values,
            valid,
            index_update=update,
        )

    assert _storage_bytes(storage) == before_bytes
    assert storage.checkpoint() == before_view
    assert storage.journal_nbytes == before_journal_nbytes
    assert storage.counters.append_calls == 0
    assert storage.counters.appended_tokens == 0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf], ids=["nan", "inf"])
def test_deferred_append_false_predicate_restores_exactly_and_allows_retry(
    bad_value: float,
) -> None:
    history = _history(1, seed=820)
    keys, values, index, positions, valid = history
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=16,
        encode_tile=_zero_tile_encoder,
    )
    _, update = storage.prepare_index(
        index,
        positions,
        Qwen4RMSNorm(128),
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    before_view = storage.checkpoint()
    before_bytes = _storage_bytes(storage)
    reservation = storage._prepare_undo_reservation(1)
    mx.eval(*storage._undo_reservation_arrays(reservation))
    storage._mark_undo_reservation_evaluated(reservation)

    bad_keys = mx.full(keys.shape, bad_value, dtype=mx.bfloat16)
    view, predicate = storage._append_with_deferred_finite(
        bad_keys,
        values,
        valid,
        index_update=update,
        undo_reservation=reservation,
        capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    )
    mx.eval(*storage._physical_arrays(), predicate)
    assert view.frontier == 1
    assert not bool(predicate.item())
    assert _storage_bytes(storage) != before_bytes

    storage.restore(before_view)
    assert _storage_bytes(storage) == before_bytes
    assert storage.checkpoint() == before_view
    assert storage.journal_nbytes == 0

    retry = storage._prepare_undo_reservation(1)
    mx.eval(*storage._undo_reservation_arrays(retry))
    storage._mark_undo_reservation_evaluated(retry)
    retry_view, retry_predicate = storage._append_with_deferred_finite(
        keys,
        values,
        valid,
        index_update=update,
        undo_reservation=retry,
        capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    )
    mx.eval(*storage._physical_arrays(), retry_predicate)
    assert bool(retry_predicate.item())
    assert retry_view.frontier == 1
    storage.commit(retry_view)
    assert storage.journal_nbytes == 0


@pytest.mark.parametrize(
    "frontier",
    [0, 3, 127, 129, 1_279],
    ids=["empty", "index-seal", "sink-edge", "tail", "tile-seal"],
)
def test_irreversible_deferred_append_matches_journaled_physical_state(
    frontier: int,
) -> None:
    history = _history(frontier + 1, seed=8_300 + frontier)
    keys, values, index, positions, valid = history
    norm = Qwen4RMSNorm(128)
    irreversible = Qwen4MutableKVarNStorage(
        max_context_tokens=max(4, frontier + 1),
        encode_tile=_zero_tile_encoder,
    )
    journaled = Qwen4MutableKVarNStorage(
        max_context_tokens=max(4, frontier + 1),
        encode_tile=_zero_tile_encoder,
    )
    if frontier:
        _append_mutable(irreversible, history, 0, frontier, norm)
        _append_mutable(journaled, history, 0, frontier, norm)
        irreversible.commit()
        journaled.commit()

    def prepare(storage: Qwen4MutableKVarNStorage):
        _, update = storage.prepare_index(
            index[:, frontier : frontier + 1],
            positions[:, :, frontier : frontier + 1],
            norm,
            rotary_dim=64,
            rope_base=10_000_000.0,
            mrope_section=(11, 11, 10),
        )
        return update

    undo_evals_before = irreversible.counters.local_undo_evals
    undo_reservations_before = irreversible.counters.undo_reservations_prepared
    irreversible_view, predicate = irreversible._append_irreversible_deferred_finite(
        keys[..., frontier : frontier + 1, :],
        values[..., frontier : frontier + 1, :],
        valid[:, frontier : frontier + 1],
        index_update=prepare(irreversible),
        capability=_QWEN4_SERIAL_IRREVOCABLE_APPEND_CAPABILITY,
    )
    journaled_view = journaled.append(
        keys[..., frontier : frontier + 1, :],
        values[..., frontier : frontier + 1, :],
        valid[:, frontier : frontier + 1],
        index_update=prepare(journaled),
    )
    mx.eval(*irreversible.state_arrays(), *journaled.state_arrays(), predicate)

    assert bool(predicate.item())
    assert _storage_bytes(irreversible) == _storage_bytes(journaled)
    for field in (
        "frontier",
        "body_frontier",
        "tail_start",
        "tail_count",
        "record_count",
        "index_group_count",
        "raw_index_count",
        "logical_nbytes",
        "schema",
    ):
        assert getattr(irreversible_view, field) == getattr(journaled_view, field)
    assert irreversible_view.mutation_cursor == 0
    assert journaled_view.mutation_cursor == 1
    assert irreversible.journal_nbytes == 0
    assert irreversible.counters.local_undo_evals == undo_evals_before
    assert irreversible.counters.undo_reservations_prepared == undo_reservations_before
    assert irreversible.counters.irreversible_append_calls == 1
    assert irreversible.counters.irreversible_appended_tokens == 1

    published = irreversible.commit(irreversible_view)
    assert published.frontier == journaled_view.frontier
    assert published.mutation_cursor == 0
    assert irreversible.journal_nbytes == 0


def test_irreversible_append_requires_capability_before_mutation() -> None:
    keys, values, index, positions, valid = _history(1, seed=8_306)
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=4,
        encode_tile=_zero_tile_encoder,
    )
    _, update = storage.prepare_index(
        index,
        positions,
        Qwen4RMSNorm(128),
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    before = _storage_bytes(storage)

    with pytest.raises(ValueError, match="capability"):
        storage._append_irreversible_deferred_finite(
            keys,
            values,
            valid,
            index_update=update,
            capability=object(),
        )

    assert _storage_bytes(storage) == before
    assert storage.checkpoint().frontier == 0
    assert not storage.abandoned


def test_false_irreversible_predicate_requires_abandon_and_refuses_reuse() -> None:
    keys, values, index, positions, valid = _history(1, seed=8_307)
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=4,
        encode_tile=_zero_tile_encoder,
    )
    _, update = storage.prepare_index(
        index,
        positions,
        Qwen4RMSNorm(128),
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    old_view = storage.checkpoint()
    next_view, predicate = storage._append_irreversible_deferred_finite(
        mx.full(keys.shape, np.nan, dtype=mx.bfloat16),
        values,
        valid,
        index_update=update,
        capability=_QWEN4_SERIAL_IRREVOCABLE_APPEND_CAPABILITY,
    )
    mx.eval(*storage.state_arrays(), predicate)
    assert not bool(predicate.item())
    assert next_view.frontier == 1
    assert storage.journal_nbytes == 0

    storage.abandon()
    storage.abandon()
    assert storage.abandoned
    assert storage.counters.abandon_calls == 1
    for operation in (
        storage.validate,
        storage.checkpoint,
        lambda: storage.restore(old_view),
        lambda: storage.commit(next_view),
        lambda: storage.gather_selected_rows(mx.zeros((1, 1, 1), dtype=mx.int32)),
    ):
        with pytest.raises(RuntimeError, match="abandoned"):
            operation()


def test_irreversible_post_mutation_failure_poisoning_is_automatic() -> None:
    fail = False

    def encoder(keys: mx.array, values: mx.array) -> mx.array:
        if fail:
            raise KeyboardInterrupt
        return _zero_tile_encoder(keys, values)

    history = _history(FIRST_SEAL, seed=8_308)
    keys, values, index, positions, valid = history
    norm = Qwen4RMSNorm(128)
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=FIRST_SEAL,
        encode_tile=encoder,
    )
    _append_mutable(storage, history, 0, FIRST_SEAL - 1, norm)
    storage.commit()
    _, update = storage.prepare_index(
        index[:, FIRST_SEAL - 1 : FIRST_SEAL],
        positions[:, :, FIRST_SEAL - 1 : FIRST_SEAL],
        norm,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )

    fail = True
    with pytest.raises(KeyboardInterrupt):
        storage._append_irreversible_deferred_finite(
            keys[..., FIRST_SEAL - 1 : FIRST_SEAL, :],
            values[..., FIRST_SEAL - 1 : FIRST_SEAL, :],
            valid[:, FIRST_SEAL - 1 : FIRST_SEAL],
            index_update=update,
            capability=_QWEN4_SERIAL_IRREVOCABLE_APPEND_CAPABILITY,
        )

    assert storage.abandoned
    assert storage.counters.post_mutation_failures == 1
    assert storage.counters.abandon_calls == 1
    assert storage.journal_nbytes == 0
    with pytest.raises(RuntimeError, match="abandoned"):
        storage.checkpoint()


@pytest.mark.parametrize(
    "frontier",
    [3, FIRST_SEAL - 1],
    ids=["index-group-seal", "kv-tile-seal"],
)
def test_deferred_append_retry_overwrites_unreachable_sealed_slots_exactly(
    frontier: int,
) -> None:
    history = _history(frontier + 1, seed=821 + frontier)
    keys, values, index, positions, valid = history
    norm = Qwen4RMSNorm(128)
    candidate = Qwen4MutableKVarNStorage(
        max_context_tokens=frontier + 1,
        encode_tile=_zero_tile_encoder,
    )
    control = Qwen4MutableKVarNStorage(
        max_context_tokens=frontier + 1,
        encode_tile=_zero_tile_encoder,
    )
    if frontier:
        _append_mutable(candidate, history, 0, frontier, norm)
        _append_mutable(control, history, 0, frontier, norm)
        candidate.commit()
        control.commit()

    def prepare(storage: Qwen4MutableKVarNStorage):
        _, update = storage.prepare_index(
            index[:, frontier : frontier + 1],
            positions[:, :, frontier : frontier + 1],
            norm,
            rotary_dim=64,
            rope_base=10_000_000.0,
            mrope_section=(11, 11, 10),
        )
        return update

    candidate_update = prepare(candidate)
    branch = candidate.checkpoint()
    reservation = candidate._prepare_undo_reservation(1)
    mx.eval(*candidate._undo_reservation_arrays(reservation))
    candidate._mark_undo_reservation_evaluated(reservation)
    _, predicate = candidate._append_with_deferred_finite(
        mx.full((1, 2, 1, 256), np.nan, dtype=mx.bfloat16),
        values[..., frontier : frontier + 1, :],
        valid[:, frontier : frontier + 1],
        index_update=candidate_update,
        undo_reservation=reservation,
        capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    )
    mx.eval(*candidate._physical_arrays(), predicate)
    assert not bool(predicate.item())
    candidate.restore(branch)

    retry = candidate._prepare_undo_reservation(1)
    mx.eval(*candidate._undo_reservation_arrays(retry))
    candidate._mark_undo_reservation_evaluated(retry)
    candidate_view, retry_predicate = candidate._append_with_deferred_finite(
        keys[..., frontier : frontier + 1, :],
        values[..., frontier : frontier + 1, :],
        valid[:, frontier : frontier + 1],
        index_update=candidate_update,
        undo_reservation=retry,
        capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    )
    control_view = control.append(
        keys[..., frontier : frontier + 1, :],
        values[..., frontier : frontier + 1, :],
        valid[:, frontier : frontier + 1],
        index_update=prepare(control),
    )
    mx.eval(*candidate._physical_arrays(), *control._physical_arrays(), retry_predicate)
    assert bool(retry_predicate.item())
    for field in (
        "frontier",
        "body_frontier",
        "tail_start",
        "tail_count",
        "record_count",
        "index_group_count",
        "raw_index_count",
        "mutation_cursor",
        "lineage",
        "logical_nbytes",
        "schema",
    ):
        assert getattr(candidate_view, field) == getattr(control_view, field)
    assert _storage_bytes(candidate) == _storage_bytes(control)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_prepared_undo_restores_bytes_after_cancellation_and_allows_retry() -> None:
    fail = False

    def encoder(keys: mx.array, values: mx.array) -> mx.array:
        if fail:
            raise KeyboardInterrupt
        return _zero_tile_encoder(keys, values)

    history = _history(FIRST_SEAL, seed=818)
    keys, values, index, positions, valid = history
    norm = Qwen4RMSNorm(128)
    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=FIRST_SEAL,
        encode_tile=encoder,
    )
    _append_mutable(storage, history, 0, FIRST_SEAL - 1, norm)
    storage.commit()
    _, update = storage.prepare_index(
        index[:, FIRST_SEAL - 1 : FIRST_SEAL],
        positions[:, :, FIRST_SEAL - 1 : FIRST_SEAL],
        norm,
        rotary_dim=64,
        rope_base=10_000_000.0,
        mrope_section=(11, 11, 10),
    )
    reservation = storage._prepare_undo_reservation(1)
    mx.eval(*storage._undo_reservation_arrays(reservation))
    storage._mark_undo_reservation_evaluated(reservation)
    before_arrays = tuple(mx.contiguous(array) for array in storage._physical_arrays())
    mx.eval(*before_arrays)
    before_bytes = tuple(bytes(memoryview(array).cast("B")) for array in before_arrays)
    before_view = storage.checkpoint()
    before_journal_nbytes = storage.journal_nbytes

    fail = True
    with pytest.raises(KeyboardInterrupt):
        storage.append(
            keys[..., FIRST_SEAL - 1 : FIRST_SEAL, :],
            values[..., FIRST_SEAL - 1 : FIRST_SEAL, :],
            valid[:, FIRST_SEAL - 1 : FIRST_SEAL],
            index_update=update,
            undo_reservation=reservation,
        )

    after_arrays = tuple(mx.contiguous(array) for array in storage._physical_arrays())
    mx.eval(*after_arrays)
    assert tuple(bytes(memoryview(array).cast("B")) for array in after_arrays) == before_bytes
    assert storage.checkpoint() == before_view
    assert storage.journal_nbytes == before_journal_nbytes

    fail = False
    retry = storage._prepare_undo_reservation(1)
    mx.eval(*storage._undo_reservation_arrays(retry))
    storage._mark_undo_reservation_evaluated(retry)
    view = storage.append(
        keys[..., FIRST_SEAL - 1 : FIRST_SEAL, :],
        values[..., FIRST_SEAL - 1 : FIRST_SEAL, :],
        valid[:, FIRST_SEAL - 1 : FIRST_SEAL],
        index_update=update,
        undo_reservation=retry,
    )
    assert view.frontier == FIRST_SEAL
    assert view.record_count == 1
