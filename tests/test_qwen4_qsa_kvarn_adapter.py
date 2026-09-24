from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import moespresso.runtime.qwen4.qsa as qsa_module
from moespresso.runtime.qwen4.model import (
    _Qwen4AppendFiniteBatch,
    _Qwen4TrustedMaskCertificate,
    _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    _QWEN4_BATCHED_UNDO_CAPABILITY,
    _QWEN4_SERIAL_LANE_CAPABILITY,
)
from moespresso.runtime.qwen4.kvarn_cache import (
    QWEN38_KVARN_EXACT_SINK,
    QWEN38_KVARN_EXACT_SUFFIX,
    QWEN38_QSA_TILE_TOKENS,
)
from moespresso.runtime.qwen4.kvarn_mutable import Qwen4MutableKVarNStorage
from moespresso.runtime.qwen4.qsa import (
    Qwen4BF16QSAStateBackend,
    Qwen4QSAAdapter,
    Qwen4QSAIndexPreparation,
    Qwen4SparseAttention,
)
from moespresso.runtime.qwen4.qsa_kvarn import (
    _Qwen4MutableIndexUpdate,
    Qwen4MutableKVarNQSAState,
    Qwen4MutableKVarNQSAStateBackend,
)


FIRST_SEAL = QWEN38_KVARN_EXACT_SINK + QWEN38_KVARN_EXACT_SUFFIX + QWEN38_QSA_TILE_TOKENS


def _module(seed: int = 701) -> Qwen4SparseAttention:
    module = Qwen4SparseAttention(hidden_size=4)
    rng = np.random.default_rng(seed)
    for projection in (
        module.indexer.index_qk_proj,
        module.q_proj,
        module.k_proj,
        module.v_proj,
        module.o_proj,
    ):
        projection.weight = mx.array(
            rng.normal(0.0, 0.02, size=projection.weight.shape).astype(np.float32)
        ).astype(mx.bfloat16)
    return module


def _history(token_count: int, seed: int = 702):
    rng = np.random.default_rng(seed)
    keys = mx.array(rng.normal(size=(1, 2, token_count, 256)).astype(np.float32)).astype(
        mx.bfloat16
    )
    values = mx.array(rng.normal(size=(1, 2, token_count, 256)).astype(np.float32)).astype(
        mx.bfloat16
    )
    index = mx.array(rng.normal(size=(1, token_count, 128)).astype(np.float32)).astype(mx.bfloat16)
    physical = np.arange(token_count, dtype=np.int32)
    positions = mx.array(np.broadcast_to(physical[None, None], (3, 1, token_count)).copy())
    valid = mx.ones((1, token_count), dtype=mx.bool_)
    return keys, values, index, positions, valid


def _mutable_state(
    module: Qwen4SparseAttention,
    token_count: int,
    *,
    max_context_tokens: int,
    seed: int,
) -> Qwen4MutableKVarNQSAState:
    keys, values, index, positions, valid = _history(token_count, seed=seed)
    storage = Qwen4MutableKVarNStorage(max_context_tokens=max_context_tokens)
    _, update = storage.prepare_index(
        index,
        positions,
        module.indexer.k_layernorm,
        rotary_dim=module.rotary_dim,
        rope_base=module.rope_base,
        mrope_section=module.mrope_section,
    )
    view = storage.append(
        keys,
        values,
        valid,
        index_update=update,
    )
    return Qwen4MutableKVarNQSAState(storage, storage.commit(view))


def _assert_mutable_state_arrays_exact(
    left: Qwen4MutableKVarNQSAState,
    right: Qwen4MutableKVarNQSAState,
) -> None:
    assert left.view == right.view
    left_arrays = left.storage._physical_arrays()
    right_arrays = right.storage._physical_arrays()
    mx.eval(*left_arrays, *right_arrays)
    for left_array, right_array in zip(left_arrays, right_arrays, strict=True):
        if left_array.dtype == mx.bfloat16:
            left_array = left_array.astype(mx.float32)
            right_array = right_array.astype(mx.float32)
        assert np.array_equal(np.asarray(left_array), np.asarray(right_array))


def _assert_mutable_state_bytes_exact(
    left: Qwen4MutableKVarNQSAState,
    right: Qwen4MutableKVarNQSAState,
) -> None:
    assert left.view == right.view
    left_arrays = tuple(mx.contiguous(array) for array in left.storage._physical_arrays())
    right_arrays = tuple(mx.contiguous(array) for array in right.storage._physical_arrays())
    mx.eval(*left_arrays, *right_arrays)
    for left_array, right_array in zip(left_arrays, right_arrays, strict=True):
        assert left_array.shape == right_array.shape
        assert left_array.dtype == right_array.dtype
        left_bytes = b"" if left_array.size == 0 else bytes(memoryview(left_array).cast("B"))
        right_bytes = b"" if right_array.size == 0 else bytes(memoryview(right_array).cast("B"))
        assert left_bytes == right_bytes


def _mutable_storage_bytes(state: Qwen4MutableKVarNQSAState) -> tuple[bytes, ...]:
    arrays = tuple(mx.contiguous(array) for array in state.storage._physical_arrays())
    mx.eval(*arrays)
    return tuple(b"" if array.size == 0 else bytes(memoryview(array).cast("B")) for array in arrays)


def _mutable_index_preparation(
    module: Qwen4SparseAttention,
    state: Qwen4MutableKVarNQSAState,
    history,
    start: int,
) -> Qwen4QSAIndexPreparation:
    _keys, _values, index, positions, _valid = history
    compressed, update = state.storage.prepare_index(
        index[:, start : start + 1],
        positions[:, :, start : start + 1],
        module.indexer.k_layernorm,
        rotary_dim=module.rotary_dim,
        rope_base=module.rope_base,
        mrope_section=module.mrope_section,
    )
    return Qwen4QSAIndexPreparation(
        compressed_keys=compressed,
        update=_Qwen4MutableIndexUpdate(state.storage, update),
    )


def test_bf16_backend_remains_the_default() -> None:
    adapter = Qwen4QSAAdapter(_module())
    assert isinstance(adapter.state_backend, Qwen4BF16QSAStateBackend)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_kvarn_certified_all_valid_step_matches_existing_checks() -> None:
    module = _module(seed=7061)
    candidate = Qwen4QSAAdapter(
        module,
        state_backend=Qwen4MutableKVarNQSAStateBackend(
            module,
            max_context_tokens=16,
        ),
    )
    incumbent = Qwen4QSAAdapter(
        module,
        state_backend=Qwen4MutableKVarNQSAStateBackend(
            module,
            max_context_tokens=16,
        ),
    )
    hidden = mx.array([[[0.25, -0.5, 0.75, 1.0]]], dtype=mx.bfloat16)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 1), dtype=mx.bool_)
    positions = mx.zeros((3, 1, 1), dtype=mx.int32)
    issuer = object()
    scope = object()
    candidate._bind_trusted_mask_issuer(issuer, scope)
    certificate = _Qwen4TrustedMaskCertificate(
        issuer=issuer,
        scope=scope,
        allowed_mixers=(candidate,),
        source_states=(None,),
        valid_tokens=valid,
        visible_history=visible,
        current_frontier=0,
        next_frontier=1,
    )

    candidate_result = candidate._step_trusted_certified(
        hidden,
        valid_tokens=valid,
        visible_history=visible,
        position_ids=positions,
        state=None,
        mask_certificate=certificate,
    )
    incumbent_result = incumbent.step_trusted(
        hidden,
        valid_tokens=valid,
        visible_history=visible,
        position_ids=positions,
        state=None,
    )
    candidate_output = mx.contiguous(candidate_result.output)
    incumbent_output = mx.contiguous(incumbent_result.output)
    mx.eval(candidate_output, incumbent_output)

    assert bytes(memoryview(candidate_output).cast("B")) == bytes(
        memoryview(incumbent_output).cast("B")
    )
    _assert_mutable_state_arrays_exact(candidate_result.state, incumbent_result.state)
    assert candidate.trusted_mask_stats() == {
        "trusted_mask_certificate_calls": 1,
        "trusted_all_valid_certificate_calls": 1,
    }
    assert incumbent.trusted_mask_stats() == {
        "trusted_mask_certificate_calls": 0,
        "trusted_all_valid_certificate_calls": 0,
    }


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_kvarn_shared_rope_step_is_byte_exact() -> None:
    module = _module(seed=7064)
    candidate = Qwen4QSAAdapter(
        module,
        state_backend=Qwen4MutableKVarNQSAStateBackend(
            module,
            max_context_tokens=132,
        ),
    )
    incumbent = Qwen4QSAAdapter(
        module,
        state_backend=Qwen4MutableKVarNQSAStateBackend(
            module,
            max_context_tokens=132,
        ),
    )
    candidate_state = _mutable_state(
        module,
        129,
        max_context_tokens=132,
        seed=7065,
    )
    incumbent_state = _mutable_state(
        module,
        129,
        max_context_tokens=132,
        seed=7065,
    )
    hidden = mx.array([[[0.25, -0.5, 0.75, 1.0]]], dtype=mx.bfloat16)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 130), dtype=mx.bool_)
    positions = mx.array([[[129]], [[17]], [[3]]], dtype=mx.int32)
    factors = candidate._prepare_shared_rope_factors(positions)

    candidate_result = candidate._step_trusted_prepared(
        hidden,
        valid_tokens=valid,
        visible_history=visible,
        position_ids=positions,
        state=candidate_state,
        mask_certificate=None,
        shared_rope_factors=factors,
    )
    incumbent_result = incumbent.step_trusted(
        hidden,
        valid_tokens=valid,
        visible_history=visible,
        position_ids=positions,
        state=incumbent_state,
    )
    candidate_output = mx.contiguous(candidate_result.output)
    incumbent_output = mx.contiguous(incumbent_result.output)
    mx.eval(candidate_output, incumbent_output)

    assert bytes(memoryview(candidate_output).cast("B")) == bytes(
        memoryview(incumbent_output).cast("B")
    )
    _assert_mutable_state_bytes_exact(candidate_result.state, incumbent_result.state)
    assert candidate.shared_rope_stats() == {
        "shared_rope_factor_calls": 1,
        "shared_rope_applications": 3,
    }
    assert incumbent.shared_rope_stats() == {
        "shared_rope_factor_calls": 0,
        "shared_rope_applications": 0,
    }


def test_mutable_kvarn_shared_rope_rejects_stale_position_object_before_append() -> None:
    module = _module(seed=7066)
    adapter = Qwen4QSAAdapter(
        module,
        state_backend=Qwen4MutableKVarNQSAStateBackend(
            module,
            max_context_tokens=4,
        ),
    )
    positions = mx.zeros((3, 1, 1), dtype=mx.int32)
    factors = adapter._prepare_shared_rope_factors(positions)
    equal_positions = mx.array(np.asarray(positions).copy())

    with pytest.raises(ValueError, match="semantic positions"):
        adapter._step_trusted_prepared(
            mx.ones((1, 1, 4), dtype=mx.bfloat16),
            valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
            visible_history=mx.ones((1, 1), dtype=mx.bool_),
            position_ids=equal_positions,
            state=None,
            mask_certificate=None,
            shared_rope_factors=factors,
        )


def test_kvarn_rejects_unissued_certificate_and_padding_hole_without_appending() -> None:
    module = _module(seed=7062)
    backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=16,
    )
    adapter = Qwen4QSAAdapter(module, state_backend=backend)
    hidden = mx.ones((1, 1, 4), dtype=mx.bfloat16)
    valid = mx.zeros((1, 1), dtype=mx.bool_)
    visible = mx.zeros((1, 1), dtype=mx.bool_)
    issuer = object()
    certificate = _Qwen4TrustedMaskCertificate(
        issuer=issuer,
        scope=object(),
        allowed_mixers=(adapter,),
        source_states=(None,),
        valid_tokens=valid,
        visible_history=visible,
        current_frontier=0,
        next_frontier=1,
    )

    with pytest.raises(ValueError, match="unissued or stale"):
        adapter._step_trusted_certified(
            hidden,
            valid_tokens=valid,
            visible_history=visible,
            position_ids=mx.zeros((3, 1, 1), dtype=mx.int32),
            state=None,
            mask_certificate=certificate,
        )
    with pytest.raises(ValueError, match="padding holes"):
        adapter.step_trusted(
            hidden,
            valid_tokens=valid,
            visible_history=visible,
            position_ids=mx.zeros((3, 1, 1), dtype=mx.int32),
            state=None,
        )

    assert backend.index_stats(None)["prepare_calls"] == 0


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_kvarn_certificate_does_not_bypass_the_finite_gate() -> None:
    module = _module(seed=7063)
    backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=16,
    )
    adapter = Qwen4QSAAdapter(module, state_backend=backend)
    first = adapter.step_trusted(
        mx.ones((1, 1, 4), dtype=mx.bfloat16),
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
        visible_history=mx.ones((1, 1), dtype=mx.bool_),
        position_ids=mx.zeros((3, 1, 1), dtype=mx.int32),
        state=None,
    )
    state = backend.commit_state(first.state)
    assert state is not None
    physical = tuple(mx.contiguous(array) for array in state.storage._physical_arrays())
    mx.eval(*physical)
    before = tuple(
        b"" if array.size == 0 else bytes(memoryview(array).cast("B")) for array in physical
    )
    before_view = state.storage.checkpoint()
    before_journal_nbytes = state.storage.journal_nbytes
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 2), dtype=mx.bool_)
    reservation = adapter._prepare_trusted_undo(
        state,
        new_tokens=1,
        capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
    )
    mx.eval(
        *adapter._trusted_undo_arrays(
            reservation,
            capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
        )
    )
    adapter._mark_trusted_undo_evaluated(
        reservation,
        capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
    )
    issuer = object()
    scope = object()
    adapter._bind_trusted_mask_issuer(issuer, scope)
    certificate = _Qwen4TrustedMaskCertificate(
        issuer=issuer,
        scope=scope,
        allowed_mixers=(adapter,),
        source_states=(state,),
        valid_tokens=valid,
        visible_history=visible,
        current_frontier=1,
        next_frontier=2,
        prepared_undos=(reservation,),
    )

    try:
        with pytest.raises(ValueError, match="finite"):
            adapter._step_trusted_certified(
                mx.array([[[np.nan, 0.0, 0.0, 0.0]]], dtype=mx.bfloat16),
                valid_tokens=valid,
                visible_history=visible,
                position_ids=mx.ones((3, 1, 1), dtype=mx.int32),
                state=state,
                mask_certificate=certificate,
            )
    finally:
        adapter._clear_trusted_mask_scope(issuer, scope)

    after_arrays = tuple(mx.contiguous(array) for array in state.storage._physical_arrays())
    mx.eval(*after_arrays)
    assert (
        tuple(
            b"" if array.size == 0 else bytes(memoryview(array).cast("B")) for array in after_arrays
        )
        == before
    )
    assert state.storage.checkpoint() == before_view
    assert state.storage.journal_nbytes == before_journal_nbytes
    assert state.storage.counters.undo_reservations_prepared == 1
    assert state.storage.counters.undo_reservations_evaluated == 1
    assert state.storage.counters.undo_reservations_consumed == 0


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_kvarn_certificate_defers_only_the_redundant_finite_check(
    monkeypatch,
) -> None:
    module = _module(seed=7064)
    backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=16,
    )
    adapter = Qwen4QSAAdapter(module, state_backend=backend)
    first = adapter.step_trusted(
        mx.ones((1, 1, 4), dtype=mx.bfloat16),
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
        visible_history=mx.ones((1, 1), dtype=mx.bool_),
        position_ids=mx.zeros((3, 1, 1), dtype=mx.int32),
        state=None,
    )
    state = backend.commit_state(first.state)
    assert state is not None
    with pytest.raises(ValueError, match="deferred finite capability"):
        backend._gather_selected_rows_deferred_finite(
            state,
            mx.ones((1, 2, 1, 256), dtype=mx.bfloat16),
            mx.ones((1, 2, 1, 256), dtype=mx.bfloat16),
            mx.zeros((1, 1, 2051), dtype=mx.int32),
            capability=object(),
        )
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 2), dtype=mx.bool_)
    issuer = object()
    scope = object()
    adapter._bind_trusted_mask_issuer(issuer, scope)
    certificate = _Qwen4TrustedMaskCertificate(
        issuer=issuer,
        scope=scope,
        allowed_mixers=(adapter,),
        source_states=(state,),
        valid_tokens=valid,
        visible_history=visible,
        current_frontier=1,
        next_frontier=2,
    )
    before = state.storage.counters.as_dict()
    backend_before = backend.index_stats(state)
    try:
        result = adapter._step_trusted_certified(
            mx.ones((1, 1, 4), dtype=mx.bfloat16),
            valid_tokens=valid,
            visible_history=visible,
            position_ids=mx.ones((3, 1, 1), dtype=mx.int32),
            state=state,
            mask_certificate=certificate,
        )
        mx.eval(result.output)
    finally:
        adapter._clear_trusted_mask_scope(issuer, scope)
    after = state.storage.counters.as_dict()
    backend_after = backend.index_stats(result.state)
    assert after["deferred_pending_value_checks"] - before["deferred_pending_value_checks"] == 1
    assert after["pending_value_checks"] == before["pending_value_checks"]
    assert after["append_value_checks"] - before["append_value_checks"] == 1
    assert (
        backend_after["deferred_pending_value_checks"]
        - backend_before["deferred_pending_value_checks"]
        == 1
    )
    assert backend_after["pending_value_checks"] == backend_before["pending_value_checks"]
    assert backend_after["append_value_checks"] - backend_before["append_value_checks"] == 1

    monkeypatch.setattr(backend, "_gather_selected_rows_deferred_finite", None)
    backend.restore_state(state)
    scope = object()
    adapter._bind_trusted_mask_issuer(issuer, scope)
    certificate = _Qwen4TrustedMaskCertificate(
        issuer=issuer,
        scope=scope,
        allowed_mixers=(adapter,),
        source_states=(state,),
        valid_tokens=valid,
        visible_history=visible,
        current_frontier=1,
        next_frontier=2,
    )
    before = state.storage.counters.as_dict()
    backend_before = backend.index_stats(state)
    try:
        result = adapter._step_trusted_certified(
            mx.ones((1, 1, 4), dtype=mx.bfloat16),
            valid_tokens=valid,
            visible_history=visible,
            position_ids=mx.ones((3, 1, 1), dtype=mx.int32),
            state=state,
            mask_certificate=certificate,
        )
        mx.eval(result.output)
    finally:
        adapter._clear_trusted_mask_scope(issuer, scope)
    after = state.storage.counters.as_dict()
    backend_after = backend.index_stats(result.state)
    assert after["pending_value_checks"] - before["pending_value_checks"] == 1
    assert after["deferred_pending_value_checks"] == before["deferred_pending_value_checks"]
    assert after["append_value_checks"] - before["append_value_checks"] == 1
    assert backend_after["pending_value_checks"] - backend_before["pending_value_checks"] == 1
    assert (
        backend_after["deferred_pending_value_checks"]
        == backend_before["deferred_pending_value_checks"]
    )
    assert backend_after["append_value_checks"] - backend_before["append_value_checks"] == 1


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_kvarn_serial_certificate_uses_irreversible_append() -> None:
    module = _module(seed=7068)
    backend = Qwen4MutableKVarNQSAStateBackend(module, max_context_tokens=16)
    adapter = Qwen4QSAAdapter(module, state_backend=backend)
    first = adapter.step_trusted(
        mx.ones((1, 1, 4), dtype=mx.bfloat16),
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
        visible_history=mx.ones((1, 1), dtype=mx.bool_),
        position_ids=mx.zeros((3, 1, 1), dtype=mx.int32),
        state=None,
    )
    state = backend.commit_state(first.state)
    assert state is not None
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 2), dtype=mx.bool_)
    issuer = object()
    scope = object()
    batch = _Qwen4AppendFiniteBatch(
        allowed_mixers=(adapter,),
        source_states=(state,),
    )
    certificate = _Qwen4TrustedMaskCertificate(
        issuer=issuer,
        scope=scope,
        allowed_mixers=(adapter,),
        source_states=(state,),
        valid_tokens=valid,
        visible_history=visible,
        current_frontier=1,
        next_frontier=2,
        append_finite_batch=batch,
        serial_capability=_QWEN4_SERIAL_LANE_CAPABILITY,
    )
    adapter._bind_trusted_mask_issuer(issuer, scope)
    try:
        result = adapter._step_trusted_certified(
            mx.ones((1, 1, 4), dtype=mx.bfloat16),
            valid_tokens=valid,
            visible_history=visible,
            position_ids=mx.ones((3, 1, 1), dtype=mx.int32),
            state=state,
            mask_certificate=certificate,
        )
        finite = batch.seal(
            ((adapter, result.state),),
            capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
        )
        mx.eval(result.output, *result.state.storage.state_arrays(), finite.predicate)
        assert bool(finite.predicate.item())
        committed = backend.commit_state(result.state)
    finally:
        adapter._clear_trusted_mask_scope(issuer, scope)

    assert committed is not None
    counters = committed.storage.counters
    assert counters.irreversible_append_calls == 1
    assert counters.irreversible_appended_tokens == 1
    assert counters.undo_reservations_prepared == 0
    assert counters.undo_reservations_consumed == 0
    assert committed.storage.journal_nbytes == 0

    adapter.abandon_serial_state(committed)
    with pytest.raises(RuntimeError, match="abandoned"):
        backend.validate_state_structure(
            committed,
            expected_frontier=2,
            batch_size=1,
            position_dtype=mx.int32,
            projection_dtype=mx.bfloat16,
        )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_kvarn_deferred_append_rejects_bad_reservations_before_mutation() -> None:
    module = _module(seed=7069)
    backend = Qwen4MutableKVarNQSAStateBackend(module, max_context_tokens=16)
    adapter = Qwen4QSAAdapter(module, state_backend=backend)
    history = _history(3, seed=7070)
    keys, values, _index, _positions, valid = history
    state = _mutable_state(module, 1, max_context_tokens=16, seed=7070)
    preparation = _mutable_index_preparation(module, state, history, 1)

    reservation = adapter._prepare_trusted_undo(
        state,
        new_tokens=1,
        capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
    )
    before = _mutable_storage_bytes(state)
    with pytest.raises(ValueError, match="append finite capability"):
        backend._append_with_prepared_undo_deferred_finite(
            state,
            keys=keys[..., 1:2, :],
            values=values[..., 1:2, :],
            valid_tokens=valid[:, 1:2],
            index_preparation=preparation,
            undo_reservation=reservation,
            capability=object(),
        )
    assert _mutable_storage_bytes(state) == before
    assert state.storage.journal_nbytes == 0

    foreign_state = _mutable_state(module, 1, max_context_tokens=16, seed=7071)
    foreign_history = _history(3, seed=7071)
    foreign_preparation = _mutable_index_preparation(module, foreign_state, foreign_history, 1)
    foreign_reservation = adapter._prepare_trusted_undo(
        foreign_state,
        new_tokens=1,
        capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
    )
    before = _mutable_storage_bytes(state)
    with pytest.raises(ValueError, match="another request"):
        backend._append_with_prepared_undo_deferred_finite(
            state,
            keys=keys[..., 1:2, :],
            values=values[..., 1:2, :],
            valid_tokens=valid[:, 1:2],
            index_preparation=foreign_preparation,
            undo_reservation=foreign_reservation,
            capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
        )
    assert _mutable_storage_bytes(state) == before
    assert state.storage.journal_nbytes == 0

    unevaluated = adapter._prepare_trusted_undo(
        state,
        new_tokens=1,
        capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
    )
    before = _mutable_storage_bytes(state)
    with pytest.raises(ValueError, match="not evaluated"):
        backend._append_with_prepared_undo_deferred_finite(
            state,
            keys=keys[..., 1:2, :],
            values=values[..., 1:2, :],
            valid_tokens=valid[:, 1:2],
            index_preparation=preparation,
            undo_reservation=unevaluated,
            capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
        )
    assert _mutable_storage_bytes(state) == before
    assert state.storage.journal_nbytes == 0

    stale = adapter._prepare_trusted_undo(
        state,
        new_tokens=1,
        capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
    )
    mx.eval(*adapter._trusted_undo_arrays(stale, capability=_QWEN4_BATCHED_UNDO_CAPABILITY))
    adapter._mark_trusted_undo_evaluated(
        stale,
        capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
    )
    next_state = backend.append(
        state,
        keys=keys[..., 1:2, :],
        values=values[..., 1:2, :],
        valid_tokens=valid[:, 1:2],
        index_preparation=preparation,
    )
    next_preparation = _mutable_index_preparation(module, next_state, history, 2)
    before = _mutable_storage_bytes(next_state)
    with pytest.raises(ValueError, match="stale"):
        backend._append_with_prepared_undo_deferred_finite(
            next_state,
            keys=keys[..., 2:3, :],
            values=values[..., 2:3, :],
            valid_tokens=valid[:, 2:3],
            index_preparation=next_preparation,
            undo_reservation=stale,
            capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
        )
    assert _mutable_storage_bytes(next_state) == before
    assert next_state.storage.checkpoint() == next_state.view


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_kvarn_prepared_undo_is_byte_exact_and_one_shot() -> None:
    module = _module(seed=7067)
    candidate_backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=132,
    )
    incumbent_backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=132,
    )
    candidate = Qwen4QSAAdapter(module, state_backend=candidate_backend)
    incumbent = Qwen4QSAAdapter(module, state_backend=incumbent_backend)
    candidate_state = _mutable_state(
        module,
        129,
        max_context_tokens=132,
        seed=7068,
    )
    incumbent_state = _mutable_state(
        module,
        129,
        max_context_tokens=132,
        seed=7068,
    )
    hidden = mx.array([[[0.25, -0.5, 0.75, 1.0]]], dtype=mx.bfloat16)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 130), dtype=mx.bool_)
    positions = mx.full((3, 1, 1), 129, dtype=mx.int32)

    reservation = candidate._prepare_trusted_undo(
        candidate_state,
        new_tokens=1,
        capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
    )
    arrays = candidate._trusted_undo_arrays(
        reservation,
        capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
    )
    mx.eval(*arrays)
    candidate._mark_trusted_undo_evaluated(
        reservation,
        capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
    )

    candidate_issuer = object()
    candidate_scope = object()
    candidate._bind_trusted_mask_issuer(candidate_issuer, candidate_scope)
    candidate_certificate = _Qwen4TrustedMaskCertificate(
        issuer=candidate_issuer,
        scope=candidate_scope,
        allowed_mixers=(candidate,),
        source_states=(candidate_state,),
        valid_tokens=valid,
        visible_history=visible,
        current_frontier=129,
        next_frontier=130,
        prepared_undos=(reservation,),
    )
    incumbent_issuer = object()
    incumbent_scope = object()
    incumbent._bind_trusted_mask_issuer(incumbent_issuer, incumbent_scope)
    incumbent_certificate = _Qwen4TrustedMaskCertificate(
        issuer=incumbent_issuer,
        scope=incumbent_scope,
        allowed_mixers=(incumbent,),
        source_states=(incumbent_state,),
        valid_tokens=valid,
        visible_history=visible,
        current_frontier=129,
        next_frontier=130,
    )
    try:
        candidate_result = candidate._step_trusted_certified(
            hidden,
            valid_tokens=valid,
            visible_history=visible,
            position_ids=positions,
            state=candidate_state,
            mask_certificate=candidate_certificate,
        )
        incumbent_result = incumbent._step_trusted_certified(
            hidden,
            valid_tokens=valid,
            visible_history=visible,
            position_ids=positions,
            state=incumbent_state,
            mask_certificate=incumbent_certificate,
        )
    finally:
        candidate._clear_trusted_mask_scope(candidate_issuer, candidate_scope)
        incumbent._clear_trusted_mask_scope(incumbent_issuer, incumbent_scope)

    candidate_output = mx.contiguous(candidate_result.output)
    incumbent_output = mx.contiguous(incumbent_result.output)
    mx.eval(candidate_output, incumbent_output)
    assert bytes(memoryview(candidate_output).cast("B")) == bytes(
        memoryview(incumbent_output).cast("B")
    )
    _assert_mutable_state_bytes_exact(candidate_result.state, incumbent_result.state)
    candidate_counters = candidate_result.state.storage.counters
    incumbent_counters = incumbent_result.state.storage.counters
    assert candidate_counters.undo_reservations_prepared == 1
    assert candidate_counters.undo_reservations_evaluated == 1
    assert candidate_counters.undo_reservations_consumed == 1
    assert candidate_counters.local_undo_evals + 1 == incumbent_counters.local_undo_evals
    with pytest.raises(ValueError, match="consumed"):
        candidate._trusted_undo_arrays(
            reservation,
            capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
        )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_backend_restores_and_commits_process_local_frontiers() -> None:
    module = _module(seed=707)
    backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=16,
    )
    adapter = Qwen4QSAAdapter(
        module,
        state_backend=backend,
        max_query_tokens=4,
    )
    rng = np.random.default_rng(707)
    hidden = mx.array(rng.normal(size=(1, 3, 4)).astype(np.float32)).astype(mx.bfloat16)
    positions = mx.array(
        np.broadcast_to(np.arange(3, dtype=np.int32)[None, None], (3, 1, 3)).copy()
    )
    first = adapter.step_trusted(
        hidden,
        valid_tokens=mx.ones((1, 3), dtype=mx.bool_),
        visible_history=mx.ones((1, 3), dtype=mx.bool_),
        position_ids=positions,
        state=None,
    )
    checkpoint = backend.snapshot_state(first.state)
    branch = backend.fork_state(checkpoint)
    assert branch is not None
    extension = adapter.step_trusted(
        hidden[:, :2],
        valid_tokens=mx.ones((1, 2), dtype=mx.bool_),
        visible_history=mx.ones((1, 5), dtype=mx.bool_),
        position_ids=positions[:, :, :2] + 3,
        state=branch,
    )
    assert extension.frontier == 5

    backend.restore_state(checkpoint)
    adapter.validate_state(
        checkpoint,
        expected_frontier=3,
        position_history=positions,
    )
    committed = backend.commit_state(checkpoint)
    assert committed is not None
    assert committed.view.frontier == 3
    assert committed.storage.journal_nbytes == 0
    with pytest.raises(ValueError, match="discarded branch|expired lineage"):
        backend.restore_state(extension.state)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_adapter_materializes_once_and_capability_fallback_restores_tiled_gathers(
    monkeypatch,
) -> None:
    module = _module(seed=711)
    candidate_backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=FIRST_SEAL + 4,
    )
    incumbent_backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=FIRST_SEAL + 4,
    )
    candidate = Qwen4QSAAdapter(
        module,
        state_backend=candidate_backend,
        max_query_tokens=2,
    )
    incumbent = Qwen4QSAAdapter(
        module,
        state_backend=incumbent_backend,
        max_query_tokens=2,
    )
    candidate_state = _mutable_state(
        module,
        FIRST_SEAL - 1,
        max_context_tokens=FIRST_SEAL + 4,
        seed=712,
    )
    incumbent_state = _mutable_state(
        module,
        FIRST_SEAL - 1,
        max_context_tokens=FIRST_SEAL + 4,
        seed=712,
    )
    rng = np.random.default_rng(713)
    hidden = mx.array(rng.normal(size=(1, 5, 4)).astype(np.float32)).astype(mx.bfloat16)
    positions = mx.array(
        np.broadcast_to(
            np.arange(FIRST_SEAL - 1, FIRST_SEAL + 4, dtype=np.int32)[None, None],
            (3, 1, 5),
        ).copy()
    )
    visible = mx.ones((1, FIRST_SEAL + 4), dtype=mx.bool_)

    candidate_result = candidate.step_trusted(
        hidden,
        valid_tokens=mx.ones((1, 5), dtype=mx.bool_),
        visible_history=visible,
        position_ids=positions,
        state=candidate_state,
    )
    monkeypatch.setattr(incumbent_backend, "prepare_prefill_selected_rows", None)
    monkeypatch.setattr(incumbent_backend, "gather_prepared_selected_rows", None)
    incumbent_result = incumbent.step_trusted(
        hidden,
        valid_tokens=mx.ones((1, 5), dtype=mx.bool_),
        visible_history=visible,
        position_ids=positions,
        state=incumbent_state,
    )
    mx.eval(candidate_result.output, incumbent_result.output)

    assert np.array_equal(
        np.asarray(candidate_result.output.astype(mx.float32)),
        np.asarray(incumbent_result.output.astype(mx.float32)),
    )
    assert candidate_result.frontier == incumbent_result.frontier == FIRST_SEAL + 4
    _assert_mutable_state_arrays_exact(candidate_result.state, incumbent_result.state)
    candidate_stats = candidate_result.state.storage.stats
    incumbent_stats = incumbent_result.state.storage.stats
    assert candidate_stats["prefill_materialize_calls"] == 1
    assert candidate_stats["prefill_materialize_lanes"] == FIRST_SEAL + 4
    assert candidate_stats["gather_calls"] == 0
    assert candidate_stats["pending_gather_calls"] == 0
    assert incumbent_stats["prefill_materialize_calls"] == 0
    assert incumbent_stats["prefill_materialize_lanes"] == 0
    assert incumbent_stats["gather_calls"] == 3
    assert incumbent_stats["pending_gather_calls"] == 3
    assert candidate_stats["tile_records_sealed"] == 1
    assert incumbent_stats["tile_records_sealed"] == 1


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mutable_adapter_keeps_cold_bounded_and_single_token_paths_on_selected_gather(
    monkeypatch,
) -> None:
    module = _module(seed=714)
    cold_backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=16,
    )
    cold_adapter = Qwen4QSAAdapter(
        module,
        state_backend=cold_backend,
        max_query_tokens=2,
    )
    cold_hidden = mx.ones((1, 2, 4), dtype=mx.bfloat16)
    cold_result = cold_adapter.step_trusted(
        cold_hidden,
        valid_tokens=mx.ones((1, 2), dtype=mx.bool_),
        visible_history=mx.ones((1, 2), dtype=mx.bool_),
        position_ids=mx.array(
            np.broadcast_to(np.arange(2, dtype=np.int32)[None, None], (3, 1, 2)).copy()
        ),
        state=None,
    )
    assert cold_result.state.storage.stats["prefill_materialize_calls"] == 0
    assert cold_result.state.storage.stats["gather_calls"] == 0

    bounded_backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=4_103,
    )
    bounded = Qwen4QSAAdapter(
        module,
        state_backend=bounded_backend,
        max_query_tokens=2,
    )
    bounded_state = _mutable_state(
        module,
        4_101,
        max_context_tokens=4_103,
        seed=716,
    )
    bounded_result = bounded.step_trusted(
        cold_hidden,
        valid_tokens=mx.ones((1, 2), dtype=mx.bool_),
        visible_history=mx.ones((1, 4_103), dtype=mx.bool_),
        position_ids=mx.array(
            np.broadcast_to(
                np.arange(4_101, 4_103, dtype=np.int32)[None, None],
                (3, 1, 2),
            ).copy()
        ),
        state=bounded_state,
    )
    bounded_stats = bounded_result.state.storage.stats
    assert bounded_stats["prefill_materialize_calls"] == 0
    assert bounded_stats["prefill_materialize_lanes"] == 0
    assert bounded_stats["gather_calls"] == 1
    assert bounded_stats["pending_gather_calls"] == 1

    enabled_backend = Qwen4MutableKVarNQSAStateBackend(
        module,
        max_context_tokens=16,
    )
    enabled = Qwen4QSAAdapter(module, state_backend=enabled_backend, max_query_tokens=2)
    enabled_state = _mutable_state(
        module,
        8,
        max_context_tokens=16,
        seed=715,
    )
    token = mx.array([[[0.25, -0.5, 0.75, 1.0]]], dtype=mx.bfloat16)
    token_positions = mx.full((3, 1, 1), 8, dtype=mx.int32)
    token_visibility = mx.ones((1, 9), dtype=mx.bool_)
    enabled_result = enabled.step_trusted(
        token,
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
        visible_history=token_visibility,
        position_ids=token_positions,
        state=enabled_state,
    )
    mx.eval(enabled_result.output)
    stats = enabled_result.state.storage.stats
    assert stats["prefill_materialize_calls"] == 0
    assert stats["prefill_materialize_lanes"] == 0
    assert stats["gather_calls"] == 1
    assert stats["pending_gather_calls"] == 1


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    ("next_frontier", "candidate_calls"),
    [(2_048, 1), (2_049, 0)],
    ids=("budget-boundary", "first-sparse-fallback"),
)
def test_mutable_single_query_short_selection_is_exact_and_bounded(
    monkeypatch,
    next_frontier: int,
    candidate_calls: int,
) -> None:
    class RecordingBackend(Qwen4MutableKVarNQSAStateBackend):
        def __init__(self, module: Qwen4SparseAttention) -> None:
            super().__init__(module, max_context_tokens=next_frontier + 1)
            self.selections: list[mx.array] = []

        def gather_selected_rows(
            self,
            state,
            pending_keys,
            pending_values,
            selected_indices,
        ):
            self.selections.append(mx.contiguous(selected_indices))
            return super().gather_selected_rows(
                state,
                pending_keys,
                pending_values,
                selected_indices,
            )

    module = _module(seed=720 + next_frontier)
    candidate_backend = RecordingBackend(module)
    incumbent_backend = RecordingBackend(module)
    candidate = Qwen4QSAAdapter(
        module,
        state_backend=candidate_backend,
        max_query_tokens=2,
    )
    incumbent = Qwen4QSAAdapter(
        module,
        state_backend=incumbent_backend,
        max_query_tokens=2,
    )
    candidate_state = _mutable_state(
        module,
        next_frontier - 1,
        max_context_tokens=next_frontier + 1,
        seed=721 + next_frontier,
    )
    incumbent_state = _mutable_state(
        module,
        next_frontier - 1,
        max_context_tokens=next_frontier + 1,
        seed=721 + next_frontier,
    )
    token = mx.array([[[0.25, -0.5, 0.75, 1.0]]], dtype=mx.bfloat16)
    token_positions = mx.full(
        (3, 1, 1),
        next_frontier - 1,
        dtype=mx.int32,
    )
    visibility = mx.ones((1, next_frontier), dtype=mx.bool_)
    score_calls = []
    score = qsa_module.qsa_index_scores

    def counted_score(*args, **kwargs):
        score_calls.append(True)
        return score(*args, **kwargs)

    monkeypatch.setattr(qsa_module, "qsa_index_scores", counted_score)
    candidate_result = candidate.step_trusted(
        token,
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
        visible_history=visibility,
        position_ids=token_positions,
        state=candidate_state,
    )
    candidate_score_calls = len(score_calls)
    score_calls.clear()
    monkeypatch.setattr(incumbent_backend, "select_all_valid_short_context", lambda *_a, **_k: None)
    incumbent_result = incumbent.step_trusted(
        token,
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
        visible_history=visibility,
        position_ids=token_positions,
        state=incumbent_state,
    )
    mx.eval(
        candidate_result.output,
        incumbent_result.output,
        *candidate_backend.selections,
        *incumbent_backend.selections,
    )

    assert candidate_score_calls == (0 if next_frontier <= 2_048 else 1)
    assert score_calls == [True]
    assert len(candidate_backend.selections) == len(incumbent_backend.selections) == 1
    candidate_selected = candidate_backend.selections[0]
    incumbent_selected = incumbent_backend.selections[0]
    assert candidate_selected.shape == incumbent_selected.shape == (1, 1, 2_051)
    expected_selected = np.full((1, 1, 2_051), -1, dtype=np.int32)
    expected_selected[0, 0, :next_frontier] = np.arange(
        next_frontier,
        dtype=np.int32,
    )
    expected_bytes = expected_selected.tobytes()
    assert bytes(memoryview(candidate_selected).cast("B")) == expected_bytes
    assert bytes(memoryview(incumbent_selected).cast("B")) == expected_bytes
    candidate_output = mx.contiguous(candidate_result.output)
    incumbent_output = mx.contiguous(incumbent_result.output)
    mx.eval(candidate_output, incumbent_output)
    assert bytes(memoryview(candidate_output).cast("B")) == bytes(
        memoryview(incumbent_output).cast("B")
    )
    _assert_mutable_state_bytes_exact(candidate_result.state, incumbent_result.state)
    candidate_stats = candidate_backend.index_stats(candidate_result.state)
    incumbent_stats = incumbent_backend.index_stats(incumbent_result.state)
    assert candidate_stats["short_all_selected_calls"] == candidate_calls
    assert candidate_stats["short_all_selected_lanes"] == (next_frontier if candidate_calls else 0)
    assert incumbent_stats["short_all_selected_calls"] == 0
    assert incumbent_stats["short_all_selected_lanes"] == 0
    for name in ("prepare_calls", "groups_sealed", "groups_reused", "retained_raw_rows"):
        assert candidate_stats[name] == incumbent_stats[name]
