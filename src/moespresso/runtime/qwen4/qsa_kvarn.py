"""K4/V4 state backends for released Qwen sparse attention."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from moespresso.runtime.qwen4.kvarn_cache import (
    Qwen4KVarNIndexUpdate,
    gather_qsa_kvarn_selected_rows_with_pending,
)
from moespresso.runtime.qwen4.qsa import (
    Qwen4QSAIndexPreparation,
    Qwen4SparseAttention,
    QSAVisiblePrefixLayout,
    _QSA_DEFER_PENDING_FINITE_CAPABILITY,
    _QSA_TRUSTED_ALL_VALID_CAPABILITY,
    _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    _QWEN4_BATCHED_UNDO_CAPABILITY,
    _QWEN4_SERIAL_LANE_CAPABILITY,
    qsa_gather_selected_rows,
)
from moespresso.runtime.qwen4.kvarn_mutable import (
    Qwen4MutableKVarNFrontier,
    Qwen4MutableKVarNStorage,
    _Qwen4MutableUndoReservation,
    _QWEN4_SERIAL_IRREVOCABLE_APPEND_CAPABILITY,
)

_RELEASED_GEOMETRY = {
    "num_query_heads": 24,
    "num_kv_heads": 2,
    "head_dim": 256,
    "index_query_heads": 4,
    "index_kv_heads": 1,
    "index_head_dim": 128,
    "token_budget": 2048,
    "compress_ratio": 4,
    "rotary_dim": 64,
}


@dataclass(frozen=True)
class Qwen4MutableKVarNQSAState:
    """One request-owned mutable store viewed at a public QSA frontier."""

    storage: Qwen4MutableKVarNStorage
    view: Qwen4MutableKVarNFrontier


@dataclass(frozen=True)
class _Qwen4MutableIndexUpdate:
    storage: Qwen4MutableKVarNStorage
    update: Qwen4KVarNIndexUpdate


@dataclass(frozen=True)
class _Qwen4MutablePreparedRows:
    """Transient logical BF16 K/V view for one multi-query segment."""

    storage: Qwen4MutableKVarNStorage
    frontier: int
    keys: mx.array
    values: mx.array


class Qwen4MutableKVarNQSAStateBackend:
    """Serve QSA from fixed-capacity K4/V4 storage owned by one request.

    The state contains only a process-local frontier marker and its storage.
    Prefix and disk caches use the independent payload in ``kvarn_snapshot``;
    they do not serialize the marker. The text shell restores a selected marker
    before publication and commits it after
    publication, which bounds the rollback journal to one proposal lineage.
    """

    supports_trusted_mask_certificate = True
    supports_batched_append_finite = True
    supports_dependency_bound_undo = True
    supports_serial_irrevocable_append = True

    def __init__(
        self,
        module: Qwen4SparseAttention,
        *,
        max_context_tokens: int,
    ) -> None:
        self.module = module
        if (
            isinstance(max_context_tokens, bool)
            or not isinstance(max_context_tokens, int)
            or max_context_tokens < module.compress_ratio
        ):
            raise ValueError(f"max_context_tokens must be an integer >= {module.compress_ratio}")
        observed = {name: getattr(module, name) for name in _RELEASED_GEOMETRY}
        if observed != _RELEASED_GEOMETRY:
            raise ValueError(
                "Qwen KVarN requires released QSA geometry: "
                f"expected {_RELEASED_GEOMETRY}, got {observed}"
            )
        self.max_context_tokens = max_context_tokens
        self.index_prepare_calls = 0
        self.index_groups_sealed = 0
        self.index_groups_reused = 0
        self.short_all_selected_calls = 0
        self.short_all_selected_lanes = 0
        self.pending_value_checks = 0
        self.deferred_pending_value_checks = 0
        self.append_value_checks = 0
        self.local_append_value_checks = 0
        self.deferred_append_value_checks = 0
        self.fused_mutable_gather_calls = 0
        self.fused_mutable_gather_lanes = 0
        self.native_select_gather_calls = 0
        self.native_select_gather_groups = 0

    def _validate_state(
        self,
        state: Qwen4MutableKVarNQSAState,
    ) -> Qwen4MutableKVarNStorage:
        if not isinstance(state, Qwen4MutableKVarNQSAState):
            raise ValueError("mutable Qwen KVarN state is incompatible")
        storage = state.storage
        if storage.max_context_tokens != self.max_context_tokens:
            raise ValueError("mutable Qwen KVarN context capacity is incompatible")
        storage.validate_frontier(state.view)
        return storage

    def fork_state(
        self,
        state: Qwen4MutableKVarNQSAState | None,
    ) -> Qwen4MutableKVarNQSAState | None:
        if state is None:
            return None
        storage = self._validate_state(state)
        return Qwen4MutableKVarNQSAState(storage, storage.fork(state.view))

    def snapshot_state(
        self,
        state: Qwen4MutableKVarNQSAState | None,
    ) -> Qwen4MutableKVarNQSAState | None:
        if state is None:
            return None
        storage = self._validate_state(state)
        current = storage.checkpoint()
        if current != state.view:
            raise ValueError("mutable Qwen KVarN state is not the active view")
        return Qwen4MutableKVarNQSAState(storage, current)

    def restore_state(
        self,
        state: Qwen4MutableKVarNQSAState | None,
    ) -> None:
        if state is None:
            return
        self._validate_state(state).restore(state.view)

    def abandon_serial_state(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        *,
        capability: object,
    ) -> None:
        """Permanently invalidate one request-private mutable storage."""

        if capability is not _QWEN4_SERIAL_LANE_CAPABILITY:
            raise ValueError("Qwen KVarN serial lane capability is invalid")
        if state is None:
            return
        if not isinstance(state, Qwen4MutableKVarNQSAState):
            raise ValueError("mutable Qwen KVarN state is incompatible")
        state.storage.abandon()

    def commit_state(
        self,
        state: Qwen4MutableKVarNQSAState | None,
    ) -> Qwen4MutableKVarNQSAState | None:
        self.preflight_commit_state(state)
        return self.commit_state_preflighted(state)

    def preflight_commit_state(
        self,
        state: Qwen4MutableKVarNQSAState | None,
    ) -> None:
        if state is None:
            return
        storage = self._validate_state(state)
        storage.preflight_commit(state.view)

    @staticmethod
    def commit_state_preflighted(
        state: Qwen4MutableKVarNQSAState | None,
    ) -> Qwen4MutableKVarNQSAState | None:
        if state is None:
            return None
        return Qwen4MutableKVarNQSAState(
            state.storage,
            state.storage.commit_preflighted(),
        )

    def frontier(self, state: Qwen4MutableKVarNQSAState | None) -> int:
        if state is None:
            return 0
        self._validate_state(state)
        return state.view.frontier

    def validate_state_structure(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        *,
        expected_frontier: int,
        batch_size: int,
        position_dtype: mx.Dtype,
        projection_dtype: mx.Dtype,
    ) -> None:
        if projection_dtype != mx.bfloat16:
            raise ValueError("Qwen KVarN requires BF16 QSA projections")
        if batch_size != 1:
            raise ValueError("Qwen KVarN requires batch size one")
        if expected_frontier == 0:
            if state is not None:
                raise ValueError("mutable Qwen KVarN state must be empty at frontier zero")
            return
        if state is None:
            raise ValueError("mutable Qwen KVarN state is missing at a live frontier")
        storage = self._validate_state(state)
        if state.view.frontier != expected_frontier:
            raise ValueError("mutable Qwen KVarN state is off the public frontier")
        if storage.position_dtype != position_dtype:
            raise ValueError("mutable Qwen KVarN cached positions have incompatible dtype")

    def validate_position_identity(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        position_history: mx.array,
    ) -> None:
        if state is None:
            if position_history.shape[-1]:
                raise ValueError("Qwen KVarN position history is live without cache state")
            return
        storage = self._validate_state(state)
        current = storage.checkpoint()
        if current != state.view:
            raise ValueError("mutable Qwen KVarN state is not the active view")
        grouped_end = state.view.index_group_count * self.module.compress_ratio
        expected_groups = position_history[:, :, : grouped_end : self.module.compress_ratio]
        expected_tail = position_history[:, :, grouped_end : state.view.frontier]
        stored_groups = storage.compressed_index_positions[:, :, : state.view.index_group_count]
        stored_tail = storage.raw_index_positions[:, :, : state.view.raw_index_count]
        mx.eval(stored_groups, expected_groups, stored_tail, expected_tail)
        if not np.array_equal(np.asarray(stored_groups), np.asarray(expected_groups)):
            raise ValueError("QSA cached positions do not match the public history")
        if not np.array_equal(np.asarray(stored_tail), np.asarray(expected_tail)):
            raise ValueError("QSA cached positions do not match the public history")

    def validate_step_inputs(
        self,
        *,
        hidden_states: mx.array,
        valid_tokens: mx.array,
        visible_history: mx.array,
        projection_dtype: mx.Dtype,
    ) -> None:
        self._validate_step_inputs(
            hidden_states=hidden_states,
            valid_tokens=valid_tokens,
            visible_history=visible_history,
            projection_dtype=projection_dtype,
            check_masks=True,
        )

    def _validate_step_inputs_certified(
        self,
        *,
        hidden_states: mx.array,
        valid_tokens: mx.array,
        visible_history: mx.array,
        projection_dtype: mx.Dtype,
        capability: object,
    ) -> None:
        if capability is not _QSA_TRUSTED_ALL_VALID_CAPABILITY:
            raise ValueError("Qwen KVarN trusted mask capability is invalid")
        self._validate_step_inputs(
            hidden_states=hidden_states,
            valid_tokens=valid_tokens,
            visible_history=visible_history,
            projection_dtype=projection_dtype,
            check_masks=False,
        )

    @staticmethod
    def _validate_step_inputs(
        *,
        hidden_states: mx.array,
        valid_tokens: mx.array,
        visible_history: mx.array,
        projection_dtype: mx.Dtype,
        check_masks: bool,
    ) -> None:
        if projection_dtype != mx.bfloat16 or hidden_states.dtype != mx.bfloat16:
            raise ValueError("Qwen KVarN requires BF16 QSA projections and hidden states")
        if hidden_states.shape[0] != 1:
            raise ValueError("Qwen KVarN requires batch size one")
        if check_masks and (
            not bool(mx.all(valid_tokens).item()) or not bool(mx.all(visible_history).item())
        ):
            raise ValueError("Qwen KVarN does not support padding holes")

    def safe_chunk_tokens(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        requested: int,
    ) -> int:
        remaining = self.max_context_tokens - self.frontier(state)
        if remaining <= 0:
            raise ValueError(
                f"Qwen KVarN context capacity {self.max_context_tokens} token(s) is exhausted"
            )
        return min(requested, remaining)

    def prepare_index(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        *,
        raw_index_keys: mx.array,
        position_ids: mx.array,
        layout: QSAVisiblePrefixLayout,
        key_norm,
        rotary_dim: int,
        rope_base: float,
        mrope_section: tuple[int, int, int],
    ) -> Qwen4QSAIndexPreparation:
        return self._prepare_index(
            state,
            raw_index_keys=raw_index_keys,
            position_ids=position_ids,
            layout=layout,
            key_norm=key_norm,
            rotary_dim=rotary_dim,
            rope_base=rope_base,
            mrope_section=mrope_section,
            check_groups=True,
        )

    def _prepare_index_certified(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        *,
        raw_index_keys: mx.array,
        position_ids: mx.array,
        layout: QSAVisiblePrefixLayout,
        key_norm,
        rotary_dim: int,
        rope_base: float,
        mrope_section: tuple[int, int, int],
        capability: object,
    ) -> Qwen4QSAIndexPreparation:
        if capability is not _QSA_TRUSTED_ALL_VALID_CAPABILITY:
            raise ValueError("Qwen KVarN trusted mask capability is invalid")
        return self._prepare_index(
            state,
            raw_index_keys=raw_index_keys,
            position_ids=position_ids,
            layout=layout,
            key_norm=key_norm,
            rotary_dim=rotary_dim,
            rope_base=rope_base,
            mrope_section=mrope_section,
            check_groups=False,
        )

    def _prepare_index(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        *,
        raw_index_keys: mx.array,
        position_ids: mx.array,
        layout: QSAVisiblePrefixLayout,
        key_norm,
        rotary_dim: int,
        rope_base: float,
        mrope_section: tuple[int, int, int],
        check_groups: bool,
    ) -> Qwen4QSAIndexPreparation:
        storage = (
            Qwen4MutableKVarNStorage(
                max_context_tokens=self.max_context_tokens,
                position_dtype=position_ids.dtype,
            )
            if state is None
            else self._validate_state(state)
        )
        compressed, update = storage.prepare_index(
            raw_index_keys,
            position_ids,
            key_norm,
            rotary_dim=rotary_dim,
            rope_base=rope_base,
            mrope_section=mrope_section,
        )
        expected_groups = layout.group_ids.shape[1]
        if compressed.shape != (1, expected_groups, self.module.index_head_dim):
            raise ValueError("Qwen KVarN prepared index does not match QSA visibility")
        if check_groups and not bool(mx.all(layout.group_valid).item()):
            raise ValueError("Qwen KVarN compressed index requires complete visible groups")
        self.index_prepare_calls += 1
        self.index_groups_sealed += int(update.new_group_keys.shape[1])
        self.index_groups_reused += int(update.old_group_count)
        return Qwen4QSAIndexPreparation(
            compressed_keys=compressed,
            update=_Qwen4MutableIndexUpdate(storage, update),
        )

    def index_stats(
        self,
        state: Qwen4MutableKVarNQSAState | None,
    ) -> dict[str, int]:
        storage_counters = None if state is None else state.storage.counters
        return {
            "prepare_calls": self.index_prepare_calls,
            "groups_sealed": self.index_groups_sealed,
            "groups_reused": self.index_groups_reused,
            "short_all_selected_calls": self.short_all_selected_calls,
            "short_all_selected_lanes": self.short_all_selected_lanes,
            "pending_value_checks": self.pending_value_checks,
            "deferred_pending_value_checks": self.deferred_pending_value_checks,
            "append_value_checks": self.append_value_checks,
            "local_append_value_checks": self.local_append_value_checks,
            "deferred_append_value_checks": self.deferred_append_value_checks,
            "fused_mutable_gather_calls": self.fused_mutable_gather_calls,
            "fused_mutable_gather_lanes": self.fused_mutable_gather_lanes,
            "native_select_gather_calls": self.native_select_gather_calls,
            "native_select_gather_groups": self.native_select_gather_groups,
            "undo_reservations_prepared": (
                0 if storage_counters is None else storage_counters.undo_reservations_prepared
            ),
            "undo_reservations_evaluated": (
                0 if storage_counters is None else storage_counters.undo_reservations_evaluated
            ),
            "undo_reservations_dependency_bound": (
                0
                if storage_counters is None
                else storage_counters.undo_reservations_dependency_bound
            ),
            "undo_reservations_consumed": (
                0 if storage_counters is None else storage_counters.undo_reservations_consumed
            ),
            "local_undo_evals": (
                0 if storage_counters is None else storage_counters.local_undo_evals
            ),
            "retained_raw_rows": 0 if state is None else state.view.raw_index_count,
            "capacity_groups": self.max_context_tokens // self.module.compress_ratio,
        }

    def select_all_valid_short_context(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        *,
        pending_tokens: int,
        token_budget: int,
    ) -> mx.array | None:
        """Return the fixed-width ascending prefix after all-valid validation."""
        if pending_tokens != 1 or token_budget != self.module.token_budget:
            return None
        next_frontier = self.frontier(state) + pending_tokens
        if next_frontier > token_budget:
            return None
        self.short_all_selected_calls += 1
        self.short_all_selected_lanes += next_frontier
        selected_width = token_budget + self.module.compress_ratio - 1
        selected = mx.pad(
            mx.arange(next_frontier, dtype=mx.int32),
            [(0, selected_width - next_frontier)],
            constant_values=-1,
        )
        return selected.reshape(1, 1, selected_width)

    def gather_selected_rows(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        pending_keys: mx.array,
        pending_values: mx.array,
        selected_indices: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if state is None:
            return gather_qsa_kvarn_selected_rows_with_pending(
                None,
                pending_keys,
                pending_values,
                selected_indices,
            )
        storage = self._validate_state(state)
        self.pending_value_checks += 1
        return storage.gather_selected_rows_with_pending(
            pending_keys,
            pending_values,
            selected_indices,
        )

    def _gather_selected_rows_deferred_finite(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        pending_keys: mx.array,
        pending_values: mx.array,
        selected_indices: mx.array,
        *,
        capability: object,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Defer the redundant pending-row check to the same-step append."""
        if capability is not _QSA_DEFER_PENDING_FINITE_CAPABILITY:
            raise ValueError("Qwen KVarN deferred finite capability is invalid")
        if state is None:
            return self.gather_selected_rows(
                state,
                pending_keys,
                pending_values,
                selected_indices,
            )
        storage = self._validate_state(state)
        self.deferred_pending_value_checks += 1
        before_calls = storage.counters.fused_mutable_gather_calls
        before_lanes = storage.counters.fused_mutable_gather_lanes
        result = storage._gather_selected_rows_with_pending_deferred_finite(
            pending_keys,
            pending_values,
            selected_indices,
        )
        self.fused_mutable_gather_calls += (
            storage.counters.fused_mutable_gather_calls - before_calls
        )
        self.fused_mutable_gather_lanes += (
            storage.counters.fused_mutable_gather_lanes - before_lanes
        )
        return result

    def _select_and_gather_rows_deferred_finite(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        pending_keys: mx.array,
        pending_values: mx.array,
        scores: mx.array,
        *,
        visible_count: int,
        capability: object,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array] | None:
        """Run stable selection and exact mutable gather without publishing state."""
        if capability is not _QSA_DEFER_PENDING_FINITE_CAPABILITY:
            raise ValueError("Qwen KVarN deferred finite capability is invalid")
        if state is None:
            return None
        import mlx_kquant

        operation = getattr(mlx_kquant, "qwen4_qsa_select_gather_k4v4", None)
        if not callable(operation):
            return None
        storage = self._validate_state(state)
        self.deferred_pending_value_checks += 1
        selected, valid, keys, values = operation(
            scores,
            storage.packed_records[: storage.record_count],
            storage.exact_sink_keys,
            storage.exact_sink_values,
            storage.exact_tail_keys,
            storage.exact_tail_values,
            pending_keys,
            pending_values,
            visible_count=visible_count,
            frontier=storage.frontier,
        )
        lanes = int(selected.size)
        storage.counters.fused_mutable_gather_calls += 1
        storage.counters.fused_mutable_gather_lanes += lanes
        self.fused_mutable_gather_calls += 1
        self.fused_mutable_gather_lanes += lanes
        self.native_select_gather_calls += 1
        self.native_select_gather_groups += int(scores.shape[-1])
        return selected, valid, keys, values

    def prepare_prefill_selected_rows(
        self,
        state: Qwen4MutableKVarNQSAState,
        pending_keys: mx.array,
        pending_values: mx.array,
    ) -> _Qwen4MutablePreparedRows:
        """Decode prior packed rows once for one live multi-query segment."""
        storage = self._validate_state(state)
        self.pending_value_checks += 1
        keys, values = storage.materialize_rows_with_pending(
            pending_keys,
            pending_values,
        )
        frontier = state.view.frontier + int(pending_keys.shape[2])
        expected = (
            1,
            self.module.num_kv_heads,
            frontier,
            self.module.head_dim,
        )
        if keys.shape != expected or values.shape != expected:
            raise ValueError("materialized Qwen KVarN rows have incompatible geometry")
        if keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16:
            raise ValueError("materialized Qwen KVarN rows must use BF16")
        mx.eval(keys, values)
        return _Qwen4MutablePreparedRows(
            storage=storage,
            frontier=frontier,
            keys=keys,
            values=values,
        )

    def gather_prepared_selected_rows(
        self,
        prepared: _Qwen4MutablePreparedRows,
        selected_indices: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Gather one selector tile from a prepared logical BF16 view."""
        if not isinstance(prepared, _Qwen4MutablePreparedRows):
            raise TypeError("prepared rows are incompatible with mutable Qwen KVarN")
        if prepared.storage.max_context_tokens != self.max_context_tokens:
            raise ValueError("prepared rows belong to an incompatible Qwen KVarN store")
        if prepared.keys.shape[2] != prepared.frontier:
            raise ValueError("prepared rows do not match their Qwen KVarN frontier")
        keys, values, valid = qsa_gather_selected_rows(
            prepared.keys,
            prepared.values,
            selected_indices,
        )
        mask = valid[:, :, None, :, None]
        return mx.where(mask, keys, 0), mx.where(mask, values, 0), valid

    def append(
        self,
        state: Qwen4MutableKVarNQSAState | None,
        *,
        keys: mx.array,
        values: mx.array,
        valid_tokens: mx.array,
        index_preparation: Qwen4QSAIndexPreparation,
    ) -> Qwen4MutableKVarNQSAState:
        prepared = index_preparation.update
        if not isinstance(prepared, _Qwen4MutableIndexUpdate):
            raise TypeError("mutable KVarN index preparation carries an incompatible update")
        if state is not None and prepared.storage is not self._validate_state(state):
            raise ValueError("mutable KVarN index preparation targets another request")
        self.append_value_checks += 1
        self.local_append_value_checks += 1
        view = prepared.storage.append(
            keys,
            values,
            valid_tokens,
            index_update=prepared.update,
        )
        return Qwen4MutableKVarNQSAState(prepared.storage, view)

    def _prepare_trusted_undo(
        self,
        state: Qwen4MutableKVarNQSAState,
        *,
        new_tokens: int,
        capability: object,
    ) -> _Qwen4MutableUndoReservation:
        """Prepare a rollback reservation for the shell's decode boundary."""

        if capability is not _QWEN4_BATCHED_UNDO_CAPABILITY:
            raise ValueError("Qwen KVarN batched undo capability is invalid")
        storage = self._validate_state(state)
        return storage._prepare_undo_reservation(new_tokens)

    def _trusted_undo_arrays(
        self,
        reservation: _Qwen4MutableUndoReservation,
        *,
        capability: object,
    ) -> tuple[mx.array, ...]:
        """Return arrays frozen by the shell's existing decode boundary."""

        if capability is not _QWEN4_BATCHED_UNDO_CAPABILITY:
            raise ValueError("Qwen KVarN batched undo capability is invalid")
        return reservation.storage._undo_reservation_arrays(reservation)

    def _mark_trusted_undo_evaluated(
        self,
        reservation: _Qwen4MutableUndoReservation,
        *,
        capability: object,
    ) -> None:
        """Mark a rollback reservation evaluated by the model shell."""

        if capability is not _QWEN4_BATCHED_UNDO_CAPABILITY:
            raise ValueError("Qwen KVarN batched undo capability is invalid")
        reservation.storage._mark_undo_reservation_evaluated(reservation)

    def _bind_trusted_undo_dependencies(
        self,
        reservation: _Qwen4MutableUndoReservation,
        inputs: tuple[mx.array, ...],
        *,
        capability: object,
    ) -> tuple[mx.array, ...]:
        """Order append inputs after lazy rollback copies without a host wait."""

        if capability is not _QWEN4_BATCHED_UNDO_CAPABILITY:
            raise ValueError("Qwen KVarN batched undo capability is invalid")
        if not isinstance(reservation, _Qwen4MutableUndoReservation):
            raise TypeError("Qwen KVarN undo reservation is incompatible")
        if reservation.storage.max_context_tokens != self.max_context_tokens:
            raise ValueError("Qwen KVarN undo reservation has incompatible capacity")
        if reservation.evaluated:
            return inputs
        return reservation.storage._bind_undo_reservation_dependencies(reservation, inputs)

    def _append_with_prepared_undo(
        self,
        state: Qwen4MutableKVarNQSAState,
        *,
        keys: mx.array,
        values: mx.array,
        valid_tokens: mx.array,
        index_preparation: Qwen4QSAIndexPreparation,
        undo_reservation: _Qwen4MutableUndoReservation,
        capability: object,
    ) -> Qwen4MutableKVarNQSAState:
        """Append using a shell-evaluated rollback reservation."""

        if capability is not _QWEN4_BATCHED_UNDO_CAPABILITY:
            raise ValueError("Qwen KVarN batched undo capability is invalid")
        prepared = index_preparation.update
        if not isinstance(prepared, _Qwen4MutableIndexUpdate):
            raise TypeError("mutable KVarN index preparation carries an incompatible update")
        storage = self._validate_state(state)
        if prepared.storage is not storage or undo_reservation.storage is not storage:
            raise ValueError("mutable KVarN prepared append targets another request")
        self.append_value_checks += 1
        self.local_append_value_checks += 1
        view = storage.append(
            keys,
            values,
            valid_tokens,
            index_update=prepared.update,
            undo_reservation=undo_reservation,
        )
        return Qwen4MutableKVarNQSAState(storage, view)

    def _append_with_prepared_undo_deferred_finite(
        self,
        state: Qwen4MutableKVarNQSAState,
        *,
        keys: mx.array,
        values: mx.array,
        valid_tokens: mx.array,
        index_preparation: Qwen4QSAIndexPreparation,
        undo_reservation: _Qwen4MutableUndoReservation,
        capability: object,
    ) -> tuple[Qwen4MutableKVarNQSAState, mx.array]:
        """Append privately and defer its finite scalar read to publication."""

        if capability is not _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY:
            raise ValueError("Qwen KVarN append finite capability is invalid")
        prepared = index_preparation.update
        if not isinstance(prepared, _Qwen4MutableIndexUpdate):
            raise TypeError("mutable KVarN index preparation carries an incompatible update")
        storage = self._validate_state(state)
        if prepared.storage is not storage or undo_reservation.storage is not storage:
            raise ValueError("mutable KVarN deferred append targets another request")
        self.append_value_checks += 1
        self.deferred_append_value_checks += 1
        view, predicate = storage._append_with_deferred_finite(
            keys,
            values,
            valid_tokens,
            index_update=prepared.update,
            undo_reservation=undo_reservation,
            capability=capability,
        )
        return Qwen4MutableKVarNQSAState(storage, view), predicate

    def _append_irreversible_deferred_finite(
        self,
        state: Qwen4MutableKVarNQSAState,
        *,
        keys: mx.array,
        values: mx.array,
        valid_tokens: mx.array,
        index_preparation: Qwen4QSAIndexPreparation,
        capability: object,
    ) -> tuple[Qwen4MutableKVarNQSAState, mx.array]:
        """Append one private serial row without preparing rollback state."""

        if capability is not _QWEN4_SERIAL_LANE_CAPABILITY:
            raise ValueError("Qwen KVarN serial lane capability is invalid")
        prepared = index_preparation.update
        if not isinstance(prepared, _Qwen4MutableIndexUpdate):
            raise TypeError("mutable KVarN index preparation carries an incompatible update")
        storage = self._validate_state(state)
        if prepared.storage is not storage:
            raise ValueError("mutable KVarN serial append targets another request")
        self.append_value_checks += 1
        self.deferred_append_value_checks += 1
        view, predicate = storage._append_irreversible_deferred_finite(
            keys,
            values,
            valid_tokens,
            index_update=prepared.update,
            capability=_QWEN4_SERIAL_IRREVOCABLE_APPEND_CAPABILITY,
        )
        return Qwen4MutableKVarNQSAState(storage, view), predicate
