"""Request-owned fixed-capacity K4/V4 storage for Qwen sparse attention.

The functional cache in :mod:`moespresso.runtime.qwen4.kvarn_cache` remains
the independent representation oracle. This module owns the mutable serving
storage: packed records, an exact 128-token sink, an exact 8,320-row tail ring,
and the already-shared incremental compressed-index arrays. Ordinary append
uses indexed writes and never grows or copies the complete K/V history.

Frontier views are process-local rollback markers. They are deliberately not a
prefix-cache or disk-checkpoint payload. ``kvarn_snapshot`` captures the live
tensors under a separate durable schema and recreates ownership on restore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import mlx.core as mx

from moespresso.runtime.qwen4.kvarn_cache import (
    QWEN38_KVARN_EXACT_SINK,
    QWEN38_QSA_TILE_TOKENS,
    Qwen4KVarNIndexUpdate,
    prepare_qsa_kvarn_index_storage,
    qsa_kvarn_partition,
)
from moespresso.runtime.qwen4.kvarn_encode import encode_qsa_kvarn_tile_mlx
from moespresso.runtime.qwen4.kvarn_layout import QWEN38_KVARN_K4V4_G128
from moespresso.runtime.qwen4.kvarn_selected_rows import (
    _decode_qsa_kvarn_rows_metal,
    gather_qsa_kvarn_mutable_rows_metal,
)
from moespresso.runtime.qwen4.qsa import (
    QWEN38_QSA_COMPRESS_RATIO,
    _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    qsa_normalize_selected_rows,
)


QWEN38_KVARN_EXACT_TAIL_CAPACITY = 8_320
_RAW_INDEX_CAPACITY = QWEN38_QSA_COMPRESS_RATIO - 1
_SCHEMA = "qwen38_qsa_kvarn_k4v4_g128_mutable_v1"
_QWEN4_SERIAL_IRREVOCABLE_APPEND_CAPABILITY = object()

TileEncoder = Callable[[mx.array, mx.array], mx.array]


@dataclass(frozen=True)
class Qwen4MutableKVarNFrontier:
    """Lightweight process-local view of one mutable storage frontier."""

    frontier: int
    body_frontier: int
    tail_start: int
    tail_count: int
    record_count: int
    index_group_count: int
    raw_index_count: int
    mutation_cursor: int
    mutation_anchor: int
    lineage: int
    logical_nbytes: int
    schema: str = _SCHEMA
    _storage_identity: object = field(repr=False, compare=False, default=None)


@dataclass
class Qwen4MutableKVarNCounters:
    """Monotonic operational counters for one request-owned storage object."""

    append_calls: int = 0
    appended_tokens: int = 0
    indexed_kv_write_calls: int = 0
    sink_rows_written: int = 0
    tail_rows_written: int = 0
    tile_records_sealed: int = 0
    index_groups_sealed: int = 0
    gather_calls: int = 0
    gather_lanes: int = 0
    pending_gather_calls: int = 0
    fused_mutable_gather_calls: int = 0
    fused_mutable_gather_lanes: int = 0
    pending_value_checks: int = 0
    deferred_pending_value_checks: int = 0
    append_value_checks: int = 0
    local_append_value_checks: int = 0
    deferred_append_value_checks: int = 0
    undo_reservations_prepared: int = 0
    undo_reservations_evaluated: int = 0
    undo_reservations_dependency_bound: int = 0
    undo_reservations_consumed: int = 0
    local_undo_evals: int = 0
    prefill_materialize_calls: int = 0
    prefill_materialize_lanes: int = 0
    restore_calls: int = 0
    committed_lineages: int = 0
    irreversible_append_calls: int = 0
    irreversible_appended_tokens: int = 0
    abandon_calls: int = 0
    post_mutation_failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class _SavedRange:
    start: int
    keys: mx.array
    values: mx.array


@dataclass(frozen=True)
class _UndoMutation:
    mutation_id: int
    frontier: int
    body_frontier: int
    record_count: int
    index_group_count: int
    raw_index_count: int
    raw_index_keys: mx.array
    raw_index_positions: mx.array
    sink_ranges: tuple[_SavedRange, ...]
    tail_ranges: tuple[_SavedRange, ...]

    @property
    def nbytes(self) -> int:
        arrays = [self.raw_index_keys, self.raw_index_positions]
        for saved in (*self.sink_ranges, *self.tail_ranges):
            arrays.extend((saved.keys, saved.values))
        return sum(int(array.nbytes) for array in arrays)


@dataclass
class _Qwen4MutableUndoReservation:
    """Side-effect-free rollback snapshot prepared for one future append."""

    storage: Any = field(repr=False)
    storage_identity: object = field(repr=False)
    lineage: int
    journal_length: int
    new_tokens: int
    mutation: _UndoMutation
    evaluated: bool = False
    dependency_bound: bool = False
    consumed: bool = False


class Qwen4MutableKVarNStorage:
    """Fixed-capacity single-row K4/V4 state owned by one live request.

    Packed records and compressed index groups are append-only within a
    lineage. Exact tail rows use deterministic logical-position slots in a
    8,320-row ring. A short undo journal preserves branch rollback without a
    whole-cache copy. Call :meth:`commit` after selecting a public frontier to
    release the journal and invalidate older process-local views.
    """

    def __init__(
        self,
        *,
        max_context_tokens: int,
        position_dtype: mx.Dtype = mx.int32,
        encode_tile: TileEncoder = encode_qsa_kvarn_tile_mlx,
    ) -> None:
        if (
            isinstance(max_context_tokens, bool)
            or not isinstance(max_context_tokens, int)
            or max_context_tokens < QWEN38_QSA_COMPRESS_RATIO
        ):
            raise ValueError(
                f"max_context_tokens must be an integer >= {QWEN38_QSA_COMPRESS_RATIO}"
            )
        if position_dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("Qwen KVarN positions must use an integer dtype")
        if not callable(encode_tile):
            raise TypeError("encode_tile must be callable")

        layout = QWEN38_KVARN_K4V4_G128
        sink_end, body_end, _ = qsa_kvarn_partition(max_context_tokens)
        record_capacity = max(0, body_end - sink_end) // layout.tile_tokens
        index_capacity = max_context_tokens // QWEN38_QSA_COMPRESS_RATIO
        self.max_context_tokens = max_context_tokens
        self.record_capacity = record_capacity
        self.index_capacity = index_capacity
        self.position_dtype = position_dtype
        self._position_itemsize = int(mx.zeros((1,), dtype=position_dtype).nbytes)
        self._encode_tile = encode_tile

        self.packed_records = mx.zeros(
            (record_capacity, layout.kv_heads, layout.head_record_bytes),
            dtype=mx.uint8,
        )
        self.exact_sink_keys = mx.zeros(
            (1, layout.kv_heads, QWEN38_KVARN_EXACT_SINK, layout.head_dim),
            dtype=mx.bfloat16,
        )
        self.exact_sink_values = mx.zeros_like(self.exact_sink_keys)
        self.exact_tail_keys = mx.zeros(
            (
                1,
                layout.kv_heads,
                QWEN38_KVARN_EXACT_TAIL_CAPACITY,
                layout.head_dim,
            ),
            dtype=mx.bfloat16,
        )
        self.exact_tail_values = mx.zeros_like(self.exact_tail_keys)
        self.compressed_index_keys = mx.zeros(
            (1, index_capacity, 128),
            dtype=mx.bfloat16,
        )
        self.compressed_index_positions = mx.zeros(
            (3, 1, index_capacity),
            dtype=position_dtype,
        )
        self.raw_index_keys = mx.zeros(
            (1, _RAW_INDEX_CAPACITY, 128),
            dtype=mx.bfloat16,
        )
        self.raw_index_positions = mx.zeros(
            (3, 1, _RAW_INDEX_CAPACITY),
            dtype=position_dtype,
        )

        self.frontier = 0
        self.body_frontier = 0
        self.record_count = 0
        self.index_group_count = 0
        self.raw_index_count = 0
        self.counters = Qwen4MutableKVarNCounters()
        self._storage_identity = object()
        self._undo_log: list[_UndoMutation] = []
        self._next_mutation_id = 1
        self._lineage = 0
        self._abandoned = False
        self.validate()

    @property
    def sink_count(self) -> int:
        return min(self.frontier, QWEN38_KVARN_EXACT_SINK)

    @property
    def tail_start(self) -> int:
        return self.body_frontier

    @property
    def tail_count(self) -> int:
        return self.frontier - self.body_frontier

    @property
    def allocated_nbytes(self) -> int:
        return sum(int(array.nbytes) for array in self._physical_arrays())

    @property
    def journal_nbytes(self) -> int:
        return sum(entry.nbytes for entry in self._undo_log)

    @property
    def physical_nbytes(self) -> int:
        return self.allocated_nbytes + self.journal_nbytes

    @property
    def logical_nbytes(self) -> int:
        return self._logical_nbytes_at(
            frontier=self.frontier,
            body_frontier=self.body_frontier,
            record_count=self.record_count,
            index_group_count=self.index_group_count,
            raw_index_count=self.raw_index_count,
        )

    def _logical_nbytes_at(
        self,
        *,
        frontier: int,
        body_frontier: int,
        record_count: int,
        index_group_count: int,
        raw_index_count: int,
    ) -> int:
        layout = QWEN38_KVARN_K4V4_G128
        bf16_row_bytes = layout.kv_heads * layout.head_dim * 2
        sink_count = min(frontier, QWEN38_KVARN_EXACT_SINK)
        tail_count = frontier - body_frontier
        return (
            record_count * layout.tile_record_bytes
            + (sink_count + tail_count) * bf16_row_bytes * 2
            + index_group_count * 128 * 2
            + index_group_count * 3 * self._position_itemsize
            + raw_index_count * 128 * 2
            + raw_index_count * 3 * self._position_itemsize
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            **self.counters.as_dict(),
            "frontier": self.frontier,
            "body_frontier": self.body_frontier,
            "tail_start": self.tail_start,
            "tail_count": self.tail_count,
            "record_count": self.record_count,
            "record_capacity": self.record_capacity,
            "index_group_count": self.index_group_count,
            "index_capacity": self.index_capacity,
            "raw_index_count": self.raw_index_count,
            "logical_nbytes": self.logical_nbytes,
            "allocated_nbytes": self.allocated_nbytes,
            "journal_nbytes": self.journal_nbytes,
            "physical_nbytes": self.physical_nbytes,
        }

    def _physical_arrays(self) -> tuple[mx.array, ...]:
        return (
            self.packed_records,
            self.exact_sink_keys,
            self.exact_sink_values,
            self.exact_tail_keys,
            self.exact_tail_values,
            self.compressed_index_keys,
            self.compressed_index_positions,
            self.raw_index_keys,
            self.raw_index_positions,
        )

    def state_arrays(self) -> tuple[mx.array, ...]:
        """Arrays that must finish before a public frontier is published."""
        return self._physical_arrays()

    @property
    def abandoned(self) -> bool:
        """Whether an irreversible request-private mutation was discarded."""

        return self._abandoned

    def abandon(self) -> None:
        """Permanently invalidate request-private storage without rolling it back."""

        if self._abandoned:
            return
        self._abandoned = True
        self.counters.abandon_calls += 1

    def _require_usable(self) -> None:
        if self._abandoned:
            raise RuntimeError("mutable Qwen KVarN storage was abandoned")

    def validate(self) -> None:
        """Fail unless physical storage and logical counters are compatible."""
        self._require_usable()
        layout = QWEN38_KVARN_K4V4_G128
        if (
            self.packed_records.shape
            != (
                self.record_capacity,
                layout.kv_heads,
                layout.head_record_bytes,
            )
            or self.packed_records.dtype != mx.uint8
        ):
            raise ValueError("mutable Qwen KVarN packed storage is incompatible")
        exact_shape = (
            1,
            layout.kv_heads,
            QWEN38_KVARN_EXACT_SINK,
            layout.head_dim,
        )
        tail_shape = (
            1,
            layout.kv_heads,
            QWEN38_KVARN_EXACT_TAIL_CAPACITY,
            layout.head_dim,
        )
        if (
            self.exact_sink_keys.shape != exact_shape
            or self.exact_sink_values.shape != exact_shape
            or self.exact_tail_keys.shape != tail_shape
            or self.exact_tail_values.shape != tail_shape
        ):
            raise ValueError("mutable Qwen KVarN exact storage is incompatible")
        if any(
            array.dtype != mx.bfloat16
            for array in (
                self.exact_sink_keys,
                self.exact_sink_values,
                self.exact_tail_keys,
                self.exact_tail_values,
            )
        ):
            raise ValueError("mutable Qwen KVarN exact storage must use BF16")
        if (
            self.compressed_index_keys.shape
            != (
                1,
                self.index_capacity,
                128,
            )
            or self.compressed_index_keys.dtype != mx.bfloat16
        ):
            raise ValueError("mutable Qwen KVarN index-key storage is incompatible")
        if self.compressed_index_positions.shape != (3, 1, self.index_capacity):
            raise ValueError("mutable Qwen KVarN index-position storage is incompatible")
        if self.compressed_index_positions.dtype != self.position_dtype:
            raise ValueError("mutable Qwen KVarN position dtype is incompatible")
        if self.raw_index_keys.shape != (1, _RAW_INDEX_CAPACITY, 128):
            raise ValueError("mutable Qwen KVarN raw-index storage is incompatible")
        if self.raw_index_keys.dtype != mx.bfloat16:
            raise ValueError("mutable Qwen KVarN raw-index storage must use BF16")
        if self.raw_index_positions.shape != (3, 1, _RAW_INDEX_CAPACITY):
            raise ValueError("mutable Qwen KVarN raw-position storage is incompatible")
        if self.raw_index_positions.dtype != self.position_dtype:
            raise ValueError("mutable Qwen KVarN raw-position dtype is incompatible")
        counters = {
            "frontier": self.frontier,
            "body_frontier": self.body_frontier,
            "record_count": self.record_count,
            "index_group_count": self.index_group_count,
            "raw_index_count": self.raw_index_count,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters.values()
        ):
            raise ValueError(f"mutable Qwen KVarN counters are invalid: {counters}")
        if self.frontier > self.max_context_tokens:
            raise ValueError("mutable Qwen KVarN frontier exceeds capacity")
        sink_end, body_end, _ = qsa_kvarn_partition(self.frontier)
        expected_records = max(0, body_end - sink_end) // layout.tile_tokens
        if self.body_frontier != body_end or self.record_count != expected_records:
            raise ValueError("mutable Qwen KVarN body counters are inconsistent")
        if self.record_count > self.record_capacity:
            raise ValueError("mutable Qwen KVarN packed capacity is exhausted")
        if self.tail_count > QWEN38_KVARN_EXACT_TAIL_CAPACITY:
            raise ValueError("mutable Qwen KVarN tail exceeds its ring")
        if self.index_group_count != self.frontier // QWEN38_QSA_COMPRESS_RATIO:
            raise ValueError("mutable Qwen KVarN index frontier is inconsistent")
        if self.index_group_count > self.index_capacity:
            raise ValueError("mutable Qwen KVarN index capacity is exhausted")
        if self.raw_index_count != self.frontier % QWEN38_QSA_COMPRESS_RATIO:
            raise ValueError("mutable Qwen KVarN raw index frontier is inconsistent")

    def checkpoint(self) -> Qwen4MutableKVarNFrontier:
        """Return a process-local frontier marker without copying cache state."""
        self.validate()
        return self._checkpoint_unchecked()

    def _checkpoint_unchecked(self) -> Qwen4MutableKVarNFrontier:
        cursor = len(self._undo_log)
        anchor = 0 if cursor == 0 else self._undo_log[cursor - 1].mutation_id
        return Qwen4MutableKVarNFrontier(
            frontier=self.frontier,
            body_frontier=self.body_frontier,
            tail_start=self.tail_start,
            tail_count=self.tail_count,
            record_count=self.record_count,
            index_group_count=self.index_group_count,
            raw_index_count=self.raw_index_count,
            mutation_cursor=cursor,
            mutation_anchor=anchor,
            lineage=self._lineage,
            logical_nbytes=self.logical_nbytes,
            _storage_identity=self._storage_identity,
        )

    def preflight_commit(
        self,
        frontier: Qwen4MutableKVarNFrontier,
    ) -> None:
        """Validate an active view before any request storage publishes."""
        self._require_active_frontier(frontier)

    def commit_preflighted(self) -> Qwen4MutableKVarNFrontier:
        """Finalize a previously validated active view without further checks."""
        self._require_usable()
        self._undo_log.clear()
        self._lineage += 1
        self.counters.committed_lineages += 1
        return self._checkpoint_unchecked()

    def validate_frontier(self, frontier: Qwen4MutableKVarNFrontier) -> None:
        """Validate a process-local frontier without changing the active view."""
        self._validate_frontier_identity(frontier)

    def fork(
        self,
        frontier: Qwen4MutableKVarNFrontier | None = None,
    ) -> Qwen4MutableKVarNFrontier:
        """Mark a branch point, restoring the requested live view if supplied."""
        if frontier is not None:
            self.restore(frontier)
        return self.checkpoint()

    def restore(self, frontier: Qwen4MutableKVarNFrontier) -> None:
        """Discard mutations after a compatible process-local frontier."""
        self._validate_frontier_identity(frontier)
        if frontier.mutation_cursor > len(self._undo_log):
            raise ValueError("mutable Qwen KVarN frontier belongs to a discarded branch")
        actual_anchor = (
            0
            if frontier.mutation_cursor == 0
            else self._undo_log[frontier.mutation_cursor - 1].mutation_id
        )
        if frontier.mutation_anchor != actual_anchor:
            raise ValueError("mutable Qwen KVarN frontier belongs to a discarded branch")
        cursor_state = (
            (
                self.frontier,
                self.body_frontier,
                self.record_count,
                self.index_group_count,
                self.raw_index_count,
            )
            if frontier.mutation_cursor == len(self._undo_log)
            else self._undo_state(self._undo_log[frontier.mutation_cursor])
        )
        expected = (
            frontier.frontier,
            frontier.body_frontier,
            frontier.record_count,
            frontier.index_group_count,
            frontier.raw_index_count,
        )
        if cursor_state != expected:
            raise ValueError("mutable Qwen KVarN frontier does not match its mutation cursor")
        while len(self._undo_log) > frontier.mutation_cursor:
            self._restore_undo(self._undo_log.pop())
        observed = (
            self.frontier,
            self.body_frontier,
            self.record_count,
            self.index_group_count,
            self.raw_index_count,
        )
        if observed != expected:
            raise ValueError("mutable Qwen KVarN rollback counters are inconsistent")
        self.counters.restore_calls += 1
        self.validate()

    def discard(self, frontier: Qwen4MutableKVarNFrontier) -> None:
        """Alias for restoring a branch point after rejecting later rows."""
        self.restore(frontier)

    def commit(
        self,
        frontier: Qwen4MutableKVarNFrontier | None = None,
    ) -> Qwen4MutableKVarNFrontier:
        """Make the active frontier the new rollback base and release its journal."""
        selected = self.checkpoint() if frontier is None else frontier
        self.preflight_commit(selected)
        return self.commit_preflighted()

    def _validate_frontier_identity(self, frontier: Qwen4MutableKVarNFrontier) -> None:
        self._require_usable()
        if not isinstance(frontier, Qwen4MutableKVarNFrontier):
            raise TypeError("frontier must be a Qwen4 mutable KVarN frontier")
        if frontier.schema != _SCHEMA:
            raise ValueError("mutable Qwen KVarN frontier schema is incompatible")
        if frontier._storage_identity is not self._storage_identity:
            raise ValueError("mutable Qwen KVarN frontier belongs to another storage")
        if frontier.lineage != self._lineage:
            raise ValueError("mutable Qwen KVarN frontier belongs to an expired lineage")
        integer_fields = {
            "frontier": frontier.frontier,
            "body_frontier": frontier.body_frontier,
            "tail_start": frontier.tail_start,
            "tail_count": frontier.tail_count,
            "record_count": frontier.record_count,
            "index_group_count": frontier.index_group_count,
            "raw_index_count": frontier.raw_index_count,
            "mutation_cursor": frontier.mutation_cursor,
            "mutation_anchor": frontier.mutation_anchor,
            "logical_nbytes": frontier.logical_nbytes,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_fields.values()
        ):
            raise ValueError(f"mutable Qwen KVarN frontier counters are invalid: {integer_fields}")
        sink_end, body_end, _ = qsa_kvarn_partition(frontier.frontier)
        expected_records = max(0, body_end - sink_end) // QWEN38_QSA_TILE_TOKENS
        expected_nbytes = self._logical_nbytes_at(
            frontier=frontier.frontier,
            body_frontier=body_end,
            record_count=expected_records,
            index_group_count=frontier.frontier // QWEN38_QSA_COMPRESS_RATIO,
            raw_index_count=frontier.frontier % QWEN38_QSA_COMPRESS_RATIO,
        )
        if (
            frontier.frontier > self.max_context_tokens
            or frontier.body_frontier != body_end
            or frontier.tail_start != body_end
            or frontier.tail_count != frontier.frontier - body_end
            or frontier.record_count != expected_records
            or frontier.index_group_count != frontier.frontier // QWEN38_QSA_COMPRESS_RATIO
            or frontier.raw_index_count != frontier.frontier % QWEN38_QSA_COMPRESS_RATIO
            or frontier.logical_nbytes != expected_nbytes
        ):
            raise ValueError("mutable Qwen KVarN frontier geometry is incompatible")

    def _require_active_frontier(self, frontier: Qwen4MutableKVarNFrontier) -> None:
        self._validate_frontier_identity(frontier)
        current = self.checkpoint()
        fields = (
            "frontier",
            "body_frontier",
            "record_count",
            "index_group_count",
            "raw_index_count",
            "mutation_cursor",
            "mutation_anchor",
        )
        if any(getattr(frontier, name) != getattr(current, name) for name in fields):
            raise ValueError("mutable Qwen KVarN frontier is not the active view")

    def prepare_index(
        self,
        raw_index_keys: mx.array,
        position_ids: mx.array,
        key_norm: Any,
        *,
        rotary_dim: int,
        rope_base: float,
        mrope_section: tuple[int, int, int],
    ) -> tuple[mx.array, Qwen4KVarNIndexUpdate]:
        """Prepare complete compressed groups without publishing any state."""
        self.validate()
        positions = position_ids
        if positions.ndim == 2:
            positions = mx.broadcast_to(positions[None], (3, *positions.shape))
        raw_tail = self.raw_index_keys[:, : self.raw_index_count]
        raw_tail_positions = self.raw_index_positions[:, :, : self.raw_index_count]
        return prepare_qsa_kvarn_index_storage(
            old_frontier=self.frontier,
            old_group_count=self.index_group_count,
            storage_keys=self.compressed_index_keys,
            storage_positions=self.compressed_index_positions,
            raw_index_tail=raw_tail,
            raw_index_tail_positions=raw_tail_positions,
            raw_index_keys=raw_index_keys,
            position_ids=positions,
            key_norm=key_norm,
            rotary_dim=rotary_dim,
            rope_base=rope_base,
            mrope_section=mrope_section,
        )

    def append(
        self,
        keys: mx.array,
        values: mx.array,
        valid_tokens: mx.array,
        *,
        index_update: Qwen4KVarNIndexUpdate,
        undo_reservation: _Qwen4MutableUndoReservation | None = None,
    ) -> Qwen4MutableKVarNFrontier:
        """Append K/V and prepared index rows with bounded indexed writes."""

        frontier, predicate = self._append_impl(
            keys,
            values,
            valid_tokens,
            index_update=index_update,
            undo_reservation=undo_reservation,
            defer_finite=False,
            irreversible=False,
        )
        if predicate is not None:
            raise AssertionError("ordinary mutable append returned a deferred predicate")
        return frontier

    def _append_with_deferred_finite(
        self,
        keys: mx.array,
        values: mx.array,
        valid_tokens: mx.array,
        *,
        index_update: Qwen4KVarNIndexUpdate,
        undo_reservation: _Qwen4MutableUndoReservation,
        capability: object,
    ) -> tuple[Qwen4MutableKVarNFrontier, mx.array]:
        """Append under shell-owned rollback and return the lazy finite predicate."""

        if capability is not _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY:
            raise ValueError("Qwen KVarN append finite capability is invalid")
        frontier, predicate = self._append_impl(
            keys,
            values,
            valid_tokens,
            index_update=index_update,
            undo_reservation=undo_reservation,
            defer_finite=True,
            irreversible=False,
        )
        if predicate is None:
            raise AssertionError("deferred mutable append did not return its predicate")
        return frontier, predicate

    def _append_irreversible_deferred_finite(
        self,
        keys: mx.array,
        values: mx.array,
        valid_tokens: mx.array,
        *,
        index_update: Qwen4KVarNIndexUpdate,
        capability: object,
    ) -> tuple[Qwen4MutableKVarNFrontier, mx.array]:
        """Append one private decode row without creating rollback state.

        The caller must evaluate the returned predicate together with every
        state array before publishing the frontier. A false predicate requires
        :meth:`abandon`; the previous frontier cannot be restored.
        """

        if capability is not _QWEN4_SERIAL_IRREVOCABLE_APPEND_CAPABILITY:
            raise ValueError("Qwen KVarN irreversible append capability is invalid")
        frontier, predicate = self._append_impl(
            keys,
            values,
            valid_tokens,
            index_update=index_update,
            undo_reservation=None,
            defer_finite=True,
            irreversible=True,
        )
        if predicate is None:
            raise AssertionError("irreversible mutable append did not return its predicate")
        return frontier, predicate

    def _append_impl(
        self,
        keys: mx.array,
        values: mx.array,
        valid_tokens: mx.array,
        *,
        index_update: Qwen4KVarNIndexUpdate,
        undo_reservation: _Qwen4MutableUndoReservation | None,
        defer_finite: bool,
        irreversible: bool,
    ) -> tuple[Qwen4MutableKVarNFrontier, mx.array | None]:
        """Validate and mutate storage, optionally deferring only the finite read."""

        self.validate()
        layout = QWEN38_KVARN_K4V4_G128
        if (
            keys.ndim != 4
            or keys.shape[:2] != (1, layout.kv_heads)
            or keys.shape[-1] != layout.head_dim
            or keys.shape[2] <= 0
        ):
            raise ValueError("keys must have shape [1, 2, tokens, 256]")
        if values.shape != keys.shape:
            raise ValueError("values must match keys")
        if keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16:
            raise ValueError("mutable Qwen KVarN K/V must use BF16")
        new_tokens = int(keys.shape[2])
        if valid_tokens.shape != (1, new_tokens) or valid_tokens.dtype != mx.bool_:
            raise ValueError("valid_tokens must match the single-row K/V chunk")
        if not isinstance(index_update, Qwen4KVarNIndexUpdate):
            raise TypeError("index_update must be a Qwen4KVarNIndexUpdate")
        if (
            index_update.old_frontier != self.frontier
            or index_update.new_tokens != new_tokens
            or index_update.old_group_count != self.index_group_count
        ):
            raise ValueError("Qwen KVarN index update is off the mutable frontier")
        if (
            index_update.storage_keys is not self.compressed_index_keys
            or index_update.storage_positions is not self.compressed_index_positions
        ):
            raise ValueError("Qwen KVarN index update targets another storage")
        next_frontier = self.frontier + new_tokens
        if next_frontier > self.max_context_tokens:
            raise ValueError(
                f"Qwen KVarN context capacity {self.max_context_tokens} token(s) "
                f"is exhausted at frontier {next_frontier}"
            )
        next_groups = self.index_group_count + int(index_update.new_group_keys.shape[1])
        next_raw_count = int(index_update.next_raw_tail.shape[1])
        if (
            index_update.new_group_keys.ndim != 3
            or index_update.new_group_keys.shape[0] != 1
            or index_update.new_group_keys.shape[2] != 128
            or index_update.new_group_keys.dtype != mx.bfloat16
            or index_update.new_group_positions.shape
            != (3, 1, index_update.new_group_keys.shape[1])
            or index_update.new_group_positions.dtype != self.position_dtype
            or next_groups != next_frontier // QWEN38_QSA_COMPRESS_RATIO
            or next_raw_count != next_frontier % QWEN38_QSA_COMPRESS_RATIO
            or index_update.next_raw_tail.shape != (1, next_raw_count, 128)
            or index_update.next_raw_tail.dtype != mx.bfloat16
            or index_update.next_raw_tail_positions.shape != (3, 1, next_raw_count)
            or index_update.next_raw_tail_positions.dtype != self.position_dtype
        ):
            raise ValueError("Qwen KVarN prepared index update is inconsistent")
        if next_groups > self.index_capacity:
            raise ValueError("Qwen KVarN compressed index capacity is exhausted")
        _, next_body, _ = qsa_kvarn_partition(next_frontier)
        next_records = max(0, next_body - QWEN38_KVARN_EXACT_SINK) // layout.tile_tokens
        if next_records > self.record_capacity:
            raise ValueError("Qwen KVarN packed-record capacity is exhausted")
        self.counters.append_value_checks += 1
        finite_and_full = mx.all(valid_tokens)
        for array in (
            keys,
            values,
            index_update.new_group_keys,
            index_update.next_raw_tail,
        ):
            finite_and_full = finite_and_full & mx.all(mx.isfinite(array))
        if irreversible and (not defer_finite or undo_reservation is not None):
            raise ValueError("irreversible Qwen KVarN append has an invalid transaction mode")
        if defer_finite:
            if new_tokens != 1 or (undo_reservation is None and not irreversible):
                raise ValueError("deferred Qwen KVarN finite validation is decode-only")
            self.counters.deferred_append_value_checks += 1
        else:
            self.counters.local_append_value_checks += 1
            if not bool(finite_and_full.item()):
                raise ValueError("mutable Qwen KVarN requires finite unpadded input")

        prior_frontier = self.frontier
        prior_record_count = self.record_count
        prior_index_group_count = self.index_group_count
        undo = None
        if not irreversible:
            undo = (
                self._capture_undo(new_tokens)
                if undo_reservation is None
                else self._consume_undo_reservation(
                    undo_reservation,
                    new_tokens=new_tokens,
                )
            )
        mutation_started = False
        try:
            cursor = 0
            working_frontier = self.frontier
            working_records = self.record_count
            while cursor < new_tokens:
                next_seal = (
                    QWEN38_KVARN_EXACT_SINK
                    + QWEN38_KVARN_EXACT_TAIL_CAPACITY
                    + working_records * QWEN38_QSA_TILE_TOKENS
                )
                width = min(new_tokens - cursor, next_seal - working_frontier)
                if width <= 0:
                    raise ValueError("mutable Qwen KVarN seal frontier is inconsistent")
                end = cursor + width
                mutation_started = True
                self._write_exact_rows(
                    working_frontier,
                    keys[..., cursor:end, :],
                    values[..., cursor:end, :],
                )
                working_frontier += width
                cursor = end
                _, working_body, _ = qsa_kvarn_partition(working_frontier)
                expected_records = (
                    max(0, working_body - QWEN38_KVARN_EXACT_SINK) // QWEN38_QSA_TILE_TOKENS
                )
                if expected_records == working_records + 1:
                    tile_start = QWEN38_KVARN_EXACT_SINK + working_records * QWEN38_QSA_TILE_TOKENS
                    slot = self._tail_slot(tile_start)
                    if slot + QWEN38_QSA_TILE_TOKENS > QWEN38_KVARN_EXACT_TAIL_CAPACITY:
                        raise ValueError("mutable Qwen KVarN tile crosses the tail-ring edge")
                    tile_keys = self.exact_tail_keys[
                        0,
                        :,
                        slot : slot + QWEN38_QSA_TILE_TOKENS,
                        :,
                    ].transpose(1, 0, 2)
                    tile_values = self.exact_tail_values[
                        0,
                        :,
                        slot : slot + QWEN38_QSA_TILE_TOKENS,
                        :,
                    ].transpose(1, 0, 2)
                    record = self._encode_tile(tile_keys, tile_values)
                    if (
                        record.shape != (layout.kv_heads, layout.head_record_bytes)
                        or record.dtype != mx.uint8
                    ):
                        raise ValueError("KVarN tile encoder returned an incompatible record")
                    mx.eval(record)
                    self.packed_records[working_records] = record
                    working_records += 1
                elif expected_records != working_records:
                    raise ValueError("mutable Qwen KVarN append crossed an invalid seal count")

            if index_update.new_group_keys.shape[1]:
                start = self.index_group_count
                self.compressed_index_keys[:, start:next_groups] = index_update.new_group_keys
                self.compressed_index_positions[:, :, start:next_groups] = (
                    index_update.new_group_positions
                )
            if next_raw_count:
                self.raw_index_keys[:, :next_raw_count] = index_update.next_raw_tail
                self.raw_index_positions[:, :, :next_raw_count] = (
                    index_update.next_raw_tail_positions
                )
            if next_raw_count < _RAW_INDEX_CAPACITY:
                self.raw_index_keys[:, next_raw_count:] = 0
                self.raw_index_positions[:, :, next_raw_count:] = 0

            self.frontier = next_frontier
            self.body_frontier = next_body
            self.record_count = next_records
            self.index_group_count = next_groups
            self.raw_index_count = next_raw_count
            if undo is not None:
                self._undo_log.append(undo)
            self.validate()
            self.counters.append_calls += 1
            self.counters.appended_tokens += new_tokens
            self.counters.indexed_kv_write_calls += 1
            sink_rows = max(
                0,
                min(next_frontier, QWEN38_KVARN_EXACT_SINK)
                - min(
                    prior_frontier,
                    QWEN38_KVARN_EXACT_SINK,
                ),
            )
            self.counters.sink_rows_written += sink_rows
            self.counters.tail_rows_written += new_tokens - sink_rows
            self.counters.tile_records_sealed += next_records - prior_record_count
            self.counters.index_groups_sealed += next_groups - prior_index_group_count
            if irreversible:
                self.counters.irreversible_append_calls += 1
                self.counters.irreversible_appended_tokens += new_tokens
            return self._checkpoint_unchecked(), finite_and_full if defer_finite else None
        except BaseException:
            if undo is not None:
                if self._undo_log and self._undo_log[-1] is undo:
                    self._undo_log.pop()
                self._restore_undo(undo)
            elif mutation_started:
                self.counters.post_mutation_failures += 1
                self.abandon()
            raise

    def gather_selected_rows(
        self,
        selected_indices: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Gather selected rows across sink, packed body, and wrapped tail."""
        self.validate()
        if selected_indices.ndim != 3 or selected_indices.shape[0] != 1:
            raise ValueError("Qwen KVarN selected rows require shape [1, queries, width]")
        normalized, valid = qsa_normalize_selected_rows(
            selected_indices,
            self.frontier,
        )
        result = self._gather_normalized_rows(normalized, valid)
        self.counters.gather_calls += 1
        self.counters.gather_lanes += int(selected_indices.size)
        return result

    def gather_selected_rows_with_pending(
        self,
        pending_keys: mx.array,
        pending_values: mx.array,
        selected_indices: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Gather live state plus exact rows that have not been published."""
        return self._gather_selected_rows_with_pending(
            pending_keys,
            pending_values,
            selected_indices,
            check_finite=True,
        )

    def _gather_selected_rows_with_pending_deferred_finite(
        self,
        pending_keys: mx.array,
        pending_values: mx.array,
        selected_indices: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Gather while deferring pending-row validation to mutable append."""
        return self._gather_selected_rows_with_pending(
            pending_keys,
            pending_values,
            selected_indices,
            check_finite=False,
        )

    def _gather_selected_rows_with_pending(
        self,
        pending_keys: mx.array,
        pending_values: mx.array,
        selected_indices: mx.array,
        *,
        check_finite: bool,
    ) -> tuple[mx.array, mx.array, mx.array]:
        self.validate()
        next_frontier = self._validate_pending_geometry(pending_keys, pending_values)
        if selected_indices.ndim != 3 or selected_indices.shape[0] != 1:
            raise ValueError("Qwen KVarN selected rows require shape [1, queries, width]")
        if check_finite:
            self._validate_pending_values(
                pending_keys,
                pending_values,
                next_frontier=next_frontier,
            )
        else:
            if next_frontier > self.max_context_tokens:
                raise ValueError("pending K/V rows exceed mutable Qwen KVarN capacity")
            self.counters.deferred_pending_value_checks += 1
        normalized, valid = qsa_normalize_selected_rows(selected_indices, next_frontier)
        fused = bool(
            not check_finite
            and pending_keys.shape[2] == 1
            and self.frontier >= QWEN38_KVARN_EXACT_SINK
            and self.record_capacity > 0
        )
        if fused:
            gathered_keys, gathered_values = gather_qsa_kvarn_mutable_rows_metal(
                self.packed_records[: max(1, self.record_count)],
                self.exact_sink_keys,
                self.exact_sink_values,
                self.exact_tail_keys,
                self.exact_tail_values,
                pending_keys,
                pending_values,
                normalized,
                valid,
                frontier=self.frontier,
                record_count=self.record_count,
            )
            self.counters.fused_mutable_gather_calls += 1
            self.counters.fused_mutable_gather_lanes += int(selected_indices.size)
        else:
            gathered_keys, gathered_values = self._gather_normalized_rows_with_pending(
                pending_keys,
                pending_values,
                normalized,
                valid,
            )
        self.counters.gather_calls += 1
        self.counters.pending_gather_calls += 1
        self.counters.gather_lanes += int(selected_indices.size)
        return gathered_keys, gathered_values, valid

    def materialize_rows_with_pending(
        self,
        pending_keys: mx.array,
        pending_values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Materialize one logical BF16 K/V view for a multi-query segment.

        Packed body rows are reconstructed once. The exact sink, logical tail,
        and pending rows retain their physical token order in the returned
        ``[batch, kv_heads, tokens, width]`` arrays.
        """
        self.validate()
        next_frontier = self._validate_pending_geometry(pending_keys, pending_values)
        self._validate_pending_values(
            pending_keys,
            pending_values,
            next_frontier=next_frontier,
        )
        normalized = mx.arange(next_frontier, dtype=mx.int32).reshape(
            1,
            1,
            next_frontier,
        )
        valid = mx.ones(normalized.shape, dtype=mx.bool_)
        gathered_keys, gathered_values = self._gather_normalized_rows_with_pending(
            pending_keys,
            pending_values,
            normalized,
            valid,
        )
        self.counters.prefill_materialize_calls += 1
        self.counters.prefill_materialize_lanes += next_frontier
        return gathered_keys[:, 0], gathered_values[:, 0]

    def _validate_pending_geometry(
        self,
        pending_keys: mx.array,
        pending_values: mx.array,
    ) -> int:
        layout = QWEN38_KVARN_K4V4_G128
        if (
            pending_keys.ndim != 4
            or pending_keys.shape[:2] != (1, layout.kv_heads)
            or pending_keys.shape[-1] != layout.head_dim
            or pending_keys.shape[2] <= 0
        ):
            raise ValueError("pending keys must have shape [1, 2, tokens, 256]")
        if pending_values.shape != pending_keys.shape:
            raise ValueError("pending values must match pending keys")
        if pending_keys.dtype != mx.bfloat16 or pending_values.dtype != mx.bfloat16:
            raise ValueError("pending K/V rows must use BF16")
        return self.frontier + int(pending_keys.shape[2])

    def _validate_pending_values(
        self,
        pending_keys: mx.array,
        pending_values: mx.array,
        *,
        next_frontier: int,
    ) -> None:
        self.counters.pending_value_checks += 1
        finite = mx.all(mx.isfinite(pending_keys)) & mx.all(mx.isfinite(pending_values))
        if not bool(finite.item()):
            raise ValueError("pending K/V rows must be finite")
        if next_frontier > self.max_context_tokens:
            raise ValueError("pending K/V rows exceed mutable Qwen KVarN capacity")

    def _gather_normalized_rows_with_pending(
        self,
        pending_keys: mx.array,
        pending_values: mx.array,
        normalized: mx.array,
        valid: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if normalized.ndim != 3 or normalized.shape[0] != 1:
            raise ValueError("normalized Qwen KVarN rows require shape [1, queries, width]")
        if normalized.dtype != mx.int32:
            raise ValueError("normalized Qwen KVarN rows must contain int32 values")
        if valid.shape != normalized.shape or valid.dtype != mx.bool_:
            raise ValueError("normalized Qwen KVarN validity must match selected rows")
        layout = QWEN38_KVARN_K4V4_G128
        safe = mx.where(valid, normalized, 0)
        prior_valid = valid & (safe < self.frontier)
        pending_valid = valid & (safe >= self.frontier)
        prior_keys, prior_values, _ = self._gather_normalized_rows(
            mx.where(prior_valid, safe, 0),
            prior_valid,
        )
        pending_ids = mx.where(pending_valid, safe - self.frontier, 0).reshape(-1)
        token_major_keys = pending_keys[0].transpose(1, 0, 2)
        token_major_values = pending_values[0].transpose(1, 0, 2)
        exact_keys = mx.take(token_major_keys, pending_ids, axis=0)
        exact_values = mx.take(token_major_values, pending_ids, axis=0)
        batch_size, query_count, selected_width = valid.shape
        gathered_shape = (
            batch_size,
            query_count,
            selected_width,
            layout.kv_heads,
            layout.head_dim,
        )
        exact_keys = exact_keys.reshape(gathered_shape).transpose(0, 1, 3, 2, 4)
        exact_values = exact_values.reshape(gathered_shape).transpose(0, 1, 3, 2, 4)
        pending_mask = pending_valid[:, :, None, :, None]
        gathered_keys = prior_keys + mx.where(pending_mask, exact_keys, 0)
        gathered_values = prior_values + mx.where(pending_mask, exact_values, 0)
        return gathered_keys, gathered_values

    def _gather_normalized_rows(
        self,
        normalized: mx.array,
        valid: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if normalized.ndim != 3 or normalized.shape[0] != 1 or normalized.dtype != mx.int32:
            raise ValueError("normalized Qwen KVarN rows require int32 [1, queries, width]")
        if valid.shape != normalized.shape or valid.dtype != mx.bool_:
            raise ValueError("normalized Qwen KVarN validity must match selected rows")
        layout = QWEN38_KVARN_K4V4_G128
        safe = mx.where(valid, normalized, 0)
        sink_mask = valid & (safe < self.sink_count)
        body_mask = valid & (safe >= QWEN38_KVARN_EXACT_SINK) & (safe < self.body_frontier)
        tail_mask = valid & (safe >= self.body_frontier)

        def exact_rows(array: mx.array, ids: mx.array) -> mx.array:
            token_major = array[0].transpose(1, 0, 2)
            return mx.take(token_major, ids.reshape(-1), axis=0)

        sink_ids = mx.where(sink_mask, safe, 0)
        sink_keys = exact_rows(self.exact_sink_keys, sink_ids)
        sink_values = exact_rows(self.exact_sink_values, sink_ids)

        body_local = safe - QWEN38_KVARN_EXACT_SINK
        tile_ids = (
            mx.where(
                body_mask,
                body_local // QWEN38_QSA_TILE_TOKENS,
                -1,
            )
            .reshape(-1)
            .astype(mx.int32)
        )
        tile_offsets = (
            mx.where(
                body_mask,
                body_local % QWEN38_QSA_TILE_TOKENS,
                0,
            )
            .reshape(-1)
            .astype(mx.int32)
        )
        body_keys, body_values, _ = _decode_qsa_kvarn_rows_metal(
            self.packed_records[: self.record_count],
            tile_ids,
            tile_offsets,
        )

        tail_slots = mx.where(
            tail_mask,
            (safe - QWEN38_KVARN_EXACT_SINK) % QWEN38_KVARN_EXACT_TAIL_CAPACITY,
            0,
        )
        tail_keys = exact_rows(self.exact_tail_keys, tail_slots)
        tail_values = exact_rows(self.exact_tail_values, tail_slots)

        flat_shape = (*safe.reshape(-1).shape, 1, 1)
        gathered_keys = mx.where(sink_mask.reshape(flat_shape), sink_keys, 0)
        gathered_values = mx.where(sink_mask.reshape(flat_shape), sink_values, 0)
        gathered_keys = gathered_keys + mx.where(body_mask.reshape(flat_shape), body_keys, 0)
        gathered_values = gathered_values + mx.where(
            body_mask.reshape(flat_shape),
            body_values,
            0,
        )
        gathered_keys = gathered_keys + mx.where(tail_mask.reshape(flat_shape), tail_keys, 0)
        gathered_values = gathered_values + mx.where(
            tail_mask.reshape(flat_shape),
            tail_values,
            0,
        )
        batch_size, query_count, selected_width = valid.shape
        provider_shape = (
            batch_size,
            query_count,
            selected_width,
            layout.kv_heads,
            layout.head_dim,
        )
        gathered_keys = gathered_keys.reshape(provider_shape).transpose(0, 1, 3, 2, 4)
        gathered_values = gathered_values.reshape(provider_shape).transpose(0, 1, 3, 2, 4)
        return gathered_keys, gathered_values, valid

    def _build_undo(self, new_tokens: int) -> _UndoMutation:
        sink_ranges = self._save_ranges(
            self.exact_sink_keys,
            self.exact_sink_values,
            self._sink_ranges(self.frontier, new_tokens),
        )
        tail_ranges = self._save_ranges(
            self.exact_tail_keys,
            self.exact_tail_values,
            self._tail_ranges(self.frontier, new_tokens),
        )
        raw_keys = self.raw_index_keys + mx.zeros_like(self.raw_index_keys)
        raw_positions = self.raw_index_positions + mx.zeros_like(self.raw_index_positions)
        return _UndoMutation(
            mutation_id=self._next_mutation_id,
            frontier=self.frontier,
            body_frontier=self.body_frontier,
            record_count=self.record_count,
            index_group_count=self.index_group_count,
            raw_index_count=self.raw_index_count,
            raw_index_keys=raw_keys,
            raw_index_positions=raw_positions,
            sink_ranges=sink_ranges,
            tail_ranges=tail_ranges,
        )

    @staticmethod
    def _undo_arrays(mutation: _UndoMutation) -> tuple[mx.array, ...]:
        arrays = [mutation.raw_index_keys, mutation.raw_index_positions]
        for saved in (*mutation.sink_ranges, *mutation.tail_ranges):
            arrays.extend((saved.keys, saved.values))
        return tuple(arrays)

    def _capture_undo(self, new_tokens: int) -> _UndoMutation:
        mutation = self._build_undo(new_tokens)
        mx.eval(*self._undo_arrays(mutation))
        self.counters.local_undo_evals += 1
        self._next_mutation_id += 1
        return mutation

    def _prepare_undo_reservation(
        self,
        new_tokens: int,
    ) -> _Qwen4MutableUndoReservation:
        """Prepare lazy rollback copies without changing mutable state."""

        if isinstance(new_tokens, bool) or not isinstance(new_tokens, int) or new_tokens <= 0:
            raise ValueError("undo reservation token count must be a positive integer")
        self.validate()
        mutation = self._build_undo(new_tokens)
        reservation = _Qwen4MutableUndoReservation(
            storage=self,
            storage_identity=self._storage_identity,
            lineage=self._lineage,
            journal_length=len(self._undo_log),
            new_tokens=new_tokens,
            mutation=mutation,
        )
        self.counters.undo_reservations_prepared += 1
        return reservation

    def _undo_reservation_arrays(
        self,
        reservation: _Qwen4MutableUndoReservation,
    ) -> tuple[mx.array, ...]:
        """Return lazy arrays belonging to one live reservation."""

        self._validate_undo_reservation(reservation, require_ready=False)
        return self._undo_arrays(reservation.mutation)

    def _mark_undo_reservation_evaluated(
        self,
        reservation: _Qwen4MutableUndoReservation,
    ) -> None:
        """Make a shell-evaluated reservation eligible for append."""

        self._validate_undo_reservation(reservation, require_ready=False)
        if reservation.evaluated or reservation.dependency_bound:
            raise ValueError("mutable Qwen KVarN undo reservation was already evaluated")
        reservation.evaluated = True
        self.counters.undo_reservations_evaluated += 1

    def _bind_undo_reservation_dependencies(
        self,
        reservation: _Qwen4MutableUndoReservation,
        inputs: tuple[mx.array, ...],
    ) -> tuple[mx.array, ...]:
        """Order one future append after its lazy rollback copies."""

        self._validate_undo_reservation(reservation, require_ready=False)
        if reservation.evaluated or reservation.dependency_bound:
            raise ValueError("mutable Qwen KVarN undo reservation is already prepared")
        if not inputs or any(not isinstance(value, mx.array) for value in inputs):
            raise TypeError("mutable Qwen KVarN undo dependencies require array inputs")
        dependencies = self._undo_arrays(reservation.mutation)
        bound = mx.depends(inputs, dependencies)
        reservation.dependency_bound = True
        self.counters.undo_reservations_dependency_bound += 1
        return tuple(bound)

    def _consume_undo_reservation(
        self,
        reservation: _Qwen4MutableUndoReservation,
        *,
        new_tokens: int,
    ) -> _UndoMutation:
        """Consume one ready reservation at its exact source frontier."""

        self._validate_undo_reservation(reservation, require_ready=True)
        if reservation.new_tokens != new_tokens:
            raise ValueError("mutable Qwen KVarN undo reservation has the wrong width")
        reservation.consumed = True
        self._next_mutation_id += 1
        self.counters.undo_reservations_consumed += 1
        return reservation.mutation

    def _validate_undo_reservation(
        self,
        reservation: _Qwen4MutableUndoReservation,
        *,
        require_ready: bool,
    ) -> None:
        if not isinstance(reservation, _Qwen4MutableUndoReservation):
            raise TypeError("undo reservation is incompatible with mutable Qwen KVarN")
        if reservation.consumed:
            raise ValueError("mutable Qwen KVarN undo reservation was already consumed")
        mutation = reservation.mutation
        if (
            reservation.storage is not self
            or reservation.storage_identity is not self._storage_identity
            or reservation.lineage != self._lineage
            or reservation.journal_length != len(self._undo_log)
            or mutation.mutation_id != self._next_mutation_id
            or mutation.frontier != self.frontier
            or mutation.body_frontier != self.body_frontier
            or mutation.record_count != self.record_count
            or mutation.index_group_count != self.index_group_count
            or mutation.raw_index_count != self.raw_index_count
        ):
            raise ValueError("mutable Qwen KVarN undo reservation is stale")
        if require_ready and not (reservation.evaluated or reservation.dependency_bound):
            raise ValueError(
                "mutable Qwen KVarN undo reservation was not evaluated or dependency-bound"
            )

    @staticmethod
    def _save_ranges(
        keys: mx.array,
        values: mx.array,
        ranges: tuple[tuple[int, int], ...],
    ) -> tuple[_SavedRange, ...]:
        saved = []
        for start, end in ranges:
            saved_keys = keys[..., start:end, :] + mx.zeros_like(keys[..., start:end, :])
            saved_values = values[..., start:end, :] + mx.zeros_like(values[..., start:end, :])
            saved.append(_SavedRange(start=start, keys=saved_keys, values=saved_values))
        return tuple(saved)

    @staticmethod
    def _sink_ranges(frontier: int, new_tokens: int) -> tuple[tuple[int, int], ...]:
        start = min(frontier, QWEN38_KVARN_EXACT_SINK)
        end = min(frontier + new_tokens, QWEN38_KVARN_EXACT_SINK)
        return () if end <= start else ((start, end),)

    @staticmethod
    def _tail_ranges(frontier: int, new_tokens: int) -> tuple[tuple[int, int], ...]:
        logical_start = max(frontier, QWEN38_KVARN_EXACT_SINK)
        logical_end = frontier + new_tokens
        if logical_end <= logical_start:
            return ()
        count = logical_end - logical_start
        if count >= QWEN38_KVARN_EXACT_TAIL_CAPACITY:
            return ((0, QWEN38_KVARN_EXACT_TAIL_CAPACITY),)
        slot = (logical_start - QWEN38_KVARN_EXACT_SINK) % QWEN38_KVARN_EXACT_TAIL_CAPACITY
        first = min(count, QWEN38_KVARN_EXACT_TAIL_CAPACITY - slot)
        result = [(slot, slot + first)]
        if first < count:
            result.append((0, count - first))
        return tuple(result)

    def _restore_undo(self, undo: _UndoMutation) -> None:
        for saved in undo.sink_ranges:
            end = saved.start + saved.keys.shape[2]
            self.exact_sink_keys[..., saved.start : end, :] = saved.keys
            self.exact_sink_values[..., saved.start : end, :] = saved.values
        for saved in undo.tail_ranges:
            end = saved.start + saved.keys.shape[2]
            self.exact_tail_keys[..., saved.start : end, :] = saved.keys
            self.exact_tail_values[..., saved.start : end, :] = saved.values
        self.raw_index_keys[:] = undo.raw_index_keys
        self.raw_index_positions[:] = undo.raw_index_positions
        self.frontier = undo.frontier
        self.body_frontier = undo.body_frontier
        self.record_count = undo.record_count
        self.index_group_count = undo.index_group_count
        self.raw_index_count = undo.raw_index_count

    @staticmethod
    def _undo_state(undo: _UndoMutation) -> tuple[int, int, int, int, int]:
        return (
            undo.frontier,
            undo.body_frontier,
            undo.record_count,
            undo.index_group_count,
            undo.raw_index_count,
        )

    def _write_exact_rows(
        self,
        logical_start: int,
        keys: mx.array,
        values: mx.array,
    ) -> None:
        token_count = int(keys.shape[2])
        cursor = 0
        if logical_start < QWEN38_KVARN_EXACT_SINK:
            sink_take = min(
                token_count,
                QWEN38_KVARN_EXACT_SINK - logical_start,
            )
            self.exact_sink_keys[..., logical_start : logical_start + sink_take, :] = keys[
                ..., :sink_take, :
            ]
            self.exact_sink_values[..., logical_start : logical_start + sink_take, :] = values[
                ..., :sink_take, :
            ]
            logical_start += sink_take
            cursor += sink_take
        while cursor < token_count:
            slot = self._tail_slot(logical_start)
            width = min(
                token_count - cursor,
                QWEN38_KVARN_EXACT_TAIL_CAPACITY - slot,
            )
            end = cursor + width
            self.exact_tail_keys[..., slot : slot + width, :] = keys[..., cursor:end, :]
            self.exact_tail_values[..., slot : slot + width, :] = values[..., cursor:end, :]
            logical_start += width
            cursor = end

    @staticmethod
    def _tail_slot(logical_position: int) -> int:
        if logical_position < QWEN38_KVARN_EXACT_SINK:
            raise ValueError("tail position is inside the exact sink")
        return (logical_position - QWEN38_KVARN_EXACT_SINK) % QWEN38_KVARN_EXACT_TAIL_CAPACITY
