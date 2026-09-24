"""K4/V4 state for Qwen sparse attention.

The state keeps the first 128 rows and a suffix of at least 8,192 rows exact.
Only committed complete 128-token tiles between those regions are encoded.
The state is single-row and does not implement trimming, prefix reuse, or disk
serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx

from moespresso.runtime.qwen4.kvarn_encode import encode_qsa_kvarn_tile_mlx
from moespresso.runtime.qwen4.kvarn_layout import QWEN38_KVARN_K4V4_G128
from moespresso.runtime.qwen4.kvarn_selected_rows import _decode_qsa_kvarn_rows_metal
from moespresso.runtime.qwen4.qsa import (
    QWEN38_QSA_COMPRESS_RATIO,
    qsa_normalize_selected_rows,
    qwen4_partial_rope,
)


QWEN38_KVARN_EXACT_SINK = 128
QWEN38_KVARN_EXACT_SUFFIX = 8_192
QWEN38_QSA_TILE_TOKENS = QWEN38_KVARN_K4V4_G128.tile_tokens
_SCHEMA = "qwen38_qsa_kvarn_k4v4_g128_state_v2"

TileEncoder = Callable[[mx.array, mx.array], mx.array]


@dataclass(frozen=True)
class Qwen4KVarNQSAState:
    """Committed packed-body and exact-boundary state at one frontier."""

    packed_records: mx.array
    exact_sink_keys: mx.array
    exact_sink_values: mx.array
    exact_tail_keys: mx.array
    exact_tail_values: mx.array
    compressed_index_keys: mx.array
    compressed_index_positions: mx.array
    index_group_count: int
    raw_index_tail: mx.array
    raw_index_tail_positions: mx.array
    frontier: int
    body_frontier: int
    schema: str = _SCHEMA

    @property
    def nbytes(self) -> int:
        return sum(
            int(array.nbytes)
            for array in (
                self.packed_records,
                self.exact_sink_keys,
                self.exact_sink_values,
                self.exact_tail_keys,
                self.exact_tail_values,
                self.compressed_index_keys,
                self.compressed_index_positions,
                self.raw_index_tail,
                self.raw_index_tail_positions,
            )
        )

    @property
    def index_stats(self) -> dict[str, int]:
        """Return inspectable logical occupancy without reading array payloads."""
        return {
            "capacity_groups": int(self.compressed_index_keys.shape[1]),
            "sealed_groups": self.index_group_count,
            "retained_raw_rows": int(self.raw_index_tail.shape[1]),
        }


@dataclass(frozen=True)
class Qwen4KVarNIndexUpdate:
    """Prepared compressed groups published only with the attention segment."""

    old_frontier: int
    new_tokens: int
    storage_keys: mx.array
    storage_positions: mx.array
    old_group_count: int
    new_group_keys: mx.array
    new_group_positions: mx.array
    next_raw_tail: mx.array
    next_raw_tail_positions: mx.array


def qsa_kvarn_partition(frontier: int) -> tuple[int, int, int]:
    """Return exact-sink end, packed-body end, and exact-tail start."""
    if isinstance(frontier, bool) or not isinstance(frontier, int) or frontier < 0:
        raise ValueError("frontier must be a nonnegative integer")
    sink_end = min(frontier, QWEN38_KVARN_EXACT_SINK)
    if frontier <= QWEN38_KVARN_EXACT_SINK:
        body_end = frontier
    else:
        eligible = max(
            0,
            frontier - QWEN38_KVARN_EXACT_SINK - QWEN38_KVARN_EXACT_SUFFIX,
        )
        body_end = (
            QWEN38_KVARN_EXACT_SINK + (eligible // QWEN38_QSA_TILE_TOKENS) * QWEN38_QSA_TILE_TOKENS
        )
    return sink_end, body_end, body_end


def _record_count(frontier: int) -> int:
    sink_end, body_end, _ = qsa_kvarn_partition(frontier)
    return max(0, body_end - sink_end) // QWEN38_QSA_TILE_TOKENS


def validate_qsa_kvarn_state(state: Qwen4KVarNQSAState | None) -> None:
    """Validate the K4/V4 state structure."""
    if state is None:
        return
    if not isinstance(state, Qwen4KVarNQSAState) or state.schema != _SCHEMA:
        raise ValueError("Qwen KVarN state schema is incompatible")
    if (
        isinstance(state.frontier, bool)
        or not isinstance(state.frontier, int)
        or state.frontier <= 0
    ):
        raise ValueError("Qwen KVarN state frontier must be a positive integer")
    sink_end, body_end, _ = qsa_kvarn_partition(state.frontier)
    if (
        isinstance(state.body_frontier, bool)
        or not isinstance(state.body_frontier, int)
        or state.body_frontier != body_end
    ):
        raise ValueError("Qwen KVarN body frontier is inconsistent")
    layout = QWEN38_KVARN_K4V4_G128
    expected_records = _record_count(state.frontier)
    if (
        state.packed_records.shape
        != (
            expected_records,
            layout.kv_heads,
            layout.head_record_bytes,
        )
        or state.packed_records.dtype != mx.uint8
    ):
        raise ValueError("Qwen KVarN packed records are inconsistent")
    tail_tokens = state.frontier - body_end
    expected_sink = (1, layout.kv_heads, sink_end, layout.head_dim)
    expected_tail = (1, layout.kv_heads, tail_tokens, layout.head_dim)
    if (
        state.exact_sink_keys.shape != expected_sink
        or state.exact_sink_values.shape != expected_sink
        or state.exact_tail_keys.shape != expected_tail
        or state.exact_tail_values.shape != expected_tail
    ):
        raise ValueError("Qwen KVarN exact K/V geometry is inconsistent")
    exact_arrays = (
        state.exact_sink_keys,
        state.exact_sink_values,
        state.exact_tail_keys,
        state.exact_tail_values,
        state.compressed_index_keys,
        state.raw_index_tail,
    )
    if any(array.dtype != mx.bfloat16 for array in exact_arrays):
        raise ValueError("Qwen KVarN exact state must use BF16")
    if (
        isinstance(state.index_group_count, bool)
        or not isinstance(state.index_group_count, int)
        or state.index_group_count != state.frontier // QWEN38_QSA_COMPRESS_RATIO
    ):
        raise ValueError("Qwen KVarN compressed index frontier is inconsistent")
    capacity_groups = state.compressed_index_keys.shape[1]
    if (
        state.compressed_index_keys.ndim != 3
        or state.compressed_index_keys.shape[:1] != (1,)
        or state.compressed_index_keys.shape[-1] != 128
        or capacity_groups < state.index_group_count
    ):
        raise ValueError("Qwen KVarN compressed index storage is inconsistent")
    if state.compressed_index_positions.shape != (3, 1, capacity_groups):
        raise ValueError("Qwen KVarN compressed index positions are inconsistent")
    raw_rows = state.frontier % QWEN38_QSA_COMPRESS_RATIO
    if state.raw_index_tail.shape != (1, raw_rows, 128):
        raise ValueError("Qwen KVarN raw index tail is inconsistent")
    if state.raw_index_tail_positions.shape != (3, 1, raw_rows):
        raise ValueError("Qwen KVarN raw index tail positions are inconsistent")
    position_arrays = (
        state.compressed_index_positions,
        state.raw_index_tail_positions,
    )
    if any(
        array.dtype
        not in (
            mx.int32,
            mx.int64,
            mx.uint32,
            mx.uint64,
        )
        for array in position_arrays
    ):
        raise ValueError("Qwen KVarN position state is inconsistent")
    if state.compressed_index_positions.dtype != state.raw_index_tail_positions.dtype:
        raise ValueError("Qwen KVarN position dtype changed within index state")
    if sink_end + expected_records * layout.tile_tokens + tail_tokens != state.frontier:
        raise ValueError("Qwen KVarN regions do not cover the frontier")
    if (
        state.frontier >= QWEN38_KVARN_EXACT_SINK + QWEN38_KVARN_EXACT_SUFFIX
        and tail_tokens < QWEN38_KVARN_EXACT_SUFFIX
    ):
        raise ValueError("Qwen KVarN exact suffix is shorter than its policy")


def qsa_kvarn_safe_chunk_tokens(
    state: Qwen4KVarNQSAState | None,
    requested: int,
) -> int:
    """Cap one update at the next tile-sealing frontier."""
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ValueError("requested must be a positive integer")
    validate_qsa_kvarn_state(state)
    frontier = 0 if state is None else state.frontier
    first_seal = QWEN38_KVARN_EXACT_SINK + QWEN38_KVARN_EXACT_SUFFIX + QWEN38_QSA_TILE_TOKENS
    next_seal = first_seal + _record_count(frontier) * QWEN38_QSA_TILE_TOKENS
    return min(requested, next_seal - frontier)


def _empty_history() -> tuple[mx.array, mx.array]:
    layout = QWEN38_KVARN_K4V4_G128
    exact = mx.zeros((1, layout.kv_heads, 0, layout.head_dim), dtype=mx.bfloat16)
    records = mx.zeros(
        (0, layout.kv_heads, layout.head_record_bytes),
        dtype=mx.uint8,
    )
    return exact, records


def prepare_qsa_kvarn_index(
    state: Qwen4KVarNQSAState | None,
    raw_index_keys: mx.array,
    position_ids: mx.array,
    key_norm: Any,
    *,
    max_index_groups: int,
    rotary_dim: int,
    rope_base: float,
    mrope_section: tuple[int, int, int],
) -> tuple[mx.array, Qwen4KVarNIndexUpdate]:
    """Prepare complete BF16 index groups without publishing them to state."""
    validate_qsa_kvarn_state(state)
    if (
        isinstance(max_index_groups, bool)
        or not isinstance(max_index_groups, int)
        or max_index_groups <= 0
    ):
        raise ValueError("max_index_groups must be a positive integer")
    if raw_index_keys.ndim != 3 or raw_index_keys.shape[0] != 1:
        raise ValueError("raw_index_keys must have shape [1, tokens, 128]")
    new_tokens = raw_index_keys.shape[1]
    if new_tokens <= 0 or raw_index_keys.shape[2] != 128:
        raise ValueError("raw_index_keys must have shape [1, tokens, 128]")
    if raw_index_keys.dtype != mx.bfloat16:
        raise ValueError("raw_index_keys must use BF16")
    positions = position_ids
    if positions.ndim == 2:
        positions = mx.broadcast_to(positions[None], (3, *positions.shape))
    if positions.shape != (3, 1, new_tokens) or positions.dtype not in (
        mx.int32,
        mx.int64,
        mx.uint32,
        mx.uint64,
    ):
        raise ValueError("position_ids must match the BF16 index-key chunk")

    old_frontier = 0 if state is None else state.frontier
    old_group_count = 0 if state is None else state.index_group_count
    if state is None:
        storage_keys = mx.zeros((1, max_index_groups, 128), dtype=mx.bfloat16)
        storage_positions = mx.zeros((3, 1, max_index_groups), dtype=positions.dtype)
        old_raw = mx.zeros((1, 0, 128), dtype=mx.bfloat16)
        old_raw_positions = mx.zeros((3, 1, 0), dtype=positions.dtype)
    else:
        if state.compressed_index_keys.shape[1] != max_index_groups:
            raise ValueError("Qwen KVarN index capacity changed at a live frontier")
        if state.compressed_index_positions.dtype != positions.dtype:
            raise ValueError("position dtype changed at a live KVarN frontier")
        storage_keys = state.compressed_index_keys
        storage_positions = state.compressed_index_positions
        old_raw = state.raw_index_tail
        old_raw_positions = state.raw_index_tail_positions

    return prepare_qsa_kvarn_index_storage(
        old_frontier=old_frontier,
        old_group_count=old_group_count,
        storage_keys=storage_keys,
        storage_positions=storage_positions,
        raw_index_tail=old_raw,
        raw_index_tail_positions=old_raw_positions,
        raw_index_keys=raw_index_keys,
        position_ids=positions,
        key_norm=key_norm,
        rotary_dim=rotary_dim,
        rope_base=rope_base,
        mrope_section=mrope_section,
    )


def prepare_qsa_kvarn_index_storage(
    *,
    old_frontier: int,
    old_group_count: int,
    storage_keys: mx.array,
    storage_positions: mx.array,
    raw_index_tail: mx.array,
    raw_index_tail_positions: mx.array,
    raw_index_keys: mx.array,
    position_ids: mx.array,
    key_norm: Any,
    rotary_dim: int,
    rope_base: float,
    mrope_section: tuple[int, int, int],
) -> tuple[mx.array, Qwen4KVarNIndexUpdate]:
    """Prepare index groups against caller-owned fixed-capacity storage.

    The returned update does not publish any group. Both the functional cache
    and the request-owned mutable cache use this seam, so their compressed-key
    arithmetic and group-frontier semantics remain one implementation.
    """
    integer_fields = {
        "old_frontier": old_frontier,
        "old_group_count": old_group_count,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_fields.values()
    ):
        raise ValueError(f"Qwen KVarN index counters are invalid: {integer_fields}")
    if (
        storage_keys.ndim != 3
        or storage_keys.shape[0] != 1
        or storage_keys.shape[-1] != 128
        or storage_keys.dtype != mx.bfloat16
    ):
        raise ValueError("Qwen KVarN index-key storage is incompatible")
    capacity_groups = int(storage_keys.shape[1])
    if storage_positions.shape != (3, 1, capacity_groups) or storage_positions.dtype not in (
        mx.int32,
        mx.int64,
        mx.uint32,
        mx.uint64,
    ):
        raise ValueError("Qwen KVarN index-position storage is incompatible")
    if old_group_count > capacity_groups:
        raise ValueError("Qwen KVarN index group count exceeds capacity")
    if old_group_count != old_frontier // QWEN38_QSA_COMPRESS_RATIO:
        raise ValueError("Qwen KVarN index group count is off the frontier")
    raw_rows = old_frontier % QWEN38_QSA_COMPRESS_RATIO
    if raw_index_tail.shape != (1, raw_rows, 128) or raw_index_tail.dtype != mx.bfloat16:
        raise ValueError("Qwen KVarN raw index tail is incompatible")
    if (
        raw_index_tail_positions.shape != (3, 1, raw_rows)
        or raw_index_tail_positions.dtype != storage_positions.dtype
    ):
        raise ValueError("Qwen KVarN raw index positions are incompatible")
    if (
        raw_index_keys.ndim != 3
        or raw_index_keys.shape[0] != 1
        or raw_index_keys.shape[2] != 128
        or raw_index_keys.shape[1] <= 0
        or raw_index_keys.dtype != mx.bfloat16
    ):
        raise ValueError("raw_index_keys must have shape [1, tokens, 128] in BF16")
    if (
        position_ids.shape != (3, 1, raw_index_keys.shape[1])
        or position_ids.dtype != storage_positions.dtype
    ):
        raise ValueError("position_ids must match the fixed index storage")

    combined = mx.concatenate([raw_index_tail, raw_index_keys], axis=1)
    combined_positions = mx.concatenate(
        [raw_index_tail_positions, position_ids],
        axis=2,
    )
    complete_rows = (combined.shape[1] // QWEN38_QSA_COMPRESS_RATIO) * QWEN38_QSA_COMPRESS_RATIO
    new_group_count = complete_rows // QWEN38_QSA_COMPRESS_RATIO
    next_group_count = old_group_count + new_group_count
    if next_group_count > capacity_groups:
        raise ValueError(
            f"Qwen KVarN compressed index capacity {capacity_groups} group(s) "
            f"is exhausted at frontier {old_frontier + raw_index_keys.shape[1]}"
        )

    if new_group_count:
        grouped = combined[:, :complete_rows].reshape(
            1,
            new_group_count,
            QWEN38_QSA_COMPRESS_RATIO,
            128,
        )
        pooled = mx.mean(grouped.astype(mx.float32), axis=2).astype(mx.bfloat16)
        normalized = key_norm(pooled)
        group_positions = combined_positions[:, :, :complete_rows:QWEN38_QSA_COMPRESS_RATIO]
        new_group_keys = qwen4_partial_rope(
            normalized[:, :, None, :],
            group_positions,
            rotary_dim=rotary_dim,
            base=rope_base,
            mrope_section=mrope_section,
        )[:, :, 0, :]
    else:
        new_group_keys = mx.zeros((1, 0, 128), dtype=mx.bfloat16)
        group_positions = mx.zeros((3, 1, 0), dtype=storage_positions.dtype)

    visible_keys = mx.concatenate(
        [storage_keys[:, :old_group_count], new_group_keys],
        axis=1,
    )
    update = Qwen4KVarNIndexUpdate(
        old_frontier=old_frontier,
        new_tokens=raw_index_keys.shape[1],
        storage_keys=storage_keys,
        storage_positions=storage_positions,
        old_group_count=old_group_count,
        new_group_keys=new_group_keys,
        new_group_positions=group_positions,
        next_raw_tail=combined[:, complete_rows:],
        next_raw_tail_positions=combined_positions[:, :, complete_rows:],
    )
    return visible_keys, update


def advance_qsa_kvarn_state(
    state: Qwen4KVarNQSAState | None,
    keys: mx.array,
    values: mx.array,
    valid_tokens: mx.array,
    *,
    index_update: Qwen4KVarNIndexUpdate,
    encode_tile: TileEncoder = encode_qsa_kvarn_tile_mlx,
) -> Qwen4KVarNQSAState:
    """Publish one segment into append-only index storage and return its state.

    Existing logical group prefixes are immutable. The fixed-capacity backing
    array may be shared with an older frontier, whose group count keeps newly
    published slots outside its visible state.
    """
    validate_qsa_kvarn_state(state)
    layout = QWEN38_KVARN_K4V4_G128
    if keys.ndim != 4 or keys.shape[:2] != (1, layout.kv_heads):
        raise ValueError("keys must have shape [1, 2, tokens, 256]")
    if keys.shape[-1] != layout.head_dim or keys.shape[2] <= 0:
        raise ValueError("keys must have shape [1, 2, tokens, 256]")
    if values.shape != keys.shape:
        raise ValueError("values must match keys")
    if keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16:
        raise ValueError("KVarN cache K/V must use BF16")
    new_tokens = keys.shape[2]
    if not isinstance(index_update, Qwen4KVarNIndexUpdate):
        raise TypeError("index_update must be a Qwen4KVarNIndexUpdate")
    old_frontier = 0 if state is None else state.frontier
    if index_update.old_frontier != old_frontier or index_update.new_tokens != new_tokens:
        raise ValueError("Qwen KVarN index update does not match the K/V frontier")
    if valid_tokens.shape != (1, new_tokens) or valid_tokens.dtype != mx.bool_:
        raise ValueError("valid_tokens must match the token chunk")
    finite_and_full = mx.all(valid_tokens)
    for array in (keys, values, index_update.new_group_keys, index_update.next_raw_tail):
        finite_and_full = finite_and_full & mx.all(mx.isfinite(array))
    if not bool(finite_and_full.item()):
        raise ValueError("KVarN cache requires finite unpadded single-row input")
    allowed = qsa_kvarn_safe_chunk_tokens(state, new_tokens)
    if allowed != new_tokens:
        raise ValueError(f"KVarN cache chunk crosses a tile seal; split after {allowed} token(s)")

    if state is None:
        old_exact, old_records = _empty_history()
        old_sink_keys = old_sink_values = old_exact
        old_tail_keys = old_tail_values = old_exact
        old_frontier = 0
        old_record_count = 0
    else:
        old_records = state.packed_records
        old_sink_keys = state.exact_sink_keys
        old_sink_values = state.exact_sink_values
        old_tail_keys = state.exact_tail_keys
        old_tail_values = state.exact_tail_values
        old_frontier = state.frontier
        old_record_count = state.packed_records.shape[0]

    sink_room = max(0, QWEN38_KVARN_EXACT_SINK - old_sink_keys.shape[2])
    sink_take = min(new_tokens, sink_room)
    sink_keys = mx.concatenate([old_sink_keys, keys[..., :sink_take, :]], axis=2)
    sink_values = mx.concatenate([old_sink_values, values[..., :sink_take, :]], axis=2)
    tail_keys = mx.concatenate([old_tail_keys, keys[..., sink_take:, :]], axis=2)
    tail_values = mx.concatenate([old_tail_values, values[..., sink_take:, :]], axis=2)
    frontier = old_frontier + new_tokens
    _, body_frontier, _ = qsa_kvarn_partition(frontier)
    record_count = _record_count(frontier)
    new_record_count = record_count - old_record_count
    if new_record_count not in (0, 1):
        raise ValueError("KVarN cache update crossed more than one tile seal")
    packed_records = old_records
    if new_record_count:
        tile_keys = tail_keys[0, :, : layout.tile_tokens, :].transpose(1, 0, 2)
        tile_values = tail_values[0, :, : layout.tile_tokens, :].transpose(1, 0, 2)
        record = encode_tile(tile_keys, tile_values)
        if record.shape != (layout.kv_heads, layout.head_record_bytes) or record.dtype != mx.uint8:
            raise ValueError("KVarN tile encoder returned an incompatible record")
        packed_records = mx.concatenate([old_records, record[None]], axis=0)
        tail_keys = tail_keys[..., layout.tile_tokens :, :]
        tail_values = tail_values[..., layout.tile_tokens :, :]

    new_index_group_count = index_update.old_group_count + index_update.new_group_keys.shape[1]
    if index_update.new_group_keys.shape[1]:
        start = index_update.old_group_count
        end = new_index_group_count
        index_update.storage_keys[:, start:end] = index_update.new_group_keys
        index_update.storage_positions[:, :, start:end] = index_update.new_group_positions

    result = Qwen4KVarNQSAState(
        packed_records=packed_records,
        exact_sink_keys=sink_keys,
        exact_sink_values=sink_values,
        exact_tail_keys=tail_keys,
        exact_tail_values=tail_values,
        compressed_index_keys=index_update.storage_keys,
        compressed_index_positions=index_update.storage_positions,
        index_group_count=new_index_group_count,
        raw_index_tail=index_update.next_raw_tail,
        raw_index_tail_positions=index_update.next_raw_tail_positions,
        frontier=frontier,
        body_frontier=body_frontier,
    )
    validate_qsa_kvarn_state(result)
    return result


def gather_qsa_kvarn_selected_rows(
    state: Qwen4KVarNQSAState,
    selected_indices: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Gather canonical selected rows from exact and packed state regions."""
    validate_qsa_kvarn_state(state)
    if selected_indices.ndim != 3 or selected_indices.shape[0] != 1:
        raise ValueError("Qwen KVarN selected rows require shape [1, queries, width]")
    normalized, valid = qsa_normalize_selected_rows(selected_indices, state.frontier)
    return _gather_qsa_kvarn_normalized_rows(state, normalized, valid)


def gather_qsa_kvarn_selected_rows_with_pending(
    state: Qwen4KVarNQSAState | None,
    pending_keys: mx.array,
    pending_values: mx.array,
    selected_indices: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Gather prior cache rows and a not-yet-sealed exact K/V chunk.

    The pending rows remain exact for the queries that produce them. Callers may
    publish newly sealed tiles only after attention has consumed this provider.
    """
    validate_qsa_kvarn_state(state)
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
    if selected_indices.ndim != 3 or selected_indices.shape[0] != 1:
        raise ValueError("Qwen KVarN selected rows require shape [1, queries, width]")

    prior_frontier = 0 if state is None else state.frontier
    pending_tokens = pending_keys.shape[2]
    next_frontier = prior_frontier + pending_tokens
    normalized, valid = qsa_normalize_selected_rows(selected_indices, next_frontier)
    safe = mx.where(valid, normalized, 0)
    prior_valid = valid & (safe < prior_frontier)
    pending_valid = valid & (safe >= prior_frontier)

    provider_shape = (
        selected_indices.shape[0],
        selected_indices.shape[1],
        layout.kv_heads,
        selected_indices.shape[2],
        layout.head_dim,
    )
    if state is None:
        prior_keys = mx.zeros(provider_shape, dtype=mx.bfloat16)
        prior_values = mx.zeros_like(prior_keys)
    else:
        prior_keys, prior_values, _ = _gather_qsa_kvarn_normalized_rows(
            state,
            mx.where(prior_valid, safe, 0),
            prior_valid,
        )

    pending_ids = mx.where(pending_valid, safe - prior_frontier, 0)
    flat_ids = pending_ids.reshape(-1)
    token_major_keys = pending_keys[0].transpose(1, 0, 2)
    token_major_values = pending_values[0].transpose(1, 0, 2)
    exact_keys = mx.take(token_major_keys, flat_ids, axis=0)
    exact_values = mx.take(token_major_values, flat_ids, axis=0)
    gathered_shape = (
        selected_indices.shape[0],
        selected_indices.shape[1],
        selected_indices.shape[2],
        layout.kv_heads,
        layout.head_dim,
    )
    exact_keys = exact_keys.reshape(gathered_shape).transpose(0, 1, 3, 2, 4)
    exact_values = exact_values.reshape(gathered_shape).transpose(0, 1, 3, 2, 4)
    pending_mask = pending_valid[:, :, None, :, None]
    gathered_keys = prior_keys + mx.where(pending_mask, exact_keys, 0)
    gathered_values = prior_values + mx.where(pending_mask, exact_values, 0)
    return gathered_keys, gathered_values, valid


def _gather_qsa_kvarn_normalized_rows(
    state: Qwen4KVarNQSAState,
    normalized: mx.array,
    valid: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Gather already-normalized physical rows without changing lane identity."""
    validate_qsa_kvarn_state(state)
    if normalized.ndim != 3 or normalized.shape[0] != 1 or normalized.dtype != mx.int32:
        raise ValueError("normalized Qwen KVarN rows require int32 [1, queries, width]")
    if valid.shape != normalized.shape or valid.dtype != mx.bool_:
        raise ValueError("normalized Qwen KVarN validity must match selected rows")
    safe = mx.where(valid, normalized, 0)
    flat_safe = safe.reshape(-1)
    sink_count = state.exact_sink_keys.shape[2]
    sink_mask = valid & (safe < sink_count)
    body_mask = valid & (safe >= sink_count) & (safe < state.body_frontier)
    tail_mask = valid & (safe >= state.body_frontier)

    def exact_rows(array: mx.array, ids: mx.array) -> mx.array:
        token_major = array[0].transpose(1, 0, 2)
        return mx.take(token_major, ids.reshape(-1), axis=0)

    sink_ids = mx.where(sink_mask, safe, 0)
    sink_keys = exact_rows(state.exact_sink_keys, sink_ids)
    sink_values = exact_rows(state.exact_sink_values, sink_ids)

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
        state.packed_records,
        tile_ids,
        tile_offsets,
    )

    if state.exact_tail_keys.shape[2]:
        tail_ids = mx.where(tail_mask, safe - state.body_frontier, 0)
        tail_keys = exact_rows(state.exact_tail_keys, tail_ids)
        tail_values = exact_rows(state.exact_tail_values, tail_ids)
    else:
        tail_keys = mx.zeros_like(body_keys)
        tail_values = mx.zeros_like(body_values)

    flat_shape = (*flat_safe.shape, 1, 1)
    gathered_keys = mx.where(sink_mask.reshape(flat_shape), sink_keys, 0)
    gathered_values = mx.where(sink_mask.reshape(flat_shape), sink_values, 0)
    gathered_keys = gathered_keys + mx.where(body_mask.reshape(flat_shape), body_keys, 0)
    gathered_values = gathered_values + mx.where(body_mask.reshape(flat_shape), body_values, 0)
    gathered_keys = gathered_keys + mx.where(tail_mask.reshape(flat_shape), tail_keys, 0)
    gathered_values = gathered_values + mx.where(tail_mask.reshape(flat_shape), tail_values, 0)
    batch_size, query_count, selected_width = valid.shape
    provider_shape = (
        batch_size,
        query_count,
        selected_width,
        QWEN38_KVARN_K4V4_G128.kv_heads,
        QWEN38_KVARN_K4V4_G128.head_dim,
    )
    gathered_keys = gathered_keys.reshape(provider_shape).transpose(0, 1, 3, 2, 4)
    gathered_values = gathered_values.reshape(provider_shape).transpose(0, 1, 3, 2, 4)
    return gathered_keys, gathered_values, valid


def fork_qsa_kvarn_state(state: Qwen4KVarNQSAState) -> Qwen4KVarNQSAState:
    """Fork exact/history arrays while sharing append-only packed records."""
    validate_qsa_kvarn_state(state)
    return Qwen4KVarNQSAState(
        packed_records=state.packed_records,
        exact_sink_keys=state.exact_sink_keys + mx.zeros_like(state.exact_sink_keys),
        exact_sink_values=state.exact_sink_values + mx.zeros_like(state.exact_sink_values),
        exact_tail_keys=state.exact_tail_keys + mx.zeros_like(state.exact_tail_keys),
        exact_tail_values=state.exact_tail_values + mx.zeros_like(state.exact_tail_values),
        compressed_index_keys=(
            state.compressed_index_keys + mx.zeros_like(state.compressed_index_keys)
        ),
        compressed_index_positions=(
            state.compressed_index_positions + mx.zeros_like(state.compressed_index_positions)
        ),
        index_group_count=state.index_group_count,
        raw_index_tail=state.raw_index_tail + mx.zeros_like(state.raw_index_tail),
        raw_index_tail_positions=(
            state.raw_index_tail_positions + mx.zeros_like(state.raw_index_tail_positions)
        ),
        frontier=state.frontier,
        body_frontier=state.body_frontier,
        schema=state.schema,
    )
