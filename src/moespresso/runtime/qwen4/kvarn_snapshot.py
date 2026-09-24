"""Self-contained snapshots of committed Qwen K4/V4 attention storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import mlx.core as mx

from moespresso.runtime.qwen4.kvarn_cache import QWEN38_KVARN_EXACT_SINK, qsa_kvarn_partition
from moespresso.runtime.qwen4.kvarn_layout import QWEN38_KVARN_K4V4_G128
from moespresso.runtime.qwen4.kvarn_mutable import (
    QWEN38_KVARN_EXACT_TAIL_CAPACITY,
    Qwen4MutableKVarNStorage,
)
from moespresso.runtime.qwen4.qsa import QWEN38_QSA_COMPRESS_RATIO
from moespresso.runtime.qwen4.qsa_kvarn import Qwen4MutableKVarNQSAState


KVARN_SNAPSHOT_SCHEMA = "qwen4-kvarn-snapshot-v1"
_POSITION_DTYPES = {str(dtype): dtype for dtype in (mx.int32, mx.int64, mx.uint32, mx.uint64)}
_META_FIELDS = frozenset({"schema", "layout", "frontier", "position_dtype"})


def _tail_slots(start: int, count: int) -> mx.array:
    return (mx.arange(start, start + count, dtype=mx.int32) - QWEN38_KVARN_EXACT_SINK) % (
        QWEN38_KVARN_EXACT_TAIL_CAPACITY
    )


def snapshot_kvarn_state(
    state: Qwen4MutableKVarNQSAState,
) -> tuple[tuple[mx.array, ...], dict[str, Any]]:
    """Copy the live payload at a committed frontier, without spare capacity.

    The caller must hold request ownership while capturing. Speculative undo
    state is not durable. Exact tail rows are stored in logical token order;
    restore reconstructs their ring placement without running the quantizer.
    """
    if not isinstance(state, Qwen4MutableKVarNQSAState):
        raise ValueError("KVarN snapshot requires mutable QSA state")
    storage = state.storage
    storage.validate_frontier(state.view)
    current = storage.checkpoint()
    if current != state.view or current.mutation_cursor:
        raise ValueError("KVarN snapshot requires the active committed frontier")
    slots = _tail_slots(storage.tail_start, storage.tail_count)
    arrays = (
        storage.packed_records[: storage.record_count],
        storage.exact_sink_keys[..., : storage.sink_count, :],
        storage.exact_sink_values[..., : storage.sink_count, :],
        mx.take(storage.exact_tail_keys, slots, axis=2),
        mx.take(storage.exact_tail_values, slots, axis=2),
        storage.compressed_index_keys[:, : storage.index_group_count],
        storage.compressed_index_positions[..., : storage.index_group_count],
        storage.raw_index_keys[:, : storage.raw_index_count],
        storage.raw_index_positions[..., : storage.raw_index_count],
    )
    arrays = tuple(mx.array(array) for array in arrays)
    mx.eval(*arrays)
    return arrays, {
        "schema": KVARN_SNAPSHOT_SCHEMA,
        "layout": QWEN38_KVARN_K4V4_G128.schema,
        "frontier": storage.frontier,
        "position_dtype": str(storage.position_dtype),
    }


def restore_kvarn_state(
    arrays: Sequence[mx.array],
    metadata: Mapping[str, Any],
    *,
    max_context_tokens: int,
) -> Qwen4MutableKVarNQSAState:
    """Validate a snapshot and construct independently owned runtime storage.

    The runtime supplies capacity; the payload cannot request an allocation.
    Tensor geometry is checked before constructing the fixed-capacity store.
    Process-local markers and mutation counters are created afresh.
    """
    layout = QWEN38_KVARN_K4V4_G128
    if not isinstance(metadata, Mapping) or set(metadata) != _META_FIELDS:
        raise ValueError("KVarN snapshot metadata fields are incompatible")
    if metadata["schema"] != KVARN_SNAPSHOT_SCHEMA or metadata["layout"] != layout.schema:
        raise ValueError("KVarN snapshot schema or layout is incompatible")
    frontier = metadata["frontier"]
    if (
        isinstance(max_context_tokens, bool)
        or not isinstance(max_context_tokens, int)
        or max_context_tokens < QWEN38_QSA_COMPRESS_RATIO
        or isinstance(frontier, bool)
        or not isinstance(frontier, int)
        or not 0 <= frontier <= max_context_tokens
    ):
        raise ValueError("KVarN snapshot frontier exceeds the runtime capacity")
    position_name = metadata["position_dtype"]
    if not isinstance(position_name, str) or position_name not in _POSITION_DTYPES:
        raise ValueError("KVarN snapshot position dtype is incompatible")
    position_dtype = _POSITION_DTYPES[position_name]
    sink_count, body_end, _ = qsa_kvarn_partition(frontier)
    record_count = (body_end - sink_count) // layout.tile_tokens
    tail_count = frontier - body_end
    group_count, raw_count = divmod(frontier, QWEN38_QSA_COMPRESS_RATIO)
    exact_prefix = (1, layout.kv_heads)
    shapes = (
        (record_count, layout.kv_heads, layout.head_record_bytes),
        (*exact_prefix, sink_count, layout.head_dim),
        (*exact_prefix, sink_count, layout.head_dim),
        (*exact_prefix, tail_count, layout.head_dim),
        (*exact_prefix, tail_count, layout.head_dim),
        (1, group_count, 128),
        (3, 1, group_count),
        (1, raw_count, 128),
        (3, 1, raw_count),
    )
    dtypes = (
        mx.uint8, mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.bfloat16,
        mx.bfloat16, position_dtype, mx.bfloat16, position_dtype,
    )
    if not isinstance(arrays, (tuple, list)) or len(arrays) != len(shapes):
        raise ValueError("KVarN snapshot tensor count is incompatible")
    for index, (array, shape, dtype) in enumerate(zip(arrays, shapes, dtypes, strict=True)):
        if not isinstance(array, mx.array) or array.shape != shape or array.dtype != dtype:
            raise ValueError(f"KVarN snapshot tensor {index} has incompatible shape or dtype")
    finite = [mx.all(mx.isfinite(array)) for array in arrays if array.dtype == mx.bfloat16]
    mx.eval(*finite)
    if not all(bool(value.item()) for value in finite):
        raise ValueError("KVarN snapshot contains non-finite exact or index state")

    storage = Qwen4MutableKVarNStorage(
        max_context_tokens=max_context_tokens, position_dtype=position_dtype,
    )
    storage.packed_records[:record_count] = arrays[0]
    storage.exact_sink_keys[..., :sink_count, :] = arrays[1]
    storage.exact_sink_values[..., :sink_count, :] = arrays[2]
    slots = _tail_slots(body_end, tail_count)
    storage.exact_tail_keys[:, :, slots, :] = arrays[3]
    storage.exact_tail_values[:, :, slots, :] = arrays[4]
    storage.compressed_index_keys[:, :group_count] = arrays[5]
    storage.compressed_index_positions[..., :group_count] = arrays[6]
    storage.raw_index_keys[:, :raw_count] = arrays[7]
    storage.raw_index_positions[..., :raw_count] = arrays[8]
    storage.frontier = frontier
    storage.body_frontier = body_end
    storage.record_count = record_count
    storage.index_group_count = group_count
    storage.raw_index_count = raw_count
    mx.eval(*storage.state_arrays())
    return Qwen4MutableKVarNQSAState(storage, storage.checkpoint())
