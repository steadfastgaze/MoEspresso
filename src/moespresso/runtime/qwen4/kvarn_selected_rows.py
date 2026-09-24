"""Selected-row Metal decoder for packed Qwen KVarN K4/V4 records.

Both kernels read only requested token-major code rows and their tile metadata.
The reconstruction kernel returns rotated FP32 rows for MLX's Hadamard
transform. The mutable gather also restores exact rows and applies the inverse
transform before its BF16 cast. Cache lifecycle, QSA selection, and attention
remain outside this module.

The numeric record contract derives from huawei-csl/KVarN revision
``7586257f1c632e63187bfacbbe21ccb51540f7b3``. MoEspresso uses a tight
token-major wire specialized for released Qwen3.8 sparse attention.
"""

from __future__ import annotations

from functools import cache as memoize

import mlx.core as mx

from moespresso.runtime.qwen4.kvarn_layout import QWEN38_KVARN_K4V4_G128
from moespresso.runtime.qwen4.qsa import qsa_normalize_selected_rows


_THREADS = 256


# Modification notice: this selected-row decoder implements the KVarN scale
# equations for MoEspresso's tight token-major Qwen QSA record. It removes the
# upstream paged allocator, dense-attention traversal, and CUDA dependencies.
_SOURCE = r"""
    uint channel = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y;

    uint row_count = (uint)tile_ids_shape[0];
    uint tile_count = (uint)records_shape[0];
    if (channel >= 256u || row >= row_count || head >= 2u) return;

    int tile = tile_ids[row];
    int token = tile_offsets[row];
    uint64_t output_index = ((uint64_t)row * 2u + head) * 256u + channel;

    if (tile < 0 || tile >= (int)tile_count || token < 0 || token >= 128) {
        k_rotated[output_index] = 0.0f;
        v_rotated[output_index] = 0.0f;
        return;
    }

    device const uchar *bytes = (device const uchar *)records;
    device const uchar *record =
        bytes + ((uint64_t)tile * 2u + head) * 35072u;
    uint code_column = channel >> 1u;
    uint shift = (channel & 1u) * 4u;

    uchar packed_k = record[(uint64_t)token * 128u + code_column];
    float k_code = float((packed_k >> shift) & 0x0fu);
    device const half *k_scale = (device const half *)(record + 16384u);
    device const half *k_zero = (device const half *)(record + 16896u);
    device const half *k_token_scale = (device const half *)(record + 17408u);
    k_rotated[output_index] =
        (k_code * float(k_scale[channel]) + float(k_zero[channel]))
        * float(k_token_scale[token]);

    uchar packed_v = record[17664u + (uint64_t)token * 128u + code_column];
    float v_code = float((packed_v >> shift) & 0x0fu);
    device const half *v_channel_scale = (device const half *)(record + 34048u);
    device const half *v_token_scale = (device const half *)(record + 34560u);
    device const half *v_zero = (device const half *)(record + 34816u);
    v_rotated[output_index] =
        (v_code * float(v_token_scale[token]) + float(v_zero[token]))
        * float(v_channel_scale[channel]);
"""


# Increasing-stride butterflies preserve the released FP32 operation order.
# SIMD exchange covers the first five stages; shared memory covers the last three.
_MUTABLE_GATHER_SOURCE = r"""
    uint channel = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y;
    if (channel >= 256u || row >= (uint)logical_ids_shape[0] || head >= 2u) return;
    int logical = logical_ids[row];
    int frontier_value = frontier[0];
    int body_frontier = 128 + record_count[0] * 128;
    uint64_t output = ((uint64_t)row * 2u + head) * 256u + channel;
    if (!valid_rows[row] || logical < 0 || logical >= frontier_value + (int)pending_keys_shape[2]) {
        gathered_keys[output] = bfloat(0.0f);
        gathered_values[output] = bfloat(0.0f);
        return;
    }
    if (logical < 128) {
        uint64_t source = ((uint64_t)head * 128u + (uint)logical) * 256u + channel;
        gathered_keys[output] = exact_sink_keys[source];
        gathered_values[output] = exact_sink_values[source];
        return;
    }
    if (logical >= body_frontier) {
        int local;
        device const bfloat *source_keys;
        device const bfloat *source_values;
        uint source_tokens;
        if (logical < frontier_value) {
            local = (logical - 128) % (int)exact_tail_keys_shape[2];
            source_keys = exact_tail_keys;
            source_values = exact_tail_values;
            source_tokens = (uint)exact_tail_keys_shape[2];
        } else {
            local = logical - frontier_value;
            source_keys = pending_keys;
            source_values = pending_values;
            source_tokens = (uint)pending_keys_shape[2];
        }
        uint64_t source = ((uint64_t)head * source_tokens + (uint)local) * 256u + channel;
        gathered_keys[output] = source_keys[source];
        gathered_values[output] = source_values[source];
        return;
    }
    int body_local = logical - 128;
    int tile = body_local / 128;
    int token = body_local % 128;
    device const uchar *record = (device const uchar *)records + ((uint64_t)tile * 2u + head) * 35072u;
    uint code_column = channel >> 1u;
    uint shift = (channel & 1u) * 4u;
    uchar packed_k = record[(uint64_t)token * 128u + code_column];
    float k_code = float((packed_k >> shift) & 0x0fu);
    device const half *k_scale = (device const half *)(record + 16384u);
    device const half *k_zero = (device const half *)(record + 16896u);
    device const half *k_token_scale = (device const half *)(record + 17408u);
    float key = (k_code * float(k_scale[channel]) + float(k_zero[channel])) * float(k_token_scale[token]);
    uchar packed_v = record[17664u + (uint64_t)token * 128u + code_column];
    float v_code = float((packed_v >> shift) & 0x0fu);
    device const half *v_channel_scale = (device const half *)(record + 34048u);
    device const half *v_token_scale = (device const half *)(record + 34560u);
    device const half *v_zero = (device const half *)(record + 34816u);
    float value = (v_code * float(v_token_scale[token]) + float(v_zero[token])) * float(v_channel_scale[channel]);
    for (uint stride = 1u; stride < 32u; stride <<= 1u) {
        float other_key = simd_shuffle_xor(key, stride);
        float other_value = simd_shuffle_xor(value, stride);
        key = (channel & stride) ? other_key - key : key + other_key;
        value = (channel & stride) ? other_value - value : value + other_value;
    }
    threadgroup float keys[256];
    threadgroup float values[256];
    keys[channel] = key;
    values[channel] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32u; stride <= 128u; stride <<= 1u) {
        float other_key = keys[channel ^ stride];
        float other_value = values[channel ^ stride];
        key = (channel & stride) ? other_key - key : key + other_key;
        value = (channel & stride) ? other_value - value : value + other_value;
        if (stride < 128u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            keys[channel] = key;
            values[channel] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    gathered_keys[output] = bfloat(key * 0.0625f);
    gathered_values[output] = bfloat(value * 0.0625f);
"""


def _metal_available() -> bool:
    return bool(
        mx.metal.is_available()
        and getattr(getattr(mx, "fast", None), "metal_kernel", None) is not None
    )


@memoize
def _selected_row_kernel():
    if not _metal_available():
        raise RuntimeError("Qwen KVarN selected-row reconstruction requires Metal")
    return mx.fast.metal_kernel(
        name="moespresso_qwen38_qsa_kvarn_k4v4_g128_selected_rows",
        input_names=["records", "tile_ids", "tile_offsets"],
        output_names=["k_rotated", "v_rotated"],
        source=_SOURCE,
    )


@memoize
def _mutable_gather_kernel():
    if not _metal_available():
        raise RuntimeError("Qwen KVarN mutable selected-row gather requires Metal")
    return mx.fast.metal_kernel(
        name="moespresso_qwen38_qsa_kvarn_k4v4_g128_mutable_gather_simd",
        input_names=[
            "records",
            "exact_sink_keys",
            "exact_sink_values",
            "exact_tail_keys",
            "exact_tail_values",
            "pending_keys",
            "pending_values",
            "logical_ids",
            "valid_rows",
            "record_count",
            "frontier",
        ],
        output_names=["gathered_keys", "gathered_values"],
        source=_MUTABLE_GATHER_SOURCE,
    )


def _integer_array(name: str, values: mx.array) -> mx.array:
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.dtype != mx.int32:
        raise ValueError(f"{name} must contain int32 values")
    return mx.contiguous(values)


def _validate_records(records: mx.array) -> None:
    layout = QWEN38_KVARN_K4V4_G128
    if records.ndim != 3 or records.shape[1:] != (
        layout.kv_heads,
        layout.head_record_bytes,
    ):
        raise ValueError(
            f"records must have shape [tiles, {layout.kv_heads}, {layout.head_record_bytes}]"
        )
    if records.dtype != mx.uint8:
        raise ValueError("records must use uint8 storage")


def _decode_qsa_kvarn_rows_metal(
    records: mx.array,
    tile_ids: mx.array,
    tile_offsets: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Decode normalized tile coordinates into BF16 K/V rows."""
    layout = QWEN38_KVARN_K4V4_G128
    _validate_records(records)
    ids = _integer_array("tile_ids", tile_ids)
    offsets = _integer_array("tile_offsets", tile_offsets)
    if ids.shape != offsets.shape:
        raise ValueError("tile ids and offsets must have matching shapes")
    valid = (ids >= 0) & (ids < records.shape[0]) & (offsets >= 0) & (offsets < layout.tile_tokens)
    row_count = ids.shape[0]
    output_shape = (row_count, layout.kv_heads, layout.head_dim)
    if row_count == 0 or records.shape[0] == 0:
        empty = mx.zeros(output_shape, dtype=mx.bfloat16)
        return empty, empty, valid

    k_rotated, v_rotated = _selected_row_kernel()(
        inputs=[mx.contiguous(records), ids, offsets],
        output_shapes=[output_shape, output_shape],
        output_dtypes=[mx.float32, mx.float32],
        grid=(row_count * _THREADS, layout.kv_heads, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    keys = mx.hadamard_transform(k_rotated).astype(mx.bfloat16)
    values = mx.hadamard_transform(v_rotated).astype(mx.bfloat16)
    return keys, values, valid


def gather_qsa_kvarn_mutable_rows_metal(
    records: mx.array,
    exact_sink_keys: mx.array,
    exact_sink_values: mx.array,
    exact_tail_keys: mx.array,
    exact_tail_values: mx.array,
    pending_keys: mx.array,
    pending_values: mx.array,
    normalized: mx.array,
    valid: mx.array,
    *,
    frontier: int,
    record_count: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Gather one normalized mutable QSA selection in a single Metal kernel."""
    layout = QWEN38_KVARN_K4V4_G128
    _validate_records(records)
    if normalized.ndim != 3 or normalized.shape[0] != 1:
        raise ValueError("normalized Qwen KVarN rows require shape [1, queries, width]")
    if normalized.dtype != mx.int32:
        raise ValueError("normalized Qwen KVarN rows must contain int32 values")
    if valid.shape != normalized.shape or valid.dtype != mx.bool_:
        raise ValueError("normalized Qwen KVarN validity must match selected rows")
    exact_shape = (1, layout.kv_heads, 128, layout.head_dim)
    if exact_sink_keys.shape != exact_shape or exact_sink_values.shape != exact_shape:
        raise ValueError("exact Qwen KVarN sink has incompatible geometry")
    if (
        exact_tail_keys.ndim != 4
        or exact_tail_keys.shape[:2] != (1, layout.kv_heads)
        or exact_tail_keys.shape[-1] != layout.head_dim
        or exact_tail_keys.shape[2] < layout.tile_tokens
        or exact_tail_keys.shape[2] % layout.tile_tokens
        or exact_tail_values.shape != exact_tail_keys.shape
    ):
        raise ValueError("exact Qwen KVarN tail has incompatible geometry")
    if (
        pending_keys.ndim != 4
        or pending_keys.shape[:2] != (1, layout.kv_heads)
        or pending_keys.shape[-1] != layout.head_dim
        or pending_keys.shape[2] <= 0
        or pending_values.shape != pending_keys.shape
    ):
        raise ValueError("pending Qwen KVarN rows have incompatible geometry")
    exact_arrays = (
        exact_sink_keys,
        exact_sink_values,
        exact_tail_keys,
        exact_tail_values,
        pending_keys,
        pending_values,
    )
    if any(array.dtype != mx.bfloat16 for array in exact_arrays):
        raise ValueError("exact and pending Qwen KVarN rows must use BF16")
    if isinstance(frontier, bool) or not isinstance(frontier, int) or frontier < 128:
        raise ValueError("Qwen KVarN mutable frontier is incompatible")
    if record_count is None:
        record_count = int(records.shape[0])
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 0
        or record_count > records.shape[0]
    ):
        raise ValueError("Qwen KVarN logical record count is incompatible")
    body_frontier = 128 + record_count * layout.tile_tokens
    if body_frontier > frontier or frontier - body_frontier > exact_tail_keys.shape[2]:
        raise ValueError("Qwen KVarN packed body and exact tail do not cover the frontier")

    row_count = normalized.size
    batch_size, query_count, selected_width = normalized.shape
    provider_shape = (
        batch_size,
        query_count,
        selected_width,
        layout.kv_heads,
        layout.head_dim,
    )
    if row_count == 0:
        empty = mx.zeros(
            (batch_size, query_count, layout.kv_heads, selected_width, layout.head_dim),
            dtype=mx.bfloat16,
        )
        return empty, empty
    flat_keys, flat_values = _mutable_gather_kernel()(
        inputs=[
            mx.contiguous(records),
            mx.contiguous(exact_sink_keys),
            mx.contiguous(exact_sink_values),
            mx.contiguous(exact_tail_keys),
            mx.contiguous(exact_tail_values),
            mx.contiguous(pending_keys),
            mx.contiguous(pending_values),
            mx.contiguous(normalized.reshape(-1)),
            mx.contiguous(valid.reshape(-1)),
            mx.array([record_count], dtype=mx.int32),
            mx.array([frontier], dtype=mx.int32),
        ],
        output_shapes=[
            (row_count, layout.kv_heads, layout.head_dim),
            (row_count, layout.kv_heads, layout.head_dim),
        ],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
        grid=(row_count * _THREADS, layout.kv_heads, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    return (
        flat_keys.reshape(provider_shape).transpose(0, 1, 3, 2, 4),
        flat_values.reshape(provider_shape).transpose(0, 1, 3, 2, 4),
    )


def reconstruct_qsa_kvarn_selected_rows_metal(
    records: mx.array,
    selected_indices: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Normalize a QSA selection and reconstruct its packed-body K/V rows."""
    layout = QWEN38_KVARN_K4V4_G128
    _validate_records(records)
    packed_tokens = records.shape[0] * layout.tile_tokens
    normalized, valid = qsa_normalize_selected_rows(selected_indices, packed_tokens)
    safe = mx.where(valid, normalized, 0)
    tile_ids = (safe // layout.tile_tokens).reshape(-1).astype(mx.int32)
    tile_offsets = (safe % layout.tile_tokens).reshape(-1).astype(mx.int32)
    flat_keys, flat_values, _ = _decode_qsa_kvarn_rows_metal(
        records,
        tile_ids,
        tile_offsets,
    )
    flat_valid = valid.reshape(-1, 1, 1)
    flat_keys = mx.where(flat_valid, flat_keys, 0)
    flat_values = mx.where(flat_valid, flat_values, 0)
    batch_size, query_count, selected_width = valid.shape
    gathered_shape = (
        batch_size,
        query_count,
        selected_width,
        layout.kv_heads,
        layout.head_dim,
    )
    gathered_keys = flat_keys.reshape(gathered_shape).transpose(0, 1, 3, 2, 4)
    gathered_values = flat_values.reshape(gathered_shape).transpose(0, 1, 3, 2, 4)
    return gathered_keys, gathered_values, valid
