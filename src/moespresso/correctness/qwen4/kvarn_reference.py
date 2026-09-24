"""Deterministic KVarN K4/V4 reference codec for Qwen sparse attention.

The numeric method adapts huawei-csl/KVarN revision
``7586257f1c632e63187bfacbbe21ccb51540f7b3``. MoEspresso fixes the geometry
to released Qwen3.8 QSA, stores both code payloads token-major, removes vLLM
allocator padding, and reconstructs only requested physical rows.
"""

from __future__ import annotations

import math

import numpy as np

from moespresso.runtime.qwen4.kvarn_layout import (
    QWEN38_KVARN_K4V4_G128,
    Qwen4KVarNLayout,
)


def normalized_sylvester_hadamard(width: int) -> np.ndarray:
    """Return the symmetric orthonormal Sylvester matrix for ``width``."""
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("Hadamard width must be a positive integer")
    if width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    matrix = np.ones((1, 1), dtype=np.float32)
    while matrix.shape[0] < width:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]]).astype(
            np.float32,
            copy=False,
        )
    return matrix * np.float32(1.0 / math.sqrt(width))


def _imbalance(values: np.ndarray) -> np.float32:
    column_std = np.std(values, axis=0, ddof=1, dtype=np.float32)
    row_std = np.std(values, axis=1, ddof=1, dtype=np.float32)
    column_min = np.maximum(column_std.min(), np.float32(1e-8))
    row_min = np.maximum(row_std.min(), np.float32(1e-8))
    return np.float32(column_std.max() / column_min + row_std.max() / row_min)


def _variance_normalize(
    values: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance row and column variance and retain the best iteration."""
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or min(source.shape) < 2:
        raise ValueError("normalization input must be a two-dimensional matrix")
    if not np.all(np.isfinite(source)):
        raise ValueError("normalization input must be finite")

    log_column = np.zeros((1, source.shape[1]), dtype=np.float32)
    log_row = np.zeros((source.shape[0], 1), dtype=np.float32)
    current = source.copy()
    best_score = _imbalance(current)
    best_column = np.ones_like(log_column)
    best_row = np.ones_like(log_row)

    for _ in range(iterations):
        column_std = np.clip(
            np.std(current, axis=0, keepdims=True, ddof=1, dtype=np.float32),
            np.float32(1e-3),
            np.float32(1e3),
        )
        log_column = np.clip(
            log_column + np.log(column_std),
            np.float32(-0.3),
            np.float32(10.0),
        ).astype(np.float32)
        current = source / np.exp(log_column) / np.exp(log_row)

        row_std = np.clip(
            np.std(current, axis=1, keepdims=True, ddof=1, dtype=np.float32),
            np.float32(1e-3),
            np.float32(1e3),
        )
        log_row = np.clip(
            log_row + np.log(row_std),
            np.float32(-0.3),
            np.float32(10.0),
        ).astype(np.float32)
        current = source / np.exp(log_column) / np.exp(log_row)
        score = _imbalance(current)
        if score <= best_score:
            best_score = score
            best_column = np.exp(log_column).astype(np.float32)
            best_row = np.exp(log_row).astype(np.float32)

    balanced = source / best_column / best_row
    return balanced.astype(np.float32), best_column, best_row


def _asymmetric_k4(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = rows.min(axis=1, keepdims=True)
    high = rows.max(axis=1, keepdims=True)
    scale = np.maximum((high - low) / np.float32(15), np.float32(1e-10))
    codes = np.clip(np.rint((rows - low) / scale), 0, 15).astype(np.uint8)
    return codes, scale.astype(np.float32), low.astype(np.float32)


def _pack_token_major_nibbles(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes)
    if values.ndim != 2 or values.shape[-1] % 2:
        raise ValueError("four-bit codes must have an even token-major width")
    if values.dtype != np.uint8 or np.any(values > 15):
        raise ValueError("four-bit codes must be uint8 values in [0, 15]")
    return values[:, 0::2] | (values[:, 1::2] << np.uint8(4))


def _unpack_token_major_nibbles(packed: np.ndarray, width: int) -> np.ndarray:
    if packed.ndim != 2 or packed.shape[-1] * 2 != width:
        raise ValueError("packed four-bit rows do not match the requested width")
    codes = np.empty((packed.shape[0], width), dtype=np.uint8)
    codes[:, 0::2] = packed & np.uint8(0x0F)
    codes[:, 1::2] = packed >> np.uint8(4)
    return codes


def _write_f16(record: np.ndarray, offset: int, values: np.ndarray) -> None:
    payload = np.asarray(values, dtype="<f2").reshape(-1).view(np.uint8)
    record[offset : offset + payload.size] = payload


def _read_f16(record: np.ndarray, offset: int, count: int) -> np.ndarray:
    raw = record[offset : offset + count * 2]
    values = np.frombuffer(raw.tobytes(), dtype="<f2", count=count).astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("KVarN record metadata must be finite")
    return values


def _validate_input_tiles(
    keys: np.ndarray,
    values: np.ndarray,
    layout: Qwen4KVarNLayout,
) -> tuple[np.ndarray, np.ndarray]:
    key = np.asarray(keys)
    value = np.asarray(values)
    expected = (layout.tile_tokens, layout.kv_heads, layout.head_dim)
    if key.shape != expected or value.shape != expected:
        raise ValueError(f"keys and values must both have shape {expected}")
    supported = {"bfloat16", "float16", "float32"}
    if key.dtype.name not in supported or value.dtype.name not in supported:
        raise ValueError("keys and values must use BF16, FP16, or FP32")
    key = key.astype(np.float32)
    value = value.astype(np.float32)
    if not np.all(np.isfinite(key)) or not np.all(np.isfinite(value)):
        raise ValueError("keys and values must be finite")
    return key, value


def encode_qsa_kv_tile(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    layout: Qwen4KVarNLayout = QWEN38_KVARN_K4V4_G128,
) -> np.ndarray:
    """Encode one released QSA tile into two tight per-head records."""
    key, value = _validate_input_tiles(keys, values, layout)
    hadamard = normalized_sylvester_hadamard(layout.head_dim)
    records = np.empty(
        (layout.kv_heads, layout.head_record_bytes),
        dtype=np.uint8,
    )

    for head in range(layout.kv_heads):
        record = records[head]
        key_rotated = key[:, head] @ hadamard
        value_rotated = value[:, head] @ hadamard

        balanced_k, token_scale_k, channel_scale_k = _variance_normalize(
            key_rotated.T,
            iterations=layout.normalization_iterations,
        )
        codes_k, rtn_scale_k, rtn_zero_k = _asymmetric_k4(balanced_k)
        packed_k = _pack_token_major_nibbles(codes_k.T)
        k_codes = layout.field("k_codes")
        record[k_codes.offset : k_codes.end] = packed_k.reshape(-1)
        _write_f16(
            record,
            layout.field("k_scale").offset,
            channel_scale_k * rtn_scale_k,
        )
        _write_f16(
            record,
            layout.field("k_zero").offset,
            channel_scale_k * rtn_zero_k,
        )
        _write_f16(
            record,
            layout.field("k_token_scale").offset,
            token_scale_k,
        )

        balanced_v, channel_scale_v, token_scale_v = _variance_normalize(
            value_rotated,
            iterations=layout.normalization_iterations,
        )
        codes_v, rtn_scale_v, rtn_zero_v = _asymmetric_k4(balanced_v)
        packed_v = _pack_token_major_nibbles(codes_v)
        v_codes = layout.field("v_codes")
        record[v_codes.offset : v_codes.end] = packed_v.reshape(-1)
        _write_f16(
            record,
            layout.field("v_channel_scale").offset,
            channel_scale_v,
        )
        _write_f16(
            record,
            layout.field("v_token_scale").offset,
            token_scale_v * rtn_scale_v,
        )
        _write_f16(
            record,
            layout.field("v_zero").offset,
            token_scale_v * rtn_zero_v,
        )
    return records


def _validate_records(
    records: np.ndarray,
    layout: Qwen4KVarNLayout,
) -> np.ndarray:
    array = np.asarray(records)
    if array.dtype != np.uint8:
        raise ValueError("KVarN records must use uint8 storage")
    if array.shape != (layout.kv_heads, layout.head_record_bytes):
        raise ValueError(
            f"KVarN records must have shape ({layout.kv_heads}, {layout.head_record_bytes})"
        )
    if not array.flags.c_contiguous:
        raise ValueError("KVarN records must be contiguous")
    layout.validate_record_nbytes(array.nbytes)
    return array


def reconstruct_qsa_kv_rows(
    records: np.ndarray,
    offsets: np.ndarray,
    *,
    layout: Qwen4KVarNLayout = QWEN38_KVARN_K4V4_G128,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct requested tile offsets in order, preserving duplicates."""
    packed = _validate_records(records, layout)
    selected = np.asarray(offsets)
    if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("offsets must be a one-dimensional integer array")
    if np.any(selected < 0) or np.any(selected >= layout.tile_tokens):
        raise ValueError("tile offset is outside the KVarN record")
    selected = selected.astype(np.int64, copy=False)
    keys = np.empty((selected.size, layout.kv_heads, layout.head_dim), dtype=np.float32)
    values = np.empty_like(keys)
    hadamard = normalized_sylvester_hadamard(layout.head_dim)

    for head in range(layout.kv_heads):
        record = packed[head]
        k_codes_field = layout.field("k_codes")
        k_codes = _unpack_token_major_nibbles(
            record[k_codes_field.offset : k_codes_field.end].reshape(
                layout.tile_tokens,
                layout.head_dim // 2,
            ),
            layout.head_dim,
        )[selected]
        k_scale = _read_f16(record, layout.field("k_scale").offset, layout.head_dim)
        k_zero = _read_f16(record, layout.field("k_zero").offset, layout.head_dim)
        k_token_scale = _read_f16(
            record,
            layout.field("k_token_scale").offset,
            layout.tile_tokens,
        )[selected]
        key_rotated = (k_codes.astype(np.float32) * k_scale[None] + k_zero[None]) * k_token_scale[
            :, None
        ]
        keys[:, head] = key_rotated @ hadamard

        v_codes_field = layout.field("v_codes")
        v_codes = _unpack_token_major_nibbles(
            record[v_codes_field.offset : v_codes_field.end].reshape(
                layout.tile_tokens,
                layout.head_dim // 2,
            ),
            layout.head_dim,
        )[selected]
        v_channel_scale = _read_f16(
            record,
            layout.field("v_channel_scale").offset,
            layout.head_dim,
        )
        v_token_scale = _read_f16(
            record,
            layout.field("v_token_scale").offset,
            layout.tile_tokens,
        )[selected]
        v_zero = _read_f16(
            record,
            layout.field("v_zero").offset,
            layout.tile_tokens,
        )[selected]
        value_rotated = (
            v_codes.astype(np.float32) * v_token_scale[:, None] + v_zero[:, None]
        ) * v_channel_scale[None]
        values[:, head] = value_rotated @ hadamard
    return keys, values
