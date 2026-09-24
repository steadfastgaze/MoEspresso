"""Device encoder for released Qwen KVarN K4/V4 cache tiles.

The numeric method derives from huawei-csl/KVarN revision
``7586257f1c632e63187bfacbbe21ccb51540f7b3``. MoEspresso fixes the geometry
to released Qwen3.8 sparse attention and writes its tight token-major record.
The QSA selector and index-key cache are outside this codec.
"""

from __future__ import annotations

import mlx.core as mx

from moespresso.runtime.qwen4.kvarn_layout import QWEN38_KVARN_K4V4_G128


# Modification notice: this encoder fixes KVarN to released Qwen QSA geometry,
# emits MoEspresso's tight token-major record, and uses MLX device operations.


def _imbalance(values: mx.array) -> mx.array:
    column_std = mx.std(values, axis=-2, ddof=1)
    row_std = mx.std(values, axis=-1, ddof=1)
    column_min = mx.maximum(mx.min(column_std, axis=-1), 1e-8)
    row_min = mx.maximum(mx.min(row_std, axis=-1), 1e-8)
    score = mx.max(column_std, axis=-1) / column_min
    score = score + mx.max(row_std, axis=-1) / row_min
    return score[:, None, None]


def _variance_normalize(
    values: mx.array,
    *,
    iterations: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Balance row and column variance independently for each KV head."""
    source = values.astype(mx.float32)
    log_column = mx.zeros((source.shape[0], 1, source.shape[2]), dtype=mx.float32)
    log_row = mx.zeros((source.shape[0], source.shape[1], 1), dtype=mx.float32)
    current = source
    best_score = _imbalance(current)
    best_column = mx.ones_like(log_column)
    best_row = mx.ones_like(log_row)

    for _ in range(iterations):
        column_std = mx.clip(
            mx.std(current, axis=-2, keepdims=True, ddof=1),
            1e-3,
            1e3,
        )
        log_column = mx.clip(log_column + mx.log(column_std), -0.3, 10.0)
        current = source / mx.exp(log_column) / mx.exp(log_row)

        row_std = mx.clip(
            mx.std(current, axis=-1, keepdims=True, ddof=1),
            1e-3,
            1e3,
        )
        log_row = mx.clip(log_row + mx.log(row_std), -0.3, 10.0)
        current = source / mx.exp(log_column) / mx.exp(log_row)
        score = _imbalance(current)
        improved = score <= best_score
        best_score = mx.where(improved, score, best_score)
        best_column = mx.where(improved, mx.exp(log_column), best_column)
        best_row = mx.where(improved, mx.exp(log_row), best_row)

    return source / best_column / best_row, best_column, best_row


def _asymmetric_k4(rows: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    low = mx.min(rows, axis=-1, keepdims=True)
    high = mx.max(rows, axis=-1, keepdims=True)
    scale = mx.maximum((high - low) / 15.0, 1e-10)
    codes = mx.clip(mx.round((rows - low) / scale), 0, 15).astype(mx.uint8)
    return codes, scale.astype(mx.float32), low.astype(mx.float32)


def _pack_token_major_nibbles(codes: mx.array) -> mx.array:
    low = codes[..., 0::2]
    high = codes[..., 1::2] << 4
    return (low | high).astype(mx.uint8)


def _metadata_half(values: mx.array) -> mx.array:
    return mx.contiguous(values.astype(mx.float16).reshape(values.shape[0], -1))


def _half_bytes(halves: mx.array) -> mx.array:
    return mx.view(halves, mx.uint8)


def encode_qsa_kvarn_tile_mlx(
    keys: mx.array,
    values: mx.array,
) -> mx.array:
    """Encode one ``[128, 2, 256]`` K/V tile into tight K4/V4 records."""
    layout = QWEN38_KVARN_K4V4_G128
    expected = (layout.tile_tokens, layout.kv_heads, layout.head_dim)
    if keys.shape != expected or values.shape != expected:
        raise ValueError(f"keys and values must both have shape {expected}")
    if keys.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise ValueError("keys must use BF16, FP16, or FP32")
    if values.dtype != keys.dtype:
        raise ValueError("keys and values must share one dtype")

    rotated_keys = mx.hadamard_transform(keys.astype(mx.float32)).transpose(1, 2, 0)
    rotated_values = mx.hadamard_transform(values.astype(mx.float32)).transpose(1, 0, 2)

    balanced_k, token_scale_k, channel_scale_k = _variance_normalize(
        rotated_keys,
        iterations=layout.normalization_iterations,
    )
    codes_k, rtn_scale_k, rtn_zero_k = _asymmetric_k4(balanced_k)
    packed_k = _pack_token_major_nibbles(codes_k.transpose(0, 2, 1))

    balanced_v, channel_scale_v, token_scale_v = _variance_normalize(
        rotated_values,
        iterations=layout.normalization_iterations,
    )
    codes_v, rtn_scale_v, rtn_zero_v = _asymmetric_k4(balanced_v)
    packed_v = _pack_token_major_nibbles(codes_v)

    metadata = {
        "k_scale": _metadata_half(channel_scale_k * rtn_scale_k),
        "k_zero": _metadata_half(channel_scale_k * rtn_zero_k),
        "k_token_scale": _metadata_half(token_scale_k),
        "v_channel_scale": _metadata_half(channel_scale_v),
        "v_token_scale": _metadata_half(token_scale_v * rtn_scale_v),
        "v_zero": _metadata_half(token_scale_v * rtn_zero_v),
    }
    finite = mx.all(mx.isfinite(keys)) & mx.all(mx.isfinite(values))
    for payload in metadata.values():
        finite = finite & mx.all(mx.isfinite(payload))
    if not bool(finite.item()):
        raise ValueError("KVarN tile and FP16 metadata must remain finite")

    records = mx.zeros(
        (layout.kv_heads, layout.head_record_bytes),
        dtype=mx.uint8,
    )

    def write(name: str, payload: mx.array) -> None:
        field = layout.field(name)
        records[:, field.offset : field.end] = payload.reshape(layout.kv_heads, -1)

    write("k_codes", packed_k)
    write("k_scale", _half_bytes(metadata["k_scale"]))
    write("k_zero", _half_bytes(metadata["k_zero"]))
    write("k_token_scale", _half_bytes(metadata["k_token_scale"]))
    write("v_codes", packed_v)
    write("v_channel_scale", _half_bytes(metadata["v_channel_scale"]))
    write("v_token_scale", _half_bytes(metadata["v_token_scale"]))
    write("v_zero", _half_bytes(metadata["v_zero"]))
    return records
