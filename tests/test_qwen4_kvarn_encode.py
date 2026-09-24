from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from moespresso.correctness.qwen4.kvarn_reference import (
    encode_qsa_kv_tile,
    reconstruct_qsa_kv_rows,
)
from moespresso.runtime.qwen4.kvarn_encode import encode_qsa_kvarn_tile_mlx
from moespresso.runtime.qwen4.kvarn_layout import QWEN38_KVARN_K4V4_G128
from moespresso.runtime.qwen4.kvarn_selected_rows import (
    reconstruct_qsa_kvarn_selected_rows_metal,
)


def _tile(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (128, 2, 256)
    return (
        rng.normal(size=shape).astype(np.float32),
        rng.normal(size=shape).astype(np.float32),
    )


def _encode(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    dtype=mx.float32,
) -> np.ndarray:
    records = encode_qsa_kvarn_tile_mlx(
        mx.array(keys).astype(dtype),
        mx.array(values).astype(dtype),
    )
    mx.eval(records)
    return np.asarray(records)


def test_device_encoder_writes_the_complete_tight_record_deterministically() -> None:
    keys, values = _tile(501)
    first = _encode(keys, values, dtype=mx.bfloat16)
    second = _encode(keys, values, dtype=mx.bfloat16)

    assert first.shape == (
        QWEN38_KVARN_K4V4_G128.kv_heads,
        QWEN38_KVARN_K4V4_G128.head_record_bytes,
    )
    assert first.dtype == np.uint8
    assert first.nbytes == QWEN38_KVARN_K4V4_G128.tile_record_bytes
    assert np.array_equal(first, second)


def test_device_encoder_tracks_the_independent_numeric_reference() -> None:
    keys, values = _tile(502)
    keys = np.asarray(mx.array(keys).astype(mx.bfloat16).astype(mx.float32))
    values = np.asarray(mx.array(values).astype(mx.bfloat16).astype(mx.float32))
    reference = encode_qsa_kv_tile(keys, values)
    device = _encode(keys, values, dtype=mx.bfloat16)
    offsets = np.array([0, 1, 17, 64, 96, 127], dtype=np.int32)
    reference_k, reference_v = reconstruct_qsa_kv_rows(reference, offsets)
    device_k, device_v = reconstruct_qsa_kv_rows(device, offsets)

    assert np.max(np.abs(device_k - reference_k)) <= 0.01
    assert np.max(np.abs(device_v - reference_v)) <= 0.05
    reference_k_rms = np.sqrt(np.mean((reference_k - keys[offsets]) ** 2))
    device_k_rms = np.sqrt(np.mean((device_k - keys[offsets]) ** 2))
    reference_v_rms = np.sqrt(np.mean((reference_v - values[offsets]) ** 2))
    device_v_rms = np.sqrt(np.mean((device_v - values[offsets]) ** 2))
    assert abs(float(device_k_rms - reference_k_rms)) <= 2e-4
    assert abs(float(device_v_rms - reference_v_rms)) <= 2e-4


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_device_record_feeds_the_selected_row_metal_decoder() -> None:
    first = _tile(503)
    second = _tile(504)
    records = mx.stack(
        [
            encode_qsa_kvarn_tile_mlx(
                mx.array(first[0]).astype(mx.bfloat16),
                mx.array(first[1]).astype(mx.bfloat16),
            ),
            encode_qsa_kvarn_tile_mlx(
                mx.array(second[0]).astype(mx.bfloat16),
                mx.array(second[1]).astype(mx.bfloat16),
            ),
        ]
    )
    selected = mx.array([[[255, 0, 192, 192, -1, 64]]], dtype=mx.int32)
    keys, values, valid = reconstruct_qsa_kvarn_selected_rows_metal(records, selected)
    mx.eval(records, keys, values, valid)

    packed = np.asarray(records)
    expected_ids = np.array([0, 64, 192, -1, 255, -1], dtype=np.int32)
    expected_k = np.zeros((6, 2, 256), dtype=np.float32)
    expected_v = np.zeros_like(expected_k)
    for tile in (0, 1):
        lanes = np.flatnonzero((expected_ids >= 0) & (expected_ids // 128 == tile))
        decoded_k, decoded_v = reconstruct_qsa_kv_rows(
            packed[tile],
            expected_ids[lanes] % 128,
        )
        expected_k[lanes] = np.asarray(mx.array(decoded_k).astype(mx.bfloat16).astype(mx.float32))
        expected_v[lanes] = np.asarray(mx.array(decoded_v).astype(mx.bfloat16).astype(mx.float32))

    got_k = np.asarray(keys.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    got_v = np.asarray(values.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    assert np.array_equal(
        np.asarray(valid),
        np.array([[[True, True, True, False, True, False]]]),
    )
    assert np.max(np.abs(got_k - expected_k)) <= 0.0078125
    assert np.max(np.abs(got_v - expected_v)) <= 0.0078125


@pytest.mark.parametrize(
    ("keys", "values", "message"),
    [
        (
            mx.zeros((127, 2, 256), dtype=mx.bfloat16),
            mx.zeros((127, 2, 256), dtype=mx.bfloat16),
            "shape",
        ),
        (
            mx.zeros((128, 2, 256), dtype=mx.int32),
            mx.zeros((128, 2, 256), dtype=mx.int32),
            "BF16",
        ),
        (
            mx.zeros((128, 2, 256), dtype=mx.float16),
            mx.zeros((128, 2, 256), dtype=mx.bfloat16),
            "share one dtype",
        ),
    ],
)
def test_device_encoder_rejects_incompatible_inputs(
    keys: mx.array,
    values: mx.array,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        encode_qsa_kvarn_tile_mlx(keys, values)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("keys", np.nan),
        ("values", np.inf),
        ("keys", np.finfo(np.float32).max),
    ],
)
def test_device_encoder_rejects_nonfinite_or_fp16_overflow(
    target: str,
    value: float,
) -> None:
    keys = np.ones((128, 2, 256), dtype=np.float32)
    values = np.ones_like(keys)
    (keys if target == "keys" else values).reshape(-1)[0] = value

    with pytest.raises(ValueError, match="finite"):
        encode_qsa_kvarn_tile_mlx(mx.array(keys), mx.array(values))
