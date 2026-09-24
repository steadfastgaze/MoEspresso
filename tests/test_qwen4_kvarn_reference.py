from __future__ import annotations

import inspect

import mlx.core as mx
import numpy as np
import pytest

import moespresso.correctness.qwen4.kvarn_reference as kvarn_reference
from moespresso.correctness.qwen4.kvarn_reference import (
    encode_qsa_kv_tile,
    normalized_sylvester_hadamard,
    reconstruct_qsa_kv_rows,
)
from moespresso.correctness.qwen4.qsa_reference import sparse_grouped_query_attention
from moespresso.runtime.qwen4.kvarn_layout import QWEN38_KVARN_K4V4_G128


def _tile(seed: int = 38) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (128, 2, 256)
    return (
        rng.normal(scale=0.4, size=shape).astype(np.float32),
        rng.normal(scale=0.6, size=shape).astype(np.float32),
    )


def test_normalized_sylvester_hadamard_is_symmetric_and_orthonormal() -> None:
    matrix = normalized_sylvester_hadamard(256)
    identity = matrix @ matrix.T

    assert matrix.dtype == np.float32
    assert np.array_equal(matrix, matrix.T)
    assert np.allclose(identity, np.eye(256, dtype=np.float32), rtol=0, atol=2e-6)
    with pytest.raises(ValueError, match="power of two"):
        normalized_sylvester_hadamard(192)
    with pytest.raises(ValueError, match="positive integer"):
        normalized_sylvester_hadamard(True)


def test_nibble_order_is_token_major_with_even_channels_in_low_nibbles() -> None:
    codes = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=np.uint8,
    )
    packed = kvarn_reference._pack_token_major_nibbles(codes)
    restored = kvarn_reference._unpack_token_major_nibbles(packed, 6)

    assert np.array_equal(
        packed,
        np.array([[0x10, 0x32, 0x54], [0x76, 0x98, 0xBA]], dtype=np.uint8),
    )
    assert np.array_equal(restored, codes)


def test_encode_is_deterministic_and_selected_rows_match_full_reconstruction() -> None:
    keys, values = _tile()
    first = encode_qsa_kv_tile(keys, values)
    second = encode_qsa_kv_tile(keys, values)
    offsets = np.array([127, 0, 64, 0, 1], dtype=np.int32)

    selected_k, selected_v = reconstruct_qsa_kv_rows(first, offsets)
    full_k, full_v = reconstruct_qsa_kv_rows(first, np.arange(128, dtype=np.int32))

    assert first.shape == (2, 35_072)
    assert first.dtype == np.uint8
    assert first.flags.c_contiguous
    assert np.array_equal(first, second)
    assert np.array_equal(selected_k, full_k[offsets])
    assert np.array_equal(selected_v, full_v[offsets])


def test_hand_authored_record_fixes_scale_absorption_and_inverse_rotation() -> None:
    layout = QWEN38_KVARN_K4V4_G128
    records = np.zeros((2, layout.head_record_bytes), dtype=np.uint8)
    token = 7
    for head in range(2):
        record = records[head]
        k_codes = layout.field("k_codes")
        k_payload = record[k_codes.offset : k_codes.end].reshape(128, 128)
        k_payload[token] = np.uint8(0x11)
        v_codes = layout.field("v_codes")
        v_payload = record[v_codes.offset : v_codes.end].reshape(128, 128)
        v_payload[token] = np.uint8(0x22)
        kvarn_reference._write_f16(
            record,
            layout.field("k_scale").offset,
            np.full(256, 2, dtype=np.float32),
        )
        kvarn_reference._write_f16(
            record,
            layout.field("k_zero").offset,
            np.full(256, 3, dtype=np.float32),
        )
        kvarn_reference._write_f16(
            record,
            layout.field("k_token_scale").offset,
            np.full(128, 4, dtype=np.float32),
        )
        kvarn_reference._write_f16(
            record,
            layout.field("v_channel_scale").offset,
            np.full(256, 5, dtype=np.float32),
        )
        kvarn_reference._write_f16(
            record,
            layout.field("v_token_scale").offset,
            np.full(128, 3, dtype=np.float32),
        )
        kvarn_reference._write_f16(
            record,
            layout.field("v_zero").offset,
            np.full(128, 4, dtype=np.float32),
        )

    keys, values = reconstruct_qsa_kv_rows(records, np.array([token], dtype=np.int32))
    expected_k = np.zeros((1, 2, 256), dtype=np.float32)
    expected_v = np.zeros_like(expected_k)
    expected_k[..., 0] = 320.0
    expected_v[..., 0] = 800.0

    assert np.allclose(keys, expected_k, rtol=0, atol=1e-5)
    assert np.allclose(values, expected_v, rtol=0, atol=1e-5)


def test_selected_bf16_workspace_matches_full_reconstruction_attention() -> None:
    keys, values = _tile(91)
    records = encode_qsa_kv_tile(keys, values)
    full_k, full_v = reconstruct_qsa_kv_rows(records, np.arange(128, dtype=np.int32))
    selected = np.array([0, 3, 64, 65, 127], dtype=np.int32)
    selected_k, selected_v = reconstruct_qsa_kv_rows(records, selected)

    def bf16(values: np.ndarray) -> np.ndarray:
        return np.asarray(mx.array(values).astype(mx.bfloat16).astype(mx.float32))

    rng = np.random.default_rng(92)
    query = rng.normal(size=(24, 256)).astype(np.float32)
    full = sparse_grouped_query_attention(
        query,
        bf16(full_k),
        bf16(full_v),
        selected,
    )
    workspace = sparse_grouped_query_attention(
        query,
        bf16(selected_k),
        bf16(selected_v),
        np.arange(selected.size, dtype=np.int32),
    )

    assert np.array_equal(full, workspace)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((128, 2, 255), dtype=np.float32),
        np.zeros((128, 2, 256), dtype=np.int32),
        np.zeros((128, 2, 256), dtype=np.float64),
    ],
)
def test_encode_rejects_invalid_inputs(bad: np.ndarray) -> None:
    keys, values = _tile()
    with pytest.raises(ValueError):
        encode_qsa_kv_tile(bad, values)
    with pytest.raises(ValueError):
        encode_qsa_kv_tile(keys, bad)


def test_encode_rejects_nonfinite_inputs() -> None:
    keys, values = _tile()
    keys[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        encode_qsa_kv_tile(keys, values)


def test_reconstruction_rejects_malformed_records_offsets_and_metadata() -> None:
    keys, values = _tile()
    records = encode_qsa_kv_tile(keys, values)
    with pytest.raises(ValueError, match="uint8"):
        reconstruct_qsa_kv_rows(records.astype(np.int16), np.array([0]))
    with pytest.raises(ValueError, match="shape"):
        reconstruct_qsa_kv_rows(records[:, :-1], np.array([0]))
    noncontiguous = np.zeros((2, 70_144), dtype=np.uint8)[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        reconstruct_qsa_kv_rows(noncontiguous, np.array([0]))
    with pytest.raises(ValueError, match="one-dimensional integer"):
        reconstruct_qsa_kv_rows(records, np.array([False]))
    with pytest.raises(ValueError, match="outside"):
        reconstruct_qsa_kv_rows(records, np.array([-1]))
    with pytest.raises(ValueError, match="outside"):
        reconstruct_qsa_kv_rows(records, np.array([128]))

    corrupted = records.copy()
    offset = QWEN38_KVARN_K4V4_G128.field("k_scale").offset
    corrupted[0, offset : offset + 2] = np.array([np.nan], dtype="<f2").view(np.uint8)
    with pytest.raises(ValueError, match="metadata must be finite"):
        reconstruct_qsa_kv_rows(corrupted, np.array([0]))


def test_codec_api_cannot_modify_qsa_index_state() -> None:
    assert "index" not in inspect.signature(encode_qsa_kv_tile).parameters
    assert "index" not in inspect.signature(reconstruct_qsa_kv_rows).parameters
