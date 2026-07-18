from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from moespresso.probe.deepseek_v4.codec import dequant_fp4_e2m1_ue8m0  # noqa: E402


def _pack_logical_fp4_codes(codes: np.ndarray) -> np.ndarray:
    codes_u8 = np.asarray(codes, dtype=np.uint8)
    low = codes_u8[:, 0::2]
    high = codes_u8[:, 1::2]
    return (low | (high << 4)).astype(np.uint8).view(np.int8)


def _repack_ds4_fp4_as_mxfp4_uint32(packed: np.ndarray) -> np.ndarray:
    packed_u8 = np.ascontiguousarray(packed).view(np.uint8)
    out_dim, packed_cols = packed_u8.shape
    in_dim = packed_cols * 2
    assert in_dim % 8 == 0
    return (
        packed_u8.reshape(out_dim, in_dim // 8, 4)
        .copy()
        .view(np.uint32)
        .reshape(out_dim, in_dim // 8)
    )


def test_deepseek_v4_fp4_repack_matches_mlx_mxfp4_dequantize():
    logical_codes = np.stack(
        [
            np.arange(64, dtype=np.uint8) % 16,
            15 - (np.arange(64, dtype=np.uint8) % 16),
        ],
        axis=0,
    )
    packed = _pack_logical_fp4_codes(logical_codes)
    scales = np.array([[126, 127], [128, 129]], dtype=np.uint8)

    expected = dequant_fp4_e2m1_ue8m0(
        packed,
        scales,
        fp4_block=32,
        out_dtype=np.float32,
    )
    mxfp4_words = _repack_ds4_fp4_as_mxfp4_uint32(packed)

    actual = mx.dequantize(
        mx.array(mxfp4_words, dtype=mx.uint32),
        mx.array(scales, dtype=mx.uint8),
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    actual = np.array(actual.astype(mx.float32))

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
