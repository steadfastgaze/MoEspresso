from __future__ import annotations

import numpy as np
import pytest

from moespresso.correctness.deepseek_v4.attention_output_replay import (
    _q8_0_ds4_activation_matmul,
    _q8_0_weight_rows,
    _quantize_q8_0_activation_np,
)


def _wire_row(scale: float, qs: np.ndarray) -> np.ndarray:
    raw = np.empty((34,), dtype=np.uint8)
    raw[:2] = np.array([scale], dtype=np.float16).view(np.uint8)
    raw[2:] = qs.astype(np.int8).view(np.uint8)
    return raw


def test_q8_0_weight_rows_reads_inline_fp16_scales() -> None:
    qs = np.arange(-16, 16, dtype=np.int8)
    wire = _wire_row(0.25, qs)[None, :]

    got_qs, got_scales = _q8_0_weight_rows(wire)

    assert got_qs.shape == (1, 1, 32)
    assert got_scales.shape == (1, 1)
    assert np.array_equal(got_qs[0, 0], qs)
    assert float(got_scales[0, 0]) == pytest.approx(0.25)


def test_q8_0_ds4_activation_matmul_matches_scalar_contract() -> None:
    x = np.linspace(-1.5, 1.5, 32, dtype=np.float32)[None, :]
    qs0 = np.arange(-16, 16, dtype=np.int8)
    qs1 = np.arange(15, -17, -1, dtype=np.int8)
    wire = np.stack([_wire_row(0.5, qs0), _wire_row(0.125, qs1)])

    xq, xscale = _quantize_q8_0_activation_np(x)
    expected0 = np.sum(qs0.astype(np.int32) * xq[0, 0].astype(np.int32))
    expected1 = np.sum(qs1.astype(np.int32) * xq[0, 0].astype(np.int32))
    expected = np.array(
        [[expected0 * 0.5 * xscale[0, 0], expected1 * 0.125 * xscale[0, 0]]],
        dtype=np.float32,
    )

    got = _q8_0_ds4_activation_matmul(x, wire)

    assert np.allclose(got, expected, rtol=0, atol=1e-6)


def test_q8_0_activation_rejects_non_block_width() -> None:
    with pytest.raises(ValueError, match="divisible by 32"):
        _quantize_q8_0_activation_np(np.ones((1, 31), dtype=np.float32))
