"""TQ format reference helpers.

These are small, pure checks for the MoEspresso-owned TQ reference code. They do not import
jang; they pin the stored-array semantics the pipeline relies on for independent evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from moespresso.correctness.tq_reference import (
    generate_random_signs,
    hadamard_inverse,
    hadamard_rotate,
    tq_decode_rows,
    unpack_tq_indices,
)


def test_unpack_tq_indices_uses_little_bits_within_each_uint32_word():
    # 2-bit values [0, 1, 2, 3] packed into the low bits of one word.
    packed = np.array([[0 | (1 << 2) | (2 << 4) | (3 << 6)]], dtype=np.uint32)
    out = unpack_tq_indices(packed, bits=2, in_features=4)
    np.testing.assert_array_equal(out, np.array([[0, 1, 2, 3]], dtype=np.uint8))


def test_unpack_tq_indices_covers_1_2_4_bit_widths():
    cases = [
        (1, [0, 1, 1, 0]),
        (2, [0, 1, 2, 3]),
        (4, [0, 5, 10, 15]),
    ]
    for bits, values in cases:
        word = 0
        for i, value in enumerate(values):
            word |= value << (i * bits)
        packed = np.array([[word]], dtype=np.uint32)
        out = unpack_tq_indices(packed, bits=bits, in_features=len(values))
        np.testing.assert_array_equal(out, np.array([values], dtype=np.uint8))


def test_hadamard_rotation_is_invertible_for_power_of_two_width():
    x = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    signs = generate_random_signs(4, seed=7)
    y = hadamard_rotate(x, signs)
    back = hadamard_inverse(y, signs)
    np.testing.assert_allclose(back, x, atol=1e-6)


def test_tq_decode_rows_returns_finite_rows_with_expected_shape():
    packed = np.array([[0 | (1 << 2) | (2 << 4) | (3 << 6)]], dtype=np.uint32)
    norms = np.array([1.5], dtype=np.float16)
    out = tq_decode_rows(packed, norms, bits=2, in_features=4, seed=42)
    assert out.shape == (1, 4)
    assert np.isfinite(out).all()


def test_tq_decode_rows_matches_jang_for_stored_sidecars():
    mx = pytest.importorskip("mlx.core")
    codebook = pytest.importorskip("jang_tools.turboquant.codebook")
    linear = pytest.importorskip("jang_tools.turboquant.linear")
    pipeline = pytest.importorskip("jang_tools.turboquant.pipeline")
    rotation = pytest.importorskip("jang_tools.turboquant.rotation")

    seed = 17
    bits = 2
    row = np.linspace(-2.0, 2.0, 16, dtype=np.float32)
    weight = np.stack([row, row[::-1]], axis=0)
    stored = linear.tq_quantize_weight(weight, bits=bits, seed=seed)

    ours = tq_decode_rows(stored["packed"], stored["norms"], bits, weight.shape[1], seed)

    cb = mx.array(np.array(codebook.compute_codebook(weight.shape[1], bits), dtype=np.float32))
    signs = rotation.generate_random_signs(weight.shape[1], seed)
    rows = []
    for packed_row, norm in zip(stored["packed"], stored["norms"], strict=True):
        idx = pipeline.unpack_bits(mx.array(packed_row), bits, weight.shape[1])
        rows.append(mx.take(cb, idx.astype(mx.uint32)) * float(norm))
    expected = rotation.hadamard_inverse(mx.stack(rows), signs)
    mx.eval(expected)

    np.testing.assert_allclose(ours, np.asarray(expected, dtype=np.float32), atol=1e-5)
