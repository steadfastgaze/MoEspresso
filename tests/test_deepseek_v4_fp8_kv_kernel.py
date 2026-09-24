from __future__ import annotations

import numpy as np
import pytest

from moespresso.runtime.deepseek_v4 import fp8_kv_kernel


def _require_metal(mx):
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")


@pytest.mark.parametrize(
    "case", ["normal", "large", "tiny", "zeros", "mixed_rows"]
)
def test_fp8_kv_prefix_rows_bit_identical_to_composed(case):
    mx = pytest.importorskip("mlx.core")
    _require_metal(mx)
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_fp8_kv_roundtrip_composed,
    )

    rng = np.random.default_rng(21)
    if case == "normal":
        rows = rng.standard_normal((97, 512)).astype(np.float32) * 0.05
    elif case == "large":
        rows = rng.standard_normal((97, 512)).astype(np.float32) * 1.0e4
    elif case == "tiny":
        rows = rng.standard_normal((97, 512)).astype(np.float32) * 1.0e-6
    elif case == "zeros":
        rows = np.zeros((5, 512), dtype=np.float32)
    else:
        rows = np.concatenate(
            [
                rng.standard_normal((32, 512)).astype(np.float32) * scale
                for scale in (1.0, 1.0e-8, 1.0e6)
            ]
        )
    x = mx.array(rows.reshape(1, -1, 512))
    mx.eval(x)

    got = fp8_kv_kernel.fp8_kv_prefix_rows(x)
    expected = _deepseek_v4_fp8_kv_roundtrip_composed(
        x, head_dim=512, rot_dim=64)
    mx.eval(got, expected)

    assert got.dtype == mx.float32
    got_bits = np.asarray(got, dtype=np.float32).view(np.uint32)
    expected_bits = np.asarray(expected, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(got_bits, expected_bits)


def test_fp8_kv_roundtrip_routes_to_kernel_when_eligible(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    _require_metal(mx)
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_fp8_kv_roundtrip,
    )

    x = mx.array(
        np.random.default_rng(3).standard_normal((1, 9, 512)).astype(np.float32)
    )
    mx.eval(x)
    calls = []
    original = fp8_kv_kernel.fp8_kv_prefix_rows

    def spy(array):
        calls.append(array.shape)
        return original(array)

    monkeypatch.setattr(fp8_kv_kernel, "fp8_kv_prefix_rows", spy)
    mx.eval(_deepseek_v4_fp8_kv_roundtrip(x))
    assert calls == [(1, 9, 512)]

    mx.eval(_deepseek_v4_fp8_kv_roundtrip(x.astype(mx.float16)))
    assert calls == [(1, 9, 512)]
