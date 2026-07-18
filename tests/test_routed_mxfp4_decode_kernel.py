from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from moespresso.probe.deepseek_v4.codec import dequant_fp4_e2m1_ue8m0  # noqa: E402
from moespresso.runtime.routed_decode_kernel import (  # noqa: E402
    make_routed_mxfp4_decode_kernel,
)


def _pack_mxfp4_source(codes: np.ndarray) -> np.ndarray:
    packed_i8 = (
        codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)
    ).astype(np.uint8).view(np.int8)
    n_exp, rows, cols2 = packed_i8.shape
    in_f = cols2 * 2
    return (
        np.ascontiguousarray(packed_i8)
        .view(np.uint8)
        .reshape(n_exp, rows, in_f // 8, 4)
        .copy()
        .view(np.uint32)
        .reshape(n_exp, rows, in_f // 8)
    ), packed_i8


def _mxfp4_tensor(rng: np.random.Generator, n_exp: int, rows: int, cols: int):
    codes = rng.integers(0, 16, (n_exp, rows, cols), dtype=np.uint8)
    packed, packed_i8 = _pack_mxfp4_source(codes)
    scales = rng.integers(126, 129, (n_exp, rows, cols // 32), dtype=np.uint8)
    dense = np.stack(
        [
            dequant_fp4_e2m1_ue8m0(packed_i8[e], scales[e], out_dtype=np.float32)
            for e in range(n_exp)
        ],
        axis=0,
    )
    return packed, scales, dense


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


@pytest.mark.parametrize("split", [1, 2, 4])
def test_fused_routed_mxfp4_decode_matches_decoded_fp4_reference(split):
    rng = np.random.default_rng(123 + split)
    in_f = 256
    out_f = 64
    k = 2

    packed_gate, scales_gate, gate_w = _mxfp4_tensor(rng, k, out_f, in_f)
    packed_up, scales_up, up_w = _mxfp4_tensor(rng, k, out_f, in_f)
    packed_down, scales_down, down_w = _mxfp4_tensor(rng, k, in_f, out_f)
    x = (rng.standard_normal((1, in_f)) * 0.05).astype(np.float32)

    expected = []
    for expert in range(k):
        gate = x @ gate_w[expert].T
        up = x @ up_w[expert].T
        gate = np.minimum(gate, 10.0)
        up = np.clip(up, -10.0, 10.0)
        act = (_silu(gate) * up).astype(np.float16).astype(np.float32)
        expected.append((act @ down_w[expert].T)[0])
    expected = np.stack(expected, axis=0)

    kernel = make_routed_mxfp4_decode_kernel(
        in_f=in_f,
        out_f=out_f,
        K=k,
        swiglu_limit=10.0,
        split=split,
    )
    assert kernel is not None
    got = kernel(
        mx.array(x),
        mx.array(packed_gate, dtype=mx.uint32),
        mx.array(scales_gate, dtype=mx.uint8),
        mx.array(packed_up, dtype=mx.uint32),
        mx.array(scales_up, dtype=mx.uint8),
        mx.array(packed_down, dtype=mx.uint32),
        mx.array(scales_down, dtype=mx.uint8),
        mx.array(np.arange(k, dtype=np.uint32)),
    )
    mx.eval(got)
    got_np = np.array(got)

    rel = np.linalg.norm(got_np - expected) / max(np.linalg.norm(expected), 1e-12)
    assert rel < 1e-5
