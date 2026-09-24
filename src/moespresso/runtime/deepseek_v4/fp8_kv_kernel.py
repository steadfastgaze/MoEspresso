"""Metal kernel for the DeepSeek-V4 compressed-KV FP8 round trip."""

from __future__ import annotations

import math

from moespresso.runtime.deepseek_v4.indexer_score_kernel import _f32_hex


_HEAD_DIM = 512
_THREADS = 128
_LN2 = math.log(2.0)
_KERNEL: object | None = None


def _e4m3fn_table_source() -> str:
    from moespresso.runtime.deepseek_v4.model import _DEEPSEEK_V4_E4M3FN_VALUES

    values = ", ".join(_f32_hex(value) for value in _DEEPSEEK_V4_E4M3FN_VALUES)
    return (
        "constant float MOESPRESSO_DSV4_E4M3FN_VALUES[127] = {"
        + values
        + "};\n"
    )


def _header() -> str:
    return (
        _e4m3fn_table_source()
        + """
METAL_FUNC float moespresso_dsv4_fp8_scale(float amax) {
    amax = metal::isnan(amax) ? amax : (amax > FP8_AMAX_FLOOR ? amax : FP8_AMAX_FLOOR);
    float log2_scale = metal::precise::log(amax / 448.0f) / LN2;
    return metal::precise::exp(metal::ceil(log2_scale) * LN2);
}

METAL_FUNC float moespresso_dsv4_fp8_roundtrip(float x, float scale) {
    float n = x / scale;
    n = (metal::isnan(n) || n > -448.0f) ? n : -448.0f;
    n = (metal::isnan(n) || n < 448.0f) ? n : 448.0f;
    float absn = metal::fabs(n);
    float best = metal::fabs(absn - MOESPRESSO_DSV4_E4M3FN_VALUES[0]);
    float qv = MOESPRESSO_DSV4_E4M3FN_VALUES[0];
    for (ushort k = 1; k < 127; k++) {
        float d = metal::fabs(absn - MOESPRESSO_DSV4_E4M3FN_VALUES[k]);
        if (d < best) {
            best = d;
            qv = MOESPRESSO_DSV4_E4M3FN_VALUES[k];
        }
    }
    float sign = n < 0.0f ? -1.0f : (n > 0.0f ? 1.0f : 0.0f);
    return (sign * qv) * scale;
}
"""
        .replace("FP8_AMAX_FLOOR", _f32_hex(1.0e-4))
        .replace("LN2", _f32_hex(_LN2))
    )


_SOURCE = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x * 4u + sg;
    uint n_rows = (uint)x_shape[0];

    if (row >= n_rows) return;

    device const float *xrow = x + (uint64_t)row * 512u;
    device float *orow = out + (uint64_t)row * 512u;

    float xv[16];
    for (ushort j = 0; j < 16; j++) {
        xv[j] = xrow[16u * lane + j];
    }
    if (lane >= 28u) {
        for (ushort j = 0; j < 16; j++) {
            orow[16u * lane + j] = xv[j];
        }
        return;
    }
    float amax = 0.0f;
    for (ushort j = 0; j < 16; j++) {
        amax = metal::max(amax, metal::fabs(xv[j]));
    }
    amax = metal::max(amax, simd_shuffle_xor(amax, 1));
    amax = metal::max(amax, simd_shuffle_xor(amax, 2));
    float scale = moespresso_dsv4_fp8_scale(amax);
    for (ushort j = 0; j < 16; j++) {
        orow[16u * lane + j] = moespresso_dsv4_fp8_roundtrip(xv[j], scale);
    }
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import mlx.core as mx

        _KERNEL = mx.fast.metal_kernel(
            name="moespresso_dsv4_fp8_kv_prefix_rows",
            input_names=["x"],
            output_names=["out"],
            source=_SOURCE,
            header=_header(),
        )
    return _KERNEL


def fp8_kv_prefix_rows(x):
    """Apply E4M3FN round-trip semantics to 512-wide float32 KV rows."""
    import mlx.core as mx

    if x.ndim < 1 or int(x.shape[-1]) != _HEAD_DIM:
        raise ValueError("fp8 rows expect 512-wide rows")
    if x.dtype != mx.float32:
        raise ValueError("fp8 rows must be float32")
    rows = x.reshape(-1, _HEAD_DIM)
    row_count = int(rows.shape[0])
    if row_count == 0:
        raise ValueError("fp8 rows expect at least one row")
    groups = (row_count + 3) // 4
    out, = _kernel()(
        inputs=[rows],
        output_shapes=[(row_count, _HEAD_DIM)],
        output_dtypes=[mx.float32],
        grid=(_THREADS * groups, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    return out.reshape(x.shape)
