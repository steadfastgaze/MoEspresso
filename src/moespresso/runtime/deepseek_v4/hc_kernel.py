"""Fused DS4 mHC kernels.

Each DS4 decoder layer runs two mHC stages around each block. The pre
stage splits the mixer row into pre weights, post gates, and the Sinkhorn
combination matrix, then collapses the four hidden-channel streams into
one embedding row. The post stage recombines the block output with the
residual channels through the post gates and the combination matrix. The
composed graph materializes a float32 ``[rows, 4, hidden]`` broadcast
product for each stage (about 252 MB per stage at a 3844-token prefill
chunk) plus the batched 4x4 recombine matmul output. This module fuses
each stage into a single Metal dispatch that reads the operands once and
writes only the stage outputs.

Prefill and decode share the kernels; the shapes differ only in the row
count. At the single-row decode shape the win is dispatch count (three
composed dispatches collapse to one per stage), while memory traffic is unchanged.

Both kernels are bit-identical to the composed ops they replace, enforced
bitwise by tests at prefill and decode shapes:

- The split math is a verbatim transcription of the jang
  ``hc_split_sinkhorn`` kernel (the served split path), so pre, post, and
  comb carry the same bits.
- The pre weighted sum reproduces the broadcast multiply plus ``mx.sum``
  contraction over the hc axis: products round individually and
  accumulate in ascending hc order from a zero accumulator, with fp
  contraction disabled so the compiler cannot fuse the product into the
  add.
- The post recombine reproduces the served float32 contract
  (``_patch_deepseek_v4_hc_post_float32``): the transposed 4x4 recombine
  contraction matches the MLX batched matmul at K=4, which accumulates an
  ascending fma chain from a zero accumulator, and the post-gate product
  and the final add round separately.

The embedded split routine derives from JANG v2.5.29 commit
`e0c5a81fb34a63f1547030902044a4b99d3f2345` under Apache-2.0. Modification
notice: MoEspresso incorporates the split math into fused pre and post stage
kernels and adds the decode-tail reduction path. See `THIRD-PARTY-NOTICES`
and `LICENSE-APACHE-2.0`.

The mix GEMM stays a composed MLX op at every shape. On multi-row
prefill chunks the flattened-row rsqrt chain also stays composed. At the
single-row decode shape the tail kernel absorbs the rsqrt chain and the
mixer scale into the split dispatch: the composed
``mx.mean(x_flat.square(), ...)`` at ``[1, 1, W]`` with the last axis
reduced is a ContiguousReduce that dispatches ``row_reduce_looped`` with
a 1024-thread threadgroup for ``W >= 4096``, and that order transcribes
exactly (each thread sums four sequential squared elements per block at
block stride 4096 from a zero accumulator, the same per-thread window
for the tail elements, one ``simd_sum`` per simdgroup, lane-ordered
``simd_sum`` over the 32 partials), followed by ``sum * float32(1/W)``
(the mean normalizer), ``+ rms_norm_eps``, and
``metal::precise::rsqrt``. The mixer scale rounds once per element like
the composed broadcast multiply. Measured 0/1400 mismatched trials
across seven widths including the served ``W = 16384``.

The adjacent RMS norms stay composed. A fold must reproduce
``mx.fast.rms_norm`` bit for bit: for float32 rows the reduction order
is transcribable (256 strided partial sums, binary tree combine, then
``(x * s) * w``, measured 0/4096 mismatches at hidden 4096), but
half-precision rows match no composed candidate order, and the norm
dispatches measure at the fence floor, so the fold stays unimplemented.

``MOESPRESSO_DSV4_HC_PREFILL_FUSED=0`` is the kill switch for multi-row
(prefill) shapes, ``MOESPRESSO_DSV4_HC_DECODE_FUSED=0`` for the
single-row decode shape, and ``MOESPRESSO_DSV4_HC_DECODE_TAIL=0`` for
the decode tail absorption alone; callers fall back to the composed
path on any precondition miss.
"""

from __future__ import annotations

import os

_PREFILL_ENV_FLAG = "MOESPRESSO_DSV4_HC_PREFILL_FUSED"
_DECODE_ENV_FLAG = "MOESPRESSO_DSV4_HC_DECODE_FUSED"
_DECODE_TAIL_ENV_FLAG = "MOESPRESSO_DSV4_HC_DECODE_TAIL"

_HC = 4
_MIX = (2 + _HC) * _HC
_THREADS_PER_GROUP = 256
# The tail kernel mirrors the composed reduce's threadgroup: at one row
# and W >= 4096 the ContiguousReduce dispatches 1024 threads.
_TAIL_THREADS_PER_GROUP = 1024
_TAIL_MIN_WIDTH = 4096

# One thread runs the whole 24-value split chain per row. The math below
# is transcribed verbatim from the jang hc_split_sinkhorn kernel body so
# the fused stage emits bit-identical pre/post/comb values, including the
# metal::fast::exp calls that kernel compiles to. The mixer row arrives
# in thread space so both the plain kernel (a pure copy of the device
# row) and the tail kernel (the row scaled by the in-kernel rsqrt) share
# one split implementation.
_SPLIT_HEADER_TEMPLATE = """
METAL_FUNC void moespresso_dsv4_hc_split_row(
        const thread float* mix,
        const thread float* scale,
        const thread float* base,
        thread float* pre_out,
        device float* post_out,
        device float* comb_out) {
    constexpr int HC = 4;
    const float epsv = EPSV;
    float pre_scale = static_cast<float>(scale[0]);
    float post_scale = static_cast<float>(scale[1]);
    float comb_scale = static_cast<float>(scale[2]);

    for (int i = 0; i < HC; ++i) {
        float z = static_cast<float>(mix[i]) * pre_scale
            + static_cast<float>(base[i]);
        pre_out[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
    }
    for (int i = 0; i < HC; ++i) {
        int off = HC + i;
        float z = static_cast<float>(mix[off]) * post_scale
            + static_cast<float>(base[off]);
        post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
    }

    float c[HC * HC];
    for (int i = 0; i < HC; ++i) {
        float row_max = -INFINITY;
        for (int j = 0; j < HC; ++j) {
            int cidx = i * HC + j;
            int off = 2 * HC + cidx;
            float v = static_cast<float>(mix[off]) * comb_scale
                + static_cast<float>(base[off]);
            c[cidx] = v;
            row_max = metal::max(row_max, v);
        }
        float row_sum = 0.0f;
        for (int j = 0; j < HC; ++j) {
            int cidx = i * HC + j;
            float v = metal::fast::exp(c[cidx] - row_max);
            c[cidx] = v;
            row_sum += v;
        }
        float inv_sum = 1.0f / row_sum;
        for (int j = 0; j < HC; ++j) {
            int cidx = i * HC + j;
            c[cidx] = c[cidx] * inv_sum + epsv;
        }
    }

    for (int j = 0; j < HC; ++j) {
        float col_sum = 0.0f;
        for (int i = 0; i < HC; ++i) {
            col_sum += c[i * HC + j];
        }
        float inv_denom = 1.0f / (col_sum + epsv);
        for (int i = 0; i < HC; ++i) {
            c[i * HC + j] *= inv_denom;
        }
    }

    for (int iter = 1; iter < ITERS; ++iter) {
        for (int i = 0; i < HC; ++i) {
            float row_sum = 0.0f;
            for (int j = 0; j < HC; ++j) {
                row_sum += c[i * HC + j];
            }
            float inv_denom = 1.0f / (row_sum + epsv);
            for (int j = 0; j < HC; ++j) {
                c[i * HC + j] *= inv_denom;
            }
        }
        for (int j = 0; j < HC; ++j) {
            float col_sum = 0.0f;
            for (int i = 0; i < HC; ++i) {
                col_sum += c[i * HC + j];
            }
            float inv_denom = 1.0f / (col_sum + epsv);
            for (int i = 0; i < HC; ++i) {
                c[i * HC + j] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC * HC; ++i) {
        comb_out[i] = c[i];
    }
}
"""

# One threadgroup owns one token row: thread 0 runs the split and shares
# the pre weights through threadgroup memory, then all threads collapse
# the four hidden channels with float4 loads. The accumulation matches
# the composed multiply-then-sum bit for bit (rounded products, ascending
# adds from zero, contraction off).
_PRE_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint ntg = threads_per_threadgroup.x;

    threadgroup float pre_sh[4];

    device const float* mix = mixes + (uint64_t)row * 24u;
    device float* post_row = post + (uint64_t)row * 4u;
    device float* comb_row = comb + (uint64_t)row * 16u;

    if (tid == 0u) {
        float mix_l[24];
        float scale_l[3];
        float base_l[24];
        for (int i = 0; i < 24; ++i) mix_l[i] = mix[i];
        for (int i = 0; i < 3; ++i) scale_l[i] = scale[i];
        for (int i = 0; i < 24; ++i) base_l[i] = base[i];
        float pre_vals[4];
        moespresso_dsv4_hc_split_row(
            mix_l, scale_l, base_l, pre_vals, post_row, comb_row);
        pre_sh[0] = pre_vals[0];
        pre_sh[1] = pre_vals[1];
        pre_sh[2] = pre_vals[2];
        pre_sh[3] = pre_vals[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float p0 = pre_sh[0];
    float p1 = pre_sh[1];
    float p2 = pre_sh[2];
    float p3 = pre_sh[3];

    uint hidden = (uint)x_shape[1] / 4u;
    uint d4 = hidden / 4u;
    device const float4* x0 =
        (device const float4*)(x + (uint64_t)row * (uint64_t)x_shape[1]);
    device const float4* x1 = x0 + d4;
    device const float4* x2 = x1 + d4;
    device const float4* x3 = x2 + d4;
    device float4* y4 = (device float4*)(y + (uint64_t)row * hidden);

    for (uint i = tid; i < d4; i += ntg) {
        #pragma clang fp contract(off)
        float4 t0 = x0[i] * p0;
        float4 t1 = x1[i] * p1;
        float4 t2 = x2[i] * p2;
        float4 t3 = x3[i] * p3;
        float4 acc = 0.0f;
        acc = acc + t0;
        acc = acc + t1;
        acc = acc + t2;
        acc = acc + t3;
        y4[i] = acc;
    }
"""

# Transcription of the composed sum-of-squares reduce at the single-row
# decode shape. mx.mean(x_flat.square(), axis=-1) on [1, 1, W] resolves
# to a ContiguousReduce and dispatches row_reduce_looped with a
# 1024-thread threadgroup for W >= 4096: each thread accumulates
# N_READS=4 sequential squared elements per block at block stride
# lsize.x * N_READS = 4096 from a zero accumulator, handles the tail
# elements through the same per-thread window, reduces each simdgroup
# with simd_sum, and combines the 32 simdgroup partials with a
# lane-ordered simd_sum. The square rounds per element exactly like the
# composed unary square dispatch. fp contraction is disabled so the
# product cannot fuse into the accumulate.
_TAIL_REDUCE_HEADER = """
METAL_FUNC float moespresso_dsv4_hc_row_sumsq(
        const device float* xrow,
        threadgroup float* shared_vals,
        uint tid,
        uint simd_gid,
        uint simd_lid,
        int W) {
    int blocks = W / (1024 * 4);
    int extra = W - blocks * (1024 * 4);
    const device float* in = xrow + tid * 4;
    float total = 0.0f;
    for (int b = 0; b < blocks; b++) {
        #pragma clang fp contract(off)
        for (int i = 0; i < 4; i++) {
            float v = in[i];
            total = (v * v) + total;
        }
        in += 1024 * 4;
    }
    int index = (int)tid * 4;
    if (index + 4 <= extra) {
        #pragma clang fp contract(off)
        for (int i = 0; i < 4; i++) {
            float v = in[i];
            total = (v * v) + total;
        }
    } else {
        #pragma clang fp contract(off)
        for (int i = 0; index + i < extra; i++) {
            float v = in[i];
            total = (v * v) + total;
        }
    }
    total = metal::simd_sum(total);
    if (simd_lid == 0) {
        shared_vals[simd_gid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float val = (tid < 32) ? shared_vals[tid] : 0.0f;
    return metal::simd_sum(val);
}
"""

# The decode-tail variant of the pre kernel. The mix GEMM stays composed
# and feeds the raw mixer row; the kernel computes the flattened-row
# rsqrt (transcribed reduce above, then the mean normalizer, the epsilon
# add, and metal::precise::rsqrt, each rounding at its composed op
# boundary), scales the 24 mixer values with one rounding per element
# like the composed broadcast multiply, and continues with the shared
# split chain and the channel weighted sum.
_TAIL_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint ntg = threads_per_threadgroup.x;
    uint simd_gid = tid / 32u;
    uint simd_lid = tid % 32u;

    threadgroup float shared_vals[32];
    threadgroup float pre_sh[4];

    int W = (int)x_shape[1];
    device const float* xrow = x + (uint64_t)row * (uint64_t)W;
    device const float* mix = mixes + (uint64_t)row * 24u;
    device float* post_row = post + (uint64_t)row * 4u;
    device float* comb_row = comb + (uint64_t)row * 16u;

    float sumsq = moespresso_dsv4_hc_row_sumsq(
        xrow, shared_vals, tid, simd_gid, simd_lid, W);

    if (tid == 0u) {
        #pragma clang fp contract(off)
        float mean_sq = sumsq * INVN;
        float rs = metal::precise::rsqrt(mean_sq + EPSR);
        float mix_l[24];
        for (int i = 0; i < 24; ++i) {
            mix_l[i] = static_cast<float>(mix[i]) * rs;
        }
        float scale_l[3];
        float base_l[24];
        for (int i = 0; i < 3; ++i) scale_l[i] = scale[i];
        for (int i = 0; i < 24; ++i) base_l[i] = base[i];
        float pre_vals[4];
        moespresso_dsv4_hc_split_row(
            mix_l, scale_l, base_l, pre_vals, post_row, comb_row);
        pre_sh[0] = pre_vals[0];
        pre_sh[1] = pre_vals[1];
        pre_sh[2] = pre_vals[2];
        pre_sh[3] = pre_vals[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float p0 = pre_sh[0];
    float p1 = pre_sh[1];
    float p2 = pre_sh[2];
    float p3 = pre_sh[3];

    uint hidden = (uint)x_shape[1] / 4u;
    uint d4 = hidden / 4u;
    device const float4* x0 =
        (device const float4*)(x + (uint64_t)row * (uint64_t)x_shape[1]);
    device const float4* x1 = x0 + d4;
    device const float4* x2 = x1 + d4;
    device const float4* x3 = x2 + d4;
    device float4* y4 = (device float4*)(y + (uint64_t)row * hidden);

    for (uint i = tid; i < d4; i += ntg) {
        #pragma clang fp contract(off)
        float4 t0 = x0[i] * p0;
        float4 t1 = x1[i] * p1;
        float4 t2 = x2[i] * p2;
        float4 t3 = x3[i] * p3;
        float4 acc = 0.0f;
        acc = acc + t0;
        acc = acc + t1;
        acc = acc + t2;
        acc = acc + t3;
        y4[i] = acc;
    }
"""

# One threadgroup owns one token row. Each thread computes all four
# destination hc streams for its embedding positions, reusing the block
# output and residual loads. The comb reads are transposed
# (comb[src * 4 + dst]) to match the served float32 recombine contract,
# and the ascending fma chain from a zero accumulator reproduces the MLX
# batched matmul at K=4 bit for bit.
_POST_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint ntg = threads_per_threadgroup.x;

    threadgroup float post_sh[4];
    threadgroup float comb_sh[16];

    if (tid < 4u) {
        post_sh[tid] = post[(uint64_t)row * 4u + tid];
    }
    if (tid < 16u) {
        comb_sh[tid] = comb[(uint64_t)row * 16u + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint hidden = (uint)x_shape[1];
    uint d4 = hidden / 4u;
    device const vec<TX, 4>* xr =
        (device const vec<TX, 4>*)(x + (uint64_t)row * hidden);
    device const vec<TR, 4>* r0 =
        (device const vec<TR, 4>*)(residual + (uint64_t)row * 4u * hidden);
    device const vec<TR, 4>* r1 = r0 + d4;
    device const vec<TR, 4>* r2 = r1 + d4;
    device const vec<TR, 4>* r3 = r2 + d4;
    device float4* out4 = (device float4*)(out + (uint64_t)row * 4u * hidden);

    for (uint i = tid; i < d4; i += ntg) {
        #pragma clang fp contract(off)
        float4 xv = float4(xr[i]);
        float4 v0 = float4(r0[i]);
        float4 v1 = float4(r1[i]);
        float4 v2 = float4(r2[i]);
        float4 v3 = float4(r3[i]);
        for (uint dst = 0u; dst < 4u; ++dst) {
            float c0 = comb_sh[0u * 4u + dst];
            float c1 = comb_sh[1u * 4u + dst];
            float c2 = comb_sh[2u * 4u + dst];
            float c3 = comb_sh[3u * 4u + dst];
            float4 m = metal::fma(v0, float4(c0), float4(0.0f));
            m = metal::fma(v1, float4(c1), m);
            m = metal::fma(v2, float4(c2), m);
            m = metal::fma(v3, float4(c3), m);
            float4 t = xv * post_sh[dst];
            out4[(uint64_t)dst * d4 + i] = t + m;
        }
    }
"""

_PRE_KERNELS: dict[tuple[int, float], object] = {}
_TAIL_KERNELS: dict[tuple[int, float, float, int], object] = {}
_POST_KERNEL: object | None = None

_METAL_AVAILABLE: bool | None = None


def _metal_available() -> bool:
    global _METAL_AVAILABLE
    if _METAL_AVAILABLE is None:
        try:
            import mlx.core as mx

            _METAL_AVAILABLE = bool(mx.metal.is_available())
        except ImportError:
            _METAL_AVAILABLE = False
    return _METAL_AVAILABLE


def hc_prefill_fused_enabled() -> bool:
    """Return True when the fused mHC path may engage on multi-row shapes."""
    if os.environ.get(_PREFILL_ENV_FLAG, "1") == "0":
        return False
    return _metal_available()


def hc_decode_fused_enabled() -> bool:
    """Return True when the fused mHC path may engage on single-row shapes."""
    if os.environ.get(_DECODE_ENV_FLAG, "1") == "0":
        return False
    return _metal_available()


def hc_decode_tail_enabled() -> bool:
    """Return True when the decode tail absorption may engage.

    The tail rides the fused decode split kernel, so it delegates to the
    decode gate: with ``MOESPRESSO_DSV4_HC_DECODE_FUSED=0`` the tail is
    off regardless of its own flag.
    """
    if os.environ.get(_DECODE_TAIL_ENV_FLAG, "1") == "0":
        return False
    return hc_decode_fused_enabled()


def hc_fused_enabled() -> bool:
    """Return True when either the prefill or the decode gate is open."""
    return hc_prefill_fused_enabled() or hc_decode_fused_enabled()


def _rows_enabled(rows: int) -> bool:
    """Route the row count to its phase gate; zero rows fail closed."""
    if rows < 1:
        return False
    if rows == 1:
        return hc_decode_fused_enabled()
    return hc_prefill_fused_enabled()


def _f32_hex(value: float) -> str:
    import numpy as np

    return float(np.float32(value)).hex() + "f"


def _element_dtypes():
    import mlx.core as mx

    return (mx.float32, mx.float16, mx.bfloat16)


def _get_pre_kernel(iters: int, eps: float):
    key = (int(iters), float(eps))
    kernel = _PRE_KERNELS.get(key)
    if kernel is None:
        import mlx.core as mx

        header = (
            _SPLIT_HEADER_TEMPLATE
            .replace("EPSV", _f32_hex(eps))
            .replace("ITERS", str(int(iters)))
        )
        kernel = mx.fast.metal_kernel(
            name=f"moespresso_dsv4_hc_split_weighted_sum_{len(_PRE_KERNELS)}",
            input_names=["mixes", "scale", "base", "x"],
            output_names=["y", "post", "comb"],
            source=_PRE_SOURCE,
            header=header,
        )
        _PRE_KERNELS[key] = kernel
    return kernel


def _get_tail_kernel(iters: int, eps: float, rms_eps: float, width: int):
    key = (int(iters), float(eps), float(rms_eps), int(width))
    kernel = _TAIL_KERNELS.get(key)
    if kernel is None:
        import mlx.core as mx

        header = (
            _SPLIT_HEADER_TEMPLATE
            .replace("EPSV", _f32_hex(eps))
            .replace("ITERS", str(int(iters)))
            + _TAIL_REDUCE_HEADER
        )
        # The mean normalizer matches mx.mean's number_of_elements
        # contract: 1/W computed in double precision, then rounded to
        # float32.
        source = (
            _TAIL_SOURCE
            .replace("INVN", _f32_hex(1.0 / float(width)))
            .replace("EPSR", _f32_hex(rms_eps))
        )
        kernel = mx.fast.metal_kernel(
            name=(
                "moespresso_dsv4_hc_split_weighted_sum_tail_"
                f"{len(_TAIL_KERNELS)}"
            ),
            input_names=["mixes", "scale", "base", "x"],
            output_names=["y", "post", "comb"],
            source=source,
            header=header,
        )
        _TAIL_KERNELS[key] = kernel
    return kernel


def _get_post_kernel():
    global _POST_KERNEL
    if _POST_KERNEL is None:
        import mlx.core as mx

        _POST_KERNEL = mx.fast.metal_kernel(
            name="moespresso_dsv4_hc_post_recombine",
            input_names=["x", "residual", "post", "comb"],
            output_names=["out"],
            source=_POST_SOURCE,
        )
    return _POST_KERNEL


def hc_split_weighted_sum_eligible(x, fn, scale, base, *, hc_mult, iters) -> bool:
    """Return True when the layer's hc pre stage can run the fused kernel.

    Eligibility is the DS4 hc layout: ``x`` is ``[batch, tokens, 4,
    hidden]`` with a float4-aligned hidden size, the mixer weight is
    float32 (so the mix GEMM emits float32 rows), and the scale and base
    vectors carry the split layout the kernel bakes in. Single-row decode
    shapes engage under the decode gate, multi-row prefill shapes under
    the prefill gate. Anything else stays on the composed path.
    """
    import mlx.core as mx

    if int(hc_mult) != _HC or int(iters) < 1:
        return False
    if x.ndim != 4 or int(x.shape[2]) != _HC:
        return False
    hidden = int(x.shape[3])
    if hidden <= 0 or hidden % 4 != 0:
        return False
    if not _rows_enabled(int(x.shape[0]) * int(x.shape[1])):
        return False
    if x.dtype not in _element_dtypes():
        return False
    if fn.ndim != 2 or tuple(int(v) for v in fn.shape) != (_MIX, _HC * hidden):
        return False
    if fn.dtype != mx.float32:
        return False
    if scale.size != 3 or scale.dtype != mx.float32:
        return False
    if base.size != _MIX or base.dtype != mx.float32:
        return False
    return True


def hc_split_weighted_sum(mixes, x_flat, scale, base, *, iters, eps):
    """Run the fused hc split plus channel weighted sum in one dispatch.

    Args:
        mixes: float32 ``[..., 24]`` mixer rows, the mix GEMM output
            already scaled by the flattened-row rsqrt.
        x_flat: float32 ``[..., 4 * hidden]`` flattened hidden-channel
            rows (the same buffer the composed weighted sum reads).
        scale: float32 ``[3]`` pre/post/comb scales.
        base: float32 ``[24]`` mixer bias.
        iters: Sinkhorn iteration count (compiled into the kernel).
        eps: Sinkhorn epsilon (compiled into the kernel).

    Returns:
        ``(y, post, comb)``: float32 ``[..., hidden]`` collapsed rows,
        float32 ``[..., 4]`` post gates, and float32 ``[..., 4, 4]``
        combination matrices, bit-identical to the composed
        ``hc_split_sinkhorn`` plus multiply-and-sum path.
    """
    import mlx.core as mx

    if mixes.ndim < 2 or int(mixes.shape[-1]) != _MIX:
        raise ValueError("mixes must have shape [..., 24]")
    if x_flat.ndim != mixes.ndim or x_flat.shape[:-1] != mixes.shape[:-1]:
        raise ValueError("x_flat leading dimensions must match mixes")
    flat_width = int(x_flat.shape[-1])
    if flat_width <= 0 or flat_width % (4 * _HC) != 0:
        raise ValueError("x_flat width must be a positive multiple of 16")
    if mixes.dtype != mx.float32 or x_flat.dtype != mx.float32:
        raise ValueError("mixes and x_flat must be float32")
    if scale.size != 3 or base.size != _MIX:
        raise ValueError("scale must carry 3 values and base 24")
    if scale.dtype != mx.float32 or base.dtype != mx.float32:
        raise ValueError("scale and base must be float32")
    if int(iters) < 1:
        raise ValueError("the split kernel requires at least one iteration")

    hidden = flat_width // _HC
    lead = mixes.shape[:-1]
    rows = 1
    for dim in lead:
        rows *= int(dim)
    if rows < 1:
        raise ValueError("mixes must contain at least one row")
    y, post, comb = _get_pre_kernel(int(iters), float(eps))(
        inputs=[
            mixes.reshape(rows, _MIX),
            scale.reshape(3),
            base.reshape(_MIX),
            x_flat.reshape(rows, flat_width),
        ],
        output_shapes=[(rows, hidden), (rows, _HC), (rows, _HC, _HC)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(rows * _THREADS_PER_GROUP, 1, 1),
        threadgroup=(_THREADS_PER_GROUP, 1, 1),
    )
    return (
        y.reshape(*lead, hidden),
        post.reshape(*lead, _HC),
        comb.reshape(*lead, _HC, _HC),
    )


def hc_split_weighted_sum_tail_eligible(
        x, fn, scale, base, *, hc_mult, iters) -> bool:
    """Return True when the decode tail absorption can run.

    On top of the split eligibility this requires the single-row decode
    shape, the tail gate, and a flattened width of at least 4096 so the
    composed reduce's 1024-thread dispatch shape holds. Anything else
    falls back to the fused split with the composed rsqrt tail.
    """
    if not hc_split_weighted_sum_eligible(
            x, fn, scale, base, hc_mult=hc_mult, iters=iters):
        return False
    if int(x.shape[0]) * int(x.shape[1]) != 1:
        return False
    if _HC * int(x.shape[3]) < _TAIL_MIN_WIDTH:
        return False
    return hc_decode_tail_enabled()


def hc_split_weighted_sum_tail(mixes_raw, x_flat, scale, base, *,
                               iters, eps, rms_eps):
    """Run the fused decode tail plus split plus weighted sum.

    Args:
        mixes_raw: float32 ``[..., 24]`` raw mix GEMM output, before the
            flattened-row rsqrt scale.
        x_flat: float32 ``[..., 4 * hidden]`` flattened hidden-channel
            rows, exactly one row.
        scale: float32 ``[3]`` pre/post/comb scales.
        base: float32 ``[24]`` mixer bias.
        iters: Sinkhorn iteration count (compiled into the kernel).
        eps: Sinkhorn epsilon (compiled into the kernel).
        rms_eps: rsqrt epsilon (compiled into the kernel).

    Returns:
        ``(y, post, comb)`` bit-identical to the composed rsqrt chain,
        mixer scale, ``hc_split_sinkhorn``, and multiply-and-sum path.
    """
    import mlx.core as mx

    if mixes_raw.ndim < 2 or int(mixes_raw.shape[-1]) != _MIX:
        raise ValueError("mixes_raw must have shape [..., 24]")
    if x_flat.ndim != mixes_raw.ndim or x_flat.shape[:-1] != mixes_raw.shape[:-1]:
        raise ValueError("x_flat leading dimensions must match mixes_raw")
    flat_width = int(x_flat.shape[-1])
    if flat_width % (4 * _HC) != 0 or flat_width < _TAIL_MIN_WIDTH:
        raise ValueError(
            "x_flat width must be a multiple of 16 and at least "
            f"{_TAIL_MIN_WIDTH}")
    if mixes_raw.dtype != mx.float32 or x_flat.dtype != mx.float32:
        raise ValueError("mixes_raw and x_flat must be float32")
    if scale.size != 3 or base.size != _MIX:
        raise ValueError("scale must carry 3 values and base 24")
    if scale.dtype != mx.float32 or base.dtype != mx.float32:
        raise ValueError("scale and base must be float32")
    if int(iters) < 1:
        raise ValueError("the split kernel requires at least one iteration")

    hidden = flat_width // _HC
    lead = mixes_raw.shape[:-1]
    rows = 1
    for dim in lead:
        rows *= int(dim)
    if rows != 1:
        raise ValueError("the tail kernel requires exactly one row")
    y, post, comb = _get_tail_kernel(
        int(iters), float(eps), float(rms_eps), flat_width)(
        inputs=[
            mixes_raw.reshape(rows, _MIX),
            scale.reshape(3),
            base.reshape(_MIX),
            x_flat.reshape(rows, flat_width),
        ],
        output_shapes=[(rows, hidden), (rows, _HC), (rows, _HC, _HC)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(rows * _TAIL_THREADS_PER_GROUP, 1, 1),
        threadgroup=(_TAIL_THREADS_PER_GROUP, 1, 1),
    )
    return (
        y.reshape(*lead, hidden),
        post.reshape(*lead, _HC),
        comb.reshape(*lead, _HC, _HC),
    )


def hc_post_recombine_eligible(x, residual, post, comb) -> bool:
    """Return True when the hc post stage can run the fused kernel.

    Eligibility mirrors the served float32 recombine contract: ``x`` is
    ``[batch, tokens, hidden]``, ``residual`` is ``[batch, tokens, 4,
    hidden]``, and the split outputs are the float32 ``[batch, tokens,
    4]`` gates and ``[batch, tokens, 4, 4]`` matrices the split stage
    emits. Single-row decode shapes engage under the decode gate,
    multi-row prefill shapes under the prefill gate.
    """
    import mlx.core as mx

    if x.ndim != 3 or residual.ndim != 4 or post.ndim != 3 or comb.ndim != 4:
        return False
    batch, tokens, hidden = (int(v) for v in x.shape)
    if hidden <= 0 or hidden % 4 != 0 or not _rows_enabled(batch * tokens):
        return False
    if tuple(int(v) for v in residual.shape) != (batch, tokens, _HC, hidden):
        return False
    if tuple(int(v) for v in post.shape) != (batch, tokens, _HC):
        return False
    if tuple(int(v) for v in comb.shape) != (batch, tokens, _HC, _HC):
        return False
    if x.dtype not in _element_dtypes() or residual.dtype not in _element_dtypes():
        return False
    if post.dtype != mx.float32 or comb.dtype != mx.float32:
        return False
    return True


def hc_post_recombine(x, residual, post, comb):
    """Run the fused hc post recombine in one dispatch.

    Computes the served float32 contract
    ``post[..., None] * x[..., None, :] + comb.T @ residual`` with the
    combination matrix read transposed, returning float32
    ``[..., 4, hidden]`` bit-identical to the composed broadcast multiply,
    batched matmul, and add. Half and bfloat16 inputs widen in-kernel,
    which is exact and matches the composed ``astype`` casts.
    """
    import mlx.core as mx

    if x.ndim < 2:
        raise ValueError("x must have shape [..., hidden]")
    hidden = int(x.shape[-1])
    if hidden <= 0 or hidden % 4 != 0:
        raise ValueError("hidden size must be a positive multiple of 4")
    lead = x.shape[:-1]
    if residual.shape[: x.ndim - 1] != lead or residual.shape[-2:] != (_HC, hidden):
        raise ValueError("residual must have shape [..., 4, hidden]")
    if post.shape != (*lead, _HC):
        raise ValueError("post must have shape [..., 4]")
    if comb.shape != (*lead, _HC, _HC):
        raise ValueError("comb must have shape [..., 4, 4]")
    if x.dtype not in _element_dtypes() or residual.dtype not in _element_dtypes():
        raise ValueError("x and residual must be float32, float16, or bfloat16")
    if post.dtype != mx.float32 or comb.dtype != mx.float32:
        raise ValueError("post and comb must be float32")

    rows = 1
    for dim in lead:
        rows *= int(dim)
    if rows < 1:
        raise ValueError("x must contain at least one row")
    out, = _get_post_kernel()(
        inputs=[
            x.reshape(rows, hidden),
            residual.reshape(rows, _HC * hidden),
            post.reshape(rows, _HC),
            comb.reshape(rows, _HC * _HC),
        ],
        template=[("TX", x.dtype), ("TR", residual.dtype)],
        output_shapes=[(rows, _HC, hidden)],
        output_dtypes=[mx.float32],
        grid=(rows * _THREADS_PER_GROUP, 1, 1),
        threadgroup=(_THREADS_PER_GROUP, 1, 1),
    )
    return out.reshape(*lead, _HC, hidden)
