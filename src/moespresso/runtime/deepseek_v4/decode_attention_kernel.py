"""Fused DS4 ratio-4 decode attention island.

The ratio-4 decode attention step is launch-bound: the composed graph issues
roughly sixty small dispatches per layer (query/kv partial rope, the FP8 KV
round trip, indexer QAT, scoring, top-k selection, the selected-row gather
and concat, SDPA, and the inverse rope), and the measured layer cost is
dominated by serialized kernel launches rather than arithmetic. This module
replaces that chain with two Metal dispatches per layer:

- ``fused_decode_prep`` (one 1024-thread threadgroup): applies the bit-exact
  e2m1 indexer QAT to the post-rope indexer queries, scores every pooled row
  (same accumulation structure as the composed score chain), selects the
  top-k rows with a deterministic radix select, and prepares the new local
  KV row (partial rope on the 64-dim tail plus the bit-exact E4M3FN round
  trip on the leading 448 dims).
- ``fused_decode_sdpa`` (one threadgroup per query head): applies the
  partial rope to the query heads, runs online-softmax attention in float32
  over the local window (with the prepared row substituted at the cache
  write slot) plus the selected pooled rows, joins the per-head sink in the
  max and the denominator exactly once, and applies the inverse partial
  rope to the float16 output.

The Q8 projections, the RMS norms, the compressor calls, and the pooled QAT
cache maintenance stay as MLX ops; the wrapper that wires these kernels into
the served layer lives in ``moespresso.runtime.deepseek_v4.model``.

Numerical contracts:

- The indexer QAT reuses the verified bit-exact Metal transcription from
  ``indexer_score_kernel`` (the DS4 e2m1 lattice contract). The QAT'd query
  heads are staged in threadgroup memory as float16; e2m1 lattice points are
  exact in float16 except at denormal group scales, where the staging loses
  bits below 2**-24 (absolute score error around 1e-37, far below any top-k
  boundary).
- The FP8 KV round trip is a bit-exact transcription of
  ``_deepseek_v4_fp8_kv_roundtrip``: ``metal::precise::log``/``exp`` for the
  power-of-two scale (matching MLX's Log and Exp kernels), the E4M3FN table
  baked as float32 hex literals, first-wins nearest selection matching
  ``mx.argmin``, and the Maximum/Minimum NaN semantics of the clip.
- Partial rope reproduces the composed float16 op sequence exactly: theta in
  float32 (``position * inv_freq``), ``metal::precise::cos``/``sin`` rounded
  to float16, then the rotation as float multiplies and adds of float16
  operands rounded to float16 at every composed-op boundary. A float32
  multiply or add of two float16 values rounded once to float16 equals the
  native float16 op (float32 carries more than 2*11+2 significand bits, so
  the double rounding is innocuous), which makes the prepared KV row
  bit-identical to the composed path and keeps the cache contents exact.
- Attention math accumulates in float32; the sink joins the softmax max and
  denominator once, mirroring the two-pass sink semantics of the fused SDPA
  kernels.
- Top-k selection is deterministic: rows scoring strictly above the k-th
  value are taken in ascending row order, and ties at the boundary are
  broken by ascending row index. ``mx.argpartition`` leaves boundary-tie
  membership undefined, so the selected set matches the composed path
  whenever the boundary is tie-free, which the parity harness checks.
"""

from __future__ import annotations

import math

from moespresso.runtime.deepseek_v4.indexer_score_kernel import (
    _f32_hex,
    _metal_available,
    _qat_header,
)

# Gate for the fused ratio-4 decode attention island. Default off, for two
# measured reasons. First, the launch-bound premise does not hold at the
# steady decode state: on a float16-seeded fenced layer A/B (layer 2, 961
# pooled rows, full window, primed pool caches, alternating 5x10 blocks)
# the island measured 1.00x against the dequant-era layer (2.733 ms
# composed vs 2.747 ms fused) and 0.89x once the q8_0 decode QMV fast path
# removed the wo dequant traffic (1.235 vs 1.386 ms); the whole composed
# attention chain the island replaces costs ~0.15 ms in the unfenced graph,
# while the island's two dispatches (amortized 0.077 + 0.054 ms) plus the
# wrapper's per-call Python and kernel-invocation overhead outweigh that.
# Second, the served decode chain carries float32 activations and a float32
# local KV window with bfloat16 projection outputs, so the float16-only
# shape predicate never engages on the served path; enabling the island
# requires dtype variants on both kernels. The kernels are
# correctness-complete on the float16 contract (identical top-k sets,
# bit-exact prepared KV rows, SDPA output within float16 tolerance) and the
# parity tests keep them honest for a future coarser island that also
# absorbs the projection seams.
_ENABLED = False

_HEAD_DIM = 512
_ROPE_DIM = 64
_INDEX_HEAD_DIM = 128
_MAX_HEADS = 64

_PREP_THREADS = 1024
_SDPA_THREADS = 128

_LN2 = math.log(2.0)


def _e4m3fn_table_source() -> str:
    """The 127 non-negative E4M3FN values as float32 hex literals."""
    from moespresso.runtime.deepseek_v4.model import _DEEPSEEK_V4_E4M3FN_VALUES

    values = ", ".join(_f32_hex(v) for v in _DEEPSEEK_V4_E4M3FN_VALUES)
    return (
        "constant float MOESPRESSO_DSV4_E4M3FN_VALUES[127] = {" + values + "};\n"
    )


def _prep_header() -> str:
    """QAT helper plus the rope/FP8 helpers shared by both kernels."""
    return (
        _qat_header()
        + _e4m3fn_table_source()
        + """
// Monotone bit mapping from float32 to uint32: preserves score order for
// the radix select (negatives inverted, positives offset).
METAL_FUNC uint moespresso_dsv4_score_key(float s) {
    uint u = as_type<uint>(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One composed-op-boundary float16 rounding step: a float32 multiply or
// add/sub of float16 operands rounded once to float16 equals the native
// float16 op (innocuous double rounding), so these mirror MLX's separate
// Multiply/Subtract/Add kernels bit for bit.
METAL_FUNC half moespresso_dsv4_h_mul(half a, half b) {
    return half(float(a) * float(b));
}
METAL_FUNC half moespresso_dsv4_h_sub(half a, half b) {
    return half(float(a) - float(b));
}
METAL_FUNC half moespresso_dsv4_h_add(half a, half b) {
    return half(float(a) + float(b));
}

// DS4 partial-rope rotation of one (even, odd) float16 pair at float16 op
// semantics. cos/sin arrive as float16 (the composed path casts the float32
// tables to the activation dtype before rotating).
METAL_FUNC void moespresso_dsv4_rope_pair(
    half x0, half x1, half c, half s, thread half* out0, thread half* out1) {
    half m0 = moespresso_dsv4_h_mul(x0, c);
    half m1 = moespresso_dsv4_h_mul(x1, s);
    half m2 = moespresso_dsv4_h_mul(x0, s);
    half m3 = moespresso_dsv4_h_mul(x1, c);
    *out0 = moespresso_dsv4_h_sub(m0, m1);
    *out1 = moespresso_dsv4_h_add(m2, m3);
}

// Bit-exact transcription of _deepseek_v4_fp8_kv_roundtrip for one float
// value given its 64-element block scale. The scale derivation mirrors
// mx.log/mx.ceil/mx.exp (metal::precise variants match MLX's kernels), the
// clip mirrors mx.clip's Maximum-then-Minimum NaN passthrough, and the
// nearest-table selection mirrors mx.argmin's first-wins tie break.
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


# Phase structure of the prep kernel (one 1024-thread threadgroup, no early
# returns so every barrier is uniform):
#   A. Each simdgroup QATs indexer query heads into the shared half buffer;
#      the last simdgroup also prepares the new local KV row (rope tail +
#      FP8 round trip) into row_out.
#   B. Each simdgroup scores pooled rows strided by simdgroup count with the
#      slice-A accumulation structure, writing float32 scores to device.
#   C. Four-pass radix select over the monotone score keys finds the k-th
#      value threshold, then a blocked prefix scan emits the selected row
#      ids deterministically (strictly-above rows in ascending row order,
#      boundary ties by ascending row index).
_PREP_SOURCE_TEMPLATE = """
    constexpr uint NT = 1024u;
    constexpr uint NSG = NT / 32u;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    uint n_heads = (uint)q_shape[0];
    uint n_rows = (uint)pooled_shape[0];

    threadgroup half qbuf[8192];
    threadgroup atomic_uint hist[256];
    threadgroup uint scan_buf[NT];
    threadgroup uint sc[2];

    // Phase A: QAT the indexer query heads into threadgroup memory. e2m1
    // lattice points are exact in float16 outside denormal scales.
    for (uint h = sg; h < n_heads; h += NSG) {
        device const T *qrow = q + (uint64_t)h * 128u;
        float4 v = float4(
            float(qrow[4u * lane + 0u]),
            float(qrow[4u * lane + 1u]),
            float(qrow[4u * lane + 2u]),
            float(qrow[4u * lane + 3u]));
        float4 qq = moespresso_dsv4_indexer_qat128(v, ushort(lane));
        threadgroup half4 *q4 = (threadgroup half4 *)(qbuf + (uint64_t)h * 128u);
        q4[lane] = half4(qq);
    }

    // Phase A2 (last simdgroup): prepare the new local KV row. Lane l holds
    // dims [16l, 16l + 16). Dims below 448 take the FP8 round trip over
    // 64-element blocks (blocks span aligned 4-lane groups, reduced with
    // simd_shuffle_xor); the 64-dim tail takes the partial rope. The rope
    // reads pre-rope values and the FP8 path reads dims the rope never
    // touches, matching the composed order (rope, then round trip).
    if (sg == NSG - 1u) {
        float xv[16];
        for (ushort j = 0; j < 16; j++) {
            xv[j] = float(kv[16u * lane + j]);
        }
        float amax = 0.0f;
        for (ushort j = 0; j < 16; j++) {
            amax = metal::max(amax, metal::fabs(xv[j]));
        }
        amax = metal::max(amax, simd_shuffle_xor(amax, 1));
        amax = metal::max(amax, simd_shuffle_xor(amax, 2));
        if (lane < 28u) {
            float scale = moespresso_dsv4_fp8_scale(amax);
            for (ushort j = 0; j < 16; j++) {
                row_out[16u * lane + j] =
                    half(moespresso_dsv4_fp8_roundtrip(xv[j], scale));
            }
        } else {
            int pos = params[0];
            for (ushort j = 0; j < 8; j++) {
                uint d0 = 16u * lane + 2u * j;
                uint p = (d0 - 448u) / 2u;
                float theta = float(pos) * inv_freq[p];
                half c = half(metal::precise::cos(theta));
                half s = half(metal::precise::sin(theta));
                half r0;
                half r1;
                moespresso_dsv4_rope_pair(
                    kv[d0], kv[d0 + 1u], c, s, &r0, &r1);
                row_out[d0] = r0;
                row_out[d0 + 1u] = r1;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: one simdgroup scores one pooled row at a time. Each lane owns
    // up to two heads and runs their full 128-dim dots serially (the row
    // reads broadcast across lanes), so the only cross-lane reduction is a
    // single simd_sum of the weighted relu terms per row. A per-head
    // simd_sum layout (the slice-A structure) serializes 64 reductions per
    // row and is several times slower at this single-threadgroup shape.
    for (uint r = sg; r < n_rows; r += NSG) {
        device const float4 *row4 =
            (device const float4 *)(pooled + (uint64_t)r * 128u);
        float acc = 0.0f;
        for (uint hh = 0u; hh < 2u; hh++) {
            uint h = lane + 32u * hh;
            if (h >= n_heads) {
                continue;
            }
            threadgroup const half4 *q4 =
                (threadgroup const half4 *)(qbuf + (uint64_t)h * 128u);
            float d = 0.0f;
            for (uint c = 0u; c < 32u; c++) {
                d += dot(float4(q4[c]), row4[c]);
            }
            acc += max(d, 0.0f) * HEAD_SCALE * weights[h];
        }
        float total = simd_sum(acc);
        if (lane == 0u) {
            scores[r] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Phase C: radix select. sc[0] is the growing key prefix, sc[1] the
    // count still to take inside the current prefix group.
    if (tid == 0u) {
        sc[0] = 0u;
        sc[1] = (uint)TOPK;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pass = 0u; pass < 4u; pass++) {
        uint shift = 24u - 8u * pass;
        if (tid < 256u) {
            atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix = sc[0];
        uint mask_hi = pass == 0u ? 0u : (0xFFFFFFFFu << (shift + 8u));
        for (uint r = tid; r < n_rows; r += NT) {
            uint key = moespresso_dsv4_score_key(scores[r]);
            if ((key & mask_hi) == prefix) {
                atomic_fetch_add_explicit(
                    &hist[(key >> shift) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint need = sc[1];
            uint b = 255u;
            for (;; b--) {
                uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (c >= need || b == 0u) {
                    break;
                }
                need -= c;
            }
            sc[0] = prefix | (b << shift);
            sc[1] = need;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint threshold = sc[0];
    uint need_ties = sc[1];
    uint n_above = (uint)TOPK - need_ties;

    // Emission: blocked row ranges plus an inclusive scan of packed
    // (above, tie) counts give each thread deterministic output slots.
    uint chunk = (n_rows + NT - 1u) / NT;
    uint r0 = tid * chunk;
    uint r1 = metal::min(n_rows, r0 + chunk);
    uint above = 0u;
    uint ties = 0u;
    for (uint r = r0; r < r1; r++) {
        uint key = moespresso_dsv4_score_key(scores[r]);
        if (key > threshold) {
            above++;
        } else if (key == threshold) {
            ties++;
        }
    }
    uint packed = (above << 16) | ties;
    scan_buf[tid] = packed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < NT; stride <<= 1u) {
        uint val = scan_buf[tid];
        uint add = tid >= stride ? scan_buf[tid - stride] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan_buf[tid] = val + add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint excl = scan_buf[tid] - packed;
    uint above_slot = excl >> 16;
    uint tie_rank = excl & 0xFFFFu;
    for (uint r = r0; r < r1; r++) {
        uint key = moespresso_dsv4_score_key(scores[r]);
        if (key > threshold) {
            sel[above_slot++] = (int)r;
        } else if (key == threshold) {
            if (tie_rank < need_ties) {
                sel[n_above + tie_rank] = (int)r;
            }
            tie_rank++;
        }
    }
"""

# One threadgroup per query head, four simdgroups striding the key rows.
# Keys are the local window (with the prepared row substituted at the cache
# write slot) followed by the selected pooled rows; values equal keys.
_SDPA_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint head = threadgroup_position_in_grid.x;

    uint window_rows = (uint)win_shape[0];
    uint n_sel = (uint)sel_shape[0];
    uint n_keys = window_rows + n_sel;
    int pos = params[0];
    uint write_idx = (uint)params[1];

    threadgroup half qh[512];
    threadgroup float m_sg[4];
    threadgroup float l_sg[4];
    threadgroup float o_tg[4 * 512];

    // Partial rope on the query head at composed float16 semantics; thread
    // t holds dims [4t, 4t + 4), so rope pairs stay in-thread.
    device const half *qrow = q + (uint64_t)head * 512u;
    {
        uint d0 = tid * 4u;
        if (d0 < 448u) {
            for (ushort j = 0; j < 4; j++) {
                qh[d0 + j] = qrow[d0 + j];
            }
        } else {
            for (ushort j = 0; j < 2; j++) {
                uint dd = d0 + 2u * j;
                uint p = (dd - 448u) / 2u;
                float theta = float(pos) * inv_freq[p];
                half c = half(metal::precise::cos(theta));
                half s = half(metal::precise::sin(theta));
                half r0;
                half r1;
                moespresso_dsv4_rope_pair(
                    qrow[dd], qrow[dd + 1u], c, s, &r0, &r1);
                qh[dd] = r0;
                qh[dd + 1u] = r1;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const half4 *qh4 = (threadgroup const half4 *)qh;
    half4 q0 = qh4[lane +  0u];
    half4 q1 = qh4[lane + 32u];
    half4 q2 = qh4[lane + 64u];
    half4 q3 = qh4[lane + 96u];

    float m = -3.402823466e38f;
    float l = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint j = sg; j < n_keys; j += 4u) {
        device const half *ksrc;
        if (j < window_rows) {
            ksrc = j == write_idx ? row : (win + (uint64_t)j * 512u);
        } else {
            uint r = (uint)sel[j - window_rows];
            ksrc = pooled + (uint64_t)r * 512u;
        }
        device const half4 *k4 = (device const half4 *)ksrc;
        half4 k0 = k4[lane +  0u];
        half4 k1 = k4[lane + 32u];
        half4 k2 = k4[lane + 64u];
        half4 k3 = k4[lane + 96u];
        float score = dot(float4(q0), float4(k0)) +
                      dot(float4(q1), float4(k1)) +
                      dot(float4(q2), float4(k2)) +
                      dot(float4(q3), float4(k3));
        score = simd_sum(score) * SOFTMAX_SCALE;
        float new_m = metal::max(m, score);
        float factor = metal::fast::exp(m - new_m);
        float w = metal::fast::exp(score - new_m);
        l = l * factor + w;
        o0 = o0 * factor + float4(k0) * w;
        o1 = o1 * factor + float4(k1) * w;
        o2 = o2 * factor + float4(k2) * w;
        o3 = o3 * factor + float4(k3) * w;
        m = new_m;
    }

    if (lane == 0u) {
        m_sg[sg] = m;
        l_sg[sg] = l;
    }
    threadgroup float *osg = o_tg + (uint64_t)sg * 512u;
    for (ushort j = 0; j < 4; j++) {
        osg[  0u + 4u * lane + j] = o0[j];
        osg[128u + 4u * lane + j] = o1[j];
        osg[256u + 4u * lane + j] = o2[j];
        osg[384u + 4u * lane + j] = o3[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Merge the simdgroup partials; the sink joins the max and the
    // denominator exactly once.
    float m_all = metal::max(
        metal::max(m_sg[0], m_sg[1]), metal::max(m_sg[2], m_sg[3]));
    float sink = float(sinks[head]);
    float m_fin = metal::max(m_all, sink);
    float l_fin = metal::fast::exp(sink - m_fin);
    float f_sg[4];
    for (ushort i = 0; i < 4; i++) {
        f_sg[i] = metal::fast::exp(m_sg[i] - m_fin);
        l_fin += l_sg[i] * f_sg[i];
    }

    // Thread t finalizes dims [4t, 4t + 4): reduce, normalize, round to
    // float16, then the inverse partial rope at float16 semantics.
    uint d0 = tid * 4u;
    half hv[4];
    for (ushort j = 0; j < 4; j++) {
        float acc = 0.0f;
        for (ushort i = 0; i < 4; i++) {
            acc += o_tg[(uint64_t)i * 512u + d0 + j] * f_sg[i];
        }
        hv[j] = half(l_fin == 0.0f ? acc : acc / l_fin);
    }
    if (d0 < 448u) {
        for (ushort j = 0; j < 4; j++) {
            out[(uint64_t)head * 512u + d0 + j] = hv[j];
        }
    } else {
        for (ushort j = 0; j < 2; j++) {
            uint dd = d0 + 2u * j;
            uint p = (dd - 448u) / 2u;
            float theta = float(pos) * inv_freq[p];
            half c = half(metal::precise::cos(theta));
            half s = half(-metal::precise::sin(theta));
            half r0;
            half r1;
            moespresso_dsv4_rope_pair(hv[2u * j], hv[2u * j + 1u], c, s, &r0, &r1);
            out[(uint64_t)head * 512u + dd] = r0;
            out[(uint64_t)head * 512u + dd + 1u] = r1;
        }
    }
"""


# Kernels are cached per (scale, topk, dtype) for prep and per softmax scale
# for SDPA; both scales are per-model constants baked as float32 hex.
_PREP_KERNELS: dict = {}
_SDPA_KERNELS: dict = {}

# Standalone E4M3FN round trip over 512-wide compressed-KV rows: one row per
# simdgroup, lane l holding dims [16l, 16l + 16). Lanes 0..27 quantize the
# 448-dim prefix over aligned 64-element blocks (amax reduced exactly with
# simd_shuffle_xor inside each 4-lane group); lanes 28..31 copy the rope tail
# through. The scale/clip/table math is the recorded bit-exact transcription
# of ``_deepseek_v4_fp8_kv_roundtrip`` from the header above.
_FP8_ROWS_SOURCE = """
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

_FP8_ROWS_KERNEL: object | None = None


def _get_fp8_rows_kernel():
    global _FP8_ROWS_KERNEL
    if _FP8_ROWS_KERNEL is None:
        import mlx.core as mx

        _FP8_ROWS_KERNEL = mx.fast.metal_kernel(
            name="moespresso_dsv4_fp8_kv_prefix_rows",
            input_names=["x"],
            output_names=["out"],
            source=_FP8_ROWS_SOURCE,
            header=_prep_header(),
        )
    return _FP8_ROWS_KERNEL


def fp8_kv_prefix_rows(x):
    """Apply DS4's E4M3FN round trip to 512-wide float32 compressed-KV rows.

    Single-dispatch equivalent of ``_deepseek_v4_fp8_kv_roundtrip`` for the
    ``head_dim=512, rot_dim=64`` contract, bit-identical through the recorded
    transcription (the composed path materializes a ``[rows, 448, 127]``
    argmin diff per call). Not gated by ``_ENABLED``; the caller gates.

    Args:
        x: ``[..., 512]`` float32 rows.

    Returns:
        float32 array of the same shape.
    """
    import mlx.core as mx

    if x.ndim < 1 or int(x.shape[-1]) != _HEAD_DIM:
        raise ValueError("fp8 rows expect 512-wide rows")
    if x.dtype != mx.float32:
        raise ValueError("fp8 rows must be float32")
    rows2d = x.reshape(-1, _HEAD_DIM)
    n_rows = int(rows2d.shape[0])
    if n_rows == 0:
        raise ValueError("fp8 rows expect at least one row")
    groups = (n_rows + 3) // 4
    out, = _get_fp8_rows_kernel()(
        inputs=[rows2d],
        output_shapes=[(n_rows, _HEAD_DIM)],
        output_dtypes=[mx.float32],
        grid=(_SDPA_THREADS * groups, 1, 1),
        threadgroup=(_SDPA_THREADS, 1, 1),
    )
    return out.reshape(x.shape)


def _query_dtypes():
    import mlx.core as mx

    return (mx.float32, mx.float16, mx.bfloat16)


def _get_prep_kernel(scale: float, topk: int):
    key = (float(scale), int(topk))
    kernel = _PREP_KERNELS.get(key)
    if kernel is None:
        import mlx.core as mx

        source = (
            _PREP_SOURCE_TEMPLATE
            .replace("HEAD_SCALE", _f32_hex(scale))
            .replace("TOPK", str(int(topk)))
        )
        kernel = mx.fast.metal_kernel(
            name=f"moespresso_dsv4_fused_decode_prep_{len(_PREP_KERNELS)}",
            input_names=["q", "pooled", "weights", "kv", "inv_freq", "params"],
            output_names=["sel", "row_out", "scores"],
            source=source,
            header=_prep_header(),
        )
        _PREP_KERNELS[key] = kernel
    return kernel


def _get_sdpa_kernel(scale: float):
    key = float(scale)
    kernel = _SDPA_KERNELS.get(key)
    if kernel is None:
        import mlx.core as mx

        kernel = mx.fast.metal_kernel(
            name=f"moespresso_dsv4_fused_decode_sdpa_{len(_SDPA_KERNELS)}",
            input_names=[
                "q", "win", "row", "pooled", "sel", "sinks", "inv_freq", "params",
            ],
            output_names=["out"],
            source=_SDPA_SOURCE.replace("SOFTMAX_SCALE", _f32_hex(scale)),
            header=_prep_header(),
        )
        _SDPA_KERNELS[key] = kernel
    return kernel


def fused_decode_prep(q_idx, pooled_qat, weights, kv_row, inv_freq, params,
                      *, scale, topk):
    """Score, select, and prepare the KV row for one ratio-4 decode token.

    Args:
        q_idx: ``[n_heads, 128]`` post-rope, pre-QAT indexer queries in
            float32, float16, or bfloat16.
        pooled_qat: float32 ``[n_rows, 128]`` QAT'd indexer pool rows.
        weights: float32 ``[n_heads]`` projected head weights, already
            multiplied by ``n_heads ** -0.5``.
        kv_row: float16 ``[512]`` post-norm, pre-rope KV projection row.
        inv_freq: float32 ``[32]`` rope inverse frequencies (YaRN folded).
        params: int32 array whose first element is the token position.
        scale: indexer score scale (``index_head_dim ** -0.5``).
        topk: selection width; must be smaller than ``n_rows``.

    Returns:
        ``(sel, row, scores)``: int32 ``[topk]`` selected pool rows
        (deterministic order), float16 ``[512]`` roped and FP8-rounded KV
        row ready for the local cache, and float32 ``[n_rows]`` scores.
    """
    import mlx.core as mx

    if q_idx.ndim != 2 or int(q_idx.shape[1]) != _INDEX_HEAD_DIM:
        raise ValueError("q_idx must have shape [n_heads, 128]")
    n_heads = int(q_idx.shape[0])
    if n_heads <= 0 or n_heads > _MAX_HEADS:
        raise ValueError("q_idx must carry between 1 and 64 heads")
    if q_idx.dtype not in _query_dtypes():
        raise ValueError("q_idx must be float32, float16, or bfloat16")
    if (
        pooled_qat.ndim != 2
        or int(pooled_qat.shape[1]) != _INDEX_HEAD_DIM
        or pooled_qat.dtype != mx.float32
    ):
        raise ValueError("pooled_qat must be float32 [n_rows, 128]")
    n_rows = int(pooled_qat.shape[0])
    topk = int(topk)
    if not 0 < topk < n_rows:
        raise ValueError("topk must be positive and smaller than n_rows")
    if n_rows >= 1 << 16:
        raise ValueError("the radix-select scan packs counts in 16 bits")
    if tuple(int(v) for v in weights.shape) != (n_heads,) or (
        weights.dtype != mx.float32
    ):
        raise ValueError("weights must be float32 [n_heads]")
    if tuple(int(v) for v in kv_row.shape) != (_HEAD_DIM,) or (
        kv_row.dtype != mx.float16
    ):
        raise ValueError("kv_row must be float16 [512]")
    if int(inv_freq.shape[-1]) != _ROPE_DIM // 2 or inv_freq.dtype != mx.float32:
        raise ValueError("inv_freq must be float32 [32]")

    sel, row, scores = _get_prep_kernel(float(scale), topk)(
        inputs=[q_idx, pooled_qat, weights, kv_row, inv_freq, params],
        template=[("T", q_idx.dtype)],
        output_shapes=[(topk,), (_HEAD_DIM,), (n_rows,)],
        output_dtypes=[mx.int32, mx.float16, mx.float32],
        grid=(_PREP_THREADS, 1, 1),
        threadgroup=(_PREP_THREADS, 1, 1),
    )
    return sel, row, scores


def fused_decode_sdpa(q, window, row, pooled, sel, sinks, inv_freq, params,
                      *, scale):
    """Attend one decode token over the window and selected pooled rows.

    Args:
        q: float16 ``[n_heads, 512]`` post-norm, pre-rope query heads.
        window: float16 ``[window_rows, 512]`` local KV buffer (pre-update).
        row: float16 ``[512]`` prepared KV row (``fused_decode_prep``).
        pooled: float16 ``[n_rows, 512]`` compressed pool rows.
        sel: int32 ``[topk]`` selected pool row ids.
        sinks: float16 ``[n_heads]`` per-head attention sinks.
        inv_freq: float32 ``[32]`` rope inverse frequencies.
        params: int32 array ``[position, write_idx]``; the prepared row
            replaces ``window[write_idx]``.
        scale: softmax scale (``head_dim ** -0.5``).

    Returns:
        float16 ``[n_heads, 512]`` attention output after the inverse rope,
        head-major, ready for the grouped output projection.
    """
    import mlx.core as mx

    if q.ndim != 2 or int(q.shape[1]) != _HEAD_DIM or q.dtype != mx.float16:
        raise ValueError("q must be float16 [n_heads, 512]")
    n_heads = int(q.shape[0])
    if n_heads <= 0:
        raise ValueError("q must carry at least one head")
    if window.ndim != 2 or int(window.shape[1]) != _HEAD_DIM or (
        window.dtype != mx.float16
    ):
        raise ValueError("window must be float16 [window_rows, 512]")
    if tuple(int(v) for v in row.shape) != (_HEAD_DIM,) or row.dtype != mx.float16:
        raise ValueError("row must be float16 [512]")
    if pooled.ndim != 2 or int(pooled.shape[1]) != _HEAD_DIM or (
        pooled.dtype != mx.float16
    ):
        raise ValueError("pooled must be float16 [n_rows, 512]")
    if sel.ndim != 1 or sel.dtype != mx.int32:
        raise ValueError("sel must be int32 [topk]")
    if tuple(int(v) for v in sinks.shape) != (n_heads,) or (
        sinks.dtype != mx.float16
    ):
        raise ValueError("sinks must be float16 [n_heads]")
    if int(inv_freq.shape[-1]) != _ROPE_DIM // 2 or inv_freq.dtype != mx.float32:
        raise ValueError("inv_freq must be float32 [32]")

    out, = _get_sdpa_kernel(float(scale))(
        inputs=[q, window, row, pooled, sel, sinks, inv_freq, params],
        output_shapes=[(n_heads, _HEAD_DIM)],
        output_dtypes=[mx.float16],
        grid=(_SDPA_THREADS * n_heads, 1, 1),
        threadgroup=(_SDPA_THREADS, 1, 1),
    )
    return out


def fused_decode_enabled() -> bool:
    """Whether the fused decode island may engage (gate plus Metal)."""
    return _ENABLED and _metal_available()
