// IQ2_K and IQ3_K relayout GEMV.
// Copyright (c) 2026 The MoEspresso authors. See LICENSE-MIT.
#include <metal_stdlib>
using namespace metal;

template <uint K, uint SG>
inline void iq2_k_gemv(
    const device half* x, const device uchar* qs, const device uchar* scl,
    const device uchar* sex, const device half* dv, const device half* vtab,
    const device uint* sel, const device uint* dims, device half* out,
    threadgroup half* values, threadgroup float* partials, uint tid, uint blk
) {
    const uint lane = tid & 31u;
    const uint sg = tid >> 5u;
    const uint rows = dims[1];
    const uint blocks_per_pair = rows / 16u;
    const uint mat = blk / blocks_per_pair;
    const uint row0 = (blk - mat * blocks_per_pair) * 16u;
    const uint expert = sel[mat];
    const uint token = mat / dims[0];
    const uint kbase = sg * 1024u + lane * 32u;
    const uint kg = kbase >> 5u;
    if (tid < 8u) { values[tid] = vtab[tid]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xv[32];
    for (uint j = 0u; j < 32u; ++j) { xv[j] = float(x[ulong(token) * K + kbase + j]); }
    for (uint r = 0u; r < 16u; ++r) {
        const ulong rid = ulong(expert) * rows + row0 + r;
        const device uint2* qp = reinterpret_cast<const device uint2*>(qs + rid * (K / 4u) + kg * 8u);
        const uint2 qw = qp[0];
        const uint sc = uint(scl[rid * (K / 32u) + kg]);
        const uint ex = uint(sex[rid * (K / 128u) + (kg >> 2u)]) >> ((kg << 1u) & 7u);
        const float d = float(dv[rid * (K / 256u) + (kg >> 3u)]);
        const float dl0 = d * float(int(sc & 15u) - 8);
        const float dl1 = d * float(int(sc >> 4u) - 8);
        const uint t0 = (ex & 1u) << 2u;
        const uint t1 = ((ex >> 1u) & 1u) << 2u;
        const float g00 = float(values[t0]);
        const float g01 = float(values[t0 + 1u]);
        const float g02 = float(values[t0 + 2u]);
        const float g03 = float(values[t0 + 3u]);
        const float g10 = float(values[t1]);
        const float g11 = float(values[t1 + 1u]);
        const float g12 = float(values[t1 + 2u]);
        const float g13 = float(values[t1 + 3u]);
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint j = 0u; j < 16u; ++j) {
            const uint code = (qw.x >> (j * 2u)) & 3u;
            const float v = (code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00);
            acc0 = fma(v, xv[j], acc0);
        }
        for (uint j = 0u; j < 16u; ++j) {
            const uint code = (qw.y >> (j * 2u)) & 3u;
            const float v = (code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10);
            acc1 = fma(v, xv[j + 16u], acc1);
        }
        const float pv = fma(dl0, acc0, dl1 * acc1);
        const float ssum = simd_sum(pv);
        if (lane == 0u) { partials[sg * 16u + r] = ssum; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16u) {
        float sum = partials[tid];
        for (uint s = 1u; s < SG; ++s) { sum += partials[s * 16u + tid]; }
        out[ulong(mat) * rows + row0 + tid] = half(sum);
    }
}

template <uint K, uint SG>
inline void iq3_k_gemv(
    const device half* x, const device uchar* qs, const device uint* qh,
    const device uchar* scl, const device uchar* sch, const device uchar* sex,
    const device half* dv, const device half* vtab, const device uint* sel,
    const device uint* dims, device half* out, threadgroup half* values,
    threadgroup float* partials, uint tid, uint blk
) {
    const uint lane = tid & 31u;
    const uint sg = tid >> 5u;
    const uint rows = dims[1];
    const uint blocks_per_pair = rows / 16u;
    const uint mat = blk / blocks_per_pair;
    const uint row0 = (blk - mat * blocks_per_pair) * 16u;
    const uint expert = sel[mat];
    const uint token = mat / dims[0];
    const uint kbase = sg * 1024u + lane * 32u;
    const uint kg = kbase >> 5u;
    if (tid < 16u) { values[tid] = vtab[tid]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xv[32];
    for (uint j = 0u; j < 32u; ++j) { xv[j] = float(x[ulong(token) * K + kbase + j]); }
    for (uint r = 0u; r < 16u; ++r) {
        const ulong rid = ulong(expert) * rows + row0 + r;
        const device uint2* qp = reinterpret_cast<const device uint2*>(qs + rid * (K / 4u) + kg * 8u);
        const uint2 qw = qp[0];
        const uint hi = qh[rid * (K / 32u) + kg];
        const uint sc = uint(scl[rid * (K / 32u) + kg]);
        const uint sh = uint(sch[rid * (K / 128u) + (kg >> 2u)]) >> ((kg << 1u) & 7u);
        const uint ex = uint(sex[rid * (K / 128u) + (kg >> 2u)]) >> ((kg << 1u) & 7u);
        const float d = float(dv[rid * (K / 256u) + (kg >> 3u)]);
        const float dl0 = d * float(int(2u * (sc & 15u) + 1u) * ((sh & 1u) ? -1 : 1));
        const float dl1 = d * float(int(2u * (sc >> 4u) + 1u) * ((sh & 2u) ? -1 : 1));
        const uint t0 = (ex & 1u) << 3u;
        const uint t1 = ((ex >> 1u) & 1u) << 3u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint j = 0u; j < 16u; ++j) {
            const uint code = ((qw.x >> (j * 2u)) & 3u) | (((hi >> j) & 1u) << 2u);
            acc0 = fma(float(values[t0 + code]), xv[j], acc0);
        }
        for (uint j = 0u; j < 16u; ++j) {
            const uint code = ((qw.y >> (j * 2u)) & 3u) | (((hi >> (j + 16u)) & 1u) << 2u);
            acc1 = fma(float(values[t1 + code]), xv[j + 16u], acc1);
        }
        const float pv = fma(dl0, acc0, dl1 * acc1);
        const float ssum = simd_sum(pv);
        if (lane == 0u) { partials[sg * 16u + r] = ssum; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16u) {
        float sum = partials[tid];
        for (uint s = 1u; s < SG; ++s) { sum += partials[s * 16u + tid]; }
        out[ulong(mat) * rows + row0 + tid] = half(sum);
    }
}

kernel void iq2_k_2560(
    const device half* x [[buffer(0)]], const device uchar* qs [[buffer(1)]],
    const device uchar* scl [[buffer(2)]], const device uchar* sex [[buffer(3)]],
    const device half* dv [[buffer(4)]], const device half* vtab [[buffer(5)]],
    const device uint* sel [[buffer(6)]], const device uint* dims [[buffer(7)]],
    device half* out [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]]
) {
    threadgroup half values[8];
    threadgroup float partials[3 * 16];
    iq2_k_gemv<2560u, 3u>(x, qs, scl, sex, dv, vtab, sel, dims, out, values, partials, tid, blk);
}

kernel void iq2_k_768(
    const device half* x [[buffer(0)]], const device uchar* qs [[buffer(1)]],
    const device uchar* scl [[buffer(2)]], const device uchar* sex [[buffer(3)]],
    const device half* dv [[buffer(4)]], const device half* vtab [[buffer(5)]],
    const device uint* sel [[buffer(6)]], const device uint* dims [[buffer(7)]],
    device half* out [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]]
) {
    threadgroup half values[8];
    threadgroup float partials[16];
    iq2_k_gemv<768u, 1u>(x, qs, scl, sex, dv, vtab, sel, dims, out, values, partials, tid, blk);
}

kernel void iq3_k_2560(
    const device half* x [[buffer(0)]], const device uchar* qs [[buffer(1)]],
    const device uint* qh [[buffer(2)]], const device uchar* scl [[buffer(3)]],
    const device uchar* sch [[buffer(4)]], const device uchar* sex [[buffer(5)]],
    const device half* dv [[buffer(6)]], const device half* vtab [[buffer(7)]],
    const device uint* sel [[buffer(8)]], const device uint* dims [[buffer(9)]],
    device half* out [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]]
) {
    threadgroup half values[16];
    threadgroup float partials[3 * 16];
    iq3_k_gemv<2560u, 3u>(x, qs, qh, scl, sch, sex, dv, vtab, sel, dims, out, values, partials, tid, blk);
}

kernel void iq3_k_768(
    const device half* x [[buffer(0)]], const device uchar* qs [[buffer(1)]],
    const device uint* qh [[buffer(2)]], const device uchar* scl [[buffer(3)]],
    const device uchar* sch [[buffer(4)]], const device uchar* sex [[buffer(5)]],
    const device half* dv [[buffer(6)]], const device half* vtab [[buffer(7)]],
    const device uint* sel [[buffer(8)]], const device uint* dims [[buffer(9)]],
    device half* out [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]]
) {
    threadgroup half values[16];
    threadgroup float partials[16];
    iq3_k_gemv<768u, 1u>(x, qs, qh, scl, sch, sex, dv, vtab, sel, dims, out, values, partials, tid, blk);
}
