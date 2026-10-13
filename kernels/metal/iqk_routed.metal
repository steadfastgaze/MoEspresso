// Two-stage routed IQ_K decode. See THIRD-PARTY-NOTICES for source attribution.
// Copyright (c) 2026 The MoEspresso authors. See LICENSE-MIT.
#include <metal_stdlib>
using namespace metal;

// The package-facing residual is BF16; the routed IQ_K gate/up decoder reads
// FP16 activation words. This conversion stays on the device.
kernel void qwen4_routed_input_fp16(
    const device ushort* source [[buffer(0)]],
    device half* converted [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index < 2560u) converted[index] = half(as_type<float>(uint(source[index]) << 16u));
}

// Load 32 FP16 activations through four 128-bit reads.
inline void routed_xload(const device half* input, thread float* xv) {
    const uint4 w0 = reinterpret_cast<const device uint4*>(input)[0];
    const half2 h00 = as_type<half2>(w0.x);
    xv[0] = float(h00.x); xv[1] = float(h00.y);
    const half2 h01 = as_type<half2>(w0.y);
    xv[2] = float(h01.x); xv[3] = float(h01.y);
    const half2 h02 = as_type<half2>(w0.z);
    xv[4] = float(h02.x); xv[5] = float(h02.y);
    const half2 h03 = as_type<half2>(w0.w);
    xv[6] = float(h03.x); xv[7] = float(h03.y);
    const uint4 w1 = reinterpret_cast<const device uint4*>(input)[1];
    const half2 h10 = as_type<half2>(w1.x);
    xv[8] = float(h10.x); xv[9] = float(h10.y);
    const half2 h11 = as_type<half2>(w1.y);
    xv[10] = float(h11.x); xv[11] = float(h11.y);
    const half2 h12 = as_type<half2>(w1.z);
    xv[12] = float(h12.x); xv[13] = float(h12.y);
    const half2 h13 = as_type<half2>(w1.w);
    xv[14] = float(h13.x); xv[15] = float(h13.y);
    const uint4 w2 = reinterpret_cast<const device uint4*>(input)[2];
    const half2 h20 = as_type<half2>(w2.x);
    xv[16] = float(h20.x); xv[17] = float(h20.y);
    const half2 h21 = as_type<half2>(w2.y);
    xv[18] = float(h21.x); xv[19] = float(h21.y);
    const half2 h22 = as_type<half2>(w2.z);
    xv[20] = float(h22.x); xv[21] = float(h22.y);
    const half2 h23 = as_type<half2>(w2.w);
    xv[22] = float(h23.x); xv[23] = float(h23.y);
    const uint4 w3 = reinterpret_cast<const device uint4*>(input)[3];
    const half2 h30 = as_type<half2>(w3.x);
    xv[24] = float(h30.x); xv[25] = float(h30.y);
    const half2 h31 = as_type<half2>(w3.y);
    xv[26] = float(h31.x); xv[27] = float(h31.y);
    const half2 h32 = as_type<half2>(w3.z);
    xv[28] = float(h32.x); xv[29] = float(h32.y);
    const half2 h33 = as_type<half2>(w3.w);
    xv[30] = float(h33.x); xv[31] = float(h33.y);
}

template<uint K, bool IQ3>
inline float routed_partial(
    const device uchar* qs, const device uint* qh, const device uchar* scl,
    const device uchar* sch, const device uchar* sex, const device half* dv,
    threadgroup const half* values, ulong rid, uint kg, thread const float* xv) {
    const uint2 q = *reinterpret_cast<const device uint2*>(qs + rid * (K / 4) + kg * 8);
    const uint hi = IQ3 ? qh[rid * (K / 32) + kg] : 0u;
    const uint sc = scl[rid * (K / 32) + kg];
    const uint sh = IQ3 ? uint(sch[rid * (K / 128) + (kg >> 2)]) >> ((kg << 1) & 7) : 0u;
    const uint ex = uint(sex[rid * (K / 128) + (kg >> 2)]) >> ((kg << 1) & 7);
    const float d = float(dv[rid * (K / 256) + (kg >> 3)]);
    const float dl0 = d * float(IQ3 ? int(2u * (sc & 15u) + 1u) * ((sh & 1u) ? -1 : 1) : int(sc & 15u) - 8);
    const float dl1 = d * float(IQ3 ? int(2u * (sc >> 4u) + 1u) * ((sh & 2u) ? -1 : 1) : int(sc >> 4u) - 8);
    const uint t0 = (ex & 1u) << (IQ3 ? 3u : 2u);
    const uint t1 = ((ex >> 1u) & 1u) << (IQ3 ? 3u : 2u);
    const float g00 = float(values[t0]), g01 = float(values[t0 + 1]), g02 = float(values[t0 + 2]), g03 = float(values[t0 + 3]);
    const float g10 = float(values[t1]), g11 = float(values[t1 + 1]), g12 = float(values[t1 + 2]), g13 = float(values[t1 + 3]);
    float acc0 = 0.0f, acc1 = 0.0f;
    { const uint code = ((q.x >> 0u) & 3u) | (IQ3 ? (((hi >> 0u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[0], acc0); }
    { const uint code = ((q.x >> 2u) & 3u) | (IQ3 ? (((hi >> 1u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[1], acc0); }
    { const uint code = ((q.x >> 4u) & 3u) | (IQ3 ? (((hi >> 2u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[2], acc0); }
    { const uint code = ((q.x >> 6u) & 3u) | (IQ3 ? (((hi >> 3u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[3], acc0); }
    { const uint code = ((q.x >> 8u) & 3u) | (IQ3 ? (((hi >> 4u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[4], acc0); }
    { const uint code = ((q.x >> 10u) & 3u) | (IQ3 ? (((hi >> 5u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[5], acc0); }
    { const uint code = ((q.x >> 12u) & 3u) | (IQ3 ? (((hi >> 6u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[6], acc0); }
    { const uint code = ((q.x >> 14u) & 3u) | (IQ3 ? (((hi >> 7u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[7], acc0); }
    { const uint code = ((q.x >> 16u) & 3u) | (IQ3 ? (((hi >> 8u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[8], acc0); }
    { const uint code = ((q.x >> 18u) & 3u) | (IQ3 ? (((hi >> 9u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[9], acc0); }
    { const uint code = ((q.x >> 20u) & 3u) | (IQ3 ? (((hi >> 10u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[10], acc0); }
    { const uint code = ((q.x >> 22u) & 3u) | (IQ3 ? (((hi >> 11u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[11], acc0); }
    { const uint code = ((q.x >> 24u) & 3u) | (IQ3 ? (((hi >> 12u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[12], acc0); }
    { const uint code = ((q.x >> 26u) & 3u) | (IQ3 ? (((hi >> 13u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[13], acc0); }
    { const uint code = ((q.x >> 28u) & 3u) | (IQ3 ? (((hi >> 14u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[14], acc0); }
    { const uint code = ((q.x >> 30u) & 3u) | (IQ3 ? (((hi >> 15u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t0 + code]) :
          ((code & 2u) ? ((code & 1u) ? g03 : g02) : ((code & 1u) ? g01 : g00));
      acc0 = fma(value, xv[15], acc0); }
    { const uint code = ((q.y >> 0u) & 3u) | (IQ3 ? (((hi >> 16u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[16], acc1); }
    { const uint code = ((q.y >> 2u) & 3u) | (IQ3 ? (((hi >> 17u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[17], acc1); }
    { const uint code = ((q.y >> 4u) & 3u) | (IQ3 ? (((hi >> 18u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[18], acc1); }
    { const uint code = ((q.y >> 6u) & 3u) | (IQ3 ? (((hi >> 19u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[19], acc1); }
    { const uint code = ((q.y >> 8u) & 3u) | (IQ3 ? (((hi >> 20u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[20], acc1); }
    { const uint code = ((q.y >> 10u) & 3u) | (IQ3 ? (((hi >> 21u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[21], acc1); }
    { const uint code = ((q.y >> 12u) & 3u) | (IQ3 ? (((hi >> 22u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[22], acc1); }
    { const uint code = ((q.y >> 14u) & 3u) | (IQ3 ? (((hi >> 23u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[23], acc1); }
    { const uint code = ((q.y >> 16u) & 3u) | (IQ3 ? (((hi >> 24u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[24], acc1); }
    { const uint code = ((q.y >> 18u) & 3u) | (IQ3 ? (((hi >> 25u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[25], acc1); }
    { const uint code = ((q.y >> 20u) & 3u) | (IQ3 ? (((hi >> 26u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[26], acc1); }
    { const uint code = ((q.y >> 22u) & 3u) | (IQ3 ? (((hi >> 27u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[27], acc1); }
    { const uint code = ((q.y >> 24u) & 3u) | (IQ3 ? (((hi >> 28u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[28], acc1); }
    { const uint code = ((q.y >> 26u) & 3u) | (IQ3 ? (((hi >> 29u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[29], acc1); }
    { const uint code = ((q.y >> 28u) & 3u) | (IQ3 ? (((hi >> 30u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[30], acc1); }
    { const uint code = ((q.y >> 30u) & 3u) | (IQ3 ? (((hi >> 31u) & 1u) << 2u) : 0u);
      const float value = IQ3 ? float(values[t1 + code]) :
          ((code & 2u) ? ((code & 1u) ? g13 : g12) : ((code & 1u) ? g11 : g10));
      acc1 = fma(value, xv[31], acc1); }
    return fma(dl0, acc0, dl1 * acc1);
}

template<bool IQ3>
inline void routed_gate_up(
    const device half* x,
    const device uchar* gqs, const device uint* gqh, const device uchar* gscl,
    const device uchar* gsch, const device uchar* gsex, const device half* gdv,
    const device half* table,
    const device uchar* uqs, const device uint* uqh, const device uchar* uscl,
    const device uchar* usch, const device uchar* usex, const device half* udv,
    const device uint* slots, const device uint* order, const device half* sigmoid,
    device half* gate, device half* up, device half* activation,
    threadgroup half* values, threadgroup float* gp, threadgroup float* upart,
    uint tid, uint block) {
    const uint position = block / 40, row0 = (block % 40) * 16;
    const uint route = order[position], expert = slots[position];
    const uint lane = tid & 31u, sg = tid >> 5u, kg = tid;
    if (tid < (IQ3 ? 16u : 8u)) values[tid] = table[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xv[32];
    routed_xload(x + kg * 32, xv);
    for (uint row = 0; row < 16; ++row) {
        const ulong rid = ulong(expert) * 640 + row0 + row;
        const float g = routed_partial<2560, IQ3>(gqs, gqh, gscl, gsch, gsex, gdv, values, rid, kg, xv);
        const float u = routed_partial<2560, IQ3>(uqs, uqh, uscl, usch, usex, udv, values, rid, kg, xv);
        const float gs = simd_sum(g), us = simd_sum(u);
        if (lane == 0) { gp[sg * 16 + row] = gs; upart[sg * 16 + row] = us; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        const half g = half(gp[tid] + gp[16 + tid] + gp[32 + tid]);
        const half u = half(upart[tid] + upart[16 + tid] + upart[32 + tid]);
        const uint column = row0 + tid;
        gate[route * 640 + column] = g;
        up[route * 640 + column] = u;
        const half sigmoid_gate = sigmoid[as_type<ushort>(g)];
        const half silu_gate = g * sigmoid_gate;
        activation[route * 768 + column] = silu_gate * u;
    }
    // Keep the public diagnostic activation's stored-width padding initialized.
    if (row0 == 0 && tid < 64) {
        activation[route * 768 + 640 + tid] = half(0);
        activation[route * 768 + 704 + tid] = half(0);
    }
}

inline ushort routed_bf16(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
inline float routed_float(ushort value) { return as_type<float>(uint(value) << 16u); }

template<bool IQ3>
inline void routed_down(
    const device half* x,
    const device uchar* qs, const device uint* qh, const device uchar* scl,
    const device uchar* sch, const device uchar* sex, const device half* dv,
    const device half* table, const device uint* slots, const device ushort* scores,
    const device uint* order, device half* down, device ushort* output,
    threadgroup half* values, threadgroup half* slot_values, uint tid, uint tile) {
    const uint position = tid >> 5u, lane = tid & 31u;
    const uint route = order[position], expert = slots[position];
    if (tid < (IQ3 ? 16u : 8u)) values[tid] = table[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xv[32];
    if (lane < 20) routed_xload(x + route * 768 + lane * 32, xv);
    else {
#pragma unroll
        for (uint j = 0; j < 32; ++j) xv[j] = 0.0f;
    }
    for (uint row = 0; row < 16; ++row) {
        float partial = 0.0f;
        if (lane < 24) {
            const ulong rid = ulong(expert) * 2560 + tile * 16 + row;
            partial = routed_partial<768, IQ3>(qs, qh, scl, sch, sex, dv, values, rid, lane, xv);
        }
        const float sum = simd_sum(partial);
        if (lane == 0) {
            const half rounded = half(sum);
            slot_values[position * 16 + row] = rounded;
            down[route * 2560 + tile * 16 + row] = rounded;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (position == 0 && lane < 16) {
        ushort total = 0;
        for (uint i = 0; i < 10; ++i) {
            const ushort value = routed_bf16(float(slot_values[i * 16 + lane]));
            const ushort product = routed_bf16(routed_float(value) * routed_float(scores[i]));
            total = routed_bf16(routed_float(total) + routed_float(product));
        }
        output[tile * 16 + lane] = total;
    }
}

#define ROUTED_GATE(NAME, IQ3) \
kernel void NAME( \
    const device half* x [[buffer(0)]], \
    const device uchar* gqs [[buffer(1)]], const device uint* gqh [[buffer(2)]], \
    const device uchar* gscl [[buffer(3)]], const device uchar* gsch [[buffer(4)]], \
    const device uchar* gsex [[buffer(5)]], const device half* gdv [[buffer(6)]], \
    const device half* table [[buffer(7)]], \
    const device uchar* uqs [[buffer(8)]], const device uint* uqh [[buffer(9)]], \
    const device uchar* uscl [[buffer(10)]], const device uchar* usch [[buffer(11)]], \
    const device uchar* usex [[buffer(12)]], const device half* udv [[buffer(13)]], \
    const device uint* slots [[buffer(14)]], const device uint* order [[buffer(15)]], \
    const device half* sigmoid [[buffer(16)]], device half* gate [[buffer(17)]], \
    device half* up [[buffer(18)]], device half* activation [[buffer(19)]], \
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) { \
    threadgroup half values[IQ3 ? 16 : 8]; \
    threadgroup float gp[48], upart[48]; \
    routed_gate_up<IQ3>(x, gqs, gqh, gscl, gsch, gsex, gdv, table, \
        uqs, uqh, uscl, usch, usex, udv, slots, order, sigmoid, gate, up, activation, \
        values, gp, upart, tid, group); \
}
#define ROUTED_DOWN(NAME, IQ3) \
kernel void NAME( \
    const device half* x [[buffer(0)]], \
    const device uchar* qs [[buffer(1)]], const device uint* qh [[buffer(2)]], \
    const device uchar* scl [[buffer(3)]], const device uchar* sch [[buffer(4)]], \
    const device uchar* sex [[buffer(5)]], const device half* dv [[buffer(6)]], \
    const device half* table [[buffer(7)]], const device uint* slots [[buffer(8)]], \
    const device ushort* scores [[buffer(9)]], const device uint* order [[buffer(10)]], \
    device half* down [[buffer(11)]], device ushort* output [[buffer(12)]], \
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) { \
    threadgroup half values[IQ3 ? 16 : 8], slot_values[160]; \
    routed_down<IQ3>(x, qs, qh, scl, sch, sex, dv, table, slots, scores, order, \
        down, output, values, slot_values, tid, group); \
}
ROUTED_GATE(routed_gate_iq2, false)
ROUTED_GATE(routed_gate_iq3, true)
ROUTED_DOWN(routed_down_iq2, false)
ROUTED_DOWN(routed_down_iq3, true)
