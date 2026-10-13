// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The MoEspresso authors. See LICENSE-MIT and THIRD-PARTY-NOTICES.
// Routed IQ_K kernels with per-slot stream addresses and source-order reduction.
template<bool IQ3>
inline void chunk_gate_up(
    const device half* x,
    const device ulong* addresses, const device half* table,
    const device uint* slots, const device uint* order, const device half* sigmoid,
    device half* gate, device half* up, device half* activation,
    threadgroup half* values, threadgroup float* gp, threadgroup float* upart,
    uint tid, uint block) {
    const uint position = block / 40, row0 = (block % 40) * 16;
    const uint route = order[position], expert = slots[position];
    const device ulong* bases = addresses + ulong(expert) * 18;
    const device uchar* gqs = reinterpret_cast<const device uchar*>(bases[0]);
    const device uint* gqh = reinterpret_cast<const device uint*>(bases[1]);
    const device uchar* gscl = reinterpret_cast<const device uchar*>(bases[2]);
    const device uchar* gsch = reinterpret_cast<const device uchar*>(bases[3]);
    const device uchar* gsex = reinterpret_cast<const device uchar*>(bases[4]);
    const device half* gdv = reinterpret_cast<const device half*>(bases[5]);
    const device uchar* uqs = reinterpret_cast<const device uchar*>(bases[6]);
    const device uint* uqh = reinterpret_cast<const device uint*>(bases[7]);
    const device uchar* uscl = reinterpret_cast<const device uchar*>(bases[8]);
    const device uchar* usch = reinterpret_cast<const device uchar*>(bases[9]);
    const device uchar* usex = reinterpret_cast<const device uchar*>(bases[10]);
    const device half* udv = reinterpret_cast<const device half*>(bases[11]);
    const uint lane = tid & 31u, sg = tid >> 5u, kg = tid;
    if (tid < (IQ3 ? 16u : 8u)) values[tid] = table[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xv[32];
    routed_xload(x + kg * 32, xv);
    for (uint row = 0; row < 16; ++row) {
        const ulong rid = row0 + row;
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

template<bool IQ3>
inline void chunk_down(
    const device half* x,
    const device ulong* addresses, const device half* table, const device uint* slots, const device ushort* scores,
    const device uint* order, device half* down, device ushort* output,
    threadgroup half* values, threadgroup half* slot_values, uint tid, uint tile) {
    const uint position = tid >> 5u, lane = tid & 31u;
    const uint route = order[position], expert = slots[position];
    const device ulong* bases = addresses + ulong(expert) * 18 + 12;
    const device uchar* qs = reinterpret_cast<const device uchar*>(bases[0]);
    const device uint* qh = reinterpret_cast<const device uint*>(bases[1]);
    const device uchar* scl = reinterpret_cast<const device uchar*>(bases[2]);
    const device uchar* sch = reinterpret_cast<const device uchar*>(bases[3]);
    const device uchar* sex = reinterpret_cast<const device uchar*>(bases[4]);
    const device half* dv = reinterpret_cast<const device half*>(bases[5]);
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
            const ulong rid = tile * 16 + row;
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

// Current capacity excludes retired suffix rows before any indirect access.
#define CHUNK_GATE(NAME, IQ3) \
kernel void NAME( \
    const device half* x [[buffer(0)]], const device ulong* addresses [[buffer(1)]], \
    const device half* table [[buffer(2)]], const device uint* slots [[buffer(3)]], \
    const device uint* order [[buffer(4)]], const device half* sigmoid [[buffer(5)]], \
    device half* gate [[buffer(6)]], device half* up [[buffer(7)]], \
    device half* activation [[buffer(8)]], const device uint* publication [[buffer(9)]], \
    constant uint& capacity [[buffer(10)]], \
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) { \
    if (publication[0] != 1u) return; \
    if (capacity < 10u || capacity >= 512u || slots[group / 40u] >= capacity) return; \
    threadgroup half values[IQ3 ? 16 : 8]; \
    threadgroup float gp[48], upart[48]; \
    chunk_gate_up<IQ3>(x, addresses, table, slots, order, sigmoid, gate, up, activation, \
        values, gp, upart, tid, group); \
}
#define CHUNK_DOWN(NAME, IQ3) \
kernel void NAME( \
    const device half* x [[buffer(0)]], const device ulong* addresses [[buffer(1)]], \
    const device half* table [[buffer(2)]], const device uint* slots [[buffer(3)]], \
    const device ushort* scores [[buffer(4)]], const device uint* order [[buffer(5)]], \
    device half* down [[buffer(6)]], device ushort* output [[buffer(7)]], \
    const device uint* publication [[buffer(8)]], \
    constant uint& capacity [[buffer(9)]], \
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) { \
    if (publication[0] != 1u) { if (tid < 16u) output[group * 16u + tid] = 0u; return; } \
    if (capacity < 10u || capacity >= 512u) { if (tid < 16u) output[group * 16u + tid] = 0u; return; } \
    for (uint i = 0; i < 10u; ++i) { \
        if (slots[i] >= capacity) { if (tid < 16u) output[group * 16u + tid] = 0u; return; } \
    } \
    threadgroup half values[IQ3 ? 16 : 8], slot_values[160]; \
    chunk_down<IQ3>(x, addresses, table, slots, scores, order, down, output, \
        values, slot_values, tid, group); \
}
CHUNK_GATE(chunk_gate_iq2, false)
CHUNK_GATE(chunk_gate_iq3, true)
CHUNK_DOWN(chunk_down_iq2, false)
CHUNK_DOWN(chunk_down_iq3, true)
