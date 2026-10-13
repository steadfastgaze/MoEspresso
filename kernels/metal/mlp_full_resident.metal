// Packed IQ2_K/IQ3_K expert tiles adapt the antirez/ds4 tile schedule (MIT).
// See THIRD-PARTY-NOTICES.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#ifndef MLP_FULL_TILE_TOKENS
#define MLP_FULL_TILE_TOKENS 32
#endif
#if MLP_FULL_TILE_TOKENS != 8 && MLP_FULL_TILE_TOKENS != 16 && MLP_FULL_TILE_TOKENS != 32
#error MLP_FULL_TILE_TOKENS must be 8, 16 or 32
#endif
#ifndef MLP_PACKED_ADDRESSED
#define MLP_PACKED_ADDRESSED 0
#endif

// Stable source grouping uses only device routes. One thread owns each source.
kernel void mlp_full_schedule(device const uint* ids [[buffer(0)]],
    constant uint* dims [[buffer(1)]], device uint* order [[buffer(2)]],
    device uint* schedule [[buffer(3)]], device uint* total [[buffer(4)]],
    uint expert [[thread_index_in_threadgroup]]) {
    threadgroup uint counts[512], starts[512], offsets[512];
    uint count = 0;
    for (uint p = 0; p < dims[0]; ++p) count += ids[p] == expert;
    counts[expert] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (expert == 0) {
        uint start = 0, tiles = 0;
        for (uint e = 0; e < 512; ++e) {
            starts[e] = start;
            offsets[e] = tiles;
            start += counts[e];
            tiles += (counts[e] + MLP_FULL_TILE_TOKENS - 1) / MLP_FULL_TILE_TOKENS;
        }
        total[0] = tiles;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = starts[expert];
    for (uint p = 0; p < dims[0]; ++p)
        if (ids[p] == expert) order[at++] = p;
    for (uint tile = 0; tile < (count + MLP_FULL_TILE_TOKENS - 1) / MLP_FULL_TILE_TOKENS; ++tile) {
        uint slot = offsets[expert] + tile;
        schedule[slot * 3] = expert;
        schedule[slot * 3 + 1] = starts[expert] + tile * MLP_FULL_TILE_TOKENS;
        schedule[slot * 3 + 2] = min(uint(MLP_FULL_TILE_TOKENS), count - tile * MLP_FULL_TILE_TOKENS);
    }
}

// Bounded large-prefill pairs arrive in stable source order. Search their
// contiguous ranges without rescanning every pair for every expert.
kernel void mlp_sorted_schedule(device const uint* sorted_ids [[buffer(0)]],
    constant uint* dims [[buffer(1)]], device uint* order [[buffer(2)]],
    device uint* schedule [[buffer(3)]], device uint* total [[buffer(4)]],
    uint expert [[thread_index_in_threadgroup]]) {
    threadgroup uint counts[512], offsets[512];
    const uint n = dims[0];
    uint lo = 0u, hi = n;
    while (lo < hi) {
        const uint mid = lo + ((hi - lo) >> 1u);
        if (sorted_ids[mid] < expert) lo = mid + 1u;
        else hi = mid;
    }
    const uint start = lo;
    hi = n;
    while (lo < hi) {
        const uint mid = lo + ((hi - lo) >> 1u);
        if (sorted_ids[mid] <= expert) lo = mid + 1u;
        else hi = mid;
    }
    const uint count = lo - start;
    counts[expert] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (expert == 0u) {
        uint tiles = 0u;
        for (uint e = 0u; e < 512u; ++e) {
            offsets[e] = tiles;
            tiles += (counts[e] + MLP_FULL_TILE_TOKENS - 1u) / MLP_FULL_TILE_TOKENS;
        }
        total[0] = tiles;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = start; p < start + count; ++p) order[p] = p;
    for (uint tile = 0u; tile < (count + MLP_FULL_TILE_TOKENS - 1u) / MLP_FULL_TILE_TOKENS; ++tile) {
        const uint slot = offsets[expert] + tile;
        schedule[slot * 3u] = expert;
        schedule[slot * 3u + 1u] = start + tile * MLP_FULL_TILE_TOKENS;
        schedule[slot * 3u + 2u] = min(uint(MLP_FULL_TILE_TOKENS), count - tile * MLP_FULL_TILE_TOKENS);
    }
}

// Bounded pools group by logical source, then resolve each scheduled source
// to its published physical slot before packed projection readers begin.
kernel void mlp_schedule_slots(
    device uint* schedule [[buffer(0)]],
    const device uint* total [[buffer(1)]],
    constant uint* source_to_slot [[buffer(2)]],
    uint tile [[thread_position_in_grid]]) {
    if (tile >= total[0]) return;
    schedule[tile * 3u] = source_to_slot[schedule[tile * 3u]];
}

// A bounded source group schedules compact local pairs, then restores each
// pair's original row/route index for packed projection and final reduction.
kernel void mlp_order_original_pairs(
    device uint* order [[buffer(0)]],
    const device uint* original_pairs [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) order[index] = original_pairs[order[index]];
}

kernel void mlp_full_route_order(device const uint* ids [[buffer(0)]],
    device uint* order [[buffer(1)]], uint row [[thread_position_in_grid]]) {
    uint routes[10];
    for (uint r = 0; r < 10; ++r) routes[r] = r;
    for (uint r = 1; r < 10; ++r) {
        uint value = routes[r], p = r;
        while (p > 0 && ids[row * 10 + routes[p - 1]] > ids[row * 10 + value]) {
            routes[p] = routes[p - 1];
            --p;
        }
        routes[p] = value;
    }
    for (uint r = 0; r < 10; ++r) order[row * 10 + r] = routes[r];
}

// The Qwen packed producer loads each 32-column code group and its metadata
// once per lane, then stages the lane's sixteen FP16 weights.
inline void mlp_grouped_weights(device const uchar* qs, device const uint* qh,
    device const uchar* scl, device const uchar* sch, device const uchar* sex,
    device const half* dv, threadgroup const half* table,
    ulong rid, uint col0, uint K, uint codec, threadgroup half* output) {
    const uint kg = col0 / 32u, sub = (col0 % 32u) / 16u;
    const device uint2* packed = reinterpret_cast<const device uint2*>(
        qs + rid * (K / 4u) + kg * 8u);
    const uint2 words = packed[0];
    const uint word = sub == 0u ? words.x : words.y;
    const uint sc = uint(scl[rid * (K / 32u) + kg]);
    const uint bit = (kg * 2u + sub) % 8u;
    const uint ex = (uint(sex[rid * (K / 128u) + kg / 4u]) >> bit) & 1u;
    const float d = float(dv[rid * (K / 256u) + kg / 8u]);
    if (codec == 3u) {
        const uint high = qh[rid * (K / 32u) + kg] >> (sub * 16u);
        const uint sign = (uint(sch[rid * (K / 128u) + kg / 4u]) >> bit) & 1u;
        const int scale = int(2u * ((sc >> (sub * 4u)) & 15u) + 1u)
            * (sign ? -1 : 1);
        const float scaled = d * float(scale);
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            const uint code = ((word >> (j * 2u)) & 3u)
                | (((high >> j) & 1u) << 2u);
            output[j] = half(scaled * float(table[code + ex * 8u]));
        }
    } else {
        const int scale = int((sc >> (sub * 4u)) & 15u) - 8;
        const float scaled = d * float(scale);
        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            const uint code = (word >> (j * 2u)) & 3u;
            output[j] = half(scaled * float(table[code + ex * 4u]));
        }
    }
}

kernel void mlp_full_gate_up(
    device const half* x [[buffer(0)]],
    device const uchar* gqs [[buffer(1)]], device const uint* gqh [[buffer(2)]],
    device const uchar* gscl [[buffer(3)]], device const uchar* gsch [[buffer(4)]],
    device const uchar* gsex [[buffer(5)]], device const half* gdv [[buffer(6)]],
    device const half* table [[buffer(7)]],
    device const uchar* uqs [[buffer(8)]], device const uint* uqh [[buffer(9)]],
    device const uchar* uscl [[buffer(10)]], device const uchar* usch [[buffer(11)]],
    device const uchar* usex [[buffer(12)]], device const half* udv [[buffer(13)]],
    device const uint* order [[buffer(14)]], device const uint* schedule [[buffer(15)]],
    device const uint* total [[buffer(16)]], device const half* sigmoid [[buffer(17)]],
    constant uint* dims [[buffer(18)]], device half* activation [[buffer(19)]],
    device half* gate [[buffer(20)]], device half* up [[buffer(21)]],
#if MLP_PACKED_ADDRESSED
    device const ulong* addresses [[buffer(22)]], constant uint& capacity [[buffer(23)]],
#endif
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint slot = group / 20, row0 = (group % 20) * 32;
    if (slot >= total[0]) return;
    uint expert = schedule[slot * 3], start = schedule[slot * 3 + 1];
    uint count = schedule[slot * 3 + 2];
#if MLP_PACKED_ADDRESSED
    if (expert >= capacity) return;
    const device ulong* bases = addresses + ulong(expert) * 18;
    gqs = reinterpret_cast<const device uchar*>(bases[0]);
    gqh = reinterpret_cast<const device uint*>(bases[1]);
    gscl = reinterpret_cast<const device uchar*>(bases[2]);
    gsch = reinterpret_cast<const device uchar*>(bases[3]);
    gsex = reinterpret_cast<const device uchar*>(bases[4]);
    gdv = reinterpret_cast<const device half*>(bases[5]);
    uqs = reinterpret_cast<const device uchar*>(bases[6]);
    uqh = reinterpret_cast<const device uint*>(bases[7]);
    uscl = reinterpret_cast<const device uchar*>(bases[8]);
    usch = reinterpret_cast<const device uchar*>(bases[9]);
    usex = reinterpret_cast<const device uchar*>(bases[10]);
    udv = reinterpret_cast<const device half*>(bases[11]);
    expert = 0;
#endif
    threadgroup half Ag[32 * 64], Au[32 * 64], Bs[64 * MLP_FULL_TILE_TOKENS], values[16];
    threadgroup float Cs[4][2][64];
    if (tid < (dims[1] == 3 ? 16u : 8u)) values[tid] = table[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Cg[MLP_FULL_TILE_TOKENS / 8], Cu[MLP_FULL_TILE_TOKENS / 8];
    for (uint nt = 0; nt < MLP_FULL_TILE_TOKENS / 8; ++nt) {
        Cg[nt] = simdgroup_float8x8(0.0f);
        Cu[nt] = simdgroup_float8x8(0.0f);
    }
    uint r = tid >> 2, q = tid & 3;
    ulong rid = ulong(expert) * 640 + row0 + r;
    uint token = tid % MLP_FULL_TILE_TOKENS;
    uint pair = token < count ? order[start + token] : 0;
    for (uint kb = 0; kb < 40; ++kb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint col0 = kb * 64u + q * 16u;
        mlp_grouped_weights(gqs, gqh, gscl, gsch, gsex, gdv, values,
            rid, col0, 2560u, dims[1], Ag + r * 64u + q * 16u);
        mlp_grouped_weights(uqs, uqh, uscl, usch, usex, udv, values,
            rid, col0, 2560u, dims[1], Au + r * 64u + q * 16u);
        uint slice = tid / MLP_FULL_TILE_TOKENS;
        constexpr uint values_per_thread = 64 * MLP_FULL_TILE_TOKENS / 128;
        for (uint j = 0; j < values_per_thread; ++j)
            Bs[(slice * values_per_thread + j) * MLP_FULL_TILE_TOKENS + token] = token < count
                ? x[(pair / 10) * 2560 + kb * 64 + slice * values_per_thread + j] : half(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub = 0; sub < 8; ++sub) {
            simdgroup_half8x8 ag, au;
            simdgroup_load(ag, Ag + sg * 8 * 64 + sub * 8, 64, 0, false);
            simdgroup_load(au, Au + sg * 8 * 64 + sub * 8, 64, 0, false);
            for (uint nt = 0; nt < MLP_FULL_TILE_TOKENS / 8; ++nt) {
                simdgroup_half8x8 b;
                simdgroup_load(b, Bs + sub * 8 * MLP_FULL_TILE_TOKENS + nt * 8, MLP_FULL_TILE_TOKENS, 0, false);
                simdgroup_multiply_accumulate(Cg[nt], ag, b, Cg[nt]);
                simdgroup_multiply_accumulate(Cu[nt], au, b, Cu[nt]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint nt = 0; nt < MLP_FULL_TILE_TOKENS / 8; ++nt) {
        simdgroup_store(Cg[nt], Cs[sg][0], 8, 0, false);
        simdgroup_store(Cu[nt], Cs[sg][1], 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < 256; index += 128) {
            uint group_id = index >> 6, element = index & 63;
            uint row = row0 + (element >> 3) + group_id * 8;
            uint token_id = nt * 8 + (element & 7);
            if (token_id < count) {
                uint output = order[start + token_id] * 640 + row;
                half g = half(Cs[group_id][0][element]), u = half(Cs[group_id][1][element]);
                half silu = g * sigmoid[as_type<ushort>(g)];
                activation[output] = silu * u;
                if (dims[3]) { gate[output] = g; up[output] = u; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void mlp_full_down(
    device const half* x [[buffer(0)]],
    device const uchar* qs [[buffer(1)]], device const uint* qh [[buffer(2)]],
    device const uchar* scl [[buffer(3)]], device const uchar* sch [[buffer(4)]],
    device const uchar* sex [[buffer(5)]], device const half* dv [[buffer(6)]],
    device const half* table [[buffer(7)]], device const uint* order [[buffer(8)]],
    device const uint* schedule [[buffer(9)]], device const uint* total [[buffer(10)]],
    constant uint* dims [[buffer(11)]], device half* output [[buffer(12)]],
#if MLP_PACKED_ADDRESSED
    device const ulong* addresses [[buffer(13)]], constant uint& capacity [[buffer(14)]],
#endif
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint slot = group / 80, row0 = (group % 80) * 32;
    if (slot >= total[0]) return;
    uint expert = schedule[slot * 3], start = schedule[slot * 3 + 1];
    uint count = schedule[slot * 3 + 2];
#if MLP_PACKED_ADDRESSED
    if (expert >= capacity) return;
    const device ulong* bases = addresses + ulong(expert) * 18 + 12;
    qs = reinterpret_cast<const device uchar*>(bases[0]);
    qh = reinterpret_cast<const device uint*>(bases[1]);
    scl = reinterpret_cast<const device uchar*>(bases[2]);
    sch = reinterpret_cast<const device uchar*>(bases[3]);
    sex = reinterpret_cast<const device uchar*>(bases[4]);
    dv = reinterpret_cast<const device half*>(bases[5]);
    expert = 0;
#endif
    threadgroup half As[32 * 64], Bs[64 * MLP_FULL_TILE_TOKENS], values[16];
    threadgroup float Cs[4][64];
    if (tid < (dims[2] == 3 ? 16u : 8u)) values[tid] = table[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 C[MLP_FULL_TILE_TOKENS / 8];
    for (uint nt = 0; nt < MLP_FULL_TILE_TOKENS / 8; ++nt) C[nt] = simdgroup_float8x8(0.0f);
    uint r = tid >> 2, q = tid & 3;
    ulong rid = ulong(expert) * 2560 + row0 + r;
    uint token = tid % MLP_FULL_TILE_TOKENS;
    uint pair = token < count ? order[start + token] : 0;
    for (uint kb = 0; kb < 10; ++kb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mlp_grouped_weights(qs, qh, scl, sch, sex, dv, values, rid,
            kb * 64u + q * 16u, 768u, dims[2], As + r * 64u + q * 16u);
        uint slice = tid / MLP_FULL_TILE_TOKENS;
        constexpr uint values_per_thread = 64 * MLP_FULL_TILE_TOKENS / 128;
        for (uint j = 0; j < values_per_thread; ++j)
            Bs[(slice * values_per_thread + j) * MLP_FULL_TILE_TOKENS + token] = token < count
                ? x[pair * 640 + kb * 64 + slice * values_per_thread + j] : half(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub = 0; sub < 8; ++sub) {
            simdgroup_half8x8 a;
            simdgroup_load(a, As + sg * 8 * 64 + sub * 8, 64, 0, false);
            for (uint nt = 0; nt < MLP_FULL_TILE_TOKENS / 8; ++nt) {
                simdgroup_half8x8 b;
                simdgroup_load(b, Bs + sub * 8 * MLP_FULL_TILE_TOKENS + nt * 8, MLP_FULL_TILE_TOKENS, 0, false);
                simdgroup_multiply_accumulate(C[nt], a, b, C[nt]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint nt = 0; nt < MLP_FULL_TILE_TOKENS / 8; ++nt) {
        simdgroup_store(C[nt], Cs[sg], 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < 256; index += 128) {
            uint group_id = index >> 6, element = index & 63;
            uint row = row0 + (element >> 3) + group_id * 8;
            uint token_id = nt * 8 + (element & 7);
            if (token_id < count)
                output[order[start + token_id] * 2560 + row] = half(Cs[group_id][element]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
