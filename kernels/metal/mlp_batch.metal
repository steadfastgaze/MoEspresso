// Row-major shared MLP and expert-major BF16 reduction follow Qwen's MLX
// operations. Scalar helpers are shared from mlp.metal (Apple MIT).
// See THIRD-PARTY-NOTICES.
#include <metal_simdgroup_matrix>

// Reconstruct 32 weights per thread from the caller's active source slots,
// sharing metadata loads and storing groups of four half values.
kernel void mlp_batch_dequant(
    device const uchar* qs [[buffer(0)]], device const uint* qh [[buffer(1)]],
    device const uchar* scl [[buffer(2)]], device const uchar* sch [[buffer(3)]],
    device const uchar* sex [[buffer(4)]], device const half* dv [[buffer(5)]],
    device const half* table [[buffer(6)]], device const uint* slots [[buffer(7)]],
    constant uint* dims [[buffer(8)]], device half* output [[buffer(9)]],
    uint i [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint K = dims[0], N = dims[1], base = dims[2], range = dims[3];
    uint groups = K / 32, kg = i % groups, linear_row = i / groups;
    uint row = linear_row % range, source = linear_row / range;
    ulong rid = ulong(slots[source]) * N + base + row;
    threadgroup half tgV[16];
    if (tid < (dims[4] == 3u ? 16u : 8u)) tgV[tid] = table[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint2 q = *((device const uint2*)(qs + rid * (K / 4) + kg * 8));
    uint sc = uint(scl[rid * (K / 32) + kg]);
    uint ex = uint(sex[rid * (K / 128) + kg / 4]) >> ((kg * 2) & 7u);
    float d = float(dv[rid * (K / 256) + kg / 8]);
    device half4* out = (device half4*)(output + ulong(linear_row) * K + kg * 32);
    if (dims[4] == 3u) {
        uint high = qh[rid * (K / 32) + kg];
        uint signs = uint(sch[rid * (K / 128) + kg / 4]) >> ((kg * 2) & 7u);
        float d0 = d * float(int(2u * (sc & 15u) + 1u) * ((signs & 1u) ? -1 : 1));
        float d1 = d * float(int(2u * (sc >> 4u) + 1u) * ((signs & 2u) ? -1 : 1));
        uint t0 = (ex & 1u) * 8u, t1 = ((ex >> 1u) & 1u) * 8u;
        float w0 = d0 * float(tgV[t0 + (((q.x >> 0u) & 3u) | (((high >> 0u) & 1u) << 2u))]);
        float w1 = d0 * float(tgV[t0 + (((q.x >> 2u) & 3u) | (((high >> 1u) & 1u) << 2u))]);
        float w2 = d0 * float(tgV[t0 + (((q.x >> 4u) & 3u) | (((high >> 2u) & 1u) << 2u))]);
        float w3 = d0 * float(tgV[t0 + (((q.x >> 6u) & 3u) | (((high >> 3u) & 1u) << 2u))]);
        float w4 = d0 * float(tgV[t0 + (((q.x >> 8u) & 3u) | (((high >> 4u) & 1u) << 2u))]);
        float w5 = d0 * float(tgV[t0 + (((q.x >> 10u) & 3u) | (((high >> 5u) & 1u) << 2u))]);
        float w6 = d0 * float(tgV[t0 + (((q.x >> 12u) & 3u) | (((high >> 6u) & 1u) << 2u))]);
        float w7 = d0 * float(tgV[t0 + (((q.x >> 14u) & 3u) | (((high >> 7u) & 1u) << 2u))]);
        float w8 = d0 * float(tgV[t0 + (((q.x >> 16u) & 3u) | (((high >> 8u) & 1u) << 2u))]);
        float w9 = d0 * float(tgV[t0 + (((q.x >> 18u) & 3u) | (((high >> 9u) & 1u) << 2u))]);
        float w10 = d0 * float(tgV[t0 + (((q.x >> 20u) & 3u) | (((high >> 10u) & 1u) << 2u))]);
        float w11 = d0 * float(tgV[t0 + (((q.x >> 22u) & 3u) | (((high >> 11u) & 1u) << 2u))]);
        float w12 = d0 * float(tgV[t0 + (((q.x >> 24u) & 3u) | (((high >> 12u) & 1u) << 2u))]);
        float w13 = d0 * float(tgV[t0 + (((q.x >> 26u) & 3u) | (((high >> 13u) & 1u) << 2u))]);
        float w14 = d0 * float(tgV[t0 + (((q.x >> 28u) & 3u) | (((high >> 14u) & 1u) << 2u))]);
        float w15 = d0 * float(tgV[t0 + (((q.x >> 30u) & 3u) | (((high >> 15u) & 1u) << 2u))]);
        float w16 = d1 * float(tgV[t1 + (((q.y >> 0u) & 3u) | (((high >> 16u) & 1u) << 2u))]);
        float w17 = d1 * float(tgV[t1 + (((q.y >> 2u) & 3u) | (((high >> 17u) & 1u) << 2u))]);
        float w18 = d1 * float(tgV[t1 + (((q.y >> 4u) & 3u) | (((high >> 18u) & 1u) << 2u))]);
        float w19 = d1 * float(tgV[t1 + (((q.y >> 6u) & 3u) | (((high >> 19u) & 1u) << 2u))]);
        float w20 = d1 * float(tgV[t1 + (((q.y >> 8u) & 3u) | (((high >> 20u) & 1u) << 2u))]);
        float w21 = d1 * float(tgV[t1 + (((q.y >> 10u) & 3u) | (((high >> 21u) & 1u) << 2u))]);
        float w22 = d1 * float(tgV[t1 + (((q.y >> 12u) & 3u) | (((high >> 22u) & 1u) << 2u))]);
        float w23 = d1 * float(tgV[t1 + (((q.y >> 14u) & 3u) | (((high >> 23u) & 1u) << 2u))]);
        float w24 = d1 * float(tgV[t1 + (((q.y >> 16u) & 3u) | (((high >> 24u) & 1u) << 2u))]);
        float w25 = d1 * float(tgV[t1 + (((q.y >> 18u) & 3u) | (((high >> 25u) & 1u) << 2u))]);
        float w26 = d1 * float(tgV[t1 + (((q.y >> 20u) & 3u) | (((high >> 26u) & 1u) << 2u))]);
        float w27 = d1 * float(tgV[t1 + (((q.y >> 22u) & 3u) | (((high >> 27u) & 1u) << 2u))]);
        float w28 = d1 * float(tgV[t1 + (((q.y >> 24u) & 3u) | (((high >> 28u) & 1u) << 2u))]);
        float w29 = d1 * float(tgV[t1 + (((q.y >> 26u) & 3u) | (((high >> 29u) & 1u) << 2u))]);
        float w30 = d1 * float(tgV[t1 + (((q.y >> 28u) & 3u) | (((high >> 30u) & 1u) << 2u))]);
        float w31 = d1 * float(tgV[t1 + (((q.y >> 30u) & 3u) | (((high >> 31u) & 1u) << 2u))]);
        out[0] = half4(half(w0), half(w1), half(w2), half(w3));
        out[1] = half4(half(w4), half(w5), half(w6), half(w7));
        out[2] = half4(half(w8), half(w9), half(w10), half(w11));
        out[3] = half4(half(w12), half(w13), half(w14), half(w15));
        out[4] = half4(half(w16), half(w17), half(w18), half(w19));
        out[5] = half4(half(w20), half(w21), half(w22), half(w23));
        out[6] = half4(half(w24), half(w25), half(w26), half(w27));
        out[7] = half4(half(w28), half(w29), half(w30), half(w31));
    } else {
        float d0 = d * float(int(sc & 15u) - 8);
        float d1 = d * float(int(sc >> 4u) - 8);
        uint t0 = (ex & 1u) * 4u, t1 = ((ex >> 1u) & 1u) * 4u;
        float g00 = float(tgV[t0 + 0u]);
        float g01 = float(tgV[t0 + 1u]);
        float g02 = float(tgV[t0 + 2u]);
        float g03 = float(tgV[t0 + 3u]);
        float g10 = float(tgV[t1 + 0u]);
        float g11 = float(tgV[t1 + 1u]);
        float g12 = float(tgV[t1 + 2u]);
        float g13 = float(tgV[t1 + 3u]);
        float w0 = d0 * ((((q.x >> 0u) & 3u) & 2u) ? ((((q.x >> 0u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 0u) & 3u) & 1u) ? g01 : g00));
        float w1 = d0 * ((((q.x >> 2u) & 3u) & 2u) ? ((((q.x >> 2u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 2u) & 3u) & 1u) ? g01 : g00));
        float w2 = d0 * ((((q.x >> 4u) & 3u) & 2u) ? ((((q.x >> 4u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 4u) & 3u) & 1u) ? g01 : g00));
        float w3 = d0 * ((((q.x >> 6u) & 3u) & 2u) ? ((((q.x >> 6u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 6u) & 3u) & 1u) ? g01 : g00));
        float w4 = d0 * ((((q.x >> 8u) & 3u) & 2u) ? ((((q.x >> 8u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 8u) & 3u) & 1u) ? g01 : g00));
        float w5 = d0 * ((((q.x >> 10u) & 3u) & 2u) ? ((((q.x >> 10u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 10u) & 3u) & 1u) ? g01 : g00));
        float w6 = d0 * ((((q.x >> 12u) & 3u) & 2u) ? ((((q.x >> 12u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 12u) & 3u) & 1u) ? g01 : g00));
        float w7 = d0 * ((((q.x >> 14u) & 3u) & 2u) ? ((((q.x >> 14u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 14u) & 3u) & 1u) ? g01 : g00));
        float w8 = d0 * ((((q.x >> 16u) & 3u) & 2u) ? ((((q.x >> 16u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 16u) & 3u) & 1u) ? g01 : g00));
        float w9 = d0 * ((((q.x >> 18u) & 3u) & 2u) ? ((((q.x >> 18u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 18u) & 3u) & 1u) ? g01 : g00));
        float w10 = d0 * ((((q.x >> 20u) & 3u) & 2u) ? ((((q.x >> 20u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 20u) & 3u) & 1u) ? g01 : g00));
        float w11 = d0 * ((((q.x >> 22u) & 3u) & 2u) ? ((((q.x >> 22u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 22u) & 3u) & 1u) ? g01 : g00));
        float w12 = d0 * ((((q.x >> 24u) & 3u) & 2u) ? ((((q.x >> 24u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 24u) & 3u) & 1u) ? g01 : g00));
        float w13 = d0 * ((((q.x >> 26u) & 3u) & 2u) ? ((((q.x >> 26u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 26u) & 3u) & 1u) ? g01 : g00));
        float w14 = d0 * ((((q.x >> 28u) & 3u) & 2u) ? ((((q.x >> 28u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 28u) & 3u) & 1u) ? g01 : g00));
        float w15 = d0 * ((((q.x >> 30u) & 3u) & 2u) ? ((((q.x >> 30u) & 3u) & 1u) ? g03 : g02) : ((((q.x >> 30u) & 3u) & 1u) ? g01 : g00));
        float w16 = d1 * ((((q.y >> 0u) & 3u) & 2u) ? ((((q.y >> 0u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 0u) & 3u) & 1u) ? g11 : g10));
        float w17 = d1 * ((((q.y >> 2u) & 3u) & 2u) ? ((((q.y >> 2u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 2u) & 3u) & 1u) ? g11 : g10));
        float w18 = d1 * ((((q.y >> 4u) & 3u) & 2u) ? ((((q.y >> 4u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 4u) & 3u) & 1u) ? g11 : g10));
        float w19 = d1 * ((((q.y >> 6u) & 3u) & 2u) ? ((((q.y >> 6u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 6u) & 3u) & 1u) ? g11 : g10));
        float w20 = d1 * ((((q.y >> 8u) & 3u) & 2u) ? ((((q.y >> 8u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 8u) & 3u) & 1u) ? g11 : g10));
        float w21 = d1 * ((((q.y >> 10u) & 3u) & 2u) ? ((((q.y >> 10u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 10u) & 3u) & 1u) ? g11 : g10));
        float w22 = d1 * ((((q.y >> 12u) & 3u) & 2u) ? ((((q.y >> 12u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 12u) & 3u) & 1u) ? g11 : g10));
        float w23 = d1 * ((((q.y >> 14u) & 3u) & 2u) ? ((((q.y >> 14u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 14u) & 3u) & 1u) ? g11 : g10));
        float w24 = d1 * ((((q.y >> 16u) & 3u) & 2u) ? ((((q.y >> 16u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 16u) & 3u) & 1u) ? g11 : g10));
        float w25 = d1 * ((((q.y >> 18u) & 3u) & 2u) ? ((((q.y >> 18u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 18u) & 3u) & 1u) ? g11 : g10));
        float w26 = d1 * ((((q.y >> 20u) & 3u) & 2u) ? ((((q.y >> 20u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 20u) & 3u) & 1u) ? g11 : g10));
        float w27 = d1 * ((((q.y >> 22u) & 3u) & 2u) ? ((((q.y >> 22u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 22u) & 3u) & 1u) ? g11 : g10));
        float w28 = d1 * ((((q.y >> 24u) & 3u) & 2u) ? ((((q.y >> 24u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 24u) & 3u) & 1u) ? g11 : g10));
        float w29 = d1 * ((((q.y >> 26u) & 3u) & 2u) ? ((((q.y >> 26u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 26u) & 3u) & 1u) ? g11 : g10));
        float w30 = d1 * ((((q.y >> 28u) & 3u) & 2u) ? ((((q.y >> 28u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 28u) & 3u) & 1u) ? g11 : g10));
        float w31 = d1 * ((((q.y >> 30u) & 3u) & 2u) ? ((((q.y >> 30u) & 3u) & 1u) ? g13 : g12) : ((((q.y >> 30u) & 3u) & 1u) ? g11 : g10));
        out[0] = half4(half(w0), half(w1), half(w2), half(w3));
        out[1] = half4(half(w4), half(w5), half(w6), half(w7));
        out[2] = half4(half(w8), half(w9), half(w10), half(w11));
        out[3] = half4(half(w12), half(w13), half(w14), half(w15));
        out[4] = half4(half(w16), half(w17), half(w18), half(w19));
        out[5] = half4(half(w20), half(w21), half(w22), half(w23));
        out[6] = half4(half(w24), half(w25), half(w26), half(w27));
        out[7] = half4(half(w28), half(w29), half(w30), half(w31));
    }
}

// MLX Steel gather_mm_rhs uses BM16/BN64/BK16 and WM1/WN2. Source-homogeneous
// tiles preserve its dot-product order. Shared inputs remain in FP16 operand
// fragments, while accumulator fragments retain the reference FP32 type.
// Steel BlockLoader copies four A values and sixteen B values per thread.
// Half alignment permits the padded shared rows to retain their 24-value stride.
struct alignas(2) MlpReadA { uchar bytes[4 * sizeof(half)]; };
struct alignas(2) MlpReadB { uchar bytes[16 * sizeof(half)]; };
kernel void mlp_batch_sorted_mm(
    device const half* input [[buffer(0)]], device const half* weights [[buffer(1)]],
    device const uint* tiles [[buffer(2)]], constant uint* dims [[buffer(3)]],
    device half* output [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint K = dims[0], stride = dims[1], range = dims[2], base = dims[3];
    uint tn = range / 64, tile = group / tn, cb = (group % tn) * 64;
    uint rb = tiles[tile * 3], end = tiles[tile * 3 + 1], source = tiles[tile * 3 + 2];
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint sn = sg * 8 + fn;
    threadgroup half a_shared[16 * 24], b_shared[64 * 24];
    uint ar = tid / 4, ak = (tid % 4) * 4;
    bool a_valid = rb + ar < end;
    device const half* a_src = input + (rb + (a_valid ? ar : 0)) * K + ak;
    device const half* b_src = weights + (source * range + cb + tid) * K;
    simdgroup_float8x8 accum[2][4];
    for (uint r = 0; r < 2; ++r)
        for (uint c = 0; c < 4; ++c) accum[r][c] = simdgroup_float8x8(0.0f);
    for (uint kb = 0; kb < K; kb += 16) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (a_valid)
            *((threadgroup MlpReadA*)(a_shared + ar * 24 + ak)) =
                *((device const MlpReadA*)a_src);
        else
            for (uint j = 0; j < 4; ++j) a_shared[ar * 24 + ak + j] = half(0);
        *((threadgroup MlpReadB*)(b_shared + tid * 24)) =
            *((device const MlpReadB*)b_src);
        a_src += 16;
        b_src += 16;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 16; kk += 8) {
            simdgroup_half8x8 a[2], b[4];
            for (uint r = 0; r < 2; ++r) {
                a[r].thread_elements()[0] = a_shared[(fm + r * 8) * 24 + kk + fn];
                a[r].thread_elements()[1] = a_shared[(fm + r * 8) * 24 + kk + fn + 1];
            }
            for (uint c = 0; c < 4; ++c) {
                b[c].thread_elements()[0] = b_shared[(sn + c * 16) * 24 + kk + fm];
                b[c].thread_elements()[1] = b_shared[(sn + c * 16 + 1) * 24 + kk + fm];
            }
            for (uint r = 0; r < 2; ++r)
                for (uint n = 0; n < 4; ++n) {
                    uint c = r % 2 ? 3 - n : n;
                    simdgroup_multiply_accumulate(accum[r][c], a[r], b[c], accum[r][c]);
                }
        }
    }
    for (uint r = 0; r < 2; ++r)
        for (uint c = 0; c < 4; ++c)
            for (uint e = 0; e < 2; ++e)
                if (rb + fm + r * 8 < end)
                    output[(rb + fm + r * 8) * stride + base + cb + sn + c * 16 + e]
                        = half(accum[r][c].thread_elements()[e]);
}

kernel void mlp_batch_gather_half(device const ushort* hidden [[buffer(0)]],
    device const uint* pairs [[buffer(1)]], device half* output [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    output[i] = half(mlp_bf16(hidden[(pairs[i / 2560] / 10) * 2560 + i % 2560]));
}

kernel void mlp_batch_scatter_half(device const half* input [[buffer(0)]],
    device const uint* pairs [[buffer(1)]], device half* output [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    output[pairs[i / 2560] * 2560 + i % 2560] = input[i];
}

kernel void mlp_batch_activate(device const ushort* gate [[buffer(0)]],
    device const ushort* up [[buffer(1)]], device ushort* output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    float g = mlp_bf16(gate[index]);
    float silu = mlp_bf16(mlp_round(g * mlp_sigmoid(g)));
    output[index] = mlp_round(silu * mlp_bf16(up[index]));
}

kernel void mlp_batch_finish(device const ushort* routed [[buffer(0)]],
    device const ushort* down [[buffer(1)]], device const ushort* logit [[buffer(2)]],
    device ushort* shared [[buffer(3)]], device ushort* output [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    float scalar = mlp_sigmoid(mlp_bf16(logit[index / 2560]));
    ushort product = mlp_round(scalar * mlp_bf16(down[index]));
    shared[index] = product;
    output[index] = mlp_round(mlp_bf16(routed[index]) + mlp_bf16(product));
}

kernel void mlp_batch_bf16_to_fp16(device const ushort* input [[buffer(0)]],
    device half* output [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    output[index] = half(mlp_bf16(input[index]));
}

kernel void mlp_batch_reduce(device const half* down [[buffer(0)]],
    device const ushort* scores [[buffer(1)]], device const uint* order [[buffer(2)]],
    device ushort* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    uint row = index / 2560, column = index % 2560;
    ushort sum = 0;
    for (uint position = 0; position < 10; ++position) {
        uint route = row * 10 + order[row * 10 + position];
        ushort value = mlp_round(float(down[route * 2560 + column]));
        ushort product = mlp_round(mlp_bf16(value) * mlp_bf16(scores[route]));
        sum = mlp_round(mlp_bf16(sum) + mlp_bf16(product));
    }
    output[index] = sum;
}
