// Q6_K/Q8_0 loaders follow mlx-kquant (MIT), with ggml block layouts.
// Matrix fragments and traversal adapt MLX Steel gemm/mma.h (MIT).
// Copyright (c) 2026 Asher Feldman. Copyright (c) 2024 Apple Inc.
// See THIRD-PARTY-NOTICES.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float db_float(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline ushort db_round(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
inline float db_half(device const uchar* p) {
    ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}

// All loader products round to BF16 before entering the FP32 matrix fragment.
// This boundary is deliberately different from scale-factored decode QMV.
template<uint Codec>
inline ushort db_weight(device const uchar* weights, uint row, uint k, uint K) {
    if (Codec == 0) return reinterpret_cast<device const ushort*>(weights)[ulong(row) * K + k];
    if (Codec == 1) {
        device const uchar* block = weights + ulong(row) * (K / 256) * 210 + (k / 256) * 210;
        uint sub = k % 256;
        uint half_index = sub / 128;
        uint quarter = (sub % 128) / 32;
        uint lane = sub % 32;
        uchar low = block[half_index * 64 + (quarter % 2) * 32 + lane];
        uchar high = block[128 + half_index * 32 + lane];
        int code = int(((low >> (quarter >= 2 ? 4 : 0)) & 15)
            | (((high >> (quarter * 2)) & 3) << 4)) - 32;
        int scale = int(reinterpret_cast<device const char*>(block + 192)[sub / 16]);
        float scaled = db_half(block + 208) * float(scale);
        return db_round(scaled * float(code));
    }
    device const uchar* block = weights + ulong(row) * (K / 32) * 34 + (k / 32) * 34;
    int code = int(reinterpret_cast<device const char*>(block + 2)[k % 32]);
    return db_round(db_half(block) * float(code));
}

// BM=BN=64, BK=32, WM=WN=2. Each SIMD group owns 4x4 8x8 fragments.
// The fragment coordinates, K traversal and serpentine MMA order match Steel.
template<uint Codec>
inline void db_gemm(
    device const uchar* weights, device const ushort* input,
    device ushort* output, device float* partials, constant uint* dims,
    uint group, uint tid, uint simd, uint lane,
    threadgroup ushort* xs, threadgroup ushort* ws) {
    uint M = dims[0], N = dims[1], K = dims[2];
    uint partitions = dims[3], partition_k = dims[4];
    uint tiles_n = (N + 63) / 64, tiles_m = (M + 63) / 64;
    uint partition = group / (tiles_n * tiles_m);
    uint tile = group % (tiles_n * tiles_m);
    uint row_base = (tile / tiles_n) * 64;
    uint col_base = (tile % tiles_n) * 64;
    uint k_begin = partition * partition_k;
    uint k_end = partition + 1 == partitions ? K : k_begin + partition_k;
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint sm = (simd / 2) * 8 + fm;
    uint sn = (simd % 2) * 8 + fn;
    // Unroll fixed fragment loops to keep matrix indices compile-time constants.
    simdgroup_float8x8 accum[4][4];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) accum[i][j] = simdgroup_float8x8(0.0f);

    for (uint kb = k_begin; kb < k_end; kb += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = tid; p < 64 * 32; p += 128) {
            uint r = p / 32, k = p % 32;
            xs[r * 40 + k] = row_base + r < M && kb + k < k_end
                ? input[ulong(row_base + r) * K + kb + k] : ushort(0);
            ws[r * 40 + k] = col_base + r < N && kb + k < k_end
                ? db_weight<Codec>(weights, col_base + r, kb + k, K) : ushort(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll_count(4)
        for (uint kk = 0; kk < 32 && kb + kk < k_end; kk += 8) {
            simdgroup_float8x8 a[4], b[4];
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                a[i].thread_elements()[0] = db_float(xs[(sm + i * 16) * 40 + kk + fn]);
                a[i].thread_elements()[1] = db_float(xs[(sm + i * 16) * 40 + kk + fn + 1]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 4; ++j) {
                b[j].thread_elements()[0] = db_float(ws[(sn + j * 16) * 40 + kk + fm]);
                b[j].thread_elements()[1] = db_float(ws[(sn + j * 16 + 1) * 40 + kk + fm]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                #pragma clang loop unroll(full)
                for (uint n = 0; n < 4; ++n) {
                    uint j = (i % 2) ? 3 - n : n;
                    simdgroup_multiply_accumulate(accum[i][j], a[i], b[j], accum[i][j]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        uint row = row_base + sm + i * 16;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2; ++e) {
                uint col = col_base + sn + j * 16 + e;
                if (row < M && col < N) {
                    float value = accum[i][j].thread_elements()[e];
                    ulong index = ulong(row) * N + col;
                    if (partitions == 1) output[index] = db_round(value);
                    else partials[ulong(partition) * M * N + index] = value;
                }
            }
        }
    }
}

#define DB_KERNEL(NAME, CODEC) \
kernel void NAME(device const uchar* w [[buffer(0)]], \
    device const ushort* x [[buffer(1)]], device ushort* y [[buffer(2)]], \
    device float* partials [[buffer(3)]], constant uint* dims [[buffer(4)]], \
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup ushort xs[64 * 40], ws[64 * 40]; \
    db_gemm<CODEC>(w, x, y, partials, dims, group, tid, simd, lane, xs, ws); \
}
DB_KERNEL(dense_batch_bf16, 0)
DB_KERNEL(dense_batch_q6_k, 1)
DB_KERNEL(dense_batch_q8_0, 2)

kernel void dense_batch_split_sum(
    device const float* partials [[buffer(0)]], device ushort* output [[buffer(1)]],
    constant uint* dims [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    uint count = dims[0] * dims[1];
    if (index >= count) return;
    float sum = 0.0f;
    for (uint part = 0; part < dims[3]; ++part) sum += partials[ulong(part) * count + index];
    output[index] = db_round(sum);
}

// These wrappers share decode helpers from kquant.metal. Rust concatenates
// both sources before compilation.
template<uint Codec>
inline void db_qmv(device const uchar* w, device const ushort* x,
    device ushort* y, constant uint* dims, uint group, uint simd, uint lane) {
    uint N = dims[1], K = dims[2];
    uint groups_per_row = (N + 7) / 8;
    uint row = group / groups_per_row;
    x += ulong(row) * K;
    device uchar* destination = reinterpret_cast<device uchar*>(y + ulong(row) * N);
    if (Codec == 0) kq_raw_bf16_matvec<false>(w, x, destination, K, N, group % groups_per_row, simd, lane);
    else if (Codec == 1) kq_q6_matvec<false>(w, x, destination, K, N, group % groups_per_row, simd, lane);
    else kq_q8_matvec<false>(w, x, destination, K, N, group % groups_per_row, simd, lane);
}

#define DB_QMV_KERNEL(NAME, CODEC) \
kernel void NAME(device const uchar* w [[buffer(0)]], device const ushort* x [[buffer(1)]], \
    device ushort* y [[buffer(2)]], device float* unused [[buffer(3)]], \
    constant uint* dims [[buffer(4)]], uint group [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    db_qmv<CODEC>(w, x, y, dims, group, simd, lane); \
}
DB_QMV_KERNEL(dense_batch_qmv_bf16, 0)
DB_QMV_KERNEL(dense_batch_qmv_q6_k, 1)
DB_QMV_KERNEL(dense_batch_qmv_q8_0, 2)

// ggml/MLX-kquant's MV-ext reconstructs FP32 chunks without a BF16 weight
// round. Keep its scale-product then offset subtraction, including byte shifts.
template<uint Codec>
inline float4x4 db_ext_chunk(device const uchar* block, uint chunk) {
    float4x4 values;
    if (Codec == 2) {
        float d = db_half(block);
        device const char* q = reinterpret_cast<device const char*>(block + 2) + chunk * 16;
        for (uint i = 0; i < 16; ++i) values[i / 4][i % 4] = float(q[i]) * d;
    } else {
        device const ushort* ql = reinterpret_cast<device const ushort*>(block);
        device const ushort* qh = reinterpret_cast<device const ushort*>(block + 128);
        ql += 32 * (chunk / 8) + 16 * ((chunk / 2) & 1) + 8 * (chunk & 1);
        qh += 16 * (chunk / 8) + 8 * (chunk & 1);
        float d = db_half(block + 208);
        float scale = float(reinterpret_cast<device const char*>(block + 192)[chunk]);
        uint quarter = (chunk / 2) & 3;
        uint high_mask = quarter > 1 ? (quarter > 2 ? 0xC0C0C0C0u : 0x30303030u)
            : (quarter > 0 ? 0x0C0C0C0Cu : 0x03030303u);
        uint low_mask = quarter > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
        float offset = d * scale * 32.0f;
        float d0 = d * scale;
        float4 ds(d0, d0 / 256.0f, d0 / (256.0f * 256.0f), d0 / (256.0f * 256.0f * 256.0f));
        uint shift_high_right = quarter > 2 ? 2 : 0;
        uint shift_high_left = quarter > 1 ? 0 : (quarter > 0 ? 2 : 4);
        uint shift_low = quarter > 1 ? 4 : 0;
        for (uint i = 0; i < 4; ++i) {
            uint low = (uint(ql[2 * i]) | (uint(ql[2 * i + 1]) << 16)) & low_mask;
            uint high = (uint(qh[2 * i]) | (uint(qh[2 * i + 1]) << 16)) & high_mask;
            uint q = ((high << shift_high_left) >> shift_high_right) | (low >> shift_low);
            for (uint j = 0; j < 4; ++j) values[i][j] = ds[j] * float(q & (0xFFu << (j * 8))) - offset;
        }
    }
    return values;
}

template<uint Codec>
inline void db_mv_ext(device const uchar* w, device const ushort* x, device ushort* y,
    constant uint* dims, uint group, uint simd, uint lane) {
    uint M = dims[0], N = dims[1], K = dims[2];
    uint tx = lane % 8, row = group * 8 + simd * 4 + lane / 8;
    uint block_width = Codec == 1 ? 256 : 32;
    uint block_bytes = Codec == 1 ? 210 : 34;
    device const uchar* wr = w + ulong(row < N ? row : 0) * (K / block_width) * block_bytes;
    float sums[12] = {0.0f};
    for (uint chunk = tx; chunk * 16 < K; chunk += 8) {
        float4x4 weights = db_ext_chunk<Codec>(wr + (chunk * 16 / block_width) * block_bytes,
            chunk % (block_width / 16));
        for (uint m = 0; m < M; ++m) {
            device const ushort* activation = x + ulong(m) * K + chunk * 16;
            float4 a[4];
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j) a[i][j] = db_float(activation[i * 4 + j]);
            sums[m] += dot(weights[0], a[0]) + dot(weights[1], a[1])
                + dot(weights[2], a[2]) + dot(weights[3], a[3]);
        }
    }
    for (uint m = 0; m < M; ++m) {
        sums[m] += simd_shuffle_down(sums[m], 4);
        sums[m] += simd_shuffle_down(sums[m], 2);
        sums[m] += simd_shuffle_down(sums[m], 1);
        if (tx == 0 && row < N) y[ulong(m) * N + row] = db_round(sums[m]);
    }
}
#define DB_EXT_KERNEL(NAME, CODEC) \
kernel void NAME(device const uchar* w [[buffer(0)]], device const ushort* x [[buffer(1)]], \
    device ushort* y [[buffer(2)]], device float* unused [[buffer(3)]], \
    constant uint* dims [[buffer(4)]], uint group [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    db_mv_ext<CODEC>(w, x, y, dims, group, simd, lane); \
}
DB_EXT_KERNEL(dense_batch_ext_q6_k, 1)
DB_EXT_KERNEL(dense_batch_ext_q8_0, 2)
