// Q6_K and Q8_0 decode and QMV folds adapted from mlx-kquant (MIT).
// Copyright (c) 2026 Asher Feldman. See THIRD-PARTY-NOTICES.
// GGUF block decode math derives from ggml (llama.cpp, MIT).
#include <metal_stdlib>
using namespace metal;

// Q6_K and Q8_0 wires and per-lane QMV folds follow mlx-kquant's
// kq_q6_k_qmv_fast_impl and kq_q8_0_qmv_fast_impl. The matrix is row-major.
inline float kq_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline ushort kq_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
inline float kq_f16(const device uchar* address) {
    ushort bits = ushort(address[0]) | (ushort(address[1]) << 8);
    return float(as_type<half>(bits));
}
template<bool FP32>
inline void kq_store(device uchar* output, uint row, float value) {
    if (FP32) reinterpret_cast<device float*>(output)[row] = value;
    else reinterpret_cast<device ushort*>(output)[row] = kq_to_bf16(value);
}

template<bool FP32>
inline void kq_q6_matvec(
    device const uchar* w, device const ushort* x, device uchar* y,
    uint K, uint N, uint group, uint sg, uint lane) {
    const uint row0 = group * 8 + sg * 4;
    const uint tid_lane = lane / 2;
    const uint ix = lane % 2;
    const uint ip = tid_lane / 8;
    const uint il = tid_lane % 8;
    const uint l0 = 4 * il;
    const uint scale_offset = 8 * ip + l0 / 16;
    const uint row_bytes = K / 256 * 210;
    const uint blocks = K / 256;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint ib = ix; ib < blocks; ib += 2) {
        uint x_base = ib * 256 + 128 * ip + l0;
        float xv[16];
        for (uint l = 0; l < 4; ++l) {
            xv[4*l] = kq_bf16(x[x_base + l]);
            xv[4*l+1] = kq_bf16(x[x_base + l + 32]);
            xv[4*l+2] = kq_bf16(x[x_base + l + 64]);
            xv[4*l+3] = kq_bf16(x[x_base + l + 96]);
        }
        for (uint r = 0; r < 4; ++r) {
            uint row = row0 + r;
            if (row >= N) continue;
            device const uchar* block = w + row * row_bytes + ib * 210;
            device const uchar* q1 = block + 64 * ip + l0;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh = block + 128 + 32 * ip + l0;
            device const char* sc = reinterpret_cast<device const char*>(block + 192 + scale_offset);
            float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint l = 0; l < 4; ++l) {
                uchar a = q1[l], b = q2[l], h = qh[l];
                int v0 = int((a & 15) | ((h & 3) << 4)) - 32;
                int v1 = int((b & 15) | ((h & 12) << 2)) - 32;
                int v2 = int((a >> 4) | (h & 48)) - 32;
                int v3 = int((b >> 4) | ((h & 192) >> 2)) - 32;
                sums[0] += xv[4*l] * float(v0);
                sums[1] += xv[4*l+1] * float(v1);
                sums[2] += xv[4*l+2] * float(v2);
                sums[3] += xv[4*l+3] * float(v3);
            }
            float d = kq_f16(block + 208);
            result[r] += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2])
                            + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
        }
    }
    for (uint r = 0; r < 4; ++r) {
        float total = simd_sum(result[r]);
        uint row = row0 + r;
        if (lane == 0 && row < N) kq_store<FP32>(y, row, total);
    }
}

template<bool FP32>
inline void kq_q8_matvec(
    device const uchar* w, device const ushort* x, device uchar* y,
    uint K, uint N, uint group, uint sg, uint lane) {
    const uint row0 = group * 8 + sg * 4;
    const uint row_bytes = K / 32 * 34;
    const uint lane_offset = lane * 8;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = lane_offset; k < K; k += 256) {
        float xv[8];
        for (uint i = 0; i < 8; ++i) xv[i] = kq_bf16(x[k + i]);
        uint block_id = k / 32;
        uint within = k % 32;
        for (uint r = 0; r < 4; ++r) {
            uint row = row0 + r;
            if (row >= N) continue;
            device const uchar* block = w + row * row_bytes + block_id * 34;
            float d = kq_f16(block);
            device const char* q = reinterpret_cast<device const char*>(block + 2 + within);
            float partial = 0.0f;
            for (uint i = 0; i < 8; ++i) partial += xv[i] * float(q[i]);
            result[r] += d * partial;
        }
    }
    for (uint r = 0; r < 4; ++r) {
        float total = simd_sum(result[r]);
        uint row = row0 + r;
        if (lane == 0 && row < N) kq_store<FP32>(y, row, total);
    }
}

template<bool FP32>
inline void kq_raw_bf16_matvec(
    device const uchar* packed_w, device const ushort* x, device uchar* y,
    uint K, uint N, uint group, uint sg, uint lane) {
    device const ushort* w = reinterpret_cast<device const ushort*>(packed_w);
    uint row0 = group * 8 + sg * 4;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = lane; k < K; k += 32) {
        float xv = kq_bf16(x[k]);
        for (uint r = 0; r < 4; ++r) {
            uint row = row0 + r;
            if (row < N) result[r] += xv * kq_bf16(w[row * K + k]);
        }
    }
    for (uint r = 0; r < 4; ++r) {
        float total = simd_sum(result[r]);
        uint row = row0 + r;
        if (lane == 0 && row < N) kq_store<FP32>(y, row, total);
    }
}

#define DEFINE_MATVEC(FORMAT, FP, NAME) \
kernel void NAME(device const uchar* w [[buffer(0)]], \
                 device const ushort* x [[buffer(1)]], \
                 device uchar* y [[buffer(2)]], \
                 constant uint* dims [[buffer(3)]], \
                 uint group [[threadgroup_position_in_grid]], \
                 uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { \
    FORMAT<FP>(w, x, y, dims[0], dims[1], group, sg, lane); \
}
DEFINE_MATVEC(kq_q6_matvec, false, dense_q6_k_bf16)
DEFINE_MATVEC(kq_q6_matvec, true, dense_q6_k_f32)
DEFINE_MATVEC(kq_q8_matvec, false, dense_q8_0_bf16)
DEFINE_MATVEC(kq_q8_matvec, true, dense_q8_0_f32)
DEFINE_MATVEC(kq_raw_bf16_matvec, false, dense_raw_bf16_bf16)
DEFINE_MATVEC(kq_raw_bf16_matvec, true, dense_raw_bf16_f32)
