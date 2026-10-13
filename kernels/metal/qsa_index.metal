// Qwen QSA index compression follows the MLX reference's FP32 mean and
// zero-centered norm, with BF16 boundaries before normalization and RoPE.
#include <metal_stdlib>
using namespace metal;

inline float qi_float(ushort v) { return as_type<float>(uint(v) << 16u); }
inline ushort qi_round(float v) {
    uint bits = as_type<uint>(v);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

kernel void qsa_index_compress(
    device const ushort* raw [[buffer(0)]],
    device const ushort* weight [[buffer(1)]],
    device const ushort* cosine [[buffer(2)]],
    device const ushort* sine [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]) {
    float values[4];
    float partial = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint d = tid * 4 + i;
        float sum = 0.0f;
        for (uint row = 0; row < 4; ++row) sum += qi_float(raw[row * 128 + d]);
        values[i] = qi_float(qi_round(sum * 0.25f));
        partial += values[i] * values[i];
    }
    float inv = precise::rsqrt(simd_sum(partial) / 128.0f + 1e-6f);
    threadgroup ushort normalized[128];
    for (uint i = 0; i < 4; ++i) {
        uint d = tid * 4 + i;
        normalized[d] = qi_round(values[i] * inv * (1.0f + qi_float(weight[d])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        uint d = tid * 4 + i;
        ushort result = normalized[d];
        if (d < 64) {
            uint partner = d < 32 ? d + 32 : d - 32;
            float rotated = qi_float(normalized[partner]) * (d < 32 ? -1.0f : 1.0f);
            float first = qi_float(qi_round(qi_float(normalized[d]) * qi_float(cosine[d])));
            float second = qi_float(qi_round(rotated * qi_float(sine[d])));
            result = qi_round(first + second);
        }
        output[d] = result;
    }
}

// One SIMD group computes eight key scores. The query buffer has eight rows;
// four zero rows pad the declared four query heads for an 8x8 matrix tile.
kernel void qsa_index_scores(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    device const uint* group_count [[buffer(3)]],
    device float* raw_scores [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    simdgroup_float8x8 accumulator(0.0f);
    for (uint k = 0; k < 128; k += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, queries + k, 128);
        simdgroup_load(b, keys + group * 8 * 128 + k, 128, ulong2(0), true);
        simdgroup_multiply_accumulate(accumulator, a, b, accumulator);
    }
    threadgroup float products[64];
    simdgroup_store(accumulator, products, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8 && group * 8 + tid < group_count[0]) {
        float sum = 0.0f;
        for (uint h = 0; h < 4; ++h) {
            raw_scores[h * group_count[0] + group * 8 + tid] = products[h * 8 + tid];
            sum += max(products[h * 8 + tid], 0.0f);
        }
        scores[group * 8 + tid] = sum / 11.313708498984761f;
    }
}
