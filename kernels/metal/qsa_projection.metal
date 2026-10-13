// Qwen QSA head normalization and partial RoPE adapted from mlx-kquant's
// kq_qwen4_qsa_project_rope (MIT). See THIRD-PARTY-NOTICES.
#include <metal_stdlib>
using namespace metal;

inline float qsa_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline ushort qsa_round(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Match MLX's FP32 power and precise trigonometric operators before the BF16
// cast. Host libm may choose different frequency bits at large positions.
inline void qsa_rope_factors_row(
    device const long* positions,
    device ushort* cosine,
    device ushort* sine,
    uint index) {
    if (index >= 32) return;
    float frequency = float(index) / 32.0f;
    float inverse = metal::pow(10000000.0f, -frequency);
    float angle = float(positions[index % 3]) * inverse;
    ushort c = qsa_round(metal::precise::cos(angle));
    ushort s = qsa_round(metal::precise::sin(angle));
    cosine[index] = cosine[index + 32] = c;
    sine[index] = sine[index + 32] = s;
}

kernel void qsa_rope_factors(
    device const long* positions [[buffer(0)]],
    device ushort* cosine [[buffer(1)]],
    device ushort* sine [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    qsa_rope_factors_row(positions, cosine, sine, index);
}

kernel void qsa_rope_factors_rows(
    device const long* positions [[buffer(0)]],
    device ushort* cosine [[buffer(1)]],
    device ushort* sine [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint row = index / 32u;
    qsa_rope_factors_row(positions + row * 3u, cosine + row * 64u,
                         sine + row * 64u, index % 32u);
}

inline void qsa_project_rope_head(
    device const ushort* index_projection,
    device const ushort* query_projection,
    device const ushort* key_projection,
    device const ushort* index_norm_weight,
    device const ushort* query_norm_weight,
    device const ushort* key_norm_weight,
    device const ushort* rope_cosine,
    device const ushort* rope_sine,
    device ushort* index_queries,
    device ushort* raw_index_key,
    device ushort* queries,
    device ushort* query_gate,
    device ushort* keys,
    device const float* epsilon,
    threadgroup ushort* normalized,
    uint logical_head,
    uint tid) {
    const uint index_heads = 4;
    const uint query_heads = 24;
    const uint index_width = 128;
    const uint head_width = 256;
    const uint rotary_width = 64;
    bool is_index = logical_head < index_heads;
    bool is_query = logical_head >= index_heads && logical_head < index_heads + query_heads;
    uint local_head = is_index ? logical_head
        : (is_query ? logical_head - index_heads : logical_head - index_heads - query_heads);
    uint width = is_index ? index_width : head_width;
    uint per_thread = width / 32;
    device const ushort* source = is_index
        ? index_projection + local_head * index_width
        : (is_query ? query_projection + local_head * 2 * head_width
                    : key_projection + local_head * head_width);
    device const ushort* weight = is_index ? index_norm_weight
        : (is_query ? query_norm_weight : key_norm_weight);
    device ushort* destination = is_index
        ? index_queries + local_head * index_width
        : (is_query ? queries + local_head * head_width : keys + local_head * head_width);

    float values[8];
    for (uint i = 0; i < 8; ++i) {
        if (i < per_thread) values[i] = qsa_bf16(source[tid * per_thread + i]);
    }
    float partial = 0.0f;
    if (width == head_width) {
        for (uint block = 0; block < 2; ++block) {
            for (uint i = 0; i < 4; ++i) {
                uint dimension = block * 128 + tid * 4 + i;
                float value = qsa_bf16(source[dimension]);
                partial += value * value;
            }
        }
    } else {
        for (uint i = 0; i < 4; ++i) partial += values[i] * values[i];
    }
    float sum_squares = simd_sum(partial);
    float inverse = precise::rsqrt(sum_squares / float(width) + epsilon[0]);

    for (uint i = 0; i < 8; ++i) {
        if (i < per_thread) {
            uint dimension = tid * per_thread + i;
            normalized[dimension] = qsa_round(
                values[i] * inverse * (1.0f + qsa_bf16(weight[dimension])));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 8; ++i) {
        if (i < per_thread) {
            uint dimension = tid * per_thread + i;
            ushort output = normalized[dimension];
            if (dimension < rotary_width) {
                uint partner = dimension < rotary_width / 2
                    ? dimension + rotary_width / 2 : dimension - rotary_width / 2;
                float rotated_half = dimension < rotary_width / 2
                    ? -qsa_bf16(normalized[partner]) : qsa_bf16(normalized[partner]);
                float first = qsa_bf16(qsa_round(
                    qsa_bf16(normalized[dimension]) * qsa_bf16(rope_cosine[dimension])));
                float second = qsa_bf16(qsa_round(
                    rotated_half * qsa_bf16(rope_sine[dimension])));
                output = qsa_round(first + second);
            }
            destination[dimension] = output;
        }
    }
    if (logical_head == 0) {
        for (uint i = 0; i < 4; ++i) {
            uint dimension = tid * 4 + i;
            raw_index_key[dimension] = index_projection[index_heads * index_width + dimension];
        }
    }
    if (is_query) {
        for (uint i = 0; i < 8; ++i) {
            uint dimension = tid * 8 + i;
            query_gate[local_head * head_width + dimension] = source[head_width + dimension];
        }
    }
}

kernel void qsa_project_rope_one(
    device const ushort* index_projection [[buffer(0)]],
    device const ushort* query_projection [[buffer(1)]],
    device const ushort* key_projection [[buffer(2)]],
    device const ushort* index_norm_weight [[buffer(3)]],
    device const ushort* query_norm_weight [[buffer(4)]],
    device const ushort* key_norm_weight [[buffer(5)]],
    device const ushort* rope_cosine [[buffer(6)]],
    device const ushort* rope_sine [[buffer(7)]],
    device ushort* index_queries [[buffer(8)]],
    device ushort* raw_index_key [[buffer(9)]],
    device ushort* queries [[buffer(10)]],
    device ushort* query_gate [[buffer(11)]],
    device ushort* keys [[buffer(12)]],
    device const float* epsilon [[buffer(13)]],
    uint logical_head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup ushort normalized[256];
    qsa_project_rope_head(index_projection, query_projection, key_projection,
        index_norm_weight, query_norm_weight, key_norm_weight, rope_cosine, rope_sine,
        index_queries, raw_index_key, queries, query_gate, keys, epsilon,
        normalized, logical_head, tid);
}

// One SIMD group per independent (row, head), as in DS4's prefill preparation.
// Flattening the grid preserves the existing 32-thread reduction and rotation.
kernel void qsa_project_rope_rows(
    device const ushort* index_projection [[buffer(0)]],
    device const ushort* query_projection [[buffer(1)]],
    device const ushort* key_projection [[buffer(2)]],
    device const ushort* index_norm_weight [[buffer(3)]],
    device const ushort* query_norm_weight [[buffer(4)]],
    device const ushort* key_norm_weight [[buffer(5)]],
    device const ushort* rope_cosine [[buffer(6)]],
    device const ushort* rope_sine [[buffer(7)]],
    device ushort* index_queries [[buffer(8)]],
    device ushort* raw_index_key [[buffer(9)]],
    device ushort* queries [[buffer(10)]],
    device ushort* query_gate [[buffer(11)]],
    device ushort* keys [[buffer(12)]],
    device const float* epsilon [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint row = group / 30u;
    threadgroup ushort normalized[256];
    qsa_project_rope_head(index_projection + row * 640u,
        query_projection + row * 12288u, key_projection + row * 512u,
        index_norm_weight, query_norm_weight, key_norm_weight,
        rope_cosine + row * 64u, rope_sine + row * 64u,
        index_queries + row * 512u, raw_index_key + row * 128u,
        queries + row * 6144u, query_gate + row * 6144u, keys + row * 512u,
        epsilon, normalized, group % 30u, tid);
}
