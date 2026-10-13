// Qwen gated-delta decode stages.
// Copyright (c) 2026 The MoEspresso authors. See LICENSE-MIT and THIRD-PARTY-NOTICES.
// Adapted MLX helpers retain the Apple MIT notice in THIRD-PARTY-NOTICES.
#include <metal_stdlib>
using namespace metal;

inline float gdn_f(ushort bits) { return as_type<float>(uint(bits) << 16u); }
inline ushort gdn_bf(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
inline ushort gdn_sigmoid_bf(ushort input) {
    const float x = gdn_f(input);
    const ushort e = gdn_bf(metal::exp(metal::abs(x)));
    const ushort d = gdn_bf(1.0f + gdn_f(e));
    const ushort y = gdn_bf(1.0f / gdn_f(d));
    return x < 0.0f ? y : gdn_bf(1.0f - gdn_f(y));
}
inline float gdn_sigmoid_f(float x) {
    const float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
    return x < 0.0f ? y : 1.0f - y;
}
inline float gdn_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == 3.402823466e+38f) { return 3.402823466e+38f; }
    if (xp1 == 1.0f) { return x; }
    return x * (metal::log(xp1) / (xp1 - 1.0f));
}

kernel void gdn_prepare(
    const device ushort* qkv [[buffer(0)]],
    const device ushort* beta_logits [[buffer(1)]],
    const device ushort* decay_logits [[buffer(2)]],
    device ushort* conv_state [[buffer(3)]],
    const device ushort* conv_weight [[buffer(4)]],
    const device ushort* a_log [[buffer(5)]],
    const device ushort* dt_bias [[buffer(6)]],
    device ushort* query [[buffer(7)]],
    device ushort* key [[buffer(8)]],
    device ushort* value [[buffer(9)]],
    device ushort* beta [[buffer(10)]],
    device float* decay [[buffer(11)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    constexpr uint conv_channels = 10240u;
    constexpr uint key_heads = 16u;
    constexpr uint value_heads = 48u;
    constexpr uint head_width = 128u;
    constexpr uint key_width = key_heads * head_width;
    uint source_base;
    device ushort* destination;
    bool normalize;
    bool scale_query;
    if (head < key_heads) {
        source_base = head * head_width;
        destination = query;
        normalize = true;
        scale_query = true;
    } else if (head < 2u * key_heads) {
        source_base = key_width + (head - key_heads) * head_width;
        destination = key;
        normalize = true;
        scale_query = false;
    } else {
        source_base = 2u * key_width + (head - 2u * key_heads) * head_width;
        destination = value;
        normalize = false;
        scale_query = false;
    }

    ushort activated[4];
    const uint first = tid * 4u;
    for (uint i = 0u; i < 4u; ++i) {
        const uint channel = source_base + first + i;
        float accumulator = 0.0f;
        for (uint tap = 0u; tap < 4u; ++tap) {
            const ushort source = tap < 3u ? conv_state[tap * conv_channels + channel] : qkv[channel];
            accumulator += gdn_f(source) * gdn_f(conv_weight[channel * 4u + tap]);
        }
        const ushort convolved = gdn_bf(accumulator);
        activated[i] = gdn_bf(gdn_f(convolved) * gdn_f(gdn_sigmoid_bf(convolved)));
        conv_state[channel] = conv_state[conv_channels + channel];
        conv_state[conv_channels + channel] = conv_state[2u * conv_channels + channel];
        conv_state[2u * conv_channels + channel] = qkv[channel];
    }

    ushort inv_norm = gdn_bf(1.0f);
    if (normalize) {
        ushort partial = gdn_bf(0.0f);
        for (uint i = 0u; i < 4u; ++i) {
            const ushort square = gdn_bf(gdn_f(activated[i]) * gdn_f(activated[i]));
            partial = gdn_bf(gdn_f(partial) + gdn_f(square));
        }
        const ushort sum_squares = gdn_bf(simd_sum(gdn_f(partial)));
        const ushort stabilized = gdn_bf(
            gdn_f(sum_squares) + gdn_f(gdn_bf(1e-6f)));
        inv_norm = gdn_bf(metal::precise::rsqrt(gdn_f(stabilized)));
    }

    const uint destination_base = normalize
        ? (head % key_heads) * head_width
        : (head - 2u * key_heads) * head_width;
    const ushort query_scale = gdn_bf(0.08838834764831845f);
    for (uint i = 0u; i < 4u; ++i) {
        ushort prepared = normalize
            ? gdn_bf(gdn_f(activated[i]) * gdn_f(inv_norm)) : activated[i];
        if (scale_query) { prepared = gdn_bf(gdn_f(prepared) * gdn_f(query_scale)); }
        destination[destination_base + first + i] = prepared;
    }

    if (head < value_heads && tid == 0u) {
        beta[head] = gdn_sigmoid_bf(beta_logits[head]);
        const float x = gdn_f(decay_logits[head]) + gdn_f(dt_bias[head]);
        const float maximum = metal::max(x, 0.0f);
        const float minimum = metal::min(x, 0.0f);
        const float softplus = maximum + gdn_log1p(metal::exp(minimum - maximum));
        const float rate = metal::precise::exp(gdn_f(a_log[head])) * softplus;
        decay[head] = metal::precise::exp(-rate);
    }
}

// Flatten the reference (32,128,48) grid into 1536 groups of 128 threads.
// Four SIMD groups in each threadgroup own distinct value rows. Every FP32
// state element has exactly one writer, so state_in and state_out may alias.
kernel void gdn_recurrence(
    const device ushort* query [[buffer(0)]],
    const device ushort* key [[buffer(1)]],
    const device ushort* value [[buffer(2)]],
    const device float* decay [[buffer(3)]],
    const device ushort* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device ushort* output [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint head_width = 128u;
    const uint value_head = group / 32u;
    const uint value_dimension = (group % 32u) * 4u + (tid >> 5u);
    const uint key_head = value_head / 3u;
    const uint key_base = key_head * head_width;
    const uint value_base = value_head * head_width;
    const uint state_base = (value_head * head_width + value_dimension) * head_width;
    float row[4];
    for (uint i = 0u; i < 4u; ++i) { row[i] = state[state_base + 4u * lane + i]; }
    float memory = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint dimension = 4u * lane + i;
        row[i] *= decay[value_head];
        memory += row[i] * gdn_f(key[key_base + dimension]);
    }
    memory = simd_sum(memory);
    const float delta = (gdn_f(value[value_base + value_dimension]) - memory)
        * gdn_f(beta[value_head]);
    float projected = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint dimension = 4u * lane + i;
        row[i] += gdn_f(key[key_base + dimension]) * delta;
        projected += row[i] * gdn_f(query[key_base + dimension]);
    }
    projected = simd_sum(projected);
    if (lane == 0u) { output[value_base + value_dimension] = gdn_bf(projected); }
    for (uint i = 0u; i < 4u; ++i) { state[state_base + 4u * lane + i] = row[i]; }
}

kernel void gdn_norm_gate(
    const device ushort* recurrence [[buffer(0)]],
    const device ushort* gate [[buffer(1)]],
    const device ushort* norm_weight [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    const constant float* eps [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint head_base = head * 128u;
    const uint first = tid * 4u;
    float source[4];
    float partial = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        source[i] = gdn_f(recurrence[head_base + first + i]);
        partial += source[i] * source[i];
    }
    const float sum_squares = simd_sum(partial);
    const float inv_rms = metal::precise::rsqrt(sum_squares / 128.0f + *eps);
    for (uint i = 0u; i < 4u; ++i) {
        const uint dimension = first + i;
        const uint index = head_base + dimension;
        const ushort normalized = gdn_bf(source[i] * inv_rms);
        const ushort weighted = gdn_bf(gdn_f(norm_weight[dimension]) * gdn_f(normalized));
        output[index] = gdn_bf(gdn_f(weighted) * gdn_sigmoid_f(gdn_f(gate[index])));
    }
}
