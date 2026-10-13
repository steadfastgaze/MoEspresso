#include <metal_stdlib>

using namespace metal;

// Reduction topology adapted from Apple MLX logsumexp.h (MIT);
// see THIRD-PARTY-NOTICES.
kernel void sampling_bf16_logsumexp_block(
    const device bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint reads = 4;
    threadgroup float local_max[32];
    threadgroup float local_normalizer[32];
    float values[reads];
    for (uint i = 0; i < reads; ++i) {
        const uint index = thread_id * reads + i;
        values[i] = index < count ? float(input[index]) : -INFINITY;
    }
    if (simdgroup == 0) {
        local_max[lane] = -INFINITY;
        local_normalizer[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maximum = -FLT_MAX;
    for (uint i = 0; i < reads; ++i) {
        maximum = maximum < values[i] ? values[i] : maximum;
    }
    maximum = simd_max(maximum);
    if (lane == 0) {
        local_max[simdgroup] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        maximum = simd_max(local_max[lane]);
        if (lane == 0) {
            local_max[0] = maximum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = local_max[0];

    float normalizer = 0.0f;
    for (uint i = 0; i < reads; ++i) {
        normalizer += fast::exp(values[i] - maximum);
    }
    normalizer = simd_sum(normalizer);
    if (lane == 0) {
        local_normalizer[simdgroup] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        normalizer = simd_sum(local_normalizer[lane]);
        if (lane == 0) {
            output[0] = isinf(maximum) ? bfloat(maximum) : bfloat(log(normalizer) + maximum);
        }
    }
}

kernel void sampling_bf16_logsumexp_looped(
    const device bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint reads = 4;
    threadgroup float local_max[32];
    threadgroup float local_normalizer[32];

    if (simdgroup == 0) {
        local_max[lane] = -INFINITY;
        local_normalizer[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maximum = -FLT_MAX;
    float normalizer = 0.0f;
    for (uint block = 0; block < (count + reads * threads - 1) / (reads * threads); ++block) {
        const uint offset = block * reads * threads + thread_id * reads;
        float values[reads];
        for (uint i = 0; i < reads; ++i) {
            values[i] = offset + i < count ? float(input[offset + i]) : -INFINITY;
        }
        const float previous_maximum = maximum;
        for (uint i = 0; i < reads; ++i) {
            maximum = maximum < values[i] ? values[i] : maximum;
        }
        normalizer *= fast::exp(previous_maximum - maximum);
        for (uint i = 0; i < reads; ++i) {
            normalizer += fast::exp(values[i] - maximum);
        }
    }
    float previous_maximum = maximum;
    maximum = simd_max(maximum);
    normalizer *= fast::exp(previous_maximum - maximum);
    normalizer = simd_sum(normalizer);

    previous_maximum = maximum;
    if (lane == 0) {
        local_max[simdgroup] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = simd_max(local_max[lane]);
    normalizer *= fast::exp(previous_maximum - maximum);
    if (lane == 0) {
        local_normalizer[simdgroup] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = simd_sum(local_normalizer[lane]);

    if (thread_id == 0) {
        output[0] = isinf(maximum) ? bfloat(maximum) : bfloat(log(normalizer) + maximum);
    }
}
