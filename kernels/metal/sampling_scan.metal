#include <metal_stdlib>

using namespace metal;

// Scan topology adapted from Apple MLX scan.h (MIT); see THIRD-PARTY-NOTICES.
// Inclusive contiguous BF16 sum with the four-value/lane scan geometry used
// by the package's reference sampler. The arithmetic stays in BF16.
kernel void sampling_bf16_cumsum(
    const device bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint reads = 4;
    constexpr uint simd_width = 32;
    threadgroup bfloat simdgroup_sums[32];
    const uint groups = threads / simd_width;
    bfloat prefix = bfloat(0.0f);

    for (uint block = 0; block < (count + reads * threads - 1) / (reads * threads); ++block) {
        const uint offset = block * reads * threads + thread_id * reads;
        bfloat values[reads];
        for (uint i = 0; i < reads; ++i) {
            values[i] = offset + i < count ? input[offset + i] : bfloat(0.0f);
        }
        for (uint i = 1; i < reads; ++i) {
            values[i] = values[i] + values[i - 1];
        }

        const bfloat previous_thread = bfloat(simd_prefix_exclusive_sum(float(values[reads - 1])));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == simd_width - 1) {
            simdgroup_sums[simdgroup] = previous_thread + values[reads - 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == 0 && lane >= groups) {
            simdgroup_sums[lane] = bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == 0) {
            simdgroup_sums[lane] = bfloat(simd_prefix_exclusive_sum(float(simdgroup_sums[lane])));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < reads; ++i) {
            values[i] = values[i] + prefix;
            values[i] = values[i] + simdgroup_sums[simdgroup];
            values[i] = values[i] + previous_thread;
            if (offset + i < count) {
                output[offset + i] = values[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == groups - 1 && lane == simd_width - 1) {
            simdgroup_sums[0] = values[reads - 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = simdgroup_sums[0];
    }
}
