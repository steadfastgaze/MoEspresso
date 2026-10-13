// Qwen PLE BF16 row reduction adapted from Apple MLX reduce_row.h and
// bf16_math.h (Copyright 2023-2024 Apple Inc., MIT license). Each product and
// per-thread addition rounds to BF16; SIMD sums use FP32 then round to BF16.
#include <metal_stdlib>
using namespace metal;

inline float ple_float(ushort bits) { return as_type<float>(uint(bits) << 16u); }
inline ushort ple_bf16(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

kernel void ple_gate_dot(
    const device bfloat* key [[buffer(0)]],
    const device bfloat* query [[buffer(1)]],
    device bfloat* dot [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint branch [[threadgroup_position_in_grid]]
) {
    constexpr uint width = 2560u;
    constexpr uint simdgroups = 20u;
    threadgroup bfloat partials[simdgroups];
    bfloat sum = bfloat(0.0f);
    const uint first = branch * width + tid * 4u;
    for (uint i = 0u; i < 4u; ++i) {
        const bfloat product = key[first + i] * query[first + i];
        sum = product + sum;
    }
    const bfloat simd_total = bfloat(simd_sum(float(sum)));
    if (lane == 0u) partials[sg] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const bfloat total = bfloat(simd_sum(lane < simdgroups ? float(partials[lane]) : 0.0f));
        if (lane == 0u) dot[branch] = total;
    }
}
