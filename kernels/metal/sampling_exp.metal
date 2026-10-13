#include <metal_stdlib>

using namespace metal;

// BF16 Exp operation adapted from Apple MLX unary_ops.h (MIT);
// see THIRD-PARTY-NOTICES.
kernel void sampling_bf16_exp(
    const device bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[index] = bfloat(metal::precise::exp(input[index]));
    }
}
