// BF16 sigmoid steps follow MLX's unary_ops.h Sigmoid<T> (MIT).
// See THIRD-PARTY-NOTICES.
#include <metal_stdlib>
using namespace metal;

inline float mlp_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline ushort mlp_round(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline float mlp_sigmoid(float value) {
    float exponential = mlp_bf16(mlp_round(precise::exp(metal::abs(value))));
    float denominator = mlp_bf16(mlp_round(1.0f + exponential));
    float reciprocal = mlp_bf16(mlp_round(1.0f / denominator));
    return value < 0.0f ? reciprocal : mlp_bf16(mlp_round(1.0f - reciprocal));
}

kernel void qwen4_sigmoid_bf16_words(
    device const ushort* input [[buffer(0)]],
    device ushort* output [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i < 65536) output[i] = mlp_round(mlp_sigmoid(mlp_bf16(input[i])));
}

// Each array is one BF16 vector. The activation's SiLU and multiply are
// separate BF16 operations in the shared branch.
kernel void qwen4_shared_activate(
    device const ushort* gate [[buffer(0)]],
    device const ushort* up [[buffer(1)]],
    device ushort* activated [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= 640) return;
    float g = mlp_bf16(gate[i]);
    float u = mlp_bf16(up[i]);
    float logistic = mlp_sigmoid(g);
    float silu = mlp_bf16(mlp_round(g * logistic));
    activated[i] = mlp_round(silu * u);
}

kernel void qwen4_shared_finish(
    device const ushort* routed [[buffer(0)]],
    device const ushort* down [[buffer(1)]],
    device const ushort* gate_logit [[buffer(2)]],
    device ushort* shared [[buffer(3)]],
    device ushort* final [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= 2560) return;
    float scalar = mlp_sigmoid(mlp_bf16(gate_logit[0]));
    ushort product = mlp_round(scalar * mlp_bf16(down[i]));
    shared[i] = product;
    final[i] = mlp_round(mlp_bf16(routed[i]) + mlp_bf16(product));
}
