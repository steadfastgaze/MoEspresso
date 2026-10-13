// Fixed Qwen routed block stages. The IQ_K projections are in iqk_gemv.metal.
#include <metal_stdlib>
using namespace metal;

kernel void swiglu_10(
    const device half* gate [[buffer(0)]],
    const device half* up [[buffer(1)]],
    const device half* sigmoid_table [[buffer(2)]],
    device half* activation [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    const uint route = id / 768u;
    const uint col = id - route * 768u;
    if (col >= 640u) {
        activation[id] = half(0.0f);
        return;
    }
    const uint src = route * 640u + col;
    const half g = gate[src];
    const half sigmoid_gate = sigmoid_table[as_type<ushort>(g)];
    const half silu_gate = g * sigmoid_gate;
    activation[id] = silu_gate * up[src];
}

inline ushort round_bf16_bits(float value) {
    const uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

inline float bf16_bits_to_float(ushort bits) {
    return as_type<float>(uint(bits) << 16u);
}

// Full residency needs no host demand publication. Preserve the routed
// reduction's logical-source order while keeping all selector records on GPU.
kernel void prepare_resident_routes(
    const device uint* ids [[buffer(0)]],
    const device ushort* scores [[buffer(1)]],
    const device uint* changed [[buffer(2)]],
    device uint* slots [[buffer(3)]],
    device ushort* ordered_scores [[buffer(4)]],
    device uint* order [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    uint permutation[10];
    bool valid = changed[0] == 0u;
    for (uint i = 0; i < 10u; ++i) {
        permutation[i] = i;
        valid = valid && ids[i] < 512u && isfinite(bf16_bits_to_float(scores[i]));
        for (uint j = 0; j < i; ++j) valid = valid && ids[i] != ids[j];
    }
    for (uint i = 1; i < 10u; ++i) {
        const uint value = permutation[i];
        uint j = i;
        while (j > 0u && ids[permutation[j - 1u]] > ids[value]) {
            permutation[j] = permutation[j - 1u];
            --j;
        }
        permutation[j] = value;
    }
    if (!valid) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    for (uint i = 0; i < 10u; ++i) {
        order[i] = permutation[i];
        slots[i] = valid ? ids[permutation[i]] : 0u;
        ordered_scores[i] = valid ? scores[permutation[i]] : 0u;
    }
}

kernel void reduce_10_bf16(
    const device half* down [[buffer(0)]],
    const device ushort* scores [[buffer(1)]],
    const device uint* source_order [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    uint col [[thread_position_in_grid]]
) {
    ushort total = 0u;
    for (uint position = 0u; position < 10u; ++position) {
        const uint route = source_order[position];
        const ushort value = round_bf16_bits(float(down[route * 2560u + col]));
        const ushort product = round_bf16_bits(
            bf16_bits_to_float(value) * bf16_bits_to_float(scores[route]));
        total = round_bf16_bits(bf16_bits_to_float(total) + bf16_bits_to_float(product));
    }
    output[col] = total;
}
