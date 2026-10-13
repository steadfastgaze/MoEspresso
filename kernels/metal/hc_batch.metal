// Batched Qwen residual reads, adapted from MoEspresso and MLX reductions.
// See THIRD-PARTY-NOTICES for retained source licenses.
#include <metal_stdlib>
using namespace metal;

inline float hcb_float(ushort value) { return as_type<float>(uint(value) << 16u); }
inline ushort hcb_word(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
inline ushort hcb_sigmoid(ushort value) {
    float x = hcb_float(value);
    ushort e = hcb_word(metal::precise::exp(metal::abs(x)));
    ushort denominator = hcb_word(1.0f + hcb_float(e));
    ushort y = hcb_word(1.0f / hcb_float(denominator));
    return x < 0.0f ? y : hcb_word(1.0f - hcb_float(y));
}

kernel void hc_batch_norm(
    const device ushort* residual [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    const device ushort* pending [[buffer(2)]],
    const device ushort* injection [[buffer(3)]],
    device ushort* normalized [[buffer(4)]],
    device ushort* updated [[buffer(5)]],
    const constant float* eps [[buffer(6)]],
    const constant uint* flags [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partials[80];
    threadgroup float inverse[4];
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint branch = 0; branch < 4; ++branch) {
        for (uint j = 0; j < 4; ++j) {
            uint d = tid * 4 + j;
            uint index = row * 10240 + branch * 2560 + d;
            ushort value = residual[index];
            if (flags[0]) {
                ushort delta = hcb_word(hcb_float(pending[row * 2560 + d])
                    * hcb_float(injection[row * 4 + branch]));
                value = hcb_word(hcb_float(value) + hcb_float(delta));
            }
            updated[index] = value;
            float x = hcb_float(value);
            volatile float square = x * x;
            sums[branch] += square;
        }
        sums[branch] = simd_sum(sums[branch]);
        if (lane == 0) partials[sg * 4 + branch] = sums[branch];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (uint branch = 0; branch < 4; ++branch) {
            float sum = simd_sum(lane < 20 ? partials[lane * 4 + branch] : 0.0f);
            if (lane == 0) {
                volatile float mean = sum * (1.0f / 2560.0f);
                volatile float stabilized = mean + eps[0];
                inverse[branch] = metal::precise::rsqrt(stabilized);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint branch = 0; branch < 4; ++branch) {
        for (uint d = tid; d < 2560; d += 640) {
            uint local = branch * 2560 + d;
            uint index = row * 10240 + local;
            float unit = hcb_float(updated[index]) * inverse[branch];
            normalized[index] = hcb_word(unit * (1.0f + hcb_float(weight[local])));
        }
    }
}

kernel void hc_batch_front(
    const device ushort* down [[buffer(0)]],
    const device ushort* projected_injection [[buffer(1)]],
    device ushort* lowrank [[buffer(2)]],
    device ushort* injection [[buffer(3)]],
    const constant uint* flags [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    ushort scaled = hcb_word(hcb_float(down[id]) * 0.25f);
    lowrank[id] = hcb_word(hcb_float(scaled) * hcb_float(hcb_sigmoid(scaled)));
    uint row = id / 320;
    uint column = id % 320;
    if (flags[1] && column < 4) {
        uint index = row * 4 + column;
        ushort scaled_injection = hcb_word(hcb_float(projected_injection[index]) * 0.25f);
        injection[index] = hcb_word(2.0f * hcb_float(hcb_sigmoid(scaled_injection)));
    }
}

kernel void hc_batch_mix(
    const device ushort* projected [[buffer(0)]],
    const device ushort* normalized [[buffer(1)]],
    device ushort* mixed [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint row = id / 2560;
    uint column = id % 2560;
    ushort sum = 0;
    for (uint branch = 0; branch < 4; ++branch) {
        uint index = row * 10240 + branch * 2560 + column;
        ushort product = hcb_word(hcb_float(hcb_sigmoid(projected[index]))
            * hcb_float(normalized[index]));
        sum = hcb_word(hcb_float(sum) + hcb_float(product));
    }
    mixed[id] = hcb_word(hcb_float(sum) * 0.25f);
}
