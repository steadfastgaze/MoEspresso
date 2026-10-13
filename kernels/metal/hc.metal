// Qwen gated-residual decode with row-parallel low-rank and optional injection projections.
// Copyright (c) 2026 The MoEspresso authors. See LICENSE-MIT and THIRD-PARTY-NOTICES.
#include <metal_stdlib>
using namespace metal;

inline float hc_float(ushort bits) { return as_type<float>(uint(bits) << 16u); }
inline ushort hc_bf16(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

// mlx-kquant evaluates the stable sigmoid in BF16 at every operator boundary.
inline ushort hc_sigmoid(ushort input) {
    const float x = hc_float(input);
    const ushort e = hc_bf16(metal::exp(metal::abs(x)));
    const ushort denominator = hc_bf16(1.0f + hc_float(e));
    const ushort y = hc_bf16(1.0f / hc_float(denominator));
    return x < 0.0f ? y : hc_bf16(1.0f - hc_float(y));
}

kernel void hc_norm(
    const device ushort* residual [[buffer(0)]],
    const device ushort* norm_weight [[buffer(1)]],
    const device ushort* pending_output [[buffer(2)]],
    const device ushort* pending_injection [[buffer(3)]],
    device ushort* normalized [[buffer(4)]],
    device ushort* updated_residual [[buffer(5)]],
    const constant float* eps [[buffer(6)]],
    const constant int* has_pending [[buffer(7)]],
    uint branch [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint width = 2560u;
    constexpr uint simdgroups = 20u;
    threadgroup float partials[simdgroups];
    threadgroup float inv_rms;
    float sum_squares = 0.0f;
    ushort kept[4];
    const uint base = branch * width;
    const uint first = tid * 4u;
#pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        const uint d = first + i;
        ushort value = residual[base + d];
        if (*has_pending != 0) {
            const ushort update = hc_bf16(
                hc_float(pending_output[d]) * hc_float(pending_injection[branch]));
            value = hc_bf16(hc_float(value) + hc_float(update));
        }
        updated_residual[base + d] = value;
        kept[i] = value;
        const float source = hc_float(value);
        volatile float square = source * source;
        sum_squares += square;
    }
    sum_squares = simd_sum(sum_squares);
    if (lane == 0u) partials[sg] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const float cross_simd = simd_sum(lane < simdgroups ? partials[lane] : 0.0f);
        if (lane == 0u) {
            volatile float mean_square = cross_simd * (1.0f / float(width));
            volatile float stabilized_mean = mean_square + *eps;
            inv_rms = metal::precise::rsqrt(stabilized_mean);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        const uint index = base + first + i;
        const float unit = hc_float(kept[i]) * inv_rms;
        const float scale = 1.0f + hc_float(norm_weight[index]);
        normalized[index] = hc_bf16(unit * scale);
    }
}

inline float hc_f16(const device uchar* address) {
    const ushort bits = ushort(address[0]) | (ushort(address[1]) << 8);
    return float(as_type<half>(bits));
}

// Each SIMD group computes one low-rank or injection row. Groups 160 and 161
// cover injection; projection rounding remains before scaling and activation.
kernel void hc_front(
    const device ushort* normalized [[buffer(0)]],
    const device uchar* down_weight [[buffer(1)]],
    const device uchar* injection_weight [[buffer(2)]],
    device ushort* lowrank [[buffer(3)]],
    device ushort* injection [[buffer(4)]],
    const constant int* has_injection [[buffer(5)]],
    device ushort* projected_down [[buffer(6)]],
    device ushort* projected_injection [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const bool inject = group >= 160;
    if (inject && *has_injection == 0) return;
    const uint out_row = (inject ? group - 160 : group) * 2 + sg;
    const device uchar* weight = inject ? injection_weight : down_weight;
    const uint tid_lane = lane / 2, ix = lane % 2;
    const uint ip = tid_lane / 8, il = tid_lane % 8;
    const uint l0 = 4 * il, scale_offset = 8 * ip + l0 / 16;
    float result[1] = {0.0f};
    for (uint ib = ix; ib < 40; ib += 2) {
        const uint x_base = ib * 256 + 128 * ip + l0;
        float values[16];
#pragma unroll
        for (uint l = 0; l < 4; ++l) {
            values[4*l] = hc_float(normalized[x_base + l]);
            values[4*l+1] = hc_float(normalized[x_base + l + 32]);
            values[4*l+2] = hc_float(normalized[x_base + l + 64]);
            values[4*l+3] = hc_float(normalized[x_base + l + 96]);
        }
#pragma unroll
        for (uint row = 0; row < 1; ++row) {
            const device uchar* block = weight + (out_row + row) * (40 * 210) + ib * 210;
            const device uchar* q1 = block + 64 * ip + l0;
            const device uchar* q2 = q1 + 32;
            const device uchar* qh = block + 128 + 32 * ip + l0;
            const device char* scales = reinterpret_cast<const device char*>(block + 192 + scale_offset);
            float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (uint l = 0; l < 4; ++l) {
                const uchar a = q1[l], b = q2[l], h = qh[l];
                const int v0 = int((a & 15) | ((h & 3) << 4)) - 32;
                const int v1 = int((b & 15) | ((h & 12) << 2)) - 32;
                const int v2 = int((a >> 4) | (h & 48)) - 32;
                const int v3 = int((b >> 4) | ((h & 192) >> 2)) - 32;
                sums[0] += values[4*l] * float(v0);
                sums[1] += values[4*l+1] * float(v1);
                sums[2] += values[4*l+2] * float(v2);
                sums[3] += values[4*l+3] * float(v3);
            }
            const float d = hc_f16(block + 208);
            result[row] += d * (sums[0] * float(scales[0]) + sums[1] * float(scales[2])
                                + sums[2] * float(scales[4]) + sums[3] * float(scales[6]));
        }
    }
#pragma unroll
    for (uint row = 0; row < 1; ++row) {
        const float reduced = simd_sum(result[row]);
        if (lane == 0) {
            const ushort projected = hc_bf16(reduced);
            const ushort scaled = hc_bf16(hc_float(projected) * 0.25f);
            const ushort gate = hc_sigmoid(scaled);
            if (inject) {
                projected_injection[out_row + row] = projected;
                injection[out_row + row] = hc_bf16(2.0f * hc_float(gate));
            } else {
                projected_down[out_row + row] = projected;
                lowrank[out_row + row] = hc_bf16(hc_float(scaled) * hc_float(gate));
            }
        }
    }
}

// Each SIMD group computes all four branches for one output channel, retaining
// the Q8 lane fold and each BF16 rounding before the branch-ordered reduction.
kernel void hc_epilogue(
    const device ushort* lowrank [[buffer(0)]],
    const device uchar* up_weight [[buffer(1)]],
    const device ushort* normalized [[buffer(2)]],
    device ushort* mixed [[buffer(3)]],
    device ushort* projected_up [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint id = group * 2 + sg;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint lane_offset = lane * 8;
    for (uint k = 0; k < 320; k += 256) {
        const int remaining = 320 - int(k) - int(lane_offset);
        const uint active = uint(clamp(remaining, 0, 8));
        float values[8];
#pragma unroll
        for (uint i = 0; i < 8; ++i)
            values[i] = i < active ? hc_float(lowrank[k + lane_offset + i]) : 0.0f;
        if (active == 0) continue;
        const uint k_global = k + lane_offset;
        const uint block_id = k_global / 32, within = k_global % 32;
#pragma unroll
        for (uint stream = 0; stream < 4; ++stream) {
            const uint row = stream * 2560 + id;
            const device uchar* block = up_weight + row * (10 * 34) + block_id * 34;
            const float scale = hc_f16(block);
            const device char* quant = reinterpret_cast<const device char*>(block + 2 + within);
            float partial = 0.0f;
#pragma unroll
            for (uint i = 0; i < 8; ++i)
                if (i < active) partial += values[i] * float(quant[i]);
            result[stream] += scale * partial;
        }
    }
#pragma unroll
    for (uint stream = 0; stream < 4; ++stream) result[stream] = simd_sum(result[stream]);
    if (lane == 0) {
        ushort total = 0u;
        for (uint stream = 0; stream < 4; ++stream) {
            const uint index = stream * 2560 + id;
            const ushort projected = hc_bf16(result[stream]);
            projected_up[index] = projected;
            const ushort gate = hc_sigmoid(projected);
            const ushort product = hc_bf16(hc_float(gate) * hc_float(normalized[index]));
            total = hc_bf16(hc_float(total) + hc_float(product));
        }
        mixed[id] = hc_bf16(hc_float(total) * 0.25f);
    }
}
