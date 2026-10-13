// Batched PLE gate reduction and dilated depthwise convolution. The grouped
// normalization helper is supplied by hc_batch.metal in the same library.
// BF16 reduction order follows the retained MLX gate-dot implementation;
// see THIRD-PARTY-NOTICES for the source license.

kernel void ple_batch_gate_dot(
    const device bfloat* key [[buffer(0)]],
    const device bfloat* query [[buffer(1)]],
    device bfloat* dot [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup bfloat partials[20];
    bfloat sum = bfloat(0.0f);
    uint first = group * 2560u + tid * 4u;
    for (uint i = 0; i < 4; ++i) {
        bfloat product = key[first + i] * query[first + i];
        sum = product + sum;
    }
    bfloat simd_total = bfloat(simd_sum(float(sum)));
    if (lane == 0) partials[sg] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        bfloat total = bfloat(simd_sum(lane < 20 ? float(partials[lane]) : 0.0f));
        if (lane == 0) dot[group] = total;
    }
}

kernel void ple_batch_gate_scalar(
    const device ushort* dot [[buffer(0)]],
    device ushort* gate [[buffer(1)]],
    device ushort* gain [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    ushort divisor = hcb_word(sqrt(2560.0f));
    ushort scaled = hcb_word(hcb_float(dot[index]) / hcb_float(divisor));
    float x = hcb_float(scaled);
    float magnitude = hcb_float(hcb_word(sqrt(max(abs(x), 1e-6f))));
    ushort signed_gate = hcb_word(x == 0.0f ? 0.0f : (x < 0.0f ? -magnitude : magnitude));
    gate[index] = signed_gate;
    gain[index] = hcb_sigmoid(signed_gate);
}

kernel void ple_batch_flat(
    const device ushort* gain [[buffer(0)]],
    const device ushort* value [[buffer(1)]],
    device ushort* flat [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint row = index / 10240u;
    uint branch = (index % 10240u) / 2560u;
    uint column = index % 2560u;
    flat[index] = hcb_word(hcb_float(gain[row * 4u + branch])
                           * hcb_float(value[row * 2560u + column]));
}

kernel void ple_batch_conv(
    const device ushort* previous [[buffer(0)]],
    const device ushort* normalized [[buffer(1)]],
    const device ushort* weight [[buffer(2)]],
    const device ushort* flat [[buffer(3)]],
    device ushort* convolved [[buffer(4)]],
    device ushort* output [[buffer(5)]],
    device ushort* next [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    uint token = index / 10240u;
    uint channel = index % 10240u;
    if (token < 9u) {
        uint position = rows + token;
        next[index] = position < 9u
            ? previous[position * 10240u + channel]
            : normalized[(position - 9u) * 10240u + channel];
    }
    if (token >= rows) return;
    float accumulator = 0.0f;
    for (uint tap = 0; tap < 4u; ++tap) {
        uint position = token + tap * 3u;
        ushort source = position < 9u
            ? previous[position * 10240u + channel]
            : normalized[(position - 9u) * 10240u + channel];
        volatile float product = hcb_float(source) * hcb_float(weight[channel * 4u + tap]);
        accumulator += product;
    }
    ushort preactivation = hcb_word(accumulator);
    ushort activated = hcb_word(hcb_float(preactivation)
                               * hcb_float(hcb_sigmoid(preactivation)));
    convolved[index] = activated;
    output[index] = hcb_word(hcb_float(flat[index]) + hcb_float(activated));
}

// One decode token advances the canonical device history and adds the PLE
// contribution to the expanded residual for the next layer.
kernel void ple_decode_conv_residual(
    const device ushort* previous [[buffer(0)]],
    const device ushort* normalized [[buffer(1)]],
    const device ushort* weight [[buffer(2)]],
    const device ushort* flat [[buffer(3)]],
    const device ushort* residual [[buffer(4)]],
    device ushort* output [[buffer(5)]],
    device ushort* next [[buffer(6)]],
    uint channel [[thread_position_in_grid]]
) {
    if (channel >= 10240u) return;
    float accumulator = 0.0f;
    for (uint tap = 0; tap < 4u; ++tap) {
        ushort source = tap < 3u
            ? previous[tap * 3u * 10240u + channel]
            : normalized[channel];
        volatile float product = hcb_float(source) * hcb_float(weight[channel * 4u + tap]);
        accumulator += product;
    }
    ushort preactivation = hcb_word(accumulator);
    ushort activated = hcb_word(hcb_float(preactivation)
                               * hcb_float(hcb_sigmoid(preactivation)));
    ushort contribution = hcb_word(hcb_float(flat[channel]) + hcb_float(activated));
    output[channel] = hcb_word(hcb_float(residual[channel]) + hcb_float(contribution));
    for (uint row = 0; row < 8u; ++row) {
        next[row * 10240u + channel] = previous[(row + 1u) * 10240u + channel];
    }
    next[8u * 10240u + channel] = normalized[channel];
}

// One group per token fuses the four grouped norms, gate dots, value gate,
// and convolution-input norms. Reduction accumulators remain in FP32.
kernel void ple_prefill_fused_gate(
    const device ushort* residual [[buffer(0)]],
    const device ushort* key [[buffer(1)]],
    const device ushort* value [[buffer(2)]],
    const device ushort* norm_key [[buffer(3)]],
    const device ushort* norm_query [[buffer(4)]],
    const device ushort* norm_conv [[buffer(5)]],
    device ushort* gated [[buffer(6)]],
    device ushort* normed [[buffer(7)]],
    const constant float* epsilon [[buffer(8)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[3][8];
    threadgroup float inverse_key;
    threadgroup float inverse_query;
    threadgroup float gain;
    threadgroup float inverse_gated;
    for (uint branch = 0; branch < 4u; ++branch) {
        uint base = token * 10240u + branch * 2560u;
        uint weight_base = branch * 2560u;
        float key_sum = 0.0f;
        float query_sum = 0.0f;
        for (uint d = tid; d < 2560u; d += 256u) {
            float k = hcb_float(key[base + d]);
            float q = hcb_float(residual[base + d]);
            key_sum += k * k;
            query_sum += q * q;
        }
        key_sum = simd_sum(key_sum);
        query_sum = simd_sum(query_sum);
        if (lane == 0u) {
            partial[0][sg] = key_sum;
            partial[1][sg] = query_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float total_key = 0.0f;
            float total_query = 0.0f;
            for (uint i = 0u; i < 8u; ++i) {
                total_key += partial[0][i];
                total_query += partial[1][i];
            }
            inverse_key = rsqrt(total_key * (1.0f / 2560.0f) + epsilon[0]);
            inverse_query = rsqrt(total_query * (1.0f / 2560.0f) + epsilon[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dot = 0.0f;
        for (uint d = tid; d < 2560u; d += 256u) {
            float k = hcb_float(key[base + d]) * inverse_key
                * (1.0f + hcb_float(norm_key[weight_base + d]));
            float q = hcb_float(residual[base + d]) * inverse_query
                * (1.0f + hcb_float(norm_query[weight_base + d]));
            dot += k * q;
        }
        dot = simd_sum(dot);
        if (lane == 0u) partial[2][sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float total_dot = 0.0f;
            for (uint i = 0u; i < 8u; ++i) total_dot += partial[2][i];
            float scaled = total_dot * rsqrt(2560.0f);
            float magnitude = sqrt(max(abs(scaled), 1e-6f));
            float signed_gate = scaled < 0.0f ? -magnitude : (scaled > 0.0f ? magnitude : 0.0f);
            float exp_term = metal::precise::exp(-abs(signed_gate));
            gain = signed_gate >= 0.0f
                ? 1.0f / (1.0f + exp_term)
                : exp_term / (1.0f + exp_term);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gated_sum = 0.0f;
        for (uint d = tid; d < 2560u; d += 256u) {
            float v = gain * hcb_float(value[token * 2560u + d]);
            gated[base + d] = hcb_word(v);
            gated_sum += v * v;
        }
        gated_sum = simd_sum(gated_sum);
        if (lane == 0u) partial[0][sg] = gated_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float total_gated = 0.0f;
            for (uint i = 0u; i < 8u; ++i) total_gated += partial[0][i];
            inverse_gated = rsqrt(total_gated * (1.0f / 2560.0f) + epsilon[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < 2560u; d += 256u) {
            float v = gain * hcb_float(value[token * 2560u + d]);
            normed[base + d] = hcb_word(v * inverse_gated
                * (1.0f + hcb_float(norm_conv[weight_base + d])));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Each channel owns its complete nine-row history, so the canonical buffer
// can be advanced in place without a token-sized host transfer or row race.
kernel void ple_prefill_fused_conv_residual(
    const device ushort* residual [[buffer(0)]],
    const device ushort* gated [[buffer(1)]],
    const device ushort* normed [[buffer(2)]],
    device ushort* history [[buffer(3)]],
    const device ushort* weight [[buffer(4)]],
    device ushort* output [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint channel [[thread_position_in_grid]]
) {
    if (channel >= 10240u) return;
    float local_history[9];
    float taps[4];
    for (uint row = 0u; row < 9u; ++row) {
        local_history[row] = hcb_float(history[row * 10240u + channel]);
    }
    for (uint tap = 0u; tap < 4u; ++tap) {
        taps[tap] = hcb_float(weight[channel * 4u + tap]);
    }
    for (uint token = 0u; token < rows; ++token) {
        uint offset = token * 10240u + channel;
        float current = hcb_float(normed[offset]);
        float acc = taps[3] * current
            + taps[0] * local_history[0]
            + taps[1] * local_history[3]
            + taps[2] * local_history[6];
        float exp_term = metal::precise::exp(-abs(acc));
        float sigmoid = acc >= 0.0f
            ? 1.0f / (1.0f + exp_term)
            : exp_term / (1.0f + exp_term);
        output[offset] = hcb_word(hcb_float(residual[offset])
            + hcb_float(gated[offset]) + acc * sigmoid);
        for (uint row = 0u; row < 8u; ++row) {
            local_history[row] = local_history[row + 1u];
        }
        local_history[8] = current;
    }
    for (uint row = 0u; row < 9u; ++row) {
        history[row * 10240u + channel] = hcb_word(local_history[row]);
    }
}
