// Generic Qwen GDN preparation and the time-loop recurrence follow MLX and
// mlx-lm (MIT), conv.metal, reduction/reduce_row.h and models/gated_delta.py.
// Copyright (c) 2023-2024 Apple Inc. See THIRD-PARTY-NOTICES.
// Scalar BF16 and decay helpers are shared with gdn.metal.

kernel void gdn_batch_conv(
    device const ushort* qkv [[buffer(0)]], device const ushort* previous [[buffer(1)]],
    device const ushort* weight [[buffer(2)]], device ushort* convolved [[buffer(3)]],
    device ushort* activated [[buffer(4)]], device ushort* next [[buffer(5)]],
    constant uint& rows [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    uint token = index / 10240, channel = index % 10240;
    if (token < 3) {
        uint position = rows + token;
        next[index] = position < 3 ? previous[position * 10240 + channel]
            : qkv[(position - 3) * 10240 + channel];
    }
    if (token >= rows) return;
    float acc = 0.0f;
    for (uint tap = 0; tap < 4; ++tap) {
        uint position = token + tap;
        ushort source = position < 3 ? previous[position * 10240 + channel]
            : qkv[(position - 3) * 10240 + channel];
        acc += gdn_f(source) * gdn_f(weight[channel * 4 + tap]);
    }
    ushort value = gdn_bf(acc);
    convolved[index] = value;
    activated[index] = gdn_bf(gdn_f(value) * gdn_f(gdn_sigmoid_bf(value)));
}

kernel void gdn_batch_prepare(
    device const ushort* activated [[buffer(0)]], device const ushort* beta_logits [[buffer(1)]],
    device const ushort* decay_logits [[buffer(2)]], device const ushort* a_log [[buffer(3)]],
    device const ushort* dt_bias [[buffer(4)]], device ushort* query [[buffer(5)]],
    device ushort* key [[buffer(6)]], device ushort* value [[buffer(7)]],
    device ushort* beta [[buffer(8)]], device float* decay [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint token = group / 80, head = group % 80, first = tid * 4;
    ushort x[4];
    for (uint i = 0; i < 4; ++i) x[i] = activated[token * 10240 + head * 128 + first + i];
    ushort inverse = gdn_bf(1.0f);
    if (head < 32) {
        ushort partial = gdn_bf(0.0f);
        for (uint i = 0; i < 4; ++i) {
            ushort square = gdn_bf(gdn_f(x[i]) * gdn_f(x[i]));
            partial = gdn_bf(gdn_f(partial) + gdn_f(square));
        }
        ushort sum = gdn_bf(simd_sum(gdn_f(partial)));
        ushort stabilized = gdn_bf(gdn_f(sum) + gdn_f(gdn_bf(1e-6f)));
        inverse = gdn_bf(precise::rsqrt(gdn_f(stabilized)));
    }
    for (uint i = 0; i < 4; ++i) {
        ushort result = head < 32 ? gdn_bf(gdn_f(x[i]) * gdn_f(inverse)) : x[i];
        if (head < 16) {
            result = gdn_bf(gdn_f(result) * gdn_f(gdn_bf(0.08838834764831845f)));
            query[token * 2048 + head * 128 + first + i] = result;
        } else if (head < 32) key[token * 2048 + (head - 16) * 128 + first + i] = result;
        else value[token * 6144 + (head - 32) * 128 + first + i] = result;
    }
    if (head < 48 && tid == 0) {
        uint at = token * 48 + head;
        beta[at] = gdn_sigmoid_bf(beta_logits[at]);
        float xg = gdn_f(decay_logits[at]) + gdn_f(dt_bias[head]);
        float upper = metal::max(xg, 0.0f), lower = metal::min(xg, 0.0f);
        float softplus = upper + gdn_log1p(metal::exp(lower - upper));
        float rate = precise::exp(gdn_f(a_log[head])) * softplus;
        decay[at] = precise::exp(-rate);
    }
}

kernel void gdn_batch_recurrence(
    device const ushort* query [[buffer(0)]], device const ushort* key [[buffer(1)]],
    device const ushort* value [[buffer(2)]], device const float* decay [[buffer(3)]],
    device const ushort* beta [[buffer(4)]], device const float* previous [[buffer(5)]],
    device float* next [[buffer(6)]], device ushort* output [[buffer(7)]],
    constant uint& rows [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint head = group / 32, dimension = (group % 32) * 4 + (tid >> 5);
    uint key_head = head / 3, state_base = (head * 128 + dimension) * 128;
    float state[4];
    for (uint i = 0; i < 4; ++i) state[i] = previous[state_base + lane * 4 + i];
    for (uint token = 0; token < rows; ++token) {
        uint kb = token * 2048 + key_head * 128;
        uint vb = token * 6144 + head * 128;
        float memory = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            state[i] = state[i] * decay[token * 48 + head];
            memory += state[i] * gdn_f(key[kb + lane * 4 + i]);
        }
        memory = simd_sum(memory);
        float delta = (gdn_f(value[vb + dimension]) - memory) * gdn_f(beta[token * 48 + head]);
        float projection = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            state[i] = state[i] + gdn_f(key[kb + lane * 4 + i]) * delta;
            projection += state[i] * gdn_f(query[kb + lane * 4 + i]);
        }
        projection = simd_sum(projection);
        if (lane == 0) output[vb + dimension] = gdn_bf(projection);
    }
    for (uint i = 0; i < 4; ++i) next[state_base + lane * 4 + i] = state[i];
}

// Four value rows per SIMD group reuse query/key loads, following the
// MIT antirez/ds4 Qwen GDN scan. Native grouped-query heads are contiguous.
kernel void gdn_batch_recurrence_r4(
    device const ushort* query [[buffer(0)]], device const ushort* key [[buffer(1)]],
    device const ushort* value [[buffer(2)]], device const float* decay [[buffer(3)]],
    device const ushort* beta [[buffer(4)]], device const float* previous [[buffer(5)]],
    device float* next [[buffer(6)]], device ushort* output [[buffer(7)]],
    constant uint& rows [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint head = group / 32, dimension = (group % 32) * 4;
    uint key_head = head / 3, state_base = (head * 128 + dimension) * 128;
    float4 state[4];
    for (uint r = 0; r < 4; ++r)
        state[r] = *((device const float4*)(previous + state_base + r * 128 + lane * 4));
    for (uint token = 0; token < rows; ++token) {
        uint kb = token * 2048 + key_head * 128 + lane * 4;
        uint vb = token * 6144 + head * 128 + dimension;
        float4 q, k;
        for (uint i = 0; i < 4; ++i) { q[i] = gdn_f(query[kb + i]); k[i] = gdn_f(key[kb + i]); }
        float g = decay[token * 48 + head], b = gdn_f(beta[token * 48 + head]);
        float memory[4], projected[4];
        for (uint r = 0; r < 4; ++r) { state[r] *= g; memory[r] = dot(state[r], k); }
        for (uint r = 0; r < 4; ++r) memory[r] = simd_sum(memory[r]);
        for (uint r = 0; r < 4; ++r) {
            state[r] += k * ((gdn_f(value[vb + r]) - memory[r]) * b);
            projected[r] = dot(state[r], q);
        }
        for (uint r = 0; r < 4; ++r) projected[r] = simd_sum(projected[r]);
        if (lane == 0) for (uint r = 0; r < 4; ++r) output[vb + r] = gdn_bf(projected[r]);
    }
    for (uint r = 0; r < 4; ++r)
        *((device float4*)(next + state_base + r * 128 + lane * 4)) = state[r];
}
