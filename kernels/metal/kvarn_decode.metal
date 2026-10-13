// KVarN decoder modified for selected rows in token-major packed records.
// See THIRD-PARTY-NOTICES for the KVarN numeric method attribution.
#include <metal_stdlib>
using namespace metal;

inline ushort kvarn_bf(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

kernel void kvarn_gather(
    const device uchar* records [[buffer(0)]],
    const device ushort* sink_keys [[buffer(1)]],
    const device ushort* sink_values [[buffer(2)]],
    const device ushort* tail_keys [[buffer(3)]],
    const device ushort* tail_values [[buffer(4)]],
    const device int* logical_ids [[buffer(5)]],
    const device uchar* valid [[buffer(6)]],
    const constant uint* counters [[buffer(7)]],
    device ushort* gathered_keys [[buffer(8)]],
    device ushort* gathered_values [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint channel [[thread_position_in_threadgroup]]
) {
    const uint row_count = counters[2];
    const uint row = group / 2u;
    const uint head = group % 2u;
    if (row >= row_count || channel >= 256u) return;
    const int logical = logical_ids[row];
    const int frontier = int(counters[0]);
    const int body_frontier = 128 + int(counters[1]) * 128;
    const uint output = (head * row_count + row) * 256u + channel;
    if (!valid[row] || logical < 0 || logical >= frontier) {
        gathered_keys[output] = 0u;
        gathered_values[output] = 0u;
        return;
    }
    if (logical < 128) {
        const uint source = (head * 128u + uint(logical)) * 256u + channel;
        gathered_keys[output] = sink_keys[source];
        gathered_values[output] = sink_values[source];
        return;
    }
    if (logical >= body_frontier) {
        const uint slot = uint(logical - 128) % 8320u;
        const uint source = (head * 8320u + slot) * 256u + channel;
        gathered_keys[output] = tail_keys[source];
        gathered_values[output] = tail_values[source];
        return;
    }

    const uint body_local = uint(logical - 128);
    const uint tile = body_local / 128u;
    const uint token = body_local % 128u;
    const device uchar* record = records + (tile * 2u + head) * 35072u;
    const uint code_column = channel >> 1u;
    const uint shift = (channel & 1u) * 4u;
    const uchar packed_k = record[token * 128u + code_column];
    const float k_code = float((packed_k >> shift) & 0x0fu);
    const device half* k_scale = (const device half*)(record + 16384u);
    const device half* k_zero = (const device half*)(record + 16896u);
    const device half* k_token_scale = (const device half*)(record + 17408u);
    float key = (k_code * float(k_scale[channel]) + float(k_zero[channel]))
        * float(k_token_scale[token]);
    const uchar packed_v = record[17664u + token * 128u + code_column];
    const float v_code = float((packed_v >> shift) & 0x0fu);
    const device half* v_channel_scale = (const device half*)(record + 34048u);
    const device half* v_token_scale = (const device half*)(record + 34560u);
    const device half* v_zero = (const device half*)(record + 34816u);
    float value = (v_code * float(v_token_scale[token]) + float(v_zero[token]))
        * float(v_channel_scale[channel]);
    for (uint stride = 1u; stride < 32u; stride <<= 1u) {
        const float other_key = simd_shuffle_xor(key, stride);
        const float other_value = simd_shuffle_xor(value, stride);
        key = (channel & stride) ? other_key - key : key + other_key;
        value = (channel & stride) ? other_value - value : value + other_value;
    }
    threadgroup float keys[256];
    threadgroup float values[256];
    keys[channel] = key;
    values[channel] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32u; stride <= 128u; stride <<= 1u) {
        const float other_key = keys[channel ^ stride];
        const float other_value = values[channel ^ stride];
        key = (channel & stride) ? other_key - key : key + other_key;
        value = (channel & stride) ? other_value - value : value + other_value;
        if (stride < 128u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            keys[channel] = key;
            values[channel] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    gathered_keys[output] = kvarn_bf(key * 0.0625f);
    gathered_values[output] = kvarn_bf(value * 0.0625f);
}

// Shared prefill reconstruction preserves the existing inverse transform and
// BF16 rounding. Every thread in a group supplies the same logical row/head.
inline ushort2 kvarn_prefill_row(
    const device uchar* records, const device ushort* sink_keys,
    const device ushort* sink_values, const device ushort* tail_keys,
    const device ushort* tail_values, int logical, uint head, uint channel,
    int body_frontier, threadgroup float* keys, threadgroup float* values) {
    if (logical < 128) {
        const uint source = (head * 128u + uint(logical)) * 256u + channel;
        return ushort2(sink_keys[source], sink_values[source]);
    }
    if (logical >= body_frontier) {
        const uint slot = uint(logical - 128) % 8320u;
        const uint source = (head * 8320u + slot) * 256u + channel;
        return ushort2(tail_keys[source], tail_values[source]);
    }
    const uint body_local = uint(logical - 128);
    const uint tile = body_local / 128u;
    const uint token = body_local % 128u;
    const device uchar* record = records + (tile * 2u + head) * 35072u;
    const uint code_column = channel >> 1u;
    const uint shift = (channel & 1u) * 4u;
    const uchar packed_k = record[token * 128u + code_column];
    const float k_code = float((packed_k >> shift) & 0x0fu);
    const device half* k_scale = (const device half*)(record + 16384u);
    const device half* k_zero = (const device half*)(record + 16896u);
    const device half* k_token_scale = (const device half*)(record + 17408u);
    float key = (k_code * float(k_scale[channel]) + float(k_zero[channel]))
        * float(k_token_scale[token]);
    const uchar packed_v = record[17664u + token * 128u + code_column];
    const float v_code = float((packed_v >> shift) & 0x0fu);
    const device half* v_channel_scale = (const device half*)(record + 34048u);
    const device half* v_token_scale = (const device half*)(record + 34560u);
    const device half* v_zero = (const device half*)(record + 34816u);
    float value = (v_code * float(v_token_scale[token]) + float(v_zero[token]))
        * float(v_channel_scale[channel]);
    for (uint stride = 1u; stride < 32u; stride <<= 1u) {
        const float other_key = simd_shuffle_xor(key, stride);
        const float other_value = simd_shuffle_xor(value, stride);
        key = (channel & stride) ? other_key - key : key + other_key;
        value = (channel & stride) ? other_value - value : value + other_value;
    }
    keys[channel] = key;
    values[channel] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32u; stride <= 128u; stride <<= 1u) {
        const float other_key = keys[channel ^ stride];
        const float other_value = values[channel ^ stride];
        key = (channel & stride) ? other_key - key : key + other_key;
        value = (channel & stride) ? other_value - value : value + other_value;
        if (stride < 128u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            keys[channel] = key;
            values[channel] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    return ushort2(kvarn_bf(key * 0.0625f), kvarn_bf(value * 0.0625f));
}

// The prefill tile shares decode's record layout and inverse transform. One
// grid covers all selected rows, retaining query-level GPU occupancy.
kernel void kvarn_gather_tile(
    const device uchar* records [[buffer(0)]],
    const device ushort* sink_keys [[buffer(1)]],
    const device ushort* sink_values [[buffer(2)]],
    const device ushort* tail_keys [[buffer(3)]],
    const device ushort* tail_values [[buffer(4)]],
    const device int* logical_ids [[buffer(5)]],
    const device uchar* valid [[buffer(6)]],
    const constant uint* counters [[buffer(7)]],
    device ushort* gathered_keys [[buffer(8)]],
    device ushort* gathered_values [[buffer(9)]],
    const constant uint* rows [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint channel [[thread_position_in_threadgroup]]) {
    constexpr uint width = 2051u;
    const uint query = group / (2u * width);
    const uint local_group = group % (2u * width);
    if (query >= rows[0] || channel >= 256u) return;
    const uint row = local_group / 2u, head = local_group % 2u;
    const int logical = logical_ids[query * width + row];
    const int frontier = int(counters[0]);
    const int body_frontier = 128 + int(counters[1]) * 128;
    const uint output = (query * 2u * width + head * width + row) * 256u + channel;
    if (!valid[query * width + row] || logical < 0 || logical >= frontier) {
        gathered_keys[output] = 0u;
        gathered_values[output] = 0u;
        return;
    }
    threadgroup float keys[256];
    threadgroup float values[256];
    const ushort2 row_value = kvarn_prefill_row(records, sink_keys, sink_values,
        tail_keys, tail_values, logical, head, channel, body_frontier, keys, values);
    gathered_keys[output] = row_value.x;
    gathered_values[output] = row_value.y;
}

// Reconstruct each logical row once into bounded token-major scratch.
kernel void kvarn_reconstruct_prefill(
    const device uchar* records [[buffer(0)]],
    const device ushort* sink_keys [[buffer(1)]],
    const device ushort* sink_values [[buffer(2)]],
    const device ushort* tail_keys [[buffer(3)]],
    const device ushort* tail_values [[buffer(4)]],
    const device ushort* pending_keys [[buffer(5)]],
    const device ushort* pending_values [[buffer(6)]],
    const constant uint* shape [[buffer(7)]],
    device ushort* keys_out [[buffer(8)]],
    device ushort* values_out [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint channel [[thread_position_in_threadgroup]]) {
    const uint logical = group / 2u, head = group % 2u;
    const uint frontier = shape[0], total = frontier + shape[2];
    if (logical >= total || channel >= 256u) return;
    const uint output = (logical * 2u + head) * 256u + channel;
    if (logical >= frontier) {
        const uint source = ((logical - frontier) * 2u + head) * 256u + channel;
        keys_out[output] = pending_keys[source];
        values_out[output] = pending_values[source];
        return;
    }
    threadgroup float keys[256];
    threadgroup float values[256];
    const ushort2 row_value = kvarn_prefill_row(records, sink_keys, sink_values,
        tail_keys, tail_values, int(logical), head, channel,
        128 + int(shape[1]) * 128, keys, values);
    keys_out[output] = row_value.x;
    values_out[output] = row_value.y;
}
