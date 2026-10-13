// KVarN variance normalization modified for the Qwen K4/V4 cache record.
// Reduction order adapts Apple MLX reduce_col.h, reduce_row.h and reduce.cpp.
// See THIRD-PARTY-NOTICES for KVarN Apache-2.0 and Apple MLX MIT notices.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Match MLX's 32x32 column reduction: each of 256 threads accumulates
// four adjacent columns over every 32nd row, then a SIMD group reduces
// the 32 row partials for each output column.
kernel void kvarn_std_col(
    const device float* values [[buffer(0)]],
    device float* output [[buffer(1)]],
    const constant uint* dimensions [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dimensions[0];
    const uint cols = dimensions[1];
    const uint tiles = cols / 32u;
    const uint head = group / tiles;
    const uint tile = group % tiles;
    const uint local_column = (lid % 8u) * 4u;
    const uint row_lane = lid / 8u;
    const uint column = tile * 32u + local_column;
    values += head * rows * cols;
    output += head * cols;
    threadgroup float shared[1024];
    threadgroup float means[32];
    float totals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint row = row_lane; row < rows; row += 32u) {
        for (uint i = 0u; i < 4u; ++i) {
            totals[i] = values[row * cols + column + i] + totals[i];
        }
    }
    for (uint i = 0u; i < 4u; ++i) {
        shared[row_lane * 32u + local_column + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        for (uint i = 0u; i < 4u; ++i) {
            // SIMD reductions run on all lanes below; lane zero writes means.
            means[simd_group * 4u + i] = 0.0f;
        }
    }
    for (uint i = 0u; i < 4u; ++i) {
        const float sum = simd_sum(shared[lane * 32u + simd_group * 4u + i]);
        if (lane == 0u) { means[simd_group * 4u + i] = sum * (1.0f / float(rows)); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < 4u; ++i) { totals[i] = 0.0f; }
    for (uint row = row_lane; row < rows; row += 32u) {
        for (uint i = 0u; i < 4u; ++i) {
            const float delta = values[row * cols + column + i] - means[local_column + i];
            const float squared = delta * delta;
            totals[i] = squared + totals[i];
        }
    }
    for (uint i = 0u; i < 4u; ++i) {
        shared[row_lane * 32u + local_column + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < 4u; ++i) {
        const float sum = simd_sum(shared[lane * 32u + simd_group * 4u + i]);
        if (lane == 0u) {
            output[tile * 32u + simd_group * 4u + i] = metal::sqrt(sum / float(rows - 1u));
        }
    }
}

// Match MLX's row_reduce_simple: one SIMD group reduces four contiguous
// rows, reading four adjacent values per lane in each 128-wide block.
kernel void kvarn_std_row(
    const device float* values [[buffer(0)]],
    device float* output [[buffer(1)]],
    const constant uint* dimensions [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    const uint rows = dimensions[0];
    const uint cols = dimensions[1];
    const uint groups_per_head = rows / 4u;
    const uint head = group / groups_per_head;
    const uint first_row = group % groups_per_head * 4u;
    values += head * rows * cols;
    output += head * rows;
    for (uint row_offset = 0u; row_offset < 4u; ++row_offset) {
        const uint row = first_row + row_offset;
        float total = 0.0f;
        for (uint block = 0u; block < cols / 128u; ++block) {
            for (uint i = 0u; i < 4u; ++i) {
                total = values[row * cols + block * 128u + lane * 4u + i] + total;
            }
        }
        const float mean = simd_sum(total) * (1.0f / float(cols));
        float variance = 0.0f;
        for (uint block = 0u; block < cols / 128u; ++block) {
            for (uint i = 0u; i < 4u; ++i) {
                const float delta = values[row * cols + block * 128u + lane * 4u + i] - mean;
                const float squared = delta * delta;
                variance = squared + variance;
            }
        }
        const float sum = simd_sum(variance);
        if (lane == 0u) { output[row] = metal::sqrt(sum / float(cols - 1u)); }
    }
}

inline float kvarn_score_from_std(
    const device float* col_std, const device float* row_std, uint rows, uint cols
) {
    float col_min = INFINITY;
    float col_max = -INFINITY;
    float row_min = INFINITY;
    float row_max = -INFINITY;
    for (uint col = 0u; col < cols; ++col) {
        col_min = metal::min(col_min, col_std[col]);
        col_max = metal::max(col_max, col_std[col]);
    }
    for (uint row = 0u; row < rows; ++row) {
        row_min = metal::min(row_min, row_std[row]);
        row_max = metal::max(row_max, row_std[row]);
    }
    return col_max / metal::max(col_min, 1e-8f)
        + row_max / metal::max(row_min, 1e-8f);
}

kernel void kvarn_init(
    const device float* source [[buffer(0)]],
    device float* current [[buffer(1)]],
    device float* log_col [[buffer(2)]],
    device float* log_row [[buffer(3)]],
    device float* best_col [[buffer(4)]],
    device float* best_row [[buffer(5)]],
    const constant uint* dimensions [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    const uint rows = dimensions[0];
    const uint cols = dimensions[1];
    const uint elements = rows * cols;
    const uint head = index / elements;
    const uint local = index % elements;
    current[index] = source[index];
    if (local < cols) {
        log_col[head * cols + local] = 0.0f;
        best_col[head * cols + local] = 1.0f;
    }
    if (local < rows) {
        log_row[head * rows + local] = 0.0f;
        best_row[head * rows + local] = 1.0f;
    }
}

kernel void kvarn_score_init(
    const device float* col_std [[buffer(0)]],
    const device float* row_std [[buffer(1)]],
    device float* best_score [[buffer(2)]],
    const constant uint* dimensions [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]]
) {
    const uint rows = dimensions[0];
    const uint cols = dimensions[1];
    best_score[head] = kvarn_score_from_std(
        col_std + head * cols, row_std + head * rows, rows, cols);
}

kernel void kvarn_update_col_log(
    const device float* col_std [[buffer(0)]],
    device float* log_col [[buffer(1)]],
    const constant uint* dimensions [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    const uint cols = dimensions[1];
    const uint head = index / cols;
    const uint col = index % cols;
    const float s = metal::clamp(col_std[head * cols + col], 1e-3f, 1e3f);
    log_col[head * cols + col] = metal::clamp(
        log_col[head * cols + col] + metal::log(s), -0.3f, 10.0f);
}

kernel void kvarn_update_row_log(
    const device float* row_std [[buffer(0)]],
    device float* log_row [[buffer(1)]],
    const constant uint* dimensions [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    const uint rows = dimensions[0];
    const uint head = index / rows;
    const uint row = index % rows;
    const float s = metal::clamp(row_std[head * rows + row], 1e-3f, 1e3f);
    log_row[head * rows + row] = metal::clamp(
        log_row[head * rows + row] + metal::log(s), -0.3f, 10.0f);
}

// A separate dispatch makes every updated log visible before any map reader.
// Each thread owns one output cell; no grid-wide barrier or intermediate copy is needed.
kernel void kvarn_update_map(
    const device float* source [[buffer(0)]],
    device float* current [[buffer(1)]],
    const device float* log_col [[buffer(2)]],
    const device float* log_row [[buffer(3)]],
    const constant uint* dimensions [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    const uint rows = dimensions[0];
    const uint cols = dimensions[1];
    const uint elements = rows * cols;
    const uint head = index / elements;
    const uint local = index % elements;
    const uint row = local / cols;
    const uint col = local % cols;
    current[index] = source[index] / metal::exp(log_col[head * cols + col])
        / metal::exp(log_row[head * rows + row]);
}

kernel void kvarn_score_select(
    const device float* col_std [[buffer(0)]],
    const device float* row_std [[buffer(1)]],
    const device float* log_col [[buffer(2)]],
    const device float* log_row [[buffer(3)]],
    device float* best_col [[buffer(4)]],
    device float* best_row [[buffer(5)]],
    device float* best_score [[buffer(6)]],
    const constant uint* dimensions [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]]
) {
    const uint rows = dimensions[0];
    const uint cols = dimensions[1];
    const float score = kvarn_score_from_std(
        col_std + head * cols, row_std + head * rows, rows, cols);
    if (score <= best_score[head]) {
        best_score[head] = score;
        log_col += head * cols;
        log_row += head * rows;
        best_col += head * cols;
        best_row += head * rows;
        for (uint col = 0u; col < cols; ++col) { best_col[col] = metal::exp(log_col[col]); }
        for (uint row = 0u; row < rows; ++row) { best_row[row] = metal::exp(log_row[row]); }
    }
}

kernel void kvarn_finalize(
    const device float* source [[buffer(0)]],
    const device float* best_col [[buffer(1)]],
    const device float* best_row [[buffer(2)]],
    device float* balanced [[buffer(3)]],
    const constant uint* dimensions [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    const uint rows = dimensions[0];
    const uint cols = dimensions[1];
    const uint elements = rows * cols;
    const uint head = index / elements;
    const uint local = index % elements;
    const uint row = local / cols;
    const uint col = local % cols;
    balanced[index] = source[index] / best_col[head * cols + col]
        / best_row[head * rows + row];
}

// The state layout and quantization fields follow the K4/V4 record contract.
// The transform uses the same 256-wide butterfly as the host codec.
kernel void kvarn_prepare_seal(
    const device ushort* tail_keys [[buffer(0)]],
    const device ushort* tail_values [[buffer(1)]],
    device float* key_matrix [[buffer(2)]],
    device float* value_matrix [[buffer(3)]],
    const constant uint* tile_start [[buffer(4)]],
    uint token_head [[threadgroup_position_in_grid]],
    uint channel [[thread_position_in_threadgroup]]
) {
    const uint head = token_head / 128u;
    const uint token = token_head % 128u;
    const uint slot = (tile_start[0] + token - 128u) % 8320u;
    const uint index = (head * 8320u + slot) * 256u + channel;
    threadgroup float k[256];
    threadgroup float v[256];
    k[channel] = as_type<float>(uint(tail_keys[index]) << 16u);
    v[channel] = as_type<float>(uint(tail_values[index]) << 16u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        const uint peer = channel ^ stride;
        const float ka = k[channel], kb = k[peer];
        const float va = v[channel], vb = v[peer];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k[channel] = (channel & stride) == 0u ? ka + kb : kb - ka;
        v[channel] = (channel & stride) == 0u ? va + vb : vb - va;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    key_matrix[(head * 256u + channel) * 128u + token] = k[channel] * 0.0625f;
    value_matrix[(head * 128u + token) * 256u + channel] = v[channel] * 0.0625f;
}

kernel void kvarn_copy_records(
    const device uchar* source [[buffer(0)]],
    device uchar* target [[buffer(1)]],
    uint byte_index [[thread_position_in_grid]]
) {
    target[byte_index] = source[byte_index];
}

kernel void kvarn_append_token(
    const device ushort* keys [[buffer(0)]],
    const device ushort* values [[buffer(1)]],
    device ushort* sink_keys [[buffer(2)]],
    device ushort* sink_values [[buffer(3)]],
    device ushort* tail_keys [[buffer(4)]],
    device ushort* tail_values [[buffer(5)]],
    const constant uint* frontier [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    const uint head = index / 256u;
    const uint channel = index % 256u;
    if (frontier[0] < 128u) {
        const uint dst = (head * 128u + frontier[0]) * 256u + channel;
        sink_keys[dst] = keys[index];
        sink_values[dst] = values[index];
    } else {
        const uint slot = (frontier[0] - 128u) % 8320u;
        const uint dst = (head * 8320u + slot) * 256u + channel;
        tail_keys[dst] = keys[index];
        tail_values[dst] = values[index];
    }
}

// A segment stops before the next seal-causing row. Distinct token/head lanes
// therefore cannot overwrite ring rows that an earlier seal still needs.
kernel void kvarn_append_chunk(
    const device ushort* keys [[buffer(0)]],
    const device ushort* values [[buffer(1)]],
    device ushort* sink_keys [[buffer(2)]],
    device ushort* sink_values [[buffer(3)]],
    device ushort* tail_keys [[buffer(4)]],
    device ushort* tail_values [[buffer(5)]],
    const constant uint* args [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    const uint row = index / 512u;
    if (row >= args[1]) return;
    const uint position = args[0] + row;
    const uint head = (index % 512u) / 256u;
    const uint channel = index % 256u;
    if (position < 128u) {
        const uint dst = (head * 128u + position) * 256u + channel;
        sink_keys[dst] = keys[index];
        sink_values[dst] = values[index];
    } else {
        const uint slot = (position - 128u) % 8320u;
        const uint dst = (head * 8320u + slot) * 256u + channel;
        tail_keys[dst] = keys[index];
        tail_values[dst] = values[index];
    }
}

inline void kvarn_check_half(float value, device atomic_uint* status) {
    if (!metal::isfinite(value) || !metal::isfinite(float(half(value)))) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}

kernel void kvarn_pack_key(
    const device float* balanced [[buffer(0)]],
    const device float* token_scale [[buffer(1)]],
    const device float* channel_scale [[buffer(2)]],
    device uchar* records [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    const constant uint* record_offset [[buffer(5)]],
    uint pair_head [[thread_position_in_grid]]
) {
    const uint head = pair_head / 128u;
    const uint pair = pair_head % 128u;
    const uint row0 = 2u * pair, row1 = row0 + 1u;
    const device float* matrix = balanced + head * 256u * 128u;
    device uchar* record = records + record_offset[0] + head * 35072u;
    float lo0 = INFINITY, hi0 = -INFINITY, lo1 = INFINITY, hi1 = -INFINITY;
    for (uint token = 0u; token < 128u; ++token) {
        const float a = matrix[row0 * 128u + token];
        const float b = matrix[row1 * 128u + token];
        lo0 = metal::min(lo0, a); hi0 = metal::max(hi0, a);
        lo1 = metal::min(lo1, b); hi1 = metal::max(hi1, b);
    }
    const float scale0 = metal::max((hi0 - lo0) / 15.0f, 1e-10f);
    const float scale1 = metal::max((hi1 - lo1) / 15.0f, 1e-10f);
    if (!metal::isfinite(lo0) || !metal::isfinite(hi0) || !metal::isfinite(lo1)
        || !metal::isfinite(hi1) || !metal::isfinite(scale0) || !metal::isfinite(scale1)) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        return;
    }
    for (uint token = 0u; token < 128u; ++token) {
        const uint a = uint(metal::clamp(metal::rint((matrix[row0 * 128u + token] - lo0) / scale0), 0.0f, 15.0f));
        const uint b = uint(metal::clamp(metal::rint((matrix[row1 * 128u + token] - lo1) / scale1), 0.0f, 15.0f));
        record[token * 128u + pair] = uchar(a | (b << 4u));
    }
    device half* meta = (device half*) record;
    const float s0 = channel_scale[head * 256u + row0];
    const float s1 = channel_scale[head * 256u + row1];
    const float values[4] = {s0 * scale0, s1 * scale1, s0 * lo0, s1 * lo1};
    for (uint i = 0u; i < 4u; ++i) kvarn_check_half(values[i], status);
    meta[(16384u / 2u) + row0] = half(values[0]);
    meta[(16384u / 2u) + row1] = half(values[1]);
    meta[(16896u / 2u) + row0] = half(values[2]);
    meta[(16896u / 2u) + row1] = half(values[3]);
    const float ts = token_scale[head * 128u + pair];
    kvarn_check_half(ts, status);
    meta[(17408u / 2u) + pair] = half(ts);
}

kernel void kvarn_pack_value(
    const device float* balanced [[buffer(0)]],
    const device float* channel_scale [[buffer(1)]],
    const device float* token_scale [[buffer(2)]],
    device uchar* records [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    const constant uint* record_offset [[buffer(5)]],
    uint token_head [[thread_position_in_grid]]
) {
    const uint head = token_head / 128u;
    const uint token = token_head % 128u;
    const device float* row = balanced + (head * 128u + token) * 256u;
    device uchar* record = records + record_offset[0] + head * 35072u;
    float low = INFINITY, high = -INFINITY;
    for (uint channel = 0u; channel < 256u; ++channel) {
        low = metal::min(low, row[channel]);
        high = metal::max(high, row[channel]);
    }
    const float scale = metal::max((high - low) / 15.0f, 1e-10f);
    if (!metal::isfinite(low) || !metal::isfinite(high) || !metal::isfinite(scale)) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        return;
    }
    for (uint pair = 0u; pair < 128u; ++pair) {
        const uint a = uint(metal::clamp(metal::rint((row[2u * pair] - low) / scale), 0.0f, 15.0f));
        const uint b = uint(metal::clamp(metal::rint((row[2u * pair + 1u] - low) / scale), 0.0f, 15.0f));
        record[17664u + token * 128u + pair] = uchar(a | (b << 4u));
    }
    device half* meta = (device half*) record;
    const float ts = token_scale[head * 128u + token];
    const float scaled = ts * scale, zero = ts * low;
    kvarn_check_half(scaled, status);
    kvarn_check_half(zero, status);
    meta[(34560u / 2u) + token] = half(scaled);
    meta[(34816u / 2u) + token] = half(zero);
    for (uint i = 0u; i < 2u; ++i) {
        const uint channel = 2u * token + i;
        const float cs = channel_scale[head * 256u + channel];
        kvarn_check_half(cs, status);
        meta[(34048u / 2u) + channel] = half(cs);
    }
}
