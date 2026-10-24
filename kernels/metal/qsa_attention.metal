// QSA selected-row arithmetic uses BF16 matmul/FP32 softmax
// boundaries. The softmax row reduction follows MLX softmax.h (MIT).
// The sigmoid follows MLX unary_ops.h (MIT). See THIRD-PARTY-NOTICES.
#include <metal_stdlib>
using namespace metal;

inline float qa_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline ushort qa_round(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Head h reads the matching KV head h/12. Slots are in ascending physical
// order, and invalid slots form a trailing suffix.
kernel void qsa_selected_logits(
    device const ushort* queries [[buffer(0)]],
    device const ushort* keys [[buffer(1)]],
    device const uchar* valid [[buffer(2)]],
    device ushort* logits [[buffer(3)]],
    device ushort* scaled [[buffer(4)]],
    device float* masked [[buffer(5)]],
    constant uint* shape [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint rows = shape[0];
    if (index >= shape[1] * 24 * rows) return;
    const uint query = index / (24 * rows);
    queries += query * 24 * 256;
    keys += query * 2 * rows * 256;
    valid += query * rows;
    uint head = (index / rows) % 24;
    uint row = index % rows;
    uint kv_head = head / 12;
    float sum = 0.0f;
    for (uint d = 0; d < 256; ++d) {
        sum = fma(qa_bf16(queries[head * 256 + d]),
            qa_bf16(keys[(kv_head * rows + row) * 256 + d]), sum);
    }
    ushort logit = qa_round(sum);
    ushort scaled_logit = qa_round(qa_bf16(logit) * 0.0625f);
    logits[index] = logit;
    scaled[index] = scaled_logit;
    masked[index] = valid[row] ? qa_bf16(scaled_logit) : -INFINITY;
}

// MLX's block float32 softmax reads four adjacent values per thread. The
// 2051-slot row uses 17 SIMD groups, 544 threads per Q head.
kernel void qsa_selected_softmax(
    device const float* masked [[buffer(0)]],
    device ushort* probabilities [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint rows = shape[0];
    if (head >= shape[1] * 24) return;
    threadgroup float local_max[32];
    threadgroup float local_sum[32];
    float values[4];
    for (uint i = 0; i < 4; ++i) {
        uint row = lid * 4 + i;
        values[i] = row < rows ? masked[head * rows + row] : -INFINITY;
    }
    if (simd == 0) {
        local_max[lane] = -INFINITY;
        local_sum[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = -MAXFLOAT;
    for (uint i = 0; i < 4; ++i) maximum = maximum < values[i] ? values[i] : maximum;
    maximum = simd_max(maximum);
    if (lane == 0) local_max[simd] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        maximum = simd_max(local_max[lane]);
        if (lane == 0) local_max[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = local_max[0];
    float total = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        values[i] = fast::exp(values[i] - maximum);
        total += values[i];
    }
    total = simd_sum(total);
    if (lane == 0) local_sum[simd] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        total = simd_sum(local_sum[lane]);
        if (lane == 0) local_sum[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = 1.0f / local_sum[0];
    for (uint i = 0; i < 4; ++i) {
        uint row = lid * 4 + i;
        if (row < rows) probabilities[head * rows + row] = qa_round(values[i] * inverse);
    }
}

inline float qa_sigmoid(float value) {
    float exponential = qa_bf16(qa_round(precise::exp(metal::abs(value))));
    float denominator = qa_bf16(qa_round(1.0f + exponential));
    float reciprocal = qa_bf16(qa_round(1.0f / denominator));
    return value < 0.0f ? reciprocal : qa_bf16(qa_round(1.0f - reciprocal));
}

kernel void qsa_selected_value_gate(
    device const ushort* probabilities [[buffer(0)]],
    device const ushort* values [[buffer(1)]],
    device const ushort* gate [[buffer(2)]],
    device ushort* attention [[buffer(3)]],
    device ushort* sigmoid [[buffer(4)]],
    device ushort* gated [[buffer(5)]],
    constant uint* shape [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint rows = shape[0];
    if (index >= shape[1] * 24 * 256) return;
    const uint query = index / (24 * 256);
    probabilities += query * 24 * rows;
    values += query * 2 * rows * 256;
    uint head = (index / 256) % 24;
    uint dimension = index % 256;
    uint kv_head = head / 12;
    float sum = 0.0f;
    // Steel's unaligned-K matmul accumulates the final partial tile before
    // the aligned tiles. At 2051 rows, the final three products come first.
    const uint aligned = rows / 32u * 32u;
    for (uint row = aligned; row < rows; ++row) {
        sum = fma(qa_bf16(probabilities[head * rows + row]),
            qa_bf16(values[(kv_head * rows + row) * 256 + dimension]), sum);
    }
    for (uint row = 0; row < aligned; ++row) {
        sum = fma(qa_bf16(probabilities[head * rows + row]),
            qa_bf16(values[(kv_head * rows + row) * 256 + dimension]), sum);
    }
    ushort attended = qa_round(sum);
    ushort logistic = qa_round(qa_sigmoid(qa_bf16(gate[index])));
    attention[index] = attended;
    sigmoid[index] = logistic;
    gated[index] = qa_round(qa_bf16(attended) * qa_bf16(logistic));
}

// Adapted from DS4 kernel_qwen4_attn_mm (MIT): 16 selected keys per online
// step, matrix-unit score/value tiles, and one group per query/KV head. The
// KVarN gather supplies query-major BF16 rows instead of a half KV
// cache, so operands are converted while staging each tile.
// Shared native reduction arithmetic. Only K/V addressing is specialized.
struct QsaPrefillShared {
    half kv[2u * 16u * 256u];
    half scaled_queries[16u * 256u];
    float score_tile[2][2][64];
    half probabilities[4][128];
    float diagonals[4][64];
    float identity[64];
    int key_positions[16];
};

inline ushort qsa_finish_gate(float attended, float logistic, device atomic_uint* status) {
    const float mixed = attended * logistic;
    if (!metal::isfinite(attended) || !metal::isfinite(logistic) || !metal::isfinite(mixed))
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    return qa_round(mixed);
}

template<bool indexed, bool DeferredGate = false>
inline void qsa_prefill_online_body(
    const device ushort* queries,
    const device ushort* gates,
    const device ushort* gathered_keys,
    const device ushort* gathered_values,
    const device uchar* valid,
    const device uint* selected_count,
    device ushort* attention,
    device ushort* sigmoid,
    device ushort* gated,
    device atomic_uint* status,
    const constant uint* rows,
    uint group_index,
    ushort tid,
    ushort simd_group,
    ushort lane,
    const device int* logical_ids, const constant uint* geometry,
    threadgroup QsaPrefillShared& shared, device float* attended_output = nullptr) {
    const uint query = group_index / 2u, kv_head = group_index % 2u;
    if (query >= rows[0]) return;
    constexpr uint D = 256u, selected = 2051u, key_tile = 16u;
    const uint n = min(selected_count[query], selected);
    const uint row_tile = simd_group & 1u, dimension_half = simd_group >> 1u;
    const uint local_row = lane >> 2u, local_column = (lane & 3u) * 4u;
    threadgroup half* staged_keys = shared.kv;
    threadgroup half* staged_values = shared.kv + key_tile * D;

    for (uint i = tid; i < 16u * D; i += 128u) {
        const uint head = i / D, dimension = i % D;
        const float value = head < 12u
            ? qa_bf16(queries[(query * 24u + kv_head * 12u + head) * D + dimension]) * 0.0625f
            : 0.0f;
        const half staged = (half)value;
        if (!metal::isfinite(value) || !metal::isfinite(float(staged)))
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        shared.scaled_queries[i] = staged;
    }
    if (tid < 64u) shared.identity[tid] = (tid >> 3u) == (tid & 7u) ? 1.0f : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 identity_matrix;
    simdgroup_load(identity_matrix, shared.identity, 8, 0, false);
    simdgroup_float8x8 output[16];
    for (uint j = 0u; j < 16u; ++j)
        output[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float maximum = -3.0e38f, denominator = 0.0f;

    for (uint first = 0u; first < n; first += key_tile) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < key_tile) {
            const uint slot = first + tid;
            int position = -1;
            if (slot < n && valid[query * selected + slot]) {
                if constexpr (indexed) {
                    position = logical_ids[query * selected + slot];
                    const uint visible = geometry[0] + geometry[1] + query + 1u;
                    if (position < 0 || uint(position) >= visible || uint(position) >= geometry[2]) {
                        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
                        position = -1;
                    }
                } else {
                    position = int(slot);
                }
            }
            shared.key_positions[tid] = position;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint key = tid >> 3u, segment = tid & 7u;
        const int slot = shared.key_positions[key];
        for (uint word = 0u; word < 32u; ++word) {
            const uint dimension = segment * 32u + word;
            const uint source = indexed
                ? (uint(max(slot, 0)) * 2u + kv_head) * D + dimension
                : ((query * 2u + kv_head) * selected + uint(max(slot, 0))) * D + dimension;
            const float k = slot >= 0 ? qa_bf16(gathered_keys[source]) : 0.0f;
            const float v = slot >= 0 ? qa_bf16(gathered_values[source]) : 0.0f;
            const half kh = (half)k, vh = (half)v;
            if (!metal::isfinite(k) || !metal::isfinite(v)
                || !metal::isfinite(float(kh)) || !metal::isfinite(float(vh)))
                atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
            staged_keys[key * D + dimension] = kh;
            staged_values[key * D + dimension] = vh;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 scores[4];
        for (uint i = 0u; i < 4u; ++i)
            scores[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k = 0u; k < 8u; ++k) {
            for (uint i = 0u; i < 4u; ++i) {
                simdgroup_half8x8 q, key_matrix;
                simdgroup_load(q, shared.scaled_queries + row_tile * 8u * D + (k * 4u + i) * 8u,
                    D, 0, false);
                simdgroup_load(key_matrix,
                    staged_keys + dimension_half * 8u * D + (k * 4u + i) * 8u,
                    D, 0, true);
                simdgroup_multiply_accumulate(scores[i], q, key_matrix, scores[i]);
            }
        }
        for (uint i = 1u; i < 4u; ++i)
            simdgroup_multiply_accumulate(scores[0], identity_matrix, scores[i], scores[0]);
        simdgroup_store(scores[0], shared.score_tile[row_tile][dimension_half], 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float probability[4];
        bool active[4];
        float tile_max = -3.0e38f;
        for (uint c = 0u; c < 4u; ++c) {
            const uint col = local_column + c;
            active[c] = shared.key_positions[col] >= 0;
            const float value = active[c]
                ? shared.score_tile[row_tile][col >> 3u][local_row * 8u + (col & 7u)]
                : -3.0e38f;
            if (!metal::isfinite(value)) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
            probability[c] = value;
            tile_max = metal::max(tile_max, value);
        }
        tile_max = metal::max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = metal::max(tile_max, simd_shuffle_xor(tile_max, 2));
        const float next_maximum = metal::max(maximum, tile_max);
        const float correction = exp(maximum - next_maximum);
        float row_sum = 0.0f;
        for (uint c = 0u; c < 4u; ++c) {
            const float weight = active[c] ? exp(probability[c] - next_maximum) : 0.0f;
            row_sum += weight;
            shared.probabilities[simd_group][local_row * 16u + local_column + c] = (half)weight;
        }
        row_sum += simd_shuffle_xor(row_sum, 1);
        row_sum += simd_shuffle_xor(row_sum, 2);
        denominator = denominator * correction + row_sum;
        maximum = next_maximum;
        const bool rescale = simd_any(correction != 1.0f);
        if (rescale) {
            for (uint i = lane; i < 64u; i += 32u)
                shared.diagonals[simd_group][i] = (i >> 3u) == (i & 7u)
                    ? simd_shuffle(correction, (ushort)((i >> 3u) * 4u)) : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (rescale) {
            simdgroup_float8x8 diagonal;
            simdgroup_load(diagonal, shared.diagonals[simd_group], 8, 0, false);
            for (uint j = 0u; j < 16u; ++j) {
                simdgroup_float8x8 scaled;
                simdgroup_multiply(scaled, diagonal, output[j]);
                output[j] = scaled;
            }
        }
        simdgroup_half8x8 p0, p1;
        simdgroup_load(p0, shared.probabilities[simd_group], 16, 0, false);
        simdgroup_load(p1, shared.probabilities[simd_group] + 8, 16, 0, false);
        for (uint j = 0u; j < 16u; ++j) {
            simdgroup_half8x8 v0, v1;
            simdgroup_load(v0, staged_values + dimension_half * 128u + j * 8u,
                D, 0, false);
            simdgroup_load(v1, staged_values + 8u * D + dimension_half * 128u + j * 8u,
                D, 0, false);
            simdgroup_multiply_accumulate(output[j], p0, v0, output[j]);
            simdgroup_multiply_accumulate(output[j], p1, v1, output[j]);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (uint i = lane; i < 64u; i += 32u)
        shared.diagonals[simd_group][i] = (i >> 3u) == (i & 7u)
            ? simd_shuffle(inverse, (ushort)((i >> 3u) * 4u)) : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 diagonal;
    simdgroup_load(diagonal, shared.diagonals[simd_group], 8, 0, false);
    threadgroup float* scaled_output = (threadgroup float*)shared.kv
        + row_tile * 8u * D + dimension_half * 128u;
    for (uint j = 0u; j < 16u; ++j) {
        simdgroup_float8x8 value;
        simdgroup_multiply(value, diagonal, output[j]);
        simdgroup_store(value, scaled_output + j * 8u, D, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 16u * D; i += 128u) {
        const uint head = i / D, dimension = i % D;
        if (head >= 12u) continue;
        const uint output_index = (query * 24u + kv_head * 12u + head) * D + dimension;
        const float attended = ((threadgroup float*)shared.kv)[head * D + dimension];
        if (DeferredGate) {
            if (!metal::isfinite(attended)) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
            attended_output[output_index] = attended;
            continue;
        }
        const float gate = qa_bf16(gates[output_index]);
        const float logistic = 1.0f / (1.0f + exp(-gate));
        if (rows[1] != 0u) {
            attention[output_index] = qa_round(attended);
            sigmoid[output_index] = qa_round(logistic);
        }
        gated[output_index] = qsa_finish_gate(attended, logistic, status);
    }
}

kernel void qsa_prefill_online_mm(
    const device ushort* queries [[buffer(0)]],
    const device ushort* gates [[buffer(1)]],
    const device ushort* gathered_keys [[buffer(2)]],
    const device ushort* gathered_values [[buffer(3)]],
    const device uchar* valid [[buffer(4)]],
    const device uint* selected_count [[buffer(5)]],
    device ushort* attention [[buffer(6)]],
    device ushort* sigmoid [[buffer(7)]],
    device ushort* gated [[buffer(8)]],
    device atomic_uint* status [[buffer(9)]],
    const constant uint* rows [[buffer(10)]],
    uint group_index [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup QsaPrefillShared shared;
    qsa_prefill_online_body<false>(queries, gates, gathered_keys, gathered_values,
        valid, selected_count, attention, sigmoid, gated, status, rows,
        group_index, tid, simd_group, lane, nullptr, nullptr, shared);
}

kernel void qsa_prefill_indexed_mm(
    const device ushort* queries [[buffer(0)]],
    const device ushort* gates [[buffer(1)]],
    const device ushort* gathered_keys [[buffer(2)]],
    const device ushort* gathered_values [[buffer(3)]],
    const device uchar* valid [[buffer(4)]],
    const device uint* selected_count [[buffer(5)]],
    device ushort* attention [[buffer(6)]],
    device ushort* sigmoid [[buffer(7)]],
    device ushort* gated [[buffer(8)]],
    device atomic_uint* status [[buffer(9)]],
    const constant uint* rows [[buffer(10)]],
    const device int* logical_ids [[buffer(11)]],
    const constant uint* geometry [[buffer(12)]],
    uint group_index [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup QsaPrefillShared shared;
    qsa_prefill_online_body<true>(queries, gates, gathered_keys, gathered_values,
        valid, selected_count, attention, sigmoid, gated, status, rows,
        group_index, tid, simd_group, lane, logical_ids, geometry, shared);
}

kernel void qsa_prefill_indexed_deferred(
    const device ushort* q [[buffer(0)]], const device ushort* unused_gate [[buffer(1)]],
    const device ushort* k [[buffer(2)]], const device ushort* v [[buffer(3)]],
    const device uchar* valid [[buffer(4)]], const device uint* count [[buffer(5)]],
    device ushort* unused_a [[buffer(6)]], device ushort* unused_s [[buffer(7)]],
    device float* attended [[buffer(8)]], device atomic_uint* status [[buffer(9)]],
    const constant uint* rows [[buffer(10)]], const device int* ids [[buffer(11)]],
    const constant uint* geometry [[buffer(12)]], uint group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup QsaPrefillShared shared;
    qsa_prefill_online_body<true,true>(q,unused_gate,k,v,valid,count,unused_a,unused_s,
        nullptr,status,rows,group,tid,simd,lane,ids,geometry,shared,attended);
}

kernel void qsa_late_gate(const device float* attended [[buffer(0)]],
    const device ushort* gate [[buffer(1)]], device ushort* gated [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float logistic = 1.0f / (1.0f + exp(-qa_bf16(gate[index])));
    gated[index] = qsa_finish_gate(attended[index], logistic, status);
}
