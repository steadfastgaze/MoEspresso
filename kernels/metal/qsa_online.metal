// Online QSA decode adapted from DS4 qwen4_attn_decode_tile<8> and its
// SIMD merge (MIT). The selected K/V rows are head-major BF16 KVarN output.
#include <metal_stdlib>
using namespace metal;

#define QSA_HEADS 24u
#define QSA_KV_HEADS 2u
#define QSA_HEAD_WIDTH 256u
#define QSA_SELECTED 2051u
#define QSA_SPLIT_KEYS 32u
#define QSA_MAX_SPLITS 64u
#define QSA_HEADS_PER_SIMD 3u
#define QSA_PART_STRIDE (2u + QSA_HEAD_WIDTH)

inline float qsa_online_float(ushort bits) {
    return as_type<float>(uint(bits) << 16u);
}

inline ushort qsa_online_word(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

inline float qsa_online_sigmoid(float x) {
    if (x >= 0.0f) {
        float e = exp(-x);
        return 1.0f / (1.0f + e);
    }
    float e = exp(x);
    return e / (1.0f + e);
}

inline uint qsa_online_splits(uint n) {
    return min(max((n + QSA_SPLIT_KEYS - 1u) / QSA_SPLIT_KEYS, 1u), QSA_MAX_SPLITS);
}

// One 128-thread group handles one key split and KV head. Its four SIMD
// groups each own three query heads, with eight dimensions per lane.
kernel void qsa_online_tile(
    const device ushort* queries [[buffer(0)]],
    const device ushort* keys [[buffer(1)]],
    const device ushort* values [[buffer(2)]],
    const device uchar* valid [[buffer(3)]],
    const device uint* selected_count [[buffer(4)]],
    device float* partials [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint split = group / QSA_KV_HEADS;
    const uint kv_head = group % QSA_KV_HEADS;
    if (split >= QSA_MAX_SPLITS) return;
    const uint n = min(selected_count[0], QSA_SELECTED);
    const uint splits = qsa_online_splits(n);
    if (split >= splits) return;
    const uint keys_per_split = (n + splits - 1u) / splits;
    const uint begin = split * keys_per_split;
    const uint end = min(n, begin + keys_per_split);

    float query[QSA_HEADS_PER_SIMD][8];
    float maximum[QSA_HEADS_PER_SIMD];
    float denominator[QSA_HEADS_PER_SIMD];
    float accumulator[QSA_HEADS_PER_SIMD][8];
    for (uint j = 0; j < QSA_HEADS_PER_SIMD; ++j) {
        const uint head = kv_head * 12u + simd * QSA_HEADS_PER_SIMD + j;
        for (uint d = 0; d < 8u; ++d) {
            query[j][d] = qsa_online_float(queries[head * QSA_HEAD_WIDTH + lane * 8u + d]) * 0.0625f;
            accumulator[j][d] = 0.0f;
        }
        maximum[j] = -3.0e38f;
        denominator[j] = 0.0f;
    }
    for (uint row = begin; row < end; ++row) {
        if (valid[row] == 0u) continue;
        const uint at = (kv_head * QSA_SELECTED + row) * QSA_HEAD_WIDTH + lane * 8u;
        float key[8], value[8];
        for (uint d = 0; d < 8u; ++d) {
            key[d] = qsa_online_float(keys[at + d]);
            value[d] = qsa_online_float(values[at + d]);
        }
        for (uint j = 0; j < QSA_HEADS_PER_SIMD; ++j) {
            float score = 0.0f;
            for (uint d = 0; d < 8u; ++d) score += query[j][d] * key[d];
            score = simd_sum(score);
            const float next_maximum = max(maximum[j], score);
            const float correction = exp(maximum[j] - next_maximum);
            const float weight = exp(score - next_maximum);
            denominator[j] = denominator[j] * correction + weight;
            for (uint d = 0; d < 8u; ++d) {
                accumulator[j][d] = accumulator[j][d] * correction + weight * value[d];
            }
            maximum[j] = next_maximum;
        }
    }
    for (uint j = 0; j < QSA_HEADS_PER_SIMD; ++j) {
        const uint head = kv_head * 12u + simd * QSA_HEADS_PER_SIMD + j;
        const uint base = (head * QSA_MAX_SPLITS + split) * QSA_PART_STRIDE;
        if (lane == 0u) {
            partials[base] = maximum[j];
            partials[base + 1u] = denominator[j];
        }
        for (uint d = 0; d < 8u; ++d) {
            partials[base + 2u + lane * 8u + d] = accumulator[j][d];
        }
    }
}

// Consume committed KVarN storage and one pending row directly. The inverse
// transform stages one selected row in threadgroup storage for all twelve
// query heads. Online attention accumulators remain FP32; sigmoid gating
// multiplies the FP32 result before the BF16 store.
kernel void qsa_kvarn_online_tile(
    const device ushort* queries [[buffer(0)]],
    const device uchar* records [[buffer(1)]],
    const device ushort* sink_keys [[buffer(2)]],
    const device ushort* sink_values [[buffer(3)]],
    const device ushort* tail_keys [[buffer(4)]],
    const device ushort* tail_values [[buffer(5)]],
    const device ushort* pending_keys [[buffer(6)]],
    const device ushort* pending_values [[buffer(7)]],
    const device int* logical_ids [[buffer(8)]],
    const device uchar* valid [[buffer(9)]],
    const device uint* selected_count [[buffer(10)]],
    const constant uint* geometry [[buffer(11)]],
    device float* partials [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint split = group / QSA_KV_HEADS;
    const uint kv_head = group % QSA_KV_HEADS;
    const uint n = min(selected_count[0], QSA_SELECTED);
    const uint splits = qsa_online_splits(n);
    if (split >= splits) return;
    const uint keys_per_split = (n + splits - 1u) / splits;
    const uint begin = split * keys_per_split;
    const uint end = min(n, begin + keys_per_split);
    const uint frontier = geometry[0];
    const uint body_frontier = 128u + geometry[1] * 128u;
    threadgroup float staged[512];
    float query[QSA_HEADS_PER_SIMD][8];
    float accumulator[QSA_HEADS_PER_SIMD][8];
    float maximum[QSA_HEADS_PER_SIMD], denominator[QSA_HEADS_PER_SIMD];
    for (uint h = 0u; h < QSA_HEADS_PER_SIMD; ++h) {
        const uint head = kv_head * 12u + simd * QSA_HEADS_PER_SIMD + h;
        for (uint d = 0u; d < 8u; ++d) {
            query[h][d] = qsa_online_float(queries[head * 256u + lane * 8u + d]) * 0.0625f;
            accumulator[h][d] = 0.0f;
        }
        maximum[h] = -3.0e38f;
        denominator[h] = 0.0f;
    }
    for (uint row = begin; row < end; ++row) {
        if (valid[row] == 0u) continue;
        const uint logical = uint(logical_ids[row]);
        float2 key, value;
        const uint channel = tid * 2u;
        if (logical >= frontier) {
            const uint at = kv_head * 256u + channel;
            key = float2(qsa_online_float(pending_keys[at]), qsa_online_float(pending_keys[at + 1u]));
            value = float2(qsa_online_float(pending_values[at]), qsa_online_float(pending_values[at + 1u]));
        } else if (logical < 128u) {
            const uint at = (kv_head * 128u + logical) * 256u + channel;
            key = float2(qsa_online_float(sink_keys[at]), qsa_online_float(sink_keys[at + 1u]));
            value = float2(qsa_online_float(sink_values[at]), qsa_online_float(sink_values[at + 1u]));
        } else if (logical >= body_frontier) {
            const uint at = (kv_head * 8320u + (logical - 128u) % 8320u) * 256u + channel;
            key = float2(qsa_online_float(tail_keys[at]), qsa_online_float(tail_keys[at + 1u]));
            value = float2(qsa_online_float(tail_values[at]), qsa_online_float(tail_values[at + 1u]));
        } else {
            const uint local = logical - 128u;
            const uint tile = local / 128u, token = local % 128u;
            const device uchar* record = records + (tile * 2u + kv_head) * 35072u;
            const uchar k_codes = record[token * 128u + tid];
            const uchar v_codes = record[17664u + token * 128u + tid];
            const device half* k_scale = (const device half*)(record + 16384u);
            const device half* k_zero = (const device half*)(record + 16896u);
            const device half* k_token_scale = (const device half*)(record + 17408u);
            const device half* v_scale = (const device half*)(record + 34048u);
            const device half* v_token_scale = (const device half*)(record + 34560u);
            const device half* v_zero = (const device half*)(record + 34816u);
            for (uint d = 0u; d < 2u; ++d) {
                const float kc = float((k_codes >> (d * 4u)) & 15u);
                const float vc = float((v_codes >> (d * 4u)) & 15u);
                key[d] = (kc * float(k_scale[channel + d]) + float(k_zero[channel + d]))
                    * float(k_token_scale[token]);
                value[d] = (vc * float(v_token_scale[token]) + float(v_zero[token]))
                    * float(v_scale[channel + d]);
            }
            key = float2(key.x + key.y, key.x - key.y);
            value = float2(value.x + value.y, value.x - value.y);
            for (uint stride = 1u; stride < 32u; stride <<= 1u) {
                const float2 k = simd_shuffle_xor(key, stride);
                const float2 v = simd_shuffle_xor(value, stride);
                key = (tid & stride) ? k - key : key + k;
                value = (tid & stride) ? v - value : value + v;
            }
            staged[channel] = key.x;
            staged[channel + 1u] = key.y;
            staged[256u + channel] = value.x;
            staged[257u + channel] = value.y;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 64u; stride <= 128u; stride <<= 1u) {
                const uint other = channel ^ stride;
                const float2 k(staged[other], staged[other + 1u]);
                const float2 v(staged[256u + other], staged[257u + other]);
                key = (channel & stride) ? k - key : key + k;
                value = (channel & stride) ? v - value : value + v;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (stride < 128u) {
                    staged[channel] = key.x;
                    staged[channel + 1u] = key.y;
                    staged[256u + channel] = value.x;
                    staged[257u + channel] = value.y;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            key = float2(qsa_online_float(qsa_online_word(key.x * 0.0625f)),
                qsa_online_float(qsa_online_word(key.y * 0.0625f)));
            value = float2(qsa_online_float(qsa_online_word(value.x * 0.0625f)),
                qsa_online_float(qsa_online_word(value.y * 0.0625f)));
        }
        staged[channel] = key.x;
        staged[channel + 1u] = key.y;
        staged[256u + channel] = value.x;
        staged[257u + channel] = value.y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float k[8], v[8];
        for (uint d = 0u; d < 8u; ++d) {
            k[d] = staged[lane * 8u + d];
            v[d] = staged[256u + lane * 8u + d];
        }
        for (uint h = 0u; h < QSA_HEADS_PER_SIMD; ++h) {
            float score = 0.0f;
            for (uint d = 0u; d < 8u; ++d) score += query[h][d] * k[d];
            score = simd_sum(score);
            const float next_maximum = max(maximum[h], score);
            const float correction = exp(maximum[h] - next_maximum);
            const float weight = exp(score - next_maximum);
            denominator[h] = denominator[h] * correction + weight;
            for (uint d = 0u; d < 8u; ++d) {
                accumulator[h][d] = accumulator[h][d] * correction + weight * v[d];
            }
            maximum[h] = next_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint h = 0u; h < QSA_HEADS_PER_SIMD; ++h) {
        const uint head = kv_head * 12u + simd * QSA_HEADS_PER_SIMD + h;
        const uint base = (head * QSA_MAX_SPLITS + split) * QSA_PART_STRIDE;
        if (lane == 0u) {
            partials[base] = maximum[h];
            partials[base + 1u] = denominator[h];
        }
        for (uint d = 0u; d < 8u; ++d) {
            partials[base + 2u + lane * 8u + d] = accumulator[h][d];
        }
    }
}

// Ordinary SIMD merge across the active key splits. One BF16 output boundary
// follows the FP32 attention and gate arithmetic.
kernel void qsa_online_merge(
    const device float* partials [[buffer(0)]],
    const device ushort* gate [[buffer(1)]],
    const device uint* selected_count [[buffer(2)]],
    device ushort* gated [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (head >= QSA_HEADS) return;
    const uint n = min(selected_count[0], QSA_SELECTED);
    const uint splits = qsa_online_splits(n);
    float maximum = -3.0e38f;
    for (uint split = 0; split < splits; ++split) {
        const uint base = (head * QSA_MAX_SPLITS + split) * QSA_PART_STRIDE;
        maximum = max(maximum, partials[base]);
    }
    float denominator = 0.0f;
    float output[8];
    for (uint d = 0; d < 8u; ++d) output[d] = 0.0f;
    for (uint split = 0; split < splits; ++split) {
        const uint base = (head * QSA_MAX_SPLITS + split) * QSA_PART_STRIDE;
        const float mass = partials[base + 1u];
        const float correction = mass > 0.0f ? exp(partials[base] - maximum) : 0.0f;
        denominator += mass * correction;
        for (uint d = 0; d < 8u; ++d) {
            output[d] += partials[base + 2u + lane * 8u + d] * correction;
        }
    }
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (uint d = 0; d < 8u; ++d) {
        const uint index = head * QSA_HEAD_WIDTH + lane * 8u + d;
        const float mixed = output[d] * inverse * qsa_online_sigmoid(qsa_online_float(gate[index]));
        gated[index] = qsa_online_word(mixed);
    }
}
