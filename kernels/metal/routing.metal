#include <metal_stdlib>
using namespace metal;

inline float routing_bf16_to_float(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

inline ushort routing_float_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Selection uses the larger ID at a top-ten cutoff. Integer ranks preserve
// float ordering while treating all NaNs alike and both zeros equally.
// See THIRD-PARTY-NOTICES for source attribution.
inline uint route_rank(float value) {
    if (metal::isnan(value)) return 0xffffffffu;
    uint bits = as_type<uint>(value == 0.0f ? 0.0f : value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

inline uint2 route_simd_max(uint rank, uint id) {
    uint top = simd_max(rank);
    return uint2(top, simd_max(rank == top ? id : 0u));
}

inline void route_top10(threadgroup float* keys, threadgroup uint* chosen,
                        threadgroup uint* list_ranks,
                        threadgroup uint* list_ids,
                        uint tid, uint sg, uint lane) {
    uint ranks[4], ids[4];
    for (uint j = 0; j < 4; ++j) {
        ids[j] = tid * 4 + j;
        ranks[j] = route_rank(keys[ids[j]]);
    }
    for (uint a = 0; a < 3; ++a) {
        for (uint b = 0; b < 3 - a; ++b) {
            bool swap = ranks[b + 1] > ranks[b]
                || (ranks[b + 1] == ranks[b] && ids[b + 1] > ids[b]);
            if (swap) {
                uint r = ranks[b]; ranks[b] = ranks[b + 1]; ranks[b + 1] = r;
                uint i = ids[b]; ids[b] = ids[b + 1]; ids[b + 1] = i;
            }
        }
    }
    // Each SIMD group produces its own top ten without a threadgroup barrier.
    uint head = 0;
    for (uint route = 0; route < 10; ++route) {
        uint rank = head < 4 ? ranks[head] : 0u;
        uint id = head < 4 ? ids[head] : 0u;
        uint2 best = route_simd_max(rank, id);
        if (head < 4 && rank == best.x && id == best.y) ++head;
        if (lane == 0) {
            list_ranks[sg * 10 + route] = best.x;
            list_ids[sg * 10 + route] = best.y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint position = 0;
        for (uint route = 0; route < 10; ++route) {
            bool live = lane < 4 && position < 10;
            uint rank = live ? list_ranks[lane * 10 + position] : 0u;
            uint id = live ? list_ids[lane * 10 + position] : 0u;
            uint2 best = route_simd_max(rank, id);
            if (live && rank == best.x && id == best.y) ++position;
            if (lane == 0) chosen[route] = best.y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// An equal probability in the selected set is emitted by smaller ID first.
inline void route_sort(threadgroup float* probabilities,
                       threadgroup uint* chosen, threadgroup uint* ranked,
                       uint tid) {
    if (tid < 10) {
        uint id = chosen[tid], rank = 0;
        float score = probabilities[id];
        for (uint j = 0; j < 10; ++j) {
            uint other_id = chosen[j];
            float other = probabilities[other_id];
            bool sn = metal::isnan(score), on = metal::isnan(other);
            bool before = (!sn && on) || (sn == on &&
                (other > score || ((sn || other == score) && other_id < id)));
            rank += uint(before);
        }
        ranked[rank] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void cache_prior_route_512(
    device const ushort* logits [[buffer(0)]],
    device const uint* gate_slots [[buffer(1)]],
    device const uint* up_slots [[buffer(2)]],
    device const uint* down_slots [[buffer(3)]],
    device const float* cache_factor [[buffer(4)]],
    device const uint* protected_routes [[buffer(5)]],
    device const uint* unbiased [[buffer(6)]],
    device uint* output_ids [[buffer(7)]],
    device ushort* output_scores [[buffer(8)]],
    device uint* changed [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    threadgroup float probabilities[512], keys[512];
    threadgroup float maxima[32], sums[32];
    threadgroup uint list_ranks[40];
    threadgroup uint list_ids[40];
    threadgroup uint invalid[4];
    threadgroup uint chosen[10], original[10], ranked[10];
    threadgroup uint apply_bias;
    threadgroup float denominator;
    float values[4];

    if (sg == 0) { maxima[lane] = -INFINITY; sums[lane] = 0.0f; }
    for (uint j = 0; j < 4; ++j) values[j] = routing_bf16_to_float(logits[row * 512 + tid * 4 + j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_max = -INFINITY;
    uint bad = 0;
    for (uint j = 0; j < 4; ++j) {
        local_max = local_max < values[j] ? values[j] : local_max;
        bad |= uint(!metal::isfinite(values[j]));
    }
    local_max = simd_max(local_max);
    bad = simd_max(bad);
    if (lane == 0) { maxima[sg] = local_max; invalid[sg] = bad; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float maximum = simd_max(maxima[lane]);
        if (lane == 0) maxima[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        values[j] = fast::exp(values[j] - maxima[0]);
        local_sum += values[j];
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) sums[sg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float total = simd_sum(sums[lane]);
        if (lane == 0) sums[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = 1.0f / sums[0];
    for (uint j = 0; j < 4; ++j) {
        uint id = tid * 4 + j;
        probabilities[id] = values[j] * inverse;
        keys[id] = probabilities[id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    route_top10(keys, chosen, list_ranks, list_ids, tid, sg, lane);
    route_sort(probabilities, chosen, original, tid);

    if (tid == 0) {
        apply_bias = 0;
        if (!unbiased[0] && cache_factor[0] > 1.0f &&
            !(invalid[0] | invalid[1] | invalid[2] | invalid[3])) {
            for (uint j = protected_routes[0]; j < 10; ++j) {
                uint id = original[j];
                if (gate_slots[id] >= 512 || up_slots[id] >= 512 || down_slots[id] >= 512)
                    apply_bias = 1;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (apply_bias) {
        for (uint j = 0; j < 4; ++j) {
            uint id = tid * 4 + j;
            bool hot = gate_slots[id] < 512 && up_slots[id] < 512 && down_slots[id] < 512;
            bool protect = false;
            for (uint k = 0; k < protected_routes[0]; ++k) protect |= id == original[k];
            keys[id] = protect ? INFINITY
                : probabilities[id] * (hot ? cache_factor[0] : 1.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        route_top10(keys, chosen, list_ranks, list_ids, tid, sg, lane);
        route_sort(probabilities, chosen, ranked, tid);
    } else {
        if (tid < 10) ranked[tid] = original[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float sum = 0.0f;
        uint replacements = 0;
        for (uint j = 0; j < 10; ++j) {
            sum = probabilities[ranked[j]] + sum;
            bool present = false;
            for (uint k = 0; k < 10; ++k) present |= ranked[j] == original[k];
            replacements += uint(!present);
        }
        denominator = sum;
        changed[row] = replacements;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 10) {
        output_ids[row * 10 + tid] = ranked[tid];
        output_scores[row * 10 + tid] = routing_float_to_bf16(probabilities[ranked[tid]] / denominator);
    }
}
