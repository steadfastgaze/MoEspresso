#include <metal_stdlib>

using namespace metal;

struct GreedyCandidate {
    float value;
    uint id;
};

static inline bool greedy_better(GreedyCandidate candidate, GreedyCandidate current) {
    return candidate.value > current.value ||
        (candidate.value == current.value && candidate.id < current.id);
}

kernel void greedy_presence_bf16(
    const device bfloat* logits [[buffer(0)]],
    const device uint* seen [[buffer(1)]],
    device bfloat* adjusted [[buffer(2)]],
    constant float& penalty [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const bool present = (seen[id >> 5u] & (1u << (id & 31u))) != 0u;
    adjusted[id] = present ? bfloat(float(logits[id]) - penalty) : logits[id];
}

kernel void greedy_logprob_blocks(
    const device bfloat* logits [[buffer(0)]],
    const device bfloat* logsumexp [[buffer(1)]],
    device GreedyCandidate* partials [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
    threadgroup GreedyCandidate candidates[256];
    const float value = id < count
        ? float(bfloat(float(logits[id]) - float(logsumexp[0])))
        : -INFINITY;
    GreedyCandidate initial = {value, id < count ? id : 0xffffffffu};
    candidates[lane] = initial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride && greedy_better(candidates[lane + stride], candidates[lane])) {
            candidates[lane] = candidates[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) partials[block] = candidates[0];
}

kernel void greedy_logprob_finish(
    const device GreedyCandidate* partials [[buffer(0)]],
    device uint* selected [[buffer(1)]],
    constant uint& blocks [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup GreedyCandidate candidates[256];
    GreedyCandidate best = {-INFINITY, 0xffffffffu};
    for (uint block = lane; block < blocks; block += 256u) {
        const GreedyCandidate candidate = partials[block];
        if (greedy_better(candidate, best)) best = candidate;
    }
    candidates[lane] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride && greedy_better(candidates[lane + stride], candidates[lane])) {
            candidates[lane] = candidates[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) selected[0] = candidates[0].id;
}
