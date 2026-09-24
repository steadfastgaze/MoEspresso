"""Fused cache-prior selection with original FP32-softmax contribution weights.

Slot maps are immutable published snapshots; 512 denotes a missing projection.
The caller owns snapshot freshness and request lifetime.
"""


from functools import cache
import math

import mlx.core as mx

from moespresso.runtime.qwen4.cache_routing_config import (
    DEFAULT_CACHE_BONUS as DEFAULT_CACHE_BONUS,
    DEFAULT_PROTECTED_ROUTES,
    validate_cache_factor,
    validate_protected_routes,
)

_HEADER = r"""
inline bool route_better(float a, uint ai, float b, uint bi) {
    bool an = metal::isnan(a), bn = metal::isnan(b);
    return (an && !bn) || (an == bn &&
        ((an && ai > bi) || (!an && (a > b || (a == b && ai > bi)))));
}

inline void route_top10(threadgroup float* keys, threadgroup uint* chosen,
                        threadgroup float* partial_values,
                        threadgroup uint* partial_ids,
                        uint tid, uint sg, uint lane) {
    float values[4];
    bool alive[4] = {true, true, true, true};
    for (uint j = 0; j < 4; ++j) values[j] = keys[tid * 4 + j];
    for (uint route = 0; route < 10; ++route) {
        float best = -INFINITY;
        uint best_id = 0;
        bool found = false;
        for (uint j = 0; j < 4; ++j) {
            uint id = tid * 4 + j;
            if (alive[j] && (!found || route_better(values[j], id, best, best_id))) {
                best = values[j]; best_id = id; found = true;
            }
        }
        uint has_nan = simd_max(uint(found && metal::isnan(best)));
        float value = simd_max(found && !metal::isnan(best) ? best : -INFINITY);
        uint id = simd_max(found && (has_nan ? metal::isnan(best) : best == value)
                          ? best_id : 0u);
        if (lane == 0) {
            partial_values[sg] = keys[id];
            partial_ids[sg] = id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            bool valid = lane < 4;
            float part = valid ? partial_values[lane] : -INFINITY;
            uint nan_group = simd_max(uint(valid && metal::isnan(part)));
            float winner = simd_max(valid && !metal::isnan(part) ? part : -INFINITY);
            uint winner_id = simd_max(valid && (nan_group ? metal::isnan(part) : part == winner)
                                     ? partial_ids[lane] : 0u);
            if (lane == 0) chosen[route] = winner_id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 4; ++j) {
            if (tid * 4 + j == chosen[route]) alive[j] = false;
        }
    }
}

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
"""

_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.x;
    threadgroup float probabilities[512], keys[512];
    threadgroup float maxima[32], sums[32];
    threadgroup float partial_values[4];
    threadgroup uint partial_ids[4];
    threadgroup uint invalid[4];
    threadgroup uint chosen[10], original[10], ranked[10];
    threadgroup uint apply_bias;
    threadgroup float denominator;
    float values[4];

    if (sg == 0) { maxima[lane] = Limits<float>::min; sums[lane] = 0.0f; }
    for (uint j = 0; j < 4; ++j) values[j] = float(logits[row * 512 + tid * 4 + j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_max = Limits<float>::finite_min;
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
    route_top10(keys, chosen, partial_values, partial_ids, tid, sg, lane);
    route_sort(probabilities, chosen, original, tid);

    if (tid == 0) {
        apply_bias = 0;
        if (cache_factor[0] > 1.0f && !(invalid[0] | invalid[1] | invalid[2] | invalid[3])) {
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
        route_top10(keys, chosen, partial_values, partial_ids, tid, sg, lane);
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
        output_scores[row * 10 + tid] = static_cast<bfloat16_t>(probabilities[ranked[tid]] / denominator);
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="moespresso_cache_prior_router",
        input_names=["logits", "gate_slots", "up_slots", "down_slots", "cache_factor", "protected_routes"],
        output_names=["output_ids", "output_scores", "changed"],
        header=_HEADER, source=_SOURCE,
    )


@cache
def _bonus(value):
    try:
        factor = validate_cache_factor(math.exp(value))
    except (OverflowError, ValueError) as exc:
        raise ValueError("cache bonus must produce a finite multiplier from 1 to 8") from exc
    return mx.array([factor], dtype=mx.float32)


@cache
def _protected_routes(value):
    return mx.array([value], dtype=mx.uint32)


def cache_prior_route(
    logits, slot_maps, *, bonus=DEFAULT_CACHE_BONUS, protected_routes=DEFAULT_PROTECTED_ROUTES,
):
    """Route independent rows against the same immutable three-map snapshot.

    Live integration uses one-row decode. Multirow inputs are supported only
    as independent stateless test rows; this does not model evolving residency
    during a prefill chunk. Ranking multiplies original FP32 probabilities by
    exp(bonus), preserving their underflow and tie behavior. The strongest
    protected_routes original routes remain selected. The multiplier must be
    between one and eight; the number of protected routes is bounded by configuration.
    """
    if type(bonus) not in (float, int) or bonus < 0:
        raise ValueError("cache bonus must be finite and nonnegative")
    factor = _bonus(bonus)
    validate_protected_routes(protected_routes)
    if (logits.ndim < 2 or logits.shape[-1] != 512 or logits.dtype != mx.bfloat16
            or logits.size == 0 or len(slot_maps) != 3
            or any(x.shape != (512,) or x.dtype != mx.uint32 for x in slot_maps)):
        raise ValueError("requires BF16 router rows and three uint32 expert-slot maps")
    rows = logits.size // 512
    shape = (*logits.shape[:-1], 10)
    return _kernel()(
        inputs=[mx.contiguous(logits), *slot_maps, factor, _protected_routes(protected_routes)],
        output_shapes=[shape, shape, (rows,)],
        output_dtypes=[mx.uint32, mx.bfloat16, mx.uint32],
        grid=(rows * 128, 1, 1), threadgroup=(128, 1, 1),
    )
