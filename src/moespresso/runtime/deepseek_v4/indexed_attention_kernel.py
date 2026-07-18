"""DS4 ratio-4 indexed mixed-attention probe kernel.

This module is intentionally not wired into serving yet. It provides a
DS4-c-shaped one-token indexed compressed-attention consumer so speed work can
measure the real candidate surface before replacing the served MLX composition.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np


_KERNEL_CACHE: dict[tuple[int, int], object] = {}
_PREFILL_KERNEL_CACHE: dict[tuple[int, int], object] = {}
_PREFILL_LIVE_F16_KERNEL_CACHE: dict[tuple[int, int], object] = {}
_PREFILL_LIVE_F32_KERNEL_CACHE: dict[tuple[int, int], object] = {}
_PREFILL_LIVE_V2_F16_KERNEL_CACHE: dict[tuple[int, int], object] = {}
_PREFILL_LIVE_V2_F32_KERNEL_CACHE: dict[tuple[int, int], object] = {}
_PREFILL_LIVE_MMA_KERNEL_CACHE: dict[tuple[int, int], object] = {}
_INDEXER_SCORE_KERNEL_CACHE: dict[tuple[int, int, str], object] = {}
_INDEXER_Q_QAT_KERNEL_CACHE: dict[tuple[int, int], object] = {}
_INDEXER_Q_QAT_V2_KERNEL_CACHE: dict[tuple[int, int], object] = {}

# Served prefill consumer engagement counts by kernel variant, exported
# through `ssd_streaming_stats` and the speed-stats count keys so served
# A/B arms can prove which consumer ran.
_CONSUMER_CALL_COUNTS = {"mma": 0, "v2": 0, "v1": 0}

# Tiled indexer score kernel engagement counts by operand dtype, exported
# the same way so served A/B arms can prove which operand form ran.
_SCORES_TILED_CALL_COUNTS = {"f16": 0, "f32": 0}


def prefill_consumer_call_counts() -> dict[str, int]:
    """Return prefill consumer engagement counts by kernel variant."""
    return dict(_CONSUMER_CALL_COUNTS)


def indexer_scores_call_counts() -> dict[str, int]:
    """Return tiled score kernel engagement counts by operand dtype."""
    return dict(_SCORES_TILED_CALL_COUNTS)


def indexer_score_operands(q, index_comp):
    """Return the tiled score kernel operand pair in float32.

    Both served prefill and the ratio-4 replay route their score-kernel
    operands through this helper so the two compositions stay identical.
    """
    import mlx.core as mx

    return q.astype(mx.float32), index_comp.astype(mx.float32)

_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint head = thread_position_in_grid.y;

    uint n_head    = meta[0];
    uint head_dim  = meta[1];
    uint n_raw     = meta[2];
    uint raw_cap   = meta[3];
    uint raw_start = meta[4];
    uint n_comp    = meta[5];
    uint top_k     = meta[6];
    uint pos0      = meta[7];
    uint window    = meta[8];
    uint ratio     = meta[9];

    if (lane >= 32u || head >= n_head) return;

    device const float4 *q4 = (device const float4 *)(q + head * head_dim);
    half4 q0 = (half4)q4[lane +  0u];
    half4 q1 = (half4)q4[lane + 32u];
    half4 q2 = (half4)q4[lane + 64u];
    half4 q3 = (half4)q4[lane + 96u];

    float M = -1.701411733e38f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;
    float scale = rsqrt((float)head_dim);

    auto attend_h4_row = [&](const half4 k0,
                             const half4 k1,
                             const half4 k2,
                             const half4 k3) {
        float score = dot((float4)q0, (float4)k0) +
                      dot((float4)q1, (float4)k1) +
                      dot((float4)q2, (float4)k2) +
                      dot((float4)q3, (float4)k3);
        score = simd_sum(score) * scale;

        float old_m = M;
        float new_m = max(M, score);
        float old_scale = exp(old_m - new_m);
        float row_scale = exp(score - new_m);

        S = S * old_scale + row_scale;
        o0 *= old_scale;
        o1 *= old_scale;
        o2 *= old_scale;
        o3 *= old_scale;

        o0 += (float4)k0 * row_scale;
        o1 += (float4)k1 * row_scale;
        o2 += (float4)k2 * row_scale;
        o3 += (float4)k3 * row_scale;
        M = new_m;
    };

    if (n_raw > 0u) {
        uint qpos = pos0;
        uint first_raw_pos = qpos + 1u - n_raw;
        uint raw_last_pos = first_raw_pos + n_raw - 1u;
        uint window_first = (window != 0u && qpos + 1u > window)
            ? qpos + 1u - window
            : 0u;
        uint first = max(first_raw_pos, window_first);
        uint last = min(qpos, raw_last_pos);
        if (first <= last) {
            for (uint pos = first; pos <= last; pos++) {
                uint logical = pos - first_raw_pos;
                uint row = (raw_start + logical) % raw_cap;
                device const half4 *src =
                    (device const half4 *)(raw_kv + row * head_dim);
                attend_h4_row(src[lane + 0u],
                              src[lane + 32u],
                              src[lane + 64u],
                              src[lane + 96u]);
            }
        }
    }

    uint visible = ratio == 0u ? n_comp : (pos0 + 1u) / ratio;
    visible = min(visible, n_comp);
    for (uint i = 0u; i < top_k; i++) {
        int idx = topk[i];
        if (idx < 0) continue;
        if ((uint)idx >= visible) break;
        device const half4 *src =
            (device const half4 *)(comp_kv + ((uint)idx) * head_dim);
        attend_h4_row(src[lane + 0u],
                      src[lane + 32u],
                      src[lane + 64u],
                      src[lane + 96u]);
    }

    float sink_score = sinks[head];
    float old_m = M;
    float new_m = max(M, sink_score);
    float old_scale = exp(old_m - new_m);
    float row_scale = exp(sink_score - new_m);
    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    float inv_s = S == 0.0f ? 0.0f : 1.0f / S;
    device float4 *dst4 = (device float4 *)(out + head * head_dim);
    dst4[lane +  0u] = o0 * inv_s;
    dst4[lane + 32u] = o1 * inv_s;
    dst4[lane + 64u] = o2 * inv_s;
    dst4[lane + 96u] = o3 * inv_s;
"""

_PREFILL_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint sg = thread_position_in_threadgroup.y;
    uint tid = sg * 32u + lane;
    uint token = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y * 8u + sg;

    uint n_tokens  = meta[0];
    uint n_head    = meta[1];
    uint head_dim  = meta[2];
    uint n_raw     = meta[3];
    uint raw_cap   = meta[4];
    uint raw_start = meta[5];
    uint n_comp    = meta[6];
    uint top_k     = meta[7];
    uint pos0      = meta[8];
    uint window    = meta[9];
    uint ratio     = meta[10];

    if (lane >= 32u || sg >= 8u || token >= n_tokens || head >= n_head) return;

    threadgroup half4 kv_shared[128];

    device const float4 *q4 = (device const float4 *)(
        q + ((uint64_t)token * n_head + head) * head_dim);
    half4 q0 = (half4)q4[lane +  0u];
    half4 q1 = (half4)q4[lane + 32u];
    half4 q2 = (half4)q4[lane + 64u];
    half4 q3 = (half4)q4[lane + 96u];

    float M = -1.701411733e38f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;
    float scale = rsqrt((float)head_dim);

    auto attend_shared = [&]() {
        half4 k0 = kv_shared[lane +  0u];
        half4 k1 = kv_shared[lane + 32u];
        half4 k2 = kv_shared[lane + 64u];
        half4 k3 = kv_shared[lane + 96u];
        float score = dot((float4)q0, (float4)k0) +
                      dot((float4)q1, (float4)k1) +
                      dot((float4)q2, (float4)k2) +
                      dot((float4)q3, (float4)k3);
        score = simd_sum(score) * scale;

        float old_m = M;
        float new_m = max(M, score);
        float old_scale = exp(old_m - new_m);
        float row_scale = exp(score - new_m);

        S = S * old_scale + row_scale;
        o0 *= old_scale;
        o1 *= old_scale;
        o2 *= old_scale;
        o3 *= old_scale;

        o0 += (float4)k0 * row_scale;
        o1 += (float4)k1 * row_scale;
        o2 += (float4)k2 * row_scale;
        o3 += (float4)k3 * row_scale;
        M = new_m;
    };

    uint qpos = pos0 + token;
    uint last_pos = pos0 + n_tokens - 1u;
    uint first_raw_pos = last_pos + 1u - n_raw;
    uint raw_last_pos = first_raw_pos + n_raw - 1u;
    uint window_first = (window != 0u && qpos + 1u > window)
        ? qpos + 1u - window
        : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos = first; pos <= last; pos++) {
            uint logical = pos - first_raw_pos;
            uint row = (raw_start + logical) % raw_cap;
            device const half4 *src =
                (device const half4 *)(raw_kv + row * head_dim);
            if (tid < 128u) kv_shared[tid] = src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            attend_shared();
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / ratio;
    visible = min(visible, n_comp);
    device const int32_t *row_topk = topk + (uint64_t)token * top_k;
    for (uint i = 0u; i < top_k; i++) {
        int idx = row_topk[i];
        if (idx < 0) continue;
        if ((uint)idx >= visible) break;
        device const half4 *src =
            (device const half4 *)(comp_kv + ((uint)idx) * head_dim);
        if (tid < 128u) kv_shared[tid] = src[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        attend_shared();
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sink_score = sinks[head];
    float old_m = M;
    float new_m = max(M, sink_score);
    float old_scale = exp(old_m - new_m);
    float row_scale = exp(sink_score - new_m);
    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    float inv_s = S == 0.0f ? 0.0f : 1.0f / S;
    device float4 *dst4 = (device float4 *)(
        out + ((uint64_t)token * n_head + head) * head_dim);
    dst4[lane +  0u] = o0 * inv_s;
    dst4[lane + 32u] = o1 * inv_s;
    dst4[lane + 64u] = o2 * inv_s;
    dst4[lane + 96u] = o3 * inv_s;
"""

_PREFILL_LIVE_F16_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint sg = thread_position_in_threadgroup.y;
    uint tid = sg * 32u + lane;
    uint token = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y * 8u + sg;
    uint batch = threadgroup_position_in_grid.z;

    uint batch_size = meta[0];
    uint n_tokens   = meta[1];
    uint n_head     = meta[2];
    uint head_dim   = meta[3];
    uint n_raw      = meta[4];
    uint raw_cap    = meta[5];
    uint raw_start  = meta[6];
    uint n_comp     = meta[7];
    uint top_k      = meta[8];
    uint pos0       = meta[9];
    uint window     = meta[10];
    uint ratio      = meta[11];

    if (lane >= 32u || sg >= 8u ||
        batch >= batch_size || token >= n_tokens || head >= n_head) return;

    threadgroup half4 kv_shared[128];

    uint64_t q_base =
        (uint64_t)batch * (uint64_t)q_strides[0] +
        (uint64_t)head  * (uint64_t)q_strides[1] +
        (uint64_t)token * (uint64_t)q_strides[2];
    device const half4 *q4 = (device const half4 *)(q + q_base);
    half4 q0 = q4[lane +  0u];
    half4 q1 = q4[lane + 32u];
    half4 q2 = q4[lane + 64u];
    half4 q3 = q4[lane + 96u];

    float M = -1.701411733e38f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;
    float scale = rsqrt((float)head_dim);

    auto attend_shared = [&]() {
        half4 k0 = kv_shared[lane +  0u];
        half4 k1 = kv_shared[lane + 32u];
        half4 k2 = kv_shared[lane + 64u];
        half4 k3 = kv_shared[lane + 96u];
        float score = dot((float4)q0, (float4)k0) +
                      dot((float4)q1, (float4)k1) +
                      dot((float4)q2, (float4)k2) +
                      dot((float4)q3, (float4)k3);
        score = simd_sum(score) * scale;

        float old_m = M;
        float new_m = max(M, score);
        float old_scale = exp(old_m - new_m);
        float row_scale = exp(score - new_m);

        S = S * old_scale + row_scale;
        o0 *= old_scale;
        o1 *= old_scale;
        o2 *= old_scale;
        o3 *= old_scale;

        o0 += (float4)k0 * row_scale;
        o1 += (float4)k1 * row_scale;
        o2 += (float4)k2 * row_scale;
        o3 += (float4)k3 * row_scale;
        M = new_m;
    };

    uint qpos = pos0 + token;
    uint last_pos = pos0 + n_tokens - 1u;
    uint first_raw_pos = last_pos + 1u - n_raw;
    uint raw_last_pos = first_raw_pos + n_raw - 1u;
    uint window_first = (window != 0u && qpos + 1u > window)
        ? qpos + 1u - window
        : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos = first; pos <= last; pos++) {
            uint logical = pos - first_raw_pos;
            uint row = (raw_start + logical) % raw_cap;
            uint64_t raw_base =
                (uint64_t)batch * (uint64_t)raw_kv_strides[0] +
                (uint64_t)row   * (uint64_t)raw_kv_strides[2];
            device const half4 *src = (device const half4 *)(raw_kv + raw_base);
            if (tid < 128u) kv_shared[tid] = src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            attend_shared();
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / ratio;
    visible = min(visible, n_comp);
    device const int32_t *row_topk = topk +
        (uint64_t)batch * (uint64_t)topk_strides[0] +
        (uint64_t)token * (uint64_t)topk_strides[1];
    for (uint i = 0u; i < top_k; i++) {
        int idx = row_topk[i];
        if (idx < 0) continue;
        if ((uint)idx >= visible) break;
        uint64_t comp_base =
            (uint64_t)batch * (uint64_t)comp_kv_strides[0] +
            (uint64_t)((uint)idx) * (uint64_t)comp_kv_strides[1];
        device const half4 *src = (device const half4 *)(comp_kv + comp_base);
        if (tid < 128u) kv_shared[tid] = src[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        attend_shared();
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sink_score = sinks[head];
    float old_m = M;
    float new_m = max(M, sink_score);
    float old_scale = exp(old_m - new_m);
    float row_scale = exp(sink_score - new_m);
    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    float inv_s = S == 0.0f ? 0.0f : 1.0f / S;
    device float4 *dst4 = (device float4 *)(
        out + (((uint64_t)batch * n_head + head) * n_tokens + token) * head_dim);
    dst4[lane +  0u] = o0 * inv_s;
    dst4[lane + 32u] = o1 * inv_s;
    dst4[lane + 64u] = o2 * inv_s;
    dst4[lane + 96u] = o3 * inv_s;
"""

_PREFILL_LIVE_F32_SOURCE = _PREFILL_LIVE_F16_SOURCE.replace(
    """device const half4 *q4 = (device const half4 *)(q + q_base);
    half4 q0 = q4[lane +  0u];
    half4 q1 = q4[lane + 32u];
    half4 q2 = q4[lane + 64u];
    half4 q3 = q4[lane + 96u];""",
    """device const float4 *q4 = (device const float4 *)(q + q_base);
    half4 q0 = (half4)q4[lane +  0u];
    half4 q1 = (half4)q4[lane + 32u];
    half4 q2 = (half4)q4[lane + 64u];
    half4 q3 = (half4)q4[lane + 96u];""",
)

_PREFILL_LIVE_V2_F16_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint sg = thread_position_in_threadgroup.y;
    uint tid = sg * 32u + lane;
    uint token = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y * 32u + sg;
    uint batch = threadgroup_position_in_grid.z;

    uint batch_size = meta[0];
    uint n_tokens   = meta[1];
    uint n_head     = meta[2];
    uint head_dim   = meta[3];
    uint n_raw      = meta[4];
    uint raw_cap    = meta[5];
    uint raw_start  = meta[6];
    uint n_comp     = meta[7];
    uint top_k      = meta[8];
    uint pos0       = meta[9];
    uint window     = meta[10];
    uint ratio      = meta[11];

    // The wrapper requires n_head % 32 == 0, so every simdgroup owns a valid
    // head and the batch/token guards below are threadgroup-uniform: no
    // thread returns while others still hit threadgroup barriers.
    if (batch >= batch_size || token >= n_tokens || head >= n_head) return;

    // 8 staged KV rows per barrier round, 32 heads sharing each staged tile.
    threadgroup half4 kv_shared[8u * 128u];
    threadgroup int staged_ok[8];
    threadgroup int staged_stop[8];

    uint srow = tid >> 7u;
    uint selem = tid & 127u;

    uint64_t q_base =
        (uint64_t)batch * (uint64_t)q_strides[0] +
        (uint64_t)head  * (uint64_t)q_strides[1] +
        (uint64_t)token * (uint64_t)q_strides[2];
    device const half4 *q4 = (device const half4 *)(q + q_base);
    half4 q0 = q4[lane +  0u];
    half4 q1 = q4[lane + 32u];
    half4 q2 = q4[lane + 64u];
    half4 q3 = q4[lane + 96u];

    float M = -1.701411733e38f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;
    float scale = rsqrt((float)head_dim);

    // Same per-row math as the one-row-per-round kernel: keeping the dot
    // structure, simd_sum order, and row visit order bit-identical is what
    // makes this a pure composition swap. Unrolled tile-wide score
    // precomputation and a two-deep score/update software pipeline both
    // measured slower than this plain loop (register pressure); the compiler
    // schedules the simple form best.
    auto row_score = [&](uint r) {
        uint off = r * 128u;
        half4 k0 = kv_shared[off + lane +  0u];
        half4 k1 = kv_shared[off + lane + 32u];
        half4 k2 = kv_shared[off + lane + 64u];
        half4 k3 = kv_shared[off + lane + 96u];
        float score = dot((float4)q0, (float4)k0) +
                      dot((float4)q1, (float4)k1) +
                      dot((float4)q2, (float4)k2) +
                      dot((float4)q3, (float4)k3);
        return simd_sum(score) * scale;
    };

    auto row_update = [&](uint r, float score) {
        uint off = r * 128u;
        half4 k0 = kv_shared[off + lane +  0u];
        half4 k1 = kv_shared[off + lane + 32u];
        half4 k2 = kv_shared[off + lane + 64u];
        half4 k3 = kv_shared[off + lane + 96u];

        float old_m = M;
        float new_m = max(M, score);
        if (new_m == old_m) {
            // Running max unchanged (the common case once it stabilizes):
            // old_scale would be exp(0) == 1.0 exactly, so S * old_scale and
            // the o rescale are bit-exact no-ops and are skipped.
            float row_scale = exp(score - new_m);
            S = S + row_scale;
            o0 += (float4)k0 * row_scale;
            o1 += (float4)k1 * row_scale;
            o2 += (float4)k2 * row_scale;
            o3 += (float4)k3 * row_scale;
            return;
        }
        float old_scale = exp(old_m - new_m);
        float row_scale = exp(score - new_m);

        S = S * old_scale + row_scale;
        o0 *= old_scale;
        o1 *= old_scale;
        o2 *= old_scale;
        o3 *= old_scale;

        o0 += (float4)k0 * row_scale;
        o1 += (float4)k1 * row_scale;
        o2 += (float4)k2 * row_scale;
        o3 += (float4)k3 * row_scale;
        M = new_m;
    };

    uint qpos = pos0 + token;
    uint last_pos = pos0 + n_tokens - 1u;
    uint first_raw_pos = last_pos + 1u - n_raw;
    uint raw_last_pos = first_raw_pos + n_raw - 1u;
    uint window_first = (window != 0u && qpos + 1u > window)
        ? qpos + 1u - window
        : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint base = first; base <= last; base += 8u) {
            uint count = min(last - base + 1u, 8u);
            if (srow < count) {
                uint pos = base + srow;
                uint logical = pos - first_raw_pos;
                uint row = (raw_start + logical) % raw_cap;
                uint64_t raw_base =
                    (uint64_t)batch * (uint64_t)raw_kv_strides[0] +
                    (uint64_t)row   * (uint64_t)raw_kv_strides[2];
                device const half4 *src = (device const half4 *)(raw_kv + raw_base);
                kv_shared[srow * 128u + selem] = src[selem];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0u; r < count; r++) {
                row_update(r, row_score(r));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / ratio;
    visible = min(visible, n_comp);
    // topk is indexed through the buffer name rather than a bound device
    // pointer: inputs under 8 elements arrive in the constant address space.
    uint64_t topk_base =
        (uint64_t)batch * (uint64_t)topk_strides[0] +
        (uint64_t)token * (uint64_t)topk_strides[1];
    for (uint i = 0u; i < top_k; i += 8u) {
        uint count = min(top_k - i, 8u);
        if (srow < count) {
            int idx = topk[topk_base + i + srow];
            bool ok = idx >= 0 && (uint)idx < visible;
            if (selem == 0u) {
                staged_ok[srow] = ok ? 1 : 0;
                // The caller passes ascending row ids, so a whole tile past
                // the visibility limit means every later tile is too; a
                // negative id (skipped, never a stop) keeps parity with the
                // one-row kernel's continue-then-break order.
                staged_stop[srow] = (idx >= 0 && (uint)idx >= visible) ? 1 : 0;
            }
            if (ok) {
                uint64_t comp_base =
                    (uint64_t)batch * (uint64_t)comp_kv_strides[0] +
                    (uint64_t)((uint)idx) * (uint64_t)comp_kv_strides[1];
                device const half4 *src = (device const half4 *)(comp_kv + comp_base);
                kv_shared[srow * 128u + selem] = src[selem];
            }
        } else if (selem == 0u) {
            staged_ok[srow] = 0;
            staged_stop[srow] = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool all_stop = true;
        for (uint r = 0u; r < count; r++) {
            if (staged_ok[r] != 0) {
                row_update(r, row_score(r));
            }
        }
        for (uint r = 0u; r < 8u; r++) {
            all_stop = all_stop && staged_stop[r] != 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (all_stop) {
            break;
        }
    }

    float sink_score = sinks[head];
    float old_m = M;
    float new_m = max(M, sink_score);
    float old_scale = exp(old_m - new_m);
    float row_scale = exp(sink_score - new_m);
    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    float inv_s = S == 0.0f ? 0.0f : 1.0f / S;
    device float4 *dst4 = (device float4 *)(
        out + (((uint64_t)batch * n_head + head) * n_tokens + token) * head_dim);
    dst4[lane +  0u] = o0 * inv_s;
    dst4[lane + 32u] = o1 * inv_s;
    dst4[lane + 64u] = o2 * inv_s;
    dst4[lane + 96u] = o3 * inv_s;
"""

_PREFILL_LIVE_V2_F32_SOURCE = _PREFILL_LIVE_V2_F16_SOURCE.replace(
    """device const half4 *q4 = (device const half4 *)(q + q_base);
    half4 q0 = q4[lane +  0u];
    half4 q1 = q4[lane + 32u];
    half4 q2 = q4[lane + 64u];
    half4 q3 = q4[lane + 96u];""",
    """device const float4 *q4 = (device const float4 *)(q + q_base);
    half4 q0 = (half4)q4[lane +  0u];
    half4 q1 = (half4)q4[lane + 32u];
    half4 q2 = (half4)q4[lane + 64u];
    half4 q3 = (half4)q4[lane + 96u];""",
)

_PREFILL_LIVE_MMA_SOURCE = """
    // Simdgroup-mma prefill consumer. One threadgroup owns one token and a
    // 16-head block; the 8 simdgroups split the work as (head tile of 8,
    // dim quarter of 128). KV rows arrive in staged 8-row tiles shared by
    // all 16 heads, scores come from 8x8 simdgroup matrix products of half
    // operands into float32 accumulators, and the P*V update runs float P
    // fragments against half KV fragments into float32 output accumulators
    // (the DS4-c flash-attention operand contract at head dim 512). The
    // queries stage once through threadgroup memory and then live in
    // register fragments for the whole row loop. Two rejected variants at
    // the anchor shape: a 16-row KV tile (56.5 versus 42.5 ms; masked
    // partial tiles still pay full tile math) and loading the strided
    // device query rows as fragments without staging (44.2 versus 42.5 ms).
    //
    // The row visit order, visibility rules, stop semantics, sink join, and
    // operand precision (queries and KV rows read as half, accumulation and
    // output in float32) match the scalar consumers, but per-row dots
    // become 8x8 tile products and the online-softmax rescale applies once
    // per 8-row tile instead of once per row, so f32 accumulation order
    // changes. The output is a numerically valid scalar-consumer variant
    // without bit identity; serving it is judged by the engaged
    // cached-path teacher-forced NLL and the generation gates (the
    // cache-less scorer forwards never reach this path).
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint token = threadgroup_position_in_grid.x;
    uint head_block = threadgroup_position_in_grid.y;
    uint batch = threadgroup_position_in_grid.z;

    uint batch_size = meta[0];
    uint n_tokens   = meta[1];
    uint n_head     = meta[2];
    uint head_dim   = meta[3];
    uint n_raw      = meta[4];
    uint raw_cap    = meta[5];
    uint raw_start  = meta[6];
    uint n_comp     = meta[7];
    uint top_k      = meta[8];
    uint pos0       = meta[9];
    uint window     = meta[10];
    uint ratio      = meta[11];

    // The wrapper sizes the grid exactly, so these guards are
    // threadgroup-uniform and barrier-safe.
    if (batch >= batch_size || token >= n_tokens ||
        head_block * 16u + 16u > n_head) return;

    threadgroup half sq[16u * 512u];       // staged query heads
    threadgroup half skv[8u * 512u];       // staged KV row tile
    threadgroup float spart[8u * 64u];     // per-simdgroup partial score tiles
    threadgroup float sp[16u * 8u];        // P tile (rows: heads, cols: kv rows)
    threadgroup float sms[16u];            // per-head tile rescale factor
    threadgroup float sdiag[2u * 64u];     // per-head-tile 8x8 diagonal matrices
    threadgroup int staged_ok[8];
    threadgroup int staged_stop[8];

    uint ht = sg >> 2u;                    // head tile owned in mma phases
    uint dq = sg & 3u;                     // dim quarter owned in mma phases
    uint head0 = head_block * 16u;

    // Off-diagonal entries are written once; only the diagonal is updated
    // per tile.
    for (uint i = tid; i < 128u; i += 256u) {
        sdiag[i] = 0.0f;
    }

    // Stage the 16 query heads through threadgroup memory with coalesced
    // device reads, cast to half like the scalar consumers' per-row query
    // loads.
    for (uint i = tid; i < 16u * 128u; i += 256u) {
        uint hh = i >> 7u;
        uint elem = i & 127u;
        uint64_t q_base =
            (uint64_t)batch * (uint64_t)q_strides[0] +
            (uint64_t)(head0 + hh) * (uint64_t)q_strides[1] +
            (uint64_t)token * (uint64_t)q_strides[2];
        device const half4 *q4 = (device const half4 *)(q + q_base);
        ((threadgroup half4 *)sq)[i] = q4[elem];
    }

    float M = -1.701411733e38f;
    float S = 0.0f;
    float scale = rsqrt((float)head_dim);

    // Softmax lanes: threads 0..127 cover the 16x8 (head, kv row) slots;
    // the 8 lanes of one head replicate its running max and sum, and the
    // shuffle trees below produce identical values in every lane of the
    // group.
    uint sm_h = tid >> 3u;
    uint sm_r = tid & 7u;

    simdgroup_float8x8 o_acc[16];
    for (uint i = 0u; i < 16u; i++) {
        o_acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup's query slice (8 heads x 128 dims) is loop-invariant;
    // holding it in register fragments saves 16 threadgroup loads per tile.
    simdgroup_half8x8 q_frag[16];
    for (uint f = 0u; f < 16u; f++) {
        simdgroup_load(q_frag[f], sq + (ht * 8u) * 512u + dq * 128u + f * 8u,
                       (ulong)512u, ulong2(0u, 0u), false);
    }

    auto process_tile = [&]() {
        // Partial QK scores for this simdgroup's head tile and dim quarter.
        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint f = 0u; f < 16u; f++) {
            simdgroup_half8x8 mk;
            simdgroup_load(mk, skv + dq * 128u + f * 8u,
                           (ulong)512u, ulong2(0u, 0u), true);
            simdgroup_multiply_accumulate(mqk, q_frag[f], mk, mqk);
        }
        simdgroup_store(mqk, spart + sg * 64u, (ulong)8u, ulong2(0u, 0u), false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-head online softmax over the staged tile.
        if (tid < 128u) {
            uint hti = sm_h >> 3u;
            uint hrow = sm_h & 7u;
            bool ok = staged_ok[sm_r] != 0;
            float s = -INFINITY;
            if (ok) {
                float part = 0.0f;
                for (uint p = 0u; p < 4u; p++) {
                    part += spart[(hti * 4u + p) * 64u + hrow * 8u + sm_r];
                }
                s = part * scale;
            }
            float gmax = s;
            gmax = max(gmax, simd_shuffle_xor(gmax, 1u));
            gmax = max(gmax, simd_shuffle_xor(gmax, 2u));
            gmax = max(gmax, simd_shuffle_xor(gmax, 4u));

            float new_m = max(M, gmax);
            // exp(0) == 1.0f exactly when the running max is unchanged, so
            // the rescale below skips as a whole-tile check.
            float ms = exp(M - new_m);
            float pv = ok ? exp(s - new_m) : 0.0f;
            float psum = pv;
            psum += simd_shuffle_xor(psum, 1u);
            psum += simd_shuffle_xor(psum, 2u);
            psum += simd_shuffle_xor(psum, 4u);

            S = S * ms + psum;
            M = new_m;
            sp[sm_h * 8u + sm_r] = pv;
            if (sm_r == 0u) {
                sms[sm_h] = ms;
                sdiag[hti * 64u + hrow * 9u] = ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale the output accumulators when any running max moved
        // (diagonal left-multiply scales the head rows).
        bool rescale = false;
        for (uint j = 0u; j < 8u; j++) {
            rescale = rescale || (sms[ht * 8u + j] != 1.0f);
        }
        if (rescale) {
            simdgroup_float8x8 mdiag;
            simdgroup_load(mdiag, sdiag + ht * 64u,
                           (ulong)8u, ulong2(0u, 0u), false);
            for (uint f = 0u; f < 16u; f++) {
                simdgroup_float8x8 t;
                simdgroup_multiply(t, mdiag, o_acc[f]);
                o_acc[f] = t;
            }
        }

        // P x V accumulate for this simdgroup's dims. Invalid rows carry
        // P == 0 against zero-filled staged rows.
        simdgroup_float8x8 mp;
        simdgroup_load(mp, sp + (ht * 8u) * 8u, (ulong)8u, ulong2(0u, 0u), false);
        for (uint f = 0u; f < 16u; f++) {
            simdgroup_half8x8 mv;
            simdgroup_load(mv, skv + dq * 128u + f * 8u,
                           (ulong)512u, ulong2(0u, 0u), false);
            simdgroup_multiply_accumulate(o_acc[f], mp, mv, o_acc[f]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    };

    uint qpos = pos0 + token;
    uint last_pos = pos0 + n_tokens - 1u;
    uint first_raw_pos = last_pos + 1u - n_raw;
    uint raw_last_pos = first_raw_pos + n_raw - 1u;
    uint window_first = (window != 0u && qpos + 1u > window)
        ? qpos + 1u - window
        : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    uint srow = tid >> 5u;                 // staging row of the 8-row tile
    uint selem = tid & 31u;                // staging quad within the row

    if (first <= last) {
        for (uint base = first; base <= last; base += 8u) {
            uint count = min(last - base + 1u, 8u);
            threadgroup half4 *dst = ((threadgroup half4 *)skv) + srow * 128u;
            if (srow < count) {
                uint pos = base + srow;
                uint logical = pos - first_raw_pos;
                uint row = (raw_start + logical) % raw_cap;
                uint64_t raw_base =
                    (uint64_t)batch * (uint64_t)raw_kv_strides[0] +
                    (uint64_t)row   * (uint64_t)raw_kv_strides[2];
                device const half4 *src = (device const half4 *)(raw_kv + raw_base);
                dst[selem +  0u] = src[selem +  0u];
                dst[selem + 32u] = src[selem + 32u];
                dst[selem + 64u] = src[selem + 64u];
                dst[selem + 96u] = src[selem + 96u];
            } else {
                half4 z = half4(0.0h);
                dst[selem +  0u] = z;
                dst[selem + 32u] = z;
                dst[selem + 64u] = z;
                dst[selem + 96u] = z;
            }
            if (selem == 0u) {
                staged_ok[srow] = srow < count ? 1 : 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            process_tile();
        }
    }

    uint visible = (qpos + 1u) / ratio;
    visible = min(visible, n_comp);
    // topk is indexed through the buffer name rather than a bound device
    // pointer: inputs under 8 elements arrive in the constant address space.
    uint64_t topk_base =
        (uint64_t)batch * (uint64_t)topk_strides[0] +
        (uint64_t)token * (uint64_t)topk_strides[1];
    for (uint i = 0u; i < top_k; i += 8u) {
        uint count = min(top_k - i, 8u);
        bool ok = false;
        threadgroup half4 *dst = ((threadgroup half4 *)skv) + srow * 128u;
        if (srow < count) {
            int idx = topk[topk_base + i + srow];
            ok = idx >= 0 && (uint)idx < visible;
            if (selem == 0u) {
                staged_ok[srow] = ok ? 1 : 0;
                // The caller passes ascending row ids, so a whole tile past
                // the visibility limit means every later tile is too; a
                // negative id (skipped, never a stop) keeps parity with the
                // scalar kernels' continue-then-break order.
                staged_stop[srow] = (idx >= 0 && (uint)idx >= visible) ? 1 : 0;
            }
            if (ok) {
                uint64_t comp_base =
                    (uint64_t)batch * (uint64_t)comp_kv_strides[0] +
                    (uint64_t)((uint)idx) * (uint64_t)comp_kv_strides[1];
                device const half4 *src = (device const half4 *)(comp_kv + comp_base);
                dst[selem +  0u] = src[selem +  0u];
                dst[selem + 32u] = src[selem + 32u];
                dst[selem + 64u] = src[selem + 64u];
                dst[selem + 96u] = src[selem + 96u];
            }
        } else if (selem == 0u) {
            staged_ok[srow] = 0;
            staged_stop[srow] = 1;
        }
        if (!ok) {
            half4 z = half4(0.0h);
            dst[selem +  0u] = z;
            dst[selem + 32u] = z;
            dst[selem + 64u] = z;
            dst[selem + 96u] = z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool all_stop = true;
        for (uint r = 0u; r < 8u; r++) {
            all_stop = all_stop && staged_stop[r] != 0;
        }
        process_tile();
        if (all_stop) {
            break;
        }
    }

    // Sink join (denominator only) and the combined final factor
    // old-scale / S, applied as one diagonal multiply on the way out.
    if (tid < 128u) {
        float sink_score = sinks[head0 + sm_h];
        float new_m = max(M, sink_score);
        float ms = exp(M - new_m);
        float vs = exp(sink_score - new_m);
        S = S * ms + vs;
        float inv_s = S == 0.0f ? 0.0f : 1.0f / S;
        if (sm_r == 0u) {
            uint hti = sm_h >> 3u;
            uint hrow = sm_h & 7u;
            sdiag[hti * 64u + hrow * 9u] = ms * inv_s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 mfin;
    simdgroup_load(mfin, sdiag + ht * 64u, (ulong)8u, ulong2(0u, 0u), false);
    uint64_t out_base =
        (((uint64_t)batch * n_head + head0 + ht * 8u) * n_tokens + token) * 512u;
    for (uint f = 0u; f < 16u; f++) {
        simdgroup_float8x8 t;
        simdgroup_multiply(t, mfin, o_acc[f]);
        simdgroup_store(t, out + out_base + dq * 128u + f * 8u,
                        (ulong)n_tokens * 512u, ulong2(0u, 0u), false);
    }
"""

_INDEXER_SCORE_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint c0 = threadgroup_position_in_grid.x * 32u;
    uint t0 = threadgroup_position_in_grid.y * 8u;
    uint batch = threadgroup_position_in_grid.z;

    uint batch_size = meta[0];
    uint n_tokens   = meta[1];
    uint n_head     = meta[2];
    uint head_dim   = meta[3];
    uint n_comp     = meta[4];
    uint pos0       = meta[5];
    uint ratio      = meta[6];

    if (batch >= batch_size || tid >= 128u) return;

    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D = 128;

    threadgroup half qtg[TM * D];
    threadgroup half ktg[TN * D];
    threadgroup float dot[TM * TN];

    uint last_token = min(t0 + TM, n_tokens);
    uint max_visible = last_token > t0
        ? min((pos0 + last_token) / ratio, n_comp)
        : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM * TN; i += 128u) {
            uint r = i / TN;
            uint cc = i - r * TN;
            uint token = t0 + r;
            uint comp = c0 + cc;
            if (token < n_tokens && comp < n_comp) {
                scores[(batch * n_tokens + token) * n_comp + comp] = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN * D; i += 128u) {
        uint cc = i / D;
        uint d = i - cc * D;
        uint comp = c0 + cc;
        half v = half(0.0f);
        if (comp < n_comp) {
            uint64_t base =
                (uint64_t)batch * (uint64_t)index_comp_strides[0] +
                (uint64_t)comp  * (uint64_t)index_comp_strides[1] +
                (uint64_t)d     * (uint64_t)index_comp_strides[2];
            v = half(index_comp[base]);
        }
        ktg[i] = v;
    }

    uint cell0 = lane;
    uint cell1 = lane + 32u;
    uint row0 = cell0 >> 3;
    uint row1 = cell1 >> 3;
    uint sub0 = cell0 & 7u;
    uint sub1 = cell1 & 7u;
    uint col0 = sg * TS + sub0;
    uint col1 = sg * TS + sub1;
    uint token0 = t0 + row0;
    uint token1 = t0 + row1;
    uint comp0 = c0 + col0;
    uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0u; head < n_head; head++) {
        for (uint i = tid; i < TM * D; i += 128u) {
            uint r = i / D;
            uint d = i - r * D;
            uint token = t0 + r;
            half v = half(0.0f);
            if (token < n_tokens) {
                uint64_t base =
                    (uint64_t)batch * (uint64_t)q_strides[0] +
                    (uint64_t)head  * (uint64_t)q_strides[1] +
                    (uint64_t)token * (uint64_t)q_strides[2] +
                    (uint64_t)d     * (uint64_t)q_strides[3];
                v = half(q[base]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0u; db < D / TS; db++) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db * TS, D, 0, false);
            simdgroup_load(mk, ktg + (sg * TS) * D + db * TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < n_tokens && comp0 < n_comp) {
            uint64_t w_base =
                (uint64_t)batch  * (uint64_t)weights_strides[0] +
                (uint64_t)token0 * (uint64_t)weights_strides[1] +
                (uint64_t)head   * (uint64_t)weights_strides[2];
            float s = dot[row0 * TN + col0];
            acc0 += max(s, 0.0f) * weights[w_base];
        }
        if (token1 < n_tokens && comp1 < n_comp) {
            uint64_t w_base =
                (uint64_t)batch  * (uint64_t)weights_strides[0] +
                (uint64_t)token1 * (uint64_t)weights_strides[1] +
                (uint64_t)head   * (uint64_t)weights_strides[2];
            float s = dot[row1 * TN + col1];
            acc1 += max(s, 0.0f) * weights[w_base];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < n_tokens && comp0 < n_comp) {
        uint visible = min((pos0 + token0 + 1u) / ratio, n_comp);
        scores[(batch * n_tokens + token0) * n_comp + comp0] =
            comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < n_tokens && comp1 < n_comp) {
        uint visible = min((pos0 + token1 + 1u) / ratio, n_comp);
        scores[(batch * n_tokens + token1) * n_comp + comp1] =
            comp1 < visible ? acc1 : -INFINITY;
    }
"""

_INDEXER_Q_QAT_SOURCE = """
    float e2m1_values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };

    auto e2m1_dequant = [&](float x) {
        float sign = x < 0.0f ? -1.0f : 1.0f;
        float ax = min(abs(x), 6.0f);
        int best = 0;
        float best_diff = abs(ax - e2m1_values[0]);
        for (int i = 1; i < 8; i++) {
            float diff = abs(ax - e2m1_values[i]);
            if (diff < best_diff ||
                (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
                best = i;
                best_diff = diff;
            }
        }
        return sign * e2m1_values[best];
    };

    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;

    uint batch_size = meta[0];
    uint n_head     = meta[1];
    uint n_tokens   = meta[2];
    uint head_dim   = meta[3];
    uint n_rows     = batch_size * n_head * n_tokens;

    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    threadgroup float vals[128];
    threadgroup float absbuf[128];

    uint rem = row;
    uint token = rem % n_tokens;
    rem = rem / n_tokens;
    uint head = rem % n_head;
    uint batch = rem / n_head;

    uint64_t in_base =
        (uint64_t)batch * (uint64_t)q_strides[0] +
        (uint64_t)head  * (uint64_t)q_strides[1] +
        (uint64_t)token * (uint64_t)q_strides[2];

    vals[tid] = q[in_base + (uint64_t)tid * (uint64_t)q_strides[3]];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint block = tid >> 5u;
    uint lane = tid & 31u;
    uint block_base = block * 32u;
    absbuf[tid] = abs(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = max(
                absbuf[block_base + lane],
                absbuf[block_base + lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float amax = max(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2(ceil(log2(amax / 6.0f)));
    out[((uint64_t)row * 128u) + tid] =
        e2m1_dequant(clamp(v / scale, -6.0f, 6.0f)) * scale;
"""


_INDEXER_Q_QAT_V2_SOURCE = """
    // Same math as the one-row-per-threadgroup QAT kernel, restructured to
    // one row per simdgroup with four elements per lane and simd_shuffle_xor
    // for the cross-lane butterfly strides. The radix-2 stage order and
    // pairing match the barrier version exactly, the 32-element group max is
    // order-exact, and the scale/tie expressions are copied verbatim, so the
    // output is bit-identical while dropping every threadgroup barrier.
    float e2m1_values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };

    auto e2m1_dequant = [&](float x) {
        float sign = x < 0.0f ? -1.0f : 1.0f;
        float ax = min(abs(x), 6.0f);
        int best = 0;
        float best_diff = abs(ax - e2m1_values[0]);
        for (int i = 1; i < 8; i++) {
            float diff = abs(ax - e2m1_values[i]);
            if (diff < best_diff ||
                (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
                best = i;
                best_diff = diff;
            }
        }
        return sign * e2m1_values[best];
    };

    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x * 4u + sg;

    uint batch_size = meta[0];
    uint n_head     = meta[1];
    uint n_tokens   = meta[2];
    uint head_dim   = meta[3];
    uint n_rows     = batch_size * n_head * n_tokens;

    if (row >= n_rows || head_dim != 128u) return;

    uint rem = row;
    uint token = rem % n_tokens;
    rem = rem / n_tokens;
    uint head = rem % n_head;
    uint batch = rem / n_head;

    uint64_t in_base =
        (uint64_t)batch * (uint64_t)q_strides[0] +
        (uint64_t)head  * (uint64_t)q_strides[1] +
        (uint64_t)token * (uint64_t)q_strides[2];

    // Lane l holds elements [4l, 4l + 3].
    float4 x;
    for (uint j = 0u; j < 4u; j++) {
        x[j] = q[in_base + (uint64_t)(4u * lane + j) * (uint64_t)q_strides[3]];
    }

    // Strides 1 and 2 inside the lane, strides 4..64 across lanes; the
    // (a + b, a - b) pairing per stage matches the barrier kernel's
    // butterfly indexing.
    float a0 = x.x + x.y;
    float a1 = x.x - x.y;
    float a2 = x.z + x.w;
    float a3 = x.z - x.w;
    x = float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);
    for (ushort m = 1; m <= 16; m <<= 1) {
        float4 other = simd_shuffle_xor(x, m);
        x = (lane & m) ? (other - x) : (x + other);
    }
    x = x * 0.08838834764831845f;

    // 32-element group max: lanes [8g, 8g + 7] hold group g; max is exact
    // in any order.
    float amax = max(max(abs(x.x), abs(x.y)), max(abs(x.z), abs(x.w)));
    amax = max(amax, simd_shuffle_xor(amax, 1));
    amax = max(amax, simd_shuffle_xor(amax, 2));
    amax = max(amax, simd_shuffle_xor(amax, 4));
    amax = max(amax, 7.052966104933725e-38f);
    float scale = exp2(ceil(log2(amax / 6.0f)));

    device float *orow = out + (uint64_t)row * 128u;
    for (uint j = 0u; j < 4u; j++) {
        orow[4u * lane + j] =
            e2m1_dequant(clamp(x[j] / scale, -6.0f, 6.0f)) * scale;
    }
"""


def _get_kernel(n_heads: int, head_dim: int):
    key = (int(n_heads), int(head_dim))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=f"moespresso_dsv4_indexed_attention_h{n_heads}_d{head_dim}",
        input_names=["q", "raw_kv", "comp_kv", "topk", "sinks", "meta"],
        output_names=["out"],
        source=_SOURCE,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _get_prefill_kernel(n_heads: int, head_dim: int):
    key = (int(n_heads), int(head_dim))
    cached = _PREFILL_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=f"moespresso_dsv4_indexed_attention_prefill_h{n_heads}_d{head_dim}",
        input_names=["q", "raw_kv", "comp_kv", "topk", "sinks", "meta"],
        output_names=["out"],
        source=_PREFILL_SOURCE,
    )
    _PREFILL_KERNEL_CACHE[key] = kernel
    return kernel


def _get_prefill_live_f16_kernel(n_heads: int, head_dim: int):
    key = (int(n_heads), int(head_dim))
    cached = _PREFILL_LIVE_F16_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=f"moespresso_dsv4_indexed_attention_prefill_live_f16_h{n_heads}_d{head_dim}",
        input_names=["q", "raw_kv", "comp_kv", "topk", "sinks", "meta"],
        output_names=["out"],
        source=_PREFILL_LIVE_F16_SOURCE,
        ensure_row_contiguous=False,
    )
    _PREFILL_LIVE_F16_KERNEL_CACHE[key] = kernel
    return kernel


def _get_prefill_live_f32_kernel(n_heads: int, head_dim: int):
    key = (int(n_heads), int(head_dim))
    cached = _PREFILL_LIVE_F32_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=f"moespresso_dsv4_indexed_attention_prefill_live_f32_h{n_heads}_d{head_dim}",
        input_names=["q", "raw_kv", "comp_kv", "topk", "sinks", "meta"],
        output_names=["out"],
        source=_PREFILL_LIVE_F32_SOURCE,
        ensure_row_contiguous=False,
    )
    _PREFILL_LIVE_F32_KERNEL_CACHE[key] = kernel
    return kernel


def _get_prefill_live_v2_f16_kernel(n_heads: int, head_dim: int):
    key = (int(n_heads), int(head_dim))
    cached = _PREFILL_LIVE_V2_F16_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=f"moespresso_dsv4_indexed_attention_prefill_live_v2_f16_h{n_heads}_d{head_dim}",
        input_names=["q", "raw_kv", "comp_kv", "topk", "sinks", "meta"],
        output_names=["out"],
        source=_PREFILL_LIVE_V2_F16_SOURCE,
        ensure_row_contiguous=False,
    )
    _PREFILL_LIVE_V2_F16_KERNEL_CACHE[key] = kernel
    return kernel


def _get_prefill_live_v2_f32_kernel(n_heads: int, head_dim: int):
    key = (int(n_heads), int(head_dim))
    cached = _PREFILL_LIVE_V2_F32_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=f"moespresso_dsv4_indexed_attention_prefill_live_v2_f32_h{n_heads}_d{head_dim}",
        input_names=["q", "raw_kv", "comp_kv", "topk", "sinks", "meta"],
        output_names=["out"],
        source=_PREFILL_LIVE_V2_F32_SOURCE,
        ensure_row_contiguous=False,
    )
    _PREFILL_LIVE_V2_F32_KERNEL_CACHE[key] = kernel
    return kernel


def _prefill_live_v2_enabled() -> bool:
    """Gate for the 8-row-staged, 32-heads-per-threadgroup prefill consumer.

    The v2 kernel is bit-identical to the one-row kernel (same per-row dot
    structure, simd_sum order, and row visit order); the gate exists so the
    two compositions can be A/B'd on the served path.
    """
    import os

    return os.environ.get("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_V2", "1") != "0"


def _get_prefill_live_mma_kernel(n_heads: int, head_dim: int):
    key = (int(n_heads), int(head_dim))
    cached = _PREFILL_LIVE_MMA_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=f"moespresso_dsv4_indexed_attention_prefill_live_mma_h{n_heads}_d{head_dim}",
        input_names=["q", "raw_kv", "comp_kv", "topk", "sinks", "meta"],
        output_names=["out"],
        source=_PREFILL_LIVE_MMA_SOURCE,
        ensure_row_contiguous=False,
    )
    _PREFILL_LIVE_MMA_KERNEL_CACHE[key] = kernel
    return kernel


def _prefill_live_mma_enabled() -> bool:
    """Gate for the simdgroup-mma prefill consumer, default on.

    Unlike the v2 gate this selects a numerically valid f32
    accumulation-order variant. Its lack of bit identity means
    the default was set through the math-change gate campaign: the engaged
    cached-path teacher-forced long-NLL holds parity with the scalar
    consumer (combined 0.0397102275 versus 0.0400776353, mixed sign across
    the two long prompts), Q2 avg_nll and Q3 recall are unchanged, Q1
    reproduces the fused-default realization token for token, and served
    64-token arms are token-identical to the fused anchor. The fenced
    same-stage A/B at the anchor prefill shape reads 62.4 versus 41.9 ms
    (1.49x). The cache-less scorer forwards (Q2, the stock long-prompt NLL
    probe) never reach the ratio-4 fast path, which is why the engaged
    cached-path probe is the discriminating NLL. Kill switch
    ``MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA=0`` falls back to the scalar
    consumers.
    """
    import os

    return os.environ.get("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1") != "0"


def _get_indexer_score_kernel(n_heads: int, head_dim: int, operand_form: str):
    # One kernel name per operand dtype: the compiled pipeline is cached by
    # name, so a float32-compiled kernel would otherwise be reused for half
    # buffers and read reinterpreted bytes.
    key = (int(n_heads), int(head_dim), str(operand_form))
    cached = _INDEXER_SCORE_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=(
            f"moespresso_dsv4_indexer_scores_tiled_{operand_form}"
            f"_h{n_heads}_d{head_dim}"
        ),
        input_names=["q", "weights", "index_comp", "meta"],
        output_names=["scores"],
        source=_INDEXER_SCORE_SOURCE,
        ensure_row_contiguous=False,
    )
    _INDEXER_SCORE_KERNEL_CACHE[key] = kernel
    return kernel


def _get_indexer_q_qat_kernel(n_heads: int, head_dim: int):
    key = (int(n_heads), int(head_dim))
    cached = _INDEXER_Q_QAT_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=f"moespresso_dsv4_indexer_q_qat_h{n_heads}_d{head_dim}",
        input_names=["q", "meta"],
        output_names=["out"],
        source=_INDEXER_Q_QAT_SOURCE,
        ensure_row_contiguous=False,
    )
    _INDEXER_Q_QAT_KERNEL_CACHE[key] = kernel
    return kernel


def indexed_mixed_attention_decode(
    q,
    raw_kv,
    comp_kv,
    topk,
    sinks,
    *,
    pos0: int,
    window: int = 128,
    ratio: int = 4,
    raw_start: int = 0,
):
    """Run the one-token DS4 indexed mixed-attention probe kernel.

    Args:
        q: float32 ``[n_heads, 512]`` query heads.
        raw_kv: float16 ``[raw_cap, 512]`` contiguous/circular raw KV rows.
        comp_kv: float16 ``[n_comp, 512]`` compressed KV rows.
        topk: int32 ``[top_k]`` selected compressed row ids.
        sinks: float32 ``[n_heads]`` attention sink logits.
    """
    import mlx.core as mx

    if q.ndim != 2 or raw_kv.ndim != 2 or comp_kv.ndim != 2:
        raise ValueError("q, raw_kv, and comp_kv must be rank-2")
    n_heads = int(q.shape[0])
    head_dim = int(q.shape[1])
    if head_dim != 512:
        raise ValueError("DS4 indexed attention probe currently requires head_dim=512")
    if int(raw_kv.shape[1]) != head_dim or int(comp_kv.shape[1]) != head_dim:
        raise ValueError("raw_kv and comp_kv width must match q")
    if topk.ndim != 1:
        raise ValueError("topk must be rank-1")
    if sinks.shape != (n_heads,):
        raise ValueError("sinks must have shape [n_heads]")
    if q.dtype != mx.float32:
        raise ValueError("q must be float32, matching DS4-c's q buffer contract")
    if raw_kv.dtype != mx.float16 or comp_kv.dtype != mx.float16:
        raise ValueError("raw_kv and comp_kv must be float16")
    if topk.dtype != mx.int32:
        topk = topk.astype(mx.int32)
    if sinks.dtype != mx.float32:
        sinks = sinks.astype(mx.float32)

    raw_cap = int(raw_kv.shape[0])
    if raw_cap <= 0:
        raise ValueError("raw_kv must contain at least one row")
    if not (0 <= int(raw_start) < raw_cap):
        raise ValueError("raw_start must be inside raw_kv")
    meta = mx.array(
        [
            n_heads,
            head_dim,
            raw_cap,
            raw_cap,
            int(raw_start),
            int(comp_kv.shape[0]),
            int(topk.shape[0]),
            int(pos0),
            int(window),
            int(ratio),
        ],
        dtype=mx.uint32,
    )
    kernel = _get_kernel(n_heads, head_dim)
    out, = kernel(
        inputs=[q, raw_kv, comp_kv, topk, sinks, meta],
        output_shapes=[(n_heads, head_dim)],
        output_dtypes=[mx.float32],
        grid=(32, n_heads, 1),
        threadgroup=(32, 1, 1),
    )
    return out


def indexed_mixed_attention_prefill_live_f16(
    q,
    raw_kv,
    comp_kv,
    topk,
    sinks,
    *,
    pos0: int,
    window: int = 128,
    ratio: int = 4,
    raw_start: int = 0,
):
    """Run indexed prefill attention directly on JANG's live BHLD layout.

    Args:
        q: float16 ``[batch, n_heads, tokens, 512]`` query heads.
        raw_kv: float16 ``[batch, 1, raw_cap, 512]`` local raw KV rows.
        comp_kv: float16 ``[batch, n_comp, 512]`` compressed KV rows.
        topk: int32 ``[batch, tokens, top_k]`` selected compressed row ids.
        sinks: float32 ``[n_heads]`` attention sink logits.
    """
    import mlx.core as mx

    if q.ndim != 4 or raw_kv.ndim != 4 or comp_kv.ndim != 3:
        raise ValueError("q/raw_kv/comp_kv must have live BHLD/B1LD/BPD layouts")
    batch = int(q.shape[0])
    n_heads = int(q.shape[1])
    n_tokens = int(q.shape[2])
    head_dim = int(q.shape[3])
    if batch <= 0 or n_tokens <= 0:
        raise ValueError("q must contain at least one batch and token")
    if head_dim != 512:
        raise ValueError("DS4 indexed prefill probe currently requires head_dim=512")
    if n_heads % 8 != 0:
        raise ValueError("DS4 indexed prefill probe requires n_heads to be divisible by 8")
    if int(raw_kv.shape[0]) != batch or int(comp_kv.shape[0]) != batch:
        raise ValueError("raw_kv and comp_kv batch size must match q")
    if int(raw_kv.shape[-1]) != head_dim or int(comp_kv.shape[-1]) != head_dim:
        raise ValueError("raw_kv and comp_kv width must match q")
    if topk.ndim != 3 or int(topk.shape[0]) != batch or int(topk.shape[1]) != n_tokens:
        raise ValueError("topk must have shape [batch, tokens, top_k]")
    if sinks.shape != (n_heads,):
        raise ValueError("sinks must have shape [n_heads]")
    if q.dtype != mx.float16:
        raise ValueError("live prefill probe requires q to be float16")
    if raw_kv.dtype != mx.float16 or comp_kv.dtype != mx.float16:
        raise ValueError("raw_kv and comp_kv must be float16")
    if topk.dtype != mx.int32:
        topk = topk.astype(mx.int32)
    if sinks.dtype != mx.float32:
        sinks = sinks.astype(mx.float32)

    raw_cap = int(raw_kv.shape[2])
    if raw_cap <= 0:
        raise ValueError("raw_kv must contain at least one row")
    if not (0 <= int(raw_start) < raw_cap):
        raise ValueError("raw_start must be inside raw_kv")
    meta = mx.array(
        [
            batch,
            n_tokens,
            n_heads,
            head_dim,
            raw_cap,
            raw_cap,
            int(raw_start),
            int(comp_kv.shape[1]),
            int(topk.shape[2]),
            int(pos0),
            int(window),
            int(ratio),
        ],
        dtype=mx.uint32,
    )
    if n_heads % 16 == 0 and _prefill_live_mma_enabled():
        kernel = _get_prefill_live_mma_kernel(n_heads, head_dim)
        _CONSUMER_CALL_COUNTS["mma"] += 1
        out, = kernel(
            inputs=[q, raw_kv, comp_kv, topk, sinks, meta],
            output_shapes=[(batch, n_heads, n_tokens, head_dim)],
            output_dtypes=[mx.float32],
            grid=(256 * n_tokens, n_heads // 16, batch),
            threadgroup=(256, 1, 1),
        )
        return out
    if n_heads % 32 == 0 and _prefill_live_v2_enabled():
        kernel = _get_prefill_live_v2_f16_kernel(n_heads, head_dim)
        _CONSUMER_CALL_COUNTS["v2"] += 1
        out, = kernel(
            inputs=[q, raw_kv, comp_kv, topk, sinks, meta],
            output_shapes=[(batch, n_heads, n_tokens, head_dim)],
            output_dtypes=[mx.float32],
            grid=(32 * n_tokens, 32 * (n_heads // 32), batch),
            threadgroup=(32, 32, 1),
        )
        return out
    kernel = _get_prefill_live_f16_kernel(n_heads, head_dim)
    _CONSUMER_CALL_COUNTS["v1"] += 1
    out, = kernel(
        inputs=[q, raw_kv, comp_kv, topk, sinks, meta],
        output_shapes=[(batch, n_heads, n_tokens, head_dim)],
        output_dtypes=[mx.float32],
        grid=(32 * n_tokens, 8 * (n_heads // 8), batch),
        threadgroup=(32, 8, 1),
    )
    return out


def indexed_mixed_attention_prefill_live_f32(
    q,
    raw_kv,
    comp_kv,
    topk,
    sinks,
    *,
    pos0: int,
    window: int = 128,
    ratio: int = 4,
    raw_start: int = 0,
):
    """Run indexed prefill attention on live BHLD layout with float32 q."""
    import mlx.core as mx

    if q.ndim != 4 or raw_kv.ndim != 4 or comp_kv.ndim != 3:
        raise ValueError("q/raw_kv/comp_kv must have live BHLD/B1LD/BPD layouts")
    batch = int(q.shape[0])
    n_heads = int(q.shape[1])
    n_tokens = int(q.shape[2])
    head_dim = int(q.shape[3])
    if batch <= 0 or n_tokens <= 0:
        raise ValueError("q must contain at least one batch and token")
    if head_dim != 512:
        raise ValueError("DS4 indexed prefill probe currently requires head_dim=512")
    if n_heads % 8 != 0:
        raise ValueError("DS4 indexed prefill probe requires n_heads to be divisible by 8")
    if int(raw_kv.shape[0]) != batch or int(comp_kv.shape[0]) != batch:
        raise ValueError("raw_kv and comp_kv batch size must match q")
    if int(raw_kv.shape[-1]) != head_dim or int(comp_kv.shape[-1]) != head_dim:
        raise ValueError("raw_kv and comp_kv width must match q")
    if topk.ndim != 3 or int(topk.shape[0]) != batch or int(topk.shape[1]) != n_tokens:
        raise ValueError("topk must have shape [batch, tokens, top_k]")
    if sinks.shape != (n_heads,):
        raise ValueError("sinks must have shape [n_heads]")
    if q.dtype != mx.float32:
        raise ValueError("live f32 prefill probe requires q to be float32")
    if raw_kv.dtype != mx.float16 or comp_kv.dtype != mx.float16:
        raise ValueError("raw_kv and comp_kv must be float16")
    if topk.dtype != mx.int32:
        topk = topk.astype(mx.int32)
    if sinks.dtype != mx.float32:
        sinks = sinks.astype(mx.float32)

    raw_cap = int(raw_kv.shape[2])
    if raw_cap <= 0:
        raise ValueError("raw_kv must contain at least one row")
    if not (0 <= int(raw_start) < raw_cap):
        raise ValueError("raw_start must be inside raw_kv")
    meta = mx.array(
        [
            batch,
            n_tokens,
            n_heads,
            head_dim,
            raw_cap,
            raw_cap,
            int(raw_start),
            int(comp_kv.shape[1]),
            int(topk.shape[2]),
            int(pos0),
            int(window),
            int(ratio),
        ],
        dtype=mx.uint32,
    )
    if n_heads % 16 == 0 and _prefill_live_mma_enabled():
        # The scalar f32 kernel casts each query row to half on load; one
        # up-front cast is the identical rounding and lets the half-operand
        # mma kernel serve both input dtypes.
        kernel = _get_prefill_live_mma_kernel(n_heads, head_dim)
        _CONSUMER_CALL_COUNTS["mma"] += 1
        out, = kernel(
            inputs=[q.astype(mx.float16), raw_kv, comp_kv, topk, sinks, meta],
            output_shapes=[(batch, n_heads, n_tokens, head_dim)],
            output_dtypes=[mx.float32],
            grid=(256 * n_tokens, n_heads // 16, batch),
            threadgroup=(256, 1, 1),
        )
        return out
    if n_heads % 32 == 0 and _prefill_live_v2_enabled():
        kernel = _get_prefill_live_v2_f32_kernel(n_heads, head_dim)
        _CONSUMER_CALL_COUNTS["v2"] += 1
        out, = kernel(
            inputs=[q, raw_kv, comp_kv, topk, sinks, meta],
            output_shapes=[(batch, n_heads, n_tokens, head_dim)],
            output_dtypes=[mx.float32],
            grid=(32 * n_tokens, 32 * (n_heads // 32), batch),
            threadgroup=(32, 32, 1),
        )
        return out
    kernel = _get_prefill_live_f32_kernel(n_heads, head_dim)
    _CONSUMER_CALL_COUNTS["v1"] += 1
    out, = kernel(
        inputs=[q, raw_kv, comp_kv, topk, sinks, meta],
        output_shapes=[(batch, n_heads, n_tokens, head_dim)],
        output_dtypes=[mx.float32],
        grid=(32 * n_tokens, 8 * (n_heads // 8), batch),
        threadgroup=(32, 8, 1),
    )
    return out


def banded_prefill_attention_live(
    q,
    raw_kv,
    comp_kv,
    topk,
    sinks,
    *,
    pos0: int,
    window: int,
    ratio: int,
):
    """Run banded prefill attention through the simdgroup-mma consumer.

    Banded layers (sliding-window and ratio-128) attend a ``window``-token
    local band plus every visible pooled row. Passing ascending pool row ids
    in ``topk`` makes the kernel's visibility rule (row id below
    ``(position + 1) / ratio``) reproduce the compressed-pool visibility
    predicate exactly, so the ratio-4 consumer kernel serves the
    attend-all-pool case unchanged. Zero-pool layers pass a one-row zero
    ``comp_kv`` with a zero-width ``topk``: the comp loop never runs, and a
    zero-element ``comp_kv`` would arrive in the constant address space and
    fail the kernel's device pointer casts at compile time.

    Args:
        q: float16 or float32 ``[batch, n_heads, tokens, 512]`` query heads.
            Float32 queries round once to half up front, the identical
            rounding the scalar f32 consumer applies per row.
        raw_kv: float16 ``[batch, 1, tokens, 512]`` local KV rows.
        comp_kv: float16 ``[batch, n_comp, 512]`` pooled KV rows.
        topk: int32 ``[batch, tokens, width]`` ascending pool row ids.
        sinks: float32 ``[n_heads]`` attention sink logits.

    The model wrapper counts engagement. The ratio-4
    ``r4_prefill_consumer_*`` counters keep counting only ratio-4 layers.
    Requires the mma consumer; callers fail closed to the batched banded
    SDPA form when ``_prefill_live_mma_enabled()`` is off or the head count
    is not a multiple of 16.
    """
    import mlx.core as mx

    if q.ndim != 4 or raw_kv.ndim != 4 or comp_kv.ndim != 3:
        raise ValueError("q/raw_kv/comp_kv must have live BHLD/B1LD/BPD layouts")
    batch = int(q.shape[0])
    n_heads = int(q.shape[1])
    n_tokens = int(q.shape[2])
    head_dim = int(q.shape[3])
    if batch <= 0 or n_tokens <= 0:
        raise ValueError("q must contain at least one batch and token")
    if head_dim != 512:
        raise ValueError("banded mma attention requires head_dim=512")
    if n_heads % 16 != 0 or not _prefill_live_mma_enabled():
        raise ValueError("banded mma attention requires the mma consumer")
    if int(raw_kv.shape[0]) != batch or int(comp_kv.shape[0]) != batch:
        raise ValueError("raw_kv and comp_kv batch size must match q")
    if int(raw_kv.shape[-1]) != head_dim or int(comp_kv.shape[-1]) != head_dim:
        raise ValueError("raw_kv and comp_kv width must match q")
    if int(comp_kv.shape[1]) <= 0:
        raise ValueError("comp_kv must contain at least one row")
    if topk.ndim != 3 or int(topk.shape[0]) != batch or int(topk.shape[1]) != n_tokens:
        raise ValueError("topk must have shape [batch, tokens, width]")
    if q.dtype not in (mx.float16, mx.float32):
        raise ValueError("banded mma attention requires q to be float16 or float32")
    if raw_kv.dtype != mx.float16 or comp_kv.dtype != mx.float16:
        raise ValueError("raw_kv and comp_kv must be float16")
    if int(window) <= 0 or int(ratio) <= 0:
        raise ValueError("banded mma attention requires positive window and ratio")
    if topk.dtype != mx.int32:
        topk = topk.astype(mx.int32)
    if sinks.shape != (n_heads,):
        raise ValueError("sinks must have shape [n_heads]")
    if sinks.dtype != mx.float32:
        sinks = sinks.astype(mx.float32)
    if q.dtype != mx.float16:
        q = q.astype(mx.float16)

    raw_cap = int(raw_kv.shape[2])
    if raw_cap <= 0:
        raise ValueError("raw_kv must contain at least one row")
    meta = mx.array(
        [
            batch,
            n_tokens,
            n_heads,
            head_dim,
            raw_cap,
            raw_cap,
            0,
            int(comp_kv.shape[1]),
            int(topk.shape[2]),
            int(pos0),
            int(window),
            int(ratio),
        ],
        dtype=mx.uint32,
    )
    kernel = _get_prefill_live_mma_kernel(n_heads, head_dim)
    out, = kernel(
        inputs=[q, raw_kv, comp_kv, topk, sinks, meta],
        output_shapes=[(batch, n_heads, n_tokens, head_dim)],
        output_dtypes=[mx.float32],
        grid=(256 * n_tokens, n_heads // 16, batch),
        threadgroup=(256, 1, 1),
    )
    return out


def indexer_scores_tiled_live(
    q,
    weights,
    index_comp,
    *,
    pos0: int,
    ratio: int = 4,
):
    """Compute DS4 ratio-4 indexer scores with a tiled Metal probe kernel.

    Args:
        q: float32 or float16 ``[batch, n_heads, tokens, 128]`` QAT indexer
            queries. The kernel stages q tiles as half either way; float16
            operands carry the identical staged values at half the device
            read width.
        weights: float32 ``[batch, tokens, n_heads]`` head weights including
            the DS4 scale factors.
        index_comp: float32 or float16 ``[batch, n_comp, 128]`` QAT
            compressed index rows; the dtype must match ``q``.
    """
    import mlx.core as mx

    if q.ndim != 4 or weights.ndim != 3 or index_comp.ndim != 3:
        raise ValueError("q, weights, and index_comp must be BHLD/BLH/BPD")
    batch = int(q.shape[0])
    n_heads = int(q.shape[1])
    n_tokens = int(q.shape[2])
    head_dim = int(q.shape[3])
    n_comp = int(index_comp.shape[1])
    if batch <= 0 or n_tokens <= 0:
        raise ValueError("q must contain at least one batch and token")
    if head_dim != 128:
        raise ValueError("DS4 indexer score probe requires head_dim=128")
    if int(index_comp.shape[0]) != batch or int(index_comp.shape[2]) != head_dim:
        raise ValueError("index_comp must have shape [batch, n_comp, 128]")
    if weights.shape != (batch, n_tokens, n_heads):
        raise ValueError("weights must have shape [batch, tokens, n_heads]")
    if weights.dtype != mx.float32:
        raise ValueError("weights must be float32")
    if q.dtype == mx.float16 or index_comp.dtype == mx.float16:
        if q.dtype != mx.float16 or index_comp.dtype != mx.float16:
            raise ValueError(
                "half score operands require both q and index_comp as float16")
        operand_form = "f16"
    elif q.dtype == mx.float32 and index_comp.dtype == mx.float32:
        operand_form = "f32"
    else:
        raise ValueError(
            "q and index_comp must both be float32 or both be float16")
    if n_comp <= 0:
        raise ValueError("index_comp must contain at least one compressed row")
    _SCORES_TILED_CALL_COUNTS[operand_form] += 1

    meta = mx.array(
        [
            batch,
            n_tokens,
            n_heads,
            head_dim,
            n_comp,
            int(pos0),
            int(ratio),
        ],
        dtype=mx.uint32,
    )
    kernel = _get_indexer_score_kernel(n_heads, head_dim, operand_form)
    scores, = kernel(
        inputs=[q, weights, index_comp, meta],
        output_shapes=[(batch, n_tokens, n_comp)],
        output_dtypes=[mx.float32],
        grid=(128 * ((n_comp + 31) // 32), (n_tokens + 7) // 8, batch),
        threadgroup=(128, 1, 1),
    )
    return scores


def _get_indexer_q_qat_v2_kernel(n_heads: int, head_dim: int):
    key = (int(n_heads), int(head_dim))
    cached = _INDEXER_Q_QAT_V2_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=f"moespresso_dsv4_indexer_q_qat_v2_h{n_heads}_d{head_dim}",
        input_names=["q", "meta"],
        output_names=["out"],
        source=_INDEXER_Q_QAT_V2_SOURCE,
        ensure_row_contiguous=False,
    )
    _INDEXER_Q_QAT_V2_KERNEL_CACHE[key] = kernel
    return kernel


def _indexer_q_qat_v2_enabled() -> bool:
    """Gate for the simdgroup-shuffle QAT restructure (bit-identical math)."""
    import os

    return os.environ.get("MOESPRESSO_DSV4_R4_PREFILL_QAT_V2", "1") != "0"


def indexer_q_qat_live(q):
    """Apply DS4 indexer Hadamard-128 + FP4 QAT to live BHLD q rows."""
    import mlx.core as mx

    if q.ndim != 4:
        raise ValueError("q must have shape [batch, n_heads, tokens, 128]")
    batch = int(q.shape[0])
    n_heads = int(q.shape[1])
    n_tokens = int(q.shape[2])
    head_dim = int(q.shape[3])
    if batch <= 0 or n_tokens <= 0:
        raise ValueError("q must contain at least one batch and token")
    if head_dim != 128:
        raise ValueError("DS4 indexer QAT probe requires head_dim=128")
    if q.dtype != mx.float32:
        raise ValueError("q must be float32")

    meta = mx.array([batch, n_heads, n_tokens, head_dim], dtype=mx.uint32)
    n_rows = batch * n_heads * n_tokens
    if _indexer_q_qat_v2_enabled():
        kernel = _get_indexer_q_qat_v2_kernel(n_heads, head_dim)
        groups = (n_rows + 3) // 4
        out, = kernel(
            inputs=[q, meta],
            output_shapes=[(batch, n_heads, n_tokens, head_dim)],
            output_dtypes=[mx.float32],
            grid=(128 * groups, 1, 1),
            threadgroup=(128, 1, 1),
        )
        return out
    kernel = _get_indexer_q_qat_kernel(n_heads, head_dim)
    out, = kernel(
        inputs=[q, meta],
        output_shapes=[(batch, n_heads, n_tokens, head_dim)],
        output_dtypes=[mx.float32],
        grid=(128 * n_rows, 1, 1),
        threadgroup=(128, 1, 1),
    )
    return out


def indexed_mixed_attention_prefill(
    q,
    raw_kv,
    comp_kv,
    topk,
    sinks,
    *,
    pos0: int,
    window: int = 128,
    ratio: int = 4,
    raw_start: int = 0,
):
    """Run a DS4-c-shaped batch-token indexed mixed-attention probe kernel.

    Args:
        q: float32 ``[tokens, n_heads, 512]`` query heads.
        raw_kv: float16 ``[raw_cap, 512]`` local raw KV rows.
        comp_kv: float16 ``[n_comp, 512]`` compressed KV rows.
        topk: int32 ``[tokens, top_k]`` selected compressed row ids.
        sinks: float32 ``[n_heads]`` attention sink logits.
    """
    import mlx.core as mx

    if q.ndim != 3 or raw_kv.ndim != 2 or comp_kv.ndim != 2:
        raise ValueError("q must be rank-3 and raw_kv/comp_kv rank-2")
    n_tokens = int(q.shape[0])
    n_heads = int(q.shape[1])
    head_dim = int(q.shape[2])
    if n_tokens <= 0:
        raise ValueError("q must contain at least one token")
    if head_dim != 512:
        raise ValueError("DS4 indexed prefill probe currently requires head_dim=512")
    if n_heads % 8 != 0:
        raise ValueError("DS4 indexed prefill probe requires n_heads to be divisible by 8")
    if int(raw_kv.shape[1]) != head_dim or int(comp_kv.shape[1]) != head_dim:
        raise ValueError("raw_kv and comp_kv width must match q")
    if topk.ndim != 2 or int(topk.shape[0]) != n_tokens:
        raise ValueError("topk must have shape [tokens, top_k]")
    if sinks.shape != (n_heads,):
        raise ValueError("sinks must have shape [n_heads]")
    if q.dtype != mx.float32:
        raise ValueError("q must be float32, matching DS4-c's q buffer contract")
    if raw_kv.dtype != mx.float16 or comp_kv.dtype != mx.float16:
        raise ValueError("raw_kv and comp_kv must be float16")
    if topk.dtype != mx.int32:
        topk = topk.astype(mx.int32)
    if sinks.dtype != mx.float32:
        sinks = sinks.astype(mx.float32)

    raw_cap = int(raw_kv.shape[0])
    if raw_cap <= 0:
        raise ValueError("raw_kv must contain at least one row")
    if not (0 <= int(raw_start) < raw_cap):
        raise ValueError("raw_start must be inside raw_kv")
    meta = mx.array(
        [
            n_tokens,
            n_heads,
            head_dim,
            raw_cap,
            raw_cap,
            int(raw_start),
            int(comp_kv.shape[0]),
            int(topk.shape[1]),
            int(pos0),
            int(window),
            int(ratio),
        ],
        dtype=mx.uint32,
    )
    kernel = _get_prefill_kernel(n_heads, head_dim)
    out, = kernel(
        inputs=[q, raw_kv, comp_kv, topk, sinks, meta],
        output_shapes=[(n_tokens, n_heads, head_dim)],
        output_dtypes=[mx.float32],
        grid=(32 * n_tokens, 8 * (n_heads // 8), 1),
        threadgroup=(32, 8, 1),
    )
    return out


def mlx_selected_rows_attention_reference(
    q,
    raw_kv,
    comp_kv,
    topk,
    sinks,
):
    """Current MLX-style selected-row consumer used as correctness/speed reference."""
    import mlx.core as mx

    selected = mx.take(comp_kv, topk.astype(mx.int32), axis=0)
    full = mx.concatenate([raw_kv, selected], axis=0)
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float16)[None, :, None, :],
        full[None, None, :, :],
        full[None, None, :, :],
        scale=float(q.shape[-1]) ** -0.5,
        mask=None,
        sinks=sinks.astype(mx.float16),
    ).reshape(q.shape).astype(mx.float32)


def mlx_indexed_mixed_attention_prefill_reference(
    q,
    raw_kv,
    comp_kv,
    topk,
    sinks,
    *,
    pos0: int,
    window: int = 128,
    ratio: int = 4,
):
    """Reference for the indexed prefill probe using MLX SDPA per token."""
    import mlx.core as mx

    n_tokens, n_heads, head_dim = q.shape
    n_raw = int(raw_kv.shape[0])
    raw_last_pos = int(pos0) + int(n_tokens) - 1
    first_raw_pos = raw_last_pos + 1 - n_raw
    rows = []
    for token in range(int(n_tokens)):
        qpos = int(pos0) + token
        window_first = qpos + 1 - int(window) if window and qpos + 1 > window else 0
        first = max(first_raw_pos, window_first)
        last = min(qpos, raw_last_pos)
        parts = []
        if first <= last:
            raw_idx = mx.arange(first - first_raw_pos, last - first_raw_pos + 1)
            parts.append(mx.take(raw_kv, raw_idx.astype(mx.int32), axis=0))
        visible = min((qpos + 1) // int(ratio), int(comp_kv.shape[0]))
        selected_np = np.asarray(topk[token], dtype=np.int32)
        selected_np = selected_np[(selected_np >= 0) & (selected_np < visible)]
        if selected_np.size:
            selected = mx.array(selected_np, dtype=mx.int32)
            parts.append(mx.take(comp_kv, selected, axis=0))
        if parts:
            full = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        else:
            full = mx.zeros((0, head_dim), dtype=mx.float16)
        out = mx.fast.scaled_dot_product_attention(
            q[token].astype(mx.float16)[None, :, None, :],
            full[None, None, :, :],
            full[None, None, :, :],
            scale=float(head_dim) ** -0.5,
            mask=None,
            sinks=sinks.astype(mx.float16),
        ).reshape(n_heads, head_dim).astype(mx.float32)
        rows.append(out)
    return mx.stack(rows, axis=0)


@dataclass(frozen=True)
class IndexedAttentionProbeCase:
    name: str
    compressed_rows: int
    topk_rows: int = 512
    raw_rows: int = 128


def default_probe_cases() -> tuple[IndexedAttentionProbeCase, ...]:
    return (
        IndexedAttentionProbeCase("bounded_long_rows961", 961),
        IndexedAttentionProbeCase("q3_scale_rows7618", 7618),
    )


def _inputs_for_case(case: IndexedAttentionProbeCase, *, seed: int):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((64, 512), dtype=np.float32))
    raw = mx.array(
        rng.standard_normal((case.raw_rows, 512), dtype=np.float32)
    ).astype(mx.float16)
    comp = mx.array(
        rng.standard_normal((case.compressed_rows, 512), dtype=np.float32)
    ).astype(mx.float16)
    topk = mx.array(np.arange(case.topk_rows, dtype=np.int32))
    sinks = mx.array(rng.standard_normal((64,), dtype=np.float32))
    pos0 = max(
        case.raw_rows - 1,
        case.compressed_rows * 4 + case.raw_rows - 1,
    )
    mx.eval(q, raw, comp, topk, sinks)
    return q, raw, comp, topk, sinks, pos0


def _time_call(fn: Callable[[], Any], *, repeats: int, warmup: int) -> float:
    import mlx.core as mx

    for _ in range(max(int(warmup), 0)):
        y = fn()
        mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(max(int(repeats), 1)):
        y = fn()
        mx.eval(y)
    return (time.perf_counter() - t0) / max(int(repeats), 1)


def run_indexed_attention_probe(
    *,
    repeats: int = 20,
    warmup: int = 3,
    cases: Sequence[IndexedAttentionProbeCase] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    import mlx.core as mx

    rows = []
    for i, case in enumerate(cases or default_probe_cases()):
        q, raw, comp, topk, sinks, pos0 = _inputs_for_case(case, seed=seed + i)
        candidate = indexed_mixed_attention_decode(
            q,
            raw,
            comp,
            topk,
            sinks,
            pos0=pos0,
        )
        reference = mlx_selected_rows_attention_reference(q, raw, comp, topk, sinks)
        mx.eval(candidate, reference)
        diff = mx.abs(candidate - reference)
        ref_scale = mx.maximum(mx.sqrt(mx.mean(reference * reference)), mx.array(1e-12))
        max_abs = float(mx.max(diff).item())
        rel_rms = float((mx.sqrt(mx.mean(diff * diff)) / ref_scale).item())
        kernel_s = _time_call(
            lambda: indexed_mixed_attention_decode(q, raw, comp, topk, sinks, pos0=pos0),
            repeats=repeats,
            warmup=warmup,
        )
        mlx_s = _time_call(
            lambda: mlx_selected_rows_attention_reference(q, raw, comp, topk, sinks),
            repeats=repeats,
            warmup=warmup,
        )
        rows.append({
            "case": case.name,
            "raw_rows": int(case.raw_rows),
            "compressed_rows": int(case.compressed_rows),
            "topk_rows": int(case.topk_rows),
            "pos0": int(pos0),
            "max_abs": max_abs,
            "rel_rms": rel_rms,
            "kernel_seconds_per_repeat": float(kernel_s),
            "mlx_seconds_per_repeat": float(mlx_s),
            "kernel_over_mlx": float(kernel_s / mlx_s) if mlx_s else None,
        })
    return {
        "metric": "ds4_indexed_mixed_attention_probe",
        "units": "seconds per one-token indexed mixed-attention call",
        "quality_note": (
            "isolated speed/correctness probe only; served integration still "
            "requires Q1/Q2/Q3 validation"
        ),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "cases": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-indexed-attention-probe",
        description=(
            "Benchmark a DS4-c-shaped one-token indexed mixed-attention kernel "
            "against the current MLX selected-row SDPA consumer."
        ),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compressed-rows",
        type=int,
        action="append",
        help="Custom compressed row count; may be passed multiple times.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    cases = None
    if args.compressed_rows:
        cases = tuple(
            IndexedAttentionProbeCase(f"rows{int(rows)}", int(rows))
            for rows in args.compressed_rows
        )
    payload = run_indexed_attention_probe(
        repeats=args.repeats,
        warmup=args.warmup,
        cases=cases,
        seed=args.seed,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
