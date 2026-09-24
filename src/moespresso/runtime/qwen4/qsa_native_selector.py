"""Stable Metal selector for trusted one-token Qwen sparse attention."""

from __future__ import annotations

from functools import cache as memoize

import mlx.core as mx


QSA_NATIVE_SELECTOR_MIN_GROUPS = 512
QSA_NATIVE_SELECTOR_MAX_GROUPS = 4_096
QSA_NATIVE_SELECTOR_BLOCK_BUDGET = 512
QSA_NATIVE_SELECTOR_COMPRESS_RATIO = 4
QSA_NATIVE_SELECTOR_WIDTH = (
    QSA_NATIVE_SELECTOR_BLOCK_BUDGET * QSA_NATIVE_SELECTOR_COMPRESS_RATIO
    + QSA_NATIVE_SELECTOR_COMPRESS_RATIO
    - 1
)
_THREADS = 512


_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint group_count = (uint)scores_shape[2];
    uint visible_count = (uint)visible_tokens[0];

    // Nonnegative IEEE-754 values retain their order as unsigned integers.
    // The inverted physical id makes the lower id win an exact score tie.
    threadgroup ulong ordered[4096];
    for (uint part = 0; part < 8u; ++part) {
        uint slot = tid + part * 512u;
        if (slot < group_count) {
            uint score_bits = as_type<uint>(scores[slot]);
            ordered[slot] = ((ulong)score_bits << 32u) |
                (ulong)(0xffffffffu - slot);
        } else {
            ordered[slot] = 0ul;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint width = 2u; width <= 4096u; width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            for (uint part = 0; part < 8u; ++part) {
                uint left = tid + part * 512u;
                uint right = left ^ stride;
                if (right <= left) continue;
                ulong left_key = ordered[left];
                ulong right_key = ordered[right];
                bool ascending = (left & width) == 0u;
                bool swap = ascending ? right_key < left_key : left_key < right_key;
                if (swap) {
                    ordered[left] = right_key;
                    ordered[right] = left_key;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    ulong selected_key = ordered[3584u + tid];
    ordered[tid] = (ulong)(0xffffffffu - (uint)selected_key);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Restore the physical context order consumed by released attention.
    for (uint width = 2u; width <= 512u; width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            uint right = tid ^ stride;
            if (right > tid) {
                ulong left_id = ordered[tid];
                ulong right_id = ordered[right];
                bool ascending = (tid & width) == 0u;
                bool swap = ascending ? right_id < left_id : left_id < right_id;
                if (swap) {
                    ordered[tid] = right_id;
                    ordered[right] = left_id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    int group = (int)ordered[tid];
    for (uint lane = 0; lane < 4u; ++lane) {
        selected[tid * 4u + lane] = group * 4 + (int)lane;
    }
    if (tid < 3u) {
        uint tail = group_count * 4u + tid;
        selected[2048u + tid] = tail < visible_count ? (int)tail : -1;
    }
"""


@memoize
def _kernel():
    if not mx.metal.is_available():
        raise RuntimeError("native QSA selection requires Metal")
    return mx.fast.metal_kernel(
        name="moespresso_qwen38_qsa_stable_select_g4096_k512",
        input_names=["scores", "visible_tokens"],
        output_names=["selected"],
        source=_SOURCE,
    )


def native_qsa_selector_eligible(
    scores: mx.array,
    *,
    visible_count: int,
    token_budget: int,
    compress_ratio: int,
) -> bool:
    """Return whether the fixed trusted-decode selector accepts this step."""

    group_count = scores.shape[-1] if scores.ndim == 3 else 0
    return bool(
        scores.shape[:2] == (1, 1)
        and scores.dtype == mx.float32
        and QSA_NATIVE_SELECTOR_MIN_GROUPS <= group_count <= QSA_NATIVE_SELECTOR_MAX_GROUPS
        and token_budget == QSA_NATIVE_SELECTOR_BLOCK_BUDGET * compress_ratio
        and compress_ratio == QSA_NATIVE_SELECTOR_COMPRESS_RATIO
        and not isinstance(visible_count, bool)
        and isinstance(visible_count, int)
        and visible_count // compress_ratio == group_count
        and 0 <= visible_count - group_count * compress_ratio < compress_ratio
    )


def native_qsa_selected_token_indices(
    scores: mx.array,
    *,
    visible_count: int,
    token_budget: int,
    compress_ratio: int,
) -> mx.array:
    """Return the released fixed-width selection for an all-visible prefix."""

    if not native_qsa_selector_eligible(
        scores,
        visible_count=visible_count,
        token_budget=token_budget,
        compress_ratio=compress_ratio,
    ):
        raise ValueError("native QSA selector received an incompatible step")
    (selected,) = _kernel()(
        inputs=[
            mx.contiguous(scores),
            mx.array([visible_count], dtype=mx.uint32),
        ],
        output_shapes=[(1, 1, QSA_NATIVE_SELECTOR_WIDTH)],
        output_dtypes=[mx.int32],
        grid=(_THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    return selected
