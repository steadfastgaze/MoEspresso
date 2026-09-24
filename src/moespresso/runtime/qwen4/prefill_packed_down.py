"""Packed IQ_K down projection for Qwen routed prefill.

The kernel reconstructs one 32-row by 64-column weight tile in threadgroup
memory and reuses it across an expert's routed-token tile.  It reads the
physical 768-column packed rows but multiplies only the 640 logical SwiGLU
columns; the omitted padded activation columns are zero.  No decoded
expert-sized weight array exists.

``packed_down`` accepts route-specific FP16 activations shaped
``indices.shape + (640,)`` and returns FP16 projections shaped
``indices.shape + (2560,)``.  The expert-tile matrix schedule adapts the Qwen
routed-prefill kernel from the MIT-licensed ``antirez/ds4`` project.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx
from mlx_iqk.format import SUB_WEIGHTS
from mlx_iqk.kernels import (
    _iq3_reads,
    _iq3_value_expr,
    _quad_load,
    _scale_block,
    _value_expr,
)
from mlx_iqk.nn import IqkSwitchLinear, member_table
from mlx_iqk.routed import DOWN_STORED, HIDDEN, INTERMEDIATE

from moespresso.runtime.qwen4.prefill_packed import _schedule_kernel


_THREADS = 128
_ROWS = 32
_K_TILE = 64
_MEMBERS = ("iq2_k", "iq2_ks", "iq3_k")


def _stream_names(member: str) -> list[str]:
    if member == "iq2_ks":
        streams = ("qs", "scl", "sch", "sex", "dv")
    elif member == "iq3_k":
        streams = ("qs", "qh", "scl", "sch", "sex", "dv")
    elif member == "iq2_k":
        streams = ("qs", "scl", "sex", "dv")
    else:
        raise ValueError(f"packed down prefill does not support {member!r}")
    return [f"d_{name}" for name in streams]


def _scale_source(member: str) -> str:
    source = _scale_block(member, DOWN_STORED, "rid", "kg")
    for name in ("scl", "sch", "sex", "dv"):
        source = source.replace(f"{name}[", f"d_{name}[")
    return source


def _stage_source(member: str) -> str:
    """Generate exact FP16 reconstruction of sixteen packed down weights."""
    if member == "iq3_k":
        reads = _iq3_reads(DOWN_STORED, "rid", "kg")
        for name in ("qs", "qh", "scl", "sch", "sex", "dv"):
            reads = reads.replace(f"{name}[", f"d_{name}[")
        reads = reads.replace("(qs +", "(d_qs +")
        low = "\n".join(
            f"dw[{i}] = half(dl0_ * {_iq3_value_expr(0, i)});" for i in range(16)
        )
        high = "\n".join(
            f"dw[{i}] = half(dl1_ * {_iq3_value_expr(1, i + 16)});"
            for i in range(16)
        )
        return f"""
    {{
{reads}
        if ((q & 1u) == 0u) {{
{low}
        }} else {{
{high}
        }}
    }}
"""
    dual = SUB_WEIGHTS[member] == 16
    halves = (0, 1) if dual else (0,)
    quads = _quad_load(halves)
    low = "\n".join(
        f"dw[{i}] = half(dl0_ * {_value_expr(0, i)});" for i in range(16)
    )
    high_half = 1 if dual else 0
    high_scale = "dl1_" if dual else "dl0_"
    high = "\n".join(
        f"dw[{i}] = half({high_scale} * {_value_expr(high_half, i + 16)});"
        for i in range(16)
    )
    return f"""
    {{
        const device uint2* qp_ = (const device uint2*)
            (d_qs + rid * {DOWN_STORED // 16}ul + (ulong)(kg << 1u));
        uint2 qw_ = qp_[0];
        uint q0_ = qw_.x;
        uint q1_ = qw_.y;
{_scale_source(member)}
{quads}
        if ((q & 1u) == 0u) {{
{low}
        }} else {{
{high}
        }}
    }}
"""


def _source(member: str, tile_tokens: int) -> str:
    token_matrices = tile_tokens // 8
    stage = _stage_source(member)
    return f"""
    uint rb = threadgroup_position_in_grid.x;
    uint schedule_slot = threadgroup_position_in_grid.y;
    if (schedule_slot >= schedule_total[0]) return;

    uint tid = thread_position_in_threadgroup.x;
    uint simdgroup = simdgroup_index_in_threadgroup;
    uint expert = schedule[schedule_slot * 3u + 0u];
    uint route_start = schedule[schedule_slot * 3u + 1u];
    uint route_count = schedule[schedule_slot * 3u + 2u];
    uint row0 = rb * {_ROWS}u;

    threadgroup half As[{_ROWS * _K_TILE}];
    threadgroup half Bs[{_K_TILE * tile_tokens}];
    threadgroup half tgV[{16 if member == 'iq3_k' else 8}];
    threadgroup float Cs[4][64];
    if (tid < {16 if member == 'iq3_k' else 8}u) tgV[tid] = vtab[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 C[{token_matrices}];
    for (uint nt = 0u; nt < {token_matrices}u; ++nt) {{
        C[nt] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }}

    uint my_token = tid % {tile_tokens}u;
    uint pair = my_token < route_count ? order[route_start + my_token] : 0u;
    for (uint kb = 0u; kb < {INTERMEDIATE // _K_TILE}u; ++kb) {{
        uint r = tid >> 2u;
        uint q = tid & 3u;
        uint kg = kb * 2u + (q >> 1u);
        ulong rid = (ulong)expert * {HIDDEN}ul + row0 + r;
        threadgroup half* dw = As + r * {_K_TILE}u + q * 16u;
{stage}

        uint activation_token = tid % {tile_tokens}u;
        uint activation_slice = tid / {tile_tokens}u;
        constexpr uint values_per_thread = {_K_TILE * tile_tokens // _THREADS};
        ulong xbase = (ulong)pair * {INTERMEDIATE}ul + kb * {_K_TILE}u
                    + activation_slice * values_per_thread;
        for (uint j = 0u; j < values_per_thread; ++j) {{
            Bs[(activation_slice * values_per_thread + j) * {tile_tokens}u
               + activation_token] = my_token < route_count ? x[xbase + j] : half(0.0f);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint sub = 0u; sub < {_K_TILE // 8}u; ++sub) {{
            simdgroup_half8x8 a;
            simdgroup_half8x8 b;
            simdgroup_load(a, As + simdgroup * 8u * {_K_TILE}u + sub * 8u,
                           {_K_TILE}, 0, false);
            for (uint nt = 0u; nt < {token_matrices}u; ++nt) {{
                simdgroup_load(b, Bs + sub * 8u * {tile_tokens}u + nt * 8u,
                               {tile_tokens}, 0, false);
                simdgroup_multiply_accumulate(C[nt], a, b, C[nt]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (uint nt = 0u; nt < {token_matrices}u; ++nt) {{
        simdgroup_store(C[nt], Cs[simdgroup], 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < 256u; index += {_THREADS}u) {{
            uint sg = index >> 6u;
            uint element = index & 63u;
            uint local_row = (element >> 3u) + sg * 8u;
            uint local_token = nt * 8u + (element & 7u);
            if (local_token < route_count) {{
                uint output_pair = order[route_start + local_token];
                ulong row = row0 + local_row;
                out[(ulong)output_pair * {HIDDEN}ul + row] = half(Cs[sg][element]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""


@cache
def _kernel(member: str, tile_tokens: int):
    return mx.fast.metal_kernel(
        name=f"qwen4_iqk_prefill_down_{member}_t{tile_tokens}",
        input_names=[
            "x", *_stream_names(member), "vtab", "order", "schedule", "schedule_total",
        ],
        output_names=["out"],
        source=_source(member, tile_tokens),
    )


def packed_down(
    down: IqkSwitchLinear,
    activation: mx.array,
    indices: mx.array,
    *,
    tile_tokens: int = 32,
) -> mx.array:
    """Return route-ordered FP16 down projections from packed IQ_K weights."""
    if not isinstance(down, IqkSwitchLinear):
        raise ValueError("packed down prefill requires an IQ_K projection")
    if down.member not in _MEMBERS:
        raise ValueError(f"packed down prefill does not support {down.member!r}")
    if (down.in_features, down.out_features) != (DOWN_STORED, HIDDEN):
        raise ValueError("packed down prefill requires the Qwen 768x2560 routed geometry")
    if not 0 < down.num_experts <= 1024:
        raise ValueError("packed down prefill requires at most 1024 experts")
    if tile_tokens not in (8, 16, 32):
        raise ValueError("tile_tokens must be 8, 16, or 32")
    expected = tuple(indices.shape) + (INTERMEDIATE,)
    if tuple(activation.shape) != expected:
        raise ValueError(f"activation must have shape {expected}, got {tuple(activation.shape)}")

    flat_ids = indices.reshape(-1).astype(mx.uint32)
    routes = int(flat_ids.size)
    order = mx.argsort(flat_ids).astype(mx.uint32)
    sorted_ids = flat_ids[order]
    max_schedule = down.num_experts + (routes + tile_tokens - 1) // tile_tokens
    dims = mx.array([routes], dtype=mx.uint32)
    schedule, schedule_total = _schedule_kernel(down.num_experts, tile_tokens)(
        inputs=[sorted_ids, dims],
        grid=(down.num_experts, 1, 1),
        threadgroup=(down.num_experts, 1, 1),
        output_shapes=[(max_schedule, 3), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    output = _kernel(down.member, tile_tokens)(
        inputs=[
            activation.reshape(routes, INTERMEDIATE).astype(mx.float16),
            *down._streams(),
            member_table(down.member),
            order,
            schedule,
            schedule_total,
        ],
        grid=(HIDDEN // _ROWS * _THREADS, max_schedule, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(routes, HIDDEN)],
        output_dtypes=[mx.float16],
    )[0]
    return output.reshape(*indices.shape, HIDDEN)


__all__ = ["packed_down"]
