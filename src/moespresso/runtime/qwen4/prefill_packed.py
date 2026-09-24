"""Packed IQ_K gate/up projection for Qwen routed prefill.

The kernel reconstructs one 32-row by 64-column weight tile in threadgroup
memory and reuses it across an expert's routed-token tile.  Gate and up share
the activation tile and matrix schedule, then round their projections through
FP16 before the SwiGLU epilogue.  No decoded expert-sized weight array exists.
The expert-tile matrix schedule adapts the Qwen routed-prefill kernel from the
MIT-licensed ``antirez/ds4`` project.

``packed_gate_up`` accepts hidden rows shaped ``[T, 2560]`` or
``[1, T, 2560]`` and source expert IDs shaped ``[T, 10]`` or ``[1, T, 10]``.
It returns FP16 activations shaped ``indices.shape + (640,)``.  One indices
array addresses both projection pools, so gate and up must use the same
source-to-slot mapping.  The full-resident identity mapping satisfies this
contract even when the two projections use distinct packed stream arrays.
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
from mlx_iqk.routed import HIDDEN, INTERMEDIATE, sigmoid_fp16_table


_THREADS = 128
_ROWS = 32
_K_TILE = 64
_MEMBERS = ("iq2_k", "iq2_ks", "iq3_k")


def _projection_stream_names(prefix: str, member: str) -> list[str]:
    if member == "iq2_ks":
        streams = ("qs", "scl", "sch", "sex", "dv")
    elif member == "iq3_k":
        streams = ("qs", "qh", "scl", "sch", "sex", "dv")
    elif member == "iq2_k":
        streams = ("qs", "scl", "sex", "dv")
    else:
        raise ValueError(f"packed prefill does not support {member!r}")
    return [f"{prefix}_{name}" for name in streams]


def _prefixed_scale(prefix: str, member: str) -> str:
    source = _scale_block(member, HIDDEN, "rid", "kg")
    for name in ("scl", "sch", "sex", "dv"):
        source = source.replace(f"{name}[", f"{prefix}_{name}[")
    return source


def _stage_projection(prefix: str, member: str, destination: str) -> str:
    """Generate one projection's exact FP16 reconstruction into a K tile."""
    if member == "iq3_k":
        reads = _iq3_reads(HIDDEN, "rid", "kg")
        for name in ("qs", "qh", "scl", "sch", "sex", "dv"):
            reads = reads.replace(f"{name}[", f"{prefix}_{name}[")
        reads = reads.replace("(qs +", f"({prefix}_qs +")
        low = "\n".join(
            f"{destination}[{i}] = half(dl0_ * {_iq3_value_expr(0, i)});"
            for i in range(16)
        )
        high = "\n".join(
            f"{destination}[{i}] = half(dl1_ * {_iq3_value_expr(1, i + 16)});"
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
        f"{destination}[{i}] = half(dl0_ * {_value_expr(0, i)});"
        for i in range(16)
    )
    high_half = 1 if dual else 0
    high_scale = "dl1_" if dual else "dl0_"
    high = "\n".join(
        f"{destination}[{i}] = half({high_scale} * {_value_expr(high_half, i + 16)});"
        for i in range(16)
    )
    return f"""
    {{
        const device uint2* qp_ = (const device uint2*)
            ({prefix}_qs + rid * {HIDDEN // 16}ul + (ulong)(kg << 1u));
        uint2 qw_ = qp_[0];
        uint q0_ = qw_.x;
        uint q1_ = qw_.y;
{_prefixed_scale(prefix, member)}
{quads}
        if ((q & 1u) == 0u) {{
{low}
        }} else {{
{high}
        }}
    }}
"""


@cache
def _schedule_kernel(num_experts: int, tile_tokens: int):
    source = f"""
    uint expert = thread_position_in_threadgroup.x;
    uint n = dims[0];
    threadgroup uint starts[{num_experts}];
    threadgroup uint tiles[{num_experts}];
    threadgroup uint offsets[{num_experts}];

    uint lo = 0u;
    uint hi = n;
    while (lo < hi) {{
        uint mid = lo + ((hi - lo) >> 1u);
        if (sorted_ids[mid] < expert) lo = mid + 1u;
        else hi = mid;
    }}
    uint start = lo;
    hi = n;
    while (lo < hi) {{
        uint mid = lo + ((hi - lo) >> 1u);
        if (sorted_ids[mid] <= expert) lo = mid + 1u;
        else hi = mid;
    }}
    uint count = lo - start;
    starts[expert] = start;
    tiles[expert] = (count + {tile_tokens - 1}u) / {tile_tokens}u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (expert == 0u) {{
        uint total = 0u;
        for (uint e = 0u; e < {num_experts}u; ++e) {{
            offsets[e] = total;
            total += tiles[e];
        }}
        schedule_total[0] = total;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile = 0u; tile < tiles[expert]; ++tile) {{
        uint slot = offsets[expert] + tile;
        uint tile_start = starts[expert] + tile * {tile_tokens}u;
        uint remaining = start + count - tile_start;
        schedule[slot * 3u + 0u] = expert;
        schedule[slot * 3u + 1u] = tile_start;
        schedule[slot * 3u + 2u] = min(remaining, {tile_tokens}u);
    }}
"""
    return mx.fast.metal_kernel(
        name=f"qwen4_iqk_prefill_schedule_e{num_experts}_t{tile_tokens}",
        input_names=["sorted_ids", "dims"],
        output_names=["schedule", "schedule_total"],
        source=source,
    )


def _main_source(member: str, tile_tokens: int, top_k: int, debug: bool) -> str:
    token_matrices = tile_tokens // 8
    gate_stage = _stage_projection("g", member, "dg")
    up_stage = _stage_projection("u", member, "du")
    if debug:
        store = """
                gate_out[(ulong)output_pair * 640ul + row] = half(g);
                up_out[(ulong)output_pair * 640ul + row] = half(u);
"""
    else:
        store = """
                half gh = half(g);
                half uh = half(u);
                half silu = gh * sigtab[as_type<ushort>(gh)];
                out[(ulong)output_pair * 640ul + row] = silu * uh;
"""
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

    threadgroup half Ag[{_ROWS * _K_TILE}];
    threadgroup half Au[{_ROWS * _K_TILE}];
    threadgroup half Bs[{_K_TILE * tile_tokens}];
    threadgroup half tgV[{16 if member == 'iq3_k' else 8}];
    threadgroup float Cs[4][2][64];
    if (tid < {16 if member == 'iq3_k' else 8}u) tgV[tid] = vtab[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 Cg[{token_matrices}];
    simdgroup_float8x8 Cu[{token_matrices}];
    for (uint nt = 0u; nt < {token_matrices}u; ++nt) {{
        Cg[nt] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        Cu[nt] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }}

    uint my_token = tid % {tile_tokens}u;
    uint pair = my_token < route_count ? order[route_start + my_token] : 0u;
    uint token = pair / {top_k}u;
    for (uint kb = 0u; kb < {HIDDEN // _K_TILE}u; ++kb) {{
        uint r = tid >> 2u;
        uint q = tid & 3u;
        uint kg = kb * 2u + (q >> 1u);
        ulong rid = (ulong)expert * {INTERMEDIATE}ul + row0 + r;
        threadgroup half* dg = Ag + r * {_K_TILE}u + q * 16u;
        threadgroup half* du = Au + r * {_K_TILE}u + q * 16u;
{gate_stage}
{up_stage}

        uint activation_token = tid % {tile_tokens}u;
        uint activation_slice = tid / {tile_tokens}u;
        constexpr uint values_per_thread = {_K_TILE * tile_tokens // _THREADS};
        ulong xbase = (ulong)token * {HIDDEN}ul + kb * {_K_TILE}u
                    + activation_slice * values_per_thread;
        for (uint j = 0u; j < values_per_thread; ++j) {{
            Bs[(activation_slice * values_per_thread + j) * {tile_tokens}u
               + activation_token] = my_token < route_count ? x[xbase + j] : half(0.0f);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint sub = 0u; sub < {_K_TILE // 8}u; ++sub) {{
            simdgroup_half8x8 ag;
            simdgroup_half8x8 au;
            simdgroup_half8x8 b;
            simdgroup_load(ag, Ag + simdgroup * 8u * {_K_TILE}u + sub * 8u,
                           {_K_TILE}, 0, false);
            simdgroup_load(au, Au + simdgroup * 8u * {_K_TILE}u + sub * 8u,
                           {_K_TILE}, 0, false);
            for (uint nt = 0u; nt < {token_matrices}u; ++nt) {{
                simdgroup_load(b, Bs + sub * 8u * {tile_tokens}u + nt * 8u,
                               {tile_tokens}, 0, false);
                simdgroup_multiply_accumulate(Cg[nt], ag, b, Cg[nt]);
                simdgroup_multiply_accumulate(Cu[nt], au, b, Cu[nt]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (uint nt = 0u; nt < {token_matrices}u; ++nt) {{
        simdgroup_store(Cg[nt], Cs[simdgroup][0], 8, 0, false);
        simdgroup_store(Cu[nt], Cs[simdgroup][1], 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < 256u; index += {_THREADS}u) {{
            uint sg = index >> 6u;
            uint element = index & 63u;
            uint local_row = (element >> 3u) + sg * 8u;
            uint local_token = nt * 8u + (element & 7u);
            if (local_token < route_count) {{
                uint output_pair = order[route_start + local_token];
                ulong row = row0 + local_row;
                float g = Cs[sg][0][element];
                float u = Cs[sg][1][element];
{store}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""


@cache
def _main_kernel(member: str, tile_tokens: int, top_k: int, debug: bool):
    input_names = [
        "x",
        *_projection_stream_names("g", member),
        *_projection_stream_names("u", member),
        "vtab",
        "order",
        "schedule",
        "schedule_total",
    ]
    if not debug:
        input_names.append("sigtab")
    output_names = ["gate_out", "up_out"] if debug else ["out"]
    return mx.fast.metal_kernel(
        name=(f"qwen4_iqk_prefill_gate_up_{member}_t{tile_tokens}_k{top_k}_"
              f"{'debug' if debug else 'silu'}"),
        input_names=input_names,
        output_names=output_names,
        source=_main_source(member, tile_tokens, top_k, debug),
    )


def _validate(
    gate: IqkSwitchLinear,
    up: IqkSwitchLinear,
    x: mx.array,
    indices: mx.array,
    tile_tokens: int,
) -> tuple[mx.array, mx.array, int, int]:
    if not isinstance(gate, IqkSwitchLinear) or not isinstance(up, IqkSwitchLinear):
        raise ValueError("packed prefill requires IQ_K gate and up projections")
    if gate.member != up.member or gate.member not in _MEMBERS:
        raise ValueError(f"packed prefill requires a matching IQ_K pair, got {(gate.member, up.member)}")
    if (gate.in_features, gate.out_features, up.in_features, up.out_features) != (
        HIDDEN, INTERMEDIATE, HIDDEN, INTERMEDIATE,
    ):
        raise ValueError("packed prefill requires the Qwen 2560x640 routed geometry")
    if gate.num_experts != up.num_experts or not 0 < gate.num_experts <= 1024:
        raise ValueError("packed prefill requires matching projection expert counts up to 1024")
    if tile_tokens not in (8, 16, 32):
        raise ValueError("tile_tokens must be 8, 16, or 32")
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError("packed prefill currently requires batch size one")
        hidden = x.reshape(x.shape[1], x.shape[2])
    elif x.ndim == 2:
        hidden = x
    else:
        raise ValueError("hidden must have shape [T, 2560] or [1, T, 2560]")
    if hidden.shape[-1] != HIDDEN:
        raise ValueError("packed prefill hidden width must be 2560")
    if tuple(indices.shape[:-1]) != tuple(x.shape[:-1]) or indices.shape[-1] <= 0:
        raise ValueError("indices must have the hidden token prefix and a positive top-k width")
    top_k = int(indices.shape[-1])
    return (
        hidden.astype(mx.float16), indices.reshape(-1).astype(mx.uint32),
        gate.num_experts, top_k,
    )


def _run(
    gate: IqkSwitchLinear,
    up: IqkSwitchLinear,
    x: mx.array,
    indices: mx.array,
    *,
    tile_tokens: int,
    debug: bool,
) -> tuple[mx.array, ...]:
    hidden, flat_ids, num_experts, top_k = _validate(gate, up, x, indices, tile_tokens)
    routes = int(flat_ids.size)
    order = mx.argsort(flat_ids).astype(mx.uint32)
    sorted_ids = flat_ids[order]
    max_schedule = num_experts + (routes + tile_tokens - 1) // tile_tokens
    dims = mx.array([routes], dtype=mx.uint32)
    schedule, schedule_total = _schedule_kernel(num_experts, tile_tokens)(
        inputs=[sorted_ids, dims],
        grid=(num_experts, 1, 1),
        threadgroup=(num_experts, 1, 1),
        output_shapes=[(max_schedule, 3), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    inputs = [
        hidden,
        *gate._streams(),
        *up._streams(),
        member_table(gate.member),
        order,
        schedule,
        schedule_total,
    ]
    if not debug:
        inputs.append(sigmoid_fp16_table())
    output_count = 2 if debug else 1
    outputs = _main_kernel(gate.member, tile_tokens, top_k, debug)(
        inputs=inputs,
        grid=(INTERMEDIATE // _ROWS * _THREADS, max_schedule, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(routes, INTERMEDIATE)] * output_count,
        output_dtypes=[mx.float16] * output_count,
    )
    shape = tuple(indices.shape) + (INTERMEDIATE,)
    return tuple(output.reshape(shape) for output in outputs)


def packed_gate_up(
    gate: IqkSwitchLinear,
    up: IqkSwitchLinear,
    x: mx.array,
    indices: mx.array,
    *,
    tile_tokens: int = 32,
) -> mx.array:
    """Return route-ordered FP16 SwiGLU activations from packed IQ_K weights."""
    return _run(gate, up, x, indices, tile_tokens=tile_tokens, debug=False)[0]


def packed_gate_up_debug(
    gate: IqkSwitchLinear,
    up: IqkSwitchLinear,
    x: mx.array,
    indices: mx.array,
    *,
    tile_tokens: int = 32,
) -> tuple[mx.array, mx.array]:
    """Return route-ordered FP16 gate and up projections before SwiGLU."""
    gate_out, up_out = _run(gate, up, x, indices, tile_tokens=tile_tokens, debug=True)
    return gate_out, up_out


__all__ = ["packed_gate_up", "packed_gate_up_debug"]
