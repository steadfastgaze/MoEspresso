"""Two-row routed verification sharing each union expert's gate/up reads."""

from functools import cache

import mlx.core as mx
from mlx_iqk import routed as r
from mlx_iqk.kernels import _xload
from mlx_iqk.nn import IqkSwitchLinear, member_table

from moespresso.runtime.qwen4.mtp_routed import _down_kernel, _streams


def _paired_decode(prefix, member):
    lines = [f"float {prefix}_a{row}_{half} = 0.0f;"
             for row in range(2) for half in range(2)]
    for position in range(32):
        half = position // 16
        lines.append(f"float {prefix}_v{position} = {r._value(prefix, half, position, member)};")
        lines.append(
            f"{prefix}_a0_{half} = fma({prefix}_v{position}, "
            f"xv0[{position}], {prefix}_a0_{half});"
        )
    lines.append("if (paired) {")
    for position in range(32):
        half = position // 16
        lines.append(
            f"{prefix}_a1_{half} = fma({prefix}_v{position}, "
            f"xv1[{position}], {prefix}_a1_{half});"
        )
    lines.append("}")
    for row in range(2):
        lines.append(
            f"float {prefix}_p{row} = fma({prefix}_dl0, {prefix}_a{row}_0, "
            f"{prefix}_dl1 * {prefix}_a{row}_1);"
        )
    return "\n".join(lines)


def _gate_source(member):
    loads = "\n".join(
        _xload(f"x + {'token * 2560u' if row == 0 else '2560u'} + kbase")
        .replace("xv", f"xv{row}").replace("xu_", f"xu{row}_")
        for row in range(2)
    )
    reads = "\n".join(
        r._projection_code_reads(prefix, f"{prefix}_rid", "kg", r.HIDDEN, member)
        + r._projection_scale(prefix, f"{prefix}_rid", "kg", r.HIDDEN, member)
        + r._projection_quads(prefix, member)
        + _paired_decode(prefix, member)
        for prefix in ("g", "u")
    )
    partials = "\n".join(
        f"float {prefix}_sum{row} = simd_sum({prefix}_p{row});\n"
        f"if (lane == 0u) {{ {prefix}_part[{row}][sg][out_row] = {prefix}_sum{row}; }}"
        for prefix in ("g", "u") for row in range(2)
    )
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint block = threadgroup_position_in_grid.x;
    uint expert = block / 40u;
    if (positions[expert * 2u] < 0 && positions[expert * 2u + 1u] < 0) return;
    uint token = positions[expert * 2u] >= 0 ? 0u : 1u;
    bool paired = positions[expert * 2u] >= 0 && positions[expert * 2u + 1u] >= 0;
    uint row0 = (block % 40u) * 16u;
    uint kbase = tid * 32u;
    uint kg = kbase >> 5u;
    threadgroup half tgV[{16 if member == 'iq3_k' else 8}];
    if (tid < {16 if member == 'iq3_k' else 8}u) {{ tgV[tid] = vtab[tid]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {loads}
    threadgroup float g_part[2][3][16];
    threadgroup float u_part[2][3][16];
    for (uint out_row = 0u; out_row < 16u; ++out_row) {{
        ulong g_rid = (ulong)slots[expert * 3u] * 640ul + row0 + out_row;
        ulong u_rid = (ulong)slots[expert * 3u + 1u] * 640ul + row0 + out_row;
        {reads}
        {partials}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16u) {{
        for (uint row = 0u; row < (paired ? 2u : 1u); ++row) {{
            uint output_token = row == 0u ? token : 1u;
            int route = positions[expert * 2u + output_token];
            if (route >= 0) {{
                half gate = half(g_part[row][0][tid] + g_part[row][1][tid] + g_part[row][2][tid]);
                half up = half(u_part[row][0][tid] + u_part[row][1][tid] + u_part[row][2][tid]);
                half silu = gate * sigtab[as_type<ushort>(gate)];
                out[((ulong)output_token * 10ul + (ulong)route) * 640ul + row0 + tid] = silu * up;
            }}
        }}
    }}
"""


@cache
def _gate_kernel(member, gate_names, up_names):
    return mx.fast.metal_kernel(
        name=f"qwen4_mtp_shared_gate_{member}",
        input_names=["x", *gate_names, *up_names, "vtab", "slots", "positions", "sigtab"],
        output_names=["out"], source=_gate_source(member),
    )


def qwen4_mtp_shared_routed_pair(
    gate: IqkSwitchLinear, up: IqkSwitchLinear, down: IqkSwitchLinear,
    hidden: mx.array, indices: mx.array, scores: mx.array,
    union_slots: mx.array, route_positions: mx.array,
    *, down_slots: mx.array | None = None,
) -> mx.array:
    """Read union gate/up weights once and compute both selected row outputs.

    The residency owner supplies one union entry per source expert. Its three
    slots address gate, up and down independently. Each token's route positions
    are a permutation of 0 through 9, with -1 for an unselected union entry.
    The owner validates those mappings and keeps all slots live until evaluation
    completes. The down stage retains independent per-token threadgroups.
    """
    if not all(isinstance(module, IqkSwitchLinear) for module in (gate, up, down)):
        raise ValueError("shared MTP experts require IQ_K projections")
    if (gate.member, up.member, down.member) not in r.SUPPORTED_CODEC_TUPLES:
        raise ValueError("shared MTP experts have an unsupported codec tuple")
    if (gate.out_features, gate.in_features, up.out_features, up.in_features,
            down.out_features, down.in_features) != (640, 2560, 640, 2560, 2560, 768):
        raise ValueError("shared MTP experts require released projection dimensions")
    if hidden.shape != (1, 2, 2560) or hidden.dtype != mx.bfloat16:
        raise ValueError("shared MTP experts require two BF16 hidden rows")
    if scores.shape != (1, 2, 10) or scores.dtype != mx.bfloat16:
        raise ValueError("shared MTP experts require two BF16 score rows")
    if indices.shape != scores.shape or indices.dtype != mx.uint32:
        raise ValueError("shared MTP experts require two uint32 source rows")
    if (union_slots.ndim != 2 or union_slots.shape[1] != 3
            or not 10 <= union_slots.shape[0] <= 20 or union_slots.dtype != mx.uint32
            or route_positions.shape != (union_slots.shape[0], 2)
            or route_positions.dtype != mx.int32):
        raise ValueError("shared MTP experts require union slots and route positions")
    activation = _gate_kernel(gate.member, tuple(_streams("g", gate)), tuple(_streams("u", up)))(
        inputs=[hidden.astype(mx.float16), *gate._streams(), *up._streams(),
                member_table(gate.member), union_slots, route_positions, r.sigmoid_fp16_table()],
        grid=(union_slots.shape[0] * 40 * 80, 1, 1), threadgroup=(80, 1, 1),
        output_shapes=[(1, 2, 10, 640)], output_dtypes=[mx.float16],
    )[0]
    # Recover per-token down addresses without imposing lockstep projection slots.
    if down_slots is None:
        down_slots = mx.sum(mx.where(
            route_positions[:, :, None] == mx.arange(10, dtype=mx.int32)[None, None, :],
            union_slots[:, 2, None, None], 0,
        ), axis=0).astype(mx.uint32)[None]
    elif down_slots.shape != indices.shape or down_slots.dtype != mx.uint32:
        raise ValueError("shared MTP down slots require two uint32 route rows")
    order = mx.argsort(indices, axis=-1)
    activation = mx.take_along_axis(activation, order[..., None], axis=2)
    return _down_kernel(down.member, tuple(_streams("d", down)))(
        inputs=[activation, *down._streams(), member_table(down.member),
                mx.take_along_axis(down_slots, order, axis=-1),
                mx.take_along_axis(scores, order, axis=-1)],
        grid=(160 * 320, 2, 1), threadgroup=(320, 1, 1),
        output_shapes=[hidden.shape], output_dtypes=[mx.bfloat16],
    )[0]
