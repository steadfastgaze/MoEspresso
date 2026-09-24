"""Two-row MTP verification using the installed IQ_K routed arithmetic."""

from functools import cache

import mlx.core as mx
from mlx_iqk.nn import IqkSwitchLinear, member_table
from mlx_iqk.routed import (
    DOWN_STORED, HIDDEN, INTERMEDIATE, SUPPORTED_CODEC_TUPLES, TOP_K,
    down_reduce_source, gate_up_source, sigmoid_fp16_table,
)


def _row_source(source: str, offsets: dict[str, int]) -> str:
    """Offset row-owned pointers; keep the dependency's arithmetic unchanged."""
    declarations = ["uint mtp_row = threadgroup_position_in_grid.y;"]
    declarations.extend(
        f"auto {name} = batch_{name} + (ulong)mtp_row * {stride}ul;"
        for name, stride in offsets.items()
    )
    declarations.append(source)
    return "\n".join(declarations)


def _streams(prefix: str, module: IqkSwitchLinear) -> list[str]:
    return [f"{prefix}_{name}" for name in module.stream_names()]


@cache
def _gate_kernel(gate_member, up_member, gate_names, up_names):
    return mx.fast.metal_kernel(
        name=f"qwen4_mtp_pair_gate_{gate_member}_{up_member}",
        input_names=["batch_x", *gate_names, *up_names, "vtab", "batch_sel", "sigtab"],
        output_names=["batch_out"],
        source=_row_source(gate_up_source(gate_member, up_member), {
            "x": HIDDEN, "sel": TOP_K, "out": TOP_K * INTERMEDIATE,
        }),
    )


@cache
def _down_kernel(member, names):
    return mx.fast.metal_kernel(
        name=f"qwen4_mtp_pair_down_{member}",
        input_names=["batch_x", *names, "vtab", "batch_sel", "batch_scores"],
        output_names=["batch_out"],
        source=_row_source(down_reduce_source(member), {
            "x": TOP_K * INTERMEDIATE, "sel": TOP_K, "scores": TOP_K, "out": HIDDEN,
        }),
    )


def qwen4_mtp_routed_pair(
    gate: IqkSwitchLinear, up: IqkSwitchLinear, down: IqkSwitchLinear,
    hidden: mx.array, indices: mx.array, scores: mx.array,
    gate_up_slots: mx.array, down_slots: mx.array,
) -> mx.array:
    """Execute two target rows in two packed expert dispatches.

    The residency owner certifies every physical slot and identical gate/up
    mappings. Slots may point into unpublished MTP scratch. The two projections
    retain separate down mappings. Source IDs determine the BF16 reduction order;
    physical slot order never changes that order. No full pool dequantization or
    union-expert execution is introduced.
    """
    if not all(isinstance(module, IqkSwitchLinear) for module in (gate, up, down)):
        raise ValueError("MTP routed verification requires IQ_K projection modules")
    if (gate.member, up.member, down.member) not in SUPPORTED_CODEC_TUPLES:
        raise ValueError("MTP routed verification has an unsupported codec tuple")
    if (gate.out_features, gate.in_features, up.out_features, up.in_features,
            down.out_features, down.in_features) != (
            INTERMEDIATE, HIDDEN, INTERMEDIATE, HIDDEN, HIDDEN, DOWN_STORED):
        raise ValueError("MTP routed verification requires the released expert dimensions")
    if len({module.num_experts for module in (gate, up, down)}) != 1:
        raise ValueError("MTP projection slot counts must agree")
    if hidden.shape != (1, 2, HIDDEN) or hidden.dtype != mx.bfloat16:
        raise ValueError("MTP routed verification requires two BF16 hidden rows")
    if scores.shape != (1, 2, TOP_K) or scores.dtype != mx.bfloat16:
        raise ValueError("MTP routed verification requires two BF16 score rows")
    if any(value.shape != scores.shape or value.dtype != mx.uint32
           for value in (indices, gate_up_slots, down_slots)):
        raise ValueError("MTP source and physical IDs must be two uint32 route rows")
    order = mx.argsort(indices, axis=-1)
    selected_gate = mx.take_along_axis(gate_up_slots, order, axis=-1)
    selected_down = mx.take_along_axis(down_slots, order, axis=-1)
    selected_scores = mx.take_along_axis(scores, order, axis=-1)
    activation = _gate_kernel(gate.member, up.member, tuple(_streams("g", gate)), tuple(_streams("u", up)))(
        inputs=[hidden.astype(mx.float16), *gate._streams(), *up._streams(),
                member_table(gate.member), selected_gate, sigmoid_fp16_table()],
        grid=(TOP_K * (INTERMEDIATE // 16) * (HIDDEN // 32), 2, 1),
        threadgroup=(HIDDEN // 32, 1, 1),
        output_shapes=[(1, 2, TOP_K, INTERMEDIATE)], output_dtypes=[mx.float16],
    )[0]
    return _down_kernel(down.member, tuple(_streams("d", down)))(
        inputs=[activation, *down._streams(), member_table(down.member), selected_down, selected_scores],
        grid=((HIDDEN // 16) * TOP_K * 32, 2, 1), threadgroup=(TOP_K * 32, 1, 1),
        output_shapes=[hidden.shape], output_dtypes=[mx.bfloat16],
    )[0]
