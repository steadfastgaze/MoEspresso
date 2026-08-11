"""Decode-shaped paired IQ_K projection kernel.

The gate and up projections consume the same activation rows and routed
expert ids.  ``mlx-iqk`` normally submits one packed-byte GEMV for each
projection.  This module preserves both GEMVs' generated arithmetic while
placing their threadgroups in one Metal dispatch.  SwiGLU and the down
projection remain on their established paths.

The implementation composes the public source generators exposed by the
pinned ``mlx-iqk`` package.  Each projection retains its own member and stream
layout, so mixed pairs such as ``IQ2_KS`` gate with ``IQ2_K`` up are covered.
"""

from __future__ import annotations

import re

import mlx.core as mx

_DUAL_GEMV_KERNELS: dict[tuple[str, str, int, int], object] = {}


def _prefixed_gemv_source(
    member: str,
    in_features: int,
    out_features: int,
    *,
    prefix: str,
) -> str:
    """Return one incumbent GEMV body with projection-local input names."""
    from mlx_iqk.kernels import gemv_input_names, gemv_source

    source = gemv_source(member, in_features, out_features)
    names = gemv_input_names(member)
    replacements = {
        name: f"{prefix}_{name}"
        for name in names
        if name not in {"x", "sel", "dims"}
    }
    replacements["out"] = f"{prefix}_out"
    for name, replacement in replacements.items():
        source = re.sub(rf"\b{re.escape(name)}\b", replacement, source)
    return source


def _dual_gemv_kernel(
    gate_member: str,
    up_member: str,
    in_features: int,
    out_features: int,
):
    """Build and cache one paired gate/up dispatch."""
    from mlx_iqk.format import check_member
    from mlx_iqk.kernels import (
        check_gemv_geometry,
        gemv_input_names,
    )

    gate_member = check_member(gate_member)
    up_member = check_member(up_member)
    in_features, out_features = check_gemv_geometry(
        in_features,
        out_features,
    )
    key = (gate_member, up_member, in_features, out_features)
    kernel = _DUAL_GEMV_KERNELS.get(key)
    if kernel is not None:
        return kernel

    gate_inputs = [
        f"gate_{name}"
        for name in gemv_input_names(gate_member)
        if name not in {"x", "sel", "dims"}
    ]
    up_inputs = [
        f"up_{name}"
        for name in gemv_input_names(up_member)
        if name not in {"x", "sel", "dims"}
    ]
    gate_source = _prefixed_gemv_source(
        gate_member,
        in_features,
        out_features,
        prefix="gate",
    )
    up_source = _prefixed_gemv_source(
        up_member,
        in_features,
        out_features,
        prefix="up",
    )
    source = f"""
    if (threadgroup_position_in_grid.y == 0u) {{
{gate_source}
    }} else {{
{up_source}
    }}
"""
    kernel = mx.fast.metal_kernel(
        name=(
            "moespresso_iqk_dual_gemv_"
            f"{gate_member}_{up_member}_{in_features}x{out_features}"
        ),
        input_names=["x", *gate_inputs, *up_inputs, "sel", "dims"],
        output_names=["gate_out", "up_out"],
        source=source,
    )
    _DUAL_GEMV_KERNELS[key] = kernel
    return kernel


def dual_gemv(gate, up, x, indices) -> tuple[mx.array, mx.array]:
    """Project gate and up in one dispatch with incumbent GEMV arithmetic."""
    from mlx_iqk.kernels import ROWS_PER_TG, gemv_threads
    from mlx_iqk.nn import member_table

    if int(gate.in_features) != int(up.in_features):
        raise ValueError("paired IQ_K projections have different input widths")
    if int(gate.out_features) != int(up.out_features):
        raise ValueError("paired IQ_K projections have different output widths")

    in_features = int(gate.in_features)
    out_features = int(gate.out_features)
    xt = x.reshape(-1, in_features).astype(mx.float16)
    sel = indices.reshape(-1).astype(mx.uint32)
    pairs = int(sel.size)
    rows = int(xt.shape[0])
    if rows <= 0 or pairs % rows:
        raise ValueError(
            f"{pairs} token-expert pairs do not tile {rows} activation rows"
        )
    dims = mx.array([pairs // rows], dtype=mx.uint32)
    blocks = pairs * (out_features // ROWS_PER_TG)
    threads = gemv_threads(in_features)
    kernel = _dual_gemv_kernel(
        gate.member,
        up.member,
        in_features,
        out_features,
    )
    gate_out, up_out = kernel(
        inputs=(
            [xt]
            + gate._streams()
            + [member_table(gate.member)]
            + up._streams()
            + [member_table(up.member), sel, dims]
        ),
        grid=(blocks * threads, 2, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (pairs * out_features,),
            (pairs * out_features,),
        ],
        output_dtypes=[mx.float16, mx.float16],
    )
    shape = [*indices.shape, out_features]
    gate_out = mx.expand_dims(gate_out.reshape(shape), -2)
    up_out = mx.expand_dims(up_out.reshape(shape), -2)
    return gate_out, up_out


def built_dual_gemv_kernels() -> list[tuple[str, str, int, int]]:
    """Return the paired kernel geometries built in this process."""
    return sorted(_DUAL_GEMV_KERNELS)


__all__ = ["built_dual_gemv_kernels", "dual_gemv"]
