"""Single-dispatch source-MXFP4 routed-MLP decode kernel.

Each threadgroup computes gate/up, SwiGLU and a slice of the down projection for
one selected expert. Input and intermediate activations use threadgroup memory.
The intermediate rounds through fp16 to match the projection-based reference.
Output splits increase occupancy; the shape gate bounds threadgroup storage.

The packed-matmul structure derives from JANG v2.5.29 commit
`e0c5a81fb34a63f1547030902044a4b99d3f2345` under Apache-2.0. Modification
notice: MoEspresso combines gather and fused gate/up structure into a routed
MXFP4 decode dispatch, changes the reduction order, and adds pool slot inputs.
See THIRD-PARTY-NOTICES and LICENSE-APACHE-2.0.
"""

from __future__ import annotations

import os

import mlx.core as mx

_SPLIT = max(1, int(os.environ.get("MOESPRESSO_ROUTED_DECODE_SPLIT", "2")))

_MXFP4_KERNEL_CACHE: dict = {}

# Each threadgroup stages the input and intermediate activations.
_TG = 256


_MXFP4_SOURCE_TEMPLATE = """
    // grid: (TG, K * split, 1); one threadgroup per (expert k, output split s)
    //
    // Source-mxfp4 gate + up + SwiGLU + down in one dispatch.
    #define MXFP4_SCALE(s) as_type<float>((((uint)(s)) == 0u) ? 0x00400000u : (((uint)(s)) << 23))
    #define MXFP4_VAL(code) (static_cast<float>(as_type<half>((ushort)(((code) & 7u) << 9))) * 16384.0f * ((((code) & 8u) != 0u) ? -1.0f : 1.0f))
    #define MXFP4_DEQ(code, scale) (MXFP4_VAL(code) * (scale))

    uint tg_idx = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;

    uint K              = meta[0];
    uint in_f           = meta[1];   // model hidden (gate/up in, down out)
    uint out_f          = meta[2];   // moe intermediate (gate/up out, down in)
    uint gu_cols        = meta[3];   // mxfp4 uint32 cols per gate/up row
    uint gu_scale_cols  = meta[4];   // ue8m0 scale cols per gate/up row
    uint dn_cols        = meta[5];   // mxfp4 uint32 cols per down row
    uint dn_scale_cols  = meta[6];   // ue8m0 scale cols per down row
    uint split          = meta[7];
    uint limit_mil      = meta[8];   // swiglu limit * 1000, 0 = off

    uint k_idx = tg_idx / split;
    uint s_idx = tg_idx % split;
    if (k_idx >= K) return;
    uint slot_gu = slot_ids_gate[k_idx];
    uint slot_dn = slot_ids_down[k_idx];

    threadgroup float xg[{IN_F}];    // raw input
    threadgroup float act[{OUT_F}];  // post-swiglu intermediate

    // ---- stage 1: load x ----
    uint ept_in = (in_f + {TG}u - 1u) / {TG}u;
    for (uint e = 0; e < ept_in; e++) {{
        uint i = tid * ept_in + e;
        if (i < in_f) xg[i] = static_cast<float>(x[i]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- stage 2: gate/up mxfp4 matmul + swiglu -> act (out_f) ----
    uint opt_out = (out_f + {TG}u - 1u) / {TG}u;   // outs per thread
    uint gu_base = slot_gu * out_f * gu_cols;
    uint gu_scale_base = slot_gu * out_f * gu_scale_cols;
    float limit = static_cast<float>(limit_mil) * 0.001f;
    for (uint o = 0; o < opt_out; o++) {{
        uint oi = tid * opt_out + o;
        if (oi >= out_f) break;
        float acc_g = 0.0f;
        float acc_u = 0.0f;
        uint row = gu_base + oi * gu_cols;
        uint scale_row = gu_scale_base + oi * gu_scale_cols;
        for (uint sc = 0; sc < gu_scale_cols; sc++) {{
            float sg = MXFP4_SCALE(scales_gate[scale_row + sc]);
            float su = MXFP4_SCALE(scales_up[scale_row + sc]);
            uint c0 = sc * 4u;
            #pragma unroll
            for (uint w = 0; w < 4u; w++) {{
                uint c = c0 + w;
                uint pg = packed_gate[row + c];
                uint pu = packed_up[row + c];
                uint i0 = c * 8u;
                #pragma unroll
                for (uint v = 0; v < 8u; v++) {{
                    uint sh = v * 4u;
                    float xv = xg[i0 + v];
                    acc_g += xv * MXFP4_DEQ((pg >> sh) & 0xFu, sg);
                    acc_u += xv * MXFP4_DEQ((pu >> sh) & 0xFu, su);
                }}
            }}
        }}
        if (limit > 0.0f) {{
            acc_g = min(acc_g, limit);
            acc_u = min(max(acc_u, -limit), limit);
        }}
        float a = (acc_g / (1.0f + exp(-acc_g))) * acc_u;
        act[oi] = static_cast<float>(static_cast<half>(a));
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- stage 3: down mxfp4 matmul for this split's output slice ----
    uint slice = in_f / split;                  // down outputs per split
    uint out0 = s_idx * slice;
    uint dn_base = slot_dn * in_f * dn_cols;
    uint dn_scale_base = slot_dn * in_f * dn_scale_cols;
    uint opt_dn = (slice + {TG}u - 1u) / {TG}u;
    for (uint o = 0; o < opt_dn; o++) {{
        uint oi = out0 + tid * opt_dn + o;
        if (oi >= out0 + slice) break;
        float acc = 0.0f;
        uint row = dn_base + oi * dn_cols;
        uint scale_row = dn_scale_base + oi * dn_scale_cols;
        for (uint sc = 0; sc < dn_scale_cols; sc++) {{
            float sd = MXFP4_SCALE(scales_down[scale_row + sc]);
            uint c0 = sc * 4u;
            #pragma unroll
            for (uint w = 0; w < 4u; w++) {{
                uint c = c0 + w;
                uint pd = packed_down[row + c];
                uint i0 = c * 8u;
                #pragma unroll
                for (uint v = 0; v < 8u; v++) {{
                    acc += act[i0 + v] * MXFP4_DEQ((pd >> (v * 4u)) & 0xFu, sd);
                }}
            }}
        }}
        out[k_idx * in_f + oi] = acc;
    }}

    #undef MXFP4_DEQ
    #undef MXFP4_VAL
    #undef MXFP4_SCALE
"""


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def routed_decode_supported(in_f: int, out_f: int) -> bool:
    """Shape gate for threadgroup storage and thread coverage."""
    return (
        _is_pow2(in_f) and _is_pow2(out_f)
        and _TG <= in_f <= 4096 and out_f <= 4096
    )


def make_routed_mxfp4_decode_kernel(
    *,
    in_f: int,
    out_f: int,
    K: int,
    swiglu_limit: float = 0.0,
    split: int | None = None,
):
    """Return fn(x_flat, pg, sg, pu, su, pd, sd, gu_slots, dn_slots) -> (K, in_f) fp32.

    Gate/up, SwiGLU and down execute in one Metal dispatch using e2m1 values
    and UE8M0 per-32 scales from DS4/MLX MXFP4 storage.
    """
    if not routed_decode_supported(in_f, out_f):
        return None
    if in_f % 32 or out_f % 32:
        return None
    split = _SPLIT if split is None else max(1, int(split))
    while in_f % split or split > K * 4:
        split -= 1
    key = (in_f, out_f, K, split, int(round(swiglu_limit * 1000)))
    cached = _MXFP4_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    source = _MXFP4_SOURCE_TEMPLATE.format(IN_F=in_f, OUT_F=out_f, TG=_TG)
    kernel = mx.fast.metal_kernel(
        name=f"moespresso_routed_mxfp4_decode_{in_f}_{out_f}",
        input_names=[
            "x",
            "packed_gate", "scales_gate",
            "packed_up", "scales_up",
            "packed_down", "scales_down",
            "slot_ids_gate", "slot_ids_down", "meta",
        ],
        output_names=["out"],
        source=source,
    )

    gu_cols = in_f // 8
    gu_scale_cols = in_f // 32
    dn_cols = out_f // 8
    dn_scale_cols = out_f // 32
    meta = mx.array(
        [K, in_f, out_f, gu_cols, gu_scale_cols, dn_cols, dn_scale_cols, split,
         max(0, int(round(float(swiglu_limit or 0.0) * 1000.0)))],
        dtype=mx.uint32,
    )

    def _fn(x_flat, pg, sg, pu, su, pd, sd, slot_ids_gate, slot_ids_down=None):
        if slot_ids_down is None:
            slot_ids_down = slot_ids_gate
        out, = kernel(
            inputs=[x_flat, pg, sg, pu, su, pd, sd, slot_ids_gate, slot_ids_down, meta],
            output_shapes=[(K, in_f)],
            output_dtypes=[mx.float32],
            grid=(_TG, K * split, 1),
            threadgroup=(_TG, 1, 1),
        )
        return out

    _MXFP4_KERNEL_CACHE[key] = _fn
    return _fn
