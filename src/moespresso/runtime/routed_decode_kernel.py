"""MoEspresso-owned single-dispatch routed-MLP decode kernel (native seam).

Rationale: after the block-exit kick the decode is GPU-bound, and the routed
island's cost is not kernel throughput (jang's kernels measure ~0.18+0.15 ms
chained-independent) but dependent-chain latency: rotate -> fused gate/up ->
rotate -> down gather are four serial Metal dispatches per layer, each paying
launch + drain before the next can start (~0.9 ms serial vs ~0.5 ms of work).
DS4 proves the fix on Apple Silicon (ds4 metal/moe.metal): fuse the whole routed
MLP into one dispatch with the intermediate activations staged in threadgroup
memory.

This kernel computes, for one decode token and K routed experts, in a single
Metal dispatch:

    x_rot   = hadamard(x * signs_in) / sqrt(in_f)          (threadgroup memory)
    act     = silu(gate(x_rot)) * up(x_rot)                (TQ dequant matmul)
    act     = float(half(act))            # fp16 bottleneck, parity with the
                                          # carried island/eager path
    act_rot = hadamard(act * signs_dn) / sqrt(out_f)       (threadgroup memory)
    y[k]    = down(act_rot)                                (TQ dequant matmul)

Grid: one threadgroup per (expert k, down-output split s). Each threadgroup
redundantly computes the full gate/up/act for its expert (the redundancy is
~1M MACs per extra split, cheap) then produces its slice of the down outputs.
The split widens occupancy beyond K threadgroups (MOESPRESSO_ROUTED_DECODE_SPLIT,
default 2 -> K*2 threadgroups).

Scope: decode shape only (batch=1, K experts), power-of-two in/out features,
gate/up sharing codebook+signs+bits (the existing fused precondition), down
with its own bits/codebook/signs. Packed layout, norms semantics, rotation
math and codebook dequant are byte-faithful to jang's kernels
(gather_tq_kernel.py / fused_gate_up_kernel.py / hadamard_kernel.py); the
reduction order differs (per-thread serial vs simd tree), so outputs are
numerically equivalent but not bit-identical: tests pin a tight tolerance.

The packed-matmul and Hadamard portions derive from JANG v2.5.29 commit
`e0c5a81fb34a63f1547030902044a4b99d3f2345` under Apache-2.0. Modification
notice: MoEspresso combines the upstream gather, fused gate/up, and Hadamard
operations into one routed decode dispatch, changes the reduction order, and
adds resident-pool slot inputs. See `THIRD-PARTY-NOTICES` and
`LICENSE-APACHE-2.0`.
"""

from __future__ import annotations

import os

import mlx.core as mx

_SPLIT = max(1, int(os.environ.get("MOESPRESSO_ROUTED_DECODE_SPLIT", "2")))

_KERNEL_CACHE: dict = {}
_MXFP4_KERNEL_CACHE: dict = {}

# Threadgroup size: 256 threads covers in_f<=4096 rotation (16 elems/thread)
# and out_f<=512 gate/up (2 outs/thread) comfortably.
_TG = 256

_SOURCE_TEMPLATE = """
    // grid: (TG, K * split, 1); one threadgroup per (expert k, output split s)
    uint tg_idx = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;

    uint K          = meta[0];
    uint in_f       = meta[1];   // model hidden (gate/up in, down out)
    uint out_f      = meta[2];   // moe intermediate (gate/up out, down in)
    uint gu_cols    = meta[3];   // packed cols per gate/up row
    uint gu_bits    = meta[4];
    uint dn_cols    = meta[5];   // packed cols per down row
    uint dn_bits    = meta[6];
    uint in_log     = meta[7];   // log2(in_f)
    uint out_log    = meta[8];   // log2(out_f)
    uint split      = meta[9];
    uint limit_mil  = meta[10];  // swiglu limit * 1000, 0 = off

    uint k_idx = tg_idx / split;
    uint s_idx = tg_idx % split;
    if (k_idx >= K) return;
    uint slot_gu = slot_ids_gate[k_idx];
    uint slot_dn = slot_ids_down[k_idx];

    threadgroup float xr[{IN_F}];     // rotated input
    threadgroup float act[{OUT_F}];   // post-swiglu intermediate

    // ---- stage 1: load + sign + hadamard rotate x (in_f) ----
    uint ept_in = in_f / {TG}u;       // elems per thread (in_f >= TG, pow2)
    for (uint e = 0; e < ept_in; e++) {{
        uint i = tid * ept_in + e;
        xr[i] = static_cast<float>(x[i]) * signs_in[i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stage = 0; stage < in_log; stage++) {{
        uint h = 1u << stage;
        uint two_h = 2u * h;
        float loc[{EPT_IN}];
        for (uint e = 0; e < ept_in; e++) {{
            uint i = tid * ept_in + e;
            uint bs = (i / two_h) * two_h;
            uint pos = i - bs;
            loc[e] = (pos < h)
                ? xr[bs + pos] + xr[bs + pos + h]
                : xr[bs + pos - h] - xr[bs + pos];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = 0; e < ept_in; e++) {{
            xr[tid * ept_in + e] = loc[e];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float nf_in = 1.0f / sqrt(static_cast<float>(in_f));
    for (uint e = 0; e < ept_in; e++) {{
        xr[tid * ept_in + e] *= nf_in;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- stage 2: gate/up dequant matmul + swiglu -> act (out_f) ----
    uint gu_vals = 32u / gu_bits;
    uint gu_mask = (1u << gu_bits) - 1u;
    uint opt_out = (out_f + {TG}u - 1u) / {TG}u;   // outs per thread
    uint gu_base = slot * out_f * gu_cols;
    float limit = static_cast<float>(limit_mil) * 0.001f;
    for (uint o = 0; o < opt_out; o++) {{
        uint oi = tid * opt_out + o;
        if (oi >= out_f) break;
        float acc_g = 0.0f;
        float acc_u = 0.0f;
        uint row = gu_base + oi * gu_cols;
        for (uint c = 0; c < gu_cols; c++) {{
            uint pg = packed_gate[row + c];
            uint pu = packed_up[row + c];
            uint i0 = c * gu_vals;
            for (uint v = 0; v < gu_vals; v++) {{
                uint i = i0 + v;
                if (i >= in_f) break;
                uint sh = v * gu_bits;
                float xv = xr[i];
                acc_g += xv * cb_gate[(pg >> sh) & gu_mask];
                acc_u += xv * cb_gate[(pu >> sh) & gu_mask];
            }}
        }}
        float g = acc_g * static_cast<float>(norms_gate[slot * out_f + oi]);
        float u = acc_u * static_cast<float>(norms_up[slot * out_f + oi]);
        if (limit > 0.0f) {{
            g = min(g, limit);
            u = min(max(u, -limit), limit);
        }}
        float a = (g / (1.0f + exp(-g))) * u;
        // fp16 bottleneck for parity with the carried eager/island path
        act[oi] = static_cast<float>(static_cast<half>(a));
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- stage 3: sign + hadamard rotate act (out_f) ----
    uint ept_out = (out_f + {TG}u - 1u) / {TG}u;
    for (uint e = 0; e < ept_out; e++) {{
        uint i = tid * ept_out + e;
        if (i < out_f) act[i] *= signs_dn[i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stage = 0; stage < out_log; stage++) {{
        uint h = 1u << stage;
        uint two_h = 2u * h;
        float loc2[{EPT_OUT}];
        for (uint e = 0; e < ept_out; e++) {{
            uint i = tid * ept_out + e;
            if (i >= out_f) continue;
            uint bs = (i / two_h) * two_h;
            uint pos = i - bs;
            loc2[e] = (pos < h)
                ? act[bs + pos] + act[bs + pos + h]
                : act[bs + pos - h] - act[bs + pos];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = 0; e < ept_out; e++) {{
            uint i = tid * ept_out + e;
            if (i < out_f) act[i] = loc2[e];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float nf_out = 1.0f / sqrt(static_cast<float>(out_f));
    for (uint e = 0; e < ept_out; e++) {{
        uint i = tid * ept_out + e;
        if (i < out_f) act[i] *= nf_out;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- stage 4: down dequant matmul for this split's output slice ----
    uint dn_vals = 32u / dn_bits;
    uint dn_mask = (1u << dn_bits) - 1u;
    uint slice = in_f / split;                  // down outputs per split
    uint out0 = s_idx * slice;
    uint dn_base = slot * in_f * dn_cols;
    uint opt_dn = (slice + {TG}u - 1u) / {TG}u;
    for (uint o = 0; o < opt_dn; o++) {{
        uint oi = out0 + tid * opt_dn + o;
        if (oi >= out0 + slice) break;
        float acc = 0.0f;
        uint row = dn_base + oi * dn_cols;
        for (uint c = 0; c < dn_cols; c++) {{
            uint pd = packed_down[row + c];
            uint i0 = c * dn_vals;
            for (uint v = 0; v < dn_vals; v++) {{
                uint i = i0 + v;
                if (i >= out_f) break;
                acc += act[i] * cb_down[(pd >> (v * dn_bits)) & dn_mask];
            }}
        }}
        out[k_idx * in_f + oi] =
            acc * static_cast<float>(norms_down[slot * in_f + oi]);
    }}
"""

_MXFP4_SOURCE_TEMPLATE = """
    // grid: (TG, K * split, 1); one threadgroup per (expert k, output split s)
    //
    // Source-mxfp4 path: raw DS4 e2m1 weights, no TQ rotation/codebook/norms.
    // This is the same fused routed-MLP shape as the TQ kernel above:
    // gate + up + SwiGLU + down in one dispatch.
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

    // ---- stage 1: load x (raw source-mxfp4 basis, no Hadamard rotation) ----
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
    """Shape gate: pow2 dims, rotation fits threadgroup memory, TG coverage."""
    return (
        _is_pow2(in_f) and _is_pow2(out_f)
        and _TG <= in_f <= 4096 and out_f <= 4096
    )


def make_routed_decode_kernel(
    *,
    in_f: int,
    out_f: int,
    gate_bits: int,
    down_bits: int,
    K: int,
    swiglu_limit: float = 0.0,
    split: int | None = None,
):
    """Return fn(x_flat, signs_in, pg, ng, pu, nu, cb_g, signs_dn, pd, nd,
    cb_d, slot_ids) -> (K, in_f) fp32, or None if the shape is unsupported.

    x_flat: (1, in_f) fp16/fp32 unrotated; slot_ids: (K,) uint32 pool slots.
    """
    if not routed_decode_supported(in_f, out_f):
        return None
    split = _SPLIT if split is None else max(1, int(split))
    # the down slice and rotation strides must divide cleanly
    while in_f % split or split > K * 4:
        split -= 1
    key = (in_f, out_f, gate_bits, down_bits, K, split,
           int(round(swiglu_limit * 1000)))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    ept_in = max(1, in_f // _TG)
    ept_out = max(1, (out_f + _TG - 1) // _TG)
    source = _SOURCE_TEMPLATE.format(
        IN_F=in_f, OUT_F=out_f, TG=_TG, EPT_IN=ept_in, EPT_OUT=ept_out)
    kernel = mx.fast.metal_kernel(
        name=f"moespresso_routed_decode_{in_f}_{out_f}_{gate_bits}_{down_bits}",
        input_names=[
            "x", "signs_in",
            "packed_gate", "norms_gate", "packed_up", "norms_up", "cb_gate",
            "signs_dn", "packed_down", "norms_down", "cb_down",
            "slot_ids", "meta",
        ],
        output_names=["out"],
        source=source,
    )

    gu_vals = 32 // gate_bits
    gu_cols = (in_f + gu_vals - 1) // gu_vals
    dn_vals = 32 // down_bits
    dn_cols = (out_f + dn_vals - 1) // dn_vals
    meta = mx.array(
        [K, in_f, out_f, gu_cols, gate_bits, dn_cols, down_bits,
         in_f.bit_length() - 1, out_f.bit_length() - 1, split,
         max(0, int(round(float(swiglu_limit or 0.0) * 1000.0)))],
        dtype=mx.uint32,
    )

    def _fn(x_flat, signs_in, pg, ng, pu, nu, cb_g,
            signs_dn, pd, nd, cb_d, slot_ids):
        out, = kernel(
            inputs=[x_flat, signs_in, pg, ng, pu, nu, cb_g,
                    signs_dn, pd, nd, cb_d, slot_ids, meta],
            output_shapes=[(K, in_f)],
            output_dtypes=[mx.float32],
            grid=(_TG, K * split, 1),
            threadgroup=(_TG, 1, 1),
        )
        return out

    _KERNEL_CACHE[key] = _fn
    return _fn


def make_routed_mxfp4_decode_kernel(
    *,
    in_f: int,
    out_f: int,
    K: int,
    swiglu_limit: float = 0.0,
    split: int | None = None,
):
    """Return fn(x_flat, pg, sg, pu, su, pd, sd, gu_slots, dn_slots) -> (K, in_f) fp32.

    This is the source-mxfp4 counterpart to ``make_routed_decode_kernel``:
    gate/up/SwiGLU + down in one Metal dispatch, using raw e2m1 values and
    UE8M0 per-32 scales from DS4/MLX mxfp4 storage. It intentionally does not
    run TQ's Hadamard rotations.
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
