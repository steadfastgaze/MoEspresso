"""Fork of jang's gather TQ kernel that decouples norms from pool slots.

Forks the minimal jang surface this runtime depends on rather than being
constrained by it. jang's `gather_tq_matmul`
(`jang_tools/turboquant/gather_tq_kernel.py`) indexes both `packed` and `norms`
with the same `rhs_indices` (the remapped resident-pool slot). For SSD streaming
that forces the per-expert `norms` to be re-`pread` and slotted alongside `packed`
on every miss. This fork adds a separate `norms_indices` so `norms` can be a
full-resident array (all experts, indexed by original expert-id) while `packed`
stays a small streaming pool (indexed by slot). The norms then never miss/stream.

This forks only the base gather kernel plus the decode/per-row/broadcast/sorted
dispatch the `PooledSwitchGLU` hot path uses. It reuses jang's `hadamard_rotate_metal`
(the input rotation, unmodified; it never touches norms) and is byte-faithful to
jang's `_GATHER_TQ_SOURCE` except for the two norms lines (input plus the read). The
opt-in MPP_NAX fast path is intentionally not forked (rarely used; falls outside).

The embedded kernel and dispatch wrappers derive from JANG v2.5.29 commit
`e0c5a81fb34a63f1547030902044a4b99d3f2345` under Apache-2.0. Modification
notice: MoEspresso adds an independent `norms_indices` input and uses it for the
norm lookup while the packed weights remain indexed by resident-pool slot. See
`THIRD-PARTY-NOTICES` and `LICENSE-APACHE-2.0`.

With `norms_indices == rhs_indices` this is identical to jang's kernel (the norms
read becomes `norms[slot * out_features + oi]`, same as upstream). A test pins
`gather_tq_matmul_split_norms(..., norms_full, expert_ids) ==
jang.gather_tq_matmul(..., norms_slotted, slots)`.
"""

from __future__ import annotations

import mlx.core as mx
from jang_tools.turboquant.gather_tq_kernel import _GATHER_OPT, _rotate_cached_by_id

# Forked from jang's _GATHER_TQ_SOURCE. The only changes
# vs upstream: (1) a new `norms_indices` kernel input; (2) the norms read uses
# `norms_expert = norms_indices[token_idx*K + k_idx]` instead of the packed `expert`
# (slot). Everything else (rotation-consuming gather, per-bit unpack, simd_sum) is
# kept verbatim so behaviour is identical when norms_indices == rhs_indices.
_GATHER_TQ_SPLIT_NORMS_SOURCE = f'''
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * {_GATHER_OPT}u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];
    uint norms_expert = norms_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc[{_GATHER_OPT}];
    #pragma unroll
    for (uint o = 0; o < {_GATHER_OPT}; o++) acc[o] = 0.0f;

    uint expert_base = expert * out_features * packed_cols;
    uint x_offset = token_idx * in_features;

    uint n_outs = {_GATHER_OPT}u;
    if (out_idx_0 + {_GATHER_OPT}u > out_features) n_outs = out_features - out_idx_0;

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {{
        uint i_base = pack_idx * vals_per_u32;
        uint pv[{_GATHER_OPT}];
        #pragma unroll
        for (uint o = 0; o < {_GATHER_OPT}; o++) {{
            pv[o] = (o < n_outs) ? packed[expert_base + (out_idx_0 + o) * packed_cols + pack_idx] : 0u;
        }}
        if (bits == 2u) {{
            #pragma unroll
            for (uint k = 0; k < 16u; k++) {{
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * 2u;
                #pragma unroll
                for (uint o = 0; o < {_GATHER_OPT}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }} else if (bits == 3u) {{
            #pragma unroll
            for (uint k = 0; k < 10u; k++) {{
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * 3u;
                #pragma unroll
                for (uint o = 0; o < {_GATHER_OPT}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }} else if (bits == 4u) {{
            #pragma unroll
            for (uint k = 0; k < 8u; k++) {{
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * 4u;
                #pragma unroll
                for (uint o = 0; o < {_GATHER_OPT}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }} else if (bits == 8u) {{
            #pragma unroll
            for (uint k = 0; k < 4u; k++) {{
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * 8u;
                #pragma unroll
                for (uint o = 0; o < {_GATHER_OPT}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }} else {{
            #pragma unroll
            for (uint k = 0; k < vals_per_u32; k++) {{
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * bits;
                #pragma unroll
                for (uint o = 0; o < {_GATHER_OPT}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }}
    }}

    #pragma unroll
    for (uint o = 0; o < {_GATHER_OPT}; o++) {{
        acc[o] = simd_sum(acc[o]);
    }}

    if (lane == 0) {{
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {{
            uint oi = out_idx_0 + o;
            float n_v = static_cast<float>(norms[norms_expert * out_features + oi]);
            out[base_off + oi] = acc[o] * n_v;
        }}
    }}
'''

_kernel_cache: dict = {}


def _get_split_norms_kernel():
    if "k" not in _kernel_cache:
        _kernel_cache["k"] = mx.fast.metal_kernel(
            name="gather_tq_matmul_split_norms",
            input_names=["x_rot", "packed", "norms", "codebook",
                         "rhs_indices", "norms_indices", "meta"],
            output_names=["out"],
            source=_GATHER_TQ_SPLIT_NORMS_SOURCE,
        )
    return _kernel_cache["k"]


def _as_uint32(indices: mx.array) -> mx.array:
    return indices if indices.dtype == mx.uint32 else indices.astype(mx.uint32)


def gather_tq_matmul_split_norms(
    x: mx.array,
    packed: mx.array,          # (pool_capacity, out_features, packed_cols) uint32
    norms: mx.array,           # (num_experts, out_features) float16: full resident
    codebook: mx.array,
    signs: mx.array,
    rhs_indices: mx.array,     # pool slots (index into packed)
    norms_indices: mx.array,   # original expert ids (index into the full norms)
    bits: int,
    sorted_indices: bool = False,
) -> mx.array:
    """gather + unpack + matmul with packed indexed by slot and norms by expert-id.

    `rhs_indices` and `norms_indices` have the same shape; the kernel reads
    `packed[rhs_indices...]` and `norms[norms_indices...]`. Mirrors jang's
    `gather_tq_matmul` dispatch for the per_row / broadcast / sorted shapes the
    pooled SwitchGLU hot path uses. Returns (..., K, 1, out_features).
    """
    in_features = x.shape[-1]
    _pool_cap, out_features, packed_cols = packed.shape

    if rhs_indices.ndim == 1:
        while x.ndim > 2 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        x_flat = x.reshape(-1, in_features)
        batch = x_flat.shape[0]
        K = 1
        idx_flat = _as_uint32(rhs_indices)
        nidx_flat = _as_uint32(norms_indices).reshape(-1)
        n_dispatches = batch
        out_shape_kind = "sorted"
    else:
        K = rhs_indices.shape[-1]
        idx_total = 1
        for s in rhs_indices.shape:
            idx_total *= s
        x_squeezed = x
        while x_squeezed.ndim > 2 and x_squeezed.shape[-2] == 1:
            x_squeezed = x_squeezed.squeeze(-2)
        x_flat = x_squeezed.reshape(-1, in_features)
        batch = x_flat.shape[0]
        if batch == idx_total:
            idx_flat = rhs_indices.reshape(-1).astype(mx.uint32)
            nidx_flat = norms_indices.reshape(-1).astype(mx.uint32)
            n_dispatches = batch
            out_shape_kind = "per_row"
        elif batch * K == idx_total:
            idx_flat = rhs_indices.reshape(-1).astype(mx.uint32)
            nidx_flat = norms_indices.reshape(-1).astype(mx.uint32)
            n_dispatches = batch * K
            out_shape_kind = "broadcast"
        else:
            raise ValueError(
                f"shape mismatch: x batch={batch}, indices total={idx_total}, K={K}")

    # Reuse jang's cached Hadamard rotation (unforked; it never touches norms).
    x_rot = _rotate_cached_by_id(x, x_flat, signs)

    kernel = _get_split_norms_kernel()
    k_meta = 1 if (rhs_indices.ndim == 1 or out_shape_kind == "per_row") else K
    meta = mx.array([k_meta, in_features, out_features, packed_cols, bits],
                    dtype=mx.uint32)
    out_groups = (out_features + _GATHER_OPT - 1) // _GATHER_OPT
    grid_x = out_groups * 32
    tg_x = min(256, grid_x)
    out = kernel(
        inputs=[x_rot, packed, norms, codebook, idx_flat, nidx_flat, meta],
        output_shapes=[(n_dispatches, out_features)],
        output_dtypes=[mx.float32],
        grid=(grid_x, n_dispatches, 1),
        threadgroup=(tg_x, 1, 1),
    )[0]

    if out_shape_kind == "sorted":
        out = out.reshape(batch, 1, out_features)
    elif out_shape_kind == "per_row":
        out = out.reshape(*rhs_indices.shape, 1, out_features)
    else:
        out = out.reshape(*rhs_indices.shape[:-1], K, 1, out_features)

    if out.dtype != x.dtype:
        out = out.astype(x.dtype)
    return out
