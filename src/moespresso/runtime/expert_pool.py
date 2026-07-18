"""Compact expert-pool primitive for SSD-streamed MoE.

The streaming runtime never holds all N experts resident. Instead it gathers the
active experts (per layer per token) into a small pool and calls jang's
`gather_tq_matmul` on the pool with indices remapped from model-expert-ids to
pool-slots. The kernel reads `n_experts` from `packed.shape`, so a compact pool of
the active experts is bit-identical to the full stack, proven in
tests/test_streaming_expert_pool.py.

Two invariants hold: math stays on mx (no numpy tensor compute), and TQ stays
packed so jang's kernel runs it (no dequant at load). This module does the index
remap + row gather with mx ops, then calls jang's kernel unchanged. See
docs/ssd_streaming_p1_jang_kernel_notes.md.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def compact_pool_for_indices(packed: mx.array, norms: mx.array,
                             indices: mx.array):
    """Gather the distinct active experts into a compact pool + remap indices.

    Args:
      packed:  (n_experts, out, packed_cols) uint32: the full (or larger) stack.
      norms:   (n_experts, out) float16.
      indices: (..., K) expert ids the router selected.

    Returns:
      (pool_packed, pool_norms, remapped_indices) where pool_* hold only the
      distinct experts referenced by `indices`, and remapped_indices has the same
      shape as `indices` but points into the pool (0..n_pool-1).

    The distinct-id set + remap is plain control logic on a handful of expert ids
    (host-side bookkeeping only; MLX/Metal owns tensor compute and
    the math, which the gather row-select + the kernel still do). The expensive part
    (gathering pool rows from `packed`) stays a single mx op.

    This convenience computes the distinct set per call. The streaming module
    does not use it directly; it holds a persistent pool + an LRU + a precomputed
    remap, so per-token work is a lookup with no repeated deduplication (see expert_index /
    the streaming module). This function exists to validate the kernel-reuse
    primitive and as a building block.
    """
    host_ids = np.asarray(indices).reshape(-1)
    uniq = np.unique(host_ids)                       # sorted distinct ids (host)
    slot_of = {int(e): i for i, e in enumerate(uniq.tolist())}
    pool_packed = packed[mx.array(uniq.astype(np.uint32))]   # one mx gather
    pool_norms = norms[mx.array(uniq.astype(np.uint32))]
    remap_host = np.vectorize(slot_of.__getitem__)(np.asarray(indices)).astype(np.uint32)
    return pool_packed, pool_norms, mx.array(remap_host)


def gather_via_pool(packed: mx.array, norms: mx.array, codebook: mx.array,
                    signs: mx.array, x: mx.array, indices: mx.array, *,
                    bits: int, sorted_indices: bool = False) -> mx.array:
    """Run jang's gather_tq_matmul through a compact pool (logit-identical).

    A convenience that compacts then dispatches, used to validate the pool path
    and as the building block the streaming module specializes (the streaming
    version supplies a persistent pool + a native miss-loader instead of gathering
    from a full resident `packed`).
    """
    from jang_tools.turboquant.gather_tq_kernel import gather_tq_matmul

    pool_packed, pool_norms, remapped = compact_pool_for_indices(
        packed, norms, indices)
    return gather_tq_matmul(x, pool_packed, pool_norms, codebook, signs,
                            remapped, bits=bits, sorted_indices=sorted_indices)
