"""Correctness of the compact-expert-pool primitive for SSD streaming.

The streaming runtime never holds all 256 experts resident. Instead it keeps a
pool of the experts it needs and calls jang's gather_tq_matmul on the pool with
indices remapped from model-expert-ids to pool-slots. These tests prove, on the
MLX/native path (not a Python end-to-end loop), that:

  gather_tq_matmul(x, FULL_packed, ..., indices)
    ==
  gather_tq_matmul(x, POOL_packed, ..., remapped_indices)

i.e. computing from a compacted subset is bit-identical to the full stack: the
foundation that allows streaming only the active experts.

Requires mlx + jang and skips cleanly in an incomplete development environment.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant.gather_tq_kernel")

import mlx.core as mx  # noqa: E402

from moespresso.runtime.expert_pool import compact_pool_for_indices  # noqa: E402


def _tiny_switch(n_experts=8, in_features=64, out_features=32, bits=2, seed=42):
    from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear
    mx.random.seed(7)
    mod = TurboQuantSwitchLinear(in_features, out_features, n_experts,
                                 bits=bits, seed=seed)
    # random packed + norms so experts differ
    vals_per_u32 = 32 // bits
    packed_cols = (in_features + vals_per_u32 - 1) // vals_per_u32
    mod.packed = mx.random.randint(0, 2**31, (n_experts, out_features, packed_cols)).astype(mx.uint32)
    mod.norms = (mx.random.normal((n_experts, out_features)) * 0.1).astype(mx.float16)
    mx.eval(mod.packed, mod.norms)
    return mod


def test_compact_pool_matches_full_stack_gate_pattern():
    mod = _tiny_switch()
    x = mx.random.normal((1, 1, 1, 64)).astype(mx.float16)   # gate/up broadcast-K
    indices = mx.array([[3, 7, 1, 5]], dtype=mx.uint32)       # (tokens=1, K=4)

    full = mod(x, indices)
    mx.eval(full)

    # Build the compact pool: only experts {1,3,5,7}, indices remapped to slots.
    pool_packed, pool_norms, remapped = compact_pool_for_indices(
        mod.packed, mod.norms, indices)
    from jang_tools.turboquant.gather_tq_kernel import gather_tq_matmul
    pooled = gather_tq_matmul(x, pool_packed, pool_norms, mod.codebook, mod.signs,
                              remapped, bits=mod.bits)
    mx.eval(pooled)

    assert pool_packed.shape[0] == 4     # only the 4 distinct active experts
    assert np.array_equal(np.array(full), np.array(pooled))


def test_compact_pool_matches_full_stack_with_duplicate_experts():
    mod = _tiny_switch()
    x = mx.random.normal((2, 1, 1, 64)).astype(mx.float16)
    # token 0 and token 1 share some experts -> pool dedups
    indices = mx.array([[3, 7, 1, 5], [7, 7, 2, 1]], dtype=mx.uint32)

    full = mod(x, indices)
    pool_packed, pool_norms, remapped = compact_pool_for_indices(
        mod.packed, mod.norms, indices)
    from jang_tools.turboquant.gather_tq_kernel import gather_tq_matmul
    pooled = gather_tq_matmul(x, pool_packed, pool_norms, mod.codebook, mod.signs,
                              remapped, bits=mod.bits)
    mx.eval(full, pooled)

    distinct = len(set(np.array(indices).flatten().tolist()))
    assert pool_packed.shape[0] == distinct
    assert np.array_equal(np.array(full), np.array(pooled))


def test_compact_pool_preserves_logits_through_full_switch_call():
    """End-to-end through the module: a streaming subclass that compacts internally
    must match the resident module's output."""
    mod = _tiny_switch(n_experts=16)
    x = mx.random.normal((3, 1, 1, 64)).astype(mx.float16)
    indices = mx.array([[3, 7, 1, 5], [9, 2, 1, 5], [15, 0, 7, 8]], dtype=mx.uint32)

    resident = mod(x, indices)

    from moespresso.runtime.expert_pool import gather_via_pool
    pooled = gather_via_pool(mod.packed, mod.norms, mod.codebook, mod.signs,
                             x, indices, bits=mod.bits)
    mx.eval(resident, pooled)
    assert np.array_equal(np.array(resident), np.array(pooled))
