"""The forked split-norms kernel must be numerically identical to jang's kernel.

Anchors the fork (`moespresso.runtime.gather_tq_split_norms`): with
`norms_indices == rhs_indices` it must byte-match jang's `gather_tq_matmul`, and the
real split case (small slot-packed pool + full-resident norms indexed by expert-id)
must match the slotted result too.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant.gather_tq_kernel")

import mlx.core as mx  # noqa: E402

from jang_tools.turboquant.codebook import compute_codebook  # noqa: E402
from jang_tools.turboquant.gather_tq_kernel import gather_tq_matmul  # noqa: E402
from jang_tools.turboquant.rotation import generate_random_signs  # noqa: E402

from moespresso.runtime.gather_tq_split_norms import (  # noqa: E402
    gather_tq_matmul_split_norms,
)


def _tables(in_f, bits):
    cb = mx.array(compute_codebook(in_f, bits), dtype=mx.float32)
    sg = mx.array(generate_random_signs(in_f, seed=42), dtype=mx.float32)
    mx.eval(cb, sg)
    return cb, sg


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_split_norms_identity_matches_jang(bits):
    """norms_indices == rhs_indices -> byte-identical to jang's kernel."""
    rng = np.random.default_rng(bits)
    in_f, out_f, ne = 512, 256, 8
    pc = in_f // (32 // bits)
    packed = mx.array(rng.integers(0, 2**32, size=(ne, out_f, pc), dtype=np.uint32))
    norms = mx.array(rng.standard_normal((ne, out_f)).astype(np.float16))
    cb, sg = _tables(in_f, bits)
    x = mx.array(rng.standard_normal((1, 4, 1, in_f)).astype(np.float32))
    idx = mx.array(np.array([[3, 5, 1, 7]], dtype=np.uint32))
    mx.eval(packed, norms, x, idx)

    a = gather_tq_matmul(x, packed, norms, cb, sg, idx, bits=bits)
    b = gather_tq_matmul_split_norms(x, packed, norms, cb, sg, idx, idx, bits=bits)
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_split_norms_pool_packed_full_norms_matches_jang(bits):
    """Small slot-packed pool + full norms (by expert-id) == jang's slotted result."""
    rng = np.random.default_rng(100 + bits)
    in_f, out_f, ne = 512, 256, 8
    pc = in_f // (32 // bits)
    packed = mx.array(rng.integers(0, 2**32, size=(ne, out_f, pc), dtype=np.uint32))
    norms = mx.array(rng.standard_normal((ne, out_f)).astype(np.float16))
    cb, sg = _tables(in_f, bits)
    x = mx.array(rng.standard_normal((1, 4, 1, in_f)).astype(np.float32))
    expert_ids = [3, 5, 1, 7]
    idx = mx.array(np.array([expert_ids], dtype=np.uint32))
    mx.eval(packed, norms, x, idx)

    ref = gather_tq_matmul(x, packed, norms, cb, sg, idx, bits=bits)

    # a 4-slot pool holding experts [3,5,1,7] at slots [0,1,2,3]
    slot_packed = mx.stack([packed[e] for e in expert_ids])
    slots = mx.array(np.array([[0, 1, 2, 3]], dtype=np.uint32))
    eids = mx.array(np.array([expert_ids], dtype=np.uint32))
    mx.eval(slot_packed, slots, eids)

    got = gather_tq_matmul_split_norms(
        x, slot_packed, norms, cb, sg, slots, eids, bits=bits)
    mx.eval(ref, got)
    assert np.array_equal(np.array(ref), np.array(got))
