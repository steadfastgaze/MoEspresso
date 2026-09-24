"""Unbiased two-row MTP experts for a fully resident Qwen4 target."""

import mlx.core as mx
from mlx_iqk.nn import IqkSwitchLinear

from moespresso.runtime.qwen4.mtp_shared_routed import qwen4_mtp_shared_routed_pair


def full_resident_pair_schedule(indices, pools):
    """Build a fixed-size shared-expert schedule without reading routes on host."""
    if indices.shape != (1, 2, 10) or indices.dtype != mx.uint32:
        raise ValueError("full-resident MTP scheduling requires two uint32 route rows")
    if len(pools) != 3:
        raise ValueError("full-resident MTP scheduling requires three projection pools")

    flat = indices.reshape(20)
    order = mx.argsort(flat)
    sorted_ids = flat[order]
    first = mx.concatenate(
        (
            mx.ones((1,), dtype=mx.bool_),
            sorted_ids[1:] != sorted_ids[:-1],
        )
    )
    routes = mx.arange(10, dtype=mx.int32)
    matches = sorted_ids[:, None, None] == indices[0][None, :, :]
    positions = mx.max(mx.where(matches, routes[None, None, :], -1), axis=-1)
    positions = mx.where(first[:, None], positions, -1).astype(mx.int32)
    union_slots = mx.stack(
        tuple(pool.remap_ondevice(sorted_ids) for pool in pools),
        axis=-1,
    ).astype(mx.uint32)
    down_slots = pools[2].remap_ondevice(indices).astype(mx.uint32)
    return union_slots, positions, down_slots


class Qwen4MTPFullResidentExpertPair:
    """Run an unbiased expert pair entirely against stable resident slots."""

    def __init__(self, block):
        if (
            block.hidden_size != 2560
            or block.num_experts != 512
            or block.top_k != 10
            or block.retained_source_ids is not None
            or block.training
        ):
            raise ValueError("full-resident MTP experts require the released inference geometry")
        executor = block.experts
        pools = tuple(executor._projection_pools_lockstep())
        if (
            len(pools) != 3
            or not all(isinstance(pool.iqk, IqkSwitchLinear) for pool in pools)
            or not all(
                pool.capacity == pool.num_experts == 512 and len(pool._slot_of) == pool.num_experts
                for pool in pools
            )
            or not executor._barrier_free_decode_ready()
        ):
            raise ValueError("full-resident MTP experts require three complete IQ_K pools")
        self.block = block
        self.pools = pools
        self._shared = mx.compile(self._shared_rows)
        self.paired_calls = 0
        self.routed_rows = 0
        self.staged_calls = 0
        self.shared_experts = None

    def _shared_rows(self, values):
        return mx.sigmoid(self.block.shared_expert_gate(values)) * self.block.shared_expert(values)

    def __call__(self, hidden):
        if hidden.shape != (1, 2, 2560) or hidden.dtype != mx.bfloat16 or self.block.training:
            raise ValueError("full-resident MTP experts require exactly two BF16 inference rows")
        # The router's two-row matmul changes BF16 rounding. Keep its decode
        # reduction while sharing the larger expert projections below.
        routes = [self.block.gate(hidden[:, row : row + 1]) for row in range(2)]
        indices = mx.concatenate([route.indices for route in routes], axis=1).astype(mx.uint32)
        scores = mx.concatenate([route.scores for route in routes], axis=1)
        union_slots, positions, down_slots = full_resident_pair_schedule(
            indices,
            self.pools,
        )
        routed = qwen4_mtp_shared_routed_pair(
            *(pool.iqk for pool in self.pools),
            hidden,
            indices,
            scores,
            union_slots,
            positions,
            down_slots=down_slots,
        )
        self.paired_calls += 1
        self.routed_rows += 2
        return routed + self._shared(hidden)


__all__ = ["Qwen4MTPFullResidentExpertPair", "full_resident_pair_schedule"]
