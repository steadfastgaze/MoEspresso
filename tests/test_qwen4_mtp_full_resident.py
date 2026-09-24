"""Full-resident Qwen4 MTP routes stay device-owned and unbiased."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from moespresso.runtime.qwen4 import mtp_full_resident as module


def test_device_schedule_deduplicates_shared_routes_and_remaps_each_projection():
    indices = mx.array([[list(range(10)), list(range(5, 15))]], dtype=mx.uint32)

    class Pool:
        def __init__(self, shift):
            self.shift = shift

        def remap_ondevice(self, values):
            return (values + self.shift).astype(mx.uint32)

    slots, positions, down = module.full_resident_pair_schedule(
        indices,
        tuple(Pool(shift) for shift in (0, 20, 40)),
    )
    mx.eval(slots, positions, down)
    assert slots.shape == (20, 3)
    assert down.shape == (1, 2, 10)
    assert np.array_equal(np.array(down), np.array(indices) + 40)
    active = np.array(positions)
    assert np.count_nonzero(np.any(active >= 0, axis=1)) == 15
    assert sorted(active[active[:, 0] >= 0, 0].tolist()) == list(range(10))
    assert sorted(active[active[:, 1] >= 0, 1].tolist()) == list(range(10))
    assert np.count_nonzero(np.all(active >= 0, axis=1)) == 5


def _pair(monkeypatch, *, capacity=512, ready=True):
    class IQK:
        pass

    class Pool:
        num_experts = 512

        def __init__(self):
            self.capacity = capacity
            self._slot_of = {i: i for i in range(capacity)}
            self.iqk = IQK()

        def remap_ondevice(self, values):
            return values

    pools = tuple(Pool() for _ in range(3))
    executor = SimpleNamespace(
        _projection_pools_lockstep=lambda: pools,
        _barrier_free_decode_ready=lambda: ready,
    )
    block = SimpleNamespace(
        hidden_size=2560,
        num_experts=512,
        top_k=10,
        retained_source_ids=None,
        training=False,
        experts=executor,
        gate=SimpleNamespace(weight=mx.zeros((512, 2560), dtype=mx.bfloat16)),
        shared_expert_gate=lambda values: mx.zeros((*values.shape[:-1], 1)),
        shared_expert=lambda values: mx.zeros(values.shape, dtype=values.dtype),
    )
    monkeypatch.setattr(module, "IqkSwitchLinear", IQK)
    monkeypatch.setattr(module.mx, "compile", lambda fn: fn)
    return block, pools


@pytest.mark.parametrize("capacity,ready", [(511, False), (512, False)])
def test_pair_requires_complete_certified_residency(monkeypatch, capacity, ready):
    block, _ = _pair(monkeypatch, capacity=capacity, ready=ready)
    with pytest.raises(ValueError, match="complete IQ_K pools"):
        module.Qwen4MTPFullResidentExpertPair(block)


def test_pair_uses_unbiased_router_and_no_transport(monkeypatch):
    block, pools = _pair(monkeypatch)
    provider = SimpleNamespace(select=lambda *a: pytest.fail("cache-biased router used"))
    block.gate._cache_routing_provider = provider
    indices = mx.array([[list(range(10)), list(range(5, 15))]], dtype=mx.uint32)
    scores = mx.ones((1, 2, 10), dtype=mx.bfloat16) / 10
    observed = {}

    class Router:
        _cache_routing_provider = provider

        def __call__(self, hidden, *, cache_routing=False):
            assert hidden.shape == (1, 1, 2560)
            assert cache_routing is False
            row = len(observed.setdefault("router_rows", []))
            observed["router_rows"].append(hidden)
            return SimpleNamespace(indices=indices[:, row:row + 1], scores=scores[:, row:row + 1])

    def routed(*args, **kwargs):
        observed["routed"] = (args, kwargs)
        return mx.zeros((1, 2, 2560), dtype=mx.bfloat16)

    block.gate = Router()
    monkeypatch.setattr(module, "qwen4_mtp_shared_routed_pair", routed)
    pair = module.Qwen4MTPFullResidentExpertPair(block)
    hidden = mx.arange(5120).reshape(1, 2, 2560).astype(mx.bfloat16)
    output = pair(hidden)
    mx.eval(output)
    assert len(observed["router_rows"]) == 2
    assert mx.array_equal(mx.concatenate(observed["router_rows"], axis=1), hidden).item()
    assert observed["routed"][0][:3] == tuple(pool.iqk for pool in pools)
    assert mx.array_equal(observed["routed"][0][4], indices).item()
    assert mx.array_equal(observed["routed"][0][5], scores).item()
    assert pair.paired_calls == 1
    assert pair.routed_rows == 2
