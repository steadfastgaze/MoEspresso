from __future__ import annotations

from types import MethodType, SimpleNamespace
import threading

import numpy as np
import pytest

import moespresso.runtime.qwen4.expert_provider as provider_module
from moespresso.runtime.qwen4.expert_provider import Qwen4PaddedPooledSwitchGLU


_SOURCE_IDS = np.array([11, 2, 9, 0, 7, 4, 6, 1, 8, 3], dtype=np.uint32)
_GATE_SLOTS = {
    expert: slot
    for expert, slot in zip(
        _SOURCE_IDS.tolist(),
        [8, 2, 5, 0, 7, 1, 9, 4, 6, 3],
        strict=True,
    )
}
_DOWN_SLOTS = {
    expert: slot
    for expert, slot in zip(
        _SOURCE_IDS.tolist(),
        [1, 9, 0, 8, 2, 7, 3, 6, 4, 5],
        strict=True,
    )
}


class _FakeMx:
    uint32 = np.uint32

    @staticmethod
    def array(value, *, dtype):
        assert dtype is np.uint32
        return np.asarray(value, dtype=np.uint32)

    @staticmethod
    def expand_dims(value, axes):
        return np.expand_dims(value, axes)


class _Projection:
    def __init__(self, name: str, pool: SimpleNamespace, calls: list) -> None:
        self.name = name
        self.pool = pool
        self.calls = calls

    def matmul_slots(self, value, slots):
        self.calls.append((self.name, np.asarray(slots).copy()))
        out_features = 2560 if self.name == "down_proj" else 640
        dtype = np.float16 if self.name == "down_proj" else np.float32
        return np.zeros((*slots.shape, 1, out_features), dtype=dtype)


def _pool(mapping: dict[int, int], *, missing: bool) -> SimpleNamespace:
    published = dict(mapping)
    if missing:
        slot = published.pop(11)
        published[5] = slot
    expert_at = [None] * 10
    for expert, slot in published.items():
        expert_at[slot] = expert
    return SimpleNamespace(
        _bk_lock=threading.Lock(),
        _slot_of=published,
        _expert_at=expert_at,
        _prefetch_reserved=set(),
        _demand_protect=set(),
        capacity=10,
        iqk=SimpleNamespace(num_experts=10),
    )


def _executor(*, missing: bool = False, up_mapping=None):
    gate_pool = _pool(_GATE_SLOTS, missing=missing)
    up_pool = _pool(_GATE_SLOTS if up_mapping is None else up_mapping, missing=missing)
    down_pool = _pool(_DOWN_SLOTS, missing=missing)
    pools = (gate_pool, up_pool, down_pool)
    events = []
    projection_calls = []

    def ensure(active):
        events.append(("ensure", frozenset(active)))
        for pool, target in zip(
            pools,
            (_GATE_SLOTS, _GATE_SLOTS if up_mapping is None else up_mapping, _DOWN_SLOTS),
            strict=True,
        ):
            pool._demand_protect = set(active)
            if 11 not in pool._slot_of:
                slot = pool._slot_of.pop(5)
                pool._expert_at[slot] = 11
                pool._slot_of[11] = slot
                assert target[11] == slot

    executor = SimpleNamespace(
        training=False,
        hidden_size=2560,
        _iqk_decode_identity_cached=False,
        total_calls=0,
        decode_calls=0,
        total_token_layers=0,
        total_unique_active_experts=0,
        max_unique_active_experts=0,
        seen_experts=set(),
        decode_seen_experts=set(),
        direct_calls=0,
        gemv_calls=0,
        gemv_pairs=0,
        barrier_free_decode_calls=0,
        iqk_two_dispatch_calls=0,
        iqk_two_dispatch_routes=0,
        iqk_bounded_two_dispatch_calls=0,
        iqk_bounded_two_dispatch_routes=0,
        iqk_bounded_two_dispatch_gate_up_slot_mismatch_fallbacks=0,
        index_resync_calls=0,
        index_resync_seconds=0.0,
        gate_proj=_Projection("gate_proj", gate_pool, projection_calls),
        up_proj=_Projection("up_proj", up_pool, projection_calls),
        down_proj=_Projection("down_proj", down_pool, projection_calls),
        activation=lambda up, gate: up + gate,
        _iqk_two_dispatch_ready=lambda: True,
        _barrier_free_decode_ready=lambda: False,
        _projection_pools_lockstep=lambda: pools,
        _ensure_projection_pools=ensure,
    )

    def record_iqk_route(pairs):
        executor.gemv_calls += 1
        executor.gemv_pairs += int(pairs)
        return False

    executor._record_iqk_route = record_iqk_route
    executor._bounded_iqk_slot_plan = MethodType(
        Qwen4PaddedPooledSwitchGLU._bounded_iqk_slot_plan,
        executor,
    )
    executor._try_bounded_weighted_decode = MethodType(
        Qwen4PaddedPooledSwitchGLU._try_bounded_weighted_decode,
        executor,
    )
    executor._bounded_iqk_incumbent_weighted_decode = MethodType(
        Qwen4PaddedPooledSwitchGLU._bounded_iqk_incumbent_weighted_decode,
        executor,
    )
    executor.projection_calls = projection_calls
    return executor, events


@pytest.mark.parametrize("missing", [False, True], ids=["all-hit", "demand-miss"])
def test_bounded_iqk_two_dispatch_uses_source_order_and_independent_down_slots(
    monkeypatch,
    missing,
) -> None:
    import mlx_iqk.routed as routed

    executor, events = _executor(missing=missing)
    calls = []

    def gate_up(gate, up, hidden, slots):
        del gate, up, hidden
        calls.append(("gate_up", np.asarray(slots).copy()))
        return np.zeros((10, 640), dtype=np.float16)

    def down_reduce(down, activation, slots, scores):
        del down, activation
        calls.append(
            (
                "down_reduce",
                np.asarray(slots).copy(),
                np.asarray(scores).copy(),
            )
        )
        return np.zeros((2560,), dtype=np.float32)

    monkeypatch.setattr(provider_module, "mx", _FakeMx)
    monkeypatch.setattr(routed, "gate_up_swiglu", gate_up)
    monkeypatch.setattr(routed, "down_reduce", down_reduce)

    hidden = np.zeros((1, 1, 2560), dtype=np.float32)
    indices = _SOURCE_IDS.reshape(1, 1, 10)
    scores = np.arange(10, dtype=np.float32).reshape(1, 1, 10)
    output = Qwen4PaddedPooledSwitchGLU.try_full_resident_weighted_decode(
        executor,
        hidden,
        indices,
        scores,
    )

    source_order = np.argsort(_SOURCE_IDS, kind="stable")
    sorted_sources = _SOURCE_IDS[source_order].tolist()
    assert events == [("ensure", frozenset(_SOURCE_IDS.tolist()))]
    assert calls[0][0] == "gate_up"
    np.testing.assert_array_equal(
        calls[0][1],
        np.array([_GATE_SLOTS[source] for source in sorted_sources], dtype=np.uint32),
    )
    assert calls[1][0] == "down_reduce"
    np.testing.assert_array_equal(
        calls[1][1],
        np.array([_DOWN_SLOTS[source] for source in sorted_sources], dtype=np.uint32),
    )
    np.testing.assert_array_equal(calls[1][2], scores.reshape(-1)[source_order])
    assert output.shape == hidden.shape
    assert executor.index_resync_calls == 1
    assert executor.total_calls == 1
    assert executor.decode_calls == 1
    assert executor.total_token_layers == 1
    assert executor.total_unique_active_experts == 10
    assert executor.max_unique_active_experts == 10
    assert executor.seen_experts == set(_SOURCE_IDS.tolist())
    assert executor.decode_seen_experts == set(_SOURCE_IDS.tolist())
    assert executor.direct_calls == 1
    assert executor.iqk_two_dispatch_calls == 1
    assert executor.iqk_bounded_two_dispatch_calls == 1
    assert executor.iqk_bounded_two_dispatch_routes == 10
    assert executor.barrier_free_decode_calls == 0


def test_bounded_iqk_two_dispatch_falls_back_when_gate_up_slots_differ(
    monkeypatch,
) -> None:
    import mlx_iqk.routed as routed
    import moespresso.runtime.qwen4.moe as moe

    up_mapping = dict(_GATE_SLOTS)
    up_mapping[0], up_mapping[1] = up_mapping[1], up_mapping[0]
    executor, events = _executor(up_mapping=up_mapping)
    native_calls = []
    monkeypatch.setattr(provider_module, "mx", _FakeMx)
    weighted_calls = []

    def weighted_sum(expert_outputs, scores, indices):
        weighted_calls.append(
            (
                np.asarray(expert_outputs).shape,
                np.asarray(expert_outputs).dtype,
                np.asarray(scores).shape,
                np.asarray(indices).copy(),
            )
        )
        return np.zeros((1, 1, 2560), dtype=np.float32)

    monkeypatch.setattr(moe, "expert_major_weighted_sum", weighted_sum)
    monkeypatch.setattr(
        routed,
        "gate_up_swiglu",
        lambda *_args: native_calls.append("gate_up"),
    )
    monkeypatch.setattr(
        routed,
        "down_reduce",
        lambda *_args: native_calls.append("down_reduce"),
    )

    output = Qwen4PaddedPooledSwitchGLU.try_full_resident_weighted_decode(
        executor,
        np.zeros((1, 1, 2560), dtype=np.float32),
        _SOURCE_IDS.reshape(1, 1, 10),
        np.ones((1, 1, 10), dtype=np.float32),
    )

    assert output.shape == (1, 1, 2560)
    assert events == [("ensure", frozenset(_SOURCE_IDS.tolist()))]
    assert native_calls == []
    assert [name for name, _slots in executor.projection_calls] == [
        "up_proj",
        "gate_proj",
        "down_proj",
    ]
    assert len(weighted_calls) == 1
    assert weighted_calls[0][1] == np.dtype(np.float32)
    assert executor.iqk_bounded_two_dispatch_gate_up_slot_mismatch_fallbacks == 1
    assert executor.iqk_two_dispatch_calls == 0
    assert executor.total_calls == 1
    assert executor.direct_calls == 1
    assert executor.gemv_calls == 1
    assert executor.gemv_pairs == 10


def test_bounded_iqk_two_dispatch_rejects_unpublished_demand_slot(monkeypatch) -> None:
    executor, _events = _executor()
    executor.gate_proj.pool._expert_at[_GATE_SLOTS[11]] = None
    with pytest.raises(RuntimeError, match="demand residency was not fully published"):
        Qwen4PaddedPooledSwitchGLU.try_full_resident_weighted_decode(
            executor,
            np.zeros((1, 1, 2560), dtype=np.float32),
            _SOURCE_IDS.reshape(1, 1, 10),
            np.ones((1, 1, 10), dtype=np.float32),
        )


def test_bounded_iqk_two_dispatch_rejects_asymmetric_slot_counts(monkeypatch) -> None:
    executor, _events = _executor()
    executor.down_proj.pool.iqk.num_experts = 11
    with pytest.raises(RuntimeError, match="projection slot counts differ"):
        Qwen4PaddedPooledSwitchGLU.try_full_resident_weighted_decode(
            executor,
            np.zeros((1, 1, 2560), dtype=np.float32),
            _SOURCE_IDS.reshape(1, 1, 10),
            np.ones((1, 1, 10), dtype=np.float32),
        )
