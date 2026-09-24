from __future__ import annotations

from types import SimpleNamespace
import threading

import numpy as np
import pytest

import moespresso.runtime.qwen4.expert_provider as provider
from moespresso.runtime.qwen4.expert_provider import Qwen4PaddedPooledSwitchGLU
from moespresso.runtime.qwen4.expert_provider import Qwen4PooledSwitchGLU


class _Pool:
    def __init__(self, slots: dict[int, int]) -> None:
        self.capacity = 4
        self._slot_of = slots
        self._expert_at = [None] * self.capacity
        for expert, slot in slots.items():
            self._expert_at[slot] = expert
        self._prefetch_reserved: set[int] = set()
        self._demand_protect = set(slots)
        self._bk_lock = threading.Lock()


def _publication_executor() -> Qwen4PooledSwitchGLU:
    executor = Qwen4PooledSwitchGLU.__new__(Qwen4PooledSwitchGLU)
    gate = _Pool({2: 3, 5: 1})
    up = _Pool({2: 0, 5: 2})
    down = _Pool({2: 1, 5: 3})
    object.__setattr__(executor, "gate_proj", SimpleNamespace(pool=gate))
    object.__setattr__(executor, "up_proj", SimpleNamespace(pool=up))
    object.__setattr__(executor, "down_proj", SimpleNamespace(pool=down))
    buffers = tuple(bytearray(8) for _ in range(3))
    object.__setattr__(
        executor,
        "_qwen4_pipe_buf_cache",
        {2: tuple(item for buffer in buffers for item in (object(), memoryview(buffer)))},
    )
    return executor


def test_qwen_pipeline_publication_keeps_three_independent_slot_maps() -> None:
    executor = _publication_executor()

    Qwen4PooledSwitchGLU._publish_pipe_slots(executor, np.asarray([2, 5], dtype=np.uint32))

    _gate, gate_view, _up, up_view, _down, down_view = executor._qwen4_pipe_buf_cache[2]
    assert np.frombuffer(gate_view, dtype=np.uint32).tolist() == [3, 1]
    assert np.frombuffer(up_view, dtype=np.uint32).tolist() == [0, 2]
    assert np.frombuffer(down_view, dtype=np.uint32).tolist() == [1, 3]
    assert gate_view.obj is not up_view.obj
    assert up_view.obj is not down_view.obj


def test_qwen_pipeline_publication_accepts_owned_demand_protected_spare() -> None:
    executor = _publication_executor()
    pool = executor.gate_proj.pool
    pool._slot_of[2] = 4
    pool._expert_at[3] = None
    pool._expert_at.append(2)

    executor._publish_pipe_slots(np.asarray([2, 5], dtype=np.uint32))

    gate_view = executor._qwen4_pipe_buf_cache[2][1]
    assert np.frombuffer(gate_view, dtype=np.uint32).tolist() == [4, 1]


class _Buffer:
    def __init__(self, name: str) -> None:
        self.name = name

    def reshape(self, shape):
        return self.name, shape


class _Value:
    def __init__(self, shape, dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype

    def squeeze(self, axis: int):
        assert axis == -2
        return self

    def astype(self, dtype: str):
        return _Value(self.shape, dtype)


def test_qwen_pipelined_graph_uses_each_slot_map_and_restores_input_dtype(monkeypatch) -> None:
    calls = []

    class FakeMx:
        @staticmethod
        def expand_dims(value, axes):
            assert axes == (-2, -3)
            return "operand"

    class Projection:
        def __init__(self, name: str, pool) -> None:
            self.name = name
            self.pool = pool

        def matmul_slots(self, operand, slots, *, sorted_indices: bool):
            assert not sorted_indices
            calls.append((self.name, operand, slots))
            if self.name == "down":
                return _Value((1, 1, 2, 1, 3), "float32")
            return self.name

    executor = Qwen4PooledSwitchGLU.__new__(Qwen4PooledSwitchGLU)
    gate_pool, up_pool, down_pool = object(), object(), object()
    object.__setattr__(executor, "gate_proj", Projection("gate", gate_pool))
    object.__setattr__(executor, "up_proj", Projection("up", up_pool))
    object.__setattr__(executor, "down_proj", Projection("down", down_pool))
    object.__setattr__(executor, "activation", lambda up, gate: (up, gate))
    object.__setattr__(executor, "pipelined_layers", 0)
    object.__setattr__(executor, "total_calls", 0)
    object.__setattr__(executor, "decode_calls", 0)
    object.__setattr__(executor, "total_token_layers", 0)
    object.__setattr__(executor, "_all_iqk", False)
    object.__setattr__(
        executor,
        "_qwen4_pipe_buf_cache",
        {2: (_Buffer("gate"), object(), _Buffer("up"), object(), _Buffer("down"), object())},
    )
    monkeypatch.setattr(provider, "mx", FakeMx)
    native_gate = SimpleNamespace(gate=lambda operand, token, sequence: "gated")

    output = Qwen4PooledSwitchGLU.build_pipelined(
        executor,
        _Value((1, 1, 3), "bfloat16"),
        SimpleNamespace(shape=(1, 1, 2)),
        event_gate=(native_gate, "token", 4),
    )

    assert calls == [
        ("up", "gated", ("up", (1, 1, 2))),
        ("gate", "gated", ("gate", (1, 1, 2))),
        ("down", ("up", "gate"), ("down", (1, 1, 2))),
    ]
    assert output.dtype == "bfloat16"
    assert executor.pipelined_layers == 1
    assert executor.total_calls == 1
    assert executor.decode_calls == 1
    assert executor.total_token_layers == 1
    assert executor._qwen4_pipe_event == (native_gate, 4)
    assert executor._qwen4_pipe_event_buffers is executor._qwen4_pipe_buf_cache[2]


@pytest.mark.parametrize("native_result", [False, True])
def test_native_adapter_keeps_shared_reader_and_terminal_signal(monkeypatch, native_result):
    calls = []

    class Publication:
        def __init__(self, executor, *, timeout_seconds):
            assert executor is provider._PIPELINE_EXECUTOR
            assert timeout_seconds == provider._RING_TIMEOUT

        def begin(self, switch, sequence, count, *, cancelled):
            calls.append(("begin", sequence, count))
            return native_result

        def finish(self, *, completed):
            calls.append(("finish", completed))

    def reader(switch, sequence, count, *, cancelled):
        calls.append(("reader", sequence, count))

    monkeypatch.setattr(provider, "Qwen4NativePublication", Publication)
    monkeypatch.setattr(provider.PooledSwitchGLU, "_ring_install_body", reader)
    switch = Qwen4PooledSwitchGLU.__new__(Qwen4PooledSwitchGLU)
    native = SimpleNamespace(signal_event=lambda seq: calls.append(("signal", seq)))
    switch.ring_install(41, 10, native)
    assert calls == [
        ("begin", 41, 10),
        ("reader", 41, 10),
        ("finish", True),
        ("signal", 41),
    ]


@pytest.mark.parametrize("failure_site", ["begin", "reader"])
def test_native_adapter_clears_receipt_and_preserves_poison_on_failure(monkeypatch, failure_site):
    calls = []

    class Publication:
        def __init__(self, executor, *, timeout_seconds):
            assert timeout_seconds == provider._RING_TIMEOUT

        def begin(self, switch, sequence, count, *, cancelled):
            if failure_site == "begin":
                raise RuntimeError("publication failed")

        def finish(self, *, completed):
            calls.append(("finish", completed))

    def reader(switch, sequence, count, *, cancelled):
        raise RuntimeError("reader failed")

    monkeypatch.setattr(provider, "Qwen4NativePublication", Publication)
    monkeypatch.setattr(provider.PooledSwitchGLU, "_ring_install_body", reader)
    switch = Qwen4PooledSwitchGLU.__new__(Qwen4PooledSwitchGLU)
    native = SimpleNamespace(signal_event=lambda seq: calls.append(("signal", seq)))
    with pytest.raises(RuntimeError, match="failed"):
        switch.ring_install(41, 10, native)
    assert calls == [("finish", False), ("signal", 41)]


def test_native_adapter_keeps_fallback_reader_without_native_capability(monkeypatch):
    calls = []
    monkeypatch.setattr(
        provider.PooledSwitchGLU,
        "_ring_install_body",
        lambda *args, **kwargs: calls.append((args[1:], kwargs)),
    )
    switch = Qwen4PooledSwitchGLU.__new__(Qwen4PooledSwitchGLU)
    switch._ring_install_body(41, 10)
    assert calls == [((41, 10), {"cancelled": None})]
    publication = switch._qwen4_native_publication
    assert publication.snapshot()["native_calls"] == 0
    assert publication.snapshot()["ineligible"] == 1


def test_resident_weighted_decode_never_uses_bounded_fallback() -> None:
    executor = SimpleNamespace(
        _qwen4_closed=False,
        training=False,
        _iqk_two_dispatch_ready=lambda: True,
        _barrier_free_decode_ready=lambda: False,
        _iqk_decode_identity_cached=True,
        _try_bounded_weighted_decode=lambda *_args: (_ for _ in ()).throw(
            AssertionError("bounded decode must not run")
        ),
    )
    executor._full_resident_weighted_decode = lambda *_args: "resident"
    value = SimpleNamespace(shape=(1, 1, 3))

    assert (
        Qwen4PaddedPooledSwitchGLU.try_resident_weighted_decode(
            executor, value, "indices", "scores"
        )
        is None
    )

    executor._barrier_free_decode_ready = lambda: True
    assert (
        Qwen4PaddedPooledSwitchGLU.try_resident_weighted_decode(
            executor, value, "indices", "scores"
        )
        == "resident"
    )
