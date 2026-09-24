from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import pytest

from moespresso.runtime.pooled_load_batch import submit_loads

pytest.importorskip("mlx.core")
pooled_switchglu = pytest.importorskip("moespresso.runtime.pooled_switchglu")


class _Pool:
    def __init__(self, ensure) -> None:
        self._ensure = ensure

    def missing_count(self, active: set[int]) -> int:
        return len(active)

    def ensure(self, active: set[int]) -> None:
        self._ensure(active)


class _Gate:
    def __init__(self) -> None:
        self.signaled = threading.Event()
        self.sequences: list[int] = []

    def signal_event(self, sequence: int) -> None:
        self.sequences.append(sequence)
        self.signaled.set()


def _switch(pools: list[_Pool]) -> SimpleNamespace:
    switch = SimpleNamespace(
        _prefill_prefetch_enabled=False,
        _prefetch_ticket=None,
        _touch_projection_pools_if_resident=lambda active: False,
        _projection_pools=lambda: pools,
        _drain_stale_prefetch_ticket=lambda: pytest.fail("unexpected prefetch drain"),
        overlap_ticket_mismatch_calls=0,
        projection_load_wait_calls=0,
        projection_load_parallel_calls=0,
        projection_sync_join_calls=0,
        projection_tracked_join_calls=0,
        overlap_load_wait_calls=0,
        projection_load_wait_seconds=0.0,
        overlap_load_wait_seconds=0.0,
        overlap_load_total_seconds=0.0,
        overlap_load_hidden_seconds=0.0,
    )
    switch._wait_projection_ticket = lambda active, ticket: (
        pooled_switchglu.PooledSwitchGLU._wait_projection_ticket(switch, active, ticket)
    )
    switch._join_projection_loads = lambda calls: (
        pooled_switchglu.PooledSwitchGLU._join_projection_loads(switch, calls)
    )
    return switch


def _join(thread: threading.Thread) -> None:
    thread.join(1)
    assert not thread.is_alive()


def test_projection_failure_waits_for_held_sibling_before_raising(monkeypatch) -> None:
    held_started, failure_started, release = (threading.Event() for _ in range(3))
    held_finished, returned = threading.Event(), threading.Event()
    outcome: list[BaseException] = []

    def failing(active: set[int]) -> None:
        assert held_started.wait(1)
        failure_started.set()
        raise OSError("first projection failed")

    def held(active: set[int]) -> None:
        held_started.set()
        release.wait()
        held_finished.set()

    executor = ThreadPoolExecutor(max_workers=3)
    monkeypatch.setattr(pooled_switchglu, "_PROJECTION_LOAD_EXECUTOR", executor)
    switch = _switch([_Pool(failing), _Pool(held), _Pool(lambda active: None)])

    def ensure() -> None:
        try:
            pooled_switchglu.PooledSwitchGLU._ensure_projection_pools(switch, {7})
        except BaseException as exc:
            outcome.append(exc)
        finally:
            returned.set()

    thread = threading.Thread(target=ensure)
    thread.start()
    try:
        assert failure_started.wait(1)
        assert not returned.wait(0.05)
        release.set()
        assert returned.wait(1)
        _join(thread)
    finally:
        release.set()
        executor.shutdown(wait=True)

    assert held_finished.is_set()
    assert len(outcome) == 1
    assert isinstance(outcome[0], OSError)
    assert str(outcome[0]) == "first projection failed"


def test_ring_poison_signal_waits_for_all_projection_writers(monkeypatch) -> None:
    held_started, failure_started, release = (threading.Event() for _ in range(3))
    held_finished, returned = threading.Event(), threading.Event()
    outcome: list[BaseException] = []

    def failing(active: set[int]) -> None:
        assert held_started.wait(1)
        failure_started.set()
        raise OSError("ring projection failed")

    def held(active: set[int]) -> None:
        held_started.set()
        release.wait()
        held_finished.set()

    executor = ThreadPoolExecutor(max_workers=3)
    monkeypatch.setattr(pooled_switchglu, "_PROJECTION_LOAD_EXECUTOR", executor)
    switch = _switch([_Pool(failing), _Pool(held), _Pool(lambda active: None)])
    switch._ring_install_body = lambda seq, width: (
        pooled_switchglu.PooledSwitchGLU._ensure_projection_pools(switch, {9})
    )
    gate = _Gate()

    def install() -> None:
        try:
            pooled_switchglu.PooledSwitchGLU.ring_install(switch, 12, 1, gate)
        except BaseException as exc:
            outcome.append(exc)
        finally:
            returned.set()

    thread = threading.Thread(target=install)
    thread.start()
    try:
        assert failure_started.wait(1)
        assert not gate.signaled.wait(0.05)
        assert not returned.is_set()
        release.set()
        assert gate.signaled.wait(1)
        assert returned.wait(1)
        _join(thread)
    finally:
        release.set()
        executor.shutdown(wait=True)

    assert held_finished.is_set()
    assert gate.sequences == [12]
    assert len(outcome) == 1
    assert isinstance(outcome[0], OSError)
    assert str(outcome[0]) == "ring projection failed"


def test_mismatched_ticket_drains_before_fallback_projection_ensure() -> None:
    held_started, held_finished, release = (threading.Event() for _ in range(3))
    fallback_started, returned = threading.Event(), threading.Event()
    executor = ThreadPoolExecutor(max_workers=2)

    def held() -> None:
        held_started.set()
        release.wait()
        held_finished.set()

    def fallback(active: set[int]) -> None:
        assert held_finished.is_set()
        fallback_started.set()

    batch = submit_loads(executor, (held,))
    ticket = pooled_switchglu._ProjectionLoadTicket(
        active={3}, batch=batch, started_at=time.perf_counter()
    )
    switch = _switch([_Pool(fallback)])

    def ensure() -> None:
        try:
            pooled_switchglu.PooledSwitchGLU._ensure_projection_pools(
                switch, {4}, load_ticket=ticket
            )
        finally:
            returned.set()

    thread = threading.Thread(target=ensure)
    thread.start()
    try:
        assert held_started.wait(1)
        assert not fallback_started.wait(0.05)
        release.set()
        assert returned.wait(1)
        _join(thread)
    finally:
        release.set()
        executor.shutdown(wait=True)

    assert held_finished.is_set()
    assert fallback_started.is_set()
    assert ticket.used
    assert switch.overlap_ticket_mismatch_calls == 1
