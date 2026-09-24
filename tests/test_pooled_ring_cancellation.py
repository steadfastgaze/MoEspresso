from __future__ import annotations

from types import MethodType, SimpleNamespace
import threading

import numpy as np
import pytest


pytest.importorskip("mlx.core")
pooled_switchglu = pytest.importorskip("moespresso.runtime.pooled_switchglu")


class _Gate:
    def __init__(self, ensure_finished: threading.Event) -> None:
        self.ensure_finished = ensure_finished
        self.signaled = threading.Event()
        self.sequence: int | None = None
        self.after_ensure = False

    def signal_event(self, sequence: int) -> None:
        self.sequence = sequence
        self.after_ensure = self.ensure_finished.is_set()
        self.signaled.set()


def _ring(sequence: int, ids: np.ndarray) -> np.ndarray:
    ring = np.zeros(8 + ids.size, dtype=np.uint32)
    ring[0] = sequence
    ring[1] = pooled_switchglu._ring_checksum(ids, sequence)
    ring[8:] = ids
    return ring


def test_ring_abort_after_ensure_skips_publish_but_signals_gate() -> None:
    sequence = 17
    ids = np.asarray([6, 1], dtype=np.uint32)
    ensure_started, ensure_finished, release = (threading.Event() for _ in range(3))
    cancelled, published = threading.Event(), threading.Event()
    outcome: list[BaseException] = []

    def ensure(active: set[int]) -> None:
        assert active == {1, 6}
        ensure_started.set()
        assert release.wait(1)
        ensure_finished.set()

    switch = SimpleNamespace(
        _ring_np=_ring(sequence, ids),
        pipeline_read_seconds=0.0,
        seen_experts=set(),
        decode_seen_experts=set(),
        _ensure_projection_pools=ensure,
        _publish_pipe_slots=lambda source_ids: published.set(),
    )
    switch._ring_install_body = MethodType(
        pooled_switchglu.PooledSwitchGLU._ring_install_body, switch
    )
    gate = _Gate(ensure_finished)

    def install() -> None:
        try:
            pooled_switchglu.PooledSwitchGLU.ring_install(
                switch,
                sequence,
                ids.size,
                gate,
                cancelled=cancelled.is_set,
            )
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=install)
    thread.start()
    try:
        assert ensure_started.wait(1)
        cancelled.set()
    finally:
        release.set()
        thread.join(1)

    assert not thread.is_alive()
    assert ensure_finished.is_set()
    assert gate.signaled.is_set()
    assert gate.sequence == sequence
    assert gate.after_ensure
    assert not published.is_set()
    assert len(outcome) == 1
    assert isinstance(outcome[0], pooled_switchglu.CancelledError)


def test_ring_install_publishes_and_signals_gate() -> None:
    sequence = 18
    ids = np.asarray([3], dtype=np.uint32)
    ensured = threading.Event()
    published = threading.Event()
    switch = SimpleNamespace(
        _ring_np=_ring(sequence, ids),
        pipeline_read_seconds=0.0,
        seen_experts=set(),
        decode_seen_experts=set(),
        _ensure_projection_pools=lambda active: ensured.set(),
        _publish_pipe_slots=lambda source_ids: published.set(),
    )
    switch._ring_install_body = MethodType(
        pooled_switchglu.PooledSwitchGLU._ring_install_body, switch
    )
    gate = _Gate(ensured)

    pooled_switchglu.PooledSwitchGLU.ring_install(switch, sequence, ids.size, gate)

    assert ensured.is_set()
    assert published.is_set()
    assert gate.signaled.is_set()
    assert gate.after_ensure
