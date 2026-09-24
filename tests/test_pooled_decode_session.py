from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from moespresso.runtime.pooled_decode_session import PooledDecodeSession


class _Gate:
    def __init__(self, value: int = 0) -> None:
        self.value = value
        self.signals: list[int] = []

    def signaled_value(self) -> int:
        return self.value

    def signal_event(self, sequence: int) -> None:
        self.signals.append(sequence)
        self.value = max(self.value, sequence)


def test_all_resident_request_and_same_owner_nesting_add_no_sync() -> None:
    session = PooledDecodeSession()
    owner = object()
    synchronized: list[object] = []

    def synchronize(root: object) -> None:
        synchronized.append(root)

    with session.request(owner, synchronize=synchronize):
        assert session.active
        assert not session.pending
        with session.request(owner, synchronize=synchronize):
            assert session.active
        with pytest.raises(RuntimeError, match="different owner"):
            with session.request(object(), synchronize=synchronize):
                pass

    assert not session.active
    assert not session.pending
    assert synchronized == []


def test_direct_root_without_event_domain_is_synchronized_on_exit() -> None:
    session = PooledDecodeSession()
    root = object()
    synchronized: list[object] = []

    with session.request(
        object(),
        synchronize=lambda value: synchronized.append(value),
    ):
        session.remember(root)
        assert session.pending

    assert synchronized == [root]
    assert not session.active
    assert not session.pending


def test_global_domain_is_lazy_exclusive_and_monotonic() -> None:
    first, second = PooledDecodeSession(), PooledDecodeSession()
    owner_first, owner_second = object(), object()
    gate = _Gate(41)

    with first.request(owner_first, synchronize=lambda root: None):
        with second.request(owner_second, synchronize=lambda root: None):
            one = first.next_sequence(gate)
            assert one > 41
            with pytest.raises(RuntimeError, match="owned by another request"):
                second.next_sequence(gate)
        two = first.next_sequence(gate)
        assert two == one + 1

    with second.request(owner_second, synchronize=lambda root: None):
        three = second.next_sequence(gate)
    assert three == two + 1


def test_submit_drain_and_request_exit_order_writer_before_sync() -> None:
    session = PooledDecodeSession()
    events: list[str] = []
    root = object()

    def synchronize(value: object) -> None:
        assert value is root
        events.append("sync")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with session.request(object(), synchronize=synchronize):
            session.next_sequence()
            future = session.submit(
                executor,
                lambda cancelled: events.append(f"worker:{cancelled()}"),
                publication_required=True,
            )
            session.remember(root)
            assert session.publication_pending
            session.drain()
            assert future.done()
            assert not session.publication_pending
            assert session.pending
            assert events == ["worker:False"]

    assert events == ["worker:False", "sync"]
    assert not session.active
    assert not session.pending


def test_body_error_cancels_active_writer_then_poison_signals_and_syncs() -> None:
    session = PooledDecodeSession()
    gate = _Gate()
    entered, cancelled = threading.Event(), threading.Event()
    root = object()
    events: list[str] = []
    error = RuntimeError("request failed")

    def writer(is_cancelled) -> None:
        entered.set()
        while not is_cancelled():
            cancelled.wait(0.001)
        cancelled.set()
        events.append("writer stopped")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError) as raised:
            with session.request(object(), synchronize=lambda value: events.append("sync")):
                sequence = session.next_sequence(gate)
                session.submit(executor, writer)
                session.remember(root)
                assert entered.wait(1)
                raise error

    assert raised.value is error
    assert cancelled.is_set()
    assert gate.signals == [sequence]
    assert events == ["writer stopped", "sync"]
    assert not session.active


def test_unknown_accepted_submission_cannot_run_writer_after_abort() -> None:
    session = PooledDecodeSession()
    gate = _Gate()
    queued: list[tuple] = []
    writes: list[str] = []

    class QueueThenRaise:
        def submit(self, call, job):
            queued.append((call, job))
            raise RuntimeError("submit raised after acceptance")

    with pytest.raises(RuntimeError, match="after acceptance"):
        with session.request(object(), synchronize=lambda root: None):
            sequence = session.next_sequence(gate)
            session.submit(QueueThenRaise(), lambda cancelled: writes.append("write"))

    queued[0][0](queued[0][1])
    assert writes == []
    assert gate.signals == [sequence]
    assert not session.active


def test_worker_failure_is_first_error_and_request_is_reusable() -> None:
    session = PooledDecodeSession()
    gate = _Gate()
    error = OSError("writer failed")

    def fail(cancelled) -> None:
        raise error

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(OSError) as raised:
            with session.request(object(), synchronize=lambda root: None):
                sequence = session.next_sequence(gate)
                session.submit(executor, fail)

    assert raised.value is error
    assert gate.signals == [sequence]
    assert not session.active
    with session.request(object(), synchronize=lambda root: None):
        pass


def test_abort_and_drain_synchronizes_device_before_return() -> None:
    session = PooledDecodeSession()
    gate = _Gate()
    root = object()
    synchronized: list[object] = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        with session.request(
            object(),
            synchronize=lambda value: synchronized.append(value),
        ):
            sequence = session.next_sequence(gate)
            session.submit(executor, lambda cancelled: None)
            session.remember(root)
            session.abort_and_drain()
            assert synchronized == [root]
            assert gate.signals == [sequence]
            assert not session.pending

    assert synchronized == [root]
    assert not session.active


def test_failed_cleanup_retains_global_ownership_until_recovered() -> None:
    session, foreign = PooledDecodeSession(), PooledDecodeSession()
    owner = object()
    gate = _Gate()
    allow_sync = [False]

    def synchronize(root: object) -> None:
        if not allow_sync[0]:
            raise RuntimeError("sync failed")

    with pytest.raises(RuntimeError, match="sync failed"):
        with session.request(owner, synchronize=synchronize):
            session.next_sequence(gate)
            session.remember(object())

    assert session.active
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        session.require_current_request()
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        session.remember(object())
    with foreign.request(object(), synchronize=lambda root: None):
        with pytest.raises(RuntimeError, match="owned by another request"):
            foreign.next_sequence(gate)

    allow_sync[0] = True
    session.abort_and_drain()
    assert not session.pending
    assert not session.active
    with foreign.request(object(), synchronize=lambda root: None):
        foreign.next_sequence(gate)


def test_foreign_thread_cannot_operate_active_request() -> None:
    session = PooledDecodeSession()
    errors: list[BaseException] = []

    with session.request(object(), synchronize=lambda root: None):
        thread = threading.Thread(
            target=lambda: _capture(errors, session.next_sequence),
        )
        thread.start()
        thread.join(1)
        assert not thread.is_alive()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "another thread" in str(errors[0])


def _capture(errors: list[BaseException], call) -> None:
    try:
        call()
    except BaseException as exc:
        errors.append(exc)
