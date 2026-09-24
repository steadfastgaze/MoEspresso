from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading

import pytest

import moespresso.runtime.pooled_load_batch as pooled_load_batch
from moespresso.runtime.pooled_load_batch import LoadBatch, submit_loads, submit_loads_and_wait


def test_wait_joins_held_sibling_before_raising_first_failure() -> None:
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    returned = threading.Event()
    outcome: list[BaseException] = []

    def held() -> None:
        started.set()
        release.wait()
        finished.set()

    def failed() -> None:
        raise RuntimeError("load fault")

    with ThreadPoolExecutor(max_workers=2) as executor:
        batch = submit_loads(executor, (failed, held))
        assert started.wait(1)

        def wait() -> None:
            try:
                batch.wait()
            except BaseException as exc:
                outcome.append(exc)
            finally:
                returned.set()

        waiter = threading.Thread(target=wait)
        waiter.start()
        assert not returned.wait(0.05)
        release.set()
        assert returned.wait(1)
        waiter.join(1)
    assert finished.is_set()
    assert len(outcome) == 1
    assert isinstance(outcome[0], RuntimeError)
    assert str(outcome[0]) == "load fault"


def test_partial_submit_failure_waits_active_writer_and_cancels_later_calls() -> None:
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    returned = threading.Event()
    outcome: list[BaseException] = []

    def held() -> None:
        started.set()
        release.wait()
        finished.set()

    class PartialExecutor:
        def __init__(self) -> None:
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.calls = 0

        def submit(self, call, job):
            self.calls += 1
            if self.calls == 1:
                return self.pool.submit(call, job)
            assert started.wait(1)
            raise RuntimeError("submit fault")

        def close(self) -> None:
            self.pool.shutdown(wait=True)

    executor = PartialExecutor()
    try:

        def submit() -> None:
            try:
                submit_loads(executor, (held, lambda: pytest.fail("must not submit")))
            except BaseException as exc:
                outcome.append(exc)
            finally:
                returned.set()

        submitter = threading.Thread(target=submit)
        submitter.start()
        assert started.wait(1)
        assert not returned.wait(0.05)
        release.set()
        assert returned.wait(1)
        submitter.join(1)
    finally:
        executor.close()
    assert finished.is_set()
    assert len(outcome) == 1
    assert isinstance(outcome[0], RuntimeError)
    assert str(outcome[0]) == "submit fault"


def test_queued_then_raised_unknown_future_skips_later_pool_mutation() -> None:
    queued, writes = [], []

    class QueueThenRaise:
        def submit(self, call, job):
            queued.append((call, job))
            raise RuntimeError("submit raised after queue")

    with pytest.raises(RuntimeError, match="after queue"):
        submit_loads(QueueThenRaise(), (lambda: writes.append("mutated"),))
    assert len(queued) == 1
    queued[0][0](queued[0][1])
    assert writes == []


def test_iterable_failure_drains_without_an_interruptible_cancelled_snapshot(monkeypatch) -> None:
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    returned = threading.Event()
    error = RuntimeError("load iterable failed")
    outcome: list[BaseException] = []

    def held() -> None:
        started.set()
        release.wait()
        finished.set()

    def calls():
        yield held
        assert started.wait(1)
        raise error

    def interrupted_snapshot(_batch):
        raise KeyboardInterrupt

    monkeypatch.setattr(LoadBatch, "cancelled", property(interrupted_snapshot))
    with ThreadPoolExecutor(max_workers=1) as executor:

        def submit() -> None:
            try:
                submit_loads(executor, calls())
            except BaseException as exc:
                outcome.append(exc)
            finally:
                returned.set()

        submitter = threading.Thread(target=submit)
        submitter.start()
        try:
            assert started.wait(1)
            assert not returned.wait(0.05)
        finally:
            release.set()
            submitter.join(1)
    assert returned.is_set()
    assert finished.is_set()
    assert outcome == [error]


def test_cancel_and_drain_is_repeatable_and_successful_calls_remain_parallel() -> None:
    entered, release = [threading.Event(), threading.Event()], threading.Event()
    active, maximum = [0], [0]
    lock = threading.Lock()

    def call(index: int) -> None:
        with lock:
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
        entered[index].set()
        release.wait()
        with lock:
            active[0] -= 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        batch = submit_loads(executor, (lambda: call(0), lambda: call(1)))
        assert all(event.wait(1) for event in entered)
        release.set()
        batch.wait()
        batch.cancel_and_drain()
        batch.cancel_and_drain()
    assert maximum == [2]


def test_keyboard_interrupt_is_deferred_until_each_active_job_is_quiescent() -> None:
    from moespresso.runtime.pooled_load_batch import LoadBatch

    class InterruptThenQuiescent:
        def __init__(self, batch: LoadBatch) -> None:
            self.batch = batch
            self.waits = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def wait(self) -> None:
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            self.batch._active_writers = 0

    batch = LoadBatch(None)
    batch._active_writers = 1
    condition = InterruptThenQuiescent(batch)
    batch._condition = condition  # type: ignore[assignment]
    with pytest.raises(KeyboardInterrupt):
        batch._cancel_and_await_quiescent()
    assert condition.waits == 2


def test_first_error_survives_cancellation_entry_and_wait_interrupts() -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    returned = threading.Event()
    cancellation_wait = threading.Event()
    outcome: list[BaseException] = []
    error = RuntimeError("first load failure")

    def failed() -> None:
        raise error

    def interrupted() -> None:
        raise KeyboardInterrupt

    def held() -> None:
        entered.set()
        release.wait()
        finished.set()

    class InterruptedCondition:
        def __init__(self, condition) -> None:
            self.condition = condition
            self.waiter_thread = None
            self.enters = 0
            self.waits = 0

        def __enter__(self):
            # Interrupt the waiting caller, never a writer's cleanup section.
            if threading.get_ident() == self.waiter_thread:
                self.enters += 1
                if self.enters == 2:
                    raise KeyboardInterrupt
            return self.condition.__enter__()

        def __exit__(self, *args) -> None:
            return self.condition.__exit__(*args)

        def wait(self) -> None:
            self.waits += 1
            if self.waits <= 3:
                raise KeyboardInterrupt
            cancellation_wait.set()
            self.condition.wait()

        def notify_all(self) -> None:
            self.condition.notify_all()

    with ThreadPoolExecutor(max_workers=3) as executor:
        batch = submit_loads(executor, (failed, interrupted, held))
        assert entered.wait(1)
        condition = InterruptedCondition(batch._condition)
        batch._condition = condition  # type: ignore[assignment]

        def wait() -> None:
            condition.waiter_thread = threading.get_ident()
            try:
                batch.wait()
            except BaseException as exc:
                outcome.append(exc)
            finally:
                returned.set()

        waiter = threading.Thread(target=wait, daemon=True)
        waiter.start()
        try:
            assert cancellation_wait.wait(1)
            assert not returned.is_set()
        finally:
            release.set()
            waiter.join(1)
    assert returned.is_set()
    assert not waiter.is_alive()
    assert finished.is_set()
    assert outcome == [error]
    assert condition.enters >= 5
    assert condition.waits == 4


def test_futures_snapshot_interrupt_cancels_and_drains_a_held_writer() -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    returned = threading.Event()
    outcome: list[BaseException] = []

    def held() -> None:
        entered.set()
        release.wait()
        finished.set()

    class InterruptFirstEnter:
        def __init__(self, condition) -> None:
            self.condition = condition
            self.interrupted = False

        def __enter__(self):
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return self.condition.__enter__()

        def __exit__(self, *args) -> None:
            return self.condition.__exit__(*args)

        def wait(self) -> None:
            self.condition.wait()

        def notify_all(self) -> None:
            self.condition.notify_all()

    with ThreadPoolExecutor(max_workers=1) as executor:
        batch = submit_loads(executor, (held,))
        assert entered.wait(1)
        condition = InterruptFirstEnter(batch._condition)
        batch._condition = condition  # type: ignore[assignment]

        def wait() -> None:
            try:
                batch.wait()
            except BaseException as exc:
                outcome.append(exc)
            finally:
                returned.set()

        waiter = threading.Thread(target=wait)
        waiter.start()
        try:
            assert not returned.wait(0.05)
        finally:
            release.set()
            waiter.join(1)
    assert returned.is_set()
    assert finished.is_set()
    assert len(outcome) == 1
    assert isinstance(outcome[0], KeyboardInterrupt)
    assert condition.interrupted


def test_cancel_does_not_invoke_future_callbacks_under_batch_lock() -> None:
    from concurrent.futures import Future

    queued, writes = [], []

    class QueuedExecutor:
        def submit(self, call, job):
            future: Future[None] = Future()
            queued.append((call, job, future))
            return future

    batch = submit_loads(QueuedExecutor(), (lambda: writes.append("write"),))
    callbacks = []
    future = batch.futures[0]
    future.add_done_callback(lambda _: callbacks.append(batch.futures))
    returned = threading.Event()

    def cancel() -> None:
        batch.cancel_and_drain()
        returned.set()

    canceller = threading.Thread(target=cancel, daemon=True)
    canceller.start()
    assert returned.wait(1)
    canceller.join(1)
    # The queued wrapper can still run after cancellation, but cannot write.
    queued[0][0](queued[0][1])
    future.set_result(None)
    assert writes == []
    assert callbacks == [[future]]


def test_public_submit_exposes_cancellation_and_pending_state() -> None:
    queued: list[tuple] = []

    class QueuedExecutor:
        def submit(self, call, job):
            from concurrent.futures import Future

            future = Future()
            queued.append((call, job, future))
            return future

    batch = LoadBatch(QueuedExecutor())
    future = batch.submit(lambda: pytest.fail("cancelled call must not run"))
    assert batch.pending
    assert not batch.cancelled
    batch.cancel_and_drain()
    assert batch.cancelled
    assert not batch.pending
    queued[0][0](queued[0][1])
    future.set_result(None)
    assert not batch.pending


def test_public_submit_retains_unknown_accepted_future_until_cancelled() -> None:
    queued: list[tuple] = []

    class QueueThenRaise:
        def submit(self, call, job):
            queued.append((call, job))
            raise RuntimeError("unknown accepted future")

    batch = LoadBatch(QueueThenRaise())
    writes: list[str] = []
    with pytest.raises(RuntimeError, match="unknown accepted future"):
        batch.submit(lambda: writes.append("write"))
    assert batch.cancelled
    assert not batch.pending
    queued[0][0](queued[0][1])
    assert writes == []


def _prestart(executor: ThreadPoolExecutor) -> None:
    entered = [threading.Event() for _ in range(executor._max_workers)]
    release = threading.Event()

    def hold(index: int) -> None:
        entered[index].set()
        release.wait()

    futures = [executor.submit(hold, index) for index in range(executor._max_workers)]
    try:
        assert all(event.wait(1) for event in entered)
    finally:
        release.set()
    for future in futures:
        future.result()
    assert len(executor._threads) == executor._max_workers
    assert all(thread.is_alive() for thread in executor._threads)


def _require_raw_join(executor: ThreadPoolExecutor) -> None:
    _prestart(executor)
    if not pooled_load_batch._raw_join_eligible(executor):
        executor.shutdown(wait=True)
        pytest.skip("stdlib worker threads lack the supported raw-join handle contract")


def test_sync_join_engages_only_after_a_full_worker_pool() -> None:
    values = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert submit_loads_and_wait(executor, (lambda: values.append("startup"),)) is False
        _prestart(executor)
        eligible = pooled_load_batch._raw_join_eligible(executor)
        raw = submit_loads_and_wait(executor, (lambda: values.append("after-startup"),))
    assert values == ["startup", "after-startup"]
    assert raw is eligible


def test_sync_join_waits_every_writer_before_propagating_a_worker_error() -> None:
    entered, release, finished, returned = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    outcome: list[BaseException] = []

    def failed() -> None:
        raise LookupError("raw worker failure")

    def held() -> None:
        entered.set()
        release.wait()
        finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        _require_raw_join(executor)

        def join() -> None:
            try:
                submit_loads_and_wait(executor, (failed, held))
            except BaseException as exc:
                outcome.append(exc)
            finally:
                returned.set()

        waiter = threading.Thread(target=join)
        waiter.start()
        try:
            assert entered.wait(1)
            assert not returned.wait(0.05)
        finally:
            release.set()
            waiter.join(1)
        assert executor.submit(lambda: "healthy").result() == "healthy"
    assert finished.is_set()
    assert outcome and isinstance(outcome[0], LookupError)


def test_sync_join_defers_wait_interrupt_until_all_writers_finish(monkeypatch) -> None:
    entered, release, finished, returned = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    outcome: list[BaseException] = []
    original = Future.result
    interrupted = False

    def result(self, *args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("synthetic wait interruption")
        return original(self, *args, **kwargs)

    def held() -> None:
        entered.set()
        release.wait()
        finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        _require_raw_join(executor)
        monkeypatch.setattr(Future, "result", result)

        def join() -> None:
            try:
                submit_loads_and_wait(executor, (held, lambda: None))
            except BaseException as exc:
                outcome.append(exc)
            finally:
                returned.set()

        waiter = threading.Thread(target=join)
        waiter.start()
        try:
            assert entered.wait(1)
            assert not returned.wait(0.05)
        finally:
            release.set()
            waiter.join(1)
    assert finished.is_set()
    assert outcome and isinstance(outcome[0], KeyboardInterrupt)


def test_sync_join_defers_transition_interrupt_until_all_writers_finish(monkeypatch) -> None:
    entered, release, finished, returned = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    outcome: list[BaseException] = []
    original = pooled_load_batch._join_raw_futures
    interrupted = False

    def interrupt_before_join(futures, first=None):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("synthetic raw-join transition interruption")
        return original(futures, first)

    def held() -> None:
        entered.set()
        release.wait()
        finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        _require_raw_join(executor)
        monkeypatch.setattr(pooled_load_batch, "_join_raw_futures", interrupt_before_join)

        def join() -> None:
            try:
                submit_loads_and_wait(executor, (held, lambda: None))
            except BaseException as exc:
                outcome.append(exc)
            finally:
                returned.set()

        waiter = threading.Thread(target=join)
        waiter.start()
        try:
            assert entered.wait(1)
            assert not returned.wait(0.05)
        finally:
            release.set()
            waiter.join(1)
    assert interrupted
    assert finished.is_set()
    assert outcome and isinstance(outcome[0], KeyboardInterrupt)


def test_sync_join_preserves_worker_error_across_later_transition_interrupt(monkeypatch) -> None:
    entered, release, finished, returned = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    outcome: list[BaseException] = []
    original = pooled_load_batch._join_raw_futures
    interrupted = False

    def failed() -> None:
        raise LookupError("first raw worker failure")

    def held() -> None:
        entered.set()
        release.wait()
        finished.set()

    def interrupt_before_later_writer(futures, first):
        nonlocal interrupted
        if not interrupted:
            original([futures[0]], first)
            interrupted = True
            raise KeyboardInterrupt("synthetic later-writer transition interruption")
        original(futures, first)

    with ThreadPoolExecutor(max_workers=2) as executor:
        _require_raw_join(executor)
        monkeypatch.setattr(pooled_load_batch, "_join_raw_futures", interrupt_before_later_writer)

        def join() -> None:
            try:
                submit_loads_and_wait(executor, (failed, held))
            except BaseException as exc:
                outcome.append(exc)
            finally:
                returned.set()

        waiter = threading.Thread(target=join)
        waiter.start()
        try:
            assert entered.wait(1)
            assert not returned.wait(0.05)
        finally:
            release.set()
            waiter.join(1)
    assert interrupted
    assert finished.is_set()
    assert outcome and isinstance(outcome[0], LookupError)


def test_raw_submit_failure_drains_and_poisoned_executor_requires_restart() -> None:
    entered, release, finished, returned = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    outcome: list[BaseException] = []
    error = OSError("accepted raw submission was not returned")

    def held() -> None:
        entered.set()
        release.wait()
        finished.set()

    executor = ThreadPoolExecutor(max_workers=1)
    _require_raw_join(executor)
    original_submit = executor.submit
    original_shutdown = executor.shutdown
    shutdown_calls = 0

    def accepted_then_raised(call, *args, **kwargs):
        original_submit(call, *args, **kwargs)
        raise error

    def interrupted_shutdown(*, wait, cancel_futures):
        nonlocal shutdown_calls
        shutdown_calls += 1
        if shutdown_calls <= 2:
            raise KeyboardInterrupt("synthetic shutdown interruption")
        return original_shutdown(wait=wait, cancel_futures=cancel_futures)

    executor.submit = accepted_then_raised  # type: ignore[method-assign]
    executor.shutdown = interrupted_shutdown  # type: ignore[method-assign]

    def submit() -> None:
        try:
            submit_loads_and_wait(executor, (held,))
        except BaseException as exc:
            outcome.append(exc)
        finally:
            returned.set()

    submitter = threading.Thread(target=submit)
    submitter.start()
    try:
        assert entered.wait(1)
        assert not returned.wait(0.05)
    finally:
        release.set()
        submitter.join(1)
    assert finished.is_set()
    assert outcome == [error]
    assert executor._shutdown
    assert shutdown_calls == 3
    assert any("permanently shut down" in note for note in error.__notes__)
    assert any("server restart is required" in note for note in error.__notes__)
    assert any("interrupted and retried" in note for note in error.__notes__)


def test_sync_join_rejects_executor_worker_and_dead_full_pool() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        _prestart(executor)
        prefix_errors: list[BaseException] = []

        def prefixed_control() -> None:
            try:
                submit_loads_and_wait(executor, (lambda: None,))
            except BaseException as exc:
                prefix_errors.append(exc)

        prefixed = threading.Thread(
            target=prefixed_control,
            name=f"{executor._thread_name_prefix}_startup",
        )
        prefixed.start()
        prefixed.join(1)
        assert prefix_errors and "cannot synchronously control itself" in str(prefix_errors[0])
        future = executor.submit(lambda: submit_loads_and_wait(executor, (lambda: None,)))
        with pytest.raises(RuntimeError, match="cannot synchronously control itself"):
            future.result()

    executor = ThreadPoolExecutor(max_workers=1)
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join(1)
    executor._threads.add(dead)
    try:
        with pytest.raises(RuntimeError, match="dead or invalid worker"):
            submit_loads_and_wait(executor, (lambda: None,))
    finally:
        executor._threads.clear()
        executor.shutdown(wait=True)


def test_sync_join_falls_back_without_supported_thread_handle(monkeypatch) -> None:
    values = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        _prestart(executor)
        worker = next(iter(executor._threads))
        original = getattr(worker, "_handle", None)
        if original is None:
            assert submit_loads_and_wait(executor, (lambda: values.append("fallback"),)) is False
        else:
            original_is_alive = threading.Thread.is_alive
            monkeypatch.setattr(
                threading.Thread,
                "is_alive",
                lambda thread: True if thread is worker else original_is_alive(thread),
            )
            worker._handle = object()  # type: ignore[attr-defined]
            try:
                assert submit_loads_and_wait(executor, (lambda: values.append("fallback"),)) is False
            finally:
                worker._handle = original  # type: ignore[attr-defined]
    assert values == ["fallback"]


def test_sync_join_falls_back_for_unsupported_executor_and_validates_before_submit() -> None:
    values = []

    class UnsupportedExecutor:
        def __init__(self) -> None:
            self.pool = ThreadPoolExecutor(max_workers=1)

        def submit(self, *args):
            return self.pool.submit(*args)

    executor = UnsupportedExecutor()
    try:
        assert submit_loads_and_wait(executor, (lambda: values.append(1),)) is False
        with pytest.raises(TypeError, match="must be callable"):
            submit_loads_and_wait(executor, (lambda: values.append(2), object()))
    finally:
        executor.pool.shutdown(wait=True)
    assert values == [1]
