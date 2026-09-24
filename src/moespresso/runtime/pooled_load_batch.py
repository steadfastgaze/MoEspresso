"""Quiescent ownership for one batch of shared pooled-load submissions."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import sys
import threading
from typing import Any, Callable, Iterable


@dataclass(eq=False)
class _Job:
    call: Callable[[], Any]
    future: Any | None = None


class LoadBatch:
    """Own submitted shared-executor calls until each active writer is quiescent."""

    def __init__(self, executor: Any) -> None:
        self._executor = executor
        self._condition = threading.Condition(threading.Lock())
        self._jobs: list[_Job] = []
        self._cancelled = False
        self._active_writers = 0

    @property
    def futures(self) -> list[Any]:
        with self._condition:
            return [job.future for job in self._jobs if job.future is not None]

    @property
    def cancelled(self) -> bool:
        """Return whether unstarted calls have been cancelled."""
        with self._condition:
            return self._cancelled

    @property
    def pending(self) -> bool:
        """Return whether a registered call can still enter or mutate storage."""
        with self._condition:
            if self._active_writers:
                return True
            if self._cancelled:
                return False
            jobs = tuple(self._jobs)
        for job in jobs:
            future = job.future
            if future is None:
                return True
            try:
                if not future.done():
                    return True
            except BaseException:
                return True
        return False

    def _run(self, job: _Job) -> Any:
        with self._condition:
            if self._cancelled:
                return None
            self._active_writers += 1
        try:
            return job.call()
        finally:
            with self._condition:
                self._active_writers -= 1
                # Normal callers join their futures directly. Cancellation is
                # the only path that waits on batch quiescence.
                if self._cancelled and self._active_writers == 0:
                    self._condition.notify_all()

    def _cancel_locked(self) -> None:
        self._cancelled = True
        # Queued wrappers observe cancellation before calling the writer. Do
        # not cancel futures here: their callbacks can re-enter the batch.
        return None

    def _cancel_and_await_quiescent(self, first: BaseException | None = None) -> None:
        """Cancel queued work and defer exceptions until active writers exit."""
        while True:
            try:
                with self._condition:
                    self._cancel_locked()
                    if self._active_writers == 0:
                        break
                    self._condition.wait()
            except BaseException as exc:
                if first is None:
                    first = exc
        if first is not None:
            raise first

    def _abort_and_drain(self, first: BaseException) -> None:
        self._cancel_and_await_quiescent(first)

    def _submit(self, call: Callable[[], Any]) -> Any:
        job = _Job(call)
        with self._condition:
            if self._cancelled:
                raise RuntimeError("pooled load batch is cancelled")
            self._jobs.append(job)
        future = self._executor.submit(self._run, job)
        with self._condition:
            job.future = future
        return future

    def submit(self, call: Callable[[], Any]) -> Any:
        """Register one call before submission and retain it through failure."""
        try:
            if not callable(call):
                raise TypeError("pooled load batch call must be callable")
            return self._submit(call)
        except BaseException as exc:
            self._abort_and_drain(exc)
            raise AssertionError("unreachable")  # pragma: no cover

    def wait(self) -> None:
        """Join every submitted writer and raise the first call failure afterwards."""

        first: BaseException | None = None
        try:
            futures = self.futures
        except BaseException as exc:
            self._abort_and_drain(exc)
            raise AssertionError("unreachable")  # pragma: no cover
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                if first is None:
                    first = exc
                if not isinstance(exc, Exception):
                    self._abort_and_drain(first)
        if first is not None:
            raise first

    def cancel_and_drain(self) -> None:
        """Prevent unstarted calls and wait until every already-active writer exits."""

        self._cancel_and_await_quiescent()


def submit_loads(executor: Any, calls: Iterable[Callable[[], Any]]) -> LoadBatch:
    """Register every call before shared submission and fail only after active writers stop."""

    batch = LoadBatch(executor)
    try:
        for call in calls:
            batch.submit(call)
    except BaseException as exc:
        batch._abort_and_drain(exc)
        raise AssertionError("unreachable")  # pragma: no cover
    return batch


def _raw_join_eligible(executor: Any) -> bool:
    """Return whether a fully started stdlib pool can synchronously join work."""
    if sys.implementation.name != "cpython" or type(executor) is not ThreadPoolExecutor:
        return False
    prefix = getattr(executor, "_thread_name_prefix", None)
    current = threading.current_thread()
    if isinstance(prefix, str) and current.name.startswith(f"{prefix}_"):
        raise RuntimeError("projection executor worker cannot synchronously control itself")
    try:
        with executor._shutdown_lock:
            if executor._shutdown or executor._broken:
                return False
            workers = tuple(executor._threads)
            maximum = executor._max_workers
    except Exception:
        return False
    if not isinstance(maximum, int) or maximum < 1:
        return False
    if len(workers) < maximum:
        return False
    if len(workers) != maximum:
        raise RuntimeError("fully registered projection executor has an invalid worker set")
    if any(not worker.is_alive() for worker in workers):
        raise RuntimeError("fully registered projection executor has a dead or invalid worker")
    handles = tuple(getattr(worker, "_handle", None) for worker in workers)
    if any(
        not callable(getattr(handle, "join", None))
        or not callable(getattr(handle, "is_done", None))
        for handle in handles
    ):
        return False
    try:
        stopped = tuple(handle.is_done() for handle in handles)
    except Exception:
        return False
    if any(stopped):
        raise RuntimeError("fully registered projection executor has a dead or invalid worker")
    if current in workers:
        raise RuntimeError("projection executor worker cannot synchronously control itself")
    return True


def _shutdown_after_raw_submit_failure(executor: ThreadPoolExecutor, first: BaseException) -> None:
    """Permanently drain an executor after raw submission ownership is uncertain."""
    interruptions = []
    while True:
        try:
            executor.shutdown(wait=True, cancel_futures=False)
        except BaseException as exc:
            interruptions.append(type(exc).__name__)
            continue
        break
    first.add_note(
        "raw projection submission failed; executor was permanently shut down and server restart is required"
    )
    if interruptions:
        first.add_note("executor shutdown was interrupted and retried: " + ", ".join(interruptions))
    raise first.with_traceback(first.__traceback__)


def _join_raw_futures(
    futures: list[Future[Any]],
    first: list[BaseException | None],
) -> None:
    """Join every submitted writer before propagating the first failure."""
    for future in futures:
        while True:
            try:
                future.result()
            except BaseException as exc:
                if first[0] is None:
                    first[0] = exc
                while True:
                    try:
                        complete = future.done()
                    except BaseException as interruption:
                        if first[0] is None:
                            first[0] = interruption
                        continue
                    break
                if complete:
                    break
            else:
                break


def submit_loads_and_wait(executor: Any, calls: Iterable[Callable[[], Any]]) -> bool:
    """Run one immediate load group and return whether the raw join engaged."""
    prepared = tuple(calls)
    if any(not callable(call) for call in prepared):
        raise TypeError("pooled load batch call must be callable")
    if not prepared:
        return False
    if not _raw_join_eligible(executor):
        submit_loads(executor, prepared).wait()
        return False

    futures: list[Future[Any]] = []
    first: list[BaseException | None] = [None]
    try:
        for call in prepared:
            future = executor.submit(call)
            if not isinstance(future, Future):
                raise TypeError("stdlib projection executor returned a non-Future")
            futures.append(future)
    except BaseException as exc:
        _shutdown_after_raw_submit_failure(executor, exc)
        raise AssertionError("unreachable")  # pragma: no cover
    while True:
        try:
            _join_raw_futures(futures, first)
            break
        except BaseException as exc:
            if first[0] is None:
                first[0] = exc
    if first[0] is not None:
        raise first[0]
    return True
