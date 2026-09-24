"""Request ownership for ordered pooled decode publication."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any, Callable, Iterator

from moespresso.runtime.pooled_load_batch import LoadBatch


_DOMAIN_LOCK = threading.Lock()
_DOMAIN_OWNER: PooledDecodeSession | None = None
_DOMAIN_SEQUENCE = 0


class PooledDecodeSession:
    """Own pooled writers and their process-global native event sequence."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owner: object | None = None
        self._owner_thread: int | None = None
        self._depth = 0
        self._synchronize: Callable[[Any], Any] | None = None
        self._failure: BaseException | None = None
        self._batch: LoadBatch | None = None
        self._executor: Any | None = None
        self._publication_pending = False
        self._root: Any | None = None
        self._domain_claimed = False
        self._gate_bound = False
        self._gate_mod: Any | None = None
        self._native_max: int | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._owner is not None

    @property
    def pending(self) -> bool:
        with self._lock:
            batch = self._batch
            root = self._root
            publication = self._publication_pending
        # A completed but undrained batch can still carry an unobserved worker
        # error. Retain request ownership until its result has been collected.
        return root is not None or publication or batch is not None

    @property
    def publication_pending(self) -> bool:
        with self._lock:
            return self._publication_pending

    def _require_owner_thread(self) -> None:
        with self._lock:
            if self._owner is None:
                raise RuntimeError("pooled decode session has no active request")
            if self._owner_thread != threading.get_ident():
                raise RuntimeError("pooled decode request is owned by another thread")

    def require_current_request(self) -> None:
        """Require an entered request owned by the calling thread."""
        self._require_owner_thread()
        with self._lock:
            if self._depth <= 0:
                raise RuntimeError("pooled decode session cleanup is incomplete")

    def _enter(self, owner: object, synchronize: Callable[[Any], Any]) -> None:
        if owner is None:
            raise TypeError("pooled decode request owner cannot be None")
        if not callable(synchronize):
            raise TypeError("pooled decode synchronize callback must be callable")
        thread = threading.get_ident()
        with self._lock:
            if self._owner is None:
                self._owner = owner
                self._owner_thread = thread
                self._synchronize = synchronize
                self._depth = 1
                self._failure = None
                return
            if self._depth == 0:
                raise RuntimeError("pooled decode session cleanup is incomplete")
            if self._owner is not owner:
                raise RuntimeError("pooled decode request has a different owner")
            if self._owner_thread != thread:
                raise RuntimeError("pooled decode request is owned by another thread")
            self._depth += 1

    @contextmanager
    def request(
        self,
        owner: object,
        *,
        synchronize: Callable[[Any], Any],
    ) -> Iterator[PooledDecodeSession]:
        """Own one request until its writers and remembered graph are quiescent."""
        self._enter(owner, synchronize)
        body_error: BaseException | None = None
        traceback = None
        try:
            yield self
        except BaseException as exc:
            body_error = exc
            traceback = exc.__traceback__

        cleanup_error = self._leave(body_error)
        if body_error is not None:
            raise body_error.with_traceback(traceback)
        if cleanup_error is not None:
            raise cleanup_error

    def next_sequence(self, gate_mod: Any | None = None) -> int:
        """Claim the global event domain and allocate its next sequence."""
        global _DOMAIN_OWNER, _DOMAIN_SEQUENCE

        self.require_current_request()
        if gate_mod is not None:
            if not callable(getattr(gate_mod, "signaled_value", None)):
                raise TypeError("native gate must expose signaled_value()")
            if not callable(getattr(gate_mod, "signal_event", None)):
                raise TypeError("native gate must expose signal_event(sequence)")
        with _DOMAIN_LOCK:
            if _DOMAIN_OWNER is not None and _DOMAIN_OWNER is not self:
                raise RuntimeError("native event domain is owned by another request")
            with self._lock:
                if self._gate_bound and self._gate_mod is not gate_mod:
                    raise RuntimeError("pooled decode request changed native gate module")
                if gate_mod is not None:
                    signaled = gate_mod.signaled_value()
                    if isinstance(signaled, bool) or not isinstance(signaled, int):
                        raise TypeError("native gate signaled value must be an integer")
                    if signaled < 0:
                        raise ValueError("native gate signaled value cannot be negative")
                    _DOMAIN_SEQUENCE = max(_DOMAIN_SEQUENCE, signaled)
                _DOMAIN_OWNER = self
                self._domain_claimed = True
                self._gate_bound = True
                self._gate_mod = gate_mod
                _DOMAIN_SEQUENCE += 1
                sequence = _DOMAIN_SEQUENCE
                if gate_mod is not None:
                    self._native_max = sequence
                return sequence

    def submit(
        self,
        executor: Any,
        call: Callable[[Callable[[], bool]], Any],
        *,
        publication_required: bool = False,
    ) -> Any:
        """Submit one owned writer with a live cancellation predicate."""
        self.require_current_request()
        if not callable(call):
            raise TypeError("pooled decode worker call must be callable")
        with self._lock:
            if not self._domain_claimed:
                raise RuntimeError("pooled decode event domain is not claimed")
            batch = self._batch
            if batch is None:
                batch = LoadBatch(executor)
                self._batch = batch
                self._executor = executor
            elif self._executor is not executor:
                raise RuntimeError("pooled decode request changed its shared executor")
            if publication_required:
                self._publication_pending = True

        return batch.submit(lambda: call(lambda: batch.cancelled))

    def remember(self, root: Any) -> None:
        """Retain the newest graph root until request cleanup synchronizes it."""
        self.require_current_request()
        if root is None:
            raise TypeError("pooled decode graph root cannot be None")
        with self._lock:
            self._root = root

    def _normal_drain(self) -> BaseException | None:
        with self._lock:
            batch = self._batch
        if batch is None:
            with self._lock:
                if self._publication_pending:
                    return RuntimeError("publication is pending without an owned load batch")
            return None
        try:
            batch.wait()
        except BaseException as exc:
            return exc
        if batch.pending:
            return RuntimeError("pooled load batch remained pending after wait")
        with self._lock:
            if self._batch is batch:
                self._batch = None
                self._executor = None
                self._publication_pending = False
        return None

    def drain(self) -> None:
        """Join the current writer group without synchronizing its graph root."""
        self.require_current_request()
        error = self._normal_drain()
        if error is not None:
            raise error

    def begin_load_batch(self, executor: Any) -> LoadBatch:
        """Own an empty direct-load batch before its first submission."""
        self.require_current_request()
        with self._lock:
            if self._batch is not None:
                raise RuntimeError("pooled decode request already owns a load batch")
            batch = LoadBatch(executor)
            self._batch = batch
            self._executor = executor
            return batch

    def _signal_abort(self) -> BaseException | None:
        with self._lock:
            gate_mod = self._gate_mod
            native_max = self._native_max
        if gate_mod is None or native_max is None:
            return None
        try:
            gate_mod.signal_event(native_max)
        except BaseException as exc:
            return exc
        return None

    def _synchronize_root(self) -> BaseException | None:
        with self._lock:
            root = self._root
            synchronize = self._synchronize
        if root is None:
            return None
        if synchronize is None:
            return RuntimeError("pooled decode request lost its synchronization callback")
        try:
            synchronize(root)
        except BaseException as exc:
            return exc
        return None

    def _clear_work(self) -> None:
        with self._lock:
            self._batch = None
            self._executor = None
            self._publication_pending = False
            self._root = None
            self._gate_bound = False
            self._gate_mod = None
            self._native_max = None

    @staticmethod
    def _first(
        first: BaseException | None,
        error: BaseException | None,
    ) -> BaseException | None:
        return first if first is not None else error

    def _abort_cleanup(
        self,
        first: BaseException | None,
    ) -> tuple[BaseException | None, bool]:
        with self._lock:
            batch = self._batch
        if batch is not None:
            try:
                batch.cancel_and_drain()
            except BaseException as exc:
                first = self._first(first, exc)
            if batch.pending:
                first = self._first(
                    first,
                    RuntimeError("pooled writers are not quiescent after cancellation"),
                )
                return first, False

        error = self._signal_abort()
        first = self._first(first, error)
        if error is not None:
            return first, False

        error = self._synchronize_root()
        first = self._first(first, error)
        if error is not None:
            return first, False

        self._clear_work()
        return first, True

    def abort_and_drain(self) -> None:
        """Cancel writers, poison native waits, and synchronize remembered work."""
        self._require_owner_thread()
        error, safe = self._abort_cleanup(None)
        with self._lock:
            detached_cleanup = self._depth == 0
        if safe and detached_cleanup:
            release_error = self._release_domain()
            error = self._first(error, release_error)
            safe = release_error is None
            if safe:
                self._clear_request()
            else:
                with self._lock:
                    self._failure = error
        if not safe and error is None:  # pragma: no cover - defensive invariant
            error = RuntimeError("pooled decode cleanup could not prove quiescence")
        if error is not None:
            raise error

    def _normal_cleanup(self) -> tuple[BaseException | None, bool]:
        error = self._normal_drain()
        if error is not None:
            return self._abort_cleanup(error)
        error = self._synchronize_root()
        if error is not None:
            return self._abort_cleanup(error)
        self._clear_work()
        return None, True

    def _release_domain(self) -> BaseException | None:
        global _DOMAIN_OWNER

        with _DOMAIN_LOCK:
            with self._lock:
                claimed = self._domain_claimed
            if claimed and _DOMAIN_OWNER is not self:
                return RuntimeError("pooled decode event domain ownership was lost")
            if claimed:
                _DOMAIN_OWNER = None
            with self._lock:
                self._domain_claimed = False
        return None

    def _clear_request(self) -> None:
        with self._lock:
            self._owner = None
            self._owner_thread = None
            self._depth = 0
            self._synchronize = None
            self._failure = None

    def _leave(self, body_error: BaseException | None) -> BaseException | None:
        self._require_owner_thread()
        with self._lock:
            if body_error is not None and self._failure is None:
                self._failure = body_error
            self._depth -= 1
            if self._depth > 0:
                return None
            first = self._failure

        if first is None:
            first, safe = self._normal_cleanup()
        else:
            first, safe = self._abort_cleanup(first)
        if safe:
            release_error = self._release_domain()
            first = self._first(first, release_error)
            safe = release_error is None
        if safe:
            self._clear_request()
        else:
            with self._lock:
                self._failure = first
                self._depth = 0
        return first
