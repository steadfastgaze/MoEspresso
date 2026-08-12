"""Bounded fresh-process recovery for server startup failures."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Mapping, MutableMapping, Sequence


_READY_FD_ENV = "_MOESPRESSO_SERVE_READY_FD"
_PARENT_FD_ENV = "_MOESPRESSO_SERVE_PARENT_FD"
_PROG_ENV = "_MOESPRESSO_SERVE_PROG"
_READY_BYTE = b"\x01"
_RETRY_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class WorkerResult:
    """Exit status and whether the worker reached its public serve boundary."""

    returncode: int
    ready: bool


@dataclass
class _SignalState:
    child: subprocess.Popen | None = None
    shutdown_signal: int | None = None
    forwarded_shutdown_signal: int | None = None

    def attach(self, child: subprocess.Popen) -> None:
        self.forwarded_shutdown_signal = None
        self.child = child
        self._forward_shutdown()

    def detach(self) -> None:
        self.child = None
        self.forwarded_shutdown_signal = None

    def _forward_shutdown(self) -> None:
        signum = self.shutdown_signal
        child = self.child
        if (
            signum is None
            or signum == self.forwarded_shutdown_signal
            or child is None
            or child.poll() is not None
        ):
            return
        self.forwarded_shutdown_signal = signum
        _signal_worker_group(child, signum)

    def handle(self, signum: int, _frame: object) -> None:
        if signum == getattr(signal, "SIGCONT", None):
            child = self.child
            if child is not None and child.poll() is None:
                _signal_worker_group(child, signum)
            return
        if signum == getattr(signal, "SIGTSTP", None):
            child = self.child
            if child is not None and child.poll() is None:
                _signal_worker_group(child, signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            signal.signal(signum, self.handle)
            return
        self.shutdown_signal = signum
        self._forward_shutdown()


def _forwarded_signals() -> tuple[int, ...]:
    names = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT", "SIGTSTP", "SIGCONT")
    return tuple(
        getattr(signal, name) for name in names if hasattr(signal, name)
    )


def _normalize_returncode(returncode: int) -> int:
    if returncode < 0:
        return 128 + abs(returncode)
    return returncode


def _signal_worker_group(child: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def _reap_failed_worker(child: subprocess.Popen) -> None:
    """Stop a worker when supervisor bookkeeping fails after it was spawned."""
    if child.poll() is not None:
        child.wait()
        return
    _signal_worker_group(child, signal.SIGTERM)
    try:
        child.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        _signal_worker_group(child, signal.SIGKILL)
        child.wait()


def _worker_command(argv: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "moespresso.runtime.http",
        *argv,
    ]


def _run_worker(
    argv: Sequence[str],
    *,
    prog: str,
    state: _SignalState,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    environ: Mapping[str, str] | None = None,
) -> WorkerResult:
    read_fd, write_fd = os.pipe()
    parent_fd, parent_hold_fd = os.pipe()
    env = dict(os.environ if environ is None else environ)
    env[_READY_FD_ENV] = str(write_fd)
    env[_PARENT_FD_ENV] = str(parent_fd)
    env[_PROG_ENV] = prog
    child = None
    try:
        try:
            child = popen(
                _worker_command(argv),
                env=env,
                pass_fds=(write_fd, parent_fd),
                process_group=0,
            )
            state.attach(child)
        except BaseException:
            os.close(read_fd)
            os.close(parent_hold_fd)
            if child is not None:
                _reap_failed_worker(child)
            raise
    finally:
        os.close(write_fd)
        os.close(parent_fd)

    ready = False
    try:
        try:
            ready = os.read(read_fd, 1) == _READY_BYTE
        finally:
            os.close(read_fd)
        returncode = child.wait()
        return WorkerResult(returncode=returncode, ready=ready)
    except BaseException:
        _reap_failed_worker(child)
        raise
    finally:
        state.detach()
        os.close(parent_hold_fd)


def _supervise(
    argv: Sequence[str],
    *,
    prog: str,
    run_worker: Callable[..., WorkerResult] = _run_worker,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    state = _SignalState()
    previous_handlers: MutableMapping[int, object] = {}
    for signum in _forwarded_signals():
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, state.handle)
    try:
        for attempt in range(2):
            if state.shutdown_signal is not None:
                return 128 + state.shutdown_signal
            try:
                result = run_worker(argv, prog=prog, state=state)
            except OSError as exc:
                print(f"FAILED: serve worker supervision failed: {exc}", flush=True)
                return 1

            if state.shutdown_signal is not None:
                if result.returncode == 0:
                    return 0
                return _normalize_returncode(result.returncode)
            retryable = (
                attempt == 0
                and not result.ready
                and result.returncode == -signal.SIGABRT
            )
            if not retryable:
                return _normalize_returncode(result.returncode)
            print(
                "[serve] startup worker aborted before readiness; "
                "retrying once in a fresh process",
                file=sys.stderr,
                flush=True,
            )
            sleep(_RETRY_DELAY_SECONDS)
            if state.shutdown_signal is not None:
                return 128 + state.shutdown_signal
    finally:
        state.detach()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    raise AssertionError("bounded serve supervision exhausted unexpectedly")


def ready_callback_from_env(
    environ: MutableMapping[str, str] | None = None,
) -> Callable[[], None] | None:
    """Consume the worker handshake descriptor and return an idempotent signal."""
    env = os.environ if environ is None else environ
    raw_fd = env.pop(_READY_FD_ENV, None)
    if raw_fd is None:
        return None
    try:
        fd = int(raw_fd)
    except ValueError as exc:
        raise RuntimeError("invalid internal serve readiness descriptor") from exc
    if fd < 0:
        raise RuntimeError("invalid internal serve readiness descriptor")
    os.set_inheritable(fd, False)
    signalled = False

    def signal_ready() -> None:
        nonlocal signalled
        if signalled:
            return
        try:
            os.write(fd, _READY_BYTE)
        finally:
            os.close(fd)
            signalled = True

    return signal_ready


def install_parent_watchdog_from_env(
    environ: MutableMapping[str, str] | None = None,
    *,
    exit_fn: Callable[[int], object] = os._exit,
):
    """Exit the worker if its supervisor closes or loses its lifetime pipe."""
    import threading

    env = os.environ if environ is None else environ
    raw_fd = env.pop(_PARENT_FD_ENV, None)
    if raw_fd is None:
        return None
    try:
        fd = int(raw_fd)
    except ValueError as exc:
        raise RuntimeError("invalid internal serve parent descriptor") from exc
    if fd < 0:
        raise RuntimeError("invalid internal serve parent descriptor")
    os.set_inheritable(fd, False)

    def watch_parent() -> None:
        try:
            while os.read(fd, 1):
                pass
        except OSError:
            pass
        finally:
            os.close(fd)
        exit_fn(1)

    thread = threading.Thread(
        target=watch_parent,
        name="moespresso-serve-parent-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def worker_prog_from_env(
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """Consume the private parser name passed from the public command."""
    env = os.environ if environ is None else environ
    return env.pop(_PROG_ENV, "moespresso-serve")


def main(
    argv: list[str] | None = None, *, prog: str = "moespresso-serve"
) -> int:
    """Run the HTTP worker with one bounded pre-readiness abort retry."""
    args = list(sys.argv[1:] if argv is None else argv)
    return _supervise(args, prog=prog)


if __name__ == "__main__":
    raise SystemExit(main())
