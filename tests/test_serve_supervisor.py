from __future__ import annotations

import os
import signal
import sys

import pytest

from moespresso import serve_supervisor as supervisor


def _run_sequence(results, seen):
    remaining = list(results)

    def run_worker(argv, *, prog, state):
        seen.append((list(argv), prog, state))
        return remaining.pop(0)

    return run_worker


def test_retries_one_pre_ready_abort_then_returns_success(capsys):
    seen = []
    slept = []
    run_worker = _run_sequence(
        [
            supervisor.WorkerResult(-signal.SIGABRT, False),
            supervisor.WorkerResult(0, True),
        ],
        seen,
    )

    result = supervisor._supervise(
        ["package with spaces", "--thinking", "on"],
        prog="moespresso serve",
        run_worker=run_worker,
        sleep=slept.append,
    )

    assert result == 0
    assert len(seen) == 2
    assert seen[0][0] == ["package with spaces", "--thinking", "on"]
    assert seen[0][1] == "moespresso serve"
    assert slept == [supervisor._RETRY_DELAY_SECONDS]
    assert "retrying once in a fresh process" in capsys.readouterr().err


def test_repeated_pre_ready_abort_stops_after_second_attempt():
    seen = []
    run_worker = _run_sequence(
        [
            supervisor.WorkerResult(-signal.SIGABRT, False),
            supervisor.WorkerResult(-signal.SIGABRT, False),
        ],
        seen,
    )

    result = supervisor._supervise(
        ["pkg"], prog="moespresso-serve", run_worker=run_worker, sleep=lambda _: None
    )

    assert result == 128 + signal.SIGABRT
    assert len(seen) == 2


def test_does_not_retry_abort_after_readiness():
    seen = []
    run_worker = _run_sequence(
        [supervisor.WorkerResult(-signal.SIGABRT, True)], seen
    )

    result = supervisor._supervise(
        ["pkg"], prog="moespresso-serve", run_worker=run_worker
    )

    assert result == 128 + signal.SIGABRT
    assert len(seen) == 1


def test_does_not_retry_ordinary_startup_failure():
    seen = []
    run_worker = _run_sequence([supervisor.WorkerResult(2, False)], seen)

    result = supervisor._supervise(
        ["missing"], prog="moespresso-serve", run_worker=run_worker
    )

    assert result == 2
    assert len(seen) == 1


def test_shutdown_signal_is_forwarded_and_suppresses_retry(monkeypatch):
    forwarded = []

    class Child:
        pid = 417

        def poll(self):
            return None

    def run_worker(_argv, *, prog, state):
        assert prog == "moespresso-serve"
        state.child = Child()
        state.handle(signal.SIGTERM, None)
        return supervisor.WorkerResult(-signal.SIGTERM, False)

    monkeypatch.setattr(
        os, "killpg", lambda pid, signum: forwarded.append((pid, signum))
    )
    result = supervisor._supervise(
        ["pkg"], prog="moespresso-serve", run_worker=run_worker
    )

    assert result == 128 + signal.SIGTERM
    assert forwarded == [(417, signal.SIGTERM)]


def test_worker_preserves_arguments_environment_and_readiness(monkeypatch):
    seen = {}

    class Child:
        pid = 418

        def poll(self):
            return None

        def wait(self):
            return 0

        def send_signal(self, _signum):
            raise AssertionError("no signal expected")

    def fake_popen(command, *, env, pass_fds, process_group):
        seen.update(
            command=command,
            env=env,
            pass_fds=pass_fds,
            process_group=process_group,
        )
        os.write(pass_fds[0], supervisor._READY_BYTE)
        return Child()

    monkeypatch.setattr(supervisor, "_worker_command", lambda argv: ["worker", *argv])
    state = supervisor._SignalState()
    result = supervisor._run_worker(
        ["package with spaces", "--thinking", "on"],
        prog="moespresso serve",
        state=state,
        popen=fake_popen,
        environ={"MOESPRESSO_NATIVE_DIR": "/native"},
    )

    assert result == supervisor.WorkerResult(0, True)
    assert seen["command"] == [
        "worker",
        "package with spaces",
        "--thinking",
        "on",
    ]
    assert seen["env"]["MOESPRESSO_NATIVE_DIR"] == "/native"
    assert seen["env"][supervisor._PROG_ENV] == "moespresso serve"
    assert seen["env"][supervisor._READY_FD_ENV] == str(seen["pass_fds"][0])
    assert seen["env"][supervisor._PARENT_FD_ENV] == str(seen["pass_fds"][1])
    assert seen["process_group"] == 0
    assert state.child is None


def test_real_subprocess_readiness_handshake(monkeypatch):
    code = (
        "import os; "
        f"fd=int(os.environ[{supervisor._READY_FD_ENV!r}]); "
        f"os.write(fd, {supervisor._READY_BYTE!r}); "
        "os.close(fd)"
    )
    monkeypatch.setattr(
        supervisor,
        "_worker_command",
        lambda _argv: [sys.executable, "-c", code],
    )

    result = supervisor._run_worker(
        [], prog="moespresso-serve", state=supervisor._SignalState()
    )

    assert result == supervisor.WorkerResult(0, True)


def test_pending_shutdown_is_forwarded_immediately_after_spawn(monkeypatch):
    forwarded = []

    class Child:
        pid = 419

        def poll(self):
            return None

        def wait(self):
            return -signal.SIGTERM

    def fake_popen(_command, **_kwargs):
        return Child()

    monkeypatch.setattr(
        os, "killpg", lambda pid, signum: forwarded.append((pid, signum))
    )
    state = supervisor._SignalState(shutdown_signal=signal.SIGTERM)

    result = supervisor._run_worker(
        ["pkg"], prog="moespresso-serve", state=state, popen=fake_popen
    )

    assert result == supervisor.WorkerResult(-signal.SIGTERM, False)
    assert forwarded == [(419, signal.SIGTERM)]
    assert state.child is None


def test_shutdown_is_forwarded_only_once_when_attach_races_with_handler(monkeypatch):
    forwarded = []

    class Child:
        pid = 422

        def poll(self):
            return None

    monkeypatch.setattr(
        os, "killpg", lambda pid, signum: forwarded.append((pid, signum))
    )
    state = supervisor._SignalState(shutdown_signal=signal.SIGINT)

    state.attach(Child())
    state.handle(signal.SIGINT, None)

    assert forwarded == [(422, signal.SIGINT)]


def test_post_spawn_supervision_error_terminates_and_reaps_worker(monkeypatch):
    forwarded = []
    waits = []

    class Child:
        pid = 420

        def poll(self):
            return None

        def wait(self, timeout=None):
            waits.append(timeout)
            return -signal.SIGTERM

    def fail_read(_fd, _count):
        raise OSError("read failed")

    monkeypatch.setattr(os, "read", fail_read)
    monkeypatch.setattr(
        os, "killpg", lambda pid, signum: forwarded.append((pid, signum))
    )
    state = supervisor._SignalState()

    with pytest.raises(OSError, match="read failed"):
        supervisor._run_worker(
            ["pkg"],
            prog="moespresso-serve",
            state=state,
            popen=lambda *_args, **_kwargs: Child(),
        )

    assert forwarded == [(420, signal.SIGTERM)]
    assert waits == [10.0]
    assert state.child is None


def test_job_control_signals_follow_the_worker_group(monkeypatch):
    forwarded = []
    self_signals = []
    handler_changes = []

    class Child:
        pid = 421

        def poll(self):
            return None

    monkeypatch.setattr(
        os, "killpg", lambda pid, signum: forwarded.append((pid, signum))
    )
    monkeypatch.setattr(
        os, "kill", lambda pid, signum: self_signals.append((pid, signum))
    )
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: handler_changes.append((signum, handler)),
    )
    state = supervisor._SignalState(child=Child())

    state.handle(signal.SIGTSTP, None)
    state.handle(signal.SIGCONT, None)

    assert forwarded == [
        (421, signal.SIGTSTP),
        (421, signal.SIGCONT),
    ]
    assert self_signals == [(os.getpid(), signal.SIGTSTP)]
    assert handler_changes[0] == (signal.SIGTSTP, signal.SIG_DFL)
    assert handler_changes[1][0] == signal.SIGTSTP
    assert state.shutdown_signal is None


def test_ready_callback_consumes_descriptor_and_signals_once():
    read_fd, write_fd = os.pipe()
    env = {supervisor._READY_FD_ENV: str(write_fd)}
    callback = supervisor.ready_callback_from_env(env)

    assert callback is not None
    assert supervisor._READY_FD_ENV not in env
    assert not os.get_inheritable(write_fd)
    callback()
    callback()
    assert os.read(read_fd, 1) == supervisor._READY_BYTE
    assert os.read(read_fd, 1) == b""
    os.close(read_fd)


def test_parent_watchdog_exits_when_supervisor_pipe_closes():
    parent_fd, parent_hold_fd = os.pipe()
    exits = []
    env = {supervisor._PARENT_FD_ENV: str(parent_fd)}

    thread = supervisor.install_parent_watchdog_from_env(
        env, exit_fn=lambda status: exits.append(status)
    )
    assert thread is not None
    assert supervisor._PARENT_FD_ENV not in env
    assert not os.get_inheritable(parent_fd)
    os.close(parent_hold_fd)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert exits == [1]


def test_worker_command_uses_current_interpreter_without_a_shell():
    assert supervisor._worker_command(["pkg", "--port", "9000"]) == [
        sys.executable,
        "-u",
        "-m",
        "moespresso.runtime.http",
        "pkg",
        "--port",
        "9000",
    ]


def test_worker_prog_is_private_and_consumed():
    env = {supervisor._PROG_ENV: "moespresso serve"}
    assert supervisor.worker_prog_from_env(env) == "moespresso serve"
    assert env == {}
    assert supervisor.worker_prog_from_env({}) == "moespresso-serve"
