"""Served-process lifecycle and shared-GPU coordination for the road-test.

The road-test owns its server: it starts ``moespresso-serve`` with the disk
KV store enabled, waits for ``/health``, restarts the process mid-session for
the resume proofs, and shuts it down at the end. Serve output streams to a
per-segment log file so the operator checkpoint lines land in the run record.

Measurement agents sharing one GPU serialize model loads through an
atomic-mkdir lock directory: the holder records its pid in ``holder.txt``, a
stale lock whose holder pid is dead is broken, and a pending ``.request``
file asks the current holder for a timing-exclusive window, so a new load
yields until it clears. The road-test holds the lock across its whole run,
server restarts included, because the served model owns the GPU for the
duration.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from moespresso.agentlib.client import ClientError, CompletionsClient

DEFAULT_GPU_LOCKDIR = "/private/tmp/ornith_gpu.lockdir"
GPU_LOCK_HOLDER = "roadtest"


class GpuLock:
    """Atomic-mkdir GPU lock following the shared measurement convention."""

    def __init__(self, lockdir: str | Path = DEFAULT_GPU_LOCKDIR,
                 *, holder: str = GPU_LOCK_HOLDER,
                 poll_seconds: float = 30.0):
        self.lockdir = Path(lockdir)
        self.request_path = Path(str(lockdir) + ".request")
        self.holder = holder
        self.poll_seconds = poll_seconds
        self.held = False

    def _holder_pid(self) -> int | None:
        try:
            for line in (self.lockdir / "holder.txt").read_text().splitlines():
                if line.startswith("pid="):
                    return int(line.split("=", 1)[1].strip())
        except (OSError, ValueError):
            return None
        return None

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def acquire(self, *, timeout: float = 4 * 3600.0,
                log_fn=print) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.request_path.exists():
                log_fn(f"[roadtest] yielding: {self.request_path} present")
                time.sleep(self.poll_seconds)
                continue
            try:
                self.lockdir.mkdir()
            except FileExistsError:
                pid = self._holder_pid()
                if pid is not None and not self._pid_alive(pid):
                    log_fn(f"[roadtest] breaking stale GPU lock (dead pid {pid})")
                    shutil.rmtree(self.lockdir, ignore_errors=True)
                    continue
                time.sleep(self.poll_seconds)
                continue
            (self.lockdir / "holder.txt").write_text(
                f"agent={self.holder}\npid={os.getpid()}\nstep=roadtest\n")
            self.held = True
            return
        raise TimeoutError(f"GPU lock not acquired within {timeout:g}s: {self.lockdir}")

    def release(self) -> None:
        if self.held:
            shutil.rmtree(self.lockdir, ignore_errors=True)
            self.held = False


class ServerController:
    """Start, health-check, restart, and stop the served model process."""

    def __init__(
        self,
        *,
        package_dir: Path,
        repo_root: Path,
        port: int,
        disk_root: Path,
        stride: int,
        budget_bytes: int | None,
        log_dir: Path,
        health_timeout: float = 2700.0,
    ):
        self.package_dir = Path(package_dir)
        self.repo_root = Path(repo_root)
        self.port = int(port)
        self.disk_root = Path(disk_root)
        self.stride = int(stride)
        self.budget_bytes = budget_bytes
        self.log_dir = Path(log_dir)
        self.health_timeout = float(health_timeout)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.segment = 0
        self._process: subprocess.Popen | None = None
        self._log_handle = None

    def _serve_env(self) -> dict:
        env = dict(os.environ)
        env["MOESPRESSO_DISK_KV"] = "frontier"
        env["MOESPRESSO_DISK_KV_ROOT"] = str(self.disk_root)
        env["MOESPRESSO_DISK_KV_STRIDE"] = str(self.stride)
        if self.budget_bytes is not None:
            env["MOESPRESSO_DISK_KV_BYTES"] = str(self.budget_bytes)
        return env

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("server already running")
        self.segment += 1
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"serve_segment_{self.segment:02d}.log"
        self._log_handle = open(log_path, "ab")
        command = [
            "uv", "run", "--locked",
            "moespresso-serve", str(self.package_dir),
            "--host", "127.0.0.1", "--port", str(self.port),
        ]
        self._process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=self._serve_env(),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def rss_bytes(self) -> int | None:
        """Resident set size of the serve process tree, in bytes.

        The serve command runs through a wrapper, so the model process is a
        child; the sum covers the whole process group the controller
        started. Returns None when no process is running or ``ps`` fails.
        """
        process = self._process
        if process is None or process.poll() is not None:
            return None
        try:
            pgid = os.getpgid(process.pid)
            listing = subprocess.run(
                ["ps", "-axo", "pgid=,rss="],
                capture_output=True, text=True, timeout=10.0, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        total_kb = 0
        for line in listing.stdout.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == str(pgid):
                try:
                    total_kb += int(fields[1])
                except ValueError:
                    continue
        return total_kb * 1024 if total_kb else None

    def wait_healthy(self, *, timeout: float | None = None) -> dict:
        """Poll ``/health`` until the server answers; return the payload."""
        timeout = self.health_timeout if timeout is None else timeout
        client = CompletionsClient(self.base_url, timeout=30.0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"server segment {self.segment} exited with code "
                    f"{self._process.returncode} before /health; see "
                    f"{self.log_dir}/serve_segment_{self.segment:02d}.log")
            try:
                return client.health()
            except ClientError:
                time.sleep(5.0)
        raise TimeoutError(f"server not healthy within {timeout:g}s")

    def stop(self, *, timeout: float = 300.0) -> None:
        """Interrupt the serve process group and wait for a clean exit."""
        process = self._process
        if process is None:
            return
        self._process = None
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            pgid = None
        for sig, grace in ((signal.SIGINT, timeout), (signal.SIGTERM, 30.0),
                           (signal.SIGKILL, 30.0)):
            if process.poll() is not None:
                break
            if pgid is not None:
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
            else:
                process.send_signal(sig)
            try:
                process.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
