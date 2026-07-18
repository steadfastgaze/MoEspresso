"""Single-owner process lock for SSD-streaming model runs.

SSD-streaming probes and servers are memory-sensitive experiments on hosts with
limited unified memory. Two concurrent runs can exceed unified-memory headroom
even when each run is safe by itself, so the streaming builder takes a
non-blocking process lock and holds it for the model lifetime.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_LOCK_PATH = Path("/tmp/moespresso-ssd-streaming.lock")
ALLOW_PARALLEL_ENV = "MOESPRESSO_ALLOW_PARALLEL_SSD_STREAMING"
LOCK_PATH_ENV = "MOESPRESSO_SSD_STREAMING_LOCK_PATH"


class SSDStreamingAlreadyRunning(RuntimeError):
    """Another MoEspresso SSD-streaming process owns the local run lock."""


class SSDStreamingProcessLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> None:
        import errno
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise SSDStreamingAlreadyRunning(
                    f"another SSD-streaming run owns {self.path}; run one "
                    "real-model experiment at a time, or set "
                    f"{ALLOW_PARALLEL_ENV}=1 if you are deliberately "
                    "overriding the memory guard") from exc
            raise
        self._fd = fd

    def close(self) -> None:
        if self._fd is None:
            return
        import fcntl

        fd = self._fd
        self._fd = None
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def acquire_ssd_streaming_process_lock() -> SSDStreamingProcessLock | None:
    if os.environ.get(ALLOW_PARALLEL_ENV) == "1":
        return None
    lock = SSDStreamingProcessLock(os.environ.get(LOCK_PATH_ENV, DEFAULT_LOCK_PATH))
    lock.acquire()
    return lock
