"""Read file byte ranges directly into writable array buffers.

The SSD-streaming runtime needs a miss-loader primitive that does not materialize expert bytes as a
Python `bytes` object and does not rebuild MLX arrays from copied host data. MLX
arrays expose a writable buffer on macOS, and `os.preadv` is a C-level syscall
binding that can fill a memoryview slice in place.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import threading


class PreadIntoError(RuntimeError):
    pass


class PreadIntoUnavailable(PreadIntoError):
    pass


class PreadIntoShortRead(PreadIntoError):
    pass


class PreadFileCache:
    """Small LRU cache of read-only file descriptors for streaming misses."""

    def __init__(self, max_open: int = 64):
        if max_open < 1:
            raise ValueError("max_open must be >= 1")
        self.max_open = int(max_open)
        self._fds: OrderedDict[Path, int] = OrderedDict()
        self._refcounts: dict[Path, int] = {}
        self._lock = threading.Lock()

    def fd(self, path: str | Path) -> int:
        path = Path(path)
        with self._lock:
            fd = self._fds.get(path)
            if fd is not None:
                self._fds.move_to_end(path)
                return fd
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
            self._fds.move_to_end(path)
            self._evict_unreferenced_locked(protected=path)
            return fd

    @contextmanager
    def acquire_fd(self, path: str | Path) -> Iterator[int]:
        path = Path(path)
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                self._fds[path] = fd
            self._fds.move_to_end(path)
            self._refcounts[path] = self._refcounts.get(path, 0) + 1
            self._evict_unreferenced_locked()
        try:
            yield fd
        finally:
            with self._lock:
                remaining = self._refcounts.get(path, 0) - 1
                if remaining > 0:
                    self._refcounts[path] = remaining
                else:
                    self._refcounts.pop(path, None)
                self._evict_unreferenced_locked()

    def _evict_unreferenced_locked(self, *, protected: Path | None = None) -> None:
        while len(self._fds) > self.max_open:
            evicted = False
            for path, fd in list(self._fds.items()):
                if path == protected:
                    continue
                if self._refcounts.get(path, 0):
                    continue
                self._fds.pop(path)
                os.close(fd)
                evicted = True
                break
            if not evicted:
                return

    def close_all(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()
            self._refcounts.clear()


_DEFAULT_FD_CACHE = PreadFileCache()


def _writable_byte_view(dst) -> memoryview:
    view = memoryview(dst)
    if view.readonly:
        raise PreadIntoError("destination buffer is read-only")
    if not view.c_contiguous:
        raise PreadIntoError("destination buffer must be C-contiguous")
    return view.cast("B")


def _pread_view_fd(view: memoryview, fd: int, *, file_offset: int, nbytes: int,
                   dst_offset: int = 0, path_for_error: str | Path = "<fd>") -> int:
    if not hasattr(os, "preadv"):
        raise PreadIntoUnavailable("os.preadv is required for direct buffer reads")
    if file_offset < 0:
        raise ValueError("file_offset must be >= 0")
    if dst_offset < 0:
        raise ValueError("dst_offset must be >= 0")
    if nbytes < 0:
        raise ValueError("nbytes must be >= 0")

    if dst_offset + nbytes > view.nbytes:
        raise ValueError(
            f"read of {nbytes} byte(s) at dst_offset={dst_offset} exceeds "
            f"destination size {view.nbytes}")
    if nbytes == 0:
        return 0

    total = 0
    while total < nbytes:
        start = dst_offset + total
        chunk = view[start:dst_offset + nbytes]
        got = os.preadv(fd, [chunk], file_offset + total)
        if got == 0:
            raise PreadIntoShortRead(
                f"short read from {path_for_error}: got {total} of {nbytes} byte(s)")
        total += got
    return total


def _pread_into_fd(dst, fd: int, *, file_offset: int, nbytes: int,
                   dst_offset: int = 0, path_for_error: str | Path = "<fd>") -> int:
    return _pread_view_fd(
        _writable_byte_view(dst),
        fd,
        file_offset=file_offset,
        nbytes=nbytes,
        dst_offset=dst_offset,
        path_for_error=path_for_error,
    )


def pread_into(dst, path: str | Path, *, file_offset: int, nbytes: int,
               dst_offset: int = 0) -> int:
    """Read exactly `nbytes` from `path:file_offset` into `dst` at `dst_offset`.

    `dst` may be an `mlx.core.array` or any other writable, C-contiguous Python
    buffer provider. Offsets are byte offsets. The implementation uses
    `os.preadv` with a memoryview slice, so no Python `bytes` payload is created.
    """
    fd = os.open(Path(path), os.O_RDONLY)
    try:
        return _pread_into_fd(
            dst,
            fd,
            file_offset=file_offset,
            nbytes=nbytes,
            dst_offset=dst_offset,
            path_for_error=path,
        )
    finally:
        os.close(fd)


def pread_into_cached(dst, path: str | Path, *, file_offset: int, nbytes: int,
                      dst_offset: int = 0,
                      cache: PreadFileCache = _DEFAULT_FD_CACHE) -> int:
    """Read through a bounded fd cache for high-frequency streaming misses."""
    with cache.acquire_fd(path) as fd:
        return _pread_into_fd(
            dst,
            fd,
            file_offset=file_offset,
            nbytes=nbytes,
            dst_offset=dst_offset,
            path_for_error=path,
        )


def pread_view_cached(view: memoryview, path: str | Path, *, file_offset: int,
                      nbytes: int, dst_offset: int = 0,
                      cache: PreadFileCache = _DEFAULT_FD_CACHE) -> int:
    """Read into an already-created writable byte view."""
    if view.readonly:
        raise PreadIntoError("destination buffer is read-only")
    if not view.c_contiguous:
        raise PreadIntoError("destination buffer must be C-contiguous")
    byte_view = view.cast("B") if view.format != "B" else view
    with cache.acquire_fd(path) as fd:
        return _pread_view_fd(
            byte_view,
            fd,
            file_offset=file_offset,
            nbytes=nbytes,
            dst_offset=dst_offset,
            path_for_error=path,
        )


def close_pread_fd_cache() -> None:
    _DEFAULT_FD_CACHE.close_all()
