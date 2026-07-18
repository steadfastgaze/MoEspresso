from __future__ import annotations

from moespresso.runtime.streaming_run_lock import (
    ALLOW_PARALLEL_ENV,
    LOCK_PATH_ENV,
    SSDStreamingAlreadyRunning,
    SSDStreamingProcessLock,
    acquire_ssd_streaming_process_lock,
)


def test_ssd_streaming_process_lock_rejects_second_owner(tmp_path):
    path = tmp_path / "streaming.lock"
    first = SSDStreamingProcessLock(path)
    first.acquire()
    try:
        second = SSDStreamingProcessLock(path)
        try:
            second.acquire()
        except SSDStreamingAlreadyRunning as exc:
            assert str(path) in str(exc)
        else:  # pragma: no cover - the assertion above is the expected path
            raise AssertionError("second lock unexpectedly acquired")
    finally:
        first.close()


def test_ssd_streaming_process_lock_releases_on_close(tmp_path):
    path = tmp_path / "streaming.lock"
    first = SSDStreamingProcessLock(path)
    first.acquire()
    first.close()

    second = SSDStreamingProcessLock(path)
    try:
        second.acquire()
    finally:
        second.close()


def test_ssd_streaming_process_lock_can_be_explicitly_bypassed(monkeypatch, tmp_path):
    monkeypatch.setenv(ALLOW_PARALLEL_ENV, "1")
    monkeypatch.setenv(LOCK_PATH_ENV, str(tmp_path / "streaming.lock"))

    assert acquire_ssd_streaming_process_lock() is None
