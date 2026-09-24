"""Direct tests for the native Qwen all-hit attempt lifecycle."""

from __future__ import annotations

import gc
import importlib
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


ABI = "qwen-native-all-hit-gate-v2"
MISSING = 512
CAPACITIES = (216, 216, 216)


@pytest.fixture(scope="module")
def native_gate():
    try:
        module = importlib.import_module("moespresso._native._moespresso_gate")
    except (ImportError, OSError) as exc:
        if os.environ.get("MOESPRESSO_REQUIRE_NATIVE_GATE") == "1":
            pytest.fail(f"required native gate could not be imported: {exc}")
        pytest.skip("native gate is not built in this environment")
    assert module.QWEN_ALL_HIT_ABI == ABI
    return module


def _checksum(ids, sequence):
    value = 2166136261
    for expert in ids:
        value = ((value ^ int(expert)) * 16777619) & 0xFFFFFFFF
    return ((value ^ sequence) * 16777619) & 0xFFFFFFFF


@pytest.mark.parametrize("count", [1, 10, 20, 64])
def test_native_ring_reader_copies_complete_export_without_writing(native_gate, count):
    ids = np.arange(count, dtype=np.uint32) * 7
    ring = np.zeros(8 + count, dtype=np.uint32)
    view = memoryview(ring).toreadonly()
    assert native_gate.read_exported_ids(view, 7, count, 0) is None
    ring[8:] = ids
    ring[1] = _checksum(ids, 7) ^ 1
    ring[0] = 7
    assert native_gate.read_exported_ids(view, 7, count, 0) is None
    ring[1] = _checksum(ids, 7)
    expected_ring = ring.copy()
    assert native_gate.read_exported_ids(view, 7, count, 0) == ids.tolist()
    np.testing.assert_array_equal(ring, expected_ring)
    with pytest.raises(RuntimeError, match="advanced beyond"):
        native_gate.read_exported_ids(view, 6, count, 0)


@pytest.mark.parametrize("field,value", [
    ("sequence", 0), ("sequence", 1 << 32), ("sequence", True),
    ("count", 0), ("count", 65), ("count", True),
    ("wait_ns", 1_000_001), ("wait_ns", True),
])
def test_native_ring_reader_bounds(native_gate, field, value):
    args = {"sequence": 1, "count": 20, "wait_ns": 0}
    args[field] = value
    with pytest.raises(ValueError):
        native_gate.read_exported_ids(memoryview(np.zeros(28, dtype=np.uint32)).toreadonly(), **args)


@pytest.mark.parametrize("buffer", [
    np.zeros(28, dtype=np.uint32),
    memoryview(np.zeros(18, dtype=np.uint32)).toreadonly(),
    memoryview(np.zeros(28, dtype=np.float32)).toreadonly(),
    memoryview(np.zeros((2, 14), dtype=np.uint32)).toreadonly(),
    memoryview(np.zeros(113, dtype=np.uint8)[1:].view(np.uint32)).toreadonly(),
])
def test_native_ring_reader_rejects_invalid_buffer(native_gate, buffer):
    with pytest.raises(ValueError, match="read-only aligned"):
        native_gate.read_exported_ids(buffer, 1, 20, 0)


def test_native_ring_wait_releases_gil_for_python_publication(native_gate):
    import sys

    ring = np.zeros(28, dtype=np.uint32)
    view = memoryview(ring).toreadonly()
    started, go = threading.Event(), threading.Event()
    ids = np.arange(20, dtype=np.uint32)

    def publish():
        started.set()
        go.wait()
        ring[8:] = ids
        ring[1] = _checksum(ids, 9)
        ring[0] = 9

    writer = threading.Thread(target=publish)
    writer.start()
    assert started.wait(timeout=1)
    interval = sys.getswitchinterval()
    result = None
    try:
        sys.setswitchinterval(1)
        go.set()
        for _ in range(50):
            result = native_gate.read_exported_ids(view, 9, 20, 1_000_000)
            if result is not None:
                break
        assert result == ids.tolist()
    finally:
        sys.setswitchinterval(interval)
        go.set()
        writer.join(timeout=1)


def _case(module, *, missing=False):
    sequence = int(module.signaled_value()) + 1
    ids = np.arange(10, dtype=np.uint32)
    ring = np.zeros(18, dtype=np.uint32)
    ring_view = ring.view()
    ring_view.flags.writeable = False
    maps = np.full((3, 512), MISSING, dtype=np.uint32)
    maps[0, ids] = np.arange(10, dtype=np.uint32)
    maps[1, ids] = np.arange(9, -1, -1, dtype=np.uint32)
    maps[2, ids] = np.roll(np.arange(10, dtype=np.uint32), 3)
    if missing:
        maps[1, int(ids[-1])] = MISSING
    maps.flags.writeable = False
    destinations = tuple(np.full(10, 77, dtype=np.uint32) for _ in range(3))
    attempt = module.AllHitAttempt(ring_view, maps, destinations, sequence, CAPACITIES)
    return SimpleNamespace(
        module=module,
        sequence=sequence,
        ids=ids,
        ring=ring,
        maps=maps,
        destinations=destinations,
        attempt=attempt,
    )


def _write_ring(case, ids=None):
    ids = case.ids if ids is None else np.asarray(ids, dtype=np.uint32)
    case.ring[8:] = ids
    case.ring[1] = _checksum(ids, case.sequence)
    case.ring[0] = case.sequence


def _poll_until_terminal(case, timeout=1.0):
    deadline = time.monotonic() + timeout
    statuses = []
    while time.monotonic() < deadline:
        status = case.attempt.poll(1_000_000)
        statuses.append(status)
        if status != "PENDING":
            return status, statuses
    pytest.fail("native all-hit attempt did not reach a terminal outcome")


def test_delayed_ring_publication_has_no_early_write_or_signal(native_gate):
    case = _case(native_gate)
    frontier = int(native_gate.signaled_value())
    before = np.stack(case.destinations).copy()
    assert case.attempt.poll(100_000) == "PENDING"
    np.testing.assert_array_equal(np.stack(case.destinations), before)
    assert native_gate.signaled_value() == frontier

    writer = threading.Thread(target=lambda: (time.sleep(0.01), _write_ring(case)))
    writer.start()
    try:
        status, _statuses = _poll_until_terminal(case)
    finally:
        writer.join(timeout=1)
    assert status == "PUBLISHED"
    np.testing.assert_array_equal(np.stack(case.destinations), case.maps[:, case.ids])
    assert native_gate.signaled_value() == case.sequence
    with pytest.raises(RuntimeError, match="attempt is terminal"):
        case.attempt.poll(0)
    case.attempt.close()
    case.attempt.close()


def test_delayed_miss_does_not_publish_or_release(native_gate):
    case = _case(native_gate, missing=True)
    frontier = int(native_gate.signaled_value())
    before = np.stack(case.destinations).copy()
    assert case.attempt.poll(0) == "PENDING"

    writer = threading.Thread(target=lambda: (time.sleep(0.005), _write_ring(case)))
    writer.start()
    try:
        status, _statuses = _poll_until_terminal(case)
    finally:
        writer.join(timeout=1)
    assert status == "MISS"
    assert native_gate.signaled_value() == frontier
    np.testing.assert_array_equal(np.stack(case.destinations), before)
    case.attempt.close()


def test_stale_and_torn_routes_remain_pending(native_gate):
    stale = _case(native_gate)
    assert stale.attempt.poll(0) == "PENDING"
    stale.attempt.close()

    torn = _case(native_gate)
    torn.ring[8:] = torn.ids
    torn.ring[1] = _checksum(torn.ids, torn.sequence) ^ 1
    torn.ring[0] = torn.sequence
    assert torn.attempt.poll(0) == "PENDING"
    torn.attempt.close()


@pytest.mark.parametrize("failure", ("future", "invalid"))
def test_future_or_invalid_routes_fail_closed(native_gate, failure):
    case = _case(native_gate)
    if failure == "future":
        case.ring[0] = case.sequence + 1
        error = RuntimeError
        match = "advanced beyond"
    else:
        ids = case.ids.copy()
        ids[-1] = 512
        _write_ring(case, ids)
        error = ValueError
        match = "out-of-range expert"
    with pytest.raises(error, match=match):
        case.attempt.poll(0)
    case.attempt.close()


def test_constructor_rejects_alias_and_invalid_buffer_contracts(native_gate):
    sequence = int(native_gate.signaled_value()) + 1
    destinations = tuple(np.zeros(10, dtype=np.uint32) for _ in range(3))
    maps = np.full((3, 512), MISSING, dtype=np.uint32)
    maps.flags.writeable = False
    writable_ring = np.zeros(18, dtype=np.uint32)
    with pytest.raises(ValueError, match="ring must be exact read-only"):
        native_gate.AllHitAttempt(writable_ring, maps, destinations, sequence, CAPACITIES)

    ring = writable_ring.view()
    ring.flags.writeable = False
    writable_maps = maps.copy()
    with pytest.raises(ValueError, match="maps must be exact read-only"):
        native_gate.AllHitAttempt(ring, writable_maps, destinations, sequence, CAPACITIES)

    readonly_destination = np.zeros(10, dtype=np.uint32)
    readonly_destination.flags.writeable = False
    with pytest.raises(ValueError, match="read-only|destination must be exact writable"):
        native_gate.AllHitAttempt(
            ring,
            maps,
            (readonly_destination, *destinations[1:]),
            sequence,
            CAPACITIES,
        )

    invalid_maps = np.full((3, 512), MISSING, dtype=np.uint32)
    invalid_maps[0, :2] = 0
    invalid_maps.flags.writeable = False
    with pytest.raises(ValueError, match="map values or slot uniqueness"):
        native_gate.AllHitAttempt(ring, invalid_maps, destinations, sequence, CAPACITIES)

    shared = np.full((3, 512), MISSING, dtype=np.uint32)
    shared_maps = shared.view()
    shared_maps.flags.writeable = False
    shared_ring = shared.reshape(-1)[:18]
    shared_ring.flags.writeable = False
    with pytest.raises(ValueError, match="must not alias"):
        native_gate.AllHitAttempt(shared_ring, shared_maps, destinations, sequence, CAPACITIES)


@pytest.mark.parametrize("slice_ns", (True, -1, 1_000_001))
def test_poll_rejects_invalid_slices(native_gate, slice_ns):
    case = _case(native_gate)
    with pytest.raises((ValueError, OverflowError)):
        case.attempt.poll(slice_ns)
    case.attempt.close()


def test_poll_and_close_validate_owner_and_closed_state(native_gate):
    case = _case(native_gate)
    errors = []

    def wrong_owner():
        for operation in (lambda: case.attempt.poll(0), case.attempt.close):
            try:
                operation()
            except RuntimeError as exc:
                errors.append(str(exc))

    thread = threading.Thread(target=wrong_owner)
    thread.start()
    thread.join(timeout=1)
    assert errors == [
        "native all-hit attempt is owned by another thread",
        "native all-hit attempt is owned by another thread",
    ]
    case.attempt.close()
    with pytest.raises(RuntimeError, match="attempt is closed"):
        case.attempt.poll(0)


def test_close_releases_every_buffer_export(native_gate):
    sequence = int(native_gate.signaled_value()) + 1
    ring_owner = bytearray(18 * 4)
    maps_owner = bytearray(3 * 512 * 4)
    destination_owners = [bytearray(10 * 4) for _ in range(3)]
    maps_array = np.frombuffer(maps_owner, dtype=np.uint32).reshape(3, 512)
    maps_array[:] = MISSING
    del maps_array
    ring_view = memoryview(ring_owner).cast("I").toreadonly()
    maps_view = memoryview(maps_owner).cast("I", shape=[3, 512]).toreadonly()
    destination_views = tuple(memoryview(owner).cast("I") for owner in destination_owners)
    attempt = native_gate.AllHitAttempt(
        ring_view, maps_view, destination_views, sequence, CAPACITIES
    )
    del ring_view, maps_view, destination_views
    gc.collect()

    for owner in (ring_owner, maps_owner, *destination_owners):
        with pytest.raises(BufferError):
            owner.extend(b"\x00")

    attempt.close()
    attempt.close()
    for owner in (ring_owner, maps_owner, *destination_owners):
        owner.extend(b"\x00")


def test_close_during_native_poll_fails(native_gate):
    shared = {}
    ready = threading.Event()
    stop = threading.Event()

    def worker():
        case = _case(native_gate)
        shared["case"] = case
        ready.set()
        try:
            while not stop.is_set():
                shared["status"] = case.attempt.poll(1_000_000)
        finally:
            case.attempt.close()

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(1)
    try:
        with pytest.raises(RuntimeError, match="active poll|owned by another thread"):
            shared["case"].attempt.close()
    finally:
        stop.set()
        thread.join(timeout=1)
    assert not thread.is_alive()
    assert shared.get("status", "PENDING") == "PENDING"
