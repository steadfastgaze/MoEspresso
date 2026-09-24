from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from moespresso.package.iqk_format import IQK_LAYOUT_IQK_RELAYOUT
from moespresso.runtime.qwen4 import native_publication


IDS = np.arange(10, dtype=np.uint32)
SEQUENCE = 41


class _Iqk:
    def __init__(self) -> None:
        self.blocks = object()

    @staticmethod
    def stream_names() -> tuple[str, ...]:
        return ("blocks",)


class _Pool:
    def __setattr__(self, name, value) -> None:
        if name == "_demand_protect" and getattr(self, "_guard_protection", False):
            assert self._bk_lock.locked()
            object.__setattr__(self, "protection_writes", self.protection_writes + 1)
        object.__setattr__(self, name, value)

    def __init__(self, projection: str, capacity: int, slots: list[int]) -> None:
        self.projection = projection
        self.capacity = capacity
        self.num_experts = 512
        self.spare_slots = 0
        self.geometry = SimpleNamespace(layout=IQK_LAYOUT_IQK_RELAYOUT)
        self._growth_pending = False
        self._loads_inflight = 0
        self._prefetch_inflight = 0
        self._prefetch_reserved: set[int] = set()
        self._bk_lock = threading.Lock()
        self._demand_protect: set[int] = set()
        self.protection_writes = 0
        self._guard_protection = True
        self._slot_of = dict(zip(map(int, IDS), slots, strict=True))
        self._expert_at: list[int | None] = [None] * capacity
        for expert, slot in self._slot_of.items():
            self._expert_at[slot] = expert
        self.total_hits = 0
        self.total_misses = 0
        self.total_loads = 0
        self.total_evictions = 0
        self._clock = 0
        self.iqk = _Iqk()
        self._iqk_views = {"blocks": memoryview(bytearray(4))}
        self._iqk_copy_plan = None


class _Native:
    QWEN_ALL_HIT_ABI = native_publication.ABI

    def __init__(self, status: str = "PUBLISHED") -> None:
        self.status = status
        self.statuses: list[str] = []
        self.value = SEQUENCE - 1
        self.calls: list[dict] = []
        self.after_publish = None
        self.after_poll = None

    def signaled_value(self) -> int:
        return self.value

    def AllHitAttempt(self, ring, maps, destinations, sequence, capacities):
        call = {
            "ring": ring,
            "maps": maps.copy(),
            "destinations_before": np.stack(destinations).copy(),
            "sequence": sequence,
            "capacities": capacities,
            "poll_slices": [],
            "close_calls": 0,
        }
        self.calls.append(call)
        native = self

        class Attempt:
            closed = False

            def poll(self, slice_ns):
                assert not self.closed
                assert all(pool._bk_lock.locked() for pool in native.switch.pools)
                call["poll_slices"].append(slice_ns)
                status = native.statuses.pop(0) if native.statuses else native.status
                if status == "PUBLISHED":
                    ids = ring[8:]
                    for projection, destination in enumerate(destinations):
                        destination[:] = maps[projection, ids]
                    native.value = max(native.value, sequence)
                    if native.after_publish is not None:
                        native.after_publish()
                if native.after_poll is not None:
                    native.after_poll(status)
                call["destinations_after"] = np.stack(destinations).copy()
                return status

            def close(self):
                call["close_calls"] += 1
                self.closed = True

        return Attempt()


class _Switch:
    def __init__(self, native, executor, *, capacity: int = 216) -> None:
        permutations = (
            list(range(10)),
            list(reversed(range(10))),
            [*range(3, 10), *range(3)],
        )
        self.pools = tuple(
            _Pool(projection, capacity, slots)
            for projection, slots in zip(("gate", "up", "down"), permutations, strict=True)
        )
        self.gate_proj = SimpleNamespace(pool=self.pools[0])
        self.up_proj = SimpleNamespace(pool=self.pools[1])
        self.down_proj = SimpleNamespace(pool=self.pools[2])
        self.training = False
        self._prefill_prefetch_enabled = False
        self._prefetch_ticket = None
        self._qwen4_pipe_event = (native, SEQUENCE)
        self._ring_np = np.zeros(18, dtype=np.uint32)
        self._ring_np[0] = SEQUENCE
        self._ring_np[1] = native_publication._checksum(IDS, SEQUENCE)
        self._ring_np[8:] = IDS
        self._ring_buf = object()
        arrays = tuple(object() for _ in range(3))
        views = tuple(memoryview(bytearray(40)) for _ in range(3))
        self._buffers = tuple(item for pair in zip(arrays, views, strict=True) for item in pair)
        self._qwen4_pipe_event_buffers = self._buffers
        self._moespresso_pooled_decode_session = SimpleNamespace(
            active=True,
            _domain_claimed=True,
            _gate_bound=True,
            _gate_mod=native,
            _executor=executor,
            _owner_thread=-1,
        )
        native.switch = self

    def _qwen4_pipe_bufs(self, width: int, *, create: bool):
        assert width == 10
        assert create is False
        return self._buffers

    def account_hits(self, source_ids=IDS) -> None:
        active = set(map(int, source_ids))
        for pool in self.pools:
            with pool._bk_lock:
                pool._demand_protect = set(active)
                for expert in active:
                    assert expert in pool._slot_of
                    pool.total_hits += 1
                    pool._clock += 1

    def original_publish(self, source_ids=IDS) -> None:
        for pool, view in zip(self.pools, self._buffers[1::2], strict=True):
            slots = [pool._slot_of[int(expert)] for expert in source_ids]
            view[:] = np.asarray(slots, dtype=np.uint32).tobytes()

    def destination_rows(self) -> np.ndarray:
        return np.stack(
            [np.frombuffer(view, dtype=np.uint32).copy() for view in self._buffers[1::2]]
        )


@pytest.fixture
def active_runtime(monkeypatch):
    executor = object()
    native = _Native()
    switch = _Switch(native, executor)
    publication = native_publication.Qwen4NativePublication(executor)
    return publication, switch, native, executor


class _Clock:
    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.now
        self.now += self.step
        return current


def test_missing_native_capability_falls_through_without_binding_or_native_work(
    active_runtime,
    monkeypatch,
):
    publication, switch, native, _executor = active_runtime
    monkeypatch.setattr(native, "QWEN_ALL_HIT_ABI", None)

    assert publication.begin(switch, SEQUENCE, 10) is False
    assert native.calls == []
    assert publication.snapshot()["ineligible"] == 1


@pytest.mark.parametrize("capacity", (10, 216, 511))
def test_publication_keeps_three_maps_and_original_lfu_accounting(capacity):
    executor = object()
    native = _Native()
    switch = _Switch(native, executor, capacity=capacity)
    publication = native_publication.Qwen4NativePublication(executor)

    assert publication.begin(switch, SEQUENCE, 10) is True
    assert all(pool._demand_protect == set(map(int, IDS)) for pool in switch.pools)
    assert all(pool.total_hits == pool._clock == 0 for pool in switch.pools)
    switch.account_hits()
    assert publication.suppress(switch, IDS) is True
    publication.finish(completed=True)

    expected = np.asarray(
        [[pool._slot_of[int(expert)] for expert in IDS] for pool in switch.pools],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(switch.destination_rows(), expected)
    assert len({tuple(row) for row in expected}) == 3
    assert native.calls[0]["capacities"] == (capacity,) * 3
    assert native.calls[0]["poll_slices"] == [native_publication._POLL_SLICE_NS]
    assert native.calls[0]["close_calls"] == 1
    assert native.calls[0]["ring"].flags.writeable is False
    assert native.calls[0]["maps"].shape == (3, 512)
    for pool in switch.pools:
        assert pool.total_hits == pool._clock == 10
        assert pool.total_misses == pool.total_loads == pool.total_evictions == 0
        assert pool.protection_writes == 2
    assert publication.snapshot() == {
        "native_calls": 1,
        "published": 1,
        "miss": 0,
        "pending": 0,
        "suppressed": 1,
        "map_builds": 1,
        "ineligible": 0,
        "poll_slices": 1,
        "poll_yields": 0,
        "timed_out": 0,
        "pending_receipts": 0,
    }


class _LegacyNative:
    QWEN_ALL_HIT_ABI = native_publication.ABI


def test_missing_legacy_capability_falls_through_without_native_work():
    executor = object()
    native = _LegacyNative()
    switch = _Switch(native, executor)
    publication = native_publication.Qwen4NativePublication(executor)
    before = switch.destination_rows()

    assert publication.begin(switch, SEQUENCE, 10) is False
    np.testing.assert_array_equal(switch.destination_rows(), before)
    assert publication.snapshot()["ineligible"] == 1
    assert publication.snapshot()["native_calls"] == 0


def test_first_use_requires_graph_owned_projection_buffers(active_runtime):
    publication, switch, native, _executor = active_runtime
    switch._qwen4_pipe_event_buffers = (*switch._buffers[:-1], object())

    with pytest.raises(RuntimeError, match="graph projection buffer ownership"):
        publication.begin(switch, SEQUENCE, 10)
    assert native.calls == []


def test_bound_projection_buffers_cannot_be_replaced_between_calls(active_runtime):
    publication, switch, native, _executor = active_runtime
    native.status = "MISS"
    assert publication.begin(switch, SEQUENCE, 10) is False
    publication.finish(completed=True)

    arrays = tuple(object() for _ in range(3))
    views = tuple(memoryview(bytearray(40)) for _ in range(3))
    switch._buffers = tuple(item for pair in zip(arrays, views, strict=True) for item in pair)
    switch._qwen4_pipe_event_buffers = switch._buffers
    switch._qwen4_pipe_event = (native, SEQUENCE + 1)
    switch._ring_np[0] = SEQUENCE + 1
    switch._ring_np[1] = native_publication._checksum(IDS, SEQUENCE + 1)

    with pytest.raises(RuntimeError, match="projection buffer ownership changed"):
        publication.begin(switch, SEQUENCE + 1, 10)
    assert len(native.calls) == 1


def test_first_use_requires_a_real_ring_owner(active_runtime):
    publication, switch, native, _executor = active_runtime
    switch._ring_buf = None

    with pytest.raises(RuntimeError, match="actual route ring"):
        publication.begin(switch, SEQUENCE, 10)
    assert native.calls == []


def test_miss_does_not_release_or_write_and_original_falls_through():
    executor = object()
    native = _Native("MISS")
    switch = _Switch(native, executor)
    publication = native_publication.Qwen4NativePublication(executor)
    before = switch.destination_rows()

    assert publication.begin(switch, SEQUENCE, 10) is False
    assert native.value == SEQUENCE - 1
    np.testing.assert_array_equal(switch.destination_rows(), before)
    switch.account_hits()
    assert publication.suppress(switch, IDS) is False
    switch.original_publish()
    publication.finish(completed=True)
    assert not np.array_equal(switch.destination_rows(), before)
    assert publication.snapshot()["miss"] == 1
    assert publication.snapshot()["pending"] == 0
    assert publication.snapshot()["suppressed"] == 0


@pytest.mark.parametrize("terminal", ("PUBLISHED", "MISS"))
def test_delayed_readiness_retries_bounded_slices_before_terminal_outcome(
    monkeypatch,
    terminal,
):
    executor = object()
    native = _Native()
    native.statuses = ["PENDING"] * 5 + [terminal]
    switch = _Switch(native, executor)
    publication = native_publication.Qwen4NativePublication(executor, timeout_seconds=1.0)
    monkeypatch.setattr(native_publication.time, "monotonic", _Clock(0.0001))

    assert publication.begin(switch, SEQUENCE, 10) is (terminal == "PUBLISHED")
    stats = publication.snapshot()
    assert stats["native_calls"] == 1
    assert stats["poll_slices"] == 6
    assert stats["poll_yields"] == 5
    assert stats[terminal.lower()] == 1
    assert stats["pending"] == stats["timed_out"] == 0
    assert len(native.calls) == 1
    assert native.calls[0]["poll_slices"] == [native_publication._POLL_SLICE_NS] * 6
    assert native.calls[0]["close_calls"] == 1
    publication.finish(completed=False)


def test_pending_uses_one_global_deadline_and_never_falls_through(monkeypatch):
    executor = object()
    native = _Native("PENDING")
    switch = _Switch(native, executor)
    publication = native_publication.Qwen4NativePublication(executor, timeout_seconds=0.002)
    monkeypatch.setattr(native_publication.time, "monotonic", _Clock(0.001))
    before = switch.destination_rows()

    with pytest.raises(TimeoutError, match="GPU export never became host-visible"):
        publication.begin(switch, SEQUENCE, 10)

    np.testing.assert_array_equal(switch.destination_rows(), before)
    assert native.value == SEQUENCE - 1
    assert native.calls[0]["poll_slices"] == [native_publication._POLL_SLICE_NS]
    assert native.calls[0]["close_calls"] == 1
    assert publication.snapshot() == {
        "native_calls": 1,
        "published": 0,
        "miss": 0,
        "pending": 1,
        "suppressed": 0,
        "map_builds": 1,
        "ineligible": 0,
        "poll_slices": 1,
        "poll_yields": 1,
        "timed_out": 1,
        "pending_receipts": 0,
    }


def test_fast_publication_after_slow_publication_starts_a_fresh_attempt(monkeypatch):
    executor = object()
    native = _Native()
    native.statuses = ["PENDING", "PENDING", "PUBLISHED"]
    switch = _Switch(native, executor)
    publication = native_publication.Qwen4NativePublication(executor, timeout_seconds=1.0)
    monkeypatch.setattr(native_publication.time, "monotonic", _Clock(0.0001))

    assert publication.begin(switch, SEQUENCE, 10) is True
    publication.finish(completed=False)

    next_sequence = SEQUENCE + 1
    switch._qwen4_pipe_event = (native, next_sequence)
    switch._ring_np[0] = next_sequence
    switch._ring_np[1] = native_publication._checksum(IDS, next_sequence)
    native.statuses = ["PUBLISHED"]
    assert publication.begin(switch, next_sequence, 10) is True
    publication.finish(completed=False)

    assert [call["poll_slices"] for call in native.calls] == [
        [native_publication._POLL_SLICE_NS] * 3,
        [native_publication._POLL_SLICE_NS],
    ]
    assert [call["close_calls"] for call in native.calls] == [1, 1]
    assert publication.snapshot()["native_calls"] == 2
    assert publication.snapshot()["published"] == 2
    assert publication.snapshot()["poll_slices"] == 4
    assert publication.snapshot()["poll_yields"] == 2


def test_pending_slice_releases_pool_locks_before_cancellation(active_runtime):
    publication, switch, native, _executor = active_runtime
    native.status = "PENDING"
    observations = []

    def cancelled():
        locked = tuple(pool._bk_lock.locked() for pool in switch.pools)
        observations.append(locked)
        polled = bool(native.calls and native.calls[0]["poll_slices"])
        return polled and not any(locked)

    with pytest.raises(CancelledError, match="after native slice"):
        publication.begin(switch, SEQUENCE, 10, cancelled=cancelled)

    assert (True, True, True) in observations
    assert observations[-1] == (False, False, False)
    assert native.calls[0]["close_calls"] == 1
    assert publication.snapshot()["poll_yields"] == 1
    assert publication.snapshot()["pending"] == 0


def test_load_epoch_change_between_slices_fails_and_closes_one_attempt(active_runtime):
    publication, switch, native, _executor = active_runtime
    native.status = "PENDING"

    def move_resident(_status):
        native.after_poll = None
        pool = switch.pools[0]
        expert = int(IDS[0])
        old = pool._slot_of[expert]
        new = pool._expert_at.index(None)
        pool._expert_at[old] = None
        pool._expert_at[new] = expert
        pool._slot_of[expert] = new
        pool.total_loads += 1
        pool.total_evictions += 1

    native.after_poll = move_resident
    with pytest.raises(RuntimeError, match="epoch|residency|storage|waiting"):
        publication.begin(switch, SEQUENCE, 10)

    assert len(native.calls) == 1
    assert native.calls[0]["close_calls"] == 1
    assert publication.snapshot()["native_calls"] == 1
    assert publication.snapshot()["poll_slices"] == 1
    assert publication.snapshot()["poll_yields"] == 1


@pytest.mark.parametrize("mutation", ("frame", "pool", "ring", "destinations", "storage"))
def test_owned_buffers_cannot_change_between_pending_slices(active_runtime, mutation):
    publication, switch, native, _executor = active_runtime
    native.status = "PENDING"

    def mutate(_status):
        native.after_poll = None
        if mutation == "frame":
            switch._qwen4_pipe_event = (native, SEQUENCE + 1)
        elif mutation == "pool":
            switch.up_proj.pool = _Pool("up", 216, list(range(10)))
        elif mutation == "ring":
            switch._ring_np = switch._ring_np.copy()
        elif mutation == "destinations":
            arrays = tuple(object() for _ in range(3))
            views = tuple(memoryview(bytearray(40)) for _ in range(3))
            switch._buffers = tuple(
                item for pair in zip(arrays, views, strict=True) for item in pair
            )
        else:
            switch.pools[0].iqk.blocks = object()

    native.after_poll = mutate
    with pytest.raises(RuntimeError, match="ownership|storage|sequence|pool"):
        publication.begin(switch, SEQUENCE, 10)

    assert native.calls[0]["close_calls"] == 1
    assert all(not pool._bk_lock.locked() for pool in switch.pools)


def test_attempt_closes_when_poll_raises(active_runtime):
    publication, switch, native, _executor = active_runtime

    def fail(_status):
        raise RuntimeError("injected poll failure")

    native.after_poll = fail
    with pytest.raises(RuntimeError, match="injected poll failure"):
        publication.begin(switch, SEQUENCE, 10)
    assert native.calls[0]["close_calls"] == 1
    assert all(not pool._bk_lock.locked() for pool in switch.pools)


@pytest.mark.parametrize("timeout", (0, -1, True, float("inf"), float("nan"), "1"))
def test_timeout_must_be_positive_finite_number(timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        native_publication.Qwen4NativePublication(object(), timeout_seconds=timeout)


@pytest.mark.parametrize(
    "mutation",
    (
        "inactive",
        "domain",
        "gate-bound",
        "module",
        "executor",
        "owner-thread",
        "event-sequence",
    ),
)
def test_capable_native_rejects_wrong_request_ownership(active_runtime, mutation):
    publication, switch, native, executor = active_runtime
    session = switch._moespresso_pooled_decode_session
    if mutation == "inactive":
        session.active = False
    elif mutation == "domain":
        session._domain_claimed = False
    elif mutation == "gate-bound":
        session._gate_bound = False
    elif mutation == "module":
        session._gate_mod = object()
    elif mutation == "executor":
        session._executor = object()
    elif mutation == "owner-thread":
        session._owner_thread = threading.get_ident()
    else:
        switch._qwen4_pipe_event = (native, SEQUENCE + 1)
    with pytest.raises(RuntimeError, match="sequence or active ownership"):
        publication.begin(switch, SEQUENCE, 10)
    assert publication.snapshot()["native_calls"] == 0
    assert session._executor is executor or mutation == "executor"


@pytest.mark.parametrize("sequence", (0, -1, True, 0x1_0000_0000))
def test_sequence_domain_is_fail_closed(active_runtime, sequence):
    publication, switch, _native, _executor = active_runtime
    with pytest.raises(RuntimeError, match="sequence or active ownership"):
        publication.begin(switch, sequence, 10)


@pytest.mark.parametrize(
    "mutation",
    (
        "source",
        "destination",
        "membership",
        "unselected-map",
        "map-epoch",
        "pool",
        "ring-array",
        "ring-owner",
    ),
)
def test_receipt_or_owned_state_mutation_is_rejected_and_finish_clears(
    active_runtime,
    mutation,
):
    publication, switch, _native, _executor = active_runtime
    assert publication.begin(switch, SEQUENCE, 10) is True
    switch.account_hits()
    source_ids = IDS.copy()
    if mutation == "source":
        source_ids[[0, 1]] = source_ids[[1, 0]]
    elif mutation == "destination":
        np.frombuffer(switch._buffers[1], dtype=np.uint32)[0] = 99
    elif mutation == "membership":
        pool = switch.pools[1]
        with pool._bk_lock:
            expert = int(IDS[0])
            slot = pool._slot_of[expert]
            pool._expert_at[slot] = None
    elif mutation == "unselected-map":
        pool = switch.pools[1]
        with pool._bk_lock:
            slot = pool._expert_at.index(None)
            pool._slot_of[100] = slot
            pool._expert_at[slot] = 100
    elif mutation == "map-epoch":
        switch.pools[1].total_loads += 1
    elif mutation == "pool":
        switch.up_proj.pool = _Pool("up", 216, list(range(10)))
    elif mutation == "ring-array":
        switch._ring_np = switch._ring_np.copy()
    else:
        switch._ring_buf = object()
    try:
        with pytest.raises(
            RuntimeError,
            match="receipt|destinations|residency|pool ownership|storage|ring ownership|membership",
        ):
            publication.suppress(switch, source_ids)
    finally:
        publication.finish(completed=False)
    assert publication.snapshot()["pending_receipts"] == 0


def test_finish_rejects_unconsumed_success_but_still_clears(active_runtime):
    publication, switch, _native, _executor = active_runtime
    assert publication.begin(switch, SEQUENCE, 10) is True
    with pytest.raises(RuntimeError, match="skipped its publication"):
        publication.finish(completed=True)
    assert publication.snapshot()["pending_receipts"] == 0


@pytest.mark.parametrize("when", ("before", "after"))
def test_cancellation_releases_locks_and_finish_clears_receipt(active_runtime, when):
    publication, switch, native, _executor = active_runtime
    cancelled = [when == "before"]
    if when == "after":
        native.after_publish = lambda: cancelled.__setitem__(0, True)
    try:
        with pytest.raises(CancelledError):
            publication.begin(switch, SEQUENCE, 10, cancelled=lambda: cancelled[0])
    finally:
        publication.finish(completed=False)
    assert all(not pool._bk_lock.locked() for pool in switch.pools)
    assert publication.snapshot()["pending_receipts"] == 0
    assert len(native.calls) == (1 if when == "after" else 0)


def test_native_slice_holds_all_pool_locks_under_contention_then_drains(active_runtime):
    publication, switch, native, _executor = active_runtime
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()

    def hold_publication():
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release the native publication")

    def worker():
        try:
            publication.begin(switch, SEQUENCE, 10, cancelled=cancelled.is_set)
        finally:
            publication.finish(completed=False)

    native.after_publish = hold_publication
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        try:
            assert entered.wait(5)
            for pool in switch.pools:
                acquired = pool._bk_lock.acquire(blocking=False)
                if acquired:
                    pool._bk_lock.release()
                assert not acquired
            cancelled.set()
        finally:
            release.set()
        with pytest.raises(CancelledError):
            future.result(timeout=5)
    assert all(not pool._bk_lock.locked() for pool in switch.pools)
    assert publication.snapshot()["pending_receipts"] == 0


@pytest.mark.parametrize(
    "mutation",
    ("capacity-low", "capacity-full", "spare", "growth", "load", "prefetch", "reserved", "layout"),
)
def test_mutable_or_unsupported_pool_state_falls_through(active_runtime, mutation):
    publication, switch, native, _executor = active_runtime
    pool = switch.pools[0]
    if mutation == "capacity-low":
        pool.capacity = 9
    elif mutation == "capacity-full":
        pool.capacity = 512
    elif mutation == "spare":
        pool.spare_slots = 1
    elif mutation == "growth":
        pool._growth_pending = True
    elif mutation == "load":
        pool._loads_inflight = 1
    elif mutation == "prefetch":
        pool._prefetch_inflight = 1
    elif mutation == "reserved":
        pool._prefetch_reserved.add(100)
    else:
        pool.geometry.layout = "other"
    assert publication.begin(switch, SEQUENCE, 10) is False
    assert native.calls == []
    assert publication.snapshot()["ineligible"] == 1


def test_map_epoch_rebuilds_after_a_real_load(active_runtime):
    publication, switch, native, _executor = active_runtime
    native.status = "MISS"
    assert publication.begin(switch, SEQUENCE, 10) is False
    publication.finish(completed=True)
    pool = switch.pools[0]
    with pool._bk_lock:
        expert = int(IDS[0])
        old = pool._slot_of[expert]
        new = pool._expert_at.index(None)
        pool._expert_at[old] = None
        pool._expert_at[new] = expert
        pool._slot_of[expert] = new
        pool.total_loads += 1
        pool.total_evictions += 1
    switch._qwen4_pipe_event = (native, SEQUENCE + 1)
    switch._ring_np[0] = SEQUENCE + 1
    switch._ring_np[1] = native_publication._checksum(IDS, SEQUENCE + 1)
    native.value = SEQUENCE
    assert publication.begin(switch, SEQUENCE + 1, 10) is False
    publication.finish(completed=True)
    assert publication.snapshot()["map_builds"] == 2


def test_close_refuses_active_receipt_then_releases_cached_owners(active_runtime):
    publication, switch, _native, _executor = active_runtime
    assert publication.begin(switch, SEQUENCE, 10) is True
    with pytest.raises(RuntimeError, match="active publication"):
        publication.close()
    publication.finish(completed=False)
    publication.close()
    publication.close()
    assert publication.snapshot()["pending_receipts"] == 0
    with pytest.raises(RuntimeError, match="lifecycle is not idle"):
        publication.begin(switch, SEQUENCE, 10)
