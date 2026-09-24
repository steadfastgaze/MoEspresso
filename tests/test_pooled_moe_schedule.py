from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from moespresso.runtime import pooled_moe
from moespresso.runtime import pooled_switchglu as pooled
from moespresso.runtime.pooled_decode_session import PooledDecodeSession
from moespresso.runtime.pooled_load_batch import LoadBatch


def test_install_assigns_one_session_to_every_routed_layer():
    model = SimpleNamespace()
    switches = [SimpleNamespace() for _ in range(3)]
    session = pooled_moe.install_pooled_decode_session(model, switches)
    assert isinstance(session, PooledDecodeSession)
    for owner in (model, *switches):
        assert getattr(owner, pooled_moe._SESSION_ATTRIBUTE, None) is session


def test_install_refuses_replacing_an_existing_session():
    model, switch = SimpleNamespace(), SimpleNamespace()
    session = pooled_moe.install_pooled_decode_session(model, [switch])
    with pytest.raises(RuntimeError, match="already installed"):
        pooled_moe.install_pooled_decode_session(model, [switch])
    assert getattr(model, pooled_moe._SESSION_ATTRIBUTE) is session
    assert getattr(switch, pooled_moe._SESSION_ATTRIBUTE) is session


class _Value:
    def __init__(self, name: str) -> None:
        self.name = name

    def __add__(self, other: _Value) -> _Value:
        return _Value(f"{self.name}+{other.name}")


class _Switch:
    def __init__(self) -> None:
        self.block_exit_kick_calls = 0
        self.decode_moe_block_calls = 0
        self.decode_moe_block_seconds = 0.0
        self.overlap_shared_eval_calls = 0
        self.overlap_shared_eval_seconds = 0.0
        self.overlap_prefill_no_eval_calls = 0
        self.shared_pooled_decode_calls = 0
        self._all_iqk = False
        self.export_inds = lambda indices, sequence: _Value("export-token")
        self.ring_install = lambda sequence, width, gate, *, cancelled: None
        self.begin_projection_load = lambda indices, *, load_owner=None: None

    @staticmethod
    def _projection_pools_lockstep():
        return (SimpleNamespace(capacity=16),)

    def __call__(self, x, indices, *, load_ticket=None):
        if load_ticket is not None:
            load_ticket.used = True
        return _Value("direct-routed")


@pytest.fixture
def schedule_runtime(monkeypatch):
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(pooled, "_PIPELINE_EXECUTOR", executor)
    monkeypatch.setattr(pooled, "_RING_DECODE", True)
    monkeypatch.setattr(pooled, "_RING_SEQ", [0])
    monkeypatch.setattr(pooled, "_token_layers", lambda x: 1)
    monkeypatch.setattr(pooled, "_record_switch_seconds", lambda *args: None)
    monkeypatch.setattr(pooled, "_ring_visibility_ok", lambda: True)
    monkeypatch.setattr(pooled, "_gate_module", lambda: None)
    yield executor
    executor.shutdown(wait=True)


def _run(switch: _Switch, *, resident=None, flush=None, last=False):
    return pooled_moe.run_pooled_moe(
        switch,
        object(),
        SimpleNamespace(shape=(1, 1, 2)),
        object(),
        shared=lambda x: _Value("shared"),
        reduce=lambda routed, scores, indices: _Value("reduced"),
        resident=resident,
        pipelined=lambda x, indices, scores, event_gate: _Value("ring-routed"),
        flush_resident=flush,
        last=last,
    )


def test_ring_submit_precedes_kick_and_kick_failure_cancels_polling_writer(
    monkeypatch,
    schedule_runtime,
) -> None:
    switch = _Switch()
    entered, stopped = threading.Event(), threading.Event()
    error = RuntimeError("kick failed")

    def install(sequence, width, gate, *, cancelled) -> None:
        entered.set()
        while not cancelled():
            stopped.wait(0.001)
        stopped.set()

    def kick(root) -> None:
        assert entered.wait(1), "writer was not submitted before the kick"
        session = getattr(switch, pooled_moe._SESSION_ATTRIBUTE)
        assert session.publication_pending
        raise error

    switch.ring_install = install
    monkeypatch.setattr(pooled, "_kick_eval", kick)
    monkeypatch.setattr(pooled_moe, "_synchronize_root", lambda root: None)

    with pytest.raises(RuntimeError) as raised:
        _run(switch)

    assert raised.value is error
    assert stopped.is_set()
    assert not getattr(switch, pooled_moe._SESSION_ATTRIBUTE).active


def test_ring_v3_publishes_before_the_consumer_graph_is_kicked(
    monkeypatch,
    schedule_runtime,
) -> None:
    switch = _Switch()
    export_kicked, published = threading.Event(), threading.Event()
    events: list[str] = []

    def install(sequence, width, gate, *, cancelled) -> None:
        assert export_kicked.wait(1)
        assert not cancelled()
        events.append("published")
        published.set()

    def kick(root: _Value) -> None:
        if root.name == "export-token":
            events.append("export-kick")
            export_kicked.set()
            return
        assert published.is_set()
        events.append("consumer-kick")

    switch.ring_install = install
    monkeypatch.setattr(pooled, "_kick_eval", kick)
    monkeypatch.setattr(
        pooled_moe,
        "_synchronize_root",
        lambda root: events.append("request-sync"),
    )

    _run(switch, last=True)

    assert events.index("published") < events.index("consumer-kick")
    assert events[-1] == "request-sync"


def test_mixed_resident_path_drains_pending_ring_before_flush(
    monkeypatch,
    schedule_runtime,
) -> None:
    switch = _Switch()
    session = PooledDecodeSession()
    object.__setattr__(switch, pooled_moe._SESSION_ATTRIBUTE, session)
    writer_started, resident_started = threading.Event(), threading.Event()
    release_writer, writer_finished = threading.Event(), threading.Event()

    def install(sequence, width, gate, *, cancelled) -> None:
        writer_started.set()
        release_writer.wait()
        assert not cancelled()
        writer_finished.set()

    def kick(root) -> None:
        assert writer_started.wait(1)

    def resident(x, indices, scores) -> _Value:
        resident_started.set()
        return _Value("resident")

    flushed: list[_Value] = []

    def flush(result: _Value) -> None:
        assert writer_finished.is_set()
        flushed.append(result)

    def release_after_resident() -> None:
        if resident_started.wait(1):
            release_writer.set()

    switch.ring_install = install
    monkeypatch.setattr(pooled, "_kick_eval", kick)
    monkeypatch.setattr(pooled_moe, "_synchronize_root", lambda root: None)
    releaser = threading.Thread(target=release_after_resident)
    releaser.start()
    try:
        with session.request(object(), synchronize=lambda root: None):
            _run(switch)
            assert session.publication_pending
            _run(
                switch,
                resident=resident,
                flush=flush,
            )
            assert not session.publication_pending
    finally:
        release_writer.set()
        releaser.join(1)

    assert writer_finished.is_set()
    assert len(flushed) == 1


def test_standalone_ring_call_drains_before_return(
    monkeypatch,
    schedule_runtime,
) -> None:
    switch = _Switch()
    export_kicked, writer_finished = threading.Event(), threading.Event()
    synchronized: list[_Value] = []

    def install(sequence, width, gate, *, cancelled) -> None:
        assert export_kicked.wait(1)
        assert not cancelled()
        writer_finished.set()

    def kick(root: _Value) -> None:
        assert root.name == "export-token"
        export_kicked.set()

    switch.ring_install = install
    monkeypatch.setattr(pooled, "_kick_eval", kick)
    monkeypatch.setattr(
        pooled_moe,
        "_synchronize_root",
        lambda root: synchronized.append(root),
    )

    _run(switch)

    assert writer_finished.is_set()
    assert len(synchronized) == 1
    assert not getattr(switch, pooled_moe._SESSION_ATTRIBUTE).active


def test_callback_failure_after_overlap_ticket_starts_quiesces_writer(
    monkeypatch,
    schedule_runtime,
) -> None:
    switch = _Switch()
    entered, stopped = threading.Event(), threading.Event()
    error = ValueError("shared callback failed")
    batches: list[LoadBatch] = []

    def begin(indices, *, load_owner):
        batch = load_owner.begin_load_batch(schedule_runtime)
        batches.append(batch)

        def writer() -> None:
            entered.set()
            while not batch.cancelled:
                stopped.wait(0.001)
            stopped.set()

        batch.submit(writer)
        assert entered.wait(1)
        return SimpleNamespace(
            active={1, 2},
            batch=batch,
            has_work=True,
            load_owner=load_owner,
            used=False,
        )

    switch.begin_projection_load = begin
    monkeypatch.setattr(pooled, "_ring_visibility_ok", lambda: False)
    monkeypatch.setattr(pooled, "_kick_eval", lambda root: None)
    monkeypatch.setattr(pooled_moe, "_synchronize_root", lambda root: None)

    def fail_shared(x) -> _Value:
        raise error

    with pytest.raises(ValueError) as raised:
        pooled_moe.run_pooled_moe(
            switch,
            object(),
            SimpleNamespace(shape=(1, 1, 2)),
            object(),
            shared=fail_shared,
            reduce=lambda routed, scores, indices: _Value("reduced"),
        )

    assert raised.value is error
    assert stopped.is_set()
    assert len(batches) == 1
    assert not batches[0].pending


def test_interrupted_ticket_submission_stays_owned_and_cannot_write(
    monkeypatch,
    schedule_runtime,
) -> None:
    queued: list[tuple] = []
    writes: list[set[int]] = []

    class QueueThenInterrupt:
        def submit(self, call, job):
            queued.append((call, job))
            raise KeyboardInterrupt("projection submission interrupted")

    class Pool:
        capacity = 4

        @staticmethod
        def missing_count(active) -> int:
            return len(active)

        @staticmethod
        def ensure(active) -> None:
            writes.append(set(active))

    pool = Pool()
    switch = SimpleNamespace(
        _barrier_free_bulk_shape=lambda indices: False,
        _projection_pools_lockstep=lambda: (pool,),
        _projection_pools=lambda: (pool,),
        index_sync_calls=0,
        index_sync_seconds=0.0,
        overlap_skipped_over_capacity_calls=0,
        overlap_no_miss_calls=0,
        overlap_load_started_calls=0,
    )
    session = PooledDecodeSession()
    monkeypatch.setattr(pooled, "_PROJECTION_LOAD_EXECUTOR", QueueThenInterrupt())

    with pytest.raises(KeyboardInterrupt, match="submission interrupted"):
        with session.request(object(), synchronize=lambda root: None):
            pooled.PooledSwitchGLU.begin_projection_load(
                switch,
                np.array([[[1, 2]]], dtype=np.uint32),
                load_owner=session,
            )

    assert len(queued) == 1
    queued[0][0](queued[0][1])
    assert writes == []
    assert not session.active


def test_owned_projection_ticket_consumption_drains_the_session(
    monkeypatch,
    schedule_runtime,
) -> None:
    loaded: list[set[int]] = []

    class Pool:
        capacity = 4

        @staticmethod
        def missing_count(active) -> int:
            return len(active)

        @staticmethod
        def ensure(active) -> None:
            loaded.append(set(active))

    pool = Pool()
    switch = SimpleNamespace(
        _barrier_free_bulk_shape=lambda indices: False,
        _projection_pools_lockstep=lambda: (pool,),
        _projection_pools=lambda: (pool,),
        index_sync_calls=0,
        index_sync_seconds=0.0,
        overlap_skipped_over_capacity_calls=0,
        overlap_no_miss_calls=0,
        overlap_load_started_calls=0,
        overlap_ticket_mismatch_calls=0,
        projection_load_wait_calls=0,
        projection_load_parallel_calls=0,
        overlap_load_wait_calls=0,
        projection_load_wait_seconds=0.0,
        overlap_load_wait_seconds=0.0,
        overlap_load_total_seconds=0.0,
        overlap_load_hidden_seconds=0.0,
    )
    session = PooledDecodeSession()
    monkeypatch.setattr(pooled, "_PROJECTION_LOAD_EXECUTOR", schedule_runtime)

    with session.request(object(), synchronize=lambda root: None):
        ticket = pooled.PooledSwitchGLU.begin_projection_load(
            switch,
            np.array([[[1, 2]]], dtype=np.uint32),
            load_owner=session,
        )
        assert ticket is not None
        assert session.pending
        assert pooled.PooledSwitchGLU._wait_projection_ticket(
            switch,
            {1, 2},
            ticket,
        )
        assert not session.pending

    assert loaded == [{1, 2}]
    assert ticket.used


def test_poisoned_active_session_rejects_resident_work_before_math(
    monkeypatch,
    schedule_runtime,
) -> None:
    switch = _Switch()
    session = PooledDecodeSession()
    object.__setattr__(switch, pooled_moe._SESSION_ATTRIBUTE, session)
    allow_sync = [False]
    resident_calls: list[str] = []

    def synchronize(root) -> None:
        if not allow_sync[0]:
            raise RuntimeError("sync failed")

    with pytest.raises(RuntimeError, match="sync failed"):
        with session.request(object(), synchronize=synchronize):
            session.remember(_Value("pending-root"))

    assert session.active
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        _run(
            switch,
            resident=lambda x, indices, scores: resident_calls.append("called"),
        )
    assert resident_calls == []

    allow_sync[0] = True
    session.abort_and_drain()
    assert not session.active
