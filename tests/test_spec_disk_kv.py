"""Model-specific DSpark integration for optional disk-KV attachments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from moespresso.runtime.deepseek_v4.spec_decode import (  # noqa: E402
    DrafterStateCapsule,
    SpecPrefillProgress,
)
from moespresso.runtime.deepseek_v4.spec_disk_kv import (  # noqa: E402
    SPEC_DISK_ATTACHMENT_KIND,
    build_spec_attachment_identity,
    make_spec_disk_kv_writer,
    restore_spec_attachment,
)
from moespresso.runtime.deepseek_v4.spec_serve import (  # noqa: E402
    SPEC_CACHE_SCHEMA,
    SPEC_PRODUCER_LATTICE,
    ServedDrafter,
    spec_cache_producer_rail,
)
from moespresso.runtime.disk_kv import (  # noqa: E402
    ATTACHMENT_STATUS_HIT,
    ATTACHMENT_STATUS_INVALID,
    ATTACHMENT_STATUS_MISSING,
    ATTACHMENT_STATUS_UNAVAILABLE,
    DiskKVAttachmentEntry,
    DiskKVAttachmentEnvelope,
    DiskKVAttachmentLookup,
    DiskKVEntry,
    DiskKVHit,
)


_SCOPE = {"schema_version": "moespresso-disk-kv-v1", "test": "spec-disk"}


@dataclass
class _State:
    frontier: int


class _FakeDrafter:
    state_capsule_kind = "deepseek_v4_dspark_window_state"
    state_capsule_schema_major = 1
    state_capsule_schema_minor = 0

    def __init__(self):
        self.export_calls = 0
        self.import_calls = 0
        self.import_error: Exception | None = None
        self.import_frontier: int | None = None
        self.export_frontier: int | None = None

    def state_frontier(self, state):
        return state.frontier

    def state_nbytes(self, state):
        return 0

    def export_state(self, state):
        self.export_calls += 1
        frontier = state.frontier if self.export_frontier is None else self.export_frontier
        return DrafterStateCapsule(
            kind=self.state_capsule_kind,
            schema_major=self.state_capsule_schema_major,
            schema_minor=self.state_capsule_schema_minor,
            frontier=frontier,
            metadata=(("stages", 3),),
            tensors=(),
        )

    def import_state(self, capsule):
        self.import_calls += 1
        if self.import_error is not None:
            raise self.import_error
        frontier = capsule.frontier if self.import_frontier is None else self.import_frontier
        return _State(frontier)


class _Cache:
    def __init__(self, offset):
        self.offset = offset
        self.state = (f"state-{offset}",)
        self.meta_state = (offset,)


class _FakeStore:
    def __init__(self, *, stride=256, write_depth=None):
        self.stride = stride
        self.write_depth_tokens = write_depth
        self.writes_disabled = False
        self.attachments_available = True
        self.targets = {}
        self.attachment_ids = set()
        self.call_order = []
        self.restore_result = DiskKVAttachmentLookup(
            status=ATTACHMENT_STATUS_MISSING)
        self.quarantined = []
        self.target_write_results = []
        self.attachment_write_results = []
        self.find_error: Exception | None = None
        self.has_error: Exception | None = None
        self.logged = []

    def _log(self, line):
        self.logged.append(line)

    def find_exact(self, scope, tokens):
        if self.find_error is not None:
            raise self.find_error
        return self.targets.get(tuple(tokens))

    def has_attachment(self, target, identity):
        if self.has_error is not None:
            raise self.has_error
        return identity.attachment_id in self.attachment_ids

    def restore_attachment(self, target, identity):
        self.call_order.append(("restore_attachment", target.cache_id))
        return self.restore_result

    def quarantine_attachment(self, entry, *, reason):
        self.quarantined.append((entry, reason))
        self.attachment_ids.discard(entry.attachment_id)

    def write_checkpoint(
        self,
        scope,
        tokens,
        *,
        cache_state_trees,
        meta_state_trees,
        cache_class_names,
        reason,
        session_cache_key,
        now,
    ):
        frontier = len(tokens)
        self.call_order.append(("target", frontier))
        if self.target_write_results:
            result = self.target_write_results.pop(0)
            if isinstance(result, Exception):
                raise result
            if result is None:
                return None
        entry = _target(tokens, now=now)
        self.targets[tuple(tokens)] = entry
        return entry

    def write_attachment(self, target, envelope, *, now):
        frontier = envelope.frontier
        self.call_order.append(("attachment", frontier))
        if self.attachment_write_results:
            result = self.attachment_write_results.pop(0)
            if isinstance(result, Exception):
                raise result
            if result is None:
                return None
        entry = DiskKVAttachmentEntry.from_identity(
            envelope.identity,
            payload_path=(
                "attachments/payloads/aa/"
                f"{envelope.identity.attachment_id}.safetensors"
            ),
            payload_bytes=1,
            now=now,
        )
        self.attachment_ids.add(entry.attachment_id)
        return entry


def _served():
    return ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/synthetic/dspark"),
        drafter=_FakeDrafter(),
        tap=None,
        artifact_id="artifact-a",
    )


def _rail(served):
    return spec_cache_producer_rail(served, schedule="fixed:3")


def _target(tokens, *, now=0):
    return DiskKVEntry.from_tokens(
        _SCOPE,
        list(tokens),
        payload_path=f"payloads/target-{len(tokens)}.safetensors",
        payload_bytes=10,
        cache_class_names=("_Cache",),
        reason="aligned_frontier",
        now=now,
    )


def _lookup_hit(target, served):
    identity = build_spec_attachment_identity(target, served, _rail(served))
    capsule = served.drafter.export_state(_State(target.token_count))
    envelope = DiskKVAttachmentEnvelope.from_capsule(identity, capsule)
    entry = DiskKVAttachmentEntry.from_identity(
        identity,
        payload_path=f"attachments/payloads/aa/{identity.attachment_id}.safetensors",
        payload_bytes=1,
    )
    return DiskKVAttachmentLookup(
        status=ATTACHMENT_STATUS_HIT, entry=entry, envelope=envelope)


def _disk_hit(target):
    cache = [_Cache(target.token_count)]
    return DiskKVHit(
        entry=target,
        prompt_cache=cache,
        suffix_tokens=[99],
        cached_tokens=target.token_count,
    )


def _writer(
    store,
    served,
    *,
    full_count=800,
    restored_prefix=0,
    default_step=512,
):
    return make_spec_disk_kv_writer(
        store,
        scope=_SCOPE,
        full_tokens=list(range(full_count)),
        restored_prefix=restored_prefix,
        served=served,
        producer_rail=_rail(served),
        default_step=default_step,
        now_fn=lambda: 7,
    )


def _event(writer, served, frontier, *, target_offset=None, state_frontier=None):
    return SpecPrefillProgress(
        processed=frontier - writer.restored_prefix,
        total=len(writer.full_tokens) - writer.restored_prefix,
        frontier=frontier,
        target_cache=[_Cache(frontier if target_offset is None else target_offset)],
        drafter_state=_State(
            frontier if state_frontier is None else state_frontier),
        state_owner=served.drafter,
    )


def test_identity_binds_fixed_schedule_rail_and_public_capsule_contract():
    served = _served()
    target = _target(range(256))
    rail = _rail(served)
    identity = build_spec_attachment_identity(target, served, rail)

    assert identity.target_cache_id == target.cache_id
    assert identity.target_scope_hash == target.scope_hash
    assert identity.target_token_count == 256
    assert identity.target_token_prefix_hash == target.token_prefix_hash
    assert identity.attachment_kind == SPEC_DISK_ATTACHMENT_KIND
    assert identity.envelope_schema == SPEC_CACHE_SCHEMA
    assert identity.drafter_family == "dspark"
    assert identity.artifact_id == "artifact-a"
    assert identity.capsule_kind == served.drafter.state_capsule_kind
    assert identity.capsule_schema_major == 1
    assert identity.capsule_schema_minor == 0
    assert identity.producer_rail == rail
    assert rail == (
        "spec",
        SPEC_CACHE_SCHEMA,
        SPEC_PRODUCER_LATTICE,
        "dspark",
        "artifact-a",
        "fixed:3",
    )


def test_identity_refuses_a_different_artifact_rail():
    served = _served()
    target = _target(range(256))
    wrong = (*_rail(served)[:4], "artifact-b", _rail(served)[-1])
    with pytest.raises(ValueError, match="producer rail"):
        build_spec_attachment_identity(target, served, wrong)


def test_valid_attachment_restore_imports_before_returning_continuation():
    served = _served()
    target = _target(range(256))
    store = _FakeStore()
    store.restore_result = _lookup_hit(target, served)
    served.drafter.export_calls = 0

    result = restore_spec_attachment(store, _disk_hit(target), served, _rail(served))

    assert result.status == ATTACHMENT_STATUS_HIT
    assert result.reason is None
    assert result.attachment_entry == store.restore_result.entry
    assert result.continuation.target_cache[0].offset == 256
    assert result.continuation.prefix_offset == 256
    assert result.continuation.companion.frontier == 256
    assert result.continuation.companion.schedule == "fixed:3"
    assert result.continuation.companion.producer_rail == _rail(served)
    assert served.drafter.import_calls == 1
    assert store.quarantined == []


@pytest.mark.parametrize(
    "status",
    [
        ATTACHMENT_STATUS_MISSING,
        ATTACHMENT_STATUS_INVALID,
        ATTACHMENT_STATUS_UNAVAILABLE,
    ],
)
def test_restore_propagates_non_hit_status_without_changing_target(status):
    served = _served()
    target = _target(range(256))
    hit = _disk_hit(target)
    store = _FakeStore()
    store.restore_result = DiskKVAttachmentLookup(status=status, reason="synthetic")

    result = restore_spec_attachment(store, hit, served, _rail(served))

    assert result.status == status
    assert result.reason == "synthetic"
    assert result.continuation is None
    assert hit.prompt_cache[0].offset == 256
    assert store.quarantined == []


def test_model_import_failure_quarantines_only_attachment_and_keeps_target():
    served = _served()
    target = _target(range(256))
    hit = _disk_hit(target)
    store = _FakeStore()
    store.restore_result = _lookup_hit(target, served)
    served.drafter.import_error = ValueError("bad DSpark geometry")

    result = restore_spec_attachment(store, hit, served, _rail(served))

    assert result.status == ATTACHMENT_STATUS_INVALID
    assert "import" in result.reason
    assert result.continuation is None
    assert store.quarantined == [
        (store.restore_result.entry, "model_import_invalid")]
    assert hit.entry == target
    assert hit.prompt_cache[0].offset == 256


def test_existing_target_with_missing_attachment_stays_capture_eligible():
    served = _served()
    store = _FakeStore()
    target = _target(range(256))
    store.targets[tuple(range(256))] = target
    writer = _writer(store, served, full_count=300)

    assert writer.capture_frontiers == (256,)
    writer.on_prefill_progress(_event(writer, served, 256))

    assert store.call_order == [("attachment", 256)]
    assert served.drafter.export_calls == 1
    assert writer.target_writes == []
    assert len(writer.attachment_writes) == 1


def test_existing_complete_pair_is_deduped_before_capsule_export():
    served = _served()
    store = _FakeStore()
    target = _target(range(256))
    store.targets[tuple(range(256))] = target
    identity = build_spec_attachment_identity(target, served, _rail(served))
    store.attachment_ids.add(identity.attachment_id)
    writer = _writer(store, served, full_count=300)

    assert writer.capture_frontiers == ()
    writer.on_prefill_progress(_event(writer, served, 256))
    assert store.call_order == []
    assert served.drafter.export_calls == 0


def test_unavailable_attachment_tier_suppresses_existing_target_work():
    served = _served()
    store = _FakeStore()
    store.attachments_available = False
    store.targets[tuple(range(256))] = _target(range(256))
    writer = _writer(store, served, full_count=300)

    assert writer.capture_frontiers == ()
    assert writer.prefill_plan == ()


def test_unavailable_attachment_tier_still_writes_a_missing_target():
    served = _served()
    store = _FakeStore()
    store.attachments_available = False
    writer = _writer(store, served, full_count=300)

    writer.on_prefill_progress(_event(writer, served, 256))

    assert store.call_order == [("target", 256)]
    assert len(writer.target_writes) == 1
    assert writer.attachment_writes == []
    assert served.drafter.export_calls == 0


def test_cold_pair_commits_target_before_attachment():
    served = _served()
    store = _FakeStore()
    writer = _writer(store, served, full_count=300)

    writer.on_prefill_progress(_event(writer, served, 256))

    assert store.call_order == [("target", 256), ("attachment", 256)]
    assert len(writer.target_writes) == 1
    assert len(writer.attachment_writes) == 1
    assert len(writer.target_blocking_seconds) == 1
    assert len(writer.attachment_blocking_seconds) == 1


def test_attachment_timing_includes_capsule_capture_after_target_commit():
    served = _served()
    store = _FakeStore()
    ticks = iter((1.0, 2.0, 3.0, 7.0))
    writer = make_spec_disk_kv_writer(
        store,
        scope=_SCOPE,
        full_tokens=list(range(300)),
        restored_prefix=0,
        served=served,
        producer_rail=_rail(served),
        default_step=512,
        now_fn=lambda: 7,
        clock_fn=lambda: next(ticks),
    )

    writer.on_prefill_progress(_event(writer, served, 256))

    assert writer.target_blocking_seconds == [1.0]
    assert writer.attachment_blocking_seconds == [4.0]
    assert writer.attachment_attempt_seconds == [4.0]


@pytest.mark.parametrize("target_result", [None, OSError("target disk full")])
def test_target_failure_or_budget_skip_blocks_attachment(target_result):
    served = _served()
    store = _FakeStore()
    store.target_write_results = [target_result]
    writer = _writer(store, served, full_count=300)

    writer.on_prefill_progress(_event(writer, served, 256))

    assert store.call_order == [("target", 256)]
    assert writer.attachment_writes == []
    assert writer.target_attempts_disabled is True
    if isinstance(target_result, Exception):
        assert len(store.logged) == 1
        assert "paired target write failed" in store.logged[0]


@pytest.mark.parametrize(
    "attachment_result", [None, OSError("attachment disk full")])
def test_attachment_failure_does_not_block_later_target_frontier(
    attachment_result,
):
    served = _served()
    store = _FakeStore()
    store.attachment_write_results = [attachment_result]
    writer = _writer(store, served, full_count=600)

    writer.on_prefill_progress(_event(writer, served, 256))
    writer.on_prefill_progress(_event(writer, served, 512))

    assert store.call_order == [
        ("target", 256),
        ("attachment", 256),
        ("target", 512),
    ]
    assert [entry.token_count for entry in writer.target_writes] == [256, 512]
    assert writer.attachment_writes == []
    assert writer.attachment_attempts_disabled is True


@pytest.mark.parametrize(
    ("target_offset", "state_frontier"),
    [(512, 256), (256, 255), (256, 256.0)],
)
def test_target_or_drafter_frontier_mismatch_refuses_capture(
    target_offset, state_frontier,
):
    served = _served()
    store = _FakeStore()
    writer = _writer(store, served, full_count=300)

    writer.on_prefill_progress(_event(
        writer,
        served,
        256,
        target_offset=target_offset,
        state_frontier=state_frontier,
    ))

    assert store.call_order == []
    assert writer.refused_events == 1
    assert served.drafter.export_calls == 0


def test_event_outside_capture_frontiers_is_ignored_without_export():
    served = _served()
    store = _FakeStore()
    writer = _writer(store, served, full_count=600)

    writer.on_prefill_progress(_event(writer, served, 384))

    assert store.call_order == []
    assert writer.ignored_events == 1
    assert served.drafter.export_calls == 0


def test_unhashable_event_frontier_is_refused_without_escaping_callback():
    served = _served()
    store = _FakeStore()
    writer = _writer(store, served, full_count=300)
    event = SpecPrefillProgress(
        processed=256,
        total=300,
        frontier=[],
        target_cache=[_Cache(256)],
        drafter_state=_State(256),
        state_owner=served.drafter,
    )

    writer.on_prefill_progress(event)

    assert store.call_order == []
    assert writer.refused_events == 1
    assert served.drafter.export_calls == 0


def test_capture_frontiers_respect_write_depth_and_final_anchor():
    served = _served()
    limited = _FakeStore(write_depth=512)
    writer = _writer(limited, served, full_count=800)
    assert writer.capture_frontiers == (256, 512)

    final_anchor = _FakeStore(write_depth=512)
    writer = _writer(final_anchor, served, full_count=512)
    assert writer.capture_frontiers == (256,)


def test_restored_prefix_plan_lands_only_on_exact_eligible_frontiers():
    served = _served()
    store = _FakeStore(write_depth=512)
    writer = _writer(
        store,
        served,
        full_count=700,
        restored_prefix=100,
        default_step=300,
    )

    assert writer.capture_frontiers == (256, 512)
    assert writer.prefill_plan == (156, 256)


def test_factory_suppresses_capture_when_exact_lookup_raises():
    served = _served()
    store = _FakeStore()
    store.find_error = OSError("target index unreadable")

    writer = _writer(store, served, full_count=600)

    assert writer.capture_frontiers == ()
    assert writer.prefill_plan == ()


def test_callback_swallows_exact_lookup_failure_and_disables_target_path():
    served = _served()
    store = _FakeStore()
    writer = _writer(store, served, full_count=300)
    store.find_error = OSError("target index unreadable")

    writer.on_prefill_progress(_event(writer, served, 256))

    assert store.call_order == []
    assert writer.target_attempts_disabled is True
    assert writer.target_write_failures == 1


def test_callback_swallows_pair_lookup_failure_and_disables_only_attachment():
    served = _served()
    store = _FakeStore()
    store.targets[tuple(range(256))] = _target(range(256))
    writer = _writer(store, served, full_count=300)
    store.has_error = OSError("attachment index unreadable")

    writer.on_prefill_progress(_event(writer, served, 256))

    assert store.call_order == []
    assert writer.attachment_attempts_disabled is True
    assert writer.target_attempts_disabled is False
    assert writer.attachment_write_failures == 1


def test_restore_normalizes_raising_attachment_lookup_to_unavailable():
    served = _served()
    target = _target(range(256))
    store = _FakeStore()

    def fail_restore(_target_entry, _identity):
        raise OSError("attachment index unreadable")

    store.restore_attachment = fail_restore
    result = restore_spec_attachment(store, _disk_hit(target), served, _rail(served))

    assert result.status == ATTACHMENT_STATUS_UNAVAILABLE
    assert "unavailable" in result.reason
    assert result.continuation is None
