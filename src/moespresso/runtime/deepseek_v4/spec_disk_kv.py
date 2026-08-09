"""DSpark companion integration for aligned disk-KV target checkpoints.

The generic disk tier owns target and attachment persistence. This module owns
the DeepSeek-V4 speculative provenance, DSpark capsule reconstruction, and the
paired prefill callback that commits a target before its optional companion.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from moespresso.runtime.disk_kv import (
    ATTACHMENT_STATUS_HIT,
    ATTACHMENT_STATUS_INVALID,
    ATTACHMENT_STATUS_UNAVAILABLE,
    DiskKVAttachmentEntry,
    DiskKVAttachmentEnvelope,
    DiskKVAttachmentIdentity,
    DiskKVEntry,
    DiskKVHit,
    FrontierTracker,
    caches_all_at_offset,
    plan_prefill_chunks,
    scope_hash,
    token_prefix_hash,
)

from .spec_decode import DrafterStateCapsule, SpecPrefillProgress

if TYPE_CHECKING:
    from .spec_serve import ServedDrafter, SpecContinuation


SPEC_DISK_ATTACHMENT_KIND = "deepseek_v4_speculative_drafter_state"


def _spec_source_contract(
    served: ServedDrafter,
    producer_rail: tuple[str, ...],
) -> tuple[str, int, int, str]:
    """Validate one resumable DSpark source and its exact producer rail."""
    from .spec_serve import ServedDrafter, spec_cache_producer_rail

    if not isinstance(served, ServedDrafter):
        raise TypeError("DSpark disk integration requires a ServedDrafter")
    if served.family != "dspark":
        raise ValueError("DSpark disk integration requires the dspark family")
    if not isinstance(served.artifact_id, str) or not served.artifact_id:
        raise ValueError("DSpark disk integration requires a sidecar artifact id")
    rail = tuple(producer_rail)
    if (
        len(rail) != 6
        or any(not isinstance(part, str) or not part for part in rail)
    ):
        raise ValueError("invalid speculative producer rail")
    schedule = rail[-1]
    expected_rail = spec_cache_producer_rail(served, schedule=schedule)
    if expected_rail is None or rail != expected_rail:
        raise ValueError("speculative producer rail does not match served DSpark")

    drafter = served.drafter
    for name in ("state_frontier", "state_nbytes", "export_state", "import_state"):
        if not callable(getattr(drafter, name, None)):
            raise ValueError(f"served DSpark lacks resumable state method {name}")
    capsule_kind = getattr(drafter, "state_capsule_kind", None)
    if not isinstance(capsule_kind, str) or not capsule_kind:
        raise ValueError("served DSpark lacks a public capsule kind")
    capsule_major = getattr(drafter, "state_capsule_schema_major", None)
    capsule_minor = getattr(drafter, "state_capsule_schema_minor", None)
    for name, value in (
        ("major", capsule_major),
        ("minor", capsule_minor),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"served DSpark capsule schema {name} is invalid")
    return capsule_kind, capsule_major, capsule_minor, schedule


def build_spec_attachment_identity(
    target: DiskKVEntry,
    served: ServedDrafter,
    producer_rail: tuple[str, ...],
) -> DiskKVAttachmentIdentity:
    """Bind the current DSpark rail and capsule contract to ``target``."""
    from .spec_serve import SPEC_CACHE_SCHEMA

    if not isinstance(target, DiskKVEntry):
        raise TypeError("speculative disk attachment requires a target entry")
    capsule_kind, capsule_major, capsule_minor, _ = _spec_source_contract(
        served, producer_rail)
    return DiskKVAttachmentIdentity.from_target(
        target,
        attachment_kind=SPEC_DISK_ATTACHMENT_KIND,
        envelope_schema=SPEC_CACHE_SCHEMA,
        drafter_family=served.family,
        artifact_id=served.artifact_id,
        capsule_kind=capsule_kind,
        capsule_schema_major=capsule_major,
        capsule_schema_minor=capsule_minor,
        producer_rail=tuple(producer_rail),
    )


@dataclass(frozen=True)
class SpecAttachmentRestore:
    """Optional DSpark continuation recovered after a valid target disk hit."""

    status: str
    reason: str | None = None
    continuation: SpecContinuation | None = None
    attachment_entry: DiskKVAttachmentEntry | None = None


def restore_spec_attachment(
    store: Any,
    disk_hit: DiskKVHit,
    served: ServedDrafter,
    producer_rail: tuple[str, ...],
) -> SpecAttachmentRestore:
    """Restore and model-validate a DSpark attachment after its target hit."""
    from .spec_serve import SpecCacheCompanion, SpecContinuation

    if not isinstance(disk_hit, DiskKVHit):
        return SpecAttachmentRestore(
            status=ATTACHMENT_STATUS_INVALID,
            reason="invalid target disk hit",
        )
    if (
        disk_hit.cached_tokens != disk_hit.entry.token_count
        or disk_hit.cached_tokens < 0
    ):
        return SpecAttachmentRestore(
            status=ATTACHMENT_STATUS_INVALID,
            reason="target disk hit frontier mismatch",
        )
    try:
        identity = build_spec_attachment_identity(
            disk_hit.entry, served, producer_rail)
    except Exception as exc:  # noqa: BLE001 - typed outcome carries the refusal
        return SpecAttachmentRestore(
            status=ATTACHMENT_STATUS_INVALID,
            reason=f"attachment identity invalid: {exc}",
        )

    try:
        lookup = store.restore_attachment(disk_hit.entry, identity)
    except Exception as exc:  # noqa: BLE001 - target hit remains independently usable
        return SpecAttachmentRestore(
            status=ATTACHMENT_STATUS_UNAVAILABLE,
            reason=f"attachment lookup unavailable: {exc}",
        )
    if lookup.status != ATTACHMENT_STATUS_HIT:
        return SpecAttachmentRestore(
            status=lookup.status,
            reason=lookup.reason,
            attachment_entry=lookup.entry,
        )
    entry = lookup.entry
    envelope = lookup.envelope
    try:
        if entry is None or envelope is None:
            raise ValueError("attachment hit did not carry an entry and envelope")
        capsule = envelope.to_capsule(DrafterStateCapsule)
        if capsule.frontier != disk_hit.cached_tokens:
            raise ValueError("DSpark capsule frontier does not match target")
        restored_state = served.drafter.import_state(capsule)
        restored_frontier = served.drafter.state_frontier(restored_state)
        if isinstance(restored_frontier, bool) or not isinstance(
            restored_frontier, int
        ):
            raise ValueError("imported DSpark state frontier is not an integer")
        if restored_frontier != disk_hit.cached_tokens:
            raise ValueError("imported DSpark state frontier does not match target")
        _, _, _, schedule = _spec_source_contract(served, producer_rail)
        companion = SpecCacheCompanion(
            family=served.family,
            artifact_id=served.artifact_id,
            schedule=schedule,
            frontier=disk_hit.cached_tokens,
            capsule=capsule,
        )
        if companion.producer_rail != tuple(producer_rail):
            raise ValueError("restored companion producer rail mismatch")
        continuation = SpecContinuation(
            target_cache=disk_hit.prompt_cache,
            companion=companion,
            prefix_offset=disk_hit.cached_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - model validation is attachment-local
        if entry is not None:
            try:
                store.quarantine_attachment(entry, reason="model_import_invalid")
            except Exception:
                pass
        return SpecAttachmentRestore(
            status=ATTACHMENT_STATUS_INVALID,
            reason=f"DSpark capsule import invalid: {exc}",
            attachment_entry=entry,
        )
    return SpecAttachmentRestore(
        status=ATTACHMENT_STATUS_HIT,
        continuation=continuation,
        attachment_entry=entry,
    )


class SpecDiskKVWriter:
    """One-request paired target and DSpark frontier writer.

    ``capture_frontiers`` is the absolute allowlist for the speculative prefill
    loop. ``on_prefill_progress`` ignores every event outside that allowlist and
    revalidates both live frontiers before any snapshot or storage call.
    """

    def __init__(
        self,
        store: Any,
        *,
        tracker: FrontierTracker,
        capture_frontiers: tuple[int, ...],
        prefill_plan: tuple[int, ...],
        full_tokens: tuple[int, ...],
        served: ServedDrafter,
        producer_rail: tuple[str, ...],
        session_cache_key: str | None,
        now_fn: Callable[[], int],
        clock_fn: Callable[[], float],
    ):
        self.store = store
        self.tracker = tracker
        self.capture_frontiers = capture_frontiers
        self.prefill_plan = prefill_plan
        self.full_tokens = full_tokens
        self.restored_prefix = tracker.restored_prefix
        self.scope = tracker.scope
        self.served = served
        self.producer_rail = producer_rail
        self.session_cache_key = session_cache_key
        self._now_fn = now_fn
        self._clock_fn = clock_fn
        self._capture_set = frozenset(capture_frontiers)
        self._handled: set[int] = set()

        self.target_writes: list[DiskKVEntry] = []
        self.attachment_writes: list[DiskKVAttachmentEntry] = []
        self.target_blocking_seconds: list[float] = []
        self.attachment_blocking_seconds: list[float] = []
        self.target_attempt_seconds: list[float] = []
        self.attachment_attempt_seconds: list[float] = []
        self.target_write_failures = 0
        self.target_write_skips = 0
        self.attachment_write_failures = 0
        self.attachment_write_skips = 0
        self.capsule_export_failures = 0
        self.refused_events = 0
        self.ignored_events = 0
        self.pair_dedupes = 0
        self.target_attempts_disabled = False
        self.attachment_attempts_disabled = False

    def _log(self, message: str) -> None:
        log = getattr(self.store, "_log", None)
        if callable(log):
            log(message)

    def _clock_value(self) -> float | None:
        try:
            return float(self._clock_fn())
        except Exception:
            return None

    def _elapsed(self, started: float | None) -> float | None:
        finished = self._clock_value()
        if started is None or finished is None:
            return None
        return max(0.0, finished - started)

    def _event_frontiers_match(self, event: SpecPrefillProgress) -> bool:
        processed = event.processed
        total = event.total
        frontier = event.frontier
        if (
            isinstance(processed, bool)
            or not isinstance(processed, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or isinstance(frontier, bool)
            or not isinstance(frontier, int)
        ):
            return False
        if total != len(self.full_tokens) - self.restored_prefix:
            return False
        if processed < 0 or processed >= total:
            return False
        if frontier != self.restored_prefix + processed:
            return False
        try:
            if not caches_all_at_offset(event.target_cache, frontier):
                return False
        except Exception:
            return False
        if event.state_owner is not self.served.drafter:
            return False
        try:
            state_frontier = event.state_owner.state_frontier(event.drafter_state)
        except Exception:
            return False
        if isinstance(state_frontier, bool) or not isinstance(state_frontier, int):
            return False
        return state_frontier == frontier

    def _target_matches_prefix(
        self,
        target: Any,
        prefix_tokens: list[int],
    ) -> bool:
        try:
            return (
                isinstance(target, DiskKVEntry)
                and target.scope_hash == scope_hash(self.scope)
                and target.token_count == len(prefix_tokens)
                and target.token_prefix_hash == token_prefix_hash(prefix_tokens)
            )
        except Exception:
            return False

    def _export_capsule(self, event: SpecPrefillProgress) -> Any | None:
        try:
            capsule = event.state_owner.export_state(event.drafter_state)
            drafter = self.served.drafter
            if getattr(capsule, "frontier", None) != event.frontier:
                raise ValueError("exported capsule frontier mismatch")
            if getattr(capsule, "kind", None) != drafter.state_capsule_kind:
                raise ValueError("exported capsule kind mismatch")
            if (
                getattr(capsule, "schema_major", None)
                != drafter.state_capsule_schema_major
            ):
                raise ValueError("exported capsule schema major mismatch")
            if (
                getattr(capsule, "schema_minor", None)
                != drafter.state_capsule_schema_minor
            ):
                raise ValueError("exported capsule schema minor mismatch")
            return capsule
        except Exception as exc:  # noqa: BLE001 - target-only capture remains valid
            self.capsule_export_failures += 1
            self.attachment_attempts_disabled = True
            self._log(
                "[disk_kv] DSpark capsule capture failed; continuing with "
                f"target checkpoints only: {exc!r}"
            )
            return None

    def _write_target(
        self,
        event: SpecPrefillProgress,
        prefix_tokens: list[int],
    ) -> DiskKVEntry | None:
        if self.target_attempts_disabled or getattr(
            self.store, "writes_disabled", False
        ):
            return None
        try:
            caches = list(event.target_cache)
            state_trees = [cache.state for cache in caches]
            meta_state_trees = [cache.meta_state for cache in caches]
            class_names = tuple(type(cache).__name__ for cache in caches)
        except Exception as exc:  # noqa: BLE001 - invalid target snapshot is request-local
            self.target_write_failures += 1
            self.target_attempts_disabled = True
            self._log(
                "[disk_kv] paired target snapshot failed "
                f"token_count={len(prefix_tokens)}; disabling checkpoint "
                f"writes for this request: {exc!r}"
            )
            return None
        started = self._clock_value()
        try:
            entry = self.store.write_checkpoint(
                self.scope,
                prefix_tokens,
                cache_state_trees=state_trees,
                meta_state_trees=meta_state_trees,
                cache_class_names=class_names,
                reason="aligned_frontier",
                session_cache_key=self.session_cache_key,
                now=self._now_fn(),
            )
        except Exception as exc:  # noqa: BLE001 - persistence never surfaces in serving
            elapsed = self._elapsed(started)
            if elapsed is not None:
                self.target_attempt_seconds.append(elapsed)
            self.target_write_failures += 1
            self.target_attempts_disabled = True
            self._log(
                "[disk_kv] paired target write failed "
                f"token_count={len(prefix_tokens)}; disabling checkpoint "
                f"writes for this request: {exc!r}"
            )
            return None
        elapsed = self._elapsed(started)
        if elapsed is not None:
            self.target_attempt_seconds.append(elapsed)
        if entry is None:
            self.target_write_skips += 1
            self.target_attempts_disabled = True
            return None
        if not self._target_matches_prefix(entry, prefix_tokens):
            self.target_write_failures += 1
            self.target_attempts_disabled = True
            self._log(
                "[disk_kv] paired target writer returned a mismatched entry "
                f"token_count={len(prefix_tokens)}; disabling checkpoint "
                "writes for this request"
            )
            return None
        self.target_writes.append(entry)
        if elapsed is not None:
            self.target_blocking_seconds.append(elapsed)
        return entry

    def _write_attachment(
        self,
        target: DiskKVEntry,
        capsule: Any,
        *,
        started: float | None,
    ) -> None:
        try:
            identity = build_spec_attachment_identity(
                target, self.served, self.producer_rail)
            envelope = DiskKVAttachmentEnvelope.from_capsule(identity, capsule)
        except Exception as exc:  # noqa: BLE001 - target commit remains valid
            elapsed = self._elapsed(started)
            if elapsed is not None:
                self.attachment_attempt_seconds.append(elapsed)
            self.attachment_write_failures += 1
            self.attachment_attempts_disabled = True
            self._log(
                "[disk_kv] DSpark attachment envelope failed; keeping target "
                f"checkpoint: {exc!r}"
            )
            return
        try:
            entry = self.store.write_attachment(
                target, envelope, now=self._now_fn())
        except Exception as exc:  # noqa: BLE001 - target commit remains valid
            elapsed = self._elapsed(started)
            if elapsed is not None:
                self.attachment_attempt_seconds.append(elapsed)
            self.attachment_write_failures += 1
            self.attachment_attempts_disabled = True
            self._log(
                "[disk_kv] DSpark attachment write failed; keeping target "
                f"checkpoint: {exc!r}"
            )
            return
        elapsed = self._elapsed(started)
        if elapsed is not None:
            self.attachment_attempt_seconds.append(elapsed)
        if entry is None:
            self.attachment_write_skips += 1
            self.attachment_attempts_disabled = True
            return
        if (
            not isinstance(entry, DiskKVAttachmentEntry)
            or entry.attachment_id != identity.attachment_id
            or entry.identity != identity
        ):
            self.attachment_write_failures += 1
            self.attachment_attempts_disabled = True
            self._log(
                "[disk_kv] DSpark attachment writer returned a mismatched "
                "entry; keeping target checkpoint"
            )
            return
        self.attachment_writes.append(entry)
        if elapsed is not None:
            self.attachment_blocking_seconds.append(elapsed)

    def on_prefill_progress(self, event: SpecPrefillProgress) -> None:
        """Capture one allowlisted, exact target and raw DSpark frontier."""
        if not isinstance(event, SpecPrefillProgress):
            self.refused_events += 1
            return
        frontier = event.frontier
        if isinstance(frontier, bool) or not isinstance(frontier, int):
            self.refused_events += 1
            return
        if frontier not in self._capture_set or frontier in self._handled:
            self.ignored_events += 1
            return
        self._handled.add(frontier)
        if not self._event_frontiers_match(event):
            self.refused_events += 1
            return

        if self.target_attempts_disabled:
            return

        prefix_tokens = list(self.full_tokens[:frontier])
        try:
            target = self.store.find_exact(self.scope, prefix_tokens)
        except Exception:  # noqa: BLE001 - disk lookup cannot fail generation
            self.target_write_failures += 1
            self.target_attempts_disabled = True
            return
        if target is not None and not self._target_matches_prefix(
            target, prefix_tokens
        ):
            self.target_write_failures += 1
            self.target_attempts_disabled = True
            return
        attachments_available = bool(getattr(
            self.store, "attachments_available", False))
        if target is not None:
            if not attachments_available or self.attachment_attempts_disabled:
                return
            try:
                identity = build_spec_attachment_identity(
                    target, self.served, self.producer_rail)
                pair_exists = self.store.has_attachment(target, identity)
            except Exception:  # noqa: BLE001 - only optional attachment use stops
                self.attachment_write_failures += 1
                self.attachment_attempts_disabled = True
                return
            if pair_exists:
                self.pair_dedupes += 1
                return
            if not bool(getattr(self.store, "attachments_available", False)):
                return
        elif self.target_attempts_disabled or getattr(
            self.store, "writes_disabled", False
        ):
            return

        want_attachment = (
            not self.attachment_attempts_disabled
            and bool(getattr(self.store, "attachments_available", False))
        )
        if target is None:
            target = self._write_target(event, prefix_tokens)
            if target is None:
                return
        if not want_attachment:
            return
        if not bool(getattr(self.store, "attachments_available", False)):
            self.attachment_attempts_disabled = True
            return
        # The reported attachment cost covers the blocking capsule copy and
        # evaluation as well as envelope construction and durable persistence.
        # Target persistence stays outside this interval and is reported through
        # the ordinary checkpoint timing.
        attachment_started = self._clock_value()
        capsule = self._export_capsule(event)
        if capsule is None:
            elapsed = self._elapsed(attachment_started)
            if elapsed is not None:
                self.attachment_attempt_seconds.append(elapsed)
            return
        self._write_attachment(target, capsule, started=attachment_started)


def make_spec_disk_kv_writer(
    store: Any,
    *,
    scope: dict,
    full_tokens: Sequence[int],
    restored_prefix: int,
    served: ServedDrafter,
    producer_rail: tuple[str, ...],
    default_step: int,
    session_cache_key: str | None = None,
    now_fn: Callable[[], int] | None = None,
    clock_fn: Callable[[], float] | None = None,
) -> SpecDiskKVWriter:
    """Plan exact paired prefill captures for one speculative request."""
    _spec_source_contract(served, producer_rail)
    tokens = tuple(int(token) for token in full_tokens)
    if not tokens:
        raise ValueError("speculative disk writer requires a non-empty prompt")
    if (
        isinstance(restored_prefix, bool)
        or not isinstance(restored_prefix, int)
        or restored_prefix < 0
        or restored_prefix >= len(tokens)
    ):
        raise ValueError("restored prefix must precede the final prompt token")
    if isinstance(default_step, bool) or not isinstance(default_step, int):
        raise ValueError("speculative prefill step must be a positive integer")
    if default_step < 1:
        raise ValueError("speculative prefill step must be a positive integer")
    stride = getattr(store, "stride", None)
    if stride is None:
        raise ValueError("speculative disk writer requires a configured stride")

    def pair_complete(candidate_scope: dict, prefix_tokens: list[int]) -> bool:
        try:
            target = store.find_exact(candidate_scope, prefix_tokens)
        except Exception:
            return True
        if target is None:
            return bool(getattr(store, "writes_disabled", False))
        if not bool(getattr(store, "attachments_available", False)):
            return True
        try:
            identity = build_spec_attachment_identity(
                target, served, producer_rail)
            if store.has_attachment(target, identity):
                return True
        except Exception:
            return True
        return not bool(getattr(store, "attachments_available", False))

    tracker = FrontierTracker(
        stride=stride,
        restored_prefix=restored_prefix,
        full_tokens=list(tokens),
        scope=scope,
        already_written=pair_complete,
        write_depth=getattr(store, "write_depth_tokens", None),
    )
    capture_frontiers = []
    position = restored_prefix
    final_prefill_frontier = len(tokens) - 1
    while True:
        frontier = tracker.next_frontier_above(position)
        if frontier is None or frontier > final_prefill_frontier:
            break
        capture_frontiers.append(frontier)
        position = frontier
    prefill_plan = plan_prefill_chunks(
        start=restored_prefix,
        boundaries=capture_frontiers,
        step=default_step,
    )
    return SpecDiskKVWriter(
        store,
        tracker=tracker,
        capture_frontiers=tuple(capture_frontiers),
        prefill_plan=tuple(prefill_plan),
        full_tokens=tokens,
        served=served,
        producer_rail=tuple(producer_rail),
        session_cache_key=session_cache_key,
        now_fn=now_fn or (lambda: int(time.time())),
        clock_fn=clock_fn or time.perf_counter,
    )
