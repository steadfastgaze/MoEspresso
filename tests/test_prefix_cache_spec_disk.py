"""Failure-first routing tests for DSpark disk-KV companions.

These tests keep target restoration, optional drafter-state restoration, and
paired prefill planning at the PrefixCacheGenerator seam. The disk payload
codecs and DSpark model-state import are covered by their owning test modules;
no model workload is needed here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from moespresso.runtime.deepseek_v4.spec_decode import (
    DrafterStateCapsule,
    SpecPrefillProgress,
)
from moespresso.runtime.deepseek_v4.spec_disk_kv import SpecAttachmentRestore
from moespresso.runtime.deepseek_v4.spec_serve import (
    ServedDrafter,
    SpecCacheCompanion,
    SpecContinuation,
    SpecContinuationError,
)
from moespresso.runtime.disk_kv import (
    ATTACHMENT_STATUS_HIT,
    ATTACHMENT_STATUS_INVALID,
    ATTACHMENT_STATUS_MISSING,
    ATTACHMENT_STATUS_UNAVAILABLE,
    DiskKVAttachmentEntry,
    DiskKVEntry,
    DiskKVHit,
)
from moespresso.runtime.generation import GenerationResult
from moespresso.runtime.kv_policy import parse_kv_policy
from moespresso.runtime.prefix_cache import PrefixCacheGenerator, PromptCacheStore


class _Tokenizer:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.bos_token = None

    def encode(self, _text, **_kwargs):
        return list(self.tokens)


class _LayerCache:
    def __init__(self, offset: int):
        self.offset = offset
        self.state = ("state", offset)
        self.meta_state = ("meta", offset)
        self.nbytes = 8


class _Drafter:
    greedy_only = False
    state_capsule_kind = "deepseek_v4_dspark_window_state"
    state_capsule_schema_major = 1
    state_capsule_schema_minor = 0

    def state_frontier(self, state):
        return state.frontier

    def state_nbytes(self, state):
        return state.nbytes

    def export_state(self, state):
        return state

    def import_state(self, capsule):
        return capsule


def _model_and_served():
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=_Drafter(),
        tap="tap",
        artifact_id="art:side",
    )
    return SimpleNamespace(_moespresso_ds4_drafter=served), served


def _companion(frontier: int) -> SpecCacheCompanion:
    return SpecCacheCompanion(
        family="dspark",
        artifact_id="art:side",
        schedule="fixed:3",
        frontier=frontier,
        capsule=SimpleNamespace(frontier=frontier, nbytes=16),
    )


def _policy():
    return parse_kv_policy({"live_kv_format": "raw"})


class _RestoringDisk:
    stride = None

    def __init__(self, target, *, frontier: int, exact: bool = False, events=None):
        self.target = target
        self.frontier = frontier
        self.exact = exact
        self.events = events if events is not None else []

    def restore(self, scope, tokens, **_kwargs):
        self.events.append("target")
        entry = DiskKVEntry.from_tokens(
            scope,
            list(tokens[: self.frontier]),
            payload_path="payloads/target.safetensors",
            payload_bytes=8,
            cache_class_names=("_LayerCache",),
        )
        suffix = [] if self.exact else list(tokens[self.frontier :])
        return DiskKVHit(
            entry=entry,
            prompt_cache=self.target,
            suffix_tokens=suffix,
            cached_tokens=self.frontier,
        )

    def stats(self):
        return {"enabled": True}


class _PlanningDisk:
    stride = 256
    write_depth_tokens = None
    writes_disabled = False
    attachments_available = True

    def __init__(self):
        self.logged = []

    def restore(self, _scope, _tokens, **_kwargs):
        return None

    def find_exact(self, _scope, _tokens):
        return None

    def has_attachment(self, _target, _identity):
        return False

    def _log(self, line):
        self.logged.append(line)

    def stats(self):
        return {"enabled": True}


def _generator(tokens, generate_fn, *, disk_store):
    model, served = _model_and_served()
    generator = PrefixCacheGenerator(
        model,
        _Tokenizer(tokens),
        {"artifact_id": "pkg", "architecture": {"family": "deepseek_v4_flash"}},
        PromptCacheStore(max_size=4),
        make_prompt_cache_fn=lambda _model: [_LayerCache(0)],
        generate_fn=generate_fn,
        disk_store=disk_store,
    )
    return generator, served


def _run(generator):
    return generator(
        "prompt",
        kv_policy=_policy(),
        effective_rendering_id="render",
        temperature=0.0,
    )


def test_disk_target_restores_before_attachment_and_resumes_speculation(
    monkeypatch,
):
    from moespresso.runtime.deepseek_v4 import spec_disk_kv, spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    events = []
    target = [_LayerCache(3)]
    disk = _RestoringDisk(target, frontier=3, events=events)
    seen = {}

    def restore_attachment(_store, disk_hit, _served, _producer_rail):
        events.append("attachment")
        return SpecAttachmentRestore(
            status=ATTACHMENT_STATUS_HIT,
            continuation=SpecContinuation(
                target_cache=disk_hit.prompt_cache,
                companion=_companion(disk_hit.cached_tokens),
                prefix_offset=disk_hit.cached_tokens,
            ),
        )

    monkeypatch.setattr(spec_disk_kv, "restore_spec_attachment", restore_attachment)

    def generate(_model, _tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        kwargs["spec_continuation_ready_callback"]()
        kwargs["first_token_callback"]()
        return GenerationResult(text="ok", speculative={"drafter": "dspark"})

    generator, _ = _generator([1, 2, 3, 4], generate, disk_store=disk)
    times = iter([10.0, 11.0, 13.5, 14.0, 15.25, 16.0, 17.5, 20.0])
    generator.clock = times.__next__
    result = generator(
        "prompt",
        kv_policy=_policy(),
        effective_rendering_id="render",
        temperature=0.0,
        ready_callback=lambda: events.append("ready"),
    )

    assert events == ["ready", "target", "attachment"]
    assert seen["prompt"] == [4]
    assert seen["prompt_cache"] is None
    assert seen["cached_tokens"] == 3
    assert seen["spec_continuation"].target_cache is target
    assert result.cache_event == "disk_hit"
    assert result.disk_restore_seconds == 2.5
    assert result.disk_attachment_restore_seconds == 2.75
    assert result.ready_to_first_token_seconds == 10.0
    assert result.disk_attachment_event == ATTACHMENT_STATUS_HIT
    assert result.speculative["cache_resume"] == {"status": "hit", "frontier": 3}


def test_disk_companion_preflight_failure_omits_restore_duration(monkeypatch):
    from moespresso.runtime.deepseek_v4 import spec_disk_kv, spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    target = [_LayerCache(3)]
    disk = _RestoringDisk(target, frontier=3)
    calls = []

    monkeypatch.setattr(
        spec_disk_kv,
        "restore_spec_attachment",
        lambda _store, disk_hit, _served, _producer_rail: SpecAttachmentRestore(
            status=ATTACHMENT_STATUS_HIT,
            continuation=SpecContinuation(
                target_cache=disk_hit.prompt_cache,
                companion=_companion(disk_hit.cached_tokens),
                prefix_offset=disk_hit.cached_tokens,
            ),
        ),
    )

    def generate(_model, _tokenizer, prompt, **kwargs):
        calls.append((list(prompt), dict(kwargs)))
        if "spec_continuation" in kwargs:
            raise SpecContinuationError("synthetic preflight rejection")
        kwargs["first_token_callback"]()
        return GenerationResult(text="ok", prompt_cache=kwargs["prompt_cache"])

    generator, _ = _generator([1, 2, 3, 4], generate, disk_store=disk)
    times = iter([10.0, 11.0, 13.0, 14.0, 15.0, 16.0, 20.0])
    generator.clock = times.__next__
    result = _run(generator)

    assert len(calls) == 2
    assert "spec_continuation_ready_callback" in calls[0][1]
    assert "spec_continuation" not in calls[1][1]
    assert result.cache_event == "disk_hit"
    assert result.disk_attachment_event == ATTACHMENT_STATUS_INVALID
    assert result.disk_attachment_restore_seconds is None
    assert result.ready_to_first_token_seconds == 10.0


@pytest.mark.parametrize(
    "status",
    [
        ATTACHMENT_STATUS_MISSING,
        ATTACHMENT_STATUS_INVALID,
        ATTACHMENT_STATUS_UNAVAILABLE,
    ],
)
def test_attachment_failure_keeps_valid_disk_target_and_runs_plain(
    monkeypatch, status,
):
    from moespresso.runtime.deepseek_v4 import spec_disk_kv, spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    target = [_LayerCache(3)]
    disk = _RestoringDisk(target, frontier=3)
    seen = {}

    monkeypatch.setattr(
        spec_disk_kv,
        "restore_spec_attachment",
        lambda *_args, **_kwargs: SpecAttachmentRestore(
            status=status, reason="optional attachment unavailable"
        ),
    )

    def generate(_model, _tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        return GenerationResult(text="ok", prompt_cache=kwargs["prompt_cache"])

    generator, _ = _generator([1, 2, 3, 4], generate, disk_store=disk)
    result = _run(generator)

    assert seen["prompt"] == [4]
    assert seen["prompt_cache"] is target
    assert seen["cached_tokens"] == 3
    assert "spec_continuation" not in seen
    assert not any(key.startswith("spec_prefill_") for key in seen)
    assert result.cache_event == "disk_hit"
    assert result.disk_attachment_event == status
    assert result.disk_attachment_restore_seconds is None


def test_exact_disk_target_does_not_attempt_attachment_restore(monkeypatch):
    from moespresso.runtime.deepseek_v4 import spec_disk_kv, spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    target = [_LayerCache(4)]
    disk = _RestoringDisk(target, frontier=4, exact=True)
    seen = {}

    def unexpected_restore(*_args, **_kwargs):
        raise AssertionError("an exact target cannot supply continuation logits")

    monkeypatch.setattr(spec_disk_kv, "restore_spec_attachment", unexpected_restore)

    def generate(_model, _tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        return GenerationResult(text="ok", speculative={"drafter": "dspark"})

    generator, _ = _generator([1, 2, 3, 4], generate, disk_store=disk)
    times = iter([10.0, 11.0, 13.0])
    generator.clock = times.__next__
    result = _run(generator)

    assert disk.events == ["target"]
    assert seen["prompt"] == [1, 2, 3, 4]
    assert seen["prompt_cache"] is None
    assert seen["cached_tokens"] == 0
    assert "spec_continuation" not in seen
    assert result.cache_event == "exact_fallback"
    assert result.disk_restore_seconds == 2.0
    assert result.ready_to_first_token_seconds is None
    assert result.disk_attachment_event is None
    assert result.disk_attachment_restore_seconds is None


def test_cold_dspark_uses_only_paired_plan_at_exact_disk_frontier(monkeypatch):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    disk = _PlanningDisk()
    seen = {}

    def generate(_model, _tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        return GenerationResult(text="ok", speculative={"drafter": "dspark"})

    generator, _ = _generator(range(1, 301), generate, disk_store=disk)
    result = _run(generator)

    assert result.text == "ok"
    assert seen["prompt_cache"] is None
    assert seen["cached_tokens"] == 0
    assert seen["spec_prefill_plan"] == [256]
    assert seen["spec_prefill_progress_frontiers"] == [256]
    assert callable(seen["spec_prefill_progress_callback"])
    assert "prefill_plan" not in seen
    assert "prompt_progress_callback" not in seen


def test_paired_callback_reports_confirmed_target_and_attachment_writes(monkeypatch):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)

    class WritingDisk(_PlanningDisk):
        def __init__(self):
            super().__init__()
            self.order = []

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
            self.order.append(("target", len(tokens)))
            return DiskKVEntry.from_tokens(
                scope,
                list(tokens),
                payload_path="payloads/target.safetensors",
                payload_bytes=8,
                cache_class_names=cache_class_names,
                reason=reason,
                session_cache_key=session_cache_key,
                now=now,
            )

        def write_attachment(self, target, envelope, *, now):
            self.order.append(("attachment", envelope.frontier))
            return DiskKVAttachmentEntry.from_identity(
                envelope.identity,
                payload_path=(
                    "attachments/payloads/aa/"
                    f"{envelope.identity.attachment_id}.safetensors"
                ),
                payload_bytes=4,
                now=now,
            )

    disk = WritingDisk()

    def generate(model, _tokenizer, prompt, **kwargs):
        capsule = DrafterStateCapsule(
            kind="deepseek_v4_dspark_window_state",
            schema_major=1,
            schema_minor=0,
            frontier=256,
            metadata=(("stages", 3),),
            tensors=(),
        )
        kwargs["spec_prefill_progress_callback"](
            SpecPrefillProgress(
                processed=256,
                total=len(prompt),
                frontier=256,
                target_cache=[_LayerCache(256)],
                drafter_state=capsule,
                state_owner=model._moespresso_ds4_drafter.drafter,
            )
        )
        return GenerationResult(text="served", speculative={"drafter": "dspark"})

    generator, _ = _generator(range(1, 301), generate, disk_store=disk)
    result = _run(generator)

    assert disk.order == [("target", 256), ("attachment", 256)]
    assert result.disk_checkpoints_written == 1
    assert len(result.disk_checkpoint_write_seconds) == 1
    assert result.disk_attachments_written == 1
    assert len(result.disk_attachment_write_seconds) == 1


def test_paired_planning_failure_never_fails_generation(monkeypatch):
    from moespresso.runtime.deepseek_v4 import spec_disk_kv, spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    disk = _PlanningDisk()
    seen = {}

    monkeypatch.setattr(
        spec_disk_kv,
        "make_spec_disk_kv_writer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("planning index fault")
        ),
    )

    def generate(_model, _tokenizer, _prompt, **kwargs):
        seen.update(kwargs)
        return GenerationResult(text="served", speculative={"drafter": "dspark"})

    generator, _ = _generator(range(1, 301), generate, disk_store=disk)
    result = _run(generator)

    assert result.text == "served"
    assert not any(key.startswith("spec_prefill_") for key in seen)
    assert any("paired checkpoint planning failed" in line for line in disk.logged)


def test_paired_callback_disk_fault_never_fails_generation(monkeypatch):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)

    class CallbackFaultDisk(_PlanningDisk):
        def __init__(self):
            super().__init__()
            self.find_calls = 0
            self.write_calls = 0

        def find_exact(self, _scope, _tokens):
            self.find_calls += 1
            if self.find_calls == 1:
                return None
            raise RuntimeError("index fault during callback")

        def write_checkpoint(self, *_args, **_kwargs):
            self.write_calls += 1
            raise AssertionError("callback must stop after the lookup fault")

    disk = CallbackFaultDisk()

    def generate(_model, _tokenizer, prompt, **kwargs):
        callback = kwargs["spec_prefill_progress_callback"]
        callback(
            SpecPrefillProgress(
                processed=256,
                total=len(prompt),
                frontier=256,
                target_cache=[_LayerCache(256)],
                drafter_state=SimpleNamespace(frontier=256, nbytes=16),
                state_owner=_model._moespresso_ds4_drafter.drafter,
            )
        )
        return GenerationResult(text="served", speculative={"drafter": "dspark"})

    generator, _ = _generator(range(1, 301), generate, disk_store=disk)
    result = _run(generator)

    assert result.text == "served"
    assert disk.find_calls == 2
    assert disk.write_calls == 0
    assert result.disk_checkpoints_written is None
    assert result.disk_attachments_written is None
