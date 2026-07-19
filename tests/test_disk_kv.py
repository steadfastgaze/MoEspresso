"""Disk prompt-cache read path.

The pure parts (root lock, token-prefix hash, scope and entry round-trip, the
JSON index longest-prefix selection, stride and config validation) run without
MLX. The payload schema round-trip and the three fail-closed cases use MLX arrays
through injected save/load callables so the codec is exercised without a model.
"""

from __future__ import annotations

import json

import pytest

from moespresso.runtime.disk_kv import (
    DiskCheckpointStore,
    DiskKVConfig,
    DiskKVEntry,
    DiskKVError,
    DiskKVIndex,
    DiskKVMetadataMismatch,
    DiskKVRootLock,
    DiskKVRootLocked,
    DiskKVStrideError,
    FrontierTracker,
    FrontierWriter,
    build_cache_scope,
    build_safety_metadata,
    caches_all_at_offset,
    default_cache_registry,
    open_disk_store,
    resolve_disk_kv_config,
    scope_hash,
    token_prefix_hash,
    validate_cache_classes,
    validate_payload_metadata,
    validate_stride,
)


def _model_key(rendering_id="render-a", *, live="raw", group=64, start=0, kind="mlx_prompt_cache"):
    return ("pkg-a", rendering_id, live, group, start, kind)


def _scope(rendering_id="render-a", *, classes=("KVCache", "KVCache"), **kwargs):
    return build_cache_scope(_model_key(rendering_id, **kwargs), classes)


# --- token-prefix hash -------------------------------------------------------


def test_token_prefix_hash_is_domain_separated_and_order_sensitive():
    h1 = token_prefix_hash([1, 2, 3])
    assert h1 == token_prefix_hash([1, 2, 3])
    assert h1 != token_prefix_hash([1, 2])
    assert h1 != token_prefix_hash([1, 3, 2])
    assert len(bytes.fromhex(h1)) == 32


def test_token_prefix_hash_refuses_negative_tokens():
    with pytest.raises(ValueError, match="non-negative"):
        token_prefix_hash([1, -2, 3])


def test_token_prefix_hash_known_vectors_are_stable():
    # Pinning known vectors guards the domain string, the count-first layout,
    # and the little-endian 64-bit encoding against silent drift.
    assert token_prefix_hash([]) == (
        "4e0184dc8ec9dbba45bb65bf07748c74b479f002ac80de8f2b3978a320eeb20e")
    assert token_prefix_hash([0]) == (
        "10f0a7b865ee2d056d6755de85468dd07a073e98273da0d93341446a31d7ed4c")
    assert token_prefix_hash([1, 2, 3]) == (
        "3545d5469d0ad0bc3dd16b15a9b2dc22ac47d7bdefdac6e3dbdc74c4c13df15b")


# --- scope round-trip --------------------------------------------------------


def test_scope_hash_includes_rendering_kv_policy_and_class_list():
    base = _scope()
    assert scope_hash(base) == scope_hash(_scope())
    assert scope_hash(base) != scope_hash(_scope("render-b"))
    assert scope_hash(base) != scope_hash(_scope(group=128))
    assert scope_hash(base) != scope_hash(_scope(start=5000))
    assert scope_hash(base) != scope_hash(_scope(live="mlx_affine_q8"))
    assert scope_hash(base) != scope_hash(_scope(classes=("KVCache",)))
    assert scope_hash(base) != scope_hash(_scope(kind="deepseek_v4_composite"))


def test_scope_carries_the_serve_six_tuple_and_class_list():
    scope = _scope("render-x", classes=("DeepseekV4Cache", "KVCache"))
    assert scope["package_manifest_artifact_id"] == "pkg-a"
    assert scope["rendering_identity"] == "render-x"
    assert scope["cache_class_names"] == ["DeepseekV4Cache", "KVCache"]
    assert scope["schema_version"]


def test_scope_keys_the_banded_prefill_offset_rail(monkeypatch):
    # The engaged route serves different attention values on chunks past
    # the first, so its checkpoints key a separate bucket. The route is
    # default-on, so the default scope carries the field; the kill switch
    # keeps the recorded disabled-route scope bytes (no field, same hash),
    # so checkpoints written under the kill switch restore there and a
    # mixed-rail restore is impossible in either direction.
    monkeypatch.setenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", "0")
    base = _scope()
    assert "dsv4_banded_prefill_offset" not in base

    monkeypatch.delenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", raising=False)
    engaged = _scope()
    assert engaged["dsv4_banded_prefill_offset"] is True
    assert scope_hash(engaged) != scope_hash(base)

    monkeypatch.setenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", "1")
    assert scope_hash(_scope()) == scope_hash(engaged)


# --- entry round-trip --------------------------------------------------------


def test_entry_from_tokens_round_trips_json_and_keeps_operational_fields_out_of_identity():
    scope = _scope()
    entry = DiskKVEntry.from_tokens(
        scope,
        [10, 20, 30],
        payload_path="payloads/aa/x.safetensors",
        payload_bytes=123,
        cache_class_names=("KVCache", "KVCache"),
        now=7,
    )
    restored = DiskKVEntry.from_json_obj(entry.to_json_obj())
    assert restored == entry
    assert restored.token_count == 3
    assert restored.token_prefix_hash == token_prefix_hash([10, 20, 30])
    assert restored.cache_class_names == ("KVCache", "KVCache")
    assert restored.created_at == 7 and restored.hit_count == 0


# --- root lock ---------------------------------------------------------------


def test_root_lock_acquires_releases_and_reacquires(tmp_path):
    lock = DiskKVRootLock(tmp_path).acquire()
    assert (tmp_path / "moespresso-disk-kv.lock").exists()
    lock.close()

    second = DiskKVRootLock(tmp_path).acquire()
    second.close()


def test_root_lock_contends_in_process_without_blocking(tmp_path):
    first = DiskKVRootLock(tmp_path).acquire()
    try:
        with pytest.raises(DiskKVRootLocked, match="already locked"):
            DiskKVRootLock(tmp_path).acquire()
    finally:
        first.close()


def test_root_lock_close_is_idempotent(tmp_path):
    lock = DiskKVRootLock(tmp_path).acquire()
    lock.close()
    lock.close()


def test_store_close_releases_root_lock(tmp_path):
    lock = DiskKVRootLock(tmp_path).acquire()
    store = DiskCheckpointStore(tmp_path, root_lock=lock)
    store.close()
    store.close()
    # The root is free again after close.
    second = DiskKVRootLock(tmp_path).acquire()
    second.close()


# --- JSON index --------------------------------------------------------------


def test_index_finds_longest_matching_token_prefix(tmp_path):
    scope = _scope()
    index = DiskKVIndex(tmp_path)
    index.put(DiskKVEntry.from_tokens(
        scope, [10, 20, 30, 40], payload_path="payloads/four.safetensors",
        payload_bytes=400, cache_class_names=("KVCache",)))
    index.put(DiskKVEntry.from_tokens(
        scope, [10, 20], payload_path="payloads/two.safetensors",
        payload_bytes=200, cache_class_names=("KVCache",)))

    found = index.find_longest(scope, [10, 20, 30, 40, 50, 60])
    assert found is not None
    assert found.token_count == 4
    assert found.payload_path == "payloads/four.safetensors"


def test_index_misses_divergent_tokens_and_wrong_scope(tmp_path):
    scope = _scope()
    index = DiskKVIndex(tmp_path)
    index.put(DiskKVEntry.from_tokens(
        scope, [10, 20, 30], payload_path="payloads/three.safetensors",
        payload_bytes=300, cache_class_names=("KVCache",)))

    assert index.find_longest(scope, [10, 99, 30, 40]) is None
    assert index.find_longest(_scope("render-b"), [10, 20, 30, 40]) is None


def test_index_ignores_checkpoints_longer_than_request(tmp_path):
    scope = _scope()
    index = DiskKVIndex(tmp_path)
    index.put(DiskKVEntry.from_tokens(
        scope, [1, 2, 3, 4], payload_path="payloads/four.safetensors",
        payload_bytes=400, cache_class_names=("KVCache",)))

    assert index.find_longest(scope, [1, 2, 3]) is None


def test_index_put_is_idempotent_and_atomic_file(tmp_path):
    scope = _scope()
    index = DiskKVIndex(tmp_path)
    entry = DiskKVEntry.from_tokens(
        scope, [1, 2, 3], payload_path="p.safetensors",
        payload_bytes=1, cache_class_names=("KVCache",))
    index.put(entry)
    index.put(entry)
    assert len(index.entries()) == 1
    # A fresh index over the same root reads the persisted file.
    assert len(DiskKVIndex(tmp_path).entries()) == 1
    assert not (tmp_path / "index.json.tmp").exists()


def test_index_mark_used_increments_hit_count_without_changing_identity(tmp_path):
    scope = _scope()
    index = DiskKVIndex(tmp_path)
    entry = DiskKVEntry.from_tokens(
        scope, [1, 2, 3], payload_path="p.safetensors",
        payload_bytes=1, cache_class_names=("KVCache",))
    index.put(entry)
    updated = index.mark_used(entry, now=99)
    assert updated.hit_count == 1
    assert updated.last_used_at == 99
    assert len(index.entries()) == 1
    assert index.find_longest(scope, [1, 2, 3, 4]).hit_count == 1


def test_index_refuses_unknown_schema(tmp_path):
    (tmp_path / "index.json").write_text(
        json.dumps({"schema_version": "other", "entries": []}))
    with pytest.raises(DiskKVError, match="schema"):
        DiskKVIndex(tmp_path)


# --- stride and config -------------------------------------------------------


def test_validate_stride_requires_positive_multiple_of_256():
    assert validate_stride(256) == 256
    assert validate_stride(4096) == 4096
    for bad in (0, -256, 255, 300, 257):
        with pytest.raises(DiskKVStrideError):
            validate_stride(bad)


def test_resolve_config_off_by_default():
    assert resolve_disk_kv_config({}) == DiskKVConfig(enabled=False)


def test_resolve_config_frontier_requires_root_and_valid_stride(tmp_path):
    cfg = resolve_disk_kv_config({
        "MOESPRESSO_DISK_KV": "frontier",
        "MOESPRESSO_DISK_KV_ROOT": str(tmp_path),
        "MOESPRESSO_DISK_KV_STRIDE": "4096",
    })
    assert cfg.enabled and cfg.root == tmp_path and cfg.stride == 4096


def test_resolve_config_rejects_unknown_mode():
    with pytest.raises(DiskKVError, match="frontier"):
        resolve_disk_kv_config({"MOESPRESSO_DISK_KV": "on"})


def test_resolve_config_requires_root_when_enabled():
    with pytest.raises(DiskKVError, match="ROOT"):
        resolve_disk_kv_config({
            "MOESPRESSO_DISK_KV": "frontier",
            "MOESPRESSO_DISK_KV_STRIDE": "256",
        })


def test_resolve_config_requires_stride_when_enabled(tmp_path):
    with pytest.raises(DiskKVError, match="STRIDE"):
        resolve_disk_kv_config({
            "MOESPRESSO_DISK_KV": "frontier",
            "MOESPRESSO_DISK_KV_ROOT": str(tmp_path),
        })


def test_resolve_config_rejects_unaligned_stride(tmp_path):
    with pytest.raises(DiskKVStrideError):
        resolve_disk_kv_config({
            "MOESPRESSO_DISK_KV": "frontier",
            "MOESPRESSO_DISK_KV_ROOT": str(tmp_path),
            "MOESPRESSO_DISK_KV_STRIDE": "1000",
        })


def test_resolve_config_serving_defaults_on(tmp_path):
    from moespresso.runtime.disk_kv import (
        DEFAULT_DISK_KV_BUDGET_BYTES,
        DEFAULT_DISK_KV_STRIDE,
    )

    package = tmp_path / "pkg"
    cfg = resolve_disk_kv_config(
        {"XDG_CACHE_HOME": str(tmp_path / "cache")}, package_dir=package)
    assert cfg.enabled
    assert cfg.explicit is False
    assert cfg.stride == DEFAULT_DISK_KV_STRIDE
    assert cfg.budget_bytes == DEFAULT_DISK_KV_BUDGET_BYTES
    assert cfg.root is not None
    assert cfg.root.parent == tmp_path / "cache" / "moespresso" / "disk_kv"


def test_resolve_config_default_root_is_per_package_and_stable(tmp_path):
    env = {"XDG_CACHE_HOME": str(tmp_path / "cache")}
    package_a = tmp_path / "pkg-a"
    package_b = tmp_path / "pkg-b"
    root_a = resolve_disk_kv_config(env, package_dir=package_a).root
    root_b = resolve_disk_kv_config(env, package_dir=package_b).root
    assert root_a != root_b
    assert root_a == resolve_disk_kv_config(env, package_dir=package_a).root


def test_resolve_config_kill_switch(tmp_path):
    for mode in ("off", "0"):
        cfg = resolve_disk_kv_config(
            {"MOESPRESSO_DISK_KV": mode}, package_dir=tmp_path / "pkg")
        assert cfg == DiskKVConfig(enabled=False)
        assert resolve_disk_kv_config(
            {"MOESPRESSO_DISK_KV": mode}) == DiskKVConfig(enabled=False)


def test_resolve_config_explicit_frontier_with_package_uses_defaults(tmp_path):
    cfg = resolve_disk_kv_config(
        {"MOESPRESSO_DISK_KV": "frontier",
         "XDG_CACHE_HOME": str(tmp_path / "cache")},
        package_dir=tmp_path / "pkg")
    assert cfg.enabled
    assert cfg.explicit is True


def test_resolve_config_env_values_override_serving_defaults(tmp_path):
    cfg = resolve_disk_kv_config(
        {
            "MOESPRESSO_DISK_KV_ROOT": str(tmp_path / "explicit-root"),
            "MOESPRESSO_DISK_KV_STRIDE": "4096",
            "MOESPRESSO_DISK_KV_BYTES": "1048576",
        },
        package_dir=tmp_path / "pkg")
    assert cfg.root == tmp_path / "explicit-root"
    assert cfg.stride == 4096
    assert cfg.budget_bytes == 1048576


def test_resolve_config_unlimited_budget_literal(tmp_path):
    cfg = resolve_disk_kv_config(
        {"MOESPRESSO_DISK_KV_BYTES": "unlimited",
         "XDG_CACHE_HOME": str(tmp_path / "cache")},
        package_dir=tmp_path / "pkg")
    assert cfg.budget_bytes is None


def test_resolve_config_write_depth_defaults_and_overrides(tmp_path):
    from moespresso.runtime.disk_kv import DEFAULT_DISK_KV_WRITE_DEPTH

    env = {"XDG_CACHE_HOME": str(tmp_path / "cache")}
    assert resolve_disk_kv_config(
        env, package_dir=tmp_path / "pkg"
    ).write_depth_tokens == DEFAULT_DISK_KV_WRITE_DEPTH
    assert resolve_disk_kv_config(
        {**env, "MOESPRESSO_DISK_KV_WRITE_DEPTH": "4096"},
        package_dir=tmp_path / "pkg").write_depth_tokens == 4096
    assert resolve_disk_kv_config(
        {**env, "MOESPRESSO_DISK_KV_WRITE_DEPTH": "unlimited"},
        package_dir=tmp_path / "pkg").write_depth_tokens is None
    explicit = resolve_disk_kv_config({
        "MOESPRESSO_DISK_KV": "frontier",
        "MOESPRESSO_DISK_KV_ROOT": str(tmp_path),
        "MOESPRESSO_DISK_KV_STRIDE": "1024",
    })
    assert explicit.write_depth_tokens is None
    for bad in ("0", "-1", "many"):
        with pytest.raises(DiskKVError, match="WRITE_DEPTH"):
            resolve_disk_kv_config(
                {**env, "MOESPRESSO_DISK_KV_WRITE_DEPTH": bad},
                package_dir=tmp_path / "pkg")


def test_tracker_write_depth_caps_proposed_frontiers():
    from moespresso.runtime.disk_kv import FrontierTracker

    tracker = FrontierTracker(
        stride=1024,
        restored_prefix=0,
        full_tokens=list(range(5000)),
        scope={"k": "v"},
        write_depth=2048,
    )
    assert tracker.crossings_up_to(5000) == [1024, 2048]
    assert tracker.next_frontier_above(2048) is None


def test_open_disk_store_normalizes_raw_failures(tmp_path):
    from moespresso.runtime.disk_kv import open_disk_store

    corrupt = tmp_path / "corrupt-root"
    corrupt.mkdir()
    (corrupt / "index.json").write_text("not json", encoding="utf-8")
    with pytest.raises(DiskKVError, match="cannot open"):
        open_disk_store(DiskKVConfig(
            enabled=True, root=corrupt, stride=1024))

    unwritable_parent = tmp_path / "sealed"
    unwritable_parent.mkdir()
    unwritable_parent.chmod(0o500)
    try:
        with pytest.raises(DiskKVError, match="cannot open"):
            open_disk_store(DiskKVConfig(
                enabled=True, root=unwritable_parent / "root", stride=1024))
    finally:
        unwritable_parent.chmod(0o700)


def test_frontier_writer_disables_after_first_hard_failure():
    from moespresso.runtime.disk_kv import FrontierTracker, FrontierWriter

    class _FailingStore:
        def __init__(self):
            self.attempts = 0
            self.logged: list[str] = []

        def _log(self, line: str) -> None:
            self.logged.append(line)

        def write_checkpoint(self, *args, **kwargs):
            self.attempts += 1
            raise OSError("disk full")

    class _Cache:
        offset = 1024
        state = ["state"]
        meta_state = ["meta"]

    tracker = FrontierTracker(
        stride=1024, restored_prefix=0,
        full_tokens=list(range(4097)), scope={"k": "v"})
    store = _FailingStore()
    writer = FrontierWriter(store, tracker=tracker, caches=[_Cache()])
    writer.on_prompt_progress(1024, 4097)
    cache = _Cache()
    cache.offset = 2048
    writer.caches = [cache]
    writer.on_prompt_progress(2048, 4097)

    assert store.attempts == 1
    assert writer.disabled is True
    assert writer.write_failures == 1
    assert any("disabling checkpoint writes" in line for line in store.logged)


def test_open_disk_store_off_returns_none():
    assert open_disk_store(DiskKVConfig(enabled=False)) is None


def test_open_disk_store_holds_lock_and_refuses_second_owner(tmp_path):
    cfg = DiskKVConfig(enabled=True, root=tmp_path, stride=256)
    store = open_disk_store(cfg)
    try:
        assert store is not None
        assert store.stats()["lock_active"] is True
        with pytest.raises(DiskKVRootLocked):
            open_disk_store(cfg)
    finally:
        store.close()


# --- safety-key gates (no MLX) -----------------------------------------------


def test_validate_payload_metadata_refuses_scope_and_prefix_mismatch():
    scope = _scope()
    entry = DiskKVEntry.from_tokens(
        scope, [1, 2, 3], payload_path="p", payload_bytes=1,
        cache_class_names=("KVCache", "KVCache"))
    good = build_safety_metadata(entry)
    validate_payload_metadata(entry, good)

    wrong_scope = dict(good, scope_hash="nope")
    with pytest.raises(DiskKVMetadataMismatch, match="scope_hash"):
        validate_payload_metadata(entry, wrong_scope)

    wrong_len = dict(good, token_count="256")
    with pytest.raises(DiskKVMetadataMismatch, match="token_count"):
        validate_payload_metadata(entry, wrong_len)


def test_validate_cache_classes_fires_registry_and_list_gates():
    registry = default_cache_registry()
    validate_cache_classes(
        ("KVCache", "DeepseekV4Cache"),
        expected_classes=("KVCache", "DeepseekV4Cache"),
        registry=registry,
    )
    with pytest.raises(DiskKVMetadataMismatch, match="unregistered"):
        validate_cache_classes(
            ("KVCache", "MysteryCache"),
            expected_classes=("KVCache", "MysteryCache"),
            registry=registry,
        )
    with pytest.raises(DiskKVMetadataMismatch, match="mismatch"):
        validate_cache_classes(
            ("KVCache", "KVCache"),
            expected_classes=("KVCache", "DeepseekV4Cache"),
            registry=registry,
        )


# --- frontier tracker accounting ---------------------------------------------


class _FakeCache:
    """A cache stub that reports an offset (a positional layer) or None (recurrent)."""

    def __init__(self, offset):
        self.offset = offset
        self.state = ["state"]
        self.meta_state = "meta"


def _tokens(n):
    return [(i % 97) + 1 for i in range(n)]


def test_tracker_fresh_miss_crosses_every_frontier_in_the_prefill():
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(800), scope=_scope())
    # A single large prefill step can jump multiple strides; all are proposed once.
    assert tracker.crossings_up_to(700) == [256, 512]
    # Re-firing the same or a lower value proposes nothing new.
    assert tracker.crossings_up_to(700) == []
    assert tracker.crossings_up_to(256) == []


def test_tracker_skips_frontiers_at_or_below_the_restored_prefix():
    tracker = FrontierTracker(
        stride=256, restored_prefix=512, full_tokens=_tokens(1200), scope=_scope())
    # 256 and 512 are already on disk (restored); only 768 and 1024 are new.
    assert tracker.crossings_up_to(1100) == [768, 1024]


def test_tracker_dedupes_against_an_already_written_frontier():
    written = {512}

    def already(scope, tokens):
        return len(tokens) in written

    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(800), scope=_scope(),
        already_written=already)
    # 512 is skipped by the dedupe read; 256 is still eligible.
    assert tracker.crossings_up_to(700) == [256]


def test_tracker_next_frontier_above_respects_prefix_and_length():
    tracker = FrontierTracker(
        stride=256, restored_prefix=300, full_tokens=_tokens(900), scope=_scope())
    # First frontier strictly above 300 is 512.
    assert tracker.next_frontier_above(300) == 512
    # Nothing above the token count.
    assert tracker.next_frontier_above(768) is None


def test_tracker_frontier_tokens_are_the_exact_prefix():
    full = _tokens(600)
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=full, scope=_scope())
    assert tracker.frontier_tokens(256) == full[:256]


# --- offset gate: writes at a non-frontier are structurally impossible --------


def test_offset_gate_requires_every_positional_cache_at_the_frontier():
    # Two positional layers plus one recurrent (offset None): the recurrent is
    # exempt, the positional layers must both equal the frontier.
    assert caches_all_at_offset(
        [_FakeCache(512), _FakeCache(None), _FakeCache(512)], 512)
    # One positional layer lags: the gate refuses.
    assert not caches_all_at_offset(
        [_FakeCache(512), _FakeCache(256)], 512)


def test_offset_gate_refuses_when_no_cache_reports_an_offset():
    # A layout that reports no offset anywhere would rest on token accounting
    # alone, which the writer refuses.
    assert not caches_all_at_offset([_FakeCache(None), _FakeCache(None)], 512)


# --- frontier writer: capture gates and atomicity ----------------------------


class _RecordingStore:
    """A store stub that records write_checkpoint calls and can inject a failure."""

    def __init__(self, *, fail=False):
        self.stride = 256
        self.calls = []
        self.fail = fail

    def has_entry(self, scope, tokens):
        return False

    def write_checkpoint(self, scope, tokens, **kwargs):
        self.calls.append((len(tokens), kwargs["reason"]))
        if self.fail:
            raise DiskKVError("injected payload write failure")
        return DiskKVEntry.from_tokens(
            scope, tokens, payload_path="p", payload_bytes=1,
            cache_class_names=kwargs["cache_class_names"])


def test_writer_captures_only_when_caches_confirm_the_offset():
    store = _RecordingStore()
    caches = [_FakeCache(256), _FakeCache(None)]
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(600), scope=_scope())
    writer = FrontierWriter(store, tracker=tracker, caches=caches, now_fn=lambda: 0)

    # The caches sit at 256: the callback for processed=256 writes.
    writer.on_prompt_progress(256, 599)
    assert [c[0] for c in store.calls] == [256]
    assert store.calls[0][1] == "aligned_frontier"


def test_writer_refuses_a_proposed_frontier_when_offsets_disagree():
    store = _RecordingStore()
    # Token accounting proposes 256, but the positional cache is still at 200.
    caches = [_FakeCache(200)]
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(600), scope=_scope())
    writer = FrontierWriter(store, tracker=tracker, caches=caches, now_fn=lambda: 0)

    writer.on_prompt_progress(256, 599)
    assert store.calls == []
    assert writer.refused_offset_mismatch == 1


def test_writer_swallows_an_injected_write_failure_and_keeps_serving():
    store = _RecordingStore(fail=True)
    caches = [_FakeCache(256)]
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(600), scope=_scope())
    writer = FrontierWriter(store, tracker=tracker, caches=caches, now_fn=lambda: 0)

    # The write raises inside the writer; the writer counts it and returns, so a
    # generation call continues rather than surfacing the fault.
    writer.on_prompt_progress(256, 599)
    assert writer.write_failures == 1
    assert writer.written == []


def _boundaries_hit(start, plan):
    positions = []
    position = start
    for size in plan:
        position += size
        positions.append(position)
    return positions


def _assert_plan_shape(plan, *, start, boundaries, step):
    """The planner contract: full chunks except one lander per boundary.

    Every boundary is a chunk end exactly once, every chunk below the full
    step ends exactly on a boundary, and the chunk count is bounded by
    ``span // step + len(boundaries)``.
    """
    ends = _boundaries_hit(start, plan)
    for boundary in boundaries:
        assert ends.count(boundary) == 1
    for size, end in zip(plan, ends):
        assert size == step or end in boundaries
    assert all(1 <= size <= step for size in plan)
    span = boundaries[-1] - start if boundaries else 0
    assert len(plan) <= span // step + len(boundaries)


def test_plan_hits_the_first_frontier_and_every_later_one_exactly_once():
    store = _RecordingStore()
    tracker = FrontierTracker(
        stride=256, restored_prefix=100, full_tokens=_tokens(1000), scope=_scope())
    writer = FrontierWriter(store, tracker=tracker, caches=[_FakeCache(0)], now_fn=lambda: 0)
    # Frontiers 256, 512, 768 sit inside the 999-token prefill span. The gap
    # from the unaligned prefix is closed by one 156-token lander; every other
    # chunk is a full stride-wide chunk under the 2048 cap.
    plan = writer.prefill_chunk_plan(2048)
    assert plan == [156, 256, 256]
    _assert_plan_shape(plan, start=100, boundaries=[256, 512, 768], step=2048)


def test_plan_is_full_size_chunks_on_a_fresh_miss():
    store = _RecordingStore()
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(1000), scope=_scope())
    writer = FrontierWriter(store, tracker=tracker, caches=[_FakeCache(0)], now_fn=lambda: 0)
    # Fresh miss: every gap equals the stride, so no lander is ever short.
    assert writer.prefill_chunk_plan(2048) == [256, 256, 256]


def test_plan_respects_a_small_default_step_between_frontiers():
    store = _RecordingStore()
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(1000), scope=_scope())
    writer = FrontierWriter(store, tracker=tracker, caches=[_FakeCache(0)], now_fn=lambda: 0)
    # A step below the stride fills each gap with full chunks plus one lander.
    plan = writer.prefill_chunk_plan(100)
    assert plan == [100, 100, 56, 100, 100, 56, 100, 100, 56]
    _assert_plan_shape(plan, start=0, boundaries=[256, 512, 768], step=100)


def test_plan_is_empty_when_no_frontier_is_reachable_during_prefill():
    store = _RecordingStore()
    # Suffix smaller than one stride: nothing to align, the whole prefill runs
    # at the caller's uniform step.
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(200), scope=_scope())
    writer = FrontierWriter(store, tracker=tracker, caches=[_FakeCache(0)], now_fn=lambda: 0)
    assert writer.prefill_chunk_plan(2048) == []


def test_plan_excludes_a_frontier_at_the_full_prompt_length():
    store = _RecordingStore()
    # The prefill loop leaves the final prompt token to the first decode step,
    # so a frontier equal to the prompt length is unreachable during prefill
    # and must not force chunks (it is proposed after the first decode and the
    # offset gate refuses it; the accounting models it as a permanent hole).
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(256), scope=_scope())
    writer = FrontierWriter(store, tracker=tracker, caches=[_FakeCache(0)], now_fn=lambda: 0)
    assert writer.prefill_chunk_plan(2048) == []


def test_plan_skips_already_written_frontiers_without_a_lander():
    written = {512}

    def already(scope, tokens):
        return len(tokens) in written

    store = _RecordingStore()
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(1000), scope=_scope(),
        already_written=already)
    writer = FrontierWriter(store, tracker=tracker, caches=[_FakeCache(0)], now_fn=lambda: 0)
    # 512 is already on disk: no chunk needs to end there, so the 256..768 gap
    # runs as one full 512-token chunk under the cap.
    assert writer.prefill_chunk_plan(2048) == [256, 512]


def test_plan_prefill_chunks_refuses_bad_step_and_non_ascending_boundaries():
    from moespresso.runtime.disk_kv import plan_prefill_chunks

    with pytest.raises(DiskKVError, match="positive"):
        plan_prefill_chunks(start=0, boundaries=[256], step=0)
    with pytest.raises(DiskKVError, match="ascend"):
        plan_prefill_chunks(start=300, boundaries=[256], step=64)
    with pytest.raises(DiskKVError, match="ascend"):
        plan_prefill_chunks(start=0, boundaries=[256, 256], step=64)


# --- the collapse cases from the road-test now get full-size chunks -----------
#
# A cumulative session's in-memory hit restores a prefix whose length is the
# previous request's full-plus-completion count, so its parity is arbitrary. A
# single uniform step must divide gcd(first_gap, stride), which collapses to a
# few tokens on such a prefix: measured roughly 11x slower prefill, and past
# roughly 20k context a reproducible Metal command-buffer out-of-memory abort.
# The variable-step plan replaces the divisor constraint with one short lander
# per frontier, so these measured requests prefill at the full step.


def test_measured_slowdown_turn_plans_full_chunks_instead_of_step_16():
    # Hit at 1648, 7907-token suffix, stride 4096: the divisor step was
    # gcd(2448, 4096) = 16 (395 s to first token). The plan covers the same
    # span to the last frontier in 4 chunks instead of 494 sixteen-token steps.
    store = _RecordingStore()
    store.stride = 4096
    tracker = FrontierTracker(
        stride=4096, restored_prefix=1648, full_tokens=_tokens(1648 + 7907),
        scope=_scope())
    writer = FrontierWriter(
        store, tracker=tracker, caches=[_FakeCache(0)], now_fn=lambda: 0)
    plan = writer.prefill_chunk_plan(2048)
    assert plan == [2048, 400, 2048, 2048]
    _assert_plan_shape(plan, start=1648, boundaries=[4096, 8192], step=2048)


def test_measured_abort_turn_plans_full_chunks_instead_of_step_1():
    # Hit at 18211, 7546-token suffix, stride 4096: the divisor step was
    # gcd(2269, 4096) = 1, which aborted the serve process mid-prefill. The
    # plan needs 4 chunks instead of 7545 single-token steps.
    store = _RecordingStore()
    store.stride = 4096
    tracker = FrontierTracker(
        stride=4096, restored_prefix=18211, full_tokens=_tokens(18211 + 7546),
        scope=_scope())
    writer = FrontierWriter(
        store, tracker=tracker, caches=[_FakeCache(0)], now_fn=lambda: 0)
    plan = writer.prefill_chunk_plan(2048)
    assert plan == [2048, 221, 2048, 2048]
    _assert_plan_shape(plan, start=18211, boundaries=[20480, 24576], step=2048)


def test_frontier_aligned_restore_plans_only_full_chunks():
    # Restored exactly on a frontier (a disk restore): every gap is a stride
    # multiple, so the plan is uniform full-size chunks, the geometry the
    # aligned arm measured at the intended prefill rate.
    store = _RecordingStore()
    store.stride = 4096
    tracker = FrontierTracker(
        stride=4096, restored_prefix=12288, full_tokens=_tokens(12288 + 11073),
        scope=_scope())
    writer = FrontierWriter(
        store, tracker=tracker, caches=[_FakeCache(0)], now_fn=lambda: 0)
    plan = writer.prefill_chunk_plan(2048)
    assert plan == [2048] * 4
    assert _boundaries_hit(12288, plan) == [14336, 16384, 18432, 20480]


# --- writer atomicity through the real store ---------------------------------


def test_frontier_write_leaves_index_untouched_when_the_payload_write_fails(tmp_path):
    def failing_save(*args, **kwargs):
        raise OSError("disk full mid-payload")

    store = DiskCheckpointStore(tmp_path, save_payload_fn=failing_save, stride=256)
    caches = [_FakeCache(256)]
    scope = _scope()
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=_tokens(600), scope=scope)
    writer = FrontierWriter(store, tracker=tracker, caches=caches, now_fn=lambda: 0)

    writer.on_prompt_progress(256, 599)
    # The injected payload failure is swallowed; the index stays empty and no
    # temp payload survives, so serving continues cold.
    assert writer.write_failures == 1
    assert store.index.entries() == []
    assert not list(tmp_path.rglob("*.safetensors"))
    assert not list(tmp_path.rglob("*.tmp.safetensors"))


# --- M3: byte budget, eviction, session key, orphan cleanup, counters --------


def _file_save_fn(nbytes: int):
    """A save_payload_fn stand-in that writes a real file of ``nbytes`` bytes.

    The store measures the on-disk size after the write, so eviction and orphan
    cleanup exercise real files without MLX. The signature matches
    ``save_prompt_cache_payload``.
    """
    from pathlib import Path as _Path

    def save_fn(root, cache_id, *, cache_state_trees, meta_state_trees,
                safety_metadata):
        rel = f"payloads/{cache_id[:2]}/{cache_id}.safetensors"
        final = _Path(root) / rel
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"\0" * nbytes)
        return rel, final.stat().st_size

    return save_fn


def _write(store, scope, tokens, *, session_cache_key=None, now=0):
    return store.write_checkpoint(
        scope, tokens,
        cache_state_trees=[], meta_state_trees=[],
        cache_class_names=("KVCache",),
        reason="aligned_frontier",
        session_cache_key=session_cache_key,
        now=now,
    )


def test_config_refuses_zero_and_negative_budget_at_startup(tmp_path):
    base = {
        "MOESPRESSO_DISK_KV": "frontier",
        "MOESPRESSO_DISK_KV_ROOT": str(tmp_path),
        "MOESPRESSO_DISK_KV_STRIDE": "256",
    }
    with pytest.raises(DiskKVError, match="MOESPRESSO_DISK_KV=off"):
        resolve_disk_kv_config({**base, "MOESPRESSO_DISK_KV_BYTES": "0"})
    with pytest.raises(DiskKVError, match="positive"):
        resolve_disk_kv_config({**base, "MOESPRESSO_DISK_KV_BYTES": "-5"})
    with pytest.raises(DiskKVError, match="integer"):
        resolve_disk_kv_config({**base, "MOESPRESSO_DISK_KV_BYTES": "big"})


def test_config_absent_budget_is_unlimited_and_positive_passes(tmp_path):
    base = {
        "MOESPRESSO_DISK_KV": "frontier",
        "MOESPRESSO_DISK_KV_ROOT": str(tmp_path),
        "MOESPRESSO_DISK_KV_STRIDE": "256",
    }
    assert resolve_disk_kv_config(base).budget_bytes is None
    cfg = resolve_disk_kv_config({**base, "MOESPRESSO_DISK_KV_BYTES": "4096"})
    assert cfg.budget_bytes == 4096


def test_budget_evicts_least_recently_used_first(tmp_path):
    scope = _scope()
    store = DiskCheckpointStore(
        tmp_path, save_payload_fn=_file_save_fn(100), budget_bytes=250,
        log_fn=lambda line: None)
    # Three entries of 100 bytes; budget 250 holds two. Created oldest-first,
    # then A is touched so B is the least-recently-used when C arrives.
    a = _write(store, scope, [1, 2], now=1)
    b = _write(store, scope, [1, 2, 3], now=2)
    store.index.mark_used(a, now=5)  # A becomes most-recently-used
    c = _write(store, scope, [1, 2, 3, 4], now=3)

    counts = {e.token_count for e in store.index.entries()}
    # B (least-recently-used) is evicted; A (touched) and C (new) remain.
    assert counts == {2, 4}
    assert store.evictions == 1
    # The evicted payload file is gone; the survivors' files remain.
    assert not (tmp_path / b.payload_path).exists()
    assert (tmp_path / a.payload_path).exists()
    assert (tmp_path / c.payload_path).exists()


def test_oversized_payload_is_skipped_not_evicting_everything(tmp_path):
    scope = _scope()
    store = DiskCheckpointStore(
        tmp_path, save_payload_fn=_file_save_fn(100), budget_bytes=250,
        log_fn=lambda line: None)
    keep = _write(store, scope, [1, 2], now=1)
    # A payload larger than the whole budget must be skipped, and the existing
    # entry must survive: the store never evicts everything for one oversized write.
    big_store = DiskCheckpointStore(
        tmp_path, index=store.index, save_payload_fn=_file_save_fn(400),
        budget_bytes=250, log_fn=lambda line: None)
    result = big_store.write_checkpoint(
        scope, [9, 9, 9], cache_state_trees=[], meta_state_trees=[],
        cache_class_names=("KVCache",), reason="aligned_frontier")

    assert result is None
    assert [e.token_count for e in store.index.entries()] == [keep.token_count]
    assert (tmp_path / keep.payload_path).exists()
    # No oversized payload survives on disk.
    assert not any(p.stat().st_size == 400 for p in tmp_path.rglob("*.safetensors"))


def test_unlimited_store_never_evicts(tmp_path):
    scope = _scope()
    store = DiskCheckpointStore(
        tmp_path, save_payload_fn=_file_save_fn(100), log_fn=lambda line: None)
    for i in range(5):
        _write(store, scope, list(range(i + 2)), now=i)
    assert len(store.index.entries()) == 5
    assert store.evictions == 0


def test_session_cache_key_is_stored_and_never_authorizes_a_load(tmp_path):
    scope = _scope()
    store = DiskCheckpointStore(
        tmp_path, save_payload_fn=_file_save_fn(10), log_fn=lambda line: None)
    entry = _write(store, scope, [1, 2, 3], session_cache_key="sess-A")
    # The hint is stored on the entry and survives a reload from the JSON file.
    assert entry.session_cache_key == "sess-A"
    reloaded = DiskKVIndex(tmp_path).find_longest(scope, [1, 2, 3, 4])
    assert reloaded.session_cache_key == "sess-A"
    # The hint is not part of identity: it never enters the safety metadata.
    assert "session_cache_key" not in build_safety_metadata(entry)
    assert "moespresso_cache_key" not in build_safety_metadata(entry)


def test_session_key_match_does_not_load_a_wrong_token_prefix(tmp_path):
    scope = _scope()
    index = DiskKVIndex(tmp_path)
    # Two entries share a session key but describe different token prefixes.
    index.put(DiskKVEntry.from_tokens(
        scope, [1, 2, 3, 4], payload_path="p4", payload_bytes=1,
        cache_class_names=("KVCache",), session_cache_key="sess-A"))
    # A request whose tokens diverge from the stored prefix finds nothing, even
    # despite the matching session key because the token prefix is authoritative.
    assert index.find_longest(scope, [1, 2, 9, 9]) is None
    # The exact-prefix request still finds it.
    assert index.find_longest(scope, [1, 2, 3, 4]).token_count == 4


def test_orphan_payload_cleanup_removes_unreferenced_files(tmp_path):
    scope = _scope()
    store = DiskCheckpointStore(
        tmp_path, save_payload_fn=_file_save_fn(10), log_fn=lambda line: None)
    kept = _write(store, scope, [1, 2, 3])
    # Drop an orphan payload no index entry references (a crash between the payload
    # rename and the index append leaves exactly this).
    orphan = tmp_path / "payloads" / "zz" / "orphan.safetensors"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"\0" * 10)

    deleted = store.cleanup_orphan_payloads()
    assert any("orphan.safetensors" in d for d in deleted)
    assert not orphan.exists()
    # The referenced payload is untouched.
    assert (tmp_path / kept.payload_path).exists()


def test_stats_counters_reflect_writes_evictions_and_quarantines(tmp_path):
    scope = _scope()
    store = DiskCheckpointStore(
        tmp_path, save_payload_fn=_file_save_fn(100), budget_bytes=150,
        log_fn=lambda line: None)
    first = _write(store, scope, [1, 2], now=1)
    _write(store, scope, [1, 2, 3], now=2)  # evicts the first (budget 150)

    stats = store.stats()
    assert stats["writes"] == 2
    assert stats["evictions"] == 1
    assert stats["quarantines"] == 0
    assert stats["budget_bytes"] == 150
    assert stats["stride"] is None
    assert stats["payload_bytes"] == 100

    # Quarantine bumps its counter.
    store.quarantine(first, reason="corrupt")
    assert store.stats()["quarantines"] == 1


def test_operator_log_line_per_decision(tmp_path):
    scope = _scope()
    lines = []
    store = DiskCheckpointStore(
        tmp_path, save_payload_fn=_file_save_fn(100), budget_bytes=150,
        log_fn=lines.append)
    _write(store, scope, [1, 2], now=1)
    _write(store, scope, [1, 2, 3], now=2)
    joined = "\n".join(lines)
    assert "write" in joined
    assert "evict" in joined
