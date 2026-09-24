from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

import moespresso.runtime.qwen4.disk_cache as qwen4_disk_cache
from moespresso.runtime.disk_kv import DiskCheckpointStore
from moespresso.runtime.kv_policy import KVPolicy, LIVE_KV_RAW
from moespresso.runtime.prefix_cache import PromptCacheStore
from moespresso.runtime.qwen4.generation import QWEN4_GENERATION_ADAPTER, Qwen4RequestGenerator
from test_qwen4_cache_snapshot import model
from test_qwen4_generation import _Tokenizer
from test_qwen4_kvarn_snapshot import same_arrays


def generator(store=None, memory=None):
    mx.random.seed(331)
    target = model()
    target.lm_head = nn.Linear(2, 12, bias=False)
    target.lm_head.set_dtype(mx.bfloat16)
    object.__setattr__(target, "_moespresso_generation_adapter", QWEN4_GENERATION_ADAPTER)
    object.__setattr__(target, "_moespresso_qwen4_stop_ids", frozenset({100}))
    object.__setattr__(target, "_moespresso_qwen4_prefill_step_size", 256)
    return Qwen4RequestGenerator(target, _Tokenizer(), 1024, disk_store=store, cache_store=memory)


def request(run, tokens, **kwargs):
    scores = []

    def observe(_step, response):
        row = mx.array(response.logprobs)
        mx.eval(row)
        scores.append(row)

    result = run(
        " ".join(map(str, tokens)), kv_policy=KVPolicy(live_kv_format=LIVE_KV_RAW),
        effective_rendering_id=kwargs.pop("effective_rendering_id", "render-1"),
        max_tokens=3, temperature=0, response_callback=observe, **kwargs,
    )
    return result, scores


def test_qwen_disk_restart_restores_prefix_and_continuation(tmp_path):
    tokens = [1, 2, 3, 1] * 65
    store = DiskCheckpointStore(tmp_path, stride=256)
    first = generator(store)
    try:
        result, _ = request(first, tokens)
        assert result.cached_tokens == 0 and result.disk_checkpoints_written == 1
        assert len(result.disk_checkpoint_write_seconds) == 1
        assert store.index.entries()[0].token_count == 256
    finally:
        first.close()
        store.close()
    store = DiskCheckpointStore(tmp_path, stride=256)
    restored = generator(store)
    cold = generator()
    try:
        resumed, resumed_scores = request(restored, tokens + [2, 3, 1])
        reference, reference_scores = request(cold, tokens + [2, 3, 1])
        assert resumed.cache_event == "disk_hit" and resumed.cached_tokens == 256
        assert resumed.prompt_tokens == len(tokens) + 3 - 256
        assert reference.prompt_tokens == len(tokens) + 3
        assert resumed.disk_restore_seconds is not None
        assert resumed.disk_checkpoints_written == 0
        assert resumed.generated_token_ids == reference.generated_token_ids
        same_arrays(resumed_scores, reference_scores)
        assert store.restores == 1
    finally:
        cold.close()
        restored.close()
        store.close()


def test_qwen_prompt_cache_schema_change_invalidates_legacy_state(
    tmp_path, monkeypatch
):
    current_schema = qwen4_disk_cache.QWEN4_PROMPT_CACHE_SCHEMA
    assert current_schema == "qwen4-prompt-kvarn4-logits-v2"
    monkeypatch.setattr(
        qwen4_disk_cache,
        "QWEN4_PROMPT_CACHE_SCHEMA",
        "qwen4-prompt-kvarn4-logits-v1",
    )
    store = DiskCheckpointStore(tmp_path, stride=256)
    writer = generator(store)
    tokens = [1, 2, 3, 1] * 65
    try:
        written, _ = request(writer, tokens)
        assert written.disk_checkpoints_written == 1
    finally:
        writer.close()

    monkeypatch.setattr(
        qwen4_disk_cache,
        "QWEN4_PROMPT_CACHE_SCHEMA",
        current_schema,
    )
    reader = generator(store)
    try:
        result, _ = request(reader, tokens)
        assert result.cache_event == "miss"
        assert result.cached_tokens == 0
        assert store.restores == 0
    finally:
        reader.close()
        store.close()


def test_qwen_exact_disk_hit_uses_saved_logits_without_prefill(tmp_path, monkeypatch):
    tokens = [1, 2, 3, 1] * 64
    store = DiskCheckpointStore(tmp_path, stride=256)
    writer = generator(store)
    expected, expected_scores = request(writer, tokens)
    writer.close()
    reader = generator(store)
    monkeypatch.setattr(reader.model, "_forward_committed_chunk", lambda *_a, **_k: pytest.fail("exact hit must not prefill"))
    try:
        result, scores = request(reader, tokens)
        assert result.cached_tokens == 256 and result.cache_event == "disk_hit"
        assert result.prompt_tokens == 0
        assert result.generated_token_ids == expected.generated_token_ids
        same_arrays(scores, expected_scores)
    finally:
        reader.close()
        store.close()


def test_qwen_memory_snapshot_is_consulted_before_disk(tmp_path, monkeypatch):
    memory = PromptCacheStore(max_size=2)
    store = DiskCheckpointStore(tmp_path, stride=256)
    run = generator(store, memory)
    tokens = [1, 2, 3, 1] * 65
    try:
        first, first_scores = request(run, tokens)
        monkeypatch.setattr(store, "restore", lambda *_a, **_k: pytest.fail("memory hit must not read disk"))
        second, second_scores = request(run, tokens)
        assert first.cache_event == "miss" and second.cache_event == "hit"
        assert second.cached_tokens == len(tokens)
        assert len(memory) == 1 and memory.nbytes > 0
        same_arrays(first_scores, second_scores)
    finally:
        run.close()
        store.close()


def test_qwen_write_fault_keeps_generation_and_disables_request_writer(tmp_path):
    calls = []

    def fail(*_args, **_kwargs):
        calls.append(1)
        raise OSError("disk full")

    store = DiskCheckpointStore(tmp_path, stride=256, save_payload_fn=fail)
    run = generator(store)
    cold = generator()
    try:
        result, scores = request(run, [1, 2, 3, 1] * 129)
        reference, reference_scores = request(cold, [1, 2, 3, 1] * 129)
        assert calls == [1] and result.disk_checkpoints_written == 0
        assert result.generated_token_ids == reference.generated_token_ids
        same_arrays(scores, reference_scores)
    finally:
        run.close()
        cold.close()
        store.close()


def test_qwen_corrupt_checkpoint_falls_back_without_counting_a_restore(tmp_path):
    store = DiskCheckpointStore(tmp_path, stride=256)
    run = generator(store)
    tokens = [1, 2, 3, 1] * 65
    try:
        expected, expected_scores = request(run, tokens)
        entry = store.index.entries()[0]
        # Corrupt only the test-owned payload while retaining a parseable container.
        arrays, meta = mx.load(str(tmp_path / entry.payload_path), return_metadata=True)
        meta["meta_state"] = '[{"schema":"invalid","composite":{}}]'
        mx.save_safetensors(str(tmp_path / entry.payload_path), arrays, meta)
        result, scores = request(run, tokens)
        assert result.cached_tokens == 0 and result.cache_event == "miss"
        assert store.quarantines == 1 and store.restores == 0
        assert result.generated_token_ids == expected.generated_token_ids
        same_arrays(scores, expected_scores)
    finally:
        run.close()
        store.close()


@pytest.mark.parametrize("different", ["render", "routing"])
def test_qwen_checkpoint_scope_separates_rendering_and_routing(tmp_path, different):
    store = DiskCheckpointStore(tmp_path, stride=256)
    run = generator(store, PromptCacheStore())
    tokens = [1, 2, 3, 1] * 65
    try:
        request(run, tokens)
        options = {}
        if different == "render":
            options["effective_rendering_id"] = "render-2"
        else:
            run.model.cache_identity += "|different-routing-policy"
        result, _ = request(run, tokens, **options)
        assert result.cached_tokens == 0 and result.cache_event == "miss"
        assert store.restores == 0
    finally:
        run.close()
        store.close()


def test_qwen_cancelled_response_does_not_publish_memory_state(tmp_path):
    store = DiskCheckpointStore(tmp_path, stride=256)
    memory = PromptCacheStore()
    run = generator(store, memory)
    tokens = [1, 2, 3, 1] * 65

    def fail(*_args):
        raise RuntimeError("client disconnected")

    try:
        with pytest.raises(RuntimeError, match="disconnected"):
            run(
                " ".join(map(str, tokens)), kv_policy=KVPolicy(live_kv_format=LIVE_KV_RAW),
                effective_rendering_id="render-1", max_tokens=3, temperature=0,
                response_callback=fail,
            )
        assert len(memory) == 0
        assert all(owner._closed for owner in run.model._coordinators)
        assert all(owner._closed for owner in run.model._plain_serial_lanes)
        result, _ = request(run, tokens)
        assert result.cached_tokens == 256 and result.cache_event == "disk_hit"
    finally:
        run.close()
        store.close()


def test_qwen_snapshot_exceeding_disk_budget_does_not_evict_or_fail_request(tmp_path):
    store = DiskCheckpointStore(tmp_path, stride=256, budget_bytes=128)
    run = generator(store)
    cold = generator()
    try:
        result, scores = request(run, [1, 2, 3, 1] * 65)
        reference, reference_scores = request(cold, [1, 2, 3, 1] * 65)
        assert result.disk_checkpoints_written == 0 and store.index.entries() == []
        assert result.generated_token_ids == reference.generated_token_ids
        same_arrays(scores, reference_scores)
    finally:
        run.close()
        cold.close()
        store.close()
