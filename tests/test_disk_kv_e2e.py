"""Disk prompt-cache codec, fail-closed loads, and the generator disk-hit path.

These run the real safetensors payload codec (leaf schema plus non-empty arrays)
and drive ``PrefixCacheGenerator`` through a disk restore on a tiny MLX model.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from moespresso.runtime.disk_kv import (  # noqa: E402
    DiskCheckpointStore,
    DiskKVInvalidPayload,
    DiskKVMetadataMismatch,
    FrontierTracker,
    FrontierWriter,
    build_cache_scope,
    caches_all_at_offset,
    decode_cache_payload,
    default_cache_registry,
    encode_cache_payload,
    load_prompt_cache_payload,
    save_prompt_cache_payload,
)
from moespresso.runtime.generation import GenerationResult  # noqa: E402
from moespresso.runtime.kv_policy import KVPolicy  # noqa: E402
from moespresso.runtime.prefix_cache import (  # noqa: E402
    PrefixCacheGenerator,
    cache_model_key,
)
from moespresso.runtime.serve import prefill_prompt_cache_chunks  # noqa: E402


# --- payload codec round-trip -------------------------------------------------


def test_leaf_schema_round_trips_arrays_none_and_zero_size_leaves():
    # Two synthetic layers whose state trees carry a real array, a zero-size
    # (empty) array, and a None slot: the DS4 frontier state has all three.
    empty = mx.zeros((1, 0, 8), dtype=mx.float16)
    real = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8)
    trees = [
        (real, empty, None),
        (mx.zeros((0,), dtype=mx.int32), (real, None)),
    ]
    arrays, schema = encode_cache_payload(trees)
    kinds = {row[1] for row in schema}
    assert kinds == {"array", "empty", "none"}

    rebuilt = decode_cache_payload(arrays, schema)
    assert np.array_equal(np.array(rebuilt[0][0]), np.array(real))
    assert rebuilt[0][1].shape == (1, 0, 8) and rebuilt[0][1].size == 0
    assert rebuilt[0][2] is None
    assert rebuilt[1][0].shape == (0,) and rebuilt[1][0].size == 0
    assert np.array_equal(np.array(rebuilt[1][1][0]), np.array(real))
    assert rebuilt[1][1][1] is None


def test_payload_save_load_round_trips_through_safetensors(tmp_path):
    real = mx.arange(12, dtype=mx.float32).reshape(1, 3, 4)
    empty = mx.zeros((1, 0, 4), dtype=mx.float32)
    trees = [(real, empty, None)]
    meta = [("dummy-meta-0",)]
    rel, size = save_prompt_cache_payload(
        tmp_path, "cafebabe",
        cache_state_trees=trees,
        meta_state_trees=meta,
        safety_metadata={"scope_hash": "abc", "token_count": "3"},
    )
    assert rel == "payloads/ca/cafebabe.safetensors"
    assert size > 0
    assert not list(tmp_path.rglob("*.tmp.safetensors"))

    state_trees, meta_trees, metadata = load_prompt_cache_payload(tmp_path, rel)
    assert metadata["scope_hash"] == "abc"
    assert metadata["token_count"] == "3"
    assert np.array_equal(np.array(state_trees[0][0]), np.array(real))
    assert state_trees[0][1].size == 0
    assert state_trees[0][2] is None
    assert meta_trees == [("dummy-meta-0",)]


def test_load_missing_payload_fails_closed(tmp_path):
    with pytest.raises(DiskKVInvalidPayload, match="missing"):
        load_prompt_cache_payload(tmp_path, "payloads/xx/nope.safetensors")


def test_load_truncated_payload_fails_closed(tmp_path):
    real = mx.arange(64, dtype=mx.float32).reshape(1, 8, 8)
    rel, _ = save_prompt_cache_payload(
        tmp_path, "deadbeef",
        cache_state_trees=[(real,)],
        meta_state_trees=[("m",)],
        safety_metadata={"scope_hash": "abc"},
    )
    path = tmp_path / rel
    full = path.stat().st_size
    with open(path, "r+b") as fh:
        fh.truncate(int(full * 0.5))
    with pytest.raises(DiskKVInvalidPayload, match="corrupt"):
        load_prompt_cache_payload(tmp_path, rel)


# --- store fail-closed on restore --------------------------------------------


def _model_key():
    return ("pkg", "render", "raw", 64, 0, "mlx_prompt_cache")


def _write_kv_checkpoint(store, scope, tokens, *, length):
    """Write a real KVCache checkpoint of `length` rows under `tokens`."""
    caches = [_kv_cache(length)]
    return store.write_manual_checkpoint(
        scope,
        tokens,
        cache_state_trees=[c.state for c in caches],
        meta_state_trees=[c.meta_state for c in caches],
        cache_class_names=tuple(type(c).__name__ for c in caches),
    )


def _kv_cache(length):
    from mlx_lm.models.cache import KVCache

    cache = KVCache()
    k = mx.random.normal((1, 2, length, 8)).astype(mx.float16)
    v = mx.random.normal((1, 2, length, 8)).astype(mx.float16)
    cache.update_and_fetch(k, v)
    mx.eval(cache.state)
    return cache


def test_restore_refuses_wrong_prefix_length_key(tmp_path):
    # A payload saved for a 500-token prefix, indexed under a rounded-down key
    # 256, is refused: the embedded token_count (500) will not equal 256.
    store = DiskCheckpointStore(tmp_path)
    scope = build_cache_scope(_model_key(), ("KVCache",))
    entry = _write_kv_checkpoint(store, scope, list(range(500)), length=500)

    # Corrupt the index entry to claim a shorter key while the payload holds 500.
    from dataclasses import replace

    from moespresso.runtime.disk_kv import token_prefix_hash

    bad = replace(
        entry,
        token_count=256,
        token_prefix_hash=token_prefix_hash(list(range(256))),
    )
    store.index.remove(entry)
    store.index.put(bad)

    with pytest.raises(DiskKVMetadataMismatch, match="token_count"):
        store.restore(
            scope, list(range(600)),
            make_cache_fn=lambda: [_kv_cache_empty()],
            registry=default_cache_registry(),
        )
    # The bad entry is quarantined and no longer reused.
    assert store.restore(
        scope, list(range(600)),
        make_cache_fn=lambda: [_kv_cache_empty()],
        registry=default_cache_registry(),
    ) is None


def _kv_cache_empty():
    from mlx_lm.models.cache import KVCache

    return KVCache()


def test_mark_used_failure_keeps_the_validated_restore(tmp_path):
    # A full or read-only disk during the LRU touch is index bookkeeping,
    # not a restore fault: the validated checkpoint must still serve.
    store = DiskCheckpointStore(tmp_path)
    scope = build_cache_scope(_model_key(), ("KVCache",))
    _write_kv_checkpoint(store, scope, list(range(512)), length=512)

    tmp_path.chmod(0o500)
    try:
        hit = store.restore(
            scope, list(range(600)),
            make_cache_fn=lambda: [_kv_cache_empty()],
            registry=default_cache_registry(),
        )
    finally:
        tmp_path.chmod(0o700)

    assert hit is not None
    assert hit.cached_tokens == 512
    assert hit.suffix_tokens == list(range(512, 600))
    assert store.restores == 1


def test_restore_refuses_class_list_the_registry_does_not_know(tmp_path):
    store = DiskCheckpointStore(tmp_path)
    scope = build_cache_scope(_model_key(), ("MysteryCache",))
    # Write a checkpoint whose recorded class name is unknown to the registry.
    caches = [_kv_cache(256)]
    entry = store.write_manual_checkpoint(
        scope,
        list(range(256)),
        cache_state_trees=[c.state for c in caches],
        meta_state_trees=[c.meta_state for c in caches],
        cache_class_names=("MysteryCache",),
    )
    assert entry is not None
    with pytest.raises(DiskKVMetadataMismatch, match="unregistered"):
        store.restore(
            scope, list(range(300)),
            make_cache_fn=lambda: [_kv_cache_empty()],
            registry=default_cache_registry(),
        )


def test_restore_refuses_corrupt_payload_and_quarantines(tmp_path):
    store = DiskCheckpointStore(tmp_path)
    scope = build_cache_scope(_model_key(), ("KVCache",))
    entry = _write_kv_checkpoint(store, scope, list(range(256)), length=256)

    path = tmp_path / entry.payload_path
    with open(path, "r+b") as fh:
        fh.truncate(int(path.stat().st_size * 0.4))

    with pytest.raises(DiskKVInvalidPayload):
        store.restore(
            scope, list(range(300)),
            make_cache_fn=lambda: [_kv_cache_empty()],
            registry=default_cache_registry(),
        )
    quarantined = list((tmp_path / "quarantine").glob("*.safetensors"))
    assert quarantined
    assert store.restore(
        scope, list(range(300)),
        make_cache_fn=lambda: [_kv_cache_empty()],
        registry=default_cache_registry(),
    ) is None


# --- generator disk-hit path on a tiny model ---------------------------------

_TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "num_experts": 2,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "rms_norm_eps": 1e-6,
    "vocab_size": 128,
    "rope_theta": 10000.0,
    "partial_rotary_factor": 0.25,
    "max_position_embeddings": 512,
    "linear_num_value_heads": 2,
    "linear_num_key_heads": 1,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 2,
    "tie_word_embeddings": False,
    "layer_types": ["linear_attention", "full_attention"],
}


def _tiny_model(**config_overrides):
    """The tiny hybrid model; overrides adjust the config per test.

    Live-KV quantization needs a head dimension divisible by the q8 group
    size, so the quantized-restore test passes a larger ``head_dim`` while
    the raw tests keep the small default.
    """
    import mlx_lm.models.qwen3_5_moe as M

    mx.random.seed(321)
    config = dict(_TEXT_CONFIG, **config_overrides)
    args = M.ModelArgs(model_type="qwen3_5_moe", text_config=config)
    model = M.Model(args)
    mx.eval(model.parameters())
    return model


def _greedy(logprobs):
    return mx.argmax(logprobs, axis=-1)


def _first_token(model, prompt, cache=None, **generate_kwargs):
    gen = generate_step(
        mx.array(prompt, dtype=mx.uint32), model, max_tokens=1,
        sampler=_greedy, prompt_cache=cache, **generate_kwargs)
    token, logprobs = next(gen)
    mx.eval(token, logprobs)
    token_id = int(token.item() if hasattr(token, "item") else token)
    return token_id, np.array(logprobs, dtype=np.float32)


class _TokenMap:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.bos_token = None

    def encode(self, text, **kwargs):
        return list(self.tokens)


class _NeverMemoryStore:
    """Memory store that always misses so the disk path is exercised."""

    def __init__(self):
        self.inserts = []

    def fetch_nearest_cache(self, model_key, tokens):
        return None, list(tokens)

    def insert_cache(self, model_key, tokens, prompt_cache):
        self.inserts.append((model_key, list(tokens), prompt_cache))

    def __len__(self):
        return len(self.inserts)


def _generate_capture(seen):
    def _fn(model, tokenizer, prompt, **kwargs):
        seen["prompt"] = list(prompt)
        seen["cached_tokens"] = kwargs["cached_tokens"]
        token, logprobs = _first_token(model, prompt, cache=kwargs["prompt_cache"])
        result = GenerationResult(
            text="",
            finish_reason="stop",
            prompt_tokens=len(prompt),
            completion_tokens=1,
            cached_tokens=kwargs["cached_tokens"],
            generated_token_ids=(token,),
            prompt_cache=kwargs["prompt_cache"],
        )
        result.logprobs = logprobs
        return result

    return _fn


def test_generator_disk_hit_restores_prefix_and_matches_memory_arm(tmp_path):
    model = _tiny_model()
    manifest = {"artifact_id": "pkg", "tokenizer": {"rendering_id": "render"}}
    policy = KVPolicy(live_kv_format="raw")
    prompt = [1, 5, 9, 13, 2, 7]
    prefix_len = 4

    # Build the disk scope exactly as the generator will, then checkpoint the
    # first `prefix_len` tokens through a fresh cache.
    model_key = cache_model_key(manifest, "render", policy)
    caches = make_prompt_cache(model)
    class_names = tuple(type(c).__name__ for c in caches)
    scope = build_cache_scope(model_key, class_names)

    prefill_caches = make_prompt_cache(model)
    model(mx.array([prompt[:prefix_len]], dtype=mx.uint32), cache=prefill_caches)
    mx.eval([c.state for c in prefill_caches])

    store = DiskCheckpointStore(tmp_path)
    store.write_manual_checkpoint(
        scope,
        prompt[:prefix_len],
        cache_state_trees=[c.state for c in prefill_caches],
        meta_state_trees=[c.meta_state for c in prefill_caches],
        cache_class_names=class_names,
    )

    # Arm M (memory): prefill the same prefix in-memory, continue with the suffix.
    mem_caches = make_prompt_cache(model)
    model(mx.array([prompt[:prefix_len]], dtype=mx.uint32), cache=mem_caches)
    mx.eval([c.state for c in mem_caches])
    mem_token, mem_logprobs = _first_token(model, prompt[prefix_len:], cache=mem_caches)

    # Arm R (disk): a fresh store simulates a restart; the generator restores.
    fresh_store = DiskCheckpointStore(tmp_path)
    seen = {}
    gen = PrefixCacheGenerator(
        model,
        _TokenMap(prompt),
        manifest,
        _NeverMemoryStore(),
        make_prompt_cache_fn=lambda m: make_prompt_cache(m),
        generate_fn=_generate_capture(seen),
        disk_store=fresh_store,
    )
    result = gen("prompt-text", kv_policy=policy, effective_rendering_id="render")

    assert result.cache_event == "disk_hit"
    assert result.cached_tokens == prefix_len
    assert seen["prompt"] == prompt[prefix_len:]
    assert seen["cached_tokens"] == prefix_len
    # R versus M at identical suffix geometry: bit-identical first token/logits.
    assert result.generated_token_ids[0] == mem_token
    assert float(np.max(np.abs(result.logprobs - mem_logprobs))) == 0.0
    # The mutated cache inserts into memory only.
    assert gen.cache_store.inserts


def test_generator_disk_hit_restores_quantized_live_kv(tmp_path):
    """Restore a checkpoint captured after live-KV quantization.

    The default KV policy quantizes the KV layers from offset zero, so a
    frontier checkpoint on a hybrid cache records QuantizedKVCache for the
    attention layers while a fresh ``make_cache`` builds raw KVCache. The
    restore path must graft into the entry's recorded classes; refusing the
    known raw-to-quantized conversion would fail every restore for a family
    whose default live KV format is quantized.
    """
    from mlx_lm.generate import maybe_quantize_kv_cache

    model = _tiny_model(head_dim=64)
    manifest = {"artifact_id": "pkg", "tokenizer": {"rendering_id": "render"}}
    policy = KVPolicy()
    assert policy.live_kv_format == "mlx_affine_q8"
    prompt = [1, 5, 9, 13, 2, 7]
    prefix_len = 4

    def quantize(caches):
        maybe_quantize_kv_cache(
            caches, quantized_kv_start=policy.quantized_kv_start,
            kv_group_size=policy.kv_group_size, kv_bits=8)

    # The scope keys on the fresh layout, exactly as the generator builds it;
    # the checkpoint is captured from a live cache after quantization, the
    # frontier writer's shape.
    model_key = cache_model_key(manifest, "render", policy)
    fresh_names = tuple(type(c).__name__ for c in make_prompt_cache(model))
    scope = build_cache_scope(model_key, fresh_names)

    prefill_caches = make_prompt_cache(model)
    model(mx.array([prompt[:prefix_len]], dtype=mx.uint32), cache=prefill_caches)
    quantize(prefill_caches)
    mx.eval([c.state for c in prefill_caches])
    live_names = tuple(type(c).__name__ for c in prefill_caches)
    assert "QuantizedKVCache" in live_names and live_names != fresh_names

    store = DiskCheckpointStore(tmp_path)
    store.write_manual_checkpoint(
        scope,
        prompt[:prefix_len],
        cache_state_trees=[c.state for c in prefill_caches],
        meta_state_trees=[c.meta_state for c in prefill_caches],
        cache_class_names=live_names,
    )

    # Arm M (memory): the same quantized prefill continued with the suffix.
    mem_caches = make_prompt_cache(model)
    model(mx.array([prompt[:prefix_len]], dtype=mx.uint32), cache=mem_caches)
    quantize(mem_caches)
    mx.eval([c.state for c in mem_caches])
    mem_token, mem_logprobs = _first_token(model, prompt[prefix_len:], cache=mem_caches)

    # Arm R (disk): a fresh store simulates a restart; the generator restores.
    fresh_store = DiskCheckpointStore(tmp_path)
    seen = {}
    gen = PrefixCacheGenerator(
        model,
        _TokenMap(prompt),
        manifest,
        _NeverMemoryStore(),
        make_prompt_cache_fn=lambda m: make_prompt_cache(m),
        generate_fn=_generate_capture(seen),
        disk_store=fresh_store,
    )
    result = gen("prompt-text", kv_policy=policy, effective_rendering_id="render")

    assert fresh_store.quarantines == 0
    assert result.cache_event == "disk_hit"
    assert result.cached_tokens == prefix_len
    assert seen["prompt"] == prompt[prefix_len:]
    restored_names = tuple(
        type(c).__name__ for c in result.prompt_cache)
    assert restored_names == live_names
    # R versus M at identical suffix geometry: bit-identical first token/logits.
    assert result.generated_token_ids[0] == mem_token
    assert float(np.max(np.abs(result.logprobs - mem_logprobs))) == 0.0


def test_reconstruct_still_refuses_a_genuinely_different_layout(tmp_path):
    """The raw-to-quantized conversion must not relax the layout gate."""
    model = _tiny_model()
    manifest = {"artifact_id": "pkg", "tokenizer": {"rendering_id": "render"}}
    policy = KVPolicy()
    prompt = [1, 5, 9, 13, 2, 7]
    prefix_len = 4

    model_key = cache_model_key(manifest, "render", policy)
    fresh_names = tuple(type(c).__name__ for c in make_prompt_cache(model))
    scope = build_cache_scope(model_key, fresh_names)

    prefill_caches = make_prompt_cache(model)
    model(mx.array([prompt[:prefix_len]], dtype=mx.uint32), cache=prefill_caches)
    mx.eval([c.state for c in prefill_caches])

    # Claim a swapped layout: the entry says the linear-attention layer held
    # a KV cache and the attention layer a recurrent one. No conversion
    # covers that; restore must refuse.
    wrong_names = tuple(
        "KVCache" if name == "ArraysCache" else "ArraysCache"
        for name in fresh_names)
    store = DiskCheckpointStore(tmp_path)
    store.write_manual_checkpoint(
        scope,
        prompt[:prefix_len],
        cache_state_trees=[c.state for c in prefill_caches],
        meta_state_trees=[c.meta_state for c in prefill_caches],
        cache_class_names=wrong_names,
    )

    fresh_store = DiskCheckpointStore(tmp_path)
    with pytest.raises(DiskKVMetadataMismatch):
        fresh_store.restore(
            scope,
            prompt,
            make_cache_fn=lambda: make_prompt_cache(model),
            registry=default_cache_registry(),
        )
    assert fresh_store.quarantines == 1


def test_generator_off_by_default_never_touches_disk(tmp_path):
    model = _tiny_model()
    manifest = {"artifact_id": "pkg", "tokenizer": {"rendering_id": "render"}}
    policy = KVPolicy(live_kv_format="raw")
    prompt = [1, 5, 9, 13]

    seen = {}
    gen = PrefixCacheGenerator(
        model,
        _TokenMap(prompt),
        manifest,
        _NeverMemoryStore(),
        make_prompt_cache_fn=lambda m: make_prompt_cache(m),
        generate_fn=_generate_capture(seen),
        disk_store=None,
    )
    result = gen("prompt-text", kv_policy=policy, effective_rendering_id="render")

    assert result.cache_event == "miss"
    assert result.cached_tokens == 0
    # No index file, no payloads: the feature never ran.
    assert not (tmp_path / "index.json").exists()
    assert not list(tmp_path.rglob("*.safetensors"))


# --- frontier writer through the real prefill loop ---------------------------


def _real_generate(model, tokenizer, prompt, **kwargs):
    """Drive one real prefill plus a single decode step, honoring the writer hooks.

    The generator passes the frontier writer's ``prefill_plan``,
    ``prefill_step_size``, and ``prompt_progress_callback`` through, so this
    stand-in runs the real chunk executor over the plan and feeds the tail to
    mlx-lm's ``generate_step`` exactly as the served path does.
    """
    cache = kwargs["prompt_cache"]
    progress = kwargs.get("prompt_progress_callback")
    consumed = 0
    tail = list(prompt)
    plan = kwargs.get("prefill_plan")
    if plan:
        consumed = prefill_prompt_cache_chunks(
            model, cache, list(prompt), plan,
            progress_callback=progress, total_tokens=len(prompt))
        tail = list(prompt[consumed:])
        if progress is not None:
            inner, offset, total = progress, consumed, len(prompt)

            def progress(processed, _total, _i=inner, _o=offset, _t=total):
                _i(_o + int(processed), _t)

    gen = generate_step(
        mx.array(tail, dtype=mx.uint32), model, max_tokens=1, sampler=_greedy,
        prompt_cache=cache,
        prefill_step_size=kwargs.get("prefill_step_size", 2048),
        prompt_progress_callback=progress,
    )
    token, logprobs = next(gen)
    mx.eval(logprobs)
    token_id = int(token.item() if hasattr(token, "item") else token)
    result = GenerationResult(
        text="", finish_reason="stop", prompt_tokens=len(prompt),
        completion_tokens=1, cached_tokens=kwargs["cached_tokens"],
        generated_token_ids=(token_id,), prompt_cache=cache)
    result.logprobs = np.array(logprobs, dtype=np.float32)
    return result


def _frontier_prompt():
    # 300 tokens crosses exactly one stride-256 frontier during prefill; the tiny
    # model's max_position_embeddings covers it.
    return [(i % 100) + 1 for i in range(300)]


def test_frontier_writer_fires_at_prefill_then_fresh_store_restores(tmp_path):
    model = _tiny_model()
    manifest = {"artifact_id": "pkg", "tokenizer": {"rendering_id": "render"}}
    policy = KVPolicy(live_kv_format="raw")
    prompt = _frontier_prompt()

    # Phase A: the writer fires at the 256 frontier during the 300-token prefill.
    store = DiskCheckpointStore(tmp_path, stride=256)
    gen = PrefixCacheGenerator(
        model, _TokenMap(prompt), manifest, _NeverMemoryStore(),
        make_prompt_cache_fn=lambda m: make_prompt_cache(m),
        generate_fn=_real_generate, disk_store=store)
    res = gen("prompt-text", kv_policy=policy, effective_rendering_id="render")

    entries = store.index.entries()
    assert [e.token_count for e in entries] == [256]
    assert res.disk_checkpoints_written == 1

    # Arm M: an in-memory restore of the 256 prefix, continued with the suffix.
    mem = make_prompt_cache(model)
    model(mx.array([prompt[:256]], dtype=mx.uint32), cache=mem)
    mx.eval([c.state for c in mem])
    mem_token, mem_logprobs = _first_token(model, prompt[256:], cache=mem)

    # Phase B: a fresh store instance is the crash-style restart. The generator
    # restores only up to the last completed checkpoint (256).
    fresh = DiskCheckpointStore(tmp_path, stride=256)
    seen = {}

    def _capture(model, tokenizer, p, **kwargs):
        seen["prompt"] = list(p)
        seen["cached"] = kwargs["cached_tokens"]
        return _real_generate(model, tokenizer, p, **kwargs)

    gen2 = PrefixCacheGenerator(
        model, _TokenMap(prompt), manifest, _NeverMemoryStore(),
        make_prompt_cache_fn=lambda m: make_prompt_cache(m),
        generate_fn=_capture, disk_store=fresh)
    res2 = gen2("prompt-text", kv_policy=policy, effective_rendering_id="render")

    assert res2.cache_event == "disk_hit"
    assert res2.cached_tokens == 256
    assert seen["prompt"] == prompt[256:]
    # R versus M at identical suffix geometry: bit-identical first token and logits.
    assert res2.generated_token_ids[0] == mem_token
    assert float(np.max(np.abs(res2.logprobs - mem_logprobs))) == 0.0


def test_frontier_writer_captures_at_exactly_the_frontier_offset(tmp_path):
    # The offset gate is the alignment invariant: every captured checkpoint's
    # cache offset equals the frontier exactly, so a write at a non-frontier
    # position is structurally impossible.
    model = _tiny_model()
    prompt = _frontier_prompt()
    caches = make_prompt_cache(model)
    tracker = FrontierTracker(
        stride=256, restored_prefix=0, full_tokens=prompt,
        scope=build_cache_scope(
            ("pkg", "render", "raw", 64, 0, "mlx_prompt_cache"),
            tuple(type(c).__name__ for c in caches)))

    class _Store:
        stride = 256

        def __init__(self):
            self.at = []

        def has_entry(self, scope, tokens):
            return False

        def write_checkpoint(self, scope, tokens, **kwargs):
            # At the moment of capture, assert the live caches are exactly here.
            assert caches_all_at_offset(caches, len(tokens))
            self.at.append(len(tokens))
            from moespresso.runtime.disk_kv import DiskKVEntry
            return DiskKVEntry.from_tokens(
                scope, tokens, payload_path="p", payload_bytes=1,
                cache_class_names=kwargs["cache_class_names"])

    store = _Store()
    writer = FrontierWriter(store, tracker=tracker, caches=caches, now_fn=lambda: 0)
    plan = writer.prefill_chunk_plan(2048)
    assert plan == [256]

    consumed = prefill_prompt_cache_chunks(
        model, caches, prompt, plan,
        progress_callback=writer.on_prompt_progress, total_tokens=len(prompt))
    gen = generate_step(
        mx.array(prompt[consumed:], dtype=mx.uint32), model, max_tokens=1,
        sampler=_greedy, prompt_cache=caches, prefill_step_size=2048)
    token, logprobs = next(gen)
    mx.eval(logprobs)

    assert store.at == [256]
    assert writer.refused_offset_mismatch == 0


def test_chunk_executor_is_bit_identical_to_the_uniform_prefill_loop():
    # The executor replaces mlx-lm's uniform-step prefill for the planned
    # span. At identical chunk boundaries the two code paths must compute
    # identical operations, so the first decode token and its logits must be
    # bit-identical; any deviation means the executor drifted from the
    # generate_step contract (stream, eval, quantization, mask).
    prompt = _frontier_prompt()

    uniform_model = _tiny_model()
    uniform_cache = make_prompt_cache(uniform_model)
    # Uniform 128-step loop chunks the 299-token prefill span as 128+128+43.
    uniform_token, uniform_logprobs = _first_token(
        uniform_model, prompt, cache=uniform_cache, prefill_step_size=128)

    planned_model = _tiny_model()
    planned_cache = make_prompt_cache(planned_model)
    consumed = prefill_prompt_cache_chunks(
        planned_model, planned_cache, prompt, [128, 128, 43],
        total_tokens=len(prompt))
    assert consumed == 299
    planned_token, planned_logprobs = _first_token(
        planned_model, prompt[consumed:], cache=planned_cache,
        prefill_step_size=128)

    assert planned_token == uniform_token
    assert float(np.max(np.abs(planned_logprobs - uniform_logprobs))) == 0.0


# --- M3: /health disk block and the session-key writer plumbing --------------


def test_cache_stats_carries_the_disk_block_when_enabled(tmp_path):
    model = _tiny_model()
    manifest = {
        "artifact_id": "pkg",
        "tokenizer": {"rendering_id": "render"},
        "architecture": {"cache_policy": {"kind": "mlx_prompt_cache"}},
    }
    store = DiskCheckpointStore(tmp_path, stride=256)
    gen = PrefixCacheGenerator(
        model, _TokenMap([1, 2, 3]), manifest, _NeverMemoryStore(),
        make_prompt_cache_fn=lambda m: make_prompt_cache(m), disk_store=store)
    stats = gen.cache_stats()
    assert stats["disk"]["enabled"] is True
    assert stats["disk"]["stride"] == 256
    assert stats["disk"]["restores"] == 0
    assert "budget_bytes" in stats["disk"]

    # With no disk store the block is a single enabled=false marker.
    gen_off = PrefixCacheGenerator(
        model, _TokenMap([1, 2, 3]), manifest, _NeverMemoryStore(),
        make_prompt_cache_fn=lambda m: make_prompt_cache(m), disk_store=None)
    assert gen_off.cache_stats()["disk"] == {"enabled": False}


def test_session_cache_key_reaches_the_written_entry(tmp_path):
    model = _tiny_model()
    manifest = {"artifact_id": "pkg", "tokenizer": {"rendering_id": "render"}}
    policy = KVPolicy(live_kv_format="raw")
    prompt = _frontier_prompt()
    store = DiskCheckpointStore(tmp_path, stride=256)
    gen = PrefixCacheGenerator(
        model, _TokenMap(prompt), manifest, _NeverMemoryStore(),
        make_prompt_cache_fn=lambda m: make_prompt_cache(m),
        generate_fn=_real_generate, disk_store=store)
    gen("prompt-text", kv_policy=policy, effective_rendering_id="render",
        session_cache_key="sess-Z")

    entries = store.index.entries()
    assert [e.token_count for e in entries] == [256]
    assert entries[0].session_cache_key == "sess-Z"
