"""Memory and disk prefix routing, publication, and fallback.

The cache manager is tested with fake tokenizers/cache stores/generators. That keeps the
logic exercisable without a GPU/model and proves the dangerous parts before MLX is involved:
exact token slicing, cache-key identity, producer rails, DSpark companions, disk arbitration,
miss/hit behavior, and publication under prompt plus generated tokens.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from moespresso.runtime.deepseek_v4.spec_serve import (
    ServedDrafter,
    SpecCacheCompanion,
    SpecContinuationError,
    spec_cache_producer_rail,
)
from moespresso.runtime.generation import GenerationResult
from moespresso.runtime.kv_policy import KVPolicyError, parse_kv_policy
from moespresso.runtime.prefix_cache import (
    CacheProbe,
    PrefixCacheGenerator,
    PromptCacheStore,
    cache_payload_kind,
    cache_model_key,
    encode_rendered_prompt,
)


class _FakeTokenizer:
    def __init__(self, tokens, *, bos_token=None):
        self.tokens = list(tokens)
        self.bos_token = bos_token
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return list(self.tokens)


class _FakeStore:
    def __init__(self, *, fetched_cache=None, rest=None):
        self.fetched_cache = fetched_cache
        self.rest = rest
        self.fetch_calls = []
        self.insert_calls = []

    def fetch_nearest_cache(self, model_key, tokens):
        self.fetch_calls.append((model_key, list(tokens)))
        if self.rest is None:
            return None, list(tokens)
        return self.fetched_cache, list(self.rest)

    def insert_cache(self, model_key, tokens, prompt_cache):
        self.insert_calls.append((model_key, list(tokens), prompt_cache))


class _FakeMakeCache:
    def __init__(self):
        self.calls = []

    def __call__(self, model):
        cache = [f"cache-for-{model}"]
        self.calls.append((model, cache))
        return cache


def test_encode_rendered_prompt_uses_mlx_string_prompt_special_token_rule():
    tok = _FakeTokenizer([1, 2], bos_token="<s>")
    assert encode_rendered_prompt(tok, "<s>hello") == [1, 2]
    assert tok.calls[-1] == ("<s>hello", {"add_special_tokens": False})

    assert encode_rendered_prompt(tok, "hello") == [1, 2]
    assert tok.calls[-1] == ("hello", {"add_special_tokens": True})

    no_bos = _FakeTokenizer([3], bos_token=None)
    encode_rendered_prompt(no_bos, "hello")
    assert no_bos.calls[-1] == ("hello", {"add_special_tokens": True})


def test_cache_model_key_uses_manifest_rendering_and_kv_policy():
    manifest = {"artifact_id": "pkg-a"}
    raw = parse_kv_policy({"live_kv_format": "raw"})
    key1 = cache_model_key(manifest, "render-a", raw)
    key2 = cache_model_key(manifest, "render-b", raw)
    assert key1 != key2
    assert key1[0] == "pkg-a"
    assert key1[1] == "render-a"
    assert key1[2] == "raw"
    assert key1[-1] == "mlx_prompt_cache"


def test_deepseek_prefix_cache_key_declares_composite_payload_and_rejects_q8():
    manifest = {
        "artifact_id": "ds4-pkg",
        "architecture": {
            "family": "deepseek_v4_flash",
            "cache_policy": {"kind": "deepseek_v4_composite", "generic_kv_bits": False},
        },
    }
    raw = parse_kv_policy({"live_kv_format": "raw"})
    assert cache_payload_kind(manifest) == "deepseek_v4_composite"
    assert cache_model_key(manifest, "render-ds4", raw)[-1] == "deepseek_v4_composite"

    tok = _FakeTokenizer([10, 20, 30])
    store = _FakeStore()
    ds4_payload = {"kind": "deepseek_v4_composite", "layers": []}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return GenerationResult(
            text="ok",
            generated_token_ids=(99,),
            prompt_cache=kwargs["prompt_cache"],
        )

    gen = PrefixCacheGenerator(
        "MODEL",
        tok,
        manifest,
        store,
        make_prompt_cache_fn=lambda model: ds4_payload,
        generate_fn=fake_generate,
    )

    assert gen.cache_stats()["supported_live_kv_formats"] == ["raw"]
    result = gen("rendered prompt", kv_policy=raw, effective_rendering_id="render-ds4")
    assert result.cache_event == "miss"
    assert store.insert_calls[0][2] is ds4_payload

    q8 = parse_kv_policy({"live_kv_format": "mlx_affine_q8"})
    try:
        gen("rendered prompt", kv_policy=q8, effective_rendering_id="render-ds4")
    except KVPolicyError as exc:
        assert "DeepSeek V4 composite cache" in str(exc)
    else:  # pragma: no cover - this should fail closed before generation
        raise AssertionError("DeepSeek V4 must not use the generic q8 KV path")


def test_raw_prefix_cache_miss_uses_full_tokens_and_inserts_prompt_plus_generated():
    tok = _FakeTokenizer([10, 20, 30])
    store = _FakeStore()
    make_cache = _FakeMakeCache()
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen["prompt"] = list(prompt)
        seen["prompt_cache"] = kwargs["prompt_cache"]
        seen["cached_tokens"] = kwargs["cached_tokens"]
        return GenerationResult(
            text="ok",
            finish_reason="stop",
            prompt_tokens=len(prompt),
            completion_tokens=1,
            generated_token_ids=(99,),
            prompt_cache=kwargs["prompt_cache"],
        )

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=make_cache,
        generate_fn=fake_generate,
    )
    result = gen("rendered prompt", kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
                 effective_rendering_id="render-id", max_tokens=1)

    assert result.text == "ok"
    assert seen["prompt"] == [10, 20, 30]
    assert seen["prompt_cache"] == ["cache-for-MODEL"]
    assert seen["cached_tokens"] == 0
    assert result.cache_event == "miss"
    assert result.cache_entries == 1
    assert result.cache_bytes is None
    assert store.fetch_calls[0][1] == [10, 20, 30]
    assert store.insert_calls[0][1] == [10, 20, 30, 99]


def test_prefix_cache_runs_after_generate_hook_after_insert():
    tok = _FakeTokenizer([10, 20, 30])
    store = _FakeStore()
    events = []

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return GenerationResult(
            text="ok",
            generated_token_ids=(99,),
            prompt_cache=kwargs["prompt_cache"],
        )

    def after_generate(model):
        events.append((model, len(store.insert_calls)))

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
        after_generate_fn=after_generate,
    )

    gen("rendered prompt", kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="render-id")

    assert events == [("MODEL", 1)]


def test_raw_prefix_cache_stats_reports_resident_cache_shape():
    tok = _FakeTokenizer([1])
    store = _FakeStore()
    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=lambda *args, **kwargs: GenerationResult(text="ok"),
    )

    assert gen.cache_stats() == {
        "default_live_kv_format": "mlx_affine_q8",
        "supported_live_kv_formats": ["raw", "mlx_affine_q8"],
        "entries": 0,
        "bytes": None,
        "disk": {"enabled": False},
    }


def test_prefix_cache_generator_close_is_idempotent():
    gen = PrefixCacheGenerator(
        "MODEL",
        _FakeTokenizer([1]),
        {"artifact_id": "pkg"},
        _FakeStore(),
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=lambda *args, **kwargs: GenerationResult(text="ok"),
    )

    gen.close()
    gen.close()

    assert gen._closed is True


def test_raw_prefix_cache_hit_uses_suffix_token_slice_not_retokenized_string():
    tok = _FakeTokenizer([10, 20, 30, 40])
    store = _FakeStore(fetched_cache=["cached"], rest=[30, 40])
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen["prompt"] = list(prompt)
        seen["cached_tokens"] = kwargs["cached_tokens"]
        return GenerationResult(
            text="hit",
            finish_reason="stop",
            prompt_tokens=len(prompt),
            completion_tokens=1,
            generated_token_ids=(50,),
            prompt_cache=kwargs["prompt_cache"],
        )

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
    )
    result = gen("rendered prompt", kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
                 effective_rendering_id="render-id")

    assert seen["prompt"] == [30, 40]
    assert seen["cached_tokens"] == 2
    assert result.cache_event == "hit"
    assert store.insert_calls
    assert len(tok.calls) == 1, "must encode full render only, never a suffix string"
    assert store.insert_calls[0][1] == [10, 20, 30, 40, 50]


def test_prefix_cache_accepts_q8_and_keeps_it_in_a_separate_cache_bucket():
    tok = _FakeTokenizer([4, 5, 6])
    store = _FakeStore()
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update(kwargs)
        return GenerationResult(
            text="q8",
            generated_token_ids=(7,),
            prompt_cache=kwargs["prompt_cache"],
        )

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
    )
    policy = parse_kv_policy({"live_kv_format": "mlx_affine_q8"})
    result = gen("rendered prompt", kv_policy=policy, effective_rendering_id="render-id")

    fetch_key = store.fetch_calls[0][0]
    insert_key = store.insert_calls[0][0]
    assert fetch_key == insert_key
    assert fetch_key[2] == "mlx_affine_q8"
    assert seen["kv_policy"].live_kv_format == "mlx_affine_q8"
    assert result.cache_event == "miss"


def _context_limit_generator(tokens, *, context_limit, store=None, seen=None):
    def fake_generate(model, tokenizer, prompt, **kwargs):
        if seen is not None:
            seen.clear()
            seen.update(kwargs)
        return GenerationResult(
            text="ok", generated_token_ids=(9,),
            prompt_cache=kwargs["prompt_cache"])

    return PrefixCacheGenerator(
        "MODEL", _FakeTokenizer(tokens), {"artifact_id": "pkg"},
        store if store is not None else _FakeStore(),
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
        context_limit=context_limit,
    )


def test_over_limit_request_is_refused_before_any_cache_access():
    from moespresso.runtime.generation import ContextLimitError

    store = _FakeStore()
    gen = _context_limit_generator(
        list(range(100)), context_limit=110, store=store)
    with pytest.raises(ContextLimitError) as e:
        gen("rendered prompt",
            kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
            effective_rendering_id="r", max_tokens=11)
    assert e.value.limit == 110
    assert e.value.prompt_tokens == 100
    assert e.value.max_tokens == 11
    assert "110" in str(e.value) and "100" in str(e.value)
    # The refusal fires before the fetch: the store hands entries out by
    # move, so a post-fetch refusal would cost the session its chain entry.
    assert store.fetch_calls == []
    assert store.insert_calls == []


def test_exactly_at_limit_request_passes():
    gen = _context_limit_generator(list(range(100)), context_limit=110)
    result = gen("rendered prompt",
                 kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
                 effective_rendering_id="r", max_tokens=10)
    assert result.text == "ok"


def test_under_limit_request_threads_internal_first_token_callback():
    seen = {}
    gen = _context_limit_generator(
        list(range(10)), context_limit=100_000, seen=seen)
    gen("rendered prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="r", max_tokens=10)
    assert set(seen) == {
        "prompt_cache", "cached_tokens", "kv_policy", "max_tokens",
        "temperature", "top_p", "first_token_callback",
    }


def test_no_declared_limit_applies_no_limit():
    gen = _context_limit_generator(list(range(100)), context_limit=None)
    result = gen("rendered prompt",
                 kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
                 effective_rendering_id="r", max_tokens=1_000_000)
    assert result.text == "ok"


def test_generator_forwards_sampling_knobs_and_omits_defaults():
    tok = _FakeTokenizer([1, 2, 3])
    store = _FakeStore()
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return GenerationResult(
            text="ok", generated_token_ids=(9,),
            prompt_cache=kwargs["prompt_cache"])

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
    )
    policy = parse_kv_policy({"live_kv_format": "raw"})

    gen("rendered prompt", kv_policy=policy, effective_rendering_id="r",
        top_k=20, min_p=0.25, presence_penalty=1.5)
    assert seen["top_k"] == 20
    assert seen["min_p"] == 0.25
    assert seen["presence_penalty"] == 1.5

    # Default-off values are not forwarded: the generate call shape is
    # byte-identical to a call made before the knobs existed.
    gen("rendered prompt", kv_policy=policy, effective_rendering_id="r")
    assert {"top_k", "min_p", "presence_penalty"}.isdisjoint(seen)


def test_sampling_knobs_never_enter_cache_identity():
    # Sampling is generation-only: two requests differing only in sampling
    # parameters must fetch and insert under the same model key and the same
    # token keys, so prefix reuse is unaffected.
    tok = _FakeTokenizer([1, 2, 3])
    store = _FakeStore()

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return GenerationResult(
            text="ok", generated_token_ids=(9,),
            prompt_cache=kwargs["prompt_cache"])

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
    )
    policy = parse_kv_policy({"live_kv_format": "raw"})

    gen("rendered prompt", kv_policy=policy, effective_rendering_id="r")
    gen("rendered prompt", kv_policy=policy, effective_rendering_id="r",
        top_k=20, min_p=0.25, presence_penalty=5.0)

    assert store.fetch_calls[0] == store.fetch_calls[1]
    assert store.insert_calls[0][0] == store.insert_calls[1][0]
    assert store.insert_calls[0][1] == store.insert_calls[1][1]


# --- the serve-side store: one live timeline per chain ------------------------


class _FakeLayerCache:
    """One fake per-layer cache with a byte size and a trimmability answer."""

    def __init__(self, nbytes=8, trimmable=False):
        self.nbytes = nbytes
        self.trimmable = trimmable


class _OffsetLayerCache(_FakeLayerCache):
    def __init__(self, offset, nbytes=8):
        super().__init__(nbytes=nbytes, trimmable=False)
        self.offset = offset


class _FakeCompanion:
    def __init__(self, nbytes):
        self.nbytes = nbytes


def _fake_cache(nbytes=8, *, trimmable=False, layers=2):
    return [_FakeLayerCache(nbytes, trimmable) for _ in range(layers)]


def _make_store(**kwargs):
    kwargs.setdefault(
        "can_trim_fn", lambda cache: all(c.trimmable for c in cache))
    kwargs.setdefault("trim_fn", lambda cache, n: None)
    return PromptCacheStore(**kwargs)


MODEL = ("pkg", "render", "raw", 64, 0, "kind")


def test_store_fetch_moves_the_chain_top_out_without_copying():
    store = _make_store()
    cache = _fake_cache(nbytes=10)
    store.insert_cache(MODEL, [1, 2, 3], cache)
    assert len(store) == 1
    assert store.nbytes == 20

    fetched, suffix = store.fetch_nearest_cache(MODEL, [1, 2, 3, 4, 5])
    # The stored object itself, by move: the store holds nothing until the
    # generated-through cache is reinserted under the extended key.
    assert fetched is cache
    assert suffix == [4, 5]
    assert len(store) == 0
    assert store.nbytes == 0


def test_store_hit_extend_cycle_keeps_one_entry_per_chain():
    # The cumulative-session pattern the road-test drives: every request
    # fetches the chain top, extends it, and reinserts full-plus-completion.
    store = _make_store()
    cache = _fake_cache()
    tokens = [1, 2]
    store.insert_cache(MODEL, tokens, cache)
    for turn in range(3, 9):
        request = tokens + [turn]
        fetched, suffix = store.fetch_nearest_cache(MODEL, request)
        assert fetched is cache
        # cached_tokens accounting is unchanged: the hit lands at exactly
        # the previous request's full-plus-completion length.
        assert len(request) - len(suffix) == len(tokens)
        tokens = request + [100 + turn]
        store.insert_cache(MODEL, tokens, cache)
        assert len(store) == 1


def test_store_insert_pops_superseded_prefixes_regardless_of_trimmability():
    # The untrimmable rotating-window case: without the popping the store
    # retains one full snapshot per request (measured 3.30 GB at 89.8k
    # retained tokens under the ten-entry cap).
    store = _make_store()
    store.insert_cache(MODEL, [1, 2], _fake_cache(trimmable=False))
    store.insert_cache(MODEL, [1, 2, 3, 4], _fake_cache(trimmable=False))
    assert len(store) == 1
    fetched, suffix = store.fetch_nearest_cache(MODEL, [1, 2, 3, 4, 5])
    assert suffix == [5]


def test_store_branch_from_an_earlier_prefix_is_a_miss():
    # The documented cost of the popping policy: a branch from an earlier
    # prefix has no in-memory entry left and falls to the disk frontier
    # restore (or a cold miss). The disk path restores exactly.
    store = _make_store()
    store.insert_cache(MODEL, [1, 2], _fake_cache(trimmable=False))
    store.insert_cache(MODEL, [1, 2, 3, 4], _fake_cache(trimmable=False))
    fetched, suffix = store.fetch_nearest_cache(MODEL, [1, 2, 9])
    assert fetched is None
    assert suffix == [1, 2, 9]


def test_store_interleaved_sessions_keep_independent_chains():
    store = _make_store()
    chain_a = _fake_cache(nbytes=3)
    chain_b = _fake_cache(nbytes=5)
    store.insert_cache(MODEL, [1, 1, 1], chain_a)
    store.insert_cache(MODEL, [2, 2], chain_b)
    assert len(store) == 2

    fetched, suffix = store.fetch_nearest_cache(MODEL, [1, 1, 1, 4])
    assert fetched is chain_a
    assert suffix == [4]
    store.insert_cache(MODEL, [1, 1, 1, 4, 5], chain_a)
    # Extending chain a never disturbs chain b.
    assert len(store) == 2
    fetched, suffix = store.fetch_nearest_cache(MODEL, [2, 2, 7])
    assert fetched is chain_b
    assert suffix == [7]


def test_store_exact_match_moves_out_and_reinsert_recovers_the_chain():
    store = _make_store()
    cache = _fake_cache()
    store.insert_cache(MODEL, [1, 2, 3], cache)
    fetched, suffix = store.fetch_nearest_cache(MODEL, [1, 2, 3])
    assert fetched is cache
    assert suffix == []
    assert len(store) == 0
    # The serve path regenerates and reinserts the extended key, so the
    # chain is covered again after the request.
    store.insert_cache(MODEL, [1, 2, 3, 4], cache)
    assert len(store) == 1


def test_store_longer_trimmable_entry_serves_a_trimmed_copy_and_stays():
    trims = []
    store = _make_store(trim_fn=lambda cache, n: trims.append(n))
    cache = _fake_cache(trimmable=True)
    store.insert_cache(MODEL, [1, 2, 3, 4, 5, 6], cache)

    fetched, suffix = store.fetch_nearest_cache(MODEL, [1, 2, 3, 9])
    # A deep copy trimmed down to the common prefix; the stored entry stays.
    assert fetched is not cache
    assert trims == [3]
    assert suffix == [9]
    assert len(store) == 1


def test_store_longer_untrimmable_entry_is_not_reused():
    store = _make_store()
    store.insert_cache(MODEL, [1, 2, 3, 4, 5, 6], _fake_cache(trimmable=False))
    fetched, suffix = store.fetch_nearest_cache(MODEL, [1, 2, 3, 9])
    assert fetched is None
    assert suffix == [1, 2, 3, 9]


def test_store_entry_cap_evicts_the_oldest_chain():
    store = _make_store(max_size=2)
    store.insert_cache(MODEL, [1], _fake_cache())
    store.insert_cache(MODEL, [2], _fake_cache())
    store.insert_cache(MODEL, [3], _fake_cache())
    assert len(store) == 2
    fetched, _ = store.fetch_nearest_cache(MODEL, [1, 5])
    assert fetched is None


def test_store_byte_bound_evicts_least_recently_inserted():
    store = _make_store(max_bytes=40)
    old = _fake_cache(nbytes=10)   # 20 bytes
    new = _fake_cache(nbytes=15)   # 30 bytes
    store.insert_cache(MODEL, [1], old)
    store.insert_cache(MODEL, [2], new)
    # 50 bytes exceeds the 40-byte bound: the oldest chain goes.
    assert len(store) == 1
    assert store.nbytes == 30
    fetched, _ = store.fetch_nearest_cache(MODEL, [2, 5])
    assert fetched is new


def test_store_nbytes_stays_exact_through_move_and_reinsert():
    store = _make_store()
    cache = _fake_cache(nbytes=7, layers=3)  # 21 bytes
    store.insert_cache(MODEL, [1, 2], cache)
    assert store.nbytes == 21
    store.fetch_nearest_cache(MODEL, [1, 2, 3])
    assert store.nbytes == 0
    store.insert_cache(MODEL, [1, 2, 3, 4], cache)
    assert store.nbytes == 21
    # Reinserting the same key replaces the entry without double counting.
    store.insert_cache(MODEL, [1, 2, 3, 4], cache)
    assert store.nbytes == 21


def test_store_model_keys_partition_entries():
    store = _make_store()
    other = ("pkg-b", "render", "raw", 64, 0, "kind")
    cache = _fake_cache()
    store.insert_cache(MODEL, [1, 2], cache)
    fetched, suffix = store.fetch_nearest_cache(other, [1, 2, 3])
    assert fetched is None
    assert suffix == [1, 2, 3]


def test_store_exact_probe_is_immutable_and_does_not_move_the_entry():
    store = _make_store()
    cache = _fake_cache(nbytes=7)
    companion = _FakeCompanion(5)
    rail = ("spec", "dspark", "sidecar-a")
    store.insert_cache_with_companion(
        MODEL,
        [1, 2, 3],
        cache,
        producer_rail=rail,
        companion=companion,
    )

    probe = store.probe_nearest_cache(
        MODEL,
        [1, 2, 3],
        producer_rail=rail,
    )

    assert probe == CacheProbe("exact", 3, True)
    with pytest.raises(FrozenInstanceError):
        probe.cached_tokens = 2
    assert len(store) == 1
    assert store.nbytes == 19
    fetched, suffix, fetched_companion = (
        store.fetch_nearest_cache_with_companion(
            MODEL,
            [1, 2, 3],
            producer_rail=rail,
        )
    )
    assert fetched is cache
    assert suffix == []
    assert fetched_companion is companion


def test_store_shorter_probe_reports_companion_without_moving_it():
    store = _make_store()
    cache = _fake_cache()
    companion = _FakeCompanion(9)
    rail = ("spec", "dspark", "sidecar-a")
    store.insert_cache_with_companion(
        MODEL,
        [1, 2, 3],
        cache,
        producer_rail=rail,
        companion=companion,
    )

    probe = store.probe_nearest_cache(
        MODEL,
        [1, 2, 3, 4, 5],
        producer_rail=rail,
    )

    assert probe == CacheProbe("shorter", 3, True)
    assert len(store) == 1
    assert store.nbytes == 25


def test_store_longer_trim_probe_does_not_copy_trim_or_remove():
    copies = []
    trims = []

    class _CopyTrackedLayer(_FakeLayerCache):
        def __deepcopy__(self, _memo):
            copies.append(self)
            return type(self)(self.nbytes, self.trimmable)

    store = _make_store(trim_fn=lambda cache, n: trims.append(n))
    cache = [_CopyTrackedLayer(trimmable=True) for _ in range(2)]
    store.insert_cache(MODEL, [1, 2, 3, 4, 5], cache)

    probe = store.probe_nearest_cache(MODEL, [1, 2, 3, 9])

    assert probe == CacheProbe("trim", 3, False)
    assert copies == []
    assert trims == []
    assert len(store) == 1
    fetched, suffix = store.fetch_nearest_cache(MODEL, [1, 2, 3, 9])
    assert fetched is not cache
    assert len(copies) == 2
    assert trims == [2]
    assert suffix == [9]


def test_store_probe_is_rail_local():
    store = _make_store()
    rail_a = ("spec", "dspark", "sidecar-a")
    rail_b = ("spec", "dspark", "sidecar-b")
    store.insert_cache_with_companion(
        MODEL,
        [1, 2, 3],
        _fake_cache(),
        producer_rail=rail_a,
    )

    assert store.probe_nearest_cache(
        MODEL,
        [1, 2, 3, 4],
        producer_rail=rail_b,
    ) == CacheProbe("miss", 0, False)
    assert len(store) == 1


def test_store_probes_allow_comparing_reusable_depth_across_rails():
    store = _make_store()
    spec_rail = ("spec", "dspark", "sidecar-a")
    store.insert_cache(MODEL, [1, 2, 3, 4, 5], _fake_cache())
    store.insert_cache_with_companion(
        MODEL,
        [1, 2, 3],
        _fake_cache(),
        producer_rail=spec_rail,
        companion=_FakeCompanion(9),
    )

    tokens = [1, 2, 3, 4, 5, 6]
    plain = store.probe_nearest_cache(MODEL, tokens)
    speculative = store.probe_nearest_cache(
        MODEL,
        tokens,
        producer_rail=spec_rail,
    )

    assert plain == CacheProbe("shorter", 5, False)
    assert speculative == CacheProbe("shorter", 3, True)
    assert plain.cached_tokens > speculative.cached_tokens
    assert len(store) == 2


def test_store_enhanced_insert_reports_oversized_self_eviction_exactly():
    store = _make_store(max_bytes=12)
    rail = ("spec", "dspark", "sidecar-a")

    retained = store.insert_cache_with_companion(
        MODEL,
        [1, 2],
        _fake_cache(nbytes=4),
        producer_rail=rail,
        companion=_FakeCompanion(5),
    )

    assert retained is False
    assert len(store) == 0
    assert store.nbytes == 0
    assert store.insert_cache(MODEL, [9], _fake_cache(nbytes=3)) is None
    assert len(store) == 1
    assert store.nbytes == 6


def test_store_old_api_remains_plain_rail_and_cannot_move_spec_entries():
    store = _make_store()
    plain_cache = _fake_cache(nbytes=3)
    spec_cache = _fake_cache(nbytes=5)
    companion = _FakeCompanion(11)
    spec_rail = ("spec", "dspark", "sidecar-a")

    store.insert_cache(MODEL, [1, 2], plain_cache)
    store.insert_cache_with_companion(
        MODEL,
        [1, 2],
        spec_cache,
        producer_rail=spec_rail,
        companion=companion,
    )

    fetched, suffix = store.fetch_nearest_cache(MODEL, [1, 2, 3])
    assert fetched is plain_cache
    assert suffix == [3]

    fetched, suffix, fetched_companion = (
        store.fetch_nearest_cache_with_companion(
            MODEL,
            [1, 2, 4],
            producer_rail=spec_rail,
        )
    )
    assert fetched is spec_cache
    assert suffix == [4]
    assert fetched_companion is companion


def test_store_companion_moves_with_target_and_accounts_bytes():
    store = _make_store()
    cache = _fake_cache(nbytes=7, layers=3)
    companion = _FakeCompanion(13)
    rail = ("spec", "dspark", "sidecar-a")
    store.insert_cache_with_companion(
        MODEL,
        [1, 2],
        cache,
        producer_rail=rail,
        companion=companion,
    )
    assert store.nbytes == 34

    fetched, suffix, fetched_companion = (
        store.fetch_nearest_cache_with_companion(
            MODEL,
            [1, 2, 3],
            producer_rail=rail,
        )
    )
    assert fetched is cache
    assert suffix == [3]
    assert fetched_companion is companion
    assert len(store) == 0
    assert store.nbytes == 0


def test_store_strict_prefix_supersession_is_scoped_to_producer_rail():
    store = _make_store()
    rail_a = ("spec", "dspark", "sidecar-a")
    rail_b = ("spec", "dspark", "sidecar-b")
    cache_a_short = _fake_cache()
    cache_a_long = _fake_cache()
    cache_b_short = _fake_cache()

    store.insert_cache_with_companion(
        MODEL, [1, 2], cache_a_short, producer_rail=rail_a
    )
    store.insert_cache_with_companion(
        MODEL, [1, 2], cache_b_short, producer_rail=rail_b
    )
    store.insert_cache_with_companion(
        MODEL, [1, 2, 3], cache_a_long, producer_rail=rail_a
    )
    assert len(store) == 2

    fetched, suffix, _ = store.fetch_nearest_cache_with_companion(
        MODEL, [1, 2, 9], producer_rail=rail_a
    )
    assert fetched is None
    assert suffix == [1, 2, 9]
    fetched, suffix, _ = store.fetch_nearest_cache_with_companion(
        MODEL, [1, 2, 9], producer_rail=rail_b
    )
    assert fetched is cache_b_short
    assert suffix == [9]


def test_store_longer_trim_search_stays_on_rail_and_leaves_companion_stored():
    trims = []
    store = _make_store(trim_fn=lambda cache, n: trims.append((cache, n)))
    rail_a = ("spec", "dspark", "sidecar-a")
    rail_b = ("spec", "dspark", "sidecar-b")
    cache_a = _fake_cache(trimmable=True)
    cache_b = _fake_cache(trimmable=True)
    companion_a = _FakeCompanion(9)
    store.insert_cache_with_companion(
        MODEL,
        [1, 2, 3, 4, 5],
        cache_a,
        producer_rail=rail_a,
        companion=companion_a,
    )
    store.insert_cache_with_companion(
        MODEL,
        [1, 2, 3, 9, 10],
        cache_b,
        producer_rail=rail_b,
    )

    fetched, suffix, companion = store.fetch_nearest_cache_with_companion(
        MODEL, [1, 2, 3, 8], producer_rail=rail_a
    )
    assert fetched is not cache_a
    assert suffix == [8]
    assert companion is None
    assert trims == [(fetched, 2)]
    assert len(store) == 2

    fetched, suffix, companion = store.fetch_nearest_cache_with_companion(
        MODEL, [1, 2, 3, 4, 5], producer_rail=rail_a
    )
    assert fetched is cache_a
    assert suffix == []
    assert companion is companion_a


def test_store_companion_bytes_participate_in_global_eviction():
    store = _make_store(max_bytes=30)
    plain_cache = _fake_cache(nbytes=4)  # 8 bytes
    spec_cache = _fake_cache(nbytes=5)   # 10 bytes
    spec_rail = ("spec", "dspark", "sidecar-a")
    store.insert_cache(MODEL, [1], plain_cache)
    store.insert_cache_with_companion(
        MODEL,
        [2],
        spec_cache,
        producer_rail=spec_rail,
        companion=object(),
        companion_nbytes=20,
    )

    # The 38-byte total exceeds the global bound. The oldest plain entry is
    # evicted and the speculative target plus declared companion occupy 30.
    assert len(store) == 1
    assert store.nbytes == 30
    fetched, suffix = store.fetch_nearest_cache(MODEL, [1, 3])
    assert fetched is None
    assert suffix == [1, 3]
    fetched, suffix, companion = store.fetch_nearest_cache_with_companion(
        MODEL, [2, 4], producer_rail=spec_rail
    )
    assert fetched is spec_cache
    assert suffix == [4]
    assert companion is not None


def test_store_plain_rail_refuses_a_companion_that_legacy_fetch_would_drop():
    store = _make_store()
    with pytest.raises(ValueError, match="plain cache rail"):
        store.insert_cache_with_companion(
            MODEL,
            [1, 2],
            _fake_cache(),
            companion=_FakeCompanion(9),
        )


def test_store_refuses_nonpositive_bounds():
    with pytest.raises(ValueError, match="max_size"):
        PromptCacheStore(max_size=0)
    with pytest.raises(ValueError, match="max_bytes"):
        PromptCacheStore(max_bytes=0)


class _ResumeDrafter:
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


def _resume_model():
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=_ResumeDrafter(),
        tap="tap",
        artifact_id="art:side",
    )
    return SimpleNamespace(_moespresso_ds4_drafter=served), served


def _resume_companion(frontier):
    return SpecCacheCompanion(
        family="dspark",
        artifact_id="art:side",
        schedule="fixed:3",
        frontier=frontier,
        capsule=SimpleNamespace(frontier=frontier, nbytes=16),
    )


def _resume_generator(tokens, store, generate_fn, *, disk_store=None, ready=None):
    model, served = _resume_model()
    generator = PrefixCacheGenerator(
        model,
        _FakeTokenizer(tokens),
        {"artifact_id": "pkg", "architecture": {"family": "deepseek_v4_flash"}},
        store,
        make_prompt_cache_fn=lambda _model: [_OffsetLayerCache(0)],
        generate_fn=generate_fn,
        disk_store=disk_store,
    )
    return generator, served


def _raw_policy():
    return parse_kv_policy({"live_kv_format": "raw"})


def test_resumable_dspark_cold_miss_publishes_frontier_pair_on_spec_rail(
    monkeypatch,
):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    store = _make_store()
    target = [_OffsetLayerCache(4)]
    companion = _resume_companion(4)
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        return GenerationResult(
            text="s",
            generated_token_ids=(9, 10),
            prompt_cache=None,
            speculative_prompt_cache=target,
            cache_frontier=4,
            cache_companion=companion,
            speculative={"drafter": "dspark", "cache_publication": {"status": "ready"}},
        )

    generator, served = _resume_generator([1, 2, 3, 4], store, fake_generate)
    result = generator(
        "prompt",
        kv_policy=_raw_policy(),
        effective_rendering_id="r",
        temperature=0.0,
    )

    assert seen["prompt"] == [1, 2, 3, 4]
    assert seen["prompt_cache"] is None
    assert "spec_continuation" not in seen
    assert result.cache_event == "miss"
    assert result.speculative["cache_publication"] == {"status": "published"}
    rail = spec_cache_producer_rail(served)
    probe = store.probe_nearest_cache(
        cache_model_key(generator.manifest, "r", _raw_policy()),
        [1, 2, 3, 4],
        producer_rail=rail,
    )
    assert probe == CacheProbe("exact", 4, True)
    assert store.probe_nearest_cache(
        cache_model_key(generator.manifest, "r", _raw_policy()),
        [1, 2, 3, 4],
    ) == CacheProbe("miss", 0, False)


def test_terminal_response_callback_routes_installed_dspark_to_plain_cache(
    monkeypatch,
):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    store = _make_store()
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        return GenerationResult(
            text="p",
            generated_token_ids=(9,),
            prompt_cache=kwargs["prompt_cache"],
        )

    generator, served = _resume_generator(
        [1, 2, 3, 4], store, fake_generate)
    def stop():
        return False
    result = generator(
        "prompt",
        kv_policy=_raw_policy(),
        effective_rendering_id="r",
        temperature=0.0,
        response_stop_callback=stop,
    )

    assert seen["prompt"] == [1, 2, 3, 4]
    assert seen["prompt_cache"] is not None
    assert seen["response_stop_callback"] is stop
    assert "spec_continuation" not in seen
    assert result.speculative is None
    model_key = cache_model_key(generator.manifest, "r", _raw_policy())
    assert store.probe_nearest_cache(
        model_key, [1, 2, 3, 4, 9]
    ) == CacheProbe("exact", 5, False)
    assert store.probe_nearest_cache(
        model_key,
        [1, 2, 3, 4, 9],
        producer_rail=spec_cache_producer_rail(served),
    ) == CacheProbe("miss", 0, False)


def test_resumable_dspark_quantized_live_kv_uses_plain_cache_rail(monkeypatch):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    store = _make_store()
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        return GenerationResult(
            text="p",
            generated_token_ids=(9,),
            prompt_cache=kwargs["prompt_cache"],
        )

    policy = parse_kv_policy({"live_kv_format": "mlx_affine_q8"})
    generator, served = _resume_generator([1, 2, 3, 4], store, fake_generate)
    result = generator(
        "prompt",
        kv_policy=policy,
        effective_rendering_id="r",
        temperature=0.0,
    )

    assert seen["prompt"] == [1, 2, 3, 4]
    assert seen["prompt_cache"][0].offset == 0
    assert "spec_continuation" not in seen
    assert not any(key.startswith("spec_prefill_") for key in seen)
    assert result.cache_event == "miss"
    model_key = cache_model_key(generator.manifest, "r", policy)
    assert store.probe_nearest_cache(
        model_key,
        [1, 2, 3, 4, 9],
    ) == CacheProbe("exact", 5, False)
    rail = spec_cache_producer_rail(served)
    assert store.probe_nearest_cache(
        model_key,
        [1, 2, 3, 4, 9],
        producer_rail=rail,
    ) == CacheProbe("miss", 0, False)


def test_resumable_dspark_hit_uses_suffix_and_republishes_at_live_frontier(
    monkeypatch,
):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    store = _make_store()
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        continuation = kwargs["spec_continuation"]
        continuation.target_cache[0].offset = 5
        return GenerationResult(
            text="s",
            generated_token_ids=(9,),
            prompt_cache=None,
            speculative_prompt_cache=continuation.target_cache,
            cache_frontier=5,
            cache_companion=_resume_companion(5),
            speculative={"drafter": "dspark", "cache_publication": {"status": "ready"}},
        )

    generator, served = _resume_generator([1, 2, 3, 4, 5], store, fake_generate)
    model_key = cache_model_key(generator.manifest, "r", _raw_policy())
    rail = spec_cache_producer_rail(served)
    target = [_OffsetLayerCache(3)]
    store.insert_cache_with_companion(
        model_key,
        [1, 2, 3],
        target,
        producer_rail=rail,
        companion=_resume_companion(3),
    )

    result = generator(
        "prompt",
        kv_policy=_raw_policy(),
        effective_rendering_id="r",
        temperature=0.0,
    )

    assert seen["prompt"] == [4, 5]
    assert seen["prompt_cache"] is None
    assert seen["cached_tokens"] == 3
    assert seen["spec_continuation"].target_cache is target
    assert result.cache_event == "hit"
    assert result.speculative["cache_resume"] == {"status": "hit", "frontier": 3}
    assert result.speculative["cache_publication"] == {"status": "published"}
    assert store.probe_nearest_cache(
        model_key,
        [1, 2, 3, 4, 5, 9],
        producer_rail=rail,
    ) == CacheProbe("shorter", 5, True)


def test_deeper_plain_cache_beats_shallower_dspark_and_leaves_it_resident(
    monkeypatch,
):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    store = _make_store()
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        return GenerationResult(
            text="p",
            generated_token_ids=(9,),
            prompt_cache=kwargs["prompt_cache"],
        )

    generator, served = _resume_generator([1, 2, 3, 4, 5], store, fake_generate)
    model_key = cache_model_key(generator.manifest, "r", _raw_policy())
    rail = spec_cache_producer_rail(served)
    plain_target = _fake_cache()
    spec_target = [_OffsetLayerCache(3)]
    store.insert_cache(model_key, [1, 2, 3, 4], plain_target)
    store.insert_cache_with_companion(
        model_key,
        [1, 2, 3],
        spec_target,
        producer_rail=rail,
        companion=_resume_companion(3),
    )

    result = generator(
        "prompt",
        kv_policy=_raw_policy(),
        effective_rendering_id="r",
        temperature=0.0,
    )

    assert seen["prompt"] == [5]
    assert seen["prompt_cache"] is plain_target
    assert "spec_continuation" not in seen
    assert result.cache_event == "hit"
    assert store.probe_nearest_cache(
        model_key,
        [1, 2, 3, 8],
        producer_rail=rail,
    ) == CacheProbe("shorter", 3, True)


def test_ready_failure_after_probe_moves_neither_cache_rail(monkeypatch):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    store = _make_store()
    generator, served = _resume_generator(
        [1, 2, 3, 4],
        store,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generation must not start")
        ),
    )
    model_key = cache_model_key(generator.manifest, "r", _raw_policy())
    rail = spec_cache_producer_rail(served)
    store.insert_cache_with_companion(
        model_key,
        [1, 2, 3],
        [_OffsetLayerCache(3)],
        producer_rail=rail,
        companion=_resume_companion(3),
    )

    with pytest.raises(RuntimeError, match="socket refused"):
        generator(
            "prompt",
            kv_policy=_raw_policy(),
            effective_rendering_id="r",
            temperature=0.0,
            ready_callback=lambda: (_ for _ in ()).throw(
                RuntimeError("socket refused")
            ),
        )

    assert store.probe_nearest_cache(
        model_key,
        [1, 2, 3, 4],
        producer_rail=rail,
    ) == CacheProbe("shorter", 3, True)


def test_bad_dspark_companion_falls_back_to_plain_on_same_rail(monkeypatch):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    store = _make_store()
    calls = []

    def fake_generate(model, tokenizer, prompt, **kwargs):
        calls.append((list(prompt), dict(kwargs)))
        if "spec_continuation" in kwargs:
            raise SpecContinuationError("bad capsule")
        kwargs["first_token_callback"]()
        target = kwargs["prompt_cache"]
        target[0].offset = 5
        return GenerationResult(
            text="p",
            generated_token_ids=(9,),
            prompt_cache=target,
        )

    generator, served = _resume_generator([1, 2, 3, 4, 5], store, fake_generate)
    times = iter([10.0, 20.0, 30.0])
    generator.clock = times.__next__
    model_key = cache_model_key(generator.manifest, "r", _raw_policy())
    rail = spec_cache_producer_rail(served)
    target = [_OffsetLayerCache(3)]
    store.insert_cache_with_companion(
        model_key,
        [1, 2, 3],
        target,
        producer_rail=rail,
        companion=_resume_companion(3),
    )

    result = generator(
        "prompt",
        kv_policy=_raw_policy(),
        effective_rendering_id="r",
        temperature=0.0,
    )

    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == [4, 5]
    assert calls[1][1]["prompt_cache"] is target
    assert "spec_continuation" not in calls[1][1]
    assert result.cache_event == "hit"
    assert result.ready_to_first_token_seconds == 10.0
    assert result.prompt_cache is None
    assert store.probe_nearest_cache(
        model_key,
        [1, 2, 3, 4, 5, 9],
        producer_rail=rail,
    ) == CacheProbe("shorter", 5, False)
    assert store.probe_nearest_cache(
        model_key,
        [1, 2, 3, 4, 5, 9],
    ) == CacheProbe("miss", 0, False)


def test_valid_spec_target_survives_companion_export_failure(monkeypatch):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    store = _make_store()
    target = [_OffsetLayerCache(4)]

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return GenerationResult(
            text="s",
            generated_token_ids=(9,),
            prompt_cache=None,
            speculative_prompt_cache=target,
            cache_frontier=4,
            cache_companion=None,
            speculative={
                "drafter": "dspark",
                "cache_publication": {
                    "status": "skipped",
                    "reason": "drafter_export_failed",
                },
            },
        )

    generator, served = _resume_generator([1, 2, 3, 4], store, fake_generate)
    result = generator(
        "prompt",
        kv_policy=_raw_policy(),
        effective_rendering_id="r",
        temperature=0.0,
    )

    assert result.speculative["cache_publication"] == {
        "status": "published_target_only",
        "reason": "drafter_export_failed",
    }
    model_key = cache_model_key(generator.manifest, "r", _raw_policy())
    assert store.probe_nearest_cache(
        model_key,
        [1, 2, 3, 4],
        producer_rail=spec_cache_producer_rail(served),
    ) == CacheProbe("exact", 4, False)


def test_exact_spec_entry_stays_resident_while_disk_target_serves_plain(
    monkeypatch,
):
    from moespresso.runtime.deepseek_v4 import spec_serve

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)

    class _RestoringDisk:
        stride = None

        def __init__(self, target):
            self.target = target
            self.restore_calls = 0

        def restore(self, scope, tokens, **kwargs):
            self.restore_calls += 1
            return SimpleNamespace(
                prompt_cache=self.target,
                suffix_tokens=[tokens[-1]],
                cached_tokens=len(tokens) - 1,
            )

        def stats(self):
            return {"enabled": True}

    store = _make_store()
    disk_target = [_OffsetLayerCache(3)]
    disk = _RestoringDisk(disk_target)
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        return GenerationResult(
            text="p",
            generated_token_ids=(9,),
            prompt_cache=kwargs["prompt_cache"],
        )

    generator, served = _resume_generator(
        [1, 2, 3, 4], store, fake_generate, disk_store=disk
    )
    model_key = cache_model_key(generator.manifest, "r", _raw_policy())
    rail = spec_cache_producer_rail(served)
    store.insert_cache_with_companion(
        model_key,
        [1, 2, 3, 4],
        [_OffsetLayerCache(4)],
        producer_rail=rail,
        companion=_resume_companion(4),
    )

    result = generator(
        "prompt",
        kv_policy=_raw_policy(),
        effective_rendering_id="r",
        temperature=0.0,
    )

    assert disk.restore_calls == 1
    assert seen["prompt"] == [4]
    assert seen["prompt_cache"] is disk_target
    assert "spec_continuation" not in seen
    assert result.cache_event == "disk_hit"
    assert store.probe_nearest_cache(
        model_key,
        [1, 2, 3, 4],
        producer_rail=rail,
    ) == CacheProbe("exact", 4, True)


class _FakeDiskStore:
    """Disk-store stand-in: a stride, a dedupe read, and no entries."""

    def __init__(self, stride=256):
        self.stride = stride

    def has_entry(self, scope, tokens):
        return False

    def stats(self):
        return {"enabled": True}


def test_plain_disk_restore_and_ready_to_first_token_are_timed():
    full = [1, 2, 3, 4]
    disk_cache = [SimpleNamespace(offset=3, nbytes=8)]

    class _RestoringDiskStore:
        stride = None

        def restore(self, _scope, tokens, **_kwargs):
            return SimpleNamespace(
                prompt_cache=disk_cache,
                suffix_tokens=[tokens[-1]],
                cached_tokens=len(tokens) - 1,
            )

        def stats(self):
            return {"enabled": True}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        kwargs["first_token_callback"]()
        return GenerationResult(
            text="ok",
            cached_tokens=kwargs["cached_tokens"],
            prompt_cache=kwargs["prompt_cache"],
        )

    gen = PrefixCacheGenerator(
        "MODEL",
        _FakeTokenizer(full),
        {"artifact_id": "pkg"},
        _FakeStore(),
        make_prompt_cache_fn=lambda model: [SimpleNamespace(offset=0)],
        generate_fn=fake_generate,
        disk_store=_RestoringDiskStore(),
    )
    times = iter([10.0, 11.0, 13.0, 20.0])
    gen.clock = times.__next__

    result = gen(
        "rendered prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="render-id",
    )

    assert result.cache_event == "disk_hit"
    assert result.cached_tokens == 3
    assert result.disk_restore_seconds == 2.0
    assert result.ready_to_first_token_seconds == 10.0


def test_a_raising_disk_store_cold_serves_instead_of_surfacing():
    # The disk store normalizes its own faults, and this caller-level
    # backstop makes the docstring contract unconditional: any disk error,
    # typed or not, is a cold miss, never a request failure.
    full = list(range(1, 401))
    logged = []

    class _RaisingDiskStore:
        stride = None

        def _log(self, line):
            logged.append(line)

        def restore(self, *args, **kwargs):
            raise RuntimeError("index exploded mid-request")

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return GenerationResult(text="ok", prompt_cache=kwargs["prompt_cache"])

    gen = PrefixCacheGenerator(
        "MODEL", _FakeTokenizer(full), {"artifact_id": "pkg"}, _FakeStore(),
        make_prompt_cache_fn=lambda model: ["fresh"],
        generate_fn=fake_generate,
        disk_store=_RaisingDiskStore(),
    )
    result = gen(
        "rendered prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="render-id",
    )
    assert result.text == "ok"
    assert any("cold serving" in line for line in logged)


def test_a_write_disabled_store_builds_no_writer():
    # After a confirmed index fault the store flags writes_disabled; the
    # writer precheck must then skip serialization entirely on every
    # request instead of paying one doomed payload write per request.
    full = list(range(1, 401))
    seen = {}

    class _DisabledDiskStore(_FakeDiskStore):
        writes_disabled = True

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen["has_plan"] = "prefill_plan" in kwargs
        return GenerationResult(text="ok", prompt_cache=kwargs["prompt_cache"])

    gen = PrefixCacheGenerator(
        "MODEL", _FakeTokenizer(full), {"artifact_id": "pkg"}, _FakeStore(),
        make_prompt_cache_fn=lambda model: ["fresh"],
        generate_fn=fake_generate,
        disk_store=_DisabledDiskStore(stride=256),
    )
    result = gen(
        "rendered prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="render-id",
    )
    assert result.text == "ok"
    assert seen["has_plan"] is False


def test_frontier_writer_kwargs_carry_a_variable_step_plan_on_an_unaligned_hit():
    full = list(range(1, 701))
    tok = _FakeTokenizer(full)
    store = _FakeStore(fetched_cache=["cached"], rest=full[100:])
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update(kwargs)
        return GenerationResult(
            text="ok",
            generated_token_ids=(999,),
            prompt_cache=kwargs["prompt_cache"],
        )

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
        disk_store=_FakeDiskStore(stride=256),
    )
    result = gen("rendered prompt",
                 kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
                 effective_rendering_id="render-id")

    assert result.cache_event == "hit"
    # Restored at 100 with frontiers 256 and 512 inside the 699-token prefill
    # span: one 156-token lander closes the unaligned gap, everything else is
    # a full chunk, and the tail runs at the default uniform step.
    assert seen["prefill_plan"] == [156, 256]
    assert seen["prefill_step_size"] == 2048
    assert callable(seen["prompt_progress_callback"])


def test_frontier_writer_kwargs_omit_the_plan_when_no_frontier_is_in_prefill_reach():
    # The only frontier sits exactly at the full prompt length, which the
    # prefill loop never reaches (the final token feeds the first decode
    # step), so no plan is built and the uniform step serves the whole
    # suffix. The writer still hooks progress for its accounting.
    full = list(range(1, 257))
    tok = _FakeTokenizer(full)
    store = _FakeStore(fetched_cache=["cached"], rest=full[100:])
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update(kwargs)
        return GenerationResult(
            text="ok",
            generated_token_ids=(999,),
            prompt_cache=kwargs["prompt_cache"],
        )

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
        disk_store=_FakeDiskStore(stride=256),
    )
    gen("rendered prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="render-id")

    assert "prefill_plan" not in seen
    assert seen["prefill_step_size"] == 2048
    assert callable(seen["prompt_progress_callback"])


def test_raw_prefix_cache_exact_empty_rest_falls_back_to_full_prompt():
    tok = _FakeTokenizer([1, 2, 3])
    store = _FakeStore(fetched_cache=["cached"], rest=[])
    make_cache = _FakeMakeCache()
    seen = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen["prompt"] = list(prompt)
        seen["cached_tokens"] = kwargs["cached_tokens"]
        return GenerationResult(
            text="again",
            prompt_tokens=len(prompt),
            completion_tokens=1,
            generated_token_ids=(4,),
            prompt_cache=kwargs["prompt_cache"],
        )

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=make_cache,
        generate_fn=fake_generate,
    )
    result = gen("rendered prompt", kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
                 effective_rendering_id="render-id")

    assert seen["prompt"] == [1, 2, 3]
    assert seen["cached_tokens"] == 0
    assert result.cache_event == "exact_fallback"
    assert make_cache.calls, "empty suffix is deliberately treated as a fresh miss"


def test_stream_ready_runs_before_cache_fetch_and_response_observes_decode():
    events = []
    tok = _FakeTokenizer([1, 2, 3])

    class Store(_FakeStore):
        def fetch_nearest_cache(self, model_key, tokens):
            events.append("fetch")
            return super().fetch_nearest_cache(model_key, tokens)

        def insert_cache(self, model_key, tokens, prompt_cache):
            events.append("insert")
            super().insert_cache(model_key, tokens, prompt_cache)

    def fake_generate(model, tokenizer, prompt, **kwargs):
        events.append("generate")
        response = type("Response", (), {"text": "piece"})()
        kwargs["response_callback"](1, response)
        return GenerationResult(
            text="piece",
            generated_token_ids=(4,),
            prompt_cache=kwargs["prompt_cache"],
        )

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, Store(),
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
    )
    seen = []
    gen(
        "rendered prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="render-id",
        ready_callback=lambda: events.append("ready"),
        response_callback=lambda step, response: seen.append((step, response.text)),
    )

    assert events == ["ready", "fetch", "generate", "insert"]
    assert seen == [(1, "piece")]


def test_stream_progress_runs_after_frontier_progress():
    events = []
    tok = _FakeTokenizer([1, 2, 3])

    def fake_generate(model, tokenizer, prompt, **kwargs):
        kwargs["prompt_progress_callback"](2, 3)
        return GenerationResult(
            text="ok",
            generated_token_ids=(4,),
            prompt_cache=kwargs["prompt_cache"],
        )

    class Writer:
        written = []

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, _FakeStore(),
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
    )
    gen._frontier_writer = lambda *args, **kwargs: (
        Writer(),
        {"prompt_progress_callback": lambda processed, total: events.append(
            ("frontier", processed, total))},
    )
    gen(
        "rendered prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="render-id",
        progress_callback=lambda processed, total: events.append(
            ("stream", processed, total)),
    )

    assert events == [("frontier", 2, 3), ("stream", 2, 3)]


def test_stream_callback_failure_does_not_publish_partial_cache():
    tok = _FakeTokenizer([1, 2, 3])
    store = _FakeStore()

    def fake_generate(model, tokenizer, prompt, **kwargs):
        response = type("Response", (), {"text": "piece"})()
        kwargs["response_callback"](1, response)
        raise ConnectionError("client disconnected")

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
    )
    with pytest.raises(ConnectionError, match="disconnected"):
        gen(
            "rendered prompt",
            kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
            effective_rendering_id="render-id",
            response_callback=lambda step, response: None,
        )
    assert store.insert_calls == []


def test_terminal_stream_stop_publishes_only_the_committed_token_frontier():
    tok = _FakeTokenizer([1, 2, 3])
    store = _FakeStore()

    def fake_generate(model, tokenizer, prompt, **kwargs):
        del model, tokenizer, prompt
        generated = []
        for step, (text, token) in enumerate(
            (("call", 4), (" close", 5), (" late", 6)), 1
        ):
            generated.append(token)
            response = SimpleNamespace(text=text, token=token)
            kwargs["response_callback"](step, response)
            if kwargs["response_stop_callback"]():
                break
        return GenerationResult(
            text="call close",
            finish_reason="stop",
            completion_tokens=len(generated),
            generated_token_ids=tuple(generated),
            prompt_cache=kwargs["prompt_cache"],
        )

    terminal = False

    def capture(_step, response):
        nonlocal terminal
        terminal = response.text == " close"

    gen = PrefixCacheGenerator(
        "MODEL", tok, {"artifact_id": "pkg"}, store,
        make_prompt_cache_fn=lambda model: ["new"],
        generate_fn=fake_generate,
    )
    result = gen(
        "rendered prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="render-id",
        response_callback=capture,
        response_stop_callback=lambda: terminal,
    )

    assert result.generated_token_ids == (4, 5)
    assert result.completion_tokens == 2
    assert store.insert_calls[0][1] == [1, 2, 3, 4, 5]


def test_streaming_preserves_frontier_plan_writes_tokens_and_cache_key():
    full = list(range(1, 701))

    class OffsetCache:
        def __init__(self, offset=100):
            self.offset = offset
            self.state = ["state"]
            self.meta_state = "meta"

    class DiskStore:
        stride = 256

        def __init__(self):
            self.writes = []

        def has_entry(self, scope, tokens):
            return False

        def write_checkpoint(self, scope, tokens, **kwargs):
            self.writes.append((len(tokens), tuple(tokens)))
            return object()

        def stats(self):
            return {"enabled": True}

    def run(streaming):
        cache = OffsetCache()
        memory = _FakeStore(fetched_cache=[cache], rest=full[100:])
        disk = DiskStore()
        record = {"deltas": [], "ready": 0}

        def fake_generate(model, tokenizer, prompt, **kwargs):
            record["plan"] = list(kwargs["prefill_plan"])
            processed = 0
            for size in kwargs["prefill_plan"]:
                processed += size
                cache.offset = kwargs["cached_tokens"] + processed
                kwargs["prompt_progress_callback"](processed, len(full))
            response = type("Response", (), {"text": "ok"})()
            if "response_callback" in kwargs:
                kwargs["response_callback"](1, response)
            return GenerationResult(
                text="ok",
                generated_token_ids=(901, 902),
                prompt_cache=kwargs["prompt_cache"],
            )

        gen = PrefixCacheGenerator(
            "MODEL", _FakeTokenizer(full), {"artifact_id": "pkg"}, memory,
            make_prompt_cache_fn=lambda model: [OffsetCache(0)],
            generate_fn=fake_generate,
            disk_store=disk,
        )
        stream_kwargs = {}
        if streaming:
            stream_kwargs = {
                "ready_callback": lambda: record.update(ready=1),
                "progress_callback": lambda processed, total: record.setdefault(
                    "progress", []).append((processed, total)),
                "response_callback": lambda step, response: record["deltas"].append(
                    (step, response.text)),
            }
        result = gen(
            "rendered prompt",
            kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
            effective_rendering_id="render-id",
            **stream_kwargs,
        )
        return {
            "plan": record["plan"],
            "writes": disk.writes,
            "insert_key": memory.insert_calls[0][1],
            "generated": result.generated_token_ids,
            "checkpoints": result.disk_checkpoints_written,
            "ready": record["ready"],
            "deltas": record["deltas"],
        }

    non_streaming = run(False)
    streaming = run(True)
    assert streaming["ready"] == 1
    assert streaming["deltas"] == [(1, "ok")]
    for field in ("plan", "writes", "insert_key", "generated", "checkpoints"):
        assert streaming[field] == non_streaming[field]
    assert streaming["plan"] == [156, 256]
    assert [length for length, _tokens in streaming["writes"]] == [256, 512]
