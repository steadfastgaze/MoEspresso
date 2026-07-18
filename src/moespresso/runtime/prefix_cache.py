"""Raw in-memory prefix reuse.

This module owns MoEspresso's cache policy glue. It stays import-light: MLX cache classes
are only imported by the factory helpers used from the serve edge.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from moespresso.runtime.generation import ContextLimitError, GenerationResult
from moespresso.runtime.kv_policy import (
    KVPolicy,
    KVPolicyError,
    LIVE_KV_Q8,
    LIVE_KV_RAW,
    validate_runtime_policy,
)
from moespresso.runtime.deepseek_v4.cache import DEEPSEEK_V4_CACHE_KIND
from moespresso.runtime.serve import generate_with_metadata


def encode_rendered_prompt(tokenizer, rendered_prompt: str) -> list[int]:
    """Encode a rendered prompt with the same string rules MLX uses.

    mlx_lm.stream_generate adds special tokens when the tokenizer has no BOS token or the
    prompt does not already start with that BOS string. Cache keys must match that token
    stream exactly, so MoEspresso uses the same rule before slicing suffix tokens.
    """
    bos = getattr(tokenizer, "bos_token", None)
    add_special_tokens = bos is None or not rendered_prompt.startswith(bos)
    return list(tokenizer.encode(rendered_prompt, add_special_tokens=add_special_tokens))


def cache_payload_kind(manifest: dict) -> str:
    """Prompt-cache payload kind stored in the prefix trie."""
    architecture = manifest.get("architecture") or {}
    cache_policy = architecture.get("cache_policy") or {}
    return cache_policy.get("kind") or "mlx_prompt_cache"


def declared_context_limit(manifest: dict) -> int | None:
    """The model's declared maximum sequence length, in tokens.

    Read from the package's embedded model config
    (``architecture.config``): ``max_position_embeddings`` at the top
    level, or under ``text_config`` for wrapped multimodal families. For
    position-scaled families the field already holds the scaled ceiling
    (DeepSeek-V4 Flash declares 1048576 via YaRN factor 16 over an
    original 65536). Returns None when the package declares nothing;
    serving then applies no limit, the pre-existing behavior.
    """
    architecture = manifest.get("architecture") or {}
    config = architecture.get("config") or {}
    for source in (config, config.get("text_config") or {}):
        value = source.get("max_position_embeddings")
        if value is not None:
            try:
                limit = int(value)
            except (TypeError, ValueError):
                return None
            return limit if limit > 0 else None
    return None


def supported_live_kv_formats(manifest: dict) -> list[str]:
    """Live KV formats allowed for this package family."""
    if cache_payload_kind(manifest) == DEEPSEEK_V4_CACHE_KIND:
        return [LIVE_KV_RAW]
    return [LIVE_KV_RAW, LIVE_KV_Q8]


def validate_manifest_cache_policy(manifest: dict, policy: KVPolicy) -> None:
    """Validate the global KV policy plus package-specific cache constraints."""
    validate_runtime_policy(policy)
    if (
        cache_payload_kind(manifest) == DEEPSEEK_V4_CACHE_KIND
        and policy.live_kv_format != LIVE_KV_RAW
    ):
        raise KVPolicyError(
            "DeepSeek V4 composite cache does not support generic mlx_affine_q8 kv_bits"
        )


def cache_model_key(manifest: dict, effective_rendering_id: str, policy: KVPolicy) -> tuple:
    """Stable model bucket for the prefix trie.

    Token ids alone are not enough: the same tokens under another package, rendering
    policy, cache payload kind, or KV format must not reuse a cache object.
    """
    return (
        manifest.get("artifact_id"),
        effective_rendering_id,
        policy.live_kv_format,
        policy.kv_group_size,
        policy.quantized_kv_start,
        cache_payload_kind(manifest),
    )


def _common_prefix_len(a: tuple, b: tuple) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class _StoreEntry:
    prompt_cache: Any
    nbytes: int


class PromptCacheStore:
    """In-memory prompt-cache store holding one live timeline per chain.

    Interface-compatible with the mlx-lm ``LRUPromptCache`` where the serve
    layer touches it (``fetch_nearest_cache``, ``insert_cache``, ``len``,
    ``nbytes``), with three policy changes, each from a measured serving
    defect on rotating-window caches that report untrimmable:

    - Fetch moves the matched entry out of the store and returns the stored
      cache object itself instead of a deep copy, which measured roughly
      34 KB per live token per request. Move semantics also keep the store
      from aliasing live mutable state: the mutated cache is published back
      only by ``insert_cache``, after generation completes, under the serve
      lock, so a concurrent reader can never observe a cache mid-mutation
      under a stale key. A request that fails mid-generation loses the
      chain's memory entry and the next request falls back to the disk
      frontier restore or a cold miss, never to a corrupt entry.
    - Insert pops strict-prefix entries of the inserted key regardless of
      trimmability, so an append-only session holds exactly one entry (the
      chain top) instead of one full snapshot per request. The cost: a
      branch from an earlier prefix loses its in-memory hit and falls to
      the disk frontier path, which restores exactly and costs about a
      minute end to end at 20k to 50k context, against gigabytes of
      retained snapshots per session without the popping (measured 3.30 GB
      at 89.8k retained tokens under the ten-entry cap).
    - ``max_bytes`` evicts least-recently-inserted entries, so resident
      cache memory is boundable for any family.

    A stored key that extends the request still serves a trimmed deep copy
    when every cache in the entry is trimmable (the stock mlx-lm branch
    behavior); untrimmable caches fall through to the shorter entry or a
    miss.
    """

    def __init__(
        self,
        max_size: int = 10,
        max_bytes: int | None = None,
        *,
        can_trim_fn: Callable | None = None,
        trim_fn: Callable | None = None,
    ):
        if max_size < 1:
            raise ValueError("prompt cache max_size must be positive")
        if max_bytes is not None and max_bytes < 1:
            raise ValueError("prompt cache max_bytes must be positive")
        self.max_size = int(max_size)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        # Insertion-ordered; recency equals insert order because a fetch
        # moves the entry out and generation reinserts the extended key.
        self._entries: dict[tuple, _StoreEntry] = {}
        self._n_bytes = 0
        self._can_trim_fn = can_trim_fn
        self._trim_fn = trim_fn

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        return self._n_bytes

    def _can_trim(self, prompt_cache) -> bool:
        if self._can_trim_fn is None:
            from mlx_lm.models.cache import can_trim_prompt_cache
            self._can_trim_fn = can_trim_prompt_cache
        return bool(self._can_trim_fn(prompt_cache))

    def _trim(self, prompt_cache, num_tokens: int) -> None:
        if self._trim_fn is None:
            from mlx_lm.models.cache import trim_prompt_cache
            self._trim_fn = trim_prompt_cache
        self._trim_fn(prompt_cache, num_tokens)

    def _pop_entry(self, store_key: tuple) -> _StoreEntry:
        entry = self._entries.pop(store_key)
        self._n_bytes -= entry.nbytes
        return entry

    def _evict_oldest(self) -> None:
        self._pop_entry(next(iter(self._entries)))

    def fetch_nearest_cache(self, model: Any, tokens: list) -> tuple[Any, list]:
        """The nearest reusable cache and the suffix left to prefill.

        An exact or strict-prefix match is moved out of the store and
        returned as-is; the caller mutates it in place and reinserts it
        under the extended key after generation.
        """
        tokens = list(tokens)
        target = tuple(tokens)
        exact = None
        shorter = None
        for entry_model, key in self._entries:
            if entry_model != model:
                continue
            if key == target:
                exact = key
            elif len(key) < len(target) and target[:len(key)] == key:
                if shorter is None or len(key) > len(shorter):
                    shorter = key
        if exact is not None:
            return self._pop_entry((model, exact)).prompt_cache, []

        short_len = len(shorter) if shorter is not None else 0
        best_longer = None
        best_common = short_len
        for entry_model, key in self._entries:
            if entry_model != model or len(key) <= len(target):
                continue
            common = _common_prefix_len(key, target)
            if common > best_common:
                best_longer, best_common = key, common
        if best_longer is not None:
            entry = self._entries[(model, best_longer)]
            if self._can_trim(entry.prompt_cache):
                cache = copy.deepcopy(entry.prompt_cache)
                prefix = min(len(target) - 1, best_common)
                self._trim(cache, len(best_longer) - prefix)
                return cache, tokens[prefix:]

        if shorter is not None:
            return self._pop_entry((model, shorter)).prompt_cache, tokens[short_len:]
        return None, tokens

    def insert_cache(self, model: Any, tokens: list, prompt_cache: Any) -> None:
        """Publish a generated-through cache under its full token key.

        Strict-prefix entries of the key are popped regardless of
        trimmability: the longer chain supersedes them, and a later branch
        from one of those prefixes is served by the disk frontier path.
        """
        key = tuple(tokens)
        entry = _StoreEntry(
            prompt_cache, sum(int(c.nbytes) for c in prompt_cache))
        previous = self._entries.pop((model, key), None)
        if previous is not None:
            self._n_bytes -= previous.nbytes
        self._entries[(model, key)] = entry
        self._n_bytes += entry.nbytes
        for entry_model, existing in list(self._entries):
            if entry_model != model or len(existing) >= len(key):
                continue
            if key[:len(existing)] == existing:
                self._pop_entry((model, existing))
        while len(self._entries) > self.max_size:
            self._evict_oldest()
        if self.max_bytes is not None:
            while self._n_bytes > self.max_bytes and self._entries:
                self._evict_oldest()


def make_prompt_cache_store(max_size: int = 10, max_bytes: int | None = None):
    return PromptCacheStore(max_size=max_size, max_bytes=max_bytes)


def make_mlx_prompt_cache(model):
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(model)


def _store_entries(cache_store) -> int | None:
    try:
        return len(cache_store)
    except TypeError:
        if hasattr(cache_store, "insert_calls"):
            return len(cache_store.insert_calls)
        return None


def _store_nbytes(cache_store) -> int | None:
    value = getattr(cache_store, "nbytes", None)
    if callable(value):
        value = value()
    return int(value) if value is not None else None


@dataclass
class PrefixCacheGenerator:
    """Generate through an MLX prompt cache when a token prefix is reusable."""

    model: Any
    tokenizer: Any
    manifest: dict
    cache_store: Any
    make_prompt_cache_fn: Callable = make_mlx_prompt_cache
    generate_fn: Callable = generate_with_metadata
    after_generate_fn: Callable | None = None
    disk_store: Any = None
    disk_registry: Any = None
    # Declared maximum sequence length in tokens (declared_context_limit);
    # None applies no limit.
    context_limit: int | None = None
    _cache_class_names: tuple[str, ...] | None = None
    _closed: bool = False

    def _cache_classes(self) -> tuple[str, ...]:
        """The live cache-class layout, built once and reused.

        The disk scope keys on this list, so a checkpoint written for one layout
        never restores into another. Building one cache reads the names; the
        instance is discarded.
        """
        if self._cache_class_names is None:
            caches = self.make_prompt_cache_fn(self.model)
            self._cache_class_names = tuple(type(c).__name__ for c in caches)
        return self._cache_class_names

    def _consult_disk(self, model_key: tuple, full_tokens: list[int]):
        """Restore the longest valid disk checkpoint, or None on miss or fault.

        A cache problem must never surface in a request. Every disk error is
        caught here and treated as a cold miss; the disk store already quarantines
        an invalid entry before raising, so the fault cannot recur silently.
        """
        from moespresso.runtime.disk_kv import (
            DiskKVError,
            build_cache_scope,
            default_cache_registry,
        )

        scope = build_cache_scope(model_key, self._cache_classes())
        registry = self.disk_registry or default_cache_registry()
        try:
            return self.disk_store.restore(
                scope,
                list(full_tokens),
                make_cache_fn=lambda: self.make_prompt_cache_fn(self.model),
                registry=registry,
            )
        except DiskKVError:
            return None

    def _frontier_writer(
        self, model_key, full_tokens, cached_tokens, prompt_cache,
        session_cache_key=None,
    ):
        """Build the frontier writer and generate kwargs for this call, or none.

        Zero overhead when the store is off, the store carries no stride (the
        read-only shape), or the request crosses no unwritten frontier. The cheap
        precheck reads only the token count and the stride before any tracker is
        built. When a frontier will be crossed the writer's variable-step chunk
        plan and progress callback are returned so a prefill chunk ends exactly
        on every frontier while all other chunks run at the full default step.
        """
        store = self.disk_store
        stride = getattr(store, "stride", None) if store is not None else None
        if store is None or stride is None:
            return None, {}
        # Cheap precheck: the smallest frontier this call could build is the first
        # stride multiple strictly above the restored prefix. If that exceeds the
        # token count, no frontier is crossed and no tracker is built.
        first_frontier = ((cached_tokens // stride) + 1) * stride
        if first_frontier > len(full_tokens):
            return None, {}

        from moespresso.runtime.disk_kv import (
            FrontierTracker,
            FrontierWriter,
            build_cache_scope,
        )

        scope = build_cache_scope(model_key, self._cache_classes())
        tracker = FrontierTracker(
            stride=stride,
            restored_prefix=cached_tokens,
            full_tokens=list(full_tokens),
            scope=scope,
            already_written=store.has_entry,
        )
        # Nothing eligible after the dedupe read: skip the writer entirely.
        if tracker.next_frontier_above(cached_tokens) is None:
            return None, {}
        writer = FrontierWriter(
            store, tracker=tracker, caches=prompt_cache,
            session_cache_key=session_cache_key)

        from moespresso.runtime.serve import _model_prefill_step_size

        default_step = _model_prefill_step_size(self.model, self.tokenizer, full_tokens)
        if default_step is None:
            default_step = 2048
        generate_kwargs = {
            "prefill_step_size": default_step,
            "prompt_progress_callback": writer.on_prompt_progress,
        }
        plan = writer.prefill_chunk_plan(default_step)
        if plan:
            generate_kwargs["prefill_plan"] = plan
        return writer, generate_kwargs

    def cache_stats(self) -> dict:
        """Small HTTP-facing snapshot of the resident prompt cache.

        When the disk store is enabled a ``disk`` block reports its counters since
        startup; when it is off the block is a single ``enabled: false`` marker.
        """
        supported = supported_live_kv_formats(self.manifest)
        stats = {
            "default_live_kv_format": supported[-1],
            "supported_live_kv_formats": supported,
            "entries": _store_entries(self.cache_store),
            "bytes": _store_nbytes(self.cache_store),
        }
        if self.disk_store is not None:
            stats["disk"] = self.disk_store.stats()
        else:
            stats["disk"] = {"enabled": False}
        return stats

    def close(self) -> None:
        self._closed = True

    def __call__(
        self,
        rendered_prompt: str,
        *,
        kv_policy: KVPolicy,
        effective_rendering_id: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float | None = None,
        session_cache_key: str | None = None,
        ready_callback: Callable[[], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        response_callback: Callable[[int, object], None] | None = None,
    ) -> GenerationResult:
        validate_manifest_cache_policy(self.manifest, kv_policy)

        # Sampling knobs are generation-only: they are forwarded to the
        # generation seam and never enter the cache model key, the disk
        # scope, or any token accounting, so prefix reuse is unaffected by a
        # client varying them turn over turn. Default-off values are not
        # forwarded, which keeps a request without them byte-identical to
        # the pre-existing generate call shape.
        sampling_kwargs = {}
        if top_k:
            sampling_kwargs["top_k"] = int(top_k)
        if min_p:
            sampling_kwargs["min_p"] = float(min_p)
        if presence_penalty is not None:
            sampling_kwargs["presence_penalty"] = float(presence_penalty)

        full_tokens = encode_rendered_prompt(self.tokenizer, rendered_prompt)

        # Refuse an over-limit request before any cache access: the store
        # hands entries out by move, so a refusal after the fetch would cost
        # the session its chain entry. Prompt plus requested completion
        # budget must fit the declared limit; a request exactly at the limit
        # passes (the sequence occupies positions 0 through limit - 1).
        if (self.context_limit is not None
                and len(full_tokens) + int(max_tokens) > self.context_limit):
            raise ContextLimitError(
                limit=self.context_limit,
                prompt_tokens=len(full_tokens),
                max_tokens=int(max_tokens),
            )

        # A streaming transport may commit its response only after all request
        # and context validation has passed.  This hook intentionally runs
        # before fetch_nearest_cache because the store hands entries out by
        # move; a failed socket write must not cost the session its chain.
        if ready_callback is not None:
            ready_callback()

        model_key = cache_model_key(self.manifest, effective_rendering_id, kv_policy)
        prompt_cache, suffix_tokens = self.cache_store.fetch_nearest_cache(
            model_key, full_tokens)
        suffix_tokens = list(suffix_tokens)
        cached_tokens = len(full_tokens) - len(suffix_tokens)
        cache_event = "hit" if cached_tokens > 0 else "miss"

        # On an in-memory miss (and only then), consult the disk store for the
        # longest exact valid checkpoint. A valid disk hit yields a live cache and
        # a suffix that generation consumes exactly like a memory hit. Any disk
        # problem returns the engine to cold serving without touching the request.
        if self.disk_store is not None and cache_event == "miss":
            disk_hit = self._consult_disk(model_key, full_tokens)
            if disk_hit is not None:
                prompt_cache = disk_hit.prompt_cache
                suffix_tokens = list(disk_hit.suffix_tokens)
                cached_tokens = disk_hit.cached_tokens
                cache_event = "disk_hit"

        # MLX generation needs at least one prompt token to compute the next token. If the
        # trie returns an exact whole-prompt cache (empty suffix), build a fresh cache so
        # there is a token to feed. Follow-up chat turns still hit normally because they
        # add user suffix tokens.
        if prompt_cache is None or not suffix_tokens:
            cache_event = "exact_fallback" if prompt_cache is not None else "miss"
            prompt_cache = self.make_prompt_cache_fn(self.model)
            suffix_tokens = list(full_tokens)
            cached_tokens = 0

        # Build the frontier writer only when the disk store is enabled, a stride is
        # configured, and this request actually crosses an unwritten frontier during
        # prefill. When any of those is false no tracker is constructed and no
        # callback is intercepted, so an off store or a short prompt adds nothing.
        writer, generate_kwargs = self._frontier_writer(
            model_key, full_tokens, cached_tokens, prompt_cache,
            session_cache_key=session_cache_key)

        # The disk writer remains first at every prompt-progress position.  A
        # passive transport observer sees the same values only after the writer
        # has evaluated the exact live-cache frontier.
        frontier_progress = generate_kwargs.pop("prompt_progress_callback", None)
        if frontier_progress is not None and progress_callback is not None:
            def combined_progress(processed: int, total: int) -> None:
                frontier_progress(processed, total)
                progress_callback(processed, total)
            generate_kwargs["prompt_progress_callback"] = combined_progress
        elif frontier_progress is not None:
            generate_kwargs["prompt_progress_callback"] = frontier_progress
        elif progress_callback is not None:
            generate_kwargs["prompt_progress_callback"] = progress_callback
        if response_callback is not None:
            generate_kwargs["response_callback"] = response_callback

        result = self.generate_fn(
            self.model,
            self.tokenizer,
            suffix_tokens,
            prompt_cache=prompt_cache,
            cached_tokens=cached_tokens,
            kv_policy=kv_policy,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **sampling_kwargs,
            **generate_kwargs,
        )
        if writer is not None and writer.written:
            result.disk_checkpoints_written = len(writer.written)
            result.disk_checkpoint_write_seconds = tuple(writer.write_seconds)

        if result.prompt_cache is not None:
            cache_key = list(full_tokens) + list(result.generated_token_ids)
            if cache_key:
                self.cache_store.insert_cache(model_key, cache_key, result.prompt_cache)
        if self.after_generate_fn is not None:
            self.after_generate_fn(self.model)
        result.cache_event = cache_event
        result.cache_entries = _store_entries(self.cache_store)
        result.cache_bytes = _store_nbytes(self.cache_store)
        return result
