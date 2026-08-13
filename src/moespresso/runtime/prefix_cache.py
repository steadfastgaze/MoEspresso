"""Memory and disk prefix reuse with producer-specific cache state.

This module owns MoEspresso's cache policy glue, memory producer rails, durable
target restores, and optional DSpark companions. It stays import-light: MLX
cache classes are imported only by factory helpers used from the serve edge.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

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


DEFAULT_CONTEXT_LIMIT = 128 * 1024

# The existing prompt-cache API always reads and writes this producer rail.
# Math-affecting generation paths use another hashable identity and therefore
# cannot reuse a cache produced on the plain path.
PLAIN_CACHE_RAIL = ("plain",)


def context_limit_warning_lines(context_limit: int) -> tuple[str, ...]:
    """Prominent product warning for a served context below the 128K target."""
    limit = int(context_limit)
    if limit >= DEFAULT_CONTEXT_LIMIT:
        return ()
    return (
        "WARNING: served context is below the 128K usability target.",
        f"WARNING: context_limit={limit} usability_target={DEFAULT_CONTEXT_LIMIT}.",
        "WARNING: usability is substantially reduced for long conversations "
        "and agentic work.",
    )


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
    original 65536). Returns None when the package declares nothing.
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


def effective_context_limit(
    manifest: dict,
    requested: int | None = None,
    runtime_default: int | None = None,
) -> int:
    """Resolve the served limit without changing the package contract.

    Packages retain their architecture limit. Serving defaults to 128K or the
    package limit, whichever is smaller. A runtime may lower that default when
    its exact memory contract cannot admit the default context alongside the
    minimum execution state. An explicit operator value remains authoritative
    and may select any positive limit up to the package maximum.
    """
    declared = declared_context_limit(manifest)
    if requested is None:
        default_limit = (
            min(DEFAULT_CONTEXT_LIMIT, declared)
            if declared
            else DEFAULT_CONTEXT_LIMIT
        )
        if runtime_default is None:
            return default_limit
        runtime_default = int(runtime_default)
        if runtime_default < 1:
            raise ValueError("runtime context default must be >= 1")
        return min(default_limit, runtime_default)

    limit = int(requested)
    if limit < 1:
        raise ValueError("--max-context-tokens must be >= 1")
    if declared is not None and limit > declared:
        raise ValueError(
            f"--max-context-tokens {limit} exceeds the package context limit "
            f"of {declared} tokens"
        )
    return limit


def validate_context_span(
    *,
    limit: int | None,
    prompt_tokens: int,
    max_tokens: int,
) -> None:
    """Refuse a prompt plus completion budget beyond the served limit."""
    if limit is not None and int(prompt_tokens) + int(max_tokens) > int(limit):
        raise ContextLimitError(
            limit=int(limit),
            prompt_tokens=int(prompt_tokens),
            max_tokens=int(max_tokens),
        )


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
    companion: Any
    nbytes: int


@dataclass(frozen=True)
class CacheProbe:
    """Read-only result of selecting one producer rail's reusable depth."""

    kind: Literal["miss", "exact", "shorter", "trim"]
    cached_tokens: int
    has_companion: bool


@dataclass(frozen=True)
class _CacheSelection:
    probe: CacheProbe
    store_key: tuple | None = None
    trim_tokens: int = 0


def _companion_nbytes(companion: Any, declared_nbytes: int | None) -> int:
    """Resolve the resident bytes of an opaque cache companion."""
    if companion is None:
        if declared_nbytes not in (None, 0):
            raise ValueError("companion_nbytes requires a companion payload")
        return 0

    value = declared_nbytes
    if value is None:
        value = getattr(companion, "nbytes", None)
        if value is None:
            raise TypeError(
                "cache companion must expose nbytes or declare companion_nbytes"
            )
        if callable(value):
            value = value()
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("companion_nbytes must be an integer")
    if value < 0:
        raise ValueError("companion_nbytes must be non-negative")
    return value


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

    Entries also carry a producer rail. The legacy API uses the plain rail;
    generation paths with different numerical histories use distinct hashable
    identities. Lookup, move, strict-prefix supersession, and trim candidate
    selection never cross rails. The entry and byte limits remain global hard
    bounds across all rails.

    An enhanced API can move an opaque companion together with its target
    cache. Companion tensor bytes count toward ``max_bytes`` through either a
    payload ``nbytes`` attribute or an explicit declaration. A stored key that
    extends the request still serves a trimmed deep copy
    when every cache in the entry is trimmable (the stock mlx-lm branch
    behavior). Its companion stays with the untrimmed stored entry because the
    generic store cannot trim opaque state. Untrimmable caches fall through to
    the shorter entry or a miss.
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
        prompt_cache, suffix_tokens, _ = self.fetch_nearest_cache_with_companion(
            model,
            tokens,
        )
        return prompt_cache, suffix_tokens

    def _select_nearest_cache(
        self,
        model: Any,
        tokens: list,
        *,
        producer_rail: Any,
    ) -> _CacheSelection:
        """Select one rail's fetch result without moving or copying it."""
        try:
            hash(producer_rail)
        except TypeError as e:
            raise TypeError("cache producer rail must be hashable") from e

        target = tuple(tokens)
        exact = None
        shorter = None
        for entry_rail, entry_model, key in self._entries:
            if entry_rail != producer_rail or entry_model != model:
                continue
            if key == target:
                exact = key
            elif len(key) < len(target) and target[:len(key)] == key:
                if shorter is None or len(key) > len(shorter):
                    shorter = key
        if exact is not None:
            store_key = (producer_rail, model, exact)
            entry = self._entries[store_key]
            return _CacheSelection(
                CacheProbe("exact", len(target), entry.companion is not None),
                store_key,
            )

        short_len = len(shorter) if shorter is not None else 0
        best_longer = None
        best_common = short_len
        for entry_rail, entry_model, key in self._entries:
            if (
                entry_rail != producer_rail
                or entry_model != model
                or len(key) <= len(target)
            ):
                continue
            common = _common_prefix_len(key, target)
            if common > best_common:
                best_longer, best_common = key, common
        if best_longer is not None:
            store_key = (producer_rail, model, best_longer)
            entry = self._entries[store_key]
            if self._can_trim(entry.prompt_cache):
                prefix = min(len(target) - 1, best_common)
                return _CacheSelection(
                    CacheProbe("trim", prefix, False),
                    store_key,
                    len(best_longer) - prefix,
                )

        if shorter is not None:
            store_key = (producer_rail, model, shorter)
            entry = self._entries[store_key]
            return _CacheSelection(
                CacheProbe("shorter", short_len, entry.companion is not None),
                store_key,
            )
        return _CacheSelection(CacheProbe("miss", 0, False))

    def probe_nearest_cache(
        self,
        model: Any,
        tokens: list,
        *,
        producer_rail: Any = PLAIN_CACHE_RAIL,
    ) -> CacheProbe:
        """Report one rail's reusable depth without changing store residency."""
        return self._select_nearest_cache(
            model,
            list(tokens),
            producer_rail=producer_rail,
        ).probe

    def fetch_nearest_cache_with_companion(
        self,
        model: Any,
        tokens: list,
        *,
        producer_rail: Any = PLAIN_CACHE_RAIL,
    ) -> tuple[Any, list, Any]:
        """Move the nearest target cache and its companion from one rail.

        Exact and shorter-prefix hits move the stored pair. A longer-prefix
        trim hit returns a target-cache copy and no companion, leaving the
        original pair in the store at its unchanged frontier.
        """
        tokens = list(tokens)
        selection = self._select_nearest_cache(
            model,
            tokens,
            producer_rail=producer_rail,
        )
        if selection.probe.kind == "miss":
            return None, tokens, None
        if selection.probe.kind == "trim":
            entry = self._entries[selection.store_key]
            cache = copy.deepcopy(entry.prompt_cache)
            self._trim(cache, selection.trim_tokens)
            return cache, tokens[selection.probe.cached_tokens:], None
        entry = self._pop_entry(selection.store_key)
        return (
            entry.prompt_cache,
            tokens[selection.probe.cached_tokens:],
            entry.companion,
        )

    def insert_cache(self, model: Any, tokens: list, prompt_cache: Any) -> None:
        """Publish a generated-through cache under its full token key.

        Strict-prefix entries of the key are popped regardless of
        trimmability: the longer chain supersedes them, and a later branch
        from one of those prefixes is served by the disk frontier path.
        """
        self.insert_cache_with_companion(model, tokens, prompt_cache)

    def insert_cache_with_companion(
        self,
        model: Any,
        tokens: list,
        prompt_cache: Any,
        *,
        producer_rail: Any = PLAIN_CACHE_RAIL,
        companion: Any = None,
        companion_nbytes: int | None = None,
    ) -> bool:
        """Publish a pair and report whether it survives global eviction."""
        try:
            hash(producer_rail)
        except TypeError as e:
            raise TypeError("cache producer rail must be hashable") from e
        if producer_rail == PLAIN_CACHE_RAIL and companion is not None:
            raise ValueError("the plain cache rail cannot carry a companion")

        key = tuple(tokens)
        cache_nbytes = sum(int(c.nbytes) for c in prompt_cache)
        payload_nbytes = _companion_nbytes(companion, companion_nbytes)
        entry = _StoreEntry(
            prompt_cache=prompt_cache,
            companion=companion,
            nbytes=cache_nbytes + payload_nbytes,
        )
        store_key = (producer_rail, model, key)
        previous = self._entries.pop(store_key, None)
        if previous is not None:
            self._n_bytes -= previous.nbytes
        self._entries[store_key] = entry
        self._n_bytes += entry.nbytes
        for entry_rail, entry_model, existing in list(self._entries):
            if (
                entry_rail != producer_rail
                or entry_model != model
                or len(existing) >= len(key)
            ):
                continue
            if key[:len(existing)] == existing:
                self._pop_entry((producer_rail, model, existing))
        while len(self._entries) > self.max_size:
            self._evict_oldest()
        if self.max_bytes is not None:
            while self._n_bytes > self.max_bytes and self._entries:
                self._evict_oldest()
        return store_key in self._entries


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


@dataclass(frozen=True)
class _MemoryRoute:
    """One claimed producer rail chosen without sacrificing cache depth."""

    kind: Literal["plain", "spec", "spec_target"]
    rail: Any
    probe: CacheProbe


def _choose_memory_route(
    plain: CacheProbe,
    speculative: CacheProbe,
    *,
    speculative_rail: Any,
) -> _MemoryRoute | None:
    """Choose the deepest usable target, using drafting only as a tie-break."""
    candidates: list[tuple[int, int, _MemoryRoute]] = []
    if plain.kind in {"shorter", "trim"} and plain.cached_tokens > 0:
        candidates.append(
            (
                plain.cached_tokens,
                1,
                _MemoryRoute("plain", PLAIN_CACHE_RAIL, plain),
            )
        )
    # A generic store cannot trim opaque drafter state. Exact entries also
    # lack the continuation logits needed to generate from an empty suffix.
    if speculative.kind == "shorter" and speculative.cached_tokens > 0:
        kind = "spec" if speculative.has_companion else "spec_target"
        priority = 2 if speculative.has_companion else 0
        candidates.append(
            (
                speculative.cached_tokens,
                priority,
                _MemoryRoute(kind, speculative_rail, speculative),
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


@dataclass
class PrefixCacheGenerator:
    """Generate from the deepest safe memory or disk prefix state."""

    model: Any
    tokenizer: Any
    manifest: dict
    cache_store: Any
    make_prompt_cache_fn: Callable = make_mlx_prompt_cache
    generate_fn: Callable = generate_with_metadata
    after_generate_fn: Callable | None = None
    disk_store: Any = None
    disk_registry: Any = None
    # Effective served sequence limit in tokens.
    context_limit: int | None = None
    clock: Callable[[], float] = time.perf_counter
    _cache_class_names: tuple[str, ...] | None = None
    _closed: bool = False
    _cold_spec_bypass_logged: bool = False
    _spec_resume_fallback_logged: bool = False

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
        except Exception as e:  # noqa: BLE001 - the docstring contract above
            log = getattr(self.disk_store, "_log", None)
            if log is not None:
                log(f"[disk_kv] restore failed; cold serving: {e!r}")
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
        if getattr(store, "writes_disabled", False):
            return None, {}
        write_depth = getattr(store, "write_depth_tokens", None)
        # Cheap precheck: the smallest frontier this call could build is the first
        # stride multiple strictly above the restored prefix. If that exceeds the
        # token count or the write-depth cap, no frontier is crossed and no
        # tracker is built.
        first_frontier = ((cached_tokens // stride) + 1) * stride
        if first_frontier > len(full_tokens):
            return None, {}
        if write_depth is not None and first_frontier > write_depth:
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
            write_depth=write_depth,
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

    def _spec_frontier_writer(
        self,
        model_key,
        full_tokens,
        cached_tokens,
        *,
        served,
        producer_rail,
        session_cache_key=None,
    ):
        """Build the paired target-plus-DSpark prefill writer, or none."""
        store = self.disk_store
        stride = getattr(store, "stride", None) if store is not None else None
        if store is None or stride is None:
            return None, {}

        from moespresso.runtime.deepseek_v4.spec_disk_kv import (
            make_spec_disk_kv_writer,
        )
        from moespresso.runtime.disk_kv import build_cache_scope
        from moespresso.runtime.serve import _model_prefill_step_size

        scope = build_cache_scope(model_key, self._cache_classes())
        default_step = _model_prefill_step_size(
            self.model, self.tokenizer, full_tokens
        )
        if default_step is None:
            default_step = 2048
        try:
            writer = make_spec_disk_kv_writer(
                store,
                scope=scope,
                full_tokens=list(full_tokens),
                restored_prefix=cached_tokens,
                served=served,
                producer_rail=producer_rail,
                default_step=default_step,
                session_cache_key=session_cache_key,
            )
        except Exception as e:  # noqa: BLE001 - disk capture is opportunistic
            log = getattr(store, "_log", None)
            if log is not None:
                log(
                    "[disk_kv] paired checkpoint planning failed; "
                    f"continuing without writes: {e!r}"
                )
            return None, {}
        if not writer.capture_frontiers:
            return None, {}
        generate_kwargs = {
            "prefill_step_size": default_step,
            "spec_prefill_progress_callback": writer.on_prefill_progress,
            "spec_prefill_progress_frontiers": list(writer.capture_frontiers),
        }
        if writer.prefill_plan:
            generate_kwargs["spec_prefill_plan"] = list(writer.prefill_plan)
        return writer, generate_kwargs

    def _spec_request_engages(
        self,
        *,
        kv_policy: KVPolicy,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float | None,
    ) -> bool:
        """True when an installed drafter will serve this request.

        Mirrors the generation seam's sampler rule so cache routing can choose
        a resumable speculative rail, a deeper plain target, or a cold path
        before the generation function makes the same engagement decision.
        """
        served = getattr(self.model, "_moespresso_ds4_drafter", None)
        if served is None:
            return False
        # The generation seam does not admit live KV quantization or an empty
        # completion budget into the speculative loop. Cache routing must make
        # the same decision before it selects a producer rail or installs a
        # speculative-only prefill callback.
        if kv_policy.live_kv_format != LIVE_KV_RAW or int(max_tokens) < 1:
            return False
        from moespresso.runtime.deepseek_v4.spec_serve import spec_sampler_eligible

        return spec_sampler_eligible(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            greedy_only=getattr(
                getattr(served, "drafter", None), "greedy_only", False
            ),
        )

    def _spec_producer_rail(self):
        """The current request's resumable DSpark rail, or ``None``."""
        served = getattr(self.model, "_moespresso_ds4_drafter", None)
        if served is None:
            return None
        from moespresso.runtime.deepseek_v4.spec_serve import (
            spec_cache_producer_rail,
        )

        return spec_cache_producer_rail(served)

    @staticmethod
    def _set_spec_publication_status(
        result: GenerationResult,
        status: str,
        *,
        reason: str | None = None,
    ) -> None:
        if result.speculative is None:
            return
        payload = {"status": status}
        if reason is not None:
            payload["reason"] = reason
        result.speculative["cache_publication"] = payload

    def _publish_spec_target(
        self,
        *,
        model_key: tuple,
        full_tokens: list[int],
        result: GenerationResult,
        target_cache,
        producer_rail,
        companion=None,
        declared_frontier: int | None = None,
    ) -> bool:
        """Publish one validated speculative-origin target on its own rail."""
        from moespresso.runtime.disk_kv import caches_shared_offset

        if target_cache is None:
            self._set_spec_publication_status(
                result, "skipped", reason="target_cache_missing"
            )
            return False
        frontier = caches_shared_offset(target_cache)
        if frontier is None:
            self._set_spec_publication_status(
                result, "skipped", reason="target_frontier_mismatch"
            )
            return False
        if declared_frontier is not None and frontier != declared_frontier:
            self._set_spec_publication_status(
                result, "skipped", reason="declared_frontier_mismatch"
            )
            return False
        timeline = list(full_tokens) + list(result.generated_token_ids)
        if frontier < len(full_tokens) or frontier > len(timeline):
            self._set_spec_publication_status(
                result, "skipped", reason="frontier_outside_public_tokens"
            )
            return False
        companion_drop_reason = None
        if companion is None and result.speculative is not None:
            previous = result.speculative.get("cache_publication") or {}
            companion_drop_reason = previous.get("reason")
        if companion is not None:
            if (
                getattr(companion, "producer_rail", None) != producer_rail
                or getattr(companion, "frontier", None) != frontier
            ):
                companion = None
                companion_drop_reason = "companion_identity_mismatch"
        try:
            retained = self.cache_store.insert_cache_with_companion(
                model_key,
                timeline[:frontier],
                target_cache,
                producer_rail=producer_rail,
                companion=companion,
            )
        except Exception:  # noqa: BLE001 - cache publication is opportunistic
            self._set_spec_publication_status(
                result, "skipped", reason="memory_store_failed"
            )
            return False
        if not retained:
            self._set_spec_publication_status(
                result, "skipped", reason="memory_budget"
            )
            return False
        if companion is None:
            self._set_spec_publication_status(
                result,
                "published_target_only",
                reason=companion_drop_reason,
            )
        else:
            self._set_spec_publication_status(result, "published")
        return True

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
        response_stop_callback: Callable[[], bool] | None = None,
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
        # budget must fit the served limit; a request exactly at the limit
        # passes (the sequence occupies positions 0 through limit - 1).
        validate_context_span(
            limit=self.context_limit,
            prompt_tokens=len(full_tokens),
            max_tokens=max_tokens,
        )

        model_key = cache_model_key(self.manifest, effective_rendering_id, kv_policy)
        spec_engaged = self._spec_request_engages(
            kv_policy=kv_policy,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
        )
        if response_stop_callback is not None:
            # The speculative loop may commit several accepted tokens at once.
            # Until it can roll back to an arbitrary text-detected boundary,
            # terminal tool-call requests use the one-token plain stream.
            spec_engaged = False
        spec_rail = self._spec_producer_rail() if spec_engaged else None
        enhanced_spec = (
            spec_rail is not None
            and callable(getattr(self.cache_store, "probe_nearest_cache", None))
            and callable(
                getattr(self.cache_store, "fetch_nearest_cache_with_companion", None)
            )
            and callable(
                getattr(self.cache_store, "insert_cache_with_companion", None)
            )
        )
        plain_probe = CacheProbe("miss", 0, False)
        spec_probe = CacheProbe("miss", 0, False)
        memory_route = None
        exact_seen = False
        if enhanced_spec:
            plain_probe = self.cache_store.probe_nearest_cache(
                model_key,
                full_tokens,
                producer_rail=PLAIN_CACHE_RAIL,
            )
            spec_probe = self.cache_store.probe_nearest_cache(
                model_key,
                full_tokens,
                producer_rail=spec_rail,
            )
            memory_route = _choose_memory_route(
                plain_probe,
                spec_probe,
                speculative_rail=spec_rail,
            )
            exact_seen = plain_probe.kind == "exact" or spec_probe.kind == "exact"

        # Probe is read-only. Commit a streaming response only after request,
        # context, and route validation, but before claiming the selected entry
        # by move. A failed socket write therefore leaves every rail resident.
        if ready_callback is not None:
            ready_callback()
        ready_at = self.clock()

        first_token_at = None

        def first_token_callback() -> None:
            nonlocal first_token_at
            if first_token_at is None:
                first_token_at = self.clock()

        prompt_cache = None
        suffix_tokens = list(full_tokens)
        cached_tokens = 0
        cache_event = "miss"
        writer = None
        spec_writer = None
        generate_kwargs = {}
        spec_continuation = None
        source_spec_rail = None
        source_spec_from_disk = False
        disk_attachment_restore = None
        disk_attachment_event = None
        disk_restore_seconds = None
        disk_attachment_restore_seconds = None
        disk_attachment_live_started = None
        disk_attachment_live_finished = False
        cache_claimed = False

        def spec_continuation_ready_callback() -> None:
            nonlocal disk_attachment_restore_seconds
            nonlocal disk_attachment_live_finished
            if (
                disk_attachment_live_started is None
                or disk_attachment_restore_seconds is None
                or disk_attachment_live_finished
            ):
                return
            disk_attachment_restore_seconds += (
                self.clock() - disk_attachment_live_started
            )
            disk_attachment_live_finished = True

        if spec_engaged and not enhanced_spec:
            # Drafters without a portable state contract retain the old cold
            # path. The cache event reports the actual outcome rather than a
            # separate speculative-only event value.
            if not self._cold_spec_bypass_logged:
                self._cold_spec_bypass_logged = True
                print(
                    "[spec] non-resumable speculative request: prompt-cache "
                    "store and disk KV are bypassed (fresh cache per request)",
                    flush=True,
                )
        elif enhanced_spec:
            if memory_route is not None:
                fetched, fetched_suffix, companion = (
                    self.cache_store.fetch_nearest_cache_with_companion(
                        model_key,
                        full_tokens,
                        producer_rail=memory_route.rail,
                    )
                )
                fetched_suffix = list(fetched_suffix)
                fetched_tokens = len(full_tokens) - len(fetched_suffix)
                route_matches = (
                    fetched is not None
                    and bool(fetched_suffix)
                    and fetched_tokens == memory_route.probe.cached_tokens
                )
                if route_matches and memory_route.kind == "plain":
                    prompt_cache = fetched
                    suffix_tokens = fetched_suffix
                    cached_tokens = fetched_tokens
                    cache_event = "hit"
                    cache_claimed = True
                elif route_matches:
                    from moespresso.runtime.disk_kv import caches_all_at_offset

                    if caches_all_at_offset(fetched, fetched_tokens):
                        suffix_tokens = fetched_suffix
                        cached_tokens = fetched_tokens
                        cache_event = "hit"
                        cache_claimed = True
                        source_spec_rail = memory_route.rail
                        if memory_route.kind == "spec" and companion is not None:
                            from moespresso.runtime.deepseek_v4.spec_serve import (
                                SpecContinuation,
                            )

                            spec_continuation = SpecContinuation(
                                target_cache=fetched,
                                companion=companion,
                                prefix_offset=fetched_tokens,
                            )
                        else:
                            prompt_cache = fetched

            # A corrupt speculative target must not hide a still-resident
            # plain candidate selected by the read-only probe.
            if (
                not cache_claimed
                and plain_probe.kind in {"shorter", "trim"}
                and plain_probe.cached_tokens > 0
            ):
                fetched, fetched_suffix, _ = (
                    self.cache_store.fetch_nearest_cache_with_companion(
                        model_key,
                        full_tokens,
                        producer_rail=PLAIN_CACHE_RAIL,
                    )
                )
                fetched_suffix = list(fetched_suffix)
                fetched_tokens = len(full_tokens) - len(fetched_suffix)
                if (
                    fetched is not None
                    and fetched_suffix
                    and fetched_tokens == plain_probe.cached_tokens
                ):
                    prompt_cache = fetched
                    suffix_tokens = fetched_suffix
                    cached_tokens = fetched_tokens
                    cache_event = "hit"
                    cache_claimed = True

            # With no usable in-memory target, consult disk even when an exact
            # speculative entry remains resident and cannot supply logits.
            if not cache_claimed and self.disk_store is not None:
                disk_restore_started = self.clock()
                disk_hit = self._consult_disk(model_key, full_tokens)
                disk_restore_finished = self.clock()
                if disk_hit is not None:
                    disk_restore_seconds = (
                        disk_restore_finished - disk_restore_started
                    )
                if disk_hit is not None and disk_hit.suffix_tokens:
                    suffix_tokens = list(disk_hit.suffix_tokens)
                    cached_tokens = disk_hit.cached_tokens
                    cache_event = "disk_hit"
                    cache_claimed = True
                    served = getattr(
                        self.model, "_moespresso_ds4_drafter", None
                    )
                    disk_attachment_started = self.clock()
                    try:
                        from moespresso.runtime.deepseek_v4.spec_disk_kv import (
                            restore_spec_attachment,
                        )

                        disk_attachment_restore = restore_spec_attachment(
                            self.disk_store,
                            disk_hit,
                            served,
                            spec_rail,
                        )
                        disk_attachment_event = disk_attachment_restore.status
                    except Exception as e:  # noqa: BLE001 - target hit stays valid
                        log = getattr(self.disk_store, "_log", None)
                        if log is not None:
                            log(
                                "[disk_kv] DSpark attachment restore failed; "
                                f"keeping target cache: {e!r}"
                            )
                        disk_attachment_restore = None
                        disk_attachment_event = "unavailable"
                    finally:
                        disk_attachment_elapsed = (
                            self.clock() - disk_attachment_started
                        )
                    if disk_attachment_event == "hit":
                        disk_attachment_restore_seconds = (
                            disk_attachment_elapsed
                        )
                    if (
                        disk_attachment_restore is not None
                        and disk_attachment_restore.continuation is not None
                    ):
                        spec_continuation = (
                            disk_attachment_restore.continuation
                        )
                        source_spec_rail = spec_rail
                        source_spec_from_disk = True
                        prompt_cache = None
                    else:
                        prompt_cache = disk_hit.prompt_cache
                elif disk_hit is not None:
                    exact_seen = True

            if not cache_claimed:
                # A cold eligible DSpark request creates its target cache inside
                # the speculative loop. Exact entries stay untouched.
                prompt_cache = None
                suffix_tokens = list(full_tokens)
                cached_tokens = 0
                cache_event = "exact_fallback" if exact_seen else "miss"
        else:
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
                disk_restore_started = self.clock()
                disk_hit = self._consult_disk(model_key, full_tokens)
                disk_restore_finished = self.clock()
                if disk_hit is not None:
                    disk_restore_seconds = (
                        disk_restore_finished - disk_restore_started
                    )
                    prompt_cache = disk_hit.prompt_cache
                    suffix_tokens = list(disk_hit.suffix_tokens)
                    cached_tokens = disk_hit.cached_tokens
                    cache_event = "disk_hit"

            # MLX generation needs at least one prompt token to compute the next token.
            # If the trie returns an exact whole-prompt cache (empty suffix), build a
            # fresh cache so there is a token to feed. Follow-up chat turns still hit
            # normally because they add user suffix tokens.
            if prompt_cache is None or not suffix_tokens:
                cache_event = "exact_fallback" if prompt_cache is not None else "miss"
                prompt_cache = self.make_prompt_cache_fn(self.model)
                suffix_tokens = list(full_tokens)
                cached_tokens = 0

        # Plain generation writes the independently valid target checkpoint.
        # A live DSpark request instead uses the paired post-ingest callback so
        # target and drafter state are captured at the same proven frontier.
        if prompt_cache is not None:
            writer, generate_kwargs = self._frontier_writer(
                model_key, full_tokens, cached_tokens, prompt_cache,
                session_cache_key=session_cache_key)
        elif enhanced_spec:
            served = getattr(self.model, "_moespresso_ds4_drafter", None)
            spec_writer, generate_kwargs = self._spec_frontier_writer(
                model_key,
                full_tokens,
                cached_tokens,
                served=served,
                producer_rail=spec_rail,
                session_cache_key=session_cache_key,
            )

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

        spec_frontier_progress = generate_kwargs.get(
            "spec_prefill_progress_callback"
        )
        if spec_frontier_progress is not None and progress_callback is not None:
            def combined_spec_progress(event) -> None:
                spec_frontier_progress(event)
                progress_callback(event.processed, event.total)

            generate_kwargs["spec_prefill_progress_callback"] = (
                combined_spec_progress
            )
        if response_callback is not None:
            generate_kwargs["response_callback"] = response_callback
        if response_stop_callback is not None:
            generate_kwargs["response_stop_callback"] = response_stop_callback
        generate_kwargs["first_token_callback"] = first_token_callback

        call_kwargs = dict(
            prompt_cache=prompt_cache,
            cached_tokens=cached_tokens,
            kv_policy=kv_policy,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **sampling_kwargs,
            **generate_kwargs,
        )
        if spec_continuation is not None:
            call_kwargs["spec_continuation"] = spec_continuation
        if source_spec_from_disk:
            disk_attachment_live_started = self.clock()
            call_kwargs["spec_continuation_ready_callback"] = (
                spec_continuation_ready_callback
            )
        try:
            result = self.generate_fn(
                self.model,
                self.tokenizer,
                suffix_tokens,
                **call_kwargs,
            )
        except Exception as exc:
            from moespresso.runtime.deepseek_v4.spec_serve import (
                SpecContinuationError,
            )

            if spec_continuation is None or not isinstance(
                exc, SpecContinuationError
            ):
                raise
            # Preflight and capsule import happen before emitter/run. Reuse a
            # still-valid target with plain generation; if its frontier changed,
            # cold-serve instead. No second generation follows streamed output.
            from moespresso.runtime.disk_kv import caches_all_at_offset

            fallback_generate_kwargs = {
                key: value
                for key, value in generate_kwargs.items()
                if not key.startswith("spec_prefill_")
            }
            if source_spec_from_disk:
                # A disk target is a generic target checkpoint. If the paired
                # continuation is rejected after restore validation, plain
                # fallback publishes it on the plain memory rail.
                source_spec_rail = None
                disk_attachment_event = "invalid"
            if caches_all_at_offset(
                spec_continuation.target_cache,
                spec_continuation.prefix_offset,
            ):
                fallback_cache = spec_continuation.target_cache
                result = self.generate_fn(
                    self.model,
                    self.tokenizer,
                    suffix_tokens,
                    prompt_cache=fallback_cache,
                    cached_tokens=cached_tokens,
                    kv_policy=kv_policy,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **sampling_kwargs,
                    **fallback_generate_kwargs,
                )
            else:
                source_spec_rail = None
                cache_event = "exact_fallback" if exact_seen else "miss"
                result = self.generate_fn(
                    self.model,
                    self.tokenizer,
                    list(full_tokens),
                    prompt_cache=None,
                    cached_tokens=0,
                    kv_policy=kv_policy,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **sampling_kwargs,
                    **fallback_generate_kwargs,
                )
            if not self._spec_resume_fallback_logged:
                self._spec_resume_fallback_logged = True
                print(
                    "[spec] cached DSpark state was rejected; retained target "
                    "cache reuse where its frontier remained valid",
                    flush=True,
                )
        else:
            if spec_continuation is not None and result.speculative is not None:
                result.speculative["cache_resume"] = {
                    "status": "hit",
                    "frontier": cached_tokens,
                }
        if writer is not None and writer.written:
            result.disk_checkpoints_written = len(writer.written)
            result.disk_checkpoint_write_seconds = tuple(writer.write_seconds)
        if spec_writer is not None:
            if spec_writer.target_writes:
                result.disk_checkpoints_written = len(spec_writer.target_writes)
                result.disk_checkpoint_write_seconds = tuple(
                    spec_writer.target_blocking_seconds
                )
            if spec_writer.attachment_writes:
                result.disk_attachments_written = len(
                    spec_writer.attachment_writes
                )
                result.disk_attachment_write_seconds = tuple(
                    spec_writer.attachment_blocking_seconds
                )
        result.disk_attachment_event = disk_attachment_event
        result.disk_restore_seconds = disk_restore_seconds
        result.disk_attachment_restore_seconds = (
            disk_attachment_restore_seconds
            if disk_attachment_event == "hit"
            else None
        )
        if first_token_at is not None:
            result.ready_to_first_token_seconds = first_token_at - ready_at

        if source_spec_rail is not None and result.prompt_cache is not None:
            target_cache = result.prompt_cache
            result.prompt_cache = None
            self._publish_spec_target(
                model_key=model_key,
                full_tokens=full_tokens,
                result=result,
                target_cache=target_cache,
                producer_rail=source_spec_rail,
            )
        elif result.prompt_cache is not None:
            cache_key = list(full_tokens) + list(result.generated_token_ids)
            if cache_key:
                self.cache_store.insert_cache(model_key, cache_key, result.prompt_cache)
        if enhanced_spec and result.speculative_prompt_cache is not None:
            self._publish_spec_target(
                model_key=model_key,
                full_tokens=full_tokens,
                result=result,
                target_cache=result.speculative_prompt_cache,
                producer_rail=spec_rail,
                companion=result.cache_companion,
                declared_frontier=result.cache_frontier,
            )
        if self.after_generate_fn is not None:
            self.after_generate_fn(self.model)
        result.cache_event = cache_event
        result.cache_entries = _store_entries(self.cache_store)
        result.cache_bytes = _store_nbytes(self.cache_store)
        return result
