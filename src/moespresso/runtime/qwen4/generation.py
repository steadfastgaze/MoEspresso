"""Owned Qwen4 generation with composite prompt checkpoint reuse."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
import time
from typing import Any, Callable

import mlx.core as mx

from moespresso.runtime.generation import ContextLimitError, GenerationResult
from moespresso.runtime.kv_policy import KVPolicy, LIVE_KV_RAW, validate_runtime_policy


QWEN4_GENERATION_ADAPTER = "qwen4_composite_v1"
QWEN4_LIVE_CACHE_FORMAT = "qwen_kvarn_k4v4"
DEFAULT_QWEN4_PREFILL_STEP_SIZE = 2048
_PRESSURE_PREFILL_STEP_SIZE = 512
_PREFILL_ACTIVE_HEADROOM_BYTES = 7 << 30
_PREFILL_HOST_RESERVE_BYTES = 3 << 30
_PREFILL_MAX_FREE_CACHE_BYTES = 4 << 30


def _prefill_memory_policy(
    *,
    capacity: int | None,
    prompt_rows: int,
    recommended_bytes: int,
    active_bytes: int,
    available_bytes: int,
) -> tuple[int, int] | None:
    """Bound long prefill work when resident expert pools leave little headroom."""
    if (
        capacity is None
        or capacity >= 512
        or prompt_rows <= _PRESSURE_PREFILL_STEP_SIZE
        or recommended_bytes - active_bytes >= _PREFILL_ACTIVE_HEADROOM_BYTES
    ):
        return None
    free_cache = min(
        _PREFILL_MAX_FREE_CACHE_BYTES,
        max(0, available_bytes - _PREFILL_HOST_RESERVE_BYTES),
    )
    return _PRESSURE_PREFILL_STEP_SIZE, free_cache


def _resolve_prefill_memory_policy(model: Any, prompt_rows: int) -> tuple[int, int] | None:
    capacity = getattr(model, "_moespresso_ssd_streaming_capacity", None)
    if not isinstance(capacity, int) or isinstance(capacity, bool):
        return None
    try:
        import psutil

        return _prefill_memory_policy(
            capacity=capacity,
            prompt_rows=prompt_rows,
            recommended_bytes=int(mx.device_info()["max_recommended_working_set_size"]),
            active_bytes=int(mx.get_active_memory()),
            available_bytes=int(psutil.virtual_memory().available),
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _split_prefill_plan(plan: list[int], maximum: int) -> list[int]:
    parts = []
    for width in plan:
        while width > maximum:
            parts.append(maximum)
            width -= maximum
        parts.append(width)
    return parts


@contextmanager
def _limited_free_cache(limit: int):
    previous = mx.set_cache_limit(0)
    try:
        mx.clear_cache()
        mx.set_cache_limit(min(previous, limit))
        yield
    finally:
        try:
            mx.clear_cache()
        finally:
            mx.set_cache_limit(previous)


def is_qwen4_generation_model(model: Any) -> bool:
    """Return whether the loader explicitly selected this generation seam."""

    return getattr(model, "_moespresso_generation_adapter", None) == QWEN4_GENERATION_ADAPTER


@dataclass(frozen=True)
class Qwen4GenerationResponse:
    """One committed public token in the shape consumed by the serve layer."""

    text: str
    token: int
    logprobs: Any
    from_draft: bool
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: str | None


class _DecodePerToken:
    """Fallback incremental decoder for synthetic and minimal tokenizers."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.last_segment = ""

    def add_token(self, token: int) -> None:
        self.last_segment = str(self._tokenizer.decode([int(token)]))

    def finalize(self) -> None:
        pass


def _detokenizer(tokenizer: Any) -> Any:
    detokenizer = getattr(tokenizer, "detokenizer", None)
    if detokenizer is not None:
        return detokenizer
    return _DecodePerToken(tokenizer)


def _prompt_tokens(tokenizer: Any, prompt: str | list[int] | Any) -> list[int]:
    if isinstance(prompt, str):
        bos = getattr(tokenizer, "bos_token", None)
        add_special_tokens = bos is None or not prompt.startswith(bos)
        prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if isinstance(prompt, mx.array):
        if prompt.ndim != 1:
            raise ValueError("Qwen4 generation requires a one-dimensional prompt")
        prompt = prompt.tolist()
    try:
        tokens = list(prompt)
    except TypeError as exc:
        raise TypeError("prompt must be rendered text or token ids") from exc
    if not tokens:
        raise ValueError("Qwen4 generation requires at least one prompt token")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
        raise ValueError("prompt token ids must be nonnegative integers")
    return tokens


def _stop_ids(model: Any, tokenizer: Any) -> frozenset[int]:
    declared = getattr(model, "_moespresso_qwen4_stop_ids", None)
    if declared is None:
        declared = getattr(tokenizer, "eos_token_ids", None)
    if not declared:
        raise RuntimeError("Qwen4 generation has no declared stop token ids")
    stop_ids = frozenset(int(token) for token in declared)
    if any(token < 0 for token in stop_ids):
        raise RuntimeError("Qwen4 generation stop token ids are invalid")
    return stop_ids


def _context_limit(model: Any, requested: int | None) -> int | None:
    value = requested
    if value is None:
        value = getattr(model, "_moespresso_qwen4_kvarn_context_tokens", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Qwen4 context limit must be a positive integer")
    return value


def _peak_memory_gb() -> float:
    try:
        return float(mx.get_peak_memory()) / 1e9
    except Exception:
        return 0.0


def _sample_next(
    logits: mx.array,
    *,
    tokens: list[int],
    generated: list[int],
    processors: list[Callable] | None,
    sampler: Callable,
) -> tuple[mx.array, mx.array]:
    """Build the next sampled token and its complete log-probability row."""

    next_logits = logits[:, -1, :]
    if processors:
        history = mx.array(tokens + generated, dtype=mx.int64)
        for processor in processors:
            next_logits = processor(history, next_logits)
    logprobs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)
    return sampler(logprobs), logprobs


def _decode_response(
    detokenizer: Any,
    *,
    token: int,
    logprobs: mx.array,
    is_stop: bool,
    is_length: bool,
    prompt_tokens: int,
    prompt_tps: float,
    generated_tokens: int,
    generation_started: float,
) -> Qwen4GenerationResponse:
    fallback = isinstance(detokenizer, _DecodePerToken)
    if not is_stop:
        detokenizer.add_token(token)
    if is_stop or is_length:
        detokenizer.finalize()
    segment = "" if is_stop and fallback else str(detokenizer.last_segment)
    elapsed = max(time.perf_counter() - generation_started, 1e-12)
    finish_reason = "stop" if is_stop else ("length" if is_length else None)
    return Qwen4GenerationResponse(
        text=segment,
        token=token,
        logprobs=logprobs,
        from_draft=False,
        prompt_tokens=prompt_tokens,
        prompt_tps=prompt_tps,
        generation_tokens=generated_tokens,
        generation_tps=generated_tokens / elapsed,
        peak_memory=_peak_memory_gb(),
        finish_reason=finish_reason,
    )


def generate_qwen4_with_metadata(
    model: Any,
    tokenizer: Any,
    prompt: str | list[int] | Any,
    *,
    prompt_cache: Any = None,
    cached_tokens: int | None = None,
    kv_policy: KVPolicy | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    presence_penalty: float | None = None,
    presence_context_size: int = 20,
    top_logprobs: int | None = None,
    prefill_step_size: int | None = None,
    prompt_progress_callback: Callable[[int, int], None] | None = None,
    response_callback: Callable[[int, object], None] | None = None,
    response_stop_callback: Callable[[], bool] | None = None,
    first_token_callback: Callable[[], None] | None = None,
    sampler_factory: Callable | None = None,
    logits_processors_factory: Callable | None = None,
    context_limit: int | None = None,
    initial_state: Any = None,
    initial_logits: Any = None,
    prefill_plan: list[int] | None = None,
    prefill_state_callback: Callable[[Any, Any], None] | None = None,
) -> GenerationResult:
    """Generate from a fresh or independently restored composite prefix.

    A coordinator owns every GDN, QSA/KVarN and PLE state object for this
    request. Prompt chunks are public and commit atomically. During decode, a
    sampled token is proposed and committed only before it is used to produce
    the following public token. The coordinator is closed on completion,
    callback cancellation, or any model error.
    """

    if not is_qwen4_generation_model(model):
        raise TypeError("model did not select the Qwen4 generation adapter")
    if prompt_cache is not None:
        raise ValueError("Qwen4 requires its composite checkpoint, not a generic prompt cache")
    cached_tokens = 0 if cached_tokens is None else cached_tokens
    if isinstance(cached_tokens, bool) or not isinstance(cached_tokens, int) or cached_tokens < 0:
        raise ValueError("cached_tokens must be a nonnegative integer")
    if initial_state is None:
        if cached_tokens or initial_logits is not None:
            raise ValueError("Qwen4 restored prefix requires composite state")
    elif initial_state.frontier != cached_tokens or cached_tokens == 0:
        raise ValueError("Qwen4 restored state does not match cached_tokens")
    if kv_policy is not None:
        validate_runtime_policy(kv_policy)
        if kv_policy.live_kv_format != LIVE_KV_RAW:
            raise ValueError(
                "Qwen4 owns its K4/V4 live-cache format; generic live KV "
                "formats are not request options"
            )
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("max_tokens must be an integer")
    if max_tokens == 0 or max_tokens < -1:
        raise ValueError("max_tokens must be positive or -1")
    if prefill_step_size is None:
        prefill_step_size = int(
            getattr(
                model,
                "_moespresso_qwen4_prefill_step_size",
                DEFAULT_QWEN4_PREFILL_STEP_SIZE,
            )
        )
    if (
        isinstance(prefill_step_size, bool)
        or not isinstance(prefill_step_size, int)
        or prefill_step_size <= 0
    ):
        raise ValueError("prefill_step_size must be a positive integer")

    tokens = _prompt_tokens(tokenizer, prompt)
    if cached_tokens > len(tokens) or (cached_tokens == len(tokens) and initial_logits is None):
        raise ValueError("Qwen4 cached prompt frontier requires matching continuation logits")
    plan = [] if prefill_plan is None else list(prefill_plan)
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in plan):
        raise ValueError("Qwen4 prefill plan must contain positive integer chunk sizes")
    if sum(plan) > len(tokens) - cached_tokens:
        raise ValueError("Qwen4 prefill plan exceeds the remaining prompt")
    memory_policy = _resolve_prefill_memory_policy(model, len(tokens) - cached_tokens)
    if memory_policy is not None:
        maximum_chunk, cache_limit = memory_policy
        prefill_step_size = min(prefill_step_size, maximum_chunk)
        plan = _split_prefill_plan(plan, maximum_chunk)
        print(
            "[serve] Qwen4 memory-aware prefill: "
            f"chunk={prefill_step_size} free_cache_limit={cache_limit / (1 << 30):.2f}GiB",
            flush=True,
        )
    planned_chunks = iter(plan)
    served_limit = _context_limit(model, context_limit)
    requested_tokens = max_tokens
    if max_tokens == -1:
        if served_limit is None:
            raise ValueError("unbounded Qwen4 generation requires a context limit")
        requested_tokens = served_limit - len(tokens)
    if served_limit is not None and len(tokens) + requested_tokens > served_limit:
        raise ContextLimitError(
            limit=served_limit,
            prompt_tokens=len(tokens),
            max_tokens=max_tokens,
        )
    if requested_tokens <= 0:
        raise ContextLimitError(
            limit=int(served_limit or len(tokens)),
            prompt_tokens=len(tokens),
            max_tokens=max_tokens,
        )

    if sampler_factory is None:
        from mlx_lm.sample_utils import make_sampler as sampler_factory
    sampler = sampler_factory(
        temp=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        min_p=float(min_p),
    )
    processors = None
    if presence_penalty:
        if logits_processors_factory is None:
            from mlx_lm.sample_utils import (
                make_logits_processors as logits_processors_factory,
            )
        processors = logits_processors_factory(
            presence_penalty=float(presence_penalty),
            presence_context_size=int(presence_context_size),
        )

    stop_ids = _stop_ids(model, tokenizer)
    detokenizer = _detokenizer(tokenizer)
    generated: list[int] = []
    text_parts: list[str] = []
    token_logprobs: list[float] = []
    captured_top_logprobs: list[tuple[dict, ...]] = []
    coordinator = None
    serial_lane = None
    contexts = ExitStack()
    pipeline_scope = ExitStack()
    started = time.perf_counter()
    first_token_seconds = None
    finish_reason = "length"
    progress = prompt_progress_callback or (lambda *_args: None)

    try:
        from mlx_lm.generate import generation_stream, wired_limit

        contexts.enter_context(wired_limit(model, [generation_stream]))
        contexts.enter_context(mx.stream(generation_stream))
        prefill_cache_scope = (
            _limited_free_cache(memory_policy[1])
            if memory_policy is not None else nullcontext()
        )
        with prefill_cache_scope:
            coordinator = (
                model.new_coordinator(1) if initial_state is None
                else model.restore_coordinator(initial_state)
            )
            progress(0, len(tokens) - cached_tokens)
            logits = initial_logits
            processed = cached_tokens
            while processed < len(tokens):
                width = next(planned_chunks, prefill_step_size)
                chunk = tokens[processed : processed + width]
                logits = coordinator.forward_chunk(mx.array([chunk], dtype=mx.int64))
                processed += len(chunk)
                if prefill_state_callback is not None:
                    prefill_state_callback(coordinator.state, logits)
                progress(processed - cached_tokens, len(tokens) - cached_tokens)
                if processed < len(tokens):
                    mx.clear_cache()
            assert logits is not None
            if processed == cached_tokens and prefill_state_callback is not None:
                prefill_state_callback(coordinator.state, logits)
        prompt_elapsed = max(time.perf_counter() - started, 1e-12)
        prompt_tps = (len(tokens) - cached_tokens) / prompt_elapsed
        generation_started = time.perf_counter()

        try_serial = getattr(coordinator, "try_enter_plain_serial_lane", None)
        if callable(try_serial):
            serial_lane = try_serial()

        pipeline_methods = (
            "prime_pipelined_step",
            "chain_pipelined_step",
            "finish_pipelined_transition",
            "finish_terminal_pipelined_step",
        )
        bounded_pipeline = bool(getattr(model, "_moespresso_pooled_decode_bounded", False))
        pipeline_enabled = bool(
            serial_lane is not None
            and processors is None
            and requested_tokens > 1
            and all(callable(getattr(serial_lane, name, None)) for name in pipeline_methods)
        )
        if pipeline_enabled and bounded_pipeline:
            from moespresso.runtime.pooled_moe import pooled_request_scope

            # The nested lane calls retain one owner until close() has either
            # committed the terminal row or cancelled and drained private work.
            pipeline_scope.enter_context(
                pooled_request_scope(model, getattr(serial_lane, "_identity", serial_lane))
            )
        pipeline_step = None
        pipeline_sampled = None
        pipeline_logprobs = None

        sampled, logprobs = _sample_next(
            logits,
            tokens=tokens,
            generated=generated,
            processors=processors,
            sampler=sampler,
        )
        if pipeline_enabled:
            assert serial_lane is not None
            pipeline_step = serial_lane.begin_step(sampled.reshape(1, 1))
            pipeline_sampled, pipeline_logprobs = _sample_next(
                pipeline_step.logits,
                tokens=tokens,
                generated=generated,
                processors=None,
                sampler=sampler,
            )
            serial_lane.prime_pipelined_step(
                pipeline_step,
                evaluated=(sampled, logprobs),
                queued=(pipeline_sampled, pipeline_logprobs),
            )
        else:
            mx.eval(sampled, logprobs)

        for step in range(1, requested_tokens + 1):
            token = int(sampled.item())
            vector_logprobs = logprobs.squeeze(0)
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - started
                if first_token_callback is not None:
                    first_token_callback()

            generated.append(token)
            is_stop = token in stop_ids
            is_length = step == requested_tokens
            response = _decode_response(
                detokenizer,
                token=token,
                logprobs=vector_logprobs,
                is_stop=is_stop,
                is_length=is_length,
                prompt_tokens=len(tokens) - cached_tokens,
                prompt_tps=prompt_tps,
                generated_tokens=step,
                generation_started=generation_started,
            )
            text_parts.append(response.text)
            if top_logprobs is not None:
                from moespresso.runtime.serve import (
                    _logprob_at,
                    _top_logprob_entries,
                )

                captured_top_logprobs.append(
                    _top_logprob_entries(
                        vector_logprobs,
                        tokenizer,
                        k=top_logprobs,
                    )
                )
                token_logprobs.append(_logprob_at(vector_logprobs, token))
            if response_callback is not None:
                response_callback(step, response)
            stopped_by_response = response_stop_callback is not None and response_stop_callback()
            if stopped_by_response:
                finish_reason = "stop"
                break
            if is_stop:
                finish_reason = "stop"
                break
            if is_length:
                finish_reason = "length"
                break

            if pipeline_step is not None:
                assert serial_lane is not None
                assert pipeline_sampled is not None
                assert pipeline_logprobs is not None
                remaining = requested_tokens - step
                if remaining == 1:
                    serial_lane.finish_terminal_pipelined_step(
                        pipeline_step,
                        evaluated=(pipeline_sampled, pipeline_logprobs),
                    )
                    sampled = pipeline_sampled
                    logprobs = pipeline_logprobs
                    pipeline_step = None
                    pipeline_sampled = None
                    pipeline_logprobs = None
                    continue

                next_step = serial_lane.chain_pipelined_step(
                    pipeline_step,
                    pipeline_sampled.reshape(1, 1),
                )
                future_sampled, future_logprobs = _sample_next(
                    next_step.logits,
                    tokens=tokens,
                    generated=generated,
                    processors=None,
                    sampler=sampler,
                )
                serial_lane.finish_pipelined_transition(
                    pipeline_step,
                    next_step,
                    evaluated=(pipeline_sampled, pipeline_logprobs),
                    queued=(future_sampled, future_logprobs),
                )
                sampled = pipeline_sampled
                logprobs = pipeline_logprobs
                pipeline_step = next_step
                pipeline_sampled = future_sampled
                pipeline_logprobs = future_logprobs
                continue

            input_ids = mx.array([[token]], dtype=mx.int64)
            if serial_lane is not None:
                serial_step = serial_lane.begin_step(input_ids)
                sampled, logprobs = _sample_next(
                    serial_step.logits,
                    tokens=tokens,
                    generated=generated,
                    processors=processors,
                    sampler=sampler,
                )
                serial_lane.finish_step(
                    serial_step,
                    evaluated=(sampled, logprobs),
                )
            else:
                candidate = coordinator.propose(input_ids)
                coordinator.commit(candidate, 1)
                logits = candidate.logits
                sampled, logprobs = _sample_next(
                    logits,
                    tokens=tokens,
                    generated=generated,
                    processors=processors,
                    sampler=sampler,
                )
                mx.eval(sampled, logprobs)
    finally:
        try:
            try:
                if serial_lane is not None:
                    serial_lane.close()
            finally:
                pipeline_scope.close()
        finally:
            try:
                if coordinator is not None:
                    coordinator.close()
            finally:
                contexts.close()

    generation_seconds = time.perf_counter() - started
    return GenerationResult(
        text="".join(text_parts),
        finish_reason=finish_reason,
        prompt_tokens=len(tokens) - cached_tokens,
        completion_tokens=len(generated),
        cached_tokens=cached_tokens,
        generated_token_ids=tuple(generated),
        prompt_cache=None,
        token_logprobs=tuple(token_logprobs),
        top_logprobs=tuple(captured_top_logprobs),
        first_token_seconds=first_token_seconds,
        generation_seconds=generation_seconds,
    )


@dataclass
class Qwen4RequestGenerator:
    """HTTP adapter with immutable prefix snapshots and independent live owners."""

    model: Any
    tokenizer: Any
    context_limit: int
    after_generate_fn: Callable | None = None
    cache_store: Any = None
    disk_store: Any = None
    clock: Callable[[], float] = time.perf_counter
    _closed: bool = False

    def cache_stats(self) -> dict[str, Any]:
        return {
            "default_live_kv_format": QWEN4_LIVE_CACHE_FORMAT,
            "supported_live_kv_formats": [QWEN4_LIVE_CACHE_FORMAT],
            "entries": len(self.cache_store) if self.cache_store is not None else 0,
            "bytes": self.cache_store.nbytes if self.cache_store is not None else 0,
            "disk": self.disk_store.stats() if self.disk_store is not None else {"enabled": False},
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.model, "close", None)
        if callable(close):
            close()

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
        if self._closed:
            raise RuntimeError("Qwen4 request generator is closed")
        validate_runtime_policy(kv_policy)
        if kv_policy.live_kv_format != LIVE_KV_RAW:
            raise ValueError(
                "Qwen4 owns its K4/V4 live-cache format; generic live KV formats "
                "are not request options"
            )
        tokens = _prompt_tokens(self.tokenizer, rendered_prompt)
        if len(tokens) + int(max_tokens) > self.context_limit:
            raise ContextLimitError(
                limit=self.context_limit,
                prompt_tokens=len(tokens),
                max_tokens=max_tokens,
            )
        if ready_callback is not None:
            ready_callback()
        ready_at = self.clock()
        first_token_at = None

        def first_token_callback() -> None:
            nonlocal first_token_at
            if first_token_at is None:
                first_token_at = self.clock()

        from moespresso.runtime.disk_kv import DiskKVError, scope_hash
        from moespresso.runtime.qwen4.disk_cache import (
            Qwen4CheckpointWriter,
            Qwen4PromptCache,
            qwen4_cache_scope,
            validate_qwen4_checkpoint,
        )

        scope = None
        if self.cache_store is not None or self.disk_store is not None:
            scope = qwen4_cache_scope(self.model, effective_rendering_id)
        model_key = scope_hash(scope) if scope is not None else None
        initial_state = initial_logits = None
        cached_tokens = 0
        cache_event = "miss" if scope is not None else "bypass"
        disk_restore_seconds = None
        if self.cache_store is not None:
            caches, suffix = self.cache_store.fetch_nearest_cache(model_key, tokens)
            if caches is not None:
                try:
                    cached_tokens = len(tokens) - len(suffix)
                    if len(caches) != 1 or not isinstance(caches[0], Qwen4PromptCache):
                        raise DiskKVError("incompatible Qwen memory checkpoint")
                    caches[0].restore(self.model, cached_tokens)
                    initial_state, initial_logits = caches[0].take_restored()
                    cache_event = "hit"
                except DiskKVError:
                    cached_tokens = 0
        if initial_state is None and self.disk_store is not None:
            started = self.clock()
            try:
                hit = self.disk_store.restore(
                    scope, tokens, make_cache_fn=lambda: [Qwen4PromptCache()],
                    registry={Qwen4PromptCache.__name__},
                    validate_fn=lambda caches, entry: validate_qwen4_checkpoint(caches, entry, self.model),
                )
                if hit is not None:
                    cached_tokens = hit.cached_tokens
                    initial_state, initial_logits = hit.prompt_cache[0].take_restored()
                    disk_restore_seconds = self.clock() - started
                    cache_event = "disk_hit"
                    hit = None
            except DiskKVError:
                cached_tokens = 0
        writer = None
        default_step = int(getattr(self.model, "_moespresso_qwen4_prefill_step_size", DEFAULT_QWEN4_PREFILL_STEP_SIZE))
        if (
            self.disk_store is not None and self.disk_store.stride is not None
            and not self.disk_store.writes_disabled
        ):
            writer = Qwen4CheckpointWriter(
                self.model, self.disk_store, scope, tokens, cached_tokens, session_cache_key,
            )
        pending_memory = None

        def capture_prefill(state, logits):
            nonlocal pending_memory
            if writer is not None:
                writer.capture(state, logits)
            if state.frontier == len(tokens) and self.cache_store is not None:
                pending_memory = Qwen4PromptCache.capture(self.model, state, logits)

        checkpoint_kwargs = {}
        if initial_state is not None:
            checkpoint_kwargs.update(initial_state=initial_state, initial_logits=initial_logits)
        if writer is not None:
            checkpoint_kwargs["prefill_plan"] = writer.prefill_plan(default_step)
        if writer is not None or self.cache_store is not None:
            checkpoint_kwargs["prefill_state_callback"] = capture_prefill

        result = generate_qwen4_with_metadata(
            self.model,
            self.tokenizer,
            tokens,
            cached_tokens=cached_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            prompt_progress_callback=progress_callback,
            response_callback=response_callback,
            response_stop_callback=response_stop_callback,
            first_token_callback=first_token_callback,
            context_limit=self.context_limit,
            **checkpoint_kwargs,
        )
        if pending_memory is not None:
            self.cache_store.insert_cache(model_key, tokens, [pending_memory])
        if writer is not None:
            result.disk_checkpoints_written = len(writer.written)
            result.disk_checkpoint_write_seconds = tuple(writer.write_seconds)
        result.disk_restore_seconds = disk_restore_seconds
        if first_token_at is not None:
            result.ready_to_first_token_seconds = first_token_at - ready_at
        if self.after_generate_fn is not None:
            self.after_generate_fn(self.model)
        result.cache_event = cache_event
        result.cache_entries = len(self.cache_store) if self.cache_store is not None else 0
        result.cache_bytes = self.cache_store.nbytes if self.cache_store is not None else 0
        return result
