"""Thin OpenAI-compatible HTTP layer over the manifest-driven serve adapter.

The model is built once from the manifest (via load_served_model) and held; each
request renders a prompt, calls the same generate_once the CLI uses, and shapes an
OpenAI chat-completions response. This module adds only request/response shaping +
routing, no quantization, no loading logic. It does not verify on load (run
moespresso-verify for the integrity gate).

Pure-core/IO-edge split: the request->response core is pure (a parsed request
dict + an injected `generate(prompt, **opts) -> str` callable -> a response dict),
so it is fully testable without a socket, mlx, or jang. The stdlib http.server
handler is the thin IO edge that parses bytes, calls the core, and writes bytes
back. Serving and model construction load the standard runtime dependencies;
module import keeps those heavy dependencies lazy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from moespresso.core.artifact import ArtifactError, read_artifact
from moespresso.package.constants import MANIFEST_NAME
from moespresso.runtime.deepseek_v4.renderer import (
    DEEPSEEK_V4_PROMPT_RENDERER,
    DEEPSEEK_V4_RENDERER_VERSION,
    render_deepseek_v4_prompt,
)
from moespresso.runtime.generation import (
    ContextLimitError,
    GenerationResult,
    as_generation_result,
)
from moespresso.runtime.chat_stream import (
    THINK_OPEN,
    ReasoningSplitter,
    split_complete_text,
)
from moespresso.runtime.kv_policy import (
    KVPolicy,
    KVPolicyError,
    LIVE_KV_RAW,
    parse_kv_policy,
    validate_runtime_policy,
)

# --- pure core: request dict + generator -> response dict / error ---

# 2048 so normal chat/reasoning replies aren't truncated (the model is thinking-on by
# default); still overridable per request and via the CLI's --max-tokens. mlx_lm requires
# an int cap, so there is no "model default" to defer to here.
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0

# Chat-template kwargs MoEspresso applies by default for thinking-capable families.
# preserve_thinking=True keeps past <think> blocks in history so the rendered prefix is
# append-only across turns (a valid KV prefix); enable_thinking=True is the model default.
DEFAULT_TEMPLATE_KWARGS = {"enable_thinking": True, "preserve_thinking": True}
PACKAGE_MANIFEST_NAME = MANIFEST_NAME
DEFAULT_PROMPT_CACHE_SIZE = 10
STARTUP_WARMUP_MAX_TOKENS = 4
STARTUP_WARMUP_MESSAGE = "Warm up."
REQUEST_KV_POLICY_FIELDS = {
    "live_kv_format",
    "kv_group_size",
    "quantized_kv_start",
    "prompt_cache_size",
    "prompt_cache_bytes",
}


class RequestError(Exception):
    """A malformed request. Carries the HTTP status to return."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class ClientDisconnected(ConnectionError):
    """The peer closed a streaming response while generation was active."""


def request_stream_options(request: dict) -> tuple[bool, bool]:
    """Validate and return ``(stream, include_usage)``."""
    stream = request.get("stream", False)
    if not isinstance(stream, bool):
        raise RequestError(400, "stream must be a boolean")
    options = request.get("stream_options")
    if options is None:
        return stream, False
    if not stream:
        raise RequestError(400, "stream_options requires stream=true")
    if not isinstance(options, dict):
        raise RequestError(400, "stream_options must be a JSON object")
    unknown = sorted(set(options) - {"include_usage"})
    if unknown:
        raise RequestError(
            400, f"unknown stream_options field(s): {', '.join(unknown)}")
    include_usage = options.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise RequestError(400, "stream_options.include_usage must be a boolean")
    return stream, include_usage


def is_deepseek_v4_renderer(prompt_renderer: str | None) -> bool:
    return prompt_renderer == DEEPSEEK_V4_PROMPT_RENDERER


def is_deepseek_v4_manifest(manifest: dict) -> bool:
    architecture = manifest.get("architecture") or {}
    return (
        architecture.get("family") == "deepseek_v4_flash"
        or is_deepseek_v4_renderer(architecture.get("prompt_renderer"))
    )


def deepseek_v4_contract_template_kwargs(thinking: str | None = None) -> dict:
    """DS4 render kwargs for a `--thinking` selection (off, on, high, max).

    `off` (the default) renders chat mode; `on` and `high` render thinking mode
    (prompt-identical in the official encoder); `max` additionally injects the
    official max reasoning-effort preamble into the first message. Every
    selection keeps history append-only (`preserve_thinking`/`drop_thinking`),
    so a growing conversation re-renders as an exact extension of its previous
    prompt and the KV prefix stays valid across turns.
    """
    selection = "on" if thinking == "high" else (thinking or "off")
    if selection not in ("off", "on", "max"):
        raise ValueError(f"invalid DeepSeek-V4 thinking selection: {thinking!r}")
    kwargs = {
        "enable_thinking": selection != "off",
        "preserve_thinking": True,
        "drop_thinking": False,
    }
    if selection == "max":
        kwargs["reasoning_effort"] = "max"
    return kwargs


def _request_template_kwargs(
    request: dict,
    *,
    prompt_renderer: str | None = None,
) -> dict:
    raw = request.get("chat_template_kwargs")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RequestError(400, "chat_template_kwargs must be a JSON object")
    if is_deepseek_v4_renderer(prompt_renderer) and raw:
        joined = ", ".join(sorted(raw))
        raise RequestError(
            400,
            "DeepSeek-V4 owns its render policy as part of the "
            f"cache/attention contract; chat_template_kwargs are not "
            f"request options: {joined}",
        )
    return raw


SUPPORTED_SAMPLING_FIELDS = (
    "temperature, top_p, top_k, min_p, presence_penalty, max_tokens"
)


def _sampling_number(request: dict, name: str) -> float | None:
    value = request.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(400, f"{name} must be a number")
    return float(value)


def _request_sampling_kwargs(request: dict) -> dict:
    """Optional sampling knobs forwarded to generation.

    Sampling parameters are generation-only: none of them enter any cache
    identity (prefix reuse keys on the package, the rendering identity, and
    the KV policy alone), so a client may vary them turn over turn on one
    session. Only fields present in the request are forwarded, which keeps a
    request without them byte-identical to the pre-existing call shape.

    ``repetition_penalty`` has no runtime implementation. Its neutral value
    1.0 is accepted as a no-op; any other value is refused loudly, because a
    silently dropped sampling parameter serves output the client did not ask
    for.
    """
    out: dict = {}
    top_k = request.get("top_k")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise RequestError(400, "top_k must be a non-negative integer")
        out["top_k"] = top_k
    min_p = _sampling_number(request, "min_p")
    if min_p is not None:
        if not 0.0 <= min_p <= 1.0:
            raise RequestError(400, "min_p must be between 0 and 1")
        out["min_p"] = min_p
    presence_penalty = _sampling_number(request, "presence_penalty")
    if presence_penalty is not None:
        out["presence_penalty"] = presence_penalty
    repetition_penalty = _sampling_number(request, "repetition_penalty")
    if repetition_penalty is not None and repetition_penalty != 1.0:
        raise RequestError(
            400,
            "repetition_penalty is not implemented (only the neutral value "
            f"1.0 is accepted); supported sampling parameters: "
            f"{SUPPORTED_SAMPLING_FIELDS}",
        )
    return out


def _request_session_cache_key(request: dict) -> str | None:
    """Read the optional client session cache key from ``metadata``.

    A client may send ``metadata.moespresso_cache_key`` to group its requests for
    disk-cache eviction preference. It is an index hint only: it is stored on the
    checkpoint entry and never enters the safety key, so it can never authorize a
    load. A request without it behaves exactly as before. The value must be a
    string when present; any other type is a malformed request.
    """
    metadata = request.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise RequestError(400, "metadata must be a JSON object")
    key = metadata.get("moespresso_cache_key")
    if key is None:
        return None
    if not isinstance(key, str):
        raise RequestError(400, "metadata.moespresso_cache_key must be a string")
    return key


def effective_template_kwargs(
    template_kwargs: dict | None = None,
    *,
    prompt_renderer: str | None = None,
) -> dict:
    """Resolve template kwargs after model-family contracts are applied."""
    caller_kwargs = template_kwargs or {}
    if is_deepseek_v4_renderer(prompt_renderer):
        # A partial dict consistent with one legal mode normalizes to that
        # mode's full contract shape ({"enable_thinking": False} is the
        # internal probe/capture shorthand for the off mode). Anything that
        # matches no mode is a contract violation and refuses.
        allowed = [
            deepseek_v4_contract_template_kwargs(selection)
            for selection in ("off", "on", "max")
        ]
        for shape in allowed:
            if all(shape.get(key) == value for key, value in caller_kwargs.items()):
                return shape
        forbidden = sorted(
            key
            for key, value in caller_kwargs.items()
            if not any(shape.get(key) == value for shape in allowed)
        )
        joined = ", ".join(forbidden or sorted(caller_kwargs))
        raise RequestError(
            400,
            "DeepSeek-V4 owns these render fields as part of its "
            f"cache/attention contract; they are not user options: {joined}",
        )
    kwargs = {**DEFAULT_TEMPLATE_KWARGS, **(template_kwargs or {})}
    return kwargs


def rendering_identity(
    rendering_id: str | None,
    template_kwargs: dict | None = None,
    *,
    prompt_renderer: str | None = None,
) -> str:
    """The effective rendering identity a prefix cache must key on.

    The rendered token stream is determined by both the tokenizer/template files (captured by
    the convert-time `rendering_id` file hash, which includes the installed chat template) AND
    the runtime template kwargs (enable_thinking/preserve_thinking change the tokens). This
    combines them into one stable id so a cache built under one render policy is never reused
    under another. Order-independent in the kwargs; deterministic.
    """
    kwargs = effective_template_kwargs(
        template_kwargs,
        prompt_renderer=prompt_renderer,
    )
    renderer_version = (
        DEEPSEEK_V4_RENDERER_VERSION
        if is_deepseek_v4_renderer(prompt_renderer)
        else prompt_renderer
    )
    payload = json.dumps(
        {
            "rendering_id": rendering_id,
            "template_kwargs": kwargs,
            "prompt_renderer": renderer_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def effective_kv_policy(
    request: dict,
    *,
    prompt_renderer: str | None = None,
) -> KVPolicy:
    """Resolve the live KV policy after model-family contracts are applied."""
    if is_deepseek_v4_renderer(prompt_renderer):
        forbidden = sorted(REQUEST_KV_POLICY_FIELDS & set(request))
        if forbidden:
            joined = ", ".join(forbidden)
            raise RequestError(
                400,
                "DeepSeek-V4 owns live KV/cache policy as part of its "
                f"attention contract; these are not request options: {joined}",
            )
        return parse_kv_policy({"live_kv_format": LIVE_KV_RAW})
    return parse_kv_policy(request)


def render_prompt(
    messages: list[dict],
    tokenizer=None,
    *,
    template_kwargs: dict | None = None,
    prompt_renderer: str | None = None,
    tools: list[dict] | None = None,
    response_format: dict | None = None,
) -> str:
    """Turn chat messages into a prompt string: the single place templating happens.

    Applies the tokenizer chat template once with MoEspresso's thinking defaults
    (DEFAULT_TEMPLATE_KWARGS), then applies family-specific render contracts.
    Generation must not template again (see serve.generate_once). Falls back to a
    plain role-tagged concatenation when there is no tokenizer, so the pure core
    stays usable in tests.

    The request's top-level `tools` and `response_format` reach the template on
    every branch. The vendored Qwen template renders tools as the leading system
    block, ahead of the message loop, so a tool set held constant across turns
    lands in the shared prefix and history stays append-only.
    """
    kwargs = effective_template_kwargs(
        template_kwargs,
        prompt_renderer=prompt_renderer,
    )
    if is_deepseek_v4_renderer(prompt_renderer):
        return render_deepseek_v4_prompt(
            messages,
            template_kwargs=kwargs,
            tools=tools,
            response_format=response_format,
        )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        # Set the tool kwargs only when present, so a tool-free request keeps
        # its exact rendered prompt (the KV prefix identity existing sessions
        # rely on). The top-level request fields overwrite any same-named
        # chat_template_kwargs entry, so apply_chat_template never receives a
        # duplicate keyword.
        if tools is not None:
            kwargs["tools"] = tools
        if response_format is not None:
            kwargs["response_format"] = response_format
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs)
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}"
                     for m in messages) + "\nassistant:"


def run_startup_warmup(
    model,
    tokenizer,
    *,
    prompt_renderer: str | None = None,
    server_template_kwargs: dict | None = None,
    generate_fn: Callable | None = None,
    clock: Callable[[], float] | None = None,
) -> float:
    """Prime generation before readiness without publishing synthetic state.

    The first generation in a freshly loaded MLX process can pay model wiring
    and kernel/graph setup inside time to first token. Serving primes those
    paths with one small deterministic generation. The direct generation seam
    bypasses ``PrefixCacheGenerator``: its prompt cache dies with this call, no
    memory or disk KV key is inserted, and synthetic expert demand is not
    persisted. In-memory expert residency and runtime counters may still
    reflect the prime.
    """
    if generate_fn is None:
        from moespresso.runtime.serve import generate_with_metadata

        generate_fn = generate_with_metadata
    if clock is None:
        import time

        clock = time.perf_counter
    prompt = render_prompt(
        [{"role": "user", "content": STARTUP_WARMUP_MESSAGE}],
        tokenizer,
        template_kwargs=server_template_kwargs,
        prompt_renderer=prompt_renderer,
    )
    kv_policy = effective_kv_policy({}, prompt_renderer=prompt_renderer)
    started = clock()
    generate_fn(
        model,
        tokenizer,
        prompt,
        prompt_cache=None,
        kv_policy=kv_policy,
        max_tokens=STARTUP_WARMUP_MAX_TOKENS,
        temperature=0.0,
        top_p=1.0,
        persist_expert_demand=False,
    )
    return clock() - started


def chat_completion(
    request: dict,
    generate: Callable[..., str | GenerationResult],
    *,
    model_id: str = "moespresso",
    tokenizer=None,
    rendering_id: str | None = None,
    prompt_renderer: str | None = None,
    server_template_kwargs: dict | None = None,
    created: int = 0,
    ready_callback: Callable[[], None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    delta_callback: Callable[[str, str], None] | None = None,
) -> dict:
    """Pure chat-completions handler: validate, render, generate, shape.

    `generate(prompt, max_tokens, temperature, top_p, kv_policy) -> str|GenerationResult`
    is injected (the real one is a closure over the loaded model; tests pass a fake).
    `created` is supplied by the caller (no wall-clock read in the core). Raises
    RequestError(400) on a malformed body.
    """
    request_stream_options(request)
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError(400, "request must include a non-empty 'messages' list")
    for m in messages:
        if not isinstance(m, dict) or "content" not in m:
            raise RequestError(400, "each message needs a 'content' field")

    try:
        kv_policy = effective_kv_policy(
            request,
            prompt_renderer=prompt_renderer,
        )
        validate_runtime_policy(kv_policy)
    except KVPolicyError as e:
        raise RequestError(400, str(e)) from e

    # precedence: module defaults < package/family contract < server launch flags
    # < per-request kwargs. DS4 removes the request layer entirely: its prompt mode
    # is fixed by the package/runtime contract and cannot be changed by callers.
    contract_template_kwargs = (
        deepseek_v4_contract_template_kwargs()
        if is_deepseek_v4_renderer(prompt_renderer)
        else {}
    )
    template_kwargs = {
        **contract_template_kwargs,
        **(server_template_kwargs or {}),
        **_request_template_kwargs(request, prompt_renderer=prompt_renderer),
    }
    resolved_template_kwargs = effective_template_kwargs(
        template_kwargs, prompt_renderer=prompt_renderer)
    thinking_enabled = bool(
        resolved_template_kwargs.get("enable_thinking", True))
    effective_rendering_id = rendering_identity(
        rendering_id,
        template_kwargs,
        prompt_renderer=prompt_renderer,
    )
    prompt = render_prompt(
        messages,
        tokenizer,
        template_kwargs=template_kwargs,
        prompt_renderer=prompt_renderer,
        tools=request.get("tools"),
        response_format=request.get("response_format"),
    )
    # An over-limit request (prompt tokens plus max_tokens past the model's
    # declared context limit) is a client error. The generator refuses it before
    # any cache access and the refusal names the
    # limit and both request-side counts. The prompt that fits but whose
    # max_tokens overruns is refused identically rather than clamped, so the
    # completion budget a client asked for is never silently shrunk.
    splitter = None
    response_callback = None
    if delta_callback is not None:
        splitter = ReasoningSplitter(
            thinking_enabled=thinking_enabled,
            emit=delta_callback,
        )

        def response_callback(_step: int, response: object) -> None:
            splitter.push(str(getattr(response, "text", "")))

    generate_kwargs = {
        "max_tokens": int(request.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "temperature": float(request.get("temperature", DEFAULT_TEMPERATURE)),
        "top_p": float(request.get("top_p", DEFAULT_TOP_P)),
        "kv_policy": kv_policy,
        "effective_rendering_id": effective_rendering_id,
        "session_cache_key": _request_session_cache_key(request),
        **_request_sampling_kwargs(request),
    }
    if ready_callback is not None:
        generate_kwargs["ready_callback"] = ready_callback
    if progress_callback is not None:
        generate_kwargs["progress_callback"] = progress_callback
    if response_callback is not None:
        generate_kwargs["response_callback"] = response_callback

    try:
        generated = as_generation_result(generate(
            prompt,
            **generate_kwargs,
        ))
    except ContextLimitError as e:
        raise RequestError(400, str(e)) from e
    if splitter is not None:
        if not splitter.saw_input:
            splitter.push(generated.text)
        splitter.finish()
        reasoning, content = splitter.reasoning, splitter.content
    else:
        reasoning, content = split_complete_text(
            generated.text, thinking_enabled=thinking_enabled,
            prompt_opened_thinking=prompt.rstrip().endswith(THINK_OPEN))
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning

    response = {
        "id": "chatcmpl-moespresso",
        "object": "chat.completion",
        "created": created,
        "model": request.get("model", model_id),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": generated.finish_reason,
        }],
    }
    if generated.prompt_tokens is not None and generated.completion_tokens is not None:
        response["usage"] = {
            "prompt_tokens": generated.prompt_tokens,
            "completion_tokens": generated.completion_tokens,
            "total_tokens": generated.prompt_tokens + generated.completion_tokens,
        }
        if generated.cached_tokens is not None:
            response["usage"]["prompt_tokens_details"] = {
                "cached_tokens": generated.cached_tokens,
            }
        if generated.cache_event is not None:
            prompt_cache_block = {
                "event": generated.cache_event,
                "entries": generated.cache_entries,
                "bytes": generated.cache_bytes,
            }
            # Surface the disk event on the request when one occurred: a disk
            # restore already shows as the ``disk_hit`` event, and a request that
            # wrote frontier checkpoints reports how many.
            if generated.disk_checkpoints_written:
                prompt_cache_block["disk_checkpoints_written"] = (
                    generated.disk_checkpoints_written
                )
            response["usage"]["prompt_cache"] = prompt_cache_block
        # first-token latency is the headline serve metric:
        # surface it on every response instead of making users probe for it.
        if generated.first_token_seconds is not None:
            tps = None
            if generated.completion_tokens and generated.generation_seconds:
                tps = round(generated.completion_tokens
                            / generated.generation_seconds, 2)
            response["usage"]["moespresso"] = {
                "first_token_seconds": round(generated.first_token_seconds, 4),
                "generation_seconds": round(
                    generated.generation_seconds or 0.0, 4),
                "generation_tps": tps,
            }
    return response


# --- IO edge: stdlib http.server handler bound to a loaded model ---

def serialized_generator(generate: Callable[..., str | GenerationResult], lock):
    """Serialize model/cache mutation across request handlers.

    The loaded model and the prompt-cache store are shared mutable objects. A
    handler must hold one lock across fetch/generate/insert: the store hands a
    matched entry out by move (no copy), so the whole
    fetch-mutate-reinsert span is one critical section. The server runs
    single-threaded (see serve()), so the lock is redundant there; it stays as
    the guard the shared state requires of any other embedding of these
    closures.
    """
    def generate_serialized(prompt, **opts):
        with lock:
            return generate(prompt, **opts)
    return generate_serialized


def serialized_stats(stats: Callable[[], dict], lock):
    """Read shared cache/model-adjacent state under the same serve lock."""
    def stats_serialized():
        with lock:
            return stats()
    return stats_serialized


class _ChatSSEWriter:
    """Write one Chat Completions SSE response."""

    def __init__(self, handler: BaseHTTPRequestHandler, *, model: str,
                 created: int, include_usage: bool):
        self.handler = handler
        self.model = model
        self.created = created
        self.include_usage = include_usage
        self.started = False

    def _write(self, payload: bytes) -> None:
        try:
            self.handler.wfile.write(payload)
            self.handler.wfile.flush()
        except OSError as e:
            raise ClientDisconnected(str(e)) from e

    def _chunk(self, choices: list[dict], *, usage=None) -> dict:
        payload = {
            "id": "chatcmpl-moespresso",
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
        }
        if self.include_usage:
            payload["usage"] = usage
        return payload

    def _event(self, payload: dict) -> None:
        data = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._write(b"data: " + data + b"\n\n")

    def start(self) -> None:
        if self.started:
            return
        self.handler.send_response(200)
        self.handler.send_header("Content-Type", "text/event-stream")
        self.handler.send_header("Cache-Control", "no-cache")
        self.handler.send_header("Connection", "close")
        self.handler.end_headers()
        self.handler.close_connection = True
        self.started = True
        self._event(self._chunk([{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }]))

    def progress(self, _processed: int, _total: int) -> None:
        self.start()
        self._write(b": prefill\n\n")

    def delta(self, kind: str, text: str) -> None:
        if not text:
            return
        self.start()
        field = "reasoning_content" if kind == "reasoning" else "content"
        self._event(self._chunk([{
            "index": 0,
            "delta": {field: text},
            "finish_reason": None,
        }]))

    def finish(self, response: dict) -> None:
        self.start()
        choice = response["choices"][0]
        self._event(self._chunk([{
            "index": 0,
            "delta": {},
            "finish_reason": choice.get("finish_reason"),
        }]))
        if self.include_usage:
            self._event(self._chunk([], usage=response.get("usage") or {}))
        self._write(b"data: [DONE]\n\n")

    def error(self, message: str) -> None:
        if not self.started:
            return
        self._event({"error": {"message": message, "type": "server_error"}})


def build_cache_generator(
    model,
    tokenizer,
    manifest: dict,
    *,
    context_limit: int | None = None,
    prompt_cache_size: int = 10,
    prompt_cache_bytes: int | None = None,
    memory_store_factory: Callable | None = None,
    disk_store=None,
):
    """Build MoEspresso's in-memory prompt-cache generator from server-owned cache config.

    ``disk_store`` is the optional two-tier disk checkpoint store. When present,
    the generator consults it on an in-memory miss and inserts the mutated cache
    into memory only; when absent (the default), the generator is memory-only.
    """
    from moespresso.runtime.prefix_cache import (
        PrefixCacheGenerator,
        effective_context_limit,
        make_prompt_cache_store,
    )

    if memory_store_factory is None:
        memory_store_factory = make_prompt_cache_store
    try:
        from moespresso.runtime.ssd_streaming_build import (
            maybe_adapt_ssd_streaming_capacity,
        )
    except ImportError:
        maybe_adapt_ssd_streaming_capacity = None

    return PrefixCacheGenerator(
        model,
        tokenizer,
        manifest,
        memory_store_factory(prompt_cache_size, prompt_cache_bytes),
        after_generate_fn=maybe_adapt_ssd_streaming_capacity,
        disk_store=disk_store,
        context_limit=(
            effective_context_limit(manifest)
            if context_limit is None
            else int(context_limit)
        ),
    )


def make_handler(
    generate: Callable[..., str],
    *,
    model_id: str,
    tokenizer=None,
    rendering_id: str | None = None,
    prompt_renderer: str | None = None,
    server_template_kwargs: dict | None = None,
    stats: Callable[[], dict] | None = None,
    runtime_stats: Callable[[], dict] | None = None,
    clock: Callable[[], int] = lambda: 0,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to a loaded generator.

    `generate` and `tokenizer` come from the loaded model; `clock` supplies the
    `created` timestamp (injected so the core stays clock-free and tests are
    deterministic). Routes: GET /health, POST /v1/chat/completions.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib dispatch name)
            if self.path == "/health":
                payload = {"status": "ok", "model": model_id}
                if stats is not None:
                    payload["prompt_cache"] = stats()
                if runtime_stats is not None:
                    payload["ssd_streaming"] = runtime_stats()
                self._send_json(200, payload)
            else:
                self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "not found"}})
                return
            stream_writer = None
            try:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(request, dict):
                    raise RequestError(400, "request body must be a JSON object")
                stream, include_usage = request_stream_options(request)
                created = clock()
                if stream:
                    stream_writer = _ChatSSEWriter(
                        self,
                        model=request.get("model", model_id),
                        created=created,
                        include_usage=include_usage,
                    )
                resp = chat_completion(
                    request, generate, model_id=model_id,
                    tokenizer=tokenizer, rendering_id=rendering_id,
                    prompt_renderer=prompt_renderer,
                    server_template_kwargs=server_template_kwargs,
                    created=created,
                    ready_callback=(stream_writer.start if stream_writer else None),
                    progress_callback=(
                        stream_writer.progress if stream_writer else None),
                    delta_callback=(stream_writer.delta if stream_writer else None),
                )
            except ClientDisconnected:
                return
            except RequestError as e:
                if stream_writer is not None and stream_writer.started:
                    try:
                        stream_writer.error(e.message)
                    except ClientDisconnected:
                        pass
                else:
                    self._send_json(e.status, {"error": {"message": e.message}})
            except json.JSONDecodeError:
                self._send_json(400, {"error": {"message": "invalid JSON body"}})
            except Exception as e:
                if stream_writer is None or not stream_writer.started:
                    raise
                try:
                    stream_writer.error(str(e))
                except ClientDisconnected:
                    pass
            else:
                if stream_writer is not None:
                    try:
                        stream_writer.finish(resp)
                    except ClientDisconnected:
                        pass
                else:
                    self._send_json(200, resp)

        def log_message(self, *args) -> None:  # quiet by default
            pass

    return Handler


def normalize_thinking_selection(thinking) -> str | None:
    """Normalize a `--thinking` selection to off, on, max, or None.

    Booleans keep their legacy programmatic meaning (True is on, False is
    off) and `high` is an alias of `on` for every family. Anything else
    refuses loudly instead of serving a silent default.
    """
    if thinking is None:
        return None
    if isinstance(thinking, bool):
        return "on" if thinking else "off"
    if thinking == "high":
        return "on"
    if thinking in ("off", "on", "max"):
        return thinking
    raise ValueError(f"invalid thinking selection: {thinking!r}")


def thinking_effort_option_error(
    thinking: str | None,
    *,
    is_deepseek_v4: bool,
) -> str | None:
    """Refuse `--thinking max` for families without an effort mechanism.

    `max` maps to DeepSeek-V4's official reasoning-effort preamble. No other
    ported family has an effort mechanism, so the flag refuses loudly there
    instead of silently serving plain thinking mode.
    """
    if thinking != "max" or is_deepseek_v4:
        return None
    return (
        "FAILED: --thinking max is a DeepSeek-V4 reasoning-effort level; "
        "this model family has no reasoning-effort mechanism."
    )


def _preflight_manifest_for_cli(package_dir: Path) -> dict | None:
    """Best-effort manifest read for CLI option gating before heavy model load."""
    manifest_path = Path(package_dir) / PACKAGE_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        return read_artifact(manifest_path)
    except (OSError, json.JSONDecodeError, ArtifactError):
        return None


def serve(
    package_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    prompt_cache_size: int | None = None,
    prompt_cache_bytes: int | None = None,
    max_context_tokens: int | None = None,
    min_resident_experts: int | None = None,
    thinking: str | None = None,
    startup_warmup: bool = True,
    load_model_fn: Callable | None = None,
    startup_warmup_fn: Callable | None = None,
) -> int:
    """Load and warm the package once, then serve OpenAI-compatible HTTP.

    Startup builds the model straight from the manifest and primes generation
    before readiness; it does not verify (run moespresso-verify for the integrity
    gate). Needs the `compute` extra; imports are lazy so the pure core stays
    importable without mlx.
    """
    import threading
    import time

    if load_model_fn is None:
        from moespresso.runtime.serve import load_served_model as load_model_fn

    thinking = normalize_thinking_selection(thinking)
    package_dir = Path(package_dir)
    print(f"Loading package: {package_dir}")
    cache_generator = None
    disk_store = None
    httpd = None

    # Acquire the disk KV root lock before the model load: a second owner must be
    # refused before the heavy load. Off by default; enabled only by
    # MOESPRESSO_DISK_KV=frontier with a root and a valid stride.
    from moespresso.runtime.disk_kv import (
        DiskKVError,
        open_disk_store,
        resolve_disk_kv_config,
    )
    from moespresso.runtime.streaming_capacity import StreamingCapacityError

    try:
        disk_kv_config = resolve_disk_kv_config()
        disk_store = open_disk_store(disk_kv_config)
    except DiskKVError as e:
        print(f"FAILED: {e}")
        return 2
    if disk_store is not None:
        print(
            f"[serve] disk_kv=frontier root={disk_kv_config.root} "
            f"stride={disk_kv_config.stride}",
            flush=True,
        )

    try:
        try:
            load_kwargs = {}
            if min_resident_experts is not None:
                load_kwargs["min_resident_experts"] = int(min_resident_experts)
            model, tokenizer, manifest = load_model_fn(package_dir, **load_kwargs)
        except (FileNotFoundError, StreamingCapacityError) as e:
            # PackageNotFoundError and friends: one clear line, no traceback
            print(f"FAILED: {e}")
            return 2
        model_id = manifest["subject"].get("source_root", "moespresso")
        from moespresso.runtime.prefix_cache import (
            declared_context_limit,
            effective_context_limit,
        )

        try:
            context_limit = effective_context_limit(
                manifest,
                requested=max_context_tokens,
            )
        except ValueError as e:
            print(f"FAILED: {e}", flush=True)
            return 2
        print(
            f"[serve] context_limit={context_limit} "
            f"package_limit={declared_context_limit(manifest) or 'unknown'}",
            flush=True,
        )

        resolved_prompt_cache_size = (
            DEFAULT_PROMPT_CACHE_SIZE
            if prompt_cache_size is None
            else prompt_cache_size
        )

        # --thinking off|on|high|max: resolve to this family's mechanism before
        # binding the socket. An unsupported selection refuses loudly at
        # startup, never silently serves the template default.
        ds4_contract = is_deepseek_v4_manifest(manifest)
        option_error = thinking_effort_option_error(
            thinking, is_deepseek_v4=ds4_contract)
        if option_error is not None:
            print(option_error, flush=True)
            return 2
        server_template_kwargs = None
        if ds4_contract:
            server_template_kwargs = deepseek_v4_contract_template_kwargs(thinking)
            effective = "max" if thinking == "max" else (
                "on" if server_template_kwargs["enable_thinking"] else "off")
            print(f"[serve] thinking={effective} via=deepseek_v4_contract",
                  flush=True)
        elif thinking is not None:
            from moespresso.runtime.thinking import resolve_thinking_kwargs
            server_template_kwargs = resolve_thinking_kwargs(
                tokenizer, thinking=thinking == "on",
                family=manifest.get("architecture", {}).get("family"))
            print(f"[serve] thinking={thinking} "
                  f"via={server_template_kwargs}", flush=True)
        else:
            print("[serve] thinking=template-default", flush=True)

        if startup_warmup:
            warmup = startup_warmup_fn or run_startup_warmup
            print(
                "[serve] warming up generation; server is not ready",
                flush=True,
            )
            try:
                warmup_seconds = warmup(
                    model,
                    tokenizer,
                    prompt_renderer=manifest.get("architecture", {}).get(
                        "prompt_renderer"),
                    server_template_kwargs=server_template_kwargs,
                )
            except Exception as e:  # noqa: BLE001 - fail before readiness
                print(f"FAILED: startup warmup: {e}", flush=True)
                return 2
            print(
                f"[serve] startup warmup {warmup_seconds:.2f}s; generation ready",
                flush=True,
            )

        print(f"  loaded ({len(manifest['tensors'])} tensors); serving on "
              f"http://{host}:{port}")

        cache_generator = build_cache_generator(
            model,
            tokenizer,
            manifest,
            context_limit=context_limit,
            prompt_cache_size=resolved_prompt_cache_size,
            prompt_cache_bytes=prompt_cache_bytes,
            disk_store=disk_store,
        )

        def generate(prompt, **opts):
            return cache_generator(prompt, **opts)

        def runtime_stats():
            from moespresso.runtime.ssd_streaming_build import (
                SSDStreamingBuildError,
                ssd_streaming_stats,
            )
            try:
                return ssd_streaming_stats(model)
            except SSDStreamingBuildError:
                return {"enabled": False}

        serve_lock = threading.Lock()
        handler = make_handler(
            serialized_generator(generate, serve_lock),
            model_id=model_id, tokenizer=tokenizer,
            rendering_id=manifest.get("tokenizer", {}).get("rendering_id"),
            prompt_renderer=manifest.get("architecture", {}).get("prompt_renderer"),
            server_template_kwargs=server_template_kwargs,
            stats=serialized_stats(cache_generator.cache_stats, serve_lock),
            runtime_stats=serialized_stats(runtime_stats, serve_lock),
            clock=lambda: int(time.time()))
        # Single-threaded on purpose: every request is handled on the thread
        # that loaded the model. The pinned MLX gives each thread its own
        # default stream, and the load path leaves deliberately lazy arrays
        # (deferred weight transforms) recorded on the loading thread's
        # stream; forcing one from a per-connection worker thread raises
        # "There is no Stream(gpu, 0) in current thread" and kills the
        # process on the first request from a fresh connection. Generation is
        # serialized by design, so a threaded server buys nothing: concurrent
        # requests queue in the listen backlog instead of on the serve lock.
        # An idle keep-alive connection would otherwise hold the single
        # handler loop, so idle connections time out and close.
        handler.timeout = 10.0
        httpd = HTTPServer((host, port), handler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        if httpd is not None:
            httpd.server_close()
        if cache_generator is not None:
            cache_generator.close()
        if disk_store is not None:
            disk_store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """`uv run moespresso-serve <package_dir> [--host H --port P]` (not a bare exec)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="moespresso-serve",
        description="Load a MoEspresso package from its manifest, then serve "
                    "OpenAI-compatible HTTP (run moespresso-verify for the sha256 gate).")
    parser.add_argument("package_dir", help="Path to the packaged model directory")
    parser.add_argument("--max-memory-gb", type=float, default=None,
                        help="Set the streamed runtime's startup capacity-planner "
                             "ceiling (GB). This selects expert-pool geometry; "
                             "RSS remains a separate process measurement.")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Maximum prompt-plus-output context tokens. Default: 131072 or "
             "the package limit, whichever is smaller.",
    )
    parser.add_argument(
        "--min-resident-experts",
        type=int,
        default=None,
        help="Require at least this many resident routed experts per layer; "
             "startup fails when the planned pool is smaller.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--prompt-cache-size", type=int, default=None,
                        help="Maximum resident in-memory prompt-cache entries "
                             "(host resource bound, any family)")
    parser.add_argument("--prompt-cache-bytes", type=int, default=None,
                        help="Maximum resident in-memory prompt-cache bytes "
                             "(host resource bound, any family)")
    parser.add_argument("--thinking", choices=("off", "on", "high", "max"),
                        default=None,
                        help="Select the model family's own thinking mode: "
                             "off, on (high is the same), or max (DeepSeek-V4 "
                             "reasoning-effort preamble; DeepSeek-V4 only). "
                             "Refuses at startup if the family has no "
                             "mechanism. Default: the template's own default "
                             "(DeepSeek-V4: off). Per-request render fields "
                             "remain enforced by the model contract.")
    parser.add_argument(
        "--startup-warmup",
        choices=("auto", "off"),
        default="auto",
        help="Prime model wiring and generation before announcing readiness "
             "(default: auto; off restores lazy first-request setup)",
    )
    args = parser.parse_args(argv)
    if args.max_context_tokens is not None and args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be >= 1")
    if args.min_resident_experts is not None and args.min_resident_experts < 1:
        parser.error("--min-resident-experts must be >= 1")
    if args.max_memory_gb is not None:
        import os as _os_cap
        _os_cap.environ["MOESPRESSO_SSD_MAX_MEMORY_GB"] = str(args.max_memory_gb)
    manifest = _preflight_manifest_for_cli(Path(args.package_dir))
    if manifest is not None:
        from moespresso.runtime.prefix_cache import effective_context_limit

        try:
            effective_context_limit(
                manifest,
                requested=args.max_context_tokens,
            )
        except ValueError as e:
            parser.error(str(e))
        option_error = thinking_effort_option_error(
            args.thinking, is_deepseek_v4=is_deepseek_v4_manifest(manifest))
        if option_error is not None:
            print(option_error, flush=True)
            return 2
    return serve(
        Path(args.package_dir),
        host=args.host,
        port=args.port,
        prompt_cache_size=args.prompt_cache_size,
        prompt_cache_bytes=args.prompt_cache_bytes,
        max_context_tokens=args.max_context_tokens,
        min_resident_experts=args.min_resident_experts,
        thinking=args.thinking,
        startup_warmup=args.startup_warmup != "off",
    )


if __name__ == "__main__":
    raise SystemExit(main())
