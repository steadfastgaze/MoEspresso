"""Chat-completions client for a local MoEspresso server.

One POST per turn over stdlib urllib, streaming by default. The client sends
the session cache key as
``metadata.moespresso_cache_key`` and surfaces the engine's per-request cache
evidence (``usage.prompt_cache`` and ``usage.prompt_tokens_details.
cached_tokens``) so a caller can assert prefix reuse turn over turn. Optional
request fields (tools, response_format, sampling knobs) are included only
when set: an absent tools field keeps the tool-free render identity, and the
server's defaults stay authoritative for the rest.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from moespresso.agentlib.conversation import Conversation
from moespresso.agentlib.sse import SSEError, iter_sse_data

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_SECONDS = 600.0


class ClientError(Exception):
    """An HTTP-level failure. Carries the status (0 for transport errors)."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}" if status else message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class ChatCompletion:
    """The parsed pieces of one chat-completions response."""

    message: dict
    finish_reason: str | None
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def content(self) -> str | None:
        return self.message.get("content")

    @property
    def reasoning_content(self) -> str | None:
        return self.message.get("reasoning_content")

    @property
    def prompt_cache(self) -> dict | None:
        """The engine's per-request cache evidence block, when present."""
        return self.usage.get("prompt_cache")

    @property
    def cached_tokens(self) -> int | None:
        details = self.usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            return None
        return details.get("cached_tokens")


class CompletionsClient:
    """Client for ``POST /v1/chat/completions`` on a MoEspresso server."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, *,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 model: str | None = None,
                 stream: bool = True):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.stream = stream

    def complete(
        self,
        conversation: Conversation | list[dict],
        *,
        session_cache_key: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        chat_template_kwargs: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool | None = None,
        verbatim_tool_calls: bool = False,
        on_start: Callable[[], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_content: Callable[[str], None] | None = None,
    ) -> ChatCompletion:
        """Send one completion request and parse the response.

        Accepts a ``Conversation`` (its messages and session cache key are
        used; an explicit ``session_cache_key`` argument wins) or a plain
        message list. ``verbatim_tool_calls`` sends
        ``metadata.moespresso_tool_calls: "verbatim"``, opting the request
        out of served tool-call parsing so the completion text carries the
        raw dialect emission; the client-side dialect parsers depend on it.
        """
        if isinstance(conversation, Conversation):
            messages = conversation.request_messages()
            if session_cache_key is None:
                session_cache_key = conversation.session_cache_key
        else:
            messages = list(conversation)

        body: dict = {"messages": messages}
        if self.model is not None:
            body["model"] = self.model
        metadata: dict = {}
        if session_cache_key is not None:
            metadata["moespresso_cache_key"] = session_cache_key
        if verbatim_tool_calls:
            metadata["moespresso_tool_calls"] = "verbatim"
        if metadata:
            body["metadata"] = metadata
        if tools is not None:
            body["tools"] = tools
        if response_format is not None:
            body["response_format"] = response_format
        if chat_template_kwargs is not None:
            body["chat_template_kwargs"] = chat_template_kwargs
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p

        use_stream = self.stream if stream is None else stream
        if not isinstance(use_stream, bool):
            raise ValueError("stream must be a boolean")
        body["stream"] = use_stream
        if use_stream:
            body["stream_options"] = {"include_usage": True}

        request = self._json_request("/v1/chat/completions", body)
        if use_stream:
            return self._send_completion_stream(
                request,
                on_start=on_start,
                on_reasoning=on_reasoning,
                on_content=on_content,
            )
        response = self._send(request)
        choices = response.get("choices") or []
        if not choices:
            raise ClientError(0, "response carries no choices")
        choice = choices[0]
        return ChatCompletion(
            message=choice.get("message") or {},
            finish_reason=choice.get("finish_reason"),
            usage=response.get("usage") or {},
            raw=response,
        )

    def health(self) -> dict:
        """Fetch ``/health`` (cumulative engine counters, disk KV included)."""
        return self._get_json("/health")

    # --- transport ---

    def _post_json(self, path: str, body: dict) -> dict:
        return self._send(self._json_request(path, body))

    def _json_request(self, path: str, body: dict) -> urllib.request.Request:
        return urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _get_json(self, path: str) -> dict:
        request = urllib.request.Request(self.base_url + path, method="GET")
        return self._send(request)

    def _send(self, request: urllib.request.Request) -> dict:
        with self._open(request) as resp:
            payload = resp.read()
        try:
            return json.loads(payload)
        except json.JSONDecodeError as e:
            raise ClientError(0, f"response is not JSON: {e}") from e

    def _open(self, request: urllib.request.Request):
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise ClientError(e.code, detail) from e
        except urllib.error.URLError as e:
            raise ClientError(0, f"cannot reach server: {e.reason}") from e
        except (http.client.HTTPException, ConnectionError, TimeoutError) as e:
            # A server process that dies mid-request surfaces as a bare
            # http.client error when the connection closes before any response.
            # Report it through the same client error type as a URLError.
            raise ClientError(0, f"transport failure: {type(e).__name__}: {e}") from e

    def _send_completion_stream(
        self,
        request: urllib.request.Request,
        *,
        on_start: Callable[[], None] | None,
        on_reasoning: Callable[[str], None] | None,
        on_content: Callable[[str], None] | None,
    ) -> ChatCompletion:
        try:
            with self._open(request) as resp:
                content_type = resp.headers.get_content_type()
                if content_type != "text/event-stream":
                    raise ClientError(
                        0, f"stream response has content type {content_type!r}")
                if on_start is not None:
                    on_start()
                return self._decode_completion_stream(
                    resp,
                    on_reasoning=on_reasoning,
                    on_content=on_content,
                )
        except SSEError as e:
            raise ClientError(0, str(e)) from e
        except json.JSONDecodeError as e:
            raise ClientError(0, f"stream event is not JSON: {e}") from e

    def _decode_completion_stream(
        self,
        lines,
        *,
        on_reasoning: Callable[[str], None] | None,
        on_content: Callable[[str], None] | None,
    ) -> ChatCompletion:
        identity: dict[str, object] = {}
        role = "assistant"
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        finish_reason = None
        usage: dict = {}
        saw_done = False
        saw_choice = False

        for data in iter_sse_data(lines):
            if data == "[DONE]":
                saw_done = True
                break
            event = json.loads(data)
            if not isinstance(event, dict):
                raise SSEError("stream event must be a JSON object")
            error = event.get("error")
            if error is not None:
                if isinstance(error, dict):
                    message = error.get("message") or json.dumps(error)
                else:
                    message = str(error)
                raise SSEError(f"server stream error: {message}")
            for identity_field in ("id", "created", "model"):
                value = event.get(identity_field)
                if value is None:
                    continue
                if (identity_field in identity
                        and identity[identity_field] != value):
                    raise SSEError(f"stream changed {identity_field}")
                identity[identity_field] = value
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices")
            if choices is None:
                continue
            if not isinstance(choices, list):
                raise SSEError("stream choices must be an array")
            if not choices:
                continue
            if len(choices) != 1 or not isinstance(choices[0], dict):
                raise SSEError("stream must carry one choice")
            saw_choice = True
            choice = choices[0]
            if choice.get("index") != 0:
                raise SSEError("stream choice index must be zero")
            finish = choice.get("finish_reason")
            if finish is not None:
                if not isinstance(finish, str):
                    raise SSEError("finish_reason must be a string or null")
                finish_reason = finish
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise SSEError("stream delta must be an object")
            if isinstance(delta.get("role"), str):
                role = delta["role"]
            reasoning = delta.get("reasoning_content", delta.get("reasoning"))
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise SSEError("reasoning delta must be a string")
                reasoning_parts.append(reasoning)
                if on_reasoning is not None:
                    on_reasoning(reasoning)
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise SSEError("content delta must be a string")
                content_parts.append(content)
                if on_content is not None:
                    on_content(content)
            calls = delta.get("tool_calls") or []
            if not isinstance(calls, list):
                raise SSEError("tool_calls delta must be an array")
            for call_delta in calls:
                self._accumulate_tool_call(tool_calls, call_delta)

        if not saw_done:
            raise SSEError("stream ended before [DONE]")
        if not saw_choice:
            raise SSEError("response carries no choices")
        message = {"role": role, "content": "".join(content_parts)}
        reasoning_content = "".join(reasoning_parts)
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        raw = {
            **identity,
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
        }
        if usage:
            raw["usage"] = usage
        return ChatCompletion(
            message=message,
            finish_reason=finish_reason,
            usage=usage,
            raw=raw,
        )

    @staticmethod
    def _accumulate_tool_call(tool_calls: dict[int, dict], delta: object) -> None:
        if not isinstance(delta, dict) or not isinstance(delta.get("index"), int):
            raise SSEError("tool call delta needs an integer index")
        index = delta["index"]
        call = tool_calls.setdefault(index, {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        })
        if isinstance(delta.get("id"), str):
            call["id"] = delta["id"]
        if isinstance(delta.get("type"), str):
            call["type"] = delta["type"]
        function = delta.get("function") or {}
        if not isinstance(function, dict):
            raise SSEError("tool call function delta must be an object")
        name = function.get("name")
        arguments = function.get("arguments")
        if name is not None:
            if not isinstance(name, str):
                raise SSEError("tool call name delta must be a string")
            call["function"]["name"] += name
        if arguments is not None:
            if not isinstance(arguments, str):
                raise SSEError("tool call arguments delta must be a string")
            call["function"]["arguments"] += arguments
