"""CompletionsClient against a fake local HTTP server (no model, no GPU).

The fake server records every request body and returns a canned response
shaped like the real serve layer (choices, usage.prompt_cache,
prompt_tokens_details). The tests pin the request contract (session cache
key under metadata, optional fields present only when set) and the response
readback (message, finish_reason, cache evidence).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from moespresso.agentlib import ClientError, CompletionsClient, Conversation

CANNED_COMPLETION = {
    "id": "chatcmpl-moespresso",
    "object": "chat.completion",
    "created": 0,
    "model": "moespresso",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "served text"},
        "finish_reason": "stop",
    }],
    "usage": {
        "prompt_tokens": 700,
        "completion_tokens": 40,
        "total_tokens": 740,
        "prompt_tokens_details": {"cached_tokens": 512},
        "prompt_cache": {
            "event": "hit",
            "entries": 3,
            "bytes": 1024,
            "disk_checkpoints_written": 2,
        },
    },
}

CANNED_HEALTH = {"status": "ok", "disk_kv": {"restores": 1, "writes": 4}}


class _ServerState:
    def __init__(self):
        self.requests = []          # (method, path, headers dict, body dict | None)
        self.status = 200
        self.completion = CANNED_COMPLETION
        self.health = CANNED_HEALTH
        self.error_body = {"error": "bad request detail"}
        self.stream_bytes = None


def _make_handler(state: _ServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            pass

        def _reply(self, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reply_stream(self, payload: dict):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            if state.stream_bytes is not None:
                self.wfile.write(state.stream_bytes)
                return
            choices = payload.get("choices") or []
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            base = {
                "id": payload.get("id", "chatcmpl-moespresso"),
                "object": "chat.completion.chunk",
                "created": payload.get("created", 0),
                "model": payload.get("model", "moespresso"),
                "usage": None,
            }
            events = []
            if choices:
                events.append({
                    **base,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
                })
            if message.get("reasoning_content"):
                events.append({
                    **base,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "reasoning_content": message["reasoning_content"],
                        },
                        "finish_reason": None,
                    }],
                })
            if "content" in message:
                content = message.get("content") or ""
                middle = len(content) // 2
                for part in (content[:middle], content[middle:]):
                    events.append({
                        **base,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": part},
                            "finish_reason": None,
                        }],
                    })
            if payload.get("choices"):
                events.append({
                    **base,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": choice.get("finish_reason"),
                    }],
                })
            if "usage" in payload:
                events.append({
                    **base,
                    "choices": [],
                    "usage": payload["usage"],
                })
            for event in events:
                data = json.dumps(event, separators=(",", ":")).encode()
                self.wfile.write(b"data: " + data + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else None
            state.requests.append(
                ("POST", self.path, dict(self.headers), body))
            if state.status == 200 and body.get("stream") is True:
                self._reply_stream(state.completion)
            else:
                self._reply(state.completion if state.status == 200
                            else state.error_body)

        def do_GET(self):
            state.requests.append(("GET", self.path, dict(self.headers), None))
            self._reply(state.health if state.status == 200
                        else state.error_body)

    return Handler


@pytest.fixture()
def fake_server():
    state = _ServerState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _last_body(state):
    return state.requests[-1][3]


def test_complete_posts_messages_to_chat_completions(fake_server):
    state, url = fake_server
    client = CompletionsClient(url)
    conv = Conversation()
    conv.add_user("hello")
    client.complete(conv)
    method, path, headers, body = state.requests[-1]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert headers["Content-Type"] == "application/json"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_session_cache_key_travels_as_metadata(fake_server):
    state, url = fake_server
    conv = Conversation(session_cache_key="road-test-1")
    conv.add_user("hi")
    CompletionsClient(url).complete(conv)
    assert _last_body(state)["metadata"] == {"moespresso_cache_key": "road-test-1"}


def test_no_metadata_without_a_session_key(fake_server):
    state, url = fake_server
    conv = Conversation()
    conv.add_user("hi")
    CompletionsClient(url).complete(conv)
    assert "metadata" not in _last_body(state)


def test_verbatim_tool_calls_travels_beside_the_cache_key(fake_server):
    state, url = fake_server
    conv = Conversation(session_cache_key="road-test-1")
    conv.add_user("hi")
    CompletionsClient(url).complete(conv, verbatim_tool_calls=True)
    assert _last_body(state)["metadata"] == {
        "moespresso_cache_key": "road-test-1",
        "moespresso_tool_calls": "verbatim",
    }


def test_explicit_session_key_wins_over_the_conversation(fake_server):
    state, url = fake_server
    conv = Conversation(session_cache_key="conv-key")
    conv.add_user("hi")
    CompletionsClient(url).complete(conv, session_cache_key="override")
    assert _last_body(state)["metadata"] == {"moespresso_cache_key": "override"}


def test_optional_fields_absent_by_default(fake_server):
    # An absent tools field keeps the tool-free render identity on the server;
    # absent sampling knobs keep the server defaults authoritative.
    state, url = fake_server
    CompletionsClient(url).complete([{"role": "user", "content": "hi"}])
    assert set(_last_body(state)) == {"messages", "stream", "stream_options"}


def test_optional_fields_forwarded_when_set(fake_server):
    state, url = fake_server
    tools = [{"type": "function", "function": {"name": "grep"}}]
    CompletionsClient(url, model="ornith").complete(
        [{"role": "user", "content": "hi"}],
        tools=tools,
        response_format={"type": "json_object"},
        chat_template_kwargs={"enable_thinking": False},
        max_tokens=64,
        temperature=0.0,
        top_p=0.9,
    )
    body = _last_body(state)
    assert body["model"] == "ornith"
    assert body["tools"] == tools
    assert body["response_format"] == {"type": "json_object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.0
    assert body["top_p"] == 0.9


def test_plain_message_list_accepted(fake_server):
    state, url = fake_server
    messages = [{"role": "user", "content": "direct"}]
    result = CompletionsClient(url).complete(messages)
    assert _last_body(state)["messages"] == messages
    assert result.content == "served text"


def test_response_readback(fake_server):
    _, url = fake_server
    conv = Conversation()
    conv.add_user("hi")
    result = CompletionsClient(url).complete(conv)
    assert result.message == {"role": "assistant", "content": "served text"}
    assert result.finish_reason == "stop"
    assert result.usage["prompt_tokens"] == 700
    assert result.raw["id"] == "chatcmpl-moespresso"


def test_non_streaming_override_uses_json_response(fake_server):
    state, url = fake_server
    result = CompletionsClient(url).complete(
        [{"role": "user", "content": "x"}], stream=False)
    assert _last_body(state)["stream"] is False
    assert "stream_options" not in _last_body(state)
    assert result.content == "served text"


def test_stream_callbacks_receive_reasoning_and_content(fake_server):
    state, url = fake_server
    state.completion = {
        **CANNED_COMPLETION,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": "considering",
                "content": "answer",
            },
            "finish_reason": "stop",
        }],
    }
    events = []
    result = CompletionsClient(url).complete(
        [{"role": "user", "content": "x"}],
        on_start=lambda: events.append(("start", "")),
        on_reasoning=lambda text: events.append(("reasoning", text)),
        on_content=lambda text: events.append(("content", text)),
    )
    assert events[0] == ("start", "")
    assert "".join(text for kind, text in events if kind == "reasoning") == (
        "considering")
    assert "".join(text for kind, text in events if kind == "content") == "answer"
    assert result.reasoning_content == "considering"
    assert result.content == "answer"


def test_prompt_cache_evidence_surfaced(fake_server):
    # The road-test asserts engine reuse from exactly these fields.
    _, url = fake_server
    result = CompletionsClient(url).complete([{"role": "user", "content": "x"}])
    assert result.prompt_cache == {
        "event": "hit",
        "entries": 3,
        "bytes": 1024,
        "disk_checkpoints_written": 2,
    }
    assert result.cached_tokens == 512


def test_missing_cache_evidence_reads_as_none(fake_server):
    state, url = fake_server
    state.completion = {
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "t"},
                     "finish_reason": "stop"}],
    }
    result = CompletionsClient(url).complete([{"role": "user", "content": "x"}])
    assert result.prompt_cache is None
    assert result.cached_tokens is None
    assert result.usage == {}


def test_http_error_raises_client_error_with_status_and_body(fake_server):
    state, url = fake_server
    state.status = 400
    with pytest.raises(ClientError) as excinfo:
        CompletionsClient(url).complete([{"role": "user", "content": "x"}])
    assert excinfo.value.status == 400
    assert "bad request detail" in excinfo.value.message


def test_unreachable_server_raises_client_error():
    client = CompletionsClient("http://127.0.0.1:9", timeout=0.5)
    with pytest.raises(ClientError) as excinfo:
        client.complete([{"role": "user", "content": "x"}])
    assert excinfo.value.status == 0


def test_response_without_choices_raises(fake_server):
    state, url = fake_server
    state.completion = {"usage": {}}
    with pytest.raises(ClientError, match="no choices"):
        CompletionsClient(url).complete([{"role": "user", "content": "x"}])


def test_stream_requires_done_marker(fake_server):
    state, url = fake_server
    state.stream_bytes = (
        b'data: {"id":"x","created":0,"model":"m","choices":'
        b'[{"index":0,"delta":{"content":"partial"},'
        b'"finish_reason":null}]}\n\n'
    )
    with pytest.raises(ClientError, match=r"before \[DONE\]"):
        CompletionsClient(url).complete([{"role": "user", "content": "x"}])


def test_stream_error_event_is_a_client_error(fake_server):
    state, url = fake_server
    state.stream_bytes = (
        b'data: {"error":{"message":"generation failed"}}\n\n'
    )
    with pytest.raises(ClientError, match="generation failed"):
        CompletionsClient(url).complete([{"role": "user", "content": "x"}])


def test_health_returns_engine_counters(fake_server):
    state, url = fake_server
    health = CompletionsClient(url).health()
    assert health == CANNED_HEALTH
    method, path, _, _ = state.requests[-1]
    assert (method, path) == ("GET", "/health")
