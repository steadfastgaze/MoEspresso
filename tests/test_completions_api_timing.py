from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from moespresso.agentlib.client import ChatCompletion, ClientError, CompletionsClient
from moespresso.runtime import completions_api_timing as timing
from moespresso.runtime.diagnostics import _digest
from moespresso.runtime.timing_tokens import LocalTokenCounter


@pytest.fixture
def counter():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({
        "[UNK]": 0, "one": 1, "two": 2, "three": 3, "four": 4, "hello": 5,
        "user": 6, "assistant": 7,
    }, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{message['role']}} {{message['content']}} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    return LocalTokenCounter(tokenizer, source="synthetic")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Client:
    base_url = "http://timing.invalid"
    model = "model"
    stream = True
    include_stream_usage = True

    def __init__(self, clock, usage=None):
        self.clock = clock
        self.usage = {} if usage is None else usage
        self.calls = []
        self.active = False

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        self.active = True
        try:
            self.clock.advance(0.25)
            kwargs["on_start"]()
            self.clock.advance(0.75)
            kwargs["on_chunk"]({"choices": [{"index": 0, "delta": {"content": "one two "}}]})
            self.clock.advance(2)
            kwargs["on_chunk"]({"choices": [{"index": 0, "delta": {"content": "three four"}}]})
            self.clock.advance(0.25)
            kwargs["on_chunk"]({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                "timings": {"predicted_per_second": 777}})
            self.clock.advance(5)
            kwargs["on_chunk"]({"choices": [], "usage": self.usage})
            return ChatCompletion(message={"role": "assistant", "content": "one two three four"},
                                  finish_reason="stop", usage=self.usage)
        finally:
            self.active = False


def run(tmp_path, counter, *, usage=None, client=None, clock=None, **kwargs):
    clock = clock or Clock()
    client = client or Client(clock, usage)
    path = tmp_path / "timing.json"
    code = timing.run_timing(client, counter, repeats=1, max_tokens=16, output=path,
                             cases=(("test", "hello"),), clock=clock, **kwargs)
    report = json.loads(path.read_text())
    digest = report.pop("content_sha256")
    assert _digest(report) == digest
    return code, report, client


@pytest.mark.parametrize("reported", [None, 4, 999, -1, "4", True, float("nan")])
def test_counts_and_clocks_do_not_depend_on_server_claims(tmp_path, counter, reported):
    code, report, client = run(tmp_path, counter, usage={"prompt_tokens": 99, "completion_tokens": reported})
    assert code == 0 and report["status"] == "completed"
    assert len(client.calls) == 2
    row = report["requests"][1]
    metrics = row["metrics"]
    assert row["local_output"]["tokens"] == 4
    assert row["local_prompt"]["tokens"] == 3
    assert metrics["first_output_seconds"] == 1
    assert metrics["response_headers_seconds"] == 0.25
    assert metrics["last_output_seconds"] == 3
    assert metrics["completion_tail_seconds"] == 5.25
    assert metrics["request_seconds"] == 8.25
    assert metrics["first_delivery_tokens"] == 2
    assert metrics["local_output_tps"] == pytest.approx(4 / 8.25)
    assert metrics["local_after_first_delivery_tps_estimate"] == 1
    assert row["server_timings"] == [{"predicted_per_second": 777}]
    assert report["measurement_contract"]["health_requests"] == 0


def test_tokenization_stays_outside_generation(tmp_path, counter, monkeypatch):
    clock = Clock()
    client = Client(clock)
    original_prompt, original_output = counter.prompt, counter.output

    def prompt(*args):
        assert not client.active and not client.calls
        return original_prompt(*args)

    def output(*args):
        assert not client.active
        return original_output(*args)

    monkeypatch.setattr(counter, "prompt", prompt)
    monkeypatch.setattr(counter, "output", output)
    assert run(tmp_path, counter, client=client, clock=clock)[0] == 0


def test_whole_text_offsets_prevent_fragment_token_overcount(counter):
    result = counter.output([
        {"seconds": 1, "content": "he"}, {"seconds": 2, "content": "llo"},
    ])
    assert result["tokens"] == 1
    assert [row["tokens"] for row in result["delivery_boundaries"]] == [0, 1]


def test_reasoning_and_content_in_same_chunk_share_one_timing_boundary(counter):
    clock = Clock()
    observer = timing.StreamClock(0, clock)
    clock.advance(1)
    observer.chunk({"choices": [{"delta": {"reasoning_content": "one two", "content": "three"}}]})
    result = counter.output(observer.events)
    assert result["tokens"] == 3 and result["reasoning_tokens"] == 2
    assert len(result["delivery_boundaries"]) == 1
    metrics = timing.calculate_metrics({"tokens": 10}, result, observer, 2, {})
    assert metrics["local_after_first_delivery_tps_estimate"] is None
    assert metrics["local_output_tps"] == 1.5


def test_slow_tokenizer_uses_post_response_prefix_retokenization(counter):
    counter.identity["is_fast"] = False
    result = counter.output([
        {"seconds": 1, "reasoning": "one "},
        {"seconds": 2, "reasoning": "two", "content": "three four"},
    ])
    assert result["method"] == "prefix_retokenization"
    assert result["tokens"] == 4
    assert [row["tokens"] for row in result["delivery_boundaries"]] == [1, 4]


def test_empty_output_does_not_invent_tokens_or_decode_rate(counter):
    result = counter.output([])
    metrics = timing.calculate_metrics({"tokens": 4}, result, timing.StreamClock(0), 1, {"completion_tokens": 9})
    assert result["tokens"] == 0
    assert metrics["first_output_seconds"] is None
    assert metrics["local_after_first_delivery_tps_estimate"] is None
    assert metrics["api_minus_local_output_tokens"] == 9


def test_rejected_usage_option_is_omitted_once_without_trusting_api_counts(tmp_path, counter):
    clock = Clock()
    client = Client(clock)
    original = client.complete
    rejected = []

    def complete(*args, **kwargs):
        if client.include_stream_usage:
            rejected.append(1)
            clock.advance(0.5)
            raise ClientError(422, "stream_options is unsupported")
        return original(*args, **kwargs)

    client.complete = complete
    code, report, _ = run(tmp_path, counter, client=client, clock=clock)
    assert code == 0 and rejected == [1]
    assert report["compatibility"][0]["rejected_warmup_seconds"] == 0.5
    assert report["requests"][1]["local_output"]["tokens"] == 4


@pytest.mark.parametrize("failure,expected", [(RuntimeError("broken"), 1), (KeyboardInterrupt(), 130)])
def test_partial_stream_survives_failure_or_interrupt(tmp_path, counter, failure, expected):
    clock = Clock()
    client = Client(clock)

    def fail(*_args, **kwargs):
        clock.advance(1)
        kwargs["on_chunk"]({"choices": [{"delta": {"content": "hello"}}]})
        raise failure

    client.complete = fail
    code, report, _ = run(tmp_path, counter, client=client, clock=clock)
    assert code == expected
    assert report["requests"][0]["partial_local_output"]["tokens"] == 1
    assert report["requests"][0]["partial_text_events"][0]["content"] == "hello"


def test_existing_output_is_not_overwritten_or_queried(tmp_path, counter):
    path = tmp_path / "keep.json"
    path.write_text("keep")
    client = Client(Clock())
    with pytest.raises(FileExistsError):
        timing.run_timing(client, counter, repeats=1, max_tokens=16, output=path)
    assert not client.calls and path.read_text() == "keep"


def test_local_tokenizer_load_requires_no_weights_or_download(tmp_path, counter):
    counter.tokenizer.save_pretrained(tmp_path)
    restored = LocalTokenCounter.load(str(tmp_path))
    assert restored.identity["vocabulary_sha256"] == counter.identity["vocabulary_sha256"]
    assert restored.prompt([{"role": "user", "content": "hello"}], {})["tokens"] == 3


def test_tokenizer_requires_chat_template(counter):
    counter.tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat_template"):
        LocalTokenCounter(counter.tokenizer, source="synthetic")


@pytest.mark.parametrize("extra", [
    ["--url", "ftp://example.org"], ["--url", "http://user:secret@example.org"],
    ["--timeout", "nan"], ["--max-tokens", "1"], ["--repeats", "0"],
    ["--chat-template-kwargs", "[]"], ["--chat-template-kwargs", '{"truncation":true}'],
])
def test_invalid_cli_fails_before_tokenizer_or_network(extra, monkeypatch):
    monkeypatch.setattr(LocalTokenCounter, "load", lambda *_a, **_k: pytest.fail("must validate first"))
    with pytest.raises(SystemExit) as error:
        timing.main(["--tokenizer", "unused", *extra])
    assert error.value.code == 2


def test_cli_accepts_v1_url_and_dispatches_with_independent_counter(tmp_path, counter, monkeypatch):
    monkeypatch.setattr(LocalTokenCounter, "load", lambda *_a, **_k: counter)
    seen = {}

    def capture(client, local, **kwargs):
        seen.update(url=client.base_url, model=client.model, counter=local, **kwargs)
        return 0

    monkeypatch.setattr(timing, "run_timing", capture)
    from moespresso.cli import main

    assert main(["completions-api-timing", "--tokenizer", "local", "--url", "http://example.org/v1/",
                 "--model", "served-model", "--output", str(tmp_path / "result.json")]) == 0
    assert seen["url"] == "http://example.org" and seen["model"] == "served-model"
    assert seen["counter"] is counter


def test_http_server_without_health_or_usage_is_measured(tmp_path, counter):
    requests = []
    gets = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            gets.append(self.path)
            self.send_error(404)

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            assert "top_k" not in body and "min_p" not in body
            assert body["model"] == "foreign-model"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for text in ("one two ", "three four"):
                event = {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = CompletionsClient(f"http://127.0.0.1:{server.server_port}", model="foreign-model")
        assert timing.run_timing(client, counter, repeats=1, max_tokens=16,
                                 output=tmp_path / "http.json", cases=(("test", "hello"),)) == 0
        report = json.loads((tmp_path / "http.json").read_text())
        assert len(requests) == 2 and gets == []
        assert all(row["local_output"]["tokens"] == 4 for row in report["requests"])
        assert report["requests"][1]["usage"] == {}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
