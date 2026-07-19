"""Road-test driver against a fake engine (no model, no GPU).

The fake server reproduces the serve layer's cache accounting over a
whitespace tokenization: suffix-only prompt_tokens, in-memory prefix hits at
the stored full-plus-completion length, frontier checkpoint writes strictly
inside the prefilled span, a disk index that survives restarts while the
memory store does not, and per-segment health counters. The fake assistant
answers scripted turns with DSML tool calls rendered through the DS4
renderer's own encoder, and the tools really execute on the generated
fixture workspace.

This pins the driver's plumbing: phase ordering, the tool loop, restart
wiring, per-request ledger and health checks, the identity probe, and the
run record. It cannot vouch for the real engine; the GPU road-test does.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from moespresso.agentlib.profile import LoopSettings
from moespresso.agentlib.roadtest.fixture import DEFECT_FIX_LINE, DEFECT_LINE
from moespresso.agentlib.roadtest.run import (
    RoadtestRun,
    RunConfig,
    _package_run_settings,
)
from moespresso.runtime.deepseek_v4.renderer import _render_tool_calls

STRIDE = 256


def test_package_run_settings_use_default_served_context_limit(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "package_manifest.json").write_text(
        json.dumps({
            "architecture": {
                "config": {"max_position_embeddings": 262144},
            },
        }),
        encoding="utf-8",
    )

    loop, context_limit = _package_run_settings(package_dir)

    assert loop is None
    assert context_limit == 131072


def _dsml(name: str, arguments: dict) -> str:
    return _render_tool_calls([{
        "id": "call_0",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }])


def _last_user_and_steps(messages: list[dict]) -> tuple[str, int]:
    """The most recent user text and the count of tool results after it."""
    user_index = max((i for i, m in enumerate(messages)
                      if m.get("role") == "user"), default=None)
    if user_index is None:
        return "", 0
    text = messages[user_index].get("content") or ""
    steps = sum(1 for m in messages[user_index + 1:]
                if m.get("role") == "tool")
    return text, steps


def _fake_assistant(messages: list[dict]) -> str:
    text, steps_done = _last_user_and_steps(messages)
    path_match = re.search(r"(data/segment_\d+\.txt|src/metrics\.py)", text)
    if "After the search" in text:
        # The subagent's two-step brief: grep, then a full segment read.
        if steps_done == 0:
            return "Searching first." + _dsml("grep", {"pattern": "TAG",
                                                       "path": "src"})
        if steps_done == 1 and path_match:
            return ("Now reading the segment." +
                    _dsml("read_file", {"path": path_match.group(1)}))
        return "Delegated checks complete: marker located, errors counted."
    if steps_done:
        return "Noted the tool output; that answers the task."
    if "acknowledge the handoff" in text:
        return "Handoff acknowledged; standing by for the subordinate report."
    if "subordinate agent's report" in text:
        return "The subordinate located the marker and counted the errors."
    if "read_file tool" in text and path_match:
        return "Reading it now." + _dsml("read_file", {"path": path_match.group(1)})
    if "bash tool" in text:
        return "Listing the project." + _dsml("bash", {"command": "ls"})
    if "edit tool" in text:
        return "Applying the fix." + _dsml("edit", {
            "path": "src/metrics.py",
            "old_string": DEFECT_LINE,
            "new_string": DEFECT_FIX_LINE,
        })
    if "grep" in text:
        return "Searching." + _dsml("grep", {"pattern": "TAG", "path": "src"})
    return "The reference declares tag cobalt-meridian-118."


def _message_tokens(messages: list[dict]) -> list[str]:
    tokens: list[str] = []
    for message in messages:
        tokens.append(f"<{message.get('role')}>")
        tokens.extend((message.get("content") or "").split())
    return tokens


class FakeEngineState:
    """Cache accounting shaped like the serve layer's, over fake tokens.

    ``shape_content`` lets a test damage the assistant text before it enters
    both the response and the fake's own memory accounting, the same way a
    served model's malformed output becomes part of the replayed history.
    ``request_bodies`` records every completion request body so a test can
    assert the request policy fields the driver sent.
    """

    def __init__(self, stride: int = STRIDE):
        self.stride = stride
        self.disk: dict[tuple, bool] = {}
        self.memory: list[tuple] = []
        self.restores = 0
        self.writes = 0
        self.shape_content = None
        self.request_bodies: list[dict] = []

    def restart(self) -> None:
        self.memory = []
        self.restores = 0
        self.writes = 0

    def _longest_prefix(self, keys, full: tuple) -> tuple | None:
        best = None
        for key in keys:
            if len(key) < len(full) and full[: len(key)] == key:
                if best is None or len(key) > len(best):
                    best = key
        return best

    def complete(self, request: dict) -> dict:
        self.request_bodies.append(request)
        messages = request["messages"]
        full = tuple(_message_tokens(messages))
        hit = self._longest_prefix(self.memory, full)
        if hit is not None:
            event, cached = "hit", len(hit)
        else:
            disk_hit = self._longest_prefix(list(self.disk), full)
            if disk_hit is not None:
                event, cached = "disk_hit", len(disk_hit)
                self.restores += 1
            else:
                event, cached = "miss", 0

        written = 0
        frontier = (cached // self.stride + 1) * self.stride
        while frontier < len(full):
            key = full[:frontier]
            if key not in self.disk:
                self.disk[key] = True
                self.writes += 1
                written += 1
            frontier += self.stride

        content = _fake_assistant(messages)
        if self.shape_content is not None:
            content = self.shape_content(content)
        generated = ["<assistant>"] + content.split()
        self.memory.append(full + tuple(generated))

        usage = {
            "prompt_tokens": len(full) - cached,
            "completion_tokens": len(generated),
            "total_tokens": len(full) - cached + len(generated),
            "prompt_tokens_details": {"cached_tokens": cached},
            "prompt_cache": {"event": event, "entries": len(self.memory),
                             "bytes": 0},
            "moespresso": {"first_token_seconds": 0.01,
                           "generation_seconds": 0.02,
                           "generation_tps": 100.0},
        }
        if written:
            usage["prompt_cache"]["disk_checkpoints_written"] = written
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": "fake",
            "prompt_cache": {
                "entries": len(self.memory),
                "bytes": 0,
                "disk": {
                    "enabled": True,
                    "root": "fake",
                    "stride": self.stride,
                    "entries": len(self.disk),
                    "payload_bytes": sum(len(k) for k in self.disk),
                    "budget_bytes": None,
                    "restores": self.restores,
                    "writes": self.writes,
                    "evictions": 0,
                    "quarantines": 0,
                    "lock_active": True,
                    "last_event": None,
                },
            },
        }


def _make_handler(state: FakeEngineState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reply_stream(self, payload: dict):
            choice = payload["choices"][0]
            message = choice["message"]
            base = {
                "id": payload["id"],
                "object": "chat.completion.chunk",
                "created": payload["created"],
                "model": payload["model"],
                "usage": None,
            }
            events = [
                {
                    **base,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                },
                {
                    **base,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": message["content"]},
                        "finish_reason": None,
                    }],
                },
                {
                    **base,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": choice["finish_reason"],
                    }],
                },
                {**base, "choices": [], "usage": payload["usage"]},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in events:
                body = json.dumps(event, separators=(",", ":")).encode()
                self.wfile.write(b"data: " + body + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length))
            response = state.complete(request)
            if request.get("stream") is True:
                self._reply_stream(response)
            else:
                self._reply(response)

        def do_GET(self):
            self._reply(state.health())

    return Handler


class FakeController:
    """Controller stand-in: restarts clear memory, the disk index survives."""

    def __init__(self, state: FakeEngineState):
        self.state = state
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self.segment = 0

    def start(self):
        self.segment += 1
        self.state.restart()

    def wait_healthy(self, *, timeout=None):
        return self.state.health()

    def stop(self, *, timeout=None):
        pass

    def rss_bytes(self):
        return None

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture()
def fake_controller():
    controller = FakeController(FakeEngineState())
    yield controller
    controller.close()


def test_full_driver_run_against_the_fake_engine(tmp_path, fake_controller):
    config = RunConfig(
        run_root=tmp_path / "run",
        stride=STRIDE,
        budget_bytes=None,
        target_tokens=15_000,
        max_tokens=700,
        request_timeout=30.0,
        memory_budget_tokens=10**9,
        align_min_context_tokens=10**9,
    )
    run = RoadtestRun(config, fake_controller, log_fn=lambda *a, **k: None)
    code = run.run()

    assert run.findings == []
    assert code == 0
    # Three server segments: startup plus the two restart-proof restarts.
    assert fake_controller.segment == 3

    events = [
        json.loads(line)
        for line in (config.run_root / "events.jsonl").read_text().splitlines()
    ]
    requests = [e for e in events if e["kind"] == "request"]
    disk_hits = [(e["session"], e["where"]) for e in requests
                 if e["event"] == "disk_hit"]
    # Both restarted probe arms restore session a from disk. The interleaved
    # session starts after the probe restarts, so with the alignment and
    # maintenance restarts disabled it never needs a disk restore here; the
    # restart-armed tests cover the session-b restore.
    assert [s for s, _ in disk_hits] == ["a", "a"]

    probes = [e for e in events if e["kind"] == "identity_probe"]
    assert len(probes) == 1
    assert probes[0]["restore_replay_identical"] is True
    assert probes[0]["pre_matches_restored"] is True

    # The subagent scenario: the child's first request is a clean miss (no
    # contamination from the parent's prefix), every later request hits on
    # the child's own chain, and the chain grows monotonically. The ledger
    # has already asserted the exact cached_tokens values (findings == []).
    sub_requests = [e for e in requests if e["session"] == "sub"]
    assert [e["event"] for e in sub_requests] == ["miss", "hit", "hit"]
    assert sub_requests[0]["cached_tokens"] == 0
    for earlier, later in zip(sub_requests, sub_requests[1:]):
        assert later["cached_tokens"] > earlier["cached_tokens"]
    results = [e for e in events if e["kind"] == "subagent_result"]
    assert len(results) == 1
    assert results[0]["ok"] is True
    assert results[0]["session_cache_key"] == "roadtest-a/sub1"
    assert results[0]["turns"] == 3

    # The fold-back landed append-only in the parent: one tool message
    # carrying the child's final answer, between the delegation
    # acknowledgment and the follow-up user turn.
    conv_a = json.loads((config.run_root / "conversation_a.json").read_text())
    folds = [i for i, m in enumerate(conv_a)
             if m["role"] == "tool" and m["content"] ==
             "Delegated checks complete: marker located, errors counted."]
    assert len(folds) == 1
    fold_index = folds[0]
    assert conv_a[fold_index - 1]["role"] == "assistant"
    assert conv_a[fold_index + 1]["role"] == "user"
    assert "subordinate agent's report" in conv_a[fold_index + 1]["content"]
    assert (config.run_root / "conversation_sub.json").is_file()

    # Real frontier checkpoints were written for both sessions.
    assert run.ledgers["a"].frontier_count > 5
    assert run.ledgers["b"].frontier_count > 0
    assert run.ledgers["a"].last_full_tokens >= config.target_tokens

    # The fixture edit really happened on disk.
    metrics = (config.run_root / "workspace" / "src" / "metrics.py").read_text()
    assert DEFECT_FIX_LINE in metrics
    assert DEFECT_LINE not in metrics

    summary = json.loads((config.run_root / "summary.json").read_text())
    assert summary["findings"] == []
    assert summary["sessions"]["a"]["frontiers_written"]
    assert (config.run_root / "conversation_a.json").is_file()
    assert (config.run_root / "conversation_b.json").is_file()


def test_smoke_mode_attaches_and_verifies_health_deltas(tmp_path, fake_controller):
    # Pre-warm the fake so the attach-time baseline counters are nonzero;
    # the smoke must verify deltas on top of them because absolute values vary.
    state = fake_controller.state
    state.complete({"messages": [{"role": "user", "content": "warm " * 600}]})
    assert state.writes > 0
    baseline_writes = state.writes

    config = RunConfig(
        run_root=tmp_path / "run",
        stride=STRIDE,
        budget_bytes=None,
        target_tokens=0,
        request_timeout=30.0,
        memory_budget_tokens=1 << 62,
        align_min_context_tokens=1 << 62,
    )
    run = RoadtestRun(config, fake_controller, log_fn=lambda *a, **k: None)
    code = run.run_smoke()
    assert run.findings == []
    assert code == 0
    # The smoke never manages the server lifecycle.
    assert fake_controller.segment == 0

    events = [
        json.loads(line)
        for line in (config.run_root / "events.jsonl").read_text().splitlines()
    ]
    assert any(e["kind"] == "smoke_start" for e in events)
    requests = [e for e in events if e["kind"] == "request"]
    assert requests
    # Salted prefixes: every session, the spawned child included, starts as
    # a fresh chain on the live server.
    first_by_session = {}
    for event in requests:
        first_by_session.setdefault(event["session"], event)
    assert set(first_by_session) == {"a", "b", "sub"}
    assert {e["event"] for e in first_by_session.values()} == {"miss"}
    assert state.writes >= baseline_writes

    # The smoke includes the subagent scenario, salted key derivation and
    # own-chain reuse included.
    sub_requests = [e for e in requests if e["session"] == "sub"]
    assert [e["event"] for e in sub_requests] == ["miss", "hit", "hit"]
    results = [e for e in events if e["kind"] == "subagent_result"]
    assert len(results) == 1
    assert results[0]["ok"] is True
    assert results[0]["session_cache_key"].startswith("smoke-a-")
    assert results[0]["session_cache_key"].endswith("/sub1")


def test_driver_records_engine_deviations_as_findings(tmp_path, fake_controller):
    # Sabotage the fake: drop every checkpoint write so the ledger's
    # disk-write expectation fails on the first long request, and the restart
    # restore then has nothing to restore from.
    state = fake_controller.state
    original_complete = state.complete

    def complete_without_writes(request):
        response = original_complete(request)
        response["usage"]["prompt_cache"].pop("disk_checkpoints_written", None)
        return response

    state.complete = complete_without_writes

    config = RunConfig(
        run_root=tmp_path / "run",
        stride=STRIDE,
        budget_bytes=None,
        target_tokens=1_000,
        request_timeout=30.0,
        memory_budget_tokens=10**9,
        align_min_context_tokens=10**9,
    )
    run = RoadtestRun(config, fake_controller, log_fn=lambda *a, **k: None)
    code = run.run()
    assert code == 1
    codes = {finding.code for finding in run.findings}
    assert "disk.checkpoints_written" in codes


def test_maintenance_restart_bounds_live_cache_memory(tmp_path, fake_controller):
    # A small memory budget forces the driver to restart the server between
    # turns; the sessions must resume from disk checkpoints with no findings.
    config = RunConfig(
        run_root=tmp_path / "run",
        stride=STRIDE,
        budget_bytes=None,
        target_tokens=1_000,
        request_timeout=30.0,
        memory_budget_tokens=12_000,
        align_min_context_tokens=10**9,
    )
    run = RoadtestRun(config, fake_controller, log_fn=lambda *a, **k: None)
    code = run.run()
    assert run.findings == []
    assert code == 0
    assert run._maintenance_restarts > 0
    events = [
        json.loads(line)
        for line in (config.run_root / "events.jsonl").read_text().splitlines()
    ]
    maintenance = [e for e in events if e["kind"] == "maintenance_restart"]
    assert maintenance
    assert all(e["projected_tokens"] >= config.memory_budget_tokens
               for e in maintenance)
    # Restarts empty the fake's memory store, so the sessions restored from
    # disk more times than the two probe restarts alone.
    requests = [e for e in events if e["kind"] == "request"]
    disk_hits = [e for e in requests if e["event"] == "disk_hit"]
    assert len(disk_hits) > 3


def test_align_restart_realigns_large_extensions(tmp_path, fake_controller):
    # With alignment restarts armed, a large tool-result extension of an
    # unaligned prefix restarts the server first, so the request restores at
    # a frontier (disk_hit with stride-aligned cached_tokens).
    config = RunConfig(
        run_root=tmp_path / "run",
        stride=STRIDE,
        budget_bytes=None,
        target_tokens=1_000,
        request_timeout=30.0,
        memory_budget_tokens=10**9,
        align_min_context_tokens=1_000,
        align_min_pending_chars=4_000,
    )
    run = RoadtestRun(config, fake_controller, log_fn=lambda *a, **k: None)
    code = run.run()
    assert run.findings == []
    assert code == 0
    assert run._align_restarts > 0
    events = [
        json.loads(line)
        for line in (config.run_root / "events.jsonl").read_text().splitlines()
    ]
    aligns = [e for e in events if e["kind"] == "align_restart"]
    assert aligns
    assert all(e["unaligned_prefix"] % 2048 != 0 for e in aligns)
    # Every restore lands exactly on a stride multiple.
    restored = [e for e in events
                if e["kind"] == "request" and e["event"] == "disk_hit"]
    assert restored
    assert all(e["cached_tokens"] % STRIDE == 0 for e in restored)


# --- profile mode -----------------------------------------------------------

PROFILE_LOOP = LoopSettings(
    dialect="dsml",
    repair=True,
    thinking_for_tools=False,
    reprompt_enabled=True,
    reprompt_limit=1,
    sampling={"temperature": 0.6, "top_p": 0.95},
)

# The recorded served malformation class: the closing quote of the parameter
# name attribute is dropped and fuses into the string attribute.
_QUOTE_DAMAGE_RE = re.compile(r'(name="[^"]*)" (string=")')


def _damage_dsml_quoting(content: str) -> str:
    return _QUOTE_DAMAGE_RE.sub(r"\1 \2", content)


def _profile_config(tmp_path, **overrides) -> RunConfig:
    fields = dict(
        run_root=tmp_path / "run",
        stride=STRIDE,
        budget_bytes=None,
        target_tokens=15_000,
        max_tokens=700,
        request_timeout=30.0,
        memory_budget_tokens=10**9,
        align_min_context_tokens=10**9,
        loop=PROFILE_LOOP,
    )
    fields.update(overrides)
    return RunConfig(**fields)


def test_profile_mode_repairs_every_call_and_enforces_request_policy(
        tmp_path, fake_controller):
    # Every tool call the fake emits carries the recorded quoting damage, so
    # strict parsing fails on every attempt and the repair layer must carry
    # the whole run, exactly the served dialect's operating point.
    state = fake_controller.state
    state.shape_content = _damage_dsml_quoting

    config = _profile_config(tmp_path)
    run = RoadtestRun(config, fake_controller, log_fn=lambda *a, **k: None)
    code = run.run()

    assert run.findings == []
    assert code == 0
    telemetry = run.repair_telemetry
    assert telemetry.fires > 0
    assert telemetry.salvaged == telemetry.fires
    assert telemetry.failed == 0

    # The run-wide request policy: thinking off on every request, and no
    # request-level tools (the dialect is taught through the system prompt).
    assert state.request_bodies
    for body in state.request_bodies:
        assert body.get("chat_template_kwargs") == {"enable_thinking": False}
        assert "tools" not in body
        system = body["messages"][0]
        assert system["role"] == "system"
        assert "tool_calls>" in system["content"]

    # The repair record: events plus the corpus file with the parsed calls.
    events = [
        json.loads(line)
        for line in (config.run_root / "events.jsonl").read_text().splitlines()
    ]
    repairs = [e for e in events if e["kind"] == "repair"]
    assert repairs
    assert all(e["salvaged"] for e in repairs)
    corpus = [
        json.loads(line)
        for line in (config.run_root / "repairs.jsonl").read_text().splitlines()
    ]
    assert len(corpus) == telemetry.fires
    assert all(entry["salvaged"] and entry["calls"] for entry in corpus)

    # The repaired calls really executed: the fixture edit landed.
    metrics = (config.run_root / "workspace" / "src" / "metrics.py").read_text()
    assert DEFECT_FIX_LINE in metrics

    summary = json.loads((config.run_root / "summary.json").read_text())
    assert summary["repair_telemetry"] == telemetry.as_dict()
    assert summary["loop"]["dialect"] == "dsml"
    assert summary["loop"]["profile_sampling_not_applied"] == {
        "temperature": 0.6, "top_p": 0.95}


def test_profile_mode_unrecoverable_repair_fails_the_run(
        tmp_path, fake_controller):
    # Mangle the invoke elements so the block parses to no call and no
    # repair transformation can rebuild one; the telemetry alarm condition
    # (failed above zero) must fail the run.
    state = fake_controller.state
    state.shape_content = lambda content: content.replace(
        "invoke", "invk")

    config = _profile_config(tmp_path, target_tokens=0)
    run = RoadtestRun(config, fake_controller, log_fn=lambda *a, **k: None)
    code = run.run()

    assert code == 1
    assert run.repair_telemetry.failed > 0
    assert run.repair_telemetry.salvaged == 0
    codes = {finding.code for finding in run.findings}
    assert "repair.failed" in codes
    # Engine-counter assertions all held; the failures are the dialect
    # contract and the scenario checks that depend on executed calls.
    assert not any(code.startswith(("cache.", "disk.", "health."))
                   for code in codes)


def test_profile_mode_nudges_a_prose_first_turn_once(tmp_path, fake_controller):
    # The first tool-requiring turn arrives as plain prose; the nudge policy
    # must re-prompt exactly once and the turn must then complete with a
    # real call, with every counter assertion still holding.
    state = fake_controller.state
    dropped = {"count": 0}

    def drop_first_block(content: str) -> str:
        if "tool_calls>" in content and dropped["count"] == 0:
            dropped["count"] += 1
            return "I will inspect the project layout first."
        return content

    state.shape_content = drop_first_block

    config = _profile_config(tmp_path, target_tokens=0)
    run = RoadtestRun(config, fake_controller, log_fn=lambda *a, **k: None)
    code = run.run()

    assert run.findings == []
    assert code == 0
    assert dropped["count"] == 1
    events = [
        json.loads(line)
        for line in (config.run_root / "events.jsonl").read_text().splitlines()
    ]
    nudges = [e for e in events if e["kind"] == "nudge"]
    assert len(nudges) == 1
    assert nudges[0]["session"] == "a"
    summary = json.loads((config.run_root / "summary.json").read_text())
    assert summary["nudges_fired"] == {"a": 1, "b": 0}


def test_subagent_failure_surfaces_as_findings_not_exceptions(
        tmp_path, fake_controller):
    # A one-step turn cap makes the child hit its turn limit mid-task. The
    # driver must record the failure as subagent findings, fold the error
    # payload into the parent as a tool result, and keep the parent session
    # serving append-only through the follow-up turn.
    config = RunConfig(
        run_root=tmp_path / "run",
        stride=STRIDE,
        budget_bytes=None,
        target_tokens=0,
        max_turn_steps=1,
        request_timeout=30.0,
        memory_budget_tokens=1 << 62,
        align_min_context_tokens=1 << 62,
    )
    run = RoadtestRun(config, fake_controller, log_fn=lambda *a, **k: None)
    code = run.run_smoke()
    assert code == 1
    codes = {finding.code for finding in run.findings}
    assert "subagent.failed" in codes
    assert "subagent.no_reuse" in codes
    # Every engine-counter assertion still held; only the scenario-level
    # subagent checks failed.
    assert all(finding.code.startswith("subagent.") for finding in run.findings)

    conv_a = run.conversations["a"].request_messages()
    folds = [m for m in conv_a if m["role"] == "tool"
             and m["content"].startswith("error: subagent (max_turns)")]
    assert len(folds) == 1
    # The follow-up turn ran after the failed fold-back.
    assert conv_a[-1]["role"] == "assistant"
