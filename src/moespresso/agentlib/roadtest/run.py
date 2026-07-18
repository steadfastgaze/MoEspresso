"""The road-test driver: phases, per-request assertions, and the run record.

The driver owns ordering. It generates the fixture workspace, starts the
served model with the disk KV store enabled, then drives the scripted
session through agentlib: every request's usage block is checked against the
session ledger, every response is followed by a ``/health`` cross-check, and
every event lands in ``events.jsonl`` under the run directory.

The two proofs are driver phases. The restart proof issues one probe turn
three times around two full server restarts: the first restart compares the
restored completion against the live pre-restart completion (suffix prefill
geometry differs between those arms, so a token difference there is recorded
and classified rather than assumed to be a restore defect), and the second
restart repeats the restore at identical geometry, where the recorded disk
KV evidence requires token-identical output. The interleaving proof runs a
second session under its own cache key between main-session turns and
resumes it from disk after the restarts. The subagent scenario spawns a
sequential child through ``SubagentRunner`` mid-session: the child runs its
brief to completion under a session key derived from the parent's, every
child request is checked against a fresh ledger (clean first miss, then
hits on the child's own chain exactly), and the folded-back tool result
must leave the parent session append-only.

A package that carries an agentic profile drives the run in profile mode:
the resolved loop settings pick the dialect adapter (system-prompt tool
teaching versus request-level tools), the thinking template kwargs ride on
every request, strict parse failures route through the repair layer with
telemetry, and the tool-nudge policy applies per session. The run fails
when any repair fire stays unsalvaged, the alarm condition the
repair-dependent dialect adoption is conditioned on. Sampling stays pinned
at temperature 0 and top_p 1 in both modes because the per-request
assertions require deterministic completions; the profile's product
sampling table is recorded in the run record as not applied.

Assertion failures are collected as findings and the run keeps going; the
exit code is nonzero when any finding was recorded. Performance numbers
(first-token latency, decode rate) are logged per request and never
asserted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from moespresso.agentlib.client import ChatCompletion, ClientError, CompletionsClient
from moespresso.agentlib.conversation import Conversation
from moespresso.agentlib.dsml import parse_dsml_tool_calls
from moespresso.agentlib.execution import execute_tool_call
from moespresso.agentlib.loop_policy import NUDGE_MESSAGE
from moespresso.agentlib.profile import (
    LoopSettings,
    load_agentic_profile,
    resolve_loop_settings,
)
from moespresso.agentlib.repair import RepairTelemetry
from moespresso.agentlib.roadtest.fixture import (
    DATA_FILE_COUNT,
    Fixture,
    generate_fixture,
)
from moespresso.agentlib.roadtest.ledger import (
    Finding,
    HealthExpectations,
    RequestCheck,
    SessionLedger,
    is_list_prefix,
)
from moespresso.agentlib.roadtest.script import (
    SYSTEM_PROMPT_A,
    SYSTEM_PROMPT_B,
    SYSTEM_PROMPT_SUB,
    RoadtestScript,
    ScriptTurn,
    build_script,
)
from moespresso.agentlib.roadtest.server import (
    DEFAULT_GPU_LOCKDIR,
    GpuLock,
    ServerController,
)
from moespresso.agentlib.subagent import (
    SubagentBrief,
    SubagentRunner,
    child_session_key,
)
from moespresso.agentlib.toolcalls import ToolCallParseError
from moespresso.agentlib.tools import build_core_registry

SESSION_CACHE_KEYS = {"a": "roadtest-a", "b": "roadtest-b"}


class AttachedController:
    """Controller stand-in for smoke runs against an already-running server.

    No spawn, no model load, no lifecycle: start and stop are no-ops and the
    health check queries the live server. Restart-dependent phases must stay
    disabled when this controller is in use.
    """

    def __init__(self, base_url: str, *, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.segment = 1
        self._client = CompletionsClient(self.base_url, timeout=timeout)

    def start(self) -> None:
        pass

    def wait_healthy(self, *, timeout: float | None = None) -> dict:
        return self._client.health()

    def stop(self, *, timeout: float | None = None) -> None:
        pass

    def rss_bytes(self) -> int | None:
        # The attached server is not this process's child; its memory is
        # not observable from here.
        return None


@dataclass(frozen=True)
class RunConfig:
    """Everything the driver needs besides the server controller."""

    run_root: Path
    stride: int
    budget_bytes: int | None
    target_tokens: int = 110_000
    max_tokens: int = 700
    max_turn_steps: int = 6
    request_timeout: float = 3600.0
    keep_disk: bool = False
    # Live-cache memory budget, in token-equivalents. A mitigation built
    # against a serve-side prompt cache that retained one full snapshot per
    # request once the rotating window made the cache untrimmable; the store
    # now pops superseded prefix entries on strict-extension inserts, so a
    # cumulative session holds one entry and this projection (which assumes
    # full retention) overestimates. The driver restarts the server before a
    # turn whenever the projected retained tokens plus twice the live
    # context would pass this budget; a restart empties the store and
    # resumes from the disk checkpoint. ``--no-mitigation-restarts``
    # disables it; the certification run of the fixed engine runs without
    # it.
    memory_budget_tokens: int = 120_000
    # Prefill-alignment restarts. A mitigation built against the uniform
    # divisor prefill step (gcd(next_frontier - restored_prefix, stride)),
    # which collapsed to a few tokens on an unaligned in-memory prefix and
    # aborted the serve process past roughly 20k context. The frontier
    # writer now plans variable-step chunks that land on each frontier
    # exactly, so the collapse no longer exists; the restart lever remains
    # for driving older engines. The driver restarts before any request
    # that follows at least ``align_min_pending_chars`` of appended text
    # when the session sits past ``align_min_context_tokens`` on an
    # unaligned in-memory prefix with a disk frontier to restore from.
    align_min_context_tokens: int = 5000
    align_min_pending_chars: int = 4000
    # Profile-driven mode. When set, the run derives its dialect behavior
    # from the resolved loop settings the served package's agentic profile
    # produced: the tool-teaching block lands in the system prompts, the
    # request tool surface comes from the dialect adapter, every request
    # carries the profile's thinking template kwargs, strict parse failures
    # route through the repair layer with telemetry, and the tool-nudge
    # policy applies per session. The run fails when any repair fire stays
    # unsalvaged. ``None`` keeps the template-native request shape: request
    # tools sent, strict DSML parsing, no repair, no template kwargs.
    # Sampling stays pinned at temperature 0 and top_p 1 either way; the
    # per-request assertions require deterministic completions, so the
    # profile's product sampling table is deliberately not applied.
    loop: LoopSettings | None = None
    # Data segments generated into the fixture workspace; the extension
    # reserve grows with it, so a deeper context target needs a larger
    # count.
    data_file_count: int = DATA_FILE_COUNT
    # The served model's declared context ceiling in tokens. The growth
    # phase stops early rather than submit a request the server would
    # refuse for exceeding the limit.
    context_limit: int | None = None


def _print_flush(*args, **kwargs) -> None:
    """Progress lines must land immediately when stdout is a file."""
    print(*args, flush=True, **kwargs)


class _LedgeredChildClient:
    """Client facade handed to the subagent runner.

    Every child request flows through the run's bookkeeping (append-only
    check, session ledger, health cross-check, event record) and the
    completion returns to the runner unchanged, so the child loop is checked
    with exactly the same rigor as the scripted sessions.
    """

    def __init__(self, run: RoadtestRun, session: str):
        self.run = run
        self.session = session
        self.steps = 0

    def complete(self, conversation: Conversation, **kwargs) -> ChatCompletion:
        self.steps += 1
        completion, _ = self.run._observed_request(
            self.session, f"sub-01/step{self.steps}", conversation, **kwargs)
        return completion


class RoadtestRun:
    """One full road-test run against a controller-managed server."""

    def __init__(self, config: RunConfig, controller, *, log_fn=_print_flush):
        self.config = config
        self.controller = controller
        self.log = log_fn
        self.client = CompletionsClient(
            controller.base_url, timeout=config.request_timeout)
        self.registry = build_core_registry()
        self.dialect = (config.loop.dialect_adapter()
                        if config.loop is not None else None)
        if self.dialect is not None:
            self.tools = self.dialect.request_tools(self.registry)
        else:
            self.tools = self.registry.openai_tools()
        self.repair_telemetry = RepairTelemetry()
        self.nudges: dict = {}
        self._calls_executed: dict[str, int] = {}
        self._child_parses = 0
        self.fixture: Fixture | None = None
        self.script: RoadtestScript | None = None
        self.conversations: dict[str, Conversation] = {}
        self.ledgers: dict[str, SessionLedger] = {}
        self.health = HealthExpectations(
            stride=config.stride, budget_bytes=config.budget_bytes)
        self.findings: list[Finding] = []
        self.completed = False
        self.last_health: dict = {}
        self._last_sent: dict[str, list | None] = {"a": None, "b": None}
        self._events_path = config.run_root / "events.jsonl"
        self._started = time.monotonic()
        # Store-key lengths inserted since the last restart; the worst-case
        # retention model keeps at most the ten most recent.
        self._keys_since_restart: list[int] = []
        self._maintenance_restarts = 0
        self._align_restarts = 0

    # --- run record ---

    def _event(self, kind: str, **payload) -> None:
        record = {"t": round(time.monotonic() - self._started, 3), "kind": kind}
        record.update(payload)
        with open(self._events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _finding(self, finding: Finding) -> None:
        self.findings.append(finding)
        self.log(f"[roadtest] FINDING {finding.code} at {finding.where}: "
                 f"{finding.message}")
        self._event("finding", **asdict(finding))

    @property
    def total_frontiers(self) -> int:
        return sum(ledger.frontier_count for ledger in self.ledgers.values())

    # --- dialect plumbing ---

    def _system_prompt(self, base: str) -> str:
        """Compose a session system prompt for the run's dialect mode.

        In profile mode a dialect taught through the system prompt (no
        request-level tools) gets its tool-teaching block appended, the
        same composition the dialect study validated.
        """
        if self.dialect is None:
            return base
        block = self.dialect.tools_block(self.registry)
        if block is None:
            return base
        return base + "\n\n" + block

    def _parse_calls(self, where: str, content: str | None):
        """Parse one assistant turn's tool calls through the run's seam.

        Legacy mode is the strict DSML parser. Profile mode uses the
        dialect adapter: strict parse first, then, when the profile
        requires repair, the repair layer with telemetry. Every repair
        fire lands in the run record (event plus ``repairs.jsonl`` with
        the raw content), salvaged or not. Raises ``ToolCallParseError``
        only when the turn stays unparseable.
        """
        if self.dialect is None:
            return parse_dsml_tool_calls(content)
        try:
            return list(self.dialect.parse_turn(content or "", self.registry).calls)
        except ToolCallParseError as error:
            if self.config.loop is None or not self.config.loop.repair:
                raise
            entry = {"where": where, "error": str(error),
                     "content": content or ""}
            try:
                calls = list(self.dialect.repair_turn(
                    content or "", self.registry).calls)
            except ToolCallParseError as repair_error:
                self.repair_telemetry.record(salvaged=False)
                entry["salvaged"] = False
                entry["repair_error"] = str(repair_error)
                self._record_repair(entry, calls=None)
                raise
            self.repair_telemetry.record(salvaged=True)
            entry["salvaged"] = True
            self._record_repair(entry, calls=calls)
            return calls

    def _record_repair(self, entry: dict, *, calls) -> None:
        record = dict(entry)
        if calls is not None:
            record["calls"] = [
                {"name": call.name, "arguments": call.arguments}
                for call in calls
            ]
        with open(self.config.run_root / "repairs.jsonl", "a",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._event("repair", where=entry["where"],
                    salvaged=entry["salvaged"], error=entry["error"])

    def _init_nudges(self) -> None:
        """One tool-nudge policy per scripted session, when the profile asks."""
        if self.config.loop is None:
            return
        for name in self.conversations:
            policy = self.config.loop.nudge_policy()
            if policy is not None:
                self.nudges[name] = policy

    def _loop_record(self) -> dict:
        """The resolved loop settings as run-record fields."""
        loop = self.config.loop
        if loop is None:
            return {"loop": None}
        return {"loop": {
            "dialect": loop.dialect,
            "repair": loop.repair,
            "thinking_for_tools": loop.thinking_for_tools,
            "reprompt_enabled": loop.reprompt_enabled,
            "reprompt_limit": loop.reprompt_limit,
            # The profile's product sampling, recorded but deliberately not
            # applied: the run pins temperature 0 and top_p 1 because the
            # per-request assertions require deterministic completions.
            "profile_sampling_not_applied": loop.request_sampling(),
            "run_sampling": {"temperature": 0.0, "top_p": 1.0},
        }}

    def _parse_child_message(self, message: dict):
        """The subagent runner's parse seam, routed through the run's own."""
        self._child_parses += 1
        return self._parse_calls(
            f"sub-01/parse{self._child_parses}", message.get("content"))

    def _maybe_nudge(self, session: str, where: str) -> bool:
        """Apply the tool-nudge policy to a zero-call final turn.

        Returns True when the nudge fired and the caller should request
        again instead of accepting the turn as final.
        """
        nudge = self.nudges.get(session)
        if nudge is None:
            return False
        if not nudge.wants_reprompt(
                final=True, calls_in_turn=0,
                calls_before_turn=self._calls_executed.get(session, 0)):
            return False
        nudge.note_fired()
        self._event("nudge", where=where, session=session,
                    fired=nudge.fired)
        self.conversations[session].add_user(NUDGE_MESSAGE)
        return True

    # --- request plumbing ---

    def _request(self, session: str, where: str) -> tuple[ChatCompletion, RequestCheck]:
        return self._observed_request(
            session, where, self.conversations[session], tools=self.tools,
            temperature=0.0, top_p=1.0, max_tokens=self.config.max_tokens)

    def _observed_request(self, session: str, where: str,
                          conversation: Conversation,
                          **request_kwargs) -> tuple[ChatCompletion, RequestCheck]:
        """Send one request and route it through the run's full bookkeeping.

        Every request the run makes flows through here, whichever
        conversation drives it: the append-only check against the session's
        previously sent messages, the ledger check of the usage block, the
        ``/health`` cross-check, and the event record. Profile mode also
        enforces the run-wide request policy at this single site: the
        dialect's request tool surface and the profile's thinking template
        kwargs apply to every request, the subagent runner's included.
        """
        if self.config.loop is not None:
            request_kwargs["tools"] = self.tools
            request_kwargs["chat_template_kwargs"] = (
                self.config.loop.chat_template_kwargs())
        messages = conversation.request_messages()
        previous = self._last_sent[session]
        if previous is not None and not is_list_prefix(previous, messages):
            self._finding(Finding(
                "structure.append_only", where,
                f"session {session}: request messages are not a list-prefix "
                f"extension of the previous request "
                f"({len(previous)} -> {len(messages)} messages)"))
        self._last_sent[session] = messages

        started = time.monotonic()
        completion = self.client.complete(conversation, **request_kwargs)
        wall = time.monotonic() - started

        check = self.ledgers[session].observe(where, completion.usage)
        for finding in check.findings:
            self._finding(finding)
        self._keys_since_restart.append(
            check.full_tokens + check.completion_tokens)
        self.health.on_request(check)
        try:
            self.last_health = self.client.health()
            for finding in self.health.verify(
                    where, self.last_health, expected_entries=self.total_frontiers):
                self._finding(finding)
        except ClientError as e:
            self._finding(Finding(
                "health.unreachable", where, f"/health failed after request: {e}"))

        perf = completion.usage.get("moespresso") or {}
        memory_cache = completion.usage.get("prompt_cache") or {}
        self._event(
            "request",
            where=where,
            session=session,
            messages=len(messages),
            event=check.event,
            cached_tokens=check.cached_tokens,
            prompt_tokens=check.prompt_tokens,
            completion_tokens=check.completion_tokens,
            full_tokens=check.full_tokens,
            new_frontiers=list(check.new_frontiers),
            finish_reason=completion.finish_reason,
            wall_seconds=round(wall, 3),
            first_token_seconds=perf.get("first_token_seconds"),
            generation_tps=perf.get("generation_tps"),
            mem_entries=memory_cache.get("entries"),
            mem_bytes=memory_cache.get("bytes"),
            retained_tokens=self.retained_store_tokens,
            rss_bytes=self.controller.rss_bytes(),
            disk=(self.last_health.get("prompt_cache") or {}).get("disk"),
        )
        self.log(f"[roadtest] {where}: event={check.event} "
                 f"cached={check.cached_tokens} suffix={check.prompt_tokens} "
                 f"full={check.full_tokens} completion={check.completion_tokens} "
                 f"writes={len(check.new_frontiers)} wall={wall:.1f}s")
        return completion, check

    def _execute_calls(self, session: str, where: str, calls) -> int:
        appended_chars = 0
        for call in calls:
            result = execute_tool_call(
                self.registry, call, workdir=self.fixture.root)
            self._event("tool", where=where, session=session, name=call.name,
                        ok=result.ok, output_chars=len(result.output))
            self.conversations[session].add_tool_result(result.output)
            appended_chars += len(result.output)
        self._calls_executed[session] = (
            self._calls_executed.get(session, 0) + len(calls))
        return appended_chars

    @property
    def retained_store_tokens(self) -> int:
        """Worst-case token-equivalents resident in the serve-side store."""
        return sum(self._keys_since_restart[-10:])

    def _maybe_maintenance_restart(self, session: str) -> None:
        """Restart before a turn that would pass the live-cache memory budget.

        The projection assumes worst-case retention: every recent request's
        snapshot still resident plus twice the session context. An engine
        that pops superseded prefix entries holds one entry per session, so
        the projection overestimates there and the restart fires early; it
        is a conservative operator lever. It does not account for the fixed
        engine. A restart empties the memory store; the session resumes
        from its disk checkpoint.
        """
        if not self._keys_since_restart:
            return
        projected = (self.retained_store_tokens
                     + 2 * self.ledgers[session].last_full_tokens)
        if projected < self.config.memory_budget_tokens:
            return
        self._maintenance_restarts += 1
        self._event("maintenance_restart", ordinal=self._maintenance_restarts,
                    retained_tokens=self.retained_store_tokens,
                    projected_tokens=projected)
        self._restart_server(f"maintenance-{self._maintenance_restarts}")

    def _maybe_align_restart(self, session: str, pending_chars: int) -> None:
        """Restart before a large extension of an unaligned in-memory prefix.

        The restored prefix then sits exactly on a frontier. A mitigation
        for the uniform divisor prefill step; the variable-step chunk plan
        removed that collapse (see RunConfig.align_min_context_tokens).
        """
        ledger = self.ledgers[session]
        event, cached = ledger.expected_event_and_cached()
        if event != "hit":
            return
        if cached < self.config.align_min_context_tokens:
            return
        if pending_chars < self.config.align_min_pending_chars:
            return
        if ledger.max_written_frontier is None:
            return
        if cached % 2048 == 0:
            return
        self._align_restarts += 1
        self._event("align_restart", ordinal=self._align_restarts,
                    session=session, unaligned_prefix=cached,
                    pending_chars=pending_chars)
        self._restart_server(f"align-{self._align_restarts}")

    def run_turn(self, session: str, turn: ScriptTurn) -> None:
        self._maybe_maintenance_restart(session)
        conversation = self.conversations[session]
        conversation.add_user(turn.user_text)
        self._event("turn", turn_id=turn.turn_id, session=session,
                    user_chars=len(turn.user_text))
        pending_chars = len(turn.user_text)
        step = 0
        while True:
            step += 1
            where = f"{turn.turn_id}/step{step}"
            self._maybe_align_restart(session, pending_chars)
            completion, _ = self._request(session, where)
            pending_chars = 0
            conversation.add_assistant_message(completion.message)
            try:
                calls = self._parse_calls(where, completion.content)
            except ToolCallParseError as e:
                self._event("note", where=where,
                            message=f"tool-call parse failed: {e}")
                break
            if not calls:
                if (step < self.config.max_turn_steps
                        and self._maybe_nudge(session, where)):
                    pending_chars = len(NUDGE_MESSAGE)
                    continue
                break
            if step >= self.config.max_turn_steps:
                self._event("note", where=where,
                            message="turn step cap reached with pending tool calls")
                break
            pending_chars = self._execute_calls(session, where, calls)

    # --- server lifecycle within the run ---

    def _restart_server(self, label: str) -> None:
        self.log(f"[roadtest] {label}: stopping server segment "
                 f"{self.controller.segment}")
        self.controller.stop()
        for ledger in self.ledgers.values():
            ledger.note_restart()
        self.health.note_restart()
        self._keys_since_restart = []
        self.controller.start()
        payload = self.controller.wait_healthy()
        for finding in self.health.verify(
                f"{label}/health", payload, expected_entries=self.total_frontiers):
            self._finding(finding)
        self._event("restart", label=label, segment=self.controller.segment)

    # --- proofs ---

    def _run_restart_proof(self) -> None:
        probe = self.script.probe_a
        self._maybe_maintenance_restart("a")
        conversation = self.conversations["a"]
        conversation.add_user(probe.user_text)
        self._event("turn", turn_id=probe.turn_id, session="a",
                    user_chars=len(probe.user_text))

        pre, pre_check = self._request("a", f"{probe.turn_id}/probe-pre-restart")
        if pre_check.prompt_tokens <= self.config.stride:
            self._finding(Finding(
                "probe.span", f"{probe.turn_id}/probe-pre-restart",
                f"probe turn added only {pre_check.prompt_tokens} suffix tokens; "
                f"it must exceed the stride ({self.config.stride}) so the "
                f"restored frontier lands inside the probe span"))

        self._restart_server("restart-1")
        mid, mid_check = self._request("a", f"{probe.turn_id}/probe-post-restart-1")
        if (mid.content, mid_check.completion_tokens) != (
                pre.content, pre_check.completion_tokens):
            self._finding(Finding(
                "identity.pre_vs_post_restart",
                f"{probe.turn_id}/probe-post-restart-1",
                f"restored completion differs from the live pre-restart "
                f"completion at temperature 0. pre: cached="
                f"{pre_check.cached_tokens} completion="
                f"{pre_check.completion_tokens} text={pre.content!r}; post: "
                f"cached={mid_check.cached_tokens} completion="
                f"{mid_check.completion_tokens} text={mid.content!r}. The arms "
                f"prefill different suffix lengths, so classify against the "
                f"identical-geometry arm before attributing this to the "
                f"restore path"))

        self._restart_server("restart-2")
        post, post_check = self._request("a", f"{probe.turn_id}/probe-post-restart-2")
        if post_check.cached_tokens != mid_check.cached_tokens:
            self._finding(Finding(
                "identity.geometry", f"{probe.turn_id}/probe-post-restart-2",
                f"the two disk restores landed on different frontiers "
                f"({mid_check.cached_tokens} vs {post_check.cached_tokens}); "
                f"the identical-geometry identity check could not run"))
        elif (post.content, post_check.completion_tokens) != (
                mid.content, mid_check.completion_tokens):
            self._finding(Finding(
                "identity.restore_replay",
                f"{probe.turn_id}/probe-post-restart-2",
                f"two disk restores at the same frontier "
                f"({post_check.cached_tokens}) and identical suffix geometry "
                f"produced different completions at temperature 0: "
                f"{mid.content!r} vs {post.content!r}"))

        self._event(
            "identity_probe",
            pre_cached=pre_check.cached_tokens,
            restored_cached=mid_check.cached_tokens,
            pre_matches_restored=(pre.content == mid.content),
            restore_replay_identical=(post.content == mid.content),
            pre_text=pre.content,
            restored_text=mid.content,
        )

        conversation.add_assistant_message(post.message)
        try:
            calls = self._parse_calls(f"{probe.turn_id}/probe-adopt", post.content)
        except ToolCallParseError:
            calls = []
        if calls:
            self._execute_calls("a", f"{probe.turn_id}/probe-adopt", calls)
            self.run_probe_followup(probe)

    def run_probe_followup(self, probe: ScriptTurn) -> None:
        """Drain unexpected tool calls the adopted probe answer made."""
        step = 0
        while step < self.config.max_turn_steps:
            step += 1
            where = f"{probe.turn_id}/probe-followup{step}"
            completion, _ = self._request("a", where)
            self.conversations["a"].add_assistant_message(completion.message)
            try:
                calls = self._parse_calls(where, completion.content)
            except ToolCallParseError:
                return
            if not calls:
                return
            self._execute_calls("a", where, calls)

    # --- the subagent scenario ---

    def run_subagent_scenario(self) -> None:
        """Spawn a sequential subagent mid-session and assert its chain.

        The parent announces the delegation as a normal scripted turn, then
        a ``SubagentRunner`` child runs the scripted brief to completion
        against the same server under the derived session key. Every child
        request flows through a fresh session ledger, so the counters must
        show a clean first miss (no contamination from the parent's prefix)
        followed by hits at exactly the child's own chain lengths, with the
        health cross-check holding throughout. The bounded result folds back
        into the parent as a tool result, and the following parent turn
        proves the fold-back kept the session append-only: its request must
        report the ledger-predicted cache event and ``cached_tokens``.
        """
        self.run_turn("a", self.script.delegate_a)
        parent = self.conversations["a"]
        expected_key = child_session_key(parent.session_cache_key, 1)
        self.ledgers["sub"] = SessionLedger(
            "sub", stride=self.health.stride,
            disk_enabled=self.health.disk_enabled)
        self._last_sent["sub"] = None
        runner = SubagentRunner(
            parent,
            _LedgeredChildClient(self, "sub"),
            self.registry,
            workdir=self.fixture.root,
            parse=self._parse_child_message,
            max_child_turns=self.config.max_turn_steps,
            max_tokens=self.config.max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        brief = SubagentBrief(
            task=self.script.subagent_task,
            system=self._system_prompt(SYSTEM_PROMPT_SUB),
            context=self.script.subagent_context,
        )
        self._event("subagent_spawn", session="sub",
                    expected_key=expected_key, task_chars=len(brief.task))
        result = runner.run(brief)
        self._event("subagent_result", session="sub", ok=result.ok,
                    reason=result.reason, turns=result.turns,
                    session_cache_key=result.session_cache_key,
                    result_chars=len(result.text), truncated=result.truncated)
        self.log(f"[roadtest] sub-01: ok={result.ok} reason={result.reason} "
                 f"turns={result.turns} key={result.session_cache_key}")
        if not result.ok:
            self._finding(Finding(
                "subagent.failed", "sub-01",
                f"child run failed ({result.reason}) after {result.turns} "
                f"request(s): {result.text}"))
        if result.session_cache_key != expected_key:
            self._finding(Finding(
                "subagent.session_key", "sub-01",
                f"child ran under {result.session_cache_key!r}, expected "
                f"the derived key {expected_key!r}"))
        if self.ledgers["sub"].requests < 2:
            self._finding(Finding(
                "subagent.no_reuse", "sub-01",
                f"the child made {self.ledgers['sub'].requests} request(s); "
                f"at least two are needed to observe reuse on its own chain"))
        last_child = self._last_sent.get("sub")
        if last_child is not None:
            (self.config.run_root / "conversation_sub.json").write_text(
                json.dumps(last_child, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8")
        self.run_turn("a", self.script.followup_a)

    # --- the full run ---

    def run(self) -> int:
        config = self.config
        config.run_root.mkdir(parents=True, exist_ok=True)
        self.fixture = generate_fixture(
            config.run_root / "workspace",
            data_file_count=config.data_file_count)
        self.script = build_script(self.fixture)
        self.conversations = {
            "a": Conversation(session_cache_key=SESSION_CACHE_KEYS["a"],
                              system=self._system_prompt(SYSTEM_PROMPT_A)),
            "b": Conversation(session_cache_key=SESSION_CACHE_KEYS["b"],
                              system=self._system_prompt(SYSTEM_PROMPT_B)),
        }
        self.ledgers = {
            "a": SessionLedger("a", stride=config.stride),
            "b": SessionLedger("b", stride=config.stride),
        }
        self._init_nudges()
        self._event("start", run_root=str(config.run_root),
                    stride=config.stride, budget_bytes=config.budget_bytes,
                    target_tokens=config.target_tokens,
                    context_limit=config.context_limit,
                    **self._loop_record())

        self.controller.start()
        payload = self.controller.wait_healthy()
        for finding in self.health.verify(
                "startup/health", payload, expected_entries=0):
            self._finding(finding)

        try:
            try:
                self._drive_phases()
            except ClientError as e:
                # A transport failure here means the serve process died; the
                # abort is itself run evidence.
                self._finding(Finding(
                    "run.aborted", "transport",
                    f"the session could not continue: {e}"))
                raise
        finally:
            self.controller.stop()
            self._write_summary()
        return 1 if self.findings else 0

    def _drive_phases(self) -> None:
        """Phase order.

        The restart proof runs right after the opening block, while the live
        pre-restart arm's unaligned prefill still sits inside the context
        range the engine demonstrably survives (finding: the gcd-collapsed
        aligned prefill aborts the process past roughly 20k context). The
        interleaving proof follows; its session resumes from disk after the
        alignment restarts the later turns trigger.
        """
        config = self.config
        for turn in self.script.opening_a:
            self.run_turn("a", turn)
        self._run_restart_proof()
        for turn in self.script.interleaved:
            self.run_turn(turn.session, turn)
        self.run_turn("b", self.script.resume_b)
        self.run_subagent_scenario()
        for turn in self.script.growth_a:
            self.run_turn("a", turn)
        for turn in self.script.extensions_a:
            if self.ledgers["a"].last_full_tokens >= config.target_tokens:
                break
            if self._at_context_ceiling(turn.turn_id):
                break
            self.run_turn("a", turn)
        self.run_turn("a", self.script.wrap_a)
        if self.ledgers["a"].last_full_tokens < config.target_tokens:
            self._finding(Finding(
                "run.target_tokens", "wrap",
                f"main session ended at "
                f"{self.ledgers['a'].last_full_tokens} context tokens, "
                f"short of the {config.target_tokens} target"))
        self._assert_repair_clean()
        self.completed = True

    # An extension turn appends the user text plus a full segment read as a
    # tool result, roughly this many prompt tokens at the upper end. The
    # bound covers tokenizers that encode the record lines expensively (a
    # segment measures near 10k tokens under the Qwen tokenizer against
    # under 4k for DeepSeek's).
    EXTENSION_TURN_TOKEN_ALLOWANCE = 12_000

    def _at_context_ceiling(self, turn_id: str) -> bool:
        """True when another extension turn could cross the declared limit.

        The server refuses a request whose prompt plus completion budget
        exceeds the model's declared context limit, so the growth phase
        stops while the next turn still fits and the run ends cleanly.
        """
        limit = self.config.context_limit
        if limit is None:
            return False
        projected = (self.ledgers["a"].last_full_tokens
                     + self.EXTENSION_TURN_TOKEN_ALLOWANCE
                     + 2 * self.config.max_tokens)
        if projected < limit:
            return False
        self._event("context_ceiling_stop", turn_id=turn_id,
                    last_full_tokens=self.ledgers["a"].last_full_tokens,
                    projected_tokens=projected, context_limit=limit)
        return True

    def _assert_repair_clean(self) -> None:
        """The repair-dependent dialect's alarm condition, checked at the end.

        The dialect adoption is conditioned on the repair telemetry staying
        at failed zero; any unsalvaged fire fails the run.
        """
        if self.config.loop is None or not self.config.loop.repair:
            return
        if self.repair_telemetry.failed:
            self._finding(Finding(
                "repair.failed", "wrap",
                f"repair telemetry ended with failed="
                f"{self.repair_telemetry.failed} of "
                f"{self.repair_telemetry.fires} fire(s); the dialect "
                f"contract requires failed == 0 (see repairs.jsonl)"))

    def run_smoke(self) -> int:
        """Short assertion pass against an already-running server.

        The middle tier between the fake-engine unit tests and the full run:
        a handful of small turns plus the subagent spawn scenario through
        the real serve path, with the same per-request ledger and
        health-delta checks, in well under the cost of one model load. Session prefixes are salted with a per-invocation
        tag so repeated smokes against one live server always start fresh
        chains, and health counters are verified as deltas over the
        attach-time baseline. The caller's config must disable the restart
        mechanisms; a smoke never manages the server lifecycle.
        """
        config = self.config
        config.run_root.mkdir(parents=True, exist_ok=True)
        tag = time.strftime("%Y%m%d-%H%M%S")
        self.fixture = generate_fixture(
            config.run_root / "workspace",
            data_file_count=config.data_file_count)
        self.script = build_script(self.fixture)
        salt = f"\nRun tag: smoke-{tag}."
        self.conversations = {
            "a": Conversation(session_cache_key=f"smoke-a-{tag}",
                              system=self._system_prompt(SYSTEM_PROMPT_A + salt)),
            "b": Conversation(session_cache_key=f"smoke-b-{tag}",
                              system=self._system_prompt(SYSTEM_PROMPT_B + salt)),
        }
        baseline = self.controller.wait_healthy()
        self.health.attach_baseline(baseline)
        self.ledgers = {
            name: SessionLedger(name, stride=self.health.stride,
                                disk_enabled=self.health.disk_enabled)
            for name in ("a", "b")
        }
        self._init_nudges()
        self._event("smoke_start", base_url=self.controller.base_url,
                    tag=tag, disk_enabled=self.health.disk_enabled,
                    stride=self.health.stride, **self._loop_record())
        try:
            for turn in self.script.opening_a[:5]:
                self.run_turn("a", turn)
            self.run_turn("b", self.script.resume_b)
            self.run_subagent_scenario()
            self._assert_repair_clean()
            self.completed = True
        finally:
            self._write_summary()
        return 1 if self.findings else 0

    def _write_summary(self) -> None:
        summary = {
            "wall_seconds": round(time.monotonic() - self._started, 1),
            "server_segments": self.controller.segment,
            "maintenance_restarts": self._maintenance_restarts,
            "align_restarts": self._align_restarts,
            "repair_telemetry": self.repair_telemetry.as_dict(),
            "nudges_fired": {name: policy.fired
                             for name, policy in self.nudges.items()},
            **self._loop_record(),
            "sessions": {
                name: {
                    "requests": ledger.requests,
                    "final_context_tokens": ledger.last_full_tokens,
                    "frontiers_written": sorted(ledger.written_frontiers),
                }
                for name, ledger in self.ledgers.items()
            },
            "final_health_disk": (
                (self.last_health.get("prompt_cache") or {}).get("disk")),
            "findings": [asdict(f) for f in self.findings],
        }
        path = self.config.run_root / "summary.json"
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for name, conversation in self.conversations.items():
            (self.config.run_root / f"conversation_{name}.json").write_text(
                json.dumps(conversation.request_messages(), ensure_ascii=False,
                           indent=1) + "\n", encoding="utf-8")
        self.log(f"[roadtest] summary written: {path}")
        self.log(json.dumps(summary, indent=2))


def _package_run_settings(package_dir: Path) -> tuple[LoopSettings | None, int | None]:
    """Loop settings and context limit a package directory declares.

    The loop settings resolve from the package's agentic profile alone
    (no user config layer: a certification run must not absorb host
    configuration), and resolution fails loudly on a malformed profile or
    an unknown dialect. A package without a profile contributes nothing
    and the run keeps the template-native request shape.
    """
    # Imported here: the manifest reader lives with the runtime cache code
    # and is only needed when a real package is on the command line.
    from moespresso.runtime.prefix_cache import declared_context_limit

    loop = None
    profile = load_agentic_profile(package_dir)
    if profile is not None:
        loop = resolve_loop_settings(
            package_profile=profile, use_user_config=False)
        loop.dialect_adapter()  # fail fast on an unknown dialect
    context_limit = None
    manifest_path = package_dir / "package_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        context_limit = declared_context_limit(manifest)
    return loop, context_limit


def _directory_bytes(root: Path) -> int:
    total = 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except OSError:
                pass
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-roadtest",
        description="Run the cumulative-session engine road-test against a "
                    "really served package (opt-in, GPU-bound).")
    parser.add_argument(
        "--package", default=os.environ.get("MOESPRESSO_ROADTEST_PACKAGE"),
        help="Packaged model directory to serve (env "
             "MOESPRESSO_ROADTEST_PACKAGE).")
    parser.add_argument(
        "--run-root", default=os.environ.get("MOESPRESSO_ROADTEST_ROOT"),
        help="Directory for the run's fixture, disk KV root, and logs "
             "(default: a fresh temp directory).")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("MOESPRESSO_ROADTEST_PORT",
                                                   "8399")))
    parser.add_argument("--stride", type=int, default=4096,
                        help="Disk checkpoint stride in tokens (multiple of 256).")
    parser.add_argument("--budget-bytes", type=int, default=450_000_000_000,
                        help="Disk KV byte budget; sized so eviction never "
                             "fires during the run.")
    parser.add_argument("--target-tokens", type=int, default=110_000)
    parser.add_argument("--memory-budget-tokens", type=int, default=120_000,
                        help="Live-cache budget in token-equivalents; the "
                             "driver restarts the server before a turn that "
                             "would pass it.")
    parser.add_argument("--no-mitigation-restarts", action="store_true",
                        help="Disable the alignment and maintenance restarts "
                             "(the driver-side mitigations for the divisor "
                             "prefill-step collapse and the store snapshot "
                             "retention). The certification run of the fixed "
                             "engine uses this so the session grows in place.")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--health-timeout", type=float, default=2700.0)
    parser.add_argument("--keep-disk", action="store_true",
                        help="Keep the disk KV payloads even on a clean run.")
    parser.add_argument("--gpu-lockdir",
                        default=os.environ.get("MOESPRESSO_ROADTEST_GPU_LOCKDIR"),
                        help="Shared-GPU lock directory (empty string disables).")
    parser.add_argument("--smoke", action="store_true",
                        help="Attach to an already-running server and run a "
                             "short assertion pass: no spawn, no model load, "
                             "no restarts. The fast tier of the fix loop. "
                             "Pass --package as well to resolve the agentic "
                             "profile the attached server is presumed to "
                             "serve.")
    parser.add_argument("--base-url", default=None,
                        help="Server URL for --smoke "
                             "(default: http://127.0.0.1:<port>).")
    parser.add_argument("--data-files", type=int, default=DATA_FILE_COUNT,
                        help="Fixture data segments to generate; sizes the "
                             "extension reserve for the context target.")
    parser.add_argument("--no-profile", action="store_true",
                        help="Ignore the package's agentic profile and drive "
                             "the template-native request shape (request "
                             "tools, strict DSML parsing, no repair).")
    args = parser.parse_args(argv)

    if args.run_root:
        run_root = Path(args.run_root)
    else:
        import tempfile
        run_root = Path(tempfile.mkdtemp(prefix="moespresso-roadtest-"))
    run_root.mkdir(parents=True, exist_ok=True)
    disk_root = run_root / "disk_kv"

    loop = None
    context_limit = None
    if args.package and not args.no_profile:
        loop, context_limit = _package_run_settings(Path(args.package))
        if loop is not None:
            print(f"[roadtest] agentic profile resolved: dialect="
                  f"{loop.dialect} repair={loop.repair} "
                  f"thinking_for_tools={loop.thinking_for_tools} "
                  f"reprompt={loop.reprompt_enabled}/{loop.reprompt_limit}")
        if context_limit is not None:
            print(f"[roadtest] declared context limit: {context_limit}")

    if args.smoke:
        base_url = args.base_url or f"http://127.0.0.1:{args.port}"
        controller = AttachedController(base_url, timeout=args.request_timeout)
        try:
            controller.wait_healthy()
        except ClientError as e:
            print(f"FAILED: no server to attach to at {base_url}: {e}")
            return 2
        config = RunConfig(
            run_root=run_root,
            stride=args.stride,
            budget_bytes=None,
            target_tokens=0,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            memory_budget_tokens=1 << 62,
            align_min_context_tokens=1 << 62,
            loop=loop,
            data_file_count=args.data_files,
            context_limit=context_limit,
        )
        print(f"[roadtest] smoke against {base_url}; run root: {run_root}")
        run = RoadtestRun(config, controller)
        code = run.run_smoke()
        if run.findings:
            print(f"[roadtest] smoke: {len(run.findings)} finding(s); "
                  f"see {run_root}/summary.json")
        else:
            print("[roadtest] smoke: all assertions held")
        return code

    if not args.package:
        parser.error("--package (or MOESPRESSO_ROADTEST_PACKAGE) is required")
    package_dir = Path(args.package)
    if not package_dir.is_dir():
        parser.error(f"package directory not found: {package_dir}")

    repo_root = Path(__file__).resolve().parents[4]
    probe_client = CompletionsClient(f"http://127.0.0.1:{args.port}", timeout=5.0)
    try:
        probe_client.health()
    except ClientError:
        pass
    else:
        print(f"FAILED: something already serves on port {args.port}")
        return 2

    controller = ServerController(
        package_dir=package_dir,
        repo_root=repo_root,
        port=args.port,
        disk_root=disk_root,
        stride=args.stride,
        budget_bytes=args.budget_bytes,
        log_dir=run_root / "logs",
        health_timeout=args.health_timeout,
    )
    config = RunConfig(
        run_root=run_root,
        stride=args.stride,
        budget_bytes=args.budget_bytes,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        keep_disk=args.keep_disk,
        memory_budget_tokens=(
            1 << 62 if args.no_mitigation_restarts
            else args.memory_budget_tokens),
        align_min_context_tokens=(
            1 << 62 if args.no_mitigation_restarts
            else RunConfig.align_min_context_tokens),
        loop=loop,
        data_file_count=args.data_files,
        context_limit=context_limit,
    )

    lock = None
    if args.gpu_lockdir != "":
        lock = GpuLock(args.gpu_lockdir or DEFAULT_GPU_LOCKDIR)
        lock.acquire()

    print(f"[roadtest] run root: {run_root}")
    run = RoadtestRun(config, controller)
    try:
        code = run.run()
    except ClientError as e:
        print(f"[roadtest] FAILED: the session could not continue: {e}")
        code = 1
    finally:
        controller.stop()
        if lock is not None:
            lock.release()
        disk_bytes = _directory_bytes(disk_root) if disk_root.exists() else 0
        print(f"[roadtest] disk KV root holds {disk_bytes / 1e9:.1f} GB "
              f"at {disk_root}")
        if (disk_root.exists() and not config.keep_disk and run.completed
                and not run.findings):
            shutil.rmtree(disk_root, ignore_errors=True)
            print("[roadtest] clean run: disk KV payloads removed")
        elif disk_root.exists():
            print("[roadtest] disk KV payloads kept for inspection")
    if not run.completed:
        print("[roadtest] run did not complete; see the logs under "
              f"{run_root}/logs")
        return max(code, 1)
    if run.findings:
        print(f"[roadtest] {len(run.findings)} finding(s) recorded; "
              f"see {run_root}/summary.json")
    else:
        print("[roadtest] all engine-counter assertions held")
    return code
