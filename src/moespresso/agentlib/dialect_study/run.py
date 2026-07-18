"""Episode loop, scoring, and the study CLI.

One invocation runs one arm over the episode set against an already-running
server, so a batch stays short and the shared GPU measurement lock is held
only for that batch. The loop is dialect-blind: the dialect adapter supplies
the system prompt, the request tool surface, parsing, repair, and result
routing, and the loop executes calls for real on a per-episode scratch
workspace through the standard execution choke point.

Per assistant turn the loop records whether the content looked like a
tool-call attempt, whether strict parsing accepted it, and, with repair
enabled, whether the repair layer salvaged it. Raw content of every strict
parse failure is appended to a malformed-output corpus file; the repair
transformations are developed and unit-tested against that corpus.

Thinking stays off for the whole study through ``chat_template_kwargs``.
Sampling defaults to greedy (temperature 0, top_p 1); the sweep batches set
the package's recommended profile through the CLI flags, and ``label`` keeps
each sweep point and repeat in its own report and corpus files.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from moespresso.agentlib.client import CompletionsClient
from moespresso.agentlib.conversation import Conversation
from moespresso.agentlib.dialect_study.dialects import ARM_NAMES, dialect_for
from moespresso.agentlib.dialect_study.episodes import (
    Episode,
    build_episodes,
    prepare_workspace,
)
from moespresso.agentlib.execution import execute_tool_call
from moespresso.agentlib.loop_policy import NUDGE_MESSAGE, ToolNudgePolicy
from moespresso.agentlib.repair import RepairTelemetry
from moespresso.agentlib.toolcalls import ToolCallParseError
from moespresso.agentlib.tools import ToolRegistry, build_core_registry

DEFAULT_MAX_TOKENS = 512
THINKING_KWARGS = {"enable_thinking": False}


@dataclass(frozen=True)
class StudyConfig:
    """One batch: an arm, its sampling point, and where records land.

    ``label`` names the batch (sampling point plus repeat ordinal); it
    suffixes the report and corpus filenames so sweep points and repeats
    never overwrite each other or mix corpora.
    """

    arm: str
    out_dir: Path
    workspace_root: Path
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    top_p: float = 1.0
    repair: bool = False
    reprompt: bool = False
    label: str = ""

    @property
    def file_stem(self) -> str:
        stem = self.arm + ("-repair" if self.repair else "")
        return stem + (f"-{self.label}" if self.label else "")


def run_episode(client, dialect, registry: ToolRegistry, episode: Episode,
                workdir: Path, config: StudyConfig) -> tuple[dict, list[dict]]:
    """Run one episode to completion; return its record and malformed corpus."""
    conversation = Conversation(system=dialect.system_prompt(registry))
    conversation.add_user(episode.prompt)
    record = {
        "episode_id": episode.episode_id,
        "family": episode.family,
        "arm": dialect.name,
        "repair": config.repair,
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "attempts": 0,
        "strict_parse_failures": 0,
        "repaired_turns": 0,
        "unrecovered_failures": 0,
        "calls": [],
        "finish_reasons": [],
        "cache_events": [],
        "cached_tokens": [],
        "request_prompt_tokens": [],
        "request_completion_tokens": [],
        "turn_log": [],
        "reprompts": 0,
        "final_reached": False,
    }
    malformed: list[dict] = []
    telemetry = RepairTelemetry()
    nudge = ToolNudgePolicy() if config.reprompt else None
    last_answer = ""
    started = time.monotonic()
    for step in range(1, episode.max_steps + 1):
        completion = client.complete(
            conversation,
            tools=dialect.request_tools(registry),
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            chat_template_kwargs=dict(THINKING_KWARGS),
        )
        record["requests"] += 1
        usage = completion.usage or {}
        record["prompt_tokens"] += usage.get("prompt_tokens", 0)
        record["completion_tokens"] += usage.get("completion_tokens", 0)
        record["request_prompt_tokens"].append(usage.get("prompt_tokens", 0))
        record["request_completion_tokens"].append(
            usage.get("completion_tokens", 0))
        record["finish_reasons"].append(completion.finish_reason)
        cache = completion.prompt_cache or {}
        record["cache_events"].append(cache.get("event"))
        record["cached_tokens"].append(completion.cached_tokens)
        conversation.add_assistant_message(completion.message)
        content = completion.content or ""
        attempted = dialect.attempted(content)
        if attempted:
            record["attempts"] += 1
        # Per-turn log, keyed by the request step: the by-ordinal axis for
        # malformation-rate analysis (does markup fidelity degrade at later
        # calls within an episode).
        turn_entry = {"step": step, "attempted": attempted,
                      "strict_parse": True, "repaired": False}
        record["turn_log"].append(turn_entry)
        turn = None
        try:
            turn = dialect.parse_turn(content, registry)
        except ToolCallParseError as error:
            turn_entry["strict_parse"] = False
            record["strict_parse_failures"] += 1
            entry = {
                "arm": dialect.name,
                "label": config.label,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "episode_id": episode.episode_id,
                "step": step,
                "error": str(error),
                "content": content,
            }
            if config.repair:
                try:
                    turn = dialect.repair_turn(content, registry)
                    record["repaired_turns"] += 1
                    entry["repaired"] = True
                    turn_entry["repaired"] = True
                    telemetry.record(salvaged=True)
                except ToolCallParseError as repair_error:
                    entry["repair_error"] = str(repair_error)
                    telemetry.record(salvaged=False)
            malformed.append(entry)
            if turn is None:
                record["unrecovered_failures"] += 1
                conversation.add_user(dialect.parse_feedback(error))
                continue
        if turn.answer_text:
            last_answer = turn.answer_text
        if turn.final:
            if nudge is not None and nudge.wants_reprompt(
                    final=True,
                    calls_in_turn=len(turn.calls),
                    calls_before_turn=len(record["calls"])):
                nudge.note_fired()
                record["reprompts"] += 1
                conversation.add_user(NUDGE_MESSAGE)
                continue
            record["final_reached"] = True
            break
        outcomes = []
        for call in turn.calls:
            result = execute_tool_call(registry, call, workdir=workdir)
            record["calls"].append({
                "tool": call.name,
                "arguments": call.arguments,
                "correct": episode.call_is_correct(call),
                "ok": result.ok,
            })
            outcomes.append((call, result))
        dialect.append_results(conversation, outcomes)
    record["wall_seconds"] = round(time.monotonic() - started, 3)
    record["answer_text"] = last_answer
    record["success"] = episode.succeeded(workdir, last_answer)
    record["repair_telemetry"] = telemetry.as_dict()
    return record, malformed


def summarize(records: list[dict]) -> dict:
    """Aggregate the per-arm metrics the study table reports."""
    calls = [call for record in records for call in record["calls"]]
    attempts = sum(r["attempts"] for r in records)
    strict_failures = sum(r["strict_parse_failures"] for r in records)
    episodes = len(records)
    completion_tokens = sum(r["completion_tokens"] for r in records)
    return {
        "episodes": episodes,
        "successes": sum(1 for r in records if r["success"]),
        "task_success_rate": _rate(
            sum(1 for r in records if r["success"]), episodes),
        "attempts": attempts,
        "strict_parse_failures": strict_failures,
        "parse_rate": _rate(attempts - strict_failures, attempts),
        "repaired_turns": sum(r["repaired_turns"] for r in records),
        "unrecovered_failures": sum(r["unrecovered_failures"] for r in records),
        "repair_telemetry": _total_telemetry(records).as_dict(),
        "reprompts": sum(r.get("reprompts", 0) for r in records),
        "calls": len(calls),
        "correct_calls": sum(1 for c in calls if c["correct"]),
        "call_correctness": _rate(
            sum(1 for c in calls if c["correct"]), len(calls)),
        "requests": sum(r["requests"] for r in records),
        "prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "completion_tokens": completion_tokens,
        "mean_completion_tokens_per_episode": (
            round(completion_tokens / episodes, 1) if episodes else None),
        "wall_seconds": round(sum(r["wall_seconds"] for r in records), 1),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def _total_telemetry(records: list[dict]) -> RepairTelemetry:
    total = RepairTelemetry()
    for record in records:
        block = record.get("repair_telemetry") or {}
        total.add(RepairTelemetry(**block))
    return total


def run_arm(client, config: StudyConfig,
            episodes: tuple[Episode, ...] | None = None,
            log_fn=print) -> dict:
    """Run every episode for one arm; write and return the arm report."""
    dialect = dialect_for(config.arm)
    registry = build_core_registry()
    episodes = build_episodes() if episodes is None else episodes
    records = []
    corpus: list[dict] = []
    for episode in episodes:
        # The stem keys the workspace too: a repeat must not inherit goal
        # state (an already-bumped VERSION, a leftover build_tag.txt) from
        # an earlier batch.
        workdir = Path(config.workspace_root) / config.file_stem / episode.episode_id
        prepare_workspace(workdir)
        record, malformed = run_episode(
            client, dialect, registry, episode, workdir, config)
        corpus.extend(malformed)
        records.append(record)
        telemetry = record["repair_telemetry"]
        log_fn(f"[dialect-study] {config.file_stem}/{episode.episode_id}: "
               f"success={record['success']} requests={record['requests']} "
               f"parse_failures={record['strict_parse_failures']} "
               f"repair={telemetry['fires']}/{telemetry['salvaged']}"
               f"/{telemetry['failed']} "
               f"reprompts={record['reprompts']} "
               f"completion_tokens={record['completion_tokens']} "
               f"wall={record['wall_seconds']}s")
    report = {
        "arm": config.arm,
        "repair": config.repair,
        "reprompt": config.reprompt,
        "label": config.label,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "thinking": "off",
        "summary": summarize(records),
        "episodes": records,
    }
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{config.file_stem}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if corpus:
        corpus_path = out_dir / f"malformed_{config.file_stem}.jsonl"
        with open(corpus_path, "a", encoding="utf-8") as handle:
            for entry in corpus:
                handle.write(json.dumps(entry) + "\n")
    log_fn(f"[dialect-study] {config.file_stem}: "
           + json.dumps(report["summary"]))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso.agentlib.dialect_study",
        description="Run one dialect arm of the Ornith tool-call study "
                    "against a running server.")
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=None,
                        help="Scratch root for episode workspaces "
                             "(default: <out-dir>/workspaces).")
    parser.add_argument("--repair", action="store_true",
                        help="Salvage strict parse failures through the "
                             "repair layer before feeding back an error.")
    parser.add_argument("--reprompt", action="store_true",
                        help="Nudge once when a final turn arrives before "
                             "any tool call (the tool-nudge loop policy).")
    parser.add_argument("--episodes", nargs="*", default=None,
                        help="Episode ids to run (default: all).")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default 0: greedy).")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--label", default="",
                        help="Batch label (sampling point, repeat ordinal); "
                             "suffixes the report and corpus filenames.")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Per-request client timeout in seconds.")
    parser.add_argument("--lockdir", default=None,
                        help="GPU measurement lock directory (default: the "
                             "shared convention). Pass --no-lock to skip.")
    parser.add_argument("--no-lock", action="store_true",
                        help="Do not take the GPU measurement lock (offline "
                             "or single-owner runs).")
    args = parser.parse_args(argv)

    episodes = build_episodes()
    if args.episodes:
        wanted = set(args.episodes)
        episodes = tuple(e for e in episodes if e.episode_id in wanted)
        missing = wanted - {e.episode_id for e in episodes}
        if missing:
            parser.error(f"unknown episode id(s): {', '.join(sorted(missing))}")

    config = StudyConfig(
        arm=args.arm,
        out_dir=args.out_dir,
        workspace_root=args.workspace_root or (args.out_dir / "workspaces"),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repair=args.repair,
        reprompt=args.reprompt,
        label=args.label,
    )
    client = CompletionsClient(args.base_url, timeout=args.timeout)
    client.health()  # fail fast when no server is up

    lock = None
    if not args.no_lock:
        from moespresso.agentlib.roadtest.server import (
            DEFAULT_GPU_LOCKDIR,
            GpuLock,
        )
        lock = GpuLock(args.lockdir or DEFAULT_GPU_LOCKDIR,
                       holder="dialect-study")
        lock.acquire(log_fn=lambda line: print(
            line.replace("[roadtest]", "[dialect-study]")))
    try:
        run_arm(client, config, episodes=episodes)
    finally:
        if lock is not None:
            lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
