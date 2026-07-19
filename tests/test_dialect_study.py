"""The dialect-study harness, driven end to end with a fake client.

Covers the episode set's integrity, the planted workspace anchors, the three
dialect adapters (prompt surface, attempt detection, parsing, result
routing), and the episode loop's bookkeeping: success judging, parse-failure
feedback, the repair path, and the arm summary. No server or GPU is
involved; the fake client returns scripted completions and captures what the
loop sends.
"""

from __future__ import annotations

import json

import pytest

from moespresso.agentlib.client import ChatCompletion
from moespresso.agentlib.dialect_study.dialects import (
    ARM_NAMES,
    BASE_SYSTEM,
    dialect_for,
)
from moespresso.agentlib.dialect_study.episodes import (
    SELFTEST_CODE,
    VERSION_VALUE,
    build_episodes,
    prepare_workspace,
)
from moespresso.agentlib.dialect_study.run import (
    StudyConfig,
    run_arm,
    run_episode,
    summarize,
)
from moespresso.agentlib.tools import build_core_registry
from moespresso.toolcalls.dsml import DSML_TOKEN
from moespresso.toolcalls.types import ToolCallParseError

REGISTRY = build_core_registry()


class FakeClient:
    """Scripted completions; captures each request's messages and options."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def complete(self, conversation, *, tools=None, temperature=None,
                 top_p=None, max_tokens=None, chat_template_kwargs=None,
                 **_ignored):
        self.requests.append({
            "messages": conversation.request_messages(),
            "tools": tools,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "chat_template_kwargs": chat_template_kwargs,
        })
        content = self.replies.pop(0)
        return ChatCompletion(
            message={"role": "assistant", "content": content},
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )


def _episode(episode_id: str):
    matches = [e for e in build_episodes() if e.episode_id == episode_id]
    assert len(matches) == 1
    return matches[0]


def _config(tmp_path, arm: str, repair: bool = False) -> StudyConfig:
    return StudyConfig(
        arm=arm,
        out_dir=tmp_path / "out",
        workspace_root=tmp_path / "ws",
        repair=repair,
    )


def _xml_call(name: str, params: dict) -> str:
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in params.items():
        lines.extend([f"<parameter={key}>", str(value), "</parameter>"])
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def _envelope_reply(commands, task_complete, analysis="working") -> str:
    return json.dumps({
        "analysis": analysis,
        "plan": "next",
        "commands": commands,
        "task_complete": task_complete,
    })


def _dsml_call(name: str, params: dict) -> str:
    t = DSML_TOKEN
    lines = [f"<{t}tool_calls>", f'<{t}invoke name="{name}">']
    for key, value in params.items():
        lines.append(
            f'<{t}parameter name="{key}" string="true">{value}</{t}parameter>')
    lines.extend([f"</{t}invoke>", f"</{t}tool_calls>"])
    return "\n".join(lines)


# --- episode set integrity -----------------------------------------------------

def test_episode_set_shape():
    episodes = build_episodes()
    assert len(episodes) == 15
    ids = [e.episode_id for e in episodes]
    assert len(set(ids)) == 15
    families = [e.family for e in episodes]
    assert families.count("read_answer") == 5
    assert families.count("grep_edit") == 5
    assert families.count("command") == 5


def test_workspace_plants_the_anchors(tmp_path):
    prepare_workspace(tmp_path)
    assert (tmp_path / "VERSION").read_text() == VERSION_VALUE + "\n"
    assert "9741" in (tmp_path / "config" / "settings.ini").read_text()
    assert "rc-4471" in (tmp_path / "notes" / "todo.txt").read_text()
    assert SELFTEST_CODE in (tmp_path / "scripts" / "selftest.sh").read_text()
    assert "len(values) - 1" in (tmp_path / "src" / "metrics.py").read_text()


def test_answer_matching_is_word_bounded(tmp_path):
    prepare_workspace(tmp_path)
    c1 = _episode("c1-src-count")
    assert c1.succeeded(tmp_path, "There are 21 files.")
    assert not c1.succeeded(tmp_path, "There are 215 files.")
    assert not c1.succeeded(tmp_path, "I could not count them.")


def test_goal_check_judges_workspace_state(tmp_path):
    prepare_workspace(tmp_path)
    g2 = _episode("g2-fix-divisor")
    assert not g2.succeeded(tmp_path, "fixed it")
    metrics = tmp_path / "src" / "metrics.py"
    metrics.write_text(metrics.read_text().replace(
        "return total / (len(values) - 1)", "return total / len(values)"))
    assert g2.succeeded(tmp_path, "fixed it")


def test_call_correctness_matchers():
    r1 = _episode("r1-version")
    from moespresso.toolcalls.types import ToolCall
    assert r1.call_is_correct(ToolCall("read_file", {"path": "VERSION"}))
    assert not r1.call_is_correct(ToolCall("bash", {"command": "cat VERSION"}))
    assert not r1.call_is_correct(ToolCall("read_file", {"path": "README.md"}))


# --- dialect adapters ------------------------------------------------------------

def test_arm_names_cover_the_three_dialects():
    assert set(ARM_NAMES) == {"native", "envelope", "dsml"}
    with pytest.raises(ValueError, match="unknown dialect arm"):
        dialect_for("openai")


def test_native_dialect_surface():
    dialect = dialect_for("native")
    assert dialect.system_prompt(REGISTRY) == BASE_SYSTEM
    tools = dialect.request_tools(REGISTRY)
    assert [t["function"]["name"] for t in tools] == [
        "read_file", "grep", "edit", "bash"]
    assert dialect.attempted("<tool_call>")
    assert dialect.attempted("<function=grep>")
    assert not dialect.attempted("plain prose")


def test_envelope_dialect_surface():
    dialect = dialect_for("envelope")
    prompt = dialect.system_prompt(REGISTRY)
    assert prompt.startswith(BASE_SYSTEM)
    assert '"name": "read_file"' in prompt
    assert "task_complete" in prompt
    assert dialect.request_tools(REGISTRY) is None
    assert dialect.attempted("anything at all")


def test_dsml_dialect_surface():
    dialect = dialect_for("dsml")
    prompt = dialect.system_prompt(REGISTRY)
    assert prompt.startswith(BASE_SYSTEM)
    assert DSML_TOKEN in prompt
    assert '"name": "bash"' in prompt
    assert dialect.request_tools(REGISTRY) is None
    assert dialect.attempted(f"<{DSML_TOKEN}invoke")
    assert not dialect.attempted("plain prose")


def test_dsml_naked_invoke_is_a_parse_failure_not_a_final_answer():
    dialect = dialect_for("dsml")
    naked = f'<{DSML_TOKEN}invoke name="bash">'
    with pytest.raises(ToolCallParseError, match="outside"):
        dialect.parse_turn(naked, REGISTRY)


def test_answer_text_excludes_tool_blocks_and_think():
    dialect = dialect_for("native")
    content = ("<think>\nplanning\n</think>\nChecking now.\n"
               + _xml_call("read_file", {"path": "VERSION"}))
    turn = dialect.parse_turn(content, REGISTRY)
    assert not turn.final
    assert "planning" not in turn.answer_text
    assert "<tool_call>" not in turn.answer_text
    assert "Checking now." in turn.answer_text


# --- episode loop, per arm --------------------------------------------------------

def test_native_episode_runs_to_success(tmp_path):
    episode = _episode("r1-version")
    workdir = tmp_path / "ws"
    prepare_workspace(workdir)
    client = FakeClient([
        _xml_call("read_file", {"path": "VERSION"}),
        f"The version is {VERSION_VALUE}.",
    ])
    record, malformed = run_episode(
        client, dialect_for("native"), REGISTRY, episode, workdir,
        _config(tmp_path, "native"))
    assert record["success"] is True
    assert record["requests"] == 2
    assert record["attempts"] == 1
    assert record["strict_parse_failures"] == 0
    assert record["final_reached"] is True
    assert malformed == []
    assert record["calls"] == [{
        "tool": "read_file", "arguments": {"path": "VERSION"},
        "correct": True, "ok": True,
    }]
    # per-turn log carries the by-ordinal parse outcome
    assert record["turn_log"] == [
        {"step": 1, "attempted": True, "strict_parse": True, "repaired": False},
        {"step": 2, "attempted": False, "strict_parse": True, "repaired": False},
    ]
    # repair is off, so the telemetry counters stay zero
    assert record["repair_telemetry"] == {"fires": 0, "salvaged": 0, "failed": 0}
    # request surface: tools ride the request; thinking is off
    assert client.requests[0]["tools"] is not None
    assert client.requests[0]["chat_template_kwargs"] == {
        "enable_thinking": False}
    assert client.requests[0]["temperature"] == 0.0
    # the tool result went back as a tool-role message
    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert VERSION_VALUE in client.requests[1]["messages"][3]["content"]


def test_envelope_episode_edits_and_reports_results_as_user(tmp_path):
    episode = _episode("g3-version-bump")
    workdir = tmp_path / "ws"
    prepare_workspace(workdir)
    client = FakeClient([
        _envelope_reply(
            [{"tool": "edit",
              "args": {"path": "VERSION", "old_string": "7.4.2",
                       "new_string": "7.4.3"}}],
            False),
        _envelope_reply([], True, analysis="Updated VERSION to 7.4.3."),
    ])
    record, _ = run_episode(
        client, dialect_for("envelope"), REGISTRY, episode, workdir,
        _config(tmp_path, "envelope"))
    assert record["success"] is True
    assert (workdir / "VERSION").read_text().strip() == "7.4.3"
    assert record["attempts"] == 2  # every envelope turn is an attempt
    assert client.requests[0]["tools"] is None
    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert client.requests[1]["messages"][3]["content"].startswith("Tool results:")


def test_dsml_episode_runs_a_real_command(tmp_path):
    episode = _episode("c3-selftest")
    workdir = tmp_path / "ws"
    prepare_workspace(workdir)
    client = FakeClient([
        _dsml_call("bash", {"command": "sh scripts/selftest.sh"}),
        f"The selftest prints code {SELFTEST_CODE}.",
    ])
    record, _ = run_episode(
        client, dialect_for("dsml"), REGISTRY, episode, workdir,
        _config(tmp_path, "dsml"))
    assert record["success"] is True
    assert record["calls"][0]["ok"] is True
    assert SELFTEST_CODE in client.requests[1]["messages"][3]["content"]


def test_parse_failure_feeds_back_and_is_counted(tmp_path):
    episode = _episode("r1-version")
    workdir = tmp_path / "ws"
    prepare_workspace(workdir)
    naked = ("<function=read_file>\n<parameter=path>\nVERSION\n"
             "</parameter>\n</function>")
    client = FakeClient([
        naked,
        _xml_call("read_file", {"path": "VERSION"}),
        f"The version is {VERSION_VALUE}.",
    ])
    record, malformed = run_episode(
        client, dialect_for("native"), REGISTRY, episode, workdir,
        _config(tmp_path, "native"))
    assert record["strict_parse_failures"] == 1
    assert record["unrecovered_failures"] == 1
    assert record["repaired_turns"] == 0
    assert record["attempts"] == 2
    assert record["success"] is True
    assert len(malformed) == 1 and malformed[0]["content"] == naked
    # the loop fed a correction prompt back as a user message
    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "could not be parsed" in client.requests[1]["messages"][3]["content"]


def test_repair_salvages_the_same_failure(tmp_path):
    episode = _episode("r1-version")
    workdir = tmp_path / "ws"
    prepare_workspace(workdir)
    naked = ("<function=read_file>\n<parameter=path>\nVERSION\n"
             "</parameter>\n</function>")
    client = FakeClient([
        naked,
        f"The version is {VERSION_VALUE}.",
    ])
    record, malformed = run_episode(
        client, dialect_for("native"), REGISTRY, episode, workdir,
        _config(tmp_path, "native", repair=True))
    assert record["strict_parse_failures"] == 1
    assert record["repaired_turns"] == 1
    assert record["unrecovered_failures"] == 0
    assert record["success"] is True
    assert malformed[0]["repaired"] is True
    assert record["calls"][0]["tool"] == "read_file"
    # the repair telemetry counts the fire and the salvage
    assert record["repair_telemetry"] == {"fires": 1, "salvaged": 1, "failed": 0}
    assert record["turn_log"][0] == {
        "step": 1, "attempted": True, "strict_parse": False, "repaired": True}


def test_reprompt_rescues_a_prose_first_turn(tmp_path):
    # The recorded failure class: a first turn of plain prose with no call
    # markup reads as a final answer. With the nudge policy on, the loop
    # re-prompts once and the episode completes.
    episode = _episode("c3-selftest")
    workdir = tmp_path / "ws"
    prepare_workspace(workdir)
    client = FakeClient([
        "I need to run the selftest script first.",
        _dsml_call("bash", {"command": "sh scripts/selftest.sh"}),
        f"The selftest prints code {SELFTEST_CODE}.",
    ])
    config = StudyConfig(arm="dsml", out_dir=tmp_path / "out",
                         workspace_root=tmp_path / "w", reprompt=True)
    record, _ = run_episode(
        client, dialect_for("dsml"), REGISTRY, episode, workdir, config)
    assert record["reprompts"] == 1
    assert record["success"] is True
    assert record["requests"] == 3
    assert [c["tool"] for c in record["calls"]] == ["bash"]
    # the nudge went back as a user message
    nudge_turn = client.requests[1]["messages"][3]
    assert nudge_turn["role"] == "user"
    assert "No tool call was found" in nudge_turn["content"]


def test_reprompt_fires_once_then_accepts_prose(tmp_path):
    episode = _episode("c3-selftest")
    workdir = tmp_path / "ws"
    prepare_workspace(workdir)
    client = FakeClient([
        "I would run the selftest script.",
        "Still just describing the plan.",
    ])
    config = StudyConfig(arm="dsml", out_dir=tmp_path / "out",
                         workspace_root=tmp_path / "w", reprompt=True)
    record, _ = run_episode(
        client, dialect_for("dsml"), REGISTRY, episode, workdir, config)
    assert record["reprompts"] == 1
    assert record["requests"] == 2
    assert record["final_reached"] is True
    assert record["success"] is False


def test_reprompt_does_not_fire_on_final_answer_after_tool_results(tmp_path):
    episode = _episode("c3-selftest")
    workdir = tmp_path / "ws"
    prepare_workspace(workdir)
    client = FakeClient([
        _dsml_call("bash", {"command": "sh scripts/selftest.sh"}),
        f"The selftest prints code {SELFTEST_CODE}.",
    ])
    config = StudyConfig(arm="dsml", out_dir=tmp_path / "out",
                         workspace_root=tmp_path / "w", reprompt=True)
    record, _ = run_episode(
        client, dialect_for("dsml"), REGISTRY, episode, workdir, config)
    assert record["reprompts"] == 0
    assert record["requests"] == 2
    assert record["success"] is True


def test_reprompt_off_keeps_the_prose_final_behavior(tmp_path):
    episode = _episode("c3-selftest")
    workdir = tmp_path / "ws"
    prepare_workspace(workdir)
    client = FakeClient(["I need to run the selftest script first."])
    record, _ = run_episode(
        client, dialect_for("dsml"), REGISTRY, episode, workdir,
        _config(tmp_path, "dsml"))
    assert record["reprompts"] == 0
    assert record["requests"] == 1
    assert record["final_reached"] is True
    assert record["success"] is False


def test_run_arm_writes_report_and_corpus(tmp_path):
    episodes = (_episode("r1-version"),)
    client = FakeClient([
        "<tool_call>\n<function=read_file>\n<parameter=path>\nVERSION",
        _xml_call("read_file", {"path": "VERSION"}),
        f"The version is {VERSION_VALUE}.",
    ])
    config = _config(tmp_path, "native")
    report = run_arm(client, config, episodes=episodes, log_fn=lambda line: None)
    summary = report["summary"]
    assert summary["episodes"] == 1
    assert summary["successes"] == 1
    assert summary["attempts"] == 2
    assert summary["strict_parse_failures"] == 1
    assert summary["parse_rate"] == 0.5
    assert summary["calls"] == 1
    assert summary["call_correctness"] == 1.0
    assert summary["completion_tokens"] == 60
    on_disk = json.loads((config.out_dir / "native.json").read_text())
    assert on_disk["summary"] == summary
    assert on_disk["thinking"] == "off"
    assert summary["repair_telemetry"] == {"fires": 0, "salvaged": 0, "failed": 0}
    corpus_lines = (config.out_dir / "malformed_native.jsonl").read_text()
    assert "read_file" in corpus_lines


def test_labeled_batch_keeps_its_own_files_and_stamps_sampling(tmp_path):
    # A sweep point runs as a labeled batch: the label suffixes the report
    # and corpus filenames and the sampling point is stamped on the report
    # and on every corpus entry. Sampling values here are arbitrary test
    # numbers; this remains an investigation fixture without product status.
    episodes = (_episode("r1-version"),)
    client = FakeClient([
        "<tool_call>\n<function=read_file>\n<parameter=path>\nVERSION",
        _xml_call("read_file", {"path": "VERSION"}),
        f"The version is {VERSION_VALUE}.",
    ])
    config = StudyConfig(
        arm="native", out_dir=tmp_path / "out",
        workspace_root=tmp_path / "ws",
        temperature=0.25, top_p=0.5, label="pt-r1")
    report = run_arm(client, config, episodes=episodes, log_fn=lambda line: None)
    assert report["label"] == "pt-r1"
    assert report["temperature"] == 0.25
    on_disk_path = config.out_dir / "native-pt-r1.json"
    assert on_disk_path.exists()
    corpus_lines = (config.out_dir / "malformed_native-pt-r1.jsonl").read_text()
    entry = json.loads(corpus_lines.splitlines()[0])
    assert entry["label"] == "pt-r1"
    assert entry["temperature"] == 0.25
    assert entry["top_p"] == 0.5


def test_summarize_handles_empty():
    summary = summarize([])
    assert summary["episodes"] == 0
    assert summary["parse_rate"] is None
    assert summary["task_success_rate"] is None
