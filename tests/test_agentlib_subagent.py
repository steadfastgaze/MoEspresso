"""Sequential subagent skeleton against a scripted fake client (no GPU).

The fake client stands in for the completions client: it records every
request's message snapshot, session key, and keyword arguments, then returns
the next scripted completion or raises the scripted error. The tests pin the
contract: a child runs a brief to completion in a forked conversation under
a derived session cache key, its bounded result folds back into the parent
as a tool result, the parent history stays a strict prefix extension across
the spawn, spawning is sequential and depth-limited, and child failures
surface as ok=False results instead of exceptions.
"""

from __future__ import annotations

import json

import pytest

from moespresso.agentlib import (
    MAX_SUBAGENT_DEPTH,
    ChatCompletion,
    ClientError,
    Conversation,
    SubagentBrief,
    SubagentConcurrencyError,
    SubagentConfigError,
    SubagentDepthError,
    SubagentRunner,
    ToolSpec,
    build_core_registry,
    child_session_key,
)


def _assistant(content=None, tool_calls=None) -> ChatCompletion:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return ChatCompletion(message=message, finish_reason="stop")


def _tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


class FakeClient:
    """Scripted stand-in for CompletionsClient.complete."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def complete(self, conversation, **kwargs):
        self.requests.append({
            "messages": conversation.request_messages(),
            "session_cache_key": conversation.session_cache_key,
            "kwargs": kwargs,
        })
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _make_runner(script, tmp_path, **overrides):
    """A parent mid-conversation (its assistant turn requested a subagent),
    a fake client with the given script, and a runner over the core tools."""
    parent = Conversation(session_cache_key="parent-key", system="parent sys")
    parent.add_user("please delegate this")
    parent.add_assistant_message({
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call("subagent", {"task": "t"},
                                  call_id="call_sub")],
    })
    client = FakeClient(script)
    runner = SubagentRunner(parent, client, build_core_registry(),
                            workdir=tmp_path, **overrides)
    return parent, client, runner


# --- the full child run ---


def test_full_child_run_executes_tools_and_folds_back(tmp_path):
    (tmp_path / "notes.txt").write_text("the answer is 42\n", encoding="utf-8")
    script = [
        _assistant(tool_calls=[_tool_call("read_file", {"path": "notes.txt"})]),
        _assistant("the file says 42"),
    ]
    parent, client, runner = _make_runner(script, tmp_path)
    result = runner.run(SubagentBrief(task="read notes.txt", system="child sys"),
                        tool_call_id="call_sub")
    assert result.ok
    assert result.reason == "completed"
    assert result.turns == 2
    assert result.text == "the file says 42"
    assert not result.truncated
    # The tool ran for real: its output is in the child's second request.
    second = client.requests[1]["messages"]
    assert second[-1] == {"role": "tool", "content": "the answer is 42\n",
                          "tool_call_id": "call_1"}
    # The fold-back is the result text, as a tool message on the parent.
    assert parent.request_messages()[-1] == {
        "role": "tool", "content": "the file says 42",
        "tool_call_id": "call_sub"}


def test_child_starts_from_the_brief_only(tmp_path):
    parent, client, runner = _make_runner([_assistant("done")], tmp_path)
    runner.run(SubagentBrief(task="the task", system="child sys"))
    assert client.requests[0]["messages"] == [
        {"role": "system", "content": "child sys"},
        {"role": "user", "content": "the task"},
    ]


def test_context_crosses_the_boundary_explicitly(tmp_path):
    parent, client, runner = _make_runner([_assistant("done")], tmp_path)
    runner.run(SubagentBrief(task="the task", context="shared excerpt"))
    assert client.requests[0]["messages"] == [
        {"role": "user", "content": "the task\n\nshared excerpt"},
    ]


def test_child_requests_grow_append_only(tmp_path):
    (tmp_path / "a.txt").write_text("aa\n", encoding="utf-8")
    script = [
        _assistant(tool_calls=[_tool_call("read_file", {"path": "a.txt"})]),
        _assistant(tool_calls=[_tool_call("grep", {"pattern": "aa"},
                                          call_id="call_2")]),
        _assistant("done"),
    ]
    parent, client, runner = _make_runner(script, tmp_path)
    runner.run(SubagentBrief(task="look around"))
    snapshots = [request["messages"] for request in client.requests]
    assert len(snapshots) == 3
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later[: len(earlier)] == earlier
        assert len(later) > len(earlier)


def test_request_options_forwarded(tmp_path):
    parent, client, runner = _make_runner(
        [_assistant("done")], tmp_path,
        max_tokens=64, temperature=0.0, top_p=1.0)
    runner.run(SubagentBrief(task="t"))
    kwargs = client.requests[0]["kwargs"]
    assert kwargs["max_tokens"] == 64
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 1.0
    names = [tool["function"]["name"] for tool in kwargs["tools"]]
    assert names == ["read_file", "grep", "edit", "bash"]


# --- session key derivation ---


def test_child_session_key_derivation():
    assert child_session_key("road-test-1", 1) == "road-test-1/sub1"
    assert child_session_key("road-test-1", 12) == "road-test-1/sub12"
    with pytest.raises(SubagentConfigError):
        child_session_key("", 1)
    with pytest.raises(SubagentConfigError):
        child_session_key("k", 0)


def test_children_run_under_derived_keys_without_collision(tmp_path):
    parent, client, runner = _make_runner(
        [_assistant("first"), _assistant("second")], tmp_path)
    first = runner.run(SubagentBrief(task="one"))
    second = runner.run(SubagentBrief(task="two"))
    assert client.requests[0]["session_cache_key"] == "parent-key/sub1"
    assert client.requests[1]["session_cache_key"] == "parent-key/sub2"
    assert first.session_cache_key == "parent-key/sub1"
    assert second.session_cache_key == "parent-key/sub2"


def test_a_failed_spawn_still_consumes_its_ordinal(tmp_path):
    parent, client, runner = _make_runner(
        [ClientError(0, "server down"), _assistant("recovered")], tmp_path)
    failed = runner.run(SubagentBrief(task="one"))
    retried = runner.run(SubagentBrief(task="two"))
    assert not failed.ok
    assert failed.session_cache_key == "parent-key/sub1"
    assert retried.session_cache_key == "parent-key/sub2"


def test_parent_without_session_key_fails_closed(tmp_path):
    parent = Conversation()
    with pytest.raises(SubagentConfigError, match="session cache key"):
        SubagentRunner(parent, FakeClient([]), build_core_registry(),
                       workdir=tmp_path)


# --- parent append-only invariant ---


def test_parent_history_is_a_strict_prefix_extension_across_a_spawn(tmp_path):
    parent, client, runner = _make_runner([_assistant("done")], tmp_path)
    before = parent.request_messages()
    runner.run(SubagentBrief(task="t"), tool_call_id="call_sub")
    after = parent.request_messages()
    assert after[: len(before)] == before
    assert len(after) == len(before) + 1
    assert after[-1]["role"] == "tool"


def test_parent_untouched_while_the_child_runs(tmp_path):
    (tmp_path / "a.txt").write_text("aa\n", encoding="utf-8")
    script = [
        _assistant(tool_calls=[_tool_call("read_file", {"path": "a.txt"})]),
        _assistant("done"),
    ]
    parent, client, runner = _make_runner(script, tmp_path)
    before = parent.request_messages()
    runner.run(SubagentBrief(task="t"))
    # Every child request went out under the child key; none extended or
    # reused the parent conversation.
    assert all(request["session_cache_key"] == "parent-key/sub1"
               for request in client.requests)
    assert all(request["messages"][: len(before)] != before
               for request in client.requests)


# --- sequential and depth enforcement ---


def test_a_second_spawn_while_one_is_open_is_refused(tmp_path):
    registry = build_core_registry()
    holder = {}

    def spawn_again(arguments, workdir):
        holder["runner"].run(SubagentBrief(task="inner"))
        return "unreachable"

    registry.register(
        ToolSpec(name="spawn", description="spawn a nested subagent",
                 parameters={"type": "object", "properties": {}}),
        spawn_again)
    parent = Conversation(session_cache_key="parent-key")
    client = FakeClient([_assistant(tool_calls=[_tool_call("spawn", {})])])
    runner = SubagentRunner(parent, client, registry, workdir=tmp_path)
    holder["runner"] = runner
    with pytest.raises(SubagentConcurrencyError):
        runner.run(SubagentBrief(task="outer"))
    # The guard releases once the failed spawn unwinds; the runner is usable.
    client.script = [_assistant("done")]
    assert runner.run(SubagentBrief(task="again")).ok


def test_depth_limit_fails_closed_at_construction(tmp_path):
    assert MAX_SUBAGENT_DEPTH == 1
    parent = Conversation(session_cache_key="parent-key")
    with pytest.raises(SubagentDepthError):
        SubagentRunner(parent, FakeClient([]), build_core_registry(),
                       workdir=tmp_path, depth=1)
    with pytest.raises(SubagentConfigError):
        SubagentRunner(parent, FakeClient([]), build_core_registry(),
                       workdir=tmp_path, depth=-1)


def test_empty_task_is_a_caller_error_and_leaves_the_parent_alone(tmp_path):
    parent, client, runner = _make_runner([], tmp_path)
    before = parent.request_messages()
    with pytest.raises(SubagentConfigError):
        runner.run(SubagentBrief(task="   "))
    assert parent.request_messages() == before
    assert client.requests == []


# --- failure surfacing ---


def test_turn_cap_surfaces_as_ok_false(tmp_path):
    (tmp_path / "a.txt").write_text("aa\n", encoding="utf-8")
    call = _tool_call("read_file", {"path": "a.txt"})
    parent, client, runner = _make_runner(
        [_assistant(tool_calls=[call]), _assistant(tool_calls=[call])],
        tmp_path, max_child_turns=2)
    result = runner.run(SubagentBrief(task="t"))
    assert not result.ok
    assert result.reason == "max_turns"
    assert result.turns == 2
    assert result.text.startswith("error: subagent (max_turns)")
    assert parent.request_messages()[-1]["content"] == result.text


def test_parse_failure_surfaces_as_ok_false(tmp_path):
    completion = ChatCompletion(
        message={"role": "assistant", "content": None,
                 "tool_calls": "not a list"},
        finish_reason="stop")
    parent, client, runner = _make_runner([completion], tmp_path)
    result = runner.run(SubagentBrief(task="t"))
    assert not result.ok
    assert result.reason == "parse_failure"
    assert parent.request_messages()[-1]["content"] == result.text


def test_client_error_surfaces_as_ok_false(tmp_path):
    parent, client, runner = _make_runner(
        [ClientError(500, "engine fault")], tmp_path)
    result = runner.run(SubagentBrief(task="t"))
    assert not result.ok
    assert result.reason == "client_error"
    assert "engine fault" in result.text
    assert parent.request_messages()[-1]["content"] == result.text


def test_malformed_assistant_message_surfaces_as_ok_false(tmp_path):
    completion = ChatCompletion(message={"role": "assistant"},
                                finish_reason="stop")
    parent, client, runner = _make_runner([completion], tmp_path)
    result = runner.run(SubagentBrief(task="t"))
    assert not result.ok
    assert result.reason == "invalid_message"
    assert parent.request_messages()[-1]["content"] == result.text


def test_a_failed_tool_call_feeds_back_and_the_child_recovers(tmp_path):
    script = [
        _assistant(tool_calls=[_tool_call("no_such_tool", {})]),
        _assistant("recovered"),
    ]
    parent, client, runner = _make_runner(script, tmp_path)
    result = runner.run(SubagentBrief(task="t"))
    assert result.ok
    fed_back = client.requests[1]["messages"][-1]
    assert fed_back["role"] == "tool"
    assert fed_back["content"].startswith("error: unknown tool")


# --- fold-back bounding and the dialect seam ---


def test_result_payload_is_bounded(tmp_path):
    parent, client, runner = _make_runner(
        [_assistant("x" * 500)], tmp_path, max_result_chars=100)
    result = runner.run(SubagentBrief(task="t"))
    assert result.ok
    assert result.truncated
    assert result.text.startswith("x" * 100)
    assert "[truncated: result exceeded 100 characters]" in result.text
    assert parent.request_messages()[-1]["content"] == result.text


def test_parse_callable_is_the_dialect_seam(tmp_path):
    # A dialect that never yields tool calls treats every assistant message
    # as the final answer, even one carrying native tool_calls.
    completion = _assistant("final text",
                            tool_calls=[_tool_call("read_file", {"path": "x"})])
    parent, client, runner = _make_runner(
        [completion], tmp_path, parse=lambda message: [])
    result = runner.run(SubagentBrief(task="t"))
    assert result.ok
    assert result.text == "final text"
    assert result.turns == 1
