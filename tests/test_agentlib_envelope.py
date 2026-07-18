"""Action-envelope dialect parsing.

The envelope dialect asks for a single JSON object per turn with exactly the
keys analysis, plan, commands, task_complete. These tests pin the strict
contract: well-formed envelopes come back with calls in order and JSON-typed
arguments, and any prose, fencing, extra keys, or mistyped fields raise so
the repair seam catches them.
"""

from __future__ import annotations

import json

import pytest

from moespresso.agentlib.envelope import (
    ActionEnvelope,
    envelope_system_block,
    parse_action_envelope,
)
from moespresso.agentlib.toolcalls import ToolCallParseError
from moespresso.agentlib.tools import build_core_registry


def _envelope(**overrides) -> str:
    body = {
        "analysis": "the file has not been read yet",
        "plan": "read it",
        "commands": [{"tool": "read_file", "args": {"path": "VERSION"}}],
        "task_complete": False,
    }
    body.update(overrides)
    return json.dumps(body)


def test_valid_envelope_parses():
    env = parse_action_envelope(_envelope())
    assert isinstance(env, ActionEnvelope)
    assert env.analysis == "the file has not been read yet"
    assert env.plan == "read it"
    assert env.task_complete is False
    assert len(env.calls) == 1
    assert env.calls[0].name == "read_file"
    assert env.calls[0].arguments == {"path": "VERSION"}


def test_commands_keep_order_and_json_types():
    content = _envelope(commands=[
        {"tool": "read_file", "args": {"path": "a", "offset": 3, "limit": 2}},
        {"tool": "bash", "args": {"command": "ls", "timeout": 30.5}},
    ])
    env = parse_action_envelope(content)
    assert [c.name for c in env.calls] == ["read_file", "bash"]
    assert env.calls[0].arguments["offset"] == 3
    assert env.calls[1].arguments["timeout"] == 30.5


def test_final_envelope_with_no_commands():
    env = parse_action_envelope(_envelope(commands=[], task_complete=True))
    assert env.task_complete is True
    assert env.calls == ()


def test_analysis_and_plan_default_to_empty():
    content = json.dumps({"commands": [], "task_complete": True})
    env = parse_action_envelope(content)
    assert env.analysis == "" and env.plan == ""


def test_args_default_to_empty_object():
    content = _envelope(commands=[{"tool": "bash"}])
    env = parse_action_envelope(content)
    assert env.calls[0].arguments == {}


def test_surrounding_whitespace_is_tolerated():
    env = parse_action_envelope("\n  " + _envelope() + "\n")
    assert env.calls[0].name == "read_file"


def test_prose_reply_raises():
    with pytest.raises(ToolCallParseError, match="not a JSON action envelope"):
        parse_action_envelope("I will read the file now.")


def test_empty_reply_raises():
    with pytest.raises(ToolCallParseError, match="empty reply"):
        parse_action_envelope("")
    with pytest.raises(ToolCallParseError, match="empty reply"):
        parse_action_envelope(None)


def test_fenced_json_raises_strictly():
    with pytest.raises(ToolCallParseError):
        parse_action_envelope("```json\n" + _envelope() + "\n```")


def test_trailing_prose_raises_strictly():
    with pytest.raises(ToolCallParseError):
        parse_action_envelope(_envelope() + "\nDone.")


def test_top_level_array_raises():
    with pytest.raises(ToolCallParseError, match="must be a JSON object"):
        parse_action_envelope("[1, 2]")


def test_unknown_top_level_key_raises():
    content = json.dumps({
        "analysis": "", "plan": "", "commands": [], "task_complete": True,
        "confidence": 0.9,
    })
    with pytest.raises(ToolCallParseError, match="confidence"):
        parse_action_envelope(content)


def test_missing_commands_raises():
    content = json.dumps({"analysis": "", "plan": "", "task_complete": False})
    with pytest.raises(ToolCallParseError, match="commands"):
        parse_action_envelope(content)


def test_missing_or_mistyped_task_complete_raises():
    content = json.dumps({"analysis": "", "plan": "", "commands": []})
    with pytest.raises(ToolCallParseError, match="task_complete"):
        parse_action_envelope(content)
    with pytest.raises(ToolCallParseError, match="task_complete"):
        parse_action_envelope(_envelope(task_complete="true"))


def test_mistyped_text_channel_raises():
    with pytest.raises(ToolCallParseError, match="analysis"):
        parse_action_envelope(_envelope(analysis=["a"]))


def test_unknown_command_key_raises():
    content = _envelope(commands=[
        {"tool": "bash", "args": {"command": "ls"}, "reason": "look around"},
    ])
    with pytest.raises(ToolCallParseError, match="reason"):
        parse_action_envelope(content)


def test_command_without_tool_raises():
    content = _envelope(commands=[{"args": {"command": "ls"}}])
    with pytest.raises(ToolCallParseError, match="tool name"):
        parse_action_envelope(content)


def test_command_args_must_be_object():
    content = _envelope(commands=[{"tool": "bash", "args": "ls"}])
    with pytest.raises(ToolCallParseError, match="args"):
        parse_action_envelope(content)


def test_system_block_embeds_schemas_and_keys():
    block = envelope_system_block(build_core_registry().openai_tools())
    for name in ("read_file", "grep", "edit", "bash"):
        assert f'"name": "{name}"' in block
    for key in ("analysis", "plan", "commands", "task_complete"):
        assert key in block
    assert "single JSON object" in block
