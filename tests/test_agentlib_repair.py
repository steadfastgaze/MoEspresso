"""Tool-call repair transformations, offline.

Each test feeds the repair layer a malformed completion of a shape the
strict parsers reject and asserts the salvage produces the intended call, or
that hopeless input still raises the same typed error. The malformed shapes
mirror the classes the repair layer is built for: code fences, truncation,
calls buried in prose, minor JSON damage, wrong quoting, and mistyped
values.
"""

from __future__ import annotations

import json

import pytest

from moespresso.agentlib.dsml import parse_dsml_tool_calls
from moespresso.agentlib.envelope import parse_action_envelope
from moespresso.agentlib.qwenxml import parse_qwenxml_tool_calls
from moespresso.agentlib.repair import (
    repair_action_envelope,
    repair_dsml_tool_calls,
    repair_qwenxml_tool_calls,
)
from moespresso.agentlib.toolcalls import ToolCallParseError
from moespresso.agentlib.tools import build_core_registry
from moespresso.runtime.deepseek_v4.renderer import DSML_TOKEN

SCHEMAS = {
    name: build_core_registry().spec(name).parameters
    for name in build_core_registry().names()
}

T = DSML_TOKEN


# --- qwen xml ------------------------------------------------------------------

def test_qwenxml_fenced_truncated_block_is_salvaged():
    # A fenced, truncated call needs two transformations: drop the fence
    # lines, then close the dangling elements.
    content = "```xml\n<tool_call>\n<function=grep>\n<parameter=pattern>\nTRACE_TAG"
    with pytest.raises(ToolCallParseError):
        parse_qwenxml_tool_calls(content, SCHEMAS)
    calls = repair_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].name == "grep"
    assert calls[0].arguments == {"pattern": "TRACE_TAG"}


def test_qwenxml_naked_function_is_wrapped():
    content = (
        "<function=read_file>\n<parameter=path>\nVERSION\n</parameter>\n"
        "</function>"
    )
    with pytest.raises(ToolCallParseError):
        parse_qwenxml_tool_calls(content, SCHEMAS)
    calls = repair_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "VERSION"}


def test_qwenxml_truncated_block_is_closed():
    content = (
        "<tool_call>\n<function=read_file>\n"
        "<parameter=path>\nsrc/metrics.py"
    )
    with pytest.raises(ToolCallParseError):
        parse_qwenxml_tool_calls(content, SCHEMAS)
    calls = repair_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].name == "read_file"
    assert calls[0].arguments["path"] == "src/metrics.py"


def test_qwenxml_named_closers_are_normalized():
    content = (
        "<tool_call>\n<function=grep>\n"
        "<parameter=pattern>\nx\n</parameter=pattern>\n"
        "</function=grep>\n</tool_call>"
    )
    calls = repair_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].arguments == {"pattern": "x"}


def test_qwenxml_quoted_integer_is_coerced():
    content = (
        "<tool_call>\n<function=read_file>\n"
        "<parameter=path>\na.txt\n</parameter>\n"
        "<parameter=offset>\n'7'\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = repair_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].arguments["offset"] == 7


def test_qwenxml_python_boolean_is_coerced():
    content = (
        "<tool_call>\n<function=edit>\n"
        "<parameter=path>\na.txt\n</parameter>\n"
        "<parameter=old_string>\nx\n</parameter>\n"
        "<parameter=new_string>\ny\n</parameter>\n"
        "<parameter=replace_all>\nTrue\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = repair_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].arguments["replace_all"] is True


def test_qwenxml_hopeless_input_raises():
    with pytest.raises(ToolCallParseError, match="unrepairable"):
        repair_qwenxml_tool_calls("<tool_call>no function here", SCHEMAS)


# --- action envelope ------------------------------------------------------------

def _envelope_body() -> dict:
    return {
        "analysis": "reading",
        "plan": "read the file",
        "commands": [{"tool": "read_file", "args": {"path": "VERSION"}}],
        "task_complete": False,
    }


def test_envelope_fenced_json_is_salvaged():
    content = "```json\n" + json.dumps(_envelope_body()) + "\n```"
    with pytest.raises(ToolCallParseError):
        parse_action_envelope(content)
    env = repair_action_envelope(content)
    assert env.calls[0].name == "read_file"


def test_envelope_buried_in_prose_is_scavenged():
    content = ("Here is my next action:\n" + json.dumps(_envelope_body())
               + "\nLet me know.")
    env = repair_action_envelope(content)
    assert env.calls[0].arguments == {"path": "VERSION"}


def test_envelope_trailing_comma_is_fixed():
    content = (
        '{"analysis": "a", "plan": "p", '
        '"commands": [{"tool": "bash", "args": {"command": "ls"},}], '
        '"task_complete": false,}'
    )
    env = repair_action_envelope(content)
    assert env.calls[0].name == "bash"


def test_envelope_python_literals_are_fixed():
    content = ('{"analysis": "a", "plan": "p", "commands": [], '
               '"task_complete": True}')
    env = repair_action_envelope(content)
    assert env.task_complete is True


def test_envelope_single_quotes_are_fixed():
    content = ("{'analysis': 'a', 'plan': 'p', 'commands': [], "
               "'task_complete': true}")
    env = repair_action_envelope(content)
    assert env.task_complete is True


def test_envelope_truncated_json_is_closed():
    full = json.dumps(_envelope_body())
    truncated = full[: full.index('"task_complete"') - 2]
    env = repair_action_envelope(truncated)
    assert env.calls[0].name == "read_file"
    assert env.task_complete is False


def test_envelope_unknown_keys_are_dropped():
    body = _envelope_body()
    body["confidence"] = 0.9
    body["commands"][0]["reason"] = "look"
    env = repair_action_envelope(json.dumps(body))
    assert env.calls[0].name == "read_file"
    assert env.calls[0].arguments == {"path": "VERSION"}


def test_envelope_key_aliases_are_canonicalized():
    content = json.dumps({
        "state_analysis": "a",
        "next_steps": "p",
        "actions": [{"name": "bash", "arguments": {"command": "ls"}}],
        "is_task_complete": False,
    })
    env = repair_action_envelope(content)
    assert env.analysis == "a" and env.plan == "p"
    assert env.calls[0].name == "bash"
    assert env.calls[0].arguments == {"command": "ls"}


def test_envelope_string_task_complete_is_coerced():
    content = json.dumps({
        "analysis": "a", "plan": "p", "commands": [], "task_complete": "true",
    })
    env = repair_action_envelope(content)
    assert env.task_complete is True


def test_envelope_string_args_are_decoded():
    content = json.dumps({
        "analysis": "a", "plan": "p",
        "commands": [{"tool": "bash", "args": "{\"command\": \"ls\"}"}],
        "task_complete": False,
    })
    env = repair_action_envelope(content)
    assert env.calls[0].arguments == {"command": "ls"}


def test_envelope_hopeless_input_raises():
    with pytest.raises(ToolCallParseError, match="unrepairable"):
        repair_action_envelope("no json here at all")


# The following envelope shapes are the served-run corpus classes: a missing
# task_complete key, a command entry losing its closing brace before the
# array closes, a bare command emitted after the model's native <tool_call>
# marker, and unrelated JSON that must stay unrepairable instead of turning
# into an empty completed envelope.

def test_envelope_missing_task_complete_defaults_from_commands():
    content = json.dumps({
        "analysis": "reading", "plan": "read",
        "commands": [{"tool": "read_file", "args": {"path": "VERSION"}}],
    })
    env = repair_action_envelope(content)
    assert env.task_complete is False
    assert env.calls[0].name == "read_file"
    done = json.dumps({"analysis": "done", "plan": "", "commands": []})
    assert repair_action_envelope(done).task_complete is True


def test_envelope_dropped_entry_brace_is_rebalanced():
    content = (
        '{"analysis": "found it", "plan": "edit the line", '
        '"commands": [{"tool": "edit", "args": {"path": "src/metrics.py", '
        '"old_string": "    return total / (len(values) - 1)", '
        '"new_string": "    return total / len(values)"}], '
        '"task_complete": false}'
    )
    env = repair_action_envelope(content)
    assert env.calls[0].name == "edit"
    assert env.calls[0].arguments["path"] == "src/metrics.py"
    assert env.task_complete is False


def test_envelope_bare_command_after_marker_is_wrapped():
    content = ('<tool_call>\n'
               '{"name": "read_file", "args": {"path": "src/module_golf.py"}}\n'
               '{"task_complete": false}')
    env = repair_action_envelope(content)
    assert env.task_complete is False
    assert env.calls[0].name == "read_file"
    assert env.calls[0].arguments == {"path": "src/module_golf.py"}


def test_envelope_unrelated_json_stays_unrepairable():
    # Neither commands nor task_complete present: repair must not invent a
    # completed envelope out of arbitrary JSON.
    with pytest.raises(ToolCallParseError, match="unrepairable"):
        repair_action_envelope('{"summary": "all good", "confidence": 0.9}')


# --- dsml -----------------------------------------------------------------------

def test_dsml_missing_string_attribute_defaults_to_string():
    content = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="grep">\n'
        f'<{T}parameter name="pattern">TRACE_TAG</{T}parameter>\n'
        f"</{T}invoke>\n"
        f"</{T}tool_calls>"
    )
    with pytest.raises(ToolCallParseError):
        parse_dsml_tool_calls(content)
    calls = repair_dsml_tool_calls(content)
    assert calls[0].arguments == {"pattern": "TRACE_TAG"}


def test_dsml_truncated_block_is_closed():
    content = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="read_file">\n'
        f'<{T}parameter name="path" string="true">VERSION'
    )
    calls = repair_dsml_tool_calls(content)
    assert calls[0].name == "read_file"
    assert calls[0].arguments["path"].startswith("VERSION")


def test_dsml_naked_invoke_is_wrapped():
    content = (
        f'<{T}invoke name="bash">\n'
        f'<{T}parameter name="command" string="true">ls</{T}parameter>\n'
        f"</{T}invoke>"
    )
    calls = repair_dsml_tool_calls(content)
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"command": "ls"}


def test_dsml_hopeless_input_raises():
    with pytest.raises(ToolCallParseError, match="unrepairable"):
        repair_dsml_tool_calls(f"<{T}tool_calls>stray only")


def test_dsml_unclosed_name_quote_is_restored():
    # The served-run corpus class: the closing quote of the name attribute is
    # dropped, fusing it into the string attribute.
    content = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="read_file">\n'
        f'<{T}parameter name="path string="true">VERSION</{T}parameter>\n'
        f"</{T}invoke>\n"
        f"</{T}tool_calls>"
    )
    with pytest.raises(ToolCallParseError):
        parse_dsml_tool_calls(content)
    calls = repair_dsml_tool_calls(content)
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "VERSION"}


def test_dsml_wellformed_name_attribute_is_untouched_by_repair_regex():
    from moespresso.agentlib.repair import _DSML_UNCLOSED_NAME_RE
    good = f'<{T}parameter name="path" string="true">VERSION</{T}parameter>'
    assert _DSML_UNCLOSED_NAME_RE.sub(r'\1"\2', good) == good
