"""Qwen XML text-dialect tool-call parsing.

The vendored qwen3_5 template instructs the model to emit tool invocations
as ``<tool_call><function=...><parameter=...>`` XML text. These tests pin
the strict parsing contract: raw string
values survive exactly (multiline included), schema-typed values decode as
JSON of the declared type, and every structural defect raises instead of
dropping or guessing a call.
"""

from __future__ import annotations

import pytest

from moespresso.agentlib.tools import build_core_registry
from moespresso.toolcalls.qwenxml import (
    has_qwenxml_tool_call,
    parse_qwenxml_tool_calls,
    strip_qwenxml_blocks,
)
from moespresso.toolcalls.types import ToolCallParseError

SCHEMAS = {
    name: build_core_registry().spec(name).parameters
    for name in build_core_registry().names()
}


def _call(name: str, params: str) -> str:
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def _param(key: str, value: str) -> str:
    return f"<parameter={key}>\n{value}\n</parameter>\n"


def test_no_marker_returns_empty():
    assert parse_qwenxml_tool_calls("just prose") == []
    assert parse_qwenxml_tool_calls("") == []
    assert parse_qwenxml_tool_calls(None) == []


def test_has_marker_detection():
    assert not has_qwenxml_tool_call("prose")
    assert has_qwenxml_tool_call("<tool_call>")
    assert has_qwenxml_tool_call("<function=grep>")


def test_single_call_single_parameter():
    content = _call("grep", _param("pattern", "TRACE_TAG"))
    calls = parse_qwenxml_tool_calls(content, SCHEMAS)
    assert len(calls) == 1
    assert calls[0].name == "grep"
    assert calls[0].arguments == {"pattern": "TRACE_TAG"}


def test_prose_around_blocks_is_tolerated():
    content = "Looking now.\n" + _call("grep", _param("pattern", "x")) + "\nDone."
    calls = parse_qwenxml_tool_calls(content, SCHEMAS)
    assert [c.name for c in calls] == ["grep"]


def test_multiple_blocks_in_emission_order():
    content = (
        _call("read_file", _param("path", "VERSION"))
        + "\n"
        + _call("bash", _param("command", "ls src"))
    )
    calls = parse_qwenxml_tool_calls(content, SCHEMAS)
    assert [c.name for c in calls] == ["read_file", "bash"]
    assert calls[1].arguments == {"command": "ls src"}


def test_multiline_string_value_survives_exactly():
    old = "    return total / (len(values) - 1)"
    new = "    return total / len(values)"
    content = _call(
        "edit",
        _param("path", "src/metrics.py")
        + _param("old_string", old)
        + _param("new_string", new),
    )
    calls = parse_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].arguments["old_string"] == old
    assert calls[0].arguments["new_string"] == new


def test_inner_newlines_kept_only_wrapping_newlines_trimmed():
    value = "line one\n\nline three"
    content = _call("bash", _param("command", value))
    calls = parse_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].arguments["command"] == value


def test_schema_typed_values_decode():
    content = _call(
        "read_file",
        _param("path", "data/segment_01.txt")
        + _param("offset", "10")
        + _param("limit", "40"),
    )
    calls = parse_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].arguments == {
        "path": "data/segment_01.txt", "offset": 10, "limit": 40,
    }


def test_boolean_and_number_types_decode():
    content = _call(
        "edit",
        _param("path", "a.txt")
        + _param("old_string", "x")
        + _param("new_string", "y")
        + _param("replace_all", "true"),
    ) + _call("bash", _param("command", "ls") + _param("timeout", "30.5"))
    calls = parse_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].arguments["replace_all"] is True
    assert calls[1].arguments["timeout"] == 30.5


def test_string_parameter_keeps_numeric_looking_text():
    content = _call("read_file", _param("path", "123"))
    calls = parse_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].arguments["path"] == "123"


def test_without_schemas_values_stay_raw_text():
    content = _call("read_file", _param("path", "a") + _param("offset", "10"))
    calls = parse_qwenxml_tool_calls(content)
    assert calls[0].arguments == {"path": "a", "offset": "10"}


def test_undecodable_typed_value_raises():
    content = _call("read_file", _param("path", "a") + _param("offset", "seven"))
    with pytest.raises(ToolCallParseError, match="offset"):
        parse_qwenxml_tool_calls(content, SCHEMAS)


def test_mistyped_typed_value_raises():
    content = _call("read_file", _param("path", "a") + _param("offset", "true"))
    with pytest.raises(ToolCallParseError, match="offset"):
        parse_qwenxml_tool_calls(content, SCHEMAS)


def test_unclosed_block_raises():
    content = "<tool_call>\n<function=grep>\n" + _param("pattern", "x")
    with pytest.raises(ToolCallParseError, match="unclosed"):
        parse_qwenxml_tool_calls(content, SCHEMAS)


def test_function_outside_block_raises():
    content = "<function=grep>\n" + _param("pattern", "x") + "</function>"
    with pytest.raises(ToolCallParseError, match="outside"):
        parse_qwenxml_tool_calls(content, SCHEMAS)


def test_two_functions_in_one_block_raise():
    content = (
        "<tool_call>\n<function=grep>\n" + _param("pattern", "x")
        + "</function>\n<function=bash>\n" + _param("command", "ls")
        + "</function>\n</tool_call>"
    )
    with pytest.raises(ToolCallParseError, match="exactly one"):
        parse_qwenxml_tool_calls(content, SCHEMAS)


def test_block_without_function_raises():
    with pytest.raises(ToolCallParseError, match="no well-formed function"):
        parse_qwenxml_tool_calls("<tool_call>\nnothing here\n</tool_call>")


def test_duplicate_parameter_raises():
    content = _call("grep", _param("pattern", "x") + _param("pattern", "y"))
    with pytest.raises(ToolCallParseError, match="duplicate"):
        parse_qwenxml_tool_calls(content, SCHEMAS)


def test_stray_text_inside_function_raises():
    content = _call("grep", _param("pattern", "x") + "stray words\n")
    with pytest.raises(ToolCallParseError, match="unparsed text"):
        parse_qwenxml_tool_calls(content, SCHEMAS)


def test_empty_function_name_raises():
    content = "<tool_call>\n<function=>\n</function>\n</tool_call>"
    with pytest.raises(ToolCallParseError, match="empty tool name"):
        parse_qwenxml_tool_calls(content)


def test_strip_blocks_leaves_surrounding_text():
    content = "before\n" + _call("grep", _param("pattern", "x")) + "\nafter"
    stripped = strip_qwenxml_blocks(content)
    assert "before" in stripped and "after" in stripped
    assert "<tool_call>" not in stripped


def test_template_replay_shape_round_trips():
    # The vendored template replays a stored tool call as
    # <parameter=k>\nVALUE\n</parameter> lines inside one function element;
    # parsing that exact shape recovers the original arguments.
    content = (
        "<tool_call>\n<function=edit>\n"
        "<parameter=path>\nsrc/metrics.py\n</parameter>\n"
        "<parameter=old_string>\nreturn total / (len(values) - 1)\n</parameter>\n"
        "<parameter=new_string>\nreturn total / len(values)\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = parse_qwenxml_tool_calls(content, SCHEMAS)
    assert calls[0].name == "edit"
    assert calls[0].arguments["path"] == "src/metrics.py"
