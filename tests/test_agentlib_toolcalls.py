"""Strict native tool-call parsing (OpenAI ``tool_calls`` in a response message).

The parser is deliberately strict: anything malformed raises
ToolCallParseError so the later repair layer has one typed seam to catch.
These tests pin both the accepted shapes and the rejections.
"""

from __future__ import annotations

import pytest

from moespresso.agentlib import ToolCall, ToolCallParseError, parse_tool_calls


def _message(tool_calls):
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def test_message_without_tool_calls_parses_empty():
    assert parse_tool_calls({"role": "assistant", "content": "hi"}) == []
    assert parse_tool_calls(_message(None)) == []


def test_arguments_as_json_string():
    calls = parse_tool_calls(_message([{
        "id": "call_1",
        "type": "function",
        "function": {"name": "grep", "arguments": '{"pattern": "def "}'},
    }]))
    assert calls == [ToolCall(name="grep", arguments={"pattern": "def "}, id="call_1")]


def test_arguments_as_decoded_object():
    calls = parse_tool_calls(_message([{
        "function": {"name": "read_file", "arguments": {"path": "a.py"}},
    }]))
    assert calls == [ToolCall(name="read_file", arguments={"path": "a.py"}, id=None)]


def test_decoded_arguments_are_copied():
    raw = {"path": "a.py", "nested": {"k": 1}}
    calls = parse_tool_calls(_message([{"function": {"name": "read_file",
                                                     "arguments": raw}}]))
    raw["nested"]["k"] = 2
    assert calls[0].arguments["nested"]["k"] == 1


def test_empty_or_missing_arguments_decode_to_empty_object():
    for arguments in (None, "", "   "):
        entry = {"function": {"name": "bash"}}
        if arguments is not None:
            entry["function"]["arguments"] = arguments
        assert parse_tool_calls(_message([entry]))[0].arguments == {}


def test_multiple_calls_keep_order_and_ids():
    calls = parse_tool_calls(_message([
        {"id": "a", "function": {"name": "grep", "arguments": "{}"}},
        {"id": "b", "function": {"name": "bash", "arguments": "{}"}},
    ]))
    assert [(c.id, c.name) for c in calls] == [("a", "grep"), ("b", "bash")]


def test_type_defaults_to_function_and_other_types_rejected():
    entry = {"function": {"name": "grep", "arguments": "{}"}}
    assert parse_tool_calls(_message([entry]))[0].name == "grep"
    with pytest.raises(ToolCallParseError, match="unsupported type"):
        parse_tool_calls(_message([{"type": "retrieval",
                                    "function": {"name": "grep"}}]))


def test_tool_calls_must_be_a_list():
    with pytest.raises(ToolCallParseError, match="must be a list"):
        parse_tool_calls(_message({"function": {"name": "grep"}}))


def test_entry_must_be_an_object():
    with pytest.raises(ToolCallParseError, match=r"tool_calls\[0\]"):
        parse_tool_calls(_message(["grep()"]))


def test_missing_function_object_rejected():
    with pytest.raises(ToolCallParseError, match="function object"):
        parse_tool_calls(_message([{"id": "x"}]))


def test_missing_or_empty_name_rejected():
    with pytest.raises(ToolCallParseError, match="function name"):
        parse_tool_calls(_message([{"function": {"arguments": "{}"}}]))
    with pytest.raises(ToolCallParseError, match="function name"):
        parse_tool_calls(_message([{"function": {"name": ""}}]))


def test_malformed_arguments_json_rejected():
    with pytest.raises(ToolCallParseError, match="not valid JSON"):
        parse_tool_calls(_message([{
            "function": {"name": "grep", "arguments": '{"pattern": "x"'},
        }]))


def test_non_object_arguments_json_rejected():
    with pytest.raises(ToolCallParseError, match="decode to an object"):
        parse_tool_calls(_message([{
            "function": {"name": "grep", "arguments": "[1, 2]"},
        }]))


def test_non_string_non_object_arguments_rejected():
    with pytest.raises(ToolCallParseError, match="JSON string or object"):
        parse_tool_calls(_message([{
            "function": {"name": "grep", "arguments": 7},
        }]))


def test_non_string_id_rejected():
    with pytest.raises(ToolCallParseError, match="id must be a string"):
        parse_tool_calls(_message([{
            "id": 5, "function": {"name": "grep", "arguments": "{}"},
        }]))


def test_error_names_the_failing_entry_index():
    with pytest.raises(ToolCallParseError, match=r"tool_calls\[1\]"):
        parse_tool_calls(_message([
            {"function": {"name": "grep", "arguments": "{}"}},
            {"function": {"arguments": "{}"}},
        ]))
