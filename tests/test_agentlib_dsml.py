"""DSML text-dialect tool-call parsing.

The DeepSeek-V4 serve layer returns tool invocations as DSML markers inside
the completion content. These tests pin the strict parsing contract: raw
string values survive unchanged (multiline included), non-string values
decode as JSON, and every structural defect raises instead of dropping or
guessing a call. The round-trip test renders calls through the DS4
renderer's own encoder and parses them back, tying the dialect's two ends
to one grammar.
"""

from __future__ import annotations

import pytest

from moespresso.agentlib import (
    ToolCallParseError,
    has_tool_call_block,
    parse_dsml_tool_calls,
)
from moespresso.runtime.deepseek_v4.renderer import DSML_TOKEN, _render_tool_calls

T = DSML_TOKEN


def _block(inner: str) -> str:
    return f"<{T}tool_calls>\n{inner}\n</{T}tool_calls>"


def _invoke(name: str, params: str) -> str:
    return f'<{T}invoke name="{name}">\n{params}\n</{T}invoke>'


def _param(name: str, value: str, *, string: str = "true") -> str:
    return f'<{T}parameter name="{name}" string="{string}">{value}</{T}parameter>'


def test_no_block_returns_empty():
    assert parse_dsml_tool_calls("just prose") == []
    assert parse_dsml_tool_calls("") == []
    assert parse_dsml_tool_calls(None) == []


def test_has_tool_call_block():
    assert not has_tool_call_block("prose")
    assert has_tool_call_block(_block(_invoke("grep", _param("pattern", "x"))))


def test_single_call_with_string_parameter():
    content = "I will search.\n\n" + _block(_invoke("grep", _param("pattern", "TRACE_TAG")))
    calls = parse_dsml_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "grep"
    assert calls[0].arguments == {"pattern": "TRACE_TAG"}
    assert calls[0].id is None


def test_non_string_parameters_decode_as_json():
    params = "\n".join([
        _param("path", "data/segment_01.txt"),
        _param("offset", "10", string="false"),
        _param("limit", "40", string="false"),
    ])
    calls = parse_dsml_tool_calls(_block(_invoke("read_file", params)))
    assert calls[0].arguments == {
        "path": "data/segment_01.txt", "offset": 10, "limit": 40,
    }


def test_multiline_string_value_survives_exactly():
    old = "    return total / (len(values) - 1)\n"
    params = "\n".join([
        _param("path", "src/metrics.py"),
        _param("old_string", old),
        _param("new_string", "    return total / len(values)\n"),
    ])
    calls = parse_dsml_tool_calls(_block(_invoke("edit", params)))
    assert calls[0].arguments["old_string"] == old


def test_multiple_invokes_keep_order():
    content = _block(
        _invoke("grep", _param("pattern", "a"))
        + "\n"
        + _invoke("read_file", _param("path", "README.md"))
    )
    calls = parse_dsml_tool_calls(content)
    assert [c.name for c in calls] == ["grep", "read_file"]


def test_unclosed_block_raises():
    with pytest.raises(ToolCallParseError, match="unclosed"):
        parse_dsml_tool_calls(f"<{T}tool_calls>\ntruncated output")


def test_block_without_invoke_raises():
    with pytest.raises(ToolCallParseError, match="no well-formed invoke"):
        parse_dsml_tool_calls(_block("stray text"))


def test_missing_string_attribute_raises():
    bad = f'<{T}parameter name="pattern">x</{T}parameter>'
    with pytest.raises(ToolCallParseError, match="unparsed text"):
        parse_dsml_tool_calls(_block(_invoke("grep", bad)))


def test_invalid_json_for_non_string_raises():
    with pytest.raises(ToolCallParseError, match="not\\s+valid JSON"):
        parse_dsml_tool_calls(
            _block(_invoke("read_file", _param("offset", "ten", string="false"))))


def test_duplicate_parameter_raises():
    params = _param("pattern", "a") + "\n" + _param("pattern", "b")
    with pytest.raises(ToolCallParseError, match="duplicate"):
        parse_dsml_tool_calls(_block(_invoke("grep", params)))


def test_stray_text_between_invokes_raises():
    content = _block(_invoke("grep", _param("pattern", "a")) + "\nleftover junk")
    with pytest.raises(ToolCallParseError, match="unparsed text"):
        parse_dsml_tool_calls(content)


def test_renderer_round_trip():
    # Render OpenAI-format calls through the DS4 renderer's encoder, then
    # parse the emitted DSML back into calls.
    openai_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "src/metrics.py", "limit": 20}',
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": '{"command": "wc -l data/segment_01.txt"}',
            },
        },
    ]
    rendered = _render_tool_calls(openai_calls)
    calls = parse_dsml_tool_calls(rendered)
    assert [c.name for c in calls] == ["read_file", "bash"]
    assert calls[0].arguments == {"path": "src/metrics.py", "limit": 20}
    assert calls[1].arguments == {"command": "wc -l data/segment_01.txt"}
