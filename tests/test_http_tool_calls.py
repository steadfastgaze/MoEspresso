"""Served tool calls: request tools in, structured tool_calls out.

The serve layer teaches a dialect through the render (the vendored
template's native XML, or the DSML block under the dsml dialect), parses
the emission back into OpenAI ``tool_calls``, and reports
``finish_reason: "tool_calls"``. These tests pin the whole round trip
against fake generators: response shaping, streaming deltas, the turn-2
request shapes a tool-calling client sends back, the render-side dialect
swap, the truncation guard, the kill switches, and the byte-stability of
replayed tool turns that prefix reuse depends on.
"""

from __future__ import annotations

import json

import pytest

# http is addressed as a module: a test elsewhere in the suite reloads
# moespresso.runtime.http, and names imported from it at collection time
# would go stale (old classes, new instances). Attribute access always
# resolves the live module contents.
import moespresso.runtime.http as http
from moespresso.package.agentic_profile import read_agentic_profile
from moespresso.runtime.deepseek_v4.renderer import DEEPSEEK_V4_PROMPT_RENDERER
from moespresso.runtime.generation import GenerationResult
from moespresso.toolcalls.dsml import (
    DSML_TOKEN,
    render_dsml_tool_calls,
    render_tools,
)

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["filePath"],
        },
    },
}

QWEN_EMISSION = (
    "checking the docs\n</think>\n"
    "<tool_call>\n<function=read>\n<parameter=filePath>\n"
    "/proj/README.md\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=read>\n<parameter=filePath>\n"
    "/proj/DEVGUIDE.md\n</parameter>\n<parameter=limit>\n5\n</parameter>\n"
    "</function>\n</tool_call>"
)

DSML_EMISSION = (
    f"<{DSML_TOKEN}tool_calls>\n"
    f'<{DSML_TOKEN}invoke name="read">\n'
    f'<{DSML_TOKEN}parameter name="filePath" string="true">/proj/README.md'
    f"</{DSML_TOKEN}parameter>\n"
    f'<{DSML_TOKEN}parameter name="limit" string="false">5</{DSML_TOKEN}parameter>\n'
    f"</{DSML_TOKEN}invoke>\n"
    f"</{DSML_TOKEN}tool_calls>"
)


def _generate_returning(text, finish_reason="stop", **fields):
    def generate(prompt, **opts):
        return GenerationResult(text=text, finish_reason=finish_reason, **fields)
    return generate


def _tool_request(**overrides):
    request = {
        "messages": [{"role": "user", "content": "what is this about"}],
        "tools": [READ_TOOL],
    }
    request.update(overrides)
    return request


class _CapturingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return "RENDERED"


# --- response shaping ---------------------------------------------------


def test_qwen_emission_becomes_structured_tool_calls():
    response = http.chat_completion(
        _tool_request(), _generate_returning(QWEN_EMISSION))
    choice = response["choices"][0]
    message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["reasoning_content"] == "checking the docs\n"
    calls = message["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["read", "read"]
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "filePath": "/proj/README.md"}
    assert json.loads(calls[1]["function"]["arguments"]) == {
        "filePath": "/proj/DEVGUIDE.md", "limit": 5}


def test_call_ids_are_deterministic_and_distinct():
    first = http.chat_completion(_tool_request(), _generate_returning(QWEN_EMISSION))
    second = http.chat_completion(_tool_request(), _generate_returning(QWEN_EMISSION))
    ids = [c["id"] for c in first["choices"][0]["message"]["tool_calls"]]
    assert len(set(ids)) == 2
    assert all(i.startswith("call_") for i in ids)
    assert ids == [c["id"] for c in second["choices"][0]["message"]["tool_calls"]]


def test_prose_around_calls_stays_content():
    text = "done thinking\n</think>\nLet me look.\n" + QWEN_EMISSION.split(
        "</think>\n", 1)[1]
    response = http.chat_completion(_tool_request(), _generate_returning(text))
    message = response["choices"][0]["message"]
    assert message["content"] == "\nLet me look.\n"
    assert len(message["tool_calls"]) == 2


def test_tool_free_request_is_untouched_by_the_tool_path():
    response = http.chat_completion(
        {"messages": [{"role": "user", "content": "hi"}]},
        _generate_returning("plan</think>final"))
    message = response["choices"][0]["message"]
    assert message == {
        "role": "assistant",
        "reasoning_content": "plan",
        "content": "final",
    }
    assert response["choices"][0]["finish_reason"] == "stop"


def test_plain_answer_with_tools_keeps_stop_finish_reason():
    response = http.chat_completion(
        _tool_request(), _generate_returning("plan</think>No tools needed."))
    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "No tools needed."
    assert "tool_calls" not in choice["message"]


def test_truncated_dangling_block_keeps_length_and_content():
    text = "t</think>\n<tool_call>\n<function=read>\n<parameter=filePath>\n/pro"
    response = http.chat_completion(
        _tool_request(), _generate_returning(text, finish_reason="length"))
    choice = response["choices"][0]
    assert choice["finish_reason"] == "length"
    assert "tool_calls" not in choice["message"]
    assert "<tool_call>" in choice["message"]["content"]


def test_repair_salvages_and_reports_telemetry_in_usage():
    text = (
        "t</think>\n<tool_call>\n<function=read>\n<parameter=filePath>\n"
        "/proj/README.md\n</parameter>\n</function=read>\n</tool_call>"
    )
    response = http.chat_completion(
        _tool_request(),
        _generate_returning(text, prompt_tokens=7, completion_tokens=9))
    message = response["choices"][0]["message"]
    assert [c["function"]["name"] for c in message["tool_calls"]] == ["read"]
    repair = response["usage"]["moespresso"]["tool_call_repair"]
    assert repair == {"fires": 1, "salvaged": 1, "failed": 0}


def test_repair_off_returns_malformed_markup_as_content():
    text = (
        "t</think>\n<tool_call>\n<function=read>\n<parameter=filePath>\n"
        "/proj/README.md\n</parameter>\n</function=read>\n</tool_call>"
    )
    response = http.chat_completion(
        _tool_request(), _generate_returning(text),
        tool_config=http.ToolCallConfig(repair=False))
    message = response["choices"][0]["message"]
    assert "tool_calls" not in message
    assert "</function=read>" in message["content"]


def test_parse_kill_switch_restores_verbatim_serving():
    response = http.chat_completion(
        _tool_request(), _generate_returning(QWEN_EMISSION),
        tool_config=http.ToolCallConfig(parse=False))
    message = response["choices"][0]["message"]
    assert "tool_calls" not in message
    assert "<tool_call>" in message["content"]
    assert response["choices"][0]["finish_reason"] == "stop"


# --- streaming ----------------------------------------------------------


def test_streaming_emits_call_deltas_and_suppresses_markup():
    chunks = [QWEN_EMISSION[i:i + 7] for i in range(0, len(QWEN_EMISSION), 7)]

    def generate(prompt, **opts):
        for step, text in enumerate(chunks, start=1):
            response = type("Response", (), {"text": text})()
            opts["response_callback"](step, response)
        return GenerationResult(text=QWEN_EMISSION, finish_reason="stop")

    deltas: list[tuple[str, str]] = []
    call_deltas: list[tuple[int, dict]] = []
    response = http.chat_completion(
        _tool_request(), generate,
        delta_callback=lambda kind, text: deltas.append((kind, text)),
        tool_delta_callback=lambda index, entry: call_deltas.append(
            (index, entry)),
    )
    reasoning = "".join(t for k, t in deltas if k == "reasoning")
    content = "".join(t for k, t in deltas if k == "content")
    assert reasoning == "checking the docs\n"
    assert "<tool_call>" not in content
    assert content.strip() == ""
    assert [index for index, _ in call_deltas] == [0, 1]
    streamed = [entry for _, entry in call_deltas]
    assert streamed == response["choices"][0]["message"]["tool_calls"]
    assert response["choices"][0]["finish_reason"] == "tool_calls"


# --- request validation and turn 2 --------------------------------------


def test_assistant_tool_calls_message_needs_no_content():
    request = {
        "messages": [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": "read",
                             "arguments": '{"filePath": "/proj/README.md"}'},
            }]},
            {"role": "tool", "content": "the file text",
             "tool_call_id": "call_0"},
        ],
        "tools": [READ_TOOL],
    }
    response = http.chat_completion(
        request, _generate_returning("t</think>All done."))
    assert response["choices"][0]["message"]["content"] == "All done."


def test_message_without_content_or_tool_calls_still_400s():
    with pytest.raises(http.RequestError, match="content"):
        http.chat_completion(
            {"messages": [{"role": "user"}]}, _generate_returning("x"))


def test_tools_entry_without_function_name_400s():
    with pytest.raises(http.RequestError, match="tools\\[0\\]"):
        http.chat_completion(
            _tool_request(tools=[{"type": "function", "function": {}}]),
            _generate_returning("x"))


def test_tool_choice_none_withholds_tools_from_the_template():
    tok = _CapturingTokenizer()
    response = http.chat_completion(
        _tool_request(tool_choice="none"),
        _generate_returning("t</think>" + QWEN_EMISSION.split("</think>\n")[1]),
        tokenizer=tok)
    assert "tools" not in tok.calls[0]["kwargs"]
    assert "tool_calls" not in response["choices"][0]["message"]


def test_forced_tool_choice_refuses():
    with pytest.raises(http.RequestError, match="tool_choice"):
        http.chat_completion(
            _tool_request(tool_choice="required"), _generate_returning("x"))


def test_native_history_arguments_decode_for_the_template():
    tok = _CapturingTokenizer()
    request = {
        "messages": [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": "read",
                             "arguments": '{"filePath": "/proj/README.md"}'},
            }]},
            {"role": "tool", "content": "text", "tool_call_id": "call_0"},
        ],
        "tools": [READ_TOOL],
    }
    http.chat_completion(request, _generate_returning("t</think>done"), tokenizer=tok)
    rendered_assistant = tok.calls[0]["messages"][1]
    assert rendered_assistant["content"] == ""
    arguments = rendered_assistant["tool_calls"][0]["function"]["arguments"]
    assert arguments == {"filePath": "/proj/README.md"}
    # The request's own message objects stay untouched.
    assert request["messages"][1]["tool_calls"][0]["function"]["arguments"] == (
        '{"filePath": "/proj/README.md"}')


# --- the dsml dialect swap ----------------------------------------------


def test_dsml_swap_teaches_dsml_and_withholds_template_tools():
    tok = _CapturingTokenizer()
    http.chat_completion(
        _tool_request(), _generate_returning("t</think>" + DSML_EMISSION),
        tokenizer=tok, tool_config=http.ToolCallConfig(dialect="dsml"))
    call = tok.calls[0]
    assert "tools" not in call["kwargs"]
    system = call["messages"][0]
    assert system["role"] == "system"
    assert DSML_TOKEN in system["content"]
    assert '"name": "read"' in system["content"]


def test_dsml_swap_parses_dsml_emission_with_typed_values():
    response = http.chat_completion(
        _tool_request(), _generate_returning("t</think>\n" + DSML_EMISSION),
        tool_config=http.ToolCallConfig(dialect="dsml"))
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "filePath": "/proj/README.md", "limit": 5}


def test_dsml_swap_still_parses_native_xml_bleed():
    response = http.chat_completion(
        _tool_request(), _generate_returning(QWEN_EMISSION),
        tool_config=http.ToolCallConfig(dialect="dsml"))
    assert len(response["choices"][0]["message"]["tool_calls"]) == 2


def test_dsml_swap_serializes_history_calls_into_content():
    tok = _CapturingTokenizer()
    request = {
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": "read",
                             "arguments": '{"filePath": "/proj/README.md"}'},
            }]},
            {"role": "tool", "content": "text", "tool_call_id": "call_0"},
        ],
        "tools": [READ_TOOL],
    }
    http.chat_completion(request, _generate_returning("t</think>done"),
                    tokenizer=tok, tool_config=http.ToolCallConfig(dialect="dsml"))
    rendered = tok.calls[0]["messages"]
    assistant = rendered[2]
    assert "tool_calls" not in assistant
    assert f'<{DSML_TOKEN}invoke name="read">' in assistant["content"]
    assert rendered[3]["role"] == "tool"


def test_dsml_history_serialization_round_trips_through_the_parser():
    from moespresso.toolcalls.dsml import parse_dsml_tool_calls

    calls = [{
        "id": "call_0", "type": "function",
        "function": {"name": "read",
                     "arguments": '{"filePath": "/proj/README.md", "limit": 5}'},
    }]
    block = render_dsml_tool_calls(calls)
    parsed = parse_dsml_tool_calls(block)
    assert render_dsml_tool_calls([{
        "type": "function",
        "function": {"name": parsed[0].name, "arguments": parsed[0].arguments},
    }]) == block


def test_dsml_swap_without_system_message_inserts_one():
    messages = [{"role": "user", "content": "read it"}]
    prepared, template_tools = http.prepare_tool_messages(
        messages, [READ_TOOL], dialect="dsml")
    assert template_tools is None
    assert prepared[0]["role"] == "system"
    assert prepared[0]["content"] == render_tools([READ_TOOL["function"]])
    assert prepared[1] == messages[0]
    assert messages == [{"role": "user", "content": "read it"}]


# --- deepseek v4 --------------------------------------------------------


def test_ds4_parses_its_native_dsml_emission():
    response = http.chat_completion(
        _tool_request(), _generate_returning(DSML_EMISSION),
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER)
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    calls = choice["message"]["tool_calls"]
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "filePath": "/proj/README.md", "limit": 5}


def test_ds4_ignores_the_dialect_selection():
    response = http.chat_completion(
        _tool_request(), _generate_returning(DSML_EMISSION),
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        tool_config=http.ToolCallConfig(dialect="native"))
    assert response["choices"][0]["finish_reason"] == "tool_calls"


# --- cache identity -----------------------------------------------------


def test_rendering_identity_unchanged_without_tool_dialect():
    assert http.rendering_identity("rid", {"enable_thinking": True}) == (
        http.rendering_identity("rid", {"enable_thinking": True},
                           tool_dialect=None))


def test_rendering_identity_separates_the_dsml_swap():
    base = http.rendering_identity("rid", {"enable_thinking": True})
    swapped = http.rendering_identity(
        "rid", {"enable_thinking": True}, tool_dialect="dsml")
    assert base != swapped


def test_tool_free_render_is_byte_identical_to_pre_tool_serving():
    tok = _CapturingTokenizer()
    http.chat_completion({"messages": [{"role": "user", "content": "hi"}]},
                    _generate_returning("x"), tokenizer=tok,
                    tool_config=http.ToolCallConfig(dialect="dsml"))
    kw = tok.calls[0]["kwargs"]
    assert "tools" not in kw
    assert tok.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


# --- config resolution --------------------------------------------------


def test_resolve_config_prefers_cli_dialect(tmp_path):
    (tmp_path / "agentic_profile.json").write_text(json.dumps({
        "schema_version": 1, "family": "qwen3_5_moe", "dialect": "dsml",
    }), encoding="utf-8")
    config = http.resolve_tool_call_config(tmp_path, dialect="native")
    assert config == http.ToolCallConfig(dialect="native")


def test_resolve_config_reads_profile_dialect(tmp_path):
    (tmp_path / "agentic_profile.json").write_text(json.dumps({
        "schema_version": 1, "family": "qwen3_5_moe", "dialect": "dsml",
    }), encoding="utf-8")
    assert http.resolve_tool_call_config(tmp_path) == http.ToolCallConfig(dialect="dsml")


def test_resolve_config_defaults_native_without_profile(tmp_path):
    assert http.resolve_tool_call_config(tmp_path) == http.ToolCallConfig()


def test_resolve_config_fails_closed_on_unknown_schema(tmp_path):
    (tmp_path / "agentic_profile.json").write_text(json.dumps({
        "schema_version": 99, "dialect": "dsml",
    }), encoding="utf-8")
    assert http.resolve_tool_call_config(tmp_path) == http.ToolCallConfig()
    assert read_agentic_profile(tmp_path) is None


def test_env_kill_switches(tmp_path, monkeypatch):
    monkeypatch.setenv("MOESPRESSO_TOOL_CALLS", "0")
    assert http.resolve_tool_call_config(tmp_path).parse is False
    monkeypatch.delenv("MOESPRESSO_TOOL_CALLS")
    monkeypatch.setenv("MOESPRESSO_TOOL_REPAIR", "0")
    config = http.resolve_tool_call_config(tmp_path)
    assert config.parse is True
    assert config.repair is False
