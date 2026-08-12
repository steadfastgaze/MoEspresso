"""Incremental reasoning and answer response shaping."""

from __future__ import annotations

import json

import pytest

from moespresso.runtime.chat_stream import ReasoningSplitter, split_complete_text
from moespresso.runtime.tool_stream import (
    DSML_DIALECT,
    QWENXML_DIALECT,
    ToolCallStreamer,
)
from moespresso.toolcalls.dsml import DSML_TOKEN

T = DSML_TOKEN

_TOOL_SCHEMAS = {
    "bash": {"type": "object", "properties": {"command": {"type": "string"}}},
}

DSML_BASH_BLOCK = (
    f"<{T}tool_calls>\n"
    f'<{T}invoke name="bash">\n'
    f'<{T}parameter name="command" string="true">ls</{T}parameter>\n'
    f"</{T}invoke>\n"
    f"</{T}tool_calls>"
)

DSML_MULTI_BLOCK = (
    f"<{T}tool_calls>\n"
    f'<{T}invoke name="bash">\n'
    f'<{T}parameter name="command" string="true">ls</{T}parameter>\n'
    f"</{T}invoke>\n"
    f'<{T}invoke name="bash">\n'
    f'<{T}parameter name="command" string="true">pwd</{T}parameter>\n'
    f"</{T}invoke>\n"
    f"</{T}tool_calls>"
)


def _route(pieces):
    """Drive text through the reasoning splitter into a tool streamer.

    Mirrors the serve path: only ``content`` deltas are pushed into the
    tool streamer, so anything the splitter classifies as reasoning never
    reaches tool extraction. Returns ``(splitter, streamer)``.
    """
    streamer = ToolCallStreamer(
        (DSML_DIALECT,), parameter_schemas=_TOOL_SCHEMAS)

    def route_delta(kind: str, text: str) -> None:
        if kind == "content":
            streamer.push(text)

    splitter = ReasoningSplitter(thinking_enabled=True, emit=route_delta)
    for piece in pieces:
        splitter.push(piece)
    splitter.finish()
    streamer.finish()
    return splitter, streamer


@pytest.mark.parametrize(
    "raw,thinking,reasoning,content",
    [
        ("plan</think>answer", True, "plan", "answer"),
        ("<think>plan</think>answer", True, "plan", "answer"),
        ("<think></think>answer", False, "", "answer"),
        ("plain answer", False, "", "plain answer"),
        ("plain legacy answer", True, "", "plain legacy answer"),
    ],
)
def test_split_complete_text(raw, thinking, reasoning, content):
    assert split_complete_text(raw, thinking_enabled=thinking) == (
        reasoning,
        content,
    )


def test_split_complete_text_truncated_thinking_is_reasoning_not_content():
    # Generation stopped before the close marker while the rendered prompt
    # itself opened the reasoning section: the whole string is reasoning.
    # Without the prompt-opened fact the legacy plain-text reading stands.
    truncated = "partial plan that ran out of tok"
    assert split_complete_text(
        truncated, thinking_enabled=True, prompt_opened_thinking=True,
    ) == (truncated, "")
    assert split_complete_text(
        truncated, thinking_enabled=True,
    ) == ("", truncated)
    assert split_complete_text(
        "plan</think>answer", thinking_enabled=True,
        prompt_opened_thinking=True,
    ) == ("plan", "answer")


@pytest.mark.parametrize("cut", range(1, len("<think>plan</think>answer")))
def test_reasoning_splitter_accepts_every_two_chunk_boundary(cut):
    raw = "<think>plan</think>answer"
    events = []
    splitter = ReasoningSplitter(
        thinking_enabled=True,
        emit=lambda kind, text: events.append((kind, text)),
    )
    splitter.push(raw[:cut])
    splitter.push(raw[cut:])
    splitter.finish()

    assert splitter.reasoning == "plan"
    assert splitter.content == "answer"
    assert "".join(text for kind, text in events if kind == "reasoning") == "plan"
    assert "".join(text for kind, text in events if kind == "content") == "answer"


def test_reasoning_splitter_accepts_one_character_chunks_after_prompt_open():
    events = []
    splitter = ReasoningSplitter(
        thinking_enabled=True,
        emit=lambda kind, text: events.append((kind, text)),
    )
    for char in "work\n</think>\nFinal":
        splitter.push(char)
    splitter.finish()

    assert splitter.reasoning == "work\n"
    assert splitter.content == "\nFinal"
    assert all("</think>" not in text for _kind, text in events)


def test_unterminated_reasoning_stays_in_reasoning_channel():
    splitter = ReasoningSplitter(thinking_enabled=True)
    splitter.push("still considering")
    splitter.finish()
    assert splitter.reasoning == "still considering"
    assert splitter.content == ""


# --- tool blocks emitted without closing the reasoning region ----------------
#
# A served turn that slides from reasoning straight into a tool-call block
# without emitting the close marker loses the call: the splitter stays in
# reasoning mode, and only content deltas reach tool extraction. These tests
# pin that channel routing, which is what decides whether a well-formed block
# is ever seen by the tool streamer at all.

def test_dsml_unclosed_think_region_routes_zero_calls():
    # Two arms differing only by the close marker, so a null result cannot
    # come from both arms running the same path.
    without_close, streamer = _route(["Planning.\n" + DSML_BASH_BLOCK])
    assert streamer.calls == []
    assert streamer.content == ""
    assert DSML_BASH_BLOCK in without_close.reasoning

    with_close, closed_streamer = _route(
        ["Planning.</think>\n" + DSML_BASH_BLOCK])
    assert [entry["function"]["name"] for entry in closed_streamer.calls] == [
        "bash"]
    assert closed_streamer.content == ""
    assert with_close.reasoning == "Planning."


def test_dsml_unclosed_think_close_marker_after_the_block_routes_zero_calls():
    # The close marker arriving after the block is no better than none: the
    # block itself was already classified as reasoning.
    splitter, streamer = _route(
        ["Planning.\n" + DSML_BASH_BLOCK + "\n</think>Done."])
    assert streamer.calls == []
    assert streamer.content == "Done."
    assert DSML_BASH_BLOCK in splitter.reasoning


def test_first_valid_outer_dsml_block_is_terminal_and_keeps_all_invokes():
    streamer = ToolCallStreamer(
        (DSML_DIALECT,), parameter_schemas=_TOOL_SCHEMAS)
    text = DSML_MULTI_BLOCK + "\n</think>ignored" + DSML_BASH_BLOCK

    for char in text:
        streamer.push(char)
    streamer.finish()

    assert streamer.terminal
    assert [entry["function"]["name"] for entry in streamer.calls] == [
        "bash", "bash"]
    assert [
        json.loads(entry["function"]["arguments"])["command"]
        for entry in streamer.calls
    ] == ["ls", "pwd"]
    assert streamer.content == ""


def test_repaired_naked_dsml_invoke_is_terminal_before_late_duplicate():
    naked = (
        f'<{T}invoke name="bash">\n'
        f'<{T}parameter name="command" string="true">ls</{T}parameter>\n'
        f"</{T}invoke>"
    )
    streamer = ToolCallStreamer(
        (DSML_DIALECT,), parameter_schemas=_TOOL_SCHEMAS)
    for char in naked + "\n" + DSML_BASH_BLOCK:
        streamer.push(char)

    assert len(streamer.calls) == 1
    assert streamer.terminal


def test_repaired_outer_dsml_block_is_terminal_before_late_duplicate():
    malformed_outer = DSML_BASH_BLOCK.replace(' string="true"', "")
    streamer = ToolCallStreamer(
        (DSML_DIALECT,), parameter_schemas=_TOOL_SCHEMAS)
    streamer.push(malformed_outer + "\n" + DSML_BASH_BLOCK)
    streamer.finish()

    assert streamer.terminal
    assert len(streamer.calls) == 1
    assert json.loads(streamer.calls[0]["function"]["arguments"]) == {
        "command": "ls"}
    assert streamer.telemetry.as_dict() == {
        "fires": 1, "salvaged": 1, "failed": 0}


def test_qwen_xml_blocks_remain_nonterminal():
    first = (
        "<tool_call>\n<function=bash>\n<parameter=command>\n"
        "ls\n</parameter>\n</function>\n</tool_call>"
    )
    second = first.replace("ls", "pwd")
    streamer = ToolCallStreamer(
        (QWENXML_DIALECT,), parameter_schemas=_TOOL_SCHEMAS)
    streamer.push(first + "\n" + second)
    streamer.finish()

    assert len(streamer.calls) == 2
    assert not streamer.terminal
