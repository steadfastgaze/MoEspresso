"""Streaming tool-call extraction from the answer channel.

ToolCallStreamer turns dialect marker blocks inside generated text into
OpenAI-format tool-call entries while surrounding text keeps flowing as
content. These tests pin the streaming contract: chunk-boundary
independence (every split of the same text yields the same calls and the
same visible content), strict-parse-first with repair only on failure,
malformed text flushing back as content, the truncation guard, and the
line-start marker rule.
"""

from __future__ import annotations

import json

import pytest

from moespresso.runtime.tool_stream import (
    DSML_DIALECT,
    QWENXML_DIALECT,
    ToolCallStreamer,
)
from moespresso.toolcalls.dsml import DSML_TOKEN

T = DSML_TOKEN

SCHEMAS = {
    "read": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "limit": {"type": "integer"},
        },
    },
}

QWEN_TWO_CALLS = (
    "<tool_call>\n"
    "<function=read>\n"
    "<parameter=filePath>\n"
    "/proj/README.md\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    "<tool_call>\n"
    "<function=read>\n"
    "<parameter=filePath>\n"
    "/proj/DEVGUIDE.md\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

DSML_ONE_CALL = (
    f"<{T}tool_calls>\n"
    f'<{T}invoke name="read">\n'
    f'<{T}parameter name="filePath" string="true">/proj/README.md'
    f"</{T}parameter>\n"
    f'<{T}parameter name="limit" string="false">5</{T}parameter>\n'
    f"</{T}invoke>\n"
    f"</{T}tool_calls>"
)


def _run(text_or_pieces, dialects=(QWENXML_DIALECT,), **kwargs):
    pieces = (
        [text_or_pieces] if isinstance(text_or_pieces, str) else text_or_pieces
    )
    content_deltas: list[str] = []
    call_events: list[tuple[int, dict]] = []
    streamer = ToolCallStreamer(
        dialects,
        parameter_schemas=SCHEMAS,
        emit_content=content_deltas.append,
        emit_tool_call=lambda index, entry: call_events.append((index, entry)),
        **kwargs,
    )
    for piece in pieces:
        streamer.push(piece)
    streamer.finish()
    return streamer, content_deltas, call_events


def _names_and_arguments(streamer):
    return [
        (entry["function"]["name"], json.loads(entry["function"]["arguments"]))
        for entry in streamer.calls
    ]


def test_two_sequential_blocks_become_indexed_calls():
    streamer, content_deltas, call_events = _run(QWEN_TWO_CALLS)
    assert _names_and_arguments(streamer) == [
        ("read", {"filePath": "/proj/README.md"}),
        ("read", {"filePath": "/proj/DEVGUIDE.md"}),
    ]
    assert [index for index, _ in call_events] == [0, 1]
    assert streamer.content == ""
    assert content_deltas == []
    assert streamer.telemetry.fires == 0


def test_every_chunk_split_yields_identical_result():
    text = "Preamble line.\n" + QWEN_TWO_CALLS + "\nDone."
    reference, _, _ = _run(text)
    for cut in range(len(text) + 1):
        streamer, _, _ = _run([text[:cut], text[cut:]])
        assert streamer.calls == reference.calls, f"cut={cut}"
        assert streamer.content == reference.content, f"cut={cut}"
    char_by_char, _, _ = _run(list(text))
    assert char_by_char.calls == reference.calls
    assert char_by_char.content == reference.content


def test_text_around_blocks_stays_content_and_furniture_is_dropped():
    # Trailing whitespace of the content stream is furniture on a tool-call
    # turn: the newline between the prose and the first block goes with it.
    text = "Looking now.\n" + QWEN_TWO_CALLS + "\n"
    streamer, content_deltas, _ = _run(text)
    assert streamer.content == "Looking now."
    assert "".join(content_deltas) == "Looking now."
    assert len(streamer.calls) == 2


def test_trailing_text_after_blocks_is_preserved():
    # Every byte outside the blocks survives: the separator newline between
    # the blocks and the newline before the prose both belong to the text
    # channel, so the content is exactly the emission minus the blocks.
    streamer, _, _ = _run(QWEN_TWO_CALLS + "\nThat covers both files.")
    assert streamer.content == "\n\nThat covers both files."
    assert len(streamer.calls) == 2


def test_typed_parameter_decodes_against_schema():
    text = (
        "<tool_call>\n<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>\n"
        "<parameter=limit>\n5\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    streamer, _, _ = _run(text)
    assert _names_and_arguments(streamer) == [
        ("read", {"filePath": "README.md", "limit": 5}),
    ]


def test_dsml_block_parses_with_string_flag_typing():
    streamer, _, _ = _run(DSML_ONE_CALL, dialects=(DSML_DIALECT,))
    assert _names_and_arguments(streamer) == [
        ("read", {"filePath": "/proj/README.md", "limit": 5}),
    ]


def test_dsml_primary_still_catches_native_qwenxml_bleed():
    streamer, _, _ = _run(
        QWEN_TWO_CALLS, dialects=(DSML_DIALECT, QWENXML_DIALECT))
    assert len(streamer.calls) == 2
    assert streamer.content == ""


def test_marker_mid_sentence_stays_prose():
    text = "The dialect wraps calls in <tool_call> tags on their own line."
    streamer, content_deltas, _ = _run(text)
    assert streamer.calls == []
    assert streamer.content == text
    assert "".join(content_deltas) == text


def test_glued_blocks_without_separator_both_parse():
    # The character after a close marker is a block boundary, so a second
    # block glued directly to the first still parses.
    glued = QWEN_TWO_CALLS.replace("</tool_call>\n<tool_call>",
                                   "</tool_call><tool_call>")
    streamer, _, _ = _run(glued)
    assert [name for name, _ in _names_and_arguments(streamer)] == [
        "read", "read"]
    assert streamer.content == ""


def test_unterminated_block_split_fuzz_matches_one_shot():
    # The resumable close-marker scan must not change behavior at any chunk
    # boundary, including when the block never closes and finish repairs it.
    text = (
        "lead-in\n<tool_call>\n<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>"
    )
    reference, _, _ = _run(text)
    for cut in range(len(text) + 1):
        streamer, _, _ = _run([text[:cut], text[cut:]])
        assert streamer.calls == reference.calls, f"cut={cut}"
        assert streamer.content == reference.content, f"cut={cut}"


def test_malformed_block_is_repaired_and_counted():
    text = (
        "<tool_call>\n<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>\n"
        "</function=read>\n</tool_call>"
    )
    streamer, content_deltas, _ = _run(text)
    assert _names_and_arguments(streamer) == [("read", {"filePath": "README.md"})]
    assert streamer.telemetry.as_dict() == {"fires": 1, "salvaged": 1, "failed": 0}
    assert content_deltas == []


def test_hopeless_block_flushes_back_as_content():
    text = "<tool_call>\nnothing resembling a function element\n</tool_call>"
    streamer, content_deltas, _ = _run(text)
    assert streamer.calls == []
    assert streamer.content == text
    assert "".join(content_deltas) == text
    assert streamer.telemetry.failed == 1


def test_repair_disabled_flushes_malformed_block_without_counting():
    text = (
        "<tool_call>\n<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>\n"
        "</function=read>\n</tool_call>"
    )
    streamer, _, _ = _run(text, repair_enabled=False)
    assert streamer.calls == []
    assert streamer.content == text
    assert streamer.telemetry.fires == 0


def test_unterminated_block_is_repaired_at_finish():
    text = (
        "<tool_call>\n<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>"
    )
    streamer, _, _ = _run(text)
    assert _names_and_arguments(streamer) == [("read", {"filePath": "README.md"})]
    assert streamer.telemetry.salvaged == 1


def test_truncated_turn_never_repairs_the_dangling_block():
    text = (
        "<tool_call>\n<function=read>\n"
        "<parameter=filePath>\n/proj/pro"
    )
    content_deltas: list[str] = []
    streamer = ToolCallStreamer(
        (QWENXML_DIALECT,),
        parameter_schemas=SCHEMAS,
        emit_content=content_deltas.append,
    )
    streamer.push(text)
    streamer.finish(truncated=True)
    assert streamer.calls == []
    assert streamer.content == text
    assert "".join(content_deltas) == text
    assert streamer.telemetry.fires == 0


def test_naked_function_is_salvaged_at_end_of_turn():
    text = (
        "<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>\n"
        "</function>"
    )
    streamer, content_deltas, _ = _run(text)
    assert _names_and_arguments(streamer) == [("read", {"filePath": "README.md"})]
    # The raw text already streamed as deltas; the shaped message drops it.
    assert streamer.content == ""
    assert "".join(content_deltas) == text
    assert streamer.telemetry.salvaged == 1


def test_prose_before_naked_function_survives_salvage():
    text = (
        "Reading the file.\n<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>\n</function>"
    )
    streamer, _, _ = _run(text)
    assert len(streamer.calls) == 1
    assert streamer.content == "Reading the file.\n"


def test_prose_after_naked_function_survives_salvage():
    # Salvage strips the attempt region, not everything after it, so the
    # non-streaming content keeps the trailing prose streamed clients saw.
    text = (
        "Reading the file.\n<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>\n</function>\n"
        "Let me know if you need more."
    )
    streamer, _, _ = _run(text)
    assert len(streamer.calls) == 1
    assert "Reading the file." in streamer.content
    assert "Let me know if you need more." in streamer.content
    assert "<function=" not in streamer.content


def test_quoted_mid_sentence_attempt_never_salvages():
    # A function element quoted inside prose (not at a line start) is
    # documentation, not an attempt; salvage must not turn it into a call.
    text = (
        "The format is <function=read>\n<parameter=filePath>\nX\n"
        "</parameter>\n</function> on its own lines."
    )
    streamer, _, _ = _run(text)
    assert streamer.calls == []
    assert streamer.content == text


def test_prose_only_turn_passes_through_untouched():
    text = "First paragraph.\n\nSecond paragraph with trailing space. \n"
    streamer, content_deltas, _ = _run(text)
    assert streamer.calls == []
    assert streamer.content == text
    assert "".join(content_deltas) == text


def test_call_entries_are_openai_shaped_with_stable_ids():
    streamer, _, _ = _run(
        QWEN_TWO_CALLS, make_call_id=lambda index: f"call_test_{index}")
    for index, entry in enumerate(streamer.calls):
        assert entry["id"] == f"call_test_{index}"
        assert entry["type"] == "function"
        assert isinstance(entry["function"]["arguments"], str)
        json.loads(entry["function"]["arguments"])


def test_push_after_finish_refuses():
    streamer, _, _ = _run("hello")
    with pytest.raises(RuntimeError):
        streamer.push("more")


def test_finish_is_idempotent():
    streamer, _, _ = _run(QWEN_TWO_CALLS)
    calls = list(streamer.calls)
    streamer.finish()
    assert streamer.calls == calls
