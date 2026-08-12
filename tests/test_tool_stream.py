"""Streaming tool-call extraction from the answer channel.

ToolCallStreamer turns dialect marker blocks inside generated text into
OpenAI-format tool-call entries while surrounding text keeps flowing as
content. These tests pin the streaming contract: chunk-boundary
independence (every split of the same text yields the same calls and the
same visible content), strict-parse-first with repair only on failure,
malformed text flushing back as content, the truncation guard, and the
line-start marker rule.

A block that never leaves the reasoning channel never reaches this streamer,
so those arms live in tests/test_chat_stream.py beside the splitter that
decides the routing.
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
    "write": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "content": {"type": "string"},
        },
    },
    "bash": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
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


def test_naked_function_buffers_in_stream_and_repairs():
    # A line-start function element with no wrapper is a call attempt: it
    # buffers through its own element markers, so the raw markup never
    # reaches the streamed content even though repair is what parses it.
    text = (
        "<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>\n"
        "</function>"
    )
    streamer, content_deltas, _ = _run(text)
    assert _names_and_arguments(streamer) == [("read", {"filePath": "README.md"})]
    assert streamer.content == ""
    assert content_deltas == []
    assert streamer.telemetry.as_dict() == {"fires": 1, "salvaged": 1, "failed": 0}


def test_naked_dsml_invoke_buffers_in_stream_and_repairs():
    text = (
        f'<{T}invoke name="read">\n'
        f'<{T}parameter name="filePath" string="true">README.md'
        f"</{T}parameter>\n"
        f"</{T}invoke>"
    )
    streamer, content_deltas, _ = _run(text, dialects=(DSML_DIALECT,))
    assert _names_and_arguments(streamer) == [("read", {"filePath": "README.md"})]
    assert content_deltas == []
    assert streamer.telemetry.salvaged == 1


def test_prose_around_naked_function_survives():
    text = (
        "Reading the file.\n<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>\n</function>\n"
        "Let me know if you need more."
    )
    streamer, content_deltas, _ = _run(text)
    assert len(streamer.calls) == 1
    assert streamer.content == (
        "Reading the file.\n\nLet me know if you need more.")
    assert "<function=" not in "".join(content_deltas)


def test_naked_function_chunk_split_fuzz_matches_one_shot():
    text = (
        "Reading the file.\n<function=read>\n"
        "<parameter=filePath>\nREADME.md\n</parameter>\n</function>\n"
        "Done."
    )
    reference, _, _ = _run(text)
    for cut in range(len(text) + 1):
        streamer, _, _ = _run([text[:cut], text[cut:]])
        assert streamer.calls == reference.calls, f"cut={cut}"
        assert streamer.content == reference.content, f"cut={cut}"


def test_truncated_naked_attempt_flushes_as_content():
    streamer = ToolCallStreamer((QWENXML_DIALECT,), parameter_schemas=SCHEMAS)
    streamer.push("<function=read>\n<parameter=filePath>\n/pro")
    streamer.finish(truncated=True)
    assert streamer.calls == []
    assert streamer.content == "<function=read>\n<parameter=filePath>\n/pro"


def test_quoted_mid_sentence_attempt_stays_prose():
    # A function element quoted inside prose (not at a line start) is
    # documentation, not an attempt; it must not buffer or become a call.
    text = (
        "The format is <function=read>\n<parameter=filePath>\nX\n"
        "</parameter>\n</function> on its own lines."
    )
    streamer, _, _ = _run(text)
    assert streamer.calls == []
    assert streamer.content == text


def test_dsml_mislabeled_string_flag_coerces_to_schema_type():
    # The model may mark an integer parameter string="true"; the declared
    # schema wins after parse, so the client receives a typed value.
    text = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="read">\n'
        f'<{T}parameter name="filePath" string="false">123</{T}parameter>\n'
        f'<{T}parameter name="limit" string="true">5</{T}parameter>\n'
        f"</{T}invoke>\n"
        f"</{T}tool_calls>"
    )
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert _names_and_arguments(streamer) == [
        ("read", {"filePath": "123", "limit": 5}),
    ]
    assert streamer.telemetry.fires == 0


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


# --- DSML emission failures under large parameter values ---------------------
#
# Three ways a served DSML tool call fails to become a tool call, observed
# through an OpenAI-compatible client. The defects are structural and
# reproduce from a short fixture; the multi-kilobyte parameter values that
# provoke the model into emitting them are not needed to pin the handling.
# Each test asserts the behavior the current implementation has, including
# where that behavior is the defect itself.

HTML_BODY = (
    "<!DOCTYPE html>\n<html>\n<head>\n<title>Demo</title>\n</head>\n"
    "<body>\n<h1>Demo</h1>\n</body>\n</html>"
)

DSML_NAKED_INVOKE = (
    f'<{T}invoke name="bash">\n'
    f'<{T}parameter name="command" string="true">ls</{T}parameter>\n'
    f"</{T}invoke>"
)

DSML_BASH_BLOCK = f"<{T}tool_calls>\n{DSML_NAKED_INVOKE}\n</{T}tool_calls>"

# The traced shape: the closing quote of the name attribute is dropped, no
# interior parameter close is emitted, and only one of the tool's two
# parameters appears.
DSML_ALL_THREE_DEFECTS = (
    f"<{T}tool_calls>\n"
    f'<{T}invoke name="write">\n'
    f'<{T}parameter name="content string="true">{HTML_BODY}\n'
    f"</{T}invoke>\n"
    f"</{T}tool_calls>"
)

# Only the missing interior parameter close.
DSML_MISSING_PARAM_CLOSE = (
    f"<{T}tool_calls>\n"
    f'<{T}invoke name="write">\n'
    f'<{T}parameter name="content" string="true">{HTML_BODY}\n'
    f"</{T}invoke>\n"
    f"</{T}tool_calls>"
)

# Only the unclosed name-attribute quote.
DSML_UNCLOSED_NAME_ONLY = (
    f"<{T}tool_calls>\n"
    f'<{T}invoke name="write">\n'
    f'<{T}parameter name="content string="true">{HTML_BODY}</{T}parameter>\n'
    f"</{T}invoke>\n"
    f"</{T}tool_calls>"
)

# The same defects on a block that never closed, which is what generation
# stopping at the token limit produces.
DSML_CUT_MID_VALUE = (
    f"<{T}tool_calls>\n"
    f'<{T}invoke name="write">\n'
    f'<{T}parameter name="content string="true"><!DOCTYPE html>\n'
    f"<html>\n<head>\n<title>De"
)


def _finish_with(text, *, dialects=(DSML_DIALECT,), truncated):
    """Push one text and finish, choosing the truncation branch explicitly."""
    content_deltas: list[str] = []
    streamer = ToolCallStreamer(
        dialects,
        parameter_schemas=SCHEMAS,
        emit_content=content_deltas.append,
    )
    streamer.push(text)
    streamer.finish(truncated=truncated)
    return streamer, content_deltas


# --- A close marker with no matching open ------------------------------------

def test_dsml_orphan_close_after_naked_invoke_still_fires_the_call():
    # The wrapper open is dropped and the wrapper close is kept. The naked
    # invoke buffers as an attempt and repair salvages the call. That call
    # unit terminates DSML, so the orphaned wrapper close cannot leak.
    streamer, content_deltas, _ = _run(
        DSML_NAKED_INVOKE + f"\n</{T}tool_calls>", dialects=(DSML_DIALECT,))
    assert _names_and_arguments(streamer) == [("bash", {"command": "ls"})]
    assert streamer.content == ""
    assert content_deltas == []
    assert streamer.telemetry.as_dict() == {
        "fires": 1, "salvaged": 1, "failed": 0}


def test_dsml_orphan_close_duplicated_after_block_still_fires_the_call():
    # A well-formed outer block terminates DSML at its own close marker, so a
    # second close cannot leak into the served turn.
    streamer, _, _ = _run(
        DSML_BASH_BLOCK + f"\n</{T}tool_calls>", dialects=(DSML_DIALECT,))
    assert _names_and_arguments(streamer) == [("bash", {"command": "ls"})]
    assert streamer.content == ""
    assert streamer.telemetry.fires == 0


def test_dsml_orphan_close_control_block_leaves_no_content():
    # The control arm: the same call with no surplus marker. A change that
    # suppresses the leaked marker by suppressing the call fails here.
    streamer, _, _ = _run(DSML_BASH_BLOCK, dialects=(DSML_DIALECT,))
    assert _names_and_arguments(streamer) == [("bash", {"command": "ls"})]
    assert streamer.content == ""
    assert streamer.telemetry.fires == 0


def test_dsml_orphan_close_invoke_marker_stays_prose():
    text = f"Done reading.\n</{T}invoke>"
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == text
    assert streamer.telemetry.fires == 0


def test_dsml_orphan_close_indented_stays_prose():
    # The line-start rule applies to opens only, so an indented close is
    # prose for the same reason a column-zero one is: nothing opened.
    text = f"Done reading.\n  </{T}tool_calls>"
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == text


def test_dsml_orphan_close_quoted_mid_sentence_stays_prose():
    text = f"The block ends with </{T}tool_calls> on its own line."
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == text


# --- Failure C: the interior parameter close is missing ----------------------

def test_dsml_interior_missing_param_close_loses_the_call():
    # The traced block. Repair restores the name-attribute quote but has no
    # transformation that inserts a closer inside an invoke body, so the
    # strict parse fails a second time and the whole block, markers
    # included, flushes as visible content.
    streamer, content_deltas, _ = _run(
        DSML_ALL_THREE_DEFECTS, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == DSML_ALL_THREE_DEFECTS
    assert "".join(content_deltas) == DSML_ALL_THREE_DEFECTS
    assert streamer.telemetry.as_dict() == {
        "fires": 1, "salvaged": 0, "failed": 1}


def test_dsml_interior_missing_param_close_alone_is_fatal():
    # Isolating defect two: a well-formed name attribute and a well-formed
    # wrapper are not enough. The missing interior closer alone loses it.
    streamer, _, _ = _run(DSML_MISSING_PARAM_CLOSE, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == DSML_MISSING_PARAM_CLOSE
    assert streamer.telemetry.failed == 1


def test_dsml_interior_missing_param_close_unclosed_name_alone_survives():
    # Isolating defect one: the unclosed name quote is already repaired, so
    # the call fires with the value byte-identical to the emission.
    streamer, _, _ = _run(DSML_UNCLOSED_NAME_ONLY, dialects=(DSML_DIALECT,))
    assert _names_and_arguments(streamer) == [("write", {"content": HTML_BODY})]
    assert streamer.content == ""
    assert streamer.telemetry.salvaged == 1


def test_dsml_interior_missing_param_close_truncated_never_repairs():
    # The truncation branch is reached only by a block that never closed.
    # Closing a half-emitted value would fabricate a plausible but wrong
    # argument, so a truncated turn flushes without ever calling repair.
    truncated, deltas = _finish_with(DSML_CUT_MID_VALUE, truncated=True)
    assert truncated.calls == []
    assert truncated.content == DSML_CUT_MID_VALUE
    assert "".join(deltas) == DSML_CUT_MID_VALUE
    assert truncated.telemetry.fires == 0

    # The same text finished normally proves the two arms differ: repair
    # closes the dangling elements and ships the partial value as a call.
    complete, _ = _finish_with(DSML_CUT_MID_VALUE, truncated=False)
    names = _names_and_arguments(complete)
    assert [name for name, _ in names] == ["write"]
    assert names[0][1]["content"].startswith("<!DOCTYPE html>")
    assert complete.telemetry.salvaged == 1

    # A block whose wrapper close did arrive never reaches the truncation
    # branch at all: it is handled during push, so the flush comes from the
    # repair failure and the truncation flag changes nothing. This shape is
    # a visual lookalike of the branch above, not the same code path.
    lookalike, _ = _finish_with(DSML_ALL_THREE_DEFECTS, truncated=True)
    assert lookalike.calls == []
    assert lookalike.content == DSML_ALL_THREE_DEFECTS
    assert lookalike.telemetry.failed == 1


# --- adjacent structural shapes ----------------------------------------------

def test_dsml_interior_missing_invoke_close_loses_the_call():
    # The parameter closes but the invoke does not. The dangling-closer pass
    # appends the invoke close after the wrapper close, outside the block,
    # so the invoke grammar still finds nothing to match.
    text = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="read">\n'
        f'<{T}parameter name="filePath" string="true">README.md'
        f"</{T}parameter>\n"
        f"</{T}tool_calls>"
    )
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == text
    assert streamer.telemetry.failed == 1


def test_dsml_interior_both_closers_missing_loses_the_call():
    # Both interior closers gone on a turn that was not truncated: repair
    # must not invent a value boundary, and it does not.
    text = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="read">\n'
        f'<{T}parameter name="filePath" string="true">README.md\n'
        f"</{T}tool_calls>"
    )
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == text
    assert streamer.telemetry.failed == 1


def test_dsml_second_parameter_before_first_close_loses_the_call():
    # A second parameter opened before the first closes is rejected rather
    # than merged: the non-greedy value grammar would swallow the second
    # parameter's markup into the first value and ship a one-argument call
    # that no counter records. The rejection routes the block through repair,
    # which declines to invent the value boundary, so the block flushes as
    # visible content and the failure is counted.
    text = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="write">\n'
        f'<{T}parameter name="filePath" string="true">/proj/page.html\n'
        f'<{T}parameter name="content" string="true">{HTML_BODY}'
        f"</{T}parameter>\n"
        f"</{T}invoke>\n"
        f"</{T}tool_calls>"
    )
    streamer, content_deltas, _ = _run(text, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == text
    assert "".join(content_deltas) == text
    assert streamer.telemetry.as_dict() == {
        "fires": 1, "salvaged": 0, "failed": 1}


def test_dsml_parameter_open_quoted_mid_line_stays_in_the_value():
    # The discriminator: a parameter open marker that does not begin a line
    # is value text, so a value documenting the dialect mid-sentence still
    # ships both arguments byte-for-byte.
    quoted = f'A parameter opens with <{T}parameter name="k" string="true">.'
    text = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="write">\n'
        f'<{T}parameter name="filePath" string="true">/proj/dsml.md'
        f"</{T}parameter>\n"
        f'<{T}parameter name="content" string="true">{quoted}'
        f"</{T}parameter>\n"
        f"</{T}invoke>\n"
        f"</{T}tool_calls>"
    )
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert _names_and_arguments(streamer) == [
        ("write", {"filePath": "/proj/dsml.md", "content": quoted}),
    ]
    assert streamer.telemetry.fires == 0


def test_qwenxml_second_parameter_before_first_close_loses_the_call():
    # The native dialect carries the same value grammar, so it gets the same
    # rejection, the same declined repair, and the same visible flush.
    text = (
        "<tool_call>\n<function=write>\n"
        "<parameter=filePath>\n/proj/page.html\n"
        f"<parameter=content>\n{HTML_BODY}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    streamer, content_deltas, _ = _run(text)
    assert streamer.calls == []
    assert streamer.content == text
    assert "".join(content_deltas) == text
    assert streamer.telemetry.as_dict() == {
        "fires": 1, "salvaged": 0, "failed": 1}


def test_dsml_string_attribute_quote_dropped_loses_the_call():
    # The mirror of the repaired defect: the dropped quote belongs to the
    # string attribute rather than the name attribute, and no repair
    # transformation covers that side.
    text = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="read">\n'
        f'<{T}parameter name="filePath" string="true>README.md'
        f"</{T}parameter>\n"
        f"</{T}invoke>\n"
        f"</{T}tool_calls>"
    )
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == text
    assert streamer.telemetry.failed == 1


def test_dsml_literal_parameter_close_inside_value_loses_the_call():
    # A value that documents the dialect terminates itself early: the first
    # closer ends the parameter and the rest of the value becomes unparsed
    # text inside the invoke.
    text = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="write">\n'
        f'<{T}parameter name="content" string="true">A value ends at '
        f"</{T}parameter> in the docs.</{T}parameter>\n"
        f"</{T}invoke>\n"
        f"</{T}tool_calls>"
    )
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == text
    assert streamer.telemetry.failed == 1


def test_dsml_indented_markers_never_reach_the_tool_streamer():
    # Markers count only at the start of a line, so an indented block never
    # buffers. The text the strict parser would accept prints verbatim
    # instead, and the repair counters stay at zero because repair is never
    # reached.
    text = (
        f"  <{T}tool_calls>\n"
        f'  <{T}invoke name="read">\n'
        f'  <{T}parameter name="filePath" string="true">README.md'
        f"</{T}parameter>\n"
        f"  </{T}invoke>\n"
        f"  </{T}tool_calls>"
    )
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == text
    assert streamer.telemetry.fires == 0


def test_dsml_duplicate_name_attribute_loses_the_call():
    # Two name attributes on one invoke: the grammar rejects rather than
    # picking one, and no repair transformation drops the surplus.
    text = (
        f"<{T}tool_calls>\n"
        f'<{T}invoke name="read" name="write">\n'
        f'<{T}parameter name="filePath" string="true">README.md'
        f"</{T}parameter>\n"
        f"</{T}invoke>\n"
        f"</{T}tool_calls>"
    )
    streamer, _, _ = _run(text, dialects=(DSML_DIALECT,))
    assert streamer.calls == []
    assert streamer.content == text
    assert streamer.telemetry.failed == 1


def test_dsml_crlf_and_trailing_space_marker_variants_still_parse():
    # Marker matching is exact-string, but neither a carriage return before
    # the newline nor a trailing space after a marker sits inside a marker,
    # so both variants parse strictly and fire the call.
    crlf, _, _ = _run(
        DSML_BASH_BLOCK.replace("\n", "\r\n"), dialects=(DSML_DIALECT,))
    assert _names_and_arguments(crlf) == [("bash", {"command": "ls"})]
    assert crlf.content == ""

    spaced = (
        f"<{T}tool_calls> \n"
        f'<{T}invoke name="bash"> \n'
        f'<{T}parameter name="command" string="true">ls</{T}parameter> \n'
        f"</{T}invoke> \n"
        f"</{T}tool_calls> "
    )
    streamer, _, _ = _run(spaced, dialects=(DSML_DIALECT,))
    assert _names_and_arguments(streamer) == [("bash", {"command": "ls"})]
    assert streamer.content == ""


@pytest.mark.parametrize(
    "text",
    [DSML_ALL_THREE_DEFECTS, DSML_NAKED_INVOKE + f"\n</{T}tool_calls>"],
    ids=["all-three-defects", "orphan-close"],
)
def test_dsml_chunk_splits_near_marker_boundaries_match_one_shot(text):
    # Chunk-split invariance on the malformed shapes. An every-cut sweep of a
    # real multi-kilobyte value is expensive, so the cuts are sampled around
    # each marker occurrence, which is where the resumable scan and the
    # held-tail logic can differ.
    markers = (
        f"<{T}tool_calls>", f"</{T}tool_calls>", f"<{T}invoke",
        f"</{T}invoke>", f"<{T}parameter", f"</{T}parameter>",
    )
    cuts: set[int] = set()
    for marker in markers:
        start = 0
        while True:
            at = text.find(marker, start)
            if at < 0:
                break
            cuts.update(
                cut for cut in range(at - 2, at + len(marker) + 3)
                if 0 <= cut <= len(text)
            )
            start = at + 1
    assert cuts, "the fixture must contain markers to sample around"
    reference, _, _ = _run(text, dialects=(DSML_DIALECT,))
    for cut in sorted(cuts):
        streamer, _, _ = _run(
            [text[:cut], text[cut:]], dialects=(DSML_DIALECT,))
        assert streamer.calls == reference.calls, f"cut={cut}"
        assert streamer.content == reference.content, f"cut={cut}"


# --- the same failure classes in the Qwen XML dialect -------------------------
#
# The dialect Ornith is served with carries the same three structural risks:
# a close marker anchors nothing, an interior closer has no repair, and a
# truncated block must never be closed by guesswork.

QWEN_NAKED_FUNCTION = (
    "<function=read>\n<parameter=filePath>\nREADME.md\n</parameter>\n"
    "</function>"
)
QWEN_ONE_CALL = f"<tool_call>\n{QWEN_NAKED_FUNCTION}\n</tool_call>"


def test_qwenxml_orphan_close_after_naked_function_still_fires_the_call():
    streamer, _, _ = _run(QWEN_NAKED_FUNCTION + "\n</tool_call>")
    assert _names_and_arguments(streamer) == [("read", {"filePath": "README.md"})]
    # The orphaned close leaks into visible content, as in the DSML dialect.
    assert streamer.content == "\n</tool_call>"
    assert streamer.telemetry.salvaged == 1


def test_qwenxml_orphan_close_duplicated_after_block_still_fires_the_call():
    streamer, _, _ = _run(QWEN_ONE_CALL + "\n</tool_call>")
    assert _names_and_arguments(streamer) == [("read", {"filePath": "README.md"})]
    assert streamer.content == "\n</tool_call>"
    assert streamer.telemetry.fires == 0


def test_qwenxml_orphan_close_control_block_leaves_no_content():
    streamer, _, _ = _run(QWEN_ONE_CALL)
    assert _names_and_arguments(streamer) == [("read", {"filePath": "README.md"})]
    assert streamer.content == ""
    assert streamer.telemetry.fires == 0


def test_qwenxml_interior_missing_param_close_loses_the_call():
    text = (
        "<tool_call>\n<function=read>\n<parameter=filePath>\nREADME.md\n"
        "</function>\n</tool_call>"
    )
    streamer, _, _ = _run(text)
    assert streamer.calls == []
    assert streamer.content == text
    assert streamer.telemetry.failed == 1


def test_qwenxml_interior_missing_function_close_loses_the_call():
    text = (
        "<tool_call>\n<function=read>\n<parameter=filePath>\nREADME.md\n"
        "</parameter>\n</tool_call>"
    )
    streamer, _, _ = _run(text)
    assert streamer.calls == []
    assert streamer.content == text
    assert streamer.telemetry.failed == 1


def test_qwenxml_interior_missing_param_close_truncated_never_repairs():
    # The unterminated form is what the token limit produces, and it is the
    # only form that reaches the truncation branch.
    text = "<tool_call>\n<function=read>\n<parameter=filePath>\n/proj/REA"
    truncated, _ = _finish_with(
        text, dialects=(QWENXML_DIALECT,), truncated=True)
    assert truncated.calls == []
    assert truncated.content == text
    assert truncated.telemetry.fires == 0

    # Finished normally the same text repairs into a call carrying the
    # half-emitted path, which is exactly what the truncation guard exists
    # to prevent from reaching a client.
    complete, _ = _finish_with(
        text, dialects=(QWENXML_DIALECT,), truncated=False)
    assert _names_and_arguments(complete) == [("read", {"filePath": "/proj/REA"})]
    assert complete.telemetry.salvaged == 1
