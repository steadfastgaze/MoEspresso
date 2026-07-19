"""Streaming tool-call extraction from generated answer text.

``chat_stream.ReasoningSplitter`` classifies decoded text into reasoning and
answer channels. When a request carries tools, the answer channel passes
through one more classifier: ``ToolCallStreamer`` buffers dialect marker
blocks out of the content stream, parses each completed block with the
strict dialect parser, and publishes structured OpenAI-format tool-call
entries. Text the model wrote around the blocks still flows as content.

The streamer holds back only the longest tail that could be the start of an
open marker, so ordinary content streams with at most a few characters of
extra latency and a marker split across detokenizer chunks is never leaked.
Markers count only at the start of a line, which is where both dialects
instruct the model to put them; a marker quoted mid-sentence stays prose.

A completed block that fails the strict parse goes through the bounded
repair layer; a block that still fails flushes back to the content channel,
so bytes are never silently dropped. An unterminated block at end of turn is
repaired the same way, except when generation stopped at the token limit:
closing a half-emitted value would fabricate a plausible but wrong argument,
so a truncated block always flushes as content. Everything here runs on
already-decoded text after each decode step; nothing touches the token loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from moespresso.runtime.chat_stream import _marker_suffix_length
from moespresso.toolcalls import dsml, qwenxml, repair
from moespresso.toolcalls.repair import RepairTelemetry
from moespresso.toolcalls.types import ToolCall, ToolCallParseError


@dataclass(frozen=True)
class ToolDialect:
    """One text dialect the streamer can extract: markers plus parse/repair.

    ``parse`` and ``repair`` take ``(text, parameter_schemas)`` and return
    the calls in emission order; both raise ``ToolCallParseError`` on text
    they cannot accept. ``attempt_markers`` are substrings that mark a
    call attempt outside a complete block (a naked function element), used
    only by the end-of-turn salvage scan.
    """

    name: str
    open_marker: str
    close_marker: str
    parse: Callable[[str, dict], list[ToolCall]]
    repair: Callable[[str, dict], list[ToolCall]]
    attempt_markers: tuple[str, ...] = ()
    attempt_span_res: tuple = ()


QWENXML_DIALECT = ToolDialect(
    name="qwenxml",
    open_marker=qwenxml.TOOL_CALL_OPEN,
    close_marker=qwenxml.TOOL_CALL_CLOSE,
    parse=qwenxml.parse_qwenxml_tool_calls,
    repair=repair.repair_qwenxml_tool_calls,
    attempt_markers=("<function=",),
    attempt_span_res=(repair._NAKED_FUNCTION_RE,),
)

DSML_DIALECT = ToolDialect(
    name="dsml",
    open_marker=dsml.TOOL_CALLS_OPEN,
    close_marker=dsml.TOOL_CALLS_CLOSE,
    parse=lambda text, schemas: dsml.parse_dsml_tool_calls(text),
    repair=lambda text, schemas: repair.repair_dsml_tool_calls(text),
    attempt_markers=(dsml.INVOKE_OPEN_PREFIX,),
    attempt_span_res=(repair._DSML_NAKED_INVOKE_RE,),
)


def _default_call_id(index: int) -> str:
    return f"call_{index}"


class ToolCallStreamer:
    """Split an answer-channel text stream into content and tool calls.

    ``emit_content`` receives visible text deltas; ``emit_tool_call``
    receives ``(index, entry)`` where ``entry`` is a message-shaped
    OpenAI tool-call object (id, type, function.name, function.arguments as
    a JSON string). Both are optional; the accumulated ``content`` string
    and ``calls`` list carry the same data for non-streaming callers.

    The trailing whitespace of the content stream is always held back
    rather than emitted, then resolved at end of turn: dropped when the
    turn produced calls (so marker-separator newlines never reach the
    content of a tool-call turn) and flushed otherwise (so a prose turn
    keeps every byte). Because the held tail is a pure function of the
    content bytes, any chunking of the same text yields the same content.
    """

    def __init__(
        self,
        dialects,
        *,
        parameter_schemas: dict[str, dict] | None = None,
        emit_content: Callable[[str], None] | None = None,
        emit_tool_call: Callable[[int, dict], None] | None = None,
        repair_enabled: bool = True,
        make_call_id: Callable[[int], str] | None = None,
    ):
        self.dialects = tuple(dialects)
        if not self.dialects:
            raise ValueError("ToolCallStreamer needs at least one dialect")
        self.schemas = parameter_schemas or {}
        self.emit_content = emit_content
        self.emit_tool_call = emit_tool_call
        self.repair_enabled = repair_enabled
        self.make_call_id = make_call_id or _default_call_id
        self.calls: list[dict] = []
        self.telemetry = RepairTelemetry()
        self.content_parts: list[str] = []
        self.buffer = ""
        self.active: ToolDialect | None = None
        self._held_ws = ""
        self._line_start = True
        self._scan_from = 0
        self._finished = False

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    # --- content channel -------------------------------------------------

    def _flush_content(self, text: str) -> None:
        if not text:
            return
        combined = self._held_ws + text
        stripped = combined.rstrip()
        if not stripped:
            self._held_ws = combined
            return
        out, self._held_ws = combined[:len(stripped)], combined[len(stripped):]
        self.content_parts.append(out)
        if self.emit_content is not None:
            self.emit_content(out)

    def _consume(self, length: int) -> str:
        consumed, self.buffer = self.buffer[:length], self.buffer[length:]
        if consumed:
            self._line_start = consumed.endswith("\n")
        return consumed

    # --- call channel ----------------------------------------------------

    def _emit_calls(self, parsed: list[ToolCall]) -> None:
        for call in parsed:
            index = len(self.calls)
            entry = {
                "id": self.make_call_id(index),
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            self.calls.append(entry)
            if self.emit_tool_call is not None:
                self.emit_tool_call(index, entry)

    def _repair_or_flush(self, dialect: ToolDialect, text: str) -> None:
        """Repair a strict-parse failure; text that still fails is content."""
        if self.repair_enabled:
            try:
                salvaged = dialect.repair(text, self.schemas)
            except ToolCallParseError:
                salvaged = []
            self.telemetry.record(salvaged=bool(salvaged))
            if salvaged:
                self._emit_calls(salvaged)
                return
        self._flush_content(text)

    def _handle_block(self, dialect: ToolDialect, block: str) -> None:
        try:
            parsed = dialect.parse(block, self.schemas)
        except ToolCallParseError:
            parsed = []
        if parsed:
            self._emit_calls(parsed)
            return
        self._repair_or_flush(dialect, block)

    # --- marker scanning -------------------------------------------------

    def _at_line_start(self, index: int) -> bool:
        if index == 0:
            return self._line_start
        return self.buffer[index - 1] == "\n"

    def _find_open(self) -> tuple[int, ToolDialect] | None:
        best: tuple[int, ToolDialect] | None = None
        for dialect in self.dialects:
            start = 0
            while True:
                index = self.buffer.find(dialect.open_marker, start)
                if index < 0:
                    break
                if self._at_line_start(index):
                    if best is None or index < best[0]:
                        best = (index, dialect)
                    break
                start = index + 1
        return best

    def _hold_length(self) -> int:
        hold = 0
        for dialect in self.dialects:
            length = _marker_suffix_length(self.buffer, dialect.open_marker)
            if length and self._at_line_start(len(self.buffer) - length):
                hold = max(hold, length)
        return hold

    def push(self, text: str) -> None:
        if self._finished:
            raise RuntimeError("push after finish")
        if not text:
            return
        self.buffer += text
        while True:
            if self.active is not None:
                end = self.buffer.find(
                    self.active.close_marker, self._scan_from)
                if end < 0:
                    # Resume the next scan where this one left off; a close
                    # marker split across chunks can begin at most
                    # len(close)-1 characters before the buffer end. Without
                    # this the whole block would rescan on every push.
                    self._scan_from = max(
                        self._scan_from,
                        len(self.buffer) - len(self.active.close_marker) + 1,
                    )
                    return
                block = self._consume(end + len(self.active.close_marker))
                dialect, self.active = self.active, None
                self._handle_block(dialect, block)
                # The character after a close marker is a block boundary;
                # counting it as a line start lets a glued next block parse.
                self._line_start = True
                continue
            found = self._find_open()
            if found is not None:
                index, dialect = found
                self._flush_content(self._consume(index))
                self.active = dialect
                self._scan_from = len(dialect.open_marker)
                continue
            hold = self._hold_length()
            self._flush_content(self._consume(len(self.buffer) - hold))
            return

    # --- end of turn -----------------------------------------------------

    @staticmethod
    def _line_start_positions(text: str, markers) -> list[int]:
        positions = []
        for marker in markers:
            start = 0
            while True:
                index = text.find(marker, start)
                if index < 0:
                    break
                if index == 0 or text[index - 1] == "\n":
                    positions.append(index)
                    break
                start = index + 1
        return positions

    def _late_salvage(self) -> None:
        """Salvage a call attempt that never formed a complete block.

        Runs only when the turn produced no calls: a naked function or
        invoke element at a line start is still an attempt, and the repair
        layer knows these shapes. A marker quoted mid-sentence never
        triggers salvage, matching the streaming rule. On success the
        attempt regions drop out of the visible content and the prose
        around them survives; streamed clients have already seen the raw
        text, and the shaped message carries the structured calls.
        """
        full = self.content + self._held_ws
        for dialect in self.dialects:
            markers = (dialect.open_marker, *dialect.attempt_markers)
            positions = self._line_start_positions(full, markers)
            if not positions:
                continue
            try:
                salvaged = dialect.repair(full, self.schemas)
            except ToolCallParseError:
                salvaged = []
            self.telemetry.record(salvaged=bool(salvaged))
            if salvaged:
                self._emit_calls(salvaged)
                visible = full
                for span_re in dialect.attempt_span_res:
                    visible = span_re.sub("", visible)
                if dialect.open_marker in visible:
                    # A leaked bare marker has no span to strip; fall back
                    # to cutting at the first attempt position.
                    visible = full[:min(positions)]
                self.content_parts = [visible] if visible.strip() else []
                self._held_ws = ""
            return

    def finish(self, *, truncated: bool = False) -> None:
        """Resolve the tail: unterminated blocks, held text, late salvage.

        ``truncated`` means generation stopped at the token limit, so an
        unterminated block is flushed as content instead of repaired.
        """
        if self._finished:
            return
        if self.active is not None:
            raw, self.buffer = self.buffer, ""
            dialect, self.active = self.active, None
            if truncated:
                self._flush_content(raw)
            else:
                self._repair_or_flush(dialect, raw)
        elif self.buffer:
            self._flush_content(self._consume(len(self.buffer)))
        # The salvage scan runs only when no repair fired yet: a block whose
        # repair already failed would fail again on the superset text and
        # double-count the same attempt.
        if (not self.calls and self.repair_enabled and not truncated
                and self.telemetry.fires == 0):
            self._late_salvage()
        if self._held_ws:
            if self.calls:
                self._held_ws = ""
            else:
                trailing, self._held_ws = self._held_ws, ""
                self.content_parts.append(trailing)
                if self.emit_content is not None:
                    self.emit_content(trailing)
        self._finished = True
