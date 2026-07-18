"""Incremental Chat Completions response shaping.

Generation remains responsible for raw text and token IDs.  This module only
classifies decoded text into reasoning and answer channels for HTTP response
shaping.  The classifier retains marker prefixes across arbitrary chunk
boundaries and never mutates generation state.
"""

from __future__ import annotations

from collections.abc import Callable

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _marker_suffix_length(text: str, marker: str) -> int:
    """Length of the longest suffix of ``text`` that prefixes ``marker``."""
    limit = min(len(text), len(marker) - 1)
    for size in range(limit, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


class ReasoningSplitter:
    """Split generated text into reasoning and answer deltas.

    ``thinking_enabled`` means the rendered generation prompt already opened
    the reasoning section.  Some renderers repeat the opening marker in model
    output; that form is accepted too.  ``emit`` receives ``("reasoning",
    text)`` or ``("content", text)``.
    """

    def __init__(
        self,
        *,
        thinking_enabled: bool,
        emit: Callable[[str, str], None] | None = None,
    ):
        self.mode = "reasoning" if thinking_enabled else "content"
        self.emit = emit
        self.buffer = ""
        self.reasoning_parts: list[str] = []
        self.content_parts: list[str] = []
        self.saw_input = False
        self._at_start = True

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts)

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    def _publish(self, kind: str, text: str) -> None:
        if not text:
            return
        if kind == "reasoning":
            self.reasoning_parts.append(text)
        else:
            self.content_parts.append(text)
        if self.emit is not None:
            self.emit(kind, text)

    def push(self, text: str) -> None:
        if not text:
            return
        self.saw_input = True
        self.buffer += text
        while self.buffer:
            if self.mode == "reasoning":
                if self._at_start:
                    if THINK_OPEN.startswith(self.buffer) and len(self.buffer) < len(THINK_OPEN):
                        return
                    if self.buffer.startswith(THINK_OPEN):
                        self.buffer = self.buffer[len(THINK_OPEN):]
                        self._at_start = False
                        continue
                    self._at_start = False
                close_at = self.buffer.find(THINK_CLOSE)
                if close_at >= 0:
                    self._publish("reasoning", self.buffer[:close_at])
                    self.buffer = self.buffer[close_at + len(THINK_CLOSE):]
                    self.mode = "content"
                    continue
                hold = _marker_suffix_length(self.buffer, THINK_CLOSE)
                end = len(self.buffer) - hold
                self._publish("reasoning", self.buffer[:end])
                self.buffer = self.buffer[end:]
                return

            open_at = self.buffer.find(THINK_OPEN)
            if open_at >= 0:
                self._publish("content", self.buffer[:open_at])
                self.buffer = self.buffer[open_at + len(THINK_OPEN):]
                self.mode = "reasoning"
                self._at_start = False
                continue
            hold = _marker_suffix_length(self.buffer, THINK_OPEN)
            end = len(self.buffer) - hold
            self._publish("content", self.buffer[:end])
            self.buffer = self.buffer[end:]
            return

    def finish(self) -> None:
        """Publish an unterminated tail in its current channel."""
        if self.buffer:
            self._publish(self.mode, self.buffer)
            self.buffer = ""


def split_complete_text(
    text: str,
    *,
    thinking_enabled: bool,
    prompt_opened_thinking: bool = False,
) -> tuple[str, str]:
    """Return ``(reasoning, content)`` for one complete generated string.

    Legacy generators and simple HTTP tests often return plain text while the
    template default says thinking is enabled.  A string without either marker
    remains ordinary content.  Real thinking output normally contains a close
    marker because the generation prompt already supplied the open marker.
    ``prompt_opened_thinking`` covers the truncated case: when the rendered
    prompt itself opened the reasoning section and generation stopped before
    the close marker, the whole string is reasoning, not content.
    """
    if THINK_OPEN not in text and THINK_CLOSE not in text:
        if thinking_enabled and prompt_opened_thinking:
            return text, ""
        return "", text
    splitter = ReasoningSplitter(thinking_enabled=thinking_enabled)
    splitter.push(text)
    splitter.finish()
    return splitter.reasoning, splitter.content
