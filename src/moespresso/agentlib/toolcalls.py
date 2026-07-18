"""Native tool-call parsing: OpenAI ``tool_calls`` in a response message.

This is the strict path. A malformed entry raises ``ToolCallParseError``
instead of being coerced; the tool-call repair layer is a separate, later
component that catches exactly this error and decides what to salvage.
Keeping the parser strict keeps that seam clean.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field


class ToolCallParseError(Exception):
    """A response message carries tool calls the strict parser cannot accept."""


@dataclass(frozen=True)
class ToolCall:
    """One parsed tool invocation: name plus decoded arguments object."""

    name: str
    arguments: dict = field(default_factory=dict)
    id: str | None = None


def parse_tool_calls(message: dict) -> list[ToolCall]:
    """Parse ``message["tool_calls"]`` into ``ToolCall`` values, in order.

    Returns an empty list when the message has no tool calls. Arguments are
    accepted as an OpenAI-style JSON string or as an already-decoded object;
    both must denote a JSON object. Anything else raises ToolCallParseError.
    """
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, list):
        raise ToolCallParseError("tool_calls must be a list")
    calls = []
    for index, entry in enumerate(raw_calls):
        calls.append(_parse_entry(entry, index))
    return calls


def _parse_entry(entry, index: int) -> ToolCall:
    if not isinstance(entry, dict):
        raise ToolCallParseError(f"tool_calls[{index}] must be an object")
    entry_type = entry.get("type", "function")
    if entry_type != "function":
        raise ToolCallParseError(
            f"tool_calls[{index}] has unsupported type {entry_type!r}")
    function = entry.get("function")
    if not isinstance(function, dict):
        raise ToolCallParseError(f"tool_calls[{index}] lacks a function object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ToolCallParseError(f"tool_calls[{index}] lacks a function name")
    call_id = entry.get("id")
    if call_id is not None and not isinstance(call_id, str):
        raise ToolCallParseError(f"tool_calls[{index}] id must be a string")
    return ToolCall(
        name=name,
        arguments=_parse_arguments(function.get("arguments"), index),
        id=call_id,
    )


def _parse_arguments(raw, index: int) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ToolCallParseError(
                f"tool_calls[{index}] arguments are not valid JSON: {e}") from e
        if not isinstance(decoded, dict):
            raise ToolCallParseError(
                f"tool_calls[{index}] arguments must decode to an object")
        return decoded
    raise ToolCallParseError(
        f"tool_calls[{index}] arguments must be a JSON string or object")
