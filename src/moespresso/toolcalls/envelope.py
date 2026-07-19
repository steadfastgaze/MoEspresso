"""Terminus-2-style JSON action envelope parsed from completion text.

The dialect asks the model to answer every turn with a single JSON object
carrying four keys: ``analysis`` (what the latest output shows), ``plan``
(the next step), ``commands`` (the tool calls to run now), and
``task_complete`` (whether the task is done). No request-level tools are
sent; the tool schemas travel inside the system prompt, and the whole tool
surface is this one envelope. The shape follows the Terminus-2 agent, whose
Terminal-Bench results are the evidence for this dialect on the Ornith
lineage; command entries here name a registry tool with a JSON ``args``
object instead of raw terminal keystrokes so the tool surface matches the
other dialects.

The parser is strict about exactly what the instructions demand: the whole
reply must be one JSON object, the four keys are the only top-level keys
allowed, and each command entry carries only ``tool`` and ``args``. Anything
else raises ``ToolCallParseError`` so the shared repair seam catches it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from moespresso.toolcalls.types import ToolCall, ToolCallParseError

ENVELOPE_KEYS = frozenset({"analysis", "plan", "commands", "task_complete"})
COMMAND_KEYS = frozenset({"tool", "args"})


@dataclass(frozen=True)
class ActionEnvelope:
    """One parsed action envelope: text channels, tool calls, completion flag."""

    analysis: str
    plan: str
    calls: tuple[ToolCall, ...]
    task_complete: bool
    raw: dict = field(default_factory=dict)


def parse_action_envelope(content: str | None) -> ActionEnvelope:
    """Parse an assistant reply as a single strict action envelope.

    Raises ``ToolCallParseError`` when the reply is not exactly one JSON
    object of the documented shape.
    """
    if not content or not content.strip():
        raise ToolCallParseError("empty reply where an action envelope is required")
    try:
        decoded = json.loads(content.strip())
    except json.JSONDecodeError as e:
        raise ToolCallParseError(f"reply is not a JSON action envelope: {e}") from e
    return envelope_from_object(decoded)


def envelope_from_object(decoded) -> ActionEnvelope:
    """Validate a decoded object against the strict envelope shape."""
    if not isinstance(decoded, dict):
        raise ToolCallParseError("action envelope must be a JSON object")
    unknown = sorted(set(decoded) - ENVELOPE_KEYS)
    if unknown:
        raise ToolCallParseError(
            f"action envelope carries unknown key(s): {', '.join(unknown)}")
    commands = decoded.get("commands")
    if not isinstance(commands, list):
        raise ToolCallParseError(
            "action envelope must carry 'commands' as a list")
    task_complete = decoded.get("task_complete")
    if not isinstance(task_complete, bool):
        raise ToolCallParseError(
            "action envelope must carry 'task_complete' as a boolean")
    analysis = _text_channel(decoded, "analysis")
    plan = _text_channel(decoded, "plan")
    calls = tuple(
        _parse_command(entry, index) for index, entry in enumerate(commands)
    )
    return ActionEnvelope(
        analysis=analysis,
        plan=plan,
        calls=calls,
        task_complete=task_complete,
        raw=decoded,
    )


def _text_channel(decoded: dict, key: str) -> str:
    value = decoded.get(key, "")
    if not isinstance(value, str):
        raise ToolCallParseError(f"action envelope key {key!r} must be a string")
    return value


def _parse_command(entry, index: int) -> ToolCall:
    if not isinstance(entry, dict):
        raise ToolCallParseError(f"commands[{index}] must be an object")
    unknown = sorted(set(entry) - COMMAND_KEYS)
    if unknown:
        raise ToolCallParseError(
            f"commands[{index}] carries unknown key(s): {', '.join(unknown)}")
    tool = entry.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ToolCallParseError(f"commands[{index}] lacks a tool name")
    args = entry.get("args", {})
    if not isinstance(args, dict):
        raise ToolCallParseError(f"commands[{index}] args must be an object")
    return ToolCall(name=tool, arguments=args)


_ENVELOPE_INSTRUCTIONS = """## Actions

You have access to the following tools:

<tools>
{schemas}
</tools>

Reply to every message with a single JSON object and nothing else: no prose
before or after it, no code fences. The object has exactly these keys:

{{
  "analysis": "what the latest output shows about the task",
  "plan": "the next step or two",
  "commands": [{{"tool": "tool_name", "args": {{"parameter": "value"}}}}],
  "task_complete": false
}}

Rules:
- "commands" lists the tool calls to run now, in order. Use [] when there is
  nothing left to run.
- Each command has exactly two keys: "tool" and "args". Argument names and
  types follow the tool schemas above.
- Set "task_complete" to true only when the task is done, and put the final
  answer for the user in "analysis"."""


def envelope_system_block(tools: list[dict]) -> str:
    """The system-prompt block teaching the envelope over the given tools.

    ``tools`` is the OpenAI-format array from ``ToolRegistry.openai_tools``;
    the schemas are embedded one JSON object per line, mirroring how the
    native template lists them.
    """
    functions = [tool.get("function", tool) for tool in tools]
    schemas = "\n".join(json.dumps(fn, ensure_ascii=False) for fn in functions)
    return _ENVELOPE_INSTRUCTIONS.format(schemas=schemas)
