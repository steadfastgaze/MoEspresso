"""The DeepSeek DSML tool-call dialect: grammar, serializers, and parser.

DSML is the text dialect the DeepSeek-V4 encoder teaches: tool invocations
are emitted inside the completion text as a marker block built from
``DSML_TOKEN``:

    <DSML tool_calls>
    <DSML invoke name="$TOOL_NAME">
    <DSML parameter name="$NAME" string="true|false">$VALUE</DSML parameter>
    </DSML invoke>
    </DSML tool_calls>

This module owns both directions of that grammar. ``render_tools`` renders
the instruction block that teaches the dialect; the DeepSeek-V4 renderer
embeds it in the system region, and the serve layer can teach it to another
family the same way. ``render_dsml_tool_calls`` serializes OpenAI-format
``tool_calls`` entries back into a DSML block for history re-rendering.
``parse_dsml_tool_calls`` is the strict parser: well-formed calls come back
as ``ToolCall`` values in emission order, and a malformed block raises
``ToolCallParseError`` so the same repair seam catches every dialect.

A parameter with ``string="true"`` carries its raw text value unchanged
(multiline values included), and ``string="false"`` carries JSON that must
decode successfully. The parser is strict: an unclosed block, an invoke
without a name, a parameter without the ``string`` attribute, stray text
between elements, or invalid JSON all raise instead of guessing. Tool-call
repair is a separate component that builds on exactly that error.
"""

from __future__ import annotations

import json
import re
from typing import Any

from moespresso.toolcalls.types import ToolCall, ToolCallParseError

DSML_TOKEN = "｜DSML｜"

_THINKING_START_TOKEN = "<think>"
_THINKING_END_TOKEN = "</think>"

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

TOOL_CALLS_OPEN = f"<{DSML_TOKEN}tool_calls>"
TOOL_CALLS_CLOSE = f"</{DSML_TOKEN}tool_calls>"
INVOKE_OPEN_PREFIX = f"<{DSML_TOKEN}invoke"
_PARAM_OPEN_PREFIX = f"<{DSML_TOKEN}parameter"

_BLOCK_RE = re.compile(
    re.escape(TOOL_CALLS_OPEN) + r"(.*?)" + re.escape(TOOL_CALLS_CLOSE), re.DOTALL
)
_INVOKE_RE = re.compile(
    re.escape(INVOKE_OPEN_PREFIX)
    + r'\s+name="([^"]*)"\s*>(.*?)'
    + re.escape(f"</{DSML_TOKEN}invoke>"),
    re.DOTALL,
)
_PARAM_RE = re.compile(
    re.escape(_PARAM_OPEN_PREFIX)
    + r'\s+name="([^"]*)"\s+string="(true|false)"\s*>(.*?)'
    + re.escape(f"</{DSML_TOKEN}parameter>"),
    re.DOTALL,
)


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_tools(tools: list[dict]) -> str:
    """The DSML tools instruction block for a list of function schemas.

    ``tools`` is the unwrapped function-object list (each entry the value of
    an OpenAI tool's ``function`` key). The block text is deterministic for a
    fixed tool list, so a tool set held constant across turns renders into a
    stable prompt prefix.
    """
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(_to_json(t) for t in tools),
        dsml_token=DSML_TOKEN,
        thinking_start_token=_THINKING_START_TOKEN,
        thinking_end_token=_THINKING_END_TOKEN,
    )


def encode_arguments_to_dsml(tool_call: dict) -> str:
    """Encode OpenAI function arguments into DSML parameter tags.

    ``arguments`` may be a decoded object or the OpenAI wire shape (a JSON
    string). Anything that does not decode to an object renders as a
    single ``arguments`` parameter carrying the raw value.
    """
    raw = tool_call["arguments"]
    if isinstance(raw, dict):
        arguments = raw
    else:
        try:
            arguments = json.loads(raw)
        except Exception:
            arguments = {"arguments": raw}
        if not isinstance(arguments, dict):
            arguments = {"arguments": raw}

    parts = []
    for key, value in arguments.items():
        is_str = isinstance(value, str)
        rendered = value if is_str else _to_json(value)
        parts.append(
            f'<{DSML_TOKEN}parameter name="{key}" string="{str(is_str).lower()}">'
            f"{rendered}</{DSML_TOKEN}parameter>"
        )
    return "\n".join(parts)


def _tool_calls_from_openai_format(tool_calls: list[dict]) -> list[dict]:
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def render_dsml_tool_calls(tool_calls: list[dict]) -> str:
    """Serialize OpenAI-format ``tool_calls`` entries into one DSML block.

    Each entry's ``function.arguments`` may be a JSON string (the OpenAI wire
    shape) or an already-decoded object. The rendered block is what an
    assistant history turn carries, so the model sees its past invocations in
    the dialect it emitted them in.
    """
    converted = _tool_calls_from_openai_format(tool_calls)
    calls = [
        f'<{DSML_TOKEN}invoke name="{tc.get("name")}">\n'
        f"{encode_arguments_to_dsml(tc)}\n"
        f"</{DSML_TOKEN}invoke>"
        for tc in converted
    ]
    rendered_calls = "\n".join(calls)
    return (
        f"\n\n<{DSML_TOKEN}tool_calls>\n"
        f"{rendered_calls}\n"
        f"</{DSML_TOKEN}tool_calls>"
    )


def has_tool_call_block(content: str | None) -> bool:
    """True when the content contains a DSML tool-calls open marker."""
    return bool(content) and TOOL_CALLS_OPEN in content


def parse_dsml_tool_calls(content: str | None) -> list[ToolCall]:
    """Parse every DSML tool-call block in ``content`` into ``ToolCall`` values.

    Returns an empty list when the content carries no block. Raises
    ``ToolCallParseError`` on any structural defect so a truncated or
    hand-mangled block never silently drops a call.
    """
    if not content or TOOL_CALLS_OPEN not in content:
        return []
    blocks = _BLOCK_RE.findall(content)
    if len(blocks) != content.count(TOOL_CALLS_OPEN):
        raise ToolCallParseError(
            "unclosed or malformed DSML tool_calls block (open markers without a close)"
        )
    calls: list[ToolCall] = []
    for block in blocks:
        calls.extend(_parse_block(block))
    return calls


def _parse_block(block: str) -> list[ToolCall]:
    remainder = block
    calls: list[ToolCall] = []
    for match in _INVOKE_RE.finditer(block):
        name, inner = match.group(1), match.group(2)
        if not name:
            raise ToolCallParseError("DSML invoke carries an empty tool name")
        calls.append(ToolCall(name=name, arguments=_parse_parameters(inner, name)))
        remainder = remainder.replace(match.group(0), "", 1)
    if not calls:
        raise ToolCallParseError("DSML tool_calls block contains no well-formed invoke")
    if remainder.strip():
        raise ToolCallParseError(
            f"unparsed text inside DSML tool_calls block: {remainder.strip()[:120]!r}"
        )
    return calls


def _parse_parameters(inner: str, tool_name: str) -> dict:
    arguments: dict = {}
    remainder = inner
    for match in _PARAM_RE.finditer(inner):
        key, is_string, raw = match.group(1), match.group(2), match.group(3)
        if not key:
            raise ToolCallParseError(f"{tool_name}: DSML parameter carries an empty name")
        if key in arguments:
            raise ToolCallParseError(f"{tool_name}: duplicate DSML parameter {key!r}")
        if is_string == "true":
            arguments[key] = raw
        else:
            try:
                arguments[key] = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ToolCallParseError(
                    f"{tool_name}: parameter {key!r} is marked non-string but is not "
                    f"valid JSON: {e}"
                ) from e
        remainder = remainder.replace(match.group(0), "", 1)
    if remainder.strip():
        raise ToolCallParseError(
            f"{tool_name}: unparsed text inside DSML invoke: {remainder.strip()[:120]!r}"
        )
    return arguments
