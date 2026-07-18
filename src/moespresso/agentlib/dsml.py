"""DeepSeek DSML tool-call parsing from completion text.

The DeepSeek-V4 renderer instructs the model to emit tool invocations as a
DSML block inside the completion text, and the serve layer returns that text
verbatim as the assistant message content without parsing it. The client
therefore extracts the calls from the text. This module is the text-dialect
counterpart of ``toolcalls.parse_tool_calls``: well-formed calls come back as
``ToolCall`` values in emission order, and a malformed block raises
``ToolCallParseError`` so the same repair seam catches both dialects.

The grammar matches the renderer's tools template
(``runtime.deepseek_v4.renderer.TOOLS_TEMPLATE``):

    <DSML tool_calls>
    <DSML invoke name="$TOOL_NAME">
    <DSML parameter name="$NAME" string="true|false">$VALUE</DSML parameter>
    </DSML invoke>
    </DSML tool_calls>

where ``DSML`` stands for the renderer's ``DSML_TOKEN``. A parameter with
``string="true"`` carries its raw text value unchanged (multiline values
included), and ``string="false"`` carries JSON that must decode successfully.
The parser is strict: an unclosed block, an invoke without a name, a
parameter without the ``string`` attribute, stray text between elements, or
invalid JSON all raise instead of guessing. Tool-call repair is a separate
component that builds on exactly that error.
"""

from __future__ import annotations

import json
import re

from moespresso.agentlib.toolcalls import ToolCall, ToolCallParseError
from moespresso.runtime.deepseek_v4.renderer import DSML_TOKEN

TOOL_CALLS_OPEN = f"<{DSML_TOKEN}tool_calls>"
TOOL_CALLS_CLOSE = f"</{DSML_TOKEN}tool_calls>"
_INVOKE_OPEN_PREFIX = f"<{DSML_TOKEN}invoke"
_PARAM_OPEN_PREFIX = f"<{DSML_TOKEN}parameter"

_BLOCK_RE = re.compile(
    re.escape(TOOL_CALLS_OPEN) + r"(.*?)" + re.escape(TOOL_CALLS_CLOSE), re.DOTALL
)
_INVOKE_RE = re.compile(
    re.escape(_INVOKE_OPEN_PREFIX)
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
