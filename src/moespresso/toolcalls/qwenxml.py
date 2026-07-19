"""Qwen XML tool-call parsing from completion text.

The vendored qwen3_5 chat template renders request tools as a leading system
block that instructs the model to emit tool invocations as XML-tagged text:

    <tool_call>
    <function=$TOOL_NAME>
    <parameter=$NAME>
    $VALUE
    </parameter>
    </function>
    </tool_call>

The serve layer converts these blocks into structured ``tool_calls`` when
the request carries tools, and the agent client parses them from verbatim
text. Both sides use this module. It is the native-dialect counterpart of
``dsml.parse_dsml_tool_calls``: well-formed calls come back as ``ToolCall``
values in emission order, and a malformed block raises
``ToolCallParseError`` so the same repair seam catches every dialect.

Parameter values arrive as raw text. The template renders one newline around
each value, so the parser trims exactly one leading and one trailing newline
and keeps everything else, multiline values included. A parameter whose
declared schema type is not string must decode as JSON of that type; the
schemas come from the caller's tool registry. The parser is strict: an
unclosed block, a function element outside a block, stray text between
elements, a duplicate parameter, or an undecodable typed value all raise
instead of guessing. Tool-call repair is a separate component that builds on
exactly that error.
"""

from __future__ import annotations

import json
import re

from moespresso.toolcalls.types import ToolCall, ToolCallParseError

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
_FUNCTION_OPEN_PREFIX = "<function="
_UNDECODABLE = object()

_BLOCK_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE), re.DOTALL
)
_FUNCTION_RE = re.compile(r"<function=([^>\n]*)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]*)>(.*?)</parameter>", re.DOTALL)

# JSON schema type name -> the Python types a decoded value may have.
# bool is checked first because bool is a subclass of int.
_JSON_TYPES = {
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def union_members(declared) -> tuple[tuple[str, ...], bool]:
    """Resolve a schema ``type`` value to ``(members, nullable)``.

    JSON Schema allows a union list such as ``["integer", "string",
    "null"]``: ``members`` is every non-null string member in declared
    order and ``nullable`` records whether null is a member. A bare
    ``"null"`` resolves to no members, and every unrecognized shape
    resolves to ``((), False)``, which downstream treats as undeclared
    (the raw text is kept).
    """
    if isinstance(declared, str):
        if declared == "null":
            return (), True
        return (declared,), False
    if isinstance(declared, list):
        members = tuple(
            t for t in declared if isinstance(t, str) and t != "null")
        nullable = any(t == "null" for t in declared)
        return members, nullable
    return (), False


def accepts_raw_text(members: tuple[str, ...]) -> bool:
    """Whether a union keeps raw text: a string or unrecognized member."""
    return any(m == "string" or m not in _JSON_TYPES for m in members)


def declared_type_for(properties: dict, key: str):
    """The raw declared ``type`` for one property, tolerating odd shapes."""
    prop = properties.get(key)
    return prop.get("type") if isinstance(prop, dict) else None


def has_qwenxml_tool_call(content: str | None) -> bool:
    """True when the content carries a tool-call or function open marker."""
    if not content:
        return False
    return TOOL_CALL_OPEN in content or _FUNCTION_OPEN_PREFIX in content


def strip_qwenxml_blocks(content: str) -> str:
    """The content with every well-formed tool-call block removed."""
    return _BLOCK_RE.sub("", content)


def parse_qwenxml_tool_calls(
    content: str | None,
    parameter_schemas: dict[str, dict] | None = None,
) -> list[ToolCall]:
    """Parse every ``<tool_call>`` block in ``content`` into ``ToolCall`` values.

    ``parameter_schemas`` maps a tool name to its JSON schema (the
    ``ToolSpec.parameters`` object); it drives the typed-value decoding.
    Returns an empty list when the content carries no tool-call markers.
    Raises ``ToolCallParseError`` on any structural defect so a truncated or
    hand-mangled block never silently drops a call.
    """
    if not has_qwenxml_tool_call(content):
        return []
    assert content is not None
    blocks = _BLOCK_RE.findall(content)
    if len(blocks) != content.count(TOOL_CALL_OPEN):
        raise ToolCallParseError(
            "unclosed or malformed <tool_call> block (open markers without a close)"
        )
    outside = strip_qwenxml_blocks(content)
    if _FUNCTION_OPEN_PREFIX in outside:
        raise ToolCallParseError(
            "function element outside a <tool_call> block"
        )
    return [_parse_block(block, parameter_schemas or {}) for block in blocks]


def _parse_block(block: str, parameter_schemas: dict[str, dict]) -> ToolCall:
    matches = list(_FUNCTION_RE.finditer(block))
    if not matches:
        raise ToolCallParseError(
            "tool_call block contains no well-formed function element"
        )
    if len(matches) > 1 or block.count(_FUNCTION_OPEN_PREFIX) > 1:
        raise ToolCallParseError(
            "tool_call block must contain exactly one function element"
        )
    match = matches[0]
    name = match.group(1)
    if not name:
        raise ToolCallParseError("function element carries an empty tool name")
    remainder = block.replace(match.group(0), "", 1)
    if remainder.strip():
        raise ToolCallParseError(
            f"unparsed text inside tool_call block: {remainder.strip()[:120]!r}"
        )
    schema = parameter_schemas.get(name) or {}
    arguments = _parse_parameters(match.group(2), name, schema)
    return ToolCall(name=name, arguments=arguments)


def _parse_parameters(inner: str, tool_name: str, schema: dict) -> dict:
    properties = schema.get("properties") or {}
    arguments: dict = {}
    remainder = inner
    for match in _PARAM_RE.finditer(inner):
        key, raw = match.group(1), match.group(2)
        if not key:
            raise ToolCallParseError(
                f"{tool_name}: parameter element carries an empty name")
        if key in arguments:
            raise ToolCallParseError(f"{tool_name}: duplicate parameter {key!r}")
        declared = declared_type_for(properties, key)
        arguments[key] = _decode_value(_trim_value(raw), declared, tool_name, key)
        remainder = remainder.replace(match.group(0), "", 1)
    if remainder.strip():
        raise ToolCallParseError(
            f"{tool_name}: unparsed text inside function element: "
            f"{remainder.strip()[:120]!r}"
        )
    return arguments


def _trim_value(raw: str) -> str:
    """Trim the single newline the format places on each side of a value."""
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    return raw


def _decode_value(value: str, declared, tool_name: str, key: str):
    """Decode a raw text value against its declared schema type.

    Undeclared and string-typed parameters keep the raw text. A declared
    union tries its typed members in declared order (the first whose JSON
    decoding matches wins), falls back to raw text when a string member is
    present, and accepts the null literal when nullable; a bare ``null``
    under ``["string", "null"]`` reads as null, because a nullable client
    schema expects the null, not the four-letter string. A value that
    matches no member raises so the repair layer sees a typed failure
    instead of an executor crash.
    """
    members, nullable = union_members(declared)
    if nullable and value == "null":
        return None
    if not members:
        if nullable:
            # A null-only type accepts exactly the null literal.
            raise ToolCallParseError(
                f"{tool_name}: parameter {key!r} must be null, got "
                f"{value[:80]!r}")
        return value
    typed = [m for m in members if m in _JSON_TYPES and m != "string"]
    if typed:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = _UNDECODABLE
        if decoded is not _UNDECODABLE:
            for member in typed:
                if isinstance(decoded, bool) and member != "boolean":
                    continue
                if isinstance(decoded, _JSON_TYPES[member]):
                    return decoded
    if accepts_raw_text(members):
        return value
    label = " or ".join(members)
    raise ToolCallParseError(
        f"{tool_name}: parameter {key!r} must be {label}, got "
        f"unmatching value {value[:80]!r}")
