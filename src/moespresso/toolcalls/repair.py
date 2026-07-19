"""Tool-call repair: salvage almost-valid completions behind the strict parsers.

Every dialect parser in this package raises ``ToolCallParseError`` instead of
guessing. This module is the component that catches exactly that seam: given
the raw completion text that failed strict parsing, it applies a bounded
sequence of text transformations (code-fence removal, truncation closing,
scavenging the call out of surrounding prose, JSON damage fixes, typed-value
coercion) and accepts the first transformed candidate the strict parser
takes. Anything still unparseable raises ``ToolCallParseError`` again, so a
caller sees either a valid call list or the same clean typed failure.

Repair targets small models specifically: the transformations encode the
malformed shapes such models actually emit, and each one is unit-tested
offline against collected outputs. Repair never invents a call: a candidate
that transforms into no call at all is a failure, because the caller only
reaches this module after the content already looked like an attempt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from moespresso.toolcalls import dsml, qwenxml
from moespresso.toolcalls.dsml import DSML_TOKEN
from moespresso.toolcalls.envelope import (
    ENVELOPE_KEYS,
    ActionEnvelope,
    envelope_from_object,
)
from moespresso.toolcalls.types import ToolCall, ToolCallParseError

_FENCE_LINE_RE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*$\n?", re.MULTILINE)
_FENCED_SEGMENT_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n(.*?)```", re.DOTALL)


@dataclass
class RepairTelemetry:
    """Counters for repair activity over one scope (episode, session, arm).

    ``fires`` counts strict parse failures handed to the repair layer,
    ``salvaged`` the fires repair turned into a valid call set, ``failed``
    the fires that stayed typed failures. A consumer keeps one instance per
    scope, records through ``record``, and aggregates with ``add``. Adopting
    a repair-dependent dialect is conditioned on watching these counters, so
    they ship here beside the repair functions for any client to reuse.
    """

    fires: int = 0
    salvaged: int = 0
    failed: int = 0

    def record(self, salvaged: bool) -> None:
        self.fires += 1
        if salvaged:
            self.salvaged += 1
        else:
            self.failed += 1

    def add(self, other: "RepairTelemetry") -> None:
        self.fires += other.fires
        self.salvaged += other.salvaged
        self.failed += other.failed

    def as_dict(self) -> dict:
        return {"fires": self.fires, "salvaged": self.salvaged,
                "failed": self.failed}


def _drop_fence_lines(text: str) -> str:
    return _FENCE_LINE_RE.sub("", text)


# --- Qwen XML dialect --------------------------------------------------------

_NAMED_CLOSER_RE = re.compile(r"</(function|parameter)=[^>\n]*>")
_NAKED_FUNCTION_RE = re.compile(r"<function=[^>\n]*>.*?</function>", re.DOTALL)


def _close_dangling(text: str, pairs: list[tuple[str, str]]) -> str:
    """Append missing closers for innermost-first tag pairs at end of text."""
    for open_marker, close_marker in pairs:
        while text.count(open_marker) > text.count(close_marker):
            text = text + "\n" + close_marker
    return text


def _wrap_naked_functions(text: str) -> str:
    if qwenxml.TOOL_CALL_OPEN in text:
        return text
    return _NAKED_FUNCTION_RE.sub(
        lambda m: f"{qwenxml.TOOL_CALL_OPEN}\n{m.group(0)}\n{qwenxml.TOOL_CALL_CLOSE}",
        text,
    )


def _coerce_qwenxml_values(text: str, parameter_schemas: dict[str, dict]) -> str:
    """Rewrite undecodable typed parameter values into decodable JSON text.

    Walks each function element with the strict grammar's own regexes so the
    repair pass and the parser agree on structure, and rewrites only values
    whose declared type is non-string and whose text fails to decode.
    """
    def fix_function(match: re.Match) -> str:
        name = match.group(1)
        properties = (parameter_schemas.get(name) or {}).get("properties") or {}

        def fix_parameter(pmatch: re.Match) -> str:
            key, raw = pmatch.group(1), pmatch.group(2)
            declared = (properties.get(key) or {}).get("type")
            if declared in (None, "string"):
                return pmatch.group(0)
            fixed = _lenient_scalar(raw.strip(), declared)
            if fixed is None:
                return pmatch.group(0)
            return f"<parameter={key}>\n{fixed}\n</parameter>"

        inner = qwenxml._PARAM_RE.sub(fix_parameter, match.group(2))
        return f"<function={name}>{inner}</function>"

    return qwenxml._FUNCTION_RE.sub(fix_function, text)


def _lenient_scalar(value: str, declared: str) -> str | None:
    """A JSON text for ``value`` matching the declared type, or None."""
    candidates = [value]
    unquoted = value
    for quote in ("'", '"'):
        if len(unquoted) >= 2 and unquoted[0] == quote and unquoted[-1] == quote:
            unquoted = unquoted[1:-1]
    if unquoted != value:
        candidates.append(unquoted)
    for candidate in list(candidates):
        lowered = candidate.strip().lower()
        if declared == "boolean" and lowered in ("true", "false", "yes", "no",
                                                 "on", "off"):
            candidates.append(
                "true" if lowered in ("true", "yes", "on") else "false")
        if declared in ("integer", "number"):
            digits = candidate.replace(",", "")
            if digits != candidate:
                candidates.append(digits)
            if declared == "integer" and re.fullmatch(r"-?\d+\.0*", digits):
                candidates.append(digits.split(".")[0])
    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        expected = qwenxml._JSON_TYPES.get(declared)
        if expected is None:
            return None
        if isinstance(decoded, bool) and declared != "boolean":
            continue
        if isinstance(decoded, expected):
            return json.dumps(decoded)
    return None


def repair_qwenxml_tool_calls(
    content: str,
    parameter_schemas: dict[str, dict] | None = None,
) -> list[ToolCall]:
    """Salvage a malformed Qwen XML tool-call completion into calls."""
    schemas = parameter_schemas or {}
    last_error: Exception | None = None
    text = content
    transformations = (
        _drop_fence_lines,
        lambda t: _NAMED_CLOSER_RE.sub(lambda m: f"</{m.group(1)}>", t),
        _wrap_naked_functions,
        lambda t: _close_dangling(t, [
            ("<parameter=", "</parameter>"),
            ("<function=", "</function>"),
            (qwenxml.TOOL_CALL_OPEN, qwenxml.TOOL_CALL_CLOSE),
        ]),
        lambda t: _coerce_qwenxml_values(t, schemas),
    )
    for transform in transformations:
        text = transform(text)
        try:
            calls = qwenxml.parse_qwenxml_tool_calls(text, schemas)
        except ToolCallParseError as e:
            last_error = e
            continue
        if calls:
            return calls
    raise ToolCallParseError(f"unrepairable tool-call text: {last_error}")


# --- action envelope dialect -------------------------------------------------

# Alternate key spellings observed in envelope-style agents; the canonical
# Terminus-2 keys win when both are present.
_ENVELOPE_KEY_ALIASES = {
    "state_analysis": "analysis",
    "explanation": "analysis",
    "next_steps": "plan",
    "actions": "commands",
    "tool_calls": "commands",
    "is_task_complete": "task_complete",
    "done": "task_complete",
}
_COMMAND_KEY_ALIASES = {
    "name": "tool",
    "function": "tool",
    "arguments": "args",
    "parameters": "args",
}


def _scavenge_json_object(text: str) -> str | None:
    """The first balanced top-level JSON object in the text, string-aware."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _close_truncated_json(text: str) -> str:
    """Close a JSON object cut off mid-stream: quote, then brackets, in order."""
    stack = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_string:
        text = text + '"'
    text = re.sub(r",\s*$", "", text)
    return text + "".join(reversed(stack))


def _rebalance_closers(text: str) -> str:
    """Insert missing closers where a closer arrives for an outer scope.

    Observed as a command entry losing its closing brace before the commands
    array closes. The scan is string-aware; a closer that does not match the
    innermost open scope first closes the scopes the text skipped.
    """
    out = []
    stack = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            while stack and stack[-1] != ch:
                out.append(stack.pop())
            if stack:
                stack.pop()
        out.append(ch)
    return "".join(out)


def _json_damage_candidates(text: str):
    yield text
    yield re.sub(r",\s*([}\]])", r"\1", text)
    yield re.sub(r"\bTrue\b", "true",
                 re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null", text)))
    yield text.replace("'", '"')
    yield _rebalance_closers(text)
    yield _close_truncated_json(text)


def _lenient_envelope_object(decoded: dict) -> dict:
    """Canonicalize aliases and drop unknown keys before the strict shaping.

    Canonically named keys win; an alias fills a key only when the canonical
    spelling is absent. One action key must already be present for the other
    to be defaulted: an object carrying neither commands nor task_complete is
    not an envelope, and defaulting both would invent a completed turn out of
    unrelated JSON.
    """
    shaped = {key: value for key, value in decoded.items() if key in ENVELOPE_KEYS}
    for key, value in decoded.items():
        canonical = _ENVELOPE_KEY_ALIASES.get(key)
        if canonical is not None and canonical not in shaped:
            shaped[canonical] = value
    if "commands" not in shaped and "task_complete" not in shaped:
        return shaped
    task_complete = shaped.get("task_complete")
    if isinstance(task_complete, str):
        shaped["task_complete"] = task_complete.strip().lower() == "true"
    commands = shaped.get("commands")
    if commands is None:
        commands = []
    if isinstance(commands, list):
        shaped["commands"] = [_lenient_command(entry) for entry in commands]
        if "task_complete" not in shaped:
            shaped["task_complete"] = not shaped["commands"]
    return shaped


# A reply that is one bare command object instead of an envelope; observed as
# the model opening its native <tool_call> marker and then emitting JSON.
_BARE_COMMAND_KEYS = frozenset(
    {"tool", "args"} | set(_COMMAND_KEY_ALIASES)
)


def _wrap_bare_command(decoded: dict) -> dict | None:
    if not decoded or not set(decoded) <= _BARE_COMMAND_KEYS:
        return None
    entry = _lenient_command(decoded)
    if not isinstance(entry.get("tool"), str) or not entry.get("tool"):
        return None
    return {"commands": [entry], "task_complete": False}


def _lenient_command(entry):
    if not isinstance(entry, dict):
        return entry
    shaped = {}
    for key, value in entry.items():
        canonical = _COMMAND_KEY_ALIASES.get(key, key)
        if canonical in ("tool", "args"):
            shaped.setdefault(canonical, value)
    args = shaped.get("args")
    if isinstance(args, str):
        try:
            decoded = json.loads(args)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            shaped["args"] = decoded
    return shaped


def repair_action_envelope(content: str) -> ActionEnvelope:
    """Salvage a malformed action-envelope completion into an envelope."""
    texts = []
    fenced = _FENCED_SEGMENT_RE.search(content)
    if fenced:
        texts.append(fenced.group(1))
    texts.append(_drop_fence_lines(content))
    scavenged = [_scavenge_json_object(text) for text in texts]
    texts.extend(s for s in scavenged if s)
    last_error: Exception | None = None
    for text in texts:
        for candidate in _json_damage_candidates(text.strip()):
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError as e:
                last_error = e
                continue
            if not isinstance(decoded, dict):
                continue
            try:
                return envelope_from_object(_lenient_envelope_object(decoded))
            except ToolCallParseError as e:
                last_error = e
            wrapped = _wrap_bare_command(decoded)
            if wrapped is not None:
                try:
                    return envelope_from_object(wrapped)
                except ToolCallParseError as e:
                    last_error = e
    raise ToolCallParseError(f"unrepairable action envelope: {last_error}")


# --- DSML dialect -------------------------------------------------------------

_DSML_PARAM_NO_STRING_RE = re.compile(
    rf'(<{re.escape(DSML_TOKEN)}parameter\s+name="[^"]*")\s*>'
)
# Wrong quoting observed in served completions: the closing quote of the name
# attribute is dropped, fusing it into the string attribute
# (name="path string="true"). Restore the quote before the attribute split.
_DSML_UNCLOSED_NAME_RE = re.compile(
    rf'(<{re.escape(DSML_TOKEN)}parameter\s+name="[^"<>]*?)(\s+string=")'
)
_DSML_NAKED_INVOKE_RE = re.compile(
    rf'<{re.escape(DSML_TOKEN)}invoke\s+name="[^"]*">.*?'
    rf"</{re.escape(DSML_TOKEN)}invoke>",
    re.DOTALL,
)


def repair_dsml_tool_calls(content: str) -> list[ToolCall]:
    """Salvage a malformed DSML tool-call completion into calls."""
    last_error: Exception | None = None
    text = content
    transformations = (
        _drop_fence_lines,
        lambda t: _DSML_UNCLOSED_NAME_RE.sub(r'\1"\2', t),
        lambda t: _DSML_PARAM_NO_STRING_RE.sub(r'\1 string="true">', t),
        _wrap_naked_dsml_invokes,
        lambda t: _close_dangling(t, [
            (f"<{DSML_TOKEN}parameter", f"</{DSML_TOKEN}parameter>"),
            (f"<{DSML_TOKEN}invoke", f"</{DSML_TOKEN}invoke>"),
            (dsml.TOOL_CALLS_OPEN, dsml.TOOL_CALLS_CLOSE),
        ]),
    )
    for transform in transformations:
        text = transform(text)
        try:
            calls = dsml.parse_dsml_tool_calls(text)
        except ToolCallParseError as e:
            last_error = e
            continue
        if calls:
            return calls
    raise ToolCallParseError(f"unrepairable DSML tool-call text: {last_error}")


def _wrap_naked_dsml_invokes(text: str) -> str:
    if dsml.TOOL_CALLS_OPEN in text:
        return text
    return _DSML_NAKED_INVOKE_RE.sub(
        lambda m: f"{dsml.TOOL_CALLS_OPEN}\n{m.group(0)}\n{dsml.TOOL_CALLS_CLOSE}",
        text,
    )
