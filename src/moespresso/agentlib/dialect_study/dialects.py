"""The three dialect arms as one adapter interface.

A dialect owns everything arm-specific about a turn: the system prompt (with
or without embedded tool schemas), whether request-level ``tools`` are sent,
how to detect a tool-call attempt in completion text, strict parsing, repair,
the answer text channel, and how tool results flow back into the
conversation. The episode loop in ``run`` is dialect-blind.

Result routing differs by dialect on purpose. The native and DSML arms
return results as ``role: "tool"`` messages, which the vendored template
wraps in ``<tool_response>`` blocks; the envelope arm receives results as a
plain user message, matching the Terminus-2 convention where everything is
text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from moespresso.agentlib.conversation import Conversation
from moespresso.agentlib.execution import ToolResult
from moespresso.agentlib.tools import ToolRegistry
from moespresso.toolcalls import repair
from moespresso.toolcalls.dsml import (
    DSML_TOKEN,
    TOOL_CALLS_CLOSE,
    TOOL_CALLS_OPEN,
    parse_dsml_tool_calls,
    render_tools,
)
from moespresso.toolcalls.envelope import (
    ActionEnvelope,
    envelope_system_block,
    parse_action_envelope,
)
from moespresso.toolcalls.qwenxml import (
    has_qwenxml_tool_call,
    parse_qwenxml_tool_calls,
    strip_qwenxml_blocks,
)
from moespresso.toolcalls.types import ToolCall, ToolCallParseError

BASE_SYSTEM = (
    "You are a coding agent working inside a small telemetry-processing "
    "project. Use the available tools to inspect and modify the project. "
    "Work in short steps: call a tool when you need facts from the "
    "workspace, and finish with a short factual answer of one or two "
    "sentences."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DSML_BLOCK_RE = re.compile(
    re.escape(TOOL_CALLS_OPEN) + r".*?" + re.escape(TOOL_CALLS_CLOSE), re.DOTALL
)


@dataclass(frozen=True)
class DialectTurn:
    """One parsed assistant turn: the calls to run, or the final answer."""

    calls: tuple[ToolCall, ...]
    final: bool
    answer_text: str


def _strip_think(content: str) -> str:
    return _THINK_RE.sub("", content).strip()


def parameter_schemas(registry: ToolRegistry) -> dict[str, dict]:
    """Tool name to JSON schema map, for typed text-value decoding."""
    return {name: registry.spec(name).parameters for name in registry.names()}


class NativeDialect:
    """Arm A: request-level tools, template-rendered XML tool calls."""

    name = "native"

    def tools_block(self, registry: ToolRegistry) -> str | None:
        # The template injects the tool instructions from the request's tools
        # array; the system prompt carries only the task framing.
        return None

    def system_prompt(self, registry: ToolRegistry) -> str:
        return BASE_SYSTEM

    def request_tools(self, registry: ToolRegistry) -> list[dict] | None:
        return registry.openai_tools()

    def attempted(self, content: str) -> bool:
        return has_qwenxml_tool_call(content)

    def parse_turn(self, content: str, registry: ToolRegistry) -> DialectTurn:
        calls = parse_qwenxml_tool_calls(content, parameter_schemas(registry))
        return self._shape(content, calls)

    def repair_turn(self, content: str, registry: ToolRegistry) -> DialectTurn:
        calls = repair.repair_qwenxml_tool_calls(
            content, parameter_schemas(registry))
        return self._shape(content, calls)

    def _shape(self, content: str, calls: list[ToolCall]) -> DialectTurn:
        answer = _strip_think(strip_qwenxml_blocks(content))
        return DialectTurn(calls=tuple(calls), final=not calls, answer_text=answer)

    def append_results(self, conversation: Conversation,
                       outcomes: list[tuple[ToolCall, ToolResult]]) -> None:
        for _call, result in outcomes:
            conversation.add_tool_result(result.output)

    def parse_feedback(self, error: Exception) -> str:
        return (
            f"Your tool call could not be parsed: {error}. Reply again using "
            "exactly the documented format: a <tool_call> block containing "
            "one <function=...> element with <parameter=...> values."
        )


class EnvelopeDialect:
    """Arm B: the Terminus-2-style JSON action envelope in plain text."""

    name = "envelope"

    def tools_block(self, registry: ToolRegistry) -> str | None:
        return envelope_system_block(registry.openai_tools())

    def system_prompt(self, registry: ToolRegistry) -> str:
        return BASE_SYSTEM + "\n\n" + self.tools_block(registry)

    def request_tools(self, registry: ToolRegistry) -> list[dict] | None:
        return None

    def attempted(self, content: str) -> bool:
        # Every envelope turn is an attempt: the dialect demands a JSON
        # envelope on every reply, so prose is a parse failure, never a
        # final answer.
        return True

    def parse_turn(self, content: str, registry: ToolRegistry) -> DialectTurn:
        return self._shape(parse_action_envelope(content))

    def repair_turn(self, content: str, registry: ToolRegistry) -> DialectTurn:
        return self._shape(repair.repair_action_envelope(content))

    def _shape(self, env: ActionEnvelope) -> DialectTurn:
        return DialectTurn(
            calls=env.calls,
            final=env.task_complete or not env.calls,
            answer_text=env.analysis.strip(),
        )

    def append_results(self, conversation: Conversation,
                       outcomes: list[tuple[ToolCall, ToolResult]]) -> None:
        parts = []
        for index, (call, result) in enumerate(outcomes, start=1):
            status = "ok" if result.ok else "error"
            parts.append(f"[{index}] {call.name} -> {status}\n{result.output}")
        conversation.add_user("Tool results:\n\n" + "\n\n".join(parts))

    def parse_feedback(self, error: Exception) -> str:
        return (
            f"Your reply was not a valid action envelope: {error}. Reply "
            "with a single JSON object carrying exactly the keys analysis, "
            "plan, commands, task_complete and nothing else."
        )


class DsmlDialect:
    """Arm C: DeepSeek DSML text markers taught through the system prompt."""

    name = "dsml"

    def tools_block(self, registry: ToolRegistry) -> str | None:
        functions = [tool["function"] for tool in registry.openai_tools()]
        return render_tools(functions)

    def system_prompt(self, registry: ToolRegistry) -> str:
        return BASE_SYSTEM + "\n\n" + self.tools_block(registry)

    def request_tools(self, registry: ToolRegistry) -> list[dict] | None:
        return None

    def attempted(self, content: str) -> bool:
        return TOOL_CALLS_OPEN in content or f"<{DSML_TOKEN}invoke" in content

    def parse_turn(self, content: str, registry: ToolRegistry) -> DialectTurn:
        calls = parse_dsml_tool_calls(content)
        if not calls and f"<{DSML_TOKEN}invoke" in content:
            # The shared parser returns empty without the block marker; an
            # invoke outside a block is an attempt and must fail as one.
            raise ToolCallParseError("invoke element outside a tool_calls block")
        return self._shape(content, calls)

    def repair_turn(self, content: str, registry: ToolRegistry) -> DialectTurn:
        return self._shape(content, repair.repair_dsml_tool_calls(content))

    def _shape(self, content: str, calls: list[ToolCall]) -> DialectTurn:
        answer = _strip_think(_DSML_BLOCK_RE.sub("", content))
        return DialectTurn(calls=tuple(calls), final=not calls, answer_text=answer)

    def append_results(self, conversation: Conversation,
                       outcomes: list[tuple[ToolCall, ToolResult]]) -> None:
        for _call, result in outcomes:
            conversation.add_tool_result(result.output)

    def parse_feedback(self, error: Exception) -> str:
        return (
            f"Your tool call could not be parsed: {error}. Reply again using "
            f"exactly the documented format: a <{DSML_TOKEN}tool_calls> "
            f"block containing <{DSML_TOKEN}invoke> elements."
        )


_DIALECTS = {
    NativeDialect.name: NativeDialect,
    EnvelopeDialect.name: EnvelopeDialect,
    DsmlDialect.name: DsmlDialect,
}

ARM_NAMES = tuple(_DIALECTS)


def dialect_for(arm: str):
    """The dialect adapter for an arm name."""
    if arm not in _DIALECTS:
        known = ", ".join(_DIALECTS)
        raise ValueError(f"unknown dialect arm {arm!r} (known: {known})")
    return _DIALECTS[arm]()
