"""Tool-call dialects: types, strict parsers, serializers, and repair.

A serving stack and an agent client meet at the same contract: the request
carries OpenAI-style ``tools``, the model emits an invocation in a text
dialect, and something converts that text back into OpenAI ``tool_calls``.
This package owns everything dialect-shaped on both sides of that contract,
so the serve layer (``runtime``) and the agent library (``agentlib``) share
one grammar, one strict parser, and one repair seam per dialect.

- ``types``: the ``ToolCall`` value, the ``ToolCallParseError`` seam, and the
  strict parser for native OpenAI ``tool_calls`` message entries.
- ``qwenxml``: the Qwen XML text dialect (``<tool_call>`` blocks with
  ``<function=...>``/``<parameter=...>`` elements).
- ``dsml``: the DeepSeek DSML text dialect, both directions: the grammar
  constants, the tools instruction block, serializers from OpenAI-format
  calls, and the strict parser.
- ``envelope``: the Terminus-2-style JSON action envelope.
- ``repair``: bounded text transformations that salvage almost-valid
  completions behind the strict parsers, plus ``RepairTelemetry``.

Every module here is pure (stdlib only) and importable without mlx or any
runtime dependency.
"""

from moespresso.toolcalls.dsml import (
    DSML_TOKEN,
    has_tool_call_block,
    parse_dsml_tool_calls,
    render_dsml_tool_calls,
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
from moespresso.toolcalls.repair import (
    RepairTelemetry,
    repair_action_envelope,
    repair_dsml_tool_calls,
    repair_qwenxml_tool_calls,
)
from moespresso.toolcalls.types import (
    ToolCall,
    ToolCallParseError,
    parse_tool_calls,
)

__all__ = [
    "ToolCall",
    "ToolCallParseError",
    "parse_tool_calls",
    "has_qwenxml_tool_call",
    "parse_qwenxml_tool_calls",
    "strip_qwenxml_blocks",
    "DSML_TOKEN",
    "has_tool_call_block",
    "parse_dsml_tool_calls",
    "render_dsml_tool_calls",
    "render_tools",
    "ActionEnvelope",
    "envelope_system_block",
    "parse_action_envelope",
    "RepairTelemetry",
    "repair_action_envelope",
    "repair_dsml_tool_calls",
    "repair_qwenxml_tool_calls",
]
