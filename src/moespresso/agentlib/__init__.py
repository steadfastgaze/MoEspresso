"""Agent-side library for driving a MoEspresso server.

agentlib is engine test infrastructure. The road-test drives a served model
through this library and asserts engine counters turn over turn, and it doubles
as the reference client for anything agentic on MoEspresso. It provides:

- ``Conversation``: append-only message history plus the session cache key,
  so every request extends the previous render as a token prefix.
- ``CompletionsClient``: one blocking chat-completions call per turn against
  the local server, carrying ``metadata.moespresso_cache_key`` and surfacing
  ``usage.prompt_cache`` for reuse assertions.
- ``ToolRegistry`` and the four core tools (read_file, grep, edit, bash),
  executed only through ``execute_tool_call``, the single choke point that
  carries the sandbox policy gate.
- ``SandboxPolicy`` and the policy engine in ``sandbox``: ordered regex and
  executable-hook rules decide whether a bash command runs unsandboxed, runs
  under a generated Seatbelt profile, or is refused, and the non-bash tools
  get path-scope checks against the writable scope.
- ``parse_tool_calls``: the strict native parser for OpenAI ``tool_calls``
  in a response message. Tool-call repair and text-envelope dialects are
  separate, later components that build on this seam.
- ``parse_dsml_tool_calls``: the strict parser for the DeepSeek DSML text
  dialect, for model families whose serve layer returns tool invocations
  inside the completion content.
- ``SubagentRunner`` and ``SubagentBrief``: the sequential subagent
  skeleton. A child runs a brief to completion in a forked conversation
  under a session cache key derived from the parent's, and its bounded
  result folds back into the parent as a tool result.
"""

from moespresso.agentlib.client import (
    ChatCompletion,
    ClientError,
    CompletionsClient,
)
from moespresso.agentlib.conversation import Conversation
from moespresso.agentlib.dsml import has_tool_call_block, parse_dsml_tool_calls
from moespresso.agentlib.execution import ToolResult, execute_tool_call
from moespresso.agentlib.sandbox import (
    Decision,
    HookRule,
    PolicyConfigError,
    PolicyDecision,
    RegexRule,
    SandboxPolicy,
    build_policy,
    default_config_path,
    evaluate_command,
    generate_profile,
    load_policy,
    path_scope_problem,
    sandboxed_command,
)
from moespresso.agentlib.subagent import (
    MAX_SUBAGENT_DEPTH,
    SubagentBrief,
    SubagentConcurrencyError,
    SubagentConfigError,
    SubagentDepthError,
    SubagentError,
    SubagentResult,
    SubagentRunner,
    child_session_key,
)
from moespresso.agentlib.toolcalls import (
    ToolCall,
    ToolCallParseError,
    parse_tool_calls,
)
from moespresso.agentlib.tools import (
    CORE_TOOL_SPECS,
    ToolRegistry,
    ToolSpec,
    build_core_registry,
)

__all__ = [
    "ChatCompletion",
    "ClientError",
    "CompletionsClient",
    "Conversation",
    "ToolResult",
    "execute_tool_call",
    "Decision",
    "HookRule",
    "PolicyConfigError",
    "PolicyDecision",
    "RegexRule",
    "SandboxPolicy",
    "build_policy",
    "default_config_path",
    "evaluate_command",
    "generate_profile",
    "load_policy",
    "path_scope_problem",
    "sandboxed_command",
    "MAX_SUBAGENT_DEPTH",
    "SubagentBrief",
    "SubagentConcurrencyError",
    "SubagentConfigError",
    "SubagentDepthError",
    "SubagentError",
    "SubagentResult",
    "SubagentRunner",
    "child_session_key",
    "ToolCall",
    "ToolCallParseError",
    "parse_tool_calls",
    "has_tool_call_block",
    "parse_dsml_tool_calls",
    "CORE_TOOL_SPECS",
    "ToolRegistry",
    "ToolSpec",
    "build_core_registry",
]
