"""The single execution choke point for tool calls.

Every tool invocation in agentlib flows through ``execute_tool_call``. The
sandbox policy gate lives inside this function, so no tool can be reached
around it: with a ``SandboxPolicy`` supplied, a bash command is evaluated
against the policy rules and runs unsandboxed, runs wrapped in the generated
sandbox-exec profile, or comes back refused; the path arguments of the other
core tools are checked against the writable scope. An ask decision resolves
to a sandboxed run here, because this executor is headless and a sandboxed
run is the default posture; a frontend with an approval prompt resolves ask
itself before reaching this layer. Sandboxed execution needs macOS
sandbox-exec; on any other platform a sandbox-requiring decision is refused.
Tools the gate has no rule class for are refused while a policy is active,
so adding a tool forces a deliberate policy mapping. With ``policy=None``
every call runs ungated.

Failures do not raise. An agent loop feeds errors back to the model as tool
results, so an unknown tool, bad arguments, a policy refusal, or an executor
failure all return ``ToolResult(ok=False, ...)`` with a message the model
can act on.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from moespresso.agentlib import sandbox
from moespresso.agentlib.sandbox import Decision, SandboxPolicy
from moespresso.toolcalls.types import ToolCall
from moespresso.agentlib.tools import ToolRegistry


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call, as text for the tool-role message."""

    ok: bool
    output: str


def execute_tool_call(registry: ToolRegistry, call: ToolCall, *,
                      workdir: Path,
                      policy: SandboxPolicy | None = None) -> ToolResult:
    """Validate, gate against the sandbox policy, and run one tool call."""
    try:
        spec = registry.spec(call.name)
    except KeyError:
        known = ", ".join(registry.names())
        return ToolResult(False, f"error: unknown tool {call.name!r} "
                                 f"(available: {known})")
    problem = _validate_arguments(spec.parameters, call.arguments)
    if problem is not None:
        return ToolResult(False, f"error: {call.name}: {problem}")
    arguments = dict(call.arguments)
    if policy is not None:
        gated = _apply_policy(policy, call.name, arguments, workdir)
        if isinstance(gated, ToolResult):
            return gated
        arguments = gated
    executor = registry.executor(call.name)
    try:
        output = executor(arguments, workdir)
    except (ValueError, TypeError, OSError) as e:
        # Arguments come from an untrusted model: a wrong-typed value surfaces
        # as TypeError inside an executor and is a tool-level failure too.
        return ToolResult(False, f"error: {call.name}: {e}")
    return ToolResult(True, output)


# The path argument each non-bash core tool exposes; the gate confines it to
# the policy's writable scope. grep defaults to the working directory.
_PATH_ARGUMENTS = {"read_file": "path", "grep": "path", "edit": "path"}


def _apply_policy(policy: SandboxPolicy, name: str, arguments: dict,
                  workdir: Path) -> dict | ToolResult:
    """Gate one validated call. Returns the arguments to execute (rewritten
    for a sandboxed bash run) or a refusal. Policy outcomes never raise."""
    if name == "bash":
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return arguments  # the executor rejects malformed commands itself
        verdict = sandbox.evaluate_command(policy, command, cwd=workdir)
        decision = verdict.decision
        if decision is Decision.DENY:
            detail = (f"{verdict.source}: {verdict.reason}" if verdict.reason
                      else verdict.source)
            return ToolResult(
                False, f"error: bash: refused by sandbox policy ({detail})")
        if decision is Decision.ASK:
            # Headless resolution: ask runs sandboxed, the default posture.
            decision = Decision.ALLOW_SANDBOXED
        if decision is Decision.ALLOW_SANDBOXED:
            if sys.platform != "darwin":
                return ToolResult(
                    False, "error: bash: sandboxed execution requires macOS "
                           "sandbox-exec; the policy fails closed elsewhere")
            profile = sandbox.generate_profile(policy)
            rewritten = dict(arguments)
            rewritten["command"] = sandbox.sandboxed_command(command, profile)
            return rewritten
        return arguments  # allow-unsandboxed
    path_key = _PATH_ARGUMENTS.get(name)
    if path_key is not None:
        raw = arguments.get(path_key, ".")
        if isinstance(raw, str):
            problem = sandbox.path_scope_problem(policy, raw, workdir)
            if problem is not None:
                return ToolResult(False, f"error: {name}: {problem}")
        return arguments
    return ToolResult(
        False, f"error: {name}: the sandbox policy has no rule class for "
               "this tool and refuses it")


def _validate_arguments(schema: dict, arguments: dict) -> str | None:
    """Check required and unknown argument names against the spec schema.

    Strict on names so a misspelled argument comes back as a correctable
    message instead of being silently ignored. Value types are left to the
    executors.
    """
    required = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    allowed = set(schema.get("properties", {}))
    unknown = [name for name in arguments if name not in allowed]
    if unknown:
        return (f"unknown argument(s): {', '.join(sorted(unknown))} "
                f"(allowed: {', '.join(sorted(allowed))})")
    return None
