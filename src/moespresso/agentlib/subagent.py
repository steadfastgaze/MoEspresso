"""Sequential subagent skeleton: a forked conversation folded back as a tool result.

A subagent runs a self-contained task in its own conversation under its own
session cache key, against the same client and tool registry as the parent.
The parent session is idle while the child runs; when the child finishes, a
bounded result payload is appended to the parent through
``Conversation.add_tool_result``, so the parent's next request is still a
strict prefix extension of its previous one. Subagents are sequential: the
runner is the only spawn path and it refuses to open a second child while one
is open, which also keeps the client one-in-flight against an engine that
serves one request at a time.

Where the design leaves details open, this module fixes them as follows:

- The child session cache key derives from the parent key: the parent key
  plus ``/sub<n>``, where ``n`` is the runner's spawn ordinal. Derivation is
  deterministic, every spawn (including a failed one) consumes an ordinal,
  and ordinals never repeat within a runner, so repeated spawns never
  collide. The shared prefix keeps a session family together for the disk
  cache's eviction grouping; the key never enters the engine's restore
  safety key.
- The child starts from an explicit ``SubagentBrief`` (system prompt, task
  message, optional context block). The parent history does not cross the
  boundary; what crosses is exactly the brief fields.
- The fold-back payload is bounded by ``max_result_chars`` and is the same
  string in both the returned ``SubagentResult`` and the appended tool
  message.
- Nesting depth is limited to one level: a runner attached to a depth-0
  conversation may spawn depth-1 children, and a runner attached at depth 1
  fails closed at construction with ``SubagentDepthError``.
- Child failures (turn cap, tool-call parse failure, transport failure, a
  malformed assistant message) come back as an ``ok=False`` result folded
  into the parent. The parent loop never sees them as exceptions; the typed
  errors in this module cover caller mistakes (miswiring, a concurrent or
  too-deep spawn), which raise before a child opens.

The ``parse`` parameter is the dialect seam: it receives the assistant
message dict and returns the tool calls to execute. The default is the
strict native parser; a text-dialect caller wraps its content parser, for
example ``lambda message: parse_dsml_tool_calls(message.get("content"))``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from moespresso.agentlib.client import ClientError, CompletionsClient
from moespresso.agentlib.conversation import Conversation
from moespresso.agentlib.execution import execute_tool_call
from moespresso.agentlib.sandbox import SandboxPolicy
from moespresso.agentlib.toolcalls import (
    ToolCall,
    ToolCallParseError,
    parse_tool_calls,
)
from moespresso.agentlib.tools import ToolRegistry

# Children nest at most this many levels below the root conversation.
MAX_SUBAGENT_DEPTH = 1
# Cap on child requests before the run fails with reason "max_turns".
DEFAULT_MAX_CHILD_TURNS = 24
# Cap on the fold-back payload length in characters.
DEFAULT_MAX_RESULT_CHARS = 8000

# The dialect seam: assistant message dict in, tool calls out.
MessageParser = Callable[[dict], list[ToolCall]]


class SubagentError(Exception):
    """Base for the typed spawn errors raised to the caller."""


class SubagentConfigError(SubagentError):
    """The runner or brief is miswired: bad limits, missing parent key."""


class SubagentDepthError(SubagentError):
    """A spawn would nest children deeper than MAX_SUBAGENT_DEPTH."""


class SubagentConcurrencyError(SubagentError):
    """A second child spawn while one is open. Subagents run sequentially."""


def child_session_key(parent_key: str, ordinal: int) -> str:
    """Derive a child session cache key from the parent key and spawn ordinal."""
    if not isinstance(parent_key, str) or not parent_key:
        raise SubagentConfigError(
            "parent session cache key must be a non-empty string")
    if ordinal < 1:
        raise SubagentConfigError("spawn ordinal must be >= 1")
    return f"{parent_key}/sub{ordinal}"


@dataclass(frozen=True)
class SubagentBrief:
    """The context that crosses the parent-child boundary.

    ``task`` becomes the child's first user message. ``system`` is the
    child's system prompt. ``context`` is material the parent chooses to
    share (file excerpts, constraints, prior findings); it is appended to
    the task message after a blank line.
    """

    task: str
    system: str | None = None
    context: str | None = None

    def opening_message(self) -> str:
        if self.context is None:
            return self.task
        return f"{self.task}\n\n{self.context}"


@dataclass(frozen=True)
class SubagentResult:
    """The outcome of one child run.

    ``text`` is exactly the payload folded into the parent as the tool
    result. ``reason`` is ``completed`` on success, or the failure mode:
    ``max_turns``, ``parse_failure``, ``client_error``, ``invalid_message``.
    """

    ok: bool
    text: str
    reason: str
    session_cache_key: str
    turns: int
    truncated: bool = False


class SubagentRunner:
    """Spawns sequential subagents for one parent conversation.

    The runner holds the shared client, tool registry, working directory,
    and sandbox policy, and is the only spawn path. Construction fails
    closed on a parent without a session cache key (child keys derive from
    it) and on a depth at or past MAX_SUBAGENT_DEPTH (a child cannot get a
    runner of its own).
    """

    def __init__(
        self,
        parent: Conversation,
        client: CompletionsClient,
        registry: ToolRegistry,
        *,
        workdir: Path,
        policy: SandboxPolicy | None = None,
        parse: MessageParser = parse_tool_calls,
        depth: int = 0,
        max_child_turns: int = DEFAULT_MAX_CHILD_TURNS,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        if depth < 0:
            raise SubagentConfigError("depth must be a non-negative integer")
        if depth >= MAX_SUBAGENT_DEPTH:
            raise SubagentDepthError(
                f"a conversation at depth {depth} cannot spawn a child; "
                f"children nest at most {MAX_SUBAGENT_DEPTH} level(s) below "
                f"the root")
        key = parent.session_cache_key
        if not isinstance(key, str) or not key:
            raise SubagentConfigError(
                "the parent conversation carries no session cache key; child "
                "keys derive from it, so a keyed parent is required")
        if max_child_turns < 1:
            raise SubagentConfigError("max_child_turns must be >= 1")
        if max_result_chars < 1:
            raise SubagentConfigError("max_result_chars must be >= 1")
        self.parent = parent
        self.client = client
        self.registry = registry
        self.workdir = workdir
        self.policy = policy
        self.parse = parse
        self.depth = depth
        self.max_child_turns = max_child_turns
        self.max_result_chars = max_result_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        # Rendered once and held constant across all children: the tools
        # array renders into the shared prompt prefix, and a stable prefix
        # is what the session-family cache grouping is for.
        self._tools = registry.openai_tools()
        self._spawned = 0
        self._open = False

    def run(self, brief: SubagentBrief, *,
            tool_call_id: str | None = None) -> SubagentResult:
        """Run one child to completion and fold its result into the parent.

        Appends the bounded result payload to the parent as a tool-role
        message (with ``tool_call_id`` when the parent's assistant turn
        carried one) and returns the same result. Child failures come back
        as ``ok=False`` results; the raises are the typed caller errors
        (an empty task, a spawn while a child is open).
        """
        if not isinstance(brief.task, str) or not brief.task.strip():
            raise SubagentConfigError("the brief task must be a non-empty string")
        if self._open:
            raise SubagentConcurrencyError(
                "a subagent is already open on this parent; subagents run "
                "sequentially, so the open child must fold back first")
        self._open = True
        try:
            self._spawned += 1
            key = child_session_key(self.parent.session_cache_key, self._spawned)
            result = self._run_child(key, brief)
            self.parent.add_tool_result(result.text, tool_call_id=tool_call_id)
        finally:
            self._open = False
        return result

    def _run_child(self, key: str, brief: SubagentBrief) -> SubagentResult:
        """The child tool loop: request, execute tool calls, repeat.

        The loop ends on the first assistant message with no tool calls;
        that message's content is the result. Failed tool calls feed back
        into the child as tool results (the executor never raises), so the
        child can correct itself within the turn budget.
        """
        child = Conversation(session_cache_key=key, system=brief.system)
        child.add_user(brief.opening_message())
        turns = 0
        while turns < self.max_child_turns:
            turns += 1
            try:
                completion = self.client.complete(
                    child, tools=self._tools, max_tokens=self.max_tokens,
                    temperature=self.temperature, top_p=self.top_p)
            except ClientError as e:
                return self._failure(key, "client_error", str(e), turns)
            try:
                child.add_assistant_message(completion.message)
            except ValueError as e:
                return self._failure(key, "invalid_message", str(e), turns)
            try:
                calls = self.parse(completion.message)
            except ToolCallParseError as e:
                return self._failure(key, "parse_failure", str(e), turns)
            if not calls:
                text, truncated = _bounded(
                    completion.content or "", self.max_result_chars)
                return SubagentResult(
                    ok=True, text=text, reason="completed",
                    session_cache_key=key, turns=turns, truncated=truncated)
            for call in calls:
                outcome = execute_tool_call(
                    self.registry, call, workdir=self.workdir,
                    policy=self.policy)
                child.add_tool_result(outcome.output, tool_call_id=call.id)
        return self._failure(
            key, "max_turns",
            f"no final answer after {turns} request(s)", turns)

    def _failure(self, key: str, reason: str, detail: str,
                 turns: int) -> SubagentResult:
        text, truncated = _bounded(
            f"error: subagent ({reason}): {detail}", self.max_result_chars)
        return SubagentResult(
            ok=False, text=text, reason=reason, session_cache_key=key,
            turns=turns, truncated=truncated)


def _bounded(text: str, limit: int) -> tuple[str, bool]:
    """Cut the payload at the limit; a marker records that a cut happened."""
    if len(text) <= limit:
        return text, False
    return (text[:limit] + f"\n[truncated: result exceeded {limit} characters]",
            True)
