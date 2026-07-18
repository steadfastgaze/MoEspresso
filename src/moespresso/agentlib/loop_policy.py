"""Reusable agent-loop policies above the dialect layer.

The tool-nudge policy covers a recorded failure class: on a task that
requires tools, the model's first turn sometimes arrives as plain prose
with no call markup at all, and a dialect-blind loop reads it as a final
answer and stops with the task unstarted. Repair cannot apply because
nothing was attempted. The policy re-prompts exactly once when a final
turn arrives before any tool call has run, then accepts the next outcome
either way. The counter guard makes a second nudge structurally
impossible, so a model that keeps narrating costs one extra request and
nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass

# The nudge is a plain user message; it names no dialect so every arm can
# use it unchanged.
NUDGE_MESSAGE = (
    "No tool call was found in your reply, but this task requires using "
    "the available tools. Either make the tool call now, using exactly "
    "the documented format, or restate your final answer if the task is "
    "already complete."
)


@dataclass
class ToolNudgePolicy:
    """Re-prompt once when a final turn arrives before any tool call.

    ``fired`` is the re-prompt counter that sits beside the repair
    telemetry counters; a consumer reads it per scope the same way it
    reads ``RepairTelemetry.fires``.
    """

    limit: int = 1
    fired: int = 0

    def wants_reprompt(self, *, final: bool, calls_in_turn: int,
                       calls_before_turn: int) -> bool:
        """True when this final turn should be nudged instead of accepted.

        Fires only for a final turn carrying no calls, before any tool
        call has run in the conversation, and at most ``limit`` times. A
        zero-call final answer after tool results is legitimate and never
        fires.
        """
        if not final or calls_in_turn:
            return False
        if calls_before_turn:
            return False
        return self.fired < self.limit

    def note_fired(self) -> None:
        self.fired += 1
