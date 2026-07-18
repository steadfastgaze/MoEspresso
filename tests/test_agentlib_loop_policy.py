"""The tool-nudge loop policy.

Pins the firing contract: a nudge fires only for a zero-call final turn
before any tool call has run, fires at most once, and never fires on a
legitimate zero-call final answer after tool results.
"""

from __future__ import annotations

from moespresso.agentlib.loop_policy import NUDGE_MESSAGE, ToolNudgePolicy


def test_fires_on_first_zero_call_final_turn():
    policy = ToolNudgePolicy()
    assert policy.wants_reprompt(final=True, calls_in_turn=0,
                                 calls_before_turn=0)


def test_never_fires_twice():
    policy = ToolNudgePolicy()
    assert policy.wants_reprompt(final=True, calls_in_turn=0,
                                 calls_before_turn=0)
    policy.note_fired()
    assert policy.fired == 1
    assert not policy.wants_reprompt(final=True, calls_in_turn=0,
                                     calls_before_turn=0)


def test_does_not_fire_after_tool_results():
    policy = ToolNudgePolicy()
    assert not policy.wants_reprompt(final=True, calls_in_turn=0,
                                     calls_before_turn=2)


def test_does_not_fire_on_non_final_or_calling_turns():
    policy = ToolNudgePolicy()
    assert not policy.wants_reprompt(final=False, calls_in_turn=0,
                                     calls_before_turn=0)
    assert not policy.wants_reprompt(final=True, calls_in_turn=1,
                                     calls_before_turn=0)


def test_nudge_message_names_no_dialect():
    for marker in ("<tool_call>", "DSML", "JSON", "envelope"):
        assert marker not in NUDGE_MESSAGE
