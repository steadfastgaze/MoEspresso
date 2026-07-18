"""The scripted user side of the road-test session.

User turns are fixed strings; the assistant turns are real model output and
the tool calls the model makes are really executed. The script is data: the
driver owns ordering, restarts, and interleaving. Turn text is written to
push the model toward one specific tool action per turn so the run stays
deterministic at temperature zero, while the engine-counter assertions never
depend on which tools the model actually picked.

The session accumulates context through full-file reads of the fixture data
segments. The extension turns are a reserve the driver consumes until the
observed context passes its token target, so the guarantee that the session
grows well past the target does not depend on token-per-line estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

from moespresso.agentlib.roadtest.fixture import (
    MARKER_ONE,
    MARKER_TWO,
    Fixture,
)

SYSTEM_PROMPT_A = (
    "You are a code auditing agent working inside a telemetry-processing "
    "project. Use the provided tools to inspect and modify the project when "
    "a task calls for it. Keep answers short and factual: report counts and "
    "findings in one or two sentences. Do not repeat file contents back in "
    "your answers."
)

SYSTEM_PROMPT_B = (
    "You are a data-verification agent double-checking record segments in a "
    "telemetry-processing project. Use the provided tools when a task calls "
    "for it. Answer in one or two short sentences with exact counts."
)

SYSTEM_PROMPT_SUB = (
    "You are a subordinate verification agent handling one delegated side "
    "task in a telemetry-processing project. Use the provided tools when the "
    "task calls for it. Answer in one or two short sentences with exact "
    "counts."
)


@dataclass(frozen=True)
class ScriptTurn:
    """One scripted user turn."""

    turn_id: str
    session: str
    user_text: str


@dataclass(frozen=True)
class RoadtestScript:
    """All scripted turns, grouped by the phase the driver runs them in."""

    opening_a: tuple[ScriptTurn, ...]
    interleaved: tuple[ScriptTurn, ...]
    probe_a: ScriptTurn
    resume_b: ScriptTurn
    delegate_a: ScriptTurn
    subagent_task: str
    subagent_context: str
    followup_a: ScriptTurn
    growth_a: tuple[ScriptTurn, ...]
    extensions_a: tuple[ScriptTurn, ...]
    wrap_a: ScriptTurn

    def all_turns(self) -> tuple[ScriptTurn, ...]:
        return (
            *self.opening_a,
            *self.interleaved,
            self.probe_a,
            self.resume_b,
            self.delegate_a,
            self.followup_a,
            *self.growth_a,
            *self.extensions_a,
            self.wrap_a,
        )


def _read_turn(turn_id: str, session: str, relpath: str) -> ScriptTurn:
    return ScriptTurn(
        turn_id=turn_id,
        session=session,
        user_text=(
            f"Read the file {relpath} in full with the read_file tool, then "
            f"report exactly how many records in it have status error."
        ),
    )


def build_script(fixture: Fixture) -> RoadtestScript:
    """Build the full turn script against a generated fixture layout."""
    segments = {info.relpath: info for info in fixture.data_files}
    seg = sorted(segments)

    opening_a = (
        ScriptTurn(
            "a-01", "a",
            "Run `ls` at the top of the project with the bash tool and give a "
            "one-line summary of the layout.",
        ),
        ScriptTurn(
            "a-02", "a",
            f"Search the src directory for the string {MARKER_ONE} with the "
            "grep tool and report which file defines it.",
        ),
        ScriptTurn(
            "a-03", "a",
            "Read src/metrics.py in full with the read_file tool and identify "
            "the defect in the mean() function in one sentence.",
        ),
        ScriptTurn(
            "a-04", "a",
            "Fix the defect in mean() with the edit tool: the divisor must be "
            "the full length of the sequence. Change only that one line.",
        ),
        ScriptTurn(
            "a-05", "a",
            "Verify the fix: grep src/metrics.py for the exact string "
            "'len(values) - 1' and confirm it no longer appears.",
        ),
        _read_turn("a-06", "a", seg[0]),
    )

    probe_a = ScriptTurn(
        "a-07", "a",
        "Here is the full text of data/large_reference.txt for reference:\n\n"
        + fixture.large_reference_text
        + "\nWithout using any tools, answer from the text above alone: which "
        "trace tag of record does it declare, and exactly how many readings "
        "are flagged REVIEW?",
    )

    interleaved = (
        _read_turn("b-01", "b", seg[15]),
        _read_turn("a-08", "a", seg[1]),
        ScriptTurn(
            "b-02", "b",
            f"Search the src directory for the string {MARKER_TWO} with the "
            "grep tool and report which file defines it.",
        ),
        _read_turn("a-09", "a", seg[2]),
        _read_turn("b-03", "b", seg[16]),
        _read_turn("a-10", "a", seg[3]),
        _read_turn("a-11", "a", seg[4]),
    )

    resume_b = ScriptTurn(
        "b-04", "b",
        "Read src/metrics.py in full with the read_file tool and confirm in "
        "one sentence whether mean() divides by the full sequence length.",
    )

    # The subagent scenario: a parent turn announcing the delegation, the
    # child's brief (task plus the explicit context that crosses the
    # boundary), and the parent turn that consumes the folded-back report.
    # The two-step task gives the child several requests on its own chain,
    # and the full segment read grows that chain with real tool output.
    delegate_a = ScriptTurn(
        "a-delegate", "a",
        "A subordinate verification agent will now take over one delegated "
        "side task; its report will arrive as a tool result. Do not use any "
        "tools this turn; acknowledge the handoff in one sentence.",
    )
    subagent_task = (
        f"Search the src directory for the string {MARKER_TWO} with the "
        f"grep tool and report which file defines it. After the search, "
        f"read the file {seg[-1]} in full with the read_file tool and "
        f"report exactly how many records in it have status error."
    )
    subagent_context = (
        "You are working in the same project checkout as the delegating "
        "agent: source files live under src and record segments under data."
    )
    followup_a = ScriptTurn(
        "a-followup", "a",
        "The subordinate agent's report has arrived as the tool result "
        "above. Without using any tools, restate its key finding in one "
        "sentence.",
    )

    growth_a = (
        _read_turn("a-12", "a", seg[5]),
    )

    # The extension reserve: fresh segments the driver consumes until the
    # session passes its token target. Segments 15 and 16 belong to session
    # b; everything else unread by the earlier phases lands here, sized so
    # the reserve covers a 110k-token target with margin.
    extension_indexes = [6, 7, 8, 9, 10, 11, 12, 13, 14] + list(
        range(17, len(seg)))
    extensions_a = tuple(
        _read_turn(f"a-{13 + i:02d}", "a", seg[index])
        for i, index in enumerate(extension_indexes)
    )

    wrap_a = ScriptTurn(
        f"a-{13 + len(extensions_a):02d}", "a",
        "Give a final audit summary in at most four sentences: the defect "
        "you fixed, the marker strings you located, and how many data "
        "segments you reviewed.",
    )

    return RoadtestScript(
        opening_a=opening_a,
        interleaved=interleaved,
        probe_a=probe_a,
        resume_b=resume_b,
        delegate_a=delegate_a,
        subagent_task=subagent_task,
        subagent_context=subagent_context,
        followup_a=followup_a,
        growth_a=growth_a,
        extensions_a=extensions_a,
        wrap_a=wrap_a,
    )
