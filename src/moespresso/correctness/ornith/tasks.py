"""Gate v2 task definitions for Ornith.

Three instrument families:

  hard_reasoning   Four exact-answer items drawn from the code-verified private
                   key (referenced by question number; the prose and answers stay
                   in the private fixture and are read by path, never embedded).
  agentic_coding   Three self-verifying tasks: the model emits an implementation
                   through a qwen3_xml tool call, and the harness runs it against
                   hidden test cases in a sandboxed subprocess. Original content,
                   so it lives in this module.
  long_context     Two exact-scored questions over a large context built from
                   real repository source with planted facts: one needle recall
                   and one aggregation. Answers verify by normalized string match.

Per-item fields: a stable seed, a token cap sized from a measured budget, and a
`token_hungry` flag so the caller can hold the extended-reasoning items to at
most one. Thinking is off for every render.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Private fixture, read by path at run time. Never embed its prose or answers.
FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "correctness" / "fixtures" / "ornith" / "private"
)
QUESTIONS_PATH = FIXTURE_DIR / "questions_reconstructed.json"
KEY_PATH = FIXTURE_DIR / "benchmark_key_verified.json"

# Repository source files concatenated to build the long-context corpus. All are
# committed source, so the context is reproducible from the tree.
REPO_ROOT = Path(__file__).resolve().parents[4]
LONG_CONTEXT_SOURCES = (
    "src/moespresso/runtime/serve.py",
    "src/moespresso/runtime/http.py",
    "src/moespresso/runtime/build.py",
    "src/moespresso/runtime/kv_policy.py",
    "src/moespresso/runtime/generation.py",
    "src/moespresso/correctness/gate.py",
    "src/moespresso/correctness/ladder.py",
    "src/moespresso/correctness/reconstruct.py",
    "src/moespresso/core/artifact.py",
)


@dataclass(frozen=True)
class HardReasoningItem:
    id: str
    question_number: str
    seed: int
    max_tokens: int
    token_hungry: bool = False


@dataclass(frozen=True)
class AgenticCodingTask:
    id: str
    seed: int
    max_tokens: int
    entry: str
    instruction: str
    hidden_tests: tuple = ()
    token_hungry: bool = False


@dataclass(frozen=True)
class LongContextItem:
    id: str
    kind: str  # "needle" or "aggregation"
    seed: int
    max_tokens: int
    question: str
    expected: str
    token_hungry: bool = False


# --- hard reasoning: four items spanning difficulty, at most one token-hungry ---
# Q1 number theory (integer), Q2 group theory (hard, integer), Q6 probability
# (fraction), Q8 combinatorics (integer). None is the resistor-network item that
# the task gate found runs to the cap.
HARD_REASONING: tuple[HardReasoningItem, ...] = (
    HardReasoningItem(id="hr_q1_number_theory", question_number="1",
                      seed=20260709, max_tokens=3072),
    HardReasoningItem(id="hr_q2_group_theory", question_number="2",
                      seed=20260709, max_tokens=10240, token_hungry=True),
    HardReasoningItem(id="hr_q6_probability", question_number="6",
                      seed=20260709, max_tokens=3072),
    HardReasoningItem(id="hr_q8_combinatorics", question_number="8",
                      seed=20260709, max_tokens=10240),
)


# --- agentic coding: three self-verifying tasks ---
# Each solution is 10-30 lines. The tool schema is passed to the template so the
# model emits a qwen3_xml tool call submitting the implementation.
SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_solution",
        "description": (
            "Submit a complete Python implementation. The code parameter must "
            "define the required top-level function and any helpers it needs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source defining the required function.",
                },
            },
            "required": ["code"],
        },
    },
}


AGENTIC_CODING: tuple[AgenticCodingTask, ...] = (
    AgenticCodingTask(
        id="ac_run_length_encode",
        seed=20260709,
        max_tokens=2048,
        entry="rle_encode",
        instruction=(
            "Implement a Python function `rle_encode(s)` that run-length encodes "
            "a string. For each maximal run of one repeated character, output the "
            "character followed by the run length as a base-10 integer. A run of "
            "length 1 still gets the count 1. Example: rle_encode('aaabbc') "
            "returns 'a3b2c1'. rle_encode('') returns ''. Submit the "
            "implementation with the submit_solution tool."
        ),
        hidden_tests=(
            {"args": ["aaabbc"], "expected": "a3b2c1"},
            {"args": [""], "expected": ""},
            {"args": ["x"], "expected": "x1"},
            {"args": ["aabbaa"], "expected": "a2b2a2"},
            {"args": ["zzzzz"], "expected": "z5"},
        ),
    ),
    AgenticCodingTask(
        id="ac_balanced_brackets",
        seed=20260709,
        max_tokens=2048,
        entry="is_balanced",
        instruction=(
            "Implement a Python function `is_balanced(s)` that returns True if "
            "every bracket in the string is correctly matched and nested, and "
            "False otherwise. Consider three bracket pairs: (), [], and {}. "
            "Characters other than brackets are ignored. Example: "
            "is_balanced('a(b[c]d)e') returns True; is_balanced('([)]') returns "
            "False. Submit the implementation with the submit_solution tool."
        ),
        hidden_tests=(
            {"args": ["a(b[c]d)e"], "expected": True},
            {"args": ["([)]"], "expected": False},
            {"args": [""], "expected": True},
            {"args": ["((("], "expected": False},
            {"args": ["{[()]}"], "expected": True},
            {"args": [")("], "expected": False},
        ),
    ),
    AgenticCodingTask(
        id="ac_word_frequency_top",
        seed=20260709,
        max_tokens=2560,
        entry="most_common_word",
        instruction=(
            "Implement a Python function `most_common_word(text)` that returns "
            "the most frequent word in the text. Words are maximal runs of "
            "letters compared case-insensitively (lowercase the result). On a "
            "tie in frequency, return the word that is smallest in alphabetical "
            "order. For empty input with no words, return ''. Example: "
            "most_common_word('The cat the CAT dog') returns 'cat'. Submit the "
            "implementation with the submit_solution tool."
        ),
        hidden_tests=(
            {"args": ["The cat the CAT dog"], "expected": "cat"},
            {"args": [""], "expected": ""},
            {"args": ["a a b b"], "expected": "a"},
            {"args": ["one, one; two. two! three three three"], "expected": "three"},
            {"args": ["Zebra zebra apple Apple"], "expected": "apple"},
        ),
    ),
)


# --- long context: planted facts and the two questions ---
# The facts are planted into the concatenated repository text as comment lines so
# they read as content rather than instructions. The needle asks for one planted
# token; the aggregation sums three planted numbers.
@dataclass(frozen=True)
class PlantedFact:
    marker: str
    text: str


NEEDLE_TOKEN = "ORCHID-4417-KESTREL"
PLANTED_NUMBERS = (271, 314, 577)  # sum = 1162

PLANTED_FACTS: tuple[PlantedFact, ...] = (
    PlantedFact(
        marker="LONGCTX_NEEDLE",
        text=(
            "# Field audit note: the sealed calibration record for this build "
            f"carries the access token {NEEDLE_TOKEN}, issued once and never "
            "rotated."
        ),
    ),
    PlantedFact(
        marker="LONGCTX_NUM_A",
        text=(
            f"# Ledger entry A: the north shard recorded {PLANTED_NUMBERS[0]} "
            "sealed checkpoints during the audit window."
        ),
    ),
    PlantedFact(
        marker="LONGCTX_NUM_B",
        text=(
            f"# Ledger entry B: the east shard recorded {PLANTED_NUMBERS[1]} "
            "sealed checkpoints during the audit window."
        ),
    ),
    PlantedFact(
        marker="LONGCTX_NUM_C",
        text=(
            f"# Ledger entry C: the west shard recorded {PLANTED_NUMBERS[2]} "
            "sealed checkpoints during the audit window."
        ),
    ),
)


LONG_CONTEXT_ITEMS: tuple[LongContextItem, ...] = (
    LongContextItem(
        id="lc_needle_token",
        kind="needle",
        seed=20260709,
        max_tokens=512,
        question=(
            "Somewhere in the document above is a field audit note stating a "
            "one-time access token for the sealed calibration record. Reply with "
            "only that access token and nothing else."
        ),
        expected=NEEDLE_TOKEN,
    ),
    LongContextItem(
        id="lc_aggregation_sum",
        kind="aggregation",
        seed=20260709,
        max_tokens=1024,
        question=(
            "The document above contains three ledger entries (A, B, and C), each "
            "stating a number of sealed checkpoints recorded by one shard. Add "
            "the three numbers and reply with only the total as a base-10 "
            "integer."
        ),
        expected=str(sum(PLANTED_NUMBERS)),
    ),
)


def load_private_questions() -> tuple[dict, dict]:
    """Load the private question prose and code-verified key by path."""
    questions = json.loads(QUESTIONS_PATH.read_text())["questions"]
    key = json.loads(KEY_PATH.read_text())
    return questions, key


def _plant_facts_into(blocks: list[str]) -> list[str]:
    """Distribute planted-fact comment lines across the source blocks.

    Facts are inserted at even intervals so the needle sits deep in the context
    rather than at an edge.
    """
    if not blocks:
        return blocks
    out = list(blocks)
    n = len(out)
    for index, fact in enumerate(PLANTED_FACTS):
        # Distribute facts across interior blocks and keep both edges untouched.
        position = 1 + int((index + 1) / (len(PLANTED_FACTS) + 1) * max(n - 2, 1))
        position = min(max(position, 1), n - 1) if n > 1 else 0
        out[position] = fact.text + "\n" + out[position]
    return out


def build_long_context(root: Path | None = None) -> str:
    """Concatenate the repository source files and plant the facts.

    Returns the raw context string. The planted-fact comment lines are woven into
    the interior so both the needle and the aggregation numbers sit inside the
    body of the document.
    """
    base = root or REPO_ROOT
    blocks = []
    for rel in LONG_CONTEXT_SOURCES:
        path = base / rel
        header = f"# ===== FILE: {rel} =====\n"
        blocks.append(header + path.read_text(encoding="utf-8"))
    blocks = _plant_facts_into(blocks)
    return "\n\n".join(blocks)
