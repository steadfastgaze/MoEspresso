"""DeepSeek-V4 Q3 long-context fact-recall gate.

Q3 is deliberately deterministic: it vendors the DS4 story generator semantics,
checks that committed golden facts align with the generated prompt, and scores the
served model's extracted ``Name=number`` answer lines.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from moespresso.core.artifact import Validation, make_artifact
from moespresso.correctness.deepseek_v4.parity import _json_safe
from moespresso.correctness.environment import mlx_wheel_tag
from moespresso.correctness.ladder import PRODUCER

Q3_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "deepseek_v4" / "q3_long_context"
)
Q3_PROMPT_ID = "long_context_story_fact_recall"
Q3_SCHEMA = "ds4-q3-long-context-v1"
Q3_MANIFEST_SCHEMA = "ds4-q3-long-context-manifest-v1"
Q3_ANSWERS_SCHEMA = "ds4-q3-long-context-answers-v1"
Q3_GENERATOR_SEED = 20260513

SYSTEM_PROMPT = (
    "You are a careful assistant. Read the story, remember the assignments, "
    "and answer the final task exactly."
)

OPENING = """\
You are reading a long story from the harbor town of Bellwether. The story is
ordinary on purpose: people speak, walk, remember, repair things, argue about
weather, and sometimes receive a private assignment number written out in
words. Your job at the end is to recover the assignment numbers.

Important rule while reading: only assignments stated as "was assigned the
number ..." count. Other ages, prices, dates, distances, room numbers, rumors,
or guesses do not count. The assignment numbers in the story are written in
words, not numerals.

"""

SCENE_TEMPLATES = [
    """\
At first light the harbor smelled of rope, rain, and cedar smoke. {lead} crossed
the quay with a folded map tucked under one arm, stopping whenever gulls made a
mess of the chalk marks near the fish stalls. {friend} had promised to fix the
south gate before supper, but the hinges complained so loudly that everyone
pretended not to hear them. In the bakery window, loaves cooled beneath linen
while a child counted shells in a wooden bowl. No one was in a hurry, because
Bellwether moved by tide and habit, not by the bells on the council tower.

The archivist Mara wrote notes in brown ink, never black, because black ink made
old ledgers look like court summonses. She watched {lead} and {friend} pass the
fountain, then added a line about the morning fog. Her notes often wandered into
small details: the color of a scarf, the chipped rim of a cup, the way a door
kept opening after it had been firmly shut.
""",
    """\
By noon the market had filled with baskets of pears, lamp oil, brass hooks, and
paper flowers. {lead} bargained for twine while {friend} listened to a sailor
describe a storm that seemed to grow taller every time he retold it. The town
clock had stopped again, but nobody agreed on when, so every shopkeeper chose a
different hour and defended it with confidence.

Mara sat outside the apothecary and copied the day's ordinary business into the
festival ledger. She liked ordinary business best. Extraordinary business came
with signatures, seals, and people who leaned over her shoulder. Ordinary
business arrived quietly, sat down, and became history before anyone noticed.
""",
    """\
In the afternoon, a rehearsal for the midsummer play blocked the west road.
{lead} carried a crate of lantern glass through the crowd while {friend} read
lines from a damp script. Someone had painted the moon too blue on the backdrop,
and three people argued about whether a theatrical moon was allowed to be wrong.
The argument lasted longer than the scene.

The ledger lay open on a bench. Mara kept it weighted with two smooth stones
from the beach. She recorded who borrowed the theater ladder, who returned the
wrong kettle, and who claimed the missing red umbrella. The handwriting was calm
even when the town was not.
""",
    """\
Evening brought a quiet wind and the sound of shutters being latched one after
another. {lead} helped carry chairs into the assembly hall, where the floor had
been scrubbed until it smelled faintly of salt. {friend} found a lost button
near the door and pinned it to the notice board with a note that said, simply,
"lonely."

Mara walked the perimeter of the hall with the ledger pressed to her chest. She
had learned that important facts hid best inside unimportant days. A missing
button, a changed route, a corrected name, a number assigned without ceremony:
these were the things that later made sense of everything else.
""",
    """\
Rain arrived after midnight and softened every sound in Bellwether. {lead}
stood beneath the awning of the rope-maker's shop, waiting for {friend}, who had
gone back for a forgotten satchel. The street lamps shone in puddles like coins
that nobody could spend. From the hill, the lighthouse blinked with patient
regularity.

Mara remained awake in the archive room. She sharpened a pencil, rejected it,
and returned to brown ink. The festival ledger had grown heavy with the week:
weather notes, repairs, errands, apologies, and a few facts she underlined only
once so they would not look too important.
""",
]

BRIDGE_SENTENCES = [
    "The town talked around the matter without naming it directly.",
    "A kettle whistled somewhere nearby and broke the silence at exactly the right moment.",
    "Mara did not decorate the sentence; she wanted it to be easy to find later.",
    "The phrase was spoken once, then folded into the rest of the day's business.",
    "No one treated the entry like a puzzle, which is why it survived unchanged.",
    "The ledger page smelled of dust, salt, and the faint sweetness of drying glue.",
]

_ANSWER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z-]*)\s*=\s*([0-9]+)\s*$")


@dataclass(frozen=True)
class Q3Fact:
    name: str
    word: str
    number: int


def _blocking(code: str, message: str, *, path: str, expected=None, actual=None) -> Validation:
    return Validation(
        "error",
        code,
        message,
        path=path,
        phase="Q3",
        blocking=True,
        expected=_json_safe(expected),
        actual=_json_safe(actual),
    )


def load_q3_facts(fixture_root: Path | None = None) -> list[Q3Fact]:
    root = Path(fixture_root or Q3_FIXTURE_ROOT)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != Q3_MANIFEST_SCHEMA:
        raise ValueError(f"bad Q3 manifest schema: {manifest.get('schema')!r}")
    answers = json.loads((root / manifest["answers_file"]).read_text(encoding="utf-8"))
    if answers.get("schema") != Q3_ANSWERS_SCHEMA:
        raise ValueError(f"bad Q3 answers schema: {answers.get('schema')!r}")
    facts = []
    for i, row in enumerate(answers.get("facts", [])):
        try:
            facts.append(Q3Fact(
                name=str(row["name"]),
                word=str(row["word"]),
                number=int(row["number"]),
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"bad Q3 fact row {i}: {row!r}") from e
    if not facts:
        raise ValueError("Q3 answers fixture is empty")
    return facts


def assignment_sentence(name: str, word: str) -> str:
    return (
        f"During that same scene, {name} was assigned the number {word}. "
        f"Mara wrote the assignment in words, closed the ledger for a moment, "
        f"and then returned to the smaller gossip of the harbor."
    )


def make_q3_story(facts: list[Q3Fact] | None = None) -> str:
    facts = facts or load_q3_facts()
    rng = random.Random(Q3_GENERATOR_SEED)
    names = [fact.name for fact in facts]
    fact_by_scene = {7 + i * 11: fact for i, fact in enumerate(facts)}
    scenes: list[str] = []

    for scene_index in range(190):
        lead = rng.choice(names)
        friend = rng.choice([n for n in names if n != lead])
        scene = SCENE_TEMPLATES[scene_index % len(SCENE_TEMPLATES)].format(
            lead=lead,
            friend=friend,
        )

        if scene_index in fact_by_scene:
            fact = fact_by_scene[scene_index]
            scene += (
                "\n"
                + rng.choice(BRIDGE_SENTENCES)
                + " "
                + assignment_sentence(fact.name, fact.word)
                + "\n"
            )
        elif scene_index % 9 == 3:
            scene += (
                "\nMara heard someone mention an old rumor about a numbered key, "
                "but she crossed it out because it was not an assignment and did "
                "not belong in the final list.\n"
            )
        elif scene_index % 11 == 6:
            scene += (
                "\nA shop sign advertised a discount in careful words, but prices "
                "and discounts were not assignment numbers, so Mara ignored them.\n"
            )

        scenes.append(scene)

    final_names = ", ".join(fact.name for fact in facts)
    question = f"""\

Final task:

Compile the assignment ledger from the story. Convert the spelled-out numbers to
ordinary decimal numerals. Write only lines in the form Name=number. The first
example line is Bob=34; include that line and all remaining people.

People to list: {final_names}.

No bullets, no prose, no explanation.
"""
    return OPENING + "\n".join(scenes) + question


def validate_q3_story_alignment(story: str, facts: list[Q3Fact]) -> list[str]:
    problems: list[str] = []
    for fact in facts:
        assignment = f"{fact.name} was assigned the number {fact.word}"
        if assignment not in story:
            problems.append(f"missing assignment sentence for {fact.name}")
        if f"{fact.name}" not in story.rsplit("People to list:", 1)[-1]:
            problems.append(f"missing final people-list entry for {fact.name}")
    return problems


def extract_q3_answer_lines(text: str) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = _ANSWER_RE.match(line)
        if match is None:
            if line.strip():
                rows.append({
                    "line": line_no,
                    "text": line,
                    "name": None,
                    "number": None,
                    "parseable": False,
                })
            continue
        rows.append({
            "line": line_no,
            "text": line,
            "name": match.group(1),
            "number": int(match.group(2)),
            "parseable": True,
        })
    return rows


def score_q3_fact_recall(text: str, facts: list[Q3Fact]) -> dict:
    expected = {fact.name: fact.number for fact in facts}
    parsed = extract_q3_answer_lines(text)
    observed: dict[str, int] = {}
    duplicate_names: list[str] = []
    unparseable = []
    for row in parsed:
        if not row["parseable"]:
            unparseable.append(row)
            continue
        name = row["name"]
        assert isinstance(name, str)
        if name in observed:
            duplicate_names.append(name)
        observed[name] = int(row["number"])

    correct = {
        name: number
        for name, number in observed.items()
        if expected.get(name) == number
    }
    wrong = {
        name: {"expected": expected[name], "actual": observed[name]}
        for name in sorted(set(expected) & set(observed))
        if expected[name] != observed[name]
    }
    missing = {name: expected[name] for name in sorted(set(expected) - set(observed))}
    extra = {name: observed[name] for name in sorted(set(observed) - set(expected))}
    exact_match = (
        len(correct) == len(expected)
        and not missing
        and not wrong
        and not extra
        and not duplicate_names
        and not unparseable
    )
    return {
        "expected": expected,
        "observed": observed,
        "parsed_lines": parsed,
        "correct": correct,
        "missing": missing,
        "wrong": wrong,
        "extra": extra,
        "duplicate_names": sorted(set(duplicate_names)),
        "unparseable_lines": unparseable,
        "correct_count": len(correct),
        "expected_count": len(expected),
        "recall": len(correct) / len(expected) if expected else 0.0,
        "exact_match": exact_match,
    }


def validate_deepseek_v4_q3_evidence(evidence: dict) -> list[Validation]:
    out: list[Validation] = []
    if evidence.get("family") != "deepseek_v4_flash":
        out.append(_blocking(
            "deepseek_v4.q3.family_mismatch",
            "Q3 evidence must be for the DeepSeek-V4-Flash family",
            path="/family",
            expected="deepseek_v4_flash",
            actual=evidence.get("family"),
        ))
    run = evidence.get("run")
    if not isinstance(run, dict):
        out.append(_blocking(
            "deepseek_v4.q3.missing_run_config",
            "Q3 evidence must describe the generation run",
            path="/run",
            expected="object",
            actual=type(run).__name__,
        ))
    else:
        if run.get("decode") != "greedy":
            out.append(_blocking(
                "deepseek_v4.q3.decode_not_greedy",
                "Q3 must use greedy decode",
                path="/run/decode",
                expected="greedy",
                actual=run.get("decode"),
            ))
        if run.get("thinking") is not False:
            out.append(_blocking(
                "deepseek_v4.q3.thinking_not_disabled",
                "Q3 must run with thinking disabled",
                path="/run/thinking",
                expected=False,
                actual=run.get("thinking"),
            ))
        if run.get("temperature") != 0:
            out.append(_blocking(
                "deepseek_v4.q3.temperature_not_zero",
                "Q3 must use temperature 0",
                path="/run/temperature",
                expected=0,
                actual=run.get("temperature"),
            ))

    reference = evidence.get("reference")
    if not isinstance(reference, dict):
        out.append(_blocking(
            "deepseek_v4.q3.missing_reference",
            "Q3 evidence must describe the deterministic fixture reference",
            path="/reference",
            expected="object",
            actual=type(reference).__name__,
        ))
    else:
        expected_ref = {
            "kind": "deterministic_long_context_fact_recall",
            "schema": Q3_SCHEMA,
            "source": "ds4-long-context-story-generator",
        }
        for key, value in expected_ref.items():
            if reference.get(key) != value:
                out.append(_blocking(
                    f"deepseek_v4.q3.reference_{key}",
                    f"Q3 reference must declare {key}={value!r}",
                    path=f"/reference/{key}",
                    expected=value,
                    actual=reference.get(key),
                ))

    candidate = evidence.get("candidate")
    if not isinstance(candidate, dict):
        out.append(_blocking(
            "deepseek_v4.q3.missing_candidate",
            "Q3 evidence must describe the MoEspresso package candidate",
            path="/candidate",
            expected="object",
            actual=type(candidate).__name__,
        ))
    elif candidate.get("kind") != "moespresso_mlx_package":
        out.append(_blocking(
            "deepseek_v4.q3.candidate_kind",
            "Q3 must judge the real MoEspresso MLX package runtime",
            path="/candidate/kind",
            expected="moespresso_mlx_package",
            actual=candidate.get("kind"),
        ))

    prompt = evidence.get("prompt")
    if not isinstance(prompt, dict):
        out.append(_blocking(
            "deepseek_v4.q3.missing_prompt",
            "Q3 evidence must carry one prompt row",
            path="/prompt",
            expected="object",
            actual=type(prompt).__name__,
        ))
    else:
        if prompt.get("id") != Q3_PROMPT_ID:
            out.append(_blocking(
                "deepseek_v4.q3.prompt_id",
                "Q3 prompt id must match the committed long-context fixture",
                path="/prompt/id",
                expected=Q3_PROMPT_ID,
                actual=prompt.get("id"),
            ))
        fixture_problems = prompt.get("fixture_alignment_errors")
        if fixture_problems:
            out.append(_blocking(
                "deepseek_v4.q3.fixture_alignment",
                "Q3 golden answers must align with the generated story",
                path="/prompt/fixture_alignment_errors",
                expected=[],
                actual=fixture_problems,
            ))

    score = evidence.get("score")
    if not isinstance(score, dict):
        out.append(_blocking(
            "deepseek_v4.q3.missing_score",
            "Q3 evidence must carry fact-recall score details",
            path="/score",
            expected="object",
            actual=type(score).__name__,
        ))
    else:
        if score.get("exact_match") is not True:
            out.append(_blocking(
                "deepseek_v4.q3.fact_recall_not_exact",
                "Q3 requires exact recall of the deterministic name=number facts",
                path="/score/exact_match",
                expected=True,
                actual=score.get("exact_match"),
            ))
        expected_count = score.get("expected_count")
        correct_count = score.get("correct_count")
        if not isinstance(expected_count, int) or not isinstance(correct_count, int):
            out.append(_blocking(
                "deepseek_v4.q3.bad_score_counts",
                "Q3 score must carry integer correct and expected counts",
                path="/score",
                expected="integer correct_count and expected_count",
                actual={"correct_count": correct_count, "expected_count": expected_count},
            ))
    return out


def make_deepseek_v4_q3_evidence(subject: dict, external_evidence: dict) -> dict:
    external_evidence = _json_safe(external_evidence)
    findings = validate_deepseek_v4_q3_evidence(external_evidence)
    blocking = any(f.blocking for f in findings)
    inputs = external_evidence.get("inputs", []) if isinstance(external_evidence, dict) else []
    score = external_evidence.get("score", {}) if isinstance(external_evidence, dict) else {}
    return make_artifact(
        "correctness_evidence",
        subject,
        PRODUCER,
        status="invalid" if blocking else "valid",
        validation=findings,
        inputs=inputs,
        rung="Q3",
        summary={
            "findings": len(findings),
            "blocking": sum(1 for f in findings if f.blocking),
            "prompt_id": Q3_PROMPT_ID,
            "correct_count": score.get("correct_count", 0),
            "expected_count": score.get("expected_count", 0),
            "recall": score.get("recall", 0.0),
            "exact_match": bool(score.get("exact_match")),
            # The wheel variant keys the quality lattice; record it so a
            # silent reinstall flip is attributable from the artifact alone.
            "mlx_wheel": mlx_wheel_tag(),
        },
        external_evidence=external_evidence,
    )


def q3_external_evidence_from_text(
    *,
    package_dir: Path,
    manifest: dict,
    candidate_text: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    finish_reason: str | None,
    fixture_root: Path | None = None,
    max_tokens: int,
) -> dict:
    fixture_root = Path(fixture_root or Q3_FIXTURE_ROOT)
    facts = load_q3_facts(fixture_root)
    story = make_q3_story(facts)
    fixture_errors = validate_q3_story_alignment(story, facts)
    score = score_q3_fact_recall(candidate_text, facts)
    return {
        "family": "deepseek_v4_flash",
        "run": {
            "decode": "greedy",
            "thinking": False,
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": int(max_tokens),
            "prompt_renderer": "deepseek_v4_dsv4",
        },
        "reference": {
            "kind": "deterministic_long_context_fact_recall",
            "schema": Q3_SCHEMA,
            "source": "ds4-long-context-story-generator",
            "generator_seed": Q3_GENERATOR_SEED,
            "fixture_root": str(fixture_root),
        },
        "candidate": {
            "kind": "moespresso_mlx_package",
            "package_dir": str(package_dir),
            "package_manifest_id": manifest.get("artifact_id"),
            "family": manifest.get("architecture", {}).get("family"),
        },
        "inputs": [
            {"path": str(fixture_root / "manifest.json"), "role": "q3_manifest"},
            {"path": str(fixture_root / "answers.json"), "role": "q3_golden_answers"},
            {"path": str(Path(package_dir) / "package_manifest.json"), "role": "candidate_package"},
        ],
        "prompt": {
            "id": Q3_PROMPT_ID,
            "story_chars": len(story),
            "system_chars": len(SYSTEM_PROMPT),
            "prompt_tokens": prompt_tokens,
            "fixture_alignment_errors": fixture_errors,
        },
        "generation": {
            "candidate_text": candidate_text,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
        },
        "score": score,
    }
