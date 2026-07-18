from __future__ import annotations

import json

from moespresso.core.artifact import compute_artifact_id, validate_base
from moespresso.correctness.environment import mlx_wheel_tag
from moespresso.correctness.deepseek_v4.q3 import (
    Q3_FIXTURE_ROOT,
    Q3_GENERATOR_SEED,
    Q3_PROMPT_ID,
    load_q3_facts,
    make_deepseek_v4_q3_evidence,
    make_q3_story,
    q3_external_evidence_from_text,
    score_q3_fact_recall,
    validate_q3_story_alignment,
)


Q3_UPSTREAM_GENERATOR = (
    "https://github.com/antirez/ds4/blob/"
    "0cba357ca1bc0e7510421cc26888e420ea942123/"
    "tests/generate_long_context_story_prompt.py"
)


def _perfect_answer() -> str:
    return "\n".join(f"{fact.name}={fact.number}" for fact in load_q3_facts())


def test_q3_manifest_records_immutable_public_generator_provenance():
    manifest = json.loads((Q3_FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["source"] == Q3_UPSTREAM_GENERATOR
    assert manifest["generator_seed"] == Q3_GENERATOR_SEED
    assert manifest["prompt_id"] == Q3_PROMPT_ID


def _external(candidate_text: str) -> dict:
    return q3_external_evidence_from_text(
        package_dir="/tmp/package",
        manifest={
            "artifact_id": "pkg:test",
            "architecture": {"family": "deepseek_v4_flash"},
        },
        candidate_text=candidate_text,
        prompt_tokens=123,
        completion_tokens=16,
        finish_reason="length",
        max_tokens=256,
    )


def test_q3_story_fixture_aligns_with_committed_goldens():
    facts = load_q3_facts()
    story = make_q3_story(facts)

    assert validate_q3_story_alignment(story, facts) == []
    assert "People to list:" in story
    assert story.count("was assigned the number") == len(facts)


def test_q3_story_alignment_fails_closed_for_bad_golden():
    facts = load_q3_facts()
    bad_facts = [*facts[:-1], type(facts[-1])("Priya", "ninety-eight", 98)]

    problems = validate_q3_story_alignment(make_q3_story(facts), bad_facts)

    assert problems == ["missing assignment sentence for Priya"]


def test_q3_scores_exact_name_number_recall():
    score = score_q3_fact_recall(_perfect_answer(), load_q3_facts())

    assert score["exact_match"] is True
    assert score["correct_count"] == score["expected_count"] == 16
    assert score["recall"] == 1.0
    assert score["missing"] == {}
    assert score["wrong"] == {}
    assert score["extra"] == {}


def test_q3_scores_missing_wrong_extra_and_unparseable_lines():
    text = "\n".join([
        "Bob=35",
        "Alice=52",
        "Mystery=7",
        "not a fact line",
    ])

    score = score_q3_fact_recall(text, load_q3_facts())

    assert score["exact_match"] is False
    assert score["correct"] == {"Alice": 52}
    assert score["wrong"] == {"Bob": {"expected": 34, "actual": 35}}
    assert score["extra"] == {"Mystery": 7}
    assert "Clara" in score["missing"]
    assert score["unparseable_lines"][0]["text"] == "not a fact line"


def test_q3_valid_external_evidence_emits_valid_artifact():
    external = _external(_perfect_answer())

    artifact = make_deepseek_v4_q3_evidence({"package_id": "pkg:test"}, external)

    assert validate_base(artifact) == []
    assert artifact["rung"] == "Q3"
    assert artifact["status"] == "valid"
    assert artifact["summary"]["prompt_id"] == Q3_PROMPT_ID
    assert artifact["summary"]["exact_match"] is True
    assert artifact["summary"]["recall"] == 1.0
    assert artifact["summary"]["mlx_wheel"] == mlx_wheel_tag()
    assert artifact["artifact_id"] == compute_artifact_id(artifact)


def test_q3_artifact_fails_closed_when_recall_is_not_exact():
    external = _external("Bob=35\nAlice=52\n")

    artifact = make_deepseek_v4_q3_evidence({"package_id": "pkg:test"}, external)

    assert artifact["status"] == "invalid"
    assert any(
        v["code"] == "deepseek_v4.q3.fact_recall_not_exact"
        for v in artifact["validation"]
    )


def test_q3_artifact_fails_closed_when_fixture_goldens_do_not_align():
    external = _external(_perfect_answer())
    external["prompt"]["fixture_alignment_errors"] = ["missing assignment sentence for Priya"]

    artifact = make_deepseek_v4_q3_evidence({"package_id": "pkg:test"}, external)

    assert artifact["status"] == "invalid"
    assert any(
        v["code"] == "deepseek_v4.q3.fixture_alignment"
        for v in artifact["validation"]
    )
