from __future__ import annotations

import json

import pytest

from moespresso.core.artifact import compute_artifact_id, validate_base
from moespresso.correctness.environment import mlx_wheel_tag
from moespresso.correctness.deepseek_v4.q2 import (
    OPENROUTER_DS4_FLASH_MODEL,
    Q2_FROZEN_PREVIEW_REFERENCE_PATH,
    Q2_NOISE_FLOOR_REFERENCE_PATH,
    Q2_PROMPTS_PATH,
    Q2_REFERENCE_SCHEMA,
    Q2_REFERENCE_PATH,
    aggregate_q2_scores,
    check_q2_capture_output_path,
    compare_q2_score_tables,
    default_q2_capture_path,
    load_q2_prompts,
    make_deepseek_v4_q2_evidence,
    score_q2_token_sequence,
    validate_q2_reference,
)


def _reference() -> dict:
    return {
        "schema": Q2_REFERENCE_SCHEMA,
        "source": "deepseek-official-api",
        "capture": {
            "transport": "openrouter",
            "endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "model": "deepseek/deepseek-v4-flash",
            "provider": {"only": ["deepseek"], "allow_fallbacks": False},
            "temperature": 0,
            "thinking": False,
            "max_tokens": 24,
            "top_logprobs": 20,
        },
        "cases": [
            {
                "id": "case_000",
                "prompt": "Explain a heap.",
                "continuation": "A heap is",
                "official_tokens": [
                    {
                        "index": 0,
                        "token": "A",
                        "bytes": [65],
                        "logprob": -0.01,
                        "top_logprobs": [
                            {"token": "A", "bytes": [65], "logprob": -0.01},
                            {"token": "B", "bytes": [66], "logprob": -9999.0},
                        ],
                    }
                ],
            }
        ],
    }


def _external(rows: list[dict]) -> dict:
    return {
        "family": "deepseek_v4_flash",
        "schema": "ds4-q2-target-token-nll-v1",
        "run": {
            "decode": "teacher_forced_target_nll",
            "thinking": False,
            "temperature": 0,
            "prompt_renderer": "deepseek_v4_dsv4",
        },
        "reference": {
            "kind": "deepseek_official_api_continuations",
            "schema": Q2_REFERENCE_SCHEMA,
            "source": "deepseek-official-api",
            "model": "deepseek/deepseek-v4-flash",
            "endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "path": "/tmp/official.json",
            "cases": len(rows),
        },
        "candidate": {
            "kind": "moespresso_mlx_package",
            "package_dir": "/tmp/package",
            "package_manifest_id": "pkg:test",
            "family": "deepseek_v4_flash",
        },
        "inputs": [],
        "score": aggregate_q2_scores(rows),
        "case_scores": rows,
    }


def test_q2_prompt_fixture_has_100_unique_cases():
    prompts = load_q2_prompts()

    assert len(prompts) == 100
    assert prompts[0].id == "case_000"
    assert prompts[-1].id == "case_099"
    assert len({p.id for p in prompts}) == 100
    assert Q2_PROMPTS_PATH.parent.parent.name == "deepseek_v4"
    assert Q2_REFERENCE_PATH.parent.parent.name == "private"


def test_q2_reference_of_record_and_its_witnesses_stay_private():
    """Every reference path is oracle material and lives outside the tree."""
    for path in (
        Q2_REFERENCE_PATH,
        Q2_NOISE_FLOOR_REFERENCE_PATH,
        Q2_FROZEN_PREVIEW_REFERENCE_PATH,
        default_q2_capture_path(),
    ):
        assert path.parent.parent.name == "private", path
    # The reference of record and the earlier frozen reference are different
    # captures of different checkpoints; scoring one against the other's number
    # is meaningless, so they are never the same file.
    assert Q2_REFERENCE_PATH != Q2_FROZEN_PREVIEW_REFERENCE_PATH
    assert Q2_REFERENCE_PATH.parent != Q2_FROZEN_PREVIEW_REFERENCE_PATH.parent


def test_q2_capture_defaults_to_a_new_timestamped_path():
    from datetime import UTC, datetime

    first = default_q2_capture_path(datetime(2026, 8, 2, 3, 4, 5, tzinfo=UTC))
    second = default_q2_capture_path(datetime(2026, 8, 2, 3, 4, 6, tzinfo=UTC))

    assert first.name == "capture_20260802T030405Z.json"
    assert first != second
    assert first != Q2_REFERENCE_PATH
    assert first != Q2_FROZEN_PREVIEW_REFERENCE_PATH


def test_q2_capture_refuses_to_overwrite_an_existing_reference(tmp_path):
    existing = tmp_path / "pass1.json"
    existing.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError) as excinfo:
        check_q2_capture_output_path(existing)

    assert "pass1.json" in str(excinfo.value)
    assert check_q2_capture_output_path(existing, force=True) == existing
    assert check_q2_capture_output_path(tmp_path / "new.json") == tmp_path / "new.json"


def test_q2_capture_model_default_is_the_dated_slug():
    """An undated slug is not a checkpoint identity, so it cannot be a default."""
    assert OPENROUTER_DS4_FLASH_MODEL.endswith("-0731")


def test_q2_reference_allows_non_selected_sentinel_top_logprobs():
    findings = validate_q2_reference(_reference())

    assert findings == []


def test_q2_reference_rejects_selected_sentinel_logprob():
    reference = _reference()
    reference["cases"][0]["official_tokens"][0]["logprob"] = -9999.0

    findings = validate_q2_reference(reference)

    assert any(v.code == "deepseek_v4.q2.selected_logprob_sentinel" for v in findings)


def test_q2_scores_token_sequence_nll_and_greedy_lcp():
    row = score_q2_token_sequence(
        case_id="case_000",
        target_token_ids=[10, 11, 12],
        target_logprobs=[-0.1, -0.2, -1.0],
        greedy_token_ids=[10, 11, 99],
    )

    assert row["target_tokens"] == 3
    assert row["nll"] == pytest.approx(1.3)
    assert row["avg_nll"] == pytest.approx(1.3 / 3)
    assert row["first_match"] == 1
    assert row["greedy_lcp"] == 2


def test_q2_aggregate_and_compare_match_ds4_metric_shape():
    old_rows = [
        score_q2_token_sequence(
            case_id="case_000",
            target_token_ids=[1, 2],
            target_logprobs=[-1.0, -1.0],
            greedy_token_ids=[1, 9],
        ),
        score_q2_token_sequence(
            case_id="case_001",
            target_token_ids=[3],
            target_logprobs=[-0.5],
            greedy_token_ids=[4],
        ),
    ]
    new_rows = [
        {**old_rows[0], "nll": 1.0, "avg_nll": 0.5},
        {**old_rows[1], "nll": 0.75, "avg_nll": 0.75},
    ]

    agg = aggregate_q2_scores(old_rows)
    delta = compare_q2_score_tables(old_rows, new_rows)

    assert agg["cases"] == 2
    assert agg["target_tokens"] == 3
    assert agg["avg_nll"] == pytest.approx(2.5 / 3)
    assert agg["first_token_matches"] == 1
    assert agg["avg_greedy_lcp"] == pytest.approx(0.5)
    assert delta["delta_new_minus_old"] == pytest.approx((1.75 - 2.5) / 3)
    assert delta["case_wins_new_old_ties"] == {"new": 1, "old": 1, "ties": 0}


def test_q2_valid_external_evidence_emits_valid_artifact():
    rows = [
        score_q2_token_sequence(
            case_id="case_000",
            target_token_ids=[10],
            target_logprobs=[-0.25],
            greedy_token_ids=[10],
        )
    ]

    artifact = make_deepseek_v4_q2_evidence({"package_id": "pkg:test"}, _external(rows))

    assert validate_base(artifact) == []
    assert artifact["rung"] == "Q2"
    assert artifact["status"] == "valid"
    assert artifact["summary"]["cases"] == 1
    assert artifact["summary"]["avg_nll"] == pytest.approx(0.25)
    assert artifact["summary"]["mlx_wheel"] == mlx_wheel_tag()
    assert artifact["artifact_id"] == compute_artifact_id(artifact)


def test_q2_artifact_rejects_bad_scoring_shape():
    rows = [
        score_q2_token_sequence(
            case_id="case_000",
            target_token_ids=[10],
            target_logprobs=[-0.25],
            greedy_token_ids=[10],
        )
    ]
    external = _external(rows)
    external["case_scores"][0]["avg_nll"] = "bad"

    artifact = make_deepseek_v4_q2_evidence({"package_id": "pkg:test"}, external)

    assert artifact["status"] == "invalid"
    assert any(v["code"] == "deepseek_v4.q2.bad_avg_nll" for v in artifact["validation"])


def test_q2_reference_fixture_is_jsonl():
    line = load_q2_prompts()[0]
    encoded = json.loads(json.dumps({"id": line.id, "prompt": line.prompt}))

    assert encoded["id"] == "case_000"
    assert encoded["prompt"]
