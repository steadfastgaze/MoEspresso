from __future__ import annotations

import json

from moespresso.core.artifact import compute_artifact_id, validate_base
from moespresso.correctness.environment import mlx_wheel_tag
from moespresso.correctness.deepseek_v4.parity import (
    DEEPSEEK_V4_Q1_PROMPT_IDS,
    DEEPSEEK_V4_Q1_TOP_LOGPROBS,
    DEEPSEEK_V4_VOCAB_SIZE,
    main,
    make_deepseek_v4_q1_evidence,
    validate_deepseek_v4_q1_evidence,
)


def _top20() -> list[dict]:
    return [
        {
            "token_id": i,
            "token": {"text": str(i), "bytes": list(str(i).encode())},
            "logprob": -float(i),
        }
        for i in range(DEEPSEEK_V4_Q1_TOP_LOGPROBS)
    ]


def _step(i: int, selected: int = 0) -> dict:
    return {
        "step": i,
        "official_selected": {
            "token_id": selected,
            "text": str(selected),
            "bytes": list(str(selected).encode()),
            "logprob": 0.0,
        },
        "candidate_selected": {
            "token_id": selected,
            "text": str(selected),
            "logprob": 0.0,
        },
        "selected_match": True,
        "selected_rank": selected,
        "official_top20_token_ids": list(range(DEEPSEEK_V4_Q1_TOP_LOGPROBS)),
        "candidate_top20_token_ids": list(range(DEEPSEEK_V4_Q1_TOP_LOGPROBS)),
        "top20_overlap_token_ids": list(range(DEEPSEEK_V4_Q1_TOP_LOGPROBS)),
        "top20_overlap_count": DEEPSEEK_V4_Q1_TOP_LOGPROBS,
        "official_logprob_delta_policy": "skipped_sentinel_non_selected",
        "candidate_top20_logprobs": _top20(),
    }


def _valid_external_evidence() -> dict:
    return {
        "family": "deepseek_v4_flash",
        "run": {
            "decode": "greedy",
            "thinking": False,
            "temperature": 0,
            "top_p": 1.0,
            "top_logprobs": DEEPSEEK_V4_Q1_TOP_LOGPROBS,
            "prompt_renderer": "deepseek_v4_dsv4",
        },
        "reference": {
            "kind": "deepseek_official_api_top20",
            "schema": "ds4-official-logprobs-v1",
            "source": "deepseek-official-api",
            "model": "deepseek-v4-flash",
            "endpoint": "https://api.deepseek.com/chat/completions",
            "logprob_policy": "skip_sentinel_non_selected",
        },
        "candidate": {
            "kind": "moespresso_mlx_package",
            "package_dir": "/tmp/package",
            "package_manifest_id": "pkg:abc",
            "family": "deepseek_v4_flash",
        },
        "thresholds": {
            "selected_rank_max": 0,
            "top20_overlap_min": 1,
        },
        "prompts": [
            {
                "id": prompt_id,
                "kind": "short",
                "prompt_file": f"prompts/{prompt_id}.txt",
                "official_file": f"official/{prompt_id}.official.json",
                "prompt_chars": 10,
                "prompt_tokens": 5,
                "candidate_text": "0",
                "finish_reason": "length",
                "steps": [_step(0)],
            }
            for prompt_id in DEEPSEEK_V4_Q1_PROMPT_IDS
        ],
    }


def _codes(findings):
    return {v.code for v in findings}


def test_deepseek_v4_q1_valid_external_evidence_emits_valid_artifact():
    external = _valid_external_evidence()

    findings = validate_deepseek_v4_q1_evidence(external)
    artifact = make_deepseek_v4_q1_evidence({"package_id": "pkg:test"}, external)

    assert findings == []
    assert validate_base(artifact) == []
    assert artifact["rung"] == "Q1"
    assert artifact["status"] == "valid"
    assert artifact["summary"]["prompts"] == len(DEEPSEEK_V4_Q1_PROMPT_IDS)
    assert artifact["summary"]["selected_matches"] == len(DEEPSEEK_V4_Q1_PROMPT_IDS)
    assert artifact["summary"]["mlx_wheel"] == mlx_wheel_tag()
    assert artifact["external_evidence"] == external
    assert artifact["artifact_id"] == compute_artifact_id(artifact)


def test_deepseek_v4_q1_requires_official_top20_reference_contract():
    external = _valid_external_evidence()
    external["reference"]["kind"] = "official_decoded_weight_reference"
    external["reference"]["schema"] = "other"
    external["reference"]["logprob_policy"] = "compare_all"

    findings = validate_deepseek_v4_q1_evidence(external)

    assert "deepseek_v4.q1.reference_kind" in _codes(findings)
    assert "deepseek_v4.q1.reference_schema" in _codes(findings)
    assert "deepseek_v4.q1.reference_logprob_policy" in _codes(findings)


def test_deepseek_v4_q1_requires_fixed_greedy_thinking_off_run():
    external = _valid_external_evidence()
    external["run"] = {
        "decode": "sampling",
        "thinking": True,
        "temperature": 0.7,
        "top_logprobs": 5,
    }

    findings = validate_deepseek_v4_q1_evidence(external)

    assert "deepseek_v4.q1.decode_not_greedy" in _codes(findings)
    assert "deepseek_v4.q1.thinking_not_disabled" in _codes(findings)
    assert "deepseek_v4.q1.temperature_not_zero" in _codes(findings)
    assert "deepseek_v4.q1.bad_top_logprobs" in _codes(findings)


def test_deepseek_v4_q1_requires_all_committed_prompt_ids():
    external = _valid_external_evidence()
    external["prompts"] = external["prompts"][:-1]

    findings = validate_deepseek_v4_q1_evidence(external)

    assert "deepseek_v4.q1.missing_prompt_ids" in _codes(findings)


def test_deepseek_v4_q1_rejects_selected_token_mismatch_and_low_overlap():
    external = _valid_external_evidence()
    step = external["prompts"][0]["steps"][0]
    step["selected_match"] = False
    step["selected_rank"] = 5
    step["top20_overlap_count"] = 0

    findings = validate_deepseek_v4_q1_evidence(external)

    assert "deepseek_v4.q1.selected_token_mismatch" in _codes(findings)
    assert "deepseek_v4.q1.selected_rank_mismatch" in _codes(findings)
    assert "deepseek_v4.q1.top20_overlap_too_low" in _codes(findings)


def test_deepseek_v4_q1_requires_structured_sorted_local_top20():
    external = _valid_external_evidence()
    top20 = external["prompts"][0]["steps"][0]["candidate_top20_logprobs"]
    top20[0] = {"token_id": DEEPSEEK_V4_VOCAB_SIZE, "logprob": -0.1}
    top20[1] = {"token_id": 1, "logprob": "bad"}
    top20[2] = ["not", "an", "object"]
    top20[5]["logprob"] = 1.0

    findings = validate_deepseek_v4_q1_evidence(external)

    assert "deepseek_v4.q1.bad_top20_token_id" in _codes(findings)
    assert "deepseek_v4.q1.bad_top20_logprob" in _codes(findings)
    assert "deepseek_v4.q1.bad_top20_entry" in _codes(findings)
    assert "deepseek_v4.q1.unsorted_top20_logprobs" in _codes(findings)


def test_deepseek_v4_q1_artifact_sanitizes_nonfinite_logprobs():
    external = _valid_external_evidence()
    top20 = external["prompts"][0]["steps"][0]["candidate_top20_logprobs"]
    top20[0]["logprob"] = float("nan")

    artifact = make_deepseek_v4_q1_evidence({"package_id": "pkg:test"}, external)

    assert artifact["status"] == "invalid"
    assert artifact["external_evidence"]["prompts"][0]["steps"][0][
        "candidate_top20_logprobs"
    ][0]["logprob"] is None
    assert any(
        v["code"] == "deepseek_v4.q1.bad_top20_logprob"
        for v in artifact["validation"]
    )
    assert artifact["artifact_id"] == compute_artifact_id(artifact)


def test_deepseek_v4_q1_cli_writes_valid_artifact(tmp_path, capsys):
    evidence = tmp_path / "q1.json"
    out = tmp_path / "correctness.json"
    evidence.write_text(json.dumps(_valid_external_evidence()), encoding="utf-8")

    rc = main([str(evidence), "--out", str(out)])

    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert artifact["status"] == "valid"
    assert artifact["artifact_id"] == compute_artifact_id(artifact)
    assert "Q1 evidence valid" in capsys.readouterr().err


def test_deepseek_v4_q1_cli_returns_nonzero_for_invalid_evidence(tmp_path, capsys):
    external = _valid_external_evidence()
    external["reference"]["kind"] = "official_decoded_weight_reference"
    evidence = tmp_path / "q1.json"
    out = tmp_path / "correctness.json"
    evidence.write_text(json.dumps(external), encoding="utf-8")

    rc = main([str(evidence), "--out", str(out)])

    artifact = json.loads(out.read_text(encoding="utf-8"))
    captured = capsys.readouterr()
    assert rc == 1
    assert artifact["status"] == "invalid"
    assert any(v["code"] == "deepseek_v4.q1.reference_kind" for v in artifact["validation"])
    assert "deepseek_v4.q1.reference_kind" in captured.err
