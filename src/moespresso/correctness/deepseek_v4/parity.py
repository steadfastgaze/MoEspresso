"""DeepSeek-V4 Q1 greedy selected-token identity evidence contract.

The old decoded-weight M21 contract required a full official PyTorch reference with
decoded FP4/FP8 weights. Q1 is deliberately cheaper and checks the real MoEspresso
serve path against captured official DeepSeek-V4-Flash generation vectors.

**The bar is selected-token identity.** Every comparable reference step must be
the local rank-0 token. A step is comparable when the reference's selected token
resolves to exactly one token under the package tokenizer; a candidate that stops
before the reference ran out of steps fails those steps rather than skipping them,
so the bar cannot be lowered by generating less.

**The top-20 overlap clause is retired.** The provider stack accepts
`top_logprobs: 20` and returns sentinel values: the selected entry carries a real
number and every other candidate sits at `-9999`, degrading to token-id-ordered
filler. An overlap of at least one is therefore satisfied by the selected token
alone and measures nothing. The candidate and reference top-20 id sets and their
overlap are still recorded in the evidence, because they are what the fixture
holds and they cost nothing to persist, but no threshold reads them. Calibrated
distribution questions belong to the KL panel in `q4.py`, which scores against
locally streamed teacher logits instead of provider logprobs.

**The step count comes from the fixture.** It is not a constant: the same five
prompts yield different step totals on different checkpoints, because a
checkpoint that selects an end-of-sequence token earlier simply records fewer
steps at the same `max_tokens`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from moespresso.core.artifact import Validation, make_artifact, write_artifact
from moespresso.correctness.environment import mlx_wheel_tag
from moespresso.correctness.ladder import PRODUCER

DEEPSEEK_V4_Q1_PROMPT_IDS = (
    "short_italian_fact",
    "short_code_completion",
    "short_reasoning_plain",
    "long_memory_archive",
    "long_code_audit",
)
DEEPSEEK_V4_Q1_TOP_LOGPROBS = 20
DEEPSEEK_V4_VOCAB_SIZE = 129280
DEEPSEEK_V4_Q1_DEFAULT_THRESHOLDS = {
    "selected_rank_max": 0,
    "selected_identity_over_comparable_steps": True,
}
# Retired threshold keys. Evidence written before the redesign still carries
# them; they are reported once and never gate.
DEEPSEEK_V4_Q1_RETIRED_THRESHOLDS = ("top20_overlap_min",)
# Reference identities the gate accepts. Upstream's published suite names the
# undated model. A self-captured reference names the dated slug it sent, and the
# provider's own generation record reads back the canonical dated id; the three
# are different naming conventions for a reference of record, not different
# contracts. The undated slug is not a checkpoint identity and must never be
# used to *capture* a reference.
DEEPSEEK_V4_Q1_REFERENCE_MODELS = (
    "deepseek-v4-flash",
    "deepseek/deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash-20260731",
)


def _blocking(code: str, message: str, *, path: str, expected=None, actual=None) -> Validation:
    return Validation(
        "error",
        code,
        message,
        path=path,
        phase="Q1",
        blocking=True,
        expected=_json_safe(expected),
        actual=_json_safe(actual),
    )


def _note(code: str, message: str, *, path: str, expected=None, actual=None) -> Validation:
    return Validation(
        "info",
        code,
        message,
        path=path,
        phase="Q1",
        blocking=False,
        expected=_json_safe(expected),
        actual=_json_safe(actual),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_json_safe(v) for v in value)
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_run_config(evidence: dict) -> list[Validation]:
    run = evidence.get("run")
    if not isinstance(run, dict):
        return [_blocking(
            "deepseek_v4.q1.missing_run_config",
            "Q1 evidence must describe the greedy, thinking-off generation run",
            path="/run",
            expected="object",
            actual=type(run).__name__,
        )]

    out: list[Validation] = []
    if run.get("decode") != "greedy":
        out.append(_blocking(
            "deepseek_v4.q1.decode_not_greedy",
            "Q1 must use greedy decode",
            path="/run/decode",
            expected="greedy",
            actual=run.get("decode"),
        ))
    if run.get("thinking") is not False:
        out.append(_blocking(
            "deepseek_v4.q1.thinking_not_disabled",
            "Q1 must run with thinking disabled",
            path="/run/thinking",
            expected=False,
            actual=run.get("thinking"),
        ))
    if run.get("top_logprobs") != DEEPSEEK_V4_Q1_TOP_LOGPROBS:
        out.append(_blocking(
            "deepseek_v4.q1.bad_top_logprobs",
            "Q1 must capture top-20 local logprobs",
            path="/run/top_logprobs",
            expected=DEEPSEEK_V4_Q1_TOP_LOGPROBS,
            actual=run.get("top_logprobs"),
        ))
    if run.get("temperature") != 0:
        out.append(_blocking(
            "deepseek_v4.q1.temperature_not_zero",
            "Q1 must use temperature 0",
            path="/run/temperature",
            expected=0,
            actual=run.get("temperature"),
        ))
    return out


def _validate_reference(evidence: dict) -> list[Validation]:
    reference = evidence.get("reference")
    if not isinstance(reference, dict):
        return [_blocking(
            "deepseek_v4.q1.missing_reference",
            "Q1 evidence must describe the committed official top-20 reference",
            path="/reference",
            expected="object",
            actual=type(reference).__name__,
        )]

    out: list[Validation] = []
    expected = {
        "kind": "deepseek_official_api_top20",
        "schema": "ds4-official-logprobs-v1",
        "source": "deepseek-official-api",
    }
    for key, value in expected.items():
        if reference.get(key) != value:
            out.append(_blocking(
                f"deepseek_v4.q1.reference_{key}",
                f"Q1 reference must declare {key}={value!r}",
                path=f"/reference/{key}",
                expected=value,
                actual=reference.get(key),
            ))
    if reference.get("model") not in DEEPSEEK_V4_Q1_REFERENCE_MODELS:
        out.append(_blocking(
            "deepseek_v4.q1.reference_model",
            "Q1 reference must name an accepted DeepSeek-V4-Flash reference identity",
            path="/reference/model",
            expected=list(DEEPSEEK_V4_Q1_REFERENCE_MODELS),
            actual=reference.get("model"),
        ))
    if reference.get("logprob_policy") != "skip_sentinel_non_selected":
        out.append(_blocking(
            "deepseek_v4.q1.reference_logprob_policy",
            "Q1 must not treat -9999 non-selected official entries as calibrated logprobs",
            path="/reference/logprob_policy",
            expected="skip_sentinel_non_selected",
            actual=reference.get("logprob_policy"),
        ))
    return out


def _validate_candidate(evidence: dict) -> list[Validation]:
    candidate = evidence.get("candidate")
    if not isinstance(candidate, dict):
        return [_blocking(
            "deepseek_v4.q1.missing_candidate",
            "Q1 evidence must describe the MoEspresso package candidate",
            path="/candidate",
            expected="object",
            actual=type(candidate).__name__,
        )]
    if candidate.get("kind") != "moespresso_mlx_package":
        return [_blocking(
            "deepseek_v4.q1.candidate_kind",
            "Q1 must judge the real MoEspresso MLX package runtime",
            path="/candidate/kind",
            expected="moespresso_mlx_package",
            actual=candidate.get("kind"),
        )]
    return []


def _validate_top20(entries: Any, *, path: str) -> list[Validation]:
    if not isinstance(entries, list) or len(entries) != DEEPSEEK_V4_Q1_TOP_LOGPROBS:
        return [_blocking(
            "deepseek_v4.q1.bad_candidate_top20",
            "each Q1 step must carry exactly 20 local top-logprob entries",
            path=path,
            expected=f"{DEEPSEEK_V4_Q1_TOP_LOGPROBS} entries",
            actual=len(entries) if isinstance(entries, list) else type(entries).__name__,
        )]

    out: list[Validation] = []
    previous_logprob: float | None = None
    seen: set[int] = set()
    for i, entry in enumerate(entries):
        entry_path = f"{path}/{i}"
        if not isinstance(entry, dict):
            out.append(_blocking(
                "deepseek_v4.q1.bad_top20_entry",
                "local top-logprob entries must be objects",
                path=entry_path,
                expected="object with token_id and logprob",
                actual=type(entry).__name__,
            ))
            continue
        token_id = _int_value(entry.get("token_id"))
        if token_id is None or token_id < 0 or token_id >= DEEPSEEK_V4_VOCAB_SIZE:
            out.append(_blocking(
                "deepseek_v4.q1.bad_top20_token_id",
                "local top-logprob entry has no valid DS4 token id",
                path=f"{entry_path}/token_id",
                expected=f"integer in [0, {DEEPSEEK_V4_VOCAB_SIZE})",
                actual=entry.get("token_id"),
            ))
        elif token_id in seen:
            out.append(_blocking(
                "deepseek_v4.q1.duplicate_top20_token_id",
                "local top-logprob entries must not repeat token ids",
                path=f"{entry_path}/token_id",
                expected="unique token id",
                actual=token_id,
            ))
        else:
            seen.add(token_id)
        logprob = _finite_number(entry.get("logprob"))
        if logprob is None:
            out.append(_blocking(
                "deepseek_v4.q1.bad_top20_logprob",
                "local top-logprob entry has no finite logprob",
                path=f"{entry_path}/logprob",
                expected="finite number",
                actual=entry.get("logprob"),
            ))
            continue
        if previous_logprob is not None and logprob > previous_logprob:
            out.append(_blocking(
                "deepseek_v4.q1.unsorted_top20_logprobs",
                "local top-logprob entries must be sorted best-first",
                path=f"{entry_path}/logprob",
                expected=f"<= previous logprob {previous_logprob}",
                actual=logprob,
            ))
        previous_logprob = logprob
    return out


def _validate_prompt_row(row: Any, *, index: int, thresholds: dict) -> list[Validation]:
    path = f"/prompts/{index}"
    if not isinstance(row, dict):
        return [_blocking(
            "deepseek_v4.q1.bad_prompt_row",
            "Q1 prompt rows must be objects",
            path=path,
            expected="object",
            actual=type(row).__name__,
        )]

    out: list[Validation] = []
    steps = row.get("steps")
    if not isinstance(steps, list) or not steps:
        return [_blocking(
            "deepseek_v4.q1.bad_steps",
            "Q1 prompt rows must carry step evidence",
            path=f"{path}/steps",
            expected="non-empty list",
            actual=type(steps).__name__,
        )]

    selected_rank_max = _int_value(thresholds.get("selected_rank_max"))
    if selected_rank_max is None:
        selected_rank_max = DEEPSEEK_V4_Q1_DEFAULT_THRESHOLDS["selected_rank_max"]

    # The reference declares how many steps it holds. A run that scored a
    # different number scored a different fixture, so the two are cross-checked
    # rather than trusting whichever one the summary happened to read.
    declared_steps = _int_value(row.get("declared_steps"))
    if declared_steps is not None and declared_steps != len(steps):
        out.append(_blocking(
            "deepseek_v4.q1.step_count_mismatch",
            "Q1 scored a different number of steps than the reference declares",
            path=f"{path}/steps",
            expected=declared_steps,
            actual=len(steps),
        ))

    for step_index, step in enumerate(steps):
        step_path = f"{path}/steps/{step_index}"
        if not isinstance(step, dict):
            out.append(_blocking(
                "deepseek_v4.q1.bad_step",
                "Q1 step evidence must be an object",
                path=step_path,
                expected="object",
                actual=type(step).__name__,
            ))
            continue
        # Comparability is a property of the reference alone: a reference step
        # whose selected token does not resolve to exactly one token under the
        # package tokenizer cannot be scored as identity. Q0 pins that every
        # reference token does resolve, so a non-comparable step means the
        # package tokenizer is not the checkpoint tokenizer and the gate fails
        # closed instead of quietly scoring fewer steps.
        if step.get("comparable") is False:
            out.append(_blocking(
                "deepseek_v4.q1.reference_step_not_comparable",
                "reference selected token does not resolve to one package token",
                path=f"{step_path}/comparable",
                expected=True,
                actual=step.get("comparable"),
            ))
            continue
        if step.get("selected_match") is not True:
            out.append(_blocking(
                "deepseek_v4.q1.selected_token_mismatch",
                "candidate greedy token must match the official selected token",
                path=f"{step_path}/selected_match",
                expected=True,
                actual=step.get("selected_match"),
            ))
        selected_rank = _int_value(step.get("selected_rank"))
        if selected_rank is None or selected_rank > selected_rank_max:
            out.append(_blocking(
                "deepseek_v4.q1.selected_rank_mismatch",
                "official selected token must appear at the required local rank",
                path=f"{step_path}/selected_rank",
                expected=f"<= {selected_rank_max}",
                actual=step.get("selected_rank"),
            ))
        if step.get("official_logprob_delta_policy") != "skipped_sentinel_non_selected":
            out.append(_blocking(
                "deepseek_v4.q1.logprob_delta_policy",
                "Q1 must explicitly skip official non-selected -9999 sentinel deltas",
                path=f"{step_path}/official_logprob_delta_policy",
                expected="skipped_sentinel_non_selected",
                actual=step.get("official_logprob_delta_policy"),
            ))
        out.extend(_validate_top20(step.get("candidate_top20_logprobs"), path=(
            f"{step_path}/candidate_top20_logprobs"
        )))
    return out


def _validate_prompts(evidence: dict) -> list[Validation]:
    prompts = evidence.get("prompts")
    if not isinstance(prompts, list):
        return [_blocking(
            "deepseek_v4.q1.missing_prompts",
            "Q1 evidence must carry all five prompt rows",
            path="/prompts",
            expected="list",
            actual=type(prompts).__name__,
        )]

    out: list[Validation] = []
    by_id: dict[str, int] = {}
    for i, row in enumerate(prompts):
        if isinstance(row, dict):
            prompt_id = row.get("id")
            if isinstance(prompt_id, str):
                if prompt_id in by_id:
                    out.append(_blocking(
                        "deepseek_v4.q1.duplicate_prompt_id",
                        f"Q1 prompt id {prompt_id!r} appears more than once",
                        path=f"/prompts/{i}/id",
                        expected="unique prompt id",
                        actual=prompt_id,
                    ))
                by_id[prompt_id] = i
        out.extend(_validate_prompt_row(
            row,
            index=i,
            thresholds=evidence.get("thresholds") if isinstance(evidence.get("thresholds"), dict) else {},
        ))

    missing = [p for p in DEEPSEEK_V4_Q1_PROMPT_IDS if p not in by_id]
    if missing:
        out.append(_blocking(
            "deepseek_v4.q1.missing_prompt_ids",
            "Q1 evidence is missing committed official-vector prompts",
            path="/prompts",
            expected=list(DEEPSEEK_V4_Q1_PROMPT_IDS),
            actual=sorted(by_id),
        ))
    return out


def validate_deepseek_v4_q1_evidence(evidence: dict) -> list[Validation]:
    """Return blocking findings for incomplete or failed DS4 Q1 evidence."""
    out: list[Validation] = []
    if evidence.get("family") != "deepseek_v4_flash":
        out.append(_blocking(
            "deepseek_v4.q1.family_mismatch",
            "Q1 evidence must be for the DeepSeek-V4-Flash family",
            path="/family",
            expected="deepseek_v4_flash",
            actual=evidence.get("family"),
        ))
    out.extend(_validate_run_config(evidence))
    out.extend(_validate_reference(evidence))
    out.extend(_validate_candidate(evidence))
    out.extend(_validate_prompts(evidence))
    thresholds = evidence.get("thresholds")
    if isinstance(thresholds, dict):
        for key in DEEPSEEK_V4_Q1_RETIRED_THRESHOLDS:
            if key in thresholds:
                out.append(_note(
                    "deepseek_v4.q1.retired_threshold",
                    f"Q1 no longer gates on {key}; the provider's non-selected "
                    "candidate logprobs are sentinel values",
                    path=f"/thresholds/{key}",
                    expected="absent",
                    actual=thresholds.get(key),
                ))
    return out


def _step_count(evidence: dict) -> int:
    prompts = evidence.get("prompts")
    if not isinstance(prompts, list):
        return 0
    return sum(len(p.get("steps", [])) for p in prompts if isinstance(p, dict))


def _selected_matches(evidence: dict) -> int:
    prompts = evidence.get("prompts")
    if not isinstance(prompts, list):
        return 0
    total = 0
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        for step in prompt.get("steps", []):
            if isinstance(step, dict) and step.get("selected_match") is True:
                total += 1
    return total


def _comparable_steps(evidence: dict) -> int:
    """Count reference steps whose selected token is scoreable as identity.

    A step without an explicit `comparable` field counts as comparable, so
    evidence written before the field existed reads the same way it always did.
    """
    prompts = evidence.get("prompts")
    if not isinstance(prompts, list):
        return 0
    total = 0
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        for step in prompt.get("steps", []):
            if isinstance(step, dict) and step.get("comparable") is not False:
                total += 1
    return total


def make_deepseek_v4_q1_evidence(subject: dict, external_evidence: dict) -> dict:
    """Wrap Q1 run results as a correctness_evidence artifact."""
    external_evidence = _json_safe(external_evidence)
    findings = validate_deepseek_v4_q1_evidence(external_evidence)
    blocking = any(f.blocking for f in findings)
    inputs = external_evidence.get("inputs", []) if isinstance(external_evidence, dict) else []
    steps = _step_count(external_evidence)
    comparable_steps = _comparable_steps(external_evidence)
    selected_matches = _selected_matches(external_evidence)
    return make_artifact(
        "correctness_evidence",
        subject,
        PRODUCER,
        status="invalid" if blocking else "valid",
        validation=findings,
        inputs=inputs,
        rung="Q1",
        summary={
            "findings": len(findings),
            "blocking": sum(1 for f in findings if f.blocking),
            "prompts": len(external_evidence.get("prompts", []))
            if isinstance(external_evidence.get("prompts"), list) else 0,
            "steps": steps,
            "comparable_steps": comparable_steps,
            "selected_matches": selected_matches,
            # Named in the artifact so a reader never has to infer which Q1 bar
            # produced the status.
            "bar": "greedy_selected_token_identity_rank0_all_comparable_steps",
            "top_logprobs": DEEPSEEK_V4_Q1_TOP_LOGPROBS,
            "top20_overlap_gated": False,
            "logprob_delta_policy": "skipped_sentinel_non_selected",
            # The wheel variant keys the quality lattice; record it so a
            # silent reinstall flip is attributable from the artifact alone.
            "mlx_wheel": mlx_wheel_tag(),
        },
        external_evidence=external_evidence,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-q1-validate",
        description="Validate DeepSeek-V4 Q1 greedy selected-token identity evidence.",
    )
    parser.add_argument("evidence", help="external Q1 evidence JSON")
    parser.add_argument("--out", help="write correctness_evidence artifact JSON")
    args = parser.parse_args(argv)

    evidence_path = Path(args.evidence)
    with open(evidence_path, encoding="utf-8") as f:
        external = json.load(f)

    artifact = make_deepseek_v4_q1_evidence({"evidence": str(evidence_path)}, external)
    if args.out:
        write_artifact(args.out, artifact)
    else:
        json.dump(artifact, sys.stdout, sort_keys=True, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")

    blocking = [v for v in artifact["validation"] if v.get("blocking")]
    if blocking:
        for v in blocking:
            print(f"{v['code']}: {v['message']}", file=sys.stderr)
        return 1
    print("deepseek_v4 Q1 evidence valid", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
