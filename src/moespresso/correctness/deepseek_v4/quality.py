"""Manual DeepSeek-V4 quality gates.

These gates are intentionally not pytest tests and are not wired into `make test`.
They run only through this explicit command.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from moespresso.correctness.deepseek_v4.q2 import (
    OPENROUTER_DS4_FLASH_MODEL,
    OPENROUTER_ENDPOINT,
    Q2_DEFAULT_MAX_TOKENS,
    Q2_DEFAULT_TOP_LOGPROBS,
    Q2_PROMPTS_PATH,
    Q2_REFERENCE_PATH,
    capture_q2_official_reference,
    compare_q2_score_tables,
    encode_no_special,
    load_openrouter_token,
    make_deepseek_v4_q2_evidence,
    read_q2_reference,
    score_model_target_tokens,
    score_q2_token_sequence,
    build_q2_external_evidence,
)
from moespresso.correctness.deepseek_v4.parity import (
    DEEPSEEK_V4_Q1_DEFAULT_THRESHOLDS,
    DEEPSEEK_V4_Q1_TOP_LOGPROBS,
    make_deepseek_v4_q1_evidence,
)
from moespresso.correctness.deepseek_v4.q3 import (
    Q3_FIXTURE_ROOT,
    SYSTEM_PROMPT as Q3_SYSTEM_PROMPT,
    load_q3_facts,
    make_deepseek_v4_q3_evidence,
    make_q3_story,
    q3_external_evidence_from_text,
)
from moespresso.correctness.environment import mlx_wheel_tag
from moespresso.correctness.goldens import q0_deepseek_v4_renderer_tokenizer_goldens
from moespresso.core.artifact import write_artifact

PACKAGE_ENV = "MOESPRESSO_DS4_QUALITY_PACKAGE"
DS4_TEST_VECTOR_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "deepseek_v4" / "test_vectors"
)


def pin_full_expert_residency() -> None:
    """Pin gate runs to fully resident expert pools before the model loads.

    The routed prefill has two numerically valid implementations: the
    full-resident barrier-free sorted GEMM and the segmented per-expert
    path. Which one serves depends on pool residency at call time, and both
    are exact to f32 rounding per operation, but their accumulation orders
    differ enough end to end to flip knife-edge parity steps (measured: the
    same package scores Q1 12/17 from a cold pool and 16/17 warm, each
    bit-reproducible). Gate evidence is comparable across runs and packages
    only at a pinned pool state, so gates prewarm every expert and always
    exercise the full-resident path. An explicit environment override still
    wins.
    """
    os.environ.setdefault("MOESPRESSO_SSD_PREWARM_EXPERTS", "all")


def _package_arg(value: str | None) -> Path:
    package = value or os.environ.get(PACKAGE_ENV)
    if not package:
        raise SystemExit(
            f"provide --package or set {PACKAGE_ENV}; DS4 quality gates never "
            "discover or run packages implicitly"
        )
    path = Path(package)
    if not path.is_dir():
        raise SystemExit(f"DS4 package directory not found: {path}")
    return path


def _emit(evidence: dict, json_out: Path | None) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        write_artifact(json_out, evidence)
    print(json.dumps({
        "rung": evidence.get("rung"),
        "status": evidence.get("status"),
        "summary": evidence.get("summary"),
        "artifact_id": evidence.get("artifact_id"),
    }, indent=2, sort_keys=True))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _single_token_id_from_bytes(tokenizer, raw: list[int]) -> tuple[int | None, str]:
    text = bytes(int(b) for b in raw).decode("utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        return None, text
    return int(token_ids[0]), text


def _official_top_ids(tokenizer, step: dict) -> list[int]:
    out = []
    for item in step.get("top_logprobs", []):
        token_id, _text = _single_token_id_from_bytes(tokenizer, item["token"]["bytes"])
        if token_id is not None:
            out.append(token_id)
    return out


def _q1_step_evidence(
    *,
    tokenizer,
    official_step: dict,
    candidate_token_id: int | None,
    candidate_logprob: float | None,
    candidate_top20: tuple[dict, ...],
) -> dict:
    expected_token_id, expected_text = _single_token_id_from_bytes(
        tokenizer,
        official_step["token"]["bytes"],
    )
    official_top_ids = _official_top_ids(tokenizer, official_step)
    candidate_top_ids = [int(item["token_id"]) for item in candidate_top20]
    selected_rank = (
        candidate_top_ids.index(expected_token_id)
        if expected_token_id in candidate_top_ids
        else None
    )
    overlap = sorted(set(official_top_ids) & set(candidate_top_ids))
    return {
        "step": int(official_step["step"]),
        "official_selected": {
            "token_id": expected_token_id,
            "text": expected_text,
            "bytes": official_step["token"]["bytes"],
            "logprob": official_step.get("logprob"),
        },
        "candidate_selected": {
            "token_id": candidate_token_id,
            "text": (
                tokenizer.decode([candidate_token_id])
                if candidate_token_id is not None
                else ""
            ),
            "logprob": candidate_logprob,
        },
        "selected_match": (
            expected_token_id is not None
            and candidate_token_id is not None
            and int(candidate_token_id) == int(expected_token_id)
        ),
        "selected_rank": selected_rank,
        "official_top20_token_ids": official_top_ids,
        "candidate_top20_token_ids": candidate_top_ids,
        "top20_overlap_token_ids": overlap,
        "top20_overlap_count": len(overlap),
        "official_logprob_delta_policy": "skipped_sentinel_non_selected",
        "candidate_top20_logprobs": [dict(item) for item in candidate_top20],
    }


def q1_deepseek_v4_official_top20_parity(
    package_dir: Path,
    *,
    fixture_root: Path | None = None,
    subject: dict | None = None,
) -> dict:
    from moespresso.runtime.deepseek_v4.renderer import DEEPSEEK_V4_PROMPT_RENDERER
    from moespresso.runtime.http import render_prompt
    from moespresso.runtime.serve import generate_with_metadata, load_served_model

    package_dir = Path(package_dir)
    fixture_root = Path(fixture_root or DS4_TEST_VECTOR_FIXTURE_ROOT)
    vector_manifest = _read_json(fixture_root / "manifest.json")
    pin_full_expert_residency()
    model, tokenizer, manifest = load_served_model(package_dir)

    prompt_rows = []
    for prompt_spec in vector_manifest["prompts"]:
        prompt_path = fixture_root / prompt_spec["prompt_file"]
        official_path = fixture_root / prompt_spec["official_file"]
        prompt_text = prompt_path.read_text(encoding="utf-8")
        official = _read_json(official_path)
        official_steps = official["steps"]
        rendered = render_prompt(
            [{"role": "user", "content": prompt_text}],
            tokenizer,
            template_kwargs={"enable_thinking": False},
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        )
        result = generate_with_metadata(
            model,
            tokenizer,
            rendered,
            max_tokens=len(official_steps),
            temperature=0.0,
            top_p=1.0,
            top_logprobs=DEEPSEEK_V4_Q1_TOP_LOGPROBS,
        )
        steps = []
        for i, official_step in enumerate(official_steps):
            candidate_token_id = (
                result.generated_token_ids[i]
                if i < len(result.generated_token_ids)
                else None
            )
            candidate_logprob = (
                result.token_logprobs[i]
                if i < len(result.token_logprobs)
                else None
            )
            candidate_top20 = (
                result.top_logprobs[i]
                if i < len(result.top_logprobs)
                else ()
            )
            steps.append(_q1_step_evidence(
                tokenizer=tokenizer,
                official_step=official_step,
                candidate_token_id=candidate_token_id,
                candidate_logprob=candidate_logprob,
                candidate_top20=candidate_top20,
            ))
        prompt_rows.append({
            "id": prompt_spec["id"],
            "kind": prompt_spec["kind"],
            "prompt_file": prompt_spec["prompt_file"],
            "official_file": prompt_spec["official_file"],
            "prompt_chars": len(prompt_text),
            "prompt_tokens": result.prompt_tokens,
            "official_message": official.get("message", {}),
            "candidate_text": result.text,
            "finish_reason": result.finish_reason,
            "steps": steps,
        })

    external = {
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
            "schema": vector_manifest["schema"].replace(
                "ds4-test-vector-manifest-v1",
                "ds4-official-logprobs-v1",
            ),
            "source": vector_manifest["source"],
            "model": vector_manifest["model"],
            "endpoint": vector_manifest["endpoint"],
            "logprob_policy": "skip_sentinel_non_selected",
            "fixture_root": str(fixture_root),
        },
        "candidate": {
            "kind": "moespresso_mlx_package",
            "package_dir": str(package_dir),
            "package_manifest_id": manifest.get("artifact_id"),
            "family": manifest.get("architecture", {}).get("family"),
        },
        "thresholds": dict(DEEPSEEK_V4_Q1_DEFAULT_THRESHOLDS),
        "inputs": [
            {"path": str(fixture_root / "manifest.json"), "role": "official_manifest"},
            {"path": str(package_dir / "package_manifest.json"), "role": "candidate_package"},
        ],
        "prompts": prompt_rows,
    }
    return make_deepseek_v4_q1_evidence(
        subject or {"package_dir": str(package_dir), "gate": "Q1"},
        external,
    )


def q3_deepseek_v4_long_context_fact_recall(
    package_dir: Path,
    *,
    fixture_root: Path | None = None,
    max_tokens: int = 256,
    subject: dict | None = None,
) -> dict:
    from moespresso.runtime.deepseek_v4.renderer import DEEPSEEK_V4_PROMPT_RENDERER
    from moespresso.runtime.http import render_prompt
    from moespresso.runtime.serve import generate_with_metadata, load_served_model

    package_dir = Path(package_dir)
    fixture_root = Path(fixture_root or Q3_FIXTURE_ROOT)
    pin_full_expert_residency()
    model, tokenizer, manifest = load_served_model(package_dir)
    story = make_q3_story(load_q3_facts(fixture_root))
    rendered = render_prompt(
        [
            {"role": "system", "content": Q3_SYSTEM_PROMPT},
            {"role": "user", "content": story},
        ],
        tokenizer,
        template_kwargs={"enable_thinking": False},
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    )
    result = generate_with_metadata(
        model,
        tokenizer,
        rendered,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    external = q3_external_evidence_from_text(
        package_dir=package_dir,
        manifest=manifest,
        candidate_text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        finish_reason=result.finish_reason,
        fixture_root=fixture_root,
        max_tokens=max_tokens,
    )
    return make_deepseek_v4_q3_evidence(
        subject or {"package_dir": str(package_dir), "gate": "Q3"},
        external,
    )


def q2_deepseek_v4_target_token_nll(
    package_dir: Path,
    *,
    reference_path: Path | None = None,
    max_cases: int | None = None,
    subject: dict | None = None,
) -> dict:
    from moespresso.runtime.deepseek_v4.renderer import DEEPSEEK_V4_PROMPT_RENDERER
    from moespresso.runtime.http import render_prompt
    from moespresso.runtime.serve import load_served_model

    package_dir = Path(package_dir)
    reference_path = Path(reference_path or Q2_REFERENCE_PATH)
    reference = read_q2_reference(reference_path)
    cases = reference["cases"]
    if max_cases is not None:
        cases = cases[:max_cases]

    pin_full_expert_residency()
    model, tokenizer, manifest = load_served_model(package_dir)
    rows = []
    for case in cases:
        rendered = render_prompt(
            [{"role": "user", "content": case["prompt"]}],
            tokenizer,
            template_kwargs={"enable_thinking": False},
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        )
        prompt_ids = encode_no_special(tokenizer, rendered)
        target_ids = encode_no_special(tokenizer, case["continuation"])
        target_logprobs, greedy_ids = score_model_target_tokens(
            model,
            prompt_ids,
            target_ids,
        )
        rows.append(score_q2_token_sequence(
            case_id=case["id"],
            target_token_ids=target_ids,
            target_logprobs=target_logprobs,
            greedy_token_ids=greedy_ids,
        ))

    external = build_q2_external_evidence(
        package_dir=package_dir,
        manifest=manifest,
        reference_path=reference_path,
        reference=reference,
        case_scores=rows,
    )
    return make_deepseek_v4_q2_evidence(
        subject or {"package_dir": str(package_dir), "gate": "Q2"},
        external,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="gate", required=True)

    q0 = sub.add_parser("q0", help="renderer/tokenizer goldens; no model forward")
    q0.add_argument("--package", help=f"DS4 package path, or {PACKAGE_ENV}")
    q0.add_argument("--fixture-root", type=Path, help="override committed Q0 fixture root")
    q0.add_argument("--json-out", type=Path, help="write full correctness_evidence JSON")

    q1 = sub.add_parser("q1", help="official top-20 parity over committed vectors")
    q1.add_argument("--package", help=f"DS4 package path, or {PACKAGE_ENV}")
    q1.add_argument("--fixture-root", type=Path, help="override committed Q1 fixture root")
    q1.add_argument("--json-out", type=Path, help="write full correctness_evidence JSON")

    q2_capture = sub.add_parser(
        "q2-capture",
        help="capture official Q2 continuations through OpenRouter",
    )
    q2_capture.add_argument("--prompts", type=Path, default=Q2_PROMPTS_PATH)
    q2_capture.add_argument("--out", type=Path, default=Q2_REFERENCE_PATH)
    q2_capture.add_argument("--endpoint", default=OPENROUTER_ENDPOINT)
    q2_capture.add_argument("--model", default=OPENROUTER_DS4_FLASH_MODEL)
    q2_capture.add_argument("--count", type=int, default=100)
    q2_capture.add_argument("--max-tokens", type=int, default=Q2_DEFAULT_MAX_TOKENS)
    q2_capture.add_argument("--top-logprobs", type=int, default=Q2_DEFAULT_TOP_LOGPROBS)

    q2 = sub.add_parser("q2", help="target-token NLL over official continuations")
    q2.add_argument("--package", help=f"DS4 package path, or {PACKAGE_ENV}")
    q2.add_argument("--reference", type=Path, default=Q2_REFERENCE_PATH)
    q2.add_argument("--max-cases", type=int, default=None)
    q2.add_argument("--json-out", type=Path, help="write full correctness_evidence JSON")

    q2_compare = sub.add_parser("q2-compare", help="compare two Q2 score artifacts")
    q2_compare.add_argument("old", type=Path, help="old Q2 correctness artifact or external JSON")
    q2_compare.add_argument("new", type=Path, help="new Q2 correctness artifact or external JSON")
    q2_compare.add_argument("--json-out", type=Path, help="write package-delta JSON")

    q3 = sub.add_parser("q3", help="deterministic long-context fact recall")
    q3.add_argument("--package", help=f"DS4 package path, or {PACKAGE_ENV}")
    q3.add_argument("--fixture-root", type=Path, help="override committed Q3 fixture root")
    q3.add_argument("--max-tokens", type=int, default=256)
    q3.add_argument("--json-out", type=Path, help="write full correctness_evidence JSON")

    args = parser.parse_args(argv)
    # Printed before any model work so an interactive run shows which wheel
    # lattice the anchors should be read against; the same tag lands in each
    # gate's evidence summary as `mlx_wheel`.
    print(f"mlx_wheel: {mlx_wheel_tag()}", flush=True)
    if args.gate == "q0":
        evidence = q0_deepseek_v4_renderer_tokenizer_goldens(
            _package_arg(args.package),
            fixture_root=args.fixture_root,
        )
        _emit(evidence, args.json_out)
        return 0 if evidence["status"] == "valid" else 1
    if args.gate == "q1":
        evidence = q1_deepseek_v4_official_top20_parity(
            _package_arg(args.package),
            fixture_root=args.fixture_root,
        )
        _emit(evidence, args.json_out)
        return 0 if evidence["status"] == "valid" else 1
    if args.gate == "q2-capture":
        token = load_openrouter_token()
        if not token:
            print("SKIPPED: OPENROUTER_TOKEN is not set in the environment or .env")
            return 0
        reference = capture_q2_official_reference(
            prompts_path=args.prompts,
            out_path=args.out,
            api_key=token,
            endpoint=args.endpoint,
            model=args.model,
            count=args.count,
            max_tokens=args.max_tokens,
            top_logprobs=args.top_logprobs,
        )
        print(json.dumps({
            "schema": reference["schema"],
            "source": reference["source"],
            "cases": len(reference["cases"]),
            "out": str(args.out),
        }, indent=2, sort_keys=True))
        return 0
    if args.gate == "q2":
        evidence = q2_deepseek_v4_target_token_nll(
            _package_arg(args.package),
            reference_path=args.reference,
            max_cases=args.max_cases,
        )
        _emit(evidence, args.json_out)
        return 0 if evidence["status"] == "valid" else 1
    if args.gate == "q2-compare":
        def _rows(path: Path) -> list[dict]:
            payload = _read_json(path)
            external = payload.get("external_evidence", payload)
            rows = external.get("case_scores")
            if not isinstance(rows, list):
                raise SystemExit(f"Q2 score file has no case_scores: {path}")
            return rows

        delta = compare_q2_score_tables(_rows(args.old), _rows(args.new))
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(delta, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps({
            "cases": delta["cases"],
            "target_tokens": delta["target_tokens"],
            "old_avg_nll": delta["old_avg_nll"],
            "new_avg_nll": delta["new_avg_nll"],
            "delta_new_minus_old": delta["delta_new_minus_old"],
            "case_wins_new_old_ties": delta["case_wins_new_old_ties"],
        }, indent=2, sort_keys=True))
        return 0
    if args.gate == "q3":
        evidence = q3_deepseek_v4_long_context_fact_recall(
            _package_arg(args.package),
            fixture_root=args.fixture_root,
            max_tokens=args.max_tokens,
        )
        _emit(evidence, args.json_out)
        return 0 if evidence["status"] == "valid" else 1
    raise AssertionError(args.gate)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
