"""DeepSeek-V4 Q2 target-token NLL quality gate."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from moespresso.core.artifact import Validation, make_artifact
from moespresso.correctness.deepseek_v4.parity import _json_safe
from moespresso.correctness.environment import mlx_wheel_tag
from moespresso.correctness.ladder import PRODUCER

Q2_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "deepseek_v4"
    / "q2_official_continuations"
)
Q2_PROMPTS_PATH = Q2_FIXTURE_ROOT / "prompts.jsonl"
Q2_PRIVATE_FIXTURE_ROOT = Q2_FIXTURE_ROOT.parent / "private" / "q2_official_continuations"
Q2_REFERENCE_PATH = Q2_PRIVATE_FIXTURE_ROOT / "official_continuations.json"
Q2_REFERENCE_SCHEMA = "ds4-q2-official-continuations-v1"
Q2_EVIDENCE_SCHEMA = "ds4-q2-target-token-nll-v1"
Q2_PROMPT_COUNT = 100
Q2_DEFAULT_MAX_TOKENS = 24
Q2_DEFAULT_TOP_LOGPROBS = 20
Q2_SENTINEL_LOGPROB = -9999.0
Q2_SENTINEL_CUTOFF = -9990.0
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DS4_FLASH_MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_PROVIDER = {
    "only": ["deepseek"],
    "order": ["deepseek"],
    "allow_fallbacks": False,
    "require_parameters": True,
}
OPENROUTER_REASONING_OFF = {"effort": "none"}


@dataclass(frozen=True)
class Q2Prompt:
    id: str
    prompt: str


def _blocking(code: str, message: str, *, path: str, expected=None, actual=None) -> Validation:
    return Validation(
        "error",
        code,
        message,
        path=path,
        phase="Q2",
        blocking=True,
        expected=_json_safe(expected),
        actual=_json_safe(actual),
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def load_q2_prompts(path: Path | None = None) -> list[Q2Prompt]:
    prompt_path = Path(path or Q2_PROMPTS_PATH)
    prompts: list[Q2Prompt] = []
    with prompt_path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = row.get("id")
            prompt = row.get("prompt")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"bad Q2 prompt id at {prompt_path}:{line_no}")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"bad Q2 prompt text at {prompt_path}:{line_no}")
            prompts.append(Q2Prompt(case_id, prompt))
    if not prompts:
        raise ValueError(f"no Q2 prompts found in {prompt_path}")
    ids = [p.id for p in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate Q2 prompt ids in {prompt_path}")
    return prompts


def _token_bytes_from_response_token(token: dict) -> list[int]:
    raw = token.get("bytes")
    if isinstance(raw, list) and all(isinstance(x, int) for x in raw):
        return [int(x) for x in raw]
    text = token.get("token")
    if isinstance(text, str):
        return list(text.encode("utf-8", "replace"))
    return []


def _normalize_top_logprobs(items: Any) -> list[dict]:
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        token = item.get("token", "")
        raw = item.get("bytes")
        if raw is None:
            raw = _token_bytes_from_response_token(item)
        out.append({
            "token": token if isinstance(token, str) else "",
            "bytes": raw if isinstance(raw, list) else [],
            "logprob": item.get("logprob"),
        })
    return out


def official_case_from_response(case_id: str, prompt: str, response: dict) -> dict:
    choice = response["choices"][0]
    message = choice.get("message", {})
    continuation = message.get("content") or ""
    token_rows = choice.get("logprobs", {}).get("content", [])
    official_tokens = []
    for i, row in enumerate(token_rows):
        if not isinstance(row, dict):
            continue
        token = row.get("token", "")
        raw = row.get("bytes")
        if raw is None:
            raw = list(str(token).encode("utf-8", "replace"))
        official_tokens.append({
            "index": i,
            "token": token if isinstance(token, str) else "",
            "bytes": raw if isinstance(raw, list) else [],
            "logprob": row.get("logprob"),
            "top_logprobs": _normalize_top_logprobs(row.get("top_logprobs")),
        })
    return {
        "id": case_id,
        "prompt": prompt,
        "continuation": continuation,
        "finish_reason": choice.get("finish_reason"),
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "official_tokens": official_tokens,
    }


def validate_q2_reference(reference: dict) -> list[Validation]:
    out: list[Validation] = []
    if reference.get("schema") != Q2_REFERENCE_SCHEMA:
        out.append(_blocking(
            "deepseek_v4.q2.reference_schema",
            "Q2 reference must use the official-continuation schema",
            path="/schema",
            expected=Q2_REFERENCE_SCHEMA,
            actual=reference.get("schema"),
        ))
    if reference.get("source") != "deepseek-official-api":
        out.append(_blocking(
            "deepseek_v4.q2.reference_source",
            "Q2 reference must come from the official DeepSeek API route",
            path="/source",
            expected="deepseek-official-api",
            actual=reference.get("source"),
        ))
    capture = reference.get("capture")
    if not isinstance(capture, dict):
        out.append(_blocking(
            "deepseek_v4.q2.missing_capture",
            "Q2 reference must record capture configuration",
            path="/capture",
            expected="object",
            actual=type(capture).__name__,
        ))
    else:
        if capture.get("temperature") != 0:
            out.append(_blocking(
                "deepseek_v4.q2.capture_temperature",
                "Q2 official continuations must be captured greedily",
                path="/capture/temperature",
                expected=0,
                actual=capture.get("temperature"),
            ))
        if capture.get("thinking") is not False:
            out.append(_blocking(
                "deepseek_v4.q2.capture_thinking",
                "Q2 official continuations must be captured with thinking disabled",
                path="/capture/thinking",
                expected=False,
                actual=capture.get("thinking"),
            ))
        top_logprobs = capture.get("top_logprobs")
        if not isinstance(top_logprobs, int) or top_logprobs < 0 or top_logprobs > 20:
            out.append(_blocking(
                "deepseek_v4.q2.capture_top_logprobs",
                "Q2 capture must request a valid top-logprob slice",
                path="/capture/top_logprobs",
                expected="integer in [0, 20]",
                actual=top_logprobs,
            ))

    cases = reference.get("cases")
    if not isinstance(cases, list) or not cases:
        return out + [_blocking(
            "deepseek_v4.q2.missing_cases",
            "Q2 reference must carry official continuation cases",
            path="/cases",
            expected="non-empty list",
            actual=type(cases).__name__,
        )]

    seen: set[str] = set()
    for i, case in enumerate(cases):
        case_path = f"/cases/{i}"
        if not isinstance(case, dict):
            out.append(_blocking(
                "deepseek_v4.q2.bad_case",
                "Q2 cases must be objects",
                path=case_path,
                expected="object",
                actual=type(case).__name__,
            ))
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            out.append(_blocking(
                "deepseek_v4.q2.bad_case_id",
                "Q2 case id must be a non-empty string",
                path=f"{case_path}/id",
                expected="case id string",
                actual=case_id,
            ))
        elif case_id in seen:
            out.append(_blocking(
                "deepseek_v4.q2.duplicate_case_id",
                "Q2 case ids must be unique",
                path=f"{case_path}/id",
                expected="unique id",
                actual=case_id,
            ))
        else:
            seen.add(case_id)
        if not isinstance(case.get("prompt"), str) or not case.get("prompt"):
            out.append(_blocking(
                "deepseek_v4.q2.bad_prompt",
                "Q2 case prompt must be present",
                path=f"{case_path}/prompt",
                expected="non-empty string",
                actual=case.get("prompt"),
            ))
        if not isinstance(case.get("continuation"), str) or not case.get("continuation"):
            out.append(_blocking(
                "deepseek_v4.q2.bad_continuation",
                "Q2 case continuation must be present",
                path=f"{case_path}/continuation",
                expected="non-empty string",
                actual=case.get("continuation"),
            ))
        tokens = case.get("official_tokens")
        if not isinstance(tokens, list) or not tokens:
            out.append(_blocking(
                "deepseek_v4.q2.missing_official_tokens",
                "Q2 official case must carry selected-token logprobs",
                path=f"{case_path}/official_tokens",
                expected="non-empty list",
                actual=type(tokens).__name__,
            ))
            continue
        for j, token in enumerate(tokens):
            token_path = f"{case_path}/official_tokens/{j}"
            if not isinstance(token, dict):
                out.append(_blocking(
                    "deepseek_v4.q2.bad_official_token",
                    "Q2 official token rows must be objects",
                    path=token_path,
                    expected="object",
                    actual=type(token).__name__,
                ))
                continue
            logprob = _finite_number(token.get("logprob"))
            if logprob is None:
                out.append(_blocking(
                    "deepseek_v4.q2.bad_official_logprob",
                    "Q2 official selected-token logprob must be finite",
                    path=f"{token_path}/logprob",
                    expected="finite selected-token logprob",
                    actual=token.get("logprob"),
                ))
            elif logprob <= Q2_SENTINEL_CUTOFF:
                out.append(_blocking(
                    "deepseek_v4.q2.selected_logprob_sentinel",
                    "Q2 must not treat -9999 sentinel logprobs as usable target probabilities",
                    path=f"{token_path}/logprob",
                    expected=f"> {Q2_SENTINEL_CUTOFF}",
                    actual=logprob,
                ))
    return out


def read_q2_reference(path: Path | None = None) -> dict:
    reference_path = Path(path or Q2_REFERENCE_PATH)
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Q2 official continuation reference is not committed: {reference_path}. "
            "Run `moespresso-ds4-quality q2-capture` to create the local ignored "
            "reference, or pass `--reference` explicitly."
        )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    findings = validate_q2_reference(reference)
    blocking = [v for v in findings if v.blocking]
    if blocking:
        codes = ", ".join(v.code for v in blocking[:5])
        raise ValueError(f"invalid Q2 reference {reference_path}: {codes}")
    return reference


def score_q2_token_sequence(
    *,
    case_id: str,
    target_token_ids: list[int],
    target_logprobs: list[float],
    greedy_token_ids: list[int],
) -> dict:
    if not target_token_ids:
        raise ValueError(f"Q2 case {case_id} has no target tokens")
    if len(target_token_ids) != len(target_logprobs):
        raise ValueError(f"Q2 case {case_id} target/logprob length mismatch")
    if len(target_token_ids) != len(greedy_token_ids):
        raise ValueError(f"Q2 case {case_id} target/greedy length mismatch")
    for i, logprob in enumerate(target_logprobs):
        if not math.isfinite(float(logprob)):
            raise ValueError(f"Q2 case {case_id} non-finite local logprob at token {i}")

    first_match = int(greedy_token_ids[0] == target_token_ids[0])
    greedy_lcp = 0
    for target, greedy in zip(target_token_ids, greedy_token_ids, strict=True):
        if int(target) != int(greedy):
            break
        greedy_lcp += 1
    nll = -sum(float(v) for v in target_logprobs)
    return {
        "id": case_id,
        "target_tokens": len(target_token_ids),
        "nll": nll,
        "avg_nll": nll / len(target_token_ids),
        "first_match": first_match,
        "greedy_lcp": greedy_lcp,
        "target_token_ids": [int(x) for x in target_token_ids],
        "target_logprobs": [float(x) for x in target_logprobs],
        "greedy_token_ids": [int(x) for x in greedy_token_ids],
    }


def aggregate_q2_scores(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot aggregate empty Q2 score rows")
    tokens = sum(int(row["target_tokens"]) for row in rows)
    if tokens <= 0:
        raise ValueError("cannot aggregate Q2 rows with zero target tokens")
    nll = sum(float(row["nll"]) for row in rows)
    return {
        "cases": len(rows),
        "target_tokens": tokens,
        "nll": nll,
        "avg_nll": nll / tokens,
        "first_token_matches": sum(int(row["first_match"]) for row in rows),
        "avg_greedy_lcp": sum(int(row["greedy_lcp"]) for row in rows) / len(rows),
    }


def compare_q2_score_tables(old_rows: list[dict], new_rows: list[dict]) -> dict:
    old = {row["id"]: row for row in old_rows}
    new = {row["id"]: row for row in new_rows}
    ids = sorted(set(old) & set(new))
    if not ids:
        raise ValueError("no common Q2 cases")
    old_total = new_total = 0.0
    tokens = 0
    new_wins = old_wins = ties = 0
    deltas = []
    for case_id in ids:
        old_row = old[case_id]
        new_row = new[case_id]
        if int(old_row["target_tokens"]) != int(new_row["target_tokens"]):
            raise ValueError(f"Q2 token-count mismatch for {case_id}")
        tokens += int(old_row["target_tokens"])
        old_total += float(old_row["nll"])
        new_total += float(new_row["nll"])
        delta = float(new_row["nll"]) - float(old_row["nll"])
        deltas.append({
            "id": case_id,
            "delta_nll": delta,
            "tokens": int(old_row["target_tokens"]),
            "old_avg_nll": float(old_row["avg_nll"]),
            "new_avg_nll": float(new_row["avg_nll"]),
        })
        if delta < -1e-9:
            new_wins += 1
        elif delta > 1e-9:
            old_wins += 1
        else:
            ties += 1
    return {
        "cases": len(ids),
        "target_tokens": tokens,
        "old_avg_nll": old_total / tokens,
        "new_avg_nll": new_total / tokens,
        "delta_new_minus_old": (new_total - old_total) / tokens,
        "relative_nll_change_pct": ((new_total / old_total) - 1.0) * 100.0
        if old_total else None,
        "case_wins_new_old_ties": {
            "new": new_wins,
            "old": old_wins,
            "ties": ties,
        },
        "case_deltas": sorted(deltas, key=lambda row: row["delta_nll"]),
    }


def validate_deepseek_v4_q2_evidence(evidence: dict) -> list[Validation]:
    out: list[Validation] = []
    if evidence.get("family") != "deepseek_v4_flash":
        out.append(_blocking(
            "deepseek_v4.q2.family_mismatch",
            "Q2 evidence must be for the DeepSeek-V4-Flash family",
            path="/family",
            expected="deepseek_v4_flash",
            actual=evidence.get("family"),
        ))
    if evidence.get("schema") != Q2_EVIDENCE_SCHEMA:
        out.append(_blocking(
            "deepseek_v4.q2.evidence_schema",
            "Q2 evidence must use the target-token NLL schema",
            path="/schema",
            expected=Q2_EVIDENCE_SCHEMA,
            actual=evidence.get("schema"),
        ))
    run = evidence.get("run")
    if not isinstance(run, dict):
        out.append(_blocking(
            "deepseek_v4.q2.missing_run_config",
            "Q2 evidence must describe the scoring run",
            path="/run",
            expected="object",
            actual=type(run).__name__,
        ))
    else:
        if run.get("decode") != "teacher_forced_target_nll":
            out.append(_blocking(
                "deepseek_v4.q2.decode_mode",
                "Q2 must score the official continuation by teacher forcing",
                path="/run/decode",
                expected="teacher_forced_target_nll",
                actual=run.get("decode"),
            ))
        if run.get("thinking") is not False:
            out.append(_blocking(
                "deepseek_v4.q2.thinking_not_disabled",
                "Q2 must score thinking-off references",
                path="/run/thinking",
                expected=False,
                actual=run.get("thinking"),
            ))

    reference = evidence.get("reference")
    if not isinstance(reference, dict):
        out.append(_blocking(
            "deepseek_v4.q2.missing_reference",
            "Q2 evidence must describe the official continuation reference",
            path="/reference",
            expected="object",
            actual=type(reference).__name__,
        ))
    else:
        if reference.get("kind") != "deepseek_official_api_continuations":
            out.append(_blocking(
                "deepseek_v4.q2.reference_kind",
                "Q2 reference must be official API continuations",
                path="/reference/kind",
                expected="deepseek_official_api_continuations",
                actual=reference.get("kind"),
            ))
        if reference.get("schema") != Q2_REFERENCE_SCHEMA:
            out.append(_blocking(
                "deepseek_v4.q2.reference_schema",
                "Q2 evidence must identify the official-continuation schema",
                path="/reference/schema",
                expected=Q2_REFERENCE_SCHEMA,
                actual=reference.get("schema"),
            ))

    candidate = evidence.get("candidate")
    if not isinstance(candidate, dict):
        out.append(_blocking(
            "deepseek_v4.q2.missing_candidate",
            "Q2 evidence must describe the MoEspresso package candidate",
            path="/candidate",
            expected="object",
            actual=type(candidate).__name__,
        ))
    elif candidate.get("kind") != "moespresso_mlx_package":
        out.append(_blocking(
            "deepseek_v4.q2.candidate_kind",
            "Q2 must judge the real MoEspresso MLX package runtime",
            path="/candidate/kind",
            expected="moespresso_mlx_package",
            actual=candidate.get("kind"),
        ))

    rows = evidence.get("case_scores")
    if not isinstance(rows, list) or not rows:
        out.append(_blocking(
            "deepseek_v4.q2.missing_case_scores",
            "Q2 evidence must carry per-case scores",
            path="/case_scores",
            expected="non-empty list",
            actual=type(rows).__name__,
        ))
    else:
        for i, row in enumerate(rows):
            path = f"/case_scores/{i}"
            if not isinstance(row, dict):
                out.append(_blocking(
                    "deepseek_v4.q2.bad_case_score",
                    "Q2 case scores must be objects",
                    path=path,
                    expected="object",
                    actual=type(row).__name__,
                ))
                continue
            for key in ("target_tokens", "nll", "avg_nll", "first_match", "greedy_lcp"):
                if _finite_number(row.get(key)) is None:
                    out.append(_blocking(
                        f"deepseek_v4.q2.bad_{key}",
                        f"Q2 case score has no finite {key}",
                        path=f"{path}/{key}",
                        expected="finite number",
                        actual=row.get(key),
                    ))
    summary = evidence.get("score")
    if not isinstance(summary, dict):
        out.append(_blocking(
            "deepseek_v4.q2.missing_score",
            "Q2 evidence must carry aggregate score",
            path="/score",
            expected="object",
            actual=type(summary).__name__,
        ))
    elif _finite_number(summary.get("avg_nll")) is None:
        out.append(_blocking(
            "deepseek_v4.q2.bad_avg_nll",
            "Q2 aggregate score must include finite avg_nll",
            path="/score/avg_nll",
            expected="finite number",
            actual=summary.get("avg_nll"),
        ))
    return out


def make_deepseek_v4_q2_evidence(subject: dict, external_evidence: dict) -> dict:
    external_evidence = _json_safe(external_evidence)
    findings = validate_deepseek_v4_q2_evidence(external_evidence)
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
        rung="Q2",
        summary={
            "findings": len(findings),
            "blocking": sum(1 for f in findings if f.blocking),
            "cases": score.get("cases", 0),
            "target_tokens": score.get("target_tokens", 0),
            "nll": score.get("nll", 0.0),
            "avg_nll": score.get("avg_nll", 0.0),
            "first_token_matches": score.get("first_token_matches", 0),
            "avg_greedy_lcp": score.get("avg_greedy_lcp", 0.0),
            # The wheel variant keys the quality lattice; record it so a
            # silent reinstall flip is attributable from the artifact alone.
            "mlx_wheel": mlx_wheel_tag(),
        },
        external_evidence=external_evidence,
    )


def encode_no_special(tokenizer, text: str) -> list[int]:
    try:
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(x) for x in tokenizer.encode(text)]


def score_model_target_tokens(model, prompt_ids: list[int], target_ids: list[int]) -> tuple[list[float], list[int]]:
    if not prompt_ids:
        raise ValueError("Q2 cannot score an empty rendered prompt")
    if not target_ids:
        raise ValueError("Q2 cannot score an empty continuation")
    import mlx.core as mx

    input_ids = prompt_ids + target_ids[:-1]
    logits = model(mx.array([input_ids], dtype=mx.uint32))
    start = len(prompt_ids) - 1
    stop = start + len(target_ids)
    next_logits = logits[0, start:stop, :].astype(mx.float32)
    logprobs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)
    greedy = mx.argmax(logprobs, axis=-1)
    target = mx.array(target_ids, dtype=mx.uint32)
    selected = logprobs[mx.arange(len(target_ids)), target]
    mx.eval(selected, greedy)
    return [float(x) for x in selected.tolist()], [int(x) for x in greedy.tolist()]


def build_q2_external_evidence(
    *,
    package_dir: Path,
    manifest: dict,
    reference_path: Path,
    reference: dict,
    case_scores: list[dict],
) -> dict:
    return {
        "family": "deepseek_v4_flash",
        "schema": Q2_EVIDENCE_SCHEMA,
        "run": {
            "decode": "teacher_forced_target_nll",
            "thinking": False,
            "temperature": 0,
            "prompt_renderer": "deepseek_v4_dsv4",
        },
        "reference": {
            "kind": "deepseek_official_api_continuations",
            "schema": reference.get("schema"),
            "source": reference.get("source"),
            "model": reference.get("capture", {}).get("model"),
            "endpoint": reference.get("capture", {}).get("endpoint"),
            "path": str(reference_path),
            "cases": len(reference.get("cases", [])),
        },
        "candidate": {
            "kind": "moespresso_mlx_package",
            "package_dir": str(package_dir),
            "package_manifest_id": manifest.get("artifact_id"),
            "family": manifest.get("architecture", {}).get("family"),
        },
        "inputs": [
            {"path": str(reference_path), "role": "official_q2_reference"},
            {"path": str(Path(package_dir) / "package_manifest.json"), "role": "candidate_package"},
        ],
        "score": aggregate_q2_scores(case_scores),
        "case_scores": case_scores,
    }


def load_openrouter_token(dotenv_path: Path = Path(".env")) -> str | None:
    token = os.environ.get("OPENROUTER_TOKEN")
    if token:
        return token
    if not dotenv_path.is_file():
        return None
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENROUTER_TOKEN":
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def request_openrouter_q2_case(
    *,
    api_key: str,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    top_logprobs: int,
    provider: dict,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "stream": False,
        "provider": provider,
        "reasoning": OPENROUTER_REASONING_OFF,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as fp:
        return json.loads(fp.read().decode("utf-8"))


def request_openrouter_q2_case_with_retry(**kwargs) -> dict:
    delay = 1.0
    last: BaseException | None = None
    for attempt in range(6):
        try:
            return request_openrouter_q2_case(**kwargs)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code < 500 and e.code != 429:
                raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}") from e
            last = RuntimeError(f"OpenRouter HTTP {e.code}: {body}")
        except Exception as e:  # noqa: BLE001 - command-line retry wrapper.
            last = e
        if attempt == 5:
            assert last is not None
            raise last
        time.sleep(delay)
        delay *= 1.7
    raise AssertionError("unreachable")


def capture_q2_official_reference(
    *,
    prompts_path: Path,
    out_path: Path,
    api_key: str,
    endpoint: str = OPENROUTER_ENDPOINT,
    model: str = OPENROUTER_DS4_FLASH_MODEL,
    count: int = Q2_PROMPT_COUNT,
    max_tokens: int = Q2_DEFAULT_MAX_TOKENS,
    top_logprobs: int = Q2_DEFAULT_TOP_LOGPROBS,
    provider: dict | None = None,
) -> dict:
    prompts = load_q2_prompts(prompts_path)
    provider = dict(provider or OPENROUTER_PROVIDER)
    total = min(int(count), len(prompts))
    cases = []
    for i, prompt in enumerate(prompts[:total]):
        print(f"official Q2 {i + 1}/{total}: {prompt.id}", flush=True)
        response = request_openrouter_q2_case_with_retry(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            prompt=prompt.prompt,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            provider=provider,
        )
        cases.append(official_case_from_response(prompt.id, prompt.prompt, response))
        time.sleep(0.05)
    reference = {
        "schema": Q2_REFERENCE_SCHEMA,
        "source": "deepseek-official-api",
        "capture": {
            "transport": "openrouter",
            "endpoint": endpoint,
            "model": model,
            "provider": provider,
            "temperature": 0,
            "thinking": False,
            "reasoning": OPENROUTER_REASONING_OFF,
            "max_tokens": int(max_tokens),
            "top_logprobs": int(top_logprobs),
            "captured_at": datetime.now(UTC).isoformat(),
            "prompts_path": str(prompts_path),
        },
        "cases": cases,
    }
    findings = validate_q2_reference(reference)
    blocking = [v for v in findings if v.blocking]
    if blocking:
        codes = ", ".join(v.code for v in blocking[:8])
        raise ValueError(f"captured invalid Q2 reference: {codes}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(reference, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return reference
