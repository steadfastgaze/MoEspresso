"""DeepSeek-V4 teacher-forced WikiText perplexity gate.

This is the natural-text arm of the acceptance bar. It exists because the
numbered gates cannot see a numeric blowup that only fires on ordinary prose:
an activation-rotation overflow once produced a non-finite perplexity here
while Q0 through Q3 and every serve probe stayed finite and the code-corpus arm
scored normally. Overflow hides on natural text, so a corpus arm stays in the
bar.

The protocol is fixed and reproducible: the first `--window-count` complete,
contiguous, non-overlapping windows of `--window-size` tokens from a
digest-pinned corpus; one independent direct forward per window with no cache,
so windows cannot leak state into each other; float32 logits; per-position loss
`logsumexp(logits) - target_logit`; an MLX sum per window and a Python float sum
across windows.

Two things are deliberately not constants.

**The corpus is configurable and is not vendored.** It is third-party text. The
gate takes `--corpus` or `MOESPRESSO_DS4_WIKITEXT_CORPUS`, verifies the bytes
against a digest, and fails closed naming the expected file and digest when it
is absent. A gate whose corpus path points into an unrelated checkout stops
working the moment that checkout moves. `--corpus-sha256` runs a different
corpus deliberately, and the evidence records which digest produced the number.

The corpus of record is the held-out `test` split of
`Salesforce/wikitext`, `wikitext-2-raw-v1`, distributed as `wiki.test.raw`.
That identity matters: an earlier pin named a file that a third-party
converter ships as calibration data, whose own notice describes it as a
contiguous subset of the **train** split. Scoring a held-out bar on training
text measures the wrong thing, and the campaign's own capture plan shows every
recorded perplexity run already used the test split, so the pin was the only
part that lagged.

**The limit is a required argument.** Perplexity is package dependent, and a
new checkpoint or quantization recipe moves it. A number that means
"acceptable" for one package is not a constant of the instrument, so the caller
declares the bar and the evidence records which bar produced the status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from moespresso.core.artifact import Validation, make_artifact, write_artifact
from moespresso.correctness.deepseek_v4.parity import _json_safe
from moespresso.correctness.deepseek_v4.quality import (
    PACKAGE_ENV,
    pin_full_expert_residency,
)
from moespresso.correctness.environment import mlx_wheel_tag
from moespresso.correctness.ladder import PRODUCER

WIKITEXT_PPL_EVIDENCE_SCHEMA = "ds4-wikitext-tf-ppl-v1"
WIKITEXT_CORPUS_ENV = "MOESPRESSO_DS4_WIKITEXT_CORPUS"

# The corpus of record for this protocol, by content rather than by location.
# The held-out `test` split of `Salesforce/wikitext`, `wikitext-2-raw-v1`, as
# the canonical `wiki.test.raw`. The digest is the public dataset file's, not a
# local artifact's, so any operator can obtain the same bytes. It is identified
# here so an operator can recognise the right file, and it is never copied into
# this repository.
CANONICAL_WIKITEXT_TEST_SHA256 = (
    "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"
)
RECORDED_CORPUS_NAME = "wiki.test.raw"
RECORDED_CORPUS_SHA256 = CANONICAL_WIKITEXT_TEST_SHA256
RECORDED_CORPUS_BYTES = 1_290_590
# Token count of the recorded corpus under the DeepSeek-V4-Flash tokenizer with
# `tokenizer.encode(text)` and default arguments. Checked only when the corpus
# bytes are the recorded ones, where a different count means the tokenizer
# changed underneath the protocol. Recorded by the phase-1 capture plan against
# a pinned tokenizer digest.
RECORDED_CORPUS_TOKEN_COUNT = 287_730

DEFAULT_WINDOW_SIZE = 2048
# The protocol of record scores 32 windows. Every recorded package limit and
# every ladder reading was taken at that count, so a default of 8 produced a
# number that could not be compared against any declared bar without the flag
# being passed. `--window-count` still overrides.
DEFAULT_WINDOW_COUNT = 32


def _blocking(code: str, message: str, *, path: str, expected=None, actual=None) -> Validation:
    return Validation(
        "error",
        code,
        message,
        path=path,
        phase="PPL",
        blocking=True,
        expected=_json_safe(expected),
        actual=_json_safe(actual),
    )


def corpus_absent_message(path: Path | None) -> str:
    """Explain what file the gate wants, by name and digest, and how to point at it."""
    where = f"not found: {path}" if path is not None else "not configured"
    return (
        f"WikiText corpus {where}\n"
        f"Expected file: {RECORDED_CORPUS_NAME}, sha256 {RECORDED_CORPUS_SHA256}, "
        f"{RECORDED_CORPUS_BYTES} bytes.\n"
        f"Pass --corpus <path> or set {WIKITEXT_CORPUS_ENV}. The corpus is "
        "third-party text and is not distributed with this project; pass "
        "--corpus-sha256 to run a different corpus deliberately."
    )


def resolve_corpus_path(value: str | Path | None) -> Path:
    """Return the configured corpus path, failing closed with a named expectation."""
    configured = value or os.environ.get(WIKITEXT_CORPUS_ENV)
    if not configured:
        raise SystemExit(corpus_absent_message(None))
    path = Path(configured)
    if not path.is_file():
        raise SystemExit(corpus_absent_message(path))
    return path


def load_wikitext_corpus(path: Path, *, expected_sha256: str | None = None) -> tuple[str, dict]:
    """Read and digest-verify the corpus, returning its text and provenance.

    `expected_sha256` defaults to the recorded digest. A mismatch is refused
    rather than reported, because a corpus swap changes what every recorded
    number on this protocol means.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(corpus_absent_message(path))
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    expected = expected_sha256 or RECORDED_CORPUS_SHA256
    if actual != expected:
        raise SystemExit(
            f"WikiText corpus digest mismatch for {path}\n"
            f"  expected sha256 {expected}\n"
            f"  actual   sha256 {actual} ({len(payload)} bytes)\n"
            "Scores on different corpus bytes are not comparable. Pass "
            "--corpus-sha256 with the digest of the corpus you intend to run."
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"WikiText corpus is not valid UTF-8: {path}") from exc
    return text, {
        "path": str(path),
        "sha256": actual,
        "bytes": len(payload),
        "matches_recorded_corpus": actual == RECORDED_CORPUS_SHA256,
    }


def wikitext_windows(
    token_ids: list[int],
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_count: int = DEFAULT_WINDOW_COUNT,
) -> list[list[int]]:
    """Return the first complete, contiguous, non-overlapping windows.

    Incomplete trailing material is dropped rather than padded, so every window
    contributes exactly `window_size - 1` scored targets.
    """
    if window_size < 2:
        raise ValueError(f"window_size must be at least 2, got {window_size}")
    if window_count < 1:
        raise ValueError(f"window_count must be at least 1, got {window_count}")
    needed = window_size * window_count
    if len(token_ids) < needed:
        raise SystemExit(
            f"corpus tokenizes to {len(token_ids)} tokens; the protocol needs "
            f"{needed} ({window_count} windows of {window_size})"
        )
    return [
        list(token_ids[i * window_size : (i + 1) * window_size])
        for i in range(window_count)
    ]


def score_wikitext_windows(model: Any, windows: list[list[int]], *, mx: Any) -> list[dict]:
    """Teacher-force each window through an independent direct forward.

    No cache object is created or passed, so no window can carry state into the
    next one and the run is order-independent by construction.
    """
    rows: list[dict] = []
    for index, window in enumerate(windows):
        logits = model(mx.array([window]))
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.astype(mx.float32)
        predicted = logits[0, :-1]
        targets = mx.array(window[1:])
        log_normalizer = mx.logsumexp(predicted, axis=-1)
        target_logits = mx.take_along_axis(predicted, targets[:, None], axis=-1)[:, 0]
        losses = log_normalizer - target_logits
        mx.eval(losses)
        window_nll = float(mx.sum(losses))
        target_count = int(losses.shape[0])
        rows.append({
            "index": index,
            "input_tokens": len(window),
            "target_tokens": target_count,
            "nll": window_nll,
            "avg_nll": window_nll / target_count if target_count else float("nan"),
            "perplexity": (
                math.exp(window_nll / target_count) if target_count else float("nan")
            ),
        })
        mx.clear_cache()
    return rows


def aggregate_wikitext_score(window_rows: list[dict], *, limit: float) -> dict:
    """Sum the per-window totals and compare the aggregate against the declared bar."""
    total_nll = math.fsum(float(row["nll"]) for row in window_rows)
    total_targets = sum(int(row["target_tokens"]) for row in window_rows)
    avg_nll = total_nll / total_targets if total_targets else float("nan")
    perplexity = math.exp(avg_nll) if math.isfinite(avg_nll) else float("nan")
    return {
        "nll": total_nll,
        "avg_nll": avg_nll,
        "perplexity": perplexity,
        "target_tokens": total_targets,
        "windows": len(window_rows),
        "limit": float(limit),
        "passed": bool(math.isfinite(perplexity) and perplexity <= float(limit)),
    }


def validate_wikitext_ppl_evidence(evidence: dict) -> list[Validation]:
    """Return blocking findings for an incomplete or failing WikiText PPL run."""
    out: list[Validation] = []
    if evidence.get("family") != "deepseek_v4_flash":
        out.append(_blocking(
            "deepseek_v4.ppl.family_mismatch",
            "WikiText PPL evidence must be for the DeepSeek-V4-Flash family",
            path="/family",
            expected="deepseek_v4_flash",
            actual=evidence.get("family"),
        ))
    if evidence.get("schema") != WIKITEXT_PPL_EVIDENCE_SCHEMA:
        out.append(_blocking(
            "deepseek_v4.ppl.schema_mismatch",
            f"WikiText PPL evidence must declare schema {WIKITEXT_PPL_EVIDENCE_SCHEMA!r}",
            path="/schema",
            expected=WIKITEXT_PPL_EVIDENCE_SCHEMA,
            actual=evidence.get("schema"),
        ))

    run = evidence.get("run")
    if not isinstance(run, dict):
        out.append(_blocking(
            "deepseek_v4.ppl.missing_run_config",
            "WikiText PPL evidence must describe the teacher-forced run",
            path="/run",
            expected="object",
            actual=type(run).__name__,
        ))
    else:
        if run.get("decode") != "teacher_forced_corpus_ppl":
            out.append(_blocking(
                "deepseek_v4.ppl.decode_not_teacher_forced",
                "WikiText PPL must teacher-force the corpus, never generate",
                path="/run/decode",
                expected="teacher_forced_corpus_ppl",
                actual=run.get("decode"),
            ))
        if run.get("cache") != "none":
            out.append(_blocking(
                "deepseek_v4.ppl.cache_not_disabled",
                "each window must be an independent forward with no cache",
                path="/run/cache",
                expected="none",
                actual=run.get("cache"),
            ))

    corpus = evidence.get("corpus")
    if not isinstance(corpus, dict) or not corpus.get("sha256"):
        out.append(_blocking(
            "deepseek_v4.ppl.missing_corpus_provenance",
            "WikiText PPL evidence must record the corpus digest it scored",
            path="/corpus/sha256",
            expected="sha256 hex digest",
            actual=corpus.get("sha256") if isinstance(corpus, dict) else None,
        ))

    candidate = evidence.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("kind") != "moespresso_mlx_package":
        out.append(_blocking(
            "deepseek_v4.ppl.candidate_kind",
            "WikiText PPL must judge the real MoEspresso MLX package runtime",
            path="/candidate/kind",
            expected="moespresso_mlx_package",
            actual=candidate.get("kind") if isinstance(candidate, dict) else None,
        ))

    windows = evidence.get("windows")
    protocol = evidence.get("protocol") if isinstance(evidence.get("protocol"), dict) else {}
    expected_windows = protocol.get("window_count")
    if not isinstance(windows, list) or not windows:
        out.append(_blocking(
            "deepseek_v4.ppl.missing_windows",
            "WikiText PPL evidence must carry per-window scores",
            path="/windows",
            expected="non-empty list",
            actual=type(windows).__name__,
        ))
    else:
        if expected_windows is not None and len(windows) != expected_windows:
            out.append(_blocking(
                "deepseek_v4.ppl.window_count_mismatch",
                "scored a different number of windows than the protocol declares",
                path="/windows",
                expected=expected_windows,
                actual=len(windows),
            ))
        for i, row in enumerate(windows):
            value = row.get("nll") if isinstance(row, dict) else None
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                # A single non-finite window is the failure mode this gate
                # exists for: it stays visible per window rather than being
                # absorbed into an aggregate.
                out.append(_blocking(
                    "deepseek_v4.ppl.non_finite_window",
                    "a scored window produced a non-finite loss",
                    path=f"/windows/{i}/nll",
                    expected="finite number",
                    actual=value,
                ))

    score = evidence.get("score")
    if not isinstance(score, dict):
        out.append(_blocking(
            "deepseek_v4.ppl.missing_score",
            "WikiText PPL evidence must carry an aggregate score",
            path="/score",
            expected="object",
            actual=type(score).__name__,
        ))
        return out

    limit = score.get("limit")
    if not isinstance(limit, (int, float)) or isinstance(limit, bool) or not math.isfinite(
        float(limit)
    ):
        out.append(_blocking(
            "deepseek_v4.ppl.missing_limit",
            "WikiText PPL is a gate only against a declared limit",
            path="/score/limit",
            expected="finite number",
            actual=limit,
        ))
        limit = None
    perplexity = score.get("perplexity")
    if not isinstance(perplexity, (int, float)) or isinstance(perplexity, bool) or not (
        math.isfinite(float(perplexity))
    ):
        out.append(_blocking(
            "deepseek_v4.ppl.non_finite_perplexity",
            "aggregate perplexity is not finite",
            path="/score/perplexity",
            expected="finite number",
            actual=perplexity,
        ))
    elif limit is not None and float(perplexity) > float(limit):
        out.append(_blocking(
            "deepseek_v4.ppl.above_limit",
            "WikiText perplexity exceeds the declared limit",
            path="/score/perplexity",
            expected=f"<= {float(limit)}",
            actual=float(perplexity),
        ))
    return out


def make_wikitext_ppl_evidence(subject: dict, external_evidence: dict) -> dict:
    """Wrap a WikiText PPL run as a correctness_evidence artifact."""
    external_evidence = _json_safe(external_evidence)
    findings = validate_wikitext_ppl_evidence(external_evidence)
    blocking = any(f.blocking for f in findings)
    score = external_evidence.get("score") if isinstance(
        external_evidence.get("score"), dict
    ) else {}
    return make_artifact(
        "correctness_evidence",
        subject,
        PRODUCER,
        status="invalid" if blocking else "valid",
        validation=findings,
        inputs=external_evidence.get("inputs", []),
        rung="PPL",
        summary={
            "findings": len(findings),
            "blocking": sum(1 for f in findings if f.blocking),
            "perplexity": score.get("perplexity"),
            "avg_nll": score.get("avg_nll"),
            "target_tokens": score.get("target_tokens"),
            "windows": score.get("windows"),
            "limit": score.get("limit"),
            "corpus_sha256": (
                external_evidence.get("corpus", {}).get("sha256")
                if isinstance(external_evidence.get("corpus"), dict)
                else None
            ),
            # The wheel variant keys the numeric lattice; record it so a silent
            # reinstall flip is attributable from the artifact alone.
            "mlx_wheel": mlx_wheel_tag(),
        },
        external_evidence=external_evidence,
    )


def run_wikitext_ppl(
    package_dir: Path,
    *,
    corpus_path: Path,
    limit: float,
    corpus_sha256: str | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_count: int = DEFAULT_WINDOW_COUNT,
    subject: dict | None = None,
) -> dict:
    """Load the package, score the corpus windows, and build the evidence artifact."""
    import mlx.core as mx

    from moespresso.runtime.serve import load_served_model

    package_dir = Path(package_dir)
    corpus_text, corpus = load_wikitext_corpus(corpus_path, expected_sha256=corpus_sha256)
    pin_full_expert_residency()
    model, tokenizer, manifest = load_served_model(package_dir)

    token_ids = [int(token_id) for token_id in tokenizer.encode(corpus_text)]
    if corpus["matches_recorded_corpus"] and len(token_ids) != RECORDED_CORPUS_TOKEN_COUNT:
        raise SystemExit(
            "the recorded corpus tokenizes to "
            f"{len(token_ids)} tokens, expected {RECORDED_CORPUS_TOKEN_COUNT}; "
            "the package tokenizer is not the one this protocol was recorded on"
        )
    windows = wikitext_windows(
        token_ids,
        window_size=window_size,
        window_count=window_count,
    )
    window_rows = score_wikitext_windows(model, windows, mx=mx)
    score = aggregate_wikitext_score(window_rows, limit=limit)

    external = {
        "family": "deepseek_v4_flash",
        "schema": WIKITEXT_PPL_EVIDENCE_SCHEMA,
        "run": {
            "decode": "teacher_forced_corpus_ppl",
            "thinking": False,
            "cache": "none",
            "logits_dtype_for_loss": "float32",
            "loss": "logsumexp(logits) - target_logit",
        },
        "protocol": {
            "window_size": window_size,
            "window_count": window_count,
            "window_layout": "first complete, contiguous, non-overlapping windows",
            "reduction": "MLX sum per window, Python float sum across windows",
            "corpus_tokens": len(token_ids),
            "scored_input_tokens": window_size * window_count,
        },
        "corpus": corpus,
        "candidate": {
            "kind": "moespresso_mlx_package",
            "package_dir": str(package_dir),
            "package_manifest_id": manifest.get("artifact_id"),
            "family": manifest.get("architecture", {}).get("family"),
        },
        "inputs": [
            {"path": corpus["path"], "role": "wikitext_corpus"},
            {"path": str(package_dir / "package_manifest.json"), "role": "candidate_package"},
        ],
        "windows": window_rows,
        "score": score,
    }
    return make_wikitext_ppl_evidence(
        subject or {"package_dir": str(package_dir), "gate": "wikitext_ppl"},
        external,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-wikitext-ppl",
        description=(
            "DeepSeek-V4 teacher-forced WikiText perplexity gate. The corpus is "
            "configured, never vendored; the limit is declared, never assumed."
        ),
    )
    parser.add_argument("--package", help=f"DS4 package path, or {PACKAGE_ENV}")
    parser.add_argument(
        "--corpus",
        help=f"path to the WikiText corpus file, or {WIKITEXT_CORPUS_ENV}",
    )
    parser.add_argument(
        "--corpus-sha256",
        help="expected corpus digest; defaults to the recorded corpus of record",
    )
    parser.add_argument(
        "--limit",
        type=float,
        required=True,
        help=(
            "perplexity bar this run must meet. Package-family dependent, so "
            "the caller declares it and the evidence records it."
        ),
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--window-count", type=int, default=DEFAULT_WINDOW_COUNT)
    parser.add_argument("--json-out", type=Path, help="write full correctness_evidence JSON")
    args = parser.parse_args(argv)

    package = args.package or os.environ.get(PACKAGE_ENV)
    if not package:
        raise SystemExit(
            f"provide --package or set {PACKAGE_ENV}; DS4 quality gates never "
            "discover or run packages implicitly"
        )
    package_dir = Path(package)
    if not package_dir.is_dir():
        raise SystemExit(f"DS4 package directory not found: {package_dir}")
    corpus_path = resolve_corpus_path(args.corpus)

    print(f"mlx_wheel: {mlx_wheel_tag()}", flush=True)
    evidence = run_wikitext_ppl(
        package_dir,
        corpus_path=corpus_path,
        limit=args.limit,
        corpus_sha256=args.corpus_sha256,
        window_size=args.window_size,
        window_count=args.window_count,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        write_artifact(args.json_out, evidence)
    print(json.dumps({
        "rung": evidence.get("rung"),
        "status": evidence.get("status"),
        "summary": evidence.get("summary"),
        "artifact_id": evidence.get("artifact_id"),
    }, indent=2, sort_keys=True))
    return 0 if evidence["status"] == "valid" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
