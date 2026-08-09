"""DeepSeek-V4 Q4: the teacher-forced KL panel.

Q0 through Q3 judge tokens. Q4 judges the distribution behind them, against a
locally streamed teacher rather than against a provider: the hosted stack
returns sentinel values for every non-selected candidate, so a calibrated
distribution can only come from a teacher-forced pass over the unquantized
weights on this host.

The panel reports five things per probe.

1. **Mean KL** over the teacher's top-K support, both sides renormalized within
   that support. Renormalizing makes it a divergence between two distributions
   on one support, so it is non-negative by construction and a negative value is
   a defect rather than a reading.
2. **Top-1 agreement**, over all scored positions and over the conditioned
   subset.
3. **Entropy-conditioned KL**: the same KL restricted to positions in the top
   quintile of teacher entropy. The threshold comes from the teacher alone, so
   the position mask does not move when the candidate changes and two candidate
   arms are conditioned identically.
4. **Marker-mass inflation**: probability mass on overthinking marker tokens,
   teacher and candidate, with the candidate/teacher ratio. Reported per surface
   form and as an aggregate, plus a narrow subset column restricted to the
   hesitation and branch-alternative classes. The narrow column exists because
   the marker set does not survive tokenization uniformly: several candidate
   surface forms have no single-token representation in this vocabulary, so the
   aggregate mixes classes with very different coverage and the narrow subset is
   the comparable fallback.
5. **Free-run length accounting**, supplied by the caller rather than computed
   here, because it needs generation rather than teacher forcing.

**Which thresholds are coded.** Only the ones that are properties of the
instrument rather than of a package: finite and non-negative KL, top-1
agreement inside [0, 1], teacher and candidate dumps covering identical
positions, a top-K mass in (0, 1], finite marker ratios. Every level -- what
mean KL is acceptable, how much marker inflation matters -- is package
dependent, so it lands in structured output for ledger comparison and gates only
against bars the caller declares.

**Dump schema.** `ds4-q4-teacher-dump-v1`, an npz carrying `ids` (C, T),
`top_ids` (C, T, K), `top_logits` (C, T, K), `log_partition` (C, T), and
`argmax` (C, T). It adapts the earlier teacher-match schema, which stored
`top_lp` already renormalized inside the top-K support: storing raw logits with
the full-vocab log-partition instead makes the top-K mass a measured quantity
rather than an assumption, so a probe whose teacher distribution is not covered
by K is visible in the panel. The earlier field name is still read, and evidence
from it records `top_k_renormalized` normalization so the two are never confused.
The candidate dump mirrors `ids` and `top_ids`, carries `top_logits` gathered
at that exact support, and adds its own `log_partition` and `argmax`. Scoring
requires exact equality of the token-position and ordered-support arrays.
Shape equality alone is not evidence that two dumps describe the same panel.

An unbarred or partially barred panel is emitted as `draft` measurement
evidence. A `valid` quality result requires all three declared bars. The CLI
requires the bars unless `--report-only` explicitly requests draft evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from moespresso.core.artifact import Validation, make_artifact, write_artifact
from moespresso.correctness.deepseek_v4.parity import _json_safe
from moespresso.correctness.environment import mlx_wheel_tag
from moespresso.correctness.ladder import PRODUCER

Q4_TEACHER_DUMP_SCHEMA = "ds4-q4-teacher-dump-v1"
Q4_EVIDENCE_SCHEMA = "ds4-q4-kl-panel-v2"
Q4_MARKERS_ENV = "MOESPRESSO_DS4_Q4_MARKERS"
Q4_BAR_KEYS = ("kl_mean_max", "top1_agreement_min", "marker_ratio_max")

# The conditioning definition, not a bar: the top quintile of teacher entropy is
# where quantization divergence concentrates, so the panel reports the whole
# population and that subset side by side.
Q4_ENTROPY_QUANTILE = 0.8
# Marker classes that form the narrow subset column.
Q4_NARROW_MARKER_CLASSES = ("hesitation_interjection", "branch_alternative")
# Numerical slack for the self-consistency checks. A renormalized KL is
# non-negative in exact arithmetic; float64 accumulation over a 64-wide support
# can land a hair below zero.
Q4_NUMERIC_TOLERANCE = 1e-9


def _blocking(code: str, message: str, *, path: str, expected=None, actual=None) -> Validation:
    return Validation(
        "error",
        code,
        message,
        path=path,
        phase="Q4",
        blocking=True,
        expected=_json_safe(expected),
        actual=_json_safe(actual),
    )


def _warning(code: str, message: str, *, path: str, expected=None, actual=None) -> Validation:
    return Validation(
        "warning",
        code,
        message,
        path=path,
        phase="Q4",
        blocking=False,
        expected=_json_safe(expected),
        actual=_json_safe(actual),
    )


@dataclass(frozen=True)
class MarkerSet:
    """Overthinking marker surface forms resolved against the model tokenizer."""

    version: int
    forms: tuple[tuple[str, int], ...]
    narrow_ids: tuple[int, ...]
    source: str

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(token_id for _text, token_id in self.forms)


@dataclass(frozen=True)
class TeacherDump:
    """Teacher top-K support with full-vocab normalization when it is available."""

    ids: np.ndarray
    top_ids: np.ndarray
    top_logprobs: np.ndarray
    argmax: np.ndarray
    normalization: str
    source: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(dim) for dim in self.top_logprobs.shape)  # type: ignore[return-value]

    @property
    def positions(self) -> int:
        chunks, steps, _k = self.shape
        return chunks * steps


@dataclass(frozen=True)
class CandidateDump:
    """Candidate log-probabilities gathered at the teacher's top-K ids."""

    ids: np.ndarray
    top_ids: np.ndarray
    top_logprobs: np.ndarray
    argmax: np.ndarray
    normalization: str
    source: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(dim) for dim in self.top_logprobs.shape)  # type: ignore[return-value]


def load_marker_set(path: str | Path | None) -> MarkerSet:
    """Read a resolved marker set, failing closed on a shape it cannot read.

    The path is always supplied. There is no default location: the marker set is
    campaign material whose home is outside the installed package, and a gate
    that guesses where it lives breaks the moment that guess is wrong.
    """
    if not path:
        raise SystemExit(
            "no marker set configured. Pass --markers <path> or set "
            f"{Q4_MARKERS_ENV}. The file resolves overthinking surface forms "
            "against the model tokenizer and must declare `version`, "
            "`resolved` (text/id pairs) and `class_of_source_word`."
        )
    marker_path = Path(path)
    if not marker_path.is_file():
        raise SystemExit(f"marker set not found: {marker_path}")
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    resolved = payload.get("resolved")
    classes = payload.get("class_of_source_word")
    if not isinstance(resolved, list) or not resolved:
        raise SystemExit(f"marker set has no resolved forms: {marker_path}")
    if not isinstance(classes, dict):
        raise SystemExit(f"marker set has no class_of_source_word map: {marker_path}")

    forms: list[tuple[str, int]] = []
    narrow: list[int] = []
    for entry in resolved:
        text = entry.get("text")
        token_id = entry.get("id")
        if not isinstance(text, str) or not isinstance(token_id, int):
            raise SystemExit(f"marker set has a malformed resolved entry: {entry!r}")
        forms.append((text, token_id))
        # The narrow subset is defined by class, not by a hand-listed id set, so
        # it follows the marker set instead of drifting from it.
        if classes.get(text.strip()) in Q4_NARROW_MARKER_CLASSES:
            narrow.append(token_id)
    return MarkerSet(
        version=int(payload.get("version", 0)),
        forms=tuple(forms),
        narrow_ids=tuple(sorted(set(narrow))),
        source=str(marker_path),
    )


def _renormalize(logprobs: np.ndarray) -> np.ndarray:
    return logprobs - np.logaddexp.reduce(logprobs, axis=-1, keepdims=True)


def _support_logprobs(payload: Any, *, source: str) -> tuple[np.ndarray, str]:
    """Return support log-probabilities and how they were normalized.

    `top_logits` plus `log_partition` gives full-vocab normalization, so the
    top-K mass is measurable. The earlier `top_lp` field is accepted and is
    already renormalized inside the support.
    """
    if "top_logits" in payload:
        logits = np.asarray(payload["top_logits"], dtype=np.float64)
        if "log_partition" not in payload:
            raise SystemExit(
                f"{source} carries top_logits without log_partition; the pair is "
                "what makes the top-K mass measurable"
            )
        partition = np.asarray(payload["log_partition"], dtype=np.float64)
        if partition.shape != logits.shape[:-1]:
            raise SystemExit(
                f"{source} log_partition shape {partition.shape} does not match "
                f"top_logits {logits.shape}"
            )
        return logits - partition[..., None], "full_vocab"
    if "top_lp" in payload:
        return np.asarray(payload["top_lp"], dtype=np.float64), "top_k_renormalized"
    raise SystemExit(f"{source} carries neither top_logits nor top_lp")


def load_teacher_dump(path: str | Path) -> TeacherDump:
    """Read a teacher dump npz in the current or the earlier schema."""
    dump_path = Path(path)
    if not dump_path.is_file():
        raise SystemExit(f"teacher dump not found: {dump_path}")
    with np.load(dump_path) as payload:
        logprobs, normalization = _support_logprobs(payload, source=str(dump_path))
        if "top_ids" not in payload:
            raise SystemExit(f"{dump_path} carries no top_ids")
        if "ids" not in payload:
            raise SystemExit(f"{dump_path} carries no ids")
        ids = np.asarray(payload["ids"], dtype=np.int64)
        top_ids = np.asarray(payload["top_ids"], dtype=np.int64)
        argmax = (
            np.asarray(payload["argmax"], dtype=np.int64)
            if "argmax" in payload
            # Without a recorded argmax the teacher's own top-K best entry is
            # the argmax, because the support is sorted best-first.
            else top_ids[..., 0]
        )
    return teacher_dump_from_arrays(
        ids=ids,
        top_ids=top_ids,
        top_logprobs=logprobs,
        argmax=argmax,
        normalization=normalization,
        source=str(dump_path),
    )


def teacher_dump_from_arrays(
    *,
    ids: np.ndarray,
    top_ids: np.ndarray,
    top_logprobs: np.ndarray,
    argmax: np.ndarray,
    normalization: str = "full_vocab",
    source: str = "arrays",
) -> TeacherDump:
    ids = np.asarray(ids, dtype=np.int64)
    top_ids = np.asarray(top_ids, dtype=np.int64)
    top_logprobs = np.asarray(top_logprobs, dtype=np.float64)
    argmax = np.asarray(argmax, dtype=np.int64)
    if top_ids.shape != top_logprobs.shape:
        raise SystemExit(
            f"teacher top_ids {top_ids.shape} and support {top_logprobs.shape} disagree"
        )
    if ids.shape != top_logprobs.shape[:-1]:
        raise SystemExit(
            f"teacher ids {ids.shape} do not cover {top_logprobs.shape[:-1]}"
        )
    if argmax.shape != top_logprobs.shape[:-1]:
        raise SystemExit(
            f"teacher argmax {argmax.shape} does not cover {top_logprobs.shape[:-1]}"
        )
    return TeacherDump(
        ids=ids,
        top_ids=top_ids,
        top_logprobs=top_logprobs,
        argmax=argmax,
        normalization=normalization,
        source=source,
    )


def load_candidate_dump(path: str | Path) -> CandidateDump:
    """Read a candidate dump npz gathered at the teacher's top-K ids."""
    dump_path = Path(path)
    if not dump_path.is_file():
        raise SystemExit(f"candidate dump not found: {dump_path}")
    with np.load(dump_path) as payload:
        logprobs, normalization = _support_logprobs(payload, source=str(dump_path))
        if "argmax" not in payload:
            raise SystemExit(f"{dump_path} carries no argmax; top-1 agreement needs it")
        if "top_ids" not in payload:
            raise SystemExit(
                f"{dump_path} carries no top_ids; candidate logits cannot be bound "
                "to the teacher support"
            )
        if "ids" not in payload:
            raise SystemExit(
                f"{dump_path} carries no ids; candidate positions cannot be bound "
                "to the teacher input"
            )
        ids = np.asarray(payload["ids"], dtype=np.int64)
        top_ids = np.asarray(payload["top_ids"], dtype=np.int64)
        argmax = np.asarray(payload["argmax"], dtype=np.int64)
    return candidate_dump_from_arrays(
        ids=ids,
        top_ids=top_ids,
        top_logprobs=logprobs,
        argmax=argmax,
        normalization=normalization,
        source=str(dump_path),
    )


def candidate_dump_from_arrays(
    *,
    ids: np.ndarray,
    top_ids: np.ndarray,
    top_logprobs: np.ndarray,
    argmax: np.ndarray,
    normalization: str = "full_vocab",
    source: str = "arrays",
) -> CandidateDump:
    ids = np.asarray(ids, dtype=np.int64)
    top_ids = np.asarray(top_ids, dtype=np.int64)
    top_logprobs = np.asarray(top_logprobs, dtype=np.float64)
    argmax = np.asarray(argmax, dtype=np.int64)
    if top_ids.shape != top_logprobs.shape:
        raise SystemExit(
            f"candidate top_ids {top_ids.shape} and support "
            f"{top_logprobs.shape} disagree"
        )
    if ids.shape != top_logprobs.shape[:-1]:
        raise SystemExit(
            f"candidate ids {ids.shape} do not cover {top_logprobs.shape[:-1]}"
        )
    if argmax.shape != top_logprobs.shape[:-1]:
        raise SystemExit(
            f"candidate argmax {argmax.shape} does not cover "
            f"{top_logprobs.shape[:-1]}"
        )
    return CandidateDump(
        ids=ids,
        top_ids=top_ids,
        top_logprobs=top_logprobs,
        argmax=argmax,
        normalization=normalization,
        source=source,
    )


def _support_sha256(ids: np.ndarray, top_ids: np.ndarray) -> str:
    """Identity of the scored token positions and ordered teacher support."""
    digest = hashlib.sha256(b"moespresso-q4-support-v1\0")
    for value in (ids, top_ids):
        array = np.ascontiguousarray(value, dtype="<i8")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _marker_block(
    *,
    markers: MarkerSet,
    top_ids: np.ndarray,
    teacher_probability: np.ndarray,
    candidate_probability: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Marker occupancy and mass for one position subset.

    Only marker ids inside the teacher's top-K support are observable, so both
    occupancy and mass are reported relative to that support and the panel says
    so rather than implying full-vocab marker mass.
    """
    scored = int(mask.sum())
    forms: list[dict] = []
    for text, token_id in markers.forms:
        member = top_ids == token_id
        present = np.any(member, axis=-1)
        teacher_mass = float(np.mean(np.sum(teacher_probability * member, axis=-1)[mask]))
        candidate_mass = float(np.mean(np.sum(candidate_probability * member, axis=-1)[mask]))
        forms.append({
            "text": text,
            "token_id": int(token_id),
            "occupancy": float(np.mean(present[mask])) if scored else 0.0,
            "teacher_mass": teacher_mass,
            "candidate_mass": candidate_mass,
            "ratio": candidate_mass / teacher_mass if teacher_mass > 0.0 else None,
        })

    def _subset(ids: tuple[int, ...]) -> dict:
        member = np.isin(top_ids, np.asarray(ids, dtype=np.int64))
        present = np.any(member, axis=-1)
        teacher_mass = float(np.mean(np.sum(teacher_probability * member, axis=-1)[mask]))
        candidate_mass = float(np.mean(np.sum(candidate_probability * member, axis=-1)[mask]))
        return {
            "token_ids": len(ids),
            "occupancy": float(np.mean(present[mask])) if scored else 0.0,
            "teacher_mass": teacher_mass,
            "candidate_mass": candidate_mass,
            "ratio": candidate_mass / teacher_mass if teacher_mass > 0.0 else None,
        }

    return {
        "scored_positions": scored,
        "support": "marker ids present in the teacher top-K support",
        "forms": forms,
        "aggregate": _subset(markers.ids),
        "narrow_subset": {
            "classes": list(Q4_NARROW_MARKER_CLASSES),
            **_subset(markers.narrow_ids),
        },
    }


def score_kl_panel(
    teacher: TeacherDump,
    candidate: CandidateDump,
    *,
    markers: MarkerSet,
    probe: str = "probe",
    entropy_quantile: float = Q4_ENTROPY_QUANTILE,
    free_run_records: list[dict] | None = None,
) -> dict:
    """Score one candidate arm against one teacher dump over the same positions."""
    if teacher.shape != candidate.shape:
        raise SystemExit(
            f"teacher dump {teacher.shape} and candidate dump {candidate.shape} "
            "do not cover the same positions"
        )
    if not np.array_equal(teacher.ids, candidate.ids):
        raise SystemExit(
            "teacher and candidate dumps have different token ids at the "
            "scored positions"
        )
    if not np.array_equal(teacher.top_ids, candidate.top_ids):
        raise SystemExit(
            "candidate logits were not gathered on the teacher's ordered "
            "top-K support"
        )
    if not 0.0 <= entropy_quantile < 1.0:
        raise ValueError(f"entropy_quantile must be in [0, 1), got {entropy_quantile}")

    teacher_support = _renormalize(teacher.top_logprobs)
    candidate_support = _renormalize(candidate.top_logprobs)
    teacher_probability_in_support = np.exp(teacher_support)
    entropy = -np.sum(teacher_probability_in_support * teacher_support, axis=-1)
    threshold = float(np.quantile(entropy, entropy_quantile))
    mask = entropy >= threshold
    everywhere = np.ones_like(mask, dtype=bool)

    kl = np.sum(teacher_probability_in_support * (teacher_support - candidate_support), axis=-1)
    top1 = teacher.argmax == candidate.argmax

    # Raw mass, not renormalized: marker inflation is about how much probability
    # the candidate spends on these tokens, which renormalizing would hide.
    teacher_probability = np.exp(teacher.top_logprobs)
    candidate_probability = np.exp(candidate.top_logprobs)
    teacher_top_k_mass = np.sum(teacher_probability, axis=-1)

    panel = {
        "probe": probe,
        "positions": int(entropy.size),
        "support_width": teacher.shape[-1],
        "support_sha256": _support_sha256(teacher.ids, teacher.top_ids),
        "normalization": {
            "teacher": teacher.normalization,
            "candidate": candidate.normalization,
        },
        "teacher_top_k_mass_mean": float(np.mean(teacher_top_k_mass)),
        "entropy": {
            "quantile": float(entropy_quantile),
            "threshold": threshold,
            "conditioned_positions": int(mask.sum()),
            "teacher_mean": float(np.mean(entropy)),
        },
        "kl": {
            "mean": float(np.mean(kl)),
            "mean_top_entropy": float(np.mean(kl[mask])) if mask.any() else None,
            "max": float(np.max(kl)),
        },
        "top1_agreement": {
            "all_positions": float(np.mean(top1)),
            "top_entropy_positions": float(np.mean(top1[mask])) if mask.any() else None,
        },
        "markers": {
            "source": markers.source,
            "version": markers.version,
            "forms": len(markers.forms),
            "narrow_ids": len(markers.narrow_ids),
            "all_positions": _marker_block(
                markers=markers,
                top_ids=teacher.top_ids,
                teacher_probability=teacher_probability,
                candidate_probability=candidate_probability,
                mask=everywhere,
            ),
            "top_entropy_positions": _marker_block(
                markers=markers,
                top_ids=teacher.top_ids,
                teacher_probability=teacher_probability,
                candidate_probability=candidate_probability,
                mask=mask,
            ),
        },
        "free_run": (
            free_run_length_accounting(free_run_records)
            if free_run_records is not None
            else None
        ),
    }
    return panel


def free_run_length_accounting(records: list[dict]) -> dict:
    """Summarize free-run completion lengths beside the teacher-forced panel.

    Length is the symptom the KL columns cannot see: a package can hold its
    distribution and still run long, and a package can shorten because it
    terminates early rather than because it answers concisely. The accounting is
    package dependent, so it reports and never judges. Records carry `id`,
    `candidate_tokens`, an optional `reference_tokens`, and an optional
    `finish_reason`.
    """
    rows: list[dict] = []
    for record in records:
        candidate_tokens = record.get("candidate_tokens")
        if not isinstance(candidate_tokens, int) or isinstance(candidate_tokens, bool):
            raise SystemExit(
                f"free-run record {record.get('id')!r} has no integer candidate_tokens"
            )
        reference_tokens = record.get("reference_tokens")
        rows.append({
            "id": record.get("id"),
            "candidate_tokens": candidate_tokens,
            "reference_tokens": reference_tokens,
            "finish_reason": record.get("finish_reason"),
            "length_ratio": (
                candidate_tokens / reference_tokens
                if isinstance(reference_tokens, int)
                and not isinstance(reference_tokens, bool)
                and reference_tokens > 0
                else None
            ),
        })

    candidate_lengths = [row["candidate_tokens"] for row in rows]
    reference_lengths = [
        row["reference_tokens"] for row in rows if isinstance(row["reference_tokens"], int)
    ]
    finish_reasons: dict[str, int] = {}
    for row in rows:
        reason = row["finish_reason"]
        if reason is not None:
            finish_reasons[str(reason)] = finish_reasons.get(str(reason), 0) + 1
    return {
        "cases": len(rows),
        "candidate_tokens_total": sum(candidate_lengths),
        "candidate_tokens_mean": (
            sum(candidate_lengths) / len(candidate_lengths) if candidate_lengths else None
        ),
        "candidate_tokens_median": _median(candidate_lengths),
        "reference_tokens_total": sum(reference_lengths) if reference_lengths else None,
        "reference_tokens_mean": (
            sum(reference_lengths) / len(reference_lengths) if reference_lengths else None
        ),
        "length_ratio_total": (
            sum(candidate_lengths) / sum(reference_lengths)
            if reference_lengths and sum(reference_lengths) > 0
            else None
        ),
        "finish_reasons": finish_reasons,
        "cases_detail": rows,
    }


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def validate_deepseek_v4_q4_evidence(evidence: dict) -> list[Validation]:
    """Return blocking findings for a malformed, self-inconsistent, or over-bar panel.

    Everything checked without a declared bar is a property of the instrument:
    a renormalized KL cannot be negative, an agreement rate cannot leave [0, 1],
    a top-K mass cannot exceed one, and teacher and candidate must cover the same
    positions. Levels are package dependent and gate only against `bars`.
    """
    out: list[Validation] = []
    if evidence.get("family") != "deepseek_v4_flash":
        out.append(_blocking(
            "deepseek_v4.q4.family_mismatch",
            "Q4 evidence must be for the DeepSeek-V4-Flash family",
            path="/family",
            expected="deepseek_v4_flash",
            actual=evidence.get("family"),
        ))
    if evidence.get("schema") != Q4_EVIDENCE_SCHEMA:
        out.append(_blocking(
            "deepseek_v4.q4.schema_mismatch",
            f"Q4 evidence must declare schema {Q4_EVIDENCE_SCHEMA!r}",
            path="/schema",
            expected=Q4_EVIDENCE_SCHEMA,
            actual=evidence.get("schema"),
        ))

    panels = evidence.get("panels")
    if not isinstance(panels, list) or not panels:
        out.append(_blocking(
            "deepseek_v4.q4.missing_panels",
            "Q4 evidence must carry at least one scored probe panel",
            path="/panels",
            expected="non-empty list",
            actual=type(panels).__name__,
        ))
        return out

    bars = evidence.get("bars") if isinstance(evidence.get("bars"), dict) else {}
    unknown_bars = sorted(set(bars) - set(Q4_BAR_KEYS))
    if unknown_bars:
        out.append(_blocking(
            "deepseek_v4.q4.unknown_bars",
            "Q4 evidence carries unknown quality bars",
            path="/bars",
            expected=list(Q4_BAR_KEYS),
            actual=unknown_bars,
        ))
    invalid_bars = sorted(key for key in Q4_BAR_KEYS if key in bars and _finite(bars[key]) is None)
    if invalid_bars:
        out.append(_blocking(
            "deepseek_v4.q4.invalid_bars",
            "declared Q4 quality bars must be finite numbers",
            path="/bars",
            expected="finite numbers",
            actual={key: bars[key] for key in invalid_bars},
        ))
    bar_values = {key: _finite(bars.get(key)) for key in Q4_BAR_KEYS}
    bad_bar_domains = {
        key: value
        for key, value in bar_values.items()
        if value is not None
        and (
            (key in {"kl_mean_max", "marker_ratio_max"} and value < 0.0)
            or (key == "top1_agreement_min" and not 0.0 <= value <= 1.0)
        )
    }
    if bad_bar_domains:
        out.append(_blocking(
            "deepseek_v4.q4.invalid_bar_domains",
            "declared Q4 quality bars must use the metric's valid domain",
            path="/bars",
            expected={
                "kl_mean_max": ">= 0",
                "top1_agreement_min": "number in [0, 1]",
                "marker_ratio_max": ">= 0",
            },
            actual=bad_bar_domains,
        ))
    missing_bars = [key for key in Q4_BAR_KEYS if _finite(bars.get(key)) is None]
    if missing_bars:
        code = (
            "deepseek_v4.q4.unbarred"
            if len(missing_bars) == len(Q4_BAR_KEYS)
            else "deepseek_v4.q4.incomplete_bars"
        )
        out.append(_warning(
            code,
            "Q4 results without all declared quality bars are measurement "
            "evidence, not a package quality pass",
            path="/bars",
            expected=list(Q4_BAR_KEYS),
            actual=sorted(bars),
        ))
    kl_max = _finite(bars.get("kl_mean_max"))
    top1_min = _finite(bars.get("top1_agreement_min"))
    marker_ratio_max = _finite(bars.get("marker_ratio_max"))

    for index, panel in enumerate(panels):
        path = f"/panels/{index}"
        if not isinstance(panel, dict):
            out.append(_blocking(
                "deepseek_v4.q4.bad_panel",
                "Q4 panels must be objects",
                path=path,
                expected="object",
                actual=type(panel).__name__,
            ))
            continue
        positions = panel.get("positions")
        if not isinstance(positions, int) or isinstance(positions, bool) or positions <= 0:
            out.append(_blocking(
                "deepseek_v4.q4.no_scored_positions",
                "a Q4 panel must score at least one position",
                path=f"{path}/positions",
                expected="positive integer",
                actual=positions,
            ))
        support_sha256 = panel.get("support_sha256")
        if (
            not isinstance(support_sha256, str)
            or len(support_sha256) != 64
            or any(char not in "0123456789abcdef" for char in support_sha256)
        ):
            out.append(_blocking(
                "deepseek_v4.q4.bad_support_identity",
                "each Q4 panel must identify its paired token positions and support",
                path=f"{path}/support_sha256",
                expected="64 lowercase hexadecimal characters",
                actual=support_sha256,
            ))

        kl = panel.get("kl") if isinstance(panel.get("kl"), dict) else {}
        kl_mean = _finite(kl.get("mean"))
        if kl_mean is None:
            out.append(_blocking(
                "deepseek_v4.q4.non_finite_kl",
                "mean KL is not finite",
                path=f"{path}/kl/mean",
                expected="finite number",
                actual=kl.get("mean"),
            ))
        elif kl_mean < -Q4_NUMERIC_TOLERANCE:
            out.append(_blocking(
                "deepseek_v4.q4.negative_kl",
                "a support-renormalized KL cannot be negative; the dumps are "
                "misaligned or a normalization is wrong",
                path=f"{path}/kl/mean",
                expected=">= 0",
                actual=kl_mean,
            ))
        elif kl_max is not None and kl_mean > kl_max:
            out.append(_blocking(
                "deepseek_v4.q4.kl_above_bar",
                "mean KL exceeds the declared bar",
                path=f"{path}/kl/mean",
                expected=f"<= {kl_max}",
                actual=kl_mean,
            ))

        agreement = (
            panel.get("top1_agreement")
            if isinstance(panel.get("top1_agreement"), dict)
            else {}
        )
        top1 = _finite(agreement.get("all_positions"))
        if top1 is None or not 0.0 <= top1 <= 1.0:
            out.append(_blocking(
                "deepseek_v4.q4.bad_top1_agreement",
                "top-1 agreement must be a rate in [0, 1]",
                path=f"{path}/top1_agreement/all_positions",
                expected="number in [0, 1]",
                actual=agreement.get("all_positions"),
            ))
        elif top1_min is not None and top1 < top1_min:
            out.append(_blocking(
                "deepseek_v4.q4.top1_below_bar",
                "top-1 agreement is below the declared bar",
                path=f"{path}/top1_agreement/all_positions",
                expected=f">= {top1_min}",
                actual=top1,
            ))

        mass = _finite(panel.get("teacher_top_k_mass_mean"))
        normalization = (
            panel.get("normalization") if isinstance(panel.get("normalization"), dict) else {}
        )
        if normalization.get("teacher") == "full_vocab":
            if mass is None or not 0.0 < mass <= 1.0 + Q4_NUMERIC_TOLERANCE:
                out.append(_blocking(
                    "deepseek_v4.q4.bad_top_k_mass",
                    "teacher top-K mass must lie in (0, 1]",
                    path=f"{path}/teacher_top_k_mass_mean",
                    expected="number in (0, 1]",
                    actual=panel.get("teacher_top_k_mass_mean"),
                ))

        markers = panel.get("markers") if isinstance(panel.get("markers"), dict) else {}
        for subset_name in ("all_positions", "top_entropy_positions"):
            block = markers.get(subset_name)
            if not isinstance(block, dict):
                out.append(_blocking(
                    "deepseek_v4.q4.missing_marker_block",
                    "each Q4 panel reports marker mass over both position subsets",
                    path=f"{path}/markers/{subset_name}",
                    expected="object",
                    actual=type(block).__name__,
                ))
                continue
            for column in ("aggregate", "narrow_subset"):
                entry = block.get(column)
                if not isinstance(entry, dict):
                    out.append(_blocking(
                        "deepseek_v4.q4.missing_marker_column",
                        f"marker reporting must carry the {column} column",
                        path=f"{path}/markers/{subset_name}/{column}",
                        expected="object",
                        actual=type(entry).__name__,
                    ))
                    continue
                ratio = entry.get("ratio")
                finite_ratio = _finite(ratio)
                if ratio is not None and finite_ratio is None:
                    out.append(_blocking(
                        "deepseek_v4.q4.non_finite_marker_ratio",
                        "a marker inflation ratio is not finite",
                        path=f"{path}/markers/{subset_name}/{column}/ratio",
                        expected="finite number or null",
                        actual=ratio,
                    ))
                elif finite_ratio is not None and finite_ratio < 0.0:
                    out.append(_blocking(
                        "deepseek_v4.q4.negative_marker_ratio",
                        "a marker inflation ratio cannot be negative",
                        path=f"{path}/markers/{subset_name}/{column}/ratio",
                        expected=">= 0",
                        actual=finite_ratio,
                    ))
                elif (
                    subset_name == "top_entropy_positions"
                    and column == "narrow_subset"
                    and marker_ratio_max is not None
                ):
                    if finite_ratio is None:
                        out.append(_blocking(
                            "deepseek_v4.q4.marker_ratio_unavailable",
                            "the declared marker bar requires an evaluated "
                            "top-entropy narrow-subset ratio",
                            path=f"{path}/markers/{subset_name}/{column}/ratio",
                            expected=f"finite number <= {marker_ratio_max}",
                            actual=ratio,
                        ))
                    elif finite_ratio > marker_ratio_max:
                        out.append(_blocking(
                            "deepseek_v4.q4.marker_ratio_above_bar",
                            "narrow-subset marker inflation exceeds the declared bar",
                            path=f"{path}/markers/{subset_name}/{column}/ratio",
                            expected=f"<= {marker_ratio_max}",
                            actual=finite_ratio,
                        ))
    return out


def make_deepseek_v4_q4_evidence(subject: dict, external_evidence: dict) -> dict:
    """Wrap a KL panel run as a correctness_evidence artifact."""
    external_evidence = _json_safe(external_evidence)
    findings = validate_deepseek_v4_q4_evidence(external_evidence)
    blocking = any(f.blocking for f in findings)
    bars = external_evidence.get("bars") if isinstance(
        external_evidence.get("bars"), dict
    ) else {}
    bars_complete = all(_finite(bars.get(key)) is not None for key in Q4_BAR_KEYS)
    status = "invalid" if blocking else "valid" if bars_complete else "draft"
    judgement = (
        "invalid"
        if blocking
        else "quality_bar_passed"
        if bars_complete
        else "instrument_valid_unbarred"
    )
    panels = external_evidence.get("panels") if isinstance(
        external_evidence.get("panels"), list
    ) else []
    return make_artifact(
        "correctness_evidence",
        subject,
        PRODUCER,
        status=status,
        validation=findings,
        inputs=external_evidence.get("inputs", []),
        rung="Q4",
        summary={
            "findings": len(findings),
            "blocking": sum(1 for f in findings if f.blocking),
            "probes": [p.get("probe") for p in panels if isinstance(p, dict)],
            "positions": sum(
                int(p.get("positions", 0)) for p in panels if isinstance(p, dict)
            ),
            "kl_mean": [
                p.get("kl", {}).get("mean") for p in panels if isinstance(p, dict)
            ],
            "top1_agreement": [
                p.get("top1_agreement", {}).get("all_positions")
                for p in panels
                if isinstance(p, dict)
            ],
            "bars": external_evidence.get("bars", {}),
            "judgement": judgement,
            # The wheel variant keys the numeric lattice; record it so a silent
            # reinstall flip is attributable from the artifact alone.
            "mlx_wheel": mlx_wheel_tag(),
        },
        external_evidence=external_evidence,
    )


def build_q4_external_evidence(
    *,
    panels: list[dict],
    teacher_sources: list[str],
    candidate_sources: list[str],
    markers: MarkerSet,
    bars: dict | None = None,
) -> dict:
    return {
        "family": "deepseek_v4_flash",
        "schema": Q4_EVIDENCE_SCHEMA,
        "run": {
            "decode": "teacher_forced_kl_panel",
            "thinking": False,
            "teacher": "locally streamed unquantized forward",
            "kl_support": "teacher top-K, both sides renormalized within it",
        },
        "markers": {
            "source": markers.source,
            "version": markers.version,
            "forms": len(markers.forms),
            "narrow_ids": len(markers.narrow_ids),
            "narrow_classes": list(Q4_NARROW_MARKER_CLASSES),
        },
        "bars": dict(bars or {}),
        "inputs": [
            *[{"path": path, "role": "q4_teacher_dump"} for path in teacher_sources],
            *[{"path": path, "role": "q4_candidate_dump"} for path in candidate_sources],
            {"path": markers.source, "role": "q4_marker_set"},
        ],
        "panels": panels,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-q4",
        description=(
            "DeepSeek-V4 Q4 KL panel. Scores a package's teacher-forced "
            "distribution against a locally streamed teacher dump. Levels are "
            "package dependent and gate only against declared bars."
        ),
    )
    parser.add_argument("--teacher", required=True, type=Path, action="append",
                        help="teacher dump npz; repeat for multiple probes")
    parser.add_argument("--candidate", required=True, type=Path, action="append",
                        help="candidate dump npz, in the same order as --teacher")
    parser.add_argument("--probe", action="append", default=None,
                        help="probe label per dump pair; defaults to the dump stem")
    parser.add_argument("--markers", help=f"resolved marker set JSON, or {Q4_MARKERS_ENV}")
    parser.add_argument("--entropy-quantile", type=float, default=Q4_ENTROPY_QUANTILE)
    parser.add_argument("--free-run", type=Path,
                        help="JSON list of free-run length records for the last probe")
    parser.add_argument("--kl-mean-max", type=float, default=None,
                        help="declared bar: mean KL at or below this value")
    parser.add_argument("--top1-agreement-min", type=float, default=None,
                        help="declared bar: top-1 agreement at or above this value")
    parser.add_argument("--marker-ratio-max", type=float, default=None,
                        help="declared bar: narrow-subset marker inflation ratio ceiling")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="emit draft measurement evidence when quality bars are incomplete",
    )
    parser.add_argument("--json-out", type=Path, help="write full correctness_evidence JSON")
    args = parser.parse_args(argv)

    if len(args.teacher) != len(args.candidate):
        raise SystemExit(
            f"{len(args.teacher)} teacher dumps against {len(args.candidate)} "
            "candidate dumps; the panel scores pairs"
        )
    markers = load_marker_set(args.markers or os.environ.get(Q4_MARKERS_ENV))
    free_run_records = (
        json.loads(args.free_run.read_text(encoding="utf-8"))
        if args.free_run is not None
        else None
    )

    panels = []
    for index, (teacher_path, candidate_path) in enumerate(zip(args.teacher, args.candidate)):
        probe = (
            args.probe[index]
            if args.probe is not None and index < len(args.probe)
            else Path(teacher_path).stem
        )
        panels.append(score_kl_panel(
            load_teacher_dump(teacher_path),
            load_candidate_dump(candidate_path),
            markers=markers,
            probe=probe,
            entropy_quantile=args.entropy_quantile,
            free_run_records=(
                free_run_records if index == len(args.teacher) - 1 else None
            ),
        ))

    bars = {
        key: value
        for key, value in (
            ("kl_mean_max", args.kl_mean_max),
            ("top1_agreement_min", args.top1_agreement_min),
            ("marker_ratio_max", args.marker_ratio_max),
        )
        if value is not None
    }
    missing_bars = [key for key in Q4_BAR_KEYS if key not in bars]
    if missing_bars and not args.report_only:
        parser.error(
            "a Q4 quality pass requires all three declared bars; missing "
            f"{', '.join(missing_bars)}. Use --report-only for unbarred "
            "measurement evidence."
        )
    external = build_q4_external_evidence(
        panels=panels,
        teacher_sources=[str(p) for p in args.teacher],
        candidate_sources=[str(p) for p in args.candidate],
        markers=markers,
        bars=bars,
    )
    evidence = make_deepseek_v4_q4_evidence({"gate": "Q4", "probes": len(panels)}, external)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        write_artifact(args.json_out, evidence)
    print(json.dumps({
        "rung": evidence.get("rung"),
        "status": evidence.get("status"),
        "summary": evidence.get("summary"),
        "artifact_id": evidence.get("artifact_id"),
    }, indent=2, sort_keys=True))
    if evidence["status"] == "draft" and args.report_only:
        return 0
    return 0 if evidence["status"] == "valid" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
