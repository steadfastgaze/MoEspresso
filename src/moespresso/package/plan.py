"""Common package-plan artifact for recipe and optimizer producers."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from copy import deepcopy
import re

from moespresso.core.artifact import Validation, compute_artifact_id, make_artifact


class PackagePlanError(ValueError):
    pass


@dataclass(frozen=True)
class ForceOverride:
    pattern: str
    target: str


_TQ_RE = re.compile(r"^tq(?P<bits>[124])$")
_AFFINE_RE = re.compile(r"^affine(?P<bits>[234568])(?::(?P<group_size>32|64|128))?$")


def parse_force_override(spec: str) -> ForceOverride:
    """Parse `pattern=format`, the command-line force override shape."""
    if "=" not in spec:
        raise PackagePlanError(
            f"force override {spec!r} must have the form PATTERN=FORMAT")
    pattern, target = (part.strip() for part in spec.split("=", 1))
    if not pattern or not target:
        raise PackagePlanError(
            f"force override {spec!r} must have a non-empty pattern and format")
    _parse_target(target)
    return ForceOverride(pattern=pattern, target=target)


def parse_force_overrides(specs: list[str] | tuple[str, ...] | None) -> list[ForceOverride]:
    return [parse_force_override(spec) for spec in (specs or [])]


def _parse_target(target: str) -> dict:
    value = target.strip().lower()
    match = _TQ_RE.match(value)
    if match:
        return {"format": "tq", "codec": "tq", "bits": int(match.group("bits"))}
    match = _AFFINE_RE.match(value)
    if match:
        return {
            "format": "affine",
            "bits": int(match.group("bits")),
            "group_size": int(match.group("group_size") or 64),
        }
    if value in {"mxfp4", "mxfp8"}:
        return {
            "format": value,
            "codec": value,
            "bits": 4 if value == "mxfp4" else 8,
            "group_size": 32,
        }
    if value.startswith("kquant:"):
        value = value.split(":", 1)[1]
    from moespresso.package.kquant_format import KQUANT_GEOMETRY

    if value in KQUANT_GEOMETRY:
        geometry = KQUANT_GEOMETRY[value]
        return {
            "format": "kquant",
            "codec": value,
            "kquant_codec": value,
            "bits": int(geometry.bits),
            "group_size": int(geometry.group_size),
        }
    raise PackagePlanError(f"unknown force override format {target!r}")


def _match_texts(alloc: dict) -> list[str]:
    return [
        str(alloc.get(key, ""))
        for key in (
            "source_name",
            "role",
            "projection",
            "format",
            "codec",
            "kquant_codec",
            "gguf_tensor",
            "module_path",
            "module_weight_key",
        )
        if alloc.get(key) is not None
    ]


def _matches(alloc: dict, pattern: str) -> bool:
    return any(fnmatchcase(text, pattern) for text in _match_texts(alloc))


def _current_format(alloc: dict) -> str:
    fmt = alloc.get("format") or alloc.get("codec") or alloc.get("kind")
    if fmt == "kquant":
        return f"kquant:{alloc.get('kquant_codec') or alloc.get('codec')}"
    if fmt == "affine":
        return f"affine{alloc.get('bits')}:{alloc.get('group_size')}"
    if fmt == "tq":
        return f"tq{alloc.get('bits')}"
    return str(fmt)


def _preview_row(alloc: dict, override: ForceOverride, target_fields: dict) -> dict:
    return {
        "source_name": alloc.get("source_name"),
        "role": alloc.get("role"),
        "kind": alloc.get("kind"),
        "layer_index": alloc.get("layer_index"),
        "projection": alloc.get("projection"),
        "gguf_tensor": alloc.get("gguf_tensor"),
        "pattern": override.pattern,
        "before": _current_format(alloc),
        "after": override.target,
        "target_format": target_fields["format"],
    }


def _apply_target(alloc: dict, override: ForceOverride, target_fields: dict) -> None:
    before = _current_format(alloc)
    target_format = target_fields["format"]
    alloc["format"] = target_format
    alloc["bits"] = int(target_fields["bits"])
    alloc["codec"] = target_fields.get("codec", target_format)
    if "group_size" in target_fields:
        alloc["group_size"] = int(target_fields["group_size"])
    if target_format == "kquant":
        alloc["kquant_codec"] = target_fields["kquant_codec"]
    else:
        alloc.pop("kquant_codec", None)
    alloc["forced_format"] = {
        "pattern": override.pattern,
        "target": override.target,
        "before": before,
    }


def apply_force_overrides(
    decision: dict,
    overrides: list[ForceOverride] | tuple[ForceOverride, ...],
    *,
    allow_unmatched: bool = False,
    dry_run: bool = False,
) -> tuple[dict, dict]:
    """Apply package-plan force overrides to a decision allocation.

    Returns `(decision, summary)`. In dry-run mode the returned decision is an
    unmodified copy and the summary still lists every planned row.
    """
    out = deepcopy(decision)
    allocation = out.get("allocation", [])
    forced_rows: list[dict] = []
    unmatched: list[str] = []
    for override in overrides:
        target_fields = _parse_target(override.target)
        matched = [alloc for alloc in allocation if _matches(alloc, override.pattern)]
        if not matched:
            unmatched.append(override.pattern)
            continue
        for alloc in matched:
            forced_rows.append(_preview_row(alloc, override, target_fields))
            if not dry_run:
                _apply_target(alloc, override, target_fields)
    if unmatched and not allow_unmatched:
        raise PackagePlanError(
            "force override pattern(s) matched no tensors: " + ", ".join(unmatched))
    summary = {
        "dry_run": bool(dry_run),
        "matched": forced_rows,
        "unmatched_patterns": unmatched,
    }
    if not dry_run:
        out.pop("artifact_id", None)
        out["artifact_id"] = compute_artifact_id(out)
    return out, summary


def force_override_preview_lines(plan: dict, *, limit: int = 20) -> list[str]:
    """Human-readable dry-run preview lines for matched force overrides."""
    preview = plan.get("force_override_preview") or {}
    matched = list(preview.get("matched", []))
    lines = [f"  matched={len(matched)}"]
    for row in matched[:limit]:
        name = (
            row.get("source_name")
            or row.get("gguf_tensor")
            or row.get("module_weight_key")
            or "<unknown>"
        )
        lines.append(f"    {name}: {row.get('before')} -> {row.get('after')}")
    remaining = len(matched) - limit
    if remaining > 0:
        lines.append(f"    ... {remaining} more")
    unmatched = list(preview.get("unmatched_patterns", []))
    if unmatched:
        lines.append("  unmatched=" + ", ".join(str(p) for p in unmatched))
    return lines


PRODUCER = {"tool": "moespresso.package.plan", "version": "1.0.0"}


def make_package_plan(
    subject: dict,
    allocation: list[dict],
    *,
    producer_kind: str,
    producer_reference: str | None = None,
    optimized_kernels_expected: bool = False,
    force_overrides: list[ForceOverride] | tuple[ForceOverride, ...] | None = None,
    allow_unmatched_force: bool = False,
    dry_run: bool = False,
    required_features: list[str] | tuple[str, ...] | None = None,
    source_decision_id: str | None = None,
    source_probe_id: str | None = None,
    source_constraints: dict | None = None,
    achieved: dict | None = None,
    validation: list[Validation] | None = None,
    status: str = "valid",
) -> tuple[dict, dict]:
    """Build the shared package-plan artifact.

    The plan is the one artifact the writer consumes. Optimizer and GGUF recipe
    producers differ only in the metadata they attach before this point.
    """
    base = make_artifact(
        "package_plan",
        subject,
        PRODUCER,
        required_features=list(required_features or []),
        status=status,
        validation=validation or [],
        source_decision_id=source_decision_id,
        source_probe_id=source_probe_id,
        producer_kind=producer_kind,
        producer_reference=producer_reference,
        optimized_kernels_expected=bool(optimized_kernels_expected),
        source_constraints=source_constraints or {},
        allocation=[dict(row) for row in allocation],
        achieved=achieved or {},
        force_overrides=[
            {"pattern": override.pattern, "target": override.target}
            for override in (force_overrides or [])
        ],
    )
    if status == "invalid":
        return base, {"dry_run": bool(dry_run), "matched": [], "unmatched_patterns": []}
    planned, summary = apply_force_overrides(
        base,
        list(force_overrides or []),
        allow_unmatched=allow_unmatched_force,
        dry_run=dry_run,
    )
    if dry_run:
        planned["force_override_preview"] = summary
        planned.pop("artifact_id", None)
        planned["artifact_id"] = compute_artifact_id(planned)
    return planned, summary


def package_plan_from_decision(
    decision: dict,
    *,
    optimized_kernels_expected: bool = False,
    force_overrides: list[ForceOverride] | tuple[ForceOverride, ...] | None = None,
    allow_unmatched_force: bool = False,
    dry_run: bool = False,
) -> tuple[dict, dict]:
    """Render an optimizer decision into the shared writer-facing plan."""
    return make_package_plan(
        decision["subject"],
        decision.get("allocation", []),
        producer_kind="probe_optimizer",
        optimized_kernels_expected=optimized_kernels_expected,
        force_overrides=force_overrides,
        allow_unmatched_force=allow_unmatched_force,
        dry_run=dry_run,
        required_features=list(decision.get("required_features", [])),
        source_decision_id=decision.get("artifact_id"),
        source_probe_id=decision.get("source_probe_id"),
        source_constraints=decision.get("constraints") or {},
        achieved=decision.get("achieved") or {},
        validation=[
            Validation(
                v.get("severity", "error"),
                v.get("code", "package_plan.source_validation"),
                v.get("message", ""),
                path=v.get("path", ""),
                phase=v.get("phase", "package_plan"),
                blocking=bool(v.get("blocking", False)),
                expected=v.get("expected"),
                actual=v.get("actual"),
            )
            for v in decision.get("validation", [])
        ],
        status=decision.get("status", "valid"),
    )
