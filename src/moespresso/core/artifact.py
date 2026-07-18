"""The artifact contract: the one rule every MoEspresso phase obeys.

Every durable thing (inventory, probe evidence, optimizer decision, package plan,
manifest) is an Artifact: a dict payload plus a content-addressed id and a small
set of required base keys. This module owns three things and nothing else:

  1. canonical serialization (sorted-key UTF-8 JSON, no NaN/Inf)  -> content hash
  2. base-key validation (fail-closed on unknown kind / major version)
  3. read/write helpers that compute and verify the id

Phase-specific schemas live next to their phase; they only add keys. Keeping the
contract in one tiny place is what stops every tool from inventing its own.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

SCHEMA_MAJOR = 1
SCHEMA_MINOR = 0

ARTIFACT_KINDS = frozenset({
    "source_inventory",
    "probe_evidence",
    "optimizer_decision",
    "package_plan",
    "package_manifest",
    # correctness ladder (standalone evidence; not wired into convert/serve/verify).
    "architecture_profile",   # the model-family contract consumed by L0-L4
    "correctness_evidence",   # what a ladder rung (L0/L1/...) actually found
})

# Short id prefix per kind, e.g. "inv:2f8b...".
_KIND_TAG = {
    "source_inventory": "inv",
    "probe_evidence": "probe",
    "optimizer_decision": "dec",
    "package_plan": "plan",
    "package_manifest": "pkg",
    "architecture_profile": "arch",
    "correctness_evidence": "correct",
}

_REQUIRED_BASE_KEYS = (
    "artifact_kind",
    "schema_version",
    "producer",
    "subject",
    "status",
)

_STATUSES = frozenset({"draft", "valid", "invalid", "superseded", "retired"})

# Keys excluded from the content hash: the id itself, and `created_at` (a wall-clock
# stamp must not perturb the content identity: two artifacts with identical content
# share an id regardless of when they were written). See canonical_json.
_HASH_EXCLUDED = ("artifact_id", "created_at")

# Feature strings a reader must understand for an artifact to load (spec: unknown
# required feature fails closed). The set grows as real features land; a format
# like mjtq declares its requirements here (e.g. "calibration").
KNOWN_FEATURES = frozenset({
    "calibration",  # probe_evidence carries calibration-dataset identity
})


class ArtifactError(Exception):
    """A base-contract violation: unknown kind, bad version, or failed hash check."""


@dataclass
class Validation:
    """One structured validation entry (spec 'Validation entries')."""

    severity: str          # "error" | "warning" | "info"
    code: str              # dotted, e.g. "tensor.shape_mismatch"
    message: str
    path: str = ""
    phase: str = ""
    blocking: bool = False
    expected: object = None
    actual: object = None

    def as_dict(self) -> dict:
        d = {
            "severity": self.severity, "code": self.code, "message": self.message,
            "path": self.path, "phase": self.phase, "blocking": self.blocking,
        }
        if self.expected is not None:
            d["expected"] = self.expected
        if self.actual is not None:
            d["actual"] = self.actual
        return d


def _assert_finite(obj, where: str = "$") -> None:
    """Reject NaN/Inf anywhere in the payload (spec: forbidden in persisted artifacts)."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ArtifactError(f"non-finite float at {where}: {obj!r}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_finite(v, f"{where}[{i}]")


def canonical_json(payload: dict) -> str:
    """Canonical UTF-8 JSON: sorted keys, compact, deterministic, no NaN/Inf.

    `artifact_id` and `created_at` are excluded: the id never depends on itself,
    and a wall-clock stamp must not change the content identity (same content ->
    same id, whenever it was written).
    """
    _assert_finite(payload)
    body = {k: v for k, v in payload.items() if k not in _HASH_EXCLUDED}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def compute_artifact_id(payload: dict) -> str:
    """Content hash of the canonical payload, prefixed by the kind tag."""
    kind = payload.get("artifact_kind")
    tag = _KIND_TAG.get(kind, "art")
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{tag}:{digest}"


def validate_base(payload: dict) -> list[Validation]:
    """Check the base contract. Returns validation entries (empty == base-valid).

    Fail-closed conditions (unknown kind, unknown major) raise ArtifactError;
    softer issues are returned as entries so a caller can decide.
    """
    if payload.get("artifact_kind") not in ARTIFACT_KINDS:
        raise ArtifactError(f"unknown artifact_kind: {payload.get('artifact_kind')!r}")

    ver = payload.get("schema_version") or {}
    if ver.get("major") != SCHEMA_MAJOR:
        raise ArtifactError(
            f"unsupported schema major {ver.get('major')!r} (this build: {SCHEMA_MAJOR})")

    out: list[Validation] = []
    for key in _REQUIRED_BASE_KEYS:
        if key not in payload:
            out.append(Validation("error", "artifact.missing_key",
                                  f"missing required base key '{key}'",
                                  path=f"/{key}", phase="contract", blocking=True))
    status = payload.get("status")
    if status is not None and status not in _STATUSES:
        out.append(Validation("error", "artifact.bad_status",
                              f"status {status!r} not in {sorted(_STATUSES)}",
                              path="/status", phase="contract", blocking=True))
    # Fail closed on a required feature this build doesn't understand (spec).
    for feat in payload.get("required_features", []):
        if feat not in KNOWN_FEATURES:
            out.append(Validation("error", "artifact.unknown_required_feature",
                                  f"required feature {feat!r} not understood "
                                  f"(known: {sorted(KNOWN_FEATURES)})",
                                  path="/required_features", phase="contract",
                                  blocking=True))
    return out


def make_artifact(
    artifact_kind: str,
    subject: dict,
    producer: dict,
    *,
    inputs: list | None = None,
    required_features: list | None = None,
    status: str = "draft",
    validation: list[Validation] | None = None,
    **fields,
) -> dict:
    """Build an artifact dict with base keys filled and a computed `artifact_id`.

    Deterministic by construction: no wall-clock is read here, so the same content
    always yields the same `artifact_id` (tests rely on this). `created_at` is not
    set here: it's stamped at write time by `write_artifact` and excluded from the
    hash, so persisting an artifact never changes its id. `required_features`
    defaults to empty; entries must be in `KNOWN_FEATURES` (validate_base fails
    closed otherwise).
    """
    if artifact_kind not in ARTIFACT_KINDS:
        raise ArtifactError(f"unknown artifact_kind: {artifact_kind!r}")
    payload = {
        "artifact_kind": artifact_kind,
        "schema_version": {"major": SCHEMA_MAJOR, "minor": SCHEMA_MINOR},
        "producer": producer,
        "subject": subject,
        "inputs": inputs or [],
        "required_features": list(required_features or []),
        "status": status,
        "validation": [v.as_dict() for v in (validation or [])],
        **fields,
    }
    base_issues = validate_base(payload)
    if base_issues:
        raise ArtifactError("; ".join(v.message for v in base_issues))
    payload["artifact_id"] = compute_artifact_id(payload)
    return payload


def write_artifact(path, payload: dict, created_at: str | None = None) -> str:
    """Write an artifact as canonical JSON (recomputing/verifying its id). Returns the id.

    Stamps `created_at` (a UTC timestamp string) at persist time if the payload
    doesn't already carry one. It's excluded from the hash, so stamping never
    changes the id, keeping `make_artifact` deterministic while the persisted
    artifact satisfies the spec's required `created_at`. The caller supplies the
    timestamp (no wall-clock is read here) so writes stay reproducible in tests.
    """
    expected = compute_artifact_id(payload)
    if payload.get("artifact_id") not in (None, expected):
        raise ArtifactError("artifact_id does not match payload content")
    payload = {**payload, "artifact_id": expected}
    if "created_at" not in payload and created_at is not None:
        payload["created_at"] = created_at
    # Stored pretty for human diffing; the *id* is over the canonical form.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    return expected


def read_artifact(path) -> dict:
    """Read an artifact, verifying its content hash and base contract (fail-closed)."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    stored = payload.get("artifact_id")
    actual = compute_artifact_id(payload)
    if stored != actual:
        raise ArtifactError(f"artifact_id mismatch: stored {stored} != computed {actual}")
    validate_base(payload)  # raises on fail-closed conditions
    return payload
