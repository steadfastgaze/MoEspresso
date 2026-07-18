"""Convert-time correctness gate: run the ladder rungs L0/L0b/L1/L2 against a freshly
written package and decide whether it may be finalized.

This is the wiring the standalone rungs were built for: a package that fails a rung must
not be shipped as if it were sound. The gate is format-agnostic: it resolves the family's
`architecture_profile` from the model config and runs only when one exists; an unprofiled
family is reported as `skipped` (the caller warns and proceeds), never silently passed and
never wrongly blocked.

L0/L0b are pure/header-only; L1/L2 reconstruct sampled tensors with the runtime stack.
Each rung's findings become a `correctness_evidence` artifact the caller persists next to
the package, so a blocked or allowed-through convert is always inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moespresso.correctness.goldens import l2_micro_goldens
from moespresso.correctness.ladder import (
    l0_static_contract,
    l0b_norm_shift_contract,
    make_correctness_evidence,
)
from moespresso.correctness.reconstruct import l1_tensor_reconstruction


@dataclass
class GateResult:
    """Outcome of the convert-time correctness gate.

    `skipped` is True when no profile matched the family (gate did not run). `evidence` is
    the list of correctness_evidence artifacts (one per rung that ran) for the caller to
    persist. `blocking` lists the (rung, code, message) of every blocking finding.
    """
    passed: bool
    skipped: bool
    evidence: list[dict]
    blocking: list[tuple[str, str, str]]


def _blocking_from_evidence(ev: dict) -> list[tuple[str, str, str]]:
    rung = ev.get("rung", "?")
    return [(rung, v["code"], v["message"])
            for v in ev.get("validation", []) if v.get("blocking")]


def run_convert_gate(
    profile: dict,
    inventory: dict,
    manifest: dict,
    source_dir: Path,
    package_dir: Path,
    *,
    subject: dict | None = None,
    expect_conv1d: bool = True,
) -> GateResult:
    """Run L0/L0b/L1/L2 against a written package; collect evidence + blocking findings.

    The caller resolves `profile` (None -> skip the gate). `expect_conv1d` says whether this
    package should carry a conv1d at all (False for a full-attention-only or smoke build) so
    L0b doesn't block a legitimately-absent one. Every rung that runs contributes a
    correctness_evidence artifact to `evidence`; `passed` is False iff any rung produced a
    blocking finding.
    """
    subject = subject or {"source_root": str(source_dir)}
    evidence: list[dict] = []

    # L0 (pure) and L0b (header-only) emit Validation lists -> wrap as evidence.
    l0 = l0_static_contract(profile, inventory, manifest)
    evidence.append(make_correctness_evidence(subject, rung="L0", findings=l0))
    l0b = l0b_norm_shift_contract(profile, package_dir, expect_conv1d=expect_conv1d)
    evidence.append(make_correctness_evidence(subject, rung="L0b", findings=l0b))

    # L1/L2 already return correctness_evidence artifacts.
    evidence.append(l1_tensor_reconstruction(
        profile, inventory, manifest, source_dir, package_dir))
    evidence.append(l2_micro_goldens(subject))

    blocking: list[tuple[str, str, str]] = []
    for ev in evidence:
        blocking.extend(_blocking_from_evidence(ev))
    return GateResult(passed=not blocking, skipped=False,
                      evidence=evidence, blocking=blocking)
