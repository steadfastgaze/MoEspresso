"""Build a `probe_evidence` artifact from a `source_inventory` + the model weights.

The probe consumes the inventory (it never re-parses tensor names or re-resolves
GGUF keys, that already happened in the inventory phase) and, for every probe
target, streams a small weight sample and measures reconstruction quality at each
candidate bit-width: TurboQuant for experts, mlx affine for non-experts. The
per-input-channel importance vector comes from the imatrix, keyed by the GGUF key
the inventory already resolved.

Fine-grained MoE: experts within a layer are near-equivalent, so we probe at most
`expert_sample` experts per stacked tensor and record the count. fp16 measurement,
one sample resident at a time, fits a few GB on a 35B model.

The result is the q-table the optimizer turns into a bit allocation. The pure
math lives in probe.quality / optimize.aggregate; this module is the imperative
shell: read inventory -> stream + measure -> emit artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moespresso.core.artifact import Validation, make_artifact
from moespresso.probe import roundtrip, weight_io

EXPERT_BITS = (1, 2, 4)
AFFINE_BITS = (2, 3, 4, 5, 6, 8)
AFFINE_GROUP_SIZES = (32, 64, 128)
DEFAULT_EXPERT_SAMPLE = 2     # fine-grained MoE: experts in a layer are equivalent
DEFAULT_SAMPLE_ROWS = 256

PRODUCER = {"tool": "moespresso.probe", "version": "1.0.0"}


def imatrix_coverage_validations(
    mapped: dict[str, int],
    targets: dict[str, int],
) -> list[Validation]:
    """Warnings for probe targets that fell back from imatrix to uniform weights."""
    out: list[Validation] = []
    for kind in ("expert", "affine"):
        total = int(targets.get(kind, 0))
        hit = int(mapped.get(kind, 0))
        if total <= 0 or hit == total:
            continue
        if hit == 0:
            out.append(Validation(
                "warning",
                "probe.no_imatrix",
                f"0/{total} {kind} units mapped to imatrix (uniform fallback)",
                phase="probe",
            ))
        else:
            out.append(Validation(
                "warning",
                "probe.partial_imatrix",
                f"{hit}/{total} {kind} units mapped to imatrix; "
                f"{total - hit} used uniform fallback",
                phase="probe",
            ))
    return out


def _importance_vector(
    gguf_key: str | None, imatrix_vectors: dict[str, np.ndarray], in_features: int,
) -> tuple[np.ndarray, bool]:
    """Per-input-channel importance for one tensor. Returns (h, used_imatrix).

    Falls back to a uniform vector (normalized MSE) when there's no matching
    imatrix vector or its length disagrees with the weight's in_features.
    """
    h = imatrix_vectors.get(gguf_key) if gguf_key else None
    if h is None or h.shape[0] != in_features:
        return np.ones(in_features, dtype=np.float32), False
    return h, True


def _scalar_importance(h: np.ndarray, used_imatrix: bool, sample: np.ndarray) -> float:
    """One importance scalar for the optimizer's fidelity weighting."""
    if used_imatrix:
        return float(np.mean(h))
    return float(np.linalg.norm(sample) / np.sqrt(sample.size)) if sample.size else 0.0


def _probe_affine(
    model_dir: Path, entry: dict, header, imatrix_vectors, sample_rows: int, seed: int,
) -> dict:
    sample = weight_io.load_2d_sample(model_dir, header, sample_rows, seed)
    in_features = sample.shape[1]
    gguf_key = entry["gguf_keys"][0] if entry["gguf_keys"] else None
    h, used = _importance_vector(gguf_key, imatrix_vectors, in_features)
    quality: dict[str, float] = {}
    for bits in AFFINE_BITS:
        for gs in AFFINE_GROUP_SIZES:
            if in_features % gs != 0:
                continue
            _, q = roundtrip.affine_quality(sample, bits, gs, h)
            quality[f"{bits}_{gs}"] = q
    return {
        "source_name": entry["source_name"],
        "kind": "affine",
        "role": entry["role"],
        "layer_index": entry.get("layer_index"),
        "shape": entry["shape"],
        "importance": _scalar_importance(h, used, sample),
        "imatrix_mapped": used,
        "quality": quality,
    }


def _probe_expert(
    model_dir: Path, entry: dict, header, imatrix_vectors, sample_rows: int,
    expert_sample: int, seed: int,
) -> list[dict]:
    """One stacked tensor -> one report per sub-projection (gate/up split if fused)."""
    layer = entry["layer_index"]
    proj = entry["projection"]
    fused = "gate_up" in proj
    raw = weight_io.load_expert_sample(model_dir, header, expert_sample, seed)
    # header.shape is [E, rows, cols]; load_expert_sample returns [n_sampled*rows, cols].
    rows = header.shape[1]
    n_sampled = raw.shape[0] // rows
    # True per-projection out_features (a fused gate_up splits into two halves). This
    # describes the tensor, so it sizes correctly; it is not the sampled-row count.
    out_features = rows // 2 if fused else rows

    if fused:
        gate, up = weight_io.split_fused_gate_up(raw, n_sampled)
        sub_samples = [("gate", gate), ("up", up)]
    else:
        sub_samples = [(proj.replace("_proj", ""), raw)]

    reports = []
    for sub, sample in sub_samples:
        # subsample rows to keep the round-trip cheap (the sample shrinks; the recorded
        # `shape` stays the true tensor geometry so the optimizer prices bytes right).
        if sample.shape[0] > sample_rows:
            idx = np.random.default_rng(seed).choice(sample.shape[0], sample_rows, replace=False)
            sample = sample[idx]
        in_features = sample.shape[1]
        gguf_key = f"blk.{layer}.ffn_{sub}_exps.weight"
        h, used = _importance_vector(gguf_key, imatrix_vectors, in_features)
        quality = {}
        for bits in EXPERT_BITS:
            _, q = roundtrip.tq_quality(sample, bits, h, seed)
            quality[str(bits)] = q
        reports.append({
            "source_name": entry["source_name"],
            "kind": "expert",
            "role": entry["role"],
            "layer_index": layer,
            "projection": sub,
            "n_experts": header.shape[0],
            "sampled": n_sampled,
            "shape": [out_features, in_features],  # true tensor geometry
            "importance": _scalar_importance(h, used, sample),
            "imatrix_mapped": used,
            "quality": quality,
        })
    return reports


def build_probe_evidence(
    inventory: dict,
    model_dir: Path,
    calibration: tuple[dict[str, np.ndarray], dict] | None = None,
    *,
    expert_sample: int = DEFAULT_EXPERT_SAMPLE,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Measure quality-vs-bits for every probe target -> probe_evidence artifact.

    `calibration` is a (vectors, identity) pair from a calibration provider (e.g.
    probe.calibration.imatrix_calibration). When given, the evidence records the
    calibration identity and declares `required_features=["calibration"]` so the
    spec's fail-closed rule applies. `calibration=None` is an explicit escape
    hatch (synthetic tests / no-imatrix research): the evidence is stamped
    `calibration={"kind": "uniform"}` and declares NO required feature, so an
    uncalibrated artifact cannot pass as calibrated. The probe
    core stays format-agnostic; mjtq's "calibration required" policy is enforced
    by the caller that chooses mjtq (see package.convert). This module stays
    format-agnostic.
    """
    if inventory.get("family") == "deepseek_v4_flash":
        from moespresso.probe.deepseek_v4.probe import (
            build_deepseek_v4_probe_evidence_from_inventory,
        )

        return build_deepseek_v4_probe_evidence_from_inventory(
            inventory,
            model_dir,
            calibration,
            expert_sample=expert_sample,
            sample_rows=sample_rows,
            seed=seed,
        )

    if calibration is not None:
        imatrix_vectors, calib_identity = calibration
    else:
        imatrix_vectors, calib_identity = {}, {"kind": "uniform"}
    catalog = weight_io.scan_offsets(model_dir)
    units: list[dict] = []
    mapped = {"expert": 0, "affine": 0}
    n_targets = {"expert": 0, "affine": 0}

    for entry in inventory["tensors"]:
        if entry["status"] not in ("required",):
            continue
        header = catalog.get(entry["source_name"])
        if header is None:
            continue
        if entry["kind"] == "expert":
            reps = _probe_expert(model_dir, entry, header, imatrix_vectors,
                                 sample_rows, expert_sample, seed)
            for r in reps:
                n_targets["expert"] += 1
                mapped["expert"] += int(r["imatrix_mapped"])
            units.extend(reps)
        elif entry["kind"] == "affine":
            r = _probe_affine(model_dir, entry, header, imatrix_vectors, sample_rows, seed)
            n_targets["affine"] += 1
            mapped["affine"] += int(r["imatrix_mapped"])
            units.append(r)
        if verbose:
            print(f"  probed {entry['source_name']}")

    validation = imatrix_coverage_validations(mapped, n_targets)

    # Calibrated evidence declares the `calibration` feature (spec fail-closed
    # gate); the uniform escape hatch declares nothing, so it can't masquerade as
    # calibrated.
    required_features = ["calibration"] if calib_identity.get("kind") != "uniform" else []

    return make_artifact(
        "probe_evidence", inventory["subject"], PRODUCER,
        status="valid", validation=validation,
        required_features=required_features,
        units=units,
        calibration=calib_identity,
        config={"expert_bits": list(EXPERT_BITS), "affine_bits": list(AFFINE_BITS),
                "affine_group_sizes": list(AFFINE_GROUP_SIZES),
                "expert_sample": expert_sample, "sample_rows": sample_rows,
                "seed": seed, "fp16_measurement": True},
        coverage={"expert_mapped": f"{mapped['expert']}/{n_targets['expert']}",
                  "affine_mapped": f"{mapped['affine']}/{n_targets['affine']}"},
        source_inventory_id=inventory.get("artifact_id"),
    )
