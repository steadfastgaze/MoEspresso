"""End-to-end streamed conversion: inventory -> probe -> optimize -> package.

One command that runs the whole pipeline in a few GB of RAM and writes a
MoEspresso package (quantized shards + `package_manifest.json`) to disk, e.g. an
SSD. Every phase streams (the probe samples by byte-range; the writer quantizes a
row-band / one expert at a time), so the full 35B converts in a bounded footprint.

This module is the imperative shell that wires the four phase functions and
persists their artifacts; the phases themselves own the science. Needs the
runtime dependencies (probe round-trips + package writer use mlx/jang).

This orchestrator is format-neutral: it does not itself decree that calibration
is mandatory. Instead it consults the target format's declared requirements
(`PACKAGE_FORMAT_FEATURES`). mjtq declares it requires "calibration", so
producing a mjtq package without an imatrix is refused. Other package formats
declare their own requirements. The probe core is likewise format-agnostic; it
records calibration identity when given one and declares
`required_features=["calibration"]` on the evidence.

The uniform-importance path still exists for research as an explicit in-process
override (`allow_uniform=True`), never via the CLI, which always produces a
calibrated mjtq package, so an uncalibrated mjtq is a deliberate library call,
never an accident.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path

from moespresso.core.artifact import write_artifact
from moespresso.inventory.build import build_inventory
from moespresso.optimize.decide import decide
from moespresso.package.constants import MANIFEST_NAME
from moespresso.probe.build import (
    DEFAULT_EXPERT_SAMPLE,
    DEFAULT_SAMPLE_ROWS,
    build_probe_evidence,
)


def _rss_bytes() -> int:
    """Current process RSS in bytes; psutil if present, else resource fallback.

    No hard psutil dependency (same posture as package/write.py's auto-sizer).
    The resource fallback reports max-RSS (ru_maxrss), already a peak.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


_GB = 1024 ** 3


def rss_summary(samples: list[int]) -> dict | None:
    """Peak/mean/median/p95 over an RSS sample series, in GB. None if empty.

    Peak alone is skew-prone (one transient spike). The distribution tells the
    real story of the streaming footprint: median ~ the typical load, p95 ~ how
    close to the edge it routinely runs, peak ~ the crash-relevant ceiling.
    """
    if not samples:
        return None
    s = sorted(samples)
    n = len(s)

    def pct(p):  # nearest-rank percentile
        return s[min(n - 1, max(0, int(round(p / 100.0 * n)) - 1))]

    return {
        "peak_gb": round(s[-1] / _GB, 3),
        "mean_gb": round(sum(s) / n / _GB, 3),
        "median_gb": round(pct(50) / _GB, 3),
        "p95_gb": round(pct(95) / _GB, 3),
        "samples": n,
    }


@contextmanager
def _rss_watch(interval: float = 2.0):
    """Sample RSS in a background thread; yields a list filled with byte samples.

    The tool measures and reports the bounded-footprint invariant: streamed
    conversion should stay within a few GB of RAM.
    Sampling is one syscall every `interval` s on a daemon thread: free against an
    IO/compute-bound convert, so it never slows the conversion. Percentiles are
    computed once at the end (rss_summary) from the collected series.
    """
    samples: list[int] = [_rss_bytes()]
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            samples.append(_rss_bytes())
            stop.wait(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield samples
    finally:
        stop.set()
        t.join(timeout=interval + 1.0)
        samples.append(_rss_bytes())


INVENTORY_NAME = "source_inventory.json"
PROBE_NAME = "probe_evidence.json"
DECISION_NAME = "optimizer_decision.json"
PACKAGE_PLAN_NAME = "package_plan.json"
REPORT_NAME = "conversion_report.json"

DEEPSEEK_V4_PRACTICAL_TARGET_SIZE_GB = 75.0
DEEPSEEK_V4_ABSOLUTE_PACKAGE_SIZE_GB = 85.0
_GIB = 1024 ** 3
CORRECTNESS_DIR = "correctness"


def _artifact_write_note(label: str, path: Path, write_intermediate: bool) -> str:
    if write_intermediate:
        return f"{label} written to {path}."
    return f"{label} artifact was not written because write_intermediate=False."


def _read_config(model_dir: Path) -> dict:
    cfg = Path(model_dir) / "config.json"
    return json.loads(cfg.read_text()) if cfg.exists() else {}


def _layer_types(config: dict) -> list[str] | None:
    return config.get("text_config", config).get("layer_types")


def _source_safetensors_total_size(model_dir: Path) -> int | None:
    """Source checkpoint tensor-data bytes from model.safetensors.index.json."""
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    data = json.loads(index_path.read_text(encoding="utf-8"))
    value = data.get("metadata", {}).get("total_size")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _package_shard_bytes(manifest: dict) -> int:
    return sum(int(f.get("size_bytes", 0)) for f in manifest.get("files", []))


def _deepseek_v4_package_size_contract(
    family: str | None,
    model_dir: Path,
    manifest: dict,
) -> dict | None:
    """Fail closed if a DS4 package exceeds source or absolute project ceiling."""
    if family != "deepseek_v4_flash":
        return None
    source_bytes = _source_safetensors_total_size(model_dir)
    if source_bytes is None:
        raise RuntimeError(
            "DeepSeek V4 package size gate requires model.safetensors.index.json "
            "metadata.total_size")
    package_bytes = _package_shard_bytes(manifest)
    absolute_max_bytes = int(DEEPSEEK_V4_ABSOLUTE_PACKAGE_SIZE_GB * _GIB)
    contract = {
        "family": family,
        "source_size_bytes": source_bytes,
        "package_size_bytes": package_bytes,
        "package_le_source": package_bytes <= source_bytes,
        "practical_target_size_gb": DEEPSEEK_V4_PRACTICAL_TARGET_SIZE_GB,
        "absolute_package_size_gb": DEEPSEEK_V4_ABSOLUTE_PACKAGE_SIZE_GB,
        "package_le_absolute_ceiling": package_bytes <= absolute_max_bytes,
    }
    if package_bytes > absolute_max_bytes:
        raise RuntimeError(
            "DeepSeek V4 package size gate FAILED: package shards "
            f"{package_bytes} bytes exceed absolute "
            f"{DEEPSEEK_V4_ABSOLUTE_PACKAGE_SIZE_GB:.1f} GiB ceiling")
    if package_bytes > source_bytes:
        raise RuntimeError(
            "DeepSeek V4 package size gate FAILED: package shards "
            f"{package_bytes} bytes exceed source checkpoint {source_bytes} bytes")
    return contract


def convert(
    model_dir: Path,
    out_dir: Path,
    *,
    imatrix_path: Path | str | None = None,
    allow_uniform: bool = False,
    allow_unhealthy: bool = False,
    allow_incomplete: bool = False,
    target_quality: float | None = None,
    target_size_gb: float | None = None,
    expert_allocation_ratio: float | None = None,
    expert_importance_norm: str = "class-mean",
    tau: float | None = None,
    alpha: float = 0.05,
    seed: int = 42,
    shard_size_gb: float = 4.0,
    chunk_bytes: int | None = None,
    expert_sample: int = DEFAULT_EXPERT_SAMPLE,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    max_experts: int | None = None,
    affine_role_weights: dict[str, float] | None = None,
    affine_role_bit_weights: dict[str, dict[int | str, float]] | None = None,
    affine_role_min_bits: dict[str, int] | None = None,
    force_tq4_lossless: bool = False,
    force_dense_lossless_mx: bool = False,
    min_routed_expert_bits: int = 1,
    optimized_kernels_expected: bool = False,
    force_format: list[str] | tuple[str, ...] | None = None,
    allow_unmatched_force: bool = False,
    force_format_dry_run: bool = False,
    write_intermediate: bool = True,
    verbose: bool = False,
) -> dict:
    """Run inventory->probe->optimize->package; write the package + manifest.

    A mjtq package needs calibration: pass `imatrix_path` and the probe is
    activation-weighted, with its identity recorded in probe_evidence.
    `allow_uniform=True` is the explicit research escape hatch (no calibration);
    without either, this raises rather than silently producing an uncalibrated
    package. Returns the `package_manifest`. Intermediate artifacts (inventory,
    probe, decision) are written next to the package when `write_intermediate` so
    the run is fully inspectable / re-enterable. Raises if the optimizer is
    infeasible (with the decision written for inspection when intermediates are
    enabled).
    """
    # Lazy: the writer pulls in mlx/jang only here, so importing this module is cheap.
    from moespresso.package.manifest import PACKAGE_FORMAT, PACKAGE_FORMAT_FEATURES
    from moespresso.package.write import write_package

    # Format-neutral gate: enforce what the target format declares it requires,
    # not a hardcoded pipeline rule.
    if "calibration" in PACKAGE_FORMAT_FEATURES and imatrix_path is None and not allow_uniform:
        raise ValueError(
            f"format {PACKAGE_FORMAT!r} requires calibration: pass imatrix_path "
            f"(allow_uniform=True is an explicit in-process "
            f"research override; the CLI never produces an uncalibrated package.)")

    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _read_config(model_dir)
    layer_types = _layer_types(config)
    from moespresso.inventory.architecture_profile import family_of
    family = family_of(config)
    from moespresso.optimize.affine_elasticity import affine_role_profile_for_family
    affine_role_profile = affine_role_profile_for_family(family)
    if affine_role_profile:
        if affine_role_weights is None:
            affine_role_weights = affine_role_profile["affine_role_weights"]
        if affine_role_bit_weights is None:
            affine_role_bit_weights = affine_role_profile["affine_role_bit_weights"]
        if affine_role_min_bits is None:
            affine_role_min_bits = affine_role_profile["affine_role_min_bits"]
    elif affine_role_min_bits is None:
        # No model-specific affine role band, and the caller passed no explicit floors.
        # The affine backbone is then protected only by the generic 2-bit minimum plus
        # the aggregate allocation health gate, with no per-role floor. That is safe at a
        # conservative size target but gets risky as the target drops and the per-byte
        # greedy starts taking backbone bits: a single critical projection can fall to
        # 2 bits without the aggregate gate flagging it. A proven family (qwen3_5_moe,
        # deepseek_v4_flash) pins the backbone at 4 bits. Strongly prefer creating a
        # band for this family in affine_elasticity.py before an aggressive convert.
        print(f"      [warn] no affine role band for family {family!r} "
              f"(model_type={config.get('model_type')!r}); the affine backbone has no "
              f"per-role floor, only the 2-bit minimum + health gate. Highly recommended: "
              f"add a model-specific affine band (affine_role_min_bits) in "
              f"affine_elasticity.py before pushing the size target down, or pass "
              f"affine_role_min_bits to convert() explicitly.", flush=True)

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # Watch RSS across the whole streamed pipeline. Sampling is one syscall every
    # 2s on a daemon thread (free against an IO/compute-bound convert), so it
    # measures the bounded-footprint invariant without slowing it.
    package_size_contract = None
    with _rss_watch() as rss_samples:
        calibration = None
        if imatrix_path is not None:
            from moespresso.probe.calibration import imatrix_calibration
            _log(f"[0/4] calibration  {imatrix_path}")
            calibration = imatrix_calibration(imatrix_path)
            _log(f"      imatrix: {calibration[1]['key_count']} keys, "
                 f"sha256 {calibration[1]['sha256'][:12]}")

        _log(f"[1/4] inventory  {model_dir}")
        imatrix_keys = set(calibration[0]) if calibration is not None else None
        inventory = build_inventory(
            model_dir,
            layer_types=layer_types,
            imatrix_keys=imatrix_keys,
            family=family,
        )
        if write_intermediate:
            write_artifact(out_dir / INVENTORY_NAME, inventory)
        _log(f"      {inventory['counts']}")
        if inventory["status"] == "invalid":
            blocking = [v for v in inventory.get("validation", []) if v.get("blocking")]
            lines = "; ".join(
                f"{v['code']}: {v['message']}" for v in blocking[:6]
            )
            note = _artifact_write_note(
                "Inventory",
                out_dir / INVENTORY_NAME,
                write_intermediate,
            )
            raise RuntimeError(
                f"source inventory failed ({len(blocking)} blocking finding(s)): "
                f"{lines}. {note}")

        _log("[2/4] probe (streamed sampling)")
        evidence = build_probe_evidence(
            inventory, model_dir, calibration,
            expert_sample=expert_sample, sample_rows=sample_rows, seed=seed,
            verbose=verbose)
        if write_intermediate:
            write_artifact(out_dir / PROBE_NAME, evidence)
        _log(f"      coverage {evidence.get('coverage')}")

        _log("[3/4] optimize")
        if affine_role_profile:
            _log(f"      affine role profile: {affine_role_profile['name']} ({family})")
        budget_split = None
        if expert_allocation_ratio is not None:
            if target_size_gb is None:
                raise ValueError(
                    "expert_allocation_ratio requires target_size_gb")
            if not (0.0 < expert_allocation_ratio <= 1.0):
                raise ValueError(
                    "expert_allocation_ratio must be in (0, 1], got "
                    f"{expert_allocation_ratio}")
            if not any(u.get("kind") == "expert" for u in evidence.get("units", [])):
                raise ValueError(
                    "expert_allocation_ratio is MoE-only: no expert units in "
                    "evidence")
            budget_split = {"experts": float(expert_allocation_ratio),
                            "affine": 1.0 - float(expert_allocation_ratio)}
        decision = decide(
            evidence, target_quality=target_quality, target_size_gb=target_size_gb,
            tau=tau, alpha=alpha, allow_unhealthy=allow_unhealthy,
            budget_split=budget_split,
            expert_importance_norm=expert_importance_norm,
            affine_role_profile_name=(
                affine_role_profile["name"] if affine_role_profile else None),
            affine_role_weights=affine_role_weights,
            affine_role_bit_weights=affine_role_bit_weights,
            affine_role_min_bits=affine_role_min_bits,
            force_tq4_lossless=force_tq4_lossless,
            force_dense_lossless_mx=force_dense_lossless_mx,
            min_routed_expert_bits=min_routed_expert_bits)
        if write_intermediate:
            write_artifact(out_dir / DECISION_NAME, decision)
        if decision["feasibility"] != "feasible":
            note = _artifact_write_note(
                "Decision",
                out_dir / DECISION_NAME,
                write_intermediate,
            )
            raise RuntimeError(
                f"optimizer infeasible ({decision['feasibility']}); {note}")
        # The allocation health gate marks a serve-unviable allocation invalid
        # (collapsed backbone / critical tensor below floor). Refuse before writing a
        # multi-GB package; the decision is written for inspection when intermediate
        # artifacts are enabled. allow_unhealthy overrides (records the findings as
        # warnings, keeps the decision valid).
        if decision["status"] == "invalid":
            bad = [v["message"] for v in decision["validation"]
                   if v.get("blocking") and v["code"].startswith("optimize.collapsed")]
            note = _artifact_write_note(
                "Decision",
                out_dir / DECISION_NAME,
                write_intermediate,
            )
            raise RuntimeError(
                "allocation failed the health gate (would serve garbage): "
                + "; ".join(bad)
                + f". {note} Re-run with a size budget (e.g. --target-size-gb 17) "
                  "or pass --allow-unhealthy to override.")
        from moespresso.package.plan import (
            package_plan_from_decision,
            parse_force_overrides,
        )

        package_plan, force_summary = package_plan_from_decision(
            decision,
            optimized_kernels_expected=optimized_kernels_expected,
            force_overrides=parse_force_overrides(force_format),
            allow_unmatched_force=allow_unmatched_force,
            dry_run=force_format_dry_run,
        )
        if write_intermediate:
            write_artifact(out_dir / PACKAGE_PLAN_NAME, package_plan)
        if force_format:
            _log(
                f"      force overrides matched {len(force_summary['matched'])} "
                f"tensor(s)")
        if force_format_dry_run:
            return package_plan
        ach = decision["achieved"]
        # sizes are allocated in binary GiB; print decimal GB too because that
        # is what file browsers report (17.06 GiB reads as "18.35 GB" on disk)
        _log(f"      feasible; fidelity={ach['fidelity']:.4f} "
             f"tail={ach['worst_layer_tail']:.4f} "
             f"size={ach['size_gb']:.2f} GiB "
             f"(= {ach['size_gb'] * 2**30 / 1e9:.2f} decimal GB, the Finder "
             f"number)")

        _log(f"[4/4] package (streamed convert) -> {out_dir}")
        # Structural tensors (norms, SSM state) travel directly from the inventory.
        # They remain outside quantization decisions while still serving the graph.
        passthrough = [e for e in inventory["tensors"] if e["kind"] == "passthrough"]
        # Tokenizer is a package contract: copy it in + record its identity so the
        # runtime tokenizes from the package, never the source (spec rendering_id). For a
        # family with a MoEspresso-owned chat template (e.g. qwen3_5_moe), the copy installs
        # it (overwriting the source template) so the package serves the contract template.
        from moespresso.package.agentic_profile import write_agentic_profile
        from moespresso.package.tokenizer import copy_tokenizer_into_package
        tokenizer = copy_tokenizer_into_package(model_dir, out_dir, family=family)
        agentic_profile = write_agentic_profile(out_dir, family=family)
        write_kwargs = {"seed": seed, "shard_size_gb": shard_size_gb,
                        "passthrough": passthrough, "tokenizer": tokenizer,
                        "agentic_profile": agentic_profile,
                        "max_experts": max_experts}
        if family == "deepseek_v4_flash":
            from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup

            write_kwargs["deepseek_v4_expert_group"] = (
                DecodedExpertGroup.from_inventory(inventory, model_dir)
            )
        if chunk_bytes is not None:
            write_kwargs["chunk_bytes"] = chunk_bytes
        if max_experts is not None:
            _log(f"      SMOKE: {max_experts} expert(s)/layer (reduced model)")
        manifest = write_package(package_plan, model_dir, config, out_dir, **write_kwargs)
        package_size_contract = _deepseek_v4_package_size_contract(
            family, model_dir, manifest)
        _log(f"      +{len(passthrough)} passthrough tensors; "
             f"tokenizer {len(tokenizer['files'])} files "
             f"(has_tokenizer={tokenizer['has_tokenizer']})")
        if package_size_contract is not None:
            _log("      [gate] DS4 package size <= source "
                 f"({package_size_contract['package_size_bytes']} <= "
                 f"{package_size_contract['source_size_bytes']} bytes)")
        write_artifact(out_dir / MANIFEST_NAME, manifest)
        # jang-compatible sidecars (config.json, jang_config.json) generated from the
        # manifest: the compat view the proven serve path (load_jangtq_model +
        # tensor_map) consumes. The manifest stays the source of truth.
        from moespresso.package.sidecars import build_sidecars
        config_json, jang_config = build_sidecars(manifest, seed=seed)
        (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))
        (out_dir / "jang_config.json").write_text(json.dumps(jang_config, indent=2))
        _log(f"      wrote {len(manifest['files'])} shard(s), "
             f"{len(manifest['tensors'])} tensors; manifest {manifest['artifact_id'][:16]}; "
             f"+ config.json/jang_config.json sidecars")
        # Cold-start expert hotlist from the imatrix routing counts:
        # serve seeds residency from it when no saved-demand hotlist exists.
        # Alignment failures skip the artifact with a loud warning: a wrong hotlist
        # would silently seed the wrong layers; no hotlist just means a colder start.
        if imatrix_path is not None:
            from moespresso.package.hotlist import (
                HotlistAlignmentError,
                write_package_expert_hotlist,
            )
            try:
                hot_layers = write_package_expert_hotlist(
                    out_dir, imatrix_path,
                    imatrix_identity=calibration[1] if calibration else None)
            except HotlistAlignmentError as e:
                print(f"      [hotlist] SKIPPED (misaligned imatrix counts): {e}",
                      flush=True)
            else:
                if hot_layers:
                    _log(f"      expert hotlist: {hot_layers} layer(s) from "
                         f"imatrix routing counts")

        # Correctness gate: run the ladder rungs the profile declares against the
        # just-written package, locally: the cheap check that catches converter
        # regressions (e.g. conv1d stored pre-transposed -> garbage) without a
        # round-trip to a high-memory parity environment. Format-agnostic: resolve the family's
        # profile; if none is registered, skip with a loud warning (never block an
        # unprofiled family, never apply the wrong contract). A blocking finding refuses
        # the package (the bytes stay for inspection, evidence is written) unless
        # allow_incomplete overrides, mirroring the health gate / allow_unhealthy.
        from moespresso.inventory.architecture_profile import family_of, profile_for
        profile = profile_for(config)
        if profile is None:
            print(f"      [gate] no architecture_profile for family "
                  f"{family_of(config)!r} (model_type={config.get('model_type')!r}), "
                  f"correctness gate SKIPPED. Package written unverified.", flush=True)
        else:
            from moespresso.correctness.gate import run_convert_gate
            # A conv1d is only expected if the model has linear-attention layers; a
            # full-attention-only or smoke build legitimately has none (don't block on that).
            expect_conv1d = bool(layer_types) and "linear_attention" in layer_types
            _log(f"      [gate] correctness ladder (family {profile['family']}, "
                 f"expect_conv1d={expect_conv1d})")
            result = run_convert_gate(
                profile, inventory, manifest, model_dir, out_dir,
                subject={"source_root": str(model_dir), "source_format": "hf_safetensors"},
                expect_conv1d=expect_conv1d)
            ev_dir = out_dir / CORRECTNESS_DIR
            ev_dir.mkdir(exist_ok=True)
            for ev in result.evidence:
                write_artifact(ev_dir / f"{ev['rung']}_evidence.json", ev)
            if result.blocking and not allow_incomplete:
                lines = "; ".join(f"[{r}] {code}: {msg}"
                                  for r, code, msg in result.blocking[:6])
                raise RuntimeError(
                    f"correctness gate FAILED ({len(result.blocking)} blocking finding(s)): "
                    f"{lines}. Evidence written to {ev_dir}. The package would not serve "
                    f"correctly; fix the converter or pass --allow-incomplete to override.")
            if result.blocking:
                print(f"      [gate] {len(result.blocking)} blocking finding(s) OVERRIDDEN "
                      f"by --allow-incomplete; see {ev_dir}", flush=True)
            else:
                _log("      [gate] all rungs passed")

    # Memory footprint is environment-specific, so it goes in the (non-hashed)
    # conversion report (the spec's Phase-4 output, distinct from the manifest),
    # never on the content-addressed manifest (which must stay deterministic).
    mem = rss_summary(rss_samples)
    if mem and write_intermediate:
        report = {
            "memory_rss": mem,
            "inventory_id": inventory["artifact_id"],
            "probe_id": evidence["artifact_id"],
            "decision_id": decision["artifact_id"],
            "package_plan_id": package_plan["artifact_id"],
            "manifest_id": manifest["artifact_id"],
            "required_features": {
                "probe": list(evidence.get("required_features", [])),
                "decision": list(decision.get("required_features", [])),
                "package_plan": list(package_plan.get("required_features", [])),
                "manifest": list(manifest.get("required_features", [])),
            },
        }
        if package_size_contract is not None:
            report["package_size_contract"] = package_size_contract
        (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2))
    if mem:
        _log(f"      memory RSS: peak={mem['peak_gb']:.2f} GB "
             f"median={mem['median_gb']:.2f} p95={mem['p95_gb']:.2f} "
             f"mean={mem['mean_gb']:.2f} (n={mem['samples']})")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """`moespresso-convert <model_dir> <out_dir>`: streamed end-to-end conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="moespresso-convert",
        description="Streamed inventory->probe->optimize->package: convert a model "
                    "to a MoEspresso package (+ manifest) in a few GB of RAM.")
    parser.add_argument("model_dir", help="Source HF model directory (safetensors)")
    parser.add_argument("out_dir", help="Output package directory (e.g. on an SSD)")
    parser.add_argument("--target-quality", type=float, default=None,
                        help="Target importance-weighted fidelity F (e.g. 0.95)")
    parser.add_argument("--target-size-gb", type=float, default=None,
                        help="Target package size in GB")
    parser.add_argument("--expert-allocation-ratio", type=float, default=None,
                        help="MoE only: expert share of the spendable budget "
                             "(0 < ratio <= 1; tuning default 0.80 when set). "
                             "Requires --target-size-gb; incompatible with "
                             "--target-quality and --tau.")
    parser.add_argument("--expert-importance-norm", choices=("class-mean", "off"),
                        default="class-mean",
                        help="MoE only: equalize expert importance class means "
                             "across projections before allocation (default ON; "
                             "raw imatrix means starve down_proj). "
                             "'off' restores raw cross-class comparison.")
    parser.add_argument("--tau", type=float, default=None,
                        help="Worst-layer tail floor (CVaR_alpha >= tau)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Tail fraction for --tau (default 0.05)")
    parser.add_argument("--shard-size-gb", type=float, default=4.0,
                        help="New shard once the current passes this size (default 4)")
    parser.add_argument("--chunk-bytes", type=int, default=None,
                        help="Affine/fp16 row-band cap in bytes (default ~200MB)")
    parser.add_argument("--expert-sample", type=int, default=DEFAULT_EXPERT_SAMPLE,
                        help="Experts sampled per stacked tensor during probe")
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS,
                        help="Rows sampled per tensor during probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imatrix", default=None,
                        help="Imatrix calibration path. Required whenever the target "
                             "format declares it needs calibration (mjtq does).")
    parser.add_argument("--max-experts-per-layer", type=int, default=None,
                        help="Smoke artifact: keep only N experts/layer (a small "
                             "REAL package to convert+serve here before a full run)")
    parser.add_argument("--smoke", action="store_true",
                        help="Shorthand for --max-experts-per-layer 1")
    parser.add_argument("--allow-unhealthy", action="store_true",
                        help="Override the allocation health gate. By default a "
                             "collapsed allocation (backbone/critical tensor at low bits "
                             "-> serves garbage) is refused; this writes it anyway, with "
                             "the findings recorded as warnings. Prefer --target-size-gb.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Override the correctness gate. By default a package "
                             "that fails a ladder rung (L0/L0b/L1/L2: wrong tensor, "
                             "sidecar, TQ bits, gate/up half, conv1d storage, or sampled "
                             "reconstruction) is refused; this writes it anyway, evidence "
                             "recorded under correctness/. For research only.")
    parser.add_argument("--force-tq4-lossless", action="store_true",
                        help="Fallback/A-B switch: keep lossless-capable DeepSeek V4 "
                             "source-FP4 routed experts on TQ4 instead of the default "
                             "native mxfp4 tier.")
    parser.add_argument("--force-dense-lossless-mx", action="store_true",
                        help="Keep source-compatible dense MX-float tensors at their "
                             "near-lossless MX floor. For DeepSeek V4 dense FP8/e4m3 "
                             "tensors this means mxfp8; routed experts are unchanged.")
    parser.add_argument("--min-routed-expert-bits", type=int, choices=(1, 2, 4),
                        default=1,
                        help="Manual routed-expert floor for ablations. Use 2 to "
                             "exclude TQ1 while still allowing TQ2 and the existing "
                             "lossless mxfp4 tier.")
    parser.add_argument("--optimized-kernels-expected", action="store_true",
                        help="Stamp the package plan/manifest as intended for "
                             "optimized kernels. Runtime fast paths still validate "
                             "the tensor formats and shapes they use.")
    parser.add_argument("--force-format", action="append", default=[],
                        metavar="PATTERN=FORMAT",
                        help="Force matched package-plan rows to a format such as "
                             "tq2, tq4, mxfp4, mxfp8, affine4, or kquant:q2_k. "
                             "Matches source, role, projection, GGUF tensor, or "
                             "module path with shell-style wildcards.")
    parser.add_argument("--allow-unmatched-force", action="store_true",
                        help="Allow --force-format patterns that match no tensors.")
    parser.add_argument("--force-format-dry-run", action="store_true",
                        help="Build inventory/probe/decision/package_plan and print "
                             "force matches without writing package shards.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    max_experts = 1 if args.smoke else args.max_experts_per_layer
    if max_experts is not None and max_experts <= 0:
        parser.error("--max-experts-per-layer must be a positive integer")

    # The CLI is format-neutral: it asks the target format whether calibration is
    # required, rather than hardcoding the rule. mjtq (the only format today)
    # declares it, so the CLI never produces an uncalibrated package.
    from moespresso.package.manifest import PACKAGE_FORMAT, PACKAGE_FORMAT_FEATURES
    if args.target_quality is None and args.target_size_gb is None:
        parser.error("provide --target-quality or --target-size-gb")
    if args.expert_allocation_ratio is not None:
        conflicts = [name for name, val in (
            ("--target-quality", args.target_quality),
            ("--tau", args.tau),
        ) if val is not None]
        if conflicts:
            parser.error(
                f"--expert-allocation-ratio is incompatible with "
                f"{', '.join(conflicts)}")
        if args.target_size_gb is None:
            parser.error("--expert-allocation-ratio requires --target-size-gb")
    if "calibration" in PACKAGE_FORMAT_FEATURES and args.imatrix is None:
        parser.error(f"format {PACKAGE_FORMAT!r} requires calibration: "
                     f"pass --imatrix <path>")

    try:
        manifest = convert(
            Path(args.model_dir), Path(args.out_dir),
            imatrix_path=args.imatrix, allow_unhealthy=args.allow_unhealthy,
            allow_incomplete=args.allow_incomplete,
            target_quality=args.target_quality, target_size_gb=args.target_size_gb,
            expert_allocation_ratio=args.expert_allocation_ratio,
            expert_importance_norm=args.expert_importance_norm,
            tau=args.tau, alpha=args.alpha, seed=args.seed,
            shard_size_gb=args.shard_size_gb, chunk_bytes=args.chunk_bytes,
            expert_sample=args.expert_sample, sample_rows=args.sample_rows,
            max_experts=max_experts,
            force_tq4_lossless=args.force_tq4_lossless,
            force_dense_lossless_mx=args.force_dense_lossless_mx,
            min_routed_expert_bits=args.min_routed_expert_bits,
            optimized_kernels_expected=args.optimized_kernels_expected,
            force_format=args.force_format,
            allow_unmatched_force=args.allow_unmatched_force,
            force_format_dry_run=args.force_format_dry_run,
            verbose=True)
    except (RuntimeError, ValueError) as e:
        print(f"FAILED: {e}")
        return 2

    if manifest.get("artifact_kind") == "package_plan":
        from moespresso.package.plan import force_override_preview_lines

        print(f"Dry run: {args.out_dir}")
        print(f"  plan={manifest['artifact_id'][:24]}")
        print(f"  allocations={len(manifest.get('allocation', []))}")
        print(f"  force_overrides={len(manifest.get('force_overrides', []))}")
        for line in force_override_preview_lines(manifest):
            print(line)
        return 0

    # The achieved fidelity/tail/size are logged during convert() (decision phase);
    # here just confirm the written package. The optimizer_decision.json next to the
    # package holds the full achieved metrics.
    print(f"Done: {args.out_dir}")
    shard_bytes = sum(f.get("size_bytes", 0) for f in manifest["files"])
    print(f"  shards={len(manifest['files'])} tensors={len(manifest['tensors'])} "
          f"size={shard_bytes / 2**30:.2f} GiB "
          f"(= {shard_bytes / 1e9:.2f} decimal GB, the Finder number)")
    print(f"  manifest={manifest['artifact_id'][:24]}  "
          f"(metrics in {DECISION_NAME})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
