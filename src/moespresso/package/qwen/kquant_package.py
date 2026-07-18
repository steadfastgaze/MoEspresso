"""Manual Qwen GGUF K-quant recipe package builder."""

from __future__ import annotations

import json
from pathlib import Path

from moespresso.core.artifact import write_artifact
from moespresso.inventory.build import build_inventory
from moespresso.package.kquant_backend import check_kquant_backend_available
from moespresso.package.kquant_cache import KQuantEncodeCache
from moespresso.package.kquant_gguf import GGUFKQuantExpertReader
from moespresso.package.kquant_recipe import (
    KQuantRecipeError,
    read_gguf_kquant_recipe,
    read_gguf_tensor_types,
    validate_kquant_target_fit,
)
from moespresso.package.qwen.expert_allocation import (
    build_tq_expert_allocations_from_decision,
    load_expert_allocation_decision,
)
from moespresso.package.qwen.recipe import (
    build_dense_kquant_targets,
    build_expert_kquant_targets,
    build_f32_passthrough,
    build_kquant_plan,
    expert_logical_shape,
)
from moespresso.package.plan import force_override_preview_lines, parse_force_overrides
from moespresso.package.write import write_package
from moespresso.probe.calibration import imatrix_calibration
from moespresso.package.convert import _layer_types, _read_config
from moespresso.package.constants import MANIFEST_NAME

KQUANT_RECIPE_REPORT_NAME = "qwen_kquant_recipe_report.json"
PACKAGE_PLAN_NAME = "package_plan.json"
QWEN_FAMILY = "qwen3_5_moe"


def _package_size_bytes(manifest: dict) -> int:
    return sum(int(f.get("size_bytes", 0)) for f in manifest.get("files", []))


def _codec_counts(targets) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in targets:
        counts[target.codec] = counts.get(target.codec, 0) + 1
    return dict(sorted(counts.items()))


def _source_entries_by_name(inventory: dict) -> dict[str, dict]:
    return {
        str(entry["source_name"]): entry
        for entry in inventory.get("tensors", [])
        if entry.get("source_name") is not None
    }


def _validate_targets_against_source(
    *,
    inventory: dict,
    dense_targets,
    expert_targets,
    imatrix_vectors: dict,
) -> dict:
    by_source = _source_entries_by_name(inventory)
    dense_shape_counts: dict[tuple[str, tuple[int, int]], int] = {}
    expert_shape_counts: dict[tuple[str, str, tuple[int, int]], int] = {}
    imatrix_lengths: dict[int, int] = {}

    for target in dense_targets:
        entry = by_source.get(target.source_name)
        if entry is None:
            raise KQuantRecipeError(
                f"{target.gguf_tensor}: missing source tensor {target.source_name}")
        shape = tuple(int(v) for v in entry.get("shape", []))
        validate_kquant_target_fit(target, shape, imatrix_vectors)
        dense_shape_counts[(target.codec, shape)] = (
            dense_shape_counts.get((target.codec, shape), 0) + 1
        )
        imatrix = imatrix_vectors.get(target.imatrix_key)
        if imatrix is not None:
            length = int(imatrix.shape[0])
            imatrix_lengths[length] = imatrix_lengths.get(length, 0) + 1

    for target in expert_targets:
        entry = by_source.get(target.source_name)
        if entry is None:
            raise KQuantRecipeError(
                f"{target.gguf_tensor}: missing source tensor {target.source_name}")
        shape = expert_logical_shape(target, entry.get("shape", []))
        validate_kquant_target_fit(target, shape, imatrix_vectors)
        key = (target.projection, target.codec, tuple(int(v) for v in shape))
        expert_shape_counts[key] = expert_shape_counts.get(key, 0) + 1
        imatrix = imatrix_vectors.get(target.imatrix_key)
        if imatrix is not None:
            length = int(imatrix.shape[0])
            imatrix_lengths[length] = imatrix_lengths.get(length, 0) + 1

    return {
        "status": "valid",
        "dense_target_shapes": [
            {"codec": codec, "shape": list(shape), "count": count}
            for (codec, shape), count in sorted(dense_shape_counts.items())
        ],
        "expert_target_shapes": [
            {
                "projection": projection,
                "codec": codec,
                "shape": list(shape),
                "count": count,
            }
            for (projection, codec, shape), count in sorted(expert_shape_counts.items())
        ],
        "imatrix_lengths": {
            str(length): count for length, count in sorted(imatrix_lengths.items())
        },
    }


def _expert_byte_source(
    *,
    copy_gguf_expert_bytes: bool,
    expert_decision_id: str | None,
    gguf_recipe_path,
) -> dict:
    """Describe where the package's routed-expert content came from."""
    if expert_decision_id is not None:
        return {"mode": "optimizer_decision_tq", "decision_id": expert_decision_id}
    if copy_gguf_expert_bytes:
        return {"mode": "gguf_bytes", "name": Path(gguf_recipe_path).name}
    return {"mode": "source_reencode", "name": None}


def _write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / KQUANT_RECIPE_REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True))


def _is_remote_ref(value: str | Path) -> bool:
    return str(value).startswith(("http://", "https://"))


def _recipe_parts(model_dir, gguf_recipe_path, imatrix_path):
    model_dir = Path(model_dir)
    config = _read_config(model_dir)
    inventory = build_inventory(model_dir, layer_types=_layer_types(config))
    imatrix_vectors, imatrix_identity = imatrix_calibration(imatrix_path)
    recipe = read_gguf_kquant_recipe(gguf_recipe_path)
    tensor_types = read_gguf_tensor_types(gguf_recipe_path)
    dense_targets = build_dense_kquant_targets(recipe, inventory)
    expert_targets = build_expert_kquant_targets(recipe, inventory)
    passthrough = build_f32_passthrough(tensor_types, inventory)
    fit = _validate_targets_against_source(
        inventory=inventory,
        dense_targets=dense_targets,
        expert_targets=expert_targets,
        imatrix_vectors=imatrix_vectors,
    )
    return {
        "config": config,
        "inventory": inventory,
        "imatrix_vectors": imatrix_vectors,
        "imatrix_identity": imatrix_identity,
        "recipe": recipe,
        "tensor_types": tensor_types,
        "dense_targets": dense_targets,
        "expert_targets": expert_targets,
        "passthrough": passthrough,
        "fit": fit,
    }


def _package_subject(inventory: dict, source_identity: str | None) -> dict:
    subject = dict(inventory["subject"])
    if source_identity is None:
        return subject
    source_identity = source_identity.strip()
    if not source_identity:
        raise ValueError("source_identity must not be empty")
    subject["source_root"] = source_identity
    return subject


def preflight_qwen_kquant_package(
    model_dir,
    *,
    gguf_recipe_path,
    imatrix_path,
) -> dict:
    """Validate Qwen source, GGUF recipe, and imatrix fit without encoding."""
    parts = _recipe_parts(model_dir, gguf_recipe_path, imatrix_path)
    return {
        "status": "valid",
        "recipe": {
            "dense_targets": len(parts["dense_targets"]),
            "expert_targets": len(parts["expert_targets"]),
            "dense_codec_counts": _codec_counts(parts["dense_targets"]),
            "expert_codec_counts": _codec_counts(parts["expert_targets"]),
            "f32_passthrough": len(parts["passthrough"]),
        },
        "fit": parts["fit"],
        "imatrix": parts["imatrix_identity"],
    }


def build_qwen_kquant_package(
    model_dir,
    out_dir,
    *,
    gguf_recipe_path,
    imatrix_path,
    seed: int = 42,
    shard_size_gb: float = 4.0,
    chunk_bytes: int | None = None,
    max_experts: int | None = None,
    kquant_cache_dir: str | Path | None = None,
    kquant_encoder=None,
    copy_gguf_expert_bytes: bool = False,
    expert_allocation_from: str | Path | None = None,
    source_identity: str | None = None,
    optimized_kernels_expected: bool = False,
    force_format: list[str] | tuple[str, ...] | None = None,
    allow_unmatched_force: bool = False,
    force_format_dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Build a Qwen package from a GGUF K-quant recipe.

    With `expert_allocation_from` set to an `optimizer_decision` artifact (or the
    package directory that contains one), the routed-expert rows come from that
    TurboQuant decision while the dense/backbone tensors keep the recipe's
    imatrix-calibrated K-quant encode. This is the hybrid arm: K-quant-calibrated
    dense plus TQ experts. Copying GGUF expert bytes cannot be combined with it
    (K-quant wire bytes are not a TQ allocation).
    """
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if copy_gguf_expert_bytes and _is_remote_ref(gguf_recipe_path):
        raise KQuantRecipeError(
            "copying GGUF expert bytes requires a local GGUF file")
    if copy_gguf_expert_bytes and expert_allocation_from is not None:
        raise KQuantRecipeError(
            "--copy-gguf-expert-bytes cannot be combined with "
            "--expert-allocation-from: GGUF K-quant wire bytes are not a "
            "TurboQuant expert allocation")

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log("[1/5] source, recipe, and imatrix preflight")
    parts = _recipe_parts(model_dir, gguf_recipe_path, imatrix_path)
    expert_allocations = None
    expert_decision_id = None
    if expert_allocation_from is not None:
        decision = load_expert_allocation_decision(expert_allocation_from)
        expert_decision_id = decision.get("artifact_id")
        expert_allocations = build_tq_expert_allocations_from_decision(
            decision, parts["inventory"])
        log(
            f"[1/5] routed experts from optimizer_decision {expert_decision_id} "
            f"({len(expert_allocations)} TQ groups); dense stays recipe K-quant")
    package_plan = build_kquant_plan(
        _package_subject(parts["inventory"], source_identity),
        parts["expert_targets"],
        parts["dense_targets"],
        recipe_source=Path(gguf_recipe_path).name,
        imatrix_identity=parts["imatrix_identity"],
        optimized_kernels_expected=optimized_kernels_expected,
        force_overrides=parse_force_overrides(force_format),
        allow_unmatched_force=allow_unmatched_force,
        dry_run=force_format_dry_run,
        expert_allocations=expert_allocations,
        source_decision_id=expert_decision_id,
    )
    if package_plan["status"] != "valid":
        blocking = [
            f"{v['code']}: {v['message']}"
            for v in package_plan.get("validation", [])
            if v.get("blocking")
        ]
        raise KQuantRecipeError(
            "invalid Qwen GGUF K-quant package plan: " + "; ".join(blocking[:6]))
    if force_format_dry_run:
        write_artifact(out_dir / PACKAGE_PLAN_NAME, package_plan)
        _write_report(out_dir, {
            "status": "valid",
            "package_plan_id": package_plan["artifact_id"],
            "force_overrides": package_plan["force_overrides"],
            "matched": (
                package_plan.get("force_override_preview") or {}).get("matched", []),
        })
        return package_plan
    if kquant_encoder is None:
        check_kquant_backend_available()
    write_artifact(out_dir / PACKAGE_PLAN_NAME, package_plan)
    cache = KQuantEncodeCache(kquant_cache_dir) if kquant_cache_dir is not None else None

    log("[2/5] tokenizer")
    from moespresso.package.agentic_profile import write_agentic_profile
    from moespresso.package.tokenizer import copy_tokenizer_into_package

    tokenizer = copy_tokenizer_into_package(
        model_dir,
        out_dir,
        family=QWEN_FAMILY,
    )
    agentic_profile = write_agentic_profile(out_dir, family=QWEN_FAMILY)
    log("[3/5] encode and write package")
    kquant_expert_loader = None
    if copy_gguf_expert_bytes:
        # The routed expert targets come straight from the GGUF recipe (there is
        # no diagnostic mode that rewrites codecs, unlike the DS4 builder), so
        # the target codecs are byte-compatible with the recipe file by
        # construction; the reader still fails closed on any codec mismatch.
        gguf_expert_reader = GGUFKQuantExpertReader(Path(gguf_recipe_path))

        def _load_gguf_expert_bytes(target, expert_index):
            return gguf_expert_reader.load_expert_weight(
                target,
                expert_index=expert_index,
            )

        kquant_expert_loader = _load_gguf_expert_bytes
    write_kwargs = {
        "seed": seed,
        "shard_size_gb": shard_size_gb,
        "passthrough": parts["passthrough"],
        "tokenizer": tokenizer,
        "agentic_profile": agentic_profile,
        "max_experts": max_experts,
        "kquant_imatrix_vectors": parts["imatrix_vectors"],
        "kquant_encoder": kquant_encoder,
        "kquant_expert_loader": kquant_expert_loader,
        "kquant_cache": cache,
        "kquant_cache_context": {
            "recipe_source": Path(gguf_recipe_path).name,
            "recipe_kind": "qwen_gguf_recipe",
        },
    }
    if chunk_bytes is not None:
        write_kwargs["chunk_bytes"] = chunk_bytes
    manifest = write_package(
        package_plan,
        model_dir,
        parts["config"],
        out_dir,
        **write_kwargs,
    )
    write_artifact(out_dir / MANIFEST_NAME, manifest)

    log("[4/5] sidecars and report")
    from moespresso.package.sidecars import build_sidecars

    config_json, jang_config = build_sidecars(manifest, seed=seed)
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))
    (out_dir / "jang_config.json").write_text(json.dumps(jang_config, indent=2))

    # Cold-start expert hotlist from the imatrix routing counts: serve seeds
    # residency from it when no saved-demand hotlist exists. Alignment
    # failures skip the artifact with a loud warning: a wrong hotlist would
    # silently seed the wrong layers; no hotlist just means a colder start.
    from moespresso.package.hotlist import (
        HotlistAlignmentError,
        write_package_expert_hotlist,
    )

    hotlist_layers = 0
    try:
        hotlist_layers = write_package_expert_hotlist(
            out_dir, imatrix_path,
            imatrix_identity=parts["imatrix_identity"])
    except HotlistAlignmentError as e:
        print(f"  [hotlist] SKIPPED (misaligned imatrix counts): {e}",
              flush=True)
    else:
        if hotlist_layers:
            log(f"  expert hotlist: {hotlist_layers} layer(s) from imatrix "
                f"routing counts")

    report = {
        "status": "valid",
        "manifest_id": manifest["artifact_id"],
        "recipe": {
            "dense_targets": len(parts["dense_targets"]),
            "expert_targets": (
                len(expert_allocations) if expert_allocations is not None
                else len(parts["expert_targets"])),
            "dense_codec_counts": _codec_counts(parts["dense_targets"]),
            "expert_codec_counts": (
                package_plan.get("achieved", {}).get("expert_tq_bit_counts")
                if expert_allocations is not None
                else _codec_counts(parts["expert_targets"])),
            "expert_byte_source": _expert_byte_source(
                copy_gguf_expert_bytes=copy_gguf_expert_bytes,
                expert_decision_id=expert_decision_id,
                gguf_recipe_path=gguf_recipe_path,
            ),
            "f32_passthrough": len(parts["passthrough"]),
        },
        "fit": parts["fit"],
        "imatrix": parts["imatrix_identity"],
        "expert_hotlist_layers": hotlist_layers,
        "package_size_bytes": _package_size_bytes(manifest),
        "kquant_cache": None if cache is None else cache.summary(),
        "package_plan_id": package_plan["artifact_id"],
    }
    _write_report(out_dir, report)
    log("[5/5] done")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """`moespresso-qwen-kquant-package <model_dir> <out_dir> --gguf-recipe ...`."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="moespresso-qwen-kquant-package",
        description="Build a Qwen MoEspresso package from a GGUF K-quant recipe.",
    )
    parser.add_argument("model_dir", help="Source Qwen HF safetensors directory")
    parser.add_argument("out_dir", help="Output MoEspresso package directory")
    parser.add_argument("--gguf-recipe", required=True, help="GGUF file used as codec recipe")
    parser.add_argument("--imatrix", required=True, help="llama.cpp imatrix file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size-gb", type=float, default=4.0)
    parser.add_argument("--chunk-bytes", type=int, default=None)
    parser.add_argument("--max-experts-per-layer", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Shorthand for --max-experts-per-layer 1")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate source, recipe, and imatrix fit without encoding")
    parser.add_argument("--kquant-cache-dir", default=None,
                        help="Cache encoded K-quant wire tensors by source+codec+imatrix")
    parser.add_argument(
        "--copy-gguf-expert-bytes", action="store_true",
        help=("Copy routed expert K-quant wire bytes directly from --gguf-recipe "
              "instead of re-encoding source experts. This is byte-faithful to "
              "the recipe GGUF for routed experts; dense tensors still follow "
              "the normal package recipe path."))
    parser.add_argument(
        "--expert-allocation-from", default=None, metavar="PATH",
        help=("Take routed-expert allocations from a TurboQuant optimizer_decision "
              "artifact (a package directory or an optimizer_decision.json). The "
              "experts become TQ per that decision; dense tensors keep the recipe's "
              "imatrix-calibrated K-quant. Cannot be combined with "
              "--copy-gguf-expert-bytes."))
    parser.add_argument(
        "--source-identity", default=None, metavar="IDENTITY",
        help=("Stable source model identity recorded in the package, for example "
              "org/model@revision. Defaults to the source directory name."))
    parser.add_argument("--optimized-kernels-expected", action="store_true",
                        help="Stamp the package as intended for optimized kernels.")
    parser.add_argument("--force-format", action="append", default=[],
                        metavar="PATTERN=FORMAT",
                        help="Force matched package-plan rows to a format such as "
                             "tq2, tq4, mxfp4, mxfp8, affine4, or kquant:q2_k.")
    parser.add_argument("--allow-unmatched-force", action="store_true")
    parser.add_argument("--force-format-dry-run", action="store_true",
                        help="Write package_plan/report and exit before encoding.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    max_experts = 1 if args.smoke else args.max_experts_per_layer
    if max_experts is not None and max_experts <= 0:
        parser.error("--max-experts-per-layer must be a positive integer")

    try:
        if args.preflight_only:
            report = preflight_qwen_kquant_package(
                args.model_dir,
                gguf_recipe_path=args.gguf_recipe,
                imatrix_path=args.imatrix,
            )
            out_dir = Path(args.out_dir)
            _write_report(out_dir, report)
            print(f"Preflight: {args.out_dir}")
            print(
                f"  dense={report['recipe']['dense_targets']} "
                f"experts={report['recipe']['expert_targets']} "
                f"f32={report['recipe']['f32_passthrough']}"
            )
            print(f"  report={out_dir / KQUANT_RECIPE_REPORT_NAME}")
            return 0
        manifest = build_qwen_kquant_package(
            args.model_dir,
            args.out_dir,
            gguf_recipe_path=args.gguf_recipe,
            imatrix_path=args.imatrix,
            seed=args.seed,
            shard_size_gb=args.shard_size_gb,
            chunk_bytes=args.chunk_bytes,
            max_experts=max_experts,
            kquant_cache_dir=args.kquant_cache_dir,
            copy_gguf_expert_bytes=args.copy_gguf_expert_bytes,
            expert_allocation_from=args.expert_allocation_from,
            source_identity=args.source_identity,
            optimized_kernels_expected=args.optimized_kernels_expected,
            force_format=args.force_format,
            allow_unmatched_force=args.allow_unmatched_force,
            force_format_dry_run=args.force_format_dry_run,
            verbose=args.verbose,
        )
    except (RuntimeError, ValueError, KQuantRecipeError) as exc:
        print(f"FAILED: {exc}")
        return 2

    if manifest.get("artifact_kind") == "package_plan":
        print(f"Dry run: {args.out_dir}")
        print(f"  plan={manifest['artifact_id'][:24]}")
        print(f"  allocations={len(manifest.get('allocation', []))}")
        print(f"  force_overrides={len(manifest.get('force_overrides', []))}")
        for line in force_override_preview_lines(manifest):
            print(line)
        return 0

    package_bytes = _package_size_bytes(manifest)
    print(f"Done: {args.out_dir}")
    print(f"  shards={len(manifest['files'])} tensors={len(manifest['tensors'])} "
          f"size={package_bytes / 2**30:.2f} GiB")
    print(f"  manifest={manifest['artifact_id'][:24]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
