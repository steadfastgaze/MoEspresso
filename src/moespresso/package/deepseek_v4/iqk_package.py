"""DeepSeek-V4 IQ_K package builder.

The routed experts of an IQ_K package are not encoded here. A conversion stage
writes one file per (layer, role) holding that cell's experts in index order,
and this builder assembles those bytes into a MoEspresso package: it reads the
allocation that chose each cell's member, checks every artifact file against
the member's own struct arithmetic, streams the expert rows into per-layer
bundles, and allocates the dense side at the requested dense codec with the
same treatment the K-quant recipe path gives it.

Mixed allocations are the normal case: a cell's member is a per-(layer, role)
fact all the way through the plan, the bundle metadata and the manifest, and no
stage carries a package-wide bit width.

The stored bytes are the quantizer's own row-major wire. The manifest records
that as `layout`, so a later build step can rearrange the same encoded bytes
into a decode kernel's layout and record the new value without re-encoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moespresso.core.artifact import write_artifact
from moespresso.inventory.build import build_inventory
from moespresso.inventory.deepseek_v4 import roles as deepseek_v4_roles
from moespresso.package.constants import MANIFEST_NAME
from moespresso.package.source import INVENTORY_NAME, _layer_types, _read_config
from moespresso.package.deepseek_v4.recipe import (
    DS4KQuantDenseTarget,
    build_ds4_iqk_expert_allocations,
    build_ds4_iqk_plan,
)
from moespresso.package.iqk_format import (
    IQK_GEOMETRY,
    IQK_LAYOUT_IK_WIRE,
    IQK_LAYOUT_IQK_RELAYOUT,
    IQKFormatError,
    validate_iqk_layout,
)
from moespresso.package.iqk_artifacts import IQKConvertedArtifacts
from moespresso.package.kquant_backend import check_kquant_backend_available
from moespresso.package.kquant_cache import KQuantEncodeCache
from moespresso.package.kquant_format import KQUANT_GEOMETRY
from moespresso.package.kquant_recipe import KQuantRecipeError, validate_kquant_target_fit
from moespresso.package.plan import parse_force_overrides
from moespresso.package.tokenizer import copy_tokenizer_into_package
from moespresso.package.write import write_package
from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup

IQK_REPORT_NAME = "iqk_package_report.json"
PACKAGE_PLAN_NAME = "package_plan.json"
DEFAULT_DENSE_CODEC = "q8_0"
PROJECTIONS = ("gate", "up", "down")
_DS4_ROUTER_GATE_ROLE = "moe.router_gate"


class IQKPackageError(ValueError):
    """The IQ_K allocation, the converted artifacts, or the dense side is unusable."""


# --------------------------------------------------------------------------
# The allocation


def read_iqk_allocation(path: str | Path, candidate: str) -> tuple[dict[int, dict[str, str]], dict]:
    """Read one candidate's per-(layer, role) member assignment.

    `candidate` matches the candidate record's `name` exactly or by prefix, so
    a caller can name `A1` without repeating the record's full label. An
    ambiguous prefix is an error rather than a first-match.
    """
    data = json.loads(Path(path).read_text())
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise IQKPackageError(f"{path}: no candidate allocations")
    names = [str(c.get("name", "")) for c in candidates]
    matched = [i for i, name in enumerate(names) if name == candidate]
    if not matched:
        matched = [i for i, name in enumerate(names) if name.startswith(candidate)]
    if not matched:
        raise IQKPackageError(
            f"{path}: no candidate named {candidate!r}; have {names}")
    if len(matched) > 1:
        raise IQKPackageError(
            f"{path}: candidate {candidate!r} is ambiguous between "
            f"{[names[i] for i in matched]}")
    record = candidates[matched[0]]
    per_layer = ((record.get("allocation_map") or {}).get("per_layer") or {})
    if not per_layer:
        raise IQKPackageError(
            f"{path}: candidate {record.get('name')!r} carries no per-layer allocation map")
    members: dict[int, dict[str, str]] = {}
    for layer_key, by_role in sorted(per_layer.items(), key=lambda kv: int(kv[0])):
        cells = {str(role): str(codec) for role, codec in by_role.items()}
        if sorted(cells) != ["down", "gate", "up"]:
            raise IQKPackageError(
                f"{path}: layer {layer_key} covers {sorted(cells)}, not gate/up/down")
        for role, codec in cells.items():
            if codec not in IQK_GEOMETRY:
                raise IQKPackageError(
                    f"{path}: layer {layer_key} {role} names {codec!r}, which is not "
                    f"an IQ_K member; known: {sorted(IQK_GEOMETRY)}")
        members[int(layer_key)] = cells
    return members, record


def allocation_member_counts(members: dict[int, dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cells in members.values():
        for codec in cells.values():
            counts[codec] = counts.get(codec, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# The converted routed stack


class IQKRoutedArtifacts(IQKConvertedArtifacts):
    """DeepSeek wrapper preserving the builder's public error type."""

    def __init__(
        self,
        root: str | Path,
        members: dict[int, dict[str, str]],
        shapes: dict[int, dict[str, tuple[int, int]]],
        num_experts: int,
        *,
        max_open: int = 6,
    ):
        super().__init__(
            root,
            members,
            shapes,
            num_experts,
            max_open=max_open,
            error_type=IQKPackageError,
        )


def routed_artifact_shapes(
    expert_group: DecodedExpertGroup,
    layers: list[int] | tuple[int, ...],
) -> dict[int, dict[str, tuple[int, int]]]:
    """Per-(layer, role) logical `[out_features, in_features]` from the source."""
    shapes: dict[int, dict[str, tuple[int, int]]] = {}
    for layer in layers:
        experts = expert_group.experts(layer)
        if not experts:
            raise IQKPackageError(f"source has no experts for layer {layer}")
        shapes[int(layer)] = {
            projection: tuple(
                int(v) for v in expert_group.logical_shape(
                    layer=layer, expert_index=experts[0], projection=projection)
            )
            for projection in PROJECTIONS
        }
    return shapes


# --------------------------------------------------------------------------
# The dense side


def _is_ds4_router_gate(entry: dict) -> bool:
    return entry.get("role") == _DS4_ROUTER_GATE_ROLE


def _passthrough_tensors(inventory: dict) -> list[dict]:
    """Structural passthrough plus DS4 router gates.

    Passthrough entries are copied whole so their inventory-declared `format`
    survives: the router's token-to-expert table is I64 and any rebuild that
    drops the raw format silently casts it.
    """
    out = [dict(e) for e in inventory.get("tensors", []) if e.get("kind") == "passthrough"]
    for entry in inventory.get("tensors", []):
        if entry.get("kind") != "affine" or not _is_ds4_router_gate(entry):
            continue
        out.append(
            {
                "source_name": entry["source_name"],
                "role": entry["role"],
                "kind": "passthrough",
                "layer_index": entry.get("layer_index"),
                "shape": entry.get("shape", []),
                "dtype": entry.get("dtype"),
                "shard": entry.get("shard"),
                "gguf_keys": [],
                "status": entry.get("status", "required"),
                "format": "fp16",
            }
        )
    return out


def _conservative_dense_allocation(entry: dict, scale_names: set[str]) -> dict:
    name = entry["source_name"]
    alloc = {
        "source_name": name,
        "kind": "affine",
        "role": entry["role"],
        "layer_index": entry.get("layer_index"),
        "bits": 8,
        "group_size": 32,
    }
    scale_name = f"{name[: -len('.weight')]}.scale" if name.endswith(".weight") else None
    if entry.get("dtype") == "F8_E4M3" and scale_name in scale_names:
        alloc.update(
            {
                "format": "mxfp8",
                "source_codec": "fp8_e4m3_ue8m0",
                "lossless": False,
            }
        )
    else:
        alloc["format"] = "affine"
    return alloc


def build_dense_allocations(inventory: dict, *, codec: str = DEFAULT_DENSE_CODEC) -> list[dict]:
    """Dense K-quant allocation for a converted routed-IQ_K package.

    The set of dense tensors that carry a GGUF key is exactly the set a
    K-quant recipe maps, so allocating that set at one codec reproduces the
    recipe path's dense treatment without depending on a recipe file. The
    remainder keeps the conservative storage the recipe path also gives it:
    8-bit affine, or mxfp8 for an fp8 source that ships its own block scales.
    The embedding has no GGUF key and therefore stays affine; the head has one
    and takes the codec, which is the split the recipe path produces.
    """
    geometry = KQUANT_GEOMETRY.get(codec)
    if geometry is None:
        raise IQKPackageError(
            f"unknown dense codec {codec!r}; known: {sorted(KQUANT_GEOMETRY)}"
        )
    scale_names = {
        entry["source_name"]
        for entry in inventory.get("tensors", [])
        if entry.get("kind") == "codec_scale"
    }
    out: list[dict] = []
    for entry in inventory.get("tensors", []):
        if entry.get("kind") != "affine" or _is_ds4_router_gate(entry):
            continue
        keys = [k for k in entry.get("gguf_keys", []) if k]
        if not keys:
            out.append(_conservative_dense_allocation(entry, scale_names))
            continue
        if len(keys) > 1:
            raise IQKPackageError(
                f"{entry['source_name']}: {len(keys)} GGUF keys, expected one")
        module_path = deepseek_v4_roles.module_path(entry["source_name"])
        target = DS4KQuantDenseTarget(
            source_name=entry["source_name"],
            role=entry["role"],
            layer_index=entry.get("layer_index"),
            codec=codec,
            gguf_tensor=keys[0],
            imatrix_key=keys[0],
            module_path=module_path,
            module_weight_key=f"{module_path}.weight",
            requires_imatrix=False,
        )
        validate_kquant_target_fit(target, entry.get("shape", []), {})
        out.append(
            {
                "source_name": target.source_name,
                "kind": "affine",
                "role": target.role,
                "layer_index": target.layer_index,
                "bits": int(geometry.bits),
                "group_size": int(geometry.group_size),
                "format": "kquant",
                "codec": target.codec,
                "kquant_codec": target.codec,
                "gguf_tensor": target.gguf_tensor,
                "imatrix_key": target.imatrix_key,
                "module_path": target.module_path,
                "module_weight_key": target.module_weight_key,
                # The writer rebuilds the target from this row and defaults
                # requires_imatrix to true; without the field an
                # imatrix-steered dense codec fails at encode even though the
                # allocation validated as unsteered above.
                "requires_imatrix": target.requires_imatrix,
            }
        )
    return sorted(out, key=lambda a: a["source_name"])


def dense_format_counts(allocation: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for alloc in allocation:
        key = alloc["format"]
        if key == "kquant":
            key = f"kquant:{alloc['kquant_codec']}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# The cold-start hotlist


def capture_expert_counts(capture_dir: str | Path, layers: list[int]) -> dict[int, np.ndarray]:
    """Per-expert route-active token counts from the calibration capture."""
    capture_dir = Path(capture_dir)
    counts: dict[int, np.ndarray] = {}
    for layer in layers:
        path = capture_dir / f"layer{layer:02d}_train.npz"
        if not path.exists():
            raise IQKPackageError(f"calibration capture missing {path}")
        with np.load(path) as data:
            if "gate_up_count" not in data:
                raise IQKPackageError(
                    f"{path}: no gate_up_count vector for the hotlist ranking")
            counts[int(layer)] = np.asarray(data["gate_up_count"], dtype=np.float64)
    return counts


# --------------------------------------------------------------------------
# The build


def _blocking_messages(artifact: dict) -> str:
    blocking = [v for v in artifact.get("validation", []) if v.get("blocking")]
    return "; ".join(f"{v['code']}: {v['message']}" for v in blocking[:6])


def _package_size_bytes(manifest: dict) -> int:
    return sum(int(f.get("size_bytes", 0)) for f in manifest.get("files", []))


def _write_report(out_dir: Path, report: dict) -> None:
    (Path(out_dir) / IQK_REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True))


def build_ds4_iqk_package(
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    routed_artifacts_dir: str | Path,
    allocation_path: str | Path,
    candidate: str,
    routed_inventory_path: str | Path | None = None,
    calibration_capture_dir: str | Path | None = None,
    dense_codec: str = DEFAULT_DENSE_CODEC,
    layout: str = IQK_LAYOUT_IK_WIRE,
    seed: int = 42,
    shard_size_gb: float = 4.0,
    chunk_bytes: int | None = None,
    max_experts: int | None = None,
    kquant_encoder=None,
    kquant_cache_dir: str | Path | None = None,
    optimized_kernels_expected: bool = False,
    force_format: list[str] | tuple[str, ...] | None = None,
    allow_unmatched_force: bool = False,
    force_format_dry_run: bool = False,
    preflight_only: bool = False,
    verbose: bool = False,
) -> dict:
    """Assemble converted IQ_K routed experts plus a q8_0 dense side."""
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    log = (lambda msg: print(msg, flush=True)) if verbose else (lambda msg: None)
    try:
        validate_iqk_layout(layout)
    except IQKFormatError as exc:
        raise IQKPackageError(str(exc)) from exc
    if layout != IQK_LAYOUT_IK_WIRE:
        # This builder stores the conversion artifacts' own bytes, which are
        # on the quantizer's ik wire. Declaring the relayout here would label
        # unrearranged bytes as rearranged: the row width is identical, so no
        # arithmetic catches it, and the installer trusts the label. The
        # relayout is a separate gated step over a built package.
        raise IQKPackageError(
            f"IQ_K wire layout {layout!r} is not a layout this builder can "
            f"store: the converted artifacts are on {IQK_LAYOUT_IK_WIRE!r} and "
            f"nothing here rearranges them. Build on {IQK_LAYOUT_IK_WIRE!r}, "
            f"then run moespresso-ds4-iqk-relayout to reach "
            f"{IQK_LAYOUT_IQK_RELAYOUT!r}")

    config = _read_config(model_dir)
    from moespresso.inventory.architecture_profile import family_of

    family = family_of(config)
    if family != "deepseek_v4_flash":
        raise IQKPackageError(
            f"{model_dir} is not a DeepSeek V4 source (resolved family {family!r})")

    log("[1/5] allocation")
    members, candidate_record = read_iqk_allocation(allocation_path, candidate)
    log(f"  candidate {candidate_record.get('name')!r}: "
        f"{allocation_member_counts(members)}")

    log("[2/5] inventory")
    inventory = build_inventory(
        model_dir,
        layer_types=_layer_types(config),
        family=family,
    )
    if inventory.get("status") == "invalid":
        raise IQKPackageError(f"source inventory failed: {_blocking_messages(inventory)}")
    expert_group = DecodedExpertGroup.from_inventory(inventory, model_dir)
    source_layers = set(expert_group.layers())
    missing = sorted(set(members) - source_layers)
    extra = sorted(source_layers - set(members))
    if missing or extra:
        raise IQKPackageError(
            f"allocation layers do not match the source routed layers "
            f"(allocation-only {missing}, source-only {extra})")
    shapes = routed_artifact_shapes(expert_group, sorted(members))
    num_experts = len(expert_group.experts(sorted(members)[0]))

    log("[3/5] converted routed stack")
    artifacts = IQKRoutedArtifacts(
        routed_artifacts_dir, members, shapes, num_experts)
    log(f"  {artifacts.identity()}")
    digest_report = None
    if routed_inventory_path is not None:
        digest_report = artifacts.verify_digests(routed_inventory_path)
        log(f"  digests: {digest_report['files_checked']} file(s) match the conversion")

    expert_allocation = build_ds4_iqk_expert_allocations(members, layout=layout)
    dense_allocation = build_dense_allocations(inventory, codec=dense_codec)
    dense_counts = dense_format_counts(dense_allocation)
    log(f"  dense: {dense_counts}")

    subject = dict(inventory["subject"])
    artifacts_identity = {
        **artifacts.identity(),
        "allocation_candidate": candidate_record.get("name"),
        "allocation_accounting": candidate_record.get("accounting"),
        "digests": digest_report,
    }
    if digest_report is not None:
        artifacts_identity["inventory_sha256"] = digest_report["inventory_sha256"]
    package_plan = build_ds4_iqk_plan(
        subject,
        expert_allocation,
        artifacts_identity=artifacts_identity,
        allocation_source=Path(allocation_path).name,
        extra_allocation=dense_allocation,
        optimized_kernels_expected=optimized_kernels_expected,
        force_overrides=parse_force_overrides(force_format or []),
        allow_unmatched_force=allow_unmatched_force,
        dry_run=force_format_dry_run,
    )
    if package_plan.get("status") == "invalid":
        raise IQKPackageError(f"package plan failed: {_blocking_messages(package_plan)}")

    if preflight_only:
        artifacts.close()
        return {
            "status": "preflight",
            "candidate": candidate_record.get("name"),
            "member_counts": allocation_member_counts(members),
            "routed": artifacts.identity(),
            "routed_digests": digest_report,
            "dense_format_counts": dense_counts,
            "package_plan_id": package_plan["artifact_id"],
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    if force_format_dry_run:
        write_artifact(out_dir / INVENTORY_NAME, inventory)
        write_artifact(out_dir / PACKAGE_PLAN_NAME, package_plan)
        artifacts.close()
        return package_plan

    if kquant_encoder is None and any(
            a.get("format") == "kquant" for a in package_plan["allocation"]):
        check_kquant_backend_available()
    write_artifact(out_dir / INVENTORY_NAME, inventory)
    write_artifact(out_dir / PACKAGE_PLAN_NAME, package_plan)

    log(f"[4/5] package {out_dir}")
    cache = KQuantEncodeCache(kquant_cache_dir) if kquant_cache_dir is not None else None
    passthrough = _passthrough_tensors(inventory)
    tokenizer = copy_tokenizer_into_package(model_dir, out_dir, family=family)
    from moespresso.package.agentic_profile import write_agentic_profile

    agentic_profile = write_agentic_profile(out_dir, family=family)
    write_kwargs = {
        "shard_size_gb": shard_size_gb,
        "passthrough": passthrough,
        "tokenizer": tokenizer,
        "agentic_profile": agentic_profile,
        "max_experts": max_experts,
        "deepseek_v4_expert_group": expert_group,
        "kquant_encoder": kquant_encoder,
        "kquant_cache": cache,
        "kquant_cache_context": {"recipe_mode": "iqk_converted_artifacts"},
        "iqk_expert_loader": artifacts.expert_blocks,
    }
    if chunk_bytes is not None:
        write_kwargs["chunk_bytes"] = chunk_bytes
    manifest = write_package(package_plan, model_dir, config, out_dir, **write_kwargs)
    artifacts.close()
    write_artifact(out_dir / MANIFEST_NAME, manifest)

    from moespresso.package.sidecars import build_sidecars

    config_json, jang_config = build_sidecars(manifest, seed=seed)
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))
    (out_dir / "jang_config.json").write_text(json.dumps(jang_config, indent=2))

    hotlist_layers, hotlist_source = _write_hotlist(
        out_dir,
        calibration_capture_dir,
        sorted(members),
        log=log,
    )

    report = {
        "allocation": {
            "source": str(allocation_path),
            "candidate": candidate_record.get("name"),
            "member_counts": allocation_member_counts(members),
            "declared_accounting": candidate_record.get("accounting"),
        },
        "routed": artifacts.identity(),
        "routed_digests": digest_report,
        "wire_layout": layout,
        "dense": {"codec": dense_codec, "format_counts": dense_counts},
        "expert_hotlist": {"layers": hotlist_layers, "source": hotlist_source},
        "package_size_bytes": _package_size_bytes(manifest),
        "kquant_cache": None if cache is None else cache.summary(),
        "package_plan_id": package_plan["artifact_id"],
        "package_manifest_id": manifest["artifact_id"],
    }
    _write_report(out_dir, report)
    log("[5/5] done")
    return manifest


def _write_hotlist(
    out_dir: Path,
    calibration_capture_dir: str | Path | None,
    layers: list[int],
    *,
    log,
) -> tuple[int, str | None]:
    """Cold-start expert prewarm ranking, from this package's own calibration.

    A misaligned ranking would seed the wrong layers' experts, so an alignment
    failure writes nothing and says why; a package with no hotlist starts
    colder, which is recoverable, and a wrong one is not.
    """
    from moespresso.package.deepseek_v4.hotlist_vector import load_vendored_expert_hotlist
    from moespresso.package.hotlist import (
        HotlistAlignmentError,
        build_package_expert_hotlist,
        write_package_expert_hotlist_from_payload,
    )
    from moespresso.runtime.expert_index import build_expert_index

    if calibration_capture_dir is not None:
        try:
            counts = capture_expert_counts(calibration_capture_dir, layers)
            index = build_expert_index(out_dir)
            payload = build_package_expert_hotlist(
                counts,
                layers_indexed=index.layers_indexed(),
                num_experts=index.num_experts,
                source={
                    "kind": "route_active_calibration_counts",
                    "capture": str(calibration_capture_dir),
                    "use": "cold-start expert prewarm ranking only",
                },
            )
            (out_dir / "expert_hotlist.json").write_text(json.dumps(payload))
            n = len(payload["layers"])
            log(f"  expert hotlist: {n} layer(s) from route-active calibration counts")
            return n, "route_active_calibration_counts"
        except (HotlistAlignmentError, IQKPackageError, ValueError) as e:
            print(f"  [hotlist] calibration counts unusable: {e}", flush=True)
    try:
        n = write_package_expert_hotlist_from_payload(
            out_dir, load_vendored_expert_hotlist())
        if n:
            log(f"  expert hotlist: {n} layer(s) from the vendored ranking")
        return n, "vendored" if n else None
    except HotlistAlignmentError as e:
        print(f"  [hotlist] SKIPPED (misaligned expert counts): {e}", flush=True)
        return 0, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a DeepSeek-V4 package from converted IQ_K routed experts")
    parser.add_argument("model_dir", help="Source DeepSeek V4 HF safetensors directory")
    parser.add_argument("out_dir", help="Output MoEspresso package directory")
    parser.add_argument(
        "--routed-artifacts", required=True,
        help="Directory of converted layer<LL>_<role>.<member> cell files")
    parser.add_argument(
        "--allocation", required=True,
        help="Candidate allocation JSON carrying allocation_map.per_layer")
    parser.add_argument(
        "--candidate", required=True,
        help="Candidate name (exact, or an unambiguous prefix)")
    parser.add_argument(
        "--routed-inventory", default=None,
        help="Conversion inventory JSON; when given, every artifact file must "
             "reproduce its recorded size and sha256")
    parser.add_argument(
        "--calibration-capture", default=None,
        help="Calibration capture imatrix directory (layer<LL>_train.npz) used "
             "for the cold-start expert hotlist ranking")
    parser.add_argument(
        "--dense-codec",
        choices=sorted(KQUANT_GEOMETRY),
        default=DEFAULT_DENSE_CODEC,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size-gb", type=float, default=4.0)
    parser.add_argument("--chunk-bytes", type=int, default=None)
    parser.add_argument("--max-experts-per-layer", type=int, default=None)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Shorthand for --max-experts-per-layer 1")
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Validate the allocation, the converted artifacts and the dense "
             "side without writing a package")
    parser.add_argument("--kquant-cache-dir", default=None)
    parser.add_argument("--optimized-kernels-expected", action="store_true")
    parser.add_argument("--force-format", action="append", default=[])
    parser.add_argument("--allow-unmatched-force", action="store_true")
    parser.add_argument("--force-format-dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    max_experts = args.max_experts_per_layer
    if args.smoke:
        max_experts = 1 if max_experts is None else max_experts
    if max_experts is not None and max_experts < 1:
        parser.error("--max-experts-per-layer must be >= 1")

    try:
        result = build_ds4_iqk_package(
            args.model_dir,
            args.out_dir,
            routed_artifacts_dir=args.routed_artifacts,
            allocation_path=args.allocation,
            candidate=args.candidate,
            routed_inventory_path=args.routed_inventory,
            calibration_capture_dir=args.calibration_capture,
            dense_codec=args.dense_codec,
            seed=args.seed,
            shard_size_gb=args.shard_size_gb,
            chunk_bytes=args.chunk_bytes,
            max_experts=max_experts,
            kquant_cache_dir=args.kquant_cache_dir,
            optimized_kernels_expected=args.optimized_kernels_expected,
            force_format=args.force_format,
            allow_unmatched_force=args.allow_unmatched_force,
            force_format_dry_run=args.force_format_dry_run,
            preflight_only=args.preflight_only,
            verbose=True,
        )
    except (RuntimeError, ValueError, KQuantRecipeError) as exc:
        print(f"error: {exc}", flush=True)
        return 2
    if args.preflight_only:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0
