"""Manual DeepSeek-V4 K-quant package builder.

This path consumes a GGUF tensor-codec recipe plus DS4 source weights and writes
a normal MoEspresso package. The GGUF file supplies routed-expert codec choices
and mapped dense codec choices; unmapped dense tensors keep conservative
MoEspresso storage so the package remains a regular manifest-driven package.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from moespresso.core.artifact import write_artifact
from moespresso.inventory.deepseek_v4 import roles as deepseek_v4_roles
from moespresso.inventory.build import build_inventory
from moespresso.package.kquant_backend import check_kquant_backend_available
from moespresso.package.kquant_cache import KQuantEncodeCache
from moespresso.package.deepseek_v4.kquant import load_ds4_kquant_imatrix_vectors
from moespresso.package.deepseek_v4.recipe import (
    DS4KQuantDenseTarget,
    DS4KQuantExpertTarget,
    build_ds4_expert_kquant_targets,
    build_ds4_kquant_plan,
)
from moespresso.package.kquant_format import KQUANT_GEOMETRY
from moespresso.package.kquant_gguf import GGUFKQuantExpertReader
from moespresso.package.kquant_recipe import (
    KQuantRecipeError,
    read_gguf_kquant_recipe,
    validate_kquant_target_fit,
)
from moespresso.package.plan import force_override_preview_lines, parse_force_overrides
from moespresso.package.convert import INVENTORY_NAME, _layer_types, _read_config
from moespresso.package.constants import MANIFEST_NAME

KQUANT_RECIPE_REPORT_NAME = "kquant_recipe_report.json"
PACKAGE_PLAN_NAME = "package_plan.json"
_DS4_ROUTER_GATE_ROLE = "moe.router_gate"
_FAST_DIAGNOSTIC_FALLBACK_CODEC = "q2_k"
_FAST_DIAGNOSTIC_MODE = {
    "mode": "fast_diagnostic",
    "faithful_ds4c_recipe": False,
    "routed_iquant_override": {
        "from": "iq* (any iq codec, e.g. iq2_xxs)",
        "to": _FAST_DIAGNOSTIC_FALLBACK_CODEC,
        "projections": "all routed (gate, up, down)",
        "reason": (
            "iq codecs have no GPU encoder and are force-pinned to the CPU stream "
            "in mlx-kquant, where the scalar grid-search encode is ~30-75x slower "
            "per tensor and dominates build time. q2_k encodes on the GPU "
            "bit-identically to its CPU encode. This diagnostic recipe differs from "
            "the faithful DS4-c artifact because codec fidelity is intentionally traded "
            "for build speed. Re-enable iq codecs only with the explicit slow flag."
        ),
    },
}


def _is_iquant_codec(codec: str) -> bool:
    return codec.startswith("iq")


def _is_ds4_router_gate(entry: dict) -> bool:
    return entry.get("role") == _DS4_ROUTER_GATE_ROLE


def _kquant_passthrough_tensors(inventory: dict) -> list[dict]:
    """Structural passthrough plus DS4 router gates for the K-quant recipe path."""
    out = [dict(e) for e in inventory.get("tensors", []) if e.get("kind") == "passthrough"]
    for entry in inventory.get("tensors", []):
        if entry.get("kind") != "affine" or not _is_ds4_router_gate(entry):
            continue
        out.append({
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
        })
    return out


def _recipe_mode(fast_diagnostic: bool, keep_iquants: bool = False) -> dict:
    if not fast_diagnostic:
        return {"mode": "faithful_recipe", "faithful_ds4c_recipe": True}
    if keep_iquants:
        # fast-diagnostic invoked with the loud opt-out: iq* codecs were NOT
        # swapped to q2_k, so the package keeps the slow CPU-only iq* encode.
        # Record the preserved iq* codecs accurately; the q2_k override did not run.
        return {
            "mode": "fast_diagnostic",
            "faithful_ds4c_recipe": True,
            "routed_iquant_override": {
                "applied": False,
                "reason": (
                    "--force-very-slow-cpu-iquant-encode kept iq* routed codecs "
                    "(no q2_k override); CPU-only iq* encode, faithful DS4-c codecs"
                ),
            },
        }
    mode = dict(_FAST_DIAGNOSTIC_MODE)
    mode["routed_iquant_override"] = {
        "applied": True,
        **mode["routed_iquant_override"],
    }
    return mode


def _apply_fast_diagnostic_targets(
    targets: list[DS4KQuantExpertTarget],
    *,
    keep_iquants: bool = False,
) -> list[DS4KQuantExpertTarget]:
    """Swap every routed iq* expert codec (gate, up, AND down) to q2_k.

    iq codecs are CPU-only in mlx-kquant and dominate build time. q2_k encodes
    on the GPU bit-identically. Covering all routed projections (not just
    gate/up) is what keeps the whole routed encode off the slow CPU path; the
    down projection was the largest remaining CPU cost when left as iq2_xxs.

    `keep_iquants=True` is the loud opt-out (from --force-very-slow-cpu-iquant-encode):
    it keeps iq* codecs inside a fast-diagnostic build, reintroducing the slow
    CPU-only encode. Only for a faithful artifact, never for iteration.
    """
    if keep_iquants:
        return list(targets)
    out: list[DS4KQuantExpertTarget] = []
    for target in targets:
        if _is_iquant_codec(target.codec):
            out.append(replace(target, codec=_FAST_DIAGNOSTIC_FALLBACK_CODEC))
        else:
            out.append(target)
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
    scale_name = f"{name[:-len('.weight')]}.scale" if name.endswith(".weight") else None
    if entry.get("dtype") == "F8_E4M3" and scale_name in scale_names:
        # Not lossless through the current writer: `mx.quantize(mode="mxfp8")`
        # re-derives each group's e8m0 scale from the group amax, which rounds
        # below the source block scale for some groups and clips their maxima
        # (measured 3955/524288 mismatched elements, rel RMS 0.035, on
        # e4m3-times-2^k lattice data). A true byte repack would need the
        # writer to carry the source e8m0 block scale per group.
        alloc.update({
            "format": "mxfp8",
            "source_codec": "fp8_e4m3_ue8m0",
            "lossless": False,
        })
    else:
        alloc["format"] = "affine"
    return alloc


def _dense_recipe_match(entry: dict, recipe: dict[str, str]) -> tuple[str, str] | None:
    matched = [(key, recipe[key]) for key in entry.get("gguf_keys", []) if key in recipe]
    if len(matched) > 1:
        keys = ", ".join(key for key, _codec in matched)
        raise KQuantRecipeError(
            f"{entry['source_name']}: multiple dense GGUF recipe keys matched: {keys}")
    return matched[0] if matched else None


def _dense_kquant_target(entry: dict, gguf_name: str, codec: str) -> DS4KQuantDenseTarget:
    module_path = deepseek_v4_roles.module_path(entry["source_name"])
    return DS4KQuantDenseTarget(
        source_name=entry["source_name"],
        role=entry["role"],
        layer_index=entry.get("layer_index"),
        codec=codec,
        gguf_tensor=gguf_name,
        imatrix_key=gguf_name,
        module_path=module_path,
        module_weight_key=f"{module_path}.weight",
    )


def _dense_kquant_allocation(target: DS4KQuantDenseTarget) -> dict:
    geometry = KQUANT_GEOMETRY.get(target.codec)
    if geometry is None:
        raise KQuantRecipeError(
            f"{target.gguf_tensor}: unknown kquant codec {target.codec!r}")
    return {
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
    }


def _dense_allocations(
    inventory: dict,
    recipe: dict[str, str],
    imatrix_vectors: dict | None = None,
) -> list[dict]:
    """Non-expert allocation used by the GGUF-recipe path."""
    scale_names = {
        entry["source_name"]
        for entry in inventory.get("tensors", [])
        if entry.get("kind") == "codec_scale"
    }
    out = []
    for entry in inventory.get("tensors", []):
        if entry.get("kind") != "affine":
            continue
        if _is_ds4_router_gate(entry):
            continue
        match = _dense_recipe_match(entry, recipe)
        if match is None:
            out.append(_conservative_dense_allocation(entry, scale_names))
            continue
        gguf_name, codec = match
        target = _dense_kquant_target(entry, gguf_name, codec)
        if imatrix_vectors is not None:
            validate_kquant_target_fit(target, entry.get("shape", []), imatrix_vectors)
        out.append(_dense_kquant_allocation(target))
    return sorted(out, key=lambda a: a["source_name"])


def _blocking_messages(artifact: dict) -> str:
    blocking = [v for v in artifact.get("validation", []) if v.get("blocking")]
    return "; ".join(f"{v['code']}: {v['message']}" for v in blocking[:6])


def _validate_targets_against_source(targets, expert_group, imatrix_vectors) -> None:
    for target in targets:
        experts = expert_group.experts(target.layer_index)
        if not experts:
            raise KQuantRecipeError(
                f"{target.gguf_tensor}: source has no experts for layer "
                f"{target.layer_index}")
        shape = expert_group.logical_shape(
            layer=target.layer_index,
            expert_index=experts[0],
            projection=target.projection,
        )
        validate_kquant_target_fit(target, shape, imatrix_vectors)


def _package_size_bytes(manifest: dict) -> int:
    return sum(int(f.get("size_bytes", 0)) for f in manifest.get("files", []))


def _codec_counts(targets) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in targets:
        counts[target.codec] = counts.get(target.codec, 0) + 1
    return dict(sorted(counts.items()))


def _requires_kquant_backend(allocation: list[dict]) -> bool:
    return any(alloc.get("format") == "kquant" for alloc in allocation)


def _is_expert_recipe_key(gguf_name: str) -> bool:
    return gguf_name.startswith("blk.") and "_exps.weight" in gguf_name


def _dense_recipe_report(recipe: dict[str, str], inventory: dict) -> dict:
    affine_by_gguf: dict[str, dict] = {}
    for entry in inventory.get("tensors", []):
        if entry.get("kind") != "affine":
            continue
        for key in entry.get("gguf_keys", []):
            affine_by_gguf[key] = entry

    codec_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    unmatched: list[str] = []
    matched = 0
    for gguf_name, codec in sorted(recipe.items()):
        if _is_expert_recipe_key(gguf_name):
            continue
        entry = affine_by_gguf.get(gguf_name)
        if entry is None:
            unmatched.append(gguf_name)
            continue
        matched += 1
        codec_counts[codec] = codec_counts.get(codec, 0) + 1
        role = str(entry.get("role"))
        role_counts[role] = role_counts.get(role, 0) + 1

    if unmatched:
        preview = ", ".join(unmatched[:8])
        if len(unmatched) > 8:
            preview += f", ... ({len(unmatched)} total)"
        raise KQuantRecipeError(
            "GGUF recipe contains DS4 non-expert tensor(s) not present in "
            f"source inventory: {preview}")

    return {
        "targets": matched,
        "codec_counts": dict(sorted(codec_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
    }


def _fit_report(targets, expert_group, imatrix_vectors) -> dict:
    shape_counts: dict[tuple[str, str, tuple[int, int]], int] = {}
    imatrix_lengths: dict[int, int] = {}
    for target in targets:
        experts = expert_group.experts(target.layer_index)
        if not experts:
            raise KQuantRecipeError(
                f"{target.gguf_tensor}: source has no experts for layer "
                f"{target.layer_index}")
        shape = tuple(int(v) for v in expert_group.logical_shape(
            layer=target.layer_index,
            expert_index=experts[0],
            projection=target.projection,
        ))
        validate_kquant_target_fit(target, shape, imatrix_vectors)
        shape_key = (target.projection, target.codec, shape)
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
        imatrix = imatrix_vectors.get(target.imatrix_key)
        if imatrix is not None:
            length = int(imatrix.shape[0])
            imatrix_lengths[length] = imatrix_lengths.get(length, 0) + 1
    return {
        "status": "valid",
        "target_shapes": [
            {
                "projection": projection,
                "codec": codec,
                "shape": list(shape),
                "count": count,
            }
            for (projection, codec, shape), count in sorted(shape_counts.items())
        ],
        "imatrix_lengths": {
            str(length): count for length, count in sorted(imatrix_lengths.items())
        },
    }


def _dense_fit_report(
    allocations: list[dict],
    inventory: dict,
    imatrix_vectors: dict,
) -> dict:
    by_source = {
        entry["source_name"]: entry
        for entry in inventory.get("tensors", [])
        if entry.get("kind") == "affine"
    }
    format_counts: dict[str, int] = {}
    shape_counts: dict[tuple[str, str, tuple[int, int]], int] = {}
    imatrix_lengths: dict[int, int] = {}
    kquant_targets = 0
    for alloc in allocations:
        if alloc.get("kind") != "affine":
            continue
        fmt = str(alloc.get("format", "affine"))
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
        if fmt != "kquant":
            continue
        entry = by_source.get(alloc["source_name"])
        if entry is None:
            raise KQuantRecipeError(
                f"{alloc['source_name']}: dense allocation has no inventory entry")
        target = _dense_kquant_target(
            entry,
            str(alloc["gguf_tensor"]),
            str(alloc["kquant_codec"]),
        )
        shape = tuple(int(v) for v in entry.get("shape", []))
        validate_kquant_target_fit(target, shape, imatrix_vectors)
        kquant_targets += 1
        shape_key = (str(alloc["role"]), str(alloc["kquant_codec"]), shape)
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
        imatrix = imatrix_vectors.get(target.imatrix_key)
        if imatrix is not None:
            length = int(imatrix.shape[0])
            imatrix_lengths[length] = imatrix_lengths.get(length, 0) + 1
    return {
        "format_counts": dict(sorted(format_counts.items())),
        "kquant_targets": kquant_targets,
        "target_shapes": [
            {
                "role": role,
                "codec": codec,
                "shape": list(shape),
                "count": count,
            }
            for (role, codec, shape), count in sorted(shape_counts.items())
        ],
        "imatrix_lengths": {
            str(length): count for length, count in sorted(imatrix_lengths.items())
        },
    }


def _write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / KQUANT_RECIPE_REPORT_NAME).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def _is_remote_ref(value: str | Path) -> bool:
    return str(value).startswith(("http://", "https://"))


def preflight_ds4_kquant_package(
    model_dir: str | Path,
    *,
    gguf_recipe_path: str | Path,
    imatrix_path: str | Path,
    fast_diagnostic: bool = False,
    keep_iquants: bool = False,
    verbose: bool = False,
) -> dict:
    """Validate DS4 source, GGUF recipe, and imatrix fit without encoding."""
    from moespresso.inventory.architecture_profile import family_of
    from moespresso.probe.calibration import calibration_identity
    from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup

    model_dir = Path(model_dir)
    imatrix_path = Path(imatrix_path)

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    config = _read_config(model_dir)
    family = family_of(config)
    if family != "deepseek_v4_flash":
        raise ValueError("K-quant recipe packaging is implemented for DeepSeek V4 Flash")

    log(f"[1/4] imatrix {imatrix_path}")
    imatrix_vectors = load_ds4_kquant_imatrix_vectors(imatrix_path)
    imatrix_identity = calibration_identity(imatrix_path, imatrix_vectors)

    log(f"[2/4] inventory {model_dir}")
    inventory = build_inventory(
        model_dir,
        layer_types=_layer_types(config),
        imatrix_keys=set(imatrix_vectors),
        family=family,
    )
    if inventory["status"] == "invalid":
        raise RuntimeError(f"source inventory failed: {_blocking_messages(inventory)}")

    expert_group = DecodedExpertGroup.from_inventory(inventory, model_dir)
    layers = expert_group.layers()
    log(f"[3/4] recipe {gguf_recipe_path}")
    recipe = read_gguf_kquant_recipe(gguf_recipe_path)
    targets = build_ds4_expert_kquant_targets(recipe, required_layers=layers)
    if fast_diagnostic:
        targets = _apply_fast_diagnostic_targets(targets, keep_iquants=keep_iquants)
    dense_recipe = _dense_recipe_report(recipe, inventory)
    dense_allocations = _dense_allocations(inventory, recipe, imatrix_vectors)

    log("[4/4] fit")
    fit = _fit_report(targets, expert_group, imatrix_vectors)
    fit["dense"] = _dense_fit_report(dense_allocations, inventory, imatrix_vectors)
    first_layer = int(layers[0]) if layers else None
    last_layer = int(layers[-1]) if layers else None
    return {
        "recipe": {
            "source": Path(gguf_recipe_path).name,
            "mode": _recipe_mode(fast_diagnostic, keep_iquants),
            "tensor_count": len(recipe),
            "expert_targets": len(targets),
            "expert_codec_counts": _codec_counts(targets),
            "dense": dense_recipe,
        },
        "source": {
            "family": family,
            "inventory_status": inventory["status"],
            "expert_layer_first": first_layer,
            "expert_layer_last": last_layer,
            "expert_layer_count": len(layers),
        },
        "imatrix": imatrix_identity,
        "fit": fit,
        "manual_q1": {"status": "not_run"},
    }


def build_ds4_kquant_package(
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    gguf_recipe_path: str | Path,
    imatrix_path: str | Path,
    seed: int = 42,
    shard_size_gb: float = 4.0,
    chunk_bytes: int | None = None,
    max_experts: int | None = None,
    kquant_encoder=None,
    kquant_cache_dir: str | Path | None = None,
    fast_diagnostic: bool = False,
    keep_iquants: bool = False,
    copy_gguf_expert_bytes: bool = False,
    optimized_kernels_expected: bool = False,
    force_format: list[str] | tuple[str, ...] | None = None,
    allow_unmatched_force: bool = False,
    force_format_dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Build a DS4 package from source weights plus a GGUF K-quant recipe."""
    from moespresso.inventory.architecture_profile import family_of
    from moespresso.package.write import write_package
    from moespresso.probe.calibration import calibration_identity
    from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup
    from moespresso.package.sidecars import build_sidecars
    from moespresso.package.tokenizer import copy_tokenizer_into_package

    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    imatrix_path = Path(imatrix_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if copy_gguf_expert_bytes and _is_remote_ref(gguf_recipe_path):
        raise KQuantRecipeError(
            "copying GGUF expert bytes requires a local GGUF file")

    config = _read_config(model_dir)
    family = family_of(config)
    if family != "deepseek_v4_flash":
        raise ValueError("K-quant recipe packaging is implemented for DeepSeek V4 Flash")

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log(f"[1/5] imatrix {imatrix_path}")
    imatrix_vectors = load_ds4_kquant_imatrix_vectors(imatrix_path)
    imatrix_identity = calibration_identity(imatrix_path, imatrix_vectors)

    log(f"[2/5] inventory {model_dir}")
    inventory = build_inventory(
        model_dir,
        layer_types=_layer_types(config),
        imatrix_keys=set(imatrix_vectors),
        family=family,
    )
    if inventory["status"] == "invalid":
        raise RuntimeError(f"source inventory failed: {_blocking_messages(inventory)}")

    expert_group = DecodedExpertGroup.from_inventory(inventory, model_dir)
    recipe = read_gguf_kquant_recipe(gguf_recipe_path)
    layers = expert_group.layers()
    log(f"[3/5] recipe {gguf_recipe_path}")
    targets = build_ds4_expert_kquant_targets(recipe, required_layers=layers)
    if fast_diagnostic:
        targets = _apply_fast_diagnostic_targets(targets, keep_iquants=keep_iquants)
    if copy_gguf_expert_bytes:
        recipe_targets = build_ds4_expert_kquant_targets(recipe, required_layers=layers)
        expected = {
            (int(target.layer_index), target.projection): target.codec
            for target in recipe_targets
        }
        got = {
            (int(target.layer_index), target.projection): target.codec
            for target in targets
        }
        if got != expected:
            raise KQuantRecipeError(
                "copying GGUF expert bytes requires the routed expert codecs to "
                "match the GGUF recipe; disable --fast-diagnostic or keep iq* "
                "codecs with --force-very-slow-cpu-iquant-encode"
            )
    dense_recipe = _dense_recipe_report(recipe, inventory)
    _validate_targets_against_source(targets, expert_group, imatrix_vectors)
    extra_allocation = _dense_allocations(inventory, recipe, imatrix_vectors)
    dense_fit = _dense_fit_report(extra_allocation, inventory, imatrix_vectors)
    package_plan = build_ds4_kquant_plan(
        inventory["subject"],
        targets,
        recipe_source=Path(gguf_recipe_path).name,
        imatrix_identity=imatrix_identity,
        extra_allocation=extra_allocation,
        diagnostic=_recipe_mode(fast_diagnostic, keep_iquants) if fast_diagnostic else None,
        optimized_kernels_expected=optimized_kernels_expected,
        force_overrides=parse_force_overrides(force_format),
        allow_unmatched_force=allow_unmatched_force,
        dry_run=force_format_dry_run,
    )
    if package_plan["status"] == "invalid":
        raise RuntimeError(
            f"K-quant recipe package plan failed: {_blocking_messages(package_plan)}")
    if force_format_dry_run:
        write_artifact(out_dir / INVENTORY_NAME, inventory)
        write_artifact(out_dir / PACKAGE_PLAN_NAME, package_plan)
        _write_report(out_dir, {
            "status": "valid",
            "package_plan_id": package_plan["artifact_id"],
            "force_overrides": package_plan["force_overrides"],
            "matched": (
                package_plan.get("force_override_preview") or {}).get("matched", []),
            "manual_q1": {"status": "not_run"},
        })
        return package_plan
    if kquant_encoder is None and _requires_kquant_backend(package_plan["allocation"]):
        check_kquant_backend_available()

    write_artifact(out_dir / INVENTORY_NAME, inventory)
    write_artifact(out_dir / PACKAGE_PLAN_NAME, package_plan)

    log(f"[4/5] package {out_dir}")
    cache = KQuantEncodeCache(kquant_cache_dir) if kquant_cache_dir is not None else None
    passthrough = _kquant_passthrough_tensors(inventory)
    tokenizer = copy_tokenizer_into_package(model_dir, out_dir, family=family)
    from moespresso.package.agentic_profile import write_agentic_profile

    agentic_profile = write_agentic_profile(out_dir, family=family)
    kquant_expert_loader = None
    if copy_gguf_expert_bytes:
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
        "passthrough": passthrough,
        "tokenizer": tokenizer,
        "agentic_profile": agentic_profile,
        "max_experts": max_experts,
        "deepseek_v4_expert_group": expert_group,
        "kquant_imatrix_vectors": imatrix_vectors,
        "kquant_encoder": kquant_encoder,
        "kquant_expert_loader": kquant_expert_loader,
        "kquant_cache": cache,
        "kquant_cache_context": {
            "recipe_mode": _recipe_mode(fast_diagnostic, keep_iquants)["mode"],
        },
    }
    if chunk_bytes is not None:
        write_kwargs["chunk_bytes"] = chunk_bytes
    manifest = write_package(package_plan, model_dir, config, out_dir, **write_kwargs)
    write_artifact(out_dir / MANIFEST_NAME, manifest)
    config_json, jang_config = build_sidecars(manifest, seed=seed)
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))
    (out_dir / "jang_config.json").write_text(json.dumps(jang_config, indent=2))

    # Cold-start expert hotlist: serve seeds residency from it when no
    # saved-demand hotlist exists. A counts-bearing (GGUF) build imatrix wins;
    # the legacy .dat build imatrix yields no expert counts, and the vendored
    # DS4 ranking (hotlist_vector.py) is the fallback. Alignment failures skip
    # the artifact with a loud warning: a wrong hotlist would silently seed
    # the wrong layers; no hotlist just means a colder start.
    from moespresso.package.deepseek_v4.hotlist_vector import (
        load_vendored_expert_hotlist,
    )
    from moespresso.package.hotlist import (
        HotlistAlignmentError,
        write_package_expert_hotlist,
        write_package_expert_hotlist_from_payload,
    )

    hotlist_layers = 0
    try:
        hotlist_layers = write_package_expert_hotlist(
            out_dir, imatrix_path, imatrix_identity=imatrix_identity)
        if hotlist_layers:
            log(f"  expert hotlist: {hotlist_layers} layer(s) from imatrix "
                f"routing counts")
        else:
            hotlist_layers = write_package_expert_hotlist_from_payload(
                out_dir, load_vendored_expert_hotlist())
            if hotlist_layers:
                log(f"  expert hotlist: {hotlist_layers} layer(s) from the "
                    f"vendored ranking")
    except HotlistAlignmentError as e:
        print(f"  [hotlist] SKIPPED (misaligned expert counts): {e}",
              flush=True)

    report = {
        "recipe": {
            "source": Path(gguf_recipe_path).name,
            "mode": _recipe_mode(fast_diagnostic, keep_iquants),
            "expert_byte_source": {
                "mode": "gguf_bytes" if copy_gguf_expert_bytes else "source_reencode",
                "name": Path(gguf_recipe_path).name if copy_gguf_expert_bytes else None,
            },
            "tensor_count": len(recipe),
            "expert_targets": len(targets),
            "expert_codec_counts": _codec_counts(targets),
            "dense": dense_recipe,
        },
        "fit": {"dense": dense_fit},
        "imatrix": imatrix_identity,
        "expert_hotlist_layers": hotlist_layers,
        "package_size_bytes": _package_size_bytes(manifest),
        "kquant_cache": None if cache is None else cache.summary(),
        "package_plan_id": package_plan["artifact_id"],
        "manual_q1": {"status": "not_run"},
    }
    _write_report(out_dir, report)
    log("[5/5] done")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """`moespresso-ds4-kquant-package <model_dir> <out_dir> --gguf-recipe ...`."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-kquant-package",
        description="Build a DeepSeek V4 MoEspresso package from a GGUF K-quant recipe.",
    )
    parser.add_argument("model_dir", help="Source DeepSeek V4 HF safetensors directory")
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
    parser.add_argument("--fast-diagnostic", action="store_true",
                        help=("Fast diagnostic recipe: swap every routed iq* expert "
                              "codec (gate, up, AND down) to q2_k, which encodes on "
                              "the GPU bit-identically to its CPU encode. Builds in "
                              "~10 min instead of the faithful recipe's many hours. "
                              "Not the faithful DS4-c artifact."))
    parser.add_argument(
        "--force-very-slow-cpu-iquant-encode", action="store_true",
        help=("DANGER / VERY SLOW: keep iq* codecs (e.g. iq2_xxs) inside a "
              "--fast-diagnostic build instead of swapping them to q2_k. iq* has "
              "NO GPU encoder and is force-pinned to a single-threaded CPU encode, "
              "making the full build MANY HOURS (measured ~8h). Only for a faithful "
              "artifact; never for iteration. No effect without --fast-diagnostic."))
    parser.add_argument("--kquant-cache-dir", default=None,
                        help="Cache encoded K-quant wire tensors by source+codec+imatrix")
    parser.add_argument(
        "--copy-gguf-expert-bytes", action="store_true",
        help=("Copy routed expert K-quant wire bytes directly from --gguf-recipe "
              "instead of re-encoding source FP4 experts. This is byte-faithful "
              "to DS4-c for routed experts; dense tensors still follow the normal "
              "package recipe path."))
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
    if args.force_very_slow_cpu_iquant_encode and not args.fast_diagnostic:
        parser.error(
            "--force-very-slow-cpu-iquant-encode only applies inside a "
            "--fast-diagnostic build (the faithful default already uses iq* codecs)")
    if args.copy_gguf_expert_bytes and args.fast_diagnostic and not args.force_very_slow_cpu_iquant_encode:
        parser.error(
            "--copy-gguf-expert-bytes requires routed codecs to match the GGUF "
            "recipe; use the faithful default or combine --fast-diagnostic with "
            "--force-very-slow-cpu-iquant-encode")
    fast_diagnostic = args.fast_diagnostic

    try:
        if args.preflight_only:
            report = preflight_ds4_kquant_package(
                args.model_dir,
                gguf_recipe_path=args.gguf_recipe,
                imatrix_path=args.imatrix,
                fast_diagnostic=fast_diagnostic,
                keep_iquants=args.force_very_slow_cpu_iquant_encode,
                verbose=args.verbose,
            )
            out_dir = Path(args.out_dir)
            _write_report(out_dir, report)
            print(f"Preflight: {args.out_dir}")
            print(
                f"  targets={report['recipe']['expert_targets']} "
                f"codecs={report['recipe']['expert_codec_counts']}"
            )
            print(f"  report={out_dir / KQUANT_RECIPE_REPORT_NAME}")
            return 0

        manifest = build_ds4_kquant_package(
            args.model_dir,
            args.out_dir,
            gguf_recipe_path=args.gguf_recipe,
            imatrix_path=args.imatrix,
            seed=args.seed,
            shard_size_gb=args.shard_size_gb,
            chunk_bytes=args.chunk_bytes,
            max_experts=max_experts,
            kquant_cache_dir=args.kquant_cache_dir,
            fast_diagnostic=fast_diagnostic,
            keep_iquants=args.force_very_slow_cpu_iquant_encode,
            copy_gguf_expert_bytes=args.copy_gguf_expert_bytes,
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
