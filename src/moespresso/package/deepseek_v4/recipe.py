"""DeepSeek-V4 GGUF K-quant recipe mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass

from moespresso.core.artifact import Validation
from moespresso.package.kquant_format import KQUANT_GEOMETRY
from moespresso.package.kquant_recipe import KQuantRecipeError
from moespresso.package.plan import make_package_plan


@dataclass(frozen=True)
class DS4KQuantExpertTarget:
    layer_index: int
    projection: str
    codec: str
    gguf_tensor: str
    imatrix_key: str
    source_weight_template: str
    source_scale_template: str
    module_path: str
    module_weight_key: str


@dataclass(frozen=True)
class DS4KQuantDenseTarget:
    source_name: str
    role: str
    layer_index: int | None
    codec: str
    gguf_tensor: str
    imatrix_key: str
    module_path: str
    module_weight_key: str
    requires_imatrix: bool = True


_DS4_GGUF_EXPERT_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<projection>gate|up|down)_exps\.weight$"
)
_SOURCE_PROJECTION = {
    "gate": "w1",
    "up": "w3",
    "down": "w2",
}
_PROJECTION_ORDER = {"gate": 0, "up": 1, "down": 2}


def build_ds4_expert_kquant_targets(
    recipe: dict[str, str],
    *,
    required_layers: range | list[int] | tuple[int, ...] | None = None,
    required_projections: tuple[str, ...] = ("gate", "up", "down"),
) -> list[DS4KQuantExpertTarget]:
    """Map a GGUF recipe onto DS4 routed expert targets.

    The GGUF recipe is stacked per layer/projection; the DS4 source checkpoint is
    per expert. The returned templates keep that distinction explicit for the
    later encoder.
    """
    by_key: dict[tuple[int, str], tuple[str, str]] = {}
    for gguf_name, codec in recipe.items():
        match = _DS4_GGUF_EXPERT_RE.match(gguf_name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        projection = match.group("projection")
        by_key[(layer, projection)] = (gguf_name, codec)

    if required_layers is None:
        layers = sorted({layer for layer, _projection in by_key})
    else:
        layers = list(required_layers)
    if not layers:
        raise KQuantRecipeError("GGUF recipe contains no DS4 routed expert tensors")

    out: list[DS4KQuantExpertTarget] = []
    missing: list[str] = []
    for layer in layers:
        for projection in required_projections:
            item = by_key.get((layer, projection))
            if item is None:
                missing.append(f"blk.{layer}.ffn_{projection}_exps.weight")
                continue
            gguf_name, codec = item
            source_projection = _SOURCE_PROJECTION[projection]
            source_base = f"layers.{layer}.ffn.experts.{{expert}}.{source_projection}"
            module_path = f"model.layers.{layer}.mlp.switch_mlp.{projection}_proj"
            out.append(DS4KQuantExpertTarget(
                layer_index=layer,
                projection=projection,
                codec=codec,
                gguf_tensor=gguf_name,
                imatrix_key=gguf_name,
                source_weight_template=f"{source_base}.weight",
                source_scale_template=f"{source_base}.scale",
                module_path=module_path,
                module_weight_key=f"{module_path}.weight",
            ))
    if missing:
        raise KQuantRecipeError(
            "GGUF recipe is missing required DS4 routed expert tensor(s): "
            + ", ".join(missing)
        )
    return out


def build_ds4_kquant_expert_allocations(
    targets: list[DS4KQuantExpertTarget] | tuple[DS4KQuantExpertTarget, ...],
) -> list[dict]:
    """Render DS4 K-quant targets as package-manifest allocation rows."""
    if not targets:
        raise KQuantRecipeError("no DS4 K-quant targets to allocate")
    out = []
    for target in sorted(
        targets,
        key=lambda t: (t.layer_index, _PROJECTION_ORDER.get(t.projection, 99)),
    ):
        geometry = KQUANT_GEOMETRY.get(target.codec)
        if geometry is None:
            raise KQuantRecipeError(
                f"{target.gguf_tensor}: unknown kquant codec {target.codec!r}")
        if target.projection not in _PROJECTION_ORDER:
            raise KQuantRecipeError(
                f"{target.gguf_tensor}: unsupported DS4 projection {target.projection!r}")
        out.append({
            "source_name": f"layers.{target.layer_index}.ffn.experts.{target.projection}",
            "kind": "expert",
            "role": f"moe.expert.{target.projection}",
            "layer_index": int(target.layer_index),
            "projection": target.projection,
            "bits": int(geometry.bits),
            "codec": target.codec,
            "format": "kquant",
            "kquant_codec": target.codec,
            "imatrix_key": target.imatrix_key,
            "gguf_tensor": target.gguf_tensor,
            "module_path": target.module_path,
            "module_weight_key": target.module_weight_key,
            "source_weight_template": target.source_weight_template,
            "source_scale_template": target.source_scale_template,
        })
    return out


def expert_target_from_allocation(alloc: dict) -> DS4KQuantExpertTarget:
    """Rebuild a DS4 expert target from one package-plan allocation row."""
    kcodec = alloc.get("kquant_codec") or alloc.get("codec")
    missing = [
        key for key in (
            "gguf_tensor",
            "imatrix_key",
            "source_weight_template",
            "source_scale_template",
            "module_path",
            "module_weight_key",
        )
        if alloc.get(key) is None
    ]
    if kcodec not in KQUANT_GEOMETRY:
        missing.append("kquant_codec")
    if missing:
        raise ValueError(
            f"K-quant allocation for {alloc.get('source_name')} is missing "
            f"required field(s): {', '.join(missing)}")
    return DS4KQuantExpertTarget(
        layer_index=int(alloc["layer_index"]),
        projection=str(alloc["projection"]),
        codec=str(kcodec),
        gguf_tensor=str(alloc["gguf_tensor"]),
        imatrix_key=str(alloc["imatrix_key"]),
        source_weight_template=str(alloc["source_weight_template"]),
        source_scale_template=str(alloc["source_scale_template"]),
        module_path=str(alloc["module_path"]),
        module_weight_key=str(alloc["module_weight_key"]),
    )


def dense_target_from_allocation(alloc: dict) -> DS4KQuantDenseTarget:
    """Rebuild a DS4 dense target from one package-plan allocation row."""
    kcodec = alloc.get("kquant_codec") or alloc.get("codec")
    missing = [
        key for key in ("gguf_tensor", "imatrix_key", "module_path", "module_weight_key")
        if alloc.get(key) is None
    ]
    if kcodec not in KQUANT_GEOMETRY:
        missing.append("kquant_codec")
    if missing:
        raise ValueError(
            f"K-quant dense allocation for {alloc.get('source_name')} is missing "
            f"required field(s): {', '.join(missing)}")
    return DS4KQuantDenseTarget(
        source_name=str(alloc["source_name"]),
        role=str(alloc["role"]),
        layer_index=(
            None if alloc.get("layer_index") is None else int(alloc["layer_index"])
        ),
        codec=str(kcodec),
        gguf_tensor=str(alloc["gguf_tensor"]),
        imatrix_key=str(alloc["imatrix_key"]),
        module_path=str(alloc["module_path"]),
        module_weight_key=str(alloc["module_weight_key"]),
        requires_imatrix=bool(alloc.get("requires_imatrix", True)),
    )


def build_ds4_kquant_plan(
    subject: dict,
    targets: list[DS4KQuantExpertTarget] | tuple[DS4KQuantExpertTarget, ...],
    *,
    recipe_source: str | None = None,
    imatrix_identity: dict | None = None,
    extra_allocation: list[dict] | tuple[dict, ...] | None = None,
    diagnostic: dict | None = None,
    optimized_kernels_expected: bool = False,
    force_overrides=None,
    allow_unmatched_force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Build the shared package-plan artifact for a DS4 GGUF recipe."""
    validation: list[Validation] = []
    try:
        allocation = build_ds4_kquant_expert_allocations(targets)
        allocation.extend(dict(a) for a in (extra_allocation or ()))
    except KQuantRecipeError as exc:
        allocation = []
        validation.append(Validation(
            "error",
            "kquant_recipe.invalid_targets",
            str(exc),
            phase="kquant_recipe",
            blocking=True,
        ))
    counts: dict[str, int] = {}
    for alloc in allocation:
        if alloc.get("format") == "kquant":
            counts[alloc["kquant_codec"]] = counts.get(alloc["kquant_codec"], 0) + 1
    format_counts: dict[str, int] = {}
    for alloc in allocation:
        fmt = alloc.get("format") or alloc.get("codec") or alloc.get("kind")
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    constraints = {
        "objective": "gguf_recipe_kquant_allocation",
        "recipe_source": recipe_source,
        "imatrix": imatrix_identity,
    }
    if diagnostic is not None:
        constraints["diagnostic"] = diagnostic
    plan, _summary = make_package_plan(
        subject,
        allocation,
        producer_kind="gguf_recipe",
        producer_reference=recipe_source,
        optimized_kernels_expected=optimized_kernels_expected,
        force_overrides=force_overrides,
        allow_unmatched_force=allow_unmatched_force,
        dry_run=dry_run,
        required_features=["calibration"],
        status="valid" if not validation else "invalid",
        validation=validation,
        source_constraints=constraints,
        achieved={
            "expert_codec_counts": dict(sorted(counts.items())),
            "format_counts": dict(sorted(format_counts.items())),
            "expert_format_counts": {"kquant": sum(counts.values())} if counts else {},
        },
    )
    return plan
