"""Qwen GGUF K-quant recipe mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moespresso.core.artifact import Validation
from moespresso.inventory import roles
from moespresso.package.kquant_format import KQUANT_GEOMETRY
from moespresso.package.kquant_recipe import KQuantRecipeError
from moespresso.package.plan import make_package_plan


@dataclass(frozen=True)
class QwenKQuantExpertTarget:
    layer_index: int
    projection: str
    codec: str
    gguf_tensor: str
    imatrix_key: str
    source_name: str
    source_projection: str
    module_path: str
    module_weight_key: str


@dataclass(frozen=True)
class QwenKQuantDenseTarget:
    source_name: str
    role: str
    layer_index: int | None
    codec: str
    gguf_tensor: str
    imatrix_key: str
    module_path: str
    module_weight_key: str
    requires_imatrix: bool = True


_PROJECTION_ORDER = {"gate": 0, "up": 1, "down": 2}
_QWEN_EXPERT_REQUIRED = ("gate", "up", "down")
_QWEN_SOURCE_PROJECTION = {
    "gate": "gate_up",
    "up": "gate_up",
    "down": "down",
}
_QWEN_GLOBAL_GGUF_KEYS = {
    "model.language_model.embed_tokens.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "model.language_model.norm.weight": "output_norm.weight",
}
_QWEN_STRUCTURAL_GGUF_SUFFIXES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
}


def module_path(source_name: str) -> str:
    """Qwen source tensor name -> installed MLX module path without `.weight`."""
    name = source_name[:-len(".weight")] if source_name.endswith(".weight") else source_name
    if name == "lm_head":
        return "language_model.lm_head"
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    return name


def source_gguf_key(entry: dict) -> str | None:
    """GGUF tensor name corresponding to a Qwen inventory entry, when known."""
    keys = entry.get("gguf_keys") or []
    if len(keys) == 1:
        return str(keys[0])
    name = str(entry.get("source_name"))
    global_key = _QWEN_GLOBAL_GGUF_KEYS.get(name)
    if global_key is not None:
        return global_key
    layer = roles.tensor_layer(name)
    if layer is None:
        return None
    marker = f".layers.{layer}."
    if marker in name:
        suffix = name[name.index(marker) + len(marker):]
    else:
        marker = f"layers.{layer}."
        if marker not in name:
            return None
        suffix = name[name.index(marker) + len(marker):]
    gguf_suffix = _QWEN_STRUCTURAL_GGUF_SUFFIXES.get(suffix)
    if gguf_suffix is None:
        return None
    return f"blk.{layer}.{gguf_suffix}"


def _expert_entries_by_layer(inventory: dict) -> dict[tuple[int, str], dict]:
    out: dict[tuple[int, str], dict] = {}
    for entry in inventory.get("tensors", []):
        if entry.get("kind") != "expert":
            continue
        projection = str(entry.get("projection"))
        if projection not in {"gate_up", "down"}:
            continue
        layer = entry.get("layer_index")
        if layer is None:
            continue
        out[(int(layer), projection)] = entry
    return out


def build_expert_kquant_targets(
    recipe: dict[str, str],
    inventory: dict,
    *,
    required_layers: range | list[int] | tuple[int, ...] | None = None,
) -> list[QwenKQuantExpertTarget]:
    """Map a GGUF recipe onto Qwen3.5/3.6 stacked expert targets.

    Qwen stores gate and up as one fused source tensor
    `mlp.experts.gate_up_proj`. The GGUF recipe still names three logical
    projections (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`), so the mapper
    intentionally emits separate gate/up targets that share one source tensor.
    """
    entries = _expert_entries_by_layer(inventory)
    if required_layers is None:
        layers = sorted({layer for layer, _projection in entries})
    else:
        layers = list(required_layers)
    if not layers:
        raise KQuantRecipeError("inventory contains no Qwen routed expert tensors")

    out: list[QwenKQuantExpertTarget] = []
    missing: list[str] = []
    for layer in layers:
        source_by_projection = {
            "gate": entries.get((layer, "gate_up")),
            "up": entries.get((layer, "gate_up")),
            "down": entries.get((layer, "down")),
        }
        for projection in _QWEN_EXPERT_REQUIRED:
            source_entry = source_by_projection[projection]
            gguf_name = f"blk.{layer}.ffn_{projection}_exps.weight"
            codec = recipe.get(gguf_name)
            if source_entry is None:
                missing.append(
                    f"Qwen source layer {layer} projection "
                    f"{_QWEN_SOURCE_PROJECTION[projection]}")
                continue
            if codec is None:
                missing.append(gguf_name)
                continue
            module = roles.switch_mlp_key(source_entry["source_name"], projection)
            out.append(QwenKQuantExpertTarget(
                layer_index=int(layer),
                projection=projection,
                codec=codec,
                gguf_tensor=gguf_name,
                imatrix_key=gguf_name,
                source_name=source_entry["source_name"],
                source_projection=str(source_entry["projection"]),
                module_path=module,
                module_weight_key=f"{module}.weight",
            ))
    if missing:
        raise KQuantRecipeError(
            "GGUF recipe/inventory is missing required Qwen routed expert "
            "target(s): " + ", ".join(missing)
        )
    return sorted(out, key=lambda t: (t.layer_index, _PROJECTION_ORDER[t.projection]))


def build_dense_kquant_targets(
    recipe: dict[str, str],
    inventory: dict,
) -> list[QwenKQuantDenseTarget]:
    """Map GGUF K-quant non-expert recipe rows onto Qwen dense source tensors."""
    out: list[QwenKQuantDenseTarget] = []
    seen_gguf: set[str] = set()
    for entry in inventory.get("tensors", []):
        if entry.get("kind") != "affine":
            continue
        gguf_name = source_gguf_key(entry)
        if gguf_name is None:
            continue
        codec = recipe.get(gguf_name)
        if codec is None:
            continue
        source_name = str(entry["source_name"])
        module = module_path(source_name)
        out.append(QwenKQuantDenseTarget(
            source_name=source_name,
            role=str(entry["role"]),
            layer_index=(
                None if entry.get("layer_index") is None
                else int(entry["layer_index"])
            ),
            codec=codec,
            gguf_tensor=gguf_name,
            imatrix_key=gguf_name,
            module_path=module,
            module_weight_key=f"{module}.weight",
            requires_imatrix=gguf_name not in {"token_embd.weight", "output.weight"},
        ))
        seen_gguf.add(gguf_name)

    missing_globals = [
        gguf_name
        for gguf_name in ("token_embd.weight", "output.weight")
        if gguf_name in recipe and gguf_name not in seen_gguf
    ]
    if missing_globals:
        raise KQuantRecipeError(
            "GGUF recipe is missing source inventory entries for Qwen global "
            "K-quant tensor(s): " + ", ".join(missing_globals)
        )
    return sorted(out, key=lambda t: (
        -1 if t.layer_index is None else t.layer_index,
        t.module_weight_key,
    ))


def build_f32_passthrough(
    tensor_types: dict[str, str],
    inventory: dict,
) -> list[dict]:
    """Return package passthrough rows for Qwen tensors stored as F32 in GGUF."""
    out: list[dict] = []
    for entry in inventory.get("tensors", []):
        if entry.get("kind") not in {"affine", "passthrough"}:
            continue
        gguf_name = source_gguf_key(entry)
        if gguf_name is None or tensor_types.get(gguf_name) != "F32":
            continue
        out.append({
            "source_name": entry["source_name"],
            "role": entry["role"],
            "kind": "passthrough",
            "layer_index": entry.get("layer_index"),
            "format": "f32_passthrough",
            "gguf_tensor": gguf_name,
        })

    seen = {row["gguf_tensor"] for row in out}
    missing = [
        gguf_name
        for gguf_name in ("output_norm.weight",)
        if tensor_types.get(gguf_name) == "F32" and gguf_name not in seen
    ]
    if missing:
        raise KQuantRecipeError(
            "GGUF recipe is missing source inventory entries for Qwen F32 "
            "passthrough tensor(s): " + ", ".join(missing)
        )
    return sorted(out, key=lambda row: (
        -1 if row.get("layer_index") is None else int(row["layer_index"]),
        str(row["source_name"]),
    ))


def expert_logical_shape(
    target: QwenKQuantExpertTarget,
    source_shape: tuple[int, int, int] | list[int],
) -> tuple[int, int]:
    """Return logical `[out_features, in_features]` for one Qwen expert target."""
    if len(source_shape) != 3:
        raise KQuantRecipeError(
            f"{target.gguf_tensor}: expected stacked source shape [experts,out,in], "
            f"got {list(source_shape)}")
    _experts, out_features, in_features = (int(v) for v in source_shape)
    if target.projection in {"gate", "up"}:
        if target.source_projection != "gate_up" or out_features % 2:
            raise KQuantRecipeError(
                f"{target.gguf_tensor}: cannot split Qwen gate_up source shape "
                f"{list(source_shape)} for projection {target.projection!r}")
        return out_features // 2, in_features
    if target.projection == "down":
        return out_features, in_features
    raise KQuantRecipeError(
        f"{target.gguf_tensor}: unsupported Qwen projection {target.projection!r}")


def build_expert_kquant_allocations(
    targets: list[QwenKQuantExpertTarget] | tuple[QwenKQuantExpertTarget, ...],
) -> list[dict]:
    """Render Qwen K-quant expert targets as package-manifest allocation rows."""
    if not targets:
        raise KQuantRecipeError("no Qwen K-quant targets to allocate")
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
                f"{target.gguf_tensor}: unsupported Qwen projection "
                f"{target.projection!r}")
        out.append({
            "source_name": target.source_name,
            "kind": "expert",
            "role": f"moe.expert.{target.projection}",
            "layer_index": int(target.layer_index),
            "projection": target.projection,
            "source_projection": target.source_projection,
            "bits": int(geometry.bits),
            "codec": target.codec,
            "format": "kquant",
            "kquant_codec": target.codec,
            "imatrix_key": target.imatrix_key,
            "gguf_tensor": target.gguf_tensor,
            "module_path": target.module_path,
            "module_weight_key": target.module_weight_key,
        })
    return out


def expert_target_from_allocation(alloc: dict) -> QwenKQuantExpertTarget:
    """Rebuild a Qwen expert target from one package-plan allocation row."""
    kcodec = alloc.get("kquant_codec") or alloc.get("codec")
    missing = [
        key for key in (
            "gguf_tensor",
            "imatrix_key",
            "source_projection",
            "module_path",
            "module_weight_key",
        )
        if alloc.get(key) is None
    ]
    if kcodec not in KQUANT_GEOMETRY:
        missing.append("kquant_codec")
    if missing:
        raise ValueError(
            f"Qwen K-quant allocation for {alloc.get('source_name')} is missing "
            f"required field(s): {', '.join(missing)}")
    return QwenKQuantExpertTarget(
        layer_index=int(alloc["layer_index"]),
        projection=str(alloc["projection"]),
        codec=str(kcodec),
        gguf_tensor=str(alloc["gguf_tensor"]),
        imatrix_key=str(alloc["imatrix_key"]),
        source_name=str(alloc["source_name"]),
        source_projection=str(alloc["source_projection"]),
        module_path=str(alloc["module_path"]),
        module_weight_key=str(alloc["module_weight_key"]),
    )


def expert_submatrix(expert: np.ndarray, target: QwenKQuantExpertTarget) -> np.ndarray:
    """Return the logical expert matrix for a Qwen gate, up, or down target."""
    if target.projection in {"gate", "up"}:
        if target.source_projection != "gate_up" or expert.shape[0] % 2:
            raise ValueError(
                f"{target.gguf_tensor}: cannot split Qwen gate_up expert shape "
                f"{list(expert.shape)} for projection {target.projection!r}")
        mid = expert.shape[0] // 2
        return expert[:mid] if target.projection == "gate" else expert[mid:]
    if target.projection == "down":
        return expert
    raise ValueError(
        f"{target.gguf_tensor}: unsupported Qwen projection {target.projection!r}")


def build_dense_kquant_allocations(
    targets: list[QwenKQuantDenseTarget] | tuple[QwenKQuantDenseTarget, ...],
) -> list[dict]:
    """Render Qwen dense K-quant targets as package-manifest allocation rows."""
    if not targets:
        raise KQuantRecipeError("no Qwen dense K-quant targets to allocate")
    out = []
    for target in sorted(
        targets,
        key=lambda t: (
            -1 if t.layer_index is None else t.layer_index,
            t.module_weight_key,
        ),
    ):
        geometry = KQUANT_GEOMETRY.get(target.codec)
        if geometry is None:
            raise KQuantRecipeError(
                f"{target.gguf_tensor}: unknown kquant codec {target.codec!r}")
        out.append({
            "source_name": target.source_name,
            "kind": "affine",
            "role": target.role,
            "layer_index": target.layer_index,
            "bits": int(geometry.bits),
            "group_size": int(geometry.group_size),
            "codec": target.codec,
            "format": "kquant",
            "kquant_codec": target.codec,
            "imatrix_key": target.imatrix_key,
            "gguf_tensor": target.gguf_tensor,
            "module_path": target.module_path,
            "module_weight_key": target.module_weight_key,
            "requires_imatrix": bool(target.requires_imatrix),
        })
    return out


def build_kquant_plan(
    subject: dict,
    expert_targets: list[QwenKQuantExpertTarget] | tuple[QwenKQuantExpertTarget, ...],
    dense_targets: list[QwenKQuantDenseTarget] | tuple[QwenKQuantDenseTarget, ...],
    *,
    recipe_source: str | None = None,
    imatrix_identity: dict | None = None,
    diagnostic: dict | None = None,
    optimized_kernels_expected: bool = False,
    force_overrides=None,
    allow_unmatched_force: bool = False,
    dry_run: bool = False,
    expert_allocations: list[dict] | tuple[dict, ...] | None = None,
    source_decision_id: str | None = None,
) -> dict:
    """Build the shared package-plan artifact for a Qwen GGUF recipe.

    The GGUF recipe supplies the dense allocation. MoEspresso's optimizer may
    supply the routed-expert override described below.
    F32 passthrough tensors travel separately through
    `write_package(..., passthrough=...)`.

    `expert_allocations` overrides the recipe's routed-expert rows with rows
    consumed from an `optimizer_decision` (the TurboQuant hybrid path). The dense
    rows stay the recipe's imatrix-calibrated K-quant. `source_decision_id`
    records the cited decision on the plan so the manifest provenance shows the
    honest mixed source; the plan producer stays `gguf_recipe` (a recipe path
    does not emit an optimizer decision, it consumes and cites one).
    """
    validation: list[Validation] = []
    try:
        allocation = build_dense_kquant_allocations(dense_targets)
        if expert_allocations is None:
            allocation.extend(build_expert_kquant_allocations(expert_targets))
        else:
            allocation.extend(dict(row) for row in expert_allocations)
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
    expert_counts: dict[str, int] = {}
    dense_counts: dict[str, int] = {}
    tq_expert_counts: dict[str, int] = {}
    format_counts: dict[str, int] = {}
    for alloc in allocation:
        fmt = alloc.get("format")
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
        if fmt == "kquant":
            codec = alloc["kquant_codec"]
            counts[codec] = counts.get(codec, 0) + 1
            by_kind = expert_counts if alloc.get("kind") == "expert" else dense_counts
            by_kind[codec] = by_kind.get(codec, 0) + 1
        elif fmt == "tq" and alloc.get("kind") == "expert":
            label = f"TQ{int(alloc.get('bits', 0))}"
            tq_expert_counts[label] = tq_expert_counts.get(label, 0) + 1
    constraints = {
        "objective": "gguf_recipe_kquant_allocation",
        "recipe_source": recipe_source,
        "imatrix": imatrix_identity,
    }
    if expert_allocations is not None:
        constraints["expert_allocation_source"] = "optimizer_decision"
        constraints["expert_allocation_decision_id"] = source_decision_id
    if diagnostic is not None:
        constraints["diagnostic"] = diagnostic
    achieved = {
        "codec_counts": dict(sorted(counts.items())),
        "dense_codec_counts": dict(sorted(dense_counts.items())),
        "expert_codec_counts": dict(sorted(expert_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "expert_format_counts": {
            fmt: sum(1 for a in allocation
                     if a.get("kind") == "expert" and a.get("format") == fmt)
            for fmt in sorted({a.get("format") for a in allocation
                               if a.get("kind") == "expert"})
        } if allocation else {},
    }
    if tq_expert_counts:
        achieved["expert_tq_bit_counts"] = dict(sorted(tq_expert_counts.items()))
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
        source_decision_id=source_decision_id,
        achieved=achieved,
    )
    return plan
