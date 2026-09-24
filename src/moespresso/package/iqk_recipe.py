"""Shared IQ_K allocation rows and plan helpers.

Model-specific recipe modules resolve source ownership and pass those typed
facts here. This module owns the common member, projection and wire-layout
checks so package builders emit one IQ_K allocation contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from moespresso.package.iqk_format import (
    IQK_DENSE_MEMBERS,
    IQK_GEOMETRY,
    IQK_LAYOUT_IK_WIRE,
    IQK_LAYOUTS,
    validate_iqk_layout,
)
from moespresso.package.plan import make_package_plan


PROJECTIONS = ("gate", "up", "down")
_MANAGED_FIELDS = frozenset(
    {
        "kind",
        "role",
        "layer_index",
        "projection",
        "bits",
        "codec",
        "format",
        "iqk_codec",
        "layout",
    }
)


class IQKRecipeError(ValueError):
    """An IQ_K allocation is incomplete or contradicts the shared format."""


@dataclass(frozen=True)
class IQKDenseTarget:
    """One dense tensor's IQ_K encode target."""

    source_name: str
    role: str
    layer_index: int | None
    codec: str
    layout: str
    gguf_tensor: str
    imatrix_key: str
    module_path: str
    module_weight_key: str


def iqk_dense_target_from_allocation(alloc: Mapping[str, object]) -> IQKDenseTarget:
    """Rebuild a dense IQ_K encode target from one package-plan row."""
    icodec = alloc.get("iqk_codec") or alloc.get("codec")
    layout = alloc.get("layout", IQK_LAYOUT_IK_WIRE)
    missing = [
        key for key in ("gguf_tensor", "imatrix_key", "module_path", "module_weight_key")
        if alloc.get(key) is None
    ]
    if icodec not in IQK_DENSE_MEMBERS:
        missing.append("iqk_codec (dense member)")
    if layout not in IQK_LAYOUTS:
        missing.append("layout")
    if missing:
        raise ValueError(
            f"IQ_K dense allocation for {alloc.get('source_name')} is missing "
            f"or misdeclares required field(s): {', '.join(missing)}"
        )
    return IQKDenseTarget(
        source_name=str(alloc["source_name"]),
        role=str(alloc["role"]),
        layer_index=(
            None if alloc.get("layer_index") is None else int(alloc["layer_index"])
        ),
        codec=str(icodec),
        layout=str(layout),
        gguf_tensor=str(alloc["gguf_tensor"]),
        imatrix_key=str(alloc["imatrix_key"]),
        module_path=str(alloc["module_path"]),
        module_weight_key=str(alloc["module_weight_key"]),
    )


def build_iqk_expert_allocations(
    member_by_layer_projection: Mapping[int, Mapping[str, str]],
    *,
    target_fields: Callable[[int, str], Mapping[str, object]],
    layout: str = IQK_LAYOUT_IK_WIRE,
) -> list[dict]:
    """Render model-resolved routed cells as package-plan allocation rows.

    ``target_fields`` supplies source and module ownership. The fields managed
    by this function cannot be overridden, which keeps member and layout facts
    identical across model families.
    """
    try:
        validate_iqk_layout(layout)
    except ValueError as exc:
        raise IQKRecipeError(str(exc)) from exc
    if not member_by_layer_projection:
        raise IQKRecipeError("no layers for IQ_K expert allocation")

    out: list[dict] = []
    for raw_layer in sorted(member_by_layer_projection):
        layer = int(raw_layer)
        if layer < 0 or layer != raw_layer:
            raise IQKRecipeError(f"invalid IQ_K layer index {raw_layer!r}")
        members = member_by_layer_projection[raw_layer]
        if sorted(members) != sorted(PROJECTIONS):
            raise IQKRecipeError(
                f"layer {layer}: IQ_K allocation must cover gate, up and down, "
                f"got {sorted(members)}"
            )
        for projection in PROJECTIONS:
            member = str(members[projection])
            geometry = IQK_GEOMETRY.get(member)
            if geometry is None:
                raise IQKRecipeError(
                    f"layer {layer} {projection}: unknown IQ_K member {member!r}"
                )
            fields = dict(target_fields(layer, projection))
            collisions = sorted(set(fields) & _MANAGED_FIELDS)
            if collisions:
                raise IQKRecipeError(
                    f"layer {layer} {projection}: target fields override shared "
                    f"IQ_K field(s) {collisions}"
                )
            source_name = fields.get("source_name")
            if not isinstance(source_name, str) or not source_name:
                raise IQKRecipeError(
                    f"layer {layer} {projection}: target has no source_name"
                )
            out.append(
                {
                    "source_name": source_name,
                    "kind": "expert",
                    "role": f"moe.expert.{projection}",
                    "layer_index": layer,
                    "projection": projection,
                    "bits": int(geometry.bits),
                    "codec": member,
                    "format": "iqk",
                    "iqk_codec": member,
                    "layout": layout,
                    **fields,
                }
            )
    return out


def build_iqk_package_plan(
    subject: dict,
    expert_allocation: list[dict],
    *,
    artifacts_identity: dict | None = None,
    allocation_source: str | None = None,
    imatrix_identity: dict | None = None,
    extra_allocation: list[dict] | tuple[dict, ...] | None = None,
    source_decision_id: str | None = None,
    source_probe_id: str | None = None,
    additional_source_constraints: dict | None = None,
    additional_achieved: dict | None = None,
    optimized_kernels_expected: bool = False,
    force_overrides=None,
    allow_unmatched_force: bool = False,
    dry_run: bool = False,
    inputs: list[str] | tuple[str, ...] | None = None,
    additional_required_features: list[str] | tuple[str, ...] | None = None,
    objective: str = "iqk_expert_allocation",
    producer_kind: str = "iqk_converted_artifacts",
) -> dict:
    """Build the shared package plan for converted IQ_K expert cells."""
    allocation = [dict(row) for row in expert_allocation]
    allocation.extend(dict(row) for row in (extra_allocation or ()))
    for row in allocation:
        if row.get("format") == "iqk" or (
            row.get("kind") == "expert" and row.get("codec") == "iqk"
        ):
            validate_iqk_layout(row.get("layout", IQK_LAYOUT_IK_WIRE))

    dense_kquant: dict[str, int] = {}
    format_counts: dict[str, int] = {}
    expert_members: dict[str, int] = {}
    dense_iqk_members: dict[str, int] = {}
    expert_formats: dict[str, int] = {}
    for row in allocation:
        tensor_format = row.get("format") or row.get("codec") or row.get("kind")
        format_counts[tensor_format] = format_counts.get(tensor_format, 0) + 1
        if row.get("kind") == "expert":
            expert_formats[tensor_format] = expert_formats.get(tensor_format, 0) + 1
        if row.get("format") == "kquant" and row.get("kind") == "affine":
            member = row["kquant_codec"]
            dense_kquant[member] = dense_kquant.get(member, 0) + 1
        if row.get("format") == "iqk" and row.get("kind") == "expert":
            member = row["iqk_codec"]
            expert_members[member] = expert_members.get(member, 0) + 1
        if row.get("format") == "iqk" and row.get("kind") == "affine":
            member = row["iqk_codec"]
            dense_iqk_members[member] = dense_iqk_members.get(member, 0) + 1

    constraints = {
        "objective": objective,
        "allocation_source": allocation_source,
        "imatrix": imatrix_identity,
    }
    if artifacts_identity is not None:
        constraints["iqk_artifacts"] = artifacts_identity
    for key, value in (additional_source_constraints or {}).items():
        if key in constraints:
            raise IQKRecipeError(
                f"additional source constraint {key!r} collides with IQ_K plan metadata"
            )
        constraints[key] = value

    achieved = {
        "expert_codec_counts": dict(sorted(expert_members.items())),
        "dense_kquant_codec_counts": dict(sorted(dense_kquant.items())),
        "dense_iqk_member_counts": dict(sorted(dense_iqk_members.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "expert_format_counts": dict(sorted(expert_formats.items())),
    }
    for key, value in (additional_achieved or {}).items():
        if key in achieved:
            raise IQKRecipeError(
                f"additional achieved field {key!r} collides with IQ_K plan metadata"
            )
        achieved[key] = value

    plan, _summary = make_package_plan(
        subject,
        allocation,
        producer_kind=producer_kind,
        producer_reference=(artifacts_identity or {}).get("inventory_sha256"),
        optimized_kernels_expected=optimized_kernels_expected,
        force_overrides=force_overrides,
        allow_unmatched_force=allow_unmatched_force,
        dry_run=dry_run,
        inputs=inputs,
        required_features=["calibration", *(additional_required_features or ())],
        source_decision_id=source_decision_id,
        source_probe_id=source_probe_id,
        status="valid",
        validation=[],
        source_constraints=constraints,
        achieved=achieved,
    )
    return plan
