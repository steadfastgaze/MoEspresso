"""Consume a TurboQuant optimizer_decision for Qwen routed-expert allocations.

The Qwen GGUF K-quant builder normally reads every allocation (dense and routed
experts) from the GGUF recipe. This module lets the builder instead take the
routed-expert rows from a probe/optimizer `optimizer_decision` artifact while the
dense/backbone tensors keep the recipe's imatrix-calibrated K-quant encode.

The recipe path consumes and cites the optimizer decision without re-emitting
it. A recipe-path package must never fabricate an `optimizer_decision`; only the
probe/optimizer route emits allocation decisions. The
decision's artifact id is recorded on the plan so the manifest provenance shows
the honest mixed source: dense from the recipe, experts from the cited decision.
"""

from __future__ import annotations

from pathlib import Path

from moespresso.core.artifact import ArtifactError, read_artifact
from moespresso.package.qwen.recipe import _expert_entries_by_layer, _PROJECTION_ORDER
from moespresso.inventory import roles
from moespresso.package.kquant_recipe import KQuantRecipeError

OPTIMIZER_DECISION_NAME = "optimizer_decision.json"

# The three logical projections every Qwen MoE layer must cover, and the source
# tensor each maps to (gate/up share the fused gate_up source).
_QWEN_EXPERT_REQUIRED = ("gate", "up", "down")
_QWEN_SOURCE_PROJECTION = {"gate": "gate_up", "up": "gate_up", "down": "down"}


class ExpertAllocationError(KQuantRecipeError):
    """The consumed optimizer decision does not supply valid TQ expert coverage."""


def _resolve_decision_path(ref: str | Path) -> Path:
    """Accept either an optimizer_decision.json file or a package directory."""
    path = Path(ref)
    if path.is_dir():
        candidate = path / OPTIMIZER_DECISION_NAME
        if not candidate.is_file():
            raise ExpertAllocationError(
                f"{path} is a directory but contains no {OPTIMIZER_DECISION_NAME}")
        return candidate
    if not path.is_file():
        raise ExpertAllocationError(
            f"expert allocation source {path} is neither a file nor a directory")
    return path


def load_expert_allocation_decision(ref: str | Path) -> dict:
    """Read and validate an optimizer_decision, fail closed on kind/version/hash.

    `read_artifact` already fails closed on an unknown artifact kind, an
    unsupported schema major, and a content-hash mismatch. This adds the
    stronger check that the artifact is specifically an `optimizer_decision`: any
    other kind (a package_plan, a manifest) is rejected for this flag.
    """
    path = _resolve_decision_path(ref)
    try:
        decision = read_artifact(path)
    except ArtifactError as exc:
        raise ExpertAllocationError(
            f"{path} is not a readable versioned artifact: {exc}") from exc
    kind = decision.get("artifact_kind")
    if kind != "optimizer_decision":
        raise ExpertAllocationError(
            f"{path} has artifact_kind {kind!r}; --expert-allocation-from requires "
            "an optimizer_decision artifact")
    return decision


def _expert_rows_by_key(decision: dict) -> dict[tuple[int, str], dict]:
    """Map (layer, projection) -> the decision's expert allocation row.

    Every routed-expert row must be TQ because this flag means TurboQuant experts.
    Any other codec produces an error.
    """
    out: dict[tuple[int, str], dict] = {}
    non_tq: list[str] = []
    for alloc in decision.get("allocation", []):
        if alloc.get("kind") != "expert":
            continue
        codec = alloc.get("codec") or alloc.get("format")
        if codec != "tq" or alloc.get("format") != "tq":
            non_tq.append(
                f"{alloc.get('source_name')}::{alloc.get('projection')}"
                f"={alloc.get('format')}/{alloc.get('codec')}")
            continue
        layer = alloc.get("layer_index")
        projection = alloc.get("projection")
        if layer is None or projection is None:
            raise ExpertAllocationError(
                "optimizer decision expert row missing layer_index/projection: "
                f"{alloc.get('source_name')}")
        key = (int(layer), str(projection))
        if key in out:
            raise ExpertAllocationError(
                f"optimizer decision has duplicate expert row for {key}")
        out[key] = alloc
    if non_tq:
        raise ExpertAllocationError(
            "--expert-allocation-from requires every routed expert to be TurboQuant "
            "(codec tq); the decision has non-tq expert row(s): "
            + ", ".join(non_tq[:8]))
    return out


def build_tq_expert_allocations_from_decision(
    decision: dict,
    inventory: dict,
) -> list[dict]:
    """Render the decision's TQ expert rows as writer-ready allocation rows.

    Coverage is checked fail-closed against the inventory: every (layer,
    projection) group the inventory's routed experts imply must be present in the
    decision, and the decision must not carry expert rows the inventory does not
    expect. The resulting rows carry exactly the fields the generic writer's TQ
    expert path reads (`source_name`, `kind`, `role`, `layer_index`,
    `projection`, `bits`, `format=tq`, `codec=tq`), plus `module_path`/
    `module_weight_key` for the manifest, all rebuilt from the inventory source
    entry so a decision produced against a different source cannot smuggle in a
    stale tensor name.
    """
    entries = _expert_entries_by_layer(inventory)
    layers = sorted({layer for layer, _projection in entries})
    if not layers:
        raise ExpertAllocationError(
            "inventory contains no Qwen routed expert tensors to cover")

    rows_by_key = _expert_rows_by_key(decision)
    expected_keys: set[tuple[int, str]] = set()
    out: list[dict] = []
    missing: list[str] = []
    for layer in layers:
        source_by_projection = {
            "gate": entries.get((layer, "gate_up")),
            "up": entries.get((layer, "gate_up")),
            "down": entries.get((layer, "down")),
        }
        for projection in _QWEN_EXPERT_REQUIRED:
            source_entry = source_by_projection[projection]
            if source_entry is None:
                missing.append(
                    f"inventory layer {layer} projection "
                    f"{_QWEN_SOURCE_PROJECTION[projection]}")
                continue
            key = (int(layer), projection)
            expected_keys.add(key)
            alloc = rows_by_key.get(key)
            if alloc is None:
                missing.append(f"decision expert row for layer {layer} {projection}")
                continue
            source_name = str(source_entry["source_name"])
            # The decision must reference the same source tensor the inventory
            # resolved; a mismatch means the decision came from a different model.
            if str(alloc.get("source_name")) != source_name:
                raise ExpertAllocationError(
                    f"decision expert row for layer {layer} {projection} names source "
                    f"{alloc.get('source_name')!r}, inventory expects {source_name!r}")
            module = roles.switch_mlp_key(source_name, projection)
            out.append({
                "source_name": source_name,
                "kind": "expert",
                "role": f"moe.expert.{projection}",
                "layer_index": int(layer),
                "projection": projection,
                "source_projection": str(source_entry["projection"]),
                "bits": int(alloc["bits"]),
                "codec": "tq",
                "format": "tq",
                "module_path": module,
                "module_weight_key": f"{module}.weight",
            })

    unexpected = sorted(rows_by_key.keys() - expected_keys)
    if unexpected:
        missing.append(
            "decision has expert rows the inventory does not expect: "
            + ", ".join(f"layer {layer} {proj}" for layer, proj in unexpected[:8]))
    if missing:
        raise ExpertAllocationError(
            "TQ expert allocation coverage mismatch: " + "; ".join(missing[:8]))
    return sorted(out, key=lambda r: (r["layer_index"], _PROJECTION_ORDER[r["projection"]]))
