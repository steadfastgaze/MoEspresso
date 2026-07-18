"""Capacity math for the SSD-streaming runtime.

The runtime derives capacity from a memory contract. One capacity unit means
"one resident expert slot per routed layer/projection", and the package's expert
index supplies its exact byte cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moespresso.runtime.expert_index import PROJECTIONS, ExpertIndex
from moespresso.inventory.safetensors_header import read_headers_with_offsets


class StreamingCapacityError(ValueError):
    pass


@dataclass(frozen=True)
class CapacityBudget:
    available_bytes: int
    resident_base_bytes: int
    kv_activation_allowance_bytes: int
    safety_margin_bytes: int
    bytes_per_capacity_unit: int
    min_capacity: int
    max_capacity: int
    runtime_resident_bytes: int = 0

    @property
    def usable_bytes(self) -> int:
        return (
            self.available_bytes
            - self.resident_base_bytes
            - self.runtime_resident_bytes
            - self.kv_activation_allowance_bytes
            - self.safety_margin_bytes
        )


def bytes_per_capacity_unit(index: ExpertIndex) -> int:
    """Bytes for one slot per indexed layer/projection/component."""
    return sum(bytes_per_layer_slot(index).values())


def bytes_per_layer_slot(index: ExpertIndex) -> dict[int, int]:
    """Bytes for one resident expert slot in each indexed routed layer."""
    result = {}
    for layer in index.layers_indexed():
        total = 0
        for projection in PROJECTIONS:
            if not index.has_projection(layer=layer, projection=projection):
                continue
            for component in index.components_for_projection(
                layer=layer,
                projection=projection,
            ):
                total += index.locate(
                    layer=layer,
                    expert=0,
                    projection=projection,
                    component=component,
                ).nbytes
        result[layer] = total
    return result


def is_routed_expert_payload_key(key: str) -> bool:
    # Bundle format: one per-layer uint8 bundle tensor carries the routed
    # expert payload (older stacked keys never reach here: the expert index
    # refuses such packages outright).
    if not key.endswith(".tq_bundle"):
        return False
    return ".switch_mlp." in key or ".ffn.experts." in key


def non_routed_payload_bytes(package_dir: str | Path) -> int:
    """Payload bytes that the SSD runtime keeps resident at startup.

    This is header-only and deliberately excludes routed expert TQ stacks, which
    are represented by the slot pools instead of resident model parameters.
    """
    total = 0
    for shard in sorted(Path(package_dir).glob("model-*.safetensors")):
        for tensor in read_headers_with_offsets(shard):
            if not is_routed_expert_payload_key(tensor.name):
                total += tensor.end - tensor.begin
    return total


def min_capacity(*, max_router_fanout: int, staging_slots: int = 2) -> int:
    """Lower bound: enough slots for top-k plus eviction/loading slack."""
    if max_router_fanout < 1:
        raise ValueError("max_router_fanout must be >= 1")
    if staging_slots < 0:
        raise ValueError("staging_slots must be >= 0")
    return max_router_fanout + staging_slots


def choose_capacity(budget: CapacityBudget) -> int:
    """Return the largest safe capacity, capped at full residency."""
    if budget.bytes_per_capacity_unit <= 0:
        raise ValueError("bytes_per_capacity_unit must be > 0")
    if budget.max_capacity < 1:
        raise ValueError("max_capacity must be >= 1")
    if budget.min_capacity < 1:
        raise ValueError("min_capacity must be >= 1")
    if budget.runtime_resident_bytes < 0:
        raise ValueError("runtime_resident_bytes must be >= 0")
    if budget.min_capacity > budget.max_capacity:
        raise StreamingCapacityError(
            f"min_capacity {budget.min_capacity} exceeds max_capacity "
            f"{budget.max_capacity}")

    capacity = budget.usable_bytes // budget.bytes_per_capacity_unit
    if capacity < budget.min_capacity:
        need = budget.min_capacity * budget.bytes_per_capacity_unit
        raise StreamingCapacityError(
            f"only {budget.usable_bytes} usable byte(s) for SSD expert slots; "
            f"need at least {need} for min_capacity={budget.min_capacity}")
    return int(min(budget.max_capacity, capacity))


def available_memory_bytes() -> int:
    import psutil

    return int(psutil.virtual_memory().available)


def package_capacity_budget(
    *,
    index: ExpertIndex,
    package_dir: str | Path,
    max_router_fanout: int,
    available_bytes: int | None = None,
    kv_activation_allowance_bytes: int | None = None,
    safety_margin_bytes: int | None = None,
    staging_slots: int = 2,
    runtime_resident_bytes: int = 0,
) -> CapacityBudget:
    """Build the package-specific capacity budget from memory + headers.

    `resident_base_bytes` is measured from the package (the model's
    non-routed core). `runtime_resident_bytes` reserves generated resident
    storage that is not present in package headers. The KV/activation allowance
    (default 1 GiB) and safety margin (default 2 GiB) are defaults tuned for a
    16 GB host: hosts with little free RAM tune them via
    MOESPRESSO_SSD_KV_ALLOWANCE_GB / MOESPRESSO_SSD_SAFETY_MARGIN_GB."""
    import os

    if kv_activation_allowance_bytes is None:
        kv_activation_allowance_bytes = int(float(
            os.environ.get("MOESPRESSO_SSD_KV_ALLOWANCE_GB", "1")) * (1 << 30))
    if safety_margin_bytes is None:
        safety_margin_bytes = int(float(
            os.environ.get("MOESPRESSO_SSD_SAFETY_MARGIN_GB", "2")) * (1 << 30))
    if available_bytes is None:
        available_bytes = available_memory_bytes()
    return CapacityBudget(
        available_bytes=int(available_bytes),
        resident_base_bytes=non_routed_payload_bytes(package_dir),
        kv_activation_allowance_bytes=int(kv_activation_allowance_bytes),
        safety_margin_bytes=int(safety_margin_bytes),
        bytes_per_capacity_unit=bytes_per_capacity_unit(index),
        min_capacity=min_capacity(
            max_router_fanout=max_router_fanout,
            staging_slots=staging_slots,
        ),
        max_capacity=index.num_experts,
        runtime_resident_bytes=int(runtime_resident_bytes),
    )


def choose_package_capacity(
    *,
    index: ExpertIndex,
    package_dir: str | Path,
    max_router_fanout: int,
    available_bytes: int | None = None,
    kv_activation_allowance_bytes: int = 1 << 30,
    safety_margin_bytes: int = 2 << 30,
    staging_slots: int = 2,
    runtime_resident_bytes: int = 0,
) -> int:
    """Choose the largest safe per-layer expert-slot capacity for a package."""
    return choose_capacity(package_capacity_budget(
        index=index,
        package_dir=package_dir,
        max_router_fanout=max_router_fanout,
        available_bytes=available_bytes,
        kv_activation_allowance_bytes=kv_activation_allowance_bytes,
        safety_margin_bytes=safety_margin_bytes,
        staging_slots=staging_slots,
        runtime_resident_bytes=runtime_resident_bytes,
    ))
