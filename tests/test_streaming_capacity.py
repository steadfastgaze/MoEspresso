"""Auto-capacity policy."""

from __future__ import annotations

from types import SimpleNamespace


import pytest

from moespresso.runtime.expert_index import build_expert_index
from moespresso.runtime.streaming_capacity import (
    CapacityBudget,
    StreamingCapacityError,
    bytes_per_capacity_unit,
    bytes_per_layer_slot,
    choose_capacity,
    choose_package_capacity,
    full_resident_expert_bytes,
    is_routed_expert_payload_key,
    min_capacity,
    non_routed_payload_bytes,
    package_capacity_budget,
    validate_min_resident_experts,
)


def _package(tmp_path, *, n_layers=2, n_exp=4, out=8, cols=2):
    from conftest import write_bundle_package
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    write_bundle_package(pkg, n_layers=n_layers, n_exp=n_exp, out=out, cols=cols,
                         extra_tensors={
                             "language_model.model.embed_tokens.weight": (
                                 "F16", (10, 8), bytes(10 * 8 * 2)),
                             "language_model.model.layers.0.mlp.shared_expert"
                             ".up_proj.weight": (
                                 "F16", (8, 8), bytes(8 * 8 * 2)),
                         })
    return pkg


def _mixed_package(tmp_path):
    from conftest import write_bundle_package

    pkg = tmp_path / "mixed-pkg"
    pkg.mkdir()
    write_bundle_package(
        pkg,
        layers=(0,),
        n_exp=8,
        shard_name="model-00001-of-00002.safetensors",
    )
    write_bundle_package(
        pkg,
        layers=(1,),
        n_exp=6,
        shard_name="model-00002-of-00002.safetensors",
    )
    return pkg


def test_bytes_per_capacity_unit_comes_from_index_geometry(tmp_path):
    index = build_expert_index(_package(tmp_path, n_layers=2, out=8, cols=2))

    # Per layer/projection: packed row = 8*2*4, norms row = 8*2.
    assert bytes_per_capacity_unit(index) == 2 * 3 * ((8 * 2 * 4) + (8 * 2))


def test_bytes_per_layer_slot_comes_from_index_geometry(tmp_path):
    index = build_expert_index(_package(tmp_path, n_layers=2, out=8, cols=2))

    assert bytes_per_layer_slot(index) == {
        0: 3 * ((8 * 2 * 4) + (8 * 2)),
        1: 3 * ((8 * 2 * 4) + (8 * 2)),
    }


def test_full_resident_bytes_sum_mixed_layer_counts(tmp_path):
    index = build_expert_index(_mixed_package(tmp_path))
    per_layer = bytes_per_layer_slot(index)

    assert full_resident_expert_bytes(index) == (
        (8 * per_layer[0]) + (6 * per_layer[1])
    )


def test_min_capacity_is_router_fanout_plus_staging():
    assert min_capacity(max_router_fanout=4) == 6
    assert min_capacity(max_router_fanout=8, staging_slots=8) == 16


def test_min_resident_experts_is_an_optional_fail_closed_floor():
    model = SimpleNamespace(
        _moespresso_ssd_streaming_capacity=48,
        _moespresso_ssd_streaming_capacity_overrides={3: 52},
    )
    validate_min_resident_experts(
        model,
        requested=None,
    )
    validate_min_resident_experts(
        model,
        requested=48,
    )
    with pytest.raises(StreamingCapacityError, match="below the requested minimum"):
        validate_min_resident_experts(
            model,
            requested=49,
        )
    with pytest.raises(StreamingCapacityError, match="must be >= 1"):
        validate_min_resident_experts(
            model,
            requested=0,
        )
    with pytest.raises(StreamingCapacityError, match="reports resident expert capacity"):
        validate_min_resident_experts(
            SimpleNamespace(),
            requested=1,
        )


def test_min_resident_experts_checks_per_layer_overrides():
    model = SimpleNamespace(
        _moespresso_ssd_streaming_capacity=48,
        _moespresso_ssd_streaming_capacity_overrides={3: 32, 9: 64},
    )

    validate_min_resident_experts(model, requested=32)
    with pytest.raises(StreamingCapacityError, match="capacity 32"):
        validate_min_resident_experts(
            model,
            requested=33,
        )


def test_min_resident_experts_uses_resolved_mixed_layer_capacities():
    model = SimpleNamespace(
        _moespresso_ssd_streaming_capacity=256,
        _moespresso_ssd_streaming_capacity_overrides={},
        _moespresso_ssd_streaming_resolved_capacities={
            0: 256,
            1: 256,
            2: 256,
            3: 64,
        },
    )

    validate_min_resident_experts(model, requested=64)
    with pytest.raises(StreamingCapacityError, match="capacity 64"):
        validate_min_resident_experts(model, requested=200)


def test_min_resident_experts_preserves_uniform_resolved_capacity():
    model = SimpleNamespace(
        _moespresso_ssd_streaming_capacity=256,
        _moespresso_ssd_streaming_capacity_overrides={},
        _moespresso_ssd_streaming_resolved_capacities={0: 256, 1: 256},
    )

    validate_min_resident_experts(model, requested=200)


def test_choose_capacity_uses_remaining_budget_and_caps_at_num_experts():
    budget = CapacityBudget(
        available_bytes=10_000,
        resident_base_bytes=1_000,
        kv_activation_allowance_bytes=1_000,
        safety_margin_bytes=1_000,
        bytes_per_capacity_unit=100,
        min_capacity=6,
        max_capacity=32,
    )

    assert budget.usable_bytes == 7_000
    assert choose_capacity(budget) == 32


def test_choose_capacity_can_return_partial_capacity():
    budget = CapacityBudget(
        available_bytes=2_250,
        resident_base_bytes=250,
        kv_activation_allowance_bytes=250,
        safety_margin_bytes=250,
        bytes_per_capacity_unit=100,
        min_capacity=6,
        max_capacity=32,
    )

    assert choose_capacity(budget) == 15


def test_choose_capacity_subtracts_runtime_resident_bytes():
    budget = CapacityBudget(
        available_bytes=739,
        resident_base_bytes=0,
        kv_activation_allowance_bytes=0,
        safety_margin_bytes=0,
        bytes_per_capacity_unit=100,
        min_capacity=1,
        max_capacity=32,
        runtime_resident_bytes=40,
    )

    assert budget.usable_bytes == 699
    assert choose_capacity(budget) == 6


def test_choose_capacity_fails_before_unsafe_tiny_capacity():
    budget = CapacityBudget(
        available_bytes=1_000,
        resident_base_bytes=250,
        kv_activation_allowance_bytes=250,
        safety_margin_bytes=250,
        bytes_per_capacity_unit=100,
        min_capacity=6,
        max_capacity=32,
    )

    with pytest.raises(StreamingCapacityError, match="need at least 600"):
        choose_capacity(budget)


def test_routed_payload_filter_matches_routed_bundle_keys():
    assert is_routed_expert_payload_key(
        "language_model.model.layers.0.mlp.switch_mlp.experts.tq_bundle")
    assert is_routed_expert_payload_key(
        "layers.0.ffn.experts.tq_bundle")
    assert not is_routed_expert_payload_key(
        "language_model.model.layers.0.mlp.shared_expert.up_proj.weight")
    # legacy stacked keys are not routed payload anymore (such packages are
    # refused by the index before capacity math ever runs)
    assert not is_routed_expert_payload_key(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.tq_packed")


def test_non_routed_payload_bytes_are_header_only_and_exclude_expert_stacks(tmp_path):
    pkg = _package(tmp_path, n_layers=1, n_exp=4, out=8, cols=2)

    assert non_routed_payload_bytes(pkg) == (10 * 8 * 2) + (8 * 8 * 2)


def test_choose_package_capacity_uses_package_geometry_and_memory_budget(tmp_path):
    pkg = _package(tmp_path, n_layers=2, n_exp=8, out=8, cols=2)
    index = build_expert_index(pkg)
    unit = bytes_per_capacity_unit(index)
    non_routed = non_routed_payload_bytes(pkg)

    capacity = choose_package_capacity(
        index=index,
        package_dir=pkg,
        max_router_fanout=4,
        available_bytes=non_routed + (unit * 7) + 100,
        kv_activation_allowance_bytes=50,
        safety_margin_bytes=50,
        staging_slots=2,
    )

    assert capacity == 7


def test_mixed_package_uses_exact_full_fit_and_conservative_partial_fit(tmp_path):
    pkg = _mixed_package(tmp_path)
    index = build_expert_index(pkg)
    exact = full_resident_expert_bytes(index)
    non_routed = non_routed_payload_bytes(pkg)

    assert choose_package_capacity(
        index=index,
        package_dir=pkg,
        max_router_fanout=4,
        available_bytes=non_routed + exact,
        kv_activation_allowance_bytes=0,
        safety_margin_bytes=0,
        staging_slots=2,
    ) == 8
    assert choose_package_capacity(
        index=index,
        package_dir=pkg,
        max_router_fanout=4,
        available_bytes=non_routed + exact - 1,
        kv_activation_allowance_bytes=0,
        safety_margin_bytes=0,
        staging_slots=2,
    ) == 6


def test_package_capacity_budget_exposes_budget_inputs(tmp_path):
    pkg = _package(tmp_path, n_layers=1, n_exp=8, out=8, cols=2)
    index = build_expert_index(pkg)

    budget = package_capacity_budget(
        index=index,
        package_dir=pkg,
        max_router_fanout=4,
        available_bytes=10_000,
        kv_activation_allowance_bytes=100,
        safety_margin_bytes=200,
        staging_slots=2,
    )

    assert budget.available_bytes == 10_000
    assert budget.resident_base_bytes == non_routed_payload_bytes(pkg)
    assert budget.runtime_resident_bytes == 0
    assert budget.bytes_per_capacity_unit == bytes_per_capacity_unit(index)
    assert budget.min_capacity == 6
    assert budget.max_capacity == 8
    assert budget.full_resident_expert_bytes == (
        budget.bytes_per_capacity_unit * budget.max_capacity
    )
