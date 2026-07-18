from __future__ import annotations

from types import SimpleNamespace

import pytest

from moespresso.runtime.deepseek_v4.speed_stats import (
    _metal_capture_metadata,
    _validate_metal_capture_phase,
    _validate_metal_capture_path,
    attention_layer_phase_deltas,
    indexer_layer_phase_deltas,
    normalize_ssd_structural_stats,
    phase_attention_by_compress_ratio,
    phase_layer_by_compress_ratio,
    phase_indexer_by_compress_ratio,
    prompt_progress_wall_deltas,
    routed_runtime_topology_summary,
    speed_snapshot_payload,
    structural_layer_phase_deltas,
    structural_phase_deltas,
    structural_prompt_prefill_delta,
)


def test_prompt_progress_wall_deltas_report_callback_intervals():
    out = prompt_progress_wall_deltas([
        {"processed_tokens": 0, "total_tokens": 9, "wall_seconds": 0.1},
        {"processed_tokens": 4, "total_tokens": 9, "wall_seconds": 2.5},
        {"processed_tokens": 9, "total_tokens": 9, "wall_seconds": 3.0},
    ])

    assert out == [
        {
            "processed_tokens": 0,
            "total_tokens": 9,
            "wall_seconds": 0.1,
            "delta_seconds": pytest.approx(0.1),
            "token_delta": 0,
        },
        {
            "processed_tokens": 4,
            "total_tokens": 9,
            "wall_seconds": 2.5,
            "delta_seconds": pytest.approx(2.4),
            "token_delta": 4,
        },
        {
            "processed_tokens": 9,
            "total_tokens": 9,
            "wall_seconds": 3.0,
            "delta_seconds": pytest.approx(0.5),
            "token_delta": 5,
        },
    ]


def _stats(**overrides):
    base = {
        "enabled": True,
        "switch_modules": 2,
        "switch_calls": 10,
        "decode_calls": 8,
        "prefill_calls": 2,
        "direct_calls": 7,
        "row_chunked_calls": 2,
        "sorted_chunked_calls": 1,
        "over_capacity_calls": 3,
        "index_sync_calls": 8,
        "index_resync_calls": 10,
        "index_resync_seconds": 99.0,
        "expert_hits": 20,
        "expert_misses": 4,
        "expert_loads": 4,
        "expert_evictions": 1,
        "bundle_row_preads": 3,
        "bundle_cached_takes": 9,
        "chunk_count": 6,
        "projection_load_wait_calls": 4,
        "projection_no_miss_calls": 5,
        "projection_load_parallel_calls": 4,
        "overlap_load_started_calls": 4,
        "overlap_load_wait_calls": 4,
        "overlap_no_miss_calls": 2,
        "overlap_ticket_mismatch_calls": 1,
        "decode_moe_block_calls": 0,
        "decode_moe_block_seconds": 4.0,
        "block_exit_kick_seconds": 1.5,
        "expert_load_seconds": 5.0,
        "index_sync_seconds": 88.0,
        "router_export_seconds": 0.75,
        "router_gate_seconds": 0.5,
        "routed_weighted_sum_calls": 8,
        "routed_weighted_sum_slot_elements": 48,
        "routed_weighted_sum_output_elements": 64,
        "pipeline_read_seconds": 8.0,
        "pipeline_join_seconds": 1.0,
        "routed_build_seconds": 2.0,
        "shared_experts_build_seconds": 1.25,
        "compiled_island_calls": 8,
        "block_exit_kick_calls": 8,
        "pipelined_layers": 8,
        "routed_matmul_calls": 24,
        "routed_matmul_slot_elements": 96,
        "routed_gate_matmul_calls": 8,
        "routed_up_matmul_calls": 8,
        "routed_down_matmul_calls": 8,
        "routed_gate_matmul_slot_elements": 32,
        "routed_up_matmul_slot_elements": 32,
        "routed_down_matmul_slot_elements": 32,
    }
    base.update(overrides)
    return base


def test_normalize_ssd_structural_stats_uses_count_metrics_not_seconds():
    out = normalize_ssd_structural_stats(
        _stats(),
        layer_stats=[{"layer": 0}, {"layer": 1}],
        prompt_tokens=12,
        completion_tokens=4,
    )

    assert out["metric"] == "ds4_ssd_structural_pressure"
    assert out["decode_layer_calls_expected"] == 8
    assert out["observed_decode_layer_calls"] == 8
    assert out["decode_layer_call_coverage"] == pytest.approx(1.0)
    assert out["per_completion_token"]["index_resync_calls"] == pytest.approx(2.5)
    assert out["per_expected_decode_layer"]["index_resync_calls"] == pytest.approx(1.25)
    assert out["per_expected_decode_layer"]["routed_matmul_calls"] == pytest.approx(3.0)
    assert out["per_completion_token"]["projection_no_miss_calls"] == pytest.approx(1.25)
    assert out["per_expected_decode_layer"]["routed_weighted_sum_calls"] == pytest.approx(1.0)
    assert out["per_expected_decode_layer"]["routed_weighted_sum_slot_elements"] == pytest.approx(6.0)
    assert out["per_switch_call"]["bundle_row_preads"] == pytest.approx(0.3)
    assert "index_resync_seconds" not in out["count_totals"]
    assert "index_resync_seconds" not in out["per_completion_token"]
    assert out["seconds_totals"]["index_resync_seconds"] == pytest.approx(99.0)
    assert out["seconds_totals"]["router_gate_seconds"] == pytest.approx(0.5)
    assert out["seconds_totals"]["router_export_seconds"] == pytest.approx(0.75)
    assert out["seconds_totals"]["shared_experts_build_seconds"] == pytest.approx(1.25)
    assert out["seconds_totals"]["block_exit_kick_seconds"] == pytest.approx(1.5)
    assert out["seconds_totals"]["pipeline_read_seconds"] == pytest.approx(8.0)
    assert out["seconds_per_completion_token"]["pipeline_read_seconds"] == pytest.approx(2.0)
    assert out["seconds_per_expected_decode_layer"][
        "pipeline_read_seconds"] == pytest.approx(1.0)


def test_normalize_ssd_structural_stats_fails_without_decode_tokens():
    with pytest.raises(ValueError, match="completion_tokens"):
        normalize_ssd_structural_stats(
            _stats(),
            layer_stats=[],
            prompt_tokens=12,
            completion_tokens=0,
        )


def test_normalize_ssd_structural_stats_fails_without_ssd_streaming():
    with pytest.raises(ValueError, match="SSD-streaming"):
        normalize_ssd_structural_stats(
            _stats(enabled=False),
            layer_stats=[],
            prompt_tokens=12,
            completion_tokens=4,
        )


def test_routed_runtime_topology_summary_flags_full_resident_pooled_path():
    structural = normalize_ssd_structural_stats(
        _stats(),
        layer_stats=[
            {
                "layer": 0,
                "capacity": 4,
                "num_experts": 4,
                "projection_pool_count": 2,
                "resident_slots": 8,
            },
            {
                "layer": 1,
                "capacity": 4,
                "num_experts": 4,
                "projection_pool_count": 2,
                "resident_slots": 8,
            },
        ],
        prompt_tokens=12,
        completion_tokens=4,
    )
    phase = structural_phase_deltas(
        start_stats=_stats(
            routed_weighted_sum_calls=0,
            routed_matmul_calls=0,
        ),
        first_token_stats=_stats(
            routed_weighted_sum_calls=2,
            routed_matmul_calls=4,
        ),
        final_stats=_stats(
            routed_weighted_sum_calls=8,
            routed_matmul_calls=16,
        ),
        routed_layers=2,
        completion_tokens=4,
    )

    out = routed_runtime_topology_summary(
        layer_stats=[
            {
                "layer": 0,
                "capacity": 4,
                "num_experts": 4,
                "projection_pool_count": 2,
                "resident_slots": 8,
            },
            {
                "layer": 1,
                "capacity": 4,
                "num_experts": 4,
                "projection_pool_count": 2,
                "resident_slots": 8,
            },
        ],
        structural=structural,
        phase_structural=phase,
    )

    assert out["metric"] == "ds4_routed_runtime_topology"
    assert out["all_layers_full_capacity"] is True
    assert out["all_layers_full_resident"] is True
    assert out["pooled_runtime_even_when_full_resident"] is True
    assert out["active_decode_topology"] == "combined_gate_up_materialized_down"


def test_routed_runtime_topology_summary_classifies_combined_materialized_path():
    structural = normalize_ssd_structural_stats(
        _stats(
            routed_weighted_sum_calls=8,
            routed_matmul_calls=16,
        ),
        layer_stats=[
            {
                "layer": 0,
                "capacity": 2,
                "num_experts": 4,
                "projection_pool_count": 2,
                "resident_slots": 4,
            },
            {
                "layer": 1,
                "capacity": 2,
                "num_experts": 4,
                "projection_pool_count": 2,
                "resident_slots": 4,
            },
        ],
        prompt_tokens=12,
        completion_tokens=4,
    )

    out = routed_runtime_topology_summary(
        layer_stats=[
            {
                "layer": 0,
                "capacity": 2,
                "num_experts": 4,
                "projection_pool_count": 2,
                "resident_slots": 4,
            },
            {
                "layer": 1,
                "capacity": 2,
                "num_experts": 4,
                "projection_pool_count": 2,
                "resident_slots": 4,
            },
        ],
        structural=structural,
    )

    assert out["all_layers_full_capacity"] is False
    assert out["all_layers_full_resident"] is False
    assert out["active_decode_topology"] == "combined_gate_up_materialized_down"


def test_speed_snapshot_payload_marks_invalid_ceiling_probe():
    result = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        finish_reason="length",
        first_token_seconds=1.0,
        generation_seconds=2.0,
        generated_token_ids=(1, 2, 3, 4),
    )

    payload = speed_snapshot_payload(
        package_dir="/tmp/pkg",
        manifest={
            "artifact_id": "pkg:test",
            "architecture": {
                "family": "deepseek_v4_flash",
                "config": {"compress_ratios": [0, 4]},
            },
        },
        prompt_source="argv:--prompt",
        prompt_mode="rendered-from-user",
        result=result,
        stats=_stats(decode_calls=0, decode_moe_block_calls=8),
        layer_stats=[{"layer": 0}, {"layer": 1}],
        quality_valid=False,
        warnings=["invalid quality arm"],
    )

    assert payload["quality_valid"] is False
    assert payload["warnings"] == ["invalid quality arm"]
    assert payload["structural"]["per_completion_token"]["decode_moe_block_calls"] == 2.0
    assert payload["runtime_topology"]["metric"] == "ds4_routed_runtime_topology"


def test_speed_snapshot_payload_records_decode_after_first_orientation_timing():
    result = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        finish_reason="length",
        first_token_seconds=1.0,
        generation_seconds=2.5,
        generated_token_ids=(1, 2, 3, 4),
    )

    payload = speed_snapshot_payload(
        package_dir="/tmp/pkg",
        manifest={
            "artifact_id": "pkg:test",
            "architecture": {
                "family": "deepseek_v4_flash",
                "config": {"compress_ratios": [0, 4]},
            },
        },
        prompt_source="argv:--prompt",
        prompt_mode="rendered-from-user",
        result=result,
        stats=_stats(),
        layer_stats=[{"layer": 0}, {"layer": 1}],
        quality_valid=True,
    )

    assert payload["generation"]["after_first_seconds"] == pytest.approx(1.5)
    assert payload["generation"]["after_first_tokens"] == 3
    assert payload["generation"]["after_first_tokens_per_second"] == pytest.approx(2.0)


def test_structural_phase_deltas_split_first_token_from_decode_tail():
    out = structural_phase_deltas(
        start_stats=_stats(
            switch_calls=0,
            decode_calls=0,
            bundle_row_preads=0,
            pipeline_read_seconds=0.0,
            index_sync_seconds=0.0,
        ),
        first_token_stats=_stats(
            switch_calls=4,
            decode_calls=2,
            bundle_row_preads=12,
            pipeline_read_seconds=1.0,
            index_sync_seconds=2.0,
        ),
        final_stats=_stats(
            switch_calls=10,
            decode_calls=8,
            bundle_row_preads=30,
            pipeline_read_seconds=7.0,
            index_sync_seconds=5.0,
        ),
        routed_layers=2,
        completion_tokens=4,
    )

    assert out["metric"] == "ds4_ssd_structural_phase_pressure"
    assert out["to_first_response"]["tokens"] == 1
    assert out["to_first_response"]["count_totals"]["switch_calls"] == 4
    assert out["to_first_response"]["per_expected_routed_layer"]["decode_calls"] == 1.0
    assert out["after_first_response"]["tokens"] == 3
    assert out["after_first_response"]["count_totals"]["switch_calls"] == 6
    assert out["after_first_response"]["per_token"]["bundle_row_preads"] == 6.0
    assert out["to_first_response"]["seconds_totals"][
        "pipeline_read_seconds"] == pytest.approx(1.0)
    assert out["after_first_response"]["seconds_totals"][
        "pipeline_read_seconds"] == pytest.approx(6.0)
    assert out["after_first_response"]["seconds_per_token"][
        "pipeline_read_seconds"] == pytest.approx(2.0)
    assert out["after_first_response"]["seconds_per_expected_routed_layer"][
        "index_sync_seconds"] == pytest.approx(0.5)


def test_structural_prompt_prefill_delta_uses_last_pre_final_progress():
    out = structural_prompt_prefill_delta(
        start_stats=_stats(switch_calls=0, prefill_calls=0,
                           routed_matmul_calls=0,
                           pipeline_read_seconds=0.0),
        prompt_progress_stats=[
            {
                "processed_tokens": 0,
                "total_tokens": 5,
                "stats": _stats(switch_calls=0, prefill_calls=0,
                                routed_matmul_calls=0),
            },
            {
                "processed_tokens": 2,
                "total_tokens": 5,
                "stats": _stats(switch_calls=4, prefill_calls=4,
                                routed_matmul_calls=8,
                                pipeline_read_seconds=0.4),
            },
            {
                "processed_tokens": 4,
                "total_tokens": 5,
                "stats": _stats(switch_calls=8, prefill_calls=8,
                                routed_matmul_calls=16,
                                pipeline_read_seconds=0.8),
            },
            {
                # mlx_lm's total callback is after decode prefetch work starts;
                # the speed metric intentionally ignores it.
                "processed_tokens": 5,
                "total_tokens": 5,
                "stats": _stats(switch_calls=99, prefill_calls=99,
                                routed_matmul_calls=99),
            },
        ],
        routed_layers=2,
    )

    assert out["metric"] == "ds4_ssd_prompt_bulk_prefill_pressure"
    assert out["processed_prompt_tokens"] == 4
    assert out["total_prompt_tokens"] == 5
    assert out["omits_final_prompt_token"] is True
    assert out["count_totals"]["switch_calls"] == 8
    assert out["per_expected_routed_layer"]["routed_matmul_calls"] == 2.0
    assert out["seconds_totals"]["pipeline_read_seconds"] == pytest.approx(0.8)
    assert out["seconds_per_expected_routed_layer"][
        "pipeline_read_seconds"] == pytest.approx(0.1)


def test_structural_layer_phase_deltas_split_per_layer_timing():
    out = structural_layer_phase_deltas(
        start_layer_stats=[
            {"layer": 0, "pipelined_layers": 0, "pipeline_read_seconds": 0.0},
            {"layer": 1, "pipelined_layers": 0, "pipeline_read_seconds": 0.0},
        ],
        first_token_layer_stats=[
            {"layer": 0, "pipelined_layers": 1, "pipeline_read_seconds": 0.2},
            {"layer": 1, "pipelined_layers": 1, "pipeline_read_seconds": 0.4},
        ],
        final_layer_stats=[
            {"layer": 0, "pipelined_layers": 3, "pipeline_read_seconds": 0.7},
            {"layer": 1, "pipelined_layers": 3, "pipeline_read_seconds": 1.1},
        ],
    )

    assert out["metric"] == "ds4_ssd_structural_layer_phase_pressure"
    after = out["after_first_response"]
    assert [row["layer"] for row in after] == [0, 1]
    assert after[0]["count_totals"]["pipelined_layers"] == 2
    assert after[1]["seconds_totals"]["pipeline_read_seconds"] == pytest.approx(0.7)


def test_phase_layer_by_compress_ratio_groups_layer_timing():
    grouped = phase_layer_by_compress_ratio(
        {
            "after_first_response": [
                {
                    "layer": 0,
                    "count_totals": {"pipelined_layers": 1},
                    "seconds_totals": {"pipeline_read_seconds": 0.2},
                },
                {
                    "layer": 1,
                    "count_totals": {"pipelined_layers": 1},
                    "seconds_totals": {"pipeline_read_seconds": 0.4},
                },
                {
                    "layer": 2,
                    "count_totals": {"pipelined_layers": 1},
                    "seconds_totals": {"pipeline_read_seconds": 0.8},
                },
            ],
        },
        compress_ratios=[0, 4, 4],
    )

    assert grouped["metric"] == "ds4_ssd_structural_layer_phase_by_compress_ratio"
    phase = grouped["phases"]["after_first_response"]
    assert phase["0"]["layers"] == [0]
    assert phase["0"]["seconds_totals"]["pipeline_read_seconds"] == pytest.approx(0.2)
    assert phase["4"]["layers"] == [1, 2]
    assert phase["4"]["seconds_totals"]["pipeline_read_seconds"] == pytest.approx(1.2)
    assert phase["4"]["seconds_per_layer"]["pipeline_read_seconds"] == pytest.approx(0.6)
    assert phase["4"]["counts_per_layer"]["pipelined_layers"] == pytest.approx(1.0)


def test_indexer_phase_and_ratio_grouping_are_structural():
    phase = indexer_layer_phase_deltas(
        start_layer_stats=[{
            "layer": 2,
            "compress_ratio": 4,
            "indexer_score_contract_calls": 1,
            "indexer_score_contract_tokens": 1,
            "indexer_score_contract_pooled_rows": 128,
            "indexer_score_contract_topk_rows": 128,
            "indexer_score_contract_score_elements": 512,
            "indexer_score_contract_qat_elements": 16512,
            "indexer_score_contract_cached_pooled_rows": 0,
            "indexer_score_contract_new_qat_pooled_rows": 128,
        }],
        first_token_layer_stats=[{
            "layer": 2,
            "compress_ratio": 4,
            "indexer_score_contract_calls": 2,
            "indexer_score_contract_tokens": 2,
            "indexer_score_contract_pooled_rows": 640,
            "indexer_score_contract_topk_rows": 640,
            "indexer_score_contract_score_elements": 2560,
            "indexer_score_contract_qat_elements": 82432,
            "indexer_score_contract_cached_pooled_rows": 128,
            "indexer_score_contract_new_qat_pooled_rows": 512,
        }],
        final_layer_stats=[{
            "layer": 2,
            "compress_ratio": 4,
            "indexer_score_contract_calls": 3,
            "indexer_score_contract_tokens": 3,
            "indexer_score_contract_pooled_rows": 1152,
            "indexer_score_contract_topk_rows": 1152,
            "indexer_score_contract_score_elements": 4608,
            "indexer_score_contract_qat_elements": 90752,
            "indexer_score_contract_cached_pooled_rows": 640,
            "indexer_score_contract_new_qat_pooled_rows": 513,
        }],
    )

    assert phase["metric"] == "ds4_indexer_score_contract_phase_pressure"
    after_first = phase["after_first_response"][0]
    assert after_first["layer"] == 2
    assert after_first["compress_ratio"] == 4
    assert after_first["count_totals"] == {
        "indexer_score_contract_calls": 1,
        "indexer_score_contract_tokens": 1,
        "indexer_score_contract_pooled_rows": 512,
        "indexer_score_contract_topk_rows": 512,
        "indexer_score_contract_score_elements": 2048,
        "indexer_score_contract_qat_elements": 8320,
        "indexer_score_contract_cached_pooled_rows": 512,
        "indexer_score_contract_new_qat_pooled_rows": 1,
        "indexer_score_contract_decode_qat_kernel_calls": 0,
        "indexer_score_contract_fixed_state_calls": 0,
        "indexer_score_contract_score_tail_kernel_calls": 0,
    }

    grouped = phase_indexer_by_compress_ratio(phase)
    bucket = grouped["phases"]["after_first_response"]["4"]
    assert bucket["layers"] == [2]
    assert bucket["count_totals"]["indexer_score_contract_pooled_rows"] == 512
    assert bucket["counts_per_token"]["indexer_score_contract_qat_elements"] == (
        pytest.approx(8320.0)
    )


def test_attention_phase_and_ratio_grouping_are_structural():
    phase = attention_layer_phase_deltas(
        start_layer_stats=[{
            "layer": 2,
            "compress_ratio": 4,
            "attention_sdpa_calls": 1,
            "attention_sdpa_tokens": 1,
            "attention_sdpa_key_rows": 2560,
            "attention_sdpa_score_elements": 163840,
            "attention_sdpa_value_elements": 1310720,
            "attention_sdpa_max_key_rows": 2560,
        }],
        first_token_layer_stats=[{
            "layer": 2,
            "compress_ratio": 4,
            "attention_sdpa_calls": 2,
            "attention_sdpa_tokens": 2,
            "attention_sdpa_key_rows": 5120,
            "attention_sdpa_score_elements": 327680,
            "attention_sdpa_value_elements": 2621440,
            "attention_sdpa_max_key_rows": 2560,
        }],
        final_layer_stats=[{
            "layer": 2,
            "compress_ratio": 4,
            "attention_sdpa_calls": 3,
            "attention_sdpa_tokens": 3,
            "attention_sdpa_key_rows": 7681,
            "attention_sdpa_score_elements": 491584,
            "attention_sdpa_value_elements": 3932672,
            "attention_sdpa_max_key_rows": 2561,
        }],
    )

    assert phase["metric"] == "ds4_attention_sdpa_shape_phase_pressure"
    after_first = phase["after_first_response"][0]
    assert after_first["layer"] == 2
    assert after_first["compress_ratio"] == 4
    assert after_first["cumulative_max_key_rows"] == 2561
    assert after_first["count_totals"] == {
        "attention_sdpa_calls": 1,
        "attention_sdpa_tokens": 1,
        "attention_sdpa_key_rows": 2561,
        "attention_sdpa_score_elements": 163904,
        "attention_sdpa_value_elements": 1311232,
        "fixed_state_decode_layers": 0,
    }

    grouped = phase_attention_by_compress_ratio(phase)
    bucket = grouped["phases"]["after_first_response"]["4"]
    assert bucket["layers"] == [2]
    assert bucket["cumulative_max_key_rows"] == 2561
    assert bucket["counts_per_token"]["attention_sdpa_key_rows"] == (
        pytest.approx(2561.0)
    )


def test_speed_snapshot_payload_includes_phase_structural_deltas():
    result = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        finish_reason="length",
        first_token_seconds=1.0,
        generation_seconds=2.0,
        generated_token_ids=(1, 2, 3, 4),
    )

    payload = speed_snapshot_payload(
        package_dir="/tmp/pkg",
        manifest={
            "artifact_id": "pkg:test",
            "architecture": {
                "family": "deepseek_v4_flash",
                "config": {"compress_ratios": [0, 4]},
            },
        },
        prompt_source="argv:--prompt",
        prompt_mode="rendered-from-user",
        result=result,
        stats=_stats(switch_calls=10, decode_calls=8),
        layer_stats=[{"layer": 0}, {"layer": 1}],
        quality_valid=True,
        phase_stats={
            "start": _stats(switch_calls=0, decode_calls=0),
            "first_token": _stats(switch_calls=4, decode_calls=2),
            "final": _stats(switch_calls=10, decode_calls=8),
        },
    )

    assert payload["phase_structural"]["to_first_response"]["count_totals"]["switch_calls"] == 4
    assert payload["phase_structural"]["after_first_response"]["count_totals"]["decode_calls"] == 6
    assert payload["phase_structural"]["after_first_response"][
        "seconds_totals"]["pipeline_read_seconds"] == pytest.approx(0.0)


def test_speed_snapshot_payload_includes_phase_layer_structural_deltas():
    result = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        finish_reason="length",
        first_token_seconds=1.0,
        generation_seconds=2.0,
        generated_token_ids=(1, 2, 3, 4),
    )

    payload = speed_snapshot_payload(
        package_dir="/tmp/pkg",
        manifest={
            "artifact_id": "pkg:test",
            "architecture": {
                "family": "deepseek_v4_flash",
                "config": {"compress_ratios": [0, 4]},
            },
        },
        prompt_source="argv:--prompt",
        prompt_mode="rendered-from-user",
        result=result,
        stats=_stats(switch_calls=10, decode_calls=8),
        layer_stats=[{"layer": 0}, {"layer": 1}],
        quality_valid=True,
        phase_layer_stats={
            "start": [
                {"layer": 0, "pipelined_layers": 0, "pipeline_read_seconds": 0.0},
            ],
            "first_token": [
                {"layer": 0, "pipelined_layers": 1, "pipeline_read_seconds": 0.2},
            ],
            "final": [
                {"layer": 0, "pipelined_layers": 4, "pipeline_read_seconds": 0.8},
            ],
        },
    )

    row = payload["phase_layer_structural"]["after_first_response"][0]
    assert row["layer"] == 0
    assert row["count_totals"]["pipelined_layers"] == 3
    assert row["seconds_totals"]["pipeline_read_seconds"] == pytest.approx(0.6)
    grouped = payload["phase_layer_by_compress_ratio"]
    assert grouped["phases"]["after_first_response"]["0"]["layers"] == [0]
    assert grouped["phases"]["after_first_response"]["0"][
        "seconds_totals"]["pipeline_read_seconds"] == pytest.approx(0.6)


def test_speed_snapshot_payload_includes_prompt_prefill_structural_deltas():
    result = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        finish_reason="length",
        first_token_seconds=1.0,
        generation_seconds=2.0,
        generated_token_ids=(1, 2, 3, 4),
    )

    payload = speed_snapshot_payload(
        package_dir="/tmp/pkg",
        manifest={
            "artifact_id": "pkg:test",
            "architecture": {"family": "deepseek_v4_flash"},
        },
        prompt_source="argv:--prompt",
        prompt_mode="rendered-from-user",
        result=result,
        stats=_stats(switch_calls=10, prefill_calls=8),
        layer_stats=[{"layer": 0}, {"layer": 1}],
        quality_valid=True,
        phase_stats={
            "start": _stats(switch_calls=0, prefill_calls=0),
            "first_token": _stats(switch_calls=4, prefill_calls=2),
            "final": _stats(switch_calls=10, prefill_calls=8),
        },
        prompt_progress_stats=[{
            "processed_tokens": 11,
            "total_tokens": 12,
            "stats": _stats(switch_calls=7, prefill_calls=7),
        }],
    )

    row = payload["prompt_prefill_structural"]
    assert row["processed_prompt_tokens"] == 11
    assert row["count_totals"]["switch_calls"] == 7
    assert row["per_expected_routed_layer"]["prefill_calls"] == pytest.approx(
        7 / 22)


def test_speed_snapshot_payload_records_metal_capture_metadata(tmp_path):
    result = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        finish_reason="length",
        first_token_seconds=1.0,
        generation_seconds=2.0,
        generated_token_ids=(1, 2, 3, 4),
    )

    capture_path = tmp_path / "tiny.gputrace"
    payload = speed_snapshot_payload(
        package_dir="/tmp/pkg",
        manifest={
            "artifact_id": "pkg:test",
            "architecture": {"family": "deepseek_v4_flash"},
        },
        prompt_source="argv:--prompt",
        prompt_mode="rendered-from-user",
        result=result,
        stats=_stats(),
        layer_stats=[{"layer": 0}, {"layer": 1}],
        quality_valid=True,
        metal_capture=_metal_capture_metadata(capture_path),
    )

    assert payload["metal_capture"] == {
        "enabled": True,
        "path": str(capture_path),
        "phase": "generation",
        "requires_mtl_capture_enabled": True,
        "bounded_by_max_tokens": True,
    }


def test_speed_snapshot_payload_records_diagnostics_metadata():
    result = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        finish_reason="length",
        first_token_seconds=1.0,
        generation_seconds=2.0,
        generated_token_ids=(1, 2, 3, 4),
    )

    payload = speed_snapshot_payload(
        package_dir="/tmp/pkg",
        manifest={
            "artifact_id": "pkg:test",
            "architecture": {"family": "deepseek_v4_flash"},
        },
        prompt_source="argv:--prompt",
        prompt_mode="rendered-from-user",
        result=result,
        stats=_stats(),
        layer_stats=[{"layer": 0}, {"layer": 1}],
        quality_valid=False,
        warnings=["diagnostic arm"],
        diagnostics={
            "sample_probe": {
                "enabled": True,
                "value": 128,
            }
        },
    )

    assert payload["quality_valid"] is False
    assert payload["diagnostics"]["sample_probe"] == {
        "enabled": True,
        "value": 128,
    }


def test_metal_capture_metadata_records_decode_only_phase(tmp_path):
    capture_path = tmp_path / "decode.gputrace"

    assert _metal_capture_metadata(
        capture_path,
        phase="after-first-response",
    ) == {
        "enabled": True,
        "path": str(capture_path),
        "phase": "after-first-response",
        "requires_mtl_capture_enabled": True,
        "bounded_by_max_tokens": True,
    }

    assert _metal_capture_metadata(None) == {
        "enabled": False,
        "path": None,
        "phase": None,
        "requires_mtl_capture_enabled": False,
        "bounded_by_max_tokens": False,
    }


def test_validate_metal_capture_path_requires_process_start_env(tmp_path):
    with pytest.raises(ValueError, match="MTL_CAPTURE_ENABLED=1"):
        _validate_metal_capture_path(tmp_path / "tiny.gputrace", {})


def test_validate_metal_capture_path_requires_gputrace_suffix(tmp_path):
    with pytest.raises(ValueError, match=".gputrace"):
        _validate_metal_capture_path(tmp_path / "tiny.trace", {"MTL_CAPTURE_ENABLED": "1"})


def test_validate_metal_capture_phase_requires_two_tokens_for_decode_only(tmp_path):
    with pytest.raises(ValueError, match="max-tokens >= 2"):
        _validate_metal_capture_phase(
            capture_path=tmp_path / "tiny.gputrace",
            phase="after-first-response",
            max_tokens=1,
        )


def test_validate_metal_capture_phase_allows_decode_only_without_capture():
    _validate_metal_capture_phase(
        capture_path=None,
        phase="after-first-response",
        max_tokens=1,
    )
