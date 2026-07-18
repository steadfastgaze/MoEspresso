"""Structural speed snapshots for DS4 SSD-streaming runs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence


_COUNT_KEYS = (
    "switch_calls",
    "decode_calls",
    "prefill_calls",
    "direct_calls",
    "row_chunked_calls",
    "sorted_chunked_calls",
    "segmented_prefill_calls",
    "barrier_free_prefill_calls",
    "barrier_free_identity_calls",
    "barrier_free_fused_swiglu_calls",
    "barrier_free_decode_calls",
    "barrier_free_decode_flush_calls",
    "decode_routed_fused_calls",
    "pipelined_decode_fused_calls",
    "hc_fused_pre_calls",
    "hc_fused_post_calls",
    "hc_fused_pre_decode_calls",
    "hc_fused_pre_tail_decode_calls",
    "hc_fused_post_decode_calls",
    "r4_prefill_consumer_mma_calls",
    "r4_prefill_consumer_scalar_calls",
    "r4_prefill_scores_f16_calls",
    "r4_prefill_scores_f32_calls",
    "wo_a_batched_decode_calls",
    "wo_a_gather_decode_calls",
    "wo_a_loop_projection_calls",
    "q8_dense_decode_qmv_calls",
    "q8_dense_decode_wire_qmv_wo_b_calls",
    "q8_dense_decode_wire_qmv_lm_head_calls",
    "q8_dense_prefill_dequant_calls",
    "affine_wo_fp32_wo_a_calls",
    "affine_wo_fp32_wo_b_calls",
    "affine_wo_fp32_delegated_calls",
    "banded_prefill_mma_calls",
    "banded_prefill_sdpa_calls",
    "attn_seam_rope_fused_calls",
    "attn_seam_rope_composed_calls",
    "router_gate_precast_calls",
    "router_gate_select_kernel_calls",
    "router_gate_select_composed_calls",
    "router_gate_composed_calls",
    "over_capacity_calls",
    "index_sync_calls",
    "index_resync_calls",
    "expert_hits",
    "expert_misses",
    "expert_loads",
    "expert_evictions",
    "bundle_row_preads",
    "bundle_cached_takes",
    "chunk_count",
    "projection_load_wait_calls",
    "projection_no_miss_calls",
    "projection_load_parallel_calls",
    "overlap_load_started_calls",
    "overlap_load_wait_calls",
    "overlap_no_miss_calls",
    "overlap_ticket_mismatch_calls",
    "lookahead_exports",
    "lookahead_prefetch_loads",
    "lookahead_ring_misses",
    "lookahead_errors",
    "lookahead_dropped",
    "expert_spec_prefetch_loads",
    "expert_spec_prefetch_skips",
    "decode_moe_block_calls",
    "routed_weighted_sum_calls",
    "routed_weighted_sum_slot_elements",
    "routed_weighted_sum_output_elements",
    "compiled_island_calls",
    "block_exit_kick_calls",
    "pipelined_layers",
    "routed_matmul_calls",
    "routed_matmul_slot_elements",
    "routed_gate_matmul_calls",
    "routed_up_matmul_calls",
    "routed_down_matmul_calls",
    "routed_gate_matmul_slot_elements",
    "routed_up_matmul_slot_elements",
    "routed_down_matmul_slot_elements",
)

_SECONDS_KEYS = (
    "block_exit_kick_seconds",
    "decode_moe_block_seconds",
    "expert_load_seconds",
    "index_sync_seconds",
    "index_resync_seconds",
    "overlap_load_hidden_seconds",
    "overlap_load_total_seconds",
    "overlap_load_wait_seconds",
    "overlap_shared_eval_seconds",
    "pipeline_join_seconds",
    "pipeline_read_seconds",
    "projection_load_wait_seconds",
    "router_export_seconds",
    "router_gate_seconds",
    "routed_build_seconds",
    "shared_experts_build_seconds",
)

_INDEXER_COUNT_KEYS = (
    "indexer_score_contract_calls",
    "indexer_score_contract_tokens",
    "indexer_score_contract_pooled_rows",
    "indexer_score_contract_topk_rows",
    "indexer_score_contract_score_elements",
    "indexer_score_contract_qat_elements",
    "indexer_score_contract_cached_pooled_rows",
    "indexer_score_contract_new_qat_pooled_rows",
    "indexer_score_contract_decode_qat_kernel_calls",
    "indexer_score_contract_fixed_state_calls",
    "indexer_score_contract_score_tail_kernel_calls",
)

_ATTENTION_COUNT_KEYS = (
    "attention_sdpa_calls",
    "attention_sdpa_tokens",
    "attention_sdpa_key_rows",
    "attention_sdpa_score_elements",
    "attention_sdpa_value_elements",
    "fixed_state_decode_layers",
)


def _count(stats: Mapping[str, Any], key: str) -> int:
    value = stats.get(key, 0)
    if value is None:
        return 0
    return int(value)


def _seconds(stats: Mapping[str, Any], key: str) -> float:
    value = stats.get(key, 0.0)
    if value is None:
        return 0.0
    return float(value)


def _rate(value: int, denominator: int) -> float:
    return float(value) / float(denominator)


def prompt_progress_wall_deltas(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int]]:
    """Return per-callback wall deltas for MLX prompt-progress callbacks."""
    out = []
    prev_wall = 0.0
    prev_processed = 0
    for row in rows:
        processed = int(row["processed_tokens"])
        total = int(row["total_tokens"])
        wall = float(row["wall_seconds"])
        out.append({
            "processed_tokens": processed,
            "total_tokens": total,
            "wall_seconds": wall,
            "delta_seconds": wall - prev_wall,
            "token_delta": processed - prev_processed,
        })
        prev_wall = wall
        prev_processed = processed
    return out


def _count_deltas(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
) -> dict[str, int]:
    return {
        key: _count(after, key) - _count(before, key)
        for key in _COUNT_KEYS
    }


def _indexer_count_deltas(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
) -> dict[str, int]:
    return {
        key: _count(after, key) - _count(before, key)
        for key in _INDEXER_COUNT_KEYS
    }


def _attention_count_deltas(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
) -> dict[str, int]:
    return {
        key: _count(after, key) - _count(before, key)
        for key in _ATTENTION_COUNT_KEYS
    }


def _seconds_deltas(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
) -> dict[str, float]:
    return {
        key: _seconds(after, key) - _seconds(before, key)
        for key in _SECONDS_KEYS
    }


def _phase_row(
    delta: Mapping[str, int],
    *,
    tokens: int,
    routed_layers: int,
    seconds_delta: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    per_token = {
        key: _rate(value, tokens)
        for key, value in delta.items()
    } if tokens > 0 else {}
    expected_layers = tokens * routed_layers
    per_expected_layer = {
        key: _rate(value, expected_layers)
        for key, value in delta.items()
    } if expected_layers > 0 else {}
    seconds_totals = dict(seconds_delta or {})
    seconds_per_token = {
        key: float(value) / float(tokens)
        for key, value in seconds_totals.items()
    } if tokens > 0 else {}
    seconds_per_expected_layer = {
        key: float(value) / float(expected_layers)
        for key, value in seconds_totals.items()
    } if expected_layers > 0 else {}
    return {
        "tokens": int(tokens),
        "expected_routed_layer_calls": int(expected_layers),
        "count_totals": dict(delta),
        "per_token": per_token,
        "per_expected_routed_layer": per_expected_layer,
        "seconds_totals": seconds_totals,
        "seconds_per_token": seconds_per_token,
        "seconds_per_expected_routed_layer": seconds_per_expected_layer,
    }


def structural_phase_deltas(
    *,
    start_stats: Mapping[str, Any],
    first_token_stats: Mapping[str, Any],
    final_stats: Mapping[str, Any],
    routed_layers: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Split non-wall-clock structural counters at the first yielded response."""
    if completion_tokens <= 0:
        raise ValueError("completion_tokens must be positive")
    if routed_layers <= 0:
        raise ValueError("routed_layers must be positive")
    to_first = _count_deltas(first_token_stats, start_stats)
    after_first = _count_deltas(final_stats, first_token_stats)
    return {
        "metric": "ds4_ssd_structural_phase_pressure",
        "units": (
            "event count and internal seconds deltas split at the first yielded "
            "response"
        ),
        "to_first_response": _phase_row(
            to_first,
            tokens=1,
            routed_layers=routed_layers,
            seconds_delta=_seconds_deltas(first_token_stats, start_stats),
        ),
        "after_first_response": _phase_row(
            after_first,
            tokens=max(int(completion_tokens) - 1, 0),
            routed_layers=routed_layers,
            seconds_delta=_seconds_deltas(final_stats, first_token_stats),
        ),
    }


def structural_prompt_prefill_delta(
    *,
    start_stats: Mapping[str, Any],
    prompt_progress_stats: Sequence[Mapping[str, Any]],
    routed_layers: int,
) -> dict[str, Any] | None:
    """Return the last MLX bulk-prefill progress delta before final sampling.

    mlx_lm calls prompt_progress_callback(processed, total) during prompt chunk
    prefill, then calls it with processed == total only after it has also started
    decode prefetch work. The useful structural prefill boundary is therefore
    the last progress point where 0 < processed < total.
    """
    if routed_layers <= 0:
        raise ValueError("routed_layers must be positive")
    candidates = [
        row
        for row in prompt_progress_stats
        if 0 < int(row.get("processed_tokens", 0)) < int(row.get("total_tokens", 0))
    ]
    if not candidates:
        return None
    row = max(candidates, key=lambda item: int(item["processed_tokens"]))
    processed_tokens = int(row["processed_tokens"])
    total_tokens = int(row["total_tokens"])
    delta = _count_deltas(row["stats"], start_stats)
    seconds_delta = _seconds_deltas(row["stats"], start_stats)
    out = _phase_row(delta, tokens=processed_tokens, routed_layers=routed_layers)
    out["seconds_totals"] = seconds_delta
    if processed_tokens > 0:
        out["seconds_per_token"] = {
            key: value / float(processed_tokens)
            for key, value in seconds_delta.items()
        }
    expected_layers = processed_tokens * routed_layers
    if expected_layers > 0:
        out["seconds_per_expected_routed_layer"] = {
            key: value / float(expected_layers)
            for key, value in seconds_delta.items()
        }
    out.update({
        "metric": "ds4_ssd_prompt_bulk_prefill_pressure",
        "units": (
            "event count and internal seconds deltas through the last mlx_lm "
            "prompt-progress callback before the final prompt-token sampling step"
        ),
        "processed_prompt_tokens": processed_tokens,
        "total_prompt_tokens": total_tokens,
        "omits_final_prompt_token": processed_tokens == total_tokens - 1,
    })
    return out


def _layer_rows_by_id(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if "layer" not in row:
            continue
        out[int(row["layer"])] = row
    return out


def layer_phase_deltas(
    *,
    before_layer_stats: Sequence[Mapping[str, Any]],
    after_layer_stats: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-layer count/seconds deltas between two stats snapshots."""
    before = _layer_rows_by_id(before_layer_stats)
    after = _layer_rows_by_id(after_layer_stats)
    rows = []
    for layer in sorted(set(before) | set(after)):
        rows.append({
            "layer": int(layer),
            "count_totals": _count_deltas(after.get(layer, {}), before.get(layer, {})),
            "seconds_totals": _seconds_deltas(
                after.get(layer, {}),
                before.get(layer, {}),
            ),
        })
    return rows


def structural_layer_phase_deltas(
    *,
    start_layer_stats: Sequence[Mapping[str, Any]],
    first_token_layer_stats: Sequence[Mapping[str, Any]],
    final_layer_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Split per-layer deltas at the first yielded response."""
    return {
        "metric": "ds4_ssd_structural_layer_phase_pressure",
        "units": (
            "per-layer event count and internal seconds deltas split at the "
            "first yielded response"
        ),
        "to_first_response": layer_phase_deltas(
            before_layer_stats=start_layer_stats,
            after_layer_stats=first_token_layer_stats,
        ),
        "after_first_response": layer_phase_deltas(
            before_layer_stats=first_token_layer_stats,
            after_layer_stats=final_layer_stats,
        ),
    }


def indexer_layer_phase_deltas(
    *,
    start_layer_stats: Sequence[Mapping[str, Any]],
    first_token_layer_stats: Sequence[Mapping[str, Any]],
    final_layer_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Split DS4 indexer score-contract counters at the first response."""
    start = _layer_rows_by_id(start_layer_stats)
    first = _layer_rows_by_id(first_token_layer_stats)
    final = _layer_rows_by_id(final_layer_stats)

    def rows_between(
        before: Mapping[int, Mapping[str, Any]],
        after: Mapping[int, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = []
        for layer in sorted(set(before) | set(after)):
            after_row = after.get(layer, {})
            before_row = before.get(layer, {})
            rows.append({
                "layer": int(layer),
                "compress_ratio": int(
                    after_row.get(
                        "compress_ratio",
                        before_row.get("compress_ratio", 0),
                    )
                    or 0
                ),
                "count_totals": _indexer_count_deltas(after_row, before_row),
            })
        return rows

    return {
        "metric": "ds4_indexer_score_contract_phase_pressure",
        "units": (
            "shape-derived indexer score-contract deltas split at the first "
            "yielded response; no wall-clock timing"
        ),
        "to_first_response": rows_between(start, first),
        "after_first_response": rows_between(first, final),
    }


def attention_layer_phase_deltas(
    *,
    start_layer_stats: Sequence[Mapping[str, Any]],
    first_token_layer_stats: Sequence[Mapping[str, Any]],
    final_layer_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Split DS4 SDPA shape counters at the first response."""
    start = _layer_rows_by_id(start_layer_stats)
    first = _layer_rows_by_id(first_token_layer_stats)
    final = _layer_rows_by_id(final_layer_stats)

    def rows_between(
        before: Mapping[int, Mapping[str, Any]],
        after: Mapping[int, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = []
        for layer in sorted(set(before) | set(after)):
            after_row = after.get(layer, {})
            before_row = before.get(layer, {})
            delta = _attention_count_deltas(after_row, before_row)
            rows.append({
                "layer": int(layer),
                "compress_ratio": int(
                    after_row.get(
                        "compress_ratio",
                        before_row.get("compress_ratio", 0),
                    )
                    or 0
                ),
                "count_totals": delta,
                "cumulative_max_key_rows": int(
                    max(
                        int(after_row.get("attention_sdpa_max_key_rows", 0) or 0),
                        int(before_row.get("attention_sdpa_max_key_rows", 0) or 0),
                    )
                ),
            })
        return rows

    return {
        "metric": "ds4_attention_sdpa_shape_phase_pressure",
        "units": (
            "shape-derived SDPA q/k burden split at the first yielded "
            "response; no wall-clock timing"
        ),
        "to_first_response": rows_between(start, first),
        "after_first_response": rows_between(first, final),
    }


def _add_dict_values(
    left: dict[str, Any],
    right: Mapping[str, Any],
) -> None:
    for key, value in right.items():
        left[key] = left.get(key, 0) + value


def _compress_ratios_from_manifest(manifest: Mapping[str, Any]) -> list[int] | None:
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        arch = manifest.get("architecture", {})
        if isinstance(arch, Mapping):
            config = arch.get("config")
    if not isinstance(config, Mapping):
        return None
    ratios = config.get("compress_ratios")
    if not isinstance(ratios, Sequence) or isinstance(ratios, (str, bytes)):
        return None
    return [int(ratio) for ratio in ratios]


def phase_layer_by_compress_ratio(
    phase_layer_structural: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    *,
    compress_ratios: Sequence[int] | None,
) -> dict[str, Any] | None:
    """Aggregate per-layer phase deltas by DS4 attention compression ratio."""
    if phase_layer_structural is None or compress_ratios is None:
        return None
    phases = {}
    for phase, rows in phase_layer_structural.items():
        if phase not in {"to_first_response", "after_first_response"}:
            continue
        buckets: dict[str, dict[str, Any]] = {}
        for row in rows:
            layer = int(row["layer"])
            if layer < 0 or layer >= len(compress_ratios):
                continue
            ratio = str(int(compress_ratios[layer]))
            bucket = buckets.setdefault(ratio, {
                "layers": [],
                "layer_count": 0,
                "count_totals": {},
                "seconds_totals": {},
            })
            bucket["layers"].append(layer)
            bucket["layer_count"] += 1
            _add_dict_values(bucket["count_totals"], row.get("count_totals", {}))
            _add_dict_values(bucket["seconds_totals"], row.get("seconds_totals", {}))
        for bucket in buckets.values():
            layer_count = int(bucket["layer_count"])
            if layer_count > 0:
                bucket["seconds_per_layer"] = {
                    key: float(value) / float(layer_count)
                    for key, value in bucket["seconds_totals"].items()
                }
                bucket["counts_per_layer"] = {
                    key: float(value) / float(layer_count)
                    for key, value in bucket["count_totals"].items()
                }
        phases[str(phase)] = buckets
    return {
        "metric": "ds4_ssd_structural_layer_phase_by_compress_ratio",
        "units": (
            "per-layer event count and internal seconds deltas grouped by "
            "manifest config.compress_ratios"
        ),
        "phases": phases,
    }


def phase_indexer_by_compress_ratio(
    phase_indexer_layer_structural: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Any] | None:
    """Aggregate indexer score-contract counters by layer compression ratio."""
    if phase_indexer_layer_structural is None:
        return None
    phases = {}
    for phase, rows in phase_indexer_layer_structural.items():
        if phase not in {"to_first_response", "after_first_response"}:
            continue
        buckets: dict[str, dict[str, Any]] = {}
        for row in rows:
            ratio = str(int(row.get("compress_ratio", 0) or 0))
            bucket = buckets.setdefault(ratio, {
                "layers": [],
                "layer_count": 0,
                "count_totals": {},
            })
            layer = int(row["layer"])
            if layer not in bucket["layers"]:
                bucket["layers"].append(layer)
                bucket["layer_count"] += 1
            _add_dict_values(bucket["count_totals"], row.get("count_totals", {}))
        for bucket in buckets.values():
            token_count = int(
                bucket["count_totals"].get("indexer_score_contract_tokens", 0)
            )
            if token_count > 0:
                bucket["counts_per_token"] = {
                    key: float(value) / float(token_count)
                    for key, value in bucket["count_totals"].items()
                }
        phases[str(phase)] = buckets
    return {
        "metric": "ds4_indexer_score_contract_by_compress_ratio",
        "units": (
            "shape-derived indexer score-contract deltas grouped by "
            "attention compression ratio"
        ),
        "phases": phases,
    }


def phase_attention_by_compress_ratio(
    phase_attention_layer_structural: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Any] | None:
    """Aggregate SDPA shape counters by layer compression ratio."""
    if phase_attention_layer_structural is None:
        return None
    phases = {}
    for phase, rows in phase_attention_layer_structural.items():
        if phase not in {"to_first_response", "after_first_response"}:
            continue
        buckets: dict[str, dict[str, Any]] = {}
        for row in rows:
            ratio = str(int(row.get("compress_ratio", 0) or 0))
            bucket = buckets.setdefault(ratio, {
                "layers": [],
                "layer_count": 0,
                "count_totals": {},
                "cumulative_max_key_rows": 0,
            })
            layer = int(row["layer"])
            if layer not in bucket["layers"]:
                bucket["layers"].append(layer)
                bucket["layer_count"] += 1
            _add_dict_values(bucket["count_totals"], row.get("count_totals", {}))
            bucket["cumulative_max_key_rows"] = max(
                int(bucket["cumulative_max_key_rows"]),
                int(row.get("cumulative_max_key_rows", 0) or 0),
            )
        for bucket in buckets.values():
            token_count = int(bucket["count_totals"].get("attention_sdpa_tokens", 0))
            if token_count > 0:
                bucket["counts_per_token"] = {
                    key: float(value) / float(token_count)
                    for key, value in bucket["count_totals"].items()
                }
        phases[str(phase)] = buckets
    return {
        "metric": "ds4_attention_sdpa_shape_by_compress_ratio",
        "units": (
            "shape-derived SDPA q/k burden grouped by attention compression ratio"
        ),
        "phases": phases,
    }


def normalize_ssd_structural_stats(
    stats: Mapping[str, Any],
    layer_stats: Sequence[Mapping[str, Any]],
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Return the non-wall-time speed metric for an SSD-streaming generation."""
    if not stats.get("enabled", False):
        raise ValueError("SSD-streaming stats are not enabled for this model")
    if prompt_tokens < 0:
        raise ValueError("prompt_tokens must be non-negative")
    if completion_tokens <= 0:
        raise ValueError("completion_tokens must be positive")

    routed_layers = _count(stats, "switch_modules")
    if routed_layers <= 0:
        routed_layers = len(layer_stats)
    if routed_layers <= 0:
        raise ValueError("stats do not contain routed layer counts")

    decode_layer_calls_expected = completion_tokens * routed_layers
    count_totals = {key: _count(stats, key) for key in _COUNT_KEYS}
    seconds_totals = {key: _seconds(stats, key) for key in _SECONDS_KEYS}
    switch_calls = count_totals["switch_calls"]
    observed_decode_layer_calls = (
        count_totals["decode_calls"] + count_totals["decode_moe_block_calls"]
    )

    per_completion_token = {
        key: _rate(value, completion_tokens)
        for key, value in count_totals.items()
    }
    per_switch_call = {
        key: _rate(value, switch_calls)
        for key, value in count_totals.items()
    } if switch_calls > 0 else {}
    per_expected_decode_layer = {
        key: _rate(value, decode_layer_calls_expected)
        for key, value in count_totals.items()
    }
    seconds_per_completion_token = {
        key: float(value) / float(completion_tokens)
        for key, value in seconds_totals.items()
    }
    seconds_per_expected_decode_layer = {
        key: float(value) / float(decode_layer_calls_expected)
        for key, value in seconds_totals.items()
    }

    return {
        "metric": "ds4_ssd_structural_pressure",
        "units": "event counts normalized by generated tokens and routed layers",
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "routed_layers": int(routed_layers),
        "decode_layer_calls_expected": int(decode_layer_calls_expected),
        "observed_decode_layer_calls": int(observed_decode_layer_calls),
        "decode_layer_call_coverage": _rate(
            observed_decode_layer_calls,
            decode_layer_calls_expected,
        ),
        "count_totals": count_totals,
        "seconds_totals": seconds_totals,
        "per_completion_token": per_completion_token,
        "seconds_per_completion_token": seconds_per_completion_token,
        "per_switch_call": per_switch_call,
        "per_expected_decode_layer": per_expected_decode_layer,
        "seconds_per_expected_decode_layer": seconds_per_expected_decode_layer,
    }


def routed_runtime_topology_summary(
    *,
    layer_stats: Sequence[Mapping[str, Any]],
    structural: Mapping[str, Any],
    phase_structural: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize the served routed runtime shape without using wall time."""
    layer_count = len(layer_stats)
    full_capacity_layers = 0
    full_resident_layers = 0
    capacities = []
    resident_slots = []
    for row in layer_stats:
        capacity = int(row.get("capacity", 0) or 0)
        num_experts = int(row.get("num_experts", 0) or 0)
        pool_count = int(row.get("projection_pool_count", 0) or 0)
        resident = int(row.get("resident_slots", 0) or 0)
        capacities.append(capacity)
        resident_slots.append(resident)
        if num_experts > 0 and capacity >= num_experts:
            full_capacity_layers += 1
        if num_experts > 0 and pool_count > 0 and resident >= num_experts * pool_count:
            full_resident_layers += 1

    phase = (
        phase_structural.get("after_first_response", {})
        if phase_structural is not None
        else {}
    )
    per_layer = phase.get("per_expected_routed_layer")
    if not per_layer:
        per_layer = structural.get("per_expected_decode_layer", {})
    materialized = float(per_layer.get("routed_weighted_sum_calls", 0.0))
    matmul_calls = float(per_layer.get("routed_matmul_calls", 0.0))

    if matmul_calls <= 2.01 and materialized > 0.01:
        active_decode_topology = "combined_gate_up_materialized_down"
    elif matmul_calls >= 2.0 or materialized > 0.01:
        active_decode_topology = "materialized_or_mixed"
    else:
        active_decode_topology = "unknown_or_inactive"

    return {
        "metric": "ds4_routed_runtime_topology",
        "units": "non-wall-clock topology and residency facts for routed layers",
        "pooled_switch_layers": int(layer_count),
        "full_capacity_layers": int(full_capacity_layers),
        "full_resident_layers": int(full_resident_layers),
        "all_layers_full_capacity": bool(
            layer_count > 0 and full_capacity_layers == layer_count),
        "all_layers_full_resident": bool(
            layer_count > 0 and full_resident_layers == layer_count),
        "pooled_runtime_even_when_full_resident": bool(
            layer_count > 0 and full_resident_layers == layer_count),
        "capacity_range": [
            min(capacities) if capacities else 0,
            max(capacities) if capacities else 0,
        ],
        "resident_slots_range": [
            min(resident_slots) if resident_slots else 0,
            max(resident_slots) if resident_slots else 0,
        ],
        "active_decode_topology": active_decode_topology,
        "per_expected_layer": {
            "routed_weighted_sum_calls": materialized,
            "routed_matmul_calls": matmul_calls,
        },
    }


def _validate_metal_capture_path(
    capture_path: Path | None,
    environ: Mapping[str, str],
) -> Path | None:
    if capture_path is None:
        return None
    if capture_path.suffix != ".gputrace":
        raise ValueError("--metal-capture-path must end with .gputrace")
    if environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise ValueError(
            "--metal-capture-path requires MTL_CAPTURE_ENABLED=1 before process start"
        )
    return capture_path


def _validate_metal_capture_phase(*, capture_path: Path | None, phase: str,
                                  max_tokens: int) -> None:
    if phase not in {"generation", "after-first-response"}:
        raise ValueError(
            "--metal-capture-phase must be 'generation' or 'after-first-response'")
    if capture_path is None:
        return
    if phase == "after-first-response" and max_tokens < 2:
        raise ValueError(
            "--metal-capture-phase=after-first-response requires --max-tokens >= 2")


def _metal_capture_metadata(capture_path: Path | None, *,
                            phase: str = "generation") -> dict[str, Any]:
    return {
        "enabled": capture_path is not None,
        "path": str(capture_path) if capture_path is not None else None,
        "phase": str(phase) if capture_path is not None else None,
        "requires_mtl_capture_enabled": capture_path is not None,
        "bounded_by_max_tokens": capture_path is not None,
    }


@contextmanager
def _metal_capture(capture_path: Path | None) -> Iterator[None]:
    if capture_path is None:
        yield
        return

    import mlx.core as mx

    capture_path.parent.mkdir(parents=True, exist_ok=True)
    mx.metal.start_capture(str(capture_path))
    try:
        yield
    finally:
        mx.metal.stop_capture()


class _DeferredMetalCapture:
    """Start a Metal capture later than the outer generate call.

    The DS4 speed loop often needs decode-only traces after prewarm. Starting
    capture from the first response callback avoids capturing prefill/TTFT while
    still using MLX's regular stream_generate path.
    """

    def __init__(self, capture_path: Path | None):
        self.capture_path = capture_path
        self.started = False

    def start(self) -> None:
        if self.capture_path is None or self.started:
            return
        import mlx.core as mx

        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        mx.metal.start_capture(str(self.capture_path))
        self.started = True

    def stop(self) -> None:
        if not self.started:
            return
        import mlx.core as mx

        mx.metal.stop_capture()
        self.started = False


def speed_snapshot_payload(
    *,
    package_dir: Path,
    manifest: Mapping[str, Any],
    prompt_source: str,
    prompt_mode: str,
    result: Any,
    stats: Mapping[str, Any],
    layer_stats: Sequence[Mapping[str, Any]],
    quality_valid: bool,
    warnings: Sequence[str] = (),
    metal_capture: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    phase_stats: Mapping[str, Mapping[str, Any]] | None = None,
    phase_layer_stats: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    phase_indexer_layer_stats: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    phase_attention_layer_stats: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    prompt_progress_stats: Sequence[Mapping[str, Any]] = (),
    prompt_progress_wall: Sequence[Mapping[str, Any]] = (),
    indexer_layer_stats: Sequence[Mapping[str, Any]] = (),
    attention_layer_stats: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a JSON-serializable speed snapshot from a completed generation."""
    normalized = normalize_ssd_structural_stats(
        stats,
        layer_stats,
        prompt_tokens=int(result.prompt_tokens),
        completion_tokens=int(result.completion_tokens),
    )
    phase_structural = None
    if phase_stats is not None:
        phase_structural = structural_phase_deltas(
            start_stats=phase_stats["start"],
            first_token_stats=phase_stats["first_token"],
            final_stats=phase_stats["final"],
            routed_layers=int(normalized["routed_layers"]),
            completion_tokens=int(result.completion_tokens),
        )
        prompt_prefill_structural = structural_prompt_prefill_delta(
            start_stats=phase_stats["start"],
            prompt_progress_stats=prompt_progress_stats,
            routed_layers=int(normalized["routed_layers"]),
        )
    else:
        prompt_prefill_structural = None
    phase_layer_structural = None
    if phase_layer_stats is not None:
        phase_layer_structural = structural_layer_phase_deltas(
            start_layer_stats=phase_layer_stats["start"],
            first_token_layer_stats=phase_layer_stats["first_token"],
            final_layer_stats=phase_layer_stats["final"],
        )
    phase_layer_by_ratio = phase_layer_by_compress_ratio(
        phase_layer_structural,
        compress_ratios=_compress_ratios_from_manifest(manifest),
    )
    phase_indexer_layer_structural = None
    if phase_indexer_layer_stats is not None:
        phase_indexer_layer_structural = indexer_layer_phase_deltas(
            start_layer_stats=phase_indexer_layer_stats["start"],
            first_token_layer_stats=phase_indexer_layer_stats["first_token"],
            final_layer_stats=phase_indexer_layer_stats["final"],
        )
    phase_indexer_by_ratio = phase_indexer_by_compress_ratio(
        phase_indexer_layer_structural
    )
    phase_attention_layer_structural = None
    if phase_attention_layer_stats is not None:
        phase_attention_layer_structural = attention_layer_phase_deltas(
            start_layer_stats=phase_attention_layer_stats["start"],
            first_token_layer_stats=phase_attention_layer_stats["first_token"],
            final_layer_stats=phase_attention_layer_stats["final"],
        )
    phase_attention_by_ratio = phase_attention_by_compress_ratio(
        phase_attention_layer_structural
    )
    after_first_tokens = max(int(result.completion_tokens) - 1, 0)
    after_first_seconds = None
    after_first_tokens_per_second = None
    if (
        result.generation_seconds is not None
        and result.first_token_seconds is not None
    ):
        after_first_seconds = (
            float(result.generation_seconds) - float(result.first_token_seconds)
        )
        if after_first_tokens > 0 and after_first_seconds > 0:
            after_first_tokens_per_second = (
                float(after_first_tokens) / float(after_first_seconds)
            )
    runtime_topology = routed_runtime_topology_summary(
        layer_stats=layer_stats,
        structural=normalized,
        phase_structural=phase_structural,
    )
    return {
        "artifact_kind": "ds4_speed_structural_snapshot",
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "package_family": manifest.get("architecture", {}).get("family"),
        "prompt_source": prompt_source,
        "prompt_mode": prompt_mode,
        "quality_valid": bool(quality_valid),
        "warnings": list(warnings),
        "metal_capture": dict(metal_capture or _metal_capture_metadata(None)),
        "diagnostics": dict(diagnostics or {}),
        "generation": {
            "prompt_tokens": int(result.prompt_tokens),
            "completion_tokens": int(result.completion_tokens),
            "finish_reason": result.finish_reason,
            "first_token_seconds": result.first_token_seconds,
            "generation_seconds": result.generation_seconds,
            "after_first_seconds": after_first_seconds,
            "after_first_tokens": after_first_tokens,
            "after_first_tokens_per_second": after_first_tokens_per_second,
            "generated_token_ids": list(result.generated_token_ids),
        },
        "structural": normalized,
        "runtime_topology": runtime_topology,
        "phase_structural": phase_structural,
        "phase_layer_structural": phase_layer_structural,
        "phase_layer_by_compress_ratio": phase_layer_by_ratio,
        "phase_indexer_layer_structural": phase_indexer_layer_structural,
        "phase_indexer_by_compress_ratio": phase_indexer_by_ratio,
        "phase_attention_layer_structural": phase_attention_layer_structural,
        "phase_attention_by_compress_ratio": phase_attention_by_ratio,
        "prompt_prefill_structural": prompt_prefill_structural,
        "prompt_progress_wall": prompt_progress_wall_deltas(prompt_progress_wall),
        "raw_ssd_streaming_stats": dict(stats),
        "layer_stats": [dict(row) for row in layer_stats],
        "indexer_layer_stats": [dict(row) for row in indexer_layer_stats],
        "attention_layer_stats": [dict(row) for row in attention_layer_stats],
    }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _prompt_from_args(args, tokenizer, manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    if args.rendered_prompt_file is not None:
        path = Path(args.rendered_prompt_file)
        return _read_text(path), str(path), "rendered"

    if args.prompt_file is not None:
        user_prompt = _read_text(Path(args.prompt_file))
        prompt_source = str(args.prompt_file)
    else:
        user_prompt = args.prompt
        prompt_source = "argv:--prompt"

    from moespresso.runtime.http import (
        deepseek_v4_contract_template_kwargs,
        is_deepseek_v4_manifest,
        render_prompt,
    )

    template_kwargs = None
    if is_deepseek_v4_manifest(manifest):
        # The speed snapshot measures the served default mode (thinking off).
        template_kwargs = deepseek_v4_contract_template_kwargs()
    rendered = render_prompt(
        [{"role": "user", "content": user_prompt}],
        tokenizer,
        template_kwargs=template_kwargs,
        prompt_renderer=manifest.get("architecture", {}).get("prompt_renderer"),
    )
    return rendered, prompt_source, "rendered-from-user"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-speed-stats",
        description="Run a small DS4 generation and dump SSD structural counters.",
    )
    parser.add_argument("package_dir", type=Path)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default="Hello")
    prompt_group.add_argument("--prompt-file", type=Path)
    prompt_group.add_argument("--rendered-prompt-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-memory-gb", type=float, default=None)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--metal-capture-path",
        type=Path,
        help=(
            "Optional .gputrace path. Requires MTL_CAPTURE_ENABLED=1 before "
            "process start and captures only the bounded generation call."
        ),
    )
    parser.add_argument(
        "--metal-capture-phase",
        choices=("generation", "after-first-response"),
        default="generation",
        help=(
            "Metal capture window when --metal-capture-path is set. "
            "'generation' captures the full bounded generate call; "
            "'after-first-response' starts after the first yielded token so "
            "small runs can inspect decode without prefill/TTFT."
        ),
    )
    args = parser.parse_args(argv)

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    try:
        metal_capture_path = _validate_metal_capture_path(
            args.metal_capture_path,
            os.environ,
        )
        _validate_metal_capture_phase(
            capture_path=metal_capture_path,
            phase=args.metal_capture_phase,
            max_tokens=args.max_tokens,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_memory_gb is not None:
        os.environ["MOESPRESSO_SSD_MAX_MEMORY_GB"] = str(args.max_memory_gb)

    from moespresso.runtime.serve import generate_with_metadata, load_served_model
    from moespresso.runtime.deepseek_v4.model import (
        deepseek_v4_attention_layer_stats,
        deepseek_v4_indexer_layer_stats,
    )
    from moespresso.runtime.ssd_streaming_build import (
        ssd_streaming_layer_stats,
        ssd_streaming_stats,
    )

    model, tokenizer, manifest = load_served_model(args.package_dir)
    prompt, prompt_source, prompt_mode = _prompt_from_args(args, tokenizer, manifest)
    start_stats = ssd_streaming_stats(model)
    start_layer_stats = ssd_streaming_layer_stats(model)
    start_indexer_layer_stats = deepseek_v4_indexer_layer_stats(model)
    start_attention_layer_stats = deepseek_v4_attention_layer_stats(model)
    first_token_stats: dict[str, Any] | None = None
    first_token_layer_stats: list[dict[str, Any]] | None = None
    first_token_indexer_layer_stats: list[dict[str, Any]] | None = None
    first_token_attention_layer_stats: list[dict[str, Any]] | None = None
    prompt_progress_stats: list[dict[str, Any]] = []
    prompt_progress_wall: list[dict[str, Any]] = []
    progress_t0 = time.perf_counter()

    def _capture_first_token_stats(step: int, _response: object) -> None:
        nonlocal first_token_stats, first_token_layer_stats
        nonlocal first_token_indexer_layer_stats
        nonlocal first_token_attention_layer_stats
        if step == 1 and first_token_stats is None:
            first_token_stats = ssd_streaming_stats(model)
            first_token_layer_stats = ssd_streaming_layer_stats(model)
            first_token_indexer_layer_stats = deepseek_v4_indexer_layer_stats(model)
            first_token_attention_layer_stats = deepseek_v4_attention_layer_stats(model)
            if args.metal_capture_phase == "after-first-response":
                deferred_capture.start()

    def _capture_prompt_progress(processed: int, total: int) -> None:
        prompt_progress_wall.append({
            "processed_tokens": int(processed),
            "total_tokens": int(total),
            "wall_seconds": time.perf_counter() - progress_t0,
        })
        if 0 < int(processed) < int(total):
            prompt_progress_stats.append({
                "processed_tokens": int(processed),
                "total_tokens": int(total),
                "stats": ssd_streaming_stats(model),
            })

    deferred_capture = _DeferredMetalCapture(metal_capture_path)
    try:
        if args.metal_capture_phase == "generation":
            with _metal_capture(metal_capture_path):
                result = generate_with_metadata(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    prompt_progress_callback=_capture_prompt_progress,
                    response_callback=_capture_first_token_stats,
                )
        else:
            result = generate_with_metadata(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                prompt_progress_callback=_capture_prompt_progress,
                response_callback=_capture_first_token_stats,
            )
    finally:
        deferred_capture.stop()
    final_stats = ssd_streaming_stats(model)
    final_layer_stats = ssd_streaming_layer_stats(model)
    final_indexer_layer_stats = deepseek_v4_indexer_layer_stats(model)
    final_attention_layer_stats = deepseek_v4_attention_layer_stats(model)
    if first_token_stats is None:
        first_token_stats = final_stats
    if first_token_layer_stats is None:
        first_token_layer_stats = final_layer_stats
    if first_token_indexer_layer_stats is None:
        first_token_indexer_layer_stats = final_indexer_layer_stats
    if first_token_attention_layer_stats is None:
        first_token_attention_layer_stats = final_attention_layer_stats
    payload = speed_snapshot_payload(
        package_dir=args.package_dir,
        manifest=manifest,
        prompt_source=prompt_source,
        prompt_mode=prompt_mode,
        result=result,
        stats=final_stats,
        layer_stats=final_layer_stats,
        quality_valid=True,
        metal_capture=_metal_capture_metadata(
            metal_capture_path,
            phase=args.metal_capture_phase,
        ),
        phase_stats={
            "start": start_stats,
            "first_token": first_token_stats,
            "final": final_stats,
        },
        phase_layer_stats={
            "start": start_layer_stats,
            "first_token": first_token_layer_stats,
            "final": final_layer_stats,
        },
        phase_indexer_layer_stats={
            "start": start_indexer_layer_stats,
            "first_token": first_token_indexer_layer_stats,
            "final": final_indexer_layer_stats,
        },
        phase_attention_layer_stats={
            "start": start_attention_layer_stats,
            "first_token": first_token_attention_layer_stats,
            "final": final_attention_layer_stats,
        },
        prompt_progress_stats=prompt_progress_stats,
        prompt_progress_wall=prompt_progress_wall,
        indexer_layer_stats=final_indexer_layer_stats,
        attention_layer_stats=final_attention_layer_stats,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out is None:
        print(text)
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
