"""Unit tests for the speculative-decoding measurement battery.

Covers report assembly, prompt-set validation, the mactop snapshot
parser, and the hidden-tap install/uninstall lifecycle on dummy layers.
No model is loaded and no package is read.
"""

import json

import mlx.core as mx
import pytest

from moespresso.correctness.deepseek_v4.spec_battery import (
    EVALUATION_CATEGORIES,
    EVALUATION_PROMPTS,
    aggregates_by_repeat,
    assemble_report,
    divergence_stats,
    parse_mactop_snapshot,
    plain_arm_record,
    repeat_aggregates,
    spec_arm_record,
    thermal_snapshot,
    uninstall_hidden_tap,
    validate_prompt_set,
)
from moespresso.runtime.deepseek_v4.spec_decode import SpecStats, install_hidden_tap


def _prompt(pid="p1", category="open_chat", text="hello"):
    return {"id": pid, "category": category, "text": text}


class TestPromptSetValidation:
    def test_shipped_prompt_set_is_valid(self):
        validate_prompt_set(EVALUATION_PROMPTS)

    def test_shipped_prompt_set_covers_every_category(self):
        covered = {p["category"] for p in EVALUATION_PROMPTS}
        assert covered == set(EVALUATION_CATEGORIES)
        assert len(EVALUATION_PROMPTS) >= 10

    def test_rejects_empty_set(self):
        with pytest.raises(ValueError, match="empty prompt set"):
            validate_prompt_set(())

    def test_rejects_duplicate_ids(self):
        prompts = [
            _prompt(pid="dup", category=c, text="t") for c in EVALUATION_CATEGORIES
        ]
        with pytest.raises(ValueError, match="duplicate prompt id"):
            validate_prompt_set(prompts)

    def test_rejects_unknown_category(self):
        prompts = list(EVALUATION_PROMPTS) + [_prompt(pid="x", category="poetry")]
        with pytest.raises(ValueError, match="unknown category"):
            validate_prompt_set(prompts)

    def test_rejects_empty_text(self):
        prompts = list(EVALUATION_PROMPTS[:-1]) + [
            _prompt(pid="x", category=EVALUATION_PROMPTS[-1]["category"], text="  ")
        ]
        with pytest.raises(ValueError, match="missing or empty 'text'"):
            validate_prompt_set(prompts)

    def test_rejects_missing_field(self):
        prompts = list(EVALUATION_PROMPTS) + [{"id": "x", "text": "t"}]
        with pytest.raises(ValueError, match="missing or empty 'category'"):
            validate_prompt_set(prompts)

    def test_rejects_missing_category_coverage(self):
        prompts = [p for p in EVALUATION_PROMPTS if p["category"] != "planning"]
        with pytest.raises(ValueError, match="does not cover categories"):
            validate_prompt_set(prompts)


class TestDivergenceStats:
    def test_identical_streams(self):
        assert divergence_stats([1, 2, 3], [1, 2, 3]) == (None, 3)

    def test_midstream_divergence(self):
        assert divergence_stats([1, 2, 9, 4], [1, 2, 3, 4]) == (2, 2)

    def test_prefix_counts_as_divergence_at_shorter_length(self):
        assert divergence_stats([1, 2], [1, 2, 3]) == (2, 2)
        assert divergence_stats([1, 2, 3], [1, 2]) == (2, 2)

    def test_empty_streams(self):
        assert divergence_stats([], []) == (None, 0)


def _stats(rows):
    """Build SpecStats from (offered, accepted) pairs plus fallbacks."""
    stats = SpecStats()
    for offered, accepted in rows:
        stats.record(offered, accepted)
        stats.submit_length_counts[offered] = (
            stats.submit_length_counts.get(offered, 0) + 1
        )
    return stats


class TestArmRecords:
    def test_plain_arm_record_fields(self):
        record = plain_arm_record([1, 2, 3, 4], 1.0, 2.0, "text")
        assert record["tokens_generated"] == 4
        assert record["decode_tok_per_s"] == pytest.approx(2.0)
        assert record["tok_per_s"] == pytest.approx(4.0 / 3.0)
        assert record["total_seconds"] == pytest.approx(3.0)
        assert record["text"] == "text"

    def test_spec_arm_record_fields(self):
        stats = _stats([(5, 3), (5, 5), (3, 1)])
        plain = plain_arm_record([1, 2, 9, 9], 1.0, 1.0, "p")
        record = spec_arm_record(
            [1, 2, 3, 4], stats, 1.0, "s", plain_record=plain, plain_tokens=[1, 2, 9, 9]
        )
        assert record["tokens_generated"] == 4
        assert record["tok_per_s"] == pytest.approx(4.0)
        assert record["ratio_vs_plain"] == pytest.approx(4.0 / plain["tok_per_s"])
        assert record["rounds"] == 3
        assert record["proposed"] == 13
        assert record["accepted"] == 9
        # tau = (accepted + rounds) / rounds: every round also emits the
        # corrected or bonus token.
        assert record["tau"] == pytest.approx(12.0 / 3.0)
        assert record["per_position_offered"] == [3, 3, 3, 2, 2]
        assert record["per_position_accepted"] == [3, 2, 2, 1, 1]
        assert record["submit_length_counts"] == {"3": 1, "5": 2}
        assert record["first_divergence_vs_plain"] == 2
        assert record["identical_prefix_length"] == 2
        assert record["tokens_equal_plain"] is False

    def test_spec_arm_record_without_plain(self):
        record = spec_arm_record([1, 2], _stats([(2, 2)]), 0.5, "s")
        assert record["ratio_vs_plain"] is None
        assert record["first_divergence_vs_plain"] is None
        assert record["identical_prefix_length"] is None
        assert record["tokens_equal_plain"] is None


def _prompt_result(category, ratios_taus):
    """One prompt's arms dict from {drafter: (ratio, tau)}."""
    arms = {"plain": {"tok_per_s": 10.0}}
    for name, (ratio, tau) in ratios_taus.items():
        arms[name] = {"ratio_vs_plain": ratio, "tau": tau}
    return {"category": category, "arms": arms}


class TestAggregates:
    def test_repeat_aggregates_mean_median_and_categories(self):
        prompts = {
            "a": _prompt_result("open_chat", {"dspark": (1.1, 2.7), "mtp": (1.0, 2.2)}),
            "b": _prompt_result("code_generation", {"dspark": (1.3, 3.1), "mtp": (1.2, 2.4)}),
            "c": _prompt_result("open_chat", {"dspark": (1.2, 2.9), "mtp": (0.8, 2.0)}),
        }
        agg = repeat_aggregates(prompts)
        assert set(agg) == {"dspark", "mtp"}
        assert agg["dspark"]["prompts"] == 3
        assert agg["dspark"]["mean_ratio"] == pytest.approx(1.2)
        assert agg["dspark"]["median_ratio"] == pytest.approx(1.2)
        assert agg["dspark"]["mean_tau"] == pytest.approx((2.7 + 3.1 + 2.9) / 3)
        chat = agg["dspark"]["per_category"]["open_chat"]
        assert chat["prompts"] == 2
        assert chat["mean_ratio"] == pytest.approx(1.15)
        code = agg["mtp"]["per_category"]["code_generation"]
        assert code["mean_ratio"] == pytest.approx(1.2)
        assert code["mean_tau"] == pytest.approx(2.4)

    def test_repeat_aggregates_skips_missing_ratio(self):
        prompts = {
            "a": _prompt_result("open_chat", {"mtp": (None, 2.0)}),
            "b": _prompt_result("open_chat", {"mtp": (1.5, 2.5)}),
        }
        agg = repeat_aggregates(prompts)
        assert agg["mtp"]["prompts"] == 2
        assert agg["mtp"]["mean_ratio"] == pytest.approx(1.5)
        assert agg["mtp"]["mean_tau"] == pytest.approx(2.25)

    def test_repeat_aggregates_plain_only_is_empty(self):
        prompts = {"a": {"category": "open_chat", "arms": {"plain": {}}}}
        assert repeat_aggregates(prompts) == {}

    def test_aggregates_by_repeat_side_by_side(self):
        entries = [
            {"aggregates": {"dspark": {"mean_ratio": 1.1, "median_ratio": 1.0,
                                       "mean_tau": 2.5}}},
            {"aggregates": {"dspark": {"mean_ratio": 1.2, "median_ratio": 1.1,
                                       "mean_tau": 2.6},
                            "mtp": {"mean_ratio": 1.0, "median_ratio": 1.0,
                                    "mean_tau": 2.1}}},
        ]
        series = aggregates_by_repeat(entries)
        assert series["dspark"]["mean_ratio"] == [1.1, 1.2]
        assert series["dspark"]["mean_tau"] == [2.5, 2.6]
        # A drafter absent from a repeat stays aligned via None.
        assert series["mtp"]["mean_ratio"] == [None, 1.0]


class TestAssembleReport:
    def test_assemble_report_shape(self):
        entries = [
            {"repeat": 0, "prompts": {}, "aggregates": {"mtp": {"mean_ratio": 1.0}}},
            {"repeat": 1, "prompts": {}, "aggregates": {"mtp": {"mean_ratio": 1.1}}},
        ]
        report = assemble_report(
            package="/pkg",
            package_artifact_id="abc123",
            sidecars={"mtp": {"path": "/sc", "artifact_id": "def456"}},
            max_new_tokens=300,
            adaptive_cap=True,
            environment={"thermal_at_start": {"available": False, "reason": "x"}},
            repeat_entries=entries,
            prompt_set=EVALUATION_PROMPTS,
        )
        assert report["repeats"] == 2
        assert report["package_artifact_id"] == "abc123"
        assert report["sidecars"]["mtp"]["artifact_id"] == "def456"
        assert report["temperature"] == 0.0
        assert len(report["prompt_set"]) == len(EVALUATION_PROMPTS)
        assert all(set(p) == {"id", "category"} for p in report["prompt_set"])
        assert report["aggregates_by_repeat"]["mtp"]["mean_ratio"] == [1.0, 1.1]
        # The report must serialize as JSON without custom encoders.
        json.dumps(report)


SAMPLE_MACTOP = json.dumps(
    [
        {
            "timestamp": "2026-07-24T00:00:00+02:00",
            "soc_metrics": {"gpu_temp": 35.17, "cpu_temp": 41.9},
            "thermal_state": "Nominal",
        }
    ]
)


class TestThermalSnapshot:
    def test_parse_ok(self):
        snap = parse_mactop_snapshot(SAMPLE_MACTOP)
        assert snap == {
            "available": True,
            "gpu_temp_c": pytest.approx(35.17),
            "thermal_state": "Nominal",
        }

    def test_parse_rejects_non_list(self):
        with pytest.raises(ValueError):
            parse_mactop_snapshot("{}")

    def test_parse_rejects_missing_fields(self):
        with pytest.raises(ValueError):
            parse_mactop_snapshot(json.dumps([{"soc_metrics": {}}]))

    def test_snapshot_degrades_without_binary(self, monkeypatch):
        from moespresso.correctness.deepseek_v4 import spec_battery

        monkeypatch.setattr(spec_battery.shutil, "which", lambda name: None)
        snap = thermal_snapshot()
        assert snap["available"] is False
        assert "mactop" in snap["reason"]


class _Layer:
    """Minimal stand-in for a decoder layer: a layer_id and a call."""

    def __init__(self, layer_id):
        self.layer_id = layer_id

    def __call__(self, x):
        return x + self.layer_id


class _Inner:
    def __init__(self, layers):
        self.layers = layers


class _DummyModel:
    def __init__(self, n_layers):
        self.model = _Inner([_Layer(i) for i in range(n_layers)])


def _identity(layer_id, out):
    return out


def _double(layer_id, out):
    return out * 2


class TestTapLifecycle:
    def test_taps_do_not_compose_on_shared_layers(self):
        # A second install on a shared layer replaces the first tap's
        # registration there: the first tap stops seeing that layer and
        # its take_rows fails on the missing id. This is the behavior the
        # battery's install/uninstall bracketing exists for.
        model = _DummyModel(3)
        tap_a = install_hidden_tap(model, [0, 2], _identity)
        tap_b = install_hidden_tap(model, [2], _double)
        assert model.model.layers[2]._moespresso_hidden_tap is tap_b
        tap_a.active = True
        for layer in model.model.layers:
            layer(mx.ones((1, 1, 2)))
        assert set(tap_a.rows) == {0}
        with pytest.raises(KeyError):
            tap_a.take_rows()
        uninstall_hidden_tap(model, tap_a)
        uninstall_hidden_tap(model, tap_b)

    def test_uninstall_removes_own_registrations_only(self):
        model = _DummyModel(3)
        tap_a = install_hidden_tap(model, [0], _identity)
        tap_b = install_hidden_tap(model, [1, 2], _identity)
        uninstall_hidden_tap(model, tap_b)
        assert tap_b.active is False
        assert model.model.layers[1]._moespresso_hidden_tap is None
        assert model.model.layers[2]._moespresso_hidden_tap is None
        assert model.model.layers[0]._moespresso_hidden_tap is tap_a
        # A layer claimed by another tap in the meantime is left alone.
        tap_c = install_hidden_tap(model, [0], _double)
        uninstall_hidden_tap(model, tap_a)
        assert model.model.layers[0]._moespresso_hidden_tap is tap_c
        uninstall_hidden_tap(model, tap_c)

    def test_sequential_reinstall_restores_shared_layer(self):
        # The battery's per-arm bracketing: after another drafter's tap
        # claimed a shared layer, a fresh install for the first drafter
        # records every one of its layers with its own transform again.
        model = _DummyModel(3)
        tap_a = install_hidden_tap(model, [0, 2], _identity)
        tap_b = install_hidden_tap(model, [2], _double)
        uninstall_hidden_tap(model, tap_a)
        uninstall_hidden_tap(model, tap_b)
        tap_a2 = install_hidden_tap(model, [0, 2], _identity)
        tap_a2.active = True
        for layer in model.model.layers:
            layer(mx.ones((1, 1, 2)))
        assert set(tap_a2.rows) == {0, 2}
        rows = tap_a2.take_rows()
        # Layer 0 emits ones, layer 2 emits threes; the tap concatenates
        # tap-order rows along the last axis.
        assert rows.shape == (1, 1, 4)
        assert mx.array_equal(rows, mx.array([[[1.0, 1.0, 3.0, 3.0]]]))
        assert tap_a2.rows == {}
        uninstall_hidden_tap(model, tap_a2)
