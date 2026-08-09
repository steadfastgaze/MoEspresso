"""The Q4 KL panel, on synthetic dumps.

Every dump here is constructed in the test so the expected KL, entropy
threshold, agreement rate, and marker mass are computable by hand. No teacher
dump, package, or oracle material is read.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from moespresso.core.artifact import compute_artifact_id, validate_base
from moespresso.correctness.deepseek_v4.q4 import (
    Q4_EVIDENCE_SCHEMA,
    Q4_MARKERS_ENV,
    Q4_NARROW_MARKER_CLASSES,
    CandidateDump,
    build_q4_external_evidence,
    candidate_dump_from_arrays,
    free_run_length_accounting,
    load_candidate_dump,
    load_marker_set,
    load_teacher_dump,
    make_deepseek_v4_q4_evidence,
    main as q4_main,
    score_kl_panel,
    teacher_dump_from_arrays,
    validate_deepseek_v4_q4_evidence,
)

# Two marker forms in the narrow classes and one outside them, so the aggregate
# and the narrow subset are provably different columns.
MARKER_PAYLOAD = {
    "version": 1,
    "resolved": [
        {"text": "wait", "id": 3},
        {"text": " Alternatively", "id": 4},
        {"text": "however", "id": 5},
    ],
    "class_of_source_word": {
        "wait": "hesitation_interjection",
        "Alternatively": "branch_alternative",
        "however": "generic_contrastive_connective",
    },
}


def _markers(tmp_path, payload=None):
    path = tmp_path / "markers.json"
    path.write_text(json.dumps(payload or MARKER_PAYLOAD), encoding="utf-8")
    return load_marker_set(path)


def _teacher(top_ids, top_logits, log_partition, argmax=None):
    top_logits = np.asarray(top_logits, dtype=np.float64)
    top_ids = np.asarray(top_ids, dtype=np.int64)
    partition = np.asarray(log_partition, dtype=np.float64)
    ids = np.tile(
        np.arange(top_logits.shape[1], dtype=np.int64),
        (top_logits.shape[0], 1),
    )
    return teacher_dump_from_arrays(
        ids=ids,
        top_ids=top_ids,
        top_logprobs=top_logits - partition[..., None],
        argmax=np.asarray(argmax if argmax is not None else top_ids[..., 0], dtype=np.int64),
    )


def _candidate(top_logits, log_partition, argmax, *, top_ids=None, ids=None):
    top_logits = np.asarray(top_logits, dtype=np.float64)
    partition = np.asarray(log_partition, dtype=np.float64)
    if top_ids is None:
        top_ids = np.tile(
            np.arange(top_logits.shape[-1], dtype=np.int64),
            top_logits.shape[:-1] + (1,),
        )
    if ids is None:
        ids = np.tile(
            np.arange(top_logits.shape[1], dtype=np.int64),
            (top_logits.shape[0], 1),
        )
    return candidate_dump_from_arrays(
        ids=ids,
        top_ids=top_ids,
        top_logprobs=top_logits - partition[..., None],
        argmax=np.asarray(argmax, dtype=np.int64),
        normalization="full_vocab",
        source="synthetic",
    )


def _uniform_pair(chunks=1, steps=4, support=4):
    """Teacher and candidate that are the identical uniform distribution."""
    top_ids = np.tile(np.arange(support, dtype=np.int64), (chunks, steps, 1))
    logits = np.zeros((chunks, steps, support))
    partition = np.full((chunks, steps), math.log(support))
    argmax = np.zeros((chunks, steps), dtype=np.int64)
    return (
        _teacher(top_ids, logits, partition, argmax),
        _candidate(logits, partition, argmax, top_ids=top_ids),
    )


def _codes(findings):
    return {v.code for v in findings}


# --- marker set --------------------------------------------------------------


def test_marker_set_requires_an_explicit_path():
    with pytest.raises(SystemExit) as excinfo:
        load_marker_set(None)

    assert Q4_MARKERS_ENV in str(excinfo.value)


def test_narrow_subset_follows_the_marker_classes(tmp_path):
    markers = _markers(tmp_path)

    assert markers.ids == (3, 4, 5)
    assert markers.narrow_ids == (3, 4)
    assert "generic_contrastive_connective" not in Q4_NARROW_MARKER_CLASSES


def test_marker_set_fails_closed_on_a_shape_it_cannot_read(tmp_path):
    path = tmp_path / "markers.json"
    path.write_text(json.dumps({"version": 1, "resolved": []}), encoding="utf-8")

    with pytest.raises(SystemExit):
        load_marker_set(path)


# --- panel math --------------------------------------------------------------


def test_identical_distributions_score_zero_kl_and_full_agreement(tmp_path):
    teacher, candidate = _uniform_pair()

    panel = score_kl_panel(teacher, candidate, markers=_markers(tmp_path), probe="ind")

    assert panel["kl"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert panel["kl"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert panel["top1_agreement"]["all_positions"] == 1.0
    assert panel["positions"] == 4
    assert panel["support_width"] == 4
    assert len(panel["support_sha256"]) == 64


def test_kl_matches_the_hand_computed_divergence(tmp_path):
    """Uniform teacher against a candidate that moved mass onto one token."""
    support = 2
    top_ids = np.array([[[0, 1]]], dtype=np.int64)
    teacher = _teacher(top_ids, [[[0.0, 0.0]]], [[math.log(support)]], [[0]])
    # Candidate log-probabilities log(0.75), log(0.25) inside the support.
    candidate = _candidate(
        [[[math.log(0.75), math.log(0.25)]]],
        [[0.0]],
        [[0]],
    )

    panel = score_kl_panel(teacher, candidate, markers=_markers(tmp_path))

    expected = 0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
    assert panel["kl"]["mean"] == pytest.approx(expected)


def test_kl_is_invariant_to_the_candidate_partition(tmp_path):
    """Both sides renormalize inside the teacher's support, so a candidate that
    spends different mass outside it does not move the KL."""
    teacher, candidate = _uniform_pair(steps=2)
    shifted = CandidateDump(
        ids=candidate.ids,
        top_ids=candidate.top_ids,
        top_logprobs=candidate.top_logprobs - 1.25,
        argmax=candidate.argmax,
        normalization="full_vocab",
        source="synthetic",
    )

    markers = _markers(tmp_path)
    base = score_kl_panel(teacher, candidate, markers=markers)
    moved = score_kl_panel(teacher, shifted, markers=markers)

    assert moved["kl"]["mean"] == pytest.approx(base["kl"]["mean"], abs=1e-12)
    # Marker mass is raw, so it does move; that is the point of not
    # renormalizing it.
    assert (
        moved["markers"]["all_positions"]["aggregate"]["candidate_mass"]
        < base["markers"]["all_positions"]["aggregate"]["candidate_mass"]
    )


def test_top_k_mass_is_measured_not_assumed(tmp_path):
    """A teacher whose support covers half the distribution says so."""
    top_ids = np.array([[[0, 1]]], dtype=np.int64)
    teacher = _teacher(top_ids, [[[0.0, 0.0]]], [[math.log(4.0)]], [[0]])
    candidate = _candidate([[[0.0, 0.0]]], [[math.log(4.0)]], [[0]])

    panel = score_kl_panel(teacher, candidate, markers=_markers(tmp_path))

    assert panel["teacher_top_k_mass_mean"] == pytest.approx(0.5)


def test_entropy_conditioning_uses_the_teacher_alone(tmp_path):
    """Two candidates against one teacher are conditioned on the same mask."""
    top_ids = np.tile(np.array([0, 1, 2, 3], dtype=np.int64), (1, 4, 1))
    # Two flat positions (high entropy) and two peaked positions (low entropy).
    logits = np.array([[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0, 0.0],
    ]])
    partition = np.zeros((1, 4))
    teacher = _teacher(top_ids, logits, partition, [[0, 0, 0, 0]])
    markers = _markers(tmp_path)

    first = score_kl_panel(
        teacher,
        _candidate(logits, partition, [[0, 0, 0, 0]]),
        markers=markers,
        entropy_quantile=0.5,
    )
    second = score_kl_panel(
        teacher,
        _candidate(logits + np.array([1.0, 0.0, 0.0, 0.0]), partition, [[1, 1, 1, 1]]),
        markers=markers,
        entropy_quantile=0.5,
    )

    assert first["entropy"]["conditioned_positions"] == 2
    assert second["entropy"]["conditioned_positions"] == 2
    assert first["entropy"]["threshold"] == second["entropy"]["threshold"]
    assert second["top1_agreement"]["all_positions"] == 0.0


def test_marker_reporting_carries_forms_aggregate_and_narrow_subset(tmp_path):
    top_ids = np.array([[[3, 4, 5, 9]]], dtype=np.int64)
    teacher = _teacher(top_ids, [[[0.0, 0.0, 0.0, 0.0]]], [[math.log(4.0)]], [[3]])
    # Candidate doubles the mass on the first marker form relative to uniform.
    candidate = _candidate(
        [[[math.log(2.0), 0.0, 0.0, 0.0]]],
        [[math.log(5.0)]],
        [[3]],
        top_ids=top_ids,
    )

    panel = score_kl_panel(teacher, candidate, markers=_markers(tmp_path))
    block = panel["markers"]["all_positions"]
    by_text = {row["text"]: row for row in block["forms"]}

    assert set(by_text) == {"wait", " Alternatively", "however"}
    assert by_text["wait"]["occupancy"] == 1.0
    assert by_text["wait"]["teacher_mass"] == pytest.approx(0.25)
    assert by_text["wait"]["candidate_mass"] == pytest.approx(0.4)
    assert by_text["wait"]["ratio"] == pytest.approx(1.6)
    # Three marker forms sit in the support, two of them in the narrow classes.
    assert block["aggregate"]["teacher_mass"] == pytest.approx(0.75)
    assert block["narrow_subset"]["teacher_mass"] == pytest.approx(0.5)
    assert block["narrow_subset"]["token_ids"] == 2
    assert block["narrow_subset"]["classes"] == list(Q4_NARROW_MARKER_CLASSES)


def test_marker_occupancy_reports_absence(tmp_path):
    """A form outside the teacher's support has zero occupancy and no ratio."""
    top_ids = np.array([[[3, 7, 8, 9]]], dtype=np.int64)
    teacher = _teacher(top_ids, [[[0.0, 0.0, 0.0, 0.0]]], [[math.log(4.0)]], [[3]])
    candidate = _candidate(
        [[[0.0, 0.0, 0.0, 0.0]]],
        [[math.log(4.0)]],
        [[3]],
        top_ids=top_ids,
    )

    block = score_kl_panel(
        teacher, candidate, markers=_markers(tmp_path)
    )["markers"]["all_positions"]
    by_text = {row["text"]: row for row in block["forms"]}

    assert by_text["however"]["occupancy"] == 0.0
    assert by_text["however"]["teacher_mass"] == 0.0
    assert by_text["however"]["ratio"] is None


def test_panel_refuses_misaligned_dumps(tmp_path):
    teacher, _candidate_ok = _uniform_pair(steps=4)
    _teacher_other, shorter = _uniform_pair(steps=2)

    with pytest.raises(SystemExit) as excinfo:
        score_kl_panel(teacher, shorter, markers=_markers(tmp_path))

    assert "same positions" in str(excinfo.value)


def test_panel_refuses_same_shape_with_different_token_positions(tmp_path):
    teacher, candidate = _uniform_pair(steps=4)
    shifted = CandidateDump(
        ids=candidate.ids + 1,
        top_ids=candidate.top_ids,
        top_logprobs=candidate.top_logprobs,
        argmax=candidate.argmax,
        normalization=candidate.normalization,
        source="shifted",
    )

    with pytest.raises(SystemExit, match="different token ids"):
        score_kl_panel(teacher, shifted, markers=_markers(tmp_path))


def test_panel_refuses_same_shape_with_different_teacher_support(tmp_path):
    teacher, candidate = _uniform_pair(steps=4)
    reordered = CandidateDump(
        ids=candidate.ids,
        top_ids=candidate.top_ids[..., ::-1],
        top_logprobs=candidate.top_logprobs,
        argmax=candidate.argmax,
        normalization=candidate.normalization,
        source="reordered",
    )

    with pytest.raises(SystemExit, match="ordered top-K support"):
        score_kl_panel(teacher, reordered, markers=_markers(tmp_path))


# --- dump IO -----------------------------------------------------------------


def test_teacher_dump_round_trips_through_npz(tmp_path):
    path = tmp_path / "teacher.npz"
    np.savez(
        path,
        ids=np.zeros((1, 2), dtype=np.int32),
        top_ids=np.array([[[0, 1], [0, 1]]], dtype=np.int32),
        top_logits=np.zeros((1, 2, 2), dtype=np.float32),
        log_partition=np.full((1, 2), math.log(2.0), dtype=np.float32),
        argmax=np.zeros((1, 2), dtype=np.int32),
    )

    dump = load_teacher_dump(path)

    assert dump.normalization == "full_vocab"
    assert dump.shape == (1, 2, 2)
    assert dump.positions == 2
    assert np.array_equal(dump.ids, np.zeros((1, 2), dtype=np.int64))
    assert np.allclose(np.exp(dump.top_logprobs).sum(axis=-1), 1.0)


def test_earlier_schema_is_read_and_labelled(tmp_path):
    """The earlier dumps stored log-probabilities already renormalized inside
    the support; reading them is fine as long as the panel says which it was."""
    path = tmp_path / "teacher.npz"
    np.savez(
        path,
        ids=np.zeros((1, 1), dtype=np.int32),
        top_ids=np.array([[[0, 1]]], dtype=np.int32),
        top_lp=np.full((1, 1, 2), -math.log(2.0), dtype=np.float32),
        argmax=np.zeros((1, 1), dtype=np.int32),
    )

    dump = load_teacher_dump(path)

    assert dump.normalization == "top_k_renormalized"


def test_top_logits_without_a_partition_is_refused(tmp_path):
    path = tmp_path / "teacher.npz"
    np.savez(
        path,
        top_ids=np.array([[[0, 1]]], dtype=np.int32),
        top_logits=np.zeros((1, 1, 2), dtype=np.float32),
        argmax=np.zeros((1, 1), dtype=np.int32),
    )

    with pytest.raises(SystemExit) as excinfo:
        load_teacher_dump(path)

    assert "log_partition" in str(excinfo.value)


def test_candidate_dump_requires_an_argmax(tmp_path):
    path = tmp_path / "candidate.npz"
    np.savez(
        path,
        top_logits=np.zeros((1, 1, 2), dtype=np.float32),
        log_partition=np.zeros((1, 1), dtype=np.float32),
    )

    with pytest.raises(SystemExit) as excinfo:
        load_candidate_dump(path)

    assert "argmax" in str(excinfo.value)


def test_candidate_dump_round_trips_with_position_and_support_identity(tmp_path):
    path = tmp_path / "candidate.npz"
    np.savez(
        path,
        ids=np.array([[7, 8]], dtype=np.int32),
        top_ids=np.array([[[3, 4], [5, 6]]], dtype=np.int32),
        top_logits=np.zeros((1, 2, 2), dtype=np.float32),
        log_partition=np.full((1, 2), math.log(2.0), dtype=np.float32),
        argmax=np.array([[3, 5]], dtype=np.int32),
    )

    dump = load_candidate_dump(path)

    assert np.array_equal(dump.ids, [[7, 8]])
    assert np.array_equal(dump.top_ids, [[[3, 4], [5, 6]]])


def test_candidate_dump_without_top_ids_is_refused(tmp_path):
    path = tmp_path / "candidate.npz"
    np.savez(
        path,
        ids=np.zeros((1, 1), dtype=np.int32),
        top_logits=np.zeros((1, 1, 2), dtype=np.float32),
        log_partition=np.zeros((1, 1), dtype=np.float32),
        argmax=np.zeros((1, 1), dtype=np.int32),
    )

    with pytest.raises(SystemExit, match="no top_ids"):
        load_candidate_dump(path)


def _write_cli_inputs(tmp_path):
    teacher = tmp_path / "teacher.npz"
    candidate = tmp_path / "candidate.npz"
    markers = tmp_path / "markers.json"
    common = {
        "ids": np.array([[7]], dtype=np.int32),
        "top_ids": np.array([[[0, 1]]], dtype=np.int32),
        "top_logits": np.zeros((1, 1, 2), dtype=np.float32),
        "log_partition": np.full((1, 1), math.log(2.0), dtype=np.float32),
        "argmax": np.zeros((1, 1), dtype=np.int32),
    }
    np.savez(teacher, **common)
    np.savez(candidate, **common)
    markers.write_text(json.dumps(MARKER_PAYLOAD), encoding="utf-8")
    return teacher, candidate, markers


def test_q4_cli_requires_bars_unless_report_only(tmp_path):
    teacher, candidate, markers = _write_cli_inputs(tmp_path)
    base = [
        "--teacher", str(teacher),
        "--candidate", str(candidate),
        "--markers", str(markers),
    ]

    with pytest.raises(SystemExit) as excinfo:
        q4_main(base)

    assert excinfo.value.code == 2
    assert q4_main([*base, "--report-only"]) == 0


# --- free-run length accounting ---------------------------------------------


def test_free_run_accounting_reports_and_does_not_judge():
    accounting = free_run_length_accounting([
        {"id": "a", "candidate_tokens": 120, "reference_tokens": 100, "finish_reason": "stop"},
        {"id": "b", "candidate_tokens": 80, "reference_tokens": 100, "finish_reason": "stop"},
        {"id": "c", "candidate_tokens": 300, "reference_tokens": 100, "finish_reason": "length"},
    ])

    assert accounting["cases"] == 3
    assert accounting["candidate_tokens_total"] == 500
    assert accounting["candidate_tokens_median"] == 120
    assert accounting["length_ratio_total"] == pytest.approx(500 / 300)
    assert accounting["finish_reasons"] == {"stop": 2, "length": 1}
    assert accounting["cases_detail"][0]["length_ratio"] == pytest.approx(1.2)
    assert "passed" not in accounting


def test_free_run_accounting_without_a_reference_reports_lengths_only():
    accounting = free_run_length_accounting([{"id": "a", "candidate_tokens": 40}])

    assert accounting["reference_tokens_total"] is None
    assert accounting["length_ratio_total"] is None
    assert accounting["cases_detail"][0]["length_ratio"] is None


def test_free_run_accounting_refuses_a_record_without_a_length():
    with pytest.raises(SystemExit):
        free_run_length_accounting([{"id": "a"}])


def test_panel_carries_free_run_accounting_when_supplied(tmp_path):
    teacher, candidate = _uniform_pair()

    without = score_kl_panel(teacher, candidate, markers=_markers(tmp_path))
    with_records = score_kl_panel(
        teacher,
        candidate,
        markers=_markers(tmp_path),
        free_run_records=[{"id": "a", "candidate_tokens": 12}],
    )

    assert without["free_run"] is None
    assert with_records["free_run"]["cases"] == 1


# --- evidence contract -------------------------------------------------------


def _external(tmp_path, bars=None):
    markers = _markers(tmp_path)
    teacher, candidate = _uniform_pair()
    panel = score_kl_panel(teacher, candidate, markers=markers, probe="ind")
    return build_q4_external_evidence(
        panels=[panel],
        teacher_sources=["/tmp/teacher_ind.npz"],
        candidate_sources=["/tmp/candidate_ind.npz"],
        markers=markers,
        bars=bars,
    )


def test_unbarred_panel_emits_draft_measurement_evidence(tmp_path):
    external = _external(tmp_path)

    findings = validate_deepseek_v4_q4_evidence(external)
    artifact = make_deepseek_v4_q4_evidence({"gate": "Q4"}, external)

    assert _codes(findings) == {"deepseek_v4.q4.unbarred"}
    assert validate_base(artifact) == []
    assert artifact["rung"] == "Q4"
    assert artifact["status"] == "draft"
    assert artifact["summary"]["judgement"] == "instrument_valid_unbarred"
    assert artifact["summary"]["probes"] == ["ind"]
    assert artifact["artifact_id"] == compute_artifact_id(artifact)


def test_complete_passing_bars_emit_quality_valid_evidence(tmp_path):
    external = _external(tmp_path, bars={
        "kl_mean_max": 0.1,
        "top1_agreement_min": 0.9,
        "marker_ratio_max": 1.1,
    })

    artifact = make_deepseek_v4_q4_evidence({"gate": "Q4"}, external)

    assert artifact["status"] == "valid"
    assert artifact["summary"]["judgement"] == "quality_bar_passed"
    assert artifact["validation"] == []


def test_levels_remain_measurements_without_declared_bars(tmp_path):
    """A large but well-formed divergence is a reading, not a failure."""
    markers = _markers(tmp_path)
    top_ids = np.array([[[0, 1]]], dtype=np.int64)
    teacher = _teacher(top_ids, [[[0.0, 0.0]]], [[math.log(2.0)]], [[0]])
    candidate = _candidate([[[math.log(0.99), math.log(0.01)]]], [[0.0]], [[1]])
    external = build_q4_external_evidence(
        panels=[score_kl_panel(teacher, candidate, markers=markers, probe="ind")],
        teacher_sources=["/tmp/t.npz"],
        candidate_sources=["/tmp/c.npz"],
        markers=markers,
    )

    assert _codes(validate_deepseek_v4_q4_evidence(external)) == {
        "deepseek_v4.q4.unbarred"
    }
    assert make_deepseek_v4_q4_evidence({"gate": "Q4"}, external)["status"] == "draft"
    assert external["panels"][0]["kl"]["mean"] > 1.0
    assert external["panels"][0]["top1_agreement"]["all_positions"] == 0.0


def test_declared_bars_gate(tmp_path):
    markers = _markers(tmp_path)
    top_ids = np.array([[[0, 1]]], dtype=np.int64)
    teacher = _teacher(top_ids, [[[0.0, 0.0]]], [[math.log(2.0)]], [[0]])
    candidate = _candidate([[[math.log(0.99), math.log(0.01)]]], [[0.0]], [[1]])
    external = build_q4_external_evidence(
        panels=[score_kl_panel(teacher, candidate, markers=markers, probe="ind")],
        teacher_sources=["/tmp/t.npz"],
        candidate_sources=["/tmp/c.npz"],
        markers=markers,
        bars={
            "kl_mean_max": 0.05,
            "top1_agreement_min": 0.9,
            "marker_ratio_max": 1.1,
        },
    )

    codes = _codes(validate_deepseek_v4_q4_evidence(external))

    assert "deepseek_v4.q4.kl_above_bar" in codes
    assert "deepseek_v4.q4.top1_below_bar" in codes


def test_narrow_subset_inflation_gates_against_a_declared_ceiling(tmp_path):
    external = _external(tmp_path, bars={
        "kl_mean_max": 0.1,
        "top1_agreement_min": 0.9,
        "marker_ratio_max": 1.05,
    })
    block = external["panels"][0]["markers"]["top_entropy_positions"]["narrow_subset"]
    block["ratio"] = 1.5

    codes = _codes(validate_deepseek_v4_q4_evidence(external))

    assert "deepseek_v4.q4.marker_ratio_above_bar" in codes


def test_declared_marker_bar_requires_an_evaluated_ratio(tmp_path):
    external = _external(tmp_path, bars={
        "kl_mean_max": 0.1,
        "top1_agreement_min": 0.9,
        "marker_ratio_max": 1.05,
    })
    block = external["panels"][0]["markers"]["top_entropy_positions"]["narrow_subset"]
    block["ratio"] = None

    artifact = make_deepseek_v4_q4_evidence({"gate": "Q4"}, external)

    assert artifact["status"] == "invalid"
    assert "deepseek_v4.q4.marker_ratio_unavailable" in {
        row["code"] for row in artifact["validation"]
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kl_mean_max", -0.1),
        ("top1_agreement_min", -0.1),
        ("top1_agreement_min", 1.1),
        ("marker_ratio_max", -0.1),
    ],
)
def test_declared_bars_must_use_metric_domains(tmp_path, key, value):
    bars = {
        "kl_mean_max": 0.1,
        "top1_agreement_min": 0.9,
        "marker_ratio_max": 1.05,
    }
    bars[key] = value

    artifact = make_deepseek_v4_q4_evidence(
        {"gate": "Q4"}, _external(tmp_path, bars=bars)
    )

    assert artifact["status"] == "invalid"
    assert "deepseek_v4.q4.invalid_bar_domains" in {
        row["code"] for row in artifact["validation"]
    }


def test_instrument_defects_gate_without_any_declared_bar(tmp_path):
    external = _external(tmp_path)
    panel = external["panels"][0]
    panel["kl"]["mean"] = -0.5
    panel["top1_agreement"]["all_positions"] = 1.5
    panel["teacher_top_k_mass_mean"] = 1.5

    codes = _codes(validate_deepseek_v4_q4_evidence(external))

    assert "deepseek_v4.q4.negative_kl" in codes
    assert "deepseek_v4.q4.bad_top1_agreement" in codes
    assert "deepseek_v4.q4.bad_top_k_mass" in codes
    assert "deepseek_v4.q4.unbarred" in codes


def test_partial_bars_cannot_produce_quality_valid_evidence(tmp_path):
    external = _external(tmp_path, bars={"kl_mean_max": 0.1})

    artifact = make_deepseek_v4_q4_evidence({"gate": "Q4"}, external)

    assert artifact["status"] == "draft"
    assert "deepseek_v4.q4.incomplete_bars" in {
        row["code"] for row in artifact["validation"]
    }


def test_non_finite_kl_is_blocking(tmp_path):
    external = _external(tmp_path)
    external["panels"][0]["kl"]["mean"] = float("nan")

    artifact = make_deepseek_v4_q4_evidence({"gate": "Q4"}, external)
    codes = {v["code"] for v in artifact["validation"]}

    assert "deepseek_v4.q4.non_finite_kl" in codes
    assert artifact["status"] == "invalid"


def test_evidence_requires_both_marker_position_subsets_and_columns(tmp_path):
    external = _external(tmp_path)
    del external["panels"][0]["markers"]["top_entropy_positions"]
    del external["panels"][0]["markers"]["all_positions"]["narrow_subset"]

    codes = _codes(validate_deepseek_v4_q4_evidence(external))

    assert "deepseek_v4.q4.missing_marker_block" in codes
    assert "deepseek_v4.q4.missing_marker_column" in codes


def test_evidence_requires_the_paired_support_identity(tmp_path):
    external = _external(tmp_path)
    del external["panels"][0]["support_sha256"]

    codes = _codes(validate_deepseek_v4_q4_evidence(external))

    assert "deepseek_v4.q4.bad_support_identity" in codes


def test_evidence_requires_a_scored_panel(tmp_path):
    external = _external(tmp_path)
    external["panels"] = []

    codes = _codes(validate_deepseek_v4_q4_evidence(external))

    assert "deepseek_v4.q4.missing_panels" in codes


def test_evidence_schema_is_pinned(tmp_path):
    external = _external(tmp_path)
    external["schema"] = "something-else"

    codes = _codes(validate_deepseek_v4_q4_evidence(external))

    assert "deepseek_v4.q4.schema_mismatch" in codes
    assert Q4_EVIDENCE_SCHEMA == "ds4-q4-kl-panel-v2"
