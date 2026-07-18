"""L2 micro-golden evidence."""

from __future__ import annotations

from moespresso.core.artifact import validate_base
from moespresso.correctness.goldens import l2_micro_goldens


def test_l2_micro_goldens_emit_valid_evidence():
    ev = l2_micro_goldens()
    assert validate_base(ev) == []
    assert ev["rung"] == "L2"
    assert ev["status"] == "valid"
    assert ev["summary"]["failed"] == 0
    assert {m["case"] for m in ev["metrics"]} >= {
        "tq_unpack_1bit",
        "tq_unpack_2bit",
        "tq_unpack_4bit",
        "fused_gate_up_split",
        "conv1d_norm_shift_trigger",
        "affine_sidecar_shapes",
    }


def test_l2_micro_goldens_record_independent_reference_provenance():
    ev = l2_micro_goldens()
    assert all(r["kind"] == "independent" for r in ev["reference_provenance"])
