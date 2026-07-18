"""optimizer_decision builder: ported optimizer proofs on typed probe_evidence.

Builds minimal real probe_evidence artifacts (typed units, no names to parse) and
pins: the tail constraint lifts the worst layer to tau, bundled tied minima don't
stall, infeasible targets are recorded (not crashed), the backbone has no per-role
floors, the decision is a deterministic content-addressed artifact, and higher
target_quality never lowers bits.
"""

from __future__ import annotations

import pytest

from moespresso.core.artifact import compute_artifact_id, make_artifact, validate_base
from moespresso.optimize.allocate import AFFINE_BITS, EXPERT_BITS, GROUP_SIZES
from moespresso.optimize.affine_elasticity import qwen35_moe_affine_role_profile_v1
from moespresso.optimize.decide import decide
from moespresso.optimize.sizes import affine_bytes

SUBJECT = {"source_root": "toy", "source_format": "hf_safetensors"}
PRODUCER = {"tool": "test", "version": "0"}


def _affine_unit(name, q_by_bits, role="attn.q_proj", layer_index=0,
                 importance=1.0, rows=64, cols=128):
    """Affine unit; quality independent of group size (bits-only control)."""
    filled, last = {}, None
    for b in AFFINE_BITS:
        last = q_by_bits.get(b, last)
        filled[b] = last
    quality = {f"{b}_{gs}": filled[b] for b in AFFINE_BITS for gs in (128, 64, 32)}
    return {"source_name": name, "kind": "affine", "role": role,
            "layer_index": layer_index, "shape": [rows, cols],
            "importance": importance, "imatrix_mapped": True, "quality": quality}


def _affine_unit_grid(name, q_by_bgs, role="attn.q_proj", layer_index=0,
                      importance=1.0, rows=64, cols=128):
    """Affine unit with explicit per-(bits,gs) quality (full 2-D control).

    q_by_bgs: {(bits, gs): q}. Missing tuples default to the lowest given q so the table
    is complete; callers should keep it monotone-in-bits per gs (the monotone envelope runs
    on it). Lets a test set a quality landscape where only a diagonal move helps.
    """
    base = min(q_by_bgs.values())
    quality = {f"{b}_{gs}": q_by_bgs.get((b, gs), base)
               for b in AFFINE_BITS for gs in (128, 64, 32)}
    return {"source_name": name, "kind": "affine", "role": role,
            "layer_index": layer_index, "shape": [rows, cols],
            "importance": importance, "imatrix_mapped": True, "quality": quality}


def _expert_unit(name, layer, projection, q_by_bits, n_experts=4,
                 importance=1.0, rows=64, cols=128):
    quality = {str(b): q_by_bits[b] for b in EXPERT_BITS}
    return {"source_name": name, "kind": "expert", "role": f"moe.expert.{projection}",
            "layer_index": layer, "projection": projection, "n_experts": n_experts,
            "sampled": 2, "shape": [rows, cols], "importance": importance,
            "imatrix_mapped": True, "quality": quality}


def _lossless_mxfp4_expert_unit(name="expert", projection="gate", q4=0.90):
    unit = _expert_unit(
        name,
        0,
        projection,
        {1: 0.10, 2: 0.50, 4: q4},
        importance=100.0,
    )
    unit["source_codec"] = "fp4_e2m1_ue8m0"
    unit["lossless_codecs"] = ["mxfp4"]
    return unit


def _evidence(units):
    return make_artifact("probe_evidence", SUBJECT, PRODUCER, status="valid", units=units)


def _good_layers(n, start=1):
    return [_affine_unit(f"L{i}.q", {2: 0.99}, layer_index=i) for i in range(start, start + n)]


_RAMP = {2: 0.5, 3: 0.7, 4: 0.9, 5: 0.93, 6: 0.97, 8: 0.99}


def _by_name(decision):
    return {a["source_name"]: a for a in decision["allocation"]}


# --- structure ---

def test_decision_is_a_valid_artifact():
    ev = _evidence(_good_layers(4, start=0))
    dec = decide(ev, target_quality=0.5)
    assert validate_base(dec) == []
    assert dec["artifact_kind"] == "optimizer_decision"
    assert dec["feasibility"] == "feasible"
    assert dec["source_probe_id"] == ev["artifact_id"]


def test_decision_preserves_probe_required_features():
    ev = make_artifact(
        "probe_evidence",
        SUBJECT,
        PRODUCER,
        status="valid",
        required_features=["calibration"],
        units=_good_layers(4, start=0),
    )
    dec = decide(ev, target_quality=0.5)

    assert validate_base(dec) == []
    assert dec["required_features"] == ["calibration"]
    assert dec["source_probe_id"] == ev["artifact_id"]


def test_allocation_bits_in_allowed_sets():
    units = _good_layers(2, start=0) + [
        _expert_unit("e0.gate", 0, "gate", {1: 0.9, 2: 0.97, 4: 0.995})]
    dec = decide(_evidence(units), target_quality=0.95)
    for a in dec["allocation"]:
        if a["kind"] == "affine":
            assert a["bits"] in AFFINE_BITS and a["group_size"] in GROUP_SIZES
        elif a["kind"] == "expert":
            assert a["bits"] in EXPERT_BITS
            assert a["codec"] in {"tq", "mxfp4"}


def test_lossless_capable_expert_defaults_four_bit_tier_to_mxfp4():
    units = [_lossless_mxfp4_expert_unit()] + _good_layers(3, start=1)

    dec = decide(_evidence(units), target_quality=0.98)
    expert = _by_name(dec)["expert"]

    assert expert["bits"] == 4
    assert expert["codec"] == "mxfp4"
    assert expert["format"] == "mxfp4"
    assert expert["lossless"] is True
    assert dec["achieved"]["expert_codec_counts"] == {"mxfp4": 1}
    assert any(
        r["choice"] == "TQ4 for lossless-capable routed experts"
        for r in dec["rejected"]
    )


def test_deepseek_v4_hash_routed_source_fp4_experts_start_lossless_four_bit():
    early = _lossless_mxfp4_expert_unit("layers.0.ffn.experts.gate", q4=0.90)
    early["layer_index"] = 0
    late = _lossless_mxfp4_expert_unit("layers.3.ffn.experts.gate", q4=0.90)
    late["layer_index"] = 3
    dec = decide(_evidence([early, late] + _good_layers(3, start=4)), target_quality=0.5)
    by = _by_name(dec)

    assert dec["status"] == "valid"
    assert by["layers.0.ffn.experts.gate"]["bits"] == 4
    assert by["layers.0.ffn.experts.gate"]["codec"] == "mxfp4"
    assert by["layers.0.ffn.experts.gate"]["lossless"] is True
    assert by["layers.3.ffn.experts.gate"]["bits"] == 1


def test_force_tq4_lossless_kill_switch_keeps_four_bit_tier_as_tq():
    units = [_lossless_mxfp4_expert_unit(q4=0.99)] + _good_layers(3, start=1)

    dec = decide(_evidence(units), target_quality=0.98, force_tq4_lossless=True)
    expert = _by_name(dec)["expert"]

    assert expert["bits"] == 4
    assert expert["codec"] == "tq"
    assert expert["format"] == "tq"
    assert expert["lossless"] is False
    assert dec["constraints"]["force_tq4_lossless"] is True
    assert dec["achieved"]["expert_codec_counts"] == {"tq": 1}
    assert any(
        r["choice"] == "source mxfp4 for lossless-capable routed experts"
        for r in dec["rejected"]
    )


def test_min_routed_expert_bits_excludes_tq1_without_forcing_lossless_tier():
    unit = _expert_unit(
        "layers.7.ffn.experts.down",
        7,
        "down",
        {1: 0.10, 2: 0.20, 4: 0.90},
    )

    dec = decide(
        _evidence([unit] + _good_layers(3, start=8)),
        target_quality=0.05,
        min_routed_expert_bits=2,
    )
    expert = _by_name(dec)["layers.7.ffn.experts.down"]

    assert expert["bits"] == 2
    assert expert["codec"] == "tq"
    assert dec["constraints"]["min_routed_expert_bits"] == 2
    assert any(r["choice"] == "TQ1 for routed experts" for r in dec["rejected"])


def test_force_dense_lossless_mx_keeps_fp8_dense_at_mxfp8_floor():
    unit = _affine_unit(
        "layers.0.attn.wq.weight",
        {2: 0.50, 3: 0.55, 4: 0.60, 5: 0.65, 6: 0.70, 8: 0.75},
        role="attn.q_proj",
        layer_index=0,
    )
    unit["source_codec"] = "fp8_e4m3_ue8m0"
    unit["lossless_codecs"] = ["mxfp8"]
    unit["dense_codec_quality"] = {"mxfp8_8_32": 1.0}

    unforced = decide(
        _evidence([unit]),
        target_quality=0.5,
        force_dense_lossless_mx=False,
    )
    assert _by_name(unforced)["layers.0.attn.wq.weight"].get("format", "affine") == "affine"

    dec = decide(
        _evidence([unit]),
        target_quality=0.5,
        force_dense_lossless_mx=True,
    )
    alloc = _by_name(dec)["layers.0.attn.wq.weight"]

    assert alloc["format"] == "mxfp8"
    assert alloc["bits"] == 8
    assert alloc["group_size"] == 32
    assert alloc["lossless"] is True
    assert dec["constraints"]["force_dense_lossless_mx"] is True
    assert dec["achieved"]["tensor_format_counts"]["mxfp8"] == 1
    assert any(
        r["choice"].startswith("lower dense FP8/e4m3 roles below mxfp8")
        for r in dec["rejected"]
    )


# --- objective + rejected (spec optimizer_decision fields) ---

def test_objective_is_named_and_reflects_constraints():
    ev = _evidence(_good_layers(4, start=0))
    dec = decide(ev, target_quality=0.9, tau=0.8, alpha=0.05)
    obj = dec["objective"]
    assert "fidelity" in obj                       # the goal is always fidelity
    assert "0.9" in obj and "0.8" in obj           # both constraints named
    # with no constraints, the objective is just the goal (no 's.t.')
    bare = decide(ev, target_size_gb=1e9)["objective"]
    assert "size_gb" in bare


def test_target_size_objective_is_honest_fill_toward_not_budget():
    # target_size_gb is fill-toward (greedy stops when size >= target, landing at
    # the target). It does not impose a strict `<=` budget, so the objective must omit `<=`,
    # which it incorrectly did. The artifact would misdescribe the contract.
    ev = _evidence(_good_layers(4, start=0))
    obj = decide(ev, target_size_gb=1.0)["objective"]
    assert "size_gb" in obj
    assert "<=" not in obj                 # the inaccurate `<=` claim is gone
    assert "fill-toward" in obj


def test_budget_split_requires_size_target_and_supported_objective():
    ev = _evidence(_good_layers(4, start=0))

    with pytest.raises(ValueError, match="requires target_size_gb"):
        decide(ev, target_quality=0.5, budget_split={"experts": 0.5, "affine": 0.5})
    with pytest.raises(ValueError, match="does not support target_quality"):
        decide(
            ev,
            target_quality=0.5,
            target_size_gb=1.0,
            budget_split={"experts": 0.5, "affine": 0.5},
        )
    with pytest.raises(ValueError, match="does not support tau"):
        decide(ev, target_size_gb=1.0, tau=0.5,
               budget_split={"experts": 0.5, "affine": 0.5})
    with pytest.raises(ValueError, match="exactly"):
        decide(ev, target_size_gb=1.0, budget_split={"expert": 0.5, "affine": 0.5})
    with pytest.raises(ValueError, match="sum to 1.0"):
        decide(ev, target_size_gb=1.0, budget_split={"experts": 0.6, "affine": 0.6})


def test_budget_split_is_absent_from_default_joint_allocator():
    ev = _evidence(_good_layers(4, start=0))

    dec = decide(ev, target_size_gb=1.0)

    assert "budget_split" not in dec["constraints"]
    assert "budget_split" not in dec["achieved"]
    assert "strict expert/affine" not in dec["objective"]


def test_budget_split_does_not_let_experts_spend_affine_budget():
    units = [
        _expert_unit("expert", 0, "gate", {1: 0.10, 2: 0.90, 4: 0.99},
                     importance=100.0),
        _affine_unit("affine", _RAMP, layer_index=0, importance=1.0),
    ] + _good_layers(3, start=1)

    dec = decide(
        _evidence(units),
        target_size_gb=1.0,
        budget_split={"experts": 0.0, "affine": 1.0},
    )

    by = _by_name(dec)
    assert by["expert"]["bits"] == 1
    assert by["affine"]["bits"] > 2
    assert dec["constraints"]["budget_split"] == {"experts": 0.0, "affine": 1.0}
    assert dec["achieved"]["budget_split"]["expert_spent_bytes"] == 0
    assert dec["achieved"]["budget_split"]["affine_spent_bytes"] > 0
    assert "strict expert/affine budget split" in dec["objective"]


def test_budget_split_does_not_let_affine_spend_expert_budget():
    units = [
        _expert_unit("expert", 0, "gate", {1: 0.10, 2: 0.90, 4: 0.99},
                     importance=1.0),
        _affine_unit("affine", _RAMP, layer_index=0, importance=100.0),
    ] + _good_layers(3, start=1)

    dec = decide(
        _evidence(units),
        target_size_gb=1.0,
        budget_split={"experts": 1.0, "affine": 0.0},
    )

    by = _by_name(dec)
    assert by["expert"]["bits"] > 1
    assert by["affine"]["bits"] == 2
    assert dec["achieved"]["budget_split"]["expert_spent_bytes"] > 0
    assert dec["achieved"]["budget_split"]["affine_spent_bytes"] == 0


def test_rejected_records_fp16_passthrough_choice():
    # a router gate -> kept fp16; the decision should record rejecting its quant.
    gate = _affine_unit("router", {2: 0.4}, role="moe.router_gate", layer_index=0)
    dec = decide(_evidence([gate] + _good_layers(3)), target_quality=0.5)
    reasons = [r for r in dec["rejected"] if "moe.router_gate" in r["choice"]]
    assert reasons and "fp16" in reasons[0]["reason"]


def test_rejected_is_honest_when_nothing_was_rejected():
    # all-affine, no tau, no fp16 roles -> nothing genuinely rejected.
    dec = decide(_evidence(_good_layers(3, start=0)), target_quality=0.5)
    assert dec["rejected"] == []


def test_infeasible_decision_still_carries_objective_and_rejected():
    bad = _affine_unit("bad", _RAMP, layer_index=0)
    dec = decide(_evidence([bad] + _good_layers(19)), target_quality=0.5,
                 tau=0.999, alpha=0.05)
    assert dec["feasibility"] == "infeasible"
    assert "fidelity" in dec["objective"]
    assert isinstance(dec["rejected"], list)  # present even on the invalid artifact


# --- tail constraint ---

def test_feasibility_lifts_worst_layer_to_tau():
    bad = _affine_unit("bad", _RAMP, layer_index=0)
    dec = decide(_evidence([bad] + _good_layers(19)), target_quality=0.5, tau=0.9, alpha=0.05)
    assert _by_name(dec)["bad"]["bits"] == 4          # first bit where q>=0.9
    assert dec["achieved"]["worst_layer_tail"] >= 0.9 - 1e-9
    assert dec["feasibility"] == "feasible"


def test_bundled_tied_minima_do_not_stall():
    t1 = _affine_unit("t1", _RAMP, role="attn.q_proj", layer_index=0)
    t2 = _affine_unit("t2", _RAMP, role="attn.k_proj", layer_index=0)
    dec = decide(_evidence([t1, t2] + _good_layers(19)), target_quality=0.5, tau=0.9, alpha=0.05)
    by = _by_name(dec)
    assert by["t1"]["bits"] == 4 and by["t2"]["bits"] == 4


def test_infeasible_when_tau_above_max_is_recorded():
    bad = _affine_unit("bad", _RAMP, layer_index=0)  # max q = 0.99
    dec = decide(_evidence([bad] + _good_layers(19)), target_quality=0.5, tau=0.999, alpha=0.05)
    assert dec["feasibility"] == "infeasible"
    assert dec["status"] == "invalid"
    assert dec["allocation"] == [] and dec["achieved"] is None
    assert any(v["code"] == "optimize.infeasible" for v in dec["validation"])


def test_infeasible_under_budget_is_recorded():
    bad = _affine_unit("bad", _RAMP, layer_index=0)
    dec = decide(_evidence([bad] + _good_layers(19)), target_size_gb=1e-9, tau=0.9, alpha=0.05)
    assert dec["feasibility"] == "infeasible_under_budget"
    assert any(v["code"] == "optimize.infeasible_under_budget" for v in dec["validation"])


# --- floors ---

def test_lm_head_and_embed_floored_regardless_of_tau():
    # Serve-critical safety floors: lm_head and embed_tokens are single global tensors
    # (each its own 1-unit "layer"), so the CVaR worst-layer tail cannot protect them.
    # A quality/tau recipe can otherwise strand them at the 2-bit floor (the collapse
    # signature where the worst-layer tail strands lm_head/embed at the bit floor). These
    # floors hold with and without tau, separate from the optimization objective.
    head = _affine_unit("head", {2: 0.99}, role="lm_head", layer_index=None)
    embed = _affine_unit("embed", {2: 0.99}, role="embed_tokens", layer_index=None)
    units = [head, embed] + _good_layers(5)
    for kwargs in ({"target_quality": 0.5},
                   {"target_quality": 0.5, "tau": 0.0},   # tau set -> floors used to drop
                   {"target_quality": 0.5, "tau": 0.9}):
        by = _by_name(decide(_evidence(units), **kwargs))
        assert by["head"]["bits"] >= 6, kwargs    # lm_head floor (_LM_HEAD_DEFAULT_BITS)
        assert by["embed"]["bits"] >= 4, kwargs   # embed_tokens floor


def test_backbone_has_no_per_role_floor():
    # The health gate protects the backbone. Broad per-role floors recreate the
    # reverted uniform-high mistake. Only lm_head/embed get
    # hard floors, so a backbone tensor whose quality is already met sits at the
    # global bit minimum with and without tau.
    o = _affine_unit("o", {2: 0.99}, role="attn.o_proj", layer_index=0)
    units = [o] + _good_layers(5)
    assert _by_name(decide(_evidence(units), target_quality=0.5))["o"]["bits"] == 2
    assert _by_name(decide(_evidence(units), target_quality=0.5, tau=0.0))["o"]["bits"] == 2


def test_router_gate_is_fp16_passthrough():
    gate = _affine_unit("router", {2: 0.4}, role="moe.router_gate", layer_index=0)
    dec = decide(_evidence([gate] + _good_layers(3)), target_quality=0.5)
    router = _by_name(dec)["router"]
    assert router["kind"] == "fp16_passthrough"
    assert "bits" not in router


def test_deepseek_v4_optimizer_ownership_roles_are_pinned():
    units = [
        _expert_unit("layers.2.ffn.experts.gate", 2, "gate", {1: 0.80, 2: 0.95, 4: 0.99}),
        _affine_unit("layers.2.attn.indexer.wq_b.weight", _RAMP,
                     role="attn.indexer.wq_b", layer_index=2),
        _affine_unit("layers.2.attn.compressor.wkv.weight", _RAMP,
                     role="attn.compressor.wkv", layer_index=2),
        _affine_unit("embed", {2: 0.90, 4: 0.97}, role="embed_tokens",
                     layer_index=None),
        _affine_unit("head", {2: 0.90, 4: 0.97, 6: 0.99}, role="lm_head",
                     layer_index=None),
        _affine_unit("router", {2: 0.4}, role="moe.router_gate", layer_index=2),
    ] + _good_layers(3, start=3)

    dec = decide(_evidence(units), target_quality=0.5)
    by = _by_name(dec)

    assert by["layers.2.ffn.experts.gate"]["kind"] == "expert"
    assert by["layers.2.attn.indexer.wq_b.weight"]["kind"] == "affine"
    assert by["layers.2.attn.compressor.wkv.weight"]["kind"] == "affine"
    assert by["router"]["kind"] == "fp16_passthrough"
    assert "bits" not in by["router"]
    assert by["embed"]["bits"] >= 4
    assert by["head"]["bits"] >= 6


def test_moe_affine_profile_keeps_routers_fp16_and_profiles_shared_experts():
    profile = qwen35_moe_affine_role_profile_v1()
    units = [
        _expert_unit("experts.gate", 0, "gate", {1: 0.80, 2: 0.90, 4: 0.99}),
        _affine_unit("router", {2: 0.4}, role="moe.router_gate", layer_index=0),
        _affine_unit("shared_router", {2: 0.4}, role="moe.shared_expert_gate", layer_index=0),
        _affine_unit("shared_down", {2: 0.90, 4: 0.95},
                     role="moe.shared_expert.down_proj", layer_index=0),
        _affine_unit("ssm_a", {2: 0.90, 4: 0.95}, role="ssm.in_proj_a", layer_index=0),
    ] + _good_layers(3, start=1)

    dec = decide(
        _evidence(units),
        target_quality=0.5,
        affine_role_profile_name=profile["name"],
        affine_role_weights=profile["affine_role_weights"],
        affine_role_bit_weights=profile["affine_role_bit_weights"],
        affine_role_min_bits=profile["affine_role_min_bits"],
    )

    by = _by_name(dec)
    assert dec["constraints"]["affine_role_profile_name"] == profile["name"]
    assert by["router"]["kind"] == "fp16_passthrough"
    assert by["shared_router"]["kind"] == "fp16_passthrough"
    assert by["experts.gate"]["kind"] == "expert"
    assert by["shared_down"]["kind"] == "affine"
    assert by["shared_down"]["bits"] >= 4
    assert by["ssm_a"]["kind"] == "affine"
    assert by["ssm_a"]["bits"] >= 4
    assert "moe.shared_expert.down_proj" in dec["constraints"]["affine_role_min_bits"]
    assert "moe.router_gate" not in dec["constraints"]["affine_role_min_bits"]


def test_ssm_in_proj_a_b_are_quantized_not_fp16():
    # in_proj_a / in_proj_b are ordinary SSM projection weights. The proven
    # converter affine-quantizes them (2D weight -> affine),
    # and the runtime serves coherently with them quantized. Keeping them fp16 (listing
    # them in FP16_ROLES) diverges from the runtime and produces garbage. They must be
    # affine.
    a = _affine_unit("ipa", {2: 0.9, 4: 0.97}, role="ssm.in_proj_a", layer_index=0)
    b = _affine_unit("ipb", {2: 0.9, 4: 0.97}, role="ssm.in_proj_b", layer_index=0)
    dec = decide(_evidence([a, b] + _good_layers(3)), target_quality=0.5)
    by = _by_name(dec)
    for name in ("ipa", "ipb"):
        assert by[name]["kind"] == "affine", f"{name} must use affine storage"
        assert "bits" in by[name]
    # only the genuine routing gates stay fp16; SSM in_proj must not be rejected.
    assert not any("ssm.in_proj" in r["choice"] for r in dec["rejected"])


def test_dense_ffn_is_quantized_not_special_cased():
    # Second-pressure (dense family): an FFN backbone tensor must be allocated
    # like any affine weight: never fp16-passthrough, never floored as a gate.
    ffn = _affine_unit("dense.down", {2: 0.9, 4: 0.97}, role="ffn.down_proj", layer_index=0)
    dec = decide(_evidence([ffn] + _good_layers(3)), target_quality=0.5)
    alloc = _by_name(dec)["dense.down"]
    assert alloc["kind"] == "affine"
    assert "bits" in alloc
    assert not any("ffn." in r["choice"] for r in dec["rejected"])


def test_missing_affine_role_weights_preserves_default_allocation_order():
    q = {2: 0.80, 4: 0.90}
    elastic = _affine_unit("elastic", q, role="ssm.in_proj_a", layer_index=0,
                           importance=1.0)
    fragile = _affine_unit("fragile", q, role="ffn.down_proj", layer_index=1,
                           importance=1.0)
    start = 2 * affine_bytes(64, 128, 2, 128)
    one_upgrade = affine_bytes(64, 128, 4, 128) - affine_bytes(64, 128, 2, 128)

    dec = decide(
        _evidence([elastic, fragile]),
        target_size_gb=(start + one_upgrade) / (1024 ** 3),
        allow_unhealthy=True,
    )

    by = _by_name(dec)
    assert by["elastic"]["bits"] == 4
    assert by["fragile"]["bits"] == 2
    assert "affine_role_weights" not in dec["constraints"]


def test_affine_role_weights_record_and_change_allocation_order():
    q = {2: 0.80, 4: 0.90}
    elastic = _affine_unit("elastic", q, role="ssm.in_proj_a", layer_index=0,
                           importance=1.0)
    fragile = _affine_unit("fragile", q, role="ffn.down_proj", layer_index=1,
                           importance=1.0)
    start = 2 * affine_bytes(64, 128, 2, 128)
    one_upgrade = affine_bytes(64, 128, 4, 128) - affine_bytes(64, 128, 2, 128)

    dec = decide(
        _evidence([elastic, fragile]),
        target_size_gb=(start + one_upgrade) / (1024 ** 3),
        allow_unhealthy=True,
        affine_role_weights={"ffn.down_proj": 10.0},
    )

    by = _by_name(dec)
    assert by["elastic"]["bits"] == 2
    assert by["fragile"]["bits"] == 4
    assert dec["constraints"]["affine_role_weights"] == {"ffn.down_proj": 10.0}
    assert "role-adjusted affine risk" in dec["objective"]


def test_affine_role_weights_reject_non_positive_values():
    ev = _evidence(_good_layers(3, start=0))

    with pytest.raises(ValueError, match="must be > 0"):
        decide(ev, target_quality=0.5, affine_role_weights={"ffn.down_proj": 0.0})


def test_affine_role_bit_weights_record_and_change_destination():
    unit = _affine_unit_grid("banded", {
        (2, 128): 0.80,
        (4, 128): 0.81,
        (8, 128): 0.95,
    }, role="ssm.in_proj_a", layer_index=0)
    start = affine_bytes(64, 128, 2, 128)
    one_upgrade = affine_bytes(64, 128, 4, 128) - affine_bytes(64, 128, 2, 128)

    dec = decide(
        _evidence([unit]),
        target_size_gb=(start + one_upgrade) / (1024 ** 3),
        allow_unhealthy=True,
        affine_role_bit_weights={"ssm.in_proj_a": {4: 100.0, 8: 0.01}},
    )

    alloc = _by_name(dec)["banded"]
    assert alloc["bits"] == 4
    assert dec["constraints"]["affine_role_bit_weights"] == {
        "ssm.in_proj_a": {"4": 100.0, "8": 0.01},
    }
    assert "role-adjusted affine risk" in dec["objective"]


def test_affine_role_bit_weights_reject_bad_values():
    ev = _evidence(_good_layers(3, start=0))

    with pytest.raises(ValueError, match="unsupported bit"):
        decide(ev, target_quality=0.5, affine_role_bit_weights={"ffn.down_proj": {7: 1.0}})
    with pytest.raises(ValueError, match="must be > 0"):
        decide(ev, target_quality=0.5, affine_role_bit_weights={"ffn.down_proj": {4: 0.0}})


def test_affine_role_min_bits_record_and_raise_floor():
    q = {2: 0.80, 4: 0.90, 8: 0.95}
    unit = _affine_unit("floor", q, role="ssm.in_proj_a", layer_index=0)
    start = affine_bytes(64, 128, 2, 128)

    dec = decide(
        _evidence([unit]),
        target_size_gb=start / (1024 ** 3),
        allow_unhealthy=True,
        affine_role_min_bits={"ssm.in_proj_a": 4},
    )

    alloc = _by_name(dec)["floor"]
    assert alloc["bits"] == 4
    assert dec["constraints"]["affine_role_min_bits"] == {"ssm.in_proj_a": 4}
    assert "role-adjusted affine risk" in dec["objective"]


def test_affine_role_min_bits_reject_bad_bits():
    ev = _evidence(_good_layers(3, start=0))

    with pytest.raises(ValueError, match="must be one of"):
        decide(ev, target_quality=0.5, affine_role_min_bits={"ffn.down_proj": 7})


def test_monotone_envelope_applied_through_decide():
    # The quality table the optimizer consumes is the monotone envelope,
    # not the raw measured q. A unit with a sampling-noise dip (4-bit measured below
    # 2-bit) must be lifted so q never decreases as bits rise, proving decide()'s table
    # parser applies the envelope (not just the pure function in isolation).
    from moespresso.optimize.decide import _quality_table_affine
    dip_unit = _affine_unit("dip", {2: 0.90, 4: 0.80}, role="attn.q_proj", layer_index=0)
    table = _quality_table_affine(dip_unit)
    # raw 4-bit was 0.80 < 0.90; enveloped it must be >= the 2-bit value, per gs.
    for gs in (128, 64, 32):
        assert table[(4, gs)] >= table[(2, gs)]
        assert table[(4, gs)] >= 0.90 - 1e-9


def test_dense_mx_monotone_envelope_applied_through_decide():
    from moespresso.optimize.decide import _quality_table_dense_mx

    unit = _affine_unit(
        "mx",
        {2: 0.50},
        role="attn.q_proj",
        layer_index=0,
    )
    unit["dense_codec_quality"] = {
        "mxfp4_4_32": 0.93,
        "mxfp8_8_32": 0.91,
    }
    table = _quality_table_dense_mx(unit)

    assert table[("mxfp4", 4, 32)] == 0.93
    assert table[("mxfp8", 8, 32)] == 0.93


def test_large_q_inversion_is_reported_not_silently_smoothed():
    # Reporting half: the envelope smooths for allocation, but a genuine
    # inversion beyond the noise band (here 0.90 -> 0.55 at a higher bit) must surface as
    # a non-blocking warning so it can be re-measured. A clean table does not
    # warn.
    bad = _affine_unit("bad", {2: 0.90, 4: 0.55}, role="attn.q_proj", layer_index=0)
    dec = decide(_evidence([bad] + _good_layers(3)), target_quality=0.5)
    inv = [v for v in dec["validation"] if v["code"] == "optimize.q_inversion"]
    assert inv and all(v["severity"] == "warning" and not v["blocking"] for v in inv)
    assert dec["status"] != "invalid"   # warning only, does not block

    clean = decide(_evidence(_good_layers(4, start=0)), target_quality=0.5)
    assert not any(v["code"] == "optimize.q_inversion" for v in clean["validation"])


def test_optimizer_takes_a_diagonal_bits_gs_move():
    # The (bits, gs) frontier fix: a tensor whose quality is flat-low along each single
    # axis from (2,128) (more bits at gs128 doesn't help, finer gs at 2b doesn't help)
    # but the diagonal (4,32) jumps to 0.99. An axis-only search could never reach
    # it; the frontier search must. Drive quality up so the greedy spends to claim it.
    diag = _affine_unit_grid("diag", {
        (2, 128): 0.80, (3, 128): 0.80, (4, 128): 0.80,   # bits alone @gs128: useless
        (2, 64): 0.80, (2, 32): 0.80,                      # gs alone @2b: useless
        (4, 32): 0.99,                                     # only the diagonal pays
    }, layer_index=0)
    dec = decide(_evidence([diag] + _good_layers(3)), target_quality=0.99,
                 allow_unhealthy=True)
    alloc = _by_name(dec)["diag"]
    assert (alloc["bits"], alloc["group_size"]) == (4, 32)   # reached the diagonal tuple


# --- health gate ---

def test_collapsed_allocation_is_rejected_invalid():
    # A backbone that collapses to the bit floor (the garbage signature: a quality
    # target met at 2-bit over a real-sized backbone) must yield status=invalid with a
    # blocking health validation, so convert can't silently ship a garbage package.
    units = [_affine_unit(f"L{i}.q", {2: 0.99}, layer_index=i) for i in range(20)]
    dec = decide(_evidence(units), target_quality=0.5)
    assert all(a["bits"] <= 3 for a in dec["allocation"] if a["kind"] == "affine")
    assert dec["status"] == "invalid"
    assert any(v["code"] == "optimize.collapsed_backbone" and v["blocking"]
               for v in dec["validation"])


def test_allow_unhealthy_keeps_collapsed_decision_valid_but_warns():
    # The explicit escape hatch: same collapse, but allow_unhealthy downgrades the
    # findings to non-blocking warnings and keeps the decision valid (still recorded).
    units = [_affine_unit(f"L{i}.q", {2: 0.99}, layer_index=i) for i in range(20)]
    dec = decide(_evidence(units), target_quality=0.5, allow_unhealthy=True)
    assert dec["status"] == "valid"
    warns = [v for v in dec["validation"] if v["code"] == "optimize.collapsed_backbone"]
    assert warns and all(not v["blocking"] and v["severity"] == "warning" for v in warns)
    assert any(v["code"] == "optimize.health_overridden" for v in dec["validation"])


def test_healthy_allocation_passes_gate_valid():
    # A high-bit backbone (target pushes bits up) is serve-viable -> valid, no health
    # findings. Guards against the gate false-positiving on the proven recipe.
    units = [_affine_unit(f"L{i}.q", _RAMP, layer_index=i) for i in range(20)]
    dec = decide(_evidence(units), target_quality=0.99)
    assert dec["status"] == "valid"
    assert not any(v["code"].startswith("optimize.collapsed") for v in dec["validation"])


# --- achieved summary ---

def test_summary_reports_fidelity_and_tail():
    dec = decide(_evidence(_good_layers(4, start=0)), target_quality=0.5)
    ach = dec["achieved"]
    assert ach["fidelity"] == pytest.approx(0.99, abs=1e-6)
    assert ach["worst_layer_tail"] == pytest.approx(0.99, abs=1e-6)


# --- determinism + monotonicity ---

def test_decision_is_deterministic():
    units = _good_layers(3, start=0) + [
        _expert_unit("e0.gate", 0, "gate", {1: 0.9, 2: 0.97, 4: 0.995})]
    a = decide(_evidence(units), target_quality=0.95, tau=0.8)
    b = decide(_evidence(units), target_quality=0.95, tau=0.8)
    assert a["artifact_id"] == b["artifact_id"] == compute_artifact_id(a)


def test_higher_quality_never_lowers_bits():
    units = [_affine_unit(f"L{i}.q", {2: 0.85, 4: 0.92, 8: 0.99}, layer_index=i)
             for i in range(3)]
    low = decide(_evidence(units), target_quality=0.85)
    high = decide(_evidence(units), target_quality=0.99)

    def avg(dec):
        bits = [a["bits"] for a in dec["allocation"] if a["kind"] == "affine"]
        return sum(bits) / max(len(bits), 1)

    assert avg(high) >= avg(low)


# --- expert importance cross-projection normalization ---

def _norm_units():
    """down has a slightly better quality curve than gate/up but 20x smaller
    raw importance (the post-SiLU input-magnitude artifact). A fair
    auction must rank down's upgrade first; the raw scalars bury it."""
    ramp = {1: 0.5, 2: 0.9, 4: 0.99}
    down_ramp = {1: 0.5, 2: 0.92, 4: 0.99}
    return _good_layers(2, start=10) + [
        _expert_unit("e0.gate", 0, "gate", ramp, importance=1.0),
        _expert_unit("e0.up", 0, "up", ramp, importance=1.0),
        _expert_unit("e0.down", 0, "down", down_ramp, importance=0.05),
    ]


def test_expert_importance_norm_default_unfreezes_starved_class():
    ev = _evidence(_norm_units())
    raw = decide(ev, target_quality=0.8, expert_importance_norm="off")
    normed = decide(ev, target_quality=0.8)  # class-mean is the default
    raw_bits = {a["source_name"]: a["bits"] for a in raw["allocation"]
                if a["kind"] == "expert"}
    norm_bits = {a["source_name"]: a["bits"] for a in normed["allocation"]
                 if a["kind"] == "expert"}
    # raw: down loses the auction despite the better quality curve
    assert raw_bits["e0.down"] < raw_bits["e0.gate"]
    # normalized: the better curve wins the auction
    assert norm_bits["e0.down"] > norm_bits["e0.gate"]
    # audit trail records the applied scales
    rec = normed["constraints"]["expert_importance_norm"]
    assert rec["mode"] == "class-mean"
    assert rec["scales"]["down"] > 1.0 > rec["scales"]["gate"]
    assert "expert_importance_norm" not in raw["constraints"]


def test_expert_importance_norm_noop_without_experts():
    ev = _evidence(_good_layers(3, start=0))
    dec = decide(ev, target_quality=0.5)
    assert "expert_importance_norm" not in dec["constraints"]
    assert dec["feasibility"] == "feasible"


def test_expert_importance_norm_validates_mode():
    ev = _evidence(_norm_units())
    with pytest.raises(ValueError, match="expert_importance_norm"):
        decide(ev, target_quality=0.8, expert_importance_norm="bogus")
