from moespresso.optimize.affine_elasticity import (
    DEEPSEEK_V4_AFFINE_ROLE_PROFILE_V1_NAME,
    PROMPTS,
    QWEN35_AFFINE_ROLE_BIT_WEIGHTS_V1,
    QWEN35_AFFINE_ROLE_MIN_BITS_V1,
    QWEN35_AFFINE_ROLE_PROFILE_V1_NAME,
    QWEN35_AFFINE_ROLE_WEIGHTS_V1,
    QWEN35_AFFINE_ROLE_WEIGHTS_V2,
    QWEN35_MOE_AFFINE_ROLE_PROFILE_V1_NAME,
    affine_role_profile_for_family,
    check_hits,
    extract_final_line,
    q8_relative_summary,
    qwen35_affine_role_bit_weights_v1,
    qwen35_affine_role_min_bits_v1,
    qwen35_moe_affine_role_bit_weights_v1,
    qwen35_moe_affine_role_min_bits_v1,
    qwen35_moe_affine_role_profile_v1,
    qwen35_moe_affine_role_weights_v1,
    qwen35_affine_role_weights_v1,
    qwen35_affine_role_weights_v2,
    deepseek_v4_affine_role_profile_v1,
    repetition_max_trigram,
    role_bit_summary,
    score_generation,
    summarize_generation,
)


def test_qwen35_affine_role_weights_v1_are_positive_and_copied():
    weights = qwen35_affine_role_weights_v1()

    assert weights == QWEN35_AFFINE_ROLE_WEIGHTS_V1
    assert all(value > 0 for value in weights.values())
    assert weights["ffn.down_proj"] > weights["ffn.gate_proj"]
    assert weights["ssm.in_proj_a"] < weights["ssm.in_proj_qkv"]
    weights["ffn.down_proj"] = 1.0
    assert QWEN35_AFFINE_ROLE_WEIGHTS_V1["ffn.down_proj"] == 128.0


def test_qwen35_affine_role_band_profile_is_positive_and_copied():
    weights = qwen35_affine_role_weights_v2()
    bit_weights = qwen35_affine_role_bit_weights_v1()
    min_bits = qwen35_affine_role_min_bits_v1()

    assert weights == QWEN35_AFFINE_ROLE_WEIGHTS_V2
    assert bit_weights == QWEN35_AFFINE_ROLE_BIT_WEIGHTS_V1
    assert min_bits == QWEN35_AFFINE_ROLE_MIN_BITS_V1
    assert all(value > 0 for value in weights.values())
    assert all(value > 0 for role in bit_weights.values() for value in role.values())
    assert set(min_bits.values()) == {4}
    assert bit_weights["ssm.in_proj_a"][4] > bit_weights["ssm.in_proj_a"][8]
    assert bit_weights["ffn.gate_proj"][8] > bit_weights["ffn.gate_proj"][4]
    assert bit_weights["attn.q_proj"][5] > bit_weights["attn.q_proj"][6]

    weights["ssm.in_proj_a"] = 1.0
    bit_weights["ssm.in_proj_a"][4] = 1.0
    min_bits["ssm.in_proj_a"] = 2

    assert QWEN35_AFFINE_ROLE_WEIGHTS_V2["ssm.in_proj_a"] == 0.05
    assert QWEN35_AFFINE_ROLE_BIT_WEIGHTS_V1["ssm.in_proj_a"][4] == 50.0
    assert QWEN35_AFFINE_ROLE_MIN_BITS_V1["ssm.in_proj_a"] == 4


def test_moe_affine_role_profile_aliases_shared_experts_and_excludes_routers():
    weights = qwen35_moe_affine_role_weights_v1()
    bit_weights = qwen35_moe_affine_role_bit_weights_v1()
    min_bits = qwen35_moe_affine_role_min_bits_v1()

    assert weights["moe.shared_expert.gate_proj"] == weights["ffn.gate_proj"]
    assert weights["moe.shared_expert.up_proj"] == weights["ffn.up_proj"]
    assert weights["moe.shared_expert.down_proj"] == weights["ffn.down_proj"]
    assert bit_weights["moe.shared_expert.gate_proj"] == bit_weights["ffn.gate_proj"]
    assert bit_weights["moe.shared_expert.up_proj"] == bit_weights["ffn.up_proj"]
    assert bit_weights["moe.shared_expert.down_proj"] == bit_weights["ffn.down_proj"]
    assert min_bits["moe.shared_expert.gate_proj"] == min_bits["ffn.gate_proj"]
    assert min_bits["moe.shared_expert.up_proj"] == min_bits["ffn.up_proj"]
    assert min_bits["moe.shared_expert.down_proj"] == min_bits["ffn.down_proj"]

    for role in (
        "moe.router_gate",
        "moe.shared_expert_gate",
        "moe.expert.gate",
        "moe.expert.up",
        "moe.expert.down",
    ):
        assert role not in weights
        assert role not in bit_weights
        assert role not in min_bits

    weights["moe.shared_expert.down_proj"] = 1.0
    bit_weights["moe.shared_expert.down_proj"][4] = 1.0
    min_bits["moe.shared_expert.down_proj"] = 2

    fresh = qwen35_moe_affine_role_profile_v1()
    assert fresh["affine_role_weights"]["moe.shared_expert.down_proj"] == 128.0
    assert fresh["affine_role_bit_weights"]["moe.shared_expert.down_proj"][4] == 20.0
    assert fresh["affine_role_min_bits"]["moe.shared_expert.down_proj"] == 4


def test_affine_role_profile_defaults_for_supported_qwen35_families():
    profile = affine_role_profile_for_family("qwen3_5_dense")
    moe_profile = affine_role_profile_for_family("qwen3_5_moe")

    assert profile["name"] == QWEN35_AFFINE_ROLE_PROFILE_V1_NAME
    assert profile["affine_role_weights"] == QWEN35_AFFINE_ROLE_WEIGHTS_V2
    assert profile["affine_role_bit_weights"] == QWEN35_AFFINE_ROLE_BIT_WEIGHTS_V1
    assert profile["affine_role_min_bits"] == QWEN35_AFFINE_ROLE_MIN_BITS_V1
    assert moe_profile["name"] == QWEN35_MOE_AFFINE_ROLE_PROFILE_V1_NAME
    assert moe_profile["affine_role_weights"]["attn.q_proj"] == profile["affine_role_weights"]["attn.q_proj"]
    assert moe_profile["affine_role_weights"]["moe.shared_expert.down_proj"] == profile["affine_role_weights"]["ffn.down_proj"]
    assert affine_role_profile_for_family(None) == {}


def test_deepseek_v4_affine_profile_is_dense_conservative_and_excludes_experts():
    profile = deepseek_v4_affine_role_profile_v1()

    assert profile["name"] == DEEPSEEK_V4_AFFINE_ROLE_PROFILE_V1_NAME
    assert set(profile["affine_role_min_bits"].values()) == {6}
    assert profile["affine_role_bit_weights"]["attn.wo_b"][6] > (
        profile["affine_role_bit_weights"]["attn.wo_b"][5])
    assert profile["affine_role_bit_weights"]["attn.wq_a"][6] > (
        profile["affine_role_bit_weights"]["attn.wq_a"][5])
    for role in (
        "moe.router_gate",
        "moe.expert.gate",
        "moe.expert.up",
        "moe.expert.down",
    ):
        assert role not in profile["affine_role_weights"]
        assert role not in profile["affine_role_bit_weights"]
        assert role not in profile["affine_role_min_bits"]

    assert (
        affine_role_profile_for_family("deepseek_v4_flash")["name"]
        == DEEPSEEK_V4_AFFINE_ROLE_PROFILE_V1_NAME
    )


def test_generation_scoring_records_checks_final_repetition_and_tokens():
    prompt = PROMPTS[0]
    text = "thinking words repeat repeat repeat\nFINAL: 410"

    row = score_generation(prompt, text, finish_reason="stop", completion_tokens=12)

    assert row["id"] == "math_rates"
    assert row["check_hits"] == 2
    assert row["check_total"] == 2
    assert row["final"] == "FINAL: 410"
    assert row["completion_tokens"] == 12
    assert row["repetition_max_trigram"] >= 1


def test_repetition_score_counts_repeated_trigrams():
    text = "a b c a b c a b c done"

    assert repetition_max_trigram(text) == 3


def test_extract_final_line_is_case_insensitive_and_first_match():
    text = "notes\n final: first\nFINAL: second"

    assert extract_final_line(text) == "final: first"


def test_check_hits_is_case_insensitive():
    assert check_hits("Seen.Add(X) is the fix", ["seen.add(x)", "missing"]) == 1


def test_generation_summary_counts_lengths_and_stops():
    rows = [
        {"id": "a", "check_hits": 2, "check_total": 3, "final": "FINAL: x",
         "finish_reason": "stop", "completion_tokens": 10, "repetition_max_trigram": 2},
        {"id": "b", "check_hits": 1, "check_total": 2, "final": "",
         "finish_reason": "length", "completion_tokens": 20, "repetition_max_trigram": 5},
    ]

    summary = summarize_generation(rows)

    assert summary == {
        "prompt_count": 2,
        "checks_hit": 3,
        "checks_total": 5,
        "final_lines": 1,
        "stops": 1,
        "length_finishes": 1,
        "max_repeated_trigram": 5,
        "completion_tokens": [10, 20],
    }


def test_q8_relative_summary_compares_by_prompt_id():
    q8 = [
        {"id": "a", "check_hits": 2, "completion_tokens": 100,
         "repetition_max_trigram": 10, "final": "FINAL: q8"},
        {"id": "b", "check_hits": 1, "completion_tokens": 50,
         "repetition_max_trigram": 5, "final": ""},
    ]
    candidate = [
        {"id": "a", "check_hits": 3, "completion_tokens": 150,
         "repetition_max_trigram": 20, "final": "FINAL: c"},
        {"id": "b", "check_hits": 0, "completion_tokens": 25,
         "repetition_max_trigram": 10, "final": ""},
    ]

    summary = q8_relative_summary(candidate, q8)

    assert summary["total_check_delta"] == 0
    assert summary["mean_completion_token_ratio"] == 1.0
    assert summary["mean_repetition_ratio"] == 2.0
    assert summary["per_prompt"][0]["candidate_final"] == "FINAL: c"


def test_role_bit_summary_counts_affine_bits_by_role_only():
    decision = {
        "allocation": [
            {"kind": "affine", "role": "ffn.down_proj", "bits": 4},
            {"kind": "affine", "role": "ffn.down_proj", "bits": 5},
            {"kind": "affine", "role": "ffn.down_proj", "bits": 4},
            {"kind": "expert", "role": "moe.expert.down", "bits": 1},
        ]
    }

    assert role_bit_summary(decision) == {"ffn.down_proj": {"4": 2, "5": 1}}
