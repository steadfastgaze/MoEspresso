"""Pure helpers for affine role-elasticity calibration experiments.

The heavy part of the calibration harness loads real packages and generates text.
This module keeps the prompt set and metrics pure so tests can pin the evidence
shape without touching MLX or a full model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


PROMPT_SET_VERSION = "qwen35_affine_role_elasticity_v5"
QWEN35_AFFINE_ROLE_PROFILE_V1_NAME = "qwen35_affine_role_band_v1"
QWEN35_MOE_AFFINE_ROLE_PROFILE_V1_NAME = "qwen35_moe_affine_role_band_v1"

# Provisional role weights derived from a 0.8B model for package experiments.
#
# This is not MoQ and not yet a default optimizer policy. It captures the first
# evidence-backed direction: protect residual-write projections and discount
# some input-side roles that otherwise dominate the local imatrix objective.
QWEN35_AFFINE_ROLE_WEIGHTS_V1: dict[str, float] = {
    "ffn.down_proj": 128.0,
    "ssm.out_proj": 128.0,
    "attn.o_proj": 128.0,
    "ssm.in_proj_a": 0.35,
    "ssm.in_proj_b": 0.35,
    "ssm.in_proj_z": 0.35,
    "ffn.gate_proj": 0.5,
    "ffn.up_proj": 0.5,
    "ssm.in_proj_qkv": 0.5,
    "attn.q_proj": 0.5,
    "attn.k_proj": 0.5,
    "attn.v_proj": 0.5,
}


def qwen35_affine_role_weights_v1() -> dict[str, float]:
    """Return a copy of the provisional Qwen3.5 affine role weights."""
    return dict(QWEN35_AFFINE_ROLE_WEIGHTS_V1)


# Provisional role-band profile that matched Q8/manual behavior on the 0.8B
# calibration harness at 0.50 GiB. This remains experimental evidence about
# allocator direction and is not the default conversion policy.
QWEN35_AFFINE_ROLE_WEIGHTS_V2: dict[str, float] = {
    **QWEN35_AFFINE_ROLE_WEIGHTS_V1,
    "ssm.in_proj_a": 0.05,
    "ssm.in_proj_b": 0.05,
    "ssm.in_proj_z": 0.05,
}

QWEN35_AFFINE_ROLE_BIT_WEIGHTS_V1: dict[str, dict[int, float]] = {
    "ffn.down_proj": {3: 0.001, 4: 20.0, 5: 20.0, 6: 0.01, 8: 0.001},
    "ssm.out_proj": {3: 0.001, 4: 20.0, 5: 20.0, 6: 0.01, 8: 0.001},
    "attn.o_proj": {3: 0.001, 4: 20.0, 5: 20.0, 6: 0.01, 8: 0.001},
    "ssm.in_proj_a": {3: 0.001, 4: 50.0, 5: 0.001, 6: 0.0001, 8: 0.00001},
    "ssm.in_proj_b": {3: 0.001, 4: 50.0, 5: 0.001, 6: 0.0001, 8: 0.00001},
    "ssm.in_proj_z": {3: 0.001, 4: 50.0, 5: 0.001, 6: 0.0001, 8: 0.00001},
    "ffn.gate_proj": {3: 0.001, 4: 0.5, 5: 1.0, 6: 5.0, 8: 5.0},
    "ffn.up_proj": {3: 0.001, 4: 0.5, 5: 1.0, 6: 5.0, 8: 5.0},
    "ssm.in_proj_qkv": {3: 0.001, 4: 1.0, 5: 10.0, 6: 0.01, 8: 0.001},
    "attn.q_proj": {3: 0.001, 4: 1.0, 5: 10.0, 6: 0.01, 8: 0.001},
    "attn.k_proj": {3: 0.001, 4: 1.0, 5: 10.0, 6: 0.01, 8: 0.001},
    "attn.v_proj": {3: 0.001, 4: 1.0, 5: 10.0, 6: 0.01, 8: 0.001},
}

QWEN35_AFFINE_ROLE_MIN_BITS_V1: dict[str, int] = {
    "attn.k_proj": 4,
    "attn.o_proj": 4,
    "attn.q_proj": 4,
    "attn.v_proj": 4,
    "ffn.down_proj": 4,
    "ffn.gate_proj": 4,
    "ffn.up_proj": 4,
    "ssm.in_proj_a": 4,
    "ssm.in_proj_b": 4,
    "ssm.in_proj_qkv": 4,
    "ssm.in_proj_z": 4,
    "ssm.out_proj": 4,
}


def qwen35_affine_role_weights_v2() -> dict[str, float]:
    """Return a copy of the provisional role-band Qwen3.5 role weights."""
    return dict(QWEN35_AFFINE_ROLE_WEIGHTS_V2)


def qwen35_affine_role_bit_weights_v1() -> dict[str, dict[int, float]]:
    """Return a deep copy of provisional Qwen3.5 role destination-bit priors."""
    return {role: dict(weights) for role, weights in QWEN35_AFFINE_ROLE_BIT_WEIGHTS_V1.items()}


def qwen35_affine_role_min_bits_v1() -> dict[str, int]:
    """Return a copy of provisional Qwen3.5 affine role min-bit floors."""
    return dict(QWEN35_AFFINE_ROLE_MIN_BITS_V1)


def qwen35_affine_role_profile_v1() -> dict:
    """Return the current experimental Qwen3.5 dense affine role profile."""
    return {
        "name": QWEN35_AFFINE_ROLE_PROFILE_V1_NAME,
        "affine_role_weights": qwen35_affine_role_weights_v2(),
        "affine_role_bit_weights": qwen35_affine_role_bit_weights_v1(),
        "affine_role_min_bits": qwen35_affine_role_min_bits_v1(),
    }


_MOE_SHARED_EXPERT_ALIASES: dict[str, str] = {
    "moe.shared_expert.gate_proj": "ffn.gate_proj",
    "moe.shared_expert.up_proj": "ffn.up_proj",
    "moe.shared_expert.down_proj": "ffn.down_proj",
}


def _with_moe_shared_expert_aliases(dense_map: dict) -> dict:
    out = dict(dense_map)
    for moe_role, dense_role in _MOE_SHARED_EXPERT_ALIASES.items():
        out[moe_role] = dense_map[dense_role]
    return out


def qwen35_moe_affine_role_weights_v1() -> dict[str, float]:
    """Return Qwen MoE affine role weights, aliasing shared experts to dense FFN roles."""
    return _with_moe_shared_expert_aliases(qwen35_affine_role_weights_v2())


def qwen35_moe_affine_role_bit_weights_v1() -> dict[str, dict[int, float]]:
    """Return Qwen MoE affine destination-bit priors with explicit shared-expert roles."""
    dense = qwen35_affine_role_bit_weights_v1()
    out = {role: dict(weights) for role, weights in dense.items()}
    for moe_role, dense_role in _MOE_SHARED_EXPERT_ALIASES.items():
        out[moe_role] = dict(dense[dense_role])
    return out


def qwen35_moe_affine_role_min_bits_v1() -> dict[str, int]:
    """Return Qwen MoE affine min-bit floors with explicit shared-expert roles."""
    return _with_moe_shared_expert_aliases(qwen35_affine_role_min_bits_v1())


def qwen35_moe_affine_role_profile_v1() -> dict:
    """Return the current Qwen3.5/3.6 MoE affine role profile.

    Routed experts are not affine units and router gates are fp16 passthrough, so neither
    appears here. The allocator receives only normal affine roles.
    """
    return {
        "name": QWEN35_MOE_AFFINE_ROLE_PROFILE_V1_NAME,
        "affine_role_weights": qwen35_moe_affine_role_weights_v1(),
        "affine_role_bit_weights": qwen35_moe_affine_role_bit_weights_v1(),
        "affine_role_min_bits": qwen35_moe_affine_role_min_bits_v1(),
    }


DEEPSEEK_V4_AFFINE_ROLE_PROFILE_V1_NAME = "deepseek_v4_dense_conservative_v2"

# DeepSeek-V4-Flash affine backbone band. Do not let the probe-derived scalar
# quality proxy trade dense/non-expert backbone bits for routed-expert bits here:
# Q1 showed that the proxy can approve garbage allocations, while routed experts
# are the intended compression reservoir. Keep non-expert affine tensors
# conservative by default and let routed experts absorb target-size pressure.
_DEEPSEEK_V4_BACKBONE_AFFINE_ROLES = (
    "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b",
    "attn.indexer.wq_b", "attn.indexer.weights_proj",
    "attn.compressor.wkv", "attn.compressor.wgate",
    "attn.indexer.compressor.wkv", "attn.indexer.compressor.wgate",
    "moe.shared_expert.gate_proj", "moe.shared_expert.up_proj",
    "moe.shared_expert.down_proj",
)
# Output-side projections carry especially reconstruction-sensitive signal. They
# get a stronger 6-bit prior, but the structural floor applies to the full DS4
# affine backbone.
_DEEPSEEK_V4_OUTPUT_AFFINE_ROLES = ("attn.wo_a", "attn.wo_b", "moe.shared_expert.down_proj")


def deepseek_v4_affine_role_profile_v1() -> dict:
    """Dense-conservative affine role band for deepseek_v4_flash.

    DS4 uses source-FP4 routed experts and TQ/mxfp4 expert bundles; those routed
    experts are the part we deliberately compress aggressively. Non-expert affine
    tensors are graph/backbone infrastructure, so they stay at 6 bits unless an
    explicit caller overrides the profile.
    """
    min_bits = {role: 6 for role in _DEEPSEEK_V4_BACKBONE_AFFINE_ROLES}
    bit_weights = {}
    for role in _DEEPSEEK_V4_BACKBONE_AFFINE_ROLES:
        if role in _DEEPSEEK_V4_OUTPUT_AFFINE_ROLES:
            bit_weights[role] = {3: 0.001, 4: 0.001, 5: 0.001, 6: 20.0, 8: 0.01}
        else:
            bit_weights[role] = {3: 0.001, 4: 0.001, 5: 0.001, 6: 5.0, 8: 0.001}
    return {
        "name": DEEPSEEK_V4_AFFINE_ROLE_PROFILE_V1_NAME,
        "affine_role_weights": {role: 1.0 for role in _DEEPSEEK_V4_BACKBONE_AFFINE_ROLES},
        "affine_role_bit_weights": bit_weights,
        "affine_role_min_bits": min_bits,
    }


def affine_role_profile_for_family(family: str | None) -> dict:
    """Return default affine-role optimizer settings for a supported family."""
    if family == "qwen3_5_dense":
        return qwen35_affine_role_profile_v1()
    if family == "qwen3_5_moe":
        return qwen35_moe_affine_role_profile_v1()
    if family == "deepseek_v4_flash":
        return deepseek_v4_affine_role_profile_v1()
    return {}


@dataclass(frozen=True)
class CalibrationPrompt:
    id: str
    topic: str
    prompt: str
    checks: tuple[str, ...]


PROMPTS: tuple[CalibrationPrompt, ...] = (
    CalibrationPrompt(
        id="math_rates",
        topic="math",
        prompt=(
            "Return exactly one line beginning with FINAL:. Do not show reasoning.\n\n"
            "Compute 391 + 19. Choices: A=391, B=410, C=420. Answer with the "
            "number, not the letter."
        ),
        checks=("FINAL", "410"),
    ),
    CalibrationPrompt(
        id="code_bug",
        topic="code",
        prompt=(
            "The corrected line is `seen.add(x)`. Return exactly this one line: "
            "FINAL: seen.add(x)\n\n"
            "def first_dup(xs):\n"
            "    seen = set()\n"
            "    for x in xs:\n"
            "        if x in seen:\n"
            "            return x\n"
            "        seen.add(xs)\n"
            "    return None"
        ),
        checks=("FINAL", "seen.add(x)"),
    ),
    CalibrationPrompt(
        id="logic_order",
        topic="logic",
        prompt=(
            "Return exactly one line beginning with FINAL:.\n\n"
            "The key is either in the blue box or the green box. It is not in the "
            "blue box. Answer with the box color."
        ),
        checks=("FINAL", "green"),
    ),
    CalibrationPrompt(
        id="science_causal",
        topic="science",
        prompt=(
            "Return exactly one line beginning with FINAL:. Answer in one sentence.\n\n"
            "Choose the correct explanation for why ice floats on liquid water: "
            "A) hydrogen bonds create an open arrangement with lower density; "
            "B) hydrogen bonds create a packed arrangement with higher density. "
            "Include the words hydrogen bonds and lower density."
        ),
        checks=("FINAL", "hydrogen", "lower density"),
    ),
    CalibrationPrompt(
        id="planning_constraints",
        topic="planning",
        prompt=(
            "A meeting ends at 10:00 and the required buffer is 20 minutes. The "
            "earliest valid focus-block start time is 10:20. Return exactly this "
            "one line: FINAL: 10:20"
        ),
        checks=("FINAL", "10:20"),
    ),
    CalibrationPrompt(
        id="language_precision",
        topic="language",
        prompt=(
            "The Italian translation is `esattamente lo stesso prefisso di token`. "
            "Return exactly this one line: FINAL: esattamente lo stesso prefisso "
            "di token"
        ),
        checks=("FINAL", "stesso", "prefisso"),
    ),
)


def repetition_max_trigram(text: str) -> int:
    """Return the highest repeated word-trigram count in generated text."""
    words = text.lower().split()
    triples = list(zip(words, words[1:], words[2:]))
    if not triples:
        return 0
    return max(Counter(triples).values())


def check_hits(text: str, checks: Iterable[str]) -> int:
    """Count case-insensitive substring checks present in the generated text."""
    lower = text.lower()
    return sum(1 for check in checks if check.lower() in lower)


def extract_final_line(text: str) -> str:
    """Return the first final-answer line, or empty string when absent."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("final:"):
            return stripped
    return ""


def score_generation(prompt: CalibrationPrompt, text: str, *, finish_reason: str,
                     completion_tokens: int) -> dict:
    """Structured per-prompt evidence row."""
    return {
        "id": prompt.id,
        "topic": prompt.topic,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "check_hits": check_hits(text, prompt.checks),
        "check_total": len(prompt.checks),
        "final": extract_final_line(text),
        "repetition_max_trigram": repetition_max_trigram(text),
        "text": text,
    }


def summarize_generation(rows: list[dict]) -> dict:
    """Aggregate behavior metrics over scored prompt rows."""
    total_checks = sum(int(r["check_hits"]) for r in rows)
    total_possible = sum(int(r["check_total"]) for r in rows)
    return {
        "prompt_count": len(rows),
        "checks_hit": total_checks,
        "checks_total": total_possible,
        "final_lines": sum(1 for r in rows if r.get("final")),
        "stops": sum(1 for r in rows if r.get("finish_reason") == "stop"),
        "length_finishes": sum(1 for r in rows if r.get("finish_reason") == "length"),
        "max_repeated_trigram": max((int(r["repetition_max_trigram"]) for r in rows), default=0),
        "completion_tokens": [int(r["completion_tokens"]) for r in rows],
    }


def q8_relative_summary(candidate_rows: list[dict], q8_rows: list[dict]) -> dict:
    """Compare candidate rows against Q8 rows by prompt id."""
    by_q8 = {r["id"]: r for r in q8_rows}
    per_prompt = []
    for row in candidate_rows:
        base = by_q8.get(row["id"])
        if base is None:
            continue
        q8_tokens = max(1, int(base["completion_tokens"]))
        q8_rep = max(1, int(base["repetition_max_trigram"]))
        per_prompt.append({
            "id": row["id"],
            "check_delta": int(row["check_hits"]) - int(base["check_hits"]),
            "completion_token_ratio": int(row["completion_tokens"]) / q8_tokens,
            "repetition_ratio": int(row["repetition_max_trigram"]) / q8_rep,
            "candidate_final": row.get("final", ""),
            "q8_final": base.get("final", ""),
        })
    if not per_prompt:
        return {"per_prompt": [], "mean_completion_token_ratio": None,
                "mean_repetition_ratio": None, "total_check_delta": None}
    return {
        "per_prompt": per_prompt,
        "mean_completion_token_ratio": (
            sum(r["completion_token_ratio"] for r in per_prompt) / len(per_prompt)
        ),
        "mean_repetition_ratio": (
            sum(r["repetition_ratio"] for r in per_prompt) / len(per_prompt)
        ),
        "total_check_delta": sum(int(r["check_delta"]) for r in per_prompt),
    }


def role_bit_summary(decision: dict) -> dict:
    """Role -> bit-count map from an optimizer_decision allocation."""
    roles: dict[str, Counter] = {}
    for item in decision.get("allocation", []):
        if item.get("kind") != "affine":
            continue
        role = item["role"]
        roles.setdefault(role, Counter())[str(item["bits"])] += 1
    return {role: dict(sorted(counts.items())) for role, counts in sorted(roles.items())}
