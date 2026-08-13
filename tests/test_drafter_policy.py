"""Bundled-drafter capacity policy: cache arithmetic and the decision.

Pure: the budget reader is injected, so every branch runs with synthetic
budgets and no sysctl or Metal read. The cache-state arithmetic is pinned
against the measured 131,072-token cache census of the served DeepSeek-V4
composite cache (33,664 bytes per token across the 43 trunk layers plus the
366,120,960-byte prefill-window constant).
"""

from __future__ import annotations

import pytest

from moespresso.inventory.architecture_profile import (
    DEEPSEEK_V4_FLASH_COMPRESS_RATIOS,
)
from moespresso.runtime.deepseek_v4.drafter_policy import (
    DEFAULT_CONTEXT_TOKENS,
    PREFILL_WINDOW_STATE_BYTES,
    WORKING_SET_MARGIN_BYTES,
    WORKING_SET_MARGIN_BYTES_PARTS32,
    WORKING_SET_MARGIN_BYTES_UNSPLIT,
    deepseek_v4_cache_state_bytes,
    dspark_state_bytes,
    evaluate_bundled_drafter_budget,
    working_set_margin_bytes,
)
from moespresso.runtime.deepseek_v4.model import (
    _deepseek_v4_auto_context_limit,
)
from moespresso.runtime.streaming_capacity import CapacityBudget

ARCHITECTURE = {
    "num_hidden_layers": 43,
    "compress_ratios": list(DEEPSEEK_V4_FLASH_COMPRESS_RATIOS),
    "attention": {"head_dim": 512, "index_head_dim": 128},
}

# The measured census at exactly 131,072 tokens: 21 ratio-4 layers at 1,280
# B/token, 20 ratio-128 layers at 32 B/token, 2 ratio-0 layers at 3,072
# B/token, plus the constant.
MEASURED_CACHE_BYTES_AT_128K = 4_778_528_768


def _manifest(weights=80 << 30, drafter=6 << 30):
    return {
        "files": [{"path": "model-00001-of-00001.safetensors",
                   "size_bytes": weights, "sha256": "0" * 64}],
        "drafter": {
            "family": "dspark",
            "files": [{"path": "model-dspark-00001-of-00001.safetensors",
                       "size_bytes": drafter, "sha256": "0" * 64}],
        },
        "architecture": ARCHITECTURE,
    }


def test_default_context_matches_the_served_context_limit():
    from moespresso.runtime.prefix_cache import DEFAULT_CONTEXT_LIMIT

    assert DEFAULT_CONTEXT_TOKENS == DEFAULT_CONTEXT_LIMIT == 131072


def test_cache_state_bytes_reproduces_the_measured_128k_census():
    got = deepseek_v4_cache_state_bytes(ARCHITECTURE, DEFAULT_CONTEXT_TOKENS)
    assert got == MEASURED_CACHE_BYTES_AT_128K


def test_cache_state_bytes_slope_is_the_measured_per_token_rate():
    at_zero = deepseek_v4_cache_state_bytes(ARCHITECTURE, 0)
    at_one_k = deepseek_v4_cache_state_bytes(ARCHITECTURE, 1024)
    assert at_zero == PREFILL_WINDOW_STATE_BYTES
    assert at_one_k - at_zero == 33_664 * 1024


def test_cache_state_bytes_fails_on_ratio_count_mismatch():
    broken = dict(ARCHITECTURE)
    broken["compress_ratios"] = [0, 0, 4]
    with pytest.raises(ValueError, match="compress ratios"):
        deepseek_v4_cache_state_bytes(broken, 1024)


def test_low_memory_context_keeps_the_minimum_expert_pool():
    gib = 1 << 30
    budget = CapacityBudget(
        available_bytes=11 * gib,
        resident_base_bytes=int(6.002493788488209 * gib),
        runtime_resident_bytes=int(0.04273653030395508 * gib),
        kv_activation_allowance_bytes=0,
        safety_margin_bytes=2 * gib,
        bytes_per_capacity_unit=int(266.86328125 * (1 << 20)),
        min_capacity=8,
        max_capacity=256,
    )

    assert _deepseek_v4_auto_context_limit(budget, ARCHITECTURE) == 16384


def test_dspark_state_is_fixed_and_context_independent():
    assert dspark_state_bytes(ARCHITECTURE, 3) == 3 * 128 * 512 * 4 == 786_432


def test_working_set_margins_by_split_setting():
    assert working_set_margin_bytes(16) == WORKING_SET_MARGIN_BYTES
    assert working_set_margin_bytes(32) == WORKING_SET_MARGIN_BYTES_PARTS32
    assert working_set_margin_bytes(64) == WORKING_SET_MARGIN_BYTES_PARTS32
    assert working_set_margin_bytes(8) == WORKING_SET_MARGIN_BYTES_UNSPLIT
    assert working_set_margin_bytes(1) == WORKING_SET_MARGIN_BYTES_UNSPLIT
    assert WORKING_SET_MARGIN_BYTES_PARTS32 < WORKING_SET_MARGIN_BYTES


def _base_bytes():
    return ((80 << 30) + (6 << 30) + MEASURED_CACHE_BYTES_AT_128K + 786_432)


def test_budget_fit_prefers_the_committed_parts_sixteen():
    manifest = _manifest()
    decision = evaluate_bundled_drafter_budget(
        manifest, n_stages=3, budget_fn=lambda: (120 << 30, "synthetic"))
    assert decision["decision"] == "on"
    assert decision["reason"] == "budget-fit"
    assert decision["budget_source"] == "synthetic"
    assert decision["sort_nsplit"] == 16
    assert decision["sort_nsplit_source"] == "default"
    assert decision["required_bytes"] == (
        _base_bytes() + WORKING_SET_MARGIN_BYTES)
    assert decision["required_bytes_by_nsplit"] == {
        "16": _base_bytes() + WORKING_SET_MARGIN_BYTES,
        "32": _base_bytes() + WORKING_SET_MARGIN_BYTES_PARTS32,
    }
    assert decision["required_bytes"] <= decision["budget_bytes"]


def test_tight_budget_selects_the_parts_32_floor_before_off():
    required16 = _base_bytes() + WORKING_SET_MARGIN_BYTES
    decision = evaluate_bundled_drafter_budget(
        _manifest(), n_stages=3,
        budget_fn=lambda: (required16 - 1, "synthetic"))
    assert decision["decision"] == "on"
    assert decision["reason"] == "budget-fit-nsplit32"
    assert decision["sort_nsplit"] == 32
    assert decision["sort_nsplit_source"] == "policy"
    assert decision["required_bytes"] == (
        _base_bytes() + WORKING_SET_MARGIN_BYTES_PARTS32)


def test_budget_exceeded_decides_off_with_both_candidates_recorded():
    decision = evaluate_bundled_drafter_budget(
        _manifest(), n_stages=3, budget_fn=lambda: (90 << 30, "synthetic"))
    assert decision["decision"] == "off"
    assert decision["reason"] == "budget-exceeded"
    assert decision["required_bytes"] > decision["budget_bytes"]
    assert set(decision["required_bytes_by_nsplit"]) == {"16", "32"}
    assert min(
        decision["required_bytes_by_nsplit"].values()
    ) > decision["budget_bytes"]


def test_budget_boundary_is_inclusive_at_both_margins():
    required16 = _base_bytes() + WORKING_SET_MARGIN_BYTES
    required32 = _base_bytes() + WORKING_SET_MARGIN_BYTES_PARTS32
    at16 = evaluate_bundled_drafter_budget(
        _manifest(), n_stages=3,
        budget_fn=lambda: (required16, "synthetic"))
    at32 = evaluate_bundled_drafter_budget(
        _manifest(), n_stages=3,
        budget_fn=lambda: (required32, "synthetic"))
    under32 = evaluate_bundled_drafter_budget(
        _manifest(), n_stages=3,
        budget_fn=lambda: (required32 - 1, "synthetic"))
    assert (at16["decision"], at16["sort_nsplit"]) == ("on", 16)
    assert (at32["decision"], at32["sort_nsplit"]) == ("on", 32)
    assert under32["decision"] == "off"


def test_env_pinned_split_is_never_overridden():
    required32 = _base_bytes() + WORKING_SET_MARGIN_BYTES_PARTS32
    decision = evaluate_bundled_drafter_budget(
        _manifest(), n_stages=3,
        budget_fn=lambda: (required32, "synthetic"),
        sort_nsplit_env="32")
    assert decision["decision"] == "on"
    assert decision["reason"] == "budget-fit"
    assert decision["sort_nsplit"] == 32
    assert decision["sort_nsplit_source"] == "env"
    assert list(decision["required_bytes_by_nsplit"]) == ["32"]

    pinned16 = evaluate_bundled_drafter_budget(
        _manifest(), n_stages=3,
        budget_fn=lambda: (required32, "synthetic"),
        sort_nsplit_env="16")
    assert pinned16["decision"] == "off"
    assert pinned16["reason"] == "budget-exceeded"
    assert pinned16["sort_nsplit_source"] == "env"


def test_env_pinned_sub_sixteen_split_reserves_the_unsplit_class():
    decision = evaluate_bundled_drafter_budget(
        _manifest(), n_stages=3,
        budget_fn=lambda: (200 << 30, "synthetic"),
        sort_nsplit_env="8")
    assert decision["decision"] == "on"
    assert decision["working_set_margin_bytes"] == (
        WORKING_SET_MARGIN_BYTES_UNSPLIT)


@pytest.mark.parametrize("value", ["banana", "0", "-4", "12"])
def test_unreadable_env_split_fails_closed_to_off(value):
    decision = evaluate_bundled_drafter_budget(
        _manifest(), n_stages=3,
        budget_fn=lambda: (200 << 30, "synthetic"),
        sort_nsplit_env=value)
    assert decision["decision"] == "off"
    assert decision["reason"] == "sort-nsplit-env-unreadable"


def test_unreadable_budget_fails_closed_to_off():
    decision = evaluate_bundled_drafter_budget(
        _manifest(), n_stages=3, budget_fn=lambda: (None, "unreadable"))
    assert decision["decision"] == "off"
    assert decision["reason"] == "budget-unreadable"


def test_unreadable_weight_bytes_fail_closed_to_off():
    manifest = _manifest()
    manifest["files"] = []
    decision = evaluate_bundled_drafter_budget(
        manifest, n_stages=3, budget_fn=lambda: (120 << 30, "synthetic"))
    assert decision == {
        "decision": "off",
        "reason": "weights-bytes-unreadable",
        "context_tokens": DEFAULT_CONTEXT_TOKENS,
    }


def test_unreadable_drafter_bytes_fail_closed_to_off():
    manifest = _manifest()
    manifest["drafter"]["files"][0]["size_bytes"] = "big"
    decision = evaluate_bundled_drafter_budget(
        manifest, n_stages=3, budget_fn=lambda: (120 << 30, "synthetic"))
    assert decision["decision"] == "off"
    assert decision["reason"] == "drafter-bytes-unreadable"


def test_unreadable_cache_contract_fails_closed_to_off():
    manifest = _manifest()
    manifest["architecture"] = {"attention": {}}
    decision = evaluate_bundled_drafter_budget(
        manifest, n_stages=3, budget_fn=lambda: (120 << 30, "synthetic"))
    assert decision["decision"] == "off"
    assert decision["reason"].startswith("cache-contract-unreadable")
