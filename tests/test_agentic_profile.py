"""The agentic profile sidecar: schema, family values, and the writer.

The profile values are pinned facts from the recorded studies; these tests
hold the committed values to the evidence records and pin the writer's
manifest block so the builder registration stays correct.
"""

from __future__ import annotations

import hashlib
import json

from moespresso.package.agentic_profile import (
    AGENTIC_PROFILE_NAME,
    SCHEMA_VERSION,
    profile_for_family,
    write_agentic_profile,
)


def test_ornith_profile_matches_the_study_record():
    profile = profile_for_family("qwen3_5_moe")
    assert profile == {
        "schema_version": SCHEMA_VERSION,
        "family": "qwen3_5_moe",
        "dialect": "dsml",
        "repair": {"required": True},
        "thinking_for_tools": False,
        "reprompt": {"enabled": True, "limit": 1},
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
        "provenance": "docs/package_format.md#agentic-profile-records",
    }


def test_ds4_profile_carries_only_proven_values():
    profile = profile_for_family("deepseek_v4_flash")
    assert profile["dialect"] == "dsml"
    assert profile["repair"] == {"required": False}
    assert profile["provenance"] == "docs/package_format.md#agentic-profile-records"
    # absent keys mean the client decides; nothing is invented
    assert "sampling" not in profile
    assert "thinking_for_tools" not in profile
    assert "reprompt" not in profile


def test_unknown_family_has_no_profile():
    assert profile_for_family("synthetic_dense") is None
    assert profile_for_family(None) is None


def test_profile_for_family_returns_a_copy():
    first = profile_for_family("qwen3_5_moe")
    first["sampling"]["temperature"] = 99.0
    assert profile_for_family("qwen3_5_moe")["sampling"]["temperature"] == 0.6


def test_write_agentic_profile_round_trips_with_manifest_block(tmp_path):
    block = write_agentic_profile(tmp_path, family="qwen3_5_moe")
    path = tmp_path / AGENTIC_PROFILE_NAME
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == profile_for_family(
        "qwen3_5_moe")
    assert block["path"] == AGENTIC_PROFILE_NAME
    assert block["family"] == "qwen3_5_moe"
    assert block["size_bytes"] == path.stat().st_size
    assert block["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_agentic_profile_skips_families_without_evidence(tmp_path):
    assert write_agentic_profile(tmp_path, family="synthetic_dense") is None
    assert not (tmp_path / AGENTIC_PROFILE_NAME).exists()
