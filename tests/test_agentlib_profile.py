"""The agentic profile loader and the loop-settings precedence chain.

Pins the resolution order (explicit arguments over user config over package
profile over built-ins), the per-key sampling merge, the clean fallthrough
for packages without a profile, and the fail-closed behavior on malformed
inputs. The Ornith end-to-end case reads a sidecar written by the package
writer itself, so the two ends of the contract stay tied to one schema.
"""

from __future__ import annotations

import json

import pytest

from moespresso.agentlib.loop_policy import ToolNudgePolicy
from moespresso.agentlib.profile import (
    AGENTIC_PROFILE_NAME,
    BUILTIN_SETTINGS,
    LoopSettings,
    ProfileError,
    load_agentic_profile,
    resolve_loop_settings,
)
from moespresso.package.agentic_profile import (
    profile_for_family,
    write_agentic_profile,
)


def _resolve(**kwargs):
    kwargs.setdefault("use_user_config", False)
    return resolve_loop_settings(**kwargs)


# --- loading -------------------------------------------------------------------

def test_missing_profile_loads_as_none(tmp_path):
    assert load_agentic_profile(tmp_path) is None


def test_written_profile_round_trips(tmp_path):
    write_agentic_profile(tmp_path, family="qwen3_5_moe")
    assert load_agentic_profile(tmp_path) == profile_for_family("qwen3_5_moe")


def test_malformed_profile_fails_closed(tmp_path):
    (tmp_path / AGENTIC_PROFILE_NAME).write_text("not json")
    with pytest.raises(ProfileError, match="unreadable"):
        load_agentic_profile(tmp_path)


def test_future_schema_version_fails_closed(tmp_path):
    (tmp_path / AGENTIC_PROFILE_NAME).write_text(
        json.dumps({"schema_version": 2, "dialect": "dsml"}))
    with pytest.raises(ProfileError, match="schema_version"):
        load_agentic_profile(tmp_path)


# --- fallthrough and package layer ----------------------------------------------

def test_no_profile_falls_through_to_builtins(tmp_path):
    settings = _resolve(package_dir=tmp_path)
    assert settings == LoopSettings(
        dialect=BUILTIN_SETTINGS["dialect"],
        repair=BUILTIN_SETTINGS["repair"],
        thinking_for_tools=None,
        reprompt_enabled=False,
        reprompt_limit=1,
        sampling={},
    )
    assert settings.nudge_policy() is None
    assert settings.chat_template_kwargs() is None
    assert settings.request_sampling() == {}


def test_ornith_profile_configures_the_loop(tmp_path):
    write_agentic_profile(tmp_path, family="qwen3_5_moe")
    settings = _resolve(package_dir=tmp_path)
    assert settings.dialect == "dsml"
    assert settings.repair is True
    assert settings.thinking_for_tools is False
    assert settings.reprompt_enabled is True
    assert settings.sampling == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0,
    }
    policy = settings.nudge_policy()
    assert isinstance(policy, ToolNudgePolicy) and policy.limit == 1
    assert settings.chat_template_kwargs() == {"enable_thinking": False}
    assert settings.dialect_adapter().name == "dsml"


def test_ds4_profile_leaves_unproven_settings_to_builtins(tmp_path):
    write_agentic_profile(tmp_path, family="deepseek_v4_flash")
    settings = _resolve(package_dir=tmp_path)
    assert settings.dialect == "dsml"
    assert settings.repair is False       # optional and disabled by default
    assert settings.thinking_for_tools is None
    assert settings.reprompt_enabled is False
    assert settings.sampling == {}


# --- precedence chain -------------------------------------------------------------

def _write_config(tmp_path, text: str):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_user_config_overrides_package_profile(tmp_path):
    write_agentic_profile(tmp_path, family="qwen3_5_moe")
    config = _write_config(tmp_path, (
        "[agent]\n"
        'dialect = "envelope"\n'
        "repair = false\n"
        "[agent.sampling]\n"
        "temperature = 0.9\n"
    ))
    settings = resolve_loop_settings(
        package_dir=tmp_path, config_path=config)
    assert settings.dialect == "envelope"
    assert settings.repair is False
    # per-key sampling merge: config pins temperature, profile keeps the rest
    assert settings.sampling["temperature"] == 0.9
    assert settings.sampling["top_p"] == 0.95
    assert settings.sampling["top_k"] == 20


def test_explicit_arguments_override_everything(tmp_path):
    write_agentic_profile(tmp_path, family="qwen3_5_moe")
    config = _write_config(tmp_path, (
        "[agent]\n"
        'dialect = "envelope"\n'
        "[agent.sampling]\n"
        "temperature = 0.9\n"
    ))
    settings = resolve_loop_settings(
        package_dir=tmp_path, config_path=config,
        dialect="native", thinking_for_tools=None,
        sampling={"top_k": 5})
    assert settings.dialect == "native"
    assert settings.thinking_for_tools is None
    assert settings.sampling["top_k"] == 5          # explicit wins
    assert settings.sampling["temperature"] == 0.9  # config layer survives
    assert settings.sampling["top_p"] == 0.95       # profile layer survives


def test_missing_config_file_contributes_nothing(tmp_path):
    write_agentic_profile(tmp_path, family="qwen3_5_moe")
    settings = resolve_loop_settings(
        package_dir=tmp_path, config_path=tmp_path / "absent.toml")
    assert settings.dialect == "dsml"


def test_package_profile_argument_wins_over_package_dir(tmp_path):
    write_agentic_profile(tmp_path, family="qwen3_5_moe")
    settings = _resolve(
        package_dir=tmp_path,
        package_profile=profile_for_family("deepseek_v4_flash"))
    assert settings.repair is False


# --- fail-closed config and profile fields ---------------------------------------

def test_unknown_agent_config_key_fails_closed(tmp_path):
    config = _write_config(tmp_path, "[agent]\nspeed = 11\n")
    with pytest.raises(ProfileError, match="speed"):
        resolve_loop_settings(config_path=config)


def test_wrong_typed_config_field_fails_closed(tmp_path):
    config = _write_config(tmp_path, "[agent]\nrepair = 'yes'\n")
    with pytest.raises(ProfileError, match="repair must be bool"):
        resolve_loop_settings(config_path=config)


def test_boolean_reprompt_limit_fails_closed(tmp_path):
    config = _write_config(tmp_path, "[agent]\nreprompt_limit = true\n")
    with pytest.raises(ProfileError, match="reprompt_limit must be int"):
        resolve_loop_settings(config_path=config)


def test_unknown_sampling_key_fails_closed(tmp_path):
    config = _write_config(
        tmp_path, "[agent]\n[agent.sampling]\nbeam_width = 4\n")
    with pytest.raises(ProfileError, match="beam_width"):
        resolve_loop_settings(config_path=config)


def test_malformed_profile_repair_block_fails_closed(tmp_path):
    (tmp_path / AGENTIC_PROFILE_NAME).write_text(json.dumps({
        "schema_version": 1, "dialect": "dsml", "repair": {"required": "yes"},
    }))
    with pytest.raises(ProfileError, match="repair"):
        _resolve(package_dir=tmp_path)
