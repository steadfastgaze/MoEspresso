"""Load the package agentic profile and resolve the agent-loop settings.

The package carries an ``agentic_profile.json`` sidecar beside its vendored
chat template (``moespresso.package.agentic_profile`` documents the schema
and owns the writer). This module is the consumer: it reads the sidecar
from a package directory and resolves the loop-facing settings through a
fixed precedence chain, highest first:

1. explicit caller arguments,
2. the user config (``~/.moespresso/config.toml``, ``[agent]`` table with
   an optional ``[agent.sampling]`` sub-table),
3. the package profile,
4. library built-ins.

Scalar settings resolve whole; the sampling table merges per key across
the layers, so a config that pins only ``temperature`` keeps the package's
other sampling defaults. A package without a profile contributes nothing
and everything falls through. Malformed inputs fail closed: invalid JSON
or TOML, a wrong-typed field, or a profile schema version above the
supported one all raise ``ProfileError`` instead of guessing.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from moespresso.agentlib.loop_policy import ToolNudgePolicy
from moespresso.agentlib.sandbox import default_config_path

# One schema constant governs every reader of the sidecar: the package
# module that writes it, the serve layer, and this client loader.
from moespresso.package.agentic_profile import (
    AGENTIC_PROFILE_NAME,
    SCHEMA_VERSION as SUPPORTED_SCHEMA_VERSION,
)

SAMPLING_KEYS = frozenset({
    "temperature", "top_p", "top_k", "min_p",
    "presence_penalty", "repetition_penalty",
})

# The last-resort layer for packages without a profile: the template-native
# dialect with strict parsing, no nudge, template-default thinking, and
# server-default sampling.
BUILTIN_SETTINGS = {
    "dialect": "native",
    "repair": False,
    "thinking_for_tools": None,
    "reprompt_enabled": False,
    "reprompt_limit": 1,
    "sampling": {},
}

_UNSET = object()


class ProfileError(Exception):
    """A malformed agentic profile or [agent] config table."""


@dataclass(frozen=True)
class LoopSettings:
    """The resolved loop configuration an agent drives a model with."""

    dialect: str
    repair: bool
    thinking_for_tools: bool | None
    reprompt_enabled: bool
    reprompt_limit: int
    sampling: dict = field(default_factory=dict)

    def nudge_policy(self) -> ToolNudgePolicy | None:
        """A fresh tool-nudge policy per session, or None when disabled."""
        if not self.reprompt_enabled:
            return None
        return ToolNudgePolicy(limit=self.reprompt_limit)

    def chat_template_kwargs(self) -> dict | None:
        """The per-request template kwargs, or None for template defaults."""
        if self.thinking_for_tools is None:
            return None
        return {"enable_thinking": self.thinking_for_tools}

    def request_sampling(self) -> dict:
        """The sampling fields to place on a completion request."""
        return dict(self.sampling)

    def dialect_adapter(self):
        """The dialect adapter (system prompt, parser, result routing)."""
        # Imported lazily: the adapter registry lives beside the study
        # harness and pulls in every dialect parser.
        from moespresso.agentlib.dialect_study.dialects import dialect_for
        return dialect_for(self.dialect)


def load_agentic_profile(package_dir: str | Path) -> dict | None:
    """Read a package's agentic profile; None when the package has none."""
    path = Path(package_dir) / AGENTIC_PROFILE_NAME
    if not path.is_file():
        return None
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ProfileError(f"{path}: unreadable agentic profile: {e}") from e
    if not isinstance(profile, dict):
        raise ProfileError(f"{path}: agentic profile must be a JSON object")
    version = profile.get("schema_version")
    if not isinstance(version, int) or version > SUPPORTED_SCHEMA_VERSION:
        raise ProfileError(
            f"{path}: unsupported agentic profile schema_version "
            f"{version!r} (supported: <= {SUPPORTED_SCHEMA_VERSION})")
    return profile


def _profile_layer(profile: dict, *, source: str) -> dict:
    """The settings layer a package profile contributes."""
    layer: dict = {}
    if "dialect" in profile:
        layer["dialect"] = _typed(profile["dialect"], str, source, "dialect")
    repair = profile.get("repair")
    if repair is not None:
        if not isinstance(repair, dict) or not isinstance(
                repair.get("required"), bool):
            raise ProfileError(
                f"{source}: repair must be an object with a boolean 'required'")
        layer["repair"] = repair["required"]
    if "thinking_for_tools" in profile:
        layer["thinking_for_tools"] = _typed(
            profile["thinking_for_tools"], bool, source, "thinking_for_tools")
    reprompt = profile.get("reprompt")
    if reprompt is not None:
        if not isinstance(reprompt, dict) or not isinstance(
                reprompt.get("enabled"), bool):
            raise ProfileError(
                f"{source}: reprompt must be an object with a boolean 'enabled'")
        layer["reprompt_enabled"] = reprompt["enabled"]
        if "limit" in reprompt:
            layer["reprompt_limit"] = _typed(
                reprompt["limit"], int, source, "reprompt.limit")
    if "sampling" in profile:
        layer["sampling"] = _sampling_table(profile["sampling"], source)
    return layer


def _config_layer(config_path: str | Path | None) -> dict:
    """The settings layer the user config's [agent] table contributes."""
    path = Path(config_path).expanduser() if config_path is not None \
        else default_config_path()
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"{path}: invalid TOML: {e}") from e
    agent = data.get("agent", {})
    if not isinstance(agent, dict):
        raise ProfileError(f"{path}: [agent] must be a table")
    source = str(path)
    known = {"dialect", "repair", "thinking_for_tools", "reprompt",
             "reprompt_limit", "sampling"}
    unknown = sorted(set(agent) - known)
    if unknown:
        raise ProfileError(
            f"{source}: unknown [agent] key(s): {', '.join(unknown)}")
    layer: dict = {}
    if "dialect" in agent:
        layer["dialect"] = _typed(agent["dialect"], str, source, "dialect")
    if "repair" in agent:
        layer["repair"] = _typed(agent["repair"], bool, source, "repair")
    if "thinking_for_tools" in agent:
        layer["thinking_for_tools"] = _typed(
            agent["thinking_for_tools"], bool, source, "thinking_for_tools")
    if "reprompt" in agent:
        layer["reprompt_enabled"] = _typed(
            agent["reprompt"], bool, source, "reprompt")
    if "reprompt_limit" in agent:
        layer["reprompt_limit"] = _typed(
            agent["reprompt_limit"], int, source, "reprompt_limit")
    if "sampling" in agent:
        layer["sampling"] = _sampling_table(agent["sampling"], source)
    return layer


def _typed(value, kind, source: str, name: str):
    # bool is an int subclass; an int field must still refuse booleans.
    if isinstance(value, kind) and not (kind is int and isinstance(value, bool)):
        return value
    raise ProfileError(
        f"{source}: {name} must be {kind.__name__}, got {type(value).__name__}")


def _sampling_table(table, source: str) -> dict:
    if not isinstance(table, dict):
        raise ProfileError(f"{source}: sampling must be a table")
    unknown = sorted(set(table) - SAMPLING_KEYS)
    if unknown:
        raise ProfileError(
            f"{source}: unknown sampling key(s): {', '.join(unknown)}")
    for key, value in table.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProfileError(
                f"{source}: sampling.{key} must be a number")
    return dict(table)


def resolve_loop_settings(
    *,
    package_dir: str | Path | None = None,
    package_profile: dict | None = None,
    config_path: str | Path | None = None,
    use_user_config: bool = True,
    dialect=_UNSET,
    repair=_UNSET,
    thinking_for_tools=_UNSET,
    reprompt_enabled=_UNSET,
    reprompt_limit=_UNSET,
    sampling=_UNSET,
) -> LoopSettings:
    """Resolve the loop settings through the documented precedence chain.

    ``package_profile`` wins over ``package_dir`` when both are given (a
    caller that already loaded the profile passes it through). Explicit
    keyword arguments override everything; ``use_user_config=False`` skips
    the config layer entirely (offline and test runs).
    """
    if package_profile is None and package_dir is not None:
        package_profile = load_agentic_profile(package_dir)

    resolved = dict(BUILTIN_SETTINGS)
    resolved["sampling"] = dict(BUILTIN_SETTINGS["sampling"])
    layers = []
    if package_profile is not None:
        layers.append(_profile_layer(
            package_profile, source=f"package profile ({AGENTIC_PROFILE_NAME})"))
    if use_user_config:
        layers.append(_config_layer(config_path))
    overrides = {
        "dialect": dialect,
        "repair": repair,
        "thinking_for_tools": thinking_for_tools,
        "reprompt_enabled": reprompt_enabled,
        "reprompt_limit": reprompt_limit,
        "sampling": sampling,
    }
    layers.append({key: value for key, value in overrides.items()
                   if value is not _UNSET})

    for layer in layers:
        for key, value in layer.items():
            if key == "sampling":
                resolved["sampling"].update(value)
            else:
                resolved[key] = value

    return LoopSettings(
        dialect=resolved["dialect"],
        repair=resolved["repair"],
        thinking_for_tools=resolved["thinking_for_tools"],
        reprompt_enabled=resolved["reprompt_enabled"],
        reprompt_limit=resolved["reprompt_limit"],
        sampling=resolved["sampling"],
    )
