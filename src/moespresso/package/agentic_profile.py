"""The agentic profile sidecar: per-model tool-dialect facts in the package.

The chat template ships in the package as model-specific serving metadata;
the tool dialect is the same kind of metadata. This sidecar records, beside
the vendored template, how an agent loop should drive the model: which
tool-call dialect it emits reliably, whether the repair layer is load-bearing
for that dialect, the thinking flag for tool work, the re-prompt policy, and
the recommended sampling defaults. agentlib reads the file and configures
its loop from it, so a client gets correct tool behavior without knowing
what a tool schema is. The profile is data; the parsers, the repair engine,
and the loop policies are code in agentlib.

Every value is pinned by a recorded study; the ``provenance`` field names
the public evidence record. Families without recorded evidence get no profile,
and a missing file means the client decides everything.

Schema (``agentic_profile.json``, schema_version 1); every key except
``schema_version``, ``family``, ``dialect``, and ``provenance`` is optional,
and an absent key means the client decides:

- ``schema_version``: integer. Readers fail closed on a version above the
  one they support.
- ``family``: the architecture family the profile was validated on.
- ``dialect``: the tool-call dialect the model emits reliably. Known
  values: ``native`` (template-rendered XML), ``envelope`` (Terminus-2
  JSON action object), ``dsml`` (DeepSeek text markers).
- ``repair``: ``{"required": bool}``. Required means the dialect is not
  viable without the repair layer, so the loop must enable it and treat a
  nonzero ``RepairTelemetry.failed`` count as the alarm condition.
  Not-required means repair is optional and the client decides.
- ``thinking_for_tools``: boolean; the thinking flag agentic sessions
  should request for tool work.
- ``reprompt``: ``{"enabled": bool, "limit": int}``; the tool-nudge loop
  policy (re-prompt when a final turn arrives before any tool call).
- ``sampling``: recommended request sampling defaults, any subset of
  ``temperature``, ``top_p``, ``top_k``, ``min_p``, ``presence_penalty``,
  ``repetition_penalty``.
- ``provenance``: the public repository locator for the evidence record the
  values come from.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

AGENTIC_PROFILE_NAME = "agentic_profile.json"
SCHEMA_VERSION = 1

# Profiles of record, keyed by architecture family. Values come from the
# named evidence records and change only with a new recorded study.
_FAMILY_PROFILES: dict[str, dict] = {
    # Ornith: the dialect study's golden form, validated end to end at this
    # sampling shape (15 of 15 with repair; the quoting malformation class
    # is fully salvaged and a nonzero failed count is the alarm).
    "qwen3_5_moe": {
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
    },
    # DeepSeek-V4 Flash: only what the road-test campaign proved. DSML is
    # the campaign's dialect with zero malformations over 40 requests, so
    # repair is optional; no sampling defaults are recorded, so the client
    # decides sampling.
    "deepseek_v4_flash": {
        "schema_version": SCHEMA_VERSION,
        "family": "deepseek_v4_flash",
        "dialect": "dsml",
        "repair": {"required": False},
        "provenance": "docs/package_format.md#agentic-profile-records",
    },
}


def profile_for_family(family: str | None) -> dict | None:
    """The agentic profile of record for a family, or None."""
    if family is None:
        return None
    profile = _FAMILY_PROFILES.get(family)
    return copy.deepcopy(profile) if profile is not None else None


def read_agentic_profile(package_dir: Path) -> dict | None:
    """Read a package's agentic profile sidecar, or None.

    Returns None when the file is absent, unreadable, not an object, or
    carries a ``schema_version`` above ``SCHEMA_VERSION``: readers fail
    closed on versions they do not understand, and a consumer treats a
    missing profile as "the client decides".
    """
    path = Path(package_dir) / AGENTIC_PROFILE_NAME
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(profile, dict):
        return None
    version = profile.get("schema_version")
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        return None
    return profile


def write_agentic_profile(package_dir: Path, *, family: str | None) -> dict | None:
    """Write the family's profile sidecar into the package.

    Returns the manifest ``agentic_profile`` block (path, sha256,
    size_bytes, family) for registration, or None when the family has no
    profile of record, in which case no file is written.
    """
    profile = profile_for_family(family)
    if profile is None:
        return None
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    path = package_dir / AGENTIC_PROFILE_NAME
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return {
        "path": AGENTIC_PROFILE_NAME,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "family": profile["family"],
    }
