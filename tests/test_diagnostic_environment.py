"""Recorded processes inherit only explicitly supported settings."""

import os

from moespresso.runtime.diagnostic_environment import SYSTEM_ENV, diagnostic_tool_environment


def test_tool_environment_copies_only_exact_names_and_uses_system_path(monkeypatch):
    allowed = {name: "allowed-" + name for name in SYSTEM_ENV}
    parent = {
        **allowed, "PATH": "/untrusted/bin", "HF_TOKEN": "synthetic-secret",
        "UNRELATED_SETTING": "synthetic-secret", "LC_PRIVATE": "synthetic-secret",
        "MLX_API_KEY": "synthetic-secret", "MOESPRESSO_API_KEY": "synthetic-secret",
        "DEVELOPER_DIR_TOKEN": "synthetic-secret", "SDKROOT_TOKEN": "synthetic-secret",
        "TOOLCHAINS_TOKEN": "synthetic-secret", "TMPDIR_TOKEN": "synthetic-secret",
        "PYTHONPATH": "/untrusted/python", "DYLD_INSERT_LIBRARIES": "/untrusted/library",
        "SSH_AUTH_SOCK": "/untrusted/socket", "UV_INDEX_PRIVATE_PASSWORD": "synthetic-secret",
    }
    monkeypatch.setattr(os, "environ", parent.copy())
    actual = diagnostic_tool_environment()
    assert actual == {**allowed, "PATH": os.defpath}
    assert os.environ == parent
    actual["HOME"] = "changed-child-value"
    assert os.environ["HOME"] == allowed["HOME"]


def test_empty_parent_environment_does_not_trigger_an_inheritance_fallback(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    assert diagnostic_tool_environment() == {"PATH": os.defpath}
