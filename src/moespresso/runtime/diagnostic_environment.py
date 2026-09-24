"""Limit inherited settings in diagnostic tools and recorded processes."""

import os


SYSTEM_ENV = frozenset({
    "HOME", "TMPDIR", "USER", "LOGNAME", "TZ", "LANG",
    "LC_ALL", "LC_CTYPE", "LC_COLLATE", "LC_MESSAGES", "LC_MONETARY",
    "LC_NUMERIC", "LC_TIME",
    "DEVELOPER_DIR", "SDKROOT", "TOOLCHAINS",
})


def diagnostic_tool_environment():
    """Build an allowlisted environment without changing the parent process.

    Xcode records target environment values in trace metadata. Unknown names,
    including names within recognized namespaces, are never inherited.
    System tools use the system executable search path.
    """
    env = {name: os.environ[name] for name in SYSTEM_ENV if name in os.environ}
    env["PATH"] = os.defpath
    return env
