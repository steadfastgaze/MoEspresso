"""Serving modules use MLX/Metal tensor compute; numpy is confined to offline tools."""

from __future__ import annotations

import re
from pathlib import Path


_ENGINE_MODULES = ("build.py", "serve.py", "http.py")
_RUNTIME_DIR = Path(__file__).resolve().parent.parent / "src" / "moespresso" / "runtime"


def test_rl1_no_numpy_in_engine_modules():
    """RL1: the engine (serve path) source has no numpy import / np. usage."""
    offenders = {}
    for mod in _ENGINE_MODULES:
        src = (_RUNTIME_DIR / mod).read_text()
        hits = []
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]  # ignore comments
            if re.search(r"\bimport numpy\b", code) or re.search(r"\bnp\.", code):
                hits.append((i, line.strip()))
        if hits:
            offenders[mod] = hits
    assert not offenders, (
        "numpy tensor compute leaked into the engine (RED LINE 1). "
        f"Move it to the edge / offline convert. Offenders: {offenders}")


def test_rl1_deleted_loader_stays_gone():
    """The numpy weight-reconstruction loader (runtime/load.py) must not return."""
    assert not (_RUNTIME_DIR / "load.py").exists(), (
        "runtime/load.py (numpy weight reconstruction) was deleted. It must not "
        "come back on the engine path (RED LINE 1).")
