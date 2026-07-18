"""Loader for the optional native MTLSharedEvent gate.

The gate is a compiled MLX extension (native/gate). It is optional: when the
.so is missing, fails to import, or fails the runtime self-test, decode falls
back to the ring path transparently. Build with `native/build.sh`.

The self-test mirrors the ring visibility self-test discipline: once per
process, prove hold + foreign-thread release + value integrity before the
gate is allowed anywhere near the product path.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

_NATIVE_DIR = Path(__file__).resolve().parents[3] / "native" / "gate" / "build"
_ENV_DIR = os.environ.get("MOESPRESSO_NATIVE_DIR")

# None = undecided, module = usable, False = unavailable/failed
_GATE: list = [None]


def _self_test(mod) -> bool:
    import mlx.core as mx
    import numpy as np

    try:
        base = int(mod.signaled_value())
        x = mx.arange(64).astype(mx.float32)
        token = mx.array([1], dtype=mx.uint32)
        mx.eval(x, token)
        gated = mod.gate(x, token, base + 1)
        out = gated * 2.0
        hold_s = 0.05
        threading.Thread(
            target=lambda: (time.sleep(hold_s),
                            mod.signal_event(base + 1)),
        ).start()
        t0 = time.perf_counter()
        mx.eval(out)
        held = (time.perf_counter() - t0) >= hold_s * 0.8
        correct = bool(np.allclose(np.array(out), np.arange(64) * 2.0))
        return held and correct
    except Exception:
        return False


def load_gate():
    """Return the gate module, or None when unavailable (ring fallback)."""
    if _GATE[0] is not None:
        return _GATE[0] or None
    if os.environ.get("MOESPRESSO_SSD_GATE_DECODE", "1") == "0":
        _GATE[0] = False
        return None
    candidates = []
    if _ENV_DIR:
        candidates.append(Path(_ENV_DIR))
    candidates.append(_NATIVE_DIR)
    mod = None
    for cand in candidates:
        if not cand.is_dir():
            continue
        sys.path.insert(0, str(cand))
        try:
            import _moespresso_gate as mod  # noqa: F401
            break
        except Exception:
            mod = None
        finally:
            sys.path.remove(str(cand))
    if mod is None or not _self_test(mod):
        if mod is not None:
            import warnings
            warnings.warn(
                "moespresso: native gate built but FAILED its self-test; "
                "decode uses the ring path.",
                RuntimeWarning,
                stacklevel=2,
            )
        _GATE[0] = False
        return None
    _GATE[0] = mod
    return mod
