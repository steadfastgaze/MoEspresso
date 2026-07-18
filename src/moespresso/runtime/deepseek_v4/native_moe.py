"""Loader for optional native DS4 routed-MoE kernels."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_NATIVE_DIR = Path(__file__).resolve().parents[4] / "native" / "ds4_moe" / "build"
_ENV_DIR = os.environ.get("MOESPRESSO_NATIVE_DS4_MOE_DIR")

# None = undecided, module = usable, False = unavailable/failed
_DS4_MOE: list = [None]


def _self_test(mod) -> bool:
    import mlx.core as mx
    import numpy as np

    try:
        rows_np = np.arange(2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8) / 64.0
        scores_np = np.array(
            [
                [0.31, 0.22, 0.17, 0.13, 0.09, 0.08],
                [0.08, 0.09, 0.13, 0.17, 0.22, 0.31],
            ],
            dtype=np.float32,
        )
        rows = mx.array(rows_np, dtype=mx.float32)
        scores = mx.array(scores_np, dtype=mx.float32)
        got = mod.weighted_sum6(rows, scores)
        expected = mx.sum(rows * scores[..., None], axis=1)
        mx.eval(got, expected)
        return bool(
            np.allclose(np.array(got), np.array(expected), rtol=1e-6, atol=1e-6))
    except Exception:
        return False


def load_ds4_moe():
    """Return the native DS4 routed-MoE module, or None when unavailable."""
    if _DS4_MOE[0] is not None:
        return _DS4_MOE[0] or None
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
            import _moespresso_ds4_moe as mod  # noqa: F401
            break
        except Exception:
            mod = None
        finally:
            sys.path.remove(str(cand))
    if mod is None or not _self_test(mod):
        if mod is not None:
            warnings.warn(
                "moespresso: native DS4 routed-MoE module built but failed "
                "its self-test.",
                RuntimeWarning,
                stacklevel=2,
            )
        _DS4_MOE[0] = False
        return None
    _DS4_MOE[0] = mod
    return mod


def weighted_sum6(rows, scores, *, stream=None):
    """Return ``sum_k(rows[:, k] * scores[:, k])`` using the native DS4 op."""
    mod = load_ds4_moe()
    if mod is None:
        raise RuntimeError("native DS4 routed-MoE module is unavailable")
    return mod.weighted_sum6(rows, scores, stream=stream)
