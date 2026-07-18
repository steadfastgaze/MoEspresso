"""TurboQuant package encoding helpers."""

from __future__ import annotations

import numpy as np


def quantize_tq(sample: np.ndarray, bits: int, seed: int) -> dict[str, np.ndarray]:
    """Quantize one dense matrix with jang TurboQuant."""
    from jang_tools.turboquant.linear import tq_quantize_weight

    result = tq_quantize_weight(sample.astype(np.float32), bits=bits, seed=seed)
    return {
        "tq_packed": np.asarray(result["packed"]),
        "tq_norms": np.asarray(result["norms"]),
        "tq_bits": np.array([bits], dtype=np.uint8),
    }
