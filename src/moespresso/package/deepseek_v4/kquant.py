"""DeepSeek-V4 source experts -> K-quant routed bundle."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from moespresso.package.kquant_backend import KQuantEncodedWeight, encode_kquant_weight
from moespresso.package.kquant_bundle import assemble_kquant_encoded_layer_bundle
from moespresso.package.deepseek_v4.recipe import DS4KQuantExpertTarget
from moespresso.package.kquant_recipe import (
    KQuantRecipeError,
    validate_kquant_target_fit,
)
from moespresso.probe.calibration import read_imatrix_vectors


class DS4KQuantEncodeError(ValueError):
    pass


Encoder = Callable[
    [np.ndarray, DS4KQuantExpertTarget, dict[str, np.ndarray]],
    KQuantEncodedWeight,
]


def load_ds4_kquant_imatrix_vectors(path) -> dict[str, np.ndarray]:
    """Load imatrix vectors through the existing calibration reader."""
    return read_imatrix_vectors(path)


def _targets_by_projection(
    targets: Sequence[DS4KQuantExpertTarget],
    *,
    layer: int,
) -> dict[str, DS4KQuantExpertTarget]:
    by_projection: dict[str, DS4KQuantExpertTarget] = {}
    for target in targets:
        if int(target.layer_index) != int(layer):
            continue
        if target.projection in by_projection:
            raise DS4KQuantEncodeError(
                f"duplicate K-quant target for layer={layer} projection={target.projection}")
        by_projection[target.projection] = target
    missing = [projection for projection in ("gate", "up", "down")
               if projection not in by_projection]
    if missing:
        raise DS4KQuantEncodeError(
            f"missing K-quant target(s) for layer={layer}: {', '.join(missing)}")
    return by_projection


def encode_ds4_kquant_layer_bundle(
    expert_group: Any,
    targets: Sequence[DS4KQuantExpertTarget],
    imatrix_vectors: dict[str, np.ndarray],
    *,
    layer: int,
    max_experts: int | None = None,
    encoder: Encoder | None = None,
) -> tuple[np.ndarray, dict]:
    """Encode one DS4 routed layer into the normal MoEspresso bundle format.

    `expert_group` is the existing `DecodedExpertGroup` interface: it supplies
    expert ids and decodes each FP4 source expert/projection to logical
    `[out_features, in_features]` float weights. `encoder` defaults to the real
    `mlx-kquant` backend wrapper, but tests inject a fake encoder.
    """
    by_projection = _targets_by_projection(targets, layer=layer)
    expert_indices = list(expert_group.experts(layer))
    if max_experts is not None:
        expert_indices = expert_indices[:max_experts]
    if not expert_indices:
        raise DS4KQuantEncodeError(f"layer={layer}: no DS4 experts to encode")
    encoder = encoder or encode_kquant_weight

    encoded_by_projection: dict[str, list[KQuantEncodedWeight]] = {
        "gate": [],
        "up": [],
        "down": [],
    }
    for projection in ("gate", "up", "down"):
        target = by_projection[projection]
        for expert_index in expert_indices:
            decoded = expert_group.decode(
                layer=layer,
                expert_index=expert_index,
                projection=projection,
                out_dtype=np.float32,
            )
            decoded = np.asarray(decoded, dtype=np.float32)
            try:
                validate_kquant_target_fit(target, decoded.shape, imatrix_vectors)
            except KQuantRecipeError as exc:
                raise DS4KQuantEncodeError(
                    f"layer={layer} expert={expert_index} projection={projection}: {exc}"
                ) from exc
            encoded = encoder(decoded, target, imatrix_vectors)
            encoded_by_projection[projection].append(encoded)

    return assemble_kquant_encoded_layer_bundle(encoded_by_projection)
