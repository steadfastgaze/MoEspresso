"""K-quant encoded expert weights -> MoEspresso routed bundle rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from moespresso.package.bundle import KQUANT_CODEC, PROJECTIONS, assemble_layer_bundle
from moespresso.package.kquant_backend import KQuantEncodedWeight
from moespresso.package.kquant_format import KQUANT_GEOMETRY


class KQuantBundleError(ValueError):
    pass


# Allocation rows and recipe targets name routed projections gate/up/down; the
# bundle metadata names them after the module path (gate_proj/up_proj/down_proj).
_BUNDLE_PROJECTIONS = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def stack_kquant_projection_components(
    encoded: Sequence[KQuantEncodedWeight],
    *,
    projection: str,
) -> tuple[str, dict[str, np.ndarray]]:
    """Stack one projection's encoded experts for `assemble_layer_bundle()`.

    Input weights are per-expert `uint8 [out_features, bytes_per_row]` wire
    bytes. The placeholder scales from mlx-kquant must be `uint8 [1]` for every
    expert; the bundle stores them as `[num_experts, 1]` so each expert row is
    independently sliceable by the existing loader/index path.
    """
    if not encoded:
        raise KQuantBundleError(f"{projection}: no encoded experts")
    codec = encoded[0].codec
    geometry = KQUANT_GEOMETRY.get(codec)
    if geometry is None:
        raise KQuantBundleError(f"{projection}: unknown K-quant codec {codec!r}")

    weights: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    shape = None
    for idx, item in enumerate(encoded):
        if item.codec != codec:
            raise KQuantBundleError(
                f"{projection}: expert {idx} codec {item.codec!r} != {codec!r}")
        weight = np.asarray(item.weight)
        scale = np.asarray(item.scales)
        if weight.dtype != np.uint8 or weight.ndim != 2:
            raise KQuantBundleError(
                f"{projection}: expert {idx} weight must be 2D uint8, got "
                f"{weight.ndim}D {weight.dtype}")
        if weight.shape[1] % geometry.bytes_per_block:
            raise KQuantBundleError(
                f"{projection}: expert {idx} bytes_per_row {weight.shape[1]} "
                f"is not divisible by {geometry.bytes_per_block} for {codec}")
        if scale.dtype != np.uint8 or scale.shape != (1,):
            raise KQuantBundleError(
                f"{projection}: expert {idx} scales must be uint8[1], got "
                f"{scale.dtype}{scale.shape}")
        if shape is None:
            shape = weight.shape
        elif weight.shape != shape:
            raise KQuantBundleError(
                f"{projection}: expert {idx} weight shape {weight.shape} != {shape}")
        weights.append(np.ascontiguousarray(weight))
        scales.append(np.ascontiguousarray(scale))

    return codec, {
        "weight": np.stack(weights, axis=0),
        "scales": np.stack(scales, axis=0),
    }


def assemble_kquant_encoded_layer_bundle(
    encoded_by_projection: Mapping[str, Sequence[KQuantEncodedWeight]],
) -> tuple[np.ndarray, dict]:
    """Build one routed layer bundle from encoded K-quant experts.

    Keys are the allocation-domain projection names: gate, up, down.
    """
    unknown = [name for name in encoded_by_projection if name not in _BUNDLE_PROJECTIONS]
    if unknown:
        raise KQuantBundleError(
            f"unknown routed projection(s): {', '.join(sorted(unknown))}")
    missing = [name for name in _BUNDLE_PROJECTIONS if name not in encoded_by_projection]
    if missing:
        raise KQuantBundleError(f"missing K-quant projection(s): {', '.join(missing)}")

    components: dict[tuple[str, str], np.ndarray] = {}
    bits: dict[str, int] = {}
    kquant_codecs: dict[str, str] = {}
    for name, projection in _BUNDLE_PROJECTIONS.items():
        codec, stacked = stack_kquant_projection_components(
            encoded_by_projection[name],
            projection=projection,
        )
        geometry = KQUANT_GEOMETRY[codec]
        components[(projection, "weight")] = stacked["weight"]
        components[(projection, "scales")] = stacked["scales"]
        bits[projection] = geometry.bits
        kquant_codecs[projection] = codec

    return assemble_layer_bundle(
        components,
        bits,
        codecs={projection: KQUANT_CODEC for projection in PROJECTIONS},
        kquant_codecs=kquant_codecs,
    )
