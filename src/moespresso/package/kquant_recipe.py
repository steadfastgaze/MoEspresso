"""Shared GGUF K-quant recipe helpers.

This module is intentionally pure: it reads GGUF tensor type metadata, maps the
GGUF tensor codecs to mlx-kquant codec names, and checks codec fit before any
mlx-kquant import or model load happens.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from moespresso.package.kquant_format import (
    IMATRIX_REQUIRED_CODECS,
    KQUANT_GEOMETRY,
)
from moespresso.probe.gguf_parse import TENSOR_TYPE_NAMES, read_gguf_metadata


class KQuantRecipeError(ValueError):
    pass


PRODUCER = {"tool": "moespresso.package.kquant_recipe", "version": "1.0.0"}


GGUF_TO_KQUANT_CODEC = {
    "Q8_0": "q8_0",
    "Q4_0": "q4_0",
    "Q4_1": "q4_1",
    "Q5_0": "q5_0",
    "Q5_1": "q5_1",
    "Q2_K": "q2_k",
    "Q3_K": "q3_k",
    "Q4_K": "q4_k",
    "Q5_K": "q5_k",
    "Q6_K": "q6_k",
    "IQ4_NL": "iq4_nl",
    "IQ4_XS": "iq4_xs",
    "IQ3_S": "iq3_s",
    "IQ3_XXS": "iq3_xxs",
    "IQ2_XXS": "iq2_xxs",
    "IQ2_XS": "iq2_xs",
    "IQ2_S": "iq2_s",
    "IQ1_S": "iq1_s",
    "IQ1_M": "iq1_m",
}

_NON_CODEC_GGUF_TYPES = frozenset({
    "F32", "F16", "I8", "I16", "I32", "I64", "F64", "BF16",
})


def _read_recipe_metadata(path: str | Path):
    value = str(path)
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        from moespresso.inventory.hf_inspect import read_remote_gguf_metadata

        return read_remote_gguf_metadata(value)
    return read_gguf_metadata(path)


def read_gguf_kquant_recipe(path: str | Path) -> dict[str, str]:
    """Return `{gguf_tensor_name: kquant_codec}` for K-quant tensors in a GGUF.

    Non-codec float/integer tensors are ignored. Quantized tensor types that
    mlx-kquant does not encode are rejected rather than guessed.
    """
    recipe: dict[str, str] = {}
    metadata = _read_recipe_metadata(path)
    for tensor in metadata.tensor_infos:
        type_name = TENSOR_TYPE_NAMES.get(tensor.type_id)
        if type_name is None:
            raise KQuantRecipeError(
                f"{tensor.name}: unknown GGUF tensor type id {tensor.type_id}")
        codec = GGUF_TO_KQUANT_CODEC.get(type_name)
        if codec is not None:
            recipe[tensor.name] = codec
            continue
        if type_name in _NON_CODEC_GGUF_TYPES:
            continue
        raise KQuantRecipeError(
            f"{tensor.name}: unsupported GGUF codec {type_name!r}")
    return recipe


def read_gguf_tensor_types(path: str | Path) -> dict[str, str]:
    """Return `{gguf_tensor_name: type_name}` for every tensor in a GGUF."""
    out: dict[str, str] = {}
    metadata = _read_recipe_metadata(path)
    for tensor in metadata.tensor_infos:
        type_name = TENSOR_TYPE_NAMES.get(tensor.type_id)
        if type_name is None:
            raise KQuantRecipeError(
                f"{tensor.name}: unknown GGUF tensor type id {tensor.type_id}")
        out[tensor.name] = type_name
    return out


def validate_kquant_target_fit(
    target,
    logical_shape: tuple[int, int] | list[int],
    imatrix_vectors: dict[str, np.ndarray],
) -> None:
    """Validate one target's orientation, row width, and imatrix vector length."""
    if len(logical_shape) != 2:
        raise KQuantRecipeError(
            f"{target.gguf_tensor}: expected 2D [out_features, in_features] "
            f"shape, got {list(logical_shape)}")
    _out_features, in_features = (int(logical_shape[0]), int(logical_shape[1]))
    geometry = KQUANT_GEOMETRY.get(target.codec)
    if geometry is None:
        raise KQuantRecipeError(f"{target.gguf_tensor}: unknown kquant codec {target.codec!r}")
    if in_features % geometry.weights_per_block:
        raise KQuantRecipeError(
            f"{target.gguf_tensor}: codec {target.codec!r} packs "
            f"{geometry.weights_per_block} weights/block, but in_features "
            f"{in_features} is not divisible by {geometry.weights_per_block}")
    if target.codec not in IMATRIX_REQUIRED_CODECS:
        return
    imatrix = imatrix_vectors.get(target.imatrix_key)
    if imatrix is None:
        if not getattr(target, "requires_imatrix", True):
            return
        raise KQuantRecipeError(
            f"{target.gguf_tensor}: missing imatrix vector {target.imatrix_key!r} "
            f"for imatrix-steered codec {target.codec!r}")
    if int(imatrix.shape[0]) != in_features:
        raise KQuantRecipeError(
            f"{target.gguf_tensor}: imatrix length {int(imatrix.shape[0])} "
            f"does not match in_features {in_features}; check tensor orientation")
