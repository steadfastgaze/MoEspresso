"""Read byte-faithful K-quant expert rows from GGUF tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moespresso.package.kquant_backend import KQuantEncodedWeight
from moespresso.package.kquant_format import KQUANT_GEOMETRY
from moespresso.package.kquant_recipe import GGUF_TO_KQUANT_CODEC
from moespresso.probe.gguf_parse import GGUFBufferParser, TENSOR_TYPE_NAMES


class GGUFKQuantError(ValueError):
    pass


def _read_gguf_directory(path: Path) -> tuple[GGUFBufferParser, int]:
    parser = GGUFBufferParser()
    with open(path, "rb") as f:
        while not parser.is_complete():
            chunk = f.read(1 << 20)
            if not chunk:
                parser.try_parse()
                break
            parser.feed(chunk)
            parser.try_parse()
    if parser.header is None or not parser.is_complete():
        raise GGUFKQuantError(f"failed to parse GGUF tensor directory from {path}")
    data_offset = ((parser.total_consumed() + 31) // 32) * 32
    return parser, data_offset


def _kquant_storage_shape(tensor, codec: str) -> tuple[int, int, int]:
    geometry = KQUANT_GEOMETRY.get(codec)
    if geometry is None:
        raise GGUFKQuantError(f"{tensor.name}: unsupported K-quant codec {codec!r}")
    if tensor.n_dimensions != 3:
        raise GGUFKQuantError(
            f"{tensor.name}: expected stacked expert tensor, got {tensor.n_dimensions}D"
        )
    in_features, out_features, experts = (int(dim) for dim in tensor.dimensions)
    if in_features % geometry.weights_per_block:
        raise GGUFKQuantError(
            f"{tensor.name}: in_features {in_features} is not divisible by "
            f"{geometry.weights_per_block} for {codec}"
        )
    packed_cols = (
        in_features // geometry.weights_per_block * geometry.bytes_per_block
    )
    return experts, out_features, packed_cols


class GGUFKQuantExpertReader:
    """Reusable byte reader for stacked GGUF routed expert tensors.

    Parsing a GGUF tensor directory is not free. A package build loads tens
    of thousands of expert rows, so the directory must be parsed once and reused.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        parser, data_offset = _read_gguf_directory(self.path)
        self._data_offset = data_offset
        self._tensors = {tensor.name: tensor for tensor in parser.tensor_infos}

    def load_expert_weight(
        self,
        target,
        *,
        expert_index: int,
    ) -> KQuantEncodedWeight:
        """Load one encoded `[out_features, bytes_per_row]` expert row from GGUF.

        `target` is a routed expert recipe target carrying `gguf_tensor` and
        `codec` fields; the DS4 and Qwen expert targets both qualify. GGUF stores
        routed experts as one stacked tensor per layer/projection, with dimensions
        `[in_features, out_features, experts]`. The byte payload is read as
        `[experts, out_features, packed_cols]`, matching the layout already
        proven by the Q3 GGUF replay harness.
        """
        tensor = self._tensors.get(target.gguf_tensor)
        if tensor is None:
            raise GGUFKQuantError(f"missing GGUF tensor {target.gguf_tensor!r}")
        type_name = TENSOR_TYPE_NAMES.get(tensor.type_id)
        codec = GGUF_TO_KQUANT_CODEC.get(type_name)
        if codec is None:
            raise GGUFKQuantError(
                f"{target.gguf_tensor}: expected K-quant tensor, got {type_name!r}"
            )
        if codec != target.codec:
            raise GGUFKQuantError(
                f"{target.gguf_tensor}: expected {target.codec}, got {codec}"
            )
        experts, out_features, packed_cols = _kquant_storage_shape(tensor, codec)
        expert = int(expert_index)
        if expert < 0 or expert >= experts:
            raise GGUFKQuantError(
                f"{target.gguf_tensor}: expert {expert} out of range [0, {experts})"
            )
        row_bytes = out_features * packed_cols
        offset = self._data_offset + int(tensor.offset) + expert * row_bytes
        with open(self.path, "rb") as f:
            f.seek(offset)
            raw = f.read(row_bytes)
        if len(raw) != row_bytes:
            raise GGUFKQuantError(
                f"{target.gguf_tensor}: short read {len(raw)} of {row_bytes} bytes "
                f"for expert {expert}"
            )
        return KQuantEncodedWeight(
            codec=codec,
            weight=np.frombuffer(raw, dtype=np.uint8)
            .reshape(out_features, packed_cols)
            .copy(),
            scales=np.zeros((1,), dtype=np.uint8),
        )


def load_gguf_kquant_expert_weight(
    path: str | Path,
    target,
    *,
    expert_index: int,
) -> KQuantEncodedWeight:
    """One-off wrapper for tests and probes; package builds should reuse a reader."""
    return GGUFKQuantExpertReader(path).load_expert_weight(
        target,
        expert_index=expert_index,
    )
