"""Logical routed-expert source adapter for DeepSeek-V4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from moespresso.probe import weight_io
from moespresso.probe.deepseek_v4.codec import load_dequantized_fp4, load_storage_tensor


class DeepseekV4ExpertAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class ExpertProjectionSource:
    layer: int
    expert_index: int
    projection: str
    weight_name: str
    scale_name: str
    logical_shape: tuple[int, int]


class DecodedExpertGroup:
    """Expose DS4 separate expert tensors as logical gate/up/down projections."""

    def __init__(
        self,
        model_dir: Path,
        sources: dict[tuple[int, int, str], ExpertProjectionSource],
        *,
        fp4_block: int = 32,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.sources = dict(sources)
        self.fp4_block = fp4_block
        self.catalog = weight_io.scan_offsets(self.model_dir)

    @classmethod
    def from_inventory(
        cls,
        inventory: dict,
        model_dir: Path,
        *,
        fp4_block: int = 32,
    ) -> "DecodedExpertGroup":
        weights: dict[tuple[int, int, str], dict] = {}
        scales: dict[tuple[int, int, str], dict] = {}
        for entry in inventory.get("tensors", []):
            if entry.get("kind") not in {"expert_source", "codec_scale"}:
                continue
            if "expert_index" not in entry:
                continue
            key = (
                int(entry["layer_index"]),
                int(entry["expert_index"]),
                str(entry["projection"]),
            )
            if entry["kind"] == "expert_source":
                weights[key] = entry
            else:
                scales[key] = entry

        required = {"gate", "up", "down"}
        expert_ids = {
            (layer, expert)
            for layer, expert, _projection in set(weights) | set(scales)
        }
        for layer, expert in sorted(expert_ids):
            weight_projections = {
                projection
                for src_layer, src_expert, projection in weights
                if src_layer == layer and src_expert == expert
            }
            scale_projections = {
                projection
                for src_layer, src_expert, projection in scales
                if src_layer == layer and src_expert == expert
            }
            missing_weights = sorted(required - weight_projections)
            missing_scales = sorted(required - scale_projections)
            if missing_weights or missing_scales:
                details = []
                if missing_weights:
                    details.append(f"missing weight(s) {missing_weights}")
                if missing_scales:
                    details.append(f"missing scale(s) {missing_scales}")
                raise DeepseekV4ExpertAdapterError(
                    "incomplete DS4 expert source triplet "
                    f"layer={layer} expert={expert}: " + "; ".join(details))

        sources: dict[tuple[int, int, str], ExpertProjectionSource] = {}
        for key, weight_entry in weights.items():
            scale_entry = scales.get(key)
            if scale_entry is None:
                raise DeepseekV4ExpertAdapterError(
                    f"missing scale for DS4 expert source {weight_entry['source_name']}")
            rows, packed_cols = weight_entry["shape"]
            sources[key] = ExpertProjectionSource(
                layer=key[0],
                expert_index=key[1],
                projection=key[2],
                weight_name=weight_entry["source_name"],
                scale_name=scale_entry["source_name"],
                logical_shape=(int(rows), int(packed_cols) * 2),
            )
        return cls(model_dir, sources, fp4_block=fp4_block)

    def layers(self) -> list[int]:
        return sorted({layer for layer, _expert, _projection in self.sources})

    def experts(self, layer: int) -> list[int]:
        return sorted({
            expert for src_layer, expert, _projection in self.sources
            if src_layer == layer
        })

    def projections(self, layer: int) -> list[str]:
        order = {"gate": 0, "up": 1, "down": 2}
        projs = {
            projection for src_layer, _expert, projection in self.sources
            if src_layer == layer
        }
        return sorted(projs, key=lambda p: order.get(p, 99))

    def logical_shape(self, *, layer: int, expert_index: int, projection: str) -> tuple[int, int]:
        return self.sources[(layer, expert_index, projection)].logical_shape

    def decode(
        self,
        *,
        layer: int,
        expert_index: int,
        projection: str,
        out_dtype=np.float16,
    ) -> np.ndarray:
        source = self.sources.get((layer, expert_index, projection))
        if source is None:
            raise DeepseekV4ExpertAdapterError(
                f"missing DS4 expert source layer={layer} expert={expert_index} "
                f"projection={projection}")
        weight_header = self.catalog[source.weight_name]
        scale_header = self.catalog[source.scale_name]
        return load_dequantized_fp4(
            self.model_dir,
            weight_header,
            scale_header,
            fp4_block=self.fp4_block,
            out_dtype=out_dtype,
        )

    def storage(
        self,
        *,
        layer: int,
        expert_index: int,
        projection: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read one DS4 expert/projection as stored FP4 bytes plus UE8M0 scales."""
        source = self.sources.get((layer, expert_index, projection))
        if source is None:
            raise DeepseekV4ExpertAdapterError(
                f"missing DS4 expert source layer={layer} expert={expert_index} "
                f"projection={projection}")
        weight_header = self.catalog[source.weight_name]
        scale_header = self.catalog[source.scale_name]
        return (
            load_storage_tensor(self.model_dir, weight_header),
            load_storage_tensor(self.model_dir, scale_header),
        )

    def iter_projection(
        self,
        *,
        layer: int,
        projection: str,
        expert_indices: list[int] | None = None,
        out_dtype=np.float16,
    ):
        experts = self.experts(layer) if expert_indices is None else list(expert_indices)
        for expert_index in experts:
            yield expert_index, self.decode(
                layer=layer,
                expert_index=expert_index,
                projection=projection,
                out_dtype=out_dtype,
            )
