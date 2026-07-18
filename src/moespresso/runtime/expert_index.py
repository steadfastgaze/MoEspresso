"""Expert byte-offset index for SSD-streamed MoE inference (bundle format).

Maps (layer, expert, projection, component) -> an exact byte range in a package
shard, so the miss-loader can `pread` expert bytes without faulting whole
tensors. In the current format, one routed layer stores ONE bundle tensor
(`...switch_mlp.experts.tq_bundle`, uint8 `[n_experts, row_bytes]`) whose row e
concatenates expert e's full payload, so the index can also hand out the WHOLE
row as a single range (`locate_row`): one pread per missed expert instead of
six.

The within-row geometry travels in each shard's safetensors `__metadata__`
(package/bundle.py is the schema's single source of truth), so this stays
header-only and import-light: safetensors headers + metadata JSON, no weight
bytes, no mlx, no jang, no manifest file.

Legacy packages (stacked `...tq_packed/tq_norms/tq_bits` tensors) are not
readable: there is no compatibility path; they fail loud here with a
re-convert message, never a silent miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from moespresso.inventory.safetensors_header import (
    read_headers_with_offsets,
    read_shard_metadata,
)
from moespresso.package.bundle import (
    COMPONENTS,
    KQUANT_CODEC,
    METADATA_KEY,
    MXFP4_CODEC,
    PROJECTIONS,
    TQ_CODEC,
    BundleFormatError,
    decode_bundle_metadata,
    row_order_for_codecs,
)

__all__ = [
    "PROJECTIONS", "COMPONENTS", "ExpertByteRange", "ProjectionGeometry",
    "ExpertIndex", "StackedLayoutError", "build_expert_index",
]

# bundle key: prefixed MoE layers or DS4 root `layers.<L>.ffn.experts.tq_bundle`
_BUNDLE_KEY = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\..*experts\.tq_bundle$")
# future routed bundle suffixes that must not be ignored by the TQ-only index.
_UNSUPPORTED_BUNDLE_KEY = re.compile(
    r"(?:^|\.)layers\.\d+\..*experts\.(?P<suffix>mxfp4_bundle)$")
# legacy stacked keys, matched only to fail loud with a useful message.
_STACKED_KEY = re.compile(
    r"(?:^|\.)layers\.\d+\..*switch_mlp\."
    r"(?:gate_proj|up_proj|down_proj)\.tq_(?:packed|norms|bits)$")


class StackedLayoutError(ValueError):
    """The package uses the legacy stacked expert layout (no longer readable)."""


@dataclass(frozen=True)
class ExpertByteRange:
    """An exact, contiguous byte range for one (layer, expert[, proj, component])."""
    shard: str
    offset: int      # absolute byte offset in the shard file
    nbytes: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ProjectionGeometry:
    """Shape facts for one routed projection."""
    codec: str
    out_features: int
    packed_cols: int
    bits: int
    packed_dtype: str
    norms_dtype: str | None = None
    scales_dtype: str | None = None
    kquant_codec: str | None = None
    group_size: int | None = None
    bytes_per_block: int | None = None
    weights_per_block: int | None = None


@dataclass(frozen=True)
class _LayerBundle:
    """One layer's bundle tensor + its validated within-row geometry."""
    shard: str
    tensor_start: int       # absolute byte where the bundle's data begins
    num_experts: int
    row_bytes: int
    total_bytes: int        # whole-tensor bytes (== num_experts * row_bytes)
    # (projection, component) -> {"offset", "nbytes", "shape", "dtype"}
    components: dict[tuple[str, str], dict]
    bits: dict[str, int]    # projection -> bits
    codecs: dict[str, str]  # projection -> codec
    projections: dict[str, dict]

    def _check_expert(self, expert: int) -> None:
        if expert < 0 or expert >= self.num_experts:
            raise IndexError(
                f"expert {expert} out of range [0, {self.num_experts})")

    def row_range(self, expert: int) -> ExpertByteRange:
        self._check_expert(expert)
        return ExpertByteRange(
            shard=self.shard,
            offset=self.tensor_start + expert * self.row_bytes,
            nbytes=self.row_bytes,
            shape=(self.row_bytes,),
            dtype="U8",
        )

    def component_range(self, expert: int, projection: str,
                        component: str) -> ExpertByteRange:
        self._check_expert(expert)
        c = self.components[(projection, component)]
        return ExpertByteRange(
            shard=self.shard,
            offset=self.tensor_start + expert * self.row_bytes + c["offset"],
            nbytes=c["nbytes"],
            shape=tuple(c["shape"]),
            dtype=c["dtype"],
        )


class ExpertIndex:
    """Per-layer bundle byte geometry with per-component and whole-row ranges."""

    def __init__(self, bundles: dict[int, _LayerBundle],
                 num_layers: int, num_experts: int):
        self._bundles = bundles
        self.num_layers = num_layers
        self.num_experts = num_experts

    def _bundle(self, layer: int) -> _LayerBundle:
        b = self._bundles.get(layer)
        if b is None:
            raise KeyError(f"no expert bundle for layer={layer}")
        return b

    def has_projection(self, *, layer: int, projection: str) -> bool:
        b = self._bundles.get(layer)
        return b is not None and projection in b.codecs

    def locate(self, *, layer: int, expert: int, projection: str,
               component: str = "packed") -> ExpertByteRange:
        b = self._bundle(layer)
        if (projection, component) not in b.components:
            raise KeyError(
                f"no component for layer={layer} proj={projection} "
                f"component={component}")
        return b.component_range(expert, projection, component)

    def locate_row(self, *, layer: int, expert: int) -> ExpertByteRange:
        """The whole bundle row for one expert: the one-pread miss load."""
        return self._bundle(layer).row_range(expert)

    def row_components(self, *, layer: int) -> dict[tuple[str, str], dict]:
        """Within-row geometry {(proj, comp) -> {offset, nbytes, shape, dtype}}.

        Offsets are relative to the row start (for splitting a staged row),
        unlike locate()'s absolute file offsets.
        """
        return dict(self._bundle(layer).components)

    def row_bytes(self, *, layer: int) -> int:
        return self._bundle(layer).row_bytes

    def bits(self, *, layer: int, projection: str) -> int:
        b = self._bundle(layer)
        if projection not in b.bits:
            raise KeyError(f"no bits for layer={layer} proj={projection}")
        return b.bits[projection]

    def codec(self, *, layer: int, projection: str) -> str:
        b = self._bundle(layer)
        if projection not in b.codecs:
            raise KeyError(f"no codec for layer={layer} proj={projection}")
        return b.codecs[projection]

    def components_for_projection(self, *, layer: int, projection: str) -> tuple[str, ...]:
        b = self._bundle(layer)
        return tuple(
            comp
            for proj, comp in row_order_for_codecs(b.codecs)
            if proj == projection
        )

    def geometry(self, *, layer: int, projection: str) -> ProjectionGeometry:
        b = self._bundle(layer)
        codec = self.codec(layer=layer, projection=projection)
        weight_component = "weight" if codec == KQUANT_CODEC else "packed"
        packed = b.components[(projection, weight_component)]
        if len(packed["shape"]) != 2:
            raise ValueError(
                f"layer={layer} {projection}.{weight_component}: expected per-expert 2D, "
                f"got {packed['shape']}")
        out_features = packed["shape"][0]
        norms_dtype = None
        scales_dtype = None
        kquant_codec = None
        group_size = None
        bytes_per_block = None
        weights_per_block = None
        if codec == TQ_CODEC:
            norms = b.components[(projection, "norms")]
            if len(norms["shape"]) != 1:
                raise ValueError(
                    f"layer={layer} {projection}.norms: expected per-expert 1D, "
                    f"got {norms['shape']}")
            if norms["shape"][0] != out_features:
                raise ValueError(
                    f"layer={layer} {projection}: packed rows {out_features} "
                    f"!= norms rows {norms['shape'][0]}")
            norms_dtype = norms["dtype"]
        elif codec == MXFP4_CODEC:
            scales = b.components[(projection, "scales")]
            if len(scales["shape"]) != 2:
                raise ValueError(
                    f"layer={layer} {projection}.scales: expected per-expert 2D, "
                    f"got {scales['shape']}")
            if scales["shape"][0] != out_features:
                raise ValueError(
                    f"layer={layer} {projection}: packed rows {out_features} "
                    f"!= scales rows {scales['shape'][0]}")
            scales_dtype = scales["dtype"]
        elif codec == KQUANT_CODEC:
            scales = b.components[(projection, "scales")]
            if len(scales["shape"]) != 1:
                raise ValueError(
                    f"layer={layer} {projection}.scales: expected per-expert 1D, "
                    f"got {scales['shape']}")
            if scales["shape"][0] != 1:
                raise ValueError(
                    f"layer={layer} {projection}: kquant scales is a placeholder "
                    f"with per-expert shape [1], got {scales['shape']}")
            scales_dtype = scales["dtype"]
            proj_geo = b.projections[projection]
            kquant_codec = proj_geo["kquant_codec"]
            group_size = proj_geo["group_size"]
            bytes_per_block = proj_geo["bytes_per_block"]
            weights_per_block = proj_geo["weights_per_block"]
        else:
            raise ValueError(f"layer={layer} {projection}: unsupported codec {codec!r}")
        return ProjectionGeometry(
            codec=codec,
            out_features=out_features,
            packed_cols=packed["shape"][1],
            bits=self.bits(layer=layer, projection=projection),
            packed_dtype=packed["dtype"],
            norms_dtype=norms_dtype,
            scales_dtype=scales_dtype,
            kquant_codec=kquant_codec,
            group_size=group_size,
            bytes_per_block=bytes_per_block,
            weights_per_block=weights_per_block,
        )

    def num_layers_indexed(self) -> int:
        """Distinct layers that have routed-expert bundles."""
        return len(self._bundles)

    def layers_indexed(self) -> tuple[int, ...]:
        """Layer indexes that have routed-expert bundles."""
        return tuple(sorted(self._bundles))

    def num_expert_slots(self) -> int:
        """Total locatable (layer, expert) pairs = layers * experts per layer."""
        return self.num_layers_indexed() * self.num_experts

    def validate(self) -> list[str]:
        """Cheap structural checks; returns a list of problems ([] == ok).

        Verifies, for every layer bundle: per-expert rows tile the tensor
        exactly, component ranges tile each row exactly in ROW_ORDER (no
        padding), and first/last expert rows stay inside the tensor.
        """
        problems: list[str] = []
        for layer, b in sorted(self._bundles.items()):
            if b.num_experts != self.num_experts:
                problems.append(
                    f"layer={layer}: num_experts {b.num_experts} "
                    f"!= {self.num_experts}")
            if b.row_bytes * b.num_experts != b.total_bytes:
                problems.append(
                    f"layer={layer}: rows {b.row_bytes}*{b.num_experts} "
                    f"!= total {b.total_bytes}")
            offset = 0
            for proj, comp in row_order_for_codecs(b.codecs):
                c = b.components.get((proj, comp))
                if c is None:
                    problems.append(f"layer={layer} {proj}.{comp}: missing")
                    continue
                if c["offset"] != offset:
                    problems.append(
                        f"layer={layer} {proj}.{comp}: offset {c['offset']} "
                        f"!= expected {offset}")
                offset += c["nbytes"]
            if offset != b.row_bytes:
                problems.append(
                    f"layer={layer}: components end at {offset} "
                    f"!= row_bytes {b.row_bytes}")
            first = b.row_range(0)
            last = b.row_range(b.num_experts - 1)
            if first.offset < b.tensor_start:
                problems.append(f"layer={layer}: first row before tensor start")
            if last.offset + last.nbytes != b.tensor_start + b.total_bytes:
                problems.append(f"layer={layer}: last row past tensor end")
        return problems


def build_expert_index(package_dir: str | Path) -> ExpertIndex:
    """Build the expert byte index from a package's headers + shard metadata.

    Scans every shard for `...experts.tq_bundle` tensors, pairs each
    with its layer's geometry from the shard's `__metadata__` (strictly
    validated by package/bundle.py), and records absolute offsets. Weight
    tensors themselves are never read. Legacy stacked packages fail loud.
    """
    package_dir = Path(package_dir)
    bundles: dict[int, _LayerBundle] = {}
    experts_seen: set[int] = set()
    stacked_keys_seen: list[str] = []

    for shard in sorted(package_dir.glob("model-*.safetensors")):
        headers = read_headers_with_offsets(shard)
        meta_text = read_shard_metadata(shard).get(METADATA_KEY)
        geometries: dict[int, dict] = {}
        if meta_text is not None:
            try:
                geometries = decode_bundle_metadata(meta_text)
            except BundleFormatError as e:
                raise ValueError(f"{shard.name}: {e}") from e

        matched_layers: set[int] = set()
        for th in headers:
            if _STACKED_KEY.search(th.name):
                stacked_keys_seen.append(th.name)
                continue
            unsupported = _UNSUPPORTED_BUNDLE_KEY.search(th.name)
            if unsupported is not None:
                suffix = unsupported.group("suffix")
                raise ValueError(
                    f"{shard.name}: unsupported routed-expert bundle suffix "
                    f"{suffix!r} in {th.name}; re-convert or use a runtime that "
                    "declares support for this expert bundle format")
            m = _BUNDLE_KEY.search(th.name)
            if m is None:
                continue
            layer = int(m.group("layer"))
            geo = geometries.get(layer)
            if geo is None:
                raise ValueError(
                    f"{shard.name}: bundle tensor {th.name} has no layer-{layer} "
                    f"entry in the shard's {METADATA_KEY} metadata")
            if layer in bundles:
                raise ValueError(
                    f"duplicate bundle for layer {layer} "
                    f"({bundles[layer].shard} and {shard.name})")
            num_experts, row_bytes = geo["num_experts"], geo["row_bytes"]
            if tuple(th.shape) != (num_experts, row_bytes) or th.dtype != "U8":
                raise ValueError(
                    f"{th.name}: header {th.dtype} {th.shape} does not match "
                    f"metadata uint8 ({num_experts}, {row_bytes})")
            codecs = {
                proj: geo["projections"][proj].get("codec", TQ_CODEC)
                for proj in PROJECTIONS
            }
            components = {
                (proj, comp): geo["projections"][proj][comp]
                for proj, comp in row_order_for_codecs(codecs)
            }
            bits = {proj: geo["projections"][proj]["bits"] for proj in PROJECTIONS}
            bundles[layer] = _LayerBundle(
                shard=th.shard,
                tensor_start=th.header_size + th.begin,
                num_experts=num_experts,
                row_bytes=row_bytes,
                total_bytes=th.end - th.begin,
                components=components,
                bits=bits,
                codecs=codecs,
                projections=geo["projections"],
            )
            matched_layers.add(layer)
            experts_seen.add(num_experts)

        unmatched = set(geometries) - matched_layers - set(bundles)
        if unmatched:
            raise ValueError(
                f"{shard.name}: {METADATA_KEY} metadata declares layer(s) "
                f"{sorted(unmatched)} but the shard has no matching bundle tensor")

    if stacked_keys_seen and not bundles:
        raise StackedLayoutError(
            f"{package_dir} uses the legacy STACKED expert layout "
            f"(e.g. {stacked_keys_seen[0]}). There is no compatibility path; "
            f"re-convert the package with the current moespresso-convert to "
            f"produce the bundle format.")
    if not bundles:
        raise ValueError(f"no routed-expert bundle tensors found in {package_dir}")
    if stacked_keys_seen:
        raise ValueError(
            f"{package_dir} mixes bundle and stacked expert tensors "
            f"(e.g. {stacked_keys_seen[0]}): corrupt or half-converted package")
    if len(experts_seen) != 1:
        raise ValueError(
            f"inconsistent num_experts across routed tensors: {sorted(experts_seen)}")
    num_experts = next(iter(experts_seen))
    return ExpertIndex(bundles=bundles,
                       num_layers=len(bundles), num_experts=num_experts)
