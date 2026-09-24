"""Shared writer for one routed IQ_K bundle row."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from moespresso.package.bundle import IQK_CODEC, PROJECTIONS, assemble_layer_bundle
from moespresso.package.iqk_format import (
    IQK_GEOMETRY,
    IQK_LAYOUT_IK_WIRE,
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
    validate_iqk_layout,
)
from moespresso.package.iqk_relayout import pack_stream_major, stream_major_spans


def _shape_pair(value: object, *, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError(f"{field} must be [out_features, in_features]")
    return int(value[0]), int(value[1])


def annotate_expert_input_geometry(
    geometry: dict,
    allocations: Mapping[str, Mapping[str, object]],
) -> dict:
    """Record and validate logical versus stored projection widths.

    Older allocations omit these fields and keep the existing metadata. New
    padded formats declare both shapes so readers never infer logical model
    geometry from an encoded row width.
    """
    projections = geometry.get("projections")
    if not isinstance(projections, dict):
        raise ValueError("expert bundle geometry has no projections")
    for projection in ("gate", "up", "down"):
        allocation = allocations[projection]
        logical_value = allocation.get("logical_shape")
        stored_value = allocation.get("stored_shape")
        if logical_value is None and stored_value is None:
            continue
        logical = _shape_pair(logical_value, field=f"{projection}.logical_shape")
        stored = _shape_pair(stored_value, field=f"{projection}.stored_shape")
        if logical[0] != stored[0] or logical[1] > stored[1]:
            raise ValueError(
                f"{projection}: logical shape {logical} is incompatible with "
                f"stored shape {stored}"
            )
        key = f"{projection}_proj"
        projection_geometry = projections[key]
        component = projection_geometry.get("blocks") or projection_geometry.get("weight")
        component_shape = component.get("shape") if isinstance(component, dict) else None
        if not isinstance(component_shape, list) or len(component_shape) != 2:
            raise ValueError(f"{projection}: encoded component has no matrix shape")
        if int(component_shape[0]) != stored[0]:
            raise ValueError(
                f"{projection}: encoded rows {component_shape[0]} do not match "
                f"stored out_features {stored[0]}"
            )

        codec = projection_geometry.get("codec")
        if codec == IQK_CODEC:
            encoded_in = int(projection_geometry["in_features"])
        elif codec == "kquant":
            from moespresso.package.kquant_format import KQUANT_GEOMETRY

            member = projection_geometry.get("kquant_codec")
            codec_geometry = KQUANT_GEOMETRY.get(member)
            if codec_geometry is None:
                raise ValueError(f"{projection}: unknown K-quant codec {member!r}")
            row_bytes = int(component_shape[1])
            if row_bytes % codec_geometry.bytes_per_block:
                raise ValueError(
                    f"{projection}: encoded row width {row_bytes} does not fit {member}"
                )
            encoded_in = (
                row_bytes // codec_geometry.bytes_per_block
            ) * codec_geometry.weights_per_block
        else:
            raise ValueError(
                f"{projection}: logical/stored geometry is unsupported for codec {codec!r}"
            )
        if encoded_in != stored[1]:
            raise ValueError(
                f"{projection}: encoded in_features {encoded_in} do not match "
                f"stored in_features {stored[1]}"
            )
        padding = stored[1] - logical[1]
        declared_padding = allocation.get("zero_padding", padding)
        if declared_padding != padding:
            raise ValueError(
                f"{projection}: zero_padding {declared_padding!r} does not match {padding}"
            )
        projection_geometry.update(
            {
                "logical_in_features": logical[1],
                "stored_in_features": stored[1],
                "zero_padding": padding,
            }
        )
    return geometry


def annotate_iqk_stream_geometry(geometry: dict) -> dict:
    """Attach exact native-stream spans to a stream-major bundle geometry."""
    projections = geometry.get("projections")
    if not isinstance(projections, dict):
        raise ValueError("expert bundle geometry has no projections")
    layouts = {params.get("layout") for params in projections.values()}
    if layouts != {IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1}:
        raise ValueError(
            "native stream geometry requires one uniform "
            f"{IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1!r} layer"
        )
    for key, params in projections.items():
        member = params.get("iqk_codec")
        blocks = params.get("blocks")
        shape = blocks.get("shape") if isinstance(blocks, dict) else None
        in_features = params.get("in_features")
        if (
            member not in IQK_GEOMETRY
            or not isinstance(shape, list)
            or len(shape) != 2
            or not isinstance(in_features, int)
        ):
            raise ValueError(f"{key}: incomplete IQ_K stream-major geometry")
        params["streams"] = [
            {
                "name": span.name,
                "dtype": span.dtype.name,
                "shape": list(span.shape),
                "offset": span.offset,
                "nbytes": span.nbytes,
            }
            for span in stream_major_spans(member, int(shape[0]), in_features)
        ]
    return geometry


def iqk_bundle_row(
    layer: int,
    expert_index: int,
    allocations: Mapping[str, Mapping[str, object]],
    *,
    expert_loader: Callable[[int, int, str], np.ndarray],
    source_layout: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Load one expert's IQ_K cells and assemble its complete bundle row."""
    components: dict[tuple[str, str], np.ndarray] = {}
    bits: dict[str, int] = {}
    members: dict[str, str] = {}
    layouts: set[str] = set()
    for projection in ("gate", "up", "down"):
        allocation = allocations[projection]
        if allocation.get("format") != "iqk":
            raise ValueError(
                f"layer={layer} {projection}: shared IQ_K writer received "
                f"format {allocation.get('format')!r}"
            )
        member = allocation.get("iqk_codec") or allocation.get("codec")
        codec_geometry = IQK_GEOMETRY.get(member)
        if codec_geometry is None:
            raise ValueError(
                f"unknown IQ_K codec {member!r} for layer={layer} "
                f"projection={projection}"
            )
        blocks = np.ascontiguousarray(
            expert_loader(layer, expert_index, projection),
            dtype=np.uint8,
        )
        if blocks.ndim != 2:
            raise ValueError(
                "IQ_K expert blocks must be 2D [out_features, bytes_per_row], "
                f"got {blocks.ndim}D for layer={layer} projection={projection}"
            )
        in_features = codec_geometry.in_features_for_row_bytes(int(blocks.shape[1]))
        output_layout = validate_iqk_layout(
            str(allocation.get("layout", IQK_LAYOUT_IK_WIRE))
        )
        input_layout = output_layout if source_layout is None else validate_iqk_layout(source_layout)
        if input_layout != output_layout:
            if (
                input_layout == IQK_LAYOUT_IQK_RELAYOUT
                and output_layout == IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1
            ):
                blocks = pack_stream_major(str(member), blocks, in_features)
            else:
                raise ValueError(
                    f"layer={layer} {projection}: unsupported IQ_K layout transform "
                    f"{input_layout!r} -> {output_layout!r}"
                )
        key = f"{projection}_proj"
        components[(key, "blocks")] = blocks[None, ...]
        bits[key] = codec_geometry.bits
        members[key] = str(member)
        layouts.add(output_layout)
    if len(layouts) != 1:
        raise ValueError(f"layer={layer} mixes IQ_K wire layouts {sorted(layouts)}")
    bundle, geometry = assemble_layer_bundle(
        components,
        bits,
        codecs={projection: IQK_CODEC for projection in PROJECTIONS},
        iqk_codecs=members,
        iqk_layout=next(iter(layouts)),
    )
    annotate_expert_input_geometry(geometry, allocations)
    if next(iter(layouts)) == IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1:
        annotate_iqk_stream_geometry(geometry)
    return np.ascontiguousarray(bundle[0]), geometry
