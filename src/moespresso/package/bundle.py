"""Per-expert bundle layout for routed experts (the streaming format).

One bundle tensor per routed layer replaces the six stacked expert tensors:
`...switch_mlp.experts.tq_bundle` is uint8 `[n_experts, row_bytes]`, where row
e concatenates expert e's full payload. TQ projections store `packed + norms`;
source-mxfp4 projections store `packed + scales`. The projection codec is
declared in the bundle metadata, so readers never infer it from bit width.

Row stride is the exact component sum, no padding: plain pread needs no
alignment and direct IO is intentionally out of scope.

The geometry contract travels in each shard's safetensors `__metadata__` under
METADATA_KEY as versioned JSON (string values are all safetensors allows), so
the expert index stays header-only: no weight reads, no manifest file. This
module is the single source of truth for that schema (the writer, the index,
the correctness ladder, and the test fixtures all build/parse through here so
the offset math can never drift apart).

Pure numpy + stdlib: no mlx, no jang (the index imports this and must stay
import-light).
"""

from __future__ import annotations

import json

import numpy as np

from moespresso.package.kquant_format import KQUANT_GEOMETRY

METADATA_KEY = "expert_bundles"
SCHEMA_VERSION = 1
BUNDLE_KEY_SUFFIX = "tq_bundle"

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
TQ_CODEC = "tq"
MXFP4_CODEC = "mxfp4"
KQUANT_CODEC = "kquant"
COMPONENTS = ("packed", "norms", "scales", "weight")
TQ_COMPONENTS = ("packed", "norms")
MXFP4_COMPONENTS = ("packed", "scales")
KQUANT_COMPONENTS = ("weight", "scales")

# The fixed within-row component order. Readers must derive offsets from the
# metadata, never from this tuple: it exists so the writer is deterministic
# and the metadata validator can require exact, gap-free tiling.
ROW_ORDER = (
    ("gate_proj", "packed"), ("gate_proj", "norms"),
    ("up_proj", "packed"), ("up_proj", "norms"),
    ("down_proj", "packed"), ("down_proj", "norms"),
)

# safetensors dtype token <-> numpy dtype, per component kind. TQ packed words
# and mxfp4 packed words are uint32; TQ norms are float16; mxfp4 scales are
# uint8 UE8M0.
_COMPONENT_DTYPES = {
    "packed": ("U32", np.uint32),
    "norms": ("F16", np.float16),
    "scales": ("U8", np.uint8),
    "weight": ("U8", np.uint8),
}
NP_DTYPES = {"U32": np.uint32, "F16": np.float16, "U8": np.uint8}


class BundleFormatError(ValueError):
    pass


def _components_for_codec(codec: str) -> tuple[str, ...]:
    if codec == TQ_CODEC:
        return TQ_COMPONENTS
    if codec == MXFP4_CODEC:
        return MXFP4_COMPONENTS
    if codec == KQUANT_CODEC:
        return KQUANT_COMPONENTS
    raise BundleFormatError(f"unsupported expert codec {codec!r}")


def _component_ndim(codec: str, component: str) -> int:
    if codec == TQ_CODEC:
        return 2 if component == "norms" else 3
    if codec == MXFP4_CODEC:
        return 3
    if codec == KQUANT_CODEC:
        return 2 if component == "scales" else 3
    raise BundleFormatError(f"unsupported expert codec {codec!r}")


def row_order_for_codecs(codecs: dict[str, str] | None = None) -> tuple[tuple[str, str], ...]:
    """Return the within-row component order for projection codecs."""
    codecs = codecs or {proj: TQ_CODEC for proj in PROJECTIONS}
    if sorted(codecs) != sorted(PROJECTIONS):
        raise BundleFormatError(
            f"codecs must cover exactly {PROJECTIONS}, got {sorted(codecs)}")
    out: list[tuple[str, str]] = []
    for proj in PROJECTIONS:
        for comp in _components_for_codec(codecs[proj]):
            out.append((proj, comp))
    return tuple(out)


def repack_ds4_fp4_as_mxfp4_uint32(packed: np.ndarray) -> np.ndarray:
    """Repack DeepSeek-V4 I8 FP4 bytes into MLX mxfp4 uint32 words."""
    packed_u8 = np.ascontiguousarray(packed).view(np.uint8)
    if packed_u8.ndim != 2:
        raise BundleFormatError(
            f"source FP4 packed bytes must be 2D, got {packed_u8.ndim}D")
    out_dim, source_byte_cols = packed_u8.shape
    in_features = source_byte_cols * 2
    if in_features % 8:
        raise BundleFormatError(
            f"in_features {in_features} is not divisible by 8")
    return (
        packed_u8.reshape(out_dim, in_features // 8, 4)
        .copy()
        .view(np.uint32)
        .reshape(out_dim, in_features // 8)
    )


def ds4_source_to_mxfp4_components(
    packed_i8: np.ndarray,
    scales_u8: np.ndarray,
) -> dict[str, np.ndarray]:
    """One DS4 FP4 expert/projection -> live mxfp4 bundle components."""
    packed = repack_ds4_fp4_as_mxfp4_uint32(packed_i8)
    scales = np.ascontiguousarray(scales_u8).view(np.uint8)
    _validate_mxfp4_packed_scales_shape("source", packed[None, ...], scales[None, ...])
    return {"packed": packed, "scales": scales}


def assemble_layer_bundle(
    components: dict[tuple[str, str], np.ndarray],
    bits: dict[str, int],
    codecs: dict[str, str] | None = None,
    kquant_codecs: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Stacked per-projection arrays -> (bundle uint8 [N, row_bytes], geometry).

    `components` maps every (projection, component) required by the projection's
    codec to its stacked array. TQ uses packed `[n_experts, out, packed_cols]`
    uint32 and norms `[n_experts, out]` float16. mxfp4 uses packed
    `[n_experts, out, in/8]` uint32 and scales `[n_experts, out, in/32]` uint8.
    The returned geometry dict is one layer's entry for shard metadata and
    records, per component, the exact within-row byte range + per-expert shape
    + dtype.
    """
    codecs = codecs or {p: TQ_CODEC for p in PROJECTIONS}
    kquant_codecs = kquant_codecs or {}
    order = row_order_for_codecs(codecs)
    missing = [k for k in order if k not in components]
    extra = [k for k in components if k not in order]
    if missing or extra:
        raise BundleFormatError(
            f"components do not match projection codecs; missing={missing} extra={extra}")
    if sorted(bits) != sorted(PROJECTIONS):
        raise BundleFormatError(
            f"bits must cover exactly {PROJECTIONS}, got {sorted(bits)}")

    n_experts = None
    for (proj, comp), arr in components.items():
        codec = codecs.get(proj)
        want_token, want_np = _COMPONENT_DTYPES[comp]
        want_ndim = _component_ndim(codec, comp)
        if arr.ndim != want_ndim or arr.dtype != want_np:
            raise BundleFormatError(
                f"{proj}.{comp}: expected {want_ndim}D {np.dtype(want_np).name}, "
                f"got {arr.ndim}D {arr.dtype}")
        if n_experts is None:
            n_experts = arr.shape[0]
        elif arr.shape[0] != n_experts:
            raise BundleFormatError(
                f"{proj}.{comp}: num_experts {arr.shape[0]} != {n_experts}")
    for proj in PROJECTIONS:
        b = bits[proj]
        codec = codecs[proj]
        if codec == TQ_CODEC:
            if components[(proj, "packed")].shape[1] != components[(proj, "norms")].shape[1]:
                raise BundleFormatError(
                    f"{proj}: packed rows {components[(proj, 'packed')].shape[1]} != "
                    f"norms rows {components[(proj, 'norms')].shape[1]}")
            if not isinstance(b, int) or not (1 <= b <= 8):
                raise BundleFormatError(f"{proj}: bits {b!r} outside [1, 8]")
        elif codec == MXFP4_CODEC:
            if b != 4:
                raise BundleFormatError(f"{proj}: mxfp4 bits must be 4, got {b!r}")
            _validate_mxfp4_packed_scales_shape(
                proj,
                components[(proj, "packed")],
                components[(proj, "scales")],
            )
        elif codec == KQUANT_CODEC:
            _validate_kquant_component_shape(
                proj,
                components[(proj, "weight")],
                components[(proj, "scales")],
                bits=b,
                kquant_codec=kquant_codecs.get(proj),
            )
        else:
            raise BundleFormatError(f"{proj}: unsupported codec {codec!r}")

    projections: dict[str, dict] = {
        p: {"codec": codecs[p], "bits": int(bits[p])}
        for p in PROJECTIONS
    }
    for p in PROJECTIONS:
        if codecs[p] == MXFP4_CODEC:
            projections[p]["group_size"] = 32
            projections[p]["scale_dtype"] = "ue8m0"
        elif codecs[p] == KQUANT_CODEC:
            kcodec = kquant_codecs[p]
            geometry = KQUANT_GEOMETRY[kcodec]
            projections[p]["kquant_codec"] = kcodec
            projections[p]["group_size"] = geometry.group_size
            projections[p]["bytes_per_block"] = geometry.bytes_per_block
            projections[p]["weights_per_block"] = geometry.weights_per_block
    offset = 0
    for proj, comp in order:
        arr = components[(proj, comp)]
        token, _ = _COMPONENT_DTYPES[comp]
        per_expert = arr.reshape(n_experts, -1)
        nbytes = per_expert.shape[1] * arr.dtype.itemsize
        projections[proj][comp] = {
            "offset": offset,
            "nbytes": nbytes,
            "shape": list(arr.shape[1:]),
            "dtype": token,
        }
        offset += nbytes
    row_bytes = offset

    bundle = np.empty((n_experts, row_bytes), dtype=np.uint8)
    for proj, comp in order:
        c = projections[proj][comp]
        arr = np.ascontiguousarray(components[(proj, comp)])
        bundle[:, c["offset"]:c["offset"] + c["nbytes"]] = (
            arr.reshape(n_experts, -1).view(np.uint8))

    geometry = {
        "num_experts": int(n_experts),
        "row_bytes": int(row_bytes),
        "projections": projections,
    }
    return bundle, geometry


def encode_bundle_metadata(layers: dict[int, dict]) -> str:
    """Per-layer geometry dicts -> the shard `__metadata__[METADATA_KEY]` string."""
    return json.dumps(
        {"version": SCHEMA_VERSION,
         "layers": {str(layer): geo for layer, geo in sorted(layers.items())}},
        sort_keys=True)


def decode_bundle_metadata(text: str) -> dict[int, dict]:
    """Parse + strictly validate the shard metadata. Raises BundleFormatError.

    Validation is exact-tiling: every component range must follow ROW_ORDER
    back-to-back and the last must end at row_bytes. The format has no padding,
    so any gap/overlap means writer/reader drift: fail loud, never guess.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise BundleFormatError(f"unparseable {METADATA_KEY} metadata: {e}") from e
    if not isinstance(data, dict) or data.get("version") != SCHEMA_VERSION:
        raise BundleFormatError(
            f"unsupported {METADATA_KEY} version {data.get('version')!r} "
            f"(expected {SCHEMA_VERSION})")
    layers_raw = data.get("layers")
    if not isinstance(layers_raw, dict) or not layers_raw:
        raise BundleFormatError(f"{METADATA_KEY} metadata has no layers")

    layers: dict[int, dict] = {}
    for layer_key, geo in layers_raw.items():
        try:
            layer = int(layer_key)
        except (TypeError, ValueError) as e:
            raise BundleFormatError(f"bad layer key {layer_key!r}") from e
        layers[layer] = _validate_layer_geometry(layer, geo)
    return layers


def _validate_layer_geometry(layer: int, geo: dict) -> dict:
    where = f"layer {layer}"
    if not isinstance(geo, dict):
        raise BundleFormatError(f"{where}: geometry is not an object")
    num_experts = geo.get("num_experts")
    row_bytes = geo.get("row_bytes")
    if not isinstance(num_experts, int) or num_experts < 1:
        raise BundleFormatError(f"{where}: bad num_experts {num_experts!r}")
    if not isinstance(row_bytes, int) or row_bytes < 1:
        raise BundleFormatError(f"{where}: bad row_bytes {row_bytes!r}")
    projections = geo.get("projections")
    if (not isinstance(projections, dict)
            or sorted(projections) != sorted(PROJECTIONS)):
        raise BundleFormatError(
            f"{where}: projections must be exactly {PROJECTIONS}, "
            f"got {sorted(projections) if isinstance(projections, dict) else projections!r}")

    codecs = {}
    for proj in PROJECTIONS:
        codec = projections[proj].get("codec", TQ_CODEC)
        _components_for_codec(codec)
        codecs[proj] = codec

    expect_offset = 0
    for proj, comp in row_order_for_codecs(codecs):
        p = projections[proj]
        b = p.get("bits")
        codec = p.get("codec", TQ_CODEC)
        if codec == TQ_CODEC:
            if not isinstance(b, int) or not (1 <= b <= 8):
                raise BundleFormatError(f"{where} {proj}: bits {b!r} outside [1, 8]")
        elif codec == MXFP4_CODEC:
            if b != 4 or p.get("group_size") != 32 or p.get("scale_dtype") != "ue8m0":
                raise BundleFormatError(f"{where} {proj}: bad mxfp4 params {p!r}")
        elif codec == KQUANT_CODEC:
            kcodec = p.get("kquant_codec")
            geometry = KQUANT_GEOMETRY.get(kcodec)
            if geometry is None:
                raise BundleFormatError(
                    f"{where} {proj}: unknown kquant codec {kcodec!r}")
            if (
                b != geometry.bits
                or p.get("group_size") != geometry.group_size
                or p.get("bytes_per_block") != geometry.bytes_per_block
                or p.get("weights_per_block") != geometry.weights_per_block
            ):
                raise BundleFormatError(f"{where} {proj}: bad kquant params {p!r}")
        c = p.get(comp)
        if not isinstance(c, dict):
            raise BundleFormatError(f"{where} {proj}.{comp}: missing component")
        token, _np_dtype = _COMPONENT_DTYPES[comp]
        if c.get("dtype") != token:
            raise BundleFormatError(
                f"{where} {proj}.{comp}: dtype {c.get('dtype')!r} != {token}")
        shape = c.get("shape")
        want_ndim = _component_ndim(codec, comp) - 1
        if (not isinstance(shape, list) or len(shape) != want_ndim
                or not all(isinstance(d, int) and d > 0 for d in shape)):
            raise BundleFormatError(f"{where} {proj}.{comp}: bad shape {shape!r}")
        elems = 1
        for d in shape:
            elems *= d
        nbytes = c.get("nbytes")
        if nbytes != elems * np.dtype(_np_dtype).itemsize:
            raise BundleFormatError(
                f"{where} {proj}.{comp}: nbytes {nbytes!r} != shape {shape} x "
                f"{np.dtype(_np_dtype).itemsize}B")
        if c.get("offset") != expect_offset:
            raise BundleFormatError(
                f"{where} {proj}.{comp}: offset {c.get('offset')!r} != expected "
                f"{expect_offset} (ROW_ORDER tiling violated)")
        expect_offset += nbytes
    if expect_offset != row_bytes:
        raise BundleFormatError(
            f"{where}: components end at {expect_offset} but row_bytes is "
            f"{row_bytes} (format has no padding)")
    return geo


def _validate_mxfp4_packed_scales_shape(
    where: str,
    packed: np.ndarray,
    scales: np.ndarray,
) -> None:
    if packed.ndim != 3 or scales.ndim != 3:
        raise BundleFormatError(
            f"{where}: packed/scales must both be 3D, got "
            f"{packed.ndim}D/{scales.ndim}D")
    if packed.shape[:2] != scales.shape[:2]:
        raise BundleFormatError(
            f"{where}: packed shape {packed.shape} and scales shape {scales.shape} "
            "disagree on experts/out rows")
    packed_words = packed.shape[2]
    if packed_words % 4:
        raise BundleFormatError(
            f"{where}: packed words {packed_words} not divisible by 4")
    expected_scale_cols = packed_words // 4
    if scales.shape[2] != expected_scale_cols:
        raise BundleFormatError(
            f"{where}: scales cols {scales.shape[2]} != expected "
            f"{expected_scale_cols} for packed words {packed_words}")


def _validate_kquant_component_shape(
    where: str,
    weight: np.ndarray,
    scales: np.ndarray,
    *,
    bits: int,
    kquant_codec: str | None,
) -> None:
    if kquant_codec is None:
        raise BundleFormatError(f"{where}: missing kquant codec")
    geometry = KQUANT_GEOMETRY.get(kquant_codec)
    if geometry is None:
        raise BundleFormatError(f"{where}: unknown kquant codec {kquant_codec!r}")
    if bits != geometry.bits:
        raise BundleFormatError(
            f"{where}: kquant codec {kquant_codec!r} uses {geometry.bits} bits, "
            f"got {bits!r}")
    if weight.ndim != 3 or scales.ndim != 2:
        raise BundleFormatError(
            f"{where}: kquant weight/scales must be 3D/2D, got "
            f"{weight.ndim}D/{scales.ndim}D")
    if weight.shape[0] != scales.shape[0]:
        raise BundleFormatError(
            f"{where}: kquant weight experts {weight.shape[0]} "
            f"!= scales experts {scales.shape[0]}")
    if scales.shape[1] != 1:
        raise BundleFormatError(
            f"{where}: kquant scales is a placeholder with shape [experts, 1], "
            f"got {scales.shape}")
    if weight.shape[2] % geometry.bytes_per_block:
        raise BundleFormatError(
            f"{where}: kquant codec {kquant_codec!r} uses "
            f"{geometry.bytes_per_block}-byte blocks, but bytes_per_row "
            f"{weight.shape[2]} is not divisible by it")


def component_array(rows: np.ndarray, component: dict) -> np.ndarray:
    """Slice one component out of bundle rows `[k, row_bytes]` -> `[k, *shape]`.

    The reader-side counterpart of assemble_layer_bundle, for the correctness
    ladder and the inspector probe. `component` is a geometry dict entry
    (offset/nbytes/shape/dtype).
    """
    if rows.ndim != 2 or rows.dtype != np.uint8:
        raise BundleFormatError(f"rows must be 2D uint8, got {rows.ndim}D {rows.dtype}")
    off, nbytes = component["offset"], component["nbytes"]
    if off + nbytes > rows.shape[1]:
        raise BundleFormatError(
            f"component range [{off}, {off + nbytes}) exceeds row_bytes {rows.shape[1]}")
    flat = np.ascontiguousarray(rows[:, off:off + nbytes])
    arr = flat.view(NP_DTYPES[component["dtype"]])
    return arr.reshape(rows.shape[0], *component["shape"])
