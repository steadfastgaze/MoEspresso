"""Write package-plan allocations as safetensors shards and emit a manifest.

Source tensors are read in bounded bands or expert rows. The plan fixes each
codec; the writer records the resulting keys, component geometry and file hashes.
Routed experts use one per-layer bundle with a contiguous payload per expert.
Shard files split at shard_size_gb and receive their final count after writing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import struct

import numpy as np

from moespresso.inventory import roles
from moespresso.package.bundle import (
    BUNDLE_KEY_SUFFIX,
    METADATA_KEY,
    encode_bundle_metadata,
)
from moespresso.package.kquant_backend import encode_kquant_weight
from moespresso.package.kquant_bundle import assemble_kquant_encoded_layer_bundle
from moespresso.package.kquant_cache import KQuantEncodeCache, source_identity_from_arrays
from moespresso.package.iqk_write import (
    annotate_expert_input_geometry,
    iqk_bundle_row,
)
from moespresso.package.iqk_recipe import iqk_dense_target_from_allocation
from moespresso.package.deepseek_v4.recipe import (
    dense_target_from_allocation as ds4_dense_target_from_allocation,
)
from moespresso.package.deepseek_v4.write import (
    bundle_row as deepseek_v4_bundle_row,
)
from moespresso.package.qwen.write import (
    encode_kquant_experts_streamed as encode_qwen_kquant_experts_streamed,
)
from moespresso.package.manifest import (
    PACKAGE_FORMAT,
    build_package_manifest,
    file_identity,
    located_key,
)
from moespresso.probe import weight_io
from moespresso.probe.deepseek_v4.codec import iter_dequantized_fp8_row_chunks
from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup

_PLACEHOLDER = "?????"

# Auto-sizing the in-memory affine/fp16 row-band (the knob that can OOM conversion).
# A band is materialized as f32, then mlx.quantize makes working copies, so peak
# is a few x the band; budget the band well below free RAM. These bound the auto
# value; an explicit chunk_bytes overrides entirely.
_CHUNK_FREE_FRACTION = 0.10   # band <= 10% of free RAM (leaves room for the few-x spike)
_CHUNK_FLOOR_BYTES = 16 * 1024 * 1024     # never smaller than 16 MB (avoids tiny reads)
_CHUNK_CEILING_BYTES = 512 * 1024 * 1024  # never larger than 512 MB (convert_moe-ish cap)
_STREAMED_AFFINE_OUTPUT_THRESHOLD_BYTES = 256 * 1024 * 1024

_SAFETENSORS_DTYPES = {
    np.dtype("float16"): "F16",
    np.dtype("float32"): "F32",
    np.dtype("int8"): "I8",
    np.dtype("int64"): "I64",
    np.dtype("uint8"): "U8",
    np.dtype("uint32"): "U32",
}


def validate_additional_file_identities(
    package_dir: str | Path,
    records: list[dict] | tuple[dict, ...] | None,
) -> list[dict]:
    """Validate package-owned non-shard files before model writing begins."""
    root = Path(package_dir).resolve()
    validated = []
    seen: set[str] = set()
    for index, raw in enumerate(records or ()):
        if not isinstance(raw, dict):
            raise ValueError(f"additional package file {index} is not an object")
        relative = raw.get("path")
        posix = PurePosixPath(relative) if isinstance(relative, str) else None
        windows = PureWindowsPath(relative) if isinstance(relative, str) else None
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or posix is None
            or posix.is_absolute()
            or windows is None
            or windows.is_absolute()
            or bool(windows.drive)
            or any(part in {"", ".", ".."} for part in posix.parts)
            or posix.as_posix() != relative
        ):
            raise ValueError(
                f"additional package file {index} has a noncanonical relative path"
            )
        if relative in seen:
            raise ValueError(f"additional package file path is duplicated: {relative}")
        seen.add(relative)
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"additional package file {relative} has an invalid size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"additional package file {relative} has an invalid sha256")
        try:
            path = (root / relative).resolve(strict=True)
            path.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"additional package file {relative} escapes the package root"
            ) from exc
        if (root / relative).is_symlink() or not path.is_file():
            raise ValueError(
                f"additional package file {relative} must be a regular package file"
            )
        if path.stat().st_size != size:
            raise ValueError(
                f"additional package file {relative} size does not match its identity"
            )
        actual = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1 << 22), b""):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError(
                f"additional package file {relative} sha256 does not match its identity"
            )
        validated.append(
            {"path": relative, "size_bytes": size, "sha256": digest}
        )
    return validated


def safe_chunk_bytes(free_bytes: int, *, fraction: float = _CHUNK_FREE_FRACTION,
                     floor: int = _CHUNK_FLOOR_BYTES,
                     ceiling: int = _CHUNK_CEILING_BYTES) -> int:
    """A row-band byte budget that fits in `free_bytes` of RAM (pure, testable).

    Returns clamp(floor, fraction*free_bytes, ceiling). Low available RAM stays
    near the floor; high available RAM caps at the ceiling.
    """
    return int(max(floor, min(int(free_bytes * fraction), ceiling)))


def _autosize_chunk_bytes() -> int:
    """Pick a safe band from free RAM (psutil). Falls back to the floor if psutil
    is unavailable: conservative, never crashes."""
    try:
        import psutil
        free = psutil.virtual_memory().available
    except Exception:
        return _CHUNK_FLOOR_BYTES
    return safe_chunk_bytes(free)


def _quantize_affine_streamed_chunks(
    chunks, bits: int, group_size: int,
) -> dict[str, np.ndarray]:
    """Affine-quantize a 2D tensor row-band at a time, never the whole tensor in RAM.

    group_size divides the row, so each row's groups are self-contained and bands
    can be quantized independently then concatenated (convert_moe row-chunked path).
    """
    import mlx.core as mx

    packed_parts, scales_parts, biases_parts = [], [], []
    for _start, chunk in chunks:
        w = mx.array(chunk)
        qw, scales, biases = mx.quantize(w, group_size=group_size, bits=bits, mode="affine")
        mx.eval(qw, scales, biases)
        packed_parts.append(np.asarray(qw))
        scales_parts.append(np.asarray(scales, dtype=np.float16))
        biases_parts.append(np.asarray(biases, dtype=np.float16))
        del w, qw, scales, biases, chunk
        mx.eval()
        mx.clear_cache()
    return {"weight": np.concatenate(packed_parts, axis=0),
            "scales": np.concatenate(scales_parts, axis=0),
            "biases": np.concatenate(biases_parts, axis=0)}


def _source_row_chunks(model_dir: Path, catalog: dict, name: str, header, chunk_bytes: int):
    prefix = name[: -len(".weight")] if name.endswith(".weight") else name
    if header.dtype != "F8_E4M3":
        return weight_io.iter_row_chunks(model_dir, header, chunk_bytes)
    scale_header = catalog.get(f"{prefix}.scale")
    if scale_header is None:
        return None
    return iter_dequantized_fp8_row_chunks(
        model_dir,
        header,
        scale_header,
        chunk_bytes,
        out_dtype=np.float32,
    )


def _quantize_mx_streamed(chunks, mode: str) -> dict[str, np.ndarray]:
    """MLX MX-float quantize row bands and concatenate weight/scales."""
    import mlx.core as mx

    packed_parts, scales_parts = [], []
    for _start, chunk in chunks:
        w = mx.array(chunk)
        qw, scales = mx.quantize(w, mode=mode)
        mx.eval(qw, scales)
        packed_parts.append(np.asarray(qw))
        scales_parts.append(np.asarray(scales, dtype=np.uint8))
        del w, qw, scales, chunk
        mx.eval()
        mx.clear_cache()
    return {"weight": np.concatenate(packed_parts, axis=0),
            "scales": np.concatenate(scales_parts, axis=0)}


def _estimated_affine_output_bytes(shape: list[int] | tuple[int, int],
                                   bits: int, group_size: int) -> int:
    rows, cols = int(shape[0]), int(shape[1])
    packed_cols = (cols * bits + 31) // 32
    groups = cols // group_size
    return rows * (packed_cols * 4 + groups * 2 + groups * 2)


def _estimated_mx_output_bytes(shape: list[int] | tuple[int, int],
                               bits: int) -> int:
    rows, cols = int(shape[0]), int(shape[1])
    packed_cols = (cols * bits + 31) // 32
    scale_cols = (cols + 31) // 32
    return rows * (packed_cols * 4 + scale_cols)


def _safetensors_dtype(arr: np.ndarray) -> str:
    tag = _SAFETENSORS_DTYPES.get(np.dtype(arr.dtype))
    if tag is None:
        raise ValueError(f"unsupported safetensors dtype {arr.dtype}")
    return tag


def _write_component_chunk(info: dict, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    rows = int(arr.shape[0])
    tail = tuple(int(x) for x in arr.shape[1:])
    dtype = _safetensors_dtype(arr)
    if info["rows"] == 0:
        info["tail"] = tail
        info["dtype"] = dtype
    elif info["tail"] != tail or info["dtype"] != dtype:
        raise ValueError(
            f"affine component changed shape/dtype from "
            f"{info['dtype']}{info['tail']} to {dtype}{tail}"
        )
    info["file"].write(arr.tobytes())
    info["rows"] += rows
    info["nbytes"] += arr.nbytes


def _quantize_affine_streamed_files(
    chunks,
    bits: int,
    group_size: int,
    tmp_dir: Path,
    tmp_prefix: str,
) -> dict[str, dict]:
    """Affine-quantize row bands directly into temporary component files."""
    import mlx.core as mx

    infos: dict[str, dict] = {}
    for suffix in ("weight", "scales", "biases"):
        path = tmp_dir / f".{tmp_prefix}.{suffix}.tmp"
        infos[suffix] = {
            "path": path,
            "file": open(path, "wb"),
            "rows": 0,
            "tail": None,
            "dtype": None,
            "nbytes": 0,
        }
    try:
        for _start, chunk in chunks:
            w = mx.array(chunk)
            qw, scales, biases = mx.quantize(
                w, group_size=group_size, bits=bits, mode="affine")
            mx.eval(qw, scales, biases)
            _write_component_chunk(infos["weight"], np.asarray(qw))
            _write_component_chunk(infos["scales"], np.asarray(scales, dtype=np.float16))
            _write_component_chunk(infos["biases"], np.asarray(biases, dtype=np.float16))
            del w, qw, scales, biases, chunk
            mx.eval()
            mx.clear_cache()
    except Exception:
        for info in infos.values():
            info["file"].close()
            info["path"].unlink(missing_ok=True)
        raise
    for info in infos.values():
        info["file"].close()
        if info["rows"] == 0:
            info["path"].unlink(missing_ok=True)
            raise ValueError("affine tensor produced no rows")
        info["shape"] = [info["rows"], *info["tail"]]
        del info["file"]
        del info["rows"]
        del info["tail"]
    return infos


def _quantize_mx_streamed_files(
    chunks,
    mode: str,
    tmp_dir: Path,
    tmp_prefix: str,
) -> dict[str, dict]:
    """MX-float quantize row bands directly into temporary component files."""
    import mlx.core as mx

    infos: dict[str, dict] = {}
    for suffix in ("weight", "scales"):
        path = tmp_dir / f".{tmp_prefix}.{suffix}.tmp"
        infos[suffix] = {
            "path": path,
            "file": open(path, "wb"),
            "rows": 0,
            "tail": None,
            "dtype": None,
            "nbytes": 0,
        }
    try:
        for _start, chunk in chunks:
            w = mx.array(chunk)
            qw, scales = mx.quantize(w, mode=mode)
            mx.eval(qw, scales)
            _write_component_chunk(infos["weight"], np.asarray(qw))
            _write_component_chunk(infos["scales"], np.asarray(scales, dtype=np.uint8))
            del w, qw, scales, chunk
            mx.eval()
            mx.clear_cache()
    except Exception:
        for info in infos.values():
            info["file"].close()
            info["path"].unlink(missing_ok=True)
        raise
    for info in infos.values():
        info["file"].close()
        if info["rows"] == 0:
            info["path"].unlink(missing_ok=True)
            raise ValueError("MX dense tensor produced no rows")
        info["shape"] = [info["rows"], *info["tail"]]
        del info["file"]
        del info["rows"]
        del info["tail"]
    return infos


def _write_streamed_dense_group(
    writer: _ShardWriter,
    located: dict,
    alloc: dict,
    prefix: str,
    files: dict[str, dict],
) -> None:
    keyed = {f"{prefix}.{k}": v for k, v in files.items()}
    shard_name = writer.add_streamed_group_from_files(keyed)
    located[located_key(alloc)] = {"shard": shard_name, "key_prefix": prefix}


def _fp16_streamed(model_dir: Path, header, chunk_bytes: int) -> np.ndarray:
    """Copy a 2D tensor to float16 a row-band at a time (passthrough)."""
    parts = [chunk.astype(np.float16)
             for _start, chunk in weight_io.iter_row_chunks(model_dir, header, chunk_bytes)]
    return np.concatenate(parts, axis=0)


def _matrix_from_row_chunks(chunks, name: str) -> np.ndarray:
    parts = [np.asarray(chunk, dtype=np.float32) for _start, chunk in chunks]
    if not parts:
        raise ValueError(f"dense tensor {name} produced no rows")
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


def _write_deepseek_v4_layer_bundle_streamed(
    writer: _ShardWriter,
    group: DecodedExpertGroup,
    layer: int,
    allocs: dict[str, dict],
    max_experts: int | None,
    *,
    kquant_imatrix_vectors: dict[str, np.ndarray] | None = None,
    kquant_encoder=None,
    kquant_expert_loader=None,
    kquant_cache: KQuantEncodeCache | None = None,
    kquant_cache_context: dict | None = None,
    iqk_expert_loader=None,
) -> tuple[str, str] | None:
    """Write one DS4 layer bundle directly as expert rows."""
    expert_indices = group.experts(layer)
    if max_experts is not None:
        expert_indices = expert_indices[:max_experts]
    if not expert_indices:
        return None

    first_row, geometry = deepseek_v4_bundle_row(
        group,
        layer,
        expert_indices[0],
        allocs,
        kquant_imatrix_vectors=kquant_imatrix_vectors,
        kquant_encoder=kquant_encoder,
        kquant_expert_loader=kquant_expert_loader,
        kquant_cache=kquant_cache,
        kquant_cache_context=kquant_cache_context,
        iqk_expert_loader=iqk_expert_loader,
    )
    geometry = dict(geometry)
    geometry["num_experts"] = len(expert_indices)
    row_bytes = int(geometry["row_bytes"])

    def rows():
        yield first_row
        for expert_index in expert_indices[1:]:
            row, _geo = deepseek_v4_bundle_row(
                group,
                layer,
                expert_index,
                allocs,
                kquant_imatrix_vectors=kquant_imatrix_vectors,
                kquant_encoder=kquant_encoder,
                kquant_expert_loader=kquant_expert_loader,
                kquant_cache=kquant_cache,
                kquant_cache_context=kquant_cache_context,
                iqk_expert_loader=iqk_expert_loader,
            )
            if row.shape != (row_bytes,):
                raise ValueError(
                    f"DS4 layer {layer} expert {expert_index} bundle row "
                    f"{row.shape} != ({row_bytes},)"
                )
            yield row

    prefix = _expert_bundle_prefix(allocs["gate"]["source_name"])
    shard_name = writer.add_streamed_bundle(
        f"{prefix}.{BUNDLE_KEY_SUFFIX}",
        layer,
        geometry,
        rows(),
    )
    return prefix, shard_name


def _write_iqk_layer_bundle_streamed(
    writer: _ShardWriter,
    layer: int,
    allocs: dict[str, dict],
    num_experts: int,
    *,
    max_experts: int | None,
    iqk_expert_loader,
    iqk_expert_source_layout: str | None = None,
    expert_source_ids: Sequence[int] | None = None,
) -> tuple[str, str] | None:
    """Write one model-independent IQ_K bundle directly as expert rows."""
    if iqk_expert_loader is None:
        raise ValueError(
            f"layer {layer} declares IQ_K experts but has no converted-artifact loader"
        )
    if expert_source_ids is None:
        selected = tuple(range(int(num_experts)))
    else:
        selected = tuple(expert_source_ids)
        if (
            not selected
            or any(isinstance(expert, bool) or not isinstance(expert, int) for expert in selected)
            or selected != tuple(sorted(set(selected)))
            or selected[0] < 0
            or selected[-1] >= int(num_experts)
        ):
            raise ValueError(
                f"layer {layer} compact expert source ids must be sorted, unique, and in range"
            )
    if max_experts is not None:
        selected = selected[: int(max_experts)]
    if not selected:
        return None
    first_row, geometry = iqk_bundle_row(
        layer,
        selected[0],
        allocs,
        expert_loader=iqk_expert_loader,
        source_layout=iqk_expert_source_layout,
    )
    geometry = dict(geometry)
    geometry["num_experts"] = len(selected)
    row_bytes = int(geometry["row_bytes"])

    def rows():
        yield first_row
        for expert_index in selected[1:]:
            row, _geometry = iqk_bundle_row(
                layer,
                expert_index,
                allocs,
                expert_loader=iqk_expert_loader,
                source_layout=iqk_expert_source_layout,
            )
            if row.shape != (row_bytes,):
                raise ValueError(
                    f"IQ_K layer {layer} expert {expert_index} bundle row "
                    f"{row.shape} != ({row_bytes},)"
                )
            yield row

    prefix = _expert_bundle_prefix(allocs["gate"]["source_name"])
    shard_name = writer.add_streamed_bundle(
        f"{prefix}.{BUNDLE_KEY_SUFFIX}",
        layer,
        geometry,
        rows(),
    )
    return prefix, shard_name


def _expert_bundle_prefix(source_name: str) -> str:
    if ".ffn.experts." in source_name and source_name.startswith("layers."):
        return source_name.split(".ffn.experts.", 1)[0] + ".ffn.experts"
    return roles.switch_mlp_bundle_prefix(source_name)


# numpy dtype name -> safetensors dtype tag, for the dtypes this writer emits.
_SAFETENSORS_DTYPE_TAG = {
    "bool": "BOOL", "uint8": "U8", "int8": "I8", "int16": "I16",
    "uint16": "U16", "float16": "F16", "int32": "I32", "uint32": "U32",
    "float32": "F32", "float64": "F64", "int64": "I64", "uint64": "U64",
}


class _BF16Codes:
    """Exact BF16 payload bits carried through NumPy's uint16 storage view."""

    def __init__(self, values: np.ndarray):
        array = np.ascontiguousarray(values)
        if array.dtype != np.uint16:
            raise ValueError(f"BF16 payload codes must use uint16, got {array.dtype}")
        self.values = array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.values.shape)

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)

# safetensors assigns data offsets in dtype-rank-descending then name-ascending
# order (the rank is the library's dtype enum position). This list is that
# descending rank order restricted to the numpy-representable tags, so shards
# written here keep the same data layout as shards written by the library.
_SAFETENSORS_LAYOUT_ORDER = (
    "U64", "I64", "F64", "F32", "U32", "I32", "BF16", "F16", "U16", "I16",
    "I8", "U8", "BOOL",
)
_SAFETENSORS_LAYOUT_RANK = {
    tag: index for index, tag in enumerate(_SAFETENSORS_LAYOUT_ORDER)
}


def _safetensors_dtype_tag(arr: np.ndarray | _BF16Codes) -> str:
    if isinstance(arr, _BF16Codes):
        return "BF16"
    tag = _SAFETENSORS_DTYPE_TAG.get(arr.dtype.name)
    if tag is None:
        raise ValueError(f"unsupported shard tensor dtype {arr.dtype!r}")
    return tag


def _write_shard_deterministic(
    path: Path,
    tensors: dict[str, np.ndarray | _BF16Codes],
    metadata: dict[str, str],
) -> None:
    """Write one safetensors shard whose bytes are a pure function of the inputs.

    The library serializer keeps `__metadata__` in hash-map order, which is
    randomized per map instance, so two builds of identical content produce
    headers whose metadata keys appear in different orders and the shard file
    hashes diverge. This writer keeps the library's data layout (dtype rank
    descending, then name ascending, the order the library assigns data
    offsets in) and serializes the header with sorted keys and 8-byte
    alignment padding, so identical inputs always produce identical files.
    """
    ordered = sorted(
        tensors.items(),
        key=lambda kv: (
            _SAFETENSORS_LAYOUT_RANK[_safetensors_dtype_tag(kv[1])],
            kv[0],
        ),
    )
    header: dict = {
        "__metadata__": {str(k): str(v) for k, v in metadata.items()},
    }
    offset = 0
    for name, arr in ordered:
        header[name] = {
            "dtype": _safetensors_dtype_tag(arr),
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + arr.nbytes],
        }
        offset += arr.nbytes
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)  # 8-byte data alignment, as the library pads
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for _name, arr in ordered:
                values = arr.values if isinstance(arr, _BF16Codes) else arr
                if values.flags.c_contiguous:
                    f.write(memoryview(values).cast("B"))
                else:
                    f.write(values.tobytes())
        tmp.rename(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


class _ShardWriter:
    """Buffers tensors and flushes to a new shard when the byte cap is exceeded.

    A "tensor group" (all the keys for one source tensor, e.g. an expert's
    packed/norms/bits) is added atomically so a group never straddles two shards;
    that keeps the loader's per-tensor read within a single file. Returns each
    group's shard name so the caller can record it in `located`.
    """

    def __init__(self, out_dir: Path, cap_bytes: int):
        self.out_dir = out_dir
        self.cap_bytes = cap_bytes  # 0 == unlimited (single shard)
        self.idx = 0
        self.buf: dict[str, np.ndarray | _BF16Codes] = {}
        self.buf_bytes = 0
        # Bundle geometry for the buffered layers: each shard's __metadata__
        # describes exactly the bundles it carries (package/bundle.py schema),
        # so the expert index stays header-only per shard.
        self.buf_bundle_geo: dict[int, dict] = {}
        self.written: list[str] = []  # shard file names, in order

    def _shard_name(self, idx: int) -> str:
        # Keep the standard MLX shard convention. The regular JANG v2 loader used
        # for dense affine packages sniffs `model-*.safetensors` to choose its mmap
        # v2 path when no model.safetensors.index.json is present.
        return f"model-{idx:05d}-of-{_PLACEHOLDER}.safetensors"

    def add_group(self, keyed: dict[str, np.ndarray | _BF16Codes],
                  bundle_geo: tuple[int, dict] | None = None) -> str:
        """Add one tensor group; flush first if it would overflow. Returns shard name.

        `bundle_geo=(layer, geometry)` accompanies an expert-bundle group; the
        geometry lands in the same shard's `__metadata__` as the bundle tensor.
        """
        group_bytes = sum(v.nbytes for v in keyed.values())
        if self.cap_bytes and self.buf and self.buf_bytes + group_bytes > self.cap_bytes:
            self._flush()
        self.buf.update(keyed)
        self.buf_bytes += group_bytes
        if bundle_geo is not None:
            layer, geo = bundle_geo
            self.buf_bundle_geo[layer] = geo
        return self._shard_name(self.idx + 1)

    def add_streamed_bundle(
        self,
        key: str,
        layer: int,
        geometry: dict,
        rows,
    ) -> str:
        """Write one expert bundle shard directly from row bytes.

        Safetensors needs its header before tensor data, so this path is only for
        a single known-shape bundle tensor. It avoids buffering a full DS4 layer
        bundle in Python memory; each yielded row is one expert's gate/up/down
        payload.
        """
        self._flush()
        self.idx += 1
        name = self._shard_name(self.idx)
        path = self.out_dir / name
        tmp = self.out_dir / f"{name}.tmp"
        row_bytes = int(geometry["row_bytes"])
        n_experts = int(geometry["num_experts"])
        header = {
            "__metadata__": {
                "format": PACKAGE_FORMAT,
                METADATA_KEY: encode_bundle_metadata({layer: geometry}),
            },
            key: {
                "dtype": "U8",
                "shape": [n_experts, row_bytes],
                "data_offsets": [0, n_experts * row_bytes],
            },
        }
        header_bytes = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        written = 0
        try:
            with open(tmp, "wb") as f:
                f.write(struct.pack("<Q", len(header_bytes)))
                f.write(header_bytes)
                for row in rows:
                    arr = np.asarray(row)
                    if arr.dtype != np.uint8 or arr.shape != (row_bytes,):
                        raise ValueError(
                            f"bundle row must be uint8[{row_bytes}], got "
                            f"{arr.dtype}{arr.shape}"
                        )
                    f.write(np.ascontiguousarray(arr).tobytes())
                    written += 1
                    if written > n_experts:
                        raise ValueError(
                            f"bundle yielded more than {n_experts} expert rows"
                        )
            if written != n_experts:
                raise ValueError(
                    f"bundle yielded {written} expert rows, expected {n_experts}"
                )
            tmp.rename(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        self.written.append(name)
        return name

    def add_streamed_group_from_files(self, keyed: dict[str, dict]) -> str:
        """Write one tensor group from temp files without resident arrays."""
        self._flush()
        self.idx += 1
        name = self._shard_name(self.idx)
        path = self.out_dir / name
        tmp = self.out_dir / f"{name}.tmp"
        header = {"__metadata__": {"format": PACKAGE_FORMAT}}
        offset = 0
        ordered = sorted(keyed.items())
        for key, info in ordered:
            nbytes = int(info["nbytes"])
            header[key] = {
                "dtype": info["dtype"],
                "shape": list(info["shape"]),
                "data_offsets": [offset, offset + nbytes],
            }
            offset += nbytes
        header_bytes = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            with open(tmp, "wb") as out:
                out.write(struct.pack("<Q", len(header_bytes)))
                out.write(header_bytes)
                for _key, info in ordered:
                    with open(info["path"], "rb") as src:
                        shutil.copyfileobj(src, out, length=8 << 20)
            tmp.rename(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        finally:
            for info in keyed.values():
                info["path"].unlink(missing_ok=True)
        self.written.append(name)
        return name

    def _flush(self) -> None:
        if not self.buf:
            return
        self.idx += 1
        name = self._shard_name(self.idx)
        metadata = {"format": PACKAGE_FORMAT}
        if self.buf_bundle_geo:
            metadata[METADATA_KEY] = encode_bundle_metadata(self.buf_bundle_geo)
        _write_shard_deterministic(self.out_dir / name, self.buf, metadata)
        self.written.append(name)
        self.buf = {}
        self.buf_bytes = 0
        self.buf_bundle_geo = {}

    def finalize(self) -> dict[str, str]:
        """Flush the tail, rename `-of-?????` -> `-of-COUNT`, return {old:new} map."""
        self._flush()
        count = self.idx
        rename = {}
        for old in self.written:
            new = old.replace(_PLACEHOLDER, f"{count:05d}")
            if new != old:
                (self.out_dir / old).rename(self.out_dir / new)
            rename[old] = new
        return rename


def _passthrough_array(
    model_dir: Path,
    header,
    fmt: str = "fp16",
) -> np.ndarray | _BF16Codes:
    """Copy a structural tensor in its declared package passthrough format.

    Every structural tensor is copied 1:1, no transpose, no value change. This is
    load-bearing for conv1d.weight specifically: mlx_lm's qwen3_5 sanitize
    (qwen3_5.py:309-330) couples two actions to the predicate `conv1d.shape[-1] != 1`:
    it transposes conv1d via moveaxis(2,1) and adds +1.0 to every RMSNorm weight.
    Norms are stored unshifted (source convention, ~0.0), so that +1.0 must fire at
    load for them to reach ~1.0. Storing conv1d pre-transposed as [out, k, 1] sets
    shape[-1] == 1, the predicate is false, the shift is skipped, and every norm loads
    ~1.0 too low -> garbage. So conv1d must stay [out, 1, k] here (mlx's nn.Conv1d
    wants [out, k, 1] and gets it from sanitize's transpose). Stored here as [out,1,k].
    Pinned by test_write.test_passthrough_structural_tensors_round_trip.
    """
    if fmt == "raw_dtype_passthrough":
        values = weight_io.load_full_raw(model_dir, header)
        return _BF16Codes(values) if header.dtype == "BF16" else values
    if fmt == "f32_passthrough":
        return weight_io.load_full(model_dir, header).astype(np.float32)
    return weight_io.load_full(model_dir, header).astype(np.float16)


def write_package(
    package_plan: dict,
    model_dir: Path,
    arch_config: dict,
    out_dir: Path,
    *,
    shard_size_gb: float = 0.0,
    chunk_bytes: int | None = None,
    passthrough: list[dict] | None = None,
    tokenizer: dict | None = None,
    agentic_profile: dict | None = None,
    max_experts: int | None = None,
    deepseek_v4_expert_group: DecodedExpertGroup | None = None,
    kquant_imatrix_vectors: dict[str, np.ndarray] | None = None,
    kquant_encoder=None,
    kquant_expert_loader=None,
    kquant_cache: KQuantEncodeCache | None = None,
    kquant_cache_context: dict | None = None,
    iqk_expert_loader=None,
    iqk_expert_source_layout: str | None = None,
    iqk_dense_encoder=None,
    expert_source_ids: Mapping[int, Sequence[int]] | None = None,
    expert_layout: dict | None = None,
    additional_files: list[dict] | tuple[dict, ...] | None = None,
    ple_provider: dict | None = None,
) -> dict:
    """Quantize per the package plan, write shard(s), return the package_manifest.

    Streams within every tensor so the full model converts in a bounded footprint:
    affine/fp16 are quantized a ~`chunk_bytes` row-band at a time; stacked experts
    are encoded or copied one expert at a time. Peak RAM is one band / one expert plus
    the current shard buffer, never a whole tensor or the whole model.

    `chunk_bytes=None` (default) auto-sizes the row-band from free RAM
    (safe_chunk_bytes). Pass an int to override. `shard_size_gb` caps each shard
    (0 = single shard, the test default): it is a disk-file split, so it keeps a
    plain default. Expert source tensors are stacked-3D; affine/fp16 are 2D.
    """
    if chunk_bytes is None:
        chunk_bytes = _autosize_chunk_bytes()
    if (expert_source_ids is None) != (expert_layout is None):
        raise ValueError(
            "compact expert source ids and expert layout must be declared together"
        )
    if expert_source_ids is None:
        compact_ids = None
    else:
        if any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in expert_source_ids
        ):
            raise ValueError("compact expert source-id layer keys must be integers")
        compact_ids = {
            layer: tuple(ids) for layer, ids in expert_source_ids.items()
        }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validated_additional_files = validate_additional_file_identities(
        out_dir,
        additional_files,
    )
    catalog = weight_io.scan_offsets(model_dir)

    cap_bytes = int(shard_size_gb * (1024 ** 3))
    writer = _ShardWriter(out_dir, cap_bytes)
    located: dict[str, dict] = {}

    expert_allocs: dict[int, dict[str, dict]] = {}
    for alloc in package_plan.get("allocation", []):
        name = alloc["source_name"]
        if alloc["kind"] == "expert":
            # Deferred: a layer's gate/up/down are written together as one
            # per-expert bundle tensor (the streaming format) after this
            # loop, so all three allocations must be collected first.
            layer = int(alloc["layer_index"])
            expert_allocs.setdefault(layer, {})[alloc["projection"]] = alloc
            continue
        header = catalog.get(name)
        if header is None:
            continue  # manifest will flag the missing location, fail-closed
        if alloc["kind"] == "affine":
            # A QuantizedLinear's params are <module_path>.{weight,scales,biases};
            # the module path is the source name without the trailing `.weight`
            # (else the key becomes `...gate_proj.weight.weight` and never binds).
            # jang's model.sanitize() renames model.language_model.* at load.
            prefix = name[: -len(".weight")] if name.endswith(".weight") else name
            fmt = alloc.get("format", "affine")
            chunks = _source_row_chunks(model_dir, catalog, name, header, chunk_bytes)
            if chunks is None:
                continue  # manifest will flag the missing location, fail-closed
            if fmt in {"mxfp4", "mxfp8"}:
                bits = 4 if fmt == "mxfp4" else 8
                stream_output = (
                    _estimated_mx_output_bytes(header.shape, bits)
                    > _STREAMED_AFFINE_OUTPUT_THRESHOLD_BYTES
                )
                if stream_output:
                    files = _quantize_mx_streamed_files(
                        chunks, fmt, out_dir, f"{fmt}.{len(located)}")
                    _write_streamed_dense_group(writer, located, alloc, prefix, files)
                    continue
                grp = _quantize_mx_streamed(chunks, fmt)
            elif fmt == "affine":
                stream_output = (
                    _estimated_affine_output_bytes(
                        header.shape,
                        int(alloc["bits"]),
                        int(alloc["group_size"]),
                    )
                    > _STREAMED_AFFINE_OUTPUT_THRESHOLD_BYTES
                )
                if stream_output:
                    files = _quantize_affine_streamed_files(
                        chunks,
                        alloc["bits"],
                        alloc["group_size"],
                        out_dir,
                        f"affine.{len(located)}",
                    )
                    _write_streamed_dense_group(writer, located, alloc, prefix, files)
                    continue
                grp = _quantize_affine_streamed_chunks(
                    chunks, alloc["bits"], alloc["group_size"])
            elif fmt == "kquant":
                target = ds4_dense_target_from_allocation(alloc)
                encoder = kquant_encoder or encode_kquant_weight
                matrix = _matrix_from_row_chunks(chunks, name)
                encoded = None
                metadata = None
                if kquant_cache is not None:
                    metadata = kquant_cache.metadata_for(
                        source=source_identity_from_arrays(
                            "dense_matrix",
                            {"weight": matrix},
                            source_name=name,
                        ),
                        target=target,
                        imatrix_vectors=kquant_imatrix_vectors or {},
                        context=kquant_cache_context,
                    )
                    encoded = kquant_cache.get(metadata)
                if encoded is None:
                    encoded = encoder(
                        matrix,
                        target,
                        kquant_imatrix_vectors or {},
                    )
                    if metadata is not None:
                        kquant_cache.put(metadata, encoded)
                if encoded.codec != target.codec:
                    raise ValueError(
                        f"K-quant encoder returned codec {encoded.codec!r}, expected "
                        f"{target.codec!r} for dense tensor {name}")
                grp = {"weight": encoded.weight, "scales": encoded.scales}
            elif fmt == "iqk":
                # Dense IQ_K: the encode is ik's own quantizer behind an
                # injected callable (the build harness links it; nothing in
                # tracked source reimplements a codec). The encoder consumes
                # the float matrix plus the target's own steering vector and
                # returns the member's row-major wire; the byte count is
                # checked against the member's struct arithmetic before a
                # byte is written, so a wrong-member or truncated encode
                # fails here rather than at serve.
                from moespresso.package.iqk_format import iqk_dense_geometry

                target = iqk_dense_target_from_allocation(alloc)
                if iqk_dense_encoder is None:
                    raise ValueError(
                        f"dense tensor {name} declares IQ_K member "
                        f"{target.codec!r} but no iqk_dense_encoder was "
                        "provided; the dense IQ_K encode requires the linked "
                        "ik quantizer")
                imatrix = (kquant_imatrix_vectors or {}).get(target.imatrix_key)
                if imatrix is None:
                    raise ValueError(
                        f"{name}: missing imatrix vector {target.imatrix_key!r}; "
                        "dense IQ_K encodes are always steered")
                geometry = iqk_dense_geometry(target.codec)
                matrix = _matrix_from_row_chunks(chunks, name)
                if int(matrix.shape[1]) != int(np.asarray(imatrix).shape[-1]):
                    raise ValueError(
                        f"{name}: imatrix vector length "
                        f"{np.asarray(imatrix).shape[-1]} does not match "
                        f"in_features {matrix.shape[1]}")
                wire = np.asarray(
                    iqk_dense_encoder(matrix, target, imatrix), dtype=np.uint8)
                expect = (
                    int(matrix.shape[0]),
                    geometry.bytes_per_row(int(matrix.shape[1])),
                )
                if tuple(int(v) for v in wire.shape) != expect:
                    raise ValueError(
                        f"{name}: IQ_K dense encoder returned shape "
                        f"{tuple(wire.shape)}, expected {expect} at "
                        f"{target.codec}")
                grp = {"weight": np.ascontiguousarray(wire)}
            else:
                raise ValueError(f"unsupported dense tensor format {fmt!r} for {name}")
            keyed = {f"{prefix}.{k}": v for k, v in grp.items()}
        else:  # fp16_passthrough
            prefix = name
            arr = _fp16_streamed(model_dir, header, chunk_bytes)
            # Smoke: the router gate is [num_experts, hidden]; with experts clamped
            # to max_experts the served router is [max_experts, hidden], so slice it
            # to match (the only fp16 tensor whose rows == num_experts).
            if (max_experts is not None and alloc.get("role") == "moe.router_gate"
                    and arr.shape[0] > max_experts):
                arr = np.ascontiguousarray(arr[:max_experts])
            keyed = {prefix: arr}

        shard_name = writer.add_group(keyed)
        located[located_key(alloc)] = {"shard": shard_name, "key_prefix": prefix}

    # Routed experts: one bundle per layer (uint8 [n_experts, row_bytes], row e =
    # expert e's full gate/up/down payload) so a streamed miss is one pread
    # instead of six scattered ones. DS4 experts write rows directly;
    # the Qwen K-quant path assembles one layer's packed stack
    # before writing its bundle.
    if compact_ids is not None and set(compact_ids) != set(expert_allocs):
        raise ValueError("compact expert source ids must cover every routed layer exactly")
    for layer in sorted(expert_allocs):
        allocs = expert_allocs[layer]
        if sorted(allocs) != ["down", "gate", "up"]:
            # Incomplete layer (missing source tensor): write nothing; the
            # manifest flags every unwritten location, fail-closed.
            continue
        formats = {str(alloc.get("format")) for alloc in allocs.values()}
        if compact_ids is not None and formats != {"iqk"}:
            raise ValueError(
                f"compact expert source ids require an IQ_K layer, got {sorted(formats)}"
            )
        if formats == {"iqk"} and deepseek_v4_expert_group is None:
            source_header = catalog.get(allocs["gate"]["source_name"])
            if source_header is None or len(source_header.shape) != 3:
                continue
            streamed = _write_iqk_layer_bundle_streamed(
                writer,
                layer,
                allocs,
                int(source_header.shape[0]),
                max_experts=max_experts,
                iqk_expert_loader=iqk_expert_loader,
                iqk_expert_source_layout=iqk_expert_source_layout,
                expert_source_ids=(
                    None if compact_ids is None else compact_ids.get(layer)
                ),
            )
            if streamed is None:
                continue
            prefix, shard_name = streamed
            for allocation in allocs.values():
                located[located_key(allocation)] = {
                    "shard": shard_name,
                    "key_prefix": prefix,
                }
            continue
        if deepseek_v4_expert_group is not None:
            streamed = _write_deepseek_v4_layer_bundle_streamed(
                writer,
                deepseek_v4_expert_group,
                layer,
                allocs,
                max_experts,
                kquant_imatrix_vectors=kquant_imatrix_vectors,
                kquant_encoder=kquant_encoder,
                kquant_expert_loader=kquant_expert_loader,
                kquant_cache=kquant_cache,
                kquant_cache_context=kquant_cache_context,
                iqk_expert_loader=iqk_expert_loader,
            )
            if streamed is None:
                continue
            prefix, shard_name = streamed
            for a in allocs.values():
                located[located_key(a)] = {"shard": shard_name, "key_prefix": prefix}
            continue

        encoded_by_projection = {}
        for projection in ("gate", "up", "down"):
            allocation = allocs[projection]
            if allocation.get("format") != "kquant":
                raise ValueError(
                    f"unsupported routed format {allocation.get('format')!r} "
                    f"for layer={layer} projection={projection}")
            header = catalog.get(allocation["source_name"])
            if header is None:
                continue
            encoded_by_projection[projection] = encode_qwen_kquant_experts_streamed(
                model_dir, header, allocation,
                max_experts=max_experts,
                kquant_imatrix_vectors=kquant_imatrix_vectors,
                kquant_encoder=kquant_encoder,
                kquant_expert_loader=kquant_expert_loader,
                kquant_cache=kquant_cache,
                kquant_cache_context=kquant_cache_context,
            )
        if len(encoded_by_projection) != 3:
            continue
        bundle_arr, geo = assemble_kquant_encoded_layer_bundle(encoded_by_projection)
        annotate_expert_input_geometry(geo, allocs)
        prefix = _expert_bundle_prefix(allocs["gate"]["source_name"])
        shard_name = writer.add_group(
            {f"{prefix}.{BUNDLE_KEY_SUFFIX}": bundle_arr}, bundle_geo=(layer, geo))
        for allocation in allocs.values():
            located[located_key(allocation)] = {"shard": shard_name, "key_prefix": prefix}

    # Structural passthrough tensors (norms, SSM state), copied verbatim so the
    # runtime builds the graph without source files. They come from the inventory,
    # not the optimizer decision (the optimizer stays pure). Tiny: read whole.
    pt_located: dict[str, dict] = {}
    for entry in passthrough or []:
        name = entry["source_name"]
        header = catalog.get(name)
        if header is None:
            continue  # manifest flags the gap, fail-closed
        arr = _passthrough_array(model_dir, header, entry.get("format", "fp16"))
        shard_name = writer.add_group({name: arr})
        pt_located[name] = {"shard": shard_name, "key_prefix": name}

    rename = writer.finalize()
    # Point every tensor's recorded shard at the finalized (renamed) file.
    for loc in (*located.values(), *pt_located.values()):
        loc["shard"] = rename.get(loc["shard"], loc["shard"])

    files = [file_identity(out_dir / new) for new in rename.values()]
    files.extend(validated_additional_files)
    paths = [record.get("path") for record in files]
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("additional package files must declare non-empty paths")
    if len(paths) != len(set(paths)):
        raise ValueError("package file identities contain duplicate paths")
    return build_package_manifest(
        package_plan,
        arch_config,
        located,
        files,
        expert_layout=expert_layout,
        passthrough=passthrough,
        passthrough_located=pt_located,
        tokenizer=tokenizer,
        agentic_profile=agentic_profile,
        ple_provider=ple_provider,
        max_experts=max_experts,
    )
