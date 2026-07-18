"""Write the MJTQ package from a package_plan, then emit its manifest.

MJTQ ("MoEspresso Jang TurboQuant") is the strict package format: it reuses jang's
TurboQuant codec + tensor conventions for compression, but the explicit manifest
(emitted here) is the contract the runtime reads: it never guesses. The upstream
"jangtq" package format is separate; jang is used only as the codec.

The imperative shell: stream each source weight, quantize it per the plan's
allocation (TQ for experts via jang, affine via mlx, fp16 passthrough), write the
packed arrays into safetensors shards, collect on-disk locations + file
identities, then hand all of that to the pure package.manifest builder.

Multi-shard streaming: a new shard is started once the current one passes a byte
cap (`--shard-size-gb`), so a 35B package doesn't have to fit one file in RAM.
The final shard count is only known at the end, so shards are written as
`model-NNNNN-of-?????` and renamed to `-of-COUNT` once done (mirrors the proven
convert_moe). `shard_size_gb=0` keeps everything in one shard (the synthetic/test
path). The manifest already takes a per-tensor `located` map + a `files` list, so
none of this changes the manifest contract.

Needs mlx + jang. The format-defining work (the manifest)
is pure and lives in package/manifest.py; this module only moves bytes.

Tensor key conventions: routed experts -> one per-layer bundle
`...switch_mlp.experts.tq_bundle` (uint8 [n_experts, row_bytes]; the
streaming format: row e is expert e's full gate/up/down payload, geometry in
the shard's `__metadata__`, see package/bundle.py); affine -> `<base>.weight` /
`.scales` / `.biases`; fp16 passthrough -> the raw array under its name.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct

import numpy as np

from moespresso.inventory import roles
from moespresso.package.bundle import (
    BUNDLE_KEY_SUFFIX,
    METADATA_KEY,
    assemble_layer_bundle,
    encode_bundle_metadata,
)
from moespresso.package.kquant_backend import encode_kquant_weight
from moespresso.package.kquant_bundle import assemble_kquant_encoded_layer_bundle
from moespresso.package.kquant_cache import KQuantEncodeCache, source_identity_from_arrays
from moespresso.package.deepseek_v4.recipe import (
    dense_target_from_allocation as ds4_dense_target_from_allocation,
)
from moespresso.package.deepseek_v4.write import (
    bundle_row as deepseek_v4_bundle_row,
    quantize_experts_streamed as quantize_deepseek_v4_experts_streamed,
)
from moespresso.package.qwen.write import (
    encode_kquant_experts_streamed as encode_qwen_kquant_experts_streamed,
)
from moespresso.package.tq import quantize_tq
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


def _quantize_experts_streamed(
    model_dir: Path, header, projection: str, bits: int, seed: int,
    max_experts: int | None = None,
) -> dict[str, np.ndarray]:
    """TQ-quantize a stacked-expert sub-projection one expert at a time -> 3D stack.

    Per-expert streaming limits the peak footprint to one [rows, cols] expert.
    For a fused gate_up source, take the gate or up
    half of each expert before quantizing. The per-expert results are stacked
    (packed [n_experts, out, packed_in], norms [n_experts, out]) as input for the
    per-layer bundle assembly (package/bundle.py); the stack itself is no longer
    written to disk. 'out' = the sub-projection's row count (moe_inter for
    gate/up, hidden for down).

    `max_experts` (smoke artifact) keeps only the first N experts; the manifest's
    config num_experts is reduced to match so the served graph is consistent.
    """
    import mlx.core as mx

    rows = header.shape[1]
    mid = rows // 2
    fused = projection in ("gate", "up")
    packed_list, norms_list = [], []
    for expert in weight_io.iter_experts(model_dir, header, max_experts=max_experts):
        sub = expert
        if fused:
            sub = expert[:mid] if projection == "gate" else expert[mid:]
        r = quantize_tq(sub, bits, seed)
        packed_list.append(r["tq_packed"])   # [out, packed_in]
        norms_list.append(r["tq_norms"])     # [out]
        del expert, sub
        mx.eval()
        mx.clear_cache()
    # Stack to 3D [n_experts, out, packed_in] / [n_experts, out]: jang's
    # prestacked switch_mlp layout, the kernel's native form.
    return {"tq_packed": np.stack(packed_list, axis=0),
            "tq_norms": np.stack(norms_list, axis=0),
            "tq_bits": np.array([bits], dtype=np.uint8)}


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
    seed: int,
    max_experts: int | None,
    *,
    kquant_imatrix_vectors: dict[str, np.ndarray] | None = None,
    kquant_encoder=None,
    kquant_expert_loader=None,
    kquant_cache: KQuantEncodeCache | None = None,
    kquant_cache_context: dict | None = None,
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
        seed,
        kquant_imatrix_vectors=kquant_imatrix_vectors,
        kquant_encoder=kquant_encoder,
        kquant_expert_loader=kquant_expert_loader,
        kquant_cache=kquant_cache,
        kquant_cache_context=kquant_cache_context,
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
                seed,
                kquant_imatrix_vectors=kquant_imatrix_vectors,
                kquant_encoder=kquant_encoder,
                kquant_expert_loader=kquant_expert_loader,
                kquant_cache=kquant_cache,
                kquant_cache_context=kquant_cache_context,
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

# safetensors assigns data offsets in dtype-rank-descending then name-ascending
# order (the rank is the library's dtype enum position). This list is that
# descending rank order restricted to the numpy-representable tags, so shards
# written here keep the same data layout as shards written by the library.
_SAFETENSORS_LAYOUT_ORDER = (
    "U64", "I64", "F64", "F32", "U32", "I32", "F16", "U16", "I16",
    "I8", "U8", "BOOL",
)
_SAFETENSORS_LAYOUT_RANK = {
    tag: index for index, tag in enumerate(_SAFETENSORS_LAYOUT_ORDER)
}


def _safetensors_dtype_tag(arr: np.ndarray) -> str:
    tag = _SAFETENSORS_DTYPE_TAG.get(arr.dtype.name)
    if tag is None:
        raise ValueError(f"unsupported shard tensor dtype {arr.dtype!r}")
    return tag


def _write_shard_deterministic(
    path: Path,
    tensors: dict[str, np.ndarray],
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
                if arr.flags.c_contiguous:
                    f.write(memoryview(arr).cast("B"))
                else:
                    f.write(arr.tobytes())
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
        self.buf: dict[str, np.ndarray] = {}
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

    def add_group(self, keyed: dict[str, np.ndarray],
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


def _passthrough_array(model_dir: Path, header, fmt: str = "fp16") -> np.ndarray:
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
        return weight_io.load_full_raw(model_dir, header)
    if fmt == "f32_passthrough":
        return weight_io.load_full(model_dir, header).astype(np.float32)
    return weight_io.load_full(model_dir, header).astype(np.float16)


def write_package(
    package_plan: dict,
    model_dir: Path,
    arch_config: dict,
    out_dir: Path,
    *,
    seed: int = 42,
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
) -> dict:
    """Quantize per the package plan, write shard(s), return the package_manifest.

    Streams within every tensor so the full model converts in a bounded footprint:
    affine/fp16 are quantized a ~`chunk_bytes` row-band at a time; stacked experts
    are TQ-quantized one expert at a time. Peak RAM is one band / one expert plus
    the current shard buffer, never a whole tensor or the whole model.

    `chunk_bytes=None` (default) auto-sizes the row-band from free RAM
    (safe_chunk_bytes). Pass an int to override. `shard_size_gb` caps each shard
    (0 = single shard, the test default): it is a disk-file split, so it keeps a
    plain default. Expert source tensors are stacked-3D; affine/fp16 are 2D.
    """
    if chunk_bytes is None:
        chunk_bytes = _autosize_chunk_bytes()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    # instead of six scattered ones. DS4 writes rows directly; the Qwen fallback
    # still assembles one layer's packed stack before writing its bundle.
    for layer in sorted(expert_allocs):
        allocs = expert_allocs[layer]
        if sorted(allocs) != ["down", "gate", "up"]:
            # Incomplete layer (missing source tensor): write nothing; the
            # manifest flags every unwritten location, fail-closed.
            continue
        if deepseek_v4_expert_group is not None:
            streamed = _write_deepseek_v4_layer_bundle_streamed(
                writer,
                deepseek_v4_expert_group,
                layer,
                allocs,
                seed,
                max_experts,
                kquant_imatrix_vectors=kquant_imatrix_vectors,
                kquant_encoder=kquant_encoder,
                kquant_expert_loader=kquant_expert_loader,
                kquant_cache=kquant_cache,
                kquant_cache_context=kquant_cache_context,
            )
            if streamed is None:
                continue
            prefix, shard_name = streamed
            for a in allocs.values():
                located[located_key(a)] = {"shard": shard_name, "key_prefix": prefix}
            continue

        comps: dict[tuple[str, str], np.ndarray] = {}
        encoded_by_projection = {}
        bits: dict[str, int] = {}
        codecs: dict[str, str] = {}
        for projection in ("gate", "up", "down"):
            a = allocs[projection]
            codec = a.get("codec", a.get("format", "tq"))
            header = catalog.get(a["source_name"])
            if a.get("format") == "kquant":
                if header is None:
                    continue
                encoded_by_projection[projection] = encode_qwen_kquant_experts_streamed(
                    model_dir,
                    header,
                    a,
                    max_experts=max_experts,
                    kquant_imatrix_vectors=kquant_imatrix_vectors,
                    kquant_encoder=kquant_encoder,
                    kquant_expert_loader=kquant_expert_loader,
                    kquant_cache=kquant_cache,
                    kquant_cache_context=kquant_cache_context,
                )
                continue
            if codec != "tq":
                raise ValueError(
                    f"source mxfp4 routed codec requires DeepSeek V4 raw expert "
                    f"storage; generic writer got codec={codec!r} for layer={layer} "
                    f"projection={projection}")
            if header is not None:
                grp = _quantize_experts_streamed(
                    model_dir,
                    header,
                    projection,
                    a["bits"],
                    seed,
                    max_experts=max_experts,
                )
            elif deepseek_v4_expert_group is not None:
                grp = quantize_deepseek_v4_experts_streamed(
                    deepseek_v4_expert_group,
                    layer,
                    projection,
                    a["bits"],
                    seed,
                    max_experts=max_experts,
                )
            else:
                grp = None
            if grp is None:
                continue
            comps[(f"{projection}_proj", "packed")] = grp["tq_packed"]
            comps[(f"{projection}_proj", "norms")] = grp["tq_norms"]
            bits[f"{projection}_proj"] = int(a["bits"])
            codecs[f"{projection}_proj"] = "tq"
        if encoded_by_projection:
            if comps:
                raise ValueError(
                    f"generic routed layer {layer} mixes K-quant and TQ expert "
                    "projections; use one expert codec family per layer")
            bundle_arr, geo = assemble_kquant_encoded_layer_bundle(encoded_by_projection)
            prefix = _expert_bundle_prefix(allocs["gate"]["source_name"])
            shard_name = writer.add_group(
                {f"{prefix}.{BUNDLE_KEY_SUFFIX}": bundle_arr}, bundle_geo=(layer, geo))
            for a in allocs.values():
                located[located_key(a)] = {"shard": shard_name, "key_prefix": prefix}
            continue
        if sorted(bits) != ["down_proj", "gate_proj", "up_proj"]:
            continue
        bundle_arr, geo = assemble_layer_bundle(comps, bits, codecs=codecs)
        del comps
        prefix = _expert_bundle_prefix(allocs["gate"]["source_name"])
        shard_name = writer.add_group(
            {f"{prefix}.{BUNDLE_KEY_SUFFIX}": bundle_arr}, bundle_geo=(layer, geo))
        for a in allocs.values():
            located[located_key(a)] = {"shard": shard_name, "key_prefix": prefix}

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
    return build_package_manifest(package_plan, arch_config, located, files, seed=seed,
                                  passthrough=passthrough, passthrough_located=pt_located,
                                  tokenizer=tokenizer, agentic_profile=agentic_profile,
                                  max_experts=max_experts)
