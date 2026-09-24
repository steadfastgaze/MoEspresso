"""Rewrite routed Qwen4 IQ_K bundles onto the stream-major package layout.

The transformation changes the byte ordering inside each IQ_K ``blocks``
component. It does not re-encode weights, alter a codec, or touch dense IQ_K
weights. A destination is always a new package directory. Its immutable
payload is hard-linked when the two package roots share a filesystem. IQ_K
routed expert shards are rewritten through atomic replacements. The exact
Q8_0 fallback lattice used by the released model remains hard-linked.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import struct
import time
from typing import Mapping

import numpy as np

from moespresso.core.artifact import ArtifactError, read_artifact, write_artifact
from moespresso.package.bundle import (
    IQK_CODEC,
    KQUANT_CODEC,
    METADATA_KEY,
    PROJECTIONS as BUNDLE_PROJECTIONS,
    decode_bundle_metadata,
    encode_bundle_metadata,
)
from moespresso.package.constants import MANIFEST_NAME
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
)
from moespresso.package.iqk_relayout import (
    check_relayout_member,
    pack_stream_major,
    unpack_stream_major,
)
from moespresso.package.iqk_write import annotate_iqk_stream_geometry
from moespresso.package.manifest import file_identity
from moespresso.package.qwen4.iqk_package import (
    IQK_REPORT_NAME,
    PACKAGE_PLAN_NAME,
    QWEN4_FAMILY,
)
from moespresso.package.sidecars import build_sidecars


REPORT_NAME = "qwen4_iqk_stream_layout_report.json"
QWEN_PROJECTIONS = ("gate", "up", "down")
FILESYSTEM_RESERVE_BYTES = 1 << 30
_Q8_EARLY_CELLS = {(layer, projection) for layer in (0, 1) for projection in QWEN_PROJECTIONS}
_ALL_EXPERT_CELLS = {(layer, projection) for layer in range(48) for projection in QWEN_PROJECTIONS}


class Qwen4IQKStreamLayoutError(ValueError):
    """The source package cannot be safely rewritten onto this layout."""


@dataclass(frozen=True)
class _ExpertInventory:
    mode: str
    rewrite_shards: dict[str, set[tuple[int, str]]]
    unchanged_shards: dict[str, set[tuple[int, str]]]
    cells: dict[tuple[int, str], tuple[str, str, str | None]]

    @property
    def rewrite_cells(self) -> set[tuple[int, str]]:
        return set().union(*self.rewrite_shards.values())

    @property
    def unchanged_cells(self) -> set[tuple[int, str]]:
        return set().union(*self.unchanged_shards.values()) if self.unchanged_shards else set()


def _read_safetensors_header(path: Path) -> tuple[dict, int]:
    try:
        with open(path, "rb") as source:
            raw_length = source.read(8)
            if len(raw_length) != 8:
                raise Qwen4IQKStreamLayoutError(f"{path.name}: too short for a safetensors header")
            length = struct.unpack("<Q", raw_length)[0]
            if length <= 0 or length > path.stat().st_size - 8:
                raise Qwen4IQKStreamLayoutError(
                    f"{path.name}: invalid safetensors header length {length}"
                )
            raw_header = source.read(length)
    except OSError as exc:
        raise Qwen4IQKStreamLayoutError(f"could not read {path}") from exc
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise Qwen4IQKStreamLayoutError(f"{path.name}: invalid safetensors header JSON") from exc
    if not isinstance(header, dict):
        raise Qwen4IQKStreamLayoutError(f"{path.name}: safetensors header is not an object")
    return header, 8 + length


def _canonical_relative_path(value: object, *, label: str) -> str:
    posix = PurePosixPath(value) if isinstance(value, str) else None
    windows = PureWindowsPath(value) if isinstance(value, str) else None
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or posix is None
        or posix.is_absolute()
        or windows is None
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != value
    ):
        raise Qwen4IQKStreamLayoutError(f"{label} has a noncanonical relative path")
    return value


def _manifest_files(package_dir: Path, manifest: Mapping[str, object]) -> dict[str, dict]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise Qwen4IQKStreamLayoutError("manifest has no declared files")
    files: dict[str, dict] = {}
    root = package_dir.resolve()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, dict):
            raise Qwen4IQKStreamLayoutError(f"manifest file {index} is not an object")
        relative = _canonical_relative_path(raw.get("path"), label=f"manifest file {index}")
        if relative in files:
            raise Qwen4IQKStreamLayoutError(f"manifest declares {relative} more than once")
        path = root / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise Qwen4IQKStreamLayoutError(
                f"manifest file {relative} is missing or escapes the package root"
            ) from exc
        if path.is_symlink() or not resolved.is_file():
            raise Qwen4IQKStreamLayoutError(
                f"manifest file {relative} must be a regular package file"
            )
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise Qwen4IQKStreamLayoutError(f"manifest file {relative} has an invalid size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Qwen4IQKStreamLayoutError(f"manifest file {relative} has an invalid sha256")
        actual = file_identity(path)
        if actual["size_bytes"] != size or actual["sha256"] != digest:
            raise Qwen4IQKStreamLayoutError(
                f"manifest file {relative} does not match its declared identity"
            )
        files[relative] = dict(raw)
    return files


def _clone(value: object) -> dict:
    try:
        copied = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise Qwen4IQKStreamLayoutError("package artifact is not JSON serializable") from exc
    if not isinstance(copied, dict):
        raise Qwen4IQKStreamLayoutError("package artifact is not an object")
    return copied


def _bundle_geometry(header: Mapping[str, object], *, shard: str) -> tuple[int, dict, str, dict]:
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or not isinstance(metadata.get(METADATA_KEY), str):
        raise Qwen4IQKStreamLayoutError(f"{shard}: missing {METADATA_KEY} metadata")
    try:
        layers = decode_bundle_metadata(metadata[METADATA_KEY])
    except ValueError as exc:
        raise Qwen4IQKStreamLayoutError(f"{shard}: invalid bundle metadata") from exc
    if len(layers) != 1:
        raise Qwen4IQKStreamLayoutError(
            f"{shard}: requires one isolated layer bundle, found {len(layers)}"
        )
    ((layer, geometry),) = layers.items()
    tensor_names = [name for name in header if name != "__metadata__"]
    if len(tensor_names) != 1:
        raise Qwen4IQKStreamLayoutError(
            f"{shard}: requires one bundle tensor, found {len(tensor_names)}"
        )
    key = tensor_names[0]
    entry = header[key]
    if not isinstance(entry, dict):
        raise Qwen4IQKStreamLayoutError(f"{shard}: bundle tensor header is invalid")
    num_experts = geometry.get("num_experts")
    row_bytes = geometry.get("row_bytes")
    if (
        entry.get("dtype") != "U8"
        or entry.get("shape") != [num_experts, row_bytes]
        or entry.get("data_offsets") != [0, int(num_experts) * int(row_bytes)]
    ):
        raise Qwen4IQKStreamLayoutError(
            f"{shard}: bundle tensor does not match its declared geometry"
        )
    projections = geometry.get("projections")
    if not isinstance(projections, dict) or set(projections) != set(BUNDLE_PROJECTIONS):
        raise Qwen4IQKStreamLayoutError(f"{shard}: bundle does not carry three IQ_K projections")
    for projection in BUNDLE_PROJECTIONS:
        params = projections[projection]
        if not isinstance(params, dict) or params.get("codec") != IQK_CODEC:
            raise Qwen4IQKStreamLayoutError(
                f"{shard}: {projection} is not IQ_K; mixed expert bundles are unsupported"
            )
        if params.get("layout") != IQK_LAYOUT_IQK_RELAYOUT:
            raise Qwen4IQKStreamLayoutError(
                f"{shard}: {projection} layout is not {IQK_LAYOUT_IQK_RELAYOUT!r}"
            )
        if params.get("streams") is not None:
            raise Qwen4IQKStreamLayoutError(
                f"{shard}: {projection} relayout must not carry stream-major metadata"
            )
    return int(layer), geometry, key, dict(metadata)


def _expert_manifest_cells(manifest: Mapping[str, object]) -> _ExpertInventory:
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise Qwen4IQKStreamLayoutError("manifest tensors are not an array")
    by_shard: dict[str, set[tuple[int, str]]] = {}
    cells: dict[tuple[int, str], tuple[str, str, str | None]] = {}
    for tensor in tensors:
        if not isinstance(tensor, dict) or tensor.get("kind") != "expert":
            continue
        params = tensor.get("format_params")
        layer = tensor.get("layer_index")
        projection = tensor.get("projection")
        shard = tensor.get("shard")
        if (
            not isinstance(params, dict)
            or not isinstance(layer, int)
            or isinstance(layer, bool)
            or projection not in QWEN_PROJECTIONS
            or not isinstance(shard, str)
        ):
            raise Qwen4IQKStreamLayoutError("expert manifest entries have invalid coordinates")
        cell = (layer, projection)
        if cell in cells:
            raise Qwen4IQKStreamLayoutError(f"manifest duplicates expert cell {cell}")
        tensor_format = tensor.get("format")
        if tensor_format == IQK_CODEC:
            codec = params.get("iqk_codec")
            layout = params.get("layout")
            if not isinstance(codec, str) or layout != IQK_LAYOUT_IQK_RELAYOUT:
                raise Qwen4IQKStreamLayoutError(
                    "expert manifest IQ_K entries do not match the relayout contract"
                )
        elif tensor_format == KQUANT_CODEC:
            codec = params.get("kquant_codec")
            layout = params.get("layout")
            if codec != "q8_0" or layout is not None:
                raise Qwen4IQKStreamLayoutError(
                    "mixed Qwen expert packages require unchanged Q8_0 entries"
                )
        else:
            raise Qwen4IQKStreamLayoutError("routed Qwen expert format is unsupported")
        cells[cell] = (tensor_format, codec, layout)
        by_shard.setdefault(shard, set()).add(cell)
    if len(by_shard) != 48 or set(cells) != _ALL_EXPERT_CELLS:
        raise Qwen4IQKStreamLayoutError(
            "Qwen stream rewrite requires exactly 48 isolated routed-expert shards"
        )
    for shard, shard_cells in by_shard.items():
        layers = {layer for layer, _projection in shard_cells}
        projections = {projection for _layer, projection in shard_cells}
        formats = {cells[cell][0] for cell in shard_cells}
        if len(layers) != 1 or projections != set(QWEN_PROJECTIONS) or len(formats) != 1:
            raise Qwen4IQKStreamLayoutError(
                f"expert shard {shard} is not one homogeneous isolated layer"
            )

    q8_cells = {cell for cell, spec in cells.items() if spec[0] == KQUANT_CODEC}
    if not q8_cells:
        mode = "all_iqk"
    else:
        iqk_cells = set(cells) - q8_cells
        if (
            q8_cells != _Q8_EARLY_CELLS
            or any(cells[cell] != (KQUANT_CODEC, "q8_0", None) for cell in q8_cells)
            or any(
                cells[cell] != (IQK_CODEC, "iq2_k", IQK_LAYOUT_IQK_RELAYOUT) for cell in iqk_cells
            )
        ):
            raise Qwen4IQKStreamLayoutError(
                "mixed Qwen experts must be six early Q8_0 cells and 138 IQ2_K cells"
            )
        mode = "q8_early_iq2_k"

    rewrite_shards = {
        shard: shard_cells
        for shard, shard_cells in by_shard.items()
        if cells[next(iter(shard_cells))][0] == IQK_CODEC
    }
    unchanged_shards = {
        shard: shard_cells
        for shard, shard_cells in by_shard.items()
        if cells[next(iter(shard_cells))][0] == KQUANT_CODEC
    }
    for tensor in tensors:
        if not isinstance(tensor, dict) or tensor.get("kind") == "expert":
            continue
        if tensor.get("format") != IQK_CODEC:
            continue
        params = tensor.get("format_params")
        if not isinstance(params, dict) or params.get("layout") != IQK_LAYOUT_IQK_RELAYOUT:
            raise Qwen4IQKStreamLayoutError("dense IQ_K tensors must remain on iqk_relayout")
    return _ExpertInventory(
        mode=mode,
        rewrite_shards=rewrite_shards,
        unchanged_shards=unchanged_shards,
        cells=cells,
    )


def _validate_plan(
    plan: Mapping[str, object],
    cells: Mapping[tuple[int, str], tuple[str, str, str | None]],
) -> None:
    if plan.get("artifact_kind") != "package_plan" or plan.get("status") != "valid":
        raise Qwen4IQKStreamLayoutError("package plan is not a valid package_plan artifact")
    allocation = plan.get("allocation")
    if not isinstance(allocation, list):
        raise Qwen4IQKStreamLayoutError("package plan has no allocation array")
    plan_cells: dict[tuple[int, str], tuple[str, str, str | None]] = {}
    for alloc in allocation:
        if not isinstance(alloc, dict) or alloc.get("kind") != "expert":
            continue
        layer = alloc.get("layer_index")
        projection = alloc.get("projection")
        if (
            not isinstance(layer, int)
            or isinstance(layer, bool)
            or projection not in QWEN_PROJECTIONS
        ):
            raise Qwen4IQKStreamLayoutError("package plan expert cell has invalid coordinates")
        cell = (layer, projection)
        if cell in plan_cells:
            raise Qwen4IQKStreamLayoutError(f"package plan duplicates expert cell {cell}")
        alloc_format = alloc.get("format")
        if alloc_format == IQK_CODEC:
            spec = (alloc_format, alloc.get("iqk_codec"), alloc.get("layout"))
        elif alloc_format == KQUANT_CODEC:
            spec = (alloc_format, alloc.get("kquant_codec"), alloc.get("layout"))
        else:
            raise Qwen4IQKStreamLayoutError("package plan routed expert format is unsupported")
        plan_cells[cell] = spec
    if plan_cells != cells:
        raise Qwen4IQKStreamLayoutError("package plan and manifest expert cells or codecs disagree")


def _validate_unchanged_q8_shard(
    path: Path,
    *,
    expected_cells: set[tuple[int, str]],
) -> None:
    """Validate one isolated Q8_0 layer before it becomes a hard link."""
    header, data_start = _read_safetensors_header(path)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or not isinstance(metadata.get(METADATA_KEY), str):
        raise Qwen4IQKStreamLayoutError(f"{path.name}: missing {METADATA_KEY} metadata")
    try:
        layers = decode_bundle_metadata(metadata[METADATA_KEY])
    except ValueError as exc:
        raise Qwen4IQKStreamLayoutError(f"{path.name}: invalid Q8_0 bundle metadata") from exc
    if len(layers) != 1:
        raise Qwen4IQKStreamLayoutError(
            f"{path.name}: requires one isolated Q8_0 layer bundle, found {len(layers)}"
        )
    ((layer, geometry),) = layers.items()
    if expected_cells != {(layer, projection) for projection in QWEN_PROJECTIONS}:
        raise Qwen4IQKStreamLayoutError(
            f"{path.name}: manifest expert cells do not match its isolated Q8_0 bundle"
        )
    names = [name for name in header if name != "__metadata__"]
    if len(names) != 1:
        raise Qwen4IQKStreamLayoutError(
            f"{path.name}: requires one Q8_0 bundle tensor, found {len(names)}"
        )
    entry = header[names[0]]
    if not isinstance(entry, dict):
        raise Qwen4IQKStreamLayoutError(f"{path.name}: Q8_0 bundle tensor header is invalid")
    num_experts = int(geometry["num_experts"])
    row_bytes = int(geometry["row_bytes"])
    if (
        entry.get("dtype") != "U8"
        or entry.get("shape") != [num_experts, row_bytes]
        or entry.get("data_offsets") != [0, num_experts * row_bytes]
        or path.stat().st_size != data_start + num_experts * row_bytes
    ):
        raise Qwen4IQKStreamLayoutError(f"{path.name}: Q8_0 bundle payload geometry drifted")
    projections = geometry.get("projections")
    if not isinstance(projections, dict) or set(projections) != set(BUNDLE_PROJECTIONS):
        raise Qwen4IQKStreamLayoutError(
            f"{path.name}: Q8_0 bundle does not carry exactly three projections"
        )
    for projection in BUNDLE_PROJECTIONS:
        params = projections[projection]
        if (
            not isinstance(params, dict)
            or params.get("codec") != KQUANT_CODEC
            or params.get("kquant_codec") != "q8_0"
            or params.get("layout") is not None
            or params.get("streams") is not None
        ):
            raise Qwen4IQKStreamLayoutError(
                f"{path.name}: {projection} is not an unchanged Q8_0 projection"
            )


def _validate_source(
    package_dir: Path,
) -> tuple[dict, dict, _ExpertInventory, dict[str, dict]]:
    try:
        manifest = read_artifact(package_dir / MANIFEST_NAME)
        plan = read_artifact(package_dir / PACKAGE_PLAN_NAME)
    except (ArtifactError, OSError) as exc:
        raise Qwen4IQKStreamLayoutError("could not read a valid package plan and manifest") from exc
    if manifest.get("artifact_kind") != "package_manifest" or manifest.get("status") != "valid":
        raise Qwen4IQKStreamLayoutError("package manifest is not valid")
    architecture = manifest.get("architecture")
    config = architecture.get("config") if isinstance(architecture, Mapping) else None
    if (
        not isinstance(architecture, Mapping)
        or architecture.get("family") != QWEN4_FAMILY
        or not isinstance(config, Mapping)
        or config.get("model_type") != "qwen4_exp_text"
    ):
        raise Qwen4IQKStreamLayoutError("package is not the released Qwen4 text family")
    inventory = _expert_manifest_cells(manifest)
    _validate_plan(plan, inventory.cells)
    files = _manifest_files(package_dir, manifest)
    expert_shards = set(inventory.rewrite_shards) | set(inventory.unchanged_shards)
    if expert_shards - set(files):
        raise Qwen4IQKStreamLayoutError("manifest expert shards are missing file identities")
    if any(Path(name).parent != Path(".") for name in expert_shards):
        raise Qwen4IQKStreamLayoutError("expert bundle shards must be package-root files")
    for name, expected_cells in sorted(inventory.unchanged_shards.items()):
        _validate_unchanged_q8_shard(package_dir / name, expected_cells=expected_cells)
    return manifest, plan, inventory, files


def _rewrite_expert_shard(
    source: Path, destination: Path, *, expected_cells: set[tuple[int, str]]
) -> dict:
    header, data_start = _read_safetensors_header(source)
    layer, geometry, key, metadata = _bundle_geometry(header, shard=source.name)
    if expected_cells != {(layer, projection) for projection in QWEN_PROJECTIONS}:
        raise Qwen4IQKStreamLayoutError(
            f"{source.name}: manifest expert cells do not match its isolated bundle"
        )
    num_experts = int(geometry["num_experts"])
    row_bytes = int(geometry["row_bytes"])
    if source.stat().st_size != data_start + num_experts * row_bytes:
        raise Qwen4IQKStreamLayoutError(f"{source.name}: unexpected trailing data")

    out_geometry = _clone(geometry)
    for projection in BUNDLE_PROJECTIONS:
        out_geometry["projections"][projection]["layout"] = IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1
    try:
        out_geometry = annotate_iqk_stream_geometry(out_geometry)
    except ValueError as exc:
        raise Qwen4IQKStreamLayoutError(
            f"{source.name}: cannot annotate stream-major geometry"
        ) from exc
    out_metadata = dict(metadata)
    out_metadata[METADATA_KEY] = encode_bundle_metadata({layer: out_geometry})
    out_header = {"__metadata__": out_metadata, key: dict(header[key])}
    blob = json.dumps(out_header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    temporary = destination.with_name(destination.name + ".stream-major.tmp")
    if temporary.exists():
        raise Qwen4IQKStreamLayoutError(f"stale atomic temporary exists: {temporary.name}")
    rows_checked = 0
    try:
        with open(source, "rb") as source_handle, open(temporary, "xb") as destination_handle:
            destination_handle.write(struct.pack("<Q", len(blob)))
            destination_handle.write(blob)
            for expert in range(num_experts):
                source_handle.seek(data_start + expert * row_bytes)
                raw = source_handle.read(row_bytes)
                if len(raw) != row_bytes:
                    raise Qwen4IQKStreamLayoutError(
                        f"{source.name}: short read for expert {expert}"
                    )
                rewritten = bytearray(raw)
                for projection in BUNDLE_PROJECTIONS:
                    params = geometry["projections"][projection]
                    blocks = params["blocks"]
                    offset = int(blocks["offset"])
                    nbytes = int(blocks["nbytes"])
                    out_features, bytes_per_row = blocks["shape"]
                    rows = np.frombuffer(raw, dtype=np.uint8, count=nbytes, offset=offset).copy()
                    rows = rows.reshape(int(out_features), int(bytes_per_row))
                    codec = check_relayout_member(params["iqk_codec"])
                    packed = pack_stream_major(codec, rows, int(params["in_features"]))
                    restored = unpack_stream_major(codec, packed, int(params["in_features"]))
                    if not np.array_equal(restored, rows):
                        raise Qwen4IQKStreamLayoutError(
                            f"{source.name}: {projection} expert {expert} fails inverse byte check"
                        )
                    rewritten[offset : offset + nbytes] = packed.tobytes()
                    rows_checked += int(out_features)
                destination_handle.write(rewritten)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "shard": destination.name,
        "layer": layer,
        "experts": num_experts,
        "projections": list(BUNDLE_PROJECTIONS),
        "rows_inverse_checked": rows_checked,
    }


def _copy_unchanged_payload(
    source_dir: Path,
    destination_dir: Path,
    *,
    rewrite_shards: set[str],
    unchanged_expert_shards: set[str],
    immutable_declared_files: set[str],
) -> dict[str, int]:
    if source_dir.stat().st_dev != destination_dir.parent.stat().st_dev:
        raise Qwen4IQKStreamLayoutError(
            "stream-layout output must share the source package filesystem"
        )
    hardlinked = copied = hardlinked_expert_shards = 0
    for source in sorted(source_dir.rglob("*")):
        relative = source.relative_to(source_dir)
        destination = destination_dir / relative
        if source.is_symlink():
            raise Qwen4IQKStreamLayoutError(f"source package contains symlink {relative}")
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=False)
            continue
        if not source.is_file():
            raise Qwen4IQKStreamLayoutError(f"source package contains non-file {relative}")
        relative_name = relative.as_posix()
        if relative_name in rewrite_shards:
            if relative.parent != Path("."):
                raise Qwen4IQKStreamLayoutError("expert bundle shards must be package-root files")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative_name in immutable_declared_files:
            os.link(source, destination)
            hardlinked += 1
            if relative_name in unchanged_expert_shards:
                hardlinked_expert_shards += 1
        else:
            shutil.copy2(source, destination)
            copied += 1
    return {
        "hardlinked_unchanged_declared_files": hardlinked,
        "hardlinked_unchanged_expert_shards": hardlinked_expert_shards,
        "copied_furniture": copied,
    }


def _storage_guard(
    destination_dir: Path,
    *,
    expert_shards: set[str],
    declared_files: Mapping[str, Mapping[str, object]],
) -> dict[str, int | bool]:
    """Require room for changed shards, one atomic temporary, and metadata."""
    sizes = []
    for name in sorted(expert_shards):
        entry = declared_files.get(name)
        size = entry.get("size_bytes") if isinstance(entry, Mapping) else None
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise Qwen4IQKStreamLayoutError(
                f"expert shard {name} has no positive declared byte size"
            )
        sizes.append(size)
    if not sizes:
        raise Qwen4IQKStreamLayoutError("storage guard requires changed expert shards")
    changed_payload = sum(sizes)
    largest_atomic = max(sizes)
    required = changed_payload + largest_atomic + FILESYSTEM_RESERVE_BYTES
    probe = destination_dir.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    available = int(shutil.disk_usage(probe).free)
    result: dict[str, int | bool] = {
        "changed_expert_shards": len(sizes),
        "changed_expert_payload_bytes": changed_payload,
        "largest_atomic_temp_payload_bytes": largest_atomic,
        "metadata_and_filesystem_reserve_bytes": FILESYSTEM_RESERVE_BYTES,
        "required_free_space_bytes": required,
        "available_free_space_bytes": available,
        "sufficient": available >= required,
    }
    if not result["sufficient"]:
        raise Qwen4IQKStreamLayoutError(
            f"insufficient output free space: {available} available, {required} required"
        )
    return result


def _validate_rewritten_expert_shard(
    path: Path,
    *,
    expected_cells: set[tuple[int, str]],
) -> None:
    """Check the committed header and payload shape after the atomic rename."""
    header, data_start = _read_safetensors_header(path)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or not isinstance(metadata.get(METADATA_KEY), str):
        raise Qwen4IQKStreamLayoutError(f"{path.name}: rewritten metadata is missing")
    try:
        layers = decode_bundle_metadata(metadata[METADATA_KEY])
    except ValueError as exc:
        raise Qwen4IQKStreamLayoutError(f"{path.name}: rewritten metadata is invalid") from exc
    if len(layers) != 1:
        raise Qwen4IQKStreamLayoutError(f"{path.name}: rewritten shard has multiple layers")
    ((layer, geometry),) = layers.items()
    if expected_cells != {(layer, projection) for projection in QWEN_PROJECTIONS}:
        raise Qwen4IQKStreamLayoutError(f"{path.name}: rewritten layer does not match manifest")
    names = [name for name in header if name != "__metadata__"]
    if len(names) != 1:
        raise Qwen4IQKStreamLayoutError(f"{path.name}: rewritten shard has extra tensors")
    entry = header[names[0]]
    if not isinstance(entry, dict):
        raise Qwen4IQKStreamLayoutError(f"{path.name}: rewritten tensor header is invalid")
    num_experts = int(geometry["num_experts"])
    row_bytes = int(geometry["row_bytes"])
    if (
        entry.get("dtype") != "U8"
        or entry.get("shape") != [num_experts, row_bytes]
        or entry.get("data_offsets") != [0, num_experts * row_bytes]
        or path.stat().st_size != data_start + num_experts * row_bytes
    ):
        raise Qwen4IQKStreamLayoutError(f"{path.name}: rewritten payload geometry drifted")
    for projection in BUNDLE_PROJECTIONS:
        params = geometry["projections"][projection]
        if params.get("layout") != IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1 or not params.get("streams"):
            raise Qwen4IQKStreamLayoutError(
                f"{path.name}: {projection} did not receive stream-major metadata"
            )


def _clear_owned_destination(destination_dir: Path, *, created_destination: bool) -> None:
    """Remove incomplete output files; retain a caller-supplied empty directory."""
    if created_destination:
        shutil.rmtree(destination_dir)
        return
    for child in destination_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _rewrite_plan(
    plan: Mapping[str, object],
    *,
    rewrite_cells: set[tuple[int, str]],
) -> tuple[dict, int]:
    out = _clone(plan)
    moved = 0
    for alloc in out["allocation"]:
        if alloc.get("kind") != "expert":
            continue
        cell = (alloc.get("layer_index"), alloc.get("projection"))
        if cell not in rewrite_cells:
            continue
        alloc["layout"] = IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1
        moved += 1
    if moved != len(rewrite_cells):
        raise Qwen4IQKStreamLayoutError("rewritten plan does not carry all changed expert cells")
    out.pop("artifact_id", None)
    out.pop("created_at", None)
    return out, moved


def _rewrite_manifest(
    manifest: Mapping[str, object],
    *,
    plan_id: str,
    files: list[dict],
    rewrite_cells: set[tuple[int, str]],
) -> tuple[dict, int]:
    out = _clone(manifest)
    moved = 0
    for tensor in out["tensors"]:
        if tensor.get("kind") != "expert":
            continue
        cell = (tensor.get("layer_index"), tensor.get("projection"))
        if cell not in rewrite_cells:
            continue
        tensor["format_params"]["layout"] = IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1
        moved += 1
    if moved != len(rewrite_cells):
        raise Qwen4IQKStreamLayoutError(
            "rewritten manifest does not carry all changed expert cells"
        )
    out["files"] = sorted(files, key=lambda entry: entry["path"])
    provenance = out.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise Qwen4IQKStreamLayoutError("manifest provenance is not an object")
    provenance["source_plan_id"] = plan_id
    out.pop("artifact_id", None)
    out.pop("created_at", None)
    return out, moved


def _regenerate_sidecars(
    destination_dir: Path, manifest: Mapping[str, object], *, seed: int
) -> list[str]:
    config, jang = build_sidecars(dict(manifest), seed=seed)
    (destination_dir / "config.json").write_text(json.dumps(config, indent=2))
    (destination_dir / "jang_config.json").write_text(json.dumps(jang, indent=2))
    return ["config.json", "jang_config.json"]


def _seed_from_sidecar(package_dir: Path) -> int:
    path = package_dir / "jang_config.json"
    if not path.is_file():
        return 42
    try:
        value = json.loads(path.read_text()).get("mxtq_seed")
    except (OSError, json.JSONDecodeError):
        return 42
    return value if isinstance(value, int) else 42


def rewrite_qwen4_iqk_stream_layout(
    package_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Write a new Qwen4 package with routed bundles on stream-major layout."""
    source_dir = Path(package_dir).resolve()
    destination_dir = Path(output_dir).resolve()
    if destination_dir == source_dir:
        raise Qwen4IQKStreamLayoutError("in-place stream-layout rewrites are not supported")
    try:
        destination_dir.relative_to(source_dir)
    except ValueError:
        pass
    else:
        raise Qwen4IQKStreamLayoutError("output directory must not be inside the source package")
    if destination_dir.exists() and any(destination_dir.iterdir()):
        raise Qwen4IQKStreamLayoutError("output directory must be absent or empty")
    if not source_dir.is_dir():
        raise Qwen4IQKStreamLayoutError("source package directory does not exist")

    manifest, plan, inventory, files = _validate_source(source_dir)
    storage_guard = _storage_guard(
        destination_dir,
        expert_shards=set(inventory.rewrite_shards),
        declared_files=files,
    )
    created_destination = not destination_dir.exists()
    destination_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        copy_report = _copy_unchanged_payload(
            source_dir,
            destination_dir,
            rewrite_shards=set(inventory.rewrite_shards),
            unchanged_expert_shards=set(inventory.unchanged_shards),
            immutable_declared_files=set(files) - set(inventory.rewrite_shards),
        )
        if copy_report["hardlinked_unchanged_declared_files"] != len(files) - len(
            inventory.rewrite_shards
        ):
            raise Qwen4IQKStreamLayoutError("not every unchanged declared file was hard-linked")
        if copy_report["hardlinked_unchanged_expert_shards"] != len(inventory.unchanged_shards):
            raise Qwen4IQKStreamLayoutError("not every unchanged expert shard was hard-linked")
        shard_reports = []
        for name in sorted(inventory.rewrite_shards):
            shard_reports.append(
                _rewrite_expert_shard(
                    source_dir / name,
                    destination_dir / name,
                    expected_cells=inventory.rewrite_shards[name],
                )
            )
            _validate_rewritten_expert_shard(
                destination_dir / name,
                expected_cells=inventory.rewrite_shards[name],
            )

        new_plan, allocations_moved = _rewrite_plan(
            plan,
            rewrite_cells=inventory.rewrite_cells,
        )
        plan_id = write_artifact(
            destination_dir / PACKAGE_PLAN_NAME,
            new_plan,
            created_at=plan.get("created_at"),
        )
        new_files = []
        for name in sorted(files):
            identity = file_identity(destination_dir / name)
            identity["path"] = name
            new_files.append(identity)
        new_manifest, tensors_moved = _rewrite_manifest(
            manifest,
            plan_id=plan_id,
            files=new_files,
            rewrite_cells=inventory.rewrite_cells,
        )
        manifest_id = write_artifact(
            destination_dir / MANIFEST_NAME,
            new_manifest,
            created_at=manifest.get("created_at"),
        )
        final_manifest = read_artifact(destination_dir / MANIFEST_NAME)
        sidecars = _regenerate_sidecars(
            destination_dir,
            final_manifest,
            seed=_seed_from_sidecar(source_dir),
        )
        if (destination_dir / IQK_REPORT_NAME).is_file():
            report = json.loads((destination_dir / IQK_REPORT_NAME).read_text())
            if not isinstance(report, dict):
                raise Qwen4IQKStreamLayoutError("Qwen IQ_K package report is not an object")
            report["expert_layout"] = IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1
            report["package_plan_id"] = plan_id
            report["package_manifest_id"] = manifest_id
            report["stream_layout"] = {"report": REPORT_NAME}
            (destination_dir / IQK_REPORT_NAME).write_text(
                json.dumps(report, indent=2, sort_keys=True)
            )
        _manifest_files(destination_dir, final_manifest)
    except BaseException:
        _clear_owned_destination(destination_dir, created_destination=created_destination)
        raise

    report = {
        "status": "valid",
        "from_layout": IQK_LAYOUT_IQK_RELAYOUT,
        "to_layout": IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
        "expert_layout_mode": inventory.mode,
        "plan_id": {"before": plan["artifact_id"], "after": plan_id},
        "manifest_id": {"before": manifest["artifact_id"], "after": manifest_id},
        "expert_shards": len(inventory.rewrite_shards) + len(inventory.unchanged_shards),
        "expert_shards_rewritten": len(inventory.rewrite_shards),
        "expert_shards_unchanged": len(inventory.unchanged_shards),
        "expert_cells": len(inventory.cells),
        "expert_cells_rewritten": len(inventory.rewrite_cells),
        "expert_cells_unchanged": len(inventory.unchanged_cells),
        "allocations_moved": allocations_moved,
        "manifest_tensors_moved": tensors_moved,
        "rows_inverse_checked": sum(item["rows_inverse_checked"] for item in shard_reports),
        "sidecars_regenerated": sidecars,
        "copy": copy_report,
        "storage_guard": storage_guard,
        "shards": shard_reports,
        "seconds": round(time.perf_counter() - started, 3),
    }
    (destination_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite a Qwen4 IQ_K package's routed bundles onto stream-major layout"
    )
    parser.add_argument("package_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    try:
        report = rewrite_qwen4_iqk_stream_layout(args.package_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", flush=True)
        return 2
    print(json.dumps({key: value for key, value in report.items() if key != "shards"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
