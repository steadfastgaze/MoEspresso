"""Stream Qwen4 PLE table payloads into package-owned row files."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from typing import Protocol

from moespresso.inventory.safetensors_header import (
    TensorHeader,
    read_headers_with_offsets,
)
from moespresso.package.constants import MANIFEST_NAME

_COPY_BYTES = 8 << 20


class Qwen4PLEWriteError(RuntimeError):
    """Raised when source PLE storage cannot produce a valid component."""


class Qwen4PLEWriteContract(Protocol):
    """Validated model facts required to write the physical PLE component."""

    layer_index: int
    dtype: str
    row_width: int
    row_bytes: int
    logical_rows: int
    padded_rows: int
    rows_per_shard: int
    shard_count: int
    ngram_size: int
    heads_per_ngram: int
    multipliers: tuple[int, ...]
    table_sizes: tuple[int, ...]
    table_offsets: tuple[int, ...]


@dataclass(frozen=True)
class Qwen4PLESourceTable:
    """One validated source tensor payload before package copying begins."""

    index: int
    tensor: str
    source_path: Path
    header: TensorHeader
    relative_path: str


@dataclass(frozen=True)
class Qwen4PLEReuseShard:
    """One manifest-bound package file eligible for same-volume reuse."""

    index: int
    source_path: Path
    relative_path: str
    size_bytes: int
    sha256: str


def _table_tensor_name(layer_index: int, shard_index: int) -> str:
    return (
        f"model.language_model.layers.{layer_index}.ple.ple_embedding."
        f"ngram_embedding.shard_{shard_index}.weight"
    )


def _source_shard(root: Path, value: object, *, tensor: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Qwen4PLEWriteError(f"{tensor} has no source shard")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or len(posix.parts) != 1
        or posix.parts[0] in {"", ".", ".."}
    ):
        raise Qwen4PLEWriteError(f"{tensor} source shard is not canonical")
    try:
        resolved_root = root.resolve()
        path = (resolved_root / value).resolve()
        allowed_roots = [resolved_root]
        if resolved_root.parent.name == "snapshots":
            blobs = resolved_root.parent.parent / "blobs"
            if blobs.is_dir():
                allowed_roots.append(blobs.resolve())
        if not any(path.is_relative_to(allowed_root) for allowed_root in allowed_roots):
            raise ValueError("resolved shard is outside the allowed source roots")
    except (OSError, RuntimeError, ValueError) as exc:
        raise Qwen4PLEWriteError(f"{tensor} source shard escapes the source root") from exc
    if not path.is_file():
        raise Qwen4PLEWriteError(f"{tensor} source shard is missing")
    return path


def _package_destination(root: Path, relative_path: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    parent = resolved_root / "ple"
    parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved_parent = parent.resolve()
        resolved_parent.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise Qwen4PLEWriteError("PLE output directory escapes the package root") from exc
    return resolved_parent / PurePosixPath(relative_path).name


def _weight_map(source_dir: Path) -> dict[str, str]:
    index_path = source_dir / "model.safetensors.index.json"
    try:
        payload = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen4PLEWriteError("could not read the source safetensors index") from exc
    mapping = payload.get("weight_map")
    if not isinstance(mapping, dict):
        raise Qwen4PLEWriteError("source safetensors index has no weight_map")
    return mapping


def _header_for_tensor(
    *,
    source_dir: Path,
    tensor: str,
    mapping: dict[str, str],
    header_cache: dict[Path, dict[str, TensorHeader]],
) -> tuple[Path, TensorHeader]:
    source_path = _source_shard(source_dir, mapping.get(tensor), tensor=tensor)
    headers = header_cache.get(source_path)
    if headers is None:
        try:
            headers = {
                header.name: header for header in read_headers_with_offsets(source_path)
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise Qwen4PLEWriteError(
                f"could not read source header for {tensor}"
            ) from exc
        header_cache[source_path] = headers
    header = headers.get(tensor)
    if header is None:
        raise Qwen4PLEWriteError(f"source shard does not contain {tensor}")
    return source_path, header


def _copy_tensor_payload(
    source_path: Path,
    header: TensorHeader,
    destination: Path,
) -> tuple[int, str]:
    expected_bytes = int(header.end - header.begin)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with open(source_path, "rb") as source:
                source.seek(int(header.header_size + header.begin))
                remaining = expected_bytes
                while remaining:
                    chunk = source.read(min(_COPY_BYTES, remaining))
                    if not chunk:
                        raise Qwen4PLEWriteError(
                            f"short source read while copying {header.name}"
                        )
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return expected_bytes, digest.hexdigest()


def inspect_qwen4_ple_source(
    source_dir: str | Path,
    *,
    expected: Qwen4PLEWriteContract,
) -> tuple[Qwen4PLESourceTable, ...]:
    """Validate all physical PLE tensors without reading their payloads."""
    if expected.dtype != "BF16" or expected.row_bytes != expected.row_width * 2:
        raise Qwen4PLEWriteError("PLE output contract requires row-major BF16")
    if expected.padded_rows != expected.rows_per_shard * expected.shard_count:
        raise Qwen4PLEWriteError("PLE output contract has inconsistent padded rows")

    source_root = Path(source_dir)
    mapping = _weight_map(source_root)
    header_cache: dict[Path, dict[str, TensorHeader]] = {}
    tables = []
    for shard_index in range(expected.shard_count):
        tensor = _table_tensor_name(expected.layer_index, shard_index)
        source_path, header = _header_for_tensor(
            source_dir=source_root,
            tensor=tensor,
            mapping=mapping,
            header_cache=header_cache,
        )
        expected_shape = (expected.rows_per_shard, expected.row_width)
        if header.dtype != expected.dtype or header.shape != expected_shape:
            raise Qwen4PLEWriteError(
                f"{tensor} has {header.dtype} {header.shape}, expected "
                f"{expected.dtype} {expected_shape}"
            )
        if header.end - header.begin != expected.rows_per_shard * expected.row_bytes:
            raise Qwen4PLEWriteError(f"{tensor} byte range does not match row geometry")
        tables.append(
            Qwen4PLESourceTable(
                index=shard_index,
                tensor=tensor,
                source_path=source_path,
                header=header,
                relative_path=(
                    f"ple/rows-{shard_index:03d}-of-{expected.shard_count:03d}.bf16"
                ),
            )
        )
    return tuple(tables)


def _provider_component(expected: Qwen4PLEWriteContract) -> dict:
    return {
        "schema": "qwen4_ple_provider_v1",
        "layer_index": expected.layer_index,
        "dtype": expected.dtype,
        "row_width": expected.row_width,
        "row_bytes": expected.row_bytes,
        "logical_rows": expected.logical_rows,
        "padded_rows": expected.padded_rows,
        "rows_per_shard": expected.rows_per_shard,
        "ngram_size": expected.ngram_size,
        "heads_per_ngram": expected.heads_per_ngram,
        "multipliers": list(expected.multipliers),
        "table_sizes": list(expected.table_sizes),
        "table_offsets": list(expected.table_offsets),
        "shards": [
            {
                "index": index,
                "path": (
                    f"ple/rows-{index:03d}-of-{expected.shard_count:03d}.bf16"
                ),
                "row_start": index * expected.rows_per_shard,
                "row_count": expected.rows_per_shard,
            }
            for index in range(expected.shard_count)
        ],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(_COPY_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Qwen4PLEWriteError(f"could not hash reusable PLE file {path}") from exc
    return digest.hexdigest()


def _manifest_digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Qwen4PLEWriteError(f"{field} is not a lowercase SHA-256 digest")
    return value


def _reuse_source_path(root: Path, relative_path: str) -> Path:
    posix = PurePosixPath(relative_path)
    windows = PureWindowsPath(relative_path)
    if (
        "\\" in relative_path
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or len(posix.parts) != 2
        or posix.parts[0] != "ple"
        or posix.parts[1] in {"", ".", ".."}
    ):
        raise Qwen4PLEWriteError("reusable PLE path is not canonical")
    path = root / posix
    try:
        resolved_root = root.resolve()
        resolved = path.resolve()
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise Qwen4PLEWriteError("reusable PLE path escapes its package") from exc
    if path.is_symlink() or not path.is_file():
        raise Qwen4PLEWriteError(f"reusable PLE file is not regular: {path}")
    return path


def inspect_qwen4_ple_reuse(
    package_dir: str | Path,
    *,
    expected: Qwen4PLEWriteContract,
    target_dir: str | Path | None = None,
) -> tuple[Qwen4PLEReuseShard, ...]:
    """Validate a package PLE manifest before same-volume hardlink reuse."""
    root = Path(package_dir)
    if root.is_symlink() or not root.is_dir():
        raise Qwen4PLEWriteError(
            f"reusable PLE package is not a regular directory: {root}"
        )
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise Qwen4PLEWriteError("reusable PLE package has no regular manifest")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen4PLEWriteError("could not read reusable PLE manifest") from exc
    if manifest.get("status") != "valid":
        raise Qwen4PLEWriteError("reusable PLE package manifest is not valid")
    component = _provider_component(expected)
    if manifest.get("ple_provider") != component:
        raise Qwen4PLEWriteError("reusable PLE provider contract differs")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise Qwen4PLEWriteError("reusable PLE manifest has no file records")
    by_path: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise Qwen4PLEWriteError("reusable PLE file record is invalid")
        path = record.get("path")
        if not isinstance(path, str):
            raise Qwen4PLEWriteError("reusable PLE file record has no path")
        if path in by_path:
            raise Qwen4PLEWriteError(f"reusable PLE file record is duplicated: {path}")
        by_path[path] = record
    target_device = None
    if target_dir is not None:
        if root.resolve() == Path(target_dir).resolve():
            raise Qwen4PLEWriteError(
                "reusable PLE package and output directory are identical"
            )
        probe = Path(target_dir)
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            target_device = probe.stat().st_dev
        except OSError as exc:
            raise Qwen4PLEWriteError("could not inspect PLE target volume") from exc
    expected_size = expected.rows_per_shard * expected.row_bytes
    shards = []
    for shard in component["shards"]:
        relative_path = shard["path"]
        record = by_path.get(relative_path)
        if record is None:
            raise Qwen4PLEWriteError(
                f"reusable PLE manifest is missing {relative_path}"
            )
        size_bytes = record.get("size_bytes")
        if isinstance(size_bytes, bool) or size_bytes != expected_size:
            raise Qwen4PLEWriteError(
                f"reusable PLE size differs for {relative_path}"
            )
        digest = _manifest_digest(
            record.get("sha256"), field=f"{relative_path}.sha256"
        )
        source_path = _reuse_source_path(root, relative_path)
        try:
            stat = source_path.stat()
        except OSError as exc:
            raise Qwen4PLEWriteError(
                f"could not inspect reusable PLE file {source_path}"
            ) from exc
        if stat.st_size != expected_size:
            raise Qwen4PLEWriteError(
                f"reusable PLE file size differs for {relative_path}"
            )
        if target_device is not None and stat.st_dev != target_device:
            raise Qwen4PLEWriteError(
                "reusable PLE files and package output are on different volumes"
            )
        shards.append(
            Qwen4PLEReuseShard(
                index=shard["index"],
                source_path=source_path,
                relative_path=relative_path,
                size_bytes=expected_size,
                sha256=digest,
            )
        )
    return tuple(shards)


def _hardlink_reused_payload(
    source: Qwen4PLEReuseShard,
    destination: Path,
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.reuse-",
            dir=destination.parent,
        ) as temporary_dir:
            temporary = Path(temporary_dir) / destination.name
            os.link(source.source_path, temporary)
            if temporary.stat().st_size != source.size_bytes:
                raise Qwen4PLEWriteError(
                    f"reusable PLE file size changed for {source.relative_path}"
                )
            digest = _sha256_file(temporary)
            if digest != source.sha256:
                raise Qwen4PLEWriteError(
                    f"reusable PLE digest differs for {source.relative_path}"
                )
            os.replace(temporary, destination)
    except Qwen4PLEWriteError:
        raise
    except OSError as exc:
        raise Qwen4PLEWriteError(
            f"could not hardlink reusable PLE file {source.relative_path}"
        ) from exc
    return source.size_bytes, source.sha256


def write_qwen4_ple_provider(
    source_dir: str | Path,
    package_dir: str | Path,
    *,
    expected: Qwen4PLEWriteContract,
    reuse_from: str | Path | None = None,
) -> tuple[dict, list[dict]]:
    """Write raw PLE rows and return the component plus file identities.

    ``expected`` is derived from the validated architecture contract. The
    source index selects storage locations only; it cannot redefine model
    geometry or hashing semantics.
    """
    package_root = Path(package_dir)
    tables = inspect_qwen4_ple_source(source_dir, expected=expected)
    reused = (
        inspect_qwen4_ple_reuse(
            reuse_from,
            expected=expected,
            target_dir=package_root,
        )
        if reuse_from is not None
        else ()
    )
    reuse_by_index = {shard.index: shard for shard in reused}
    shards = []
    files = []
    for table in tables:
        destination = _package_destination(package_root, table.relative_path)
        reuse = reuse_by_index.get(table.index)
        if reuse is None:
            size_bytes, digest = _copy_tensor_payload(
                table.source_path,
                table.header,
                destination,
            )
        else:
            if reuse.relative_path != table.relative_path:
                raise Qwen4PLEWriteError(
                    f"reusable PLE shard path differs at index {table.index}"
                )
            size_bytes, digest = _hardlink_reused_payload(reuse, destination)
        files.append(
            {
                "path": table.relative_path,
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )
        shards.append(
            {
                "index": table.index,
                "path": table.relative_path,
                "row_start": table.index * expected.rows_per_shard,
                "row_count": expected.rows_per_shard,
            }
        )

    component = _provider_component(expected)
    if component["shards"] != shards:
        raise AssertionError("PLE provider shard construction drifted")
    return component, files
