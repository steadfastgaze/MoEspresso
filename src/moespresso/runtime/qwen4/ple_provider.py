"""Manifest-backed direct-row storage for Qwen4 PLE embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import threading

import mlx.core as mx
import numpy as np

from moespresso.runtime.pread_into import (
    PreadFileCache,
    PreadIntoError,
    pread_view_cached,
)
from moespresso.runtime.pooled_load_batch import submit_loads
from moespresso.runtime.qwen4.ple_contract import (
    Qwen4PLEProviderContract,
    Qwen4PLEProviderError,
    parse_qwen4_ple_component_contract,
    qwen4_ple_contract_mismatches,
)


_DEFAULT_MAX_OPEN_FILES = 128
_PARALLEL_READ_MIN_RUNS = 1_024
_READ_WORKERS = 4


@dataclass(frozen=True)
class _Qwen4PLEReadRun:
    shard: Qwen4PLEShardRecord
    unique_offset: int
    row_start: int
    row_count: int


@dataclass(frozen=True)
class Qwen4PLEShardRecord:
    """One contiguous range of physical PLE rows and its file identity."""

    index: int
    path: str
    file_path: Path
    row_start: int
    row_count: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Qwen4PLEProviderLayout:
    """Immutable physical and hashing metadata for a PLE table set."""

    layer_index: int
    dtype: str
    row_width: int
    row_bytes: int
    logical_rows: int
    padded_rows: int
    rows_per_shard: int
    ngram_size: int
    heads_per_ngram: int
    multipliers: tuple[int, ...]
    table_sizes: tuple[int, ...]
    table_offsets: tuple[int, ...]
    shards: tuple[Qwen4PLEShardRecord, ...]


_SHARD_FIELDS = frozenset({"index", "path", "row_start", "row_count"})


def _fail(message: str) -> Qwen4PLEProviderError:
    return Qwen4PLEProviderError(message)


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{field} must be an integer >= {minimum}")
    return value


def _declared_path(package_dir: Path, value: object, *, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise _fail(f"{field} must be a non-empty string")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    parts = value.split("/")
    if (
        "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise _fail(f"{field} escapes the package root or is not canonical")
    try:
        root = package_dir.resolve()
        resolved = (root / value).resolve()
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _fail(f"{field} escapes the package root or cannot be resolved") from exc
    return value, resolved


def _file_identities(
    manifest: Mapping[str, object],
    package_dir: Path,
) -> dict[str, tuple[Path, int, str]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise _fail("files must be an array of file identities")
    identities: dict[str, tuple[Path, int, str]] = {}
    for index, identity in enumerate(files):
        if not isinstance(identity, Mapping):
            raise _fail(f"files[{index}] must be an object")
        if set(identity) != {"path", "size_bytes", "sha256"}:
            raise _fail(f"files[{index}] has invalid identity fields")
        path, resolved = _declared_path(
            package_dir,
            identity.get("path"),
            field=f"files[{index}].path",
        )
        if path in identities:
            raise _fail(f"duplicate file identity for {path}")
        size_bytes = _integer(
            identity.get("size_bytes"),
            field=f"files[{index}].size_bytes",
        )
        digest = identity.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise _fail(f"files[{index}].sha256 must be a lowercase SHA-256 digest")
        identities[path] = (resolved, size_bytes, digest)
    return identities


def parse_qwen4_ple_provider(
    manifest: Mapping[str, object],
    package_dir: str | Path,
    *,
    expected: Qwen4PLEProviderContract,
) -> Qwen4PLEProviderLayout:
    """Parse and validate the ``qwen4_ple_provider_v1`` package component.

    Hashes remain the package verification gate's responsibility. This parser
    binds every shard to its canonical top-level file identity and checks its
    current byte size before the runtime opens it. ``expected`` must come from
    the independently validated model contract, not from ``ple_provider``.
    """
    if not isinstance(manifest, Mapping):
        raise _fail("package manifest must be an object")
    if not isinstance(expected, Qwen4PLEProviderContract):
        raise TypeError("expected must be a Qwen4PLEProviderContract")
    root = Path(package_dir)
    component = manifest.get("ple_provider")
    if not isinstance(component, Mapping):
        raise _fail("ple_provider must be an object")
    actual_contract = parse_qwen4_ple_component_contract(component)
    raw_shards = component.get("shards")
    assert isinstance(raw_shards, list)
    mismatches = qwen4_ple_contract_mismatches(actual_contract, expected)
    if mismatches:
        raise _fail(
            "ple_provider does not match the validated model contract: " + ", ".join(mismatches)
        )

    identities = _file_identities(manifest, root)
    shards = []
    seen_paths = set()
    expected_size = actual_contract.rows_per_shard * actual_contract.row_bytes
    for position, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping) or set(raw) != _SHARD_FIELDS:
            raise _fail(f"shards[{position}] has invalid fields")
        index = _integer(raw.get("index"), field=f"shards[{position}].index")
        row_start = _integer(raw.get("row_start"), field=f"shards[{position}].row_start")
        row_count = _integer(raw.get("row_count"), field=f"shards[{position}].row_count", minimum=1)
        if index != position:
            raise _fail("shards must be ordered by contiguous index")
        if (
            row_start != position * actual_contract.rows_per_shard
            or row_count != actual_contract.rows_per_shard
        ):
            raise _fail("shards must partition padded rows uniformly and in order")
        path = raw.get("path")
        if not isinstance(path, str) or path not in identities:
            raise _fail(f"shards[{position}].path has no top-level file identity")
        if path in seen_paths:
            raise _fail(f"duplicate shard path {path}")
        seen_paths.add(path)
        file_path, size_bytes, digest = identities[path]
        if size_bytes != expected_size:
            raise _fail(f"shard {path} identity size does not match its row range")
        if not file_path.is_file():
            raise _fail(f"shard file {path} is missing")
        try:
            actual_size = file_path.stat().st_size
        except OSError as exc:
            raise _fail(f"could not stat shard file {path}") from exc
        if actual_size != size_bytes:
            raise _fail(f"shard file {path} has size {actual_size}, expected {size_bytes}")
        shards.append(
            Qwen4PLEShardRecord(
                index=index,
                path=path,
                file_path=file_path,
                row_start=row_start,
                row_count=row_count,
                size_bytes=size_bytes,
                sha256=digest,
            )
        )

    layout = Qwen4PLEProviderLayout(
        layer_index=actual_contract.layer_index,
        dtype=actual_contract.dtype,
        row_width=actual_contract.row_width,
        row_bytes=actual_contract.row_bytes,
        logical_rows=actual_contract.logical_rows,
        padded_rows=actual_contract.padded_rows,
        rows_per_shard=actual_contract.rows_per_shard,
        ngram_size=actual_contract.ngram_size,
        heads_per_ngram=actual_contract.heads_per_ngram,
        multipliers=actual_contract.multipliers,
        table_sizes=actual_contract.table_sizes,
        table_offsets=actual_contract.table_offsets,
        shards=tuple(shards),
    )
    return layout


class Qwen4PLEDirectRowProvider:
    """Materialize only requested BF16 PLE rows through direct file reads."""

    def __init__(
        self,
        layout: Qwen4PLEProviderLayout,
        *,
        file_cache: PreadFileCache | None = None,
        max_open_files: int = _DEFAULT_MAX_OPEN_FILES,
    ) -> None:
        if file_cache is not None and max_open_files != _DEFAULT_MAX_OPEN_FILES:
            raise ValueError("max_open_files cannot be combined with file_cache")
        self.layout = layout
        self.file_cache = file_cache or PreadFileCache(max_open=max_open_files)
        self._owns_cache = file_cache is None
        self._provider_lock = threading.Lock()
        self._read_executor: ThreadPoolExecutor | None = None
        self._closed = False

    def close(self) -> None:
        """Close resources owned by this provider."""
        with self._provider_lock:
            if self._closed:
                return
            self._closed = True
            read_executor = self._read_executor
        if read_executor is not None:
            read_executor.shutdown(wait=True, cancel_futures=True)
        if self._owns_cache:
            self.file_cache.close_all()

    def _read_plan(self, unique_ids: np.ndarray) -> tuple[_Qwen4PLEReadRun, ...]:
        runs = []
        cursor = 0
        while cursor < unique_ids.size:
            global_row = int(unique_ids[cursor])
            shard_index = global_row // self.layout.rows_per_shard
            shard = self.layout.shards[shard_index]
            run_end = cursor + 1
            while (
                run_end < unique_ids.size
                and int(unique_ids[run_end]) == int(unique_ids[run_end - 1]) + 1
                and int(unique_ids[run_end]) < shard.row_start + shard.row_count
            ):
                run_end += 1
            runs.append(
                _Qwen4PLEReadRun(
                    shard=shard,
                    unique_offset=cursor,
                    row_start=global_row,
                    row_count=run_end - cursor,
                )
            )
            cursor = run_end
        return tuple(runs)

    def _read_runs(
        self,
        view: memoryview,
        runs: tuple[_Qwen4PLEReadRun, ...],
    ) -> None:
        """Fill disjoint row ranges and join every writer before publication."""
        def read_part(part: tuple[_Qwen4PLEReadRun, ...]) -> None:
            for run in part:
                pread_view_cached(
                    view,
                    run.shard.file_path,
                    file_offset=(run.row_start - run.shard.row_start) * self.layout.row_bytes,
                    nbytes=run.row_count * self.layout.row_bytes,
                    dst_offset=run.unique_offset * self.layout.row_bytes,
                    cache=self.file_cache,
                )

        if len(runs) < _PARALLEL_READ_MIN_RUNS:
            read_part(runs)
            return
        with self._provider_lock:
            if self._closed:
                raise _fail("PLE row provider is closed")
            if self._read_executor is None:
                self._read_executor = ThreadPoolExecutor(
                    max_workers=_READ_WORKERS,
                    thread_name_prefix="moespresso-qwen4-ple-read",
                )
            executor = self._read_executor
        # Coarse batches bound queue growth and retain sorted reads within each
        # range. Workers touch host buffers only; MLX publication stays on the caller.
        width = (len(runs) + _READ_WORKERS - 1) // _READ_WORKERS
        batch = submit_loads(
            executor,
            (
                lambda part=runs[start:start + width]: read_part(part)
                for start in range(0, len(runs), width)
            ),
        )
        batch.wait()

    def lookup(self, row_ids: mx.array) -> mx.array:
        """Read sorted unique rows, then restore the caller's shape and order."""
        if not isinstance(row_ids, mx.array):
            raise TypeError("row_ids must be an MLX array")
        mx.eval(row_ids)
        host_ids = np.asarray(row_ids)
        if not np.issubdtype(host_ids.dtype, np.integer) or np.issubdtype(host_ids.dtype, np.bool_):
            raise TypeError("row_ids must contain integers")
        if host_ids.size == 0:
            return mx.zeros((*host_ids.shape, self.layout.row_width), dtype=mx.bfloat16)
        if np.any(host_ids < 0):
            raise ValueError("row_ids must be nonnegative")
        if np.any(host_ids >= self.layout.logical_rows):
            raise ValueError("row_ids exceed the logical PLE row range")

        unique_ids, inverse = np.unique(host_ids.reshape(-1), return_inverse=True)
        unique_ids = unique_ids.astype(np.int64, copy=False)
        staging = mx.zeros(
            (unique_ids.size, self.layout.row_width),
            dtype=mx.bfloat16,
        )
        mx.eval(staging)
        view = memoryview(staging).cast("B")

        try:
            self._read_runs(view, self._read_plan(unique_ids))
        except (OSError, PreadIntoError) as exc:
            raise _fail(f"failed to read selected PLE rows: {exc}") from exc

        mx.eval(staging)
        restored = staging[mx.array(inverse, dtype=mx.int64)]
        return restored.reshape(*host_ids.shape, self.layout.row_width)
