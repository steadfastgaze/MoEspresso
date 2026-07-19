"""Disk prompt-cache tier (frontier checkpoints): writer and read path.

Restore a served model's prompt-cache prefix from disk and prefill only the
suffix, after a restart or from a new session sharing a long prompt prefix.
Serving enables the store by default (``MOESPRESSO_DISK_KV=off`` disables
it). The store is single-process per root by contract: one process owns a
disk root through a non-blocking file lock, so the index is a single JSON
file rewritten under that lock with a temp file and an atomic rename.

Boundaries this module keeps explicit:

- MoEspresso owns token-prefix identity and metadata trust. The safety key is the
  in-memory cache scope (the serve 6-tuple) plus the cache-class list and the
  disk schema version, joined with the token-prefix hash and the prefix length.
- MLX owns prompt-cache payload serialization. Payloads are direct, uncompressed
  safetensors written through a per-leaf schema so the DeepSeek-V4 composite state
  tree round-trips: that tree carries None slots (which stock save refuses) and
  zero-size arrays for empty frontier buffers (which safetensors refuses). The
  schema records each leaf as array, empty, or none; the loader rebuilds the
  empty and none leaves from the schema and grafts the arrays from the file.
- The loader is fail-closed. Every load re-checks the embedded safetensors
  metadata against the index entry, refuses a payload whose stored prefix length
  does not equal the claimed key (no rounding a key down), refuses a class list
  the registry does not know, and refuses a corrupt or truncated payload before
  any cache reaches the model. A refusal quarantines the entry and returns the
  engine to cold serving.

The frontier writer runs during prefill only: token accounting proposes
aligned frontiers, every positional cache must independently report exactly
the frontier offset before a write, and by default only frontiers within the
write-depth cap are written, because the shallow shared-prefix region is
where cross-session restores land while deep cumulative snapshots only add
write traffic. The manual checkpoint helper creates fixtures through the
same write path.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import struct
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = "moespresso-disk-kv-v1"

# The safetensors metadata leaf kinds. A payload leaf is either a real array (its
# bytes are in the file), an empty array (zero-size, shape and dtype recorded so
# the loader can rebuild it), or a None slot (an absent buffer at a frontier).
_LEAF_ARRAY = "array"
_LEAF_EMPTY = "empty"
_LEAF_NONE = "none"


def _default_log(line: str) -> None:
    """One plain operator line per checkpoint decision, on the serve stderr."""
    import sys

    print(line, file=sys.stderr, flush=True)


class DiskKVError(RuntimeError):
    """Base class for disk prompt-cache failures."""


class DiskKVInvalidPayload(DiskKVError):
    """A payload path is missing, corrupt, or failed to load.

    This is corrupt or absent evidence: the file cannot be read as a cache.
    """


class DiskKVMetadataMismatch(DiskKVError):
    """A loaded payload's embedded metadata does not match the index entry.

    This is a trust failure: the file loaded, but its safety key disagrees with
    what the index claims, so the checkpoint is not the one the caller asked for.
    """


class DiskKVRootLocked(DiskKVError):
    """The disk root is owned by another process."""


class DiskKVStrideError(DiskKVError):
    """The configured checkpoint stride is not a positive multiple of 256."""


# --- canonical hashing -------------------------------------------------------


def _json_dumps(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _json_bytes(data: dict) -> bytes:
    return _json_dumps(data).encode()


def _hash_json(domain: str, data: dict) -> str:
    h = hashlib.sha256()
    h.update(domain.encode())
    h.update(b"\0")
    h.update(_json_bytes(data))
    return h.hexdigest()


def token_prefix_hash(tokens: list[int] | tuple[int, ...]) -> str:
    """Hash a claimed token prefix.

    The input uses a binary format so a future implementation in another language
    can reproduce it exactly. The domain string and a fixed version guard against a
    hash from another purpose colliding. The token count is hashed first, then the
    token ids as unsigned little-endian 64-bit integers. Only the claimed prefix
    is hashed; the caller slices before calling.
    """
    h = hashlib.sha256()
    h.update(b"moespresso.disk-kv.token-prefix.v1\0")
    h.update(struct.pack("<Q", len(tokens)))
    for token in tokens:
        if token < 0:
            raise ValueError("token ids must be non-negative")
        h.update(struct.pack("<Q", int(token)))
    return h.hexdigest()


def _dsv4_banded_prefill_offset_route() -> bool:
    """Whether the DS4 banded-prefill offset route is enabled for this process.

    The offset route changes the attention values on every chunk past the
    first, so hidden states downstream of any later chunk differ from the
    composed lattice. A checkpoint written under one route must never
    restore under the other; the scope carries the route flag so the two
    rails key separate buckets. Frontier geometry is unchanged either way.
    """
    from moespresso.runtime.deepseek_v4.model import (
        _banded_prefill_offset_enabled,
    )

    return _banded_prefill_offset_enabled()


def build_cache_scope(model_key: tuple, cache_class_names: tuple[str, ...]) -> dict:
    """Fields that decide whether one checkpoint may restore for another request.

    ``model_key`` is the in-memory serve 6-tuple from
    ``prefix_cache.cache_model_key`` (artifact id, rendering id, live KV format,
    group size, quantized KV start, cache payload kind). The scope extends it with
    the cache-class list and the disk schema version, so a checkpoint written for
    one cache-class layout never restores into another. Math-affecting
    route flags join the scope only when engaged: a disabled route keeps
    the recorded scope bytes, and an engaged rail keys its own bucket, so
    a checkpoint written under one rail never restores under the other.
    """
    scope = {
        "schema_version": SCHEMA_VERSION,
        "package_manifest_artifact_id": model_key[0],
        "rendering_identity": model_key[1],
        "live_kv_format": model_key[2],
        "kv_group_size": model_key[3],
        "quantized_kv_start": model_key[4],
        "cache_payload_kind": model_key[5],
        "cache_class_names": list(cache_class_names),
    }
    if _dsv4_banded_prefill_offset_route():
        scope["dsv4_banded_prefill_offset"] = True
    return scope


def scope_hash(scope: dict) -> str:
    """Stable bucket id for a cache scope."""
    return _hash_json("moespresso.disk-kv.cache-scope.v1", scope)


def validate_stride(stride: int) -> int:
    """A checkpoint stride must be a positive multiple of the 256-token frontier."""
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise DiskKVStrideError("disk KV stride must be an integer")
    if stride <= 0 or stride % 256 != 0:
        raise DiskKVStrideError(
            f"disk KV stride must be a positive multiple of 256, got {stride}")
    return stride


# --- root lock ---------------------------------------------------------------


class DiskKVRootLock:
    """Exclusive, non-blocking process lock for one disk root.

    One process owns a disk root through ``fcntl.flock(LOCK_EX | LOCK_NB)`` on a
    lockfile. The lock is acquired before model load and held for the store
    lifetime. There is no blocking mode and no lock stealing: a second owner is
    refused loudly at startup rather than waiting.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.fd: int | None = None

    def acquire(self) -> "DiskKVRootLock":
        if self.fd is not None:
            return self
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "moespresso-disk-kv.lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            if e.errno in (errno.EACCES, errno.EAGAIN):
                raise DiskKVRootLocked(
                    f"disk KV root is already locked by another process: {self.root}"
                ) from e
            raise
        self.fd = fd
        return self

    @property
    def locked(self) -> bool:
        return self.fd is not None

    def close(self) -> None:
        if self.fd is None:
            return
        fd = self.fd
        self.fd = None
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# --- entry schema ------------------------------------------------------------


@dataclass(frozen=True)
class DiskKVEntry:
    """Index metadata for one saved prompt-cache payload.

    The content fields (scope, token count, prefix hash, class list, payload
    path) define identity. The operational fields (created, last used, hit count)
    describe use and never enter any hash.
    """

    schema_version: str
    scope: dict
    scope_hash: str
    cache_id: str
    token_count: int
    token_prefix_hash: str
    payload_path: str
    payload_bytes: int
    cache_class_names: tuple[str, ...]
    reason: str
    created_at: int
    last_used_at: int
    hit_count: int
    # An optional client-supplied grouping hint (metadata.moespresso_cache_key). It
    # is an operational field: it is stored, used to prefer entries of the same
    # session for eviction, and never enters any content hash or the safety key. It
    # never authorizes a load; the token-prefix and scope validation decide that.
    session_cache_key: str | None = None

    @classmethod
    def from_tokens(
        cls,
        scope: dict,
        tokens: list[int],
        *,
        payload_path: str,
        payload_bytes: int,
        cache_class_names: tuple[str, ...],
        cache_id: str | None = None,
        reason: str = "aligned_frontier",
        session_cache_key: str | None = None,
        now: int = 0,
    ) -> "DiskKVEntry":
        prefix_hash = token_prefix_hash(tokens)
        sid = scope_hash(scope)
        return cls(
            schema_version=SCHEMA_VERSION,
            scope=dict(scope),
            scope_hash=sid,
            cache_id=cache_id or prefix_hash,
            token_count=len(tokens),
            token_prefix_hash=prefix_hash,
            payload_path=payload_path,
            payload_bytes=int(payload_bytes),
            cache_class_names=tuple(cache_class_names),
            reason=reason,
            created_at=now,
            last_used_at=now,
            hit_count=0,
            session_cache_key=session_cache_key,
        )

    def to_json_obj(self) -> dict:
        data = asdict(self)
        data["cache_class_names"] = list(self.cache_class_names)
        return data

    @classmethod
    def from_json_obj(cls, raw: dict) -> "DiskKVEntry":
        raw = dict(raw)
        raw["cache_class_names"] = tuple(raw["cache_class_names"])
        # Older index files predate the session hint; default it absent.
        raw.setdefault("session_cache_key", None)
        return cls(**raw)


# --- JSON index --------------------------------------------------------------


class DiskKVIndex:
    """One JSON file per disk root, rewritten atomically under the root lock.

    The store is single-process by contract and payloads are gigabytes each, so
    the index holds at most hundreds of entries. A JSON file loaded whole and
    rewritten with a temp file and an atomic rename gives crash consistency at no
    dependency cost. Every mutation loads the current file, edits in memory, and
    replaces the file. Mutable operational fields stay out of the entry hash.
    """

    INDEX_NAME = "index.json"

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.INDEX_NAME
        if not self.path.exists():
            self._write({"schema_version": SCHEMA_VERSION, "entries": []})
        else:
            data = self._read()
            stored = data.get("schema_version")
            if stored != SCHEMA_VERSION:
                raise DiskKVError(
                    f"unsupported disk KV index schema {stored!r}")

    def close(self) -> None:
        # The JSON index holds no file handles between calls; nothing to release.
        return None

    def _read(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(_json_dumps(data))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def entries(self) -> list[DiskKVEntry]:
        data = self._read()
        return [DiskKVEntry.from_json_obj(obj) for obj in data.get("entries", [])]

    def _entry_identity(self, entry: DiskKVEntry) -> tuple:
        return (entry.scope_hash, entry.token_count, entry.token_prefix_hash)

    def put(self, entry: DiskKVEntry) -> None:
        data = self._read()
        identity = self._entry_identity(entry)
        kept = [
            obj for obj in data.get("entries", [])
            if self._entry_identity(DiskKVEntry.from_json_obj(obj)) != identity
        ]
        kept.append(entry.to_json_obj())
        data["entries"] = kept
        self._write(data)

    def find_longest(self, scope: dict, tokens: list[int]) -> DiskKVEntry | None:
        sid = scope_hash(scope)
        entries = [e for e in self.entries() if e.scope_hash == sid]
        # Descending token count: the longest exact prefix wins.
        for entry in sorted(entries, key=lambda e: e.token_count, reverse=True):
            if entry.token_count > len(tokens):
                continue
            if entry.token_prefix_hash == token_prefix_hash(tokens[:entry.token_count]):
                return entry
        return None

    def mark_used(self, entry: DiskKVEntry, *, now: int | None = None) -> DiskKVEntry:
        now = int(time.time()) if now is None else int(now)
        updated = replace(entry, last_used_at=now, hit_count=entry.hit_count + 1)
        self.put(updated)
        return updated

    def remove(self, entry: DiskKVEntry) -> None:
        data = self._read()
        identity = self._entry_identity(entry)
        data["entries"] = [
            obj for obj in data.get("entries", [])
            if self._entry_identity(DiskKVEntry.from_json_obj(obj)) != identity
        ]
        self._write(data)


# --- payload codec (per-leaf schema + non-empty arrays) ----------------------

_DTYPE_NAMES = (
    "float32", "float16", "bfloat16", "int32", "uint32",
    "int8", "uint8", "int16", "uint16", "int64", "uint64", "bool_",
)


def _dtype_map():
    import mlx.core as mx

    table = {}
    for name in _DTYPE_NAMES:
        dtype = getattr(mx, name, None)
        if dtype is not None:
            table[name] = dtype
            table[str(dtype)] = dtype
    return table


def _collect_leaves(tree, out):
    """Flatten the cache state tree into a stable list of (path, leaf) pairs.

    The state getter returns nested tuples and lists with array, None, and
    occasionally scalar leaves. A stable positional path lets the loader rebuild
    the identical structure, so the tree does not need mlx.utils here.
    """
    def _walk(node, path):
        if isinstance(node, (tuple, list)):
            for i, child in enumerate(node):
                _walk(child, path + (i,))
        else:
            out.append((path, node))

    _walk(tree, ())


def _rebuild_tree(schema, arrays):
    """Rebuild the nested state tree from the leaf schema and the loaded arrays."""
    import mlx.core as mx

    dtype_map = _dtype_map()
    # Reconstruct as nested lists indexed by the positional path, then leave the
    # lists in place (the cache state setters accept lists as well as tuples).
    root: list = []

    def _ensure(container, index):
        while len(container) <= index:
            container.append(None)

    for key, kind, shape, dtype in schema:
        path = tuple(int(p) for p in key.split("."))
        if kind == _LEAF_ARRAY:
            leaf = arrays[key]
        elif kind == _LEAF_EMPTY:
            leaf = mx.zeros(tuple(shape), dtype=dtype_map.get(dtype, mx.float32))
        elif kind == _LEAF_NONE:
            leaf = None
        else:
            raise DiskKVInvalidPayload(f"unknown payload leaf kind: {kind!r}")
        container = root
        for depth, index in enumerate(path):
            _ensure(container, index)
            if depth == len(path) - 1:
                container[index] = leaf
            else:
                if not isinstance(container[index], list):
                    container[index] = []
                container = container[index]
    return root


def encode_cache_payload(cache_state_trees) -> tuple[dict, list]:
    """Serialize per-layer cache state into arrays plus a per-leaf schema.

    Returns ``(arrays, schema)`` where ``arrays`` maps a dotted positional key to
    a non-empty MLX array and ``schema`` is a list of ``[key, kind, shape, dtype]``
    covering every leaf including the empty and None ones.
    """
    import mlx.core as mx

    arrays: dict = {}
    schema: list = []
    for layer_index, tree in enumerate(cache_state_trees):
        leaves: list = []
        _collect_leaves(tree, leaves)
        for path, leaf in leaves:
            key = ".".join(str(p) for p in (layer_index,) + path)
            if isinstance(leaf, mx.array):
                if leaf.size == 0:
                    schema.append([key, _LEAF_EMPTY, list(leaf.shape), str(leaf.dtype)])
                else:
                    arrays[key] = leaf
                    schema.append([key, _LEAF_ARRAY, list(leaf.shape), str(leaf.dtype)])
            elif leaf is None:
                schema.append([key, _LEAF_NONE, None, None])
            else:
                raise DiskKVInvalidPayload(
                    f"cache state carries a non-array, non-None leaf at {key}: "
                    f"{type(leaf).__name__}")
    return arrays, schema


def decode_cache_payload(arrays: dict, schema: list) -> list:
    """Rebuild per-layer state trees from arrays and the leaf schema.

    The schema keys are ``<layer>.<path...>``. Leaves are grouped by their leading
    layer index and each layer's sub-tree is rebuilt positionally.
    """
    by_layer: dict[int, list] = {}
    for entry in schema:
        key = entry[0]
        layer_index = int(key.split(".", 1)[0])
        sub_key = key.split(".", 1)[1] if "." in key else ""
        by_layer.setdefault(layer_index, []).append(
            [sub_key, entry[1], entry[2], entry[3]])
    layer_arrays: dict[int, dict] = {}
    for key, value in arrays.items():
        layer_index = int(key.split(".", 1)[0])
        sub_key = key.split(".", 1)[1] if "." in key else ""
        layer_arrays.setdefault(layer_index, {})[sub_key] = value
    trees = []
    for layer_index in sorted(by_layer):
        trees.append(_rebuild_tree(by_layer[layer_index], layer_arrays.get(layer_index, {})))
    return trees


def _meta_state_to_json(meta_state_trees) -> str:
    def _norm(node):
        if isinstance(node, (tuple, list)):
            return [_norm(child) for child in node]
        return node

    return json.dumps([_norm(tree) for tree in meta_state_trees])


def _meta_state_from_json(text: str) -> list:
    raw = json.loads(text)

    def _norm(node):
        if isinstance(node, list):
            return tuple(_norm(child) for child in node)
        return node

    return [_norm(tree) for tree in raw]


def save_prompt_cache_payload(
    root: Path | str,
    cache_id: str,
    *,
    cache_state_trees,
    meta_state_trees,
    safety_metadata: dict[str, str],
    save_fn: Callable | None = None,
) -> tuple[str, int]:
    """Write a prompt cache to a direct safetensors payload with a leaf schema.

    The payload holds only non-empty arrays. Its safetensors metadata carries the
    per-leaf schema, the per-layer meta_state as JSON, and the full safety key so
    a load can re-check the file against the index entry. The write is a temp file,
    fsync, and an atomic rename.
    """
    import mlx.core as mx

    root = Path(root)
    rel_path = Path("payloads") / cache_id[:2] / f"{cache_id}.safetensors"
    final_path = root / rel_path
    tmp_path = final_path.with_suffix(".tmp.safetensors")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    arrays, schema = encode_cache_payload(cache_state_trees)
    metadata = dict(safety_metadata)
    metadata["leaf_schema"] = _json_dumps({"schema": schema})
    metadata["meta_state"] = _meta_state_to_json(meta_state_trees)

    if save_fn is None:
        save_fn = mx.save_safetensors
    save_fn(str(tmp_path), arrays, metadata)
    with open(tmp_path, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp_path, final_path)
    return rel_path.as_posix(), final_path.stat().st_size


def load_prompt_cache_payload(
    root: Path | str,
    payload_path: str,
    *,
    load_fn: Callable | None = None,
) -> tuple[list, list, dict]:
    """Load a payload into (state trees, meta_state trees, safety metadata).

    Fails closed on a missing or corrupt file. The safetensors read raises before
    any cache reaches the model, so a truncated payload is refused here.
    """
    import mlx.core as mx

    path = Path(root) / payload_path
    if not path.exists():
        raise DiskKVInvalidPayload(f"missing prompt-cache payload: {payload_path}")
    if load_fn is None:
        load_fn = mx.load
    try:
        arrays, metadata = load_fn(str(path), return_metadata=True)
        schema = json.loads(metadata["leaf_schema"])["schema"]
        state_trees = decode_cache_payload(arrays, schema)
        meta_state_trees = _meta_state_from_json(metadata["meta_state"])
    except DiskKVError:
        raise
    except Exception as e:
        raise DiskKVInvalidPayload(
            f"corrupt prompt-cache payload: {payload_path}") from e
    return state_trees, meta_state_trees, metadata


# --- safety-key metadata gate ------------------------------------------------


def build_safety_metadata(entry: DiskKVEntry) -> dict[str, str]:
    """The safety key embedded in the payload, mirrored by the index entry."""
    return {
        "cache_schema_version": entry.schema_version,
        "scope_hash": entry.scope_hash,
        "token_count": str(entry.token_count),
        "token_prefix_hash": entry.token_prefix_hash,
        "cache_class_names": _json_dumps({"classes": list(entry.cache_class_names)}),
    }


def validate_payload_metadata(entry: DiskKVEntry, metadata: dict[str, Any]) -> None:
    """Re-check the embedded safety key against the index entry.

    The prefix-length check is exact: a payload holding a different token count
    than the entry claims is refused, so a key can never be rounded down to a
    checkpoint that does not describe the payload.
    """
    expected = build_safety_metadata(entry)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise DiskKVMetadataMismatch(
                f"payload metadata mismatch for {key}: "
                f"expected {value!r}, got {metadata.get(key)!r}")


def validate_cache_classes(
    stored_classes: tuple[str, ...],
    *,
    expected_classes: tuple[str, ...],
    registry: set[str],
) -> None:
    """Refuse a class list the registry does not know or that disagrees.

    Two gates fire independently. The registry gate refuses a class name the
    loader cannot reconstruct. The expected-list gate refuses a payload whose
    class layout is not the one the live model builds.
    """
    for name in stored_classes:
        if name not in registry:
            raise DiskKVMetadataMismatch(f"unregistered cache class: {name}")
    if tuple(stored_classes) != tuple(expected_classes):
        raise DiskKVMetadataMismatch(
            f"cache-class list mismatch: stored {list(stored_classes)} != "
            f"expected {list(expected_classes)}")


# --- store -------------------------------------------------------------------


@dataclass(frozen=True)
class DiskKVHit:
    """A disk restore plus the suffix that still needs generation."""

    entry: DiskKVEntry
    prompt_cache: Any
    suffix_tokens: list[int]
    cached_tokens: int


class DiskCheckpointStore:
    """JSON metadata index plus direct safetensors payloads, read path only.

    ``find_longest`` selects the longest exact token-prefix checkpoint in scope.
    ``restore`` loads and validates one, reconstructs the live caches through the
    model's own ``make_cache`` (the explicit registry), grafts the state, and
    returns the suffix to prefill. Any validation failure quarantines the entry
    and raises, and the caller falls back to cold serving.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        index: DiskKVIndex | None = None,
        load_payload_fn: Callable | None = None,
        save_payload_fn: Callable | None = None,
        root_lock: DiskKVRootLock | None = None,
        stride: int | None = None,
        budget_bytes: int | None = None,
        write_depth_tokens: int | None = None,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = index if index is not None else DiskKVIndex(self.root)
        self.load_payload_fn = load_payload_fn or load_prompt_cache_payload
        self.save_payload_fn = save_payload_fn or save_prompt_cache_payload
        self.root_lock = root_lock
        # The frontier writer reads the stride to place checkpoints and the
        # write-depth cap to bound them; the read path never uses either.
        # None stride means no writer runs (the read-only shape).
        self.stride = validate_stride(stride) if stride is not None else None
        self.write_depth_tokens = write_depth_tokens
        # None means an unbounded store. A positive cap evicts least-recently-used
        # entries under the root lock before a write that would exceed it.
        if budget_bytes is not None and budget_bytes <= 0:
            raise DiskKVError("disk KV byte budget must be positive when set")
        self.budget_bytes = budget_bytes
        self._log = log_fn or _default_log
        # Session counters since store open, surfaced on /health.
        self.restores = 0
        self.writes = 0
        self.evictions = 0
        self.quarantines = 0
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.index.close()
        finally:
            if self.root_lock is not None:
                self.root_lock.close()

    def stats(self, *, last_event: str | None = None) -> dict:
        entries = self.index.entries()
        return {
            "enabled": True,
            "root": str(self.root),
            "stride": self.stride,
            "entries": len(entries),
            "payload_bytes": sum(entry.payload_bytes for entry in entries),
            "budget_bytes": self.budget_bytes,
            "restores": self.restores,
            "writes": self.writes,
            "evictions": self.evictions,
            "quarantines": self.quarantines,
            "lock_active": bool(
                self.root_lock is not None and getattr(self.root_lock, "locked", False)
            ),
            "last_event": last_event,
        }

    def find_longest(self, scope: dict, tokens: list[int]) -> DiskKVEntry | None:
        return self.index.find_longest(scope, tokens)

    def restore(
        self,
        scope: dict,
        full_tokens: list[int],
        *,
        make_cache_fn: Callable[[], list],
        registry: set[str],
    ) -> DiskKVHit | None:
        """Restore the longest valid checkpoint for these tokens, or return None.

        Fail-closed order: find the longest exact prefix entry; load and validate
        the payload (integrity, then embedded metadata vs the index, then the
        prefix-length and class-list gates); reconstruct live caches and graft the
        state. A refusal quarantines the entry and re-raises the domain error.
        """
        entry = self.index.find_longest(scope, full_tokens)
        if entry is None:
            return None
        try:
            state_trees, meta_state_trees, metadata = self.load_payload_fn(
                self.root, entry.payload_path)
            validate_payload_metadata(entry, metadata)
            stored_classes = tuple(
                json.loads(metadata["cache_class_names"])["classes"])
            validate_cache_classes(
                stored_classes,
                expected_classes=entry.cache_class_names,
                registry=registry,
            )
            prompt_cache = self._reconstruct(
                make_cache_fn, state_trees, meta_state_trees, entry)
        except DiskKVError:
            self.quarantine(entry)
            raise
        updated = self.index.mark_used(entry)
        self.restores += 1
        self._log(f"[disk_kv] restore cached_tokens={entry.token_count}")
        return DiskKVHit(
            entry=updated,
            prompt_cache=prompt_cache,
            suffix_tokens=list(full_tokens[entry.token_count:]),
            cached_tokens=entry.token_count,
        )

    def _reconstruct(self, make_cache_fn, state_trees, meta_state_trees, entry):
        import mlx.core as mx

        caches = make_cache_fn()
        if len(caches) != len(state_trees):
            raise DiskKVMetadataMismatch(
                f"cache layer count mismatch: model built {len(caches)} caches, "
                f"payload holds {len(state_trees)}")
        caches = [
            _graft_target(cache, expected)
            for cache, expected in zip(caches, entry.cache_class_names)
        ]
        live_classes = tuple(type(c).__name__ for c in caches)
        if live_classes != tuple(entry.cache_class_names):
            raise DiskKVMetadataMismatch(
                f"live cache-class list {list(live_classes)} != entry "
                f"{list(entry.cache_class_names)}")
        for cache, state, meta in zip(caches, state_trees, meta_state_trees):
            cache.state = state
            cache.meta_state = meta
        arrays = []
        for cache in caches:
            _collect_arrays(cache.state, arrays)
        if arrays:
            mx.eval(*arrays)
        return caches

    def quarantine(self, entry: DiskKVEntry, *, reason: str = "invalid") -> None:
        """Drop an invalid index entry and move its payload aside if present."""
        self.index.remove(entry)
        self.quarantines += 1
        self._log(f"[disk_kv] quarantine reason={reason}")
        payload = self.root / entry.payload_path
        if not payload.exists():
            return
        quarantine_dir = self.root / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        target = quarantine_dir / payload.name
        if target.exists():
            target = quarantine_dir / f"{entry.cache_id}.{int(time.time())}.safetensors"
        shutil.move(str(payload), str(target))

    def cleanup_stale_temps(self) -> list[str]:
        """Delete leftover ``.tmp.safetensors`` files under the startup lock.

        A temp payload can only belong to a crashed previous owner because the
        root lock makes the store single-process. Serve startup calls this after
        acquiring the lock.
        """
        payload_root = self.root / "payloads"
        if not payload_root.exists():
            return []
        deleted: list[str] = []
        for path in sorted(payload_root.rglob("*.tmp.safetensors")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            path.unlink()
            deleted.append(rel)
        return deleted

    def cleanup_orphan_payloads(self) -> list[str]:
        """Delete payload files no index entry references, under the startup lock.

        A payload without an index entry is an orphan: it can only be left by a
        crash between the payload rename and the index append, or by an eviction
        that removed the entry before the payload delete. The root lock makes the
        store single-process, so a payload the current index does not name is dead.
        Serve startup calls this after acquiring the lock, alongside the stale-temp
        sweep. The quarantine directory is left alone; its aging is out of scope.
        """
        payload_root = self.root / "payloads"
        if not payload_root.exists():
            return []
        referenced = {
            (self.root / e.payload_path).resolve()
            for e in self.index.entries()
            if e.payload_path
        }
        deleted: list[str] = []
        for path in sorted(payload_root.rglob("*.safetensors")):
            if not path.is_file():
                continue
            if path.name.endswith(".tmp.safetensors"):
                continue
            if path.resolve() in referenced:
                continue
            rel = path.relative_to(self.root).as_posix()
            path.unlink()
            deleted.append(rel)
        return deleted

    def has_entry(self, scope: dict, tokens: list[int]) -> bool:
        """Whether an index entry already covers this exact scope and token prefix.

        The frontier writer reads this before a write so a checkpoint that already
        exists (from a previous session or an earlier request) is not rewritten.
        The check is on the same identity the index dedupes on: scope hash, token
        count, and token-prefix hash.
        """
        target = (scope_hash(scope), len(tokens), token_prefix_hash(tokens))
        for entry in self.index.entries():
            if (entry.scope_hash, entry.token_count, entry.token_prefix_hash) == target:
                return True
        return False

    def _evict_to_fit(self, incoming_bytes: int, *, keep_id: str) -> bool:
        """Make room under the byte budget for a new payload, under the root lock.

        Returns True when the store can hold ``incoming_bytes`` after evicting, and
        False when the incoming payload alone exceeds the whole budget (the caller
        then skips the write rather than evicting everything for one oversized
        payload). Eviction order is least-recently-used first: ``last_used_at``,
        then ``created_at`` as a tiebreak. The entry with ``keep_id`` (the one being
        written) is never a candidate. Each eviction removes the index entry first,
        then deletes the payload, so a crash mid-eviction leaves an orphan payload
        (cleaned up at startup) rather than a dangling index reference.
        """
        if self.budget_bytes is None:
            return True
        if incoming_bytes > self.budget_bytes:
            return False
        candidates = [e for e in self.index.entries() if e.cache_id != keep_id]
        resident = sum(e.payload_bytes for e in candidates)
        # Least-recently-used first: oldest last_used_at, then oldest created_at.
        candidates.sort(key=lambda e: (e.last_used_at, e.created_at))
        i = 0
        while resident + incoming_bytes > self.budget_bytes and i < len(candidates):
            victim = candidates[i]
            self.index.remove(victim)
            _delete_file(self.root / victim.payload_path)
            resident -= victim.payload_bytes
            self.evictions += 1
            self._log(
                f"[disk_kv] evict token_count={victim.token_count} "
                f"bytes={victim.payload_bytes}")
            i += 1
        return resident + incoming_bytes <= self.budget_bytes

    def write_checkpoint(
        self,
        scope: dict,
        tokens: list[int],
        *,
        cache_state_trees,
        meta_state_trees,
        cache_class_names: tuple[str, ...],
        reason: str,
        session_cache_key: str | None = None,
        now: int = 0,
    ) -> DiskKVEntry | None:
        """Write a checkpoint: the payload first, then the index entry last.

        The payload lands with a temp file, fsync, and an atomic rename, then the
        index entry is appended, so a crash between the two leaves an orphan payload
        rather than a dangling index reference. Any failure during the payload write
        or the index append cleans up the partial payload and re-raises, and the
        caller returns the engine to cold serving. Both the manual fixture writer
        and the frontier writer share this one path.

        Under a byte budget the payload is written first so its exact on-disk size is
        known, then least-recently-used entries are evicted to make room. A payload
        that alone exceeds the whole budget is deleted and the write is skipped
        (returns None); the store never evicts every other entry for one oversized
        payload.
        """
        if not tokens:
            raise DiskKVError("cannot checkpoint an empty token sequence")
        prefix_hash = token_prefix_hash(tokens)
        sid = scope_hash(scope)
        cache_id = _hash_json("moespresso.disk-kv.cache-id.v1", {
            "scope_hash": sid,
            "token_count": len(tokens),
            "token_prefix_hash": prefix_hash,
        })
        entry = DiskKVEntry.from_tokens(
            scope,
            tokens,
            payload_path="",  # filled after the payload write
            payload_bytes=0,
            cache_class_names=cache_class_names,
            cache_id=cache_id,
            reason=reason,
            session_cache_key=session_cache_key,
            now=now,
        )
        safety_metadata = build_safety_metadata(entry)
        payload_path, payload_bytes = self.save_payload_fn(
            self.root,
            cache_id,
            cache_state_trees=cache_state_trees,
            meta_state_trees=meta_state_trees,
            safety_metadata=safety_metadata,
        )
        entry = replace(entry, payload_path=payload_path, payload_bytes=payload_bytes)
        if not self._evict_to_fit(payload_bytes, keep_id=cache_id):
            _delete_file(self.root / payload_path)
            self._log(
                f"[disk_kv] skip reason=budget token_count={len(tokens)} "
                f"bytes={payload_bytes} budget={self.budget_bytes}")
            return None
        try:
            self.index.put(entry)
        except Exception:
            _delete_file(self.root / payload_path)
            raise
        self.writes += 1
        self._log(
            f"[disk_kv] write token_count={len(tokens)} bytes={payload_bytes}")
        return entry

    def write_manual_checkpoint(
        self,
        scope: dict,
        tokens: list[int],
        *,
        cache_state_trees,
        meta_state_trees,
        cache_class_names: tuple[str, ...],
        reason: str = "manual_fixture",
        session_cache_key: str | None = None,
        now: int = 0,
    ) -> DiskKVEntry | None:
        """Write a checkpoint fixture through the shared writer."""
        return self.write_checkpoint(
            scope,
            tokens,
            cache_state_trees=cache_state_trees,
            meta_state_trees=meta_state_trees,
            cache_class_names=cache_class_names,
            reason=reason,
            session_cache_key=session_cache_key,
            now=now,
        )


def _graft_target(cache, expected_class: str):
    """The cache instance to graft a stored layer state into.

    A checkpoint records the live cache classes at capture time. Under a
    quantized live-KV policy the KV layers convert from KVCache to
    QuantizedKVCache once the offset passes the policy threshold, so an
    aligned save on such a session records the quantized class while a
    fresh ``make_cache`` builds the raw one. The fresh cache is empty, so
    converting it is stateless, and the grafted ``meta_state`` restores the
    recorded offset, group size, and bits. Every other class disagreement
    stays a layout mismatch for the caller's gate to refuse.
    """
    live_class = type(cache).__name__
    if (expected_class == "QuantizedKVCache" and live_class == "KVCache"
            and hasattr(cache, "to_quantized")):
        return cache.to_quantized()
    return cache


def _collect_arrays(tree, out):
    import mlx.core as mx

    if isinstance(tree, mx.array):
        out.append(tree)
    elif isinstance(tree, (tuple, list)):
        for child in tree:
            _collect_arrays(child, out)


def _delete_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# --- frontier accounting and capture -----------------------------------------


def _cache_offset(cache) -> int | None:
    """The live position of one cache object, or None when it reports none.

    Every mlx-lm cache exposes ``offset`` (KVCache, RotatingKVCache,
    QuantizedKVCache) and the jang DeepSeek-V4 composite cache exposes it too. The
    frontier writer trusts these offsets over token accounting when deciding
    whether the cache is exactly at a frontier. A cache that reports no offset is
    treated as a disagreement (no write) rather than assumed aligned.
    """
    value = getattr(cache, "offset", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def caches_all_at_offset(caches, expected: int) -> bool:
    """Whether the live caches confirm they are exactly at ``expected``.

    This is the last gate before a write and the source of truth over token
    accounting. Two families of cache appear together in a hybrid layout:

    - Positional caches (``KVCache``, ``RotatingKVCache``, ``QuantizedKVCache``,
      the DeepSeek-V4 composite) expose ``offset``. Every one of them must report
      exactly ``expected``; a single layer at a different offset fails the gate,
      so a write at a non-frontier position is structurally impossible.
    - Recurrent caches (``ArraysCache``) fold every token into whole state and
      expose no offset. They carry no alignment concern (their whole state
      round-trips bit-identically) and are exempt from the equality check.

    At least one positional cache must confirm ``expected``. A layout that reports
    no offset anywhere would leave the frontier resting on token accounting alone,
    which is the failure mode the writer refuses, so that case fails the gate.
    """
    confirmed = False
    for cache in caches:
        offset = _cache_offset(cache)
        if offset is None:
            continue
        if offset != int(expected):
            return False
        confirmed = True
    return confirmed


class FrontierTracker:
    """Absolute token accounting for one generation call.

    Owns the running position of the live cache across a single request: the
    restored prefix length (from the disk or memory hit), the suffix prefill
    progress reported by the mlx-lm prompt-progress callback, and the decode
    tokens appended one at a time. It decides which configured frontiers (multiples
    of ``stride``) this call crosses for the first time and are eligible to write.

    A frontier is eligible only when it is strictly above the restored prefix (a
    frontier at or below the restored prefix is already on disk, or was
    deliberately not written, and this call did not build it) and has no existing
    index entry for the same scope and token-prefix hash (a dedupe read before any
    write). Absolute position is ``restored_prefix + processed``, where
    ``processed`` is the callback's ``prompt_processed_tokens`` during prefill and
    grows by one per appended decode token afterward.

    The tracker proposes; it never writes. The caller cross-checks the proposed
    frontier against the live caches' own offsets and refuses on any disagreement.
    """

    def __init__(
        self,
        *,
        stride: int,
        restored_prefix: int,
        full_tokens: list[int],
        scope: dict,
        already_written: Callable[[dict, list[int]], bool] | None = None,
        write_depth: int | None = None,
    ):
        validate_stride(stride)
        if restored_prefix < 0:
            raise DiskKVError("restored prefix length must be non-negative")
        self.stride = int(stride)
        self.restored_prefix = int(restored_prefix)
        self.full_tokens = list(full_tokens)
        self.scope = dict(scope)
        self._already_written = already_written or (lambda scope, tokens: False)
        # Frontiers above the write-depth cap are never proposed: cumulative
        # snapshot bytes grow with depth while cross-session restores land in
        # the shallow shared-prefix region, so the deep tail is all cost.
        self.write_depth = None if write_depth is None else int(write_depth)
        # Frontiers already handled in this call, so a repeated callback value or a
        # decode step that revisits the same boundary does not propose twice.
        self._handled: set[int] = set()

    def frontier_tokens(self, frontier: int) -> list[int]:
        """The exact token prefix a checkpoint at ``frontier`` describes."""
        return self.full_tokens[:frontier]

    def _eligible(self, frontier: int) -> bool:
        if frontier <= self.restored_prefix:
            return False
        if frontier > len(self.full_tokens):
            return False
        if self.write_depth is not None and frontier > self.write_depth:
            return False
        if frontier in self._handled:
            return False
        if self._already_written(self.scope, self.frontier_tokens(frontier)):
            self._handled.add(frontier)
            return False
        return True

    def crossings_up_to(self, absolute: int) -> list[int]:
        """Frontiers newly reached at absolute position ``absolute``.

        Returns each eligible frontier at or below ``absolute`` that has not been
        proposed yet, in ascending order, and marks them handled. A single prefill
        step can jump several strides at once when the step size is large; a normal
        aligned step reaches exactly one.
        """
        if absolute < self.stride:
            return []
        highest = (absolute // self.stride) * self.stride
        out: list[int] = []
        frontier = self.stride
        while frontier <= highest:
            if self._eligible(frontier):
                self._handled.add(frontier)
                out.append(frontier)
            frontier += self.stride
        return out

    def next_frontier_above(self, absolute: int) -> int | None:
        """The smallest eligible frontier strictly above ``absolute``, or None.

        Used to align the prefill step size so a callback firing lands on a
        frontier exactly. Only frontiers this call can build (above the restored
        prefix within the full token count, excluding entries already written or handled).
        """
        floor = max(self.restored_prefix, absolute)
        frontier = ((floor // self.stride) + 1) * self.stride
        while frontier <= len(self.full_tokens):
            if self._eligible(frontier):
                return frontier
            frontier += self.stride
        return None


def _cache_state_trees(caches) -> tuple[list, list, tuple[str, ...]]:
    """Snapshot the live caches' serializable state at a frontier.

    Returns the per-layer state trees, the per-layer meta_state trees, and the
    cache-class list. The caller has already gated on the caches being exactly at
    the frontier and quiesced, so this only reads the state getters.
    """
    state_trees = [c.state for c in caches]
    meta_state_trees = [c.meta_state for c in caches]
    class_names = tuple(type(c).__name__ for c in caches)
    return state_trees, meta_state_trees, class_names


def plan_prefill_chunks(
    *, start: int, boundaries: Sequence[int], step: int,
) -> list[int]:
    """Chunk sizes that land a chunk end on every boundary exactly.

    ``start`` is the absolute restored-prefix position, ``boundaries`` the
    ascending absolute positions a chunk must end on (checkpoint frontiers),
    ``step`` the full chunk size. Chunks run at ``step``; where a full chunk
    would overshoot a boundary, one shorter chunk closes it. Every chunk is
    therefore ``step`` tokens except the single frontier-landing chunk per
    gap, so the chunk count is bounded by ``span // step + len(boundaries)``.

    Returns the chunk sizes covering ``start`` through the last boundary.
    The caller prefills the remaining tail at its uniform step; past the
    last boundary there is no alignment constraint left.
    """
    step = int(step)
    if step < 1:
        raise DiskKVError("prefill chunk step must be positive")
    plan: list[int] = []
    position = int(start)
    for boundary in boundaries:
        boundary = int(boundary)
        if boundary <= position:
            raise DiskKVError(
                f"prefill chunk boundaries must ascend from the start "
                f"position (boundary {boundary} at position {position})")
        while position < boundary:
            size = min(step, boundary - position)
            plan.append(size)
            position += size
    return plan


class FrontierWriter:
    """Blocking frontier checkpoint writer, driven from the prompt-progress hook.

    Built for one generation call when the disk store is enabled and the request
    crosses at least one unwritten frontier. The callback the mlx-lm prefill loop
    fires runs through :meth:`on_prompt_progress`; in the pinned mlx-lm that
    callback fires after the cache for the step is mutated and evaluated, so the
    live cache offset equals the restored prefix plus the processed count exactly
    at that moment.

    Capture is gated twice. The tracker proposes a frontier from token accounting.
    Then :func:`caches_all_at_offset` confirms every live cache's own reported
    offset equals that frontier. Only when both agree does the writer serialize the
    state and write the checkpoint through the store's shared writer, blocking under
    the caller's serve lock. Any write failure is caught, counted, and swallowed so
    the request continues; the store already quarantines a bad payload, and a
    failed write leaves the index untouched.
    """

    def __init__(
        self,
        store,
        *,
        tracker: FrontierTracker,
        caches,
        session_cache_key: str | None = None,
        now_fn: Callable[[], int] | None = None,
    ):
        self.store = store
        self.tracker = tracker
        self.caches = caches
        self.session_cache_key = session_cache_key
        self._now_fn = now_fn or (lambda: int(time.time()))
        self.written: list[DiskKVEntry] = []
        self.write_seconds: list[float] = []
        self.refused_offset_mismatch = 0
        self.write_failures = 0
        self.disabled = False

    def prefill_chunk_plan(self, default_step: int) -> list[int]:
        """Variable-step chunk plan that lands a callback on every frontier.

        The callback fires after each prefill chunk, so a chunk end must sit
        exactly on a frontier for the writer to capture there. A single
        uniform step can only do that by dividing
        ``gcd(first_gap, stride)``, which collapses to a few tokens whenever
        the restored prefix has arbitrary parity (an in-memory hit on a
        cumulative session); the collapsed step prefills an order of
        magnitude slower and, past roughly 20k context, exhausts Metal
        command-buffer memory. The plan instead runs full ``default_step``
        chunks with one shorter chunk closing each frontier exactly.

        Boundaries are the eligible frontiers at most ``total - 1``: the
        prefill loop leaves the final prompt token to the first decode step,
        so a frontier at the full prompt length is proposed only after that
        step has advanced the cache and the offset gate refuses it (the
        accounting treats it as a permanent hole). An empty plan means no
        frontier is reachable during prefill and the whole suffix runs at
        the caller's uniform step.
        """
        step = max(1, int(default_step))
        prefill_end = len(self.tracker.full_tokens) - 1
        boundaries: list[int] = []
        position = self.tracker.restored_prefix
        while True:
            frontier = self.tracker.next_frontier_above(position)
            if frontier is None or frontier > prefill_end:
                break
            boundaries.append(frontier)
            position = frontier
        return plan_prefill_chunks(
            start=self.tracker.restored_prefix, boundaries=boundaries, step=step)

    def on_prompt_progress(self, processed: int, total: int) -> None:
        """The mlx-lm prompt-progress callback: capture at each crossed frontier.

        ``processed`` is the suffix tokens prefilled so far; the absolute cache
        position is ``restored_prefix + processed``. For each frontier this reaches
        for the first time, confirm the live caches are exactly there and write.
        """
        absolute = self.tracker.restored_prefix + int(processed)
        for frontier in self.tracker.crossings_up_to(absolute):
            self._capture(frontier)

    def on_decode_token(self, absolute: int) -> None:
        """Optional decode-time hook: capture when a decode append hits a frontier.

        ``absolute`` is the full token position after the decode append (prompt
        plus generated). Unused in v1 unless a generation seam offers a per-token
        hook with the cache quiesced; the same offset gate applies.
        """
        for frontier in self.tracker.crossings_up_to(int(absolute)):
            self._capture(frontier)

    def _capture(self, frontier: int) -> None:
        # A hard write fault (full or failing disk) disables the writer for
        # the rest of the request: each later frontier would serialize an
        # even larger snapshot into the same fault, all inside TTFT.
        if self.disabled:
            return
        # The cache classes' own offsets are the source of truth. Refuse the write
        # on any disagreement between the proposed frontier and the live offsets:
        # a non-frontier offset makes a write structurally impossible here.
        if not caches_all_at_offset(self.caches, frontier):
            self.refused_offset_mismatch += 1
            return
        tokens = self.tracker.frontier_tokens(frontier)
        state_trees, meta_state_trees, class_names = _cache_state_trees(self.caches)
        start = time.perf_counter()
        try:
            entry = self.store.write_checkpoint(
                self.tracker.scope,
                tokens,
                cache_state_trees=state_trees,
                meta_state_trees=meta_state_trees,
                cache_class_names=class_names,
                reason="aligned_frontier",
                session_cache_key=self.session_cache_key,
                now=self._now_fn(),
            )
        except Exception as e:  # noqa: BLE001 - a write fault never surfaces in a request
            self.write_failures += 1
            self.disabled = True
            log = getattr(self.store, "_log", _default_log)
            log(f"[disk_kv] write failed token_count={frontier}; disabling "
                f"checkpoint writes for this request: {e!r}")
            return
        # The budget path may skip an oversized payload (entry is None); it logged
        # its own reason and left nothing to record here.
        if entry is None:
            return
        self.write_seconds.append(time.perf_counter() - start)
        self.written.append(entry)


# --- cache-class registry ----------------------------------------------------


def default_cache_registry() -> set[str]:
    """Cache-class names the loader may reconstruct.

    The stock mlx-lm classes plus the jang DeepSeek-V4 composite class. The names
    are strings so the registry can be built without importing mlx at module load;
    the live model's ``make_cache`` constructs the instances.
    """
    return {"KVCache", "QuantizedKVCache", "ArraysCache", "RotatingKVCache",
            "DeepseekV4Cache"}


# --- operator surface (env gating) -------------------------------------------

DISK_KV_MODE_ENV = "MOESPRESSO_DISK_KV"
DISK_KV_ROOT_ENV = "MOESPRESSO_DISK_KV_ROOT"
DISK_KV_STRIDE_ENV = "MOESPRESSO_DISK_KV_STRIDE"
DISK_KV_BYTES_ENV = "MOESPRESSO_DISK_KV_BYTES"
DISK_KV_WRITE_DEPTH_ENV = "MOESPRESSO_DISK_KV_WRITE_DEPTH"

DISK_KV_MODE_FRONTIER = "frontier"
_DISK_KV_MODES_OFF = ("off", "0")
DISK_KV_BYTES_UNLIMITED = "unlimited"

# Serving defaults. The stride trades first-write cost against restore
# granularity: checkpoints are written only during prefill (once per new
# prefix region, deduplicated against the store), and a restore re-prefills
# at most one stride of tail, so 1024 keeps a cold 10k-token prompt to a few
# writes while a later restore re-prefills only a few seconds of tokens.
# The budget bounds each per-package root; eviction is least-recently-used.
# A checkpoint set covering one long agent prompt runs to a few GiB, so the
# default holds a handful of hot prefix regions without claiming a large
# slice of the host disk.
#
# The write-depth cap bounds the cumulative-snapshot cost: each checkpoint
# is a complete snapshot, so written bytes grow quadratically with region
# depth while cross-session restores land in the shallow shared-prefix
# region (an agent client's system prompt and tools). Capping writes at
# 16k tokens keeps the whole shared-prefix win and drops the deep-tail
# traffic that a long conversation would otherwise pay once per region.
DEFAULT_DISK_KV_STRIDE = 1024
DEFAULT_DISK_KV_BUDGET_BYTES = 8 * 1024**3
DEFAULT_DISK_KV_WRITE_DEPTH = 16384


@dataclass(frozen=True)
class DiskKVConfig:
    """Resolved disk KV operator config. ``enabled`` is false when the flag is off.

    ``explicit`` records whether the operator asked for the store by setting
    the mode flag. Serving fails loudly when an explicitly requested store
    cannot open, and degrades to memory-only serving when a default-enabled
    store cannot (a locked root or an unwritable cache directory must not
    take the server down when nobody asked for disk KV).
    """

    enabled: bool
    root: Path | None = None
    stride: int | None = None
    # None means an unbounded store (no eviction). A positive value caps the total
    # payload bytes on disk; a write that would exceed it evicts least-recently-used
    # entries first. Zero and negative are refused at startup by config resolution.
    budget_bytes: int | None = None
    # None means frontiers at any depth are written; a positive value writes
    # only frontiers at or below this token depth.
    write_depth_tokens: int | None = None
    explicit: bool = True


def default_disk_kv_root(package_dir: Path | str,
                         env: dict[str, str] | None = None) -> Path:
    """The serving default disk KV root for one package.

    Lives under the user cache directory (``XDG_CACHE_HOME`` or
    ``~/.cache``) as ``moespresso/disk_kv/<package-fingerprint>``. The
    per-package fingerprint exists for the single-owner root lock: servers
    for different packages run concurrently without contending for one
    root. Correctness never depends on the split; the checkpoint scope hash
    gates every restore regardless of which root holds the entry. Deleting
    the whole ``moespresso`` cache directory is always safe: a missing
    checkpoint means cold serving, never a wrong restore.
    """
    env = os.environ if env is None else env
    cache_home = env.get("XDG_CACHE_HOME")
    if cache_home:
        base = Path(cache_home)
    else:
        home = env.get("HOME")
        base = (Path(home) if home else Path.home()) / ".cache"
    fingerprint = hashlib.sha256(
        str(Path(package_dir).resolve()).encode()).hexdigest()[:12]
    return base / "moespresso" / "disk_kv" / fingerprint


def resolve_disk_kv_config(
    env: dict[str, str] | None = None,
    *,
    package_dir: Path | str | None = None,
) -> DiskKVConfig:
    """Read the disk KV operator surface from the environment.

    With ``package_dir`` (the serving path) the store defaults on: the root
    derives from the package under the user cache directory, the stride
    defaults to ``DEFAULT_DISK_KV_STRIDE``, and the byte budget defaults to
    ``DEFAULT_DISK_KV_BUDGET_BYTES``. ``MOESPRESSO_DISK_KV=off`` (or ``0``)
    is the kill switch, and every explicit environment value overrides its
    default. Without ``package_dir`` no default root exists, so the store
    stays off unless ``MOESPRESSO_DISK_KV=frontier`` names a root and a
    stride explicitly, and an absent budget means unbounded. An unknown
    mode or a half-configured explicit store fails closed at startup.
    """
    env = os.environ if env is None else env
    mode = env.get(DISK_KV_MODE_ENV)
    if mode in _DISK_KV_MODES_OFF:
        return DiskKVConfig(enabled=False)
    if mode and mode != DISK_KV_MODE_FRONTIER:
        raise DiskKVError(
            f"{DISK_KV_MODE_ENV} must be {DISK_KV_MODE_FRONTIER!r} or "
            f"'off' (got {mode!r})")
    explicit = bool(mode)
    if not mode and package_dir is None:
        return DiskKVConfig(enabled=False)

    root = env.get(DISK_KV_ROOT_ENV)
    if not root:
        if package_dir is None:
            raise DiskKVError(
                f"{DISK_KV_MODE_ENV}={DISK_KV_MODE_FRONTIER} requires "
                f"{DISK_KV_ROOT_ENV}")
        resolved_root = default_disk_kv_root(package_dir, env)
    else:
        resolved_root = Path(root)

    stride_raw = env.get(DISK_KV_STRIDE_ENV)
    if stride_raw is None:
        if package_dir is None:
            raise DiskKVError(
                f"{DISK_KV_MODE_ENV}={DISK_KV_MODE_FRONTIER} requires "
                f"{DISK_KV_STRIDE_ENV}")
        stride = DEFAULT_DISK_KV_STRIDE
    else:
        try:
            stride = int(stride_raw)
        except ValueError as e:
            raise DiskKVStrideError(
                f"{DISK_KV_STRIDE_ENV} must be an integer, got {stride_raw!r}"
            ) from e
    validate_stride(stride)
    budget_bytes = _resolve_budget_bytes(
        env.get(DISK_KV_BYTES_ENV),
        default=(DEFAULT_DISK_KV_BUDGET_BYTES if package_dir is not None
                 else None),
    )
    write_depth = _resolve_write_depth(
        env.get(DISK_KV_WRITE_DEPTH_ENV),
        default=(DEFAULT_DISK_KV_WRITE_DEPTH if package_dir is not None
                 else None),
    )
    return DiskKVConfig(
        enabled=True, root=resolved_root, stride=stride,
        budget_bytes=budget_bytes, write_depth_tokens=write_depth,
        explicit=explicit)


def _resolve_budget_bytes(raw: str | None, *,
                          default: int | None = None) -> int | None:
    """Read the disk KV byte budget.

    Absent means ``default`` (the serving path passes the bounded serving
    default; direct callers keep unbounded). The literal ``unlimited``
    disables eviction explicitly. Zero is refused with a message pointing
    the operator at the kill switch: a zero budget cannot hold any
    checkpoint, so the intent is to disable the feature. A negative budget
    is refused as a misconfiguration.
    """
    if raw is None:
        return default
    if raw == DISK_KV_BYTES_UNLIMITED:
        return None
    try:
        value = int(raw)
    except ValueError as e:
        raise DiskKVError(
            f"{DISK_KV_BYTES_ENV} must be an integer or "
            f"{DISK_KV_BYTES_UNLIMITED!r}, got {raw!r}") from e
    if value == 0:
        raise DiskKVError(
            f"{DISK_KV_BYTES_ENV}=0 cannot hold any checkpoint; to turn the "
            f"feature off set {DISK_KV_MODE_ENV}=off instead of a zero budget")
    if value < 0:
        raise DiskKVError(
            f"{DISK_KV_BYTES_ENV} must be positive or "
            f"{DISK_KV_BYTES_UNLIMITED!r}, got {value}")
    return value


def _resolve_write_depth(raw: str | None, *,
                         default: int | None = None) -> int | None:
    """Read the checkpoint write-depth cap in tokens.

    Absent means ``default`` (the serving path passes the bounded serving
    default; direct callers keep unlimited). The literal ``unlimited``
    writes frontiers at any depth. Zero and negative are refused: a store
    that must not write is disabled with the mode flag, not a zero depth.
    """
    if raw is None:
        return default
    if raw == DISK_KV_BYTES_UNLIMITED:
        return None
    try:
        value = int(raw)
    except ValueError as e:
        raise DiskKVError(
            f"{DISK_KV_WRITE_DEPTH_ENV} must be an integer or "
            f"{DISK_KV_BYTES_UNLIMITED!r}, got {raw!r}") from e
    if value <= 0:
        raise DiskKVError(
            f"{DISK_KV_WRITE_DEPTH_ENV} must be positive or "
            f"{DISK_KV_BYTES_UNLIMITED!r}, got {value}")
    return value


def create_manual_checkpoint(
    package_dir: Path | str,
    rendered_prompt_path: Path | str,
    checkpoint_k: int,
    *,
    root: Path | str,
    load_model_fn: Callable | None = None,
    encode_fn: Callable | None = None,
    kv_policy=None,
    now: int = 0,
) -> DiskKVEntry:
    """Prefill exactly K tokens as one chunk and write a checkpoint fixture.

    The only writer that runs a model in M1. It loads the served model, encodes
    the rendered prompt with the serve BOS rule, prefills the first ``K`` tokens as
    a single chunk into a fresh cache, and writes the payload plus the index entry.
    ``K`` must be a positive multiple of 256 (a frontier) and no longer than the
    prompt. The disk scope keys on the same serve 6-tuple the read path uses.
    """
    import mlx.core as mx

    from moespresso.runtime.kv_policy import KVPolicy
    from moespresso.runtime.prefix_cache import (
        cache_model_key,
        encode_rendered_prompt,
    )
    from moespresso.runtime.serve import load_served_model

    validate_stride(checkpoint_k)
    load_model_fn = load_model_fn or load_served_model
    encode_fn = encode_fn or encode_rendered_prompt
    policy = kv_policy if kv_policy is not None else KVPolicy(live_kv_format="raw")

    model, tokenizer, manifest = load_model_fn(Path(package_dir))
    rendered = Path(rendered_prompt_path).read_text(encoding="utf-8")
    tokens = encode_fn(tokenizer, rendered)
    if checkpoint_k > len(tokens):
        raise DiskKVError(
            f"checkpoint K={checkpoint_k} exceeds prompt length {len(tokens)}")

    caches = model.make_cache()
    logits = model(mx.array([tokens[:checkpoint_k]], dtype=mx.int32), cache=caches)
    mx.eval(logits)

    cache_class_names = tuple(type(c).__name__ for c in caches)
    state_trees = [c.state for c in caches]
    meta_state_trees = [c.meta_state for c in caches]

    rendering_id = (manifest.get("tokenizer") or {}).get("rendering_id")
    model_key = cache_model_key(manifest, rendering_id, policy)
    scope = build_cache_scope(model_key, cache_class_names)

    store = DiskCheckpointStore(Path(root))
    return store.write_manual_checkpoint(
        scope,
        tokens[:checkpoint_k],
        cache_state_trees=state_trees,
        meta_state_trees=meta_state_trees,
        cache_class_names=cache_class_names,
        now=now,
    )


def open_disk_store(config: DiskKVConfig) -> DiskCheckpointStore | None:
    """Acquire the root lock and open the store for an enabled config.

    Returns None when the feature is off. The lock is acquired before the model
    load in serve startup, so a second owner is refused before the heavy load. On
    open the stale temp payloads of a crashed previous owner are cleaned up under
    the held lock.

    Every open-time fault surfaces as ``DiskKVError``: an unwritable root,
    a permission fault, or a corrupt index must land in the one exception
    type the serve startup policy dispatches on (explicit stores refuse
    startup, default-enabled stores degrade to memory-only serving).
    """
    if not config.enabled:
        return None
    try:
        lock = DiskKVRootLock(config.root).acquire()
    except DiskKVError:
        raise
    except Exception as e:
        raise DiskKVError(
            f"disk KV root {config.root} cannot open: {e}") from e
    try:
        store = DiskCheckpointStore(
            config.root,
            root_lock=lock,
            stride=config.stride,
            budget_bytes=config.budget_bytes,
            write_depth_tokens=config.write_depth_tokens,
        )
        store.cleanup_stale_temps()
        store.cleanup_orphan_payloads()
    except Exception as e:
        lock.close()
        if isinstance(e, DiskKVError):
            raise
        raise DiskKVError(
            f"disk KV store at {config.root} cannot open: {e}") from e
    return store
