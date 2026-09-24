"""Content identity for the released Qwen4 source checkpoint."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


QWEN4_HF_SNAPSHOT_SOURCE_SCHEMA = "qwen4_hf_snapshot_source_v1"
QWEN4_SHARD_COUNT = 131
QWEN4_MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
_TEACHER_FIELDS = frozenset(
    {"model_id", "revision", "config_sha256", "index_sha256"}
)


class SourceIdentityError(ValueError):
    """A checkpoint path cannot reproduce its declared immutable identity."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, *, field: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SourceIdentityError(
            f"{field} must be a lowercase {length}-character hexadecimal identity"
        )
    return value


def _canonical_sha256(payload: object) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _teacher_identity(value: object) -> dict:
    if not isinstance(value, Mapping) or set(value) != _TEACHER_FIELDS:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise SourceIdentityError(
            "teacher source identity fields do not match the Qwen4 contract: "
            f"{actual}"
        )
    model_id = value.get("model_id")
    if model_id != QWEN4_MODEL_ID:
        raise SourceIdentityError(
            f"teacher source model_id must be {QWEN4_MODEL_ID}"
        )
    return {
        "model_id": model_id,
        "revision": _sha256(value.get("revision"), field="revision", length=40),
        "config_sha256": _sha256(
            value.get("config_sha256"), field="config_sha256"
        ),
        "index_sha256": _sha256(
            value.get("index_sha256"), field="index_sha256"
        ),
    }


def _canonical_qwen4_shards(weight_map: Mapping[str, object]) -> list[str]:
    values = list(weight_map.values())
    if any(not isinstance(value, str) for value in values):
        raise SourceIdentityError("safetensors weight_map contains a non-string shard")
    actual = set(values)
    expected = {
        f"model-{index:05d}-of-{QWEN4_SHARD_COUNT:05d}.safetensors"
        for index in range(1, QWEN4_SHARD_COUNT + 1)
    }
    if actual != expected or len(values) < QWEN4_SHARD_COUNT:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SourceIdentityError(
            f"Qwen4 index must reference exactly {QWEN4_SHARD_COUNT} canonical shards "
            f"(missing={missing[:4]} extra={extra[:4]})"
        )
    return sorted(expected)


def _resolved_hf_blob(snapshot: Path, shard: str) -> tuple[str, int]:
    relative = PurePosixPath(shard)
    if relative.name != shard or len(relative.parts) != 1:
        raise SourceIdentityError(f"indexed shard name is not canonical: {shard!r}")
    path = snapshot / shard
    if not path.is_symlink():
        raise SourceIdentityError(f"indexed shard is not an HF blob symlink: {shard}")
    try:
        resolved = path.resolve(strict=True)
        blob_root = (snapshot.parent.parent / "blobs").resolve(strict=True)
        resolved.relative_to(blob_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SourceIdentityError(
            f"indexed shard does not resolve inside the HF blob store: {shard}"
        ) from exc
    if resolved.parent != blob_root:
        raise SourceIdentityError(f"indexed shard resolves below a nested blob path: {shard}")
    blob = _sha256(resolved.name, field=f"{shard} HF blob")
    if not resolved.is_file():
        raise SourceIdentityError(f"indexed HF blob is not a regular file: {shard}")
    size = int(resolved.stat().st_size)
    if size <= 0:
        raise SourceIdentityError(f"indexed HF blob is empty: {shard}")
    return blob, size


def qwen4_hf_snapshot_source_identity(
    model_dir: str | Path,
    *,
    teacher_source_identity: Mapping[str, object],
    expected_source_identity: Mapping[str, object] | None = None,
) -> dict:
    """Bind the released Qwen4 snapshot metadata and every physical shard.

    Shard content is named by Hugging Face's immutable 64-hex blob identity.
    Recording that name with the exact file size detects retargeted symlinks
    and incomplete reconstruction while binding the cache's declared content
    identity without rereading 360 GB.
    """
    snapshot = Path(model_dir)
    teacher = _teacher_identity(teacher_source_identity)
    if snapshot.name != teacher["revision"] or snapshot.parent.name != "snapshots":
        raise SourceIdentityError(
            "Qwen4 source must be the declared Hugging Face snapshot revision"
        )
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise SourceIdentityError("Qwen4 snapshot is missing config or safetensors index")
    if _sha256_file(config_path) != teacher["config_sha256"]:
        raise SourceIdentityError("Qwen4 config does not match teacher source identity")
    if _sha256_file(index_path) != teacher["index_sha256"]:
        raise SourceIdentityError("Qwen4 index does not match teacher source identity")
    try:
        index = json.loads(index_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceIdentityError("could not read Qwen4 safetensors index") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise SourceIdentityError("Qwen4 safetensors index has no weight_map")
    canonical_shards = _canonical_qwen4_shards(weight_map)
    physical_shards = {
        path.name for path in snapshot.glob("model-*-of-*.safetensors")
    }
    if physical_shards != set(canonical_shards):
        missing = sorted(set(canonical_shards) - physical_shards)
        extra = sorted(physical_shards - set(canonical_shards))
        raise SourceIdentityError(
            "Qwen4 snapshot shard paths do not match the index "
            f"(missing={missing[:4]} extra={extra[:4]})"
        )
    shards = []
    for name in canonical_shards:
        blob, size = _resolved_hf_blob(snapshot, name)
        shards.append(
            {
                "name": name,
                "hf_blob_sha256": blob,
                "size_bytes": size,
            }
        )
    manifest_sha256 = _canonical_sha256(shards)
    identity = {
        "schema": QWEN4_HF_SNAPSHOT_SOURCE_SCHEMA,
        "teacher_source_identity": teacher,
        "shards": shards,
        "shard_manifest_sha256": manifest_sha256,
    }
    identity["snapshot_identity_sha256"] = _canonical_sha256(identity)
    if expected_source_identity is not None and identity != dict(
        expected_source_identity
    ):
        raise SourceIdentityError("Qwen4 snapshot source identity drifted")
    return identity
