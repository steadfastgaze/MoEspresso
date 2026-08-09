"""Loader for the MTP draft-model sidecar package.

Reads the sidecar directory written by
`moespresso.package.deepseek_v4.mtp_sidecar`: validates the content-hashed
manifest (kind, schema version, per-file sha256), rebuilds the model arguments
from the stored source config, quantizes the draft module tree to match the
manifest's per-tensor formats, and performs a strict `load_weights`. Tensor
names were resolved once at build time; nothing here parses names beyond
locating each tensor's owning module. Every mismatch is an error; there is no
silent fallback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from jang_tools.dsv4.mlx_model import ModelArgs

from moespresso.core.artifact import compute_artifact_id
from moespresso.core.paths import UnsafeArtifactPathError, resolve_artifact_file
from moespresso.package.deepseek_v4.mtp_sidecar import (
    KNOWN_FORMATS,
    FORMAT_AFFINE8,
    FORMAT_MXFP4,
    SIDECAR_KIND,
    SIDECAR_MANIFEST_NAME,
    SIDECAR_SCHEMA_MAJOR,
    MTPSidecarError,
)
from moespresso.runtime.deepseek_v4.mtp_model import MTPArgs, MTPDraftModel
from moespresso.runtime.deepseek_v4.spec_decode import strip_draft_skeleton


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mx_dtype_name(dtype) -> str:
    return str(dtype).split(".")[-1]


def read_sidecar_manifest(sidecar_dir: Path, verify_files: bool = True) -> dict:
    """Read and validate the sidecar manifest, failing closed on any mismatch."""
    manifest_path = Path(sidecar_dir) / SIDECAR_MANIFEST_NAME
    if not manifest_path.exists():
        raise MTPSidecarError(f"missing sidecar manifest {manifest_path}")
    payload = json.loads(manifest_path.read_text())

    kind = payload.get("artifact_kind")
    if kind != SIDECAR_KIND:
        raise MTPSidecarError(f"unknown sidecar artifact kind {kind!r}")
    major = (payload.get("schema_version") or {}).get("major")
    if major != SIDECAR_SCHEMA_MAJOR:
        raise MTPSidecarError(
            f"unsupported sidecar schema major {major!r} "
            f"(this build: {SIDECAR_SCHEMA_MAJOR})")
    stored_id = payload.get("artifact_id")
    computed_id = compute_artifact_id(payload)
    if stored_id != computed_id:
        raise MTPSidecarError(
            f"sidecar manifest hash mismatch: stored {stored_id} != computed {computed_id}")

    tensors = payload.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise MTPSidecarError("sidecar manifest has no tensor table")
    file_sha256 = (payload.get("provenance") or {}).get("file_sha256")
    if not isinstance(file_sha256, dict) or not file_sha256:
        raise MTPSidecarError("sidecar manifest has no file hashes")
    try:
        shard_paths = {
            file_name: resolve_artifact_file(sidecar_dir, file_name)
            for file_name in file_sha256
        }
    except UnsafeArtifactPathError as exc:
        raise MTPSidecarError(str(exc)) from exc
    for name, row in tensors.items():
        fmt = row.get("format")
        if fmt not in KNOWN_FORMATS:
            raise MTPSidecarError(f"{name}: unknown tensor format {fmt!r}")
        if row.get("file") not in file_sha256:
            raise MTPSidecarError(f"{name}: file {row.get('file')!r} is not hashed")

    if verify_files:
        for file_name, expected in sorted(file_sha256.items()):
            shard_path = shard_paths[file_name]
            if not shard_path.exists():
                raise MTPSidecarError(f"missing sidecar shard {shard_path}")
            actual = _sha256_file(shard_path)
            if actual != expected:
                raise MTPSidecarError(
                    f"sidecar shard {file_name} hash mismatch: "
                    f"{actual} != recorded {expected}")
    return payload


def _module_formats(manifest: dict) -> dict[str, dict]:
    """Map each quantized module path to its quantization parameters."""
    formats = manifest.get("formats") or {}
    module_params: dict[str, dict] = {}
    for name, row in manifest["tensors"].items():
        fmt = row["format"]
        if fmt not in (FORMAT_MXFP4, FORMAT_AFFINE8):
            continue
        params = formats.get(fmt) or row.get("quant")
        if not params:
            raise MTPSidecarError(f"{name}: quantized format {fmt} has no parameters")
        module = name.rsplit(".", 1)[0]
        known = module_params.setdefault(module, dict(params))
        if known != dict(params):
            raise MTPSidecarError(
                f"{module}: conflicting quantization parameters across tensors")
    return module_params


def load_mtp_sidecar(
    sidecar_dir, embed, lm_head,
) -> tuple[MTPDraftModel, MTPArgs]:
    """Load the draft model from a sidecar directory.

    `embed` and `lm_head` are the target model's frozen modules; the sidecar
    does not carry them.
    """
    sidecar_dir = Path(sidecar_dir)
    manifest = read_sidecar_manifest(sidecar_dir)

    margs = ModelArgs.from_dict(manifest["source_config"])
    m = manifest["mtp"]
    if int(m.get("n_mtp_layers", 0)) != 1:
        raise MTPSidecarError(
            f"the MTP drafter chains a single module; manifest declares "
            f"{m.get('n_mtp_layers')!r}")
    args = MTPArgs(model_args=margs, block_size=int(m["block_size"]))
    model = MTPDraftModel(args, embed=embed, lm_head=lm_head)

    module_params = _module_formats(manifest)
    if module_params:
        nn.quantize(
            model,
            class_predicate=lambda path, mod: dict(module_params[path])
            if path in module_params else False,
        )
    # Drop the constructed skeleton before any shard is read: the module
    # tree still references lazy random-init arrays (and the quantization
    # graphs built over them), and an evaluation reaching the tree before
    # the strict load would allocate the full float32 skeleton.
    strip_draft_skeleton(model)

    rows = manifest["tensors"]
    weights: dict[str, mx.array] = {}
    for file_name in sorted(manifest["provenance"]["file_sha256"]):
        try:
            shard_path = resolve_artifact_file(sidecar_dir, file_name)
        except UnsafeArtifactPathError as exc:
            raise MTPSidecarError(str(exc)) from exc
        part = mx.load(str(shard_path))
        overlap = set(part) & set(weights)
        if overlap:
            raise MTPSidecarError(
                f"tensors repeated across sidecar shards: {sorted(overlap)[:8]}")
        weights.update(part)

    missing = sorted(set(rows) - set(weights))
    unexpected = sorted(set(weights) - set(rows))
    if missing or unexpected:
        raise MTPSidecarError(
            f"sidecar shards do not match the manifest tensor table; "
            f"missing={missing[:8]} unexpected={unexpected[:8]}")
    for name, arr in weights.items():
        row = rows[name]
        if list(arr.shape) != list(row["shape"]):
            raise MTPSidecarError(
                f"{name}: shard shape {list(arr.shape)} != manifest {row['shape']}")
        if _mx_dtype_name(arr.dtype) != row["dtype"]:
            raise MTPSidecarError(
                f"{name}: shard dtype {_mx_dtype_name(arr.dtype)} != "
                f"manifest {row['dtype']}")

    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model, args
