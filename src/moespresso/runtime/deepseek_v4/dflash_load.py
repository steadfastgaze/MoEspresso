"""Loader for the DFlash draft-model sidecar package.

Reads the sidecar directory written by
`moespresso.package.deepseek_v4.dflash_sidecar`: validates the
content-hashed manifest (kind, schema version, per-file sha256), rebuilds
the drafter arguments from the stored speculator config, quantizes the
draft module tree to match the manifest's per-tensor formats, and performs
a strict `load_weights`. Tensor names were resolved once at build time;
nothing here parses names beyond locating each tensor's owning module.
Every mismatch is an error; there is no silent fallback.

The sidecar carries sample rows of the dropped verifier embedding. After
loading, the target embedding is evaluated at the sampled token ids and
the maximum absolute delta is logged. Quantization-level differences are
expected when the target embedding is quantized; a large delta means the
sidecar is paired with a different target, which costs acceptance only
and never correctness (verification compares every draft token against
the target's own logits), so the check logs and does not fail.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from moespresso.core.artifact import compute_artifact_id
from moespresso.package.deepseek_v4.dflash_sidecar import (
    EMBED_SAMPLE_NAMES,
    EMBED_SAMPLE_ROWS,
    EMBED_SAMPLE_TOKEN_IDS,
    FORMAT_AFFINE8,
    KNOWN_FORMATS,
    SIDECAR_KIND,
    SIDECAR_MANIFEST_NAME,
    SIDECAR_SCHEMA_MAJOR,
    DFlashSidecarError,
)
from moespresso.runtime.deepseek_v4.dflash_model import DFlashArgs, DFlashDraftModel
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
        raise DFlashSidecarError(f"missing sidecar manifest {manifest_path}")
    payload = json.loads(manifest_path.read_text())

    kind = payload.get("artifact_kind")
    if kind != SIDECAR_KIND:
        raise DFlashSidecarError(f"unknown sidecar artifact kind {kind!r}")
    major = (payload.get("schema_version") or {}).get("major")
    if major != SIDECAR_SCHEMA_MAJOR:
        raise DFlashSidecarError(
            f"unsupported sidecar schema major {major!r} "
            f"(this build: {SIDECAR_SCHEMA_MAJOR})")
    stored_id = payload.get("artifact_id")
    computed_id = compute_artifact_id(payload)
    if stored_id != computed_id:
        raise DFlashSidecarError(
            f"sidecar manifest hash mismatch: stored {stored_id} != "
            f"computed {computed_id}")

    tensors = payload.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise DFlashSidecarError("sidecar manifest has no tensor table")
    file_sha256 = (payload.get("provenance") or {}).get("file_sha256")
    if not isinstance(file_sha256, dict) or not file_sha256:
        raise DFlashSidecarError("sidecar manifest has no file hashes")
    for name, row in tensors.items():
        fmt = row.get("format")
        if fmt not in KNOWN_FORMATS:
            raise DFlashSidecarError(f"{name}: unknown tensor format {fmt!r}")
        if row.get("file") not in file_sha256:
            raise DFlashSidecarError(f"{name}: file {row.get('file')!r} is not hashed")

    if verify_files:
        for file_name, expected in sorted(file_sha256.items()):
            shard_path = Path(sidecar_dir) / file_name
            if not shard_path.exists():
                raise DFlashSidecarError(f"missing sidecar shard {shard_path}")
            actual = _sha256_file(shard_path)
            if actual != expected:
                raise DFlashSidecarError(
                    f"sidecar shard {file_name} hash mismatch: "
                    f"{actual} != recorded {expected}")
    return payload


def _module_formats(manifest: dict) -> dict[str, dict]:
    """Map each quantized module path to its quantization parameters."""
    formats = manifest.get("formats") or {}
    module_params: dict[str, dict] = {}
    for name, row in manifest["tensors"].items():
        fmt = row["format"]
        if fmt != FORMAT_AFFINE8:
            continue
        params = formats.get(fmt) or row.get("quant")
        if not params:
            raise DFlashSidecarError(
                f"{name}: quantized format {fmt} has no parameters")
        module = name.rsplit(".", 1)[0]
        known = module_params.setdefault(module, dict(params))
        if known != dict(params):
            raise DFlashSidecarError(
                f"{module}: conflicting quantization parameters across tensors")
    return module_params


def _validate_dflash_block(manifest: dict, args: DFlashArgs) -> None:
    """The manifest's dflash block must restate the parsed config exactly."""
    block = manifest.get("dflash") or {}
    expected = {
        "block_size": args.block_size,
        "speculative_tokens": args.speculative_tokens,
        "mask_token_id": args.mask_token_id,
        "draft_vocab_size": args.draft_vocab_size,
        "target_vocab_size": args.target_vocab_size,
        "sliding_window": args.sliding_window,
        "aux_hidden_state_layer_ids": list(args.aux_layer_ids),
        "tap_layer_ids": list(args.tap_layer_ids),
    }
    for key, value in expected.items():
        if block.get(key) != value:
            raise DFlashSidecarError(
                f"manifest dflash.{key} is {block.get(key)!r}; the stored "
                f"source config implies {value!r}")


def _embed_sample_check(model: DFlashDraftModel, ids: mx.array, rows: mx.array) -> None:
    """Compare stored verifier-embedding rows against the target embedding.

    Logged only: a quantized target embedding differs by quantization
    error, and a wrong pairing costs acceptance, never correctness.
    """
    got = model.embed(ids[None])[0].astype(mx.float32)
    delta = mx.abs(got - rows.astype(mx.float32)).max()
    mx.eval(delta)
    print(
        f"[dflash] verifier embed sample delta max={float(delta):.3e} over "
        f"{int(ids.shape[0])} rows (quantization-level differences are "
        "expected; a large delta means the sidecar is paired with a "
        "different target)",
        flush=True,
    )


def load_dflash_sidecar(sidecar_dir, embed) -> tuple[DFlashDraftModel, DFlashArgs]:
    """Load the draft model from a sidecar directory.

    `embed` is the target model's frozen embedding; the sidecar does not
    carry one. The drafter owns its pruned language-model head.
    """
    sidecar_dir = Path(sidecar_dir)
    manifest = read_sidecar_manifest(sidecar_dir)

    try:
        args = DFlashArgs.from_config(manifest["source_config"])
    except (KeyError, ValueError) as e:
        raise DFlashSidecarError(f"invalid stored source config: {e}") from e
    _validate_dflash_block(manifest, args)
    model = DFlashDraftModel(args, embed=embed)

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
        part = mx.load(str(sidecar_dir / file_name))
        overlap = set(part) & set(weights)
        if overlap:
            raise DFlashSidecarError(
                f"tensors repeated across sidecar shards: {sorted(overlap)[:8]}")
        weights.update(part)

    missing = sorted(set(rows) - set(weights))
    unexpected = sorted(set(weights) - set(rows))
    if missing or unexpected:
        raise DFlashSidecarError(
            f"sidecar shards do not match the manifest tensor table; "
            f"missing={missing[:8]} unexpected={unexpected[:8]}")
    for name, arr in weights.items():
        row = rows[name]
        if list(arr.shape) != list(row["shape"]):
            raise DFlashSidecarError(
                f"{name}: shard shape {list(arr.shape)} != manifest {row['shape']}")
        if _mx_dtype_name(arr.dtype) != row["dtype"]:
            raise DFlashSidecarError(
                f"{name}: shard dtype {_mx_dtype_name(arr.dtype)} != "
                f"manifest {row['dtype']}")

    if EMBED_SAMPLE_NAMES - set(weights):
        raise DFlashSidecarError(
            "sidecar carries no verifier embedding sample rows")
    sample_rows = weights.pop(EMBED_SAMPLE_ROWS)
    sample_ids = weights.pop(EMBED_SAMPLE_TOKEN_IDS)

    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    if embed is not None:
        _embed_sample_check(model, sample_ids, sample_rows)
    return model, args
