"""DFlash draft-model sidecar builder for DeepSeek-V4-Flash.

Reads the RedHatAI/DeepSeek-V4-Flash-speculator.dflash checkpoint (a single
safetensors shard plus the speculators config.json) and writes a standalone
sidecar package: one safetensors shard holding final module-path tensor
names plus a content-hashed manifest. The checkpoint stores final names
already, so the naming map is the identity apart from dropping the
verifier embedding copy; `sanitize_dflash_weights` is the naming
authority and the loader never parses names.

Per-tensor treatment, recorded tensor by tensor in the manifest:

- Attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`) and
  the pruned `lm_head`: affine-quantized at 8 bits, group size 32.
- The `fc` feature projection: affine-quantized at 8 bits, group size 32,
  by default. Quantizing `fc` measured no acceptance cost on the real
  checkpoint (fixed-schedule tau 2.31 both ways, per-position matched
  curves equal within noise) and shrinks the per-ingest read of the
  largest drafter tensor from about 671 MB to about 378 MB, cutting the
  per-round ingest cost by roughly a third. `--fc-format bf16` builds a
  source-dtype passthrough variant; the chosen format is recorded per
  tensor and under `build_options` in the manifest.
- Every norm weight: passthrough at the source dtype. The norms are
  precision-sensitive and are never quantized.
- `d2t` (int64 offset table) and `t2d` (bool membership table): raw
  passthrough. `t2d` is training-time documentation, unused at inference.
- `embed_tokens` is a verifier copy of the target embedding and is
  dropped; the runtime injects the target model's own embedding. Because
  the target package is not an input of this builder, the equality check
  moves to load time: the builder stores a few sample embedding rows
  under the reserved `embed_sample.*` names, and the loader compares them
  against the target embedding and logs the delta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from moespresso.core.artifact import (
    artifact_producer,
    compute_artifact_id,
    write_artifact,
)
from moespresso.inventory.safetensors_header import (
    TensorHeader,
    read_headers_with_offsets,
)
from moespresso.runtime.deepseek_v4.dflash_model import (
    DFlashArgs,
    DFlashDraftModel,
    sanitize_dflash_weights,
)

SIDECAR_KIND = "deepseek_v4_dflash_sidecar"
SIDECAR_SCHEMA_MAJOR = 1
SIDECAR_SCHEMA_MINOR = 1
SIDECAR_MANIFEST_NAME = "dflash_sidecar.json"

FORMAT_AFFINE8 = "affine8"
FORMAT_PASSTHROUGH = "passthrough"
KNOWN_FORMATS = frozenset({FORMAT_AFFINE8, FORMAT_PASSTHROUGH})

# Builder choices for the fc feature projection: affine 8-bit
# quantization (the default) or source-dtype passthrough.
FC_FORMAT_BF16 = "bf16"
FC_FORMAT_AFFINE8 = "affine8"
FC_FORMATS = (FC_FORMAT_AFFINE8, FC_FORMAT_BF16)
FC_FORMAT_DEFAULT = FC_FORMAT_AFFINE8

AFFINE8_PARAMS = {"group_size": 32, "bits": 8, "mode": "affine"}

PRODUCER = artifact_producer("moespresso.package.deepseek_v4.dflash_sidecar")
SUBJECT = {"family": "deepseek_v4_flash_dflash", "role": "draft_sidecar"}

HUB_CACHE = Path("~/.cache/huggingface/hub").expanduser()
SHARD_NAME = "model-dflash-00001-of-00001.safetensors"

EMBED_SOURCE_NAME = "embed_tokens.weight"
# Reserved shard names for the load-time embedding check; not parameters.
EMBED_SAMPLE_ROWS = "embed_sample.rows"
EMBED_SAMPLE_TOKEN_IDS = "embed_sample.token_ids"
EMBED_SAMPLE_NAMES = frozenset({EMBED_SAMPLE_ROWS, EMBED_SAMPLE_TOKEN_IDS})

# Storage dtype token -> numpy dtype used to view the raw bytes. BF16 is
# read as uint16 and widened to float32 (an exact conversion).
_STORAGE_NP = {
    "F16": np.float16,
    "F32": np.float32,
    "BF16": np.uint16,
    "I64": np.int64,
    "BOOL": np.bool_,
}
_FLOAT_TOKENS = ("F16", "F32", "BF16")
_MX_FLOAT = {"F16": mx.float16, "F32": mx.float32, "BF16": mx.bfloat16}

_PROJ_RE = re.compile(
    r"layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$")
_NORM_RE = re.compile(
    r"layers\.\d+\.(input_layernorm|post_attention_layernorm"
    r"|self_attn\.(q_norm|k_norm))\.weight$")
_KEEP_NAMES = frozenset({"hidden_norm.weight", "norm.weight"})


class DFlashSidecarError(ValueError):
    """A sidecar build or load contract violation."""


def _mx_dtype_name(dtype) -> str:
    return str(dtype).split(".")[-1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_storage(source_dir: Path, header: TensorHeader) -> np.ndarray:
    """Read one source tensor preserving its storage codes."""
    np_dtype = _STORAGE_NP.get(header.dtype)
    if np_dtype is None:
        raise DFlashSidecarError(
            f"unsupported source storage dtype {header.dtype!r} for {header.name}")
    nbytes = header.end - header.begin
    with open(source_dir / header.shard, "rb") as f:
        f.seek(header.header_size + header.begin)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise DFlashSidecarError(
            f"short read for {header.name}: {len(raw)} of {nbytes} bytes")
    arr = np.frombuffer(raw, dtype=np_dtype)
    expected = int(np.prod(header.shape)) if header.shape else 1
    if arr.size != expected:
        raise DFlashSidecarError(
            f"{header.name}: {arr.size} elements on disk, header says {expected}")
    return arr.reshape(header.shape)


def _widen_float32(codes: np.ndarray, dtype_token: str, name: str) -> np.ndarray:
    if dtype_token == "BF16":
        return (codes.astype(np.uint32) << 16).view(np.float32)
    if dtype_token in ("F16", "F32"):
        return codes.astype(np.float32)
    raise DFlashSidecarError(f"{name}: dtype {dtype_token} is not a float dtype")


def _load_float_mx(source_dir: Path, header: TensorHeader) -> mx.array:
    """Read one float source tensor as an mx array at its source dtype."""
    codes = _read_storage(source_dir, header)
    widened = _widen_float32(codes, header.dtype, header.name)
    return mx.array(widened).astype(_MX_FLOAT[header.dtype])


def _read_float_rows(
    source_dir: Path, header: TensorHeader, row_ids: list[int],
) -> mx.array:
    """Read selected rows of a 2-D float tensor without loading the whole
    tensor (the verifier embedding is large)."""
    np_dtype = _STORAGE_NP.get(header.dtype)
    if header.dtype not in _FLOAT_TOKENS or np_dtype is None:
        raise DFlashSidecarError(
            f"{header.name}: unsupported embedding dtype {header.dtype!r}")
    if len(header.shape) != 2:
        raise DFlashSidecarError(
            f"{header.name}: expected a 2-D embedding, got shape {header.shape}")
    rows, cols = header.shape
    itemsize = np.dtype(np_dtype).itemsize
    row_bytes = cols * itemsize
    out = np.empty((len(row_ids), cols), dtype=np.float32)
    with open(source_dir / header.shard, "rb") as f:
        for i, row in enumerate(row_ids):
            if not (0 <= row < rows):
                raise DFlashSidecarError(
                    f"{header.name}: sample row {row} outside {rows} rows")
            f.seek(header.header_size + header.begin + row * row_bytes)
            raw = f.read(row_bytes)
            if len(raw) != row_bytes:
                raise DFlashSidecarError(
                    f"short read for {header.name} row {row}")
            codes = np.frombuffer(raw, dtype=np_dtype)
            out[i] = _widen_float32(codes, header.dtype, header.name)
    return mx.array(out).astype(_MX_FLOAT[header.dtype])


def _load_args(config: dict) -> DFlashArgs:
    try:
        return DFlashArgs.from_config(config)
    except ValueError as e:
        raise DFlashSidecarError(f"invalid speculator config: {e}") from e


def _collect_source_headers(source_dir: Path) -> dict[str, TensorHeader]:
    """Map raw tensor names to shard headers across every shard on disk."""
    shard_paths = sorted(source_dir.glob("*.safetensors"))
    if not shard_paths:
        raise DFlashSidecarError(f"no safetensors shards found under {source_dir}")
    headers: dict[str, TensorHeader] = {}
    for shard_path in shard_paths:
        for header in read_headers_with_offsets(shard_path):
            if header.name in headers:
                raise DFlashSidecarError(f"duplicate source tensor {header.name}")
            headers[header.name] = header
    return headers


def _treatment_for(name: str, fc_format: str = FC_FORMAT_DEFAULT) -> str:
    """Conversion treatment for one final tensor name, failing closed."""
    if name == "fc.weight":
        return "affine8" if fc_format == FC_FORMAT_AFFINE8 else "keep"
    if _PROJ_RE.fullmatch(name) or name == "lm_head.weight":
        return "affine8"
    if name in _KEEP_NAMES or _NORM_RE.fullmatch(name):
        return "keep"
    if name in ("d2t", "t2d"):
        return "raw"
    raise DFlashSidecarError(f"no conversion rule for source tensor {name}")


def _expected_params(args: DFlashArgs, module_formats: dict[str, dict]) -> set[str]:
    """Parameter names of the draft model under the given quant formats.

    The model is built lazily (arrays are never evaluated), so enumerating
    the tree is cheap.
    """
    model = DFlashDraftModel(args)
    if module_formats:
        nn.quantize(
            model,
            class_predicate=lambda path, mod: dict(module_formats[path])
            if path in module_formats else False,
        )
    return {name for name, _ in tree_flatten(model.parameters())}


def _check_key_sets(expected: set[str], produced: set[str], what: str) -> None:
    missing = sorted(expected - produced)
    unexpected = sorted(produced - expected)
    if missing or unexpected:
        raise DFlashSidecarError(
            f"{what}: tensor set does not match the draft model; "
            f"missing={missing[:8]} unexpected={unexpected[:8]}")


def _validate_tables(
    args: DFlashArgs, headers: dict[str, TensorHeader], tensors: dict[str, mx.array],
) -> None:
    """Shape and range checks on the vocabulary tables."""
    d2t = tensors.get("d2t")
    t2d = tensors.get("t2d")
    if d2t is None or headers["d2t"].dtype != "I64":
        raise DFlashSidecarError("d2t must be an int64 offset table")
    if t2d is None or headers["t2d"].dtype != "BOOL":
        raise DFlashSidecarError("t2d must be a bool membership table")
    if tuple(d2t.shape) != (args.draft_vocab_size,):
        raise DFlashSidecarError(
            f"d2t shape {tuple(d2t.shape)} != ({args.draft_vocab_size},)")
    if tuple(t2d.shape) != (args.target_vocab_size,):
        raise DFlashSidecarError(
            f"t2d shape {tuple(t2d.shape)} != ({args.target_vocab_size},)")
    mapped = np.arange(args.draft_vocab_size, dtype=np.int64) + np.array(d2t)
    if mapped.min() < 0 or mapped.max() >= args.target_vocab_size:
        raise DFlashSidecarError(
            f"d2t maps draft indices outside the target vocabulary "
            f"(range {int(mapped.min())}..{int(mapped.max())}, "
            f"target vocab {args.target_vocab_size})")


def _embed_sample_ids(args: DFlashArgs) -> list[int]:
    """Deterministic sample rows for the load-time embedding check."""
    vocab = args.target_vocab_size
    return sorted({0, args.mask_token_id, vocab // 2, vocab - 1})


def build_dflash_sidecar(
    source_dir: Path, output_dir: Path, fc_format: str = FC_FORMAT_DEFAULT,
) -> dict:
    """Build the sidecar package. Returns the written manifest payload.

    `fc_format` selects the fc feature-projection treatment: affine 8-bit
    quantization (the default) or source-dtype passthrough.
    """
    if fc_format not in FC_FORMATS:
        raise DFlashSidecarError(
            f"unknown fc format {fc_format!r}; known formats: {FC_FORMATS}")
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    config_path = source_dir / "config.json"
    if not config_path.exists():
        raise DFlashSidecarError(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    args = _load_args(config)

    headers = _collect_source_headers(source_dir)
    embed_header = headers.pop(EMBED_SOURCE_NAME, None)
    if embed_header is None:
        raise DFlashSidecarError(
            f"source has no {EMBED_SOURCE_NAME}; the checkpoint carries a "
            "verifier embedding copy and its absence means an incomplete "
            "download")
    if list(embed_header.shape) != [args.target_vocab_size, args.hidden_size]:
        raise DFlashSidecarError(
            f"{EMBED_SOURCE_NAME} shape {embed_header.shape} != "
            f"[{args.target_vocab_size}, {args.hidden_size}]")

    # Resolve names once through the naming authority. The map is the
    # identity for every kept tensor, asserted here.
    placeholder = mx.zeros((1,))
    try:
        finals = sanitize_dflash_weights(
            {name: placeholder for name in headers}, args)
    except ValueError as e:
        raise DFlashSidecarError(str(e)) from e
    if set(finals) != set(headers):
        raise DFlashSidecarError(
            "sanitize changed tensor names; the DFlash checkpoint stores "
            "final module-path names")

    plan = {name: _treatment_for(name, fc_format) for name in sorted(headers)}
    _check_key_sets(_expected_params(args, {}), set(plan), "source resolution")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise DFlashSidecarError(f"output directory {output_dir} is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_rows: dict[str, dict] = {}
    module_formats: dict[str, dict] = {}
    out_tensors: dict[str, mx.array] = {}

    def _take(name: str, arr: mx.array, row: dict) -> None:
        out_tensors[name] = arr
        row = dict(row)
        row.update({
            "dtype": _mx_dtype_name(arr.dtype),
            "shape": [int(d) for d in arr.shape],
            "file": SHARD_NAME,
        })
        tensor_rows[name] = row

    for name, treatment in plan.items():
        header = headers[name]
        if treatment == "affine8":
            if header.dtype not in _FLOAT_TOKENS:
                raise DFlashSidecarError(
                    f"{name}: unsupported dense source dtype {header.dtype}")
            w = _load_float_mx(source_dir, header)
            qw, qs, qb = mx.quantize(w, **AFFINE8_PARAMS)
            mx.eval(qw, qs, qb)
            base = name[: -len(".weight")]
            row = {
                "format": FORMAT_AFFINE8,
                "quant": dict(AFFINE8_PARAMS),
                "source_dtype": header.dtype,
            }
            _take(f"{base}.weight", qw, row)
            _take(f"{base}.scales", qs, row)
            _take(f"{base}.biases", qb, row)
            module_formats[base] = dict(AFFINE8_PARAMS)
        elif treatment == "keep":
            arr = _load_float_mx(source_dir, header)
            _take(name, arr, {
                "format": FORMAT_PASSTHROUGH, "source_dtype": header.dtype,
            })
        elif treatment == "raw":
            arr = mx.array(_read_storage(source_dir, header))
            _take(name, arr, {
                "format": FORMAT_PASSTHROUGH, "source_dtype": header.dtype,
            })
        else:
            raise DFlashSidecarError(f"unknown treatment {treatment!r} for {name}")

    _validate_tables(args, headers, out_tensors)

    # Sample rows of the dropped verifier embedding, for the load-time
    # comparison against the target package embedding.
    sample_ids = _embed_sample_ids(args)
    sample_rows = _read_float_rows(source_dir, embed_header, sample_ids)
    note = (
        "verifier embedding sample rows for the load-time target-embedding "
        "check; not draft parameters")
    _take(EMBED_SAMPLE_ROWS, sample_rows, {
        "format": FORMAT_PASSTHROUGH, "source_dtype": embed_header.dtype,
        "note": note,
    })
    _take(EMBED_SAMPLE_TOKEN_IDS, mx.array(sample_ids, dtype=mx.int64), {
        "format": FORMAT_PASSTHROUGH, "source_dtype": "I64", "note": note,
    })

    shard_path = output_dir / SHARD_NAME
    mx.save_safetensors(str(shard_path), out_tensors, metadata={"format": "mlx"})
    file_sha256 = {SHARD_NAME: _sha256_file(shard_path)}

    _check_key_sets(
        _expected_params(args, module_formats),
        set(out_tensors) - EMBED_SAMPLE_NAMES,
        "sidecar output",
    )

    snapshot = source_dir.resolve()
    payload = {
        "artifact_kind": SIDECAR_KIND,
        "schema_version": {
            "major": SIDECAR_SCHEMA_MAJOR, "minor": SIDECAR_SCHEMA_MINOR,
        },
        "producer": dict(PRODUCER),
        "subject": dict(SUBJECT),
        "status": "valid",
        "formats": {FORMAT_AFFINE8: dict(AFFINE8_PARAMS)},
        "build_options": {"fc_format": fc_format},
        "provenance": {
            "source_snapshot": str(snapshot),
            "source_commit": snapshot.name,
            "file_sha256": file_sha256,
        },
        "dflash": {
            "block_size": args.block_size,
            "speculative_tokens": args.speculative_tokens,
            "mask_token_id": args.mask_token_id,
            "draft_vocab_size": args.draft_vocab_size,
            "target_vocab_size": args.target_vocab_size,
            "sliding_window": args.sliding_window,
            "aux_hidden_state_layer_ids": list(args.aux_layer_ids),
            "tap_layer_ids": list(args.tap_layer_ids),
        },
        "source_config": config,
        "tensors": tensor_rows,
    }
    payload["artifact_id"] = compute_artifact_id(payload)
    write_artifact(
        output_dir / SIDECAR_MANIFEST_NAME,
        payload,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return payload


def resolve_output_dir(value: str) -> Path:
    """A bare name goes under the HF hub cache; a path is used as given."""
    if "/" in value or value.startswith("~"):
        return Path(value).expanduser()
    return HUB_CACHE / value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the DFlash draft-model sidecar from the "
                    "DeepSeek-V4-Flash speculator HF snapshot.")
    parser.add_argument(
        "--source", required=True,
        help="speculator snapshot directory (config.json plus the "
             "safetensors shard)")
    parser.add_argument(
        "--output", required=True,
        help="sidecar directory: a path, or a bare name placed under the HF "
             "hub cache (moespresso-ds4-dflash-sidecar is the conventional "
             "name)")
    parser.add_argument(
        "--fc-format", choices=FC_FORMATS, default=FC_FORMAT_DEFAULT,
        help="fc feature-projection treatment: affine 8-bit quantization "
             "(affine8, the default; measured acceptance-neutral and "
             "cheaper to ingest) or source-dtype passthrough (bf16)")
    args = parser.parse_args(argv)

    try:
        payload = build_dflash_sidecar(
            Path(args.source), resolve_output_dir(args.output),
            fc_format=args.fc_format)
    except DFlashSidecarError as e:
        parser.exit(2, f"error: {e}\n")
    counts: dict[str, int] = {}
    for row in payload["tensors"].values():
        counts[row["format"]] = counts.get(row["format"], 0) + 1
    summary = ", ".join(f"{fmt}: {n}" for fmt, n in sorted(counts.items()))
    print(f"wrote {len(payload['tensors'])} tensors ({summary})")
    print(f"manifest {payload['artifact_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
