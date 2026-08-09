"""DSpark draft-model sidecar builder for DeepSeek-V4-Flash.

Reads the `mtp.*` draft tensors from a DeepSeek-V4-Flash-DSpark HF snapshot
and writes a standalone sidecar package: per-stage safetensors shards holding
final module-path tensor names plus a content-hashed manifest. The sidecar
carries only the drafter (three DSpark blocks, Markov head, confidence head);
the shared embedding and language-model head stay with the target package, so
the source `mtp` embed/head tensors are dropped, matching the reference
checkpoint conversion.

Source names are resolved once at build time: raw checkpoint names are
normalized with the reference conversion renames, mapped onto the draft module
tree with `sanitize_dspark_weights`, and stored under their final names. The
loader (`moespresso.runtime.deepseek_v4.dspark_load`) performs a strict
`load_weights` and never parses names.

Per-tensor treatment, recorded tensor by tensor in the manifest:

- Routed expert projections: source FP4 (E2M1 packed, UE8M0 per-32 scales) is
  repacked byte-losslessly into the MLX mxfp4 layout; a per-group sample is
  dequantized both ways and must match exactly, otherwise the group falls back
  to affine 8-bit. Float sources are quantized with `mx.quantize(mode="mxfp4")`
  and recorded as not lossless.
- Dense attention/shared-expert/main projections: FP8 (E4M3, UE8M0 block
  scales) is dequantized to bf16 with the probe codec, then affine-quantized
  at 8 bits, group size 32. Float sources are quantized directly.
- Router gate weight and bias: unquantized passthrough (the gate computes in
  fp32 at runtime); an FP8 gate weight is dequantized to fp32.
- Norm weights and Markov head tables: passthrough at the source dtype.
- Hyper-connection parameters, attention sink, and the confidence projection:
  fp32 passthrough.

A second experts mode (`--experts-format iqk --routed-artifacts <dir>`) takes
the routed projections from pre-encoded IQ_K conversion artifacts instead of
the source checkpoint. The staging directory holds one raw cell file per
stage and projection (`tensors/layerNN_{gate,up,down}.<member>`: expert-major
concatenated ik-wire rows) plus an `inventory.json` with per-file digests
that the converter writes only after every unit completes; a staging
directory without the inventory is refused, and every consumed file is
digest-checked first. The builder rearranges each expert row onto the decode
kernels' `iqk_relayout` and stores one uint8 tensor per stage and projection
under the switch seam, shaped `[num_experts, out_features, row_bytes]`. Two
gates run on the rewritten bytes, the pair the package relayout step also
runs: every row is unpacked back to wire bytes and must match exactly, and a
deterministic sample of rows per expert is decoded through ik's CPU
dequantizer and through the relayout reference decode and compared as raw
fp16 bit patterns. Members follow the relayout definition (`iq2_ks`,
`iq2_k`, `iq1_s_r4`); a member with build-side geometry but no relayout and
no decode kernels is refused. Manifest rows record the member, layout, and geometry
per tensor, so a later mixed-member allocation is a manifest fact rather
than a format change. Everything outside the routed experts keeps the
treatment above, and the default mode is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from jang_tools.dsv4.mlx_model import ModelArgs

from moespresso.core.artifact import compute_artifact_id, write_artifact
from moespresso.inventory.safetensors_header import TensorHeader, read_headers_with_offsets
from moespresso.package.bundle import IQK_CODEC, ds4_source_to_mxfp4_components
from moespresso.package.deepseek_v4.iqk_relayout import _sample_row_indices
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    IQKFormatError,
    iqk_geometry,
)
from moespresso.package.iqk_relayout import (
    check_relayout_member,
    decode_rows,
    pack_rows,
    unpack_rows,
    wire_group_rows,
)
from moespresso.runtime.deepseek_v4.dspark_model import (
    DSparkArgs,
    DSparkDraftModel,
    sanitize_dspark_weights,
)

SIDECAR_KIND = "deepseek_v4_dspark_sidecar"
SIDECAR_SCHEMA_MAJOR = 1
SIDECAR_SCHEMA_MINOR = 0
SIDECAR_MANIFEST_NAME = "dspark_sidecar.json"

FORMAT_MXFP4 = "mxfp4"
FORMAT_AFFINE8 = "affine8"
FORMAT_PASSTHROUGH = "passthrough"
FORMAT_IQK = IQK_CODEC
KNOWN_FORMATS = frozenset({
    FORMAT_MXFP4, FORMAT_AFFINE8, FORMAT_PASSTHROUGH, FORMAT_IQK,
})

MXFP4_PARAMS = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
AFFINE8_PARAMS = {"group_size": 32, "bits": 8, "mode": "affine"}

# IQ_K experts mode: routed projections in stage order, the staging file stem
# each maps to, the loader-owned tensor component name, and the row sample
# size of the reference-decode gate (the package relayout step's own default).
IQK_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_IQK_PROJ_FILE = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
IQK_BLOCKS_COMPONENT = "iqk_blocks"
IQK_SAMPLE_ROWS = 64

PRODUCER = {"tool": "moespresso.package.deepseek_v4.dspark_sidecar", "version": "2.0.0"}
SUBJECT = {"family": "deepseek_v4_flash_dspark", "role": "draft_sidecar"}

HUB_CACHE = Path("~/.cache/huggingface/hub").expanduser()
INDEX_NAME = "model.safetensors.index.json"

# Storage dtype token -> numpy dtype used to view the raw bytes. FP8/UE8M0
# codes are read as uint8 and decoded by the probe codec; BF16 is read as
# uint16 and widened to float32 (an exact conversion).
_STORAGE_NP = {
    "I8": np.int8,
    "U8": np.uint8,
    "F8_E4M3": np.uint8,
    "F8_E8M0": np.uint8,
    "F16": np.float16,
    "F32": np.float32,
    "BF16": np.uint16,
}
_FLOAT_TOKENS = ("F16", "F32", "BF16")
_MX_FLOAT = {"F16": mx.float16, "F32": mx.float32, "BF16": mx.bfloat16}

# Stage-relative module bases (final name minus "blocks.N." and the trailing
# component) treated as dense affine 8-bit projections.
_DENSE8_BASES = frozenset({
    "self_attn.wq_a", "self_attn.wq_b", "self_attn.wkv",
    "self_attn.wo_a", "self_attn.wo_b",
    "mlp.shared_experts.gate_proj", "mlp.shared_experts.down_proj",
    "mlp.shared_experts.up_proj",
    "main_proj",
})
_GATE_BASE = "mlp.gate"
# Stage-relative full names kept at the source dtype.
_KEEP_DTYPE_NAMES = frozenset({
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_norm.weight", "self_attn.kv_norm.weight",
    "main_norm.weight", "norm.weight",
    "markov_head.markov_w1.weight", "markov_head.markov_w2.weight",
})
# Stage-relative full names stored as fp32 (a lossless widening of bf16/fp16).
_F32_NAMES = frozenset({
    "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
    "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
    "hc_head_fn", "hc_head_base", "hc_head_scale",
    "self_attn.attn_sink",
    "confidence_head.proj.weight",
})

_EXPERT_RAW_RE = re.compile(r"mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
_FINAL_STAGE_RE = re.compile(r"blocks\.(\d+)\.(.+)$")


class DSparkSidecarError(ValueError):
    """A sidecar build or load contract violation."""


def _mx_dtype_name(dtype) -> str:
    return str(dtype).split(".")[-1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_source_name(name: str) -> str:
    """Apply the reference conversion renames to one raw checkpoint name."""
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    return name


def _is_shared_with_target(name: str) -> bool:
    """Drop mtp embed/head tensors; the target model owns them."""
    return "emb" in name or name.endswith("head.weight")


def _read_storage(source_dir: Path, header: TensorHeader) -> np.ndarray:
    """Read one source tensor preserving its storage codes."""
    np_dtype = _STORAGE_NP.get(header.dtype)
    if np_dtype is None:
        raise DSparkSidecarError(
            f"unsupported source storage dtype {header.dtype!r} for {header.name}")
    nbytes = header.end - header.begin
    with open(source_dir / header.shard, "rb") as f:
        f.seek(header.header_size + header.begin)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise DSparkSidecarError(
            f"short read for {header.name}: {len(raw)} of {nbytes} bytes")
    arr = np.frombuffer(raw, dtype=np_dtype)
    expected = int(np.prod(header.shape)) if header.shape else 1
    if arr.size != expected:
        raise DSparkSidecarError(
            f"{header.name}: {arr.size} elements on disk, header says {expected}")
    return arr.reshape(header.shape)


def _read_float32(source_dir: Path, header: TensorHeader) -> np.ndarray:
    """Read one float source tensor as float32 (exact for bf16/fp16)."""
    codes = _read_storage(source_dir, header)
    if header.dtype == "BF16":
        return (codes.astype(np.uint32) << 16).view(np.float32)
    if header.dtype in ("F16", "F32"):
        return codes.astype(np.float32)
    raise DSparkSidecarError(f"{header.name}: dtype {header.dtype} is not a float dtype")


def _load_float_mx(source_dir: Path, header: TensorHeader) -> mx.array:
    """Read one float source tensor as an mx array at its source dtype."""
    return mx.array(_read_float32(source_dir, header)).astype(_MX_FLOAT[header.dtype])


@dataclass
class _ExpertGroup:
    """One routed projection's per-expert source tensors, in expert order."""

    final_weight: str
    weights: list[TensorHeader]
    scales: list[TensorHeader] | None = None


@dataclass
class _Unit:
    """One module-level conversion unit (weight plus optional scale/bias)."""

    treatment: str  # "dense8" | "gate" | "keep" | "f32"
    weight: TensorHeader | None = None
    scale: TensorHeader | None = None
    bias: TensorHeader | None = None


@dataclass
class _Plan:
    """Resolved build plan: final names bound to source tensors."""

    units: dict[str, _Unit] = field(default_factory=dict)  # keyed by final base name
    expert_groups: dict[str, _ExpertGroup] = field(default_factory=dict)


def _resolve_stage_count(config: dict, source_dir: Path) -> tuple[int, str]:
    """Resolve the declared drafter stage count and the file declaring it.

    The stage count is a declared architecture fact and is unrelated to
    `len(dspark_target_layer_ids)`, which counts tapped main-stack layers,
    so the builder never infers one from the other. The top-level
    `config.json` takes precedence; checkpoints that keep the drafter
    fields in `inference/config.json` are read second; a source that
    declares the count in neither file is refused.
    """
    inference_path = Path(source_dir) / "inference" / "config.json"
    n = config.get("n_mtp_layers")
    declared_by = "config.json"
    if n is None and inference_path.exists():
        inference_config = json.loads(inference_path.read_text())
        n = inference_config.get("n_mtp_layers")
        declared_by = "inference/config.json"
    if n is None:
        raise DSparkSidecarError(
            "source declares no n_mtp_layers; looked in "
            f"{Path(source_dir) / 'config.json'} and {inference_path}. "
            "The drafter stage count must be declared; it cannot be "
            "inferred from dspark_target_layer_ids.")
    n = int(n)
    if n < 1:
        raise DSparkSidecarError(
            f"{declared_by} declares n_mtp_layers={n}; the drafter needs at "
            "least one stage")
    return n, declared_by


def _dspark_fields(config: dict, n_mtp_layers: int) -> dict:
    """Extract the drafter configuration fields from the source config."""
    try:
        fields_out = {
            "n_mtp_layers": int(n_mtp_layers),
            "block_size": int(config["dspark_block_size"]),
            "noise_token_id": int(config["dspark_noise_token_id"]),
            "target_layer_ids": [int(i) for i in config["dspark_target_layer_ids"]],
            "markov_rank": int(config["dspark_markov_rank"]),
        }
    except KeyError as e:
        raise DSparkSidecarError(f"source config is missing dspark field {e}") from e
    return fields_out


def _build_dspark_args(config: dict, n_mtp_layers: int) -> DSparkArgs:
    fields_out = _dspark_fields(config, n_mtp_layers)
    margs = ModelArgs.from_dict(config)
    return DSparkArgs(
        model_args=margs,
        n_mtp_layers=fields_out["n_mtp_layers"],
        block_size=fields_out["block_size"],
        noise_token_id=fields_out["noise_token_id"],
        target_layer_ids=tuple(fields_out["target_layer_ids"]),
        markov_rank=fields_out["markov_rank"],
    )


def _collect_source_headers(source_dir: Path) -> dict[str, TensorHeader]:
    """Map normalized raw mtp names to shard headers, opening only needed shards.

    With the index present, only shards that hold mtp tensors are opened, and
    missing shard files fail with an explicit list (an incomplete download).
    Without the index, every shard on disk is header-scanned; missing tensors
    then surface in the completeness check.
    """
    index_path = source_dir / INDEX_NAME
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        needed_shards = sorted({
            shard for raw, shard in weight_map.items()
            if _normalize_source_name(raw).startswith("mtp.")
        })
        missing = [s for s in needed_shards if not (source_dir / s).exists()]
        if missing:
            raise DSparkSidecarError(
                "source download is incomplete; missing shard files: "
                + ", ".join(missing))
        shard_paths = [source_dir / s for s in needed_shards]
    else:
        shard_paths = sorted(source_dir.glob("*.safetensors"))
    if not shard_paths:
        raise DSparkSidecarError(f"no safetensors shards found under {source_dir}")

    headers: dict[str, TensorHeader] = {}
    for shard_path in shard_paths:
        for header in read_headers_with_offsets(shard_path):
            name = _normalize_source_name(header.name)
            if not name.startswith("mtp.") or _is_shared_with_target(name):
                continue
            if name in headers:
                raise DSparkSidecarError(f"duplicate source tensor {name}")
            headers[name] = header
    if not headers:
        raise DSparkSidecarError(f"no mtp.* tensors found under {source_dir}")
    return headers


def _final_name_for(raw_names: dict[str, object], dargs: DSparkArgs) -> str:
    """Resolve one sanitize call that must produce exactly one final name."""
    out = sanitize_dspark_weights(raw_names, dargs)
    if len(out) != 1:
        raise DSparkSidecarError(
            f"source tensors {sorted(raw_names)[:3]}... resolved to "
            f"{sorted(out)} (expected exactly one final name)")
    return next(iter(out))


def _split_stage_rest(final: str) -> tuple[int, str]:
    m = _FINAL_STAGE_RE.match(final)
    if not m:
        raise DSparkSidecarError(f"final name {final} is not stage-scoped")
    return int(m.group(1)), m.group(2)


def _resolve_plan(headers: dict[str, TensorHeader], dargs: DSparkArgs) -> _Plan:
    """Resolve every source tensor to its final name and conversion unit.

    Naming goes through `sanitize_dspark_weights` (the single naming
    authority); this function only groups the results into units and fails
    closed on anything it cannot place.
    """
    n_experts = dargs.model_args.n_routed_experts
    placeholder = mx.zeros((1,))

    expert_raw: dict[tuple[int, str, str], dict[int, TensorHeader]] = {}
    singles: dict[str, TensorHeader] = {}
    for raw, header in headers.items():
        m = _EXPERT_RAW_RE.match(raw)
        if m:
            key = (int(m.group(1)), m.group(3), m.group(4))
            expert_raw.setdefault(key, {})[int(m.group(2))] = header
        else:
            singles[raw] = header

    plan = _Plan()

    for (stage, wkey, kind), by_expert in sorted(expert_raw.items()):
        missing = sorted(set(range(n_experts)) - set(by_expert))
        extra = sorted(set(by_expert) - set(range(n_experts)))
        if missing or extra:
            raise DSparkSidecarError(
                f"stage {stage} {wkey}.{kind}: expert set mismatch; "
                f"missing={missing[:8]} unexpected={extra[:8]}")
        final = _final_name_for(
            {f"mtp.{stage}.ffn.experts.{e}.{wkey}.{kind}": placeholder
             for e in range(n_experts)},
            dargs,
        )
        ordered = [by_expert[e] for e in range(n_experts)]
        base, comp = final.rsplit(".", 1)
        group = plan.expert_groups.setdefault(
            f"{base}.weight" if comp != "weight" else final,
            _ExpertGroup(final_weight=f"{base}.weight", weights=[]),
        )
        if comp == "weight":
            group.weights = ordered
        elif comp == "scale":
            group.scales = ordered
        else:
            raise DSparkSidecarError(f"unexpected expert component {final}")

    final_singles: dict[str, TensorHeader] = {}
    for raw, header in sorted(singles.items()):
        final = _final_name_for({raw: placeholder}, dargs)
        if final in final_singles:
            raise DSparkSidecarError(f"{raw} collides with another tensor at {final}")
        final_singles[final] = header

    for final, header in final_singles.items():
        stage, rest = _split_stage_rest(final)
        base_final = final.rsplit(".", 1)[0]
        if rest.endswith(".scale"):
            base_rest = rest[: -len(".scale")]
            if base_rest in _DENSE8_BASES:
                unit = plan.units.setdefault(base_final, _Unit(treatment="dense8"))
            elif base_rest == _GATE_BASE:
                unit = plan.units.setdefault(base_final, _Unit(treatment="gate"))
            else:
                raise DSparkSidecarError(f"scale tensor {final} has no quantized owner")
            unit.scale = header
        elif rest.endswith(".weight") and rest[: -len(".weight")] in _DENSE8_BASES:
            unit = plan.units.setdefault(base_final, _Unit(treatment="dense8"))
            unit.weight = header
        elif rest == f"{_GATE_BASE}.weight":
            unit = plan.units.setdefault(base_final, _Unit(treatment="gate"))
            unit.weight = header
        elif rest == f"{_GATE_BASE}.bias":
            unit = plan.units.setdefault(base_final, _Unit(treatment="gate"))
            unit.bias = header
        elif rest in _KEEP_DTYPE_NAMES:
            plan.units[final] = _Unit(treatment="keep", weight=header)
        elif rest in _F32_NAMES:
            plan.units[final] = _Unit(treatment="f32", weight=header)
        else:
            raise DSparkSidecarError(f"no conversion rule for source tensor {final}")

    for base, unit in plan.units.items():
        if unit.treatment in ("dense8", "gate") and unit.weight is None and unit.bias is None:
            raise DSparkSidecarError(f"{base}: scale present without weight")
    for key, group in plan.expert_groups.items():
        if not group.weights:
            raise DSparkSidecarError(f"{key}: expert scales present without weights")
        if group.scales is not None and len(group.scales) != len(group.weights):
            raise DSparkSidecarError(f"{key}: expert weight/scale counts differ")
    return plan


def _logical_names(plan: _Plan) -> set[str]:
    """Final parameter names the plan will populate, before quant expansion."""
    names: set[str] = set()
    for base, unit in plan.units.items():
        if unit.treatment in ("keep", "f32"):
            names.add(base)
        else:
            if unit.weight is not None:
                names.add(f"{base}.weight")
            if unit.bias is not None:
                names.add(f"{base}.bias")
    for group in plan.expert_groups.values():
        names.add(group.final_weight)
    return names


def _expected_params(dargs: DSparkArgs, module_formats: dict[str, dict]) -> dict[str, tuple]:
    """Parameter names and shapes of the draft model under the given formats.

    The model is built lazily (arrays are never evaluated), so enumerating the
    real-shape tree is cheap.
    """
    model = DSparkDraftModel(dargs)
    if module_formats:
        nn.quantize(
            model,
            class_predicate=lambda path, m: dict(module_formats[path])
            if path in module_formats else False,
        )
    return {name: tuple(a.shape) for name, a in tree_flatten(model.parameters())}


def _check_key_sets(
    expected: dict[str, tuple],
    produced: dict[str, tuple],
    what: str,
    check_shapes: bool = True,
) -> None:
    missing = sorted(set(expected) - set(produced))
    unexpected = sorted(set(produced) - set(expected))
    if missing or unexpected:
        raise DSparkSidecarError(
            f"{what}: tensor set does not match the draft model; "
            f"missing={missing} unexpected={unexpected}")
    if not check_shapes:
        return
    bad_shapes = [
        f"{name}: {produced[name]} != {expected[name]}"
        for name in sorted(expected)
        if expected[name] != produced[name]
    ]
    if bad_shapes:
        raise DSparkSidecarError(f"{what}: shape mismatches; " + "; ".join(bad_shapes))


def _process_expert_group(
    source_dir: Path, group: _ExpertGroup,
) -> tuple[dict[str, mx.array], dict[str, dict], dict]:
    """Convert one routed projection to stacked quantized parameters.

    Returns (tensors, manifest rows keyed by final name, module format params).
    """
    from moespresso.probe.deepseek_v4.codec import dequant_fp4_e2m1_ue8m0

    base = group.final_weight[: -len(".weight")]
    n_experts = len(group.weights)
    w_dtypes = {h.dtype for h in group.weights}

    if w_dtypes == {"I8"} and group.scales is not None:
        out_dim, packed_cols = group.weights[0].shape
        in_dim = packed_cols * 2
        packed = np.empty((n_experts, out_dim, in_dim // 8), dtype=np.uint32)
        scales = np.empty((n_experts, out_dim, in_dim // 32), dtype=np.uint8)
        for e in range(n_experts):
            comp = ds4_source_to_mxfp4_components(
                _read_storage(source_dir, group.weights[e]),
                _read_storage(source_dir, group.scales[e]),
            )
            packed[e] = comp["packed"]
            scales[e] = comp["scales"]

        # Sample identity check: the repacked bytes must dequantize to exactly
        # the source values. On any difference the group falls back to affine
        # 8-bit built from the dequantized source.
        ref = dequant_fp4_e2m1_ue8m0(
            _read_storage(source_dir, group.weights[0]),
            _read_storage(source_dir, group.scales[0]),
            out_dtype=np.float32,
        )
        got = np.array(mx.dequantize(
            mx.array(packed[0]), mx.array(scales[0]), None, **MXFP4_PARAMS,
        ).astype(mx.float32))
        if not np.array_equal(got, ref):
            return _expert_group_affine8_from_fp4(source_dir, group)

        tensors = {
            f"{base}.weight": mx.array(packed),
            f"{base}.scales": mx.array(scales),
        }
        rows = {
            name: {
                "format": FORMAT_MXFP4,
                "quant": dict(MXFP4_PARAMS),
                "lossless": True,
                "source_dtype": "I8+F8_E8M0",
            }
            for name in tensors
        }
        return tensors, rows, {base: dict(MXFP4_PARAMS)}

    if w_dtypes <= set(_FLOAT_TOKENS) and group.scales is None:
        packed_rows = []
        scale_rows = []
        for header in group.weights:
            w = _load_float_mx(source_dir, header)
            qw, qs = mx.quantize(w, **MXFP4_PARAMS)
            packed_rows.append(qw)
            scale_rows.append(qs)
        tensors = {
            f"{base}.weight": mx.stack(packed_rows),
            f"{base}.scales": mx.stack(scale_rows),
        }
        mx.eval(tensors[f"{base}.weight"], tensors[f"{base}.scales"])
        rows = {
            name: {
                "format": FORMAT_MXFP4,
                "quant": dict(MXFP4_PARAMS),
                "lossless": False,
                "source_dtype": group.weights[0].dtype,
                "note": "requantized from a float source; not a byte repack",
            }
            for name in tensors
        }
        return tensors, rows, {base: dict(MXFP4_PARAMS)}

    raise DSparkSidecarError(
        f"{base}: unsupported expert source combination "
        f"(dtypes {sorted(w_dtypes)}, scales={'yes' if group.scales else 'no'})")


def _expert_group_affine8_from_fp4(
    source_dir: Path, group: _ExpertGroup,
) -> tuple[dict[str, mx.array], dict[str, dict], dict]:
    """Fallback: dequantize FP4 experts and store affine 8-bit parameters."""
    from moespresso.probe.deepseek_v4.codec import dequant_fp4_e2m1_ue8m0

    base = group.final_weight[: -len(".weight")]
    weights, scales, biases = [], [], []
    for w_header, s_header in zip(group.weights, group.scales):
        w = dequant_fp4_e2m1_ue8m0(
            _read_storage(source_dir, w_header),
            _read_storage(source_dir, s_header),
            out_dtype=np.float32,
        )
        qw, qs, qb = mx.quantize(mx.array(w).astype(mx.bfloat16), **AFFINE8_PARAMS)
        weights.append(qw)
        scales.append(qs)
        biases.append(qb)
    tensors = {
        f"{base}.weight": mx.stack(weights),
        f"{base}.scales": mx.stack(scales),
        f"{base}.biases": mx.stack(biases),
    }
    mx.eval(*tensors.values())
    rows = {
        name: {
            "format": FORMAT_AFFINE8,
            "quant": dict(AFFINE8_PARAMS),
            "source_dtype": "I8+F8_E8M0",
            "note": "mxfp4 repack sample check failed; dequantized and requantized",
        }
        for name in tensors
    }
    return tensors, rows, {base: dict(AFFINE8_PARAMS)}


def iqk_blocks_name(stage: int, projection: str) -> str:
    """Tensor name of one stage projection's IQ_K expert rows.

    The name sits under the switch seam the loader replaces; it is a stored
    payload the loader consumes, not a module parameter of the draft tree.
    """
    return f"blocks.{stage}.mlp.switch_mlp.{projection}.{IQK_BLOCKS_COMPONENT}"


def iqk_cell_features(margs, projection: str) -> tuple[int, int]:
    """(in_features, out_features) of one routed projection."""
    if projection == "down_proj":
        return int(margs.moe_intermediate_size), int(margs.hidden_size)
    return int(margs.hidden_size), int(margs.moe_intermediate_size)


@dataclass
class _IqkStaging:
    """A verified view of one member's conversion staging directory."""

    directory: Path
    inventory_path: Path
    member: str
    files: dict


def _read_iqk_staging(routed_artifacts: Path) -> _IqkStaging:
    """Open a staging directory, refusing an incomplete or unknown one."""
    directory = Path(routed_artifacts)
    inventory_path = directory / "inventory.json"
    if not inventory_path.is_file():
        raise DSparkSidecarError(
            f"missing {inventory_path}; the converter writes the inventory "
            "only after every unit completes, so a staging directory without "
            "one is incomplete")
    inventory = json.loads(inventory_path.read_text())
    member = inventory.get("member")
    try:
        iqk_geometry(member)
        check_relayout_member(member)
    except IQKFormatError as e:
        raise DSparkSidecarError(f"{inventory_path}: {e}") from e
    files = inventory.get("files")
    if not isinstance(files, dict) or not files:
        raise DSparkSidecarError(f"{inventory_path} lists no tensor files")
    return _IqkStaging(
        directory=directory,
        inventory_path=inventory_path,
        member=member,
        files=files,
    )


def _convert_iqk_expert_cell(
    staging: _IqkStaging, stage: int, projection: str, dargs: DSparkArgs,
) -> tuple[dict[str, mx.array], dict[str, dict], dict]:
    """Relayout one staged routed projection and gate the rewritten bytes.

    Returns (tensors, manifest rows, gate facts). Both relayout gates run:
    every expert row is unpacked back to wire bytes and compared exactly, and
    a deterministic row sample per expert is decoded through ik's CPU
    dequantizer and the relayout reference and compared as fp16 bit patterns.
    """
    from mlx_iqk import codec as iqk_codec

    margs = dargs.model_args
    member = staging.member
    layer_index = margs.num_hidden_layers + stage
    file_name = f"layer{layer_index:02d}_{_IQK_PROJ_FILE[projection]}.{member}"
    entry = staging.files.get(file_name)
    if entry is None:
        raise DSparkSidecarError(
            f"staging inventory {staging.inventory_path} does not list "
            f"{file_name}")
    path = staging.directory / "tensors" / file_name
    if not path.is_file():
        raise DSparkSidecarError(f"missing staging tensor file {path}")

    num_experts = int(margs.n_routed_experts)
    in_features, out_features = iqk_cell_features(margs, projection)
    try:
        row_bytes = iqk_geometry(member).bytes_per_row(in_features)
    except ValueError as e:
        raise DSparkSidecarError(f"{file_name}: {e}") from e
    expert_bytes = row_bytes * out_features
    want_bytes = expert_bytes * num_experts
    declared = int(entry.get("bytes", -1))
    actual = path.stat().st_size
    if actual != want_bytes or declared != want_bytes:
        raise DSparkSidecarError(
            f"{file_name}: {actual} B on disk and {declared} B in the "
            f"inventory; {num_experts} experts of {out_features} rows x "
            f"{row_bytes} B need {want_bytes} B")
    per_expert = entry.get("bytes_per_expert")
    if per_expert is not None and int(per_expert) != expert_bytes:
        raise DSparkSidecarError(
            f"{file_name}: inventory bytes_per_expert {per_expert} against "
            f"the {member} geometry's {expert_bytes}")
    digest = _sha256_file(path)
    if digest != entry.get("sha256"):
        raise DSparkSidecarError(
            f"{file_name}: sha256 {digest} does not match the inventory "
            f"digest {entry.get('sha256')}")

    blocks = np.empty((num_experts, out_features, row_bytes), dtype=np.uint8)
    sampled = 0
    with open(path, "rb") as f:
        for expert in range(num_experts):
            raw = f.read(expert_bytes)
            if len(raw) != expert_bytes:
                raise DSparkSidecarError(
                    f"{file_name}: expert {expert} short read "
                    f"({len(raw)} of {expert_bytes} B)")
            wire = np.frombuffer(raw, dtype=np.uint8).reshape(
                out_features, row_bytes)
            packed = pack_rows(member, wire, in_features)
            if not np.array_equal(unpack_rows(member, packed, in_features), wire):
                raise DSparkSidecarError(
                    f"{file_name}: expert {expert} does not round trip; the "
                    "rearranged bytes are not the staged bytes")
            picks = _sample_row_indices(
                out_features, IQK_SAMPLE_ROWS,
                salt=layer_index * 1_000_003 + expert * 1009 + len(projection),
                group=wire_group_rows(member))
            if picks.size:
                want = iqk_codec.dequantize(
                    member, np.ascontiguousarray(wire[picks]), in_features,
                ).astype(np.float16)
                got = decode_rows(
                    member, np.ascontiguousarray(packed[picks]), in_features,
                ).astype(np.float16)
                bad = int(np.count_nonzero(
                    want.view(np.uint16) != got.view(np.uint16)))
                if bad:
                    raise DSparkSidecarError(
                        f"{file_name}: expert {expert} decodes to {bad} "
                        "differing fp16 values through the two references")
                sampled += int(picks.size)
            blocks[expert] = packed

    final = iqk_blocks_name(stage, projection)
    tensors = {final: mx.array(blocks)}
    rows = {
        final: {
            "format": FORMAT_IQK,
            "iqk_codec": member,
            "layout": IQK_LAYOUT_IQK_RELAYOUT,
            "num_experts": num_experts,
            "out_features": out_features,
            "in_features": in_features,
            "stage": stage,
            "projection": projection,
            "staging_file": file_name,
        }
    }
    gate_facts = {
        "file": file_name,
        "sha256": digest,
        "rows_reference_decoded": sampled,
    }
    return tensors, rows, gate_facts


def _load_dense_float(source_dir: Path, unit: _Unit) -> tuple[mx.array, str]:
    """Load a dense weight as a float mx array, dequantizing FP8 sources."""
    header = unit.weight
    if header.dtype == "F8_E4M3":
        if unit.scale is None:
            raise DSparkSidecarError(f"{header.name}: FP8 weight without a scale tensor")
        from moespresso.probe.deepseek_v4.codec import dequant_fp8_e4m3_ue8m0

        w = dequant_fp8_e4m3_ue8m0(
            _read_storage(source_dir, header),
            _read_storage(source_dir, unit.scale),
            out_dtype=np.float32,
        )
        return mx.array(w).astype(mx.bfloat16), f"{header.dtype}+{unit.scale.dtype}"
    if header.dtype in _FLOAT_TOKENS:
        if unit.scale is not None:
            raise DSparkSidecarError(f"{header.name}: float weight with a scale tensor")
        return _load_float_mx(source_dir, header), header.dtype
    raise DSparkSidecarError(f"{header.name}: unsupported dense source dtype {header.dtype}")


def _process_unit(
    source_dir: Path, base: str, unit: _Unit,
) -> tuple[dict[str, mx.array], dict[str, dict], dict]:
    """Convert one non-expert unit. Returns (tensors, rows, module formats)."""
    if unit.treatment == "dense8":
        w, source_dtype = _load_dense_float(source_dir, unit)
        qw, qs, qb = mx.quantize(w, **AFFINE8_PARAMS)
        tensors = {f"{base}.weight": qw, f"{base}.scales": qs, f"{base}.biases": qb}
        mx.eval(*tensors.values())
        rows = {
            name: {
                "format": FORMAT_AFFINE8,
                "quant": dict(AFFINE8_PARAMS),
                "source_dtype": source_dtype,
            }
            for name in tensors
        }
        return tensors, rows, {base: dict(AFFINE8_PARAMS)}

    if unit.treatment == "gate":
        tensors = {}
        rows = {}
        if unit.weight is not None:
            if unit.weight.dtype == "F8_E4M3":
                if unit.scale is None:
                    raise DSparkSidecarError(
                        f"{unit.weight.name}: FP8 gate weight without a scale tensor")
                from moespresso.probe.deepseek_v4.codec import dequant_fp8_e4m3_ue8m0

                w = mx.array(dequant_fp8_e4m3_ue8m0(
                    _read_storage(source_dir, unit.weight),
                    _read_storage(source_dir, unit.scale),
                    out_dtype=np.float32,
                ))
                source_dtype = f"{unit.weight.dtype}+{unit.scale.dtype}"
                note = "gate weight dequantized to fp32; router stays unquantized"
            else:
                w, source_dtype = _load_dense_float(source_dir, unit)
                note = None
            tensors[f"{base}.weight"] = w
            rows[f"{base}.weight"] = {
                "format": FORMAT_PASSTHROUGH, "source_dtype": source_dtype,
            }
            if note:
                rows[f"{base}.weight"]["note"] = note
        if unit.bias is not None:
            tensors[f"{base}.bias"] = _load_float_mx(source_dir, unit.bias)
            rows[f"{base}.bias"] = {
                "format": FORMAT_PASSTHROUGH, "source_dtype": unit.bias.dtype,
            }
        return tensors, rows, {}

    if unit.treatment == "keep":
        arr = _load_float_mx(source_dir, unit.weight)
        return (
            {base: arr},
            {base: {"format": FORMAT_PASSTHROUGH, "source_dtype": unit.weight.dtype}},
            {},
        )

    if unit.treatment == "f32":
        arr = mx.array(_read_float32(source_dir, unit.weight))
        return (
            {base: arr},
            {base: {"format": FORMAT_PASSTHROUGH, "source_dtype": unit.weight.dtype}},
            {},
        )

    raise DSparkSidecarError(f"unknown treatment {unit.treatment!r} for {base}")


def build_dspark_sidecar(
    source_dir: Path,
    output_dir: Path,
    experts_format: str = FORMAT_MXFP4,
    routed_artifacts: Path | None = None,
) -> dict:
    """Build the sidecar package. Returns the written manifest payload.

    `experts_format` selects the routed-expert treatment: the default
    `mxfp4` mode converts the source checkpoint's own expert tensors, and
    `iqk` mode takes them from the `routed_artifacts` staging directory
    instead. The source resolution, completeness checks, and every
    non-expert treatment are the same in both modes.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if experts_format not in (FORMAT_MXFP4, FORMAT_IQK):
        raise DSparkSidecarError(
            f"unknown experts format {experts_format!r}; known: "
            f"{[FORMAT_MXFP4, FORMAT_IQK]}")
    staging = None
    if experts_format == FORMAT_IQK:
        if routed_artifacts is None:
            raise DSparkSidecarError(
                "experts format iqk needs --routed-artifacts, the staging "
                "directory the conversion wrote")
        staging = _read_iqk_staging(routed_artifacts)
    elif routed_artifacts is not None:
        raise DSparkSidecarError(
            "--routed-artifacts is an iqk-mode input; the mxfp4 mode reads "
            "the routed experts from the source checkpoint")
    config_path = source_dir / "config.json"
    if not config_path.exists():
        raise DSparkSidecarError(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    n_mtp_layers, stage_count_source = _resolve_stage_count(config, source_dir)
    dargs = _build_dspark_args(config, n_mtp_layers)
    n_stages = dargs.n_mtp_layers

    headers = _collect_source_headers(source_dir)
    plan = _resolve_plan(headers, dargs)

    # Pre-flight: the plan must cover the draft parameter tree exactly before
    # any heavy conversion work starts. Shapes are checked on the produced
    # tensors after conversion.
    _check_key_sets(
        _expected_params(dargs, {}),
        {name: () for name in _logical_names(plan)},
        "source resolution",
        check_shapes=False,
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise DSparkSidecarError(f"output directory {output_dir} is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_names = [
        f"model-dspark-{i + 1:05d}-of-{n_stages:05d}.safetensors" for i in range(n_stages)
    ]
    tensor_rows: dict[str, dict] = {}
    module_formats: dict[str, dict] = {}
    produced: dict[str, tuple] = {}
    file_sha256: dict[str, str] = {}

    def _take(
        stage_tensors: dict, tensors: dict, rows: dict, formats: dict, shard: str,
    ) -> None:
        for name, arr in tensors.items():
            stage_tensors[name] = arr
            produced[name] = tuple(arr.shape)
            row = dict(rows[name])
            row.update({
                "dtype": _mx_dtype_name(arr.dtype),
                "shape": [int(d) for d in arr.shape],
                "file": shard,
            })
            tensor_rows[name] = row
        module_formats.update(formats)

    iqk_file_sha256: dict[str, str] = {}
    iqk_rows_decoded = 0
    for stage in range(n_stages):
        stage_prefix = f"blocks.{stage}."
        stage_tensors: dict[str, mx.array] = {}
        if staging is None:
            for key in sorted(plan.expert_groups):
                if not key.startswith(stage_prefix):
                    continue
                tensors, rows, formats = _process_expert_group(
                    source_dir, plan.expert_groups[key])
                _take(stage_tensors, tensors, rows, formats, shard_names[stage])
        else:
            # The plan's expert groups proved the source complete; their
            # bytes are not read. The routed projections come from the
            # staging cells, gated row by row.
            for projection in IQK_PROJECTIONS:
                tensors, rows, gate_facts = _convert_iqk_expert_cell(
                    staging, stage, projection, dargs)
                _take(stage_tensors, tensors, rows, {}, shard_names[stage])
                iqk_file_sha256[gate_facts["file"]] = gate_facts["sha256"]
                iqk_rows_decoded += gate_facts["rows_reference_decoded"]
        for base in sorted(plan.units):
            if not base.startswith(stage_prefix):
                continue
            tensors, rows, formats = _process_unit(source_dir, base, plan.units[base])
            _take(stage_tensors, tensors, rows, formats, shard_names[stage])

        if not stage_tensors:
            raise DSparkSidecarError(f"stage {stage} produced no tensors")
        shard_path = output_dir / shard_names[stage]
        mx.save_safetensors(str(shard_path), stage_tensors, metadata={"format": "mlx"})
        file_sha256[shard_names[stage]] = _sha256_file(shard_path)

    expected = _expected_params(dargs, module_formats)
    if staging is None:
        _check_key_sets(expected, produced, "sidecar output")
    else:
        # The IQ_K expert rows are stored payloads for the loader's switch
        # install, not draft-tree parameters, so the tree comparison covers
        # everything outside the switch seam and the expert keys are checked
        # against the exact seam set.
        want_iqk = {
            iqk_blocks_name(s, p)
            for s in range(n_stages) for p in IQK_PROJECTIONS
        }
        got_iqk = {
            name for name, row in tensor_rows.items()
            if row["format"] == FORMAT_IQK
        }
        if got_iqk != want_iqk:
            raise DSparkSidecarError(
                "sidecar output: IQ_K expert rows do not cover the switch "
                f"seams; missing={sorted(want_iqk - got_iqk)} "
                f"unexpected={sorted(got_iqk - want_iqk)}")
        switch_bases = tuple(
            f"blocks.{s}.mlp.switch_mlp." for s in range(n_stages))
        expected = {
            k: v for k, v in expected.items() if not k.startswith(switch_bases)
        }
        rest = {k: v for k, v in produced.items() if k not in want_iqk}
        _check_key_sets(expected, rest, "sidecar output")

    snapshot = source_dir.resolve()
    commit = snapshot.name
    payload = {
        "artifact_kind": SIDECAR_KIND,
        "schema_version": {"major": SIDECAR_SCHEMA_MAJOR, "minor": SIDECAR_SCHEMA_MINOR},
        "producer": dict(PRODUCER),
        "subject": dict(SUBJECT),
        "status": "valid",
        "formats": {FORMAT_MXFP4: dict(MXFP4_PARAMS), FORMAT_AFFINE8: dict(AFFINE8_PARAMS)},
        "provenance": {
            "source_snapshot": str(snapshot),
            "source_commit": commit,
            "file_sha256": file_sha256,
            # The declared stage count and the config file that declared it;
            # the count is never inferred from dspark_target_layer_ids.
            "n_mtp_layers": n_mtp_layers,
            "n_mtp_layers_source": stage_count_source,
        },
        "dspark": _dspark_fields(config, n_mtp_layers),
        "source_config": config,
        "tensors": tensor_rows,
    }
    if staging is not None:
        payload["provenance"]["iqk_staging"] = {
            "dir": str(staging.directory.resolve()),
            "member": staging.member,
            "file_sha256": dict(sorted(iqk_file_sha256.items())),
        }
        payload["provenance"]["iqk_gates"] = {
            "round_trip": "every row",
            "reference_decode_sample_rows": IQK_SAMPLE_ROWS,
            "rows_reference_decoded": iqk_rows_decoded,
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
        description="Build the DSpark draft-model sidecar from an HF snapshot.")
    parser.add_argument(
        "--source", required=True,
        help="DeepSeek-V4-Flash-DSpark snapshot directory (config.json + shards)")
    parser.add_argument(
        "--output", required=True,
        help="sidecar directory: a path, or a bare name placed under the HF hub cache")
    parser.add_argument(
        "--experts-format", choices=[FORMAT_MXFP4, FORMAT_IQK],
        default=FORMAT_MXFP4,
        help="routed-expert treatment: mxfp4 converts the source checkpoint's "
             "experts (the default); iqk relays pre-encoded IQ_K conversion "
             "artifacts onto the decode kernels' layout")
    parser.add_argument(
        "--routed-artifacts", default=None,
        help="iqk mode only: staging directory holding tensors/layerNN_* "
             "cell files and the converter's inventory.json")
    args = parser.parse_args(argv)

    try:
        payload = build_dspark_sidecar(
            Path(args.source),
            resolve_output_dir(args.output),
            experts_format=args.experts_format,
            routed_artifacts=(
                Path(args.routed_artifacts).expanduser()
                if args.routed_artifacts is not None else None),
        )
    except DSparkSidecarError as e:
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
