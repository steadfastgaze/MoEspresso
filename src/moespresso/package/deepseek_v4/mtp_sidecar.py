"""MTP draft-model sidecar builder for DeepSeek-V4-Flash.

Reads the `mtp.0.*` draft tensors from the DeepSeek-V4-Flash base HF
snapshot and writes a standalone sidecar package: one safetensors shard
holding final module-path tensor names plus a content-hashed manifest. The
sidecar carries only the drafter (one vendored decoder block, the fusion
projections and norms, the hc-head parameters); the shared embedding and
language-model head stay with the target package, so the source `mtp`
embed/head tensors are dropped, matching the reference checkpoint layout.

Source names are resolved once at build time: raw checkpoint names are
normalized with the reference conversion renames, mapped onto the draft
module tree with `sanitize_mtp_weights`, and stored under their final
names. The loader (`moespresso.runtime.deepseek_v4.mtp_load`) performs a
strict `load_weights` and never parses names.

Per-tensor treatment, recorded tensor by tensor in the manifest:

- Routed expert projections: source FP4 (E2M1 packed, UE8M0 per-32 scales) is
  repacked byte-losslessly into the MLX mxfp4 layout; a per-group sample is
  dequantized both ways and must match exactly, otherwise the group falls back
  to affine 8-bit. Float sources are quantized with `mx.quantize(mode="mxfp4")`
  and recorded as not lossless.
- Dense attention/shared-expert projections and the fusion projections
  e_proj/h_proj: FP8 (E4M3, UE8M0 block scales) is dequantized to bf16 with
  the probe codec, then affine-quantized at 8 bits, group size 32. Float
  sources are quantized directly.
- Router gate weight and bias: unquantized passthrough (the gate computes in
  fp32 at runtime); an FP8 gate weight is dequantized to fp32.
- Fusion and block norm weights: passthrough at the source dtype. The norms
  and fusion glue are precision-sensitive; they are never quantized.
- Hyper-connection parameters and the attention sink: fp32 passthrough.
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
from moespresso.package.bundle import ds4_source_to_mxfp4_components
from moespresso.runtime.deepseek_v4.mtp_model import (
    MTPArgs,
    MTPDraftModel,
    sanitize_mtp_weights,
)

SIDECAR_KIND = "deepseek_v4_mtp_sidecar"
SIDECAR_SCHEMA_MAJOR = 1
SIDECAR_SCHEMA_MINOR = 0
SIDECAR_MANIFEST_NAME = "mtp_sidecar.json"

FORMAT_MXFP4 = "mxfp4"
FORMAT_AFFINE8 = "affine8"
FORMAT_PASSTHROUGH = "passthrough"
KNOWN_FORMATS = frozenset({FORMAT_MXFP4, FORMAT_AFFINE8, FORMAT_PASSTHROUGH})

MXFP4_PARAMS = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
AFFINE8_PARAMS = {"group_size": 32, "bits": 8, "mode": "affine"}

PRODUCER = {"tool": "moespresso.package.deepseek_v4.mtp_sidecar", "version": "2.0.0"}
SUBJECT = {"family": "deepseek_v4_flash_mtp", "role": "draft_sidecar"}

# Chained draft depth cap recorded in the manifest. Chaining the single
# vendor-trained module to depth 3 is the proven operating range; the
# adaptive scheduler chooses the per-round depth below the cap.
DEFAULT_BLOCK_SIZE = 3

HUB_CACHE = Path("~/.cache/huggingface/hub").expanduser()
INDEX_NAME = "model.safetensors.index.json"
SHARD_NAME = "model-mtp-00001-of-00001.safetensors"

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

# Final module bases (final tensor name minus the trailing component)
# treated as dense affine 8-bit projections.
_DENSE8_BASES = frozenset({
    "block.self_attn.wq_a", "block.self_attn.wq_b", "block.self_attn.wkv",
    "block.self_attn.wo_a", "block.self_attn.wo_b",
    "block.mlp.shared_experts.gate_proj", "block.mlp.shared_experts.down_proj",
    "block.mlp.shared_experts.up_proj",
    "e_proj", "h_proj",
})
_GATE_BASE = "block.mlp.gate"
# Final full names kept at the source dtype.
_KEEP_DTYPE_NAMES = frozenset({
    "block.input_layernorm.weight", "block.post_attention_layernorm.weight",
    "block.self_attn.q_norm.weight", "block.self_attn.kv_norm.weight",
    "enorm.weight", "hnorm.weight", "norm.weight",
})
# Final full names stored as fp32 (a lossless widening of bf16/fp16).
_F32_NAMES = frozenset({
    "block.hc_attn_fn", "block.hc_attn_base", "block.hc_attn_scale",
    "block.hc_ffn_fn", "block.hc_ffn_base", "block.hc_ffn_scale",
    "hc_head_fn", "hc_head_base", "hc_head_scale",
    "block.self_attn.attn_sink",
})

_EXPERT_RAW_RE = re.compile(r"mtp\.0\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")


class MTPSidecarError(ValueError):
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
        raise MTPSidecarError(
            f"unsupported source storage dtype {header.dtype!r} for {header.name}")
    nbytes = header.end - header.begin
    with open(source_dir / header.shard, "rb") as f:
        f.seek(header.header_size + header.begin)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise MTPSidecarError(
            f"short read for {header.name}: {len(raw)} of {nbytes} bytes")
    arr = np.frombuffer(raw, dtype=np_dtype)
    expected = int(np.prod(header.shape)) if header.shape else 1
    if arr.size != expected:
        raise MTPSidecarError(
            f"{header.name}: {arr.size} elements on disk, header says {expected}")
    return arr.reshape(header.shape)


def _read_float32(source_dir: Path, header: TensorHeader) -> np.ndarray:
    """Read one float source tensor as float32 (exact for bf16/fp16)."""
    codes = _read_storage(source_dir, header)
    if header.dtype == "BF16":
        return (codes.astype(np.uint32) << 16).view(np.float32)
    if header.dtype in ("F16", "F32"):
        return codes.astype(np.float32)
    raise MTPSidecarError(f"{header.name}: dtype {header.dtype} is not a float dtype")


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


def _mtp_fields(config: dict) -> dict:
    """Extract and validate the drafter configuration from the source config."""
    n = config.get("num_nextn_predict_layers", config.get("n_mtp_layers"))
    if n is None:
        raise MTPSidecarError("source config carries no MTP module count")
    if int(n) != 1:
        raise MTPSidecarError(
            f"the MTP drafter chains a single module; source declares {n}")
    return {"n_mtp_layers": 1, "block_size": DEFAULT_BLOCK_SIZE}


def _build_mtp_args(config: dict) -> MTPArgs:
    fields_out = _mtp_fields(config)
    margs = ModelArgs.from_dict(config)
    return MTPArgs(model_args=margs, block_size=fields_out["block_size"])


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
            raise MTPSidecarError(
                "source download is incomplete; missing shard files: "
                + ", ".join(missing))
        shard_paths = [source_dir / s for s in needed_shards]
    else:
        shard_paths = sorted(source_dir.glob("*.safetensors"))
    if not shard_paths:
        raise MTPSidecarError(f"no safetensors shards found under {source_dir}")

    headers: dict[str, TensorHeader] = {}
    for shard_path in shard_paths:
        for header in read_headers_with_offsets(shard_path):
            name = _normalize_source_name(header.name)
            if not name.startswith("mtp.") or _is_shared_with_target(name):
                continue
            if name in headers:
                raise MTPSidecarError(f"duplicate source tensor {name}")
            headers[name] = header
    if not headers:
        raise MTPSidecarError(f"no mtp.* tensors found under {source_dir}")
    return headers


def _final_name_for(raw_names: dict[str, object], args: MTPArgs) -> str:
    """Resolve one sanitize call that must produce exactly one final name."""
    out = sanitize_mtp_weights(raw_names, args)
    if len(out) != 1:
        raise MTPSidecarError(
            f"source tensors {sorted(raw_names)[:3]}... resolved to "
            f"{sorted(out)} (expected exactly one final name)")
    return next(iter(out))


def _resolve_plan(headers: dict[str, TensorHeader], args: MTPArgs) -> _Plan:
    """Resolve every source tensor to its final name and conversion unit.

    Naming goes through `sanitize_mtp_weights` (the single naming
    authority); this function only groups the results into units and fails
    closed on anything it cannot place.
    """
    n_experts = args.model_args.n_routed_experts
    placeholder = mx.zeros((1,))

    expert_raw: dict[tuple[str, str], dict[int, TensorHeader]] = {}
    singles: dict[str, TensorHeader] = {}
    for raw, header in headers.items():
        m = _EXPERT_RAW_RE.match(raw)
        if m:
            key = (m.group(2), m.group(3))
            expert_raw.setdefault(key, {})[int(m.group(1))] = header
        else:
            singles[raw] = header

    plan = _Plan()

    for (wkey, kind), by_expert in sorted(expert_raw.items()):
        missing = sorted(set(range(n_experts)) - set(by_expert))
        extra = sorted(set(by_expert) - set(range(n_experts)))
        if missing or extra:
            raise MTPSidecarError(
                f"{wkey}.{kind}: expert set mismatch; "
                f"missing={missing[:8]} unexpected={extra[:8]}")
        final = _final_name_for(
            {f"mtp.0.ffn.experts.{e}.{wkey}.{kind}": placeholder
             for e in range(n_experts)},
            args,
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
            raise MTPSidecarError(f"unexpected expert component {final}")

    final_singles: dict[str, TensorHeader] = {}
    for raw, header in sorted(singles.items()):
        final = _final_name_for({raw: placeholder}, args)
        if final in final_singles:
            raise MTPSidecarError(f"{raw} collides with another tensor at {final}")
        final_singles[final] = header

    for final, header in final_singles.items():
        base_final = final.rsplit(".", 1)[0]
        if final.endswith(".scale"):
            base = final[: -len(".scale")]
            if base in _DENSE8_BASES:
                unit = plan.units.setdefault(base_final, _Unit(treatment="dense8"))
            elif base == _GATE_BASE:
                unit = plan.units.setdefault(base_final, _Unit(treatment="gate"))
            else:
                raise MTPSidecarError(f"scale tensor {final} has no quantized owner")
            unit.scale = header
        elif final.endswith(".weight") and final[: -len(".weight")] in _DENSE8_BASES:
            unit = plan.units.setdefault(base_final, _Unit(treatment="dense8"))
            unit.weight = header
        elif final == f"{_GATE_BASE}.weight":
            unit = plan.units.setdefault(base_final, _Unit(treatment="gate"))
            unit.weight = header
        elif final == f"{_GATE_BASE}.bias":
            unit = plan.units.setdefault(base_final, _Unit(treatment="gate"))
            unit.bias = header
        elif final in _KEEP_DTYPE_NAMES:
            plan.units[final] = _Unit(treatment="keep", weight=header)
        elif final in _F32_NAMES:
            plan.units[final] = _Unit(treatment="f32", weight=header)
        else:
            raise MTPSidecarError(f"no conversion rule for source tensor {final}")

    for base, unit in plan.units.items():
        if unit.treatment in ("dense8", "gate") and unit.weight is None and unit.bias is None:
            raise MTPSidecarError(f"{base}: scale present without weight")
    for key, group in plan.expert_groups.items():
        if not group.weights:
            raise MTPSidecarError(f"{key}: expert scales present without weights")
        if group.scales is not None and len(group.scales) != len(group.weights):
            raise MTPSidecarError(f"{key}: expert weight/scale counts differ")
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


def _expected_params(args: MTPArgs, module_formats: dict[str, dict]) -> dict[str, tuple]:
    """Parameter names and shapes of the draft model under the given formats.

    The model is built lazily (arrays are never evaluated), so enumerating the
    real-shape tree is cheap.
    """
    model = MTPDraftModel(args)
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
        raise MTPSidecarError(
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
        raise MTPSidecarError(f"{what}: shape mismatches; " + "; ".join(bad_shapes))


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

    raise MTPSidecarError(
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


def _load_dense_float(source_dir: Path, unit: _Unit) -> tuple[mx.array, str]:
    """Load a dense weight as a float mx array, dequantizing FP8 sources."""
    header = unit.weight
    if header.dtype == "F8_E4M3":
        if unit.scale is None:
            raise MTPSidecarError(f"{header.name}: FP8 weight without a scale tensor")
        from moespresso.probe.deepseek_v4.codec import dequant_fp8_e4m3_ue8m0

        w = dequant_fp8_e4m3_ue8m0(
            _read_storage(source_dir, header),
            _read_storage(source_dir, unit.scale),
            out_dtype=np.float32,
        )
        return mx.array(w).astype(mx.bfloat16), f"{header.dtype}+{unit.scale.dtype}"
    if header.dtype in _FLOAT_TOKENS:
        if unit.scale is not None:
            raise MTPSidecarError(f"{header.name}: float weight with a scale tensor")
        return _load_float_mx(source_dir, header), header.dtype
    raise MTPSidecarError(f"{header.name}: unsupported dense source dtype {header.dtype}")


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
                    raise MTPSidecarError(
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

    raise MTPSidecarError(f"unknown treatment {unit.treatment!r} for {base}")


def build_mtp_sidecar(source_dir: Path, output_dir: Path) -> dict:
    """Build the sidecar package. Returns the written manifest payload."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    config_path = source_dir / "config.json"
    if not config_path.exists():
        raise MTPSidecarError(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    args = _build_mtp_args(config)

    headers = _collect_source_headers(source_dir)
    plan = _resolve_plan(headers, args)

    # Pre-flight: the plan must cover the draft parameter tree exactly before
    # any heavy conversion work starts. Shapes are checked on the produced
    # tensors after conversion.
    _check_key_sets(
        _expected_params(args, {}),
        {name: () for name in _logical_names(plan)},
        "source resolution",
        check_shapes=False,
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise MTPSidecarError(f"output directory {output_dir} is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_rows: dict[str, dict] = {}
    module_formats: dict[str, dict] = {}
    produced: dict[str, tuple] = {}
    out_tensors: dict[str, mx.array] = {}

    def _take(tensors: dict, rows: dict, formats: dict) -> None:
        for name, arr in tensors.items():
            out_tensors[name] = arr
            produced[name] = tuple(arr.shape)
            row = dict(rows[name])
            row.update({
                "dtype": _mx_dtype_name(arr.dtype),
                "shape": [int(d) for d in arr.shape],
                "file": SHARD_NAME,
            })
            tensor_rows[name] = row
        module_formats.update(formats)

    for key in sorted(plan.expert_groups):
        tensors, rows, formats = _process_expert_group(
            source_dir, plan.expert_groups[key])
        _take(tensors, rows, formats)
    for base in sorted(plan.units):
        tensors, rows, formats = _process_unit(source_dir, base, plan.units[base])
        _take(tensors, rows, formats)

    if not out_tensors:
        raise MTPSidecarError("sidecar build produced no tensors")
    shard_path = output_dir / SHARD_NAME
    mx.save_safetensors(str(shard_path), out_tensors, metadata={"format": "mlx"})
    file_sha256 = {SHARD_NAME: _sha256_file(shard_path)}

    _check_key_sets(_expected_params(args, module_formats), produced, "sidecar output")

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
        },
        "mtp": _mtp_fields(config),
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
        description="Build a research MTP draft-model sidecar from a compatible "
                    "DeepSeek-V4-Flash HF snapshot.")
    parser.add_argument(
        "--source", required=True,
        help="DeepSeek-V4-Flash snapshot directory (config.json + shards)")
    parser.add_argument(
        "--output", required=True,
        help="sidecar directory: a path, or a bare name placed under the HF hub "
             "cache")
    args = parser.parse_args(argv)

    try:
        payload = build_mtp_sidecar(Path(args.source), resolve_output_dir(args.output))
    except MTPSidecarError as e:
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
