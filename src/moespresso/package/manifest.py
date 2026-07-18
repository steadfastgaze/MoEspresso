"""Build a `package_manifest` artifact: the package's self-description.

This is the phase that pays off the artifact-centered design: the runtime reads
the manifest and never guesses. The manifest declares, explicitly, the
architecture facts, every packed tensor's on-disk location + weight format +
transform, the expert layout, the ops the engine must support, and the file
identities (path/size/sha256) of every shard so a tampered/partial package fails
closed.

Pure: a function of the package_plan + the source architecture config + the
list of written-file identities. No mlx, no jang, no weight bytes, fully
testable anywhere (no model load). The actual TQ packing + safetensors writing
lives in package/write.py (the imperative shell with the MLX/JANG runtime).

`tq` transform is declared by versioned reference (`format: "tq", format_params:
{tq_version, bits, seed}`): the engine knows what tq_version 1 means (Hadamard
rotation with seed, per-row norms, bit-packing), but versioned so it cannot
silently drift. The `format_params` sub-object leaves room to declare the
transform structurally later without a major bump.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from moespresso.core.artifact import Validation, make_artifact
from moespresso.inventory.architecture_profile import deepseek_v4_flash_profile, family_of
from moespresso.package.kquant_format import KQUANT_GEOMETRY

PRODUCER = {"tool": "moespresso.package", "version": "1.0.0"}

# The package format. MJTQ = "MoEspresso Jang TurboQuant": it reuses jang's TurboQuant codec
# + tensor conventions (.tq_packed/.tq_norms/.tq_bits) for compression, but adds
# the strict, fully-explicit manifest layer on top: the runtime reads the
# manifest and never guesses. The upstream "jangtq" format is a different,
# third-party package format. Jang supplies the compression backend for mjtq;
# this manifest defines the package format.
PACKAGE_FORMAT = "mjtq"
PACKAGE_FORMAT_VERSION = 1

# What this format requires of the pipeline: its own strictness, declared here,
# not baked into the generic convert orchestrator. mjtq requires calibration: a
# mjtq package's probe evidence must be activation-weighted by a real imatrix
# (the spec's "calibration dataset identity" requirement; an uncalibrated mjtq is
# the red flag the spec names). Other package formats declare their own feature
# sets, and the convert pipeline consults those declarations instead of hardcoding
# requirements. Each entry must be in core.artifact.KNOWN_FEATURES.
PACKAGE_FORMAT_FEATURES = frozenset({"calibration"})

TQ_VERSION = 1
_PASSTHROUGH_FORMATS = frozenset({"fp16", "f32_passthrough", "raw_dtype_passthrough"})
_RAW_DTYPE_PASSTHROUGH_ROLES = frozenset({
    "attn.attn_sink",
    "attn.compressor.ape",
    "attn.indexer.compressor.ape",
    "moe.router_bias",
    "moe.router_tid2eid",
    "hc.control",
})

# A readable summary of the structural facts (for humans + the existing arch
# tests). The runtime does not build from this subset: it builds from the full
# `config` the manifest also carries (see _architecture). Kept so a glance at the
# manifest shows the shape without reading the whole config blob.
_ARCH_SUMMARY_FIELDS = (
    "num_hidden_layers", "hidden_size", "num_attention_heads",
    "num_key_value_heads", "num_experts", "num_experts_per_tok",
    "moe_intermediate_size", "layer_types", "vocab_size", "head_dim",
    "rms_norm_eps", "rope_theta",
)

_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")

# Tensors this package deliberately does not serve. The source may be a VL/MTP
# checkpoint; mjtq serves the text model only (for now), and declares it: a declared
# support scope, never a silent assumption (spec: Support Status Model). A future
# vision mjtq would declare modality="multimodal" and carry the vision config too.
_EXCLUDED_MODALITIES = ("vision", "mtp")


def file_identity(path: Path) -> dict:
    """{path (name), size_bytes, sha256} for a written shard (fail-closed identity)."""
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": path.name, "size_bytes": path.stat().st_size, "sha256": h.hexdigest()}


def _declared_expert_count(config: dict) -> int:
    for key in _EXPERT_COUNT_KEYS:
        value = config.get(key)
        if value is None:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return 0


def _clamp_expert_count(config: dict, max_experts: int) -> None:
    for key in _EXPERT_COUNT_KEYS:
        if key in config:
            config[key] = max_experts
    if "num_experts_per_tok" in config:
        config["num_experts_per_tok"] = min(
            int(config.get("num_experts_per_tok", 1)), max_experts)


def _architecture(arch_config: dict, max_experts: int | None = None) -> dict:
    """The architecture the runtime builds the graph from, fully self-contained.

    Carries the complete config (the text config), so the engine instantiates the
    model from the manifest alone and never reads the source config.json (spec:
    "runtime never performs source archaeology"). A trimmed field set fails here:
    it drops the linear-attn/SSM fields the graph needs, forcing the loader back to
    config.json. `family` selects the model class; `config` is
    everything that class's ModelArgs.from_dict needs.

    mjtq serves the text model: `text_config` is unwrapped (the source may be a VL
    checkpoint) and the modality + exclusions (vision/mtp) are declared explicitly.
    A future vision mjtq would declare a different modality.

    `max_experts` (smoke artifact): the package stored only N experts/layer, so the
    config's num_experts is clamped to N (and num_experts_per_tok to <= N): the
    served graph then has exactly the experts on disk. A reduced-expert smoke is a
    declared smaller model used for crash and coherence checks; it carries no quality claim.
    """
    is_wrapped = "text_config" in arch_config
    text = arch_config.get("text_config", arch_config)
    # The class to instantiate: the wrapper family (qwen3_5_moe) when the source is
    # wrapped: mlx_lm's wrapper builds the text model from text_config and handles
    # the model.language_model.* key nesting itself; else the model_type directly.
    raw_family = (arch_config.get("model_type") if is_wrapped
                  else arch_config.get("model_type") or text.get("model_type") or "unknown")
    resolved_family = family_of(arch_config)
    # Keep the established MoE manifest family stable for existing packages/tests.
    if resolved_family in {"qwen3_5_dense", "deepseek_v4_flash"}:
        family = resolved_family
    else:
        family = raw_family
    config = dict(text)
    smoke = max_experts is not None and max_experts < _declared_expert_count(config)
    if smoke:
        _clamp_expert_count(config, max_experts)
    summary = {k: config[k] for k in _ARCH_SUMMARY_FIELDS if k in config}
    architecture = {
        **summary,                       # readable shape (back-compat with arch tests)
        "family": family,
        "modality": "text",              # this package serves text only (declared)
        "excludes": ["mtp"] if resolved_family == "deepseek_v4_flash"
        else list(_EXCLUDED_MODALITIES),
        "smoke_max_experts": max_experts if smoke else None,
        "source_nesting": "" if resolved_family == "deepseek_v4_flash" else "model.language_model.",
        # The full config the runtime builds from: no source file needed at load.
        "config": config,
    }
    if is_wrapped:
        architecture["wrapper_model_type"] = arch_config.get("model_type")
        architecture["text_model_type"] = text.get("model_type")
    if resolved_family == "deepseek_v4_flash":
        profile = deepseek_v4_flash_profile()
        architecture.update({
            "compress_ratios": profile["compress_ratios"],
            "layer_kinds": profile["layer_kinds"],
            "rope_base_by_layer": profile["rope_base_by_layer"],
            "yarn_by_layer": profile["yarn_by_layer"],
            "yarn": profile["yarn"],
            "attention": profile["attention"],
            "hyper_connections": profile["hyper_connections"],
            "router": profile["router"],
            "cache_policy": profile["cache_policy"],
            "tokenizer": profile["tokenizer"],
            "prompt_renderer": profile["tokenizer"]["renderer"],
            "expert_source_layout": profile["router"]["expert_source_layout"],
        })
    return architecture


def located_key(alloc: dict) -> str:
    """Key for the `located` map. Experts share a source_name (fused gate_up), so
    they need a projection-qualified key; affine/fp16 use the bare source_name.
    The single source of this rule: the writer and manifest both call it."""
    if alloc["kind"] == "expert":
        return f"{alloc['source_name']}::{alloc['projection']}"
    return alloc["source_name"]


def _tensor_entry(alloc: dict, located: dict, seed: int) -> dict:
    """One allocation entry + its on-disk location -> a manifest tensor entry."""
    kind = alloc["kind"]
    entry = {
        "source_name": alloc["source_name"],
        "role": alloc["role"],
        "kind": kind,
        "shard": located["shard"],
        "key_prefix": located["key_prefix"],
    }
    if alloc.get("forced_format") is not None:
        entry["format_decision"] = {"forced": dict(alloc["forced_format"])}
    if kind == "expert":
        fmt = alloc.get("format") or alloc.get("codec") or "tq"
        entry["layer_index"] = alloc["layer_index"]
        entry["projection"] = alloc["projection"]
        if fmt == "tq":
            entry["format"] = "tq"
            entry["format_params"] = {
                "tq_version": TQ_VERSION,
                "bits": alloc["bits"],
                "seed": seed,
            }
        elif fmt == "mxfp4":
            entry["format"] = "mxfp4"
            entry["format_params"] = {
                "bits": 4,
                "group_size": 32,
                "scale_dtype": "ue8m0",
                "source_codec": alloc.get("source_codec", "fp4_e2m1_ue8m0"),
                "lossless": bool(alloc.get("lossless", True)),
            }
        elif fmt == "kquant":
            kcodec = alloc.get("kquant_codec") or alloc.get("codec")
            geometry = KQUANT_GEOMETRY.get(kcodec)
            entry["format"] = "kquant"
            entry["format_params"] = {
                "kquant_codec": kcodec,
            }
            if alloc.get("module_weight_key") is not None:
                entry["module_weight_key"] = alloc["module_weight_key"]
            if alloc.get("module_path") is not None:
                entry["module_path"] = alloc["module_path"]
            if geometry is not None:
                entry["format_params"].update({
                    "bits": geometry.bits,
                    "group_size": geometry.group_size,
                    "bytes_per_block": geometry.bytes_per_block,
                    "weights_per_block": geometry.weights_per_block,
                })
            if alloc.get("imatrix_key") is not None:
                entry["format_params"]["imatrix_key"] = alloc["imatrix_key"]
        else:
            entry["format"] = fmt
            entry["format_params"] = {}
    elif kind == "affine":
        fmt = alloc.get("format", "affine")
        entry["format"] = fmt
        if fmt == "affine":
            entry["format_params"] = {
                "bits": alloc["bits"],
                "group_size": alloc["group_size"],
            }
        elif fmt in {"mxfp4", "mxfp8"}:
            entry["format_params"] = {
                "bits": 4 if fmt == "mxfp4" else 8,
                "group_size": 32,
                "scale_dtype": "ue8m0",
                "source_codec": alloc.get("source_codec"),
                "lossless": bool(alloc.get("lossless", False)),
            }
        elif fmt == "kquant":
            kcodec = alloc.get("kquant_codec") or alloc.get("codec")
            geometry = KQUANT_GEOMETRY.get(kcodec)
            entry["format_params"] = {"kquant_codec": kcodec}
            if alloc.get("module_weight_key") is not None:
                entry["module_weight_key"] = alloc["module_weight_key"]
            if alloc.get("module_path") is not None:
                entry["module_path"] = alloc["module_path"]
            if geometry is not None:
                entry["format_params"].update({
                    "bits": geometry.bits,
                    "group_size": geometry.group_size,
                    "bytes_per_block": geometry.bytes_per_block,
                    "weights_per_block": geometry.weights_per_block,
                })
            if alloc.get("imatrix_key") is not None:
                entry["format_params"]["imatrix_key"] = alloc["imatrix_key"]
        else:
            entry["format_params"] = {}
    elif kind == "fp16_passthrough":
        entry["format"] = "fp16"
        entry["format_params"] = {}
    elif kind == "raw_dtype_passthrough":
        entry["format"] = "raw_dtype_passthrough"
        entry["format_params"] = {}
    else:  # pragma: no cover - guarded by validation below
        entry["format"] = "unknown"
        entry["format_params"] = {}
    return entry


def _passthrough_entry(pt: dict, loc: dict) -> dict:
    """A structural passthrough tensor's manifest entry."""
    fmt = pt.get("format", "fp16")
    return {
        "source_name": pt["source_name"],
        "role": pt["role"],
        "kind": "passthrough",
        "layer_index": pt.get("layer_index"),
        "shard": loc["shard"],
        "key_prefix": loc["key_prefix"],
        "format": fmt,
        "format_params": {},
    }


def build_package_manifest(
    package_plan: dict,
    arch_config: dict,
    located: dict[str, dict],
    files: list[dict],
    *,
    seed: int = 42,
    expert_layout: dict | None = None,
    passthrough: list[dict] | None = None,
    passthrough_located: dict[str, dict] | None = None,
    tokenizer: dict | None = None,
    agentic_profile: dict | None = None,
    max_experts: int | None = None,
) -> dict:
    """Assemble a package_manifest artifact (pure).

    `located` maps each allocation's source_name -> {shard, key_prefix} (where the
    converter wrote it). `files` is a list of file_identity dicts for every shard.
    `expert_layout` declares the stacked-expert sharding/keying convention.
    `passthrough` is the inventory's structural tensors (norms, SSM state) copied
    verbatim; `passthrough_located` maps each to where it was written. They flow
    directly from the inventory, preserving optimizer purity.
    `agentic_profile` is the identity block of the agentic profile sidecar
    (package/agentic_profile.py); families without one omit the key.
    """
    if package_plan.get("artifact_kind") != "package_plan":
        raise ValueError("build_package_manifest requires a package_plan artifact")
    arch = _architecture(arch_config, max_experts=max_experts)
    tensors, validation = [], []
    file_names = {f["path"] for f in files}

    for alloc in package_plan.get("allocation", []):
        name = alloc["source_name"]
        loc = located.get(located_key(alloc))
        if loc is None:
            validation.append(Validation(
                "error", "package.unwritten_tensor",
                f"allocation references {name} but no written location was provided",
                path=f"/{name}", phase="package", blocking=True))
            continue
        if loc["shard"] not in file_names:
            validation.append(Validation(
                "error", "package.missing_shard",
                f"{name} -> shard {loc['shard']} not in written files",
                path=f"/{name}", phase="package", blocking=True))
            continue
        entry = _tensor_entry(alloc, loc, seed)
        if entry["kind"] == "expert" and entry["format"] not in {"tq", "mxfp4", "kquant"}:
            validation.append(Validation(
                "error", "package.unsupported_expert_format",
                f"{name} declares expert format {entry['format']!r}; routed experts "
                "support only TQ, source-mxfp4, or K-quant",
                path=f"/{name}", phase="package", blocking=True,
                expected=["tq", "mxfp4", "kquant"], actual=entry["format"]))
        if entry["format"] == "kquant":
            kcodec = entry["format_params"].get("kquant_codec")
            if kcodec not in KQUANT_GEOMETRY:
                validation.append(Validation(
                    "error", "package.unsupported_kquant_codec",
                    f"{name} declares unsupported K-quant codec {kcodec!r}",
                    path=f"/{name}", phase="package", blocking=True,
                    expected=sorted(KQUANT_GEOMETRY), actual=kcodec))
            if not isinstance(entry.get("module_weight_key"), str):
                validation.append(Validation(
                    "error", "package.missing_kquant_module_weight_key",
                    f"{name} declares K-quant but no module_weight_key for the "
                    "mlx-kquant installer",
                    path=f"/{name}", phase="package", blocking=True))
        if entry["kind"] == "affine" and entry["format"] not in {
            "affine", "mxfp4", "mxfp8", "kquant",
        }:
            validation.append(Validation(
                "error", "package.unsupported_dense_format",
                f"{name} declares dense format {entry['format']!r}; dense tensors "
                "support affine, mxfp4, mxfp8, or K-quant",
                path=f"/{name}", phase="package", blocking=True,
                expected=["affine", "mxfp4", "mxfp8", "kquant"],
                actual=entry["format"]))
        tensors.append(entry)

    pt_located = passthrough_located or {}
    for pt in passthrough or []:
        name = pt["source_name"]
        fmt = pt.get("format", "fp16")
        if fmt not in _PASSTHROUGH_FORMATS:
            validation.append(Validation(
                "error", "package.unsupported_passthrough_format",
                f"passthrough {name} declares unsupported format {fmt!r}",
                path=f"/{name}", phase="package", blocking=True))
            continue
        if pt.get("role") in _RAW_DTYPE_PASSTHROUGH_ROLES and fmt != "raw_dtype_passthrough":
            validation.append(Validation(
                "error", "package.control_tensor_downcast",
                f"passthrough {name} role {pt.get('role')!r} requires raw dtype storage",
                path=f"/{name}", phase="package", blocking=True))
            continue
        loc = pt_located.get(name)
        if loc is None or loc["shard"] not in file_names:
            validation.append(Validation(
                "error", "package.unwritten_tensor",
                f"passthrough {name} has no written location",
                path=f"/{name}", phase="package", blocking=True))
            continue
        tensors.append(_passthrough_entry(pt, loc))

    if not package_plan.get("allocation"):
        validation.append(Validation(
            "error", "package.empty_plan",
            "package plan has no allocation, nothing to package",
            phase="package", blocking=True))

    format_ops = {
        "tq": "tq_dequant",
        "mxfp4": "mxfp4_dequant",
        "kquant": "kquant_dequant",
        "mxfp8": "mxfp8_dequant",
        "affine": "affine_dequant",
        "fp16": "fp16_passthrough",
        "f32_passthrough": "f32_passthrough",
        "raw_dtype_passthrough": "raw_dtype_passthrough",
    }
    required_ops_set = set()
    for t in tensors:
        op = format_ops.get(t["format"])
        if op is None:
            validation.append(Validation(
                "error", "package.unsupported_tensor_format",
                f"{t['source_name']} declares unsupported tensor format {t['format']!r}",
                path=f"/{t['source_name']}", phase="package", blocking=True))
            continue
        required_ops_set.add(op)
    required_ops = sorted(required_ops_set)

    status = "valid" if not any(v.blocking for v in validation) else "invalid"
    manifest_fields = {
        "architecture": arch,
        "tensors": sorted(tensors, key=lambda t: (
            t.get("layer_index") if t.get("layer_index") is not None else -1,
            t.get("projection", ""), t["source_name"])),
        "required_ops": required_ops,
        "files": sorted(files, key=lambda f: f["path"]),
        "tokenizer": tokenizer or {"files": [], "rendering_id": None, "has_tokenizer": False},
        "optimized_kernels_expected": bool(
            package_plan.get("optimized_kernels_expected", False)),
        "provenance": {
            "source_plan_id": package_plan.get("artifact_id"),
            "source_decision_id": package_plan.get("source_decision_id"),
            "source_probe_id": package_plan.get("source_probe_id"),
            "package_plan": {
                "producer_kind": package_plan.get("producer_kind"),
                "producer_reference": package_plan.get("producer_reference"),
                "optimized_kernels_expected": bool(
                    package_plan.get("optimized_kernels_expected", False)),
                "force_overrides": list(package_plan.get("force_overrides", [])),
            },
        },
    }
    if agentic_profile is not None:
        manifest_fields["agentic_profile"] = agentic_profile
    diagnostic = (package_plan.get("source_constraints") or {}).get("diagnostic")
    if diagnostic is not None:
        manifest_fields["provenance"]["diagnostic"] = diagnostic
    if (
        any(
            t["format"] in {"tq", "mxfp4", "kquant"} and t.get("kind") == "expert"
            for t in tensors
        )
        or expert_layout is not None
    ):
        manifest_fields["expert_layout"] = expert_layout or _DEFAULT_EXPERT_LAYOUT

    return make_artifact(
        "package_manifest", package_plan["subject"], PRODUCER,
        required_features=list(package_plan.get("required_features", [])),
        status=status, validation=validation,
        package_format=PACKAGE_FORMAT,
        package_format_version=PACKAGE_FORMAT_VERSION,
        **manifest_fields,
    )


# How routed experts are sharded and keyed on disk (the bundle convention).
# One uint8 bundle tensor per layer (`...switch_mlp.experts.tq_bundle`,
# [n_experts, row_bytes]) whose row e concatenates expert e's full payload in
# row_order; the exact per-component geometry travels in each shard's
# safetensors __metadata__ (package/bundle.py is the schema's source of truth).
# Older stacked packages (tq_packed/tq_norms/tq_bits) are not readable: the
# runtime fails loud with a re-convert message (no compatibility path).
_DEFAULT_EXPERT_LAYOUT = {
    "stacked": False,
    "bundled": True,
    "fused_gate_up": True,        # gate_up_proj splits into gate + up sub-projections
    "shard_per_layer": False,     # bundles live in the model-*-of-* shards
    "key_suffixes": ["tq_bundle"],
    "row_order": [
        ["gate_proj", "packed"], ["gate_proj", "norms"],
        ["up_proj", "packed"], ["up_proj", "norms"],
        ["down_proj", "packed"], ["down_proj", "norms"],
    ],
}
