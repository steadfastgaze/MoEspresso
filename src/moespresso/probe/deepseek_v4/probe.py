"""DeepSeek-V4 probe unit adapters.

The generic probe builder assumes Qwen-style stacked experts and ordinary numeric
2D safetensors. DS4 source tensors need a family adapter: routed experts are
separate FP4 sources exposed through `DecodedExpertGroup`, and decoded FP8/BF16
affine samples arrive as already-materialized arrays from the streaming decoder.
This module emits the same `probe_evidence.units` shape the optimizer already
understands.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from moespresso.core.artifact import make_artifact
from moespresso.probe import roundtrip, weight_io
from moespresso.probe.build import (
    AFFINE_BITS,
    AFFINE_GROUP_SIZES,
    EXPERT_BITS,
    PRODUCER,
    imatrix_coverage_validations,
)
from moespresso.probe.deepseek_v4.codec import load_dequantized_fp8_rows
from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup


def _importance_vector(
    gguf_key: str | None,
    imatrix_vectors: dict[str, np.ndarray],
    in_features: int,
) -> tuple[np.ndarray, bool]:
    h = imatrix_vectors.get(gguf_key) if gguf_key else None
    if h is None or h.shape[0] != in_features:
        return np.ones(in_features, dtype=np.float32), False
    return h.astype(np.float32, copy=False), True


def _scalar_importance(h: np.ndarray, used_imatrix: bool, sample: np.ndarray) -> float:
    if used_imatrix:
        return float(np.mean(h))
    return float(np.linalg.norm(sample) / np.sqrt(sample.size)) if sample.size else 0.0


def _subsample_rows(sample: np.ndarray, sample_rows: int, seed: int) -> np.ndarray:
    if sample.shape[0] <= sample_rows:
        return sample
    idx = np.random.default_rng(seed).choice(sample.shape[0], sample_rows, replace=False)
    return sample[idx]


def probe_deepseek_v4_affine_sample(
    *,
    source_name: str,
    role: str,
    sample: np.ndarray,
    layer_index: int | None = None,
    shape: tuple[int, int] | None = None,
    gguf_key: str | None = None,
    imatrix_vectors: dict[str, np.ndarray] | None = None,
    source_codec: str | None = None,
    lossless_codecs: list[str] | None = None,
    sample_rows: int = 256,
    seed: int = 42,
) -> dict:
    """Probe one decoded DS4 non-expert dense sample."""
    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim != 2:
        raise ValueError("dense sample must be 2D")
    sample = _subsample_rows(sample, sample_rows, seed)
    in_features = sample.shape[1]
    h, used = _importance_vector(gguf_key, imatrix_vectors or {}, in_features)
    quality = {}
    for bits in AFFINE_BITS:
        for group_size in AFFINE_GROUP_SIZES:
            if in_features % group_size != 0:
                continue
            _, q = roundtrip.affine_quality(sample, bits, group_size, h)
            quality[f"{bits}_{group_size}"] = q
    dense_codec_quality = {}
    if in_features % 32 == 0:
        for mode, bits in (("mxfp4", 4), ("mxfp8", 8)):
            _, q = roundtrip.mx_float_quality(sample, mode, h)
            dense_codec_quality[f"{mode}_{bits}_32"] = q
    unit = {
        "source_name": source_name,
        "kind": "affine",
        "role": role,
        "layer_index": layer_index,
        "shape": list(shape or sample.shape),
        "importance": _scalar_importance(h, used, sample),
        "imatrix_mapped": used,
        "quality": quality,
    }
    if dense_codec_quality:
        unit["dense_codec_quality"] = dense_codec_quality
    if source_codec is not None:
        unit["source_codec"] = source_codec
    if lossless_codecs:
        unit["lossless_codecs"] = list(lossless_codecs)
    return unit


def probe_deepseek_v4_expert_group(
    group: DecodedExpertGroup,
    *,
    imatrix_vectors: dict[str, np.ndarray] | None = None,
    expert_sample: int = 2,
    sample_rows: int = 256,
    seed: int = 42,
) -> list[dict]:
    """Probe decoded DS4 FP4 experts as logical gate/up/down units."""
    vectors = imatrix_vectors or {}
    units: list[dict] = []
    rng = np.random.default_rng(seed)
    for layer in group.layers():
        all_experts = group.experts(layer)
        if not all_experts:
            continue
        selected = all_experts
        if len(all_experts) > expert_sample:
            selected = sorted(rng.choice(all_experts, expert_sample, replace=False).tolist())
        for projection in group.projections(layer):
            samples = []
            logical_shape = group.logical_shape(
                layer=layer,
                expert_index=selected[0],
                projection=projection,
            )
            for expert_index in selected:
                decoded = group.decode(
                    layer=layer,
                    expert_index=expert_index,
                    projection=projection,
                    out_dtype=np.float32,
                )
                samples.append(_subsample_rows(decoded, sample_rows, seed + expert_index))
            sample = np.concatenate(samples, axis=0)
            if sample.shape[0] > sample_rows:
                sample = _subsample_rows(sample, sample_rows, seed + layer)
            in_features = sample.shape[1]
            gguf_key = f"blk.{layer}.ffn_{projection}_exps.weight"
            h, used = _importance_vector(gguf_key, vectors, in_features)
            quality = {}
            for bits in EXPERT_BITS:
                _, q = roundtrip.tq_quality(sample, bits, h, seed)
                quality[str(bits)] = q
            units.append({
                "source_name": f"layers.{layer}.ffn.experts.{projection}",
                "kind": "expert",
                "role": f"moe.expert.{projection}",
                "layer_index": layer,
                "projection": projection,
                "n_experts": len(all_experts),
                "sampled": len(selected),
                "shape": list(logical_shape),
                "importance": _scalar_importance(h, used, sample),
                "imatrix_mapped": used,
                "source_codec": "fp4_e2m1_ue8m0",
                "lossless_codecs": ["mxfp4"],
                "quality": quality,
            })
    return units


def _scale_name_for_weight(source_name: str) -> str:
    if not source_name.endswith(".weight"):
        raise ValueError(f"DS4 FP8 affine source is not a weight tensor: {source_name}")
    return source_name[:-len(".weight")] + ".scale"


def _load_deepseek_v4_affine_sample(
    model_dir: Path,
    header,
    scale_header,
    *,
    rows: np.ndarray,
    fp8_block: tuple[int, int],
) -> np.ndarray:
    if header.dtype == "F8_E4M3":
        if scale_header is None:
            raise ValueError(f"missing FP8 scale tensor for {header.name}")
        return load_dequantized_fp8_rows(
            model_dir,
            header,
            scale_header,
            rows,
            fp8_block=fp8_block,
            out_dtype=np.float32,
        )
    if header.dtype in {"BF16", "F16", "F32", "FLOAT"}:
        return weight_io.load_2d_rows(model_dir, header, rows)
    raise ValueError(f"unsupported DeepSeek V4 affine dtype {header.dtype!r}")


def _dense_source_codec(header) -> tuple[str | None, list[str]]:
    if header.dtype == "F8_E4M3":
        return "fp8_e4m3_ue8m0", ["mxfp8"]
    if header.dtype == "BF16":
        return "bf16", []
    if header.dtype == "F16":
        return "fp16", []
    if header.dtype in {"F32", "FLOAT"}:
        return "fp32", []
    return None, []


def iter_deepseek_v4_affine_samples(
    inventory: dict,
    model_dir: Path,
    *,
    sample_rows: int = 256,
    seed: int = 42,
    fp8_block: tuple[int, int] = (128, 128),
) -> Iterable[dict]:
    """Yield bounded decoded DS4 affine samples from source safetensors."""
    model_dir = Path(model_dir)
    catalog = weight_io.scan_offsets(model_dir)
    entries = {
        entry["source_name"]: entry
        for entry in inventory.get("tensors", [])
    }
    for entry in inventory.get("tensors", []):
        if entry.get("status") != "required" or entry.get("kind") != "affine":
            continue
        header = catalog.get(entry["source_name"])
        if header is None:
            continue
        if len(header.shape) != 2:
            raise ValueError(f"{header.name} is not 2D: {header.shape}")
        rows = weight_io.sample_indices(header.shape[0], sample_rows, seed)
        scale_header = None
        if header.dtype == "F8_E4M3":
            scale_name = _scale_name_for_weight(entry["source_name"])
            scale_entry = entries.get(scale_name)
            if scale_entry is None or scale_entry.get("kind") != "codec_scale":
                raise ValueError(f"missing FP8 scale tensor for {entry['source_name']}")
            scale_header = catalog.get(scale_name)
            if scale_header is None:
                raise ValueError(f"missing FP8 scale bytes for {entry['source_name']}")
        sample = _load_deepseek_v4_affine_sample(
            model_dir,
            header,
            scale_header,
            rows=rows,
            fp8_block=fp8_block,
        )
        gguf_key = entry["gguf_keys"][0] if entry.get("gguf_keys") else None
        source_codec, lossless_codecs = _dense_source_codec(header)
        yield {
            "source_name": entry["source_name"],
            "role": entry["role"],
            "layer_index": entry.get("layer_index"),
            "shape": entry["shape"],
            "sample": sample,
            "gguf_key": gguf_key,
            "source_codec": source_codec,
            "lossless_codecs": lossless_codecs,
        }


def build_deepseek_v4_probe_evidence_from_inventory(
    inventory: dict,
    model_dir: Path,
    calibration: tuple[dict[str, np.ndarray], dict] | None = None,
    *,
    expert_sample: int = 2,
    sample_rows: int = 256,
    seed: int = 42,
    fp4_block: int = 32,
    fp8_block: tuple[int, int] = (128, 128),
) -> dict:
    """Build DS4 probe evidence by sampling real source tensors through inventory."""
    expert_group = None
    if any(e.get("kind") == "expert_source" for e in inventory.get("tensors", [])):
        expert_group = DecodedExpertGroup.from_inventory(
            inventory, model_dir, fp4_block=fp4_block)
    affine_samples = iter_deepseek_v4_affine_samples(
        inventory,
        model_dir,
        sample_rows=sample_rows,
        seed=seed,
        fp8_block=fp8_block,
    )
    return build_deepseek_v4_probe_evidence(
        inventory["subject"],
        expert_group=expert_group,
        affine_samples=affine_samples,
        calibration=calibration,
        expert_sample=expert_sample,
        sample_rows=sample_rows,
        seed=seed,
        source_inventory_id=inventory.get("artifact_id"),
    )


def build_deepseek_v4_probe_evidence(
    subject: dict,
    *,
    expert_group: DecodedExpertGroup | None = None,
    affine_samples: Iterable[dict] = (),
    calibration: tuple[dict[str, np.ndarray], dict] | None = None,
    expert_sample: int = 2,
    sample_rows: int = 256,
    seed: int = 42,
    source_inventory_id: str | None = None,
) -> dict:
    """Build a DS4 probe_evidence artifact from decoded streaming samples."""
    if calibration is not None:
        imatrix_vectors, calibration_identity = calibration
    else:
        imatrix_vectors, calibration_identity = {}, {"kind": "uniform"}

    units: list[dict] = []
    if expert_group is not None:
        units.extend(
            probe_deepseek_v4_expert_group(
                expert_group,
                imatrix_vectors=imatrix_vectors,
                expert_sample=expert_sample,
                sample_rows=sample_rows,
                seed=seed,
            )
        )
    for spec in affine_samples:
        units.append(
            probe_deepseek_v4_affine_sample(
                source_name=spec["source_name"],
                role=spec["role"],
                sample=spec["sample"],
                layer_index=spec.get("layer_index"),
                shape=tuple(spec["shape"]) if spec.get("shape") is not None else None,
                gguf_key=spec.get("gguf_key"),
                imatrix_vectors=imatrix_vectors,
                source_codec=spec.get("source_codec"),
                lossless_codecs=spec.get("lossless_codecs"),
                sample_rows=sample_rows,
                seed=seed,
            )
        )

    targets = {"expert": 0, "affine": 0}
    mapped = {"expert": 0, "affine": 0}
    for unit in units:
        if unit["kind"] in targets:
            targets[unit["kind"]] += 1
            mapped[unit["kind"]] += int(unit["imatrix_mapped"])

    validation = imatrix_coverage_validations(mapped, targets)

    required_features = (
        ["calibration"] if calibration_identity.get("kind") != "uniform" else []
    )
    return make_artifact(
        "probe_evidence",
        subject,
        PRODUCER,
        status="valid",
        validation=validation,
        required_features=required_features,
        units=units,
        calibration=calibration_identity,
        config={
            "family": "deepseek_v4_flash",
            "expert_bits": list(EXPERT_BITS),
            "affine_bits": list(AFFINE_BITS),
            "affine_group_sizes": list(AFFINE_GROUP_SIZES),
            "dense_mx_modes": ["mxfp4", "mxfp8"],
            "expert_sample": expert_sample,
            "sample_rows": sample_rows,
            "seed": seed,
            "fp16_measurement": True,
        },
        coverage={
            "expert_mapped": f"{mapped['expert']}/{targets['expert']}",
            "affine_mapped": f"{mapped['affine']}/{targets['affine']}",
        },
        source_inventory_id=source_inventory_id,
    )
