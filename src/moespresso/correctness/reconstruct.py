"""L1 tensor reconstruction evidence.

No model graph, no full package serve. This module samples source/package tensors and
checks that the stored package data reconstructs according to the manifest and the family
profile. Heavy references (MLX) are imported lazily here, so L0/L0b stay lightweight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moespresso.core.artifact import Validation, make_artifact
from moespresso.correctness.tq_reference import tq_decode_rows
from moespresso.inventory.safetensors_header import read_shard_metadata
from moespresso.package import bundle as bundle_mod
from moespresso.probe import weight_io
from moespresso.probe.deepseek_v4.codec import load_dequantized_fp8_rows
from moespresso.probe.deepseek_v4.experts import DecodedExpertGroup

PRODUCER = {"tool": "moespresso.correctness", "version": "1.1.0"}

DEFAULT_SAMPLE_POLICY = {
    "seed": 42,
    "affine_tensors": 8,
    "rows_per_tensor": 32,
    "tq_tensors": 24,
    "tq_experts": 2,
    "rows_per_expert": 8,
}

DEFAULT_TOLERANCES = {
    "passthrough_max_abs": 2e-3,
    "affine_relative_rms": 1.25,
    "tq_relative_rms": 1.25,
}

_HIGH_RISK = (
    "lm_head", "embed_tokens", "in_proj_a", "in_proj_b", "conv1d", "norm", "gate_up"
)


def _policy(overrides: dict | None) -> dict:
    return {**DEFAULT_SAMPLE_POLICY, **(overrides or {})}


def _stable_seed(seed: int, *parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    return seed + (sum(ord(c) for c in text) % 100_000)


def _priority(t: dict) -> tuple[int, str, str]:
    text = " ".join(str(t.get(k, "")) for k in ("source_name", "role", "projection"))
    return (0 if any(x in text for x in _HIGH_RISK) else 1,
            t.get("source_name", ""), t.get("projection", ""))


def _limit(entries: list[dict], n: int) -> list[dict]:
    return sorted(entries, key=_priority)[:max(0, n)]


def _sidecar_keys(t: dict, suffixes: tuple[str, ...]) -> list[str]:
    prefix = t["key_prefix"]
    return [f"{prefix}.{suffix}" for suffix in suffixes]


def _missing(out: list[Validation], key: str, t: dict) -> None:
    out.append(Validation(
        "error", "correctness.missing_package_tensor",
        f"{key} is required by manifest entry {t['source_name']} but is missing on disk",
        path=f"/{key}", phase="L1", blocking=True))


def _source_missing(out: list[Validation], name: str) -> None:
    out.append(Validation(
        "error", "correctness.missing_source_tensor",
        f"{name} is referenced by the package manifest but missing from source tensors",
        path=f"/{name}", phase="L1", blocking=True))


def _errors(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    diff = actual - expected
    rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    src_rms = float(np.sqrt(np.mean(expected * expected))) if expected.size else 0.0
    rel = rms / max(src_rms, 1e-12)
    return {
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms": rms,
        "relative_rms": float(rel),
        "nonfinite": int(np.size(actual) - np.isfinite(actual).sum()),
    }


def _record_metric(metrics: list[dict], t: dict, fmt: str, err: dict,
                   rows: list[int] | None = None, experts: list[int] | None = None) -> None:
    metrics.append({
        "source_name": t["source_name"],
        "projection": t.get("projection"),
        "format": fmt,
        "rows": rows or [],
        "experts": experts or [],
        **err,
    })


def _block_on_error(out: list[Validation], t: dict, err: dict, tolerance: float,
                    metric: str = "relative_rms") -> None:
    if err["nonfinite"]:
        out.append(Validation(
            "error", "correctness.nonfinite_reconstruction",
            f"{t['source_name']} reconstructed with {err['nonfinite']} non-finite values",
            path=f"/{t['source_name']}", phase="L1", blocking=True,
            expected=0, actual=err["nonfinite"]))
        return
    if err[metric] > tolerance:
        out.append(Validation(
            "error", "correctness.reconstruction_error",
            f"{t['source_name']} reconstruction {metric}={err[metric]:.6g} exceeds "
            f"tolerance {tolerance:.6g}",
            path=f"/{t['source_name']}", phase="L1", blocking=True,
            expected=f"<={tolerance}", actual=err[metric]))


def _smoke_max_experts(manifest: dict) -> int | None:
    value = (manifest.get("architecture") or {}).get("smoke_max_experts")
    if value is None:
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _router_gate_source_slice(t, src, pkg, smoke_max_experts, out, name):
    if t.get("role") != "moe.router_gate" or smoke_max_experts is None:
        return src
    if not (src.ndim == pkg.ndim == 2 and src.shape[1:] == pkg.shape[1:]):
        return src
    expected_shape = (smoke_max_experts, *src.shape[1:])
    if pkg.shape != expected_shape:
        out.append(Validation(
            "error", "correctness.shape_mismatch",
            f"{name} smoke router gate stored shape {pkg.shape} does not match "
            f"declared smoke expert count {smoke_max_experts}",
            path=f"/{name}", phase="L1", blocking=True,
            expected=list(expected_shape), actual=list(pkg.shape)))
        return None
    if src.shape[0] < smoke_max_experts:
        out.append(Validation(
            "error", "correctness.shape_mismatch",
            f"{name} source router gate has {src.shape[0]} rows, fewer than declared "
            f"smoke expert count {smoke_max_experts}",
            path=f"/{name}", phase="L1", blocking=True,
            expected=f">={smoke_max_experts}", actual=src.shape[0]))
        return None
    return src[:smoke_max_experts]


def _check_passthrough(
    t,
    src_cat,
    pkg_cat,
    source_dir,
    package_dir,
    smoke_max_experts,
    out,
    metrics,
    counts,
):
    name = t["source_name"]
    src_h = src_cat.get(name)
    pkg_h = pkg_cat.get(t["key_prefix"])
    if src_h is None:
        _source_missing(out, name)
        return
    if pkg_h is None:
        _missing(out, t["key_prefix"], t)
        return
    src = weight_io.load_full(source_dir, src_h)
    pkg = weight_io.load_full(package_dir, pkg_h)
    src = _router_gate_source_slice(t, src, pkg, smoke_max_experts, out, name)
    if src is None:
        return
    if src.shape != pkg.shape:
        out.append(Validation(
            "error", "correctness.shape_mismatch",
            f"{name} stored shape {pkg.shape} does not match source shape {src.shape}",
            path=f"/{name}", phase="L1", blocking=True,
            expected=list(src.shape), actual=list(pkg.shape)))
        return
    err = _errors(src.astype(np.float16).astype(np.float32), pkg)
    _record_metric(metrics, t, "fp16", err)
    _block_on_error(out, t, err, DEFAULT_TOLERANCES["passthrough_max_abs"], "max_abs")
    counts["fp16"] += 1


def _check_f32_passthrough(
    t,
    src_cat,
    pkg_cat,
    source_dir,
    package_dir,
    out,
    metrics,
    counts,
):
    name = t["source_name"]
    src_h = src_cat.get(name)
    pkg_h = pkg_cat.get(t["key_prefix"])
    if src_h is None:
        _source_missing(out, name)
        return
    if pkg_h is None:
        _missing(out, t["key_prefix"], t)
        return
    if pkg_h.dtype != "F32":
        out.append(Validation(
            "error", "correctness.dtype_mismatch",
            f"{name} stored dtype {pkg_h.dtype} does not match declared F32 passthrough",
            path=f"/{name}", phase="L1", blocking=True,
            expected="F32", actual=pkg_h.dtype))
        return
    src = weight_io.load_full(source_dir, src_h).astype(np.float32)
    pkg = weight_io.load_full(package_dir, pkg_h)
    if src.shape != pkg.shape:
        out.append(Validation(
            "error", "correctness.shape_mismatch",
            f"{name} stored shape {pkg.shape} does not match source shape {src.shape}",
            path=f"/{name}", phase="L1", blocking=True,
            expected=list(src.shape), actual=list(pkg.shape)))
        return
    err = _errors(src, pkg)
    _record_metric(metrics, t, "f32_passthrough", err)
    _block_on_error(out, t, err, DEFAULT_TOLERANCES["passthrough_max_abs"], "max_abs")
    counts["f32_passthrough"] += 1


def _check_raw_passthrough(t, src_cat, pkg_cat, source_dir, package_dir, out, metrics, counts):
    name = t["source_name"]
    src_h = src_cat.get(name)
    pkg_h = pkg_cat.get(t["key_prefix"])
    if src_h is None:
        _source_missing(out, name)
        return
    if pkg_h is None:
        _missing(out, t["key_prefix"], t)
        return
    src = weight_io.load_full_raw(source_dir, src_h)
    pkg = weight_io.load_full_raw(package_dir, pkg_h)
    if src.shape != pkg.shape:
        out.append(Validation(
            "error", "correctness.shape_mismatch",
            f"{name} stored shape {pkg.shape} does not match source shape {src.shape}",
            path=f"/{name}", phase="L1", blocking=True,
            expected=list(src.shape), actual=list(pkg.shape)))
        return
    if src.dtype != pkg.dtype:
        out.append(Validation(
            "error", "correctness.dtype_mismatch",
            f"{name} stored dtype {pkg.dtype} does not match source dtype {src.dtype}",
            path=f"/{name}", phase="L1", blocking=True,
            expected=str(src.dtype), actual=str(pkg.dtype)))
        return
    err = _errors(src, pkg)
    _record_metric(metrics, t, "raw_dtype_passthrough", err)
    if not np.array_equal(src, pkg):
        out.append(Validation(
            "error", "correctness.raw_passthrough_mismatch",
            f"{name} raw passthrough storage differs from source storage",
            path=f"/{name}", phase="L1", blocking=True))
    counts["raw_dtype_passthrough"] += 1


def _mx_dequant(qw: np.ndarray, scales: np.ndarray, biases: np.ndarray | None,
                bits: int, group_size: int, mode: str = "affine") -> np.ndarray:
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError("mlx is required for affine reconstruction") from e
    if mode == "affine":
        w = mx.dequantize(mx.array(qw), mx.array(scales), mx.array(biases),
                          bits=bits, group_size=group_size, mode=mode)
    else:
        w = mx.dequantize(mx.array(qw), mx.array(scales), mode=mode).astype(mx.float32)
    mx.eval(w)
    out = np.asarray(w, dtype=np.float32)
    del w
    mx.clear_cache()
    return out


def _scale_name_for_weight(name: str) -> str:
    if not name.endswith(".weight"):
        return f"{name}.scale"
    return name[:-len(".weight")] + ".scale"


def _load_affine_source_rows(t, src_cat, source_dir, rows, out):
    name = t["source_name"]
    src_h = src_cat.get(name)
    if src_h is None:
        _source_missing(out, name)
        return None
    if src_h.dtype == "F8_E4M3":
        scale_name = _scale_name_for_weight(name)
        scale_h = src_cat.get(scale_name)
        if scale_h is None:
            out.append(Validation(
                "error", "correctness.missing_source_tensor",
                f"{name} is FP8 but matching scale tensor {scale_name} is missing",
                path=f"/{name}", phase="L1", blocking=True))
            return None
        return load_dequantized_fp8_rows(
            source_dir,
            src_h,
            scale_h,
            rows,
            out_dtype=np.float32,
        )
    return weight_io.load_2d_rows(source_dir, src_h, rows)


def _check_affine(t, src_cat, pkg_cat, source_dir, package_dir, policy, out, metrics, counts):
    name = t["source_name"]
    src_h = src_cat.get(name)
    if src_h is None:
        _source_missing(out, name)
        return
    keys = _sidecar_keys(t, ("weight", "scales", "biases"))
    headers = [pkg_cat.get(k) for k in keys]
    for key, header in zip(keys, headers, strict=True):
        if header is None:
            _missing(out, key, t)
            return
    bits = int(t.get("format_params", {}).get("bits", 0))
    group_size = int(t.get("format_params", {}).get("group_size", 0))
    if bits <= 0 or group_size <= 0:
        out.append(Validation(
            "error", "correctness.bad_format_params",
            f"{name} affine entry has invalid format_params {t.get('format_params')!r}",
            path=f"/{name}", phase="L1", blocking=True))
        return
    rows = weight_io.sample_indices(src_h.shape[0], int(policy["rows_per_tensor"]),
                                    _stable_seed(policy["seed"], name, "affine"))
    source = _load_affine_source_rows(t, src_cat, source_dir, rows, out)
    if source is None:
        return
    qw = weight_io.load_2d_rows_raw(package_dir, headers[0], rows)
    scales = weight_io.load_2d_rows_raw(package_dir, headers[1], rows)
    biases = weight_io.load_2d_rows_raw(package_dir, headers[2], rows)
    try:
        recon = _mx_dequant(qw, scales, biases, bits, group_size)
    except RuntimeError as e:
        out.append(Validation(
            "error", "correctness.reference_unavailable", str(e),
            path=f"/{name}", phase="L1", blocking=True))
        return
    err = _errors(source, recon)
    _record_metric(metrics, t, "affine", err, rows=rows.tolist())
    _block_on_error(out, t, err, DEFAULT_TOLERANCES["affine_relative_rms"])
    counts["affine"] += 1


def _check_mx_dense(t, src_cat, pkg_cat, source_dir, package_dir, policy, out, metrics, counts):
    name = t["source_name"]
    src_h = src_cat.get(name)
    if src_h is None:
        _source_missing(out, name)
        return
    keys = _sidecar_keys(t, ("weight", "scales"))
    headers = [pkg_cat.get(k) for k in keys]
    for key, header in zip(keys, headers, strict=True):
        if header is None:
            _missing(out, key, t)
            return
    fmt = t.get("format")
    bits = int(t.get("format_params", {}).get("bits", 0))
    group_size = int(t.get("format_params", {}).get("group_size", 0))
    if fmt not in {"mxfp4", "mxfp8"} or bits <= 0 or group_size != 32:
        out.append(Validation(
            "error", "correctness.bad_format_params",
            f"{name} MX dense entry has invalid format_params {t.get('format_params')!r}",
            path=f"/{name}", phase="L1", blocking=True))
        return
    rows = weight_io.sample_indices(src_h.shape[0], int(policy["rows_per_tensor"]),
                                    _stable_seed(policy["seed"], name, fmt))
    source = _load_affine_source_rows(t, src_cat, source_dir, rows, out)
    if source is None:
        return
    qw = weight_io.load_2d_rows_raw(package_dir, headers[0], rows)
    scales = weight_io.load_2d_rows_raw(package_dir, headers[1], rows)
    try:
        recon = _mx_dequant(qw, scales, None, bits, group_size, mode=fmt)
    except RuntimeError as e:
        out.append(Validation(
            "error", "correctness.reference_unavailable", str(e),
            path=f"/{name}", phase="L1", blocking=True))
        return
    err = _errors(source, recon)
    _record_metric(metrics, t, fmt, err, rows=rows.tolist())
    _block_on_error(out, t, err, DEFAULT_TOLERANCES["affine_relative_rms"])
    counts[fmt] += 1


def _source_rows_for_projection(source_rows: int, projection: str, sampled_rows: np.ndarray):
    if projection in ("gate", "up"):
        mid = source_rows // 2
        if projection == "gate":
            return sampled_rows, mid
        return sampled_rows + mid, mid
    return sampled_rows, source_rows


def _bundle_projection_geometry(t, bundle_h, package_dir, out):
    """The (layer, projection) geometry from the bundle shard's metadata.

    Returns the projection's geometry dict or None (with a blocking validation
    appended). This is the same metadata the runtime's expert index loads from,
    so L1 verifies the exact contract the loader will follow.
    """
    key = f"/{t['key_prefix']}.{bundle_mod.BUNDLE_KEY_SUFFIX}"
    meta_text = read_shard_metadata(
        Path(package_dir) / bundle_h.shard).get(bundle_mod.METADATA_KEY)
    if meta_text is None:
        out.append(Validation(
            "error", "correctness.missing_bundle_metadata",
            f"shard {bundle_h.shard} carries {bundle_h.name} but no "
            f"{bundle_mod.METADATA_KEY} metadata",
            path=key, phase="L1", blocking=True))
        return None
    try:
        layers = bundle_mod.decode_bundle_metadata(meta_text)
    except bundle_mod.BundleFormatError as e:
        out.append(Validation(
            "error", "correctness.bad_bundle_metadata", str(e),
            path=key, phase="L1", blocking=True))
        return None
    layer = t.get("layer_index")
    geo = layers.get(int(layer)) if layer is not None else None
    if geo is None:
        out.append(Validation(
            "error", "correctness.missing_bundle_metadata",
            f"{bundle_mod.METADATA_KEY} metadata has no layer-{layer} entry "
            f"for {t['source_name']}",
            path=key, phase="L1", blocking=True))
        return None
    proj = t.get("projection", "")
    proj_key = proj if proj.endswith("_proj") else f"{proj}_proj"
    return geo["projections"].get(proj_key)


def _check_tq(t, src_cat, pkg_cat, source_dir, package_dir, policy, out, metrics, counts):
    name = t["source_name"]
    src_h = src_cat.get(name)
    if src_h is None:
        _source_missing(out, name)
        return
    (bundle_key,) = _sidecar_keys(t, (bundle_mod.BUNDLE_KEY_SUFFIX,))
    bundle_h = pkg_cat.get(bundle_key)
    if bundle_h is None:
        _missing(out, bundle_key, t)
        return
    pgeo = _bundle_projection_geometry(t, bundle_h, package_dir, out)
    if pgeo is None:
        return
    params = t.get("format_params", {})
    expected_bits = int(params.get("bits", 0))
    seed = int(params.get("seed", policy["seed"]))
    stored_bits = int(pgeo["bits"])
    if stored_bits != expected_bits:
        out.append(Validation(
            "error", "correctness.tq_bits_mismatch",
            f"{name} manifest bits {expected_bits} but bundle metadata stores "
            f"{stored_bits}",
            path=f"/{bundle_key}", phase="L1", blocking=True,
            expected=expected_bits, actual=stored_bits))
        return
    n_experts, source_rows, in_features = src_h.shape
    expert_idx = weight_io.sample_indices(n_experts, int(policy["tq_experts"]),
                                          _stable_seed(policy["seed"], name, "experts"))
    projection = t.get("projection", "")
    row_limit = int(pgeo["packed"]["shape"][0])  # out_features in the package
    row_idx = weight_io.sample_indices(row_limit, int(policy["rows_per_expert"]),
                                       _stable_seed(policy["seed"], name, projection, "rows"))
    source_row_idx, out_features = _source_rows_for_projection(source_rows, projection, row_idx)
    if out_features != row_limit:
        out.append(Validation(
            "error", "correctness.shape_mismatch",
            f"{name} projection {projection!r} package rows {row_limit} do not match "
            f"source projection rows {out_features}",
            path=f"/{name}", phase="L1", blocking=True,
            expected=out_features, actual=row_limit))
        return
    source = weight_io.load_3d_rows(source_dir, src_h, expert_idx, source_row_idx)
    source = source.reshape(-1, in_features)
    # One read per sampled expert: the bundle row is the expert's payload (the
    # exact byte ranges the runtime preads), then slice components per metadata.
    bundle_rows = weight_io.load_2d_rows_raw(package_dir, bundle_h, expert_idx)
    try:
        packed_all = bundle_mod.component_array(bundle_rows, pgeo["packed"])
        norms_all = bundle_mod.component_array(bundle_rows, pgeo["norms"])
    except bundle_mod.BundleFormatError as e:
        out.append(Validation(
            "error", "correctness.bad_bundle_metadata", str(e),
            path=f"/{bundle_key}", phase="L1", blocking=True))
        return
    packed = packed_all[:, row_idx, :].reshape(-1, packed_all.shape[-1])
    norms = norms_all[:, row_idx].reshape(-1)
    try:
        recon = tq_decode_rows(packed, norms, stored_bits, in_features, seed)
    except ValueError as e:
        out.append(Validation(
            "error", "correctness.tq_decode_failed", str(e),
            path=f"/{name}", phase="L1", blocking=True))
        return
    err = _errors(source, recon)
    _record_metric(metrics, t, "tq", err, rows=row_idx.tolist(), experts=expert_idx.tolist())
    _block_on_error(out, t, err, DEFAULT_TOLERANCES["tq_relative_rms"])
    counts["tq"] += 1


def _check_deepseek_v4_tq(
    t,
    group: DecodedExpertGroup,
    pkg_cat,
    package_dir,
    policy,
    out,
    metrics,
    counts,
):
    name = t["source_name"]
    (bundle_key,) = _sidecar_keys(t, (bundle_mod.BUNDLE_KEY_SUFFIX,))
    bundle_h = pkg_cat.get(bundle_key)
    if bundle_h is None:
        _missing(out, bundle_key, t)
        return
    pgeo = _bundle_projection_geometry(t, bundle_h, package_dir, out)
    if pgeo is None:
        return
    params = t.get("format_params", {})
    expected_bits = int(params.get("bits", 0))
    seed = int(params.get("seed", policy["seed"]))
    stored_bits = int(pgeo["bits"])
    if stored_bits != expected_bits:
        out.append(Validation(
            "error", "correctness.tq_bits_mismatch",
            f"{name} manifest bits {expected_bits} but bundle metadata stores "
            f"{stored_bits}",
            path=f"/{bundle_key}", phase="L1", blocking=True,
            expected=expected_bits, actual=stored_bits))
        return

    layer = int(t["layer_index"])
    projection = str(t.get("projection", ""))
    source_experts = group.experts(layer)
    available = min(len(source_experts), int(bundle_h.shape[0]))
    if available <= 0:
        out.append(Validation(
            "error", "correctness.missing_source_tensor",
            f"{name} has no DS4 expert rows available for reconstruction",
            path=f"/{name}", phase="L1", blocking=True))
        return
    expert_rows = weight_io.sample_indices(
        available,
        int(policy["tq_experts"]),
        _stable_seed(policy["seed"], name, "experts"),
    )
    row_limit = int(pgeo["packed"]["shape"][0])
    row_idx = weight_io.sample_indices(
        row_limit,
        int(policy["rows_per_expert"]),
        _stable_seed(policy["seed"], name, projection, "rows"),
    )
    source_parts = []
    for row in expert_rows:
        expert_index = source_experts[int(row)]
        decoded = group.decode(
            layer=layer,
            expert_index=expert_index,
            projection=projection,
            out_dtype=np.float32,
        )
        if decoded.shape[0] != row_limit:
            out.append(Validation(
                "error", "correctness.shape_mismatch",
                f"{name} projection {projection!r} package rows {row_limit} do not "
                f"match source projection rows {decoded.shape[0]}",
                path=f"/{name}", phase="L1", blocking=True,
                expected=decoded.shape[0], actual=row_limit))
            return
        source_parts.append(decoded[row_idx])
    source = np.concatenate(source_parts, axis=0)
    in_features = int(source.shape[1])

    bundle_rows = weight_io.load_2d_rows_raw(package_dir, bundle_h, expert_rows)
    try:
        packed_all = bundle_mod.component_array(bundle_rows, pgeo["packed"])
        norms_all = bundle_mod.component_array(bundle_rows, pgeo["norms"])
    except bundle_mod.BundleFormatError as e:
        out.append(Validation(
            "error", "correctness.bad_bundle_metadata", str(e),
            path=f"/{bundle_key}", phase="L1", blocking=True))
        return
    packed = packed_all[:, row_idx, :].reshape(-1, packed_all.shape[-1])
    norms = norms_all[:, row_idx].reshape(-1)
    try:
        recon = tq_decode_rows(packed, norms, stored_bits, in_features, seed)
    except ValueError as e:
        out.append(Validation(
            "error", "correctness.tq_decode_failed", str(e),
            path=f"/{name}", phase="L1", blocking=True))
        return
    err = _errors(source, recon)
    _record_metric(
        metrics,
        t,
        "tq",
        err,
        rows=row_idx.tolist(),
        experts=expert_rows.tolist(),
    )
    _block_on_error(out, t, err, DEFAULT_TOLERANCES["tq_relative_rms"])
    counts["tq"] += 1


def _provenance(tq_seen: bool, affine_seen: bool, passthrough_seen: bool) -> list[dict]:
    out = []
    if passthrough_seen:
        out.append({"component": "passthrough", "kind": "independent",
                    "identity": "source/package passthrough comparison",
                    "shared_with": []})
    if affine_seen:
        out.append({"component": "affine", "kind": "external_codec",
                    "identity": "mlx.core.dequantize",
                    "shared_with": ["mlx affine codec"]})
    if tq_seen:
        out.append({"component": "tq", "kind": "independent",
                    "identity": "moespresso.correctness.tq_reference",
                    "shared_with": []})
    return out


def l1_tensor_reconstruction(
    profile: dict,
    inventory: dict,
    manifest: dict,
    source_dir: Path,
    package_dir: Path,
    *,
    sample_policy: dict | None = None,
) -> dict:
    """Emit L1 sampled tensor-reconstruction evidence."""
    del profile  # profile drives L0/L0b; L1 consumes the concrete manifest formats.
    policy = _policy(sample_policy)
    source_dir = Path(source_dir)
    package_dir = Path(package_dir)
    src_cat = weight_io.scan_offsets(source_dir)
    pkg_cat = weight_io.scan_offsets(package_dir)
    validations: list[Validation] = []
    metrics: list[dict] = []
    counts = {
        "fp16": 0,
        "f32_passthrough": 0,
        "raw_dtype_passthrough": 0,
        "affine": 0,
        "mxfp4": 0,
        "mxfp8": 0,
        "tq": 0,
    }
    smoke_max_experts = _smoke_max_experts(manifest)

    tensors = manifest.get("tensors", [])
    passthrough = [t for t in tensors if t.get("format") == "fp16"]
    f32_passthrough = [t for t in tensors if t.get("format") == "f32_passthrough"]
    raw_passthrough = [t for t in tensors if t.get("format") == "raw_dtype_passthrough"]
    affine = _limit([t for t in tensors if t.get("format") == "affine"],
                    int(policy["affine_tensors"]))
    mx_dense = _limit(
        [
            t for t in tensors
            if t.get("format") in {"mxfp4", "mxfp8"} and t.get("kind") != "expert"
        ],
        int(policy["affine_tensors"]),
    )
    tq = _limit([t for t in tensors if t.get("format") == "tq"], int(policy["tq_tensors"]))
    ds4_expert_group = None
    family = manifest.get("architecture", {}).get("family")
    if family == "deepseek_v4_flash" and tq:
        try:
            ds4_expert_group = DecodedExpertGroup.from_inventory(inventory, source_dir)
        except ValueError as e:
            validations.append(Validation(
                "error", "correctness.source_adapter_unavailable",
                f"DeepSeek V4 expert reconstruction needs a valid source adapter: {e}",
                phase="L1", blocking=True))

    for t in passthrough:
        _check_passthrough(t, src_cat, pkg_cat, source_dir, package_dir,
                           smoke_max_experts, validations, metrics, counts)
    for t in f32_passthrough:
        _check_f32_passthrough(t, src_cat, pkg_cat, source_dir, package_dir,
                               validations, metrics, counts)
    for t in raw_passthrough:
        _check_raw_passthrough(t, src_cat, pkg_cat, source_dir, package_dir,
                               validations, metrics, counts)
    for t in affine:
        _check_affine(t, src_cat, pkg_cat, source_dir, package_dir, policy,
                      validations, metrics, counts)
    for t in mx_dense:
        _check_mx_dense(t, src_cat, pkg_cat, source_dir, package_dir, policy,
                        validations, metrics, counts)
    for t in tq:
        if (ds4_expert_group is not None
                and t.get("source_name", "").startswith("layers.")
                and ".ffn.experts." in t.get("source_name", "")):
            _check_deepseek_v4_tq(
                t, ds4_expert_group, pkg_cat, package_dir, policy,
                validations, metrics, counts)
        else:
            _check_tq(t, src_cat, pkg_cat, source_dir, package_dir, policy,
                      validations, metrics, counts)

    for fmt, present in (
        ("affine", any(t.get("format") == "affine" for t in tensors)),
        ("mxfp4", any(t.get("format") == "mxfp4" and t.get("kind") != "expert"
                      for t in tensors)),
        ("mxfp8", any(t.get("format") == "mxfp8" for t in tensors)),
        ("tq", any(t.get("format") == "tq" for t in tensors)),
        ("fp16", any(t.get("format") == "fp16" for t in tensors)),
        ("f32_passthrough", any(t.get("format") == "f32_passthrough" for t in tensors)),
        ("raw_dtype_passthrough", any(t.get("format") == "raw_dtype_passthrough"
                                      for t in tensors)),
    ):
        if present and counts[fmt] == 0:
            validations.append(Validation(
                "error", "correctness.no_samples",
                f"manifest contains {fmt} tensors but L1 sampled none successfully",
                path="/tensors", phase="L1", blocking=True))

    blocking = any(v.blocking for v in validations)
    worst = {}
    for fmt in (
        "fp16", "f32_passthrough", "raw_dtype_passthrough",
        "affine", "mxfp4", "mxfp8", "tq",
    ):
        vals = [m for m in metrics if m["format"] == fmt]
        if vals:
            worst[fmt] = {
                "max_abs": max(v["max_abs"] for v in vals),
                "relative_rms": max(v["relative_rms"] for v in vals),
            }
    return make_artifact(
        "correctness_evidence",
        inventory.get("subject", manifest.get("subject", {"source_root": "unknown"})),
        PRODUCER,
        status="invalid" if blocking else "valid",
        validation=validations,
        inputs=[x for x in (inventory.get("artifact_id"), manifest.get("artifact_id")) if x],
        rung="L1",
        sample_policy=policy,
        tolerances=dict(DEFAULT_TOLERANCES),
        reference_provenance=_provenance(
            bool(tq),
            bool(affine or mx_dense),
            bool(passthrough or f32_passthrough or raw_passthrough),
        ),
        metrics=metrics,
        summary={
            "findings": len(validations),
            "blocking": sum(1 for v in validations if v.blocking),
            "sampled_by_format": counts,
            "worst_by_format": worst,
        },
    )
