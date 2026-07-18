"""Build a `source_inventory` artifact from a model's safetensors headers.

One pass over headers: classify each tensor (expert / non-expert / skip),
resolve its internal role and GGUF imatrix key at the inventory boundary, and
record a typed entry. The result is the first durable artifact every later phase
consumes, so probes/optimizer/converter never re-parse raw names.

Pure except for header reading at the edge. No weight bytes, no MLX.
"""

from __future__ import annotations

from pathlib import Path

from moespresso.core.artifact import Validation, make_artifact
from moespresso.inventory import roles
from moespresso.inventory.deepseek_v4 import roles as deepseek_v4_roles
from moespresso.inventory.safetensors_header import TensorHeader, scan_headers

# Substrings marking a tensor we never quantize/probe (norms, biases, rope,
# vision/audio, mtp). Explicit and minimal.
_SKIP_SUBSTR = (
    "norm", "bias", "rotary", "rope", "conv", "_scale",
    "vision", "visual", "audio", "image", "mtp",
)

PRODUCER = {"tool": "moespresso.inventory", "version": "1.0.0"}


def _classify_deepseek_v4(h: TensorHeader) -> dict | None:
    """One DS4 tensor -> inventory entry, or None for explicit exclusions."""
    name = h.name
    resolved = deepseek_v4_roles.tensor_role(name)
    if resolved is None:
        return None
    entry = {
        "source_name": name,
        "role": resolved["role"],
        "kind": resolved["kind"],
        "layer_index": resolved.get("layer_index", deepseek_v4_roles.tensor_layer(name)),
        "shape": list(h.shape),
        "dtype": h.dtype,
        "shard": h.shard,
        "gguf_keys": [],
        "status": "unknown" if resolved["kind"] == "unknown" else "required",
    }
    if resolved["kind"] == "expert_source":
        entry["gguf_keys"] = roles.expert_gguf_keys(
            int(entry["layer_index"]), resolved["projection"])
    elif resolved["kind"] == "affine":
        gguf_key = deepseek_v4_roles.gguf_key(name)
        if gguf_key is not None:
            entry["gguf_keys"] = [gguf_key]
    for key in ("projection", "expert_index", "source_projection", "format"):
        if key in resolved:
            entry[key] = resolved[key]
    return entry


def _classify(
    h: TensorHeader,
    layer_types: list[str] | None,
    family: str | None = None,
) -> dict | None:
    """One tensor -> inventory entry, or None to skip."""
    name = h.name
    if family == "deepseek_v4_flash":
        return _classify_deepseek_v4(h)

    # Structural text tensors (norms, SSM state) are not quantized but the graph
    # needs them. Carry them as passthrough. Checked before _SKIP_SUBSTR, which
    # would otherwise drop them on 'norm'/'conv'/'bias'. mtp/vision return None.
    srole = roles.structural_role(name)
    if srole is not None:
        return {
            "source_name": name,
            "role": srole,
            "kind": "passthrough",
            "layer_index": roles.tensor_layer(name),
            "shape": list(h.shape),
            "dtype": h.dtype,
            "shard": h.shard,
            "gguf_keys": [],
            "status": "required",
        }

    if any(s in name for s in _SKIP_SUBSTR):
        return None

    stacked = roles.parse_stacked_expert(name)
    if stacked and len(h.shape) == 3:
        layer, proj = stacked
        return {
            "source_name": name,
            "role": roles.expert_role(proj),
            "kind": "expert",
            "layer_index": layer,
            "projection": proj,
            "shape": list(h.shape),
            "dtype": h.dtype,
            "shard": h.shard,
            "gguf_keys": roles.expert_gguf_keys(layer, proj),
            "status": "required",
        }

    if len(h.shape) != 2:
        return None  # not a 2D weight we handle

    role = roles.non_expert_role(name, layer_types)
    key = roles.non_expert_gguf_key(name, layer_types)
    return {
        "source_name": name,
        "role": role or "unknown",
        "kind": "affine",
        "layer_index": roles.tensor_layer(name),
        "shape": list(h.shape),
        "dtype": h.dtype,
        "shard": h.shard,
        "gguf_keys": [key] if key else [],
        "status": "required" if role else "unknown",
    }


def build_inventory_from_headers(
    headers: list[TensorHeader],
    subject: dict,
    layer_types: list[str] | None,
    imatrix_keys: set[str] | None = None,
    family: str | None = None,
) -> dict:
    """Build the `source_inventory` artifact from already-read headers.

    `imatrix_keys`, if given (the real GGUF key set), records coverage and flags
    any resolved key unexpectedly absent: the guard against a silent mapping
    regression.
    """
    entries = []
    for h in sorted(headers, key=lambda x: x.name):
        e = _classify(h, layer_types, family)
        if e is not None:
            entries.append(e)

    n_expert = sum(1 for e in entries if e["kind"] == "expert")
    n_expert_source = sum(1 for e in entries if e["kind"] == "expert_source")
    n_affine = sum(1 for e in entries if e["kind"] == "affine")
    n_codec_scale = sum(1 for e in entries if e["kind"] == "codec_scale")
    n_passthrough = sum(1 for e in entries if e["kind"] == "passthrough")
    n_unknown = sum(1 for e in entries if e["status"] == "unknown")

    validation: list[Validation] = []
    coverage = None
    if imatrix_keys is not None:
        resolved = mapped = absent = 0
        for e in entries:
            for k in e["gguf_keys"]:
                resolved += 1
                if k in imatrix_keys:
                    mapped += 1
                else:
                    absent += 1
                    validation.append(Validation(
                        "warning", "imatrix.key_absent",
                        f"resolved key {k} not in imatrix",
                        path=f"/{e['source_name']}", phase="inventory"))
        coverage = {"resolved_keys": resolved, "present_in_imatrix": mapped,
                    "absent": absent}

    if n_unknown and family == "deepseek_v4_flash":
        validation.append(Validation(
            "error", "inventory.unknown_tensors",
            f"{n_unknown} DeepSeek V4 tensors had no role",
            phase="inventory", blocking=True))
    elif n_unknown:
        validation.append(Validation(
            "warning", "inventory.unknown_tensors",
            f"{n_unknown} 2D weight tensors had no role", phase="inventory"))

    status = "valid" if not any(v.blocking for v in validation) else "invalid"
    return make_artifact(
        "source_inventory", subject, PRODUCER,
        status=status, validation=validation,
        tensors=entries,
        counts={"expert": n_expert, "affine": n_affine,
                "expert_source": n_expert_source, "codec_scale": n_codec_scale,
                "passthrough": n_passthrough, "unknown": n_unknown,
                "total": len(entries)},
        imatrix_coverage=coverage,
        layer_types=layer_types,
        family=family,
    )


def build_inventory(
    model_dir: Path,
    layer_types: list[str] | None = None,
    imatrix_keys: set[str] | None = None,
    family: str | None = None,
) -> dict:
    """Scan a model dir's headers and build its source_inventory artifact."""
    headers = scan_headers(model_dir)
    subject = {"source_root": model_dir.name, "source_format": "hf_safetensors"}
    return build_inventory_from_headers(headers, subject, layer_types, imatrix_keys, family)
