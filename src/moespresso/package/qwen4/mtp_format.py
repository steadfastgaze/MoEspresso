"""IQ2_K projection geometry for the optional Qwen MTP sidecar."""

from __future__ import annotations

from dataclasses import dataclass

from moespresso.package.iqk_format import IQK_LAYOUT_IQK_RELAYOUT, iqk_geometry


MTP_CODEC = "iq2_k"
MTP_GRAPH_CONTRACT = "qwen4_mtp_full_width_fusion_v1"
MTP_SIDECAR_SCHEMA = "qwen4_mtp_iq2_sidecar_v1"
MTP_SIDECAR_NAME = "mtp_manifest.json"
_INPUT_WIDTHS = (768, 2048, 2560, 4096)
_OUTPUT_ALIGNMENT = 16


@dataclass(frozen=True)
class MTPInputSlice:
    """One contiguous reduction slice, padded for an existing IQ2 kernel."""

    begin: int
    end: int
    stored_width: int


@dataclass(frozen=True)
class MTPProjectionPlan:
    """Logical projection and its existing-kernel storage geometry."""

    num_experts: int
    out_features: int
    in_features: int
    stored_out_features: int
    slices: tuple[MTPInputSlice, ...]

    @property
    def encoded_bytes(self) -> int:
        geometry = iqk_geometry(MTP_CODEC)
        return self.num_experts * self.stored_out_features * sum(
            geometry.bytes_per_row(part.stored_width) for part in self.slices
        )


def mtp_iq2_projection_plan(shape: tuple[int, ...]) -> MTPProjectionPlan:
    """Cover a dense or expert matrix using supported IQ2 reduction widths.

    Wide matrices use adjacent block-aligned slices. Short reductions and output
    rows are zero-padded. No new codec or byte layout is introduced.
    """
    if len(shape) not in (2, 3) or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("MTP projection shape must contain two or three positive integers")
    experts = shape[0] if len(shape) == 3 else 1
    rows, columns = shape[-2:]
    slices = []
    begin = 0
    while begin < columns:
        remaining = columns - begin
        width = next((width for width in _INPUT_WIDTHS if width >= remaining), _INPUT_WIDTHS[-1])
        end = min(begin + width, columns)
        slices.append(MTPInputSlice(begin, end, width))
        begin = end
    stored_rows = (rows + _OUTPUT_ALIGNMENT - 1) // _OUTPUT_ALIGNMENT * _OUTPUT_ALIGNMENT
    return MTPProjectionPlan(experts, rows, columns, stored_rows, tuple(slices))


def mtp_graph_config(config: dict) -> dict:
    """Select the self-contained draft geometry shared with the target."""
    import math

    integer_fields = (
        "hidden_size", "hc_count", "hc_lowrank", "head_dim", "num_attention_heads",
        "num_key_value_heads", "indexer_n_heads", "indexer_kv_heads", "indexer_head_dim",
        "indexer_budget", "indexer_compress_ratio", "moe_intermediate_size",
        "shared_expert_intermediate_size", "num_experts", "num_experts_per_tok", "vocab_size",
    )
    out = {}
    for name in integer_fields:
        value = config.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"MTP config {name} must be a positive integer")
        out[name] = value
    rope = config.get("rope_parameters", {})
    eps, base = config.get("rms_norm_eps"), rope.get("rope_theta")
    factor = config.get("partial_rotary_factor")
    for name, value in (("rms_norm_eps", eps), ("rope_theta", base), ("partial_rotary_factor", factor)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"MTP config {name} must be finite and positive")
    rotary = out["head_dim"] * factor
    section = rope.get("mrope_section")
    if (
        rotary != int(rotary) or rotary % 2 or rotary > min(out["head_dim"], out["indexer_head_dim"])
        or not isinstance(section, (list, tuple)) or len(section) != 3
        or any(type(value) is not int or value < 0 for value in section)
        or sum(section) != rotary // 2
    ):
        raise ValueError("MTP rotary geometry is invalid")
    out.update(rms_norm_eps=float(eps), rope_base=float(base), rotary_dim=int(rotary),
               mrope_section=list(section))
    return out


def read_mtp_sidecar_manifest(sidecar_dir, target: dict) -> dict:
    """Validate identities, geometry and file headers without hashing payloads."""
    from dataclasses import asdict
    from pathlib import Path

    from moespresso.core.artifact import canonical_json, compute_artifact_id, read_artifact, validate_base
    from moespresso.core.paths import resolve_artifact_file
    from moespresso.inventory.safetensors_header import read_headers_with_offsets

    if (
        target.get("artifact_kind") != "package_manifest"
        or target.get("status") != "valid"
        or target.get("architecture", {}).get("family") != "qwen4_exp"
        or target.get("artifact_id") != compute_artifact_id(target)
        or any(issue.blocking for issue in validate_base(target))
    ):
        raise ValueError("MTP target manifest is invalid")
    root = Path(sidecar_dir)
    manifest = read_artifact(root / MTP_SIDECAR_NAME)
    if (
        manifest["artifact_kind"] != "package_manifest" or manifest["status"] != "valid"
        or manifest.get("subject") != {"family": "qwen4_exp", "role": "mtp_sidecar"}
        or manifest.get("sidecar_schema") != MTP_SIDECAR_SCHEMA
        or manifest.get("graph_contract") != MTP_GRAPH_CONTRACT
        or manifest.get("quantization") != {"codec": MTP_CODEC, "objective": "ik_unweighted", "calibrated": False}
    ):
        raise ValueError("MTP sidecar contract is unsupported")
    if manifest.get("target_artifact_id") != target["artifact_id"]:
        raise ValueError("MTP sidecar belongs to a different target package")
    if manifest.get("graph") != mtp_graph_config(target["architecture"]["config"]):
        raise ValueError("MTP sidecar geometry differs from the target")
    records = manifest.get("tensors")
    files = manifest.get("files")
    if not isinstance(records, list) or not records or not isinstance(files, list) or not files:
        raise ValueError("MTP sidecar requires tensors and files")
    headers = {}
    file_names = set()
    for record in files:
        path = resolve_artifact_file(root, record["path"])
        digest = record.get("sha256")
        if (
            path.name in file_names or type(record.get("size_bytes")) is not int
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not path.is_file() or path.stat().st_size != record["size_bytes"]
        ):
            raise ValueError(f"MTP sidecar file is missing, incomplete or duplicated: {path.name}")
        file_names.add(path.name)
        for header in read_headers_with_offsets(path):
            if header.begin < 0 or header.end < header.begin or header.header_size + header.end > record["size_bytes"]:
                raise ValueError("MTP tensor exceeds its payload file")
            headers[(path.name, header.name)] = header
    seen = set()
    for tensor in records:
        kind = tensor.get("kind")
        shape = tuple(tensor.get("logical_shape", ()))
        if kind == "passthrough":
            if tensor.get("format") != "raw_dtype_passthrough" or len(shape) != 1:
                raise ValueError("MTP structural tensor contract is invalid")
        elif kind in {"matrix", "expert"}:
            plan = mtp_iq2_projection_plan(shape)
            if (
                tensor.get("format") != "iqk" or tensor.get("codec") != MTP_CODEC
                or tensor.get("layout") != IQK_LAYOUT_IQK_RELAYOUT
                or canonical_json(tensor.get("projection_plan", {})) != canonical_json(asdict(plan))
                or tensor.get("encoded_bytes") != plan.encoded_bytes
            ):
                raise ValueError("MTP IQ2 tensor contract is invalid")
        else:
            raise ValueError("MTP tensor kind is unsupported")
        payloads = tensor.get("payloads")
        if not isinstance(payloads, list) or not payloads:
            raise ValueError("MTP tensor has no declared payload")
        for payload in payloads:
            key = (payload["file"], payload["key"])
            header = headers.get(key)
            if (
                key in seen or header is None
                or payload["dtype"] != header.dtype or tuple(payload["shape"]) != header.shape
            ):
                raise ValueError("MTP payload header disagrees with the manifest")
            seen.add(key)
    if seen != set(headers):
        raise ValueError("MTP payload files contain undeclared tensors")
    return manifest
