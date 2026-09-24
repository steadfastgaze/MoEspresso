"""Resolve the released Qwen MTP source without reading target weight payloads."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from moespresso.inventory.qwen4.roles import module_path, tensor_role
from moespresso.inventory.qwen4.static import (
    EXPECTED_MTP_TENSOR_COUNT,
    expected_qwen38_flash_next_header_specs,
    validate_qwen38_flash_next_static,
)
from moespresso.inventory.safetensors_header import TensorHeader, read_headers_with_offsets


class Qwen4MTPSourceError(ValueError):
    """An MTP source fails its checkpoint or tensor contract."""


@dataclass(frozen=True)
class Qwen4MTPSourceTensor:
    """One resolved source tensor and its drafter-owned destination."""

    header: TensorHeader
    module_path: str
    kind: Literal["matrix", "expert", "passthrough"]
    role: str
    projection: str | None = None


def _source_aliases() -> dict[str, str]:
    """Map the MTP block onto the existing full-attention tensor contract."""
    out = {}
    for name in expected_qwen38_flash_next_header_specs():
        if name.startswith("model.language_model.layers.3."):
            out[name.replace("model.language_model.layers.3.", "mtp.layers.0.", 1)] = name
        elif name.startswith("model.language_model.hyper_connection_mixer."):
            out[name.replace("model.language_model.", "mtp.", 1)] = name
    return out


def expected_qwen4_mtp_header_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    """Exact source shapes, including the MTP-specific fusion projections."""
    trunk = expected_qwen38_flash_next_header_specs()
    specs = {name: trunk[alias] for name, alias in _source_aliases().items()}
    specs.update({
        "mtp.pre_fc_norm_embedding.weight": ("BF16", (2560,)),
        "mtp.pre_fc_norm_hidden.weight": ("BF16", (10240,)),
        "mtp.fc_embedding.weight": ("BF16", (2560, 2560)),
        "mtp.fc_hidden.weight": ("BF16", (2560, 2560)),
    })
    if len(specs) != EXPECTED_MTP_TENSOR_COUNT:
        raise AssertionError("MTP source contract has an inconsistent tensor count")
    return specs


def resolve_qwen4_mtp_headers(headers: list[TensorHeader]) -> tuple[Qwen4MTPSourceTensor, ...]:
    """Validate every MTP tensor and resolve names once for subsequent phases."""
    specs = expected_qwen4_mtp_header_specs()
    by_name = {}
    for header in headers:
        if header.name in by_name:
            raise Qwen4MTPSourceError(f"duplicate MTP source tensor: {header.name}")
        by_name[header.name] = header
    if set(by_name) != set(specs):
        raise Qwen4MTPSourceError(
            f"MTP tensor set differs: missing={sorted(set(specs) - set(by_name))}, "
            f"unexpected={sorted(set(by_name) - set(specs))}"
        )
    aliases = _source_aliases()
    out = []
    for name, (dtype, shape) in sorted(specs.items()):
        header = by_name[name]
        if header.dtype != dtype or header.shape != shape:
            raise Qwen4MTPSourceError(
                f"{name}: expected {dtype} {shape}, got {header.dtype} {header.shape}"
            )
        if (
            header.header_size < 8
            or header.begin < 0
            or header.end - header.begin != math.prod(shape) * 2
        ):
            raise Qwen4MTPSourceError(f"{name}: invalid source byte range")
        if name in aliases:
            alias = aliases[name]
            resolved = tensor_role(alias)
            destination = module_path(alias)
            if resolved is None or destination is None:
                raise AssertionError(f"MTP alias has no existing tensor role: {alias}")
            kind = resolved["kind"]
            out.append(Qwen4MTPSourceTensor(
                header=header,
                module_path=destination.replace("layers.3.", "layers.0.", 1),
                kind="matrix" if kind == "affine" else kind,
                role=resolved["role"],
                projection=resolved.get("projection"),
            ))
        else:
            out.append(Qwen4MTPSourceTensor(
                header=header,
                module_path=name.removeprefix("mtp.").removesuffix(".weight"),
                kind="passthrough" if len(shape) == 1 else "matrix",
                role="mtp.norm" if len(shape) == 1 else "mtp.fusion",
            ))
    return tuple(out)


def inspect_qwen4_mtp_source(source_dir: Path) -> tuple[dict, tuple[Qwen4MTPSourceTensor, ...]]:
    """Read config, index and only headers of shards carrying MTP tensors."""
    source_dir = Path(source_dir)
    config = json.loads((source_dir / "config.json").read_text())
    index = json.loads((source_dir / "model.safetensors.index.json").read_text())
    issues = validate_qwen38_flash_next_static(config, index)
    errors = [issue.message for issue in issues if issue.blocking or issue.severity == "error"]
    if errors:
        raise Qwen4MTPSourceError("; ".join(errors))
    names = {name: shard for name, shard in index["weight_map"].items() if name.startswith("mtp.")}
    headers = []
    for shard in sorted(set(names.values())):
        path = source_dir / shard
        if Path(shard).name != shard or not path.is_file():
            raise Qwen4MTPSourceError(f"invalid or missing MTP shard: {shard}")
        size = path.stat().st_size
        for header in read_headers_with_offsets(path):
            if not header.name.startswith("mtp."):
                continue
            if names.get(header.name) != shard:
                raise Qwen4MTPSourceError(f"index disagrees with MTP shard: {header.name}")
            if header.header_size + header.end > size:
                raise Qwen4MTPSourceError(f"MTP tensor extends beyond its shard: {header.name}")
            headers.append(header)
    return config["text_config"], resolve_qwen4_mtp_headers(headers)
