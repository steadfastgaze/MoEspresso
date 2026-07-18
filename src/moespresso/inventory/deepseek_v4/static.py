"""DeepSeek-V4-Flash static checkpoint checks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from moespresso.core.artifact import Validation
from moespresso.inventory.architecture_profile import DEEPSEEK_V4_FLASH_COMPRESS_RATIOS
from moespresso.inventory.safetensors_header import TensorHeader, scan_headers

EXPECTED_TOTAL_SIZE = 159_609_485_896
EXPECTED_SHARD_COUNT = 46
EXPECTED_TENSOR_COUNT = 69_187

_CONFIG_FACTS = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "sliding_window": 128,
    "index_topk": 512,
    "compress_rope_theta": 160000,
}

_TENSOR_FACTS = {
    "embed.weight": ("BF16", [129280, 4096]),
    "head.weight": ("BF16", [129280, 4096]),
    "layers.0.attn.wq_a.weight": ("F8_E4M3", [1024, 4096]),
    "layers.0.attn.wq_a.scale": ("F8_E8M0", [8, 32]),
    "layers.2.attn.compressor.ape": ("F32", [4, 1024]),
    "layers.2.attn.indexer.wq_b.weight": ("F8_E4M3", [8192, 1024]),
    "layers.3.attn.compressor.ape": ("F32", [128, 512]),
    "layers.0.ffn.gate.tid2eid": ("I64", [129280, 6]),
    "layers.3.ffn.gate.bias": ("F32", [256]),
    "layers.0.ffn.experts.0.w1.weight": ("I8", [2048, 2048]),
    "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", [2048, 128]),
    "layers.0.ffn.shared_experts.w1.weight": ("F8_E4M3", [2048, 4096]),
    "layers.0.ffn.shared_experts.w2.weight": ("F8_E4M3", [4096, 2048]),
    "layers.0.hc_attn_fn": ("F32", [24, 16384]),
    "hc_head_fn": ("F32", [4, 16384]),
    "mtp.0.e_proj.weight": ("F8_E4M3", [4096, 4096]),
}


def _error(code: str, message: str, *, path: str, expected=None, actual=None) -> Validation:
    return Validation(
        "error",
        code,
        message,
        path=path,
        phase="deepseek_v4_static",
        blocking=True,
        expected=expected,
        actual=actual,
    )


def _check_equal(out: list[Validation], path: str, actual, expected) -> None:
    if actual != expected:
        out.append(_error(
            "deepseek_v4.static_mismatch",
            f"{path} is {actual!r}, expected {expected!r}",
            path=path,
            expected=expected,
            actual=actual,
        ))


def _index_facts(model_dir: Path) -> dict:
    idx = Path(model_dir) / "model.safetensors.index.json"
    if not idx.exists():
        return {}
    data = json.loads(idx.read_text())
    weight_map = data.get("weight_map", {})
    return {
        "tensor_count": len(weight_map),
        "shard_count": len(set(weight_map.values())),
        "total_size": data.get("metadata", {}).get("total_size"),
    }


def validate_deepseek_v4_static(
    config: dict,
    headers: Sequence[TensorHeader],
    *,
    shard_count: int | None = None,
    tensor_count: int | None = None,
    total_size: int | None = None,
) -> list[Validation]:
    """Validate static DeepSeek-V4-Flash config and safetensors header facts."""
    out: list[Validation] = []

    for key, expected in _CONFIG_FACTS.items():
        _check_equal(out, f"/config/{key}", config.get(key), expected)

    ratios = config.get("compress_ratios")
    _check_equal(out, "/config/compress_ratios", ratios, DEEPSEEK_V4_FLASH_COMPRESS_RATIOS)
    if isinstance(ratios, list):
        _check_equal(out, "/config/compress_ratios/42", ratios[42] if len(ratios) > 42 else None, 4)
        _check_equal(out, "/config/compress_ratios/43", ratios[43] if len(ratios) > 43 else None, 0)

    if shard_count is not None:
        _check_equal(out, "/index/shard_count", shard_count, EXPECTED_SHARD_COUNT)
    if tensor_count is not None:
        _check_equal(out, "/index/tensor_count", tensor_count, EXPECTED_TENSOR_COUNT)
    if total_size is not None:
        _check_equal(out, "/index/total_size", total_size, EXPECTED_TOTAL_SIZE)

    by_name = {h.name: h for h in headers}
    for name, (expected_dtype, expected_shape) in _TENSOR_FACTS.items():
        header = by_name.get(name)
        if header is None:
            out.append(_error(
                "deepseek_v4.missing_tensor",
                f"{name} is missing from safetensors headers",
                path=f"/tensors/{name}",
                expected={"dtype": expected_dtype, "shape": expected_shape},
                actual=None,
            ))
            continue
        if header.dtype != expected_dtype:
            out.append(_error(
                "deepseek_v4.dtype_mismatch",
                f"{name} dtype is {header.dtype}, expected {expected_dtype}",
                path=f"/tensors/{name}/dtype",
                expected=expected_dtype,
                actual=header.dtype,
            ))
        shape = list(header.shape)
        if shape != expected_shape:
            out.append(_error(
                "deepseek_v4.shape_mismatch",
                f"{name} shape is {shape}, expected {expected_shape}",
                path=f"/tensors/{name}/shape",
                expected=expected_shape,
                actual=shape,
            ))

    return out


def validate_deepseek_v4_model_dir(model_dir: Path) -> list[Validation]:
    """Run the static DS4 validator on a source directory."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    facts = _index_facts(model_dir)
    return validate_deepseek_v4_static(
        config,
        scan_headers(model_dir),
        shard_count=facts.get("shard_count"),
        tensor_count=facts.get("tensor_count"),
        total_size=facts.get("total_size"),
    )
