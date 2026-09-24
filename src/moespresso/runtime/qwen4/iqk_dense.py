"""Manifest-owned dense IQ_K projections for the Qwen4 runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    iqk_dense_geometry,
    normalize_iqk_layout,
)


_DENSE_QMV_ENV = "MOESPRESSO_QWEN4_IQK_DENSE_QMV"
_CALL_COUNTS = {
    "decode_gemv": 0,
    "prefill_dequant": 0,
    "delegated": 0,
}


class Qwen4IQKDenseError(RuntimeError):
    """A dense IQ_K manifest, module geometry, or runtime call is invalid."""


def qwen4_iqk_dense_call_counts() -> dict[str, int]:
    """Return route engagement counts for served-path evidence."""
    return dict(_CALL_COUNTS)


def qwen4_iqk_dense_qmv_enabled() -> bool:
    """Return whether decode-shaped calls use the packed GEMV kernels."""
    return os.environ.get(_DENSE_QMV_ENV, "1") != "0"


def qwen4_iqk_dense_weight_map(
    manifest: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    """Return dense IQ_K install facts keyed by exact module weight key."""
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise Qwen4IQKDenseError("manifest tensors must be an array")
    out: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(tensors):
        if not isinstance(raw, Mapping):
            raise Qwen4IQKDenseError(f"manifest tensor {index} must be an object")
        if raw.get("format") != "iqk" or raw.get("kind") == "expert":
            continue
        params = raw.get("format_params")
        if not isinstance(params, Mapping):
            raise Qwen4IQKDenseError(f"{raw.get('source_name')}: format_params must be an object")
        member = params.get("iqk_codec")
        try:
            iqk_dense_geometry(str(member))
        except ValueError as exc:
            raise Qwen4IQKDenseError(f"{raw.get('source_name')}: {exc}") from exc
        layout = normalize_iqk_layout(params.get("layout"))
        if layout != IQK_LAYOUT_IQK_RELAYOUT:
            raise Qwen4IQKDenseError(
                f"{raw.get('source_name')}: dense IQ_K requires "
                f"{IQK_LAYOUT_IQK_RELAYOUT!r}, got {layout!r}"
            )
        path = raw.get("module_path")
        key = raw.get("module_weight_key")
        if not isinstance(path, str) or key != f"{path}.weight":
            raise Qwen4IQKDenseError(
                f"{raw.get('source_name')}: dense IQ_K lacks exact module ownership"
            )
        facts = {
            "member": str(member),
            "layout": str(layout),
            "source_name": str(raw.get("source_name")),
        }
        if key in out:
            raise Qwen4IQKDenseError(f"duplicate dense IQ_K destination: {key}")
        out[key] = facts
    return out


_DENSE_CLASS = None


def _dense_class():
    global _DENSE_CLASS
    if _DENSE_CLASS is not None:
        return _DENSE_CLASS

    import mlx.core as mx
    import mlx.nn as nn

    class Qwen4IQKDenseLinear(nn.Module):
        """One packed dense projection with decode and bulk routes."""

        mode = "iqk_dense"

        def __init__(
            self,
            member: str,
            layout: str,
            out_features: int,
            in_features: int,
        ) -> None:
            super().__init__()
            if layout != IQK_LAYOUT_IQK_RELAYOUT:
                raise Qwen4IQKDenseError(f"unsupported dense IQ_K layout {layout!r}")
            geometry = iqk_dense_geometry(member)
            self.member = str(member)
            self.layout = str(layout)
            self.out_features = int(out_features)
            self.in_features = int(in_features)
            self.bytes_per_row = geometry.bytes_per_row(self.in_features)
            self.weight = mx.zeros((self.out_features, self.bytes_per_row), dtype=mx.uint8)
            self.freeze()

        def __call__(self, values):
            if int(values.shape[-1]) != self.in_features:
                raise Qwen4IQKDenseError(
                    f"activation width {int(values.shape[-1])} does not match {self.in_features}"
                )
            rows = 1
            for dimension in values.shape[:-1]:
                rows *= int(dimension)
            expected = (self.out_features, self.bytes_per_row)
            if tuple(int(value) for value in self.weight.shape) != expected:
                raise Qwen4IQKDenseError(
                    f"{self.member} weight has shape {tuple(self.weight.shape)}, expected {expected}"
                )
            from mlx_iqk.dense import dense_linear_packed

            if rows == 1 and qwen4_iqk_dense_qmv_enabled():
                _CALL_COUNTS["decode_gemv"] += 1
                token_limit = 1
            elif rows > 1:
                _CALL_COUNTS["prefill_dequant"] += 1
                token_limit = 0
            else:
                _CALL_COUNTS["delegated"] += 1
                token_limit = 0
            result = dense_linear_packed(
                self.member,
                values,
                self.weight,
                self.out_features,
                self.in_features,
                token_limit=token_limit,
            )
            return result.astype(values.dtype)

    _DENSE_CLASS = Qwen4IQKDenseLinear
    return _DENSE_CLASS


def _walk_module_path(model: Any, path: str) -> tuple[Any, str]:
    parts = path.split(".")
    current = model
    try:
        for part in parts[:-1]:
            current = current[int(part)] if part.isdigit() else getattr(current, part)
        getattr(current, parts[-1])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise Qwen4IQKDenseError(f"manifest module path does not exist: {path}") from exc
    return current, parts[-1]


def _logical_shape(module: Any) -> tuple[int, int]:
    weight = getattr(module, "weight", None)
    if weight is None or len(weight.shape) != 2:
        raise Qwen4IQKDenseError(f"{type(module).__name__} does not carry a two-dimensional weight")
    rows = int(weight.shape[0])
    scales = getattr(module, "scales", None)
    group_size = getattr(module, "group_size", None)
    if scales is not None and group_size is not None:
        return rows, int(scales.shape[-1]) * int(group_size)
    return rows, int(weight.shape[-1])


def install_qwen4_iqk_dense_modules(
    model: Any,
    manifest: Mapping[str, object],
) -> int:
    """Swap manifest-declared dense IQ_K leaves before package hydration."""
    weight_map = qwen4_iqk_dense_weight_map(manifest)
    if not weight_map:
        return 0
    dense_class = _dense_class()
    members: dict[str, int] = {}
    for key in sorted(weight_map):
        module_path = key.removesuffix(".weight")
        facts = weight_map[key]
        parent, attribute = _walk_module_path(model, module_path)
        current = getattr(parent, attribute)
        out_features, in_features = _logical_shape(current)
        replacement = dense_class(
            facts["member"],
            facts["layout"],
            out_features,
            in_features,
        )
        replacement.eval()
        setattr(parent, attribute, replacement)
        members[facts["member"]] = members.get(facts["member"], 0) + 1
    install = {
        "modules": len(weight_map),
        "member_counts": dict(sorted(members.items())),
        "layout": IQK_LAYOUT_IQK_RELAYOUT,
    }
    object.__setattr__(model, "_moespresso_qwen4_iqk_dense_install", install)
    return len(weight_map)


def qwen4_iqk_dense_engagement(model: Any) -> dict[str, object]:
    """Return install facts and route counts for runtime probes."""
    return {
        "install": getattr(model, "_moespresso_qwen4_iqk_dense_install", None),
        "counts": qwen4_iqk_dense_call_counts(),
    }


__all__ = [
    "Qwen4IQKDenseError",
    "install_qwen4_iqk_dense_modules",
    "qwen4_iqk_dense_call_counts",
    "qwen4_iqk_dense_engagement",
    "qwen4_iqk_dense_qmv_enabled",
    "qwen4_iqk_dense_weight_map",
]
