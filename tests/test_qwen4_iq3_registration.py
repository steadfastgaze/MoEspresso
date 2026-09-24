"""Focused Qwen4 IQ3_K registration and stored-width checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from moespresso.package.iqk_format import IQK_GEOMETRY, iqk_geometry
from moespresso.runtime.qwen4.expert_provider import _qwen4_projection_geometry
from moespresso.runtime.qwen4.load import Qwen4PackageLoadError
from moespresso.runtime.qwen4.load import _qwen4_expert_entries
from moespresso.runtime.ssd_streaming_build import SSDStreamingBuildError


def _manifest() -> dict:
    tensors = []
    for layer in range(48):
        for projection in ("gate", "up", "down"):
            logical = [2560, 640] if projection == "down" else [640, 2560]
            module = f"layers.{layer}.mlp.experts.{projection}_proj"
            if layer < 2:
                params = {
                    "kquant_codec": "q8_0",
                    "logical_shape": logical,
                    "stored_shape": logical,
                    "zero_padding": 0,
                }
                fmt = "kquant"
            else:
                member = "iq3_k" if layer == 2 else "iq2_k"
                stored = [2560, 768] if projection == "down" else logical
                params = {
                    "iqk_codec": member,
                    "layout": "iqk_relayout",
                    "logical_shape": logical,
                    "stored_shape": stored,
                    "zero_padding": stored[1] - logical[1],
                    "calibration_policy": "per_expert_route_active",
                }
                fmt = "iqk"
            tensors.append(
                {
                    "kind": "expert",
                    "layer_index": layer,
                    "projection": projection,
                    "format": fmt,
                    "format_params": params,
                    "module_path": module,
                    "module_weight_key": f"{module}.weight",
                }
            )
    return {"tensors": tensors}


def _projection_index(member: str) -> object:
    class Index:
        def geometry(self, *, layer, projection):
            return SimpleNamespace(
                codec="iqk",
                out_features=2560 if projection == "down_proj" else 640,
                in_features=768 if projection == "down_proj" else 2560,
                iqk_codec=member,
            )

    return Index()


def test_iq3_k_geometry_is_exactly_three_bit_110_bytes_without_row_meta() -> None:
    geometry = IQK_GEOMETRY["iq3_k"]

    assert geometry.bits == 3
    assert geometry.weights_per_block == 256
    assert geometry.bytes_per_block == 110
    assert geometry.row_meta_bytes == 0
    assert geometry.bytes_per_row(256) == 110
    assert geometry.bpw(256) == 3.4375
    assert iqk_geometry("iq3_k") is geometry


def test_loader_accepts_iq3_k_down_at_the_padded_qwen_width() -> None:
    entries = _qwen4_expert_entries(_manifest())
    params = entries[(2, "down_proj")]["format_params"]

    assert params["iqk_codec"] == "iq3_k"
    assert params["logical_shape"] == [2560, 640]
    assert params["stored_shape"] == [2560, 768]
    assert params["zero_padding"] == 128


def test_loader_rejects_iq3_k_down_at_native_width() -> None:
    manifest = _manifest()
    row = next(
        entry
        for entry in manifest["tensors"]
        if entry["layer_index"] == 2 and entry["projection"] == "down"
    )
    row["format_params"]["stored_shape"] = [2560, 640]
    row["format_params"]["zero_padding"] = 0

    with pytest.raises(Qwen4PackageLoadError, match="stored geometry"):
        _qwen4_expert_entries(manifest)


def test_projection_geometry_accepts_iq3_k_and_keeps_down_padded() -> None:
    codec, down_stored = _qwen4_projection_geometry(
        index=_projection_index("iq3_k"),
        layer=2,
        hidden_size=2560,
        intermediate_size=640,
    )

    assert codec == "iqk"
    assert down_stored == 768


def test_projection_geometry_rejects_unregistered_iqk_members() -> None:
    with pytest.raises(SSDStreamingBuildError, match="unsupported IQ_K member"):
        _qwen4_projection_geometry(
            index=_projection_index("iq4_k"),
            layer=2,
            hidden_size=2560,
            intermediate_size=640,
        )
