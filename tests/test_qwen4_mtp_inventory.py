"""MTP source contracts are independent of the unchanged target inventory."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from moespresso.inventory.qwen4.mtp import (
    Qwen4MTPSourceError,
    expected_qwen4_mtp_header_specs,
    resolve_qwen4_mtp_headers,
)
from moespresso.inventory.qwen4.roles import tensor_role
from moespresso.inventory.safetensors_header import TensorHeader
from moespresso.package.qwen4.mtp_sidecar import preflight_qwen4_mtp_sidecar


def _headers():
    return [
        TensorHeader(name, shape, dtype, "mtp.safetensors", 8, 0, math.prod(shape) * 2)
        for name, (dtype, shape) in expected_qwen4_mtp_header_specs().items()
    ]


def test_mtp_resolves_all_source_tensors_without_changing_target_roles():
    headers = _headers()
    resolved = resolve_qwen4_mtp_headers(headers)
    assert len(resolved) == 31
    assert all(tensor_role(header.name) is None for header in headers)
    by_name = {item.header.name: item for item in resolved}
    assert by_name["mtp.layers.0.self_attn.q_proj.weight"].module_path == "layers.0.mixer.module.q_proj"
    assert by_name["mtp.hyper_connection_mixer.hc_norm.weight"].module_path == "final_residual.hc_norm"
    assert by_name["mtp.fc_hidden.weight"].module_path == "fc_hidden"
    assert by_name["mtp.pre_fc_norm_hidden.weight"].kind == "passthrough"
    gate = by_name["mtp.layers.0.mlp.experts.gate_up_proj"]
    assert gate.kind == "expert"
    assert gate.projection == "gate_up"
    assert gate.module_path == "layers.0.mlp.experts"


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "shape", "dtype", "bytes"])
def test_mtp_inventory_rejects_source_drift(mutation):
    headers = _headers()
    if mutation == "missing":
        headers.pop()
    elif mutation == "extra":
        headers.append(replace(headers[0], name="mtp.unexpected.weight"))
    elif mutation == "duplicate":
        headers.append(headers[0])
    elif mutation == "shape":
        headers[0] = replace(headers[0], shape=(1,))
    elif mutation == "dtype":
        headers[0] = replace(headers[0], dtype="F16")
    else:
        headers[0] = replace(headers[0], end=headers[0].end - 1)
    with pytest.raises(Qwen4MTPSourceError):
        resolve_qwen4_mtp_headers(headers)


def test_mtp_preflight_prices_only_iq2_and_structural_passthrough(monkeypatch):
    resolved = resolve_qwen4_mtp_headers(_headers())
    monkeypatch.setattr(
        "moespresso.package.qwen4.mtp_sidecar.inspect_qwen4_mtp_source",
        lambda _path: ({"model_type": "qwen4_exp_text"}, resolved),
    )
    inventory = preflight_qwen4_mtp_sidecar("unused")
    assert inventory["target_payload_mutation"] is False
    assert inventory["calibration"] == "not_performed"
    assert inventory["byte_estimate"]["source_payload"] == 5_214_301_696
    by_name = {item["header"]["name"]: item for item in inventory["tensors"]}
    down = by_name["mtp.layers.0.mlp.experts.down_proj"]
    assert down["codec"] == "iq2_k"
    assert down["logical_in_features"] == 640
    assert down["stored_in_features"] == 768
    assert down["encoded_bytes"] == 512 * 2560 * 3 * 76
    assert {item["codec"] for item in inventory["tensors"] if item["format"] == "iqk"} == {"iq2_k"}
    assert inventory == preflight_qwen4_mtp_sidecar("unused")
