from __future__ import annotations

from copy import deepcopy

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from moespresso.runtime.qwen4.load import (
    Qwen4PackageLoadError,
    hydrate_qwen4_package_weights,
    validate_qwen4_direct_parameter_coverage,
)
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    iqk_dense_geometry,
)
from moespresso.package.iqk_dense_relayout import pack_dense_rows
from moespresso.runtime.qwen4.iqk_dense import (
    Qwen4IQKDenseError,
    install_qwen4_iqk_dense_modules,
    qwen4_iqk_dense_weight_map,
)


class _Graph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 32)
        self.linear = nn.Linear(32, 16, bias=False)
        self.conv = nn.Conv1d(4, 4, 3, groups=4, bias=False)


class _PackedLinear(nn.Module):
    def __init__(self, rows: int, row_bytes: int) -> None:
        super().__init__()
        self.weight = mx.zeros((rows, row_bytes), dtype=mx.uint8)
        self.scales = mx.zeros((1,), dtype=mx.uint8)


class _PackedGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q6 = _PackedLinear(4, 54)
        self.q8 = _PackedLinear(3, 34)


class _DenseIQKGraph(nn.Module):
    def __init__(self, in_features: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 4, bias=False)


def _package(tmp_path):
    embedding_weight = mx.arange(8 * 32, dtype=mx.bfloat16).reshape(8, 32)
    linear_weight = mx.arange(16 * 32, dtype=mx.bfloat16).reshape(16, 32)
    raw_conv = mx.arange(12, dtype=mx.bfloat16).reshape(4, 1, 3)
    mx.eval(
        embedding_weight,
        linear_weight,
        raw_conv,
    )
    shard = tmp_path / "model-00001-of-00001.safetensors"
    payload = {
        "embed.weight": embedding_weight,
        "head.weight": linear_weight,
        "conv": raw_conv,
    }
    mx.save_safetensors(str(shard), payload)
    manifest = {
        "tensors": [
            {
                "source_name": "model.language_model.embed_tokens.weight",
                "kind": "passthrough",
                "format": "raw_dtype_passthrough",
                "format_params": {},
                "module_path": "embedding",
                "module_weight_key": "embedding.weight",
                "shard": shard.name,
                "key_prefix": "embed.weight",
            },
            {
                "source_name": "lm_head.weight",
                "kind": "passthrough",
                "format": "raw_dtype_passthrough",
                "format_params": {},
                "module_path": "linear",
                "module_weight_key": "linear.weight",
                "shard": shard.name,
                "key_prefix": "head.weight",
            },
            {
                "source_name": "model.language_model.layers.1.ple.conv1d.weight",
                "kind": "passthrough",
                "format": "raw_dtype_passthrough",
                "format_params": {},
                "module_path": "conv",
                "module_weight_key": "conv.weight",
                "shard": shard.name,
                "key_prefix": "conv",
            },
            {
                "source_name": "model.language_model.layers.0.mlp.experts.gate_up_proj",
                "kind": "expert",
                "format": "iqk",
                "module_path": "experts",
                "shard": shard.name,
                "key_prefix": "experts",
            },
        ]
    }
    return manifest, payload


def test_qwen4_package_hydrates_direct_weights_by_manifest_key(tmp_path) -> None:
    manifest, payload = _package(tmp_path)
    model = _Graph()

    assert isinstance(model.embedding, nn.Embedding)
    assert isinstance(model.linear, nn.Linear)
    assert hydrate_qwen4_package_weights(model, manifest, tmp_path) == 3

    mx.eval(
        model.embedding.weight,
        model.linear.weight,
        model.conv.weight,
    )
    assert np.array_equal(
        np.asarray(model.embedding.weight.astype(mx.float32)),
        np.asarray(payload["embed.weight"].astype(mx.float32)),
    )
    assert np.array_equal(
        np.asarray(model.linear.weight.astype(mx.float32)),
        np.asarray(payload["head.weight"].astype(mx.float32)),
    )
    assert np.array_equal(
        np.asarray(model.conv.weight.view(mx.uint16)),
        np.swapaxes(np.asarray(payload["conv"].view(mx.uint16)), 1, 2),
    )


def test_qwen4_hydration_reads_each_shard_once(tmp_path) -> None:
    manifest, payload = _package(tmp_path)
    model = _Graph()
    calls = []

    def loader(path: str):
        calls.append(path)
        return payload

    hydrate_qwen4_package_weights(model, manifest, tmp_path, shard_loader=loader)

    assert len(calls) == 1


def test_qwen4_hydration_refuses_duplicate_parameter_destinations(tmp_path) -> None:
    manifest, _payload = _package(tmp_path)
    duplicate = deepcopy(manifest["tensors"][2])
    manifest["tensors"].append(duplicate)
    model = _Graph()

    with pytest.raises(Qwen4PackageLoadError, match="duplicate Qwen4 parameter"):
        hydrate_qwen4_package_weights(model, manifest, tmp_path)


def test_qwen4_hydration_refuses_missing_and_wrong_shape_payloads(tmp_path) -> None:
    manifest, payload = _package(tmp_path)
    model = _Graph()
    missing = dict(payload)
    del missing["conv"]
    with pytest.raises(Qwen4PackageLoadError, match="missing declared tensor key conv"):
        hydrate_qwen4_package_weights(
            model,
            manifest,
            tmp_path,
            shard_loader=lambda _path: missing,
        )

    wrong = dict(payload)
    wrong["conv"] = mx.zeros((4, 1, 2), dtype=mx.bfloat16)
    with pytest.raises(Qwen4PackageLoadError, match="destination expects"):
        hydrate_qwen4_package_weights(
            model,
            manifest,
            tmp_path,
            shard_loader=lambda _path: wrong,
        )


def test_qwen4_hydration_refuses_shard_path_escape(tmp_path) -> None:
    manifest, _payload = _package(tmp_path)
    manifest["tensors"][0]["shard"] = "../outside.safetensors"
    model = _Graph()

    with pytest.raises(Qwen4PackageLoadError, match="escapes the package root"):
        hydrate_qwen4_package_weights(model, manifest, tmp_path)


@pytest.mark.parametrize("format", ("affine", "mxfp4"))
def test_qwen4_hydration_refuses_unreleased_direct_formats(tmp_path, format) -> None:
    manifest, _payload = _package(tmp_path)
    manifest["tensors"][0]["format"] = format

    with pytest.raises(Qwen4PackageLoadError, match="unsupported Qwen4 direct format"):
        hydrate_qwen4_package_weights(_Graph(), manifest, tmp_path)


def test_qwen4_hydration_loads_direct_q6_and_q8_wire_tensors(tmp_path) -> None:
    graph = _PackedGraph()
    shard = tmp_path / "model.safetensors"
    payload = {
        "q6.weight": mx.arange(4 * 54, dtype=mx.uint8).reshape(4, 54),
        "q6.scales": mx.zeros((1,), dtype=mx.uint8),
        "q8.weight": mx.arange(3 * 34, dtype=mx.uint8).reshape(3, 34),
        "q8.scales": mx.zeros((1,), dtype=mx.uint8),
    }
    mx.save_safetensors(str(shard), payload)
    manifest = {
        "tensors": [
            {
                "source_name": "dense.q6.weight",
                "kind": "affine",
                "format": "kquant",
                "format_params": {"kquant_codec": "q6_k"},
                "module_path": "q6",
                "module_weight_key": "q6.weight",
                "shard": shard.name,
                "key_prefix": "q6",
            },
            {
                "source_name": "dense.q8.weight",
                "kind": "affine",
                "format": "kquant",
                "format_params": {"kquant_codec": "q8_0"},
                "module_path": "q8",
                "module_weight_key": "q8.weight",
                "shard": shard.name,
                "key_prefix": "q8",
            },
        ]
    }

    assert hydrate_qwen4_package_weights(graph, manifest, tmp_path) == 4
    validate_qwen4_direct_parameter_coverage(graph, manifest)
    mx.eval(graph.q6.weight, graph.q8.weight)

    assert np.array_equal(np.asarray(graph.q6.weight), np.asarray(payload["q6.weight"]))
    assert np.array_equal(np.asarray(graph.q8.weight), np.asarray(payload["q8.weight"]))


def test_qwen4_installs_and_hydrates_dense_iqk_by_manifest_key(tmp_path) -> None:
    row_bytes = iqk_dense_geometry("iq4_k").bytes_per_row(256)
    payload = {"dense.weight": mx.arange(4 * row_bytes, dtype=mx.uint8).reshape(4, row_bytes)}
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(str(shard), payload)
    manifest = {
        "tensors": [
            {
                "source_name": "dense.weight",
                "kind": "affine",
                "format": "iqk",
                "format_params": {
                    "iqk_codec": "iq4_k",
                    "layout": IQK_LAYOUT_IQK_RELAYOUT,
                },
                "module_path": "linear",
                "module_weight_key": "linear.weight",
                "shard": shard.name,
                "key_prefix": "dense",
            }
        ]
    }
    graph = _DenseIQKGraph()

    assert qwen4_iqk_dense_weight_map(manifest) == {
        "linear.weight": {
            "member": "iq4_k",
            "layout": IQK_LAYOUT_IQK_RELAYOUT,
            "source_name": "dense.weight",
        }
    }
    assert install_qwen4_iqk_dense_modules(graph, manifest) == 1
    assert graph.linear.mode == "iqk_dense"
    assert tuple(graph.linear.weight.shape) == (4, row_bytes)
    assert hydrate_qwen4_package_weights(graph, manifest, tmp_path) == 1
    validate_qwen4_direct_parameter_coverage(graph, manifest)
    mx.eval(graph.linear.weight)
    assert np.array_equal(
        np.asarray(graph.linear.weight),
        np.asarray(payload["dense.weight"]),
    )


def test_qwen4_dense_iqk_install_refuses_quantizer_wire_layout() -> None:
    manifest = {
        "tensors": [
            {
                "source_name": "dense.weight",
                "kind": "affine",
                "format": "iqk",
                "format_params": {
                    "iqk_codec": "iq4_k",
                    "layout": "ik_wire",
                },
                "module_path": "linear",
                "module_weight_key": "linear.weight",
            }
        ]
    }

    with pytest.raises(Qwen4IQKDenseError, match="requires"):
        install_qwen4_iqk_dense_modules(_DenseIQKGraph(), manifest)


@pytest.mark.parametrize("member", ("iq4_ks", "iq4_k", "iq5_k", "iq6_k"))
def test_qwen4_dense_iqk_decode_matches_its_dequantized_route(
    member: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_iqk import codec as iqk_codec

    rng = np.random.default_rng(73)
    in_features = 512
    weights = rng.standard_normal((4, in_features)).astype(np.float32) * 0.02
    importance = np.square(rng.standard_normal(in_features)).astype(np.float32) + 0.01
    wire = iqk_codec.quantize(member, weights, importance)
    packed = pack_dense_rows(member, wire, in_features)
    manifest = {
        "tensors": [
            {
                "source_name": "dense.weight",
                "kind": "affine",
                "format": "iqk",
                "format_params": {
                    "iqk_codec": member,
                    "layout": IQK_LAYOUT_IQK_RELAYOUT,
                },
                "module_path": "linear",
                "module_weight_key": "linear.weight",
            }
        ]
    }
    graph = _DenseIQKGraph(in_features)
    install_qwen4_iqk_dense_modules(graph, manifest)
    graph.load_weights([("linear.weight", mx.array(packed))], strict=False)
    values = mx.array(rng.standard_normal((1, 1, in_features)).astype(np.float32)).astype(
        mx.bfloat16
    )

    monkeypatch.setenv("MOESPRESSO_QWEN4_IQK_DENSE_QMV", "1")
    decode = graph.linear(values)
    mx.eval(decode)
    assert not hasattr(graph.linear, "_moespresso_iqk_streams")
    assert tuple(graph.linear.weight.shape) == packed.shape
    monkeypatch.setenv("MOESPRESSO_QWEN4_IQK_DENSE_QMV", "0")
    reference = graph.linear(values)
    mx.eval(reference)
    assert not hasattr(graph.linear, "_moespresso_iqk_streams")

    got = np.asarray(decode.astype(mx.float32))
    want = np.asarray(reference.astype(mx.float32))
    scale = max(float(np.max(np.abs(want))), 1e-6)
    assert float(np.max(np.abs(got - want))) / scale < 3e-3
