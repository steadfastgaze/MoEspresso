from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import moespresso.runtime.qwen4.load as load_module
from moespresso.package import iqk_relayout
from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata
from moespresso.package.kquant_format import KQUANT_GEOMETRY
from moespresso.package.iqk_format import (
    IQK_GEOMETRY,
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
)
from moespresso.runtime.expert_index import build_expert_index
from moespresso.runtime.qwen4.expert_provider import Qwen4PaddedPooledSwitchGLU
from moespresso.runtime.qwen4.expert_provider import Qwen4ZeroPaddedDownProjection
from moespresso.runtime.qwen4.expert_provider import build_qwen4_pooled_expert_executor
from moespresso.runtime.qwen4.load import (
    Qwen4PackageLoadError,
    _qwen4_non_routed_payload_bytes,
    _qwen4_expert_entries,
    _qwen4_stop_ids,
    load_qwen4_iqk_package_model,
    qwen4_kvarn_runtime_bytes,
    validate_qwen4_expert_package_contract,
)
from moespresso.runtime.qwen4.primitives import qwen4_projection_compute_dtype


def _architecture() -> dict:
    return {
        "family": "qwen4_exp",
        "text_model_type": "qwen4_exp_text",
        "config": {
            "model_type": "qwen4_exp_text",
            "dtype": "bfloat16",
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "hidden_act": "silu",
            "hc_count": 4,
            "hc_lowrank": 320,
            "head_dim": 256,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "norm_topk_prob": True,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ple_layer_ids": [2],
            "ngram_vocab_size_base": 20000000,
            "make_ngram_vocab_size_divisible_by": 128,
            "split_ngram_parts": 128,
            "seed": 1234,
            "eos_token_id": 248044,
            "rms_norm_eps": 1e-6,
            "output_gate_type": "sigmoid",
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
            "tie_word_embeddings": False,
            "layer_types": [
                "full_attention" if layer % 4 == 3 else "linear_attention"
                for layer in range(48)
            ],
        },
    }


def _manifest() -> dict:
    tensors = [
        {
            "source_name": "model.language_model.embed_tokens.weight",
            "kind": "affine",
            "format": "kquant",
            "format_params": {"kquant_codec": "q8_0"},
            "module_path": "embedding",
            "module_weight_key": "embedding.weight",
            "shard": "model.safetensors",
            "key_prefix": "embedding",
        },
        {
            "source_name": "model.language_model.layers.0.input_layernorm.weight",
            "kind": "passthrough",
            "format": "raw_dtype_passthrough",
            "format_params": {},
            "module_path": "layers.0.attention_residual.norm",
            "module_weight_key": "layers.0.attention_residual.norm.weight",
            "shard": "model.safetensors",
            "key_prefix": "norm",
        },
    ]
    for layer in range(48):
        for projection in ("gate", "up", "down"):
            module_projection = f"{projection}_proj"
            logical = [2560, 640] if projection == "down" else [640, 2560]
            if layer < 2:
                fmt = "kquant"
                params = {
                    "kquant_codec": "q8_0",
                    "logical_shape": logical,
                    "stored_shape": logical,
                    "zero_padding": 0,
                }
            else:
                fmt = "iqk"
                params = {
                    "iqk_codec": (
                        "iq2_ks" if (layer, projection) == (2, "gate") else "iq2_k"
                    ),
                    "layout": "iqk_relayout",
                    "logical_shape": logical,
                    "stored_shape": [2560, 768] if projection == "down" else logical,
                    "zero_padding": 128 if projection == "down" else 0,
                }
            module = f"layers.{layer}.mlp.experts.{module_projection}"
            tensors.append(
                {
                    "source_name": f"source.layers.{layer}.{projection}",
                    "kind": "expert",
                    "layer_index": layer,
                    "projection": projection,
                    "format": fmt,
                    "format_params": params,
                    "module_path": module,
                    "module_weight_key": f"{module}.weight",
                    "shard": "model.safetensors",
                    "key_prefix": f"experts.{layer}",
                }
            )
    return {
        "artifact_kind": "package_manifest",
        "artifact_id": "pkg:qwen4-loader-test",
        "status": "valid",
        "required_ops": [
            "iqk_dequant",
            "kquant_dequant",
            "raw_dtype_passthrough",
        ],
        "architecture": _architecture(),
        "tensors": tensors,
        "tokenizer": {
            "files": [
                {
                    "path": "generation_config.json",
                    "size_bytes": 32,
                    "sha256": "0" * 64,
                }
            ]
        },
    }


def _all_iq2_k_manifest() -> dict:
    manifest = _manifest()
    zero_experts = {0: [1, 7], 1: [3]}
    for entry in manifest["tensors"]:
        if entry.get("kind") != "expert":
            continue
        layer = int(entry["layer_index"])
        projection = str(entry["projection"])
        logical = [2560, 640] if projection == "down" else [640, 2560]
        params = {
            "iqk_codec": "iq2_k",
            "layout": "iqk_relayout",
            "logical_shape": logical,
            "stored_shape": [2560, 768] if projection == "down" else logical,
            "zero_padding": 128 if projection == "down" else 0,
            "calibration_policy": "per_expert_route_active",
        }
        if layer in zero_experts:
            params.update(
                {
                    "zero_count_experts": zero_experts[layer],
                    "zero_count_fallback_policy": (
                        "same_layer_projection_mean_normalized_route_active_v1"
                    ),
                }
            )
        entry["format"] = "iqk"
        entry["format_params"] = params
    return manifest


_EARLY_IQ3_ZERO_COUNT_PAIRS = (
    (0, 181),
    (0, 193),
    (0, 236),
    (0, 244),
    (0, 271),
    (0, 413),
    (0, 424),
    (0, 477),
    (1, 116),
)


def _early_iq3_k_manifest() -> dict:
    manifest = _all_iq2_k_manifest()
    zeros_by_layer = {
        layer: [
            expert
            for pair_layer, expert in _EARLY_IQ3_ZERO_COUNT_PAIRS
            if pair_layer == layer
        ]
        for layer in range(48)
    }
    for entry in manifest["tensors"]:
        if entry.get("kind") != "expert":
            continue
        layer = int(entry["layer_index"])
        params = entry["format_params"]
        params["iqk_codec"] = "iq3_k" if layer < 2 else "iq2_k"
        if zeros_by_layer[layer]:
            params["zero_count_experts"] = zeros_by_layer[layer]
        else:
            params.pop("zero_count_experts", None)
            params.pop("zero_count_fallback_policy", None)
    return manifest


def _with_compact_layout(manifest: dict, count: int = 448) -> dict:
    selection_id = "select:" + "b" * 64
    source_ids = list(range(count))
    manifest["required_features"] = ["qwen4_per_layer_experts"]
    manifest["inputs"] = [selection_id]
    manifest["expert_layout"] = {
        "per_layer_experts": {
            "source_selection_artifact_id": selection_id,
            "source_package_manifest_id": "pkg:" + "a" * 64,
            "source_num_experts": 512,
            "top_k": 10,
            "layers": {
                str(layer): {
                    "num_experts": count,
                    "source_expert_ids": source_ids,
                }
                for layer in range(48)
            },
        }
    }
    return manifest


class _Index:
    def layers_indexed(self):
        return tuple(range(48))

    def validate(self):
        return []

    def num_experts_for_layer(self, _layer):
        return 512

    def geometry(self, *, layer, projection):
        out_features = 2560 if projection == "down_proj" else 640
        if layer < 2:
            in_features = 640 if projection == "down_proj" else 2560
            return SimpleNamespace(
                codec="kquant",
                out_features=out_features,
                packed_cols=in_features // 32 * 34,
                bytes_per_block=34,
                weights_per_block=32,
                kquant_codec="q8_0",
                iqk_codec=None,
                layout=None,
                in_features=None,
            )
        return SimpleNamespace(
            codec="iqk",
            out_features=out_features,
            packed_cols=0,
            bytes_per_block=76,
            weights_per_block=256,
            kquant_codec=None,
            iqk_codec=(
                "iq2_ks" if (layer, projection) == (2, "gate_proj") else "iq2_k"
            ),
            layout="iqk_relayout",
            in_features=768 if projection == "down_proj" else 2560,
        )


class _AllIQ2Index(_Index):
    def geometry(self, *, layer, projection):
        out_features = 2560 if projection == "down_proj" else 640
        return SimpleNamespace(
            codec="iqk",
            out_features=out_features,
            packed_cols=0,
            bytes_per_block=76,
            weights_per_block=256,
            kquant_codec=None,
            iqk_codec="iq2_k",
            layout="iqk_relayout",
            in_features=768 if projection == "down_proj" else 2560,
        )


class _EarlyIQ3Index(_AllIQ2Index):
    def geometry(self, *, layer, projection):
        geometry = super().geometry(layer=layer, projection=projection)
        if layer >= 2:
            return geometry
        return SimpleNamespace(
            **{
                **vars(geometry),
                "bytes_per_block": 110,
                "iqk_codec": "iq3_k",
            }
        )


class _AllIQ2StreamMajorIndex(_AllIQ2Index):
    def geometry(self, *, layer, projection):
        geometry = super().geometry(layer=layer, projection=projection)
        return SimpleNamespace(
            **{
                **vars(geometry),
                "layout": IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
            }
        )


class _CompactIndex(_Index):
    def num_experts_for_layer(self, _layer):
        return 448


class _CompactMixedIndex(_CompactIndex):
    def geometry(self, *, layer, projection):
        out_features = 2560 if projection == "down_proj" else 640
        if (layer, projection) == (0, "gate_proj"):
            member = "iq1_s_r4"
            bytes_per_block = 6
            weights_per_block = 32
        elif (layer, projection) == (1, "gate_proj"):
            member = "iq2_ks"
            bytes_per_block = 84
            weights_per_block = 256
        else:
            member = "iq2_k"
            bytes_per_block = 76
            weights_per_block = 256
        return SimpleNamespace(
            codec="iqk",
            out_features=out_features,
            packed_cols=0,
            bytes_per_block=bytes_per_block,
            weights_per_block=weights_per_block,
            kquant_codec=None,
            iqk_codec=member,
            layout="iqk_relayout",
            in_features=(
                640
                if projection == "down_proj" and member == "iq1_s_r4"
                else 768
                if projection == "down_proj"
                else 2560
            ),
        )


class _NativeIQ1DownIndex(_Index):
    def geometry(self, *, layer, projection):
        if layer != 2 or projection != "down_proj":
            return super().geometry(layer=layer, projection=projection)
        return SimpleNamespace(
            codec="iqk",
            out_features=2560,
            packed_cols=0,
            bytes_per_block=6,
            weights_per_block=32,
            kquant_codec=None,
            iqk_codec="iq1_s_r4",
            layout="iqk_relayout",
            in_features=640,
        )


def _write_q8_executor_package(tmp_path):
    from conftest import write_safetensors_raw

    package = tmp_path / "q8-package"
    package.mkdir()
    geometry = KQUANT_GEOMETRY["q8_0"]
    components = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        components[(projection, "weight")] = np.zeros(
            (4, 256, 256 // geometry.weights_per_block * geometry.bytes_per_block),
            dtype=np.uint8,
        )
        components[(projection, "scales")] = np.zeros((4, 1), dtype=np.uint8)
    bundle, metadata = assemble_layer_bundle(
        components,
        {
            projection: geometry.bits
            for projection in ("gate_proj", "up_proj", "down_proj")
        },
        codecs={
            projection: "kquant"
            for projection in ("gate_proj", "up_proj", "down_proj")
        },
        kquant_codecs={
            projection: "q8_0"
            for projection in ("gate_proj", "up_proj", "down_proj")
        },
    )
    write_safetensors_raw(
        package / "model-00001-of-00001.safetensors",
        {
            "layers.0.mlp.experts.tq_bundle": (
                "U8",
                bundle.shape,
                bundle.tobytes(),
            )
        },
        metadata={"expert_bundles": encode_bundle_metadata({0: metadata})},
    )
    return package


def test_released_manifest_and_bundle_accept_q8_fallback_and_mixed_iqk_members() -> None:
    manifest = _manifest()
    entries = validate_qwen4_expert_package_contract(manifest, _Index())

    assert entries[(0, "down_proj")]["format_params"]["stored_shape"] == [2560, 640]
    assert entries[(1, "gate_proj")]["format_params"]["kquant_codec"] == "q8_0"
    assert entries[(2, "gate_proj")]["format_params"]["iqk_codec"] == "iq2_ks"
    assert entries[(2, "up_proj")]["format_params"]["iqk_codec"] == "iq2_k"
    assert entries[(47, "down_proj")]["format_params"]["zero_padding"] == 128


def test_compact_manifest_requires_bundle_counts_from_embedded_source_map() -> None:
    manifest = _with_compact_layout(_manifest())

    entries = validate_qwen4_expert_package_contract(manifest, _CompactIndex())
    assert len(entries) == 144

    with pytest.raises(Qwen4PackageLoadError, match="count mismatch at layer 0"):
        validate_qwen4_expert_package_contract(manifest, _Index())


def test_compact_manifest_accepts_mixed_iqk_when_calibration_holes_are_removed() -> None:
    manifest = _with_compact_layout(_all_iq2_k_manifest())
    experts = [entry for entry in manifest["tensors"] if entry.get("kind") == "expert"]
    next(
        entry
        for entry in experts
        if entry["layer_index"] == 0 and entry["projection"] == "gate"
    )["format_params"]["iqk_codec"] = "iq1_s_r4"
    next(
        entry
        for entry in experts
        if entry["layer_index"] == 1 and entry["projection"] == "gate"
    )["format_params"]["iqk_codec"] = "iq2_ks"
    layout = manifest["expert_layout"]["per_layer_experts"]["layers"]
    layout["0"]["source_expert_ids"] = [
        expert for expert in range(512) if expert not in {1, 7}
    ][:448]
    layout["1"]["source_expert_ids"] = [
        expert for expert in range(512) if expert != 3
    ][:448]

    entries = validate_qwen4_expert_package_contract(
        manifest,
        _CompactMixedIndex(),
    )
    assert entries[(0, "gate_proj")]["format_params"]["iqk_codec"] == "iq1_s_r4"
    assert entries[(1, "gate_proj")]["format_params"]["iqk_codec"] == "iq2_ks"

    broken = deepcopy(manifest)
    source_ids = broken["expert_layout"]["per_layer_experts"]["layers"]["0"][
        "source_expert_ids"
    ]
    source_ids[-1] = 1
    source_ids.sort()
    with pytest.raises(Qwen4PackageLoadError, match="retains unobserved"):
        validate_qwen4_expert_package_contract(broken, _CompactMixedIndex())


def test_released_manifest_accepts_native_width_iq1_down_cell() -> None:
    manifest = _manifest()
    down = next(
        entry
        for entry in manifest["tensors"]
        if entry.get("kind") == "expert"
        and entry["layer_index"] == 2
        and entry["projection"] == "down"
    )
    down["format_params"].update(
        {
            "iqk_codec": "iq1_s_r4",
            "stored_shape": [2560, 640],
            "zero_padding": 0,
        }
    )

    entries = validate_qwen4_expert_package_contract(
        manifest,
        _NativeIQ1DownIndex(),
    )
    params = entries[(2, "down_proj")]["format_params"]

    assert params["iqk_codec"] == "iq1_s_r4"
    assert params["stored_shape"] == [2560, 640]
    assert params["zero_padding"] == 0


def test_released_manifest_accepts_declared_all_iq2_k_zero_count_policy() -> None:
    entries = validate_qwen4_expert_package_contract(
        _all_iq2_k_manifest(), _AllIQ2Index()
    )

    assert entries[(0, "gate_proj")]["format_params"]["zero_count_experts"] == [
        1,
        7,
    ]
    assert entries[(1, "down_proj")]["format_params"]["zero_count_experts"] == [3]
    assert entries[(2, "gate_proj")]["format_params"]["iqk_codec"] == "iq2_k"


def test_released_manifest_accepts_exact_full_early_iq3_k_lattice() -> None:
    entries = validate_qwen4_expert_package_contract(
        _early_iq3_k_manifest(), _EarlyIQ3Index()
    )

    assert entries[(0, "gate_proj")]["format_params"]["iqk_codec"] == "iq3_k"
    assert entries[(1, "down_proj")]["format_params"]["stored_shape"] == [2560, 768]
    assert entries[(2, "gate_proj")]["format_params"]["iqk_codec"] == "iq2_k"


@pytest.mark.parametrize(
    "mutation",
    ("pair_missing", "pair_extra", "early_codec", "late_codec", "down_geometry"),
)
def test_released_manifest_refuses_early_iq3_k_lattice_drift(mutation: str) -> None:
    manifest = _early_iq3_k_manifest()
    experts = [entry for entry in manifest["tensors"] if entry.get("kind") == "expert"]
    if mutation in {"pair_missing", "pair_extra"}:
        for entry in experts:
            if entry["layer_index"] != 0:
                continue
            declared = entry["format_params"]["zero_count_experts"]
            entry["format_params"]["zero_count_experts"] = (
                declared[:-1] if mutation == "pair_missing" else [*declared, 500]
            )
    elif mutation == "early_codec":
        experts[0]["format_params"]["iqk_codec"] = "iq2_k"
    elif mutation == "late_codec":
        next(entry for entry in experts if entry["layer_index"] == 2)[
            "format_params"
        ]["iqk_codec"] = "iq3_k"
    else:
        down = next(
            entry
            for entry in experts
            if entry["layer_index"] == 0 and entry["projection"] == "down"
        )
        down["format_params"]["stored_shape"] = [2560, 640]
        down["format_params"]["zero_padding"] = 0

    with pytest.raises(
        Qwen4PackageLoadError,
        match=(
            "layer 0 down_proj: calibration holes"
            if mutation == "down_geometry"
            else "early-IQ3_K|six layer|stored geometry"
        ),
    ):
        _qwen4_expert_entries(manifest)


def test_released_manifest_and_bundle_accept_matching_stream_major_experts() -> None:
    manifest = _all_iq2_k_manifest()
    for entry in manifest["tensors"]:
        if entry.get("kind") == "expert":
            entry["format_params"]["layout"] = IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1

    entries = validate_qwen4_expert_package_contract(
        manifest,
        _AllIQ2StreamMajorIndex(),
    )
    assert {
        entry["format_params"]["layout"] for entry in entries.values()
    } == {IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1}

    with pytest.raises(Qwen4PackageLoadError, match="does not match manifest"):
        validate_qwen4_expert_package_contract(manifest, _AllIQ2Index())


@pytest.mark.parametrize("mutation", ("projection", "member", "policy"))
def test_released_manifest_refuses_all_iq2_k_policy_drift(mutation: str) -> None:
    manifest = _all_iq2_k_manifest()
    experts = [entry for entry in manifest["tensors"] if entry.get("kind") == "expert"]
    if mutation == "projection":
        next(
            entry
            for entry in experts
            if entry["layer_index"] == 0 and entry["projection"] == "up"
        )["format_params"]["zero_count_experts"] = [1]
    elif mutation == "member":
        experts[-1]["format_params"]["iqk_codec"] = "iq2_ks"
    else:
        experts[0]["format_params"]["zero_count_fallback_policy"] = "uniform"

    with pytest.raises(Qwen4PackageLoadError, match="zero-count|all 144"):
        validate_qwen4_expert_package_contract(manifest, _AllIQ2Index())


def test_layer_zero_q8_fallback_builds_the_same_pooled_graph_at_bounded_capacity(
    tmp_path,
) -> None:
    package = _write_q8_executor_package(tmp_path)
    executor = build_qwen4_pooled_expert_executor(
        package_dir=package,
        index=build_expert_index(package),
        layer=0,
        hidden_size=256,
        intermediate_size=256,
        num_experts=4,
        capacity=2,
    )

    assert executor.resolved_capacity == 2
    assert executor.gate_proj.kquant_type == "q8_0"
    assert executor.up_proj.pool is executor.gate_proj.pool
    assert executor.down_proj.kquant_type == "q8_0"
    assert executor.down_proj.in_features == 256
    pool = executor.gate_proj.pool
    with pool._bk_lock:
        pool._loads_inflight = 1
    with pytest.raises(RuntimeError, match="not quiescent"):
        executor.close()
    assert pool.weight is not None
    with pool._bk_lock:
        pool._loads_inflight = 0
    executor.close()
    assert pool.weight is None


def test_qwen4_full_resident_decode_refuses_training_and_dispatches_uint32(
    tmp_path,
    monkeypatch,
) -> None:
    package = _write_q8_executor_package(tmp_path)
    executor = build_qwen4_pooled_expert_executor(
        package_dir=package,
        index=build_expert_index(package),
        layer=0,
        hidden_size=256,
        intermediate_size=256,
        num_experts=4,
        capacity=4,
    )
    for pool in executor._unique_projection_pools(lockstep=True):
        pool.ensure(range(4))

    hidden = mx.ones((1, 1, 256), dtype=mx.bfloat16)
    indices = mx.array([[[3, 1]]], dtype=mx.int64)
    executor.train()
    assert executor.try_full_resident_decode(hidden, indices) is None
    assert executor.total_calls == 0
    assert executor.decode_calls == 0
    assert executor.barrier_free_decode_calls == 0

    executor.eval()
    dispatched = []
    original = executor.build_barrier_free_decode

    def capture(value, routed_indices):
        dispatched.append(routed_indices)
        return original(value, routed_indices)

    monkeypatch.setattr(executor, "build_barrier_free_decode", capture)
    output = executor.try_full_resident_decode(hidden, indices)
    assert output is not None
    mx.eval(output, *dispatched)

    assert output.shape == (1, 1, 2, 256)
    assert output.dtype == hidden.dtype
    assert len(dispatched) == 1
    assert dispatched[0].dtype == mx.uint32
    assert np.array_equal(np.asarray(dispatched[0]), np.array([[[3, 1]]], dtype=np.uint32))
    assert executor.total_calls == 1
    assert executor.decode_calls == 1
    assert executor.total_token_layers == 1
    assert executor.barrier_free_decode_calls == 1
    assert executor.barrier_free_prefill_calls == 0
    assert executor.index_sync_calls == 0
    assert executor.index_resync_calls == 0
    assert executor.direct_calls == 0
    executor.close()


def _value_iqk_wire(member: str, rows: int, in_features: int, seed: int) -> np.ndarray:
    geometry = IQK_GEOMETRY[member]
    rng = np.random.default_rng(seed)
    wire = rng.integers(
        0,
        256,
        size=(rows, geometry.bytes_per_row(in_features)),
        dtype=np.uint8,
    )
    scale_offsets = (
        [0]
        if member == "iq2_ks"
        else [block * geometry.bytes_per_block for block in range(in_features // 256)]
    )
    scales = (
        rng.standard_normal((rows, len(scale_offsets))).astype(np.float32) * 0.004
    ).astype(np.float16)
    scale_bytes = scales.view(np.uint8).reshape(rows, len(scale_offsets), 2)
    for offset_index, offset in enumerate(scale_offsets):
        wire[:, offset : offset + 2] = scale_bytes[:, offset_index]
    return wire


def _write_value_iqk_executor_package(tmp_path):
    from conftest import write_safetensors_raw

    package = tmp_path / "value-iqk-package"
    package.mkdir()
    members = {
        "gate_proj": "iq2_ks",
        "up_proj": "iq2_k",
        "down_proj": "iq2_k",
    }
    shapes = {
        "gate_proj": (640, 2560),
        "up_proj": (640, 2560),
        "down_proj": (2560, 768),
    }
    components = {}
    for seed, (projection, member) in enumerate(members.items(), start=811):
        out_features, in_features = shapes[projection]
        wire = _value_iqk_wire(
            member,
            4 * out_features,
            in_features,
            seed,
        )
        rows = iqk_relayout.pack_rows(member, wire, in_features)
        components[(projection, "blocks")] = rows.reshape(4, out_features, -1)
    bundle, metadata = assemble_layer_bundle(
        components,
        {
            projection: IQK_GEOMETRY[member].bits
            for projection, member in members.items()
        },
        codecs={projection: "iqk" for projection in members},
        iqk_codecs=members,
        iqk_layout=IQK_LAYOUT_IQK_RELAYOUT,
    )
    write_safetensors_raw(
        package / "model-00001-of-00001.safetensors",
        {
            "layers.2.mlp.experts.tq_bundle": (
                "U8",
                bundle.shape,
                bundle.tobytes(),
            )
        },
        metadata={"expert_bundles": encode_bundle_metadata({2: metadata})},
    )
    return package


def test_qwen4_padded_iqk_full_resident_decode_matches_incumbent_with_zero_lanes(
    tmp_path,
    monkeypatch,
) -> None:
    package = _write_value_iqk_executor_package(tmp_path)

    def build_executor():
        executor = build_qwen4_pooled_expert_executor(
            package_dir=package,
            index=build_expert_index(package),
            layer=2,
            hidden_size=2560,
            intermediate_size=640,
            num_experts=4,
            capacity=4,
        )
        for pool in executor._unique_projection_pools(lockstep=True):
            pool.ensure(range(4))
        executor.eval()
        return executor

    incumbent = build_executor()
    optional = build_executor()
    rng = np.random.default_rng(815)
    hidden = mx.array(
        (rng.standard_normal((1, 1, 2560)).astype(np.float32) * 0.05)
    ).astype(mx.bfloat16)
    indices = mx.array([[[3, 1, 0, 2]]], dtype=mx.int64)
    padded_operands = []
    inner_down = optional.down_proj.projection
    original_down = inner_down.matmul_slots

    def capture_down(value, routed_indices, *, sorted_indices=False):
        padded_operands.append(value)
        return original_down(
            value,
            routed_indices,
            sorted_indices=sorted_indices,
        )

    monkeypatch.setattr(inner_down, "matmul_slots", capture_down)
    expected = incumbent(hidden, indices)
    actual = optional.try_full_resident_decode(hidden, indices)
    assert actual is not None
    mx.eval(expected, actual, *padded_operands)

    expected_host = np.asarray(expected.astype(mx.float32))
    actual_host = np.asarray(actual.astype(mx.float32))
    assert np.count_nonzero(expected_host) > 0
    assert np.array_equal(actual_host, expected_host)
    assert len(padded_operands) == 1
    operand = np.asarray(padded_operands[0].astype(mx.float32))
    assert operand.shape[-1] == 768
    assert np.count_nonzero(operand[..., :640]) > 0
    assert np.array_equal(operand[..., 640:], np.zeros_like(operand[..., 640:]))
    assert optional.barrier_free_decode_calls == 1
    assert optional.barrier_free_prefill_calls == 0
    assert optional.index_sync_calls == 0
    assert optional.index_resync_calls == 0
    assert incumbent.barrier_free_decode_calls == 0
    assert incumbent.barrier_free_prefill_calls == 1
    incumbent.close()
    optional.close()


def test_mixed_iq2_ks_iq2_k_layer_builds_with_logical_down_padding(tmp_path) -> None:
    from conftest import write_safetensors_raw

    package = tmp_path / "iqk-package"
    package.mkdir()
    members = {
        "gate_proj": "iq2_ks",
        "up_proj": "iq2_k",
        "down_proj": "iq2_k",
    }
    shapes = {
        "gate_proj": (640, 2560),
        "up_proj": (640, 2560),
        "down_proj": (2560, 768),
    }
    components = {}
    for projection, member in members.items():
        out_features, in_features = shapes[projection]
        components[(projection, "blocks")] = np.zeros(
            (
                4,
                out_features,
                IQK_GEOMETRY[member].bytes_per_row(in_features),
            ),
            dtype=np.uint8,
        )
    bundle, metadata = assemble_layer_bundle(
        components,
        {projection: IQK_GEOMETRY[member].bits for projection, member in members.items()},
        codecs={projection: "iqk" for projection in members},
        iqk_codecs=members,
        iqk_layout=IQK_LAYOUT_IQK_RELAYOUT,
    )
    write_safetensors_raw(
        package / "model-00001-of-00001.safetensors",
        {
            "layers.2.mlp.experts.tq_bundle": (
                "U8",
                bundle.shape,
                bundle.tobytes(),
            )
        },
        metadata={"expert_bundles": encode_bundle_metadata({2: metadata})},
    )
    executor = build_qwen4_pooled_expert_executor(
        package_dir=package,
        index=build_expert_index(package),
        layer=2,
        hidden_size=2560,
        intermediate_size=640,
        num_experts=4,
        capacity=4,
    )

    assert executor.members == members
    assert executor.down_proj.in_features == 640
    assert executor.down_proj.stored_in_features == 768
    assert executor.down_proj.zero_padding == 128
    assert executor._iqk_sorted_parts() == 10

    indices = mx.array([0, 1, 2, 3], dtype=mx.uint32)
    rng = np.random.default_rng(771)
    for projection, input_width, range_rows in (
        (executor.gate_proj, 2560, 64),
        (executor.up_proj, 2560, 64),
        (executor.down_proj, 640, 256),
    ):
        projection.pool.ensure(range(4))
        slots = projection.pool.remap(indices)
        values = mx.array(
            rng.normal(size=(4, 1, input_width)).astype(np.float16)
        )
        full = projection.matmul_slots(values, slots, sorted_indices=True)
        decode = projection.matmul_slots(values, slots, sorted_indices=False)
        ranged = mx.concatenate(
            [
                projection.sorted_matmul_range(
                    values,
                    slots,
                    part * range_rows,
                    range_rows,
                )
                for part in range(10)
            ],
            axis=-1,
        )
        mx.eval(full, decode, ranged)
        assert full.shape[-1] == range_rows * 10
        assert np.array_equal(np.asarray(ranged), np.asarray(full))
        assert np.array_equal(np.asarray(decode), np.asarray(full))
    executor.close()


def test_native_width_iq1_down_layer_builds_without_padding(tmp_path) -> None:
    from conftest import write_safetensors_raw

    package = tmp_path / "iq1-down-package"
    package.mkdir()
    members = {
        "gate_proj": "iq2_ks",
        "up_proj": "iq2_k",
        "down_proj": "iq1_s_r4",
    }
    shapes = {
        "gate_proj": (640, 2560),
        "up_proj": (640, 2560),
        "down_proj": (2560, 640),
    }
    components = {}
    for projection, member in members.items():
        out_features, in_features = shapes[projection]
        components[(projection, "blocks")] = np.zeros(
            (
                4,
                out_features,
                IQK_GEOMETRY[member].bytes_per_row(in_features),
            ),
            dtype=np.uint8,
        )
    bundle, metadata = assemble_layer_bundle(
        components,
        {
            projection: IQK_GEOMETRY[member].bits
            for projection, member in members.items()
        },
        codecs={projection: "iqk" for projection in members},
        iqk_codecs=members,
        iqk_layout=IQK_LAYOUT_IQK_RELAYOUT,
    )
    write_safetensors_raw(
        package / "model-00001-of-00001.safetensors",
        {
            "layers.2.mlp.experts.tq_bundle": (
                "U8",
                bundle.shape,
                bundle.tobytes(),
            )
        },
        metadata={"expert_bundles": encode_bundle_metadata({2: metadata})},
    )

    index = build_expert_index(package)
    assert {
        projection: (
            index.geometry(layer=2, projection=projection).iqk_codec,
            index.geometry(layer=2, projection=projection).in_features,
        )
        for projection in members
    } == {
        "gate_proj": ("iq2_ks", 2560),
        "up_proj": ("iq2_k", 2560),
        "down_proj": ("iq1_s_r4", 640),
    }
    executor = build_qwen4_pooled_expert_executor(
        package_dir=package,
        index=index,
        layer=2,
        hidden_size=2560,
        intermediate_size=640,
        num_experts=4,
        capacity=4,
    )

    assert executor.members == members
    assert executor.down_proj.in_features == 640
    assert not isinstance(executor.down_proj, Qwen4ZeroPaddedDownProjection)
    assert executor.down_proj.member == "iq1_s_r4"

    executor.down_proj.pool.ensure(range(4))
    indices = mx.array([0, 1, 2, 3], dtype=mx.uint32)
    slots = executor.down_proj.pool.remap(indices)
    values = mx.zeros((4, 1, 640), dtype=mx.float16)
    output = executor.down_proj.matmul_slots(values, slots)
    mx.eval(output)
    assert output.shape == (4, 1, 2560)
    assert np.array_equal(
        np.asarray(output),
        np.zeros((4, 1, 2560), dtype=np.float16),
    )
    executor.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("logical_shape", [2560, 768], "logical shape"),
        ("stored_shape", [2560, 640], "stored geometry"),
        ("zero_padding", 0, "stored geometry"),
        ("layout", "ik_wire", "stored geometry"),
    ],
)
def test_manifest_refuses_inconsistent_iqk_down_geometry(field, value, message) -> None:
    manifest = _manifest()
    row = next(
        tensor
        for tensor in manifest["tensors"]
        if tensor.get("layer_index") == 2 and tensor.get("projection") == "down"
    )
    row["format_params"][field] = value

    with pytest.raises(Qwen4PackageLoadError, match=message):
        _qwen4_expert_entries(manifest)


def test_manifest_refuses_padded_iq1_down_geometry() -> None:
    manifest = _manifest()
    row = next(
        tensor
        for tensor in manifest["tensors"]
        if tensor.get("layer_index") == 2 and tensor.get("projection") == "down"
    )
    row["format_params"]["iqk_codec"] = "iq1_s_r4"

    with pytest.raises(Qwen4PackageLoadError, match="stored geometry"):
        _qwen4_expert_entries(manifest)


class _CapturedProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 768
        self.out_features = 2560
        self.num_experts = 512
        self.pool = object()
        self.codec = "iqk"
        self.bits = 2
        self.member = "iq2_k"
        self.matmul_slot_calls = 0
        self.matmul_slot_elements = 0
        self.operands = []

    def matmul_slots(self, value, indices, *, sorted_indices=False):
        del indices, sorted_indices
        self.operands.append(value)
        return value

    def sorted_matmul_range(self, value, indices, start, rows):
        del indices, start, rows
        self.operands.append(value)
        return value

    def __call__(self, value, indices, *, sorted_indices=False):
        return self.matmul_slots(value, indices, sorted_indices=sorted_indices)


def _two_dispatch_executor(
    *,
    num_experts: int = 256,
    members: tuple[str, str, str] = ("iq2_k", "iq2_k", "iq2_k"),
) -> Qwen4PaddedPooledSwitchGLU:
    executor = Qwen4PaddedPooledSwitchGLU.__new__(Qwen4PaddedPooledSwitchGLU)
    down = Qwen4ZeroPaddedDownProjection.__new__(Qwen4ZeroPaddedDownProjection)
    object.__setattr__(down, "stored_in_features", 768)
    for name, value in {
        "_all_iqk": True,
        "_iqk_two_dispatch_ready_cached": None,
        "members": dict(
            zip(("gate_proj", "up_proj", "down_proj"), members, strict=True)
        ),
        "hidden_size": 2560,
        "intermediate_size": 640,
        "num_experts": num_experts,
        "down_proj": down,
    }.items():
        object.__setattr__(executor, name, value)
    return executor


@pytest.mark.parametrize("num_experts", [256, 320, 512])
def test_iqk_two_dispatch_accepts_compact_released_expert_counts(num_experts) -> None:
    executor = _two_dispatch_executor(num_experts=num_experts)

    assert executor._iqk_two_dispatch_ready()


@pytest.mark.parametrize(
    "members",
    [
        ("iq2_k", "iq2_k", "iq2_k"),
        ("iq3_k", "iq3_k", "iq2_k"),
        ("iq3_k", "iq3_k", "iq3_k"),
    ],
)
def test_iqk_two_dispatch_accepts_declared_compile_time_tuples(members) -> None:
    assert _two_dispatch_executor(members=members)._iqk_two_dispatch_ready()


@pytest.mark.parametrize(
    "members",
    [
        ("iq2_k", "iq3_k", "iq2_k"),
        ("iq3_k", "iq2_k", "iq3_k"),
        ("iq2_ks", "iq2_ks", "iq2_k"),
    ],
)
def test_iqk_two_dispatch_refuses_unsupported_codec_tuples(members) -> None:
    assert not _two_dispatch_executor(members=members)._iqk_two_dispatch_ready()


@pytest.mark.parametrize("num_experts", [9, 513])
def test_iqk_two_dispatch_refuses_expert_counts_outside_source_geometry(
    num_experts,
) -> None:
    executor = _two_dispatch_executor(num_experts=num_experts)

    assert not executor._iqk_two_dispatch_ready()


def test_iqk_down_projection_appends_exact_zero_lanes_for_every_entrypoint() -> None:
    inner = _CapturedProjection()
    padded = Qwen4ZeroPaddedDownProjection(inner, logical_in_features=640)
    values = mx.arange(2 * 640, dtype=mx.float32).reshape(2, 640)
    indices = mx.array([0, 1], dtype=mx.uint32)

    padded(values, indices)
    padded.matmul_slots(values, indices)
    padded.sorted_matmul_range(values, indices, 0, 32)
    mx.eval(*inner.operands)

    for operand in inner.operands:
        host = np.asarray(operand)
        assert host.shape == (2, 768)
        assert np.array_equal(host[:, :640], np.asarray(values))
        assert np.array_equal(host[:, 640:], np.zeros((2, 128), dtype=np.float32))


class _Provider:
    def __init__(self, layout):
        self.layout = layout
        self.closed = False

    def close(self):
        self.closed = True


class _Graph:
    def __init__(self):
        self.evaluated = False
        self.layers = [SimpleNamespace(mlp=SimpleNamespace()) for _ in range(48)]

    def parameters(self):
        return []

    def eval(self):
        self.evaluated = True


class _LoaderExecutor:
    def __init__(self, capacity, layer, closed_experts):
        self.resolved_capacity = capacity
        self.layer = layer
        self.closed_experts = closed_experts
        self.pools = tuple(
            SimpleNamespace(capacity=capacity, num_experts=512) for _ in range(3)
        )

    def _projection_pools_lockstep(self):
        return self.pools

    def close(self):
        self.closed_experts.append(self.layer)


@pytest.mark.parametrize("capacity", [512, 37])
@pytest.mark.parametrize("routing_options", [
    {},
    {"cache_routing": "prefer-resident", "cache_routing_factor": 4,
     "cache_routing_protected_routes": 0},
])
def test_top_level_loader_uses_one_graph_for_full_and_bounded_residency(
    monkeypatch,
    tmp_path,
    capacity,
    routing_options,
) -> None:
    manifest = _manifest()
    generation = b'{"eos_token_id":[248046,248044]}'
    (tmp_path / "generation_config.json").write_bytes(generation)
    manifest["tokenizer"]["files"][0]["size_bytes"] = len(generation)
    provider_box = {}
    expert_calls = []
    closed_experts = []
    executors = []
    graph_args = {}
    seed_calls = []
    routing_calls = []
    monkeypatch.setattr(
        "moespresso.runtime.qwen4.cache_routing.configure_cache_routing",
        lambda model, policy, **kwargs: routing_calls.append((model, policy, kwargs)),
    )

    def provider_factory(layout):
        provider_box["provider"] = _Provider(layout)
        return provider_box["provider"]

    def expert_builder(**kwargs):
        expert_calls.append(kwargs)
        executor = _LoaderExecutor(kwargs["capacity"], kwargs["layer"], closed_experts)
        executors.append(executor)
        return executor

    def graph_builder(_manifest, **kwargs):
        graph_args.update(kwargs)
        for layer in range(48):
            kwargs["expert_factory"](layer, 2560, 640, 512)
        return _Graph()

    def seed_expert_residency(model, package_dir):
        seed_calls.append((model, package_dir))
        return {"source": "test", "seeded": capacity * 48 * 3}

    monkeypatch.setattr(load_module.mx, "eval", lambda *_args: None)
    model, tokenizer = load_qwen4_iqk_package_model(
        manifest,
        tmp_path,
        capacity_per_layer=capacity,
        max_context_tokens=4096,
        build_index_fn=lambda _root: _Index(),
        expert_builder=expert_builder,
        graph_builder=graph_builder,
        install_kquant_fn=lambda _model, _manifest: 1,
        validate_direct_coverage_fn=lambda *_args: None,
        hydrate_fn=lambda *_args: 0,
        parse_ple_fn=lambda *_args, **_kwargs: "PLE-LAYOUT",
        ple_provider_factory=provider_factory,
        load_tokenizer_fn=lambda _root, *, eos_token_ids: ("TOK", eos_token_ids),
        seed_expert_residency_fn=seed_expert_residency,
        **routing_options,
    )

    assert model.evaluated
    assert routing_calls == (
        [(model, "prefer-resident", {"factor": 4, "protected_routes": 0})]
        if routing_options else (
            [(model, "prefer-resident", {"factor": 2.0, "protected_routes": 2})]
            if capacity < 512 else []
        )
    )
    session = getattr(model, "_moespresso_pooled_decode_session", None)
    assert session is not None
    assert model._moespresso_owns_pooled_request_scope is True
    assert model.layers[-1].mlp.pipeline_is_last is True
    for executor in executors:
        assert getattr(executor, "_moespresso_pooled_decode_session", None) is session
    assert model._moespresso_pooled_decode_bounded is (capacity < 512)
    assert not session.active and not session.pending
    assert tokenizer == ("TOK", {248044, 248046})
    assert len(expert_calls) == 48
    assert {call["capacity"] for call in expert_calls} == {capacity}
    assert graph_args["ple_provider"] is provider_box["provider"]
    assert graph_args["qsa_max_query_tokens"] == 64
    assert callable(graph_args["qsa_state_backend_factory"])
    assert seed_calls == [(model, tmp_path)]
    assert model._moespresso_ssd_hotlist == {
        "source": "test",
        "seeded": capacity * 48 * 3,
    }
    assert model._moespresso_ssd_streaming_capacity == capacity
    assert model._moespresso_generation_adapter == "qwen4_composite_v1"
    assert model._moespresso_qwen4_stop_ids == {248044, 248046}
    assert set(model._moespresso_ssd_streaming_resolved_capacities.values()) == {
        capacity
    }
    model._moespresso_qwen4_runtime_resources.close()
    assert provider_box["provider"].closed
    assert closed_experts == list(range(48))


def test_loader_closes_runtime_resources_when_expert_seeding_fails(
    monkeypatch,
    tmp_path,
) -> None:
    provider = _Provider("layout")
    closed_experts = []

    def expert_builder(**kwargs):
        return _LoaderExecutor(kwargs["capacity"], kwargs["layer"], closed_experts)

    def graph_builder(_manifest, **kwargs):
        for layer in range(48):
            kwargs["expert_factory"](layer, 2560, 640, 512)
        return _Graph()

    monkeypatch.setattr(load_module.mx, "eval", lambda *_args: None)
    with pytest.raises(RuntimeError, match="seed failed"):
        load_qwen4_iqk_package_model(
            _manifest(),
            tmp_path,
            capacity_per_layer=512,
            max_context_tokens=4096,
            build_index_fn=lambda _root: _Index(),
            expert_builder=expert_builder,
            graph_builder=graph_builder,
            install_kquant_fn=lambda _model, _manifest: 1,
            validate_direct_coverage_fn=lambda *_args: None,
            hydrate_fn=lambda *_args: 0,
            parse_ple_fn=lambda *_args, **_kwargs: "layout",
            ple_provider_factory=lambda _layout: provider,
            seed_expert_residency_fn=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("seed failed")
            ),
        )

    assert provider.closed
    assert closed_experts == list(range(48))


def test_loader_closes_ple_provider_when_graph_construction_fails(tmp_path) -> None:
    provider = _Provider("layout")

    with pytest.raises(RuntimeError, match="graph failed"):
        load_qwen4_iqk_package_model(
            _manifest(),
            tmp_path,
            capacity_per_layer=32,
            max_context_tokens=4096,
            build_index_fn=lambda _root: _Index(),
            graph_builder=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("graph failed")
            ),
            parse_ple_fn=lambda *_args, **_kwargs: "layout",
            ple_provider_factory=lambda _layout: provider,
        )

    assert provider.closed


def test_loader_refuses_invalid_manifest_before_opening_ple(tmp_path) -> None:
    manifest = deepcopy(_manifest())
    manifest["required_ops"].append("future_op")
    opened = []

    with pytest.raises(Qwen4PackageLoadError, match="required_ops"):
        load_qwen4_iqk_package_model(
            manifest,
            tmp_path,
            capacity_per_layer=32,
            parse_ple_fn=lambda *_args, **_kwargs: opened.append(True),
        )

    assert opened == []


def test_qwen4_kvarn_budget_is_explicit() -> None:
    assert qwen4_kvarn_runtime_bytes(131072) > qwen4_kvarn_runtime_bytes(4096) > 0


def test_qwen4_capacity_budget_uses_live_memory_ceiling(monkeypatch, tmp_path) -> None:
    from moespresso.runtime import ssd_streaming_build, streaming_capacity

    monkeypatch.delenv("MOESPRESSO_SSD_KV_ALLOWANCE_GB", raising=False)
    calls = []

    def resolved_available_bytes(**kwargs):
        calls.append(kwargs)
        return 9 << 30, {"limiting_source": "live-available"}

    monkeypatch.setattr(
        ssd_streaming_build, "_resolved_available_bytes", resolved_available_bytes,
    )
    monkeypatch.setattr(load_module, "_qwen4_non_routed_payload_bytes", lambda _: 100)
    monkeypatch.setattr(load_module, "qwen4_kvarn_runtime_bytes", lambda _: 100)
    monkeypatch.setattr(streaming_capacity, "bytes_per_capacity_unit", lambda _: 100)
    monkeypatch.setattr(streaming_capacity, "full_resident_expert_bytes", lambda _: 1000)
    monkeypatch.setattr(streaming_capacity, "min_capacity", lambda **_: 12)
    resolution = {}

    budget = load_module._qwen4_capacity_budget(
        index=object(), package_dir=tmp_path, max_context_tokens=4096,
        resolution_out=resolution,
    )

    assert calls == [{"strict_live_available": True}]
    assert budget.available_bytes == 9 << 30
    assert budget.kv_activation_allowance_bytes == 2 << 30
    assert resolution == {"limiting_source": "live-available"}


def test_qwen4_stop_policy_uses_the_package_generation_contract(tmp_path) -> None:
    manifest = _manifest()
    payload = b'{"eos_token_id":[248046,248044]}'
    (tmp_path / "generation_config.json").write_bytes(payload)
    manifest["tokenizer"]["files"][0]["size_bytes"] = len(payload)

    assert _qwen4_stop_ids(
        manifest,
        tmp_path,
        manifest["architecture"]["config"],
    ) == {248044, 248046}

    (tmp_path / "generation_config.json").write_text('{"eos_token_id":248044}')
    manifest["tokenizer"]["files"][0]["size_bytes"] = (
        tmp_path / "generation_config.json"
    ).stat().st_size
    with pytest.raises(Qwen4PackageLoadError, match="stop ids"):
        _qwen4_stop_ids(
            manifest,
            tmp_path,
            manifest["architecture"]["config"],
        )


def test_qwen4_resident_base_excludes_all_48_expert_bundles(tmp_path) -> None:
    from conftest import write_bundle_package

    package = tmp_path / "package"
    package.mkdir()
    write_bundle_package(
        package,
        layers=tuple(range(48)),
        n_exp=1,
        out=2,
        cols=1,
        extra_tensors={"layers.0.norm.weight": ("F16", (4,), bytes(8))},
    )

    assert _qwen4_non_routed_payload_bytes(package) == 8


def test_kquant_wire_dtype_does_not_replace_the_bf16_compute_contract() -> None:
    packed = SimpleNamespace(mode="kquant", weight=mx.zeros((2, 34), dtype=mx.uint8))
    plain = SimpleNamespace(weight=mx.zeros((2, 32), dtype=mx.float16))

    assert qwen4_projection_compute_dtype(packed) == mx.bfloat16
    assert qwen4_projection_compute_dtype(plain) == mx.float16
