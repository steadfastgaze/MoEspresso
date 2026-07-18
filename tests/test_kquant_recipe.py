from __future__ import annotations

import struct
from unittest.mock import patch

import numpy as np
import pytest

from moespresso.package.deepseek_v4.recipe import (
    DS4KQuantExpertTarget,
    build_ds4_expert_kquant_targets,
    build_ds4_kquant_expert_allocations,
    build_ds4_kquant_plan,
)
from moespresso.package.kquant_recipe import (
    KQuantRecipeError,
    read_gguf_kquant_recipe,
    read_gguf_tensor_types,
    validate_kquant_target_fit,
)
from moespresso.package.qwen.recipe import (
    build_dense_kquant_allocations as build_qwen_dense_kquant_allocations,
    build_dense_kquant_targets as build_qwen_dense_kquant_targets,
    build_expert_kquant_allocations as build_qwen_expert_kquant_allocations,
    build_expert_kquant_targets as build_qwen_expert_kquant_targets,
    build_f32_passthrough as build_qwen_f32_passthrough,
    build_kquant_plan as build_qwen_kquant_plan,
    expert_logical_shape as qwen_expert_logical_shape,
    module_path as qwen_module_path,
    source_gguf_key as qwen_source_gguf_key,
)
from moespresso.core.artifact import validate_base
from moespresso.probe.gguf_parse import GGUF_MAGIC, read_gguf_metadata


def _gguf_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_recipe_gguf(path, tensors):
    header = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), 0)
    infos = bytearray()
    for offset, (name, type_id, dims) in enumerate(tensors):
        infos += _gguf_string(name)
        infos += struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<I", type_id)
        infos += struct.pack("<Q", offset)
    path.write_bytes(header + bytes(infos))


def test_reads_gguf_metadata_without_tensor_payload(tmp_path):
    path = tmp_path / "recipe.gguf"
    _write_recipe_gguf(path, [
        ("blk.3.ffn_down_exps.weight", 10, [2048, 4096]),
        ("blk.3.ffn_gate_exps.weight", 16, [4096, 2048]),
    ])

    metadata = read_gguf_metadata(path, chunk_bytes=17)

    assert metadata.header.tensor_count == 2
    assert [t.name for t in metadata.tensor_infos] == [
        "blk.3.ffn_down_exps.weight",
        "blk.3.ffn_gate_exps.weight",
    ]


def test_reads_gguf_kquant_recipe_and_skips_float_tensors(tmp_path):
    path = tmp_path / "recipe.gguf"
    _write_recipe_gguf(path, [
        ("blk.3.ffn_down_exps.weight", 10, [2048, 4096]),  # Q2_K
        ("blk.3.ffn_gate_exps.weight", 16, [4096, 2048]),  # IQ2_XXS
        ("blk.3.attn_output.weight", 8, [4096, 4096]),     # Q8_0
        ("blk.3.attn_norm.weight", 0, [4096]),             # F32 has no codec recipe
    ])

    recipe = read_gguf_kquant_recipe(path)

    assert recipe == {
        "blk.3.ffn_down_exps.weight": "q2_k",
        "blk.3.ffn_gate_exps.weight": "iq2_xxs",
        "blk.3.attn_output.weight": "q8_0",
    }


def test_reads_gguf_kquant_recipe_from_hf_url_with_range_requests():
    gguf_data = struct.pack("<IIQQ", GGUF_MAGIC, 3, 2, 0)
    infos = bytearray()
    for offset, (name, type_id, dims) in enumerate([
        ("blk.3.ffn_down_exps.weight", 10, [2048, 4096]),
        ("blk.3.ffn_gate_exps.weight", 16, [4096, 2048]),
    ]):
        infos += _gguf_string(name)
        infos += struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<I", type_id)
        infos += struct.pack("<Q", offset)
    gguf_data += bytes(infos)
    requested: list[tuple[str, int, int]] = []

    def fake_fetch_range(url: str, start: int, end: int) -> bytes:
        requested.append((url, start, end))
        return gguf_data[start:end + 1]

    with patch("moespresso.inventory.hf_inspect._fetch_range", fake_fetch_range):
        recipe = read_gguf_kquant_recipe(
            "https://huggingface.co/org/repo/blob/main/model.gguf")

    assert requested[0][0] == "https://huggingface.co/org/repo/resolve/main/model.gguf"
    assert recipe == {
        "blk.3.ffn_down_exps.weight": "q2_k",
        "blk.3.ffn_gate_exps.weight": "iq2_xxs",
    }


def test_reads_gguf_tensor_types_including_f32_and_kquant(tmp_path):
    path = tmp_path / "recipe.gguf"
    _write_recipe_gguf(path, [
        ("blk.3.ssm_alpha.weight", 0, [2048, 32]),  # F32
        ("blk.3.attn_output.weight", 13, [4096, 4096]),  # Q5_K
        ("token_embd.weight", 12, [2048, 248320]),  # Q4_K
    ])

    tensor_types = read_gguf_tensor_types(path)

    assert tensor_types == {
        "blk.3.ssm_alpha.weight": "F32",
        "blk.3.attn_output.weight": "Q5_K",
        "token_embd.weight": "Q4_K",
    }


def test_gguf_recipe_fails_closed_on_unsupported_quant_codec(tmp_path):
    path = tmp_path / "recipe.gguf"
    _write_recipe_gguf(path, [
        ("blk.0.bad.weight", 15, [4096, 4096]),  # unsupported mlx-kquant Q8_K
    ])

    with pytest.raises(KQuantRecipeError, match="unsupported GGUF codec 'Q8_K'"):
        read_gguf_kquant_recipe(path)


def test_gguf_recipe_fails_closed_on_unknown_tensor_type(tmp_path):
    path = tmp_path / "recipe.gguf"
    _write_recipe_gguf(path, [
        ("blk.0.bad.weight", 999, [4096, 4096]),
    ])

    with pytest.raises(KQuantRecipeError, match="unknown GGUF tensor type id 999"):
        read_gguf_kquant_recipe(path)


def test_maps_ds4_gguf_expert_recipe_to_source_templates_and_module_keys():
    recipe = {
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }

    targets = build_ds4_expert_kquant_targets(recipe, required_layers=[7])

    by_projection = {target.projection: target for target in targets}
    assert by_projection["gate"].source_weight_template == (
        "layers.7.ffn.experts.{expert}.w1.weight")
    assert by_projection["up"].source_weight_template == (
        "layers.7.ffn.experts.{expert}.w3.weight")
    assert by_projection["down"].source_weight_template == (
        "layers.7.ffn.experts.{expert}.w2.weight")
    assert by_projection["down"].source_scale_template == (
        "layers.7.ffn.experts.{expert}.w2.scale")
    assert by_projection["down"].module_weight_key == (
        "model.layers.7.mlp.switch_mlp.down_proj.weight")
    assert by_projection["down"].imatrix_key == "blk.7.ffn_down_exps.weight"
    assert by_projection["down"].codec == "q2_k"


def test_ds4_mapping_fails_closed_when_required_projection_is_missing():
    recipe = {
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }

    with pytest.raises(KQuantRecipeError, match="blk.7.ffn_up_exps.weight"):
        build_ds4_expert_kquant_targets(recipe, required_layers=[7])


def test_kquant_fit_accepts_valid_q2k_down_projection_with_matching_imatrix():
    target = build_ds4_expert_kquant_targets({
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }, required_layers=[7])[2]

    validate_kquant_target_fit(
        target,
        (4096, 2048),
        {"blk.7.ffn_down_exps.weight": np.ones(2048, dtype=np.float32)},
    )


def test_kquant_fit_rejects_bad_superblock_width():
    target = build_ds4_expert_kquant_targets({
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }, required_layers=[7])[2]

    with pytest.raises(KQuantRecipeError, match="in_features 2050 is not divisible"):
        validate_kquant_target_fit(
            target,
            (4096, 2050),
            {"blk.7.ffn_down_exps.weight": np.ones(2050, dtype=np.float32)},
        )


def test_kquant_fit_rejects_transposed_imatrix_evidence():
    target = build_ds4_expert_kquant_targets({
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }, required_layers=[7])[2]

    with pytest.raises(KQuantRecipeError, match="check tensor orientation"):
        validate_kquant_target_fit(
            target,
            (4096, 2048),
            {"blk.7.ffn_down_exps.weight": np.ones(4096, dtype=np.float32)},
        )


def test_kquant_fit_requires_imatrix_for_iq2xxs_gate_projection():
    target = build_ds4_expert_kquant_targets({
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }, required_layers=[7])[0]

    with pytest.raises(KQuantRecipeError, match="missing imatrix vector"):
        validate_kquant_target_fit(target, (2048, 4096), {})


def test_kquant_fit_does_not_require_imatrix_for_q8_block_codec():
    target = DS4KQuantExpertTarget(
        layer_index=7,
        projection="down",
        codec="q8_0",
        gguf_tensor="blk.7.attn_output.weight",
        imatrix_key="blk.7.attn_output.weight",
        source_weight_template="layers.7.attn.wo_b.weight",
        source_scale_template="layers.7.attn.wo_b.scale",
        module_path="model.layers.7.self_attn.wo_b",
        module_weight_key="model.layers.7.self_attn.wo_b.weight",
    )

    validate_kquant_target_fit(target, (4096, 4096), {})


def test_builds_ds4_kquant_expert_allocations_for_package_manifest():
    targets = build_ds4_expert_kquant_targets({
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }, required_layers=[7])

    allocation = build_ds4_kquant_expert_allocations(targets)

    assert [a["projection"] for a in allocation] == ["gate", "up", "down"]
    gate, _up, down = allocation
    assert gate["source_name"] == "layers.7.ffn.experts.gate"
    assert gate["role"] == "moe.expert.gate"
    assert gate["format"] == "kquant"
    assert gate["codec"] == gate["kquant_codec"] == "iq2_xxs"
    assert gate["bits"] == 2
    assert gate["imatrix_key"] == "blk.7.ffn_gate_exps.weight"
    assert gate["module_weight_key"] == (
        "model.layers.7.mlp.switch_mlp.gate_proj.weight"
    )
    assert down["source_name"] == "layers.7.ffn.experts.down"
    assert down["codec"] == down["kquant_codec"] == "q2_k"
    assert down["source_weight_template"] == "layers.7.ffn.experts.{expert}.w2.weight"


def test_builds_ds4_kquant_plan_artifact_from_recipe_targets():
    targets = build_ds4_expert_kquant_targets({
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }, required_layers=[7])
    subject = {"source_root": "/models/ds4", "source_format": "hf_safetensors"}
    imatrix_identity = {"name": "imatrix.dat", "sha256": "abc", "key_count": 129}

    plan = build_ds4_kquant_plan(
        subject,
        targets,
        recipe_source="ds4.gguf",
        imatrix_identity=imatrix_identity,
    )

    assert validate_base(plan) == []
    assert plan["artifact_kind"] == "package_plan"
    assert plan["producer_kind"] == "gguf_recipe"
    assert plan["producer_reference"] == "ds4.gguf"
    assert plan["status"] == "valid"
    assert plan["required_features"] == ["calibration"]
    assert len(plan["allocation"]) == 3
    assert plan["source_constraints"]["objective"] == "gguf_recipe_kquant_allocation"
    assert plan["source_constraints"]["recipe_source"] == "ds4.gguf"
    assert plan["source_constraints"]["imatrix"] == imatrix_identity
    assert plan["achieved"]["expert_codec_counts"] == {"iq2_xxs": 2, "q2_k": 1}


def test_ds4_kquant_plan_fails_closed_on_unknown_target_codec():
    bad = DS4KQuantExpertTarget(
        layer_index=7,
        projection="down",
        codec="q2_not_real",
        gguf_tensor="blk.7.ffn_down_exps.weight",
        imatrix_key="blk.7.ffn_down_exps.weight",
        source_weight_template="layers.7.ffn.experts.{expert}.w2.weight",
        source_scale_template="layers.7.ffn.experts.{expert}.w2.scale",
        module_path="model.layers.7.mlp.switch_mlp.down_proj",
        module_weight_key="model.layers.7.mlp.switch_mlp.down_proj.weight",
    )

    plan = build_ds4_kquant_plan(
        {"source_root": "/models/ds4", "source_format": "hf_safetensors"},
        [bad],
    )

    assert plan["status"] == "invalid"
    assert plan["allocation"] == []
    assert any(
        v["code"] == "kquant_recipe.invalid_targets"
        and "unknown kquant codec" in v["message"]
        for v in plan["validation"]
    )


def _qwen_inventory(layer: int = 7) -> dict:
    base = f"model.language_model.layers.{layer}.mlp.experts"
    return {
        "tensors": [
            {
                "source_name": f"{base}.gate_up_proj",
                "role": "moe.expert.gate_up",
                "kind": "expert",
                "layer_index": layer,
                "projection": "gate_up",
                "shape": [256, 1024, 2048],
                "gguf_keys": [
                    f"blk.{layer}.ffn_gate_exps.weight",
                    f"blk.{layer}.ffn_up_exps.weight",
                ],
                "status": "required",
            },
            {
                "source_name": f"{base}.down_proj",
                "role": "moe.expert.down",
                "kind": "expert",
                "layer_index": layer,
                "projection": "down",
                "shape": [256, 2048, 512],
                "gguf_keys": [f"blk.{layer}.ffn_down_exps.weight"],
                "status": "required",
            },
        ]
    }


def _qwen_full_recipe_inventory(layer: int = 7) -> dict:
    inv = _qwen_inventory(layer)
    inv["tensors"].extend([
        {
            "source_name": f"model.language_model.layers.{layer}.self_attn.q_proj.weight",
            "role": "attn.q_proj",
            "kind": "affine",
            "layer_index": layer,
            "shape": [8192, 2048],
            "gguf_keys": [f"blk.{layer}.attn_q.weight"],
            "status": "required",
        },
        {
            "source_name": f"model.language_model.layers.{layer}.linear_attn.in_proj_a.weight",
            "role": "ssm.in_proj_a",
            "kind": "affine",
            "layer_index": layer,
            "shape": [32, 2048],
            "gguf_keys": [f"blk.{layer}.ssm_alpha.weight"],
            "status": "required",
        },
        {
            "source_name": f"model.language_model.layers.{layer}.mlp.gate.weight",
            "role": "moe.router_gate",
            "kind": "affine",
            "layer_index": layer,
            "shape": [256, 2048],
            "gguf_keys": [f"blk.{layer}.ffn_gate_inp.weight"],
            "status": "required",
        },
        {
            "source_name": f"model.language_model.layers.{layer}.input_layernorm.weight",
            "role": "norm.input_layernorm",
            "kind": "passthrough",
            "layer_index": layer,
            "shape": [2048],
            "gguf_keys": [],
            "status": "required",
        },
        {
            "source_name": "model.language_model.embed_tokens.weight",
            "role": "embed_tokens",
            "kind": "affine",
            "layer_index": None,
            "shape": [248320, 2048],
            "gguf_keys": [],
            "status": "required",
        },
        {
            "source_name": "lm_head.weight",
            "role": "lm_head",
            "kind": "affine",
            "layer_index": None,
            "shape": [248320, 2048],
            "gguf_keys": [],
            "status": "required",
        },
        {
            "source_name": "model.language_model.norm.weight",
            "role": "norm.final",
            "kind": "passthrough",
            "layer_index": None,
            "shape": [2048],
            "gguf_keys": [],
            "status": "required",
        },
    ])
    return inv


def test_maps_qwen_stacked_expert_recipe_to_split_gate_up_targets():
    recipe = {
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }

    targets = build_qwen_expert_kquant_targets(
        recipe,
        _qwen_inventory(),
        required_layers=[7],
    )

    by_projection = {target.projection: target for target in targets}
    assert list(by_projection) == ["gate", "up", "down"]
    assert by_projection["gate"].source_name == (
        "model.language_model.layers.7.mlp.experts.gate_up_proj"
    )
    assert by_projection["up"].source_name == by_projection["gate"].source_name
    assert by_projection["down"].source_name == (
        "model.language_model.layers.7.mlp.experts.down_proj"
    )
    assert by_projection["gate"].source_projection == "gate_up"
    assert by_projection["up"].source_projection == "gate_up"
    assert by_projection["down"].source_projection == "down"
    assert by_projection["gate"].module_weight_key == (
        "language_model.model.layers.7.mlp.switch_mlp.gate_proj.weight"
    )
    assert by_projection["up"].module_weight_key == (
        "language_model.model.layers.7.mlp.switch_mlp.up_proj.weight"
    )
    assert by_projection["down"].module_weight_key == (
        "language_model.model.layers.7.mlp.switch_mlp.down_proj.weight"
    )
    assert by_projection["down"].codec == "q2_k"


def test_qwen_mapping_fails_closed_when_required_split_projection_is_missing():
    recipe = {
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }

    with pytest.raises(KQuantRecipeError, match="blk.7.ffn_up_exps.weight"):
        build_qwen_expert_kquant_targets(recipe, _qwen_inventory(), required_layers=[7])


def test_qwen_expert_logical_shape_splits_gate_up_and_keeps_down_orientation():
    recipe = {
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }
    targets = build_qwen_expert_kquant_targets(
        recipe,
        _qwen_inventory(),
        required_layers=[7],
    )
    by_projection = {target.projection: target for target in targets}

    assert qwen_expert_logical_shape(
        by_projection["gate"],
        [256, 1024, 2048],
    ) == (512, 2048)
    assert qwen_expert_logical_shape(
        by_projection["up"],
        [256, 1024, 2048],
    ) == (512, 2048)
    assert qwen_expert_logical_shape(
        by_projection["down"],
        [256, 2048, 512],
    ) == (2048, 512)


def test_builds_qwen_kquant_expert_allocations_for_package_manifest():
    targets = build_qwen_expert_kquant_targets({
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "q2_k",
    }, _qwen_inventory(), required_layers=[7])

    allocation = build_qwen_expert_kquant_allocations(targets)

    assert [a["projection"] for a in allocation] == ["gate", "up", "down"]
    gate, up, down = allocation
    assert gate["source_name"] == (
        "model.language_model.layers.7.mlp.experts.gate_up_proj"
    )
    assert gate["role"] == "moe.expert.gate"
    assert gate["format"] == "kquant"
    assert gate["codec"] == gate["kquant_codec"] == "iq2_xxs"
    assert gate["bits"] == 2
    assert gate["source_projection"] == "gate_up"
    assert up["source_name"] == gate["source_name"]
    assert up["source_projection"] == "gate_up"
    assert down["source_name"] == (
        "model.language_model.layers.7.mlp.experts.down_proj"
    )
    assert down["codec"] == down["kquant_codec"] == "q2_k"


def test_qwen_source_gguf_key_maps_globals_and_structural_tensors():
    inventory = _qwen_full_recipe_inventory(layer=7)
    by_name = {entry["source_name"]: entry for entry in inventory["tensors"]}

    assert qwen_source_gguf_key(by_name[
        "model.language_model.layers.7.input_layernorm.weight"
    ]) == "blk.7.attn_norm.weight"
    assert qwen_source_gguf_key(by_name[
        "model.language_model.embed_tokens.weight"
    ]) == "token_embd.weight"
    assert qwen_source_gguf_key(by_name["lm_head.weight"]) == "output.weight"
    assert qwen_source_gguf_key(by_name[
        "model.language_model.norm.weight"
    ]) == "output_norm.weight"


def test_qwen_module_path_matches_sanitized_jang_wrapper_path():
    assert qwen_module_path(
        "model.language_model.layers.7.self_attn.q_proj.weight"
    ) == "language_model.model.layers.7.self_attn.q_proj"
    assert qwen_module_path("lm_head.weight") == "language_model.lm_head"


def test_maps_qwen_dense_kquant_recipe_to_allocations_including_globals():
    recipe = {
        "blk.7.attn_q.weight": "q5_k",
        "token_embd.weight": "q4_k",
        "output.weight": "q4_k",
    }

    targets = build_qwen_dense_kquant_targets(
        recipe,
        _qwen_full_recipe_inventory(layer=7),
    )
    allocation = build_qwen_dense_kquant_allocations(targets)

    by_source = {row["source_name"]: row for row in allocation}
    q_proj = by_source["model.language_model.layers.7.self_attn.q_proj.weight"]
    assert q_proj["format"] == "kquant"
    assert q_proj["codec"] == q_proj["kquant_codec"] == "q5_k"
    assert q_proj["gguf_tensor"] == "blk.7.attn_q.weight"
    assert q_proj["module_weight_key"] == (
        "language_model.model.layers.7.self_attn.q_proj.weight"
    )

    embed = by_source["model.language_model.embed_tokens.weight"]
    assert embed["codec"] == embed["kquant_codec"] == "q4_k"
    assert embed["gguf_tensor"] == "token_embd.weight"
    assert embed["module_weight_key"] == "language_model.model.embed_tokens.weight"

    lm_head = by_source["lm_head.weight"]
    assert lm_head["codec"] == lm_head["kquant_codec"] == "q4_k"
    assert lm_head["gguf_tensor"] == "output.weight"
    assert lm_head["module_weight_key"] == "language_model.lm_head.weight"


def test_maps_qwen_gguf_f32_tensors_to_f32_passthrough_entries():
    tensor_types = {
        "blk.7.ssm_alpha.weight": "F32",
        "blk.7.ffn_gate_inp.weight": "F32",
        "blk.7.attn_norm.weight": "F32",
        "output_norm.weight": "F32",
        "blk.7.attn_q.weight": "Q5_K",
    }

    passthrough = build_qwen_f32_passthrough(
        tensor_types,
        _qwen_full_recipe_inventory(layer=7),
    )

    by_source = {row["source_name"]: row for row in passthrough}
    for source_name in (
        "model.language_model.layers.7.linear_attn.in_proj_a.weight",
        "model.language_model.layers.7.mlp.gate.weight",
        "model.language_model.layers.7.input_layernorm.weight",
        "model.language_model.norm.weight",
    ):
        assert by_source[source_name]["format"] == "f32_passthrough"
        assert by_source[source_name]["kind"] == "passthrough"

    assert "model.language_model.layers.7.self_attn.q_proj.weight" not in by_source


def test_builds_qwen_kquant_plan_from_dense_and_expert_recipe_targets():
    inventory = _qwen_full_recipe_inventory(layer=7)
    recipe = {
        "blk.7.ffn_gate_exps.weight": "iq2_xxs",
        "blk.7.ffn_up_exps.weight": "iq2_xxs",
        "blk.7.ffn_down_exps.weight": "iq2_s",
        "blk.7.attn_q.weight": "q5_k",
        "token_embd.weight": "q4_k",
        "output.weight": "q4_k",
    }
    expert_targets = build_qwen_expert_kquant_targets(
        recipe,
        inventory,
        required_layers=[7],
    )
    dense_targets = build_qwen_dense_kquant_targets(recipe, inventory)

    plan = build_qwen_kquant_plan(
        {"source_root": "/models/qwen", "source_format": "hf_safetensors"},
        expert_targets,
        dense_targets,
        recipe_source="qwen.gguf",
        imatrix_identity={"sha256": "abc", "key_count": 510},
    )

    assert validate_base(plan) == []
    assert plan["artifact_kind"] == "package_plan"
    assert plan["producer_kind"] == "gguf_recipe"
    assert plan["producer_reference"] == "qwen.gguf"
    assert plan["status"] == "valid"
    assert len(plan["allocation"]) == 6
    assert plan["source_constraints"]["recipe_source"] == "qwen.gguf"
    assert plan["achieved"]["dense_codec_counts"] == {"q4_k": 2, "q5_k": 1}
    assert plan["achieved"]["expert_codec_counts"] == {
        "iq2_s": 1,
        "iq2_xxs": 2,
    }
    assert plan["achieved"]["format_counts"] == {"kquant": 6}
