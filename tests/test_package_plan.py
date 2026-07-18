from __future__ import annotations

import pytest

from moespresso.core.artifact import make_artifact
from moespresso.package.manifest import build_package_manifest, located_key
from moespresso.package.plan import (
    PackagePlanError,
    make_package_plan,
    parse_force_overrides,
)


SUBJECT = {"source_root": "toy", "source_format": "hf_safetensors"}
PRODUCER = {"tool": "test", "version": "0"}
ARCH = {
    "model_type": "qwen3_moe",
    "text_config": {
        "num_hidden_layers": 1,
        "hidden_size": 256,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 128,
        "layer_types": ["full_attention"],
        "vocab_size": 512,
    },
}


def _decision() -> dict:
    return make_artifact(
        "optimizer_decision",
        SUBJECT,
        PRODUCER,
        status="valid",
        source_probe_id="probe:test",
        allocation=[
            {
                "source_name": "model.layers.0.mlp.experts.gate_up_proj",
                "kind": "expert",
                "role": "moe.expert.gate",
                "layer_index": 0,
                "projection": "gate",
                "source_projection": "gate_up",
                "format": "kquant",
                "codec": "iq2_xxs",
                "kquant_codec": "iq2_xxs",
                "bits": 2,
                "imatrix_key": "blk.0.ffn_gate_exps.weight",
                "gguf_tensor": "blk.0.ffn_gate_exps.weight",
                "module_path": "model.layers.0.mlp.switch_mlp.gate_proj",
                "module_weight_key": "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            },
            {
                "source_name": "model.layers.0.self_attn.q_proj.weight",
                "kind": "affine",
                "role": "attn.q_proj",
                "layer_index": 0,
                "format": "affine",
                "bits": 4,
                "group_size": 64,
            },
        ],
        constraints={"objective": "test"},
        achieved={},
    )


def _located(decision: dict) -> dict:
    return {
        located_key(alloc): {
            "shard": "model-00001-of-00001.safetensors",
            "key_prefix": alloc["source_name"],
        }
        for alloc in decision["allocation"]
    }


def test_package_plan_force_override_dry_run_does_not_mutate_decision():
    decision = _decision()
    overrides = parse_force_overrides(["*ffn_gate_exps.weight=tq2"])

    planned, summary = make_package_plan(
        decision["subject"],
        decision["allocation"],
        producer_kind="gguf_recipe",
        producer_reference="recipe.gguf",
        force_overrides=overrides,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["matched"][0]["before"] == "kquant:iq2_xxs"
    assert summary["matched"][0]["after"] == "tq2"
    assert planned["force_override_preview"]["matched"][0]["before"] == "kquant:iq2_xxs"
    assert planned["force_override_preview"]["matched"][0]["after"] == "tq2"
    assert planned["allocation"][0]["format"] == "kquant"
    assert planned["artifact_kind"] == "package_plan"
    assert planned["producer_kind"] == "gguf_recipe"
    assert planned["producer_reference"] == "recipe.gguf"
    assert planned["optimized_kernels_expected"] is False
    assert planned["force_overrides"] == [
        {"pattern": "*ffn_gate_exps.weight", "target": "tq2"}
    ]


def test_package_plan_force_override_changes_format_and_records_reason():
    decision = _decision()
    overrides = parse_force_overrides(["*ffn_gate_exps.weight=tq2"])

    planned, summary = make_package_plan(
        decision["subject"],
        decision["allocation"],
        producer_kind="gguf_recipe",
        force_overrides=overrides,
    )

    forced = planned["allocation"][0]
    assert summary["matched"][0]["source_name"] == forced["source_name"]
    assert forced["format"] == "tq"
    assert forced["codec"] == "tq"
    assert forced["bits"] == 2
    assert "kquant_codec" not in forced
    assert forced["forced_format"] == {
        "pattern": "*ffn_gate_exps.weight",
        "target": "tq2",
        "before": "kquant:iq2_xxs",
    }


def test_package_plan_rejects_unknown_force_format():
    with pytest.raises(PackagePlanError, match="unknown force override format"):
        parse_force_overrides(["*gate*=fp39393939"])


def test_package_plan_rejects_unmatched_force_pattern_by_default():
    with pytest.raises(PackagePlanError, match="matched no tensors"):
        decision = _decision()
        make_package_plan(
            decision["subject"],
            decision["allocation"],
            producer_kind="optimizer",
            force_overrides=parse_force_overrides(["*nope*=tq2"]),
        )


def test_manifest_records_package_plan_metadata_and_forced_tensor(tmp_path):
    source = _decision()
    decision, _summary = make_package_plan(
        source["subject"],
        source["allocation"],
        producer_kind="gguf_recipe",
        producer_reference="recipe.gguf",
        optimized_kernels_expected=True,
        force_overrides=parse_force_overrides(["*ffn_gate_exps.weight=tq2"]),
    )
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"fake")
    files = [{
        "path": shard.name,
        "size_bytes": shard.stat().st_size,
        "sha256": "0" * 64,
    }]

    manifest = build_package_manifest(decision, ARCH, _located(decision), files)

    assert manifest["optimized_kernels_expected"] is True
    assert manifest["provenance"]["package_plan"]["producer_kind"] == "gguf_recipe"
    gate = next(t for t in manifest["tensors"] if t.get("projection") == "gate")
    assert gate["format"] == "tq"
    assert gate["format_decision"]["forced"]["target"] == "tq2"
