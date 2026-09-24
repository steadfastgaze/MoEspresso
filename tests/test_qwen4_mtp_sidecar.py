"""Tiny target-bound IQ2 sidecars exercise real writing and loading."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from moespresso.core.artifact import make_artifact, read_artifact, write_artifact
from moespresso.inventory.qwen4.mtp import resolve_qwen4_mtp_headers
from moespresso.inventory.safetensors_header import read_headers_with_offsets
from moespresso.package.constants import MANIFEST_NAME
from moespresso.package.qwen4.mtp_format import MTP_SIDECAR_NAME
from moespresso.package.qwen4.mtp_sidecar import (
    build_qwen4_mtp_sidecar,
    plan_qwen4_mtp_sidecar,
    verify_qwen4_mtp_sidecar,
)
from moespresso.package.write import _BF16Codes, _write_shard_deterministic
from moespresso.runtime.qwen4.iqk_dense import _walk_module_path
from moespresso.runtime.qwen4.mtp_load import load_qwen4_mtp_sidecar

from test_qwen4_mtp import _head, _inputs
from test_qwen4_mtp_inventory import _headers


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    head = _head()
    head.apply(lambda value: value.astype(mx.bfloat16))
    records = resolve_qwen4_mtp_headers(_headers())
    tensors = {}
    for record in records:
        parent, name = _walk_module_path(head, record.module_path)
        module = getattr(parent, name)
        values = (getattr(module, {"gate_up": "gate_up_proj", "down": "down_proj"}[record.projection])
                  if record.kind == "expert" else module.weight)
        tensors[record.header.name] = _BF16Codes(np.asarray(values.view(mx.uint16)))
    shard = source / "source.safetensors"
    _write_shard_deterministic(shard, tensors, {"fixture": "synthetic"})
    headers = {header.name: header for header in read_headers_with_offsets(shard)}
    resolved = tuple(replace(record, header=headers[record.header.name]) for record in records)
    config = {
        "hidden_size": 8, "hc_count": 4, "hc_lowrank": 4, "head_dim": 4,
        "num_attention_heads": 2, "num_key_value_heads": 1, "indexer_n_heads": 1,
        "indexer_kv_heads": 1, "indexer_head_dim": 4, "indexer_budget": 4,
        "indexer_compress_ratio": 2, "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8, "num_experts": 4,
        "num_experts_per_tok": 2, "vocab_size": 13, "rms_norm_eps": 1e-6,
        "partial_rotary_factor": 1.0,
        "rope_parameters": {"rope_theta": 1e7, "mrope_section": [1, 1, 0]},
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
        "tie_word_embeddings": False, "mtp": {"layer_types": ["full_attention"]},
    }
    manifest = make_artifact(
        "package_manifest", producer={"tool": "test", "version": "1"},
        subject={"fixture": "synthetic"}, status="valid",
        architecture={"family": "qwen4_exp", "config": config},
    )
    write_artifact(target / MANIFEST_NAME, manifest)
    monkeypatch.setattr("moespresso.package.qwen4.mtp_sidecar.inspect_qwen4_mtp_source",
                        lambda _path: (config, resolved))
    monkeypatch.setattr("moespresso.package.qwen4.mtp_sidecar._source_binding",
                        lambda _path, _target: {"fixture": "synthetic-source-binding"})
    model = SimpleNamespace(cache_identity=manifest["artifact_id"] + "|test-routing",
                            embedding=nn.Embedding(13, 8), lm_head=head.lm_head)
    return SimpleNamespace(source=source, target=target, manifest=manifest, model=model,
                           norm=head.pre_fc_norm_hidden.weight)


def _build(fixture, output):
    plan = plan_qwen4_mtp_sidecar(fixture.source, fixture.target, allow_uncalibrated=True)
    return build_qwen4_mtp_sidecar(fixture.source, fixture.target, output, plan=plan)


def test_sidecar_roundtrip_preserves_target_and_raw_norms(fixture, tmp_path):
    before = (fixture.target / MANIFEST_NAME).read_bytes()
    output = tmp_path / "sidecar"
    manifest = _build(fixture, output)
    verified = verify_qwen4_mtp_sidecar(output, fixture.manifest)
    assert verified["artifact_id"] == manifest["artifact_id"]
    loaded = load_qwen4_mtp_sidecar(output, fixture.manifest, target_model=fixture.model)
    assert loaded.embedding is fixture.model.embedding
    assert loaded.head.lm_head is fixture.model.lm_head
    assert loaded.target_cache_identity == fixture.model.cache_identity
    assert loaded.payload_bytes == manifest["byte_estimate"]["iq2_k_payload_with_input_padding"]
    np.testing.assert_array_equal(np.asarray(loaded.head.pre_fc_norm_hidden.weight.view(mx.uint16)),
                                  np.asarray(fixture.norm.view(mx.uint16)))
    hidden, embeddings = (value.astype(mx.bfloat16) for value in _inputs(3))
    result = loaded.head(hidden, embeddings)
    assert result.logits.shape == (1, 3, 13)
    assert bool(mx.all(mx.isfinite(result.logits)))
    repeat = load_qwen4_mtp_sidecar(output, fixture.manifest, target_model=fixture.model)
    np.testing.assert_array_equal(np.asarray(result.logits.astype(mx.float32)),
                                  np.asarray(repeat.head(hidden, embeddings).logits.astype(mx.float32)))
    assert (fixture.target / MANIFEST_NAME).read_bytes() == before
    assert list(fixture.target.iterdir()) == [fixture.target / MANIFEST_NAME]


def test_sidecar_build_is_reproducible_and_requires_explicit_objective(fixture, tmp_path):
    with pytest.raises(ValueError, match="uncalibrated"):
        plan_qwen4_mtp_sidecar(fixture.source, fixture.target)
    first, second = _build(fixture, tmp_path / "a"), _build(fixture, tmp_path / "b")
    assert first["artifact_id"] == second["artifact_id"]
    assert first["files"] == second["files"]
    for file in first["files"]:
        assert (tmp_path / "a" / file["path"]).read_bytes() == (tmp_path / "b" / file["path"]).read_bytes()


def test_sidecar_refuses_changed_plan_and_unsafe_output(fixture, tmp_path):
    plan = plan_qwen4_mtp_sidecar(fixture.source, fixture.target, allow_uncalibrated=True)
    plan["graph"]["hc_count"] += 1
    with pytest.raises(ValueError, match="plan"):
        build_qwen4_mtp_sidecar(fixture.source, fixture.target, tmp_path / "out", plan=plan)
    with pytest.raises(ValueError, match="outside"):
        build_qwen4_mtp_sidecar(fixture.source, fixture.target, fixture.target / "out", plan=plan)
    _build(fixture, tmp_path / "occupied")
    with pytest.raises(ValueError, match="empty"):
        _build(fixture, tmp_path / "occupied")


def test_sidecar_failed_build_never_publishes_a_manifest(fixture, tmp_path):
    plan = plan_qwen4_mtp_sidecar(fixture.source, fixture.target, allow_uncalibrated=True)
    output = tmp_path / "failed"

    def fail(_progress):
        raise RuntimeError("injected construction failure")

    with pytest.raises(RuntimeError, match="construction failure"):
        build_qwen4_mtp_sidecar(fixture.source, fixture.target, output, plan=plan, progress=fail)
    assert not (output / MTP_SIDECAR_NAME).exists()
    assert not list(output.glob("mtp-components-*"))


@pytest.mark.parametrize("mutation", ["schema", "target", "port", "shape", "path", "missing"])
def test_sidecar_loader_refuses_manifest_drift(fixture, tmp_path, mutation):
    output = tmp_path / "sidecar"
    _build(fixture, output)
    manifest = read_artifact(output / MTP_SIDECAR_NAME)
    if mutation == "schema":
        manifest["sidecar_schema"] = "unknown"
    elif mutation == "target":
        manifest["target_artifact_id"] = "pkg:" + "0" * 64
    elif mutation == "port":
        manifest["tensors"][0]["module_path"] = "lm_head"
    elif mutation == "shape":
        manifest["tensors"][0]["logical_shape"][0] += 1
    elif mutation == "path":
        manifest["files"][0]["path"] = "../outside.safetensors"
    else:
        (output / manifest["files"][0]["path"]).unlink()
    manifest.pop("artifact_id")
    write_artifact(output / MTP_SIDECAR_NAME, manifest)
    with pytest.raises(ValueError):
        load_qwen4_mtp_sidecar(output, fixture.manifest, target_model=fixture.model)


def test_sidecar_payload_hash_verification_is_separate(fixture, tmp_path):
    output = tmp_path / "sidecar"
    manifest = _build(fixture, output)
    path = output / manifest["files"][0]["path"]
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="content verification"):
        verify_qwen4_mtp_sidecar(output, fixture.manifest)
