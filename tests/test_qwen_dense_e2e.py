"""End-to-end loader-contract test for the dense Qwen path.

This is the smallest useful proof for the dense path: build a small real Qwen3.5
hybrid text model, dump source-style weights, package them through MoEspresso's
streaming writer, emit the regular JANG v2 sidecars, then load through
`jang_tools.loader.load_jang_model` and run one forward.

It deliberately does not load a real model.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("jang_tools.loader")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from moespresso.inventory.build import build_inventory  # noqa: E402
from moespresso.optimize.affine_elasticity import QWEN35_AFFINE_ROLE_PROFILE_V1_NAME  # noqa: E402
from moespresso.optimize.decide import decide  # noqa: E402
from moespresso.package.plan import package_plan_from_decision  # noqa: E402
from moespresso.package.write import write_package  # noqa: E402
from moespresso.probe.build import build_probe_evidence  # noqa: E402


def _dense_text_config(**overrides) -> dict:
    cfg = {
        "model_type": "qwen3_5_text",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "max_position_embeddings": 4096,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "tie_word_embeddings": False,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "decoder_sparse_step": 1,
        "shared_expert_intermediate_size": 0,
        "moe_intermediate_size": 0,
        "layer_types": ["linear_attention", "full_attention"],
    }
    cfg.update(overrides)
    return cfg


def _arch(**text_overrides) -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": _dense_text_config(**text_overrides),
    }


def _reference_model(**text_overrides):
    import mlx_lm.models.qwen3_5 as M

    mx.random.seed(7)
    args = M.ModelArgs(model_type="qwen3_5", text_config=_dense_text_config(**text_overrides))
    model = M.Model(args)
    mx.eval(model.parameters())
    return model


def _param_to_source_name(param_path: str) -> str:
    if param_path.startswith("language_model.model."):
        return "model.language_model." + param_path[len("language_model.model."):]
    if param_path.startswith("language_model.lm_head."):
        return "lm_head." + param_path[len("language_model.lm_head."):]
    return param_path


_SANITIZE_SHIFT_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _dump_source_safetensors(model, path):
    tensors = {}
    for path_key, arr in tree_flatten(model.parameters()):
        a = np.array(arr, dtype=np.float32)
        name = _param_to_source_name(path_key)
        if name.endswith("linear_attn.conv1d.weight") and a.ndim == 3 and a.shape[2] == 1:
            a = np.ascontiguousarray(np.moveaxis(a, 1, 2))
        if a.ndim == 1 and any(name.endswith(sfx) for sfx in _SANITIZE_SHIFT_NORM_SUFFIXES):
            a = a - 1.0
        tensors[name] = a
    _write_safetensors(path, tensors)


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        b = a.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(a.shape),
            "data_offsets": [off, off + len(b)],
        }
        blob += b
        off += len(b)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _emit_sidecars(manifest, out_dir):
    from moespresso.package.sidecars import build_sidecars

    config_json, jang_config = build_sidecars(manifest)
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))
    (out_dir / "jang_config.json").write_text(json.dumps(jang_config, indent=2))


def _logits(model, ids):
    out = model(mx.array([ids]))
    mx.eval(out)
    return np.array(out[0, -1], dtype=np.float32)


def _package_dense(tmp_path, *, tie_word_embeddings=False):
    ref = _reference_model(tie_word_embeddings=tie_word_embeddings)
    src = tmp_path / "src"
    src.mkdir()
    arch = _arch(tie_word_embeddings=tie_word_embeddings)
    _dump_source_safetensors(ref, src / "model-00001.safetensors")
    (src / "config.json").write_text(json.dumps(arch))

    inv = build_inventory(src, layer_types=arch["text_config"]["layer_types"])
    assert inv["counts"]["expert"] == 0
    assert inv["counts"]["affine"] > 0
    assert inv["counts"]["passthrough"] > 0

    passthrough = [e for e in inv["tensors"] if e["kind"] == "passthrough"]
    ev = build_probe_evidence(inv, src, sample_rows=64)
    dec = decide(ev, target_quality=0.5)
    plan, _summary = package_plan_from_decision(dec)
    out = tmp_path / "pkg"
    man = write_package(plan, src, arch, out, passthrough=passthrough)
    assert man["architecture"]["family"] == "qwen3_5_dense"
    assert man["status"] == "valid"
    _emit_sidecars(man, out)
    return ref, man, out


def test_dense_affine_package_loads_through_regular_jang_v2(monkeypatch, tmp_path):
    _ref, _man, out = _package_dense(tmp_path)
    import mlx_lm.utils as mlx_utils
    from jang_tools.loader import load_jang_model

    monkeypatch.setattr(mlx_utils, "load_tokenizer", lambda *args, **kwargs: object())
    served, _tokenizer = load_jang_model(out)

    got = _logits(served, [1, 5, 9, 13, 2, 7])
    assert np.isfinite(got).all()
    assert float(np.std(got)) > 1e-3


def test_dense_tied_embedding_package_builds_through_manifest_runtime(monkeypatch, tmp_path):
    ref, man, out = _package_dense(tmp_path, tie_word_embeddings=True)
    assert not any(t["source_name"] == "lm_head.weight" for t in man["tensors"])
    assert man["architecture"]["config"]["tie_word_embeddings"] is True

    import mlx_lm.utils as mlx_utils
    from moespresso.runtime.build import build_model

    monkeypatch.setattr(mlx_utils, "load_tokenizer", lambda *args, **kwargs: object())
    served, _tokenizer = build_model(man, out)

    ids = [1, 5, 9, 13, 2, 7]
    ref_logits = _logits(ref, ids)
    got_logits = _logits(served, ids)

    assert np.isfinite(got_logits).all()
    assert float(np.std(got_logits)) > 1e-3
    corr = float(np.corrcoef(ref_logits, got_logits)[0, 1])
    assert corr > 0.7, f"dense tied-embedding logits barely track reference (r={corr:.3f})"


def test_dense_convert_orchestrator_writes_regular_jang_v2_and_builds(
    monkeypatch,
    tmp_path,
):
    ref = _reference_model(tie_word_embeddings=True)
    src = tmp_path / "src"
    src.mkdir()
    arch = _arch(tie_word_embeddings=True)
    _dump_source_safetensors(ref, src / "model-00001.safetensors")
    (src / "config.json").write_text(json.dumps(arch))

    from moespresso.package.convert import convert

    out = tmp_path / "pkg"
    man = convert(src, out, allow_uniform=True, target_size_gb=0.001, shard_size_gb=0.0)
    jang_config = json.loads((out / "jang_config.json").read_text())
    decision = json.loads((out / "optimizer_decision.json").read_text())

    assert man["architecture"]["family"] == "qwen3_5_dense"
    assert decision["constraints"]["affine_role_profile_name"] == QWEN35_AFFINE_ROLE_PROFILE_V1_NAME
    assert decision["constraints"]["affine_role_min_bits"]["ffn.down_proj"] == 4
    assert jang_config["format"] == "jang"
    assert jang_config["format_version"] == "2.0"
    assert "MOESPRESSO_MOE" not in str(jang_config)

    import mlx_lm.utils as mlx_utils
    from moespresso.runtime.build import build_model

    monkeypatch.setattr(mlx_utils, "load_tokenizer", lambda *args, **kwargs: object())
    served, _tokenizer = build_model(man, out)

    ids = [1, 5, 9, 13, 2, 7]
    ref_logits = _logits(ref, ids)
    got_logits = _logits(served, ids)
    corr = float(np.corrcoef(ref_logits, got_logits)[0, 1])
    assert corr > 0.7, f"dense convert logits barely track reference (r={corr:.3f})"
