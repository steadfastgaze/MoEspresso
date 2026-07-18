from __future__ import annotations

from moespresso.inventory.architecture_profile import DEEPSEEK_V4_FLASH_COMPRESS_RATIOS
from moespresso.inventory.deepseek_v4.static import (
    EXPECTED_SHARD_COUNT,
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TOTAL_SIZE,
    validate_deepseek_v4_static,
)
from moespresso.inventory.safetensors_header import TensorHeader


def _config():
    return {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "sliding_window": 128,
        "index_topk": 512,
        "compress_rope_theta": 160000,
        "compress_ratios": list(DEEPSEEK_V4_FLASH_COMPRESS_RATIOS),
    }


def _h(name, dtype, shape):
    return TensorHeader(name=name, dtype=dtype, shape=tuple(shape), shard="model-00001.safetensors")


def _headers():
    return [
        _h("embed.weight", "BF16", [129280, 4096]),
        _h("head.weight", "BF16", [129280, 4096]),
        _h("layers.0.attn.wq_a.weight", "F8_E4M3", [1024, 4096]),
        _h("layers.0.attn.wq_a.scale", "F8_E8M0", [8, 32]),
        _h("layers.2.attn.compressor.ape", "F32", [4, 1024]),
        _h("layers.2.attn.indexer.wq_b.weight", "F8_E4M3", [8192, 1024]),
        _h("layers.3.attn.compressor.ape", "F32", [128, 512]),
        _h("layers.0.ffn.gate.tid2eid", "I64", [129280, 6]),
        _h("layers.3.ffn.gate.bias", "F32", [256]),
        _h("layers.0.ffn.experts.0.w1.weight", "I8", [2048, 2048]),
        _h("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", [2048, 128]),
        _h("layers.0.ffn.shared_experts.w1.weight", "F8_E4M3", [2048, 4096]),
        _h("layers.0.ffn.shared_experts.w2.weight", "F8_E4M3", [4096, 2048]),
        _h("layers.0.hc_attn_fn", "F32", [24, 16384]),
        _h("hc_head_fn", "F32", [4, 16384]),
        _h("mtp.0.e_proj.weight", "F8_E4M3", [4096, 4096]),
    ]


def test_deepseek_v4_static_validator_accepts_expected_header_facts():
    issues = validate_deepseek_v4_static(
        _config(),
        _headers(),
        shard_count=EXPECTED_SHARD_COUNT,
        tensor_count=EXPECTED_TENSOR_COUNT,
        total_size=EXPECTED_TOTAL_SIZE,
    )
    assert issues == []


def test_deepseek_v4_static_validator_rejects_missing_layer_42_csa_ratio():
    config = _config()
    config["compress_ratios"] = config["compress_ratios"][:43]
    issues = validate_deepseek_v4_static(config, _headers())
    paths = {v.path for v in issues}
    assert "/config/compress_ratios" in paths
    assert "/config/compress_ratios/43" in paths
    assert all(v.blocking for v in issues)


def test_deepseek_v4_static_validator_rejects_layer_42_swa_ratio():
    config = _config()
    config["compress_ratios"][42] = 128
    issues = validate_deepseek_v4_static(config, _headers())
    assert any(
        v.code == "deepseek_v4.static_mismatch"
        and v.path == "/config/compress_ratios/42"
        and v.expected == 4
        and v.actual == 128
        for v in issues
    )


def test_deepseek_v4_static_validator_rejects_wrong_control_dtype():
    headers = [
        _h("layers.0.hc_attn_fn", "F8_E4M3", [24, 16384])
        if h.name == "layers.0.hc_attn_fn" else h
        for h in _headers()
    ]
    issues = validate_deepseek_v4_static(_config(), headers)
    assert any(v.code == "deepseek_v4.dtype_mismatch" for v in issues)
    assert any(v.path == "/tensors/layers.0.hc_attn_fn/dtype" for v in issues)


def test_deepseek_v4_static_validator_rejects_wrong_down_projection_shape():
    headers = [
        _h("layers.0.ffn.shared_experts.w2.weight", "F8_E4M3", [2048, 4096])
        if h.name == "layers.0.ffn.shared_experts.w2.weight" else h
        for h in _headers()
    ]
    issues = validate_deepseek_v4_static(_config(), headers)
    assert any(v.code == "deepseek_v4.shape_mismatch" for v in issues)
    assert any(v.path == "/tensors/layers.0.ffn.shared_experts.w2.weight/shape" for v in issues)


def test_deepseek_v4_static_validator_rejects_missing_control_tensor():
    headers = [h for h in _headers() if h.name != "layers.2.attn.compressor.ape"]
    issues = validate_deepseek_v4_static(_config(), headers)
    assert any(v.code == "deepseek_v4.missing_tensor" for v in issues)
    assert any(v.path == "/tensors/layers.2.attn.compressor.ape" for v in issues)


def test_deepseek_v4_static_validator_rejects_wrong_index_counts():
    issues = validate_deepseek_v4_static(
        _config(),
        _headers(),
        shard_count=45,
        tensor_count=EXPECTED_TENSOR_COUNT - 1,
        total_size=EXPECTED_TOTAL_SIZE - 1,
    )
    paths = {v.path for v in issues}
    assert "/index/shard_count" in paths
    assert "/index/tensor_count" in paths
    assert "/index/total_size" in paths
