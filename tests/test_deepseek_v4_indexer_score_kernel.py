from __future__ import annotations

import numpy as np
import pytest

from moespresso.runtime.deepseek_v4 import indexer_score_kernel


def _composed_reference(mx, q, pooled, weights, scale):
    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

    q = _dsv4_indexer_qat(mx, q)
    scores = (
        q.astype(mx.float32)
        @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
    )
    scores = mx.maximum(scores, 0) * scale
    return (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)


def _decode_inputs(mx, *, n_heads, head_dim, n_rows, seed, q_dtype=None):
    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((1, n_heads, 1, head_dim), dtype=np.float32))
    if q_dtype is not None:
        q = q.astype(q_dtype)
    pooled = _dsv4_indexer_qat(
        mx,
        mx.array(rng.standard_normal((1, n_rows, head_dim), dtype=np.float32)),
    )
    weights = mx.array(
        rng.standard_normal((1, 1, n_heads), dtype=np.float32)
    ) * (n_heads ** -0.5)
    return q, pooled, weights


def _qat_case_rows(name, rng):
    if name == "normal":
        return rng.standard_normal((1024, 128)).astype(np.float32)
    if name == "scaled_2p30":
        return (rng.standard_normal((1024, 128)) * 2.0 ** 30).astype(np.float32)
    if name == "scaled_2m30":
        return (rng.standard_normal((1024, 128)) * 2.0 ** -30).astype(np.float32)
    if name == "amax_near_lattice_boundary":
        # Rows rescaled so each 32-element group amax lands within a few ulp
        # of 6 * 2^k, the ceil boundary of the group scale derivation.
        base = rng.standard_normal((1024, 128)).astype(np.float32)
        groups = base.reshape(1024, 4, 32)
        amax = np.abs(groups).max(axis=-1, keepdims=True)
        target = 6.0 * (2.0 ** rng.integers(-40, 40, size=(1024, 4, 1)))
        jitter = 1.0 + rng.integers(-4, 5, size=(1024, 4, 1)) * np.float32(
            1.1920929e-07
        )
        return (groups / amax * target * jitter).astype(np.float32).reshape(
            1024, 128
        )
    if name == "lattice_midpoints":
        # Exact ties between adjacent e2m1 lattice values; mx.argmin keeps
        # the first (lower) value, so a round-to-even tie break diverges.
        mids = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
        signs = np.where(rng.random((256, 128)) < 0.5, 1.0, -1.0)
        return (rng.choice(mids, size=(256, 128)) * signs).astype(np.float32)
    if name == "zeros_and_denormals":
        rows = np.zeros((8, 128), dtype=np.float32)
        rows[1] = -0.0
        rows[2] = 1e-39
        rows[3] = rng.standard_normal(128) * 1e-38
        rows[4] = rng.standard_normal(128) * 1e-44
        return rows
    raise AssertionError(name)


@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "scaled_2p30",
        "scaled_2m30",
        "amax_near_lattice_boundary",
        "lattice_midpoints",
        "zeros_and_denormals",
    ],
)
def test_indexer_qat_rows_bitexact_float32(case):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

    rows = mx.array(_qat_case_rows(case, np.random.default_rng(1234)))
    got = indexer_score_kernel.indexer_qat_rows(rows)
    expected = _dsv4_indexer_qat(mx, rows)
    mx.eval(got, expected)

    assert got.dtype == mx.float32
    got_bits = np.asarray(got, dtype=np.float32).view(np.uint32)
    expected_bits = np.asarray(expected, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(got_bits, expected_bits)


@pytest.mark.parametrize("q_dtype_name", ["float16", "bfloat16"])
def test_indexer_qat_rows_bitexact_low_precision_query(q_dtype_name):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

    q_dtype = getattr(mx, q_dtype_name)
    rng = np.random.default_rng(77)
    rows = mx.array(
        rng.standard_normal((1, 64, 1, 128), dtype=np.float32)
    ).astype(q_dtype)
    got = indexer_score_kernel.indexer_qat_rows(rows)
    expected = _dsv4_indexer_qat(mx, rows)
    mx.eval(got, expected)

    assert got.shape == rows.shape
    assert got.dtype == mx.float32
    got_bits = np.asarray(got, dtype=np.float32).view(np.uint32)
    expected_bits = np.asarray(expected, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(got_bits, expected_bits)


def test_indexer_qat_rows_rejects_bad_inputs():
    mx = pytest.importorskip("mlx.core")

    with pytest.raises(ValueError, match="128-wide"):
        indexer_score_kernel.indexer_qat_rows(mx.zeros((4, 64), dtype=mx.float32))
    with pytest.raises(ValueError, match="float32, float16, or bfloat16"):
        indexer_score_kernel.indexer_qat_rows(mx.zeros((4, 128), dtype=mx.int32))


# Served DS4 decode shape from the package config: index_n_heads=64,
# index_head_dim=128. The odd row counts cover the row-tail guard, 961
# being the fenced A/B pool size.
@pytest.mark.parametrize("n_rows", [3, 512, 961, 977])
def test_fused_qat_indexer_scores_matches_composed_chain(n_rows):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    n_heads = 64
    head_dim = 128
    scale = head_dim ** -0.5
    q, pooled, weights = _decode_inputs(
        mx, n_heads=n_heads, head_dim=head_dim, n_rows=n_rows, seed=1000 + n_rows
    )

    got = indexer_score_kernel.fused_qat_indexer_scores(q, pooled, weights, scale)
    expected = _composed_reference(mx, q, pooled, weights, scale)
    mx.eval(got, expected)

    assert got.shape == (n_rows,)
    assert got.dtype == mx.float32
    got_row = np.asarray(got)
    expected_row = np.asarray(expected).reshape(-1)
    np.testing.assert_allclose(got_row, expected_row, rtol=1.0e-5, atol=1.0e-5)

    k = min(512, n_rows)
    got_topk = set(np.argpartition(-got_row, k - 1)[:k].tolist())
    expected_topk = set(np.argpartition(-expected_row, k - 1)[:k].tolist())
    assert got_topk == expected_topk


def test_fused_qat_indexer_scores_matches_composed_chain_f16_query():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    scale = 128 ** -0.5
    q, pooled, weights = _decode_inputs(
        mx, n_heads=64, head_dim=128, n_rows=961, seed=9, q_dtype=mx.float16
    )

    got = indexer_score_kernel.fused_qat_indexer_scores(q, pooled, weights, scale)
    expected = _composed_reference(mx, q, pooled, weights, scale)
    mx.eval(got, expected)

    got_row = np.asarray(got)
    expected_row = np.asarray(expected).reshape(-1)
    np.testing.assert_allclose(got_row, expected_row, rtol=1.0e-5, atol=1.0e-5)

    got_topk = set(np.argpartition(-got_row, 511)[:512].tolist())
    expected_topk = set(np.argpartition(-expected_row, 511)[:512].tolist())
    assert got_topk == expected_topk


def test_fused_qat_indexer_scores_matches_composed_chain_single_head():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    scale = 128 ** -0.5
    q, pooled, weights = _decode_inputs(
        mx, n_heads=1, head_dim=128, n_rows=5, seed=42
    )

    got = indexer_score_kernel.fused_qat_indexer_scores(q, pooled, weights, scale)
    expected = _composed_reference(mx, q, pooled, weights, scale)
    mx.eval(got, expected)

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected).reshape(-1),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_fused_qat_indexer_scores_rejects_non_decode_shapes():
    mx = pytest.importorskip("mlx.core")

    q = mx.zeros((1, 4, 2, 128), dtype=mx.float32)
    pooled = mx.zeros((1, 3, 128), dtype=mx.float32)
    weights = mx.zeros((1, 1, 4), dtype=mx.float32)
    with pytest.raises(ValueError, match="one decode token"):
        indexer_score_kernel.fused_qat_indexer_scores(q, pooled, weights, 1.0)

    q = mx.zeros((1, 4, 1, 64), dtype=mx.float32)
    with pytest.raises(ValueError, match="head_dim=128"):
        indexer_score_kernel.fused_qat_indexer_scores(q, pooled, weights, 1.0)

    q = mx.zeros((1, 65, 1, 128), dtype=mx.float32)
    with pytest.raises(ValueError, match="between 1 and 64 heads"):
        indexer_score_kernel.fused_qat_indexer_scores(
            q, pooled, mx.zeros((1, 1, 65), dtype=mx.float32), 1.0
        )

    q = mx.zeros((1, 4, 1, 128), dtype=mx.int32)
    with pytest.raises(ValueError, match="float32, float16, or bfloat16"):
        indexer_score_kernel.fused_qat_indexer_scores(q, pooled, weights, 1.0)

    q = mx.zeros((1, 4, 1, 128), dtype=mx.float32)
    with pytest.raises(ValueError, match="pooled_qat and weights must be float32"):
        indexer_score_kernel.fused_qat_indexer_scores(
            q, pooled.astype(mx.float16), weights, 1.0
        )


def test_fused_qat_scores_eligible_covers_decode_shape_predicate(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setattr(indexer_score_kernel, "_metal_available", lambda: True)

    q = mx.zeros((1, 4, 1, 128), dtype=mx.float32)
    pooled = mx.zeros((1, 3, 128), dtype=mx.float32)
    weights = mx.zeros((1, 1, 4), dtype=mx.float32)

    # The fused decode path ships gated off: the fenced served-layer A/B
    # measured 1.33x on the indexer segment, below the 1.8x retention bar.
    assert indexer_score_kernel._ENABLED is False
    assert not indexer_score_kernel.fused_qat_scores_eligible(q, pooled, weights)

    monkeypatch.setattr(indexer_score_kernel, "_ENABLED", True)
    assert indexer_score_kernel.fused_qat_scores_eligible(q, pooled, weights)

    # The query is pre-QAT, so the served half/bfloat activations qualify.
    assert indexer_score_kernel.fused_qat_scores_eligible(
        q.astype(mx.float16), pooled, weights
    )
    assert indexer_score_kernel.fused_qat_scores_eligible(
        q.astype(mx.bfloat16), pooled, weights
    )

    prefill_q = mx.zeros((1, 4, 2, 128), dtype=mx.float32)
    assert not indexer_score_kernel.fused_qat_scores_eligible(
        prefill_q, pooled, weights
    )
    # The threadgroup staging buffer is sized for the DS4 index head count.
    assert not indexer_score_kernel.fused_qat_scores_eligible(
        mx.zeros((1, 65, 1, 128), dtype=mx.float32),
        pooled,
        mx.zeros((1, 1, 65), dtype=mx.float32),
    )
    assert not indexer_score_kernel.fused_qat_scores_eligible(
        q.astype(mx.int32), pooled, weights
    )
    assert not indexer_score_kernel.fused_qat_scores_eligible(
        q, pooled.astype(mx.float16), weights
    )
    assert not indexer_score_kernel.fused_qat_scores_eligible(
        q, pooled, weights.astype(mx.float16)
    )
    assert not indexer_score_kernel.fused_qat_scores_eligible(
        q, pooled, mx.zeros((1, 2, 4), dtype=mx.float32)
    )


def _tiny_ratio4_manifest(index_n_heads: int = 2):
    config = {
        "model_type": "deepseek_v4",
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 16,
        "o_lora_rank": 16,
        "o_groups": 2,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "num_hash_layers": 0,
        "num_nextn_predict_layers": 1,
        "sliding_window": 8,
        "compress_ratios": [4],
        "index_n_heads": index_n_heads,
        "index_head_dim": 128,
        "index_topk": 2,
    }
    return {
        "architecture": {
            "family": "deepseek_v4_flash",
            "config": config,
            "compress_ratios": [4],
        }
    }


def test_indexer_score_contract_routes_decode_fused_and_prefill_composed(
    monkeypatch,
):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("jang_tools.dsv4.mlx_model")
    from moespresso.runtime.deepseek_v4.model import (
        _patch_deepseek_v4_indexer_score_contract,
        build_deepseek_v4_graph_from_manifest,
    )

    manifest = _tiny_ratio4_manifest()
    cfg = manifest["architecture"]["config"]
    model = build_deepseek_v4_graph_from_manifest(manifest)
    assert _patch_deepseek_v4_indexer_score_contract(model) == 1
    indexer = model.layers[0].self_attn.indexer

    class FakeCompressor:
        def __call__(self, x, rope, cache, start_pos, state_key="indexer_state"):
            del x, rope, cache, start_pos, state_key
            rows = np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1)
            return mx.array(np.repeat(rows, cfg["index_head_dim"], axis=-1))

    class FakeLinear:
        def __init__(self, width):
            self.width = width

        def __call__(self, x):
            return mx.ones((x.shape[0], x.shape[1], self.width), dtype=mx.float32)

    class IdentityRope:
        dims = cfg["index_head_dim"]

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            return x

    fused_calls = []

    def spy_fused(q, pooled_qat, weights, scale):
        fused_calls.append(
            (tuple(q.shape), tuple(pooled_qat.shape), float(scale))
        )
        scores = _composed_reference(mx, q, pooled_qat, weights, scale)
        return scores.reshape(-1)

    indexer._original.compressor = FakeCompressor()
    indexer._original.wq_b = FakeLinear(
        cfg["index_n_heads"] * cfg["index_head_dim"]
    )
    indexer._original.weights_proj = FakeLinear(cfg["index_n_heads"])
    monkeypatch.setattr(indexer_score_kernel, "_ENABLED", True)
    monkeypatch.setattr(indexer_score_kernel, "_metal_available", lambda: True)
    monkeypatch.setattr(
        indexer_score_kernel, "fused_qat_indexer_scores", spy_fused
    )

    decode_topk = indexer(
        mx.zeros((1, 1, cfg["hidden_size"]), dtype=mx.float32),
        mx.zeros((1, 1, cfg["q_lora_rank"]), dtype=mx.float32),
        IdentityRope(),
        IdentityRope(),
        None,
        16,
    )
    mx.eval(decode_topk)

    assert len(fused_calls) == 1
    assert fused_calls[0][0] == (1, cfg["index_n_heads"], 1, 128)
    assert fused_calls[0][1] == (1, 4, 128)
    assert indexer._moespresso_dsv4_indexer_score_contract_fused_score_calls == 1
    assert np.array(decode_topk).shape == (1, 1, 2)

    prefill_topk = indexer(
        mx.zeros((1, 2, cfg["hidden_size"]), dtype=mx.float32),
        mx.zeros((1, 2, cfg["q_lora_rank"]), dtype=mx.float32),
        IdentityRope(),
        IdentityRope(),
        None,
        7,
    )
    mx.eval(prefill_topk)

    assert len(fused_calls) == 1
    assert indexer._moespresso_dsv4_indexer_score_contract_fused_score_calls == 1
    assert np.array(prefill_topk).shape == (1, 2, 2)


def test_indexer_score_contract_decode_matches_composed_scores(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    jang_dsv4 = pytest.importorskip("jang_tools.dsv4.mlx_model")
    from moespresso.runtime.deepseek_v4.model import (
        _patch_deepseek_v4_indexer_score_contract,
        build_deepseek_v4_graph_from_manifest,
    )

    manifest = _tiny_ratio4_manifest()
    cfg = manifest["architecture"]["config"]
    model = build_deepseek_v4_graph_from_manifest(manifest)
    assert _patch_deepseek_v4_indexer_score_contract(model) == 1
    indexer = model.layers[0].self_attn.indexer

    rng = np.random.default_rng(7)
    pooled_rows = rng.standard_normal((1, 4, cfg["index_head_dim"])).astype(
        np.float32
    )
    q_rows = rng.standard_normal(
        (1, 1, cfg["index_n_heads"] * cfg["index_head_dim"])
    ).astype(np.float32)
    weight_rows = rng.standard_normal((1, 1, cfg["index_n_heads"])).astype(
        np.float32
    )

    class FakeCompressor:
        def __call__(self, x, rope, cache, start_pos, state_key="indexer_state"):
            del x, rope, cache, start_pos, state_key
            return mx.array(pooled_rows)

    class FakeWqB:
        def __call__(self, x):
            del x
            return mx.array(q_rows)

    class FakeWeightsProj:
        def __call__(self, x):
            del x
            return mx.array(weight_rows)

    class IdentityRope:
        dims = cfg["index_head_dim"]

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            return x

    captured = {}
    real_argpartition = jang_dsv4.mx.argpartition

    def spy_argpartition(values, kth, axis=-1):
        captured.setdefault("scores", []).append(np.array(-values))
        return real_argpartition(values, kth=kth, axis=axis)

    indexer._original.compressor = FakeCompressor()
    indexer._original.wq_b = FakeWqB()
    indexer._original.weights_proj = FakeWeightsProj()
    monkeypatch.setattr(jang_dsv4.mx, "argpartition", spy_argpartition)

    x = mx.zeros((1, 1, cfg["hidden_size"]), dtype=mx.float32)
    q_residual = mx.zeros((1, 1, cfg["q_lora_rank"]), dtype=mx.float32)

    monkeypatch.setattr(indexer_score_kernel, "_ENABLED", True)
    fused_topk = indexer(
        x, q_residual, IdentityRope(), IdentityRope(), None, 16
    )
    mx.eval(fused_topk)
    assert indexer._moespresso_dsv4_indexer_score_contract_fused_score_calls == 1

    monkeypatch.setattr(indexer_score_kernel, "_ENABLED", False)
    composed_topk = indexer(
        x, q_residual, IdentityRope(), IdentityRope(), None, 16
    )
    mx.eval(composed_topk)
    assert indexer._moespresso_dsv4_indexer_score_contract_fused_score_calls == 1

    fused_scores, composed_scores = captured["scores"]
    np.testing.assert_allclose(
        fused_scores,
        composed_scores,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(np.array(fused_topk), np.array(composed_topk))


def test_decode_indexer_qat_routes_by_gate_and_shape(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    from moespresso.runtime.deepseek_v4.model import (
        _dsv4_decode_indexer_qat,
        _dsv4_indexer_qat,
    )

    x = mx.array(
        np.random.default_rng(3).standard_normal(
            (1, 64, 1, 128)).astype(np.float32))
    mx.eval(x)
    calls = []
    original = indexer_score_kernel.indexer_qat_rows

    def spy(rows):
        calls.append(tuple(int(v) for v in rows.shape))
        return original(rows)

    monkeypatch.setattr(indexer_score_kernel, "indexer_qat_rows", spy)

    monkeypatch.delenv(
        "MOESPRESSO_DSV4_INDEXER_DECODE_QAT_KERNEL", raising=False)
    routed, used = _dsv4_decode_indexer_qat(mx, x)
    composed = _dsv4_indexer_qat(mx, x)
    mx.eval(routed, composed)
    assert used
    assert calls == [(1, 64, 1, 128)]
    np.testing.assert_array_equal(
        np.asarray(routed, dtype=np.float32).view(np.uint32),
        np.asarray(composed, dtype=np.float32).view(np.uint32))

    # Kill switch restores the composed chain.
    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_DECODE_QAT_KERNEL", "0")
    fallback, used = _dsv4_decode_indexer_qat(mx, x)
    mx.eval(fallback)
    assert not used
    assert calls == [(1, 64, 1, 128)]
    np.testing.assert_array_equal(
        np.asarray(fallback, dtype=np.float32).view(np.uint32),
        np.asarray(composed, dtype=np.float32).view(np.uint32))

    # Unsupported dtypes fail closed to the composed chain. Non-128-wide
    # rows are outside the QAT contract on both paths (the composed chain
    # raises the same load error the kernel wrapper would).
    monkeypatch.delenv(
        "MOESPRESSO_DSV4_INDEXER_DECODE_QAT_KERNEL", raising=False)
    _out, used = _dsv4_decode_indexer_qat(
        mx, mx.zeros((1, 2, 1, 128), dtype=mx.int32))
    assert not used
    assert calls == [(1, 64, 1, 128)]


def test_indexer_score_contract_counts_decode_qat_kernel(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    pytest.importorskip("jang_tools.dsv4.mlx_model")
    from moespresso.runtime.deepseek_v4.model import (
        _patch_deepseek_v4_indexer_score_contract,
        build_deepseek_v4_graph_from_manifest,
        deepseek_v4_indexer_layer_stats,
    )

    manifest = _tiny_ratio4_manifest()
    cfg = manifest["architecture"]["config"]
    model = build_deepseek_v4_graph_from_manifest(manifest)
    assert _patch_deepseek_v4_indexer_score_contract(model) == 1
    indexer = model.layers[0].self_attn.indexer

    class FakeCompressor:
        def __call__(self, x, rope, cache, start_pos, state_key="indexer_state"):
            del x, rope, cache, start_pos, state_key
            rows = np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1)
            return mx.array(np.repeat(rows, cfg["index_head_dim"], axis=-1))

    class FakeLinear:
        def __init__(self, width):
            self.width = width

        def __call__(self, x):
            return mx.ones((x.shape[0], x.shape[1], self.width), dtype=mx.float32)

    class IdentityRope:
        dims = cfg["index_head_dim"]

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            return x

    indexer._original.compressor = FakeCompressor()
    indexer._original.wq_b = FakeLinear(
        cfg["index_n_heads"] * cfg["index_head_dim"])
    indexer._original.weights_proj = FakeLinear(cfg["index_n_heads"])
    monkeypatch.setattr(indexer_score_kernel, "_ENABLED", False)
    monkeypatch.delenv(
        "MOESPRESSO_DSV4_INDEXER_DECODE_QAT_KERNEL", raising=False)

    x = mx.zeros((1, 1, cfg["hidden_size"]), dtype=mx.float32)
    q_residual = mx.zeros((1, 1, cfg["q_lora_rank"]), dtype=mx.float32)
    mx.eval(indexer(x, q_residual, IdentityRope(), IdentityRope(), None, 16))
    counter = (
        indexer._moespresso_dsv4_indexer_score_contract_decode_qat_kernel_calls
    )
    assert counter == 1

    # Prefill shapes keep the composed chain regardless of the gate.
    mx.eval(indexer(
        mx.zeros((1, 2, cfg["hidden_size"]), dtype=mx.float32),
        mx.zeros((1, 2, cfg["q_lora_rank"]), dtype=mx.float32),
        IdentityRope(), IdentityRope(), None, 7))
    counter = (
        indexer._moespresso_dsv4_indexer_score_contract_decode_qat_kernel_calls
    )
    assert counter == 1

    # Kill switch keeps the counter flat on decode shapes.
    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_DECODE_QAT_KERNEL", "0")
    mx.eval(indexer(x, q_residual, IdentityRope(), IdentityRope(), None, 24))
    counter = (
        indexer._moespresso_dsv4_indexer_score_contract_decode_qat_kernel_calls
    )
    assert counter == 1

    stats = deepseek_v4_indexer_layer_stats(model)[0]
    assert stats["indexer_score_contract_decode_qat_kernel_calls"] == 1


def test_prefill_pooled_qat_routes_to_kernel_and_matches_composed(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    from moespresso.runtime.deepseek_v4.model import (
        _dsv4_indexer_qat,
        _dsv4_prefill_pooled_qat,
    )

    x = mx.array(
        np.random.default_rng(77).standard_normal((1, 33, 128)).astype(np.float32)
    )
    mx.eval(x)

    calls = []
    original = indexer_score_kernel.indexer_qat_rows

    def spy(rows):
        calls.append(rows.shape)
        return original(rows)

    monkeypatch.setattr(indexer_score_kernel, "indexer_qat_rows", spy)
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_POOLED_QAT_KERNEL", "1")
    got = _dsv4_prefill_pooled_qat(mx, x)
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_POOLED_QAT_KERNEL", "0")
    composed = _dsv4_prefill_pooled_qat(mx, x)
    expected = _dsv4_indexer_qat(mx, x)
    mx.eval(got, composed, expected)

    assert calls == [(1, 33, 128)]
    got_bits = np.asarray(got, dtype=np.float32).view(np.uint32)
    expected_bits = np.asarray(expected, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(got_bits, expected_bits)
    composed_bits = np.asarray(composed, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(composed_bits, expected_bits)


def _composed_score_tail_reference(mx, scores4, weights_raw, *, valid, scale):
    """The composed fixed-state score tail, negated for the selection."""
    from moespresso.runtime.deepseek_v4 import fixed_decode_state as fds

    n_heads = int(scores4.shape[1])
    weights = weights_raw.astype(mx.float32) * (n_heads ** -0.5)
    fixed = mx.maximum(scores4, 0) * scale
    fixed = (fixed * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
    params = fds.decode_step_params(mx, offset=3844, pool_rows=valid)
    fixed = fds.pad_scores_to_capacity(mx, fixed, params)
    return -fixed


def _score_tail_inputs(mx, *, n_heads, capacity, seed, w_dtype, special=False):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_heads, capacity)).astype(np.float32)
    raw *= np.float32(10.0) ** rng.integers(
        -8, 8, size=(n_heads, capacity)).astype(np.float32)
    if special:
        raw.flat[rng.integers(0, raw.size, 64)] = -0.0
        raw.flat[rng.integers(0, raw.size, 8)] = np.inf
        raw.flat[rng.integers(0, raw.size, 8)] = -np.inf
        raw.flat[rng.integers(0, raw.size, 8)] = np.nan
        raw.flat[rng.integers(0, raw.size, 32)] = 1e-42
    w = rng.standard_normal(n_heads).astype(np.float32)
    w *= np.float32(10.0) ** rng.integers(-3, 3, size=n_heads).astype(
        np.float32)
    scores4 = mx.array(raw).reshape(1, n_heads, 1, capacity)
    weights_raw = mx.array(w).astype(w_dtype).reshape(1, 1, n_heads)
    mx.eval(scores4, weights_raw)
    return scores4, weights_raw


# The served fixed-state decode shape is 64 index heads over a 1024-row
# capacity buffer; 963 is the seeded-probe valid-row count, the smaller
# geometries cover the strided-loop and column tails, and 32 is the lower
# edge of the col_reduce_looped selection window the transcription
# mirrors.
@pytest.mark.parametrize("n_heads", [32, 48, 64])
@pytest.mark.parametrize(
    "capacity,valid", [(1024, 963), (1024, 1024), (963, 963), (33, 20)]
)
@pytest.mark.parametrize("w_dtype_name", ["bfloat16", "float16", "float32"])
def test_fused_score_tail_matches_composed_tail(
    n_heads, capacity, valid, w_dtype_name
):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    scale = 128 ** -0.5
    scores4, weights_raw = _score_tail_inputs(
        mx, n_heads=n_heads, capacity=capacity, seed=100 + n_heads + capacity,
        w_dtype=getattr(mx, w_dtype_name))
    params = mx.array([3844, valid], dtype=mx.int32)
    got = indexer_score_kernel.fused_score_tail(
        scores4, weights_raw, params, scale=scale)
    expected = _composed_score_tail_reference(
        mx, scores4, weights_raw, valid=valid, scale=scale)
    mx.eval(got, expected)

    assert got.shape == (1, 1, capacity)
    assert got.dtype == mx.float32
    np.testing.assert_array_equal(
        np.asarray(got, dtype=np.float32).view(np.uint32),
        np.asarray(expected, dtype=np.float32).view(np.uint32))


def test_fused_score_tail_matches_composed_tail_special_values():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    scale = 128 ** -0.5
    scores4, weights_raw = _score_tail_inputs(
        mx, n_heads=64, capacity=1024, seed=41, w_dtype=mx.bfloat16,
        special=True)
    params = mx.array([3844, 963], dtype=mx.int32)
    got = indexer_score_kernel.fused_score_tail(
        scores4, weights_raw, params, scale=scale)
    expected = _composed_score_tail_reference(
        mx, scores4, weights_raw, valid=963, scale=scale)
    mx.eval(got, expected)

    np.testing.assert_array_equal(
        np.asarray(got, dtype=np.float32).view(np.uint32),
        np.asarray(expected, dtype=np.float32).view(np.uint32))


def test_fused_score_tail_selection_set_and_order_parity():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    scale = 128 ** -0.5
    for topk, seed in ((512, 7), (13, 8)):
        scores4, weights_raw = _score_tail_inputs(
            mx, n_heads=64, capacity=1024, seed=seed, w_dtype=mx.bfloat16)
        params = mx.array([3844, 963], dtype=mx.int32)
        neg = indexer_score_kernel.fused_score_tail(
            scores4, weights_raw, params, scale=scale)
        expected = _composed_score_tail_reference(
            mx, scores4, weights_raw, valid=963, scale=scale)
        got_topk = mx.argpartition(
            neg[..., :963], kth=topk - 1, axis=-1)[..., :topk]
        ref_topk = mx.argpartition(
            expected[..., :963], kth=topk - 1, axis=-1)[..., :topk]
        mx.eval(got_topk, ref_topk)
        # Bit-identical selection input means both the set and the order
        # match the composed path exactly.
        np.testing.assert_array_equal(
            np.asarray(got_topk), np.asarray(ref_topk))


def test_fused_score_tail_boundary_ties_match_composed_selection():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    # Duplicated score columns force ties at the selection boundary; the
    # fused tail feeds the argpartition the same bits, so the tie
    # resolution is exactly the stock one.
    scale = 128 ** -0.5
    rng = np.random.default_rng(23)
    base = rng.standard_normal((64, 8)).astype(np.float32)
    raw = np.repeat(base, 128, axis=1)[:, :1000]
    scores4 = mx.array(raw).reshape(1, 64, 1, 1000)
    weights_raw = mx.array(
        rng.standard_normal(64).astype(np.float32)).reshape(1, 1, 64)
    params = mx.array([3844, 990], dtype=mx.int32)
    mx.eval(scores4, weights_raw)

    neg = indexer_score_kernel.fused_score_tail(
        scores4, weights_raw, params, scale=scale)
    expected = _composed_score_tail_reference(
        mx, scores4, weights_raw, valid=990, scale=scale)
    mx.eval(neg, expected)
    np.testing.assert_array_equal(
        np.asarray(neg, dtype=np.float32).view(np.uint32),
        np.asarray(expected, dtype=np.float32).view(np.uint32))

    got_topk = mx.argpartition(
        neg[..., :990], kth=511, axis=-1)[..., :512]
    ref_topk = mx.argpartition(
        expected[..., :990], kth=511, axis=-1)[..., :512]
    mx.eval(got_topk, ref_topk)
    np.testing.assert_array_equal(np.asarray(got_topk), np.asarray(ref_topk))


def test_fused_score_tail_rejects_bad_inputs():
    mx = pytest.importorskip("mlx.core")

    scores = mx.zeros((1, 64, 1, 1024), dtype=mx.float32)
    weights = mx.zeros((1, 1, 64), dtype=mx.float32)
    params = mx.zeros((2,), dtype=mx.int32)

    def call(**overrides):
        args = dict(scores=scores, weights_raw=weights, params=params,
                    scale=1.0)
        args.update(overrides)
        return indexer_score_kernel.fused_score_tail(
            args["scores"], args["weights_raw"], args["params"],
            scale=args["scale"])

    with pytest.raises(ValueError, match="shape"):
        call(scores=mx.zeros((64, 1024), dtype=mx.float32))
    with pytest.raises(ValueError, match="one decode token"):
        call(scores=mx.zeros((1, 64, 2, 1024), dtype=mx.float32))
    with pytest.raises(ValueError, match="one decode token"):
        call(scores=mx.zeros((2, 64, 1, 1024), dtype=mx.float32))
    with pytest.raises(ValueError, match="32 to 64 index heads"):
        call(scores=mx.zeros((1, 31, 1, 1024), dtype=mx.float32),
             weights_raw=mx.zeros((1, 1, 31), dtype=mx.float32))
    with pytest.raises(ValueError, match="32 to 64 index heads"):
        call(scores=mx.zeros((1, 65, 1, 1024), dtype=mx.float32),
             weights_raw=mx.zeros((1, 1, 65), dtype=mx.float32))
    with pytest.raises(ValueError, match="scores must be float32"):
        call(scores=mx.zeros((1, 64, 1, 1024), dtype=mx.float16))
    with pytest.raises(ValueError, match="weights_raw must have shape"):
        call(weights_raw=mx.zeros((1, 1, 32), dtype=mx.float32))
    with pytest.raises(ValueError, match="float32, float16, or bfloat16"):
        call(weights_raw=mx.zeros((1, 1, 64), dtype=mx.int32))
    with pytest.raises(ValueError, match="params must be int32"):
        call(params=mx.zeros((2,), dtype=mx.int64))
    with pytest.raises(ValueError, match="params must be int32"):
        call(params=mx.zeros((1,), dtype=mx.int32))


def test_score_tail_eligible_gate_and_shape_predicate(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setattr(indexer_score_kernel, "_metal_available", lambda: True)

    scores = mx.zeros((1, 64, 1, 1024), dtype=mx.float32)
    weights = mx.zeros((1, 1, 64), dtype=mx.bfloat16)
    params = mx.zeros((2,), dtype=mx.int32)

    def eligible(**overrides):
        args = dict(scores=scores, weights_raw=weights, params=params)
        args.update(overrides)
        return indexer_score_kernel.score_tail_eligible(
            args["scores"], args["weights_raw"], args["params"])

    monkeypatch.delenv("MOESPRESSO_DSV4_INDEXER_CHAIN_TRIMS", raising=False)
    monkeypatch.delenv("MOESPRESSO_DSV4_INDEXER_SCORE_TAIL", raising=False)
    assert indexer_score_kernel.score_tail_enabled()
    assert eligible()
    assert eligible(weights_raw=weights.astype(mx.float16))
    assert eligible(weights_raw=weights.astype(mx.float32))
    assert eligible(scores=mx.zeros((1, 32, 1, 7), dtype=mx.float32),
                    weights_raw=mx.zeros((1, 1, 32), dtype=mx.float32))

    # Both kill switches are read per call: the family switch covers every
    # indexer chain trim, the piece switch only this seam.
    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_CHAIN_TRIMS", "0")
    assert not eligible()
    monkeypatch.delenv("MOESPRESSO_DSV4_INDEXER_CHAIN_TRIMS", raising=False)
    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_SCORE_TAIL", "0")
    assert not eligible()
    monkeypatch.delenv("MOESPRESSO_DSV4_INDEXER_SCORE_TAIL", raising=False)
    assert eligible()

    # Shape and dtype vetoes fail closed.
    assert not eligible(scores=mx.zeros((1, 64, 2, 1024), dtype=mx.float32))
    assert not eligible(scores=mx.zeros((2, 64, 1, 1024), dtype=mx.float32))
    assert not eligible(scores=mx.zeros((1, 31, 1, 1024), dtype=mx.float32),
                        weights_raw=mx.zeros((1, 1, 31), dtype=mx.float32))
    assert not eligible(scores=mx.zeros((1, 65, 1, 1024), dtype=mx.float32),
                        weights_raw=mx.zeros((1, 1, 65), dtype=mx.float32))
    assert not eligible(scores=scores.astype(mx.float16))
    assert not eligible(weights_raw=mx.zeros((1, 2, 64), dtype=mx.float32))
    assert not eligible(weights_raw=mx.zeros((1, 1, 64), dtype=mx.int32))
    assert not eligible(params=mx.zeros((2,), dtype=mx.int64))
    assert not eligible(params=mx.zeros((1,), dtype=mx.int32))

    monkeypatch.setattr(indexer_score_kernel, "_metal_available", lambda: False)
    assert not eligible()


def _engaged_indexer_fixed_cache(offset, *, pool_capacity=8):
    """A cache stub whose indexer state carries an engaged fixed branch."""
    from moespresso.runtime.deepseek_v4 import fixed_decode_state as fds

    class _FixedStateCache:
        pass

    cache = _FixedStateCache()
    state: dict = {}
    cache.indexer_state = state
    branch = fds._FixedBranchState(
        ratio=4, overlap=True, head_dim=128, pool_capacity=pool_capacity)
    branch.engaged_pos = int(offset)
    for key in ("buffer_kv", "buffer_gate", "pooled"):
        branch.mirrors[key] = state.get(key)
    setattr(cache, fds._BRANCHES_ATTR, {"indexer_state": branch})
    return cache


def test_indexer_score_contract_score_tail_matches_composed(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    pytest.importorskip("jang_tools.dsv4.mlx_model")
    from moespresso.runtime.deepseek_v4.model import (
        _patch_deepseek_v4_indexer_score_contract,
        build_deepseek_v4_graph_from_manifest,
        deepseek_v4_indexer_layer_stats,
    )

    manifest = _tiny_ratio4_manifest(index_n_heads=32)
    cfg = manifest["architecture"]["config"]
    model = build_deepseek_v4_graph_from_manifest(manifest)
    assert _patch_deepseek_v4_indexer_score_contract(model) == 1
    indexer = model.layers[0].self_attn.indexer

    rng = np.random.default_rng(19)
    pooled_rows = rng.standard_normal((1, 4, 128)).astype(np.float32)
    q_rows = rng.standard_normal((1, 1, 32 * 128)).astype(np.float32)
    weight_rows = rng.standard_normal((1, 1, 32)).astype(np.float32)

    class FakeCompressor:
        def __call__(self, x, rope, cache, start_pos, state_key="indexer_state"):
            del x, rope, cache, start_pos, state_key
            return mx.array(pooled_rows)

    class FakeWqB:
        def __call__(self, x):
            del x
            return mx.array(q_rows)

    class FakeWeightsProj:
        def __call__(self, x):
            del x
            return mx.array(weight_rows).astype(mx.bfloat16)

    class IdentityRope:
        dims = 128

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            return x

    indexer._original.compressor = FakeCompressor()
    indexer._original.wq_b = FakeWqB()
    indexer._original.weights_proj = FakeWeightsProj()
    monkeypatch.setattr(indexer_score_kernel, "_ENABLED", False)
    monkeypatch.delenv("MOESPRESSO_DSV4_INDEXER_CHAIN_TRIMS", raising=False)
    monkeypatch.delenv("MOESPRESSO_DSV4_INDEXER_SCORE_TAIL", raising=False)
    monkeypatch.delenv("MOESPRESSO_DSV4_INDEXER_DUMP_PREFIX", raising=False)

    x = mx.zeros((1, 1, cfg["hidden_size"]), dtype=mx.float32)
    q_residual = mx.zeros((1, 1, cfg["q_lora_rank"]), dtype=mx.float32)

    fused_topk = indexer(
        x, q_residual, IdentityRope(), IdentityRope(),
        _engaged_indexer_fixed_cache(16), 16)
    mx.eval(fused_topk)
    counts = deepseek_v4_indexer_layer_stats(model)[0]
    assert counts["indexer_score_contract_score_tail_kernel_calls"] == 1
    assert counts["indexer_score_contract_fixed_state_calls"] == 1
    assert counts["indexer_score_contract_decode_qat_kernel_calls"] == 1

    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_SCORE_TAIL", "0")
    composed_topk = indexer(
        x, q_residual, IdentityRope(), IdentityRope(),
        _engaged_indexer_fixed_cache(16), 16)
    mx.eval(composed_topk)
    counts = deepseek_v4_indexer_layer_stats(model)[0]
    assert counts["indexer_score_contract_score_tail_kernel_calls"] == 1
    assert counts["indexer_score_contract_fixed_state_calls"] == 2

    # The fused tail feeds the stock argpartition bit-identical input, so
    # the selection matches in both set and order.
    np.testing.assert_array_equal(
        np.asarray(fused_topk), np.asarray(composed_topk))

    # The selection dump rides the composed chain, so the dump env vetoes
    # the fused tail.
    monkeypatch.delenv("MOESPRESSO_DSV4_INDEXER_SCORE_TAIL", raising=False)
    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_DUMP_PREFIX", "/tmp/never")
    dump_topk = indexer(
        x, q_residual, IdentityRope(), IdentityRope(),
        _engaged_indexer_fixed_cache(16), 16)
    mx.eval(dump_topk)
    counts = deepseek_v4_indexer_layer_stats(model)[0]
    assert counts["indexer_score_contract_score_tail_kernel_calls"] == 1
    assert counts["indexer_score_contract_fixed_state_calls"] == 3
    np.testing.assert_array_equal(
        np.asarray(dump_topk), np.asarray(composed_topk))
