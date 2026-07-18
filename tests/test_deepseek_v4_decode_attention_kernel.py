from __future__ import annotations

import numpy as np
import pytest

from moespresso.runtime.deepseek_v4 import decode_attention_kernel


YARN_CONFIG = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "type": "yarn",
}
OFFSET = 3844


def _require_metal(mx):
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")


def _compress_rope(jm):
    return jm.DeepseekV4RoPE(64, 160000, YARN_CONFIG, 1048576)


def _composed_scores(mx, q_roped, pooled_qat, weights, scale):
    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

    q = _dsv4_indexer_qat(mx, q_roped)
    scores = (
        q.astype(mx.float32)
        @ pooled_qat[:, None].swapaxes(-1, -2).astype(mx.float32)
    )
    scores = mx.maximum(scores, 0) * scale
    return (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)


def _prep_inputs(mx, jm, *, n_rows, n_heads=64, seed=0):
    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

    rng = np.random.default_rng(seed)
    rope = _compress_rope(jm)
    q_idx = mx.array(
        rng.standard_normal((1, n_heads, 1, 128), dtype=np.float32)
    ).astype(mx.float16)
    q_roped = jm._apply_partial_rope(q_idx, rope, OFFSET)
    pooled_qat = _dsv4_indexer_qat(
        mx, mx.array(rng.standard_normal((1, n_rows, 128), dtype=np.float32)))
    weights = mx.array(
        rng.standard_normal((1, 1, n_heads), dtype=np.float32)
    ) * (n_heads ** -0.5)
    kv_row = mx.array(
        rng.standard_normal((1, 1, 1, 512), dtype=np.float32) * 0.05
    ).astype(mx.float16)
    params = mx.array([OFFSET, 0], dtype=mx.int32)
    mx.eval(q_roped, pooled_qat, weights, kv_row)
    return rope, q_roped, pooled_qat, weights, kv_row, params


# The served decode shape is 64 index heads over a pool larger than the 512
# top-k width; 961 is the fenced A/B pool size and the odd counts cover the
# blocked-emission tail.
@pytest.mark.parametrize("n_rows", [531, 961, 977])
def test_fused_decode_prep_scores_and_topk_match_composed(n_rows):
    mx = pytest.importorskip("mlx.core")
    _require_metal(mx)
    jm = pytest.importorskip("jang_tools.dsv4.mlx_model")

    rope, q_roped, pooled_qat, weights, kv_row, params = _prep_inputs(
        mx, jm, n_rows=n_rows, seed=100 + n_rows)
    scale = 128 ** -0.5
    topk = 512

    sel, _row, scores = decode_attention_kernel.fused_decode_prep(
        q_roped.reshape(64, 128),
        pooled_qat.reshape(n_rows, 128),
        weights.reshape(64),
        kv_row.reshape(512),
        rope.inv_freq,
        params,
        scale=scale,
        topk=topk,
    )
    ref_scores = _composed_scores(mx, q_roped, pooled_qat, weights, scale)
    ref_topk = mx.argpartition(-ref_scores, kth=topk - 1, axis=-1)[..., :topk]
    mx.eval(sel, scores, ref_scores, ref_topk)

    np.testing.assert_allclose(
        np.asarray(scores),
        np.asarray(ref_scores).reshape(-1),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    got = set(np.asarray(sel).tolist())
    expected = set(np.asarray(ref_topk).reshape(-1).tolist())
    assert len(got) == topk
    assert got == expected


@pytest.mark.parametrize("case", ["normal", "large", "tiny", "zeros"])
def test_fused_decode_prep_kv_row_bitexact(case):
    mx = pytest.importorskip("mlx.core")
    _require_metal(mx)
    jm = pytest.importorskip("jang_tools.dsv4.mlx_model")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_fp8_kv_roundtrip,
    )

    rng = np.random.default_rng(7)
    if case == "normal":
        row = rng.standard_normal(512).astype(np.float32) * 0.05
    elif case == "large":
        row = rng.standard_normal(512).astype(np.float32) * 1.0e4
    elif case == "tiny":
        row = rng.standard_normal(512).astype(np.float32) * 1.0e-6
    else:
        row = np.zeros(512, dtype=np.float32)
    rope, q_roped, pooled_qat, weights, _kv, params = _prep_inputs(
        mx, jm, n_rows=531, seed=8)
    kv_row = mx.array(row.reshape(1, 1, 1, 512)).astype(mx.float16)
    mx.eval(kv_row)

    _sel, got_row, _scores = decode_attention_kernel.fused_decode_prep(
        q_roped.reshape(64, 128),
        pooled_qat.reshape(531, 128),
        weights.reshape(64),
        kv_row.reshape(512),
        rope.inv_freq,
        params,
        scale=128 ** -0.5,
        topk=512,
    )
    expected = _deepseek_v4_fp8_kv_roundtrip(
        jm._apply_partial_rope(kv_row, rope, OFFSET))
    mx.eval(got_row, expected)

    got_bits = np.asarray(got_row).view(np.uint16)
    expected_bits = np.asarray(expected).reshape(-1).view(np.uint16)
    np.testing.assert_array_equal(got_bits, expected_bits)


@pytest.mark.parametrize(
    "case", ["normal", "large", "tiny", "zeros", "mixed_rows"]
)
def test_fp8_kv_prefix_rows_bit_identical_to_composed(monkeypatch, case):
    mx = pytest.importorskip("mlx.core")
    _require_metal(mx)
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_fp8_kv_roundtrip,
    )

    rng = np.random.default_rng(21)
    if case == "normal":
        rows = rng.standard_normal((97, 512)).astype(np.float32) * 0.05
    elif case == "large":
        rows = rng.standard_normal((97, 512)).astype(np.float32) * 1.0e4
    elif case == "tiny":
        rows = rng.standard_normal((97, 512)).astype(np.float32) * 1.0e-6
    elif case == "zeros":
        rows = np.zeros((5, 512), dtype=np.float32)
    else:
        rows = np.concatenate(
            [
                rng.standard_normal((32, 512)).astype(np.float32) * s
                for s in (1.0, 1.0e-8, 1.0e6)
            ]
        )
    x = mx.array(rows.reshape(1, -1, 512))
    mx.eval(x)

    got = decode_attention_kernel.fp8_kv_prefix_rows(x)
    monkeypatch.setenv("MOESPRESSO_DSV4_FP8_KV_KERNEL", "0")
    expected = _deepseek_v4_fp8_kv_roundtrip(x)
    mx.eval(got, expected)

    assert got.dtype == mx.float32
    got_bits = np.asarray(got, dtype=np.float32).view(np.uint32)
    expected_bits = np.asarray(expected, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(got_bits, expected_bits)


def test_fp8_kv_roundtrip_routes_to_kernel_by_gate(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    _require_metal(mx)
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_fp8_kv_roundtrip,
    )

    x = mx.array(
        np.random.default_rng(3).standard_normal((1, 9, 512)).astype(np.float32))
    mx.eval(x)
    calls = []
    original = decode_attention_kernel.fp8_kv_prefix_rows

    def spy(arr):
        calls.append(arr.shape)
        return original(arr)

    monkeypatch.setattr(decode_attention_kernel, "fp8_kv_prefix_rows", spy)
    monkeypatch.setenv("MOESPRESSO_DSV4_FP8_KV_KERNEL", "1")
    mx.eval(_deepseek_v4_fp8_kv_roundtrip(x))
    assert calls == [(1, 9, 512)]

    monkeypatch.setenv("MOESPRESSO_DSV4_FP8_KV_KERNEL", "0")
    mx.eval(_deepseek_v4_fp8_kv_roundtrip(x))
    assert calls == [(1, 9, 512)]
    # Non-float32 rows stay on the composed path regardless of the gate.
    monkeypatch.setenv("MOESPRESSO_DSV4_FP8_KV_KERNEL", "1")
    mx.eval(_deepseek_v4_fp8_kv_roundtrip(x.astype(mx.float16)))
    assert calls == [(1, 9, 512)]


def test_fused_decode_prep_boundary_ties_take_lowest_rows():
    mx = pytest.importorskip("mlx.core")
    _require_metal(mx)
    jm = pytest.importorskip("jang_tools.dsv4.mlx_model")

    # Every pooled row identical: every score ties at the top-k boundary and
    # the deterministic contract selects the lowest row indices in order.
    n_rows, topk = 600, 512
    rope, q_roped, _pooled, weights, kv_row, params = _prep_inputs(
        mx, jm, n_rows=n_rows, seed=9)
    one_row = mx.random.normal((1, 1, 128))
    pooled_qat = mx.broadcast_to(one_row, (1, n_rows, 128)).astype(mx.float32)
    pooled_qat = mx.contiguous(pooled_qat)
    mx.eval(pooled_qat)

    sel, _row, scores = decode_attention_kernel.fused_decode_prep(
        q_roped.reshape(64, 128),
        pooled_qat.reshape(n_rows, 128),
        weights.reshape(64),
        kv_row.reshape(512),
        rope.inv_freq,
        params,
        scale=128 ** -0.5,
        topk=topk,
    )
    mx.eval(sel, scores)
    assert len(np.unique(np.asarray(scores))) == 1
    np.testing.assert_array_equal(
        np.asarray(sel), np.arange(topk, dtype=np.int32))


def test_fused_decode_prep_rejects_bad_inputs():
    mx = pytest.importorskip("mlx.core")

    q = mx.zeros((64, 128), dtype=mx.float16)
    pooled = mx.zeros((600, 128), dtype=mx.float32)
    weights = mx.zeros((64,), dtype=mx.float32)
    kv = mx.zeros((512,), dtype=mx.float16)
    inv_freq = mx.zeros((32,), dtype=mx.float32)
    params = mx.zeros((2,), dtype=mx.int32)

    def call(**overrides):
        kwargs = dict(scale=1.0, topk=512)
        args = dict(q_idx=q, pooled_qat=pooled, weights=weights, kv_row=kv,
                    inv_freq=inv_freq, params=params)
        args.update(overrides)
        return decode_attention_kernel.fused_decode_prep(
            args["q_idx"], args["pooled_qat"], args["weights"],
            args["kv_row"], args["inv_freq"], args["params"], **kwargs)

    with pytest.raises(ValueError, match="n_heads, 128"):
        call(q_idx=mx.zeros((64, 64), dtype=mx.float16))
    with pytest.raises(ValueError, match="1 and 64 heads"):
        call(q_idx=mx.zeros((65, 128), dtype=mx.float16))
    with pytest.raises(ValueError, match="float32, float16, or bfloat16"):
        call(q_idx=mx.zeros((64, 128), dtype=mx.int32))
    with pytest.raises(ValueError, match="float32 \\[n_rows, 128\\]"):
        call(pooled_qat=pooled.astype(mx.float16))
    with pytest.raises(ValueError, match="smaller than n_rows"):
        call(pooled_qat=mx.zeros((512, 128), dtype=mx.float32))
    with pytest.raises(ValueError, match="16 bits"):
        call(pooled_qat=mx.zeros((1 << 16, 128), dtype=mx.float32))
    with pytest.raises(ValueError, match="float32 \\[n_heads\\]"):
        call(weights=weights.astype(mx.float16))
    with pytest.raises(ValueError, match="float16 \\[512\\]"):
        call(kv_row=mx.zeros((512,), dtype=mx.float32))
    with pytest.raises(ValueError, match="float32 \\[32\\]"):
        call(inv_freq=mx.zeros((16,), dtype=mx.float32))


def test_fused_decode_sdpa_matches_composed_chain():
    mx = pytest.importorskip("mlx.core")
    _require_metal(mx)
    jm = pytest.importorskip("jang_tools.dsv4.mlx_model")
    from mlx_lm.models.base import scaled_dot_product_attention

    rng = np.random.default_rng(21)
    rope = _compress_rope(jm)
    n_rows, topk, write_idx = 961, 512, 37
    q = mx.array(
        rng.standard_normal((1, 1, 64, 512), dtype=np.float32) * 0.1
    ).astype(mx.float16)
    window = mx.array(
        rng.standard_normal((1, 1, 128, 512), dtype=np.float32) * 0.1
    ).astype(mx.float16)
    row = mx.array(
        rng.standard_normal((512,), dtype=np.float32) * 0.1
    ).astype(mx.float16)
    pooled = mx.array(
        rng.standard_normal((1, n_rows, 512), dtype=np.float32) * 0.1
    ).astype(mx.float16)
    sel = mx.array(np.sort(
        rng.choice(n_rows, size=topk, replace=False)).astype(np.int32))
    sinks = mx.array(
        rng.standard_normal((64,), dtype=np.float32)).astype(mx.float16)
    params = mx.array([OFFSET, write_idx], dtype=mx.int32)
    mx.eval(q, window, row, pooled, sel, sinks)

    heads = decode_attention_kernel.fused_decode_sdpa(
        q.reshape(64, 512), window.reshape(128, 512), row,
        pooled.reshape(n_rows, 512), sel, sinks, rope.inv_freq, params,
        scale=512 ** -0.5,
    )

    q_ref = jm._apply_partial_rope(q.transpose(0, 2, 1, 3), rope, OFFSET)
    window_sub = mx.concatenate(
        [window[:, :, :write_idx], row.reshape(1, 1, 1, 512),
         window[:, :, write_idx + 1:]], axis=2)
    full_kv = mx.concatenate(
        [window_sub, pooled[0][sel][None, None]], axis=2)
    ref = scaled_dot_product_attention(
        q_ref, full_kv, full_kv,
        cache=None, scale=512 ** -0.5, mask=None, sinks=sinks,
    )
    ref = jm._apply_partial_rope(ref, rope, OFFSET, inverse=True)
    ref = ref.transpose(0, 2, 1, 3).reshape(64, 512)
    mx.eval(heads, ref)

    diff = np.abs(
        np.asarray(heads, dtype=np.float32) - np.asarray(ref, dtype=np.float32))
    # Float16 tolerance surface: the accumulation partitioning differs from
    # the mx.fast SDPA kernel. The banded-attention change was gate-safe at
    # 7e-3 on the layer output; this bound is far inside that scale.
    assert float(diff.max()) < 2.0e-3


def test_fused_decode_sdpa_rejects_bad_inputs():
    mx = pytest.importorskip("mlx.core")

    q = mx.zeros((64, 512), dtype=mx.float16)
    window = mx.zeros((128, 512), dtype=mx.float16)
    row = mx.zeros((512,), dtype=mx.float16)
    pooled = mx.zeros((961, 512), dtype=mx.float16)
    sel = mx.zeros((512,), dtype=mx.int32)
    sinks = mx.zeros((64,), dtype=mx.float16)
    inv_freq = mx.zeros((32,), dtype=mx.float32)
    params = mx.zeros((2,), dtype=mx.int32)

    def call(**overrides):
        args = dict(q=q, window=window, row=row, pooled=pooled, sel=sel,
                    sinks=sinks, inv_freq=inv_freq, params=params)
        args.update(overrides)
        return decode_attention_kernel.fused_decode_sdpa(
            args["q"], args["window"], args["row"], args["pooled"],
            args["sel"], args["sinks"], args["inv_freq"], args["params"],
            scale=1.0)

    with pytest.raises(ValueError, match="float16 \\[n_heads, 512\\]"):
        call(q=q.astype(mx.float32))
    with pytest.raises(ValueError, match="window_rows"):
        call(window=mx.zeros((128, 256), dtype=mx.float16))
    with pytest.raises(ValueError, match="row must be float16"):
        call(row=mx.zeros((512,), dtype=mx.float32))
    with pytest.raises(ValueError, match="pooled must be float16"):
        call(pooled=pooled.astype(mx.float32))
    with pytest.raises(ValueError, match="int32"):
        call(sel=sel.astype(mx.int64))
    with pytest.raises(ValueError, match="sinks"):
        call(sinks=mx.zeros((32,), dtype=mx.float16))


def _build_synthetic_attention(mx, jm):
    """Reduced-hidden DS4 attention with the real head geometry."""
    from mlx.utils import tree_map
    from moespresso.runtime.deepseek_v4.model import (
        _AttentionCompressorFp8KV,
        _IndexerDS4ScoreContract,
    )

    args = jm.ModelArgs(
        hidden_size=256,
        num_attention_heads=8,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=64,
        o_lora_rank=64,
        o_groups=2,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=64,
        sliding_window=128,
        compress_ratios=[4],
        rope_scaling=YARN_CONFIG,
    )
    mx.random.seed(11)
    attn = jm.DeepseekV4Attention(args, layer_id=0)
    attn.update(tree_map(lambda p: p.astype(mx.float16), attn.parameters()))
    attn.attn_sink = (mx.random.normal((8,)) * 0.3).astype(mx.float32)
    object.__setattr__(
        attn, "indexer",
        _IndexerDS4ScoreContract(
            attn.indexer, 4, layer_index=0, mx=mx, dsv4_model=jm))
    object.__setattr__(
        attn, "compressor",
        _AttentionCompressorFp8KV(attn.compressor, indexer=attn.indexer))
    return args, attn


def _wrap_fp8_cache(cache):
    """Install the served FP8 KV cache update contract."""
    from moespresso.runtime.deepseek_v4.model import _cache_with_fp8_kv_roundtrip

    return _cache_with_fp8_kv_roundtrip(cache)


def _make_steady_cache(mx, jm, *, rows, offset=OFFSET, fp8_marker=True):
    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

    rng = np.random.default_rng(1234)
    cache = jm.DeepseekV4Cache(128, compress_ratio=4)
    keys = mx.array(
        rng.standard_normal((1, 1, 128, 512), dtype=np.float32) * 0.1
    ).astype(mx.float16)
    cache.local.keys = keys
    cache.local.values = keys
    cache.local.offset = offset
    cache.local._idx = 37
    cache.compressor_state["pooled"] = mx.array(
        rng.standard_normal((1, rows, 512), dtype=np.float32) * 0.1
    ).astype(mx.float16)
    idx_pool = mx.array(
        rng.standard_normal((1, rows, 128), dtype=np.float32)
    ).astype(mx.float16)
    cache.indexer_state["pooled"] = idx_pool
    cache.indexer_state["pooled_qat"] = _dsv4_indexer_qat(mx, idx_pool)
    cache.indexer_state["pooled_qat_rows"] = rows
    mx.eval(cache.local.keys, cache.compressor_state["pooled"],
            cache.indexer_state["pooled_qat"])
    if fp8_marker:
        return _wrap_fp8_cache(cache)
    return cache


def _cache_signature(mx, cache):
    return {
        "keys": np.asarray(cache.local.keys.astype(mx.float32)),
        "values": np.asarray(cache.local.values.astype(mx.float32)),
        "offset": cache.local.offset,
        "idx": cache.local._idx,
        "qat_rows": cache.indexer_state.get("pooled_qat_rows"),
        "qat": np.asarray(cache.indexer_state["pooled_qat"]),
    }


def test_fused_decode_wrapper_matches_composed_layer(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    _require_metal(mx)
    jm = pytest.importorskip("jang_tools.dsv4.mlx_model")
    from moespresso.runtime.deepseek_v4.model import (
        _Ratio4DecodeFusedAttention,
    )

    monkeypatch.setattr(decode_attention_kernel, "_ENABLED", True)
    args, attn = _build_synthetic_attention(mx, jm)
    wrapper = _Ratio4DecodeFusedAttention(
        attn, mx=mx, dsv4_model=jm, cache_cls=jm.DeepseekV4Cache)
    x = (mx.random.normal((1, 1, args.hidden_size)) * 0.05).astype(mx.float16)
    mx.eval(x)

    cache_ref = _make_steady_cache(mx, jm, rows=96)
    out_ref = attn(x, mask=None, cache=cache_ref)
    cache_fused = _make_steady_cache(mx, jm, rows=96)
    out_fused = wrapper(x, mask=None, cache=cache_fused)
    mx.eval(out_ref, out_fused,
            cache_ref.local.keys, cache_fused.local.keys,
            cache_ref.local.values, cache_fused.local.values)

    assert wrapper.fused_decode_calls == 1
    assert wrapper.fused_decode_composed_tail_calls == 0

    sig_ref = _cache_signature(mx, cache_ref)
    sig_fused = _cache_signature(mx, cache_fused)
    assert sig_ref["offset"] == sig_fused["offset"]
    assert sig_ref["idx"] == sig_fused["idx"]
    assert sig_ref["qat_rows"] == sig_fused["qat_rows"]
    # The prepared KV row contract is bit identity with the composed rope +
    # FP8 round trip, so the whole cache stays bit-equal.
    np.testing.assert_array_equal(sig_ref["keys"], sig_fused["keys"])
    np.testing.assert_array_equal(sig_ref["values"], sig_fused["values"])
    np.testing.assert_array_equal(sig_ref["qat"], sig_fused["qat"])

    diff = np.abs(
        np.asarray(out_ref.astype(mx.float32))
        - np.asarray(out_fused.astype(mx.float32)))
    assert float(diff.max()) < 2.0e-3


def test_fused_decode_wrapper_composed_tail_is_bit_identical(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    _require_metal(mx)
    jm = pytest.importorskip("jang_tools.dsv4.mlx_model")
    from moespresso.runtime.deepseek_v4.model import (
        _Ratio4DecodeFusedAttention,
    )

    monkeypatch.setattr(decode_attention_kernel, "_ENABLED", True)
    args, attn = _build_synthetic_attention(mx, jm)
    wrapper = _Ratio4DecodeFusedAttention(
        attn, mx=mx, dsv4_model=jm, cache_cls=jm.DeepseekV4Cache)
    x = (mx.random.normal((1, 1, args.hidden_size)) * 0.05).astype(mx.float16)
    mx.eval(x)

    # R <= index_topk: the fused path hands off to the composed tail, which
    # must reproduce the composed op sequence exactly.
    cache_ref = _make_steady_cache(mx, jm, rows=32)
    out_ref = attn(x, mask=None, cache=cache_ref)
    cache_tail = _make_steady_cache(mx, jm, rows=32)
    out_tail = wrapper(x, mask=None, cache=cache_tail)
    mx.eval(out_ref, out_tail)

    assert wrapper.fused_decode_calls == 0
    assert wrapper.fused_decode_composed_tail_calls == 1
    np.testing.assert_array_equal(
        np.asarray(out_ref).view(np.uint16),
        np.asarray(out_tail).view(np.uint16))


def test_fused_decode_wrapper_delegates_outside_contract(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    jm = pytest.importorskip("jang_tools.dsv4.mlx_model")
    from moespresso.runtime.deepseek_v4.model import (
        _Ratio4DecodeFusedAttention,
    )

    calls = []

    class FakeRope:
        dims = 64

    class FakeIndexer:
        head_dim = 128
        n_heads = 64
        index_topk = 512

    class FakeAttn:
        compress_ratio = 4
        head_dim = 512
        rope = FakeRope()
        indexer = FakeIndexer()

        def __call__(self, x, mask=None, cache=None):
            calls.append((x.shape, cache))
            return x

    wrapper = _Ratio4DecodeFusedAttention(
        FakeAttn(), mx=mx, dsv4_model=jm, cache_cls=jm.DeepseekV4Cache)
    x = mx.zeros((1, 1, 8), dtype=mx.float16)

    # Gate off: always delegates.
    monkeypatch.setattr(decode_attention_kernel, "_ENABLED", False)
    cache = jm.DeepseekV4Cache(128, compress_ratio=4)
    wrapper(x, mask=None, cache=cache)
    assert len(calls) == 1

    # Gate on but no FP8 cache contract marker: delegates before any
    # cache mutation.
    monkeypatch.setattr(decode_attention_kernel, "_ENABLED", True)
    wrapper(x, mask=None, cache=cache)
    assert len(calls) == 2

    # Prefill shape delegates.
    wrapper(mx.zeros((1, 4, 8), dtype=mx.float16), mask=None, cache=cache)
    assert len(calls) == 3
    assert wrapper.fused_decode_calls == 0


def test_patch_installs_fused_decode_wrapper_by_gate(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("jang_tools.dsv4.mlx_model")
    del mx
    from moespresso.runtime.deepseek_v4.model import (
        _patch_deepseek_v4_ratio4_decode_fused_attention,
        build_deepseek_v4_graph_from_manifest,
    )

    manifest = {
        "architecture": {
            "family": "deepseek_v4_flash",
            "config": {
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
                "index_n_heads": 2,
                "index_head_dim": 128,
                "index_topk": 2,
            },
            "compress_ratios": [4],
        }
    }

    monkeypatch.setattr(decode_attention_kernel, "_ENABLED", False)
    model = build_deepseek_v4_graph_from_manifest(manifest)
    assert _patch_deepseek_v4_ratio4_decode_fused_attention(model) == 0
    assert model._moespresso_dsv4_fused_decode_attention_layers == 0

    monkeypatch.setattr(decode_attention_kernel, "_ENABLED", True)
    monkeypatch.setattr(
        decode_attention_kernel, "_metal_available", lambda: True)
    model = build_deepseek_v4_graph_from_manifest(manifest)
    assert _patch_deepseek_v4_ratio4_decode_fused_attention(model) == 1
    attn = model.layers[0].self_attn
    assert attn._moespresso_dsv4_fused_decode_attention
    # Re-patching is a no-op.
    assert _patch_deepseek_v4_ratio4_decode_fused_attention(model) == 0
