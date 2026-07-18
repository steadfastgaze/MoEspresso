from __future__ import annotations

import json

import numpy as np
import pytest

from moespresso.runtime.deepseek_v4 import indexed_attention_kernel as indexed


def test_indexed_attention_probe_default_cases_cover_goal_scales():
    cases = indexed.default_probe_cases()

    assert [case.name for case in cases] == [
        "bounded_long_rows961",
        "q3_scale_rows7618",
    ]
    assert [case.compressed_rows for case in cases] == [961, 7618]
    assert all(case.topk_rows == 512 for case in cases)
    assert all(case.raw_rows == 128 for case in cases)


def test_indexed_attention_probe_cli_writes_json(monkeypatch, tmp_path):
    out = tmp_path / "probe.json"
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {
            "metric": "ds4_indexed_mixed_attention_probe",
            "cases": [{"case": "rows961"}],
        }

    monkeypatch.setattr(indexed, "run_indexed_attention_probe", fake_run)

    assert indexed.main([
        "--repeats",
        "3",
        "--warmup",
        "1",
        "--compressed-rows",
        "961",
        "--json-out",
        str(out),
    ]) == 0

    assert json.loads(out.read_text())["metric"] == "ds4_indexed_mixed_attention_probe"
    assert calls == [{
        "repeats": 3,
        "warmup": 1,
        "cases": (indexed.IndexedAttentionProbeCase("rows961", 961),),
        "seed": 0,
    }]


def test_indexed_attention_probe_cli_rejects_invalid_repeats():
    with pytest.raises(SystemExit):
        indexed.main(["--repeats", "0"])


def test_indexed_mixed_attention_probe_kernel_matches_mlx_reference():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    rng = np.random.default_rng(123)
    q = mx.array(rng.standard_normal((8, 512), dtype=np.float32))
    raw = mx.array(rng.standard_normal((9, 512), dtype=np.float32)).astype(mx.float16)
    comp = mx.array(rng.standard_normal((32, 512), dtype=np.float32)).astype(mx.float16)
    topk = mx.array(np.arange(16, dtype=np.int32))
    sinks = mx.array(rng.standard_normal((8,), dtype=np.float32))

    got = indexed.indexed_mixed_attention_decode(
        q,
        raw,
        comp,
        topk,
        sinks,
        pos0=9 + 32 * 4 - 1,
    )
    expected = indexed.mlx_selected_rows_attention_reference(q, raw, comp, topk, sinks)
    mx.eval(got, expected)

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_indexed_mixed_attention_prefill_kernel_matches_mlx_reference():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    rng = np.random.default_rng(456)
    q = mx.array(rng.standard_normal((3, 8, 512), dtype=np.float32))
    raw = mx.array(rng.standard_normal((3, 512), dtype=np.float32)).astype(mx.float16)
    comp = mx.array(rng.standard_normal((12, 512), dtype=np.float32)).astype(mx.float16)
    topk = mx.array(
        np.tile(np.arange(8, dtype=np.int32), (3, 1)),
        dtype=mx.int32,
    )
    sinks = mx.array(rng.standard_normal((8,), dtype=np.float32))

    got = indexed.indexed_mixed_attention_prefill(
        q,
        raw,
        comp,
        topk,
        sinks,
        pos0=0,
        window=128,
    )
    expected = indexed.mlx_indexed_mixed_attention_prefill_reference(
        q,
        raw,
        comp,
        topk,
        sinks,
        pos0=0,
        window=128,
    )
    mx.eval(got, expected)

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_indexed_mixed_attention_prefill_live_f16_matches_mlx_reference():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    rng = np.random.default_rng(789)
    base_q = mx.array(rng.standard_normal((1, 3, 8, 512), dtype=np.float32)).astype(
        mx.float16
    )
    q = base_q.transpose(0, 2, 1, 3)
    raw = mx.array(rng.standard_normal((1, 1, 3, 512), dtype=np.float32)).astype(
        mx.float16
    )
    comp = mx.array(rng.standard_normal((1, 12, 512), dtype=np.float32)).astype(
        mx.float16
    )
    topk = mx.array(
        np.tile(np.arange(8, dtype=np.int32), (1, 3, 1)),
        dtype=mx.int32,
    )
    sinks = mx.array(rng.standard_normal((8,), dtype=np.float32))

    got = indexed.indexed_mixed_attention_prefill_live_f16(
        q,
        raw,
        comp,
        topk,
        sinks,
        pos0=0,
        window=128,
    )
    expected_tld = indexed.mlx_indexed_mixed_attention_prefill_reference(
        q[0].transpose(1, 0, 2).astype(mx.float32),
        raw[0, 0],
        comp[0],
        topk[0],
        sinks,
        pos0=0,
        window=128,
    )
    expected = expected_tld[None].transpose(0, 2, 1, 3)
    mx.eval(got, expected)

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_indexed_mixed_attention_prefill_live_f32_matches_mlx_reference():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    rng = np.random.default_rng(790)
    base_q = mx.array(rng.standard_normal((1, 3, 8, 512), dtype=np.float32))
    q = base_q.transpose(0, 2, 1, 3)
    raw = mx.array(rng.standard_normal((1, 1, 3, 512), dtype=np.float32)).astype(
        mx.float16
    )
    comp = mx.array(rng.standard_normal((1, 12, 512), dtype=np.float32)).astype(
        mx.float16
    )
    topk = mx.array(
        np.tile(np.arange(8, dtype=np.int32), (1, 3, 1)),
        dtype=mx.int32,
    )
    sinks = mx.array(rng.standard_normal((8,), dtype=np.float32))

    got = indexed.indexed_mixed_attention_prefill_live_f32(
        q,
        raw,
        comp,
        topk,
        sinks,
        pos0=0,
        window=128,
    )
    expected_tld = indexed.mlx_indexed_mixed_attention_prefill_reference(
        q[0].transpose(1, 0, 2),
        raw[0, 0],
        comp[0],
        topk[0],
        sinks,
        pos0=0,
        window=128,
    )
    expected = expected_tld[None].transpose(0, 2, 1, 3)
    mx.eval(got, expected)

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_indexer_scores_tiled_live_matches_mlx_reference():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    rng = np.random.default_rng(321)
    q_base = mx.array(rng.standard_normal((1, 5, 8, 128), dtype=np.float32))
    q = q_base.transpose(0, 2, 1, 3)
    weights = mx.array(rng.standard_normal((1, 5, 8), dtype=np.float32)) * (
        8 ** -0.5
    ) * (128 ** -0.5)
    comp = mx.array(rng.standard_normal((1, 19, 128), dtype=np.float32))

    got = indexed.indexer_scores_tiled_live(q, weights, comp, pos0=0, ratio=4)
    raw = q @ comp[:, None].swapaxes(-1, -2)
    expected = mx.maximum(raw, 0) * weights.swapaxes(-1, -2)[..., None]
    expected = expected.sum(axis=1)
    q_pos = mx.arange(5)
    k_idx = mx.arange(19)
    visible = ((k_idx[None, :] + 1) * 4) <= (q_pos[:, None] + 1)
    expected = mx.where(visible[None], expected, -mx.inf)
    mx.eval(got, expected)

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        rtol=3.0e-2,
        atol=3.0e-2,
    )


def test_indexer_q_qat_live_matches_existing_contract():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

    rng = np.random.default_rng(654)
    q_base = mx.array(rng.standard_normal((1, 4, 8, 128), dtype=np.float32))
    q = q_base.transpose(0, 2, 1, 3)

    got = indexed.indexer_q_qat_live(q)
    expected = _dsv4_indexer_qat(mx, q)
    mx.eval(got, expected)

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def _live_prefill_case(mx, *, tokens, n_comp, heads=64, seed=99):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((1, heads, tokens, 512), dtype=np.float32))
    raw = mx.array(
        rng.standard_normal((1, 1, tokens, 512), dtype=np.float32)
    ).astype(mx.float16)
    comp = mx.array(
        rng.standard_normal((1, n_comp, 512), dtype=np.float32)
    ).astype(mx.float16)
    width = min(512, n_comp)
    ids = np.stack(
        [
            np.sort(rng.choice(n_comp, size=width, replace=False))
            for _ in range(tokens)
        ]
    )[None].astype(np.int32)
    topk = mx.array(ids)
    sinks = mx.array(rng.standard_normal((heads,), dtype=np.float32))
    return q, raw, comp, topk, sinks


@pytest.mark.parametrize("tokens,n_comp", [(200, 50), (137, 34), (96, 512)])
@pytest.mark.parametrize("q_dtype", ["float32", "float16"])
def test_indexed_mixed_attention_prefill_live_v2_bit_identical(
    monkeypatch, tokens, n_comp, q_dtype
):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    q, raw, comp, topk, sinks = _live_prefill_case(mx, tokens=tokens, n_comp=n_comp)
    if q_dtype == "float16":
        q = q.astype(mx.float16)
        fn = indexed.indexed_mixed_attention_prefill_live_f16
    else:
        fn = indexed.indexed_mixed_attention_prefill_live_f32

    # Pin the mma consumer off so both arms exercise the scalar kernels.
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "0")
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_V2", "0")
    v1 = fn(q, raw, comp, topk, sinks, pos0=0, window=128, ratio=4)
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_V2", "1")
    v2 = fn(q, raw, comp, topk, sinks, pos0=0, window=128, ratio=4)
    mx.eval(v1, v2)

    v1_bits = np.asarray(v1, dtype=np.float32).view(np.uint32)
    v2_bits = np.asarray(v2, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(v1_bits, v2_bits)


@pytest.mark.parametrize("tokens,n_comp", [(200, 50), (137, 34), (96, 512)])
@pytest.mark.parametrize("q_dtype", ["float32", "float16"])
def test_indexed_mixed_attention_prefill_live_mma_matches_v2(
    monkeypatch, tokens, n_comp, q_dtype
):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    q, raw, comp, topk, sinks = _live_prefill_case(mx, tokens=tokens, n_comp=n_comp)
    if q_dtype == "float16":
        q = q.astype(mx.float16)
        fn = indexed.indexed_mixed_attention_prefill_live_f16
    else:
        fn = indexed.indexed_mixed_attention_prefill_live_f32

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "0")
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_V2", "1")
    v2 = fn(q, raw, comp, topk, sinks, pos0=0, window=128, ratio=4)
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    mma = fn(q, raw, comp, topk, sinks, pos0=0, window=128, ratio=4)
    mx.eval(v2, mma)

    # The mma consumer is a valid f32 accumulation-order variant of v2
    # (identical row sets and operand precision, different summation
    # order), so it agrees to f32 rounding rather than bit-for-bit.
    np.testing.assert_allclose(
        np.asarray(mma),
        np.asarray(v2),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_indexed_mixed_attention_prefill_live_mma_matches_mlx_reference(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    rng = np.random.default_rng(791)
    tokens = 11
    heads = 16
    q = mx.array(rng.standard_normal((1, heads, tokens, 512), dtype=np.float32))
    raw = mx.array(
        rng.standard_normal((1, 1, tokens, 512), dtype=np.float32)
    ).astype(mx.float16)
    comp = mx.array(rng.standard_normal((1, 12, 512), dtype=np.float32)).astype(
        mx.float16
    )
    topk = mx.array(
        np.tile(np.arange(8, dtype=np.int32), (1, tokens, 1)),
        dtype=mx.int32,
    )
    sinks = mx.array(rng.standard_normal((heads,), dtype=np.float32))

    got = indexed.indexed_mixed_attention_prefill_live_f32(
        q,
        raw,
        comp,
        topk,
        sinks,
        pos0=0,
        window=128,
    )
    expected_tld = indexed.mlx_indexed_mixed_attention_prefill_reference(
        q[0].transpose(1, 0, 2),
        raw[0, 0],
        comp[0],
        topk[0],
        sinks,
        pos0=0,
        window=128,
    )
    expected = expected_tld[None].transpose(0, 2, 1, 3)
    mx.eval(got, expected)

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_indexed_mixed_attention_prefill_live_mma_negative_and_stop_ids(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    rng = np.random.default_rng(792)
    tokens = 64
    n_comp = 40
    q = mx.array(rng.standard_normal((1, 64, tokens, 512), dtype=np.float32))
    raw = mx.array(
        rng.standard_normal((1, 1, tokens, 512), dtype=np.float32)
    ).astype(mx.float16)
    comp = mx.array(
        rng.standard_normal((1, n_comp, 512), dtype=np.float32)
    ).astype(mx.float16)
    # Ascending ids with leading skips (negative) and a tail past every
    # token's visibility limit, exercising the continue-then-break order.
    ids = np.tile(np.arange(-3, 61, dtype=np.int32), (1, tokens, 1))
    topk = mx.array(ids)
    sinks = mx.array(rng.standard_normal((64,), dtype=np.float32))

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "0")
    v2 = indexed.indexed_mixed_attention_prefill_live_f32(
        q, raw, comp, topk, sinks, pos0=0, window=128, ratio=4
    )
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    mma = indexed.indexed_mixed_attention_prefill_live_f32(
        q, raw, comp, topk, sinks, pos0=0, window=128, ratio=4
    )
    mx.eval(v2, mma)

    np.testing.assert_allclose(
        np.asarray(mma),
        np.asarray(v2),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_indexed_mixed_attention_prefill_live_mma_circular_raw_start(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    rng = np.random.default_rng(793)
    tokens = 24
    raw_cap = 160
    n_comp = 96
    q = mx.array(rng.standard_normal((1, 64, tokens, 512), dtype=np.float32))
    raw = mx.array(
        rng.standard_normal((1, 1, raw_cap, 512), dtype=np.float32)
    ).astype(mx.float16)
    comp = mx.array(
        rng.standard_normal((1, n_comp, 512), dtype=np.float32)
    ).astype(mx.float16)
    ids = np.stack(
        [
            np.sort(rng.choice(n_comp, size=64, replace=False))
            for _ in range(tokens)
        ]
    )[None].astype(np.int32)
    topk = mx.array(ids)
    sinks = mx.array(rng.standard_normal((64,), dtype=np.float32))
    pos0 = 400

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "0")
    v2 = indexed.indexed_mixed_attention_prefill_live_f32(
        q, raw, comp, topk, sinks, pos0=pos0, window=128, ratio=4, raw_start=37
    )
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    mma = indexed.indexed_mixed_attention_prefill_live_f32(
        q, raw, comp, topk, sinks, pos0=pos0, window=128, ratio=4, raw_start=37
    )
    mx.eval(v2, mma)

    np.testing.assert_allclose(
        np.asarray(mma),
        np.asarray(v2),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_prefill_consumer_call_counts_track_variant(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    q, raw, comp, topk, sinks = _live_prefill_case(mx, tokens=16, n_comp=8)

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    before = indexed.prefill_consumer_call_counts()
    out = indexed.indexed_mixed_attention_prefill_live_f32(
        q, raw, comp, topk, sinks, pos0=0, window=128, ratio=4
    )
    mx.eval(out)
    after = indexed.prefill_consumer_call_counts()
    assert after["mma"] == before["mma"] + 1
    assert after["v2"] == before["v2"]

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "0")
    out = indexed.indexed_mixed_attention_prefill_live_f32(
        q, raw, comp, topk, sinks, pos0=0, window=128, ratio=4
    )
    mx.eval(out)
    final = indexed.prefill_consumer_call_counts()
    assert final["mma"] == after["mma"]
    assert final["v2"] == after["v2"] + 1


def test_prefill_consumer_mma_requires_heads16(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    q, raw, comp, topk, sinks = _live_prefill_case(
        mx, tokens=16, n_comp=8, heads=8
    )

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    before = indexed.prefill_consumer_call_counts()
    out = indexed.indexed_mixed_attention_prefill_live_f32(
        q, raw, comp, topk, sinks, pos0=0, window=128, ratio=4
    )
    mx.eval(out)
    after = indexed.prefill_consumer_call_counts()
    assert after["mma"] == before["mma"]
    assert after["v1"] == before["v1"] + 1


def _banded_live_case(mx, *, tokens, n_comp, heads=16, seed=811):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((1, heads, tokens, 512), dtype=np.float32))
    raw = mx.array(
        rng.standard_normal((1, 1, tokens, 512), dtype=np.float32)
    ).astype(mx.float16)
    comp = mx.array(
        rng.standard_normal((1, max(n_comp, 1), 512), dtype=np.float32)
    ).astype(mx.float16)
    topk = mx.array(
        np.tile(np.arange(n_comp, dtype=np.int32), (1, tokens, 1)),
        dtype=mx.int32,
    )
    sinks = mx.array(rng.standard_normal((heads,), dtype=np.float32))
    return q, raw, comp, topk, sinks


@pytest.mark.parametrize("q_dtype", ["float32", "float16"])
def test_banded_prefill_attention_live_matches_mlx_reference(
    monkeypatch, q_dtype
):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    tokens, n_comp, ratio, window = 200, 12, 16, 64
    q, raw, comp, topk, sinks = _banded_live_case(
        mx, tokens=tokens, n_comp=n_comp)
    if q_dtype == "float16":
        q = q.astype(mx.float16)

    got = indexed.banded_prefill_attention_live(
        q, raw, comp, topk, sinks, pos0=0, window=window, ratio=ratio
    )
    # The reference applies the same ascending-id visibility rule
    # ((row + 1) * ratio <= position + 1), so all-pool-rows ids reproduce
    # the compressed-pool visibility predicate of the banded plan.
    expected_tld = indexed.mlx_indexed_mixed_attention_prefill_reference(
        q[0].transpose(1, 0, 2),
        raw[0, 0],
        comp[0],
        topk[0],
        sinks,
        pos0=0,
        window=window,
        ratio=ratio,
    )
    expected = expected_tld[None].transpose(0, 2, 1, 3)
    mx.eval(got, expected)

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_banded_prefill_attention_live_zero_pool_dummy(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    tokens, window = 150, 32
    q, raw, comp, _topk, sinks = _banded_live_case(
        mx, tokens=tokens, n_comp=0)
    topk = mx.zeros((1, tokens, 0), dtype=mx.int32)

    # The one-row zero dummy comp buffer is never read (topk width 0).
    got = indexed.banded_prefill_attention_live(
        q, raw, mx.zeros((1, 1, 512), dtype=mx.float16), topk, sinks,
        pos0=0, window=window, ratio=1,
    )
    expected_tld = indexed.mlx_indexed_mixed_attention_prefill_reference(
        q[0].transpose(1, 0, 2),
        raw[0, 0],
        comp[0],
        topk[0],
        sinks,
        pos0=0,
        window=window,
        ratio=1,
    )
    expected = expected_tld[None].transpose(0, 2, 1, 3)
    mx.eval(got, expected)

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def _banded_offset_case(mx, *, n_tokens, n_comp, heads=64, seed=1234):
    """Design-of-record probe inputs: half operands at the served width."""
    rng = np.random.default_rng(seed)
    q = mx.array(
        rng.standard_normal((1, heads, n_tokens, 512)).astype(np.float16))
    kv = mx.array(
        rng.standard_normal((1, 1, n_tokens, 512)).astype(np.float16))
    if n_comp > 0:
        comp = mx.array(
            rng.standard_normal((1, n_comp, 512)).astype(np.float16))
    else:
        comp = mx.zeros((1, 1, 512), dtype=mx.float16)
    sinks = mx.array(rng.standard_normal((heads,)).astype(np.float32))
    return q, kv, comp, sinks


def _banded_topk_all(mx, n_tokens, n_comp):
    if n_comp <= 0:
        return mx.zeros((1, n_tokens, 0), dtype=mx.int32)
    ids = mx.arange(n_comp, dtype=mx.int32)
    return mx.broadcast_to(ids[None, None, :], (1, n_tokens, n_comp))


@pytest.mark.parametrize(
    "name, ratio, n_comp, offset, trailing_raw",
    [
        # r128 class: the rotating cache hands window - 1 lead-in rows
        # plus the chunk; aligned and unaligned offsets.
        ("r128_aligned_offset", 128, 4, 256, 127 + 256),
        ("r128_unaligned_offset", 128, 4, 250, 127 + 262),
        # swa class: zero pool, full-history raw rows (KVCache semantics);
        # the offset below the window exercises band clamping at zero.
        ("swa_full_history", 0, 0, 256, 512),
        ("swa_offset_below_window", 0, 0, 64, 512),
    ],
)
def test_banded_prefill_attention_live_offset_chunk_is_bit_identical(
    monkeypatch, name, ratio, n_comp, offset, trailing_raw
):
    """Chunk invariance at pos0: the offset arm reads the trailing raw rows
    a cache would return and must reproduce the single-call rows bit for
    bit, because the kernel keys band tile bases to absolute position and
    pool tiles to list index."""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    n_tokens, window = 512, 128
    q, kv, comp, sinks = _banded_offset_case(
        mx, n_tokens=n_tokens, n_comp=n_comp)
    kernel_ratio = ratio if n_comp > 0 else 1

    full = indexed.banded_prefill_attention_live(
        q, kv, comp, _banded_topk_all(mx, n_tokens, n_comp), sinks,
        pos0=0, window=window, ratio=kernel_ratio,
    )
    chunk_tokens = n_tokens - offset
    chunk = indexed.banded_prefill_attention_live(
        q[:, :, offset:, :],
        kv[:, :, n_tokens - trailing_raw:, :],
        comp,
        _banded_topk_all(mx, chunk_tokens, n_comp),
        sinks,
        pos0=offset, window=window, ratio=kernel_ratio,
    )
    mx.eval(full, chunk)

    a = np.asarray(full[:, :, offset:, :])
    b = np.asarray(chunk)
    assert a.dtype == np.float32 and b.dtype == np.float32
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


def test_banded_prefill_attention_live_offset_matches_reference(monkeypatch):
    """The offset arm also agrees with the per-token reference within the
    half staging tolerance on the r128 geometry."""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    n_tokens, window, ratio, n_comp, offset = 512, 128, 128, 4, 256
    check_tokens = 32
    q, kv, comp, sinks = _banded_offset_case(
        mx, n_tokens=n_tokens, n_comp=n_comp)

    q_chunk = q[:, :, offset:offset + check_tokens, :]
    raw_rows = (window - 1) + check_tokens
    raw_start = offset - (window - 1)
    raw_chunk = kv[:, :, raw_start:raw_start + raw_rows, :]
    out = indexed.banded_prefill_attention_live(
        q_chunk, raw_chunk, comp,
        _banded_topk_all(mx, check_tokens, n_comp), sinks,
        pos0=offset, window=window, ratio=ratio,
    )
    ref_topk = np.tile(
        np.arange(n_comp, dtype=np.int32), (check_tokens, 1))
    ref = indexed.mlx_indexed_mixed_attention_prefill_reference(
        q_chunk[0].transpose(1, 0, 2),
        raw_chunk[0, 0],
        comp[0],
        mx.array(ref_topk),
        sinks,
        pos0=offset,
        window=window,
        ratio=ratio,
    )
    mx.eval(out, ref)
    np.testing.assert_allclose(
        np.asarray(out[0].transpose(1, 0, 2)),
        np.asarray(ref),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_banded_prefill_attention_live_fails_closed(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    q, raw, comp, topk, sinks = _banded_live_case(mx, tokens=8, n_comp=4)

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "0")
    with pytest.raises(ValueError, match="mma consumer"):
        indexed.banded_prefill_attention_live(
            q, raw, comp, topk, sinks, pos0=0, window=32, ratio=16
        )

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    with pytest.raises(ValueError, match="mma consumer"):
        indexed.banded_prefill_attention_live(
            q[:, :8], raw, comp, topk, sinks, pos0=0, window=32, ratio=16
        )
    with pytest.raises(ValueError, match="float16 or float32"):
        indexed.banded_prefill_attention_live(
            q.astype(mx.bfloat16), raw, comp, topk, sinks,
            pos0=0, window=32, ratio=16,
        )
    with pytest.raises(ValueError, match="at least one row"):
        indexed.banded_prefill_attention_live(
            q, raw, comp[:, :0], topk, sinks, pos0=0, window=32, ratio=16
        )
    with pytest.raises(ValueError, match="positive window and ratio"):
        indexed.banded_prefill_attention_live(
            q, raw, comp, topk, sinks, pos0=0, window=32, ratio=0
        )
    # The consumer counts stay ratio-4-only: none of the calls above, nor a
    # successful banded call, moves them.
    before = indexed.prefill_consumer_call_counts()
    out = indexed.banded_prefill_attention_live(
        q, raw, comp, topk, sinks, pos0=0, window=32, ratio=16
    )
    mx.eval(out)
    after = indexed.prefill_consumer_call_counts()
    assert after == before


def test_indexer_q_qat_live_v2_bit_identical(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    rng = np.random.default_rng(11)
    parts = [
        rng.standard_normal((1, 64, 40, 128), dtype=np.float32),
        rng.standard_normal((1, 64, 8, 128), dtype=np.float32) * 2.0 ** 30,
        rng.standard_normal((1, 64, 8, 128), dtype=np.float32) * 2.0 ** -30,
        np.zeros((1, 64, 3, 128), dtype=np.float32),
    ]
    q = mx.array(np.concatenate(parts, axis=2))

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_QAT_V2", "0")
    v1 = indexed.indexer_q_qat_live(q)
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_QAT_V2", "1")
    v2 = indexed.indexer_q_qat_live(q)
    mx.eval(v1, v2)

    v1_bits = np.asarray(v1, dtype=np.float32).view(np.uint32)
    v2_bits = np.asarray(v2, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(v1_bits, v2_bits)


def test_indexed_mixed_attention_prefill_rejects_non_heads8():
    mx = pytest.importorskip("mlx.core")
    q = mx.zeros((1, 7, 512), dtype=mx.float32)
    raw = mx.zeros((1, 512), dtype=mx.float16)
    comp = mx.zeros((1, 512), dtype=mx.float16)
    topk = mx.zeros((1, 1), dtype=mx.int32)
    sinks = mx.zeros((7,), dtype=mx.float32)

    with pytest.raises(ValueError, match="divisible by 8"):
        indexed.indexed_mixed_attention_prefill(q, raw, comp, topk, sinks, pos0=0)


def _score_case(mx, *, tokens=37, n_comp=40, heads=8, seed=17):
    rng = np.random.default_rng(seed)
    q = mx.array(
        rng.standard_normal((1, heads, tokens, 128), dtype=np.float32))
    weights = mx.array(
        rng.standard_normal((1, tokens, heads), dtype=np.float32)
    ) * (heads ** -0.5) * (128 ** -0.5)
    comp = mx.array(rng.standard_normal((1, n_comp, 128), dtype=np.float32))
    return q, weights, comp


def test_indexer_scores_tiled_live_f16_operands_bit_identical():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    # Multiple token and comp tiles, including masked (-inf) regions.
    q, weights, comp = _score_case(mx)

    f32 = indexed.indexer_scores_tiled_live(q, weights, comp, pos0=0, ratio=4)
    f16 = indexed.indexer_scores_tiled_live(
        q.astype(mx.float16), weights, comp.astype(mx.float16),
        pos0=0, ratio=4,
    )
    mx.eval(f32, f16)

    np.testing.assert_array_equal(
        np.asarray(f32).view(np.uint32),
        np.asarray(f16).view(np.uint32),
    )


def test_indexer_scores_tiled_live_rejects_mixed_operand_dtypes():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    q, weights, comp = _score_case(mx, tokens=9, n_comp=5)

    with pytest.raises(ValueError, match="both q and index_comp as float16"):
        indexed.indexer_scores_tiled_live(
            q.astype(mx.float16), weights, comp, pos0=0, ratio=4)
    with pytest.raises(ValueError, match="both q and index_comp as float16"):
        indexed.indexer_scores_tiled_live(
            q, weights, comp.astype(mx.float16), pos0=0, ratio=4)
    with pytest.raises(ValueError, match="weights must be float32"):
        indexed.indexer_scores_tiled_live(
            q, weights.astype(mx.float16), comp, pos0=0, ratio=4)
    with pytest.raises(ValueError, match="both be float32 or both be float16"):
        indexed.indexer_scores_tiled_live(
            q.astype(mx.bfloat16), weights, comp.astype(mx.bfloat16),
            pos0=0, ratio=4)


def test_indexer_score_operands_are_float32():
    mx = pytest.importorskip("mlx.core")

    q = mx.zeros((1, 2, 3, 128), dtype=mx.float16)
    comp = mx.zeros((1, 5, 128), dtype=mx.float16)

    q_out, comp_out = indexed.indexer_score_operands(q, comp)
    assert q_out.dtype == mx.float32
    assert comp_out.dtype == mx.float32


def test_indexer_scores_call_counts_track_operand_form():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")

    q, weights, comp = _score_case(mx, tokens=9, n_comp=5)

    before = indexed.indexer_scores_call_counts()
    out = indexed.indexer_scores_tiled_live(q, weights, comp, pos0=0, ratio=4)
    mx.eval(out)
    after = indexed.indexer_scores_call_counts()
    assert after["f32"] == before["f32"] + 1
    assert after["f16"] == before["f16"]

    out = indexed.indexer_scores_tiled_live(
        q.astype(mx.float16), weights, comp.astype(mx.float16),
        pos0=0, ratio=4,
    )
    mx.eval(out)
    final = indexed.indexer_scores_call_counts()
    assert final["f32"] == after["f32"]
    assert final["f16"] == after["f16"] + 1
