"""Fixed-shape DS4 decode cache contract: identity against the stock layout.

Every test drives a stock (concat-growth) cache and a fixed-state cache
with identical inputs and asserts bitwise equality of the returned values
and the published legacy state keys. The fixed layout must be
undetectable from values; only the storage and the append mechanism may
differ.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from moespresso.runtime.deepseek_v4 import fixed_decode_state as fds


def _mx():
    return pytest.importorskip("mlx.core")


def _jm():
    return pytest.importorskip("jang_tools.dsv4.mlx_model")


def _make_caches(jm, monkeypatch, *, ratio, window=8, max_context=256):
    monkeypatch.setenv("MOESPRESSO_DSV4_DECODE_MAX_CONTEXT", str(max_context))
    monkeypatch.delenv("MOESPRESSO_DSV4_DECODE_FIXED_STATE", raising=False)
    legacy = jm.DeepseekV4Cache(window, compress_ratio=ratio)
    fixed = fds.install_fixed_decode_state(
        jm.DeepseekV4Cache(window, compress_ratio=ratio))
    assert getattr(fixed, "_moespresso_dsv4_fixed_decode_state", False)
    return legacy, fixed


def _rows(mx, rng, count, out_dim):
    return mx.array(rng.standard_normal((1, count, out_dim), dtype=np.float32))


def _drive_overlap(mx, cache, kv, gate, pos, *, ratio, head_dim, state_key):
    rows, gate_rows, base = cache.accumulate_overlap_windows(
        kv, gate, state_key, ratio, pos, head_dim)
    if int(rows.shape[1]):
        weights = mx.softmax(gate_rows.astype(mx.float32), axis=2, precise=True)
        new_pooled = (rows.astype(mx.float32) * weights).sum(axis=2)
    else:
        new_pooled = mx.zeros((int(kv.shape[0]), 0, head_dim), dtype=mx.float32)
    return cache.update_pool(new_pooled, state_key), base


def _drive_plain(mx, cache, kv, gate, pos, *, ratio, state_key):
    out_dim = int(kv.shape[-1])
    ready_kv, ready_gate, base = cache.accumulate_windows(
        kv, gate, state_key, ratio, pos)
    if int(ready_kv.shape[1]):
        windows = int(ready_kv.shape[1]) // ratio
        kv_r = ready_kv.reshape(1, windows, ratio, out_dim)
        gate_r = ready_gate.reshape(1, windows, ratio, out_dim)
        weights = mx.softmax(gate_r.astype(mx.float32), axis=2, precise=True)
        new_pooled = (kv_r.astype(mx.float32) * weights).sum(axis=2)
    else:
        new_pooled = mx.zeros((1, 0, out_dim), dtype=mx.float32)
    return cache.update_pool(new_pooled, state_key), base


def _assert_arrays_equal(mx, a, b, label):
    if a is None or b is None:
        assert a is None and b is None, f"{label}: one side is None"
        return
    assert tuple(a.shape) == tuple(b.shape), f"{label}: shape mismatch"
    assert a.dtype == b.dtype, f"{label}: dtype mismatch"
    assert bool(mx.array_equal(a, b)), f"{label}: values differ"


def _assert_state_equal(mx, legacy_state, fixed_state, label):
    for key in ("buffer_kv", "buffer_gate", "pooled"):
        _assert_arrays_equal(
            mx, legacy_state.get(key), fixed_state.get(key), f"{label}:{key}")


def test_overlap_decode_parity(monkeypatch):
    mx = _mx()
    jm = _jm()
    ratio, head_dim = 4, 8
    out_dim = 2 * head_dim
    legacy, fixed = _make_caches(jm, monkeypatch, ratio=ratio)
    rng = np.random.default_rng(11)

    chunk_kv = _rows(mx, rng, 10, out_dim)
    chunk_gate = _rows(mx, rng, 10, out_dim)
    for cache in (legacy, fixed):
        _drive_overlap(mx, cache, chunk_kv, chunk_gate, 0,
                       ratio=ratio, head_dim=head_dim,
                       state_key="compressor_state")
    for pos in range(10, 33):
        kv = _rows(mx, rng, 1, out_dim)
        gate = _rows(mx, rng, 1, out_dim)
        pool_l, base_l = _drive_overlap(
            mx, legacy, kv, gate, pos, ratio=ratio, head_dim=head_dim,
            state_key="compressor_state")
        pool_f, base_f = _drive_overlap(
            mx, fixed, kv, gate, pos, ratio=ratio, head_dim=head_dim,
            state_key="compressor_state")
        assert base_l == base_f
        _assert_arrays_equal(mx, pool_l, pool_f, f"pool@{pos}")
        _assert_state_equal(
            mx, legacy.compressor_state, fixed.compressor_state, f"state@{pos}")
        assert fds.engaged_branch(fixed, "compressor_state", pos) is not None
        assert fds.engaged_branch(legacy, "compressor_state", pos) is None


def test_plain_decode_parity(monkeypatch):
    mx = _mx()
    jm = _jm()
    ratio, out_dim = 8, 12
    legacy, fixed = _make_caches(jm, monkeypatch, ratio=ratio)
    rng = np.random.default_rng(13)

    chunk_kv = _rows(mx, rng, 21, out_dim)
    chunk_gate = _rows(mx, rng, 21, out_dim)
    for cache in (legacy, fixed):
        _drive_plain(mx, cache, chunk_kv, chunk_gate, 0,
                     ratio=ratio, state_key="compressor_state")
    for pos in range(21, 60):
        kv = _rows(mx, rng, 1, out_dim)
        gate = _rows(mx, rng, 1, out_dim)
        pool_l, base_l = _drive_plain(
            mx, legacy, kv, gate, pos, ratio=ratio, state_key="compressor_state")
        pool_f, base_f = _drive_plain(
            mx, fixed, kv, gate, pos, ratio=ratio, state_key="compressor_state")
        assert base_l == base_f
        _assert_arrays_equal(mx, pool_l, pool_f, f"pool@{pos}")
        _assert_state_equal(
            mx, legacy.compressor_state, fixed.compressor_state, f"state@{pos}")
        assert fds.engaged_branch(fixed, "compressor_state", pos) is not None


def test_trim_parity(monkeypatch):
    mx = _mx()
    jm = _jm()
    ratio, head_dim = 4, 8
    out_dim = 2 * head_dim
    legacy, fixed = _make_caches(jm, monkeypatch, ratio=ratio)
    rng = np.random.default_rng(17)

    chunk_kv = _rows(mx, rng, 12, out_dim)
    chunk_gate = _rows(mx, rng, 12, out_dim)
    for cache in (legacy, fixed):
        _drive_overlap(mx, cache, chunk_kv, chunk_gate, 0,
                       ratio=ratio, head_dim=head_dim,
                       state_key="compressor_state")
    for pos in range(12, 19):
        kv = _rows(mx, rng, 1, out_dim)
        gate = _rows(mx, rng, 1, out_dim)
        _drive_overlap(mx, legacy, kv, gate, pos, ratio=ratio,
                       head_dim=head_dim, state_key="compressor_state")
        _drive_overlap(mx, fixed, kv, gate, pos, ratio=ratio,
                       head_dim=head_dim, state_key="compressor_state")

    legacy.trim(5)
    fixed.trim(5)
    _assert_state_equal(
        mx, legacy.compressor_state, fixed.compressor_state, "post-trim")

    # Trim rewinds row counts on the same capacity buffer; it must not
    # reallocate. The branch survives and re-engages on the next step.
    branch = fixed._moespresso_dsv4_fixed_branches["compressor_state"]
    cap_before = branch.pool_cap
    for pos in range(14, 26):
        kv = _rows(mx, rng, 1, out_dim)
        gate = _rows(mx, rng, 1, out_dim)
        pool_l, _ = _drive_overlap(
            mx, legacy, kv, gate, pos, ratio=ratio, head_dim=head_dim,
            state_key="compressor_state")
        pool_f, _ = _drive_overlap(
            mx, fixed, kv, gate, pos, ratio=ratio, head_dim=head_dim,
            state_key="compressor_state")
        _assert_arrays_equal(mx, pool_l, pool_f, f"pool@{pos}")
        _assert_state_equal(
            mx, legacy.compressor_state, fixed.compressor_state, f"state@{pos}")
    assert fixed._moespresso_dsv4_fixed_branches["compressor_state"] is branch
    assert branch.pool_cap is cap_before


def test_fork_deepcopy_isolation(monkeypatch):
    mx = _mx()
    jm = _jm()
    ratio, head_dim = 4, 8
    out_dim = 2 * head_dim
    legacy, fixed = _make_caches(jm, monkeypatch, ratio=ratio)
    rng = np.random.default_rng(19)

    chunk_kv = _rows(mx, rng, 12, out_dim)
    chunk_gate = _rows(mx, rng, 12, out_dim)
    for cache in (legacy, fixed):
        _drive_overlap(mx, cache, chunk_kv, chunk_gate, 0,
                       ratio=ratio, head_dim=head_dim,
                       state_key="compressor_state")
    for pos in range(12, 17):
        kv = _rows(mx, rng, 1, out_dim)
        gate = _rows(mx, rng, 1, out_dim)
        _drive_overlap(mx, legacy, kv, gate, pos, ratio=ratio,
                       head_dim=head_dim, state_key="compressor_state")
        _drive_overlap(mx, fixed, kv, gate, pos, ratio=ratio,
                       head_dim=head_dim, state_key="compressor_state")

    fork = copy.deepcopy(fixed)
    snapshot = np.asarray(fixed.compressor_state["pooled"])

    step_kv = _rows(mx, rng, 1, out_dim)
    step_gate = _rows(mx, rng, 1, out_dim)
    pool_l, _ = _drive_overlap(
        mx, legacy, step_kv, step_gate, 17, ratio=ratio, head_dim=head_dim,
        state_key="compressor_state")
    pool_fork, _ = _drive_overlap(
        mx, fork, step_kv, step_gate, 17, ratio=ratio, head_dim=head_dim,
        state_key="compressor_state")
    _assert_arrays_equal(mx, pool_l, pool_fork, "fork pool")

    # Driving the fork must not perturb the source cache's state.
    after = np.asarray(fixed.compressor_state["pooled"])
    assert np.array_equal(snapshot, after)

    pool_src, _ = _drive_overlap(
        mx, fixed, step_kv, step_gate, 17, ratio=ratio, head_dim=head_dim,
        state_key="compressor_state")
    _assert_arrays_equal(mx, pool_l, pool_src, "source pool")


def test_capacity_exhaustion_fails_closed(monkeypatch):
    mx = _mx()
    jm = _jm()
    ratio, head_dim = 4, 8
    out_dim = 2 * head_dim
    # Capacity of eight pool rows; the run below needs ten.
    legacy, fixed = _make_caches(jm, monkeypatch, ratio=ratio, max_context=32)
    rng = np.random.default_rng(23)

    dropped = False
    for pos in range(1, 41):
        kv = _rows(mx, rng, 1, out_dim)
        gate = _rows(mx, rng, 1, out_dim)
        pool_l, base_l = _drive_overlap(
            mx, legacy, kv, gate, pos, ratio=ratio, head_dim=head_dim,
            state_key="compressor_state")
        pool_f, base_f = _drive_overlap(
            mx, fixed, kv, gate, pos, ratio=ratio, head_dim=head_dim,
            state_key="compressor_state")
        assert base_l == base_f
        _assert_arrays_equal(mx, pool_l, pool_f, f"pool@{pos}")
        _assert_state_equal(
            mx, legacy.compressor_state, fixed.compressor_state, f"state@{pos}")
        if fds.engaged_branch(fixed, "compressor_state", pos) is None:
            dropped = True
    assert dropped
    assert fixed._moespresso_dsv4_fixed_branches.get("compressor_state") is None
    assert int(fixed.compressor_state["pooled"].shape[1]) == 10


def test_adoption_refuses_oversized_pool(monkeypatch):
    mx = _mx()
    jm = _jm()
    ratio, head_dim = 4, 8
    out_dim = 2 * head_dim
    # Capacity of two pool rows against a prefill that built five.
    legacy, fixed = _make_caches(jm, monkeypatch, ratio=ratio, max_context=8)
    rng = np.random.default_rng(29)

    chunk_kv = _rows(mx, rng, 20, out_dim)
    chunk_gate = _rows(mx, rng, 20, out_dim)
    for cache in (legacy, fixed):
        _drive_overlap(mx, cache, chunk_kv, chunk_gate, 0,
                       ratio=ratio, head_dim=head_dim,
                       state_key="compressor_state")
    for pos in range(20, 26):
        kv = _rows(mx, rng, 1, out_dim)
        gate = _rows(mx, rng, 1, out_dim)
        pool_l, _ = _drive_overlap(
            mx, legacy, kv, gate, pos, ratio=ratio, head_dim=head_dim,
            state_key="compressor_state")
        pool_f, _ = _drive_overlap(
            mx, fixed, kv, gate, pos, ratio=ratio, head_dim=head_dim,
            state_key="compressor_state")
        _assert_arrays_equal(mx, pool_l, pool_f, f"pool@{pos}")
        assert fds.engaged_branch(fixed, "compressor_state", pos) is None


def test_kill_switch_leaves_stock_contract(monkeypatch):
    jm = _jm()
    monkeypatch.setenv("MOESPRESSO_DSV4_DECODE_FIXED_STATE", "0")
    cache = jm.DeepseekV4Cache(8, compress_ratio=4)
    out = fds.install_fixed_decode_state(cache)
    assert out is cache
    assert not getattr(cache, "_moespresso_dsv4_fixed_decode_state", False)
    for name in ("accumulate_windows", "accumulate_overlap_windows",
                 "update_pool", "trim"):
        assert name not in cache.__dict__


def test_external_state_replacement_readopts(monkeypatch):
    mx = _mx()
    jm = _jm()
    ratio, head_dim = 4, 8
    out_dim = 2 * head_dim
    legacy, fixed = _make_caches(jm, monkeypatch, ratio=ratio)
    rng = np.random.default_rng(31)

    for pos in range(1, 10):
        kv = _rows(mx, rng, 1, out_dim)
        gate = _rows(mx, rng, 1, out_dim)
        _drive_overlap(mx, legacy, kv, gate, pos, ratio=ratio,
                       head_dim=head_dim, state_key="compressor_state")
        _drive_overlap(mx, fixed, kv, gate, pos, ratio=ratio,
                       head_dim=head_dim, state_key="compressor_state")

    # Replace the published pool on both caches with equal-valued fresh
    # arrays, as a replay or seeding tool would.
    replacement = np.asarray(legacy.compressor_state["pooled"])
    legacy.compressor_state["pooled"] = mx.array(replacement)
    fixed.compressor_state["pooled"] = mx.array(replacement)

    for pos in range(10, 20):
        kv = _rows(mx, rng, 1, out_dim)
        gate = _rows(mx, rng, 1, out_dim)
        pool_l, _ = _drive_overlap(
            mx, legacy, kv, gate, pos, ratio=ratio, head_dim=head_dim,
            state_key="compressor_state")
        pool_f, _ = _drive_overlap(
            mx, fixed, kv, gate, pos, ratio=ratio, head_dim=head_dim,
            state_key="compressor_state")
        _assert_arrays_equal(mx, pool_l, pool_f, f"pool@{pos}")
    assert fds.engaged_branch(fixed, "compressor_state", 19) is not None


def test_padded_scores_match_valid_prefix():
    mx = _mx()
    rng = np.random.default_rng(37)
    n_rows, capacity, heads, dim, topk = 961, 1024, 64, 128, 512

    q = mx.array(rng.standard_normal((1, heads, 1, dim), dtype=np.float32))
    pool = mx.array(rng.standard_normal((1, n_rows, dim), dtype=np.float32))
    cap = mx.zeros((1, capacity, dim), dtype=mx.float32)
    cap[:, :n_rows] = pool
    weights = mx.array(rng.standard_normal((1, 1, heads), dtype=np.float32))
    scale = dim ** -0.5

    def score(rows):
        s = q @ rows[:, None].swapaxes(-1, -2)
        s = mx.maximum(s, 0) * scale
        return (s * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)

    valid = score(pool)
    params = fds.decode_step_params(mx, offset=3844, pool_rows=n_rows)
    padded = fds.pad_scores_to_capacity(mx, score(cap), params)
    mx.eval(valid, padded)
    # The valid prefix of the padded fixed-width score row is bitwise
    # identical to the concat-layout scores.
    assert bool(mx.array_equal(valid, padded[..., :n_rows]))
    assert bool(
        mx.array_equal(
            padded[..., n_rows:],
            mx.full((1, 1, capacity - n_rows), -mx.inf, dtype=mx.float32),
        )
    )
    # Stage-3 readiness evidence: selecting over the padded row returns
    # the same top-k order as the valid row when the boundary is tie-free.
    top_valid = mx.argpartition(-valid, kth=topk - 1, axis=-1)[..., :topk]
    top_padded = mx.argpartition(-padded, kth=topk - 1, axis=-1)[..., :topk]
    assert bool(mx.array_equal(top_valid, top_padded))


class _StubCompressor:
    """Deterministic stand-in for the jang compressor.

    Projects the hidden row with a fixed matrix and drives the cache
    exactly as the stock overlap compressor does: accumulate, combine on
    window completion, append through `update_pool`.
    """

    def __init__(self, mx, *, ratio, head_dim, hidden, seed):
        rng = np.random.default_rng(seed)
        self._mx = mx
        self.ratio = ratio
        self.head_dim = head_dim
        self.out_dim = 2 * head_dim
        self._w_kv = mx.array(
            rng.standard_normal((hidden, self.out_dim), dtype=np.float32))
        self._w_gate = mx.array(
            rng.standard_normal((hidden, self.out_dim), dtype=np.float32))

    def __call__(self, x, rope, cache, start_pos, state_key="compressor_state"):
        mx = self._mx
        kv = x.astype(mx.float32) @ self._w_kv
        gate = x.astype(mx.float32) @ self._w_gate
        rows, gate_rows, _base = cache.accumulate_overlap_windows(
            kv, gate, state_key, self.ratio, start_pos, self.head_dim)
        if int(rows.shape[1]):
            weights = mx.softmax(
                gate_rows.astype(mx.float32), axis=2, precise=True)
            new_pooled = (rows.astype(mx.float32) * weights).sum(axis=2)
        else:
            new_pooled = mx.zeros(
                (int(x.shape[0]), 0, self.head_dim), dtype=mx.float32)
        return cache.update_pool(new_pooled, state_key)


def test_fp8_wrapper_parity_and_engagement(monkeypatch):
    mx = _mx()
    jm = _jm()
    from moespresso.runtime.deepseek_v4.model import _AttentionCompressorFp8KV

    ratio, head_dim, hidden = 4, 512, 16
    legacy, fixed = _make_caches(jm, monkeypatch, ratio=ratio)
    compressor = _StubCompressor(
        mx, ratio=ratio, head_dim=head_dim, hidden=hidden, seed=41)
    wrap_legacy = _AttentionCompressorFp8KV(compressor)
    wrap_fixed = _AttentionCompressorFp8KV(compressor)
    rng = np.random.default_rng(43)

    chunk = mx.array(rng.standard_normal((1, 9, hidden), dtype=np.float32))
    wrap_legacy(chunk, None, legacy, 0)
    wrap_fixed(chunk, None, fixed, 0)
    assert wrap_fixed.fixed_state_decode_calls == 0

    for pos in range(9, 27):
        x = mx.array(rng.standard_normal((1, 1, hidden), dtype=np.float32))
        pooled_l = wrap_legacy(x, None, legacy, pos)
        pooled_f = wrap_fixed(x, None, fixed, pos)
        _assert_arrays_equal(mx, pooled_l, pooled_f, f"fp8 pool@{pos}")
        _assert_arrays_equal(
            mx,
            legacy.compressor_state.get("pooled_fp8"),
            fixed.compressor_state.get("pooled_fp8"),
            f"fp8 cache@{pos}",
        )
        assert (
            legacy.compressor_state.get("pooled_fp8_rows")
            == fixed.compressor_state.get("pooled_fp8_rows")
        )
    assert wrap_legacy.fixed_state_decode_calls == 0
    assert wrap_fixed.fixed_state_decode_calls == 18


class _StubIndexer:
    """Minimal indexer surface for `_IndexerDS4ScoreContract`."""

    def __init__(self, mx, *, hidden, q_lora, n_heads, head_dim, index_topk,
                 ratio, seed):
        rng = np.random.default_rng(seed)
        self._mx = mx
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.index_topk = index_topk
        self.scale = head_dim ** -0.5
        self._w_q = mx.array(
            rng.standard_normal((q_lora, n_heads * head_dim), dtype=np.float32))
        self._w_weights = mx.array(
            rng.standard_normal((hidden, n_heads), dtype=np.float32))
        self.compressor = _StubCompressor(
            mx, ratio=ratio, head_dim=head_dim, hidden=hidden, seed=seed + 1)

    def wq_b(self, q_residual):
        return q_residual.astype(self._mx.float32) @ self._w_q

    def weights_proj(self, x):
        return x.astype(self._mx.float32) @ self._w_weights


def test_indexer_score_contract_fixed_parity(monkeypatch):
    mx = _mx()
    jm = _jm()
    from moespresso.runtime.deepseek_v4.model import _IndexerDS4ScoreContract

    ratio, hidden, q_lora, n_heads, head_dim = 4, 16, 12, 2, 128
    legacy, fixed = _make_caches(jm, monkeypatch, ratio=ratio)
    indexer = _StubIndexer(
        mx, hidden=hidden, q_lora=q_lora, n_heads=n_heads, head_dim=head_dim,
        index_topk=2, ratio=ratio, seed=47)
    rope = jm.DeepseekV4RoPE(64, 10000.0, None, 4096)

    contract_legacy = _IndexerDS4ScoreContract(
        indexer, ratio, layer_index=0, mx=mx, dsv4_model=jm)
    contract_fixed = _IndexerDS4ScoreContract(
        indexer, ratio, layer_index=0, mx=mx, dsv4_model=jm)
    rng = np.random.default_rng(53)

    chunk = mx.array(rng.standard_normal((1, 13, hidden), dtype=np.float32))
    indexer.compressor(chunk, rope, legacy, 0, state_key="indexer_state")
    indexer.compressor(chunk, rope, fixed, 0, state_key="indexer_state")

    for pos in range(13, 31):
        x = mx.array(rng.standard_normal((1, 1, hidden), dtype=np.float32))
        q_residual = mx.array(
            rng.standard_normal((1, 1, q_lora), dtype=np.float32))
        topk_l = contract_legacy(x, q_residual, rope, rope, legacy, pos)
        topk_f = contract_fixed(x, q_residual, rope, rope, fixed, pos)
        _assert_arrays_equal(mx, topk_l, topk_f, f"topk@{pos}")
        _assert_arrays_equal(
            mx,
            legacy.indexer_state.get("pooled_qat"),
            fixed.indexer_state.get("pooled_qat"),
            f"qat cache@{pos}",
        )
    calls = contract_fixed._moespresso_dsv4_indexer_score_contract_fixed_state_calls
    assert calls == 18
    assert contract_legacy._moespresso_dsv4_indexer_score_contract_fixed_state_calls == 0


def test_make_cache_installs_fixed_state(monkeypatch):
    jm = _jm()
    from moespresso.runtime.deepseek_v4.model import (
        _patch_deepseek_v4_required_attention_cache,
    )

    monkeypatch.delenv("MOESPRESSO_DSV4_DECODE_FIXED_STATE", raising=False)

    class _Attn:
        def __init__(self, ratio):
            self.compress_ratio = ratio

    class _Layer:
        def __init__(self, ratio):
            self.self_attn = _Attn(ratio)

    class _Inner:
        def __init__(self):
            self.layers = [_Layer(0), _Layer(4), _Layer(128)]

    class _Model:
        def __init__(self):
            self.args = type("Args", (), {"sliding_window": 8})()
            self.model = _Inner()

    class _PlainKV:
        def update_and_fetch(self, keys, values):
            return keys, values

    model = _Model()
    _patch_deepseek_v4_required_attention_cache(
        model,
        kv_cache_cls=_PlainKV,
        deepseek_cache_cls=jm.DeepseekV4Cache,
    )
    caches = model.make_cache()
    assert not getattr(caches[0], "_moespresso_dsv4_fixed_decode_state", False)
    assert getattr(caches[1], "_moespresso_dsv4_fixed_decode_state", False)
    assert getattr(caches[2], "_moespresso_dsv4_fixed_decode_state", False)

    monkeypatch.setenv("MOESPRESSO_DSV4_DECODE_FIXED_STATE", "0")
    caches_off = model.make_cache()
    assert not getattr(caches_off[1], "_moespresso_dsv4_fixed_decode_state", False)
    assert "trim" in caches_off[1].__dict__  # stock wrapper chain intact
    assert "accumulate_windows" not in caches_off[1].__dict__
