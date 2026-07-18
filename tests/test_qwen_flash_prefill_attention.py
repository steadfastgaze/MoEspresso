"""Flash D=256 prefill attention dispatch for the qwen full-attention layers.

The route ships default on; MOESPRESSO_QWEN_PREFILL_FLASH_D256=0 is the kill
switch. These
tests prove the kill switch installs nothing, the enabled route wraps exactly
the full-attention layers, an eligible prefill chunk engages the flash kernel
with a numeric bound against the stock composed path, every fallback branch
falls back to the stock path with its counter incremented, the decode fallback
is bit-identical to the stock module, and the stats aggregator sums the
counters. The served speed effect and the token rail are the campaign's job,
not this unit's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402

kq = pytest.importorskip("mlx_kquant")

if getattr(kq, "sdpa_fa_prefill_q8", None) is None:
    pytest.skip(
        "installed mlx_kquant build lacks sdpa_fa_prefill_q8",
        allow_module_level=True,
    )

from mlx_lm.models.cache import KVCache, QuantizedKVCache  # noqa: E402
from mlx_lm.models.qwen3_next import Qwen3NextAttention  # noqa: E402

from moespresso.runtime.qwen import full_attention as fa  # noqa: E402
from moespresso.runtime.qwen.full_attention import (  # noqa: E402
    FlashPrefillD256Attention,
    flash_prefill_attention_stats,
    install_flash_prefill_attention,
)

HIDDEN = 64


def _attention(n_q_heads=16, n_kv_heads=2, head_dim=256, seed=0):
    args = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=n_q_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        attention_bias=False,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
        rope_theta=10000000.0,
        rope_scaling=None,
        max_position_embeddings=262144,
    )
    mx.random.seed(seed)
    attn = Qwen3NextAttention(args)
    mx.eval(attn.parameters())
    return attn


def _x(L, seed=1):
    mx.random.seed(seed)
    x = 0.1 * mx.random.normal((1, L, HIDDEN)).astype(mx.float32)
    mx.eval(x)
    return x


def _model_with_layers(attn_list):
    layers = []
    for attn in attn_list:
        layers.append(SimpleNamespace(is_linear=False, self_attn=attn))
    layers.append(SimpleNamespace(is_linear=True))
    return SimpleNamespace(layers=layers)


def _rel(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    return float(mx.linalg.norm(af - bf) / (mx.linalg.norm(bf) + 1e-9))


def test_kill_switch_installs_nothing(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_PREFILL_FLASH_D256", False)
    monkeypatch.setattr(fa, "_QWEN_DECODE_Q8_TILE16", False)
    model = _model_with_layers([_attention()])
    assert install_flash_prefill_attention(model) == 0
    assert isinstance(model.layers[0].self_attn, Qwen3NextAttention)


def test_install_requires_kernel(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_PREFILL_FLASH_D256", True)
    monkeypatch.setattr(fa, "_QWEN_DECODE_Q8_TILE16", False)
    monkeypatch.setattr(fa, "_kernel_available", lambda: False)
    model = _model_with_layers([_attention()])
    assert install_flash_prefill_attention(model) == 0


def test_install_wraps_full_attention_layers_idempotently(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_PREFILL_FLASH_D256", True)
    monkeypatch.setattr(fa, "_QWEN_DECODE_Q8_TILE16", False)
    model = _model_with_layers([_attention(), _attention(seed=2)])
    assert install_flash_prefill_attention(model) == 2
    assert isinstance(model.layers[0].self_attn, FlashPrefillD256Attention)
    assert isinstance(model.layers[1].self_attn, FlashPrefillD256Attention)
    # The linear layer carries no self_attn and is untouched.
    assert not hasattr(model.layers[2], "self_attn")
    # Re-running wraps nothing new.
    assert install_flash_prefill_attention(model) == 0


def test_engagement_and_numeric_bound_vs_stock():
    stock = _attention(seed=3)
    wrapped = FlashPrefillD256Attention(stock)
    cache_w = QuantizedKVCache(group_size=64, bits=8)
    cache_s = QuantizedKVCache(group_size=64, bits=8)
    x0 = _x(64, seed=10)
    x1 = _x(48, seed=11)

    # Chunk 0: empty quantized cache, fail-closed to the stock path.
    out0_w = wrapped(x0, mask="causal", cache=cache_w)
    out0_s = stock(x0, mask="causal", cache=cache_s)
    mx.eval(out0_w, out0_s)
    assert wrapped.flash_calls == 0
    assert wrapped.fallback_cache == 1
    assert bool(mx.array_equal(out0_w, out0_s))

    # Chunk 1: nonempty q8 past, causal mask, served geometry: engages.
    out1_w = wrapped(x1, mask="causal", cache=cache_w)
    out1_s = stock(x1, mask="causal", cache=cache_s)
    mx.eval(out1_w, out1_s)
    assert wrapped.flash_calls == 1
    rel = _rel(out1_w, out1_s)
    # At the default float32 staging the flash arm attends the same values
    # the composed arm reconstructs (q8 round-tripped self chunk), so the
    # output difference is accumulation order only.
    assert rel < 1e-4, f"flash vs composed rel {rel:.3e}"
    # Both caches advanced identically (the wrapper still runs the update).
    assert cache_w.offset == cache_s.offset == 112


def test_engagement_on_converted_cache():
    # The served path converts the first chunk's dense KVCache to quantized
    # (to_quantized), which stores bfloat16 V scales from the bfloat16 values;
    # the route must engage on that cache form too.
    stock = _attention(seed=14)
    wrapped = FlashPrefillD256Attention(stock)
    dense = KVCache()
    x0 = _x(64, seed=30)
    out0 = wrapped(x0, mask="causal", cache=dense)
    mx.eval(out0)
    # The served dense cache holds bfloat16 values (the value projection is a
    # bfloat16 weight); the all-float32 toy module stores float32, so cast the
    # buffer to reproduce the served cache form before conversion.
    dense.values = dense.values.astype(mx.bfloat16)
    cache = dense.to_quantized(group_size=64, bits=8)
    assert cache.values[1].dtype == mx.bfloat16
    out1 = wrapped(_x(32, seed=31), mask="causal", cache=cache)
    mx.eval(out1)
    assert wrapped.flash_calls == 1
    assert bool(mx.all(mx.isfinite(out1)).item())


def test_decode_fallback_is_bit_identical_to_stock(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_DECODE_Q8_TILE16", False)
    stock = _attention(seed=4)
    wrapped = FlashPrefillD256Attention(stock)
    cache_w = QuantizedKVCache(group_size=64, bits=8)
    cache_s = QuantizedKVCache(group_size=64, bits=8)
    x0 = _x(32, seed=20)
    xd = _x(1, seed=21)
    wrapped(x0, mask="causal", cache=cache_w)
    stock(x0, mask="causal", cache=cache_s)
    out_w = wrapped(xd, mask=None, cache=cache_w)
    out_s = stock(xd, mask=None, cache=cache_s)
    mx.eval(out_w, out_s)
    assert wrapped.fallback_decode == 1
    assert bool(mx.array_equal(out_w, out_s))


def test_decode_tile16_engages_with_numeric_bound(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_DECODE_Q8_TILE16", True)
    monkeypatch.setattr(fa, "_DECODE_Q8_TILE16_MIN_KEYS", 1)
    stock = _attention(seed=42)
    wrapped = FlashPrefillD256Attention(stock)
    cache_w = QuantizedKVCache(group_size=64, bits=8)
    cache_s = QuantizedKVCache(group_size=64, bits=8)
    x0 = _x(32, seed=50)
    xd = _x(1, seed=51)
    wrapped(x0, mask="causal", cache=cache_w)
    stock(x0, mask="causal", cache=cache_s)
    out_w = wrapped(xd, mask=None, cache=cache_w)
    out_s = stock(xd, mask=None, cache=cache_s)
    mx.eval(out_w, out_s)
    assert wrapped.decode_calls == 1
    assert wrapped.fallback_decode == 0
    assert cache_w.offset == cache_s.offset == 33
    rel = _rel(out_w, out_s)
    assert rel < 1e-4, f"tile16 decode vs composed rel {rel:.3e}"


@pytest.mark.parametrize(
    ("enabled", "capability", "expected_option"),
    [(True, True, True), (False, True, None), (True, False, None)],
)
def test_decode_dimension_merge_is_capability_and_kill_switch_gated(
    monkeypatch, enabled, capability, expected_option
):
    monkeypatch.setattr(fa, "_QWEN_DECODE_Q8_TILE16", True)
    monkeypatch.setattr(fa, "_QWEN_DECODE_Q8_DIMENSION_MERGE", enabled)
    monkeypatch.setattr(fa, "_decode_dimension_merge_available", lambda: capability)
    monkeypatch.setattr(fa, "_DECODE_Q8_TILE16_MIN_KEYS", 1)

    original = kq.sdpa_decode_q8
    observed_options = []

    def capture_option(*args, **kwargs):
        observed_options.append(kwargs.pop("dimension_parallel_merge", None))
        return original(*args, **kwargs)

    monkeypatch.setattr(kq, "sdpa_decode_q8", capture_option)
    wrapped = FlashPrefillD256Attention(_attention(seed=142))
    cache = QuantizedKVCache(group_size=64, bits=8)
    wrapped(_x(32, seed=150), mask="causal", cache=cache)
    output = wrapped(_x(1, seed=151), mask=None, cache=cache)
    mx.eval(output)

    assert observed_options == [expected_option]
    assert wrapped.decode_calls == 1
    assert wrapped.decode_dimension_merge_calls == int(expected_option is True)


def test_decode_tile16_falls_back_below_depth(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_DECODE_Q8_TILE16", True)
    monkeypatch.setattr(fa, "_DECODE_Q8_TILE16_MIN_KEYS", 64)
    stock = _attention(seed=43)
    wrapped = FlashPrefillD256Attention(stock)
    cache_w = QuantizedKVCache(group_size=64, bits=8)
    cache_s = QuantizedKVCache(group_size=64, bits=8)
    x0 = _x(32, seed=52)
    xd = _x(1, seed=53)
    wrapped(x0, mask="causal", cache=cache_w)
    stock(x0, mask="causal", cache=cache_s)
    out_w = wrapped(xd, mask=None, cache=cache_w)
    out_s = stock(xd, mask=None, cache=cache_s)
    mx.eval(out_w, out_s)
    assert wrapped.decode_calls == 0
    assert wrapped.fallback_decode == 1
    assert bool(mx.array_equal(out_w, out_s))


def test_fallback_mask_branch():
    wrapped = FlashPrefillD256Attention(_attention(seed=5))
    cache = QuantizedKVCache(group_size=64, bits=8)
    wrapped(_x(32), mask="causal", cache=cache)  # chunk 0, fallback_cache
    out = wrapped(_x(16), mask=None, cache=cache)  # multi-row, no causal str
    mx.eval(out)
    assert wrapped.flash_calls == 0
    assert wrapped.fallback_mask == 1


def test_fallback_dense_cache_branch():
    wrapped = FlashPrefillD256Attention(_attention(seed=6))
    cache = KVCache()
    out0 = wrapped(_x(32), mask="causal", cache=cache)
    out1 = wrapped(_x(16), mask="causal", cache=cache)
    mx.eval(out0, out1)
    assert wrapped.flash_calls == 0
    assert wrapped.fallback_cache == 2


def test_fallback_geometry_branch():
    # An off-contract head layout (8 query heads) never engages.
    wrapped = FlashPrefillD256Attention(
        _attention(n_q_heads=8, n_kv_heads=2, seed=7)
    )
    cache = QuantizedKVCache(group_size=64, bits=8)
    wrapped(_x(32), mask="causal", cache=cache)
    out = wrapped(_x(16), mask="causal", cache=cache)
    mx.eval(out)
    assert wrapped.flash_calls == 0
    assert wrapped.fallback_geometry == 1


def test_fallback_kernel_branch(monkeypatch):
    wrapped = FlashPrefillD256Attention(_attention(seed=8))
    cache = QuantizedKVCache(group_size=64, bits=8)
    wrapped(_x(32), mask="causal", cache=cache)
    monkeypatch.setattr(fa, "_kernel_available", lambda: False)
    out = wrapped(_x(16), mask="causal", cache=cache)
    mx.eval(out)
    assert wrapped.flash_calls == 0
    assert wrapped.fallback_kernel == 1


def test_fallback_no_cache_branch():
    wrapped = FlashPrefillD256Attention(_attention(seed=9))
    out = wrapped(_x(8), mask="causal", cache=None)
    mx.eval(out)
    assert wrapped.fallback_no_cache == 1
    assert wrapped.flash_calls == 0


def test_stats_aggregation(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_PREFILL_FLASH_D256", True)
    monkeypatch.setattr(fa, "_QWEN_DECODE_Q8_TILE16", True)
    monkeypatch.setattr(fa, "_DECODE_Q8_TILE16_MIN_KEYS", 1)
    model = _model_with_layers([_attention(seed=12), _attention(seed=13)])
    install_flash_prefill_attention(model)
    for layer in model.layers[:2]:
        cache = QuantizedKVCache(group_size=64, bits=8)
        layer.self_attn(_x(32), mask="causal", cache=cache)
        layer.self_attn(_x(16), mask="causal", cache=cache)
        layer.self_attn(_x(1), mask=None, cache=cache)
    model.layers[0].self_attn.decode_dimension_merge_calls = 1
    model.layers[1].self_attn.decode_dimension_merge_calls = 2
    stats = flash_prefill_attention_stats(model)
    assert stats["wrapped_layers"] == 2
    assert stats["flash_calls"] == 2
    assert stats["decode_calls"] == 2
    assert stats["decode_dimension_merge_calls"] == 3
    assert stats["fallback_cache"] == 2


# ---- BQ=64 float32-staging width -------------------------------------------


def test_f32_default_is_bq64_and_f32w32_is_bq32():
    # The float32 default runs the wide BQ=64 query tile; f32w32 keeps the
    # BQ=32 width selectable for re-pricing. Both hold the same staging id and
    # key tile; only the query tile differs.
    assert fa._STAGE_CONFIGS["f32"] == (2, 64, 16)
    assert fa._STAGE_CONFIGS["f32w32"] == (2, 32, 16)
    assert fa._STAGE_CONFIGS["f32"][0] == fa._STAGE_CONFIGS["f32w32"][0]
    assert fa._STAGE_CONFIGS["f32"][2] == fa._STAGE_CONFIGS["f32w32"][2]


def test_bq64_width_engages_and_matches_stock(monkeypatch):
    # The wrapper reads the width from the module globals; select the BQ=64
    # float32 width and confirm it engages and stays within the same
    # accumulation-order bound against the composed path as the BQ=32 width.
    monkeypatch.setattr(fa, "_FLASH_STAGE", 2)
    monkeypatch.setattr(fa, "_FLASH_BQ", 64)
    monkeypatch.setattr(fa, "_FLASH_BK", 16)
    stock = _attention(seed=33)
    wrapped = FlashPrefillD256Attention(stock)
    cache_w = QuantizedKVCache(group_size=64, bits=8)
    cache_s = QuantizedKVCache(group_size=64, bits=8)
    wrapped(_x(64, seed=40), mask="causal", cache=cache_w)
    stock(_x(64, seed=40), mask="causal", cache=cache_s)
    out1_w = wrapped(_x(48, seed=41), mask="causal", cache=cache_w)
    out1_s = stock(_x(48, seed=41), mask="causal", cache=cache_s)
    mx.eval(out1_w, out1_s)
    assert wrapped.flash_calls == 1
    rel = _rel(out1_w, out1_s)
    assert rel < 1e-4, f"f32w64 flash vs composed rel {rel:.3e}"


def _prefill_q8(q, past_k, past_v, self_k, self_v, scale, bq, bk):
    gs, bits = 64, 8
    pk = mx.quantize(past_k, group_size=gs, bits=bits)
    pv = mx.quantize(past_v, group_size=gs, bits=bits)
    sk = mx.dequantize(
        *mx.quantize(self_k, group_size=gs, bits=bits), group_size=gs, bits=bits
    )
    sv = mx.dequantize(
        *mx.quantize(self_v, group_size=gs, bits=bits), group_size=gs, bits=bits
    )
    out = kq.sdpa_fa_prefill_q8(
        q, *pk, *pv, sk, sv, scale, bq=bq, bk=bk, stage=2
    )
    mx.eval(out)
    return out


def _prefill_inputs(past_len, q_len, seed):
    n_q, n_kv, D = 16, 2, 256
    mx.random.seed(seed)
    q = 0.1 * mx.random.normal((1, n_q, q_len, D)).astype(mx.float32)
    past_k = 0.1 * mx.random.normal((1, n_kv, past_len, D)).astype(mx.float32)
    past_v = 0.1 * mx.random.normal((1, n_kv, past_len, D)).astype(mx.float32)
    self_k = 0.1 * mx.random.normal((1, n_kv, q_len, D)).astype(mx.float32)
    self_v = 0.1 * mx.random.normal((1, n_kv, q_len, D)).astype(mx.float32)
    mx.eval(q, past_k, past_v, self_k, self_v)
    return q, past_k, past_v, self_k, self_v, 1.0 / (D ** 0.5)


def test_bq64_vs_bq32_bit_identical_on_full_chunk():
    # At depth both widths pin at 16 splits (by_depth dominates) and the
    # key-axis walk is BK-tile sequential with BK unchanged, so the per-row
    # accumulation order is identical: a full chunk is bit-identical.
    q, pk, pv, sk, sv, scale = _prefill_inputs(past_len=32768, q_len=256, seed=7)
    out32 = _prefill_q8(q, pk, pv, sk, sv, scale, bq=32, bk=16)
    out64 = _prefill_q8(q, pk, pv, sk, sv, scale, bq=64, bk=16)
    assert bool(mx.array_equal(out32, out64)), (
        "full-chunk BQ=64 not bit-identical to BQ=32"
    )


def test_bq64_vs_bq32_forks_only_on_short_tail():
    # A short tail near N=2200 shifts the occupancy split floor (BQ=64 gives 8
    # splits, BQ=32 gives 4), which changes the pass-2 merge count, so the
    # outputs fork. The fork is a scheduling reorder only: the values stay
    # numerically close because the staging precision is unchanged.
    q, pk, pv, sk, sv, scale = _prefill_inputs(past_len=2065, q_len=135, seed=8)
    out32 = _prefill_q8(q, pk, pv, sk, sv, scale, bq=32, bk=16)
    out64 = _prefill_q8(q, pk, pv, sk, sv, scale, bq=64, bk=16)
    assert not bool(mx.array_equal(out32, out64)), (
        "short-tail split counts should differ, so the outputs fork"
    )
    assert _rel(out64, out32) < 1e-5, "short-tail fork should be a tiny reorder"
