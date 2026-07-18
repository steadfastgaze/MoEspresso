"""Flash D=256 prefill attention on the SSD-streaming build.

The streamed build wires the same `install_flash_prefill_attention` route the
resident build uses, on the identical full-attention `self_attn` modules (the
pooled MoE swap leaves them alone). These tests prove the installer wraps the
streamed model's full-attention layers, is idempotent, installs nothing under
the family kill switch, falls back bit-identically on a single-row decode call
(the flash wrapper counts only `fallback_decode`, zero `flash_calls`), and that
the streamed stats surface the flash engagement counters. They stay synthetic:
no real model is loaded. The engaged-kernel numeric bound and every eligibility
fallback branch are covered by test_qwen_flash_prefill_attention.py against the
shared wrapper.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

kq = pytest.importorskip("mlx_kquant")

if getattr(kq, "sdpa_fa_prefill_q8", None) is None:
    pytest.skip(
        "installed mlx_kquant build lacks sdpa_fa_prefill_q8",
        allow_module_level=True,
    )

from mlx_lm.models.cache import QuantizedKVCache  # noqa: E402
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


class _PooledMoE(nn.Module):
    """Stand-in for the streamed build's pooled MoE block.

    The streamed build replaces the routed SwitchGLU with a pooled block and
    leaves `self_attn` on the full-attention layers alone. The flash installer
    walks `self_attn`, so this block only needs to be a distinct object that the
    installer never touches.
    """

    def __init__(self):
        super().__init__()
        self.marker = "pooled"


def _streamed_model(*, n_full=2, n_linear=1):
    """A streamed-shaped model: full-attention layers carry `self_attn` and a
    pooled MoE block; linear (GatedDeltaNet) layers set `is_linear` and carry no
    `self_attn`. Uses the `language_model.model.layers` shape the streamed build
    exposes."""
    layers = []
    for i in range(n_full):
        layer = SimpleNamespace(
            is_linear=False,
            self_attn=_attention(seed=100 + i),
            mlp=_PooledMoE(),
        )
        layers.append(layer)
    for _ in range(n_linear):
        layers.append(SimpleNamespace(is_linear=True, mlp=_PooledMoE()))
    inner = SimpleNamespace(layers=layers)
    return SimpleNamespace(language_model=SimpleNamespace(model=inner))


def test_installer_wraps_streamed_full_attention_idempotently(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_PREFILL_FLASH_D256", True)
    model = _streamed_model(n_full=2, n_linear=1)
    assert install_flash_prefill_attention(model) == 2
    layers = model.language_model.model.layers
    assert isinstance(layers[0].self_attn, FlashPrefillD256Attention)
    assert isinstance(layers[1].self_attn, FlashPrefillD256Attention)
    # The pooled MoE block is left alone, and the linear layer carries no
    # self_attn to wrap.
    assert layers[0].mlp.marker == "pooled"
    assert not hasattr(layers[2], "self_attn")
    # Re-running wraps nothing new.
    assert install_flash_prefill_attention(model) == 0
    assert isinstance(layers[0].self_attn, FlashPrefillD256Attention)


def test_kill_switch_installs_nothing_on_streamed(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_PREFILL_FLASH_D256", False)
    monkeypatch.setattr(fa, "_QWEN_DECODE_Q8_TILE16", False)
    model = _streamed_model(n_full=2, n_linear=1)
    assert install_flash_prefill_attention(model) == 0
    layers = model.language_model.model.layers
    assert isinstance(layers[0].self_attn, Qwen3NextAttention)
    assert isinstance(layers[1].self_attn, Qwen3NextAttention)


def test_decode_falls_back_bit_identically_on_streamed(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_PREFILL_FLASH_D256", True)
    model = _streamed_model(n_full=1, n_linear=0)
    stock = _attention(seed=200)
    model.language_model.model.layers[0].self_attn = stock
    install_flash_prefill_attention(model)
    wrapped = model.language_model.model.layers[0].self_attn
    assert isinstance(wrapped, FlashPrefillD256Attention)

    # Prime a nonempty q8 cache on both the wrapped and a stock reference.
    ref = _attention(seed=200)
    cache_w = QuantizedKVCache(group_size=64, bits=8)
    cache_s = QuantizedKVCache(group_size=64, bits=8)
    x0 = _x(32, seed=20)
    wrapped(x0, mask="causal", cache=cache_w)
    ref(x0, mask="causal", cache=cache_s)

    # A single-row decode call falls back to the stock path bit-identically and
    # counts only fallback_decode (zero flash_calls).
    xd = _x(1, seed=21)
    out_w = wrapped(xd, mask=None, cache=cache_w)
    out_s = ref(xd, mask=None, cache=cache_s)
    mx.eval(out_w, out_s)
    assert wrapped.fallback_decode == 1
    assert wrapped.flash_calls == 0
    assert bool(mx.array_equal(out_w, out_s))


def test_streamed_stats_surface_flash_counters(monkeypatch):
    monkeypatch.setattr(fa, "_QWEN_PREFILL_FLASH_D256", True)
    model = _streamed_model(n_full=2, n_linear=1)
    install_flash_prefill_attention(model)
    # Drive one prefill chunk (empty cache, fail-closed to stock) plus a decode
    # step on each wrapped layer so the counters carry engagement evidence.
    for layer in model.language_model.model.layers[:2]:
        cache = QuantizedKVCache(group_size=64, bits=8)
        layer.self_attn(_x(32), mask="causal", cache=cache)
        layer.self_attn(_x(1), mask=None, cache=cache)
    stats = flash_prefill_attention_stats(model)
    assert stats["wrapped_layers"] == 2
    # First-chunk cache fallback (empty q8 past) plus the decode fallback.
    assert stats["fallback_cache"] == 2
    assert stats["fallback_decode"] == 2
    assert stats["flash_calls"] == 0
