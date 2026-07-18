"""D=256 flash prefill attention dispatch for the qwen full-attention layers.

The served product configuration holds a q8 KV cache (`QuantizedKVCache`, group
64, 8 bits) at the ten full-attention layers, and every prefill chunk after the
first attends it through the composed `quantized_scaled_dot_product_attention`,
which materializes the quadratic score tensor per chunk (fenced at 188 ms per
layer per 2048-query chunk at 37K depth, with a 5.1 GB per-layer peak). The
flash route dispatches `mlx_kquant.sdpa_fa_prefill_q8` instead: float32 queries
and accumulators, the past prefix dequantized from the quantized cache into
float32 threadgroup staging during load, the fresh self chunk read dense, an
online softmax, and no score tensor. Fenced at the same shape the flash route
runs at roughly composed time at a 0.78 GB peak (6.5x lower).

The route is math-affecting: the flash kernel accumulates in a different order
than the composed path. The attended values themselves match the composed path
(the self chunk goes through the same q8 round trip
`QuantizedKVCache.update_and_fetch` applies; attending it dense measured a
+0.084 combined engaged prefill NLL regression and is not served). At the
float32 staging the engaged prefill NLL improves against the composed path
(-0.029 combined on the 8K and 37K audits). Half-precision staging rounds the
attended values and lost the engaged prefill NLL bar by two orders of
magnitude (float16 +0.1019 combined against a 5e-4 bar), so the route stages
in float32 only. The quality ladder judges any staging change, and an on-rail
fork is expected.

Kill switch and default: the route is on by default (the memory-lever form
passed the full quality ladder); `MOESPRESSO_QWEN_PREFILL_FLASH_D256=0` is the
kill switch, which installs nothing and serves the stock composed path byte
for byte on its own recorded rail. `MOESPRESSO_QWEN_PREFILL_FLASH_D256_STAGE`
selects the query-tile width (f32, the default, is the wide BQ=64 tile;
f32w32 is the BQ=32 width for re-pricing, bit-identical to the default).
With the route on, eligibility is fail-closed per call: the wrapper falls back
to the stock path unless the call is prefill shaped (more than one query row),
the cache is a `QuantizedKVCache` at group 64 and 8 bits with a nonempty past,
the mask is the causal string, the geometry is the served head layout (16
query heads, 2 KV heads, head dim 256, batch 1), the queries are float32, and
the kernel is present in the installed mlx_kquant build. Every fallback branch
is counted, and the wrapper is installed only on the resident K-quant build.

The first prefill chunk attends a dense `KVCache` (the serving policy converts
the cache to quantized after the first chunk), so it stays on the stock fused
dense path by construction; the flash route serves chunks two onward, which
carry the depth-scaling cost.

An optional decode route dispatches long q8 prefixes to
`mlx_kquant.sdpa_decode_q8` with the 16-key SIMD staging tile. It keeps the
float32 query, dequantization, softmax, accumulation, and output contract, but
changes the split reduction order. The route is on by default after passing
the long-context NLL checks and the full Ornith quality gate. Setting
`MOESPRESSO_QWEN_DECODE_Q8_TILE16=0` disables it. It applies at cache depths of
at least 8,192 keys; shorter prefixes retain the stock composed path.

When the installed mlx-kquant exposes its exact dimension-parallel split
merge, the decode route requests it by default. The merge preserves every
float32 reduction order and is bit-identical to the shared merge while
distributing the 256 output dimensions across eight SIMD groups. Setting
`MOESPRESSO_QWEN_DECODE_Q8_DIMENSION_MERGE=0` retains the shared merge without
disabling the tile-16 decode route. An older mlx-kquant build fails closed to
the shared merge.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

# Family kill switch. Default on (the float32 memory-lever form, full ladder
# green). Setting MOESPRESSO_QWEN_PREFILL_FLASH_D256=0 installs nothing and
# restores the stock composed path exactly.
_QWEN_PREFILL_FLASH_D256 = (
    os.environ.get("MOESPRESSO_QWEN_PREFILL_FLASH_D256", "1") == "1"
)

# Long-context q8 decode route. The 16-key SIMD stage has a depth-scaled gain;
# below 8K the served graph sits at its submission/overlap floor. Default on
# after the accumulation-order variant passed the long-context NLL checks and
# the full Ornith quality gate. Setting the environment variable to 0 restores
# the stock composed path.
_QWEN_DECODE_Q8_TILE16 = (
    os.environ.get("MOESPRESSO_QWEN_DECODE_Q8_TILE16", "1") == "1"
)
_QWEN_DECODE_Q8_DIMENSION_MERGE = (
    os.environ.get("MOESPRESSO_QWEN_DECODE_Q8_DIMENSION_MERGE", "1") == "1"
)
_DECODE_Q8_TILE16_MIN_KEYS = 8192

# Staging form. The route stages in float32, the memory-lever form: it carries
# no staging round, so with the composed-equivalent self values the engaged
# prefill NLL improves against the composed path (measured -0.029 combined on
# the 8K and 37K audits) at roughly composed time and a 6.5x lower per-layer
# attention peak. Half-precision staging (a 1.21x speed form over the composed
# path at 37K) rounds the attended values and lost the engaged prefill NLL bar
# by two orders of magnitude (float16 +0.1019 combined against the 5e-4 bar),
# so no half-precision stage is selectable.
#
# The default runs the wide query tile (BQ=64). The threadgroup staging memory
# depends on BK and D alone (20 KB at BK=16, D=256), so BK stays 16 and BQ is
# the only fold; BQ=64 folds twice the query positions onto each staged KV
# tile, halving the per-row staging and dequant work. The staging precision is
# identical to the BQ=32 width, so the served token stream is unchanged: the
# 4K anchor rail and the 37K greedy stream are bit-identical to the BQ=32
# width (full chunks pin at 16 splits at depth; only a short tail could shift
# the split count, which the served chunk layout does not hit). It recovers
# about +37 t/s at 37K prefill (789 to 826 resident, 789 to 821 streamed
# full-cap) at an unchanged peak. "f32w32" selects the BQ=32 width for
# re-pricing.
_STAGE_CONFIGS = {
    # stage id, query rows per threadgroup, keys per tile
    "f32": (2, 64, 16),
    "f32w32": (2, 32, 16),
}
_STAGE_NAME = os.environ.get("MOESPRESSO_QWEN_PREFILL_FLASH_D256_STAGE", "f32")
if _STAGE_NAME not in _STAGE_CONFIGS:
    _STAGE_NAME = "f32"  # unknown value fails closed to the memory-lever form
_FLASH_STAGE, _FLASH_BQ, _FLASH_BK = _STAGE_CONFIGS[_STAGE_NAME]

# The served full-attention geometry the kernel is fenced and certified for.
_FLASH_N_Q_HEADS = 16
_FLASH_N_KV_HEADS = 2
_FLASH_HEAD_DIM = 256


def flash_prefill_enabled() -> bool:
    """Whether the flash prefill route is enabled for this process."""
    return _QWEN_PREFILL_FLASH_D256


def _kernel_available() -> bool:
    try:
        import mlx_kquant as kq
    except ImportError:
        return False
    return getattr(kq, "sdpa_fa_prefill_q8", None) is not None


def _decode_kernel_available() -> bool:
    try:
        import mlx_kquant as kq
    except ImportError:
        return False
    return getattr(kq, "sdpa_decode_q8", None) is not None


def _decode_dimension_merge_available() -> bool:
    try:
        import mlx_kquant as kq
    except ImportError:
        return False
    return bool(
        getattr(
            kq,
            "HAS_SDPA_DECODE_Q8_DIMENSION_PARALLEL_MERGE",
            False,
        )
    )


def _quantized_cache_class():
    try:
        from mlx_lm.models.cache import QuantizedKVCache
    except ImportError:
        return None
    return QuantizedKVCache


class FlashPrefillD256Attention(nn.Module):
    """Dispatch eligible prefill and decode calls to q8 attention kernels.

    Wraps a stock `Qwen3NextAttention` module. The forward reproduces the stock
    body exactly up to the attention dispatch (projections, norms, partial
    RoPE, the cache update, the output gate, and the output projection are the
    stock modules), so an ineligible call computes the stock result through the
    stock operations.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.decode_dimension_merge_enabled = (
            _QWEN_DECODE_Q8_DIMENSION_MERGE and _decode_dimension_merge_available()
        )

        # Engagement counters (read through flash_prefill_attention_stats).
        self.flash_calls = 0
        self.decode_calls = 0
        self.decode_dimension_merge_calls = 0
        self.fallback_prefill_disabled = 0
        self.fallback_no_cache = 0
        self.fallback_decode = 0
        self.fallback_cache = 0
        self.fallback_mask = 0
        self.fallback_geometry = 0
        self.fallback_dtype = 0
        self.fallback_kernel = 0

    # ---- eligibility ----------------------------------------------------

    def _flash_eligible(self, queries, cache, mask) -> bool:
        """Per-call fail-closed eligibility; counts the first failing branch."""
        if not _QWEN_PREFILL_FLASH_D256:
            self.fallback_prefill_disabled += 1
            return False
        if queries.shape[2] <= 1:
            self.fallback_decode += 1
            return False
        qcls = _quantized_cache_class()
        if (
            qcls is None
            or not isinstance(cache, qcls)
            or getattr(cache, "bits", None) != 8
            or getattr(cache, "group_size", None) != 64
            or int(getattr(cache, "offset", 0)) < 1
            or cache.keys is None
        ):
            self.fallback_cache += 1
            return False
        if not (isinstance(mask, str) and mask == "causal"):
            self.fallback_mask += 1
            return False
        if (
            queries.shape[0] != 1
            or queries.shape[1] != _FLASH_N_Q_HEADS
            or queries.shape[3] != _FLASH_HEAD_DIM
            or self.inner.num_key_value_heads != _FLASH_N_KV_HEADS
        ):
            self.fallback_geometry += 1
            return False
        if queries.dtype != mx.float32:
            self.fallback_dtype += 1
            return False
        if not _kernel_available():
            self.fallback_kernel += 1
            return False
        return True

    def _decode_eligible(self, queries, cache, mask) -> bool:
        """Whether one decode row can use the long-context q8 kernel."""
        if not _QWEN_DECODE_Q8_TILE16:
            self.fallback_decode += 1
            return False
        qcls = _quantized_cache_class()
        if (
            qcls is None
            or not isinstance(cache, qcls)
            or getattr(cache, "bits", None) != 8
            or getattr(cache, "group_size", None) != 64
            or int(getattr(cache, "offset", 0)) < 1
            or cache.keys is None
        ):
            self.fallback_cache += 1
            return False
        if int(cache.offset) + int(queries.shape[2]) < _DECODE_Q8_TILE16_MIN_KEYS:
            self.fallback_decode += 1
            return False
        if mask is not None:
            self.fallback_mask += 1
            return False
        if (
            queries.shape[0] != 1
            or queries.shape[1] != _FLASH_N_Q_HEADS
            or queries.shape[2] != 1
            or queries.shape[3] != _FLASH_HEAD_DIM
            or self.inner.num_key_value_heads != _FLASH_N_KV_HEADS
        ):
            self.fallback_geometry += 1
            return False
        if queries.dtype != mx.float32:
            self.fallback_dtype += 1
            return False
        if not _decode_kernel_available():
            self.fallback_kernel += 1
            return False
        return True

    # ---- forward ---------------------------------------------------------

    def __call__(self, x, mask=None, cache=None):
        inner = self.inner
        B, L, _ = x.shape

        q_proj_output = inner.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, inner.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(B, L, -1)

        keys, values = inner.k_proj(x), inner.v_proj(x)

        queries = inner.q_norm(queries).transpose(0, 2, 1, 3)
        keys = inner.k_norm(
            keys.reshape(B, L, inner.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, inner.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is None:
            self.fallback_no_cache += 1
            queries = inner.rope(queries)
            keys = inner.rope(keys)
            from mlx_lm.models.base import scaled_dot_product_attention

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=inner.scale, mask=mask
            )
        else:
            queries = inner.rope(queries, offset=cache.offset)
            keys = inner.rope(keys, offset=cache.offset)
            if queries.shape[2] == 1:
                if self._decode_eligible(queries, cache, mask):
                    q_keys, q_values = cache.update_and_fetch(keys, values)
                    import mlx_kquant as kq

                    self.decode_calls += 1
                    decode_options = {}
                    if self.decode_dimension_merge_enabled:
                        self.decode_dimension_merge_calls += 1
                        decode_options["dimension_parallel_merge"] = True
                    output = kq.sdpa_decode_q8(
                        queries,
                        *q_keys,
                        *q_values,
                        inner.scale,
                        group_size=64,
                        bits=8,
                        splits=128,
                        stage=2,
                        compute=1,
                        tile_c=16,
                        **decode_options,
                    )
                else:
                    keys, values = cache.update_and_fetch(keys, values)
                    from mlx_lm.models.base import scaled_dot_product_attention

                    output = scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        cache=cache,
                        scale=inner.scale,
                        mask=mask,
                    )
            elif self._flash_eligible(queries, cache, mask):
                # Read the past prefix views before the update appends the
                # fresh chunk; the update keeps the cache correct for later
                # chunks and decode, and its quantized return is not used.
                past_k, past_v = cache.state
                cache.update_and_fetch(keys, values)
                import mlx_kquant as kq

                # The self chunk attends through the q8 round trip, matching
                # the values the composed path reconstructs from the cache.
                # Attending the fresh chunk dense measured a large engaged
                # prefill NLL regression (+0.084 combined at float32 staging)
                # where the round-tripped form measured an improvement, so the
                # composed-equivalent values are the route's contract.
                gs = cache.group_size
                bits = cache.bits
                self_k = mx.dequantize(
                    *mx.quantize(keys, group_size=gs, bits=bits),
                    group_size=gs, bits=bits)
                self_v = mx.dequantize(
                    *mx.quantize(values, group_size=gs, bits=bits),
                    group_size=gs, bits=bits)

                self.flash_calls += 1
                output = kq.sdpa_fa_prefill_q8(
                    queries,
                    *past_k,
                    *past_v,
                    self_k,
                    self_v,
                    inner.scale,
                    bq=_FLASH_BQ,
                    bk=_FLASH_BK,
                    stage=_FLASH_STAGE,
                )
            else:
                keys, values = cache.update_and_fetch(keys, values)
                from mlx_lm.models.base import scaled_dot_product_attention

                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=inner.scale,
                    mask=mask,
                )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return inner.o_proj(output * mx.sigmoid(gate))


def _iter_full_attention_layers(model):
    """Yield every decoder layer that carries a full-attention block."""
    for path in (
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("layers",),
    ):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            layers = obj
            break
    else:
        return
    for layer in layers:
        if getattr(layer, "is_linear", True):
            continue
        if getattr(layer, "self_attn", None) is None:
            continue
        yield layer


def install_flash_prefill_attention(model) -> int:
    """Wrap full-attention layers for enabled q8 attention dispatches.

    A no-op returning 0 when both routes are disabled or their kernels are
    unavailable. Layers already wrapped are left alone (idempotent). Returns
    the number of layers wrapped.
    """
    prefill_ready = _QWEN_PREFILL_FLASH_D256 and _kernel_available()
    decode_ready = _QWEN_DECODE_Q8_TILE16 and _decode_kernel_available()
    if not (prefill_ready or decode_ready):
        return 0
    installed = 0
    for layer in _iter_full_attention_layers(model):
        if isinstance(layer.self_attn, FlashPrefillD256Attention):
            continue
        layer.self_attn = FlashPrefillD256Attention(layer.self_attn)
        installed += 1
    return installed


def flash_prefill_attention_stats(model) -> dict:
    """Aggregate the flash-route engagement counters across wrapped layers."""
    stats = {
        "wrapped_layers": 0,
        "flash_calls": 0,
        "decode_calls": 0,
        "decode_dimension_merge_calls": 0,
        "fallback_prefill_disabled": 0,
        "fallback_no_cache": 0,
        "fallback_decode": 0,
        "fallback_cache": 0,
        "fallback_mask": 0,
        "fallback_geometry": 0,
        "fallback_dtype": 0,
        "fallback_kernel": 0,
    }
    for layer in _iter_full_attention_layers(model):
        attn = layer.self_attn
        if not isinstance(attn, FlashPrefillD256Attention):
            continue
        stats["wrapped_layers"] += 1
        for key in list(stats.keys()):
            if key == "wrapped_layers":
                continue
            stats[key] += int(getattr(attn, key))
    return stats
