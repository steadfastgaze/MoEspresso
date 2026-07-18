"""Manifest-driven DeepSeek-V4 MLX graph and package loading."""

from __future__ import annotations

import gc
import json
import math
import os
import tempfile
from contextvars import ContextVar
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Mapping

from moespresso.runtime.deepseek_v4 import fixed_decode_state as _fixed_decode_state


class DeepseekV4GraphError(ValueError):
    """Invalid or unsupported DS4 graph manifest."""


class DeepseekV4RuntimeLoadError(RuntimeError):
    """The DS4 package cannot be loaded into the runtime graph."""


_DSV4_ATTENTION_STATS_CONTEXT: ContextVar[Any] = ContextVar(
    "moespresso_dsv4_attention_stats_context",
    default=None,
)


def _architecture_config(manifest: dict) -> dict:
    architecture = manifest.get("architecture") or {}
    if architecture.get("family") != "deepseek_v4_flash":
        raise DeepseekV4GraphError("manifest is not a DeepSeek V4 package")
    config = dict(architecture.get("config") or {})
    if not config:
        raise DeepseekV4GraphError("DeepSeek V4 manifest is missing architecture.config")
    ratios = config.get("compress_ratios") or architecture.get("compress_ratios")
    if ratios is None:
        raise DeepseekV4GraphError("DeepSeek V4 manifest must carry explicit compress_ratios")
    n_layers = int(config.get("num_hidden_layers", 0))
    if n_layers <= 0:
        raise DeepseekV4GraphError("DeepSeek V4 config must carry num_hidden_layers")
    if len(ratios) not in {n_layers, n_layers + 1}:
        raise DeepseekV4GraphError(
            "DeepSeek V4 compress_ratios length must match hidden layers or include MTP"
        )
    config["compress_ratios"] = list(ratios)
    return config


def build_deepseek_v4_graph_from_manifest(
    manifest: dict,
    *,
    model_cls: Callable[[Any], Any] | None = None,
    args_cls: Any | None = None,
) -> Any:
    """Instantiate the DS4 MLX model graph from a package manifest."""
    config = _architecture_config(manifest)
    if model_cls is None or args_cls is None:
        try:
            from jang_tools.dsv4.mlx_model import Model, ModelArgs
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DeepseekV4GraphError("jang_tools.dsv4.mlx_model is required") from exc
        model_cls = model_cls or Model
        args_cls = args_cls or ModelArgs
    args = args_cls.from_dict(config) if hasattr(args_cls, "from_dict") else args_cls(**config)
    return model_cls(args)


def load_deepseek_v4_package_graph(manifest: dict, package_dir: Path) -> Any:
    """Build only the DS4 graph.

    This helper deliberately does not bind package weights yet. The full DS4 runtime
    loader must map MJTQ affine/TQ/raw tensors into the graph before generation.
    """
    _ = Path(package_dir)
    return build_deepseek_v4_graph_from_manifest(manifest)


def _is_deepseek_v4_bundle_key(key: str) -> bool:
    return key.endswith(".tq_bundle") and ".ffn.experts" in key


def _manifest_requires_routed_bundles(manifest: dict) -> bool:
    return any(
        t.get("format") in {"tq", "mxfp4", "kquant"} and t.get("kind") == "expert"
        for t in manifest.get("tensors", [])
    )


def _install_deepseek_v4_pooled_bundles(
    model,
    package_dir: Path,
    index,
    *,
    seed: int,
    capacity_per_layer: int | None = None,
    capacity_overrides: Mapping[int, int] | None = None,
    eviction_policy: str = "lfu",
) -> int:
    """Install DS4 routed experts as SSD-backed pooled SwitchGLUs.

    This is the default DS4 routed-expert path. The older resident installer
    materializes TurboQuant modules and is TQ-only; pooled installation keeps the
    bundle contract codec-aware and routes source-mxfp4 experts to the fused
    decode kernel.
    """
    from moespresso.runtime.ssd_streaming_build import (
        _budget_payload,
        _deterministic_available_bytes,
        install_pooled_switchglus,
    )
    from moespresso.runtime.streaming_capacity import (
        choose_capacity,
        package_capacity_budget,
    )

    budget_payload = None
    if capacity_per_layer is None:
        args = getattr(model, "args", None)
        max_router_fanout = int(getattr(args, "num_experts_per_tok", 1) or 1)
        budget = package_capacity_budget(
            index=index,
            package_dir=package_dir,
            max_router_fanout=max_router_fanout,
            available_bytes=_deterministic_available_bytes(),
        )
        if index.num_experts < budget.min_capacity:
            capacity_per_layer = index.num_experts
        else:
            capacity_per_layer = choose_capacity(budget)
        budget_payload = _budget_payload(budget)
    capacity_per_layer = int(capacity_per_layer)
    # Cross-layer decode lookahead (opt-in, MOESPRESSO_SSD_LOOKAHEAD=<delta>).
    # The prediction export exists only on the native-gate decode path, and
    # at full residency there are no misses to hide while carving spares
    # would break the full-residency certificate, so both cases refuse with
    # a printed reason and zero engagement. Spares come out of the same
    # capacity budget (the honest A/B against the LFU slots they replace).
    lookahead_env = int(
        os.environ.get("MOESPRESSO_SSD_LOOKAHEAD", "0") or "0")
    spare_slots = 0
    if lookahead_env > 0:
        from moespresso.runtime.pooled_switchglu import (
            _RING_DECODE,
            _gate_module,
            _ring_visibility_ok,
        )
        gate_live = (_RING_DECODE and _ring_visibility_ok()
                     and _gate_module() is not None)
        if not gate_live:
            print("[ds4-streaming] lookahead requested but the native-gate "
                  "decode path is not live - DISABLED (no capacity carved)",
                  flush=True)
            lookahead_env = 0
        elif capacity_per_layer >= index.num_experts:
            print("[ds4-streaming] lookahead requested at full residency - "
                  "DISABLED (no misses to hide; spares would break the "
                  "full-residency certificate)", flush=True)
            lookahead_env = 0
    if lookahead_env > 0:
        spare_slots = min(16, max(0, capacity_per_layer - 24))
        capacity_per_layer -= spare_slots
        if spare_slots <= 0:
            lookahead_env = 0
    installed = install_pooled_switchglus(
        model,
        package_dir=package_dir,
        index=index,
        capacity_per_layer=capacity_per_layer,
        capacity_overrides=capacity_overrides,
        eviction_policy=eviction_policy,
        seed=seed,
        spare_slots=spare_slots,
        wrap_deepseek_v4_moe=True,
    )
    if lookahead_env > 0:
        from moespresso.runtime.ssd_streaming_build import install_lookahead

        wired = install_lookahead(model, lookahead_env)
        object.__setattr__(model, "_moespresso_ssd_lookahead",
                           {"delta": lookahead_env, "wired": wired,
                            "spare_slots": spare_slots})
    object.__setattr__(model, "_moespresso_ssd_streaming_capacity",
                       int(capacity_per_layer))
    object.__setattr__(
        model,
        "_moespresso_ssd_streaming_capacity_overrides",
        {
            int(layer): int(capacity)
            for layer, capacity in dict(capacity_overrides or {}).items()
        },
    )
    object.__setattr__(model, "_moespresso_ssd_streaming_eviction_policy",
                       eviction_policy)
    if budget_payload is not None:
        object.__setattr__(
            model,
            "_moespresso_ssd_streaming_capacity_budget",
            budget_payload,
        )
    model.eval()
    return installed


def _patch_deepseek_v4_hc_post_float32(model) -> int:
    """Keep DS4 mHC recombine activations in the reference fp32 contract."""
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover - DS4 runtime already checks this
        return 0

    def _hc_post_float32(self, x, residual, post, comb):
        del self
        # DS4's C/CUDA/Metal expand kernels read the split-combine matrix as
        # comb[dst + src * n_hc], i.e. the transpose of the row-major Sinkhorn
        # buffer emitted by the split step.
        y = post[..., None] * x[..., None, :].astype(mx.float32) + mx.matmul(
            mx.swapaxes(comb, -1, -2).astype(mx.float32),
            residual.astype(mx.float32),
        )
        return y

    layers = getattr(getattr(model, "model", None), "layers", ())
    patched = 0
    for layer in layers:
        if not hasattr(layer, "_hc_post"):
            continue
        object.__setattr__(
            layer, "_hc_post", MethodType(_hc_post_float32, layer))
        patched += 1
    object.__setattr__(
        model, "_moespresso_dsv4_hc_post_dtype",
        "float32" if patched else "unchanged",
    )
    object.__setattr__(model, "_moespresso_dsv4_hc_post_patched_layers", patched)
    return patched


def _patch_deepseek_v4_hc_fused(model) -> int:
    """Route DS4 mHC stages through the fused Metal kernels.

    Wraps each layer's ``_hc_pre`` and ``_hc_post`` with dispatchers that
    run the fused split/weighted-sum and recombine kernels on eligible
    float chunks and fall back to the composed path on any precondition
    miss. Both kernels are bit-identical to the composed ops they replace
    (see ``hc_kernel``); the fused post stage implements the served
    float32 recombine contract, so it only installs after
    ``_patch_deepseek_v4_hc_post_float32`` has marked the model.
    Multi-row prefill chunks engage under
    ``MOESPRESSO_DSV4_HC_PREFILL_FUSED`` and single-row decode steps
    under ``MOESPRESSO_DSV4_HC_DECODE_FUSED``; with both gates at ``0``
    the wrap does not install. Eligible single-row decode steps also
    absorb the pre stage's rsqrt chain and mixer scale into the split
    kernel unless ``MOESPRESSO_DSV4_HC_DECODE_TAIL=0``.
    """
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover - DS4 runtime already checks this
        return 0
    from moespresso.runtime.deepseek_v4 import hc_kernel

    if (
        not hc_kernel.hc_fused_enabled()
        or getattr(model, "_moespresso_dsv4_hc_post_dtype", None) != "float32"
    ):
        object.__setattr__(model, "_moespresso_dsv4_hc_fused_layers", 0)
        return 0

    layers = getattr(getattr(model, "model", None), "layers", ())
    patched = 0
    for layer in layers:
        if not hasattr(layer, "_hc_pre") or not hasattr(layer, "_hc_post"):
            continue
        if getattr(layer, "_moespresso_dsv4_hc_fused", False):
            continue
        original_pre = layer._hc_pre
        original_post = layer._hc_post

        def _hc_pre_fused(self, x, fn, scale, base, _original=original_pre):
            args = getattr(self, "args", None)
            if args is None or not hc_kernel.hc_split_weighted_sum_eligible(
                x, fn, scale, base,
                hc_mult=int(getattr(args, "hc_mult", 0) or 0),
                iters=int(getattr(args, "hc_sinkhorn_iters", 0) or 0),
            ):
                self._moespresso_dsv4_hc_fused_pre_fallback_calls += 1
                return _original(x, fn, scale, base)
            # The flatten and mix GEMM mirror the composed stage op for
            # op. On the single-row decode shape the tail kernel absorbs
            # the rsqrt chain and the mixer scale along with the split
            # and the weighted sum; elsewhere the rsqrt chain stays
            # composed and the kernel covers the split dispatch and the
            # broadcast multiply plus hc-axis sum.
            x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
            if hc_kernel.hc_split_weighted_sum_tail_eligible(
                x, fn, scale, base,
                hc_mult=int(getattr(args, "hc_mult", 0) or 0),
                iters=int(getattr(args, "hc_sinkhorn_iters", 0) or 0),
            ):
                mixes_raw = x_flat @ fn.T
                if mixes_raw.dtype == mx.float32:
                    y, post, comb = hc_kernel.hc_split_weighted_sum_tail(
                        mixes_raw, x_flat, scale, base,
                        iters=int(args.hc_sinkhorn_iters),
                        eps=float(args.hc_eps),
                        rms_eps=float(args.rms_norm_eps),
                    )
                    self._moespresso_dsv4_hc_fused_pre_calls += 1
                    self._moespresso_dsv4_hc_fused_pre_decode_calls += 1
                    self._moespresso_dsv4_hc_fused_pre_tail_decode_calls += 1
                    return y.astype(x.dtype), post, comb
            rsqrt = mx.rsqrt(
                mx.mean(x_flat.square(), axis=-1, keepdims=True)
                + args.rms_norm_eps
            )
            mixes = (x_flat @ fn.T) * rsqrt
            if mixes.dtype != mx.float32:
                self._moespresso_dsv4_hc_fused_pre_fallback_calls += 1
                return _original(x, fn, scale, base)
            y, post, comb = hc_kernel.hc_split_weighted_sum(
                mixes, x_flat, scale, base,
                iters=int(args.hc_sinkhorn_iters),
                eps=float(args.hc_eps),
            )
            self._moespresso_dsv4_hc_fused_pre_calls += 1
            if int(x.shape[0]) * int(x.shape[1]) == 1:
                self._moespresso_dsv4_hc_fused_pre_decode_calls += 1
            return y.astype(x.dtype), post, comb

        def _hc_post_fused(self, x, residual, post, comb, _original=original_post):
            if not hc_kernel.hc_post_recombine_eligible(x, residual, post, comb):
                self._moespresso_dsv4_hc_fused_post_fallback_calls += 1
                return _original(x, residual, post, comb)
            out = hc_kernel.hc_post_recombine(x, residual, post, comb)
            self._moespresso_dsv4_hc_fused_post_calls += 1
            if int(x.shape[0]) * int(x.shape[1]) == 1:
                self._moespresso_dsv4_hc_fused_post_decode_calls += 1
            return out

        object.__setattr__(layer, "_moespresso_dsv4_hc_fused_pre_calls", 0)
        object.__setattr__(layer, "_moespresso_dsv4_hc_fused_post_calls", 0)
        object.__setattr__(
            layer, "_moespresso_dsv4_hc_fused_pre_decode_calls", 0)
        object.__setattr__(
            layer, "_moespresso_dsv4_hc_fused_pre_tail_decode_calls", 0)
        object.__setattr__(
            layer, "_moespresso_dsv4_hc_fused_post_decode_calls", 0)
        object.__setattr__(
            layer, "_moespresso_dsv4_hc_fused_pre_fallback_calls", 0)
        object.__setattr__(
            layer, "_moespresso_dsv4_hc_fused_post_fallback_calls", 0)
        object.__setattr__(layer, "_hc_pre", MethodType(_hc_pre_fused, layer))
        object.__setattr__(layer, "_hc_post", MethodType(_hc_post_fused, layer))
        object.__setattr__(layer, "_moespresso_dsv4_hc_fused", True)
        patched += 1
    object.__setattr__(model, "_moespresso_dsv4_hc_fused_layers", patched)
    return patched


_CACHE_WRAPPER_ORIGINALS_ATTR = "_moespresso_dsv4_cache_wrapper_originals"


def _cache_wrapper_originals(cache) -> dict:
    """Original bound methods stashed for the shipped cache wrappers.

    Originals live in an instance dict so `copy.deepcopy` (the prefix cache
    fork) rebinds them to the copied cache through the deepcopy memo. A
    closure over a bound method keeps pointing at the source cache after a
    fork, so driving the copy mutates the source cache's offset and pools.
    """
    originals = getattr(cache, _CACHE_WRAPPER_ORIGINALS_ATTR, None)
    if not isinstance(originals, dict):
        originals = {}
        object.__setattr__(cache, _CACHE_WRAPPER_ORIGINALS_ATTR, originals)
    return originals


def _update_and_fetch_with_fp8_kv(self, keys, values):
    original_update = getattr(self, _CACHE_WRAPPER_ORIGINALS_ATTR)["update_and_fetch"]
    rounded_keys = _deepseek_v4_fp8_kv_roundtrip(keys)
    rounded_values = (
        rounded_keys
        if values is keys
        else _deepseek_v4_fp8_kv_roundtrip(values)
    )
    return original_update(rounded_keys, rounded_values)


def _cache_with_fp8_kv_roundtrip(cache):
    if getattr(cache, "_moespresso_dsv4_fp8_kv_cache", False):
        return cache
    _cache_wrapper_originals(cache)["update_and_fetch"] = cache.update_and_fetch
    object.__setattr__(
        cache,
        "update_and_fetch",
        MethodType(_update_and_fetch_with_fp8_kv, cache),
    )
    object.__setattr__(cache, "_moespresso_dsv4_fp8_kv_cache", True)
    return cache


def _trim_with_compressed_pool_aux_clear(self, n):
    result = getattr(self, _CACHE_WRAPPER_ORIGINALS_ATTR)["trim"](n)
    for attr, keys in (
        ("indexer_state", ("pooled_qat", "pooled_qat_rows")),
        ("compressor_state", ("pooled_fp8", "pooled_fp8_rows")),
    ):
        state = getattr(self, attr, None)
        if isinstance(state, dict):
            for key in keys:
                state.pop(key, None)
    return result


def _cache_with_compressed_pool_aux_trim_clear(cache):
    if getattr(cache, "_moespresso_dsv4_compressed_pool_aux_trim_clear", False):
        return cache
    original_trim = getattr(cache, "trim", None)
    if original_trim is None:
        return cache
    _cache_wrapper_originals(cache)["trim"] = original_trim
    object.__setattr__(
        cache,
        "trim",
        MethodType(_trim_with_compressed_pool_aux_clear, cache),
    )
    object.__setattr__(
        cache,
        "_moespresso_dsv4_compressed_pool_aux_trim_clear",
        True,
    )
    return cache


def _deepseek_v4_cache_store_nbytes(self) -> int:
    """Prompt-cache store sizing tolerant of non-array aux state.

    The stock `DeepseekV4Cache.nbytes` sums `.nbytes` over every non-None
    value in the compressor and indexer state dicts. Post-decode those
    dicts also carry the aux row counts (`pooled_fp8_rows`,
    `pooled_qat_rows`) as plain ints, so the stock sum raises
    AttributeError and `LRUPromptCache.insert_cache` fails the request
    before the entry lands, losing multi-turn prefix reuse. Values without
    an integer `nbytes` size as zero: the row counts mirror array shapes
    whose bytes are already counted through the arrays themselves, and the
    store byte budget tracks tensor storage.
    """
    local_size = getattr(getattr(self, "local", None), "nbytes", None)
    total = local_size if isinstance(local_size, int) else 0
    for state_key in ("compressor_state", "indexer_state"):
        state = getattr(self, state_key, None)
        if not isinstance(state, dict):
            continue
        for value in state.values():
            size = getattr(value, "nbytes", None)
            if isinstance(size, int):
                total += size
    return total


def _patch_deepseek_v4_cache_store_nbytes(cache_cls) -> bool:
    """Replace the composite cache class `nbytes` with aux-tolerant sizing.

    `nbytes` is a read-only property on the cache class, so the instance
    method overrides used by the other cache wrappers cannot shadow it;
    the replacement lands on the class. Classes that do not define an
    `nbytes` property are left alone.
    """
    if getattr(cache_cls, "_moespresso_dsv4_store_nbytes", False):
        return False
    if not isinstance(getattr(cache_cls, "nbytes", None), property):
        return False
    cache_cls.nbytes = property(_deepseek_v4_cache_store_nbytes)
    cache_cls._moespresso_dsv4_store_nbytes = True
    return True


def _patch_deepseek_v4_required_attention_cache(
    model,
    *,
    kv_cache_cls: Callable[..., Any] | None = None,
    deepseek_cache_cls: Callable[..., Any] | None = None,
) -> None:
    """Enforce DS4's HSA/CSA attention-cache contract.

    A compressed DS4 layer served with plain KVCache has broken attention: it
    drops the compressed pool rows that are part of the model architecture. This
    is not a user-selectable long-context mode.
    """
    if deepseek_cache_cls is None:
        from jang_tools.dsv4.mlx_model import DeepseekV4Cache as deepseek_cache_cls
    if kv_cache_cls is None:
        from mlx_lm.models.cache import KVCache as kv_cache_cls
    _patch_deepseek_v4_cache_store_nbytes(deepseek_cache_cls)

    def make_deepseek_v4_required_cache(self):
        caches = []
        sliding_window = getattr(self.args, "sliding_window", None)
        for layer in self.model.layers:
            compress_ratio = int(getattr(layer.self_attn, "compress_ratio", 0) or 0)
            if compress_ratio:
                cache = deepseek_cache_cls(
                    sliding_window,
                    compress_ratio=compress_ratio,
                )
            else:
                cache = kv_cache_cls()
            wrapped = _cache_with_compressed_pool_aux_trim_clear(
                _cache_with_fp8_kv_roundtrip(cache)
            )
            if compress_ratio:
                # Fixed-shape decode state (capacity buffers plus row
                # counts) for the compressed-layer pool state. No-op when
                # MOESPRESSO_DSV4_DECODE_FIXED_STATE=0.
                wrapped = _fixed_decode_state.install_fixed_decode_state(wrapped)
            caches.append(wrapped)
        return caches

    object.__setattr__(
        model, "make_cache", MethodType(make_deepseek_v4_required_cache, model))
    object.__setattr__(
        model,
        "_moespresso_dsv4_attention_cache_contract",
        "required_hsa_csa_for_compressed_layers",
    )


def _deepseek_v4_e4m3fn_values() -> tuple[float, ...]:
    values = []
    for code in range(127):
        exp = (code >> 3) & 0x0F
        mant = code & 0x07
        if exp == 0:
            values.append(float(mant) * 0.001953125)
        else:
            values.append((1.0 + float(mant) * 0.125) * float(2 ** (exp - 7)))
    return tuple(values)


_DEEPSEEK_V4_E4M3FN_VALUES = _deepseek_v4_e4m3fn_values()
_DEEPSEEK_V4_E4M3FN_MX = None


def _deepseek_v4_e4m3fn_mx():
    global _DEEPSEEK_V4_E4M3FN_MX
    if _DEEPSEEK_V4_E4M3FN_MX is None:
        import mlx.core as mx

        _DEEPSEEK_V4_E4M3FN_MX = mx.array(
            _DEEPSEEK_V4_E4M3FN_VALUES,
            dtype=mx.float32,
        )
    return _DEEPSEEK_V4_E4M3FN_MX


def _deepseek_v4_fp8_kv_roundtrip(x, *, head_dim: int = 512, rot_dim: int = 64):
    """Apply DS4's E4M3FN round trip to the non-RoPE compressed-KV prefix."""
    import mlx.core as mx

    if not x.shape or int(x.shape[-1]) != int(head_dim):
        return x
    if x.size == 0:
        return x
    n_nope = int(head_dim) - int(rot_dim)
    if n_nope <= 0:
        return x
    if n_nope % 64:
        raise DeepseekV4RuntimeLoadError(
            "DeepSeek V4 compressed KV non-RoPE width must be divisible by 64"
        )

    if (
        int(head_dim) == 512
        and int(rot_dim) == 64
        and x.dtype == mx.float32
        and os.environ.get("MOESPRESSO_DSV4_FP8_KV_KERNEL", "1") != "0"
    ):
        # Single-dispatch bit-exact transcription; the composed path below
        # materializes a [rows, 448, 127] float32 argmin diff per call.
        from moespresso.runtime.deepseek_v4 import decode_attention_kernel

        return decode_attention_kernel.fp8_kv_prefix_rows(x)

    original_dtype = x.dtype
    prefix = x[..., :n_nope].astype(mx.float32)
    tail = x[..., n_nope:]
    blocks = prefix.reshape((-1, n_nope // 64, 64))
    amax = mx.max(mx.abs(blocks), axis=-1, keepdims=True)
    amax = mx.maximum(amax, mx.array(1.0e-4, dtype=mx.float32))
    log2 = mx.log(amax / 448.0) / math.log(2.0)
    scale = mx.exp(mx.ceil(log2) * math.log(2.0))
    scaled = mx.clip(blocks / scale, -448.0, 448.0)

    table = _deepseek_v4_e4m3fn_mx()
    abs_scaled = mx.abs(scaled)
    diff = mx.abs(abs_scaled[..., None] - table)
    nearest = mx.argmin(diff, axis=-1)
    quantized = mx.take(table, nearest)
    sign = mx.where(
        scaled < 0,
        mx.array(-1.0, dtype=mx.float32),
        mx.where(scaled > 0, mx.array(1.0, dtype=mx.float32), mx.array(0.0, dtype=mx.float32)),
    )
    rounded = (sign * quantized * scale).reshape(prefix.shape)
    out = mx.concatenate([rounded, tail.astype(mx.float32)], axis=-1)
    return out.astype(original_dtype)


class _AttentionCompressorFp8KV:
    def __init__(self, original, *, indexer=None):
        self._original = original
        self._indexer = indexer
        self._moespresso_dsv4_fp8_kv_wrapper = True
        # Decode steps whose compressor state, pool append, and fp8 cache
        # all ran on the fixed-shape layout (fixed_decode_state).
        self.fixed_state_decode_calls = 0

    def __getattr__(self, name: str):
        return getattr(self._original, name)

    def _fp8_roundtrip_pooled(self, pooled, cache, state_key, fixed_branch=None):
        if cache is None or state_key != "compressor_state":
            return _deepseek_v4_fp8_kv_roundtrip(pooled)
        state = getattr(cache, "compressor_state", None)
        if not isinstance(state, dict):
            return _deepseek_v4_fp8_kv_roundtrip(pooled)
        if fixed_branch is not None:
            updated = _fixed_decode_state.aux_fixed_update(
                fixed_branch,
                state,
                name="pooled_fp8",
                rows_key="pooled_fp8_rows",
                pooled=pooled,
                transform=_deepseek_v4_fp8_kv_roundtrip,
            )
            if updated is not None:
                self.fixed_state_decode_calls += 1
                return updated[0]

        pooled_rows = int(pooled.shape[1]) if len(pooled.shape) > 1 else 0
        cached = state.get("pooled_fp8")
        cached_rows = int(state.get("pooled_fp8_rows", 0) or 0)
        if cached is not None and 0 < cached_rows <= pooled_rows:
            if cached_rows == pooled_rows:
                return cached
            import mlx.core as mx

            tail = _deepseek_v4_fp8_kv_roundtrip(pooled[:, cached_rows:, :])
            rounded = mx.concatenate([cached, tail], axis=1)
        else:
            rounded = _deepseek_v4_fp8_kv_roundtrip(pooled)

        state["pooled_fp8"] = rounded
        state["pooled_fp8_rows"] = pooled_rows
        return rounded

    def __call__(self, x, rope, cache, start_pos, state_key="compressor_state"):
        pooled = self._original(x, rope, cache, start_pos, state_key=state_key)
        indexer = self._indexer
        indexer_compressor = getattr(indexer, "compressor", None)
        index_topk = int(getattr(indexer, "index_topk", 0) or 0)
        if (
            cache is not None
            and state_key == "compressor_state"
            and indexer_compressor is not None
            and index_topk > 0
            and pooled.shape[1] > 0
            and pooled.shape[1] <= index_topk
        ):
            # DS4's indexer compressor state must advance even when selecting
            # sparse top-k rows is unnecessary because every compressed row is
            # still visible. JANG skipped the whole indexer call in this case,
            # dropping the first 512 ratio-4 rows from later long-context top-k.
            indexer_compressor(
                x,
                rope,
                cache,
                start_pos,
                state_key="indexer_state",
            )
        fixed_branch = None
        if cache is not None and state_key == "compressor_state":
            fixed_branch = _fixed_decode_state.engaged_branch(
                cache, "compressor_state", start_pos)
        return self._fp8_roundtrip_pooled(
            pooled, cache, state_key, fixed_branch=fixed_branch)


def _patch_deepseek_v4_attention_compressor_fp8_kv(model) -> int:
    """Match DS4's compressed attention KV round trip before attention consumes it."""
    patched = 0
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        compressor = getattr(attn, "compressor", None)
        if compressor is None:
            continue
        if getattr(compressor, "_moespresso_dsv4_fp8_kv_wrapper", False):
            continue
        object.__setattr__(
            attn,
            "compressor",
            _AttentionCompressorFp8KV(
                compressor,
                indexer=getattr(attn, "indexer", None),
            ),
        )
        patched += 1
    object.__setattr__(
        model,
        "_moespresso_dsv4_attention_compressor_fp8_kv_layers",
        patched,
    )
    return patched


def _dsv4_indexer_hadamard128(mx, x):
    if x.shape[-1] != 128:
        raise DeepseekV4RuntimeLoadError(
            f"DS4 indexer QAT expects 128-wide rows, got {x.shape[-1]}"
        )
    return mx.hadamard_transform(x.astype(mx.float32), scale=0.08838834764831845)


def _dsv4_indexer_fp4_act_roundtrip(mx, x):
    values = mx.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=mx.float32,
    )
    y = x.astype(mx.float32).reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = mx.max(mx.abs(y), axis=-1, keepdims=True)
    amax = mx.maximum(amax, mx.array(7.052966104933725e-38, dtype=mx.float32))
    log2_scale = mx.log(amax / 6.0) / math.log(2.0)
    scale = mx.exp(mx.ceil(log2_scale) * math.log(2.0))
    normalized = mx.clip(y / scale, -6.0, 6.0)
    nearest = mx.argmin(mx.abs(mx.abs(normalized)[..., None] - values), axis=-1)
    dequant = mx.take(values, nearest)
    dequant = mx.where(normalized < 0, -dequant, dequant)
    return (dequant * scale).reshape(*x.shape)


def _dsv4_indexer_qat(mx, x):
    return _dsv4_indexer_fp4_act_roundtrip(
        mx,
        _dsv4_indexer_hadamard128(mx, x),
    )


def _dsv4_prefill_pooled_qat(mx, x):
    """QAT pooled indexer rows on the prefill path.

    Routes to ``indexer_qat_rows``, the single-dispatch Metal transcription
    whose parity tests prove bit identity with ``_dsv4_indexer_qat``; the
    composed op chain costs a dozen dispatches and an argmin materialization
    per prefill chunk. ``MOESPRESSO_DSV4_R4_PREFILL_POOLED_QAT_KERNEL=0``
    falls back to the composed chain.
    """
    if os.environ.get("MOESPRESSO_DSV4_R4_PREFILL_POOLED_QAT_KERNEL", "1") != "0":
        from moespresso.runtime.deepseek_v4 import indexer_score_kernel

        if x.dtype in (mx.float32, mx.float16, mx.bfloat16) and x.size > 0:
            return indexer_score_kernel.indexer_qat_rows(x)
    return _dsv4_indexer_qat(mx, x)


_DSV4_DECODE_QAT_KERNEL_ENV = "MOESPRESSO_DSV4_INDEXER_DECODE_QAT_KERNEL"


def _dsv4_decode_indexer_qat(mx, x):
    """QAT decode indexer rows through the single-dispatch kernel.

    The decode analog of ``_dsv4_prefill_pooled_qat``: ``indexer_qat_rows``
    is bit-identical to ``_dsv4_indexer_qat`` per element, while the
    composed chain issues roughly a dozen elementwise dispatches and
    materializes an argmin lattice diff per call. On the serial decode
    spine that composed chain is the fattest line of the indexer segment
    (0.443 ms fenced against a 0.186 ms kernel segment at the served
    steady state), so the kernel routing is a measured whole-token win at
    identical bits. ``MOESPRESSO_DSV4_INDEXER_DECODE_QAT_KERNEL=0`` is the
    kill switch and restores the composed chain.

    Returns ``(rows, used_kernel)`` so the caller can count engagement.
    """
    if os.environ.get(_DSV4_DECODE_QAT_KERNEL_ENV, "1") != "0":
        from moespresso.runtime.deepseek_v4 import indexer_score_kernel

        if (
            indexer_score_kernel._metal_available()
            and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
            and x.size > 0
            and int(x.shape[-1]) == 128
        ):
            return indexer_score_kernel.indexer_qat_rows(x), True
    return _dsv4_indexer_qat(mx, x), False


def _maybe_dump_dsv4_indexer_selection(
    *,
    mx,
    scores,
    topk,
    layer_index: int,
    start_pos: int,
) -> None:
    prefix = os.environ.get("MOESPRESSO_DSV4_INDEXER_DUMP_PREFIX")
    if not prefix:
        return
    layer_env = os.environ.get("MOESPRESSO_DSV4_INDEXER_DUMP_LAYER")
    if layer_env and layer_env != "all" and int(layer_env) != int(layer_index):
        return
    pos_env = os.environ.get("MOESPRESSO_DSV4_INDEXER_DUMP_POS")
    if not pos_env:
        return
    pos = int(pos_env)
    local_row = pos - int(start_pos)
    if local_row < 0 or local_row >= int(topk.shape[1]):
        return

    import numpy as np

    mx.eval(scores, topk)
    score_row = np.asarray(scores[0, local_row].astype(mx.float32))
    topk_row = np.asarray(topk[0, local_row].astype(mx.int32))
    base = Path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    score_row.tofile(f"{prefix}_indexer_scores-{layer_index}_pos{pos}.bin")
    topk_row.tofile(f"{prefix}_indexer_topk-{layer_index}_pos{pos}.i32")


class _IndexerDS4ScoreContract:
    """Apply DS4 indexer QAT and visibility before top-k selection."""

    def __init__(
        self,
        original,
        compress_ratio: int,
        *,
        layer_index: int,
        mx,
        dsv4_model,
    ):
        self._original = original
        self._compress_ratio = int(compress_ratio)
        self._layer_index = int(layer_index)
        self._mx = mx
        self._dsv4_model = dsv4_model
        self._moespresso_dsv4_indexer_score_contract = True
        self._moespresso_dsv4_indexer_score_contract_calls = 0
        self._moespresso_dsv4_indexer_score_contract_tokens = 0
        self._moespresso_dsv4_indexer_score_contract_pooled_rows = 0
        self._moespresso_dsv4_indexer_score_contract_topk_rows = 0
        self._moespresso_dsv4_indexer_score_contract_score_elements = 0
        self._moespresso_dsv4_indexer_score_contract_qat_elements = 0
        self._moespresso_dsv4_indexer_score_contract_cached_pooled_rows = 0
        self._moespresso_dsv4_indexer_score_contract_new_qat_pooled_rows = 0
        self._moespresso_dsv4_indexer_score_contract_fused_score_calls = 0
        self._moespresso_dsv4_indexer_score_contract_decode_qat_kernel_calls = 0
        self._moespresso_dsv4_indexer_score_contract_fixed_state_calls = 0
        self._moespresso_dsv4_indexer_score_contract_score_tail_kernel_calls = 0

    def __getattr__(self, name: str):
        return getattr(self._original, name)

    def __call__(self, x, q_residual, rope, position_rope, cache, start_pos):
        mx = self._mx
        B, L, _ = x.shape
        pooled = self._original.compressor(
            x, rope, cache, start_pos, state_key="indexer_state")
        if pooled.shape[1] == 0:
            return None
        pooled_rows = int(pooled.shape[1])
        tokens = int(B) * int(L)
        heads = int(self._original.n_heads)
        head_dim = int(self._original.head_dim)
        topk_rows = min(int(self._original.index_topk), pooled_rows)
        self._moespresso_dsv4_indexer_score_contract_calls += 1
        self._moespresso_dsv4_indexer_score_contract_tokens += tokens
        self._moespresso_dsv4_indexer_score_contract_pooled_rows += (
            tokens * pooled_rows
        )
        self._moespresso_dsv4_indexer_score_contract_topk_rows += (
            tokens * topk_rows
        )
        self._moespresso_dsv4_indexer_score_contract_score_elements += (
            tokens * heads * pooled_rows
        )
        offset = start_pos
        q = self._original.wq_b(q_residual).reshape(
            B, L, self._original.n_heads, self._original.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = self._dsv4_model._apply_partial_rope(q, position_rope, offset)
        q_qat_elements = tokens * heads * head_dim
        pool_qat_elements = int(B) * pooled_rows * head_dim
        cached_pooled_rows = 0
        new_qat_pooled_rows = pooled_rows
        state = getattr(cache, "indexer_state", None)
        fixed_branch = None
        fixed_qat_cap = None
        if int(B) == 1 and int(L) == 1 and isinstance(state, dict):
            fixed_branch = _fixed_decode_state.engaged_branch(
                cache, "indexer_state", offset)
        if fixed_branch is not None:
            updated = _fixed_decode_state.aux_fixed_update(
                fixed_branch,
                state,
                name="pooled_qat",
                rows_key="pooled_qat_rows",
                pooled=pooled,
                transform=lambda rows: _dsv4_decode_indexer_qat(mx, rows)[0],
            )
            if updated is None:
                fixed_branch = None
        if fixed_branch is not None:
            pooled, prior_qat_rows = updated
            cached_pooled_rows = (
                prior_qat_rows if 0 < prior_qat_rows <= pooled_rows else 0
            )
            new_qat_pooled_rows = pooled_rows - cached_pooled_rows
            pool_qat_elements = int(B) * new_qat_pooled_rows * head_dim
            fixed_qat_cap = _fixed_decode_state.aux_capacity_buffer(
                fixed_branch, "pooled_qat")
        elif isinstance(state, dict):
            cached = state.get("pooled_qat")
            cached_rows = int(state.get("pooled_qat_rows", 0) or 0)
            if cached is not None and 0 < cached_rows <= pooled_rows:
                cached_pooled_rows = cached_rows
                if cached_rows == pooled_rows:
                    pooled = cached
                    new_qat_pooled_rows = 0
                    pool_qat_elements = 0
                else:
                    pooled_tail = _dsv4_indexer_qat(mx, pooled[:, cached_rows:, :])
                    pooled = mx.concatenate([cached, pooled_tail], axis=1)
                    new_qat_pooled_rows = pooled_rows - cached_rows
                    pool_qat_elements = int(B) * new_qat_pooled_rows * head_dim
                    state["pooled_qat"] = pooled
                    state["pooled_qat_rows"] = pooled_rows
            else:
                pooled = _dsv4_indexer_qat(mx, pooled)
                state["pooled_qat"] = pooled
                state["pooled_qat_rows"] = pooled_rows
        else:
            pooled = _dsv4_indexer_qat(mx, pooled)
        self._moespresso_dsv4_indexer_score_contract_qat_elements += (
            q_qat_elements + pool_qat_elements
        )
        self._moespresso_dsv4_indexer_score_contract_cached_pooled_rows += (
            tokens * cached_pooled_rows
        )
        self._moespresso_dsv4_indexer_score_contract_new_qat_pooled_rows += (
            tokens * new_qat_pooled_rows
        )
        weights_raw = self._original.weights_proj(x)
        # Lazy: the float32 cast and head-count scale only execute on paths
        # that consume `weights`; the fused score tail consumes the raw
        # projection row and applies both in-kernel.
        weights = weights_raw.astype(mx.float32) * (
            self._original.n_heads ** -0.5
        )
        fused_row = None
        fixed_scores = None
        if fixed_qat_cap is not None:
            from moespresso.runtime.deepseek_v4 import indexer_score_kernel

            # Fixed-shape decode contract: score the full capacity buffer
            # with the stock op sequence, carry the valid-row count as a
            # per-token params array, and pad the invalid tail with -inf.
            # Selection runs on the valid-prefix slice, whose values are
            # bit-identical to the concat-layout scores, so the top-k set
            # and its order match the stock path exactly.
            q, used_kernel = _dsv4_decode_indexer_qat(mx, q)
            if used_kernel:
                self._moespresso_dsv4_indexer_score_contract_decode_qat_kernel_calls += 1
            raw_scores = (
                q.astype(mx.float32)
                @ fixed_qat_cap[:, None].swapaxes(-1, -2).astype(mx.float32)
            )
            params = _fixed_decode_state.decode_step_params(
                mx, offset=offset, pool_rows=pooled_rows)
            if (
                not os.environ.get("MOESPRESSO_DSV4_INDEXER_DUMP_PREFIX")
                and indexer_score_kernel.score_tail_eligible(
                    raw_scores, weights_raw, params)
            ):
                # One wide dispatch replaces the composed tail between the
                # score matmul and the selection (relu, scales, head sum,
                # capacity pad, negation), bit-identical to the composed
                # chain, and the argpartition consumes the negated row
                # directly. The selection dump stays on the composed chain
                # (the dump env is an eligibility veto above).
                neg_scores = indexer_score_kernel.fused_score_tail(
                    raw_scores,
                    weights_raw,
                    params,
                    scale=float(self._original.scale),
                )
                self._moespresso_dsv4_indexer_score_contract_score_tail_kernel_calls += 1
                self._moespresso_dsv4_indexer_score_contract_fixed_state_calls += 1
                k = topk_rows
                return mx.argpartition(
                    neg_scores[..., :pooled_rows], kth=k - 1, axis=-1,
                )[..., :k]
            fixed_scores = mx.maximum(raw_scores, 0) * self._original.scale
            fixed_scores = (
                fixed_scores * weights.swapaxes(-1, -2)[..., None]
            ).sum(axis=1)
            fixed_scores = _fixed_decode_state.pad_scores_to_capacity(
                mx, fixed_scores, params)
            self._moespresso_dsv4_indexer_score_contract_fixed_state_calls += 1
        elif int(B) == 1 and int(L) == 1:
            # Decode shape: every pooled row is visible, so the pre-top-k
            # chain needs no mask. The fused kernel absorbs the query QAT
            # lattice (bit-exact with _dsv4_indexer_qat) and the score
            # chain into one dispatch.
            from moespresso.runtime.deepseek_v4 import indexer_score_kernel

            if indexer_score_kernel.fused_qat_scores_eligible(q, pooled, weights):
                fused_row = indexer_score_kernel.fused_qat_indexer_scores(
                    q,
                    pooled,
                    weights,
                    float(self._original.scale),
                )
        if fixed_scores is not None:
            scores = fixed_scores[..., :pooled_rows]
        elif fused_row is not None:
            self._moespresso_dsv4_indexer_score_contract_fused_score_calls += 1
            scores = fused_row.reshape(1, 1, pooled_rows)
        else:
            if int(B) == 1 and int(L) == 1:
                # Decode-shaped fallback (fixed state not engaged): the
                # query QAT keeps the single-dispatch kernel routing.
                q, used_kernel = _dsv4_decode_indexer_qat(mx, q)
                if used_kernel:
                    self._moespresso_dsv4_indexer_score_contract_decode_qat_kernel_calls += 1
            else:
                q = _dsv4_indexer_qat(mx, q)
            scores = (
                q.astype(mx.float32)
                @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
            )
            scores = mx.maximum(scores, 0) * self._original.scale
            scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        if int(L) != 1:
            visible = self._dsv4_model._dsv4_compressed_visibility(
                B,
                L,
                offset,
                pooled.shape[1],
                self._compress_ratio,
            )[:, 0, :, :]
            scores = mx.where(visible, scores, mx.array(-mx.inf, dtype=scores.dtype))
        k = topk_rows
        topk = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        _maybe_dump_dsv4_indexer_selection(
            mx=mx,
            scores=scores,
            topk=topk,
            layer_index=self._layer_index,
            start_pos=offset,
        )
        return topk


def _patch_deepseek_v4_indexer_score_contract(model) -> int:
    """Match DS4's indexer QAT and mask invisible rows before top-k."""
    try:
        import mlx.core as mx
        import jang_tools.dsv4.mlx_model as dsv4_model
    except ImportError:  # pragma: no cover - DS4 runtime already checks this
        return 0

    patched = 0
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer_index, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        indexer = getattr(attn, "indexer", None)
        if indexer is None:
            continue
        if getattr(indexer, "_moespresso_dsv4_indexer_score_contract", False):
            continue
        compress_ratio = int(getattr(attn, "compress_ratio", 0) or 0)
        if compress_ratio <= 0:
            continue
        object.__setattr__(
            attn,
            "indexer",
            _IndexerDS4ScoreContract(
                indexer,
                compress_ratio,
                layer_index=layer_index,
                mx=mx,
                dsv4_model=dsv4_model,
            ),
        )
        patched += 1
    object.__setattr__(
        model,
        "_moespresso_dsv4_indexer_pre_topk_visibility_layers",
        patched,
    )
    return patched


def _patch_deepseek_v4_indexer_pre_topk_visibility(model) -> int:
    """Backward-compatible name for the full DS4 indexer score contract patch."""
    return _patch_deepseek_v4_indexer_score_contract(model)


# Router gate trim engagement counts by form, exported through
# `ssd_streaming_stats` and the speed-stats count keys so served A/B arms
# can prove which composition ran. "precast" counts gate calls on the
# hoisted weight operand, "select_kernel" counts fused select-tail calls,
# "select_composed" counts trimmed calls that kept the composed select
# (prefill shapes and per-call eligibility failures), and "composed"
# counts full delegations to the stock gate.
_ROUTER_GATE_TRIM_CALL_COUNTS = {
    "precast": 0,
    "select_kernel": 0,
    "select_composed": 0,
    "composed": 0,
}


def router_gate_trim_call_counts() -> dict[str, int]:
    """Return router gate trim engagement counts by form."""
    return dict(_ROUTER_GATE_TRIM_CALL_COUNTS)


class _DeepseekV4RouterGateContract:
    """DS4 MoE router gate with the decode router trims.

    Two bit-identical trims over the stock jang gate:

    - The router weight operand ``weight.T.astype(float32)`` is hoisted
      into a cached device array. The composed path re-materializes the
      float32 copy of the fp16 router weight on every call (one dispatch
      plus several megabytes of wire per layer per token at decode);
      hoisting the subexpression leaves the matmul operand bytes and
      layout unchanged, so the gate logits are unchanged bit for bit.
    - At the single-token decode shape the select chain around
      ``mx.argpartition`` collapses into the two ``router_kernel``
      dispatches: ``fused_score_head`` before the selection and
      ``fused_topk_weights`` after it, replacing the compiled elementwise
      segment, the index slice-cast, the selected-score gather, the
      renorm sum, the divide/scale pair, and the downstream uint32 index
      cast (the returned ids already are uint32, the argpartition output
      dtype). The argpartition input equals the composed
      ``-(scores + bias)`` bit for bit and the selection op is shared, so
      the selected set and order match the stock path exactly, ties
      included.

    Hash-routed layers keep the stock eager select chain and take only
    the hoisted weight operand. Anything off the decode single-token
    shape or outside the transcribed envelope fails closed to the stock
    composed forms; the kill switches are read per call.
    """

    def __init__(self, original, *, mx, dsv4_model):
        self._original = original
        self._mx = mx
        self._dsv4_model = dsv4_model
        self._moespresso_dsv4_router_gate_contract = True
        self._precast_weight = None
        self._precast_ok_cached: bool | None = None
        self._select_ok_cached: bool | None = None

    def __getattr__(self, name: str):
        return getattr(self._original, name)

    def _precast_ok(self) -> bool:
        ok = self._precast_ok_cached
        if ok is None:
            weight = getattr(self._original, "weight", None)
            ok = (
                weight is not None
                and int(getattr(weight, "ndim", 0)) == 2
                and "float" in str(getattr(weight, "dtype", ""))
            )
            self._precast_ok_cached = ok
        return ok

    def _select_static_ok(self) -> bool:
        ok = self._select_ok_cached
        if ok is None:
            original = self._original
            args = original.args
            bias = getattr(original, "bias", None)
            k = int(getattr(args, "num_experts_per_tok", 0) or 0)
            ok = (
                not original.hash
                and bias is not None
                and bool(getattr(args, "norm_topk_prob", False))
                and 1 < k <= 64
                and math.isfinite(float(args.routed_scaling_factor))
            )
            self._select_ok_cached = ok
        return ok

    def _gates(self, x, *, precast: bool):
        mx = self._mx
        original = self._original
        if precast:
            w32 = self._precast_weight
            if w32 is None:
                w32 = original.weight.T.astype(mx.float32)
                self._precast_weight = w32
            _ROUTER_GATE_TRIM_CALL_COUNTS["precast"] += 1
            return x.astype(mx.float32) @ w32
        return x.astype(mx.float32) @ original.weight.T.astype(mx.float32)

    def __call__(self, x, input_ids=None):
        from moespresso.runtime.deepseek_v4 import router_kernel

        mx = self._mx
        original = self._original
        precast = router_kernel.router_precast_enabled() and self._precast_ok()
        rows = 1
        for dim in x.shape[:-1]:
            rows *= int(dim)
        select = (
            rows == 1
            and x.ndim == 3
            and router_kernel.router_select_enabled()
            and self._select_static_ok()
        )
        if not (precast or select):
            _ROUTER_GATE_TRIM_CALL_COUNTS["composed"] += 1
            return original(x, input_ids=input_ids)

        gates = self._gates(x, precast=precast)
        if original.hash:
            # The stock eager hash branch on the hoisted-operand logits.
            scores = mx.sqrt(mx.log1p(mx.exp(gates)))
            assert input_ids is not None, "hash-routed layer requires input_ids"
            inds = original.tid2eid[input_ids].astype(mx.int32)
            weights = mx.take_along_axis(scores, inds, axis=-1)
            if original.args.norm_topk_prob:
                weights = weights / mx.sum(weights, axis=-1, keepdims=True)
            weights = weights * original.args.routed_scaling_factor
            return inds, weights

        args = original.args
        k = int(args.num_experts_per_tok)
        if select and router_kernel.score_head_eligible(gates, original.bias):
            neg, orig_scores = router_kernel.fused_score_head(
                gates, original.bias)
            sel = mx.argpartition(neg, kth=k - 1, axis=-1)[..., :k]
            if router_kernel.topk_weights_eligible(orig_scores, sel):
                weights = router_kernel.fused_topk_weights(
                    orig_scores,
                    sel,
                    scaling=float(args.routed_scaling_factor),
                )
                _ROUTER_GATE_TRIM_CALL_COUNTS["select_kernel"] += 1
                return sel, weights
        _ROUTER_GATE_TRIM_CALL_COUNTS["select_composed"] += 1
        return self._dsv4_model.sqrtsoftplus_select(
            gates,
            original.bias,
            k,
            args.routed_scaling_factor,
            args.norm_topk_prob,
        )


def _patch_deepseek_v4_router_gate_trims(model) -> int:
    """Wrap DS4 MoE router gates in the decode router trim contract."""
    try:
        import mlx.core as mx
        import jang_tools.dsv4.mlx_model as dsv4_model
    except ImportError:  # pragma: no cover - DS4 runtime already checks this
        return 0

    patched = 0
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        if gate is None:
            continue
        if getattr(gate, "_moespresso_dsv4_router_gate_contract", False):
            continue
        # The transcription pins the jang gate contract exactly; foreign
        # gate classes fail closed to their own composed path.
        if type(gate) is not dsv4_model.Gate:
            continue
        object.__setattr__(
            mlp,
            "gate",
            _DeepseekV4RouterGateContract(gate, mx=mx, dsv4_model=dsv4_model),
        )
        patched += 1
    object.__setattr__(
        model,
        "_moespresso_dsv4_router_gate_trim_layers",
        patched,
    )
    return patched


class _Ratio4PrefillFastAttention:
    """DS4 ratio-4 prefill fast path based on DS4-c kernels."""

    def __init__(self, original, *, mx, dsv4_model, cache_cls):
        self._original = original
        self._mx = mx
        self._dsv4_model = dsv4_model
        self._cache_cls = cache_cls
        self._moespresso_dsv4_ratio4_prefill_fast_path = True
        self._moespresso_dsv4_ratio4_prefill_fast_calls = 0
        self._moespresso_dsv4_ratio4_prefill_fast_tokens = 0

    def __getattr__(self, name: str):
        return getattr(self._original, name)

    def __call__(self, x, mask=None, cache=None):
        mx = self._mx
        attn = self._original
        batch, tokens, _ = x.shape
        if (
            int(tokens) <= 1
            or int(batch) != 1
            or int(getattr(attn, "compress_ratio", 0) or 0) != 4
            or not isinstance(cache, self._cache_cls)
            or not hasattr(attn, "indexer")
        ):
            return attn(x, mask=mask, cache=cache)

        from moespresso.runtime.deepseek_v4.indexed_attention_kernel import (
            indexed_mixed_attention_prefill_live_f32,
            indexed_mixed_attention_prefill_live_f16,
            indexer_q_qat_live,
            indexer_score_operands,
            indexer_scores_tiled_live,
        )

        offset = int(cache.offset)
        q_residual = attn.q_norm(attn.wq_a(x))
        q = attn.wq_b(q_residual).reshape(
            batch, tokens, attn.n_heads, attn.head_dim,
        )
        q = mx.fast.rms_norm(
            q,
            weight=self._dsv4_model._get_q_norm_ones(attn.head_dim, q.dtype),
            eps=attn.args.rms_norm_eps,
        )
        q = q.transpose(0, 2, 1, 3)

        kv = attn.kv_norm(attn.wkv(x)).reshape(
            batch, tokens, 1, attn.head_dim,
        ).transpose(0, 2, 1, 3)

        q = self._dsv4_model._apply_partial_rope(q, attn.rope, offset)
        kv = self._dsv4_model._apply_partial_rope(kv, attn.rope, offset)
        kv, _ = cache.update_and_fetch(kv, kv)

        pooled = attn.compressor(x, attn.compress_rope, cache, offset)
        if pooled.shape[1] <= 0:
            topk = mx.zeros((batch, tokens, 0), dtype=mx.int32)
            if q.dtype == mx.float16:
                indexed_attention = indexed_mixed_attention_prefill_live_f16
                q_for_attention = q
            else:
                indexed_attention = indexed_mixed_attention_prefill_live_f32
                q_for_attention = q if q.dtype == mx.float32 else q.astype(mx.float32)
            kv_for_attention = kv if kv.dtype == mx.float16 else kv.astype(mx.float16)
            pooled_for_attention = (
                pooled if pooled.dtype == mx.float16 else pooled.astype(mx.float16)
            )
            heads = indexed_attention(
                q_for_attention,
                kv_for_attention,
                pooled_for_attention,
                topk,
                attn.attn_sink.astype(mx.float32),
                pos0=offset,
                window=int(attn.args.sliding_window),
                ratio=int(attn.compress_ratio),
            )
            out = self._dsv4_model._apply_partial_rope(
                heads, attn.rope, offset, inverse=True)
            out = out.transpose(0, 2, 1, 3).reshape(
                batch, tokens, attn.n_heads * attn.head_dim,
            )
            out = attn._grouped_output_projection(out)
            return attn.wo_b(out)

        indexer = attn.indexer
        if pooled.shape[1] <= indexer.index_topk:
            row_ids = mx.arange(pooled.shape[1], dtype=mx.int32)
            topk = mx.broadcast_to(row_ids[None, None, :], (batch, tokens, pooled.shape[1]))
        else:
            index_pooled = indexer.compressor(
                x, attn.compress_rope, cache, offset, state_key="indexer_state"
            )

            q_idx = indexer.wq_b(q_residual).reshape(
                batch, tokens, indexer.n_heads, indexer.head_dim,
            )
            q_idx = q_idx.transpose(0, 2, 1, 3)
            q_idx = self._dsv4_model._apply_partial_rope(q_idx, attn.rope, offset)
            q_idx = indexer_q_qat_live(q_idx.astype(mx.float32))

            state = getattr(cache, "indexer_state", None)
            if isinstance(state, dict):
                cached = state.get("pooled_qat")
                cached_rows = int(state.get("pooled_qat_rows", 0) or 0)
            else:
                cached = None
                cached_rows = 0
            if cached is not None and 0 < cached_rows <= int(index_pooled.shape[1]):
                if cached_rows == int(index_pooled.shape[1]):
                    index_pooled = cached
                else:
                    pooled_tail = _dsv4_prefill_pooled_qat(
                        mx, index_pooled[:, cached_rows:, :])
                    index_pooled = mx.concatenate([cached, pooled_tail], axis=1)
                    if isinstance(state, dict):
                        state["pooled_qat"] = index_pooled
                        state["pooled_qat_rows"] = int(index_pooled.shape[1])
            else:
                index_pooled = _dsv4_prefill_pooled_qat(mx, index_pooled)
                if isinstance(state, dict):
                    state["pooled_qat"] = index_pooled
                    state["pooled_qat_rows"] = int(index_pooled.shape[1])

            weights = indexer.weights_proj(x).astype(mx.float32) * (
                indexer.n_heads ** -0.5
            ) * indexer.scale
            q_scores, comp_scores = indexer_score_operands(q_idx, index_pooled)
            scores = indexer_scores_tiled_live(
                q_scores,
                weights,
                comp_scores,
                pos0=offset,
                ratio=int(attn.compress_ratio),
            )
            topk_rows = min(int(indexer.index_topk), int(index_pooled.shape[1]))
            topk = mx.argpartition(-scores, kth=topk_rows - 1, axis=-1)[..., :topk_rows]
            _maybe_dump_dsv4_indexer_selection(
                mx=mx,
                scores=scores,
                topk=topk,
                layer_index=getattr(indexer, "_layer_index", -1),
                start_pos=offset,
            )
            topk = mx.sort(topk.astype(mx.int32), axis=-1)

        if q.dtype == mx.float16:
            indexed_attention = indexed_mixed_attention_prefill_live_f16
            q_for_attention = q
        else:
            indexed_attention = indexed_mixed_attention_prefill_live_f32
            q_for_attention = q if q.dtype == mx.float32 else q.astype(mx.float32)
        kv_for_attention = kv if kv.dtype == mx.float16 else kv.astype(mx.float16)
        pooled_for_attention = (
            pooled if pooled.dtype == mx.float16 else pooled.astype(mx.float16)
        )
        heads = indexed_attention(
            q_for_attention,
            kv_for_attention,
            pooled_for_attention,
            topk,
            attn.attn_sink.astype(mx.float32),
            pos0=offset,
            window=int(attn.args.sliding_window),
            ratio=int(attn.compress_ratio),
        )
        out = self._dsv4_model._apply_partial_rope(heads, attn.rope, offset, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(
            batch, tokens, attn.n_heads * attn.head_dim,
        )
        out = attn._grouped_output_projection(out)
        self._moespresso_dsv4_ratio4_prefill_fast_calls += 1
        self._moespresso_dsv4_ratio4_prefill_fast_tokens += int(tokens)
        return attn.wo_b(out)


def _patch_deepseek_v4_ratio4_prefill_fast_path(model) -> int:
    """Install the product ratio-4 prefill fast path for DS4 packages."""
    try:
        import mlx.core as mx
        import jang_tools.dsv4.mlx_model as dsv4_model
        from jang_tools.dsv4.mlx_model import DeepseekV4Cache
    except ImportError:  # pragma: no cover - DS4 runtime already checks this
        return 0

    patched = 0
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if getattr(attn, "_moespresso_dsv4_ratio4_prefill_fast_path", False):
            continue
        if int(getattr(attn, "compress_ratio", 0) or 0) != 4:
            continue
        if not hasattr(attn, "indexer"):
            continue
        object.__setattr__(
            layer,
            "self_attn",
            _Ratio4PrefillFastAttention(
                attn,
                mx=mx,
                dsv4_model=dsv4_model,
                cache_cls=DeepseekV4Cache,
            ),
        )
        patched += 1
    object.__setattr__(model, "_moespresso_dsv4_ratio4_prefill_fast_path_layers", patched)
    return patched


class _Ratio4DecodeFusedAttention:
    """DS4 ratio-4 decode attention as two fused Metal dispatches.

    The composed decode step issues roughly sixty small dispatches per
    ratio-4 layer and is launch-bound; this wrapper routes the single-token
    shape to ``decode_attention_kernel`` (indexer QAT + scoring + top-k +
    KV-row prep in one dispatch, rope + selected-row SDPA + inverse rope in
    a second). Projections, norms, and compressor calls stay as MLX ops.
    The prepared KV row is bit-identical to the composed rope + FP8 round
    trip, so the local cache append bypasses the FP8 wrapper and replicates
    ``RotatingKVCache._update_in_place`` bookkeeping directly.

    Any shape, dtype, or cache-state condition outside the proven decode
    contract delegates to the composed path before any cache mutation; the
    checks that can only run after the compressor has advanced fall back to
    ``_composed_decode_tail``, which finishes the token with exactly the
    composed op sequence.
    """

    def __init__(self, original, *, mx, dsv4_model, cache_cls):
        self._original = original
        self._mx = mx
        self._dsv4_model = dsv4_model
        self._cache_cls = cache_cls
        self._moespresso_dsv4_fused_decode_attention = True
        self.fused_decode_calls = 0
        self.fused_decode_composed_tail_calls = 0
        self._fused_sinks_f16 = None

    def __getattr__(self, name: str):
        return getattr(self._original, name)

    def _decode_eligible(self, x, cache) -> bool:
        from moespresso.runtime.deepseek_v4 import decode_attention_kernel

        if not decode_attention_kernel.fused_decode_enabled():
            return False
        if os.environ.get("MOESPRESSO_DSV4_INDEXER_DUMP_PREFIX"):
            return False
        mx = self._mx
        attn = self._original
        if x.ndim != 3 or int(x.shape[0]) != 1 or int(x.shape[1]) != 1:
            return False
        if x.dtype != mx.float16:
            return False
        if int(getattr(attn, "compress_ratio", 0) or 0) != 4:
            return False
        if not hasattr(attn, "indexer"):
            return False
        if int(attn.head_dim) != 512 or int(getattr(attn.rope, "dims", 0)) != 64:
            return False
        indexer = attn.indexer
        if int(indexer.head_dim) != 128 or not 0 < int(indexer.n_heads) <= 64:
            return False
        if int(indexer.index_topk) <= 0:
            return False
        if not isinstance(cache, self._cache_cls):
            return False
        # The prepared KV row bakes in the FP8 round trip, so the cache must
        # carry the served FP8 update contract for the direct append to be
        # equivalent to update_and_fetch.
        if not getattr(cache, "_moespresso_dsv4_fp8_kv_cache", False):
            return False
        local = getattr(cache, "local", None)
        window = getattr(attn.args, "sliding_window", None)
        max_size = getattr(local, "max_size", None)
        if window is None or max_size is None or int(max_size) != int(window):
            return False
        if int(getattr(local, "keep", -1)) != 0:
            return False
        keys = getattr(local, "keys", None)
        values = getattr(local, "values", None)
        if keys is None or values is None:
            return False
        if int(local.offset) < int(max_size):
            return False
        expected = (1, 1, int(max_size), int(attn.head_dim))
        if tuple(int(v) for v in keys.shape) != expected:
            return False
        if tuple(int(v) for v in values.shape) != expected:
            return False
        if keys.dtype != mx.float16 or values.dtype != mx.float16:
            return False
        return True

    def _composed_decode_tail(self, *, x, q_residual, q, kv, pooled, cache,
                              offset, run_indexer, topk):
        """Finish the token with the composed op sequence.

        ``q`` and ``kv`` arrive pre-rope in the ``[1, 1, heads, dim]``
        layout; the compressor has already advanced. ``run_indexer`` re-uses
        the wrapped indexer (which advances its own compressor state);
        ``topk`` carries an already-computed selection when the fused path
        bailed after the indexer state advance.
        """
        mx = self._mx
        attn = self._original
        dsv4 = self._dsv4_model
        self.fused_decode_composed_tail_calls += 1
        q = q.transpose(0, 2, 1, 3)
        kv = kv.transpose(0, 2, 1, 3)
        q = dsv4._apply_partial_rope(q, attn.rope, offset)
        kv = dsv4._apply_partial_rope(kv, attn.rope, offset)
        kv, _ = cache.update_and_fetch(kv, kv)
        if run_indexer and pooled.shape[1] > attn.indexer.index_topk:
            topk = attn.indexer(
                x, q_residual, attn.compress_rope, attn.rope, cache, offset,
            )
        if topk is not None:
            idx = topk[:, None, :, :, None]
            expanded = mx.broadcast_to(
                pooled[:, None, None, :, :],
                (1, 1, 1, pooled.shape[1], attn.head_dim),
            )
            pooled_kv = mx.take_along_axis(
                expanded,
                mx.broadcast_to(idx, idx.shape[:-1] + (attn.head_dim,)),
                axis=3,
            ).reshape(1, 1, -1, attn.head_dim)
        else:
            pooled_kv = pooled[:, None]
        full_kv = mx.concatenate([kv, pooled_kv], axis=2)
        out = dsv4.scaled_dot_product_attention(
            q, full_kv, full_kv,
            cache=cache, scale=attn.softmax_scale, mask=None,
            sinks=attn.attn_sink.astype(q.dtype),
        )
        out = dsv4._apply_partial_rope(out, attn.rope, offset, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(
            1, 1, attn.n_heads * attn.head_dim)
        out = attn._grouped_output_projection(out)
        return attn.wo_b(out)

    def __call__(self, x, mask=None, cache=None):
        if cache is None or not self._decode_eligible(x, cache):
            return self._original(x, mask=mask, cache=cache)

        from moespresso.runtime.deepseek_v4 import decode_attention_kernel

        mx = self._mx
        attn = self._original
        dsv4 = self._dsv4_model
        indexer = attn.indexer
        local = cache.local
        offset = int(cache.offset)
        write_idx = int(local._idx)
        if write_idx == int(local.max_size):
            write_idx = int(local.keep)

        q_residual = attn.q_norm(attn.wq_a(x))
        q = attn.wq_b(q_residual).reshape(1, 1, attn.n_heads, attn.head_dim)
        q = mx.fast.rms_norm(
            q,
            weight=dsv4._get_q_norm_ones(attn.head_dim, q.dtype),
            eps=attn.args.rms_norm_eps,
        )
        kv = attn.kv_norm(attn.wkv(x)).reshape(1, 1, 1, attn.head_dim)

        pooled = attn.compressor(x, attn.compress_rope, cache, offset)
        n_rows = int(pooled.shape[1])
        topk_width = int(indexer.index_topk)
        if (
            n_rows <= topk_width
            or n_rows >= (1 << 16)
            or pooled.dtype != mx.float16
            or q.dtype != mx.float16
            or kv.dtype != mx.float16
        ):
            # The composed tail re-runs the wrapped indexer, which advances
            # the indexer compressor state exactly as the composed path.
            return self._composed_decode_tail(
                x=x, q_residual=q_residual, q=q, kv=kv, pooled=pooled,
                cache=cache, offset=offset, run_indexer=True, topk=None,
            )

        # Indexer-side prep, mirroring _IndexerDS4ScoreContract: advance the
        # indexer compressor, maintain the pooled QAT cache, and project the
        # indexer queries and head weights.
        index_pooled = indexer.compressor(
            x, attn.compress_rope, cache, offset, state_key="indexer_state")
        state = getattr(cache, "indexer_state", None)
        cached = state.get("pooled_qat") if isinstance(state, dict) else None
        cached_rows = (
            int(state.get("pooled_qat_rows", 0) or 0)
            if isinstance(state, dict) else 0
        )
        index_rows = int(index_pooled.shape[1])
        if cached is not None and 0 < cached_rows <= index_rows:
            if cached_rows == index_rows:
                index_pooled_qat = cached
            else:
                tail_rows = _dsv4_indexer_qat(mx, index_pooled[:, cached_rows:, :])
                index_pooled_qat = mx.concatenate([cached, tail_rows], axis=1)
                if isinstance(state, dict):
                    state["pooled_qat"] = index_pooled_qat
                    state["pooled_qat_rows"] = index_rows
        else:
            index_pooled_qat = _dsv4_indexer_qat(mx, index_pooled)
            if isinstance(state, dict):
                state["pooled_qat"] = index_pooled_qat
                state["pooled_qat_rows"] = index_rows

        q_idx = indexer.wq_b(q_residual).reshape(
            1, 1, indexer.n_heads, indexer.head_dim).transpose(0, 2, 1, 3)
        q_idx = dsv4._apply_partial_rope(q_idx, attn.rope, offset)
        weights = indexer.weights_proj(x).astype(mx.float32) * (
            indexer.n_heads ** -0.5
        )

        if index_rows != n_rows or q_idx.dtype not in (
            mx.float32, mx.float16, mx.bfloat16,
        ):
            # The indexer state already advanced; score with the composed
            # ops instead of re-entering the wrapped indexer.
            q_idx_qat = _dsv4_indexer_qat(mx, q_idx)
            scores = (
                q_idx_qat.astype(mx.float32)
                @ index_pooled_qat[:, None].swapaxes(-1, -2).astype(mx.float32)
            )
            scores = mx.maximum(scores, 0) * indexer.scale
            scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
            k = min(topk_width, index_rows)
            topk = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
            return self._composed_decode_tail(
                x=x, q_residual=q_residual, q=q, kv=kv, pooled=pooled,
                cache=cache, offset=offset, run_indexer=False, topk=topk,
            )

        params = mx.array([offset, write_idx], dtype=mx.int32)
        inv_freq = attn.rope.inv_freq
        sinks = self._fused_sinks_f16
        if sinks is None:
            sinks = attn.attn_sink.astype(mx.float16)
            mx.eval(sinks)
            self._fused_sinks_f16 = sinks

        sel, row, _scores = decode_attention_kernel.fused_decode_prep(
            q_idx.reshape(int(indexer.n_heads), int(indexer.head_dim)),
            index_pooled_qat.reshape(n_rows, int(indexer.head_dim)),
            weights.reshape(int(indexer.n_heads)),
            kv.reshape(int(attn.head_dim)),
            inv_freq,
            params,
            scale=float(indexer.scale),
            topk=topk_width,
        )
        heads = decode_attention_kernel.fused_decode_sdpa(
            q.reshape(int(attn.n_heads), int(attn.head_dim)),
            local.keys.reshape(int(local.max_size), int(attn.head_dim)),
            row,
            pooled.reshape(n_rows, int(attn.head_dim)),
            sel,
            sinks,
            inv_freq,
            params,
            scale=float(attn.softmax_scale),
        )

        # The prepared row is bit-identical to the composed rope + FP8 round
        # trip, so append it directly, replicating the steady-state
        # RotatingKVCache._update_in_place bookkeeping (no growth, no trim).
        row4 = row.reshape(1, 1, 1, int(attn.head_dim))
        local.keys[..., write_idx:write_idx + 1, :] = row4
        local.values[..., write_idx:write_idx + 1, :] = row4
        local.offset += 1
        local._idx = write_idx + 1

        self.fused_decode_calls += 1
        out = heads.reshape(1, 1, attn.n_heads * attn.head_dim)
        out = attn._grouped_output_projection(out)
        return attn.wo_b(out)


def _patch_deepseek_v4_ratio4_decode_fused_attention(model) -> int:
    """Install the fused ratio-4 decode attention island on DS4 layers."""
    try:
        import mlx.core as mx
        import jang_tools.dsv4.mlx_model as dsv4_model
        from jang_tools.dsv4.mlx_model import DeepseekV4Cache
    except ImportError:  # pragma: no cover - DS4 runtime already checks this
        return 0
    from moespresso.runtime.deepseek_v4 import decode_attention_kernel

    if not decode_attention_kernel.fused_decode_enabled():
        object.__setattr__(
            model, "_moespresso_dsv4_fused_decode_attention_layers", 0)
        return 0

    patched = 0
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if getattr(attn, "_moespresso_dsv4_fused_decode_attention", False):
            continue
        if int(getattr(attn, "compress_ratio", 0) or 0) != 4:
            continue
        if not hasattr(attn, "indexer"):
            continue
        object.__setattr__(
            layer,
            "self_attn",
            _Ratio4DecodeFusedAttention(
                attn,
                mx=mx,
                dsv4_model=dsv4_model,
                cache_cls=DeepseekV4Cache,
            ),
        )
        patched += 1
    object.__setattr__(
        model, "_moespresso_dsv4_fused_decode_attention_layers", patched)
    return patched


_DEEPSEEK_V4_BANDED_PREFILL_BLOCK = 512
_DEEPSEEK_V4_BANDED_PREFILL_PLAN_CACHE: dict = {}
_DEEPSEEK_V4_BANDED_PREFILL_PLAN_CACHE_MAX = 8
_DEEPSEEK_V4_SHARED_WINDOW_MASK_MEMO: dict = {}
_DEEPSEEK_V4_SHARED_WINDOW_MASK_MEMO_MAX = 32

_BANDED_PREFILL_CALL_COUNTS = {
    "mma": 0,
    "sdpa": 0,
    "mma_offset": 0,
    "composed_offset": 0,
}


def banded_prefill_call_counts() -> dict[str, int]:
    """Snapshot of banded prefill attention engagements by form."""
    return dict(_BANDED_PREFILL_CALL_COUNTS)


def _banded_prefill_mma_enabled() -> bool:
    """Gate for routing banded prefill layers through the mma consumer.

    The mma route computes the same band-plus-pool visibility with half
    operand staging and float32 accumulation (the operand contract the
    ratio-4 layers already serve), replacing the composed SDPA fallback
    that materializes the [blocks, heads, block, band] score tensor at head
    dim 512. This numerically valid variant changes accumulation order and
    therefore lacks bit identity. The default rides the math-change gate campaign
    recorded in the speed log.
    Kill switch ``MOESPRESSO_DSV4_BANDED_PREFILL_MMA=0`` restores the
    batched banded SDPA form.
    """
    import os

    return os.environ.get("MOESPRESSO_DSV4_BANDED_PREFILL_MMA", "1") != "0"


def _banded_prefill_offset_enabled() -> bool:
    """Gate for serving banded prefill chunks at cache offsets past zero.

    Default on. The route passes the true cache offset through the banded
    mma consumer, which computes every visibility predicate from ``pos0``;
    the ratio-4 path serves the same kernel family at offset on every
    chunk. It engages only past the 4096 single-chunk gate and measured
    1.10x faster prefill at depth 7698 and 1.14x at 15406 with transient
    peaks of 10.7 to 11.1 GiB against 12.2 to 16.5 GiB for the composed
    fallback, with engaged teacher-forced NLL reading better at both
    depths. Kill switch ``MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET=0``
    restores the offset-zero-only behavior byte-identically: every chunk
    after the first drops the banded layers to the composed SDPA fallback
    that materializes the score tensor at head dim 512.
    """
    import os

    return os.environ.get("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", "1") != "0"


def _deepseek_v4_banded_mma_offset_ready(
    mx, *, q, kv, ratio, scale, cache, cache_cls,
) -> bool:
    """Pre-mutation eligibility for the banded mma route at offset.

    Offset chunks have no in-wrapper fallback: the batched SDPA plan
    assumes local key positions starting at zero, and the composed
    original must see an unmutated cache. Every predicate the mma engine
    checks on wrapper-supplied operands is decided here, before
    ``update_and_fetch`` or the compressor advance state, so an ineligible
    call fails closed to the composed form with the cache untouched.
    """
    from moespresso.runtime.deepseek_v4.indexed_attention_kernel import (
        _prefill_live_mma_enabled,
    )

    if not _banded_prefill_mma_enabled() or not _prefill_live_mma_enabled():
        return False
    n_heads = int(q.shape[1])
    head_dim = int(q.shape[3])
    if head_dim != 512 or n_heads % 16 != 0:
        return False
    if float(scale) != float(head_dim) ** -0.5:
        return False
    floats = (mx.float16, mx.bfloat16, mx.float32)
    if q.dtype not in floats or kv.dtype not in floats:
        return False
    # The cumulative pool for the prefix lives in the composite cache; a
    # compressed layer without one cannot rebuild the prefix pool rows
    # mid-stream.
    if int(ratio) > 0 and not isinstance(cache, cache_cls):
        return False
    return True


def _deepseek_v4_banded_mma_attention(
    mx, *, q, kv, pooled, sinks, window, ratio, scale, pos0=0,
):
    """Banded heads through the mma consumer, or None when ineligible.

    Fail-closed: any geometry or dtype outside the kernel contract returns
    None and the caller keeps the batched banded SDPA form. Ascending pool
    row ids plus the kernel's visibility rule reproduce the compressed-pool
    visibility predicate, so ratio-128 layers attend exactly the rows the
    banded plan admits; sliding-window layers pass a one-row zero comp
    buffer that the zero-width id list never reads. ``pos0`` is the
    absolute position of the first query row; the kernel keys the band and
    pool predicates to absolute positions, so a chunk at offset reads the
    trailing KV rows the cache returns and attends the same band a
    single-chunk call would.
    """
    from moespresso.runtime.deepseek_v4.indexed_attention_kernel import (
        _prefill_live_mma_enabled,
        banded_prefill_attention_live,
    )

    n_heads = int(q.shape[1])
    tokens = int(q.shape[2])
    head_dim = int(q.shape[3])
    if head_dim != 512 or n_heads % 16 != 0:
        return None
    # The consumer kernel applies rsqrt(head_dim); any other served scale
    # keeps the SDPA form.
    if float(scale) != float(head_dim) ** -0.5:
        return None
    if not _prefill_live_mma_enabled():
        return None
    floats = (mx.float16, mx.bfloat16, mx.float32)
    if q.dtype not in floats or kv.dtype not in floats:
        return None
    pooled_rows = 0 if pooled is None else int(pooled.shape[1])
    if pooled_rows > 0 and (int(ratio) <= 0 or pooled.dtype not in floats):
        return None
    kv_f16 = kv if kv.dtype == mx.float16 else kv.astype(mx.float16)
    if pooled_rows > 0:
        comp = pooled if pooled.dtype == mx.float16 else pooled.astype(mx.float16)
        topk = mx.broadcast_to(
            mx.arange(pooled_rows, dtype=mx.int32)[None, None, :],
            (1, tokens, pooled_rows),
        )
        kernel_ratio = int(ratio)
    else:
        comp = mx.zeros((1, 1, head_dim), dtype=mx.float16)
        topk = mx.zeros((1, tokens, 0), dtype=mx.int32)
        kernel_ratio = max(int(ratio), 1)
    # bfloat16 queries round once to half (bfloat16 values are exact in
    # float32, so the direct cast equals the scalar consumers' f32-to-half
    # per-row rounding); float16 and float32 pass through unchanged.
    q_in = q if q.dtype in (mx.float16, mx.float32) else q.astype(mx.float16)
    return banded_prefill_attention_live(
        q_in,
        kv_f16,
        comp,
        topk,
        sinks.astype(mx.float32),
        pos0=int(pos0),
        window=int(window),
        ratio=kernel_ratio,
    )


def _deepseek_v4_shared_window_mask_matches(mask, *, tokens, offset, window) -> bool:
    """Check that a mask is exactly the shared sliding-window bool mask.

    The shared DS4 model mask admits key column ``k`` for query row ``r``
    (absolute position ``offset + r``) iff ``k <= offset + r`` and
    ``k > offset + r - window``. Accepts the ``[tokens, S]`` and
    ``[1, 1, tokens, S]`` layouts with ``S == offset + tokens``. The content
    comparison runs once per mask array and is memoized by identity and
    dimensions.
    """
    import mlx.core as mx

    if mask is None or window is None or int(window) <= 0:
        return False
    if getattr(mask, "dtype", None) != mx.bool_:
        return False
    shape = tuple(int(dim) for dim in mask.shape)
    if len(shape) == 4:
        if shape[0] != 1 or shape[1] != 1:
            return False
        rows, cols = shape[2], shape[3]
    elif len(shape) == 2:
        rows, cols = shape
    else:
        return False
    if rows != int(tokens) or cols != int(offset) + int(tokens):
        return False
    key = (id(mask), shape, int(offset), int(window))
    cached = _DEEPSEEK_V4_SHARED_WINDOW_MASK_MEMO.get(key)
    if cached is not None:
        return cached
    content = mask[0, 0] if len(shape) == 4 else mask
    q_pos = int(offset) + mx.arange(rows)[:, None]
    k_pos = mx.arange(cols)[None, :]
    expected = (k_pos <= q_pos) & (k_pos > q_pos - int(window))
    matches = bool(mx.array_equal(content, expected))
    if len(_DEEPSEEK_V4_SHARED_WINDOW_MASK_MEMO) >= (
        _DEEPSEEK_V4_SHARED_WINDOW_MASK_MEMO_MAX
    ):
        _DEEPSEEK_V4_SHARED_WINDOW_MASK_MEMO.clear()
    _DEEPSEEK_V4_SHARED_WINDOW_MASK_MEMO[key] = matches
    return matches


def _deepseek_v4_banded_prefill_plan(mx, *, tokens, pooled_rows, window, ratio, block):
    """Build gather indices and the bool block mask for one banded SDPA.

    Query blocks of ``block`` rows fold into the SDPA batch dim. Each block
    gathers its overlapping KV band (``window - 1`` lead-in keys plus the
    block's own ``block`` keys) followed by the pooled tail columns. The
    block mask carries the sliding-window predicate for band columns and the
    compressed-pool visibility predicate for pooled columns. Query rows past
    ``tokens`` are zero padding: each keeps its band's first (in-range)
    column visible so the softmax stays finite, and its output row is
    sliced away.
    """
    key = (int(tokens), int(pooled_rows), int(window), int(ratio), int(block))
    plan = _DEEPSEEK_V4_BANDED_PREFILL_PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    tokens, pooled_rows, window, ratio, block = key
    blocks = (tokens + block - 1) // block
    band_local = window - 1 + block
    starts = mx.arange(blocks) * block - (window - 1)
    # Band columns hold absolute key positions; clamp the gather so every
    # index stays inside the local KV (out-of-range columns are masked).
    band_idx = mx.clip(
        starts[:, None] + mx.arange(band_local)[None, :], 0, tokens - 1)
    row = mx.arange(block)
    col = mx.arange(band_local)
    abs_q = starts[:, None, None] + (window - 1) + row[None, :, None]
    abs_k = starts[:, None, None] + col[None, None, :]
    band_mask = (
        (abs_k <= abs_q)
        & (abs_k > abs_q - window)
        & (abs_k >= 0)
        & (abs_q < tokens)
    )
    band_mask = band_mask | ((abs_q >= tokens) & (col[None, None, :] == 0))
    if pooled_rows > 0:
        # Exact _dsv4_compressed_visibility predicate: pool row k is visible
        # to absolute query position q iff (k + 1) * ratio <= q + 1.
        pool_idx = mx.arange(pooled_rows)
        pooled_mask = ((pool_idx[None, None, :] + 1) * ratio) <= (abs_q + 1)
        block_mask = mx.concatenate([band_mask, pooled_mask], axis=2)
        gather_idx = mx.concatenate(
            [
                band_idx,
                mx.broadcast_to(
                    (tokens + pool_idx)[None, :], (blocks, pooled_rows)),
            ],
            axis=1,
        )
    else:
        block_mask = band_mask
        gather_idx = band_idx
    block_mask = block_mask[:, None]
    mx.eval(gather_idx, block_mask)
    if len(_DEEPSEEK_V4_BANDED_PREFILL_PLAN_CACHE) >= (
        _DEEPSEEK_V4_BANDED_PREFILL_PLAN_CACHE_MAX
    ):
        _DEEPSEEK_V4_BANDED_PREFILL_PLAN_CACHE.clear()
    plan = (gather_idx, block_mask, blocks)
    _DEEPSEEK_V4_BANDED_PREFILL_PLAN_CACHE[key] = plan
    return plan


class _BandedPrefillAttention:
    """Banded DS4 prefill attention over the local band plus pool.

    Sliding-window layers only attend a ``window``-token local band plus a
    small pooled tail, but the dense served path pays quadratic SDPA FLOPs
    over the full key length. Eligible chunks route through the
    simdgroup-mma consumer (one fused dispatch with online softmax; head
    dim 512 has no fused MLX SDPA kernel, so the SDPA form runs the
    composed fallback that materializes the score tensor). The fallback
    form folds query blocks into the SDPA batch dim against materialized
    overlapping KV bands and computes the same result in a single call. A
    per-block Python loop over the same decomposition measured slower than
    the dense path (272 vs 176 ms on a captured ratio-128 layer); the
    banded win only exists in the batched or fused forms.

    Chunks at cache offsets past zero serve the mma form by default: the
    true offset flows to the rope sites, the compressor, and the kernel
    ``pos0``, and the rotating cache hands exactly the band lead-in rows.
    The batched SDPA plan assumes local key positions starting at zero, so
    at offset the only fallback is the composed original and every
    eligibility predicate is decided before any cache mutation. The kill
    switch ``MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET=0`` serves the composed
    original unchanged on every offset chunk.
    """

    def __init__(self, original, *, mx, dsv4_model, cache_cls):
        self._original = original
        self._mx = mx
        self._dsv4_model = dsv4_model
        self._cache_cls = cache_cls
        self._moespresso_dsv4_banded_prefill = True
        self.banded_prefill_calls = 0

    def __getattr__(self, name: str):
        return getattr(self._original, name)

    def __call__(self, x, mask=None, cache=None):
        mx = self._mx
        attn = self._original
        batch, tokens, _ = x.shape
        tokens = int(tokens)
        block = int(_DEEPSEEK_V4_BANDED_PREFILL_BLOCK)
        window = getattr(attn.args, "sliding_window", None)
        ratio = int(getattr(attn, "compress_ratio", 0) or 0)
        offset = getattr(cache, "offset", None) if cache is not None else None
        if (
            int(batch) != 1
            or tokens <= 1
            or offset is None
            or hasattr(cache, "bits")
            or window is None
        ):
            return attn(x, mask=mask, cache=cache)
        offset = int(offset)
        if offset != 0 and not _banded_prefill_offset_enabled():
            return attn(x, mask=mask, cache=cache)
        if (
            # The band must be narrower than the dense key length or the
            # banded call is dense work plus padding overhead. The gate is
            # scoped to offset zero: at offset the composed alternative
            # reads at least ``window - 1 + tokens`` keys plus pool, so the
            # banded form is never worse and the route stays uniform across
            # later chunks, including a short trailing chunk.
            (offset == 0 and tokens <= int(window) + block)
            or not _deepseek_v4_shared_window_mask_matches(
                mask, tokens=tokens, offset=offset, window=int(window))
        ):
            if offset != 0:
                _BANDED_PREFILL_CALL_COUNTS["composed_offset"] += 1
            return attn(x, mask=mask, cache=cache)
        window = int(window)
        # Mirror of the original's indexer condition in cumulative form:
        # after this chunk the compressor pool holds
        # ``(offset + tokens) // ratio`` rows, so top-k selection would
        # engage iff that exceeds index_topk. The patched classes carry no
        # indexer (ratio-128 layers have none in the jang graph), so this
        # stays a safety mirror. Decide before any cache mutation.
        if (
            ratio > 0
            and hasattr(attn, "indexer")
            and (offset + tokens) // ratio > int(attn.indexer.index_topk)
        ):
            if offset != 0:
                _BANDED_PREFILL_CALL_COUNTS["composed_offset"] += 1
            return attn(x, mask=mask, cache=cache)

        q_residual = attn.q_norm(attn.wq_a(x))
        q = attn.wq_b(q_residual).reshape(1, tokens, attn.n_heads, attn.head_dim)
        q = mx.fast.rms_norm(
            q,
            weight=self._dsv4_model._get_q_norm_ones(attn.head_dim, q.dtype),
            eps=attn.args.rms_norm_eps,
        )
        q = q.transpose(0, 2, 1, 3)
        kv = attn.kv_norm(attn.wkv(x)).reshape(
            1, tokens, 1, attn.head_dim).transpose(0, 2, 1, 3)
        if offset != 0 and not _deepseek_v4_banded_mma_offset_ready(
            mx,
            q=q,
            kv=kv,
            ratio=ratio,
            scale=attn.softmax_scale,
            cache=cache,
            cache_cls=self._cache_cls,
        ):
            _BANDED_PREFILL_CALL_COUNTS["composed_offset"] += 1
            return attn(x, mask=mask, cache=cache)
        q = self._dsv4_model._apply_partial_rope(q, attn.rope, offset)
        kv = self._dsv4_model._apply_partial_rope(kv, attn.rope, offset)
        kv, _ = cache.update_and_fetch(kv, kv)

        pooled = None
        if ratio > 0:
            v4_cache = cache if isinstance(cache, self._cache_cls) else None
            if v4_cache is not None or tokens >= ratio:
                candidate = attn.compressor(
                    x, attn.compress_rope, v4_cache, offset)
                if int(candidate.shape[1]) > 0:
                    pooled = candidate

        pooled_rows = 0 if pooled is None else int(pooled.shape[1])
        heads = None
        if offset != 0 or _banded_prefill_mma_enabled():
            heads = _deepseek_v4_banded_mma_attention(
                mx,
                q=q,
                kv=kv,
                pooled=pooled,
                sinks=attn.attn_sink,
                window=window,
                ratio=ratio,
                scale=attn.softmax_scale,
                pos0=offset,
            )
        if offset != 0:
            if heads is None:
                # The hoisted eligibility check decides every engine
                # predicate before the cache advances, so no fallback can
                # serve here: the SDPA plan is offset-zero geometry and the
                # composed original would double-append the chunk.
                raise RuntimeError(
                    "banded prefill offset route lost mma eligibility "
                    "after cache mutation"
                )
            _BANDED_PREFILL_CALL_COUNTS["mma_offset"] += 1
        elif heads is not None:
            _BANDED_PREFILL_CALL_COUNTS["mma"] += 1
        else:
            _BANDED_PREFILL_CALL_COUNTS["sdpa"] += 1
            gather_idx, block_mask, blocks = _deepseek_v4_banded_prefill_plan(
                mx,
                tokens=tokens,
                pooled_rows=pooled_rows,
                window=window,
                ratio=ratio,
                block=block,
            )
            full_kv = kv if pooled is None else mx.concatenate(
                [kv, pooled[:, None]], axis=2)
            kv_bands = full_kv[0, 0][gather_idx][:, None]
            padded = blocks * block
            if padded != tokens:
                q = mx.concatenate(
                    [
                        q,
                        mx.zeros(
                            (1, attn.n_heads, padded - tokens, attn.head_dim),
                            dtype=q.dtype,
                        ),
                    ],
                    axis=2,
                )
            q_blocks = q[0].reshape(
                attn.n_heads, blocks, block, attn.head_dim).transpose(1, 0, 2, 3)
            heads = self._dsv4_model.scaled_dot_product_attention(
                q_blocks,
                kv_bands,
                kv_bands,
                cache=None,
                scale=attn.softmax_scale,
                mask=block_mask,
                sinks=attn.attn_sink.astype(q.dtype),
            )
            heads = heads.transpose(1, 0, 2, 3).reshape(
                1, attn.n_heads, padded, attn.head_dim)
            if padded != tokens:
                heads = heads[:, :, :tokens]
        out = self._dsv4_model._apply_partial_rope(
            heads, attn.rope, offset, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(
            1, tokens, attn.n_heads * attn.head_dim)
        out = attn._grouped_output_projection(out)
        self.banded_prefill_calls += 1
        return attn.wo_b(out)


def _patch_deepseek_v4_banded_prefill_attention(model) -> int:
    """Install the batched banded prefill attention on DS4 window layers."""
    try:
        import mlx.core as mx
        import jang_tools.dsv4.mlx_model as dsv4_model
        from jang_tools.dsv4.mlx_model import DeepseekV4Cache
    except ImportError:  # pragma: no cover - DS4 runtime already checks this
        return 0

    patched = 0
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if getattr(attn, "_moespresso_dsv4_banded_prefill", False):
            continue
        if int(getattr(attn, "compress_ratio", 0) or 0) not in (0, 128):
            continue
        object.__setattr__(
            layer,
            "self_attn",
            _BandedPrefillAttention(
                attn,
                mx=mx,
                dsv4_model=dsv4_model,
                cache_cls=DeepseekV4Cache,
            ),
        )
        patched += 1
    object.__setattr__(model, "_moespresso_dsv4_banded_prefill_layers", patched)
    return patched


def deepseek_v4_indexer_layer_stats(model) -> list[dict[str, int]]:
    """Return shape-derived DS4 indexer scoring counters by layer."""
    rows = []
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer_index, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        indexer = getattr(attn, "indexer", None)
        if indexer is None:
            continue
        if not getattr(indexer, "_moespresso_dsv4_indexer_score_contract", False):
            continue
        rows.append({
            "layer": int(layer_index),
            "compress_ratio": int(getattr(attn, "compress_ratio", 0) or 0),
            "indexer_score_contract_calls": int(
                indexer._moespresso_dsv4_indexer_score_contract_calls
            ),
            "indexer_score_contract_tokens": int(
                indexer._moespresso_dsv4_indexer_score_contract_tokens
            ),
            "indexer_score_contract_pooled_rows": int(
                indexer._moespresso_dsv4_indexer_score_contract_pooled_rows
            ),
            "indexer_score_contract_topk_rows": int(
                indexer._moespresso_dsv4_indexer_score_contract_topk_rows
            ),
            "indexer_score_contract_score_elements": int(
                indexer._moespresso_dsv4_indexer_score_contract_score_elements
            ),
            "indexer_score_contract_qat_elements": int(
                indexer._moespresso_dsv4_indexer_score_contract_qat_elements
            ),
            "indexer_score_contract_cached_pooled_rows": int(
                indexer._moespresso_dsv4_indexer_score_contract_cached_pooled_rows
            ),
            "indexer_score_contract_new_qat_pooled_rows": int(
                indexer._moespresso_dsv4_indexer_score_contract_new_qat_pooled_rows
            ),
            "indexer_score_contract_fused_score_calls": int(
                indexer._moespresso_dsv4_indexer_score_contract_fused_score_calls
            ),
            "indexer_score_contract_decode_qat_kernel_calls": int(
                indexer._moespresso_dsv4_indexer_score_contract_decode_qat_kernel_calls
            ),
            "indexer_score_contract_fixed_state_calls": int(
                indexer._moespresso_dsv4_indexer_score_contract_fixed_state_calls
            ),
            "indexer_score_contract_score_tail_kernel_calls": int(
                indexer._moespresso_dsv4_indexer_score_contract_score_tail_kernel_calls
            ),
        })
    return rows


class _DeepseekV4AttentionShapeStats:
    """Proxy a DS4 attention module and count SDPA shape burden."""

    def __init__(self, original, *, layer_index: int):
        self._original = original
        self._layer_index = int(layer_index)
        self._moespresso_dsv4_attention_shape_stats = True
        self._moespresso_dsv4_attention_sdpa_calls = 0
        self._moespresso_dsv4_attention_sdpa_tokens = 0
        self._moespresso_dsv4_attention_sdpa_key_rows = 0
        self._moespresso_dsv4_attention_sdpa_score_elements = 0
        self._moespresso_dsv4_attention_sdpa_value_elements = 0
        self._moespresso_dsv4_attention_sdpa_max_key_rows = 0

    def __getattr__(self, name: str):
        return getattr(self._original, name)

    def __call__(self, *args, **kwargs):
        token = _DSV4_ATTENTION_STATS_CONTEXT.set(self)
        try:
            return self._original(*args, **kwargs)
        finally:
            _DSV4_ATTENTION_STATS_CONTEXT.reset(token)

    def _record_sdpa(self, q, k, v) -> None:
        del v
        if len(q.shape) < 4 or len(k.shape) < 4:
            return
        batch = int(q.shape[0])
        heads = int(q.shape[1])
        tokens = int(q.shape[2])
        key_rows = int(k.shape[2])
        head_dim = int(k.shape[-1])
        self._moespresso_dsv4_attention_sdpa_calls += 1
        self._moespresso_dsv4_attention_sdpa_tokens += batch * tokens
        self._moespresso_dsv4_attention_sdpa_key_rows += batch * tokens * key_rows
        self._moespresso_dsv4_attention_sdpa_score_elements += (
            batch * heads * tokens * key_rows
        )
        self._moespresso_dsv4_attention_sdpa_value_elements += (
            batch * tokens * key_rows * head_dim
        )
        self._moespresso_dsv4_attention_sdpa_max_key_rows = max(
            self._moespresso_dsv4_attention_sdpa_max_key_rows,
            key_rows,
        )


def _patch_deepseek_v4_attention_shape_stats(model) -> int:
    """Install per-layer SDPA shape counters for DS4 speed diagnostics."""
    patched = 0
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer_index, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if getattr(attn, "_moespresso_dsv4_attention_shape_stats", False):
            continue
        object.__setattr__(
            layer,
            "self_attn",
            _DeepseekV4AttentionShapeStats(attn, layer_index=layer_index),
        )
        patched += 1
    object.__setattr__(
        model,
        "_moespresso_dsv4_attention_shape_stats_layers",
        patched,
    )
    return patched


def deepseek_v4_attention_layer_stats(model) -> list[dict[str, int]]:
    """Return shape-derived DS4 SDPA counters by layer."""
    rows = []
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer_index, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if not getattr(attn, "_moespresso_dsv4_attention_shape_stats", False):
            continue
        rows.append({
            "layer": int(layer_index),
            "compress_ratio": int(getattr(attn, "compress_ratio", 0) or 0),
            "attention_sdpa_calls": int(
                attn._moespresso_dsv4_attention_sdpa_calls
            ),
            "attention_sdpa_tokens": int(
                attn._moespresso_dsv4_attention_sdpa_tokens
            ),
            "attention_sdpa_key_rows": int(
                attn._moespresso_dsv4_attention_sdpa_key_rows
            ),
            "attention_sdpa_score_elements": int(
                attn._moespresso_dsv4_attention_sdpa_score_elements
            ),
            "attention_sdpa_value_elements": int(
                attn._moespresso_dsv4_attention_sdpa_value_elements
            ),
            "attention_sdpa_max_key_rows": int(
                attn._moespresso_dsv4_attention_sdpa_max_key_rows
            ),
            # Proxied down the wrapper chain to the fused decode wrapper;
            # zero when the fused island is absent or never engaged.
            "fused_decode_attention_calls": int(
                getattr(attn, "fused_decode_calls", 0) or 0
            ),
            "fused_decode_composed_tail_calls": int(
                getattr(attn, "fused_decode_composed_tail_calls", 0) or 0
            ),
            # Decode steps this layer ran under the fixed-shape decode
            # cache contract; zero for sliding-window layers, which keep
            # no compressed pool state.
            "fixed_state_decode_layers": int(
                getattr(
                    getattr(attn, "compressor", None),
                    "fixed_state_decode_calls",
                    0,
                ) or 0
            ),
        })
    return rows


def _patch_deepseek_v4_compressor_ape_float16(model) -> int:
    """Match DS4 compressor APE storage before compressor score addition."""
    import mlx.core as mx

    patched = 0
    layers = getattr(getattr(model, "model", model), "layers", ())
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        compressors = []
        compressor = getattr(attn, "compressor", None)
        if compressor is not None:
            compressors.append(compressor)
        indexer = getattr(attn, "indexer", None)
        indexer_compressor = getattr(indexer, "compressor", None)
        if indexer_compressor is not None:
            compressors.append(indexer_compressor)
        for compressor in compressors:
            ape = getattr(compressor, "ape", None)
            if ape is None or getattr(ape, "dtype", None) == mx.float16:
                continue
            compressor.ape = ape.astype(mx.float16)
            patched += 1
    object.__setattr__(
        model,
        "_moespresso_dsv4_compressor_ape_dtype",
        "float16",
    )
    object.__setattr__(
        model,
        "_moespresso_dsv4_compressor_ape_tensors",
        patched,
    )
    return patched


def _patch_deepseek_v4_attention_fp16_qkv(model) -> bool:
    """Cast DS4 attention Q/K/V inputs to fp16 before SDPA."""
    try:
        import mlx.core as mx
        import jang_tools.dsv4.mlx_model as dsv4_model
    except ImportError:  # pragma: no cover - DS4 runtime already checks this
        return False

    current = dsv4_model.scaled_dot_product_attention
    if getattr(current, "_moespresso_dsv4_fp16_qkv_attention", False):
        patched = True
    else:
        original = getattr(
            dsv4_model,
            "_moespresso_original_scaled_dot_product_attention",
            current,
        )
        if not hasattr(dsv4_model, "_moespresso_original_scaled_dot_product_attention"):
            setattr(
                dsv4_model,
                "_moespresso_original_scaled_dot_product_attention",
                original,
            )

        def _scaled_dot_product_attention_fp16_qkv(q, k, v, *args, **kwargs):
            out_dtype = q.dtype
            stats_context = _DSV4_ATTENTION_STATS_CONTEXT.get()
            if stats_context is not None:
                stats_context._record_sdpa(q, k, v)
            if kwargs.get("sinks") is not None:
                kwargs = dict(kwargs)
                kwargs["sinks"] = kwargs["sinks"].astype(mx.float16)
            out = original(
                q.astype(mx.float16),
                k.astype(mx.float16),
                v.astype(mx.float16),
                *args,
                **kwargs,
            )
            return out.astype(out_dtype)

        setattr(
            _scaled_dot_product_attention_fp16_qkv,
            "_moespresso_dsv4_fp16_qkv_attention",
            True,
        )
        dsv4_model.scaled_dot_product_attention = _scaled_dot_product_attention_fp16_qkv
        patched = True

    object.__setattr__(
        model,
        "_moespresso_dsv4_attention_qkv_dtype",
        "float16",
    )
    return patched


# Fused rope seam engagement counts by form, exported through
# `ssd_streaming_stats` and the speed-stats count keys so served A/B arms
# can prove which composition ran. The composed count includes every
# ineligible call (foreign rope objects, out-of-range positions, and
# prefill chunks when the prefill row-cap extension is killed), so
# per-token deltas expose the engagement rate.
_ATTN_SEAM_ROPE_CALL_COUNTS = {"fused": 0, "composed": 0}


def attention_seam_rope_call_counts() -> dict[str, int]:
    """Return fused rope seam engagement counts by form."""
    return dict(_ATTN_SEAM_ROPE_CALL_COUNTS)


def _patch_deepseek_v4_attention_seam_rope(model) -> bool:
    """Route partial-rope calls through one fused dispatch.

    The composed ``_apply_partial_rope`` expands into roughly a dozen
    elementwise dispatches per call, and a ratio-4 decode layer pays it
    four times (query, KV, inverse output, indexer query) at near-zero
    traffic; the decode ledger prices the three attention rope stages at
    about 0.29 ms/layer fenced marginal. Prefill row counts serve the
    same fused dispatch by default (`MOESPRESSO_DSV4_SEAM_ROPE_PREFILL=0`
    restores the decode-only row cap); the composed prefill assemblies
    fence at 222.8 ms of removable data movement per anchor chunk and the
    extension served 0.14 s off the anchor chunk wall (14.538 against
    14.677 s, medians of three). The fused kernel is a bit-exact per-op
    transcription (`attention_seam_kernel`), so the patch changes
    dispatch structure only.

    The patch replaces the jang module-global (the fp16 SDPA precedent):
    every caller routes through the module attribute, including the
    indexer score contract and the compressor's pooled-row rope.
    Eligibility fails closed to the composed path per call; the kill
    switches are read per call so serving and tests can toggle them
    without reinstalling.
    """
    try:
        import jang_tools.dsv4.mlx_model as dsv4_model
    except ImportError:  # pragma: no cover - DS4 runtime already checks this
        return False
    from moespresso.runtime.deepseek_v4 import attention_seam_kernel

    current = dsv4_model._apply_partial_rope
    if getattr(current, "_moespresso_dsv4_attn_seam_rope", False):
        patched = True
    else:
        original = getattr(
            dsv4_model,
            "_moespresso_original_apply_partial_rope",
            current,
        )
        if not hasattr(dsv4_model, "_moespresso_original_apply_partial_rope"):
            setattr(
                dsv4_model,
                "_moespresso_original_apply_partial_rope",
                original,
            )
        rope_cls = dsv4_model.DeepseekV4RoPE

        def _apply_partial_rope_seam(
            x, rope, offset=0, inverse=False, positions=None,
        ):
            if (
                attention_seam_kernel.rope_seam_enabled()
                and type(rope) is rope_cls
                and int(rope.dims) == 64
                and attention_seam_kernel.partial_rope_eligible(
                    x, rope.inv_freq, offset=offset, positions=positions,
                )
            ):
                _ATTN_SEAM_ROPE_CALL_COUNTS["fused"] += 1
                return attention_seam_kernel.fused_partial_rope(
                    x,
                    rope.inv_freq,
                    offset=offset,
                    inverse=inverse,
                    positions=positions,
                )
            _ATTN_SEAM_ROPE_CALL_COUNTS["composed"] += 1
            return original(
                x, rope, offset=offset, inverse=inverse, positions=positions,
            )

        setattr(_apply_partial_rope_seam, "_moespresso_dsv4_attn_seam_rope", True)
        dsv4_model._apply_partial_rope = _apply_partial_rope_seam
        patched = True

    object.__setattr__(
        model,
        "_moespresso_dsv4_attn_seam_rope_installed",
        patched,
    )
    return patched


# Kill switch for the decode-shaped q8_0 affine-QMV fast path below.
_DSV4_Q8_DECODE_QMV = os.environ.get("MOESPRESSO_DSV4_Q8_DECODE_QMV", "1") != "0"

# The wire-QMV decode route on wo_b and lm_head below is unconditional at
# eligible decode shapes. At the served decode shapes the mlx-kquant QMV
# on the resident q8_0 wire bytes beats the affine QMV at lm_head (1.718
# versus 2.532 ms/call) and wo_b (0.275 versus 0.322 ms/layer), about 2.8
# ms/token across the 43 layers plus the head. The route rides the
# mlx-kquant activation contract (float32 activations pre-round to
# bfloat16 and the kernel accumulates on the bfloat16 lattice), so it is
# math-affecting at the fp32-protected q8 seams; measured drift against
# the affine route on real served wire is 0.021 max-abs at wo_b and 0.127
# at lm_head logits, and the route landed through the full math-change
# gate campaign without moving served tokens off the banded rail. wo_a
# keeps the batched affine decode form, which beats every wire
# alternative (the wire QMV has no batched-weight form), and multi-row
# prefill calls keep their current paths. MOESPRESSO_DSV4_Q8_DECODE_QMV=0
# closes the whole decode QMV family, wire route included.

_DEEPSEEK_V4_Q8_BLOCK_BYTES = 34
_DEEPSEEK_V4_Q8_GROUP = 32

# Grouped wo_a projection engagement counts by form, exported through
# `ssd_streaming_stats` and the speed-stats count keys so served A/B arms
# can prove which composition ran.
_WO_A_PROJECTION_CALL_COUNTS = {
    "batched_decode": 0,
    "gather_decode": 0,
    "loop": 0,
}

# The wo_a gather-QMV decode route below is unconditional at eligible
# decode shapes. The grouped decode projection is one activation row per
# group against that group's [rank, group_feat] weight block;
# mlx-kquant's gather_qmv_kq runs all groups as expert slots in one
# dispatch on the resident q8_0 wire bytes (fenced 0.280 versus 0.313
# ms/layer against the batched affine QMV). The kernel takes bfloat16
# activations, so the route shares the wire family's bfloat16-lattice
# contract and is math-affecting at the fp32-protected q8 seam (0.0076
# max-abs against the affine form at the served decode shape); it
# certified ahead of the reference engine with the full quality ladder
# on the woagather rail. MOESPRESSO_DSV4_Q8_DECODE_QMV=0 closes the
# whole decode QMV family, this route included, and per-call
# eligibility fails closed to the batched affine form.


def wo_a_projection_call_counts() -> dict[str, int]:
    """Return grouped wo_a projection engagement counts by form."""
    return dict(_WO_A_PROJECTION_CALL_COUNTS)


# Dense q8_0 matmul engagement counts by form (decode affine QMV, decode
# wire QMV per site, prefill dequant bridge), exported through
# `ssd_streaming_stats` so served A/B arms can prove which composition
# ran at each site.
_Q8_DENSE_MATMUL_CALL_COUNTS = {
    "decode_qmv": 0,
    "decode_wire_qmv_wo_b": 0,
    "decode_wire_qmv_lm_head": 0,
    "prefill_dequant": 0,
}


def q8_dense_matmul_call_counts() -> dict[str, int]:
    """Return dense q8_0 matmul engagement counts by form."""
    return dict(_Q8_DENSE_MATMUL_CALL_COUNTS)


# Kill switch for the fp32 seam contract on affine dense wo modules below.
# Setting MOESPRESSO_DSV4_AFFINE_WO_FP32=0 restores the stock float16 wo
# seams for A/B arms; default serving keeps the fp32 contract.
_DSV4_AFFINE_WO_FP32 = (
    os.environ.get("MOESPRESSO_DSV4_AFFINE_WO_FP32", "1") != "0"
)

# Affine dense wo fp32-seam engagement counts by module, exported through
# `ssd_streaming_stats` and the speed-stats count keys so served A/B arms
# can prove which seam contract ran. "delegated" counts kill-switch
# delegations to the stock float16 path.
_AFFINE_WO_FP32_CALL_COUNTS = {"wo_a": 0, "wo_b": 0, "delegated": 0}


def affine_wo_fp32_call_counts() -> dict[str, int]:
    """Return affine dense wo fp32-seam engagement counts by module."""
    return dict(_AFFINE_WO_FP32_CALL_COUNTS)


def _dsv4_wo_a_batched_decode_enabled() -> bool:
    """Kill switch for the single-dispatch decode form of the grouped wo_a.

    Decode-shaped (single activation row) grouped projections run one
    batched `mx.quantized_matmul` over the stacked affine group views
    instead of eight per-group QMV dispatches plus a concatenate. Each
    output element reduces the same group_feat products through the same
    per-batch QMV kernel, so the batched form is bit-identical to the
    slice loop (0 mismatched output bits per group across all 43 layers'
    served steady-state activations on real q8_0 wire at the served
    [8, 1024, 4096] geometry). Unlike the batched prefill form, the decode
    shape is dispatch-bound rather than GEMM-floor-bound: the fenced
    same-stage A/B at the seeded steady state reads loop 0.516-0.543
    versus batched 0.329-0.337 ms medians across the layer classes, and
    the whole-token alternating A/B reads 57.4 versus 52.8 ms/token.

    Default on. ``MOESPRESSO_DSV4_WO_A_BATCHED_DECODE=0`` restores the
    per-group slice loop. Ineligible calls (non-q8_0 wire, missing affine
    views, geometry mismatches, or the affine decode QMV kill switch)
    always fall back to the loop.
    """
    return os.environ.get("MOESPRESSO_DSV4_WO_A_BATCHED_DECODE", "1") != "0"


def _deepseek_v4_q8_affine_views(module, *, mx):
    """Repack a q8_0 wire tensor as MLX affine (8 bit, group 32) views.

    q8_0 stores per-32 blocks of one float16 scale ``d`` plus 32 signed int8
    quants; the identical lattice in MLX affine form is ``scales = d``,
    ``biases = -128 * d``, ``w_q = q + 128``, so `mx.quantized_matmul` on
    the repacked tensor computes the same dequantized weights with float32
    accumulation and a float32 output for float32 activations. The repack is
    built lazily on first use (module weights are placeholders until the
    package shards load) and cached on the module. Returns None for tensors
    that are not q8_0 wire shaped.
    """
    cached = getattr(module, "_moespresso_dsv4_q8_affine", None)
    if cached is not None:
        return cached
    wire = module["weight"]
    if wire.dtype != mx.uint8 or wire.ndim != 2:
        return None
    rows, bpr = (int(v) for v in wire.shape)
    if bpr <= 0 or bpr % _DEEPSEEK_V4_Q8_BLOCK_BYTES:
        return None
    blocks = wire.reshape(rows, bpr // _DEEPSEEK_V4_Q8_BLOCK_BYTES,
                          _DEEPSEEK_V4_Q8_BLOCK_BYTES)
    scales = mx.view(mx.contiguous(blocks[:, :, :2]), mx.float16).reshape(
        rows, -1)
    quants = mx.view(mx.contiguous(blocks[:, :, 2:]), mx.int8)
    unsigned = (quants.astype(mx.int16) + 128).astype(mx.uint8).reshape(
        rows, -1)
    w_q = mx.view(unsigned, mx.uint32).reshape(rows, -1)
    biases = (scales.astype(mx.float32) * -128.0).astype(mx.float16)
    mx.eval(w_q, scales, biases)
    affine = (w_q, scales, biases)
    object.__setattr__(module, "_moespresso_dsv4_q8_affine", affine)
    return affine


def _deepseek_v4_q8_affine_group_views(module, *, groups, rank, group_feat,
                                       mx):
    """Stack a module's q8_0 affine views along a leading group axis.

    Returns cached ``[groups, rank, ...]`` views of the affine repack for
    the batched decode form of the grouped output projection, or None when
    the affine views are missing or their geometry does not match the
    grouped projection contract (fail closed to the per-group slice loop).
    """
    affine = _deepseek_v4_q8_affine_views(module, mx=mx)
    if affine is None:
        return None
    w_q, scales, biases = affine
    if (
        int(w_q.shape[0]) != groups * rank
        # w_q packs four uint8 quants per uint32; scales and biases carry
        # one entry per 32-weight quantization group.
        or int(w_q.shape[1]) * 4 != group_feat
        or int(scales.shape[1]) * _DEEPSEEK_V4_Q8_GROUP != group_feat
        or int(biases.shape[1]) * _DEEPSEEK_V4_Q8_GROUP != group_feat
    ):
        return None
    cached = getattr(module, "_moespresso_dsv4_q8_affine_grouped", None)
    if (
        cached is not None
        and int(cached[0].shape[0]) == groups
        and int(cached[0].shape[1]) == rank
    ):
        return cached
    grouped = tuple(
        mx.contiguous(tensor.reshape(groups, rank, -1)) for tensor in affine)
    mx.eval(*grouped)
    object.__setattr__(module, "_moespresso_dsv4_q8_affine_grouped", grouped)
    return grouped


def _deepseek_v4_q8_wire_gather_stack(module, *, groups, rank, group_feat,
                                      mx):
    """Cached ``[groups, rank, bytes_per_row]`` wire stack plus slot ids.

    Returns the module's q8_0 wire bytes reshaped for the gather-QMV
    decode route, or None when the geometry does not match the grouped
    projection contract or the gather kernel's activation contract
    (fail closed to the batched affine form).
    """
    weight = module["weight"]
    if weight.dtype != mx.uint8 or weight.ndim != 2:
        return None
    bytes_per_row = int(weight.shape[1])
    if (
        int(weight.shape[0]) != groups * rank
        or bytes_per_row <= 0
        or bytes_per_row % _DEEPSEEK_V4_Q8_BLOCK_BYTES != 0
        # gather_qmv_kq requires the reduced dimension in whole 256-wide
        # tiles and an output width in whole simdgroup rows of 8.
        or group_feat % 256 != 0
        or rank % 8 != 0
    ):
        return None
    cached = getattr(module, "_moespresso_dsv4_q8_wire_gather", None)
    if (
        cached is not None
        and int(cached[0].shape[0]) == groups
        and int(cached[0].shape[1]) == rank
    ):
        return cached
    stack = weight.reshape(groups, rank, bytes_per_row)
    ids = mx.array([list(range(groups))], dtype=mx.uint32)
    mx.eval(stack, ids)
    cached = (stack, ids)
    object.__setattr__(module, "_moespresso_dsv4_q8_wire_gather", cached)
    return cached


def _deepseek_v4_q8_wire_decode_eligible(x, weight, site, *, mx) -> bool:
    """Fail-closed eligibility for the wire-QMV decode route.

    The route runs `kq.quantized_matmul` directly on the module's resident
    q8_0 wire bytes, so it requires a known counter site, a floating-point
    activation, and a weight tensor that is actually q8_0 wire shaped
    (uint8 rows of whole 34-byte blocks). Anything else falls through to
    the affine QMV or the dequant bridge.
    """
    if ("decode_wire_qmv_" + str(site)) not in _Q8_DENSE_MATMUL_CALL_COUNTS:
        return False
    if x.dtype not in (mx.float32, mx.bfloat16, mx.float16):
        return False
    if weight.dtype != mx.uint8 or weight.ndim != 2:
        return False
    bytes_per_row = int(weight.shape[1])
    return (
        bytes_per_row > 0
        and bytes_per_row % _DEEPSEEK_V4_Q8_BLOCK_BYTES == 0
    )


def _kquant_matmul_ds4_fp32(x, weight, scales, kquant_type: str, bias=None, *,
                            mx, kq, affine=None, wire_decode_site=None):
    if kquant_type == "q8_0":
        rows = 1
        for dim in x.shape[:-1]:
            rows *= int(dim)
        if (
            rows == 1
            and wire_decode_site is not None
            and _DSV4_Q8_DECODE_QMV
            and _deepseek_v4_q8_wire_decode_eligible(
                x, weight, wire_decode_site, mx=mx)
        ):
            # Decode-shaped wire route: one activation row through the
            # mlx-kquant QMV on the resident q8_0 wire bytes. The kernel
            # pre-rounds a float32 activation to bfloat16 and accumulates
            # on the bfloat16 lattice, so this arm drifts from the affine
            # QMV (details in the route note above); the final cast
            # restores the float32 seam contract for downstream consumers.
            _Q8_DENSE_MATMUL_CALL_COUNTS[
                "decode_wire_qmv_" + wire_decode_site] += 1
            y = kq.quantized_matmul(
                x.astype(mx.float32),
                weight,
                scales,
                kquant_type,
                transpose=True,
            ).astype(mx.float32)
        elif (rows == 1 and _DSV4_Q8_DECODE_QMV) and (
            # `affine` may be a zero-arg callable so the repack (a full
            # extra copy of the wire tensor) only materializes when this
            # branch actually consumes it; the wire and dequant branches
            # never resolve it.
            affine := (affine() if callable(affine) else affine)
        ) is not None:
            # Decode-shaped fast path: one activation row through the MLX
            # affine QMV on the repacked q8_0 lattice. The kernel
            # dequantizes weight groups in registers and accumulates in
            # float32 with a float32 output, so the only difference from
            # the dequantize + float32 matmul below is accumulation order
            # (measured 4e-7 rel on the served wo shapes), without
            # materializing hundreds of MB of float32 weights per token.
            # Multi-row calls stay on the dequant path: the affine QMM
            # tile path has a different (float16 staging) contract at
            # bulk shapes and prefill amortizes the dequant.
            _Q8_DENSE_MATMUL_CALL_COUNTS["decode_qmv"] += 1
            w_q, affine_scales, affine_biases = affine
            y = mx.quantized_matmul(
                x.astype(mx.float32),
                w_q,
                scales=affine_scales,
                biases=affine_biases,
                transpose=True,
                group_size=_DEEPSEEK_V4_Q8_GROUP,
                bits=8,
            )
        else:
            if rows > 1:
                _Q8_DENSE_MATMUL_CALL_COUNTS["prefill_dequant"] += 1
            weight = kq.dequantize(
                weight,
                scales,
                kquant_type,
                dtype=mx.float32,
            )
            y = mx.matmul(x.astype(mx.float32), weight.T)
    else:
        y = kq.quantized_matmul(
            x,
            weight,
            scales,
            kquant_type,
            transpose=True,
        )
    if bias is not None:
        y = y + bias
    return y


def _patch_deepseek_v4_kquant_grouped_output_projection(model) -> int:
    """Teach DS4 grouped attention output projection about KQuantLinear."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx_kquant as kq
    except ImportError:  # pragma: no cover - guarded by kquant runtime install
        return 0

    class _DS4Fp32KQuantLinear(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
            self.mode = getattr(original, "mode", None)
            self.kquant_type = getattr(original, "kquant_type", None)
            self.group_size = getattr(original, "group_size", None)
            self.bits = getattr(original, "bits", None)
            self.biases = getattr(original, "biases", None)
            self.freeze()

        def __call__(self, x):
            affine = None
            if self.original.kquant_type == "q8_0":
                # A thunk defers the affine views. The wire route serves decode rows
                # and prefill takes the dequant bridge, so the affine
                # repack only materializes if a fallback branch asks.
                original = self.original
                affine = lambda: _deepseek_v4_q8_affine_views(  # noqa: E731
                    original, mx=mx)
            return _kquant_matmul_ds4_fp32(
                x,
                self.original["weight"],
                self.original["scales"],
                self.original.kquant_type,
                self.original["bias"] if "bias" in self.original else None,
                mx=mx,
                kq=kq,
                affine=affine,
                wire_decode_site="wo_b",
            )

    def _kquant_grouped_output_projection(self, out):
        wo_a = self.wo_a
        if getattr(wo_a, "mode", None) != "kquant":
            raise DeepseekV4RuntimeLoadError(
                "K-quant grouped output projection was installed on a non-kquant wo_a")

        bsz, length = out.shape[:2]
        groups = int(self.o_groups)
        rank = int(self.o_lora_rank)
        group_feat = (self.n_heads * self.head_dim) // groups
        weight = wo_a["weight"]
        if int(weight.shape[0]) != groups * rank:
            raise DeepseekV4RuntimeLoadError(
                "K-quant wo_a rows do not match DS4 grouped projection geometry: "
                f"rows={weight.shape[0]} groups={groups} rank={rank}")

        grouped = out.reshape(bsz, length, groups, group_feat)
        rows = int(bsz) * int(length)
        if (
            rows == 1
            and wo_a.kquant_type == "q8_0"
            and _DSV4_Q8_DECODE_QMV
            and hasattr(kq, "gather_qmv_kq")
            and grouped.dtype in (mx.float32, mx.bfloat16, mx.float16)
        ):
            gather = _deepseek_v4_q8_wire_gather_stack(
                wo_a, groups=groups, rank=rank, group_feat=group_feat,
                mx=mx)
            if gather is not None:
                # Gather decode form: all groups as expert slots in one
                # gather-QMV dispatch on the resident q8_0 wire bytes.
                # The kernel consumes bfloat16 activations and
                # accumulates on that lattice, so this arm drifts from
                # the affine form (details on the kill switch above);
                # the final cast restores the float32 seam contract.
                # The [1, groups, rank] result matches the loop's
                # last-axis concatenation order.
                _WO_A_PROJECTION_CALL_COUNTS["gather_decode"] += 1
                stack, slot_ids = gather
                y = kq.gather_qmv_kq(
                    grouped.reshape(1, groups, group_feat).astype(
                        mx.bfloat16),
                    stack,
                    wo_a.kquant_type,
                    slot_ids,
                ).astype(mx.float32)
                if "bias" in wo_a:
                    y = y + wo_a["bias"].reshape(groups, rank)
                return y.reshape(bsz, length, groups * rank)
        if (
            rows == 1
            and wo_a.kquant_type == "q8_0"
            and _DSV4_Q8_DECODE_QMV
            and _dsv4_wo_a_batched_decode_enabled()
        ):
            group_views = _deepseek_v4_q8_affine_group_views(
                wo_a, groups=groups, rank=rank, group_feat=group_feat, mx=mx)
            if group_views is not None:
                # Batched decode form: one QMV dispatch over the stacked
                # affine group views instead of eight per-group slices plus
                # a concatenate. The per-batch QMV reduces each output
                # element in the same order as the sliced dispatch, so the
                # result is bit-identical to the loop below; only the
                # dispatch count changes. The reshape of the [groups, 1,
                # rank] result matches the loop's last-axis concatenation.
                _WO_A_PROJECTION_CALL_COUNTS["batched_decode"] += 1
                w_qg, group_scales, group_biases = group_views
                y = mx.quantized_matmul(
                    grouped.astype(mx.float32).reshape(groups, 1, group_feat),
                    w_qg,
                    scales=group_scales,
                    biases=group_biases,
                    transpose=True,
                    group_size=_DEEPSEEK_V4_Q8_GROUP,
                    bits=8,
                )
                if "bias" in wo_a:
                    y = y + wo_a["bias"].reshape(groups, 1, rank)
                return y.reshape(bsz, length, groups * rank)
        _WO_A_PROJECTION_CALL_COUNTS["loop"] += 1

        def _group_affine_thunk(row_start, row_end):
            # A thunk defers the affine views. On the default routes (gather decode,
            # dequant-bridge prefill) the loop's affine slices are never
            # consumed, so the full repack only materializes if a
            # fallback branch asks; the module-level cache makes the
            # per-group builds free after the first.
            def build():
                affine = _deepseek_v4_q8_affine_views(wo_a, mx=mx)
                if affine is None:
                    return None
                return tuple(
                    tensor[row_start:row_end] for tensor in affine)
            return build

        pieces = []
        for group in range(groups):
            row_start = group * rank
            row_end = row_start + rank
            group_affine = None
            if wo_a.kquant_type == "q8_0":
                group_affine = _group_affine_thunk(row_start, row_end)
            pieces.append(
                _kquant_matmul_ds4_fp32(
                    grouped[:, :, group, :],
                    weight[row_start:row_end],
                    wo_a["scales"],
                    wo_a.kquant_type,
                    wo_a["bias"][row_start:row_end] if "bias" in wo_a else None,
                    mx=mx,
                    kq=kq,
                    affine=group_affine,
                )
            )
        out = mx.concatenate(pieces, axis=-1)
        return out

    patched = 0
    layers = getattr(getattr(model, "model", None), "layers", ())
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(getattr(attn, "wo_a", None), "mode", None) != "kquant":
            continue
        object.__setattr__(
            attn,
            "_grouped_output_projection",
            MethodType(_kquant_grouped_output_projection, attn),
        )
        wo_b = getattr(attn, "wo_b", None)
        if (
            getattr(wo_b, "mode", None) == "kquant"
            and not getattr(wo_b, "_moespresso_dsv4_q8_fp32_linear", False)
        ):
            wrapped = _DS4Fp32KQuantLinear(wo_b)
            object.__setattr__(wrapped, "_moespresso_dsv4_q8_fp32_linear", True)
            object.__setattr__(attn, "wo_b", wrapped)
        patched += 1
    object.__setattr__(
        model, "_moespresso_dsv4_kquant_grouped_output_projection_layers", patched)
    return patched


def _patch_deepseek_v4_affine_wo_fp32(model) -> int:
    """Enforce the fp32 seam contract on affine dense wo modules.

    Served attention runs SDPA in float16, so the stock QuantizedLinear wo
    path computes the grouped wo_a QMM and the wo_b matmul at float16
    activations with float16 outputs. A float16 output representation at a
    quantized dense seam is a recorded quality failure (0.26 max-abs
    single-step logit drift, two orders above accepted f32-rounding drifts),
    and the kquant bridge (`_kquant_matmul_ds4_fp32`) deliberately casts the
    same seams to float32 end to end. This patch gives affine-quantized wo
    modules the matching contract: the activation is cast to float32 before
    `mx.quantized_matmul`, so the kernel dequantizes weight groups in
    registers and accumulates in float32 with a float32 output. No extra
    weight buffers are built; the module's packed affine tensors serve both
    forms, keeping the single-copy memory story of an affine wo recipe.

    Eligibility is fail-closed per module: only `nn.QuantizedLinear` wo_a
    and wo_b in affine mode are patched, and wo_a additionally must match
    the grouped projection geometry. K-quant wo modules are not
    QuantizedLinear and keep their existing bridge; plain and mx-mode
    modules keep the stock path. `MOESPRESSO_DSV4_AFFINE_WO_FP32=0` is the
    kill switch: patched modules then delegate per call to the stock
    float16 path (counted as "delegated") so A/B arms can prove which seam
    contract ran.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:  # pragma: no cover - guarded by the runtime installation
        return 0

    def _eligible(module):
        return (
            isinstance(module, nn.QuantizedLinear)
            and getattr(module, "mode", "affine") == "affine"
            and getattr(module, "biases", None) is not None
        )

    class _DS4Fp32AffineLinear(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
            self.mode = getattr(original, "mode", "affine")
            self.group_size = original.group_size
            self.bits = original.bits
            self.freeze()

        def __call__(self, x):
            original = self.original
            if not _DSV4_AFFINE_WO_FP32:
                _AFFINE_WO_FP32_CALL_COUNTS["delegated"] += 1
                return original(x)
            _AFFINE_WO_FP32_CALL_COUNTS["wo_b"] += 1
            y = mx.quantized_matmul(
                x.astype(mx.float32),
                original["weight"],
                scales=original["scales"],
                biases=original["biases"],
                transpose=True,
                group_size=original.group_size,
                bits=original.bits,
                mode="affine",
            )
            if "bias" in original:
                y = y + original["bias"].astype(mx.float32)
            return y

    def _make_projection(stock):
        def _affine_fp32_grouped_output_projection(self, out):
            if not _DSV4_AFFINE_WO_FP32:
                _AFFINE_WO_FP32_CALL_COUNTS["delegated"] += 1
                return stock(out)
            wo_a = self.wo_a
            bsz, length = out.shape[:2]
            groups = int(self.o_groups)
            rank = int(self.o_lora_rank)
            group_feat = (self.n_heads * self.head_dim) // groups
            _AFFINE_WO_FP32_CALL_COUNTS["wo_a"] += 1
            # The stock grouped QMM at the float32 activation dtype: same
            # batched dispatch over the packed group views, float32
            # accumulation and output at the seam.
            x = out.reshape(bsz, length, groups, group_feat)
            x = x.astype(mx.float32).transpose(2, 0, 1, 3)
            weight = wo_a["weight"].reshape(groups, rank, -1)[:, None]
            scales = wo_a["scales"].reshape(groups, rank, -1)[:, None]
            biases = wo_a["biases"].reshape(groups, rank, -1)[:, None]
            y = mx.quantized_matmul(
                x,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=wo_a.group_size,
                bits=wo_a.bits,
                mode="affine",
            )
            y = y.transpose(1, 2, 0, 3).reshape(bsz, length, groups * rank)
            if "bias" in wo_a:
                y = y + wo_a["bias"].astype(mx.float32)
            return y
        return _affine_fp32_grouped_output_projection

    patched = 0
    layers = getattr(getattr(model, "model", None), "layers", ())
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "_moespresso_dsv4_affine_wo_fp32", False):
            continue
        layer_patched = False
        wo_a = getattr(attn, "wo_a", None)
        groups = getattr(attn, "o_groups", None)
        rank = getattr(attn, "o_lora_rank", None)
        stock = getattr(attn, "_grouped_output_projection", None)
        if (
            _eligible(wo_a)
            and callable(stock)
            and groups is not None
            and rank is not None
            and int(wo_a["weight"].shape[0]) == int(groups) * int(rank)
        ):
            object.__setattr__(
                attn,
                "_grouped_output_projection",
                MethodType(_make_projection(stock), attn),
            )
            layer_patched = True
        wo_b = getattr(attn, "wo_b", None)
        if _eligible(wo_b):
            wrapped = _DS4Fp32AffineLinear(wo_b)
            object.__setattr__(attn, "wo_b", wrapped)
            layer_patched = True
        if layer_patched:
            object.__setattr__(attn, "_moespresso_dsv4_affine_wo_fp32", True)
            patched += 1
    object.__setattr__(
        model, "_moespresso_dsv4_affine_wo_fp32_layers", patched)
    return patched


def _patch_deepseek_v4_kquant_lm_head(model) -> bool:
    """Teach DS4's FP32 logits path about a K-quant lm_head."""
    lm_head = getattr(model, "lm_head", None)
    if getattr(lm_head, "mode", None) != "kquant":
        object.__setattr__(model, "_moespresso_dsv4_kquant_lm_head", False)
        return False
    try:
        import mlx.core as mx
        import mlx_kquant as kq
    except ImportError:  # pragma: no cover - guarded by kquant runtime install
        return False

    cls = type(model)
    original_call = getattr(cls, "_moespresso_original_call", cls.__call__)
    if not hasattr(cls, "_moespresso_original_call"):
        setattr(cls, "_moespresso_original_call", original_call)

    def _kquant_lm_head_call(self, input_ids, cache=None, mask=None):
        if getattr(getattr(self, "lm_head", None), "mode", None) != "kquant":
            return original_call(self, input_ids, cache=cache, mask=mask)
        h = self.model(input_ids, cache=cache, mask=mask)
        if cache is not None and int(h.shape[1]) > 1:
            # Generation prefill: the sampler consumes only the newest
            # position's logits (the generate loop slices [:, -1]), while the
            # scorer paths (Q1/Q2 teacher forcing) call without a cache and
            # keep every row. Slicing before the head skips a [L, vocab] fp32
            # matmul and the full lm_head dequantization per prefill chunk;
            # the surviving row takes the single-row affine QMV path.
            h = h[:, -1:, :]
        affine = None
        if self.lm_head.kquant_type == "q8_0":
            # A thunk defers the affine views. The wire route serves the decode row and
            # the scorer path takes the dequant bridge, so the affine
            # repack only materializes if a fallback branch asks.
            lm_head = self.lm_head
            affine = lambda: _deepseek_v4_q8_affine_views(  # noqa: E731
                lm_head, mx=mx)
        return _kquant_matmul_ds4_fp32(
            h.astype(mx.float32),
            self.lm_head["weight"],
            self.lm_head["scales"],
            self.lm_head.kquant_type,
            mx=mx,
            kq=kq,
            affine=affine,
            wire_decode_site="lm_head",
        )

    setattr(cls, "__call__", _kquant_lm_head_call)
    object.__setattr__(model, "_moespresso_dsv4_kquant_lm_head", True)
    return True


def _load_deepseek_v4_regular_weights(
    model,
    package_dir: Path,
    *,
    load_shard_fn: Callable[[Path], dict] | None = None,
) -> tuple[int, int]:
    """Load every non-bundle DS4 package tensor through the graph sanitizer."""
    package_dir = Path(package_dir)
    shards = sorted(package_dir.glob("model-*.safetensors"))
    if not shards:
        raise DeepseekV4RuntimeLoadError(
            f"DeepSeek V4 package has no safetensors shards: {package_dir}")

    if load_shard_fn is None:
        import mlx.core as mx

        def load_shard_fn(path: Path) -> dict:
            return mx.load(str(path))

    loaded = 0
    skipped_bundles = 0
    for shard in shards:
        weights = load_shard_fn(shard)
        regular = {}
        for key, value in weights.items():
            if _is_deepseek_v4_bundle_key(key):
                skipped_bundles += 1
            else:
                regular[key] = value
        if regular:
            if hasattr(model, "sanitize"):
                regular = model.sanitize(regular)
            if regular:
                model.load_weights(list(regular.items()), strict=False)
                loaded += len(regular)
        del weights, regular
        gc.collect()
    return loaded, skipped_bundles


def _validate_deepseek_v4_router_gate_dtypes(model) -> int:
    """Reject raw integer router-gate storage before it reaches routing math."""
    bad = []
    checked = 0
    layers = getattr(getattr(model, "model", None), "layers", ())
    for layer_idx, layer in enumerate(layers):
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        weight = getattr(gate, "weight", None)
        if weight is None:
            continue
        checked += 1
        dtype = getattr(weight, "dtype", None)
        if "float" not in str(dtype):
            bad.append((layer_idx, str(dtype)))
    if bad:
        sample = ", ".join(f"layer {idx}: {dtype}" for idx, dtype in bad[:4])
        raise DeepseekV4RuntimeLoadError(
            "DeepSeek V4 router gate weights must load as floating tensors; "
            f"got {sample}. Rebuild the package with fp16 router-gate passthrough.")
    object.__setattr__(model, "_moespresso_dsv4_router_gates_checked", checked)
    return checked


def _read_jang_config(package_dir: Path) -> dict:
    path = Path(package_dir) / "jang_config.json"
    if not path.is_file():
        return {}
    with open(path) as f:
        return json.load(f)


def _deepseek_v4_skeleton_config(model_config: dict) -> dict:
    """Config view for building a DS4 skeleton with mlx_lm's generic loader."""
    out = dict(model_config)
    quantization = out.get("quantization")
    if isinstance(quantization, dict):
        out["quantization"] = {
            key: value
            for key, value in quantization.items()
            if not (
                isinstance(key, str)
                and key.startswith("model.layers.")
                and key.endswith(".mlp.gate")
            )
        }
    return out


def _load_empty_deepseek_v4_skeleton(
    package_dir: Path,
    *,
    model_config: dict,
    lazy: bool = True,
    strict: bool = False,
    load_model_fn: Callable[..., tuple[Any, Any]] | None = None,
) -> tuple[Any, Any]:
    """Build the quantized graph without reading package weight shards."""
    if load_model_fn is None:
        import jang_tools.dsv4  # noqa: F401
        from mlx_lm.utils import load_model as load_model_fn

    with tempfile.TemporaryDirectory(prefix="moespresso-dsv4-skeleton-") as tmp:
        tmp_dir = Path(tmp)
        skeleton_config = _deepseek_v4_skeleton_config(model_config)
        (tmp_dir / "config.json").write_text(json.dumps(skeleton_config))
        return load_model_fn(
            tmp_dir,
            lazy=lazy,
            strict=False,
            model_config=skeleton_config,
        )


_DSV4_PREFILL_SINGLE_CHUNK_ENV = "MOESPRESSO_DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS"
_DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS = 4096


def _dsv4_prefill_single_chunk_max_tokens() -> int:
    """Largest prompt served as one prefill chunk, in tokens.

    The banded prefill attention route requires cache offset 0, so a
    prompt split into multiple chunks drops the 22 banded layers to a
    composed SDPA fallback on every chunk past the first; head dim 512
    has no fused MLX kernel there, and the fallback materializes score
    tensors of 1.2 to 2.1 GB per layer per chunk. Serving a deeper
    prompt as one chunk keeps every layer on its fused route and
    measured 1.34x served TTFT at 7698 tokens (37.5 s chunked to 28.1 s
    single-chunk, medians, one process per arm) with a request transient
    peak of 15.7 GiB above the resident set against 11.4 GiB chunked on
    a 128 GB host.

    The default stays at 4096 because the two chunk geometries are
    different math: at 7698 tokens the chunked and single-chunk arms are
    each bit-reproducible across processes and fork from each other at
    token 35 of a 56-token greedy continuation. Chunk size moves both
    the attention route (banded mma against the composed fallback) and
    the ratio-4 selection engagement, so raising the default is a
    math-affecting change that needs the full quality adjudication
    (teacher-forced NLL, task gates, and a depth rail) before it can
    ship. ``MOESPRESSO_DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS``
    overrides the cap for explicit re-pricing runs; ``0`` disables the
    single-chunk policy at every depth.
    """
    return int(os.environ.get(
        _DSV4_PREFILL_SINGLE_CHUNK_ENV,
        str(_DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS),
    ))


_DSV4_WIRED_PREWARM_ENV = "MOESPRESSO_DSV4_WIRED_PREWARM"


def _prewarm_wired_limit(model, *, wired_limit_fn=None, streams=None) -> float | None:
    """Enter and exit the generation wired-limit context once at load.

    The first effective wired-limit entry after a DS4 package load wires
    the resident set; on the recorded ~72 GiB resident set it measures
    2.7 to 2.9 s, while every later entry costs under 1 ms. The serve
    path enters the context on every request, so without this warm the
    whole wiring cost lands inside the first request's TTFT. Entering
    once here moves the cost into load time; the load takes
    correspondingly longer and the elapsed seconds are recorded on the
    model and printed.

    Raising the wired limit wires nothing until the process has
    submitted its first GPU command; the load path allocates the
    resident set without submitting any, so a bare entry at load is a
    no-op and the cost stays in the first request. The warm therefore
    evaluates a one-element sentinel first (scheduling-only, no model
    math) and then enters the context. Returns the elapsed entry
    seconds, or None when disabled or when mlx_lm is absent.
    ``MOESPRESSO_DSV4_WIRED_PREWARM=0`` is the kill switch.
    """
    if os.environ.get(_DSV4_WIRED_PREWARM_ENV, "1") == "0":
        return None
    if wired_limit_fn is None:
        try:
            import mlx.core as mx
            from mlx_lm.generate import generation_stream, wired_limit
        except ImportError:  # pragma: no cover - loader already needs mlx_lm
            return None
        mx.eval(mx.array([1.0]) + 1.0)
        mx.synchronize()
        wired_limit_fn = wired_limit
        streams = [generation_stream]
    import time

    start = time.perf_counter()
    with wired_limit_fn(model, streams):
        pass
    elapsed = time.perf_counter() - start
    object.__setattr__(model, "_moespresso_dsv4_wired_prewarm_seconds", elapsed)
    return elapsed


def load_deepseek_v4_package_model(
    manifest: dict,
    package_dir: Path,
    *,
    load_config_fn: Callable[[Path], dict] | None = None,
    load_skeleton_fn: Callable[..., tuple[Any, Any]] | None = None,
    load_tokenizer_fn: Callable[..., Any] | None = None,
    load_shard_fn: Callable[[Path], dict] | None = None,
    expert_index_fn: Callable[[Path], Any] | None = None,
    install_bundles_fn: Callable[..., int] | None = None,
    wrap_switchglus_fn: Callable[..., int] | None = None,
    install_kquant_modules_fn: Callable[[Any, dict[str, str]], int] | None = None,
    apply_tensor_map_fn: Callable[[Any, dict], None] | None = None,
    read_jang_config_fn: Callable[[Path], dict] = _read_jang_config,
    capacity_per_layer: int | None = None,
    capacity_overrides: Mapping[int, int] | None = None,
    eviction_policy: str = "lfu",
) -> tuple[Any, Any]:
    """Load a MoEspresso DS4 package into the JANG DS4 graph."""
    architecture = manifest.get("architecture") or {}
    if architecture.get("family") != "deepseek_v4_flash":
        raise DeepseekV4RuntimeLoadError("manifest is not a DeepSeek V4 package")

    package_dir = Path(package_dir)
    if load_config_fn is None or load_skeleton_fn is None or load_tokenizer_fn is None:
        try:
            import jang_tools.dsv4  # noqa: F401
            from mlx_lm.utils import load_config, load_tokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DeepseekV4RuntimeLoadError(
                "DeepSeek V4 runtime requires jang_tools.dsv4 and mlx_lm") from exc
        load_config_fn = load_config_fn or load_config
        load_skeleton_fn = load_skeleton_fn or _load_empty_deepseek_v4_skeleton
        load_tokenizer_fn = load_tokenizer_fn or load_tokenizer

    default_pooled_install = install_bundles_fn is None
    if expert_index_fn is None:
        from moespresso.runtime.build import (
            _apply_tensor_map,
            _mixed_gate_up_layers,
            _routed_expert_index,
        )
        expert_index_fn = expert_index_fn or _routed_expert_index
        apply_tensor_map_fn = apply_tensor_map_fn or _apply_tensor_map
    else:
        from moespresso.runtime.build import _mixed_gate_up_layers

    if install_bundles_fn is None:
        def install_bundles_fn(model_arg, package_dir_arg, index_arg, *, seed):
            return _install_deepseek_v4_pooled_bundles(
                model_arg,
                package_dir_arg,
                index_arg,
                seed=seed,
                capacity_per_layer=capacity_per_layer,
                capacity_overrides=capacity_overrides,
                eviction_policy=eviction_policy,
            )
    if wrap_switchglus_fn is None:
        if default_pooled_install:
            def wrap_switchglus_fn(_model_arg, *, required_mixed_layers):
                del required_mixed_layers
                return 0
        else:
            from moespresso.runtime.build import _wrap_mixed_bit_switchglus
            wrap_switchglus_fn = _wrap_mixed_bit_switchglus

    if apply_tensor_map_fn is None:
        from moespresso.runtime.build import _apply_tensor_map
        apply_tensor_map_fn = _apply_tensor_map

    model_config = load_config_fn(package_dir)
    if model_config.get("model_type") != "deepseek_v4":
        raise DeepseekV4RuntimeLoadError(
            "DeepSeek V4 runtime requires config.json model_type='deepseek_v4'")
    model, _model_config = load_skeleton_fn(
        package_dir, lazy=True, strict=False, model_config=model_config)
    if "kquant_dequant" in set(manifest.get("required_ops", [])):
        from moespresso.runtime.kquant_install import install_manifest_kquant_modules

        install_manifest_kquant_modules(
            model,
            manifest,
            installer=install_kquant_modules_fn,
        )
        _patch_deepseek_v4_kquant_grouped_output_projection(model)
        _patch_deepseek_v4_kquant_lm_head(model)
    _patch_deepseek_v4_affine_wo_fp32(model)
    _patch_deepseek_v4_hc_post_float32(model)
    _patch_deepseek_v4_hc_fused(model)
    _patch_deepseek_v4_required_attention_cache(model)
    _patch_deepseek_v4_attention_fp16_qkv(model)
    _patch_deepseek_v4_attention_seam_rope(model)

    loaded_regular, skipped_bundles = _load_deepseek_v4_regular_weights(
        model, package_dir, load_shard_fn=load_shard_fn)
    if loaded_regular == 0:
        raise DeepseekV4RuntimeLoadError("DeepSeek V4 package loaded no regular tensors")
    _validate_deepseek_v4_router_gate_dtypes(model)
    _patch_deepseek_v4_compressor_ape_float16(model)
    _patch_deepseek_v4_attention_compressor_fp8_kv(model)
    _patch_deepseek_v4_indexer_score_contract(model)
    _patch_deepseek_v4_router_gate_trims(model)
    _patch_deepseek_v4_ratio4_decode_fused_attention(model)
    _patch_deepseek_v4_ratio4_prefill_fast_path(model)
    _patch_deepseek_v4_banded_prefill_attention(model)
    _patch_deepseek_v4_attention_shape_stats(model)
    if _manifest_requires_routed_bundles(manifest) and skipped_bundles == 0:
        raise DeepseekV4RuntimeLoadError(
            "DeepSeek V4 manifest declares routed experts but no expert bundle was found")

    jang_cfg = read_jang_config_fn(package_dir)
    tensor_map = jang_cfg.get("quantization", {}).get("tensor_map", {})
    if tensor_map:
        apply_tensor_map_fn(model, tensor_map)

    index = expert_index_fn(package_dir)
    if index is None and _manifest_requires_routed_bundles(manifest):
        raise DeepseekV4RuntimeLoadError(
            "DeepSeek V4 manifest declares routed experts but no expert index was found")
    if index is not None:
        seed = int(jang_cfg.get("mxtq_seed", 42))
        install_bundles_fn(model, package_dir, index, seed=seed)
        if default_pooled_install:
            from moespresso.runtime.ssd_streaming_build import seed_expert_residency

            object.__setattr__(
                model,
                "_moespresso_ssd_hotlist",
                seed_expert_residency(model, package_dir),
            )
        wrap_switchglus_fn(model, required_mixed_layers=_mixed_gate_up_layers(index))

    tokenizer = load_tokenizer_fn(package_dir)
    model._moespresso_dsv4_regular_tensors_loaded = loaded_regular
    model._moespresso_dsv4_bundles_seen = skipped_bundles
    # One prefill chunk up to the cap; past it the serve seam threads no
    # step size and mlx_lm falls back to 2048-token chunks. The cap and
    # its rationale live in _dsv4_prefill_single_chunk_max_tokens.
    single_chunk_cap = _dsv4_prefill_single_chunk_max_tokens()
    model._moespresso_prefill_step_size = single_chunk_cap
    model._moespresso_prefill_step_size_max_prompt_tokens = single_chunk_cap
    prewarm_seconds = _prewarm_wired_limit(model)
    if prewarm_seconds is not None:
        print(
            f"[serve] wired-limit prewarm {prewarm_seconds:.2f}s at load; "
            "the first request no longer pays it inside TTFT",
            flush=True,
        )
    return model, tokenizer
