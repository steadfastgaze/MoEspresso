"""Manual one-layer DS4 decoder-stage speed replay.

This speed diagnostic loads the served DS4 graph from a real package, runs one
decoder layer over synthetic activations, and
forces evaluation at the main stage boundaries. The point is to explain served
``index_sync_seconds`` waits with the same pre-MoE graph stages that feed router
index export, without running full-model generation for every hypothesis.
It carries no quality claim and must be paired with the model-specific gates.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from moespresso.runtime.deepseek_v4.ratio4_attention_replay import _StageProfile, _pool_rows


def _to_jsonable_shape(value: Any) -> list[int]:
    return [int(v) for v in getattr(value, "shape", ())]


def _make_synthetic_layer_input(mx: Any, *, args: Any, tokens: int, dtype: Any) -> Any:
    hidden = int(getattr(args, "hidden_size", 4096))
    hc_mult = int(getattr(args, "hc_mult", 4))
    return mx.random.normal((1, int(tokens), hc_mult, hidden)).astype(dtype)


def _make_input_ids(mx: Any, *, args: Any, tokens: int) -> Any:
    vocab_size = int(getattr(args, "vocab_size", 1)) or 1
    return (mx.arange(int(tokens), dtype=mx.int32) % vocab_size).reshape(1, int(tokens))


def _make_layer_cache(model: Any, *, layer: int) -> Any:
    caches = model.make_cache()
    if layer < 0 or layer >= len(caches):
        raise ValueError(f"layer {layer} is out of range for cache length {len(caches)}")
    return caches[int(layer)]


def _seed_compressed_pool(
    *,
    mx: Any,
    cache: Any,
    attn: Any,
    pooled_rows: int,
    dtype: Any,
    prime_indexer_qat: bool,
) -> None:
    if cache is None or int(pooled_rows) <= 0:
        return
    if not hasattr(cache, "compressor_state"):
        raise ValueError("--pooled-rows requires a DeepseekV4Cache-style layer cache")
    head_dim = int(getattr(attn, "head_dim", 512))
    compressor = mx.random.normal((1, int(pooled_rows), head_dim)).astype(dtype)
    cache.compressor_state["pooled"] = compressor
    values = [compressor]

    indexer = getattr(attn, "indexer", None)
    if indexer is not None and hasattr(cache, "indexer_state"):
        index_head_dim = int(getattr(indexer, "head_dim", 128))
        index_pooled = mx.random.normal((1, int(pooled_rows), index_head_dim)).astype(dtype)
        cache.indexer_state["pooled"] = index_pooled
        values.append(index_pooled)
        if prime_indexer_qat:
            from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

            qat = _dsv4_indexer_qat(mx, index_pooled)
            cache.indexer_state["pooled_qat"] = qat
            cache.indexer_state["pooled_qat_rows"] = int(pooled_rows)
            values.append(qat)
    mx.eval(*values)


def _make_layer_mask(
    *,
    mx: Any,
    model: Any,
    cache: Any,
    x: Any,
    tokens: int,
) -> Any:
    if int(tokens) <= 1:
        return None
    from mlx_lm.models.base import create_attention_mask

    mask_input = x[:, :, 0, :]
    mask = create_attention_mask(
        mask_input,
        cache,
        window_size=int(getattr(model.args, "sliding_window", 128)),
        return_array=True,
    )
    mx.eval(mask)
    return mask


def _run_layer_until_router(
    *,
    mx: Any,
    layer_mod: Any,
    x: Any,
    mask: Any,
    cache: Any,
    input_ids: Any,
    profile: _StageProfile,
    include_moe: bool,
) -> dict[str, Any]:
    residual = x
    x, attn_post, attn_comb = profile.record(
        "hc_attn_pre",
        lambda: layer_mod._hc_pre(
            x,
            layer_mod.hc_attn_fn,
            layer_mod.hc_attn_scale,
            layer_mod.hc_attn_base,
        ),
    )
    x = profile.record("input_layernorm", lambda: layer_mod.input_layernorm(x))
    x = profile.record("self_attn", lambda: layer_mod.self_attn(x, mask=mask, cache=cache))
    x = profile.record(
        "hc_attn_post",
        lambda: layer_mod._hc_post(x, residual, attn_post, attn_comb),
    )

    residual = x
    x, ffn_post, ffn_comb = profile.record(
        "hc_ffn_pre",
        lambda: layer_mod._hc_pre(
            x,
            layer_mod.hc_ffn_fn,
            layer_mod.hc_ffn_scale,
            layer_mod.hc_ffn_base,
        ),
    )
    x = profile.record(
        "post_attention_layernorm",
        lambda: layer_mod.post_attention_layernorm(x),
    )
    inds, scores = profile.record(
        "router_gate",
        lambda: layer_mod.mlp.gate(x, input_ids=input_ids),
    )
    inds = profile.record("router_indices_cast", lambda: inds.astype(mx.uint32))
    if not include_moe:
        return {
            "stopped_after_router": True,
            "router_indices_shape": _to_jsonable_shape(inds),
            "router_scores_shape": _to_jsonable_shape(scores),
            "output_shape": None,
        }

    # Replay the same served MoE contract explicitly so router timing stays
    # visible instead of being hidden inside layer_mod.mlp(...).
    y = profile.record("routed_switch_mlp", lambda: layer_mod.mlp.switch_mlp(x, inds))
    y = profile.record(
        "route_weighted_sum",
        lambda: (y * scores[..., None]).sum(axis=-2).astype(y.dtype).reshape(x.shape),
    )
    shared = profile.record("shared_experts", lambda: layer_mod.mlp.shared_experts(x))
    x = profile.record("moe_add_shared", lambda: y + shared)
    x = profile.record(
        "hc_ffn_post",
        lambda: layer_mod._hc_post(x, residual, ffn_post, ffn_comb),
    )
    return {
        "stopped_after_router": False,
        "router_indices_shape": _to_jsonable_shape(inds),
        "router_scores_shape": _to_jsonable_shape(scores),
        "output_shape": _to_jsonable_shape(x),
    }


def _time_layer_case(
    *,
    mx: Any,
    model: Any,
    layer: int,
    tokens: int,
    repeats: int,
    warmup: int,
    pooled_rows: int,
    prime_indexer_qat: bool,
    include_moe: bool,
) -> dict[str, Any]:
    layers = getattr(getattr(model, "model", model), "layers", ())
    layer_mod = layers[int(layer)]
    attn = layer_mod.self_attn
    dtype = mx.float16

    def make_inputs():
        x = _make_synthetic_layer_input(mx, args=model.args, tokens=tokens, dtype=dtype)
        input_ids = _make_input_ids(mx, args=model.args, tokens=tokens)
        cache = _make_layer_cache(model, layer=layer)
        _seed_compressed_pool(
            mx=mx,
            cache=cache,
            attn=getattr(attn, "_original", attn),
            pooled_rows=pooled_rows,
            dtype=dtype,
            prime_indexer_qat=prime_indexer_qat,
        )
        mask = _make_layer_mask(mx=mx, model=model, cache=cache, x=x, tokens=tokens)
        mx.eval(x, input_ids)
        return x, input_ids, cache, mask

    start_cache = _make_layer_cache(model, layer=layer)
    _seed_compressed_pool(
        mx=mx,
        cache=start_cache,
        attn=getattr(attn, "_original", attn),
        pooled_rows=pooled_rows,
        dtype=dtype,
        prime_indexer_qat=prime_indexer_qat,
    )
    start_pool_rows = _pool_rows(start_cache)

    profile = _StageProfile(mx)
    last_result: dict[str, Any] | None = None
    last_cache = None
    for _ in range(max(int(warmup), 0)):
        x, input_ids, cache, mask = make_inputs()
        _run_layer_until_router(
            mx=mx,
            layer_mod=layer_mod,
            x=x,
            mask=mask,
            cache=cache,
            input_ids=input_ids,
            profile=profile,
            include_moe=include_moe,
        )
    profile.reset()

    elapsed = 0.0
    for _ in range(max(int(repeats), 1)):
        x, input_ids, cache, mask = make_inputs()
        t0 = time.perf_counter()
        last_result = _run_layer_until_router(
            mx=mx,
            layer_mod=layer_mod,
            x=x,
            mask=mask,
            cache=cache,
            input_ids=input_ids,
            profile=profile,
            include_moe=include_moe,
        )
        elapsed += time.perf_counter() - t0
        last_cache = cache
    end_pool_rows = _pool_rows(last_cache)

    if last_result is None:
        last_result = {}
    return {
        "layer": int(layer),
        "compress_ratio": int(getattr(attn, "compress_ratio", 0) or 0),
        "input_tokens": int(tokens),
        "pooled_rows_requested": int(pooled_rows),
        "prime_indexer_qat": bool(prime_indexer_qat),
        "include_moe": bool(include_moe),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "seconds_total": float(elapsed),
        "seconds_per_repeat": float(elapsed / max(int(repeats), 1)),
        "pool_rows_start": start_pool_rows,
        "pool_rows_end": end_pool_rows,
        "stage_seconds": profile.payload(repeats),
        **last_result,
    }


def run_ds4_layer_stage_replay(
    package_dir: Path,
    *,
    layer: int = 3,
    input_tokens: int = 3844,
    pooled_rows: int = 0,
    repeats: int = 1,
    warmup: int = 1,
    max_memory_gb: float | None = None,
    prime_indexer_qat: bool = True,
    include_moe: bool = False,
    load_served_model_fn: Callable[..., tuple[Any, Any, dict]] | None = None,
) -> dict[str, Any]:
    import mlx.core as mx

    if max_memory_gb is not None:
        os.environ["MOESPRESSO_SSD_MAX_MEMORY_GB"] = str(max_memory_gb)
    if load_served_model_fn is None:
        from moespresso.runtime.serve import load_served_model

        load_served_model_fn = load_served_model

    model, _tokenizer, manifest = load_served_model_fn(Path(package_dir))
    layers = getattr(getattr(model, "model", model), "layers", ())
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"layer {layer} is out of range for {len(layers)} layers")
    case = _time_layer_case(
        mx=mx,
        model=model,
        layer=layer,
        tokens=input_tokens,
        repeats=repeats,
        warmup=warmup,
        pooled_rows=pooled_rows,
        prime_indexer_qat=prime_indexer_qat,
        include_moe=include_moe,
    )
    return {
        "metric": "ds4_decoder_layer_stage_replay",
        "units": "seconds per replay through one served DS4 decoder layer stage graph",
        "package": str(package_dir),
        "package_artifact_id": manifest.get("artifact_id"),
        "quality_note": (
            "speed diagnostic only; synthetic activations do not replace Q1/Q2/Q3"
        ),
        "case": case,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-layer-stage-replay",
        description="Time one served DS4 decoder layer split at major stage boundaries.",
    )
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--input-tokens", type=int, default=3844)
    parser.add_argument(
        "--pooled-rows",
        type=int,
        default=0,
        help=(
            "Optional decode-style compressed rows to seed before the replay. "
            "Default 0 matches the first prompt chunk."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-memory-gb", type=float, default=None)
    parser.add_argument(
        "--no-prime-indexer-qat",
        action="store_true",
        help="Do not pre-seed the indexer QAT cache for seeded compressed rows.",
    )
    parser.add_argument(
        "--include-moe",
        action="store_true",
        help=(
            "Also replay routed switch_mlp, route-weighted sum, shared experts, "
            "and the final HC FFN post stage. Default stops after router export inputs."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.layer < 0:
        parser.error("--layer must be non-negative")
    if args.input_tokens <= 0:
        parser.error("--input-tokens must be positive")
    if args.pooled_rows < 0:
        parser.error("--pooled-rows must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.max_memory_gb is not None and args.max_memory_gb <= 0:
        parser.error("--max-memory-gb must be positive")
    payload = run_ds4_layer_stage_replay(
        args.package_dir,
        layer=args.layer,
        input_tokens=args.input_tokens,
        pooled_rows=args.pooled_rows,
        repeats=args.repeats,
        warmup=args.warmup,
        max_memory_gb=args.max_memory_gb,
        prime_indexer_qat=not args.no_prime_indexer_qat,
        include_moe=args.include_moe,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
