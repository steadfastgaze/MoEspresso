"""Manual one-layer DS4 attention speed replay.

This speed diagnostic loads a real served DS4 package, seeds one layer's
compressed pools when applicable, and times calls
through that layer's actual attention module. The goal is to keep the hot
compressed-attention surfaces measurable without running a full model generation
benchmark for every hypothesis. Quality promotion still requires the
model-specific gates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from types import MethodType
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class Ratio4ReplayCase:
    name: str
    pooled_rows: int
    index_topk: int
    cache: bool = True


def default_replay_cases() -> tuple[Ratio4ReplayCase, ...]:
    """Return the current speed-goal replay cases.

    The row counts intentionally mirror the consolidated speed log: 961 rows for
    the bounded long prompt scale, and 7618 rows for the Q3-scale long-context
    surface. The top-k skipped cases keep the previous Entry 99 split available
    without making it the main optimization target.
    """
    return (
        Ratio4ReplayCase("local_no_compressed_pool", 0, 512, cache=False),
        Ratio4ReplayCase("ratio4_rows961_topk_active", 961, 512),
        Ratio4ReplayCase("ratio4_rows7618_topk_active", 7618, 512),
        Ratio4ReplayCase("ratio4_rows961_topk_skipped", 961, 2000),
        Ratio4ReplayCase("ratio4_rows512_topk_skipped", 512, 512),
    )


def selected_replay_cases(
    *,
    include_splits: bool,
    pooled_rows: Sequence[int] | None = None,
    index_topk: int = 512,
    ratio_label: str = "ratio4",
) -> tuple[Ratio4ReplayCase, ...]:
    if pooled_rows:
        return tuple(
            Ratio4ReplayCase(
                f"{ratio_label}_rows{int(rows)}_topk{int(index_topk)}",
                int(rows),
                int(index_topk),
            )
            for rows in pooled_rows
        )
    cases = default_replay_cases()
    if include_splits:
        return cases
    return cases[:3]


def _set_index_topk(attn: Any, topk: int) -> None:
    indexer = getattr(attn, "indexer", None)
    if indexer is None:
        return
    object.__setattr__(indexer, "index_topk", int(topk))
    original = getattr(indexer, "_original", None)
    if original is not None and hasattr(original, "index_topk"):
        object.__setattr__(original, "index_topk", int(topk))


def _make_seeded_cache(
    *,
    mx: Any,
    cache_cls: Callable[..., Any],
    sliding_window: int,
    compress_ratio: int,
    pooled_rows: int,
    head_dim: int,
    index_head_dim: int,
    dtype: Any,
    prime_indexer_qat: bool,
    seed_indexer_state: bool,
) -> Any:
    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

    cache = cache_cls(sliding_window, compress_ratio=int(compress_ratio))
    if pooled_rows <= 0:
        return cache
    compressed = mx.random.normal((1, int(pooled_rows), int(head_dim))).astype(dtype)
    cache.compressor_state["pooled"] = compressed
    values = [compressed]
    if seed_indexer_state:
        indexer = mx.random.normal((1, int(pooled_rows), int(index_head_dim))).astype(dtype)
        cache.indexer_state["pooled"] = indexer
        values.append(indexer)
        if prime_indexer_qat:
            cache.indexer_state["pooled_qat"] = _dsv4_indexer_qat(mx, indexer)
            cache.indexer_state["pooled_qat_rows"] = int(pooled_rows)
            values.append(cache.indexer_state["pooled_qat"])
    mx.eval(*values)
    return cache


def _pool_rows(cache: Any) -> dict[str, int]:
    if cache is None:
        return {"compressor": 0, "indexer": 0}
    compressor = cache.compressor_state.get("pooled")
    indexer = cache.indexer_state.get("pooled")
    return {
        "compressor": 0 if compressor is None else int(compressor.shape[1]),
        "indexer": 0 if indexer is None else int(indexer.shape[1]),
    }


class _StageProfile:
    def __init__(self, mx: Any):
        self._mx = mx
        self.rows: dict[str, dict[str, float | int]] = {}

    def reset(self) -> None:
        self.rows.clear()

    def _eval(self, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, tuple | list):
            values = [item for item in value if item is not None]
            if values:
                self._mx.eval(*values)
            return
        self._mx.eval(value)

    def record(self, name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        value = fn()
        self._eval(value)
        elapsed = time.perf_counter() - t0
        row = self.rows.setdefault(name, {"calls": 0, "seconds": 0.0})
        row["calls"] = int(row["calls"]) + 1
        row["seconds"] = float(row["seconds"]) + float(elapsed)
        return value

    def payload(self, repeats: int) -> dict[str, dict[str, float | int]]:
        n = max(int(repeats), 1)
        return {
            name: {
                "calls": int(row["calls"]),
                "seconds_total": float(row["seconds"]),
                "seconds_per_repeat": float(row["seconds"]) / n,
            }
            for name, row in sorted(self.rows.items())
        }


class _TimedCallable:
    def __init__(self, original: Any, *, name: str, profile: _StageProfile):
        self._original = original
        self._name = name
        self._profile = profile

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._profile.record(
            self._name,
            lambda: self._original(*args, **kwargs),
        )


class _StagePatch:
    def __init__(self, *, attn: Any, profile: _StageProfile):
        self._attn = attn
        self._target = getattr(attn, "_original", attn)
        self._profile = profile
        self._restore: list[Callable[[], None]] = []

    def __enter__(self):
        import jang_tools.dsv4.mlx_model as dsv4_model

        attn = self._target
        for attr, name in (
            ("compressor", "compressor"),
            ("indexer", "indexer"),
            ("wo_b", "wo_b"),
        ):
            original = getattr(attn, attr, None)
            if original is None:
                continue
            object.__setattr__(
                attn,
                attr,
                _TimedCallable(original, name=name, profile=self._profile),
            )
            self._restore.append(
                lambda attr=attr, original=original: object.__setattr__(
                    attn, attr, original)
            )

        original_grouped = getattr(attn, "_grouped_output_projection", None)
        if original_grouped is not None:
            def grouped(self, out):
                return self_profile.record(
                    "grouped_output_projection",
                    lambda: original_grouped(out),
                )

            self_profile = self._profile
            object.__setattr__(
                attn,
                "_grouped_output_projection",
                MethodType(grouped, attn),
            )
            self._restore.append(
                lambda original=original_grouped: object.__setattr__(
                    attn, "_grouped_output_projection", original)
            )

        original_sdpa = dsv4_model.scaled_dot_product_attention

        def sdpa(*args, **kwargs):
            return self._profile.record(
                "scaled_dot_product_attention",
                lambda: original_sdpa(*args, **kwargs),
            )

        dsv4_model.scaled_dot_product_attention = sdpa
        self._restore.append(
            lambda original=original_sdpa: setattr(
                dsv4_model, "scaled_dot_product_attention", original)
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        while self._restore:
            self._restore.pop()()


def _profile_record(profile: _StageProfile | None, name: str, fn: Callable[[], Any]) -> Any:
    if profile is None:
        return fn()
    return profile.record(name, fn)


def _run_indexed_prefill_consumer(
    *,
    mx: Any,
    attn: Any,
    x: Any,
    cache: Any,
    profile: _StageProfile | None,
    tiled_indexer_scores: bool,
) -> Any:
    """Replay one ratio-4 prefill call with a DS4-c-shaped indexed consumer."""
    from jang_tools.dsv4.mlx_model import _apply_partial_rope, _get_q_norm_ones
    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat
    from moespresso.runtime.deepseek_v4.indexed_attention_kernel import (
        indexed_mixed_attention_prefill_live_f16,
        indexer_q_qat_live,
        indexer_score_operands,
        indexer_scores_tiled_live,
    )

    if cache is None:
        raise ValueError("indexed prefill consumer replay requires DeepseekV4Cache")
    batch, tokens, _ = x.shape
    if int(batch) != 1:
        raise ValueError("indexed prefill consumer replay currently requires batch=1")
    if int(tokens) <= 1:
        raise ValueError("indexed prefill consumer replay requires input_tokens > 1")
    offset = int(cache.offset)

    q_residual = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(q_residual).reshape(batch, tokens, attn.n_heads, attn.head_dim)
    q = mx.fast.rms_norm(
        q,
        weight=_get_q_norm_ones(attn.head_dim, q.dtype),
        eps=attn.args.rms_norm_eps,
    )
    q = q.transpose(0, 2, 1, 3)

    kv = attn.kv_norm(attn.wkv(x)).reshape(
        batch, tokens, 1, attn.head_dim,
    ).transpose(0, 2, 1, 3)

    q = _apply_partial_rope(q, attn.rope, offset)
    kv = _apply_partial_rope(kv, attn.rope, offset)
    kv, _ = cache.update_and_fetch(kv, kv)

    pooled = _profile_record(
        profile,
        "compressor",
        lambda: attn.compressor(x, attn.compress_rope, cache, offset),
    )
    if pooled.shape[1] <= 0:
        raise ValueError("indexed prefill consumer replay requires compressed rows")
    if not hasattr(attn, "indexer"):
        raise ValueError("indexed prefill consumer replay requires an indexer")
    if pooled.shape[1] <= attn.indexer.index_topk:
        raise ValueError("indexed prefill consumer replay requires active sparse top-k")
    if tiled_indexer_scores:
        indexer = attn.indexer
        index_pooled = _profile_record(
            profile,
            "indexer_compressor",
            lambda: indexer.compressor(
                x, attn.compress_rope, cache, offset, state_key="indexer_state"
            ),
        )
        q_idx = indexer.wq_b(q_residual).reshape(
            batch, tokens, indexer.n_heads, indexer.head_dim,
        )
        q_idx = q_idx.transpose(0, 2, 1, 3)
        q_idx = _apply_partial_rope(q_idx, attn.rope, offset)
        q_idx = _profile_record(
            profile,
            "indexer_q_qat",
            lambda: indexer_q_qat_live(q_idx.astype(mx.float32)),
        )
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
                tail = _profile_record(
                    profile,
                    "indexer_pool_qat",
                    lambda: _dsv4_indexer_qat(mx, index_pooled[:, cached_rows:, :]),
                )
                index_pooled = mx.concatenate([cached, tail], axis=1)
                if isinstance(state, dict):
                    state["pooled_qat"] = index_pooled
                    state["pooled_qat_rows"] = int(index_pooled.shape[1])
        else:
            index_pooled = _profile_record(
                profile,
                "indexer_pool_qat",
                lambda: _dsv4_indexer_qat(mx, index_pooled),
            )
            if isinstance(state, dict):
                state["pooled_qat"] = index_pooled
                state["pooled_qat_rows"] = int(index_pooled.shape[1])
        weights = indexer.weights_proj(x).astype(mx.float32) * (
            indexer.n_heads ** -0.5
        ) * indexer.scale
        q_scores, comp_scores = indexer_score_operands(q_idx, index_pooled)
        scores = _profile_record(
            profile,
            "indexer_scores_tiled",
            lambda: indexer_scores_tiled_live(
                q_scores,
                weights,
                comp_scores,
                pos0=offset,
                ratio=int(attn.compress_ratio),
            ),
        )
        topk = _profile_record(
            profile,
            "indexer_topk",
            lambda: mx.argpartition(
                -scores,
                kth=min(int(indexer.index_topk), int(index_pooled.shape[1])) - 1,
                axis=-1,
            )[..., : min(int(indexer.index_topk), int(index_pooled.shape[1]))],
        )
    else:
        topk = _profile_record(
            profile,
            "indexer",
            lambda: attn.indexer(
                x, q_residual, attn.compress_rope, attn.rope, cache, offset,
            ),
        )
    if topk is None:
        raise ValueError("indexed prefill consumer replay expected non-empty top-k")
    topk = mx.sort(topk[0].astype(mx.int32), axis=-1)
    heads = _profile_record(
        profile,
        "indexed_prefill_attention",
        lambda: indexed_mixed_attention_prefill_live_f16(
            q,
            kv,
            pooled,
            topk[None],
            attn.attn_sink.astype(mx.float32),
            pos0=offset,
            window=int(attn.args.sliding_window),
            ratio=int(attn.compress_ratio),
        ),
    )
    out = _apply_partial_rope(heads, attn.rope, offset, inverse=True)
    out = out.transpose(0, 2, 1, 3).reshape(
        batch, tokens, attn.n_heads * attn.head_dim,
    )
    out = _profile_record(
        profile,
        "grouped_output_projection",
        lambda: attn._grouped_output_projection(out),
    )
    return _profile_record(profile, "wo_b", lambda: attn.wo_b(out))


def _time_case(
    *,
    mx: Any,
    attn: Any,
    cache_cls: Callable[..., Any],
    sliding_window: int,
    hidden_size: int,
    case: Ratio4ReplayCase,
    repeats: int,
    warmup: int,
    dtype: Any,
    prime_indexer_qat: bool,
    stage_profile: bool = False,
    input_tokens: int = 1,
    indexed_prefill_consumer: bool = False,
    tiled_indexer_scores: bool = False,
) -> dict[str, Any]:
    head_dim = int(getattr(attn, "head_dim", 512))
    compress_ratio = int(getattr(attn, "compress_ratio", 0) or 0)
    indexer = getattr(attn, "indexer", None)
    index_head_dim = int(getattr(indexer, "head_dim", 128))
    _set_index_topk(attn, case.index_topk)
    tokens = int(input_tokens)
    if tokens <= 0:
        raise ValueError("input_tokens must be positive")
    x = mx.random.normal((1, tokens, int(hidden_size))).astype(dtype)
    def make_cache():
        if not case.cache or compress_ratio <= 0:
            return None
        return _make_seeded_cache(
            mx=mx,
            cache_cls=cache_cls,
            sliding_window=sliding_window,
            compress_ratio=compress_ratio,
            pooled_rows=case.pooled_rows,
            head_dim=head_dim,
            index_head_dim=index_head_dim,
            dtype=dtype,
            prime_indexer_qat=prime_indexer_qat,
            seed_indexer_state=indexer is not None,
        )

    cache = make_cache()
    mask = None
    if tokens > 1:
        from mlx_lm.models.base import create_attention_mask

        mask = create_attention_mask(
            x,
            cache,
            window_size=int(sliding_window),
            return_array=True,
        )
    mx.eval(x)
    start_pool_rows = _pool_rows(cache)
    reset_cache_each_call = tokens > 1

    profile = _StageProfile(mx) if stage_profile else None
    patch = (
        None
        if profile is None or indexed_prefill_consumer
        else _StagePatch(attn=attn, profile=profile)
    )
    last_cache = cache

    def run_once():
        nonlocal last_cache
        active_cache = make_cache() if reset_cache_each_call else last_cache
        if indexed_prefill_consumer:
            if compress_ratio != 4:
                raise ValueError(
                    "indexed prefill consumer replay is only defined for ratio-4 layers")
            y = _run_indexed_prefill_consumer(
                mx=mx,
                attn=getattr(attn, "_original", attn),
                x=x,
                cache=active_cache,
                profile=profile,
                tiled_indexer_scores=tiled_indexer_scores,
            )
        else:
            y = attn(x, mask=mask, cache=active_cache)
        last_cache = active_cache
        mx.eval(y)
        return y

    if patch is None:
        for _ in range(max(int(warmup), 0)):
            run_once()

        t0 = time.perf_counter()
        for _ in range(max(int(repeats), 1)):
            run_once()
        elapsed = time.perf_counter() - t0
    else:
        with patch:
            for _ in range(max(int(warmup), 0)):
                run_once()
            profile.reset()
            t0 = time.perf_counter()
            for _ in range(max(int(repeats), 1)):
                run_once()
            elapsed = time.perf_counter() - t0
    end_pool_rows = _pool_rows(last_cache)
    out = {
        "case": case.name,
        "pooled_rows_requested": int(case.pooled_rows),
        "index_topk": int(case.index_topk),
        "cache": bool(case.cache),
        "prime_indexer_qat": bool(prime_indexer_qat),
        "stage_profile": bool(stage_profile),
        "input_tokens": int(tokens),
        "compress_ratio": int(compress_ratio),
        "indexed_prefill_consumer": bool(indexed_prefill_consumer),
        "tiled_indexer_scores": bool(tiled_indexer_scores),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "seconds_total": float(elapsed),
        "seconds_per_repeat": float(elapsed / max(int(repeats), 1)),
        "pool_rows_start": start_pool_rows,
        "pool_rows_end": end_pool_rows,
    }
    if profile is not None:
        out["stage_seconds"] = profile.payload(repeats)
    return out


def run_ratio4_attention_replay(
    package_dir: Path,
    *,
    layer: int = 2,
    repeats: int = 5,
    warmup: int = 1,
    include_splits: bool = False,
    pooled_rows: Sequence[int] | None = None,
    index_topk: int = 512,
    prime_indexer_qat: bool = True,
    stage_profile: bool = False,
    input_tokens: int = 1,
    indexed_prefill_consumer: bool = False,
    tiled_indexer_scores: bool = False,
    load_served_model_fn: Callable[..., tuple[Any, Any, dict]] | None = None,
) -> dict[str, Any]:
    import mlx.core as mx
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache

    if load_served_model_fn is None:
        from moespresso.runtime.serve import load_served_model

        load_served_model_fn = load_served_model

    model, _tokenizer, manifest = load_served_model_fn(Path(package_dir))
    layers = getattr(getattr(model, "model", model), "layers", ())
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"layer {layer} is out of range for {len(layers)} layers")
    attn = layers[int(layer)].self_attn
    compress_ratio = int(getattr(attn, "compress_ratio", 0) or 0)
    if (indexed_prefill_consumer or tiled_indexer_scores) and compress_ratio != 4:
        raise ValueError(
            f"layer {layer} is compress_ratio={compress_ratio}; "
            "indexed prefill consumer probes require a ratio-4 layer")
    hidden_size = int(getattr(model.args, "hidden_size", getattr(attn, "hidden_size", 4096)))
    sliding_window = int(getattr(model.args, "sliding_window", 128))
    cases = selected_replay_cases(
        include_splits=include_splits,
        pooled_rows=pooled_rows,
        index_topk=index_topk,
        ratio_label=f"ratio{compress_ratio}",
    )
    results = [
        _time_case(
            mx=mx,
            attn=attn,
            cache_cls=DeepseekV4Cache,
            sliding_window=sliding_window,
            hidden_size=hidden_size,
            case=case,
            repeats=repeats,
            warmup=warmup,
            dtype=mx.float16,
            prime_indexer_qat=prime_indexer_qat,
            stage_profile=stage_profile,
            input_tokens=input_tokens,
            indexed_prefill_consumer=indexed_prefill_consumer,
            tiled_indexer_scores=tiled_indexer_scores,
        )
        for case in cases
    ]
    return {
        "metric": "ds4_attention_one_layer_replay",
        "units": "seconds per call through one served DS4 attention layer",
        "package": str(package_dir),
        "package_artifact_id": manifest.get("artifact_id"),
        "layer": int(layer),
        "compress_ratio": compress_ratio,
        "stage_profile": bool(stage_profile),
        "input_tokens": int(input_tokens),
        "indexed_prefill_consumer": bool(indexed_prefill_consumer),
        "tiled_indexer_scores": bool(tiled_indexer_scores),
        "quality_note": (
            "speed diagnostic only; does not replace Q1/Q2/Q3 after retained math changes"
        ),
        "cases": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-ratio4-attention-replay",
        description=(
            "Time one served DS4 attention layer with seeded compressed pools."
        ),
    )
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--include-splits",
        action="store_true",
        help="Also run the top-k skipped split cases from the speed log.",
    )
    parser.add_argument(
        "--pooled-rows",
        type=int,
        action="append",
        help="Custom active-topk compressed-pool row count; may be passed multiple times.",
    )
    parser.add_argument(
        "--index-topk",
        type=int,
        default=512,
        help=(
            "Diagnostic sparse-indexer top-k for custom --pooled-rows cases. "
            "Use a value above the final pool row count to skip top-k."
        ),
    )
    parser.add_argument(
        "--no-prime-indexer-qat",
        action="store_true",
        help="Do not pre-seed the indexer QAT cache before timing.",
    )
    parser.add_argument(
        "--stage-profile",
        action="store_true",
        help=(
            "Diagnostic split: force evaluation at compressor/indexer/SDPA/"
            "output-projection boundaries and report substage timings."
        ),
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=1,
        help=(
            "Input sequence length for each replay call. The default one-token "
            "shape measures decode; larger values measure a one-layer prefill shape."
        ),
    )
    parser.add_argument(
        "--indexed-prefill-consumer",
        action="store_true",
        help=(
            "Diagnostic only: for prefill cases with active sparse top-k, replay "
            "the attention prefix and replace generic SDPA with the indexed "
            "prefill probe kernel. This is not served-path wiring."
        ),
    )
    parser.add_argument(
        "--tiled-indexer-scores",
        action="store_true",
        help=(
            "Diagnostic only, with --indexed-prefill-consumer: replace generic "
            "indexer score construction with a DS4-c-shaped tiled score kernel."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.input_tokens <= 0:
        parser.error("--input-tokens must be positive")
    if args.index_topk <= 0:
        parser.error("--index-topk must be positive")
    if args.tiled_indexer_scores and not args.indexed_prefill_consumer:
        parser.error("--tiled-indexer-scores requires --indexed-prefill-consumer")

    payload = run_ratio4_attention_replay(
        args.package_dir,
        layer=args.layer,
        repeats=args.repeats,
        warmup=args.warmup,
        include_splits=args.include_splits,
        pooled_rows=args.pooled_rows,
        index_topk=args.index_topk,
        prime_indexer_qat=not args.no_prime_indexer_qat,
        stage_profile=args.stage_profile,
        input_tokens=args.input_tokens,
        indexed_prefill_consumer=args.indexed_prefill_consumer,
        tiled_indexer_scores=args.tiled_indexer_scores,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
