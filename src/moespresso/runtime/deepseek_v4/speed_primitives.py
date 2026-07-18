"""Synthetic MLX primitive counts for DeepSeek-V4 speed work.

This structural diagnostic uses ``mx.export_function`` to count graph primitives
for small DS4-shaped decode
stages so speed ideas can be screened before touching the served hot path.
It does not measure quality or wall-clock performance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable, Sequence


def _export_primitives(fn: Callable[..., Any], *args: Any) -> list[str]:
    import mlx.core as mx

    names: list[str] = []

    def callback(payload: dict[str, Any]) -> None:
        if payload.get("type") == "primitive":
            names.append(str(payload.get("name")))

    mx.export_function(callback, fn, *args)
    return names


def _stage_summary(name: str, fn: Callable[..., Any], *args: Any) -> dict[str, Any]:
    names = _export_primitives(fn, *args)
    counts = Counter(names)
    return {
        "stage": name,
        "primitive_count": len(names),
        "primitive_counts": dict(sorted(counts.items())),
        "primitives": names,
    }


def build_ds4_primitive_summary(
    *,
    pooled_rows: int = 640,
    topk_rows: int = 512,
    local_rows: int = 128,
    head_dim: int = 128,
    hidden_dim: int = 512,
    index_heads: int = 64,
) -> dict[str, Any]:
    """Return primitive summaries for DS4-shaped decode subgraphs."""
    import mlx.core as mx
    import jang_tools.dsv4.mlx_model as dsv4

    from moespresso.runtime.deepseek_v4.model import _dsv4_indexer_qat

    batch = 1
    tokens = 1
    selected_pooled_rows = min(int(topk_rows), int(pooled_rows))
    pooled_kv = mx.zeros((batch, pooled_rows, hidden_dim), dtype=mx.float16)
    topk = mx.arange(selected_pooled_rows).reshape(
        batch, tokens, selected_pooled_rows).astype(mx.int32)

    def selected_rows_current(pooled, selected):
        idx = selected[:, None, :, :, None]
        expanded = mx.broadcast_to(
            pooled[:, None, None, :, :],
            (batch, 1, tokens, pooled_rows, hidden_dim),
        )
        return mx.take_along_axis(
            expanded,
            mx.broadcast_to(idx, idx.shape[:-1] + (hidden_dim,)),
            axis=3,
        ).reshape(batch, 1, -1, hidden_dim)

    def selected_rows_direct_take(pooled, selected):
        return mx.take(pooled, selected.reshape(-1), axis=1)[:, None]

    q = mx.zeros((batch, index_heads, tokens, head_dim), dtype=mx.float16)
    local_kv = mx.zeros((batch, 1, local_rows, head_dim), dtype=mx.float16)
    selected_kv = mx.zeros(
        (batch, 1, selected_pooled_rows, head_dim), dtype=mx.float16)
    sinks = mx.zeros((index_heads,), dtype=mx.float16)

    def ratio4_attention_consumer(q_in, local, selected, sinks_in):
        full = mx.concatenate([local, selected], axis=2)
        return dsv4.scaled_dot_product_attention(
            q_in,
            full,
            full,
            None,
            head_dim**-0.5,
            None,
            sinks=sinks_in,
        )

    indexer_pooled_rows = max(int(pooled_rows), int(topk_rows))
    indexer_pooled = mx.zeros(
        (batch, indexer_pooled_rows, head_dim), dtype=mx.float16)
    indexer_q = mx.zeros(
        (batch, index_heads, tokens, head_dim), dtype=mx.float16)
    weights = mx.ones((batch, tokens, index_heads), dtype=mx.float32)
    visible = mx.ones((batch, tokens, indexer_pooled_rows), dtype=mx.bool_)

    def indexer_decode_cached_pool_score(q_in, pooled, weight_in):
        q_qat = _dsv4_indexer_qat(mx, q_in)
        scores = (
            q_qat.astype(mx.float32)
            @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
        )
        scores = mx.maximum(scores, 0) * (head_dim**-0.5)
        scores = (scores * weight_in.swapaxes(-1, -2)[..., None]).sum(axis=1)
        return mx.argpartition(-scores, kth=selected_pooled_rows - 1, axis=-1)[
            ..., :selected_pooled_rows
        ]

    def indexer_prefill_masked_score(q_in, pooled, weight_in, visible_in):
        q_qat = _dsv4_indexer_qat(mx, q_in)
        scores = (
            q_qat.astype(mx.float32)
            @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
        )
        scores = mx.maximum(scores, 0) * (head_dim**-0.5)
        scores = (scores * weight_in.swapaxes(-1, -2)[..., None]).sum(axis=1)
        scores = mx.where(visible_in, scores, mx.array(-mx.inf, dtype=scores.dtype))
        return mx.argpartition(-scores, kth=selected_pooled_rows - 1, axis=-1)[
            ..., :selected_pooled_rows
        ]

    stages = [
        _stage_summary(
            "decode_selected_rows_current",
            selected_rows_current,
            pooled_kv,
            topk,
        ),
        _stage_summary(
            "decode_selected_rows_direct_take",
            selected_rows_direct_take,
            pooled_kv,
            topk,
        ),
        _stage_summary(
            "ratio4_attention_consumer",
            ratio4_attention_consumer,
            q,
            local_kv,
            selected_kv,
            sinks,
        ),
        _stage_summary(
            "indexer_decode_cached_pool_score",
            indexer_decode_cached_pool_score,
            indexer_q,
            indexer_pooled,
            weights,
        ),
        _stage_summary(
            "indexer_prefill_masked_score",
            indexer_prefill_masked_score,
            indexer_q,
            indexer_pooled,
            weights,
            visible,
        ),
    ]
    return {
        "metric": "ds4_synthetic_mlx_primitive_counts",
        "units": "MLX export primitive counts; no model load, no generation",
        "shape": {
            "batch": batch,
            "tokens": tokens,
            "pooled_rows": int(pooled_rows),
            "topk_rows": int(topk_rows),
            "selected_pooled_rows": selected_pooled_rows,
            "local_rows": int(local_rows),
            "head_dim": int(head_dim),
            "hidden_dim": int(hidden_dim),
            "index_heads": int(index_heads),
        },
        "stages": stages,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-speed-primitives",
        description="Dump synthetic MLX primitive counts for DS4 speed probes.",
    )
    parser.add_argument("--pooled-rows", type=int, default=640)
    parser.add_argument("--topk-rows", type=int, default=512)
    parser.add_argument("--local-rows", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--index-heads", type=int, default=64)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    payload = build_ds4_primitive_summary(
        pooled_rows=args.pooled_rows,
        topk_rows=args.topk_rows,
        local_rows=args.local_rows,
        head_dim=args.head_dim,
        hidden_dim=args.hidden_dim,
        index_heads=args.index_heads,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
