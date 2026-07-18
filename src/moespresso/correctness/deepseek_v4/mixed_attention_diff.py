"""Replay DS4 static mixed attention dumps with MLX attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moespresso.correctness.deepseek_v4.attention_diff import (
    _layers,
    _load_package_sink,
    _read_pos_dump,
    _reshape_kv,
    _reshape_q,
)
from moespresso.correctness.deepseek_v4.rope_diff import _metrics


def _mixed_visibility(
    *,
    tokens: int,
    n_comp: int,
    window: int,
    ratio: int,
) -> np.ndarray:
    raw = np.zeros((tokens, tokens), dtype=np.bool_)
    comp = np.zeros((tokens, n_comp), dtype=np.bool_)
    for query in range(tokens):
        for key in range(tokens):
            causal = key <= query
            in_window = window == 0 or query - key < window
            raw[query, key] = causal and in_window
        visible = (query + 1) // ratio
        comp[query, :visible] = True
    return np.concatenate([raw, comp], axis=1).reshape(1, 1, tokens, tokens + n_comp)


def _mlx_mixed_attention(
    *,
    qcur: np.ndarray,
    kvcur: np.ndarray,
    kvcompress: np.ndarray,
    sinks: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention

    tokens, heads, head_dim = qcur.shape
    q = mx.array(qcur.transpose(1, 0, 2)[None], dtype=mx.float16)
    raw = mx.array(kvcur.reshape(1, 1, tokens, head_dim), dtype=mx.float16)
    comp = mx.array(kvcompress.reshape(1, 1, kvcompress.shape[0], head_dim), dtype=mx.float16)
    full_kv = mx.concatenate([raw, comp], axis=2)
    out = scaled_dot_product_attention(
        q,
        full_kv,
        full_kv,
        None,
        scale=scale,
        mask=mx.array(mask),
        sinks=mx.array(sinks, dtype=mx.float16),
    )
    mx.eval(out)
    return np.asarray(out.astype(mx.float32))[0].transpose(1, 0, 2)


def compare_mixed_attention_dumps(
    *,
    dump_prefix: Path,
    package_dir: Path,
    config_path: Path,
    layers: list[int],
    final_row: int,
    pos: int,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    n_heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    scale = float(head_dim ** -0.5)
    window = int(config.get("sliding_window") or 0)
    compress_ratios = list(config.get("compress_ratios") or [])

    rows = []
    for layer in layers:
        ratio = int(compress_ratios[layer]) if layer < len(compress_ratios) else 0
        if ratio <= 0:
            raise ValueError(
                "mixed attention replay only supports compressed layers; "
                f"layer {layer} has compress_ratio={ratio}"
            )
        qcur = _reshape_q(
            "Qcur",
            _read_pos_dump(dump_prefix, "Qcur", layer, pos),
            n_heads=n_heads,
            head_dim=head_dim,
        )
        tokens = qcur.shape[0]
        if final_row >= tokens:
            raise ValueError(
                f"final row {final_row} is outside layer {layer} dump with {tokens} rows"
            )
        kvcur = _reshape_kv(
            "KVcur",
            _read_pos_dump(dump_prefix, "KVcur", layer, pos),
            tokens=tokens,
            head_dim=head_dim,
        )
        ref = _reshape_q(
            "kqv_out",
            _read_pos_dump(dump_prefix, "kqv_out", layer, pos),
            n_heads=n_heads,
            head_dim=head_dim,
        )
        kvcompress = _read_pos_dump(dump_prefix, "KVcompress", layer, pos)
        if kvcompress.size % head_dim:
            raise ValueError(f"bad KVcompress dump size: {kvcompress.size}")
        kvcompress = kvcompress.reshape(kvcompress.size // head_dim, head_dim)
        mask = _mixed_visibility(
            tokens=tokens,
            n_comp=kvcompress.shape[0],
            window=window,
            ratio=ratio,
        )
        got = _mlx_mixed_attention(
            qcur=qcur,
            kvcur=kvcur,
            kvcompress=kvcompress,
            sinks=_load_package_sink(package_dir, layer),
            mask=mask,
            scale=scale,
        )
        if got.shape != ref.shape:
            raise ValueError(f"MLX output shape {got.shape} does not match {ref.shape}")
        rows.append({
            "layer": int(layer),
            "compress_ratio": ratio,
            "tokens": int(tokens),
            "compressed_rows": int(kvcompress.shape[0]),
            "metrics": {
                "all": _metrics(got.reshape(-1), ref.reshape(-1)),
                "final": _metrics(got[final_row].reshape(-1), ref[final_row].reshape(-1)),
            },
        })

    return {
        "dump_prefix": str(dump_prefix),
        "package_dir": str(package_dir),
        "pos": int(pos),
        "final_row": int(final_row),
        "layers": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--layers", default="2", type=_layers)
    parser.add_argument("--final-row", required=True, type=int)
    parser.add_argument("--pos", default=0, type=int)
    args = parser.parse_args(argv)
    result = compare_mixed_attention_dumps(
        dump_prefix=args.dump_prefix,
        package_dir=args.package,
        config_path=args.config,
        layers=args.layers,
        final_row=args.final_row,
        pos=args.pos,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
