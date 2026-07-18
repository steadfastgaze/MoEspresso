"""Compare DS4 reference RoPE dumps against the served MLX RoPE implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


N_HEADS = 64
HEAD_DIM = 512
Q_DIM = N_HEADS * HEAD_DIM


def _read_dump(prefix: Path, name: str, layer: int) -> np.ndarray:
    path = Path(f"{prefix}_{name}-{layer}_pos0.bin")
    if not path.is_file():
        raise FileNotFoundError(f"missing DS4 dump: {path}")
    return np.fromfile(path, dtype=np.float32)


def _metrics(got: np.ndarray, ref: np.ndarray) -> dict:
    diff = got.astype(np.float64) - ref.astype(np.float64)
    ref64 = ref.astype(np.float64)
    denom = float(np.sqrt(np.mean(ref64 * ref64)))
    rms = float(np.sqrt(np.mean(diff * diff)))
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rms": rms,
        "rel_rms": rms / denom if denom else float("inf"),
    }


def _q_apply(qnorm, *, layout: str, tokens: int, rope):
    import mlx.core as mx

    from jang_tools.dsv4.mlx_model import _apply_partial_rope

    if layout == "token_head_dim":
        q = qnorm.reshape(tokens, N_HEADS, HEAD_DIM).transpose(1, 0, 2)[None]
        out = _apply_partial_rope(mx.array(q), rope, offset=0)
        return np.asarray(out).squeeze(0).transpose(1, 0, 2).reshape(-1)
    if layout == "head_token_dim":
        q = qnorm.reshape(N_HEADS, tokens, HEAD_DIM)[None]
        out = _apply_partial_rope(mx.array(q), rope, offset=0)
        return np.asarray(out).squeeze(0).reshape(-1)
    raise ValueError(f"unknown Q layout: {layout}")


def _kv_apply(kvnorm, *, tokens: int, rope):
    import mlx.core as mx

    from jang_tools.dsv4.mlx_model import _apply_partial_rope

    kv = kvnorm.reshape(tokens, HEAD_DIM)[None, None]
    out = _apply_partial_rope(mx.array(kv), rope, offset=0)
    return np.asarray(out).reshape(-1)


def _rope_for(args, layer: int):
    from jang_tools.dsv4.mlx_model import DeepseekV4RoPE

    ratio = int((args.compress_ratios or [])[layer])
    if ratio:
        theta = float(args.compress_rope_theta)
        scaling = args.rope_scaling
    else:
        theta = float(args.rope_theta)
        scaling = None
    return ratio, DeepseekV4RoPE(
        args.qk_rope_head_dim,
        theta,
        scaling,
        args.max_position_embeddings,
    )


def compare_rope_dumps(
    *,
    dump_prefix: Path,
    config_path: Path,
    layers: list[int],
    final_row: int,
) -> list[dict]:
    from jang_tools.dsv4.mlx_model import ModelArgs

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    args = ModelArgs.from_dict(cfg)
    rows = []
    for layer in layers:
        ratio, rope = _rope_for(args, layer)
        qnorm = _read_dump(dump_prefix, "Qnorm", layer)
        qcur = _read_dump(dump_prefix, "Qcur", layer)
        kvnorm = _read_dump(dump_prefix, "KVnorm", layer)
        kvrope = _read_dump(dump_prefix, "KVrope", layer)
        tokens = qnorm.size // Q_DIM
        if tokens * Q_DIM != qnorm.size:
            raise ValueError(f"bad Q dump size for layer {layer}: {qnorm.size}")
        if kvnorm.size != tokens * HEAD_DIM:
            raise ValueError(f"bad KV dump size for layer {layer}: {kvnorm.size}")
        if final_row >= tokens:
            raise ValueError(
                f"final row {final_row} is outside layer {layer} dump with {tokens} rows"
            )

        q_layouts = {}
        for layout in ("token_head_dim", "head_token_dim"):
            qgot = _q_apply(qnorm, layout=layout, tokens=tokens, rope=rope)
            q_layouts[layout] = {
                "all": _metrics(qgot, qcur),
                "final": _metrics(
                    qgot.reshape(tokens, Q_DIM)[final_row],
                    qcur.reshape(tokens, Q_DIM)[final_row],
                ),
            }
        kvgot = _kv_apply(kvnorm, tokens=tokens, rope=rope)
        rows.append({
            "layer": int(layer),
            "compress_ratio": ratio,
            "tokens": int(tokens),
            "q_layouts": q_layouts,
            "kv": {
                "all": _metrics(kvgot, kvrope),
                "final": _metrics(
                    kvgot.reshape(tokens, HEAD_DIM)[final_row],
                    kvrope.reshape(tokens, HEAD_DIM)[final_row],
                ),
            },
        })
    return rows


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--layers", default="0,1,2", type=_layers)
    parser.add_argument("--final-row", required=True, type=int)
    args = parser.parse_args(argv)
    result = compare_rope_dumps(
        dump_prefix=args.dump_prefix,
        config_path=args.config,
        layers=args.layers,
        final_row=args.final_row,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
