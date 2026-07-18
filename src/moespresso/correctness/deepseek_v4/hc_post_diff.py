"""Replay DS4 mHC post-combine outputs from reference dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moespresso.correctness.deepseek_v4.hc_input_diff import (
    HC_DIM,
    HIDDEN_SIZE,
    N_HC,
    _load_package_tensor,
    _read_pos_dump,
    _replay_hc_pre,
    _reshape_hc,
    _reshape_hidden,
)
from moespresso.correctness.deepseek_v4.rope_diff import _metrics


def _as_numpy(value) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def _hc_post_variants(
    *,
    x_hidden: np.ndarray,
    residual_hc: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
) -> dict[str, np.ndarray]:
    import mlx.core as mx

    x = mx.array(x_hidden.reshape(1, x_hidden.shape[0], HIDDEN_SIZE), dtype=mx.float32)
    residual = mx.array(
        residual_hc.reshape(1, residual_hc.shape[0], N_HC, HIDDEN_SIZE),
        dtype=mx.float32,
    )
    post_mx = mx.array(post.reshape(1, post.shape[0], N_HC), dtype=mx.float32)
    comb_mx = mx.array(
        comb.reshape(1, comb.shape[0], N_HC, N_HC),
        dtype=mx.float32,
    )
    transposed = post_mx[..., None] * x[..., None, :] + mx.matmul(
        mx.swapaxes(comb_mx, -1, -2),
        residual,
    )
    untransposed = post_mx[..., None] * x[..., None, :] + mx.matmul(comb_mx, residual)
    mx.eval(transposed, untransposed)
    return {
        "transposed_f32": _as_numpy(transposed).reshape(-1, N_HC, HIDDEN_SIZE),
        "transposed_bf16": _as_numpy(transposed.astype(mx.bfloat16)).reshape(
            -1,
            N_HC,
            HIDDEN_SIZE,
        ),
        "untransposed_f32": _as_numpy(untransposed).reshape(-1, N_HC, HIDDEN_SIZE),
    }


def compare_hc_post_dumps(
    *,
    package_dir: Path,
    dump_prefix: Path,
    residual_name: str,
    residual_layer: int,
    x_name: str,
    x_layer: int,
    target_layer: int,
    family: str,
    pos: int,
    final_row: int,
) -> dict:
    if family not in {"attn", "ffn"}:
        raise ValueError(f"unknown HC family: {family}")
    residual_hc = _reshape_hc(
        residual_name,
        _read_pos_dump(dump_prefix, residual_name, residual_layer, pos),
    )
    tokens = residual_hc.shape[0]
    if final_row >= tokens:
        raise ValueError(f"final row {final_row} is outside dump with {tokens} rows")
    x_hidden = _reshape_hidden(
        x_name,
        _read_pos_dump(dump_prefix, x_name, x_layer, pos),
        tokens,
    )

    prefix = f"layers.{target_layer}.hc_{family}"
    fn = _load_package_tensor(package_dir, f"{prefix}_fn")
    scale = _load_package_tensor(package_dir, f"{prefix}_scale")
    base = _load_package_tensor(package_dir, f"{prefix}_base")
    got_mixes, got_pre = _replay_hc_pre(residual_hc, fn, scale, base)
    del got_mixes

    from jang_tools.dsv4.mlx_model import hc_split_sinkhorn
    import mlx.core as mx

    x_flat = residual_hc.reshape(tokens, HC_DIM).astype(np.float32)
    rsqrt = 1.0 / np.sqrt(np.mean(x_flat * x_flat, axis=-1, keepdims=True) + 1.0e-6)
    mixes = (x_flat @ fn.T.astype(np.float32)) * rsqrt
    pre, post, comb = hc_split_sinkhorn(
        mx.array(mixes.reshape(1, tokens, -1), dtype=mx.float32),
        mx.array(scale, dtype=mx.float32),
        mx.array(base, dtype=mx.float32),
        N_HC,
        20,
        1.0e-6,
    )
    mx.eval(pre, post, comb)
    post_np = np.asarray(post.astype(mx.float32)).reshape(tokens, N_HC)
    comb_np = np.asarray(comb.astype(mx.float32)).reshape(tokens, N_HC, N_HC)

    ref_pre = _reshape_hidden(
        f"hc_{family}_pre",
        _read_pos_dump(dump_prefix, f"hc_{family}_pre", target_layer, pos),
        tokens,
    )
    ref_post = _reshape_hc(
        f"hc_{family}_post",
        _read_pos_dump(dump_prefix, f"hc_{family}_post", target_layer, pos),
    )
    variants = _hc_post_variants(
        x_hidden=x_hidden,
        residual_hc=residual_hc,
        post=post_np,
        comb=comb_np,
    )
    return {
        "package_dir": str(package_dir),
        "dump_prefix": str(dump_prefix),
        "residual_name": residual_name,
        "residual_layer": int(residual_layer),
        "x_name": x_name,
        "x_layer": int(x_layer),
        "target_layer": int(target_layer),
        "family": family,
        "pos": int(pos),
        "final_row": int(final_row),
        "tokens": int(tokens),
        "pre_hidden": {
            "all": _metrics(got_pre.reshape(-1), ref_pre.reshape(-1)),
            "final": _metrics(got_pre[final_row], ref_pre[final_row]),
        },
        "post_hidden": {
            name: {
                "all": _metrics(value.reshape(-1), ref_post.reshape(-1)),
                "final": _metrics(value[final_row].reshape(-1), ref_post[final_row].reshape(-1)),
            }
            for name, value in variants.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--residual-name", required=True)
    parser.add_argument("--residual-layer", required=True, type=int)
    parser.add_argument("--x-name", required=True)
    parser.add_argument("--x-layer", required=True, type=int)
    parser.add_argument("--target-layer", required=True, type=int)
    parser.add_argument("--family", choices=("attn", "ffn"), required=True)
    parser.add_argument("--pos", default=0, type=int)
    parser.add_argument("--final-row", required=True, type=int)
    args = parser.parse_args(argv)
    result = compare_hc_post_dumps(
        package_dir=args.package,
        dump_prefix=args.dump_prefix,
        residual_name=args.residual_name,
        residual_layer=args.residual_layer,
        x_name=args.x_name,
        x_layer=args.x_layer,
        target_layer=args.target_layer,
        family=args.family,
        pos=args.pos,
        final_row=args.final_row,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
