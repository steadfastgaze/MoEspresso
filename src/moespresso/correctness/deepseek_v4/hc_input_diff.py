"""Replay DS4 mHC pre-combine inputs from reference dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moespresso.correctness.deepseek_v4.rope_diff import _metrics


HIDDEN_SIZE = 4096
N_HC = 4
HC_DIM = HIDDEN_SIZE * N_HC
MIX_HC = (2 + N_HC) * N_HC
RMS_EPS = 1.0e-6
HC_EPS = 1.0e-6
SINKHORN_ITERS = 20


def _read_pos_dump(prefix: Path, name: str, layer: int, pos: int) -> np.ndarray:
    path = Path(f"{prefix}_{name}-{layer}_pos{pos}.bin")
    if not path.is_file():
        raise FileNotFoundError(f"missing DS4 dump: {path}")
    return np.fromfile(path, dtype=np.float32)


def _load_package_tensor(package_dir: Path, key: str) -> np.ndarray:
    from safetensors import safe_open

    for shard in sorted(Path(package_dir).glob("model-*.safetensors")):
        with safe_open(shard, framework="np") as handle:
            if key in handle.keys():
                return np.asarray(handle.get_tensor(key), dtype=np.float32)
    raise FileNotFoundError(f"missing package tensor {key} under {package_dir}")


def _reshape_hc(name: str, values: np.ndarray) -> np.ndarray:
    if values.size % HC_DIM:
        raise ValueError(f"bad {name} dump size: {values.size}")
    return values.reshape(values.size // HC_DIM, N_HC, HIDDEN_SIZE)


def _reshape_hidden(name: str, values: np.ndarray, tokens: int) -> np.ndarray:
    if values.size != tokens * HIDDEN_SIZE:
        raise ValueError(
            f"bad {name} dump size: {values.size}; expected {tokens * HIDDEN_SIZE}"
        )
    return values.reshape(tokens, HIDDEN_SIZE)


def _reshape_mixes(name: str, values: np.ndarray, tokens: int) -> np.ndarray:
    if values.size != tokens * MIX_HC:
        raise ValueError(
            f"bad {name} dump size: {values.size}; expected {tokens * MIX_HC}"
        )
    return values.reshape(tokens, MIX_HC)


def _replay_hc_pre(
    x_hc: np.ndarray,
    fn: np.ndarray,
    scale: np.ndarray,
    base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import mlx.core as mx

    from jang_tools.dsv4.mlx_model import hc_split_sinkhorn

    tokens = x_hc.shape[0]
    x_flat = x_hc.reshape(tokens, HC_DIM).astype(np.float32)
    rsqrt = 1.0 / np.sqrt(np.mean(x_flat * x_flat, axis=-1, keepdims=True) + RMS_EPS)
    mixes = (x_flat @ fn.T.astype(np.float32)) * rsqrt

    pre, _post, _comb = hc_split_sinkhorn(
        mx.array(mixes.reshape(1, tokens, MIX_HC), dtype=mx.float32),
        mx.array(scale, dtype=mx.float32),
        mx.array(base, dtype=mx.float32),
        N_HC,
        SINKHORN_ITERS,
        HC_EPS,
    )
    y = mx.sum(
        pre[..., None] * mx.array(x_hc.reshape(1, tokens, N_HC, HIDDEN_SIZE)),
        axis=2,
    )
    mx.eval(y)
    return mixes.astype(np.float32), np.asarray(y.astype(mx.float32)).reshape(
        tokens,
        HIDDEN_SIZE,
    )


def compare_hc_input_dumps(
    *,
    package_dir: Path,
    dump_prefix: Path,
    input_name: str,
    input_layer: int,
    target_layer: int,
    family: str,
    pos: int,
    final_row: int,
) -> dict:
    if family not in {"attn", "ffn"}:
        raise ValueError(f"unknown HC family: {family}")
    x_hc = _reshape_hc(
        input_name,
        _read_pos_dump(dump_prefix, input_name, input_layer, pos),
    )
    tokens = x_hc.shape[0]
    if final_row >= tokens:
        raise ValueError(f"final row {final_row} is outside dump with {tokens} rows")

    prefix = f"layers.{target_layer}.hc_{family}"
    fn = _load_package_tensor(package_dir, f"{prefix}_fn")
    scale = _load_package_tensor(package_dir, f"{prefix}_scale")
    base = _load_package_tensor(package_dir, f"{prefix}_base")
    if fn.shape != (MIX_HC, HC_DIM):
        raise ValueError(f"bad {prefix}_fn shape: {fn.shape}")

    got_mixes, got_hidden = _replay_hc_pre(x_hc, fn, scale, base)
    ref_mixes = None
    try:
        ref_mixes = _reshape_mixes(
            f"hc_{family}_pre_mixes",
            _read_pos_dump(dump_prefix, f"hc_{family}_pre_mixes", target_layer, pos),
            tokens,
        )
    except FileNotFoundError:
        pass
    ref_hidden = _reshape_hidden(
        f"hc_{family}_pre",
        _read_pos_dump(dump_prefix, f"hc_{family}_pre", target_layer, pos),
        tokens,
    )
    return {
        "package_dir": str(package_dir),
        "dump_prefix": str(dump_prefix),
        "input_name": input_name,
        "input_layer": int(input_layer),
        "target_layer": int(target_layer),
        "family": family,
        "pos": int(pos),
        "final_row": int(final_row),
        "tokens": int(tokens),
        "mixes": None if ref_mixes is None else {
            "all": _metrics(got_mixes.reshape(-1), ref_mixes.reshape(-1)),
            "final": _metrics(got_mixes[final_row], ref_mixes[final_row]),
        },
        "hidden": {
            "all": _metrics(got_hidden.reshape(-1), ref_hidden.reshape(-1)),
            "final": _metrics(got_hidden[final_row], ref_hidden[final_row]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--input-name", required=True)
    parser.add_argument("--input-layer", required=True, type=int)
    parser.add_argument("--target-layer", required=True, type=int)
    parser.add_argument("--family", choices=("attn", "ffn"), required=True)
    parser.add_argument("--pos", default=0, type=int)
    parser.add_argument("--final-row", required=True, type=int)
    args = parser.parse_args(argv)
    result = compare_hc_input_dumps(
        package_dir=args.package,
        dump_prefix=args.dump_prefix,
        input_name=args.input_name,
        input_layer=args.input_layer,
        target_layer=args.target_layer,
        family=args.family,
        pos=args.pos,
        final_row=args.final_row,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
