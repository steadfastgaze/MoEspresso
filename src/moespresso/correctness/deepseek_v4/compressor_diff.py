"""Compare DS4 compressor dumps against the served MLX compressor replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moespresso.correctness.deepseek_v4.rope_diff import _metrics
from moespresso.runtime.serve import load_served_model


HEAD_DIM = 512
ROT_DIM = 64


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _read_pos_dump(prefix: Path, name: str, layer: int, pos: int) -> np.ndarray:
    path = Path(f"{prefix}_{name}-{layer}_pos{pos}.bin")
    if not path.is_file():
        raise FileNotFoundError(f"missing DS4 dump: {path}")
    return np.fromfile(path, dtype=np.float32)


def _as_numpy(value) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def _layer_at(model, layer: int):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("loaded model has no model.layers")
    return layers[layer]


def _e4m3fn_values() -> np.ndarray:
    values = []
    for code in range(127):
        exp = (code >> 3) & 0x0F
        mant = code & 0x07
        if exp == 0:
            values.append(float(mant) * 0.001953125)
        else:
            values.append((1.0 + float(mant) * 0.125) * float(2 ** (exp - 7)))
    return np.array(values, dtype=np.float32)


E4M3FN_VALUES = _e4m3fn_values()


def _e4m3fn_dequant(values: np.ndarray) -> np.ndarray:
    clipped = np.minimum(np.abs(values).astype(np.float32), np.float32(448.0))
    idx = np.searchsorted(E4M3FN_VALUES, clipped, side="right") - 1
    idx = np.clip(idx, 0, E4M3FN_VALUES.size - 1)
    next_idx = np.minimum(idx + 1, E4M3FN_VALUES.size - 1)
    best = E4M3FN_VALUES[idx]
    next_best = E4M3FN_VALUES[next_idx]
    best_diff = np.abs(clipped - best)
    next_diff = np.abs(clipped - next_best)
    tie_to_even = (next_diff == best_diff) & ((next_idx & 1) == 0) & ((idx & 1) != 0)
    use_next = (next_diff < best_diff) | tie_to_even
    quantized = np.where(use_next, next_best, best)
    return np.sign(values).astype(np.float32) * quantized


def _fp8_kv_round(rows: np.ndarray) -> np.ndarray:
    out = rows.astype(np.float32).copy()
    n_nope = HEAD_DIM - ROT_DIM
    for row in out:
        for off in range(0, n_nope, 64):
            block = row[off: off + 64]
            amax = float(np.max(np.abs(block)))
            if amax < 1.0e-4:
                amax = 1.0e-4
            scale = np.float32(2.0 ** np.ceil(np.log2(amax / 448.0)))
            clipped = np.clip(block / scale, -448.0, 448.0)
            row[off: off + 64] = _e4m3fn_dequant(clipped) * scale
    return out


def _replay_attention_compressor(
    attn,
    kv_raw: np.ndarray,
    score_raw: np.ndarray,
    *,
    ape_dtype: str = "package",
) -> np.ndarray:
    import mlx.core as mx

    from jang_tools.dsv4.mlx_model import _apply_partial_rope

    comp = attn.compressor
    ratio = int(comp.compress_ratio)
    out_dim = int(comp.out_dim)
    tokens = kv_raw.size // out_dim
    if tokens * out_dim != kv_raw.size or score_raw.size != kv_raw.size:
        raise ValueError(
            f"bad compressor raw sizes: kv={kv_raw.size} score={score_raw.size} "
            f"out_dim={out_dim}"
        )
    usable = (tokens // ratio) * ratio
    if usable == 0:
        return np.zeros((0, HEAD_DIM), dtype=np.float32)

    kv = mx.array(kv_raw.reshape(1, tokens, out_dim)[:, :usable], dtype=mx.float32)
    gate = mx.array(score_raw.reshape(1, tokens, out_dim)[:, :usable], dtype=mx.float32)
    ape = comp.ape.astype(mx.float32)
    if ape_dtype == "f16":
        ape = ape.astype(mx.float16).astype(mx.float32)
    elif ape_dtype != "package":
        raise ValueError(f"unknown APE dtype variant: {ape_dtype}")
    windows = usable // ratio
    kv = kv.reshape(1, windows, ratio, out_dim)
    gate = gate.reshape(1, windows, ratio, out_dim) + ape
    if comp.overlap:
        rows = mx.zeros((1, windows, 2 * ratio, HEAD_DIM), dtype=mx.float32)
        gate_rows = mx.full(
            (1, windows, 2 * ratio, HEAD_DIM),
            -float("inf"),
            dtype=mx.float32,
        )
        rows[:, :, ratio:] = kv[:, :, :, HEAD_DIM:]
        rows[:, 1:, :ratio] = kv[:, :-1, :, :HEAD_DIM]
        gate_rows[:, :, ratio:] = gate[:, :, :, HEAD_DIM:]
        gate_rows[:, 1:, :ratio] = gate[:, :-1, :, :HEAD_DIM]
        kv = rows
        gate = gate_rows
    weights = mx.softmax(gate, axis=2, precise=True)
    pooled = (kv * weights).sum(axis=2)
    pooled = comp.norm(pooled)
    positions = mx.arange(pooled.shape[1], dtype=mx.float32) * ratio
    pooled = _apply_partial_rope(pooled[:, None], attn.compress_rope, positions=positions)
    return _as_numpy(pooled.squeeze(1)).reshape(windows, HEAD_DIM)


def compare_compressor_dumps(
    *,
    package_dir: Path,
    dump_prefix: Path,
    layers: list[int],
    pos: int,
) -> dict:
    model, _tokenizer, manifest = load_served_model(package_dir)
    rows = []
    for layer_id in layers:
        layer = _layer_at(model, layer_id)
        attn = layer.self_attn
        ratio = int(attn.compress_ratio)
        kv_raw = _read_pos_dump(dump_prefix, "attn_comp_kv_raw", layer_id, pos)
        score_raw = _read_pos_dump(dump_prefix, "attn_comp_score_raw", layer_id, pos)
        ref = _read_pos_dump(dump_prefix, "KVcompress", layer_id, pos)
        ref_rows = ref.reshape(-1, HEAD_DIM)
        ape_variants = {}
        for ape_dtype in ("package", "f16"):
            got = _replay_attention_compressor(
                attn,
                kv_raw,
                score_raw,
                ape_dtype=ape_dtype,
            )
            if got.shape != ref_rows.shape:
                raise ValueError(
                    f"compressor row mismatch for layer {layer_id}: "
                    f"MoEspresso={got.shape} DS4={ref_rows.shape}"
                )
            variants = {
                "f32": got,
                "f16": got.astype(np.float16).astype(np.float32),
                "fp8": _fp8_kv_round(got),
                "fp8_f16": _fp8_kv_round(got).astype(np.float16).astype(np.float32),
            }
            ape_variants[ape_dtype] = {
                name: _metrics(value.reshape(-1), ref_rows.reshape(-1))
                for name, value in variants.items()
            }
        rows.append({
            "layer": int(layer_id),
            "compress_ratio": ratio,
            "rows": int(ref_rows.shape[0]),
            "ape_variants": ape_variants,
        })
    return {
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "dump_prefix": str(dump_prefix),
        "pos": int(pos),
        "layers": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--layers", default="2", type=_layers)
    parser.add_argument("--pos", required=True, type=int)
    args = parser.parse_args(argv)
    result = compare_compressor_dumps(
        package_dir=args.package,
        dump_prefix=args.dump_prefix,
        layers=args.layers,
        pos=args.pos,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
