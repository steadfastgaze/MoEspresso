"""Replay DS4 raw attention dumps against the served MLX attention contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moespresso.correctness.deepseek_v4.rope_diff import _metrics


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _read_pos_dump(prefix: Path, name: str, layer: int, pos: int) -> np.ndarray:
    path = Path(f"{prefix}_{name}-{layer}_pos{pos}.bin")
    if not path.is_file():
        raise FileNotFoundError(f"missing DS4 dump: {path}")
    return np.fromfile(path, dtype=np.float32)


def _load_package_sink(package_dir: Path, layer: int) -> np.ndarray:
    from safetensors import safe_open

    key = f"layers.{layer}.attn.attn_sink"
    for shard in sorted(Path(package_dir).glob("model-*.safetensors")):
        with safe_open(shard, framework="np") as handle:
            if key in handle.keys():
                return np.asarray(handle.get_tensor(key), dtype=np.float32)
    raise FileNotFoundError(f"missing package tensor {key} under {package_dir}")


def _half_to_float(value: np.ndarray) -> np.ndarray:
    return value.astype(np.float16).astype(np.float32)


def _attention_one(
    query: np.ndarray,
    keys: np.ndarray,
    sink: float,
    *,
    scale: float,
    qk_half: bool,
    value_half: bool,
) -> np.ndarray:
    q = _half_to_float(query) if qk_half else query.astype(np.float32)
    k = _half_to_float(keys) if qk_half else keys.astype(np.float32)
    v = _half_to_float(keys) if value_half else keys.astype(np.float32)
    scores = k @ q * np.float32(scale)
    sink_score = np.float32(sink)
    max_score = np.maximum(np.max(scores), sink_score)
    weights = np.exp(scores - max_score, dtype=np.float32)
    denom = np.sum(weights, dtype=np.float32) + np.exp(
        sink_score - max_score,
        dtype=np.float32,
    )
    if denom == 0:
        return np.zeros((keys.shape[-1],), dtype=np.float32)
    return (weights.astype(np.float32) @ v.astype(np.float32)) / denom


def raw_prefill_attention(
    qcur: np.ndarray,
    kvcur: np.ndarray,
    sinks: np.ndarray,
    *,
    scale: float,
    window: int,
    qk_half: bool,
    value_half: bool,
) -> np.ndarray:
    if qcur.ndim != 3:
        raise ValueError(f"qcur must have shape (tokens, heads, dim), got {qcur.shape}")
    if kvcur.ndim != 2:
        raise ValueError(f"kvcur must have shape (tokens, dim), got {kvcur.shape}")
    tokens, n_heads, head_dim = qcur.shape
    if kvcur.shape != (tokens, head_dim):
        raise ValueError(f"KV shape {kvcur.shape} does not match Q shape {qcur.shape}")
    if sinks.shape != (n_heads,):
        raise ValueError(f"sink shape {sinks.shape} does not match {n_heads} heads")

    out = np.empty((tokens, n_heads, head_dim), dtype=np.float32)
    for token in range(tokens):
        start = 0 if window <= 0 else max(0, token + 1 - int(window))
        keys = kvcur[start : token + 1]
        for head in range(n_heads):
            out[token, head] = _attention_one(
                qcur[token, head],
                keys,
                float(sinks[head]),
                scale=scale,
                qk_half=qk_half,
                value_half=value_half,
            )
    return out


def _raw_visibility(*, tokens: int, window: int) -> np.ndarray:
    mask = np.zeros((tokens, tokens), dtype=np.bool_)
    for query in range(tokens):
        for key in range(tokens):
            causal = key <= query
            in_window = window <= 0 or query - key < window
            mask[query, key] = causal and in_window
    return mask.reshape(1, 1, tokens, tokens)


def mlx_raw_prefill_attention(
    qcur: np.ndarray,
    kvcur: np.ndarray,
    sinks: np.ndarray,
    *,
    scale: float,
    window: int,
    mask_kind: str,
) -> np.ndarray:
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention

    tokens, heads, head_dim = qcur.shape
    q = mx.array(qcur.transpose(1, 0, 2)[None], dtype=mx.float16)
    kv = mx.array(kvcur.reshape(1, 1, tokens, head_dim), dtype=mx.float16)
    if mask_kind == "bool":
        mask = mx.array(_raw_visibility(tokens=tokens, window=window))
    elif mask_kind == "causal":
        mask = "causal"
    else:
        raise ValueError(f"unknown MLX attention mask kind: {mask_kind}")
    out = scaled_dot_product_attention(
        q,
        kv,
        kv,
        None,
        scale=scale,
        mask=mask,
        sinks=mx.array(sinks, dtype=mx.float16),
    )
    mx.eval(out)
    return np.asarray(out.astype(mx.float32))[0].transpose(1, 0, 2)


def _reshape_q(name: str, values: np.ndarray, *, n_heads: int, head_dim: int) -> np.ndarray:
    width = n_heads * head_dim
    if values.size % width:
        raise ValueError(f"bad {name} dump size: {values.size}")
    return values.reshape(values.size // width, n_heads, head_dim)


def _reshape_kv(name: str, values: np.ndarray, *, tokens: int, head_dim: int) -> np.ndarray:
    if values.size != tokens * head_dim:
        raise ValueError(
            f"bad {name} dump size: {values.size}; expected {tokens * head_dim}"
        )
    return values.reshape(tokens, head_dim)


def compare_attention_dumps(
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
        if ratio:
            raise ValueError(
                "raw attention replay only supports compress_ratio=0 layers; "
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
        if ref.shape != qcur.shape:
            raise ValueError(
                f"kqv_out shape {ref.shape} does not match Q shape {qcur.shape}"
            )

        sinks = _load_package_sink(package_dir, layer)
        variants = {}
        for name, qk_half, value_half in (
            ("fp32_qkv", False, False),
            ("f16_qk_fp32_v", True, False),
            ("f16_qkv", True, True),
        ):
            got = raw_prefill_attention(
                qcur,
                kvcur,
                sinks,
                scale=scale,
                window=window,
                qk_half=qk_half,
                value_half=value_half,
            )
            variants[name] = {
                "all": _metrics(got.reshape(-1), ref.reshape(-1)),
                "final": _metrics(
                    got[final_row].reshape(-1),
                    ref[final_row].reshape(-1),
                ),
            }
        for mask_kind in ("bool", "causal"):
            mlx_got = mlx_raw_prefill_attention(
                qcur,
                kvcur,
                sinks,
                scale=scale,
                window=window,
                mask_kind=mask_kind,
            )
            variants[f"mlx_sdpa_f16_qkv_{mask_kind}_mask"] = {
                "all": _metrics(mlx_got.reshape(-1), ref.reshape(-1)),
                "final": _metrics(
                    mlx_got[final_row].reshape(-1),
                    ref[final_row].reshape(-1),
                ),
            }

        rows.append({
            "layer": int(layer),
            "compress_ratio": ratio,
            "tokens": int(tokens),
            "variants": variants,
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
    parser.add_argument("--layers", default="0,1,2", type=_layers)
    parser.add_argument("--final-row", required=True, type=int)
    parser.add_argument("--pos", default=0, type=int)
    args = parser.parse_args(argv)
    result = compare_attention_dumps(
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
