"""Replay one DS4 MoE layer through the direct and pooled-block formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import numpy as np


def _layers(model) -> list[Any]:
    candidates = (
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("layers",),
    )
    for path in candidates:
        cur = model
        for name in path:
            cur = getattr(cur, name, None)
            if cur is None:
                break
        if cur is not None:
            return list(cur)
    raise ValueError("could not find decoder layers on model")


def _metrics(a: mx.array, b: mx.array) -> dict[str, Any]:
    mx.eval(a, b)
    an = np.asarray(a, dtype=np.float32)
    bn = np.asarray(b, dtype=np.float32)
    diff = bn - an
    rms = float(np.sqrt(np.mean(diff * diff)))
    ref_rms = float(np.sqrt(np.mean(an * an)))
    return {
        "shape": list(an.shape),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms": rms,
        "rel_rms": rms / ref_rms if ref_rms else 0.0,
        "array_equal": bool(np.array_equal(an, bn)),
    }


def _make_inputs(
    *,
    hidden_size: int,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch_size, seq_len, hidden_size)).astype(np.float16)
    ids = (
        np.arange(batch_size * seq_len, dtype=np.int32).reshape(batch_size, seq_len)
        + seed
    ) % max(1, vocab_size)
    return mx.array(x), mx.array(ids.astype(np.int32))


def compare_direct_and_pooled_block(
    mlp,
    *,
    x: mx.array,
    input_ids: mx.array,
) -> dict[str, Any]:
    from moespresso.runtime.pooled_switchglu import PooledDeepseekV4MoEBlock

    direct = mlp(x, input_ids=input_ids)
    wrapped = PooledDeepseekV4MoEBlock(mlp)
    wrapped.eval()
    candidate = wrapped(x, input_ids=input_ids)
    metrics = _metrics(direct, candidate)
    metrics["direct_type"] = type(mlp).__name__
    metrics["candidate_type"] = type(wrapped).__name__
    return metrics


def run_replay(
    *,
    package_dir: Path,
    layer: int,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    from moespresso.runtime.serve import load_served_model
    from moespresso.runtime.ssd_streaming_build import ssd_streaming_stats

    model, _tokenizer, manifest = load_served_model(package_dir)
    layers = _layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"layer {layer} out of range for {len(layers)} layers")
    mlp = getattr(layers[layer], "mlp", None)
    if mlp is None:
        raise ValueError(f"layer {layer} has no mlp")
    for name in ("gate", "switch_mlp", "shared_experts"):
        if not hasattr(mlp, name):
            raise ValueError(f"layer {layer} mlp is not a DS4 MoE: missing {name}")
    hidden_size = int(getattr(mlp.switch_mlp.gate_proj, "in_features"))
    architecture = manifest.get("architecture") or {}
    config = architecture.get("config") or {}
    vocab_size = int(config.get("vocab_size") or 1)
    x, input_ids = _make_inputs(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed,
    )
    metrics = compare_direct_and_pooled_block(mlp, x=x, input_ids=input_ids)
    return {
        "artifact_kind": "deepseek_v4_moe_block_replay",
        "package_manifest_id": manifest.get("artifact_id"),
        "package_family": architecture.get("family"),
        "layer": int(layer),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "seed": int(seed),
        "input_shape": list(x.shape),
        "input_ids_shape": list(input_ids.shape),
        "metrics": metrics,
        "ssd_streaming_stats": ssd_streaming_stats(model),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-moe-block-replay",
        description=(
            "Compare one loaded DS4 MoE layer through the live direct formula "
            "and the opt-in pooled block wrapper."
        ),
    )
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.seq_len <= 0:
        parser.error("--seq-len must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    payload = run_replay(
        package_dir=args.package,
        layer=args.layer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out is None:
        print(text)
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
