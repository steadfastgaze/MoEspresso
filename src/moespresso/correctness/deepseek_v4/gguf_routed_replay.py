"""Replay DS4 GGUF routed experts and compare against DS4 routed dumps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from moespresso.correctness.deepseek_v4.rope_diff import _metrics, _read_dump


HIDDEN_SIZE = 4096
EXPERT_SIZE = 2048
TOP_K = 6
GGUF_RECIPE_ENV = "MOESPRESSO_DS4_KQUANT_GGUF_RECIPE"


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


DEFAULT_GGUF = _path_from_env(GGUF_RECIPE_ENV)


def _read_i32_dump(prefix: Path, name: str, layer: int) -> np.ndarray:
    path = Path(f"{prefix}_{name}-{layer}_pos0.i32")
    if not path.is_file():
        raise FileNotFoundError(f"missing DS4 dump: {path}")
    return np.fromfile(path, dtype=np.int32)


def _mx_dtype(mx, name: str):
    if name == "float16":
        return mx.float16
    if name == "float32":
        return mx.float32
    raise ValueError(f"unknown activation dtype: {name}")


def _as_float_numpy(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def _load_expert_tensor(arrays: dict, codecs: dict, layer: int, projection: str):
    stem = f"blk.{layer}.ffn_{projection}_exps"
    weight_key = f"{stem}.weight"
    scale_key = f"{stem}.scales"
    if weight_key not in arrays:
        raise ValueError(f"GGUF is missing {weight_key}")
    if scale_key not in arrays:
        raise ValueError(f"GGUF is missing {scale_key}")
    codec = codecs.get(weight_key)
    if codec is None:
        raise ValueError(f"GGUF tensor {weight_key} has no K-quant codec")
    return arrays[weight_key], arrays[scale_key], codec


def _load_shared_tensor(arrays: dict, codecs: dict, layer: int, projection: str):
    stem = f"blk.{layer}.ffn_{projection}_shexp"
    weight_key = f"{stem}.weight"
    scale_key = f"{stem}.scales"
    if weight_key not in arrays:
        raise ValueError(f"GGUF is missing {weight_key}")
    if scale_key not in arrays:
        raise ValueError(f"GGUF is missing {scale_key}")
    codec = codecs.get(weight_key)
    if codec is None:
        raise ValueError(f"GGUF tensor {weight_key} has no K-quant codec")
    return arrays[weight_key], arrays[scale_key], codec


def _gather(
    *,
    kq,
    x,
    weight,
    scales,
    codec: str,
    indices,
) -> Any:
    return kq.gather_qmm(
        x,
        weight,
        scales,
        codec,
        rhs_indices=indices,
        transpose=True,
        sorted_indices=False,
    )


def _reshape_gather(value: Any, *, tokens: int, top_k: int, width: int) -> Any:
    import mlx.core as mx

    if int(value.size) != tokens * top_k * width:
        raise ValueError(
            "unexpected gather output size: "
            f"got {value.shape} size={value.size}, "
            f"expected tokens={tokens} top_k={top_k} width={width}"
        )
    return mx.reshape(value, (tokens, top_k, width))


def _stage_result(
    *,
    got: np.ndarray,
    ref: np.ndarray,
    width: int,
    tokens: int,
    final_row: int,
) -> dict[str, Any]:
    if got.size != ref.size:
        raise ValueError(f"size mismatch: replay={got.size} DS4={ref.size}")
    got_rows = got.reshape(tokens, width)
    ref_rows = ref.reshape(tokens, width)
    return {
        "all": _metrics(got, ref),
        "final": _metrics(got_rows[final_row], ref_rows[final_row]),
    }


def _compare_stage(
    *,
    dump_prefix: Path,
    layer: int,
    stage: str,
    got: np.ndarray,
    width: int,
    tokens: int,
    final_row: int,
) -> dict[str, Any]:
    ref = _read_dump(dump_prefix, stage, layer)
    dump_tokens = ref.size // width
    if dump_tokens != tokens:
        raise ValueError(
            f"token count mismatch for {stage}: replay={tokens} DS4={dump_tokens}"
        )
    return _stage_result(
        got=got,
        ref=ref,
        width=width,
        tokens=tokens,
        final_row=final_row,
    )


def replay_gguf_routed(
    *,
    gguf_path: Path,
    dump_prefix: Path,
    layer: int,
    final_row: int,
    activation_dtype: str,
    swiglu_limit: float,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_kquant as kq

    arrays, codecs, _metadata, shapes = kq.load_gguf(str(gguf_path), zero_copy=True)
    norm = _read_dump(dump_prefix, "ffn_norm", layer)
    if norm.size % HIDDEN_SIZE:
        raise ValueError(f"bad ffn_norm dump size: {norm.size}")
    tokens = norm.size // HIDDEN_SIZE
    topk = _read_i32_dump(dump_prefix, "ffn_moe_topk", layer).reshape(tokens, TOP_K)
    weights = _read_dump(dump_prefix, "ffn_moe_weights_scaled", layer).reshape(
        tokens,
        TOP_K,
    )

    dtype = _mx_dtype(mx, activation_dtype)
    x = mx.array(norm.reshape(tokens, 1, 1, HIDDEN_SIZE), dtype=dtype)
    idx = mx.array(topk.astype(np.uint32), dtype=mx.uint32)
    route_weights = mx.array(weights, dtype=dtype)

    gate_w, gate_scales, gate_codec = _load_expert_tensor(
        arrays, codecs, layer, "gate")
    up_w, up_scales, up_codec = _load_expert_tensor(arrays, codecs, layer, "up")
    down_w, down_scales, down_codec = _load_expert_tensor(
        arrays, codecs, layer, "down")
    shared_gate_w, shared_gate_scales, shared_gate_codec = _load_shared_tensor(
        arrays, codecs, layer, "gate")
    shared_up_w, shared_up_scales, shared_up_codec = _load_shared_tensor(
        arrays, codecs, layer, "up")
    shared_down_w, shared_down_scales, shared_down_codec = _load_shared_tensor(
        arrays, codecs, layer, "down")

    gate_raw = _reshape_gather(
        _gather(
            kq=kq,
            x=x,
            weight=gate_w,
            scales=gate_scales,
            codec=gate_codec,
            indices=idx,
        ),
        tokens=tokens,
        top_k=TOP_K,
        width=EXPERT_SIZE,
    )
    up_raw = _reshape_gather(
        _gather(
            kq=kq,
            x=x,
            weight=up_w,
            scales=up_scales,
            codec=up_codec,
            indices=idx,
        ),
        tokens=tokens,
        top_k=TOP_K,
        width=EXPERT_SIZE,
    )
    gate = mx.minimum(gate_raw, mx.array(swiglu_limit, dtype=gate_raw.dtype))
    up = mx.clip(up_raw, -swiglu_limit, swiglu_limit)
    mid = nn.silu(gate.astype(mx.float32)) * up.astype(mx.float32)
    mid = mid * route_weights.astype(mx.float32)[..., None]
    down = _reshape_gather(
        _gather(
            kq=kq,
            x=mx.expand_dims(mid.astype(dtype), -2),
            weight=down_w,
            scales=down_scales,
            codec=down_codec,
            indices=idx,
        ),
        tokens=tokens,
        top_k=TOP_K,
        width=HIDDEN_SIZE,
    )
    routed_out = mx.sum(down.astype(mx.float32), axis=1)
    shared_x = mx.array(norm.reshape(tokens, HIDDEN_SIZE), dtype=dtype)
    shared_gate_raw = kq.quantized_matmul(
        shared_x,
        shared_gate_w,
        shared_gate_scales,
        shared_gate_codec,
        transpose=True,
    )
    shared_up_raw = kq.quantized_matmul(
        shared_x,
        shared_up_w,
        shared_up_scales,
        shared_up_codec,
        transpose=True,
    )
    shared_gate = mx.minimum(
        shared_gate_raw,
        mx.array(swiglu_limit, dtype=shared_gate_raw.dtype),
    )
    shared_up = mx.clip(shared_up_raw, -swiglu_limit, swiglu_limit)
    shared_mid = nn.silu(shared_gate.astype(mx.float32)) * shared_up.astype(mx.float32)
    shared_out = kq.quantized_matmul(
        shared_mid.astype(dtype),
        shared_down_w,
        shared_down_scales,
        shared_down_codec,
        transpose=True,
    ).astype(mx.float32)
    ffn_out = routed_out + shared_out
    mx.eval(gate, up, mid, down, routed_out, shared_out, ffn_out)

    stage_rows = {
        "ffn_moe_gate_clamped": _compare_stage(
            dump_prefix=dump_prefix,
            layer=layer,
            stage="ffn_moe_gate_clamped",
            got=_as_float_numpy(gate).reshape(-1),
            width=TOP_K * EXPERT_SIZE,
            tokens=tokens,
            final_row=final_row,
        ),
        "ffn_moe_up_clamped": _compare_stage(
            dump_prefix=dump_prefix,
            layer=layer,
            stage="ffn_moe_up_clamped",
            got=_as_float_numpy(up).reshape(-1),
            width=TOP_K * EXPERT_SIZE,
            tokens=tokens,
            final_row=final_row,
        ),
        "ffn_moe_weighted_swiglu": _compare_stage(
            dump_prefix=dump_prefix,
            layer=layer,
            stage="ffn_moe_weighted_swiglu",
            got=_as_float_numpy(mid).reshape(-1),
            width=TOP_K * EXPERT_SIZE,
            tokens=tokens,
            final_row=final_row,
        ),
        "ffn_moe_down": _compare_stage(
            dump_prefix=dump_prefix,
            layer=layer,
            stage="ffn_moe_down",
            got=_as_float_numpy(down).reshape(-1),
            width=TOP_K * HIDDEN_SIZE,
            tokens=tokens,
            final_row=final_row,
        ),
        "ffn_moe_out": _compare_stage(
            dump_prefix=dump_prefix,
            layer=layer,
            stage="ffn_moe_out",
            got=_as_float_numpy(routed_out).reshape(-1),
            width=HIDDEN_SIZE,
            tokens=tokens,
            final_row=final_row,
        ),
        "ffn_shexp": _compare_stage(
            dump_prefix=dump_prefix,
            layer=layer,
            stage="ffn_shexp",
            got=_as_float_numpy(shared_out).reshape(-1),
            width=HIDDEN_SIZE,
            tokens=tokens,
            final_row=final_row,
        ),
        "ffn_out": _compare_stage(
            dump_prefix=dump_prefix,
            layer=layer,
            stage="ffn_out",
            got=_as_float_numpy(ffn_out).reshape(-1),
            width=HIDDEN_SIZE,
            tokens=tokens,
            final_row=final_row,
        ),
    }

    return {
        "gguf_path": str(gguf_path),
        "dump_prefix": str(dump_prefix),
        "layer": int(layer),
        "activation_dtype": activation_dtype,
        "swiglu_limit": float(swiglu_limit),
        "prompt_tokens": int(tokens),
        "final_row": int(final_row),
        "final_topk": [int(v) for v in topk[final_row]],
        "codecs": {
            "gate": gate_codec,
            "up": up_codec,
            "down": down_codec,
            "shared_gate": shared_gate_codec,
            "shared_up": shared_up_codec,
            "shared_down": shared_down_codec,
        },
        "logical_shapes": {
            "gate": list(shapes[f"blk.{layer}.ffn_gate_exps.weight"]),
            "up": list(shapes[f"blk.{layer}.ffn_up_exps.weight"]),
            "down": list(shapes[f"blk.{layer}.ffn_down_exps.weight"]),
            "shared_gate": list(shapes[f"blk.{layer}.ffn_gate_shexp.weight"]),
            "shared_up": list(shapes[f"blk.{layer}.ffn_up_shexp.weight"]),
            "shared_down": list(shapes[f"blk.{layer}.ffn_down_shexp.weight"]),
        },
        "stages": stage_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gguf",
        type=Path,
        default=DEFAULT_GGUF,
        help=f"DS4 GGUF recipe path. Defaults to ${GGUF_RECIPE_ENV}.",
    )
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--layer", default=0, type=int)
    parser.add_argument("--final-row", required=True, type=int)
    parser.add_argument(
        "--activation-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--swiglu-limit", default=10.0, type=float)
    args = parser.parse_args(argv)
    if args.gguf is None:
        parser.error(f"--gguf is required unless ${GGUF_RECIPE_ENV} is set")
    result = replay_gguf_routed(
        gguf_path=args.gguf,
        dump_prefix=args.dump_prefix,
        layer=args.layer,
        final_row=args.final_row,
        activation_dtype=args.activation_dtype,
        swiglu_limit=args.swiglu_limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
