"""Replay DS4 attention output projection from reference attention-head dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from moespresso.correctness.deepseek_v4.attention_diff import _read_pos_dump, _reshape_q
from moespresso.correctness.deepseek_v4.rope_diff import HEAD_DIM, N_HEADS, Q_DIM, _metrics
from moespresso.correctness.deepseek_v4.gguf_routed_replay import (
    DEFAULT_GGUF,
    GGUF_RECIPE_ENV,
)
from moespresso.runtime.serve import load_served_model


HIDDEN_SIZE = 4096
N_OUT_GROUPS = 8
OUT_RANK = 1024
ATTN_LOW_SIZE = N_OUT_GROUPS * OUT_RANK


def _fp16_bytes_to_float32(bytes2: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(bytes2, dtype=np.uint8)
    return packed.view("<f2").astype(np.float32)


def _quantize_q8_0_activation_np(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"expected a 2D activation array, got shape {x.shape}")
    if x.shape[1] % 32 != 0:
        raise ValueError(f"Q8_0 activation width must be divisible by 32, got {x.shape[1]}")
    rows, width = x.shape
    blocks = width // 32
    xb = np.asarray(x, dtype=np.float32).reshape(rows, blocks, 32)
    amax = np.max(np.abs(xb), axis=2)
    scale = (amax / np.float32(127.0)).astype(np.float32)
    inv = np.divide(
        np.float32(1.0),
        scale,
        out=np.zeros_like(scale, dtype=np.float32),
        where=scale != 0,
    )
    q = np.rint(xb * inv[:, :, None])
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scale


def _q8_0_weight_rows(wire_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if wire_rows.ndim != 2:
        raise ValueError(f"expected 2D Q8_0 wire rows, got shape {wire_rows.shape}")
    if wire_rows.shape[1] % 34 != 0:
        raise ValueError(
            f"Q8_0 wire-row byte width must be divisible by 34, got {wire_rows.shape[1]}"
        )
    packed = np.ascontiguousarray(wire_rows, dtype=np.uint8).reshape(
        wire_rows.shape[0], wire_rows.shape[1] // 34, 34
    )
    scales = _fp16_bytes_to_float32(packed[:, :, :2].reshape(-1, 2)).reshape(
        wire_rows.shape[0], wire_rows.shape[1] // 34
    )
    qs = packed[:, :, 2:].view(np.int8)
    return qs, scales


def _q8_0_ds4_activation_matmul(x: np.ndarray, wire_rows: np.ndarray) -> np.ndarray:
    """Replay DS4's Q8_0 activation-dot against row-major GGUF Q8_0 weights."""

    xq, xscale = _quantize_q8_0_activation_np(x)
    wq, wscale = _q8_0_weight_rows(wire_rows)
    if xq.shape[1:] != wq.shape[1:]:
        raise ValueError(
            "activation/weight Q8_0 block mismatch: "
            f"activation blocks {xq.shape[1:]}, weight blocks {wq.shape[1:]}"
        )

    out = np.zeros((xq.shape[0], wq.shape[0]), dtype=np.float32)
    for block in range(xq.shape[1]):
        dots = xq[:, block, :].astype(np.int32) @ wq[:, block, :].astype(np.int32).T
        out += (
            dots.astype(np.float32)
            * xscale[:, block, None]
            * wscale[None, :, block]
        )
    return out


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _as_float_numpy(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def _layer_at(model, layer: int):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("loaded model has no model.layers")
    return layers[layer]


def _load_q8(arrays: dict, codecs: dict, layer: int, projection: str):
    stem = f"blk.{layer}.attn_output_{projection}"
    weight_key = f"{stem}.weight"
    scale_key = f"{stem}.scales"
    if weight_key not in arrays:
        raise ValueError(f"GGUF is missing {weight_key}")
    if scale_key not in arrays:
        raise ValueError(f"GGUF is missing {scale_key}")
    codec = codecs.get(weight_key)
    if codec != "q8_0":
        raise ValueError(f"{weight_key}: expected q8_0, got {codec!r}")
    return arrays[weight_key], arrays[scale_key], codec


def _stage_result(
    *,
    got: np.ndarray,
    ref: np.ndarray,
    width: int,
    final_row: int,
) -> dict[str, Any]:
    if got.size != ref.size:
        raise ValueError(f"size mismatch: replay={got.size} DS4={ref.size}")
    tokens = ref.size // width
    got_rows = got.reshape(tokens, width)
    ref_rows = ref.reshape(tokens, width)
    return {
        "all": _metrics(got, ref),
        "final": _metrics(got_rows[final_row], ref_rows[final_row]),
    }


def replay_attention_output(
    *,
    package_dir: Path,
    gguf_path: Path,
    dump_prefix: Path,
    layers: list[int],
    final_row: int,
    pos: int,
    include_ds4_q8: bool,
    include_dequant_f32: bool,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx_kquant as kq
    from jang_tools.dsv4.mlx_model import _apply_partial_rope

    model, _tokenizer, manifest = load_served_model(package_dir)
    arrays, codecs, _metadata, shapes = kq.load_gguf(str(gguf_path), zero_copy=True)

    rows = []
    group_dim = Q_DIM // N_OUT_GROUPS
    for layer_id in layers:
        layer = _layer_at(model, layer_id)
        attn = layer.self_attn
        q_heads = _reshape_q(
            "kqv_out",
            _read_pos_dump(dump_prefix, "kqv_out", layer_id, pos),
            n_heads=N_HEADS,
            head_dim=HEAD_DIM,
        )
        tokens = q_heads.shape[0]
        if final_row >= tokens:
            raise ValueError(
                f"final row {final_row} is outside layer {layer_id} dump with {tokens} rows"
            )
        heads = mx.array(q_heads.transpose(1, 0, 2)[None], dtype=mx.float32)
        heads_back = _apply_partial_rope(
            heads,
            attn.rope,
            offset=0,
            inverse=True,
        )
        flat = heads_back.transpose(0, 2, 1, 3).reshape(1, tokens, Q_DIM)

        weight_a, scales_a, codec_a = _load_q8(arrays, codecs, layer_id, "a")
        weight_b, scales_b, codec_b = _load_q8(arrays, codecs, layer_id, "b")
        pieces = []
        exact_pieces = []
        dequant_f32_pieces = []
        flat_np = None
        if include_ds4_q8:
            flat_np = _as_float_numpy(flat).reshape(tokens, Q_DIM)
        for group in range(N_OUT_GROUPS):
            row_start = group * OUT_RANK
            row_end = row_start + OUT_RANK
            col_start = group * group_dim
            col_end = col_start + group_dim
            pieces.append(
                kq.quantized_matmul(
                    flat[:, :, col_start:col_end],
                    weight_a[row_start:row_end],
                    scales_a,
                    codec_a,
                    transpose=True,
                )
            )
            if flat_np is not None:
                exact_pieces.append(
                    _q8_0_ds4_activation_matmul(
                        flat_np[:, col_start:col_end],
                        np.asarray(weight_a[row_start:row_end], dtype=np.uint8),
                    )
                )
            if include_dequant_f32:
                deq_a = kq.dequantize(
                    weight_a[row_start:row_end],
                    scales_a,
                    codec_a,
                    dtype=mx.float32,
                )
                dequant_f32_pieces.append(
                    mx.matmul(flat[:, :, col_start:col_end], deq_a.T)
                )
        low = mx.concatenate(pieces, axis=-1)
        out = kq.quantized_matmul(
            low,
            weight_b,
            scales_b,
            codec_b,
            transpose=True,
        )
        dequant_f32_low = None
        dequant_f32_out = None
        if dequant_f32_pieces:
            dequant_f32_low = mx.concatenate(dequant_f32_pieces, axis=-1)
            deq_b = kq.dequantize(weight_b, scales_b, codec_b, dtype=mx.float32)
            dequant_f32_out = mx.matmul(dequant_f32_low, deq_b.T)
            mx.eval(dequant_f32_low, dequant_f32_out)
        mx.eval(heads_back, low, out)

        stages = {
            "kqv_back": _stage_result(
                got=_as_float_numpy(heads_back)[0].transpose(1, 0, 2).reshape(-1),
                ref=_read_pos_dump(dump_prefix, "kqv_back", layer_id, pos),
                width=Q_DIM,
                final_row=final_row,
            ),
            "attn_low": _stage_result(
                got=_as_float_numpy(low).reshape(-1),
                ref=_read_pos_dump(dump_prefix, "attn_low", layer_id, pos),
                width=ATTN_LOW_SIZE,
                final_row=final_row,
            ),
            "attn_out": _stage_result(
                got=_as_float_numpy(out).reshape(-1),
                ref=_read_pos_dump(dump_prefix, "attn_out", layer_id, pos),
                width=HIDDEN_SIZE,
                final_row=final_row,
            ),
        }
        if exact_pieces:
            exact_low_np = np.concatenate(exact_pieces, axis=1)
            exact_out_np = _q8_0_ds4_activation_matmul(
                exact_low_np,
                np.asarray(weight_b, dtype=np.uint8),
            )
            stages["attn_low_ds4_q8"] = _stage_result(
                got=exact_low_np.reshape(-1),
                ref=_read_pos_dump(dump_prefix, "attn_low", layer_id, pos),
                width=ATTN_LOW_SIZE,
                final_row=final_row,
            )
            stages["attn_out_ds4_q8"] = _stage_result(
                got=exact_out_np.reshape(-1),
                ref=_read_pos_dump(dump_prefix, "attn_out", layer_id, pos),
                width=HIDDEN_SIZE,
                final_row=final_row,
            )
        if dequant_f32_low is not None and dequant_f32_out is not None:
            stages["attn_low_dequant_f32"] = _stage_result(
                got=_as_float_numpy(dequant_f32_low).reshape(-1),
                ref=_read_pos_dump(dump_prefix, "attn_low", layer_id, pos),
                width=ATTN_LOW_SIZE,
                final_row=final_row,
            )
            stages["attn_out_dequant_f32"] = _stage_result(
                got=_as_float_numpy(dequant_f32_out).reshape(-1),
                ref=_read_pos_dump(dump_prefix, "attn_out", layer_id, pos),
                width=HIDDEN_SIZE,
                final_row=final_row,
            )
        rows.append({
            "layer": int(layer_id),
            "tokens": int(tokens),
            "group_dim": int(group_dim),
            "codecs": {
                "attn_output_a": codec_a,
                "attn_output_b": codec_b,
            },
            "matmul_dtypes": {
                "mlx_kquant_low": str(low.dtype),
                "mlx_kquant_out": str(out.dtype),
                "dequant_f32_low": None
                if dequant_f32_low is None
                else str(dequant_f32_low.dtype),
                "dequant_f32_out": None
                if dequant_f32_out is None
                else str(dequant_f32_out.dtype),
            },
            "logical_shapes": {
                "attn_output_a": list(shapes[f"blk.{layer_id}.attn_output_a.weight"]),
                "attn_output_b": list(shapes[f"blk.{layer_id}.attn_output_b.weight"]),
            },
            "stages": stages,
        })

    return {
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "gguf_path": str(gguf_path),
        "dump_prefix": str(dump_prefix),
        "pos": int(pos),
        "final_row": int(final_row),
        "layers": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument(
        "--gguf",
        type=Path,
        default=DEFAULT_GGUF,
        help=f"DS4 GGUF recipe path. Defaults to ${GGUF_RECIPE_ENV}.",
    )
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--layers", default="2", type=_layers)
    parser.add_argument("--final-row", required=True, type=int)
    parser.add_argument("--pos", default=0, type=int)
    parser.add_argument(
        "--include-ds4-q8",
        action="store_true",
        help="also replay DS4's activation-quantized Q8_0 dot, not only x @ dequant(w)",
    )
    parser.add_argument(
        "--include-dequant-f32",
        action="store_true",
        help="also replay Q8_0 as explicit fp32 dequantized weights plus mx.matmul",
    )
    args = parser.parse_args(argv)
    if args.gguf is None:
        parser.error(f"--gguf is required unless ${GGUF_RECIPE_ENV} is set")
    result = replay_attention_output(
        package_dir=args.package,
        gguf_path=args.gguf,
        dump_prefix=args.dump_prefix,
        layers=args.layers,
        final_row=args.final_row,
        pos=args.pos,
        include_ds4_q8=args.include_ds4_q8,
        include_dequant_f32=args.include_dequant_f32,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
