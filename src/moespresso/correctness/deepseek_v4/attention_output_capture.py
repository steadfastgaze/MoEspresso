"""Compare served DS4 layer-0 attention output stages against DS4 dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from moespresso.correctness.deepseek_v4.rope_diff import (
    HEAD_DIM,
    N_HEADS,
    Q_DIM,
    _metrics,
    _read_dump,
)
from moespresso.correctness.deepseek_v4.short_probe import (
    _DEFAULT_GGUF,
    _DEFAULT_SOURCE,
    _patch_attention_projection_arm,
    _patch_prompt_embedding_rows,
    _validate_probe_input_paths,
)
from moespresso.runtime.deepseek_v4.renderer import DEEPSEEK_V4_PROMPT_RENDERER
from moespresso.runtime.http import render_prompt
from moespresso.runtime.serve import load_served_model


HIDDEN_SIZE = 4096
ATTN_LOW_SIZE = 8192


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _key(layer: int, name: str) -> str:
    return f"{layer}:{name}"


def _as_float_numpy(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def _flatten_q(value: Any) -> np.ndarray:
    arr = _as_float_numpy(value)
    if arr.ndim != 4 or arr.shape[1] != N_HEADS or arr.shape[-1] != HEAD_DIM:
        raise ValueError(f"unexpected Q-like capture shape: {arr.shape}")
    return arr[0].transpose(1, 0, 2).reshape(-1)


def _flatten_hidden(value: Any, width: int) -> np.ndarray:
    arr = _as_float_numpy(value)
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[-1] != width:
        raise ValueError(f"unexpected hidden capture shape: {arr.shape}")
    return arr[0].reshape(-1)


def _apply_mask(scores, mask):
    import mlx.core as mx

    if mask is None:
        return scores
    if isinstance(mask, str) and mask == "causal":
        queries = scores.shape[-2]
        keys = scores.shape[-1]
        q_pos = mx.arange(queries) + (keys - queries)
        k_pos = mx.arange(keys)
        visible = k_pos[None, :] <= q_pos[:, None]
        return mx.where(visible[None, None], scores, -mx.inf)
    if getattr(mask, "dtype", None) == mx.bool_:
        return mx.where(mask, scores, -mx.inf)
    return scores + mask


def _install_attention_output_capture(
    model,
    layers: list[int],
    captures: dict[str, np.ndarray],
    *,
    kv_cache_arm: str,
):
    import mlx.core as mx
    import jang_tools.dsv4.mlx_model as dsv4_model

    targets = set(int(layer) for layer in layers)
    original = dsv4_model.DeepseekV4Attention.__call__

    def traced(self, x, mask=None, cache=None):
        layer = int(self.layer_id)
        if layer not in targets:
            return original(self, x, mask=mask, cache=cache)
        if int(getattr(self, "compress_ratio", 0)):
            raise ValueError(
                "attention output capture only supports non-compressed layers"
            )

        batch, length, _ = x.shape
        local_cache = cache
        offset = local_cache.offset if local_cache is not None else 0

        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(
            batch,
            length,
            self.n_heads,
            self.head_dim,
        )
        q = mx.fast.rms_norm(
            q,
            weight=dsv4_model._get_q_norm_ones(self.head_dim, q.dtype),
            eps=self.args.rms_norm_eps,
        ).transpose(0, 2, 1, 3)
        kv = self.kv_norm(self.wkv(x)).reshape(
            batch,
            length,
            1,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

        q = dsv4_model._apply_partial_rope(q, self.rope, offset)
        kv = dsv4_model._apply_partial_rope(kv, self.rope, offset)
        if kv_cache_arm == "ds4_fp8_roundtrip":
            from moespresso.runtime.deepseek_v4.model import (
                _deepseek_v4_fp8_kv_roundtrip,
            )

            kv = _deepseek_v4_fp8_kv_roundtrip(kv)
        elif kv_cache_arm != "direct":
            raise ValueError(f"unknown KV cache arm: {kv_cache_arm}")
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, kv)
        attn_mask = mask
        if attn_mask is not None:
            if attn_mask.shape[-1] > kv.shape[2]:
                attn_mask = attn_mask[..., -kv.shape[2]:]
            elif kv.shape[2] > attn_mask.shape[-1]:
                if getattr(attn_mask, "dtype", None) == mx.bool_:
                    pad = mx.ones(
                        attn_mask.shape[:-1] + (kv.shape[2] - attn_mask.shape[-1],),
                        dtype=mx.bool_,
                    )
                else:
                    pad = mx.zeros(
                        attn_mask.shape[:-1] + (kv.shape[2] - attn_mask.shape[-1],),
                        dtype=attn_mask.dtype,
                    )
                attn_mask = mx.concatenate([attn_mask, pad], axis=-1)

        heads = dsv4_model.scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=local_cache,
            scale=self.softmax_scale,
            mask=attn_mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        captures[_key(layer, "kqv_out")] = _flatten_q(heads)
        heads = dsv4_model._apply_partial_rope(heads, self.rope, offset, inverse=True)
        captures[_key(layer, "kqv_back")] = _flatten_q(heads)
        flat = heads.transpose(0, 2, 1, 3).reshape(
            batch,
            length,
            self.n_heads * self.head_dim,
        )
        low = self._grouped_output_projection(flat)
        captures[_key(layer, "attn_low")] = _flatten_hidden(
            low,
            int(self.o_groups * self.o_lora_rank),
        )
        out = self.wo_b(low)
        captures[_key(layer, "attn_out")] = _flatten_hidden(out, HIDDEN_SIZE)
        return out

    dsv4_model.DeepseekV4Attention.__call__ = traced
    return original


def _restore_attention_output_capture(original) -> None:
    import jang_tools.dsv4.mlx_model as dsv4_model

    dsv4_model.DeepseekV4Attention.__call__ = original


def _stage_result(
    *,
    got: np.ndarray,
    ref: np.ndarray,
    width: int,
    tokens: int,
    final_row: int,
) -> dict[str, Any]:
    if got.size != ref.size:
        raise ValueError(f"size mismatch: MoEspresso={got.size} DS4={ref.size}")
    got_rows = got.reshape(tokens, width)
    ref_rows = ref.reshape(tokens, width)
    return {
        "all": _metrics(got, ref),
        "final": _metrics(got_rows[final_row], ref_rows[final_row]),
    }


def compare_attention_output_dumps(
    *,
    package_dir: Path,
    prompt_path: Path,
    dump_prefix: Path,
    layers: list[int],
    final_row: int,
    embedding_arm: str,
    attention_projection_arm: str,
    kv_cache_arm: str,
    source_dir: Path,
    gguf_path: Path,
) -> dict[str, Any]:
    import mlx.core as mx

    model, tokenizer, manifest = load_served_model(package_dir)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    rendered = render_prompt(
        [{"role": "user", "content": prompt_text}],
        tokenizer,
        template_kwargs={"enable_thinking": False},
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    )
    tokens = tokenizer.encode(rendered)
    input_ids = mx.array(tokens, dtype=mx.int32)[None]
    if embedding_arm == "source_prompt_rows":
        _patch_prompt_embedding_rows(model, source_dir, tokens)
    elif embedding_arm != "default":
        raise ValueError(f"unknown embedding arm: {embedding_arm}")
    _patch_attention_projection_arm(
        model, attention_projection_arm, source_dir, gguf_path)

    captures: dict[str, np.ndarray] = {}
    original = _install_attention_output_capture(
        model,
        layers,
        captures,
        kv_cache_arm=kv_cache_arm,
    )
    try:
        logits = model(input_ids)
        mx.eval(logits)
    finally:
        _restore_attention_output_capture(original)

    widths = {
        "kqv_out": Q_DIM,
        "kqv_back": Q_DIM,
        "attn_low": ATTN_LOW_SIZE,
        "attn_out": HIDDEN_SIZE,
    }
    rows = []
    for layer in layers:
        stage_rows: dict[str, Any] = {}
        for stage, width in widths.items():
            ref = _read_dump(dump_prefix, stage, layer)
            tokens_in_dump = ref.size // width
            if tokens_in_dump != len(tokens):
                raise ValueError(
                    f"token count mismatch for {stage} layer {layer}: "
                    f"MoEspresso={len(tokens)} DS4={tokens_in_dump}"
                )
            stage_rows[stage] = _stage_result(
                got=captures[_key(layer, stage)],
                ref=ref,
                width=width,
                tokens=len(tokens),
                final_row=final_row,
            )
        rows.append({
            "layer": int(layer),
            "tokens": len(tokens),
            "stages": stage_rows,
        })

    return {
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "embedding_arm": embedding_arm,
        "attention_projection_arm": attention_projection_arm,
        "kv_cache_arm": kv_cache_arm,
        "prompt_tokens": len(tokens),
        "final_row": int(final_row),
        "layers": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--layers", default="0", type=_layers)
    parser.add_argument("--final-row", required=True, type=int)
    parser.add_argument(
        "--embedding-arm",
        choices=("default", "source_prompt_rows"),
        default="default",
    )
    parser.add_argument(
        "--attention-projection-arm",
        choices=(
            "default",
            "source_layer0_all",
            "source_layers0_3_all",
            "ds4_gguf_layer0_all",
        ),
        default="default",
    )
    parser.add_argument(
        "--kv-cache-arm",
        choices=("direct", "ds4_fp8_roundtrip"),
        default="direct",
    )
    parser.add_argument("--source-dir", type=Path, default=_DEFAULT_SOURCE)
    parser.add_argument("--gguf-path", type=Path, default=_DEFAULT_GGUF)
    args = parser.parse_args(argv)
    try:
        _validate_probe_input_paths(
            embedding_arm=args.embedding_arm,
            attention_projection_arm=args.attention_projection_arm,
            source_dir=args.source_dir,
            gguf_path=args.gguf_path,
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = compare_attention_output_dumps(
        package_dir=args.package,
        prompt_path=args.prompt_file,
        dump_prefix=args.dump_prefix,
        layers=args.layers,
        final_row=args.final_row,
        embedding_arm=args.embedding_arm,
        attention_projection_arm=args.attention_projection_arm,
        kv_cache_arm=args.kv_cache_arm,
        source_dir=args.source_dir,
        gguf_path=args.gguf_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
