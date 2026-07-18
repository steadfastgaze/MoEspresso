"""Compare served DS4 attention projection stages against DS4 dumps."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
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


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _key(layer: int, name: str) -> str:
    return f"{layer}:{name}"


def _as_float_numpy(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def _flatten_hidden(value: Any, width: int) -> np.ndarray:
    arr = _as_float_numpy(value)
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[-1] != width:
        raise ValueError(f"unexpected hidden capture shape: {arr.shape}")
    return arr[0].reshape(-1)


def _flatten_q(value: Any) -> np.ndarray:
    arr = _as_float_numpy(value)
    if arr.ndim != 4 or arr.shape[1] != N_HEADS or arr.shape[-1] != HEAD_DIM:
        raise ValueError(f"unexpected Q capture shape: {arr.shape}")
    return arr[0].transpose(1, 0, 2).reshape(-1)


def _flatten_kv(value: Any) -> np.ndarray:
    arr = _as_float_numpy(value)
    if arr.ndim == 3:
        if arr.shape[0] != 1 or arr.shape[-1] != HEAD_DIM:
            raise ValueError(f"unexpected KV capture shape: {arr.shape}")
        return arr[0].reshape(-1)
    if arr.ndim == 4 and arr.shape[1] == 1 and arr.shape[-1] == HEAD_DIM:
        return arr[0, 0].reshape(-1)
    raise ValueError(f"unexpected KV capture shape: {arr.shape}")


def _install_attention_projection_capture(
    model,
    layers: Iterable[int],
    captures: dict[str, np.ndarray],
):
    import mlx.core as mx
    import jang_tools.dsv4.mlx_model as dsv4_model

    targets = set(int(layer) for layer in layers)
    original = dsv4_model.DeepseekV4Attention.__call__

    def traced(self, x, mask=None, cache=None):
        layer = int(self.layer_id)
        if layer in targets:
            q_lora = self.wq_a(x)
            captures[_key(layer, "q_lora")] = _flatten_hidden(
                q_lora,
                int(self.q_lora_rank),
            )
            q_lora_norm = self.q_norm(q_lora)
            captures[_key(layer, "q_lora_norm")] = _flatten_hidden(
                q_lora_norm,
                int(self.q_lora_rank),
            )
            q_raw = self.wq_b(q_lora_norm).reshape(
                x.shape[0],
                x.shape[1],
                self.n_heads,
                self.head_dim,
            )
            captures[_key(layer, "Qraw")] = _flatten_q(q_raw.transpose(0, 2, 1, 3))
            q_norm = mx.fast.rms_norm(
                q_raw,
                weight=dsv4_model._get_q_norm_ones(self.head_dim, q_raw.dtype),
                eps=self.args.rms_norm_eps,
            ).transpose(0, 2, 1, 3)
            captures[_key(layer, "Qnorm")] = _flatten_q(q_norm)

            kv_raw = self.wkv(x)
            captures[_key(layer, "KVraw")] = _flatten_kv(kv_raw)
            kv_norm = self.kv_norm(kv_raw)
            captures[_key(layer, "KVnorm")] = _flatten_kv(kv_norm)
        return original(self, x, mask=mask, cache=cache)

    dsv4_model.DeepseekV4Attention.__call__ = traced
    return original


def _restore_attention_projection_capture(original) -> None:
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


def compare_attention_projection_dumps(
    *,
    package_dir: Path,
    prompt_path: Path,
    dump_prefix: Path,
    layers: list[int],
    final_row: int,
    embedding_arm: str,
    attention_projection_arm: str,
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
    original = _install_attention_projection_capture(model, layers, captures)
    try:
        logits = model(input_ids)
        mx.eval(logits)
    finally:
        _restore_attention_projection_capture(original)

    rows = []
    for layer in layers:
        layer_rows: dict[str, Any] = {}
        q_lora = captures[_key(layer, "q_lora")]
        q_lora_width = q_lora.size // len(tokens)
        stages = {
            "q_lora": q_lora_width,
            "q_lora_norm": q_lora_width,
            "Qraw": Q_DIM,
            "Qnorm": Q_DIM,
            "KVraw": HEAD_DIM,
            "KVnorm": HEAD_DIM,
        }
        for stage, width in stages.items():
            ref = _read_dump(dump_prefix, stage, layer)
            dump_tokens = ref.size // width
            if dump_tokens != len(tokens):
                raise ValueError(
                    f"token count mismatch for {stage} layer {layer}: "
                    f"MoEspresso={len(tokens)} DS4={dump_tokens}"
                )
            layer_rows[stage] = _stage_result(
                got=captures[_key(layer, stage)],
                ref=ref,
                width=width,
                tokens=len(tokens),
                final_row=final_row,
            )
        rows.append({
            "layer": int(layer),
            "tokens": len(tokens),
            "stages": layer_rows,
        })

    return {
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "embedding_arm": embedding_arm,
        "attention_projection_arm": attention_projection_arm,
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
    result = compare_attention_projection_dumps(
        package_dir=args.package,
        prompt_path=args.prompt_file,
        dump_prefix=args.dump_prefix,
        layers=args.layers,
        final_row=args.final_row,
        embedding_arm=args.embedding_arm,
        attention_projection_arm=args.attention_projection_arm,
        source_dir=args.source_dir,
        gguf_path=args.gguf_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
