"""Capture served DS4 Q/K RoPE tensors and compare them with DS4 reference dumps."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

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


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _capture_key(layer: int, name: str) -> str:
    return f"{layer}:{name}"


def _as_numpy(value) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value)


def _flatten_q(value: np.ndarray) -> np.ndarray:
    if value.ndim != 4 or value.shape[1] != N_HEADS or value.shape[-1] != HEAD_DIM:
        raise ValueError(f"unexpected Q capture shape: {value.shape}")
    return value[0].transpose(1, 0, 2).reshape(-1)


def _flatten_kv(value: np.ndarray) -> np.ndarray:
    if value.ndim != 4 or value.shape[1] != 1 or value.shape[-1] != HEAD_DIM:
        raise ValueError(f"unexpected KV capture shape: {value.shape}")
    return value[0, 0].reshape(-1)


def _capture_rope_calls(model, layers: Iterable[int], captures: dict[str, np.ndarray]):
    import jang_tools.dsv4.mlx_model as dsv4_model

    target_layers = set(int(layer) for layer in layers)
    rope_to_layer = {
        id(layer.self_attn.rope): int(i)
        for i, layer in enumerate(model.model.layers)
        if int(i) in target_layers
    }
    original = dsv4_model._apply_partial_rope

    def wrapped(x, rope, offset=0, inverse=False, positions=None):
        out = original(x, rope, offset=offset, inverse=inverse, positions=positions)
        layer = rope_to_layer.get(id(rope))
        if layer is None or inverse or positions is not None:
            return out
        shape = tuple(x.shape)
        if len(shape) == 4 and shape[1] == N_HEADS and shape[-1] == HEAD_DIM:
            captures.setdefault(_capture_key(layer, "Qnorm"), _flatten_q(_as_numpy(x)))
            captures.setdefault(_capture_key(layer, "Qcur"), _flatten_q(_as_numpy(out)))
        elif len(shape) == 4 and shape[1] == 1 and shape[-1] == HEAD_DIM:
            captures.setdefault(_capture_key(layer, "KVnorm"), _flatten_kv(_as_numpy(x)))
            captures.setdefault(_capture_key(layer, "KVrope"), _flatten_kv(_as_numpy(out)))
        return out

    dsv4_model._apply_partial_rope = wrapped
    return original


def _restore_rope_capture(original) -> None:
    import jang_tools.dsv4.mlx_model as dsv4_model

    dsv4_model._apply_partial_rope = original


def compare_served_rope(
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
) -> dict:
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
    original = _capture_rope_calls(model, layers, captures)
    try:
        logits = model(input_ids)
        mx.eval(logits)
    finally:
        _restore_rope_capture(original)

    rows = []
    for layer in layers:
        qnorm_ref = _read_dump(dump_prefix, "Qnorm", layer)
        qcur_ref = _read_dump(dump_prefix, "Qcur", layer)
        kvnorm_ref = _read_dump(dump_prefix, "KVnorm", layer)
        kvrope_ref = _read_dump(dump_prefix, "KVrope", layer)
        tokens_in_dump = qnorm_ref.size // Q_DIM
        if tokens_in_dump != len(tokens):
            raise ValueError(
                f"token count mismatch for layer {layer}: "
                f"MoEspresso={len(tokens)} DS4={tokens_in_dump}"
            )
        if final_row >= len(tokens):
            raise ValueError(f"final row {final_row} outside {len(tokens)} tokens")

        qnorm = captures[_capture_key(layer, "Qnorm")]
        qcur = captures[_capture_key(layer, "Qcur")]
        kvnorm = captures[_capture_key(layer, "KVnorm")]
        kvrope = captures[_capture_key(layer, "KVrope")]
        rows.append({
            "layer": int(layer),
            "tokens": len(tokens),
            "qnorm": {
                "all": _metrics(qnorm, qnorm_ref),
                "final": _metrics(
                    qnorm.reshape(len(tokens), Q_DIM)[final_row],
                    qnorm_ref.reshape(len(tokens), Q_DIM)[final_row],
                ),
            },
            "qcur": {
                "all": _metrics(qcur, qcur_ref),
                "final": _metrics(
                    qcur.reshape(len(tokens), Q_DIM)[final_row],
                    qcur_ref.reshape(len(tokens), Q_DIM)[final_row],
                ),
            },
            "kvnorm": {
                "all": _metrics(kvnorm, kvnorm_ref),
                "final": _metrics(
                    kvnorm.reshape(len(tokens), HEAD_DIM)[final_row],
                    kvnorm_ref.reshape(len(tokens), HEAD_DIM)[final_row],
                ),
            },
            "kvrope": {
                "all": _metrics(kvrope, kvrope_ref),
                "final": _metrics(
                    kvrope.reshape(len(tokens), HEAD_DIM)[final_row],
                    kvrope_ref.reshape(len(tokens), HEAD_DIM)[final_row],
                ),
            },
        })
    return {
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "embedding_arm": embedding_arm,
        "attention_projection_arm": attention_projection_arm,
        "prompt_tokens": len(tokens),
        "final_row": final_row,
        "layers": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--layers", default="0,1,2", type=_layers)
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
    result = compare_served_rope(
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
