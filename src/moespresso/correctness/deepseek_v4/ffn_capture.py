"""Compare served DS4 FFN substages against DS4 reference dumps."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from moespresso.correctness.deepseek_v4.rope_diff import _metrics
from moespresso.correctness.deepseek_v4.stage_capture import (
    PROMPT_MODES,
    _prompt_tokens,
    _read_stage_dump,
    _reference_row,
)
from moespresso.correctness.deepseek_v4.short_probe import (
    _DEFAULT_GGUF,
    _DEFAULT_SOURCE,
    _patch_attention_projection_arm,
    _patch_prompt_embedding_rows,
    _validate_probe_input_paths,
)
from moespresso.runtime.serve import load_served_model


HIDDEN_SIZE = 4096
TOP_K = 6

STAGES = (
    "ffn_norm",
    "ffn_moe_out",
    "ffn_shexp",
    "ffn_out",
)


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _key(layer: int, stage: str) -> str:
    return f"{layer}:{stage}"


def _read_i32_dump(prefix: Path, name: str, layer: int, dump_pos: int) -> np.ndarray:
    path = Path(f"{prefix}_{name}-{layer}_pos{dump_pos}.i32")
    if not path.is_file():
        raise FileNotFoundError(f"missing DS4 dump: {path}")
    return np.fromfile(path, dtype=np.int32)


def _as_float_numpy(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def _flatten_hidden(value: Any) -> np.ndarray:
    arr = _as_float_numpy(value)
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[-1] != HIDDEN_SIZE:
        raise ValueError(f"unexpected FFN capture shape: {arr.shape}")
    return arr[0].reshape(-1)


def _install_ffn_capture(
    layers: Iterable[int],
    captures: dict[str, np.ndarray],
):
    import jang_tools.dsv4.mlx_model as dsv4_model

    targets = set(int(layer) for layer in layers)
    original = dsv4_model.MoE.__call__

    def traced(self, x, input_ids=None):
        layer = int(self.layer_id)
        if layer not in targets:
            return original(self, x, input_ids=input_ids)
        return _capture_moe_stages(layer, self, x, input_ids, captures)

    dsv4_model.MoE.__call__ = traced
    return original


def _restore_ffn_capture(original) -> None:
    import jang_tools.dsv4.mlx_model as dsv4_model

    dsv4_model.MoE.__call__ = original


def _capture_moe_stages(
    layer: int,
    mlp,
    x,
    input_ids,
    captures: dict[str, np.ndarray],
):
    import mlx.core as mx

    captures[_key(layer, "ffn_norm")] = _flatten_hidden(x)
    inds, scores = mlp.gate(x, input_ids=input_ids)
    inds = inds.astype(mx.uint32)
    routed_by_expert = mlp.switch_mlp(x, inds)
    routed = (
        routed_by_expert * scores[..., None]
    ).sum(axis=-2).astype(routed_by_expert.dtype).reshape(x.shape)
    captures[_key(layer, "ffn_moe_out")] = _flatten_hidden(routed)
    shared = mlp.shared_experts(x)
    captures[_key(layer, "ffn_shexp")] = _flatten_hidden(shared)
    out = routed + shared
    captures[_key(layer, "ffn_out")] = _flatten_hidden(out)
    return out


def _capture_moe_with_ds4_routes(
    *,
    layer: int,
    mlp,
    ffn_norm: np.ndarray,
    topk: np.ndarray,
    route_weights: np.ndarray,
    captures: dict[str, np.ndarray],
) -> None:
    import mlx.core as mx

    if ffn_norm.shape != (HIDDEN_SIZE,):
        raise ValueError(f"bad DS4 ffn_norm row shape: {ffn_norm.shape}")
    if topk.shape != (TOP_K,):
        raise ValueError(f"bad DS4 top-k row shape: {topk.shape}")
    if route_weights.shape != (TOP_K,):
        raise ValueError(f"bad DS4 route-weight row shape: {route_weights.shape}")

    x = mx.array(ffn_norm.reshape(1, 1, HIDDEN_SIZE), dtype=mx.float32)
    inds = mx.array(topk.reshape(1, 1, TOP_K).astype(np.uint32), dtype=mx.uint32)
    scores = mx.array(route_weights.reshape(1, 1, TOP_K), dtype=mx.float32)

    captures[_key(layer, "ffn_norm")] = ffn_norm.astype(np.float32)
    routed_by_expert = mlp.switch_mlp(x, inds)
    routed = (
        routed_by_expert * scores[..., None]
    ).sum(axis=-2).astype(routed_by_expert.dtype).reshape(x.shape)
    captures[_key(layer, "ffn_moe_out")] = _flatten_hidden(routed)
    shared = mlp.shared_experts(x)
    captures[_key(layer, "ffn_shexp")] = _flatten_hidden(shared)
    out = routed + shared
    captures[_key(layer, "ffn_out")] = _flatten_hidden(out)


def _stage_result(
    *,
    got: np.ndarray,
    ref: np.ndarray,
    tokens: int,
    ref_rows: int,
    final_row: int,
    ref_final_row: int | None,
    mode: str,
) -> dict[str, Any]:
    ref_row = _reference_row(
        mode=mode,
        tokens=tokens,
        ref_rows=ref_rows,
        final_row=final_row,
        ref_final_row=ref_final_row,
    )
    if mode == "full":
        if got.size != ref.size:
            raise ValueError(f"size mismatch: MoEspresso={got.size} DS4={ref.size}")
        got_rows = got.reshape(tokens, HIDDEN_SIZE)
        all_metrics = _metrics(got, ref)
        got_final = got_rows[final_row]
    elif mode == "split-final":
        if got.size != HIDDEN_SIZE:
            raise ValueError(
                f"split-final size mismatch: MoEspresso={got.size} expected={HIDDEN_SIZE}"
            )
        all_metrics = None
        got_final = got.reshape(1, HIDDEN_SIZE)[0]
    else:
        raise ValueError(f"unknown capture mode: {mode}")
    ref_rows_arr = ref.reshape(ref_rows, HIDDEN_SIZE)
    return {
        "all": all_metrics,
        "final": _metrics(got_final, ref_rows_arr[ref_row]),
        "ds4_dump_rows": ref_rows,
        "ds4_ref_row": ref_row,
    }


def compare_ffn_dumps(
    *,
    package_dir: Path,
    prompt_path: Path,
    dump_prefix: Path,
    dump_pos: int,
    layers: list[int],
    final_row: int,
    ref_final_row: int | None,
    mode: str,
    prompt_mode: str,
    input_source: str,
    embedding_arm: str,
    attention_projection_arm: str,
    kv_cache_arm: str,
    source_dir: Path,
    gguf_path: Path,
) -> dict[str, Any]:
    import mlx.core as mx

    model, tokenizer, manifest = load_served_model(package_dir)
    tokens = _prompt_tokens(tokenizer, prompt_path, prompt_mode)
    input_ids = mx.array(tokens, dtype=mx.int32)[None]
    if input_source == "served":
        if embedding_arm == "source_prompt_rows":
            _patch_prompt_embedding_rows(model, source_dir, tokens)
        elif embedding_arm != "default":
            raise ValueError(f"unknown embedding arm: {embedding_arm}")
        _patch_attention_projection_arm(
            model, attention_projection_arm, source_dir, gguf_path)
    elif input_source in {"ds4", "ds4-routed"}:
        if embedding_arm != "default":
            raise ValueError("embedding arm is only valid with served input source")
        if attention_projection_arm != "default":
            raise ValueError(
                "attention projection arm is only valid with served input source")
        if kv_cache_arm != "direct":
            raise ValueError("KV cache arm is only valid with served input source")
    else:
        raise ValueError(f"unknown input source: {input_source}")

    attention_original = None
    if input_source == "served" and kv_cache_arm != "direct":
        from moespresso.correctness.deepseek_v4.attention_output_capture import (
            _install_attention_output_capture,
        )

        attention_original = _install_attention_output_capture(
            model,
            layers,
            {},
            kv_cache_arm=kv_cache_arm,
        )

    captures: dict[str, np.ndarray] = {}
    if input_source == "served":
        original = _install_ffn_capture(layers, captures)
        try:
            if mode == "full":
                logits = model(input_ids)
                mx.eval(logits)
            elif mode == "split-final":
                from mlx_lm.generate import generate_step

                list(generate_step(
                    mx.array(tokens, dtype=mx.int32),
                    model,
                    max_tokens=0,
                    prefill_step_size=2048,
                ))
            else:
                raise ValueError(f"unknown capture mode: {mode}")
        finally:
            _restore_ffn_capture(original)
            if attention_original is not None:
                from moespresso.correctness.deepseek_v4.attention_output_capture import (
                    _restore_attention_output_capture,
                )

                _restore_attention_output_capture(attention_original)
    elif input_source == "ds4":
        if mode != "full":
            raise ValueError("input_source=ds4 only supports mode=full")
        model_layers = getattr(getattr(model, "model", model), "layers", None)
        if model_layers is None:
            raise ValueError("loaded model has no model.layers container")
        for layer in layers:
            ref_norm = _read_stage_dump(dump_prefix, "ffn_norm", layer, dump_pos)
            dump_tokens = ref_norm.size // HIDDEN_SIZE
            if dump_tokens != len(tokens):
                raise ValueError(
                    f"token count mismatch for ffn_norm layer {layer}: "
                    f"MoEspresso={len(tokens)} DS4={dump_tokens}"
                )
            mlp = getattr(model_layers[layer], "mlp", None)
            if mlp is None:
                raise ValueError(f"layer {layer} has no mlp")
            x = mx.array(
                ref_norm.reshape(1, dump_tokens, HIDDEN_SIZE),
                dtype=mx.float32,
            )
            _capture_moe_stages(int(layer), mlp, x, input_ids, captures)
    elif input_source == "ds4-routed":
        if mode != "split-final":
            raise ValueError("input_source=ds4-routed only supports mode=split-final")
        if ref_final_row is None:
            raise ValueError("input_source=ds4-routed requires --ref-final-row")
        model_layers = getattr(getattr(model, "model", model), "layers", None)
        if model_layers is None:
            raise ValueError("loaded model has no model.layers container")
        for layer in layers:
            ref_norm = _read_stage_dump(dump_prefix, "ffn_norm", layer, dump_pos)
            if ref_norm.size % HIDDEN_SIZE:
                raise ValueError(
                    f"bad DS4 ffn_norm dump size for layer {layer}: {ref_norm.size}"
                )
            dump_tokens = ref_norm.size // HIDDEN_SIZE
            row = _reference_row(
                mode="split-final",
                tokens=len(tokens),
                ref_rows=dump_tokens,
                final_row=final_row,
                ref_final_row=ref_final_row,
            )
            topk = _read_i32_dump(dump_prefix, "ffn_moe_topk", layer, dump_pos)
            route_weights = _read_stage_dump(
                dump_prefix,
                "ffn_moe_weights_scaled",
                layer,
                dump_pos,
            )
            if topk.size != dump_tokens * TOP_K:
                raise ValueError(
                    f"bad DS4 top-k dump size for layer {layer}: {topk.size}"
                )
            if route_weights.size != dump_tokens * TOP_K:
                raise ValueError(
                    "bad DS4 route-weight dump size for layer "
                    f"{layer}: {route_weights.size}"
                )
            mlp = getattr(model_layers[layer], "mlp", None)
            if mlp is None:
                raise ValueError(f"layer {layer} has no mlp")
            _capture_moe_with_ds4_routes(
                layer=int(layer),
                mlp=mlp,
                ffn_norm=ref_norm.reshape(dump_tokens, HIDDEN_SIZE)[row],
                topk=topk.reshape(dump_tokens, TOP_K)[row],
                route_weights=route_weights.reshape(dump_tokens, TOP_K)[row],
                captures=captures,
            )

    rows = []
    for layer in layers:
        stage_rows: dict[str, Any] = {}
        for stage in STAGES:
            ref = _read_stage_dump(dump_prefix, stage, layer, dump_pos)
            dump_tokens = ref.size // HIDDEN_SIZE
            stage_rows[stage] = _stage_result(
                got=captures[_key(layer, stage)],
                ref=ref,
                tokens=len(tokens),
                ref_rows=dump_tokens,
                final_row=final_row,
                ref_final_row=ref_final_row,
                mode=mode,
            )
        rows.append({
            "layer": int(layer),
            "tokens": len(tokens),
            "stages": stage_rows,
        })

    return {
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "mode": mode,
        "prompt_mode": prompt_mode,
        "input_source": input_source,
        "embedding_arm": embedding_arm,
        "attention_projection_arm": attention_projection_arm,
        "kv_cache_arm": kv_cache_arm,
        "prompt_tokens": len(tokens),
        "dump_pos": int(dump_pos),
        "final_row": int(final_row),
        "ref_final_row": ref_final_row,
        "layers": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--dump-pos", default=0, type=int)
    parser.add_argument("--layers", default="0", type=_layers)
    parser.add_argument("--final-row", required=True, type=int)
    parser.add_argument("--ref-final-row", default=None, type=int)
    parser.add_argument(
        "--mode",
        choices=("full", "split-final"),
        default="full",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default="raw-user",
    )
    parser.add_argument(
        "--input-source",
        choices=("served", "ds4", "ds4-routed"),
        default="served",
        help=(
            "Use served hidden states, feed DS4 ffn_norm dumps into the package "
            "gate, or feed DS4 ffn_norm plus DS4 routes directly into package "
            "experts."
        ),
    )
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
    result = compare_ffn_dumps(
        package_dir=args.package,
        prompt_path=args.prompt_file,
        dump_prefix=args.dump_prefix,
        dump_pos=args.dump_pos,
        layers=args.layers,
        final_row=args.final_row,
        ref_final_row=args.ref_final_row,
        mode=args.mode,
        prompt_mode=args.prompt_mode,
        input_source=args.input_source,
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
