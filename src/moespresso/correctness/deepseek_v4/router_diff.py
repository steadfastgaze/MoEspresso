"""Compare served DS4 router stages against DS4 reference dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from moespresso.correctness.deepseek_v4.rope_diff import _metrics
from moespresso.correctness.deepseek_v4.short_probe import (
    _DEFAULT_SOURCE,
    _patch_prompt_embedding_rows,
    _validate_probe_input_paths,
)
from moespresso.correctness.deepseek_v4.stage_capture import (
    PROMPT_MODES,
    _prompt_tokens,
    _read_stage_dump,
    _reference_row,
)
from moespresso.runtime.serve import load_served_model


HIDDEN_SIZE = 4096
N_EXPERT = 256
TOP_K = 6

FLOAT_STAGES: dict[str, int] = {
    "ffn_norm": HIDDEN_SIZE,
    "ffn_moe_logits": N_EXPERT,
    "ffn_moe_probs": N_EXPERT,
    "ffn_moe_weights_scaled": TOP_K,
}


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _key(layer: int, stage: str) -> str:
    return f"{layer}:{stage}"


def _as_float_numpy(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def _as_i32_numpy(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.int32))


def _flatten_float(value: Any, width: int) -> np.ndarray:
    arr = _as_float_numpy(value)
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[-1] != width:
        raise ValueError(f"unexpected router float capture shape: {arr.shape}")
    return arr[0].reshape(-1)


def _flatten_i32(value: Any) -> np.ndarray:
    arr = _as_i32_numpy(value)
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[-1] != TOP_K:
        raise ValueError(f"unexpected router top-k capture shape: {arr.shape}")
    return arr[0].reshape(-1)


def _read_i32_dump(prefix: Path, name: str, layer: int, dump_pos: int) -> np.ndarray:
    path = Path(f"{prefix}_{name}-{layer}_pos{dump_pos}.i32")
    if not path.is_file():
        raise FileNotFoundError(f"missing DS4 dump: {path}")
    return np.fromfile(path, dtype=np.int32)


def _capture_gate_stages(
    *,
    layer_id: int,
    gate,
    x,
    input_ids,
    captures: dict[str, np.ndarray],
):
    import mlx.core as mx

    captures[_key(layer_id, "ffn_norm")] = _flatten_float(x, HIDDEN_SIZE)
    logits = x.astype(mx.float32) @ gate.weight.T.astype(mx.float32)
    captures[_key(layer_id, "ffn_moe_logits")] = _flatten_float(
        logits,
        N_EXPERT,
    )
    probs = mx.sqrt(mx.log1p(mx.exp(logits)))
    captures[_key(layer_id, "ffn_moe_probs")] = _flatten_float(
        probs,
        N_EXPERT,
    )

    if gate.hash:
        if input_ids is None:
            raise ValueError(
                f"hash-routed layer {layer_id} was called without input_ids"
            )
        inds = gate.tid2eid[input_ids].astype(mx.int32)
        weights = mx.take_along_axis(probs, inds, axis=-1)
    else:
        scores = probs + gate.bias
        k = int(gate.args.num_experts_per_tok)
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k].astype(
            mx.int32
        )
        weights = mx.take_along_axis(probs, inds, axis=-1)

    if gate.args.norm_topk_prob:
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    weights = weights * gate.args.routed_scaling_factor
    captures[_key(layer_id, "ffn_moe_topk")] = _flatten_i32(inds)
    captures[_key(layer_id, "ffn_moe_weights_scaled")] = _flatten_float(
        weights,
        TOP_K,
    )


def _install_gate_capture(
    layers: list[int],
    captures: dict[str, np.ndarray],
):
    import jang_tools.dsv4.mlx_model as dsv4_model

    targets = set(int(layer) for layer in layers)
    original = dsv4_model.Gate.__call__

    def traced(self, x, input_ids=None):
        layer_id = int(self.layer_id)
        if layer_id not in targets:
            return original(self, x, input_ids=input_ids)

        _capture_gate_stages(
            layer_id=layer_id,
            gate=self,
            x=x,
            input_ids=input_ids,
            captures=captures,
        )
        return original(self, x, input_ids=input_ids)

    dsv4_model.Gate.__call__ = traced
    return original


def _restore_gate_capture(original) -> None:
    import jang_tools.dsv4.mlx_model as dsv4_model

    dsv4_model.Gate.__call__ = original


def _float_stage_result(
    *,
    got: np.ndarray,
    ref: np.ndarray,
    width: int,
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
        got_rows = got.reshape(tokens, width)
        all_metrics = _metrics(got, ref)
        got_final = got_rows[final_row]
    elif mode == "split-final":
        if got.size != width:
            raise ValueError(
                f"split-final size mismatch: MoEspresso={got.size} expected={width}"
            )
        all_metrics = None
        got_final = got.reshape(1, width)[0]
    else:
        raise ValueError(f"unknown capture mode: {mode}")
    ref_rows_arr = ref.reshape(ref_rows, width)
    return {
        "all": all_metrics,
        "final": _metrics(got_final, ref_rows_arr[ref_row]),
        "ds4_dump_rows": ref_rows,
        "ds4_ref_row": ref_row,
    }


def _topk_result(
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
            raise ValueError(
                f"top-k size mismatch: MoEspresso={got.size} DS4={ref.size}"
            )
        got_rows = got.reshape(tokens, TOP_K)
        ref_rows_arr = ref.reshape(tokens, TOP_K)
        ordered = np.all(got_rows == ref_rows_arr, axis=1)
        set_equal = np.array([
            set(int(v) for v in got_row) == set(int(v) for v in ref_row_values)
            for got_row, ref_row_values in zip(got_rows, ref_rows_arr, strict=True)
        ])
        final_got = got_rows[final_row]
        final_ref = ref_rows_arr[final_row]
    elif mode == "split-final":
        if got.size != TOP_K:
            raise ValueError(
                f"split-final top-k size mismatch: MoEspresso={got.size} "
                f"expected={TOP_K}"
            )
        got_rows = got.reshape(1, TOP_K)
        ref_rows_arr = ref.reshape(ref_rows, TOP_K)
        final_got = got_rows[0]
        final_ref = ref_rows_arr[ref_row]
        ordered = np.array([np.all(final_got == final_ref)])
        set_equal = np.array([
            set(int(v) for v in final_got) == set(int(v) for v in final_ref)
        ])
    else:
        raise ValueError(f"unknown capture mode: {mode}")
    return {
        "ordered_matches": int(np.sum(ordered)),
        "set_matches": int(np.sum(set_equal)),
        "tokens": int(tokens if mode == "full" else 1),
        "ds4_dump_rows": int(ref_rows),
        "ds4_ref_row": int(ref_row),
        "final_ordered_match": bool(ordered[-1]),
        "final_set_match": bool(set_equal[-1]),
        "final_moespresso": [int(v) for v in final_got],
        "final_ds4": [int(v) for v in final_ref],
        "first_ordered_mismatch": (
            None if bool(np.all(ordered)) else int(np.flatnonzero(~ordered)[0])
        ),
        "first_set_mismatch": (
            None if bool(np.all(set_equal)) else int(np.flatnonzero(~set_equal)[0])
        ),
    }


def compare_router_dumps(
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
    source_dir: Path,
) -> dict[str, Any]:
    import mlx.core as mx

    model, tokenizer, manifest = load_served_model(package_dir)
    tokens = _prompt_tokens(tokenizer, prompt_path, prompt_mode)
    input_ids = mx.array(tokens, dtype=mx.int32)[None]

    captures: dict[str, np.ndarray] = {}
    if input_source == "served":
        if embedding_arm == "source_prompt_rows":
            _patch_prompt_embedding_rows(model, source_dir, tokens)
        elif embedding_arm != "default":
            raise ValueError(f"unknown embedding arm: {embedding_arm}")
        original = _install_gate_capture(layers, captures)
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
            _restore_gate_capture(original)
    elif input_source == "ds4":
        if mode != "full":
            raise ValueError("input_source=ds4 only supports mode=full")
        if embedding_arm != "default":
            raise ValueError("embedding arm is only valid with served input source")
        model_layers = getattr(getattr(model, "model", model), "layers", None)
        if model_layers is None:
            raise ValueError("loaded model has no model.layers container")
        for layer in layers:
            ref_norm = _read_stage_dump(dump_prefix, "ffn_norm", layer, dump_pos)
            if ref_norm.size % HIDDEN_SIZE != 0:
                raise ValueError(
                    f"bad DS4 ffn_norm dump size for layer {layer}: "
                    f"{ref_norm.size}"
                )
            dump_tokens = ref_norm.size // HIDDEN_SIZE
            if dump_tokens != len(tokens):
                raise ValueError(
                    f"token count mismatch for ffn_norm layer {layer}: "
                    f"prompt={len(tokens)} DS4={dump_tokens}"
                )
            gate = getattr(getattr(model_layers[layer], "mlp", None), "gate", None)
            if gate is None:
                raise ValueError(f"layer {layer} has no mlp.gate")
            x = mx.array(
                ref_norm.reshape(1, dump_tokens, HIDDEN_SIZE),
                dtype=mx.float32,
            )
            _capture_gate_stages(
                layer_id=int(layer),
                gate=gate,
                x=x,
                input_ids=input_ids,
                captures=captures,
            )
    else:
        raise ValueError(f"unknown input source: {input_source}")

    rows = []
    for layer in layers:
        stage_rows: dict[str, Any] = {}
        for stage, width in FLOAT_STAGES.items():
            ref = _read_stage_dump(dump_prefix, stage, layer, dump_pos)
            got = captures[_key(layer, stage)]
            if ref.size % width != 0:
                raise ValueError(
                    f"bad DS4 dump size for {stage} layer {layer}: {ref.size}"
                )
            dump_tokens = ref.size // width
            stage_rows[stage] = _float_stage_result(
                got=got,
                ref=ref,
                width=width,
                tokens=len(tokens),
                ref_rows=dump_tokens,
                final_row=final_row,
                ref_final_row=ref_final_row,
                mode=mode,
            )

        ref_topk = _read_i32_dump(dump_prefix, "ffn_moe_topk", layer, dump_pos)
        dump_tokens = ref_topk.size // TOP_K
        got_topk = captures[_key(layer, "ffn_moe_topk")]
        stage_rows["ffn_moe_topk"] = _topk_result(
            got=got_topk,
            ref=ref_topk,
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
    parser.add_argument("--layers", default="3", type=_layers)
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
        choices=("served", "ds4"),
        default="served",
        help="Use served hidden states or feed DS4 ffn_norm dumps into the gate.",
    )
    parser.add_argument(
        "--embedding-arm",
        choices=("default", "source_prompt_rows"),
        default="default",
    )
    parser.add_argument("--source-dir", type=Path, default=_DEFAULT_SOURCE)
    args = parser.parse_args(argv)
    try:
        _validate_probe_input_paths(
            embedding_arm=args.embedding_arm,
            source_dir=args.source_dir,
            gguf_path=None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = compare_router_dumps(
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
        source_dir=args.source_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
