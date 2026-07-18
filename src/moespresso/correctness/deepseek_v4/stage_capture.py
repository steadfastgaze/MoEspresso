"""Capture served DS4 layer stages and compare them with DS4 reference dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moespresso.correctness.deepseek_v4.rope_diff import _metrics
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
N_HC = 4
HC_DIM = HIDDEN_SIZE * N_HC

STAGES = (
    "hc_attn_pre",
    "attn_norm",
    "attn_out",
    "hc_attn_post",
    "hc_ffn_pre",
    "ffn_norm",
    "ffn_out",
    "hc_ffn_post",
)
PROMPT_MODES = ("raw-user", "rendered")


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _stages(value: str) -> list[str]:
    stages = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s): {','.join(unknown)}"
        )
    return stages


def _key(layer: int, stage: str) -> str:
    return f"{layer}:{stage}"


def _read_stage_dump(prefix: Path, stage: str, layer: int, dump_pos: int) -> np.ndarray:
    path = Path(f"{prefix}_{stage}-{layer}_pos{dump_pos}.bin")
    if not path.is_file():
        raise FileNotFoundError(f"missing DS4 dump: {path}")
    return np.fromfile(path, dtype=np.float32)


def _as_numpy(value) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def _prompt_tokens(tokenizer, prompt_path: Path, prompt_mode: str) -> list[int]:
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if prompt_mode == "raw-user":
        rendered = render_prompt(
            [{"role": "user", "content": prompt_text}],
            tokenizer,
            template_kwargs={"enable_thinking": False},
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        )
        return list(tokenizer.encode(rendered))
    if prompt_mode == "rendered":
        return list(tokenizer.encode(prompt_text, add_special_tokens=False))
    raise ValueError(f"unknown prompt mode: {prompt_mode}")


def _reference_row(
    *,
    mode: str,
    tokens: int,
    ref_rows: int,
    final_row: int,
    ref_final_row: int | None,
) -> int:
    row = final_row if ref_final_row is None else int(ref_final_row)
    if mode == "full" and ref_rows != tokens:
        raise ValueError(
            f"token count mismatch: MoEspresso={tokens} DS4={ref_rows}"
        )
    if mode == "split-final" and ref_rows not in {tokens, 1} and ref_final_row is None:
        raise ValueError(
            "split-final reference dumps with a chunked DS4 prefill need "
            "--ref-final-row because the dump does not contain the full prompt "
            f"(MoEspresso={tokens} DS4={ref_rows})"
        )
    if row < 0 or row >= ref_rows:
        raise ValueError(
            f"reference row {row} is outside DS4 dump with {ref_rows} rows"
        )
    return row


def _flatten_stage(value: np.ndarray, stage: str) -> np.ndarray:
    if stage in {"hc_attn_post", "hc_ffn_post"}:
        if value.ndim != 4 or value.shape[2] != N_HC or value.shape[-1] != HIDDEN_SIZE:
            raise ValueError(f"unexpected {stage} capture shape: {value.shape}")
        return value[0].reshape(value.shape[1], HC_DIM).reshape(-1)
    if value.ndim != 3 or value.shape[-1] != HIDDEN_SIZE:
        raise ValueError(f"unexpected {stage} capture shape: {value.shape}")
    return value[0].reshape(-1)


def _record(captures: dict[str, np.ndarray], layer: int, stage: str, value) -> None:
    captures[_key(layer, stage)] = _flatten_stage(_as_numpy(value), stage)


def _install_layer_capture(model, layers: list[int], captures: dict[str, np.ndarray]):
    import jang_tools.dsv4.mlx_model as dsv4_model

    targets = set(int(layer) for layer in layers)
    original = dsv4_model.DeepseekV4DecoderLayer.__call__

    def traced(self, x, mask=None, cache=None, input_ids=None):
        layer_id = int(self.layer_id)
        if layer_id not in targets:
            return original(self, x, mask=mask, cache=cache, input_ids=input_ids)

        residual = x
        x, post, comb = self._hc_pre(
            x,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        _record(captures, layer_id, "hc_attn_pre", x)
        x = self.input_layernorm(x)
        _record(captures, layer_id, "attn_norm", x)
        x = self.self_attn(x, mask=mask, cache=cache)
        _record(captures, layer_id, "attn_out", x)
        x = self._hc_post(x, residual, post, comb)
        _record(captures, layer_id, "hc_attn_post", x)

        residual = x
        x, post, comb = self._hc_pre(
            x,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        _record(captures, layer_id, "hc_ffn_pre", x)
        x = self.post_attention_layernorm(x)
        _record(captures, layer_id, "ffn_norm", x)
        x = self.mlp(x, input_ids=input_ids)
        _record(captures, layer_id, "ffn_out", x)
        x = self._hc_post(x, residual, post, comb)
        _record(captures, layer_id, "hc_ffn_post", x)
        return x

    dsv4_model.DeepseekV4DecoderLayer.__call__ = traced
    return original


def _restore_layer_capture(original) -> None:
    import jang_tools.dsv4.mlx_model as dsv4_model

    dsv4_model.DeepseekV4DecoderLayer.__call__ = original


def compare_served_stages(
    *,
    package_dir: Path,
    prompt_path: Path,
    dump_prefix: Path,
    dump_pos: int,
    layers: list[int],
    stages: list[str],
    final_row: int,
    ref_final_row: int | None,
    mode: str,
    prompt_mode: str,
    embedding_arm: str,
    attention_projection_arm: str,
    kv_cache_arm: str,
    source_dir: Path,
    gguf_path: Path,
) -> dict:
    import mlx.core as mx

    model, tokenizer, manifest = load_served_model(package_dir)
    tokens = _prompt_tokens(tokenizer, prompt_path, prompt_mode)
    input_ids = mx.array(tokens, dtype=mx.int32)[None]
    if embedding_arm == "source_prompt_rows":
        _patch_prompt_embedding_rows(model, source_dir, tokens)
    elif embedding_arm != "default":
        raise ValueError(f"unknown embedding arm: {embedding_arm}")
    _patch_attention_projection_arm(
        model, attention_projection_arm, source_dir, gguf_path)

    captures: dict[str, np.ndarray] = {}
    attention_original = None
    if kv_cache_arm != "direct":
        from moespresso.correctness.deepseek_v4.attention_output_capture import (
            _install_attention_output_capture,
        )

        attention_original = _install_attention_output_capture(
            model,
            layers,
            {},
            kv_cache_arm=kv_cache_arm,
        )
    original = _install_layer_capture(model, layers, captures)
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
        _restore_layer_capture(original)
        if attention_original is not None:
            from moespresso.correctness.deepseek_v4.attention_output_capture import (
                _restore_attention_output_capture,
            )

            _restore_attention_output_capture(attention_original)

    rows = []
    for layer in layers:
        stage_rows = {}
        for stage in stages:
            ref = _read_stage_dump(dump_prefix, stage, layer, dump_pos)
            got = captures[_key(layer, stage)]
            width = HC_DIM if stage in {"hc_attn_post", "hc_ffn_post"} else HIDDEN_SIZE
            tokens_in_dump = ref.size // width
            ref_row = _reference_row(
                mode=mode,
                tokens=len(tokens),
                ref_rows=tokens_in_dump,
                final_row=final_row,
                ref_final_row=ref_final_row,
            )
            if mode == "full":
                if got.size != ref.size:
                    raise ValueError(
                        f"size mismatch for {stage} layer {layer}: "
                        f"MoEspresso={got.size} DS4={ref.size}"
                    )
                all_metrics = _metrics(got, ref)
                got_final = got.reshape(len(tokens), width)[final_row]
            elif mode == "split-final":
                if got.size != width:
                    raise ValueError(
                        f"split-final size mismatch for {stage} layer {layer}: "
                        f"MoEspresso={got.size} expected={width}"
                    )
                all_metrics = None
                got_final = got.reshape(1, width)[0]
            else:
                raise ValueError(
                    f"unknown capture mode during comparison: {mode}"
                )
            ref_final = ref.reshape(tokens_in_dump, width)[ref_row]
            stage_rows[stage] = {
                "all": all_metrics,
                "final": _metrics(got_final, ref_final),
                "ds4_dump_rows": tokens_in_dump,
                "ds4_ref_row": ref_row,
            }
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
        "embedding_arm": embedding_arm,
        "attention_projection_arm": attention_projection_arm,
        "kv_cache_arm": kv_cache_arm,
        "prompt_tokens": len(tokens),
        "dump_pos": dump_pos,
        "final_row": final_row,
        "ref_final_row": ref_final_row,
        "layers": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--dump-pos", default=0, type=int)
    parser.add_argument("--layers", default="0,1,2", type=_layers)
    parser.add_argument("--stages", default=",".join(STAGES), type=_stages)
    parser.add_argument("--final-row", required=True, type=int)
    parser.add_argument(
        "--ref-final-row",
        type=int,
        default=None,
        help=(
            "row inside the DS4 reference dump to compare. Needed for chunked "
            "prefill dumps that contain only the final prompt chunk."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("full", "split-final"),
        default="full",
        help="Capture direct full-prompt forward or the generation split final-token call.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default="raw-user",
        help="raw user text to render, or an already-rendered DS4 chat prompt.",
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
    result = compare_served_stages(
        package_dir=args.package,
        prompt_path=args.prompt_file,
        dump_prefix=args.dump_prefix,
        dump_pos=args.dump_pos,
        layers=args.layers,
        stages=args.stages,
        final_row=args.final_row,
        ref_final_row=args.ref_final_row,
        mode=args.mode,
        prompt_mode=args.prompt_mode,
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
