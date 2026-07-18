"""Compare direct DS4 graph states with the opt-in pooled MoE block graph."""

from __future__ import annotations

import argparse
import gc
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from moespresso.correctness.deepseek_v4.stage_capture import (
    PROMPT_MODES,
    _install_layer_capture,
    _prompt_tokens,
    _restore_layer_capture,
    _stages,
)
from moespresso.runtime.deepseek_v4.renderer import DEEPSEEK_V4_PROMPT_RENDERER
from moespresso.runtime.http import render_prompt


DEFAULT_STAGES = ("ffn_out", "hc_ffn_post")
MODES = ("full", "generate-step", "stream-generate", "metadata")
DEFAULT_TOP_LOGPROBS = 20


def _layers_arg(value: str) -> list[int]:
    if value.strip() == "all":
        return []
    return [int(item) for item in value.split(",") if item.strip()]


def _metric(got: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    if got.shape != ref.shape:
        return {
            "shape": {"direct": list(ref.shape), "wrapped": list(got.shape)},
            "array_equal": False,
            "max_abs": float("inf"),
            "rms": float("inf"),
            "rel_rms": float("inf"),
        }
    diff = got.astype(np.float64) - ref.astype(np.float64)
    ref64 = ref.astype(np.float64)
    denom = float(np.sqrt(np.mean(ref64 * ref64))) if ref64.size else 0.0
    rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    return {
        "shape": list(ref.shape),
        "array_equal": bool(np.array_equal(got, ref)),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms": rms,
        "rel_rms": rms / denom if denom else 0.0,
    }


def _prompt_text(prompt_path: Path, prompt_mode: str, tokenizer) -> str:
    text = prompt_path.read_text(encoding="utf-8")
    if prompt_mode == "raw-user":
        return render_prompt(
            [{"role": "user", "content": text}],
            tokenizer,
            template_kwargs={"enable_thinking": False},
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        )
    if prompt_mode == "rendered":
        return text
    raise ValueError(f"unknown prompt mode: {prompt_mode}")


def _first_non_equal(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        for stage, metrics in row["stages"].items():
            if not metrics["all"]["array_equal"]:
                return {
                    "layer": row["layer"],
                    "stage": stage,
                    "scope": "all",
                    "metrics": metrics["all"],
                }
            if not metrics["final"]["array_equal"]:
                return {
                    "layer": row["layer"],
                    "stage": stage,
                    "scope": "final",
                    "metrics": metrics["final"],
                }
    return None


def _score_summary(direct_logits, wrapped_logits) -> dict[str, Any]:
    import mlx.core as mx

    mx.eval(direct_logits, wrapped_logits)
    direct = np.asarray(direct_logits.astype(mx.float32), dtype=np.float32)
    wrapped = np.asarray(wrapped_logits.astype(mx.float32), dtype=np.float32)
    all_metrics = _metric(wrapped, direct)
    if direct.ndim == 1:
        direct_final = direct
        wrapped_final = wrapped
    else:
        direct_final = direct[0, -1]
        wrapped_final = wrapped[0, -1]
    final_metrics = _metric(wrapped_final, direct_final)
    direct_top = int(np.argmax(direct_final))
    wrapped_top = int(np.argmax(wrapped_final))
    return {
        "all": all_metrics,
        "final": final_metrics,
        "direct_final_argmax": direct_top,
        "wrapped_final_argmax": wrapped_top,
        "final_argmax_equal": direct_top == wrapped_top,
    }


def _top_logprob_signature(top_logprobs: Sequence[Sequence[Mapping[str, Any]]]) -> list[list[int]]:
    return [
        [int(item["token_id"]) for item in step]
        for step in top_logprobs
    ]


@contextmanager
def _temporary_env(updates: Mapping[str, str | None]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _cleanup_mlx() -> None:
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def _load_capture_run(
    *,
    package_dir: Path,
    prompt_path: Path,
    prompt_mode: str,
    layers: list[int],
    stages: Sequence[str],
    wrap_moe_block: bool,
    mode: str,
    max_tokens: int,
) -> dict[str, Any]:
    import mlx.core as mx

    from moespresso.runtime.serve import load_served_model
    from moespresso.runtime.ssd_streaming_build import ssd_streaming_stats

    del wrap_moe_block
    with _temporary_env({}):
        model, tokenizer, manifest = load_served_model(package_dir)
        if not layers:
            layers = list(range(len(model.layers)))
        tokens = _prompt_tokens(tokenizer, prompt_path, prompt_mode)
        rendered_prompt = _prompt_text(prompt_path, prompt_mode, tokenizer)
        input_ids = mx.array(tokens, dtype=mx.int32)[None]
        captures: dict[str, np.ndarray] = {}
        original = _install_layer_capture(model, layers, captures)
        try:
            generated_tokens = []
            token_logprobs = []
            top_logprobs = []
            if mode == "full":
                scores = model(input_ids)
                mx.eval(scores)
            elif mode == "generate-step":
                from mlx_lm.generate import generate_step

                responses = list(generate_step(
                    mx.array(tokens, dtype=mx.int32),
                    model,
                    max_tokens=max_tokens,
                    prefill_step_size=2048,
                ))
                if not responses:
                    raise ValueError("generate-step produced no response")
                generated_tokens = [
                    int(item[0].item() if hasattr(item[0], "item") else item[0])
                    if isinstance(item, tuple)
                    else int(item.token.item())
                    for item in responses
                ]
                response = responses[-1]
                if isinstance(response, tuple):
                    token, scores = response
                    _ = int(token.item() if hasattr(token, "item") else token)
                else:
                    scores = response.logprobs
                mx.eval(scores)
            elif mode == "stream-generate":
                from mlx_lm import stream_generate
                from mlx_lm.sample_utils import make_sampler

                responses = list(stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=rendered_prompt,
                    max_tokens=max_tokens,
                    sampler=make_sampler(temp=0.0, top_p=1.0),
                ))
                if not responses:
                    raise ValueError("stream-generate produced no response")
                generated_tokens = [
                    int(
                        response.token.item()
                        if hasattr(response.token, "item")
                        else response.token
                    )
                    for response in responses
                ]
                scores = responses[-1].logprobs
                mx.eval(scores)
            elif mode == "metadata":
                from moespresso.runtime.serve import generate_with_metadata

                result = generate_with_metadata(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=rendered_prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    top_logprobs=DEFAULT_TOP_LOGPROBS,
                )
                generated_tokens = list(result.generated_token_ids)
                token_logprobs = list(result.token_logprobs)
                top_logprobs = [list(step) for step in result.top_logprobs]
                if not generated_tokens:
                    raise ValueError("metadata generation produced no response")
                scores = mx.array([token_logprobs[-1]], dtype=mx.float32)
                mx.eval(scores)
            else:
                raise ValueError(f"unknown replay mode: {mode}")
        finally:
            _restore_layer_capture(original)
        stats = ssd_streaming_stats(model)
        return {
            "manifest": manifest,
            "tokens": tokens,
            "rendered_prompt": rendered_prompt,
            "layers": layers,
            "captures": captures,
            "scores": scores,
            "generated_tokens": generated_tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "stats": stats,
        }


def compare_graph_runs(
    direct: Mapping[str, Any],
    wrapped: Mapping[str, Any],
    *,
    stages: Sequence[str],
) -> dict[str, Any]:
    if direct["tokens"] != wrapped["tokens"]:
        raise ValueError("direct and wrapped token streams differ")
    if direct["rendered_prompt"] != wrapped["rendered_prompt"]:
        raise ValueError("direct and wrapped rendered prompts differ")
    rows = []
    for layer in direct["layers"]:
        stage_rows = {}
        for stage in stages:
            key = f"{layer}:{stage}"
            ref = direct["captures"][key]
            got = wrapped["captures"][key]
            width = 4096 * 4 if stage in {"hc_attn_post", "hc_ffn_post"} else 4096
            ref_rows = ref.reshape(-1, width)
            got_rows = got.reshape(-1, width)
            stage_rows[stage] = {
                "all": _metric(got_rows, ref_rows),
                "final": _metric(got_rows[-1], ref_rows[-1]),
            }
        rows.append({"layer": int(layer), "stages": stage_rows})
    scores = _score_summary(direct["scores"], wrapped["scores"])
    direct_top_logprob_ids = _top_logprob_signature(direct.get("top_logprobs") or [])
    wrapped_top_logprob_ids = _top_logprob_signature(wrapped.get("top_logprobs") or [])
    return {
        "prompt_tokens": len(direct["tokens"]),
        "package_manifest_id": direct["manifest"].get("artifact_id"),
        "layers": rows,
        "scores": scores,
        "direct_generated_tokens": list(direct.get("generated_tokens") or []),
        "wrapped_generated_tokens": list(wrapped.get("generated_tokens") or []),
        "generated_tokens_equal": (
            list(direct.get("generated_tokens") or [])
            == list(wrapped.get("generated_tokens") or [])
        ),
        "direct_token_logprobs": list(direct.get("token_logprobs") or []),
        "wrapped_token_logprobs": list(wrapped.get("token_logprobs") or []),
        "direct_top_logprob_token_ids": direct_top_logprob_ids,
        "wrapped_top_logprob_token_ids": wrapped_top_logprob_ids,
        "top_logprob_token_ids_equal": direct_top_logprob_ids == wrapped_top_logprob_ids,
        "first_non_equal": _first_non_equal(rows),
        "direct_ssd_streaming_stats": dict(direct["stats"]),
        "wrapped_ssd_streaming_stats": dict(wrapped["stats"]),
    }


def run_graph_replay(
    *,
    package_dir: Path,
    prompt_path: Path,
    prompt_mode: str,
    layers: list[int],
    stages: Sequence[str],
    mode: str = "full",
    max_tokens: int = 1,
) -> dict[str, Any]:
    direct = _load_capture_run(
        package_dir=package_dir,
        prompt_path=prompt_path,
        prompt_mode=prompt_mode,
        layers=layers,
        stages=stages,
        wrap_moe_block=False,
        mode=mode,
        max_tokens=max_tokens,
    )
    direct_layers = list(direct["layers"])
    _cleanup_mlx()
    wrapped = _load_capture_run(
        package_dir=package_dir,
        prompt_path=prompt_path,
        prompt_mode=prompt_mode,
        layers=direct_layers,
        stages=stages,
        wrap_moe_block=True,
        mode=mode,
        max_tokens=max_tokens,
    )
    comparison = compare_graph_runs(direct, wrapped, stages=stages)
    return {
        "artifact_kind": "deepseek_v4_moe_graph_replay",
        "package_dir": str(package_dir),
        "prompt_path": str(prompt_path),
        "prompt_mode": prompt_mode,
        "mode": mode,
        "max_tokens": (
            int(max_tokens)
            if mode in {"generate-step", "stream-generate", "metadata"}
            else None
        ),
        "requested_layers": "all" if not layers else list(layers),
        "stages": list(stages),
        **comparison,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moespresso-ds4-moe-graph-replay",
        description=(
            "Run one full-forward DS4 prompt through the normal direct graph and "
            "the opt-in DS4 pooled-MoE graph, then compare layer captures."
        ),
    )
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="raw-user")
    parser.add_argument("--layers", type=_layers_arg, default=[])
    parser.add_argument("--stages", type=_stages, default=",".join(DEFAULT_STAGES))
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    payload = run_graph_replay(
        package_dir=args.package,
        prompt_path=args.prompt_file,
        prompt_mode=args.prompt_mode,
        layers=args.layers,
        stages=args.stages,
        mode=args.mode,
        max_tokens=args.max_tokens,
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
