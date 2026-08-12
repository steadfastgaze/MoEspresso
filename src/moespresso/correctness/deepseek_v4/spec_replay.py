"""A/B replay for speculative decoding against plain decoding.

Loads the DeepSeek-V4 target package and a drafter sidecar, runs the same
prompt through plain greedy decoding and speculative decoding, and
reports token identity, acceptance statistics, and wall-clock speed.
Speculative decoding must be lossless: with temperature 0 the two token
streams are expected to match exactly; any divergence is reported with
its position so knife-edge argmax flips can be inspected.

Both arms share the same prefill schedule (chunked over all but the final
prompt token, then a single-token forward) so the comparison isolates the
decode strategy.

`--drafter` selects the drafter family. DSpark and DFlash are the wired
families; each shares the target's embedding, DSpark also shares the
language-model head, and each installs its own hidden tap. DFlash is
greedy-only: it proposes argmax tokens with no draft distribution over
the target vocabulary, so `--temperature` above 0 is rejected for it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

import mlx.core as mx

from moespresso.runtime.deepseek_v4.spec_decode import (
    install_hidden_tap,
    spec_generate,
)
from moespresso.runtime.serve import load_served_model

DRAFTER_FAMILIES = ("dspark", "dflash")


def build_drafter(family: str, sidecar_dir: Path, model):
    """Load the drafter sidecar for `family`, sharing the target's embed
    (and lm head where the drafter has none). Fails closed on an unknown
    family."""
    if family == "dspark":
        from moespresso.runtime.deepseek_v4.dspark_load import load_dspark_sidecar

        drafter, _ = load_dspark_sidecar(
            sidecar_dir, embed=model.model.embed, lm_head=model.lm_head
        )
        return drafter
    if family == "dflash":
        from moespresso.runtime.deepseek_v4.dflash_load import load_dflash_sidecar

        drafter, _ = load_dflash_sidecar(sidecar_dir, embed=model.model.embed)
        return drafter
    raise ValueError(f"unknown drafter family: {family!r}")


def _prefill_step_size(model, fallback: int = 2048) -> int:
    return int(getattr(model, "_moespresso_prefill_step_size", fallback) or fallback)


def plain_greedy_generate(
    model,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    eos_ids: Optional[Sequence[int]] = None,
    prefill_step_size: int = 2048,
) -> tuple[List[int], float, float]:
    """Plain one-token-per-step greedy decode. Returns (tokens,
    prefill_seconds, decode_seconds)."""
    eos = set(eos_ids or [])
    cache = model.make_cache()
    prompt = mx.array(list(prompt_ids), dtype=mx.int64)[None]
    n_prompt = prompt.shape[1]

    t0 = time.perf_counter()
    for start in range(0, n_prompt - 1, prefill_step_size):
        chunk = prompt[:, start : min(start + prefill_step_size, n_prompt - 1)]
        model(chunk, cache=cache)
        mx.eval(*(c.state for c in cache if c is not None))
    logits = model(prompt[:, -1:], cache=cache)
    token = int(mx.argmax(logits[0, -1].astype(mx.float32)))
    t1 = time.perf_counter()

    out = [token]
    while len(out) < max_new_tokens and token not in eos:
        logits = model(mx.array([[token]], dtype=mx.int64), cache=cache)
        token = int(mx.argmax(logits[0, -1].astype(mx.float32)))
        out.append(token)
    from moespresso.runtime.deepseek_v4.model import (
        evaluate_deepseek_v4_cache_state,
    )

    evaluate_deepseek_v4_cache_state(cache, asynchronous=False)
    t2 = time.perf_counter()
    return out, t1 - t0, t2 - t1


def run_replay(
    package_dir: Path,
    sidecar_dir: Path,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    confidence_threshold: float = 0.0,
    skip_plain: bool = False,
    adaptive_cap: bool = True,
    drafter_family: str = "dspark",
) -> dict:
    model, tokenizer, manifest = load_served_model(package_dir)
    drafter = build_drafter(drafter_family, sidecar_dir, model)
    if temperature > 0 and getattr(drafter, "greedy_only", False):
        raise ValueError(
            f"the {drafter_family} drafter is greedy-only; run with "
            "--temperature 0")
    tap = install_hidden_tap(model, drafter.tap_layer_ids, drafter.tap_transform)

    prompt_ids = tokenizer.encode(prompt_text)
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    step = _prefill_step_size(model)

    report: dict = {
        "package": str(package_dir),
        "sidecar": str(sidecar_dir),
        "drafter": drafter_family,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "block_size": drafter.block_size,
    }

    t0 = time.perf_counter()
    spec = spec_generate(
        model,
        drafter,
        tap,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_ids=eos_ids,
        confidence_threshold=confidence_threshold,
        prefill_step_size=step,
        adaptive_cap=adaptive_cap,
    )
    spec_seconds = time.perf_counter() - t0
    stats = spec.stats
    report["adaptive_cap"] = adaptive_cap
    report["speculative"] = {
        "tokens_generated": len(spec.tokens),
        "total_seconds": spec_seconds,
        "tok_per_s": len(spec.tokens) / max(spec_seconds, 1e-9),
        "rounds": stats.rounds,
        "proposed": stats.proposed,
        "accepted": stats.accepted,
        "mean_accepted_length": stats.mean_accepted_length,
        "per_position_offered": stats.per_position_offered,
        "per_position_accepted": stats.per_position_accept,
        "per_position_matched": stats.per_position_matched,
        "plain_fallbacks": stats.plain_fallbacks,
        "submit_length_counts": {
            str(k): v for k, v in sorted(stats.submit_length_counts.items())
        },
        "text": tokenizer.decode(spec.tokens),
    }

    if not skip_plain and temperature <= 0:
        plain_tokens, prefill_s, decode_s = plain_greedy_generate(
            model, prompt_ids, max_new_tokens, eos_ids, step
        )
        n = len(plain_tokens)
        report["plain"] = {
            "tokens_generated": n,
            "prefill_seconds": prefill_s,
            "decode_seconds": decode_s,
            "decode_tok_per_s": n / max(decode_s, 1e-9),
            "text": tokenizer.decode(plain_tokens),
        }
        divergence = next(
            (
                i
                for i, (a, b) in enumerate(zip(spec.tokens, plain_tokens))
                if a != b
            ),
            None,
        )
        identical = divergence is None and len(spec.tokens) == len(plain_tokens)
        report["identity"] = {
            "tokens_equal": identical,
            "first_divergence": divergence,
            "spec_len": len(spec.tokens),
            "plain_len": len(plain_tokens),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--drafter", choices=DRAFTER_FAMILIES, default="dspark",
                        help="drafter family to load from the sidecar")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--no-adaptive", action="store_true",
                        help="fixed full-block schedule instead of the "
                             "adaptive verify-length cap")
    parser.add_argument("--skip-plain", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if (args.prompt is None) == (args.prompt_file is None):
        parser.error("provide exactly one of --prompt or --prompt-file")
    prompt_text = (
        args.prompt
        if args.prompt is not None
        else args.prompt_file.read_text(encoding="utf-8")
    )

    report = run_replay(
        args.package,
        args.sidecar,
        prompt_text,
        args.max_new_tokens,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
        skip_plain=args.skip_plain,
        adaptive_cap=not args.no_adaptive,
        drafter_family=args.drafter,
    )
    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    identity = report.get("identity")
    if identity is not None and not identity["tokens_equal"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
