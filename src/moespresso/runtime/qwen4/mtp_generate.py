"""Experimental full-resident Qwen4 MTP generation with shared-weight verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import mlx.core as mx
from mlx_lm.generate import generation_stream, wired_limit

from moespresso.runtime.qwen4.mtp_drafter import Qwen4MTPDrafter
from moespresso.runtime.qwen4.mtp_compiled_verify import Qwen4MTPCompiledVerifier
from moespresso.runtime.qwen4.mtp_full_resident import Qwen4MTPFullResidentExpertPair
from moespresso.runtime.qwen4.mtp_verify import Qwen4MTPVerifier


def qwen4_mtp_compiled_enabled() -> bool:
    """Return whether eligible MTP rounds use the compiled target core."""

    return os.environ.get("MOESPRESSO_QWEN4_MTP_COMPILED", "1") != "0"


def make_mtp_verifier(model, **kwargs):
    """Construct the configured verifier for one fresh MTP request."""

    verifier_type = Qwen4MTPCompiledVerifier if qwen4_mtp_compiled_enabled() else Qwen4MTPVerifier
    return verifier_type(model, **kwargs)


def _require_full_resident_target(model):
    capacity = getattr(model, "_moespresso_ssd_streaming_capacity", None)
    bounded = getattr(model, "_moespresso_pooled_decode_bounded", None)
    cache_routing = getattr(model, "_cache_routing_enabled", None)
    if capacity != 512 or bounded is not False or cache_routing is not False:
        raise ValueError("Qwen4 MTP requires full target expert residency with cache routing off")


def generate_mtp(
    model,
    tokenizer,
    prompt_tokens,
    loaded,
    *,
    max_tokens=128,
    prefill_step_size=256,
    response_callback=None,
    cancelled=None,
    native_gdn_boundaries=True,
):
    """Generate from fresh request state; no speculative state enters cache tiers."""
    if max_tokens < 1 or prefill_step_size < 1 or not prompt_tokens:
        raise ValueError("MTP requires a nonempty prompt and positive generation sizes")
    limit = getattr(model, "_moespresso_qwen4_kvarn_context_tokens", None)
    if limit is not None and len(prompt_tokens) + max_tokens > limit:
        raise ValueError("prompt and requested completion exceed the configured context")
    _require_full_resident_target(model)
    drafter = Qwen4MTPDrafter(loaded)
    state = None
    verifier = None
    coordinator = None
    tokens, parts, arrivals, rounds = [], [], [], []
    started = time.perf_counter()
    first = None

    def check_cancel():
        if cancelled is not None and cancelled():
            raise InterruptedError("MTP generation cancelled")

    def log_probabilities(logits):
        probabilities = logits.astype(mx.float32)
        probabilities = probabilities - mx.logsumexp(probabilities, axis=-1, keepdims=True)
        finite = mx.all(mx.isfinite(probabilities))
        mx.eval(probabilities, finite)
        if not bool(finite.item()):
            raise ValueError("MTP target produced nonfinite log probabilities")
        return probabilities

    def emit(token, probabilities, *, from_draft=False):
        nonlocal first
        check_cancel()
        now = time.perf_counter()
        if first is None:
            first = now
        tokens.append(token)
        arrivals.append(now)
        response = _decode_response(
            detokenizer,
            token=token,
            logprobs=probabilities,
            is_stop=token in stops,
            is_length=len(tokens) == max_tokens,
            prompt_tokens=len(prompt_tokens),
            prompt_tps=len(prompt_tokens) / max(first - started, 1e-12),
            generated_tokens=len(tokens),
            generation_started=first,
        )
        if from_draft:
            from dataclasses import replace

            response = replace(response, from_draft=True)
        parts.append(response.text)
        if response_callback is not None:
            response_callback(len(tokens), response)

    try:
        from moespresso.runtime.qwen4.generation import _decode_response, _detokenizer, _stop_ids

        state = drafter.make_state()
        verifier = make_mtp_verifier(
            model,
            expert_pair_factory=Qwen4MTPFullResidentExpertPair,
            shared_projections=True,
            native_gdn_boundaries=native_gdn_boundaries,
        )
        detokenizer = _detokenizer(tokenizer)
        stops = _stop_ids(model, tokenizer)
        with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
            coordinator = model.new_coordinator(1)
            for begin in range(0, len(prompt_tokens), prefill_step_size):
                check_cancel()
                chunk = prompt_tokens[begin : begin + prefill_step_size]
                logits, hidden = coordinator.forward_chunk_with_widened(
                    mx.array([chunk], dtype=mx.int64)
                )
                drafter.ingest(state, hidden, list(range(begin, begin + len(chunk))), chunk)
            anchor = int(mx.argmax(logits[0, -1]).item())
            emit(anchor, log_probabilities(logits[0, -1]))
            while len(tokens) < max_tokens and tokens[-1] not in stops:
                check_cancel()
                tick = time.perf_counter()
                frontier = coordinator.state.frontier
                if max_tokens - len(tokens) == 1:
                    result = verifier.verify_plain(
                        coordinator, mx.array([[anchor]], dtype=mx.int64), cancelled=cancelled
                    )
                    emit(
                        int(mx.argmax(result.logits[0, 0]).item()),
                        log_probabilities(result.logits[0, 0]),
                    )
                    rounds.append(
                        {
                            "accepted": 0,
                            "drafted": False,
                            "seconds": time.perf_counter() - tick,
                            "emitted": 1,
                        }
                    )
                    break
                proposal = drafter.draft(state, anchor, frontier, 0)
                finite = mx.all(mx.isfinite(proposal.logits))
                mx.eval(proposal.tokens, finite)
                if not bool(finite.item()):
                    raise ValueError("MTP drafter produced nonfinite logits")
                draft = int(proposal.tokens.item())
                after_draft = time.perf_counter()
                result = verifier.verify(
                    coordinator, mx.array([[anchor, draft]], dtype=mx.int64), cancelled=cancelled
                )
                after_verify = time.perf_counter()
                emitted = result.acceptance.emitted
                keep = result.acceptance.accepted + 1
                terminal = (
                    any(token in stops for token in emitted)
                    or len(tokens) + len(emitted) >= max_tokens
                )
                if not terminal:
                    drafter.ingest(
                        state,
                        result.widened,
                        list(range(frontier, frontier + keep)),
                        [anchor, draft][:keep],
                    )
                before = len(tokens)
                probabilities = log_probabilities(result.logits[0])
                for row, token in enumerate(emitted):
                    check_cancel()
                    emit(token, probabilities[row], from_draft=row < result.acceptance.accepted)
                    if token in stops or len(tokens) == max_tokens:
                        break
                rounds.append(
                    {
                        "accepted": result.acceptance.accepted,
                        "drafted": True,
                        "seconds": time.perf_counter() - tick,
                        "draft_seconds": after_draft - tick,
                        "verify_seconds": after_verify - after_draft,
                        "emitted": len(tokens) - before,
                    }
                )
                anchor = result.acceptance.next_token
    finally:
        try:
            if verifier is not None:
                verifier.close()
        finally:
            try:
                if coordinator is not None:
                    coordinator.close()
            finally:
                if state is not None:
                    drafter.close_state(state)
    decode_seconds = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0
    drafted = sum(row["drafted"] for row in rounds)
    compiled_stats = getattr(verifier, "compiled_verification_stats", None)
    compiled_stats = (
        compiled_stats()
        if callable(compiled_stats)
        else {"compiled_rounds": 0, "fallback_rounds": 0}
    )
    shared_expert_count = getattr(verifier, "shared_expert_call_count", None)
    shared_expert_calls = (
        shared_expert_count()
        if callable(shared_expert_count)
        else sum(pair.paired_calls for pair in verifier.expert_pairs)
    )
    return {
        "tokens": tokens,
        "text": "".join(parts),
        "rounds": rounds,
        "first_token_seconds": first - started,
        "request_seconds": time.perf_counter() - started,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (len(tokens) - 1) / decode_seconds if decode_seconds else None,
        "acceptance_rate": sum(row["accepted"] for row in rounds) / drafted if drafted else None,
        "shared_expert_calls": shared_expert_calls,
        "staged_expert_calls": sum(
            getattr(pair, "staged_calls", 0) for pair in verifier.expert_pairs
        ),
        "shared_experts": None,
        "prefix_submissions": getattr(verifier, "prefix_submissions", 0),
        "compiled_verification_rounds": compiled_stats["compiled_rounds"],
        "fallback_verification_rounds": compiled_stats["fallback_rounds"],
        "expert_residency": "full",
        "cache_routing": "off",
        "native_gdn_boundaries": native_gdn_boundaries,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "routing_policy": "unbiased-full-resident",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    os.environ["MOESPRESSO_DISK_KV"] = "off"
    os.environ["MOESPRESSO_DS4_DRAFTER"] = "off"
    os.environ["MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB"] = "0"
    if args.memory_gb is not None:
        os.environ["MOESPRESSO_SSD_MAX_MEMORY_GB"] = str(args.memory_gb)
    from moespresso.core.artifact import read_artifact
    from moespresso.runtime.build import _silence_known_transformers_warnings
    from moespresso.package.qwen4.mtp_format import read_mtp_sidecar_manifest
    from moespresso.runtime.http import qwen4_contract_template_kwargs, render_prompt
    from moespresso.runtime.prefix_cache import encode_rendered_prompt
    from moespresso.runtime.qwen4.load import load_qwen4_iqk_package_model
    from moespresso.runtime.qwen4.mtp_load import load_qwen4_mtp_sidecar

    _silence_known_transformers_warnings()
    manifest = read_artifact(args.package / "package_manifest.json")
    sidecar = read_mtp_sidecar_manifest(args.sidecar, manifest)
    graph = sidecar["graph"]
    draft_state_bytes = args.max_context_tokens * (
        (
            2 * graph["num_key_value_heads"] * graph["head_dim"]
            + graph["indexer_kv_heads"] * graph["indexer_head_dim"]
        )
        * 4
        + 24
    )
    reservation = sidecar["byte_estimate"]["iq2_k_payload_with_input_padding"] + draft_state_bytes
    model = None
    try:
        model, tokenizer = load_qwen4_iqk_package_model(
            manifest,
            args.package,
            additional_resident_bytes=reservation,
            max_context_tokens=args.max_context_tokens,
            cache_routing="off",
        )
        _require_full_resident_target(model)
        loaded = load_qwen4_mtp_sidecar(args.sidecar, manifest, target_model=model)
        rendered = render_prompt(
            [{"role": "user", "content": args.prompt}],
            tokenizer,
            template_kwargs=qwen4_contract_template_kwargs(
                tokenizer, "off", family=manifest["architecture"]["family"]
            ),
            prompt_renderer=manifest["architecture"].get("prompt_renderer"),
        )
        prompt = encode_rendered_prompt(tokenizer, rendered)
        result = generate_mtp(
            model,
            tokenizer,
            prompt,
            loaded,
            max_tokens=args.max_tokens,
            response_callback=lambda _i, response: print(response.text, end="", flush=True),
        )
        result.update(
            package_artifact=manifest["artifact_id"],
            sidecar_artifact=sidecar["artifact_id"],
            device=mx.device_info(),
            context_tokens=args.max_context_tokens,
            target_capacity_per_layer=model._moespresso_ssd_streaming_capacity,
            cache_routing="off",
            drafter_reserved_bytes=reservation,
        )
        print(
            "\n"
            + json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"text", "tokens", "rounds"}
                }
            ),
            file=sys.stderr,
        )
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if model is not None:
            model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
