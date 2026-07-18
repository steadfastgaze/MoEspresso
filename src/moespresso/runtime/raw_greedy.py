"""Raw greedy generation over the pinned MLX LM cache contract.

The deterministic product path selects the next token directly from processed
logits. It preserves mlx-lm's prefill, cache advancement, asynchronous
lookahead, detokenization, and response accounting without constructing a
log-probability vector.

The generation loop adapts ``mlx_lm.generate.generate_step`` and
``mlx_lm.generate.stream_generate`` from MLX LM v0.31.3. See
``THIRD-PARTY-NOTICES`` for
source attribution.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Sequence
from typing import Any


# Copyright © 2023-2024 Apple Inc.
# Modification notice: this specializes MLX LM v0.31.3 generation for the
# deterministic MoEspresso route. It selects argmax directly from logits and
# omits the normalized full-vocabulary log-probability vector.


def raw_greedy_generate_step(
    prompt,
    model,
    *,
    max_tokens: int = 256,
    logits_processors: Sequence[Callable] | None = None,
    max_kv_size: int | None = None,
    prompt_cache: Any | None = None,
    prefill_step_size: int = 2048,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Callable[[int, int], None] | None = None,
) -> Generator[int, None, None]:
    """Yield raw-argmax token ids while updating ``prompt_cache`` in place."""
    import mlx.core as mx
    from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
    from mlx_lm.models.cache import make_prompt_cache

    if len(prompt) == 0:
        raise ValueError("raw greedy generation requires at least one prompt token")
    if prefill_step_size < 1:
        raise ValueError("prefill_step_size must be positive")

    token_history = None
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model, max_kv_size=max_kv_size)

    progress = prompt_progress_callback or (lambda *_: None)

    def quantize_cache() -> None:
        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=quantized_kv_start,
            kv_group_size=kv_group_size,
            kv_bits=kv_bits,
        )

    def model_call(input_tokens):
        return model(input_tokens[None], cache=prompt_cache)

    def step(input_tokens):
        nonlocal token_history

        with mx.stream(generation_stream):
            logits = model_call(input_tokens)[:, -1, :]
            if logits_processors and len(input_tokens) > 0:
                token_history = (
                    mx.concat([token_history, input_tokens])
                    if token_history is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(token_history, logits)
            quantize_cache()
            return mx.argmax(logits, axis=-1)

    with mx.stream(generation_stream):
        total_prompt_tokens = len(prompt)
        processed = 0
        progress(processed, total_prompt_tokens)
        while total_prompt_tokens - processed > 1:
            remaining = total_prompt_tokens - processed - 1
            count = min(prefill_step_size, remaining)
            model_call(prompt[:count])
            quantize_cache()
            mx.eval([entry.state for entry in prompt_cache])
            processed += count
            progress(processed, total_prompt_tokens)
            prompt = prompt[count:]
            mx.clear_cache()
        token = step(prompt)

    mx.async_eval(token)
    generated = 0
    while True:
        if generated != max_tokens:
            next_token = step(token)
            mx.async_eval(next_token)
        if generated == 0:
            mx.eval(token)
            progress(total_prompt_tokens, total_prompt_tokens)
        if generated == max_tokens:
            break
        yield token.item()
        if generated % 256 == 0:
            mx.clear_cache()
        token = next_token
        generated += 1


def stream_raw_greedy(
    model,
    tokenizer,
    prompt,
    max_tokens: int = 256,
    **kwargs,
) -> Generator[Any, None, None]:
    """Yield decoded raw-greedy responses with mlx-lm stream semantics."""
    import mlx.core as mx
    from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if max_tokens == 0 or max_tokens < -1:
        raise ValueError("max_tokens must be positive or -1")
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    prompt_tokens = int(prompt.size)
    detokenizer = tokenizer.detokenizer
    token_generator = raw_greedy_generate_step(
        prompt,
        model,
        max_tokens=max_tokens,
        **kwargs,
    )

    with wired_limit(model, [generation_stream]):
        started = time.perf_counter()
        for index, token in enumerate(token_generator):
            if index == 0:
                prompt_time = time.perf_counter() - started
                prompt_tps = prompt_tokens / prompt_time
                started = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break

            detokenizer.add_token(token)
            if index + 1 == max_tokens:
                break

            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=None,
                from_draft=False,
                prompt_tokens=prompt_tokens,
                prompt_tps=prompt_tps,
                generation_tokens=index + 1,
                generation_tps=(index + 1) / (time.perf_counter() - started),
                peak_memory=mx.get_peak_memory() / 1e9,
            )

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=None,
            from_draft=False,
            prompt_tokens=prompt_tokens,
            prompt_tps=prompt_tps,
            generation_tokens=index + 1,
            generation_tps=(index + 1) / (time.perf_counter() - started),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
        )
