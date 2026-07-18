"""Raw prefix-cache correctness on a tiny real MLX model.

This does not load a full MoEspresso package. It exercises MLX's real cache classes on a
small Qwen3.5/3.6-style hybrid model: scratch prompt vs prefilled-prefix + suffix must
produce the same next-token logprobs/top token. That is the minimum correctness evidence
before MoEspresso's HTTP path enables raw prefix reuse.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402

from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from moespresso.runtime.generation import GenerationResult  # noqa: E402
from moespresso.runtime.kv_policy import parse_kv_policy, stream_generate_kv_kwargs  # noqa: E402
from moespresso.runtime.prefix_cache import (  # noqa: E402
    PrefixCacheGenerator,
    make_prompt_cache_store,
)


_TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "num_experts": 2,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "rms_norm_eps": 1e-6,
    "vocab_size": 128,
    "rope_theta": 10000.0,
    "partial_rotary_factor": 0.25,
    "max_position_embeddings": 512,
    "linear_num_value_heads": 2,
    "linear_num_key_heads": 1,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 2,
    "tie_word_embeddings": False,
    "layer_types": ["linear_attention", "full_attention"],
}


def _greedy(logprobs):
    return mx.argmax(logprobs, axis=-1)


def _tiny_model(text_config_overrides=None):
    import mlx_lm.models.qwen3_5_moe as M
    mx.random.seed(123)
    text_config = dict(_TEXT_CONFIG)
    text_config.update(text_config_overrides or {})
    args = M.ModelArgs(model_type="qwen3_5_moe", text_config=text_config)
    model = M.Model(args)
    mx.eval(model.parameters())
    return model


def _first_token_and_logprobs(model, prompt, cache=None, **generate_kwargs):
    gen = generate_step(
        mx.array(prompt, dtype=mx.uint32),
        model,
        max_tokens=1,
        sampler=_greedy,
        prompt_cache=cache,
        **generate_kwargs,
    )
    token, logprobs = next(gen)
    mx.eval(token, logprobs)
    token_id = int(token.item() if hasattr(token, "item") else token)
    return token_id, np.array(logprobs, dtype=np.float32)


def _prefill(model, prefix, cache):
    model(mx.array([prefix], dtype=mx.uint32), cache=cache)
    mx.eval([c.state for c in cache])


class _TokenMap:
    def __init__(self, mapping):
        self.mapping = {k: list(v) for k, v in mapping.items()}
        self.bos_token = None

    def encode(self, text, **kwargs):
        return list(self.mapping[text])


def _generate_one_token(model, tokenizer, prompt, **kwargs):
    kv_kwargs = {}
    if "kv_policy" in kwargs:
        kv_kwargs = stream_generate_kv_kwargs(kwargs["kv_policy"])
    token, logprobs = _first_token_and_logprobs(
        model,
        prompt,
        cache=kwargs["prompt_cache"],
        **kv_kwargs,
    )
    result = GenerationResult(
        text="",
        finish_reason="stop",
        prompt_tokens=len(prompt),
        completion_tokens=1,
        cached_tokens=kwargs.get("cached_tokens"),
        generated_token_ids=(token,),
        prompt_cache=kwargs["prompt_cache"],
    )
    result.logprobs = logprobs
    return result


def test_raw_prefix_cache_matches_scratch_logits_on_tiny_qwen_hybrid_model():
    model = _tiny_model()
    cache = make_prompt_cache(model)
    names = {type(c).__name__ for c in cache}
    assert {"ArraysCache", "KVCache"}.issubset(names)

    prompt = [1, 5, 9, 13, 2, 7]
    prefix = prompt[:3]
    suffix = prompt[3:]

    scratch_token, scratch_logprobs = _first_token_and_logprobs(model, prompt)

    _prefill(model, prefix, cache)
    cached_token, cached_logprobs = _first_token_and_logprobs(model, suffix, cache)

    assert cached_token == scratch_token
    assert float(np.mean((cached_logprobs - scratch_logprobs) ** 2)) < 1e-10
    assert float(np.max(np.abs(cached_logprobs - scratch_logprobs))) < 1e-5


def test_raw_prefix_cache_generator_matches_scratch_logits_on_tiny_qwen_hybrid_model():
    model = _tiny_model()
    prompt1 = [1, 5, 9]
    tokenizer = _TokenMap({"turn1": prompt1})
    generator = PrefixCacheGenerator(
        model,
        tokenizer,
        {"artifact_id": "tiny-pkg"},
        make_prompt_cache_store(max_size=4),
        generate_fn=_generate_one_token,
    )
    policy = parse_kv_policy({"live_kv_format": "raw"})

    first = generator("turn1", kv_policy=policy, effective_rendering_id="render-a")
    prompt2 = prompt1 + list(first.generated_token_ids) + [13, 2, 7]
    tokenizer.mapping["turn2"] = prompt2

    scratch_token, scratch_logprobs = _first_token_and_logprobs(model, prompt2)
    cached = generator("turn2", kv_policy=policy, effective_rendering_id="render-a")

    assert cached.cached_tokens == len(prompt1) + len(first.generated_token_ids)
    assert cached.generated_token_ids == (scratch_token,)
    assert float(np.mean((cached.logprobs - scratch_logprobs) ** 2)) < 1e-10
    assert float(np.max(np.abs(cached.logprobs - scratch_logprobs))) < 1e-5


def test_q8_live_kv_keeps_top_token_and_quantizes_only_full_attention_cache():
    model = _tiny_model({"head_dim": 32})
    prompt = [1, 5, 9, 13, 2, 7]

    raw_cache = make_prompt_cache(model)
    q8_cache = make_prompt_cache(model)
    assert [type(c).__name__ for c in raw_cache] == ["ArraysCache", "KVCache"]

    raw_token, raw_logprobs = _first_token_and_logprobs(model, prompt, cache=raw_cache)
    q8_token, q8_logprobs = _first_token_and_logprobs(
        model,
        prompt,
        cache=q8_cache,
        kv_bits=8,
        kv_group_size=32,
        quantized_kv_start=0,
    )

    assert [type(c).__name__ for c in q8_cache] == ["ArraysCache", "QuantizedKVCache"]
    assert q8_token == raw_token
    assert float(np.mean((q8_logprobs - raw_logprobs) ** 2)) < 1e-4
