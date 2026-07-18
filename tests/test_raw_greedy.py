"""Behavioral checks for the MoEspresso-owned deterministic generation path."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.tokenizer_utils import TokenizerWrapper

import moespresso.runtime.raw_greedy as raw_greedy


class _CountingCache:
    def __init__(self):
        self.offset = 0

    @property
    def state(self):
        return mx.array([self.offset])


class _NextTokenModel:
    """Emit a unique argmax at input token plus one and count cached rows."""

    def __call__(self, tokens, *, cache):
        for entry in cache:
            entry.offset += int(tokens.shape[1])
        targets = (tokens.astype(mx.int32) + 1) % 16
        vocabulary = mx.arange(16, dtype=mx.int32)
        return -mx.abs(vocabulary[None, None, :] - targets[:, :, None]).astype(
            mx.float32
        )


class _TextTokenizer:
    eos_token_id = 9
    bos_token = None
    chat_template = None
    clean_up_tokenization_spaces = False

    def get_vocab(self):
        return {}

    def encode(self, text, *, add_special_tokens=True):
        del add_special_tokens
        return [ord(char) - ord("A") for char in text]

    def decode(self, tokens):
        return "".join(chr(ord("A") + int(token)) for token in tokens)


def test_raw_greedy_matches_stock_tokens_callbacks_and_cache_frontier():
    prompt = mx.array([2, 4, 6], dtype=mx.int32)
    raw_cache = [_CountingCache()]
    stock_cache = [_CountingCache()]
    raw_progress = []
    stock_progress = []

    raw_tokens = list(
        raw_greedy.raw_greedy_generate_step(
            prompt,
            _NextTokenModel(),
            max_tokens=4,
            prompt_cache=raw_cache,
            prefill_step_size=2,
            prompt_progress_callback=lambda processed, total: raw_progress.append(
                (processed, total)
            ),
        )
    )
    stock_tokens = [
        token
        for token, _logprobs in generate_step(
            prompt,
            _NextTokenModel(),
            max_tokens=4,
            prompt_cache=stock_cache,
            prefill_step_size=2,
            prompt_progress_callback=lambda processed, total: stock_progress.append(
                (processed, total)
            ),
        )
    ]

    assert raw_tokens == stock_tokens == [7, 8, 9, 10]
    assert raw_progress == stock_progress == [(0, 3), (2, 3), (3, 3)]
    assert raw_cache[0].offset == stock_cache[0].offset == len(prompt) + len(raw_tokens)


def test_raw_greedy_never_normalizes_the_vocabulary(monkeypatch):
    def fail_logsumexp(*_args, **_kwargs):
        raise AssertionError("raw greedy generation must not normalize logits")

    monkeypatch.setattr(mx, "logsumexp", fail_logsumexp)
    tokens = list(
        raw_greedy.raw_greedy_generate_step(
            mx.array([2, 4, 6], dtype=mx.int32),
            _NextTokenModel(),
            max_tokens=3,
            prompt_cache=[_CountingCache()],
        )
    )

    assert tokens == [7, 8, 9]


@contextmanager
def _no_wired_limit(*_args, **_kwargs):
    yield


def test_stream_raw_greedy_preserves_eos_response_semantics(monkeypatch):
    monkeypatch.setattr(
        raw_greedy,
        "raw_greedy_generate_step",
        lambda *_args, **_kwargs: iter([1, 2, 9, 7]),
    )
    mlx_generate = import_module("mlx_lm.generate")
    monkeypatch.setattr(mlx_generate, "wired_limit", _no_wired_limit)
    responses = list(
        raw_greedy.stream_raw_greedy(
            object(),
            TokenizerWrapper(_TextTokenizer()),
            [3, 4],
            max_tokens=4,
        )
    )

    assert [response.token for response in responses] == [1, 2, 9]
    assert "".join(response.text for response in responses) == "BC"
    assert responses[-1].generation_tokens == 3
    assert responses[-1].finish_reason == "stop"
    assert all(response.logprobs is None for response in responses)


def test_stream_raw_greedy_preserves_length_response_semantics(monkeypatch):
    monkeypatch.setattr(
        raw_greedy,
        "raw_greedy_generate_step",
        lambda *_args, **_kwargs: iter([1, 2, 3, 4]),
    )
    mlx_generate = import_module("mlx_lm.generate")
    monkeypatch.setattr(mlx_generate, "wired_limit", _no_wired_limit)
    responses = list(
        raw_greedy.stream_raw_greedy(
            object(),
            TokenizerWrapper(_TextTokenizer()),
            [3, 4],
            max_tokens=3,
        )
    )

    assert [response.token for response in responses] == [1, 2, 3]
    assert "".join(response.text for response in responses) == "BCD"
    assert responses[-1].generation_tokens == 3
    assert responses[-1].finish_reason == "length"
