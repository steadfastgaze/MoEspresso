"""Served prefill chunk override for the qwen3_5_moe sorted routed-MoE path.

The mechanism is default-on for long prompts: the shipped default raises the
prompt chunk to 4096 and gates it to prompts longer than the chunk, so short
prompts keep the mlx_lm chunk. These tests prove the shipped default sets the
4096 chunk with the long-prompt gate, that `MOESPRESSO_QWEN_PREFILL_CHUNK`
overrides the value, that the no-op values (the mlx_lm default and 0) leave the
attributes unset, and that a malformed value fails closed. The served speed
effect and token rail belong to the campaign-level quality gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from moespresso.runtime.qwen.prefill_chunk import (
    _MLX_DEFAULT_PREFILL_STEP,
    _QWEN_PREFILL_CHUNK_DEFAULT,
    install_prefill_chunk,
)


def _model():
    return SimpleNamespace()


def test_default_promotes_chunk_for_long_prompts(monkeypatch):
    # The shipped default raises the chunk to 4096 and gates it to prompts
    # longer than the chunk, so short prompts keep the mlx_lm chunk.
    monkeypatch.delenv("MOESPRESSO_QWEN_PREFILL_CHUNK", raising=False)
    assert _QWEN_PREFILL_CHUNK_DEFAULT == 4096
    assert _QWEN_PREFILL_CHUNK_DEFAULT != _MLX_DEFAULT_PREFILL_STEP
    model = _model()
    result = install_prefill_chunk(model)
    assert result == 4096
    assert model._moespresso_prefill_step_size == 4096
    # The long-prompt gate is the chunk plus one, so a prompt of exactly the
    # chunk length stays on the stock path.
    assert model._moespresso_prefill_step_size_min_prompt_tokens == 4097


def test_env_override_sets_requested_chunk(monkeypatch):
    monkeypatch.setenv("MOESPRESSO_QWEN_PREFILL_CHUNK", "8192")
    model = _model()
    result = install_prefill_chunk(model)
    assert result == 8192
    assert model._moespresso_prefill_step_size == 8192
    # The long-prompt gate tracks the resolved chunk.
    assert model._moespresso_prefill_step_size_min_prompt_tokens == 8193


def test_kill_switch_default_value_is_noop(monkeypatch):
    # Setting the env to the mlx_lm default is the kill switch: the attributes
    # are left unset so serving is byte-for-byte the stock chunk granularity.
    monkeypatch.setenv(
        "MOESPRESSO_QWEN_PREFILL_CHUNK", str(_MLX_DEFAULT_PREFILL_STEP)
    )
    model = _model()
    result = install_prefill_chunk(model)
    assert result is None
    assert not hasattr(model, "_moespresso_prefill_step_size")
    assert not hasattr(model, "_moespresso_prefill_step_size_min_prompt_tokens")


def test_kill_switch_zero_is_noop(monkeypatch):
    monkeypatch.setenv("MOESPRESSO_QWEN_PREFILL_CHUNK", "0")
    model = _model()
    result = install_prefill_chunk(model)
    assert result is None
    assert not hasattr(model, "_moespresso_prefill_step_size")
    assert not hasattr(model, "_moespresso_prefill_step_size_min_prompt_tokens")


@pytest.mark.parametrize("bad", ["-1", "notanint", "  ", "1.5"])
def test_malformed_value_fails_closed(monkeypatch, bad):
    monkeypatch.setenv("MOESPRESSO_QWEN_PREFILL_CHUNK", bad)
    model = _model()
    result = install_prefill_chunk(model)
    assert result is None
    assert not hasattr(model, "_moespresso_prefill_step_size")
    assert not hasattr(model, "_moespresso_prefill_step_size_min_prompt_tokens")


def test_idempotent_same_env(monkeypatch):
    monkeypatch.setenv("MOESPRESSO_QWEN_PREFILL_CHUNK", "16384")
    model = _model()
    first = install_prefill_chunk(model)
    second = install_prefill_chunk(model)
    assert first == second == 16384
    assert model._moespresso_prefill_step_size == 16384
    assert model._moespresso_prefill_step_size_min_prompt_tokens == 16385
