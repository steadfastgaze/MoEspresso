"""Opt-in MTP hidden capture retains the ordinary target computation."""

import mlx.core as mx
import numpy as np
import pytest

from test_qwen4_model_shell import _model, _state_arrays


def test_widened_prefill_capture_is_pre_final_mixer_and_matches_ordinary_state():
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int64)
    captured_model, ordinary_model = _model([]), _model([])
    captured = captured_model.new_coordinator(1)
    ordinary = ordinary_model.new_coordinator(1)
    logits, hidden = captured.forward_chunk_with_widened(tokens)
    expected = ordinary.forward_chunk(tokens)
    assert hidden.shape == (1, 4, 4)
    assert logits.shape == (1, 4, 2)
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(captured_model.lm_head(captured_model.final_residual(hidden))),
                                  np.asarray(logits))
    for got, wanted in zip(_state_arrays(captured.state), _state_arrays(ordinary.state), strict=True):
        np.testing.assert_array_equal(got, wanted)


def test_widened_capture_preserves_alignment_across_prompt_chunks():
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int64)
    model = _model([])
    all_at_once = model.new_coordinator(1)
    _logits, all_hidden = all_at_once.forward_chunk_with_widened(tokens)
    chunked = _model([]).new_coordinator(1)
    parts = [chunked.forward_chunk_with_widened(tokens[:, :1])[1],
             chunked.forward_chunk_with_widened(tokens[:, 1:])[1]]
    np.testing.assert_array_equal(np.asarray(mx.concatenate(parts, axis=1)), np.asarray(all_hidden))


def test_widened_proposal_capture_preserves_partial_commit_and_continuation():
    model, reference_model = _model([]), _model([])
    coordinator, reference = model.new_coordinator(1), reference_model.new_coordinator(1)
    tokens = mx.array([[1, 2, 3]], dtype=mx.int64)
    candidate = coordinator.propose(tokens, capture_widened=True)
    ordinary = reference.propose(tokens)
    assert candidate.widened.shape == (1, 3, 4)
    assert ordinary.widened is None
    np.testing.assert_array_equal(np.asarray(candidate.logits), np.asarray(ordinary.logits))
    coordinator.commit(candidate, 1)
    reference.commit(ordinary, 1)
    continuation = mx.array([[4]], dtype=mx.int64)
    got = coordinator.propose(continuation, capture_widened=True)
    expected = reference.propose(continuation)
    np.testing.assert_array_equal(np.asarray(got.logits), np.asarray(expected.logits))
    np.testing.assert_array_equal(np.asarray(model.lm_head(model.final_residual(got.widened))),
                                  np.asarray(got.logits))


def test_widened_capture_does_not_bypass_cache_routing_guard():
    model = _model([])
    model._cache_routing_enabled = True
    coordinator = model.new_coordinator(1)
    with pytest.raises(ValueError, match="single-token"):
        coordinator.propose(mx.array([[1, 2]]), capture_widened=True)


def test_widened_capture_failure_keeps_the_previous_committed_state():
    model = _model([])
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk_with_widened(mx.array([[1, 2]]))
    previous = coordinator.state
    expected = [value.copy() for value in _state_arrays(previous)]

    def fail(_hidden):
        raise RuntimeError("injected target head failure")

    model.lm_head = fail
    with pytest.raises(RuntimeError, match="target head failure"):
        coordinator.forward_chunk_with_widened(mx.array([[3]]))
    assert coordinator.state is previous
    for got, wanted in zip(_state_arrays(previous), expected, strict=True):
        np.testing.assert_array_equal(got, wanted)
