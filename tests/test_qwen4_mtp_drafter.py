"""Shifted target/MTP state alignment, acceptance and request isolation."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from moespresso.runtime.qwen4.mtp_drafter import Qwen4MTPDrafter
from moespresso.runtime.qwen4.mtp_load import LoadedQwen4MTP

from test_qwen4_mtp import _head, _inputs


def _drafter():
    head = _head()
    return Qwen4MTPDrafter(LoadedQwen4MTP(
        head=head, embedding=nn.Embedding(13, 8), artifact_id="synthetic-drafter",
        target_cache_identity="synthetic-target", payload_bytes=0, vocab_size=13,
    ))


def _equal_attention(actual, expected):
    assert actual.frontier == expected.frontier
    for name in ("keys", "values", "raw_index_keys", "position_ids"):
        np.testing.assert_allclose(np.asarray(getattr(actual.attention, name)),
                                   np.asarray(getattr(expected.attention, name)), rtol=2e-5, atol=2e-6)


def _reference(drafter, rows, tokens):
    return drafter.loaded.head.append_context(
        rows[:, :-1], drafter.loaded.embedding(mx.array([tokens[1:]], dtype=mx.int32)),
    )


def test_mtp_context_only_matches_full_forward_without_unused_moe_or_head():
    head = _head()
    rows, embeddings = _inputs()
    full = head(rows, embeddings)

    def unused(_values):
        raise AssertionError("context ingestion must not run the MTP MoE or language head")

    head.layers[0].mlp = unused
    head.lm_head = unused
    context = head.append_context(rows, embeddings)
    _equal_attention(context, full.state)


@pytest.mark.parametrize("cut", [1, 2, 3, 4])
def test_mtp_ingest_preserves_shifted_pairs_across_prompt_chunks(cut):
    drafter = _drafter()
    state = drafter.make_state()
    rows, _embeddings = _inputs()
    tokens = [1, 2, 3, 4, 5]
    drafter.ingest(state, rows[:, :cut], list(range(cut)), tokens[:cut])
    drafter.ingest(state, rows[:, cut:], list(range(cut, 5)), tokens[cut:])
    _equal_attention(state.attention, _reference(drafter, rows, tokens))
    np.testing.assert_array_equal(np.asarray(state.pending_hidden), np.asarray(rows[:, -1:]))
    assert state.frontier == 5
    assert state.attention.frontier == 4


def test_mtp_one_token_prompt_and_draft_do_not_commit_a_speculative_frontier():
    drafter = _drafter()
    state = drafter.make_state()
    rows, _embeddings = _inputs(1)
    drafter.ingest(state, rows, [0], [1])
    assert state.attention is None and state.frontier == 1
    proposal = drafter.draft(state, 2, 1, 0)
    expected = drafter.loaded.head(rows, drafter.loaded.embedding(mx.array([[2]])))
    np.testing.assert_array_equal(np.asarray(proposal.logits), np.asarray(expected.logits))
    np.testing.assert_array_equal(np.asarray(proposal.tokens), np.asarray(mx.argmax(expected.logits, axis=-1)))
    assert state.attention is None and state.frontier == 1
    assert state.anchor.attention.frontier == 1


@pytest.mark.parametrize("accepted", [0, 1])
def test_mtp_reconciles_verified_rows_and_reuses_only_the_authoritative_anchor(accepted, monkeypatch):
    drafter = _drafter()
    state = drafter.make_state()
    rows, _embeddings = _inputs(5)
    tokens = [1, 2, 3]
    drafter.ingest(state, rows[:, :3], [0, 1, 2], tokens)
    proposal = drafter.draft(state, 4, 3, 0)
    proposed = int(proposal.tokens.item())
    committed = [4, proposed][:1 + accepted]
    start = state.frontier
    append = drafter.loaded.head.append_context
    observed = []

    def record(inputs, embeddings, *, state):
        observed.append(inputs)
        return append(inputs, embeddings, state=state)

    monkeypatch.setattr(drafter.loaded.head, "append_context", record)
    drafter.ingest(state, rows[:, 3:4 + accepted], list(range(start, start + 1 + accepted)), committed)
    assert len(observed) == accepted
    if accepted:
        np.testing.assert_array_equal(np.asarray(observed[0]), np.asarray(rows[:, 3:4]))
    monkeypatch.setattr(drafter.loaded.head, "append_context", append)
    _equal_attention(state.attention, _reference(drafter, rows[:, :4 + accepted], tokens + committed))
    assert state.frontier == 4 + accepted
    assert state.anchor is None
    next_proposal = drafter.draft(state, 6, state.frontier, 0)
    assert next_proposal.tokens.shape == (1, 1)


def test_mtp_mismatched_or_discarded_anchor_falls_back_to_authoritative_ingest():
    drafter = _drafter()
    rows, _embeddings = _inputs(4)
    for discard in (False, True):
        state = drafter.make_state()
        drafter.ingest(state, rows[:, :3], [0, 1, 2], [1, 2, 3])
        drafter.draft(state, 5, 3, 0)
        if discard:
            drafter.discard_draft(state)
        drafter.ingest(state, rows[:, 3:], [3], [4])
        _equal_attention(state.attention, _reference(drafter, rows, [1, 2, 3, 4]))


def test_mtp_failed_ingest_preserves_committed_state(monkeypatch):
    drafter = _drafter()
    state = drafter.make_state()
    rows, _embeddings = _inputs(4)
    drafter.ingest(state, rows[:, :3], [0, 1, 2], [1, 2, 3])
    previous = state.attention, state.pending_hidden, state.frontier

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected MTP append failure")

    monkeypatch.setattr(drafter.loaded.head, "append_context", fail)
    with pytest.raises(RuntimeError, match="append failure"):
        drafter.ingest(state, rows[:, 3:], [3], [4])
    assert state.attention is previous[0] and state.pending_hidden is previous[1]
    assert state.frontier == previous[2]


def test_mtp_closed_foreign_and_misaligned_request_states_are_refused():
    drafter = _drafter()
    state = drafter.make_state()
    rows, _embeddings = _inputs(1)
    with pytest.raises(ValueError, match="contiguous"):
        drafter.ingest(state, rows, [1], [1])
    with pytest.raises(ValueError, match="vocabulary"):
        drafter.ingest(state, rows, [0], [13])
    drafter.ingest(state, rows, [0], [1])
    with pytest.raises(ValueError, match="greedy"):
        drafter.draft(state, 2, 1, 0.5)
    with pytest.raises(ValueError, match="frontier"):
        drafter.draft(state, 2, 0, 0)
    with pytest.raises(ValueError, match="another"):
        _drafter().draft(state, 2, 1, 0)
    assert drafter.state_nbytes(state) > 0
    drafter.close_state(state)
    assert state.pending_hidden is None and state.attention is None and state.anchor is None
    with pytest.raises(RuntimeError, match="closed"):
        drafter.draft(state, 2, 1, 0)
