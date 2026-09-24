"""Request-owned shifted-row state for the single-token Qwen MTP drafter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx

from moespresso.runtime.deepseek_v4.spec_decode import DraftProposal
from moespresso.runtime.qwen4.mtp import Qwen4MTPState
from moespresso.runtime.qwen4.mtp_load import LoadedQwen4MTP


@dataclass(frozen=True)
class _AnchorState:
    frontier: int
    token: int
    attention: Qwen4MTPState


@dataclass
class Qwen4MTPRequestState:
    owner: object
    frontier: int = 0
    attention: Qwen4MTPState | None = None
    pending_hidden: mx.array | None = None
    anchor: _AnchorState | None = None
    closed: bool = False


def _state_arrays(state: Qwen4MTPState | None) -> tuple[mx.array, ...]:
    if state is None:
        return ()
    attention = state.attention
    return attention.keys, attention.values, attention.raw_index_keys, attention.position_ids


class Qwen4MTPDrafter:
    """Produce one greedy draft using the existing proposal/acceptance contract.

    Committed target rows advance this request state. A draft never advances its
    committed frontier. Its first attention row may be reused when the matching
    public anchor is subsequently ingested; all later context uses target hidden.
    """

    greedy_only = True
    block_size = 1

    def __init__(self, loaded: LoadedQwen4MTP):
        self.loaded = loaded
        self._identity = object()

    def make_state(self) -> Qwen4MTPRequestState:
        return Qwen4MTPRequestState(self._identity)

    def _check(self, state):
        if not isinstance(state, Qwen4MTPRequestState) or state.owner is not self._identity:
            raise ValueError("MTP request state belongs to another drafter")
        if state.closed:
            raise RuntimeError("MTP request state is closed")
        if (0 if state.attention is None else state.attention.frontier) != max(0, state.frontier - 1):
            raise ValueError("MTP attention is off the committed target frontier")
        if (state.pending_hidden is None) != (state.frontier == 0):
            raise ValueError("MTP pending hidden row is off the target frontier")

    def _tokens(self, token_ids: Sequence[int]) -> mx.array:
        if any(type(token) is not int or not 0 <= token < self.loaded.vocab_size for token in token_ids):
            raise ValueError("MTP tokens must be valid integer vocabulary IDs")
        return mx.array([list(token_ids)], dtype=mx.int32)

    def ingest(
        self, state: Qwen4MTPRequestState, rows: mx.array,
        positions: Sequence[int], token_ids: Sequence[int],
    ) -> None:
        """Atomically append target-authoritative rows at contiguous positions."""
        self._check(state)
        count = len(token_ids)
        expected_shape = (1, count, self.loaded.head.hidden_size * self.loaded.head.branch_count)
        if (
            count == 0 or rows.shape != expected_shape
            or rows.dtype not in (mx.float16, mx.bfloat16, mx.float32)
            or any(type(position) is not int for position in positions)
            or list(positions) != list(range(state.frontier, state.frontier + count))
        ):
            raise ValueError("MTP ingest rows must align with contiguous target positions")
        tokens = self._tokens(token_ids)
        attention = state.attention
        if state.frontier == 0:
            inputs, shifted_tokens = rows[:, :-1], tokens[:, 1:]
        elif state.anchor is not None and (
            state.anchor.frontier == state.frontier and state.anchor.token == token_ids[0]
        ):
            attention = state.anchor.attention
            inputs, shifted_tokens = rows[:, :-1], tokens[:, 1:]
        else:
            inputs = mx.concatenate([state.pending_hidden, rows[:, :-1]], axis=1)
            shifted_tokens = tokens
        if shifted_tokens.shape[1]:
            embeddings = self.loaded.embedding(shifted_tokens).astype(rows.dtype)
            attention = self.loaded.head.append_context(inputs, embeddings, state=attention)
        pending = mx.contiguous(rows[:, -1:])
        frontier = state.frontier + count
        if (0 if attention is None else attention.frontier) != frontier - 1:
            raise ValueError("MTP ingest produced the wrong attention frontier")
        mx.eval(pending, *_state_arrays(attention))
        state.attention = attention
        state.pending_hidden = pending
        state.frontier = frontier
        state.anchor = None

    def draft(
        self, state: Qwen4MTPRequestState, anchor_token: int, anchor_pos: int, temperature: float,
    ) -> DraftProposal:
        """Build a proposal without committing target or drafter state."""
        self._check(state)
        if temperature != 0:
            raise ValueError("Qwen MTP currently supports greedy proposals only")
        if state.frontier == 0 or type(anchor_pos) is not int or anchor_pos != state.frontier:
            raise ValueError("MTP anchor must follow the committed target frontier")
        ids = self._tokens([anchor_token])
        embeddings = self.loaded.embedding(ids).astype(state.pending_hidden.dtype)
        output = self.loaded.head(state.pending_hidden, embeddings, state=state.attention)
        logits = output.logits.astype(mx.float32)
        if logits.shape != (1, 1, self.loaded.vocab_size):
            raise ValueError("MTP proposal logits do not cover the target vocabulary")
        proposal = DraftProposal(tokens=mx.argmax(logits, axis=-1), logits=logits)
        state.anchor = _AnchorState(state.frontier, anchor_token, output.state)
        return proposal

    def discard_draft(self, state: Qwen4MTPRequestState) -> None:
        """Drop an uncommitted anchor cache after cancellation or failed evaluation."""
        self._check(state)
        state.anchor = None

    def state_nbytes(self, state: Qwen4MTPRequestState) -> int:
        self._check(state)
        arrays = list(_state_arrays(state.attention))
        if state.pending_hidden is not None:
            arrays.append(state.pending_hidden)
        if state.anchor is not None:
            arrays.extend(_state_arrays(state.anchor.attention))
        return sum(value.nbytes for value in arrays)

    def close_state(self, state: Qwen4MTPRequestState) -> None:
        """Release request-private arrays; the resident head remains shared."""
        if state.owner is not self._identity:
            raise ValueError("MTP request state belongs to another drafter")
        state.attention = None
        state.pending_hidden = None
        state.anchor = None
        state.closed = True
