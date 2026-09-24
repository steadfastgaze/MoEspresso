"""Qwen4-Exp n-gram hashing and selected-row PLE lookup contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from moespresso.runtime.qwen4.primitives import (
    Qwen4RMSNorm,
    qwen4_projection_compute_dtype,
)


class Qwen4PLEEmbeddingProvider(Protocol):
    """Materialize the selected PLE table rows for physical row ids."""

    def lookup(self, row_ids: mx.array) -> mx.array:
        """Return ``[*row_ids.shape, row_width]`` selected embedding rows."""
        ...


@dataclass(frozen=True)
class Qwen4NGramHashResult:
    """Physical PLE row ids and the next short token history."""

    row_ids: mx.array
    next_context: mx.array


@dataclass(frozen=True)
class Qwen4PLEState:
    """Committed token and dilated-convolution history at one frontier."""

    token_context: mx.array
    conv_state: mx.array
    offset: int


@dataclass(frozen=True)
class Qwen4PLEOutput:
    """PLE output and candidate state for an atomic caller-side commit."""

    output: mx.array
    state: Qwen4PLEState


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _metadata_vector(name: str, values: object, *, length: int) -> np.ndarray:
    vector = np.asarray(values)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.issubdtype(vector.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    return vector.astype(np.int64, copy=False)


def _shift_right_ignore_eos(
    token_ids: mx.array,
    *,
    shift: int,
    eos_token_id: int,
) -> mx.array:
    """Shift tokens without carrying an n-gram across an EOS boundary."""
    if shift == 0:
        return token_ids
    batch_size, sequence_length = token_ids.shape
    positions = mx.arange(sequence_length, dtype=mx.int64)
    eos_positions = mx.where(
        token_ids == eos_token_id,
        mx.broadcast_to(positions, (batch_size, sequence_length)),
        mx.array(-1, dtype=mx.int64),
    )
    previous_eos_inclusive = mx.cummax(eos_positions, axis=1)
    previous_eos = mx.concatenate(
        [
            mx.full((batch_size, 1), -1, dtype=mx.int64),
            previous_eos_inclusive[:, :-1],
        ],
        axis=1,
    )
    segment_start = previous_eos + 1
    position_in_segment = positions[None, :] - segment_start
    source_positions = positions - shift
    gather_positions = mx.broadcast_to(
        mx.maximum(source_positions, mx.array(0, dtype=mx.int64))[None, :],
        (batch_size, sequence_length),
    )
    shifted = mx.take_along_axis(token_ids, gather_positions, axis=1)
    valid = (position_in_segment >= shift) & (source_positions[None, :] >= 0)
    return mx.where(valid, shifted, mx.array(eos_token_id, dtype=mx.int64))


class Qwen4NGramHasher:
    """Map token histories to the released bigram and trigram PLE rows."""

    def __init__(
        self,
        *,
        eos_token_id: int,
        ngram_size: int,
        heads_per_ngram: int,
        multipliers: object,
        table_sizes: object,
        table_offsets: object,
    ) -> None:
        self.ngram_size = _positive_integer("ngram_size", ngram_size)
        if self.ngram_size < 2:
            raise ValueError("ngram_size must be at least two")
        self.heads_per_ngram = _positive_integer("heads_per_ngram", heads_per_ngram)
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
            raise TypeError("eos_token_id must be an int")
        self.eos_token_id = eos_token_id
        self.context_length = self.ngram_size - 1
        self.ngram_heads = self.context_length * self.heads_per_ngram

        multiplier_values = _metadata_vector(
            "multipliers",
            multipliers,
            length=self.ngram_size,
        )
        size_values = _metadata_vector(
            "table_sizes",
            table_sizes,
            length=self.ngram_heads,
        )
        offset_values = _metadata_vector(
            "table_offsets",
            table_offsets,
            length=self.ngram_heads,
        )
        if np.any(size_values <= 0):
            raise ValueError("table_sizes must be positive")
        if np.any(offset_values < 0):
            raise ValueError("table_offsets must be nonnegative")
        expected_offsets = np.zeros_like(offset_values)
        if expected_offsets.size > 1:
            expected_offsets[1:] = np.cumsum(size_values[:-1], dtype=np.int64)
        if not np.array_equal(offset_values, expected_offsets):
            raise ValueError("table_offsets must concatenate table_sizes")

        self.multipliers = mx.array(multiplier_values, dtype=mx.int64)
        self.table_sizes = mx.array(size_values, dtype=mx.int64)
        self.table_offsets = mx.array(offset_values, dtype=mx.int64)
    def __call__(
        self,
        input_ids: mx.array,
        *,
        previous_context: mx.array | None = None,
        valid_tokens: mx.array | None = None,
    ) -> Qwen4NGramHashResult:
        if input_ids.ndim != 2 or input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
            raise ValueError("input_ids must have shape [batch, tokens]")
        current = input_ids.astype(mx.int64)
        if valid_tokens is not None:
            if valid_tokens.shape != current.shape:
                raise ValueError("valid_tokens must match input_ids")
            current = mx.where(
                valid_tokens.astype(mx.bool_),
                current,
                mx.array(self.eos_token_id, dtype=mx.int64),
            )

        batch_size = current.shape[0]
        if previous_context is None:
            previous = mx.full(
                (batch_size, self.context_length),
                self.eos_token_id,
                dtype=mx.int64,
            )
        else:
            if previous_context.shape != (batch_size, self.context_length):
                raise ValueError("previous_context has an invalid shape")
            previous = previous_context.astype(mx.int64)

        history = mx.concatenate([previous, current], axis=1)
        shifted = [
            _shift_right_ignore_eos(
                history,
                shift=shift,
                eos_token_id=self.eos_token_id,
            )
            for shift in range(self.ngram_size)
        ]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.multipliers[0]
            for position in range(1, ngram):
                mixed = mx.bitwise_xor(
                    mixed,
                    shifted[position] * self.multipliers[position],
                )
            row_ids = mx.remainder(mixed[..., None], self.table_sizes[start:end])
            blocks.append(row_ids + self.table_offsets[start:end])

        return Qwen4NGramHashResult(
            row_ids=mx.concatenate(blocks, axis=-1)[:, -current.shape[1] :],
            next_context=history[:, -self.context_length :],
        )


def lookup_ple_rows(
    provider: Qwen4PLEEmbeddingProvider,
    row_ids: mx.array,
    *,
    row_width: int,
) -> mx.array:
    """Materialize and concatenate selected per-head PLE rows."""
    width = _positive_integer("row_width", row_width)
    if row_ids.ndim != 3:
        raise ValueError("row_ids must have shape [batch, tokens, heads]")
    rows = provider.lookup(row_ids)
    expected = (*row_ids.shape, width)
    if rows.shape != expected:
        raise ValueError(f"PLE provider returned {rows.shape}; expected {expected}")
    return rows.reshape(*row_ids.shape[:-1], row_ids.shape[-1] * width)


class Qwen4PLELayer(nn.Module):
    """Correctness-first released PLE compute path with explicit state."""

    def __init__(
        self,
        hasher: Qwen4NGramHasher,
        provider: Qwen4PLEEmbeddingProvider,
        *,
        row_width: int,
        hidden_size: int,
        branch_count: int,
        conv_kernel_size: int,
        conv_dilation: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hasher = hasher
        self.provider = provider
        self.row_width = _positive_integer("row_width", row_width)
        self.hidden_size = _positive_integer("hidden_size", hidden_size)
        self.branch_count = _positive_integer("branch_count", branch_count)
        self.conv_kernel_size = _positive_integer("conv_kernel_size", conv_kernel_size)
        self.conv_dilation = _positive_integer("conv_dilation", conv_dilation)
        self.short_conv_state_len = (self.conv_kernel_size - 1) * self.conv_dilation
        self.expanded_size = self.hidden_size * self.branch_count
        self.embedding_size = self.hasher.ngram_heads * self.row_width

        self.key_proj = nn.Linear(self.embedding_size, self.expanded_size, bias=False)
        self.value_proj = nn.Linear(self.embedding_size, self.hidden_size, bias=False)
        self.norm_key = Qwen4RMSNorm(
            self.expanded_size,
            eps=eps,
            group_size=self.hidden_size,
        )
        self.norm_query = Qwen4RMSNorm(
            self.expanded_size,
            eps=eps,
            group_size=self.hidden_size,
        )
        self.norm_conv = Qwen4RMSNorm(
            self.expanded_size,
            eps=eps,
            group_size=self.hidden_size,
        )
        self.conv1d = nn.Conv1d(
            self.expanded_size,
            self.expanded_size,
            self.conv_kernel_size,
            dilation=self.conv_dilation,
            groups=self.expanded_size,
            bias=False,
        )

    def validate_state(
        self,
        state: Qwen4PLEState | None,
        *,
        expected_frontier: int,
        batch_size: int,
    ) -> None:
        """Fail unless PLE state is complete at the public token frontier."""
        if (
            isinstance(expected_frontier, bool)
            or not isinstance(expected_frontier, int)
            or expected_frontier < 0
        ):
            raise ValueError("PLE frontier must be a nonnegative integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("PLE batch size must be a positive integer")
        if expected_frontier == 0:
            if state is not None:
                raise ValueError("PLE state must be empty at frontier zero")
            return
        if not isinstance(state, Qwen4PLEState):
            raise ValueError("PLE state is missing at a live frontier")
        if isinstance(state.offset, bool) or not isinstance(state.offset, int):
            raise ValueError("PLE state offset must be a nonnegative integer")
        if state.offset != expected_frontier:
            raise ValueError("PLE state is off the public frontier")
        if state.token_context.shape != (batch_size, self.hasher.context_length):
            raise ValueError("PLE token context does not match the request batch")
        if state.token_context.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("PLE token context must contain integers")
        expected_conv = (batch_size, self.short_conv_state_len, self.expanded_size)
        if state.conv_state.shape != expected_conv:
            raise ValueError("PLE convolution state does not match the released geometry")
        if state.conv_state.dtype != qwen4_projection_compute_dtype(self.key_proj):
            raise ValueError("PLE convolution state has an incompatible dtype")

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        *,
        state: Qwen4PLEState | None = None,
        valid_tokens: mx.array | None = None,
    ) -> Qwen4PLEOutput:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.expanded_size:
            raise ValueError("hidden_states must match the expanded residual width")
        if input_ids.shape != hidden_states.shape[:2]:
            raise ValueError("input_ids must match hidden_states batch and token axes")
        if valid_tokens is not None:
            if valid_tokens.shape != input_ids.shape or valid_tokens.dtype != mx.bool_:
                raise ValueError("valid_tokens must be a boolean input-id mask")

        batch_size, sequence_length = input_ids.shape
        previous_context = None
        previous_conv = None
        offset = 0
        if state is not None:
            self.validate_state(
                state,
                expected_frontier=state.offset,
                batch_size=batch_size,
            )
            previous_context = state.token_context
            previous_conv = state.conv_state
            offset = state.offset

        hashed = self.hasher(
            input_ids,
            previous_context=previous_context,
            valid_tokens=valid_tokens,
        )
        embeddings = lookup_ple_rows(
            self.provider,
            hashed.row_ids,
            row_width=self.row_width,
        )
        next_context = hashed.next_context
        key = self.norm_key(self.key_proj(embeddings)).reshape(
            batch_size,
            sequence_length,
            self.branch_count,
            self.hidden_size,
        )
        value = self.value_proj(embeddings)
        query = self.norm_query(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.branch_count,
            self.hidden_size,
        )
        gate = mx.sum(key * query, axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated_value = mx.sigmoid(gate) * value[:, :, None, :]
        flattened = gated_value.reshape(batch_size, sequence_length, self.expanded_size)
        normalized = self.norm_conv(flattened)
        if valid_tokens is not None:
            flattened = mx.where(valid_tokens[..., None], flattened, 0)
            normalized = mx.where(valid_tokens[..., None], normalized, 0)

        if previous_conv is None:
            previous_conv = mx.zeros(
                (batch_size, self.short_conv_state_len, self.expanded_size),
                dtype=normalized.dtype,
            )
        history = mx.concatenate([previous_conv, normalized], axis=1)
        convolved = nn.silu(self.conv1d(history))
        next_conv = (
            history[:, -self.short_conv_state_len :]
            if self.short_conv_state_len
            else history[:, :0]
        )
        return Qwen4PLEOutput(
            output=flattened + convolved,
            state=Qwen4PLEState(
                token_context=next_context,
                conv_state=next_conv,
                offset=offset + sequence_length,
            ),
        )
