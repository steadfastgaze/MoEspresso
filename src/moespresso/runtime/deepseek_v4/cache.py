"""Synthetic DeepSeek-V4 composite cache adapter.

The real DS4 runtime cache will hold MLX tensors. This module pins the structural
contract in a pure representation: raw SWA rows, compressed attention rows,
indexer rows for CSA layers, partial compressor state, trim, fork, and payload
serialization. It is deliberately model-free and small enough for unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from moespresso.inventory.architecture_profile import DEEPSEEK_V4_FLASH_COMPRESS_RATIOS

DEEPSEEK_V4_CACHE_KIND = "deepseek_v4_composite"
DEEPSEEK_V4_LAYER_COUNT = 43
DEEPSEEK_V4_SLIDING_WINDOW = 128


@dataclass(frozen=True)
class DeepseekV4PoolRow:
    row_index: int
    raw_start: int
    raw_end: int

    def to_payload(self) -> dict:
        return {
            "row_index": self.row_index,
            "raw_start": self.raw_start,
            "raw_end": self.raw_end,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "DeepseekV4PoolRow":
        return cls(
            row_index=int(payload["row_index"]),
            raw_start=int(payload["raw_start"]),
            raw_end=int(payload["raw_end"]),
        )


@dataclass(frozen=True)
class DeepseekV4LayerCacheState:
    layer_index: int
    compress_ratio: int
    sliding_window: int
    raw_positions: tuple[int, ...]
    compressed_rows: tuple[DeepseekV4PoolRow, ...]
    indexer_rows: tuple[DeepseekV4PoolRow, ...]
    partial_positions: tuple[int, ...]
    indexer_partial_positions: tuple[int, ...]

    @property
    def has_compressor(self) -> bool:
        return self.compress_ratio > 0

    @property
    def has_indexer(self) -> bool:
        return self.compress_ratio == 4

    def to_payload(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "compress_ratio": self.compress_ratio,
            "sliding_window": self.sliding_window,
            "raw_positions": list(self.raw_positions),
            "compressed_rows": [r.to_payload() for r in self.compressed_rows],
            "indexer_rows": [r.to_payload() for r in self.indexer_rows],
            "partial_positions": list(self.partial_positions),
            "indexer_partial_positions": list(self.indexer_partial_positions),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "DeepseekV4LayerCacheState":
        return cls(
            layer_index=int(payload["layer_index"]),
            compress_ratio=int(payload["compress_ratio"]),
            sliding_window=int(payload["sliding_window"]),
            raw_positions=tuple(int(v) for v in payload["raw_positions"]),
            compressed_rows=tuple(
                DeepseekV4PoolRow.from_payload(r) for r in payload["compressed_rows"]
            ),
            indexer_rows=tuple(
                DeepseekV4PoolRow.from_payload(r) for r in payload["indexer_rows"]
            ),
            partial_positions=tuple(int(v) for v in payload["partial_positions"]),
            indexer_partial_positions=tuple(
                int(v) for v in payload["indexer_partial_positions"]
            ),
        )


@dataclass(frozen=True)
class DeepseekV4DecodeCompressionStep:
    start_pos: int
    write_slot: int
    ape_slot: int
    should_compress: bool
    compression_positions: tuple[int | None, ...]


class DeepseekV4DecodeCompressionState:
    """Pure slot-state model for DS4 decode-time compressed attention."""

    def __init__(self, compress_ratio: int) -> None:
        ratio = int(compress_ratio)
        if ratio <= 0:
            raise ValueError("compress_ratio must be positive")
        self.compress_ratio = ratio
        self.has_overlap = ratio == 4
        self._slot_count = ratio * 2 if self.has_overlap else ratio
        self._positions: list[int | None] = [None] * self._slot_count
        self._scores: list[float] = [-math.inf] * self._slot_count
        self._next_pos = 0

    @property
    def position_slots(self) -> tuple[int | None, ...]:
        return tuple(self._positions)

    @property
    def score_state(self) -> tuple[float, ...]:
        return tuple(self._scores)

    def append_token(self, start_pos: int) -> DeepseekV4DecodeCompressionStep:
        pos = int(start_pos)
        if pos != self._next_pos:
            raise ValueError("start_pos must advance one token at a time")

        slot = pos % self.compress_ratio
        write_slot = self.compress_ratio + slot if self.has_overlap else slot
        self._positions[write_slot] = pos
        self._scores[write_slot] = 0.0

        should_compress = (pos + 1) % self.compress_ratio == 0
        compression_positions: tuple[int | None, ...] = ()
        if should_compress:
            compression_positions = tuple(self._positions)
            if self.has_overlap:
                self._positions[:self.compress_ratio] = self._positions[
                    self.compress_ratio:
                ]
                self._positions[self.compress_ratio:] = [None] * self.compress_ratio
                self._scores[:self.compress_ratio] = self._scores[self.compress_ratio:]
                self._scores[self.compress_ratio:] = [-math.inf] * self.compress_ratio
            else:
                self._positions = [None] * self._slot_count
                self._scores = [-math.inf] * self._slot_count

        self._next_pos += 1
        return DeepseekV4DecodeCompressionStep(
            start_pos=pos,
            write_slot=write_slot,
            ape_slot=slot,
            should_compress=should_compress,
            compression_positions=compression_positions,
        )


def _layer_state_from_length(
    layer_index: int,
    compress_ratio: int,
    token_count: int,
    sliding_window: int,
) -> DeepseekV4LayerCacheState:
    raw_start = max(0, token_count - sliding_window)
    raw_positions = tuple(range(raw_start, token_count))
    compressed_rows: tuple[DeepseekV4PoolRow, ...] = ()
    indexer_rows: tuple[DeepseekV4PoolRow, ...] = ()
    partial_positions: tuple[int, ...] = ()
    indexer_partial_positions: tuple[int, ...] = ()

    if compress_ratio > 0:
        complete_rows = token_count // compress_ratio
        compressed_rows = tuple(
            DeepseekV4PoolRow(
                row_index=row,
                raw_start=row * compress_ratio,
                raw_end=(row + 1) * compress_ratio,
            )
            for row in range(complete_rows)
        )
        partial_start = complete_rows * compress_ratio
        partial_positions = tuple(range(partial_start, token_count))
        if compress_ratio == 4:
            indexer_rows = compressed_rows
            indexer_partial_positions = partial_positions

    return DeepseekV4LayerCacheState(
        layer_index=layer_index,
        compress_ratio=compress_ratio,
        sliding_window=sliding_window,
        raw_positions=raw_positions,
        compressed_rows=compressed_rows,
        indexer_rows=indexer_rows,
        partial_positions=partial_positions,
        indexer_partial_positions=indexer_partial_positions,
    )


class DeepseekV4Cache:
    """Pure structural adapter for DS4 prefix-cache payloads."""

    def __init__(
        self,
        *,
        compress_ratios: Sequence[int] = DEEPSEEK_V4_FLASH_COMPRESS_RATIOS,
        sliding_window: int = DEEPSEEK_V4_SLIDING_WINDOW,
        layer_count: int = DEEPSEEK_V4_LAYER_COUNT,
        token_count: int = 0,
    ) -> None:
        if layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if len(compress_ratios) < layer_count:
            raise ValueError("compress_ratios must include every served layer")
        if sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        self.compress_ratios = tuple(int(v) for v in compress_ratios[:layer_count])
        self.sliding_window = int(sliding_window)
        self.layer_count = int(layer_count)
        self.token_count = int(token_count)
        self.layers: tuple[DeepseekV4LayerCacheState, ...] = ()
        self._rebuild()

    def _rebuild(self) -> None:
        self.layers = tuple(
            _layer_state_from_length(
                layer_index=i,
                compress_ratio=ratio,
                token_count=self.token_count,
                sliding_window=self.sliding_window,
            )
            for i, ratio in enumerate(self.compress_ratios)
        )

    def layer(self, layer_index: int) -> DeepseekV4LayerCacheState:
        if layer_index < 0 or layer_index >= self.layer_count:
            raise ValueError(f"layer_index {layer_index} outside cache")
        return self.layers[layer_index]

    def append_tokens(self, count: int) -> "DeepseekV4Cache":
        if count < 0:
            raise ValueError("count must be non-negative")
        self.token_count += int(count)
        self._rebuild()
        return self

    def trim_to_length(self, prefix_len: int) -> "DeepseekV4Cache":
        if prefix_len < 0 or prefix_len > self.token_count:
            raise ValueError("prefix_len must be within the current cache length")
        self.token_count = int(prefix_len)
        self._rebuild()
        return self

    def fork(self) -> "DeepseekV4Cache":
        return DeepseekV4Cache(
            compress_ratios=self.compress_ratios,
            sliding_window=self.sliding_window,
            layer_count=self.layer_count,
            token_count=self.token_count,
        )

    def to_payload(self) -> dict:
        return {
            "kind": DEEPSEEK_V4_CACHE_KIND,
            "token_count": self.token_count,
            "sliding_window": self.sliding_window,
            "compress_ratios": list(self.compress_ratios),
            "layers": [layer.to_payload() for layer in self.layers],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "DeepseekV4Cache":
        if payload.get("kind") != DEEPSEEK_V4_CACHE_KIND:
            raise ValueError("payload is not a DeepSeek V4 composite cache")
        cache = cls(
            compress_ratios=payload["compress_ratios"],
            sliding_window=int(payload["sliding_window"]),
            layer_count=len(payload["layers"]),
            token_count=int(payload["token_count"]),
        )
        # Preserve the serialized layer payload exactly so tests catch accidental
        # payload shape drift, even though rebuild would derive the same structure.
        cache.layers = tuple(
            DeepseekV4LayerCacheState.from_payload(layer) for layer in payload["layers"]
        )
        return cache


def make_deepseek_v4_cache() -> DeepseekV4Cache:
    """Factory mirroring the future runtime prompt-cache constructor."""
    return DeepseekV4Cache()
