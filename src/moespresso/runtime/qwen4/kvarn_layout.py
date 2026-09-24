"""Packed KVarN K4/V4 cache layout for released Qwen sparse attention.

The numeric codec is intentionally outside the runtime package. This module is
the shared representation contract used by correctness tools and, after the
quality gates pass, the serving cache and Metal kernels.
"""

from __future__ import annotations

from dataclasses import dataclass


QWEN38_KVARN_SCHEMA = "qwen38-qsa-kvarn-k4v4-g128-d256-h2-i8-fp16meta-sylvester-v1"


@dataclass(frozen=True)
class Qwen4KVarNField:
    """One byte range in a per-head KVarN tile record."""

    name: str
    offset: int
    nbytes: int

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class Qwen4KVarNLayout:
    """Narrow packed-record contract for released Qwen3.8 QSA K/V."""

    kv_heads: int
    head_dim: int
    tile_tokens: int
    key_bits: int
    value_bits: int
    normalization_iterations: int
    schema: str = QWEN38_KVARN_SCHEMA

    def __post_init__(self) -> None:
        integers = {
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "tile_tokens": self.tile_tokens,
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "normalization_iterations": self.normalization_iterations,
        }
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.kv_heads != 2:
            raise ValueError("released Qwen3.8 QSA requires two K/V heads")
        if self.head_dim != 256 or self.head_dim & (self.head_dim - 1):
            raise ValueError("released Qwen3.8 QSA requires power-of-two head dimension 256")
        if self.tile_tokens != 128:
            raise ValueError("the first Qwen3.8 KVarN layout supports 128-token tiles")
        if self.key_bits != 4 or self.value_bits != 4:
            raise ValueError("the first Qwen3.8 KVarN layout supports only K4/V4")
        if self.normalization_iterations != 8:
            raise ValueError("the Qwen3.8 KVarN numeric contract requires eight iterations")
        if self.schema != QWEN38_KVARN_SCHEMA:
            raise ValueError("unsupported Qwen3.8 KVarN schema")

    @property
    def code_bytes(self) -> int:
        return self.tile_tokens * self.head_dim // 2

    @property
    def fields(self) -> tuple[Qwen4KVarNField, ...]:
        offset = 0
        result = []
        for name, nbytes in (
            ("k_codes", self.code_bytes),
            ("k_scale", self.head_dim * 2),
            ("k_zero", self.head_dim * 2),
            ("k_token_scale", self.tile_tokens * 2),
            ("v_codes", self.code_bytes),
            ("v_channel_scale", self.head_dim * 2),
            ("v_token_scale", self.tile_tokens * 2),
            ("v_zero", self.tile_tokens * 2),
        ):
            result.append(Qwen4KVarNField(name=name, offset=offset, nbytes=nbytes))
            offset += nbytes
        return tuple(result)

    def field(self, name: str) -> Qwen4KVarNField:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    @property
    def head_record_bytes(self) -> int:
        return self.fields[-1].end

    @property
    def tile_record_bytes(self) -> int:
        return self.kv_heads * self.head_record_bytes

    @property
    def bf16_tile_bytes(self) -> int:
        return self.tile_tokens * self.kv_heads * self.head_dim * 2 * 2

    @property
    def compression_ratio(self) -> float:
        return self.bf16_tile_bytes / self.tile_record_bytes

    @property
    def effective_bits_per_element(self) -> float:
        elements = self.tile_tokens * self.kv_heads * self.head_dim * 2
        return self.tile_record_bytes * 8 / elements

    def validate_record_nbytes(self, nbytes: int) -> None:
        if isinstance(nbytes, bool) or not isinstance(nbytes, int):
            raise ValueError("record byte count must be an integer")
        if nbytes != self.tile_record_bytes:
            raise ValueError(
                f"KVarN tile record has {nbytes} bytes; expected {self.tile_record_bytes}"
            )


QWEN38_KVARN_K4V4_G128 = Qwen4KVarNLayout(
    kv_heads=2,
    head_dim=256,
    tile_tokens=128,
    key_bits=4,
    value_bits=4,
    normalization_iterations=8,
)
