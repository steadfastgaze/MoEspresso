from __future__ import annotations

import math

import pytest

from moespresso.runtime.qwen4.kvarn_layout import (
    QWEN38_KVARN_K4V4_G128,
    QWEN38_KVARN_SCHEMA,
    Qwen4KVarNLayout,
)


def _layout(**overrides) -> Qwen4KVarNLayout:
    values = {
        "kv_heads": 2,
        "head_dim": 256,
        "tile_tokens": 128,
        "key_bits": 4,
        "value_bits": 4,
        "normalization_iterations": 8,
        "schema": QWEN38_KVARN_SCHEMA,
    }
    values.update(overrides)
    return Qwen4KVarNLayout(**values)


def test_released_k4v4_layout_is_tight_and_fully_aligned() -> None:
    layout = QWEN38_KVARN_K4V4_G128
    expected = {
        "k_codes": (0, 16_384),
        "k_scale": (16_384, 512),
        "k_zero": (16_896, 512),
        "k_token_scale": (17_408, 256),
        "v_codes": (17_664, 16_384),
        "v_channel_scale": (34_048, 512),
        "v_token_scale": (34_560, 256),
        "v_zero": (34_816, 256),
    }

    assert {field.name: (field.offset, field.nbytes) for field in layout.fields} == expected
    assert all(field.offset % 256 == 0 for field in layout.fields)
    assert layout.head_record_bytes == 35_072
    assert layout.tile_record_bytes == 70_144
    assert layout.bf16_tile_bytes == 262_144
    assert math.isclose(layout.compression_ratio, 262_144 / 70_144)
    assert layout.effective_bits_per_element == 4.28125
    layout.validate_record_nbytes(70_144)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kv_heads", 1, "two K/V heads"),
        ("head_dim", 128, "dimension 256"),
        ("tile_tokens", 64, "128-token tiles"),
        ("key_bits", 3, "only K4/V4"),
        ("value_bits", 2, "only K4/V4"),
        ("normalization_iterations", 7, "eight iterations"),
        ("schema", "another", "unsupported"),
    ],
)
def test_layout_rejects_unproven_variants(field: str, value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _layout(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "kv_heads",
        "head_dim",
        "tile_tokens",
        "key_bits",
        "value_bits",
        "normalization_iterations",
    ],
)
def test_layout_rejects_boolean_geometry(field: str) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _layout(**{field: True})


def test_layout_rejects_unknown_fields_and_record_lengths() -> None:
    with pytest.raises(KeyError):
        QWEN38_KVARN_K4V4_G128.field("missing")
    with pytest.raises(ValueError, match="expected 70144"):
        QWEN38_KVARN_K4V4_G128.validate_record_nbytes(70_143)
    with pytest.raises(ValueError, match="must be an integer"):
        QWEN38_KVARN_K4V4_G128.validate_record_nbytes(True)
