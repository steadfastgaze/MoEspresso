"""Opt-in real-source smoke for local DeepSeek-V4-Flash artifacts.

Default pytest stays model-free. Enable explicitly:

    MOESPRESSO_RUN_DEEPSEEK_V4_REAL_SMOKE=1 uv run --locked python -m pytest \
        tests/test_deepseek_v4_real_smoke.py -s

Required:
    MOESPRESSO_DEEPSEEK_V4_SOURCE=<HF safetensors checkpoint directory>

Optional:
    MOESPRESSO_DEEPSEEK_V4_IMATRIX=<imatrix calibration file>
    MOESPRESSO_DEEPSEEK_V4_OUT=<artifact directory>
    MOESPRESSO_DEEPSEEK_V4_SAMPLE_ROWS=8
    MOESPRESSO_DEEPSEEK_V4_EXPERT_SAMPLE=1
    MOESPRESSO_DEEPSEEK_V4_TARGET_GB=120.0
    MOESPRESSO_DEEPSEEK_V4_MAX_RSS_GB=4.0

Package smoke:
    MOESPRESSO_RUN_DEEPSEEK_V4_PACKAGE_SMOKE=1 uv run --locked python -m pytest \
        tests/test_deepseek_v4_real_smoke.py -s

Optional package knobs:
    MOESPRESSO_DEEPSEEK_V4_PACKAGE_OUT=<package directory>
    MOESPRESSO_DEEPSEEK_V4_PACKAGE_MAX_RSS_GB=6.0
    MOESPRESSO_DEEPSEEK_V4_MAX_EXPERTS=1
    MOESPRESSO_DEEPSEEK_V4_CHUNK_BYTES=16777216

FP4 nibble oracle:
    MOESPRESSO_RUN_DEEPSEEK_V4_FP4_ORACLE=1 uv run --locked python -m pytest \
        tests/test_deepseek_v4_real_smoke.py -s

The probe smoke does not package or execute the model. The package smoke writes a
reduced real package (`max_experts=1` by default) and verifies it, but still does
not execute the model.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _require_source() -> Path:
    source = _env_path("MOESPRESSO_DEEPSEEK_V4_SOURCE")
    if source is None:
        pytest.skip("set MOESPRESSO_DEEPSEEK_V4_SOURCE=<checkpoint directory>")
    if not source.exists():
        pytest.skip(f"source model not found: {source}")
    return source


def _optional_imatrix() -> Path | None:
    imatrix = _env_path("MOESPRESSO_DEEPSEEK_V4_IMATRIX")
    if imatrix is not None and not imatrix.exists():
        pytest.skip(f"imatrix not found: {imatrix}")
    return imatrix


def _released_converter_uses_low_then_high_fp4(convert_text: str) -> bool:
    return bool(
        re.search(r"\blow\s*=\s*x\s*&\s*0x0F", convert_text)
        and re.search(r"\bhigh\s*=\s*\(x\s*>>\s*4\)\s*&\s*0x0F", convert_text)
        and re.search(
            r"torch\.stack\(\s*\[\s*FP4_TABLE\[low\.long\(\)\]\s*,\s*"
            r"FP4_TABLE\[high\.long\(\)\]",
            convert_text,
            re.DOTALL,
        )
    )


def _first_fp4_expert_pair(source: Path) -> tuple[str, str, str]:
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip("source index not found")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    for name in sorted(weight_map):
        if not (
            name.startswith("layers.")
            and ".ffn.experts." in name
            and name.endswith(".w1.weight")
        ):
            continue
        scale_name = name.removesuffix(".weight") + ".scale"
        shard = weight_map[name]
        if weight_map.get(scale_name) == shard:
            return name, scale_name, shard
    pytest.skip("no routed FP4 expert tensor pair found")


@pytest.mark.real_model
def test_deepseek_v4_real_fp4_nibble_order_matches_released_converter():
    if os.environ.get("MOESPRESSO_RUN_DEEPSEEK_V4_FP4_ORACLE") != "1":
        pytest.skip(
            "set MOESPRESSO_RUN_DEEPSEEK_V4_FP4_ORACLE=1 to run DS4 FP4 oracle"
        )

    source = _require_source()
    convert_py = source / "inference" / "convert.py"
    if not convert_py.exists():
        pytest.skip("released converter not found under source inference directory")
    assert _released_converter_uses_low_then_high_fp4(
        convert_py.read_text(encoding="utf-8")
    )

    from moespresso.inventory.safetensors_header import read_headers_with_offsets
    from moespresso.probe.deepseek_v4.codec import (
        FP4_E2M1_TABLE,
        dequant_fp4_e2m1_ue8m0,
        load_storage_rows,
        ue8m0_to_float32,
    )

    weight_name, scale_name, shard = _first_fp4_expert_pair(source)
    headers = {h.name: h for h in read_headers_with_offsets(source / shard)}
    weight_header = headers[weight_name]
    scale_header = headers[scale_name]
    assert weight_header.dtype == "I8"
    assert scale_header.dtype == "F8_E8M0"

    rows = np.arange(min(4, weight_header.shape[0]), dtype=np.int64)
    packed = load_storage_rows(source, weight_header, rows)
    scales = load_storage_rows(source, scale_header, rows)
    packed_u8 = packed.view(np.uint8)
    assert np.any((packed_u8 & 0x0F) != ((packed_u8 >> 4) & 0x0F))

    decoded = dequant_fp4_e2m1_ue8m0(
        packed,
        scales,
        fp4_block=32,
        out_dtype=np.float32,
    )
    scale_expanded = np.repeat(ue8m0_to_float32(scales), 32, axis=1)
    high_first = np.stack(
        [
            FP4_E2M1_TABLE[(packed_u8 >> 4) & 0x0F],
            FP4_E2M1_TABLE[packed_u8 & 0x0F],
        ],
        axis=-1,
    ).reshape(decoded.shape)
    high_first = (high_first * scale_expanded).astype(np.float32)

    first_scale = ue8m0_to_float32(scales[:1, :1])[0, 0]
    first_byte = int(packed_u8[0, 0])
    expected_first_pair = np.array(
        [
            FP4_E2M1_TABLE[first_byte & 0x0F] * first_scale,
            FP4_E2M1_TABLE[(first_byte >> 4) & 0x0F] * first_scale,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(decoded[0, :2], expected_first_pair)
    assert float(np.max(np.abs(decoded - high_first))) > 0.0
