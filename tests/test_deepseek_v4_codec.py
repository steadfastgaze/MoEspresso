from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from moespresso.inventory.safetensors_header import read_headers_with_offsets
from moespresso.probe.deepseek_v4.codec import (
    dequant_fp4_e2m1_ue8m0,
    dequant_fp8_e4m3_ue8m0,
    fits_float16,
    fp8_e4m3_to_float32,
    iter_dequantized_fp8_row_chunks,
    load_dequantized_fp4,
    load_dequantized_fp8,
    load_dequantized_fp8_rows,
    load_storage_tensor,
    ue8m0_to_float32,
)


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, (dtype, arr) in tensors.items():
        a = np.ascontiguousarray(arr)
        data = a.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(a.shape),
            "data_offsets": [off, off + len(data)],
        }
        blob += data
        off += len(data)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def test_ue8m0_decodes_power_of_two_scales():
    vals = ue8m0_to_float32(np.array([126, 127, 128, 129], dtype=np.uint8))
    np.testing.assert_allclose(vals, np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32))


def test_ue8m0_reserved_code_fails():
    with pytest.raises(ValueError, match="reserved"):
        ue8m0_to_float32(np.array([0xFF], dtype=np.uint8))


def test_fp8_e4m3_decodes_known_values():
    codes = np.array([0x00, 0x30, 0x38, 0x40, 0x7E, 0xB8], dtype=np.uint8)
    vals = fp8_e4m3_to_float32(codes)
    np.testing.assert_allclose(
        vals,
        np.array([0.0, 0.5, 1.0, 2.0, 448.0, -1.0], dtype=np.float32),
    )


def test_fp8_e4m3_nan_codes_fail_closed():
    with pytest.raises(ValueError, match="NaN/reserved"):
        fp8_e4m3_to_float32(np.array([0x7F], dtype=np.uint8))
    with pytest.raises(ValueError, match="NaN/reserved"):
        fp8_e4m3_to_float32(np.array([0xFF], dtype=np.uint8))


def test_dequant_fp4_e2m1_uses_low_then_high_nibbles_and_block_scales():
    packed = np.array([[0x21, 0xA7]], dtype=np.uint8).view(np.int8)
    scales = np.array([[127, 128]], dtype=np.uint8)
    out = dequant_fp4_e2m1_ue8m0(packed, scales, fp4_block=2, out_dtype=np.float32)
    np.testing.assert_allclose(out, np.array([[0.5, 1.0, 12.0, -2.0]], dtype=np.float32))


def test_dequant_fp8_e4m3_uses_2d_block_scales():
    weight = np.array([[0x38, 0x40], [0xB8, 0x30]], dtype=np.uint8)
    scales = np.array([[128]], dtype=np.uint8)
    out = dequant_fp8_e4m3_ue8m0(weight, scales, fp8_block=(2, 2), out_dtype=np.float32)
    np.testing.assert_allclose(out, np.array([[2.0, 4.0], [-2.0, 1.0]], dtype=np.float32))


def test_dequant_fp8_rows_and_chunks_match_full_decode(tmp_path):
    model = tmp_path / "model-00001.safetensors"
    weight = np.array(
        [
            [0x38, 0x40, 0x30, 0x00],
            [0xB8, 0x30, 0x38, 0x40],
            [0x40, 0x38, 0xB8, 0x30],
            [0x30, 0x00, 0x40, 0x38],
        ],
        dtype=np.uint8,
    )
    scales = np.array([[127, 128], [129, 130]], dtype=np.uint8)
    _write_safetensors(model, {
        "dense.weight": ("F8_E4M3", weight),
        "dense.scale": ("F8_E8M0", scales),
    })
    headers = {h.name: h for h in read_headers_with_offsets(model)}
    full = dequant_fp8_e4m3_ue8m0(weight, scales, fp8_block=(2, 2), out_dtype=np.float32)

    rows = load_dequantized_fp8_rows(
        tmp_path,
        headers["dense.weight"],
        headers["dense.scale"],
        np.array([1, 3], dtype=np.int64),
        fp8_block=(2, 2),
        out_dtype=np.float32,
    )
    np.testing.assert_allclose(rows, full[[1, 3]])

    chunks = [
        chunk for _start, chunk in iter_dequantized_fp8_row_chunks(
            tmp_path,
            headers["dense.weight"],
            headers["dense.scale"],
            max_chunk_bytes=16,
            fp8_block=(2, 2),
            out_dtype=np.float32,
        )
    ]
    np.testing.assert_allclose(np.concatenate(chunks, axis=0), full)


def test_dequant_validates_scale_shape():
    with pytest.raises(ValueError, match="scale shape"):
        dequant_fp4_e2m1_ue8m0(
            np.zeros((1, 2), dtype=np.int8),
            np.zeros((1, 1), dtype=np.uint8),
            fp4_block=2,
        )


def test_source_reader_loads_and_dequants_tiny_fp4_and_fp8_safetensors(tmp_path):
    model = tmp_path / "model-00001.safetensors"
    _write_safetensors(model, {
        "expert.weight": ("I8", np.array([[0x21, 0xA7]], dtype=np.uint8).view(np.int8)),
        "expert.scale": ("F8_E8M0", np.array([[127, 128]], dtype=np.uint8)),
        "dense.weight": ("F8_E4M3", np.array([[0x38, 0x40], [0xB8, 0x30]], dtype=np.uint8)),
        "dense.scale": ("F8_E8M0", np.array([[128]], dtype=np.uint8)),
    })
    headers = {h.name: h for h in read_headers_with_offsets(model)}

    raw = load_storage_tensor(tmp_path, headers["expert.weight"])
    assert raw.dtype == np.int8
    fp4 = load_dequantized_fp4(
        tmp_path,
        headers["expert.weight"],
        headers["expert.scale"],
        fp4_block=2,
        out_dtype=np.float32,
    )
    fp8 = load_dequantized_fp8(
        tmp_path,
        headers["dense.weight"],
        headers["dense.scale"],
        fp8_block=(2, 2),
        out_dtype=np.float32,
    )

    np.testing.assert_allclose(fp4, np.array([[0.5, 1.0, 12.0, -2.0]], dtype=np.float32))
    np.testing.assert_allclose(fp8, np.array([[2.0, 4.0], [-2.0, 1.0]], dtype=np.float32))


def test_fits_float16_checks_range_and_finiteness():
    assert fits_float16(np.array([0.0, 65504.0], dtype=np.float32))
    assert not fits_float16(np.array([65520.0], dtype=np.float32))
    assert not fits_float16(np.array([np.inf], dtype=np.float32))
