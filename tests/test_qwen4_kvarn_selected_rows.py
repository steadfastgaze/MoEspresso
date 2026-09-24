from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from moespresso.correctness.qwen4.kvarn_reference import (
    encode_qsa_kv_tile,
    reconstruct_qsa_kv_rows,
)
from moespresso.runtime.qwen4.kvarn_layout import QWEN38_KVARN_K4V4_G128
from moespresso.runtime.qwen4.kvarn_selected_rows import (
    gather_qsa_kvarn_mutable_rows_metal,
    reconstruct_qsa_kvarn_selected_rows_metal,
)
from moespresso.runtime.qwen4.qsa import (
    qsa_attention_from_selected_rows,
    qsa_gather_selected_rows,
    qsa_normalize_selected_rows,
)


pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _record(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (128, 2, 256)
    keys = rng.normal(size=shape).astype(np.float32)
    values = rng.normal(size=shape).astype(np.float32)
    return encode_qsa_kv_tile(keys, values)


def _bf16(values: np.ndarray) -> np.ndarray:
    return np.asarray(mx.array(values).astype(mx.bfloat16).astype(mx.float32))


def test_metal_selected_rows_match_reference_across_tiles_and_invalid_lanes() -> None:
    records = np.stack([_record(401), _record(402)], axis=0)
    selected = np.array([[[255, 0, 192, -1, 64, 257]]], dtype=np.int32)

    keys, values, valid = reconstruct_qsa_kvarn_selected_rows_metal(
        mx.array(records),
        mx.array(selected),
    )
    mx.eval(keys, values, valid)

    expected_ids = np.array([0, 64, 192, 255], dtype=np.int32)
    expected_k = np.zeros((selected.size, 2, 256), dtype=np.float32)
    expected_v = np.zeros_like(expected_k)
    for tile in (0, 1):
        lanes = np.flatnonzero(expected_ids // 128 == tile)
        decoded_k, decoded_v = reconstruct_qsa_kv_rows(records[tile], expected_ids[lanes] % 128)
        expected_k[lanes] = _bf16(decoded_k)
        expected_v[lanes] = _bf16(decoded_v)

    got_k = np.asarray(keys.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    got_v = np.asarray(values.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    assert np.array_equal(
        np.asarray(valid),
        np.array([[[True, True, True, True, False, False]]]),
    )
    assert np.max(np.abs(got_k - expected_k)) <= 0.0078125
    assert np.max(np.abs(got_v - expected_v)) <= 0.0078125
    invalid = ~np.asarray(valid)[0, 0]
    assert np.array_equal(got_k[invalid], np.zeros((2, 2, 256)))
    assert np.array_equal(got_v[invalid], np.zeros((2, 2, 256)))


def test_metal_decoder_reads_only_requested_tile_payload() -> None:
    records = np.stack([_record(403), _record(404)], axis=0)
    broken = records.copy()
    offset = QWEN38_KVARN_K4V4_G128.field("k_scale").offset
    broken[1, :, offset : offset + 2] = np.array([np.nan], dtype="<f2").view(np.uint8)

    keys, values, valid = reconstruct_qsa_kvarn_selected_rows_metal(
        mx.array(broken),
        mx.array([[[3, 127]]], dtype=mx.int32),
    )
    mx.eval(keys, values, valid)

    assert bool(mx.all(valid).item())
    assert bool(mx.all(mx.isfinite(keys)).item())
    assert bool(mx.all(mx.isfinite(values)).item())


def test_fused_mutable_gather_matches_each_storage_class() -> None:
    rng = np.random.default_rng(408)
    records = np.stack([_record(409), _record(410)], axis=0)
    sink_keys = mx.array(rng.normal(size=(1, 2, 128, 256)).astype(np.float32)).astype(mx.bfloat16)
    sink_values = mx.array(rng.normal(size=(1, 2, 128, 256)).astype(np.float32)).astype(mx.bfloat16)
    tail_keys = mx.array(rng.normal(size=(1, 2, 1_152, 256)).astype(np.float32)).astype(mx.bfloat16)
    tail_values = mx.array(rng.normal(size=(1, 2, 1_152, 256)).astype(np.float32)).astype(
        mx.bfloat16
    )
    pending_keys = mx.array(rng.normal(size=(1, 2, 2, 256)).astype(np.float32)).astype(mx.bfloat16)
    pending_values = mx.array(rng.normal(size=(1, 2, 2, 256)).astype(np.float32)).astype(
        mx.bfloat16
    )
    selected = mx.array(
        [[[0, 127, 128, 255, 256, 383, 384, 499, 500, 501, 501, -1, 999]]],
        dtype=mx.int32,
    )
    normalized, valid = qsa_normalize_selected_rows(selected, 502)
    keys, values = gather_qsa_kvarn_mutable_rows_metal(
        mx.array(records),
        sink_keys,
        sink_values,
        tail_keys,
        tail_values,
        pending_keys,
        pending_values,
        normalized,
        valid,
        frontier=500,
    )
    mx.eval(keys, values, normalized, valid)

    expected_keys = np.zeros((13, 2, 256), dtype=np.float32)
    expected_values = np.zeros_like(expected_keys)
    normalized_host = np.asarray(normalized)[0, 0]
    valid_host = np.asarray(valid)[0, 0]
    decoded_keys, decoded_values, _ = reconstruct_qsa_kvarn_selected_rows_metal(
        mx.array(records),
        mx.array([[[0, 127, 128, 255]]], dtype=mx.int32),
    )
    mx.eval(decoded_keys, decoded_values)
    decoded_keys_host = np.asarray(decoded_keys.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    decoded_values_host = np.asarray(decoded_values.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    sink_keys_host = np.asarray(sink_keys.astype(mx.float32))[0].transpose(1, 0, 2)
    sink_values_host = np.asarray(sink_values.astype(mx.float32))[0].transpose(1, 0, 2)
    tail_keys_host = np.asarray(tail_keys.astype(mx.float32))[0].transpose(1, 0, 2)
    tail_values_host = np.asarray(tail_values.astype(mx.float32))[0].transpose(1, 0, 2)
    pending_keys_host = np.asarray(pending_keys.astype(mx.float32))[0].transpose(1, 0, 2)
    pending_values_host = np.asarray(pending_values.astype(mx.float32))[0].transpose(1, 0, 2)
    for lane, logical in enumerate(normalized_host):
        if not valid_host[lane]:
            continue
        if logical < 128:
            expected_keys[lane] = sink_keys_host[logical]
            expected_values[lane] = sink_values_host[logical]
        elif logical < 384:
            body_lane = (0, 1, 2, 3)[(0, 127, 128, 255).index(logical - 128)]
            expected_keys[lane] = decoded_keys_host[body_lane]
            expected_values[lane] = decoded_values_host[body_lane]
        elif logical < 500:
            slot = (logical - 128) % 1_152
            expected_keys[lane] = tail_keys_host[slot]
            expected_values[lane] = tail_values_host[slot]
        else:
            expected_keys[lane] = pending_keys_host[logical - 500]
            expected_values[lane] = pending_values_host[logical - 500]

    got_keys = np.asarray(keys.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    got_values = np.asarray(values.astype(mx.float32))[0, 0].transpose(1, 0, 2)
    assert np.array_equal(got_keys, expected_keys)
    assert np.array_equal(got_values, expected_values)


@pytest.mark.parametrize("record_count", (0, 3))
def test_mutable_gather_matches_logical_rows_with_multiple_queries_and_wrapped_tail(
    record_count: int,
) -> None:
    rng = np.random.default_rng(411 + record_count)
    records = mx.array(np.stack([_record(420 + i) for i in range(max(1, record_count))]))
    exact = [
        mx.array(rng.normal(size=(1, 2, width, 256)).astype(np.float32)).astype(mx.bfloat16)
        for width in (128, 128, 256, 256, 2, 2)
    ]
    sink_k, sink_v, tail_k, tail_v, pending_k, pending_v = exact
    body_end = 128 + record_count * 128
    frontier = body_end + 173
    ids = np.concatenate((np.arange(frontier + 2), [-1, frontier + 5, 128, 128])).astype(np.int32)
    selected = mx.array(np.stack((ids, np.roll(ids, 7)))[None])
    normalized, valid = qsa_normalize_selected_rows(selected, frontier + 2)
    actual = gather_qsa_kvarn_mutable_rows_metal(
        records, *exact, normalized, valid, frontier=frontier, record_count=record_count,
    )
    if record_count:
        body_k, body_v, _ = reconstruct_qsa_kvarn_selected_rows_metal(
            records,
            mx.arange(record_count * 128, dtype=mx.int32).reshape(1, 1, -1),
        )
        body_k, body_v = body_k[:, 0], body_v[:, 0]
    else:
        body_k = body_v = mx.zeros((1, 2, 0, 256), dtype=mx.bfloat16)
    tail_ids = (mx.arange(body_end, frontier) - 128) % 256
    logical_k = mx.concatenate((sink_k, body_k, tail_k[:, :, tail_ids], pending_k), axis=2)
    logical_v = mx.concatenate((sink_v, body_v, tail_v[:, :, tail_ids], pending_v), axis=2)
    expected_k, expected_v, expected_valid = qsa_gather_selected_rows(
        logical_k, logical_v, selected,
    )
    mask = expected_valid[:, :, None, :, None]
    expected = (mx.where(mask, expected_k, 0), mx.where(mask, expected_v, 0))
    mx.eval(*actual, *expected, valid, expected_valid)
    assert np.array_equal(np.asarray(valid), np.asarray(expected_valid))
    for result, reference in zip(actual, expected, strict=True):
        assert np.array_equal(
            np.asarray(result.view(mx.uint16)), np.asarray(reference.view(mx.uint16)),
        )


def test_metal_decoder_accepts_empty_requests_without_compiling_a_grid() -> None:
    keys, values, valid = reconstruct_qsa_kvarn_selected_rows_metal(
        mx.zeros((0, 2, 35_072), dtype=mx.uint8),
        mx.zeros((1, 1, 0), dtype=mx.int32),
    )

    assert keys.shape == values.shape == (1, 1, 2, 0, 256)
    assert keys.dtype == values.dtype == mx.bfloat16
    assert valid.shape == (1, 1, 0)


@pytest.mark.parametrize(
    ("records", "selected", "message"),
    [
        (
            mx.zeros((1, 2, 35_071), dtype=mx.uint8),
            mx.array([[[0]]], dtype=mx.int32),
            "records must have shape",
        ),
        (
            mx.zeros((1, 2, 35_072), dtype=mx.int16),
            mx.array([[[0]]], dtype=mx.int32),
            "uint8",
        ),
        (
            mx.zeros((1, 2, 35_072), dtype=mx.uint8),
            mx.array([0], dtype=mx.int32),
            "shape",
        ),
        (
            mx.zeros((1, 2, 35_072), dtype=mx.uint8),
            mx.array([[[0]]], dtype=mx.uint32),
            "int32",
        ),
    ],
)
def test_metal_decoder_rejects_incompatible_inputs(
    records: mx.array,
    selected: mx.array,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reconstruct_qsa_kvarn_selected_rows_metal(records, selected)


def test_packed_provider_canonicalizes_duplicate_and_invalid_rows() -> None:
    records = np.stack([_record(405), _record(406)], axis=0)
    selected = mx.array(
        [
            [[255, 0, 192, 192, -1, 64], [129, 1, 128, 999, 1, -1]],
            [[127, 64, 0, -1, 64, 5], [200, 199, 130, 128, 130, -5]],
        ],
        dtype=mx.int32,
    )
    keys, values, valid = reconstruct_qsa_kvarn_selected_rows_metal(
        mx.array(records),
        selected,
    )
    rng = np.random.default_rng(407)
    queries = mx.array(rng.normal(size=(2, 2, 4, 256)).astype(np.float32)).astype(mx.bfloat16)
    output = qsa_attention_from_selected_rows(queries, keys, values, valid)
    canonical = mx.array(
        [
            [[0, 64, 192, 255, -1, -1], [1, 128, 129, -1, -1, -1]],
            [[0, 5, 64, 127, -1, -1], [128, 130, 199, 200, -1, -1]],
        ],
        dtype=mx.int32,
    )
    canonical_keys, canonical_values, canonical_valid = reconstruct_qsa_kvarn_selected_rows_metal(
        mx.array(records), canonical
    )
    canonical_output = qsa_attention_from_selected_rows(
        queries,
        canonical_keys,
        canonical_values,
        canonical_valid,
    )
    mx.eval(output, canonical_output, valid, canonical_valid)

    assert np.array_equal(
        np.asarray(mx.sum(valid, axis=-1)),
        np.array([[4, 3], [4, 4]], dtype=np.uint32),
    )
    assert np.array_equal(
        np.asarray(output.astype(mx.float32)),
        np.asarray(canonical_output.astype(mx.float32)),
    )
