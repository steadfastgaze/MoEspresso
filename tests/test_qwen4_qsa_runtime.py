from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import moespresso.runtime.qwen4.qsa as qsa_runtime
from moespresso.correctness.qwen4.qsa_reference import (
    expand_visible_group_indices,
    score_compressed_groups,
    sparse_grouped_query_attention,
    stable_topk_groups,
    zero_centered_rms_norm,
)
from moespresso.runtime.qwen4.primitives import Qwen4RMSNorm
from moespresso.runtime.qwen4.qsa import (
    qsa_attention_from_selected_rows,
    qsa_causal_prefix_layout,
    qsa_compress_index_keys,
    qsa_gather_selected_rows,
    qsa_index_scores,
    qsa_selected_attention,
    qsa_selected_token_indices,
    qwen4_partial_rope,
)


def _released_rope_reference(
    values: np.ndarray,
    position_ids: np.ndarray,
    *,
    rotary_dim: int,
    base: float,
) -> np.ndarray:
    positions = np.asarray(position_ids)
    if positions.ndim == 2:
        positions = np.broadcast_to(positions[None], (3, *positions.shape))
    axis_map = np.array([0, 1, 2] * 10 + [0, 1], dtype=np.int64)
    inverse = 1.0 / (
        np.float32(base) ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim))
    )
    frequencies = positions.astype(np.float32)[..., None] * inverse
    phases = np.stack(
        [frequencies[axis, ..., index] for index, axis in enumerate(axis_map)],
        axis=-1,
    )
    angles = np.concatenate([phases, phases], axis=-1)
    cosine = np.cos(angles).astype(values.dtype)[:, :, None]
    sine = np.sin(angles).astype(values.dtype)[:, :, None]
    rope = values[..., :rotary_dim]
    half = rotary_dim // 2
    rotated_half = np.concatenate([-rope[..., half:], rope[..., :half]], axis=-1)
    rotated = rope * cosine + rotated_half * sine
    return np.concatenate([rotated, values[..., rotary_dim:]], axis=-1)


def test_explicit_partial_mrope_matches_released_axis_interleaving() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(1, 2, 2, 80)).astype(np.float32)
    positions = np.array(
        [
            [[7, 8]],
            [[70, 80]],
            [[700, 800]],
        ],
        dtype=np.int32,
    )
    got = qwen4_partial_rope(mx.array(values), mx.array(positions))
    expected = _released_rope_reference(values, positions, rotary_dim=64, base=1.0e7)
    mx.eval(got)

    assert np.allclose(np.asarray(got), expected, rtol=0, atol=1e-5)
    assert np.array_equal(np.asarray(got)[..., 64:], values[..., 64:])


def test_text_positions_broadcast_to_all_three_mrope_axes() -> None:
    rng = np.random.default_rng(19)
    values = rng.normal(size=(1, 3, 1, 72)).astype(np.float32)
    positions = np.array([[2, 5, 11]], dtype=np.int32)

    text = qwen4_partial_rope(mx.array(values), mx.array(positions))
    expanded = qwen4_partial_rope(
        mx.array(values),
        mx.array(np.broadcast_to(positions[None], (3, *positions.shape)).copy()),
    )
    mx.eval(text, expanded)

    assert np.array_equal(np.asarray(text), np.asarray(expanded))


def test_bf16_correctness_path_does_not_collapse_to_fused_rope() -> None:
    rng = np.random.default_rng(31)
    values = mx.array(rng.normal(size=(1, 2, 2, 256)).astype(np.float32)).astype(mx.bfloat16)
    positions = mx.array([[4096, 4097]], dtype=mx.int32)

    explicit = qwen4_partial_rope(values, positions).astype(mx.float32)
    fused = mx.fast.rope(
        values.transpose(0, 2, 1, 3),
        64,
        traditional=False,
        base=1.0e7,
        scale=1.0,
        offset=4096,
    ).transpose(0, 2, 1, 3)
    fused = fused.astype(mx.float32)
    mx.eval(explicit, fused)

    difference = np.abs(np.asarray(explicit) - np.asarray(fused))
    assert difference.max() == 0.015625
    assert np.count_nonzero(difference) > 0


def test_request_local_partial_rope_factors_are_byte_exact_across_qsa_projections() -> None:
    rng = np.random.default_rng(37)
    positions = mx.array(
        np.array([[[4097]], [[83]], [[12]]], dtype=np.int32),
    )
    factors = qsa_runtime._qwen4_partial_rope_factors(
        positions,
        batch_size=1,
        token_count=1,
        rotary_dim=64,
        base=1.0e7,
        mrope_section=(11, 11, 10),
        dtype=mx.bfloat16,
    )
    incumbent_outputs = []
    candidate_outputs = []
    for heads in (2, 8, 16):
        values = mx.array(
            rng.normal(size=(1, 1, heads, 256)).astype(np.float32),
        ).astype(mx.bfloat16)
        incumbent_outputs.append(qwen4_partial_rope(values, positions))
        candidate_outputs.append(
            qsa_runtime._apply_qwen4_partial_rope_factors(
                values,
                positions,
                factors,
                rotary_dim=64,
                base=1.0e7,
                mrope_section=(11, 11, 10),
            )
        )

    incumbent_outputs = tuple(mx.contiguous(value) for value in incumbent_outputs)
    candidate_outputs = tuple(mx.contiguous(value) for value in candidate_outputs)
    mx.eval(*incumbent_outputs, *candidate_outputs)

    for incumbent, candidate in zip(
        incumbent_outputs,
        candidate_outputs,
        strict=True,
    ):
        assert bytes(memoryview(candidate).cast("B")) == bytes(
            memoryview(incumbent).cast("B")
        )


def test_request_local_partial_rope_factors_reject_equal_but_distinct_positions() -> None:
    positions = mx.array([[[7]], [[8]], [[9]]], dtype=mx.int32)
    factors = qsa_runtime._qwen4_partial_rope_factors(
        positions,
        batch_size=1,
        token_count=1,
        rotary_dim=64,
        base=1.0e7,
        mrope_section=(11, 11, 10),
        dtype=mx.bfloat16,
    )
    values = mx.ones((1, 1, 2, 256), dtype=mx.bfloat16)
    equal_positions = mx.array(np.asarray(positions).copy())

    with pytest.raises(ValueError, match="semantic positions"):
        qsa_runtime._apply_qwen4_partial_rope_factors(
            values,
            equal_positions,
            factors,
            rotary_dim=64,
            base=1.0e7,
            mrope_section=(11, 11, 10),
        )


def test_runtime_index_scores_match_numpy_reference() -> None:
    rng = np.random.default_rng(23)
    queries = rng.normal(size=(2, 3, 4, 8)).astype(np.float32)
    keys = rng.normal(size=(2, 5, 8)).astype(np.float32)

    got = qsa_index_scores(mx.array(queries), mx.array(keys))
    expected = np.stack(
        [
            np.stack(
                [score_compressed_groups(query, keys[batch]) for query in queries[batch]],
                axis=0,
            )
            for batch in range(queries.shape[0])
        ],
        axis=0,
    )
    mx.eval(got)

    assert np.allclose(np.asarray(got), expected, rtol=0, atol=2e-6)


def test_compressed_index_keys_pool_before_norm_and_use_group_start_positions() -> None:
    rng = np.random.default_rng(24)
    raw = rng.normal(size=(1, 7, 128)).astype(np.float32)
    weight = rng.normal(scale=0.05, size=(128,)).astype(np.float32)
    positions = np.stack(
        [
            np.arange(10, 17, dtype=np.int32),
            np.arange(110, 117, dtype=np.int32),
            np.arange(1010, 1017, dtype=np.int32),
        ],
        axis=0,
    )[:, None]
    norm = Qwen4RMSNorm(128)
    norm.weight = mx.array(weight)
    layout = qsa_causal_prefix_layout(
        mx.ones((1, 7), dtype=mx.bool_),
        mx.array([[6]], dtype=mx.int32),
    )

    got = qsa_compress_index_keys(mx.array(raw), mx.array(positions), norm, layout)
    pooled = raw[:, :4].astype(np.float32).mean(axis=1, keepdims=True).astype(raw.dtype)
    normalized = zero_centered_rms_norm(pooled, weight)
    expected = _released_rope_reference(
        normalized[:, :, None, :],
        positions[..., :4:4],
        rotary_dim=64,
        base=1.0e7,
    )[:, :, 0]
    mx.eval(got)

    assert got.shape == (1, 1, 128)
    assert np.allclose(np.asarray(got), expected, rtol=0, atol=2e-5)


def test_selection_restores_ascending_context_order_without_dense_mask() -> None:
    scores = np.array(
        [
            [
                [0.1, 0.9, 0.8],
                [0.1, 0.9, 0.8],
                [0.7, 0.1, 0.9],
            ]
        ],
        dtype=np.float32,
    )
    layout = qsa_causal_prefix_layout(
        mx.ones((1, 7), dtype=mx.bool_),
        mx.array([[0, 3, 6]], dtype=mx.int32),
        compress_ratio=2,
    )

    got = qsa_selected_token_indices(
        mx.array(scores),
        layout,
        token_budget=4,
        compress_ratio=2,
    )
    mx.eval(got)

    expected = np.array(
        [
            [
                [0, -1, -1, -1, -1],
                [0, 1, 2, 3, -1],
                [0, 1, 4, 5, 6],
            ]
        ],
        dtype=np.int32,
    )
    assert np.array_equal(np.asarray(got), expected)


def test_released_selection_becomes_sparse_at_2052_visible_tokens() -> None:
    scores = np.arange(513, dtype=np.float32)[None, None]
    scores = np.repeat(scores, 2, axis=1)
    layout = qsa_causal_prefix_layout(
        mx.ones((1, 2052), dtype=mx.bool_),
        mx.array([[2050, 2051]], dtype=mx.int32),
    )

    got = qsa_selected_token_indices(
        mx.array(scores),
        layout,
    )
    mx.eval(got)
    rows = np.asarray(got)[0]

    assert np.array_equal(rows[0], np.arange(2051, dtype=np.int32))
    valid_last = rows[1][rows[1] >= 0]
    assert valid_last.size == 2048
    assert np.array_equal(valid_last, np.arange(4, 2052, dtype=np.int32))
    assert np.array_equal(rows[1, -3:], np.full(3, -1, dtype=np.int32))


def test_compressed_index_keys_follow_visible_rank_and_physical_group_starts() -> None:
    rng = np.random.default_rng(43)
    raw = rng.normal(size=(2, 8, 128)).astype(np.float32)
    weight = rng.normal(scale=0.05, size=(128,)).astype(np.float32)
    positions = np.stack(
        [
            np.broadcast_to(np.arange(8, dtype=np.int32), (2, 8)),
            np.broadcast_to(np.arange(100, 108, dtype=np.int32), (2, 8)),
            np.broadcast_to(np.arange(1000, 1008, dtype=np.int32), (2, 8)),
        ],
        axis=0,
    )
    visible = np.array(
        [
            [2, 3, 4, 5, 6, -1],
            [0, 1, 3, 7, -1, -1],
        ],
        dtype=np.int32,
    )
    norm = Qwen4RMSNorm(128)
    norm.weight = mx.array(weight)
    visible_mask = np.zeros((2, 8), dtype=bool)
    visible_mask[0, visible[0, visible[0] >= 0]] = True
    visible_mask[1, visible[1, visible[1] >= 0]] = True
    layout = qsa_causal_prefix_layout(
        mx.array(visible_mask),
        mx.array([[6], [7]], dtype=mx.int32),
    )

    got = qsa_compress_index_keys(
        mx.array(raw),
        mx.array(positions),
        norm,
        layout,
    )
    expected_rows = []
    for batch in range(2):
        physical = visible[batch, :4]
        pooled = raw[batch, physical].astype(np.float32).mean(axis=0, keepdims=True)
        normalized = zero_centered_rms_norm(pooled, weight)
        expected_rows.append(
            _released_rope_reference(
                normalized[:, None, None, :],
                positions[:, batch : batch + 1, physical[:1]],
                rotary_dim=64,
                base=1.0e7,
            )[0, 0, 0]
        )
    expected = np.stack(expected_rows, axis=0)[:, None]
    mx.eval(got)

    assert got.shape == (2, 2, 128)
    assert np.allclose(np.asarray(got)[:, :1], expected, rtol=0, atol=2e-5)
    assert np.array_equal(np.asarray(got)[:, 1], np.zeros((2, 128), dtype=np.float32))


def test_selection_maps_visible_ranks_back_to_noncontiguous_physical_ids() -> None:
    compact_scores = np.array(
        [
            [[0.2, 0.9, 0.1], [0.8, 0.1, 0.7]],
            [[0.4, 0.3, 0.2], [0.1, 0.8, 0.9]],
        ],
        dtype=np.float32,
    )
    visible = np.array(
        [
            [2, 3, 4, 5, 6, -1],
            [0, 2, 4, 7, 8, 9],
        ],
        dtype=np.int32,
    )
    scores = np.pad(compact_scores, ((0, 0), (0, 0), (0, 2)))
    visible_mask = np.zeros((2, 10), dtype=bool)
    visible_mask[0, visible[0, visible[0] >= 0]] = True
    visible_mask[1, visible[1, visible[1] >= 0]] = True
    query_indices = np.array([[2, 6], [7, 9]], dtype=np.int32)
    counts = np.array([[1, 5], [4, 6]], dtype=np.int32)
    layout = qsa_causal_prefix_layout(
        mx.array(visible_mask),
        mx.array(query_indices),
        compress_ratio=2,
    )

    got = qsa_selected_token_indices(
        mx.array(scores),
        layout,
        token_budget=4,
        compress_ratio=2,
    )
    mx.eval(got)

    expected = np.full((2, 2, 5), -1, dtype=np.int32)
    for batch in range(2):
        for query in range(2):
            visible_prefix = visible[batch, : counts[batch, query]]
            group_count = visible_prefix.size // 2
            groups = stable_topk_groups(
                scores[batch, query],
                visible_groups=group_count,
                block_topk=2,
            )
            expanded = expand_visible_group_indices(
                groups,
                visible_indices=visible_prefix,
                compress_ratio=2,
                token_topk=4,
            )
            valid = np.sort(expanded[expanded >= 0])
            expected[batch, query, : valid.size] = valid

    assert np.array_equal(np.asarray(got), expected)


def test_selection_rejects_score_geometry_from_another_visible_layout() -> None:
    layout = qsa_causal_prefix_layout(
        mx.array([[False, False, True, True, True]]),
        mx.array([[4]], dtype=mx.int32),
        compress_ratio=2,
    )
    with pytest.raises(ValueError, match="score groups must match"):
        qsa_selected_token_indices(
            mx.zeros((1, 1, 3), dtype=mx.float32),
            layout,
            token_budget=4,
            compress_ratio=2,
        )


def test_causal_prefix_layout_packs_padding_holes_without_host_shapes() -> None:
    mask = mx.array(
        [
            [False, False, True, True, True, True, True, True, True],
            [True, False, True, True, False, True, True, False, True],
        ]
    )
    queries = mx.array([[6, 8], [5, 8]], dtype=mx.int32)

    layout = qsa_causal_prefix_layout(mask, queries, compress_ratio=2)
    mx.eval(
        layout.physical_ids,
        layout.valid,
        layout.visible_counts,
        layout.query_valid,
        layout.group_ids,
        layout.group_valid,
    )

    assert np.array_equal(
        np.asarray(layout.physical_ids),
        np.array(
            [[2, 3, 4, 5, 6, 7, 8, -1, -1], [0, 2, 3, 5, 6, 8, -1, -1, -1]],
            dtype=np.int32,
        ),
    )
    assert np.array_equal(np.asarray(layout.visible_counts), np.array([[5, 7], [4, 6]]))
    assert np.array_equal(np.asarray(layout.query_valid), np.ones((2, 2), dtype=bool))
    assert np.array_equal(
        np.asarray(layout.group_ids)[:, :3],
        np.array(
            [
                [[2, 3], [4, 5], [6, 7]],
                [[0, 2], [3, 5], [6, 8]],
            ],
            dtype=np.int32,
        ),
    )
    assert np.array_equal(
        np.asarray(layout.group_valid),
        np.array([[True, True, True, False], [True, True, True, False]]),
    )


def test_causal_prefix_layout_keeps_direct_reduction_for_small_query_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cumsum(*args, **kwargs):
        raise AssertionError("small query batches must not use the prefix-sum path")

    monkeypatch.setattr(qsa_runtime.mx, "cumsum", fail_cumsum)
    layout = qsa_causal_prefix_layout(
        mx.ones((1, 128), dtype=mx.bool_),
        mx.arange(65, 128, dtype=mx.int32)[None],
    )
    mx.eval(layout.visible_counts)

    assert np.array_equal(np.asarray(layout.visible_counts), np.arange(66, 129)[None])


def test_causal_prefix_layout_uses_prefix_sum_for_large_query_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_cumsum = qsa_runtime.mx.cumsum
    calls = []

    def tracked_cumsum(*args, **kwargs):
        calls.append(args[0].shape)
        return real_cumsum(*args, **kwargs)

    monkeypatch.setattr(qsa_runtime.mx, "cumsum", tracked_cumsum)
    positions = np.arange(128, dtype=np.int32)
    mask_values = (positions % 7 != 0)[None]
    queries = positions[-64:][None]
    layout = qsa_causal_prefix_layout(
        mx.array(mask_values),
        mx.array(queries),
    )
    mx.eval(layout.visible_counts)

    expected = np.take_along_axis(np.cumsum(mask_values, axis=-1), queries, axis=-1)
    assert calls == [(1, 128)]
    assert np.array_equal(np.asarray(layout.visible_counts), expected)


def test_causal_prefix_sum_masks_invalid_query_rows() -> None:
    mask_values = (np.arange(128, dtype=np.int32) % 5 != 0)[None]
    queries = np.array([[-1, *range(62), 128]], dtype=np.int32)

    layout = qsa_causal_prefix_layout(
        mx.array(mask_values),
        mx.array(queries),
    )
    mx.eval(layout.visible_counts, layout.query_valid)

    expected = np.zeros_like(queries)
    expected[:, 1:-1] = np.take_along_axis(
        np.cumsum(mask_values, axis=-1),
        queries[:, 1:-1],
        axis=-1,
    )
    assert not bool(np.asarray(layout.query_valid)[0, 0])
    assert not bool(np.asarray(layout.query_valid)[0, -1])
    assert np.array_equal(np.asarray(layout.visible_counts), expected)


@pytest.mark.parametrize(
    ("mlx_dtype", "numpy_dtype"),
    [
        (mx.int32, np.int32),
        (mx.int64, np.int64),
        (mx.uint32, np.uint32),
        (mx.uint64, np.uint64),
    ],
)
def test_causal_prefix_sum_matches_batch_holes_for_every_query_dtype(
    mlx_dtype,
    numpy_dtype,
) -> None:
    positions = np.arange(128, dtype=np.int64)
    mask_values = np.stack([positions % 5 != 0, positions % 7 != 0])
    limit = np.iinfo(numpy_dtype).max
    if np.issubdtype(numpy_dtype, np.signedinteger):
        row = np.array([-1, *range(62), limit], dtype=numpy_dtype)
    else:
        row = np.array([*range(63), limit], dtype=numpy_dtype)
    queries = np.stack([row, row])

    layout = qsa_causal_prefix_layout(
        mx.array(mask_values),
        mx.array(queries, dtype=mlx_dtype),
    )
    mx.eval(layout.visible_counts, layout.query_valid)

    expected_valid = (queries >= 0) & (queries < mask_values.shape[1])
    safe = np.clip(queries, 0, mask_values.shape[1] - 1).astype(np.int64)
    expected_counts = np.take_along_axis(
        np.cumsum(mask_values, axis=-1, dtype=np.int32),
        safe,
        axis=-1,
    )
    expected_counts = np.where(expected_valid, expected_counts, 0)
    assert np.array_equal(np.asarray(layout.query_valid), expected_valid)
    assert np.array_equal(np.asarray(layout.visible_counts), expected_counts)


def test_sparse_selection_and_attention_match_padded_physical_token_sets() -> None:
    mask = mx.array(
        [
            [False, False, True, True, True, True, True, True, True],
            [True, False, True, True, False, True, True, False, True],
        ]
    )
    layout = qsa_causal_prefix_layout(
        mask,
        mx.array([[6, 8], [5, 8]], dtype=mx.int32),
        compress_ratio=2,
    )
    scores = mx.array(
        [
            [[0.2, 0.9, 0.7, 0.0], [0.2, 0.7, 0.9, 0.0]],
            [[0.1, 0.8, 0.9, 0.0], [0.1, 0.8, 0.9, 0.0]],
        ],
        dtype=mx.float32,
    )
    selected = qsa_selected_token_indices(
        scores,
        layout,
        token_budget=2,
        compress_ratio=2,
    )
    keys = mx.zeros((2, 1, 9, 1), dtype=mx.float32)
    physical_values = np.arange(9, dtype=np.float32)[None, None, :, None]
    values = mx.array(np.broadcast_to(physical_values, (2, 1, 9, 1)).copy())
    queries = mx.zeros((2, 2, 1, 1), dtype=mx.float32)
    output = qsa_selected_attention(queries, keys, values, selected, scale=1.0)
    mx.eval(selected, output)

    assert np.array_equal(
        np.asarray(selected),
        np.array(
            [
                [[4, 5, 6], [6, 7, 8]],
                [[3, 5, -1], [6, 8, -1]],
            ],
            dtype=np.int32,
        ),
    )
    assert np.allclose(
        np.asarray(output)[..., 0, 0],
        np.array([[5.0, 7.0], [4.0, 7.0]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_selected_attention_matches_set_mask_reference() -> None:
    rng = np.random.default_rng(29)
    queries = rng.normal(size=(1, 2, 4, 3)).astype(np.float32)
    keys_token_major = rng.normal(size=(6, 2, 3)).astype(np.float32)
    values_token_major = rng.normal(size=(6, 2, 3)).astype(np.float32)
    selected = np.array([[[0, 3, 5, -1], [1, 2, 4, 5]]], dtype=np.int32)
    keys = keys_token_major.transpose(1, 0, 2)[None]
    values = values_token_major.transpose(1, 0, 2)[None]

    got = qsa_selected_attention(
        mx.array(queries),
        mx.array(keys),
        mx.array(values),
        mx.array(selected),
        scale=3**-0.5,
    )
    expected = np.stack(
        [
            sparse_grouped_query_attention(
                queries[0, row],
                keys_token_major,
                values_token_major,
                selected[0, row],
                softmax_scale=3**-0.5,
            )
            for row in range(queries.shape[1])
        ],
        axis=0,
    )[None]
    mx.eval(got)

    assert np.allclose(np.asarray(got), expected, rtol=0, atol=3e-6)


def test_selected_attention_preserves_released_bfloat16_qk_and_pv_lattice() -> None:
    rng = np.random.default_rng(1)
    magnitude = 10 ** rng.uniform(-1.5, 1.5)
    queries = mx.array((rng.normal(size=(1, 1, 1, 32)) * magnitude).astype(np.float32)).astype(
        mx.bfloat16
    )
    keys = mx.array((rng.normal(size=(1, 1, 9, 32)) * magnitude).astype(np.float32)).astype(
        mx.bfloat16
    )
    values = mx.array(rng.normal(size=(1, 1, 9, 32)).astype(np.float32)).astype(mx.bfloat16)
    selected = mx.arange(9, dtype=mx.int32).reshape(1, 1, 9)
    scale = 32**-0.5

    got = qsa_selected_attention(queries, keys, values, selected, scale=scale)
    grouped_queries = queries.reshape(1, 1, 1, 1, 32)
    gathered_keys = keys[:, None]
    gathered_values = values[:, None]
    released_logits = (grouped_queries @ gathered_keys.swapaxes(-1, -2)) * scale
    released_probabilities_fp32 = mx.softmax(
        released_logits.astype(mx.float32),
        axis=-1,
        precise=True,
    )
    released = (released_probabilities_fp32.astype(mx.bfloat16) @ gathered_values).reshape(
        got.shape
    )
    widened_qk_logits = (
        grouped_queries.astype(mx.float32) @ gathered_keys.astype(mx.float32).swapaxes(-1, -2)
    ) * scale
    widened_qk = (
        mx.softmax(widened_qk_logits, axis=-1, precise=True).astype(mx.bfloat16) @ gathered_values
    ).reshape(got.shape)
    widened_pv = (
        (released_probabilities_fp32 @ gathered_values.astype(mx.float32))
        .astype(mx.bfloat16)
        .reshape(got.shape)
    )
    mx.eval(got, released, widened_qk, widened_pv)

    got_fp32 = np.asarray(got.astype(mx.float32))
    assert np.array_equal(got_fp32, np.asarray(released.astype(mx.float32)))
    assert not np.array_equal(got_fp32, np.asarray(widened_qk.astype(mx.float32)))
    assert not np.array_equal(got_fp32, np.asarray(widened_pv.astype(mx.float32)))
    assert got_fp32.sum() == np.float32(2.3981934)


def test_selected_attention_deduplicates_rows_like_the_released_set_mask() -> None:
    queries = mx.zeros((1, 1, 1, 1), dtype=mx.float32)
    keys = mx.zeros((1, 1, 3, 1), dtype=mx.float32)
    values = mx.array([[[[1.0], [10.0], [100.0]]]])

    got = qsa_selected_attention(
        queries,
        keys,
        values,
        mx.array([[[2, 0, 2, -1]]], dtype=mx.int32),
        scale=1.0,
    )
    mx.eval(got)

    assert np.array_equal(np.asarray(got), np.array([[[[50.5]]]], dtype=np.float32))


def test_selected_row_provider_preserves_existing_qsa_arithmetic_exactly() -> None:
    rng = np.random.default_rng(67)
    queries = mx.array(rng.normal(size=(1, 2, 4, 8)).astype(np.float32)).astype(mx.bfloat16)
    keys = mx.array(rng.normal(size=(1, 2, 7, 8)).astype(np.float32)).astype(mx.bfloat16)
    values = mx.array(rng.normal(size=(1, 2, 7, 8)).astype(np.float32)).astype(mx.bfloat16)
    selected = mx.array(
        [[[6, 0, 3, 3, -1], [1, 5, 2, 99, -1]]],
        dtype=mx.int32,
    )

    direct = qsa_selected_attention(queries, keys, values, selected)
    gathered_k, gathered_v, valid = qsa_gather_selected_rows(keys, values, selected)
    provided = qsa_attention_from_selected_rows(
        queries,
        gathered_k,
        gathered_v,
        valid,
    )
    mx.eval(direct, provided)

    assert np.array_equal(
        np.asarray(direct.astype(mx.float32)),
        np.asarray(provided.astype(mx.float32)),
    )
