from __future__ import annotations

import numpy as np
import pytest

from moespresso.correctness.qwen4.qsa_reference import (
    QWEN38_QSA_COMPRESS_RATIO,
    QWEN38_QSA_HEAD_DIM,
    QWEN38_QSA_INDEX_HEAD_DIM,
    QWEN38_QSA_INDEX_HEADS,
    QWEN38_QSA_KV_HEADS,
    QWEN38_QSA_QUERY_HEADS,
    QWEN38_QSA_ROTARY_DIM,
    QWEN38_QSA_TOKEN_BUDGET,
    apply_partial_rope,
    compress_index_keys,
    expand_group_indices,
    expand_visible_group_indices,
    prepare_index_queries,
    project_qsa_output,
    qsa_reference_decode,
    score_compressed_groups,
    selected_index_mask,
    sigmoid_output_gate,
    split_query_gate_projection,
    sparse_grouped_query_attention,
    stable_topk_groups,
    visible_group_rows,
    visible_complete_groups,
    zero_centered_rms_norm,
)


def test_released_qsa_geometry_is_explicit() -> None:
    assert (
        QWEN38_QSA_QUERY_HEADS,
        QWEN38_QSA_KV_HEADS,
        QWEN38_QSA_HEAD_DIM,
        QWEN38_QSA_INDEX_HEADS,
        QWEN38_QSA_INDEX_HEAD_DIM,
        QWEN38_QSA_ROTARY_DIM,
        QWEN38_QSA_COMPRESS_RATIO,
        QWEN38_QSA_TOKEN_BUDGET,
    ) == (24, 2, 256, 4, 128, 64, 4, 2048)


def test_zero_centered_rms_norm_uses_one_plus_weight() -> None:
    values = np.array([[3.0, 4.0]], dtype=np.float32)
    weight = np.array([0.5, -0.25], dtype=np.float32)

    got = zero_centered_rms_norm(values, weight, eps=1e-6)

    rms = np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + 1e-6)
    expected = values / rms * np.array([[1.5, 0.75]], dtype=np.float32)
    assert np.allclose(got, expected, rtol=0, atol=1e-7)


def test_partial_rope_rotates_only_the_leading_dimensions() -> None:
    values = np.array([[1.0, 2.0, 3.0, 4.0, 9.0, 10.0]], dtype=np.float32)
    cos = np.zeros((1, 4), dtype=np.float32)
    sin = np.ones((1, 4), dtype=np.float32)

    got = apply_partial_rope(values, cos, sin)

    assert np.array_equal(got, np.array([[-3.0, -4.0, 1.0, 2.0, 9.0, 10.0]]))


def test_released_indexer_pools_before_norm_and_block_start_rope() -> None:
    raw_keys = np.array(
        [[1.0, 3.0, 10.0, 20.0], [3.0, 5.0, 30.0, 40.0]],
        dtype=np.float32,
    )
    weight = np.zeros(4, dtype=np.float32)
    cos = np.zeros((1, 2), dtype=np.float32)
    sin = np.ones((1, 2), dtype=np.float32)

    got = compress_index_keys(
        raw_keys,
        weight,
        cos,
        sin,
        compress_ratio=2,
    )

    pooled = np.array([[2.0, 4.0, 20.0, 30.0]], dtype=np.float32)
    normalized = pooled / np.sqrt(np.mean(pooled * pooled, axis=-1, keepdims=True) + 1e-6)
    expected = np.concatenate((-normalized[:, 1:2], normalized[:, 0:1], normalized[:, 2:]), axis=-1)
    assert np.allclose(got, expected, rtol=0, atol=1e-7)


def test_released_index_queries_normalize_before_rope() -> None:
    raw = np.array([[[3.0, 4.0, 8.0, 6.0]]], dtype=np.float32)
    weight = np.zeros(4, dtype=np.float32)
    cos = np.zeros((1, 1, 2), dtype=np.float32)
    sin = np.ones((1, 1, 2), dtype=np.float32)

    got = prepare_index_queries(raw, weight, cos, sin)

    normalized = raw / np.sqrt(np.mean(raw * raw, axis=-1, keepdims=True) + 1e-6)
    expected = np.concatenate((-normalized[..., 1:2], normalized[..., 0:1], normalized[..., 2:]), axis=-1)
    assert np.allclose(got, expected, rtol=0, atol=1e-7)


def test_sigmoid_output_gate_accepts_flattened_gate_projection() -> None:
    output = np.array([[2.0, -4.0], [1.0, 3.0]], dtype=np.float32)
    gate = np.array([0.0, 0.0, np.log(3.0), -np.log(3.0)], dtype=np.float32)

    got = sigmoid_output_gate(output, gate)

    assert np.allclose(got, np.array([[1.0, -2.0], [0.75, 0.75]]), rtol=0, atol=1e-7)


def test_query_and_gate_projection_is_interleaved_per_head() -> None:
    projected = np.arange(16, dtype=np.float32)

    query, gate = split_query_gate_projection(projected, query_heads=2, head_dim=4)

    assert np.array_equal(query, np.array([[0, 1, 2, 3], [8, 9, 10, 11]]))
    assert np.array_equal(gate, np.array([[4, 5, 6, 7], [12, 13, 14, 15]]))


def test_qsa_output_gate_precedes_output_projection() -> None:
    heads = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    gate = np.zeros(4, dtype=np.float32)
    output_weight = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

    got = project_qsa_output(heads, gate, output_weight)

    assert np.array_equal(got, np.array([4.0, 6.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("query_position", "context_length", "expected"),
    ((0, 1, 0), (2, 3, 0), (3, 4, 1), (6, 7, 1), (7, 8, 2), (15, 7, 1)),
)
def test_visible_complete_groups_respects_causality_and_context(
    query_position: int,
    context_length: int,
    expected: int,
) -> None:
    assert visible_complete_groups(query_position, context_length, 4) == expected


def test_score_compressed_groups_sums_positive_scores_across_index_heads() -> None:
    query = np.array([[1, 0], [0, 1]], dtype=np.float32)
    keys = np.array([[1, -1], [-1, 1], [1, 1]], dtype=np.float32)

    got = score_compressed_groups(query, keys)

    expected = np.array([1, 1, 2], dtype=np.float32) / np.sqrt(2)
    assert np.allclose(got, expected, rtol=0, atol=1e-7)


def test_stable_topk_groups_uses_lower_group_id_to_break_ties() -> None:
    scores = np.array([0.5, 0.75, 0.75, 4.0], dtype=np.float32)

    got = stable_topk_groups(scores, visible_groups=3, block_topk=4)

    assert np.array_equal(got, np.array([1, 2, 0, -1], dtype=np.int32))


def test_topk_pads_when_release_budget_exceeds_short_context() -> None:
    scores = np.array([3.0, 1.0], dtype=np.float32)

    got = stable_topk_groups(scores, visible_groups=2, block_topk=4)

    assert np.array_equal(got, np.array([0, 1, -1, -1], dtype=np.int32))


def test_expand_group_indices_appends_incomplete_causal_tail() -> None:
    selected = np.array([1, 0], dtype=np.int32)

    got = expand_group_indices(
        selected,
        query_position=10,
        context_length=11,
        compress_ratio=4,
        token_topk=8,
    )

    assert np.array_equal(
        got,
        np.array([4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10], dtype=np.int32),
    )


def test_expand_group_indices_leaves_unused_selection_columns_invalid() -> None:
    selected = np.array([0, -1], dtype=np.int32)

    got = expand_group_indices(
        selected,
        query_position=4,
        context_length=5,
        compress_ratio=4,
        token_topk=8,
    )

    assert np.array_equal(
        got,
        np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1], dtype=np.int32),
    )


def test_contiguous_wrapper_uses_live_context_to_form_tail() -> None:
    got = expand_group_indices(
        np.array([0, -1], dtype=np.int32),
        query_position=15,
        context_length=7,
        compress_ratio=4,
        token_topk=8,
    )

    assert np.array_equal(
        got,
        np.array([0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1], dtype=np.int32),
    )


def test_visible_groups_preserve_actual_left_padded_token_ids() -> None:
    groups, tail = visible_group_rows(
        np.array([2, 3, 4, 5, 6], dtype=np.int32),
        compress_ratio=4,
    )

    assert np.array_equal(groups, np.array([[2, 3, 4, 5]], dtype=np.int64))
    assert np.array_equal(tail, np.array([6], dtype=np.int64))
    got = expand_visible_group_indices(
        np.array([0, -1], dtype=np.int32),
        visible_indices=np.array([2, 3, 4, 5, 6], dtype=np.int32),
        compress_ratio=4,
        token_topk=8,
    )
    assert np.array_equal(
        got,
        np.array([2, 3, 4, 5, 6, -1, -1, -1, -1, -1, -1], dtype=np.int32),
    )


def test_selected_mask_deduplicates_and_intersects_visibility() -> None:
    got = selected_index_mask(
        np.array([4, 4, 2, 8, -1], dtype=np.int32),
        context_length=7,
        visible_indices=np.array([1, 2, 3, 4], dtype=np.int32),
    )

    assert np.array_equal(
        got,
        np.array([False, False, True, False, True, False, False]),
    )


def test_sparse_grouped_query_attention_maps_query_groups_to_kv_heads() -> None:
    query = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]], dtype=np.float32)
    keys = np.array(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[1, 1], [-1, 0]],
        ],
        dtype=np.float32,
    )
    values = np.array(
        [
            [[10, 0], [0, 10]],
            [[0, 20], [20, 0]],
            [[30, 30], [40, 40]],
        ],
        dtype=np.float32,
    )
    logical_indices = np.array([2, 0, -1], dtype=np.int32)

    got = sparse_grouped_query_attention(
        query,
        keys,
        values,
        logical_indices,
        softmax_scale=1.0,
    )

    expected = np.empty_like(got)
    for head, kv_head in enumerate((0, 0, 1, 1)):
        selected_keys = keys[[2, 0], kv_head]
        logits = query[head] @ selected_keys.T
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        expected[head] = probabilities @ values[[2, 0], kv_head]
    assert np.allclose(got, expected, rtol=0, atol=1e-6)


def test_sparse_grouped_query_attention_returns_zero_without_valid_indices() -> None:
    got = sparse_grouped_query_attention(
        np.ones((2, 3), dtype=np.float32),
        np.ones((4, 1, 3), dtype=np.float32),
        np.ones((4, 1, 3), dtype=np.float32),
        np.array([-1, 9], dtype=np.int32),
    )

    assert np.array_equal(got, np.zeros((2, 3), dtype=np.float32))


def test_qsa_reference_decode_keeps_selected_groups_and_tail_observable() -> None:
    keys = np.zeros((5, 1, 2), dtype=np.float32)
    values = np.arange(10, dtype=np.float32).reshape(5, 1, 2)

    got = qsa_reference_decode(
        index_query=np.array([[0, 1]], dtype=np.float32),
        compressed_keys=np.array([[1, 0], [0, 2]], dtype=np.float32),
        attention_query=np.array([[0, 0]], dtype=np.float32),
        keys=keys,
        values=values,
        query_position=4,
        context_length=5,
        token_topk=2,
        compress_ratio=2,
    )

    assert got.visible_groups == 2
    assert np.array_equal(got.selected_groups, np.array([1], dtype=np.int32))
    assert np.array_equal(got.logical_indices, np.array([2, 3, 4], dtype=np.int32))
    assert np.allclose(got.output, np.array([[6, 7]], dtype=np.float32), atol=1e-6)


def test_qsa_reference_decode_handles_zero_complete_groups() -> None:
    got = qsa_reference_decode(
        index_query=np.zeros((1, 2), dtype=np.float32),
        compressed_keys=np.empty((0, 2), dtype=np.float32),
        attention_query=np.zeros((1, 2), dtype=np.float32),
        keys=np.zeros((3, 1, 2), dtype=np.float32),
        values=np.arange(6, dtype=np.float32).reshape(3, 1, 2),
        query_position=2,
        context_length=3,
        token_topk=2048,
        compress_ratio=4,
    )

    assert got.visible_groups == 0
    assert np.array_equal(got.logical_indices[:3], np.array([0, 1, 2], dtype=np.int32))
    assert np.all(got.logical_indices[3:] == -1)
    assert np.allclose(got.output, np.array([[2.0, 3.0]], dtype=np.float32), atol=1e-6)
