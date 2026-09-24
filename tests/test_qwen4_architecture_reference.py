from __future__ import annotations

import numpy as np
import pytest

from moespresso.correctness.qwen4.architecture_reference import (
    gated_delta_step,
    gated_residual_read,
    gated_residual_write,
    grouped_zero_centered_rms_norm,
    ngram_embedding_row_ids,
    ngram_layer_multipliers,
    ngram_table_geometry,
    ple_depthwise_convolution,
    ple_finish,
    ple_gated_value,
)


def test_grouped_rms_norm_keeps_residual_branches_independent() -> None:
    values = np.array([[3.0, 4.0, 0.0, 10.0]], dtype=np.float32)

    got = grouped_zero_centered_rms_norm(
        values,
        np.zeros(4, dtype=np.float32),
        group_size=2,
        eps=1e-6,
    )

    expected = np.array(
        [[3.0, 4.0, 0.0, 10.0]]
        / np.sqrt(np.array([[12.5, 12.5, 50.0, 50.0]]) + 1e-6),
        dtype=np.float32,
    )
    assert np.allclose(got, expected, rtol=0, atol=2e-7)


def test_gated_residual_zero_weights_average_half_of_normalized_streams() -> None:
    hyper_input = np.array([[3.0, 4.0, 0.0, 5.0]], dtype=np.float32)

    got = gated_residual_read(
        hyper_input,
        np.zeros(4, dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((4, 1), dtype=np.float32),
        branch_count=2,
        hidden_size=2,
        injection_weight=np.zeros((2, 4), dtype=np.float32),
    )

    streams = got.normalized_input.reshape(1, 2, 2)
    assert np.allclose(got.input_mix_weight, 0.5, rtol=0, atol=0)
    assert np.allclose(got.mixed_input, 0.5 * streams.mean(axis=1), rtol=0, atol=1e-7)
    assert np.array_equal(got.injection_weights, np.ones((1, 2), dtype=np.float32))


def test_gated_residual_write_injects_each_branch_separately() -> None:
    got = gated_residual_write(
        np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        np.array([[2.0, 4.0]], dtype=np.float32),
        np.array([[0.5, 2.0]], dtype=np.float32),
    )

    assert np.array_equal(got, np.array([[11.0, 22.0, 34.0, 48.0]], dtype=np.float32))


def test_gated_residual_rejects_ordinary_hidden_width_input() -> None:
    with pytest.raises(ValueError, match="branch geometry"):
        gated_residual_read(
            np.ones((1, 2), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            np.zeros((4, 1), dtype=np.float32),
            branch_count=2,
            hidden_size=2,
        )


def test_gated_delta_step_decays_predicts_residual_and_commits() -> None:
    state = np.array([[[2.0], [0.0]]], dtype=np.float32)

    got = gated_delta_step(
        query=np.array([[1.0, 0.0]], dtype=np.float32),
        key=np.array([[1.0, 0.0]], dtype=np.float32),
        value=np.array([[3.0]], dtype=np.float32),
        log_decay=np.array([np.log(0.5)], dtype=np.float32),
        beta=np.array([0.25], dtype=np.float32),
        state=state,
        eps=1e-12,
    )

    assert np.allclose(got.state, np.array([[[1.5], [0.0]]]), rtol=0, atol=1e-7)
    assert np.allclose(got.output, np.array([[1.5 / np.sqrt(2)]]), rtol=0, atol=1e-7)


def test_ngram_table_geometry_uses_distinct_primes_and_offsets() -> None:
    got = ngram_table_geometry(
        ngram_size=3,
        heads_per_ngram=2,
        vocab_size_base=11,
        ple_layer_index=0,
    )

    assert np.array_equal(got.sizes, np.array([11, 13, 17, 19], dtype=np.int64))
    assert np.array_equal(got.offsets, np.array([0, 11, 24, 41], dtype=np.int64))
    assert got.total_rows == 60
    assert got.padded_rows == 128


def test_ngram_release_geometry_matches_pinned_transformers_derivation() -> None:
    got = ngram_table_geometry(
        ngram_size=3,
        heads_per_ngram=8,
        vocab_size_base=20_000_000,
        ple_layer_index=0,
    )

    assert np.array_equal(
        ngram_layer_multipliers(248320, 3, ple_layer_index=0, seed=1234),
        np.array([23703573157769, 20109073645365, 8052911324071], dtype=np.int64),
    )
    assert np.array_equal(
        got.sizes,
        np.array(
            [
                20000003,
                20000023,
                20000033,
                20000047,
                20000059,
                20000063,
                20000069,
                20000077,
                20000081,
                20000093,
                20000107,
                20000147,
                20000153,
                20000159,
                20000161,
                20000171,
            ],
            dtype=np.int64,
        ),
    )
    assert got.total_rows == 320_001_446
    assert got.padded_rows == 320_001_536
    assert got.padded_rows // 128 == 2_500_012


def test_ngram_multipliers_are_odd_deterministic_and_layer_specific() -> None:
    first = ngram_layer_multipliers(31, 3, ple_layer_index=0, seed=7)
    repeated = ngram_layer_multipliers(31, 3, ple_layer_index=0, seed=7)
    next_layer = ngram_layer_multipliers(31, 3, ple_layer_index=1, seed=7)

    assert np.array_equal(first, repeated)
    assert np.all(first % 2 == 1)
    assert not np.array_equal(first, next_layer)


def test_ngram_hash_does_not_cross_eos_and_preserves_decode_context() -> None:
    geometry = ngram_table_geometry(
        ngram_size=3,
        heads_per_ngram=1,
        vocab_size_base=11,
        ple_layer_index=0,
    )
    multipliers = np.array([3, 5, 7], dtype=np.int64)

    whole = ngram_embedding_row_ids(
        np.array([[2, 4, 9, 6]], dtype=np.int64),
        eos_token_id=9,
        ngram_size=3,
        heads_per_ngram=1,
        multipliers=multipliers,
        table_geometry=geometry,
    )
    first = ngram_embedding_row_ids(
        np.array([[2, 4, 9]], dtype=np.int64),
        eos_token_id=9,
        ngram_size=3,
        heads_per_ngram=1,
        multipliers=multipliers,
        table_geometry=geometry,
    )
    last = ngram_embedding_row_ids(
        np.array([[6]], dtype=np.int64),
        eos_token_id=9,
        ngram_size=3,
        heads_per_ngram=1,
        multipliers=multipliers,
        table_geometry=geometry,
        previous_context=first.next_context,
    )

    assert np.array_equal(last.row_ids, whole.row_ids[:, -1:])
    bigram_expected = ((6 * 3) ^ (9 * 5)) % 11
    trigram_expected = 11 + ((6 * 3) ^ (9 * 5) ^ (9 * 7)) % 13
    assert np.array_equal(last.row_ids[0, 0], np.array([bigram_expected, trigram_expected]))


def test_ngram_hash_replaces_padding_with_eos_before_hash_and_cache() -> None:
    geometry = ngram_table_geometry(
        ngram_size=3,
        heads_per_ngram=1,
        vocab_size_base=11,
        ple_layer_index=0,
    )
    kwargs = {
        "eos_token_id": 9,
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "multipliers": np.array([3, 5, 7], dtype=np.int64),
        "table_geometry": geometry,
    }

    padded = ngram_embedding_row_ids(
        np.array([[8, 2, 4]], dtype=np.int64),
        valid_tokens=np.array([[False, True, True]]),
        **kwargs,
    )
    explicit = ngram_embedding_row_ids(np.array([[9, 2, 4]], dtype=np.int64), **kwargs)

    assert np.array_equal(padded.row_ids, explicit.row_ids)
    assert np.array_equal(padded.next_context, explicit.next_context)


def test_ple_gate_uses_signed_square_root_before_sigmoid() -> None:
    hidden = np.array([[1.0, 0.0, -1.0, 0.0]], dtype=np.float32)
    key = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    value = np.array([[2.0, 4.0]], dtype=np.float32)

    got = ple_gated_value(
        hidden,
        key,
        value,
        np.zeros(4, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        branch_count=2,
        hidden_size=2,
        eps=1e-6,
    )

    score_magnitude = np.sqrt(2.0)
    positive_gate = 1.0 / (1.0 + np.exp(-np.sqrt(score_magnitude)))
    negative_gate = 1.0 / (1.0 + np.exp(np.sqrt(score_magnitude)))
    expected = np.array(
        [
            [
                [2.0 * positive_gate, 4.0 * positive_gate],
                [2.0 * negative_gate, 4.0 * negative_gate],
            ]
        ],
        dtype=np.float32,
    )
    assert np.allclose(got, expected, rtol=0, atol=2e-6)


def test_ple_depthwise_convolution_is_causal_and_dilated() -> None:
    values = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)

    got = ple_depthwise_convolution(
        values,
        np.array([[1.0, 10.0]], dtype=np.float32),
        dilation=2,
    )

    expected_pre_activation = np.array([[[10.0], [20.0], [31.0]]], dtype=np.float32)
    expected = expected_pre_activation / (1.0 + np.exp(-expected_pre_activation))
    assert np.allclose(got.output, expected, rtol=0, atol=2e-6)
    assert np.array_equal(got.next_state, np.swapaxes(values[:, -2:], 1, 2))


def test_ple_convolution_split_decode_matches_full_sequence() -> None:
    values = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    weight = np.array([[0.5, -0.25, 1.0]], dtype=np.float32)

    whole = ple_depthwise_convolution(values, weight, dilation=3)
    first = ple_depthwise_convolution(values[:, :3], weight, dilation=3)
    second = ple_depthwise_convolution(
        values[:, 3:],
        weight,
        dilation=3,
        previous_state=first.next_state,
    )

    assert np.array_equal(second.output, whole.output[:, 3:])
    assert np.array_equal(second.next_state, whole.next_state)


def test_ple_finish_masks_padding_before_state_commit() -> None:
    gated = np.array([[[[1.0]], [[9.0]]]], dtype=np.float32)

    got = ple_finish(
        gated,
        np.zeros(1, dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        branch_count=1,
        hidden_size=1,
        dilation=3,
        valid_tokens=np.array([[True, False]]),
    )

    expected_first = np.float32(1.0 + 1.0 / (1.0 + np.exp(-1.0)))
    assert np.allclose(got.output[0, 0, 0], expected_first, rtol=0, atol=2e-6)
    assert got.output[0, 1, 0] == 0
    assert got.next_conv_state.shape == (1, 1, 0)
