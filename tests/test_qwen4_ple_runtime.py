from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from moespresso.correctness.qwen4.architecture_reference import (
    ngram_embedding_row_ids,
    ngram_table_geometry,
    ple_finish,
    ple_gated_value,
)
from moespresso.runtime.qwen4.ple import (
    Qwen4NGramHasher,
    Qwen4PLELayer,
    Qwen4PLEState,
    lookup_ple_rows,
)


def _array(values: np.ndarray) -> mx.array:
    return mx.array(values)


def _tiny_hasher() -> Qwen4NGramHasher:
    geometry = ngram_table_geometry(
        ngram_size=3,
        heads_per_ngram=2,
        vocab_size_base=11,
        ple_layer_index=0,
    )
    return Qwen4NGramHasher(
        eos_token_id=9,
        ngram_size=3,
        heads_per_ngram=2,
        multipliers=np.array([3, 5, 7], dtype=np.int64),
        table_sizes=geometry.sizes,
        table_offsets=geometry.offsets,
    )


def test_runtime_ngram_hash_matches_numpy_reference_with_eos_and_padding() -> None:
    hasher = _tiny_hasher()
    tokens = np.array([[8, 2, 4, 9, 6], [1, 9, 3, 5, 7]], dtype=np.int64)
    valid = np.array(
        [[False, True, True, True, True], [True, True, True, True, False]],
        dtype=bool,
    )
    geometry = ngram_table_geometry(
        ngram_size=3,
        heads_per_ngram=2,
        vocab_size_base=11,
        ple_layer_index=0,
    )

    got = hasher(_array(tokens), valid_tokens=_array(valid))
    expected = ngram_embedding_row_ids(
        tokens,
        eos_token_id=9,
        ngram_size=3,
        heads_per_ngram=2,
        multipliers=np.array([3, 5, 7], dtype=np.int64),
        table_geometry=geometry,
        valid_tokens=valid,
    )
    mx.eval(got.row_ids, got.next_context)

    assert np.array_equal(np.asarray(got.row_ids), expected.row_ids)
    assert np.array_equal(np.asarray(got.next_context), expected.next_context)


def test_runtime_ngram_hash_matches_release_scale_int64_lattice() -> None:
    hasher = Qwen4NGramHasher(
        eos_token_id=248044,
        ngram_size=3,
        heads_per_ngram=1,
        multipliers=np.array(
            [23703573157769, 20109073645365, 8052911324071],
            dtype=np.int64,
        ),
        table_sizes=np.array([20000003, 20000023], dtype=np.int64),
        table_offsets=np.array([0, 20000003], dtype=np.int64),
    )
    tokens = np.array([[248319, 248318, 248317]], dtype=np.int64)
    geometry = ngram_table_geometry(
        ngram_size=3,
        heads_per_ngram=1,
        vocab_size_base=20_000_000,
        ple_layer_index=0,
    )

    got = hasher(_array(tokens))
    with np.errstate(over="ignore"):
        expected = ngram_embedding_row_ids(
            tokens,
            eos_token_id=248044,
            ngram_size=3,
            heads_per_ngram=1,
            multipliers=np.array(
                [23703573157769, 20109073645365, 8052911324071],
                dtype=np.int64,
            ),
            table_geometry=geometry,
        )
    mx.eval(got.row_ids, got.next_context)

    assert np.array_equal(np.asarray(got.row_ids), expected.row_ids)
    assert np.array_equal(np.asarray(got.next_context), expected.next_context)


def test_runtime_ngram_hash_full_chunks_and_tokens_are_exact() -> None:
    hasher = _tiny_hasher()
    tokens = _array(np.array([[2, 4, 9, 6, 3, 1, 9, 5]], dtype=np.int64))

    full = hasher(tokens)

    context = None
    chunks = []
    for start, end in ((0, 3), (3, 5), (5, 8)):
        result = hasher(tokens[:, start:end], previous_context=context)
        chunks.append(result.row_ids)
        context = result.next_context

    token_context = None
    token_rows = []
    for position in range(tokens.shape[1]):
        result = hasher(
            tokens[:, position : position + 1],
            previous_context=token_context,
        )
        token_rows.append(result.row_ids)
        token_context = result.next_context

    chunked = mx.concatenate(chunks, axis=1)
    tokenwise = mx.concatenate(token_rows, axis=1)
    mx.eval(full.row_ids, full.next_context, chunked, context, tokenwise, token_context)

    assert np.array_equal(np.asarray(chunked), np.asarray(full.row_ids))
    assert np.array_equal(np.asarray(tokenwise), np.asarray(full.row_ids))
    assert np.array_equal(np.asarray(context), np.asarray(full.next_context))
    assert np.array_equal(np.asarray(token_context), np.asarray(full.next_context))


class _SelectedRowProvider:
    def lookup(self, row_ids: mx.array) -> mx.array:
        return mx.stack(
            [row_ids.astype(mx.float32), row_ids.astype(mx.float32) + 0.5],
            axis=-1,
        )


def test_ple_provider_boundary_materializes_only_selected_rows() -> None:
    row_ids = _array(np.array([[[2, 5], [7, 11]]], dtype=np.int64))

    got = lookup_ple_rows(_SelectedRowProvider(), row_ids, row_width=2)
    mx.eval(got)

    expected = np.array([[[2.0, 2.5, 5.0, 5.5], [7.0, 7.5, 11.0, 11.5]]], dtype=np.float32)
    assert np.array_equal(np.asarray(got), expected)


def test_ple_provider_boundary_fails_closed_on_wrong_shape() -> None:
    class BadProvider:
        def lookup(self, row_ids: mx.array) -> mx.array:
            return mx.zeros(row_ids.shape)

    with pytest.raises(ValueError, match="provider returned"):
        lookup_ple_rows(
            BadProvider(),
            _array(np.array([[[1, 2]]], dtype=np.int64)),
            row_width=2,
        )


def test_runtime_ngram_hash_fails_closed_on_noncontiguous_table_offsets() -> None:
    with pytest.raises(ValueError, match="concatenate table_sizes"):
        Qwen4NGramHasher(
            eos_token_id=9,
            ngram_size=3,
            heads_per_ngram=1,
            multipliers=np.array([3, 5, 7], dtype=np.int64),
            table_sizes=np.array([11, 13], dtype=np.int64),
            table_offsets=np.array([0, 12], dtype=np.int64),
        )


class _ScalarRowProvider:
    def lookup(self, row_ids: mx.array) -> mx.array:
        return row_ids.astype(mx.float32)[..., None] * 0.01


def _tiny_ple_layer() -> Qwen4PLELayer:
    layer = Qwen4PLELayer(
        _tiny_hasher(),
        _ScalarRowProvider(),
        row_width=1,
        hidden_size=2,
        branch_count=2,
        conv_kernel_size=2,
        conv_dilation=2,
    )
    layer.key_proj.weight = _array(
        np.array(
            [
                [0.2, -0.1, 0.3, 0.4],
                [-0.4, 0.2, 0.1, 0.3],
                [0.1, 0.5, -0.2, 0.2],
                [0.3, -0.3, 0.4, -0.1],
            ],
            dtype=np.float32,
        )
    )
    layer.value_proj.weight = _array(
        np.array(
            [[0.5, -0.2, 0.1, 0.3], [-0.1, 0.4, 0.2, -0.3]],
            dtype=np.float32,
        )
    )
    layer.norm_key.weight = mx.zeros((4,), dtype=mx.float32)
    layer.norm_query.weight = mx.zeros((4,), dtype=mx.float32)
    layer.norm_conv.weight = mx.zeros((4,), dtype=mx.float32)
    layer.conv1d.weight = _array(
        np.array(
            [
                [[0.2], [0.5]],
                [[-0.3], [0.1]],
                [[0.4], [-0.2]],
                [[0.1], [0.3]],
            ],
            dtype=np.float32,
        )
    )
    return layer


def test_runtime_ple_compute_matches_independent_gate_and_convolution_reference() -> None:
    layer = _tiny_ple_layer()
    tokens = np.array([[2, 4, 9, 6]], dtype=np.int64)
    hidden = np.array(
        [
            [
                [0.2, -0.4, 0.3, 0.5],
                [0.7, 0.1, -0.2, 0.4],
                [-0.3, 0.6, 0.8, -0.1],
                [0.5, -0.5, 0.2, 0.9],
            ]
        ],
        dtype=np.float32,
    )

    got = layer(_array(hidden), _array(tokens))
    hashed = _tiny_hasher()(_array(tokens))
    mx.eval(hashed.row_ids)
    embeddings = np.asarray(hashed.row_ids).astype(np.float32) * np.float32(0.01)
    key_weight = np.asarray(layer.key_proj.weight)
    value_weight = np.asarray(layer.value_proj.weight)
    projected_key = embeddings @ key_weight.T
    projected_value = embeddings @ value_weight.T
    gated = ple_gated_value(
        hidden,
        projected_key,
        projected_value,
        np.zeros(4, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        branch_count=2,
        hidden_size=2,
    )
    expected = ple_finish(
        gated,
        np.zeros(4, dtype=np.float32),
        np.asarray(layer.conv1d.weight)[..., 0],
        branch_count=2,
        hidden_size=2,
        dilation=2,
    )
    mx.eval(got.output, got.state.token_context, got.state.conv_state)

    assert np.allclose(np.asarray(got.output), expected.output, rtol=0, atol=2e-6)
    assert np.array_equal(np.asarray(got.state.token_context), np.asarray(hashed.next_context))
    assert np.allclose(
        np.asarray(got.state.conv_state),
        expected.next_conv_state.transpose(0, 2, 1),
        rtol=0,
        atol=2e-7,
    )
    assert got.state.offset == tokens.shape[1]


def test_runtime_ple_state_matches_full_chunks_and_tokens() -> None:
    layer = _tiny_ple_layer()
    tokens = _array(np.array([[2, 4, 9, 6, 3]], dtype=np.int64))
    hidden = _array(np.random.default_rng(47).normal(size=(1, 5, 4)).astype(np.float32))

    full = layer(hidden, tokens)

    chunk_state = None
    chunks = []
    for start, end in ((0, 2), (2, 5)):
        result = layer(hidden[:, start:end], tokens[:, start:end], state=chunk_state)
        chunks.append(result.output)
        chunk_state = result.state

    token_state = None
    token_outputs = []
    for position in range(tokens.shape[1]):
        result = layer(
            hidden[:, position : position + 1],
            tokens[:, position : position + 1],
            state=token_state,
        )
        token_outputs.append(result.output)
        token_state = result.state

    chunked = mx.concatenate(chunks, axis=1)
    tokenwise = mx.concatenate(token_outputs, axis=1)
    mx.eval(
        full.output,
        full.state.token_context,
        full.state.conv_state,
        chunked,
        chunk_state.token_context,
        chunk_state.conv_state,
        tokenwise,
        token_state.token_context,
        token_state.conv_state,
    )

    assert np.allclose(np.asarray(chunked), np.asarray(full.output), rtol=0, atol=2e-6)
    assert np.allclose(np.asarray(tokenwise), np.asarray(full.output), rtol=0, atol=2e-6)
    assert np.array_equal(
        np.asarray(chunk_state.token_context), np.asarray(full.state.token_context)
    )
    assert np.array_equal(
        np.asarray(token_state.token_context), np.asarray(full.state.token_context)
    )
    assert np.allclose(
        np.asarray(chunk_state.conv_state),
        np.asarray(full.state.conv_state),
        rtol=0,
        atol=2e-7,
    )
    assert np.allclose(
        np.asarray(token_state.conv_state),
        np.asarray(full.state.conv_state),
        rtol=0,
        atol=2e-7,
    )
    assert chunk_state.offset == token_state.offset == full.state.offset == 5


def test_runtime_ple_rejects_state_from_another_geometry() -> None:
    layer = _tiny_ple_layer()
    bad_state = Qwen4PLEState(
        token_context=mx.full((1, 2), 9, dtype=mx.int64),
        conv_state=mx.zeros((1, 3, 4), dtype=mx.float32),
        offset=1,
    )

    with pytest.raises(ValueError, match="convolution state"):
        layer(
            mx.zeros((1, 1, 4), dtype=mx.float32),
            mx.array([[2]], dtype=mx.int64),
            state=bad_state,
        )


def test_runtime_ple_state_contract_fails_closed() -> None:
    layer = _tiny_ple_layer()
    valid = Qwen4PLEState(
        token_context=mx.full((1, 2), 9, dtype=mx.int64),
        conv_state=mx.zeros((1, 2, 4), dtype=mx.float32),
        offset=1,
    )
    layer.validate_state(valid, expected_frontier=1, batch_size=1)

    with pytest.raises(ValueError, match="empty at frontier zero"):
        layer.validate_state(valid, expected_frontier=0, batch_size=1)
    with pytest.raises(ValueError, match="must contain integers"):
        layer.validate_state(
            Qwen4PLEState(
                token_context=mx.zeros((1, 2), dtype=mx.float32),
                conv_state=valid.conv_state,
                offset=1,
            ),
            expected_frontier=1,
            batch_size=1,
        )
    with pytest.raises(ValueError, match="incompatible dtype"):
        layer.validate_state(
            Qwen4PLEState(
                token_context=valid.token_context,
                conv_state=mx.zeros((1, 2, 4), dtype=mx.float16),
                offset=1,
            ),
            expected_frontier=1,
            batch_size=1,
        )
