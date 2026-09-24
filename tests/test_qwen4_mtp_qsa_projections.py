"""Two-row QSA projections retain sequential attention and cache updates."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from moespresso.runtime.qwen4.mtp_qsa_projections import prepare_mtp_qsa_projection_rows
from test_qwen4_qsa_adapter import _small_qsa


class _Projection:
    def __init__(self, input_width, output_width):
        self.weight = mx.zeros((output_width, input_width), dtype=mx.bfloat16)
        self.output_width = output_width
        self.calls = []

    def __call__(self, hidden):
        self.calls.append(tuple(hidden.shape))
        return mx.zeros((*hidden.shape[:-1], self.output_width), dtype=hidden.dtype)


def _fake_mixer():
    index = _Projection(4, 6)
    query = _Projection(4, 16)
    key = _Projection(4, 4)
    value = _Projection(4, 4)
    index_norm = SimpleNamespace(weight=mx.zeros((2,), dtype=mx.bfloat16), eps=1e-6)
    query_norm = SimpleNamespace(weight=mx.zeros((4,), dtype=mx.bfloat16), eps=1e-6)
    key_norm = SimpleNamespace(weight=mx.zeros((4,), dtype=mx.bfloat16), eps=1e-6)
    module = SimpleNamespace(
        hidden_size=4,
        index_query_heads=2,
        index_kv_heads=1,
        index_head_dim=2,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=4,
        rotary_dim=2,
        indexer=SimpleNamespace(index_qk_proj=index, q_layernorm=index_norm),
        q_proj=query,
        k_proj=key,
        v_proj=value,
        q_norm=query_norm,
        k_norm=key_norm,
    )
    factors = SimpleNamespace(
        cosine=mx.ones((1, 2, 1, 2), dtype=mx.bfloat16),
        sine=mx.zeros((1, 2, 1, 2), dtype=mx.bfloat16),
    )
    mixer = SimpleNamespace(
        module=module,
        _projection_dtype=lambda: mx.bfloat16,
        _prepare_mtp_rope_factors=lambda _positions: factors,
    )
    return mixer, (index, query, key, value)


def _positions():
    return mx.broadcast_to(mx.arange(2, dtype=mx.int32)[None, None], (3, 1, 2))


def test_projection_rows_issue_one_two_row_call_per_dense_input_projection():
    mixer, projections = _fake_mixer()
    hidden = mx.zeros((1, 2, 4), dtype=mx.bfloat16)

    prepared = prepare_mtp_qsa_projection_rows(mixer, hidden, _positions())

    assert all(projection.calls == [(1, 2, 4)] for projection in projections)
    assert tuple(row.hidden_states.shape for row in prepared.rows) == ((1, 1, 4), (1, 1, 4))
    assert tuple(row.index_queries.shape for row in prepared.rows) == ((1, 1, 2, 2), (1, 1, 2, 2))


def test_projection_row_is_bound_to_its_mixer_and_internal_hidden_view():
    mixer, _projections = _fake_mixer()
    prepared = prepare_mtp_qsa_projection_rows(
        mixer,
        mx.zeros((1, 2, 4), dtype=mx.bfloat16),
        _positions(),
    )
    row = prepared.rows[0]

    row.validate(mixer, row.hidden_states)
    with pytest.raises(ValueError, match="do not belong"):
        row.validate(object(), row.hidden_states)
    with pytest.raises(ValueError, match="do not belong"):
        row.validate(mixer, mx.zeros((1, 1, 4), dtype=mx.bfloat16))


def _advance_two_rows(adapter, hidden, *, prepared=None):
    state = None
    outputs = []
    for row in range(2):
        projection = None if prepared is None else prepared.rows[row]
        row_hidden = hidden[:, row : row + 1] if projection is None else projection.hidden_states
        result = adapter.step_trusted(
            row_hidden,
            valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
            visible_history=mx.ones((1, row + 1), dtype=mx.bool_),
            position_ids=mx.full((3, 1, 1), row, dtype=mx.int32),
            state=state,
            prepared_projections=projection,
        )
        outputs.append(result.output)
        state = result.state
    return mx.concatenate(outputs, axis=1), state


def test_prepared_input_projections_keep_sequential_attention_and_cache_results():
    _serial_module, serial = _small_qsa()
    _shared_module, shared = _small_qsa()
    hidden = mx.array([[[0.25, -0.5], [0.75, 0.125]]], dtype=mx.float32)

    expected_output, expected_state = _advance_two_rows(serial, hidden)
    prepared = prepare_mtp_qsa_projection_rows(shared, hidden, _positions())
    actual_output, actual_state = _advance_two_rows(shared, hidden, prepared=prepared)
    mx.eval(actual_output, expected_output, actual_state.keys, expected_state.keys)

    np.testing.assert_allclose(
        np.asarray(actual_output), np.asarray(expected_output), rtol=1e-5, atol=1e-6
    )
    assert actual_state.offset == expected_state.offset == 2
    for actual, expected in (
        (actual_state.keys, expected_state.keys),
        (actual_state.values, expected_state.values),
        (actual_state.raw_index_keys, expected_state.raw_index_keys),
        (actual_state.position_ids, expected_state.position_ids),
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)
