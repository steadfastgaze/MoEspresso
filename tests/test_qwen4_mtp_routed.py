"""Paired target expert kernels preserve the incumbent per-row arithmetic."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_iqk.format import component_dtypes, component_shapes
from mlx_iqk.nn import IqkSwitchLinear
from mlx_iqk.routed import down_reduce, gate_up_swiglu, SUPPORTED_CODEC_TUPLES

from moespresso.runtime.qwen4.mtp_routed import qwen4_mtp_routed_pair


def _projection(member, output, width, seed):
    rng = np.random.default_rng(seed)
    module = IqkSwitchLinear(member, 20, output, width)
    streams = {}
    for name, shape in component_shapes(member, 20, output, width).items():
        dtype = component_dtypes(member)[name]
        if np.issubdtype(dtype, np.floating):
            value = (rng.standard_normal(shape) * 0.0001).astype(dtype)
        else:
            value = rng.integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
        streams[name] = mx.array(value)
    module.load_streams(streams)
    mx.eval(*module._streams())
    return module


@pytest.fixture(scope="module", params=SUPPORTED_CODEC_TUPLES)
def projections(request):
    gate, up, down = request.param
    return (_projection(gate, 640, 2560, 101), _projection(up, 640, 2560, 103),
            _projection(down, 2560, 768, 107))


def _inputs(overlap):
    rng = np.random.default_rng(37 + overlap)
    rows = np.array([rng.permutation(10), rng.permutation(np.arange(10 - overlap, 20 - overlap))])
    source_ids = (rng.permutation(20) * 19 + 5).astype(np.uint32)
    gate_map, down_map = rng.permutation(20).astype(np.uint32), rng.permutation(20).astype(np.uint32)
    hidden = mx.array(rng.standard_normal((1, 2, 2560)).astype(np.float32)).astype(mx.bfloat16)
    weights = mx.array(rng.random((1, 2, 10)).astype(np.float32)).astype(mx.bfloat16)
    scores = (weights / mx.sum(weights, axis=-1, keepdims=True)).astype(mx.bfloat16)
    return (hidden, mx.array(source_ids[rows][None]), scores,
            mx.array(gate_map[rows][None]), mx.array(down_map[rows][None]))


def _serial(projections, values):
    gate, up, down = projections
    hidden, source, scores, gate_slots, down_slots = values
    rows = []
    for row in range(2):
        order = mx.argsort(source[0, row])
        activation = gate_up_swiglu(gate, up, hidden[:, row:row + 1], gate_slots[0, row][order])
        rows.append(down_reduce(down, activation, down_slots[0, row][order], scores[0, row][order]))
    return mx.stack(rows)[None]


@pytest.mark.parametrize("overlap", [0, 5, 10])
def test_mtp_pair_matches_two_incumbent_rows_bit_for_bit(projections, overlap):
    values = _inputs(overlap)
    want = _serial(projections, values)
    got = qwen4_mtp_routed_pair(*projections, *values)
    mx.eval(want, got)
    assert bool(mx.all(mx.isfinite(want)))
    np.testing.assert_array_equal(np.asarray(got.view(mx.uint16)), np.asarray(want.view(mx.uint16)))


@pytest.mark.parametrize("change", ["hidden_rows", "hidden_dtype", "score_dtype", "slot_dtype", "slot_shape"])
def test_mtp_pair_refuses_an_incompatible_tensor_contract(projections, change):
    values = list(_inputs(5))
    if change == "hidden_rows":
        values[0] = values[0][:, :1]
    elif change == "hidden_dtype":
        values[0] = values[0].astype(mx.float16)
    elif change == "score_dtype":
        values[2] = values[2].astype(mx.float32)
    elif change == "slot_dtype":
        values[3] = values[3].astype(mx.int32)
    else:
        values[4] = values[4].reshape(2, 10)
    with pytest.raises(ValueError):
        qwen4_mtp_routed_pair(*projections, *values)
