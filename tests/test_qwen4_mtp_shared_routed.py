"""Shared gate/up verification addresses independent projection slot maps."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_iqk.format import component_dtypes, component_shapes
from mlx_iqk.nn import IqkSwitchLinear
from mlx_iqk.routed import SUPPORTED_CODEC_TUPLES

from moespresso.runtime.qwen4.moe import expert_major_weighted_sum
from moespresso.runtime.qwen4.mtp_shared_routed import (
    _gate_source, qwen4_mtp_shared_routed_pair,
)


def _projection(member, output, width, seed):
    rng = np.random.default_rng(seed)
    module = IqkSwitchLinear(member, 20, output, width)
    streams = {}
    for name, shape in component_shapes(member, 20, output, width).items():
        dtype = component_dtypes(member)[name]
        if np.issubdtype(dtype, np.floating):
            values = (rng.standard_normal(shape) * 0.002).astype(dtype)
        else:
            values = rng.integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
        streams[name] = mx.array(values)
    module.load_streams(streams)
    mx.eval(*module._streams())
    return module


@pytest.fixture(scope="module", params=SUPPORTED_CODEC_TUPLES)
def projections(request):
    gate, up, down = request.param
    return (_projection(gate, 640, 2560, 101), _projection(up, 640, 2560, 103),
            _projection(down, 2560, 768, 107))


def _inputs(overlap):
    rng = np.random.default_rng(71 + overlap)
    routes = np.array([rng.permutation(10),
                       rng.permutation(np.arange(10 - overlap, 20 - overlap))])
    source_ids = (rng.permutation(20) * 19 + 5).astype(np.uint32)
    union = np.unique(routes)
    positions = np.full((len(union), 2), -1, dtype=np.int32)
    for entry, expert in enumerate(union):
        for row in range(2):
            hits = np.nonzero(routes[row] == expert)[0]
            if len(hits):
                positions[entry, row] = hits[0]
    maps = np.stack([rng.permutation(20) for _ in range(3)], axis=1).astype(np.uint32)
    hidden = mx.array(rng.normal(scale=0.2, size=(1, 2, 2560))).astype(mx.bfloat16)
    scores = mx.array(rng.uniform(0.03, 0.17, size=(1, 2, 10))).astype(mx.bfloat16)
    return (hidden, mx.array(source_ids[routes][None]), scores,
            mx.array(maps[union]), mx.array(positions)), maps[routes]


def _serial(projections, values, row_slots):
    gate, up, down = projections
    hidden, indices, scores, *_ = values
    outputs = []
    for row in range(2):
        x = hidden[:, row:row + 1, None, None, :]
        gs, us, ds = (mx.array(row_slots[row, :, column][None, None]) for column in range(3))
        g, u = gate.gemv(x, gs), up.gemv(x, us)
        activation = (g * mx.sigmoid(g)) * u
        padded = mx.pad(activation, [(0, 0)] * (activation.ndim - 1) + [(0, 128)])
        result = down.gemv(padded, ds).squeeze(-2).astype(mx.bfloat16)
        outputs.append(expert_major_weighted_sum(result, scores[:, row:row + 1], indices[:, row:row + 1]))
    return mx.concatenate(outputs, axis=1)


@pytest.mark.parametrize("overlap", [0, 4, 10])
@pytest.mark.parametrize("precomputed_down", [False, True])
def test_shared_pair_matches_selected_expert_reference(projections, overlap, precomputed_down):
    values, slots = _inputs(overlap)
    kwargs = {"down_slots": mx.array(slots[:, :, 2][None])} if precomputed_down else {}
    actual = qwen4_mtp_shared_routed_pair(*projections, *values, **kwargs)
    expected = _serial(projections, values, slots)
    mx.eval(actual, expected)
    assert bool(mx.all(mx.isfinite(actual)))
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32)),
        rtol=0.02, atol=2e-4,
    )


@pytest.mark.parametrize("member", ["iq2_k", "iq3_k"])
def test_generated_gate_decodes_each_weight_for_both_accumulators(member):
    source = _gate_source(member)
    for prefix in ("g", "u"):
        assert source.count(f"uint2 {prefix}qw =") == 1
        assert source.count(f"float {prefix}_v0 =") == 1
        assert f"fma({prefix}_v0, xv0[0]" in source
        assert f"fma({prefix}_v0, xv1[0]" in source
    assert "threadgroup_position_in_grid.y" not in source
