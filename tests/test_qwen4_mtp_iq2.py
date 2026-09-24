"""IQ2 MTP projections preserve planned bytes and explicit numeric boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from moespresso.package.qwen4.mtp_format import mtp_iq2_projection_plan
from moespresso.package.qwen4.mtp_sidecar import encode_mtp_iq2_projection


@pytest.mark.parametrize("shape,widths,stored_rows", [
    ((4, 320), (768,), 16),
    ((1, 10240), (4096, 4096, 2048), 16),
    ((16, 6144), (4096, 2048), 16),
    ((512, 2560, 640), (768,), 2560),
    ((512, 1280, 2560), (2560,), 1280),
])
def test_iq2_projection_plan_covers_mtp_geometry(shape, widths, stored_rows):
    plan = mtp_iq2_projection_plan(shape)
    assert tuple(part.stored_width for part in plan.slices) == widths
    assert plan.stored_out_features == stored_rows
    assert plan.slices[0].begin == 0
    assert plan.slices[-1].end == shape[-1]
    assert all(a.end == b.begin for a, b in zip(plan.slices, plan.slices[1:]))
    assert plan.encoded_bytes == plan.num_experts * stored_rows * sum(widths) // 256 * 76


@pytest.mark.parametrize("shape", [(0, 320), (1, -1), (2,), (1, 2, 3, 4), (True, 640), (1, 640.0)])
def test_iq2_projection_plan_refuses_invalid_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        mtp_iq2_projection_plan(shape)


def test_iq2_encoding_requires_an_explicit_objective():
    weights = np.zeros((1, 320), dtype=np.float32)
    plan = mtp_iq2_projection_plan(weights.shape)
    with pytest.raises(ValueError, match="uncalibrated"):
        encode_mtp_iq2_projection(weights, plan)
    for importance in (np.ones(319), np.zeros(320), np.full(320, np.nan), -np.ones(320)):
        with pytest.raises(ValueError, match="importance"):
            encode_mtp_iq2_projection(weights, plan, importance)


def _encoded(shape):
    from mlx_iqk.codec import load

    load(build_if_missing=False)
    rng = np.random.default_rng(902)
    weights = rng.standard_normal(shape).astype(np.float32) * 0.02
    plan = mtp_iq2_projection_plan(shape)
    streams = encode_mtp_iq2_projection(weights, plan, np.ones(shape[-1], dtype=np.float32))
    assert sum(array.nbytes for part in streams for array in part.values()) == plan.encoded_bytes
    return plan, streams


def _load(module, streams):
    import mlx.core as mx

    module.load_streams([{name: mx.array(value) for name, value in part.items()} for part in streams])


def _decoded(streams, width):
    from mlx_iqk.codec import dequantize
    from mlx_iqk.format import unpack

    experts, rows = streams["qs"].shape[:2]
    flat = {name: value.reshape(experts * rows, -1) for name, value in streams.items()}
    wire = unpack("iq2_k", flat, width)
    return dequantize("iq2_k", wire, width).reshape(experts, rows, width)


@pytest.mark.parametrize("width", [320, 640, 2560, 6144, 10240])
def test_iq2_dense_matches_independent_decode_with_padding_and_slices(width):
    import mlx.core as mx

    from moespresso.runtime.qwen4.mtp_iq2 import Qwen4MTPIQ2Linear
    from moespresso.runtime.qwen4.primitives import qwen4_projection_compute_dtype

    plan, streams = _encoded((4, width))
    module = Qwen4MTPIQ2Linear(4, width)
    _load(module, streams)
    values = mx.array(np.random.default_rng(31).standard_normal((1, 3, width))).astype(mx.bfloat16)
    result = module(values)
    reference = None
    for part, packed in zip(plan.slices, streams, strict=True):
        weights = _decoded(packed, part.stored_width)[0, :4, :part.end - part.begin]
        operand = np.asarray(values.astype(mx.float32))[..., part.begin:part.end]
        projected = operand @ weights.T
        reference = projected if reference is None else reference + projected
    mx.eval(result)
    got = np.asarray(result.astype(mx.float32))
    scale = max(float(np.max(np.abs(reference))), 1e-6)
    assert float(np.max(np.abs(got - reference))) / scale < 0.012
    assert result.shape == (1, 3, 4)
    assert result.dtype == mx.bfloat16
    assert qwen4_projection_compute_dtype(module) == mx.bfloat16


def test_iq2_experts_preserve_pair_rows_and_do_not_dequantize_large_batches(monkeypatch):
    import mlx.core as mx
    from mlx_iqk.nn import IqkSwitchLinear

    from moespresso.runtime.qwen4.mtp_iq2 import Qwen4MTPIQ2SwitchLinear

    plan, streams = _encoded((3, 16, 640))
    module = Qwen4MTPIQ2SwitchLinear(3, 16, 640)
    _load(module, streams)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("MTP must not dequantize the entire expert stack")

    monkeypatch.setattr(IqkSwitchLinear, "dequantized", forbidden)
    ids = mx.array(np.tile([2, 0, 1], (32, 1)), dtype=mx.uint32)
    values = mx.array(np.random.default_rng(7).standard_normal((32, 640))).astype(mx.float16)
    shared = module(values, ids)
    per_pair = module(mx.broadcast_to(values[:, None, :], (32, 3, 640)), ids)
    weights = _decoded(streams[0], plan.slices[0].stored_width)[:, :, :640]
    expected = np.einsum("ti,tkoi->tko", np.asarray(values).astype(np.float32), weights[np.asarray(ids)])
    mx.eval(shared, per_pair)
    np.testing.assert_array_equal(np.asarray(shared), np.asarray(per_pair))
    scale = float(np.max(np.abs(expected)))
    assert float(np.max(np.abs(np.asarray(shared) - expected))) / scale < 0.003


def test_iq2_stream_loading_is_validated_before_installation():
    import mlx.core as mx

    from moespresso.runtime.qwen4.mtp_iq2 import Qwen4MTPIQ2Linear

    _plan, encoded = _encoded((4, 6144))
    streams = [{name: mx.array(value) for name, value in part.items()} for part in encoded]
    module = Qwen4MTPIQ2Linear(4, 6144)
    invalid = [dict(part) for part in streams]
    key = next(iter(invalid[-1]))
    invalid[-1][key] = invalid[-1][key].astype(mx.float32)
    with pytest.raises(ValueError, match="shape or dtype"):
        module.load_streams(invalid)
    assert module._loaded is False
    with pytest.raises(ValueError, match="not been loaded"):
        module(mx.zeros((1, 6144)))
    module.load_streams(streams)
    with pytest.raises(ValueError, match="immutable"):
        module.load_streams(streams)
    with pytest.raises(ValueError, match="activation shape"):
        module(mx.zeros((1, 640)))
