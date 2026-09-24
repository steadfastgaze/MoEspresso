"""Packed prefill preserves expert assignments and reconstructed operands."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_iqk.format import component_dtypes, component_shapes
from mlx_iqk.nn import IqkSwitchLinear

from moespresso.runtime.qwen4.prefill_packed import packed_gate_up
from moespresso.runtime.qwen4.prefill_packed_down import packed_down
from moespresso.runtime.pooled_switchglu import PooledSwitchGLU
from moespresso.runtime.qwen4.expert_provider import (
    Qwen4PaddedPooledSwitchGLU,
    Qwen4PooledSwitchGLU,
    Qwen4ZeroPaddedDownProjection,
)


class _ArrayStub:
    def __init__(self, shape, dtype="bfloat16"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.casts = []

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def astype(self, dtype):
        self.casts.append(dtype)
        return _ArrayStub(self.shape, dtype)


class _PoolStub:
    def __init__(self, iqk, identity=True):
        self.iqk = iqk
        self.identity = identity

    def slot_table_is_identity(self):
        return self.identity


def _integration_executor(
    *,
    members=("iq2_k", "iq2_k", "iq2_k"),
    identities=(True, True, True),
    hidden_size=2560,
    intermediate_size=640,
    padded_down=True,
    stored_down=768,
):
    executor = Qwen4PaddedPooledSwitchGLU.__new__(Qwen4PaddedPooledSwitchGLU)
    pools = tuple(
        _PoolStub(SimpleNamespace(member=member), identity)
        for member, identity in zip(members, identities, strict=True)
    )
    object.__setattr__(executor, "gate_proj", SimpleNamespace(pool=pools[0]))
    object.__setattr__(executor, "up_proj", SimpleNamespace(pool=pools[1]))
    if padded_down:
        down = Qwen4ZeroPaddedDownProjection.__new__(Qwen4ZeroPaddedDownProjection)
        object.__setattr__(down, "pool", pools[2])
        object.__setattr__(down, "stored_in_features", stored_down)
    else:
        down = SimpleNamespace(pool=pools[2], stored_in_features=stored_down)
    object.__setattr__(executor, "down_proj", down)
    object.__setattr__(
        executor,
        "members",
        dict(zip(("gate_proj", "up_proj", "down_proj"), members, strict=True)),
    )
    object.__setattr__(executor, "hidden_size", hidden_size)
    object.__setattr__(executor, "intermediate_size", intermediate_size)
    object.__setattr__(executor, "packed_prefill_calls", 0)
    object.__setattr__(executor, "packed_prefill_pairs", 0)
    return executor


def _projection(seed, output=640, width=2560, member="iq2_k"):
    rng = np.random.default_rng(seed)
    module = IqkSwitchLinear(member, 12, output, width)
    streams = {}
    for name, shape in component_shapes(member, 12, output, width).items():
        dtype = component_dtypes(member)[name]
        if np.issubdtype(dtype, np.floating):
            values = rng.uniform(0.0001, 0.0004, shape).astype(dtype)
        else:
            values = rng.integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
        streams[name] = mx.array(values)
    module.load_streams(streams)
    mx.eval(*module._streams())
    return module


@pytest.fixture(scope="module", params=["iq2_k", "iq2_ks", "iq3_k"])
def projections(request):
    return _projection(13, member=request.param), _projection(17, member=request.param)


def _reference(gate, up, x, indices):
    order = mx.argsort(indices.reshape(-1))
    slots = indices.reshape(-1)[order]
    rows = x.reshape(-1, 2560)[order // indices.shape[-1]]
    rows = rows.astype(mx.float16).reshape(-1, 1, 2560)
    g = mx.gather_mm(rows, gate.dequantized().swapaxes(-1, -2),
                     rhs_indices=slots, sorted_indices=True)
    u = mx.gather_mm(rows, up.dequantized().swapaxes(-1, -2),
                     rhs_indices=slots, sorted_indices=True)
    out = (g * mx.sigmoid(g)) * u
    return out.reshape(-1, 640)[mx.argsort(order)].reshape(*indices.shape, 640)


@pytest.mark.parametrize("tokens", [3, 19, 67])
def test_packed_gate_up_handles_uneven_expert_counts_and_tile_tails(projections, tokens):
    rng = np.random.default_rng(tokens)
    x = mx.array(rng.normal(0, 0.2, (1, tokens, 2560))).astype(mx.bfloat16)
    # Every token includes the same hot expert; the other routes vary.
    ids = np.array([[0, *rng.choice(np.arange(1, 12), 3, replace=False)]
                    for _ in range(tokens)], dtype=np.uint32)[None]
    indices = mx.array(ids)
    actual = packed_gate_up(*projections, x, indices, tile_tokens=32)
    expected = _reference(*projections, x, indices)
    mx.eval(actual, expected)
    assert actual.shape == (*indices.shape, 640)
    assert actual.dtype == mx.float16
    assert bool(mx.all(mx.isfinite(actual)))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.02, atol=2e-4)


def test_packed_gate_up_keeps_route_order(projections):
    rng = np.random.default_rng(23)
    x = mx.array(rng.normal(0, 0.2, (1, 5, 2560))).astype(mx.bfloat16)
    indices = mx.array([[[3, 0, 7, 9], [0, 4, 6, 2], [2, 8, 0, 1],
                         [7, 4, 0, 9], [0, 11, 10, 6]]], dtype=mx.uint32)
    forward = packed_gate_up(*projections, x, indices)
    reversed_routes = packed_gate_up(*projections, x, indices[..., ::-1])
    mx.eval(forward, reversed_routes)
    np.testing.assert_array_equal(np.asarray(forward), np.asarray(reversed_routes[..., ::-1, :]))


@pytest.fixture(scope="module", params=["iq2_k", "iq2_ks", "iq3_k"])
def down_projection(request):
    return _projection(29, output=2560, width=768, member=request.param)


@pytest.mark.parametrize("tokens", [3, 19, 67])
def test_packed_down_uses_each_routes_activation_and_ignores_zero_padding(down_projection, tokens):
    rng = np.random.default_rng(tokens + 101)
    activation = mx.array(rng.normal(0, 0.2, (1, tokens, 4, 640))).astype(mx.float16)
    ids = np.array([[0, *rng.choice(np.arange(1, 12), 3, replace=False)]
                    for _ in range(tokens)], dtype=np.uint32)[None]
    indices = mx.array(ids)
    order = mx.argsort(indices.reshape(-1))
    slots = indices.reshape(-1)[order]
    operand = mx.pad(activation.reshape(-1, 640)[order], [(0, 0), (0, 128)])
    expected = mx.gather_mm(
        operand.reshape(-1, 1, 768), down_projection.dequantized().swapaxes(-1, -2),
        rhs_indices=slots, sorted_indices=True,
    ).reshape(-1, 2560)[mx.argsort(order)].reshape(*indices.shape, 2560)
    actual = packed_down(down_projection, activation, indices)
    mx.eval(actual, expected)
    assert actual.shape == (*indices.shape, 2560)
    assert actual.dtype == mx.float16
    assert bool(mx.all(mx.isfinite(actual)))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.02, atol=2e-4)


@pytest.mark.parametrize(
    ("case", "executor_kwargs", "x_shape"),
    [
        ("decode", {}, (1, 1, 2560)),
        ("short_prefill", {}, (1, 64, 2560)),
        ("final_partial_chunk", {}, (1, 256, 2560)),
        ("crossover_minus_one", {}, (1, 1023, 2560)),
        ("rank_two", {}, (1024, 2560)),
        ("batched", {}, (2, 1024, 2560)),
        ("hidden_geometry", {"hidden_size": 2048}, (1, 1024, 2560)),
        (
            "unpaired_gate_up",
            {"members": ("iq3_k", "iq2_k", "iq2_k")},
            (1, 1024, 2560),
        ),
        (
            "unsupported_codec",
            {"members": ("iq1_s_r4", "iq1_s_r4", "iq2_k")},
            (1, 1024, 2560),
        ),
        ("unwrapped_down", {"padded_down": False}, (1, 1024, 2560)),
        ("wrong_down_width", {"stored_down": 640}, (1, 1024, 2560)),
        (
            "nonidentity_slots",
            {"identities": (True, False, True)},
            (1, 1024, 2560),
        ),
    ],
)
def test_packed_prefill_integration_falls_back(monkeypatch, case, executor_kwargs, x_shape):
    executor = _integration_executor(**executor_kwargs)
    x = _ArrayStub(x_shape)
    indices = _ArrayStub((*x_shape[:-1], 4), "uint32")
    fallback = object()
    calls = []

    def fallback_call(self, value, selected):
        calls.append((self, value, selected))
        return fallback

    def unexpected_packed(*args, **kwargs):
        raise AssertionError(f"packed path engaged for fallback case {case}")

    monkeypatch.setattr(
        Qwen4PaddedPooledSwitchGLU, "_iqk_sorted_threshold", staticmethod(lambda: 4096),
    )
    monkeypatch.setattr(PooledSwitchGLU, "_call_iqk_full_resident", fallback_call)
    monkeypatch.setattr(
        "moespresso.runtime.qwen4.prefill_packed.packed_gate_up", unexpected_packed,
    )
    monkeypatch.setattr(
        "moespresso.runtime.qwen4.prefill_packed_down.packed_down", unexpected_packed,
    )

    result = Qwen4PaddedPooledSwitchGLU._call_iqk_full_resident(executor, x, indices)

    assert result is fallback
    assert calls == [(executor, x, indices)]
    assert executor.packed_prefill_calls == 0
    assert executor.packed_prefill_pairs == 0


def test_packed_prefill_integration_accepts_supported_projection_pair(monkeypatch):
    executor = _integration_executor(members=("iq3_k", "iq3_k", "iq2_k"))
    x = _ArrayStub((1, 1024, 2560), mx.bfloat16)
    indices = _ArrayStub((1, 1024, 4), mx.uint32)
    activation = object()
    output = _ArrayStub((1, 1024, 4, 2560), mx.float16)
    calls = []

    def fail_fallback(*args, **kwargs):
        raise AssertionError("supported packed prefill fell back")

    def gate_up(gate, up, hidden, selected):
        calls.append(("gate_up", gate, up, hidden, selected))
        return activation

    def down(projection, value, selected):
        calls.append(("down", projection, value, selected))
        return output

    monkeypatch.setattr(
        Qwen4PaddedPooledSwitchGLU, "_iqk_sorted_threshold", staticmethod(lambda: 4096),
    )
    monkeypatch.setattr(PooledSwitchGLU, "_call_iqk_full_resident", fail_fallback)
    monkeypatch.setattr("moespresso.runtime.qwen4.prefill_packed.packed_gate_up", gate_up)
    monkeypatch.setattr("moespresso.runtime.qwen4.prefill_packed_down.packed_down", down)

    result = Qwen4PaddedPooledSwitchGLU._call_iqk_full_resident(executor, x, indices)

    pools = (executor.gate_proj.pool, executor.up_proj.pool, executor.down_proj.pool)
    assert result is output
    assert calls == [
        ("gate_up", pools[0].iqk, pools[1].iqk, x, indices),
        ("down", pools[2].iqk, activation, indices),
    ]
    assert executor.packed_prefill_calls == 1
    assert executor.packed_prefill_pairs == 4096


@pytest.mark.parametrize(("tokens", "engages"), [(2, False), (3, True)])
def test_packed_prefill_integration_respects_configured_pair_threshold(
    monkeypatch, tokens, engages,
):
    executor = _integration_executor()
    x = _ArrayStub((1, tokens, 2560), mx.bfloat16)
    indices = _ArrayStub((1, tokens, 4), mx.uint32)
    activation = object()
    packed_output = object()
    fallback_output = object()
    calls = []

    def fallback(*args, **kwargs):
        calls.append("fallback")
        return fallback_output

    def gate_up(*args, **kwargs):
        calls.append("gate_up")
        return activation

    def down(projection, value, selected):
        assert value is activation
        calls.append("down")
        return packed_output

    monkeypatch.setattr(
        Qwen4PaddedPooledSwitchGLU, "_iqk_sorted_threshold", staticmethod(lambda: 12),
    )
    monkeypatch.setattr(PooledSwitchGLU, "_call_iqk_full_resident", fallback)
    monkeypatch.setattr("moespresso.runtime.qwen4.prefill_packed.packed_gate_up", gate_up)
    monkeypatch.setattr("moespresso.runtime.qwen4.prefill_packed_down.packed_down", down)

    result = Qwen4PaddedPooledSwitchGLU._call_iqk_full_resident(executor, x, indices)

    assert result is (packed_output if engages else fallback_output)
    assert calls == (["gate_up", "down"] if engages else ["fallback"])
    assert executor.packed_prefill_calls == int(engages)
    assert executor.packed_prefill_pairs == (tokens * 4 if engages else 0)


def test_qwen_switchglu_preserves_input_dtype_after_packed_output(monkeypatch):
    executor = Qwen4PooledSwitchGLU.__new__(Qwen4PooledSwitchGLU)
    value = _ArrayStub((1, 256, 2560), mx.bfloat16)
    indices = _ArrayStub((1, 256, 4), mx.uint32)
    packed_output = _ArrayStub((1, 256, 4, 2560), mx.float16)
    monkeypatch.setattr(PooledSwitchGLU, "__call__", lambda *args, **kwargs: packed_output)

    output = Qwen4PooledSwitchGLU.__call__(executor, value, indices)

    assert output.dtype == mx.bfloat16
    assert packed_output.casts == [mx.bfloat16]
