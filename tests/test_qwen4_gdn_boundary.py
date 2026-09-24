from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import moespresso.runtime.qwen4.gdn as qwen4_gdn


class _Projection:
    def __init__(self, width: int) -> None:
        self.width = width
        self.inputs: list[mx.array] = []

    def __call__(self, inputs: mx.array) -> mx.array:
        self.inputs.append(inputs)
        return mx.zeros((*inputs.shape[:-1], self.width), dtype=mx.bfloat16)


class _Conv:
    def __init__(self) -> None:
        self.weight = mx.zeros((10240, 4, 1), dtype=mx.bfloat16)

    def __call__(self, inputs: mx.array) -> mx.array:
        return mx.zeros(
            (inputs.shape[0], inputs.shape[1] - 3, 10240),
            dtype=mx.bfloat16,
        )


class _Norm:
    eps = 1e-6

    def __init__(self) -> None:
        self.weight = mx.zeros((128,), dtype=mx.bfloat16)

    def __call__(self, values: mx.array, gate: mx.array) -> mx.array:
        assert values.shape == gate.shape
        return values


class _OutProjection:
    def __call__(self, values: mx.array) -> mx.array:
        return mx.zeros((*values.shape[:-1], 2560), dtype=mx.bfloat16)


class _Cache:
    lengths = None
    left_padding = None

    def __init__(self) -> None:
        self.state = [
            mx.zeros((1, 3, 10240), dtype=mx.bfloat16),
            mx.zeros((1, 48, 128, 128), dtype=mx.float32),
        ]
        self.advances: list[int] = []

    def __getitem__(self, index: int) -> mx.array:
        return self.state[index]

    def __setitem__(self, index: int, value: mx.array) -> None:
        self.state[index] = value

    def advance(self, length: int) -> None:
        self.advances.append(length)


def _module() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2560,
        conv_dim=10240,
        conv_kernel_size=4,
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        key_dim=2048,
        value_dim=6144,
        training=False,
        sharding_group=None,
        in_proj_qkv=_Projection(10240),
        in_proj_z=_Projection(6144),
        in_proj_b=_Projection(48),
        in_proj_a=_Projection(48),
        conv1d=_Conv(),
        A_log=mx.zeros((48,), dtype=mx.bfloat16),
        dt_bias=mx.zeros((48,), dtype=mx.bfloat16),
        norm=_Norm(),
        out_proj=_OutProjection(),
    )


def _generic_update(
    _query,
    _key,
    value,
    _decay_logits,
    _beta_logits,
    _a_log,
    _dt_bias,
    state,
    *,
    use_kernel,
):
    del use_kernel
    return value, state


def _call(module, inputs, *, mask, cache):
    return qwen4_gdn.Qwen4GatedDeltaNet.__call__(
        module,
        inputs,
        mask=mask,
        cache=cache,
    )


def test_gdn_prerouter_returns_coordinator_owned_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx_kquant as kq

    def projection():
        return SimpleNamespace(
            weight=mx.zeros((1,), dtype=mx.uint8),
            scales=mx.zeros((1,), dtype=mx.uint8),
        )

    attention = SimpleNamespace(
        hc_norm=SimpleNamespace(weight=mx.zeros((10240,), dtype=mx.bfloat16), eps=1e-6),
        input_mix_weight_down=projection(),
        block_inject_weight=projection(),
        input_mix_weight_up=projection(),
    )
    mlp = SimpleNamespace(
        hc_norm=SimpleNamespace(weight=mx.zeros((10240,), dtype=mx.bfloat16), eps=1e-6),
        input_mix_weight_down=projection(),
        block_inject_weight=projection(),
        input_mix_weight_up=projection(),
    )
    module = SimpleNamespace(
        in_proj_qkv=projection(),
        in_proj_z=projection(),
        in_proj_b=projection(),
        in_proj_a=projection(),
        conv1d=SimpleNamespace(weight=mx.zeros((10240, 4, 1), dtype=mx.bfloat16)),
        A_log=mx.zeros((48,), dtype=mx.bfloat16),
        dt_bias=mx.zeros((48,), dtype=mx.bfloat16),
        norm=SimpleNamespace(weight=mx.zeros((128,), dtype=mx.bfloat16), eps=1e-6),
        out_proj=projection(),
    )
    layer = SimpleNamespace(
        mixer_kind="gdn",
        attention_residual=attention,
        mlp_residual=mlp,
        mixer=SimpleNamespace(module=module),
    )
    state = qwen4_gdn.Qwen4GDNState(
        conv_state=mx.zeros((1, 3, 10240), dtype=mx.bfloat16),
        recurrent_state=mx.zeros((1, 48, 128, 128), dtype=mx.float32),
        offset=8192,
    )
    expected = (
        mx.zeros((1, 1, 2560), dtype=mx.bfloat16),
        mx.zeros((1, 1, 10240), dtype=mx.bfloat16),
        mx.zeros((1, 1, 4), dtype=mx.bfloat16),
        mx.ones((1, 3, 10240), dtype=mx.bfloat16),
        mx.ones((1, 48, 128, 128), dtype=mx.float32),
    )
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(qwen4_gdn, "_qwen4_gdn_prerouter_contract", lambda *_args: True)
    monkeypatch.setattr(kq, "qwen4_gdn_prerouter_q6", native)
    before = qwen4_gdn.qwen4_gdn_prerouter_call_counts()
    result = qwen4_gdn.qwen4_gdn_prerouter_step(
        layer,
        mx.zeros((1, 1, 10240), dtype=mx.bfloat16),
        state=state,
        pending_output=None,
        pending_injection=None,
        certified_all_valid=True,
    )
    after = qwen4_gdn.qwen4_gdn_prerouter_call_counts()

    assert result is not None
    assert result.mlp_hidden is expected[0]
    assert result.residual is expected[1]
    assert result.injection is expected[2]
    assert result.state.conv_state is expected[3]
    assert result.state.recurrent_state is expected[4]
    assert result.frontier == 8193
    assert len(calls) == 1
    assert after["native_calls"] - before["native_calls"] == 1


def test_gdn_boundary_is_automatic_and_masked_rows_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    cache = _Cache()
    inputs = mx.ones((1, 1, 2560), dtype=mx.bfloat16)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    invalid = mx.zeros((1, 1), dtype=mx.bool_)
    next_conv_state = mx.ones((1, 3, 10240), dtype=mx.bfloat16)
    next_recurrent_state = mx.ones((1, 48, 128, 128), dtype=mx.float32)
    native_calls = []

    def prepare(qkv, beta_logits, decay_logits, conv_state, *_weights):
        native_calls.append((qkv, beta_logits, decay_logits, conv_state))
        return (
            mx.zeros((1, 1, 16, 128), dtype=mx.bfloat16),
            mx.zeros((1, 1, 16, 128), dtype=mx.bfloat16),
            mx.zeros((1, 1, 48, 128), dtype=mx.bfloat16),
            mx.zeros((1, 1, 48), dtype=mx.bfloat16),
            mx.ones((1, 1, 48), dtype=mx.float32),
            next_conv_state,
        )

    def recurrence(query, key, value, decay, beta, state, mask):
        assert query.shape == key.shape == (1, 1, 16, 128)
        assert value.shape == (1, 1, 48, 128)
        assert decay.shape == beta.shape == (1, 1, 48)
        assert state.shape == (1, 48, 128, 128)
        assert mask is None
        return value, next_recurrent_state

    def norm_gate(recurrence_output, gate, weight, *, eps):
        assert recurrence_output.shape == (1, 1, 48, 128)
        assert gate.shape == (1, 1, 6144)
        assert weight is module.norm.weight
        assert eps == module.norm.eps
        return recurrence_output.reshape(1, 1, 6144)

    monkeypatch.setattr(qwen4_gdn, "_qwen4_gated_delta_update", _generic_update)
    monkeypatch.setattr(qwen4_gdn, "_qwen4_gdn_native_contract", lambda *_args: True)
    monkeypatch.setattr(
        qwen4_gdn,
        "_qwen4_gdn_native_helpers",
        lambda: (prepare, norm_gate),
    )
    monkeypatch.setattr(qwen4_gdn, "gated_delta_kernel", recurrence)
    before = qwen4_gdn.qwen4_gdn_call_counts()

    _call(module, inputs, mask=valid, cache=cache)
    _call(module, inputs, mask=invalid, cache=cache)

    assert cache[0] is next_conv_state
    assert cache[1] is next_recurrent_state
    assert cache.advances == [1, 1]
    assert len(native_calls) == 2
    masked_input = module.in_proj_qkv.inputs[1].astype(mx.float32)
    mx.eval(masked_input)
    assert np.count_nonzero(np.asarray(masked_input)) == 0

    after = qwen4_gdn.qwen4_gdn_call_counts()

    assert after["native_calls"] - before["native_calls"] == 2
    assert after["generic_calls"] == before["generic_calls"]
    assert after["fallback_input"] == before["fallback_input"]
    assert after["fallback_contract"] == before["fallback_contract"]


def test_gdn_boundary_prefill_stays_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    cache = _Cache()
    monkeypatch.setattr(qwen4_gdn, "_qwen4_gated_delta_update", _generic_update)
    monkeypatch.setattr(
        qwen4_gdn,
        "_qwen4_gdn_native_helpers",
        lambda: pytest.fail("prefill loaded the native GDN helpers"),
    )
    before = qwen4_gdn.qwen4_gdn_call_counts()

    output = _call(
        module,
        mx.ones((1, 2, 2560), dtype=mx.bfloat16),
        mask=mx.ones((1, 2), dtype=mx.bool_),
        cache=cache,
    )
    after = qwen4_gdn.qwen4_gdn_call_counts()

    assert output.shape == (1, 2, 2560)
    assert after["generic_calls"] - before["generic_calls"] == 1
    assert after["fallback_input"] - before["fallback_input"] == 1
    assert after["native_calls"] == before["native_calls"]


@pytest.mark.parametrize("missing_dependency", [False, True])
def test_gdn_boundary_contract_failures_stay_generic(
    monkeypatch: pytest.MonkeyPatch,
    missing_dependency: bool,
) -> None:
    module = _module()
    cache = _Cache()
    monkeypatch.setattr(qwen4_gdn, "_qwen4_gated_delta_update", _generic_update)
    if missing_dependency:
        monkeypatch.setattr(qwen4_gdn, "_qwen4_gdn_native_contract", lambda *_args: True)
        monkeypatch.setattr(qwen4_gdn, "_qwen4_gdn_native_helpers", lambda: None)
    else:
        module.training = True
        monkeypatch.setattr(
            qwen4_gdn,
            "_qwen4_gdn_native_helpers",
            lambda: pytest.fail("training loaded the native GDN helpers"),
        )
    before = qwen4_gdn.qwen4_gdn_call_counts()

    output = _call(
        module,
        mx.ones((1, 1, 2560), dtype=mx.bfloat16),
        mask=mx.ones((1, 1), dtype=mx.bool_),
        cache=cache,
    )
    after = qwen4_gdn.qwen4_gdn_call_counts()

    assert output.shape == (1, 1, 2560)
    assert after["generic_calls"] - before["generic_calls"] == 1
    assert after["fallback_contract"] - before["fallback_contract"] == 1
    assert after["native_calls"] == before["native_calls"]


def test_gdn_boundary_input_contract_is_exact() -> None:
    module = SimpleNamespace(hidden_size=2560)

    assert qwen4_gdn._qwen4_gdn_native_input_supported(
        module,
        mx.zeros((1, 1, 2560), dtype=mx.bfloat16),
    )
    assert not qwen4_gdn._qwen4_gdn_native_input_supported(
        module,
        mx.zeros((1, 2, 2560), dtype=mx.bfloat16),
    )
    assert not qwen4_gdn._qwen4_gdn_native_input_supported(
        module,
        mx.zeros((1, 1, 2560), dtype=mx.float32),
    )
