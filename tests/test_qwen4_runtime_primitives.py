from __future__ import annotations

import mlx.core as mx
import numpy as np
from types import SimpleNamespace
from mlx_lm.models.cache import ArraysCache
import pytest

from moespresso.correctness.qwen4.architecture_reference import (
    gated_residual_read as reference_gated_residual_read,
)
from moespresso.correctness.qwen4.architecture_reference import (
    gated_residual_write as reference_gated_residual_write,
)
from moespresso.correctness.qwen4.architecture_reference import (
    grouped_zero_centered_rms_norm,
)
from moespresso.runtime.qwen4.primitives import (
    Qwen4GatedResidual,
    Qwen4RMSNorm,
    Qwen4SigmoidRMSNormGated,
    gated_residual_write,
)
from moespresso.runtime.qwen4.gdn import (
    Qwen4GDNAdapter,
    Qwen4GDNState,
    Qwen4GatedDeltaNet,
    _decay_multiplier,
    _l2_normalize,
)
import moespresso.runtime.qwen4.primitives as qwen4_primitives


def _array(values: np.ndarray) -> mx.array:
    return mx.array(values)


def test_runtime_grouped_zero_centered_rms_norm_matches_reference() -> None:
    values = np.array([[3.0, 4.0, 1.0, -2.0]], dtype=np.float32)
    weight = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    module = Qwen4RMSNorm(4, group_size=2)
    module.weight = _array(weight)

    got = module(_array(values))
    expected = grouped_zero_centered_rms_norm(values, weight, group_size=2)
    mx.eval(got)

    assert np.allclose(np.asarray(got), expected, rtol=0, atol=2e-7)


def test_runtime_gated_residual_read_and_write_match_reference() -> None:
    hyper_input = np.array([[3.0, 4.0, -2.0, 1.0]], dtype=np.float32)
    norm_weight = np.array([0.1, 0.2, -0.1, 0.05], dtype=np.float32)
    down_weight = np.array([[0.2, -0.1, 0.3, 0.4]], dtype=np.float32)
    up_weight = np.array([[0.5], [-0.3], [0.2], [0.1]], dtype=np.float32)
    inject_weight = np.array([[0.1, 0.2, -0.4, 0.3], [-0.2, 0.5, 0.1, -0.3]], dtype=np.float32)
    block_output = np.array([[0.25, -0.5]], dtype=np.float32)

    module = Qwen4GatedResidual(2, 2, 1)
    module.hc_norm.weight = _array(norm_weight)
    module.input_mix_weight_down.weight = _array(down_weight)
    module.input_mix_weight_up.weight = _array(up_weight)
    module.block_inject_weight.weight = _array(inject_weight)

    mixed, residual, injection = module(_array(hyper_input))
    got = gated_residual_write(residual, _array(block_output), injection)
    expected_read = reference_gated_residual_read(
        hyper_input,
        norm_weight,
        down_weight,
        up_weight,
        branch_count=2,
        hidden_size=2,
        injection_weight=inject_weight,
    )
    expected = reference_gated_residual_write(
        hyper_input,
        block_output,
        expected_read.injection_weights,
    )
    mx.eval(mixed, injection, got)

    assert np.allclose(np.asarray(mixed), expected_read.mixed_input, rtol=0, atol=2e-7)
    assert np.allclose(np.asarray(injection), expected_read.injection_weights, rtol=0, atol=2e-7)
    assert np.allclose(np.asarray(got), expected, rtol=0, atol=2e-7)


def test_runtime_gated_residual_uses_enabled_native_projection_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Qwen4GatedResidual(2, 2, 1)
    values = _array(np.array([[3.0, 4.0, -2.0, 1.0]], dtype=np.float32))
    expected_mixed = _array(np.array([[0.5, -0.25]], dtype=np.float32))
    expected_injection = _array(np.array([[0.75, 1.25]], dtype=np.float32))
    observed = {}

    def native_projection_read(target, normalized):
        observed["target"] = target
        observed["normalized"] = normalized
        return expected_mixed, expected_injection

    monkeypatch.setattr(
        qwen4_primitives,
        "_qwen4_hc_native_read_contract",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        qwen4_primitives,
        "_qwen4_hc_native_projection_contract",
        lambda target, normalized: target is module and normalized is values,
    )
    monkeypatch.setattr(
        qwen4_primitives,
        "_qwen4_hc_native_projection_read",
        native_projection_read,
    )
    before = qwen4_primitives.qwen4_hc_call_counts()

    mixed, residual, injection = module(values)
    mx.eval(observed["normalized"], mixed, injection)
    after = qwen4_primitives.qwen4_hc_call_counts()

    assert observed["target"] is module
    assert np.array_equal(np.asarray(mixed), np.asarray(expected_mixed))
    assert residual is values
    assert np.array_equal(np.asarray(injection), np.asarray(expected_injection))
    assert after["fused_projection_reads"] - before["fused_projection_reads"] == 1
    assert after["generic_reads"] == before["generic_reads"]


def test_runtime_gated_residual_native_projection_pair_falls_back_on_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qwen4_primitives,
        "_qwen4_hc_native_projection_contract",
        lambda _target, _normalized: False,
    )
    monkeypatch.setattr(
        qwen4_primitives,
        "_qwen4_hc_native_projection_read",
        lambda _target, _normalized: pytest.fail("ineligible call used the native projection pair"),
    )
    module = Qwen4GatedResidual(2, 2, 1)
    before = qwen4_primitives.qwen4_hc_call_counts()

    mixed, _residual, injection = module(_array(np.ones((1, 4), dtype=np.float32)))
    mx.eval(mixed, injection)
    after = qwen4_primitives.qwen4_hc_call_counts()

    assert after["fused_projection_reads"] == before["fused_projection_reads"]
    assert after["generic_reads"] - before["generic_reads"] == 1


def test_runtime_gated_residual_native_read_folds_pending_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Qwen4GatedResidual(2, 2, 1)
    values = _array(np.array([[3.0, 4.0, -2.0, 1.0]], dtype=np.float32))
    block_output = _array(np.array([[0.25, -0.5]], dtype=np.float32))
    pending_injection = _array(np.array([[0.75, 1.25]], dtype=np.float32))
    expected_mixed = _array(np.array([[0.5, -0.25]], dtype=np.float32))
    expected_updated = _array(np.array([[3.25, 3.5, -1.6875, 0.375]], dtype=np.float32))
    expected_injection = _array(np.array([[0.5, 1.5]], dtype=np.float32))
    observed = {}

    def native_read(target, residual, output, injection):
        observed["arguments"] = (target, residual, output, injection)
        return expected_mixed, expected_updated, expected_injection

    monkeypatch.setattr(
        qwen4_primitives,
        "_qwen4_hc_native_read_contract",
        lambda target, residual, output, injection: (
            target is module
            and residual is values
            and output is block_output
            and injection is pending_injection
        ),
    )
    monkeypatch.setattr(qwen4_primitives, "_qwen4_hc_native_read", native_read)
    before = qwen4_primitives.qwen4_hc_call_counts()

    mixed, updated, injection = module.read_with_pending(
        values,
        block_output,
        pending_injection,
    )
    after = qwen4_primitives.qwen4_hc_call_counts()

    target, residual, output, observed_injection = observed["arguments"]
    assert target is module
    assert residual is values
    assert output is block_output
    assert observed_injection is pending_injection
    assert mixed is expected_mixed
    assert updated is expected_updated
    assert injection is expected_injection
    assert after["fused_projection_reads"] - before["fused_projection_reads"] == 1
    assert after["native_norm_reads"] - before["native_norm_reads"] == 1
    assert after["native_folded_writes"] - before["native_folded_writes"] == 1


def test_runtime_gated_residual_pending_read_falls_back_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Qwen4GatedResidual(2, 2, 1)
    values = _array(np.array([[3.0, 4.0, -2.0, 1.0]], dtype=np.float32))
    block_output = _array(np.array([[0.25, -0.5]], dtype=np.float32))
    pending_injection = _array(np.array([[0.75, 1.25]], dtype=np.float32))
    monkeypatch.setattr(
        qwen4_primitives,
        "_qwen4_hc_native_read_contract",
        lambda *_args: False,
    )

    expected_updated = gated_residual_write(values, block_output, pending_injection)
    expected = module(expected_updated)
    got = module.read_with_pending(values, block_output, pending_injection)
    mx.eval(*expected, *got)

    for actual, reference in zip(got, expected, strict=True):
        assert np.array_equal(np.asarray(actual), np.asarray(reference))


def test_runtime_sigmoid_gated_norm_does_not_apply_silu_gate() -> None:
    values = _array(np.array([[[1.0, -2.0]]], dtype=np.float32))
    gate = _array(np.array([[[0.5, -1.0]]], dtype=np.float32))
    module = Qwen4SigmoidRMSNormGated(2)
    module.weight = _array(np.array([1.25, 0.75], dtype=np.float32))

    got = module(values, gate)
    normalized = np.asarray(values) / np.sqrt(
        np.mean(np.square(np.asarray(values)), axis=-1, keepdims=True) + 1e-6
    )
    expected = normalized * np.array([1.25, 0.75], dtype=np.float32)
    expected *= 1.0 / (1.0 + np.exp(-np.asarray(gate)))
    mx.eval(got)

    assert np.allclose(np.asarray(got), expected, rtol=0, atol=2e-7)


def _tiny_gdn() -> Qwen4GatedDeltaNet:
    module = Qwen4GatedDeltaNet(
        SimpleNamespace(
            hidden_size=2,
            linear_num_value_heads=1,
            linear_num_key_heads=1,
            linear_key_head_dim=2,
            linear_value_head_dim=1,
            linear_conv_kernel_dim=1,
            rms_norm_eps=1e-6,
        )
    )
    module.in_proj_qkv.weight = _array(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    module.in_proj_z.weight = _array(np.array([[0.5, 0.0]], dtype=np.float32))
    module.in_proj_b.weight = _array(np.zeros((1, 2), dtype=np.float32))
    module.in_proj_a.weight = _array(np.zeros((1, 2), dtype=np.float32))
    module.conv1d.weight = _array(np.ones((5, 1, 1), dtype=np.float32))
    module.A_log = _array(np.zeros(1, dtype=np.float32))
    module.dt_bias = _array(np.zeros(1, dtype=np.float32))
    module.norm.weight = _array(np.ones(1, dtype=np.float32))
    module.out_proj.weight = _array(np.array([[1.0], [2.0]], dtype=np.float32))
    module.train()
    return module


def test_runtime_gdn_uses_released_sigmoid_epilogue() -> None:
    module = _tiny_gdn()

    got = module(_array(np.array([[[1.0, 2.0]]], dtype=np.float32)))
    gate = 1.0 / (1.0 + np.exp(-0.5))
    expected = np.array([[[gate, 2 * gate]]], dtype=np.float32)
    mx.eval(got)

    assert np.allclose(np.asarray(got), expected, rtol=0, atol=2e-6)


def test_runtime_gdn_masks_every_projection_input() -> None:
    module = _tiny_gdn()

    got = module(
        _array(np.array([[[1.0, 2.0]]], dtype=np.float32)),
        mask=_array(np.array([[False]])),
    )
    mx.eval(got)

    assert np.array_equal(np.asarray(got), np.zeros((1, 1, 2), dtype=np.float32))


def test_runtime_gdn_invalid_physical_row_decays_live_recurrent_state() -> None:
    inputs = np.array([[[0.75, -0.5], [4.0, -3.0], [0.2, 0.3]]], dtype=np.float32)
    valid = np.array([[True, False, True]])
    masked_module = _cached_gdn()
    masked_cache = ArraysCache(size=2)
    got = masked_module(_array(inputs), mask=_array(valid), cache=masked_cache)

    explicit_module = _cached_gdn()
    explicit_cache = ArraysCache(size=2)
    expected = explicit_module(
        _array(np.where(valid[..., None], inputs, 0)),
        cache=explicit_cache,
    )

    mx.eval(got, expected, *masked_cache.state, *explicit_cache.state)

    assert np.allclose(np.asarray(got), np.asarray(expected), rtol=0, atol=2e-7)
    assert not np.array_equal(np.asarray(got)[:, 1], np.zeros((1, 2), dtype=np.float32))
    for actual_state, expected_state in zip(
        masked_cache.state,
        explicit_cache.state,
        strict=True,
    ):
        assert np.allclose(np.asarray(actual_state), np.asarray(expected_state), rtol=0, atol=2e-7)


def test_runtime_gdn_l2_epsilon_is_not_scaled_by_head_width() -> None:
    values = np.array([[[[1.0e-4, -2.0e-4, 3.0e-4, -4.0e-4]]]], dtype=np.float32)

    got = _l2_normalize(_array(values))
    expected = values / np.sqrt(np.sum(np.square(values), axis=-1, keepdims=True) + 1.0e-6)
    rms_based = values / np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True) + 1.0e-6)
    mx.eval(got)

    assert np.allclose(np.asarray(got), expected, rtol=0, atol=2e-7)
    assert not np.allclose(np.asarray(got), rms_based, rtol=0, atol=1e-3)


def test_runtime_gdn_decay_casts_logits_and_bias_before_addition() -> None:
    logits = _array(np.array([[[-4.21875]]], dtype=np.float16))
    a_log = _array(np.array([0.375], dtype=np.float32))
    bias = _array(np.array([0.00391], dtype=np.float16))

    got = _decay_multiplier(logits, a_log, bias)
    logits32 = np.asarray(logits).astype(np.float32)
    bias32 = np.asarray(bias).astype(np.float32)
    softplus = np.logaddexp(np.float32(0), logits32 + bias32)
    expected = np.exp(-np.exp(np.asarray(a_log)) * softplus)
    mx.eval(got)

    assert got.dtype == mx.float32
    assert np.allclose(np.asarray(got), expected, rtol=0, atol=2e-7)


def _cached_gdn() -> Qwen4GatedDeltaNet:
    rng = np.random.default_rng(41)
    module = Qwen4GatedDeltaNet(
        SimpleNamespace(
            hidden_size=2,
            linear_num_value_heads=1,
            linear_num_key_heads=1,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
        )
    )
    module.in_proj_qkv.weight = _array(rng.normal(scale=0.2, size=(6, 2)).astype(np.float32))
    module.in_proj_z.weight = _array(rng.normal(scale=0.2, size=(2, 2)).astype(np.float32))
    module.in_proj_b.weight = _array(rng.normal(scale=0.2, size=(1, 2)).astype(np.float32))
    module.in_proj_a.weight = _array(rng.normal(scale=0.2, size=(1, 2)).astype(np.float32))
    module.conv1d.weight = _array(rng.normal(scale=0.2, size=(6, 4, 1)).astype(np.float32))
    module.A_log = _array(np.array([0.2], dtype=np.float32))
    module.dt_bias = _array(np.array([-0.1], dtype=np.float32))
    module.norm.weight = _array(np.array([0.8, 1.2], dtype=np.float32))
    module.out_proj.weight = _array(rng.normal(scale=0.2, size=(2, 2)).astype(np.float32))
    module.train()
    return module


def test_runtime_gdn_width_four_cache_matches_full_chunks_and_tokens() -> None:
    rng = np.random.default_rng(43)
    inputs = _array(rng.normal(size=(1, 7, 2)).astype(np.float32))
    module = _cached_gdn()

    full_cache = ArraysCache(size=2)
    full = module(inputs, cache=full_cache)

    chunk_cache = ArraysCache(size=2)
    chunks = [
        module(inputs[:, :2], cache=chunk_cache),
        module(inputs[:, 2:6], cache=chunk_cache),
        module(inputs[:, 6:], cache=chunk_cache),
    ]
    chunked = mx.concatenate(chunks, axis=1)

    token_cache = ArraysCache(size=2)
    tokenwise = mx.concatenate(
        [module(inputs[:, index : index + 1], cache=token_cache) for index in range(7)],
        axis=1,
    )
    mx.eval(
        full,
        chunked,
        tokenwise,
        *full_cache.state,
        *chunk_cache.state,
        *token_cache.state,
    )

    assert np.allclose(np.asarray(chunked), np.asarray(full), rtol=0, atol=2e-6)
    assert np.allclose(np.asarray(tokenwise), np.asarray(full), rtol=0, atol=2e-6)
    for full_state, chunk_state, token_state in zip(
        full_cache.state,
        chunk_cache.state,
        token_cache.state,
        strict=True,
    ):
        assert np.allclose(np.asarray(chunk_state), np.asarray(full_state), rtol=0, atol=2e-6)
        assert np.allclose(np.asarray(token_state), np.asarray(full_state), rtol=0, atol=2e-6)


def _adapter_call(
    adapter: Qwen4GDNAdapter,
    state: Qwen4GDNState | None,
    inputs: mx.array,
    valid: mx.array,
):
    base = 0 if state is None else state.offset
    batch, tokens, _ = inputs.shape
    visible = mx.concatenate(
        [mx.ones((batch, base), dtype=mx.bool_), valid],
        axis=1,
    )
    positions = mx.broadcast_to(
        mx.arange(base, base + tokens, dtype=mx.int64)[None, None],
        (3, batch, tokens),
    )
    return adapter(
        inputs,
        valid_tokens=valid,
        visible_history=visible,
        position_ids=positions,
        state=state,
    )


def test_runtime_gdn_adapter_matches_full_chunks_and_tokens() -> None:
    rng = np.random.default_rng(47)
    inputs = _array(rng.normal(size=(1, 7, 2)).astype(np.float32))
    valid = _array(np.array([[True, False, True, True, False, True, True]]))

    full_adapter = Qwen4GDNAdapter(_cached_gdn())
    full = _adapter_call(full_adapter, None, inputs, valid)

    chunk_adapter = Qwen4GDNAdapter(_cached_gdn())
    chunk_state = None
    chunk_outputs = []
    for start, end in ((0, 2), (2, 6), (6, 7)):
        result = _adapter_call(
            chunk_adapter,
            chunk_state,
            inputs[:, start:end],
            valid[:, start:end],
        )
        chunk_outputs.append(result.output)
        chunk_state = result.state

    token_adapter = Qwen4GDNAdapter(_cached_gdn())
    token_state = None
    token_outputs = []
    for position in range(7):
        result = _adapter_call(
            token_adapter,
            token_state,
            inputs[:, position : position + 1],
            valid[:, position : position + 1],
        )
        token_outputs.append(result.output)
        token_state = result.state

    chunked = mx.concatenate(chunk_outputs, axis=1)
    tokenwise = mx.concatenate(token_outputs, axis=1)
    mx.eval(
        full.output,
        chunked,
        tokenwise,
        full.state.conv_state,
        full.state.recurrent_state,
        chunk_state.conv_state,
        chunk_state.recurrent_state,
        token_state.conv_state,
        token_state.recurrent_state,
    )

    assert np.allclose(np.asarray(chunked), np.asarray(full.output), rtol=0, atol=2e-6)
    assert np.allclose(np.asarray(tokenwise), np.asarray(full.output), rtol=0, atol=2e-6)
    assert full.frontier == chunk_state.offset == token_state.offset == 7
    assert np.allclose(
        np.asarray(chunk_state.conv_state),
        np.asarray(full.state.conv_state),
        rtol=0,
        atol=2e-6,
    )
    assert np.allclose(
        np.asarray(token_state.conv_state),
        np.asarray(full.state.conv_state),
        rtol=0,
        atol=2e-6,
    )
    assert np.allclose(
        np.asarray(chunk_state.recurrent_state),
        np.asarray(full.state.recurrent_state),
        rtol=0,
        atol=2e-6,
    )
    assert np.allclose(
        np.asarray(token_state.recurrent_state),
        np.asarray(full.state.recurrent_state),
        rtol=0,
        atol=2e-6,
    )


def test_runtime_gdn_adapter_state_contract_and_copy_isolation() -> None:
    adapter = Qwen4GDNAdapter(_cached_gdn())
    empty_positions = mx.zeros((3, 1, 0), dtype=mx.int64)
    adapter.validate_state(None, expected_frontier=0, position_history=empty_positions)
    with np.testing.assert_raises_regex(ValueError, "empty at frontier zero"):
        adapter.validate_state(
            Qwen4GDNState(
                conv_state=mx.zeros((1, 3, 6)),
                recurrent_state=mx.zeros((1, 1, 2, 2)),
                offset=0,
            ),
            expected_frontier=0,
            position_history=empty_positions,
        )

    result = _adapter_call(
        adapter,
        None,
        _array(np.array([[[0.25, -0.75]]], dtype=np.float32)),
        _array(np.array([[True]])),
    )
    state = result.state
    fork = adapter.fork_state(state)
    snapshot = adapter.snapshot_state(state)
    fork.conv_state[0, 0, 0] = np.float32(99)
    fork.recurrent_state[0, 0, 0, 0] = np.float32(77)
    mx.eval(
        state.conv_state,
        state.recurrent_state,
        fork.conv_state,
        fork.recurrent_state,
        snapshot.conv_state,
        snapshot.recurrent_state,
    )

    assert not np.array_equal(np.asarray(fork.conv_state), np.asarray(state.conv_state))
    assert not np.array_equal(
        np.asarray(fork.recurrent_state),
        np.asarray(state.recurrent_state),
    )
    assert np.array_equal(np.asarray(snapshot.conv_state), np.asarray(state.conv_state))
    assert np.array_equal(
        np.asarray(snapshot.recurrent_state),
        np.asarray(state.recurrent_state),
    )
    live_positions = mx.zeros((3, 1, 1), dtype=mx.int64)
    with np.testing.assert_raises_regex(ValueError, "schema or recurrent layout"):
        adapter.validate_state(
            Qwen4GDNState(
                conv_state=state.conv_state,
                recurrent_state=state.recurrent_state,
                offset=1,
                recurrent_layout="batch-head-key-value",
            ),
            expected_frontier=1,
            position_history=live_positions,
        )
