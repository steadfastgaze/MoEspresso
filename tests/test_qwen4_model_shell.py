from __future__ import annotations

from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

import mlx.core as mx
import numpy as np
import pytest

import moespresso.runtime.qwen4.model as model_module
from moespresso.runtime.qwen4.model import (
    Qwen4CompositeState,
    Qwen4DecoderLayer,
    Qwen4LayerState,
    Qwen4MixerOutput,
    Qwen4TextModelShell,
    _Qwen4TrustedMaskCertificate,
)
from moespresso.runtime.qwen4.gdn import (
    Qwen4GDNAdapter,
    Qwen4GDNState,
    Qwen4GatedDeltaNet,
)
from moespresso.runtime.qwen4.ple import (
    Qwen4PLEOutput,
    Qwen4PLEState,
)
from moespresso.runtime.qwen4.qsa import Qwen4QSAAdapter, Qwen4SparseAttention


class _Embedding:
    def __call__(self, input_ids: mx.array) -> mx.array:
        values = input_ids.astype(mx.float32)
        return mx.stack([values, values + 1], axis=-1)


class _Residual:
    def __init__(self, events: list[str], label: str, bias: float) -> None:
        self.events = events
        self.label = label
        self.bias = bias

    def __call__(self, hidden_states: mx.array):
        self.events.append(self.label)
        streams = hidden_states.reshape(*hidden_states.shape[:-1], 2, 2)
        mixed = mx.sum(streams, axis=-2) + self.bias
        injection = mx.broadcast_to(
            mx.array([0.25, 0.5], dtype=hidden_states.dtype),
            (*hidden_states.shape[:-1], 2),
        )
        return mixed, hidden_states, injection


class _Mixer:
    def __init__(
        self,
        events: list[str],
        label: str,
        scale: float,
        bias: float,
    ) -> None:
        self.events = events
        self.label = label
        self.scale = scale
        self.bias = bias
        self.visible_histories: list[mx.array] = []
        self.position_ids: list[mx.array] = []

    def fork_state(self, state):
        return state

    def snapshot_state(self, state):
        return state

    def validate_state(
        self,
        state,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        del position_history
        expected = None if expected_frontier == 0 else expected_frontier
        if state != expected:
            raise ValueError("synthetic mixer state is stale")

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state,
    ) -> Qwen4MixerOutput:
        self.events.append(self.label)
        self.visible_histories.append(visible_history)
        self.position_ids.append(position_ids)
        token_count = hidden_states.shape[1]
        step = 0 if state is None else state
        assert visible_history.shape[1] == step + token_count
        assert position_ids.shape == (3, hidden_states.shape[0], token_count)
        offsets = mx.arange(step, step + token_count, dtype=hidden_states.dtype)
        output = mx.where(
            valid_tokens[..., None],
            hidden_states * self.scale + self.bias + 0.01 * offsets[None, :, None],
            0,
        )
        return Qwen4MixerOutput(
            output=output,
            state=step + token_count,
            frontier=step + token_count,
        )


class _TrustedMixer(_Mixer):
    def __init__(
        self,
        events: list[str],
        label: str,
        scale: float,
        bias: float,
    ) -> None:
        super().__init__(events, label, scale, bias)
        self.strict_frontiers: list[int] = []
        self.trusted_frontiers: list[int] = []
        self.trusted_steps = 0

    def validate_state(
        self,
        state,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        self.strict_frontiers.append(expected_frontier)
        super().validate_state(
            state,
            expected_frontier=expected_frontier,
            position_history=position_history,
        )

    def validate_state_trusted(
        self,
        state,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        self.trusted_frontiers.append(expected_frontier)
        super().validate_state(
            state,
            expected_frontier=expected_frontier,
            position_history=position_history,
        )

    def step_trusted(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state,
    ) -> Qwen4MixerOutput:
        self.trusted_steps += 1
        return super().__call__(
            hidden_states,
            valid_tokens=valid_tokens,
            visible_history=visible_history,
            position_ids=position_ids,
            state=state,
        )


class _BatchedFiniteMixer(_TrustedMixer):
    supports_trusted_mask_certificate = True
    supports_batched_append_finite = True

    def __init__(self, label: str, *, finite: bool = True) -> None:
        super().__init__([], label, 0.6, 0.15)
        self.finite = finite
        self.prepared = []
        self.certificates: list[_Qwen4TrustedMaskCertificate] = []

    def _prepare_trusted_undo(self, state, *, new_tokens, capability):
        assert capability is model_module._QWEN4_BATCHED_UNDO_CAPABILITY
        reservation = SimpleNamespace(
            state=state,
            new_tokens=new_tokens,
            array=mx.array([state], dtype=mx.int32) + 0,
            evaluated=False,
        )
        self.prepared.append(reservation)
        return reservation

    def _trusted_undo_arrays(self, reservation, *, capability):
        assert capability is model_module._QWEN4_BATCHED_UNDO_CAPABILITY
        return (reservation.array,)

    def _mark_trusted_undo_evaluated(self, reservation, *, capability):
        assert capability is model_module._QWEN4_BATCHED_UNDO_CAPABILITY
        reservation.evaluated = True

    def _step_trusted_certified(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state,
        mask_certificate: _Qwen4TrustedMaskCertificate,
    ) -> Qwen4MixerOutput:
        self.certificates.append(mask_certificate)
        result = self.step_trusted(
            hidden_states,
            valid_tokens=valid_tokens,
            visible_history=visible_history,
            position_ids=position_ids,
            state=state,
        )
        append_finite_batch = mask_certificate.append_finite_batch
        if append_finite_batch is not None:
            append_finite_batch.register(
                mixer=self,
                source_state=state,
                result_state=result.state,
                predicate=mx.array(self.finite, dtype=mx.bool_),
                capability=model_module._QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
            )
        return result


class _MutableMixer(_Mixer):
    def fork_state(self, state):
        if state is None:
            return None
        return {
            "offset": state["offset"],
            "buffer": mx.array(np.asarray(state["buffer"]).copy()),
        }

    def snapshot_state(self, state):
        return self.fork_state(state)

    def validate_state(
        self,
        state,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        del position_history
        if expected_frontier == 0:
            if state is not None:
                raise ValueError("mutable mixer state exists at frontier zero")
            return
        if state is None or state["offset"] != expected_frontier:
            raise ValueError("mutable mixer state is stale")

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state,
    ) -> Qwen4MixerOutput:
        del visible_history, position_ids
        token_count = hidden_states.shape[1]
        if state is None:
            state = {"offset": 0, "buffer": mx.zeros((1,), dtype=mx.int32)}
        state["buffer"][0] = state["buffer"][0] + token_count
        state["offset"] += token_count
        output = mx.where(valid_tokens[..., None], hidden_states * 0.1, 0)
        return Qwen4MixerOutput(
            output=output,
            state=state,
            frontier=state["offset"],
        )


class _LazyMutableMixer(_MutableMixer):
    def fork_state(self, state):
        if state is None:
            return None
        return {
            "offset": state["offset"],
            "buffer": state["buffer"] + mx.zeros_like(state["buffer"]),
        }

    def snapshot_state(self, state):
        return {
            "offset": state["offset"],
            "buffer": state["buffer"] + mx.zeros_like(state["buffer"]),
        }

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state,
    ) -> Qwen4MixerOutput:
        del visible_history, position_ids
        token_count = hidden_states.shape[1]
        if state is None:
            state = {"offset": 0, "buffer": mx.zeros((1,), dtype=mx.float32)}
        state["buffer"][0] = state["buffer"][0] + token_count
        state["offset"] += token_count
        output = mx.where(
            valid_tokens[..., None],
            hidden_states * 0.1 + state["buffer"],
            0,
        )
        return Qwen4MixerOutput(
            output=output,
            state=state,
            frontier=state["offset"],
        )


class _MLP:
    def __init__(
        self,
        events: list[str],
        label: str,
        scale: float,
        bias: float,
    ) -> None:
        self.events = events
        self.label = label
        self.scale = scale
        self.bias = bias

    def __call__(self, hidden_states: mx.array) -> mx.array:
        self.events.append(self.label)
        return hidden_states * self.scale + self.bias


class _CompiledGDNMixer:
    def validate_state(
        self,
        state: Qwen4GDNState,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        del position_history
        assert isinstance(state, Qwen4GDNState)
        assert state.offset == expected_frontier


class _PLE:
    def __init__(self, events: list[str], bias: float) -> None:
        self.events = events
        self.bias = bias

    def validate_state(
        self,
        state: Qwen4PLEState | None,
        *,
        expected_frontier: int,
        batch_size: int,
    ) -> None:
        if expected_frontier == 0:
            if state is not None:
                raise ValueError("PLE state must be empty at frontier zero")
            return
        if state is None or state.offset != expected_frontier:
            raise ValueError("PLE state is off the public frontier")
        if state.token_context.shape != (batch_size, 2):
            raise ValueError("PLE token context has an invalid shape")
        if state.conv_state.shape != (batch_size, 1, 4):
            raise ValueError("PLE convolution state has an invalid shape")

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        *,
        state: Qwen4PLEState | None = None,
        valid_tokens: mx.array | None = None,
    ) -> Qwen4PLEOutput:
        self.events.append("L1 PLE")
        offset = 0 if state is None else state.offset
        values = (
            self.bias
            + 0.02
            * mx.arange(
                offset,
                offset + input_ids.shape[1],
                dtype=hidden_states.dtype,
            )[None, :, None]
        )
        output = mx.where(
            valid_tokens[..., None],
            mx.broadcast_to(values, hidden_states.shape),
            0,
        )
        next_state = Qwen4PLEState(
            token_context=mx.broadcast_to(input_ids[:, -1:], (input_ids.shape[0], 2)),
            conv_state=mx.full(
                (input_ids.shape[0], 1, 4),
                offset + input_ids.shape[1],
                mx.float32,
            ),
            offset=offset + input_ids.shape[1],
        )
        return Qwen4PLEOutput(output=output, state=next_state)


class _FinalResidual:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self, hidden_states: mx.array) -> mx.array:
        self.events.append("final read")
        return mx.mean(hidden_states.reshape(*hidden_states.shape[:-1], 2, 2), axis=-2)


class _IdentityHead:
    def __call__(self, hidden_states: mx.array) -> mx.array:
        return hidden_states


def _model(events: list[str]) -> Qwen4TextModelShell:
    layers = []
    kinds = ("gdn", "gdn", "gdn", "qsa")
    for index, kind in enumerate(kinds):
        layers.append(
            Qwen4DecoderLayer(
                mixer_kind=kind,
                attention_residual=_Residual(
                    events,
                    f"L{index} attention read",
                    0.1 * (index + 1),
                ),
                mixer=_Mixer(
                    events,
                    f"L{index} {kind.upper()}",
                    0.3 + 0.1 * index,
                    0.05 * index,
                ),
                mlp_residual=_Residual(
                    events,
                    f"L{index} MLP read",
                    -0.05 * index,
                ),
                mlp=_MLP(
                    events,
                    f"L{index} MoE",
                    0.2 + 0.05 * index,
                    -0.03 * index,
                ),
                ple=_PLE(events, 0.4) if index == 1 else None,
            )
        )
    return Qwen4TextModelShell(
        cache_identity="qwen4-shell-test",
        embedding=_Embedding(),
        layers=tuple(layers),
        final_residual=_FinalResidual(events),
        lm_head=_IdentityHead(),
        hidden_size=2,
        branch_count=2,
    )


def _all_gdn_model() -> Qwen4TextModelShell:
    model = _model([])
    model.layers = tuple(
        replace(layer, mixer_kind="gdn") if layer.mixer_kind == "qsa" else layer
        for layer in model.layers
    )
    return model


def _model_with_real_qsa() -> Qwen4TextModelShell:
    events: list[str] = []
    module = Qwen4SparseAttention(
        hidden_size=2,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=2,
        index_query_heads=1,
        index_kv_heads=1,
        index_head_dim=2,
        token_budget=2,
        compress_ratio=2,
        rotary_dim=2,
        rope_base=10_000.0,
        mrope_section=(1, 0, 0),
    )
    layer = Qwen4DecoderLayer(
        mixer_kind="qsa",
        attention_residual=_Residual(events, "QSA attention read", 0.1),
        mixer=Qwen4QSAAdapter(module),
        mlp_residual=_Residual(events, "QSA MLP read", 0.0),
        mlp=_MLP(events, "QSA MLP", 0.2, 0.0),
    )
    return Qwen4TextModelShell(
        cache_identity="qwen4-real-qsa-shell-test",
        embedding=_Embedding(),
        layers=(layer,),
        final_residual=_FinalResidual(events),
        lm_head=_IdentityHead(),
        hidden_size=2,
        branch_count=2,
    )


def _model_with_real_gdn() -> Qwen4TextModelShell:
    events: list[str] = []
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
    module.in_proj_qkv.weight = mx.array(rng.normal(scale=0.2, size=(6, 2)).astype(np.float32))
    module.in_proj_z.weight = mx.array(rng.normal(scale=0.2, size=(2, 2)).astype(np.float32))
    module.in_proj_b.weight = mx.array(rng.normal(scale=0.2, size=(1, 2)).astype(np.float32))
    module.in_proj_a.weight = mx.array(rng.normal(scale=0.2, size=(1, 2)).astype(np.float32))
    module.conv1d.weight = mx.array(rng.normal(scale=0.2, size=(6, 4, 1)).astype(np.float32))
    module.A_log = mx.array([0.2], dtype=mx.float32)
    module.dt_bias = mx.array([-0.1], dtype=mx.float32)
    module.norm.weight = mx.array([0.8, 1.2], dtype=mx.float32)
    module.out_proj.weight = mx.array(rng.normal(scale=0.2, size=(2, 2)).astype(np.float32))
    module.train()
    layer = Qwen4DecoderLayer(
        mixer_kind="gdn",
        attention_residual=_Residual(events, "GDN attention read", 0.1),
        mixer=Qwen4GDNAdapter(module),
        mlp_residual=_Residual(events, "GDN MLP read", 0.0),
        mlp=_MLP(events, "GDN MLP", 0.2, 0.0),
    )
    return Qwen4TextModelShell(
        cache_identity="qwen4-real-gdn-shell-test",
        embedding=_Embedding(),
        layers=(layer,),
        final_residual=_FinalResidual(events),
        lm_head=_IdentityHead(),
        hidden_size=2,
        branch_count=2,
    )


def test_cache_routing_is_decode_only_even_for_one_token_prefill() -> None:
    model = _model([])
    model._cache_routing_enabled = True
    calls = []
    for layer in model.layers:
        original = layer.mlp

        def mlp(value, *, cache_routing=False, original=original):
            calls.append(cache_routing)
            return original(value)

        layer.mlp = mlp
    coordinator = model.new_coordinator(1)
    try:
        coordinator.forward_chunk(mx.array([[1]], dtype=mx.int64))
        assert calls and not any(calls)
        calls.clear()
        candidate = coordinator.propose(mx.array([[2]], dtype=mx.int64))
        assert calls and all(calls)
        coordinator.commit(candidate, 1)
        calls.clear()
        coordinator.forward_chunk(mx.array([[3]], dtype=mx.int64))
        assert calls and not any(calls)
        with pytest.raises(ValueError, match="single-token"):
            coordinator.propose(mx.array([[4, 5]], dtype=mx.int64))
    finally:
        coordinator.close()
        model.close()


def test_qwen4_shell_exposes_injected_modules_to_mlx_hydration() -> None:
    model = _model_with_real_qsa()

    module_names = {name for name, _module in model.named_modules()}
    assert "layers.0.mixer" in module_names
    assert "layers.0.mixer.module" in module_names
    assert "layers.0.mixer.module.indexer.index_qk_proj" in module_names
    assert "layers.0.mixer.module.q_proj" in module_names
    assert "layers" in model.parameters()

    model.layers = tuple(model.layers)
    assert isinstance(model.layers, list)
    assert "layers.0.mixer.module.q_proj" in {name for name, _module in model.named_modules()}


def _reference_token(token: int, step: int) -> np.ndarray:
    hidden = np.tile(np.array([token, token + 1], dtype=np.float32), 2)
    for layer in range(4):
        if layer == 1:
            hidden = hidden + np.float32(0.4 + 0.02 * step)
        streams = hidden.reshape(2, 2)
        mixed = streams.sum(axis=0) + np.float32(0.1 * (layer + 1))
        block = mixed * np.float32(0.3 + 0.1 * layer)
        block += np.float32(0.05 * layer + 0.01 * step)
        hidden = hidden + (block[None] * np.array([[0.25], [0.5]], dtype=np.float32)).reshape(-1)
        streams = hidden.reshape(2, 2)
        mixed = streams.sum(axis=0) - np.float32(0.05 * layer)
        block = mixed * np.float32(0.2 + 0.05 * layer) - np.float32(0.03 * layer)
        hidden = hidden + (block[None] * np.array([[0.25], [0.5]], dtype=np.float32)).reshape(-1)
    return hidden.reshape(2, 2).mean(axis=0)


def _state_arrays(state: Qwen4CompositeState) -> list[np.ndarray]:
    arrays = [np.asarray(state.valid_history), np.asarray(state.position_history)]
    for layer in state.layers:
        if layer.ple_state is not None:
            arrays.append(np.asarray(layer.ple_state.token_context))
            arrays.append(np.asarray(layer.ple_state.conv_state))
    return arrays


def test_qwen4_model_close_owns_provider_and_composite_state_lifetimes() -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)

    class _Resources:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    resources = _Resources()
    object.__setattr__(model, "_moespresso_qwen4_runtime_resources", resources)

    model.close()
    model.close()

    assert resources.closed
    with pytest.raises(RuntimeError, match="coordinator is closed"):
        _ = coordinator.state
    with pytest.raises(RuntimeError, match="model is closed"):
        model.new_state(1)


def test_qwen4_model_close_preflights_resources_before_coordinator_state() -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)

    class _Resources:
        busy = True
        closed = False

        def assert_quiescent(self):
            if self.busy:
                raise RuntimeError("resources are busy")

        def close(self):
            self.closed = True

    resources = _Resources()
    object.__setattr__(model, "_moespresso_qwen4_runtime_resources", resources)

    with pytest.raises(RuntimeError, match="resources are busy"):
        model.close()
    assert coordinator.state.frontier == 0
    assert model.new_state(1).frontier == 0

    resources.busy = False
    model.close()
    assert resources.closed
    with pytest.raises(RuntimeError, match="coordinator is closed"):
        _ = coordinator.state


def test_qwen4_model_close_releases_compiled_mtp_core_after_quiescence() -> None:
    model = _model([])

    class _Resources:
        busy = True

        def assert_quiescent(self):
            if self.busy:
                raise RuntimeError("resources are busy")

        def close(self):
            pass

    class _Core:
        def __init__(self, owner):
            self.model = owner

    resources = _Resources()
    core = _Core(model)
    reference = weakref.ref(core)
    object.__setattr__(model, "_moespresso_qwen4_runtime_resources", resources)
    object.__setattr__(model, "_moespresso_qwen4_mtp_compiled_core", core)
    del core

    with pytest.raises(RuntimeError, match="resources are busy"):
        model.close()
    assert model._moespresso_qwen4_mtp_compiled_core is reference()

    resources.busy = False
    model.close()
    gc.collect()
    assert model._moespresso_qwen4_mtp_compiled_core is None
    assert reference() is None


def test_qwen4_shell_pins_released_four_layer_order_and_numeric_dataflow() -> None:
    events: list[str] = []
    model = _model(events)
    candidate = model.propose(model.new_coordinator(1), mx.array([[2]], dtype=mx.int64))
    mx.eval(candidate.logits)

    assert events == [
        "L0 attention read",
        "L0 GDN",
        "L0 MLP read",
        "L0 MoE",
        "L1 PLE",
        "L1 attention read",
        "L1 GDN",
        "L1 MLP read",
        "L1 MoE",
        "L2 attention read",
        "L2 GDN",
        "L2 MLP read",
        "L2 MoE",
        "L3 attention read",
        "L3 QSA",
        "L3 MLP read",
        "L3 MoE",
        "final read",
    ]
    assert np.allclose(np.asarray(candidate.logits)[0, 0], _reference_token(2, 0), atol=2e-6)


def test_qwen4_forward_chunk_matches_direct_tokenwise_execution() -> None:
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int64)
    chunk_events: list[str] = []
    chunk_model = _model(chunk_events)
    initial = chunk_model.new_state(1)

    chunk_logits, chunk_state = chunk_model.forward_chunk(initial, tokens)

    token_model = _model([])
    token_state = token_model.new_state(1)
    token_logits = []
    for position in range(tokens.shape[1]):
        logits, token_state = token_model.forward_chunk(
            token_state,
            tokens[:, position : position + 1],
        )
        token_logits.append(logits)
    token_logits = mx.concatenate(token_logits, axis=1)
    mx.eval(chunk_logits, token_logits)

    assert chunk_events == [
        "L0 attention read",
        "L0 GDN",
        "L0 MLP read",
        "L0 MoE",
        "L1 PLE",
        "L1 attention read",
        "L1 GDN",
        "L1 MLP read",
        "L1 MoE",
        "L2 attention read",
        "L2 GDN",
        "L2 MLP read",
        "L2 MoE",
        "L3 attention read",
        "L3 QSA",
        "L3 MLP read",
        "L3 MoE",
        "final read",
    ]
    assert np.allclose(np.asarray(chunk_logits), np.asarray(token_logits), rtol=0, atol=2e-6)
    assert initial.frontier == initial.revision == 0
    assert chunk_state.frontier == token_state.frontier == tokens.shape[1]
    assert chunk_state.revision == 1
    assert token_state.revision == tokens.shape[1]
    assert [layer.mixer_state for layer in chunk_state.layers] == [tokens.shape[1]] * 4
    for expected, actual in zip(
        _state_arrays(token_state),
        _state_arrays(chunk_state),
        strict=True,
    ):
        assert np.array_equal(actual, expected)


def test_qwen4_forward_chunk_matches_real_qsa_continuation_state() -> None:
    tokens = mx.array([[3, 4, 5, 6]], dtype=mx.int64)
    valid = mx.array([[True, False, True, True]])
    positions = mx.array(
        [
            [[7, 8, 9, 10]],
            [[3, 3, 4, 4]],
            [[11, 12, 12, 13]],
        ],
        dtype=mx.int64,
    )
    model = _model_with_real_qsa()
    initial = model.new_state(1)
    full_logits, full_state = model.forward_chunk(
        initial,
        tokens,
        valid_tokens=valid,
        position_ids=positions,
    )

    split_a, split_state = model.forward_chunk(
        initial,
        tokens[:, :3],
        valid_tokens=valid[:, :3],
        position_ids=positions[:, :, :3],
    )
    split_b, split_state = model.forward_chunk(
        split_state,
        tokens[:, 3:],
        valid_tokens=valid[:, 3:],
        position_ids=positions[:, :, 3:],
    )
    split_logits = mx.concatenate([split_a, split_b], axis=1)
    full_qsa = full_state.layers[0].mixer_state
    split_qsa = split_state.layers[0].mixer_state
    mx.eval(
        full_logits,
        split_logits,
        full_qsa.keys,
        full_qsa.values,
        full_qsa.raw_index_keys,
        split_qsa.keys,
        split_qsa.values,
        split_qsa.raw_index_keys,
    )

    assert np.array_equal(np.asarray(full_logits), np.asarray(split_logits))
    assert full_state.frontier == split_state.frontier == tokens.shape[1]
    assert full_state.revision == 1
    assert split_state.revision == 2
    assert np.array_equal(np.asarray(full_qsa.keys), np.asarray(split_qsa.keys))
    assert np.array_equal(np.asarray(full_qsa.values), np.asarray(split_qsa.values))
    assert np.array_equal(
        np.asarray(full_qsa.raw_index_keys),
        np.asarray(split_qsa.raw_index_keys),
    )
    assert np.array_equal(np.asarray(full_qsa.position_ids), np.asarray(positions))
    assert np.array_equal(np.asarray(split_qsa.position_ids), np.asarray(positions))
    assert np.array_equal(np.asarray(full_state.valid_history), np.asarray(valid))
    assert np.array_equal(np.asarray(split_state.valid_history), np.asarray(valid))
    assert np.array_equal(np.asarray(full_state.position_history), np.asarray(positions))
    assert np.array_equal(np.asarray(split_state.position_history), np.asarray(positions))


def test_qwen4_forward_chunk_matches_real_gdn_continuation_state() -> None:
    tokens = mx.array([[3, 4, 5, 6]], dtype=mx.int64)
    valid = mx.array([[True, False, True, True]])
    model = _model_with_real_gdn()
    initial = model.new_state(1)
    full_logits, full_state = model.forward_chunk(
        initial,
        tokens,
        valid_tokens=valid,
    )

    split_a, split_state = model.forward_chunk(
        initial,
        tokens[:, :2],
        valid_tokens=valid[:, :2],
    )
    split_b, split_state = model.forward_chunk(
        split_state,
        tokens[:, 2:],
        valid_tokens=valid[:, 2:],
    )
    split_logits = mx.concatenate([split_a, split_b], axis=1)
    full_gdn = full_state.layers[0].mixer_state
    split_gdn = split_state.layers[0].mixer_state
    mx.eval(
        full_logits,
        split_logits,
        full_gdn.conv_state,
        full_gdn.recurrent_state,
        split_gdn.conv_state,
        split_gdn.recurrent_state,
    )

    assert np.allclose(np.asarray(full_logits), np.asarray(split_logits), rtol=0, atol=2e-6)
    assert np.allclose(
        np.asarray(full_gdn.conv_state),
        np.asarray(split_gdn.conv_state),
        rtol=0,
        atol=2e-6,
    )
    assert np.allclose(
        np.asarray(full_gdn.recurrent_state),
        np.asarray(split_gdn.recurrent_state),
        rtol=0,
        atol=2e-6,
    )
    assert full_state.frontier == split_state.frontier == tokens.shape[1]
    assert full_state.revision == 1
    assert split_state.revision == 2
    assert np.array_equal(np.asarray(full_state.valid_history), np.asarray(valid))
    assert np.array_equal(np.asarray(split_state.valid_history), np.asarray(valid))


def test_qwen4_shell_matches_proposal_partitions_and_committed_tokens() -> None:
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int64)

    full_model = _model([])
    full_coordinator = full_model.new_coordinator(1)
    full = full_model.propose(full_coordinator, tokens)
    full_state = full_coordinator.commit(full, 5)

    chunk_model = _model([])
    chunk_coordinator = chunk_model.new_coordinator(1)
    chunk_a = chunk_model.propose(chunk_coordinator, tokens[:, :2])
    chunk_coordinator.commit(chunk_a, 2)
    chunk_b = chunk_model.propose(chunk_coordinator, tokens[:, 2:])
    chunk_state = chunk_coordinator.commit(chunk_b, 3)
    chunk_logits = mx.concatenate([chunk_a.logits, chunk_b.logits], axis=1)

    token_model = _model([])
    token_coordinator = token_model.new_coordinator(1)
    token_logits = []
    for position in range(tokens.shape[1]):
        candidate = token_model.propose(token_coordinator, tokens[:, position : position + 1])
        token_coordinator.commit(candidate, 1)
        token_logits.append(candidate.logits)
    token_state = token_coordinator.state
    token_logits = mx.concatenate(token_logits, axis=1)
    mx.eval(full.logits, chunk_logits, token_logits)

    assert np.allclose(np.asarray(chunk_logits), np.asarray(full.logits), rtol=0, atol=2e-6)
    assert np.allclose(np.asarray(token_logits), np.asarray(full.logits), rtol=0, atol=2e-6)
    assert full_state.frontier == chunk_state.frontier == token_state.frontier == 5
    for expected, actual in zip(_state_arrays(full_state), _state_arrays(chunk_state), strict=True):
        assert np.array_equal(actual, expected)
    for expected, actual in zip(_state_arrays(full_state), _state_arrays(token_state), strict=True):
        assert np.array_equal(actual, expected)
    assert [layer.mixer_state for layer in full_state.layers] == [5] * 4
    assert [layer.mixer_state for layer in chunk_state.layers] == [5] * 4
    assert [layer.mixer_state for layer in token_state.layers] == [5] * 4


def test_qwen4_coordinator_commits_prefill_chunks_atomically() -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int64)

    first = coordinator.forward_chunk(tokens[:, :2])
    second = coordinator.forward_chunk(tokens[:, 2:])
    mx.eval(first, second)

    assert first.shape[1] == second.shape[1] == 2
    assert coordinator.state.frontier == 4
    assert coordinator.state.revision == 2
    assert [layer.mixer_state for layer in coordinator.state.layers] == [4] * 4


def test_qwen4_committed_chunk_eval_failure_restores_coordinator_and_mutable_storage(
    monkeypatch,
) -> None:
    class JournalState:
        def __init__(self, mixer, offset: int) -> None:
            self.mixer = mixer
            self.offset = offset

        def state_arrays(self):
            return (self.mixer.buffer,)

    class JournalMixer(_Mixer):
        def __init__(self, events) -> None:
            super().__init__(events, "journal", 0.1, 0.0)
            self.buffer = mx.zeros((1,), dtype=mx.int32)
            self.frontier = 0
            self.undo: list[tuple[int, int]] = []

        def fork_state(self, state):
            return state

        def snapshot_state(self, state):
            return state

        def restore_state(self, state) -> None:
            target = 0 if state is None else state.offset
            while self.undo and self.frontier > target:
                frontier, value = self.undo.pop()
                self.buffer[0] = value
                self.frontier = frontier

        def validate_state(self, state, *, expected_frontier, position_history) -> None:
            del position_history
            observed = 0 if state is None else state.offset
            if observed != expected_frontier or self.frontier != expected_frontier:
                raise ValueError("journal state is stale")

        def __call__(self, hidden_states, **kwargs):
            del kwargs
            previous = int(np.asarray(self.buffer)[0])
            self.undo.append((self.frontier, previous))
            self.frontier += hidden_states.shape[1]
            self.buffer[0] = previous + hidden_states.shape[1]
            return Qwen4MixerOutput(
                output=mx.zeros_like(hidden_states),
                state=JournalState(self, self.frontier),
                frontier=self.frontier,
            )

    model = _model([])
    journal = JournalMixer([])
    layers = list(model.layers)
    layers[0] = replace(layers[0], mixer=journal)
    model.layers = tuple(layers)
    coordinator = model.new_coordinator(1)
    original = coordinator.state
    mx.eval(journal.buffer)
    before = bytes(memoryview(mx.contiguous(journal.buffer)).cast("B"))
    real_eval = model_module.mx.eval

    def fail_eval(*_arrays) -> None:
        raise RuntimeError("injected committed eval failure")

    monkeypatch.setattr(model_module.mx, "eval", fail_eval)
    with pytest.raises(RuntimeError, match="injected committed eval failure"):
        coordinator.forward_chunk(mx.array([[1]], dtype=mx.int64))
    monkeypatch.setattr(model_module.mx, "eval", real_eval)
    mx.eval(journal.buffer)

    assert coordinator.state is original
    assert coordinator.state.frontier == coordinator.state.revision == 0
    assert bytes(memoryview(mx.contiguous(journal.buffer)).cast("B")) == before
    assert journal.frontier == 0
    assert journal.undo == []


def test_qwen4_committed_chunk_preflights_every_layer_before_commit() -> None:
    class CommitMixer(_Mixer):
        def __init__(self, events, label: str, *, fail: bool = False) -> None:
            super().__init__(events, label, 0.1, 0.0)
            self.fail = fail
            self.preflights = 0
            self.finalizes = 0

        def preflight_commit_state(self, state) -> None:
            del state
            self.preflights += 1
            if self.fail:
                raise RuntimeError("commit preflight failed")

        def commit_state_preflighted(self, state):
            self.finalizes += 1
            return state

    model = _model([])
    first = CommitMixer([], "first")
    second = CommitMixer([], "second", fail=True)
    layers = list(model.layers)
    layers[0] = replace(layers[0], mixer=first)
    layers[1] = replace(layers[1], mixer=second)
    model.layers = tuple(layers)
    coordinator = model.new_coordinator(1)
    original = coordinator.state

    with pytest.raises(RuntimeError, match="commit preflight failed"):
        coordinator.forward_chunk(mx.array([[1]], dtype=mx.int64))

    assert coordinator.state is original
    assert first.preflights == second.preflights == 1
    assert first.finalizes == second.finalizes == 0


def test_qwen4_coordinator_generation_stays_on_trusted_validation_path() -> None:
    model = _model([])
    layers = list(model.layers)
    tracker = _TrustedMixer([], "trusted QSA", 0.6, 0.15)
    layers[3] = replace(layers[3], mixer=tracker)
    model.layers = tuple(layers)
    coordinator = model.new_coordinator(1)

    coordinator.forward_chunk(mx.array([[1, 2]], dtype=mx.int64))
    candidate = coordinator.propose(mx.array([[3]], dtype=mx.int64))
    coordinator.commit(candidate, 1)

    assert coordinator.state.frontier == 3
    assert tracker.strict_frontiers == [0]
    assert tracker.trusted_frontiers
    assert tracker.trusted_steps == 2


def test_qwen4_shell_builds_one_bound_mask_certificate_for_all_qsa_layers() -> None:
    class CertifiedMixer(_TrustedMixer):
        supports_trusted_mask_certificate = True

        def __init__(self, label: str) -> None:
            super().__init__([], label, 0.6, 0.15)
            self.certificates: list[_Qwen4TrustedMaskCertificate] = []

        def _step_trusted_certified(
            self,
            hidden_states: mx.array,
            *,
            valid_tokens: mx.array,
            visible_history: mx.array,
            position_ids: mx.array,
            state,
            mask_certificate: _Qwen4TrustedMaskCertificate,
        ) -> Qwen4MixerOutput:
            self.certificates.append(mask_certificate)
            return self.step_trusted(
                hidden_states,
                valid_tokens=valid_tokens,
                visible_history=visible_history,
                position_ids=position_ids,
                state=state,
            )

    model = _model([])
    first = CertifiedMixer("first QSA")
    second = CertifiedMixer("second QSA")
    layers = list(model.layers)
    layers[0] = replace(layers[0], mixer_kind="qsa", mixer=first)
    layers[3] = replace(layers[3], mixer_kind="qsa", mixer=second)
    model.layers = tuple(layers)
    coordinator = model.new_coordinator(1)

    coordinator.forward_chunk(mx.array([[1, 2]], dtype=mx.int64))

    assert len(first.certificates) == len(second.certificates) == 1
    certificate = first.certificates[0]
    assert second.certificates[0] is certificate
    assert certificate.valid_tokens.shape == (1, 2)
    assert certificate.visible_history.shape == (1, 2)
    assert certificate.current_frontier == 0
    assert certificate.next_frontier == 2
    assert certificate.allowed_mixers == (first, second)
    assert certificate.source_states == (None, None)
    assert model.trusted_mask_stats() == {
        "trusted_mask_certificate_builds": 1,
        "trusted_all_valid_reductions": 1,
        "batched_qsa_undo_batches": 0,
        "batched_qsa_undo_reservations": 0,
    }


def test_qwen4_shell_batches_decode_undo_at_the_all_valid_boundary() -> None:
    model = _model([])
    first = _BatchedFiniteMixer("first QSA")
    second = _BatchedFiniteMixer("second QSA")
    layers = list(model.layers)
    layers[0] = replace(layers[0], mixer_kind="qsa", mixer=first)
    layers[3] = replace(layers[3], mixer_kind="qsa", mixer=second)
    model.layers = tuple(layers)
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1, 2]], dtype=mx.int64))

    coordinator.propose(mx.array([[3]], dtype=mx.int64))

    assert len(first.prepared) == len(second.prepared) == 1
    assert first.prepared[0].evaluated
    assert second.prepared[0].evaluated
    certificate = first.certificates[-1]
    assert second.certificates[-1] is certificate
    assert certificate.prepared_undos == (first.prepared[0], second.prepared[0])
    assert model.trusted_mask_stats() == {
        "trusted_mask_certificate_builds": 2,
        "trusted_all_valid_reductions": 2,
        "batched_qsa_undo_batches": 1,
        "batched_qsa_undo_reservations": 2,
    }
    assert model.qsa_append_finite_stats() == {
        "batched_qsa_append_finite_batches": 1,
        "batched_qsa_append_finite_predicates": 2,
        "batched_qsa_append_finite_failures": 0,
    }


def test_qwen4_shell_rejects_batched_nonfinite_append_before_publication_and_retries() -> None:
    model = _model([])
    first = _BatchedFiniteMixer("first QSA")
    second = _BatchedFiniteMixer("second QSA", finite=False)
    layers = list(model.layers)
    layers[0] = replace(layers[0], mixer_kind="qsa", mixer=first)
    layers[3] = replace(layers[3], mixer_kind="qsa", mixer=second)
    model.layers = tuple(layers)
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1, 2]], dtype=mx.int64))
    original = coordinator.state

    with pytest.raises(ValueError, match="finite unpadded"):
        coordinator.propose(mx.array([[3]], dtype=mx.int64))

    assert coordinator.state is original
    assert coordinator.state.frontier == 2
    assert coordinator.state.revision == 1
    assert model.qsa_append_finite_stats() == {
        "batched_qsa_append_finite_batches": 1,
        "batched_qsa_append_finite_predicates": 2,
        "batched_qsa_append_finite_failures": 1,
    }

    second.finite = True
    candidate = coordinator.propose(mx.array([[3]], dtype=mx.int64))
    mx.eval(candidate.logits)
    coordinator.commit(candidate, 1)
    assert coordinator.state.frontier == 3
    assert coordinator.state.revision == 2
    assert model.qsa_append_finite_stats() == {
        "batched_qsa_append_finite_batches": 2,
        "batched_qsa_append_finite_predicates": 4,
        "batched_qsa_append_finite_failures": 1,
    }


def test_qwen4_append_finite_batch_rejects_missing_registration() -> None:
    first = object()
    second = object()
    first_source = object()
    second_source = object()
    first_result = object()
    batch = model_module._Qwen4AppendFiniteBatch(
        allowed_mixers=(first, second),
        source_states=(first_source, second_source),
    )
    batch.register(
        mixer=first,
        source_state=first_source,
        result_state=first_result,
        predicate=mx.array(True, dtype=mx.bool_),
        capability=model_module._QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    )

    with pytest.raises(ValueError, match="batch is incomplete"):
        batch.seal(
            ((first, first_result), (second, None)),
            capability=model_module._QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
        )


def test_qwen4_append_finite_batch_rejects_duplicate_registration() -> None:
    mixer = object()
    source = object()
    result = object()
    batch = model_module._Qwen4AppendFiniteBatch(
        allowed_mixers=(mixer,),
        source_states=(source,),
    )
    kwargs = {
        "mixer": mixer,
        "source_state": source,
        "result_state": result,
        "predicate": mx.array(True, dtype=mx.bool_),
        "capability": model_module._QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    }
    batch.register(**kwargs)

    with pytest.raises(ValueError, match="registered twice"):
        batch.register(**kwargs)


def test_qwen4_shell_does_not_issue_a_certificate_for_padding_holes() -> None:
    class CertifiedMixer(_TrustedMixer):
        supports_trusted_mask_certificate = True

        def _step_trusted_certified(self, *args, **kwargs) -> Qwen4MixerOutput:
            raise AssertionError("padding-hole request received a certificate")

    model = _model([])
    tracker = CertifiedMixer([], "QSA", 0.6, 0.15)
    layers = list(model.layers)
    layers[3] = replace(layers[3], mixer_kind="qsa", mixer=tracker)
    model.layers = tuple(layers)
    coordinator = model.new_coordinator(1)

    coordinator.forward_chunk(
        mx.array([[1]], dtype=mx.int64),
        valid_tokens=mx.zeros((1, 1), dtype=mx.bool_),
    )

    assert tracker.trusted_steps == 1
    assert model.trusted_mask_stats() == {
        "trusted_mask_certificate_builds": 0,
        "trusted_all_valid_reductions": 1,
        "batched_qsa_undo_batches": 0,
        "batched_qsa_undo_reservations": 0,
    }


def test_qwen4_shell_shares_one_rope_factor_set_for_one_token_qsa_decode() -> None:
    model = _model_with_real_qsa()
    coordinator = model.new_coordinator(1)

    coordinator.forward_chunk(mx.array([[1, 2]], dtype=mx.int64))
    assert model.shared_qsa_rope_stats() == {
        "qsa_layers": 1,
        "shared_qsa_rope_factor_builds": 0,
        "shared_rope_factor_calls": 0,
        "shared_rope_applications": 0,
    }

    candidate = coordinator.propose(mx.array([[3]], dtype=mx.int64))
    mx.eval(candidate.logits)

    assert model.shared_qsa_rope_stats() == {
        "qsa_layers": 1,
        "shared_qsa_rope_factor_builds": 1,
        "shared_rope_factor_calls": 1,
        "shared_rope_applications": 3,
    }


def test_qwen4_shell_clears_every_bound_mask_scope_after_layer_failure() -> None:
    class CertifiedMixer(_TrustedMixer):
        supports_trusted_mask_certificate = True

        def __init__(self, label: str, *, fail: bool = False) -> None:
            super().__init__([], label, 0.6, 0.15)
            self.fail = fail
            self.binds = 0
            self.clears = 0
            self.active_scope = None

        def _bind_trusted_mask_issuer(self, issuer: object, scope: object) -> None:
            del issuer
            assert self.active_scope is None
            self.active_scope = scope
            self.binds += 1

        def _clear_trusted_mask_scope(self, issuer: object, scope: object) -> None:
            del issuer
            assert self.active_scope is scope
            self.active_scope = None
            self.clears += 1

        def _step_trusted_certified(
            self,
            hidden_states: mx.array,
            *,
            valid_tokens: mx.array,
            visible_history: mx.array,
            position_ids: mx.array,
            state,
            mask_certificate: _Qwen4TrustedMaskCertificate,
        ) -> Qwen4MixerOutput:
            assert self.active_scope is mask_certificate.scope
            if self.fail:
                raise RuntimeError("certified layer failed")
            return self.step_trusted(
                hidden_states,
                valid_tokens=valid_tokens,
                visible_history=visible_history,
                position_ids=position_ids,
                state=state,
            )

    model = _model([])
    first = CertifiedMixer("first QSA")
    second = CertifiedMixer("second QSA", fail=True)
    layers = list(model.layers)
    layers[0] = replace(layers[0], mixer_kind="qsa", mixer=first)
    layers[3] = replace(layers[3], mixer_kind="qsa", mixer=second)
    model.layers = tuple(layers)
    coordinator = model.new_coordinator(1)
    original = coordinator.state

    with pytest.raises(RuntimeError, match="certified layer failed"):
        coordinator.forward_chunk(mx.array([[1]], dtype=mx.int64))

    assert coordinator.state is original
    assert first.active_scope is second.active_scope is None
    assert first.binds == first.clears == 1
    assert second.binds == second.clears == 1


@pytest.mark.parametrize(
    "partitions",
    [(5,), (2, 3), (1, 1, 1, 1, 1)],
    ids=("full", "chunked", "tokenwise"),
)
def test_qwen4_shell_uses_trusted_validation_only_inside_owned_progression(
    partitions: tuple[int, ...],
) -> None:
    model = _model([])
    layers = list(model.layers)
    tracker = _TrustedMixer([], "trusted QSA", 0.6, 0.15)
    layers[3] = replace(layers[3], mixer=tracker)
    model.layers = tuple(layers)
    coordinator = model.new_coordinator(1)
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int64)

    cursor = 0
    expected_strict_frontiers = [0]
    for width in partitions:
        candidate = model.propose(coordinator, tokens[:, cursor : cursor + width])
        expected_strict_frontiers.append(cursor)
        coordinator.commit(candidate, width)
        cursor += width
        expected_strict_frontiers.extend((cursor, cursor))

    assert cursor == tokens.shape[1]
    assert coordinator.state.frontier == tokens.shape[1]
    assert tracker.strict_frontiers == expected_strict_frontiers
    assert tracker.trusted_steps == tokens.shape[1]
    assert tracker.trusted_frontiers

    model.validate_state(coordinator.state)
    assert tracker.strict_frontiers == [*expected_strict_frontiers, tokens.shape[1]]


def test_qwen4_propose_rejects_one_sided_same_shape_position_replacement() -> None:
    model = _model_with_real_qsa()
    coordinator = model.new_coordinator(1)
    seed = model.propose(coordinator, mx.array([[1]], dtype=mx.int64))
    committed = coordinator.commit(seed, 1)
    layer = committed.layers[0]
    corrupted_mixer_state = replace(
        layer.mixer_state,
        position_ids=layer.mixer_state.position_ids
        + mx.array([[[0]], [[1]], [[0]]], dtype=mx.int64),
    )
    coordinator._state = replace(
        committed,
        layers=(replace(layer, mixer_state=corrupted_mixer_state),),
    )

    with pytest.raises(ValueError, match="do not match"):
        model.propose(coordinator, mx.array([[2]], dtype=mx.int64))


def test_qwen4_commit_rejects_one_sided_same_shape_position_replacement() -> None:
    model = _model_with_real_qsa()
    coordinator = model.new_coordinator(1)
    candidate = model.propose(coordinator, mx.array([[1]], dtype=mx.int64))
    checkpoint = candidate.checkpoints[0]
    layer = checkpoint.layers[0]
    corrupted_mixer_state = replace(
        layer.mixer_state,
        position_ids=layer.mixer_state.position_ids
        + mx.array([[[0]], [[1]], [[0]]], dtype=mx.int64),
    )
    candidate.checkpoints = (
        replace(
            checkpoint,
            layers=(replace(layer, mixer_state=corrupted_mixer_state),),
        ),
    )

    with pytest.raises(ValueError, match="do not match"):
        coordinator.commit(candidate, 1)

    assert candidate.consumed is False
    assert coordinator.state.frontier == 0


def test_qwen4_shell_partial_commit_is_atomic_and_rejects_stale_sibling() -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)
    original = coordinator.state
    tokens = mx.array([[7, 8, 9]], dtype=mx.int64)
    rejected = model.propose(coordinator, tokens)
    accepted = model.propose(coordinator, tokens)
    stale = model.propose(coordinator, tokens)

    assert coordinator.state is original
    assert coordinator.commit(rejected, 0) is original
    with pytest.raises(ValueError, match="already been consumed"):
        coordinator.commit(rejected, 0)
    committed = coordinator.commit(accepted, 2)

    assert committed.frontier == 2
    assert committed.revision == 1
    assert np.array_equal(np.asarray(committed.valid_history), np.ones((1, 2), dtype=bool))
    assert np.array_equal(np.asarray(committed.position_history)[0, 0], [0, 1])
    assert all(layer.mixer_offset == 2 for layer in committed.layers)
    assert committed.layers[1].ple_state.offset == 2
    with pytest.raises(ValueError, match="does not extend"):
        coordinator.commit(stale, 1)


def test_qwen4_commit_reuses_only_its_sealed_evaluated_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)
    candidate = model.propose(coordinator, mx.array([[7]], dtype=mx.int64))
    real_eval = model_module.mx.eval
    evaluated: list[tuple[mx.array, ...]] = []

    def record_eval(*arrays: mx.array) -> None:
        evaluated.append(arrays)
        real_eval(*arrays)

    monkeypatch.setattr(model_module.mx, "eval", record_eval)
    coordinator.commit(candidate, 1)

    assert [len(arrays) for arrays in evaluated] == [1]
    assert model.commit_boundary_stats() == {
        "evaluated_commit_boundary_calls": 1,
        "full_commit_boundary_calls": 0,
    }


def test_qwen4_commit_falls_back_when_checkpoint_tuple_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)
    candidate = model.propose(coordinator, mx.array([[7]], dtype=mx.int64))
    candidate.checkpoints = tuple(list(candidate.checkpoints))
    real_eval = model_module.mx.eval
    evaluated: list[tuple[mx.array, ...]] = []

    def record_eval(*arrays: mx.array) -> None:
        evaluated.append(arrays)
        real_eval(*arrays)

    monkeypatch.setattr(model_module.mx, "eval", record_eval)
    coordinator.commit(candidate, 1)

    assert len(evaluated) == 1
    assert len(evaluated[0]) > 1
    assert model.commit_boundary_stats() == {
        "evaluated_commit_boundary_calls": 0,
        "full_commit_boundary_calls": 1,
    }


def test_qwen4_plain_serial_lane_consumes_coordinator_without_branch_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _all_gdn_model()
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1, 2]], dtype=mx.int64))
    lane = coordinator.try_enter_plain_serial_lane()
    assert lane is not None

    def forbidden(*_args, **_kwargs):
        raise AssertionError("plain serial decode must not use branch state")

    monkeypatch.setattr(model, "_fork_state", forbidden)
    monkeypatch.setattr(model, "_snapshot_state", forbidden)
    monkeypatch.setattr(model, "_restore_state", forbidden)

    step = lane.begin_step(mx.array([[3]], dtype=mx.int64))
    lane.finish_step(step, evaluated=(step.logits,))

    assert lane.frontier == 3
    assert step.consumed is True
    assert model.plain_serial_lane_stats() == {
        "plain_serial_lane_entries": 1,
        "plain_serial_steps": 1,
        "plain_serial_failures": 0,
    }
    assert model.compiled_gdn_run_stats() == {
        "builds": 0,
        "calls": 0,
        "layer_calls": 0,
        "ineligible_calls": 1,
        "failures": 0,
    }
    with pytest.raises(RuntimeError, match="closed"):
        _ = coordinator.state
    with pytest.raises(ValueError, match="unissued or stale"):
        lane.finish_step(step, evaluated=(step.logits,))
    lane.close()


def test_qwen4_plain_serial_lane_chains_one_private_target_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _all_gdn_model()
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1]], dtype=mx.int64))
    lane = coordinator.enter_plain_serial_lane()

    first = lane.begin_step(mx.array([[2]], dtype=mx.int64))
    lane.prime_pipelined_step(
        first,
        evaluated=(first.logits,),
        queued=(),
    )
    second = lane.chain_pipelined_step(first, mx.array([[3]], dtype=mx.int64))
    lane.finish_pipelined_transition(
        first,
        second,
        evaluated=(first.logits,),
        queued=(second.logits,),
    )

    assert first.consumed is True
    assert second.consumed is False
    lane.finish_terminal_pipelined_step(second, evaluated=(second.logits,))
    assert second.consumed is True
    assert lane.frontier == 3
    assert model.plain_serial_lane_stats() == {
        "plain_serial_lane_entries": 1,
        "plain_serial_steps": 2,
        "plain_serial_failures": 0,
    }
    assert model.plain_serial_pipeline_stats() == {
        "primes": 1,
        "transitions": 1,
        "terminal_finishes": 1,
        "discards": 0,
        "discard_drain_failures": 0,
    }
    lane.close()


def test_qwen4_plain_serial_pipeline_close_drains_and_discards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _all_gdn_model()
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1]], dtype=mx.int64))
    lane = coordinator.enter_plain_serial_lane()
    step = lane.begin_step(mx.array([[2]], dtype=mx.int64))
    lane.prime_pipelined_step(step, evaluated=(step.logits,), queued=())

    lane.close()

    assert step.consumed is True
    assert model.plain_serial_lane_stats() == {
        "plain_serial_lane_entries": 1,
        "plain_serial_steps": 0,
        "plain_serial_failures": 0,
    }
    assert model.plain_serial_pipeline_stats() == {
        "primes": 1,
        "transitions": 0,
        "terminal_finishes": 0,
        "discards": 1,
        "discard_drain_failures": 0,
    }
    with pytest.raises(RuntimeError, match="closed"):
        _ = lane.frontier


def test_qwen4_compiled_gdn_run_schedule_preserves_eager_boundaries() -> None:
    assert [
        (spec.indices, spec.expects_pending, spec.publishes_final_write)
        for spec in model_module._QWEN4_COMPILED_GDN_RUN_SPECS
    ] == [
        ((0,), False, True),
        ((1,), False, False),
        ((2,), True, False),
        *[((start, start + 1, start + 2), True, False) for start in range(4, 45, 4)],
    ]


def test_qwen4_compiled_gdn_full_residency_uses_physical_expert_counts() -> None:
    counts = (320, 384, 512)
    layers = tuple(SimpleNamespace(mlp=SimpleNamespace(physical_experts=count)) for count in counts)
    capacities = {index: count for index, count in enumerate(counts)}

    assert model_module._compiled_gdn_full_residency_matches(layers, capacities) is True
    assert (
        model_module._compiled_gdn_full_residency_matches(
            layers,
            capacities | {1: 383},
        )
        is False
    )
    assert (
        model_module._compiled_gdn_full_residency_matches(
            layers,
            {0: 320, 1: 384},
        )
        is False
    )


def test_qwen4_compiled_gdn_run_rebuilds_functional_state_and_clears_on_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moespresso.runtime.qwen4.gdn as gdn_module

    monkeypatch.setattr(model_module.mx, "compile", lambda function: function)

    def fake_prerouter_step(
        layer,
        hidden_states: mx.array,
        *,
        state: Qwen4GDNState,
        pending_output: mx.array | None,
        pending_injection: mx.array | None,
        certified_all_valid: bool,
    ):
        del layer
        assert certified_all_valid is True
        if pending_output is not None:
            hidden_states = model_module.gated_residual_write(
                hidden_states,
                pending_output,
                pending_injection,
            )
        mlp_hidden = mx.sum(
            hidden_states.reshape(*hidden_states.shape[:-1], 2, 2),
            axis=-2,
        )
        return SimpleNamespace(
            mlp_hidden=mlp_hidden,
            residual=hidden_states,
            injection=mx.full((1, 1, 2), 0.25, dtype=hidden_states.dtype),
            state=Qwen4GDNState(
                conv_state=state.conv_state + 1,
                recurrent_state=state.recurrent_state + 2,
                offset=state.offset + 1,
            ),
        )

    monkeypatch.setattr(gdn_module, "qwen4_gdn_prerouter_step", fake_prerouter_step)
    events: list[str] = []
    layers = []
    for index in range(48):
        kind = "qsa" if index in range(3, 48, 4) else "gdn"
        layers.append(
            Qwen4DecoderLayer(
                mixer_kind=kind,
                attention_residual=_Residual(events, f"L{index} attention", 0.0),
                mixer=_Mixer(events, f"L{index} mixer", 1.0, 0.0)
                if kind == "qsa"
                else _CompiledGDNMixer(),
                mlp_residual=_Residual(events, f"L{index} MLP", 0.0),
                mlp=_MLP(events, f"L{index} experts", 0.5, 0.0),
                ple=_PLE(events, 0.0) if index == 1 else None,
            )
        )
    model = Qwen4TextModelShell(
        cache_identity="compiled-gdn-test",
        embedding=_Embedding(),
        layers=tuple(layers),
        final_residual=_FinalResidual(events),
        lm_head=_IdentityHead(),
        hidden_size=2,
        branch_count=2,
    )
    source_states = tuple(
        Qwen4LayerState(
            mixer_kind=layer.mixer_kind,
            mixer_state=None
            if layer.mixer_kind == "qsa"
            else Qwen4GDNState(
                conv_state=mx.array([index], dtype=mx.float32),
                recurrent_state=mx.array([index + 1], dtype=mx.float32),
                offset=5,
            ),
            mixer_offset=5,
        )
        for index, layer in enumerate(layers)
    )
    source = Qwen4CompositeState(
        cache_identity="compiled-gdn-test",
        revision=0,
        frontier=5,
        batch_size=1,
        valid_history=mx.ones((1, 5), dtype=mx.bool_),
        position_history=mx.zeros((3, 1, 5), dtype=mx.int64),
        layers=source_states,
    )

    runs = model._build_compiled_gdn_runs()
    object.__setattr__(model, "_compiled_gdn_runs", runs)
    hidden, pending_output, pending_injection, states = model._execute_compiled_gdn_run(
        runs[4],
        mx.ones((1, 1, 4), dtype=mx.float32),
        mx.ones((1, 1, 2), dtype=mx.float32),
        mx.full((1, 1, 2), 0.5, dtype=mx.float32),
        source,
        next_frontier=6,
        position_history=mx.zeros((3, 1, 6), dtype=mx.int64),
        ple_state=None,
    )
    mx.eval(hidden, pending_output, pending_injection)

    assert hidden.shape == (1, 1, 4)
    assert pending_output.shape == (1, 1, 2)
    assert pending_injection.shape == (1, 1, 2)
    assert [state.mixer_offset for state in states] == [6, 6, 6]
    assert [state.mixer_state.offset for state in states] == [6, 6, 6]
    assert model.compiled_gdn_run_stats() == {
        "builds": 14,
        "calls": 1,
        "layer_calls": 3,
        "ineligible_calls": 0,
        "failures": 0,
    }

    model.close()
    assert model._compiled_gdn_runs is None


def test_qwen4_plain_serial_lane_failure_is_request_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _all_gdn_model()
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1]], dtype=mx.int64))
    lane = coordinator.enter_plain_serial_lane()
    step = lane.begin_step(mx.array([[2]], dtype=mx.int64))

    monkeypatch.setattr(
        model_module.mx,
        "eval",
        lambda *_arrays: (_ for _ in ()).throw(RuntimeError("device failure")),
    )
    with pytest.raises(RuntimeError, match="device failure"):
        lane.finish_step(step, evaluated=(step.logits,))

    assert step.consumed is True
    assert model.plain_serial_lane_stats() == {
        "plain_serial_lane_entries": 1,
        "plain_serial_steps": 0,
        "plain_serial_failures": 1,
    }
    with pytest.raises(RuntimeError, match="failed"):
        _ = lane.frontier


def test_qwen4_plain_serial_lane_close_abandons_pending_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _all_gdn_model()
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1]], dtype=mx.int64))
    lane = coordinator.enter_plain_serial_lane()
    step = lane.begin_step(mx.array([[2]], dtype=mx.int64))

    lane.close()

    assert step.consumed is True
    assert model.plain_serial_lane_stats() == {
        "plain_serial_lane_entries": 1,
        "plain_serial_steps": 0,
        "plain_serial_failures": 1,
    }
    with pytest.raises(RuntimeError, match="failed"):
        _ = lane.frontier


def test_qwen4_plain_serial_lane_refuses_externally_masked_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _all_gdn_model()
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(
        mx.array([[1]], dtype=mx.int64),
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
    )

    assert coordinator.try_enter_plain_serial_lane() is None


def test_qwen4_shell_rejects_candidate_from_sibling_coordinator() -> None:
    model = _model([])
    first = model.new_coordinator(1)
    second = model.new_coordinator(1)
    candidate = model.propose(first, mx.array([[7]], dtype=mx.int64))

    with pytest.raises(ValueError, match="does not extend"):
        second.commit(candidate, 1)

    assert first.state.frontier == second.state.frontier == 0
    assert candidate.consumed is False


def test_qwen4_shell_carries_physical_padding_history_into_qsa() -> None:
    model = _model([])
    coordinator = model.new_coordinator(2)
    valid = mx.array([[False, True], [True, True]])
    semantic_positions = mx.array([[0, 0], [0, 1]], dtype=mx.int64)
    candidate = model.propose(
        coordinator,
        mx.array([[0, 5], [6, 7]], dtype=mx.int64),
        valid_tokens=valid,
        position_ids=semantic_positions,
    )
    qsa = model.layers[3].mixer
    assert isinstance(qsa, _Mixer)
    mx.eval(*qsa.visible_histories, *qsa.position_ids)

    assert np.array_equal(
        np.asarray(qsa.visible_histories[0]),
        np.array([[False], [True]]),
    )
    assert np.array_equal(np.asarray(qsa.visible_histories[1]), np.asarray(valid))
    assert np.array_equal(np.asarray(qsa.position_ids[0])[0, :, 0], [0, 0])
    assert np.array_equal(np.asarray(qsa.position_ids[1])[0, :, 0], [0, 1])
    assert np.array_equal(
        np.asarray(candidate.checkpoints[-1].valid_history),
        np.asarray(valid),
    )
    assert np.array_equal(
        np.asarray(candidate.checkpoints[-1].position_history)[0],
        np.asarray(semantic_positions),
    )


def test_qwen4_shell_fails_before_layers_on_frontier_or_position_geometry_mismatch() -> None:
    events: list[str] = []
    model = _model(events)
    coordinator = model.new_coordinator(1)
    invalid_layer = replace(coordinator.state.layers[0], mixer_offset=1)
    invalid = replace(
        coordinator.state,
        layers=(invalid_layer, *coordinator.state.layers[1:]),
    )
    coordinator._state = invalid

    with pytest.raises(ValueError, match="off the public frontier"):
        model.propose(coordinator, mx.array([[1]], dtype=mx.int64))
    assert events == []

    coordinator._state = model.new_coordinator(1).state
    with pytest.raises(ValueError, match="match input_ids"):
        model.propose(
            coordinator,
            mx.array([[1, 2]], dtype=mx.int64),
            position_ids=mx.array([[0, 1, 2]], dtype=mx.int64),
        )
    assert events == []


def test_qwen4_shell_rejects_ple_state_at_frontier_zero() -> None:
    model = _model([])
    state = model.new_state(1)
    layer = state.layers[1]
    unexpected = Qwen4PLEState(
        token_context=mx.zeros((1, 2), dtype=mx.int64),
        conv_state=mx.zeros((1, 1, 4), dtype=mx.float32),
        offset=0,
    )
    invalid = replace(
        state,
        layers=(
            state.layers[0],
            replace(layer, ple_state=unexpected),
            *state.layers[2:],
        ),
    )

    with pytest.raises(ValueError, match="empty at frontier zero"):
        model.validate_state(invalid)


def test_qwen4_shell_preserves_three_axis_mrope_positions() -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)
    mrope = mx.array(
        [
            [[4, 5]],
            [[8, 8]],
            [[11, 12]],
        ],
        dtype=mx.int64,
    )
    candidate = model.propose(
        coordinator,
        mx.array([[17, 18]], dtype=mx.int64),
        position_ids=mrope,
    )
    committed = coordinator.commit(candidate, 2)
    mx.eval(committed.position_history)

    assert committed.frontier == 2
    assert np.array_equal(np.asarray(committed.position_history), np.asarray(mrope))


def test_qwen4_shell_accepts_released_four_plane_position_contract() -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)
    released_positions = mx.array(
        [
            [[0, 1]],
            [[4, 5]],
            [[8, 8]],
            [[11, 12]],
        ],
        dtype=mx.int64,
    )
    candidate = model.propose(
        coordinator,
        mx.array([[17, 18]], dtype=mx.int64),
        position_ids=released_positions,
    )
    committed = coordinator.commit(candidate, 2)
    mx.eval(committed.position_history)

    assert np.array_equal(
        np.asarray(committed.position_history),
        np.asarray(released_positions)[1:],
    )


def test_qwen4_shell_forks_mutable_state_and_snapshots_each_token() -> None:
    model = _model([])
    layers = list(model.layers)
    layers[0] = replace(
        layers[0],
        mixer=_MutableMixer([], "mutable", 0.1, 0),
    )
    model.layers = tuple(layers)
    coordinator = model.new_coordinator(1)
    seed = model.propose(coordinator, mx.array([[1]], dtype=mx.int64))
    committed = coordinator.commit(seed, 1)
    committed_buffer = np.asarray(committed.layers[0].mixer_state["buffer"]).copy()

    candidate = model.propose(
        coordinator,
        mx.array([[2, 3, 4]], dtype=mx.int64),
    )
    checkpoint_buffers = [
        np.asarray(checkpoint.layers[0].mixer_state["buffer"]).copy()
        for checkpoint in candidate.checkpoints
    ]

    assert np.array_equal(
        np.asarray(coordinator.state.layers[0].mixer_state["buffer"]),
        committed_buffer,
    )
    assert [int(buffer[0]) for buffer in checkpoint_buffers] == [2, 3, 4]


def test_qwen4_forward_chunk_does_not_mutate_committed_input_state() -> None:
    model = _model([])
    layers = list(model.layers)
    layers[0] = replace(
        layers[0],
        mixer=_MutableMixer([], "mutable", 0.1, 0),
    )
    model.layers = tuple(layers)
    seed_logits, committed = model.forward_chunk(
        model.new_state(1),
        mx.array([[1]], dtype=mx.int64),
    )
    mx.eval(seed_logits, committed.layers[0].mixer_state["buffer"])
    original_buffer = np.asarray(committed.layers[0].mixer_state["buffer"]).copy()

    _, advanced = model.forward_chunk(
        committed,
        mx.array([[2, 3, 4]], dtype=mx.int64),
    )
    mx.eval(
        committed.layers[0].mixer_state["buffer"],
        advanced.layers[0].mixer_state["buffer"],
    )

    assert committed.frontier == 1
    assert committed.revision == 1
    assert np.array_equal(
        np.asarray(committed.layers[0].mixer_state["buffer"]),
        original_buffer,
    )
    assert advanced.frontier == 4
    assert advanced.revision == 2
    assert int(np.asarray(advanced.layers[0].mixer_state["buffer"])[0]) == 4


def test_qwen4_shell_materializes_lazy_token_outputs_and_snapshots() -> None:
    def lazy_model() -> Qwen4TextModelShell:
        model = _model([])
        layers = list(model.layers)
        layers[0] = replace(
            layers[0],
            mixer=_LazyMutableMixer([], "lazy mutable", 0.1, 0),
        )
        model.layers = tuple(layers)
        return model

    reference_model = lazy_model()
    reference = reference_model.propose(
        reference_model.new_coordinator(1),
        mx.array([[2]], dtype=mx.int64),
    )
    expected_first_logits = np.asarray(reference.logits[:, :1]).copy()

    model = lazy_model()
    candidate = model.propose(
        model.new_coordinator(1),
        mx.array([[2, 3, 4]], dtype=mx.int64),
    )
    checkpoint_buffers = [
        np.asarray(checkpoint.layers[0].mixer_state["buffer"]).copy()
        for checkpoint in candidate.checkpoints
    ]

    assert np.array_equal(np.asarray(candidate.logits[:, :1]), expected_first_logits)
    assert [float(buffer[0]) for buffer in checkpoint_buffers] == [1.0, 2.0, 3.0]


def test_qwen4_shell_rejects_stale_reported_mixer_frontier() -> None:
    class _StaleMixer(_Mixer):
        def __call__(self, hidden_states: mx.array, **kwargs) -> Qwen4MixerOutput:
            del kwargs
            return Qwen4MixerOutput(output=mx.zeros_like(hidden_states), state=0, frontier=0)

    events: list[str] = []
    model = _model(events)
    layers = list(model.layers)
    layers[0] = replace(layers[0], mixer=_StaleMixer(events, "stale", 0, 0))
    model.layers = tuple(layers)

    with pytest.raises(ValueError, match="did not advance"):
        model.propose(model.new_coordinator(1), mx.array([[1]], dtype=mx.int64))


def test_qwen4_commit_materializes_logits_before_publication(monkeypatch) -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)
    candidate = model.propose(coordinator, mx.array([[1]], dtype=mx.int64))
    original = coordinator.state
    expected_logits = np.asarray(candidate.logits).copy()

    monkeypatch.setattr(model, "validate_state", lambda state: None)

    def fail_eval(*arrays) -> None:
        assert np.array_equal(np.asarray(arrays[0]), expected_logits)
        raise RuntimeError("lazy model failure")

    monkeypatch.setattr("moespresso.runtime.qwen4.model.mx.eval", fail_eval)
    with pytest.raises(RuntimeError, match="lazy model failure"):
        coordinator.commit(candidate, 1)

    assert coordinator.state is original
    assert candidate.consumed is False


def test_qwen4_commit_rejects_corrupt_checkpoint_identity() -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)
    candidate = model.propose(coordinator, mx.array([[1]], dtype=mx.int64))
    candidate.checkpoints = (
        replace(candidate.checkpoints[0], cache_identity="another-checkpoint"),
    )

    with pytest.raises(ValueError, match="does not match its token frontier"):
        coordinator.commit(candidate, 1)

    assert candidate.consumed is False
    assert coordinator.state.frontier == 0


def test_qwen4_state_rejects_position_history_with_wrong_geometry() -> None:
    model = _model([])
    coordinator = model.new_coordinator(1)
    candidate = model.propose(coordinator, mx.array([[1]], dtype=mx.int64))
    committed = coordinator.commit(candidate, 1)
    corrupt = replace(
        committed,
        position_history=mx.full((2, 1, 1), 7, dtype=mx.int64),
    )

    with pytest.raises(ValueError, match="does not share the public frontier"):
        model.validate_state(corrupt)
