"""Released Qwen4-Exp gated-delta block on the pinned MLX recurrence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops
from mlx_lm.models.qwen3_5 import GatedDeltaNet

from moespresso.runtime.qwen4.model import Qwen4MixerOutput
from moespresso.runtime.qwen4.primitives import (
    Qwen4SigmoidRMSNormGated,
    qwen4_projection_compute_dtype,
)


_GDN_STATE_SCHEMA = "qwen4_gdn_state_v1"
_GDN_RECURRENT_LAYOUT = "batch_value-head_value-width_key-width"
GDN_BOUNDARY_COUNTER_FIELDS = (
    "native_calls",
    "generic_calls",
    "fallback_input",
    "fallback_contract",
)
_GDN_CALL_COUNTS = dict.fromkeys(GDN_BOUNDARY_COUNTER_FIELDS, 0)
GDN_PREROUTER_COUNTER_FIELDS = (
    "native_calls",
    "fallback_input",
    "fallback_mask",
    "fallback_contract",
)
_GDN_PREROUTER_CALL_COUNTS = dict.fromkeys(GDN_PREROUTER_COUNTER_FIELDS, 0)


def qwen4_gdn_call_counts() -> dict[str, int]:
    """Return GDN native-boundary engagement and fallback counts."""
    return dict(_GDN_CALL_COUNTS)


def qwen4_gdn_prerouter_call_counts() -> dict[str, int]:
    """Return native pre-router engagement and fallback counts."""
    return dict(_GDN_PREROUTER_CALL_COUNTS)


@dataclass(frozen=True)
class Qwen4GDNPreRouterResult:
    """Outputs from the fixed one-token GDN envelope before expert routing."""

    mlp_hidden: mx.array
    residual: mx.array
    injection: mx.array
    state: "Qwen4GDNState"
    frontier: int


def _released_kquant_projection(
    module: Any,
    *,
    codec: str,
    shape: tuple[int, int],
) -> bool:
    return (
        getattr(module, "mode", None) == "kquant"
        and getattr(module, "kquant_type", None) == codec
        and "bias" not in module
        and isinstance(getattr(module, "weight", None), mx.array)
        and tuple(module.weight.shape) == shape
        and isinstance(getattr(module, "scales", None), mx.array)
        and module.scales.dtype == mx.uint8
        and module.scales.size == 1
    )


def _released_hc_contract(module: Any) -> bool:
    norm = getattr(module, "hc_norm", None)
    injection = getattr(module, "block_inject_weight", None)
    return (
        getattr(module, "hidden_size", None) == 2560
        and getattr(module, "branch_count", None) == 4
        and getattr(module, "lowrank_size", None) == 320
        and getattr(module, "expanded_size", None) == 10240
        and isinstance(getattr(norm, "weight", None), mx.array)
        and tuple(norm.weight.shape) == (10240,)
        and norm.weight.dtype == mx.bfloat16
        and isinstance(getattr(norm, "eps", None), (float, int))
        and _released_kquant_projection(
            module.input_mix_weight_down,
            codec="q6_k",
            shape=(320, 8400),
        )
        and injection is not None
        and _released_kquant_projection(
            injection,
            codec="q6_k",
            shape=(4, 8400),
        )
        and _released_kquant_projection(
            module.input_mix_weight_up,
            codec="q8_0",
            shape=(10240, 340),
        )
    )


def _qwen4_gdn_prerouter_contract(layer: Any, state: Any) -> bool:
    adapter = getattr(layer, "mixer", None)
    module = getattr(adapter, "module", None)
    attention = getattr(layer, "attention_residual", None)
    mlp = getattr(layer, "mlp_residual", None)
    if module is None or not _released_hc_contract(attention) or not _released_hc_contract(mlp):
        return False
    projections = (
        (module.in_proj_qkv, (10240, 2100)),
        (module.in_proj_z, (6144, 2100)),
        (module.in_proj_b, (48, 2100)),
        (module.in_proj_a, (48, 2100)),
        (module.out_proj, (2560, 5040)),
    )
    norm_eps = (
        float(attention.hc_norm.eps),
        float(mlp.hc_norm.eps),
        float(module.norm.eps),
    )
    return (
        getattr(layer, "mixer_kind", None) == "gdn"
        and all(
            _released_kquant_projection(projection, codec="q6_k", shape=shape)
            for projection, shape in projections
        )
        and _qwen4_gdn_native_contract(module, [state.conv_state, state.recurrent_state])
        and norm_eps[0] == norm_eps[1] == norm_eps[2]
    )


def qwen4_gdn_prerouter_step(
    layer: Any,
    hidden_states: mx.array,
    *,
    state: "Qwen4GDNState | None",
    pending_output: mx.array | None,
    pending_injection: mx.array | None,
    certified_all_valid: bool,
) -> Qwen4GDNPreRouterResult | None:
    """Run the fixed one-token pre-router envelope when its contract is certified."""
    if (
        tuple(hidden_states.shape) != (1, 1, 10240)
        or hidden_states.dtype != mx.bfloat16
        or (pending_output is None) != (pending_injection is None)
    ):
        _GDN_PREROUTER_CALL_COUNTS["fallback_input"] += 1
        return None
    if not certified_all_valid:
        _GDN_PREROUTER_CALL_COUNTS["fallback_mask"] += 1
        return None
    if state is None or not _qwen4_gdn_prerouter_contract(layer, state):
        _GDN_PREROUTER_CALL_COUNTS["fallback_contract"] += 1
        return None
    try:
        from mlx_kquant import qwen4_gdn_prerouter_q6
    except ImportError:
        _GDN_PREROUTER_CALL_COUNTS["fallback_contract"] += 1
        return None
    if not callable(qwen4_gdn_prerouter_q6):
        _GDN_PREROUTER_CALL_COUNTS["fallback_contract"] += 1
        return None

    attention = layer.attention_residual
    mlp = layer.mlp_residual
    module = layer.mixer.module
    outputs = qwen4_gdn_prerouter_q6(
        hidden_states,
        attention.hc_norm.weight,
        attention.input_mix_weight_down.weight,
        attention.input_mix_weight_down.scales,
        attention.block_inject_weight.weight,
        attention.block_inject_weight.scales,
        attention.input_mix_weight_up.weight,
        attention.input_mix_weight_up.scales,
        mlp.hc_norm.weight,
        mlp.input_mix_weight_down.weight,
        mlp.input_mix_weight_down.scales,
        mlp.block_inject_weight.weight,
        mlp.block_inject_weight.scales,
        mlp.input_mix_weight_up.weight,
        mlp.input_mix_weight_up.scales,
        module.in_proj_qkv.weight,
        module.in_proj_qkv.scales,
        module.in_proj_z.weight,
        module.in_proj_z.scales,
        module.in_proj_b.weight,
        module.in_proj_b.scales,
        module.in_proj_a.weight,
        module.in_proj_a.scales,
        state.conv_state,
        module.conv1d.weight,
        module.A_log,
        module.dt_bias,
        state.recurrent_state,
        module.norm.weight,
        module.out_proj.weight,
        module.out_proj.scales,
        pending_output,
        pending_injection,
        eps=float(module.norm.eps),
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise RuntimeError("native Qwen GDN pre-router returned an invalid result")
    mlp_hidden, residual, injection, conv_state, recurrent_state = outputs
    next_state = Qwen4GDNState(
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        offset=state.offset + 1,
    )
    _GDN_PREROUTER_CALL_COUNTS["native_calls"] += 1
    return Qwen4GDNPreRouterResult(
        mlp_hidden=mlp_hidden,
        residual=residual,
        injection=injection,
        state=next_state,
        frontier=next_state.offset,
    )


def _qwen4_gdn_native_input_supported(module: Any, inputs: mx.array) -> bool:
    return (
        tuple(inputs.shape) == (1, 1, 2560)
        and inputs.dtype == mx.bfloat16
        and getattr(module, "hidden_size", None) == 2560
    )


def _qwen4_gdn_native_contract(module: Any, cache: Any) -> bool:
    if (
        getattr(module, "training", True)
        or getattr(module, "sharding_group", None) is not None
        or cache is None
        or getattr(cache, "lengths", None) is not None
        or getattr(cache, "left_padding", None) is not None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return False
    geometry = {
        "conv_dim": 10240,
        "conv_kernel_size": 4,
        "num_k_heads": 16,
        "num_v_heads": 48,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "key_dim": 2048,
        "value_dim": 6144,
    }
    if any(getattr(module, name, None) != expected for name, expected in geometry.items()):
        return False
    try:
        conv_state = cache[0]
        recurrent_state = cache[1]
    except (IndexError, KeyError, TypeError):
        return False
    conv_weight = getattr(getattr(module, "conv1d", None), "weight", None)
    norm = getattr(module, "norm", None)
    norm_weight = getattr(norm, "weight", None)
    a_log = getattr(module, "A_log", None)
    dt_bias = getattr(module, "dt_bias", None)
    return (
        _array_contract(conv_state, (1, 3, 10240), mx.bfloat16)
        and _array_contract(recurrent_state, (1, 48, 128, 128), mx.float32)
        and _array_contract(conv_weight, (10240, 4, 1), mx.bfloat16)
        and _array_contract(norm_weight, (128,), mx.bfloat16)
        and _array_contract(a_log, (48,), mx.bfloat16)
        and _array_contract(dt_bias, (48,), mx.bfloat16)
        and isinstance(getattr(norm, "eps", None), (float, int))
        and norm.eps > 0
    )


def _qwen4_gdn_projected_contract(
    qkv: mx.array,
    gate: mx.array,
    beta_logits: mx.array,
    decay_logits: mx.array,
) -> bool:
    return (
        _array_contract(qkv, (1, 1, 10240), mx.bfloat16)
        and _array_contract(gate, (1, 1, 48, 128), mx.bfloat16)
        and _array_contract(beta_logits, (1, 1, 48), mx.bfloat16)
        and _array_contract(decay_logits, (1, 1, 48), mx.bfloat16)
    )


def _array_contract(value: Any, shape: tuple[int, ...], dtype: mx.Dtype) -> bool:
    return isinstance(value, mx.array) and tuple(value.shape) == shape and value.dtype == dtype


def _qwen4_gdn_native_helpers():
    try:
        from mlx_kquant import qwen4_gdn_norm_gate, qwen4_gdn_prepare
    except ImportError:
        return None
    if not callable(qwen4_gdn_prepare) or not callable(qwen4_gdn_norm_gate):
        return None
    return qwen4_gdn_prepare, qwen4_gdn_norm_gate


def _qwen4_gdn_record_generic(reason: str) -> None:
    _GDN_CALL_COUNTS["generic_calls"] += 1
    _GDN_CALL_COUNTS[reason] += 1


@dataclass(frozen=True)
class Qwen4GDNState:
    """Explicit GDN continuation state at one physical token frontier."""

    conv_state: mx.array
    recurrent_state: mx.array
    offset: int
    schema: str = _GDN_STATE_SCHEMA
    recurrent_layout: str = _GDN_RECURRENT_LAYOUT


class Qwen4GatedDeltaNet(GatedDeltaNet):
    """Qwen3.5-compatible recurrence with Qwen4 masking and sigmoid output gate."""

    def __init__(self, config):
        super().__init__(config)
        self.norm = Qwen4SigmoidRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
        )

    def __call__(
        self, inputs: mx.array, mask: mx.array | None = None, cache=None,
        *, prefix_states: list[tuple[mx.array, mx.array]] | None = None,
    ) -> mx.array:
        if prefix_states is not None and (
            inputs.shape[1] != 2 or cache is None or cache.lengths is not None or prefix_states
        ):
            raise ValueError("GDN prefix capture requires two rows and an empty checkpoint list")
        if mask is not None:
            inputs = mx.where(mask[..., None], inputs, 0)
        batch_size, sequence_length, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        gate = self.in_proj_z(inputs).reshape(
            batch_size,
            sequence_length,
            self.num_v_heads,
            self.head_v_dim,
        )
        beta_logits = self.in_proj_b(inputs)
        decay_logits = self.in_proj_a(inputs)

        native_helpers = None
        if not _qwen4_gdn_native_input_supported(self, inputs):
            fallback = "fallback_input"
        elif not _qwen4_gdn_native_contract(self, cache) or not _qwen4_gdn_projected_contract(
            qkv,
            gate,
            beta_logits,
            decay_logits,
        ):
            fallback = "fallback_contract"
        else:
            native_helpers = _qwen4_gdn_native_helpers()
            fallback = "fallback_contract" if native_helpers is None else None

        if native_helpers is not None:
            qwen4_gdn_prepare, qwen4_gdn_norm_gate = native_helpers
            query, key, value, beta, decay, next_conv_state = qwen4_gdn_prepare(
                qkv,
                beta_logits,
                decay_logits,
                cache[0],
                self.conv1d.weight,
                self.A_log,
                self.dt_bias,
            )
            output, next_recurrent_state = gated_delta_kernel(
                query,
                key,
                value,
                decay,
                beta,
                cache[1],
                None,
            )
            cache[0] = next_conv_state
            cache[1] = next_recurrent_state
            cache.advance(sequence_length)
            output = qwen4_gdn_norm_gate(
                output,
                gate.reshape(batch_size, sequence_length, -1),
                self.norm.weight,
                eps=self.norm.eps,
            )
            output = self.out_proj(output.reshape(batch_size, sequence_length, -1))
            _GDN_CALL_COUNTS["native_calls"] += 1
            return output

        if fallback is None:
            raise RuntimeError("GDN boundary routing produced no implementation")
        _qwen4_gdn_record_generic(fallback)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (batch_size, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, sequence_length)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -keep:, :])

        conv_output = nn.silu(self.conv1d(conv_input))
        query, key, value = [
            tensor.reshape(batch_size, sequence_length, heads, width)
            for tensor, heads, width in zip(
                mx.split(conv_output, [self.key_dim, 2 * self.key_dim], axis=-1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]

        query = _l2_normalize(query)
        key = _l2_normalize(key)
        query = query * (self.head_k_dim**-0.5)
        state = cache[1] if cache else None
        recurrent_prefix = [] if prefix_states is not None else None
        capture = {} if recurrent_prefix is None else {"prefix_states": recurrent_prefix}
        output, state = _qwen4_gated_delta_update(
            query,
            key,
            value,
            decay_logits,
            beta_logits,
            self.A_log,
            self.dt_bias,
            state,
            use_kernel=not self.training,
            **capture,
        )
        if prefix_states is not None:
            prefix_states.append((
                mx.contiguous(conv_input[:, 1:self.conv_kernel_size, :]), recurrent_prefix[0],
            ))

        if cache is not None:
            cache[1] = state
            cache.advance(sequence_length)

        output = self.norm(output, gate)
        output = self.out_proj(output.reshape(batch_size, sequence_length, -1))
        if self.sharding_group is not None:
            output = mx.distributed.all_sum(output, group=self.sharding_group)
        return output


class Qwen4GDNAdapter(nn.Module):
    """Bind explicit Qwen4 GDN state to the model-shell mixer protocol."""

    def __init__(self, module: Qwen4GatedDeltaNet):
        super().__init__()
        self.module = module

    def fork_state(self, state: Qwen4GDNState | None) -> Qwen4GDNState | None:
        return self._copy_state(state)

    def snapshot_state(self, state: Qwen4GDNState | None) -> Qwen4GDNState | None:
        return self._copy_state(state)

    def validate_state(
        self,
        state: Qwen4GDNState | None,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        if (
            isinstance(expected_frontier, bool)
            or not isinstance(expected_frontier, int)
            or expected_frontier < 0
        ):
            raise ValueError("GDN frontier must be a nonnegative integer")
        if (
            position_history.ndim != 3
            or position_history.shape[0] != 3
            or position_history.shape[-1] != expected_frontier
        ):
            raise ValueError("GDN position history does not share the public frontier")
        batch_size = position_history.shape[1]
        if expected_frontier == 0:
            if state is not None:
                raise ValueError("GDN state must be empty at frontier zero")
            return
        if not isinstance(state, Qwen4GDNState):
            raise ValueError("GDN state is missing at a live frontier")
        if state.schema != _GDN_STATE_SCHEMA or state.recurrent_layout != _GDN_RECURRENT_LAYOUT:
            raise ValueError("GDN state schema or recurrent layout is incompatible")
        if state.offset != expected_frontier:
            raise ValueError("GDN state is off the public frontier")
        conv_shape = (
            batch_size,
            self.module.conv_kernel_size - 1,
            self.module.conv_dim,
        )
        recurrent_shape = (
            batch_size,
            self.module.num_v_heads,
            self.module.head_v_dim,
            self.module.head_k_dim,
        )
        if state.conv_state.shape != conv_shape:
            raise ValueError("GDN convolution state has incompatible geometry")
        if state.recurrent_state.shape != recurrent_shape:
            raise ValueError("GDN recurrent state has incompatible geometry")
        if state.conv_state.dtype != qwen4_projection_compute_dtype(self.module.in_proj_qkv):
            raise ValueError("GDN convolution state has incompatible dtype")
        if state.recurrent_state.dtype != mx.float32:
            raise ValueError("GDN recurrent state must be FP32")

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state: Qwen4GDNState | None,
    ) -> Qwen4MixerOutput:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.module.hidden_size:
            raise ValueError("GDN hidden states have incompatible geometry")
        batch_size, token_count, _ = hidden_states.shape
        if hidden_states.dtype != qwen4_projection_compute_dtype(self.module.in_proj_qkv):
            raise ValueError("GDN hidden states must match the projection dtype")
        if valid_tokens.shape != (batch_size, token_count) or valid_tokens.dtype != mx.bool_:
            raise ValueError("GDN valid-token mask has incompatible geometry")
        current_frontier = 0 if state is None else state.offset
        if visible_history.shape != (batch_size, current_frontier + token_count):
            raise ValueError("GDN visible history does not reach the proposed frontier")
        if visible_history.dtype != mx.bool_:
            raise ValueError("GDN visible history must be boolean")
        if position_ids.shape != (3, batch_size, token_count):
            raise ValueError("GDN position ids have incompatible geometry")
        current_positions = mx.zeros((3, batch_size, current_frontier), dtype=position_ids.dtype)
        self.validate_state(
            state,
            expected_frontier=current_frontier,
            position_history=current_positions,
        )

        cache = ArraysCache(size=2)
        if state is not None:
            cache.state = [state.conv_state, state.recurrent_state]
        output = self.module(hidden_states, mask=valid_tokens, cache=cache)
        conv_state, recurrent_state = cache.state
        next_frontier = current_frontier + token_count
        next_state = Qwen4GDNState(
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            offset=next_frontier,
        )
        next_positions = mx.zeros((3, batch_size, next_frontier), dtype=position_ids.dtype)
        self.validate_state(
            next_state,
            expected_frontier=next_frontier,
            position_history=next_positions,
        )
        return Qwen4MixerOutput(
            output=output,
            state=next_state,
            frontier=next_frontier,
        )

    @staticmethod
    def _copy_state(state: Qwen4GDNState | None) -> Qwen4GDNState | None:
        if state is None:
            return None
        return Qwen4GDNState(
            conv_state=mx.array(state.conv_state),
            recurrent_state=mx.array(state.recurrent_state),
            offset=state.offset,
            schema=state.schema,
            recurrent_layout=state.recurrent_layout,
        )


def _l2_normalize(values: mx.array, *, eps: float = 1e-6) -> mx.array:
    """Normalize with the released sum-of-squares epsilon contract."""
    return values * mx.rsqrt(mx.sum(mx.square(values), axis=-1, keepdims=True) + eps)


def _qwen4_gated_delta_update(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    decay_logits: mx.array,
    beta_logits: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    *,
    use_kernel: bool,
    prefix_states: list[mx.array] | None = None,
) -> tuple[mx.array, mx.array]:
    """Apply the released FP32 decay parametrization and MLX recurrence."""
    beta = mx.sigmoid(beta_logits)
    decay = _decay_multiplier(decay_logits, a_log, dt_bias)
    if state is None:
        batch_size, _, _, key_width = query.shape
        value_heads, value_width = value.shape[-2:]
        state = mx.zeros(
            (batch_size, value_heads, value_width, key_width),
            dtype=mx.float32,
        )
    update = (gated_delta_ops
              if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available()
              else gated_delta_kernel)
    if prefix_states is None:
        return update(query, key, value, decay, beta, state, None)
    if query.shape[1] != 2 or prefix_states:
        raise ValueError("GDN recurrent prefix capture requires exactly two rows")
    first, checkpoint = update(query[:, :1], key[:, :1], value[:, :1], decay[:, :1], beta[:, :1], state, None)
    second, final = update(query[:, 1:], key[:, 1:], value[:, 1:], decay[:, 1:], beta[:, 1:], checkpoint, None)
    prefix_states.append(checkpoint)
    return mx.concatenate([first, second], axis=1), final


def _decay_multiplier(
    decay_logits: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
) -> mx.array:
    """Evaluate both decay-gate operands in FP32 before softplus."""
    return mx.exp(
        -mx.exp(a_log.astype(mx.float32))
        * nn.softplus(decay_logits.astype(mx.float32) + dt_bias.astype(mx.float32))
    )


__all__ = [
    "GDN_BOUNDARY_COUNTER_FIELDS",
    "GDN_PREROUTER_COUNTER_FIELDS",
    "Qwen4GDNAdapter",
    "Qwen4GDNPreRouterResult",
    "Qwen4GDNState",
    "Qwen4GatedDeltaNet",
    "qwen4_gdn_call_counts",
    "qwen4_gdn_prerouter_call_counts",
    "qwen4_gdn_prerouter_step",
]
