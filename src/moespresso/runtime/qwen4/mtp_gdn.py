"""Compiled two-row GDN verification with an explicit first-row checkpoint."""

from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from moespresso.runtime.qwen4.gdn import (
    Qwen4GDNState, _GDN_RECURRENT_LAYOUT, _GDN_STATE_SCHEMA, _qwen4_gdn_prerouter_contract,
    gated_delta_kernel,
)


@dataclass(frozen=True)
class Qwen4MTPGDNPairOutput:
    mlp_hidden: mx.array
    residual: mx.array
    injection: mx.array
    first_state: Qwen4GDNState
    final_state: Qwen4GDNState


class Qwen4MTPGDNPair:
    """Batch the existing projections while retaining one-row recurrence states.

    The compiled graph has no frontier-shaped inputs, pool IO or publication.
    It owns one fixed layer's weights. Request frontiers remain outside the graph
    so growing context does not retrace this position-independent recurrence.
    """

    def __init__(self, layer, *, native_boundaries=True):
        self.layer = layer
        self.native_boundaries = native_boundaries
        self._plain = mx.compile(lambda h, c, r: self._compute(h, c, r))
        self._pending = mx.compile(lambda h, c, r, p, i: self._compute(h, c, r, p, i))

    def _compute(self, hidden, conv, recurrent, pending=None, pending_injection=None):
        layer = self.layer
        attention_rows = []
        for row in range(2):
            values = hidden[:, row : row + 1]
            if pending is None:
                result = layer.attention_residual(values)
            else:
                result = layer.attention_residual.read_with_pending(
                    values,
                    pending[:, row : row + 1],
                    pending_injection[:, row : row + 1],
                )
            attention_rows.append(result)
        mixed, residual, injection = (
            mx.concatenate([values[index] for values in attention_rows], axis=1)
            for index in range(3)
        )
        if self.native_boundaries:
            attention, conv, recurrent, first_conv, first_recurrent = self._native_attention(
                mixed, conv, recurrent,
            )
        else:
            cache = ArraysCache(size=2)
            cache.state = [conv, recurrent]
            checkpoints = []
            attention = layer.mixer.module(mixed, cache=cache, prefix_states=checkpoints)
            conv, recurrent = cache.state
            first_conv, first_recurrent = checkpoints[0]
        mlp_rows = [
            layer.mlp_residual.read_with_pending(
                residual[:, row : row + 1],
                attention[:, row : row + 1],
                injection[:, row : row + 1],
            )
            for row in range(2)
        ]
        mixed, residual, injection = (
            mx.concatenate([values[index] for values in mlp_rows], axis=1)
            for index in range(3)
        )
        return mixed, residual, injection, conv, recurrent, first_conv, first_recurrent

    def _native_attention(self, hidden, conv, recurrent):
        """Share projection reads while advancing fused recurrent cells in order."""
        from mlx_kquant import qwen4_gdn_norm_gate, qwen4_gdn_prepare

        module = self.layer.mixer.module
        qkv = module.in_proj_qkv(hidden)
        gate = module.in_proj_z(hidden)
        beta_logits = module.in_proj_b(hidden)
        decay_logits = module.in_proj_a(hidden)
        outputs, checkpoints = [], []
        for row in range(2):
            query, key, value, beta, decay, conv = qwen4_gdn_prepare(
                qkv[:, row:row + 1], beta_logits[:, row:row + 1], decay_logits[:, row:row + 1],
                conv, module.conv1d.weight, module.A_log, module.dt_bias,
            )
            output, recurrent = gated_delta_kernel(query, key, value, decay, beta, recurrent, None)
            outputs.append(qwen4_gdn_norm_gate(
                output, gate[:, row:row + 1], module.norm.weight, eps=float(module.norm.eps),
            ))
            checkpoints.append((conv, recurrent))
        attention = module.out_proj(mx.concatenate(outputs, axis=1))
        return attention, conv, recurrent, *checkpoints[0]

    def __call__(
        self, hidden: mx.array, *, state: Qwen4GDNState,
        pending_output: mx.array | None = None, pending_injection: mx.array | None = None,
    ) -> Qwen4MTPGDNPairOutput:
        if (hidden.shape != (1, 2, 10240) or hidden.dtype != mx.bfloat16
                or not isinstance(state, Qwen4GDNState) or type(state.offset) is not int or state.offset <= 0
                or state.schema != _GDN_STATE_SCHEMA or state.recurrent_layout != _GDN_RECURRENT_LAYOUT
                or not _qwen4_gdn_prerouter_contract(self.layer, state)
                or self.layer.mixer.module.training):
            raise ValueError("MTP GDN pair requires the released inference geometry and live state")
        if (pending_output is None) != (pending_injection is None):
            raise ValueError("MTP GDN pending output and injection must be supplied together")
        if pending_output is None:
            values = self._plain(hidden, state.conv_state, state.recurrent_state)
        else:
            if (pending_output.shape != (1, 2, 2560) or pending_output.dtype != mx.bfloat16
                    or pending_injection.shape != (1, 2, 4) or pending_injection.dtype != mx.bfloat16):
                raise ValueError("MTP GDN pending residual has an incompatible shape or dtype")
            values = self._pending(hidden, state.conv_state, state.recurrent_state,
                                   pending_output, pending_injection)
        mixed, residual, injection, conv, recurrent, first_conv, first_recurrent = values
        return Qwen4MTPGDNPairOutput(
            mixed, residual, injection,
            Qwen4GDNState(first_conv, first_recurrent, state.offset + 1),
            Qwen4GDNState(conv, recurrent, state.offset + 2),
        )
