"""Compiled two-row Qwen4 target math for the exact short-context region."""

import math

import mlx.core as mx
import mlx.nn as nn

from moespresso.runtime.qwen4.mtp_full_resident import Qwen4MTPFullResidentExpertPair
from moespresso.runtime.qwen4.mtp_gdn import Qwen4MTPGDNPair
from moespresso.runtime.qwen4.mtp_qsa_projections import prepare_mtp_qsa_projection_rows
from moespresso.runtime.qwen4.mtp_verify import _read
from moespresso.runtime.qwen4.ple import lookup_ple_rows
from moespresso.runtime.qwen4.primitives import gated_residual_write
from moespresso.runtime.qwen4.qsa import (
    QWEN38_QSA_COMPRESS_RATIO, QWEN38_QSA_TOKEN_BUDGET,
    qsa_attention_from_selected_rows,
)


QWEN4_MTP_COMPILED_SELECTED_WIDTH = QWEN38_QSA_TOKEN_BUDGET + QWEN38_QSA_COMPRESS_RATIO - 1
QWEN4_MTP_COMPILED_MAX_BASE_FRONTIER = QWEN38_QSA_TOKEN_BUDGET - 2


def _append_and_gather_qsa_row(keys, values, row_keys, row_values, offset):
    """Functionally append one row and expose a fixed-width visible prefix."""

    start = offset.reshape(1)
    keys = mx.slice_update(keys, row_keys, start, axes=(2,))
    values = mx.slice_update(values, row_values, start, axes=(2,))
    valid = (mx.arange(keys.shape[2]) < offset + 1).reshape(1, 1, -1)
    mask = valid[:, :, None, :, None]
    return keys, values, mx.where(mask, keys[:, None], 0), mx.where(
        mask, values[:, None], 0
    ), valid


class Qwen4MTPCompiledCore:
    """Pure fixed-shape numerical core with explicit request-state inputs."""

    def __init__(self, model):
        self.model = model
        self.width = QWEN4_MTP_COMPILED_SELECTED_WIDTH
        self.trace_count = 0
        self.gdn = {
            index: Qwen4MTPGDNPair(layer)
            for index, layer in enumerate(model.layers)
            if layer.mixer_kind == "gdn"
        }
        self.experts = tuple(
            Qwen4MTPFullResidentExpertPair(layer.mlp) for layer in model.layers
        )
        self.compiled = mx.compile(self.forward)

    def state_from_product(self, state):
        """Materialize one private fixed-shape bank from committed product state."""

        if not 0 < state.frontier <= QWEN4_MTP_COMPILED_MAX_BASE_FRONTIER:
            raise ValueError("compiled MTP core requires a live short-context frontier")
        states = []
        for layer, item in zip(self.model.layers, state.layers, strict=True):
            mixer = item.mixer_state
            if layer.mixer_kind == "gdn":
                values = (mixer.conv_state, mixer.recurrent_state)
            else:
                selected = mx.arange(self.width, dtype=mx.int32).reshape(1, 1, -1)
                selected = mx.where(selected < state.frontier, selected, -1)
                keys, cached_values, _ = mixer.storage.gather_selected_rows(selected)
                values = (mx.contiguous(keys[:, 0]), mx.contiguous(cached_values[:, 0]))
            if layer.ple is not None:
                values += (item.ple_state.conv_state,)
            states.append(values)
        mx.eval(states)
        return tuple(states)

    def ple_inputs(self, input_ids, contexts):
        """Resolve provider-backed PLE rows before entering the compiled graph."""

        embeddings, next_contexts = [], []
        for layer, context in zip(self.model.layers, contexts, strict=True):
            if layer.ple is None:
                embeddings.append(())
                next_contexts.append(())
                continue
            hashed = layer.ple.hasher(input_ids, previous_context=context)
            embeddings.append(
                lookup_ple_rows(
                    layer.ple.provider,
                    hashed.row_ids,
                    row_width=layer.ple.row_width,
                )
            )
            history = mx.concatenate((context, input_ids), axis=1)
            width = layer.ple.hasher.context_length
            next_contexts.append(
                tuple(history[:, row + 1 : row + 1 + width] for row in range(2))
            )
        mx.eval(embeddings, next_contexts)
        return tuple(embeddings), tuple(next_contexts)

    @staticmethod
    def _ple(layer, hidden, embedding, conv):
        ple = layer.ple
        key = ple.norm_key(ple.key_proj(embedding)).reshape(
            1, 1, ple.branch_count, ple.hidden_size
        )
        value = ple.value_proj(embedding)
        query = ple.norm_query(hidden).reshape(1, 1, ple.branch_count, ple.hidden_size)
        gate = mx.sum(key * query, axis=-1, keepdims=True) / math.sqrt(ple.hidden_size)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        flattened = (mx.sigmoid(gate) * value[:, :, None, :]).reshape(
            1, 1, ple.expanded_size
        )
        normalized = ple.norm_conv(flattened)
        history = mx.concatenate((conv, normalized), axis=1)
        output = flattened + nn.silu(ple.conv1d(history))
        next_conv = (
            history[:, -ple.short_conv_state_len :]
            if ple.short_conv_state_len
            else history[:, :0]
        )
        return hidden + output, next_conv

    def forward(self, input_ids, positions, offset, states, embeddings):
        """Return logits, widened rows, and both candidate state checkpoints."""

        self.trace_count += 1
        hidden = mx.tile(self.model.embedding(input_ids), (1, 1, self.model.branch_count))
        pending = injection = None
        first_states, final_states, raw_index_rows = [], [], []
        for index, (layer, state, experts) in enumerate(
            zip(self.model.layers, states, self.experts, strict=True)
        ):
            first_ple = final_ple = ()
            if layer.ple is not None:
                if pending is not None:
                    hidden = gated_residual_write(hidden, pending, injection)
                    pending = injection = None
                conv = state[2]
                rows, convs = [], []
                for row in range(2):
                    value, conv = self._ple(
                        layer,
                        hidden[:, row : row + 1],
                        embeddings[index][:, row : row + 1],
                        conv,
                    )
                    rows.append(value)
                    convs.append(conv)
                hidden = mx.concatenate(rows, axis=1)
                first_ple, final_ple = (convs[0],), (convs[1],)
            if index in self.gdn:
                values = self.gdn[index]._compute(
                    hidden, state[0], state[1], pending, injection
                )
                mixed, residual, weights, conv, recurrent, first_conv, first_recurrent = values
                first_states.append((first_conv, first_recurrent) + first_ple)
                final_states.append((conv, recurrent) + final_ple)
                raw_index_rows.append(())
            else:
                prepared = _read(layer.attention_residual, hidden, pending, injection)
                projections = prepare_mtp_qsa_projection_rows(
                    layer.mixer, prepared[0], positions
                )
                keys, cached_values = state[:2]
                mixed_rows, residual_rows, weight_rows = [], [], []
                key_states, value_states = [], []
                for row in range(2):
                    projected = projections.rows[row]
                    keys, cached_values, gathered_keys, gathered_values, valid = (
                        _append_and_gather_qsa_row(
                            keys,
                            cached_values,
                            projected.keys,
                            projected.values,
                            offset + row,
                        )
                    )
                    attention = qsa_attention_from_selected_rows(
                        projected.queries,
                        gathered_keys,
                        gathered_values,
                        valid,
                    )
                    gated = attention.reshape(1, 1, -1) * mx.sigmoid(projected.gate)
                    output = layer.mixer.module.o_proj(gated)
                    result = _read(
                        layer.mlp_residual,
                        prepared[1][:, row : row + 1],
                        output,
                        prepared[2][:, row : row + 1],
                    )
                    mixed_rows.append(result[0])
                    residual_rows.append(result[1])
                    weight_rows.append(result[2])
                    key_states.append(keys)
                    value_states.append(cached_values)
                mixed, residual, weights = (
                    mx.concatenate(rows, axis=1)
                    for rows in (mixed_rows, residual_rows, weight_rows)
                )
                first_states.append((key_states[0], value_states[0]) + first_ple)
                final_states.append((key_states[1], value_states[1]) + final_ple)
                raw_index_rows.append(
                    mx.concatenate(
                        [row.raw_index_keys for row in projections.rows], axis=1
                    )
                )
            output = experts(mixed)
            next_ple = (
                index + 1 < len(self.model.layers)
                and self.model.layers[index + 1].ple is not None
            )
            if index + 1 == len(self.model.layers) or next_ple:
                hidden = gated_residual_write(residual, output, weights)
                pending = injection = None
            else:
                hidden, pending, injection = residual, output, weights
        logits = self.model.lm_head(self.model.final_residual(hidden))
        return logits, hidden, tuple(first_states), tuple(final_states), tuple(raw_index_rows)


__all__ = [
    "QWEN4_MTP_COMPILED_MAX_BASE_FRONTIER",
    "QWEN4_MTP_COMPILED_SELECTED_WIDTH",
    "Qwen4MTPCompiledCore",
]
