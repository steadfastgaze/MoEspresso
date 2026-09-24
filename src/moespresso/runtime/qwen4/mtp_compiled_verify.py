"""Product-state bridge for short-context compiled Qwen4 MTP verification."""

from dataclasses import replace

import mlx.core as mx

from moespresso.runtime.qwen4.gdn import Qwen4GDNState
from moespresso.runtime.qwen4.model import (
    Qwen4Candidate,
    Qwen4LayerState,
    _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
    _QWEN4_BATCHED_UNDO_CAPABILITY,
    _Qwen4EvaluatedBoundary,
)
from moespresso.runtime.qwen4.mtp_compiled_core import (
    QWEN4_MTP_COMPILED_MAX_BASE_FRONTIER,
    Qwen4MTPCompiledCore,
)
from moespresso.runtime.qwen4.mtp_qsa import Qwen4MTPQSARows
from moespresso.runtime.qwen4.mtp_verify import Qwen4MTPVerifier
from moespresso.runtime.qwen4.ple import Qwen4PLEState
from moespresso.runtime.qwen4.qsa import (
    QWEN38_QSA_COMPRESS_RATIO,
    QWEN38_QSA_TOKEN_BUDGET,
    _QSA_TRUSTED_ALL_VALID_CAPABILITY,
    qsa_causal_prefix_layout,
)
from moespresso.runtime.qwen4.qsa_kvarn import Qwen4MutableKVarNQSAState


_MODEL_CORE_ATTRIBUTE = "_moespresso_qwen4_mtp_compiled_core"


def _model_core(model):
    core = getattr(model, _MODEL_CORE_ATTRIBUTE, None)
    if core is None:
        core = Qwen4MTPCompiledCore(model)
        object.__setattr__(model, _MODEL_CORE_ATTRIBUTE, core)
    elif not isinstance(core, Qwen4MTPCompiledCore) or core.model is not model:
        raise RuntimeError("cached compiled MTP core belongs to another model")
    return core


def _undo_for(certificate, mixer, source):
    matches = [
        undo
        for allowed, allowed_source, undo in zip(
            certificate.allowed_mixers,
            certificate.source_states,
            certificate.prepared_undos,
            strict=True,
        )
        if allowed is mixer and allowed_source is source
    ]
    if len(matches) != 1 or matches[0] is None:
        raise RuntimeError("compiled MTP verification requires one prepared QSA undo")
    return matches[0]


class Qwen4MTPCompiledVerifier(Qwen4MTPVerifier):
    """Compile eligible target math and retain the existing atomic state commit."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.compiled_rounds = 0
        self.fallback_rounds = 0
        self._compiled_states = None
        self._compiled_state_key = None
        self._pending_compiled_states = None
        self._round_kind = None

    def compiled_verification_stats(self):
        return {
            "compiled_rounds": self.compiled_rounds,
            "fallback_rounds": self.fallback_rounds,
        }

    def shared_expert_call_count(self):
        fallback_calls = sum(pair.paired_calls for pair in self.expert_pairs)
        return self.compiled_rounds * len(self.model.layers) + fallback_calls

    def _clear_compiled_state(self):
        self._compiled_states = None
        self._compiled_state_key = None
        self._pending_compiled_states = None
        self._round_kind = None

    @staticmethod
    def _state_key(coordinator, state):
        return coordinator._identity, state.cache_identity, state.revision, state.frontier

    def _compiled_state_eligible(self, state):
        if (
            state.frontier <= 0
            or state.frontier > QWEN4_MTP_COMPILED_MAX_BASE_FRONTIER
            or not self._full_resident_prefix_enabled
            or not self.shared_projections
            or not all(pair.native_boundaries for pair in self.gdn_pairs.values())
        ):
            return False
        return all(
            layer.mixer_kind != "qsa"
            or (
                isinstance(item.mixer_state, Qwen4MutableKVarNQSAState)
                and item.mixer_state.view.frontier == state.frontier
                and layer.mixer.module.token_budget == QWEN38_QSA_TOKEN_BUDGET
                and layer.mixer.module.compress_ratio == QWEN38_QSA_COMPRESS_RATIO
            )
            for layer, item in zip(self.model.layers, state.layers, strict=True)
        )

    def verify(self, coordinator, input_ids, *, cancelled=None):
        try:
            result = super().verify(coordinator, input_ids, cancelled=cancelled)
        except BaseException:
            self._pending_compiled_states = None
            self._round_kind = None
            raise
        kind = self._round_kind
        self._round_kind = None
        if kind == "compiled":
            first, final = self._pending_compiled_states
            self._compiled_states = final if result.acceptance.accepted else first
            self._compiled_state_key = self._state_key(coordinator, result.state)
            self.compiled_rounds += 1
        elif kind == "fallback":
            self.fallback_rounds += 1
        self._pending_compiled_states = None
        return result

    def close(self):
        super().close()
        self._clear_compiled_state()

    def _forward_pair(
        self,
        coordinator,
        base,
        input_ids,
        valid,
        positions,
        roots,
        cancelled,
    ):
        if not self._compiled_state_eligible(base):
            self._clear_compiled_state()
            self._round_kind = "fallback"
            return super()._forward_pair(
                coordinator, base, input_ids, valid, positions, roots, cancelled
            )

        core = _model_core(self.model)
        state_key = self._state_key(coordinator, base)
        if self._compiled_states is None or self._compiled_state_key != state_key:
            self._compiled_states = core.state_from_product(base)
            self._compiled_state_key = state_key
        histories = tuple(
            mx.concatenate([base.valid_history, valid[:, :count]], axis=1)
            for count in (1, 2)
        )
        position_histories = tuple(
            mx.concatenate([base.position_history, positions[:, :, :count]], axis=2)
            for count in (1, 2)
        )
        contexts = tuple(
            () if item.ple_state is None else item.ple_state.token_context
            for item in base.layers
        )
        embeddings, next_contexts = core.ple_inputs(input_ids, contexts)
        qsa = Qwen4MTPQSARows(self.model, base, valid, positions, histories)
        if (
            not qsa.certified
            or qsa.first_certificate is None
            or qsa.first_certificate.append_finite_batch is None
            or not qsa.first_certificate.prepared_undos
        ):
            raise RuntimeError("compiled MTP verification requires transactional certified QSA")

        output = core.compiled(
            input_ids,
            positions,
            mx.array(base.frontier, dtype=mx.int32),
            self._compiled_states,
            embeddings,
        )
        logits, widened, first_core, final_core, raw_index_rows = output
        roots.append((output, embeddings, next_contexts))
        session = getattr(self.model, "_moespresso_pooled_decode_session", None)
        if session is not None:
            session.remember(roots)
        mx.async_eval(output)

        layer_states = ([], [])
        for index, (layer, previous) in enumerate(
            zip(self.model.layers, base.layers, strict=True)
        ):
            first_values = first_core[index]
            final_values = final_core[index]
            ple_states = (None, None)
            if layer.ple is not None:
                ple_states = tuple(
                    Qwen4PLEState(
                        next_contexts[index][row],
                        (first_values if row == 0 else final_values)[2],
                        base.frontier + row + 1,
                    )
                    for row in range(2)
                )
            if layer.mixer_kind == "gdn":
                mixer_states = tuple(
                    Qwen4GDNState(
                        values[0],
                        values[1],
                        base.frontier + row + 1,
                        schema=previous.mixer_state.schema,
                        recurrent_layout=previous.mixer_state.recurrent_layout,
                    )
                    for row, values in enumerate((first_values, final_values))
                )
            else:
                mixer_states = self._append_qsa_rows(
                    qsa,
                    index,
                    layer.mixer,
                    raw_index_rows[index],
                    final_values[0],
                    final_values[1],
                    histories,
                    positions,
                    base.frontier,
                )
            for row in range(2):
                layer_states[row].append(
                    Qwen4LayerState(
                        layer.mixer_kind,
                        mixer_states[row],
                        base.frontier + row + 1,
                        ple_states[row],
                    )
                )

        checkpoints = tuple(
            replace(
                base,
                frontier=base.frontier + row + 1,
                valid_history=histories[row],
                position_history=position_histories[row],
                layers=tuple(layer_states[row]),
            )
            for row in range(2)
        )
        for state in checkpoints:
            self.model._validate_trusted_state(state)
        self._pending_compiled_states = (first_core, final_core)
        self._round_kind = "compiled"
        candidate = Qwen4Candidate(
            coordinator._identity,
            base.cache_identity,
            base.revision,
            base.frontier,
            logits,
            checkpoints,
            trusted_lineage=True,
            widened=widened,
            _evaluated_boundary=_Qwen4EvaluatedBoundary(
                coordinator._identity,
                base.cache_identity,
                base.revision,
                base.frontier,
                checkpoints,
            ),
        )
        return candidate, qsa.seal()

    @staticmethod
    def _append_qsa_rows(
        qsa,
        index,
        mixer,
        raw_rows,
        keys,
        values,
        histories,
        positions,
        frontier,
    ):
        states = []
        state = qsa.sources[index]
        backend = mixer.state_backend
        module = mixer.module
        for row in range(2):
            certificate = (
                qsa.first_certificate
                if row == 0
                else qsa._certificate((mixer,), (state,), row)
            )
            if certificate is None or certificate.append_finite_batch is None:
                raise RuntimeError("compiled MTP verification requires deferred finite append")
            undo = _undo_for(certificate, mixer, state)
            row_raw, row_keys, row_values, row_position = (
                backend._bind_trusted_undo_dependencies(
                    undo,
                    (
                        raw_rows[:, row : row + 1],
                        keys[:, :, frontier + row : frontier + row + 1],
                        values[:, :, frontier + row : frontier + row + 1],
                        positions[:, :, row : row + 1],
                    ),
                    capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
                )
            )
            layout = qsa_causal_prefix_layout(
                histories[row],
                mx.full((1, 1), frontier + row, dtype=mx.int32),
            )
            prepared = backend._prepare_index_certified(
                state,
                raw_index_keys=row_raw,
                position_ids=row_position,
                layout=layout,
                key_norm=module.indexer.k_layernorm,
                rotary_dim=module.rotary_dim,
                rope_base=module.rope_base,
                mrope_section=module.mrope_section,
                capability=_QSA_TRUSTED_ALL_VALID_CAPABILITY,
            )
            result, predicate = backend._append_with_prepared_undo_deferred_finite(
                state,
                keys=row_keys,
                values=row_values,
                valid_tokens=qsa.valid[row],
                index_preparation=prepared,
                undo_reservation=undo,
                capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
            )
            certificate.append_finite_batch.register(
                mixer=mixer,
                source_state=state,
                result_state=result,
                predicate=predicate,
                capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
            )
            qsa.seen.add((index, row))
            states.append(mixer.snapshot_state(result))
            if row == 0:
                qsa.first_results[index] = result
            else:
                qsa.checks.append(
                    certificate.append_finite_batch.seal(
                        ((mixer, result),),
                        capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
                    )
                )
            state = result
        return tuple(states)


__all__ = ["Qwen4MTPCompiledVerifier"]
