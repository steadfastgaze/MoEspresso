"""Layer-major two-row verification under one pooled request owner."""

from dataclasses import dataclass, replace

import mlx.core as mx

from moespresso.runtime.deepseek_v4.spec_decode import AcceptResult, greedy_accept
from moespresso.runtime.pooled_moe import pooled_request_scope
from moespresso.runtime.qwen4.model import (
    Qwen4Candidate, Qwen4CompositeState, Qwen4LayerState,
    _Qwen4EvaluatedBoundary, _state_arrays,
)
from moespresso.runtime.qwen4.mtp_full_resident import Qwen4MTPFullResidentExpertPair
from moespresso.runtime.qwen4.mtp_gdn import Qwen4MTPGDNPair
from moespresso.runtime.qwen4.mtp_qsa import Qwen4MTPQSARows
from moespresso.runtime.qwen4.primitives import gated_residual_write


def _read(residual, hidden, pending=None, injection=None):
    if pending is None:
        return residual(hidden)
    pending_read = getattr(residual, "read_with_pending", None)
    if callable(pending_read):
        return pending_read(hidden, pending, injection)
    return residual(gated_residual_write(hidden, pending, injection))


@dataclass(frozen=True)
class Qwen4MTPVerification:
    """Committed target input rows and the next unconsumed anchor."""

    acceptance: AcceptResult
    logits: mx.array
    widened: mx.array
    state: Qwen4CompositeState


@dataclass(frozen=True)
class Qwen4MTPPlainVerification:
    """One committed ordinary target row."""

    logits: mx.array
    widened: mx.array
    state: Qwen4CompositeState


class Qwen4MTPVerifier:
    """Verify resident expert pairs and commit accepted composite state.

    GDN projections and routed experts execute pairs. QSA and PLE retain their
    existing one-row update and checkpoint contracts. Failure invalidates the
    coordinator; close retries draining if a backend could not release resources.
    """

    def __init__(self, model, *, expert_pair_factory=None, shared_projections=False, native_gdn_boundaries=True):
        if expert_pair_factory is None:
            expert_pair_factory = Qwen4MTPFullResidentExpertPair
        if not callable(expert_pair_factory):
            raise TypeError("MTP expert pair factory must be callable")
        self.model = model
        self.shared_projections = shared_projections
        self.gdn_pairs = {i: (Qwen4MTPGDNPair(layer) if native_gdn_boundaries else
                             Qwen4MTPGDNPair(layer, native_boundaries=False)) for i, layer in enumerate(model.layers)
                          if layer.mixer_kind == "gdn"}
        self.expert_pairs = tuple(expert_pair_factory(layer.mlp) for layer in model.layers)
        self._full_resident_prefix_enabled = bool(self.expert_pairs) and all(
            type(pair) is Qwen4MTPFullResidentExpertPair
            for pair in self.expert_pairs
        )
        self._inflight = None
        self._failed = False
        self.rounds = 0
        self.accepted_drafts = 0
        self.plain_steps = 0
        self.prefix_submissions = 0

    def _submit_full_resident_prefix(self, output, *, layer_index: int) -> None:
        """Submit a completed full-resident four-layer prefix without waiting."""
        if (
            self._full_resident_prefix_enabled
            and (layer_index + 1) % 4 == 0
            and layer_index + 1 < len(self.expert_pairs)
        ):
            mx.async_eval(output)
            self.prefix_submissions += 1

    def close(self):
        """Drain and discard a failed round, retaining ownership on drain failure."""
        if self._inflight is None:
            return
        coordinator, base, _roots = self._inflight
        with pooled_request_scope(self.model, coordinator._identity):
            session = getattr(self.model, "_moespresso_pooled_decode_session", None)
            if session is not None:
                session.abort_and_drain()
            mx.synchronize()
            self.model._restore_state(base)
            self._inflight = None

    def verify(self, coordinator, input_ids: mx.array, *, cancelled=None) -> Qwen4MTPVerification:
        """Verify [anchor, one draft] greedily and commit its accepted input prefix."""
        model = self.model
        model._require_open()
        if self._failed or self._inflight is not None:
            raise RuntimeError("MTP verifier has a failed or unfinished round")
        if coordinator._model is not model:
            raise ValueError("MTP coordinator belongs to another model")
        base = coordinator.state
        if (base.batch_size != 1 or base.frontier <= 0 or input_ids.shape != (1, 2)
                or not coordinator._plain_all_valid_lineage):
            raise ValueError("MTP verification requires two rows after all-valid prefill")
        model._validate_trusted_state(base)
        valid, positions = model._normalize_inputs(base, input_ids, valid_tokens=None, position_ids=None)
        roots = []
        with pooled_request_scope(model, coordinator._identity):
            self._inflight = coordinator, base, roots
            try:
                self._check_cancel(cancelled)
                working = model._fork_state(base)
                candidate, finite = self._forward_pair(coordinator, working, input_ids, valid, positions,
                                                       roots, cancelled)
                logits_finite = mx.all(mx.isfinite(candidate.logits))
                mx.eval(candidate.logits, candidate.widened, logits_finite,
                        *tuple(_state_arrays(candidate.checkpoints)),
                        *((finite.predicate,) if finite is not None else ()))
                session = getattr(model, "_moespresso_pooled_decode_session", None)
                if session is not None:
                    session.drain()
                if finite is not None:
                    model.batched_qsa_append_finite_batches += 1
                    model.batched_qsa_append_finite_predicates += finite.predicate_count
                    if not bool(finite.predicate.item()):
                        model.batched_qsa_append_finite_failures += 1
                        raise ValueError("MTP QSA requires finite unpadded input")
                if not bool(logits_finite.item()):
                    raise ValueError("MTP target produced nonfinite logits")
                result = greedy_accept(input_ids[:, 1:], candidate.logits)
                self._check_cancel(cancelled)
                committed = coordinator.commit(candidate, result.accepted + 1)
                self._inflight = None
                self.rounds += 1
                self.accepted_drafts += result.accepted
                return Qwen4MTPVerification(result, candidate.logits[:, :result.accepted + 1],
                                            candidate.widened[:, :result.accepted + 1], committed)
            except BaseException as error:
                self._failed = True
                coordinator._closed = True
                coordinator._state = None
                try:
                    self.close()
                except BaseException as cleanup:
                    error.add_note(f"MTP cleanup requires another drain: {cleanup}")
                raise

    def verify_plain(
        self,
        coordinator,
        input_ids: mx.array,
        *,
        cancelled=None,
    ) -> Qwen4MTPPlainVerification:
        """Commit one ordinary row when no draft fits the remaining token budget."""

        model = self.model
        model._require_open()
        if self._failed or self._inflight is not None:
            raise RuntimeError("MTP verifier has a failed or unfinished round")
        if coordinator._model is not model:
            raise ValueError("MTP coordinator belongs to another model")
        base = coordinator.state
        if (
            base.batch_size != 1
            or base.frontier <= 0
            or input_ids.shape != (1, 1)
            or not coordinator._plain_all_valid_lineage
        ):
            raise ValueError("MTP plain verification requires one row after all-valid prefill")
        model._validate_trusted_state(base)
        with pooled_request_scope(model, coordinator._identity):
            self._inflight = coordinator, base, []
            try:
                self._check_cancel(cancelled)
                candidate = coordinator.propose(input_ids, capture_widened=True)
                mx.eval(
                    candidate.logits,
                    candidate.widened,
                    *tuple(_state_arrays(candidate.checkpoints)),
                )
                self._check_cancel(cancelled)
                committed = coordinator.commit(candidate, 1)
                self._inflight = None
                self.plain_steps += 1
                return Qwen4MTPPlainVerification(
                    candidate.logits,
                    candidate.widened,
                    committed,
                )
            except BaseException as error:
                self._failed = True
                coordinator._closed = True
                coordinator._state = None
                try:
                    self.close()
                except BaseException as cleanup:
                    error.add_note(f"MTP cleanup requires another drain: {cleanup}")
                raise

    @staticmethod
    def _check_cancel(cancelled):
        if cancelled is not None and cancelled():
            raise InterruptedError("MTP verification cancelled")

    def _forward_pair(self, coordinator, base, input_ids, valid, positions,
                      roots, cancelled):
        model = self.model
        histories = tuple(mx.concatenate([base.valid_history, valid[:, :count]], axis=1)
                          for count in (1, 2))
        position_histories = tuple(mx.concatenate([base.position_history, positions[:, :, :count]], axis=2)
                                   for count in (1, 2))
        qsa = Qwen4MTPQSARows(model, base, valid, positions, histories)
        hidden = mx.tile(model.embedding(input_ids), (1, 1, model.branch_count))
        layer_states = ([], [])
        pending = injection = None
        for index, (layer, previous, experts) in enumerate(zip(model.layers, base.layers, self.expert_pairs, strict=True)):
            self._check_cancel(cancelled)
            ple_states = (None, None)
            if layer.ple is not None:
                if pending is not None:
                    hidden = gated_residual_write(hidden, pending, injection)
                    pending = injection = None
                outputs, states = [], []
                ple_state = previous.ple_state
                for row in range(2):
                    item = layer.ple(hidden[:, row:row + 1], input_ids[:, row:row + 1],
                                     state=ple_state, valid_tokens=valid[:, row:row + 1])
                    outputs.append(hidden[:, row:row + 1] + item.output)
                    ple_state = item.state
                    states.append(ple_state)
                hidden = mx.concatenate(outputs, axis=1)
                ple_states = tuple(states)
            if index in self.gdn_pairs:
                pair = self.gdn_pairs[index](hidden, state=previous.mixer_state,
                                            pending_output=pending, pending_injection=injection)
                mixed, residual, weights = pair.mlp_hidden, pair.residual, pair.injection
                mixer_states = pair.first_state, pair.final_state
            else:
                mixed_rows, residual_rows, weight_rows, states = [], [], [], []
                mixer_state = previous.mixer_state
                prepared = None
                if self.shared_projections:
                    prepared = _read(layer.attention_residual, hidden, pending, injection)
                    qsa.prepare(index, prepared[0])
                for row in range(2):
                    if prepared is None:
                        mixed, residual, weights = _read(
                            layer.attention_residual, hidden[:, row:row + 1],
                            None if pending is None else pending[:, row:row + 1],
                            None if injection is None else injection[:, row:row + 1],
                        )
                    else:
                        mixed, residual, weights = (value[:, row:row + 1] for value in prepared)
                    result = qsa.step(index, row, mixed, mixer_state)
                    if result.frontier != base.frontier + row + 1:
                        raise ValueError("MTP mixer did not advance by one row")
                    # Mutable QSA snapshots must be taken before advancing the active view.
                    states.append(layer.mixer.snapshot_state(result.state))
                    mixer_state = result.state
                    mixed, residual, weights = _read(layer.mlp_residual, residual, result.output, weights)
                    mixed_rows.append(mixed)
                    residual_rows.append(residual)
                    weight_rows.append(weights)
                mixed, residual, weights = (mx.concatenate(rows, axis=1)
                                            for rows in (mixed_rows, residual_rows, weight_rows))
                mixer_states = tuple(states)
            output = experts(mixed)
            roots.append((output, mixer_states, ple_states))
            session = getattr(model, "_moespresso_pooled_decode_session", None)
            if session is not None:
                session.remember(roots)
            self._submit_full_resident_prefix(output, layer_index=index)
            next_has_ple = index + 1 < len(model.layers) and model.layers[index + 1].ple is not None
            if index + 1 == len(model.layers) or next_has_ple:
                hidden = gated_residual_write(residual, output, weights)
                pending = injection = None
            else:
                hidden, pending, injection = residual, output, weights
            for row in range(2):
                layer_states[row].append(Qwen4LayerState(layer.mixer_kind, mixer_states[row],
                                                       base.frontier + row + 1, ple_states[row]))
        if pending is not None:
            raise RuntimeError("MTP final layer left a pending residual")
        if self.shared_projections:
            logits = model.lm_head(model.final_residual(hidden))
        else:
            logits = mx.concatenate([model.lm_head(model.final_residual(hidden[:, row:row + 1]))
                                     for row in range(2)], axis=1)
        checkpoints = tuple(replace(base, frontier=base.frontier + row + 1,
                                    valid_history=histories[row], position_history=position_histories[row],
                                    layers=tuple(layer_states[row])) for row in range(2))
        for state in checkpoints:
            model._validate_trusted_state(state)
        return Qwen4Candidate(coordinator._identity, base.cache_identity, base.revision, base.frontier,
                              logits, checkpoints, trusted_lineage=True, widened=hidden,
                              _evaluated_boundary=_Qwen4EvaluatedBoundary(
                                  coordinator._identity, base.cache_identity, base.revision, base.frontier, checkpoints)), qsa.seal()
