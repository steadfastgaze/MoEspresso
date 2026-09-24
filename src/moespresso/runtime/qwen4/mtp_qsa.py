"""Reuse the shell's certified QSA operations for two verification frontiers."""

import mlx.core as mx

from moespresso.runtime.qwen4.model import (
    _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY, _QWEN4_BATCHED_UNDO_CAPABILITY,
    _Qwen4AppendFiniteBatch, _Qwen4AppendFiniteCheck, _Qwen4TrustedMaskCertificate,
)
from moespresso.runtime.qwen4.mtp_qsa_projections import prepare_mtp_qsa_projection_rows


class Qwen4MTPQSARows:
    """Batch anchor undo copies and certify each actual one-row state transition.

    The second undo is prepared after the anchor append. A pre-anchor snapshot
    cannot restore a rejected second row to the accepted anchor frontier.
    Each append still registers a finite predicate for the final publication
    boundary, using the existing backend capabilities and numerical kernels.
    """

    def __init__(self, model, base, valid, positions, histories, *, defer_undo=True):
        if type(defer_undo) is not bool:
            raise TypeError("MTP QSA undo deferral must be boolean")
        self.model = model
        self.defer_undo = defer_undo
        self.frontier = base.frontier
        self.valid = tuple(valid[:, row:row + 1] for row in range(2))
        self.pair_positions = positions
        self.positions = tuple(positions[:, :, row:row + 1] for row in range(2))
        self.histories = histories
        self.mixers = {i: layer.mixer for i, layer in enumerate(model.layers) if layer.mixer_kind == "qsa"}
        self.sources = {i: base.layers[i].mixer_state for i in self.mixers}
        self.first_results = {}
        self.prepared_projections = {}
        self.seen = set()
        self.checks = []
        self.certified = bool(self.mixers) and all(
            getattr(mixer, "supports_trusted_mask_certificate", False) for mixer in self.mixers.values())
        self.factors = (None, None)
        self.first_certificate = None
        if not self.mixers:
            return
        if not bool(mx.all(histories[-1]).item()):
            raise ValueError("MTP QSA requires an all-valid history")
        if self.certified:
            model.trusted_all_valid_reductions += 1
            self.first_certificate = self._certificate(tuple(self.mixers.values()), tuple(self.sources.values()), 0)
        mixers = tuple(self.mixers.values())
        signatures = tuple(getattr(mixer, "shared_rope_signature", None) for mixer in mixers)
        prepare = getattr(mixers[0], "_prepare_shared_rope_factors", None)
        if (signatures[0] is not None
                and all(signature == signatures[0] for signature in signatures)
                and callable(prepare)
                and all(callable(getattr(mixer, "_step_trusted_prepared", None)) for mixer in mixers)):
            self.factors = tuple(prepare(position) for position in self.positions)
            model.shared_qsa_rope_factor_builds += 2

    def prepare(self, index, hidden):
        """Batch one QSA layer's input projections without advancing its state."""

        if index not in self.mixers or index in self.prepared_projections:
            raise ValueError("MTP QSA projection layer is unknown or already prepared")
        if any(seen_index == index for seen_index, _row in self.seen):
            raise ValueError("MTP QSA projections must be prepared before either row")
        self.prepared_projections[index] = prepare_mtp_qsa_projection_rows(
            self.mixers[index],
            hidden,
            self.pair_positions,
            self.factors,
        )

    def _certificate(self, mixers, sources, row):
        names = ("_prepare_trusted_undo", "_trusted_undo_arrays", "_mark_trusted_undo_evaluated")
        undos = ()
        if (all(source is not None for source in sources)
                and all(callable(getattr(mixer, name, None)) for mixer in mixers for name in names)):
            undos = tuple(mixer._prepare_trusted_undo(source, new_tokens=1, capability=_QWEN4_BATCHED_UNDO_CAPABILITY)
                          for mixer, source in zip(mixers, sources, strict=True))
            arrays = tuple(array for mixer, undo in zip(mixers, undos, strict=True)
                           for array in mixer._trusted_undo_arrays(undo, capability=_QWEN4_BATCHED_UNDO_CAPABILITY))
            dependency_bound = (
                self.defer_undo
                and row == 1
                and all(undo is not None for undo in undos)
                and all(getattr(mixer, "supports_dependency_bound_undo", False) for mixer in mixers)
            )
            if not dependency_bound:
                mx.eval(*arrays)
                for mixer, undo in zip(mixers, undos, strict=True):
                    mixer._mark_trusted_undo_evaluated(
                        undo, capability=_QWEN4_BATCHED_UNDO_CAPABILITY)
            self.model.batched_qsa_undo_batches += 1
            self.model.batched_qsa_undo_reservations += len(undos)
        finite = None
        if (undos
                and all(getattr(mixer, "supports_batched_append_finite", False) for mixer in mixers)):
            finite = _Qwen4AppendFiniteBatch(mixers, sources)
        self.model.trusted_mask_certificate_builds += 1
        return _Qwen4TrustedMaskCertificate(
            issuer=self.model._trusted_mask_issuer, scope=object(), allowed_mixers=mixers,
            source_states=sources, valid_tokens=self.valid[row], visible_history=self.histories[row],
            current_frontier=self.frontier + row, next_frontier=self.frontier + row + 1,
            prepared_undos=undos, append_finite_batch=finite,
        )

    def step(self, index, row, hidden, state):
        if row not in (0, 1) or index not in self.mixers or (index, row) in self.seen:
            raise ValueError("MTP QSA row is unknown or already consumed")
        source = self.sources[index] if row == 0 else self.first_results.get(index)
        if state is not source or (row == 1 and index not in self.first_results):
            raise ValueError("MTP QSA row does not extend its owned prefix")
        mixer = self.mixers[index]
        prepared = self.prepared_projections.get(index)
        prepared_row = None if prepared is None else prepared.rows[row]
        if prepared_row is not None:
            if (
                hidden.shape != prepared_row.hidden_states.shape
                or hidden.dtype != prepared_row.hidden_states.dtype
            ):
                raise ValueError("MTP QSA prepared row has incompatible hidden states")
            hidden = prepared_row.hidden_states
        certificate = self.first_certificate if row == 0 else (
            self._certificate((mixer,), (state,), row) if self.certified else None)
        kwargs = dict(valid_tokens=self.valid[row], visible_history=self.histories[row],
                      position_ids=self.positions[row], state=state)
        if prepared_row is not None:
            kwargs["prepared_projections"] = prepared_row
        if certificate is not None:
            mixer._bind_trusted_mask_issuer(certificate.issuer, certificate.scope)
        try:
            if self.factors[row] is not None:
                result = mixer._step_trusted_prepared(hidden, **kwargs, mask_certificate=certificate,
                                                      shared_rope_factors=self.factors[row])
            elif certificate is not None:
                result = mixer._step_trusted_certified(hidden, **kwargs, mask_certificate=certificate)
            else:
                result = getattr(mixer, "step_trusted", mixer)(hidden, **kwargs)
        finally:
            if certificate is not None:
                mixer._clear_trusted_mask_scope(certificate.issuer, certificate.scope)
        self.seen.add((index, row))
        if row == 0:
            self.first_results[index] = result.state
        elif certificate is not None and certificate.append_finite_batch is not None:
            self.checks.append(certificate.append_finite_batch.seal(
                ((mixer, result.state),), capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY))
        return result

    def seal(self):
        if self.seen != {(index, row) for index in self.mixers for row in (0, 1)}:
            raise ValueError("MTP QSA did not complete both verification rows")
        certificate = self.first_certificate
        checks = list(self.checks)
        if certificate is not None and certificate.append_finite_batch is not None:
            checks.append(certificate.append_finite_batch.seal(
                tuple((mixer, self.first_results[index]) for index, mixer in self.mixers.items()),
                capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY))
        if not checks:
            return None
        return _Qwen4AppendFiniteCheck(mx.all(mx.stack([check.predicate for check in checks])),
                                      sum(check.predicate_count for check in checks))
