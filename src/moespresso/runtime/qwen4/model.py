"""Architecture-faithful Qwen4 text orchestration and cache publication shell."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Protocol
import weakref

import mlx.core as mx
import mlx.nn as nn

from moespresso.runtime.pooled_moe import abort_pooled_request, pooled_state_scope
from moespresso.runtime.qwen4.ple import (
    Qwen4PLEOutput,
    Qwen4PLEState,
)
from moespresso.runtime.qwen4.primitives import gated_residual_write


class Qwen4StatefulMixer(Protocol):
    """Uniform state boundary for a GDN or QSA layer adapter."""

    def fork_state(self, state: Any) -> Any:
        """Return a functional copy or rollback-safe working marker."""
        ...

    def snapshot_state(self, state: Any) -> Any:
        """Return an immutable checkpoint that does not alias working state."""
        ...

    def validate_state(
        self,
        state: Any,
        *,
        expected_frontier: int,
        position_history: mx.array,
    ) -> None:
        """Fail unless the opaque state is complete at the expected frontier."""
        ...

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        valid_tokens: mx.array,
        visible_history: mx.array,
        position_ids: mx.array,
        state: Any,
    ) -> "Qwen4MixerOutput": ...


class Qwen4PLE(Protocol):
    """PLE boundary consumed by the text orchestration shell."""

    def validate_state(
        self,
        state: Qwen4PLEState | None,
        *,
        expected_frontier: int,
        batch_size: int,
    ) -> None:
        """Fail unless state is complete at the expected frontier."""
        ...

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        *,
        state: Qwen4PLEState | None = None,
        valid_tokens: mx.array | None = None,
    ) -> Qwen4PLEOutput: ...


@dataclass(frozen=True)
class Qwen4MixerOutput:
    """One mixer output and its uncommitted continuation state."""

    output: mx.array
    state: Any
    frontier: int


@dataclass(frozen=True)
class _Qwen4TrustedMaskCertificate:
    """Request-local proof for masks derived by the model shell."""

    issuer: object
    scope: object
    allowed_mixers: tuple[object, ...]
    source_states: tuple[Any, ...]
    valid_tokens: mx.array
    visible_history: mx.array
    current_frontier: int
    next_frontier: int
    prepared_undos: tuple[Any | None, ...] = ()
    append_finite_batch: _Qwen4AppendFiniteBatch | None = None
    serial_capability: object | None = None


@dataclass(frozen=True)
class _Qwen4AppendFiniteCheck:
    """One sealed request-local predicate evaluated at publication."""

    predicate: mx.array
    predicate_count: int


@dataclass
class _Qwen4AppendFiniteBatch:
    """Collect one deferred append predicate from every trusted QSA layer."""

    allowed_mixers: tuple[object, ...]
    source_states: tuple[Any, ...]
    _results: list[Any | None] = field(init=False, repr=False)
    _predicates: list[mx.array | None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.allowed_mixers or len(self.allowed_mixers) != len(self.source_states):
            raise ValueError("QSA append finite batch has incompatible mixer state")
        for index, mixer in enumerate(self.allowed_mixers):
            if any(mixer is prior for prior in self.allowed_mixers[:index]):
                raise ValueError("QSA append finite batch contains a duplicate mixer")
        self._results = [None] * len(self.allowed_mixers)
        self._predicates = [None] * len(self.allowed_mixers)

    def register(
        self,
        *,
        mixer: object,
        source_state: Any,
        result_state: Any,
        predicate: mx.array,
        capability: object,
    ) -> None:
        """Register one private append result without evaluating its predicate."""

        if capability is not _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY:
            raise ValueError("QSA append finite capability is invalid")
        if predicate.shape != () or predicate.dtype != mx.bool_:
            raise ValueError("QSA append finite predicate must be a boolean scalar")
        matches = [
            index
            for index, (allowed, source) in enumerate(
                zip(self.allowed_mixers, self.source_states, strict=True)
            )
            if allowed is mixer and source is source_state
        ]
        if len(matches) != 1:
            raise ValueError("QSA append finite registration is unissued or stale")
        index = matches[0]
        if self._predicates[index] is not None:
            raise ValueError("QSA append finite predicate was registered twice")
        self._results[index] = result_state
        self._predicates[index] = predicate

    def seal(
        self,
        result_pairs: tuple[tuple[object, Any], ...],
        *,
        capability: object,
    ) -> _Qwen4AppendFiniteCheck:
        """Fail closed on incomplete registration and combine all predicates."""

        if capability is not _QWEN4_BATCHED_APPEND_FINITE_CAPABILITY:
            raise ValueError("QSA append finite capability is invalid")
        if len(result_pairs) != len(self.allowed_mixers):
            raise ValueError("QSA append finite batch did not observe every mixer")
        predicates: list[mx.array] = []
        for index, (mixer, result_state) in enumerate(result_pairs):
            if mixer is not self.allowed_mixers[index] or result_state is not self._results[index]:
                raise ValueError("QSA append finite result does not match its registration")
            predicate = self._predicates[index]
            if predicate is None:
                raise ValueError("QSA append finite batch is incomplete")
            predicates.append(predicate)
        return _Qwen4AppendFiniteCheck(
            predicate=mx.all(mx.stack(predicates)),
            predicate_count=len(predicates),
        )


_QWEN4_BATCHED_UNDO_CAPABILITY = object()
_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY = object()
_QWEN4_SERIAL_LANE_CAPABILITY = object()


def _compiled_gdn_full_residency_matches(
    layers: tuple[Any, ...],
    capacities: Any,
) -> bool:
    """Check that every routed layer is resident at its physical expert count."""

    if not isinstance(capacities, dict) or set(capacities) != set(range(len(layers))):
        return False
    for index, layer in enumerate(layers):
        physical_experts = getattr(getattr(layer, "mlp", None), "physical_experts", None)
        capacity = capacities[index]
        if (
            isinstance(physical_experts, bool)
            or not isinstance(physical_experts, int)
            or physical_experts <= 0
            or isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity != physical_experts
        ):
            return False
    return True


@dataclass(frozen=True)
class _Qwen4CompiledGDNRunSpec:
    indices: tuple[int, ...]
    expects_pending: bool
    publishes_final_write: bool = False


_QWEN4_COMPILED_GDN_RUN_SPECS = (
    _Qwen4CompiledGDNRunSpec((0,), False, True),
    _Qwen4CompiledGDNRunSpec((1,), False),
    _Qwen4CompiledGDNRunSpec((2,), True),
    *tuple(
        _Qwen4CompiledGDNRunSpec((start, start + 1, start + 2), True) for start in range(4, 45, 4)
    ),
)


@dataclass(frozen=True)
class _Qwen4CompiledGDNRun:
    spec: _Qwen4CompiledGDNRunSpec
    function: Any


@dataclass
class Qwen4DecoderLayer(nn.Module):
    """Injected modules for one released Qwen4 decoder layer."""

    mixer_kind: str
    attention_residual: Any
    mixer: Qwen4StatefulMixer
    mlp_residual: Any
    mlp: Any
    ple: Qwen4PLE | None = field(default_factory=lambda: None)

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        if self.mixer_kind not in {"gdn", "qsa"}:
            raise ValueError("mixer_kind must be 'gdn' or 'qsa'")


@dataclass(frozen=True)
class Qwen4LayerState:
    """Opaque mixer state bound to one public model frontier."""

    mixer_kind: str
    mixer_state: Any
    mixer_offset: int
    ple_state: Qwen4PLEState | None = None


@dataclass(frozen=True)
class Qwen4CompositeState:
    """All text-model state published at one physical token frontier."""

    cache_identity: str
    revision: int
    frontier: int
    batch_size: int
    valid_history: mx.array
    position_history: mx.array
    layers: tuple[Qwen4LayerState, ...]


@dataclass(frozen=True)
class _Qwen4EvaluatedBoundary:
    """Seal the exact checkpoint tuple evaluated by one proposal."""

    coordinator_identity: object
    cache_identity: str
    base_revision: int
    base_frontier: int
    checkpoints: tuple[Qwen4CompositeState, ...]


@dataclass
class Qwen4Candidate:
    """Tokenwise checkpoints proposed against one committed state revision."""

    coordinator_identity: object
    cache_identity: str
    base_revision: int
    base_frontier: int
    logits: mx.array
    checkpoints: tuple[Qwen4CompositeState, ...]
    trusted_lineage: bool = False
    _evaluated_boundary: _Qwen4EvaluatedBoundary | None = field(
        default=None,
        repr=False,
    )
    consumed: bool = False
    widened: mx.array | None = None


@dataclass
class _Qwen4PlainSerialStep:
    """One request-private decode row awaiting a single publication boundary."""

    lane_identity: object
    cache_identity: str
    base_revision: int
    base_frontier: int
    logits: mx.array
    state: Qwen4CompositeState
    append_finite_check: _Qwen4AppendFiniteCheck | None
    consumed: bool = False


class Qwen4PlainSerialLane:
    """Advance ordinary single-row decode without branch rollback state."""

    def __init__(
        self,
        model: "Qwen4TextModelShell",
        state: Qwen4CompositeState,
    ) -> None:
        self._model = model
        self._identity = object()
        self._state: Qwen4CompositeState | None = state
        self._pending: _Qwen4PlainSerialStep | None = None
        self._superseded: _Qwen4PlainSerialStep | None = None
        self._scheduled_roots: tuple[mx.array, ...] = ()
        self._closed = False
        self._failed = False

    @property
    def frontier(self) -> int:
        """Return the last fully published private frontier."""

        state = self._require_ready()
        return state.frontier

    def _require_ready(self) -> Qwen4CompositeState:
        if self._closed or self._state is None:
            reason = "failed" if self._failed else "closed"
            raise RuntimeError(f"Qwen4 plain serial lane is {reason}")
        if self._pending is not None:
            raise RuntimeError("Qwen4 plain serial lane already has a pending step")
        return self._state

    @pooled_state_scope
    def begin_step(self, input_ids: mx.array) -> _Qwen4PlainSerialStep:
        """Build one irreversible request-private decode row lazily."""

        state = self._require_ready()
        if input_ids.shape != (1, 1):
            raise ValueError("Qwen4 plain serial lane requires one token and batch size one")
        if input_ids.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("Qwen4 plain serial token must be an integer")
        valid_tokens = mx.ones((1, 1), dtype=mx.bool_)
        position_ids = _normalize_position_ids(
            None,
            batch_size=1,
            token_count=1,
            physical_frontier=state.frontier,
        )
        try:
            logits, next_state, append_finite_check = self._model._forward_chunk(
                state,
                input_ids,
                valid_tokens,
                position_ids,
                trusted=True,
                serial_capability=_QWEN4_SERIAL_LANE_CAPABILITY,
            )
        except BaseException:
            self._fail(state)
            raise
        step = _Qwen4PlainSerialStep(
            lane_identity=self._identity,
            cache_identity=state.cache_identity,
            base_revision=state.revision,
            base_frontier=state.frontier,
            logits=logits,
            state=next_state,
            append_finite_check=append_finite_check,
        )
        self._pending = step
        return step

    @pooled_state_scope
    def prime_pipelined_step(
        self,
        step: _Qwen4PlainSerialStep,
        *,
        evaluated: tuple[mx.array, ...],
        queued: tuple[mx.array, ...],
    ) -> None:
        """Submit one request-private target row without publishing its frontier."""

        self._require_pipeline_step(step)
        roots = (*evaluated, *tuple(_state_arrays(step.state)), *queued)
        if step.append_finite_check is not None:
            roots = (*roots, step.append_finite_check.predicate)
        mx.async_eval(*roots)
        self._scheduled_roots = roots
        self._model.plain_serial_pipeline_primes += 1

    @pooled_state_scope
    def chain_pipelined_step(
        self,
        step: _Qwen4PlainSerialStep,
        input_ids: mx.array,
    ) -> _Qwen4PlainSerialStep:
        """Build the next request-private row before publishing the preceding row."""

        self._require_pipeline_step(step)
        if self._superseded is not None:
            raise RuntimeError("Qwen4 plain serial pipeline has an unfinished transition")
        if input_ids.shape != (1, 1):
            raise ValueError("Qwen4 plain serial pipeline requires one token")
        if input_ids.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("Qwen4 plain serial token must be an integer")
        base = replace(step.state, revision=step.base_revision + 1)
        valid_tokens = mx.ones((1, 1), dtype=mx.bool_)
        position_ids = _normalize_position_ids(
            None,
            batch_size=1,
            token_count=1,
            physical_frontier=base.frontier,
        )
        try:
            logits, next_state, append_finite_check = self._model._forward_chunk(
                base,
                input_ids,
                valid_tokens,
                position_ids,
                trusted=True,
                serial_capability=_QWEN4_SERIAL_LANE_CAPABILITY,
            )
        except BaseException:
            step.consumed = True
            self._fail(step.state)
            raise
        next_step = _Qwen4PlainSerialStep(
            lane_identity=self._identity,
            cache_identity=base.cache_identity,
            base_revision=base.revision,
            base_frontier=base.frontier,
            logits=logits,
            state=next_state,
            append_finite_check=append_finite_check,
        )
        self._superseded = step
        self._pending = next_step
        self._state = next_state
        return next_step

    @pooled_state_scope
    def finish_pipelined_transition(
        self,
        step: _Qwen4PlainSerialStep,
        next_step: _Qwen4PlainSerialStep,
        *,
        evaluated: tuple[mx.array, ...],
        queued: tuple[mx.array, ...],
    ) -> None:
        """Validate one superseded row while the next private row executes."""

        if (
            self._closed
            or self._state is None
            or step is not self._superseded
            or next_step is not self._pending
            or step.lane_identity is not self._identity
            or next_step.lane_identity is not self._identity
        ):
            raise ValueError("Qwen4 plain serial pipeline transition is stale")
        if step.consumed or next_step.consumed:
            raise ValueError("Qwen4 plain serial pipeline step has already been consumed")
        if (
            step.state.frontier != next_step.base_frontier
            or next_step.state.frontier != step.state.frontier + 1
            or next_step.base_revision != step.base_revision + 1
        ):
            raise ValueError("Qwen4 plain serial pipeline is not contiguous")
        if not evaluated or any(not isinstance(array, mx.array) for array in evaluated):
            raise TypeError("evaluated must contain MLX arrays")
        if any(not isinstance(array, mx.array) for array in queued):
            raise TypeError("queued must contain MLX arrays")
        roots = (*evaluated, *tuple(_state_arrays(next_step.state)), *queued)
        if step.append_finite_check is not None:
            roots = (*roots, step.append_finite_check.predicate)
        if next_step.append_finite_check is not None:
            roots = (*roots, next_step.append_finite_check.predicate)
        try:
            mx.async_eval(*roots)
            next_roots = (*tuple(_state_arrays(next_step.state)), *queued)
            if next_step.append_finite_check is not None:
                next_roots = (*next_roots, next_step.append_finite_check.predicate)
            self._scheduled_roots = next_roots
            self._eval_pipelined_step(step, evaluated=evaluated)
        except BaseException:
            step.consumed = True
            next_step.consumed = True
            self._fail(next_step.state)
            raise
        step.consumed = True
        self._superseded = None
        self._model.plain_serial_steps += 1
        self._model.plain_serial_pipeline_transitions += 1

    @pooled_state_scope
    def finish_terminal_pipelined_step(
        self,
        step: _Qwen4PlainSerialStep,
        *,
        evaluated: tuple[mx.array, ...],
    ) -> None:
        """Evaluate and publish the terminal request-private frontier."""

        self._require_pipeline_step(step)
        try:
            self._model._eval_serial_boundary(
                evaluated,
                step.state,
                step.append_finite_check,
            )
            committed = self._model._commit_state(step.state)
            committed = replace(committed, revision=step.base_revision + 1)
            self._model._validate_trusted_state(committed)
        except BaseException:
            step.consumed = True
            self._fail(step.state)
            raise
        self._state = committed
        self._pending = None
        self._scheduled_roots = ()
        step.consumed = True
        self._model.plain_serial_steps += 1
        self._model.plain_serial_pipeline_terminal_finishes += 1

    def _require_pipeline_step(self, step: _Qwen4PlainSerialStep) -> None:
        if (
            self._closed
            or self._state is None
            or step is not self._pending
            or step.lane_identity is not self._identity
        ):
            raise ValueError("Qwen4 plain serial pipeline step is unissued or stale")
        if step.consumed:
            raise ValueError("Qwen4 plain serial pipeline step has already been consumed")

    def _eval_pipelined_step(
        self,
        step: _Qwen4PlainSerialStep,
        *,
        evaluated: tuple[mx.array, ...],
    ) -> None:
        mx.eval(*evaluated)
        finite = step.append_finite_check
        if finite is None:
            return
        self._model.batched_qsa_append_finite_batches += 1
        self._model.batched_qsa_append_finite_predicates += finite.predicate_count
        mx.eval(finite.predicate)
        if not bool(finite.predicate.item()):
            self._model.batched_qsa_append_finite_failures += 1
            raise ValueError("mutable Qwen KVarN requires finite unpadded input")

    @pooled_state_scope
    def finish_step(
        self,
        step: _Qwen4PlainSerialStep,
        *,
        evaluated: tuple[mx.array, ...],
    ) -> None:
        """Evaluate sampling and state together, then publish the private row."""

        if self._closed or self._state is None:
            raise RuntimeError("Qwen4 plain serial lane is closed")
        if step is not self._pending or step.lane_identity is not self._identity:
            raise ValueError("Qwen4 plain serial step is unissued or stale")
        if step.consumed:
            raise ValueError("Qwen4 plain serial step has already been consumed")
        current = self._state
        if (
            step.cache_identity != current.cache_identity
            or step.base_revision != current.revision
            or step.base_frontier != current.frontier
            or step.state.frontier != current.frontier + 1
        ):
            raise ValueError("Qwen4 plain serial step does not extend the private frontier")
        if not evaluated or any(not isinstance(array, mx.array) for array in evaluated):
            raise TypeError("evaluated must contain MLX arrays")
        try:
            self._model._eval_serial_boundary(
                evaluated,
                step.state,
                step.append_finite_check,
            )
            committed = self._model._commit_state(step.state)
            self._model._validate_trusted_state(committed)
        except BaseException:
            step.consumed = True
            self._fail(step.state)
            raise
        self._state = replace(committed, revision=current.revision + 1)
        self._pending = None
        step.consumed = True
        self._model.plain_serial_steps += 1

    def _fail(self, state: Qwen4CompositeState) -> None:
        abort_pooled_request(self._model)
        try:
            try:
                self._drain_scheduled()
                self._model._abandon_serial_state(state)
            except Exception:
                # Preserve the request failure after an irreversible mutation.
                pass
        finally:
            if self._pending is not None:
                self._pending.consumed = True
            if self._superseded is not None:
                self._superseded.consumed = True
            self._pending = None
            self._superseded = None
            self._scheduled_roots = ()
            self._state = None
            self._closed = True
            self._failed = True
            self._model.plain_serial_failures += 1

    def _drain_scheduled(self) -> None:
        roots = self._scheduled_roots
        self._scheduled_roots = ()
        if roots:
            mx.eval(*roots)

    def _discard_pending(self, state: Qwen4CompositeState) -> None:
        abort_pooled_request(self._model)
        try:
            try:
                self._drain_scheduled()
            except Exception:
                self._model.plain_serial_pipeline_discard_drain_failures += 1
            try:
                self._model._abandon_serial_state(state)
            except Exception:
                pass
        finally:
            if self._pending is not None:
                self._pending.consumed = True
            if self._superseded is not None:
                self._superseded.consumed = True
            self._pending = None
            self._superseded = None
            self._scheduled_roots = ()
            self._state = None
            self._closed = True
            self._model.plain_serial_pipeline_discards += 1

    @pooled_state_scope
    def close(self) -> None:
        """Discard this request's private state and any unfinished mutation."""

        if self._closed:
            return
        pending = self._pending
        if pending is not None:
            if self._scheduled_roots or self._superseded is not None:
                self._discard_pending(pending.state)
            else:
                pending.consumed = True
                self._fail(pending.state)
            return
        self._state = None
        self._closed = True


class Qwen4StateCoordinator:
    """Atomically publish one candidate checkpoint after evaluation."""

    def __init__(self, model: "Qwen4TextModelShell", state: Qwen4CompositeState) -> None:
        self._model = model
        self._identity = object()
        self._closed = False
        self._plain_all_valid_lineage = True
        model.validate_state(state)
        self._state: Qwen4CompositeState | None = state

    @property
    def state(self) -> Qwen4CompositeState:
        if self._closed or self._state is None:
            raise RuntimeError("Qwen4 state coordinator is closed")
        return self._state

    @pooled_state_scope
    def close(self) -> None:
        """Release this request's composite QSA, GDN and PLE state."""
        if self._closed:
            return
        state = self._state
        if state is not None:
            self._model._restore_state(state)
        self._state = None
        self._closed = True

    @pooled_state_scope
    def forward_chunk(
        self,
        input_ids: mx.array,
        *,
        valid_tokens: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        """Commit one public prefill chunk and return its logits.

        Ordinary prefill does not need tokenwise proposal checkpoints. The
        model still evaluates and commits the complete composite state before
        this coordinator publishes the new revision. A failed chunk restores
        the previously committed revision inside ``model.forward_chunk``.
        """

        current = self.state
        logits, committed = self._model._forward_committed_chunk(
            current,
            input_ids,
            valid_tokens=valid_tokens,
            position_ids=position_ids,
            trusted=True,
        )
        self._state = committed
        if valid_tokens is not None:
            self._plain_all_valid_lineage = False
        return logits

    def enter_plain_serial_lane(self) -> Qwen4PlainSerialLane:
        """Consume this coordinator into the guarded ordinary-decode lane."""

        current = self.state
        if not self._plain_all_valid_lineage:
            raise ValueError("Qwen4 plain serial lane requires an all-valid lineage")
        if not self._model._plain_serial_lane_eligible(current):
            raise ValueError("Qwen4 plain serial lane is unavailable for this state")
        lane = Qwen4PlainSerialLane(self._model, current)
        self._state = None
        self._closed = True
        self._model._plain_serial_lanes.add(lane)
        self._model.plain_serial_lane_entries += 1
        return lane

    @pooled_state_scope
    def forward_chunk_with_widened(
        self,
        input_ids: mx.array,
        *,
        valid_tokens: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Commit a prefill chunk and retain its pre-final-mixer hidden rows."""
        captured: list[mx.array] = []
        logits, committed = self._model._forward_committed_chunk(
            self.state, input_ids, valid_tokens=valid_tokens,
            position_ids=position_ids, trusted=True, widened_capture=captured,
        )
        if len(captured) != 1:
            raise RuntimeError("Qwen widened capture did not return one committed chunk")
        self._state = committed
        if valid_tokens is not None:
            self._plain_all_valid_lineage = False
        return logits, captured[0]

    def try_enter_plain_serial_lane(self) -> Qwen4PlainSerialLane | None:
        """Consume this coordinator when the private serial contract is available."""

        current = self.state
        if not self._plain_all_valid_lineage or not self._model._plain_serial_lane_eligible(
            current
        ):
            return None
        return self.enter_plain_serial_lane()

    @pooled_state_scope
    def propose(
        self,
        input_ids: mx.array,
        *,
        valid_tokens: mx.array | None = None,
        position_ids: mx.array | None = None,
        capture_widened: bool = False,
    ) -> Qwen4Candidate:
        """Propose decode rows from this coordinator's trusted lineage."""

        return self._model._propose(
            self,
            input_ids,
            valid_tokens=valid_tokens,
            position_ids=position_ids,
            trusted=True,
            capture_widened=capture_widened,
        )

    @pooled_state_scope
    def commit(self, candidate: Qwen4Candidate, keep_tokens: int) -> Qwen4CompositeState:
        if isinstance(keep_tokens, bool) or not isinstance(keep_tokens, int):
            raise TypeError("keep_tokens must be an int")
        if not 0 <= keep_tokens <= len(candidate.checkpoints):
            raise ValueError("keep_tokens is outside the candidate checkpoint range")
        if candidate.consumed:
            raise ValueError("candidate has already been consumed")
        current = self.state
        if (
            candidate.coordinator_identity is not self._identity
            or candidate.cache_identity != current.cache_identity
            or candidate.base_revision != current.revision
            or candidate.base_frontier != current.frontier
        ):
            raise ValueError("candidate does not extend the committed cache revision")
        if keep_tokens == 0:
            self._model._restore_state(current)
            candidate.consumed = True
            return current

        selected = candidate.checkpoints[keep_tokens - 1]
        if (
            selected.cache_identity != candidate.cache_identity
            or selected.revision != candidate.base_revision
            or selected.frontier != candidate.base_frontier + keep_tokens
            or selected.batch_size != current.batch_size
        ):
            raise ValueError("candidate checkpoint does not match its token frontier")
        self._model._restore_state(selected)
        validate = (
            self._model._validate_trusted_state
            if candidate.trusted_lineage
            else self._model.validate_state
        )
        validate(selected)
        accepted_logits = candidate.logits[:, :keep_tokens]
        evaluated_boundary = candidate._evaluated_boundary
        can_reuse_evaluated_state = (
            evaluated_boundary is not None
            and evaluated_boundary.coordinator_identity is self._identity
            and evaluated_boundary.cache_identity == candidate.cache_identity
            and evaluated_boundary.base_revision == candidate.base_revision
            and evaluated_boundary.base_frontier == candidate.base_frontier
            and evaluated_boundary.checkpoints is candidate.checkpoints
        )
        if can_reuse_evaluated_state:
            mx.eval(accepted_logits)
            self._model.evaluated_commit_boundary_calls += 1
        else:
            mx.eval(accepted_logits, *tuple(_state_arrays(selected)))
            self._model.full_commit_boundary_calls += 1
        selected = self._model._commit_state(selected)
        validate(selected)
        self._state = replace(selected, revision=current.revision + 1)
        candidate.consumed = True
        return self._state


class Qwen4TextModelShell(nn.Module):
    """Released text-layer order and functional cache frontiers.

    Mixer state is either functional or a process-local rollback marker. The
    shell restores a selected marker, evaluates its arrays and commits it before
    the coordinator swap. Proposals remain tokenwise so any accepted prefix can
    be committed. ``forward_chunk`` executes ordinary prefill as one chunk.
    """

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "layers" and isinstance(value, tuple):
            value = list(value)
        super().__setattr__(key, value)

    def __init__(
        self,
        *,
        cache_identity: str,
        embedding: Any,
        layers: tuple[Qwen4DecoderLayer, ...],
        final_residual: Any,
        lm_head: Any,
        hidden_size: int,
        branch_count: int,
    ) -> None:
        super().__init__()
        if not cache_identity:
            raise ValueError("cache_identity must not be empty")
        if hidden_size <= 0 or branch_count <= 0:
            raise ValueError("model geometry must be positive")
        if not layers:
            raise ValueError("at least one decoder layer is required")
        self.cache_identity = cache_identity
        self.embedding = embedding
        self.layers = list(layers)
        self.final_residual = final_residual
        self.lm_head = lm_head
        self.hidden_size = hidden_size
        self.branch_count = branch_count
        self.expanded_size = hidden_size * branch_count
        self.trusted_mask_certificate_builds = 0
        self.trusted_all_valid_reductions = 0
        self.batched_qsa_undo_batches = 0
        self.batched_qsa_undo_reservations = 0
        self.batched_qsa_append_finite_batches = 0
        self.batched_qsa_append_finite_predicates = 0
        self.batched_qsa_append_finite_failures = 0
        self.shared_qsa_rope_factor_builds = 0
        self.evaluated_commit_boundary_calls = 0
        self.full_commit_boundary_calls = 0
        self.plain_serial_lane_entries = 0
        self.plain_serial_steps = 0
        self.plain_serial_failures = 0
        self.plain_serial_pipeline_primes = 0
        self.plain_serial_pipeline_transitions = 0
        self.plain_serial_pipeline_terminal_finishes = 0
        self.plain_serial_pipeline_discards = 0
        self.plain_serial_pipeline_discard_drain_failures = 0
        self.compiled_gdn_run_builds = 0
        self.compiled_gdn_run_calls = 0
        self.compiled_gdn_run_layer_calls = 0
        self.compiled_gdn_run_ineligible_calls = 0
        self.compiled_gdn_run_failures = 0
        object.__setattr__(self, "_trusted_mask_issuer", object())
        object.__setattr__(self, "_closed", False)
        object.__setattr__(self, "_closing", False)
        object.__setattr__(
            self,
            "_coordinators",
            weakref.WeakSet(),
        )
        object.__setattr__(
            self,
            "_plain_serial_lanes",
            weakref.WeakSet(),
        )
        object.__setattr__(self, "_compiled_gdn_runs", None)
        object.__setattr__(self, "_moespresso_qwen4_mtp_compiled_core", None)
        object.__setattr__(self, "_cache_routing_enabled", False)

    def trusted_mask_stats(self) -> dict[str, int]:
        """Return monotonic counters for trusted mask certificates."""

        return {
            "trusted_mask_certificate_builds": self.trusted_mask_certificate_builds,
            "trusted_all_valid_reductions": self.trusted_all_valid_reductions,
            "batched_qsa_undo_batches": self.batched_qsa_undo_batches,
            "batched_qsa_undo_reservations": self.batched_qsa_undo_reservations,
        }

    def shared_qsa_rope_stats(self) -> dict[str, int]:
        """Return monotonic counters for request-local QSA RoPE reuse."""

        calls = 0
        applications = 0
        layers = 0
        for layer in self.layers:
            if layer.mixer_kind != "qsa":
                continue
            stats = getattr(layer.mixer, "shared_rope_stats", None)
            if not callable(stats):
                continue
            values = stats()
            layers += 1
            calls += int(values["shared_rope_factor_calls"])
            applications += int(values["shared_rope_applications"])
        return {
            "qsa_layers": layers,
            "shared_qsa_rope_factor_builds": self.shared_qsa_rope_factor_builds,
            "shared_rope_factor_calls": calls,
            "shared_rope_applications": applications,
        }

    def qsa_native_selector_stats(self) -> dict[str, int]:
        """Return aggregate counters for native trusted-decode selection."""

        totals = {
            "qsa_layers": 0,
            "native_selector_calls": 0,
            "native_selector_groups": 0,
            "native_selector_ineligible_calls": 0,
            "native_select_gather_calls": 0,
            "native_select_gather_groups": 0,
            "native_select_gather_ineligible_calls": 0,
            "native_select_gather_unavailable_calls": 0,
            "native_project_rope_calls": 0,
            "native_project_rope_ineligible_calls": 0,
            "native_project_rope_unavailable_calls": 0,
        }
        for layer in self.layers:
            if layer.mixer_kind != "qsa":
                continue
            stats = getattr(layer.mixer, "native_selector_stats", None)
            if not callable(stats):
                continue
            values = stats()
            totals["qsa_layers"] += 1
            for name in tuple(totals)[1:]:
                totals[name] += int(values[name])
        return totals

    def cache_routing_stats(self) -> dict[str, Any]:
        from moespresso.runtime.qwen4.cache_routing import cache_routing_stats

        return cache_routing_stats(self)

    def fused_exact_router_stats(self) -> dict[str, int]:
        """Return aggregate counters for exact fused router selection."""

        totals = {
            "router_layers": 0,
            "fused_exact_calls": 0,
            "fused_exact_rows": 0,
            "fused_exact_ineligible_calls": 0,
            "fused_exact_unavailable_calls": 0,
        }
        for layer in self.layers:
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            stats = getattr(gate, "fused_exact_stats", None)
            if not callable(stats):
                continue
            values = stats()
            totals["router_layers"] += 1
            for name in tuple(totals)[1:]:
                totals[name] += int(values[name])
        return totals

    def qsa_append_finite_stats(self) -> dict[str, int]:
        """Return monotonic counters for private append-finite batching."""

        return {
            "batched_qsa_append_finite_batches": self.batched_qsa_append_finite_batches,
            "batched_qsa_append_finite_predicates": (self.batched_qsa_append_finite_predicates),
            "batched_qsa_append_finite_failures": self.batched_qsa_append_finite_failures,
        }

    def commit_boundary_stats(self) -> dict[str, int]:
        """Return monotonic counters for state evaluation at commit."""

        return {
            "evaluated_commit_boundary_calls": self.evaluated_commit_boundary_calls,
            "full_commit_boundary_calls": self.full_commit_boundary_calls,
        }

    def plain_serial_lane_stats(self) -> dict[str, int]:
        """Return monotonic counters for ordinary no-rollback decode."""

        return {
            "plain_serial_lane_entries": self.plain_serial_lane_entries,
            "plain_serial_steps": self.plain_serial_steps,
            "plain_serial_failures": self.plain_serial_failures,
        }

    def plain_serial_pipeline_stats(self) -> dict[str, int]:
        """Return monotonic counters for private one-row target pipelining."""

        return {
            "primes": self.plain_serial_pipeline_primes,
            "transitions": self.plain_serial_pipeline_transitions,
            "terminal_finishes": self.plain_serial_pipeline_terminal_finishes,
            "discards": self.plain_serial_pipeline_discards,
            "discard_drain_failures": self.plain_serial_pipeline_discard_drain_failures,
        }

    def compiled_gdn_run_stats(self) -> dict[str, int]:
        """Return monotonic counters for the guarded compiled GDN schedule."""

        return {
            "builds": self.compiled_gdn_run_builds,
            "calls": self.compiled_gdn_run_calls,
            "layer_calls": self.compiled_gdn_run_layer_calls,
            "ineligible_calls": self.compiled_gdn_run_ineligible_calls,
            "failures": self.compiled_gdn_run_failures,
        }

    def _eval_forward_boundary(
        self,
        output: mx.array,
        state: Qwen4CompositeState,
        append_finite_check: _Qwen4AppendFiniteCheck | None,
    ) -> None:
        """Evaluate output, state, and any deferred append predicate together."""

        arrays = (output, *tuple(_state_arrays(state)))
        if append_finite_check is None:
            mx.eval(*arrays)
            return
        self.batched_qsa_append_finite_batches += 1
        self.batched_qsa_append_finite_predicates += append_finite_check.predicate_count
        mx.eval(*arrays, append_finite_check.predicate)
        if not bool(append_finite_check.predicate.item()):
            self.batched_qsa_append_finite_failures += 1
            raise ValueError("mutable Qwen KVarN requires finite unpadded input")

    def _eval_serial_boundary(
        self,
        evaluated: tuple[mx.array, ...],
        state: Qwen4CompositeState,
        append_finite_check: _Qwen4AppendFiniteCheck | None,
    ) -> None:
        """Evaluate next-token sampling and private state at one boundary."""

        arrays = (*evaluated, *tuple(_state_arrays(state)))
        if append_finite_check is None:
            mx.eval(*arrays)
            return
        self.batched_qsa_append_finite_batches += 1
        self.batched_qsa_append_finite_predicates += append_finite_check.predicate_count
        mx.eval(*arrays, append_finite_check.predicate)
        if not bool(append_finite_check.predicate.item()):
            self.batched_qsa_append_finite_failures += 1
            raise ValueError("mutable Qwen KVarN requires finite unpadded input")

    def _compiled_gdn_run_eligible(self, state: Qwen4CompositeState) -> bool:
        """Recognize the exact released full-resident serial GDN schedule."""

        if (
            self.training
            or self.hidden_size != 2_560
            or self.branch_count != 4
            or self.expanded_size != 10_240
            or len(self.layers) != 48
            or len(state.layers) != 48
        ):
            return False
        capacities = getattr(
            self,
            "_moespresso_ssd_streaming_resolved_capacities",
            None,
        )
        if not _compiled_gdn_full_residency_matches(self.layers, capacities):
            return False
        if self._cache_routing_enabled and any(
            not layer.mlp.gate._cache_routing_provider.is_fully_resident()
            for layer in self.layers[2:]
        ):
            return False
        expected_qsa = set(range(3, 48, 4))
        if any(
            layer.mixer_kind != ("qsa" if index in expected_qsa else "gdn")
            or (layer.ple is not None) != (index == 1)
            for index, layer in enumerate(self.layers)
        ):
            return False
        from moespresso.runtime.qwen4.gdn import (
            Qwen4GDNState,
            _qwen4_gdn_prerouter_contract,
        )

        try:
            from mlx_kquant import qwen4_gdn_prerouter_q6
        except ImportError:
            return False
        if not callable(qwen4_gdn_prerouter_q6):
            return False
        for index, (layer, layer_state) in enumerate(zip(self.layers, state.layers, strict=True)):
            if index in expected_qsa:
                continue
            mixer_state = layer_state.mixer_state
            if not isinstance(mixer_state, Qwen4GDNState) or not _qwen4_gdn_prerouter_contract(
                layer,
                mixer_state,
            ):
                return False
        return True

    def _build_compiled_gdn_runs(self) -> dict[int, _Qwen4CompiledGDNRun]:
        """Compile the fixed released GDN segments without executing them."""

        from moespresso.runtime.qwen4.gdn import (
            Qwen4GDNState,
            qwen4_gdn_prerouter_step,
        )

        built: dict[int, _Qwen4CompiledGDNRun] = {}

        def make_eager_run(
            layers: tuple[Qwen4DecoderLayer, ...],
            spec: _Qwen4CompiledGDNRunSpec,
        ):
            def eager_run(
                hidden: mx.array,
                *values: mx.array,
            ) -> tuple[mx.array, ...]:
                cursor = 0
                if spec.expects_pending:
                    pending_output, pending_injection = values[:2]
                    cursor = 2
                else:
                    pending_output = pending_injection = None
                flat_states = values[cursor:]
                outputs: list[mx.array] = []
                for offset, layer in enumerate(layers):
                    state = Qwen4GDNState(
                        conv_state=flat_states[2 * offset],
                        recurrent_state=flat_states[2 * offset + 1],
                        offset=0,
                    )
                    result = qwen4_gdn_prerouter_step(
                        layer,
                        hidden,
                        state=state,
                        pending_output=pending_output,
                        pending_injection=pending_injection,
                        certified_all_valid=True,
                    )
                    if result is None:
                        raise RuntimeError("compiled Qwen GDN run refused the native envelope")
                    pending_output = layer.mlp(result.mlp_hidden)
                    pending_injection = result.injection
                    hidden = result.residual
                    outputs.extend((result.state.conv_state, result.state.recurrent_state))
                if spec.publishes_final_write:
                    hidden = gated_residual_write(
                        hidden,
                        pending_output,
                        pending_injection,
                    )
                    return (hidden, *outputs)
                return (hidden, pending_output, pending_injection, *outputs)

            return eager_run

        for spec in _QWEN4_COMPILED_GDN_RUN_SPECS:
            layers = tuple(self.layers[index] for index in spec.indices)

            built[spec.indices[0]] = _Qwen4CompiledGDNRun(
                spec=spec,
                function=mx.compile(make_eager_run(layers, spec)),
            )
            self.compiled_gdn_run_builds += 1
        return built

    def _compiled_gdn_runs_for_serial_step(
        self,
        state: Qwen4CompositeState,
    ) -> dict[int, _Qwen4CompiledGDNRun] | None:
        """Resolve the guarded schedule before any mutable QSA layer executes."""

        if not self._compiled_gdn_run_eligible(state):
            self.compiled_gdn_run_ineligible_calls += 1
            return None
        runs = self._compiled_gdn_runs
        if runs is None:
            try:
                runs = self._build_compiled_gdn_runs()
            except BaseException:
                self.compiled_gdn_run_failures += 1
                object.__setattr__(self, "_compiled_gdn_runs", None)
                return None
            object.__setattr__(self, "_compiled_gdn_runs", runs)
        return runs

    def _execute_compiled_gdn_run(
        self,
        run: _Qwen4CompiledGDNRun,
        hidden_states: mx.array,
        pending_output: mx.array | None,
        pending_injection: mx.array | None,
        state: Qwen4CompositeState,
        *,
        next_frontier: int,
        position_history: mx.array,
        ple_state: Qwen4PLEState | None,
    ) -> tuple[
        mx.array,
        mx.array | None,
        mx.array | None,
        tuple[Qwen4LayerState, ...],
    ]:
        """Replay one compiled segment and rebuild its public state objects."""

        from moespresso.runtime.qwen4.gdn import Qwen4GDNState

        spec = run.spec
        inputs: list[mx.array] = [hidden_states]
        if spec.expects_pending:
            if pending_output is None or pending_injection is None:
                raise RuntimeError("compiled Qwen GDN run requires a pending residual write")
            inputs.extend((pending_output, pending_injection))
        elif pending_output is not None or pending_injection is not None:
            raise RuntimeError("compiled Qwen GDN run received an unexpected pending write")
        for index in spec.indices:
            mixer_state = state.layers[index].mixer_state
            if not isinstance(mixer_state, Qwen4GDNState):
                raise RuntimeError("compiled Qwen GDN run received incompatible state")
            inputs.extend((mixer_state.conv_state, mixer_state.recurrent_state))
        try:
            outputs = tuple(run.function(*inputs))
            state_cursor = 1 if spec.publishes_final_write else 3
            expected_outputs = state_cursor + 2 * len(spec.indices)
            if len(outputs) != expected_outputs:
                raise RuntimeError("compiled Qwen GDN run returned an invalid result")
            hidden_states = outputs[0]
            if hidden_states.shape != (1, 1, self.expanded_size):
                raise RuntimeError("compiled Qwen GDN run returned invalid hidden state")
            if spec.publishes_final_write:
                pending_output = pending_injection = None
            else:
                pending_output, pending_injection = outputs[1:3]
                if pending_output.shape != (1, 1, self.hidden_size) or pending_injection.shape != (
                    1,
                    1,
                    self.branch_count,
                ):
                    raise RuntimeError("compiled Qwen GDN run returned an invalid pending write")
            layer_states = []
            for offset, index in enumerate(spec.indices):
                layer = self.layers[index]
                mixer_state = Qwen4GDNState(
                    conv_state=outputs[state_cursor + 2 * offset],
                    recurrent_state=outputs[state_cursor + 2 * offset + 1],
                    offset=next_frontier,
                )
                validator = getattr(
                    layer.mixer,
                    "validate_state_trusted",
                    layer.mixer.validate_state,
                )
                validator(
                    mixer_state,
                    expected_frontier=next_frontier,
                    position_history=position_history,
                )
                layer_states.append(
                    Qwen4LayerState(
                        mixer_kind="gdn",
                        mixer_state=mixer_state,
                        mixer_offset=next_frontier,
                        ple_state=ple_state if offset == 0 else None,
                    )
                )
        except BaseException:
            self.compiled_gdn_run_failures += 1
            raise
        self.compiled_gdn_run_calls += 1
        self.compiled_gdn_run_layer_calls += len(spec.indices)
        return hidden_states, pending_output, pending_injection, tuple(layer_states)

    def close(self) -> None:
        """Close package-backed resources owned by the loaded graph."""
        if self._closed:
            return
        if self._closing:
            raise RuntimeError("Qwen4 model close is already in progress")
        object.__setattr__(self, "_closing", True)
        try:
            resources = getattr(self, "_moespresso_qwen4_runtime_resources", None)
            if resources is not None:
                assert_quiescent = getattr(resources, "assert_quiescent", None)
                if callable(assert_quiescent):
                    assert_quiescent()
            for coordinator in tuple(self._coordinators):
                coordinator.close()
            for lane in tuple(self._plain_serial_lanes):
                lane.close()
            if resources is not None:
                resources.close()
                object.__setattr__(self, "_moespresso_qwen4_runtime_resources", None)
            object.__setattr__(self, "_closed", True)
            object.__setattr__(self, "_moespresso_qwen4_mtp_compiled_core", None)
        finally:
            object.__setattr__(self, "_compiled_gdn_runs", None)
            object.__setattr__(self, "_closing", False)

    def new_coordinator(self, batch_size: int) -> Qwen4StateCoordinator:
        self._require_open()
        coordinator = Qwen4StateCoordinator(self, self.new_state(batch_size))
        self._coordinators.add(coordinator)
        return coordinator

    def restore_coordinator(self, state: Qwen4CompositeState) -> Qwen4StateCoordinator:
        """Admit independently restored committed state under a new request owner."""
        self._require_open()
        coordinator = Qwen4StateCoordinator(self, state)
        coordinator._plain_all_valid_lineage = bool(mx.all(state.valid_history).item())
        self._coordinators.add(coordinator)
        return coordinator

    def _plain_serial_lane_eligible(self, state: Qwen4CompositeState) -> bool:
        """Return whether one private state can enter ordinary serial decode."""

        if (
            state.batch_size != 1
        ):
            return False
        for layer, layer_state in zip(self.layers, state.layers, strict=True):
            if layer.mixer_kind != "qsa":
                continue
            if layer_state.mixer_state is None:
                return False
            if not bool(
                getattr(
                    layer.mixer,
                    "supports_serial_irrevocable_append",
                    False,
                )
            ):
                return False
            if not callable(getattr(layer.mixer, "abandon_serial_state", None)):
                return False
        return True

    def _abandon_serial_state(self, state: Qwen4CompositeState) -> None:
        """Invalidate mutable QSA storage after an irreversible lane failure."""

        for layer, layer_state in zip(self.layers, state.layers, strict=True):
            if layer.mixer_kind != "qsa":
                continue
            abandon = getattr(layer.mixer, "abandon_serial_state", None)
            if callable(abandon):
                abandon(layer_state.mixer_state)

    def new_state(self, batch_size: int) -> Qwen4CompositeState:
        """Return an empty functional state for direct chunk execution."""
        self._require_open()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return Qwen4CompositeState(
            cache_identity=self.cache_identity,
            revision=0,
            frontier=0,
            batch_size=batch_size,
            valid_history=mx.zeros((batch_size, 0), dtype=mx.bool_),
            position_history=mx.zeros((3, batch_size, 0), dtype=mx.int64),
            layers=tuple(
                Qwen4LayerState(
                    mixer_kind=layer.mixer_kind,
                    mixer_state=None,
                    mixer_offset=0,
                )
                for layer in self.layers
            ),
        )

    def validate_state(self, state: Qwen4CompositeState) -> None:
        """Strictly validate externally supplied or restored composite state."""
        self._validate_state(state, trusted=False)

    def _validate_trusted_state(self, state: Qwen4CompositeState) -> None:
        """Validate state produced from this shell's committed lineage."""
        self._validate_state(state, trusted=True)

    def _validate_state(self, state: Qwen4CompositeState, *, trusted: bool) -> None:
        if state.cache_identity != self.cache_identity:
            raise ValueError("cache identity does not match this model")
        if (
            isinstance(state.revision, bool)
            or not isinstance(state.revision, int)
            or state.revision < 0
        ):
            raise ValueError("cache revision must be a nonnegative integer")
        if (
            isinstance(state.frontier, bool)
            or not isinstance(state.frontier, int)
            or state.frontier < 0
        ):
            raise ValueError("cache frontier must be a nonnegative integer")
        if state.batch_size <= 0:
            raise ValueError("cache batch size must be positive")
        if state.valid_history.shape != (state.batch_size, state.frontier):
            raise ValueError("valid-token history does not share the public frontier")
        if state.valid_history.dtype != mx.bool_:
            raise ValueError("valid-token history must be boolean")
        if state.position_history.shape != (3, state.batch_size, state.frontier):
            raise ValueError("position history does not share the public frontier")
        if state.position_history.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("position history must contain integers")
        if len(state.layers) != len(self.layers):
            raise ValueError("cache layer count does not match this model")
        for index, (layer, layer_state) in enumerate(zip(self.layers, state.layers, strict=True)):
            if layer_state.mixer_kind != layer.mixer_kind:
                raise ValueError(f"layer {index} mixer kind does not match the model")
            if layer_state.mixer_offset != state.frontier:
                raise ValueError(f"layer {index} mixer state is off the public frontier")
            if state.frontier and layer_state.mixer_state is None:
                raise ValueError(f"layer {index} has no mixer state at a live frontier")
            validator = layer.mixer.validate_state
            if trusted:
                validator = getattr(layer.mixer, "validate_state_trusted", validator)
            validator(
                layer_state.mixer_state,
                expected_frontier=state.frontier,
                position_history=state.position_history,
            )
            if layer.ple is None:
                if layer_state.ple_state is not None:
                    raise ValueError(f"layer {index} unexpectedly carries PLE state")
            else:
                layer.ple.validate_state(
                    layer_state.ple_state,
                    expected_frontier=state.frontier,
                    batch_size=state.batch_size,
                )

    def propose(
        self,
        coordinator: Qwen4StateCoordinator,
        input_ids: mx.array,
        *,
        valid_tokens: mx.array | None = None,
        position_ids: mx.array | None = None,
        capture_widened: bool = False,
    ) -> Qwen4Candidate:
        return self._propose(
            coordinator,
            input_ids,
            valid_tokens=valid_tokens,
            position_ids=position_ids,
            trusted=False,
            capture_widened=capture_widened,
        )

    def _propose(
        self,
        coordinator: Qwen4StateCoordinator,
        input_ids: mx.array,
        *,
        valid_tokens: mx.array | None,
        position_ids: mx.array | None,
        trusted: bool,
        capture_widened: bool = False,
    ) -> Qwen4Candidate:
        self._require_open()
        if self._cache_routing_enabled and input_ids.shape != (1, 1):
            raise ValueError("cache-conditioned routing supports single-token proposals only")
        if coordinator._model is not self:
            raise ValueError("state coordinator belongs to another model")
        base = coordinator.state
        if trusted:
            self._validate_trusted_state(base)
        else:
            self.validate_state(base)
        valid_tokens, semantic_positions = self._normalize_inputs(
            base,
            input_ids,
            valid_tokens=valid_tokens,
            position_ids=position_ids,
        )

        checkpoints = []
        logits = []
        widened = [] if capture_widened else None
        working = self._fork_state(base)
        try:
            for token_index in range(input_ids.shape[1]):
                token_ids = input_ids[:, token_index : token_index + 1]
                token_valid = valid_tokens[:, token_index : token_index + 1]
                token_positions = semantic_positions[:, :, token_index : token_index + 1]
                output, working, append_finite_check = self._forward_chunk(
                    working,
                    token_ids,
                    token_valid,
                    token_positions,
                    trusted=trusted,
                    cache_routing=True,
                    widened_capture=widened,
                )
                checkpoint = self._snapshot_state(working)
                self._eval_forward_boundary(output, checkpoint, append_finite_check)
                logits.append(output)
                checkpoints.append(checkpoint)
        except BaseException:
            self._restore_state(base)
            raise
        all_logits = mx.concatenate(logits, axis=1)
        checkpoints_tuple = tuple(checkpoints)
        evaluated_boundary = _Qwen4EvaluatedBoundary(
            coordinator_identity=coordinator._identity,
            cache_identity=base.cache_identity,
            base_revision=base.revision,
            base_frontier=base.frontier,
            checkpoints=checkpoints_tuple,
        )
        return Qwen4Candidate(
            coordinator_identity=coordinator._identity,
            cache_identity=base.cache_identity,
            base_revision=base.revision,
            base_frontier=base.frontier,
            logits=all_logits,
            checkpoints=checkpoints_tuple,
            trusted_lineage=trusted,
            _evaluated_boundary=evaluated_boundary,
            widened=mx.concatenate(widened, axis=1) if widened is not None else None,
        )

    def forward_chunk(
        self,
        state: Qwen4CompositeState,
        input_ids: mx.array,
        *,
        valid_tokens: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> tuple[mx.array, Qwen4CompositeState]:
        """Execute and commit one ordinary prefill or decode chunk."""

        return self._forward_committed_chunk(
            state,
            input_ids,
            valid_tokens=valid_tokens,
            position_ids=position_ids,
            trusted=False,
        )

    def _forward_committed_chunk(
        self,
        state: Qwen4CompositeState,
        input_ids: mx.array,
        *,
        valid_tokens: mx.array | None,
        position_ids: mx.array | None,
        trusted: bool,
        widened_capture: list[mx.array] | None = None,
    ) -> tuple[mx.array, Qwen4CompositeState]:
        """Commit a chunk from an external or coordinator-owned state."""

        self._require_open()
        if trusted:
            self._validate_trusted_state(state)
        else:
            self.validate_state(state)
        valid_tokens, semantic_positions = self._normalize_inputs(
            state,
            input_ids,
            valid_tokens=valid_tokens,
            position_ids=position_ids,
        )
        working = self._fork_state(state)
        try:
            logits, next_state, append_finite_check = self._forward_chunk(
                working,
                input_ids,
                valid_tokens,
                semantic_positions,
                trusted=trusted,
                widened_capture=widened_capture,
            )
            committed = self._snapshot_state(next_state)
            self._eval_forward_boundary(logits, committed, append_finite_check)
            committed = self._commit_state(committed)
        except BaseException:
            self._restore_state(state)
            raise
        committed = replace(committed, revision=state.revision + 1)
        return logits, committed

    def _normalize_inputs(
        self,
        state: Qwen4CompositeState,
        input_ids: mx.array,
        *,
        valid_tokens: mx.array | None,
        position_ids: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        if input_ids.ndim != 2 or input_ids.shape[0] != state.batch_size or input_ids.shape[1] <= 0:
            raise ValueError("input_ids must match the cache batch and contain tokens")
        if input_ids.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("input_ids must contain integers")
        if valid_tokens is None:
            valid_tokens = mx.ones(input_ids.shape, dtype=mx.bool_)
        if valid_tokens.shape != input_ids.shape or valid_tokens.dtype != mx.bool_:
            raise ValueError("valid_tokens must be a boolean input-id mask")

        semantic_positions = _normalize_position_ids(
            position_ids,
            batch_size=state.batch_size,
            token_count=input_ids.shape[1],
            physical_frontier=state.frontier,
        )
        return valid_tokens, semantic_positions

    def _forward_chunk(
        self,
        state: Qwen4CompositeState,
        input_ids: mx.array,
        valid_tokens: mx.array,
        position_ids: mx.array,
        *,
        trusted: bool,
        serial_capability: object | None = None,
        cache_routing: bool = False,
        widened_capture: list[mx.array] | None = None,
    ) -> tuple[mx.array, Qwen4CompositeState, _Qwen4AppendFiniteCheck | None]:
        serial_lane = serial_capability is _QWEN4_SERIAL_LANE_CAPABILITY
        use_cache_routing = self._cache_routing_enabled and (serial_lane or cache_routing)
        if serial_capability is not None and not serial_lane:
            raise ValueError("Qwen4 plain serial capability is invalid")
        if serial_lane and (not trusted or state.batch_size != 1 or input_ids.shape != (1, 1)):
            raise ValueError("Qwen4 plain serial lane requires trusted one-token decode")
        next_frontier = state.frontier + input_ids.shape[1]
        valid_history = mx.concatenate([state.valid_history, valid_tokens], axis=1)
        position_history = mx.concatenate([state.position_history, position_ids], axis=2)
        qsa_pairs = (
            tuple(
                (layer.mixer, layer_state.mixer_state)
                for layer, layer_state in zip(self.layers, state.layers, strict=True)
                if layer.mixer_kind == "qsa"
            )
            if trusted
            else ()
        )
        qsa_mixers = tuple(mixer for mixer, _ in qsa_pairs)
        mask_certificate = None
        if trusted:
            certificate_eligible = bool(qsa_mixers) and all(
                bool(
                    getattr(
                        mixer,
                        "supports_trusted_mask_certificate",
                        False,
                    )
                )
                for mixer in qsa_mixers
            )
            if certificate_eligible:
                prepared_undos: tuple[Any | None, ...] = (None,) * len(qsa_pairs)
                append_finite_batch = None
                undo_preparations: tuple[Any, ...] = ()
                undo_arrays: tuple[mx.array, ...] = ()
                if serial_lane:
                    if not all(
                        bool(
                            getattr(
                                mixer,
                                "supports_serial_irrevocable_append",
                                False,
                            )
                        )
                        for mixer in qsa_mixers
                    ):
                        raise ValueError("Qwen4 plain serial lane requires irreversible QSA append")
                    append_finite_batch = _Qwen4AppendFiniteBatch(
                        allowed_mixers=qsa_mixers,
                        source_states=tuple(source for _, source in qsa_pairs),
                    )
                elif (
                    input_ids.shape[1] == 1
                    and all(source is not None for _, source in qsa_pairs)
                ):
                    prepare_undo = tuple(
                        getattr(mixer, "_prepare_trusted_undo", None) for mixer in qsa_mixers
                    )
                    undo_array_getters = tuple(
                        getattr(mixer, "_trusted_undo_arrays", None) for mixer in qsa_mixers
                    )
                    mark_undo = tuple(
                        getattr(mixer, "_mark_trusted_undo_evaluated", None) for mixer in qsa_mixers
                    )
                    if (
                        all(callable(method) for method in prepare_undo)
                        and all(callable(method) for method in undo_array_getters)
                        and all(callable(method) for method in mark_undo)
                    ):
                        prepared = []
                        arrays = []
                        for (mixer, source), prepare, get_arrays in zip(
                            qsa_pairs,
                            prepare_undo,
                            undo_array_getters,
                            strict=True,
                        ):
                            reservation = prepare(
                                source,
                                new_tokens=1,
                                capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
                            )
                            prepared.append(reservation)
                            arrays.extend(
                                get_arrays(
                                    reservation,
                                    capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
                                )
                            )
                        undo_preparations = tuple(prepared)
                        undo_arrays = tuple(arrays)
                if serial_lane:
                    self.trusted_mask_certificate_builds += 1
                    mask_certificate = _Qwen4TrustedMaskCertificate(
                        issuer=self._trusted_mask_issuer,
                        scope=object(),
                        allowed_mixers=qsa_mixers,
                        source_states=tuple(source for _, source in qsa_pairs),
                        prepared_undos=(),
                        append_finite_batch=append_finite_batch,
                        serial_capability=_QWEN4_SERIAL_LANE_CAPABILITY,
                        valid_tokens=valid_tokens,
                        visible_history=valid_history,
                        current_frontier=state.frontier,
                        next_frontier=next_frontier,
                    )
                else:
                    self.trusted_all_valid_reductions += 1
                    all_valid = mx.all(valid_history)
                    if undo_arrays:
                        mx.eval(all_valid, *undo_arrays)
                    all_valid_value = bool(all_valid.item())
                if not serial_lane and all_valid_value:
                    if undo_preparations:
                        for mixer, reservation, mark in zip(
                            qsa_mixers,
                            undo_preparations,
                            mark_undo,
                            strict=True,
                        ):
                            mark(
                                reservation,
                                capability=_QWEN4_BATCHED_UNDO_CAPABILITY,
                            )
                        prepared_undos = undo_preparations
                        self.batched_qsa_undo_batches += 1
                        self.batched_qsa_undo_reservations += len(prepared_undos)
                        if all(
                            bool(
                                getattr(
                                    mixer,
                                    "supports_batched_append_finite",
                                    False,
                                )
                            )
                            for mixer in qsa_mixers
                        ):
                            append_finite_batch = _Qwen4AppendFiniteBatch(
                                allowed_mixers=qsa_mixers,
                                source_states=tuple(source for _, source in qsa_pairs),
                            )
                    self.trusted_mask_certificate_builds += 1
                    mask_certificate = _Qwen4TrustedMaskCertificate(
                        issuer=self._trusted_mask_issuer,
                        scope=object(),
                        allowed_mixers=qsa_mixers,
                        source_states=tuple(source for _, source in qsa_pairs),
                        prepared_undos=prepared_undos,
                        append_finite_batch=append_finite_batch,
                        valid_tokens=valid_tokens,
                        visible_history=valid_history,
                        current_frontier=state.frontier,
                        next_frontier=next_frontier,
                    )
            elif serial_lane and qsa_mixers:
                raise ValueError("Qwen4 plain serial lane requires certified QSA masks")
        shared_rope_factors = None
        if trusted and input_ids.shape[1] == 1 and qsa_mixers:
            signatures = tuple(
                getattr(mixer, "shared_rope_signature", None) for mixer in qsa_mixers
            )
            prepared_steps = tuple(
                getattr(mixer, "_step_trusted_prepared", None) for mixer in qsa_mixers
            )
            prepare_factors = getattr(
                qsa_mixers[0],
                "_prepare_shared_rope_factors",
                None,
            )
            if (
                signatures[0] is not None
                and all(signature == signatures[0] for signature in signatures)
                and all(callable(step) for step in prepared_steps)
                and callable(prepare_factors)
            ):
                shared_rope_factors = prepare_factors(position_ids)
                self.shared_qsa_rope_factor_builds += 1
        compiled_gdn_runs = self._compiled_gdn_runs_for_serial_step(state) if serial_lane else None
        bound_mask_mixers: list[Any] = []
        body_error: BaseException | None = None
        try:
            embedded = self.embedding(input_ids)
            if embedded.shape != (*input_ids.shape, self.hidden_size):
                raise ValueError("embedding output does not match the model hidden size")
            hidden_states = mx.tile(embedded, (1, 1, self.branch_count))

            layer_states = []
            pending_block_output = None
            pending_injection_weights = None
            compiled_skip_until = 0
            for index, (layer, previous) in enumerate(zip(self.layers, state.layers, strict=True)):
                if index < compiled_skip_until:
                    continue
                ple_state = None
                if pending_block_output is not None and layer.ple is not None:
                    hidden_states = gated_residual_write(
                        hidden_states,
                        pending_block_output,
                        pending_injection_weights,
                    )
                    pending_block_output = None
                    pending_injection_weights = None
                if layer.ple is not None:
                    ple_result = layer.ple(
                        hidden_states,
                        input_ids,
                        state=previous.ple_state,
                        valid_tokens=valid_tokens,
                    )
                    if ple_result.output.shape != hidden_states.shape:
                        raise ValueError(f"layer {index} PLE output has an invalid shape")
                    hidden_states = hidden_states + ple_result.output
                    ple_state = ple_result.state

                compiled_run = None if compiled_gdn_runs is None else compiled_gdn_runs.get(index)
                if compiled_run is not None:
                    (
                        hidden_states,
                        pending_block_output,
                        pending_injection_weights,
                        compiled_states,
                    ) = self._execute_compiled_gdn_run(
                        compiled_run,
                        hidden_states,
                        pending_block_output,
                        pending_injection_weights,
                        state,
                        next_frontier=next_frontier,
                        position_history=position_history,
                        ple_state=ple_state,
                    )
                    layer_states.extend(compiled_states)
                    compiled_skip_until = compiled_run.spec.indices[-1] + 1
                    continue

                prerouter_result = None
                if layer.mixer_kind == "gdn":
                    from moespresso.runtime.qwen4.gdn import qwen4_gdn_prerouter_step

                    prerouter_result = qwen4_gdn_prerouter_step(
                        layer,
                        hidden_states,
                        state=previous.mixer_state,
                        pending_output=pending_block_output,
                        pending_injection=pending_injection_weights,
                        certified_all_valid=mask_certificate is not None,
                    )

                if prerouter_result is None:
                    if pending_block_output is None:
                        mixed, residual, injection = layer.attention_residual(hidden_states)
                    else:
                        pending_read = getattr(
                            layer.attention_residual,
                            "read_with_pending",
                            None,
                        )
                        if callable(pending_read):
                            mixed, residual, injection = pending_read(
                                hidden_states,
                                pending_block_output,
                                pending_injection_weights,
                            )
                        else:
                            hidden_states = gated_residual_write(
                                hidden_states,
                                pending_block_output,
                                pending_injection_weights,
                            )
                            mixed, residual, injection = layer.attention_residual(hidden_states)
                        pending_block_output = None
                        pending_injection_weights = None
                    certified_step = getattr(
                        layer.mixer,
                        "_step_trusted_certified",
                        None,
                    )
                    prepared_step = getattr(
                        layer.mixer,
                        "_step_trusted_prepared",
                        None,
                    )
                    if shared_rope_factors is not None and callable(prepared_step):
                        if mask_certificate is not None:
                            bind_issuer = getattr(
                                layer.mixer,
                                "_bind_trusted_mask_issuer",
                                None,
                            )
                            if callable(bind_issuer):
                                bind_issuer(
                                    self._trusted_mask_issuer,
                                    mask_certificate.scope,
                                )
                                bound_mask_mixers.append(layer.mixer)
                        mixer_result = prepared_step(
                            mixed,
                            valid_tokens=valid_tokens,
                            visible_history=valid_history,
                            position_ids=position_ids,
                            state=previous.mixer_state,
                            mask_certificate=mask_certificate,
                            shared_rope_factors=shared_rope_factors,
                        )
                    elif mask_certificate is not None and callable(certified_step):
                        bind_issuer = getattr(
                            layer.mixer,
                            "_bind_trusted_mask_issuer",
                            None,
                        )
                        if callable(bind_issuer):
                            bind_issuer(
                                self._trusted_mask_issuer,
                                mask_certificate.scope,
                            )
                            bound_mask_mixers.append(layer.mixer)
                        mixer_result = certified_step(
                            mixed,
                            valid_tokens=valid_tokens,
                            visible_history=valid_history,
                            position_ids=position_ids,
                            state=previous.mixer_state,
                            mask_certificate=mask_certificate,
                        )
                    else:
                        mixer_step = getattr(layer.mixer, "step_trusted", layer.mixer)
                        mixer_result = mixer_step(
                            mixed,
                            valid_tokens=valid_tokens,
                            visible_history=valid_history,
                            position_ids=position_ids,
                            state=previous.mixer_state,
                        )
                    if mixer_result.output.shape != mixed.shape:
                        raise ValueError(f"layer {index} mixer output has an invalid shape")
                    mixer_state = mixer_result.state
                    mixer_frontier = mixer_result.frontier
                else:
                    mixed = prerouter_result.mlp_hidden
                    residual = prerouter_result.residual
                    injection = prerouter_result.injection
                    mixer_state = prerouter_result.state
                    mixer_frontier = prerouter_result.frontier
                    pending_block_output = None
                    pending_injection_weights = None

                if mixer_frontier != next_frontier:
                    raise ValueError(f"layer {index} mixer did not advance to the next frontier")
                validator = getattr(
                    layer.mixer,
                    "validate_state_trusted",
                    layer.mixer.validate_state,
                )
                validator(
                    mixer_state,
                    expected_frontier=next_frontier,
                    position_history=position_history,
                )
                if prerouter_result is None:
                    pending_read = getattr(layer.mlp_residual, "read_with_pending", None)
                    if callable(pending_read):
                        mixed, residual, injection = pending_read(
                            residual,
                            mixer_result.output,
                            injection,
                        )
                    else:
                        hidden_states = gated_residual_write(
                            residual,
                            mixer_result.output,
                            injection,
                        )
                        mixed, residual, injection = layer.mlp_residual(hidden_states)
                mlp_output = (layer.mlp(mixed, cache_routing=True) if use_cache_routing
                              else layer.mlp(mixed))
                if mlp_output.shape != mixed.shape:
                    raise ValueError(f"layer {index} MLP output has an invalid shape")
                next_layer_has_ple = (
                    index + 1 < len(self.layers) and self.layers[index + 1].ple is not None
                )
                if index + 1 == len(self.layers) or next_layer_has_ple:
                    hidden_states = gated_residual_write(residual, mlp_output, injection)
                else:
                    hidden_states = residual
                    pending_block_output = mlp_output
                    pending_injection_weights = injection
                layer_states.append(
                    Qwen4LayerState(
                        mixer_kind=layer.mixer_kind,
                        mixer_state=mixer_state,
                        mixer_offset=mixer_frontier,
                        ple_state=ple_state,
                    )
                )
            if pending_block_output is not None:
                raise RuntimeError("final Qwen layer left an unpublished residual write")
            hidden = self.final_residual(hidden_states)
            if hidden.shape != (*input_ids.shape, self.hidden_size):
                raise ValueError("final residual output does not match hidden_size")
            logits = self.lm_head(hidden)
            if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
                raise ValueError("lm_head output must have batch and token axes")
            next_state = Qwen4CompositeState(
                cache_identity=state.cache_identity,
                revision=state.revision,
                frontier=next_frontier,
                batch_size=state.batch_size,
                valid_history=valid_history,
                position_history=position_history,
                layers=tuple(layer_states),
            )
            self._validate_trusted_state(next_state)
            append_finite_check = None
            if mask_certificate is not None and mask_certificate.append_finite_batch is not None:
                result_pairs = tuple(
                    (layer.mixer, layer_state.mixer_state)
                    for layer, layer_state in zip(self.layers, layer_states, strict=True)
                    if layer.mixer_kind == "qsa"
                )
                append_finite_check = mask_certificate.append_finite_batch.seal(
                    result_pairs,
                    capability=_QWEN4_BATCHED_APPEND_FINITE_CAPABILITY,
                )
            if widened_capture is not None:
                widened_capture.append(hidden_states)
            return logits, next_state, append_finite_check
        except BaseException as exc:
            body_error = exc
            abort_pooled_request(self)
            raise
        finally:
            cleanup_error: BaseException | None = None
            if mask_certificate is not None:
                for mixer in reversed(bound_mask_mixers):
                    clear_issuer = getattr(
                        mixer,
                        "_clear_trusted_mask_scope",
                        None,
                    )
                    if callable(clear_issuer):
                        try:
                            clear_issuer(
                                self._trusted_mask_issuer,
                                mask_certificate.scope,
                            )
                        except BaseException as exc:
                            if cleanup_error is None:
                                cleanup_error = exc
            if body_error is None and cleanup_error is not None:
                raise cleanup_error

    def _fork_state(self, state: Qwen4CompositeState) -> Qwen4CompositeState:
        layers = tuple(
            replace(
                layer_state,
                mixer_state=layer.mixer.fork_state(layer_state.mixer_state),
            )
            for layer, layer_state in zip(self.layers, state.layers, strict=True)
        )
        return replace(state, layers=layers)

    def _snapshot_state(self, state: Qwen4CompositeState) -> Qwen4CompositeState:
        layers = tuple(
            replace(
                layer_state,
                mixer_state=layer.mixer.snapshot_state(layer_state.mixer_state),
            )
            for layer, layer_state in zip(self.layers, state.layers, strict=True)
        )
        snapshot = replace(state, layers=layers)
        self._validate_trusted_state(snapshot)
        return snapshot

    def _restore_state(self, state: Qwen4CompositeState) -> None:
        abort_pooled_request(self)
        for layer, layer_state in zip(self.layers, state.layers, strict=True):
            restore = getattr(layer.mixer, "restore_state", None)
            if callable(restore):
                restore(layer_state.mixer_state)

    def _commit_state(self, state: Qwen4CompositeState) -> Qwen4CompositeState:
        committers = []
        for layer, layer_state in zip(self.layers, state.layers, strict=True):
            preflight = getattr(layer.mixer, "preflight_commit_state", None)
            finalize = getattr(layer.mixer, "commit_state_preflighted", None)
            legacy = getattr(layer.mixer, "commit_state", None)
            if callable(preflight) != callable(finalize):
                raise TypeError("mixer has an incomplete preflighted commit seam")
            if callable(preflight):
                preflight(layer_state.mixer_state)
                committers.append(finalize)
            elif callable(legacy):
                raise TypeError("mutable mixer commit requires a preflighted seam")
            else:
                committers.append(None)
        layers = tuple(
            replace(
                layer_state,
                mixer_state=(
                    finalize(layer_state.mixer_state)
                    if callable(finalize)
                    else layer_state.mixer_state
                ),
            )
            for layer_state, finalize in zip(state.layers, committers, strict=True)
        )
        return replace(state, layers=layers)

    def _require_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("Qwen4 model is closed")


def _normalize_position_ids(
    position_ids: mx.array | None,
    *,
    batch_size: int,
    token_count: int,
    physical_frontier: int,
) -> mx.array:
    """Return the three semantic RoPE axes for the current physical rows."""
    if position_ids is None:
        physical_rows = mx.arange(
            physical_frontier,
            physical_frontier + token_count,
            dtype=mx.int64,
        )
        return mx.broadcast_to(physical_rows[None, None], (3, batch_size, token_count))
    if position_ids.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
        raise ValueError("position_ids must contain integers")
    if position_ids.ndim == 2:
        if position_ids.shape != (batch_size, token_count):
            raise ValueError("text position_ids must match input_ids")
        return mx.broadcast_to(position_ids[None], (3, batch_size, token_count))
    if position_ids.ndim != 3 or position_ids.shape[1:] != (batch_size, token_count):
        raise ValueError("position_ids must have text or MRoPE geometry")
    if position_ids.shape[0] == 1:
        return mx.broadcast_to(position_ids, (3, batch_size, token_count))
    if position_ids.shape[0] == 3:
        return position_ids
    if position_ids.shape[0] == 4:
        return position_ids[1:]
    raise ValueError("position_ids must carry one, three, or four position planes")


def _state_arrays(value: Any):
    if isinstance(value, mx.array):
        yield value
    elif callable(getattr(value, "state_arrays", None)):
        yield from value.state_arrays()
    elif is_dataclass(value):
        for field in fields(value):
            yield from _state_arrays(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _state_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _state_arrays(item)
