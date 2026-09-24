"""Shared scheduling for routed MoE math over persistent expert pools."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import time


_SESSION_ATTRIBUTE = "_moespresso_pooled_decode_session"


def install_pooled_decode_session(model, switches):
    """Give every routed layer of a model the same execution owner."""
    from moespresso.runtime.pooled_decode_session import PooledDecodeSession

    if getattr(model, _SESSION_ATTRIBUTE, None) is not None:
        raise RuntimeError("pooled decode session is already installed")
    session = PooledDecodeSession()
    object.__setattr__(model, _SESSION_ATTRIBUTE, session)
    for switch in switches:
        object.__setattr__(switch, _SESSION_ATTRIBUTE, session)
    return session


def _synchronize_root(_root):
    import mlx.core as mx

    # Keep the root alive while draining submitted work. An aborted graph may
    # also contain invalid, unsubmitted nodes; do not evaluate those nodes.
    mx.synchronize()


@contextmanager
def pooled_request_scope(model, owner):
    """Keep routed writers and submitted readers inside one request lifetime."""
    session = getattr(model, _SESSION_ATTRIBUTE, None)
    if session is None:
        yield
    else:
        with session.request(owner, synchronize=_synchronize_root):
            yield


def pooled_generation_scope(function):
    """Own generic generation; composite-state models own their direct scopes."""

    @wraps(function)
    def generate(model, *args, **kwargs):
        if getattr(model, "_moespresso_owns_pooled_request_scope", False):
            return function(model, *args, **kwargs)
        with pooled_request_scope(model, object()):
            return function(model, *args, **kwargs)

    return generate


def pooled_state_scope(function):
    """Scope a composite-state coordinator or serial-lane operation."""

    @wraps(function)
    def advance(state_owner, *args, **kwargs):
        with pooled_request_scope(state_owner._model, state_owner._identity):
            return function(state_owner, *args, **kwargs)

    return advance


def abort_pooled_request(model):
    """Quiesce routed writes and readers before restoring mutable model state."""
    session = getattr(model, _SESSION_ATTRIBUTE, None)
    if session is not None and session.active:
        session.abort_and_drain()


def _switch_session(switch):
    session = getattr(switch, _SESSION_ATTRIBUTE, None)
    if session is None:
        from moespresso.runtime.pooled_decode_session import PooledDecodeSession

        session = PooledDecodeSession()
        object.__setattr__(switch, _SESSION_ATTRIBUTE, session)
    return session


def run_pooled_moe(
    switch,
    x,
    indices,
    scores,
    *,
    shared,
    reduce,
    resident=None,
    pipelined=None,
    direct=None,
    flush_resident=None,
    training=False,
    last=False,
    block_started=None,
):
    """Schedule one pooled MoE without changing its router or reduction math.

    Math callbacks return a reduced routed row. Full residency can omit all
    export and loading work. Bounded execution uses the same ordered ring
    publisher and optional event gate for every model family.
    """
    from moespresso.runtime import pooled_switchglu as pooled

    session = _switch_session(switch)
    if not session.active:
        # Direct single-block callers still have complete failure cleanup.
        # Model/generation scopes keep the owner across multiple routed layers.
        with session.request(switch, synchronize=_synchronize_root):
            return run_pooled_moe(
                switch,
                x,
                indices,
                scores,
                shared=shared,
                reduce=reduce,
                resident=resident,
                pipelined=pipelined,
                direct=direct,
                flush_resident=flush_resident,
                training=training,
                last=last,
                block_started=block_started,
            )
    session.require_current_request()

    object.__setattr__(
        switch,
        "shared_pooled_decode_calls",
        int(getattr(switch, "shared_pooled_decode_calls", 0)) + 1,
    )

    decode = pooled._token_layers(x) == 1
    if block_started is None and decode:
        block_started = time.perf_counter()

    def record(name, started):
        if decode:
            pooled._record_switch_seconds(switch, name, time.perf_counter() - started)

    def shared_output():
        started = time.perf_counter()
        result = shared(x)
        record("shared_experts_build_seconds", started)
        return result

    def join():
        started = time.perf_counter()
        session.drain()
        record("pipeline_join_seconds", started)

    def kick(root):
        session.remember(root)
        started = time.perf_counter()
        pooled._kick_eval(root)
        record("block_exit_kick_seconds", started)
        switch.block_exit_kick_calls += 1

    def finish(result):
        if decode:
            switch.decode_moe_block_calls += 1
            switch.decode_moe_block_seconds += time.perf_counter() - block_started
        return result

    if decode and not training and resident is not None:
        started = time.perf_counter()
        result = resident(x, indices, scores)
        record("routed_build_seconds", started)
        if result is not None:
            result = result + shared_output()
            if session.publication_pending or (last and session.pending):
                join()
            if flush_resident is not None:
                flush_resident(result)
            return finish(result)

    width = int(indices.shape[-1])
    supports_ring = pipelined is not None and all(
        callable(getattr(switch, name, None)) for name in ("export_inds", "ring_install")
    )
    capacity = min(pool.capacity for pool in switch._projection_pools_lockstep())
    if (
        decode
        and not training
        and supports_ring
        and pooled._RING_DECODE
        and width <= capacity
        and pooled._ring_visibility_ok()
    ):
        gate = pooled._gate_module()
        sequence = session.next_sequence(gate)
        pooled._RING_SEQ[0] = max(pooled._RING_SEQ[0], sequence)
        started = time.perf_counter()
        token = switch.export_inds(indices, sequence)
        record("router_export_seconds", started)
        started = time.perf_counter()
        event_gate = (gate, token, sequence) if gate is not None else None
        result = pipelined(x, indices, scores, event_gate=event_gate)
        record("routed_build_seconds", started)
        result = result + shared_output()
        if session.publication_pending:
            join()
        session.submit(
            pooled._PIPELINE_EXECUTOR,
            lambda cancelled: switch.ring_install(
                sequence,
                width,
                gate,
                cancelled=cancelled,
            ),
            publication_required=gate is None,
        )
        # Submission ownership precedes GPU commitment. A submit failure cannot
        # leave a committed wait with no cancellable writer behind it.
        session.remember(result)
        kick(result if gate is not None else token)
        session.remember(result)
        if last:
            join()
            if gate is None:
                kick(result)
        return finish(result)

    if session.publication_pending:
        join()
    ticket = None
    try:
        # The direct overlap uses a different executor from ring publication.
        # Join an earlier group, then own the direct batch before submission.
        session.drain()
        ticket = switch.begin_projection_load(indices, load_owner=session)
        shared_result = shared_output()
        if ticket is not None and ticket.has_work:
            if decode:
                started = time.perf_counter()
                session.remember(shared_result)
                pooled._kick_eval(shared_result)
                switch.overlap_shared_eval_calls += 1
                switch.overlap_shared_eval_seconds += time.perf_counter() - started
            else:
                switch.overlap_prefill_no_eval_calls += 1
        started = time.perf_counter()
        if direct is None:
            routed = switch(x, indices, load_ticket=ticket)
            result = reduce(routed, scores, indices)
        else:
            result = direct(x, indices, scores, load_ticket=ticket)
        record("routed_build_seconds", started)
        result = result + shared_result
    except BaseException:
        # The request owns the ticket batch from before its first submission.
        # Its outer failure path cancels and joins the batch before rollback.
        raise
    else:
        if ticket is not None and not ticket.used:
            session.drain()
    if last and session.pending:
        join()
    if decode:
        kick(result)
    elif bool(getattr(switch, "_all_iqk", False)):
        switch.commit_iqk_output(result, rows=pooled._token_layers(x))
    return finish(result)
