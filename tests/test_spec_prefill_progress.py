"""Paired target and DSpark progress during speculative prefill."""

from dataclasses import dataclass

import mlx.core as mx
import pytest

from moespresso.runtime.deepseek_v4 import spec_decode
from moespresso.runtime.deepseek_v4.spec_decode import (
    DraftProposal,
    FixedSubmitDrafter,
    SpecPrefillProgress,
    spec_generate,
)


@dataclass
class _State:
    frontier: int
    kv: mx.array


class _Cache:
    def __init__(self, offset: int):
        self.offset = offset
        self.state = mx.array([offset], dtype=mx.int32)


class _Tap:
    def __init__(self):
        self.active = False
        self.width = 0

    def take_rows(self):
        rows = mx.zeros((1, self.width, 1), dtype=mx.float32)
        self.width = 0
        return rows


class _Target:
    def __init__(self, tap, events, *, initial_offset=0, offset_delta=0):
        self.tap = tap
        self.events = events
        self.initial_offset = initial_offset
        self.offset_delta = offset_delta
        self.lm_head = lambda hidden: hidden

    def make_cache(self):
        return [_Cache(self.initial_offset)]

    def __call__(self, tokens, *, cache):
        width = int(tokens.shape[1])
        self.events.append(("target", width))
        self.tap.width = width
        cache[0].offset += width + self.offset_delta
        cache[0].state = mx.array([cache[0].offset], dtype=mx.int32)
        return mx.zeros((1, width, 17), dtype=mx.float32)


class _Drafter:
    greedy_only = False
    block_size = 1
    tap_layer_ids = ()

    def __init__(self, events, *, initial_frontier=0, frontier_delta=0):
        self.events = events
        self.initial_frontier = initial_frontier
        self.frontier_delta = frontier_delta

    def tap_transform(self, layer_id, out):
        return out

    def make_state(self):
        return _State(
            frontier=self.initial_frontier,
            kv=mx.array([self.initial_frontier], dtype=mx.int32),
        )

    def ingest(self, state, rows, positions, token_ids):
        self.events.append(("ingest", tuple(positions)))
        state.frontier = positions[-1] + 1 + self.frontier_delta
        state.kv = rows.sum() + mx.array([state.frontier], dtype=mx.float32)

    def state_frontier(self, state):
        self.events.append(("validate", state.frontier))
        return state.frontier

    def draft(self, state, anchor_token, anchor_pos, temperature):
        return DraftProposal(
            tokens=mx.array([[anchor_token]], dtype=mx.int64),
            logits=mx.zeros((1, 1, 17), dtype=mx.float32),
        )


class _PlainOnlyCost:
    def expected_ms(self, submitted):
        return 1.0 if submitted == 0 else 1_000_000.0

    def observe(self, submitted, wall_ms):
        return None


def _run(
    *,
    prompt=(1, 2, 3, 4, 5),
    prompt_offset=0,
    plan=(2, 1),
    max_new_tokens=1,
    target_offset_delta=0,
    drafter_frontier_delta=0,
    fixed_wrapper=False,
    monkeypatch=None,
    progress=None,
    events=None,
    progress_frontiers=None,
):
    if events is None:
        events = []
    tap = _Tap()
    target = _Target(
        tap,
        events,
        initial_offset=prompt_offset,
        offset_delta=target_offset_delta,
    )
    raw = _Drafter(
        events,
        initial_frontier=prompt_offset,
        frontier_delta=drafter_frontier_delta,
    )
    drafter = FixedSubmitDrafter(raw, 1) if fixed_wrapper else raw
    state = raw.make_state()
    cache = [_Cache(prompt_offset)]
    if progress is None:
        progress = []

    def record_progress(event):
        progress.append(event)
        events.append(("callback", event.frontier))

    if monkeypatch is not None:
        real_eval = mx.eval

        def traced_eval(*arrays):
            events.append(("materialize",))
            return real_eval(*arrays)

        monkeypatch.setattr(spec_decode.mx, "eval", traced_eval)

    result = spec_generate(
        target,
        drafter,
        tap,
        prompt,
        max_new_tokens=max_new_tokens,
        prefill_step_size=8,
        prefill_plan=plan,
        target_cache=cache,
        drafter_state=state,
        prompt_offset=prompt_offset,
        state_owner=raw,
        prefill_progress_callback=record_progress,
        prefill_progress_frontiers=progress_frontiers,
        calibration_warmup=0,
        probe_every=0,
        cost_model=_PlainOnlyCost(),
    )
    return result, progress, events, raw, state, cache


def test_paired_prefill_progress_orders_and_materializes_each_chunk(monkeypatch):
    _, progress, events, raw, state, cache = _run(monkeypatch=monkeypatch)

    assert [event.frontier for event in progress] == [2, 3, 4]
    assert [(event.processed, event.total) for event in progress] == [
        (2, 5),
        (3, 5),
        (4, 5),
    ]
    assert all(isinstance(event, SpecPrefillProgress) for event in progress)
    assert all(event.target_cache is cache for event in progress)
    assert all(event.drafter_state is state for event in progress)
    assert all(event.state_owner is raw for event in progress)

    chunk_events = [
        event for event in events
        if event[0] in {"target", "ingest", "materialize", "validate", "callback"}
    ]
    # The initial state validation precedes all model calls. Every callback is
    # reached only after its target, ingest, materialization, and validation.
    assert chunk_events[0] == ("validate", 0)
    for progress_event in progress:
        callback_index = events.index(("callback", progress_event.frontier))
        prefix = events[:callback_index]
        assert prefix[-4:] == [
            ("target", 2 if progress_event.frontier == 2 else 1),
            (
                "ingest",
                tuple(range(
                    0 if progress_event.frontier == 2 else progress_event.frontier - 1,
                    progress_event.frontier,
                )),
            ),
            ("materialize",),
            ("validate", progress_event.frontier),
        ]


def test_paired_prefill_progress_uses_absolute_restored_frontiers():
    _, progress, _, _, _, _ = _run(
        prompt=(7, 8, 9, 10),
        prompt_offset=256,
        plan=(2,),
    )

    assert [(event.processed, event.total, event.frontier) for event in progress] == [
        (2, 4, 258),
        (3, 4, 259),
    ]


def test_paired_prefill_frontier_filter_avoids_later_drafter_materialization(
    monkeypatch,
):
    drafter_materializations = []
    real_arrays = spec_decode._drafter_state_arrays

    def record_arrays(state):
        drafter_materializations.append(state)
        return real_arrays(state)

    monkeypatch.setattr(spec_decode, "_drafter_state_arrays", record_arrays)
    _, progress, events, _, state, _ = _run(
        progress_frontiers=(2,),
    )

    assert [event.frontier for event in progress] == [2]
    assert drafter_materializations == [state]
    assert ("validate", 3) not in events
    assert ("validate", 4) not in events


@pytest.mark.parametrize(
    "frontiers,error",
    [
        ((1,), "chunk ends"),
        ((5,), "chunk ends"),
        ((2, 2), "unique and ascending"),
        ((2.0,), "must be ints"),
    ],
)
def test_paired_prefill_frontier_filter_fails_before_model_forward(
    frontiers, error,
):
    events = []
    with pytest.raises(ValueError, match=error):
        _run(progress_frontiers=frontiers, events=events)
    assert not any(event[0] == "target" for event in events)


@pytest.mark.parametrize("plan", [(True,), (1.0,), ("1",)])
def test_spec_prefill_plan_rejects_non_int_chunks(plan):
    events = []
    with pytest.raises(ValueError, match="positive ints"):
        _run(plan=plan, events=events)
    assert not any(event[0] == "target" for event in events)


@pytest.mark.parametrize(
    "target_delta,drafter_delta,error",
    [
        (1, 0, "target cache"),
        (0, 1, "drafter state"),
    ],
)
def test_paired_prefill_frontier_mismatch_refuses_without_callback(
    target_delta, drafter_delta, error,
):
    progress = []
    with pytest.raises(RuntimeError, match=error):
        _run(
            target_offset_delta=target_delta,
            drafter_frontier_delta=drafter_delta,
            progress=progress,
        )
    assert progress == []


def test_paired_callback_excludes_anchor_decode_and_uses_raw_fixed_owner():
    result, progress, events, raw, _, _ = _run(
        prompt=(1, 2, 3, 4),
        plan=(2,),
        max_new_tokens=2,
        fixed_wrapper=True,
    )

    assert len(result.tokens) == 2
    assert [event.frontier for event in progress] == [2, 3]
    assert all(event.state_owner is raw for event in progress)
    # The final prompt anchor and the plain decode fallback both forward one
    # token, but neither can create another prefill-progress event.
    assert [event for event in events if event[0] == "target"] == [
        ("target", 2),
        ("target", 1),
        ("target", 1),
        ("target", 1),
    ]
