"""Atomic MTP verification preserves accepted target frontiers and ownership."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import moespresso.runtime.qwen4.mtp_verify as module
from moespresso.runtime.pooled_moe import install_pooled_decode_session, pooled_request_scope
from moespresso.runtime.qwen4.mtp_verify import Qwen4MTPVerifier, _read
from test_qwen4_model_shell import _model, _model_with_real_qsa, _state_arrays


class _RowReferenceGDN:
    """A layer-local reference; real paired GDN arithmetic has separate tests."""

    def __init__(self, layer):
        self.layer = layer

    def __call__(self, hidden, *, state, pending_output, pending_injection):
        layer = self.layer
        parts, states = [], []
        for row in range(2):
            mixed, residual, weights = _read(
                layer.attention_residual,
                hidden[:, row : row + 1],
                None if pending_output is None else pending_output[:, row : row + 1],
                None if pending_injection is None else pending_injection[:, row : row + 1],
            )
            frontier = 0 if state is None else state
            result = layer.mixer(
                mixed,
                valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
                visible_history=mx.ones((1, frontier + 1), dtype=mx.bool_),
                position_ids=mx.full((3, 1, 1), frontier, dtype=mx.int64),
                state=state,
            )
            state = result.state
            states.append(layer.mixer.snapshot_state(state))
            parts.append(_read(layer.mlp_residual, residual, result.output, weights))
        return SimpleNamespace(
            mlp_hidden=mx.concatenate([p[0] for p in parts], axis=1),
            residual=mx.concatenate([p[1] for p in parts], axis=1),
            injection=mx.concatenate([p[2] for p in parts], axis=1),
            first_state=states[0],
            final_state=states[1],
        )


@pytest.fixture
def paired_adapters(monkeypatch):
    observed = []

    def expert_pair(mlp):
        def run(hidden):
            observed.append(hidden)
            return mx.concatenate([mlp(hidden[:, row : row + 1]) for row in range(2)], axis=1)

        return run

    monkeypatch.setattr(module, "Qwen4MTPGDNPair", _RowReferenceGDN)
    monkeypatch.setattr(module, "Qwen4MTPFullResidentExpertPair", expert_pair)
    return observed


def _assert_state_equal(actual, expected):
    assert actual.frontier == expected.frontier
    assert actual.revision == expected.revision
    for left, right in zip(_state_arrays(actual), _state_arrays(expected), strict=True):
        np.testing.assert_array_equal(left, right)


def test_full_resident_prefix_submits_four_layer_boundaries_only(monkeypatch):
    class FullResidentPair:
        pass

    monkeypatch.setattr(module, "Qwen4MTPFullResidentExpertPair", FullResidentPair)
    verifier = object.__new__(Qwen4MTPVerifier)
    verifier.expert_pairs = tuple(FullResidentPair() for _ in range(48))
    verifier._full_resident_prefix_enabled = all(
        type(pair) is module.Qwen4MTPFullResidentExpertPair for pair in verifier.expert_pairs
    )
    verifier.prefix_submissions = 0
    submitted = []
    monkeypatch.setattr(module.mx, "async_eval", lambda output: submitted.append(output))

    for layer in range(48):
        verifier._submit_full_resident_prefix(layer, layer_index=layer)

    assert submitted == list(range(3, 44, 4))
    assert verifier.prefix_submissions == 11


def test_unmarked_expert_adapter_never_submits_prefixes(monkeypatch):
    verifier = object.__new__(Qwen4MTPVerifier)
    verifier.expert_pairs = (object(),) * 48
    verifier._full_resident_prefix_enabled = False
    verifier.prefix_submissions = 0
    monkeypatch.setattr(module.mx, "async_eval", lambda *_args: pytest.fail("legacy prefix submit"))

    for layer in range(48):
        verifier._submit_full_resident_prefix(object(), layer_index=layer)

    assert verifier.prefix_submissions == 0


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("factory", [lambda: _model([]), _model_with_real_qsa])
def test_pair_matches_serial_logits_state_and_continuation(paired_adapters, accepted, factory):
    mx.random.seed(19)
    model = factory()
    mx.random.seed(19)
    reference = factory().new_coordinator(1)
    coordinator = model.new_coordinator(1)
    prompt = mx.array([[1, 2, 3]], dtype=mx.int64)
    coordinator.forward_chunk(prompt)
    reference.forward_chunk(prompt)
    anchor = mx.array([[1]], dtype=mx.int64)
    probe = reference.propose(anchor)
    wanted = int(mx.argmax(probe.logits[0, 0]).item())
    reference.commit(probe, 0)
    draft = wanted if accepted else 1 - wanted
    inputs = mx.array([[1, draft]], dtype=mx.int64)
    expected = reference.propose(inputs, capture_widened=True)
    keep = 2 if accepted else 1
    reference.commit(expected, keep)
    verifier = Qwen4MTPVerifier(model)
    result = verifier.verify(coordinator, inputs)
    assert result.acceptance.accepted == int(accepted)
    assert len(paired_adapters) == len(model.layers)
    assert all(hidden.shape[1] == 2 for hidden in paired_adapters)
    np.testing.assert_array_equal(np.asarray(result.logits), np.asarray(expected.logits[:, :keep]))
    np.testing.assert_array_equal(
        np.asarray(result.widened), np.asarray(expected.widened[:, :keep])
    )
    _assert_state_equal(coordinator.state, reference.state)
    continuation = mx.array([[result.acceptance.next_token]], dtype=mx.int64)
    actual_next = coordinator.propose(continuation, capture_widened=True)
    expected_next = reference.propose(continuation, capture_widened=True)
    np.testing.assert_array_equal(np.asarray(actual_next.logits), np.asarray(expected_next.logits))
    coordinator.commit(actual_next, 1)
    reference.commit(expected_next, 1)
    _assert_state_equal(coordinator.state, reference.state)
    assert verifier.rounds == 1 and verifier.accepted_drafts == int(accepted)


def test_pair_uses_one_pooled_owner_through_acceptance_and_commit(paired_adapters, monkeypatch):
    model = _model([])
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1]]))
    session = install_pooled_decode_session(model, [])
    original = module.greedy_accept

    def accept(*args):
        assert session.active
        with pytest.raises(RuntimeError, match="different owner"):
            with pooled_request_scope(model, object()):
                pass
        return original(*args)

    monkeypatch.setattr(module, "greedy_accept", accept)
    Qwen4MTPVerifier(model).verify(coordinator, mx.array([[1, 1]]))
    assert not session.active and not session.pending
    assert len(paired_adapters) == len(model.layers)


def test_pair_publication_failure_precedes_acceptance(paired_adapters, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    model = _model([])
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1]]))
    session = install_pooled_decode_session(model, [])
    verifier = Qwen4MTPVerifier(model)
    pair = verifier.expert_pairs[0]

    def forbidden(*args):
        raise AssertionError("failed publication reached acceptance")

    def failure(_cancelled):
        raise RuntimeError("injected publication failure")

    with ThreadPoolExecutor(max_workers=1) as executor:

        def submit(hidden):
            session.next_sequence()
            session.submit(executor, failure)
            return pair(hidden)

        verifier.expert_pairs = (submit, *verifier.expert_pairs[1:])
        monkeypatch.setattr(module, "greedy_accept", forbidden)
        with pytest.raises(RuntimeError, match="injected publication failure"):
            verifier.verify(coordinator, mx.array([[1, 1]]))
    assert verifier.rounds == 0 and verifier._inflight is None
    assert not session.active and not session.pending


def test_pair_cancels_publication_before_restoring_state(paired_adapters, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    model = _model([])
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1]]))
    session = install_pooled_decode_session(model, [])
    verifier = Qwen4MTPVerifier(model)
    pair = verifier.expert_pairs[0]
    started, finished = threading.Event(), threading.Event()
    restore = model._restore_state

    def worker(cancelled):
        started.set()
        try:
            deadline = time.monotonic() + 2
            while not cancelled():
                if time.monotonic() >= deadline:
                    raise RuntimeError("publication was not cancelled")
                time.sleep(0.0001)
        finally:
            finished.set()

    def restored(state):
        assert finished.is_set(), "state restored while a publication writer is live"
        return restore(state)

    def fail_head(_hidden):
        raise RuntimeError("injected head failure")

    with ThreadPoolExecutor(max_workers=1) as executor:

        def submit(hidden):
            session.next_sequence()
            session.submit(executor, worker)
            assert started.wait(1)
            return pair(hidden)

        verifier.expert_pairs = (submit, *verifier.expert_pairs[1:])
        monkeypatch.setattr(model, "_restore_state", restored)
        model.lm_head = fail_head
        with pytest.raises(RuntimeError, match="injected head failure"):
            verifier.verify(coordinator, mx.array([[1, 1]]))
    assert finished.is_set() and verifier._inflight is None
    assert not session.active and not session.pending


@pytest.mark.parametrize("failure", ["cancel", "head", "accept", "commit"])
def test_pair_failure_restores_state_and_poisons_coordinator(
    paired_adapters, monkeypatch, failure
):
    model = _model([])
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1]]))
    verifier = Qwen4MTPVerifier(model)

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    cancelled = None
    if failure == "cancel":

        def cancelled():
            return len(paired_adapters) == len(model.layers)
    elif failure == "head":
        model.lm_head = fail
    elif failure == "accept":
        monkeypatch.setattr(module, "greedy_accept", fail)
    else:
        monkeypatch.setattr(coordinator, "commit", fail)
    error_type = InterruptedError if failure == "cancel" else RuntimeError
    with pytest.raises(error_type, match="injected failure|cancelled"):
        verifier.verify(coordinator, mx.array([[1, 1]]), cancelled=cancelled)
    assert verifier._inflight is None
    with pytest.raises(RuntimeError, match="coordinator is closed"):
        _ = coordinator.state
    with pytest.raises(RuntimeError, match="failed or unfinished"):
        verifier.verify(coordinator, mx.array([[1, 1]]))


def test_failed_cleanup_drain_retains_inflight_until_close_retry(paired_adapters, monkeypatch):
    model = _model([])
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1]]))
    verifier = Qwen4MTPVerifier(model)
    synchronize = mx.synchronize

    def fail():
        raise RuntimeError("injected drain failure")

    def fail_head(_hidden):
        raise RuntimeError("injected head failure")

    model.lm_head = fail_head
    monkeypatch.setattr(mx, "synchronize", fail)
    with pytest.raises(RuntimeError, match="head failure") as raised:
        verifier.verify(coordinator, mx.array([[1, 1]]))
    assert any("cleanup requires another drain" in note for note in raised.value.__notes__)
    assert verifier._inflight is not None
    monkeypatch.setattr(mx, "synchronize", synchronize)
    verifier.close()
    assert verifier._inflight is None


def test_pair_refuses_non_decode_or_masked_lineage_before_pool_work(paired_adapters):
    model = _model([])
    coordinator = model.new_coordinator(1)
    verifier = Qwen4MTPVerifier(model)
    with pytest.raises(ValueError, match="all-valid prefill"):
        verifier.verify(coordinator, mx.array([[1, 1]]))
    coordinator.forward_chunk(mx.array([[1]]), valid_tokens=mx.array([[False]]))
    with pytest.raises(ValueError, match="all-valid prefill"):
        verifier.verify(coordinator, mx.array([[1, 1]]))
    assert paired_adapters == []


def _kvarn_model(*, layer_count=1):
    from moespresso.runtime.qwen4.model import Qwen4DecoderLayer, Qwen4TextModelShell
    from moespresso.runtime.qwen4.qsa import Qwen4QSAAdapter
    from moespresso.runtime.qwen4.qsa_kvarn import Qwen4MutableKVarNQSAStateBackend
    from test_qwen4_qsa_kvarn_adapter import FIRST_SEAL, _module

    def embedding(ids):
        return ids[..., None].astype(mx.bfloat16) * 0.1 + mx.array(
            [0.1, 0.2, 0.3, 0.4], dtype=mx.bfloat16
        )

    def residual(hidden):
        mixed = mx.sum(hidden.reshape(*hidden.shape[:-1], 2, 4), axis=-2)
        return mixed, hidden, mx.full((*hidden.shape[:-1], 2), 0.01, dtype=hidden.dtype)

    def final(hidden):
        return mx.sum(hidden.reshape(*hidden.shape[:-1], 2, 4), axis=-2)

    def head(hidden):
        return hidden.astype(mx.float32) + mx.array([0, 0, 0, 100], dtype=mx.float32)

    layers = []
    for index in range(layer_count):
        attention = _module(seed=701 + index)
        layers.append(
            Qwen4DecoderLayer(
                mixer_kind="qsa",
                attention_residual=residual,
                mixer=Qwen4QSAAdapter(
                    attention,
                    state_backend=Qwen4MutableKVarNQSAStateBackend(
                        attention, max_context_tokens=FIRST_SEAL + 64
                    ),
                ),
                mlp_residual=residual,
                mlp=lambda h: h * 0.01,
            )
        )
    return Qwen4TextModelShell(
        cache_identity="mtp-mutable-qsa-test",
        embedding=embedding,
        layers=tuple(layers),
        final_residual=final,
        lm_head=head,
        hidden_size=4,
        branch_count=2,
    )


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("cross_seal", [False, True])
def test_pair_mutable_kvarn_prefix_snapshot_seal_and_continuation(
    paired_adapters, accepted, cross_seal
):
    from test_qwen4_qsa_kvarn_adapter import FIRST_SEAL, _assert_mutable_state_bytes_exact

    model, reference_model = _kvarn_model(), _kvarn_model()
    coordinator, reference = model.new_coordinator(1), reference_model.new_coordinator(1)
    length = FIRST_SEAL - 1 if cross_seal else 3
    for begin in range(0, length, 64):
        tokens = (mx.arange(begin, min(begin + 64, length), dtype=mx.int32) % 4)[None]
        coordinator.forward_chunk(tokens)
        reference.forward_chunk(tokens)
    dependency_bound = coordinator.state.layers[
        0
    ].mixer_state.storage.counters.undo_reservations_dependency_bound
    inputs = mx.array([[1, 3 if accepted else 0]], dtype=mx.int32)
    expected = reference.propose(inputs, capture_widened=True)
    keep = 2 if accepted else 1
    reference.commit(expected, keep)
    result = Qwen4MTPVerifier(model).verify(coordinator, inputs)
    assert result.acceptance.accepted == int(accepted)
    np.testing.assert_array_equal(np.asarray(result.logits), np.asarray(expected.logits[:, :keep]))
    np.testing.assert_array_equal(
        np.asarray(result.widened.view(mx.uint16)),
        np.asarray(expected.widened[:, :keep].view(mx.uint16)),
    )
    _assert_mutable_state_bytes_exact(
        coordinator.state.layers[0].mixer_state, reference.state.layers[0].mixer_state
    )
    counters = coordinator.state.layers[0].mixer_state.storage.counters
    assert counters.undo_reservations_dependency_bound - dependency_bound == 1
    next_ids = mx.array([[result.acceptance.next_token]], dtype=mx.int32)
    actual = coordinator.propose(next_ids)
    expected = reference.propose(next_ids)
    np.testing.assert_array_equal(np.asarray(actual.logits), np.asarray(expected.logits))
    coordinator.commit(actual, 1)
    reference.commit(expected, 1)
    _assert_mutable_state_bytes_exact(
        coordinator.state.layers[0].mixer_state, reference.state.layers[0].mixer_state
    )


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("frontier", [255, pytest.param("first-seal", id="first-seal")])
def test_prefix_submission_preserves_mutable_kvarn_boundaries(
    paired_adapters,
    accepted,
    frontier,
):
    from test_qwen4_qsa_kvarn_adapter import FIRST_SEAL, _assert_mutable_state_bytes_exact

    length = FIRST_SEAL - 1 if frontier == "first-seal" else frontier
    model, reference_model = _kvarn_model(layer_count=5), _kvarn_model(layer_count=5)
    coordinator, reference = model.new_coordinator(1), reference_model.new_coordinator(1)
    for begin in range(0, length, 256):
        tokens = (mx.arange(begin, min(begin + 256, length), dtype=mx.int32) % 4)[None]
        coordinator.forward_chunk(tokens)
        reference.forward_chunk(tokens)
    inputs = mx.array([[1, 3 if accepted else 0]], dtype=mx.int32)
    expected = reference.propose(inputs, capture_widened=True)
    keep = 2 if accepted else 1
    reference.commit(expected, keep)
    verifier = Qwen4MTPVerifier(model)
    verifier._full_resident_prefix_enabled = True
    result = verifier.verify(coordinator, inputs)
    assert verifier.prefix_submissions == 1
    assert result.acceptance.accepted == int(accepted)
    np.testing.assert_array_equal(np.asarray(result.logits), np.asarray(expected.logits[:, :keep]))
    for actual_layer, expected_layer in zip(
        coordinator.state.layers,
        reference.state.layers,
        strict=True,
    ):
        _assert_mutable_state_bytes_exact(actual_layer.mixer_state, expected_layer.mixer_state)


def test_pair_reuses_certified_qsa_controls_with_distinct_prefix_undo(paired_adapters, monkeypatch):
    model = _kvarn_model()
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[0, 1, 2]], dtype=mx.int32))
    mixer = model.layers[0].mixer
    storage = coordinator.state.layers[0].mixer_state.storage
    before = dict(storage.stats)
    masks, rope = mixer.trusted_mask_certificate_calls, mixer.shared_rope_factor_calls
    prepare = mixer._prepare_trusted_undo
    frontiers = []

    def recorded(state, **kwargs):
        frontiers.append((state.view.frontier, kwargs["new_tokens"]))
        return prepare(state, **kwargs)

    monkeypatch.setattr(mixer, "_prepare_trusted_undo", recorded)
    result = Qwen4MTPVerifier(model).verify(coordinator, mx.array([[1, 3]], dtype=mx.int32))
    assert result.acceptance.accepted == 1
    assert mixer.trusted_mask_certificate_calls - masks == 2
    assert mixer.shared_rope_factor_calls - rope == 2
    assert frontiers == [(3, 1), (4, 1)]
    assert storage.stats["local_undo_evals"] == before["local_undo_evals"]
    assert storage.stats["undo_reservations_consumed"] - before["undo_reservations_consumed"] == 2


def test_pair_can_force_eager_second_undo_for_comparison(paired_adapters, monkeypatch):
    model = _kvarn_model()
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[0, 1, 2]], dtype=mx.int32))
    storage = coordinator.state.layers[0].mixer_state.storage
    evaluated = storage.counters.undo_reservations_evaluated
    dependency_bound = storage.counters.undo_reservations_dependency_bound
    rows = module.Qwen4MTPQSARows

    def eager(*args, **kwargs):
        return rows(*args, **kwargs, defer_undo=False)

    monkeypatch.setattr(module, "Qwen4MTPQSARows", eager)
    Qwen4MTPVerifier(model).verify(coordinator, mx.array([[1, 3]], dtype=mx.int32))

    assert storage.counters.undo_reservations_evaluated - evaluated == 2
    assert storage.counters.undo_reservations_dependency_bound == dependency_bound


@pytest.mark.parametrize("bad_row", [0, 1])
def test_pair_deferred_finite_failure_precedes_acceptance_and_restores_cache(
    paired_adapters, monkeypatch, bad_row
):
    from test_qwen4_qsa_kvarn_adapter import _mutable_storage_bytes

    model = _kvarn_model()
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[0, 1, 2]], dtype=mx.int32))
    state = coordinator.state.layers[0].mixer_state
    before = _mutable_storage_bytes(state)
    mixer = model.layers[0].mixer
    step = mixer._step_trusted_prepared
    calls = []

    def injected(hidden, **kwargs):
        row = len(calls)
        calls.append(row)
        if row == bad_row:
            hidden = mx.full(hidden.shape, float("nan"), dtype=hidden.dtype)
        return step(hidden, **kwargs)

    def forbidden(*args):
        raise AssertionError("nonfinite verification reached acceptance")

    monkeypatch.setattr(mixer, "_step_trusted_prepared", injected)
    monkeypatch.setattr(module, "greedy_accept", forbidden)
    verifier = Qwen4MTPVerifier(model)
    with pytest.raises(ValueError, match="finite unpadded"):
        verifier.verify(coordinator, mx.array([[1, 3]], dtype=mx.int32))
    assert model.batched_qsa_append_finite_failures == 1
    assert verifier._inflight is None
    assert mixer._trusted_mask_scope is None
    storage = state.storage
    assert storage.frontier == 3 and storage.raw_index_count == 3 and storage.index_group_count == 0
    restored = _mutable_storage_bytes(state)
    # Newly appended compressed-index rows are outside the restored frontier;
    # rollback restores their count, not the unreachable physical bytes.
    assert restored[:5] + restored[7:] == before[:5] + before[7:]
    assert paired_adapters


def test_pair_batches_first_undo_across_qsa_layers(paired_adapters):
    model = _kvarn_model()
    other = _kvarn_model()
    model.layers = (*model.layers, *other.layers)
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[0, 1, 2]], dtype=mx.int32))
    batches, reservations = model.batched_qsa_undo_batches, model.batched_qsa_undo_reservations
    predicates = model.batched_qsa_append_finite_predicates
    Qwen4MTPVerifier(model).verify(coordinator, mx.array([[1, 3]], dtype=mx.int32))
    assert model.batched_qsa_undo_batches - batches == 3
    assert model.batched_qsa_undo_reservations - reservations == 4
    assert model.batched_qsa_append_finite_predicates - predicates == 4


def test_plain_fallback_matches_serial_state_and_logits(paired_adapters):
    model, reference_model = _model([]), _model([])
    coordinator = model.new_coordinator(1)
    reference = reference_model.new_coordinator(1)
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    coordinator.forward_chunk(prompt)
    reference.forward_chunk(prompt)
    expected = reference.propose(mx.array([[1]], dtype=mx.int32), capture_widened=True)
    reference.commit(expected, 1)
    verifier = Qwen4MTPVerifier(model)
    result = verifier.verify_plain(coordinator, mx.array([[1]], dtype=mx.int32))

    np.testing.assert_array_equal(np.asarray(result.logits), np.asarray(expected.logits))
    np.testing.assert_array_equal(np.asarray(result.widened), np.asarray(expected.widened))
    _assert_state_equal(result.state, reference.state)
    assert paired_adapters == []
    assert verifier.plain_steps == 1


def test_plain_fallback_commit_failure_invalidates_coordinator(paired_adapters, monkeypatch):
    model = _model([])
    coordinator = model.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1]], dtype=mx.int32))
    verifier = Qwen4MTPVerifier(model)

    def fail(*_args):
        raise RuntimeError("injected plain commit failure")

    monkeypatch.setattr(coordinator, "commit", fail)
    with pytest.raises(RuntimeError, match="plain commit failure"):
        verifier.verify_plain(coordinator, mx.array([[1]], dtype=mx.int32))
    assert verifier._inflight is None and verifier.plain_steps == 0
    assert paired_adapters == []
    with pytest.raises(RuntimeError, match="coordinator is closed"):
        _ = coordinator.state
