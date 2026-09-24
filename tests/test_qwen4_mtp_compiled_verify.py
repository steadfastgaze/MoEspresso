"""Selection and state contracts for short-context compiled MTP verification."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from moespresso.runtime.qwen4 import mtp_generate
from moespresso.runtime.qwen4.mtp_compiled_core import _append_and_gather_qsa_row
from moespresso.runtime.qwen4.mtp_compiled_verify import Qwen4MTPCompiledVerifier
from moespresso.runtime.qwen4.mtp_verify import Qwen4MTPVerifier
from moespresso.runtime.qwen4.qsa_kvarn import Qwen4MutableKVarNQSAState


def test_factory_selects_compiled_by_default_and_base_with_kill_switch(monkeypatch):
    compiled = object()
    base = object()
    monkeypatch.setattr(mtp_generate, "Qwen4MTPCompiledVerifier", lambda *a, **k: compiled)
    monkeypatch.setattr(mtp_generate, "Qwen4MTPVerifier", lambda *a, **k: base)
    monkeypatch.delenv("MOESPRESSO_QWEN4_MTP_COMPILED", raising=False)
    assert mtp_generate.make_mtp_verifier(object()) is compiled

    monkeypatch.setenv("MOESPRESSO_QWEN4_MTP_COMPILED", "0")
    assert mtp_generate.make_mtp_verifier(object()) is base


def test_fixed_qsa_bank_uses_live_tensor_offset():
    keys = mx.zeros((1, 1, 4, 1), dtype=mx.float32)
    values = mx.zeros_like(keys)
    row_keys = mx.array([[[[7.0]]]])
    row_values = mx.array([[[[9.0]]]])
    compiled = mx.compile(_append_and_gather_qsa_row)

    first = compiled(keys, values, row_keys, row_values, mx.array(0, dtype=mx.int32))
    third = compiled(keys, values, row_keys, row_values, mx.array(2, dtype=mx.int32))
    mx.eval(first, third)

    np.testing.assert_array_equal(np.asarray(first[0]).reshape(-1), [7, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(third[0]).reshape(-1), [0, 0, 7, 0])
    assert int(mx.sum(first[-1]).item()) == 1
    assert int(mx.sum(third[-1]).item()) == 3


def test_private_state_key_separates_coordinators_at_the_same_frontier():
    state = SimpleNamespace(cache_identity="model", revision=2, frontier=7)
    first = SimpleNamespace(_identity=object())
    second = SimpleNamespace(_identity=object())
    key = Qwen4MTPCompiledVerifier._state_key
    assert key(first, state) == key(first, state)
    assert key(first, state) != key(second, state)


def _eligible_verifier(frontier):
    verifier = object.__new__(Qwen4MTPCompiledVerifier)
    verifier._full_resident_prefix_enabled = True
    verifier.shared_projections = True
    verifier.gdn_pairs = {}
    qsa_state = Qwen4MutableKVarNQSAState(
        storage=object(), view=SimpleNamespace(frontier=frontier)
    )
    module = SimpleNamespace(token_budget=2048, compress_ratio=4)
    verifier.model = SimpleNamespace(layers=(
        SimpleNamespace(mixer_kind="qsa", mixer=SimpleNamespace(module=module)),
    ))
    state = SimpleNamespace(
        frontier=frontier,
        layers=(SimpleNamespace(mixer_state=qsa_state),),
    )
    return verifier, state


def test_compiled_selection_is_limited_to_exact_short_context():
    verifier, state = _eligible_verifier(2046)
    assert verifier._compiled_state_eligible(state)
    state.frontier = 2047
    assert not verifier._compiled_state_eligible(state)


def test_compiled_selection_preserves_projection_and_gdn_modes():
    verifier, state = _eligible_verifier(20)
    verifier.shared_projections = False
    assert not verifier._compiled_state_eligible(state)
    verifier.shared_projections = True
    verifier.gdn_pairs = {0: SimpleNamespace(native_boundaries=False)}
    assert not verifier._compiled_state_eligible(state)


def test_compiled_selection_preserves_sparse_attention_budget():
    verifier, state = _eligible_verifier(1024)
    verifier.model.layers[0].mixer.module.token_budget = 512
    assert not verifier._compiled_state_eligible(state)


def test_long_context_fallback_does_not_construct_core(monkeypatch):
    verifier, base = _eligible_verifier(2047)
    verifier._compiled_states = object()
    verifier._compiled_state_key = object()
    verifier._pending_compiled_states = object()
    verifier._round_kind = None
    sentinel = object()
    monkeypatch.setattr(
        Qwen4MTPVerifier,
        "_forward_pair",
        lambda *args, **kwargs: sentinel,
    )

    assert verifier._forward_pair(None, base, None, None, None, [], None) is sentinel
    assert verifier._compiled_states is None
    assert verifier._round_kind == "fallback"
    assert not hasattr(verifier.model, "_moespresso_qwen4_mtp_compiled_core")


@pytest.mark.parametrize("accepted,expected", [(0, "first"), (1, "final")])
def test_successful_compiled_round_retains_selected_private_checkpoint(
    monkeypatch, accepted, expected
):
    verifier = object.__new__(Qwen4MTPCompiledVerifier)
    verifier._pending_compiled_states = ("first", "final")
    verifier._round_kind = "compiled"
    verifier._compiled_states = None
    verifier._compiled_state_key = None
    verifier.compiled_rounds = 0
    verifier.fallback_rounds = 0
    state = SimpleNamespace(cache_identity="cache", revision=2, frontier=7)
    coordinator = SimpleNamespace(_identity=object())
    result = SimpleNamespace(acceptance=SimpleNamespace(accepted=accepted), state=state)
    monkeypatch.setattr(Qwen4MTPVerifier, "verify", lambda *args, **kwargs: result)

    assert verifier.verify(coordinator, None) is result
    assert verifier._compiled_states == expected
    assert verifier._compiled_state_key == (coordinator._identity, "cache", 2, 7)
    assert verifier.compiled_rounds == 1


def test_close_clears_private_banks_only_after_base_cleanup(monkeypatch):
    verifier = object.__new__(Qwen4MTPCompiledVerifier)
    verifier._compiled_states = "states"
    verifier._compiled_state_key = "key"
    verifier._pending_compiled_states = "pending"
    verifier._round_kind = "compiled"
    monkeypatch.setattr(Qwen4MTPVerifier, "close", lambda self: None)
    verifier.close()
    assert verifier._compiled_states is None
    assert verifier._pending_compiled_states is None

    verifier._compiled_states = "states"
    monkeypatch.setattr(
        Qwen4MTPVerifier,
        "close",
        lambda self: (_ for _ in ()).throw(RuntimeError("drain failed")),
    )
    with pytest.raises(RuntimeError, match="drain failed"):
        verifier.close()
    assert verifier._compiled_states == "states"


def test_shared_expert_counter_counts_compiled_graph_executions():
    verifier = object.__new__(Qwen4MTPCompiledVerifier)
    verifier.compiled_rounds = 2
    verifier.model = SimpleNamespace(layers=(None,) * 48)
    verifier.expert_pairs = (SimpleNamespace(paired_calls=3),)
    assert verifier.shared_expert_call_count() == 99
