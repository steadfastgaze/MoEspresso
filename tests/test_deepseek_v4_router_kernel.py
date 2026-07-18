from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from moespresso.runtime.deepseek_v4 import router_kernel


def _require_metal():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    return mx


def _jang_model():
    return pytest.importorskip("jang_tools.dsv4.mlx_model")


def _assert_bits_equal(a, b, label):
    a_bits = np.asarray(a, dtype=np.float32).view(np.uint32)
    b_bits = np.asarray(b, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(a_bits, b_bits, err_msg=label)


def _composed_head(mx, gates, bias):
    scores = mx.sqrt(mx.log1p(mx.exp(gates)))
    return -(scores + bias), scores


def _composed_weights(mx, orig, sel, scaling):
    picked = mx.take_along_axis(orig, sel.astype(mx.int32), axis=-1)
    picked = picked / mx.sum(picked, axis=-1, keepdims=True)
    return picked * scaling


def _gates_case(name, rng, n_experts):
    if name == "normal":
        return rng.normal(0.0, 4.0, n_experts).astype(np.float32)
    if name == "wide_magnitude":
        vals = rng.normal(0.0, 1.0, n_experts).astype(np.float32)
        vals[:8] = [30.0, -30.0, 10.0, 1.0, 0.0, -5.0, 20.0, 15.0]
        return vals
    if name == "special_values":
        vals = rng.normal(0.0, 30.0, n_experts).astype(np.float32)
        vals[:14] = [
            0.0, -0.0, np.inf, -np.inf, np.nan, 1e38, -1e38,
            88.8, 89.0, -87.0, 1e-38, -1e-38, 5e-39, 700.0,
        ]
        return vals
    raise AssertionError(name)


@pytest.mark.parametrize("case", ["normal", "wide_magnitude", "special_values"])
@pytest.mark.parametrize("bias_dtype_name", ["float32", "float16", "bfloat16"])
def test_fused_score_head_bitexact(case, bias_dtype_name):
    mx = _require_metal()
    rng = np.random.default_rng(31)
    n_experts = 256
    gates = mx.array(_gates_case(case, rng, n_experts)).reshape(1, 1, n_experts)
    bias = mx.array(rng.normal(0.0, 1.0, n_experts).astype(np.float32))
    bias = bias.astype(getattr(mx, bias_dtype_name))

    neg, orig = router_kernel.fused_score_head(gates, bias)
    exp_neg, exp_orig = _composed_head(mx, gates, bias)
    mx.eval(neg, orig, exp_neg, exp_orig)

    assert neg.shape == gates.shape and orig.shape == gates.shape
    assert neg.dtype == mx.float32 and orig.dtype == mx.float32
    _assert_bits_equal(neg, exp_neg, f"neg [{case}/{bias_dtype_name}]")
    _assert_bits_equal(orig, exp_orig, f"orig [{case}/{bias_dtype_name}]")


@pytest.mark.parametrize("n_experts", [64, 288, 300])
def test_fused_score_head_bitexact_expert_counts(n_experts):
    mx = _require_metal()
    rng = np.random.default_rng(5)
    gates = mx.array(
        rng.normal(0.0, 4.0, n_experts).astype(np.float32)
    ).reshape(1, 1, n_experts)
    bias = mx.array(rng.normal(0.0, 1.0, n_experts).astype(np.float32))

    neg, orig = router_kernel.fused_score_head(gates, bias)
    exp_neg, exp_orig = _composed_head(mx, gates, bias)
    mx.eval(neg, orig, exp_neg, exp_orig)
    _assert_bits_equal(neg, exp_neg, f"neg [{n_experts}]")
    _assert_bits_equal(orig, exp_orig, f"orig [{n_experts}]")


@pytest.mark.parametrize("k", [2, 6, 8, 64])
@pytest.mark.parametrize("sel_dtype_name", ["uint32", "int32"])
def test_fused_topk_weights_bitexact(k, sel_dtype_name):
    mx = _require_metal()
    rng = np.random.default_rng(k)
    n_experts = 256
    orig = mx.array(
        np.abs(rng.normal(0.5, 0.3, n_experts)).astype(np.float32)
    ).reshape(1, 1, n_experts)
    sel_np = rng.choice(n_experts, size=k, replace=False).astype(np.int64)
    sel = mx.array(sel_np.reshape(1, 1, k)).astype(getattr(mx, sel_dtype_name))

    got = router_kernel.fused_topk_weights(orig, sel, scaling=1.5)
    expected = _composed_weights(mx, orig, sel, 1.5)
    mx.eval(got, expected)

    assert got.shape == (1, 1, k)
    assert got.dtype == mx.float32
    _assert_bits_equal(got, expected, f"weights [k={k}/{sel_dtype_name}]")


def test_fused_topk_weights_bitexact_special_values():
    mx = _require_metal()
    n_experts = 32
    vals = np.linspace(0.1, 1.0, n_experts).astype(np.float32)
    vals[3] = np.inf
    vals[5] = np.nan
    vals[7] = 0.0
    orig = mx.array(vals).reshape(1, 1, n_experts)
    sel = mx.array(
        np.array([3, 5, 7, 1, 2, 9], dtype=np.uint32).reshape(1, 1, 6))

    got = router_kernel.fused_topk_weights(orig, sel, scaling=1.5)
    expected = _composed_weights(mx, orig, sel, 1.5)
    mx.eval(got, expected)
    _assert_bits_equal(got, expected, "weights [special]")


@pytest.mark.parametrize("k", [2, 6, 64])
def test_trimmed_select_matches_sqrtsoftplus_select(k):
    mx = _require_metal()
    dsv4 = _jang_model()
    rng = np.random.default_rng(97)
    n_experts = 256
    for trial in range(8):
        gates = mx.array(
            rng.normal(0.0, 4.0, n_experts).astype(np.float32)
        ).reshape(1, 1, n_experts)
        bias = mx.array(rng.normal(0.0, 1.0, n_experts).astype(np.float32))
        ref_inds, ref_scores = dsv4.sqrtsoftplus_select(
            gates, bias, k, 1.5, True)

        neg, orig = router_kernel.fused_score_head(gates, bias)
        sel = mx.argpartition(neg, kth=k - 1, axis=-1)[..., :k]
        weights = router_kernel.fused_topk_weights(orig, sel, scaling=1.5)
        mx.eval(ref_inds, ref_scores, sel, weights)

        np.testing.assert_array_equal(
            np.asarray(ref_inds).astype(np.int64),
            np.asarray(sel).astype(np.int64),
            err_msg=f"ids [k={k} trial={trial}]",
        )
        _assert_bits_equal(
            ref_scores, weights, f"weights [k={k} trial={trial}]")


def test_trimmed_select_boundary_ties_match_composed():
    mx = _require_metal()
    dsv4 = _jang_model()
    n_experts, k = 256, 6
    # Twenty identical top candidates across the k boundary: the selected
    # subset and its order are implementation-defined in mx.argpartition,
    # and both paths must make the identical choice.
    vals = np.full(n_experts, -1.0, dtype=np.float32)
    vals[10:30] = 2.0
    gates = mx.array(vals).reshape(1, 1, n_experts)
    bias = mx.zeros((n_experts,), dtype=mx.float32)
    ref_inds, ref_scores = dsv4.sqrtsoftplus_select(gates, bias, k, 1.5, True)

    neg, orig = router_kernel.fused_score_head(gates, bias)
    sel = mx.argpartition(neg, kth=k - 1, axis=-1)[..., :k]
    weights = router_kernel.fused_topk_weights(orig, sel, scaling=1.5)
    mx.eval(ref_inds, ref_scores, sel, weights)

    np.testing.assert_array_equal(
        np.asarray(ref_inds).astype(np.int64),
        np.asarray(sel).astype(np.int64),
        err_msg="tie ids",
    )
    _assert_bits_equal(ref_scores, weights, "tie weights")


def test_router_kernel_rejects_bad_inputs():
    mx = pytest.importorskip("mlx.core")

    good_gates = mx.zeros((1, 1, 64), dtype=mx.float32)
    good_bias = mx.zeros((64,), dtype=mx.float32)
    with pytest.raises(ValueError, match="shape \\[1, 1, n_experts\\]"):
        router_kernel.fused_score_head(mx.zeros((1, 64)), good_bias)
    with pytest.raises(ValueError, match="one decode token"):
        router_kernel.fused_score_head(
            mx.zeros((1, 2, 64), dtype=mx.float32), good_bias)
    with pytest.raises(ValueError, match="must be float32"):
        router_kernel.fused_score_head(
            good_gates.astype(mx.float16), good_bias)
    with pytest.raises(ValueError, match="bias"):
        router_kernel.fused_score_head(
            good_gates, mx.zeros((32,), dtype=mx.float32))
    with pytest.raises(ValueError, match="float32, float16, or bfloat16"):
        router_kernel.fused_score_head(
            good_gates, good_bias.astype(mx.int32))

    good_orig = mx.zeros((1, 1, 64), dtype=mx.float32)
    good_sel = mx.zeros((1, 1, 6), dtype=mx.uint32)
    with pytest.raises(ValueError, match="must be float32"):
        router_kernel.fused_topk_weights(
            good_orig.astype(mx.float16), good_sel, scaling=1.5)
    with pytest.raises(ValueError, match="2 to 64"):
        router_kernel.fused_topk_weights(
            good_orig, mx.zeros((1, 1, 1), dtype=mx.uint32), scaling=1.5)
    with pytest.raises(ValueError, match="2 to 64"):
        router_kernel.fused_topk_weights(
            good_orig, mx.zeros((1, 1, 65), dtype=mx.uint32), scaling=1.5)
    with pytest.raises(ValueError, match="uint32 or int32"):
        router_kernel.fused_topk_weights(
            good_orig, good_sel.astype(mx.float32), scaling=1.5)
    with pytest.raises(ValueError, match="finite"):
        router_kernel.fused_topk_weights(
            good_orig, good_sel, scaling=float("nan"))


def test_eligibility_predicates(monkeypatch):
    mx = _require_metal()
    monkeypatch.delenv(router_kernel._ROUTER_TRIMS_ENV, raising=False)
    monkeypatch.delenv(router_kernel._ROUTER_SELECT_ENV, raising=False)

    gates = mx.zeros((1, 1, 64), dtype=mx.float32)
    bias = mx.zeros((64,), dtype=mx.float32)
    orig = mx.zeros((1, 1, 64), dtype=mx.float32)
    sel = mx.zeros((1, 1, 6), dtype=mx.uint32)
    assert router_kernel.score_head_eligible(gates, bias)
    assert router_kernel.topk_weights_eligible(orig, sel)

    # Off-contract shapes and dtypes fail closed.
    assert not router_kernel.score_head_eligible(
        mx.zeros((1, 2, 64), dtype=mx.float32), bias)
    assert not router_kernel.score_head_eligible(
        gates.astype(mx.float16), bias)
    assert not router_kernel.score_head_eligible(
        gates, mx.zeros((32,), dtype=mx.float32))
    assert not router_kernel.score_head_eligible(
        gates, bias.astype(mx.int32))
    assert not router_kernel.topk_weights_eligible(
        orig, mx.zeros((1, 1, 1), dtype=mx.uint32))
    assert not router_kernel.topk_weights_eligible(
        orig, mx.zeros((1, 1, 65), dtype=mx.uint32))
    assert not router_kernel.topk_weights_eligible(
        orig, sel.astype(mx.float32))

    # The family and per-piece kill switches are read per call.
    monkeypatch.setenv(router_kernel._ROUTER_TRIMS_ENV, "0")
    assert not router_kernel.score_head_eligible(gates, bias)
    assert not router_kernel.router_precast_enabled()
    monkeypatch.delenv(router_kernel._ROUTER_TRIMS_ENV)
    monkeypatch.setenv(router_kernel._ROUTER_SELECT_ENV, "0")
    assert not router_kernel.score_head_eligible(gates, bias)
    assert router_kernel.router_precast_enabled()
    monkeypatch.delenv(router_kernel._ROUTER_SELECT_ENV)
    monkeypatch.setenv(router_kernel._ROUTER_PRECAST_ENV, "0")
    assert not router_kernel.router_precast_enabled()
    assert router_kernel.router_select_enabled()


def _model_args(dsv4, *, hidden=128, experts=64, k=6, hash_layers=1):
    return dsv4.ModelArgs(
        hidden_size=hidden,
        n_routed_experts=experts,
        num_experts_per_tok=k,
        num_hash_layers=hash_layers,
        vocab_size=64,
    )


def _make_gate(mx, dsv4, *, layer_id, seed=17, args=None):
    args = args or _model_args(dsv4)
    gate = dsv4.Gate(args, layer_id)
    rng = np.random.default_rng(seed)
    gate.weight = mx.array(
        rng.normal(0.0, 0.5, (args.n_routed_experts, args.hidden_size))
        .astype(np.float32)
    ).astype(mx.float16)
    if gate.hash:
        gate.tid2eid = mx.array(
            rng.integers(
                0,
                args.n_routed_experts,
                size=(args.vocab_size, args.num_experts_per_tok),
            ).astype(np.int32)
        )
    else:
        gate.bias = mx.array(
            rng.normal(0.0, 0.2, args.n_routed_experts).astype(np.float32))
    return gate, args


def _gate_contract(gate):
    import mlx.core as mx

    from moespresso.runtime.deepseek_v4.model import (
        _DeepseekV4RouterGateContract,
    )

    dsv4 = _jang_model()
    return _DeepseekV4RouterGateContract(gate, mx=mx, dsv4_model=dsv4)


def _assert_gate_parity(mx, expected, got, label):
    exp_inds, exp_scores = expected
    got_inds, got_scores = got
    mx.eval(exp_inds, exp_scores, got_inds, got_scores)
    np.testing.assert_array_equal(
        np.asarray(exp_inds).astype(np.int64),
        np.asarray(got_inds).astype(np.int64),
        err_msg=f"ids [{label}]",
    )
    _assert_bits_equal(exp_scores, got_scores, f"weights [{label}]")


@pytest.mark.parametrize("x_dtype_name", ["float32", "bfloat16"])
def test_gate_contract_decode_parity_nonhash(monkeypatch, x_dtype_name):
    mx = _require_metal()
    dsv4 = _jang_model()
    from moespresso.runtime.deepseek_v4 import model as ds4m

    monkeypatch.delenv(router_kernel._ROUTER_TRIMS_ENV, raising=False)
    gate, args = _make_gate(mx, dsv4, layer_id=2)
    contract = _gate_contract(gate)
    rng = np.random.default_rng(3)
    x = mx.array(
        rng.normal(0.0, 1.0, (1, 1, args.hidden_size)).astype(np.float32)
    ).astype(getattr(mx, x_dtype_name))

    before = ds4m.router_gate_trim_call_counts()
    got = contract(x)
    after = ds4m.router_gate_trim_call_counts()
    assert after["precast"] == before["precast"] + 1
    assert after["select_kernel"] == before["select_kernel"] + 1
    assert got[0].dtype == mx.uint32
    _assert_gate_parity(mx, gate(x), got, f"decode {x_dtype_name}")


def test_gate_contract_prefill_parity_nonhash(monkeypatch):
    mx = _require_metal()
    dsv4 = _jang_model()
    from moespresso.runtime.deepseek_v4 import model as ds4m

    monkeypatch.delenv(router_kernel._ROUTER_TRIMS_ENV, raising=False)
    gate, args = _make_gate(mx, dsv4, layer_id=2)
    contract = _gate_contract(gate)
    rng = np.random.default_rng(4)
    x = mx.array(
        rng.normal(0.0, 1.0, (1, 5, args.hidden_size)).astype(np.float32))

    before = ds4m.router_gate_trim_call_counts()
    got = contract(x)
    after = ds4m.router_gate_trim_call_counts()
    # Prefill keeps the composed select; the hoisted operand still engages.
    assert after["precast"] == before["precast"] + 1
    assert after["select_kernel"] == before["select_kernel"]
    assert after["select_composed"] == before["select_composed"] + 1
    _assert_gate_parity(mx, gate(x), got, "prefill")


def test_gate_contract_hash_parity(monkeypatch):
    mx = _require_metal()
    dsv4 = _jang_model()

    monkeypatch.delenv(router_kernel._ROUTER_TRIMS_ENV, raising=False)
    gate, args = _make_gate(mx, dsv4, layer_id=0)
    assert gate.hash
    contract = _gate_contract(gate)
    rng = np.random.default_rng(6)
    x = mx.array(
        rng.normal(0.0, 1.0, (1, 1, args.hidden_size)).astype(np.float32))
    input_ids = mx.array(np.array([[7]], dtype=np.int32))

    got = contract(x, input_ids=input_ids)
    _assert_gate_parity(
        mx, gate(x, input_ids=input_ids), got, "hash decode")


def test_gate_contract_kill_switches(monkeypatch):
    mx = _require_metal()
    dsv4 = _jang_model()
    from moespresso.runtime.deepseek_v4 import model as ds4m

    gate, args = _make_gate(mx, dsv4, layer_id=2)
    contract = _gate_contract(gate)
    rng = np.random.default_rng(9)
    x = mx.array(
        rng.normal(0.0, 1.0, (1, 1, args.hidden_size)).astype(np.float32))

    # Family kill: full delegation to the stock gate.
    monkeypatch.setenv(router_kernel._ROUTER_TRIMS_ENV, "0")
    before = ds4m.router_gate_trim_call_counts()
    got = contract(x)
    after = ds4m.router_gate_trim_call_counts()
    assert after["composed"] == before["composed"] + 1
    assert after["precast"] == before["precast"]
    assert after["select_kernel"] == before["select_kernel"]
    _assert_gate_parity(mx, gate(x), got, "family kill")
    monkeypatch.delenv(router_kernel._ROUTER_TRIMS_ENV)

    # Precast kill: per-call weight cast with the fused select.
    monkeypatch.setenv(router_kernel._ROUTER_PRECAST_ENV, "0")
    before = ds4m.router_gate_trim_call_counts()
    got = contract(x)
    after = ds4m.router_gate_trim_call_counts()
    assert after["precast"] == before["precast"]
    assert after["select_kernel"] == before["select_kernel"] + 1
    _assert_gate_parity(mx, gate(x), got, "precast kill")
    monkeypatch.delenv(router_kernel._ROUTER_PRECAST_ENV)

    # Select kill: hoisted operand with the composed select.
    monkeypatch.setenv(router_kernel._ROUTER_SELECT_ENV, "0")
    before = ds4m.router_gate_trim_call_counts()
    got = contract(x)
    after = ds4m.router_gate_trim_call_counts()
    assert after["precast"] == before["precast"] + 1
    assert after["select_kernel"] == before["select_kernel"]
    assert after["select_composed"] == before["select_composed"] + 1
    _assert_gate_parity(mx, gate(x), got, "select kill")
    monkeypatch.delenv(router_kernel._ROUTER_SELECT_ENV)


def test_gate_contract_fails_closed_on_static_contract(monkeypatch):
    mx = _require_metal()
    dsv4 = _jang_model()

    monkeypatch.delenv(router_kernel._ROUTER_TRIMS_ENV, raising=False)
    args = _model_args(dsv4, k=1)
    gate, args = _make_gate(mx, dsv4, layer_id=2, args=args)
    contract = _gate_contract(gate)
    assert not contract._select_static_ok()

    args2 = _model_args(dsv4)
    args2.norm_topk_prob = False
    gate2, args2 = _make_gate(mx, dsv4, layer_id=2, args=args2)
    contract2 = _gate_contract(gate2)
    assert not contract2._select_static_ok()
    rng = np.random.default_rng(12)
    x = mx.array(
        rng.normal(0.0, 1.0, (1, 1, args2.hidden_size)).astype(np.float32))
    _assert_gate_parity(mx, gate2(x), contract2(x), "norm_topk_prob off")


def test_patch_installs_contract_and_is_idempotent(monkeypatch):
    mx = _require_metal()
    dsv4 = _jang_model()
    from moespresso.runtime.deepseek_v4.model import (
        _DeepseekV4RouterGateContract,
        _patch_deepseek_v4_router_gate_trims,
    )

    gate, _args = _make_gate(mx, dsv4, layer_id=2)
    hash_gate, _hash_args = _make_gate(mx, dsv4, layer_id=0)

    class _ForeignGate:
        pass

    layers = [
        SimpleNamespace(mlp=SimpleNamespace(gate=gate)),
        SimpleNamespace(mlp=SimpleNamespace(gate=hash_gate)),
        SimpleNamespace(mlp=SimpleNamespace(gate=_ForeignGate())),
        SimpleNamespace(mlp=None),
    ]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    assert _patch_deepseek_v4_router_gate_trims(model) == 2
    assert isinstance(layers[0].mlp.gate, _DeepseekV4RouterGateContract)
    assert isinstance(layers[1].mlp.gate, _DeepseekV4RouterGateContract)
    assert isinstance(layers[2].mlp.gate, _ForeignGate)
    assert model._moespresso_dsv4_router_gate_trim_layers == 2

    # Re-running finds the contracts installed and patches nothing.
    assert _patch_deepseek_v4_router_gate_trims(model) == 0
    assert layers[0].mlp.gate._original is gate


def test_router_trim_counters_export_keys():
    from moespresso.runtime.deepseek_v4.model import (
        _ROUTER_GATE_TRIM_CALL_COUNTS,
        router_gate_trim_call_counts,
    )
    from moespresso.runtime.deepseek_v4.speed_stats import _COUNT_KEYS

    counts = router_gate_trim_call_counts()
    assert set(counts) == {
        "precast", "select_kernel", "select_composed", "composed"}
    assert counts is not _ROUTER_GATE_TRIM_CALL_COUNTS
    assert "router_gate_precast_calls" in _COUNT_KEYS
    assert "router_gate_select_kernel_calls" in _COUNT_KEYS
    assert "router_gate_select_composed_calls" in _COUNT_KEYS
    assert "router_gate_composed_calls" in _COUNT_KEYS
