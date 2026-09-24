from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from moespresso.correctness.qwen4.architecture_reference import (
    qwen4_sparse_moe,
    qwen4_topk_router,
)
import moespresso.runtime.qwen4.moe as moe_module
from moespresso.runtime.qwen4.moe import (
    Qwen4ExpertStack,
    Qwen4SparseMoEBlock,
    Qwen4TopKRouter,
    expert_major_weighted_sum,
)


def _array(values: np.ndarray) -> mx.array:
    return mx.array(values)


class _RecordingExpertExecutor:
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.calls: list[tuple[mx.array, mx.array]] = []

    def __call__(self, hidden_states: mx.array, indices: mx.array) -> mx.array:
        self.calls.append((hidden_states, indices))
        values = indices.astype(hidden_states.dtype)[..., None]
        return mx.broadcast_to(values, (*indices.shape, self.hidden_size))


class _OptionalDecodeExecutor(_RecordingExpertExecutor):
    def __init__(self, *args, enabled: bool = True) -> None:
        super().__init__(*args)
        self.enabled = enabled
        self.decode_attempts = 0

    def try_full_resident_decode(
        self,
        hidden_states: mx.array,
        indices: mx.array,
    ) -> mx.array | None:
        self.decode_attempts += 1
        rows = int(np.prod(hidden_states.shape[:-1]))
        if not self.enabled or rows != 1:
            return None
        values = indices.astype(hidden_states.dtype)[..., None]
        return mx.broadcast_to(values, (*indices.shape, self.hidden_size))


class _OptionalWeightedExecutor(_OptionalDecodeExecutor):
    def __init__(self, *args, weighted_enabled: bool = True) -> None:
        super().__init__(*args)
        self.weighted_enabled = weighted_enabled
        self.weighted_attempts = 0

    def try_full_resident_weighted_decode(
        self,
        hidden_states: mx.array,
        indices: mx.array,
        scores: mx.array,
    ) -> mx.array | None:
        self.weighted_attempts += 1
        rows = int(np.prod(hidden_states.shape[:-1]))
        if not self.weighted_enabled or rows != 1:
            return None
        values = indices.astype(hidden_states.dtype)[..., None]
        outputs = mx.broadcast_to(values, (*indices.shape, self.hidden_size))
        return expert_major_weighted_sum(outputs, scores, indices)


def test_qwen4_runtime_router_matches_fp32_softmax_reference() -> None:
    hidden = np.array([[[1.0, -2.0, 0.5], [-1.5, 0.25, 2.0]]], dtype=np.float32)
    weight = np.array(
        [
            [0.5, -0.25, 0.1],
            [-0.4, 0.75, 0.2],
            [0.3, 0.1, -0.6],
            [0.2, -0.5, 0.9],
        ],
        dtype=np.float32,
    )
    module = Qwen4TopKRouter(3, 4, 2, normalize_topk=True)
    module.weight = _array(weight)

    got = module(_array(hidden))
    expected = qwen4_topk_router(
        hidden,
        weight,
        top_k=2,
        normalize_topk=True,
    )
    mx.eval(got.logits, got.scores, got.indices)

    assert np.all(np.diff(np.asarray(got.scores), axis=-1) <= 0)
    assert np.allclose(np.asarray(got.logits), expected.logits, rtol=0, atol=2e-7)
    assert np.allclose(np.asarray(got.scores), expected.scores, rtol=0, atol=2e-7)
    assert np.array_equal(np.asarray(got.indices), expected.indices)


def test_qwen4_retained_router_matches_independent_subset_reference() -> None:
    hidden = np.array([[[1.0, -2.0, 0.5], [-1.5, 0.25, 2.0]]], dtype=np.float32)
    weight = np.array(
        [
            [0.5, -0.25, 0.1],
            [40.0, 40.0, 40.0],
            [0.3, 0.1, -0.6],
            [0.2, -0.5, 0.9],
        ],
        dtype=np.float32,
    )
    retained = (0, 2, 3)
    module = Qwen4TopKRouter(
        3,
        4,
        2,
        normalize_topk=True,
        retained_source_ids=retained,
    )
    module.weight = _array(weight)

    got = module(_array(hidden))
    subset = qwen4_topk_router(
        hidden,
        weight[list(retained)],
        top_k=2,
        normalize_topk=True,
    )
    expected_source_ids = np.take(np.asarray(retained), subset.indices)
    mx.eval(got.logits, got.scores, got.indices)

    assert np.allclose(np.asarray(got.logits), hidden @ weight.T, rtol=0, atol=2e-7)
    assert np.allclose(np.asarray(got.scores), subset.scores, rtol=0, atol=2e-7)
    assert np.array_equal(np.asarray(got.indices), expected_source_ids)
    assert 1 not in np.asarray(got.indices)
    assert module.retained_routing_stats() == {
        "retained_experts": 3,
        "retained_mask_calls": 1,
        "retained_mask_rows": 2,
        "source_experts": 4,
    }


@pytest.mark.parametrize(
    "retained",
    [
        (0,),
        (0, 0),
        (1, 0),
        (0, 4),
    ],
)
def test_qwen4_retained_router_refuses_invalid_source_maps(retained) -> None:
    with pytest.raises(ValueError, match="sorted, unique, in range, and fit top_k"):
        Qwen4TopKRouter(3, 4, 2, retained_source_ids=retained)


def _released_router() -> Qwen4TopKRouter:
    module = Qwen4TopKRouter(2560, 512, 10, normalize_topk=True)
    module.weight = mx.zeros((512, 2560), dtype=mx.bfloat16)
    return module


def test_qwen4_fused_exact_router_is_automatic_for_released_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fused(logits: mx.array):
        calls.append(logits)
        rows = logits.shape[0]
        return (
            mx.broadcast_to(mx.arange(10, dtype=mx.uint32), (rows, 10)),
            mx.full((rows, 10), 0.1, dtype=mx.bfloat16),
        )

    monkeypatch.setattr(moe_module, "_resolve_qwen4_fused_exact_router_symbol", lambda: fused)
    module = _released_router()
    hidden = mx.zeros((1, 1, 2560), dtype=mx.bfloat16)

    output = module(hidden)
    mx.eval(output.logits, output.scores, output.indices)

    assert len(calls) == 1
    assert calls[0].shape == (1, 512)
    assert output.logits.shape == (1, 1, 512)
    assert output.scores.shape == (1, 1, 10)
    assert output.indices.shape == (1, 1, 10)
    assert output.scores.dtype == mx.bfloat16
    assert output.indices.dtype == mx.uint32
    assert module.fused_exact_stats() == {
        "fused_exact_calls": 1,
        "fused_exact_rows": 1,
        "fused_exact_ineligible_calls": 0,
        "fused_exact_unavailable_calls": 0,
    }


def test_qwen4_fused_exact_router_receives_retained_mask_and_returns_raw_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    retained = tuple(range(10)) + tuple(range(500, 510))

    def fused(logits: mx.array):
        calls.append(logits)
        rows = logits.shape[0]
        return (
            mx.broadcast_to(mx.arange(10, dtype=mx.uint32), (rows, 10)),
            mx.full((rows, 10), 0.1, dtype=mx.bfloat16),
        )

    monkeypatch.setattr(moe_module, "_resolve_qwen4_fused_exact_router_symbol", lambda: fused)
    module = Qwen4TopKRouter(
        2560,
        512,
        10,
        normalize_topk=True,
        retained_source_ids=retained,
    )
    module.weight = mx.zeros((512, 2560), dtype=mx.bfloat16)

    output = module(mx.zeros((1, 1, 2560), dtype=mx.bfloat16))
    mx.eval(output.logits, output.scores, output.indices, *calls)
    masked = np.asarray(calls[0].astype(mx.float32))

    assert len(calls) == 1
    assert np.all(masked[:, list(retained)] == 0)
    assert np.all(np.isneginf(masked[:, 10:500]))
    assert np.all(np.asarray(output.logits.astype(mx.float32)) == 0)
    assert module.fused_exact_stats()["fused_exact_calls"] == 1
    assert module.retained_routing_stats()["retained_mask_calls"] == 1


def test_qwen4_fused_exact_router_leaves_prefill_outside_decode_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        moe_module,
        "_resolve_qwen4_fused_exact_router_symbol",
        lambda: lambda logits: pytest.fail("multirow prefill dispatched the decode-only symbol"),
    )
    module = _released_router()

    output = module(mx.zeros((1, 2, 2560), dtype=mx.bfloat16))
    mx.eval(output.logits, output.scores, output.indices)

    assert module.fused_exact_stats() == {
        "fused_exact_calls": 0,
        "fused_exact_rows": 0,
        "fused_exact_ineligible_calls": 0,
        "fused_exact_unavailable_calls": 0,
    }


@pytest.mark.parametrize("failure", ["ineligible", "unavailable"])
def test_qwen4_fused_exact_router_falls_back_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    if failure == "ineligible":
        monkeypatch.setattr(
            moe_module,
            "_resolve_qwen4_fused_exact_router_symbol",
            lambda: (
                lambda logits: pytest.fail("ineligible router path dispatched the optional symbol")
            ),
        )
        module = Qwen4TopKRouter(3, 4, 2, normalize_topk=True)
        module.weight = mx.zeros((4, 3), dtype=mx.bfloat16)
        hidden = mx.zeros((1, 1, 3), dtype=mx.bfloat16)
    else:
        monkeypatch.setattr(
            moe_module,
            "_resolve_qwen4_fused_exact_router_symbol",
            lambda: None,
        )
        module = _released_router()
        hidden = mx.zeros((1, 1, 2560), dtype=mx.bfloat16)

    output = module(hidden)
    mx.eval(output.logits, output.scores, output.indices)
    stats = module.fused_exact_stats()

    assert stats["fused_exact_calls"] == 0
    assert stats[f"fused_exact_{failure}_calls"] == 1


def test_qwen4_fused_exact_router_fails_closed_on_bad_kernel_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        moe_module,
        "_resolve_qwen4_fused_exact_router_symbol",
        lambda: lambda logits: (mx.zeros((logits.shape[0], 10), dtype=mx.int32),),
    )
    module = _released_router()

    with pytest.raises(RuntimeError, match="invalid result"):
        module(mx.zeros((1, 1, 2560), dtype=mx.bfloat16))
    assert module.fused_exact_stats()["fused_exact_calls"] == 0


def test_qwen4_model_aggregates_fused_exact_router_counters() -> None:
    first = _released_router()
    second = _released_router()
    first.fused_exact_calls = 2
    first.fused_exact_rows = 3
    second.fused_exact_ineligible_calls = 4
    shell = SimpleNamespace(
        layers=[
            SimpleNamespace(mlp=SimpleNamespace(gate=first)),
            SimpleNamespace(mlp=SimpleNamespace(gate=second)),
        ]
    )

    from moespresso.runtime.qwen4.model import Qwen4TextModelShell

    assert Qwen4TextModelShell.fused_exact_router_stats(shell) == {
        "router_layers": 2,
        "fused_exact_calls": 2,
        "fused_exact_rows": 3,
        "fused_exact_ineligible_calls": 4,
        "fused_exact_unavailable_calls": 0,
    }


def test_qwen4_expert_major_sum_is_not_router_rank_order() -> None:
    outputs = _array(np.array([[[[10_000.0], [-10_000.0], [1.0]]]], dtype=np.float16))
    scores = _array(np.ones((1, 1, 3), dtype=np.float16))
    indices = _array(np.array([[[2, 0, 1]]], dtype=np.int64))

    got = expert_major_weighted_sum(outputs, scores, indices)
    compiled = moe_module._compiled_expert_major_weighted_sum(
        outputs,
        scores,
        indices,
    )
    router_order = (
        outputs[..., 0, :] * scores[..., 0, None]
        + outputs[..., 1, :] * scores[..., 1, None]
        + outputs[..., 2, :] * scores[..., 2, None]
    )
    mx.eval(got, compiled, router_order)

    assert np.array_equal(np.asarray(got), np.array([[[0.0]]], dtype=np.float16))
    assert np.array_equal(np.asarray(compiled), np.asarray(got))
    assert np.array_equal(np.asarray(router_order), np.array([[[1.0]]], dtype=np.float16))


def test_qwen4_sparse_moe_matches_independent_reference() -> None:
    rng = np.random.default_rng(73)
    hidden = rng.normal(scale=0.5, size=(1, 3, 3)).astype(np.float32)
    router_weight = rng.normal(scale=0.3, size=(4, 3)).astype(np.float32)
    gate_up = rng.normal(scale=0.2, size=(4, 4, 3)).astype(np.float32)
    down = rng.normal(scale=0.2, size=(4, 3, 2)).astype(np.float32)
    shared_gate = rng.normal(scale=0.2, size=(2, 3)).astype(np.float32)
    shared_up = rng.normal(scale=0.2, size=(2, 3)).astype(np.float32)
    shared_down = rng.normal(scale=0.2, size=(3, 2)).astype(np.float32)
    shared_router = rng.normal(scale=0.2, size=(1, 3)).astype(np.float32)

    module = Qwen4SparseMoEBlock(3, 2, 2, 4, 2, normalize_topk=True)
    module.gate.weight = _array(router_weight)
    module.experts.gate_up_proj = _array(gate_up)
    module.experts.down_proj = _array(down)
    module.shared_expert.gate_proj.weight = _array(shared_gate)
    module.shared_expert.up_proj.weight = _array(shared_up)
    module.shared_expert.down_proj.weight = _array(shared_down)
    module.shared_expert_gate.weight = _array(shared_router)

    got = module(_array(hidden))
    expected = qwen4_sparse_moe(
        hidden,
        router_weight,
        gate_up,
        down,
        shared_gate,
        shared_up,
        shared_down,
        shared_router,
        top_k=2,
        normalize_topk=True,
    )
    mx.eval(got)

    assert np.allclose(np.asarray(got), expected.output, rtol=0, atol=3e-7)


def test_qwen4_sparse_moe_accepts_an_injected_expert_executor() -> None:
    executor = _RecordingExpertExecutor(3, 2, 4)
    module = Qwen4SparseMoEBlock(
        3,
        2,
        2,
        4,
        2,
        normalize_topk=True,
        expert_executor=executor,
    )
    module.gate.weight = _array(
        np.array(
            [
                [0.5, -0.25, 0.1],
                [-0.4, 0.75, 0.2],
                [0.3, 0.1, -0.6],
                [0.2, -0.5, 0.9],
            ],
            dtype=np.float32,
        )
    )
    module.shared_expert.gate_proj.weight = mx.zeros((2, 3))
    module.shared_expert.up_proj.weight = mx.zeros((2, 3))
    module.shared_expert.down_proj.weight = mx.zeros((3, 2))
    module.shared_expert_gate.weight = mx.zeros((1, 3))
    hidden = _array(np.array([[[1.0, -2.0, 0.5]]], dtype=np.float32))

    router = module.gate(hidden)
    expected_outputs = executor(hidden, router.indices)
    expected = expert_major_weighted_sum(
        expected_outputs,
        router.scores,
        router.indices,
    )
    executor.calls.clear()
    got = module(hidden)
    mx.eval(got, expected)

    assert module.experts is executor
    assert not hasattr(module.experts, "gate_up_proj")
    assert not hasattr(module.experts, "down_proj")
    assert len(executor.calls) == 1
    assert np.array_equal(np.asarray(executor.calls[0][1]), np.asarray(router.indices))
    assert np.array_equal(np.asarray(got), np.asarray(expected))


def test_qwen4_sparse_moe_maps_retained_source_ids_to_compact_executor_ids() -> None:
    retained = (0, 2, 3)
    executor = _RecordingExpertExecutor(3, 2, len(retained))
    module = Qwen4SparseMoEBlock(
        3,
        2,
        2,
        4,
        2,
        normalize_topk=True,
        expert_executor=executor,
        retained_source_ids=retained,
    )
    module.gate.weight = _array(
        np.array(
            [
                [0.5, -0.25, 0.1],
                [40.0, 40.0, 40.0],
                [0.3, 0.1, -0.6],
                [0.2, -0.5, 0.9],
            ],
            dtype=np.float32,
        )
    )
    module.shared_expert.gate_proj.weight = mx.zeros((2, 3))
    module.shared_expert.up_proj.weight = mx.zeros((2, 3))
    module.shared_expert.down_proj.weight = mx.zeros((3, 2))
    module.shared_expert_gate.weight = mx.zeros((1, 3))
    hidden = _array(np.array([[[1.0, -2.0, 0.5]]], dtype=np.float32))

    router = module.gate(hidden)
    lookup = np.full(4, -1, dtype=np.int64)
    lookup[list(retained)] = np.arange(len(retained))
    compact_ids = mx.array(lookup[np.asarray(router.indices)], dtype=mx.uint32)
    expected_outputs = executor(hidden, compact_ids)
    expected = expert_major_weighted_sum(
        expected_outputs,
        router.scores,
        compact_ids,
    )
    executor.calls.clear()
    got = module(hidden)
    mx.eval(got, expected)

    assert module.num_experts == 4
    assert module.physical_experts == 3
    assert len(executor.calls) == 1
    assert np.array_equal(np.asarray(executor.calls[0][1]), np.asarray(compact_ids))
    assert np.all(np.asarray(executor.calls[0][1]) < 3)
    assert np.array_equal(np.asarray(got), np.asarray(expected))


def test_qwen4_sparse_moe_uses_optional_full_resident_decode_only_for_one_row() -> None:
    executor = _OptionalDecodeExecutor(3, 2, 4)
    module = Qwen4SparseMoEBlock(
        3,
        2,
        2,
        4,
        2,
        expert_executor=executor,
    )
    module.shared_expert.gate_proj.weight = mx.zeros((2, 3))
    module.shared_expert.up_proj.weight = mx.zeros((2, 3))
    module.shared_expert.down_proj.weight = mx.zeros((3, 2))
    module.shared_expert_gate.weight = mx.zeros((1, 3))

    single = module(mx.ones((1, 1, 3)))
    multi = module(mx.ones((1, 2, 3)))
    mx.eval(single, multi)

    assert executor.decode_attempts == 2
    assert len(executor.calls) == 1
    assert executor.calls[0][0].shape == (1, 2, 3)
    assert module.compiled_routed_sum_calls == 1
    assert module.compiled_routed_sum_slot_elements == 2
    assert module.compiled_routed_sum_output_elements == 3


def test_qwen4_sparse_moe_uses_optional_weighted_decode_before_expert_rows() -> None:
    executor = _OptionalWeightedExecutor(3, 2, 4)
    module = Qwen4SparseMoEBlock(
        3,
        2,
        2,
        4,
        2,
        expert_executor=executor,
    )
    module.shared_expert.gate_proj.weight = mx.zeros((2, 3))
    module.shared_expert.up_proj.weight = mx.zeros((2, 3))
    module.shared_expert.down_proj.weight = mx.zeros((3, 2))
    module.shared_expert_gate.weight = mx.zeros((1, 3))
    hidden = mx.ones((1, 1, 3))

    router = module.gate(hidden)
    expected = executor.try_full_resident_weighted_decode(
        hidden,
        router.indices,
        router.scores,
    )
    executor.weighted_attempts = 0
    output = module(hidden)
    mx.eval(output, expected)

    assert executor.weighted_attempts == 1
    assert executor.decode_attempts == 0
    assert executor.calls == []
    assert module.weighted_decode_calls == 1
    assert module.weighted_decode_output_elements == 3
    assert module.compiled_routed_sum_calls == 0
    assert np.array_equal(np.asarray(output), np.asarray(expected))


def test_qwen4_sparse_moe_falls_back_when_optional_weighted_decode_refuses() -> None:
    executor = _OptionalWeightedExecutor(3, 2, 4, weighted_enabled=False)
    module = Qwen4SparseMoEBlock(
        3,
        2,
        2,
        4,
        2,
        expert_executor=executor,
    )

    output = module(mx.ones((1, 1, 3)))
    mx.eval(output)

    assert executor.weighted_attempts == 1
    assert executor.decode_attempts == 1
    assert executor.calls == []
    assert module.weighted_decode_calls == 0
    assert module.compiled_routed_sum_calls == 1


def test_qwen4_sparse_moe_falls_back_when_optional_decode_refuses() -> None:
    executor = _OptionalDecodeExecutor(3, 2, 4, enabled=False)
    module = Qwen4SparseMoEBlock(
        3,
        2,
        2,
        4,
        2,
        expert_executor=executor,
    )

    output = module(mx.ones((1, 1, 3)))
    mx.eval(output)

    assert executor.decode_attempts == 1
    assert len(executor.calls) == 1


def test_qwen4_sparse_moe_rejects_mismatched_expert_executor_geometry() -> None:
    executor = _RecordingExpertExecutor(4, 2, 4)

    with np.testing.assert_raises_regex(ValueError, "executor geometry"):
        Qwen4SparseMoEBlock(
            3,
            2,
            2,
            4,
            2,
            expert_executor=executor,
        )


def test_qwen4_sparse_moe_injected_executor_matches_resident_multi_token() -> None:
    rng = np.random.default_rng(379)
    hidden = rng.normal(scale=0.5, size=(1, 4, 3)).astype(np.float32)
    router_weight = rng.normal(scale=0.3, size=(4, 3)).astype(np.float32)
    gate_up = rng.normal(scale=0.2, size=(4, 4, 3)).astype(np.float32)
    down = rng.normal(scale=0.2, size=(4, 3, 2)).astype(np.float32)
    shared_gate = rng.normal(scale=0.2, size=(2, 3)).astype(np.float32)
    shared_up = rng.normal(scale=0.2, size=(2, 3)).astype(np.float32)
    shared_down = rng.normal(scale=0.2, size=(3, 2)).astype(np.float32)
    shared_router = rng.normal(scale=0.2, size=(1, 3)).astype(np.float32)

    resident = Qwen4SparseMoEBlock(3, 2, 2, 4, 2)
    injected_stack = Qwen4ExpertStack(3, 2, 4)
    injected = Qwen4SparseMoEBlock(
        3,
        2,
        2,
        4,
        2,
        expert_executor=injected_stack,
    )
    for module in (resident, injected):
        module.gate.weight = _array(router_weight)
        module.experts.gate_up_proj = _array(gate_up)
        module.experts.down_proj = _array(down)
        module.shared_expert.gate_proj.weight = _array(shared_gate)
        module.shared_expert.up_proj.weight = _array(shared_up)
        module.shared_expert.down_proj.weight = _array(shared_down)
        module.shared_expert_gate.weight = _array(shared_router)

    resident_output = resident(_array(hidden))
    injected_output = injected(_array(hidden))
    mx.eval(resident_output, injected_output)

    assert np.array_equal(np.asarray(injected_output), np.asarray(resident_output))


def test_qwen4_sparse_moe_rejects_invalid_expert_executor_output_shape() -> None:
    class _BadShapeExecutor(_RecordingExpertExecutor):
        def __call__(self, hidden_states: mx.array, indices: mx.array) -> mx.array:
            return mx.zeros((*indices.shape[:-1], self.hidden_size))

    module = Qwen4SparseMoEBlock(
        3,
        2,
        2,
        4,
        2,
        expert_executor=_BadShapeExecutor(3, 2, 4),
    )

    with np.testing.assert_raises_regex(ValueError, "invalid output shape"):
        module(mx.zeros((1, 1, 3)))


def test_qwen4_sparse_moe_rejects_invalid_expert_executor_output_dtype() -> None:
    class _BadDtypeExecutor(_RecordingExpertExecutor):
        def __call__(self, hidden_states: mx.array, indices: mx.array) -> mx.array:
            return mx.zeros((*indices.shape, self.hidden_size), dtype=mx.float16)

    module = Qwen4SparseMoEBlock(
        3,
        2,
        2,
        4,
        2,
        expert_executor=_BadDtypeExecutor(3, 2, 4),
    )

    with np.testing.assert_raises_regex(ValueError, "invalid output dtype"):
        module(mx.zeros((1, 1, 3), dtype=mx.float32))
