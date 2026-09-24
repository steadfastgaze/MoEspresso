from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import moespresso.runtime.qwen4.qsa as qsa_runtime
from moespresso.runtime.qwen4.model import Qwen4TextModelShell, _Qwen4TrustedMaskCertificate
from moespresso.runtime.qwen4.qsa import (
    Qwen4BF16QSAStateBackend,
    Qwen4QSAAdapter,
    Qwen4QSAState,
    Qwen4SparseAttention,
)
from moespresso.runtime.qwen4.primitives import Qwen4RMSNorm


def _small_qsa() -> tuple[Qwen4SparseAttention, Qwen4QSAAdapter]:
    module = Qwen4SparseAttention(
        hidden_size=2,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=2,
        index_query_heads=1,
        index_kv_heads=1,
        index_head_dim=2,
        token_budget=2,
        compress_ratio=2,
        rotary_dim=2,
        rope_base=10_000.0,
        mrope_section=(1, 0, 0),
    )
    module.indexer.index_qk_proj.weight = mx.zeros((4, 2), dtype=mx.float32)
    module.q_proj.weight = mx.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=mx.float32,
    )
    module.k_proj.weight = mx.zeros((2, 2), dtype=mx.float32)
    module.v_proj.weight = mx.eye(2, dtype=mx.float32)
    module.o_proj.weight = mx.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.5, 1.5, -2.0],
        ],
        dtype=mx.float32,
    )
    return module, Qwen4QSAAdapter(module)


class _ObservedBF16Backend(Qwen4BF16QSAStateBackend):
    def __init__(self, module: Qwen4SparseAttention) -> None:
        super().__init__(module)
        self.append_calls = 0
        self.selected_query_widths: list[int] = []
        self.selected_row_widths: list[int] = []

    def gather_selected_rows(self, state, pending_keys, pending_values, selected_indices):
        self.selected_query_widths.append(selected_indices.shape[1])
        self.selected_row_widths.append(selected_indices.shape[-1])
        return super().gather_selected_rows(
            state,
            pending_keys,
            pending_values,
            selected_indices,
        )

    def append(self, state, **kwargs):
        self.append_calls += 1
        return super().append(state, **kwargs)


class _CertifiedBF16Backend(Qwen4BF16QSAStateBackend):
    supports_trusted_mask_certificate = True

    def __init__(self, module: Qwen4SparseAttention) -> None:
        super().__init__(module)
        self.validation_certificates: list[bool | None] = []
        self.index_certificates: list[bool | None] = []

    def _validate_step_inputs_certified(self, *, capability, **kwargs) -> None:
        self.validation_certificates.append(
            capability is qsa_runtime._QSA_TRUSTED_ALL_VALID_CAPABILITY
        )
        super().validate_step_inputs(**kwargs)

    def _prepare_index_certified(self, state, *, capability, **kwargs):
        self.index_certificates.append(capability is qsa_runtime._QSA_TRUSTED_ALL_VALID_CAPABILITY)
        return super().prepare_index(state, **kwargs)


class _ResourceShapeBF16Backend(_ObservedBF16Backend):
    def gather_selected_rows(self, state, pending_keys, pending_values, selected_indices):
        self.selected_query_widths.append(selected_indices.shape[1])
        self.selected_row_widths.append(selected_indices.shape[-1])
        batch_size, query_count, _ = selected_indices.shape
        row_shape = (
            batch_size,
            query_count,
            pending_keys.shape[1],
            1,
            pending_keys.shape[-1],
        )
        return (
            mx.zeros(row_shape, dtype=pending_keys.dtype),
            mx.zeros(row_shape, dtype=pending_values.dtype),
            mx.ones((batch_size, query_count, 1), dtype=mx.bool_),
        )


class _FakeReleasedQ6Projection:
    mode = "kquant"
    kquant_type = "q6_k"

    def __init__(self, rows: int, output: mx.array) -> None:
        self.weight = mx.broadcast_to(
            mx.zeros((1, 1), dtype=mx.uint8),
            (rows, 2100),
        )
        self.scales = mx.zeros((1,), dtype=mx.uint8)
        self.output = output

    def __contains__(self, name: str) -> bool:
        return False

    def __call__(self, values: mx.array) -> mx.array:
        assert tuple(values.shape) == (1, 1, 2560)
        return self.output


class _FakeReleasedOutputProjection:
    def __init__(self) -> None:
        self.inputs: list[mx.array] = []

    def __call__(self, values: mx.array) -> mx.array:
        self.inputs.append(values)
        return mx.zeros((1, 1, 2560), dtype=values.dtype)


def _released_qsa_for_projection_test():
    index_projection = mx.zeros((1, 1, 640), dtype=mx.bfloat16)
    query_projection = mx.zeros((1, 1, 12288), dtype=mx.bfloat16)
    key_projection = mx.ones((1, 1, 512), dtype=mx.bfloat16)
    value_projection = mx.full((1, 1, 512), 3, dtype=mx.bfloat16)
    indexer = SimpleNamespace(
        index_qk_proj=_FakeReleasedQ6Projection(640, index_projection),
        q_layernorm=Qwen4RMSNorm(128, eps=1e-6),
        k_layernorm=Qwen4RMSNorm(128, eps=1e-6),
    )
    indexer.q_layernorm.weight = mx.zeros((128,), dtype=mx.bfloat16)
    indexer.k_layernorm.weight = mx.zeros((128,), dtype=mx.bfloat16)
    output = _FakeReleasedOutputProjection()
    module = SimpleNamespace(
        hidden_size=2560,
        num_query_heads=24,
        num_kv_heads=2,
        head_dim=256,
        index_query_heads=4,
        index_kv_heads=1,
        index_head_dim=128,
        token_budget=2048,
        compress_ratio=4,
        rotary_dim=64,
        rope_base=qsa_runtime.QWEN38_QSA_ROPE_BASE,
        mrope_section=qsa_runtime.QWEN38_QSA_MROPE_SECTION,
        indexer=indexer,
        q_proj=_FakeReleasedQ6Projection(12288, query_projection),
        k_proj=_FakeReleasedQ6Projection(512, key_projection),
        v_proj=_FakeReleasedQ6Projection(512, value_projection),
        o_proj=output,
        q_norm=Qwen4RMSNorm(256, eps=1e-6),
        k_norm=Qwen4RMSNorm(256, eps=1e-6),
    )
    module.q_norm.weight = mx.zeros((256,), dtype=mx.bfloat16)
    module.k_norm.weight = mx.zeros((256,), dtype=mx.bfloat16)
    return module, output


def _run_released_projection_step(
    monkeypatch: pytest.MonkeyPatch,
    *,
    operation,
):
    module, output = _released_qsa_for_projection_test()
    backend = _CertifiedBF16Backend(module)
    adapter = Qwen4QSAAdapter(module, state_backend=backend)
    hidden = mx.ones((1, 1, 2560), dtype=mx.bfloat16)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 1), dtype=mx.bool_)
    positions = mx.zeros((3, 1, 1), dtype=mx.int32)
    factors = adapter._prepare_shared_rope_factors(positions)
    issuer = object()
    scope = object()
    adapter._bind_trusted_mask_issuer(issuer, scope)
    certificate = _Qwen4TrustedMaskCertificate(
        issuer=issuer,
        scope=scope,
        allowed_mixers=(adapter,),
        source_states=(None,),
        valid_tokens=valid,
        visible_history=visible,
        current_frontier=0,
        next_frontier=1,
    )
    monkeypatch.setattr(qsa_runtime, "_qsa_project_rope_operation", lambda: operation)
    result = adapter._step_trusted_prepared(
        hidden,
        valid_tokens=valid,
        visible_history=visible,
        position_ids=positions,
        state=None,
        mask_certificate=certificate,
        shared_rope_factors=factors,
    )
    mx.eval(result.output, result.state.keys, result.state.values, result.state.raw_index_keys)
    return adapter, output, result, factors


def test_certified_qsa_step_uses_exact_native_projection_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    native = (
        mx.zeros((1, 1, 4, 128), dtype=mx.bfloat16),
        mx.full((1, 1, 128), 0.25, dtype=mx.bfloat16),
        mx.ones((1, 1, 24, 256), dtype=mx.bfloat16),
        mx.zeros((1, 1, 6144), dtype=mx.bfloat16),
        mx.full((1, 2, 1, 256), 2, dtype=mx.bfloat16),
        mx.full((1, 2, 1, 256), 3, dtype=mx.bfloat16),
    )

    def operation(*args, eps):
        calls.append((args, eps))
        return native

    adapter, output, result, factors = _run_released_projection_step(
        monkeypatch,
        operation=operation,
    )
    assert len(calls) == 1
    args, eps = calls[0]
    assert args[-3] is factors.cosine
    assert args[-2] is factors.sine
    assert args[-1] is factors.position_ids
    assert eps == 1e-6
    assert adapter.native_selector_stats()["native_project_rope_calls"] == 1
    assert adapter.shared_rope_stats() == {
        "shared_rope_factor_calls": 1,
        "shared_rope_applications": 3,
    }
    assert np.array_equal(
        np.asarray(result.state.raw_index_keys.astype(mx.float32)),
        np.full((1, 1, 128), 0.25, dtype=np.float32),
    )
    assert np.array_equal(
        np.asarray(result.state.keys.astype(mx.float32)),
        np.full((1, 2, 1, 256), 2, dtype=np.float32),
    )
    assert np.array_equal(
        np.asarray(result.state.values.astype(mx.float32)),
        np.full((1, 2, 1, 256), 3, dtype=np.float32),
    )
    mx.eval(output.inputs[0])
    assert np.array_equal(
        np.asarray(output.inputs[0].astype(mx.float32)),
        np.full((1, 1, 6144), 1.5, dtype=np.float32),
    )

    shell = SimpleNamespace(
        layers=(SimpleNamespace(mixer_kind="qsa", mixer=adapter),),
    )
    aggregate = Qwen4TextModelShell.qsa_native_selector_stats(shell)
    assert aggregate["qsa_layers"] == 1
    assert aggregate["native_project_rope_calls"] == 1


def test_certified_qsa_projection_unavailable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, result, _ = _run_released_projection_step(
        monkeypatch,
        operation=None,
    )
    assert result.frontier == 1
    stats = adapter.native_selector_stats()
    assert stats["native_project_rope_calls"] == 0
    assert stats["native_project_rope_unavailable_calls"] == 1


def test_certified_qsa_projection_ineligible_contract_preserves_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qsa_runtime, "_qsa_project_rope_contract", lambda *args: False)
    adapter, _, result, _ = _run_released_projection_step(
        monkeypatch,
        operation=lambda *args, **kwargs: pytest.fail("ineligible native call"),
    )
    assert result.frontier == 1
    stats = adapter.native_selector_stats()
    assert stats["native_project_rope_calls"] == 0
    assert stats["native_project_rope_ineligible_calls"] == 1


def test_qsa_adapter_uses_released_per_head_gate_split_before_output_projection() -> None:
    module, adapter = _small_qsa()
    hidden = mx.array([[[1.0, 2.0]]], dtype=mx.float32)

    result = adapter(
        hidden,
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
        visible_history=mx.ones((1, 1), dtype=mx.bool_),
        position_ids=mx.zeros((3, 1, 1), dtype=mx.int32),
        state=None,
    )
    mx.eval(result.output, result.state.keys, result.state.values)

    gates = np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float32)
    attention = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    gated = attention / (1.0 + np.exp(-gates))
    expected = gated @ np.asarray(module.o_proj.weight).T

    assert np.allclose(np.asarray(result.output)[0, 0], expected, rtol=0, atol=1e-6)
    assert result.frontier == 1
    assert result.state.offset == 1
    assert result.state.keys.shape == (1, 1, 1, 2)
    assert result.state.values.shape == (1, 1, 1, 2)
    assert result.state.raw_index_keys.shape == (1, 1, 2)
    assert result.state.position_ids.shape == (3, 1, 1)


def test_certified_qsa_step_binds_masks_frontiers_and_backend_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _small_qsa()
    backend = _CertifiedBF16Backend(module)
    adapter = Qwen4QSAAdapter(module, state_backend=backend)
    hidden = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 1), dtype=mx.bool_)
    positions = mx.zeros((3, 1, 1), dtype=mx.int32)
    issuer = object()
    scope = object()
    adapter._bind_trusted_mask_issuer(issuer, scope)
    certificate = _Qwen4TrustedMaskCertificate(
        issuer=issuer,
        scope=scope,
        allowed_mixers=(adapter,),
        source_states=(None,),
        valid_tokens=valid,
        visible_history=visible,
        current_frontier=0,
        next_frontier=1,
    )

    def unexpected_array_equal(*args, **kwargs):
        raise AssertionError("certified step evaluated the derived mask suffix")

    monkeypatch.setattr(qsa_runtime.mx, "array_equal", unexpected_array_equal)
    result = adapter._step_trusted_certified(
        hidden,
        valid_tokens=valid,
        visible_history=visible,
        position_ids=positions,
        state=None,
        mask_certificate=certificate,
    )
    mx.eval(result.output)

    assert backend.validation_certificates == [True]
    assert backend.index_certificates == [True]
    assert adapter.trusted_mask_stats() == {
        "trusted_mask_certificate_calls": 1,
        "trusted_all_valid_certificate_calls": 1,
    }
    adapter._clear_trusted_mask_scope(issuer, scope)
    with pytest.raises(ValueError, match="unissued or stale"):
        adapter._step_trusted_certified(
            hidden,
            valid_tokens=valid,
            visible_history=visible,
            position_ids=positions,
            state=None,
            mask_certificate=certificate,
        )


def test_certified_qsa_step_uses_native_selector_when_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _small_qsa()
    hidden = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 1), dtype=mx.bool_)
    positions = mx.zeros((3, 1, 1), dtype=mx.int32)

    monkeypatch.setattr(qsa_runtime, "native_qsa_selector_eligible", lambda *args, **kwargs: True)
    native_calls = []

    def native_selection(*args, **kwargs):
        native_calls.append(kwargs)
        return mx.array([[[0, -1, -1]]], dtype=mx.int32)

    monkeypatch.setattr(qsa_runtime, "native_qsa_selected_token_indices", native_selection)

    def run():
        backend = _CertifiedBF16Backend(module)
        adapter = Qwen4QSAAdapter(module, state_backend=backend)
        issuer = object()
        scope = object()
        adapter._bind_trusted_mask_issuer(issuer, scope)
        certificate = _Qwen4TrustedMaskCertificate(
            issuer=issuer,
            scope=scope,
            allowed_mixers=(adapter,),
            source_states=(None,),
            valid_tokens=valid,
            visible_history=visible,
            current_frontier=0,
            next_frontier=1,
        )
        result = adapter._step_trusted_certified(
            hidden,
            valid_tokens=valid,
            visible_history=visible,
            position_ids=positions,
            state=None,
            mask_certificate=certificate,
        )
        mx.eval(result.output)
        return adapter.native_selector_stats()

    enabled_stats = run()
    assert len(native_calls) == 1
    assert enabled_stats == {
        "native_selector_calls": 1,
        "native_selector_groups": 0,
        "native_selector_ineligible_calls": 0,
        "native_select_gather_calls": 0,
        "native_select_gather_groups": 0,
        "native_select_gather_ineligible_calls": 1,
        "native_select_gather_unavailable_calls": 0,
        "native_project_rope_calls": 0,
        "native_project_rope_ineligible_calls": 0,
        "native_project_rope_unavailable_calls": 0,
    }


def test_certified_qsa_step_combines_native_selection_and_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _small_qsa()
    hidden = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 1), dtype=mx.bool_)
    positions = mx.zeros((3, 1, 1), dtype=mx.int32)
    monkeypatch.setattr(qsa_runtime, "native_qsa_selector_eligible", lambda *args, **kwargs: True)
    native_calls = []

    def native_selection(*args, **kwargs):
        native_calls.append(kwargs)
        return mx.array([[[0, -1, -1]]], dtype=mx.int32)

    monkeypatch.setattr(qsa_runtime, "native_qsa_selected_token_indices", native_selection)

    class CompositeBackend(_CertifiedBF16Backend):
        def __init__(self, module):
            super().__init__(module)
            self.composite_calls = 0

        def _select_and_gather_rows_deferred_finite(
            self,
            state,
            pending_keys,
            pending_values,
            scores,
            *,
            visible_count,
            capability,
        ):
            assert capability is qsa_runtime._QSA_DEFER_PENDING_FINITE_CAPABILITY
            assert visible_count == 1
            self.composite_calls += 1
            selected = mx.array([[[0, -1, -1]]], dtype=mx.int32)
            selected_valid = selected >= 0
            keys = mx.pad(pending_keys[:, None], [(0, 0), (0, 0), (0, 0), (0, 2), (0, 0)])
            values = mx.pad(
                pending_values[:, None],
                [(0, 0), (0, 0), (0, 0), (0, 2), (0, 0)],
            )
            return selected, selected_valid, keys, values

    def run():
        backend = CompositeBackend(module)
        adapter = Qwen4QSAAdapter(module, state_backend=backend)
        issuer = object()
        scope = object()
        adapter._bind_trusted_mask_issuer(issuer, scope)
        certificate = _Qwen4TrustedMaskCertificate(
            issuer=issuer,
            scope=scope,
            allowed_mixers=(adapter,),
            source_states=(None,),
            valid_tokens=valid,
            visible_history=visible,
            current_frontier=0,
            next_frontier=1,
        )
        result = adapter._step_trusted_certified(
            hidden,
            valid_tokens=valid,
            visible_history=visible,
            position_ids=positions,
            state=None,
            mask_certificate=certificate,
        )
        mx.eval(result.output)
        return backend, adapter.native_selector_stats()

    enabled_backend, enabled_stats = run()
    assert enabled_backend.composite_calls == 1
    assert native_calls == []
    assert enabled_stats["native_selector_calls"] == 1
    assert enabled_stats["native_select_gather_calls"] == 1


def test_certified_qsa_step_rejects_rebound_masks_and_frontiers() -> None:
    module, adapter = _small_qsa()
    hidden = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    visible = mx.ones((1, 1), dtype=mx.bool_)
    positions = mx.zeros((3, 1, 1), dtype=mx.int32)
    issuer = object()
    scope = object()
    adapter._bind_trusted_mask_issuer(issuer, scope)

    for certificate in (
        _Qwen4TrustedMaskCertificate(
            issuer=issuer,
            scope=scope,
            allowed_mixers=(adapter,),
            source_states=(None,),
            valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
            visible_history=visible,
            current_frontier=0,
            next_frontier=1,
        ),
        _Qwen4TrustedMaskCertificate(
            issuer=issuer,
            scope=scope,
            allowed_mixers=(adapter,),
            source_states=(None,),
            valid_tokens=valid,
            visible_history=visible,
            current_frontier=1,
            next_frontier=2,
        ),
        _Qwen4TrustedMaskCertificate(
            issuer=issuer,
            scope=scope,
            allowed_mixers=(adapter,),
            source_states=(object(),),
            valid_tokens=valid,
            visible_history=visible,
            current_frontier=0,
            next_frontier=1,
        ),
    ):
        with pytest.raises(ValueError, match="unissued or stale"):
            adapter._step_trusted_certified(
                hidden,
                valid_tokens=valid,
                visible_history=visible,
                position_ids=positions,
                state=None,
                mask_certificate=certificate,
            )


def test_qsa_trusted_mask_scope_rejects_overlap_and_foreign_shells() -> None:
    _, adapter = _small_qsa()
    issuer = object()
    first_scope = object()
    adapter._bind_trusted_mask_issuer(issuer, first_scope)

    with pytest.raises(ValueError, match="active trusted mask scope"):
        adapter._bind_trusted_mask_issuer(issuer, object())
    with pytest.raises(ValueError, match="another model shell"):
        adapter._bind_trusted_mask_issuer(object(), object())

    adapter._clear_trusted_mask_scope(issuer, first_scope)
    second_scope = object()
    adapter._bind_trusted_mask_issuer(issuer, second_scope)
    adapter._clear_trusted_mask_scope(issuer, second_scope)
    with pytest.raises(ValueError, match="another model shell"):
        adapter._bind_trusted_mask_issuer(object(), object())


def test_qsa_short_context_capability_does_not_change_bf16_selection() -> None:
    module, _ = _small_qsa()
    backend = _ObservedBF16Backend(module)
    adapter = Qwen4QSAAdapter(module, state_backend=backend)

    result = adapter(
        mx.array([[[1.0, 2.0]]], dtype=mx.float32),
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
        visible_history=mx.ones((1, 1), dtype=mx.bool_),
        position_ids=mx.zeros((3, 1, 1), dtype=mx.int32),
        state=None,
    )
    mx.eval(result.output)

    assert backend.selected_query_widths == [1]
    assert backend.selected_row_widths == [3]


def test_qsa_adapter_gate_epilogue_stays_on_the_released_bfloat16_lattice() -> None:
    module, adapter = _small_qsa()
    module.indexer.index_qk_proj.weight = module.indexer.index_qk_proj.weight.astype(mx.bfloat16)
    module.q_proj.weight = mx.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-4.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=mx.bfloat16,
    )
    module.k_proj.weight = module.k_proj.weight.astype(mx.bfloat16)
    module.v_proj.weight = mx.array(
        [[0.0, 0.0], [0.0625, 0.0]],
        dtype=mx.bfloat16,
    )
    module.o_proj.weight = mx.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=mx.bfloat16,
    )
    hidden = mx.array([[[1.0, 0.0]]], dtype=mx.bfloat16)

    result = adapter(
        hidden,
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
        visible_history=mx.ones((1, 1), dtype=mx.bool_),
        position_ids=mx.zeros((3, 1, 1), dtype=mx.int32),
        state=None,
    )
    attention = mx.array(0.0625, dtype=mx.bfloat16)
    gates = mx.array(-4.0, dtype=mx.bfloat16)
    released = attention * mx.sigmoid(gates)
    widened = (attention.astype(mx.float32) * mx.sigmoid(gates.astype(mx.float32))).astype(
        mx.bfloat16
    )
    mx.eval(result.output, released, widened)

    got = np.asarray(result.output.astype(mx.float32))[0, 0, 0]
    expected = np.asarray(released.astype(mx.float32)).item()
    old_path = np.asarray(widened.astype(mx.float32)).item()
    assert got == expected
    assert expected != old_path


def test_qsa_adapter_keeps_released_key_only_padding_semantics() -> None:
    module, adapter = _small_qsa()
    hidden = mx.array([[[1.0, 2.0], [3.0, -1.0]]], dtype=mx.float32)
    positions = mx.zeros((3, 1, 2), dtype=mx.int32)

    prefix = adapter(
        hidden[:, :1],
        valid_tokens=mx.array([[True]], dtype=mx.bool_),
        visible_history=mx.array([[True]], dtype=mx.bool_),
        position_ids=positions[:, :, :1],
        state=None,
    )
    result = adapter(
        hidden[:, 1:],
        valid_tokens=mx.array([[False]], dtype=mx.bool_),
        visible_history=mx.array([[True, False]], dtype=mx.bool_),
        position_ids=positions[:, :, 1:],
        state=prefix.state,
        prior_position_history=positions[:, :, :1],
    )
    mx.eval(result.output)

    gates = np.array([3.0, -1.0, -3.0, 1.0], dtype=np.float32)
    attention = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    expected = (attention / (1.0 + np.exp(-gates))) @ np.asarray(module.o_proj.weight).T
    assert np.allclose(np.asarray(result.output)[0, 0], expected, rtol=0, atol=1e-6)
    assert np.any(np.asarray(result.output)[0, 0] != 0)
    assert result.state.offset == 2


def test_qsa_adapter_wires_index_rope_selection_into_sparse_gqa() -> None:
    module = Qwen4SparseAttention(
        hidden_size=4,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=2,
        index_query_heads=1,
        index_kv_heads=1,
        index_head_dim=2,
        token_budget=2,
        compress_ratio=2,
        rotary_dim=2,
        rope_base=10_000.0,
        mrope_section=(1, 0, 0),
    )
    module.indexer.index_qk_proj.weight = mx.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=mx.bfloat16,
    )
    module.q_proj.weight = mx.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=mx.bfloat16,
    )
    module.k_proj.weight = mx.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=mx.bfloat16,
    )
    module.v_proj.weight = mx.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=mx.bfloat16,
    )
    module.o_proj.weight = mx.eye(4, dtype=mx.bfloat16)
    adapter = Qwen4QSAAdapter(module)
    hidden = mx.array(
        [
            [
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.87758255, 0.47942555, 0.0, 1.0],
                [0.87758255, 0.47942555, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=mx.bfloat16,
    )
    positions = mx.array(
        [
            [[0, 0, 77, 1, 1, 1]],
            [[10, 10, 177, 20, 20, 20]],
            [[30, 30, 277, 40, 40, 40]],
        ],
        dtype=mx.int32,
    )
    visible = mx.array([[True, True, False, True, True, True]], dtype=mx.bool_)

    result = adapter(
        hidden,
        valid_tokens=visible,
        visible_history=visible,
        position_ids=positions,
        state=None,
    )
    without_semantic_rope = adapter(
        hidden,
        valid_tokens=visible,
        visible_history=visible,
        position_ids=mx.zeros((3, 1, 6), dtype=mx.int32),
        state=None,
    )
    mx.eval(result.output, without_semantic_rope.output, result.state.raw_index_keys)

    output = np.asarray(result.output.astype(mx.float32))
    output_without_semantic_rope = np.asarray(without_semantic_rope.output.astype(mx.float32))
    assert np.allclose(
        output[0, -1],
        np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    assert not np.allclose(
        output_without_semantic_rope[0, -1],
        output[0, -1],
        rtol=0,
        atol=1e-6,
    )
    assert np.array_equal(
        np.asarray(result.state.raw_index_keys.astype(mx.float32))[0, [0, 1, 5]],
        np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (3, 1)),
    )


def test_qsa_adapter_chunk_matches_tokenwise_with_padding_holes_and_semantic_positions() -> None:
    _, adapter = _small_qsa()
    hidden = mx.array(
        [
            [[0.5, -1.0], [1.0, 0.25], [-0.5, 2.0], [3.0, -2.0], [0.75, 1.5]],
            [[-2.0, 1.0], [0.25, 0.5], [1.5, -0.75], [-1.0, -1.5], [2.0, 0.25]],
        ],
        dtype=mx.float32,
    )
    visible = mx.array(
        [
            [False, True, True, False, True],
            [True, False, True, True, True],
        ],
        dtype=mx.bool_,
    )
    semantic = mx.array(
        [
            [[40, 7, 8, 41, 9], [3, 50, 4, 5, 6]],
            [[140, 17, 18, 141, 19], [13, 150, 14, 15, 16]],
            [[240, 27, 28, 241, 29], [23, 250, 24, 25, 26]],
        ],
        dtype=mx.int32,
    )

    chunk = adapter(
        hidden,
        valid_tokens=visible,
        visible_history=visible,
        position_ids=semantic,
        state=None,
    )

    token_outputs = []
    state = None
    for index in range(hidden.shape[1]):
        token = adapter(
            hidden[:, index : index + 1],
            valid_tokens=visible[:, index : index + 1],
            visible_history=visible[:, : index + 1],
            position_ids=semantic[:, :, index : index + 1],
            state=state,
            prior_position_history=semantic[:, :, :index],
        )
        token_outputs.append(token.output)
        state = token.state
    assert state is not None
    tokenwise = mx.concatenate(token_outputs, axis=1)
    mx.eval(
        chunk.output,
        tokenwise,
        chunk.state.keys,
        state.keys,
        chunk.state.values,
        state.values,
        chunk.state.raw_index_keys,
        state.raw_index_keys,
        chunk.state.position_ids,
        state.position_ids,
    )

    assert np.allclose(np.asarray(chunk.output), np.asarray(tokenwise), rtol=0, atol=2e-6)
    assert np.allclose(np.asarray(chunk.state.keys), np.asarray(state.keys), rtol=0, atol=1e-6)
    assert np.array_equal(np.asarray(chunk.state.values), np.asarray(state.values))
    assert np.array_equal(
        np.asarray(chunk.state.raw_index_keys),
        np.asarray(state.raw_index_keys),
    )
    assert np.array_equal(np.asarray(chunk.state.position_ids), np.asarray(semantic))
    assert np.array_equal(np.asarray(state.position_ids), np.asarray(semantic))
    assert chunk.state.offset == state.offset == hidden.shape[1]


def test_qsa_query_tiling_preserves_multibatch_padding_selection_output_and_state(
    monkeypatch,
) -> None:
    module, untiled = _small_qsa()
    tiled_backend = _ObservedBF16Backend(module)
    tiled = Qwen4QSAAdapter(
        module,
        state_backend=tiled_backend,
        max_query_tokens=2,
    )
    hidden = mx.array(
        [
            [[0.5, -1.0], [1.0, 0.25], [-0.5, 2.0], [3.0, -2.0], [0.75, 1.5]],
            [[-2.0, 1.0], [0.25, 0.5], [1.5, -0.75], [-1.0, -1.5], [2.0, 0.25]],
        ],
        dtype=mx.float32,
    )
    visible = mx.array(
        [[False, True, True, False, True], [True, False, True, True, True]],
        dtype=mx.bool_,
    )
    positions = mx.array(
        [
            [[40, 7, 8, 41, 9], [3, 50, 4, 5, 6]],
            [[140, 17, 18, 141, 19], [13, 150, 14, 15, 16]],
            [[240, 27, 28, 241, 29], [23, 250, 24, 25, 26]],
        ],
        dtype=mx.int32,
    )
    original_select = qsa_runtime.qsa_selected_token_indices
    captured: list[mx.array] = []

    def capture_selection(*args, **kwargs):
        selected = original_select(*args, **kwargs)
        captured.append(selected)
        return selected

    monkeypatch.setattr(qsa_runtime, "qsa_selected_token_indices", capture_selection)
    untiled_result = untiled(
        hidden,
        valid_tokens=visible,
        visible_history=visible,
        position_ids=positions,
        state=None,
    )
    mx.eval(untiled_result.output, *captured)
    untiled_selected = np.asarray(captured[0])
    captured.clear()

    tiled_result = tiled(
        hidden,
        valid_tokens=visible,
        visible_history=visible,
        position_ids=positions,
        state=None,
    )
    mx.eval(
        tiled_result.output,
        tiled_result.state.keys,
        tiled_result.state.values,
        tiled_result.state.raw_index_keys,
        *captured,
    )
    tiled_selected = np.concatenate([np.asarray(value) for value in captured], axis=1)

    assert np.array_equal(tiled_selected, untiled_selected)
    untiled_output = np.asarray(untiled_result.output)
    tiled_output = np.asarray(tiled_result.output)
    output_delta = tiled_output.astype(np.float64) - untiled_output.astype(np.float64)
    output_rms = float(np.sqrt(np.mean(np.square(output_delta))))
    output_mre = float(
        np.mean(
            np.abs(output_delta) / np.maximum(np.abs(untiled_output), np.finfo(np.float32).tiny)
        )
    )
    reference_rms = float(np.sqrt(np.mean(np.square(untiled_output.astype(np.float64)))))
    assert np.array_equal(tiled_output, untiled_output)
    assert float(np.max(np.abs(output_delta))) == 0.0
    assert output_rms == 0.0
    assert output_mre == 0.0
    assert output_rms / reference_rms == 0.0
    assert np.array_equal(
        np.asarray(tiled_result.state.keys),
        np.asarray(untiled_result.state.keys),
    )
    assert np.array_equal(
        np.asarray(tiled_result.state.values),
        np.asarray(untiled_result.state.values),
    )
    assert np.array_equal(
        np.asarray(tiled_result.state.raw_index_keys),
        np.asarray(untiled_result.state.raw_index_keys),
    )
    assert tiled_backend.selected_query_widths == [2, 2, 1]
    assert tiled_backend.append_calls == 1


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_qsa_query_tile_policy_rejects_invalid_width(value) -> None:
    module, _ = _small_qsa()
    with pytest.raises(ValueError, match="max_query_tokens"):
        Qwen4QSAAdapter(module, max_query_tokens=value)


def test_qsa_8192_query_resource_shape_never_builds_an_untiled_selected_workspace(
    monkeypatch,
) -> None:
    token_count = 8_192
    tile_width = 64
    module = Qwen4SparseAttention(
        hidden_size=2,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=2,
        index_query_heads=1,
        index_kv_heads=1,
        index_head_dim=2,
        token_budget=2_048,
        compress_ratio=4,
        rotary_dim=2,
        rope_base=10_000.0,
        mrope_section=(1, 0, 0),
    )
    for projection in (
        module.indexer.index_qk_proj,
        module.q_proj,
        module.k_proj,
        module.v_proj,
        module.o_proj,
    ):
        projection.weight = mx.zeros_like(projection.weight)
    backend = _ResourceShapeBF16Backend(module)
    adapter = Qwen4QSAAdapter(
        module,
        state_backend=backend,
        max_query_tokens=tile_width,
    )
    score_widths: list[int] = []
    compressed_calls = 0
    original_scores = qsa_runtime.qsa_index_scores
    original_compress = qsa_runtime.qsa_compress_index_keys

    def capture_scores(index_queries, compressed_keys):
        score_widths.append(index_queries.shape[1])
        return original_scores(index_queries, compressed_keys)

    def fixed_width_selection(scores, _layout, *, token_budget, compress_ratio):
        return mx.zeros(
            (scores.shape[0], scores.shape[1], token_budget + compress_ratio - 1),
            dtype=mx.int32,
        )

    def capture_compress(*args, **kwargs):
        nonlocal compressed_calls
        compressed_calls += 1
        return original_compress(*args, **kwargs)

    monkeypatch.setattr(qsa_runtime, "qsa_index_scores", capture_scores)
    monkeypatch.setattr(qsa_runtime, "qsa_selected_token_indices", fixed_width_selection)
    monkeypatch.setattr(qsa_runtime, "qsa_compress_index_keys", capture_compress)
    hidden = mx.zeros((1, token_count, 2), dtype=mx.float32)
    positions = mx.broadcast_to(
        mx.arange(token_count, dtype=mx.int32)[None, None],
        (3, 1, token_count),
    )
    result = adapter(
        hidden,
        valid_tokens=mx.ones((1, token_count), dtype=mx.bool_),
        visible_history=mx.ones((1, token_count), dtype=mx.bool_),
        position_ids=positions,
        state=None,
    )
    mx.eval(result.output, result.state.keys, result.state.raw_index_keys)

    expected_tiles = token_count // tile_width
    assert result.output.shape == hidden.shape
    assert len(score_widths) == expected_tiles
    assert max(score_widths) == tile_width
    assert backend.selected_query_widths == [tile_width] * expected_tiles
    assert backend.append_calls == 1
    assert compressed_calls == 1


@pytest.mark.parametrize(
    "partitions",
    [(5,), (2, 3), (1, 1, 1, 1, 1)],
    ids=("full", "chunked", "tokenwise"),
)
def test_qsa_trusted_steps_do_not_host_materialize_position_history(
    monkeypatch,
    partitions: tuple[int, ...],
) -> None:
    _, adapter = _small_qsa()
    hidden = mx.array(
        [[[0.5, -1.0], [1.0, 0.25], [-0.5, 2.0], [3.0, -2.0], [0.75, 1.5]]],
        dtype=mx.float32,
    )
    visible = mx.array([[False, True, True, False, True]], dtype=mx.bool_)
    positions = mx.array(
        [[[40, 7, 8, 41, 9]], [[140, 17, 18, 141, 19]], [[240, 27, 28, 241, 29]]],
        dtype=mx.int32,
    )
    real_eval = mx.eval
    real_asarray = np.asarray

    def is_position_history(value) -> bool:
        shape = getattr(value, "shape", ())
        dtype = getattr(value, "dtype", None)
        return (
            len(shape) == 3
            and shape[0] == 3
            and dtype in (mx.int32, mx.int64, mx.uint32, mx.uint64)
        )

    def guarded_eval(*values) -> None:
        if any(is_position_history(value) for value in values):
            raise AssertionError("trusted validation evaluated semantic-position history")
        real_eval(*values)

    def guarded_asarray(value, *args, **kwargs):
        if is_position_history(value):
            raise AssertionError("trusted validation copied semantic-position history to host")
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(qsa_runtime.mx, "eval", guarded_eval)
    monkeypatch.setattr(qsa_runtime.np, "asarray", guarded_asarray)

    cursor = 0
    state = None
    for width in partitions:
        result = adapter.step_trusted(
            hidden[:, cursor : cursor + width],
            valid_tokens=visible[:, cursor : cursor + width],
            visible_history=visible[:, : cursor + width],
            position_ids=positions[:, :, cursor : cursor + width],
            state=state,
        )
        real_eval(result.output)
        cursor += width
        state = result.state

    assert cursor == hidden.shape[1]
    assert state is not None
    assert state.offset == hidden.shape[1]


def test_qsa_state_fork_snapshot_and_validation_are_schema_bound() -> None:
    _, adapter = _small_qsa()
    hidden = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
    positions = mx.array(
        [[[7, 8]], [[17, 18]], [[27, 28]]],
        dtype=mx.int32,
    )
    result = adapter(
        hidden,
        valid_tokens=mx.ones((1, 2), dtype=mx.bool_),
        visible_history=mx.ones((1, 2), dtype=mx.bool_),
        position_ids=positions,
        state=None,
    )
    forked = adapter.fork_state(result.state)
    snapshot = adapter.snapshot_state(result.state)
    assert forked is not None
    assert snapshot is not None
    mx.eval(result.state.keys, forked.keys, snapshot.keys)

    forked.keys[0, 0, 0, 0] = 99.0
    mx.eval(result.state.keys, forked.keys, snapshot.keys)
    assert np.asarray(result.state.keys)[0, 0, 0, 0] != 99.0
    assert np.asarray(snapshot.keys)[0, 0, 0, 0] != 99.0

    adapter.validate_state(
        snapshot,
        expected_frontier=2,
        position_history=positions,
    )
    with pytest.raises(ValueError, match="schema"):
        adapter.validate_state(
            replace(snapshot, schema="unsupported"),
            expected_frontier=2,
            position_history=positions,
        )
    with pytest.raises(ValueError, match="frontier"):
        adapter.validate_state(
            snapshot,
            expected_frontier=1,
            position_history=positions[:, :, :1],
        )
    corrupted_positions = positions + mx.array(
        [[[0, 0]], [[0, 1]], [[0, 0]]],
        dtype=mx.int32,
    )
    with pytest.raises(ValueError, match="do not match"):
        adapter.validate_state_strict(
            snapshot,
            expected_frontier=2,
            position_history=corrupted_positions,
        )
    with pytest.raises(ValueError, match="cached tensors"):
        adapter.validate_state(
            replace(
                snapshot,
                keys=snapshot.keys.astype(mx.float16),
                values=snapshot.values.astype(mx.float16),
                raw_index_keys=snapshot.raw_index_keys.astype(mx.float16),
            ),
            expected_frontier=2,
            position_history=positions,
        )
    with pytest.raises(ValueError, match="offset"):
        adapter.validate_state(
            replace(snapshot, offset=True),
            expected_frontier=2,
            position_history=positions,
        )


def test_qsa_adapter_rejects_validity_suffix_drift() -> None:
    _, adapter = _small_qsa()
    with pytest.raises(ValueError, match="current validity mask"):
        adapter(
            mx.ones((1, 1, 2), dtype=mx.float32),
            valid_tokens=mx.array([[True]], dtype=mx.bool_),
            visible_history=mx.array([[False]], dtype=mx.bool_),
            position_ids=mx.zeros((3, 1, 1), dtype=mx.int32),
            state=None,
        )


def test_qsa_external_step_requires_independent_matching_position_history() -> None:
    _, adapter = _small_qsa()
    positions = mx.array([[[7, 8]], [[17, 18]], [[27, 28]]], dtype=mx.int32)
    prefix = adapter(
        mx.ones((1, 1, 2), dtype=mx.float32),
        valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
        visible_history=mx.ones((1, 1), dtype=mx.bool_),
        position_ids=positions[:, :, :1],
        state=None,
    )

    with pytest.raises(ValueError, match="requires prior_position_history"):
        adapter(
            mx.ones((1, 1, 2), dtype=mx.float32),
            valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
            visible_history=mx.ones((1, 2), dtype=mx.bool_),
            position_ids=positions[:, :, 1:],
            state=prefix.state,
        )

    corrupted_prior = positions[:, :, :1] + mx.array(
        [[[0]], [[1]], [[0]]],
        dtype=mx.int32,
    )
    with pytest.raises(ValueError, match="do not match"):
        adapter(
            mx.ones((1, 1, 2), dtype=mx.float32),
            valid_tokens=mx.ones((1, 1), dtype=mx.bool_),
            visible_history=mx.ones((1, 2), dtype=mx.bool_),
            position_ids=positions[:, :, 1:],
            state=prefix.state,
            prior_position_history=corrupted_prior,
        )


def test_qsa_geometry_rejects_unsupported_head_groupings() -> None:
    with pytest.raises(ValueError, match="equal groups"):
        Qwen4SparseAttention(num_query_heads=3, num_kv_heads=2)
    with pytest.raises(ValueError, match="one index KV head"):
        Qwen4SparseAttention(index_kv_heads=2)
    with pytest.raises(ValueError, match="divisible"):
        Qwen4SparseAttention(token_budget=3, compress_ratio=2)


def test_qsa_state_is_frozen() -> None:
    state = Qwen4QSAState(
        keys=mx.zeros((1, 1, 1, 2)),
        values=mx.zeros((1, 1, 1, 2)),
        raw_index_keys=mx.zeros((1, 1, 2)),
        position_ids=mx.zeros((3, 1, 1), dtype=mx.int32),
        offset=1,
    )
    with pytest.raises(AttributeError):
        state.offset = 2
