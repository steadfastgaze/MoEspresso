"""SSD-streaming correctness scaffolding.

These are not a streaming runtime. They pin the reference shapes that the
SSD-streaming runtime must later match: unsorted decode, sorted prefill, an opt-in real-package
resident-logit fixture, and a pure forced-cold-miss scenario.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant.tq_kernel")

import mlx.core as mx  # noqa: E402

from moespresso.runtime.owned_switchglu import OwnedSwitchGLU  # noqa: E402


def _resident_projection(n_experts, in_features, out_features, *, bits=2, seed=42):
    from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear

    mod = TurboQuantSwitchLinear(
        in_features, out_features, n_experts, bits=bits, seed=seed)
    vals_per_u32 = 32 // bits
    cols = (in_features + vals_per_u32 - 1) // vals_per_u32
    mod.packed = mx.random.randint(
        0, 2**31, (n_experts, out_features, cols)).astype(mx.uint32)
    mod.norms = (mx.random.normal((n_experts, out_features)) * 0.1).astype(mx.float16)
    mx.eval(mod.packed, mod.norms)
    return mod


def _owned_switchglu(*, n_experts=16, in_features=64, hidden_features=32,
                    gate_bits=2, up_bits=4, down_bits=2):
    from mlx_lm.models.switch_layers import SwitchGLU

    mx.random.seed(17)
    resident_shape = SwitchGLU(in_features, hidden_features, n_experts)
    gate = _resident_projection(
        n_experts, in_features, hidden_features, bits=gate_bits)
    up = _resident_projection(
        n_experts, in_features, hidden_features, bits=up_bits)
    down = _resident_projection(
        n_experts, hidden_features, in_features, bits=down_bits)
    return OwnedSwitchGLU(
        gate_proj=gate,
        up_proj=up,
        down_proj=down,
        activation=resident_shape.activation,
    )


def _manual_switchglu_reference(switch, x, indices):
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)

    x_up = switch.up_proj(x, idx, sorted_indices=do_sort)
    x_gate = switch.gate_proj(x, idx, sorted_indices=do_sort)
    out = switch.down_proj(
        switch.activation(x_up, x_gate),
        idx,
        sorted_indices=do_sort,
    )

    if do_sort:
        out = _scatter_unsort(out, inv_order, indices.shape)
    return out.squeeze(-2)


def _assert_same(a, b):
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


def test_resident_reference_shape_matches_unsorted_decode():
    switch = _owned_switchglu()
    x = mx.random.normal((1, 64)).astype(mx.float16)
    indices = mx.array([[3, 7, 1, 5]], dtype=mx.uint32)

    _assert_same(
        switch(x, indices),
        _manual_switchglu_reference(switch, x, indices),
    )


def test_resident_reference_shape_matches_sorted_prefill():
    switch = _owned_switchglu()
    x = mx.random.normal((20, 64)).astype(mx.float16)
    indices = mx.array(
        [[(t + j * 3) % 16 for j in range(4)] for t in range(20)],
        dtype=mx.uint32,
    )

    assert indices.size >= 64
    _assert_same(
        switch(x, indices),
        _manual_switchglu_reference(switch, x, indices),
    )


def _available_memory_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


def test_real_package_resident_logits_fixture_is_available_when_enabled():
    if os.environ.get("MOESPRESSO_M13_RUN_REAL_PACKAGE") != "1":
        pytest.skip("set MOESPRESSO_M13_RUN_REAL_PACKAGE=1 to run real-package fixture")

    package_env = os.environ.get("MOESPRESSO_M13_PACKAGE")
    if not package_env:
        pytest.skip("set MOESPRESSO_M13_PACKAGE to run real-package fixture")

    free = _available_memory_bytes()
    min_free_gb = float(os.environ.get("MOESPRESSO_M13_MIN_FREE_GB", "12"))
    if free is not None and free < min_free_gb * 1024**3:
        pytest.skip(
            f"only {free / 1024**3:.1f} GiB free; need {min_free_gb:.1f} GiB "
            "or lower MOESPRESSO_M13_MIN_FREE_GB deliberately")

    package = Path(package_env)
    if not package.exists():
        pytest.skip(f"real package does not exist: {package}")

    from moespresso.runtime.http import render_prompt
    from moespresso.runtime.serve import load_served_model
    from moespresso.runtime.streaming_run_lock import acquire_ssd_streaming_process_lock

    lock = acquire_ssd_streaming_process_lock()
    try:
        model, tokenizer, manifest = load_served_model(package)
        prompt_renderer = manifest.get("architecture", {}).get("prompt_renderer")
        ids = tokenizer.encode(render_prompt(
            [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            tokenizer,
            prompt_renderer=prompt_renderer,
        ))
        logits = np.array(model(mx.array([ids]))[0, -1], dtype=np.float32)
    finally:
        if lock is not None:
            lock.close()

    assert np.isfinite(logits).all()
    assert float(np.std(logits)) > 1e-3
    assert int(np.argmax(logits)) >= 0


@dataclass(frozen=True)
class ForcedColdMissCase:
    layer: int
    selected_experts: tuple[int, ...]
    initially_resident: frozenset[int]
    capacity: int

    @property
    def missing_experts(self) -> tuple[int, ...]:
        return tuple(
            e for e in self.selected_experts if e not in self.initially_resident)

    @property
    def resident_after_load(self) -> frozenset[int]:
        return self.initially_resident | frozenset(self.missing_experts)


def test_forced_cold_miss_harness_shape_is_explicit():
    case = ForcedColdMissCase(
        layer=3,
        selected_experts=(7, 11, 7, 42),
        initially_resident=frozenset({7, 42}),
        capacity=4,
    )

    assert case.missing_experts == (11,)
    assert len(case.resident_after_load) <= case.capacity
    assert 11 not in case.initially_resident
    assert 11 in case.resident_after_load
