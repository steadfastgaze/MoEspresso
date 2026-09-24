from __future__ import annotations

import mlx.core as mx
import pytest

import moespresso.runtime.qwen4.qsa_native_selector as native_selector


def test_native_selector_eligibility_is_fixed_to_released_trusted_geometry() -> None:
    scores = mx.zeros((1, 1, 512), dtype=mx.float32)
    assert native_selector.native_qsa_selector_eligible(
        scores,
        visible_count=2_051,
        token_budget=2_048,
        compress_ratio=4,
    )
    assert not native_selector.native_qsa_selector_eligible(
        scores,
        visible_count=2_052,
        token_budget=2_048,
        compress_ratio=4,
    )
    assert not native_selector.native_qsa_selector_eligible(
        scores.astype(mx.bfloat16),
        visible_count=2_051,
        token_budget=2_048,
        compress_ratio=4,
    )
    too_wide = mx.zeros((1, 1, 4_097), dtype=mx.float32)
    assert not native_selector.native_qsa_selector_eligible(
        too_wide,
        visible_count=16_388,
        token_budget=2_048,
        compress_ratio=4,
    )


def test_native_selector_wrapper_preserves_fixed_width_and_launch_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches = []

    def fake_kernel(*, inputs, output_shapes, output_dtypes, grid, threadgroup):
        launches.append(
            {
                "inputs": inputs,
                "output_shapes": output_shapes,
                "output_dtypes": output_dtypes,
                "grid": grid,
                "threadgroup": threadgroup,
            }
        )
        return (mx.zeros(output_shapes[0], dtype=output_dtypes[0]),)

    monkeypatch.setattr(native_selector, "_kernel", lambda: fake_kernel)
    scores = mx.zeros((1, 1, 2_079), dtype=mx.float32)
    selected = native_selector.native_qsa_selected_token_indices(
        scores,
        visible_count=8_319,
        token_budget=2_048,
        compress_ratio=4,
    )
    assert selected.shape == (1, 1, 2_051)
    assert selected.dtype == mx.int32
    assert launches[0]["output_shapes"] == [(1, 1, 2_051)]
    assert launches[0]["grid"] == (512, 1, 1)
    assert launches[0]["threadgroup"] == (512, 1, 1)


def test_native_selector_wrapper_rejects_incompatible_geometry() -> None:
    with pytest.raises(ValueError, match="incompatible step"):
        native_selector.native_qsa_selected_token_indices(
            mx.zeros((1, 2, 512), dtype=mx.float32),
            visible_count=2_051,
            token_budget=2_048,
            compress_ratio=4,
        )
