from __future__ import annotations

import numpy as np

from moespresso.correctness.deepseek_v4.router_diff import (
    N_EXPERT,
    _float_stage_result,
    _topk_result,
)


def test_topk_result_compares_split_final_against_chunk_ref_row():
    got = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    ref = np.array([
        [9, 9, 9, 9, 9, 9],
        [8, 8, 8, 8, 8, 8],
        [1, 2, 3, 4, 5, 6],
    ], dtype=np.int32)

    result = _topk_result(
        got=got,
        ref=ref.reshape(-1),
        tokens=30_474,
        ref_rows=3,
        final_row=30_473,
        ref_final_row=2,
        mode="split-final",
    )

    assert result["final_ordered_match"] is True
    assert result["final_set_match"] is True
    assert result["ds4_dump_rows"] == 3
    assert result["ds4_ref_row"] == 2


def test_float_stage_result_compares_split_final_against_chunk_ref_row():
    got = np.zeros((N_EXPERT,), dtype=np.float32)
    ref = np.zeros((3, N_EXPERT), dtype=np.float32)
    got[0] = 3.0
    ref[2, 0] = 3.0

    result = _float_stage_result(
        got=got,
        ref=ref.reshape(-1),
        width=N_EXPERT,
        tokens=30_474,
        ref_rows=3,
        final_row=30_473,
        ref_final_row=2,
        mode="split-final",
    )

    assert result["all"] is None
    assert result["final"]["max_abs"] == 0.0
