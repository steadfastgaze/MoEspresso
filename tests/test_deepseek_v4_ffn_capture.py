from __future__ import annotations

import numpy as np

from moespresso.correctness.deepseek_v4.ffn_capture import (
    HIDDEN_SIZE,
    _read_i32_dump,
    _stage_result,
)


def test_stage_result_compares_split_final_against_chunk_ref_row():
    got = np.zeros((HIDDEN_SIZE,), dtype=np.float32)
    ref = np.zeros((3, HIDDEN_SIZE), dtype=np.float32)
    ref[2, 0] = 1.0
    got[0] = 1.0

    result = _stage_result(
        got=got,
        ref=ref.reshape(-1),
        tokens=30_474,
        ref_rows=3,
        final_row=30_473,
        ref_final_row=2,
        mode="split-final",
    )

    assert result["all"] is None
    assert result["ds4_dump_rows"] == 3
    assert result["ds4_ref_row"] == 2
    assert result["final"]["max_abs"] == 0.0


def test_read_i32_dump_uses_requested_dump_pos(tmp_path):
    prefix = tmp_path / "q3_router"
    path = tmp_path / "q3_router_ffn_moe_topk-0_pos30471.i32"
    expected = np.array([107, 85, 126, 215, 44, 129], dtype=np.int32)
    expected.tofile(path)

    got = _read_i32_dump(prefix, "ffn_moe_topk", 0, 30471)

    np.testing.assert_array_equal(got, expected)
