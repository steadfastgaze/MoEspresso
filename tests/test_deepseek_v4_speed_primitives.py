from __future__ import annotations

import json

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.dsv4.mlx_model")

from moespresso.runtime.deepseek_v4.speed_primitives import (  # noqa: E402
    build_ds4_primitive_summary,
    main,
)


def _stage(payload, name):
    return next(row for row in payload["stages"] if row["stage"] == name)


def test_ds4_primitive_summary_marks_row_gather_rewrite_as_not_a_boundary_cut():
    payload = build_ds4_primitive_summary(pooled_rows=640, topk_rows=512)

    current = _stage(payload, "decode_selected_rows_current")
    direct = _stage(payload, "decode_selected_rows_direct_take")

    assert payload["metric"] == "ds4_synthetic_mlx_primitive_counts"
    assert current["primitive_count"] == 5
    assert direct["primitive_count"] == 5
    assert current["primitive_counts"]["GatherAxis"] == 1
    assert direct["primitive_counts"]["Gather"] == 1


def test_ds4_primitive_summary_exposes_attention_and_indexer_stage_shape():
    payload = build_ds4_primitive_summary(pooled_rows=640, topk_rows=512)

    attention = _stage(payload, "ratio4_attention_consumer")
    indexer = _stage(payload, "indexer_decode_cached_pool_score")
    masked = _stage(payload, "indexer_prefill_masked_score")

    assert attention["primitive_counts"]["Matmul"] == 2
    assert attention["primitive_counts"]["Softmax"] == 1
    assert attention["primitive_counts"]["Concatenate"] >= 1
    assert indexer["primitive_counts"]["ArgPartition"] == 1
    assert indexer["primitive_counts"]["Hadamard"] == 1
    assert indexer["primitive_counts"]["Matmul"] == 1
    assert indexer["primitive_count"] < masked["primitive_count"]
    assert masked["primitive_counts"]["Select"] > indexer["primitive_counts"]["Select"]


def test_ds4_speed_primitives_cli_writes_json(tmp_path):
    out = tmp_path / "primitives.json"

    assert main(["--pooled-rows", "640", "--json-out", str(out)]) == 0

    payload = json.loads(out.read_text())
    assert payload["shape"]["pooled_rows"] == 640
    assert {row["stage"] for row in payload["stages"]} == {
        "decode_selected_rows_current",
        "decode_selected_rows_direct_take",
        "indexer_decode_cached_pool_score",
        "indexer_prefill_masked_score",
        "ratio4_attention_consumer",
    }
