from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from moespresso.runtime.deepseek_v4 import ratio4_attention_replay as replay


def test_ratio4_attention_replay_default_cases_include_required_scales():
    cases = replay.selected_replay_cases(include_splits=False)

    assert [case.name for case in cases] == [
        "local_no_compressed_pool",
        "ratio4_rows961_topk_active",
        "ratio4_rows7618_topk_active",
    ]
    assert [(case.pooled_rows, case.index_topk, case.cache) for case in cases] == [
        (0, 512, False),
        (961, 512, True),
        (7618, 512, True),
    ]


def test_ratio4_attention_replay_custom_rows_replace_defaults():
    cases = replay.selected_replay_cases(
        include_splits=True,
        pooled_rows=[123, 456],
        index_topk=999,
    )

    assert [case.name for case in cases] == [
        "ratio4_rows123_topk999",
        "ratio4_rows456_topk999",
    ]
    assert [case.pooled_rows for case in cases] == [123, 456]
    assert [case.index_topk for case in cases] == [999, 999]


def test_attention_replay_custom_rows_use_ratio_label():
    cases = replay.selected_replay_cases(
        include_splits=False,
        pooled_rows=[30],
        index_topk=512,
        ratio_label="ratio128",
    )

    assert [case.name for case in cases] == ["ratio128_rows30_topk512"]


def test_ratio4_attention_replay_sets_wrapped_index_topk():
    original = SimpleNamespace(index_topk=512)
    wrapper = SimpleNamespace(index_topk=512, _original=original)
    attn = SimpleNamespace(indexer=wrapper)

    replay._set_index_topk(attn, 2000)

    assert wrapper.index_topk == 2000
    assert original.index_topk == 2000


def test_ratio4_attention_replay_cli_writes_json(monkeypatch, tmp_path):
    out = tmp_path / "replay.json"
    calls = []

    def fake_run(package_dir, **kwargs):
        calls.append((package_dir, kwargs))
        return {
            "metric": "ds4_attention_one_layer_replay",
            "cases": [{"case": "ratio4_rows961_topk_active"}],
        }

    monkeypatch.setattr(replay, "run_ratio4_attention_replay", fake_run)

    assert replay.main([
        "/pkg",
        "--layer",
        "2",
        "--repeats",
        "3",
        "--warmup",
        "0",
        "--include-splits",
        "--json-out",
        str(out),
    ]) == 0

    payload = json.loads(out.read_text())
    assert payload["metric"] == "ds4_attention_one_layer_replay"
    assert calls[0][0].as_posix() == "/pkg"
    assert calls[0][1] == {
        "layer": 2,
        "repeats": 3,
        "warmup": 0,
        "include_splits": True,
        "pooled_rows": None,
        "index_topk": 512,
        "prime_indexer_qat": True,
        "stage_profile": False,
        "input_tokens": 1,
        "indexed_prefill_consumer": False,
        "tiled_indexer_scores": False,
    }


def test_ratio4_attention_replay_cli_rejects_invalid_repeats():
    with pytest.raises(SystemExit):
        replay.main(["/pkg", "--repeats", "0"])


def test_ratio4_attention_replay_cli_rejects_invalid_input_tokens():
    with pytest.raises(SystemExit):
        replay.main(["/pkg", "--input-tokens", "0"])


def test_ratio4_attention_replay_cli_rejects_invalid_index_topk():
    with pytest.raises(SystemExit):
        replay.main(["/pkg", "--index-topk", "0"])


def test_ratio4_attention_replay_cli_rejects_tiled_indexer_without_indexed_consumer():
    with pytest.raises(SystemExit):
        replay.main(["/pkg", "--tiled-indexer-scores"])


def test_ratio4_stage_profile_records_seconds_per_repeat():
    class _Mx:
        def eval(self, *_values):
            return None

    profile = replay._StageProfile(_Mx())

    assert profile.record("stage", lambda: "value") == "value"
    assert profile.record("stage", lambda: None) is None

    payload = profile.payload(repeats=2)
    assert payload["stage"]["calls"] == 2
    assert payload["stage"]["seconds_total"] >= 0.0
    assert payload["stage"]["seconds_per_repeat"] >= 0.0
