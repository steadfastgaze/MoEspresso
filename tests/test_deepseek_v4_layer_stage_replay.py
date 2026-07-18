from __future__ import annotations

import json

import pytest

from moespresso.runtime.deepseek_v4 import layer_stage_replay as replay


def test_layer_stage_replay_cli_writes_json(monkeypatch, tmp_path):
    out = tmp_path / "layer-stage.json"
    calls = []

    def fake_run(package_dir, **kwargs):
        calls.append((package_dir, kwargs))
        return {
            "metric": "ds4_decoder_layer_stage_replay",
            "case": {"layer": kwargs["layer"], "input_tokens": kwargs["input_tokens"]},
        }

    monkeypatch.setattr(replay, "run_ds4_layer_stage_replay", fake_run)

    assert replay.main([
        "/pkg",
        "--layer",
        "3",
        "--input-tokens",
        "3844",
        "--pooled-rows",
        "961",
        "--repeats",
        "2",
        "--warmup",
        "0",
        "--max-memory-gb",
        "512",
        "--no-prime-indexer-qat",
        "--include-moe",
        "--json-out",
        str(out),
    ]) == 0

    payload = json.loads(out.read_text())
    assert payload["metric"] == "ds4_decoder_layer_stage_replay"
    assert calls[0][0].as_posix() == "/pkg"
    assert calls[0][1] == {
        "layer": 3,
        "input_tokens": 3844,
        "pooled_rows": 961,
        "repeats": 2,
        "warmup": 0,
        "max_memory_gb": 512.0,
        "prime_indexer_qat": False,
        "include_moe": True,
    }


@pytest.mark.parametrize(
    "args",
    [
        ["/pkg", "--layer", "-1"],
        ["/pkg", "--input-tokens", "0"],
        ["/pkg", "--pooled-rows", "-1"],
        ["/pkg", "--repeats", "0"],
        ["/pkg", "--warmup", "-1"],
        ["/pkg", "--max-memory-gb", "0"],
    ],
)
def test_layer_stage_replay_cli_rejects_invalid_numeric_args(args):
    with pytest.raises(SystemExit):
        replay.main(args)
