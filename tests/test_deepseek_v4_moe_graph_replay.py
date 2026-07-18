from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from moespresso.correctness.deepseek_v4 import moe_graph_replay as replay


def _run_payload(*, value: float):
    capture = np.full((2, 4096), value, dtype=np.float32).reshape(-1)
    scores = mx.array(np.full((1, 2, 8), value, dtype=np.float32))
    return {
        "manifest": {"artifact_id": "pkg:test"},
        "tokens": [1, 2],
        "rendered_prompt": "prompt",
        "layers": [0],
        "captures": {"0:ffn_out": capture},
        "scores": scores,
        "generated_tokens": [],
        "token_logprobs": [],
        "top_logprobs": [],
        "stats": {"enabled": True, "switch_calls": int(value)},
    }


def test_compare_graph_runs_reports_equal_states():
    direct = _run_payload(value=1.0)
    wrapped = _run_payload(value=1.0)

    payload = replay.compare_graph_runs(direct, wrapped, stages=("ffn_out",))

    assert payload["first_non_equal"] is None
    assert payload["layers"][0]["stages"]["ffn_out"]["all"]["array_equal"] is True
    assert payload["scores"]["final_argmax_equal"] is True
    assert payload["direct_ssd_streaming_stats"]["switch_calls"] == 1


def test_compare_graph_runs_reports_first_non_equal_layer_and_stage():
    direct = _run_payload(value=1.0)
    wrapped = _run_payload(value=1.0)
    wrapped["captures"]["0:ffn_out"] = np.full((2, 4096), 1.0, dtype=np.float32)
    wrapped["captures"]["0:ffn_out"][1, 0] = 2.0
    wrapped["captures"]["0:ffn_out"] = wrapped["captures"]["0:ffn_out"].reshape(-1)

    payload = replay.compare_graph_runs(direct, wrapped, stages=("ffn_out",))

    assert payload["first_non_equal"]["layer"] == 0
    assert payload["first_non_equal"]["stage"] == "ffn_out"
    assert payload["first_non_equal"]["scope"] == "all"
    assert payload["layers"][0]["stages"]["ffn_out"]["final"]["max_abs"] == 1.0


def test_compare_graph_runs_reports_top_logprob_token_id_equality():
    direct = _run_payload(value=1.0)
    wrapped = _run_payload(value=1.0)
    top_logprobs = [
        [{"token_id": 3, "logprob": -0.1}, {"token_id": 4, "logprob": -0.2}],
        [{"token_id": 5, "logprob": -0.3}],
    ]
    direct["top_logprobs"] = top_logprobs
    wrapped["top_logprobs"] = top_logprobs

    payload = replay.compare_graph_runs(direct, wrapped, stages=("ffn_out",))

    assert payload["top_logprob_token_ids_equal"] is True
    assert payload["direct_top_logprob_token_ids"] == [[3, 4], [5]]


def test_run_graph_replay_loads_direct_then_wrapped(monkeypatch, tmp_path):
    calls = []

    def fake_capture_run(
        *,
        package_dir: Path,
        prompt_path: Path,
        prompt_mode: str,
        layers: list[int],
        stages,
        wrap_moe_block: bool,
        mode: str,
        max_tokens: int,
    ):
        del package_dir, prompt_path, prompt_mode, stages
        calls.append(wrap_moe_block)
        value = 2.0 if wrap_moe_block else 2.0
        payload = _run_payload(value=value)
        payload["layers"] = list(layers or [0])
        if mode in {"generate-step", "metadata"}:
            payload["generated_tokens"] = [7] * max_tokens
        if mode == "metadata":
            payload["token_logprobs"] = [-0.5] * max_tokens
            payload["top_logprobs"] = [
                [{"token_id": 7, "logprob": -0.5}]
                for _ in range(max_tokens)
            ]
        if mode in {"generate-step", "metadata"}:
            payload["scores"] = mx.array(np.full((8,), value, dtype=np.float32))
        return payload

    monkeypatch.setattr(replay, "_load_capture_run", fake_capture_run)
    monkeypatch.setattr(replay, "_cleanup_mlx", lambda: None)

    payload = replay.run_graph_replay(
        package_dir=tmp_path,
        prompt_path=tmp_path / "prompt.txt",
        prompt_mode="raw-user",
        layers=[],
        stages=("ffn_out",),
        mode="generate-step",
        max_tokens=2,
    )

    assert calls == [False, True]
    assert payload["requested_layers"] == "all"
    assert payload["first_non_equal"] is None
    assert payload["generated_tokens_equal"] is True
    assert payload["direct_generated_tokens"] == [7, 7]


def test_run_graph_replay_supports_metadata_mode(monkeypatch, tmp_path):
    calls = []

    def fake_capture_run(
        *,
        package_dir: Path,
        prompt_path: Path,
        prompt_mode: str,
        layers: list[int],
        stages,
        wrap_moe_block: bool,
        mode: str,
        max_tokens: int,
    ):
        del package_dir, prompt_path, prompt_mode, stages
        calls.append((wrap_moe_block, mode))
        payload = _run_payload(value=3.0)
        payload["layers"] = list(layers or [0])
        payload["generated_tokens"] = [9] * max_tokens
        payload["token_logprobs"] = [-0.25] * max_tokens
        payload["top_logprobs"] = [
            [{"token_id": 9, "logprob": -0.25}]
            for _ in range(max_tokens)
        ]
        return payload

    monkeypatch.setattr(replay, "_load_capture_run", fake_capture_run)
    monkeypatch.setattr(replay, "_cleanup_mlx", lambda: None)

    payload = replay.run_graph_replay(
        package_dir=tmp_path,
        prompt_path=tmp_path / "prompt.txt",
        prompt_mode="raw-user",
        layers=[],
        stages=("ffn_out",),
        mode="metadata",
        max_tokens=3,
    )

    assert calls == [(False, "metadata"), (True, "metadata")]
    assert payload["max_tokens"] == 3
    assert payload["generated_tokens_equal"] is True
    assert payload["top_logprob_token_ids_equal"] is True
    assert payload["direct_token_logprobs"] == [-0.25, -0.25, -0.25]
