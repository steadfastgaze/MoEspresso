from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from moespresso.correctness.deepseek_v4.moe_block_replay import (
    _make_inputs,
    compare_direct_and_pooled_block,
    run_replay,
)


class _Gate(nn.Module):
    def __init__(self, *, n_experts=8, top_k=4):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.seen_input_ids = []

    def __call__(self, x, input_ids=None):
        if input_ids is None:
            raise AssertionError("input_ids required")
        self.seen_input_ids.append(input_ids)
        base = input_ids.astype(mx.uint32)
        inds = mx.concatenate(
            [((base + i) % self.n_experts)[..., None] for i in range(self.top_k)],
            axis=-1,
        )
        scores = mx.ones(inds.shape, dtype=x.dtype) / self.top_k
        return inds, scores


class _Switch(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.gate_proj = SimpleNamespace(in_features=hidden)
        self.block_exit_kick_calls = 0
        self.decode_moe_block_calls = 0
        self.decode_moe_block_seconds = 0.0

    def begin_projection_load(self, _indices):
        return None

    def __call__(self, x, inds, *, load_ticket=None):
        del load_ticket
        k = int(inds.shape[-1])
        expanded = mx.expand_dims(x, -2)
        return mx.broadcast_to(expanded, (*x.shape[:-1], k, x.shape[-1]))


class _Shared(nn.Module):
    def __call__(self, x):
        return x * mx.array(0.25, dtype=x.dtype)


class _Mlp(nn.Module):
    def __init__(self, *, hidden=8):
        super().__init__()
        self.gate = _Gate()
        self.switch_mlp = _Switch(hidden=hidden)
        self.shared_experts = _Shared()

    def __call__(self, x, input_ids=None):
        inds, scores = self.gate(x, input_ids=input_ids)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype).reshape(x.shape)
        return y + self.shared_experts(x)


def test_make_inputs_is_deterministic():
    a_x, a_ids = _make_inputs(
        hidden_size=8,
        vocab_size=100,
        batch_size=2,
        seq_len=3,
        seed=7,
    )
    b_x, b_ids = _make_inputs(
        hidden_size=8,
        vocab_size=100,
        batch_size=2,
        seq_len=3,
        seed=7,
    )

    np.testing.assert_array_equal(np.array(a_x), np.array(b_x))
    np.testing.assert_array_equal(np.array(a_ids), np.array(b_ids))


def test_compare_direct_and_pooled_block_matches_formula():
    mlp = _Mlp(hidden=8)
    x, input_ids = _make_inputs(
        hidden_size=8,
        vocab_size=100,
        batch_size=1,
        seq_len=2,
        seed=11,
    )

    metrics = compare_direct_and_pooled_block(mlp, x=x, input_ids=input_ids)

    assert metrics["array_equal"] is True
    assert metrics["max_abs"] == 0.0
    assert metrics["direct_type"] == "_Mlp"
    assert metrics["candidate_type"] == "PooledDeepseekV4MoEBlock"
    assert len(mlp.gate.seen_input_ids) == 2


def test_run_replay_uses_loader_and_reports_metrics(monkeypatch, tmp_path):
    import moespresso.runtime.serve as serve
    import moespresso.runtime.ssd_streaming_build as streaming

    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(mlp=_Mlp(hidden=8)),
            ],
        ),
    )
    manifest = {
        "artifact_id": "pkg:test",
        "architecture": {
            "family": "deepseek_v4_flash",
            "config": {"vocab_size": 100},
        },
    }

    monkeypatch.setattr(
        serve,
        "load_served_model",
        lambda _package: (model, object(), manifest),
    )
    monkeypatch.setattr(
        streaming,
        "ssd_streaming_stats",
        lambda _model: {"enabled": True, "switch_calls": 2},
    )

    payload = run_replay(
        package_dir=tmp_path,
        layer=0,
        seq_len=1,
        batch_size=1,
        seed=13,
    )

    assert payload["artifact_kind"] == "deepseek_v4_moe_block_replay"
    assert payload["package_manifest_id"] == "pkg:test"
    assert payload["metrics"]["array_equal"] is True
    assert payload["ssd_streaming_stats"] == {"enabled": True, "switch_calls": 2}
