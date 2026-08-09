"""Focused DSpark dense-attention geometry checks.

The DeepSeek reference implements DSpark with its generic indexed-attention
kernel, but supplies every live main-stream window row and every draft-block
row to every draft query. These tests compare the MLX path with a direct dense
calculation so a target-model sparse-attention path cannot enter by accident.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from jang_tools.dsv4.mlx_model import (
    ModelArgs,
    _apply_partial_rope,
    _get_q_norm_ones,
)
from mlx.utils import tree_map

from moespresso.runtime.deepseek_v4 import dspark_model
from moespresso.runtime.deepseek_v4.dspark_model import (
    DSparkArgs,
    DSparkDraftAttention,
    DSparkWindowCache,
    _fp8_kv_roundtrip,
)


def _model_args() -> ModelArgs:
    return ModelArgs(
        model_type="deepseek_v4",
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        num_hash_layers=0,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        hc_mult=2,
        hc_sinkhorn_iters=5,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        sliding_window=4,
        rms_norm_eps=1e-6,
        compress_ratios=[0],
    )


def _attention(seed: int = 31) -> DSparkDraftAttention:
    args = _model_args()
    dargs = DSparkArgs(
        model_args=args,
        n_mtp_layers=1,
        block_size=3,
        noise_token_id=31,
        target_layer_ids=(0,),
        markov_rank=8,
    )
    attention = DSparkDraftAttention(
        dargs.model_args,
        layer_id=dargs.model_args.num_hidden_layers,
    )

    def random_value(value):
        nonlocal seed
        seed += 1
        return mx.random.normal(value.shape, key=mx.random.key(seed)) * 0.08

    attention.update(tree_map(random_value, attention.parameters()))
    mx.eval(attention.parameters())
    return attention


def _project_block(attention, x, positions):
    batch, length, _ = x.shape
    position_array = mx.array(positions, dtype=mx.int32)

    q = attention.q_norm(attention.wq_a(x))
    q = attention.wq_b(q).reshape(batch, length, attention.n_heads, attention.head_dim)
    q = mx.fast.rms_norm(
        q,
        weight=_get_q_norm_ones(attention.head_dim, q.dtype),
        eps=attention.args.rms_norm_eps,
    )
    q = q.transpose(0, 2, 1, 3)
    q = _apply_partial_rope(q, attention.rope, positions=position_array)

    kv = attention.kv_norm(attention.wkv(x)).reshape(batch, length, 1, attention.head_dim)
    kv = kv.transpose(0, 2, 1, 3)
    kv = _apply_partial_rope(kv, attention.rope, positions=position_array)
    return q, _fp8_kv_roundtrip(kv)


def _dense_reference(attention, x, positions, window):
    """Direct dense attention over compacted live ring rows plus the block."""
    q, block_kv = _project_block(attention, x, positions)
    live_rows = min(window.window, window.last_pos + 1)
    main_kv = window.kv[:, :, :live_rows, :].astype(block_kv.dtype)
    full_kv = mx.concatenate([main_kv, block_kv], axis=2)

    scores = mx.matmul(
        q.astype(mx.float32),
        mx.swapaxes(full_kv, -1, -2).astype(mx.float32),
    )
    scores = scores * attention.softmax_scale
    sink = attention.attn_sink.astype(q.dtype).astype(mx.float32)
    sink = mx.broadcast_to(sink[None, :, None, None], scores.shape[:-1] + (1,))
    probabilities = mx.softmax(mx.concatenate([sink, scores], axis=-1), axis=-1, precise=True)[
        ..., 1:
    ]
    out = mx.matmul(probabilities, full_kv.astype(mx.float32)).astype(q.dtype)

    position_array = mx.array(positions, dtype=mx.int32)
    out = _apply_partial_rope(out, attention.rope, positions=position_array, inverse=True)
    batch, _, length, _ = out.shape
    out = out.transpose(0, 2, 1, 3).reshape(batch, length, attention.n_heads * attention.head_dim)
    group_features = (attention.n_heads * attention.head_dim) // attention.o_groups
    out = out.reshape(batch, length, attention.o_groups, group_features)
    wo_a = attention.wo_a.weight.reshape(attention.o_groups, attention.o_lora_rank, group_features)
    out = mx.einsum("bsgd,grd->bsgr", out, wo_a)
    out = out.reshape(batch, length, attention.o_groups * attention.o_lora_rank)
    return out @ attention.wo_b.weight.T


def _window_case(wrapped: bool):
    window = DSparkWindowCache(window=4, head_dim=32)
    if wrapped:
        rows = mx.random.normal((1, 6, 32), key=mx.random.key(41))
        window.write(rows[:, :4], [0, 1, 2, 3])
        window.write(rows[:, 4:], [4, 5])
        positions = [6, 7, 8]
    else:
        rows = mx.random.normal((1, 2, 32), key=mx.random.key(42))
        window.write(rows, [0, 1])
        # Poison unused physical slots. Only the visibility mask may exclude
        # them; zero-filled slots could let a missing mask pass accidentally.
        window.kv[:, :, 2:, :] = 1000.0
        positions = [2, 3, 4]
    x = mx.random.normal((1, 3, 64), key=mx.random.key(43 + wrapped))
    mx.eval(window.kv, x)
    return window, positions, x


@pytest.mark.parametrize("wrapped", [False, True], ids=["pre_wrap", "post_wrap"])
def test_dspark_attention_is_dense_over_live_window_and_full_block(monkeypatch, wrapped):
    attention = _attention()
    window, positions, x = _window_case(wrapped)

    # Draft stages are declared dense before the forward starts. No target
    # compressor, indexer, or compressed-cache state is constructed.
    assert attention.compress_ratio == 0
    assert not hasattr(attention, "compressor")
    assert not hasattr(attention, "indexer")

    original_attention = dspark_model.scaled_dot_product_attention
    captured = {}

    def capture(q, keys, values, **kwargs):
        captured.update(q=q, keys=keys, values=values, kwargs=kwargs)
        return original_attention(q, keys, values, **kwargs)

    monkeypatch.setattr(dspark_model, "scaled_dot_product_attention", capture)
    actual = attention(x, positions=positions, window=window)
    expected = _dense_reference(attention, x, positions, window)
    expected_q, expected_block_kv = _project_block(attention, x, positions)
    mx.eval(actual, expected, expected_q, expected_block_kv)

    # The query sequence is exactly the draft block. The physical key/value
    # sequence is the main-stream ring followed by every draft-block row.
    np.testing.assert_array_equal(np.asarray(captured["q"]), np.asarray(expected_q))
    expected_keys = mx.concatenate(
        [window.kv.astype(expected_block_kv.dtype), expected_block_kv], axis=2
    )
    mx.eval(expected_keys)
    np.testing.assert_array_equal(np.asarray(captured["keys"]), np.asarray(expected_keys))
    np.testing.assert_array_equal(np.asarray(captured["values"]), np.asarray(expected_keys))

    mask = captured["kwargs"]["mask"]
    if wrapped:
        assert mask is None
    else:
        expected_visibility = np.array(
            [[[[True, True, False, False, True, True, True]] * 3]],
            dtype=np.bool_,
        )
        np.testing.assert_array_equal(np.asarray(mask), expected_visibility)

    # The production fused SDPA and the compact explicit calculation differ
    # only in reduction order. Inputs are float32, so a 3e-5 tolerance is well
    # below the draft checkpoint's bf16 activation precision.
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=3e-5, atol=3e-5)
