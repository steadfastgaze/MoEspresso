from __future__ import annotations

import copy
import math

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.dsv4.mlx_model")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import jang_tools.dsv4.mlx_model as jang_dsv4  # noqa: E402

from moespresso.runtime.deepseek_v4.model import (  # noqa: E402
    DeepseekV4GraphError,
    _AttentionCompressorFp8KV,
    _BandedPrefillAttention,
    _patch_deepseek_v4_attention_compressor_fp8_kv,
    _patch_deepseek_v4_attention_fp16_qkv,
    _patch_deepseek_v4_attention_shape_stats,
    _patch_deepseek_v4_banded_prefill_attention,
    _patch_deepseek_v4_indexer_score_contract,
    _patch_deepseek_v4_indexer_pre_topk_visibility,
    _patch_deepseek_v4_ratio4_prefill_fast_path,
    _patch_deepseek_v4_required_attention_cache,
    _load_empty_deepseek_v4_skeleton,
    build_deepseek_v4_graph_from_manifest,
    deepseek_v4_attention_layer_stats,
    deepseek_v4_indexer_layer_stats,
)
from moespresso.package.sidecars import build_sidecars  # noqa: E402


def _tiny_manifest():
    config = {
        "model_type": "deepseek_v4",
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 16,
        "o_lora_rank": 16,
        "o_groups": 2,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "num_hash_layers": 0,
        "num_nextn_predict_layers": 1,
        "sliding_window": 8,
        "compress_ratios": [0],
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
    }
    return {
        "architecture": {
            "family": "deepseek_v4_flash",
            "config": config,
            "compress_ratios": [0],
        }
    }


def test_deepseek_v4_graph_builds_from_tiny_manifest_and_runs_forward():
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    tokens = mx.array([[1, 2, 3]], dtype=mx.uint32)

    logits = model(tokens)
    mx.eval(logits)

    assert logits.shape == (1, 3, 128)


def test_deepseek_v4_graph_requires_family_and_explicit_compress_ratios():
    manifest = _tiny_manifest()
    manifest["architecture"]["family"] = "qwen3_5_moe"
    with pytest.raises(DeepseekV4GraphError, match="not a DeepSeek"):
        build_deepseek_v4_graph_from_manifest(manifest)

    manifest = _tiny_manifest()
    manifest["architecture"]["config"].pop("compress_ratios")
    manifest["architecture"].pop("compress_ratios")
    with pytest.raises(DeepseekV4GraphError, match="compress_ratios"):
        build_deepseek_v4_graph_from_manifest(manifest)


def test_deepseek_v4_graph_rejects_ratio_length_mismatch():
    manifest = _tiny_manifest()
    manifest["architecture"]["config"]["compress_ratios"] = [0, 4, 128]

    with pytest.raises(DeepseekV4GraphError, match="compress_ratios length"):
        build_deepseek_v4_graph_from_manifest(manifest)


def test_deepseek_v4_graph_pins_attention_scale_without_mscale():
    manifest = _tiny_manifest()
    cfg = manifest["architecture"]["config"]
    cfg["compress_ratios"] = [4]
    cfg["rope_scaling"] = {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    manifest["architecture"]["compress_ratios"] = [4]

    model = build_deepseek_v4_graph_from_manifest(manifest)
    attention = model.layers[0].self_attn

    assert math.isclose(attention.softmax_scale, cfg["head_dim"] ** -0.5)
    assert not hasattr(attention.rope, "mscale")
    assert not hasattr(attention.rope, "attention_factor")


def test_deepseek_v4_served_indexer_applies_head_and_head_count_scale(monkeypatch):
    manifest = _tiny_manifest()
    cfg = manifest["architecture"]["config"]
    cfg["compress_ratios"] = [4]
    cfg["index_n_heads"] = 2
    cfg["index_head_dim"] = 8
    cfg["index_topk"] = 2
    manifest["architecture"]["compress_ratios"] = [4]

    model = build_deepseek_v4_graph_from_manifest(manifest)
    indexer = model.layers[0].self_attn.indexer

    class FakeCompressor:
        def __call__(self, x, rope, cache, start_pos, state_key="indexer_state"):
            del rope, cache, start_pos, state_key
            return mx.ones((x.shape[0], 3, cfg["index_head_dim"]), dtype=mx.float32)

    class FakeLinear:
        def __init__(self, width):
            self.width = width

        def __call__(self, x):
            return mx.ones((x.shape[0], x.shape[1], self.width), dtype=mx.float32)

    class IdentityRope:
        dims = cfg["index_head_dim"]

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            return x

    captured = {}
    real_argpartition = jang_dsv4.mx.argpartition

    def spy_argpartition(scores, kth, axis=-1):
        captured["scores"] = np.array(-scores)
        return real_argpartition(scores, kth=kth, axis=axis)

    indexer.compressor = FakeCompressor()
    indexer.wq_b = FakeLinear(cfg["index_n_heads"] * cfg["index_head_dim"])
    indexer.weights_proj = FakeLinear(cfg["index_n_heads"])
    monkeypatch.setattr(jang_dsv4.mx, "argpartition", spy_argpartition)

    topk = indexer(
        mx.zeros((1, 1, cfg["hidden_size"]), dtype=mx.float32),
        mx.zeros((1, 1, cfg["q_lora_rank"]), dtype=mx.float32),
        IdentityRope(),
        IdentityRope(),
        None,
        0,
    )
    mx.eval(topk)

    expected = (cfg["index_head_dim"] ** 0.5) * (cfg["index_n_heads"] ** 0.5)
    np.testing.assert_allclose(
        captured["scores"],
        np.full((1, 1, 3), expected, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_deepseek_v4_served_indexer_masks_invisible_rows_before_topk(monkeypatch):
    manifest = _tiny_manifest()
    cfg = manifest["architecture"]["config"]
    cfg["compress_ratios"] = [4]
    cfg["index_n_heads"] = 1
    cfg["index_head_dim"] = 128
    cfg["index_topk"] = 2
    manifest["architecture"]["compress_ratios"] = [4]

    model = build_deepseek_v4_graph_from_manifest(manifest)
    assert _patch_deepseek_v4_indexer_pre_topk_visibility(model) == 1
    indexer = model.layers[0].self_attn.indexer

    class FakeCompressor:
        def __call__(self, x, rope, cache, start_pos, state_key="indexer_state"):
            del x, rope, cache, start_pos, state_key
            rows = np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1)
            return mx.array(np.repeat(rows, cfg["index_head_dim"], axis=-1))

    class FakeLinear:
        def __init__(self, width):
            self.width = width

        def __call__(self, x):
            return mx.ones((x.shape[0], x.shape[1], self.width), dtype=mx.float32)

    class IdentityRope:
        dims = cfg["index_head_dim"]

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            return x

    captured = {}
    real_argpartition = jang_dsv4.mx.argpartition

    def spy_argpartition(values, kth, axis=-1):
        captured["scores"] = np.array(-values)
        return real_argpartition(values, kth=kth, axis=axis)

    indexer._original.compressor = FakeCompressor()
    indexer._original.wq_b = FakeLinear(cfg["index_n_heads"] * cfg["index_head_dim"])
    indexer._original.weights_proj = FakeLinear(cfg["index_n_heads"])
    monkeypatch.setattr(jang_dsv4.mx, "argpartition", spy_argpartition)

    topk = indexer(
        mx.zeros((1, 2, cfg["hidden_size"]), dtype=mx.float32),
        mx.zeros((1, 2, cfg["q_lora_rank"]), dtype=mx.float32),
        IdentityRope(),
        IdentityRope(),
        None,
        7,
    )
    mx.eval(topk)

    assert captured["scores"].shape == (1, 2, 4)
    assert np.all(np.isfinite(captured["scores"][0, :, :2]))
    assert np.all(np.isneginf(captured["scores"][0, :, 2:]))
    assert set(np.array(topk).reshape(-1).tolist()) <= {0, 1}


def test_deepseek_v4_served_indexer_skips_visibility_mask_for_decode(
    monkeypatch,
):
    manifest = _tiny_manifest()
    cfg = manifest["architecture"]["config"]
    cfg["compress_ratios"] = [4]
    cfg["index_n_heads"] = 1
    cfg["index_head_dim"] = 128
    cfg["index_topk"] = 2
    manifest["architecture"]["compress_ratios"] = [4]

    model = build_deepseek_v4_graph_from_manifest(manifest)
    assert _patch_deepseek_v4_indexer_score_contract(model) == 1
    indexer = model.layers[0].self_attn.indexer

    class FakeCompressor:
        def __call__(self, x, rope, cache, start_pos, state_key="indexer_state"):
            del x, rope, cache, start_pos, state_key
            rows = np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1)
            return mx.array(np.repeat(rows, cfg["index_head_dim"], axis=-1))

    class FakeLinear:
        def __init__(self, width):
            self.width = width

        def __call__(self, x):
            return mx.ones((x.shape[0], x.shape[1], self.width), dtype=mx.float32)

    class IdentityRope:
        dims = cfg["index_head_dim"]

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            return x

    def fail_visibility(*args, **kwargs):
        del args, kwargs
        raise AssertionError("decode indexer should not build a visibility mask")

    indexer._original.compressor = FakeCompressor()
    indexer._original.wq_b = FakeLinear(cfg["index_n_heads"] * cfg["index_head_dim"])
    indexer._original.weights_proj = FakeLinear(cfg["index_n_heads"])
    monkeypatch.setattr(jang_dsv4, "_dsv4_compressed_visibility", fail_visibility)

    topk = indexer(
        mx.zeros((1, 1, cfg["hidden_size"]), dtype=mx.float32),
        mx.zeros((1, 1, cfg["q_lora_rank"]), dtype=mx.float32),
        IdentityRope(),
        IdentityRope(),
        None,
        7,
    )
    mx.eval(topk)

    assert np.array(topk).shape == (1, 1, 2)


def test_deepseek_v4_ratio4_prefill_fast_path_patch_wraps_ratio4_attention():
    manifest = _tiny_manifest()
    cfg = manifest["architecture"]["config"]
    cfg["compress_ratios"] = [4]
    cfg["index_n_heads"] = 1
    cfg["index_head_dim"] = 128
    cfg["index_topk"] = 2
    manifest["architecture"]["compress_ratios"] = [4]

    model = build_deepseek_v4_graph_from_manifest(manifest)

    assert _patch_deepseek_v4_ratio4_prefill_fast_path(model) == 1
    wrapped = model.layers[0].self_attn
    assert wrapped._moespresso_dsv4_ratio4_prefill_fast_path is True
    assert wrapped._original is not None
    assert _patch_deepseek_v4_ratio4_prefill_fast_path(model) == 0


def _shared_window_mask(tokens, offset, window):
    q_pos = offset + mx.arange(tokens)[:, None]
    k_pos = mx.arange(offset + tokens)[None, :]
    return (k_pos <= q_pos) & (k_pos > q_pos - window)


def test_deepseek_v4_banded_prefill_matches_dense_on_ratio0_layer(monkeypatch):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    attn = model.layers[0].self_attn
    tokens = 13
    x = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    mask = _shared_window_mask(tokens, 0, 8)
    reference = attn(x, mask=mask, cache=KVCache())
    mx.eval(reference)

    assert _patch_deepseek_v4_banded_prefill_attention(model) == 1
    assert model._moespresso_dsv4_banded_prefill_layers == 1
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 0
    wrapped = model.layers[0].self_attn
    assert wrapped._moespresso_dsv4_banded_prefill is True

    banded = wrapped(x, mask=mask, cache=KVCache())
    mx.eval(banded)

    assert wrapped.banded_prefill_calls == 1
    assert np.allclose(np.asarray(reference), np.asarray(banded), atol=2e-3)


def test_deepseek_v4_banded_prefill_matches_dense_with_pooled_tail(monkeypatch):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch

    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    manifest = _tiny_manifest()
    cfg = manifest["architecture"]["config"]
    cfg["compress_ratios"] = [4]
    cfg["index_topk"] = 64
    manifest["architecture"]["compress_ratios"] = [4]
    model = build_deepseek_v4_graph_from_manifest(manifest)
    attn = model.layers[0].self_attn

    # The installer only targets ratio-0 and ratio-128 layers; a ratio-4
    # attention is wrapped by hand to exercise the pooled visibility tail.
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 0
    wrapped = _BandedPrefillAttention(
        attn,
        mx=mx,
        dsv4_model=jang_dsv4,
        cache_cls=jang_dsv4.DeepseekV4Cache,
    )

    tokens = 14
    x = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    mask = _shared_window_mask(tokens, 0, 8)
    reference = attn(
        x, mask=mask, cache=jang_dsv4.DeepseekV4Cache(8, compress_ratio=4))
    mx.eval(reference)

    banded = wrapped(
        x, mask=mask, cache=jang_dsv4.DeepseekV4Cache(8, compress_ratio=4))
    mx.eval(banded)

    assert wrapped.banded_prefill_calls == 1
    assert np.allclose(np.asarray(reference), np.asarray(banded), atol=2e-3)


def test_deepseek_v4_banded_prefill_delegates_second_chunk_and_float_mask(
    monkeypatch,
):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    attn = model.layers[0].self_attn
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 1
    wrapped = model.layers[0].self_attn

    tokens = 13
    x_first = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    x_second = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    bool_first = _shared_window_mask(tokens, 0, 8)
    float_first = mx.where(
        bool_first,
        mx.array(0.0, dtype=mx.float32),
        mx.array(-float("inf"), dtype=mx.float32),
    )
    bool_second = _shared_window_mask(tokens, tokens, 8)

    cache_reference = KVCache()
    reference_first = attn(x_first, mask=float_first, cache=cache_reference)
    reference_second = attn(x_second, mask=bool_second, cache=cache_reference)
    mx.eval(reference_first, reference_second)

    cache_wrapped = KVCache()
    banded_first = wrapped(x_first, mask=float_first, cache=cache_wrapped)
    banded_second = wrapped(x_second, mask=bool_second, cache=cache_wrapped)
    mx.eval(banded_first, banded_second)

    assert wrapped.banded_prefill_calls == 0
    np.testing.assert_array_equal(
        np.asarray(reference_first), np.asarray(banded_first))
    np.testing.assert_array_equal(
        np.asarray(reference_second), np.asarray(banded_second))


def test_deepseek_v4_banded_prefill_small_geometry_keeps_sdpa_form(monkeypatch):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from moespresso.runtime.deepseek_v4.model import banded_prefill_call_counts
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 1
    wrapped = model.layers[0].self_attn

    tokens = 13
    x = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    mask = _shared_window_mask(tokens, 0, 8)
    before = banded_prefill_call_counts()
    out = wrapped(x, mask=mask, cache=KVCache())
    mx.eval(out)
    after = banded_prefill_call_counts()

    # head_dim 32 fails the mma eligibility, so the engaged form is the
    # batched banded SDPA even with the route gate at its default.
    assert after["sdpa"] == before["sdpa"] + 1
    assert after["mma"] == before["mma"]
    assert wrapped.banded_prefill_calls == 1


def test_deepseek_v4_banded_prefill_mma_dispatch_and_kill_switch(monkeypatch):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from moespresso.runtime.deepseek_v4.model import banded_prefill_call_counts
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    attn = model.layers[0].self_attn
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 1
    wrapped = model.layers[0].self_attn

    tokens = 13
    x = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    mask = _shared_window_mask(tokens, 0, 8)
    reference = attn(x, mask=mask, cache=KVCache())
    mx.eval(reference)

    calls = []

    def fake_engine(mx_mod, **kwargs):
        calls.append(kwargs)
        q = kwargs["q"]
        return mx.zeros(
            (1, int(q.shape[1]), int(q.shape[2]), int(q.shape[3])),
            dtype=mx.float32,
        )

    monkeypatch.setattr(
        dsv4_patch, "_deepseek_v4_banded_mma_attention", fake_engine)

    before = banded_prefill_call_counts()
    out = wrapped(x, mask=mask, cache=KVCache())
    mx.eval(out)
    after = banded_prefill_call_counts()
    assert len(calls) == 1
    assert after["mma"] == before["mma"] + 1
    assert after["sdpa"] == before["sdpa"]
    assert int(calls[0]["window"]) == 8
    assert float(calls[0]["scale"]) == float(attn.softmax_scale)

    # The kill switch restores the batched banded SDPA form without
    # touching the engine.
    monkeypatch.setenv("MOESPRESSO_DSV4_BANDED_PREFILL_MMA", "0")
    out_off = wrapped(x, mask=mask, cache=KVCache())
    mx.eval(out_off)
    final = banded_prefill_call_counts()
    assert len(calls) == 1
    assert final["sdpa"] == after["sdpa"] + 1
    assert final["mma"] == after["mma"]
    assert np.allclose(np.asarray(reference), np.asarray(out_off), atol=2e-3)


def _seed_composed_first_chunk(attn, cache, tokens, window, seed=311):
    x = mx.array(
        np.random.default_rng(seed).standard_normal(
            (1, tokens, 64)).astype(np.float32) * 0.1)
    out = attn(x, mask=_shared_window_mask(tokens, 0, window), cache=cache)
    mx.eval(out)
    return out


def test_deepseek_v4_banded_prefill_offset_kill_switch_restores_composed(
    monkeypatch,
):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from moespresso.runtime.deepseek_v4.model import banded_prefill_call_counts
    from mlx_lm.models.cache import KVCache

    monkeypatch.setenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", "0")
    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    attn = model.layers[0].self_attn
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 1
    wrapped = model.layers[0].self_attn

    tokens, window = 13, 8
    cache_reference = KVCache()
    cache_wrapped = KVCache()
    _seed_composed_first_chunk(attn, cache_reference, tokens, window)
    _seed_composed_first_chunk(attn, cache_wrapped, tokens, window)

    x_second = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    mask_second = _shared_window_mask(tokens, tokens, window)
    reference = attn(x_second, mask=mask_second, cache=cache_reference)
    before = banded_prefill_call_counts()
    got = wrapped(x_second, mask=mask_second, cache=cache_wrapped)
    mx.eval(reference, got)

    # The kill-switch arm never engages, never spies on the mask, and
    # never moves any counter, including the offset-keyed ones.
    assert banded_prefill_call_counts() == before
    assert wrapped.banded_prefill_calls == 0
    np.testing.assert_array_equal(np.asarray(reference), np.asarray(got))

    # Default on: with the variable unset the wrapper attempts the offset
    # route. The tiny graph fails the hoisted mma eligibility (head dim
    # 32), so the call falls closed to the composed original and keys the
    # offset counter, which proves the gate itself was open.
    monkeypatch.delenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", raising=False)
    got_default = wrapped(
        mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1,
        mask=_shared_window_mask(tokens, 2 * tokens, window),
        cache=cache_wrapped,
    )
    mx.eval(got_default)
    after = banded_prefill_call_counts()
    assert after["composed_offset"] == before["composed_offset"] + 1
    assert wrapped.banded_prefill_calls == 0


def test_deepseek_v4_banded_prefill_offset_engages_mma_with_counters(
    monkeypatch,
):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from moespresso.runtime.deepseek_v4.model import banded_prefill_call_counts
    from mlx_lm.models.cache import KVCache

    # The route is default-on; the unset environment pins the shipped
    # default rather than an explicit opt-in.
    monkeypatch.delenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", raising=False)
    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    attn = model.layers[0].self_attn
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 1
    wrapped = model.layers[0].self_attn

    tokens, window = 13, 8
    cache = KVCache()
    _seed_composed_first_chunk(attn, cache, tokens, window)

    engine_calls = []

    def fake_engine(mx_mod, **kwargs):
        engine_calls.append(kwargs)
        q = kwargs["q"]
        return mx.zeros(
            (1, int(q.shape[1]), int(q.shape[2]), int(q.shape[3])),
            dtype=mx.float32,
        )

    rope_calls = []
    real_rope = jang_dsv4._apply_partial_rope

    def spy_rope(x, rope, offset, inverse=False, **kwargs):
        rope_calls.append((int(offset), bool(inverse)))
        return real_rope(x, rope, offset, inverse=inverse, **kwargs)

    monkeypatch.setattr(
        dsv4_patch, "_deepseek_v4_banded_mma_attention", fake_engine)
    monkeypatch.setattr(
        dsv4_patch,
        "_deepseek_v4_banded_mma_offset_ready",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(jang_dsv4, "_apply_partial_rope", spy_rope)

    x_second = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    mask_second = _shared_window_mask(tokens, tokens, window)
    before = banded_prefill_call_counts()
    out = wrapped(x_second, mask=mask_second, cache=cache)
    mx.eval(out)
    after = banded_prefill_call_counts()

    # The engine receives the true offset as pos0 and the counters key the
    # engagement by offset.
    assert len(engine_calls) == 1
    assert int(engine_calls[0]["pos0"]) == tokens
    assert int(engine_calls[0]["window"]) == window
    assert after["mma_offset"] == before["mma_offset"] + 1
    assert after["mma"] == before["mma"]
    assert after["sdpa"] == before["sdpa"]
    assert after["composed_offset"] == before["composed_offset"]
    assert wrapped.banded_prefill_calls == 1
    # Forward rope on q and kv and the inverse rope on the head output all
    # run at the true offset.
    assert rope_calls.count((tokens, False)) == 2
    assert rope_calls.count((tokens, True)) == 1
    # The cache advanced exactly one chunk.
    assert int(cache.offset) == 2 * tokens


def test_deepseek_v4_banded_prefill_offset_fails_closed_before_mutation(
    monkeypatch,
):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from moespresso.runtime.deepseek_v4.model import banded_prefill_call_counts
    from mlx_lm.models.cache import KVCache

    monkeypatch.setenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", "1")
    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    attn = model.layers[0].self_attn
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 1
    wrapped = model.layers[0].self_attn

    tokens, window = 13, 8
    cache_reference = KVCache()
    cache_wrapped = KVCache()
    _seed_composed_first_chunk(attn, cache_reference, tokens, window)
    _seed_composed_first_chunk(attn, cache_wrapped, tokens, window)

    update_calls = []
    original_update = cache_wrapped.update_and_fetch

    def spy_update(keys, values):
        update_calls.append(int(keys.shape[2]))
        return original_update(keys, values)

    cache_wrapped.update_and_fetch = spy_update

    x_second = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    mask_second = _shared_window_mask(tokens, tokens, window)
    reference = attn(x_second, mask=mask_second, cache=cache_reference)
    before = banded_prefill_call_counts()
    # head_dim 32 fails the hoisted mma eligibility, so the wrapper must
    # fall closed to the composed original with the cache untouched.
    got = wrapped(x_second, mask=mask_second, cache=cache_wrapped)
    mx.eval(reference, got)
    after = banded_prefill_call_counts()

    # Exactly one cache mutation happened, and the composed original made
    # it; a hoist failure after update_and_fetch would show two.
    assert update_calls == [tokens]
    assert int(cache_wrapped.offset) == 2 * tokens
    assert after["composed_offset"] == before["composed_offset"] + 1
    assert after["mma_offset"] == before["mma_offset"]
    assert after["mma"] == before["mma"]
    assert after["sdpa"] == before["sdpa"]
    assert wrapped.banded_prefill_calls == 0
    np.testing.assert_array_equal(np.asarray(reference), np.asarray(got))


def test_deepseek_v4_banded_prefill_offset_mask_mismatch_fails_closed(
    monkeypatch,
):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from moespresso.runtime.deepseek_v4.model import banded_prefill_call_counts
    from mlx_lm.models.cache import KVCache

    monkeypatch.setenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", "1")
    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    attn = model.layers[0].self_attn
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 1
    wrapped = model.layers[0].self_attn

    tokens, window = 13, 8
    cache = KVCache()
    _seed_composed_first_chunk(attn, cache, tokens, window)

    x_second = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    # An offset-zero-shaped mask cannot certify an offset chunk.
    wrong_mask = _shared_window_mask(tokens, 0, window)
    before = banded_prefill_call_counts()
    out = wrapped(x_second, mask=wrong_mask, cache=cache)
    mx.eval(out)
    after = banded_prefill_call_counts()

    assert after["composed_offset"] == before["composed_offset"] + 1
    assert after["mma_offset"] == before["mma_offset"]
    assert wrapped.banded_prefill_calls == 0


def test_deepseek_v4_banded_prefill_offset_indexer_mirror_is_cumulative(
    monkeypatch,
):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from moespresso.runtime.deepseek_v4.model import banded_prefill_call_counts

    monkeypatch.setenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", "1")
    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    manifest = _tiny_manifest()
    cfg = manifest["architecture"]["config"]
    cfg["compress_ratios"] = [4]
    cfg["index_topk"] = 4
    manifest["architecture"]["compress_ratios"] = [4]
    model = build_deepseek_v4_graph_from_manifest(manifest)
    attn = model.layers[0].self_attn
    wrapped = _BandedPrefillAttention(
        attn,
        mx=mx,
        dsv4_model=jang_dsv4,
        cache_cls=jang_dsv4.DeepseekV4Cache,
    )

    tokens, window = 14, 8
    cache = jang_dsv4.DeepseekV4Cache(window, compress_ratio=4)
    _seed_composed_first_chunk(attn, cache, tokens, window)

    x_second = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    mask_second = _shared_window_mask(tokens, tokens, window)
    before = banded_prefill_call_counts()
    # tokens // ratio == 3 stays under index_topk 4, but the cumulative
    # (offset + tokens) // ratio == 7 exceeds it, so the mirror must fall
    # closed to the composed original at offset.
    out = wrapped(x_second, mask=mask_second, cache=cache)
    mx.eval(out)
    after = banded_prefill_call_counts()

    assert after["composed_offset"] == before["composed_offset"] + 1
    assert after["mma_offset"] == before["mma_offset"]
    assert wrapped.banded_prefill_calls == 0


def test_deepseek_v4_banded_prefill_offset_skips_band_economics_gate(
    monkeypatch,
):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from moespresso.runtime.deepseek_v4.model import banded_prefill_call_counts
    from mlx_lm.models.cache import KVCache

    monkeypatch.setenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", "1")
    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    attn = model.layers[0].self_attn
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 1
    wrapped = model.layers[0].self_attn

    def fake_engine(mx_mod, **kwargs):
        q = kwargs["q"]
        return mx.zeros(
            (1, int(q.shape[1]), int(q.shape[2]), int(q.shape[3])),
            dtype=mx.float32,
        )

    monkeypatch.setattr(
        dsv4_patch, "_deepseek_v4_banded_mma_attention", fake_engine)
    monkeypatch.setattr(
        dsv4_patch,
        "_deepseek_v4_banded_mma_offset_ready",
        lambda *a, **k: True,
    )

    # tokens 10 sits at or under window + block == 12: composed at offset
    # zero (dense work plus padding), banded at offset (the composed
    # alternative reads window - 1 + tokens keys, so the band is never
    # worse and short trailing chunks stay on the route).
    tokens, window = 10, 8
    cache_zero = KVCache()
    before = banded_prefill_call_counts()
    out_zero = wrapped(
        mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1,
        mask=_shared_window_mask(tokens, 0, window),
        cache=cache_zero,
    )
    mx.eval(out_zero)
    assert banded_prefill_call_counts() == before
    assert wrapped.banded_prefill_calls == 0

    cache_offset = KVCache()
    _seed_composed_first_chunk(attn, cache_offset, 13, window)
    out_offset = wrapped(
        mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1,
        mask=_shared_window_mask(tokens, 13, window),
        cache=cache_offset,
    )
    mx.eval(out_offset)
    after = banded_prefill_call_counts()
    assert after["mma_offset"] == before["mma_offset"] + 1
    assert wrapped.banded_prefill_calls == 1


def test_deepseek_v4_banded_prefill_offset_plumbing_matches_composed(
    monkeypatch,
):
    import moespresso.runtime.deepseek_v4.model as dsv4_patch
    from mlx_lm.models.cache import KVCache

    monkeypatch.setenv("MOESPRESSO_DSV4_BANDED_PREFILL_OFFSET", "1")
    monkeypatch.setattr(dsv4_patch, "_DEEPSEEK_V4_BANDED_PREFILL_BLOCK", 4)
    model = build_deepseek_v4_graph_from_manifest(_tiny_manifest())
    attn = model.layers[0].self_attn
    assert _patch_deepseek_v4_banded_prefill_attention(model) == 1
    wrapped = model.layers[0].self_attn

    # An exact engine running the composed SDPA over the fetched history
    # isolates the wrapper plumbing: if the forward rope, the cache read,
    # or the inverse rope used a wrong position, the outputs would differ
    # from the composed original.
    def exact_engine(mx_mod, *, q, kv, pooled, sinks, window, ratio, scale,
                     pos0=0):
        del pooled, ratio
        tokens_in = int(q.shape[2])
        assert int(kv.shape[2]) == int(pos0) + tokens_in
        mask = _shared_window_mask(tokens_in, int(pos0), int(window))
        return jang_dsv4.scaled_dot_product_attention(
            q, kv, kv, cache=None, scale=scale, mask=mask,
            sinks=sinks.astype(q.dtype),
        )

    monkeypatch.setattr(
        dsv4_patch, "_deepseek_v4_banded_mma_attention", exact_engine)
    monkeypatch.setattr(
        dsv4_patch,
        "_deepseek_v4_banded_mma_offset_ready",
        lambda *a, **k: True,
    )

    tokens, window = 13, 8
    cache_reference = KVCache()
    cache_wrapped = KVCache()
    _seed_composed_first_chunk(attn, cache_reference, tokens, window)
    _seed_composed_first_chunk(attn, cache_wrapped, tokens, window)

    x_second = mx.random.normal((1, tokens, 64)).astype(mx.float32) * 0.1
    mask_second = _shared_window_mask(tokens, tokens, window)
    reference = attn(x_second, mask=mask_second, cache=cache_reference)
    got = wrapped(x_second, mask=mask_second, cache=cache_wrapped)
    mx.eval(reference, got)

    assert wrapped.banded_prefill_calls == 1
    np.testing.assert_array_equal(np.asarray(reference), np.asarray(got))
    np.testing.assert_array_equal(
        np.asarray(cache_reference.keys), np.asarray(cache_wrapped.keys))


def test_deepseek_v4_banded_mma_engine_fails_closed_without_kernel_contract(
    monkeypatch,
):
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_banded_mma_attention,
    )

    tokens = 6
    q = mx.zeros((1, 16, tokens, 32), dtype=mx.float32)
    kv = mx.zeros((1, 1, tokens, 32), dtype=mx.float32)
    sinks = mx.zeros((16,), dtype=mx.float32)

    # head_dim below 512 never reaches the kernel.
    assert _deepseek_v4_banded_mma_attention(
        mx, q=q, kv=kv, pooled=None, sinks=sinks,
        window=8, ratio=0, scale=32 ** -0.5,
    ) is None

    q512 = mx.zeros((1, 24, tokens, 512), dtype=mx.float32)
    kv512 = mx.zeros((1, 1, tokens, 512), dtype=mx.float32)
    sinks24 = mx.zeros((24,), dtype=mx.float32)
    # Head counts off the 16-head tile fail closed.
    assert _deepseek_v4_banded_mma_attention(
        mx, q=q512, kv=kv512, pooled=None, sinks=sinks24,
        window=8, ratio=0, scale=512 ** -0.5,
    ) is None

    q16 = mx.zeros((1, 16, tokens, 512), dtype=mx.float32)
    # A served scale other than rsqrt(head_dim) fails closed.
    assert _deepseek_v4_banded_mma_attention(
        mx, q=q16, kv=kv512, pooled=None, sinks=sinks,
        window=8, ratio=0, scale=0.5,
    ) is None
    # The consumer kernel kill switch closes the route.
    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "0")
    assert _deepseek_v4_banded_mma_attention(
        mx, q=q16, kv=kv512, pooled=None, sinks=sinks,
        window=8, ratio=0, scale=512 ** -0.5,
    ) is None
    # Integer routed ids are not a float pool; a pooled tensor outside the
    # float dtypes fails closed.
    monkeypatch.delenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", raising=False)
    assert _deepseek_v4_banded_mma_attention(
        mx, q=q16, kv=kv512,
        pooled=mx.zeros((1, 3, 512), dtype=mx.int32),
        sinks=sinks, window=8, ratio=128, scale=512 ** -0.5,
    ) is None


def test_deepseek_v4_banded_mma_engine_matches_banded_sdpa_math(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_banded_mma_attention,
        _deepseek_v4_banded_prefill_plan,
    )

    monkeypatch.setenv("MOESPRESSO_DSV4_R4_PREFILL_CONSUMER_MMA", "1")
    tokens, heads, dim = 200, 16, 512
    window, ratio, block = 64, 16, 64
    pooled_rows = tokens // ratio
    rng = np.random.default_rng(97)
    q = mx.array(
        rng.standard_normal((1, heads, tokens, dim), dtype=np.float32) * 0.1)
    kv = mx.array(
        rng.standard_normal((1, 1, tokens, dim), dtype=np.float32) * 0.1)
    pooled = mx.array(
        rng.standard_normal((1, pooled_rows, dim), dtype=np.float32) * 0.1)
    sinks = mx.array(rng.standard_normal((heads,), dtype=np.float32))
    scale = dim ** -0.5

    got = _deepseek_v4_banded_mma_attention(
        mx, q=q, kv=kv, pooled=pooled, sinks=sinks,
        window=window, ratio=ratio, scale=scale,
    )
    assert got is not None

    gather_idx, block_mask, blocks = _deepseek_v4_banded_prefill_plan(
        mx,
        tokens=tokens,
        pooled_rows=pooled_rows,
        window=window,
        ratio=ratio,
        block=block,
    )
    full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
    kv_bands = full_kv[0, 0][gather_idx][:, None]
    padded = blocks * block
    qq = q
    if padded != tokens:
        qq = mx.concatenate(
            [qq, mx.zeros((1, heads, padded - tokens, dim), dtype=q.dtype)],
            axis=2,
        )
    q_blocks = qq[0].reshape(heads, blocks, block, dim).transpose(1, 0, 2, 3)
    expected = jang_dsv4.scaled_dot_product_attention(
        q_blocks, kv_bands, kv_bands,
        cache=None, scale=scale, mask=block_mask, sinks=sinks,
    )
    expected = expected.transpose(1, 0, 2, 3).reshape(1, heads, padded, dim)
    if padded != tokens:
        expected = expected[:, :, :tokens]
    mx.eval(got, expected)

    # The mma route stages operands as half with float32 accumulation, a
    # numerically valid variant of the float32 SDPA composition.
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected, dtype=np.float32),
        rtol=5.0e-3,
        atol=5.0e-3,
    )

    # bfloat16 queries round once to half, matching a pre-rounded call.
    got_bf16 = _deepseek_v4_banded_mma_attention(
        mx, q=q.astype(mx.bfloat16), kv=kv, pooled=pooled, sinks=sinks,
        window=window, ratio=ratio, scale=scale,
    )
    got_pre = _deepseek_v4_banded_mma_attention(
        mx, q=q.astype(mx.bfloat16).astype(mx.float16), kv=kv,
        pooled=pooled, sinks=sinks,
        window=window, ratio=ratio, scale=scale,
    )
    mx.eval(got_bf16, got_pre)
    np.testing.assert_array_equal(np.asarray(got_bf16), np.asarray(got_pre))


def test_deepseek_v4_served_indexer_applies_hadamard_fp4_qat_before_scores(
    monkeypatch,
    tmp_path,
):
    # Pin the composed decode chain; the fused pre-top-k kernel and the
    # decode QAT kernel routing have their own parity coverage in
    # test_deepseek_v4_indexer_score_kernel.py.
    from moespresso.runtime.deepseek_v4 import indexer_score_kernel

    monkeypatch.setattr(indexer_score_kernel, "_ENABLED", False)
    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_DECODE_QAT_KERNEL", "0")
    manifest = _tiny_manifest()
    cfg = manifest["architecture"]["config"]
    cfg["compress_ratios"] = [4]
    cfg["index_n_heads"] = 1
    cfg["index_head_dim"] = 128
    cfg["index_topk"] = 1
    manifest["architecture"]["compress_ratios"] = [4]

    model = build_deepseek_v4_graph_from_manifest(manifest)
    assert _patch_deepseek_v4_indexer_score_contract(model) == 1
    indexer = model.layers[0].self_attn.indexer

    class FakeCompressor:
        def __call__(self, x, rope, cache, start_pos, state_key="indexer_state"):
            del x, rope, cache, start_pos, state_key
            return mx.ones((1, 1, cfg["index_head_dim"]), dtype=mx.float32)

    class FakeLinear:
        def __init__(self, width):
            self.width = width

        def __call__(self, x):
            return mx.ones((x.shape[0], x.shape[1], self.width), dtype=mx.float32)

    class IdentityRope:
        dims = cfg["index_head_dim"]

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            return x

    captured = {}
    real_argpartition = jang_dsv4.mx.argpartition

    def spy_argpartition(values, kth, axis=-1):
        captured["scores"] = np.array(-values)
        return real_argpartition(values, kth=kth, axis=axis)

    indexer._original.compressor = FakeCompressor()
    indexer._original.wq_b = FakeLinear(cfg["index_n_heads"] * cfg["index_head_dim"])
    indexer._original.weights_proj = FakeLinear(cfg["index_n_heads"])
    monkeypatch.setattr(jang_dsv4.mx, "argpartition", spy_argpartition)
    dump_prefix = tmp_path / "indexer"
    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_DUMP_PREFIX", str(dump_prefix))
    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_DUMP_LAYER", "0")
    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_DUMP_POS", "3")

    topk = indexer(
        mx.zeros((1, 1, cfg["hidden_size"]), dtype=mx.float32),
        mx.zeros((1, 1, cfg["q_lora_rank"]), dtype=mx.float32),
        IdentityRope(),
        IdentityRope(),
        None,
        3,
    )
    mx.eval(topk)

    expected = 144.0 / math.sqrt(128.0)
    np.testing.assert_allclose(
        captured["scores"],
        np.array([[[expected]]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.array(topk).reshape(-1).tolist() == [0]
    np.testing.assert_array_equal(
        np.fromfile(f"{dump_prefix}_indexer_topk-0_pos3.i32", dtype=np.int32),
        np.array([0], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.fromfile(f"{dump_prefix}_indexer_scores-0_pos3.bin", dtype=np.float32),
        np.array([expected], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    stats = deepseek_v4_indexer_layer_stats(model)
    assert stats == [{
        "layer": 0,
        "compress_ratio": 4,
        "indexer_score_contract_calls": 1,
        "indexer_score_contract_tokens": 1,
        "indexer_score_contract_pooled_rows": 1,
        "indexer_score_contract_topk_rows": 1,
        "indexer_score_contract_score_elements": 1,
        "indexer_score_contract_qat_elements": 256,
        "indexer_score_contract_cached_pooled_rows": 0,
        "indexer_score_contract_new_qat_pooled_rows": 1,
        "indexer_score_contract_fused_score_calls": 0,
        "indexer_score_contract_decode_qat_kernel_calls": 0,
        "indexer_score_contract_fixed_state_calls": 0,
        "indexer_score_contract_score_tail_kernel_calls": 0,
    }]


def test_deepseek_v4_served_indexer_reuses_cached_qat_pool(monkeypatch):
    # Pin the composed decode chain so the monkeypatched QAT observes the
    # query roundtrip; the fused kernel applies the query QAT in-kernel and
    # the decode QAT kernel routing bypasses the composed helper.
    from moespresso.runtime.deepseek_v4 import indexer_score_kernel

    monkeypatch.setattr(indexer_score_kernel, "_ENABLED", False)
    monkeypatch.setenv("MOESPRESSO_DSV4_INDEXER_DECODE_QAT_KERNEL", "0")
    manifest = _tiny_manifest()
    cfg = manifest["architecture"]["config"]
    cfg["compress_ratios"] = [4]
    cfg["index_n_heads"] = 1
    cfg["index_head_dim"] = 128
    cfg["index_topk"] = 2
    manifest["architecture"]["compress_ratios"] = [4]

    model = build_deepseek_v4_graph_from_manifest(manifest)
    assert _patch_deepseek_v4_indexer_score_contract(model) == 1
    indexer = model.layers[0].self_attn.indexer

    class FakeCompressor:
        def __init__(self):
            self.rows = 3

        def __call__(self, x, rope, cache, start_pos, state_key="indexer_state"):
            del x, rope, cache, start_pos, state_key
            values = np.arange(1, self.rows + 1, dtype=np.float32).reshape(
                1, self.rows, 1)
            return mx.array(np.repeat(values, cfg["index_head_dim"], axis=-1))

    class FakeLinear:
        def __init__(self, width):
            self.width = width

        def __call__(self, x):
            return mx.ones((x.shape[0], x.shape[1], self.width), dtype=mx.float32)

    class IdentityRope:
        dims = cfg["index_head_dim"]

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            return x

    class FakeCache:
        def __init__(self):
            self.indexer_state = {}

    qat_shapes = []

    def fake_qat(_mx, x):
        qat_shapes.append(tuple(x.shape))
        return x

    compressor = FakeCompressor()
    indexer._original.compressor = compressor
    indexer._original.wq_b = FakeLinear(cfg["index_n_heads"] * cfg["index_head_dim"])
    indexer._original.weights_proj = FakeLinear(cfg["index_n_heads"])
    monkeypatch.setattr(
        "moespresso.runtime.deepseek_v4.model._dsv4_indexer_qat",
        fake_qat,
    )
    cache = FakeCache()
    x = mx.zeros((1, 1, cfg["hidden_size"]), dtype=mx.float32)
    q_residual = mx.zeros((1, 1, cfg["q_lora_rank"]), dtype=mx.float32)

    mx.eval(indexer(x, q_residual, IdentityRope(), IdentityRope(), cache, 2048))
    compressor.rows = 4
    mx.eval(indexer(x, q_residual, IdentityRope(), IdentityRope(), cache, 2052))

    # Per step: the pool QAT runs first (full pool, then only the one new
    # tail row on the cached second step), then the query QAT.
    assert qat_shapes == [
        (1, 3, 128),
        (1, 1, 1, 128),
        (1, 1, 128),
        (1, 1, 1, 128),
    ]
    stats = deepseek_v4_indexer_layer_stats(model)[0]
    assert stats["indexer_score_contract_cached_pooled_rows"] == 3
    assert stats["indexer_score_contract_new_qat_pooled_rows"] == 4
    assert stats["indexer_score_contract_qat_elements"] == (
        128 + 3 * 128 + 128 + 1 * 128
    )


def test_deepseek_v4_attention_shape_stats_counts_served_sdpa(monkeypatch):
    original_attention = jang_dsv4.scaled_dot_product_attention
    had_saved = hasattr(jang_dsv4, "_moespresso_original_scaled_dot_product_attention")
    saved_attention = getattr(
        jang_dsv4,
        "_moespresso_original_scaled_dot_product_attention",
        None,
    )

    def fake_attention(q, k, v, *args, **kwargs):
        del k, v, args, kwargs
        return mx.zeros(q.shape, dtype=q.dtype)

    class FakeAttention:
        compress_ratio = 4

        def __call__(self, x, mask=None, cache=None):
            del x, mask, cache
            q = mx.zeros((1, 2, 3, 4), dtype=mx.float32)
            k = mx.zeros((1, 1, 5, 4), dtype=mx.float32)
            return jang_dsv4.scaled_dot_product_attention(q, k, k, scale=1.0)

    layer = type("Layer", (), {"self_attn": FakeAttention()})()
    model = type("Model", (), {"model": type("Inner", (), {"layers": [layer]})()})()

    try:
        if hasattr(jang_dsv4, "_moespresso_original_scaled_dot_product_attention"):
            delattr(jang_dsv4, "_moespresso_original_scaled_dot_product_attention")
        jang_dsv4.scaled_dot_product_attention = fake_attention
        assert _patch_deepseek_v4_attention_fp16_qkv(model) is True
        assert _patch_deepseek_v4_attention_shape_stats(model) == 1
        out = model.model.layers[0].self_attn(
            mx.zeros((1, 3, 8), dtype=mx.float32))
        mx.eval(out)

        assert deepseek_v4_attention_layer_stats(model) == [{
            "layer": 0,
            "compress_ratio": 4,
            "attention_sdpa_calls": 1,
            "attention_sdpa_tokens": 3,
            "attention_sdpa_key_rows": 15,
            "attention_sdpa_score_elements": 30,
            "attention_sdpa_value_elements": 60,
            "attention_sdpa_max_key_rows": 5,
            "fused_decode_attention_calls": 0,
            "fused_decode_composed_tail_calls": 0,
            "fixed_state_decode_layers": 0,
        }]
    finally:
        jang_dsv4.scaled_dot_product_attention = original_attention
        if had_saved:
            jang_dsv4._moespresso_original_scaled_dot_product_attention = (
                saved_attention
            )
        elif hasattr(jang_dsv4, "_moespresso_original_scaled_dot_product_attention"):
            delattr(jang_dsv4, "_moespresso_original_scaled_dot_product_attention")


def test_deepseek_v4_attention_compressor_fp8_roundtrip_wraps_attention_only():
    class FakeCompressor:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def __call__(self, x, rope, cache, start_pos, state_key="compressor_state"):
            del x, rope, cache, start_pos, state_key
            self.calls += 1
            return mx.array(self.value, dtype=mx.float32)

    values = np.zeros((1, 1, 512), dtype=np.float32)
    values[..., :448] = 0.12345
    values[..., 448:] = 0.333
    attn_compressor = FakeCompressor(values)
    indexer_compressor = FakeCompressor(values)
    attn = type("Attention", (), {})()
    attn.compressor = attn_compressor
    indexer = type("Indexer", (), {})()
    indexer.compressor = indexer_compressor
    attn.indexer = indexer
    layer = type("Layer", (), {"self_attn": attn})()
    inner = type("Inner", (), {"layers": [layer]})()
    model = type("Model", (), {"model": inner})()

    assert _patch_deepseek_v4_attention_compressor_fp8_kv(model) == 1
    assert _patch_deepseek_v4_attention_compressor_fp8_kv(model) == 0

    rounded = np.asarray(attn.compressor(None, None, None, 0))
    unwrapped_indexer = np.asarray(indexer.compressor(None, None, None, 0))

    assert attn_compressor.calls == 1
    assert indexer_compressor.calls == 1
    assert not np.allclose(rounded[..., :448], unwrapped_indexer[..., :448])
    np.testing.assert_allclose(rounded[..., 448:], unwrapped_indexer[..., 448:])


def test_deepseek_v4_attention_compressor_fp8_caches_rounded_prefix(monkeypatch):
    class FakeCompressor:
        def __init__(self, values):
            self.values = list(values)

        def __call__(self, x, rope, cache, start_pos, state_key="compressor_state"):
            del x, rope, cache, start_pos, state_key
            return self.values.pop(0)

    first = mx.zeros((1, 2, 512), dtype=mx.float32)
    tail = mx.ones((1, 1, 512), dtype=mx.float32)
    second = mx.concatenate([first, tail], axis=1)
    calls = []

    def fake_roundtrip(x):
        calls.append(tuple(x.shape))
        return x + mx.array(float(len(calls)), dtype=x.dtype)

    import moespresso.runtime.deepseek_v4.model as dsv4_patch

    monkeypatch.setattr(dsv4_patch, "_deepseek_v4_fp8_kv_roundtrip", fake_roundtrip)
    cache = type("Cache", (), {"compressor_state": {}})()
    compressor = _AttentionCompressorFp8KV(FakeCompressor([first, second]))

    out_first = compressor(None, None, cache, 0)
    out_second = compressor(None, None, cache, 4)
    mx.eval(out_first, out_second)

    assert calls == [(1, 2, 512), (1, 1, 512)]
    np.testing.assert_allclose(
        np.asarray(out_second[:, :2, :]),
        np.asarray(out_first),
    )
    np.testing.assert_allclose(
        np.asarray(out_second[:, 2:, :]),
        np.asarray(tail + 2.0),
    )
    assert cache.compressor_state["pooled_fp8_rows"] == 3


def test_deepseek_v4_attention_compressor_primes_indexer_state_before_topk():
    class FakeCompressor:
        def __init__(self, value):
            self.value = value
            self.calls = []

        def __call__(self, x, rope, cache, start_pos, state_key="compressor_state"):
            self.calls.append({
                "cache": cache,
                "start_pos": start_pos,
                "state_key": state_key,
                "x": x,
                "rope": rope,
            })
            return mx.array(self.value, dtype=mx.float32)

    attention_pool = np.zeros((1, 512, 512), dtype=np.float32)
    indexer_pool = np.zeros((1, 512, 128), dtype=np.float32)
    attn_compressor = FakeCompressor(attention_pool)
    indexer_compressor = FakeCompressor(indexer_pool)
    attn = type("Attention", (), {})()
    attn.compressor = attn_compressor
    indexer = type("Indexer", (), {"index_topk": 512})()
    indexer.compressor = indexer_compressor
    attn.indexer = indexer
    layer = type("Layer", (), {"self_attn": attn})()
    inner = type("Inner", (), {"layers": [layer]})()
    model = type("Model", (), {"model": inner})()
    cache = object()
    x = object()
    rope = object()

    assert _patch_deepseek_v4_attention_compressor_fp8_kv(model) == 1
    attn.compressor(x, rope, cache, 0)

    assert len(attn_compressor.calls) == 1
    assert len(indexer_compressor.calls) == 1
    assert indexer_compressor.calls[0] == {
        "cache": cache,
        "start_pos": 0,
        "state_key": "indexer_state",
        "x": x,
        "rope": rope,
    }


def test_deepseek_v4_attention_compressor_does_not_double_update_indexer_after_topk():
    class FakeCompressor:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def __call__(self, x, rope, cache, start_pos, state_key="compressor_state"):
            del x, rope, cache, start_pos, state_key
            self.calls += 1
            return mx.array(self.value, dtype=mx.float32)

    attention_pool = np.zeros((1, 513, 512), dtype=np.float32)
    indexer_pool = np.zeros((1, 513, 128), dtype=np.float32)
    attn = type("Attention", (), {})()
    attn.compressor = FakeCompressor(attention_pool)
    indexer = type("Indexer", (), {"index_topk": 512})()
    indexer.compressor = FakeCompressor(indexer_pool)
    attn.indexer = indexer
    layer = type("Layer", (), {"self_attn": attn})()
    inner = type("Inner", (), {"layers": [layer]})()
    model = type("Model", (), {"model": inner})()

    assert _patch_deepseek_v4_attention_compressor_fp8_kv(model) == 1
    attn.compressor(None, None, object(), 2048)

    assert attn.compressor._original.calls == 1
    assert indexer.compressor.calls == 0


def test_deepseek_v4_sidecar_builds_tiny_empty_skeleton():
    manifest = _tiny_manifest()
    manifest["tensors"] = [{
        "source_name": "layers.0.attn.wq_a.weight",
        "format": "affine",
        "format_params": {"bits": 4, "group_size": 64},
    }]
    config_json, _jang_config = build_sidecars(manifest)

    model, config = _load_empty_deepseek_v4_skeleton(".", model_config=config_json)

    assert config["model_type"] == "deepseek_v4"
    assert len(model.layers) == 1
    assert model.args.hidden_size == 64


def _wrapped_cache_model(*, ratio: int = 4, sliding_window: int = 8):
    """Minimal model surface for make_cache with the wrapper patches installed."""

    class _Attn:
        def __init__(self, compress_ratio):
            self.compress_ratio = compress_ratio

    class _Layer:
        def __init__(self, compress_ratio):
            self.self_attn = _Attn(compress_ratio)

    class _Inner:
        def __init__(self):
            self.layers = [_Layer(ratio)]

    class _Model:
        def __init__(self):
            self.args = type("Args", (), {"sliding_window": sliding_window})()
            self.model = _Inner()

    model = _Model()
    _patch_deepseek_v4_required_attention_cache(
        model, deepseek_cache_cls=jang_dsv4.DeepseekV4Cache)
    return model


def _seed_pool_and_aux(cache, rng, *, rows: int = 5):
    """Populate pool rows and derived aux entries through the stock surface."""
    pooled = mx.array(rng.standard_normal((1, rows, 16), dtype=np.float32))
    cache.update_pool(pooled, "compressor_state")
    cache.update_pool(pooled, "indexer_state")
    cache.compressor_state["pooled_fp8"] = cache.compressor_state["pooled"]
    cache.compressor_state["pooled_fp8_rows"] = rows
    cache.indexer_state["pooled_qat"] = cache.indexer_state["pooled"]
    cache.indexer_state["pooled_qat_rows"] = rows


def test_cache_wrapper_fork_isolation_after_deepcopy(monkeypatch):
    """A deepcopied wrapped cache must never mutate its source.

    The shipped wrappers replace `update_and_fetch` and `trim` with bound
    methods reading originals from an instance dict, so the deepcopy memo
    rebinds them to the copy. Storing the originals in closures instead
    left the copy driving the source cache's offset, pools, and aux state.
    """
    monkeypatch.setenv("MOESPRESSO_DSV4_DECODE_FIXED_STATE", "0")
    cache = _wrapped_cache_model().make_cache()[0]
    assert cache._moespresso_dsv4_fp8_kv_cache
    assert cache._moespresso_dsv4_compressed_pool_aux_trim_clear

    rng = np.random.default_rng(61)
    prefill = mx.array(rng.standard_normal((1, 1, 6, 512), dtype=np.float32))
    cache.update_and_fetch(prefill, prefill)
    _seed_pool_and_aux(cache, rng)

    fork = copy.deepcopy(cache)
    src_offset = int(cache.offset)
    src_keys = np.asarray(cache.local.keys)
    src_pool = np.asarray(cache.compressor_state["pooled"])

    step = mx.array(rng.standard_normal((1, 1, 1, 512), dtype=np.float32))
    fork.update_and_fetch(step, step)
    assert int(fork.offset) == src_offset + 1
    assert int(cache.offset) == src_offset
    assert np.array_equal(np.asarray(cache.local.keys), src_keys)

    fork.trim(4)
    assert "pooled_fp8" not in fork.compressor_state
    assert "pooled_qat" not in fork.indexer_state
    assert int(fork.compressor_state["pooled"].shape[1]) == 4
    assert int(cache.offset) == src_offset
    assert np.array_equal(np.asarray(cache.compressor_state["pooled"]), src_pool)
    assert cache.compressor_state.get("pooled_fp8_rows") == 5
    assert cache.indexer_state.get("pooled_qat_rows") == 5

    # Driving the source must not perturb the fork either.
    fork_offset = int(fork.offset)
    fork_keys = np.asarray(fork.local.keys)
    fork_pool = np.asarray(fork.compressor_state["pooled"])
    cache.update_and_fetch(step, step)
    cache.trim(4)
    assert int(fork.offset) == fork_offset
    assert np.array_equal(np.asarray(fork.local.keys), fork_keys)
    assert np.array_equal(np.asarray(fork.compressor_state["pooled"]), fork_pool)
    # Identical drive sequences leave source and fork in identical state.
    assert int(cache.offset) == fork_offset
    assert np.array_equal(np.asarray(cache.compressor_state["pooled"]), fork_pool)
    assert "pooled_fp8" not in cache.compressor_state


def test_cache_wrapper_fork_isolation_with_fixed_decode_state(monkeypatch):
    """Post-fork trim isolation through the full shipped wrapper stack.

    With fixed decode state installed and no live branches, `trim` falls
    back through the fixed-state originals dict to the aux-clear wrapper
    and then to the stock trim. Every hop must land on the copy.
    """
    monkeypatch.delenv("MOESPRESSO_DSV4_DECODE_FIXED_STATE", raising=False)
    cache = _wrapped_cache_model().make_cache()[0]
    assert cache._moespresso_dsv4_fixed_decode_state

    rng = np.random.default_rng(67)
    prefill = mx.array(rng.standard_normal((1, 1, 6, 512), dtype=np.float32))
    cache.update_and_fetch(prefill, prefill)
    _seed_pool_and_aux(cache, rng)

    fork = copy.deepcopy(cache)
    src_offset = int(cache.offset)
    src_pool = np.asarray(cache.compressor_state["pooled"])

    fork.trim(4)
    assert int(fork.offset) == src_offset - 4
    assert int(fork.compressor_state["pooled"].shape[1]) == 4
    assert "pooled_fp8" not in fork.compressor_state
    assert "pooled_qat" not in fork.indexer_state
    assert int(cache.offset) == src_offset
    assert np.array_equal(np.asarray(cache.compressor_state["pooled"]), src_pool)
    assert cache.compressor_state.get("pooled_fp8_rows") == 5
    assert cache.indexer_state.get("pooled_qat_rows") == 5

    step = mx.array(rng.standard_normal((1, 1, 1, 512), dtype=np.float32))
    fork.update_and_fetch(step, step)
    assert int(fork.offset) == src_offset - 3
    assert int(cache.offset) == src_offset


def test_cache_store_fork_leaves_stored_ds4_entry_untouched(monkeypatch):
    """The prefix-store fork path must not corrupt the stored DS4 entry.

    For a shorter follow-up prompt against a longer trimmable entry,
    `PromptCacheStore.fetch_nearest_cache` deepcopies the stored cache
    list and trims the copy before returning it. Both the trim and later
    decode steps on the returned fork must leave the stored entry
    byte-identical.
    """
    from moespresso.runtime.prefix_cache import make_prompt_cache_store

    monkeypatch.delenv("MOESPRESSO_DSV4_DECODE_FIXED_STATE", raising=False)
    cache_list = _wrapped_cache_model(sliding_window=32).make_cache()
    cache = cache_list[0]

    rng = np.random.default_rng(71)
    for _ in range(10):
        row = mx.array(rng.standard_normal((1, 1, 1, 512), dtype=np.float32))
        cache.update_and_fetch(row, row)

    store = make_prompt_cache_store(max_size=4)
    key = ("pkg", "render", "raw", 64, 0, "deepseek_v4_composite")
    tokens = list(range(10))
    store.insert_cache(key, tokens, cache_list)
    # The store keeps the inserted list by reference, so state written
    # after insert lands on the stored entry. Insert with the aux state
    # already present is covered by
    # test_cache_store_insert_accepts_post_decode_aux_state.
    _seed_pool_and_aux(cache, rng, rows=2)

    src_offset = int(cache.offset)
    src_keys = np.asarray(cache.local.keys)
    src_pool = np.asarray(cache.compressor_state["pooled"])

    # A 6-token common prefix against the 10-token entry takes the
    # longer-entry path: deepcopy plus trim(4) on the copy.
    fork_list, suffix = store.fetch_nearest_cache(key, tokens[:6] + [99])
    assert suffix == [99]
    fork = fork_list[0]
    assert int(fork.offset) == 6
    assert int(fork.compressor_state["pooled"].shape[1]) == 1
    assert "pooled_fp8" not in fork.compressor_state
    assert "pooled_qat" not in fork.indexer_state

    step = mx.array(rng.standard_normal((1, 1, 1, 512), dtype=np.float32))
    fork.update_and_fetch(step, step)
    assert int(fork.offset) == 7

    assert int(cache.offset) == src_offset
    assert np.array_equal(np.asarray(cache.local.keys), src_keys)
    assert np.array_equal(np.asarray(cache.compressor_state["pooled"]), src_pool)
    assert cache.compressor_state.get("pooled_fp8_rows") == 2
    assert cache.indexer_state.get("pooled_qat_rows") == 2


@pytest.mark.parametrize("fixed_state", ["0", "1"])
def test_cache_store_insert_accepts_post_decode_aux_state(monkeypatch, fixed_state):
    """`insert_cache` must size a post-decode DS4 cache without raising.

    Post-decode, the compressor and indexer state dicts hold the aux row
    counts (`pooled_fp8_rows`, `pooled_qat_rows`) as plain ints next to
    the array entries. The stock `DeepseekV4Cache.nbytes` called
    `.nbytes` on every non-None dict value, so the serve-path insert
    raised AttributeError before the entry landed and multi-turn prefix
    reuse never happened. The insert must succeed, size the entry from
    array bytes, and leave it retrievable and forkable.
    """
    from moespresso.runtime.prefix_cache import make_prompt_cache_store

    monkeypatch.setenv("MOESPRESSO_DSV4_DECODE_FIXED_STATE", fixed_state)
    cache_list = _wrapped_cache_model(sliding_window=32).make_cache()
    cache = cache_list[0]

    rng = np.random.default_rng(73)
    for _ in range(10):
        row = mx.array(rng.standard_normal((1, 1, 1, 512), dtype=np.float32))
        cache.update_and_fetch(row, row)
    _seed_pool_and_aux(cache, rng, rows=2)

    store = make_prompt_cache_store(max_size=4)
    key = ("pkg", "render", "raw", 64, 0, "deepseek_v4_composite")
    tokens = list(range(10))
    store.insert_cache(key, tokens, cache_list)
    assert isinstance(store.nbytes, int)
    assert store.nbytes > 0

    # Exact-token fetch moves the stored entry out (no copy) with the aux
    # state intact; the serve path reinserts it after generation, replayed
    # here so the longer-entry fork below has an entry to fork from.
    exact_list, exact_suffix = store.fetch_nearest_cache(key, tokens)
    assert exact_suffix == []
    assert exact_list is cache_list
    exact = exact_list[0]
    assert int(exact.offset) == 10
    assert exact.compressor_state.get("pooled_fp8_rows") == 2
    assert exact.indexer_state.get("pooled_qat_rows") == 2
    assert len(store) == 0
    store.insert_cache(key, tokens, cache_list)

    src_offset = int(cache.offset)
    src_keys = np.asarray(cache.local.keys)
    src_pool = np.asarray(cache.compressor_state["pooled"])

    # A 6-token common prefix against the 10-token entry takes the
    # longer-entry path: deepcopy plus trim(4) on the copy.
    fork_list, suffix = store.fetch_nearest_cache(key, tokens[:6] + [99])
    assert suffix == [99]
    fork = fork_list[0]
    assert int(fork.offset) == 6
    assert "pooled_fp8" not in fork.compressor_state
    assert "pooled_qat" not in fork.indexer_state

    step = mx.array(rng.standard_normal((1, 1, 1, 512), dtype=np.float32))
    fork.update_and_fetch(step, step)
    assert int(fork.offset) == 7

    assert int(cache.offset) == src_offset
    assert np.array_equal(np.asarray(cache.local.keys), src_keys)
    assert np.array_equal(np.asarray(cache.compressor_state["pooled"]), src_pool)
    assert cache.compressor_state.get("pooled_fp8_rows") == 2
    assert cache.indexer_state.get("pooled_qat_rows") == 2
