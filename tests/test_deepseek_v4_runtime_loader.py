from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import moespresso.runtime.deepseek_v4.model as dsv4_runtime
from moespresso.runtime.deepseek_v4.model import (
    DeepseekV4RuntimeLoadError,
    _install_deepseek_v4_pooled_bundles,
    _load_empty_deepseek_v4_skeleton,
    _load_deepseek_v4_regular_weights,
    _patch_deepseek_v4_affine_wo_fp32,
    _patch_deepseek_v4_attention_fp16_qkv,
    _patch_deepseek_v4_compressor_ape_float16,
    _patch_deepseek_v4_hc_post_float32,
    _patch_deepseek_v4_kquant_grouped_output_projection,
    _patch_deepseek_v4_kquant_lm_head,
    _patch_deepseek_v4_q8_ffn_hc_post,
    _patch_deepseek_v4_q8_hc_post,
    _patch_deepseek_v4_required_attention_cache,
    _validate_deepseek_v4_router_gate_dtypes,
    load_deepseek_v4_package_model,
)


class _Model:
    def __init__(self):
        self.loaded = []

    def eval(self):
        self.evaluated = True

    def sanitize(self, weights):
        return {f"sanitized.{k}": v for k, v in weights.items()}

    def load_weights(self, weights, *, strict):
        self.loaded.append((dict(weights), strict))


class _Layer:
    def _hc_post(self, x, residual, post, comb):
        del residual, post, comb
        return x


class _LayerContainer:
    def __init__(self, layers):
        self.layers = layers


class _WrappedModel:
    def __init__(self, layers):
        self.model = _LayerContainer(layers)


class _Args:
    sliding_window = 128


class _SelfAttention:
    def __init__(self, compress_ratio):
        self.compress_ratio = compress_ratio


class _KQuantModule:
    mode = "kquant"
    kquant_type = "q8_0"
    group_size = 32
    bits = 8
    biases = None

    def __init__(self, weight, scales, bias=None):
        self._params = {"weight": weight, "scales": scales}
        if bias is not None:
            self._params["bias"] = bias

    def __getitem__(self, key):
        return self._params[key]

    def __contains__(self, key):
        return key in self._params


class _KQuantAttention:
    def __init__(self, wo_a, wo_b):
        self.n_heads = 2
        self.head_dim = 2
        self.o_groups = 2
        self.o_lora_rank = 1
        self.wo_a = wo_a
        self.wo_b = wo_b


class _Identity:
    def __call__(self, x):
        return x


class _IdentityMlp:
    def __call__(self, x, *, input_ids=None):
        del input_ids
        return x


class _Q8HcAttention(_KQuantAttention):
    def __init__(self, wo_a, wo_b, projection):
        super().__init__(wo_a, wo_b)
        self.projection = projection
        self.raise_on_call = False

    def __call__(self, x, *, mask=None, cache=None):
        del x, mask, cache
        if self.raise_on_call:
            raise RuntimeError("attention failed")
        return self.wo_b(self.projection)


class _Q8HcDecoderLayer:
    def __init__(self, attn, mx):
        self.layer_id = 0
        self.self_attn = attn
        self.input_layernorm = _Identity()
        self.post_attention_layernorm = _Identity()
        self.mlp = _IdentityMlp()
        self.hc_attn_fn = None
        self.hc_attn_scale = None
        self.hc_attn_base = None
        self.hc_ffn_fn = None
        self.hc_ffn_scale = None
        self.hc_ffn_base = None
        self._mx = mx
        self.hc_post_calls = 0
        self.original_calls = 0

    def _hc_pre(self, x, fn, scale, base):
        del fn, scale, base
        batch, length, hc, _hidden = (int(dim) for dim in x.shape)
        post = self._mx.ones((batch, length, hc), dtype=self._mx.float32)
        comb = self._mx.broadcast_to(
            self._mx.eye(hc, dtype=self._mx.float32),
            (batch, length, hc, hc),
        )
        return x[..., 0, :], post, comb

    def _hc_post(self, x, residual, post, comb):
        del post, comb
        self.hc_post_calls += 1
        return residual + x[..., None, :]

    def __call__(self, x, mask=None, cache=None, input_ids=None):
        del mask, cache, input_ids
        self.original_calls += 1
        return x


class _AttentionLayer:
    def __init__(self, attn):
        self.self_attn = attn


class _Gate:
    def __init__(self, weight):
        self.weight = weight


class _MLP:
    def __init__(self, gate):
        self.gate = gate


class _RouterLayer:
    def __init__(self, gate):
        self.mlp = _MLP(gate)


class _Compressor:
    def __init__(self, ape):
        self.ape = ape


class _Indexer:
    def __init__(self, compressor):
        self.compressor = compressor


class _AttentionWithCompressors:
    def __init__(self, ape, indexer_ape):
        self.compressor = _Compressor(ape)
        self.indexer = _Indexer(_Compressor(indexer_ape))


class _ApeLayer:
    def __init__(self, ape, indexer_ape):
        self.self_attn = _AttentionWithCompressors(ape, indexer_ape)


class _CacheLayer:
    def __init__(self, compress_ratio):
        self.self_attn = _SelfAttention(compress_ratio)


class _CacheWrappedModel:
    def __init__(self, ratios):
        self.args = _Args()
        self.model = _LayerContainer([_CacheLayer(ratio) for ratio in ratios])


class _FakeKVCache:
    def update_and_fetch(self, keys, values):
        return keys, values


class _FakeDeepseekV4Cache:
    def __init__(self, sliding_window, *, compress_ratio):
        self.sliding_window = sliding_window
        self.compress_ratio = compress_ratio
        self.compressor_state = {}
        self.indexer_state = {}

    def update_and_fetch(self, keys, values):
        return keys, values

    def trim(self, n):
        return n


class _ExpertIndex:
    def layers_indexed(self):
        return [0]

    def has_projection(self, *, layer, projection):
        return layer == 0 and projection in {"gate_proj", "up_proj", "down_proj"}

    def bits(self, *, layer, projection):
        return {"gate_proj": 4, "up_proj": 6, "down_proj": 4}[projection]


def _manifest():
    return {
        "architecture": {"family": "deepseek_v4_flash"},
        "required_ops": ["affine_dequant", "mxfp4_dequant"],
        "tensors": [{"format": "affine"}, {"format": "mxfp4"}],
    }


def _kquant_manifest():
    man = _manifest()
    man["required_ops"] = ["affine_dequant", "kquant_dequant"]
    man["tensors"] = [
        {"format": "affine"},
        {
            "source_name": "layers.0.ffn.experts",
            "kind": "expert",
            "format": "kquant",
            "format_params": {"kquant_codec": "q2_k"},
            "module_weight_key": "model.layers.0.mlp.switch_mlp.down_proj.weight",
        },
    ]
    return man


def _iqk_expert_manifest():
    man = _manifest()
    man["required_ops"] = ["affine_dequant", "iqk_dequant"]
    man["tensors"] = [
        {"format": "affine"},
        {
            "source_name": "layers.0.ffn.experts",
            "kind": "expert",
            "format": "iqk",
            "format_params": {
                "iqk_codec": "iq2_ks",
                "layout": "iqk_relayout",
            },
        },
    ]
    return man


def _dense_kquant_manifest():
    man = _manifest()
    man["required_ops"] = ["affine_dequant", "kquant_dequant"]
    man["tensors"] = [
        {"format": "affine"},
        {
            "source_name": "layers.0.attn.wq_a.weight",
            "kind": "affine",
            "format": "kquant",
            "format_params": {"kquant_codec": "q8_0"},
            "module_weight_key": "model.layers.0.self_attn.wq_a.weight",
        },
    ]
    return man


def _routed_tq_manifest():
    man = _manifest()
    man["tensors"] = [
        {"format": "affine"},
        {
            "source_name": "layers.0.ffn.experts.gate",
            "kind": "expert",
            "format": "mxfp4",
            "format_params": {"bits": 2},
        },
    ]
    return man


def _touch_shard(package_dir: Path):
    (package_dir / "model-00001-of-00001.safetensors").write_bytes(b"")


def test_deepseek_v4_regular_loader_skips_only_expert_bundles(tmp_path):
    _touch_shard(tmp_path)
    model = _Model()

    loaded, skipped = _load_deepseek_v4_regular_weights(
        model,
        tmp_path,
        load_shard_fn=lambda _path: {
            "embed.weight": "E",
            "layers.0.ffn.experts.tq_bundle": "BUNDLE",
        },
    )

    assert loaded == 1
    assert skipped == 1
    assert model.loaded == [({"sanitized.embed.weight": "E"}, False)]


def test_deepseek_v4_regular_loader_reads_only_declared_shards(tmp_path):
    """A bundled drafter's model-dspark-* shards sit beside the model
    shards; the trunk load must consume only the manifest-declared set."""
    _touch_shard(tmp_path)
    (tmp_path / "model-dspark-00001-of-00001.safetensors").write_bytes(b"")
    model = _Model()
    read: list[str] = []

    def record_load(path: Path) -> dict:
        read.append(path.name)
        return {"embed.weight": "E"}

    loaded, skipped = _load_deepseek_v4_regular_weights(
        model,
        tmp_path,
        load_shard_fn=record_load,
        shard_names=["model-00001-of-00001.safetensors"],
    )

    assert loaded == 1
    assert skipped == 0
    assert read == ["model-00001-of-00001.safetensors"]


def test_deepseek_v4_hc_post_patch_returns_float32_before_fp16_overflow():
    mx = pytest.importorskip("mlx.core")
    model = _WrappedModel([_Layer()])

    patched = _patch_deepseek_v4_hc_post_float32(model)

    assert patched == 1
    assert model._moespresso_dsv4_hc_post_dtype == "float32"
    x = mx.array([[60000.0]], dtype=mx.float16)
    post = mx.array([1.2], dtype=mx.float16)
    residual = mx.zeros((1, 1, 1), dtype=mx.float16)
    comb = mx.zeros((1, 1, 1), dtype=mx.float16)
    out = model.model.layers[0]._hc_post(x, residual, post, comb)
    mx.eval(out)

    assert out.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(out)).item())
    assert float(mx.max(mx.abs(out)).item()) > 65504.0


def test_deepseek_v4_hc_post_matches_ds4_reference_comb_orientation():
    mx = pytest.importorskip("mlx.core")
    model = _WrappedModel([_Layer()])

    _patch_deepseek_v4_hc_post_float32(model)

    x = mx.zeros((1, 1, 2), dtype=mx.float32)
    post = mx.zeros((1, 1, 2), dtype=mx.float32)
    residual = mx.array(
        [[[[10.0, 100.0], [1.0, 2.0]]]],
        dtype=mx.float32,
    )
    comb = mx.array(
        [[[[1.0, 2.0], [3.0, 4.0]]]],
        dtype=mx.float32,
    )
    out = model.model.layers[0]._hc_post(x, residual, post, comb)
    mx.eval(out)

    expected = mx.array(
        [[[[13.0, 106.0], [24.0, 208.0]]]],
        dtype=mx.float32,
    )
    assert bool(mx.allclose(out, expected).item())


def _fake_q8_kquant_module(monkeypatch, calls):
    mx = pytest.importorskip("mlx.core")

    def fake_dequantize(weight, scales, codec, *, dtype=None):
        del scales
        calls["dequantize"] += 1
        assert codec == "q8_0"
        assert dtype == mx.float32
        return weight.astype(dtype)

    def fake_quantized_matmul(x, weight, scales, codec, *, transpose=True):
        del scales
        calls["quantized_matmul"] += 1
        assert codec == "q8_0"
        assert transpose is True
        # Mirror the QMV contract: float32 accumulation, output in x.dtype.
        return mx.matmul(
            x.astype(mx.float32), weight.astype(mx.float32).T
        ).astype(x.dtype)

    monkeypatch.setitem(
        sys.modules,
        "mlx_kquant",
        SimpleNamespace(
            dequantize=fake_dequantize,
            quantized_matmul=fake_quantized_matmul,
        ),
    )


def _kquant_attention_fixture(mx):
    wo_a = _KQuantModule(
        mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32),
        mx.zeros((1,), dtype=mx.uint8),
    )
    wo_b = _KQuantModule(
        mx.array([[5.0, 7.0], [11.0, 13.0]], dtype=mx.float32),
        mx.zeros((1,), dtype=mx.uint8),
    )
    return _KQuantAttention(wo_a, wo_b)


def test_deepseek_v4_kquant_attention_output_uses_fp32_q8_bridge(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    calls = {"dequantize": 0, "quantized_matmul": 0}
    _fake_q8_kquant_module(monkeypatch, calls)

    attn = _kquant_attention_fixture(mx)
    model = _WrappedModel([_AttentionLayer(attn)])

    patched = _patch_deepseek_v4_kquant_grouped_output_projection(model)

    assert patched == 1
    assert model._moespresso_dsv4_kquant_grouped_output_projection_layers == 1
    # Two token rows: the multi-row (prefill) shape must stay on the float32
    # dequant bridge; the tiled kquant qmm stages float16 weight tiles, a
    # measured quality regression at bulk shapes.
    hidden = mx.array(
        [[[2.0, 3.0, 5.0, 7.0], [1.0, 1.0, 1.0, 1.0]]], dtype=mx.float16)
    low = attn._grouped_output_projection(hidden)
    out = attn.wo_b(low)
    mx.eval(low, out)

    assert low.dtype == mx.float32
    assert out.dtype == mx.float32
    np.testing.assert_allclose(np.asarray(low), [[[8.0, 43.0], [3.0, 7.0]]])
    np.testing.assert_allclose(
        np.asarray(out), [[[341.0, 647.0], [64.0, 124.0]]])
    assert calls == {"dequantize": 3, "quantized_matmul": 0}


def _real_q8_module(kq, mx, out_dims=8, in_dims=64, seed=3):
    rng = np.random.default_rng(seed)
    dense = mx.array(
        rng.standard_normal((out_dims, in_dims), dtype=np.float32))
    wire, scales = kq.quantize(dense, "q8_0")
    mx.eval(wire, scales)
    return _KQuantModule(wire, scales)


def _q8_hc_model_fixture(
    mx, *, hidden=16, input_size=256, layer_cls=_Q8HcDecoderLayer
):
    wire_bytes = input_size // 32 * 34
    wo_a = _KQuantModule(
        mx.zeros((1, 34), dtype=mx.uint8),
        mx.zeros((1,), dtype=mx.uint8),
    )
    wo_b = _KQuantModule(
        mx.zeros((hidden, wire_bytes), dtype=mx.uint8),
        mx.zeros((1,), dtype=mx.uint8),
    )
    projection = mx.ones((1, 1, input_size), dtype=mx.float32)
    attn = _Q8HcAttention(wo_a, wo_b, projection)
    layer = layer_cls(attn, mx)
    return _WrappedModel([layer]), layer, attn


class _Q8FfnDown(_KQuantModule):
    def __init__(self, mx, *, hidden, input_size, events):
        super().__init__(
            mx.zeros((hidden, input_size // 32 * 34), dtype=mx.uint8),
            mx.zeros((1,), dtype=mx.uint8),
        )
        self.mx = mx
        self.hidden = hidden
        self.events = events
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        self.events.append("down_fallback")
        return self.mx.full(
            (*x.shape[:-1], self.hidden), 0.25, dtype=self.mx.bfloat16)


class _Q8FfnGate:
    def __init__(self, mx, events):
        self.mx = mx
        self.events = events
        self.calls = 0

    def __call__(self, x, *, input_ids=None):
        del input_ids
        self.calls += 1
        self.events.append("gate")
        rows = tuple(int(dim) for dim in x.shape[:-1])
        return (
            self.mx.zeros((*rows, 1), dtype=self.mx.uint32),
            self.mx.ones((*rows, 1), dtype=self.mx.float32),
        )


class _Q8FfnShared:
    def __init__(self, down_proj, down_input, events):
        self.down_proj = down_proj
        self.down_input = down_input
        self.events = events
        self.calls = 0
        self.raise_on_call = False

    def __call__(self, x):
        del x
        self.calls += 1
        self.events.append("shared")
        if self.raise_on_call:
            raise RuntimeError("shared expert failed")
        return self.down_proj(self.down_input)


def _configure_q8_ffn_hc_post(
    monkeypatch,
    mx,
    kq,
    *,
    hidden=16,
    attn_input=256,
    ffn_input=32,
):
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_HIDDEN", hidden)
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_INPUT", attn_input)
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_FFN_HC_POST_HIDDEN", hidden)
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_FFN_HC_POST_INPUT", ffn_input)
    monkeypatch.setattr(
        dsv4_runtime,
        "_DSV4_Q8_FFN_HC_POST_WIRE_BYTES",
        ffn_input // 32 * 34,
    )
    calls = {"attention": 0, "ffn": 0}

    def attention_native(
        x, weight, scales, residual, post, comb, kquant_type
    ):
        del scales, post, comb
        calls["attention"] += 1
        assert x.dtype == mx.float32
        assert x.shape == (1, 1, attn_input)
        assert weight.shape == (hidden, attn_input // 32 * 34)
        assert residual.shape == (4, hidden)
        assert kquant_type == "q8_0"
        return residual

    def ffn_native(
        x,
        weight,
        scales,
        routed,
        residual,
        post,
        comb,
        kquant_type,
    ):
        del scales
        calls["ffn"] += 1
        assert x.dtype == mx.bfloat16
        assert x.shape == (1, 1, ffn_input)
        assert weight.shape == (hidden, ffn_input // 32 * 34)
        assert routed.dtype == mx.float16
        assert routed.shape == (hidden,)
        assert residual.dtype == mx.float32
        assert residual.shape == (4, hidden)
        assert post.dtype == mx.float32
        assert post.shape == (4,)
        assert comb.dtype == mx.float32
        assert comb.shape == (4, 4)
        assert kquant_type == "q8_0"
        return residual + routed.astype(mx.float32)[None, :]

    monkeypatch.setattr(
        kq,
        "quantized_matmul_qmv_hc_post",
        attention_native,
        raising=False,
    )
    monkeypatch.setattr(
        kq,
        "quantized_matmul_qmv_add_hc_post",
        ffn_native,
        raising=False,
    )
    return calls


def test_deepseek_v4_q8_hc_post_requires_explicit_native_api(monkeypatch):
    pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    model = _WrappedModel([])
    monkeypatch.delattr(
        kq, "quantized_matmul_qmv_hc_post", raising=False)

    with pytest.raises(
        DeepseekV4RuntimeLoadError,
        match="mlx_kquant.quantized_matmul_qmv_hc_post",
    ):
        _patch_deepseek_v4_q8_hc_post(model)
    assert model._moespresso_dsv4_q8_hc_post_enabled is False
    assert model._moespresso_dsv4_q8_hc_post_native_api is False

    monkeypatch.setattr(
        kq,
        "quantized_matmul_qmv_hc_post",
        lambda *args, **kwargs: None,
        raising=False,
    )
    assert _patch_deepseek_v4_q8_hc_post(model) == 0
    assert model._moespresso_dsv4_q8_hc_post_enabled is False
    assert model._moespresso_dsv4_q8_hc_post_native_api is True
    assert model._moespresso_dsv4_q8_hc_post_layers == 0


def test_deepseek_v4_q8_hc_post_engages_and_skips_only_attention_post(
        monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    hidden = 16
    input_size = 256
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_HIDDEN", hidden)
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_INPUT", input_size)
    calls = {"native": 0}

    def fake_native(x, weight, scales, residual, post, comb, kquant_type):
        del scales
        calls["native"] += 1
        assert x.dtype == mx.float32
        assert x.shape == (1, 1, input_size)
        assert weight.shape == (hidden, input_size // 32 * 34)
        assert residual.shape == (4, hidden)
        assert post.shape == (4,)
        assert comb.shape == (4, 4)
        assert kquant_type == "q8_0"
        return residual + post[:, None]

    monkeypatch.setattr(
        kq, "quantized_matmul_qmv_hc_post", fake_native, raising=False)
    model, layer, attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=input_size)
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    before = dsv4_runtime.q8_hc_post_call_counts()
    q8_before = dsv4_runtime.q8_dense_matmul_call_counts()

    x = mx.ones((1, 1, 4, hidden), dtype=mx.float32)
    out = layer(x)
    mx.eval(out)
    after = dsv4_runtime.q8_hc_post_call_counts()
    q8_after = dsv4_runtime.q8_dense_matmul_call_counts()

    assert out.shape == (1, 1, 4, hidden)
    assert calls["native"] == 1
    assert layer.hc_post_calls == 1
    assert layer.original_calls == 0
    assert layer._moespresso_dsv4_q8_hc_post_engaged_calls == 1
    assert layer._moespresso_dsv4_q8_hc_post_fallback_calls == 0
    assert after["engaged"] == before["engaged"] + 1
    assert after["fallback"] == before["fallback"]
    assert q8_after["decode_wire_qmv_wo_b"] == (
        q8_before["decode_wire_qmv_wo_b"] + 1)
    assert model._moespresso_dsv4_q8_hc_post_enabled is True
    assert model._moespresso_dsv4_q8_hc_post_native_api is True
    assert model._moespresso_dsv4_q8_hc_post_layers == 1
    assert getattr(attn.wo_b, "_moespresso_dsv4_q8_hc_post_eligible")
    assert dsv4_runtime._DSV4_Q8_HC_POST_CONTEXT.get() is None


def test_a_drafter_stage_loaded_after_the_patch_stays_on_the_stock_route(
        monkeypatch):
    """The hC-post patch replaces `__call__` on the layer *class*.

    DSpark draft stages subclass the trunk decoder layer, so a drafter loaded
    after the patch inherits the fused call. Only the per-instance eligibility
    flag keeps it off the fused route, and nothing pinned that: a refactor
    that hoisted eligibility to the class would send draft stages through a
    fusion sized for the trunk. Commissioned by errata E4.
    """
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    hidden = 16
    input_size = 256
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_HIDDEN", hidden)
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_INPUT", input_size)
    calls = {"native": 0}

    def fake_native(x, weight, scales, residual, post, comb, kquant_type):
        del x, weight, scales, post, comb, kquant_type
        calls["native"] += 1
        return residual

    monkeypatch.setattr(
        kq, "quantized_matmul_qmv_hc_post", fake_native, raising=False)

    model, layer, _attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=input_size)
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    x = mx.ones((1, 1, 4, hidden), dtype=mx.float32)
    mx.eval(layer(x))
    assert calls["native"] == 1

    # The drafter's stage class subclasses the trunk layer, and it is
    # constructed after the class patch is already installed.
    class _DraftStage(type(layer)):
        pass

    _stage_model, stage, _stage_attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=input_size, layer_cls=_DraftStage)
    assert type(stage).__call__ is type(layer).__call__

    before = dsv4_runtime.q8_hc_post_call_counts()
    native_before = calls["native"]
    out = stage(x)
    mx.eval(out)
    after = dsv4_runtime.q8_hc_post_call_counts()

    # The stage runs the stock body: no fused kernel, no engagement.
    assert calls["native"] == native_before
    assert stage.original_calls == 1
    assert stage.hc_post_calls == 0
    assert after["engaged"] == before["engaged"]
    assert after["fallback"] == before["fallback"]
    assert not getattr(
        stage, "_moespresso_dsv4_q8_hc_post_eligible", False)


def test_deepseek_v4_q8_hc_post_falls_back_delegates_and_resets(
        monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    hidden = 16
    input_size = 256
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_HIDDEN", hidden)
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_INPUT", input_size)
    native_calls = {"count": 0}

    def fake_native(*args, **kwargs):
        del args, kwargs
        native_calls["count"] += 1
        return mx.zeros((4, hidden), dtype=mx.float32)

    monkeypatch.setattr(
        kq, "quantized_matmul_qmv_hc_post", fake_native, raising=False)
    model, layer, attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=input_size)
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1

    monkeypatch.setattr(
        dsv4_runtime,
        "_deepseek_v4_q8_hc_post_eligible",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        dsv4_runtime,
        "_kquant_matmul_ds4_fp32",
        lambda *args, **kwargs: mx.zeros(
            (1, 1, hidden), dtype=mx.float32),
    )
    before = dsv4_runtime.q8_hc_post_call_counts()
    x = mx.ones((1, 1, 4, hidden), dtype=mx.float32)
    out = layer(x)
    mx.eval(out)
    after = dsv4_runtime.q8_hc_post_call_counts()
    assert native_calls["count"] == 0
    assert layer.hc_post_calls == 2
    assert after["fallback"] == before["fallback"] + 1

    delegated = layer(x.astype(mx.bfloat16))
    mx.eval(delegated)
    final = dsv4_runtime.q8_hc_post_call_counts()
    assert layer.original_calls == 1
    assert final["delegated"] == after["delegated"] + 1

    attn.raise_on_call = True
    with pytest.raises(RuntimeError, match="attention failed"):
        layer(x)
    assert dsv4_runtime._DSV4_Q8_HC_POST_CONTEXT.get() is None


def test_deepseek_v4_q8_hc_post_leaves_ineligible_instances_on_stock_route(
        monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    hidden = 16
    input_size = 256
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_HIDDEN", hidden)
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_INPUT", input_size)
    calls = {"native": 0}

    def fake_native(x, weight, scales, residual, post, comb, kquant_type):
        del x, weight, scales, post, comb, kquant_type
        calls["native"] += 1
        return residual

    monkeypatch.setattr(
        kq, "quantized_matmul_qmv_hc_post", fake_native, raising=False)
    model, layer, _attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=input_size)
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    x = mx.ones((1, 1, 4, hidden), dtype=mx.float32)
    mx.eval(layer(x))
    assert calls["native"] == 1

    # The layer class now carries the fusion wrapper. A second model whose
    # wo_b never received the fp32 linear seam contract is not eligible, so
    # it keeps the stock layer body even though the wrapper is installed.
    fresh, fresh_layer, fresh_attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=input_size)
    assert _patch_deepseek_v4_q8_hc_post(fresh) == 0
    assert fresh._moespresso_dsv4_q8_hc_post_enabled is False
    assert fresh._moespresso_dsv4_q8_hc_post_layers == 0
    assert not getattr(
        fresh_layer, "_moespresso_dsv4_q8_hc_post_eligible", False)
    assert not getattr(
        fresh_attn.wo_b, "_moespresso_dsv4_q8_hc_post_eligible", False)
    mx.eval(fresh_layer(x))
    assert fresh_layer.original_calls == 1
    assert calls["native"] == 1


def test_deepseek_v4_q8_hc_post_composes_with_hidden_tap_orders(
        monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.spec_decode import install_hidden_tap

    hidden = 16
    input_size = 256
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_HIDDEN", hidden)
    monkeypatch.setattr(dsv4_runtime, "_DSV4_Q8_HC_POST_INPUT", input_size)
    native_calls = {"count": 0}

    def fake_native(x, weight, scales, residual, post, comb, kquant_type):
        del x, weight, scales, post, comb, kquant_type
        native_calls["count"] += 1
        return residual

    monkeypatch.setattr(
        kq, "quantized_matmul_qmv_hc_post", fake_native, raising=False)
    x = mx.ones((1, 1, 4, hidden), dtype=mx.float32)

    class InnerTapLayer(_Q8HcDecoderLayer):
        pass

    # The tap wraps the layer class first, so the fusion wrapper installed
    # by the next model finds the tap already inside the layer body. The
    # first model is ineligible (no fp32 linear seam contract on wo_b), so
    # its class stays unwrapped until the eligible model arrives.
    unfused_model, unfused_layer, _unfused_attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=input_size, layer_cls=InnerTapLayer)
    assert _patch_deepseek_v4_q8_hc_post(unfused_model) == 0
    inner_transform_calls = {"count": 0}

    def inner_transform(layer_id, out):
        assert layer_id == 0
        inner_transform_calls["count"] += 1
        return out

    old_tap = install_hidden_tap(unfused_model, [0], inner_transform)
    old_tap.active = True
    mx.eval(unfused_layer(x))
    assert inner_transform_calls["count"] == 1
    old_tap.rows.clear()

    inner_model, inner_layer, _inner_attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=input_size, layer_cls=InnerTapLayer)
    assert _patch_deepseek_v4_kquant_grouped_output_projection(inner_model) == 1
    assert _patch_deepseek_v4_q8_hc_post(inner_model) == 1
    inner_tap = install_hidden_tap(inner_model, [0], inner_transform)
    inner_tap.active = True
    inner_out = inner_layer(x)
    mx.eval(inner_out)
    assert inner_transform_calls["count"] == 2
    assert 0 in inner_tap.rows
    assert old_tap.rows == {}

    class OuterTapLayer(_Q8HcDecoderLayer):
        pass

    outer_model, outer_layer, _outer_attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=input_size, layer_cls=OuterTapLayer)
    assert _patch_deepseek_v4_kquant_grouped_output_projection(outer_model) == 1
    assert _patch_deepseek_v4_q8_hc_post(outer_model) == 1
    outer_transform_calls = {"count": 0}

    def outer_transform(layer_id, out):
        assert layer_id == 0
        outer_transform_calls["count"] += 1
        return out

    outer_tap = install_hidden_tap(outer_model, [0], outer_transform)
    outer_tap.active = True
    outer_out = outer_layer(x)
    mx.eval(outer_out)
    assert outer_transform_calls["count"] == 1
    assert 0 in outer_tap.rows
    assert native_calls["count"] == 2


def test_deepseek_v4_q8_ffn_hc_post_requires_explicit_contract(monkeypatch):
    pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    monkeypatch.delattr(
        kq, "quantized_matmul_qmv_add_hc_post", raising=False)

    # The FFN fusion rides inside the attention fusion, so a graph the
    # attention fusion never claimed stays on the stock route without
    # consulting the native API at all.
    model = _WrappedModel([])
    assert _patch_deepseek_v4_q8_ffn_hc_post(model) == 0
    assert model._moespresso_dsv4_q8_ffn_hc_post_enabled is False
    assert model._moespresso_dsv4_q8_ffn_hc_post_native_api is False

    claimed = _WrappedModel(
        [SimpleNamespace(_moespresso_dsv4_q8_hc_post_eligible=True, mlp=None)]
    )
    with pytest.raises(
        DeepseekV4RuntimeLoadError,
        match="mlx_kquant.quantized_matmul_qmv_add_hc_post",
    ):
        _patch_deepseek_v4_q8_ffn_hc_post(claimed)
    assert claimed._moespresso_dsv4_q8_ffn_hc_post_native_api is False

    monkeypatch.setattr(
        kq,
        "quantized_matmul_qmv_add_hc_post",
        lambda *args, **kwargs: None,
        raising=False,
    )
    assert _patch_deepseek_v4_q8_ffn_hc_post(claimed) == 0
    assert claimed._moespresso_dsv4_q8_ffn_hc_post_enabled is False
    assert claimed._moespresso_dsv4_q8_ffn_hc_post_native_api is True
    assert claimed._moespresso_dsv4_q8_ffn_hc_post_layers == 0


def test_deepseek_v4_q8_ffn_hc_post_composes_attention_and_c6(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    calls = _configure_q8_ffn_hc_post(monkeypatch, mx, kq)
    model, layer, attn, block, shared, down, switch, events = (
        _q8_iqk_ffn_hc_model_fixture(mx)
    )
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    assert _patch_deepseek_v4_q8_ffn_hc_post(model) == 1
    attention_before = dsv4_runtime.q8_hc_post_call_counts()
    ffn_before = dsv4_runtime.q8_ffn_hc_post_call_counts()
    dense_before = dsv4_runtime.q8_dense_matmul_call_counts()

    out = layer(mx.ones((1, 1, 4, 16), dtype=mx.float32))
    mx.eval(out)

    attention_after = dsv4_runtime.q8_hc_post_call_counts()
    ffn_after = dsv4_runtime.q8_ffn_hc_post_call_counts()
    dense_after = dsv4_runtime.q8_dense_matmul_call_counts()
    assert out.dtype == mx.float32
    assert out.shape == (1, 1, 4, 16)
    assert calls == {"attention": 1, "ffn": 1}
    assert events == ["gate", "switch", "shared"]
    assert switch.calls == 1
    assert shared.calls == 1
    assert down.calls == 0
    assert layer.hc_post_calls == 0
    assert layer.original_calls == 0
    assert attention_after["engaged"] == attention_before["engaged"] + 1
    assert attention_after["fallback"] == attention_before["fallback"]
    assert ffn_after["engaged"] == ffn_before["engaged"] + 1
    assert ffn_after["fallback"] == ffn_before["fallback"]
    assert ffn_after["delegated"] == ffn_before["delegated"]
    assert dense_after["decode_wire_qmv_wo_b"] == (
        dense_before["decode_wire_qmv_wo_b"] + 1)
    assert model._moespresso_dsv4_q8_ffn_hc_post_enabled is True
    assert model._moespresso_dsv4_q8_ffn_hc_post_native_api is True
    assert model._moespresso_dsv4_q8_ffn_hc_post_layers == 1
    assert block._moespresso_dsv4_q8_ffn_hc_post_eligible
    assert shared.down_proj._moespresso_dsv4_q8_ffn_hc_post_eligible
    assert attn.wo_b._moespresso_dsv4_q8_hc_post_eligible
    assert dsv4_runtime._DSV4_Q8_HC_POST_CONTEXT.get() is None
    assert dsv4_runtime._DSV4_Q8_FFN_HC_POST_CONTEXT.get() is None


@pytest.mark.parametrize("entry_dtype_name", ["float16", "bfloat16"])
def test_deepseek_v4_q8_ffn_hc_post_layer0_delegates_only_attention(
    monkeypatch,
    entry_dtype_name,
):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    calls = _configure_q8_ffn_hc_post(monkeypatch, mx, kq)
    model, layer, _attn, _block, shared, down, switch, _events = (
        _q8_iqk_ffn_hc_model_fixture(mx)
    )
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    assert _patch_deepseek_v4_q8_ffn_hc_post(model) == 1
    monkeypatch.setattr(
        dsv4_runtime,
        "_kquant_matmul_ds4_fp32",
        lambda *args, **kwargs: mx.zeros(
            (1, 1, 16), dtype=mx.float32),
    )
    attention_before = dsv4_runtime.q8_hc_post_call_counts()
    ffn_before = dsv4_runtime.q8_ffn_hc_post_call_counts()

    out = layer(
        mx.ones((1, 1, 4, 16), dtype=getattr(mx, entry_dtype_name))
    )
    mx.eval(out)

    attention_after = dsv4_runtime.q8_hc_post_call_counts()
    ffn_after = dsv4_runtime.q8_ffn_hc_post_call_counts()
    assert out.dtype == mx.float32
    assert out.shape == (1, 1, 4, 16)
    assert calls == {"attention": 0, "ffn": 1}
    assert switch.calls == 1
    assert shared.calls == 1
    assert down.calls == 0
    assert layer.hc_post_calls == 1
    assert layer.original_calls == 0
    assert attention_after["engaged"] == attention_before["engaged"]
    assert attention_after["fallback"] == attention_before["fallback"]
    assert attention_after["delegated"] == (
        attention_before["delegated"] + 1)
    assert ffn_after["engaged"] == ffn_before["engaged"] + 1
    assert ffn_after["fallback"] == ffn_before["fallback"]
    assert ffn_after["delegated"] == ffn_before["delegated"]
    assert dsv4_runtime._DSV4_Q8_HC_POST_CONTEXT.get() is None
    assert dsv4_runtime._DSV4_Q8_FFN_HC_POST_CONTEXT.get() is None


def test_deepseek_v4_q8_ffn_hc_post_falls_back_delegates_and_resets(
    monkeypatch,
):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    calls = _configure_q8_ffn_hc_post(monkeypatch, mx, kq)
    model, layer, _attn, _block, shared, down, switch, events = (
        _q8_iqk_ffn_hc_model_fixture(mx)
    )
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    assert _patch_deepseek_v4_q8_ffn_hc_post(model) == 1
    monkeypatch.setattr(
        dsv4_runtime,
        "_deepseek_v4_q8_ffn_hc_post_eligible",
        lambda *args, **kwargs: False,
    )
    attention_before = dsv4_runtime.q8_hc_post_call_counts()
    ffn_before = dsv4_runtime.q8_ffn_hc_post_call_counts()
    decode = mx.ones((1, 1, 4, 16), dtype=mx.float32)

    out = layer(decode)
    mx.eval(out)

    attention_after = dsv4_runtime.q8_hc_post_call_counts()
    ffn_after = dsv4_runtime.q8_ffn_hc_post_call_counts()
    assert out.dtype == mx.float32
    assert calls == {"attention": 1, "ffn": 0}
    assert events == ["gate", "switch", "shared", "down_fallback"]
    assert switch.calls == 1
    assert shared.calls == 1
    assert down.calls == 1
    assert layer.hc_post_calls == 1
    assert attention_after["engaged"] == attention_before["engaged"] + 1
    assert ffn_after["fallback"] == ffn_before["fallback"] + 1
    assert ffn_after["engaged"] == ffn_before["engaged"]

    delegated = layer(mx.ones((1, 2, 4, 16), dtype=mx.float32))
    mx.eval(delegated)
    attention_delegated = dsv4_runtime.q8_hc_post_call_counts()
    ffn_delegated = dsv4_runtime.q8_ffn_hc_post_call_counts()
    assert layer.original_calls == 1
    assert attention_delegated["delegated"] == (
        attention_after["delegated"] + 1)
    assert ffn_delegated["delegated"] == ffn_after["delegated"] + 1

    shared.raise_on_call = True
    with pytest.raises(RuntimeError, match="shared expert failed"):
        layer(decode)
    assert dsv4_runtime._DSV4_Q8_HC_POST_CONTEXT.get() is None
    assert dsv4_runtime._DSV4_Q8_FFN_HC_POST_CONTEXT.get() is None


def test_deepseek_v4_q8_ffn_hc_post_ineligible_block_keeps_stock_route(
        monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    calls = _configure_q8_ffn_hc_post(monkeypatch, mx, kq)
    installed, _layer, _attn, _block, _shared, _down, _switch, _events = (
        _q8_iqk_ffn_hc_model_fixture(mx)
    )
    assert _patch_deepseek_v4_kquant_grouped_output_projection(installed) == 1
    assert _patch_deepseek_v4_q8_hc_post(installed) == 1
    assert _patch_deepseek_v4_q8_ffn_hc_post(installed) == 1

    # The IQ_K block class now carries the fusion wrapper. A block whose
    # shared-expert down projection is off the declared wire shape is
    # not a candidate, so its layer keeps the stock routed-plus-shared sum
    # and the stock FFN hC post.
    model, layer, _attn, block, shared, down, switch, events = (
        _q8_iqk_ffn_hc_model_fixture(mx, ffn_input=64)
    )
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    assert _patch_deepseek_v4_q8_ffn_hc_post(model) == 0
    assert model._moespresso_dsv4_q8_ffn_hc_post_enabled is False
    assert model._moespresso_dsv4_q8_ffn_hc_post_native_api is True
    assert not getattr(
        layer, "_moespresso_dsv4_q8_ffn_hc_post_eligible", False)
    assert not getattr(
        block, "_moespresso_dsv4_q8_ffn_hc_post_eligible", False)
    assert shared.down_proj is down

    out = layer(mx.ones((1, 1, 4, 16), dtype=mx.float32))
    mx.eval(out)
    assert calls == {"attention": 1, "ffn": 0}
    assert events == ["gate", "switch", "shared", "down_fallback"]
    assert switch.calls == 1
    assert down.calls == 1
    assert layer.hc_post_calls == 1
    assert dsv4_runtime._DSV4_Q8_FFN_HC_POST_CONTEXT.get() is None


def _q8_iqk_ffn_hc_model_fixture(
    mx, *, hidden=16, attn_input=256, ffn_input=32
):
    """The FFN fixture over an IQ_K-form MoE block.

    The block classes are created per call so one test's class-level
    fusion wrap can never leak into another test.
    """
    model, layer, attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=attn_input)
    events = []
    down = _Q8FfnDown(
        mx, hidden=hidden, input_size=ffn_input, events=events)
    shared = _Q8FfnShared(
        down,
        mx.ones((1, 1, ffn_input), dtype=mx.bfloat16),
        events,
    )

    class IqkDeepseekV4SwitchGLU:
        def __init__(self):
            self.calls = 0

        def __call__(self, x, indices):
            del indices
            self.calls += 1
            events.append("switch")
            rows = tuple(int(dim) for dim in x.shape[:-1])
            return mx.full((*rows, 1, hidden), 0.5, dtype=mx.float16)

    class _IqkMoEBlock:
        def __init__(self):
            self.gate = _Q8FfnGate(mx, events)
            self.switch_mlp = IqkDeepseekV4SwitchGLU()
            self.shared_experts = shared
            self.stock_calls = 0

        def __call__(self, x, input_ids=None):
            self.stock_calls += 1
            inds, scores = self.gate(x, input_ids=input_ids)
            inds = inds.astype(mx.uint32)
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2).astype(
                y.dtype).reshape(x.shape)
            return y + self.shared_experts(x)

    block = _IqkMoEBlock()
    layer.mlp = block
    return model, layer, attn, block, shared, down, block.switch_mlp, events


def _q8_pooled_iqk_ffn_hc_model_fixture(
    mx, *, full_residency=True, hidden=16, attn_input=256, ffn_input=32
):
    model, layer, attn = _q8_hc_model_fixture(
        mx, hidden=hidden, input_size=attn_input)
    events = []
    down = _Q8FfnDown(
        mx, hidden=hidden, input_size=ffn_input, events=events)
    shared = _Q8FfnShared(
        down,
        mx.ones((1, 1, ffn_input), dtype=mx.bfloat16),
        events,
    )

    class PooledSwitchGLU:
        _all_iqk = True

        def __init__(self):
            self.calls = 0
            self.commits = []

        def _barrier_free_decode_ready(self):
            return full_residency

        def __call__(self, x, indices):
            del indices
            self.calls += 1
            events.append("switch")
            rows = tuple(int(dim) for dim in x.shape[:-1])
            return mx.full((*rows, 1, hidden), 0.5, dtype=mx.float16)

        def commit_iqk_output(self, output, *, rows):
            del output
            self.commits.append(rows)

    class _PooledIqkMoEBlock:
        def __init__(self):
            self.gate = _Q8FfnGate(mx, events)
            self.switch_mlp = PooledSwitchGLU()
            self.shared_experts = shared
            self.stock_calls = 0

        def __call__(self, x, input_ids=None):
            self.stock_calls += 1
            inds, scores = self.gate(x, input_ids=input_ids)
            inds = inds.astype(mx.uint32)
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2).astype(
                y.dtype).reshape(x.shape)
            return y + self.shared_experts(x)

    block = _PooledIqkMoEBlock()
    layer.mlp = block
    return model, layer, attn, block, shared, down, block.switch_mlp, events


def test_deepseek_v4_q8_ffn_hc_post_iqk_block_off_contract_keeps_stock(
        monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    _configure_q8_ffn_hc_post(monkeypatch, mx, kq)
    # A shared-expert down projection off the declared wire shape keeps
    # the IQ_K block on its stock route; the fusion installs nowhere.
    model, layer, _attn, block, shared, down, _switch, _events = (
        _q8_iqk_ffn_hc_model_fixture(mx, ffn_input=64)
    )
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    assert _patch_deepseek_v4_q8_ffn_hc_post(model) == 0
    assert model._moespresso_dsv4_q8_ffn_hc_post_enabled is False
    assert model._moespresso_dsv4_q8_ffn_hc_post_layers == 0
    assert shared.down_proj is down
    assert not getattr(
        layer, "_moespresso_dsv4_q8_ffn_hc_post_eligible", False)
    assert not type(block).__dict__.get(
        "_moespresso_dsv4_q8_ffn_hc_post_wrapped", False)


def test_deepseek_v4_q8_ffn_hc_post_pooled_iqk_requires_full_residency(
    monkeypatch,
):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    _configure_q8_ffn_hc_post(monkeypatch, mx, kq)
    model, layer, _attn, block, shared, down, _switch, _events = (
        _q8_pooled_iqk_ffn_hc_model_fixture(mx, full_residency=False)
    )
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    assert _patch_deepseek_v4_q8_ffn_hc_post(model) == 0
    assert model._moespresso_dsv4_q8_ffn_hc_post_enabled is False
    assert shared.down_proj is down
    assert not getattr(
        layer, "_moespresso_dsv4_q8_ffn_hc_post_eligible", False)
    assert not getattr(
        block, "_moespresso_dsv4_q8_ffn_hc_post_eligible", False)


def test_deepseek_v4_q8_ffn_hc_post_pooled_iqk_engages_and_commits(
    monkeypatch,
):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    calls = _configure_q8_ffn_hc_post(monkeypatch, mx, kq)
    model, layer, _attn, block, _shared, down, switch, events = (
        _q8_pooled_iqk_ffn_hc_model_fixture(mx)
    )
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    assert _patch_deepseek_v4_q8_ffn_hc_post(model) == 1

    out = layer(mx.ones((1, 1, 4, 16), dtype=mx.float32))
    mx.eval(out)

    assert calls == {"attention": 1, "ffn": 1}
    assert events == ["gate", "switch", "shared"]
    assert switch.calls == 1
    assert switch.commits == [1]
    assert down.calls == 0
    assert block.stock_calls == 0


def test_deepseek_v4_q8_ffn_hc_post_iqk_block_engages_structurally(
        monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    calls = _configure_q8_ffn_hc_post(monkeypatch, mx, kq)
    model, layer, _attn, block, shared, down, switch, events = (
        _q8_iqk_ffn_hc_model_fixture(mx)
    )
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    assert _patch_deepseek_v4_q8_ffn_hc_post(model) == 1
    ffn_before = dsv4_runtime.q8_ffn_hc_post_call_counts()

    out = layer(mx.ones((1, 1, 4, 16), dtype=mx.float32))
    mx.eval(out)

    ffn_after = dsv4_runtime.q8_ffn_hc_post_call_counts()
    assert out.dtype == mx.float32
    assert out.shape == (1, 1, 4, 16)
    assert calls == {"attention": 1, "ffn": 1}
    assert events == ["gate", "switch", "shared"]
    assert switch.calls == 1
    assert shared.calls == 1
    assert down.calls == 0
    assert block.stock_calls == 0
    assert layer.hc_post_calls == 0
    assert layer.original_calls == 0
    assert ffn_after["engaged"] == ffn_before["engaged"] + 1
    assert ffn_after["fallback"] == ffn_before["fallback"]
    assert ffn_after["delegated"] == ffn_before["delegated"]
    assert model._moespresso_dsv4_q8_ffn_hc_post_enabled is True
    assert model._moespresso_dsv4_q8_ffn_hc_post_layers == 1
    assert block._moespresso_dsv4_q8_ffn_hc_post_eligible
    assert shared.down_proj._moespresso_dsv4_q8_ffn_hc_post_eligible
    assert type(block).__dict__.get(
        "_moespresso_dsv4_q8_ffn_hc_post_wrapped", False)
    assert dsv4_runtime._DSV4_Q8_HC_POST_CONTEXT.get() is None
    assert dsv4_runtime._DSV4_Q8_FFN_HC_POST_CONTEXT.get() is None


def test_deepseek_v4_q8_ffn_hc_post_iqk_block_falls_back_and_delegates(
        monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    calls = _configure_q8_ffn_hc_post(monkeypatch, mx, kq)
    model, layer, _attn, _block, shared, down, switch, events = (
        _q8_iqk_ffn_hc_model_fixture(mx)
    )
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    assert _patch_deepseek_v4_q8_hc_post(model) == 1
    assert _patch_deepseek_v4_q8_ffn_hc_post(model) == 1
    monkeypatch.setattr(
        dsv4_runtime,
        "_deepseek_v4_q8_ffn_hc_post_eligible",
        lambda *args, **kwargs: False,
    )
    ffn_before = dsv4_runtime.q8_ffn_hc_post_call_counts()
    decode = mx.ones((1, 1, 4, 16), dtype=mx.float32)

    out = layer(decode)
    mx.eval(out)

    ffn_after = dsv4_runtime.q8_ffn_hc_post_call_counts()
    assert out.dtype == mx.float32
    assert calls == {"attention": 1, "ffn": 0}
    assert events == ["gate", "switch", "shared", "down_fallback"]
    assert switch.calls == 1
    assert shared.calls == 1
    assert down.calls == 1
    assert layer.hc_post_calls == 1
    assert ffn_after["fallback"] == ffn_before["fallback"] + 1
    assert ffn_after["engaged"] == ffn_before["engaged"]

    delegated = layer(mx.ones((1, 2, 4, 16), dtype=mx.float32))
    mx.eval(delegated)
    ffn_delegated = dsv4_runtime.q8_ffn_hc_post_call_counts()
    assert layer.original_calls == 1
    assert ffn_delegated["delegated"] == ffn_after["delegated"] + 1

    shared.raise_on_call = True
    with pytest.raises(RuntimeError, match="shared expert failed"):
        layer(decode)
    assert dsv4_runtime._DSV4_Q8_HC_POST_CONTEXT.get() is None
    assert dsv4_runtime._DSV4_Q8_FFN_HC_POST_CONTEXT.get() is None


def test_deepseek_v4_q8_affine_views_match_wire_lattice():
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_q8_affine_views,
    )

    module = _real_q8_module(kq, mx)
    affine = _deepseek_v4_q8_affine_views(module, mx=mx)
    assert affine is not None
    w_q, scales, biases = affine
    # Cached on the module after the first build.
    assert _deepseek_v4_q8_affine_views(module, mx=mx) is affine

    # Manual float32 dequant of the affine views reproduces the q8_0 wire
    # dequant bit for bit: both compute d * q in float32.
    rows = int(w_q.shape[0])
    unsigned = mx.view(w_q, mx.uint8).reshape(rows, -1, 32)
    manual = (unsigned.astype(mx.float32) - 128.0) * (
        scales.astype(mx.float32)[..., None]
    )
    manual = manual.reshape(rows, -1)
    reference = kq.dequantize(
        module["weight"], module["scales"], "q8_0", dtype=mx.float32)
    mx.eval(manual, reference)
    np.testing.assert_array_equal(
        np.asarray(manual).view(np.uint32),
        np.asarray(reference).view(np.uint32),
    )

    # Non-wire tensors return None instead of a bogus repack.
    bad = _KQuantModule(
        mx.zeros((2, 2), dtype=mx.uint8), mx.zeros((1,), dtype=mx.uint8))
    assert _deepseek_v4_q8_affine_views(bad, mx=mx) is None


@pytest.mark.parametrize("x_dtype_name", ["float32", "float16", "bfloat16"])
def test_deepseek_v4_q8_decode_row_uses_affine_qmv(monkeypatch, x_dtype_name):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_q8_affine_views,
        _kquant_matmul_ds4_fp32,
    )

    module = _real_q8_module(kq, mx)
    affine = _deepseek_v4_q8_affine_views(module, mx=mx)
    calls = {"dequantize": 0}
    real_dequantize = kq.dequantize

    def spy_dequantize(*args, **kwargs):
        calls["dequantize"] += 1
        return real_dequantize(*args, **kwargs)

    spy_kq = SimpleNamespace(
        dequantize=spy_dequantize, quantized_matmul=kq.quantized_matmul)

    x = mx.array(
        np.random.default_rng(4).standard_normal((1, 1, 64), dtype=np.float32)
    ).astype(getattr(mx, x_dtype_name))
    out = _kquant_matmul_ds4_fp32(
        x, module["weight"], module["scales"], "q8_0",
        mx=mx, kq=spy_kq, affine=affine,
    )
    reference = mx.matmul(
        x.astype(mx.float32),
        real_dequantize(
            module["weight"], module["scales"], "q8_0", dtype=mx.float32).T,
    )
    mx.eval(out, reference)

    assert calls["dequantize"] == 0
    assert out.dtype == mx.float32
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(reference), rtol=1.0e-5, atol=1.0e-5)


def test_deepseek_v4_q8_multi_row_stays_on_dequant():
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_q8_affine_views,
        _kquant_matmul_ds4_fp32,
    )

    module = _real_q8_module(kq, mx)
    affine = _deepseek_v4_q8_affine_views(module, mx=mx)
    calls = {"dequantize": 0}
    real_dequantize = kq.dequantize

    def spy_dequantize(*args, **kwargs):
        calls["dequantize"] += 1
        return real_dequantize(*args, **kwargs)

    spy_kq = SimpleNamespace(
        dequantize=spy_dequantize, quantized_matmul=kq.quantized_matmul)

    # Multi-row (prefill) calls keep the float32 dequant bridge.
    x = mx.zeros((1, 3, 64), dtype=mx.float16)
    out = _kquant_matmul_ds4_fp32(
        x, module["weight"], module["scales"], "q8_0",
        mx=mx, kq=spy_kq, affine=affine,
    )
    mx.eval(out)
    assert out.dtype == mx.float32
    assert calls["dequantize"] == 1

def test_deepseek_v4_non_q8_multi_row_takes_the_dequant_bridge(monkeypatch):
    # The defective kq.quantized_matmul returned wrong values when an
    # unevaluated row-strided activation view met any multi-row shape in
    # one call, which is exactly the shape the grouped output projection
    # produces at prefill and scorer widths on a non-q8_0 dense package
    # (measured rel error ~1.4 against the dequantized reference). On an
    # unverified kernel build (the strided-bulk probe reads False) bulk
    # multi-row calls take the dequant bridge, mirroring the q8_0 split;
    # single-row decode calls stay on the kernel.
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    import moespresso.runtime.deepseek_v4.model as ds4_model
    from moespresso.runtime.deepseek_v4.model import _kquant_matmul_ds4_fp32

    monkeypatch.setattr(ds4_model, "_DSV4_KQUANT_STRIDED_BULK_FIXED", False)
    bridge_before = ds4_model.kquant_bulk_route_call_counts()["bridge"]

    rng = np.random.default_rng(11)
    dense = mx.array(rng.standard_normal((16, 256), dtype=np.float32))
    wire, scales = kq.quantize(dense, "q6_k")
    mx.eval(wire, scales)
    calls = {"dequantize": 0, "quantized_matmul": 0}
    real_dequantize = kq.dequantize
    real_qmm = kq.quantized_matmul

    def spy_dequantize(*args, **kwargs):
        calls["dequantize"] += 1
        return real_dequantize(*args, **kwargs)

    def spy_qmm(*args, **kwargs):
        calls["quantized_matmul"] += 1
        return real_qmm(*args, **kwargs)

    spy_kq = SimpleNamespace(dequantize=spy_dequantize, quantized_matmul=spy_qmm)

    # The defect shape: a row-slice of the wire and a row-strided slice of
    # a grouped activation, multi-row.
    weight_view = wire[8:, :]
    grouped = mx.array(
        rng.standard_normal((1, 3, 2, 256), dtype=np.float32)
    ).astype(mx.bfloat16)
    x = grouped[:, :, 1, :]
    out = _kquant_matmul_ds4_fp32(
        x, weight_view, scales, "q6_k", mx=mx, kq=spy_kq)
    reference = mx.matmul(
        mx.contiguous(x).astype(mx.float32).reshape(-1, 256),
        real_dequantize(
            mx.contiguous(weight_view), scales, "q6_k", dtype=mx.float32).T,
    )
    mx.eval(out, reference)
    assert calls["dequantize"] == 1
    assert calls["quantized_matmul"] == 0
    assert (
        ds4_model.kquant_bulk_route_call_counts()["bridge"]
        == bridge_before + 1
    )
    np.testing.assert_allclose(
        np.asarray(out.reshape(-1, 8)), np.asarray(reference),
        rtol=2.0e-2, atol=2.0e-2)

    # Single-row decode calls keep the kernel.
    single = mx.array(
        rng.standard_normal((1, 1, 256), dtype=np.float32))
    out = _kquant_matmul_ds4_fp32(
        single, wire, scales, "q6_k", mx=mx, kq=spy_kq)
    mx.eval(out)
    assert calls["quantized_matmul"] == 1
    assert calls["dequantize"] == 1


def test_deepseek_v4_non_q8_multi_row_kernel_route_when_probe_verified(
    monkeypatch,
):
    # On a verified kernel build (the strided-bulk probe reads True) bulk
    # verify-shaped tiny multi-row calls go straight to kq.quantized_matmul
    # with no dequant bridge, while wider (prefill/scorer) calls keep the
    # bridge: the kernel emits on the bfloat16 lattice and routing every
    # width through it measured Q2 avg NLL 0.40094 against 0.39634 through
    # the bridge on the ship artifact. The operands here are dense (correct
    # on every build), so the value check holds regardless of which
    # mlx-kquant is installed; the bit-level strided pins live in the
    # mlx-kquant suite beside the fix.
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    import moespresso.runtime.deepseek_v4.model as ds4_model
    from moespresso.runtime.deepseek_v4.model import _kquant_matmul_ds4_fp32

    monkeypatch.setattr(ds4_model, "_DSV4_KQUANT_STRIDED_BULK_FIXED", True)
    counts_before = ds4_model.kquant_bulk_route_call_counts()

    rng = np.random.default_rng(11)
    dense = mx.array(rng.standard_normal((16, 256), dtype=np.float32))
    wire, scales = kq.quantize(dense, "q6_k")
    mx.eval(wire, scales)
    calls = {"dequantize": 0, "quantized_matmul": 0}
    real_dequantize = kq.dequantize
    real_qmm = kq.quantized_matmul

    def spy_dequantize(*args, **kwargs):
        calls["dequantize"] += 1
        return real_dequantize(*args, **kwargs)

    def spy_qmm(*args, **kwargs):
        calls["quantized_matmul"] += 1
        return real_qmm(*args, **kwargs)

    spy_kq = SimpleNamespace(dequantize=spy_dequantize, quantized_matmul=spy_qmm)

    x = mx.array(rng.standard_normal((1, 3, 256), dtype=np.float32)).astype(
        mx.bfloat16)
    mx.eval(x)
    out = _kquant_matmul_ds4_fp32(x, wire, scales, "q6_k", mx=mx, kq=spy_kq)
    reference = mx.matmul(
        x.astype(mx.float32).reshape(-1, 256),
        real_dequantize(wire, scales, "q6_k", dtype=mx.float32).T,
    )
    mx.eval(out, reference)
    assert calls["quantized_matmul"] == 1
    assert calls["dequantize"] == 0
    np.testing.assert_allclose(
        np.asarray(out.astype(mx.float32).reshape(-1, 16)),
        np.asarray(reference),
        rtol=2.0e-2, atol=2.0e-2)

    # Wider than the tiny-M cap: the bridge stays engaged under auto.
    wide = mx.array(rng.standard_normal((1, 16, 256), dtype=np.float32)).astype(
        mx.bfloat16)
    mx.eval(wide)
    out = _kquant_matmul_ds4_fp32(wide, wire, scales, "q6_k", mx=mx, kq=spy_kq)
    mx.eval(out)
    assert calls["quantized_matmul"] == 1
    assert calls["dequantize"] == 1
    counts_after = ds4_model.kquant_bulk_route_call_counts()
    assert counts_after["kernel"] == counts_before["kernel"] + 1
    assert counts_after["bridge"] == counts_before["bridge"] + 1


def test_kquant_bulk_route_counters_reach_all_three_census_surfaces(monkeypatch):
    """A counter on one surface defeats a census-gated instrument.

    The bulk-route arms read `kquant_bulk_route_call_counts`,
    `ssd_streaming_stats`, and the speed-stats count keys interchangeably,
    so a route counter that exists on only one of them reports no arm
    difference on the other two.
    """
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    import moespresso.runtime.deepseek_v4.model as ds4_model
    from moespresso.runtime.deepseek_v4.model import _kquant_matmul_ds4_fp32
    from moespresso.runtime.deepseek_v4.speed_stats import _COUNT_KEYS
    from moespresso.runtime.ssd_streaming_build import ssd_streaming_stats

    monkeypatch.setattr(ds4_model, "_DSV4_KQUANT_STRIDED_BULK_FIXED", True)
    before = ssd_streaming_stats(_WrappedModel([]))

    rng = np.random.default_rng(17)
    dense = mx.array(rng.standard_normal((16, 256), dtype=np.float32))
    wire, scales = kq.quantize(dense, "q6_k")
    # A verify-shaped tiny multi-row call takes the kernel route under auto;
    # a prefill-width call takes the dequant bridge. One of each moves both
    # counters, so neither pin below can pass on an untouched zero.
    tiny = mx.array(
        rng.standard_normal((1, 3, 256), dtype=np.float32)).astype(mx.bfloat16)
    wide = mx.array(
        rng.standard_normal((1, 16, 256), dtype=np.float32)).astype(mx.bfloat16)
    mx.eval(wire, scales, tiny, wide)
    mx.eval(_kquant_matmul_ds4_fp32(tiny, wire, scales, "q6_k", mx=mx, kq=kq))
    mx.eval(_kquant_matmul_ds4_fp32(wide, wire, scales, "q6_k", mx=mx, kq=kq))

    counts = ds4_model.kquant_bulk_route_call_counts()
    census = ssd_streaming_stats(_WrappedModel([]))
    for route, census_key in (
        ("kernel", "kquant_bulk_kernel_calls"),
        ("bridge", "kquant_bulk_bridge_calls"),
    ):
        assert census[census_key] == counts[route], census_key
        assert census_key in _COUNT_KEYS, census_key
        assert census[census_key] == before[census_key] + 1, census_key


def test_deepseek_v4_kquant_strided_bulk_probe_caches(monkeypatch):
    # The probe runs the recorded defect pair once against the installed
    # kernel, returns a bool, and caches the verdict for the process; a
    # second call must not touch the kernel again.
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    import moespresso.runtime.deepseek_v4.model as ds4_model

    monkeypatch.setattr(ds4_model, "_DSV4_KQUANT_STRIDED_BULK_FIXED", None)
    verdict = ds4_model._dsv4_kquant_strided_bulk_fixed(mx=mx, kq=kq)
    assert isinstance(verdict, bool)

    def explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("probe re-ran after caching")

    poisoned_kq = SimpleNamespace(quantize=explode, quantized_matmul=explode)
    assert (
        ds4_model._dsv4_kquant_strided_bulk_fixed(mx=mx, kq=poisoned_kq)
        is verdict
    )


def test_deepseek_v4_q8_tiny_m_qmm_engages_at_declared_site(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_q8_affine_views,
        _kquant_matmul_ds4_fp32,
        q8_dense_matmul_call_counts,
    )

    module = _real_q8_module(kq, mx, out_dims=32, in_dims=128, seed=51)
    affine = _deepseek_v4_q8_affine_views(module, mx=mx)
    x = mx.array(np.random.default_rng(52).standard_normal(
        (1, 6, 128), dtype=np.float32))

    # Tiny multi-row calls at the declared site take the affine QMM.
    before = q8_dense_matmul_call_counts()
    tiny_out = _kquant_matmul_ds4_fp32(
        x, module["weight"], module["scales"], "q8_0",
        mx=mx, kq=kq, affine=affine, wire_decode_site="wo_b",
        tiny_m_site="wo_b",
    )
    reference = mx.matmul(
        x.astype(mx.float32),
        kq.dequantize(
            module["weight"], module["scales"], "q8_0", dtype=mx.float32).T,
    )
    mx.eval(tiny_out, reference)
    after = q8_dense_matmul_call_counts()
    assert after["tiny_m_qmm_wo_b"] == before["tiny_m_qmm_wo_b"] + 1
    assert after["prefill_dequant"] == before["prefill_dequant"]
    assert tiny_out.dtype == mx.float32
    # The affine repack reproduces the q8_0 lattice exactly, so the route
    # differs from the dequant bridge only in float32 accumulation order.
    np.testing.assert_allclose(
        np.asarray(tiny_out), np.asarray(reference), rtol=1.0e-5, atol=1.0e-5)

    # Undeclared sites and the operator-controlled cap keep the dequant bridge.
    for kwargs, env_max in (
        (dict(), None),
        (dict(tiny_m_site="wq_b"), None),
        (dict(tiny_m_site="wo_b"), "0"),
    ):
        if env_max is not None:
            monkeypatch.setenv("MOESPRESSO_DSV4_Q8_TINY_M_MAX", env_max)
        else:
            monkeypatch.delenv("MOESPRESSO_DSV4_Q8_TINY_M_MAX", raising=False)
        before = q8_dense_matmul_call_counts()
        out = _kquant_matmul_ds4_fp32(
            x, module["weight"], module["scales"], "q8_0",
            mx=mx, kq=kq, affine=affine, **kwargs,
        )
        mx.eval(out)
        after = q8_dense_matmul_call_counts()
        assert after["prefill_dequant"] == before["prefill_dequant"] + 1
        assert after["tiny_m_qmm_wo_b"] == before["tiny_m_qmm_wo_b"]
    wide = mx.array(np.random.default_rng(53).standard_normal(
        (1, 9, 128), dtype=np.float32))
    before = q8_dense_matmul_call_counts()
    out = _kquant_matmul_ds4_fp32(
        wide, module["weight"], module["scales"], "q8_0",
        mx=mx, kq=kq, affine=affine, tiny_m_site="wo_b",
    )
    mx.eval(out)
    after = q8_dense_matmul_call_counts()
    assert after["prefill_dequant"] == before["prefill_dequant"] + 1
    assert after["tiny_m_qmm_wo_b"] == before["tiny_m_qmm_wo_b"]


@pytest.mark.parametrize(
    ("site", "in_dims"),
    [("wo_b", 8192), ("lm_head", 4096)],
)
def test_deepseek_v4_q8_wire_decode_qmv_engages_and_bounds_drift(
        site, in_dims):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_q8_affine_views,
        _kquant_matmul_ds4_fp32,
        q8_dense_matmul_call_counts,
    )

    # Served reduction lengths (wo_b K=8192, lm_head K=4096) with a
    # reduced output width; the drift mechanism scales with K independently of
    # the output count.
    module = _real_q8_module(kq, mx, out_dims=256, in_dims=in_dims, seed=11)
    affine = _deepseek_v4_q8_affine_views(module, mx=mx)
    x = mx.array(np.random.default_rng(12).standard_normal(
        (1, 1, in_dims), dtype=np.float32))

    before = q8_dense_matmul_call_counts()
    wire_out = _kquant_matmul_ds4_fp32(
        x, module["weight"], module["scales"], "q8_0",
        mx=mx, kq=kq, affine=affine, wire_decode_site=site,
    )
    affine_out = _kquant_matmul_ds4_fp32(
        x, module["weight"], module["scales"], "q8_0",
        mx=mx, kq=kq, affine=affine,
    )
    mx.eval(wire_out, affine_out)
    after = q8_dense_matmul_call_counts()

    key = "decode_wire_qmv_" + site
    assert after[key] == before[key] + 1
    assert after["decode_qmv"] == before["decode_qmv"] + 1
    assert wire_out.dtype == mx.float32

    # The wire kernel pre-rounds the float32 activation to bfloat16 and
    # accumulates on the bfloat16 lattice (about 2^-9 relative per
    # element, growing with the root of the reduction length), so the
    # routes must differ but stay drift-bounded. Measured at these shapes
    # on standard normal data: 0.0018-0.0028 of the output scale; the
    # served drifts are 0.021 max-abs at wo_b and 0.127 at lm_head
    # logits. The 0.01 relative bound carries better than 3x margin.
    diff = float(mx.max(mx.abs(wire_out - affine_out)).item())
    scale = float(mx.max(mx.abs(affine_out)).item())
    assert diff > 0.0
    assert diff <= 0.01 * scale


def test_deepseek_v4_q8_wire_decode_eligibility_fails_closed():
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_q8_affine_views,
        _deepseek_v4_q8_wire_decode_eligible,
        _kquant_matmul_ds4_fp32,
        q8_dense_matmul_call_counts,
    )

    module = _real_q8_module(kq, mx, out_dims=32, in_dims=128, seed=15)
    affine = _deepseek_v4_q8_affine_views(module, mx=mx)
    x = mx.array(np.random.default_rng(16).standard_normal(
        (1, 1, 128), dtype=np.float32))

    # The eligibility helper: only the two known sites, floating
    # activations, and q8_0 wire shaped weights qualify.
    wire = module["weight"]
    assert _deepseek_v4_q8_wire_decode_eligible(x, wire, "wo_b", mx=mx)
    assert _deepseek_v4_q8_wire_decode_eligible(x, wire, "lm_head", mx=mx)
    assert not _deepseek_v4_q8_wire_decode_eligible(x, wire, "wq_b", mx=mx)
    assert not _deepseek_v4_q8_wire_decode_eligible(x, wire, None, mx=mx)
    assert not _deepseek_v4_q8_wire_decode_eligible(
        x.astype(mx.uint32), wire, "wo_b", mx=mx)
    assert not _deepseek_v4_q8_wire_decode_eligible(
        x, mx.zeros((32, 136), dtype=mx.float32), "wo_b", mx=mx)
    assert not _deepseek_v4_q8_wire_decode_eligible(
        x, mx.zeros((32, 100), dtype=mx.uint8), "wo_b", mx=mx)
    assert not _deepseek_v4_q8_wire_decode_eligible(
        x, mx.zeros((32, 4, 34), dtype=mx.uint8), "wo_b", mx=mx)

    # Multi-row calls keep the prefill dequant bridge even with a site.
    before = q8_dense_matmul_call_counts()
    out = _kquant_matmul_ds4_fp32(
        mx.zeros((1, 3, 128), dtype=mx.float16),
        module["weight"], module["scales"], "q8_0",
        mx=mx, kq=kq, affine=affine, wire_decode_site="wo_b",
    )
    mx.eval(out)
    after = q8_dense_matmul_call_counts()
    assert after["prefill_dequant"] == before["prefill_dequant"] + 1
    assert after["decode_wire_qmv_wo_b"] == before["decode_wire_qmv_wo_b"]

    # An unknown site label falls back to the affine route.
    before = q8_dense_matmul_call_counts()
    out = _kquant_matmul_ds4_fp32(
        x, module["weight"], module["scales"], "q8_0",
        mx=mx, kq=kq, affine=affine, wire_decode_site="wq_b",
    )
    mx.eval(out)
    after = q8_dense_matmul_call_counts()
    assert after["decode_qmv"] == before["decode_qmv"] + 1
    assert after == dict(before, decode_qmv=before["decode_qmv"] + 1)

    # A non-wire weight tensor falls back to the affine route even with a
    # valid site (the affine views carry the math; the wire route must
    # never touch a placeholder).
    before = q8_dense_matmul_call_counts()
    out = _kquant_matmul_ds4_fp32(
        x, mx.zeros((32, 128), dtype=mx.float32), module["scales"], "q8_0",
        mx=mx, kq=kq, affine=affine, wire_decode_site="wo_b",
    )
    mx.eval(out)
    after = q8_dense_matmul_call_counts()
    assert after["decode_qmv"] == before["decode_qmv"] + 1
    assert after["decode_wire_qmv_wo_b"] == before["decode_wire_qmv_wo_b"]


def test_deepseek_v4_q8_wire_decode_routes_wo_b_and_lm_head_sites():
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        q8_dense_matmul_call_counts,
    )

    # wo_b: the grouped-projection patch wraps wo_b with the fp32 bridge,
    # whose decode rows land on the wo_b wire counter.
    attn = _real_wo_a_attention_fixture(kq, mx)
    attn.wo_b = _real_q8_module(kq, mx, out_dims=16, in_dims=64, seed=17)
    model = _WrappedModel([_AttentionLayer(attn)])
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    before = q8_dense_matmul_call_counts()
    out = attn.wo_b(mx.zeros((1, 1, 64), dtype=mx.float32))
    mx.eval(out)
    after = q8_dense_matmul_call_counts()
    assert after["decode_wire_qmv_wo_b"] == (
        before["decode_wire_qmv_wo_b"] + 1)
    assert after["decode_wire_qmv_lm_head"] == (
        before["decode_wire_qmv_lm_head"])

    # lm_head: the patched fp32 logits path routes its single decode row
    # through the lm_head wire counter; scorer-shaped no-cache calls stay
    # on the multi-row dequant bridge.
    hidden = mx.array(np.random.default_rng(18).standard_normal(
        (1, 1, 64), dtype=np.float32))
    lm_model = _lm_head_model_fixture(mx, hidden)
    lm_model.lm_head = _real_q8_module(kq, mx, out_dims=16, in_dims=64,
                                       seed=19)
    assert _patch_deepseek_v4_kquant_lm_head(lm_model) is True
    before = q8_dense_matmul_call_counts()
    out = lm_model(mx.array([[1]], dtype=mx.int32))
    mx.eval(out)
    after = q8_dense_matmul_call_counts()
    assert after["decode_wire_qmv_lm_head"] == (
        before["decode_wire_qmv_lm_head"] + 1)

    scorer_hidden = mx.array(np.random.default_rng(20).standard_normal(
        (1, 3, 64), dtype=np.float32))
    scorer_model = _lm_head_model_fixture(mx, scorer_hidden)
    scorer_model.lm_head = lm_model.lm_head
    assert _patch_deepseek_v4_kquant_lm_head(scorer_model) is True
    before = q8_dense_matmul_call_counts()
    out = scorer_model(mx.array([[1, 2, 3]], dtype=mx.int32))
    mx.eval(out)
    after = q8_dense_matmul_call_counts()
    assert after["prefill_dequant"] == before["prefill_dequant"] + 1
    assert after["decode_wire_qmv_lm_head"] == (
        before["decode_wire_qmv_lm_head"])


def _real_wo_a_attention_fixture(kq, mx, *, groups=2, rank=4, group_feat=64,
                                 seed=23):
    attn = _KQuantAttention(
        _real_q8_module(
            kq, mx, out_dims=groups * rank, in_dims=group_feat, seed=seed),
        None,
    )
    # Geometry consistent with the grouped projection contract:
    # n_heads * head_dim == groups * group_feat.
    attn.n_heads = 4
    attn.head_dim = (groups * group_feat) // attn.n_heads
    attn.o_groups = groups
    attn.o_lora_rank = rank
    return attn


def test_deepseek_v4_wo_a_multi_row_routes_by_tiny_m_cap(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        wo_a_projection_call_counts,
    )

    attn = _real_wo_a_attention_fixture(kq, mx)
    model = _WrappedModel([_AttentionLayer(attn)])
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1

    # Tiny multi-row (verify-shaped) calls take the batched tiny-M form;
    # the decode single-dispatch forms stay row-one only.
    before = wo_a_projection_call_counts()
    out = attn._grouped_output_projection(
        mx.zeros((1, 3, 128), dtype=mx.bfloat16))
    mx.eval(out)
    after = wo_a_projection_call_counts()
    assert after["batched_tiny_m"] == before["batched_tiny_m"] + 1
    assert after["loop"] == before["loop"]
    assert after["batched_decode"] == before["batched_decode"]
    assert after["gather_decode"] == before["gather_decode"]

    # Rows past the cap take the per-group loop (bulk prefill contract).
    before = after
    out = attn._grouped_output_projection(
        mx.zeros((1, 9, 128), dtype=mx.bfloat16))
    mx.eval(out)
    after = wo_a_projection_call_counts()
    assert after["loop"] == before["loop"] + 1
    assert after["batched_tiny_m"] == before["batched_tiny_m"]

    # The tiny-M cap remains a live operator control.
    monkeypatch.setenv("MOESPRESSO_DSV4_Q8_TINY_M_MAX", "0")
    before = after
    out = attn._grouped_output_projection(
        mx.zeros((1, 3, 128), dtype=mx.bfloat16))
    mx.eval(out)
    after = wo_a_projection_call_counts()
    assert after["loop"] == before["loop"] + 1
    assert after["batched_tiny_m"] == before["batched_tiny_m"]

def test_deepseek_v4_wo_a_tiny_m_matches_loop_bridge(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        wo_a_projection_call_counts,
    )

    attn = _real_wo_a_attention_fixture(kq, mx, seed=61)
    model = _WrappedModel([_AttentionLayer(attn)])
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    rng = np.random.default_rng(62)
    hidden = mx.array(rng.standard_normal((1, 6, 128), dtype=np.float32))

    # Reference arm: the per-group loop on the float32 dequant bridge.
    before = wo_a_projection_call_counts()
    monkeypatch.setenv("MOESPRESSO_DSV4_Q8_TINY_M_MAX", "0")
    reference = attn._grouped_output_projection(hidden)
    mx.eval(reference)
    monkeypatch.delenv("MOESPRESSO_DSV4_Q8_TINY_M_MAX", raising=False)
    after_loop = wo_a_projection_call_counts()
    assert after_loop["loop"] == before["loop"] + 1

    tiny_out = attn._grouped_output_projection(hidden)
    mx.eval(tiny_out)
    after_tiny = wo_a_projection_call_counts()
    assert after_tiny["batched_tiny_m"] == after_loop["batched_tiny_m"] + 1

    assert tiny_out.dtype == mx.float32
    assert tiny_out.shape == reference.shape
    # The affine repack reproduces the q8_0 lattice exactly, so the tiny
    # form differs from the loop's dequant bridge only in float32
    # accumulation order.
    np.testing.assert_allclose(
        np.asarray(tiny_out), np.asarray(reference), rtol=1.0e-5, atol=1.0e-5)


def test_deepseek_v4_wo_a_gather_decode_engages_and_bounds_drift(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    if not hasattr(kq, "gather_qmv_kq"):
        pytest.skip("mlx_kquant build lacks gather_qmv_kq")
    from moespresso.runtime.deepseek_v4.model import (
        wo_a_projection_call_counts,
    )

    # gather_qmv_kq requires the reduced dimension in whole 256-wide
    # tiles and rank in whole rows of 8; the batched affine form is the
    # exact reference arm.
    attn = _real_wo_a_attention_fixture(
        kq, mx, group_feat=256, rank=8, seed=41)
    model = _WrappedModel([_AttentionLayer(attn)])
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    hidden = mx.array(np.random.default_rng(42).standard_normal(
        (1, 1, 512), dtype=np.float32))

    # Reference arm through the fail-closed path: an ineligible gather
    # stack lands the call on the batched affine form.
    monkeypatch.setattr(
        dsv4_runtime, "_deepseek_v4_q8_wire_gather_stack",
        lambda *args, **kwargs: None)
    before = wo_a_projection_call_counts()
    affine_out = attn._grouped_output_projection(hidden)
    mx.eval(affine_out)
    after_affine = wo_a_projection_call_counts()
    assert after_affine["batched_decode"] == before["batched_decode"] + 1
    assert after_affine["gather_decode"] == before["gather_decode"]

    monkeypatch.undo()
    gather_out = attn._grouped_output_projection(hidden)
    mx.eval(gather_out)
    after_gather = wo_a_projection_call_counts()
    assert after_gather["gather_decode"] == after_affine["gather_decode"] + 1
    assert after_gather["batched_decode"] == after_affine["batched_decode"]

    assert gather_out.dtype == mx.float32
    assert gather_out.shape == affine_out.shape
    # The gather kernel accumulates on the bfloat16 activation lattice;
    # the 0.01 relative bound matches the wire decode engagement test.
    diff = float(mx.max(mx.abs(gather_out - affine_out)).item())
    scale = float(mx.max(mx.abs(affine_out)).item())
    assert diff > 0.0
    assert diff <= 0.01 * scale


def test_deepseek_v4_wo_a_gather_decode_fails_closed():
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    if not hasattr(kq, "gather_qmv_kq"):
        pytest.skip("mlx_kquant build lacks gather_qmv_kq")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_q8_wire_gather_stack,
        wo_a_projection_call_counts,
    )

    attn = _real_wo_a_attention_fixture(
        kq, mx, group_feat=256, rank=8, seed=43)

    # Geometry outside the gather contract fails closed to the batched
    # affine form: a 64-wide group_feat is not tileable by 256.
    narrow = _real_wo_a_attention_fixture(kq, mx, group_feat=64, seed=44)
    narrow_model = _WrappedModel([_AttentionLayer(narrow)])
    assert _patch_deepseek_v4_kquant_grouped_output_projection(
        narrow_model) == 1
    assert _deepseek_v4_q8_wire_gather_stack(
        narrow.wo_a, groups=2, rank=4, group_feat=64, mx=mx) is None
    before = wo_a_projection_call_counts()
    out = narrow._grouped_output_projection(
        mx.zeros((1, 1, 128), dtype=mx.float32))
    mx.eval(out)
    after = wo_a_projection_call_counts()
    assert after["batched_decode"] == before["batched_decode"] + 1
    assert after["gather_decode"] == before["gather_decode"]

    # Row-count mismatches fail closed at the stack helper.
    assert _deepseek_v4_q8_wire_gather_stack(
        attn.wo_a, groups=4, rank=8, group_feat=256, mx=mx) is None


def test_deepseek_v4_q8_affine_views_defer_until_fallback(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    if not hasattr(kq, "gather_qmv_kq"):
        pytest.skip("mlx_kquant build lacks gather_qmv_kq")

    # Default routes (gather at wo_a, wire QMV at wo_b) never consume the
    # affine repack, so it must not materialize; an ineligible gather
    # stack builds it on demand for the batched affine fallback.
    attn = _real_wo_a_attention_fixture(
        kq, mx, group_feat=256, rank=8, seed=47)
    attn.wo_b = _real_q8_module(kq, mx, out_dims=16, in_dims=64, seed=48)
    model = _WrappedModel([_AttentionLayer(attn)])
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1
    hidden = mx.array(np.random.default_rng(49).standard_normal(
        (1, 1, 512), dtype=np.float32))

    gather_out = attn._grouped_output_projection(hidden)
    wob_out = attn.wo_b(mx.zeros((1, 1, 64), dtype=mx.float32))
    mx.eval(gather_out, wob_out)
    assert getattr(attn.wo_a, "_moespresso_dsv4_q8_affine", None) is None
    assert getattr(
        attn.wo_b.original, "_moespresso_dsv4_q8_affine", None) is None

    monkeypatch.setattr(
        dsv4_runtime, "_deepseek_v4_q8_wire_gather_stack",
        lambda *args, **kwargs: None)
    affine_out = attn._grouped_output_projection(hidden)
    mx.eval(affine_out)
    assert getattr(attn.wo_a, "_moespresso_dsv4_q8_affine", None) is not None
    diff = float(mx.max(mx.abs(gather_out - affine_out)).item())
    scale = float(mx.max(mx.abs(affine_out)).item())
    assert diff <= 0.01 * scale


def test_deepseek_v4_wo_a_batched_decode_bit_identical_per_group():
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_q8_affine_views,
        _kquant_matmul_ds4_fp32,
        wo_a_projection_call_counts,
    )

    attn = _real_wo_a_attention_fixture(kq, mx)
    model = _WrappedModel([_AttentionLayer(attn)])
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1

    rng = np.random.default_rng(41)
    hidden = mx.array(
        rng.standard_normal((1, 1, 128), dtype=np.float32)
    ).astype(mx.bfloat16)

    groups = int(attn.o_groups)
    rank = int(attn.o_lora_rank)
    group_feat = (attn.n_heads * attn.head_dim) // groups
    grouped = hidden.reshape(1, 1, groups, group_feat)
    affine = _deepseek_v4_q8_affine_views(attn.wo_a, mx=mx)
    assert affine is not None
    pieces = []
    for group in range(groups):
        row_start = group * rank
        row_end = row_start + rank
        pieces.append(_kquant_matmul_ds4_fp32(
            grouped[:, :, group, :],
            attn.wo_a["weight"][row_start:row_end],
            attn.wo_a["scales"],
            attn.wo_a.kquant_type,
            mx=mx,
            kq=kq,
            affine=tuple(
                tensor[row_start:row_end] for tensor in affine),
        ))
    loop_out = mx.concatenate(pieces, axis=-1)
    mx.eval(loop_out)

    before = wo_a_projection_call_counts()
    batched_out = attn._grouped_output_projection(hidden)
    mx.eval(batched_out)
    after_batched = wo_a_projection_call_counts()
    assert after_batched["batched_decode"] == before["batched_decode"] + 1

    assert batched_out.dtype == mx.float32
    loop_bits = np.asarray(loop_out).reshape(groups, rank).view(np.uint32)
    batched_bits = np.asarray(batched_out).reshape(
        groups, rank).view(np.uint32)
    for group in range(groups):
        np.testing.assert_array_equal(
            loop_bits[group], batched_bits[group],
            err_msg=f"group {group} outputs diverge from the slice loop")


def test_deepseek_v4_wo_a_batched_decode_single_dispatch_and_fail_closed(
        monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    from moespresso.runtime.deepseek_v4.model import (
        _deepseek_v4_q8_affine_group_views,
    )

    attn = _real_wo_a_attention_fixture(kq, mx)
    model = _WrappedModel([_AttentionLayer(attn)])
    assert _patch_deepseek_v4_kquant_grouped_output_projection(model) == 1

    calls = {"quantized_matmul": 0}
    real_quantized_matmul = mx.quantized_matmul

    def spy_quantized_matmul(*args, **kwargs):
        calls["quantized_matmul"] += 1
        return real_quantized_matmul(*args, **kwargs)

    monkeypatch.setattr(mx, "quantized_matmul", spy_quantized_matmul)
    hidden = mx.zeros((1, 1, 128), dtype=mx.bfloat16)

    # The eligible decode form issues one QMV dispatch for the whole projection.
    out = attn._grouped_output_projection(hidden)
    mx.eval(out)
    assert calls["quantized_matmul"] == 1

    # Geometry mismatches fail closed to None instead of a bogus stack.
    wo_a = attn.wo_a
    groups = int(attn.o_groups)
    rank = int(attn.o_lora_rank)
    assert _deepseek_v4_q8_affine_group_views(
        wo_a, groups=groups, rank=rank + 1, group_feat=64, mx=mx) is None
    assert _deepseek_v4_q8_affine_group_views(
        wo_a, groups=groups, rank=rank, group_feat=32, mx=mx) is None
    views = _deepseek_v4_q8_affine_group_views(
        wo_a, groups=groups, rank=rank, group_feat=64, mx=mx)
    assert views is not None
    # Cached on the module after the first build.
    assert _deepseek_v4_q8_affine_group_views(
        wo_a, groups=groups, rank=rank, group_feat=64, mx=mx) is views


class _AffineAttention:
    """DS4 attention stub with affine-quantized wo modules.

    `_grouped_output_projection` transcribes the stock QuantizedLinear
    branch (the float16-seam path the fp32 patch replaces), so delegation
    tests compare against the exact stock math.
    """

    def __init__(self, mx, nn, *, groups, rank, group_feat, hidden,
                 bits=8, group_size=32, seed=7):
        rng = np.random.default_rng(seed)
        self.n_heads = 4
        self.head_dim = (groups * group_feat) // self.n_heads
        self.o_groups = groups
        self.o_lora_rank = rank
        wo_a = nn.Linear(group_feat, groups * rank, bias=False)
        wo_a.weight = mx.array(rng.standard_normal(
            (groups * rank, group_feat), dtype=np.float32))
        self.wo_a = wo_a.to_quantized(group_size=group_size, bits=bits)
        wo_b = nn.Linear(groups * rank, hidden, bias=False)
        wo_b.weight = mx.array(rng.standard_normal(
            (hidden, groups * rank), dtype=np.float32))
        self.wo_b = wo_b.to_quantized(group_size=group_size, bits=bits)
        # Package affine scales and biases are stored float16; the stock
        # QMM output dtype follows them, which is the float16 seam under
        # test.
        for module in (self.wo_a, self.wo_b):
            module.scales = module.scales.astype(mx.float16)
            module.biases = module.biases.astype(mx.float16)

    def _grouped_output_projection(self, out):
        import mlx.core as mx

        bsz, length = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.o_groups
        out = out.reshape(bsz, length, self.o_groups, group_feat)
        out = out.transpose(2, 0, 1, 3)
        weight = self.wo_a.weight.reshape(
            self.o_groups, self.o_lora_rank, -1)[:, None]
        scales = self.wo_a.scales.reshape(
            self.o_groups, self.o_lora_rank, -1)[:, None]
        biases = self.wo_a.biases.reshape(
            self.o_groups, self.o_lora_rank, -1)[:, None]
        out = mx.quantized_matmul(
            out, weight, scales=scales, biases=biases, transpose=True,
            group_size=self.wo_a.group_size, bits=self.wo_a.bits,
            mode="affine",
        )
        return out.transpose(1, 2, 0, 3).reshape(
            bsz, length, self.o_groups * self.o_lora_rank)


def _affine_wo_attention_fixture(mx, nn, *, served_shapes=False, seed=7):
    if served_shapes:
        # Served DS4 wo geometry: 8 groups, rank 1024, 4096 features per
        # group (n_heads * head_dim == 32768), hidden 4096.
        return _AffineAttention(
            mx, nn, groups=8, rank=1024, group_feat=4096, hidden=4096,
            seed=seed)
    return _AffineAttention(
        mx, nn, groups=2, rank=16, group_feat=64, hidden=32, seed=seed)


def _dequantized_f64(module, mx):
    # The QMM kernel dequantizes in registers at float32, so the reference
    # lattice is the float32 dequant of the stored float16 scales/biases,
    # not a float16-rounded dequant.
    return np.asarray(mx.dequantize(
        module.weight,
        module.scales.astype(mx.float32),
        module.biases.astype(mx.float32),
        group_size=module.group_size,
        bits=module.bits,
    )).astype(np.float64)


def _rel_err(actual, reference):
    return float(
        np.linalg.norm(actual.astype(np.float64) - reference)
        / np.linalg.norm(reference))


def test_deepseek_v4_affine_wo_fp32_contract_parity_vs_f64_reference():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from moespresso.runtime.deepseek_v4.model import (
        affine_wo_fp32_call_counts,
    )

    attn = _affine_wo_attention_fixture(mx, nn, served_shapes=True)
    stock_projection = attn._grouped_output_projection
    model = _WrappedModel([_AttentionLayer(attn)])
    assert _patch_deepseek_v4_affine_wo_fp32(model) == 1

    rng = np.random.default_rng(53)
    groups = attn.o_groups
    rank = attn.o_lora_rank
    group_feat = (attn.n_heads * attn.head_dim) // groups
    wo_a_f64 = _dequantized_f64(attn.wo_a, mx).reshape(
        groups, rank, group_feat)
    wo_b_f64 = _dequantized_f64(attn.wo_b.original, mx)

    for rows in (1, 4):
        x = mx.array(rng.standard_normal(
            (1, rows, groups * group_feat), dtype=np.float32)
        ).astype(mx.float16)

        before = affine_wo_fp32_call_counts()
        fp32_out = attn._grouped_output_projection(x)
        mx.eval(fp32_out)
        after = affine_wo_fp32_call_counts()
        assert after["wo_a"] == before["wo_a"] + 1
        assert fp32_out.dtype == mx.float32

        stock_out = stock_projection(x)
        mx.eval(stock_out)
        assert stock_out.dtype == mx.float16

        x_f64 = np.asarray(x.astype(mx.float32)).astype(np.float64).reshape(
            rows, groups, group_feat)
        ref = np.concatenate(
            [x_f64[:, g, :] @ wo_a_f64[g].T for g in range(groups)],
            axis=-1,
        ).reshape(1, rows, groups * rank)
        fp32_err = _rel_err(np.asarray(fp32_out), ref)
        stock_err = _rel_err(np.asarray(stock_out.astype(mx.float32)), ref)
        assert fp32_err < 1e-5, f"rows={rows}: fp32 seam error {fp32_err}"
        assert stock_err > 1e-4, f"rows={rows}: stock f16 error {stock_err}"
        assert fp32_err < stock_err

        # wo_b seam: the wrapper takes the projection output.
        y = mx.array(rng.standard_normal(
            (1, rows, groups * rank), dtype=np.float32)).astype(mx.float16)
        before = affine_wo_fp32_call_counts()
        fp32_b = attn.wo_b(y)
        mx.eval(fp32_b)
        after = affine_wo_fp32_call_counts()
        assert after["wo_b"] == before["wo_b"] + 1
        assert fp32_b.dtype == mx.float32

        stock_b = attn.wo_b.original(y)
        mx.eval(stock_b)
        assert stock_b.dtype == mx.float16

        y_f64 = np.asarray(y.astype(mx.float32)).astype(np.float64)
        ref_b = y_f64 @ wo_b_f64.T
        fp32_b_err = _rel_err(np.asarray(fp32_b), ref_b)
        stock_b_err = _rel_err(np.asarray(stock_b.astype(mx.float32)), ref_b)
        assert fp32_b_err < 1e-5, f"rows={rows}: wo_b fp32 error {fp32_b_err}"
        assert stock_b_err > 1e-4, f"rows={rows}: wo_b f16 error {stock_b_err}"
        assert fp32_b_err < stock_b_err


def test_deepseek_v4_affine_wo_fp32_fail_closed_eligibility():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    # K-quant wo modules are not QuantizedLinear; the kquant bridge owns
    # them and the affine patch must not touch the layer.
    kquant_attn = _KQuantAttention(
        _KQuantModule(mx.zeros((2, 4)), mx.zeros((1,), dtype=mx.uint8)),
        _KQuantModule(mx.zeros((2, 4)), mx.zeros((1,), dtype=mx.uint8)),
    )
    assert _patch_deepseek_v4_affine_wo_fp32(
        _WrappedModel([_AttentionLayer(kquant_attn)])) == 0
    assert not isinstance(kquant_attn.wo_b, nn.Module)

    # Plain (unquantized) wo modules keep the stock path.
    plain_attn = _AffineAttention(
        mx, nn, groups=2, rank=16, group_feat=64, hidden=32)
    plain_attn.wo_a = nn.Linear(64, 8, bias=False)
    plain_attn.wo_b = nn.Linear(8, 32, bias=False)
    plain_wo_b = plain_attn.wo_b
    assert _patch_deepseek_v4_affine_wo_fp32(
        _WrappedModel([_AttentionLayer(plain_attn)])) == 0
    assert plain_attn.wo_b is plain_wo_b

    # A non-affine QuantizedLinear mode is ineligible on both modules.
    mx_attn = _AffineAttention(
        mx, nn, groups=2, rank=16, group_feat=64, hidden=32)
    mx_attn.wo_a = nn.QuantizedLinear(64, 32, bias=False, mode="mxfp8")
    mx_attn.wo_b = nn.QuantizedLinear(64, 32, bias=False, mode="mxfp8")
    mx_wo_b = mx_attn.wo_b
    assert _patch_deepseek_v4_affine_wo_fp32(
        _WrappedModel([_AttentionLayer(mx_attn)])) == 0
    assert mx_attn.wo_b is mx_wo_b

    # A wo_a geometry mismatch keeps the stock projection while the
    # independent wo_b seam still gets the fp32 contract.
    geo_attn = _AffineAttention(
        mx, nn, groups=2, rank=16, group_feat=64, hidden=32)
    geo_attn.o_lora_rank = 15
    stock = geo_attn._grouped_output_projection
    assert _patch_deepseek_v4_affine_wo_fp32(
        _WrappedModel([_AttentionLayer(geo_attn)])) == 1
    assert geo_attn._grouped_output_projection.__func__ is stock.__func__
    assert geo_attn.wo_b.original is not None

    # Idempotent: a second pass finds the marker and rewraps nothing.
    attn = _affine_wo_attention_fixture(mx, nn)
    model = _WrappedModel([_AttentionLayer(attn)])
    assert _patch_deepseek_v4_affine_wo_fp32(model) == 1
    wrapped = attn.wo_b
    assert _patch_deepseek_v4_affine_wo_fp32(model) == 0
    assert attn.wo_b is wrapped
    assert not hasattr(wrapped.original, "original")


def test_deepseek_v4_prefill_candidate_counters_export_via_streaming_stats():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_kquant")
    from moespresso.runtime.ssd_streaming_build import ssd_streaming_stats
    from moespresso.runtime.deepseek_v4.indexed_attention_kernel import (
        indexer_scores_call_counts,
    )
    from moespresso.runtime.deepseek_v4.model import (
        affine_wo_fp32_call_counts,
        banded_prefill_call_counts,
        q8_dense_matmul_call_counts,
        q8_ffn_hc_post_call_counts,
        q8_hc_post_call_counts,
        wo_a_projection_call_counts,
    )

    stats = ssd_streaming_stats(_WrappedModel([]))

    scores_counts = indexer_scores_call_counts()
    wo_a_counts = wo_a_projection_call_counts()
    banded_counts = banded_prefill_call_counts()
    q8_dense_counts = q8_dense_matmul_call_counts()
    q8_hc_post_counts = q8_hc_post_call_counts()
    q8_ffn_hc_post_counts = q8_ffn_hc_post_call_counts()
    affine_wo_counts = affine_wo_fp32_call_counts()
    assert stats["r4_prefill_scores_f16_calls"] == scores_counts["f16"]
    assert stats["r4_prefill_scores_f32_calls"] == scores_counts["f32"]
    assert stats["wo_a_batched_decode_calls"] == wo_a_counts["batched_decode"]
    assert stats["wo_a_loop_projection_calls"] == wo_a_counts["loop"]
    assert stats["banded_prefill_mma_calls"] == banded_counts["mma"]
    assert stats["banded_prefill_sdpa_calls"] == banded_counts["sdpa"]
    assert stats["banded_prefill_mma_offset_calls"] == (
        banded_counts["mma_offset"])
    assert stats["banded_prefill_composed_offset_calls"] == (
        banded_counts["composed_offset"])
    assert stats["q8_dense_decode_qmv_calls"] == q8_dense_counts["decode_qmv"]
    assert stats["q8_dense_decode_wire_qmv_wo_b_calls"] == (
        q8_dense_counts["decode_wire_qmv_wo_b"])
    assert stats["q8_dense_decode_wire_qmv_lm_head_calls"] == (
        q8_dense_counts["decode_wire_qmv_lm_head"])
    assert stats["q8_dense_prefill_dequant_calls"] == (
        q8_dense_counts["prefill_dequant"])
    assert stats["q8_hc_post_engaged_calls"] == q8_hc_post_counts["engaged"]
    assert stats["q8_hc_post_fallback_calls"] == q8_hc_post_counts["fallback"]
    assert stats["q8_hc_post_delegated_calls"] == (
        q8_hc_post_counts["delegated"])
    assert stats["q8_ffn_hc_post_engaged_calls"] == (
        q8_ffn_hc_post_counts["engaged"])
    assert stats["q8_ffn_hc_post_fallback_calls"] == (
        q8_ffn_hc_post_counts["fallback"])
    assert stats["q8_ffn_hc_post_delegated_calls"] == (
        q8_ffn_hc_post_counts["delegated"])
    assert stats["affine_wo_fp32_wo_a_calls"] == affine_wo_counts["wo_a"]
    assert stats["affine_wo_fp32_wo_b_calls"] == affine_wo_counts["wo_b"]


def _lm_head_model_fixture(mx, hidden):
    class _Body:
        def __call__(self, input_ids, cache=None, mask=None):
            del input_ids, cache, mask
            return hidden

    class _LMHeadModel:
        def __init__(self):
            self.model = _Body()
            self.lm_head = _KQuantModule(
                mx.array([[5.0, 7.0], [11.0, 13.0]], dtype=mx.float32),
                mx.zeros((1,), dtype=mx.uint8),
            )

        def __call__(self, input_ids, cache=None, mask=None):
            del input_ids, cache, mask
            raise AssertionError("patched K-quant lm_head should replace __call__")

    return _LMHeadModel()


def test_deepseek_v4_kquant_lm_head_uses_fp32_q8_bridge(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    calls = {"dequantize": 0, "quantized_matmul": 0}
    _fake_q8_kquant_module(monkeypatch, calls)

    # Two token rows without a cache: scorer-shaped calls (Q1/Q2 teacher
    # forcing) keep every row on the float32 bridge.
    hidden = mx.array([[[2.0, 3.0], [1.0, 1.0]]], dtype=mx.float16)
    model = _lm_head_model_fixture(mx, hidden)

    patched = _patch_deepseek_v4_kquant_lm_head(model)

    assert patched is True
    assert model._moespresso_dsv4_kquant_lm_head is True
    out = model(mx.array([[1, 2]], dtype=mx.int32))
    mx.eval(out)

    assert out.dtype == mx.float32
    np.testing.assert_allclose(
        np.asarray(out), [[[31.0, 61.0], [12.0, 24.0]]])
    assert calls == {"dequantize": 1, "quantized_matmul": 0}


def test_deepseek_v4_kquant_lm_head_prefill_with_cache_slices_last_row(
    monkeypatch,
):
    mx = pytest.importorskip("mlx.core")
    calls = {"dequantize": 0, "quantized_matmul": 0}
    _fake_q8_kquant_module(monkeypatch, calls)

    # Two token rows with a cache: generation prefill only consumes the
    # newest position, so the head sees one row and its logits equal the
    # full bridge's last row.
    hidden = mx.array([[[2.0, 3.0], [1.0, 1.0]]], dtype=mx.float16)
    model = _lm_head_model_fixture(mx, hidden)
    assert _patch_deepseek_v4_kquant_lm_head(model) is True

    out = model(mx.array([[1, 2]], dtype=mx.int32), cache=object())
    mx.eval(out)

    assert out.shape == (1, 1, 2)
    np.testing.assert_allclose(np.asarray(out), [[[12.0, 24.0]]])


def test_deepseek_v4_kquant_lm_head_decode_row_uses_wire_qmv(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    kq = pytest.importorskip("mlx_kquant")
    # The decode row must run the wire QMV on the resident q8_0 bytes,
    # not the dequant bridge.
    calls = {"dequantize": 0}
    real_dequantize = kq.dequantize

    def spy_dequantize(*args, **kwargs):
        calls["dequantize"] += 1
        return real_dequantize(*args, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "mlx_kquant",
        SimpleNamespace(
            dequantize=spy_dequantize,
            quantized_matmul=kq.quantized_matmul,
        ),
    )

    hidden = mx.array(
        np.random.default_rng(6).standard_normal(
            (1, 1, 64), dtype=np.float32)).astype(mx.float16)
    model = _lm_head_model_fixture(mx, hidden)
    model.lm_head = _real_q8_module(kq, mx)
    assert _patch_deepseek_v4_kquant_lm_head(model) is True

    from moespresso.runtime.deepseek_v4.model import (
        q8_dense_matmul_call_counts,
    )

    before = q8_dense_matmul_call_counts()
    out = model(mx.array([[1]], dtype=mx.int32))
    reference = mx.matmul(
        hidden.astype(mx.float32),
        real_dequantize(
            model.lm_head["weight"], model.lm_head["scales"], "q8_0",
            dtype=mx.float32).T,
    )
    mx.eval(out, reference)
    after = q8_dense_matmul_call_counts()

    assert out.dtype == mx.float32
    assert calls["dequantize"] == 0
    assert after["decode_wire_qmv_lm_head"] == (
        before["decode_wire_qmv_lm_head"] + 1)
    # The wire QMV accumulates on the bfloat16 activation lattice; the
    # 0.01 relative bound matches the engagement test above.
    diff = float(np.max(np.abs(np.asarray(out) - np.asarray(reference))))
    scale = float(np.max(np.abs(np.asarray(reference))))
    assert diff <= 0.01 * scale


def test_deepseek_v4_router_gate_dtype_guard_rejects_raw_bf16_bits():
    mx = pytest.importorskip("mlx.core")
    model = _WrappedModel([
        _RouterLayer(_Gate(mx.zeros((2, 4), dtype=mx.uint16))),
    ])

    with pytest.raises(DeepseekV4RuntimeLoadError, match="router gate weights"):
        _validate_deepseek_v4_router_gate_dtypes(model)


def test_deepseek_v4_router_gate_dtype_guard_accepts_float_weights():
    mx = pytest.importorskip("mlx.core")
    model = _WrappedModel([
        _RouterLayer(_Gate(mx.zeros((2, 4), dtype=mx.float16))),
        _RouterLayer(_Gate(mx.zeros((2, 4), dtype=mx.float32))),
    ])

    checked = _validate_deepseek_v4_router_gate_dtypes(model)

    assert checked == 2
    assert model._moespresso_dsv4_router_gates_checked == 2


def test_deepseek_v4_attention_cache_contract_uses_dsv4_cache_for_compressed_layers():
    model = _CacheWrappedModel([0, 4, 128])

    _patch_deepseek_v4_required_attention_cache(
        model,
        kv_cache_cls=_FakeKVCache,
        deepseek_cache_cls=_FakeDeepseekV4Cache,
    )

    cache = model.make_cache()

    assert isinstance(cache[0], _FakeKVCache)
    assert isinstance(cache[1], _FakeDeepseekV4Cache)
    assert cache[1].sliding_window == 128
    assert cache[1].compress_ratio == 4
    assert isinstance(cache[2], _FakeDeepseekV4Cache)
    assert cache[2].sliding_window == 128
    assert cache[2].compress_ratio == 128
    cache[1].indexer_state["pooled_qat"] = object()
    cache[1].indexer_state["pooled_qat_rows"] = 512
    cache[1].compressor_state["pooled_fp8"] = object()
    cache[1].compressor_state["pooled_fp8_rows"] = 512
    assert cache[1].trim(4) == 4
    assert "pooled_qat" not in cache[1].indexer_state
    assert "pooled_qat_rows" not in cache[1].indexer_state
    assert "pooled_fp8" not in cache[1].compressor_state
    assert "pooled_fp8_rows" not in cache[1].compressor_state
    assert (
        model._moespresso_dsv4_attention_cache_contract
        == "required_hsa_csa_for_compressed_layers"
    )


def test_deepseek_v4_attention_cache_contract_rounds_raw_kv_before_cache():
    mx = pytest.importorskip("mlx.core")
    model = _CacheWrappedModel([0])

    _patch_deepseek_v4_required_attention_cache(
        model,
        kv_cache_cls=_FakeKVCache,
        deepseek_cache_cls=_FakeDeepseekV4Cache,
    )

    cache = model.make_cache()[0]
    values_np = np.zeros((1, 1, 512), dtype=np.float32)
    values_np[..., :448] = 0.12345
    values_np[..., 448:] = 0.333
    values = mx.array(values_np, dtype=mx.float32)
    keys, out_values = cache.update_and_fetch(values, values)
    mx.eval(keys, out_values)

    keys_np = np.asarray(keys)
    original_np = np.asarray(values)
    assert cache._moespresso_dsv4_fp8_kv_cache is True
    assert not np.allclose(keys_np[..., :448], original_np[..., :448])
    np.testing.assert_allclose(keys_np[..., 448:], original_np[..., 448:])
    np.testing.assert_allclose(keys_np, np.asarray(out_values))


def test_deepseek_v4_attention_fp16_qkv_patch_casts_attention_inputs():
    mx = pytest.importorskip("mlx.core")
    dsv4_model = pytest.importorskip("jang_tools.dsv4.mlx_model")
    model = _Model()
    calls = {}
    original_attention = dsv4_model.scaled_dot_product_attention
    had_saved = hasattr(dsv4_model, "_moespresso_original_scaled_dot_product_attention")
    saved_attention = getattr(
        dsv4_model,
        "_moespresso_original_scaled_dot_product_attention",
        None,
    )

    def fake_attention(q, k, v, *args, **kwargs):
        del args
        calls["dtypes"] = (q.dtype, k.dtype, v.dtype)
        calls["mask_dtype"] = kwargs["mask"].dtype
        return (q + k + v).astype(mx.float32)

    try:
        if had_saved:
            delattr(dsv4_model, "_moespresso_original_scaled_dot_product_attention")
        dsv4_model.scaled_dot_product_attention = fake_attention

        assert _patch_deepseek_v4_attention_fp16_qkv(model) is True
        q = mx.ones((1, 1, 1, 2), dtype=mx.float32)
        mask = mx.zeros((1, 1), dtype=mx.float32)
        out = dsv4_model.scaled_dot_product_attention(
            q,
            q,
            q,
            scale=1.0,
            mask=mask,
        )
        mx.eval(out)

        assert calls["dtypes"] == (mx.float16, mx.float16, mx.float16)
        assert calls["mask_dtype"] == mx.float16
        assert out.dtype == mx.float32
        assert model._moespresso_dsv4_attention_qkv_dtype == "float16"
    finally:
        dsv4_model.scaled_dot_product_attention = original_attention
        if had_saved:
            dsv4_model._moespresso_original_scaled_dot_product_attention = saved_attention
        elif hasattr(dsv4_model, "_moespresso_original_scaled_dot_product_attention"):
            delattr(dsv4_model, "_moespresso_original_scaled_dot_product_attention")


def test_deepseek_v4_compressor_ape_patch_casts_attention_and_indexer_ape():
    mx = pytest.importorskip("mlx.core")
    ape = mx.array([[0.1, 0.2]], dtype=mx.float32)
    indexer_ape = mx.array([[0.3, 0.4]], dtype=mx.float32)
    model = _WrappedModel([_ApeLayer(ape, indexer_ape)])

    patched = _patch_deepseek_v4_compressor_ape_float16(model)

    layer = model.model.layers[0]
    assert patched == 2
    assert layer.self_attn.compressor.ape.dtype == mx.float16
    assert layer.self_attn.indexer.compressor.ape.dtype == mx.float16
    assert model._moespresso_dsv4_compressor_ape_dtype == "float16"
    assert model._moespresso_dsv4_compressor_ape_tensors == 2


def test_empty_skeleton_loader_does_not_read_package_shards(tmp_path):
    _touch_shard(tmp_path)
    calls = []

    def fake_load_model(path, **kwargs):
        calls.append((path, kwargs, list(path.glob("model-*.safetensors"))))
        return "MODEL", kwargs["model_config"]

    assert _load_empty_deepseek_v4_skeleton(
        tmp_path,
        model_config={"model_type": "deepseek_v4"},
        load_model_fn=fake_load_model,
    ) == ("MODEL", {"model_type": "deepseek_v4"})

    load_path, kwargs, shards = calls[0]
    assert load_path != tmp_path
    assert shards == []
    assert kwargs == {
        "lazy": True,
        "strict": False,
        "model_config": {"model_type": "deepseek_v4"},
    }


def test_deepseek_v4_package_loader_binds_regular_and_bundle_weights(tmp_path):
    _touch_shard(tmp_path)
    model = _Model()
    calls = []

    def load_skeleton(package_dir, **kwargs):
        calls.append(("skeleton", package_dir, kwargs))
        return model, kwargs["model_config"]

    def install_bundles(model_arg, package_dir, index):
        calls.append(("install", model_arg, package_dir, index))
        return 1

    def wrap_switchglus(model_arg, *, required_mixed_layers):
        calls.append(("wrap", model_arg, required_mixed_layers))
        return 1

    def apply_tensor_map(model_arg, tensor_map):
        calls.append(("tensor_map", model_arg, tensor_map))

    result = load_deepseek_v4_package_model(
        _manifest(),
        tmp_path,
        load_config_fn=lambda _path: {"model_type": "deepseek_v4"},
        load_skeleton_fn=load_skeleton,
        load_tokenizer_fn=lambda _path: "TOK",
        load_shard_fn=lambda _path: {
            "embed.weight": "E",
            "layers.0.ffn.experts.tq_bundle": "BUNDLE",
        },
        expert_index_fn=lambda _path: _ExpertIndex(),
        install_bundles_fn=install_bundles,
        wrap_switchglus_fn=wrap_switchglus,
        apply_tensor_map_fn=apply_tensor_map,
        read_jang_config_fn=lambda _path: {
            "mxtq_seed": 123,
            "quantization": {"tensor_map": {"model.embed": {"bits": 6}}},
        },
    )

    assert result == (model, "TOK")
    assert model.loaded == [({"sanitized.embed.weight": "E"}, False)]
    assert model._moespresso_dsv4_regular_tensors_loaded == 1
    assert model._moespresso_dsv4_bundles_seen == 1
    assert (
        model._moespresso_dsv4_attention_cache_contract
        == "required_hsa_csa_for_compressed_layers"
    )
    assert calls[0] == (
        "skeleton",
        tmp_path,
        {
            "lazy": True,
            "strict": False,
            "model_config": {"model_type": "deepseek_v4"},
        },
    )
    assert calls[1] == ("tensor_map", model, {"model.embed": {"bits": 6}})
    install_call = calls[2]
    assert install_call[:3] == ("install", model, tmp_path)
    assert isinstance(install_call[3], _ExpertIndex)
    assert len(install_call) == 4
    assert calls[3] == ("wrap", model, {0})


def test_deepseek_v4_package_loader_refuses_dense_iqk(tmp_path):
    manifest = _manifest()
    manifest["tensors"].append({"kind": "affine", "format": "iqk"})

    with pytest.raises(
        DeepseekV4RuntimeLoadError,
        match="do not support dense IQ_K tensors",
    ):
        load_deepseek_v4_package_model(manifest, tmp_path)


def _load_package_with_fakes(model, tmp_path):
    return load_deepseek_v4_package_model(
        _manifest(),
        tmp_path,
        load_config_fn=lambda _path: {"model_type": "deepseek_v4"},
        load_skeleton_fn=lambda _path, **kwargs: (model, kwargs["model_config"]),
        load_tokenizer_fn=lambda _path: "TOK",
        load_shard_fn=lambda _path: {"embed.weight": "E"},
        expert_index_fn=lambda _path: None,
        read_jang_config_fn=lambda _path: {},
    )


def test_dsv4_prefill_single_chunk_cap_default_and_env_override(monkeypatch):
    monkeypatch.delenv(
        "MOESPRESSO_DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS", raising=False)
    assert (
        dsv4_runtime._dsv4_prefill_single_chunk_max_tokens()
        == dsv4_runtime._DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS
    )

    monkeypatch.setenv("MOESPRESSO_DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS", "4096")
    assert dsv4_runtime._dsv4_prefill_single_chunk_max_tokens() == 4096

    monkeypatch.setenv("MOESPRESSO_DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS", "0")
    assert dsv4_runtime._dsv4_prefill_single_chunk_max_tokens() == 0


def test_deepseek_v4_package_loader_applies_single_chunk_cap_env(
        tmp_path, monkeypatch):
    _touch_shard(tmp_path)
    model = _Model()
    monkeypatch.setenv("MOESPRESSO_DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS", "4096")

    result = _load_package_with_fakes(model, tmp_path)

    assert result == (model, "TOK")
    assert model._moespresso_prefill_step_size == 4096
    assert model._moespresso_prefill_step_size_max_prompt_tokens == 4096


def test_deepseek_v4_single_chunk_policy_gates_by_prompt_depth(tmp_path):
    from moespresso.runtime.serve import _model_prefill_step_size

    _touch_shard(tmp_path)
    model = _Model()
    _load_package_with_fakes(model, tmp_path)

    cap = dsv4_runtime._DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS
    assert _model_prefill_step_size(model, None, list(range(cap))) == cap
    assert _model_prefill_step_size(model, None, list(range(cap + 1))) is None


def test_prewarm_wired_limit_enters_context_and_records_elapsed():
    import contextlib

    events = []
    model = SimpleNamespace()

    @contextlib.contextmanager
    def fake_wired_limit(model_arg, streams):
        events.append(("enter", model_arg, streams))
        yield
        events.append(("exit",))

    elapsed = dsv4_runtime._prewarm_wired_limit(
        model, wired_limit_fn=fake_wired_limit, streams=["S"])

    assert events == [("enter", model, ["S"]), ("exit",)]
    assert elapsed is not None
    assert elapsed >= 0.0
    assert model._moespresso_dsv4_wired_prewarm_seconds == elapsed


def test_deepseek_v4_package_loader_prewarms_wired_limit_at_load(
        tmp_path, monkeypatch, capsys):
    _touch_shard(tmp_path)
    model = _Model()
    seen = []

    def fake_prewarm(model_arg, **kwargs):
        seen.append(model_arg)
        return 1.23

    monkeypatch.setattr(dsv4_runtime, "_prewarm_wired_limit", fake_prewarm)
    result = _load_package_with_fakes(model, tmp_path)

    assert result == (model, "TOK")
    assert seen == [model]
    assert capsys.readouterr().out == "[serve] wired-limit prewarm 1.23s at load\n"


def test_deepseek_v4_package_loader_installs_ratio4_fast_prefill_by_default(
    tmp_path,
    monkeypatch,
):
    _touch_shard(tmp_path)
    model = _Model()
    calls = []

    def load_skeleton(package_dir, **kwargs):
        calls.append(("skeleton", package_dir, kwargs))
        return model, kwargs["model_config"]

    def patch_fast_prefill(model_arg):
        calls.append(("fast_ratio4_prefill", model_arg))
        return 0

    monkeypatch.setattr(
        dsv4_runtime,
        "_patch_deepseek_v4_ratio4_prefill_fast_path",
        patch_fast_prefill,
    )
    monkeypatch.setattr(
        dsv4_runtime,
        "_patch_deepseek_v4_attention_fp16_qkv",
        lambda _model: False,
    )

    result = load_deepseek_v4_package_model(
        _manifest(),
        tmp_path,
        load_config_fn=lambda _path: {"model_type": "deepseek_v4"},
        load_skeleton_fn=load_skeleton,
        load_tokenizer_fn=lambda _path: "TOK",
        load_shard_fn=lambda _path: {"embed.weight": "E"},
        expert_index_fn=lambda _path: None,
        read_jang_config_fn=lambda _path: {},
    )

    assert result == (model, "TOK")
    assert ("fast_ratio4_prefill", model) in calls


def test_deepseek_v4_default_pooled_loader_seeds_residency(
        tmp_path, monkeypatch):
    _touch_shard(tmp_path)
    import moespresso.runtime.ssd_streaming_build as streaming

    model = _Model()
    calls = []

    def load_skeleton(package_dir, **kwargs):
        calls.append(("skeleton", package_dir, kwargs))
        return model, kwargs["model_config"]

    def install_pooled_switchglus(*args, **kwargs):
        calls.append(("install_pooled", args, kwargs))
        return 1

    def seed_expert_residency(model_arg, package_dir):
        calls.append(("seed", model_arg, package_dir))
        return {"source": "all", "path": None, "seeded": 12}

    monkeypatch.setattr(streaming, "install_pooled_switchglus",
                        install_pooled_switchglus)
    monkeypatch.setattr(streaming, "seed_expert_residency",
                        seed_expert_residency)

    result = load_deepseek_v4_package_model(
        _manifest(),
        tmp_path,
        load_config_fn=lambda _path: {"model_type": "deepseek_v4"},
        load_skeleton_fn=load_skeleton,
        load_tokenizer_fn=lambda _path: "TOK",
        load_shard_fn=lambda _path: {
            "embed.weight": "E",
            "layers.0.ffn.experts.tq_bundle": "BUNDLE",
        },
        expert_index_fn=lambda _path: _ExpertIndex(),
        read_jang_config_fn=lambda _path: {"mxtq_seed": 123},
        capacity_per_layer=4,
    )

    assert result == (model, "TOK")
    assert model._moespresso_ssd_hotlist == {
        "source": "all",
        "path": None,
        "seeded": 12,
    }
    assert calls[1][0] == "install_pooled"
    assert calls[2] == ("seed", model, tmp_path)


def test_deepseek_v4_iqk_manifest_uses_default_pool_and_seeds_residency(
    tmp_path,
    monkeypatch,
):
    _touch_shard(tmp_path)
    import moespresso.runtime.deepseek_v4.iqk_experts as iqk
    import moespresso.runtime.ssd_streaming_build as streaming

    model = _Model()
    calls = []

    def install_pooled_switchglus(*args, **kwargs):
        calls.append(("install_pooled", args, kwargs))
        return 1

    def seed_expert_residency(model_arg, package_dir):
        calls.append(("seed", model_arg, package_dir))
        return {"source": "all-default", "path": None, "seeded": 256}

    def resident_iqk_installer(*_args, **_kwargs):
        raise AssertionError("the resident-only IQ_K installer must not run")

    monkeypatch.setattr(streaming, "install_pooled_switchglus",
                        install_pooled_switchglus)
    monkeypatch.setattr(streaming, "seed_expert_residency",
                        seed_expert_residency)
    monkeypatch.setattr(iqk, "install_deepseek_v4_iqk_experts",
                        resident_iqk_installer)

    result = load_deepseek_v4_package_model(
        _iqk_expert_manifest(),
        tmp_path,
        load_config_fn=lambda _path: {"model_type": "deepseek_v4"},
        load_skeleton_fn=lambda _path, **kwargs: (
            model, kwargs["model_config"]),
        load_tokenizer_fn=lambda _path: "TOK",
        load_shard_fn=lambda _path: {
            "embed.weight": "E",
            "layers.0.ffn.experts.tq_bundle": "BUNDLE",
        },
        expert_index_fn=lambda _path: _ExpertIndex(),
        read_jang_config_fn=lambda _path: {"mxtq_seed": 123},
        capacity_per_layer=256,
    )

    assert result == (model, "TOK")
    assert calls[0][0] == "install_pooled"
    assert calls[1] == ("seed", model, tmp_path)


def test_deepseek_v4_pooled_bundle_installer_wraps_moe_by_default(
    tmp_path,
    monkeypatch,
):
    import moespresso.runtime.ssd_streaming_build as streaming

    calls = []

    def install_pooled_switchglus(*args, **kwargs):
        calls.append((args, kwargs))
        return 1

    monkeypatch.setattr(streaming, "install_pooled_switchglus", install_pooled_switchglus)

    model = _Model()
    installed = _install_deepseek_v4_pooled_bundles(
        model,
        tmp_path,
        object(),

        capacity_per_layer=4,
    )

    assert installed == 1
    assert calls[0][1]["wrap_deepseek_v4_moe"] is True
    assert model._moespresso_ssd_streaming_capacity == 4


def test_deepseek_v4_package_loader_leaves_kquant_expert_bundles_to_pool_installer(
    tmp_path,
):
    _touch_shard(tmp_path)
    model = _Model()
    calls = []

    def load_skeleton(package_dir, **kwargs):
        calls.append(("skeleton", package_dir, kwargs))
        return model, kwargs["model_config"]

    def install_kquant(model_arg, codec_map):
        calls.append(("kquant", model_arg, codec_map))
        return 1

    def load_shard(_path):
        calls.append(("load_shard",))
        return {
            "embed.weight": "E",
            "layers.0.ffn.experts.tq_bundle": "BUNDLE",
        }

    result = load_deepseek_v4_package_model(
        _kquant_manifest(),
        tmp_path,
        load_config_fn=lambda _path: {"model_type": "deepseek_v4"},
        load_skeleton_fn=load_skeleton,
        load_tokenizer_fn=lambda _path: "TOK",
        load_shard_fn=load_shard,
        expert_index_fn=lambda _path: _ExpertIndex(),
        install_bundles_fn=lambda *_args, **_kwargs: 1,
        wrap_switchglus_fn=lambda *_args, **_kwargs: 1,
        install_kquant_modules_fn=install_kquant,
        apply_tensor_map_fn=lambda *_args, **_kwargs: None,
        read_jang_config_fn=lambda _path: {},
    )

    assert result == (model, "TOK")
    assert calls[0][0] == "skeleton"
    assert calls[1] == ("load_shard",)
    assert not any(call[0] == "kquant" for call in calls)
    assert not hasattr(model, "_moespresso_kquant_modules_installed")
    assert model.loaded == [({"sanitized.embed.weight": "E"}, False)]


def test_deepseek_v4_package_loader_installs_dense_kquant_modules(tmp_path):
    _touch_shard(tmp_path)
    model = _Model()
    calls = []

    def load_skeleton(package_dir, **kwargs):
        calls.append(("skeleton", package_dir, kwargs))
        return model, kwargs["model_config"]

    def install_kquant(model_arg, codec_map):
        calls.append(("kquant", model_arg, codec_map))
        return 1

    result = load_deepseek_v4_package_model(
        _dense_kquant_manifest(),
        tmp_path,
        load_config_fn=lambda _path: {"model_type": "deepseek_v4"},
        load_skeleton_fn=load_skeleton,
        load_tokenizer_fn=lambda _path: "TOK",
        load_shard_fn=lambda _path: {
            "layers.0.attn.wq_a.weight": "W",
            "layers.0.attn.wq_a.scales": "S",
        },
        expert_index_fn=lambda _path: _ExpertIndex(),
        install_bundles_fn=lambda *_args, **_kwargs: 1,
        wrap_switchglus_fn=lambda *_args, **_kwargs: 1,
        install_kquant_modules_fn=install_kquant,
        apply_tensor_map_fn=lambda *_args, **_kwargs: None,
        read_jang_config_fn=lambda _path: {},
    )

    assert result == (model, "TOK")
    assert calls[1] == (
        "kquant",
        model,
        {"model.layers.0.self_attn.wq_a.weight": "q8_0"},
    )
    assert model._moespresso_kquant_modules_installed == 1
    assert model.loaded == [
        ({
            "sanitized.layers.0.attn.wq_a.weight": "W",
            "sanitized.layers.0.attn.wq_a.scales": "S",
        }, False)
    ]


def test_deepseek_v4_package_loader_fails_when_routed_bundle_is_absent(tmp_path):
    _touch_shard(tmp_path)

    with pytest.raises(DeepseekV4RuntimeLoadError, match="no expert bundle"):
        load_deepseek_v4_package_model(
            _routed_tq_manifest(),
            tmp_path,
            load_config_fn=lambda _path: {"model_type": "deepseek_v4"},
            load_skeleton_fn=lambda _path, **_kwargs: (_Model(), {}),
            load_tokenizer_fn=lambda _path: "TOK",
            load_shard_fn=lambda _path: {"embed.weight": "E"},
            expert_index_fn=lambda _path: _ExpertIndex(),
            install_bundles_fn=lambda *_args, **_kwargs: 1,
            wrap_switchglus_fn=lambda *_args, **_kwargs: 1,
            apply_tensor_map_fn=lambda *_args, **_kwargs: None,
            read_jang_config_fn=lambda _path: {},
        )


def test_deepseek_v4_package_loader_requires_deepseek_model_type(tmp_path):
    _touch_shard(tmp_path)

    with pytest.raises(DeepseekV4RuntimeLoadError, match="model_type='deepseek_v4'"):
        load_deepseek_v4_package_model(
            _manifest(),
            tmp_path,
            load_config_fn=lambda _path: {"model_type": "deepseek_v4_flash"},
            load_skeleton_fn=lambda _path, **_kwargs: (_Model(), {}),
            load_tokenizer_fn=lambda _path: "TOK",
            load_shard_fn=lambda _path: {
                "embed.weight": "E",
                "layers.0.ffn.experts.tq_bundle": "BUNDLE",
            },
            expert_index_fn=lambda _path: _ExpertIndex(),
            install_bundles_fn=lambda *_args, **_kwargs: 1,
            wrap_switchglus_fn=lambda *_args, **_kwargs: 1,
            apply_tensor_map_fn=lambda *_args, **_kwargs: None,
            read_jang_config_fn=lambda _path: {},
        )


def test_deepseek_v4_tiny_package_loads_renders_prefills_and_decodes(tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("jang_tools.dsv4")
    import json
    import numpy as np
    from safetensors.numpy import save_file

    from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata
    from moespresso.runtime.deepseek_v4.renderer import render_deepseek_v4_prompt
    from moespresso.runtime.serve import load_served_model
    from moespresso.package.sidecars import build_sidecars

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
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 32,
        "num_hash_layers": 0,
        "num_nextn_predict_layers": 1,
        "sliding_window": 8,
        "compress_ratios": [0],
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
    }
    manifest = {
        "architecture": {"family": "deepseek_v4_flash", "config": config},
        "required_ops": ["raw_dtype_passthrough", "mxfp4_dequant"],
        "tensors": [
            {"source_name": "layers.0.attn.attn_sink", "format": "raw_dtype_passthrough"},
            {
                "source_name": "layers.0.ffn.experts",
                "kind": "expert",
                "format": "mxfp4",
                "layer_index": 0,
                "projection": "gate",
                "format_params": {"bits": 4},
            },
            {
                "source_name": "layers.0.ffn.experts",
                "kind": "expert",
                "format": "mxfp4",
                "layer_index": 0,
                "projection": "up",
                "format_params": {"bits": 4},
            },
            {
                "source_name": "layers.0.ffn.experts",
                "kind": "expert",
                "format": "mxfp4",
                "layer_index": 0,
                "projection": "down",
                "format_params": {"bits": 4},
            },
        ],
    }
    config_json, jang_config = build_sidecars(manifest)
    (tmp_path / "config.json").write_text(json.dumps(config_json))
    (tmp_path / "jang_config.json").write_text(json.dumps(jang_config))

    n_exp, hidden, inter, bits = 2, 64, 32, 4
    vals_per_word = 32 // bits
    components = {
        ("gate_proj", "packed"): np.zeros(
            (n_exp, inter, hidden // vals_per_word), dtype=np.uint32),
        ("gate_proj", "scales"): np.full((n_exp, inter, hidden // 32), 127, dtype=np.uint8),
        ("up_proj", "packed"): np.zeros(
            (n_exp, inter, hidden // vals_per_word), dtype=np.uint32),
        ("up_proj", "scales"): np.full((n_exp, inter, hidden // 32), 127, dtype=np.uint8),
        ("down_proj", "packed"): np.zeros(
            (n_exp, hidden, inter // vals_per_word), dtype=np.uint32),
        ("down_proj", "scales"): np.full((n_exp, hidden, inter // 32), 127, dtype=np.uint8),
    }
    bundle, geometry = assemble_layer_bundle(
        components,
        {"gate_proj": bits, "up_proj": bits, "down_proj": bits},
        codecs={p: "mxfp4" for p in ("gate_proj", "up_proj", "down_proj")},
    )
    save_file(
        {
            "layers.0.attn.attn_sink": np.zeros((2,), dtype=np.float32),
            "layers.0.ffn.experts.tq_bundle": bundle,
        },
        str(tmp_path / "model-00001-of-00001.safetensors"),
        metadata={"expert_bundles": encode_bundle_metadata({0: geometry})},
    )

    prompt = render_deepseek_v4_prompt([{"role": "user", "content": "hello"}])
    served_manifest = {**manifest, "artifact_id": "pkg:tiny-dsv4"}
    model, tokenizer, loaded_manifest = load_served_model(
        tmp_path,
        manifest=served_manifest,
        build_fn=lambda man, package_dir: load_deepseek_v4_package_model(
            man, package_dir, load_tokenizer_fn=lambda _path: "TOK"),
    )
    cache = model.make_cache()
    prefill = model(mx.array([[1, 2, 3]], dtype=mx.uint32), cache=cache)
    decode = model(mx.array([[4]], dtype=mx.uint32), cache=cache)
    mx.eval(prefill, decode)

    assert "<｜User｜>" in prompt
    assert tokenizer == "TOK"
    assert loaded_manifest is served_manifest
    assert (
        model._moespresso_dsv4_attention_cache_contract
        == "required_hsa_csa_for_compressed_layers"
    )
    assert (
        model._moespresso_prefill_step_size
        == dsv4_runtime._DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS
    )
    assert (
        model._moespresso_prefill_step_size_max_prompt_tokens
        == dsv4_runtime._DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS
    )
    assert prefill.shape == (1, 3, 128)
    assert decode.shape == (1, 1, 128)


def test_deepseek_v4_tiny_mxfp4_package_loads_prefills_and_decodes(tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("jang_tools.dsv4")
    import json
    import numpy as np
    from safetensors.numpy import save_file

    from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata
    from moespresso.runtime.serve import load_served_model
    from moespresso.package.sidecars import build_sidecars
    from moespresso.runtime.pooled_switchglu import PooledSwitchGLU

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
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 32,
        "num_hash_layers": 0,
        "num_nextn_predict_layers": 1,
        "sliding_window": 8,
        "compress_ratios": [0],
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
    }
    manifest = {
        "architecture": {"family": "deepseek_v4_flash", "config": config},
        "required_ops": ["raw_dtype_passthrough", "mxfp4_dequant"],
        "tensors": [
            {"source_name": "layers.0.attn.attn_sink", "format": "raw_dtype_passthrough"},
            {
                "source_name": "layers.0.ffn.experts",
                "kind": "expert",
                "format": "mxfp4",
                "layer_index": 0,
                "projection": "gate",
                "format_params": {"bits": 4, "group_size": 32, "scale_dtype": "ue8m0"},
            },
            {
                "source_name": "layers.0.ffn.experts",
                "kind": "expert",
                "format": "mxfp4",
                "layer_index": 0,
                "projection": "up",
                "format_params": {"bits": 4, "group_size": 32, "scale_dtype": "ue8m0"},
            },
            {
                "source_name": "layers.0.ffn.experts",
                "kind": "expert",
                "format": "mxfp4",
                "layer_index": 0,
                "projection": "down",
                "format_params": {"bits": 4, "group_size": 32, "scale_dtype": "ue8m0"},
            },
        ],
    }
    config_json, jang_config = build_sidecars(manifest)
    (tmp_path / "config.json").write_text(json.dumps(config_json))
    (tmp_path / "jang_config.json").write_text(json.dumps(jang_config))

    n_exp, hidden, inter = 2, 64, 32
    components = {
        ("gate_proj", "packed"): np.zeros((n_exp, inter, hidden // 8), dtype=np.uint32),
        ("gate_proj", "scales"): np.full((n_exp, inter, hidden // 32), 127, dtype=np.uint8),
        ("up_proj", "packed"): np.zeros((n_exp, inter, hidden // 8), dtype=np.uint32),
        ("up_proj", "scales"): np.full((n_exp, inter, hidden // 32), 127, dtype=np.uint8),
        ("down_proj", "packed"): np.zeros((n_exp, hidden, inter // 8), dtype=np.uint32),
        ("down_proj", "scales"): np.full((n_exp, hidden, inter // 32), 127, dtype=np.uint8),
    }
    codecs = {p: "mxfp4" for p in ("gate_proj", "up_proj", "down_proj")}
    bundle, geometry = assemble_layer_bundle(
        components,
        {"gate_proj": 4, "up_proj": 4, "down_proj": 4},
        codecs=codecs,
    )
    save_file(
        {
            "layers.0.attn.attn_sink": np.zeros((2,), dtype=np.float32),
            "layers.0.ffn.experts.tq_bundle": bundle,
        },
        str(tmp_path / "model-00001-of-00001.safetensors"),
        metadata={"expert_bundles": encode_bundle_metadata({0: geometry})},
    )

    served_manifest = {**manifest, "artifact_id": "pkg:tiny-dsv4-mxfp4"}
    model, tokenizer, loaded_manifest = load_served_model(
        tmp_path,
        manifest=served_manifest,
        build_fn=lambda man, package_dir: load_deepseek_v4_package_model(
            man, package_dir, load_tokenizer_fn=lambda _path: "TOK"),
    )
    switch = model.layers[0].mlp.switch_mlp
    cache = model.make_cache()
    prefill = model(mx.array([[1, 2, 3]], dtype=mx.uint32), cache=cache)
    decode = model(mx.array([[4]], dtype=mx.uint32), cache=cache)
    mx.eval(prefill, decode)

    assert isinstance(switch, PooledSwitchGLU)
    assert switch._all_mxfp4
    assert tokenizer == "TOK"
    assert loaded_manifest is served_manifest
    assert (
        model._moespresso_dsv4_attention_cache_contract
        == "required_hsa_csa_for_compressed_layers"
    )
    assert (
        model._moespresso_prefill_step_size
        == dsv4_runtime._DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS
    )
    assert (
        model._moespresso_prefill_step_size_max_prompt_tokens
        == dsv4_runtime._DSV4_PREFILL_SINGLE_CHUNK_MAX_TOKENS
    )
    assert prefill.shape == (1, 3, 128)
    assert decode.shape == (1, 1, 128)
