from __future__ import annotations

import builtins

import numpy as np
import pytest

from moespresso.package.kquant_backend import (
    KQuantBackendError,
    KQuantEncodedWeight,
    KQuantRuntime,
    check_kquant_backend_available,
    decode_kquant_weight,
    encode_kquant_weight,
    is_kquant_module,
    kquant_roundtrip_relative_error,
)
from moespresso.package.deepseek_v4.recipe import DS4KQuantExpertTarget
from moespresso.package.kquant_recipe import KQuantRecipeError


def _target(codec="q2_k"):
    return DS4KQuantExpertTarget(
        layer_index=7,
        projection="down",
        codec=codec,
        gguf_tensor="blk.7.ffn_down_exps.weight",
        imatrix_key="blk.7.ffn_down_exps.weight",
        source_weight_template="layers.7.ffn.experts.{expert}.w2.weight",
        source_scale_template="layers.7.ffn.experts.{expert}.w2.scale",
        module_path="model.layers.7.mlp.switch_mlp.down_proj",
        module_weight_key="model.layers.7.mlp.switch_mlp.down_proj.weight",
    )


class _FakeMx:
    cpu = "cpu-stream"
    gpu = "gpu-stream"
    float32 = np.float32
    float16 = np.float16

    def __init__(self):
        self.evaluated = []

    def array(self, value, dtype=None):
        return np.asarray(value, dtype=dtype)

    def eval(self, *values):
        self.evaluated.append(tuple(np.asarray(v).shape for v in values))


class _FakeKq:
    def __init__(self):
        self.quantize_calls = []
        self.dequantize_calls = []

    def quantize(self, weight, codec, *, imatrix=None, stream=None):
        self.quantize_calls.append({
            "weight": np.asarray(weight).copy(),
            "codec": codec,
            "imatrix": None if imatrix is None else np.asarray(imatrix).copy(),
            "stream": stream,
        })
        return np.full((weight.shape[0], 84), 7, dtype=np.uint8), np.array([0], dtype=np.uint8)

    def dequantize(self, weight, scales, codec, *, dtype=None, stream=None):
        self.dequantize_calls.append({
            "weight": np.asarray(weight).copy(),
            "scales": np.asarray(scales).copy(),
            "codec": codec,
            "dtype": dtype,
            "stream": stream,
        })
        return np.ones((weight.shape[0], 256), dtype=dtype)


def _runtime():
    return KQuantRuntime(mx=_FakeMx(), kq=_FakeKq())


def test_encode_kquant_weight_auto_routes_gpu_capable_codec_to_gpu_stream():
    # q2_k has a GPU encoder, so the default stream auto-resolves to GPU (~30x
    # faster, bit-identical). The CPU default was a stale iq2_xxs-era choice.
    rt = _runtime()
    weight = np.arange(512, dtype=np.float32).reshape(2, 256)
    imatrix = {"blk.7.ffn_down_exps.weight": np.ones(256, dtype=np.float32)}

    encoded = encode_kquant_weight(weight, _target(), imatrix, runtime=rt)

    assert encoded.codec == "q2_k"
    assert encoded.weight.dtype == np.uint8
    assert encoded.weight.shape == (2, 84)
    assert encoded.scales.tolist() == [0]
    [call] = rt.kq.quantize_calls
    assert call["codec"] == "q2_k"
    assert call["stream"] == "gpu-stream"
    np.testing.assert_array_equal(call["weight"], weight)
    np.testing.assert_array_equal(call["imatrix"], np.ones(256, dtype=np.float32))
    assert rt.mx.evaluated == [((2, 84), (1,))]


def test_encode_kquant_weight_auto_routes_iquant_codec_to_cpu_stream():
    # iq* codecs have no GPU encoder; they auto-resolve to the CPU stream.
    rt = _runtime()
    weight = np.arange(512, dtype=np.float32).reshape(2, 256)
    imatrix = {"blk.7.ffn_down_exps.weight": np.ones(256, dtype=np.float32)}

    encode_kquant_weight(weight, _target(codec="iq2_xxs"), imatrix, runtime=rt)

    [call] = rt.kq.quantize_calls
    assert call["codec"] == "iq2_xxs"
    assert call["stream"] == "cpu-stream"


def test_encode_kquant_weight_explicit_stream_overrides_auto():
    # An explicit stream still wins over the codec-based auto choice.
    rt = _runtime()
    weight = np.arange(512, dtype=np.float32).reshape(2, 256)
    imatrix = {"blk.7.ffn_down_exps.weight": np.ones(256, dtype=np.float32)}

    encode_kquant_weight(weight, _target(), imatrix, runtime=rt, stream="cpu")

    [call] = rt.kq.quantize_calls
    assert call["stream"] == "cpu-stream"


def test_encode_fails_before_backend_on_bad_imatrix_orientation():
    class _ExplodingKq:
        def quantize(self, *args, **kwargs):
            raise AssertionError("backend should not be called after failed fit check")

    rt = KQuantRuntime(mx=_FakeMx(), kq=_ExplodingKq())
    weight = np.ones((2, 256), dtype=np.float32)

    with pytest.raises(KQuantRecipeError, match="check tensor orientation"):
        encode_kquant_weight(
            weight,
            _target(),
            {"blk.7.ffn_down_exps.weight": np.ones(2, dtype=np.float32)},
            runtime=rt,
        )


def test_decode_kquant_weight_uses_codec_scales_dtype_and_stream():
    rt = _runtime()
    encoded = KQuantEncodedWeight(
        codec="q2_k",
        weight=np.full((2, 84), 3, dtype=np.uint8),
        scales=np.array([0], dtype=np.uint8),
    )

    decoded = decode_kquant_weight(encoded, dtype=np.float16, stream="gpu", runtime=rt)

    assert decoded.dtype == np.float16
    assert decoded.shape == (2, 256)
    [call] = rt.kq.dequantize_calls
    assert call["codec"] == "q2_k"
    assert call["stream"] == "gpu-stream"
    assert call["dtype"] == np.float16
    np.testing.assert_array_equal(call["scales"], [0])


def test_roundtrip_reports_relative_error_from_backend_decode():
    rt = _runtime()
    weight = np.ones((2, 256), dtype=np.float32)
    imatrix = {"blk.7.ffn_down_exps.weight": np.ones(256, dtype=np.float32)}

    encoded, err = kquant_roundtrip_relative_error(weight, _target(), imatrix, runtime=rt)

    assert encoded.codec == "q2_k"
    assert err == pytest.approx(0.0)


def test_kquant_module_detection_is_duck_typed_by_mode_only():
    class LooksLike:
        mode = "kquant"

    class Different:
        mode = "affine"

    assert is_kquant_module(LooksLike()) is True
    assert is_kquant_module(Different()) is False
    assert is_kquant_module(object()) is False


def test_invalid_stream_fails_closed():
    with pytest.raises(KQuantBackendError, match="unsupported mlx-kquant stream"):
        decode_kquant_weight(
            KQuantEncodedWeight(
                codec="q2_k",
                weight=np.full((2, 84), 3, dtype=np.uint8),
                scales=np.array([0], dtype=np.uint8),
            ),
            stream="metal",
            runtime=_runtime(),
        )


def test_missing_kquant_backend_reports_broken_install(monkeypatch):
    from moespresso.package import kquant_backend

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"mlx.core", "mlx_kquant"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(KQuantBackendError, match="Reinstall MoEspresso"):
        kquant_backend._load_kquant_runtime()


def test_backend_availability_check_uses_lazy_loader(monkeypatch):
    from moespresso.package import kquant_backend

    calls = []

    def fake_load():
        calls.append("load")
        return object()

    monkeypatch.setattr(kquant_backend, "_load_kquant_runtime", fake_load)

    check_kquant_backend_available()

    assert calls == ["load"]
