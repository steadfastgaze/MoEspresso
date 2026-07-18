"""Bit-parity, eligibility, and dispatch tests for the fused rope seam."""

from types import SimpleNamespace

import numpy as np
import pytest

from moespresso.runtime.deepseek_v4 import attention_seam_kernel as seam
from moespresso.runtime.deepseek_v4.model import (
    _ATTN_SEAM_ROPE_CALL_COUNTS,
    _patch_deepseek_v4_attention_seam_rope,
    attention_seam_rope_call_counts,
)


def _require_metal():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal is required for mx.fast.metal_kernel")
    return mx


def _jang_model():
    return pytest.importorskip("jang_tools.dsv4.mlx_model")


_YARN = {
    "rope_type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 65536,
    "beta_fast": 32,
    "beta_slow": 1,
}


def _ropes(jm):
    return {
        "compress_yarn": jm.DeepseekV4RoPE(64, 160000.0, _YARN),
        "swa_plain": jm.DeepseekV4RoPE(64, 10000.0, None),
    }


def _assert_bit_equal(mx, expected, actual, label):
    assert expected.shape == actual.shape, label
    assert expected.dtype == actual.dtype, label
    itype = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[expected.dtype.size]
    exp = np.asarray(mx.view(expected.reshape(-1), itype))
    got = np.asarray(mx.view(actual.reshape(-1), itype))
    mismatches = int((exp != got).sum())
    assert mismatches == 0, f"{label}: {mismatches}/{exp.size} bit mismatches"


def _rope_input(mx, shape, dtype, seed):
    mx.random.seed(seed)
    x = (mx.random.normal(shape) * 2.0).astype(dtype)
    zero_mask = mx.random.uniform(shape=shape) < 0.01
    return mx.where(zero_mask, mx.array(-0.0, dtype=dtype), x)


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(
    "shape",
    [(1, 64, 1, 512), (1, 1, 1, 512), (1, 64, 1, 128), (1, 1, 1, 64)],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_fused_partial_rope_bit_identical(dtype_name, shape, inverse):
    mx = _require_metal()
    jm = _jang_model()
    dtype = getattr(mx, dtype_name)
    for rope_name, rope in _ropes(jm).items():
        for offset in (0, 1, 3855, (1 << 24) - 2):
            x = _rope_input(mx, shape, dtype, offset + shape[1] + shape[3])
            ref = jm._moespresso_original_apply_partial_rope(
                x, rope, offset=offset, inverse=inverse,
            ) if hasattr(jm, "_moespresso_original_apply_partial_rope") else (
                jm._apply_partial_rope(x, rope, offset=offset, inverse=inverse)
            )
            got = seam.fused_partial_rope(
                x, rope.inv_freq, offset=offset, inverse=inverse)
            mx.eval(ref, got)
            _assert_bit_equal(
                mx, ref, got, f"{rope_name} offset={offset} inverse={inverse}")


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("rows", [1, 3])
def test_fused_partial_rope_positions_form_bit_identical(dtype_name, rows):
    mx = _require_metal()
    jm = _jang_model()
    dtype = getattr(mx, dtype_name)
    rope = _ropes(jm)["compress_yarn"]
    positions = mx.arange(rows, dtype=mx.float32) * 4.0 + 3844.0
    x = _rope_input(mx, (1, 1, rows, 512), dtype, rows)
    composed = getattr(jm, "_moespresso_original_apply_partial_rope",
                       jm._apply_partial_rope)
    ref = composed(x, rope, positions=positions)
    got = seam.fused_partial_rope(x, rope.inv_freq, positions=positions)
    mx.eval(ref, got)
    _assert_bit_equal(mx, ref, got, f"positions rows={rows}")


# Served prefill shapes and dtypes from the glue census: the six composed
# rope assembly sites at the 3843-row anchor chunk, the served tail chunks
# at depth (7698 -> 1554 rows at offset 6144, 15406 -> 1070 rows at offset
# 14336), and a short trailing chunk. The forward query runs bfloat16;
# the inverse output, KV, indexer, and pooled sites run float32.
_PREFILL_OFFSET_CASES = [
    ((1, 64, 3843, 512), "bfloat16", 0, False),      # query assembly
    ((1, 64, 3843, 512), "float32", 0, True),        # inverse-out assembly
    ((1, 1, 3843, 512), "float32", 0, False),        # kv assembly
    ((1, 64, 3843, 128), "float32", 0, False),       # indexer query
    ((1, 64, 1554, 512), "bfloat16", 6144, False),   # depth-7698 tail
    ((1, 64, 1554, 512), "float32", 6144, True),
    ((1, 64, 1070, 512), "bfloat16", 14336, False),  # depth-15406 tail
    ((1, 64, 1070, 512), "float32", 14336, True),
    ((1, 64, 7, 512), "bfloat16", 4096, False),      # short trailing chunk
    ((1, 64, 7, 512), "float32", 4096, True),
]


@pytest.mark.parametrize("shape,dtype_name,offset,inverse", _PREFILL_OFFSET_CASES)
def test_fused_partial_rope_prefill_rows_bit_identical(
    shape, dtype_name, offset, inverse,
):
    mx = _require_metal()
    jm = _jang_model()
    dtype = getattr(mx, dtype_name)
    composed = getattr(jm, "_moespresso_original_apply_partial_rope",
                       jm._apply_partial_rope)
    for rope_name, rope in _ropes(jm).items():
        x = _rope_input(mx, shape, dtype, offset + shape[2] + shape[3])
        ref = composed(x, rope, offset=offset, inverse=inverse)
        got = seam.fused_partial_rope(
            x, rope.inv_freq, offset=offset, inverse=inverse)
        mx.eval(ref, got)
        _assert_bit_equal(
            mx, ref, got,
            f"{rope_name} shape={shape} {dtype_name} offset={offset} "
            f"inverse={inverse}",
        )


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("width", [512, 128])
def test_fused_partial_rope_prefill_positions_form_bit_identical(
    dtype_name, width,
):
    # The ratio-4 pooled-row rope sites serve the positions form at 960
    # rows per chunk; pooled positions are the strided pool row centers.
    mx = _require_metal()
    jm = _jang_model()
    dtype = getattr(mx, dtype_name)
    rope = _ropes(jm)["compress_yarn"]
    rows = 960
    positions = mx.arange(rows, dtype=mx.float32) * 4.0 + 3.0
    x = _rope_input(mx, (1, 1, rows, width), dtype, rows + width)
    composed = getattr(jm, "_moespresso_original_apply_partial_rope",
                       jm._apply_partial_rope)
    ref = composed(x, rope, positions=positions)
    got = seam.fused_partial_rope(x, rope.inv_freq, positions=positions)
    mx.eval(ref, got)
    _assert_bit_equal(
        mx, ref, got, f"pooled positions rows={rows} width={width}")


def test_prefill_rope_seam_gate_and_row_caps(monkeypatch):
    mx = _require_metal()
    inv_freq = mx.zeros((32,), dtype=mx.float32)
    prefill_shaped = mx.zeros((1, 64, 2, 512), dtype=mx.float32)

    # Default on: prefill row counts are eligible.
    monkeypatch.delenv(seam._PREFILL_ENV_FLAG, raising=False)
    assert seam.prefill_rope_seam_enabled()
    assert seam.partial_rope_eligible(
        prefill_shaped, inv_freq, offset=0, positions=None)

    # The kill switch restores the decode-only cap without touching
    # decode eligibility.
    monkeypatch.setenv(seam._PREFILL_ENV_FLAG, "0")
    assert not seam.prefill_rope_seam_enabled()
    assert not seam.partial_rope_eligible(
        prefill_shaped, inv_freq, offset=0, positions=None)
    assert seam.partial_rope_eligible(
        mx.zeros((1, 64, 1, 512), dtype=mx.float32),
        inv_freq, offset=0, positions=None)

    # The structural ceiling still fails closed with the extension on.
    monkeypatch.delenv(seam._PREFILL_ENV_FLAG, raising=False)
    monkeypatch.setattr(seam, "_MAX_ROWS_PREFILL", 100)
    assert not seam.partial_rope_eligible(
        prefill_shaped, inv_freq, offset=0, positions=None)


def test_fused_partial_rope_kernel_names_carry_operand_form():
    mx = _require_metal()
    jm = _jang_model()
    rope = _ropes(jm)["compress_yarn"]
    x = _rope_input(mx, (1, 2, 1, 128), mx.float32, 7)
    mx.eval(seam.fused_partial_rope(x, rope.inv_freq, offset=5))
    mx.eval(seam.fused_partial_rope(x, rope.inv_freq, offset=5, inverse=True))
    positions = mx.array([5.0], dtype=mx.float32)
    mx.eval(seam.fused_partial_rope(x, rope.inv_freq, positions=positions))
    assert ("off", False) in seam._ROPE_KERNELS
    assert ("off", True) in seam._ROPE_KERNELS
    assert ("pos", False) in seam._ROPE_KERNELS
    assert len({id(k) for k in seam._ROPE_KERNELS.values()}) == len(
        seam._ROPE_KERNELS)


def test_partial_rope_eligibility_fails_closed():
    mx = _require_metal()
    inv_freq = mx.zeros((32,), dtype=mx.float32)
    good = mx.zeros((1, 64, 1, 512), dtype=mx.float32)
    assert seam.partial_rope_eligible(good, inv_freq, offset=0, positions=None)

    checks = [
        # width below the rope tail
        (mx.zeros((1, 1, 1, 32), dtype=mx.float32), inv_freq, 0, None),
        # width not float4-aligned
        (mx.zeros((1, 1, 1, 66), dtype=mx.float32), inv_freq, 0, None),
        # unsupported element dtype
        (mx.zeros((1, 1, 1, 512), dtype=mx.int32), inv_freq, 0, None),
        # positions outside the float32 exact-integer range
        (good, inv_freq, 1 << 24, None),
        # negative offset
        (good, inv_freq, -1, None),
        # non-integer offset
        (good, inv_freq, 3.5, None),
        # inv_freq dtype mismatch
        (good, inv_freq.astype(mx.float16), 0, None),
        # inv_freq width mismatch
        (good, mx.zeros((16,), dtype=mx.float32), 0, None),
        # positions dtype mismatch
        (good, inv_freq, 0, mx.zeros((1,), dtype=mx.float16)),
        # positions length mismatch against the sequence axis
        (good, inv_freq, 0, mx.zeros((2,), dtype=mx.float32)),
        # positions of a non-array type
        (good, inv_freq, 0, [0.0]),
    ]
    for x, freqs, offset, positions in checks:
        assert not seam.partial_rope_eligible(
            x, freqs, offset=offset, positions=positions)


def test_fused_partial_rope_raises_outside_contract(monkeypatch):
    mx = _require_metal()
    inv_freq = mx.zeros((32,), dtype=mx.float32)
    # Structurally ineligible: width below the rope tail.
    with pytest.raises(ValueError):
        seam.fused_partial_rope(
            mx.zeros((1, 64, 1, 32), dtype=mx.float32), inv_freq, offset=0)
    # Over the decode row cap with the prefill extension killed.
    monkeypatch.setenv(seam._PREFILL_ENV_FLAG, "0")
    with pytest.raises(ValueError):
        seam.fused_partial_rope(
            mx.zeros((1, 64, 2, 512), dtype=mx.float32), inv_freq, offset=0)


def test_rope_seam_gates(monkeypatch):
    if not seam._metal_available():
        pytest.skip("Metal is required for the gate check")
    monkeypatch.delenv(seam._FAMILY_ENV_FLAG, raising=False)
    monkeypatch.delenv(seam._ROPE_ENV_FLAG, raising=False)
    assert seam.rope_seam_enabled()
    monkeypatch.setenv(seam._FAMILY_ENV_FLAG, "0")
    assert not seam.rope_seam_enabled()
    monkeypatch.delenv(seam._FAMILY_ENV_FLAG)
    monkeypatch.setenv(seam._ROPE_ENV_FLAG, "0")
    assert not seam.rope_seam_enabled()


class _RestoredRopePatch:
    """Install the module-global rope patch and restore jang on exit."""

    def __init__(self, jm):
        self.jm = jm
        self.original = jm._apply_partial_rope
        self.had_saved = hasattr(jm, "_moespresso_original_apply_partial_rope")
        self.saved = getattr(jm, "_moespresso_original_apply_partial_rope", None)

    def __enter__(self):
        model = SimpleNamespace()
        assert _patch_deepseek_v4_attention_seam_rope(model) is True
        assert model._moespresso_dsv4_attn_seam_rope_installed is True
        return model

    def __exit__(self, *exc):
        self.jm._apply_partial_rope = self.original
        if self.had_saved:
            self.jm._moespresso_original_apply_partial_rope = self.saved
        elif hasattr(self.jm, "_moespresso_original_apply_partial_rope"):
            delattr(self.jm, "_moespresso_original_apply_partial_rope")
        return False


def test_seam_rope_patch_dispatch_and_counters(monkeypatch):
    mx = _require_metal()
    jm = _jang_model()
    monkeypatch.delenv(seam._FAMILY_ENV_FLAG, raising=False)
    monkeypatch.delenv(seam._ROPE_ENV_FLAG, raising=False)
    rope = _ropes(jm)["compress_yarn"]
    decode_x = _rope_input(mx, (1, 64, 1, 512), mx.bfloat16, 3)
    prefill_x = _rope_input(mx, (1, 64, 128, 512), mx.bfloat16, 4)

    with _RestoredRopePatch(jm):
        assert getattr(
            jm._apply_partial_rope, "_moespresso_dsv4_attn_seam_rope", False)
        composed = jm._moespresso_original_apply_partial_rope

        before = attention_seam_rope_call_counts()
        got = jm._apply_partial_rope(decode_x, rope, 3855)
        mx.eval(got)
        after = attention_seam_rope_call_counts()
        assert after["fused"] == before["fused"] + 1
        assert after["composed"] == before["composed"]
        _assert_bit_equal(
            mx, composed(decode_x, rope, 3855), got, "patched decode call")

        # Prefill-shaped rows serve the fused dispatch by default.
        monkeypatch.delenv(seam._PREFILL_ENV_FLAG, raising=False)
        before = attention_seam_rope_call_counts()
        got_prefill = jm._apply_partial_rope(prefill_x, rope, 0)
        mx.eval(got_prefill)
        after = attention_seam_rope_call_counts()
        assert after["fused"] == before["fused"] + 1
        assert after["composed"] == before["composed"]
        _assert_bit_equal(
            mx, composed(prefill_x, rope, 0), got_prefill,
            "patched prefill call")

        # The prefill kill switch restores the composed path for
        # prefill-shaped rows only.
        monkeypatch.setenv(seam._PREFILL_ENV_FLAG, "0")
        before = attention_seam_rope_call_counts()
        mx.eval(jm._apply_partial_rope(prefill_x, rope, 0))
        after = attention_seam_rope_call_counts()
        assert after["fused"] == before["fused"]
        assert after["composed"] == before["composed"] + 1
        monkeypatch.delenv(seam._PREFILL_ENV_FLAG)

        # A foreign rope object fails closed to the composed path.
        class _OtherRope:
            dims = 64
            inv_freq = rope.inv_freq

            def __call__(self, x, offset=0, inverse=False, positions=None):
                return x

        before = attention_seam_rope_call_counts()
        jm._apply_partial_rope(decode_x, _OtherRope(), 3855)
        after = attention_seam_rope_call_counts()
        assert after["fused"] == before["fused"]
        assert after["composed"] == before["composed"] + 1

        # The kill switch forces the composed path per call.
        monkeypatch.setenv(seam._FAMILY_ENV_FLAG, "0")
        before = attention_seam_rope_call_counts()
        killed = jm._apply_partial_rope(decode_x, rope, 3855)
        mx.eval(killed)
        after = attention_seam_rope_call_counts()
        assert after["fused"] == before["fused"]
        assert after["composed"] == before["composed"] + 1
        _assert_bit_equal(
            mx, composed(decode_x, rope, 3855), killed, "kill-switch call")
        monkeypatch.delenv(seam._FAMILY_ENV_FLAG)


def test_seam_rope_patch_is_idempotent():
    _require_metal()
    jm = _jang_model()
    with _RestoredRopePatch(jm):
        first = jm._apply_partial_rope
        assert _patch_deepseek_v4_attention_seam_rope(SimpleNamespace()) is True
        assert jm._apply_partial_rope is first


def test_seam_rope_counters_export_keys():
    counts = attention_seam_rope_call_counts()
    assert set(counts) == {"fused", "composed"}
    assert counts is not _ATTN_SEAM_ROPE_CALL_COUNTS
    from moespresso.runtime.deepseek_v4.speed_stats import _COUNT_KEYS

    assert "attn_seam_rope_fused_calls" in _COUNT_KEYS
    assert "attn_seam_rope_composed_calls" in _COUNT_KEYS
