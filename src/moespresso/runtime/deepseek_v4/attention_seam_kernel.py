"""Fused DS4 decode attention seam kernels.

The decode ledger's attention seam glue is near-zero-traffic composed work:
each partial-rope call site (query rope, KV rope, the inverse rope on the
attention output, and the indexer query rope) expands into roughly a dozen
small elementwise dispatches (split, arange, the position-frequency outer
product, cos, sin, casts, four rotation multiplies, stack, concat) whose
fenced cost comes from dispatch count rather than arithmetic. This module fuses one
partial-rope call into one Metal dispatch. Decode row counts (up to 64
flattened rows) are always eligible; prefill row counts are eligible by
default because the kernel is row-parallel with 64-bit element indexing,
so the geometry is unchanged at any served row count, and the composed
prefill rope assemblies fence at 222.8 ms of removable data movement per
anchor chunk across their six call sites.

Bit identity is a per-op transcription of the composed sequence, following
the recorded rope facts from the fused decode island:

- Positions and angles stay float32: ``theta = position * inv_freq[p]`` is
  the same single float32 multiply the composed outer product performs.
- ``metal::precise::cos``/``sin`` match the MLX Cos and Sin kernels. The
  inverse form negates the float32 sine before the cast, matching the
  composed ``sin = -sin`` then ``astype`` order (negation is exact in
  either order).
- The rotation rounds every composed-op boundary: a float32 multiply or
  add/sub of two 16-bit values rounded once to the element type equals the
  native 16-bit op (the double rounding is innocuous because float32
  carries more than twice the significand bits), and float32 rows run the
  same float32 ops with fp contraction disabled so the compiler cannot
  fuse a rotation multiply into the following add.
- The non-rotated leading channels copy through unchanged, replacing the
  composed split and concat.

Two operand forms exist and each carries its form in the kernel name (a
pipeline is cached by name, and calling a kernel compiled for one operand
layout with another reads reinterpreted bytes): the ``off`` form takes an
int32 ``[offset, seq_len]`` params array and reproduces
``mx.arange(offset, offset + L)`` in-kernel (engaged only while
``offset + L`` stays inside float32's exact-integer range), and the ``pos``
form takes the caller's float32 positions row. The element dtype is a
template parameter, so each dtype compiles its own pipeline.

``MOESPRESSO_DSV4_ATTN_SEAM_FUSED=0`` kills every attention seam fusion;
``MOESPRESSO_DSV4_ATTN_SEAM_ROPE=0`` kills only the rope seam;
``MOESPRESSO_DSV4_SEAM_ROPE_PREFILL=0`` restores the decode-only row cap
so prefill-shaped calls stay composed. Callers fall back to the composed
path on any precondition miss.
"""

from __future__ import annotations

import os

_FAMILY_ENV_FLAG = "MOESPRESSO_DSV4_ATTN_SEAM_FUSED"
_ROPE_ENV_FLAG = "MOESPRESSO_DSV4_ATTN_SEAM_ROPE"
_PREFILL_ENV_FLAG = "MOESPRESSO_DSV4_SEAM_ROPE_PREFILL"

_ROPE_DIM = 64
_ROPE_FREQS = _ROPE_DIM // 2

# Decode row cap: 64 query heads is the widest served decode call, and
# the cap is the eligibility contract when the prefill extension is
# killed.
_MAX_ROWS_DECODE = 64

# Structural row ceiling for prefill-shaped calls. The kernel is
# row-parallel (one grid row per flattened input row), reads the row
# count through an int32 shape entry, and indexes element offsets in
# 64 bits, so the geometry holds to the int32 ceiling; served prefill
# shapes (64 heads times the chunk row count) stay orders of magnitude
# below it.
_MAX_ROWS_PREFILL = (1 << 31) - 1

# The composed path materializes positions with mx.arange in float32; the
# offset form reproduces that in-kernel, which is exact only while every
# position is an exactly representable float32 integer.
_MAX_EXACT_POSITION = 1 << 24

_ROPE_SOURCE_TEMPLATE = """
    uint tx = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint width = (uint)x_shape[1];
    uint rows = (uint)x_shape[0];
    uint quads = width / 4u;
    if (tx >= quads || row >= rows) {
        return;
    }
    uint d0 = 4u * tx;
    device const T *xrow = x + (uint64_t)row * width;
    device T *orow = out + (uint64_t)row * width;
    uint tail = width - 64u;
    if (d0 < tail) {
        for (ushort j = 0; j < 4; j++) {
            orow[d0 + j] = xrow[d0 + j];
        }
        return;
    }
    float pos = POS_EXPR;
    {
        #pragma clang fp contract(off)
        for (ushort j = 0; j < 2; j++) {
            uint dd = d0 + 2u * j;
            uint p = (dd - tail) / 2u;
            float theta = pos * inv_freq[p];
            float c_f = metal::precise::cos(theta);
            float s_f = SIGN(metal::precise::sin(theta));
            T c = T(c_f);
            T s = T(s_f);
            T x0 = xrow[dd];
            T x1 = xrow[dd + 1u];
            T m0 = T(float(x0) * float(c));
            T m1 = T(float(x1) * float(s));
            T m2 = T(float(x0) * float(s));
            T m3 = T(float(x1) * float(c));
            orow[dd] = T(float(m0) - float(m1));
            orow[dd + 1u] = T(float(m2) + float(m3));
        }
    }
"""

_POS_EXPR_OFFSET = "(float)(params[0] + (int)(row % (uint)params[1]))"
_POS_EXPR_ARRAY = "positions[row % (uint)positions_shape[0]]"

_ROPE_KERNELS: dict[tuple[str, bool], object] = {}

_METAL_AVAILABLE: bool | None = None


def _metal_available() -> bool:
    global _METAL_AVAILABLE
    if _METAL_AVAILABLE is None:
        try:
            import mlx.core as mx

            _METAL_AVAILABLE = bool(mx.metal.is_available())
        except ImportError:
            _METAL_AVAILABLE = False
    return _METAL_AVAILABLE


def rope_seam_enabled() -> bool:
    """Return True when the fused rope seam may engage."""
    if os.environ.get(_FAMILY_ENV_FLAG, "1") == "0":
        return False
    if os.environ.get(_ROPE_ENV_FLAG, "1") == "0":
        return False
    return _metal_available()


def prefill_rope_seam_enabled() -> bool:
    """Return True when prefill row counts may serve the fused dispatch.

    Default on: the fused form is bit-identical to the composed chain at
    every served prefill shape and dtype (zero mismatched bits across the
    six fenced call sites), so the extension is an eligibility widening on
    an existing kernel rather than a math change. The served anchor A/B
    measured the chunk wall at 14.538 s against 14.677 with the cap
    restored (medians of three, all 192 prefill rope calls converted, the
    token rail identical on both arms).
    ``MOESPRESSO_DSV4_SEAM_ROPE_PREFILL=0`` restores the decode-only row
    cap.
    """
    return os.environ.get(_PREFILL_ENV_FLAG, "1") != "0"


def _element_dtypes():
    import mlx.core as mx

    return (mx.float32, mx.float16, mx.bfloat16)


def _get_rope_kernel(form: str, inverse: bool):
    key = (form, bool(inverse))
    kernel = _ROPE_KERNELS.get(key)
    if kernel is None:
        import mlx.core as mx

        if form == "off":
            input_names = ["x", "inv_freq", "params"]
            pos_expr = _POS_EXPR_OFFSET
        else:
            input_names = ["x", "inv_freq", "positions"]
            pos_expr = _POS_EXPR_ARRAY
        source = (
            _ROPE_SOURCE_TEMPLATE
            .replace("POS_EXPR", pos_expr)
            .replace("SIGN(", "-(" if inverse else "(")
        )
        direction = "inv" if inverse else "fwd"
        kernel = mx.fast.metal_kernel(
            name=f"moespresso_dsv4_attn_seam_rope_{form}_{direction}",
            input_names=input_names,
            output_names=["out"],
            source=source,
        )
        _ROPE_KERNELS[key] = kernel
    return kernel


def partial_rope_eligible(x, inv_freq, *, offset, positions) -> bool:
    """Return True when a partial-rope call can run the fused dispatch.

    Eligibility is the DS4 rope contract: a float32, float16, or bfloat16
    operand whose last axis carries at least the 64-channel rope tail on a
    float4-aligned width, and positions that are either an in-range
    integer offset or a caller-supplied float32 positions row matching the
    sequence axis. Row counts up to 64 (the widest decode call) are always
    eligible; larger prefill-shaped row counts are eligible up to the
    kernel's structural int32 ceiling unless
    ``MOESPRESSO_DSV4_SEAM_ROPE_PREFILL=0`` restores the decode-only cap.
    """
    import mlx.core as mx

    if x.ndim < 2 or x.dtype not in _element_dtypes():
        return False
    width = int(x.shape[-1])
    if width < _ROPE_DIM or width % 4 != 0:
        return False
    seq_len = int(x.shape[-2])
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    max_rows = (
        _MAX_ROWS_PREFILL if prefill_rope_seam_enabled() else _MAX_ROWS_DECODE
    )
    if not 1 <= rows <= max_rows:
        return False
    if (
        inv_freq.ndim != 1
        or int(inv_freq.shape[0]) != _ROPE_FREQS
        or inv_freq.dtype != mx.float32
    ):
        return False
    if positions is None:
        if not isinstance(offset, int):
            return False
        if offset < 0 or offset + seq_len > _MAX_EXACT_POSITION:
            return False
        return True
    if not isinstance(positions, mx.array):
        return False
    if positions.dtype != mx.float32 or positions.size != seq_len:
        return False
    return True


def fused_partial_rope(x, inv_freq, *, offset=0, inverse=False, positions=None):
    """Apply the DS4 partial rope in one Metal dispatch.

    Args:
        x: ``[..., seq, width]`` float32, float16, or bfloat16 rows; the
            final 64 channels rotate and the leading channels copy through.
        inv_freq: float32 ``[32]`` rope inverse frequencies (YaRN folded).
        offset: absolute position of the first sequence row; used when
            ``positions`` is None.
        inverse: negate the rotation, matching the composed inverse rope.
        positions: optional float32 positions for the sequence axis,
            replacing the offset-derived arange.

    Returns:
        An array of the same shape and dtype, bit-identical to the
        composed ``_apply_partial_rope`` sequence.
    """
    import mlx.core as mx

    if not partial_rope_eligible(x, inv_freq, offset=offset, positions=positions):
        raise ValueError("operands are outside the fused partial-rope contract")

    width = int(x.shape[-1])
    seq_len = int(x.shape[-2])
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    if positions is None:
        form = "off"
        pos_input = mx.array([int(offset), seq_len], dtype=mx.int32)
    else:
        form = "pos"
        pos_input = positions.reshape(seq_len)
    out, = _get_rope_kernel(form, inverse)(
        inputs=[x.reshape(rows, width), inv_freq, pos_input],
        template=[("T", x.dtype)],
        output_shapes=[(rows, width)],
        output_dtypes=[x.dtype],
        grid=(width // 4, rows, 1),
        threadgroup=(min(width // 4, 128), 1, 1),
    )
    return out.reshape(x.shape)
