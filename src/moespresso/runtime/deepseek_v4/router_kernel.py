"""Fused DS4 decode router select kernels.

The DS4 MoE router at the single-token decode shape is a small matmul
followed by a chain of tiny dispatches: the compiled sqrtsoftplus scoring
segment, the top-k ``mx.argpartition``, the index slice-cast, the
``take_along_axis`` gather, the renorm sum, the divide/scale pair, and the
downstream uint32 index cast. Everything except the matmul and the
selection is launch cost over a few hundred floats. This module collapses
that chain into two wide dispatches around the composed ``mx.argpartition``:

- ``fused_score_head`` runs the pre-selection elementwise segment
  (``sqrt(log1p(exp(gates)))``, the bias add, and the negation feeding the
  selection) in one dispatch and also emits the unbiased scores the
  post-selection tail needs.
- ``fused_topk_weights`` runs the post-selection tail (the selected-score
  gather, the renorm sum, the divide, and the routed scaling multiply) in
  one dispatch, consuming the ``mx.argpartition`` output ids directly.

The matmul and the argpartition stay composed, so the selected set and
order match the stock path exactly, ties included: the argpartition input
equals the composed ``-(scores + bias)`` bit for bit and the selection op
is shared.

Bit identity is a per-op transcription against the compiled
``sqrtsoftplus_select`` (custom kernels and MLX's own kernels compile with
fast math disabled, so the same functions round the same way):

- ``exp`` and ``sqrt`` are ``metal::precise::exp`` and
  ``metal::precise::sqrt``, matching MLX's Exp and Sqrt functors.
- ``log1p`` is the MLX metal helper algorithm transcribed exactly: with
  ``xp1 = 1 + x``, saturate to infinity, return ``x`` when ``xp1 == 1``,
  else ``x * (log(xp1) / (xp1 - 1))``.
- The bias add, the negation, the divide, and the scale multiply each
  round once at the composed-op boundary with fp contraction disabled.
- The renorm sum mirrors ``row_reduce_small``, the reduce kernel MLX
  selects for a contiguous sum over a row of at most 64 elements with no
  non-row reductions: one thread accumulates the whole row sequentially
  into a float32 total initialized at +0.0. Selection widths above 64
  pick a different reduce kernel and fail closed to the composed chain.

``MOESPRESSO_DSV4_ROUTER_TRIMS=0`` kills every router trim;
``MOESPRESSO_DSV4_ROUTER_PRECAST=0`` kills only the hoisted router weight
operand; ``MOESPRESSO_DSV4_ROUTER_SELECT_TAIL=0`` kills only the two
select kernels. All three are read per call.
"""

from __future__ import annotations

import math
import os


_ROUTER_TRIMS_ENV = "MOESPRESSO_DSV4_ROUTER_TRIMS"
_ROUTER_PRECAST_ENV = "MOESPRESSO_DSV4_ROUTER_PRECAST"
_ROUTER_SELECT_ENV = "MOESPRESSO_DSV4_ROUTER_SELECT_TAIL"

# row_reduce_small accumulates a whole row in one thread only while the
# row fits 64 elements; wider selections dispatch a different reduce
# kernel with a different order, so they stay composed.
_MAX_TOPK = 64

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


def router_trims_enabled() -> bool:
    """Return True unless the router trim family is killed."""
    return os.environ.get(_ROUTER_TRIMS_ENV, "1") != "0"


def router_precast_enabled() -> bool:
    """Return True when the hoisted router weight operand may engage."""
    if not router_trims_enabled():
        return False
    return os.environ.get(_ROUTER_PRECAST_ENV, "1") != "0"


def router_select_enabled() -> bool:
    """Return True when the fused router select kernels may engage."""
    if not router_trims_enabled():
        return False
    if os.environ.get(_ROUTER_SELECT_ENV, "1") != "0":
        return _metal_available()
    return False


def _f32_hex(value: float) -> str:
    return float(value).hex() + "f"


_LOG1P_HEADER = """
// This function derives from the MLX metal log1p(float) helper at commit
// 39886de461d4f014cd60a3a68ad42fd8d776754a under the MIT license.
// Copyright © 2023 Apple Inc. This attribution applies only to
// moespresso_dsv4_log1p. See THIRD-PARTY-NOTICES and LICENSE-MIT.
// The helper
// calls metal::log, which rounds like the precise variant here because
// MLX compiles runtime kernels with fast math disabled.
METAL_FUNC float moespresso_dsv4_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == metal::numeric_limits<float>::infinity()) {
        return metal::numeric_limits<float>::infinity();
    }
    if (xp1 == 1.0f) {
        return x;
    }
    return x * (metal::log(xp1) / (xp1 - 1.0f));
}
"""

_HEAD_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n = (uint)gates_shape[0];
    if (i >= n) {
        return;
    }
    {
        #pragma clang fp contract(off)
        float g = gates[i];
        float e = metal::precise::exp(g);
        float l = moespresso_dsv4_log1p(e);
        float s = metal::precise::sqrt(l);
        float b = s + float(bias[i]);
        neg[i] = -b;
        orig[i] = s;
    }
"""

_WEIGHTS_SOURCE = """
    // One simdgroup. Every lane recomputes the identical sequential renorm
    // sum (the row_reduce_small order: one thread, a float32 total
    // initialized at +0.0, terms accumulated in selection order), so no
    // cross-lane reduction or broadcast is needed at these widths.
    uint lane = thread_position_in_grid.x;
    uint k = (uint)sel_shape[sel_ndim - 1];
    float total = 0.0f;
    {
        #pragma clang fp contract(off)
        for (uint j = 0; j < k; j++) {
            float v = orig[(uint)sel[j]];
            total = v + total;
        }
        for (uint j = lane; j < k; j += 32u) {
            float t = orig[(uint)sel[j]];
            float w = t / total;
            w = w * SCALING;
            weights[j] = w;
        }
    }
"""

_HEAD_KERNEL: object | None = None

# One compiled weights kernel per routed scaling factor; the factor is a
# per-model constant baked into the source, and each value carries its own
# kernel name (a pipeline is cached by name).
_WEIGHTS_KERNELS: dict[float, object] = {}


def _get_head_kernel():
    global _HEAD_KERNEL
    if _HEAD_KERNEL is None:
        import mlx.core as mx

        _HEAD_KERNEL = mx.fast.metal_kernel(
            name="moespresso_dsv4_router_score_head",
            input_names=["gates", "bias"],
            output_names=["neg", "orig"],
            source=_HEAD_SOURCE,
            header=_LOG1P_HEADER,
        )
    return _HEAD_KERNEL


def _get_weights_kernel(scaling: float):
    key = float(scaling)
    kernel = _WEIGHTS_KERNELS.get(key)
    if kernel is None:
        import mlx.core as mx

        kernel = mx.fast.metal_kernel(
            name=f"moespresso_dsv4_router_topk_weights_{len(_WEIGHTS_KERNELS)}",
            input_names=["orig", "sel"],
            output_names=["weights"],
            source=_WEIGHTS_SOURCE.replace("SCALING", _f32_hex(scaling)),
        )
        _WEIGHTS_KERNELS[key] = kernel
    return kernel


def _bias_dtypes():
    import mlx.core as mx

    return (mx.float32, mx.float16, mx.bfloat16)


def _index_dtypes():
    import mlx.core as mx

    return (mx.uint32, mx.int32)


def score_head_eligible(gates, bias) -> bool:
    """Return True when the router score head can run fused.

    Eligibility is the single-token decode router shape: float32
    ``[1, 1, n_experts]`` gate logits and a one-dimensional bias of the
    same expert count in a supported dtype. Everything else fails closed
    to the composed chain.
    """
    if not router_select_enabled():
        return False
    import mlx.core as mx

    if gates.ndim != 3 or bias.ndim != 1:
        return False
    batch, tokens, n_experts = (int(v) for v in gates.shape)
    if batch != 1 or tokens != 1 or n_experts <= 0:
        return False
    if int(bias.shape[0]) != n_experts:
        return False
    if gates.dtype != mx.float32:
        return False
    if bias.dtype not in _bias_dtypes():
        return False
    return True


def fused_score_head(gates, bias):
    """Run the pre-selection router scoring segment in one dispatch.

    Args:
        gates: float32 ``[1, 1, n_experts]`` router gate logits.
        bias: ``[n_experts]`` selection bias in float32, float16, or
            bfloat16; the kernel applies the float32 promotion of the
            composed add.

    Returns:
        ``(neg, orig)``: float32 ``[1, 1, n_experts]`` arrays. ``neg`` is
        the negated biased score row, bit-identical to the composed
        ``-(sqrt(log1p(exp(gates))) + bias)`` and ready for
        ``mx.argpartition``; ``orig`` is the unbiased score row feeding
        the post-selection weights.
    """
    import mlx.core as mx

    if gates.ndim != 3:
        raise ValueError("gates must have shape [1, 1, n_experts]")
    batch, tokens, n_experts = (int(v) for v in gates.shape)
    if batch != 1 or tokens != 1:
        raise ValueError("the fused score head handles one decode token")
    if n_experts <= 0:
        raise ValueError("gates must cover at least one expert")
    if gates.dtype != mx.float32:
        raise ValueError("gates must be float32")
    if bias.ndim != 1 or int(bias.shape[0]) != n_experts:
        raise ValueError("bias must have shape [n_experts]")
    if bias.dtype not in _bias_dtypes():
        raise ValueError("bias must be float32, float16, or bfloat16")

    groups = (n_experts + 255) // 256
    neg, orig = _get_head_kernel()(
        inputs=[gates.reshape(n_experts), bias],
        template=[("B", bias.dtype)],
        output_shapes=[(n_experts,), (n_experts,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(256 * groups, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return neg.reshape(1, 1, n_experts), orig.reshape(1, 1, n_experts)


def topk_weights_eligible(orig, sel) -> bool:
    """Return True when the post-selection weights can run fused.

    Eligibility is the single-token decode selection shape: the float32
    ``[1, 1, n_experts]`` unbiased score row, and ``[1, 1, k]`` selected
    ids from ``mx.argpartition`` with ``1 < k <= 64`` inside the
    transcribed row_reduce_small envelope. Everything else fails closed
    to the composed chain.
    """
    if not router_select_enabled():
        return False
    import mlx.core as mx

    if orig.ndim != 3 or sel.ndim != 3:
        return False
    batch, tokens, n_experts = (int(v) for v in orig.shape)
    if batch != 1 or tokens != 1 or n_experts <= 0:
        return False
    if orig.dtype != mx.float32:
        return False
    if int(sel.shape[0]) != 1 or int(sel.shape[1]) != 1:
        return False
    k = int(sel.shape[2])
    if not 1 < k <= min(_MAX_TOPK, n_experts):
        return False
    if sel.dtype not in _index_dtypes():
        return False
    return True


def fused_topk_weights(orig, sel, *, scaling: float):
    """Finish the router select tail in one dispatch.

    Args:
        orig: float32 ``[1, 1, n_experts]`` unbiased score row from
            ``fused_score_head``.
        sel: ``[1, 1, k]`` selected expert ids from ``mx.argpartition``
            over the negated biased scores, in uint32 or int32; every id
            must index ``orig`` (the argpartition contract).
        scaling: routed scaling factor applied after the renorm.

    Returns:
        float32 ``[1, 1, k]`` route weights, bit-identical to the
        composed ``take_along_axis`` / renorm-sum / divide / scale chain.
    """
    import mlx.core as mx

    if orig.ndim != 3:
        raise ValueError("orig must have shape [1, 1, n_experts]")
    batch, tokens, n_experts = (int(v) for v in orig.shape)
    if batch != 1 or tokens != 1 or n_experts <= 0:
        raise ValueError("the fused select tail handles one decode token")
    if orig.dtype != mx.float32:
        raise ValueError("orig must be float32")
    if sel.ndim != 3 or int(sel.shape[0]) != 1 or int(sel.shape[1]) != 1:
        raise ValueError("sel must have shape [1, 1, k]")
    k = int(sel.shape[2])
    if not 1 < k <= min(_MAX_TOPK, n_experts):
        raise ValueError(
            "the select tail transcription covers 2 to 64 selected experts")
    if sel.dtype not in _index_dtypes():
        raise ValueError("sel must be uint32 or int32")
    scaling = float(scaling)
    if not math.isfinite(scaling):
        raise ValueError("scaling must be finite")

    weights, = _get_weights_kernel(scaling)(
        inputs=[orig.reshape(n_experts), sel.reshape(k)],
        template=[("I", sel.dtype)],
        output_shapes=[(k,)],
        output_dtypes=[mx.float32],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
    )
    return weights.reshape(1, 1, k)
