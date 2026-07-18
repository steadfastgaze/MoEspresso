"""Fused DS4 decode indexer pre-top-k kernel.

The DS4 decode indexer chain between the rope projection and the top-k
selection is launch-bound: the query QAT lattice alone is about a dozen
small dispatches and the score chain (query-pool dot products, relu, scale,
head weighting, head sum) is another eight, all over a few MFLOP of work.
This module fuses that whole chain into one Metal dispatch for the
single-token decode shape: the kernel takes the post-rope query heads, the
cached QAT'd pool rows, and the projected head weights, applies the query
QAT in-kernel, and emits the float32 score row. Top-k selection stays
outside the kernel.

The in-kernel QAT is a bit-exact transcription of ``_dsv4_indexer_qat``
(the DS4 e2m1 lattice contract). The parts that matter for bit identity:

- ``mx.hadamard_transform`` reduces to radix-2 butterflies applied in
  increasing-stride order with float32 intermediates, so any radix
  decomposition that keeps the stride order reproduces it bit for bit. The
  kernel runs the transform per simdgroup with four elements per lane and
  ``simd_shuffle_xor`` for the cross-lane strides.
- The group scale is ``exp(ceil(log(amax / 6) / log(2)) * log(2))`` through
  ``metal::precise::log`` and ``metal::precise::exp``, matching MLX's Log
  and Exp kernels. Rewriting it as ``exp2(ceil(log2(...)))`` is cleaner
  math but is not bit-exact: for large exponents ``exp(k * log(2))`` in
  float32 is several ulp away from an exact power of two, and the lattice
  contract keeps that value.
- Nearest-lattice selection breaks ties like ``mx.argmin``: the first
  (lower) lattice value wins. A round-to-even tie break is not equivalent.
- The clip and the amax floor mirror MLX's Maximum/Minimum semantics,
  including NaN passthrough.

Bit identity against ``_dsv4_indexer_qat`` is enforced by a dedicated
parity test through ``indexer_qat_rows``, a QAT-only kernel variant that
shares the same Metal helper the fused score kernel calls.

The second kernel (``fused_score_tail``) is the bit-exact score-chain tail
for the fixed-state decode path: everything between the score matmul and
the top-k selection (relu, score scale, head-weight cast and scale, the
head-axis sum, the capacity pad, and the selection-input negation) in one
wide dispatch, keeping the matmul and ``mx.argpartition`` composed. See
the section comment above the kernel for the transcription facts.
"""

from __future__ import annotations

import math
import os


# Gate for the fused decode pre-top-k path (query QAT + score chain in one
# dispatch). Default off: the fenced served-layer A/B (layer 2, 961 pooled
# rows, alternating 20-repeat blocks) measured the indexer segment at
# 0.544 ms fused vs 0.724 ms composed (1.33x), below the 1.8x retention
# bar, even though the one dispatch replaces the ~20-dispatch query QAT
# plus score chain and the isolated chain microbench improves 1.63x
# (0.450 -> 0.276 ms). The per-eval command-buffer round trip and the
# chain's remaining ops (compressor, wq_b, rope, weights_proj, top-k)
# floor the fenced segment above the bar, so absorbing these dispatches
# pays off only from a coarser fused decode island that also absorbs the
# eval boundary. Q1 holds the 16/17 known-miss anchor (blocking 2) with
# the path forced on, so the kernel is correctness-safe to enable from
# such an island; the parity tests keep it honest.
_ENABLED = False

_HEAD_DIM = 128
# The score kernel stages the QAT'd query heads in a fixed threadgroup
# buffer sized for the DS4 index head count.
_MAX_HEADS = 64
_ROWS_PER_GROUP = 4
_THREADS_PER_GROUP = 128

# Constants from _dsv4_indexer_qat, embedded as float32 hex literals so the
# compiled source carries exactly the values the MLX reference ops see.
_HADAMARD_SCALE = 0.08838834764831845
_AMAX_FLOOR = 7.052966104933725e-38
_LN2 = math.log(2.0)

_QAT_HEADER_TEMPLATE = """
// Bit-exact transcription of _dsv4_indexer_qat for one 128-wide row held
// across one simdgroup, four elements per lane (lane l holds elements
// [4l, 4l+3]). Every lane of the simdgroup must be active.
METAL_FUNC float4 moespresso_dsv4_indexer_qat128(float4 v, ushort lane) {
    // Walsh-Hadamard transform of size 128 as radix-2 butterflies in
    // increasing-stride order, float32 throughout, matching
    // mx.hadamard_transform. Strides 1 and 2 stay inside the lane;
    // strides 4..64 pair lanes through simd_shuffle_xor.
    float a0 = v.x + v.y;
    float a1 = v.x - v.y;
    float a2 = v.z + v.w;
    float a3 = v.z - v.w;
    float4 x = float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);
    for (ushort m = 1; m <= 16; m <<= 1) {
        float4 other = simd_shuffle_xor(x, m);
        x = (lane & m) ? (other - x) : (x + other);
    }
    x = x * HADAMARD_SCALE;

    // e2m1 activation roundtrip over 32-element groups. Lanes [8g, 8g+7]
    // hold group g; the xor reductions keep the max inside the group.
    float amax = metal::max(
        metal::max(metal::fabs(x.x), metal::fabs(x.y)),
        metal::max(metal::fabs(x.z), metal::fabs(x.w)));
    amax = metal::max(amax, simd_shuffle_xor(amax, 1));
    amax = metal::max(amax, simd_shuffle_xor(amax, 2));
    amax = metal::max(amax, simd_shuffle_xor(amax, 4));
    // mx.maximum(amax, floor) with MLX's NaN-passthrough semantics.
    amax = metal::isnan(amax) ? amax : (amax > AMAX_FLOOR ? amax : AMAX_FLOOR);
    float log2_scale = metal::precise::log(amax / 6.0f) / LN2;
    float scale = metal::precise::exp(metal::ceil(log2_scale) * LN2);

    const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float4 out;
    for (ushort j = 0; j < 4; j++) {
        float n = x[j] / scale;
        // mx.clip(n, -6, 6) is maximum then minimum, NaN passthrough.
        n = (metal::isnan(n) || n > -6.0f) ? n : -6.0f;
        n = (metal::isnan(n) || n < 6.0f) ? n : 6.0f;
        float absn = metal::fabs(n);
        // mx.argmin tie break: the first (lower) lattice value wins.
        float best = metal::fabs(absn - values[0]);
        float dq = values[0];
        for (ushort k = 1; k < 8; k++) {
            float d = metal::fabs(absn - values[k]);
            if (d < best) {
                best = d;
                dq = values[k];
            }
        }
        dq = n < 0.0f ? -dq : dq;
        out[j] = dq * scale;
    }
    return out;
}
"""

_SCORE_SOURCE_TEMPLATE = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x * 4u + sg;

    uint n_heads = (uint)q_shape[1];
    uint n_rows  = (uint)pooled_shape[1];

    // The four simdgroups cooperatively QAT the query heads into
    // threadgroup memory once per threadgroup (the buffer caps n_heads at
    // 64, the DS4 index head count), then each simdgroup scores one
    // pooled row against the shared lattice bits. The cooperative pass
    // measured half the per-dispatch GPU time of re-deriving the QAT in
    // registers per simdgroup (0.055 vs 0.111 ms amortized at 961 rows).
    threadgroup float qbuf[8192];

    for (uint h = sg; h < n_heads; h += 4u) {
        device const T *qrow = q + (uint64_t)h * 128u;
        float4 v = float4(
            float(qrow[4u * lane + 0u]),
            float(qrow[4u * lane + 1u]),
            float(qrow[4u * lane + 2u]),
            float(qrow[4u * lane + 3u]));
        float4 qq = moespresso_dsv4_indexer_qat128(v, ushort(lane));
        threadgroup float4 *q4 =
            (threadgroup float4 *)(qbuf + (uint64_t)h * 128u);
        q4[lane] = qq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The row-tail return sits after the only barrier, so it is safe.
    if (row >= n_rows) return;

    // One simdgroup owns one pooled row. head_dim is 128, so each of the
    // 32 lanes holds four row elements as one float4 and every head's dot
    // product reduces with a single simd_sum.
    device const float4 *row4 =
        (device const float4 *)(pooled + (uint64_t)row * 128u);
    float4 p = row4[lane];

    float acc = 0.0f;
    for (uint h = 0u; h < n_heads; h++) {
        threadgroup const float4 *q4 =
            (threadgroup const float4 *)(qbuf + (uint64_t)h * 128u);
        float partial = dot(q4[lane], p);
        float relu = max(simd_sum(partial), 0.0f);
        acc += relu * HEAD_SCALE * weights[h];
    }
    if (lane == 0u) {
        scores[row] = acc;
    }
"""

_QAT_ROWS_SOURCE = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x * 4u + sg;
    uint n_rows = (uint)q_shape[0];

    if (row >= n_rows) return;

    device const T *qrow = q + (uint64_t)row * 128u;
    float4 v = float4(
        float(qrow[4u * lane + 0u]),
        float(qrow[4u * lane + 1u]),
        float(qrow[4u * lane + 2u]),
        float(qrow[4u * lane + 3u]));
    float4 y = moespresso_dsv4_indexer_qat128(v, ushort(lane));
    device float *orow = out + (uint64_t)row * 128u;
    orow[4u * lane + 0u] = y.x;
    orow[4u * lane + 1u] = y.y;
    orow[4u * lane + 2u] = y.z;
    orow[4u * lane + 3u] = y.w;
"""


def _f32_hex(value: float) -> str:
    return float(value).hex() + "f"


def _qat_header() -> str:
    return (
        _QAT_HEADER_TEMPLATE
        .replace("HADAMARD_SCALE", _f32_hex(_HADAMARD_SCALE))
        .replace("AMAX_FLOOR", _f32_hex(_AMAX_FLOOR))
        .replace("LN2", _f32_hex(_LN2))
    )


# The score scale is a per-model constant (index head_dim ** -0.5), so it is
# baked into the compiled source instead of shipping a one-float buffer with
# every decode token. One kernel per distinct scale value.
_SCORE_KERNELS: dict[float, object] = {}

_QAT_ROWS_KERNEL: object | None = None

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


def _query_dtypes():
    import mlx.core as mx

    return (mx.float32, mx.float16, mx.bfloat16)


def _get_score_kernel(scale: float):
    kernel = _SCORE_KERNELS.get(scale)
    if kernel is None:
        import mlx.core as mx

        kernel = mx.fast.metal_kernel(
            name=f"moespresso_dsv4_fused_qat_indexer_scores_{len(_SCORE_KERNELS)}",
            input_names=["q", "pooled", "weights"],
            output_names=["scores"],
            source=_SCORE_SOURCE_TEMPLATE.replace("HEAD_SCALE", _f32_hex(scale)),
            header=_qat_header(),
        )
        _SCORE_KERNELS[scale] = kernel
    return kernel


def _get_qat_rows_kernel():
    global _QAT_ROWS_KERNEL
    if _QAT_ROWS_KERNEL is None:
        import mlx.core as mx

        _QAT_ROWS_KERNEL = mx.fast.metal_kernel(
            name="moespresso_dsv4_indexer_qat_rows",
            input_names=["q"],
            output_names=["out"],
            source=_QAT_ROWS_SOURCE,
            header=_qat_header(),
        )
    return _QAT_ROWS_KERNEL


def indexer_qat_rows(x):
    """Run the in-kernel DS4 indexer QAT over 128-wide rows.

    This is the parity surface for the fused kernel's QAT: it dispatches the
    same Metal helper the score kernel calls, so a bit compare against
    ``_dsv4_indexer_qat`` tests the shipped transcription. It is not wired
    into the serve path.

    Args:
        x: ``[..., 128]`` float32, float16, or bfloat16 rows.

    Returns:
        float32 array of the same shape, bit-identical to
        ``_dsv4_indexer_qat(mx, x)``.
    """
    import mlx.core as mx

    if x.ndim < 1 or int(x.shape[-1]) != _HEAD_DIM:
        raise ValueError("indexer QAT expects 128-wide rows")
    if x.dtype not in _query_dtypes():
        raise ValueError("indexer QAT rows must be float32, float16, or bfloat16")
    rows2d = x.reshape(-1, _HEAD_DIM)
    n_rows = int(rows2d.shape[0])
    if n_rows == 0:
        raise ValueError("indexer QAT expects at least one row")
    groups = (n_rows + _ROWS_PER_GROUP - 1) // _ROWS_PER_GROUP
    out, = _get_qat_rows_kernel()(
        inputs=[rows2d],
        template=[("T", x.dtype)],
        output_shapes=[(n_rows, _HEAD_DIM)],
        output_dtypes=[mx.float32],
        grid=(_THREADS_PER_GROUP * groups, 1, 1),
        threadgroup=(_THREADS_PER_GROUP, 1, 1),
    )
    return out.reshape(x.shape)


def fused_qat_scores_eligible(q, pooled_qat, weights) -> bool:
    """Return True when the decode pre-top-k chain can use the fused kernel.

    Eligibility is the single-token decode shape: one batch row, one query
    token (so every pooled row is visible and no mask is needed), the DS4
    index head width, a float query in a supported dtype, and float32
    pooled rows and head weights. The query is the post-rope, pre-QAT
    projection; the kernel applies the QAT itself.
    """
    if not _ENABLED or not _metal_available():
        return False
    import mlx.core as mx

    if q.ndim != 4 or pooled_qat.ndim != 3 or weights.ndim != 3:
        return False
    batch, n_heads, tokens, head_dim = (int(v) for v in q.shape)
    if batch != 1 or tokens != 1 or head_dim != _HEAD_DIM:
        return False
    if n_heads <= 0 or n_heads > _MAX_HEADS:
        return False
    if int(pooled_qat.shape[0]) != 1 or int(pooled_qat.shape[2]) != head_dim:
        return False
    if int(pooled_qat.shape[1]) <= 0:
        return False
    if tuple(int(v) for v in weights.shape) != (1, 1, n_heads):
        return False
    if q.dtype not in _query_dtypes():
        return False
    if pooled_qat.dtype != mx.float32 or weights.dtype != mx.float32:
        return False
    return True


def fused_qat_indexer_scores(q, pooled_qat, weights, scale):
    """Compute the DS4 decode indexer score row in one Metal dispatch.

    The kernel applies the query QAT lattice in-kernel (bit-exact with
    ``_dsv4_indexer_qat``) and then the score chain, replacing the roughly
    twenty dispatches the composed graph issues per decode token.

    Args:
        q: ``[1, n_heads, 1, 128]`` post-rope, pre-QAT indexer queries in
            float32, float16, or bfloat16.
        pooled_qat: float32 ``[1, n_rows, 128]`` QAT'd compressed pool rows.
        weights: float32 ``[1, 1, n_heads]`` projected head weights, already
            multiplied by ``n_heads ** -0.5``.
        scale: per-head score scale (``head_dim ** -0.5``).

    Returns:
        float32 ``[n_rows]`` scores:
        ``scores[r] = sum_h max(qat(q_h) . pooled_r, 0) * scale * weights[h]``.
    """
    import mlx.core as mx

    if q.ndim != 4:
        raise ValueError("q must have shape [1, n_heads, 1, 128]")
    batch, n_heads, tokens, head_dim = (int(v) for v in q.shape)
    if batch != 1 or tokens != 1:
        raise ValueError("fused indexer scoring handles one decode token")
    if head_dim != _HEAD_DIM:
        raise ValueError("DS4 fused indexer scoring requires head_dim=128")
    if n_heads <= 0 or n_heads > _MAX_HEADS:
        raise ValueError("q must carry between 1 and 64 heads")
    if (
        pooled_qat.ndim != 3
        or int(pooled_qat.shape[0]) != 1
        or int(pooled_qat.shape[2]) != head_dim
    ):
        raise ValueError("pooled_qat must have shape [1, n_rows, 128]")
    n_rows = int(pooled_qat.shape[1])
    if n_rows <= 0:
        raise ValueError("pooled_qat must contain at least one row")
    if tuple(int(v) for v in weights.shape) != (1, 1, n_heads):
        raise ValueError("weights must have shape [1, 1, n_heads]")
    if q.dtype not in _query_dtypes():
        raise ValueError("q must be float32, float16, or bfloat16")
    if pooled_qat.dtype != mx.float32 or weights.dtype != mx.float32:
        raise ValueError("pooled_qat and weights must be float32")

    groups = (n_rows + _ROWS_PER_GROUP - 1) // _ROWS_PER_GROUP
    scores, = _get_score_kernel(float(scale))(
        inputs=[q, pooled_qat, weights],
        template=[("T", q.dtype)],
        output_shapes=[(n_rows,)],
        output_dtypes=[mx.float32],
        grid=(_THREADS_PER_GROUP * groups, 1, 1),
        threadgroup=(_THREADS_PER_GROUP, 1, 1),
    )
    return scores


# Fused decode indexer score tail. The composed fixed-state chain between
# the score matmul and the top-k selection is about eight small dispatches
# over a [1, heads, 1, capacity] float32 buffer: relu, the score scale,
# the head-weight cast and scale, the broadcast head-weight multiply, the
# head-axis sum, the arange/compare/where capacity pad, and the negation
# that feeds `mx.argpartition`. One wide dispatch (one simdgroup per
# capacity column, multiple threadgroups) replaces them, while the matmul
# and the argpartition stay composed; keeping the selection composed
# preserves the stock selection order, which the parked fused top-k unit
# broke.
#
# Bit identity is a per-op transcription:
#
# - Every elementwise step rounds at the composed-op boundary with fp
#   contraction disabled, and `mx.maximum(x, 0)` keeps MLX's
#   NaN-passthrough semantics.
# - The head-weight operand is the raw projection output; the kernel
#   applies the float32 cast and the `n_heads ** -0.5` multiply in-kernel,
#   matching the composed `astype` plus scalar-multiply rounding.
# - The head-axis sum mirrors the reduction order of MLX's
#   `col_reduce_looped` BM=32/BN=32 kernel, the one the reduce dispatch
#   selects for a [1, heads, 1, capacity] float32 sum over the head axis
#   when 32 <= heads <= 256: lane l accumulates head rows l, l + 32, ...
#   sequentially into a float32 partial initialized at +0.0, and
#   `simd_sum` combines the 32 partials. Eligibility stays inside the DS4
#   head-count envelope the parity tests pin (32 to 64 heads); other head
#   counts select a different reduce kernel and fail closed to the
#   composed chain.
# - Valid columns carry the negated sum and the invalid tail carries
#   positive infinity, the exact negation of `pad_scores_to_capacity`;
#   negation is a sign-bit flip, so the argpartition input equals the
#   composed `-scores` bit for bit and the selected set and order match
#   the stock path exactly.
#
# `MOESPRESSO_DSV4_INDEXER_CHAIN_TRIMS=0` kills every indexer chain trim;
# `MOESPRESSO_DSV4_INDEXER_SCORE_TAIL=0` kills only this seam. Both are
# read per call.
_CHAIN_TRIMS_ENV = "MOESPRESSO_DSV4_INDEXER_CHAIN_TRIMS"
_SCORE_TAIL_ENV = "MOESPRESSO_DSV4_INDEXER_SCORE_TAIL"

# col_reduce_looped BM=32/BN=32 is selected for head-axis sums with at
# least 32 and at most 256 head rows; the DS4 index head count is 64 and
# the parity tests pin 32 through 64.
_TAIL_MIN_HEADS = 32
_TAIL_MAX_HEADS = 64

_TAIL_COLS_PER_GROUP = 8

_TAIL_SOURCE_TEMPLATE = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint col = threadgroup_position_in_grid.x * 8u + sg;

    uint n_heads = (uint)s_shape[0];
    uint capacity = (uint)s_shape[1];
    uint valid = metal::min((uint)params[1], capacity);
    if (col >= capacity) {
        return;
    }

    // Lane l owns head rows l, l + 32, ... (the col_reduce_looped BM=32
    // partial layout); each term rounds at every composed-op boundary.
    float partial = 0.0f;
    {
        #pragma clang fp contract(off)
        for (uint h = lane; h < n_heads; h += 32u) {
            float w = float(wraw[h]) * HEAD_NORM;
            float t = s[h * capacity + col];
            // mx.maximum(t, 0) with MLX's NaN-passthrough semantics.
            t = metal::isnan(t) ? t : (t > 0.0f ? t : 0.0f);
            t = t * SCORE_SCALE;
            t = t * w;
            partial = partial + t;
        }
    }
    float total = metal::simd_sum(partial);
    if (lane == 0u) {
        neg[col] = col < valid ? -total : as_type<float>(0x7F800000u);
    }
"""

# Kernels are cached per (scale, n_heads); both are per-model constants
# baked into the compiled source, so each pair carries its own kernel
# name (a pipeline is cached by name).
_TAIL_KERNELS: dict = {}


def score_tail_enabled() -> bool:
    """Return True when the fused score tail may engage."""
    if os.environ.get(_CHAIN_TRIMS_ENV, "1") == "0":
        return False
    if os.environ.get(_SCORE_TAIL_ENV, "1") == "0":
        return False
    return _metal_available()


def _get_tail_kernel(scale: float, n_heads: int):
    key = (float(scale), int(n_heads))
    kernel = _TAIL_KERNELS.get(key)
    if kernel is None:
        import mlx.core as mx

        head_norm = float(int(n_heads) ** -0.5)
        source = (
            _TAIL_SOURCE_TEMPLATE
            .replace("HEAD_NORM", _f32_hex(head_norm))
            .replace("SCORE_SCALE", _f32_hex(scale))
        )
        kernel = mx.fast.metal_kernel(
            name=f"moespresso_dsv4_indexer_score_tail_{len(_TAIL_KERNELS)}",
            input_names=["s", "wraw", "params"],
            output_names=["neg"],
            source=source,
        )
        _TAIL_KERNELS[key] = kernel
    return kernel


def score_tail_eligible(scores, weights_raw, params) -> bool:
    """Return True when the fixed-state score tail can run fused.

    Eligibility is the fixed-state single-token decode shape: a float32
    ``[1, n_heads, 1, capacity]`` score matmul output whose head count
    sits inside the parity-tested col_reduce_looped envelope, the raw
    head-weight projection row in a supported dtype, and the int32
    per-token params array carrying the valid-row count. Everything else
    fails closed to the composed chain.
    """
    if not score_tail_enabled():
        return False
    import mlx.core as mx

    if scores.ndim != 4 or weights_raw.ndim != 3:
        return False
    batch, n_heads, tokens, capacity = (int(v) for v in scores.shape)
    if batch != 1 or tokens != 1 or capacity <= 0:
        return False
    if not _TAIL_MIN_HEADS <= n_heads <= _TAIL_MAX_HEADS:
        return False
    if scores.dtype != mx.float32:
        return False
    if tuple(int(v) for v in weights_raw.shape) != (1, 1, n_heads):
        return False
    if weights_raw.dtype not in _query_dtypes():
        return False
    if params.dtype != mx.int32 or params.size < 2:
        return False
    return True


def fused_score_tail(scores, weights_raw, params, *, scale):
    """Finish the fixed-state indexer score chain in one wide dispatch.

    Args:
        scores: float32 ``[1, n_heads, 1, capacity]`` score matmul output
            over the fixed-state capacity buffer.
        weights_raw: ``[1, 1, n_heads]`` raw ``weights_proj`` output in
            float32, float16, or bfloat16; the kernel applies the float32
            cast and the ``n_heads ** -0.5`` scale.
        params: int32 per-token params array whose second element is the
            valid pool row count (``decode_step_params`` layout).
        scale: per-head score scale (``index_head_dim ** -0.5``).

    Returns:
        float32 ``[1, 1, capacity]`` negated padded scores: the valid
        prefix equals the composed ``-scores`` bit for bit and the
        invalid tail is positive infinity, ready for ``mx.argpartition``.
    """
    import mlx.core as mx

    if scores.ndim != 4:
        raise ValueError("scores must have shape [1, n_heads, 1, capacity]")
    batch, n_heads, tokens, capacity = (int(v) for v in scores.shape)
    if batch != 1 or tokens != 1:
        raise ValueError("the fused score tail handles one decode token")
    if capacity <= 0:
        raise ValueError("scores must cover at least one capacity column")
    if not _TAIL_MIN_HEADS <= n_heads <= _TAIL_MAX_HEADS:
        raise ValueError(
            "the score tail transcription covers 32 to 64 index heads")
    if scores.dtype != mx.float32:
        raise ValueError("scores must be float32")
    if tuple(int(v) for v in weights_raw.shape) != (1, 1, n_heads):
        raise ValueError("weights_raw must have shape [1, 1, n_heads]")
    if weights_raw.dtype not in _query_dtypes():
        raise ValueError(
            "weights_raw must be float32, float16, or bfloat16")
    if params.dtype != mx.int32 or params.size < 2:
        raise ValueError("params must be int32 with the row count at index 1")

    groups = (capacity + _TAIL_COLS_PER_GROUP - 1) // _TAIL_COLS_PER_GROUP
    neg, = _get_tail_kernel(float(scale), n_heads)(
        inputs=[
            scores.reshape(n_heads, capacity),
            weights_raw.reshape(n_heads),
            params,
        ],
        template=[("W", weights_raw.dtype)],
        output_shapes=[(capacity,)],
        output_dtypes=[mx.float32],
        grid=(256 * groups, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return neg.reshape(1, 1, capacity)
