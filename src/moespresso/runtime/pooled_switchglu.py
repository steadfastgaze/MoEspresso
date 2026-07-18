"""SwitchGLU over persistent SSD-backed expert pools.

The module owns the whole SwitchGLU seam (sort/gather, gate/up activation, down,
scatter) so it cannot be bypassed by JANG's class-level SwitchGLU monkeypatch.
Each projection reads selected experts from a persistent `ExpertSlotPool`; misses
are loaded directly into MLX buffers via `pread_into`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import threading
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from moespresso.runtime.expert_index import ExpertIndex
from moespresso.runtime.expert_slot_pool import ExpertCapacityExceeded
from moespresso.runtime.expert_slot_pool import ExpertSlotPool
from moespresso.package.bundle import KQUANT_CODEC, MXFP4_CODEC, TQ_CODEC

_PROJECTION_LOAD_EXECUTOR = ThreadPoolExecutor(
    max_workers=3,
    thread_name_prefix="moespresso-ssd-proj",
)

# Single ORDERED worker for the pipelined decode: FIFO == layer order, so
# kicks commit in layer order and the eviction fence semantics are preserved.
_PIPELINE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="moespresso-ssd-pipe",
)

# Speculative prefetch runs off the ordered pipeline worker so a
# slow speculative pread can never delay a demand ensure. Lazy: most
# sessions never enable lookahead.
_LOOKAHEAD_EXECUTOR_BOX: list = [None]

# Load shedding for the speculative executor: predictions arrive once per
# routed layer per decode step, but a placement task preads up to sixteen
# expert rows, which on large-expert models takes longer than a layer
# step. An unbounded queue makes every placement land steps late (stale
# speculation) and leaves a backlog the process must drain at exit, so a
# submission is dropped instead of queued whenever the executor already
# holds this many in-flight tasks (one running plus one queued keeps the
# worker fed without building a backlog). Dropped predictions cost
# nothing: the next layer step submits a fresh, current one.
_LOOKAHEAD_MAX_PENDING = 2
_LOOKAHEAD_PENDING_LOCK = threading.Lock()
_LOOKAHEAD_PENDING = [0]


def _lookahead_executor() -> ThreadPoolExecutor:
    if _LOOKAHEAD_EXECUTOR_BOX[0] is None:
        _LOOKAHEAD_EXECUTOR_BOX[0] = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="moespresso-ssd-lookahead")
    return _LOOKAHEAD_EXECUTOR_BOX[0]

# On-device remap: after ensure() loads any missing experts, remap routed ids ->
# pool slots via an on-device gather instead of rebuilding slot indices on the
# host for each projection. This is an exact seam cleanup; e2e speed impact has
# measured roughly neutral so far. Default ON; set MOESPRESSO_SSD_ONDEVICE_REMAP=0
# to fall back to the proven host remap_loaded.
_ONDEVICE_REMAP = os.environ.get("MOESPRESSO_SSD_ONDEVICE_REMAP", "1") != "0"

# Fused gate+up: jang's fused_gate_up_swiglu_matmul computes SiLU(gate)*up in one
# Metal dispatch (vs the pooled path's 2 separate gather kernels + a Python
# `activation(x_up, x_gate)`). Restores a fast path the pre-pooled streaming seam
# had and the pooled rewrite dropped. This restores the previous behavior.
# Default ON; MOESPRESSO_SSD_FUSED_GATE_UP=0 falls back to the exact separate path.
# Precondition (guarded in PooledSwitchGLU.__init__): gate and up share codebook,
# signs and bits, which is true for real mjtq packages (same in_features/bits/seed).
_FUSED_GATE_UP = os.environ.get("MOESPRESSO_SSD_FUSED_GATE_UP", "1") != "0"

# Compiled-island decode path: for decode tokens, run the
# whole routed MLP (slot remap x2 + rotate + fused gate/up/SwiGLU + rotate +
# down gather) as one mx.compile'd closure, following jang's own decode patch
# shape (jangrt/switchglu_decode.py). Slot-bank mutation (ensure/pread, slot
# table rebuilds) stays outside the island; only stable MLX arrays enter.
# Default ON; MOESPRESSO_SSD_COMPILED_ISLAND=0 falls back to the eager path.
_COMPILED_ISLAND = os.environ.get("MOESPRESSO_SSD_COMPILED_ISLAND", "1") != "0"

# Cross-chunk predictive expert prefetch for streamed prefill. On the
# over-capacity sorted-chunked path a layer's demand set is nearly stationary
# across prompt chunks, and the whole prompt chunk of GPU work between a
# layer's consecutive MoE calls is an unused overlap window. After a layer's
# over-capacity call for prompt chunk N finishes, this submits a background
# best-effort prefetch of the experts that call used, keyed on the block. The
# next call for the same layer (prompt chunk N+1) awaits the prefetch before
# its own chunk-ahead path runs, so slots that would miss are already warm.
# Per-layer pools make the prefetch the only pool mutator between the two
# calls, and prefetch protects the last demand set, so the still-executing
# final chunk's slots are never evicted. The prefetch pre-fills slots and never
# substitutes for the per-call sync and ensure; a mismatched or stale ticket is
# counted and discarded, and the normal path services any difference.
# Served A/B at cap-192: 37K prefill 516.7 to 560.8 t/s (+8.5%), 4K prefill
# 620.2 to 677.4 t/s (+9.2%), miss volume down about 30 percent, token-identical
# across capacities and 9/9 on the quality gate. The full-capacity certificate
# path never dispatches over capacity, so it never submits or consumes a ticket
# and stays untouched (826 t/s at 37K, unchanged). A consumption-order-aligned
# variant that truncated the prediction to the lowest-id non-resident slice
# sized to the free-plus-evictable budget measured slower (525.6 vs 560.8 t/s:
# it warms only the non-resident actives, 123k vs 299k experts per run), so the
# whole-set submission ships. Default ON; MOESPRESSO_SSD_PREFETCH=0 is the
# kill switch.
_PREFILL_PREFETCH = os.environ.get("MOESPRESSO_SSD_PREFETCH", "1") != "0"

# Spec-prefetch oracle study: opt-in route tracing. When started, every
# layer's routed expert ids are recorded (decode from the ring worker, the
# ids are already in hand there, zero extra sync) and prefill/legacy paths
# from __call__ (one extra host sync per layer call, acceptable for a study
# run, never enabled in product). Entries:
#   ("decode", seq, layer, [ids])        ring/gate worker, seq orders steps
#   ("prefill", layer, [[ids]*T])        position-intact, T>1
#   ("decode_direct", layer, [[ids]])    legacy non-ring decode
_ROUTE_TRACE: list | None = None
# additionally capture decode router-input hidden states (("hidden", layer,
# float16 ndarray) entries) for study runs; large, so a separate switch
_ROUTE_TRACE_HIDDEN = os.environ.get("MOESPRESSO_SSD_ROUTE_TRACE_HIDDEN", "0") == "1"


def _should_sort_routed_indices(indices) -> bool:
    return bool(indices.size >= 64)


# Bulk sorted prefill: on hardware without the NAX gather_qmm_rhs kernel,
# gather_qmm lowers to one vector-matmul per token-expert pair, so every
# expert's quantized weights are re-read once per assigned token (~100 GB of
# redundant weight traffic per routed layer at the 3844-token prompt shape).
# Splitting the sorted rows into contiguous per-expert segments and running one
# f32 dequantize + f32 GEMM per active expert reads each expert's weights once:
# one layer's gate/up+down measured 842 ms -> 308 ms at 23058 sorted pairs.
# The f32 pair matters: the tiled K-quant qmm and an f16 dequant+GEMM both hold
# dequantized weights in f16 and measured 1.6x the gather path's mean abs error
# against an f32 reference, which moved the Q1 gate from 16/17 to 12/17. The
# f32 dequant + f32 GEMM error profile is identical to the gather path's, so
# the served numerics keep the quality anchor. Below this row count the gather
# path stays; the per-pair vector kernel wins at decode scale.
_SEGMENTED_PREFILL_MIN_ROWS = 4096

# Barrier-free full-resident bulk prefill: when every projection pool holds its
# whole expert set (capacity == num_experts, the prewarm-all serving
# configuration), the per-layer blocking np.asarray(indices) host read exists
# only to feed miss handling and the host-built sort/segment table, and neither
# is needed: there are no misses to load, routing can stay on device (argsort
# plus a slot-table gather), and mlx_kquant.gather_qmm_sorted derives each
# expert's row range in-kernel from the sorted slot ids, so the entire prefill
# queues as one lazy graph with no per-layer drain.
#
# Measured on the 3844-token bounded served checkpoint, alternating arms in
# one session (all thermal Nominal, GPU 43-54C): route on 25.107 / 25.306 /
# 25.068 s TTFT, route off 26.810 s taken between the second and third on-arm,
# with the off arm running cooler than the last on arm. Engagement counters
# prove the arms differ: on-arm index_sync_calls and index_resync_calls both 0
# (off-arm 43 each), barrier_free_prefill_calls 43, routed gate/down 43 calls
# each, bundle_row_preads 0. Net ~1.5-1.7 s TTFT at matched warmth. Full
# 64-token A/B/A with the route on: TTFT 25.009 / 25.478 s (certified
# pre-change anchors 26.048 / 26.251 s), decode 15.25 / 14.38 tok/s, both arms
# 46 tokens with stop and token-identical to the certified anchor
# continuation. Gates with the route engaged: Q1 16/17 (blocking 2, the known
# anchor), Q2 avg_nll 0.3914699462823993 bit-identical, Q3 16/16. This
# constant is the route's kill switch; the eligibility check fails closed to
# the segmented path when it is False or when any precondition (combined
# K-quant gate/up, K-quant down, full residency, kernel availability) does not
# hold.
_BARRIER_FREE_PREFILL = True

# Fused sorted SwiGLU for the barrier-free identity route: when mlx_kquant
# ships gather_qmm_sorted_swiglu, the combined gate/up GEMM and the SwiGLU
# activation collapse into one kernel whose epilogue applies the activation
# on the float32 accumulators, so the [rows, 2N] intermediate and the
# elementwise activation pass disappear. Same formula as the activation
# module (gate upper-clamped to swiglu_limit, up clamped symmetrically,
# silu(gate) * up in float32), but the epilogue skips the intermediate
# rounding to the row dtype, so fused-on/off is numerically equivalent
# rather than bit-identical for f16 rows. Identity-slot route only: the
# route already shares one sorted id array between both GEMMs there.
# Default ON; MOESPRESSO_SSD_FUSED_SORTED_SWIGLU=0 is the kill switch back
# to the unfused gather_qmm_sorted + activation pair.
_FUSED_SORTED_SWIGLU = (
    os.environ.get("MOESPRESSO_SSD_FUSED_SORTED_SWIGLU", "1") != "0"
)

# Unified sorted prefill: run the partial-residency sorted-chunked prefill
# through the same fused sorted K-quant kernels the full-resident barrier-free
# route uses (gather_qmm_sorted_swiglu for gate/up + SwiGLU, gather_qmm_sorted
# for down), over slot ids, per capacity-chunk. Without this the chunked
# prefill computes each chunk through the unfused segmented f32 GEMM (or the
# general gather) plus a separate activation, whose reconstruction and SwiGLU
# epilogue differ from the fused kernel, so partial residency forks the served
# tokens from the full-residency rail. Prefill routed MoE carries no cross-row
# reduction (each output row is one token against one expert), so splitting an
# expert's rows across capacity-chunks and running the same fused sorted kernel
# per chunk yields the identical per-row output as one un-split segment. This
# is the residency-decides-where-not-how contract: the misses are filled into
# slots first, then the compute is the same kernel the full path runs.
#
# Engages only when the combined K-quant gate/up pool, the K-quant down pool,
# and the fused sorted SwiGLU kernel are all present; anything else falls back
# to the segmented/general chunked compute. Default ON;
# MOESPRESSO_SSD_UNIFIED_PREFILL=0 restores the pre-unification chunked compute
# (the kill switch, and the pre-unification OFF arm for the price A/B).
_UNIFIED_SORTED_PREFILL = (
    os.environ.get("MOESPRESSO_SSD_UNIFIED_PREFILL", "1") != "0"
)

# Barrier-free full-resident decode: the decode analog of the barrier-free
# prefill route. When the one-shot residency certificate holds (every
# projection pool has capacity == num_experts and a fully populated slot
# table, read under the pool bookkeeping locks), the DS4 MoE block skips the
# ring export kernel, the event gate, the worker submit, and the per-layer
# kick, and consumes router indices on device (identity slot tables on the
# prewarm-all fill order, else one on-device slot-table gather per pool).
# The routed math is the same combined gate/up gather, activation, and down
# gather the pipelined builder emits, so the route is bit-identical to the
# ring path by construction; only the scheduling changes. The token graph
# queues lazily and _DECODE_FLUSH_LAYERS controls the intermediate commits.
# Default ON; MOESPRESSO_SSD_BARRIER_FREE_DECODE=0 is the kill switch back
# to the ring/native-gate decode, which also remains the product path for
# any partial-residency session (the certificate fails closed).
_BARRIER_FREE_DECODE = (
    os.environ.get("MOESPRESSO_SSD_BARRIER_FREE_DECODE", "1") != "0"
)

# Qwen-style decode scheduling: when the full-residency certificate holds, the
# Qwen sparse MoE block takes the barrier-free full-resident decode route that
# the DS4 block already runs, instead of the ring-export + native-gate pipeline.
# The pipeline exists to overlap expert-miss service with compute; at full
# residency there are no misses to hide, so its per-layer block-exit kick emits
# forty async_eval graph flushes per token where the resident runtime builds one
# lazy graph. The barrier-free route queues the token graph lazily and commits
# every _DECODE_FLUSH_LAYERS layers, with no ring export, no event gate, and no
# worker submit. The routed math is the same combined gate/up gather, activation,
# and down gather build_pipelined's separate-kernel branch emits, so the route is
# bit-identical to the ring path (only the index source and the scheduling
# differ). Gated by the shared `_barrier_free_decode_ready` certificate, which
# fails closed to the ring path for any partial-residency session. Default ON;
# MOESPRESSO_SSD_DECODE_SCHED=0 restores the current pipelined Qwen decode
# scheduling exactly, without touching the DS4 route.
_QWEN_DECODE_SCHED = (
    os.environ.get("MOESPRESSO_SSD_DECODE_SCHED", "1") != "0"
)

# Ornith's routed Q6_K down projections have the dedicated gathered-QMV
# geometry: one token, eight expert rows, 512 inputs, and 2,048 outputs. The
# dedicated leaf preserves the generic gather's BF16 result while avoiding its
# shape-general wrapper. Set MOESPRESSO_QWEN_DOWN_Q6_QMV=0 to restore the
# generic gather.
_QWEN_DOWN_Q6_QMV = (
    os.environ.get("MOESPRESSO_QWEN_DOWN_Q6_QMV", "1") == "1"
)

# Flush depth for the barrier-free decode route: commit the queued token
# graph after every N MoE layers; the generator's own async_eval commits the
# tail. Depth 4 mirrors the DS4-c split-after-an-early-layer shape; depth 1
# approximates the per-layer commit cadence without the ring machinery. The
# default stays 4. The landing sweep measured depth 1 tying inside noise
# (17.99 versus 17.97-17.99 tok/s). After the fused-kernel levers, an
# in-process ledger probe favored depth 1 by 0.8 ms/token (43.25 versus
# 44.05 median), but the served alternating A/B did not reproduce the gap:
# depth-1 arms at 22.997 and 23.020 tok/s interleaved with depth-4 arms at
# 23.016 and 22.966, token-identical throughout, so the ledger delta does
# not survive the serve path. The knob gates decode commits only. Values
# below 1 disable the intermediate flushes entirely.
_DECODE_FLUSH_LAYERS = int(
    os.environ.get("MOESPRESSO_DSV4_DECODE_FLUSH_LAYERS", "4")
)

# Fused decode routed matvec family (the DS4-c decode MoE contract): on the
# barrier-free decode route, the routed block collapses to two dispatches.
# gather_qmv_pair_swiglu runs one matvec per routed expert over the combined
# gate/up pool with the SwiGLU applied to the float32 accumulators and the
# route weight baked into the stored intermediate; gather_qmv_expert_sum runs
# the down matvec with the sum over the token's routed experts inside the
# kernel, which removes the separate route-weighted-sum reduction outright.
# Math-affecting by construction: route weights multiply before the down
# matvec instead of after, the cross-expert sum accumulates per output
# element in float32, and the intermediate skips the bfloat16 round-trip the
# unfused composition takes, so fused-on/off is numerically equivalent rather
# than bit-identical and the change is judged by the full quality campaign.
# Engages only when the barrier-free decode certificate holds, both slot
# tables are the identity map, and the pool codecs match the instantiated
# kernels (iq2_xxs combined gate/up, q2_k down); anything else falls back to
# the unfused barrier-free route. Against an f64 reference of the routed
# block on real served states the fused form reads rel ~2.5e-7 versus the
# composition's ~4.5e-3 (the bf16 intermediate lattice), and the served
# 64-token anchor A/B measured decode 20.33-20.34 to 21.92-21.93 tok/s with
# token identity on the anchor rail. Default ON;
# MOESPRESSO_DSV4_DECODE_ROUTED_FUSED=0 is the kill switch back to the
# unfused barrier-free route.
_DECODE_ROUTED_FUSED = (
    os.environ.get("MOESPRESSO_DSV4_DECODE_ROUTED_FUSED", "1") != "0"
)

# Ring-path fused decode routed matvec (the bounded-residency decode
# unification): the fused pair above does not require full residency, only
# correct row indices into the pool stacks, and the ring worker already
# publishes per-layer slot-id buffers in router order after ensure(). At
# partial residency the DS4 block therefore runs the same two kernels over
# the published slot ids, with the route weights in the same router order,
# so per-token math is identical to the full-resident fused route by
# construction: same kernels, same entry order (gather_qmv_expert_sum
# accumulates in id-array order, which the worker preserves), same float32
# accumulation, and the indexed rows hold the same bytes. Residency decides
# which pool row holds an expert; it never touches the dispatch or the
# math. This closes the residency-keyed decode route split that left the
# streamed tier off the full-resident rail (bounded arms forked at decode
# knife-edges). Default ON; MOESPRESSO_DSV4_DECODE_RING_FUSED=0 restores
# the unfused ring composition at partial residency only, and the family
# switch MOESPRESSO_DSV4_DECODE_ROUTED_FUSED=0 keeps killing the fused
# kernels everywhere.
_DECODE_RING_FUSED = (
    os.environ.get("MOESPRESSO_DSV4_DECODE_RING_FUSED", "1") != "0"
)


def _sorted_expert_segments(idx_host) -> list:
    """(expert_id, start, end) runs over flat, already-sorted expert ids."""
    flat = np.asarray(idx_host).reshape(-1)
    if flat.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [flat.size]))
    return [(int(flat[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _segmented_kquant_matmul(proj, x, idx_host):
    """One f32 dequantize + f32 GEMM per sorted-expert segment of `x` rows.

    `idx_host` holds the flat, sorted expert ids for the rows of `x`. Each
    active expert's weights are read once for its whole row segment, instead
    of once per row as the gather kernel does without the NAX sorted-rhs path.
    Dequantization and the GEMM run in f32 so the error profile matches the
    gather kernel's f32 accumulation; the output returns to the input dtype.
    Callers guarantee the segment experts are resident (the sorted-chunked
    prefill ensures its chunk's active set before building)."""
    import mlx_kquant as kq

    proj.matmul_slot_calls += 1
    proj.segmented_matmul_calls += 1
    rows = x.reshape(-1, proj.in_features)
    outs = []
    for expert, start, end in _sorted_expert_segments(idx_host):
        slot = proj.pool._slot_of[expert]
        wf = kq.dequantize(
            proj.pool.weight[slot],
            proj.pool.scales[slot],
            proj.kquant_type,
            dtype=mx.float32,
        )
        seg = rows[start:end].astype(mx.float32) @ wf.T
        outs.append(seg.astype(x.dtype))
    out = mx.concatenate(outs, axis=0)
    return out.reshape(*x.shape[:-1], out.shape[-1])


def route_trace_start() -> None:
    global _ROUTE_TRACE
    _ROUTE_TRACE = []


def route_trace_stop() -> list:
    global _ROUTE_TRACE
    out, _ROUTE_TRACE = _ROUTE_TRACE or [], None
    return out

# v3 ring-export decode: like v2 commits stay on main, but the worker's only
# MLX call
# (the wait-only np.asarray that appeared to serialize against main's evals)
# is replaced by a GPU-side export: a tiny kernel writes the router indices +
# a sequence number into a persistent per-layer ring buffer (relaxed
# device-scope atomics, Metal has nothing stronger, with an FNV checksum
# guarding torn/stale reads, see _ring_checksum),
# and the worker seqlock-polls raw memory (zero MLX), probe-measured ~0.5 ms
# end-to-end. Main never blocks in MLX per layer: it waits a threading.Event
# (set by the worker after ensure+publish) before the async_eval that
# transitively commits the previous layer's routed graph.
# Measured: 9.13/9.50 tok/s (32/64 tok) vs
# 5.40-5.68 for the legacy path, 11.28 with 0.5 GiB adaptive growth.
# Identical greedy output, same MLX peak. Default ON;
# MOESPRESSO_SSD_RING_DECODE=0 restores the legacy ticket/kick path. The
# GPU->host mid-buffer visibility this relies on is probe-validated
# (doorbell_probe.py) but not Metal-spec-guaranteed: the ring watchdog
# raises a loud TimeoutError rather than ever serving stale routing.
_RING_DECODE = os.environ.get("MOESPRESSO_SSD_RING_DECODE", "1") != "0"

# Worker poll timeout for the ring seq (seconds). Generous: covers cold-cache
# kernel JIT of the export on first use.
_RING_TIMEOUT = float(os.environ.get("MOESPRESSO_SSD_RING_TIMEOUT", "10.0"))

# v4 gate-decode: when the optional native gate
# extension is built and passes its self-test, each decode layer's routed
# island sits behind an MTLSharedEvent wait encoded in-stream; main commits
# the whole layer immediately (no per-layer join), and the IO worker signals
# the event after ensure+publish. Kernels wait for IO; threads never wait
# for kernels. MOESPRESSO_SSD_GATE_DECODE=0 disables (ring v3 fallback);
# the gate always gets signaled, even on worker error (poison), so a stuck
# GPU wait cannot outlive a token. Errors surface at the once-per-token
# future drain.
_GATE_MOD: list = [None]  # None = unresolved, False = unavailable, module


def _gate_module():
    if _GATE_MOD[0] is None:
        from moespresso.runtime.native_gate import load_gate
        mod = load_gate()
        if mod is not None:
            # The event value domain is shared with the loader's self-test
            # (and anything else that signaled this process's event). Seq
            # values must start above the current signaled value, or a
            # layer's gate is already open before its worker publishes:
            # the island would read pre-publish slot buffers (cold-step
            # corruption, caught by the two-layer equivalence test).
            _RING_SEQ[0] = max(_RING_SEQ[0], int(mod.signaled_value()))
        _GATE_MOD[0] = mod or False
    return _GATE_MOD[0] or None

_EXPORT_SOURCE = """
    // inputs: inds (uint, K), ring (uint, 8+K), target (uint, 1)
    // output: token (uint, 1)
    // ring layout: [0]=seq, [1]=checksum, [2..7]=reserved, [8..8+K-1]=ids.
    //
    // ORDERING: MSL device-scope atomics only offer
    // memory_order_relaxed: there is no release order to use here. The
    // GPU-side ordering mechanism is the threadgroup_barrier(mem_device)
    // between the id writes and the seq store. Because the host CPU may
    // also reorder its loads, the protocol does not rely on ordering alone:
    // the host verifies seq == expected and checksum(ids) and re-reads seq
    // (seqlock). checksum is written by thread 0 from threadgroup staging,
    // so a stale-id read cannot pass all three checks.
    uint i = thread_position_in_grid.x;
    uint K = inds_shape[0];
    device uint* ring_w = (device uint*)(ring);
    threadgroup uint stage[64];
    if (i < K) {
        stage[i] = inds[i];
        ring_w[8u + i] = inds[i];
    }
    // barriers are uniform (outside any divergent branch, Metal requires it)
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (i == 0) {
        uint sum = 2166136261u;  // FNV-ish mix over ids + seq
        for (uint k = 0; k < K; k++) {
            sum = (sum ^ stage[k]) * 16777619u;
        }
        sum = (sum ^ target[0]) * 16777619u;
        ring_w[1] = sum;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (i == 0) {
        device atomic_uint* seq = (device atomic_uint*)(ring_w);
        atomic_store_explicit(seq, target[0], memory_order_relaxed);
        token[0] = 1u;
    }
"""


def _ring_checksum(ids, seq: int) -> int:
    total = 2166136261
    for v in ids:
        total = ((total ^ int(v)) * 16777619) & 0xFFFFFFFF
    return ((total ^ seq) * 16777619) & 0xFFFFFFFF

_EXPORT_KERNEL = None


def _get_export_kernel():
    global _EXPORT_KERNEL
    if _EXPORT_KERNEL is None:
        _EXPORT_KERNEL = mx.fast.metal_kernel(
            name="moespresso_ring_export",
            input_names=["inds", "ring", "target"],
            output_names=["token"],
            source=_EXPORT_SOURCE,
        )
    return _EXPORT_KERNEL


# Ring self-test: the GPU->host mid-buffer visibility the ring relies on is
# validated by a runtime probe because the Metal specification does not guarantee it.
# Verify it once per process at
# first use; if the export does not become host-visible promptly, force the
# legacy path for the whole process instead of relying on per-layer timeouts.
# None = untested, True = ring usable, False = fall back to legacy.
_RING_SELF_TEST: list = [None]
_RING_SELF_TEST_BUDGET_S = 0.25


def _ring_visibility_ok() -> bool:
    if _RING_SELF_TEST[0] is not None:
        return _RING_SELF_TEST[0]
    try:
        K = 8
        inds = mx.array(np.arange(1, K + 1, dtype=np.uint32))
        ring = mx.array(np.zeros(8 + K, dtype=np.uint32))
        target = mx.array(np.array([12345], dtype=np.uint32))
        mx.eval(inds, ring, target)
        ring_np = np.frombuffer(memoryview(ring).cast("B"), dtype=np.uint32)
        token, = _get_export_kernel()(
            inputs=[inds, ring, target],
            output_shapes=[(1,)],
            output_dtypes=[mx.uint32],
            grid=(K, 1, 1),
            threadgroup=(K, 1, 1),
        )
        mx.async_eval(token)
        deadline = time.perf_counter() + _RING_SELF_TEST_BUDGET_S
        ok = False
        while time.perf_counter() < deadline:
            if int(ring_np[0]) == 12345:
                ok = bool(
                    np.array_equal(ring_np[8:8 + K],
                                   np.arange(1, K + 1, dtype=np.uint32)))
                break
        mx.eval(token)
    except Exception:
        ok = False
    _RING_SELF_TEST[0] = ok
    if not ok:
        import warnings
        warnings.warn(
            "moespresso: ring-export visibility self-test FAILED for this "
            "MLX build; decode falls back to the legacy ticket path "
            "(slower but proven). Set MOESPRESSO_SSD_RING_DECODE=0 to silence.",
            RuntimeWarning,
            stacklevel=2,
        )
    return ok


@dataclass
class _ProjectionLoadTicket:
    active: set[int]
    futures: list
    started_at: float
    used: bool = False

    @property
    def has_work(self) -> bool:
        return bool(self.futures)


@dataclass
class _PrefetchTicket:
    """Cross-chunk predictive prefetch handle stored on a layer's switch.

    `predicted` is the demand set of the prompt chunk that submitted it, warmed
    on the IO executor so the layer's next call finds those slots resident.
    `futures` complete when every pool's prefetch has published. The consumer
    awaits them before its chunk-ahead path touches the pools, then discards the
    ticket. A submitted set that does not match the consumer's actual demand is
    counted as a mismatch but still awaited, because the prefetch's bytes are
    already landing into slots the pool now owns; abandoning mid-flight would
    leave reserved-but-unpublished slots.
    """

    predicted: frozenset[int]
    futures: list
    submitted_at: float


def _token_layers(x) -> int:
    n = 1
    for dim in x.shape[:-1]:
        n *= int(dim)
    return n


def _kick_eval(x) -> None:
    async_eval = getattr(mx, "async_eval", None)
    if async_eval is not None:
        async_eval(x)
    else:  # pragma: no cover - old MLX fallback
        mx.eval(x)


def _record_routed_weighted_sum(switch, scores, *, out_features: int) -> None:
    """Record the unfused route-score reduction after routed down projection."""
    if not hasattr(switch, "routed_weighted_sum_calls"):
        return
    switch.routed_weighted_sum_calls += 1
    switch.routed_weighted_sum_slot_elements += int(np.prod(scores.shape))
    token_layers = 1
    for dim in scores.shape[:-1]:
        token_layers *= int(dim)
    switch.routed_weighted_sum_output_elements += (
        token_layers * int(out_features))


def _record_switch_seconds(switch, attr: str, seconds: float) -> None:
    """Record optional timing counters on real switches and replay fakes."""
    setattr(switch, attr, float(getattr(switch, attr, 0.0)) + float(seconds))


def _deepseek_v4_weighted_sum(y, scores):
    return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)


class PooledTurboQuantSwitchLinear(nn.Module):
    """A routed TQ projection backed by an `ExpertSlotPool`."""

    def __init__(
        self,
        *,
        package_dir,
        index: ExpertIndex,
        layer: int,
        projection: str,
        capacity: int,
        codebook,
        signs,
        eviction_policy: str = "lfu",
        row_cache=None,
        spare_slots: int = 0,
    ):
        super().__init__()
        self.pool = ExpertSlotPool(
            package_dir=package_dir,
            index=index,
            layer=layer,
            projection=projection,
            capacity=capacity,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=spare_slots,
        )
        if self.pool.codec != TQ_CODEC:
            raise ValueError(
                f"{projection} declares codec {self.pool.codec!r}, expected 'tq'")
        self.codec = TQ_CODEC
        self.bits = self.pool.bits
        self.num_experts = self.pool.num_experts
        self.out_features = self.pool.geometry.out_features
        self.in_features = self.pool.geometry.packed_cols * (32 // self.bits)
        self.codebook = codebook
        self.signs = signs
        self.matmul_slot_calls = 0
        self.matmul_slot_elements = 0

    def __call__(self, x, indices, *, sorted_indices: bool = False):
        remapped = self.pool.remap(indices)
        return self.matmul_slots(x, remapped, sorted_indices=sorted_indices)

    def matmul_slots(self, x, remapped_indices, *, sorted_indices: bool = False):
        from jang_tools.turboquant.gather_tq_kernel import gather_tq_matmul

        self.matmul_slot_calls += 1
        self.matmul_slot_elements += int(np.prod(remapped_indices.shape))
        return gather_tq_matmul(
            x,
            self.pool.packed,
            self.pool.norms,
            self.codebook,
            self.signs,
            remapped_indices,
            bits=self.bits,
            sorted_indices=sorted_indices,
        )


class PooledMxfp4SwitchLinear(nn.Module):
    """A routed source-mxfp4 projection backed by an `ExpertSlotPool`."""

    def __init__(
        self,
        *,
        package_dir,
        index: ExpertIndex,
        layer: int,
        projection: str,
        capacity: int,
        eviction_policy: str = "lfu",
        row_cache=None,
        spare_slots: int = 0,
    ):
        super().__init__()
        self.pool = ExpertSlotPool(
            package_dir=package_dir,
            index=index,
            layer=layer,
            projection=projection,
            capacity=capacity,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=spare_slots,
        )
        if self.pool.codec != MXFP4_CODEC:
            raise ValueError(
                f"{projection} declares codec {self.pool.codec!r}, expected 'mxfp4'")
        self.codec = MXFP4_CODEC
        self.bits = self.pool.bits
        self.num_experts = self.pool.num_experts
        self.out_features = self.pool.geometry.out_features
        self.in_features = self.pool.geometry.packed_cols * (32 // self.bits)
        self.matmul_slot_calls = 0
        self.matmul_slot_elements = 0

    def __call__(self, x, indices, *, sorted_indices: bool = False):
        remapped = self.pool.remap(indices)
        return self.matmul_slots(x, remapped, sorted_indices=sorted_indices)

    def matmul_slots(self, x, remapped_indices, *, sorted_indices: bool = False):
        self.matmul_slot_calls += 1
        self.matmul_slot_elements += int(np.prod(remapped_indices.shape))
        return mx.gather_qmm(
            x,
            self.pool.packed,
            self.pool.scales,
            None,
            rhs_indices=remapped_indices,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
            sorted_indices=sorted_indices,
        )


class PooledKQuantSwitchLinear(nn.Module):
    """A routed K-quant projection backed by an `ExpertSlotPool`."""

    def __init__(
        self,
        *,
        package_dir,
        index: ExpertIndex,
        layer: int,
        projection: str,
        capacity: int,
        eviction_policy: str = "lfu",
        row_cache=None,
        spare_slots: int = 0,
    ):
        super().__init__()
        self.pool = ExpertSlotPool(
            package_dir=package_dir,
            index=index,
            layer=layer,
            projection=projection,
            capacity=capacity,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=spare_slots,
        )
        if self.pool.codec != KQUANT_CODEC:
            raise ValueError(
                f"{projection} declares codec {self.pool.codec!r}, expected 'kquant'")
        self.codec = KQUANT_CODEC
        self.bits = self.pool.bits
        self.kquant_type = self.pool.geometry.kquant_codec
        self.num_experts = self.pool.num_experts
        self.out_features = self.pool.geometry.out_features
        bytes_per_block = int(self.pool.geometry.bytes_per_block or 0)
        weights_per_block = int(self.pool.geometry.weights_per_block or 0)
        if bytes_per_block <= 0 or weights_per_block <= 0:
            raise ValueError(f"{projection}: missing K-quant geometry")
        if self.pool.geometry.packed_cols % bytes_per_block:
            raise ValueError(
                f"{projection}: K-quant bytes_per_row {self.pool.geometry.packed_cols} "
                f"is not divisible by {bytes_per_block}")
        self.in_features = (
            self.pool.geometry.packed_cols // bytes_per_block * weights_per_block
        )
        self.matmul_slot_calls = 0
        self.matmul_slot_elements = 0
        self.segmented_matmul_calls = 0
        self.decode_q6_qmv_calls = 0

    def __call__(self, x, indices, *, sorted_indices: bool = False):
        remapped = self.pool.remap(indices)
        return self.matmul_slots(x, remapped, sorted_indices=sorted_indices)

    def matmul_slots(self, x, remapped_indices, *, sorted_indices: bool = False):
        import mlx_kquant as kq

        self.matmul_slot_calls += 1
        self.matmul_slot_elements += int(np.prod(remapped_indices.shape))
        if (
            _QWEN_DOWN_Q6_QMV
            and not sorted_indices
            and self.pool.projection == "down_proj"
            and self.kquant_type == "q6_k"
            and self.in_features == 512
            and self.out_features == 2048
            and x.dtype == mx.bfloat16
            and x.ndim == remapped_indices.ndim + 2
            and tuple(x.shape[:-2]) == tuple(remapped_indices.shape)
            and int(x.shape[-2]) == 1
            and int(remapped_indices.shape[-1]) == 8
            and getattr(kq, "gather_qmv_kq", None) is not None
        ):
            routes = int(remapped_indices.shape[-1])
            self.decode_q6_qmv_calls += 1
            out = kq.gather_qmv_kq(
                x.reshape(-1, routes, self.in_features),
                self.pool.weight,
                self.kquant_type,
                remapped_indices.reshape(-1, routes),
            )
            return out.reshape(
                *remapped_indices.shape,
                1,
                self.out_features,
            )
        return kq.gather_qmm(
            x,
            self.pool.weight,
            self.pool.scales,
            self.kquant_type,
            rhs_indices=remapped_indices,
            transpose=True,
            sorted_indices=sorted_indices,
        )

    def matmul_slots_segmented(self, x, idx_host):
        return _segmented_kquant_matmul(self, x, idx_host)


class _PooledCombinedGateUpKQuantAlias:
    """Compatibility alias for the up half of a combined gate/up K-quant pool."""

    def __init__(self, parent: "PooledCombinedGateUpKQuantLinear"):
        self._parent = parent
        self.pool = parent.pool
        self.codec = parent.codec
        self.bits = parent.bits
        self.kquant_type = parent.kquant_type
        self.num_experts = parent.num_experts
        self.in_features = parent.in_features
        self.out_features = parent.up_out_features
        self.matmul_slot_calls = 0
        self.matmul_slot_elements = 0

    def __call__(self, *_args, **_kwargs):
        raise RuntimeError(
            "combined K-quant up projection is only callable through "
            "PooledSwitchGLU")

    def matmul_slots(self, *_args, **_kwargs):
        raise RuntimeError(
            "combined K-quant up projection is only callable through "
            "PooledSwitchGLU")


class PooledCombinedGateUpKQuantLinear(nn.Module):
    """One resident K-quant pool for gate+up routed projections."""

    def __init__(
        self,
        *,
        package_dir,
        index: ExpertIndex,
        layer: int,
        capacity: int,
        eviction_policy: str = "lfu",
        row_cache=None,
        spare_slots: int = 0,
    ):
        super().__init__()
        gate_geo = index.geometry(layer=layer, projection="gate_proj")
        up_geo = index.geometry(layer=layer, projection="up_proj")
        self.pool = ExpertSlotPool(
            package_dir=package_dir,
            index=index,
            layer=layer,
            projection="gate_proj",
            capacity=capacity,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=spare_slots,
            combined_kquant_projection="up_proj",
        )
        if self.pool.codec != KQUANT_CODEC:
            raise ValueError(
                f"gate/up declares codec {self.pool.codec!r}, expected 'kquant'")
        self.codec = KQUANT_CODEC
        self.bits = self.pool.bits
        self.kquant_type = self.pool.geometry.kquant_codec
        self.num_experts = self.pool.num_experts
        self.gate_out_features = int(gate_geo.out_features)
        self.up_out_features = int(up_geo.out_features)
        self.out_features = self.gate_out_features
        bytes_per_block = int(self.pool.geometry.bytes_per_block or 0)
        weights_per_block = int(self.pool.geometry.weights_per_block or 0)
        if bytes_per_block <= 0 or weights_per_block <= 0:
            raise ValueError("combined gate/up: missing K-quant geometry")
        if self.pool.geometry.packed_cols % bytes_per_block:
            raise ValueError(
                "combined gate/up: K-quant bytes_per_row "
                f"{self.pool.geometry.packed_cols} is not divisible by "
                f"{bytes_per_block}")
        self.in_features = (
            self.pool.geometry.packed_cols // bytes_per_block * weights_per_block
        )
        self.matmul_slot_calls = 0
        self.matmul_slot_elements = 0
        self.segmented_matmul_calls = 0
        self.up_alias = _PooledCombinedGateUpKQuantAlias(self)

    def __call__(self, x, indices, *, sorted_indices: bool = False):
        remapped = self.pool.remap(indices)
        return self.matmul_slots(x, remapped, sorted_indices=sorted_indices)

    def matmul_slots(self, x, remapped_indices, *, sorted_indices: bool = False):
        import mlx_kquant as kq

        self.matmul_slot_calls += 1
        self.matmul_slot_elements += int(np.prod(remapped_indices.shape))
        return kq.gather_qmm(
            x,
            self.pool.weight,
            self.pool.scales,
            self.kquant_type,
            rhs_indices=remapped_indices,
            transpose=True,
            sorted_indices=sorted_indices,
        )

    def matmul_slots_segmented(self, x, idx_host):
        return _segmented_kquant_matmul(self, x, idx_host)

    def matmul_gate_up_slots(
        self,
        x,
        remapped_indices,
        *,
        sorted_indices: bool = False,
    ):
        combined = self.matmul_slots(
            x,
            remapped_indices,
            sorted_indices=sorted_indices,
        )
        gate = combined[..., :self.gate_out_features]
        up = combined[..., self.gate_out_features:]
        return gate, up


class PooledSwitchGLU(nn.Module):
    """Whole-SwitchGLU correctness seam over pooled projections."""

    def __init__(self, *, gate_proj, up_proj, down_proj, activation):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.activation = activation
        # Fused gate+up: SiLU(gate)*up in one Metal dispatch (vs two gather kernels
        # plus a Python `activation(x_up, x_gate)`). Restores a fast path the
        # pre-pooled streaming seam had and the pooled rewrite dropped. This restores
        # the previous behavior.
        # Precondition: jang's fused kernel takes one codebook+signs+bits for both
        # gate and up, so they must match. Real mjtq packages share them (gate/up have
        # the same in_features/bits/seed); guard so anything else falls back to the
        # exact separate path (correctness over speed).
        self._all_mxfp4 = (
            getattr(gate_proj, "codec", None) == MXFP4_CODEC
            and getattr(up_proj, "codec", None) == MXFP4_CODEC
            and getattr(down_proj, "codec", None) == MXFP4_CODEC
        )
        self._all_tq = (
            getattr(gate_proj, "codec", None) == TQ_CODEC
            and getattr(up_proj, "codec", None) == TQ_CODEC
            and getattr(down_proj, "codec", None) == TQ_CODEC
        )
        self._gate_up_tq = (
            getattr(gate_proj, "codec", None) == TQ_CODEC
            and getattr(up_proj, "codec", None) == TQ_CODEC
        )
        self._combined_gate_up_kquant = (
            isinstance(gate_proj, PooledCombinedGateUpKQuantLinear)
            and getattr(up_proj, "_parent", None) is gate_proj
        )
        self._fused_gate_up = (
            _FUSED_GATE_UP
            and not self._all_mxfp4
            and self._gate_up_tq
            and gate_proj.bits == up_proj.bits
            and bool(mx.array_equal(gate_proj.codebook, up_proj.codebook).item())
            and bool(mx.array_equal(gate_proj.signs, up_proj.signs).item())
        )
        self.fused_gate_up_calls = 0
        self.total_calls = 0
        self.decode_calls = 0
        self.prefill_calls = 0
        self.direct_calls = 0
        self.row_chunked_calls = 0
        self.sorted_chunked_calls = 0
        self.segmented_prefill_calls = 0
        self.unified_sorted_prefill_calls = 0
        self.barrier_free_prefill_calls = 0
        self.barrier_free_identity_calls = 0
        self.barrier_free_fused_swiglu_calls = 0
        self.barrier_free_decode_calls = 0
        self.barrier_free_decode_flush_calls = 0
        self.decode_routed_fused_calls = 0
        self.pipelined_decode_fused_calls = 0
        # One-shot eligibility verdict for the barrier-free prefill route
        # (None until the first bulk-prefill-shaped call decides it).
        self._barrier_free_ready_cached: bool | None = None
        # One-shot eligibility verdict for the barrier-free decode route
        # (None until the first decode-shaped call decides it).
        self._barrier_free_decode_ready_cached: bool | None = None
        # One-shot eligibility verdict for the fused decode routed matvec
        # family (None until the first engagement check decides it).
        self._decode_routed_fused_ready_cached: bool | None = None
        self.over_capacity_calls = 0
        self.total_token_layers = 0
        self.total_unique_active_experts = 0
        self.max_unique_active_experts = 0
        self.total_chunks = 0
        self.projection_load_wait_calls = 0
        self.projection_no_miss_calls = 0
        self.projection_load_wait_seconds = 0.0
        self.projection_load_parallel_calls = 0
        self.overlap_load_started_calls = 0
        self.overlap_load_wait_calls = 0
        self.overlap_load_wait_seconds = 0.0
        self.overlap_load_total_seconds = 0.0
        self.overlap_load_hidden_seconds = 0.0
        self.overlap_shared_eval_calls = 0
        self.overlap_shared_eval_seconds = 0.0
        self.overlap_prefill_no_eval_calls = 0
        self.overlap_no_miss_calls = 0
        self.overlap_skipped_over_capacity_calls = 0
        self.overlap_ticket_mismatch_calls = 0
        # Cross-chunk predictive prefetch (see _PREFILL_PREFETCH). One ticket
        # per layer at a time; submitted after an over-capacity call, consumed
        # (awaited) at the layer's next over-capacity call, then discarded.
        self._prefetch_ticket: _PrefetchTicket | None = None
        self.prefetch_ticket_submitted = 0
        self.prefetch_ticket_consumed = 0
        self.prefetch_ticket_mismatched = 0
        self.prefetch_ticket_stale = 0
        self.prefetch_ticket_experts = 0
        self.prefetch_ticket_loaded = 0
        self.prefetch_ticket_wait_seconds = 0.0
        self.remap_ondevice_calls = 0
        # Phase instrumentation (host-side wall time).
        # index_sync is the blocking np.asarray(indices) that forces
        # eval of all pending compute up to this layer's router: its duration
        # is where the GPU drain hides. index_resync is the second host read in
        # __call__ (expected cheap; measured to prove it). routed_build is the
        # host time spent building the routed graph (remap + kernel calls)
        # after misses are resident. It measures Python/graph overhead and excludes GPU work.
        self.index_sync_calls = 0
        self.index_sync_seconds = 0.0
        self.index_resync_calls = 0
        self.index_resync_seconds = 0.0
        self.routed_build_seconds = 0.0
        self.decode_moe_block_calls = 0
        self.decode_moe_block_seconds = 0.0
        self.router_gate_seconds = 0.0
        self.router_export_seconds = 0.0
        self.shared_experts_build_seconds = 0.0
        self.block_exit_kick_seconds = 0.0
        self.routed_weighted_sum_calls = 0
        self.routed_weighted_sum_slot_elements = 0
        self.routed_weighted_sum_output_elements = 0
        self.compiled_island_calls = 0
        self.block_exit_kick_calls = 0
        self._island_cache: dict = {}
        self._mxfp4_kernel_cache: dict = {}
        # Cross-layer lookahead state (install_lookahead wires these):
        # lookahead_w = fp16 router weight of layer L+Delta; lookahead_target
        # = that layer's PooledSwitchGLU (whose pools get the prefetch);
        # lookahead_b = that layer's per-expert selection bias when its gate
        # carries one (the DS4 score gate), None otherwise. The hot path only
        # checks `lookahead_w is not None`.
        self.lookahead_w = None
        self.lookahead_b = None
        self.lookahead_target = None
        self._pred_ring_buf = None
        self._pred_ring_np = None
        self._last_active: set[int] = set()
        self._spare_rr = 0
        self.lookahead_exports = 0
        self.lookahead_prefetch_loads = 0
        self.lookahead_ring_misses = 0
        self.lookahead_errors = 0
        self.lookahead_dropped = 0
        # Pipelined builder state (see build_pipelined)
        self.pipelined_layers = 0
        self.pipeline_read_seconds = 0.0
        self.pipeline_join_seconds = 0.0
        self._pipe_island_cache: dict = {}
        self._pipe_buf_cache: dict = {}
        # v3 ring-export state (see _RING_DECODE)
        self._ring_buf = None
        self._ring_np = None
        self.seen_experts: set[int] = set()
        self.prefill_seen_experts: set[int] = set()
        self.decode_seen_experts: set[int] = set()

    def grow_capacity(self, capacity: int) -> None:
        for pool in self._unique_projection_pools(lockstep=True):
            pool.grow(capacity)

    def seed_hot_free_slots(self) -> int:
        seeded = 0
        for pool in self._unique_projection_pools(lockstep=True):
            seeded += len(pool.seed_hot())
        return seeded

    def _unique_projection_pools(self, *, lockstep: bool = False):
        pools = (
            (self.gate_proj.pool, self.up_proj.pool, self.down_proj.pool)
            if lockstep
            else (self.up_proj.pool, self.gate_proj.pool, self.down_proj.pool)
        )
        out = []
        seen = set()
        for pool in pools:
            ident = id(pool)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(pool)
        return tuple(out)

    def _projection_pools(self):
        return self._unique_projection_pools(lockstep=False)

    def _projection_pools_lockstep(self):
        return self._unique_projection_pools(lockstep=True)

    def _touch_projection_pools_if_resident(self, active: set[int]) -> bool:
        """Fail-closed all-resident certificate for decode.

        All three projection pools must already contain every active expert
        while their bookkeeping locks are held. Only then can decode skip the
        demand ensure/wait path. Touching under the same locks preserves LFU
        accounting and eviction recency; if any expert is missing, nothing is
        published and the caller falls back to `ensure`.
        """
        pools = self._projection_pools_lockstep()
        locks = [pool._bk_lock for pool in pools]
        for lock in locks:
            lock.acquire()
        try:
            for pool in pools:
                if len(active) > pool.capacity:
                    return False
                for expert in active:
                    if expert < 0 or expert >= pool.num_experts:
                        raise IndexError(
                            f"expert {expert} out of range [0, {pool.num_experts})")
                    if expert not in pool._slot_of:
                        return False
            ordered = sorted(active)
            for pool in pools:
                pool._demand_protect = set(active)
                for expert in ordered:
                    pool.total_hits += 1
                    pool._touch(expert)
            return True
        finally:
            for lock in reversed(locks):
                lock.release()

    def begin_projection_load(self, indices) -> _ProjectionLoadTicket | None:
        """Start routed expert loads before the routed matmul needs them.

        This is the overlap seam. It intentionally starts after router
        indices are known and before shared-expert compute. The returned ticket is
        consumed by `__call__(..., load_ticket=ticket)`, which waits only for the
        unresolved tail. If the active set cannot fit in the pool, the caller uses
        the normal chunked path instead.
        """
        # Barrier-free bulk prefill never takes the overlap: every expert is
        # resident so there are no misses to start, and the np.asarray below
        # would reintroduce the per-layer graph drain the route removes.
        if self._barrier_free_bulk_shape(indices) and self._barrier_free_ready():
            return None
        capacity = min(pool.capacity for pool in self._projection_pools_lockstep())
        t0 = time.perf_counter()
        idx_host = np.asarray(indices).reshape(-1, indices.shape[-1])
        self.index_sync_calls += 1
        self.index_sync_seconds += time.perf_counter() - t0
        active = {int(e) for e in idx_host.reshape(-1).tolist()}
        if len(active) > capacity:
            self.overlap_skipped_over_capacity_calls += 1
            return None

        pools = self._projection_pools()
        missing_pools = [pool for pool in pools if pool.missing_count(active)]
        if not missing_pools:
            self.overlap_no_miss_calls += 1
            return None

        self.overlap_load_started_calls += 1
        futures = [
            _PROJECTION_LOAD_EXECUTOR.submit(pool.ensure, active)
            for pool in pools
        ]
        return _ProjectionLoadTicket(
            active=active,
            futures=futures,
            started_at=time.perf_counter(),
        )

    def _wait_projection_ticket(
        self,
        active: set[int],
        ticket: _ProjectionLoadTicket | None,
    ) -> bool:
        if ticket is None or ticket.used or ticket.active != active:
            if ticket is not None:
                self.overlap_ticket_mismatch_calls += 1
            return False

        ticket.used = True
        self.projection_load_wait_calls += 1
        self.projection_load_parallel_calls += 1
        self.overlap_load_wait_calls += 1
        wait_started = time.perf_counter()
        try:
            for future in ticket.futures:
                future.result()
        finally:
            done = time.perf_counter()
            wait_seconds = done - wait_started
            total_seconds = done - ticket.started_at
            self.projection_load_wait_seconds += wait_seconds
            self.overlap_load_wait_seconds += wait_seconds
            self.overlap_load_total_seconds += total_seconds
            self.overlap_load_hidden_seconds += max(0.0, total_seconds - wait_seconds)
        return True

    def _ensure_projection_pools(
        self,
        active: set[int],
        load_ticket: _ProjectionLoadTicket | None = None,
    ) -> None:
        if _PREFILL_PREFETCH and self._prefetch_ticket is not None:
            # A prefetch the previous over-capacity call submitted is still in
            # flight, and this call reached the demand path instead of the
            # sorted-chunked consume point (the layer's next call was not
            # over-capacity). Drain it before the demand ensure so the pool is
            # quiesced and the ticket lifecycle stays exactly-once.
            self._drain_stale_prefetch_ticket()
        if self._wait_projection_ticket(active, load_ticket):
            return
        if self._touch_projection_pools_if_resident(active):
            self.projection_no_miss_calls += 1
            return

        self.projection_load_wait_calls += 1
        t0 = time.perf_counter()
        pools = self._projection_pools()
        missing_pools = [
            pool
            for pool in pools
            if pool.missing_count(active)
        ]
        if len(missing_pools) < 2:
            try:
                for pool in pools:
                    pool.ensure(active)
            finally:
                self.projection_load_wait_seconds += time.perf_counter() - t0
            return

        self.projection_load_parallel_calls += 1
        try:
            futures = [
                _PROJECTION_LOAD_EXECUTOR.submit(pool.ensure, active)
                for pool in pools
            ]
            for future in futures:
                future.result()
        finally:
            self.projection_load_wait_seconds += time.perf_counter() - t0

    def _project_triplet(
        self,
        x,
        idx,
        idx_host,
        *,
        sorted_indices: bool,
        load_ticket: _ProjectionLoadTicket | None = None,
        preensured: bool = False,
    ):
        idx_shape = idx.shape
        idx_host = np.asarray(idx_host).reshape(-1)
        if not preensured:
            active = {int(e) for e in idx_host.tolist()}
            self._ensure_projection_pools(active, load_ticket=load_ticket)
        build_t0 = time.perf_counter()
        try:
            return self._project_triplet_resident(
                x, idx, idx_shape, idx_host, sorted_indices=sorted_indices)
        finally:
            self.routed_build_seconds += time.perf_counter() - build_t0

    def _get_compiled_island(self, K: int):
        """Build (once per K) the mx.compile'd routed-MLP closure.

        Follows jang's decode patch (jangrt/switchglu_decode.py:_mlp): the
        traced graph is remap-gather x2 -> rotate -> fused gate/up/SwiGLU ->
        rotate -> down gather. The decode factories bake shapes/meta as traced
        constants; pool buffers, slot tables and indices are runtime inputs, so
        residency changes outside the island never invalidate the trace."""
        island = self._island_cache.get(K)
        if island is not None:
            return island
        from jang_tools.turboquant.fused_gate_up_kernel import (
            make_fused_gate_up_swiglu_decode,
        )
        from jang_tools.turboquant.gather_tq_kernel import (
            make_gather_tq_decode_per_row,
        )
        from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal

        in_f = self.gate_proj.in_features
        out_f = self.gate_proj.out_features
        swiglu_limit = getattr(self.activation, "swiglu_limit", 0.0) or 0.0
        fused_gu = make_fused_gate_up_swiglu_decode(
            in_f, out_f, self.gate_proj.bits, K, swiglu_limit=swiglu_limit)
        gather_dn = make_gather_tq_decode_per_row(
            out_f, in_f, self.down_proj.bits, K)

        def _island(x_flat, gate_table, down_table, idx_flat,
                    pg, ng, pu, nu, pd, nd, cb_g, cb_d, s_in, s_dn):
            slot_g = gate_table[idx_flat]
            slot_d = down_table[idx_flat]
            x_rot = hadamard_rotate_metal(x_flat, s_in)
            x_act = fused_gu(x_rot, pg, ng, pu, nu, cb_g, slot_g)
            # fp16 bottleneck on purpose: the eager path converts the fused
            # output to the activation dtype before the down gather. Keeping
            # the same conversion makes island-on/off bit-exact (kill-switch
            # A/Bs compare equal), at negligible compiled cost.
            x_act = x_act.astype(mx.float16).astype(mx.float32)
            x_act_rot = hadamard_rotate_metal(x_act, s_dn)
            return gather_dn(x_act_rot, pd, nd, cb_d, slot_d)

        island = mx.compile(_island)
        self._island_cache[K] = island
        return island

    def _get_mxfp4_kernel(self, K: int):
        kernel = self._mxfp4_kernel_cache.get(K)
        if kernel is not None:
            return kernel
        from moespresso.runtime.routed_decode_kernel import make_routed_mxfp4_decode_kernel

        swiglu_limit = getattr(self.activation, "swiglu_limit", 0.0) or 0.0
        kernel = make_routed_mxfp4_decode_kernel(
            in_f=self.gate_proj.in_features,
            out_f=self.gate_proj.out_features,
            K=K,
            swiglu_limit=swiglu_limit,
        )
        if kernel is None:
            raise RuntimeError(
                "source-mxfp4 routed decode kernel is unsupported for "
                f"in_f={self.gate_proj.in_features} out_f={self.gate_proj.out_features}")
        self._mxfp4_kernel_cache[K] = kernel
        return kernel

    def _project_triplet_resident(self, x, idx, idx_shape, idx_host,
                                  *, sorted_indices: bool):
        # Compiled-island decode fast path: single token (idx is one row of
        # K experts), unsorted, on-device remap, fused preconditions hold.
        K = idx_shape[-1] if len(idx_shape) > 0 else 0
        if (
            _COMPILED_ISLAND
            and self._all_mxfp4
            and _ONDEVICE_REMAP
            and not sorted_indices
            and K > 0
            and idx.size == K
            and not self.training
        ):
            try:
                kernel = self._get_mxfp4_kernel(K)
            except RuntimeError:
                kernel = None
            if kernel is not None:
                self.compiled_island_calls += 1
                self.fused_gate_up_calls += 1
                gate_table = self.gate_proj.pool._ensure_slot_table()
                down_table = self.down_proj.pool._ensure_slot_table()
                idx_flat = idx.reshape(-1)
                x_flat = x.reshape(-1, self.gate_proj.in_features).astype(mx.float32)
                y = kernel(
                    x_flat,
                    self.gate_proj.pool.packed,
                    self.gate_proj.pool.scales,
                    self.up_proj.pool.packed,
                    self.up_proj.pool.scales,
                    self.down_proj.pool.packed,
                    self.down_proj.pool.scales,
                    gate_table[idx_flat],
                    down_table[idx_flat],
                )
                out = y.reshape(*idx_shape[:-1], K, 1, self.down_proj.out_features)
                if out.dtype != x.dtype:
                    out = out.astype(x.dtype)
                return out
        if (
            _COMPILED_ISLAND
            and self._fused_gate_up
            and self._all_tq
            and _ONDEVICE_REMAP
            and not sorted_indices
            and K > 0
            and idx.size == K
            and not self.training
        ):
            self.compiled_island_calls += 1
            self.fused_gate_up_calls += 1  # the fused kernel runs inside the island
            island = self._get_compiled_island(K)
            gate_table = self.gate_proj.pool._ensure_slot_table()
            down_table = self.down_proj.pool._ensure_slot_table()
            x_flat = x.reshape(-1, self.gate_proj.in_features).astype(mx.float32)
            y = island(
                x_flat,
                gate_table, down_table, idx.reshape(-1),
                self.gate_proj.pool.packed, self.gate_proj.pool.norms,
                self.up_proj.pool.packed, self.up_proj.pool.norms,
                self.down_proj.pool.packed, self.down_proj.pool.norms,
                self.gate_proj.codebook, self.down_proj.codebook,
                self.gate_proj.signs, self.down_proj.signs,
            )
            out = y.reshape(*idx_shape[:-1], K, 1, self.down_proj.out_features)
            if out.dtype != x.dtype:
                out = out.astype(x.dtype)
            return out
        # Unified sorted prefill: the partial-residency chunked path computes
        # each pre-ensured, expert-sorted chunk through the same fused sorted
        # kernels the full-resident barrier-free route runs, so the served
        # tokens match the full-residency rail at any capacity (see
        # _UNIFIED_SORTED_PREFILL). Covers both the large-chunk (segmented) and
        # small-chunk (general gather) cases with one kernel, since prefill
        # carries no cross-row reduction.
        if (
            sorted_indices
            and self._combined_gate_up_kquant
            and idx_host is not None
            and self._unified_sorted_ready()
        ):
            self.unified_sorted_prefill_calls += 1
            return self._call_sorted_fused(x, idx_host)
        # Bulk sorted prefill: per-expert segments read each expert's weights
        # once (see _SEGMENTED_PREFILL_MIN_ROWS). Slot lookup is host-side, so
        # this path needs no remapped index tensors at all. This is the
        # pre-unification compute, reached with MOESPRESSO_SSD_UNIFIED_PREFILL=0
        # or when the fused sorted kernel is unavailable.
        if (
            sorted_indices
            and self._combined_gate_up_kquant
            and idx_host is not None
            and int(np.size(idx_host)) >= _SEGMENTED_PREFILL_MIN_ROWS
        ):
            self.segmented_prefill_calls += 1
            combined = self.gate_proj.matmul_slots_segmented(x, idx_host)
            gate_n = self.gate_proj.gate_out_features
            x_act = self.activation(
                combined[..., gate_n:], combined[..., :gate_n])
            return self.down_proj.matmul_slots_segmented(x_act, idx_host)

        # After _ensure_projection_pools, every active expert is resident in all three
        # pools, so the on-device gather is exact (no sentinel). It keeps the index
        # tensors on-device into the kernel instead of the 3x host round-trip.
        if _ONDEVICE_REMAP:
            self.remap_ondevice_calls += 1
            up_idx = self.up_proj.pool.remap_ondevice(idx)
            gate_idx = self.gate_proj.pool.remap_ondevice(idx)
            down_idx = self.down_proj.pool.remap_ondevice(idx)
        else:
            up_idx = self.up_proj.pool.remap_loaded(idx_host, idx_shape)
            gate_idx = self.gate_proj.pool.remap_loaded(idx_host, idx_shape)
            down_idx = self.down_proj.pool.remap_loaded(idx_host, idx_shape)

        if self._combined_gate_up_kquant:
            x_gate, x_up = self.gate_proj.matmul_gate_up_slots(
                x,
                gate_idx,
                sorted_indices=sorted_indices,
            )
            out = self.down_proj.matmul_slots(
                self.activation(x_up, x_gate),
                down_idx,
                sorted_indices=sorted_indices,
            )
            return out

        if self._fused_gate_up:
            # One Metal dispatch for SiLU(gate)*up (vs 2 gather kernels + a Python
            # activation). gate/up pools are loaded together so slot N holds the same
            # expert in both (pinned by test); gate_idx indexes packed_gate and
            # packed_up. Norms are slotted. Down stays on the gather path.
            from jang_tools.turboquant.fused_gate_up_kernel import (
                fused_gate_up_swiglu_matmul,
            )
            self.fused_gate_up_calls += 1
            swiglu_limit = getattr(self.activation, "swiglu_limit", 0.0) or 0.0
            x_act = fused_gate_up_swiglu_matmul(
                x,
                self.gate_proj.pool.packed, self.gate_proj.pool.norms,
                self.up_proj.pool.packed, self.up_proj.pool.norms,
                self.gate_proj.codebook, self.gate_proj.signs,
                gate_idx,
                bits=self.gate_proj.bits,
                swiglu_limit=swiglu_limit,
            )
            return self.down_proj.matmul_slots(
                x_act, down_idx, sorted_indices=sorted_indices)

        x_up = self.up_proj.matmul_slots(
            x,
            up_idx,
            sorted_indices=sorted_indices,
        )
        x_gate = self.gate_proj.matmul_slots(
            x,
            gate_idx,
            sorted_indices=sorted_indices,
        )
        out = self.down_proj.matmul_slots(
            self.activation(x_up, x_gate),
            down_idx,
            sorted_indices=sorted_indices,
        )
        return out

    def _unified_sorted_ready(self) -> bool:
        """Whether the unified fused sorted prefill compute is usable.

        Static preconditions only: the kill switch, the combined K-quant
        gate/up pool, the K-quant down pool, and the installed
        gather_qmm_sorted_swiglu / gather_qmm_sorted kernels. A False verdict
        falls back to the pre-unification chunked compute (segmented f32 GEMM
        or general gather plus a separate activation), the fail-closed
        direction. Decided per call rather than cached because it depends only
        on process-stable facts and the check is a handful of attribute reads.
        """
        if not _UNIFIED_SORTED_PREFILL:
            return False
        if not (_FUSED_SORTED_SWIGLU and self._combined_gate_up_kquant):
            return False
        if getattr(self.down_proj, "codec", None) != KQUANT_CODEC:
            return False
        try:
            import mlx_kquant as kq
        except ImportError:
            return False
        return (
            getattr(kq, "gather_qmm_sorted_swiglu", None) is not None
            and getattr(kq, "gather_qmm_sorted", None) is not None
        )

    def _call_sorted_fused(self, x, idx_host_sorted) -> mx.array:
        """Fused sorted prefill compute for a pre-ensured, expert-sorted chunk.

        `x` holds the chunk's routed rows already sorted by expert id; the
        matching flat expert ids are `idx_host_sorted`. Every listed expert is
        resident (the caller ensured the chunk's active set into slots), so the
        expert ids remap to valid slots. This runs the same fused kernels the
        full-resident barrier-free route runs: gather_qmm_sorted_swiglu for the
        combined gate/up GEMM with the SwiGLU applied on the float32
        accumulators, then gather_qmm_sorted for the down GEMM, over slot ids.
        Because prefill carries no cross-row reduction, per-row output equals
        the full path's output for the same rows regardless of how the sorted
        rows are chunked. `x` may carry singleton leading axes (the callers
        expand `[rows, in]` to `[rows, 1, in]` before the gather sort); the
        output keeps `x`'s leading shape with `out_features` last, matching the
        segmented/general branch this replaces so the caller's unsort and
        squeeze contract is unchanged.
        """
        import mlx_kquant as kq

        lead_shape = x.shape[:-1]
        flat_ids = mx.array(np.asarray(idx_host_sorted).reshape(-1).astype(np.uint32))
        # Remap expert ids to slot ids (identity when the pool is prewarmed at
        # slot == expert id), then re-sort by slot so each expert's rows are one
        # contiguous segment the sorted kernel derives in-kernel. The gate/up
        # and down pools may hold an expert at different slots, so each gets its
        # own slot order (mirrors _call_barrier_free's non-identity branch).
        gate_slots = self.gate_proj.pool.remap_ondevice(flat_ids)
        down_slots = self.down_proj.pool.remap_ondevice(flat_ids)
        order_g = mx.argsort(gate_slots)
        order_d = mx.argsort(down_slots)
        gate_sorted = gate_slots[order_g]
        down_sorted = down_slots[order_d]

        x_rows = x.reshape(-1, self.gate_proj.in_features)
        x_g = x_rows[order_g]
        self.gate_proj.matmul_slot_calls += 1
        self.gate_proj.matmul_slot_elements += int(flat_ids.size)
        gate_n = self.gate_proj.gate_out_features
        swiglu_limit = getattr(self.activation, "swiglu_limit", 0.0) or 0.0
        x_act = kq.gather_qmm_sorted_swiglu(
            x_g,
            self.gate_proj.pool.weight,
            self.gate_proj.pool.scales,
            self.gate_proj.kquant_type,
            gate_sorted,
            gate_n,
            swiglu_limit,
        )
        # Re-permute the activation rows from gate-slot order into down-slot
        # order (a no-op gather when both slot tables agree, as under prewarm).
        x_act = x_act[mx.argsort(order_g)[order_d]]
        self.down_proj.matmul_slot_calls += 1
        self.down_proj.matmul_slot_elements += int(flat_ids.size)
        down = kq.gather_qmm_sorted(
            x_act,
            self.down_proj.pool.weight,
            self.down_proj.pool.scales,
            self.down_proj.kquant_type,
            down_sorted,
        )
        # Unsort to the incoming expert-sorted row order, then restore the
        # caller's leading shape (out_features replaces in_features).
        out = down[mx.argsort(order_d)]
        return out.reshape(*lead_shape, self.down_proj.out_features)

    def _barrier_free_bulk_shape(self, indices) -> bool:
        """Bulk-prefill shape gate for the barrier-free route.

        Mirrors the sorted-path gate (>= 64 routed pairs) and the segmented
        row threshold; indices is [..., top_k], so token rows x top_k is
        exactly indices.size. Shape-only: never touches index values."""
        return bool(
            indices.size >= 64
            and indices.size >= _SEGMENTED_PREFILL_MIN_ROWS
        )

    def _barrier_free_ready(self) -> bool:
        """One-shot fail-closed eligibility check for barrier-free prefill.

        The verdict is decided once and cached: at capacity == num_experts a
        fully resident pool never evicts (ensure can never miss), so a True
        verdict is stable for the process; a False verdict (partial residency,
        smaller capacity, non-K-quant projections, or a kernel-less
        mlx_kquant) keeps the route off for the process, which is the
        fail-closed direction. Serving reaches the first bulk prefill only
        after the build-time prewarm, so full residency is already
        established when the verdict is taken."""
        ready = self._barrier_free_ready_cached
        if ready is None:
            ready = self._barrier_free_eligible()
            self._barrier_free_ready_cached = ready
        return ready

    def _barrier_free_eligible(self) -> bool:
        if not _BARRIER_FREE_PREFILL:
            return False
        if not self._combined_gate_up_kquant:
            return False
        if getattr(self.down_proj, "codec", None) != KQUANT_CODEC:
            return False
        try:
            import mlx_kquant as kq
        except ImportError:
            return False
        if getattr(kq, "gather_qmm_sorted", None) is None:
            return False
        for pool in self._projection_pools_lockstep():
            if pool.capacity != pool.num_experts:
                return False
            if len(pool._slot_of) != pool.num_experts:
                return False
        return True

    def _call_barrier_free(self, x, indices) -> mx.array:
        """Full-resident bulk prefill with zero host synchronization.

        Device-only route: gather each projection's slot ids from its
        on-device slot table, argsort them, run the sorted K-quant GEMM
        (mlx_kquant.gather_qmm_sorted derives the per-expert row ranges
        in-kernel), and scatter-unsort, so the routed block queues into the
        same lazy graph as the rest of prefill. On the identity route the
        gate/up GEMM and the SwiGLU fuse into the single
        gather_qmm_sorted_swiglu kernel when the installed mlx_kquant ships
        it (see _FUSED_SORTED_SWIGLU). Rows keep their incoming
        dtype end to end (the kernel stages weights in f32 for every I/O
        dtype), preserving the segmented path's f32 weight-decode contract.
        Callers guaranteed full residency, so no ensure(), no miss handling,
        and no index_sync/index_resync host read happens here at all.

        Full prewarm seeds ascending experts into ascending slots, so both
        slot tables are usually the identity map. Routed ids then already
        are slot ids for both projections: one argsort serves both GEMMs and
        the inter-GEMM re-permutation (a no-op gather in that case, but a
        full [rows, in_features] copy of the activation tensor) disappears.
        Per-row math is unchanged, so the identity route is bit-identical to
        the general one; pools filled in any other order keep the general
        per-pool remap."""
        import mlx_kquant as kq

        top_k = int(indices.shape[-1])
        flat_idx = indices.reshape(-1)
        identity = (
            self.gate_proj.pool.slot_table_is_identity()
            and self.down_proj.pool.slot_table_is_identity()
        )
        if identity:
            self.barrier_free_identity_calls += 1
            order_g = mx.argsort(flat_idx)
            order_d = order_g
            gate_sorted = flat_idx[order_g]
            down_sorted = gate_sorted
        else:
            gate_slots = self.gate_proj.pool._ensure_slot_table()[flat_idx]
            down_slots = self.down_proj.pool._ensure_slot_table()[flat_idx]
            order_g = mx.argsort(gate_slots)
            order_d = mx.argsort(down_slots)
            gate_sorted = gate_slots[order_g]
            down_sorted = down_slots[order_d]

        x_tokens = x.reshape(-1, self.gate_proj.in_features)
        x_g = x_tokens[order_g // top_k]
        self.gate_proj.matmul_slot_calls += 1
        self.gate_proj.matmul_slot_elements += int(indices.size)
        gate_n = self.gate_proj.gate_out_features
        # Fused gate/up + SwiGLU on the identity route: one kernel replaces
        # the combined GEMM plus the elementwise activation, applying the
        # same formula in its epilogue on the float32 accumulators (see
        # _FUSED_SORTED_SWIGLU for the numerics note).
        if (
            identity
            and _FUSED_SORTED_SWIGLU
            and getattr(kq, "gather_qmm_sorted_swiglu", None) is not None
        ):
            self.barrier_free_fused_swiglu_calls += 1
            swiglu_limit = getattr(self.activation, "swiglu_limit", 0.0) or 0.0
            x_act = kq.gather_qmm_sorted_swiglu(
                x_g,
                self.gate_proj.pool.weight,
                self.gate_proj.pool.scales,
                self.gate_proj.kquant_type,
                gate_sorted,
                gate_n,
                swiglu_limit,
            )
        else:
            combined = kq.gather_qmm_sorted(
                x_g,
                self.gate_proj.pool.weight,
                self.gate_proj.pool.scales,
                self.gate_proj.kquant_type,
                gate_sorted,
            )
            x_act = self.activation(
                combined[..., gate_n:], combined[..., :gate_n])

        if not identity:
            # Re-permute the activation rows from gate-slot order into
            # down-slot order: row j of the down input is the gate output row
            # holding original pair order_d[j], i.e. argsort(order_g)[order_d[j]].
            x_act = x_act[mx.argsort(order_g)[order_d]]
        self.down_proj.matmul_slot_calls += 1
        self.down_proj.matmul_slot_elements += int(indices.size)
        down = kq.gather_qmm_sorted(
            x_act,
            self.down_proj.pool.weight,
            self.down_proj.pool.scales,
            self.down_proj.kquant_type,
            down_sorted,
        )
        # Unsort to flat (token, route) order, then to the caller's
        # [..., top_k, out_features] contract (matching _call_direct).
        out = down[mx.argsort(order_d)]
        return mx.unflatten(out, 0, indices.shape)

    def _barrier_free_decode_ready(self) -> bool:
        """One-shot fail-closed eligibility check for barrier-free decode.

        Decode analog of `_barrier_free_ready`: the verdict is decided once
        and cached. At capacity == num_experts a fully resident pool never
        evicts, so a True verdict is stable for the process; a False verdict
        (partial residency, smaller capacity, non-K-quant projections, or a
        kernel-less mlx_kquant) keeps the ring/native-gate decode path for
        the process, which is the fail-closed direction. The ring path stays
        the product path for every partial-residency session."""
        ready = self._barrier_free_decode_ready_cached
        if ready is None:
            ready = self._barrier_free_decode_eligible()
            self._barrier_free_decode_ready_cached = ready
        return ready

    def _barrier_free_decode_eligible(self) -> bool:
        if not _BARRIER_FREE_DECODE:
            return False
        if not self._combined_gate_up_kquant:
            return False
        if getattr(self.down_proj, "codec", None) != KQUANT_CODEC:
            return False
        try:
            import mlx_kquant as kq
        except ImportError:
            return False
        if getattr(kq, "gather_qmm", None) is None:
            return False
        # Residency is read under the pool bookkeeping locks (same acquire
        # order as _touch_projection_pools_if_resident) so the verdict
        # cannot race a concurrent load or eviction mid-check.
        pools = self._projection_pools_lockstep()
        locks = [pool._bk_lock for pool in pools]
        for lock in locks:
            lock.acquire()
        try:
            for pool in pools:
                if pool.capacity != pool.num_experts:
                    return False
                if len(pool._slot_of) != pool.num_experts:
                    return False
        finally:
            for lock in reversed(locks):
                lock.release()
        return True

    def build_barrier_free_decode(self, x, idx) -> mx.array:
        """Full-resident decode routed MLP over device-resident router ids.

        Callers hold the `_barrier_free_decode_ready` certificate, so there
        are no misses to load and routing never touches the host: no ring
        export, no event gate, no worker submit. The routed graph is the
        same combined gate/up gather, activation, and down gather the
        pipelined builder emits; only the index source differs (router ids
        consumed on device instead of worker-published slot buffers). On
        the prewarm-all fill order both slot tables are the identity map
        and routed ids already are slot ids; any other full-resident fill
        order takes one on-device slot-table gather per pool. Per-row math
        is unchanged either way, so the route is bit-identical to the ring
        path. LFU touch accounting is skipped: a full pool never evicts,
        matching the barrier-free prefill counter policy."""
        self.barrier_free_decode_calls += 1
        if (
            self.gate_proj.pool.slot_table_is_identity()
            and self.down_proj.pool.slot_table_is_identity()
        ):
            gate_idx = idx
            down_idx = idx
        else:
            gate_idx = self.gate_proj.pool.remap_ondevice(idx)
            down_idx = self.down_proj.pool.remap_ondevice(idx)
        x4 = mx.expand_dims(x, (-2, -3))
        x_gate, x_up = self.gate_proj.matmul_gate_up_slots(
            x4,
            gate_idx,
            sorted_indices=False,
        )
        out = self.down_proj.matmul_slots(
            self.activation(x_up, x_gate),
            down_idx,
            sorted_indices=False,
        )
        return out.squeeze(-2)

    def _decode_routed_fused_ready(self) -> bool:
        """One-shot fail-closed eligibility check for the fused decode
        routed matvec family.

        Static facts only (env flag, pool layout, codecs, kernel geometry,
        installed mlx_kquant surface); the verdict is decided once and
        cached. The per-call identity-slot condition lives in
        `decode_routed_fused_engaged`, and callers only reach either check
        while holding the `_barrier_free_decode_ready` certificate. A False
        verdict keeps the unfused barrier-free route, which is the
        fail-closed direction."""
        ready = self._decode_routed_fused_ready_cached
        if ready is None:
            ready = self._decode_routed_fused_eligible()
            self._decode_routed_fused_ready_cached = ready
        return ready

    def _decode_routed_fused_eligible(self) -> bool:
        if not _DECODE_ROUTED_FUSED:
            return False
        if not self._combined_gate_up_kquant:
            return False
        if getattr(self.down_proj, "codec", None) != KQUANT_CODEC:
            return False
        # The decode matvec kernels are instantiated for the DS4 routed
        # codec pair only; other codecs keep the unfused route.
        if getattr(self.gate_proj, "kquant_type", None) != "iq2_xxs":
            return False
        if getattr(self.down_proj, "kquant_type", None) != "q2_k":
            return False
        try:
            import mlx_kquant as kq
        except ImportError:
            return False
        if getattr(kq, "gather_qmv_pair_swiglu", None) is None:
            return False
        if getattr(kq, "gather_qmv_expert_sum", None) is None:
            return False
        gate_out = int(self.gate_proj.gate_out_features)
        if gate_out != int(self.gate_proj.up_out_features):
            return False
        if int(self.down_proj.in_features) != gate_out:
            return False
        # Kernel geometry: whole super-blocks on both inner dims, 4-row
        # output blocks on both output dims.
        if int(self.gate_proj.in_features) % 256 or gate_out % 256:
            return False
        if gate_out % 4 or int(self.down_proj.out_features) % 4:
            return False
        return True

    def decode_routed_fused_engaged(self) -> bool:
        """Per-call engagement check for the fused decode routed matvec
        family: the one-shot eligibility verdict plus identity slot tables
        (router ids then already are slot ids for both pools, so the
        kernels index the pool stacks directly). Any other fill order keeps
        the unfused barrier-free route."""
        if not self._decode_routed_fused_ready():
            return False
        return (
            self.gate_proj.pool.slot_table_is_identity()
            and self.down_proj.pool.slot_table_is_identity()
        )

    def build_barrier_free_decode_fused(self, x, idx, scores) -> mx.array:
        """Fused decode routed MLP: two dispatches, route weights baked in.

        Callers hold the `_barrier_free_decode_ready` certificate and the
        `decode_routed_fused_engaged` verdict. gather_qmv_pair_swiglu
        computes every routed expert's SwiGLU intermediate with the route
        weight baked in; gather_qmv_expert_sum computes the down matvec and
        sums the experts in-kernel, so the caller applies no weighted sum.
        Returns the summed routed output shaped like x."""
        import mlx_kquant as kq

        self.barrier_free_decode_calls += 1
        self.decode_routed_fused_calls += 1
        ids = idx.reshape(-1)
        x_flat = x.reshape(1, self.gate_proj.in_features)
        weights = scores.reshape(-1).astype(mx.float32)
        swiglu_limit = getattr(self.activation, "swiglu_limit", 0.0) or 0.0
        mid = kq.gather_qmv_pair_swiglu(
            x_flat,
            self.gate_proj.pool.weight,
            self.gate_proj.pool.scales,
            self.gate_proj.kquant_type,
            ids,
            weights,
            self.gate_proj.gate_out_features,
            swiglu_limit,
        )
        y = kq.gather_qmv_expert_sum(
            mid,
            self.down_proj.pool.weight,
            self.down_proj.pool.scales,
            self.down_proj.kquant_type,
            ids,
        )
        return y.reshape(*x.shape[:-1], self.down_proj.out_features)

    def weighted_output(
        self,
        x,
        indices,
        scores,
        *,
        load_ticket: _ProjectionLoadTicket | None = None,
    ) -> mx.array:
        """Return the route-weighted SwitchGLU output."""
        y = self(x, indices, load_ticket=load_ticket)
        _record_routed_weighted_sum(self, scores, out_features=int(y.shape[-1]))
        return _deepseek_v4_weighted_sum(y, scores).reshape(
            *x.shape[:-1],
            y.shape[-1],
        )

    def __call__(
        self,
        x,
        indices,
        *,
        load_ticket: _ProjectionLoadTicket | None = None,
    ) -> mx.array:
        if _ROUTE_TRACE is not None:
            arr = np.asarray(indices).reshape(-1, indices.shape[-1])
            tag = "prefill" if arr.shape[0] > 1 else "decode_direct"
            _ROUTE_TRACE.append(
                (tag, self.gate_proj.pool.layer, arr.tolist()))
        capacity = min(pool.capacity for pool in self._projection_pools_lockstep())
        bulk_rows = 1
        for dim in indices.shape[:-1]:
            bulk_rows *= int(dim)
        # Barrier-free full-resident bulk prefill: leaves before the blocking
        # np.asarray(indices) below, so index_sync/index_resync stay untouched
        # on this route (their absence in the stats is the engagement
        # evidence, next to barrier_free_prefill_calls). The expert-set
        # counters (seen/unique) also stay untouched: updating them would
        # need the very host read the route removes.
        if self._barrier_free_bulk_shape(indices) and self._barrier_free_ready():
            self.total_calls += 1
            self.prefill_calls += 1
            self.total_token_layers += bulk_rows
            self.barrier_free_prefill_calls += 1
            return self._call_barrier_free(x, indices)
        # When begin_projection_load already synced this layer's indices, this
        # re-read is a cheap host copy; with overlap off (or prefill) it is the
        # blocking sync itself. Timed separately from index_sync so the two
        # cases stay distinguishable in stats.
        t0 = time.perf_counter()
        idx_host = np.asarray(indices).reshape(-1, indices.shape[-1])
        self.index_resync_calls += 1
        self.index_resync_seconds += time.perf_counter() - t0
        token_layers = int(idx_host.shape[0])
        active = {int(e) for e in idx_host.reshape(-1).tolist()}
        self.total_calls += 1
        self.total_token_layers += token_layers
        self.total_unique_active_experts += len(active)
        self.max_unique_active_experts = max(
            self.max_unique_active_experts,
            len(active),
        )
        self.seen_experts.update(active)
        if token_layers == 1:
            self.decode_calls += 1
            self.decode_seen_experts.update(active)
        else:
            self.prefill_calls += 1
            self.prefill_seen_experts.update(active)
        if len(active) > capacity:
            self.over_capacity_calls += 1
            if indices.size >= 64:
                self.sorted_chunked_calls += 1
                return self._call_sorted_chunked(x, indices, capacity)
            self.row_chunked_calls += 1
            return self._call_chunked(x, indices, idx_host, capacity)
        self.direct_calls += 1
        return self._call_direct(
            x,
            indices,
            load_ticket=load_ticket,
            idx_host_flat=idx_host.reshape(-1),
        )

    def _call_chunked(self, x, indices, idx_host, capacity: int) -> mx.array:
        x_flat = x.reshape(-1, x.shape[-1])
        idx_flat = indices.reshape(-1, indices.shape[-1])
        if x_flat.shape[0] != idx_host.shape[0]:
            raise ExpertCapacityExceeded(
                "cannot chunk pooled SwitchGLU: x/indices token counts differ")

        chunks = []
        start = 0
        active: set[int] = set()
        for row, expert_row in enumerate(idx_host):
            row_active = {int(e) for e in expert_row.tolist()}
            if len(row_active) > capacity:
                raise ExpertCapacityExceeded(
                    f"capacity {capacity} cannot hold active experts "
                    f"{sorted(row_active)}")
            if row > start and len(active | row_active) > capacity:
                chunks.append((start, row))
                start = row
                active = set(row_active)
            else:
                active |= row_active
        chunks.append((start, idx_host.shape[0]))
        self.total_chunks += len(chunks)

        outputs = []
        for s, e in chunks:
            out = self._call_direct(x_flat[s:e], idx_flat[s:e])
            mx.eval(out)
            outputs.append(out)
        out = mx.concatenate(outputs, axis=0)
        return out.reshape(*x.shape[:-1], *out.shape[1:])

    def _consume_prefetch_ticket(self, actual_active: set[int]) -> None:
        """Await and discard this layer's pending cross-chunk prefetch ticket.

        Called before an over-capacity call touches its pools. The prefetch runs
        on the IO executor between this layer's calls and is the only mutator in
        that gap (per-layer pools), so awaiting its futures here quiesces it
        before the chunk-ahead path begins. A predicted set that differs from
        the call's actual demand is counted as a mismatch; the prefetch still
        pre-filled whatever slots it hit, and the normal path services the rest.
        """
        ticket = self._prefetch_ticket
        if ticket is None:
            return
        self._prefetch_ticket = None
        self.prefetch_ticket_consumed += 1
        if ticket.predicted != frozenset(actual_active):
            self.prefetch_ticket_mismatched += 1
        wait_started = time.perf_counter()
        try:
            for future in ticket.futures:
                self.prefetch_ticket_loaded += int(future.result())
        finally:
            self.prefetch_ticket_wait_seconds += (
                time.perf_counter() - wait_started)

    def _drain_stale_prefetch_ticket(self) -> None:
        """Await and discard a pending prefetch ticket the demand path did not
        consume (the layer's next call did not take the over-capacity sorted
        path). The prefetch and a demand ensure already coexist safely on the
        pool, but draining keeps the ticket lifecycle exactly-once and the
        pool quiesced before the demand ensure runs."""
        ticket = self._prefetch_ticket
        if ticket is None:
            return
        self._prefetch_ticket = None
        self.prefetch_ticket_stale += 1
        for future in ticket.futures:
            self.prefetch_ticket_loaded += int(future.result())

    def _submit_prefetch_ticket(
        self,
        predicted: set[int],
        protect: set[int] | None = None,
    ) -> None:
        """Submit a background best-effort prefetch of `predicted` (the demand
        set the just-finished over-capacity call used) on the IO executor, and
        store it as this layer's ticket. Best-effort `prefetch` never raises on
        capacity. `protect` pins the experts whose slots may still have
        readers in flight (the caller's final capacity-chunk); the caller
        drains every other chunk's readers before submitting, so the
        prefetch's victims are exactly the drained chunks. The layer's next
        call awaits this ticket before it touches the pools."""
        if not predicted:
            return
        # A prior ticket that was never consumed (unusual: the layer's previous
        # call took the sorted path but a still-earlier ticket lingered) is
        # drained here so only one prefetch is ever in flight per layer.
        self._drain_stale_prefetch_ticket()
        ordered = sorted(predicted)
        pools = self._projection_pools()
        # Reserve a floor of free slots for a demand ensure that could race the
        # prefetch (the pool contract): capped at the pool's own default at the
        # 192-slot product capacity, and shrunk on the tiny pools tests build so
        # the prefetch still fills slots there. prefetch also protects
        # _demand_protect regardless of the explicit set.
        capacity = min(pool.capacity for pool in pools)
        reserve_floor = min(16, capacity // 2)
        futures = [
            _PROJECTION_LOAD_EXECUTOR.submit(
                pool.prefetch, ordered, protect=protect,
                reserve_floor=reserve_floor)
            for pool in pools
        ]
        self._prefetch_ticket = _PrefetchTicket(
            predicted=frozenset(predicted),
            futures=futures,
            submitted_at=time.perf_counter(),
        )
        self.prefetch_ticket_submitted += 1
        self.prefetch_ticket_experts += len(predicted)

    def _call_sorted_chunked(self, x, indices, capacity: int) -> mx.array:
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))
        x_sorted, idx_sorted, inv_order = _gather_sort(x, indices)
        idx_host = np.asarray(idx_sorted).reshape(-1)
        # Chunk-ahead overlap: chunk to half capacity so chunks
        # i and i+1 coexist in the pool (chunk i+1's experts pread on the IO
        # executor while the GPU computes chunk i; its evictions can only
        # touch chunks <= i-1, already kicked; the batched fence covers slot
        # safety). A one-slot pool cannot hold two chunks, so it takes the
        # full-capacity eval-per-chunk path (IO and compute strictly
        # alternate there).
        overlap = capacity >= 2
        chunk_capacity = capacity // 2 if overlap else capacity
        chunks = []
        start = 0
        active: set[int] = set()
        for pos, expert in enumerate(idx_host.tolist()):
            expert = int(expert)
            if (
                pos > start
                and expert not in active
                and len(active) >= chunk_capacity
            ):
                chunks.append((start, pos))
                start = pos
                active = {expert}
            else:
                active.add(expert)
        chunks.append((start, idx_host.shape[0]))
        self.total_chunks += len(chunks)

        call_active = {int(e) for e in idx_host.tolist()}
        if _PREFILL_PREFETCH:
            # Consume before any pool touch this call: await the ticket the
            # previous over-capacity call submitted, so its prefetch is quiesced
            # before the chunk-ahead ensures run.
            self._consume_prefetch_ticket(call_active)

        if not overlap or len(chunks) == 1:
            outputs = []
            for s, e in chunks:
                idx = idx_sorted[s:e]
                out = self._project_triplet(
                    x_sorted[s:e],
                    idx,
                    idx_host[s:e],
                    sorted_indices=True,
                )
                mx.eval(out)
                outputs.append(out)
            out = mx.concatenate(outputs, axis=0)
            out = _scatter_unsort(out, inv_order, indices.shape)
            # Every chunk's output is evaluated above, so the pool is quiesced;
            # submit the next-chunk prefetch for this layer's next call.
            if _PREFILL_PREFETCH:
                self._submit_prefetch_ticket(call_active)
            return out.squeeze(-2)

        chunk_sets = [
            {int(e) for e in idx_host[s:e].tolist()} for s, e in chunks
        ]
        pools = self._projection_pools()

        def _ensure_ahead(active_set, protect_set):
            # fence=False: a worker thread cannot fence stream 0 (thread_local
            # streams). The targeted main-thread wait below replaces it.
            return [
                _PROJECTION_LOAD_EXECUTOR.submit(
                    pool.ensure, active_set, protect=protect_set, fence=False)
                for pool in pools
            ]

        # chunk 0 loads up front (nothing to overlap with yet)
        for future in _ensure_ahead(chunk_sets[0], set()):
            future.result()

        # Loop invariants (the correctness story of the overlap):
        #  - the ahead ensure is the only pool mutator and is fully awaited
        #    before main touches the pools again (remap in the next build),
        #    so pool state is quiesced whenever main reads it;
        #  - ensure-ahead(i+1) protects chunk i, so victims are chunks <= i-1;
        #  - before submitting an ensure-ahead, main waits mx.eval(out_{i-1})
        #    (a targeted drain of every possible victim reader, chunks <=
        #    i-1), while chunk i keeps executing (kicked, never waited);
        #  - half-capacity chunks guarantee chunk i and i+1 coexist.
        outputs = []
        for i, (s, e) in enumerate(chunks):
            out = self._project_triplet(
                x_sorted[s:e],
                idx_sorted[s:e],
                idx_host[s:e],
                sorted_indices=True,
                preensured=True,  # loaded by the previous iteration's ahead
            )
            _kick_eval(out)  # chunk i executes while chunk i+1's IO runs
            outputs.append(out)
            if i + 1 < len(chunks):
                if i >= 1:
                    mx.eval(outputs[i - 1])  # victims' readers are done
                t0 = time.perf_counter()
                for future in _ensure_ahead(chunk_sets[i + 1], chunk_sets[i]):
                    future.result()
                self.projection_load_wait_seconds += time.perf_counter() - t0
        # Cross-chunk prefetch submit, under the loop's own targeted-drain
        # invariant extended across calls: a pool mutation that can evict
        # must never race a possible reader of a victim slot. The loop's
        # evals drained readers only through chunk n-3, and the penultimate
        # chunk's output is merely kicked, so drain it here before handing
        # the pool to the background prefetch; without this wait the
        # prefetch can overwrite the penultimate chunk's slots while its
        # gather kernels still execute (measured as nondeterministic
        # knife-edge token flips across processes at the 64 GB budgets).
        # The final chunk stays async: its set rides as the explicit
        # protect, so its slots are never victims.
        if _PREFILL_PREFETCH:
            if len(outputs) >= 2:
                mx.eval(outputs[-2])
            self._submit_prefetch_ticket(call_active, protect=chunk_sets[-1])
        out = mx.concatenate(outputs, axis=0)
        out = _scatter_unsort(out, inv_order, indices.shape)
        return out.squeeze(-2)

    def _call_direct(
        self,
        x,
        indices,
        *,
        load_ticket: _ProjectionLoadTicket | None = None,
        idx_host_flat=None,
    ) -> mx.array:
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))
        do_sort = _should_sort_routed_indices(indices)
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        if idx_host_flat is not None and not do_sort:
            idx_host = idx_host_flat
        else:
            idx_host = np.asarray(idx).reshape(-1)
        x = self._project_triplet(
            x,
            idx,
            idx_host,
            sorted_indices=do_sort,
            load_ticket=load_ticket,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)

    # ---- pipelined decode ------------------------------------

    def _pipe_bufs(self, K: int):
        """Persistent per-layer slot-id buffers (gate/up share, down own).

        The builder wires these into the routed graph before the slot values
        exist; the worker writes the values in place (memoryview, the same
        mechanism the pools use for packed/norms) before committing the layer,
        so kernels always execute against post-ensure slots."""
        bufs = self._pipe_buf_cache.get(K)
        if bufs is None:
            gate_buf = mx.array(np.zeros(K, dtype=np.uint32))
            down_buf = mx.array(np.zeros(K, dtype=np.uint32))
            mx.eval(gate_buf, down_buf)
            bufs = (
                gate_buf, memoryview(gate_buf).cast("B"),
                down_buf, memoryview(down_buf).cast("B"),
            )
            self._pipe_buf_cache[K] = bufs
        return bufs

    def _get_pipe_island(self, K: int):
        """Compiled routed-MLP closure taking slot IDS directly (no table
        gather): the pipelined path resolves slots on the worker, host-side."""
        island = self._pipe_island_cache.get(K)
        if island is not None:
            return island
        from jang_tools.turboquant.fused_gate_up_kernel import (
            make_fused_gate_up_swiglu_decode,
        )
        from jang_tools.turboquant.gather_tq_kernel import (
            make_gather_tq_decode_per_row,
        )
        from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal

        in_f = self.gate_proj.in_features
        out_f = self.gate_proj.out_features
        swiglu_limit = getattr(self.activation, "swiglu_limit", 0.0) or 0.0
        fused_gu = make_fused_gate_up_swiglu_decode(
            in_f, out_f, self.gate_proj.bits, K, swiglu_limit=swiglu_limit)
        gather_dn = make_gather_tq_decode_per_row(
            out_f, in_f, self.down_proj.bits, K)

        def _island(x_flat, slot_g, slot_d,
                    pg, ng, pu, nu, pd, nd, cb_g, cb_d, s_in, s_dn):
            x_rot = hadamard_rotate_metal(x_flat, s_in)
            x_act = fused_gu(x_rot, pg, ng, pu, nu, cb_g, slot_g)
            x_act = x_act.astype(mx.float16).astype(mx.float32)
            x_act_rot = hadamard_rotate_metal(x_act, s_dn)
            return gather_dn(x_act_rot, pd, nd, cb_d, slot_d)

        island = mx.compile(_island)
        self._pipe_island_cache[K] = island
        return island

    def build_pipelined(self, x, idx, *, event_gate=None) -> mx.array:
        """Build the routed MLP graph without any host read (builder thread).

        Output values are only correct once the worker has run ensure() and
        written the slot-id buffers for this layer: in v3 that ordering is
        commit-after-publish; in v4 `event_gate=(module, token, seq)` encodes
        an in-stream MTLSharedEvent wait in front of the island instead, the
        token input pins the encode order strictly after the ring export."""
        K = int(idx.shape[-1])
        gate_buf, _gv, down_buf, _dv = self._pipe_bufs(K)
        self.pipelined_layers += 1
        x4 = mx.expand_dims(x, (-2, -3))
        if event_gate is not None:
            gate_mod, token, seq = event_gate
            x4 = gate_mod.gate(x4, token, seq)
        if self._all_mxfp4:
            try:
                kernel = self._get_mxfp4_kernel(K)
            except RuntimeError:
                kernel = None
            if kernel is not None:
                self.compiled_island_calls += 1
                self.fused_gate_up_calls += 1
                x_flat = x4.reshape(-1, self.gate_proj.in_features).astype(mx.float32)
                y = kernel(
                    x_flat,
                    self.gate_proj.pool.packed,
                    self.gate_proj.pool.scales,
                    self.up_proj.pool.packed,
                    self.up_proj.pool.scales,
                    self.down_proj.pool.packed,
                    self.down_proj.pool.scales,
                    gate_buf,
                    down_buf,
                )
                out = y.reshape(*idx.shape[:-1], K, 1, self.down_proj.out_features)
                if out.dtype != x.dtype:
                    out = out.astype(x.dtype)
                return out.squeeze(-2)
        if self._fused_gate_up and self._all_tq:
            self.compiled_island_calls += 1
            self.fused_gate_up_calls += 1
            island = self._get_pipe_island(K)
            x_flat = x4.reshape(-1, self.gate_proj.in_features).astype(mx.float32)
            y = island(
                x_flat, gate_buf, down_buf,
                self.gate_proj.pool.packed, self.gate_proj.pool.norms,
                self.up_proj.pool.packed, self.up_proj.pool.norms,
                self.down_proj.pool.packed, self.down_proj.pool.norms,
                self.gate_proj.codebook, self.down_proj.codebook,
                self.gate_proj.signs, self.down_proj.signs,
            )
            out = y.reshape(*idx.shape[:-1], K, 1, self.down_proj.out_features)
            if out.dtype != x.dtype:
                out = out.astype(x.dtype)
            return out.squeeze(-2)
        # separate-kernel build (non-matching codebooks): gate/up pools share
        # slot assignment (pinned by test), down uses its own buffer
        gate_idx = gate_buf.reshape(idx.shape)
        down_idx = down_buf.reshape(idx.shape)
        if self._combined_gate_up_kquant:
            x_gate, x_up = self.gate_proj.matmul_gate_up_slots(
                x4,
                gate_idx,
                sorted_indices=False,
            )
            out = self.down_proj.matmul_slots(
                self.activation(x_up, x_gate), down_idx, sorted_indices=False)
            return out.squeeze(-2)
        x_up = self.up_proj.matmul_slots(x4, gate_idx, sorted_indices=False)
        x_gate = self.gate_proj.matmul_slots(x4, gate_idx, sorted_indices=False)
        out = self.down_proj.matmul_slots(
            self.activation(x_up, x_gate), down_idx, sorted_indices=False)
        return out.squeeze(-2)

    def pipelined_decode_fused_engaged(self) -> bool:
        """Engagement check for the ring-path fused decode routed matvec:
        the family's static eligibility (`_decode_routed_fused_ready`) plus
        the ring-scoped kill switch. No residency or slot-table condition:
        the worker-published slot-id buffers already carry valid resident
        rows for the token's experts. A False verdict keeps the unfused
        ring composition, the fail-closed direction."""
        if not _DECODE_RING_FUSED:
            return False
        return self._decode_routed_fused_ready()

    def build_pipelined_fused(self, x, idx, scores, *, event_gate=None):
        """Fused decode routed MLP on the ring path (builder thread).

        Runs the same two-dispatch matvec family the full-resident
        certificate route runs (gather_qmv_pair_swiglu with the route
        weights baked in, gather_qmv_expert_sum with the cross-expert sum
        in-kernel), consuming the worker-published slot-id buffers instead
        of router expert ids. Entry order of both id buffers is the router
        order the worker preserves and the route weights ride in the same
        order, so per-token math is identical to
        `build_barrier_free_decode_fused` on the same inputs; only the
        index source differs. The caller applies no route-weighted sum.
        Output values are only correct once the worker has run ensure()
        and written the slot-id buffers for this layer, the same publish
        contract as `build_pipelined` (v3 commit-after-publish, or the
        v4 `event_gate=(module, token, seq)` in-stream wait)."""
        import mlx_kquant as kq

        K = int(idx.shape[-1])
        gate_buf, _gv, down_buf, _dv = self._pipe_bufs(K)
        self.pipelined_layers += 1
        self.pipelined_decode_fused_calls += 1
        x_flat = x.reshape(1, self.gate_proj.in_features)
        if event_gate is not None:
            gate_mod, token, seq = event_gate
            x_flat = gate_mod.gate(x_flat, token, seq)
        weights = scores.reshape(-1).astype(mx.float32)
        swiglu_limit = getattr(self.activation, "swiglu_limit", 0.0) or 0.0
        mid = kq.gather_qmv_pair_swiglu(
            x_flat,
            self.gate_proj.pool.weight,
            self.gate_proj.pool.scales,
            self.gate_proj.kquant_type,
            gate_buf,
            weights,
            self.gate_proj.gate_out_features,
            swiglu_limit,
        )
        y = kq.gather_qmv_expert_sum(
            mid,
            self.down_proj.pool.weight,
            self.down_proj.pool.scales,
            self.down_proj.kquant_type,
            down_buf,
        )
        return y.reshape(*x.shape[:-1], self.down_proj.out_features)

    def export_inds(self, inds, seq: int):
        """Build the GPU-side export of routed ids + seq into this layer's
        persistent ring buffer. Returns the token array; `mx.async_eval` of
        the token commits everything through this layer's router plus the
        export itself."""
        K = int(inds.shape[-1])
        if K > 64:
            raise ValueError(
                f"ring export supports top_k <= 64 (threadgroup staging), "
                f"got {K}")
        if self._ring_buf is None:
            self._ring_buf = mx.array(np.zeros(8 + K, dtype=np.uint32))
            mx.eval(self._ring_buf)
            self._ring_np = np.frombuffer(
                memoryview(self._ring_buf).cast("B"), dtype=np.uint32)
        target = mx.array(np.array([seq], dtype=np.uint32))
        token, = _get_export_kernel()(
            inputs=[inds.reshape(-1), self._ring_buf, target],
            output_shapes=[(1,)],
            output_dtypes=[mx.uint32],
            grid=(K, 1, 1),
            threadgroup=(K, 1, 1),
        )
        return token

    def export_pred(self, pred_ids, seq: int):
        """Export the lookahead's predicted ids into the prediction ring
        (same kernel/protocol as export_inds, separate buffer). Returns the
        export token; it rides the same per-layer kick as the island."""
        n = int(pred_ids.shape[-1])
        if self._pred_ring_buf is None:
            self._pred_ring_buf = mx.array(np.zeros(8 + n, dtype=np.uint32))
            mx.eval(self._pred_ring_buf)
            self._pred_ring_np = np.frombuffer(
                memoryview(self._pred_ring_buf).cast("B"), dtype=np.uint32)
        target = mx.array(np.array([seq], dtype=np.uint32))
        token, = _get_export_kernel()(
            inputs=[pred_ids.reshape(-1), self._pred_ring_buf, target],
            output_shapes=[(1,)],
            output_dtypes=[mx.uint32],
            grid=(n, 1, 1),
            threadgroup=(n, 1, 1),
        )
        self.lookahead_exports += 1
        return token

    def _read_pred_ring(self, seq: int, n: int, deadline_s: float = 0.05):
        """Seqlock+checksum read of the prediction ring; None on timeout
        (prefetch is best-effort, a missed read skips one prefetch)."""
        ring = self._pred_ring_np
        if ring is None:
            return None
        deadline = time.perf_counter() + deadline_s
        while True:
            if int(ring[0]) == seq:
                ids = ring[8:8 + n].copy()
                if (int(ring[0]) == seq
                        and int(ring[1]) == _ring_checksum(ids, seq)):
                    return ids
            if time.perf_counter() > deadline:
                self.lookahead_ring_misses += 1
                return None
            time.sleep(0)

    def _maybe_lookahead(self, seq: int) -> None:
        """Worker-side: hand both the prediction-ring read and the prefetch
        to the lookahead executor. The ordered pipeline worker must never
        block on the prediction ring: its progress is what signals the
        gates the GPU (and thus the export) may be queued behind. The
        submission is dropped when the executor already holds
        `_LOOKAHEAD_MAX_PENDING` tasks (see the load-shedding note on the
        executor): stale queued speculation is worthless and its backlog
        outlives the request."""
        if self.lookahead_w is None or self.lookahead_target is None:
            return
        with _LOOKAHEAD_PENDING_LOCK:
            if _LOOKAHEAD_PENDING[0] >= _LOOKAHEAD_MAX_PENDING:
                self.lookahead_dropped += 1
                return
            _LOOKAHEAD_PENDING[0] += 1
        _lookahead_executor().submit(self._lookahead_task, seq)

    def _lookahead_task(self, seq: int) -> None:
        try:
            ids = self._read_pred_ring(seq, 16)
            if ids is not None:
                self.lookahead_target._prefetch_pools(ids)
        finally:
            with _LOOKAHEAD_PENDING_LOCK:
                _LOOKAHEAD_PENDING[0] -= 1

    def _prefetch_pools(self, pred_ids) -> None:
        """Executor-side: load predicted experts into the layer's
        dedicated spare slots via the atomic trio placement (same spare
        index across gate/up/down, the islands' slot-map lockstep holds by
        construction), round-robin over the spare ring, never evicting the
        live LFU pool. The pools are deduplicated: a combined K-quant
        gate/up projection shares one physical pool behind both aliases,
        and the trio placement would otherwise pread the same combined
        row twice per placement."""
        from moespresso.runtime.expert_slot_pool import place_spare_trio

        try:
            gate = self.gate_proj.pool
            if gate.spare_slots <= 0:
                return
            pools = self._unique_projection_pools(lockstep=True)
            for expert in pred_ids:
                expert = int(expert)
                if expert in gate._slot_of:
                    continue
                spare = self._spare_rr % gate.spare_slots
                self._spare_rr += 1
                if place_spare_trio(pools, expert, spare):
                    self.lookahead_prefetch_loads += 1
        except Exception:
            self.lookahead_errors += 1  # speculative path: never fail decode

    def ring_install(self, seq: int, K: int, gate_mod=None) -> None:
        """Worker-side per-layer step, zero MLX calls on the read path:
        seqlock-poll the ring for this layer's seq, read the expert ids from
        raw memory, then ensure() the misses and publish the slot-id buffers
        in place (same as publish_slots, without the np.asarray).

        v4: when `gate_mod` is given, the layer's event is signaled with
        `seq` in a finally block, always, even on error (poison): a routed
        island must never wait forever on a dead worker; the error itself
        re-raises and surfaces at the once-per-token future drain."""
        if gate_mod is not None:
            try:
                self._ring_install_body(seq, K)
            finally:
                gate_mod.signal_event(seq)
            self._maybe_lookahead(seq)
            return
        self._ring_install_body(seq, K)
        self._maybe_lookahead(seq)

    def _ring_install_body(self, seq: int, K: int) -> None:
        ring = self._ring_np
        t0 = time.perf_counter()
        deadline = t0 + _RING_TIMEOUT
        while True:
            if int(ring[0]) == seq:
                ids = ring[8:8 + K].copy()
                checksum = int(ring[1])
                # seqlock + checksum: seq stable around the
                # id reads and the GPU-computed checksum matches the ids+seq
                # we read; a stale or torn id snapshot cannot pass both.
                if int(ring[0]) == seq and checksum == _ring_checksum(ids, seq):
                    break
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"ring seq {seq} not observed within {_RING_TIMEOUT}s "
                    f"(layer={self.gate_proj.pool.layer}); GPU export never "
                    "became host-visible")
            # GIL-friendly poll: a pure busy-spin would hold the GIL for up
            # to the 5 ms switch interval and starve the builder thread.
            time.sleep(0)
        self.pipeline_read_seconds += time.perf_counter() - t0
        idx_host = ids
        self._last_active = {int(e) for e in ids.tolist()}
        if _ROUTE_TRACE is not None:
            _ROUTE_TRACE.append(
                ("decode", seq, self.gate_proj.pool.layer, ids.tolist()))
        active = {int(e) for e in idx_host.tolist()}
        self.seen_experts.update(active)
        self.decode_seen_experts.update(active)
        self._ensure_projection_pools(active)
        _gb, gate_view, _db, down_view = self._pipe_bufs(K)
        gate_view[:] = np.fromiter(
            (self.gate_proj.pool._slot_of[e] for e in idx_host),
            dtype=np.uint32, count=K).tobytes()
        down_view[:] = np.fromiter(
            (self.down_proj.pool._slot_of[e] for e in idx_host),
            dtype=np.uint32, count=K).tobytes()

    def publish_slots(self, inds) -> None:
        """Synchronous slot publication: read `inds` on the host, load the
        misses, and write the slot-id buffers `build_pipelined` reads. The
        serve path does this on the ring worker (`ring_install`); this method
        drives `build_pipelined` directly, for tests and probes."""
        t0 = time.perf_counter()
        idx_host = np.asarray(inds).reshape(-1)
        self.pipeline_read_seconds += time.perf_counter() - t0
        active = {int(e) for e in idx_host.tolist()}
        self.seen_experts.update(active)
        self.decode_seen_experts.update(active)
        self._ensure_projection_pools(active)
        K = idx_host.shape[0]
        _gb, gate_view, _db, down_view = self._pipe_bufs(K)
        gate_slots = np.fromiter(
            (self.gate_proj.pool._slot_of[e] for e in idx_host),
            dtype=np.uint32, count=K)
        down_slots = np.fromiter(
            (self.down_proj.pool._slot_of[e] for e in idx_host),
            dtype=np.uint32, count=K)
        gate_view[:] = gate_slots.tobytes()
        down_view[:] = down_slots.tobytes()


# Ring-decode cross-layer state (decode is single-threaded on the builder
# side). _PIPE_PREV holds the previous MoE layer's worker future: the next
# layer waits it before committing, which is the moment the previous layer's
# routed graph gets committed, guaranteeing publish(L) precedes
# commit-of-routed(L). The last MoE layer drains it synchronously.
_PIPE_PREV: list = []

# v3 ring-decode shares the ordering discipline; a monotonically increasing
# sequence number distinguishes layer-steps across tokens in the ring buffers
# and doubles as the MTLSharedEvent value in v4 (Metal requires monotonic
# nondecreasing signal values; one global counter provides that).
_RING_SEQ = [0]

# v4 gate-decode worker futures for the once-per-token error drain at the
# last MoE layer (workers signal the gates independently; the drain exists
# so worker exceptions surface on the builder thread every token).
_GATE_PENDING: list = []


class PooledSparseMoeBlock(nn.Module):
    """Qwen3Next sparse MoE block that overlaps routed misses with shared expert.

    The math mirrors mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock. The only
    scheduling change is decode-only: once router indices are known, start SSD
    loads for missing routed experts, then force the resident shared expert while
    those reads are in flight. The existing `PooledSwitchGLU` consumes the load
    ticket and waits for the unresolved tail before routed matmul.

    On ring decode the builder thread never reads indices; one ordered worker
    does read+ensure+publish per layer while commits stay on main.
    `pipeline_is_last` is set at install time on the deepest MoE layer.
    """

    pipeline_is_last: bool = False

    def __init__(self, original):
        super().__init__()
        self.gate = original.gate
        self.switch_mlp = original.switch_mlp
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.norm_topk_prob = original.norm_topk_prob
        self.num_experts = original.num_experts
        self.top_k = original.top_k
        self.sharding_group = getattr(original, "sharding_group", None)

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            from mlx.nn.layers.distributed import sum_gradients
            x = sum_gradients(self.sharding_group)(x)

        # Study capture: decode-only capture of the router input hidden state
        # (the residual-stream view each layer's router actually sees). One
        # host sync per layer per step, study runs only, gated twice.
        if (_ROUTE_TRACE is not None and _ROUTE_TRACE_HIDDEN
                and _token_layers(x) == 1):
            _ROUTE_TRACE.append((
                "hidden",
                self.switch_mlp.gate_proj.pool.layer,
                np.asarray(x).reshape(-1).astype(np.float16),
            ))

        # Decode-only block wall time (host side; includes the blocking index
        # sync + IO wait + graph building). The denominator for phase shares.
        block_t0 = time.perf_counter() if _token_layers(x) == 1 else None

        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # Barrier-free full-resident decode: when the certificate holds every
        # projection pool is full-resident, so the routed ids never leave the
        # device and the layer emits no ring export, no event gate, no worker
        # submit, and no per-layer block-exit kick. The token graph queues
        # lazily; the flush knob commits after every _DECODE_FLUSH_LAYERS
        # layers, so the forty-layer decode builds one lazy graph the way the
        # resident runtime does instead of forty async_eval flushes. The routed
        # math is the same separate-kernel combined gate/up gather, activation,
        # and down gather that build_pipelined emits, so this route is
        # bit-identical to the ring path; only the index source (router ids on
        # device instead of worker-published slot buffers) and the scheduling
        # change. Route tracing needs the host read this route removes, so
        # study runs keep the ring path.
        switch = self.switch_mlp
        barrier_free_ready = getattr(
            switch, "_barrier_free_decode_ready", None)
        if (
            _QWEN_DECODE_SCHED
            and block_t0 is not None
            and not self.training
            and _ROUTE_TRACE is None
            and barrier_free_ready is not None
            and barrier_free_ready()
        ):
            bf_inds = inds.astype(mx.uint32)
            y = switch.build_barrier_free_decode(x, bf_inds)
            _record_routed_weighted_sum(
                switch, scores, out_features=int(y.shape[-1]))
            y = (y * scores[..., None]).sum(axis=-2)
            shared_y = self.shared_expert(x)
            y = y + mx.sigmoid(self.shared_expert_gate(x)) * shared_y
            if _PIPE_PREV or _GATE_PENDING:
                # Mixed-path session: an earlier partial-residency layer took
                # the ring path this token. Its worker must publish before any
                # commit that can reach its routed island (the ring ordering
                # contract), and its errors must still surface once per token,
                # so drain here. Both lists stay empty on a pure barrier-free
                # run and this branch never executes.
                while _PIPE_PREV:
                    _PIPE_PREV.pop(0).result()
                if self.pipeline_is_last:
                    pending, _GATE_PENDING[:] = list(_GATE_PENDING), []
                    for future in pending:
                        future.result()
            if (
                _DECODE_FLUSH_LAYERS > 0
                and (int(switch.gate_proj.pool.layer) + 1)
                % _DECODE_FLUSH_LAYERS == 0
            ):
                _kick_eval(y)
                switch.barrier_free_decode_flush_calls += 1
            switch.decode_moe_block_calls += 1
            switch.decode_moe_block_seconds += time.perf_counter() - block_t0
            return y

        if (
            _RING_DECODE
            and block_t0 is not None
            and not self.training
            and _ring_visibility_ok()
            and _gate_module() is not None
        ):
            # v4 gate-decode: main never waits per layer at all. The
            # island sits behind an in-stream event wait (encoded after the
            # ring export via the token dependency); the worker signals after
            # ensure+publish. Commit the whole layer immediately and move on;
            # worker errors surface at the once-per-token future drain.
            gate_mod = _gate_module()
            switch = self.switch_mlp
            K = int(inds.shape[-1])
            _RING_SEQ[0] += 1
            seq = _RING_SEQ[0]
            token = switch.export_inds(inds, seq)
            # Cross-layer lookahead: run layer L+Delta's router on
            # this layer's input hidden, export top-16 predicted experts via
            # the prediction ring. Pre-gate (depends only on x), so the GPU
            # can execute it immediately; the worker hands the prefetch to
            # its own executor. Measured offline: catches ~59% of real
            # decode misses at Delta=4 with ~10 ms of lead.
            la_token = None
            if switch.lookahead_w is not None:
                la_logits = (
                    x.reshape(-1, switch.gate_proj.in_features)
                    .astype(switch.lookahead_w.dtype)
                    @ switch.lookahead_w.T
                ).reshape(-1)
                la_top = mx.argpartition(la_logits, kth=-16)[-16:].astype(
                    mx.uint32)
                la_token = switch.export_pred(la_top, seq)
                # Order pin: without a dependency, the scheduler may encode
                # this export behind a later layer's gate wait, and nothing
                # signals that gate before the export is needed. Folding
                # la_token into the gate's token input pins both exports
                # strictly before the gate wait.
                token = token + la_token * 0
            y = switch.build_pipelined(
                x, inds, event_gate=(gate_mod, token, seq))
            _record_routed_weighted_sum(
                switch, scores, out_features=int(y.shape[-1]))
            y = (y * scores[..., None]).sum(axis=-2)
            shared_y = self.shared_expert(x)
            y = y + mx.sigmoid(self.shared_expert_gate(x)) * shared_y
            _kick_eval(y)
            switch.block_exit_kick_calls += 1
            _GATE_PENDING.append(_PIPELINE_EXECUTOR.submit(
                switch.ring_install, seq, K, gate_mod))
            if self.pipeline_is_last:
                t0 = time.perf_counter()
                pending, _GATE_PENDING[:] = list(_GATE_PENDING), []
                for future in pending:
                    future.result()  # error drain; gates already signaled
                switch.pipeline_join_seconds += time.perf_counter() - t0
            switch.decode_moe_block_calls += 1
            switch.decode_moe_block_seconds += time.perf_counter() - block_t0
            return y

        if (
            _RING_DECODE
            and block_t0 is not None
            and not self.training
            and _ring_visibility_ok()
        ):
            # v3: zero MLX on the worker, zero per-layer blocking MLX on main.
            switch = self.switch_mlp
            K = int(inds.shape[-1])
            _RING_SEQ[0] += 1
            seq = _RING_SEQ[0]
            token = switch.export_inds(inds, seq)
            y = switch.build_pipelined(x, inds)
            _record_routed_weighted_sum(
                switch, scores, out_features=int(y.shape[-1]))
            y = (y * scores[..., None]).sum(axis=-2)
            shared_y = self.shared_expert(x)
            y = y + mx.sigmoid(self.shared_expert_gate(x)) * shared_y
            t0 = time.perf_counter()
            while _PIPE_PREV:
                _PIPE_PREV.pop(0).result()  # publish(L-1) precedes the commit
            switch.pipeline_join_seconds += time.perf_counter() - t0
            # Commits L-1's routed graph (via attention_L), this layer's
            # attention+router chain, and the export, but not this layer's
            # routed island (token does not depend on it).
            _kick_eval(token)
            switch.block_exit_kick_calls += 1
            future = _PIPELINE_EXECUTOR.submit(switch.ring_install, seq, K)
            if self.pipeline_is_last:
                t1 = time.perf_counter()
                future.result()
                switch.pipeline_join_seconds += time.perf_counter() - t1
                _kick_eval(y)  # commit the last routed island post-publish
            else:
                _PIPE_PREV.append(future)
            switch.decode_moe_block_calls += 1
            switch.decode_moe_block_seconds += time.perf_counter() - block_t0
            return y

        # Committing the router chain early via async_eval(inds) and building
        # the shared graph before the host read moved ~0.4 ms/layer between
        # counters but left e2e flat (5.46 vs 5.48 tok/s): the read must wait
        # for the same GPU chain wherever it sits. Do not retry the reorder;
        # the round-trip cost itself is the native-scheduling target.
        # Whole-MoE overlap: start routed expert loads as soon as the router
        # indices are known, then compute the always-resident shared expert
        # while those reads are in flight.
        load_ticket = self.switch_mlp.begin_projection_load(inds)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        if load_ticket is not None and load_ticket.has_work:
            # Extending this kick to all-hit decode layers as well measured
            # flat (5.42 vs 5.48 tok/s): the routed graph builds fast enough
            # that the no-miss idle window is tiny once the block-exit kick
            # exists, so the gate stays miss-only. Prefill does not force the
            # eval: the intermediate can be large and the prefill path already
            # has different scheduling pressure.
            if _token_layers(x) == 1:
                t0 = time.perf_counter()
                _kick_eval(shared_y)
                self.switch_mlp.overlap_shared_eval_calls += 1
                self.switch_mlp.overlap_shared_eval_seconds += time.perf_counter() - t0
            else:
                self.switch_mlp.overlap_prefill_no_eval_calls += 1

        y = self.switch_mlp(x, inds, load_ticket=load_ticket)
        _record_routed_weighted_sum(
            self.switch_mlp, scores, out_features=int(y.shape[-1]))
        y = (y * scores[..., None]).sum(axis=-2)
        y = y + shared_y

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        if block_t0 is not None:
            # Block-exit kick: commit the finished block so the GPU runs layer
            # L's routed work while python builds layer L+1's graph.
            _kick_eval(y)
            self.switch_mlp.block_exit_kick_calls += 1
            self.switch_mlp.decode_moe_block_calls += 1
            self.switch_mlp.decode_moe_block_seconds += (
                time.perf_counter() - block_t0)

        return y


class PooledDeepseekV4MoEBlock(nn.Module):
    """DeepSeek-V4 MoE block over an SSD-backed pooled SwitchGLU.

    This mirrors jang_tools.dsv4.mlx_model.MoE: the DS4 gate already returns
    routed expert ids and scores, hash layers require input_ids, and the shared
    expert is added directly without a separate shared gate. Keep this separate
    from PooledSparseMoeBlock; the Qwen-style softmax router contract is not the
    DS4 contract.
    """

    pipeline_is_last: bool = False

    def __init__(self, original):
        super().__init__()
        self.gate = original.gate
        self.switch_mlp = original.switch_mlp
        self.shared_experts = original.shared_experts
        self.sharding_group = getattr(original, "sharding_group", None)

    def __call__(self, x: mx.array, input_ids=None) -> mx.array:
        if self.sharding_group is not None:
            from mlx.nn.layers.distributed import sum_gradients
            x = sum_gradients(self.sharding_group)(x)

        if (_ROUTE_TRACE is not None and _ROUTE_TRACE_HIDDEN
                and _token_layers(x) == 1):
            _ROUTE_TRACE.append((
                "hidden",
                self.switch_mlp.gate_proj.pool.layer,
                np.asarray(x).reshape(-1).astype(np.float16),
            ))

        block_t0 = time.perf_counter() if _token_layers(x) == 1 else None

        switch = self.switch_mlp
        gate_t0 = time.perf_counter() if block_t0 is not None else None
        inds, scores = self.gate(x, input_ids=input_ids)
        if gate_t0 is not None:
            _record_switch_seconds(
                switch, "router_gate_seconds", time.perf_counter() - gate_t0)
        inds = inds.astype(mx.uint32)
        supports_ring = all(hasattr(switch, name) for name in (
            "build_pipelined",
            "export_inds",
            "ring_install",
        ))

        # Barrier-free full-resident decode: the certificate holds, so the
        # routed ids never leave the device and the layer emits no ring
        # export, no event gate, no worker submit, and no per-layer kick.
        # The token graph queues lazily; the flush knob commits after every
        # _DECODE_FLUSH_LAYERS layers. Route tracing needs the host read
        # this route removes, so study runs keep the ring path.
        barrier_free_ready = getattr(
            switch, "_barrier_free_decode_ready", None)
        if (
            block_t0 is not None
            and not self.training
            and _ROUTE_TRACE is None
            and barrier_free_ready is not None
            and barrier_free_ready()
        ):
            t0 = time.perf_counter()
            # Fused decode routed matvec family: route weights bake into the
            # pair+SwiGLU intermediate and the down kernel sums the experts,
            # so this arm emits no route-weighted sum at all (its counters
            # staying at zero is the engagement evidence, next to
            # decode_routed_fused_calls).
            fused_engaged = getattr(
                switch, "decode_routed_fused_engaged", None)
            if fused_engaged is not None and fused_engaged():
                y = switch.build_barrier_free_decode_fused(x, inds, scores)
            else:
                y = switch.build_barrier_free_decode(x, inds)
                _record_routed_weighted_sum(
                    switch, scores, out_features=int(y.shape[-1]))
                y = _deepseek_v4_weighted_sum(y, scores).reshape(x.shape)
            _record_switch_seconds(
                switch, "routed_build_seconds", time.perf_counter() - t0)
            t0 = time.perf_counter()
            y = y + self.shared_experts(x)
            _record_switch_seconds(
                switch,
                "shared_experts_build_seconds",
                time.perf_counter() - t0,
            )
            if _PIPE_PREV or _GATE_PENDING:
                # Mixed-path session: an earlier partial-residency layer took
                # the ring path this token. Its worker must publish before
                # any commit that can reach its routed island (the ring
                # ordering contract), and its errors must still surface once
                # per token, so drain here. Both lists stay empty on a pure
                # barrier-free run and this branch never executes.
                t0 = time.perf_counter()
                while _PIPE_PREV:
                    _PIPE_PREV.pop(0).result()
                if self.pipeline_is_last:
                    pending, _GATE_PENDING[:] = list(_GATE_PENDING), []
                    for future in pending:
                        future.result()
                switch.pipeline_join_seconds += time.perf_counter() - t0
            if (
                _DECODE_FLUSH_LAYERS > 0
                and (int(switch.gate_proj.pool.layer) + 1)
                % _DECODE_FLUSH_LAYERS == 0
            ):
                _kick_eval(y)
                switch.barrier_free_decode_flush_calls += 1
            switch.decode_moe_block_calls += 1
            switch.decode_moe_block_seconds += time.perf_counter() - block_t0
            return y

        if (
            supports_ring
            and _RING_DECODE
            and block_t0 is not None
            and not self.training
            and _ring_visibility_ok()
            and _gate_module() is not None
        ):
            gate_mod = _gate_module()
            K = int(inds.shape[-1])
            _RING_SEQ[0] += 1
            seq = _RING_SEQ[0]
            t0 = time.perf_counter()
            token = switch.export_inds(inds, seq)
            _record_switch_seconds(
                switch, "router_export_seconds", time.perf_counter() - t0)
            # Cross-layer lookahead: run layer L+Delta's router scoring on
            # this layer's input hidden and export the top-16 predicted
            # experts via the prediction ring. The DS4 selection form is the
            # monotone softplus transform plus a per-expert bias
            # (sqrt(log1p(exp(logits))) + bias); the bias reorders
            # candidates, so ranking raw logits would mispredict. Pre-gate
            # (depends only on x), and the token dependency pins both
            # exports strictly before the event-gate wait, mirroring the
            # Qwen block's order pin.
            if switch.lookahead_w is not None:
                la_logits = (
                    x.reshape(-1, switch.gate_proj.in_features)
                    .astype(switch.lookahead_w.dtype)
                    @ switch.lookahead_w.T
                ).reshape(-1)
                la_scores = mx.sqrt(
                    mx.log1p(mx.exp(la_logits.astype(mx.float32))))
                if switch.lookahead_b is not None:
                    la_scores = la_scores + switch.lookahead_b
                la_top = mx.argpartition(la_scores, kth=-16)[-16:].astype(
                    mx.uint32)
                la_token = switch.export_pred(la_top, seq)
                token = token + la_token * 0
            t0 = time.perf_counter()
            # Ring-path fused decode: the same matvec pair the full-resident
            # certificate route runs, over the worker-published slot ids, so
            # bounded residency serves the full-resident decode lattice. The
            # fused kernels bake the route weights and sum the experts, so
            # this arm emits no route-weighted sum (its counters staying at
            # zero is the engagement evidence, next to
            # pipelined_decode_fused_calls).
            ring_fused = getattr(
                switch, "pipelined_decode_fused_engaged", None)
            if ring_fused is not None and ring_fused():
                y = switch.build_pipelined_fused(
                    x,
                    inds,
                    scores,
                    event_gate=(gate_mod, token, seq),
                ).reshape(x.shape)
            else:
                y = switch.build_pipelined(
                    x,
                    inds,
                    event_gate=(gate_mod, token, seq),
                )
                _record_routed_weighted_sum(
                    switch, scores, out_features=int(y.shape[-1]))
                y = _deepseek_v4_weighted_sum(y, scores).reshape(x.shape)
            _record_switch_seconds(
                switch, "routed_build_seconds", time.perf_counter() - t0)
            t0 = time.perf_counter()
            y = y + self.shared_experts(x)
            _record_switch_seconds(
                switch,
                "shared_experts_build_seconds",
                time.perf_counter() - t0,
            )
            t0 = time.perf_counter()
            _kick_eval(y)
            _record_switch_seconds(
                switch, "block_exit_kick_seconds", time.perf_counter() - t0)
            switch.block_exit_kick_calls += 1
            _GATE_PENDING.append(_PIPELINE_EXECUTOR.submit(
                switch.ring_install, seq, K, gate_mod))
            if self.pipeline_is_last:
                t0 = time.perf_counter()
                pending, _GATE_PENDING[:] = list(_GATE_PENDING), []
                for future in pending:
                    future.result()
                switch.pipeline_join_seconds += time.perf_counter() - t0
            switch.decode_moe_block_calls += 1
            switch.decode_moe_block_seconds += time.perf_counter() - block_t0
            return y

        if (
            supports_ring
            and _RING_DECODE
            and block_t0 is not None
            and not self.training
            and _ring_visibility_ok()
        ):
            K = int(inds.shape[-1])
            _RING_SEQ[0] += 1
            seq = _RING_SEQ[0]
            t0 = time.perf_counter()
            token = switch.export_inds(inds, seq)
            _record_switch_seconds(
                switch, "router_export_seconds", time.perf_counter() - t0)
            t0 = time.perf_counter()
            # Ring-path fused decode, v3 scheduling: same unification as the
            # v4 branch above; publish-before-commit orders the kernels
            # after the worker's slot writes.
            ring_fused = getattr(
                switch, "pipelined_decode_fused_engaged", None)
            if ring_fused is not None and ring_fused():
                y = switch.build_pipelined_fused(
                    x, inds, scores).reshape(x.shape)
            else:
                y = switch.build_pipelined(x, inds)
                _record_routed_weighted_sum(
                    switch, scores, out_features=int(y.shape[-1]))
                y = _deepseek_v4_weighted_sum(y, scores).reshape(x.shape)
            _record_switch_seconds(
                switch, "routed_build_seconds", time.perf_counter() - t0)
            t0 = time.perf_counter()
            y = y + self.shared_experts(x)
            _record_switch_seconds(
                switch,
                "shared_experts_build_seconds",
                time.perf_counter() - t0,
            )
            t0 = time.perf_counter()
            while _PIPE_PREV:
                _PIPE_PREV.pop(0).result()
            switch.pipeline_join_seconds += time.perf_counter() - t0
            _kick_eval(token)
            switch.block_exit_kick_calls += 1
            future = _PIPELINE_EXECUTOR.submit(switch.ring_install, seq, K)
            if self.pipeline_is_last:
                t1 = time.perf_counter()
                future.result()
                switch.pipeline_join_seconds += time.perf_counter() - t1
                _kick_eval(y)
            else:
                _PIPE_PREV.append(future)
            switch.decode_moe_block_calls += 1
            switch.decode_moe_block_seconds += time.perf_counter() - block_t0
            return y

        # Whole-MoE overlap: start routed expert loads as soon as the router
        # indices are known, then compute the always-resident shared experts
        # while those reads are in flight.
        load_ticket = switch.begin_projection_load(inds)

        t0 = time.perf_counter() if block_t0 is not None else None
        shared_y = self.shared_experts(x)
        if t0 is not None:
            _record_switch_seconds(
                switch,
                "shared_experts_build_seconds",
                time.perf_counter() - t0,
            )
        if load_ticket is not None and load_ticket.has_work:
            if _token_layers(x) == 1:
                t0 = time.perf_counter()
                _kick_eval(shared_y)
                switch.overlap_shared_eval_calls += 1
                switch.overlap_shared_eval_seconds += time.perf_counter() - t0
            else:
                switch.overlap_prefill_no_eval_calls += 1

        weighted_output = getattr(switch, "weighted_output", None)
        t0 = time.perf_counter() if block_t0 is not None else None
        if callable(weighted_output):
            y = weighted_output(
                x,
                inds,
                scores,
                load_ticket=load_ticket,
            ).reshape(x.shape)
        else:
            y = switch(x, inds)
            y = _deepseek_v4_weighted_sum(y, scores).reshape(x.shape)
        if t0 is not None:
            _record_switch_seconds(
                switch, "routed_build_seconds", time.perf_counter() - t0)
        y = y + shared_y

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        if block_t0 is not None:
            # Block-exit kick: commit the finished block so the GPU runs layer
            # L's routed work while python builds layer L+1's graph.
            t0 = time.perf_counter()
            _kick_eval(y)
            _record_switch_seconds(
                switch,
                "block_exit_kick_seconds",
                time.perf_counter() - t0,
            )
            switch.block_exit_kick_calls += 1
            switch.decode_moe_block_calls += 1
            switch.decode_moe_block_seconds += (
                time.perf_counter() - block_t0)

        return y
