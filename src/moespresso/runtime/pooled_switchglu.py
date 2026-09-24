"""SwitchGLU over persistent SSD-backed expert pools.

The module owns the whole SwitchGLU seam (sort/gather, gate/up activation, down,
scatter) so it cannot be bypassed by JANG's class-level SwitchGLU monkeypatch.
Each projection reads selected experts from a persistent `ExpertSlotPool`; misses
are loaded directly into MLX buffers via `pread_into`.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import os
import time
from types import MethodType

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from moespresso.runtime.expert_index import ExpertIndex
from moespresso.runtime.expert_slot_pool import ExpertCapacityExceeded
from moespresso.runtime.expert_slot_pool import ExpertSlotPool
from moespresso.runtime.expert_slot_pool import grow_expert_slot_pools
from moespresso.runtime.expert_slot_pool import seed_hot_expert_slot_pools
from moespresso.runtime.pooled_load_batch import LoadBatch, submit_loads, submit_loads_and_wait
from moespresso.package.bundle import IQK_CODEC, KQUANT_CODEC, MXFP4_CODEC
from moespresso.package.iqk_format import IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1

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
    batch: LoadBatch
    started_at: float
    load_owner: object | None = None
    used: bool = False

    @property
    def has_work(self) -> bool:
        return bool(self.batch.futures)


@dataclass
class _PrefetchTicket:
    """Cross-chunk predictive prefetch handle stored on a layer's switch.

    `predicted` is the demand set of the prompt chunk that submitted it, warmed
    on the IO executor so the layer's next call finds those slots resident.
    `batch` completes when every pool's prefetch has published. The consumer
    awaits them before its chunk-ahead path touches the pools, then discards the
    ticket. A submitted set that does not match the consumer's actual demand is
    counted as a mismatch but still awaited, because the prefetch's bytes are
    already landing into slots the pool now owns; abandoning mid-flight would
    leave reserved-but-unpublished slots.
    """

    predicted: frozenset[int]
    batch: LoadBatch


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


class PooledIqkSwitchLinear(nn.Module):
    """An IQ_K routed projection backed by an ``ExpertSlotPool``."""

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
        if self.pool.codec != IQK_CODEC or self.pool.iqk is None:
            raise ValueError(
                f"{projection} declares codec {self.pool.codec!r}, expected 'iqk'")
        self.codec = IQK_CODEC
        self.bits = self.pool.bits
        self.member = self.pool.geometry.iqk_codec
        self.num_experts = self.pool.num_experts
        self.out_features = self.pool.geometry.out_features
        self.in_features = int(self.pool.geometry.in_features or 0)
        self.matmul_slot_calls = 0
        self.matmul_slot_elements = 0

    def __call__(self, x, indices, *, sorted_indices: bool = False):
        remapped = self.pool.remap(indices)
        return self.matmul_slots(x, remapped, sorted_indices=sorted_indices)

    def matmul_slots(self, x, remapped_indices, *, sorted_indices: bool = False):
        self.matmul_slot_calls += 1
        self.matmul_slot_elements += int(remapped_indices.size)
        if sorted_indices:
            return self.pool.iqk(x, remapped_indices, sorted_indices=True)
        return self.pool.iqk.gemv(x, remapped_indices)

    def sorted_matmul_range(self, x, remapped_indices, start: int, rows: int):
        self.matmul_slot_calls += 1
        self.matmul_slot_elements += int(remapped_indices.size)
        return self.pool.iqk.sorted_matmul_range(
            x,
            remapped_indices,
            start,
            rows,
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
            not sorted_indices
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
        self.hidden_size = int(gate_proj.in_features)
        self.intermediate_size = int(gate_proj.out_features)
        self.num_experts = int(gate_proj.num_experts)
        projection_geometry = (
            int(up_proj.in_features),
            int(up_proj.out_features),
            int(up_proj.num_experts),
            int(down_proj.in_features),
            int(down_proj.out_features),
            int(down_proj.num_experts),
        )
        expected_geometry = (
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.intermediate_size,
            self.hidden_size,
            self.num_experts,
        )
        if projection_geometry != expected_geometry:
            raise ValueError("pooled SwitchGLU projection geometry is inconsistent")
        self._all_mxfp4 = (
            getattr(gate_proj, "codec", None) == MXFP4_CODEC
            and getattr(up_proj, "codec", None) == MXFP4_CODEC
            and getattr(down_proj, "codec", None) == MXFP4_CODEC
        )
        self._all_iqk = (
            getattr(gate_proj, "codec", None) == IQK_CODEC
            and getattr(up_proj, "codec", None) == IQK_CODEC
            and getattr(down_proj, "codec", None) == IQK_CODEC
        )
        # The cross-call prefetch has measured wins for K-quant prompt chunks,
        # but an IQ_K miss also splits each stored row into kernel streams. On
        # a terminal one-chunk prefill that work cannot be consumed and instead
        # evicts the demand residency needed by decode. Keep the proven
        # K-quant policy; IQ_K stays demand-driven until a multi-chunk served
        # arm establishes a codec-specific win.
        self._prefill_prefetch_enabled = not self._all_iqk
        self.layer = int(gate_proj.pool.layer)
        self.iqk_ordinal = 0
        self.members = (
            {
                "gate_proj": gate_proj.member,
                "up_proj": up_proj.member,
                "down_proj": down_proj.member,
            }
            if self._all_iqk
            else {}
        )
        self._combined_gate_up_kquant = (
            isinstance(gate_proj, PooledCombinedGateUpKQuantLinear)
            and getattr(up_proj, "_parent", None) is gate_proj
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
        self.gemv_calls = 0
        self.gemv_pairs = 0
        self.sorted_prefill_calls = 0
        self.sorted_prefill_pairs = 0
        self.sorted_nsplit_calls = 0
        self.sorted_nsplit_parts = 0
        self.iqk_decode_flush_calls = 0
        self.iqk_verify_flush_calls = 0
        self.decode_routed_fused_calls = 0
        self.pipelined_decode_fused_calls = 0
        # One-shot eligibility verdict for the barrier-free prefill route
        # (None until the first bulk-prefill-shaped call decides it).
        self._barrier_free_ready_cached: bool | None = None
        # One-shot eligibility verdict for the barrier-free decode route
        # (None until the first decode-shaped call decides it).
        self._barrier_free_decode_ready_cached: bool | None = None
        self._iqk_decode_identity_cached: bool | None = None
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
        self.projection_sync_join_calls = 0
        self.projection_tracked_join_calls = 0
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
        # Cross-chunk predictive prefetch. One ticket
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
        self._mxfp4_kernel_cache: dict = {}
        # Pipelined builder state (see build_pipelined)
        self.pipelined_layers = 0
        self.pipeline_read_seconds = 0.0
        self.pipeline_join_seconds = 0.0
        self._pipe_buf_cache: dict = {}
        # v3 ring-export state (see _RING_DECODE)
        self._ring_buf = None
        self._ring_np = None
        self.seen_experts: set[int] = set()
        self.prefill_seen_experts: set[int] = set()
        self.decode_seen_experts: set[int] = set()

    def grow_capacity(self, capacity: int) -> None:
        def reset_route_certificates() -> None:
            self._barrier_free_ready_cached = None
            self._barrier_free_decode_ready_cached = None
            self._iqk_decode_identity_cached = None

        grow_expert_slot_pools(
            self._unique_projection_pools(lockstep=True),
            capacity,
            after_publish=reset_route_certificates,
        )

    def seed_hot_free_slots(self) -> int:
        return seed_hot_expert_slot_pools(
            self._unique_projection_pools(lockstep=True),
        )

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

    def _join_projection_loads(self, calls) -> None:
        """Run one immediate projection group with the safe join fallback."""
        if submit_loads_and_wait(_PROJECTION_LOAD_EXECUTOR, calls):
            self.projection_sync_join_calls += 1
        else:
            self.projection_tracked_join_calls += 1

    def _ensure_stream_major_projection_batches(
        self,
        active: set[int],
        *,
        protect: set[int],
        fence: bool,
    ) -> bool:
        """Load a sorted-prefill active set within the shared row-cache window.

        Stream-major copies can let one projection consume rows substantially
        faster than the other two. If the active set exceeds the shared cache,
        that producer can evict rows before the remaining projections consume
        them. Coordinating the existing projection ensures in cache-sized
        rounds preserves one row read per expert without changing pool
        publication or routed compute.

        This helper is deliberately limited to the Qwen stream-major package
        layout. Other layouts retain their established scheduling.
        """
        pools = self._projection_pools_lockstep()
        if len(pools) != 3:
            return False
        row_cache = pools[0].row_cache
        if (
            row_cache is None
            or row_cache.max_rows < 1
            or len(active) <= row_cache.max_rows
            or any(pool.row_cache is not row_cache for pool in pools[1:])
            or any(
                pool.geometry.layout != IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1
                for pool in pools
            )
        ):
            return False

        ordered = sorted(active)
        for start in range(0, len(ordered), row_cache.max_rows):
            batch = set(ordered[start:start + row_cache.max_rows])
            # The unbatched ensure protects its complete active set while it
            # chooses victims. Preserve that eligibility: a later sub-round's
            # resident expert must not be evicted by an earlier sub-round.
            batch_protect = set(protect) | (active - batch)
            try:
                self._join_projection_loads(
                    partial(pool.ensure, batch, protect=batch_protect, fence=fence)
                    for pool in pools
                )
            finally:
                # Submission and execution failures drain active writers before
                # returning. Cached rows no longer have projection consumers.
                for expert in batch:
                    row_cache.discard(expert)
        return True

    @staticmethod
    def _iqk_sorted_threshold() -> int:
        from moespresso.runtime.deepseek_v4.iqk_experts import (
            sorted_prefill_min_pairs,
        )

        return sorted_prefill_min_pairs()

    @staticmethod
    def _iqk_sorted_parts() -> int:
        from moespresso.runtime.deepseek_v4.iqk_experts import (
            sorted_prefill_nsplit,
        )

        return sorted_prefill_nsplit()

    def _record_iqk_route(self, pairs: int) -> bool:
        """Record and return the incumbent IQ_K sorted-route decision."""
        if pairs < self._iqk_sorted_threshold():
            self.gemv_calls += 1
            self.gemv_pairs += pairs
            return False
        self.sorted_prefill_calls += 1
        self.sorted_prefill_pairs += pairs
        parts = self._iqk_sorted_parts()
        if parts > 1:
            self.sorted_nsplit_calls += 1
            self.sorted_nsplit_parts = parts
        return True

    def commit_iqk_output(self, output, *, rows: int) -> bool:
        """Apply the incumbent IQ_K decode/verify commit cadence."""
        if not self._all_iqk:
            return False
        from moespresso.runtime.deepseek_v4.iqk_experts import (
            _VERIFY_FLUSH_MAX_ROWS,
            iqk_decode_flush_layers,
        )

        cadence = iqk_decode_flush_layers()
        if cadence < 1 or (self.iqk_ordinal + 1) % cadence:
            return False
        if rows == 1:
            mx.async_eval(output)
            self.iqk_decode_flush_calls += 1
            return True
        if rows <= _VERIFY_FLUSH_MAX_ROWS:
            mx.async_eval(output)
            self.iqk_verify_flush_calls += 1
            return True
        return False

    def _iqk_sorted_projection(self, projection, x_rows, slot_ids):
        """Project pairwise rows after sorting by the projection's slot ids."""
        rows = int(slot_ids.size)
        flat_slots = slot_ids.reshape(-1)
        order = mx.argsort(flat_slots)
        sorted_slots = flat_slots[order]
        operand = x_rows.reshape(rows, projection.in_features)[order]
        operand = operand.reshape(rows, 1, projection.in_features).astype(mx.float16)
        parts = self._iqk_sorted_parts()
        step = projection.out_features // parts
        out = mx.concatenate(
            [
                projection.sorted_matmul_range(
                    operand,
                    sorted_slots,
                    part * step,
                    step,
                )
                for part in range(parts)
            ],
            axis=-1,
        )
        return out.reshape(rows, projection.out_features)[mx.argsort(order)]

    def _iqk_sorted_triplet(self, x_rows, gate_slots, up_slots, down_slots):
        gate = self._iqk_sorted_projection(self.gate_proj, x_rows, gate_slots)
        up = self._iqk_sorted_projection(self.up_proj, x_rows, up_slots)
        activated = self.activation(up, gate)
        return self._iqk_sorted_projection(
            self.down_proj,
            activated,
            down_slots,
        )

    def _call_iqk_full_resident(self, x, indices) -> mx.array:
        """IQ_K compute with full-resident routing kept entirely on device."""
        pairs = int(indices.size)
        identity = all(
            pool.slot_table_is_identity()
            for pool in self._projection_pools_lockstep()
        )
        if identity:
            gate_slots = up_slots = down_slots = indices
        else:
            gate_slots = self.gate_proj.pool.remap_ondevice(indices)
            up_slots = self.up_proj.pool.remap_ondevice(indices)
            down_slots = self.down_proj.pool.remap_ondevice(indices)
        if pairs < self._iqk_sorted_threshold():
            x4 = mx.expand_dims(x, (-2, -3))
            up = self.up_proj.matmul_slots(x4, up_slots, sorted_indices=False)
            gate = self.gate_proj.matmul_slots(x4, gate_slots, sorted_indices=False)
            out = self.down_proj.matmul_slots(
                self.activation(up, gate),
                down_slots,
                sorted_indices=False,
            )
            return out.squeeze(-2)

        top_k = int(indices.shape[-1])
        flat = indices.reshape(-1)
        if identity:
            order = mx.argsort(flat)
            sorted_ids = flat[order]
            rows = int(flat.size)
            gathered = x.reshape(-1, self.gate_proj.in_features)[order // top_k]
            # Match IqkDeepseekV4SwitchGLU._call_sorted: the kernel-facing
            # activation is fp16 even when the trunk hidden state is bf16.
            # Leaving it bf16 promotes the large sorted matmuls to fp32,
            # increasing both prefill wall time and transient memory.
            xg = gathered.reshape(
                rows, 1, self.gate_proj.in_features).astype(mx.float16)
            parts = self._iqk_sorted_parts()
            step = self.gate_proj.out_features // parts
            up = mx.concatenate(
                [
                    self.up_proj.sorted_matmul_range(
                        xg, sorted_ids, part * step, step)
                    for part in range(parts)
                ],
                axis=-1,
            )
            gate = mx.concatenate(
                [
                    self.gate_proj.sorted_matmul_range(
                        xg, sorted_ids, part * step, step)
                    for part in range(parts)
                ],
                axis=-1,
            )
            activated = self.activation(up, gate)
            down_step = self.down_proj.out_features // parts
            down = mx.concatenate(
                [
                    self.down_proj.sorted_matmul_range(
                        activated, sorted_ids, part * down_step, down_step)
                    for part in range(parts)
                ],
                axis=-1,
            )
            out = down.reshape(rows, -1)[mx.argsort(order)]
        else:
            pair_rows = x.reshape(-1, self.gate_proj.in_features)[
                mx.arange(pairs) // top_k]
            out = self._iqk_sorted_triplet(
                pair_rows,
                gate_slots,
                up_slots,
                down_slots,
            )
        return mx.unflatten(out, 0, tuple(indices.shape))

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
                if getattr(pool, "_staging_owner", None) is not None:
                    raise RuntimeError("expert pool is owned by staged verification")
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

    def begin_projection_load(
        self,
        indices,
        *,
        load_owner=None,
    ) -> _ProjectionLoadTicket | None:
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
        if load_owner is None:
            batch = submit_loads(
                _PROJECTION_LOAD_EXECUTOR,
                (partial(pool.ensure, active) for pool in pools),
            )
        else:
            batch = load_owner.begin_load_batch(_PROJECTION_LOAD_EXECUTOR)
            for pool in pools:
                batch.submit(partial(pool.ensure, active))
        return _ProjectionLoadTicket(
            active=active,
            batch=batch,
            started_at=time.perf_counter(),
            load_owner=load_owner,
        )

    def _wait_projection_ticket(
        self,
        active: set[int],
        ticket: _ProjectionLoadTicket | None,
    ) -> bool:
        if ticket is None:
            return False
        if ticket.used:
            self.overlap_ticket_mismatch_calls += 1
            return False

        matches = ticket.active == active
        if not matches:
            self.overlap_ticket_mismatch_calls += 1

        ticket.used = True
        self.projection_load_wait_calls += 1
        self.projection_load_parallel_calls += 1
        self.overlap_load_wait_calls += 1
        wait_started = time.perf_counter()
        try:
            if ticket.load_owner is None:
                ticket.batch.wait()
            else:
                ticket.load_owner.drain()
        finally:
            done = time.perf_counter()
            wait_seconds = done - wait_started
            total_seconds = done - ticket.started_at
            self.projection_load_wait_seconds += wait_seconds
            self.overlap_load_wait_seconds += wait_seconds
            self.overlap_load_total_seconds += total_seconds
            self.overlap_load_hidden_seconds += max(0.0, total_seconds - wait_seconds)
        # A different demand cannot touch the pools until the old writers stop.
        return matches

    def _ensure_projection_pools(
        self,
        active: set[int],
        load_ticket: _ProjectionLoadTicket | None = None,
    ) -> None:
        if self._prefill_prefetch_enabled and self._prefetch_ticket is not None:
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
            self._join_projection_loads(partial(pool.ensure, active) for pool in pools)
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
            self._all_mxfp4
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
        # Unified sorted prefill: the partial-residency chunked path computes
        # each pre-ensured, expert-sorted chunk through the same fused sorted
        # kernels the full-resident barrier-free route runs, so the served
        # tokens match the full-residency rail at any capacity. Covers both
        # the large-chunk (segmented) and
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
        # this path needs no remapped index tensors at all. It serves shapes
        # or dependencies unsupported by the fused sorted path.
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
        self.remap_ondevice_calls += 1
        up_idx = self.up_proj.pool.remap_ondevice(idx)
        gate_idx = self.gate_proj.pool.remap_ondevice(idx)
        down_idx = self.down_proj.pool.remap_ondevice(idx)

        if self._all_iqk and sorted_indices:
            rows = int(idx.size)
            out = self._iqk_sorted_triplet(
                x.reshape(rows, self.gate_proj.in_features),
                gate_idx,
                up_idx,
                down_idx,
            )
            return out.reshape(*idx_shape, 1, self.down_proj.out_features)

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

        Static preconditions only: the combined K-quant
        gate/up pool, the K-quant down pool, and the installed
        gather_qmm_sorted_swiglu / gather_qmm_sorted kernels. A False verdict
        falls back to the pre-unification chunked compute (segmented f32 GEMM
        or general gather plus a separate activation), the fail-closed
        direction. Decided per call rather than cached because it depends only
        on process-stable facts and the check is a handful of attribute reads.
        """
        if not self._combined_gate_up_kquant:
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
        if self._all_iqk:
            return bool(indices.size)
        return bool(
            indices.size >= 64
            and indices.size >= _SEGMENTED_PREFILL_MIN_ROWS
        )

    def _barrier_free_ready(self) -> bool:
        """Fail-closed eligibility check for barrier-free prefill.

        A successful verdict is stable because a fully resident full-capacity
        pool cannot evict. A negative verdict at smaller capacity is stable
        until growth resets the cache. At full capacity, however, an explicit
        no-prewarm setting can leave the pool cold; revisit that negative
        verdict so demand loading can eventually earn the optimized route."""
        ready = self._barrier_free_ready_cached
        full_capacity = all(
            pool.capacity == pool.num_experts
            for pool in self._projection_pools_lockstep()
        )
        if ready is None or (ready is False and full_capacity):
            ready = self._barrier_free_eligible()
            self._barrier_free_ready_cached = ready
        return ready

    def _barrier_free_eligible(self) -> bool:
        if not self._all_iqk:
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
        it. Rows keep their incoming
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
        if self._all_iqk:
            return self._call_iqk_full_resident(x, indices)

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
        # same formula in its epilogue on float32 accumulators, without an
        # intermediate rounding to the row dtype.
        if (
            identity
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
        """Fail-closed eligibility check for barrier-free decode.

        Decode analog of `_barrier_free_ready`: successful and bounded-capacity
        verdicts are stable, while a cold full-capacity pool rechecks after the
        ring path loads more experts. The ring path stays the product path for
        every partial-capacity session."""
        ready = self._barrier_free_decode_ready_cached
        full_capacity = all(
            pool.capacity == pool.num_experts
            for pool in self._projection_pools_lockstep()
        )
        if ready is None or (ready is False and full_capacity):
            ready = self._barrier_free_decode_eligible()
            self._barrier_free_decode_ready_cached = ready
        return ready

    def _barrier_free_decode_eligible(self) -> bool:
        if not self._all_iqk:
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
            if self._all_iqk:
                self._iqk_decode_identity_cached = all(
                    all(pool._slot_of.get(expert) == expert
                        for expert in range(pool.num_experts))
                    for pool in pools
                )
        finally:
            for lock in reversed(locks):
                lock.release()
        return True

    def _iqk_dual_gemv_engaged(self, compact_source_ids) -> bool:
        """Whether the compact-only paired gate/up dispatch may run."""
        if not self._all_iqk:
            return False
        if self._iqk_decode_identity_cached is not True:
            return False
        if compact_source_ids is None:
            return False
        compact_count = int(getattr(compact_source_ids, "size", 0) or 0)
        if compact_count != int(self.gate_proj.pool.num_experts):
            return False
        return (
            int(self.gate_proj.in_features) == int(self.up_proj.in_features)
            and int(self.gate_proj.out_features) == int(self.up_proj.out_features)
        )

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
        if self._all_iqk:
            self._record_iqk_route(int(idx.size))
            if self._iqk_decode_identity_cached:
                gate_idx = up_idx = down_idx = idx
            else:
                gate_idx = self.gate_proj.pool.remap_ondevice(idx)
                up_idx = self.up_proj.pool.remap_ondevice(idx)
                down_idx = self.down_proj.pool.remap_ondevice(idx)
            elements = int(idx.size)
            for projection in (
                self.up_proj,
                self.gate_proj,
                self.down_proj,
            ):
                projection.matmul_slot_calls += 1
                projection.matmul_slot_elements += elements
            x4 = mx.expand_dims(x, (-2, -3))
            x_up = self.up_proj.pool.iqk.gemv(x4, up_idx)
            x_gate = self.gate_proj.pool.iqk.gemv(x4, gate_idx)
            out = self.down_proj.pool.iqk.gemv(
                self.activation(x_up, x_gate),
                down_idx,
            )
            return out.squeeze(-2)
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

    def build_compact_barrier_free_decode(self, x, idx) -> mx.array:
        """Paired IQ_K decode installed only on compact learned layers."""
        compact_source_ids = getattr(
            self,
            "_moespresso_compact_source_ids",
            None,
        )
        if not self._iqk_dual_gemv_engaged(compact_source_ids):
            return PooledSwitchGLU.build_barrier_free_decode(self, x, idx)

        self.barrier_free_decode_calls += 1
        self._record_iqk_route(int(idx.size))
        elements = int(idx.size)
        for projection in (
            self.up_proj,
            self.gate_proj,
            self.down_proj,
        ):
            projection.matmul_slot_calls += 1
            projection.matmul_slot_elements += elements
        x4 = mx.expand_dims(x, (-2, -3))
        from moespresso.runtime.deepseek_v4.iqk_decode_kernel import dual_gemv

        x_gate, x_up = dual_gemv(
            self.gate_proj.pool.iqk,
            self.up_proj.pool.iqk,
            x4,
            idx,
        )
        self.iqk_dual_gemv_calls += 1
        self.iqk_dual_gemv_pairs += elements
        out = self.down_proj.pool.iqk.gemv(
            self.activation(x_up, x_gate),
            idx,
        )
        return out.squeeze(-2)

    def _decode_routed_fused_ready(self) -> bool:
        """One-shot fail-closed eligibility check for the fused decode
        routed matvec family.

        Static facts only (pool layout, codecs, kernel geometry,
        installed mlx_kquant surface); the verdict is decided once and
        cached. Resident calls also require the full-residency certificate
        and the identity-slot check in `decode_routed_fused_engaged`. Ring
        calls use the worker-published slot indices after demand loading.
        Unsupported kernels or layouts retain the composed route."""
        ready = self._decode_routed_fused_ready_cached
        if ready is None:
            ready = self._decode_routed_fused_eligible()
            self._decode_routed_fused_ready_cached = ready
        return ready

    def _decode_routed_fused_eligible(self) -> bool:
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
        iqk_sorted = (
            self._record_iqk_route(int(indices.size))
            if self._all_iqk
            else False
        )
        # Barrier-free full-resident bulk prefill: leaves before the blocking
        # np.asarray(indices) below, so index_sync/index_resync stay untouched
        # on this route (their absence in the stats is the engagement
        # evidence, next to barrier_free_prefill_calls). The expert-set
        # counters (seen/unique) also stay untouched: updating them would
        # need the very host read the route removes.
        if self._barrier_free_bulk_shape(indices) and self._barrier_free_ready():
            self.total_calls += 1
            if bulk_rows == 1:
                self.decode_calls += 1
            else:
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
            if iqk_sorted or (not self._all_iqk and indices.size >= 64):
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
            ticket.batch.wait()
            for future in ticket.batch.futures:
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
        ticket.batch.wait()
        for future in ticket.batch.futures:
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
        batch = submit_loads(
            _PROJECTION_LOAD_EXECUTOR,
            (partial(pool.prefetch, ordered, protect=protect,
                     reserve_floor=reserve_floor) for pool in pools),
        )
        self._prefetch_ticket = _PrefetchTicket(
            predicted=frozenset(predicted),
            batch=batch,
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
        if self._prefill_prefetch_enabled:
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
            if self._prefill_prefetch_enabled:
                self._submit_prefetch_ticket(call_active)
            return out.squeeze(-2)

        chunk_sets = [
            {int(e) for e in idx_host[s:e].tolist()} for s, e in chunks
        ]
        pools = self._projection_pools()

        def _ensure_ahead(active_set, protect_set):
            # fence=False: a worker thread cannot fence stream 0 (thread_local
            # streams). The targeted main-thread wait below replaces it.
            if self._ensure_stream_major_projection_batches(
                active_set,
                protect=protect_set,
                fence=False,
            ):
                return
            self._join_projection_loads(
                partial(pool.ensure, active_set, protect=protect_set, fence=False)
                for pool in pools
            )

        # chunk 0 loads up front (nothing to overlap with yet)
        _ensure_ahead(chunk_sets[0], set())

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
                _ensure_ahead(chunk_sets[i + 1], chunk_sets[i])
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
        if self._prefill_prefetch_enabled:
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
        do_sort = (
            indices.size >= self._iqk_sorted_threshold()
            if self._all_iqk
            else _should_sort_routed_indices(indices)
        )
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
        mechanism the pools use for encoded weights) before committing the layer,
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
        if self._all_iqk:
            self._record_iqk_route(int(idx.size))
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
        # Separate projections: gate/up pools share
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
        if self._all_iqk:
            self._record_iqk_route(int(idx.size))
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

    def ring_install(self, seq: int, K: int, gate_mod=None, *, cancelled=None) -> None:
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
                if cancelled is None:
                    self._ring_install_body(seq, K)
                else:
                    self._ring_install_body(seq, K, cancelled=cancelled)
            finally:
                gate_mod.signal_event(seq)
            return
        if cancelled is None:
            self._ring_install_body(seq, K)
        else:
            self._ring_install_body(seq, K, cancelled=cancelled)

    def _ring_install_body(self, seq: int, K: int, *, cancelled=None) -> None:
        ring = self._ring_np
        t0 = time.perf_counter()
        deadline = t0 + _RING_TIMEOUT
        while True:
            if cancelled is not None and cancelled():
                raise CancelledError("pooled route publication was cancelled")
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
        if _ROUTE_TRACE is not None:
            _ROUTE_TRACE.append(
                ("decode", seq, self.gate_proj.pool.layer, ids.tolist()))
        active = {int(e) for e in idx_host.tolist()}
        self.seen_experts.update(active)
        self.decode_seen_experts.update(active)
        self._ensure_projection_pools(active)
        if cancelled is not None and cancelled():
            raise CancelledError("pooled route publication was cancelled")
        self._publish_pipe_slots(idx_host)

    def _publish_pipe_slots(self, idx_host: np.ndarray) -> None:
        """Publish the shared gate/up and independent down slot indices."""
        K = int(idx_host.size)
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
        self._publish_pipe_slots(idx_host.astype(np.uint32, copy=False))


# A monotonically increasing sequence number distinguishes layer-steps across
# tokens in the ring buffers. The shared request session also binds this value
# to its native event domain.
_RING_SEQ = [0]


def install_compact_iqk_dual_gemv(switch, compact_source_ids) -> bool:
    """Install the paired decode method on one compact IQ_K switch."""
    if (
        not isinstance(switch, PooledSwitchGLU)
        or not switch._all_iqk
    ):
        return False
    compact_count = int(getattr(compact_source_ids, "size", 0) or 0)
    if compact_count <= 0 or any(
        int(pool.num_experts) != compact_count
        for pool in switch._projection_pools_lockstep()
    ):
        return False
    object.__setattr__(
        switch,
        "_moespresso_compact_source_ids",
        compact_source_ids,
    )
    object.__setattr__(switch, "iqk_dual_gemv_calls", 0)
    object.__setattr__(switch, "iqk_dual_gemv_pairs", 0)
    object.__setattr__(
        switch,
        "build_barrier_free_decode",
        MethodType(PooledSwitchGLU.build_compact_barrier_free_decode, switch),
    )
    return True


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
        from moespresso.runtime.pooled_moe_blocks import _run

        return _run(self, x, deepseek=False)


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
        from moespresso.runtime.pooled_moe_blocks import _run

        return _run(self, x, deepseek=True, input_ids=input_ids)
