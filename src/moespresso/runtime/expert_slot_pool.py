"""Persistent routed-expert slots for the SSD-streaming runtime.

This is the bridge-B shape: a fixed MLX pool per routed projection, host-side
integer slot bookkeeping, and direct `pread_into` on misses. The pool arrays are
created once and reused; misses overwrite slots in place after an explicit MLX
synchronization fence.

The package stores one bundle row per (layer, expert): the expert's
full gate/up/down payload, contiguous. A per-layer `BundleRowCache` shared by
the layer's three projection pools turns a miss into one pread + six host
memcpys (was six scattered preads); pools built standalone (no cache) fall back
to exact per-component preads through the same index.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np

from moespresso.runtime.expert_index import ExpertIndex, ProjectionGeometry
from moespresso.package.bundle import KQUANT_CODEC, MXFP4_CODEC, TQ_CODEC
from moespresso.runtime.pread_into import pread_view_cached


class ExpertCapacityExceeded(RuntimeError):
    pass


# In-session hotness decay: halve LFU counters every N
# touches so the pool follows topic shifts in long sessions instead of
# letting early-session experts squat on inflated counts. ~8 touches per
# token per pool at top-8 => 128 touches ~ 16 tokens.
# 0 disables. Long-session serving is a product goal, so this defaults ON
# with the kill switch.
_DECAY_EVERY_TOUCHES = int(
    os.environ.get("MOESPRESSO_SSD_HOTNESS_DECAY_TOUCHES", "128") or "0")

# Page-cache hygiene: after a demand eviction, advise the
# kernel to drop the evicted row's file pages. Advisory-only on a read-only
# mapping - cannot corrupt; worst cases are an ignored hint or a slower
# re-read. Default ON; MOESPRESSO_SSD_EVICT_DONTNEED=0 is the kill switch.
_EVICT_DONTNEED = os.environ.get(
    "MOESPRESSO_SSD_EVICT_DONTNEED", "1") == "1"


class BundleRowCache:
    """One layer's bundle-row reader, shared by its three projection pools.

    A missed expert costs one row pread; each pool then memcpys its two slices.
    The first pool to ask loads the row into a staging buffer; the other two
    consume it from the cache; after `consumers` takes the row is dropped, so
    the steady state is an empty cache (rows only outlive a miss round when
    pool residencies diverge, bounded by `max_rows` LRU).

    Thread-safe: the three pools may ensure in parallel (the projection-load
    executor): an in-flight marker dedups concurrent loads of the same row,
    so the row is pread exactly once even under racing ensures. A failed pread
    leaves nothing published (waiters retry, become the loader, and surface
    the same error to their own ensure, fail-closed, same as before).
    """

    def __init__(self, *, package_dir: str | Path, index: ExpertIndex,
                 layer: int, consumers: int = 3, max_rows: int = 32):
        self.package_dir = Path(package_dir)
        self.index = index
        self.layer = int(layer)
        self.consumers = int(consumers)
        self.max_rows = int(max_rows)
        self._lock = threading.Lock()
        self._rows: OrderedDict[int, list] = OrderedDict()  # expert -> [buf, takes]
        self._inflight: dict[int, threading.Event] = {}
        self.total_preads = 0
        self.total_cached_takes = 0

    def take(self, expert: int) -> memoryview:
        """The expert's bundle row bytes (read-only use; do not mutate)."""
        expert = int(expert)
        while True:
            with self._lock:
                entry = self._rows.get(expert)
                if entry is not None:
                    entry[1] += 1
                    if entry[1] >= self.consumers:
                        del self._rows[expert]
                    else:
                        self._rows.move_to_end(expert)
                    self.total_cached_takes += 1
                    return memoryview(entry[0])
                event = self._inflight.get(expert)
                if event is None:
                    event = threading.Event()
                    self._inflight[expert] = event
                    break  # this thread loads the row
            event.wait()  # another thread is loading it; then retry

        try:
            br = self.index.locate_row(layer=self.layer, expert=expert)
            buf = bytearray(br.nbytes)
            pread_view_cached(
                memoryview(buf),
                self.package_dir / br.shard,
                file_offset=br.offset,
                nbytes=br.nbytes,
            )
            with self._lock:
                self.total_preads += 1
                # the loader's own take counts as the first consumption
                if self.consumers > 1:
                    self._rows[expert] = [buf, 1]
                    self._rows.move_to_end(expert)
                    while len(self._rows) > self.max_rows:
                        self._rows.popitem(last=False)
            return memoryview(buf)
        finally:
            with self._lock:
                self._inflight.pop(expert, None)
            event.set()


class ExpertSlotPool:
    """Fixed-capacity persistent MLX slots for one layer/projection."""

    def __init__(
        self,
        *,
        package_dir: str | Path,
        index: ExpertIndex,
        layer: int,
        projection: str,
        capacity: int,
        eviction_policy: str = "lfu",
        row_cache: BundleRowCache | None = None,
        spare_slots: int = 0,
        combined_kquant_projection: str | None = None,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if spare_slots < 0:
            raise ValueError("spare_slots must be >= 0")
        if eviction_policy not in {"lru", "lfu"}:
            raise ValueError("eviction_policy must be 'lru' or 'lfu'")
        self.package_dir = Path(package_dir)
        self.index = index
        self.layer = layer
        self.projection = projection
        self.combined_kquant_projection = combined_kquant_projection
        self.capacity = int(capacity)
        # Spare slots live above the demand capacity, written only
        # by speculative prefetch (place_spare), never chosen by the demand
        # allocator, never evicted by LFU. A correct prediction becomes a
        # plain hit (the expert is published at its spare slot); a wrong one
        # is overwritten by a later prefetch round-robin. This keeps the
        # live pool's residency untouched by speculation.
        self.spare_slots = int(spare_slots)
        self.eviction_policy = eviction_policy
        self.geometry = index.geometry(layer=layer, projection=projection)
        self.bits = self.geometry.bits
        self.num_experts = index.num_experts
        if capacity + spare_slots > self.num_experts:
            raise ValueError("capacity+spare_slots cannot exceed num_experts")
        # One-pread miss path: the layer's shared BundleRowCache plus this
        # projection's within-row slice geometry. Without a cache (standalone
        # pools, tests) misses fall back to exact per-component preads.
        self.row_cache = row_cache
        comps = index.row_components(layer=layer)
        self.codec = self.geometry.codec
        self.components = index.components_for_projection(layer=layer, projection=projection)
        if self.codec == KQUANT_CODEC:
            self.components = tuple(c for c in self.components if c != "scales")
        weight_component = "weight" if self.codec == KQUANT_CODEC else "packed"
        self._comp_packed = comps.get((projection, weight_component))
        if self._comp_packed is None:
            raise ValueError(
                f"{projection}: {self.codec} pool has no {weight_component} component")
        self._combined_comp_packed = None
        if combined_kquant_projection is not None:
            self._init_combined_kquant_geometry(
                comps=comps,
                projection=projection,
                combined_projection=combined_kquant_projection,
            )
        self._comp_norms = comps.get((projection, "norms"))
        self._comp_scales = comps.get((projection, "scales"))

        total_slots = capacity + self.spare_slots
        packed_dtype = mx.uint8 if self.codec == KQUANT_CODEC else mx.uint32
        self.packed = mx.zeros(
            (total_slots, self.geometry.out_features, self.geometry.packed_cols),
            dtype=packed_dtype,
        )
        self.weight = self.packed if self.codec == KQUANT_CODEC else None
        self.norms = None
        self.scales = None
        if self.codec == TQ_CODEC:
            self.norms = mx.zeros((total_slots, self.geometry.out_features),
                                  dtype=mx.float16)
            mx.eval(self.packed, self.norms)
        elif self.codec == MXFP4_CODEC:
            if self._comp_scales is None:
                raise ValueError(f"{projection}: mxfp4 pool has no scales component")
            scale_shape = tuple(self._comp_scales["shape"])
            self.scales = mx.zeros((total_slots, *scale_shape), dtype=mx.uint8)
            mx.eval(self.packed, self.scales)
        elif self.codec == KQUANT_CODEC:
            self.scales = mx.zeros((1,), dtype=mx.uint8)
            mx.eval(self.packed, self.scales)
        else:
            raise ValueError(f"{projection}: unsupported expert codec {self.codec!r}")
        self._packed_view = memoryview(self.packed).cast("B")
        self._norms_view = memoryview(self.norms).cast("B") if self.norms is not None else None
        self._scales_view = memoryview(self.scales).cast("B") if self.scales is not None else None

        self._slot_of: dict[int, int] = {}
        self._expert_at: list[int | None] = [None] * total_slots
        # Bookkeeping lock: speculative prefetch runs on its own
        # executor concurrently with the worker's demand ensure. Only the
        # dict/list bookkeeping is locked; preads happen outside the lock so
        # a slow prefetch read can never block a demand ensure's bookkeeping.
        self._bk_lock = threading.Lock()
        self._prefetch_inflight = 0
        # expert ids currently reserved by an in-flight prefetch (bytes
        # landing). A demand ensure for such an expert waits for the publish
        # instead of double-loading into a second slot: double-residency
        # leaks the loser slot forever (_expert_at ghost never freed; surfaces
        # as a slow capacity leak ending in spurious
        # ExpertCapacityExceeded after ~17k prefetch evictions).
        self._prefetch_reserved: set[int] = set()
        # The most recent demand active set, updated atomically inside
        # ensure()'s locked phase 1. Prefetch always protects it: a stale
        # protect snapshot taken on the prefetch executor one token earlier
        # must never evict an expert the current step just published (surfaces
        # as a KeyError in the worker's slot read).
        self._demand_protect: set[int] = set()
        self._pending_advise: list[int] = []
        self.total_dontneed = 0
        self.total_dontneed_errors = 0
        self.total_prefetch_loads = 0
        self.total_prefetch_skips = 0
        # Recency stamps replace the old _lru python list: the list cost O(n)
        # per touch (membership + remove) and O(n^2) per eviction (index() in
        # the LFU tie-break) at capacity-70 pools on every decode layer. A
        # monotonic clock preserves the exact ordering semantics (smaller
        # stamp == older == old list position) at O(1) per touch.
        self._recency: dict[int, int] = {}
        self._clock = 0
        self._freq: dict[int, int] = {}
        self.total_hits = 0
        self.total_misses = 0
        self.total_loads = 0
        self.total_evictions = 0
        self.total_load_seconds = 0.0

        # On-device slot table for the sync-free remap (remap_ondevice). The
        # sentinel num_experts (an invalid slot) marks an absent expert. Rebuilt
        # lazily from _slot_of whenever residency changes (_slot_table_dirty). This avoids
        # rebuilding projection slot indices on the host; measured e2e speed impact
        # has been roughly neutral. Treat this as seam cleanup without a speed claim.
        self._slot_sentinel = self.num_experts
        self._slot_table: mx.array | None = None
        self._slot_table_identity = False
        self._slot_table_dirty = True
        # Instrumentation: each rebuild is one small host->device upload on
        # the decode path. This event proxies dispatch engagement and carries no timing.
        self.slot_table_rebuilds = 0

    def _init_combined_kquant_geometry(
        self,
        *,
        comps: dict[tuple[str, str], dict],
        projection: str,
        combined_projection: str,
    ) -> None:
        if self.codec != KQUANT_CODEC:
            raise ValueError(
                f"{projection}+{combined_projection}: combined pool requires kquant, "
                f"got {self.codec!r}")
        other = self.index.geometry(
            layer=self.layer,
            projection=combined_projection,
        )
        if other.codec != KQUANT_CODEC:
            raise ValueError(
                f"{projection}+{combined_projection}: combined pool requires both "
                f"projections to be kquant, got {other.codec!r}")
        checks = (
            "bits",
            "packed_cols",
            "packed_dtype",
            "kquant_codec",
            "group_size",
            "bytes_per_block",
            "weights_per_block",
        )
        mismatched = [
            name
            for name in checks
            if getattr(self.geometry, name) != getattr(other, name)
        ]
        if mismatched:
            raise ValueError(
                f"{projection}+{combined_projection}: incompatible kquant geometry "
                f"for combined pool ({', '.join(mismatched)})")
        comp = comps.get((combined_projection, "weight"))
        if comp is None:
            raise ValueError(
                f"{projection}+{combined_projection}: missing {combined_projection}.weight")
        if comp["nbytes"] != self._comp_packed["nbytes"]:
            raise ValueError(
                f"{projection}+{combined_projection}: weight byte rows differ "
                f"({self._comp_packed['nbytes']} != {comp['nbytes']})")
        self._combined_comp_packed = comp
        self.geometry = ProjectionGeometry(
            codec=KQUANT_CODEC,
            out_features=self.geometry.out_features + other.out_features,
            packed_cols=self.geometry.packed_cols,
            bits=self.geometry.bits,
            packed_dtype=self.geometry.packed_dtype,
            scales_dtype=self.geometry.scales_dtype,
            kquant_codec=self.geometry.kquant_codec,
            group_size=self.geometry.group_size,
            bytes_per_block=self.geometry.bytes_per_block,
            weights_per_block=self.geometry.weights_per_block,
        )

    @property
    def is_combined_kquant(self) -> bool:
        return self._combined_comp_packed is not None

    def _packed_row_nbytes(self) -> int:
        total = int(self._comp_packed["nbytes"])
        if self._combined_comp_packed is not None:
            total += int(self._combined_comp_packed["nbytes"])
        return total

    def resident_ids(self) -> set[int]:
        return set(self._slot_of)

    def slot_of(self, expert: int) -> int:
        return self._slot_of[int(expert)]

    def free_slots(self) -> int:
        return self.capacity - len(self._slot_of)

    def grow(self, capacity: int) -> None:
        capacity = int(capacity)
        if capacity < self.capacity:
            raise ValueError("ExpertSlotPool.grow cannot shrink capacity")
        if capacity + self.spare_slots > self.num_experts:
            raise ValueError("capacity cannot exceed num_experts")
        if capacity == self.capacity:
            return

        with self._bk_lock:
            delta = capacity - self.capacity
            old_capacity = self.capacity
            total = capacity + self.spare_slots
            packed_dtype = mx.uint8 if self.codec == KQUANT_CODEC else mx.uint32
            packed = mx.zeros(
                (total, self.geometry.out_features, self.geometry.packed_cols),
                dtype=packed_dtype,
            )
            norms = None
            scales = None
            if self.codec == TQ_CODEC:
                norms = mx.zeros((total, self.geometry.out_features),
                                 dtype=mx.float16)
                mx.eval(packed, norms)
            elif self.codec == MXFP4_CODEC:
                assert self._comp_scales is not None
                scales = mx.zeros((total, *self._comp_scales["shape"]), dtype=mx.uint8)
                mx.eval(packed, scales)
            elif self.codec == KQUANT_CODEC:
                scales = self.scales
                mx.eval(packed)
            else:
                raise ValueError(f"{self.projection}: unsupported expert codec {self.codec!r}")
            packed_view = memoryview(packed).cast("B")
            norms_view = memoryview(norms).cast("B") if norms is not None else None
            scales_view = memoryview(scales).cast("B") if scales is not None else None
            prow = self._packed_row_nbytes()
            nrow = self.geometry.out_features * 2 if norms is not None else 0
            srow = (
                self._comp_scales["nbytes"]
                if self.codec == MXFP4_CODEC and scales is not None
                else 0
            )
            # demand region copies in place; the spare region moves up by
            # delta rows. Growth must not drop the spare rows while _slot_of
            # still points at them (that caused out-of-bounds writes on the next
            # place_spare_trio when lookahead + growth both ran).
            packed_view[:old_capacity * prow] = (
                self._packed_view[:old_capacity * prow])
            if norms_view is not None and self._norms_view is not None:
                norms_view[:old_capacity * nrow] = (
                    self._norms_view[:old_capacity * nrow])
            if scales_view is not None and self._scales_view is not None:
                scales_view[:old_capacity * srow] = (
                    self._scales_view[:old_capacity * srow])
            if self.spare_slots:
                packed_view[capacity * prow:total * prow] = (
                    self._packed_view[old_capacity * prow:
                                      (old_capacity + self.spare_slots) * prow])
                if norms_view is not None and self._norms_view is not None:
                    norms_view[capacity * nrow:total * nrow] = (
                        self._norms_view[old_capacity * nrow:
                                         (old_capacity + self.spare_slots) * nrow])
                if scales_view is not None and self._scales_view is not None:
                    scales_view[capacity * srow:total * srow] = (
                        self._scales_view[old_capacity * srow:
                                          (old_capacity + self.spare_slots) * srow])

            self.packed = packed
            self.weight = packed if self.codec == KQUANT_CODEC else None
            self.norms = norms
            self.scales = scales
            self._packed_view = packed_view
            self._norms_view = norms_view
            self._scales_view = scales_view
            spare_tail = self._expert_at[old_capacity:]
            self._expert_at = (self._expert_at[:old_capacity]
                               + [None] * delta + spare_tail)
            self._slot_of = {
                expert: (slot + delta if slot >= old_capacity else slot)
                for expert, slot in self._slot_of.items()
            }
            self._slot_table_dirty = True
            self.capacity = capacity

    def seed_hot(self, limit: int | None = None) -> list[int]:
        free = self.free_slots()
        if free <= 0:
            return []
        if limit is not None:
            free = min(free, int(limit))
        if free <= 0:
            return []
        ranked = sorted(
            (
                (-count, expert)
                for expert, count in self._freq.items()
                if expert not in self._slot_of
            ),
        )
        experts = [expert for _neg_count, expert in ranked[:free]]
        if experts:
            self.ensure(experts)
        return experts

    def _touch(self, expert: int) -> None:
        self._freq[expert] = self._freq.get(expert, 0) + 1
        self._clock += 1
        self._recency[expert] = self._clock
        if _DECAY_EVERY_TOUCHES and self._clock % _DECAY_EVERY_TOUCHES == 0:
            # halve everything: the effective window is the recent few
            # hundred touches, so a topic shift overturns the pool in
            # ~100-200 tokens instead of never. Recency tie-break is
            # unchanged; persisted hotlists now reflect recent demand.
            self._freq = {e: c >> 1 for e, c in self._freq.items()}

    def _choose_slot(self, protected: set[int]) -> tuple[int, bool]:
        """Reserve a slot (bookkeeping only). Returns (slot, evicted).

        Does not synchronize: the caller fences once per ensure() batch before
        any reclaimed slot is overwritten (same use-after-free safety as the
        old per-eviction mx.synchronize(), without N pipeline drains per batch,
        measured ~3.5 evictions/decode-layer warm, each draining the GPU and
        defeating the block-exit kick's overlap)."""
        for slot in range(self.capacity):
            if self._expert_at[slot] is None:
                return slot, False
        candidates = [expert for expert, slot in self._slot_of.items()
                      if expert not in protected and slot < self.capacity]
        if candidates:
            if self.eviction_policy == "lfu":
                # tie-break on recency: smaller stamp == older (same ordering
                # the old _lru.index() tie-break produced)
                expert = min(candidates, key=lambda e: (self._freq.get(e, 0),
                                                        self._recency.get(e, 0)))
            else:
                expert = min(candidates, key=lambda e: self._recency.get(e, 0))
            slot = self._slot_of.pop(expert)
            self._expert_at[slot] = None
            self._recency.pop(expert, None)
            self.total_evictions += 1
            self._slot_table_dirty = True
            if _EVICT_DONTNEED and self.projection == "gate_proj":
                self._pending_advise.append(expert)
            return slot, True
        # Diagnostic state in the message: this raise is fail-closed and
        # rare, and the slot ledger is exactly what an investigation needs
        # (a slot that is occupied but unpublished points at a leaked or
        # in-flight reservation; protected residents point at a caller
        # whose active-plus-protect set exceeds capacity).
        demand_res = sum(
            1 for _e, sl in self._slot_of.items() if sl < self.capacity)
        spare_res = sum(
            1 for _e, sl in self._slot_of.items() if sl >= self.capacity)
        protected_res = sum(
            1 for e, sl in self._slot_of.items()
            if sl < self.capacity and e in protected)
        occupied = sum(
            1 for sl in range(self.capacity)
            if self._expert_at[sl] is not None)
        unpublished = sum(
            1 for sl in range(self.capacity)
            if self._expert_at[sl] is not None
            and self._slot_of.get(self._expert_at[sl]) != sl)
        raise ExpertCapacityExceeded(
            f"capacity {self.capacity} cannot hold active experts "
            f"{sorted(protected)} (demand residents {demand_res}, protected "
            f"among them {protected_res}, spare residents {spare_res}, "
            f"occupied demand slots {occupied}, unpublished occupancies "
            f"{unpublished}, prefetch inflight {self._prefetch_inflight}, "
            f"prefetch reserved {sorted(self._prefetch_reserved)})")

    def _load_component(self, *, expert: int, slot: int, component: str) -> None:
        if self.is_combined_kquant:
            raise ValueError(
                f"{self.projection}+{self.combined_kquant_projection}: "
                "_load_component is not used for combined K-quant pools")
        br = self.index.locate(
            layer=self.layer,
            expert=expert,
            projection=self.projection,
            component=component,
        )
        if component in {"packed", "weight"}:
            dst = self._packed_view
        elif component == "norms" and self._norms_view is not None:
            dst = self._norms_view
        elif component == "scales" and self._scales_view is not None:
            dst = self._scales_view
        else:
            raise ValueError(f"{self.projection}: no {component} buffer for codec {self.codec}")
        pread_view_cached(
            dst,
            self.package_dir / br.shard,
            file_offset=br.offset,
            nbytes=br.nbytes,
            dst_offset=slot * br.nbytes,
        )

    def slot_nbytes(self) -> int:
        total = self._packed_row_nbytes()
        if self._comp_norms is not None:
            total += int(self._comp_norms["nbytes"])
        if self.codec == MXFP4_CODEC and self._comp_scales is not None:
            total += int(self._comp_scales["nbytes"])
        return total

    def _load_expert(self, *, expert: int, slot: int) -> None:
        """Land one expert's projection components in `slot` (fail-closed on raise).

        With a shared row cache this is one bundle-row pread for the whole
        layer (the cache dedups across the three pools) + two host memcpys;
        without, two exact per-component preads.
        """
        if self.row_cache is None:
            if self.is_combined_kquant:
                assert self._combined_comp_packed is not None
                primary = self.index.locate(
                    layer=self.layer,
                    expert=expert,
                    projection=self.projection,
                    component="weight",
                )
                secondary = self.index.locate(
                    layer=self.layer,
                    expert=expert,
                    projection=self.combined_kquant_projection,
                    component="weight",
                )
                dst_offset = slot * self._packed_row_nbytes()
                pread_view_cached(
                    self._packed_view,
                    self.package_dir / primary.shard,
                    file_offset=primary.offset,
                    nbytes=primary.nbytes,
                    dst_offset=dst_offset,
                )
                pread_view_cached(
                    self._packed_view,
                    self.package_dir / secondary.shard,
                    file_offset=secondary.offset,
                    nbytes=secondary.nbytes,
                    dst_offset=dst_offset + primary.nbytes,
                )
                return
            for component in self.components:
                self._load_component(expert=expert, slot=slot, component=component)
            return
        row = self.row_cache.take(expert)
        pc = self._comp_packed
        pn = pc["nbytes"]
        packed_row = self._packed_row_nbytes()
        base = slot * packed_row
        self._packed_view[base:base + pn] = row[pc["offset"]:pc["offset"] + pn]
        if self.is_combined_kquant:
            assert self._combined_comp_packed is not None
            uc = self._combined_comp_packed
            un = uc["nbytes"]
            self._packed_view[base + pn:base + pn + un] = (
                row[uc["offset"]:uc["offset"] + un])
            return
        if self._comp_norms is not None and self._norms_view is not None:
            nc = self._comp_norms
            nn = nc["nbytes"]
            self._norms_view[slot * nn:(slot + 1) * nn] = (
                row[nc["offset"]:nc["offset"] + nn])
        if (
            self.codec == MXFP4_CODEC
            and self._comp_scales is not None
            and self._scales_view is not None
        ):
            sc = self._comp_scales
            sn = sc["nbytes"]
            self._scales_view[slot * sn:(slot + 1) * sn] = (
                row[sc["offset"]:sc["offset"] + sn])

    def _active_set(self, expert_ids: list[int] | tuple[int, ...] | set[int]) -> set[int]:
        active = {int(e) for e in expert_ids}
        if len(active) > self.capacity:
            raise ExpertCapacityExceeded(
                f"capacity {self.capacity} cannot hold active experts {sorted(active)}")
        for expert in active:
            if expert < 0 or expert >= self.num_experts:
                raise IndexError(f"expert {expert} out of range [0, {self.num_experts})")
        return active

    def missing_count(self, expert_ids: list[int] | tuple[int, ...] | set[int]) -> int:
        active = self._active_set(expert_ids)
        return sum(1 for expert in active if expert not in self._slot_of)

    def ensure(
        self,
        expert_ids: list[int] | tuple[int, ...] | set[int],
        *,
        protect: set[int] | None = None,
        fence: bool = True,
    ) -> None:
        """Make every expert in `expert_ids` resident.

        `protect` pins additional residents against eviction during this call
        (the prefill chunk-ahead overlap loads chunk i+1 while the GPU is
        still computing chunk i: chunk i's experts must not be victims). The
        caller guarantees |active ∪ protect| <= capacity.

        Two-phase: (1) bookkeeping, count hits, reserve a slot for every miss
        (evictions are dict/list ops only); (2) one mx.synchronize() fence if
        any reclaimed slot will be overwritten, so in-flight GPU work that may
        still read old slot contents finishes before pread rewrites them; then
        (3) all pread loads. Safety is identical to the old per-eviction sync
        (the fence still strictly precedes every overwrite of a reclaimed
        slot); the per-batch fence count drops from N evictions to <=1."""
        active = self._active_set(expert_ids)
        protected = active if not protect else active | set(protect)
        t0 = time.perf_counter()
        placements: list[tuple[int, int]] = []
        evicted_any = False
        while True:
            with self._bk_lock:
                self._demand_protect = set(active)
                # Demand outranks speculation: in-flight prefetch
                # reservations are invisible to _choose_slot, so a big
                # demand batch (a prefill chunk wants ~capacity slots) could
                # see spurious capacity exhaustion. If the batch does not
                # fit immediately but prefetch reservations are in flight,
                # wait for them to drain and retry instead of raising
                # (surfaced in the warm-pass prefill). Feasibility and
                # phase 1 share one lock acquisition so no new reservation
                # can slip between them.
                needed = sum(1 for e in active if e not in self._slot_of)
                free = sum(1 for sl in range(self.capacity)
                           if self._expert_at[sl] is None)
                # Demand slots only: a spare-resident expert is never an
                # eviction candidate (the demand allocator cannot reclaim
                # slots above capacity), so counting it here would let the
                # batch proceed into a deficit instead of waiting out the
                # in-flight reservations.
                evictable = sum(
                    1 for e, sl in self._slot_of.items()
                    if e not in protected and sl < self.capacity)
                if active & self._prefetch_reserved:
                    retry = True  # an active's bytes are landing: await publish
                elif needed > free + evictable and self._prefetch_inflight > 0:
                    retry = True
                else:
                    retry = False
                    try:
                        for expert in sorted(active):
                            if expert in self._slot_of:
                                self.total_hits += 1
                                self._touch(expert)
                                continue
                            self.total_misses += 1
                            slot, evicted = self._choose_slot(protected)
                            evicted_any = evicted_any or evicted
                            # Reserve occupancy only (so this batch cannot
                            # double-assign the slot). Residency (_slot_of) is
                            # published per expert in phase 3, after its bytes
                            # landed. A failed pread must never leave the pool
                            # believing an expert is resident over stale slot
                            # bytes (fail-closed; regression caught by review of
                            # the two-phase refactor). _touch here is safe:
                            # freq/recency entries for not-yet-resident experts
                            # are already legitimate (seed_hot).
                            self._expert_at[slot] = expert
                            self._touch(expert)
                            placements.append((expert, slot))
                    except BaseException:
                        # Release this batch's occupancy reservations before
                        # surfacing: a mid-batch capacity raise must not leak
                        # occupied-but-unpublished slots (each leaked slot is
                        # neither free nor evictable and permanently shrinks
                        # the pool).
                        for expert, slot in placements:
                            self._expert_at[slot] = None
                        raise
            if not retry:
                break
            time.sleep(0.0005)

        if _EVICT_DONTNEED and self._pending_advise:
            with self._bk_lock:
                advise, self._pending_advise = self._pending_advise, []
            for evicted in advise:  # advisory syscalls OUTSIDE the lock
                self._advise_dontneed(evicted)
        if not placements:
            return
        if evicted_any and fence:
            # From a worker thread this does not reliably fence stream 0 (MLX
            # streams are thread_local; synchronize(s) on a foreign stream
            # raises, no-arg behavior is undefined for the main GPU stream).
            # Decode paths are safe without it (per-layer pools +
            # publish-before-commit + token-boundary evals mean no in-flight
            # reader of a victim slot can exist). The prefill chunk-ahead path
            # passes fence=False and performs a targeted main-thread wait
            # (mx.eval of the chunk i-1 output) before any evicting
            # ensure-ahead is submitted.
            mx.synchronize()
        loaded = 0
        try:
            for expert, slot in placements:
                self._load_expert(expert=expert, slot=slot)
                with self._bk_lock:
                    self._slot_of[expert] = slot
                    self._slot_table_dirty = True
                self.total_loads += 1
                loaded += 1
        except BaseException:
            # Unwind the unpublished reservations: the failing expert and any
            # not-yet-loaded ones simply are not resident; a retry re-loads
            # them. Already-published experts have good bytes and stay.
            with self._bk_lock:
                for expert, slot in placements[loaded:]:
                    self._expert_at[slot] = None
            raise
        self.total_load_seconds += time.perf_counter() - t0

    def prefetch(self, expert_ids, *, protect: set[int] | None = None,
                 reserve_floor: int = 16) -> int:
        """Speculatively load predicted experts. Best-effort: skips
        already-resident ids, never raises on capacity (stops reserving when
        only protected residents remain), never fences (between the per-token
        drain and this layer's publish no kernel reads this pool's slots,
        the same ordering argument as the worker's fence-free decode ensure).

        `reserve_floor`: in-flight prefetch reservations make slots invisible
        to a concurrent demand ensure (reserved-but-unpublished slots are
        neither free nor evictable), so prefetch caps its in-flight
        reservations at capacity - reserve_floor: the demand path always
        finds at least reserve_floor reachable slots (callers pass >= 2x
        top_k: demand needs <= top_k and holds <= top_k of its own transient
        reservations). A wrong prediction costs only the wasted pread.
        Returns loads."""
        protected = {int(e) for e in (protect or ())}
        placements: list[tuple[int, int]] = []
        with self._bk_lock:
            protected |= self._demand_protect
            budget = self.capacity - reserve_floor - self._prefetch_inflight
            for expert in expert_ids:
                if budget <= 0:
                    self.total_prefetch_skips += 1
                    break
                expert = int(expert)
                if (expert < 0 or expert >= self.num_experts
                        or expert in self._slot_of
                        or expert in self._prefetch_reserved):
                    continue
                try:
                    slot, _evicted = self._choose_slot(protected)
                except ExpertCapacityExceeded:
                    self.total_prefetch_skips += 1
                    break
                self._expert_at[slot] = expert
                self._prefetch_reserved.add(expert)
                self._touch(expert)
                placements.append((expert, slot))
                budget -= 1
            self._prefetch_inflight += len(placements)
        loaded = 0
        try:
            for expert, slot in placements:
                self._load_expert(expert=expert, slot=slot)
                with self._bk_lock:
                    self._slot_of[expert] = slot
                    self._prefetch_reserved.discard(expert)
                    self._slot_table_dirty = True
                self.total_prefetch_loads += 1
                loaded += 1
        except BaseException:
            with self._bk_lock:
                for expert, slot in placements[loaded:]:
                    self._prefetch_reserved.discard(expert)
                    if self._expert_at[slot] == expert:
                        self._expert_at[slot] = None
            raise
        finally:
            with self._bk_lock:
                self._prefetch_inflight -= len(placements)
                for expert, _slot in placements:
                    self._prefetch_reserved.discard(expert)
        return loaded

    def _advise_dontneed(self, expert: int) -> None:
        """Best-effort: drop the evicted expert's file pages from the page
        cache (mmap read-only + MADV_DONTNEED + munmap; macOS treats it as
        a hint). Never raises; counts errors. ~5-10us per call, demand
        evictions only, gate pool only (one row covers all projections)."""
        import mmap as _mmaplib
        try:
            br = self.index.locate_row(layer=self.layer, expert=expert)
            gran = _mmaplib.ALLOCATIONGRANULARITY
            start = (br.offset // gran) * gran
            length = (br.offset + br.nbytes) - start
            fd = os.open(self.package_dir / br.shard, os.O_RDONLY)
            try:
                m = _mmaplib.mmap(fd, length, prot=_mmaplib.PROT_READ,
                                  offset=start)
                try:
                    m.madvise(_mmaplib.MADV_DONTNEED)
                finally:
                    m.close()
            finally:
                os.close(fd)
            self.total_dontneed += 1
        except Exception:
            self.total_dontneed_errors += 1

    def remap_loaded(self, indices_host, shape: tuple[int, ...]) -> mx.array:
        host = np.asarray(indices_host).reshape(-1)
        remapped = np.array([self._slot_of[int(e)] for e in host], dtype=np.uint32)
        return mx.array(remapped.reshape(shape))

    def _ensure_slot_table(self) -> mx.array:
        """Build/refresh the on-device expert_id -> slot table from _slot_of.

        Rebuilt only when residency changed (cheap, num_experts-length, and it does
        not depend on the routing `indices`, so it never forces the per-token routing
        graph, unlike remap_loaded's per-call host materialization)."""
        if self._slot_table is None or self._slot_table_dirty:
            host = np.full(self.num_experts, self._slot_sentinel, dtype=np.uint32)
            for expert, slot in self._slot_of.items():
                host[expert] = slot
            self._slot_table = mx.array(host)
            self._slot_table_identity = (
                len(self._slot_of) == self.num_experts
                and bool(np.array_equal(
                    host, np.arange(self.num_experts, dtype=np.uint32)))
            )
            self._slot_table_dirty = False
            self.slot_table_rebuilds += 1
        return self._slot_table

    def slot_table_is_identity(self) -> bool:
        """True when every expert is resident at slot == expert id.

        Full prewarm produces this layout (experts seed in ascending order
        into ascending free slots of an empty pool), and a fully resident
        pool never evicts, so the verdict is stable once reached. Routed ids
        then already are slot ids, which lets the bulk prefill share one
        sort between projections instead of remapping and re-permuting rows
        per pool."""
        self._ensure_slot_table()
        return self._slot_table_identity

    def remap_ondevice(self, indices: mx.array) -> mx.array:
        """Sync-free remap of routed expert ids -> resident pool slots.

        `slot_table[indices]` is a pure on-device gather (no np.asarray of `indices`,
        no Python loop, no host->device round-trip). Proven numerically identical to
        remap_loaded. The caller must have `ensure`d all active experts
        first, so every gathered value is a valid resident slot (no sentinel reaches
        the kernel). Returns uint32, shape == indices.shape."""
        table = self._ensure_slot_table()
        return table[indices]

    def remap(self, indices: mx.array) -> mx.array:
        host = np.asarray(indices).reshape(-1)
        active = [int(e) for e in host.tolist()]
        self.ensure(active)
        return self.remap_loaded(host, indices.shape)


def place_spare_trio(pools, expert: int, spare_index: int) -> bool:
    """Atomically place `expert` into the same spare slot of all three
    projection pools. All-or-nothing under all three bookkeeping
    locks (fixed order; demand only ever takes single locks, so no cycle):
    the fused islands index the up pool with the gate pool's slot ids, so a
    partial trio placement physically desynchronizes gate/up bytes, the
    exact corruption this guards against. Publication happens only after
    all three loads landed; a demand ensure arriving mid-flight waits on the
    reservation registry. Returns True when loaded."""
    expert = int(expert)
    locks = [pool._bk_lock for pool in pools]
    for lock in locks:
        lock.acquire()
    try:
        for pool in pools:
            if not (0 <= spare_index < pool.spare_slots):
                return False
            slot = pool.capacity + spare_index
            if (expert in pool._slot_of or expert in pool._prefetch_reserved
                    or expert < 0 or expert >= pool.num_experts):
                return False
            # A demand ensure's phase-1 placement reserves occupancy only
            # (`_expert_at`), publishing `_slot_of` after the bytes land.
            # Those in-flight placements are invisible to the two checks
            # above, and placing the same expert into a spare here would
            # split it across two slots; whichever publishes last strands
            # the other slot as an occupied-but-unpublished leak (neither
            # free nor evictable, a permanent capacity loss that ends in
            # spurious ExpertCapacityExceeded). Refuse on any occupancy.
            if expert in pool._expert_at:
                return False
            occupant = pool._expert_at[slot]
            if occupant is not None and (
                    occupant in pool._demand_protect
                    or occupant in pool._prefetch_reserved):
                return False
        for pool in pools:
            slot = pool.capacity + spare_index
            occupant = pool._expert_at[slot]
            if occupant is not None:
                # Sever only the spare mapping this eviction owns: if the
                # occupant's published residency points elsewhere (stale
                # occupancy metadata), popping it would strand that other
                # slot as an occupied-but-unpublished leak.
                if pool._slot_of.get(occupant) == slot:
                    pool._slot_of.pop(occupant, None)
                    pool._recency.pop(occupant, None)
                pool._slot_table_dirty = True
            pool._expert_at[slot] = expert
            pool._prefetch_reserved.add(expert)
            pool._prefetch_inflight += 1
            pool._touch(expert)
    finally:
        for lock in reversed(locks):
            lock.release()
    try:
        for pool in pools:
            pool._load_expert(expert=expert, slot=pool.capacity + spare_index)
        for pool in pools:
            with pool._bk_lock:
                pool._slot_of[expert] = pool.capacity + spare_index
                pool._slot_table_dirty = True
                pool.total_prefetch_loads += 1
        return True
    except BaseException:
        for pool in pools:
            with pool._bk_lock:
                slot = pool.capacity + spare_index
                if pool._expert_at[slot] == expert and expert not in pool._slot_of:
                    pool._expert_at[slot] = None
        raise
    finally:
        for pool in pools:
            with pool._bk_lock:
                pool._prefetch_reserved.discard(expert)
                pool._prefetch_inflight -= 1
