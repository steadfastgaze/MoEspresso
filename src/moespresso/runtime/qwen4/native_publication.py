"""Publish resident Qwen projection slots through the shared native event."""

from __future__ import annotations

from concurrent.futures import CancelledError
from contextlib import contextmanager
import math
import threading
import time

import numpy as np

from moespresso.package.iqk_format import IQK_LAYOUT_IQK_RELAYOUT


ABI = "qwen-native-all-hit-gate-v2"
_ACTIVE = 10
_EXPERTS = 512
# Yield ownership and check cancellation between bounded native polling slices.
_POLL_SLICE_NS = 1_000_000


def _eligible_pool(pool) -> bool:
    return (
        type(pool.capacity) is int
        and _ACTIVE <= pool.capacity < _EXPERTS
        and pool.num_experts == _EXPERTS
        and getattr(pool, "spare_slots", None) == 0
        and getattr(pool, "_growth_pending", None) is False
        and getattr(pool, "_loads_inflight", None) == 0
        and getattr(pool, "_prefetch_inflight", None) == 0
        and type(getattr(pool, "_prefetch_reserved", None)) is set
        and not pool._prefetch_reserved
        and getattr(getattr(pool, "geometry", None), "layout", None) == IQK_LAYOUT_IQK_RELAYOUT
        and getattr(pool, "iqk", None) is not None
    )


def _storage(pool):
    module = pool.iqk
    names = tuple(module.stream_names())
    if not names or len(set(names)) != len(names) or set(names) != set(pool._iqk_views):
        raise RuntimeError("native all-hit IQK stream ownership differs")
    return (
        id(pool),
        pool.capacity,
        id(module),
        tuple((name, id(getattr(module, name)), id(pool._iqk_views[name])) for name in names),
        id(pool._iqk_copy_plan),
    )


def _epochs(pools):
    return tuple((pool.total_loads, pool.total_evictions) for pool in pools)


def _checksum(ids, sequence):
    value = 2166136261
    for expert in ids:
        value = ((value ^ int(expert)) * 16777619) & 0xFFFFFFFF
    return ((value ^ sequence) * 16777619) & 0xFFFFFFFF


class Qwen4NativePublication:
    """Bind one Qwen switch to its existing FIFO publication worker.

    The native call holds all projection locks while checking the actual GPU
    route and releasing its gate. The shared reader then performs the usual
    demand accounting. Its publication consumes a receipt without rewriting
    slot buffers that the GPU may already be reading. Delayed route exports
    remain eligible across polling slices. Misses use the shared reader.
    """

    def __init__(self, expected_executor, *, timeout_seconds=10.0):
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("native publication timeout must be positive and finite")
        self._timeout_seconds = float(timeout_seconds)
        self._executor = expected_executor
        self._switch = None
        self._pools = None
        self._storage_key = None
        self._maps = None
        self._map_key = None
        self._memberships = None
        self._backpoints = None
        self._bufs = None
        self._destinations = None
        self._ring = None
        self._ring_owner = None
        self._receipt = None
        self._closed = False
        self._stats = dict.fromkeys(
            (
                "native_calls",
                "published",
                "miss",
                "pending",
                "suppressed",
                "map_builds",
                "ineligible",
                "poll_slices",
                "poll_yields",
                "timed_out",
            ),
            0,
        )

    @contextmanager
    def _locked(self, pools):
        acquired = []
        try:
            for pool in pools:
                pool._bk_lock.acquire()
                acquired.append(pool._bk_lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()

    def _prepare_maps(self, pools):
        storage = tuple(_storage(pool) for pool in pools)
        key = _epochs(pools)
        if self._maps is not None and storage == self._storage_key and key == self._map_key:
            if any(
                pool._slot_of != members or pool._expert_at != backpoints
                for pool, members, backpoints in zip(
                    pools, self._memberships, self._backpoints, strict=True
                )
            ):
                raise RuntimeError("native all-hit membership changed without a load or eviction")
            return self._maps
        maps = np.full((3, _EXPERTS), _EXPERTS, dtype=np.uint32)
        for projection, pool in enumerate(pools):
            if (
                type(pool._slot_of) is not dict
                or type(pool._expert_at) is not list
                or len(pool._expert_at) != pool.capacity
            ):
                raise RuntimeError("native all-hit pool membership geometry differs")
            for expert, slot in pool._slot_of.items():
                if (
                    type(expert) is not int
                    or not 0 <= expert < _EXPERTS
                    or type(slot) is not int
                    or not 0 <= slot < pool.capacity
                    or pool._expert_at[slot] != expert
                ):
                    raise RuntimeError("native all-hit membership backpointer differs")
                maps[projection, expert] = slot
            for slot, expert in enumerate(pool._expert_at):
                if expert is not None and pool._slot_of.get(expert) != slot:
                    raise RuntimeError("native all-hit has unpublished slot occupancy")
        maps.flags.writeable = False
        self._pools = pools
        self._storage_key = storage
        self._maps = maps
        self._map_key = key
        self._memberships = tuple(dict(pool._slot_of) for pool in pools)
        self._backpoints = tuple(list(pool._expert_at) for pool in pools)
        self._stats["map_builds"] += 1
        return maps

    def _prepare_destinations(self, switch):
        bufs = switch._qwen4_pipe_bufs(_ACTIVE, create=False)
        if len(bufs) != 6:
            raise RuntimeError("native all-hit projection buffers differ")
        graph_bufs = getattr(switch, "_qwen4_pipe_event_buffers", None)
        if (
            type(graph_bufs) is not tuple
            or len(graph_bufs) != 6
            or any(a is not b for a, b in zip(bufs, graph_bufs, strict=True))
        ):
            raise RuntimeError("native all-hit graph projection buffer ownership changed")
        if self._bufs is None:
            destinations = tuple(np.frombuffer(view, dtype=np.uint32) for view in bufs[1::2])
            if any(
                value.shape != (_ACTIVE,) or not value.flags.writeable for value in destinations
            ):
                raise RuntimeError("native all-hit slot destinations differ")
            self._bufs = bufs
            self._destinations = destinations
        elif any(a is not b for a, b in zip(bufs, self._bufs, strict=True)):
            raise RuntimeError("native all-hit projection buffer ownership changed")
        return self._destinations

    def begin(self, switch, sequence, count, *, cancelled=None) -> bool:
        """Await the actual route with bounded ownership and cancellation checks."""
        if self._closed or self._receipt is not None:
            raise RuntimeError("native all-hit publication lifecycle is not idle")
        if self._switch is not None and self._switch is not switch:
            raise RuntimeError("native all-hit switch ownership changed")
        frame = getattr(switch, "_qwen4_pipe_event", None)
        native = frame[0] if type(frame) is tuple and len(frame) == 2 else None
        if (
            count != _ACTIVE
            or getattr(native, "QWEN_ALL_HIT_ABI", None) != ABI
            or not callable(getattr(native, "AllHitAttempt", None))
            or getattr(switch, "training", True)
            or getattr(switch, "_prefill_prefetch_enabled", None) is not False
            or getattr(switch, "_prefetch_ticket", None) is not None
        ):
            self._stats["ineligible"] += 1
            return False
        session = getattr(switch, "_moespresso_pooled_decode_session", None)
        if (
            type(sequence) is not int
            or not 0 < sequence <= 0xFFFFFFFF
            or frame[1] != sequence
            or session is None
            or not session.active
            or not session._domain_claimed
            or not session._gate_bound
            or session._gate_mod is not native
            or session._executor is not self._executor
            or session._owner_thread == threading.get_ident()
        ):
            raise RuntimeError("native all-hit sequence or active ownership differs")
        if cancelled is not None and cancelled():
            raise CancelledError("native all-hit was cancelled before publication")
        pools = (switch.gate_proj.pool, switch.up_proj.pool, switch.down_proj.pool)
        if len({id(pool) for pool in pools}) != 3:
            self._stats["ineligible"] += 1
            return False
        deadline = time.monotonic() + self._timeout_seconds
        attempt = None
        initial_storage = None
        initial_epochs = None
        try:
            while True:
                if cancelled is not None and cancelled():
                    raise CancelledError("native all-hit was cancelled between native slices")
                with self._locked(pools):
                    if switch._qwen4_pipe_event != frame:
                        raise RuntimeError("native all-hit event ownership changed while waiting")
                    current_pools = (
                        switch.gate_proj.pool,
                        switch.up_proj.pool,
                        switch.down_proj.pool,
                    )
                    if any(a is not b for a, b in zip(pools, current_pools, strict=True)):
                        raise RuntimeError("native all-hit pool ownership changed while waiting")
                    if (
                        initial_storage is not None
                        and tuple(_storage(pool) for pool in pools) != initial_storage
                    ):
                        raise RuntimeError("native all-hit pool storage changed while waiting")
                    if initial_epochs is not None and _epochs(pools) != initial_epochs:
                        raise RuntimeError("native all-hit pool membership changed while waiting")
                    if not all(_eligible_pool(pool) for pool in pools):
                        if attempt is not None:
                            raise RuntimeError(
                                "native all-hit pool eligibility changed while waiting"
                            )
                        self._stats["ineligible"] += 1
                        return False
                    self._switch = switch
                    maps = self._prepare_maps(pools)
                    if initial_storage is None:
                        initial_storage = self._storage_key
                        initial_epochs = self._map_key
                    elif initial_storage != self._storage_key:
                        raise RuntimeError("native all-hit pool storage changed while waiting")
                    destinations = self._prepare_destinations(switch)
                    real = switch._ring_np
                    owner = getattr(switch, "_ring_buf", None)
                    if (
                        type(real) is not np.ndarray
                        or real.dtype != np.uint32
                        or real.shape != (18,)
                        or not real.flags.c_contiguous
                        or not real.flags.aligned
                        or owner is None
                    ):
                        raise RuntimeError("native all-hit actual route ring differs")
                    if self._ring is None:
                        self._ring, self._ring_owner = real, owner
                    elif self._ring is not real or self._ring_owner is not owner:
                        raise RuntimeError("native all-hit actual route ring ownership changed")
                    if cancelled is not None and cancelled():
                        raise CancelledError("native all-hit was cancelled before native slice")
                    remaining_ns = int((deadline - time.monotonic()) * 1_000_000_000)
                    if remaining_ns <= 0:
                        self._stats["pending"] += 1
                        self._stats["timed_out"] += 1
                        raise TimeoutError(
                            f"ring seq {sequence} not observed within "
                            f"{self._timeout_seconds}s; GPU export never became host-visible"
                        )
                    if attempt is None:
                        ring = real.view()
                        ring.flags.writeable = False
                        attempt = native.AllHitAttempt(
                            ring,
                            maps,
                            destinations,
                            sequence,
                            tuple(pool.capacity for pool in pools),
                        )
                        self._stats["native_calls"] += 1
                    frontier = native.signaled_value()
                    self._stats["poll_slices"] += 1
                    status = attempt.poll(min(_POLL_SLICE_NS, remaining_ns))
                    if status == "PUBLISHED":
                        self._record_published(sequence, native, real, maps, destinations, pools)
                    elif status in ("MISS", "PENDING"):
                        if native.signaled_value() != frontier:
                            raise RuntimeError(
                                "native all-hit fallback advanced the release frontier"
                            )
                        if status == "MISS":
                            self._stats["miss"] += 1
                        else:
                            self._stats["poll_yields"] += 1
                    else:
                        raise RuntimeError("native all-hit returned an unknown status")
                if cancelled is not None and cancelled():
                    raise CancelledError("native all-hit was cancelled after native slice")
                if status != "PENDING":
                    return status == "PUBLISHED"
        finally:
            if attempt is not None:
                attempt.close()

    def _record_published(self, sequence, native, real, maps, destinations, pools):
        """Protect published rows and bind their receipt while all pool locks are held."""
        self._stats["published"] += 1
        ids = real[8:].copy()
        if (
            int(real[0]) != sequence
            or int(real[1]) != _checksum(ids, sequence)
            or len(set(ids.tolist())) != _ACTIVE
            or np.any(ids >= _EXPERTS)
            or native.signaled_value() != sequence
        ):
            raise RuntimeError("native all-hit published receipt differs from actual ring")
        slots = maps[:, ids].copy()
        if not np.array_equal(np.stack(destinations), slots):
            raise RuntimeError("native all-hit published different projection slots")
        active = set(int(value) for value in ids)
        for pool in pools:
            pool._demand_protect = set(active)
        self._receipt = {
            "sequence": sequence,
            "thread": threading.get_ident(),
            "ids": ids,
            "slots": slots,
            "active": active,
            "consumed": False,
        }

    def suppress(self, switch, source_ids) -> bool:
        """Consume one native publication after normal shared-reader accounting."""
        receipt = self._receipt
        if receipt is None:
            return False
        if (
            switch is not self._switch
            or receipt["consumed"]
            or receipt["thread"] != threading.get_ident()
            or type(source_ids) is not np.ndarray
            or source_ids.dtype != np.uint32
            or source_ids.shape != (_ACTIVE,)
            or not np.array_equal(source_ids, receipt["ids"])
        ):
            raise RuntimeError("native all-hit duplicate publication receipt mismatched")
        if switch._ring_np is not self._ring or switch._ring_buf is not self._ring_owner:
            raise RuntimeError("native all-hit actual route ring ownership changed")
        pools = (switch.gate_proj.pool, switch.up_proj.pool, switch.down_proj.pool)
        if any(a is not b for a, b in zip(pools, self._pools, strict=True)):
            raise RuntimeError("native all-hit pool ownership changed after publication")
        with self._locked(pools):
            if (
                not all(_eligible_pool(pool) for pool in pools)
                or tuple(_storage(pool) for pool in pools) != self._storage_key
                or _epochs(pools) != self._map_key
            ):
                raise RuntimeError("native all-hit pool storage changed after publication")
            self._prepare_maps(pools)
            destinations = self._prepare_destinations(switch)
            if not np.array_equal(np.stack(destinations), receipt["slots"]):
                raise RuntimeError("native all-hit slot destinations changed before accounting")
            for projection, pool in enumerate(pools):
                if pool._demand_protect != receipt["active"]:
                    raise RuntimeError("native all-hit lost demand protection")
                for expert, slot in zip(
                    source_ids.tolist(), receipt["slots"][projection].tolist(), strict=True
                ):
                    if pool._slot_of.get(expert) != slot or pool._expert_at[slot] != expert:
                        raise RuntimeError("native all-hit selected slot residency changed")
            receipt["consumed"] = True
            self._stats["suppressed"] += 1
        return True

    def finish(self, *, completed: bool) -> None:
        """Clear per-call ownership on both normal and exceptional reader exit."""
        try:
            if completed and self._receipt is not None and not self._receipt["consumed"]:
                raise RuntimeError("native all-hit original reader skipped its publication")
        finally:
            self._receipt = None

    def snapshot(self) -> dict[str, int]:
        return {**self._stats, "pending_receipts": int(self._receipt is not None)}

    def close(self) -> None:
        """Release cached buffer owners after the shared worker has drained."""
        if self._receipt is not None:
            raise RuntimeError("native all-hit cannot close an active publication")
        self._closed = True
        self._switch = self._pools = self._storage_key = None
        self._maps = self._map_key = self._memberships = self._backpoints = None
        self._bufs = self._destinations = None
        self._ring = self._ring_owner = None
        self._executor = None
