"""Package-backed routed expert execution for Qwen4."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from moespresso.package.bundle import (
    IQK_CODEC,
    KQUANT_CODEC,
    MXFP4_CODEC,
)
from moespresso.runtime.expert_index import ExpertIndex
from moespresso.runtime.expert_slot_pool import BundleRowCache
from moespresso.runtime.pooled_switchglu import (
    PooledCombinedGateUpKQuantLinear,
    PooledIqkSwitchLinear,
    PooledKQuantSwitchLinear,
    PooledSwitchGLU,
    _PIPELINE_EXECUTOR,
    _RING_TIMEOUT,
)
from moespresso.runtime.qwen4.native_publication import (
    Qwen4NativePublication,
)
from moespresso.runtime.ssd_streaming_build import (
    SSDStreamingBuildError,
    build_pooled_switchglu,
)


_QWEN4_ROUTED_TOP_K = 10
_QWEN4_SOURCE_EXPERTS = 512


class _Qwen4SwiGLU:
    def __call__(self, up: mx.array, gate: mx.array) -> mx.array:
        return nn.silu(gate) * up


class Qwen4ZeroPaddedDownProjection(nn.Module):
    """Present a logical down width while serving a wider IQ_K row.

    The encoder pads the released 640-wide down rows to 768 weights. The
    runtime appends exactly zero activation lanes before every kernel call;
    callers and the surrounding SwitchGLU continue to see the logical width.
    """

    def __init__(self, projection: PooledIqkSwitchLinear, *, logical_in_features: int):
        super().__init__()
        stored = int(projection.in_features)
        logical = int(logical_in_features)
        if logical <= 0 or stored <= logical:
            raise ValueError("padded Qwen4 down geometry must widen a positive input")
        self.projection = projection
        self.pool = projection.pool
        self.codec = projection.codec
        self.bits = projection.bits
        self.member = projection.member
        self.num_experts = projection.num_experts
        self.out_features = projection.out_features
        self.in_features = logical
        self.stored_in_features = stored
        self.zero_padding = stored - logical

    @property
    def matmul_slot_calls(self) -> int:
        return int(self.projection.matmul_slot_calls)

    @matmul_slot_calls.setter
    def matmul_slot_calls(self, value: int) -> None:
        self.projection.matmul_slot_calls = int(value)

    @property
    def matmul_slot_elements(self) -> int:
        return int(self.projection.matmul_slot_elements)

    @matmul_slot_elements.setter
    def matmul_slot_elements(self, value: int) -> None:
        self.projection.matmul_slot_elements = int(value)

    def _operand(self, value: mx.array) -> mx.array:
        if value.shape[-1] != self.in_features:
            raise ValueError(
                f"Qwen4 down input width {value.shape[-1]} does not match "
                f"logical width {self.in_features}"
            )
        zeros = mx.zeros((*value.shape[:-1], self.zero_padding), dtype=value.dtype)
        return mx.concatenate((value, zeros), axis=-1)

    def __call__(self, value, indices, *, sorted_indices: bool = False):
        return self.projection(
            self._operand(value),
            indices,
            sorted_indices=sorted_indices,
        )

    def matmul_slots(self, value, indices, *, sorted_indices: bool = False):
        return self.projection.matmul_slots(
            self._operand(value),
            indices,
            sorted_indices=sorted_indices,
        )

    def sorted_matmul_range(self, value, indices, start: int, rows: int):
        return self.projection.sorted_matmul_range(
            self._operand(value),
            indices,
            start,
            rows,
        )


class Qwen4PooledSwitchGLU(PooledSwitchGLU):
    """Qwen4 executor preserving the trunk activation dtype."""

    def assert_quiescent(self) -> None:
        """Fail unless no loader, prefetch or growth can still write slots.

        The owning graph calls this only after request execution has stopped.
        It does not cancel work: closing an active executor is a lifecycle bug,
        so teardown refuses before changing any shared pool state.
        """
        if self._prefetch_ticket is not None:
            raise RuntimeError("Qwen4 expert executor has an undrained prefetch ticket")
        pools = self._unique_projection_pools(lockstep=True)
        pool_locks = [pool._bk_lock for pool in pools]
        row_caches = []
        seen_caches = set()
        for pool in pools:
            cache = pool.row_cache
            if cache is not None and id(cache) not in seen_caches:
                seen_caches.add(id(cache))
                row_caches.append(cache)
        for lock in pool_locks:
            lock.acquire()
        try:
            problems = []
            for pool in pools:
                if getattr(pool, "_staging_owner", None) is not None:
                    problems.append(f"{pool.projection}:verification-pending")
                if pool._loads_inflight:
                    problems.append(f"{pool.projection}:loads={pool._loads_inflight}")
                if pool._prefetch_inflight:
                    problems.append(f"{pool.projection}:prefetch={pool._prefetch_inflight}")
                if pool._growth_pending:
                    problems.append(f"{pool.projection}:growth-pending")
                if pool._prefetch_reserved:
                    problems.append(f"{pool.projection}:reserved={len(pool._prefetch_reserved)}")
            if problems:
                raise RuntimeError("Qwen4 expert executor is not quiescent: " + ", ".join(problems))
            for cache in row_caches:
                with cache._lock:
                    if cache._inflight:
                        raise RuntimeError("Qwen4 expert executor has an active bundle-row read")
        finally:
            for lock in reversed(pool_locks):
                lock.release()

    def close(self) -> None:
        """Release expert slots after the owning graph has become quiescent."""
        if getattr(self, "_qwen4_closed", False):
            return
        self.assert_quiescent()
        publication = getattr(self, "_qwen4_native_publication", None)
        if publication is not None:
            publication.close()
        object.__setattr__(self, "_qwen4_closed", True)
        object.__setattr__(self, "_qwen4_pipe_event", None)
        object.__setattr__(self, "_qwen4_pipe_event_buffers", None)
        for pool in self._unique_projection_pools(lockstep=True):
            pool._iqk_copy_plan = None
            pool.iqk = None
            pool._iqk_views.clear()
            pool._iqk_slot_nbytes.clear()
            pool.packed = None
            pool.weight = None
            pool.scales = None
            pool._packed_view = None
            pool._scales_view = None
            pool._slot_of.clear()
            pool._expert_at.clear()
            if pool.row_cache is not None:
                with pool.row_cache._lock:
                    pool.row_cache._rows.clear()
        self._mxfp4_kernel_cache.clear()
        self._pipe_buf_cache.clear()
        cache = getattr(self, "_qwen4_pipe_buf_cache", None)
        if cache is not None:
            cache.clear()

    def _qwen4_pipe_bufs(self, width: int, *, create: bool) -> tuple:
        """Return Qwen's independent persistent projection-slot buffers."""
        cache = getattr(self, "_qwen4_pipe_buf_cache", None)
        if cache is None:
            cache = {}
            self._qwen4_pipe_buf_cache = cache
        bufs = cache.get(width)
        if bufs is None:
            if not create:
                raise RuntimeError("Qwen4 pipeline slot buffers were not built")
            arrays = tuple(mx.array(np.zeros(width, dtype=np.uint32)) for _ in range(3))
            mx.eval(*arrays)
            bufs = tuple(item for array in arrays for item in (array, memoryview(array).cast("B")))
            cache[width] = bufs
        return bufs

    def _ring_install_body(self, seq: int, K: int, *, cancelled=None) -> None:
        """Publish eligible resident slots natively before shared demand accounting."""
        publication = getattr(self, "_qwen4_native_publication", None)
        if publication is None:
            publication = Qwen4NativePublication(_PIPELINE_EXECUTOR, timeout_seconds=_RING_TIMEOUT)
            object.__setattr__(self, "_qwen4_native_publication", publication)
        completed = False
        try:
            publication.begin(self, seq, K, cancelled=cancelled)
            super()._ring_install_body(seq, K, cancelled=cancelled)
            completed = True
        finally:
            publication.finish(completed=completed)

    def _publish_pipe_slots(self, source_ids: np.ndarray) -> None:
        """Publish independently mapped Qwen projection slots after demand ensure."""
        publication = getattr(self, "_qwen4_native_publication", None)
        if publication is not None and publication.suppress(self, source_ids):
            return
        if (
            not isinstance(source_ids, np.ndarray)
            or source_ids.dtype != np.uint32
            or source_ids.ndim != 1
        ):
            raise TypeError("Qwen4 pipeline source ids must be a one-dimensional uint32 array")
        width = int(source_ids.size)
        _gate, gate_view, _up, up_view, _down, down_view = self._qwen4_pipe_bufs(
            width, create=False
        )
        projection_pools = (
            self.gate_proj.pool,
            self.up_proj.pool,
            self.down_proj.pool,
        )
        pools = []
        for pool in projection_pools:
            if all(pool is not existing for existing in pools):
                pools.append(pool)
        locks = [pool._bk_lock for pool in pools]
        for lock in locks:
            lock.acquire()
        try:
            slot_sets: list[list[int]] = [[], [], []]
            for source_id in source_ids.tolist():
                expert = int(source_id)
                for slots, pool in zip(slot_sets, projection_pools, strict=True):
                    slot = pool._slot_of.get(expert)
                    if (
                        slot is None
                        or slot < 0
                        or slot >= len(pool._expert_at)
                        or pool._expert_at[slot] != expert
                        or expert in pool._prefetch_reserved
                        or expert not in pool._demand_protect
                    ):
                        raise RuntimeError(
                            "Qwen4 pipeline demand residency was not fully published"
                        )
                    slots.append(slot)
            for view, slots in zip((gate_view, up_view, down_view), slot_sets, strict=True):
                view[:] = np.asarray(slots, dtype=np.uint32).tobytes()
        finally:
            for lock in reversed(locks):
                lock.release()

    def build_pipelined(self, value, indices, *, event_gate=None) -> mx.array:
        """Build the generic Qwen routed graph from independently published slots."""
        width = int(indices.shape[-1])
        bufs = self._qwen4_pipe_bufs(width, create=True)
        gate_buf, _gate_view, up_buf, _up_view, down_buf, _down_view = bufs
        self.pipelined_layers += 1
        self.total_calls += 1
        self.decode_calls += 1
        self.total_token_layers += 1
        if self._all_iqk:
            self._record_iqk_route(int(indices.size))
        operand = mx.expand_dims(value, (-2, -3))
        object.__setattr__(self, "_qwen4_pipe_event", None)
        object.__setattr__(self, "_qwen4_pipe_event_buffers", None)
        if event_gate is not None:
            gate_mod, token, sequence = event_gate
            operand = gate_mod.gate(operand, token, sequence)
            object.__setattr__(self, "_qwen4_pipe_event", (gate_mod, sequence))
            object.__setattr__(self, "_qwen4_pipe_event_buffers", bufs)
        gate_slots = gate_buf.reshape(indices.shape)
        up_slots = up_buf.reshape(indices.shape)
        down_slots = down_buf.reshape(indices.shape)
        up = self.up_proj.matmul_slots(operand, up_slots, sorted_indices=False)
        gate = self.gate_proj.matmul_slots(operand, gate_slots, sorted_indices=False)
        output = self.down_proj.matmul_slots(
            self.activation(up, gate), down_slots, sorted_indices=False
        ).squeeze(-2)
        return output if output.dtype == value.dtype else output.astype(value.dtype)

    def try_full_resident_decode(self, value, indices) -> mx.array | None:
        """Use the no-sync decode path for one row when residency proves it."""
        if getattr(self, "_qwen4_closed", False):
            raise RuntimeError("Qwen4 expert executor is closed")
        rows = 1
        for dimension in value.shape[:-1]:
            rows *= int(dimension)
        if self.training or rows != 1 or not self._barrier_free_decode_ready():
            return None
        self.total_calls += 1
        self.decode_calls += 1
        self.total_token_layers += 1
        output = self.build_barrier_free_decode(
            value,
            indices.astype(mx.uint32),
        )
        return output if output.dtype == value.dtype else output.astype(value.dtype)

    def __call__(self, value, indices, *, load_ticket=None) -> mx.array:
        if getattr(self, "_qwen4_closed", False):
            raise RuntimeError("Qwen4 expert executor is closed")
        output = super().__call__(value, indices, load_ticket=load_ticket)
        return output if output.dtype == value.dtype else output.astype(value.dtype)


class Qwen4PaddedPooledSwitchGLU(Qwen4PooledSwitchGLU):
    """Pooled IQ_K SwitchGLU for the released Qwen expert geometries."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.iqk_two_dispatch_calls = 0
        self.iqk_two_dispatch_routes = 0
        self.iqk_bounded_two_dispatch_calls = 0
        self.iqk_bounded_two_dispatch_routes = 0
        self.iqk_bounded_two_dispatch_gate_up_slot_mismatch_fallbacks = 0
        self._iqk_two_dispatch_ready_cached: bool | None = None
        self.packed_prefill_calls = 0
        self.packed_prefill_pairs = 0

    def _call_iqk_full_resident(self, x, indices) -> mx.array:
        members = tuple(self.members.get(name) for name in ("gate_proj", "up_proj", "down_proj"))
        if (
            x.ndim != 3
            or x.shape[0] != 1
            or indices.size < self._iqk_sorted_threshold()
            or self.hidden_size != 2560
            or self.intermediate_size != 640
            or members[0] != members[1]
            or any(member not in ("iq2_k", "iq2_ks", "iq3_k") for member in members)
            or not isinstance(self.down_proj, Qwen4ZeroPaddedDownProjection)
            or self.down_proj.stored_in_features != 768
        ):
            return super()._call_iqk_full_resident(x, indices)
        pools = (self.gate_proj.pool, self.up_proj.pool, self.down_proj.pool)
        if not all(pool.slot_table_is_identity() for pool in pools):
            return super()._call_iqk_full_resident(x, indices)
        from moespresso.runtime.qwen4.prefill_packed import packed_gate_up
        from moespresso.runtime.qwen4.prefill_packed_down import packed_down

        activation = packed_gate_up(pools[0].iqk, pools[1].iqk, x, indices)
        output = packed_down(pools[2].iqk, activation, indices)
        self.packed_prefill_calls += 1
        self.packed_prefill_pairs += int(indices.size)
        return output

    @staticmethod
    def _iqk_sorted_parts() -> int:
        """Use power-of-two output ranges for released 640/2,560 widths."""
        return 10

    def _iqk_two_dispatch_ready(self) -> bool:
        ready = self._iqk_two_dispatch_ready_cached
        if ready is not None:
            return ready
        codec_tuple = tuple(
            self.members.get(projection) for projection in ("gate_proj", "up_proj", "down_proj")
        )
        ready = bool(
            self._all_iqk
            and set(self.members) == {"gate_proj", "up_proj", "down_proj"}
            and self.hidden_size == 2560
            and self.intermediate_size == 640
            and _QWEN4_ROUTED_TOP_K <= self.num_experts <= _QWEN4_SOURCE_EXPERTS
            and isinstance(self.down_proj, Qwen4ZeroPaddedDownProjection)
            and self.down_proj.stored_in_features == 768
        )
        if ready:
            try:
                from mlx_iqk.routed import (
                    SUPPORTED_CODEC_TUPLES,
                    down_reduce,
                    gate_up_swiglu,
                    routed_moe,
                )
            except ImportError:
                ready = False
            else:
                ready = bool(
                    callable(routed_moe)
                    and callable(gate_up_swiglu)
                    and callable(down_reduce)
                    and codec_tuple in SUPPORTED_CODEC_TUPLES
                )
        self._iqk_two_dispatch_ready_cached = ready
        return ready

    def _bounded_iqk_slot_plan(
        self,
        source_ids: np.ndarray,
    ) -> tuple[object, object, object, np.ndarray, np.ndarray, np.ndarray]:
        """Snapshot every selected projection slot after demand publication."""
        pools = self._projection_pools_lockstep()
        if len(pools) != 3:
            raise RuntimeError("Qwen4 bounded IQ_K requires three projection pools")
        gate_pool = self.gate_proj.pool
        up_pool = self.up_proj.pool
        down_pool = self.down_proj.pool
        if tuple(map(id, pools)) != tuple(map(id, (gate_pool, up_pool, down_pool))):
            raise RuntimeError("Qwen4 bounded IQ_K projection pool order is inconsistent")

        locks = [pool._bk_lock for pool in pools]
        for lock in locks:
            lock.acquire()
        try:
            modules = (gate_pool.iqk, up_pool.iqk, down_pool.iqk)
            if any(module is None for module in modules):
                raise RuntimeError("Qwen4 bounded IQ_K projection storage is unavailable")
            if len({int(module.num_experts) for module in modules}) != 1:
                raise RuntimeError("Qwen4 bounded IQ_K projection slot counts differ")

            gate_slots = []
            up_slots = []
            down_slots = []
            for source_id in source_ids.tolist():
                expert = int(source_id)
                selected_slots = []
                for pool, module in zip(pools, modules, strict=True):
                    slot = pool._slot_of.get(expert)
                    if (
                        slot is None
                        or slot < 0
                        or slot >= int(module.num_experts)
                        or slot >= len(pool._expert_at)
                        or pool._expert_at[slot] != expert
                        or expert in pool._prefetch_reserved
                        or expert not in pool._demand_protect
                    ):
                        raise RuntimeError(
                            "Qwen4 bounded IQ_K demand residency was not fully published"
                        )
                    selected_slots.append(slot)
                gate_slot, up_slot, down_slot = selected_slots
                gate_slots.append(gate_slot)
                up_slots.append(up_slot)
                down_slots.append(down_slot)

            return (
                *modules,
                np.asarray(gate_slots, dtype=np.uint32),
                np.asarray(up_slots, dtype=np.uint32),
                np.asarray(down_slots, dtype=np.uint32),
            )
        finally:
            for lock in reversed(locks):
                lock.release()

    def _bounded_iqk_incumbent_weighted_decode(
        self,
        value,
        source_indices,
        scores,
        gate_slots: np.ndarray,
        up_slots: np.ndarray,
        down_slots: np.ndarray,
    ) -> mx.array:
        """Use the incumbent three-GEMV path without repeating demand ensure."""
        operand = mx.expand_dims(value, (-2, -3))
        up = self.up_proj.matmul_slots(
            operand,
            mx.array(up_slots.reshape(source_indices.shape), dtype=mx.uint32),
        )
        gate = self.gate_proj.matmul_slots(
            operand,
            mx.array(gate_slots.reshape(source_indices.shape), dtype=mx.uint32),
        )
        expert_outputs = self.down_proj.matmul_slots(
            self.activation(up, gate),
            mx.array(down_slots.reshape(source_indices.shape), dtype=mx.uint32),
        ).squeeze(-2)
        if expert_outputs.dtype != value.dtype:
            expert_outputs = expert_outputs.astype(value.dtype)
        from moespresso.runtime.qwen4.moe import expert_major_weighted_sum

        return expert_major_weighted_sum(expert_outputs, scores, source_indices)

    def _try_bounded_weighted_decode(
        self,
        value,
        source_indices,
        scores,
    ) -> mx.array | None:
        """Run both routed stages after one bounded-pool demand ensure."""
        if int(source_indices.size) != _QWEN4_ROUTED_TOP_K:
            return None
        if source_indices.shape != scores.shape:
            return None

        sync_started = time.perf_counter()
        source_ids = np.asarray(source_indices).reshape(-1)
        self.index_resync_calls += 1
        self.index_resync_seconds += time.perf_counter() - sync_started
        active = {int(expert) for expert in source_ids.tolist()}
        capacity = min(pool.capacity for pool in self._projection_pools_lockstep())
        if len(active) != _QWEN4_ROUTED_TOP_K or len(active) > capacity:
            return None

        self.total_calls += 1
        self.decode_calls += 1
        self.total_token_layers += 1
        self.total_unique_active_experts += len(active)
        self.max_unique_active_experts = max(
            self.max_unique_active_experts,
            len(active),
        )
        self.seen_experts.update(active)
        self.decode_seen_experts.update(active)
        self.direct_calls += 1
        # This is the ordinary bounded-pool demand barrier. It waits for every
        # selected projection load and publishes demand protection before the
        # joint slot-map snapshot below.
        self._ensure_projection_pools(active)
        gate, up, down, gate_slots, up_slots, down_slots = self._bounded_iqk_slot_plan(source_ids)
        if not np.array_equal(gate_slots, up_slots):
            self.iqk_bounded_two_dispatch_gate_up_slot_mismatch_fallbacks += 1
            self._record_iqk_route(int(source_indices.size))
            return self._bounded_iqk_incumbent_weighted_decode(
                value,
                source_indices,
                scores,
                gate_slots,
                up_slots,
                down_slots,
            )

        from mlx_iqk.routed import down_reduce, gate_up_swiglu

        # routed_moe sorts its single lookup-id array. Bounded slot ids are not
        # source-monotone, and sorting them would change Qwen's BF16 reduction
        # order. Sort source experts here, then pass the corresponding gate/up
        # and down lookup slots to the two public stages separately.
        source_order = np.argsort(source_ids, kind="stable")
        order = mx.array(source_order, dtype=mx.uint32)
        sorted_scores = scores.reshape(-1)[order]
        activation = gate_up_swiglu(
            gate,
            up,
            value,
            mx.array(gate_slots[source_order], dtype=mx.uint32),
        )
        output = down_reduce(
            down,
            activation,
            mx.array(down_slots[source_order], dtype=mx.uint32),
            sorted_scores,
        ).reshape(*value.shape[:-1], self.hidden_size)

        self.iqk_two_dispatch_calls += 1
        self.iqk_two_dispatch_routes += int(source_indices.size)
        self.iqk_bounded_two_dispatch_calls += 1
        self.iqk_bounded_two_dispatch_routes += int(source_indices.size)
        return output

    def _full_resident_weighted_decode(self, value, source_indices, scores) -> mx.array:
        """Run the existing two-stage full-resident IQ_K decode."""
        from mlx_iqk.routed import routed_moe

        self.total_calls += 1
        self.decode_calls += 1
        self.total_token_layers += 1
        self.barrier_free_decode_calls += 1
        self.iqk_two_dispatch_calls += 1
        self.iqk_two_dispatch_routes += int(source_indices.size)
        return routed_moe(
            self.gate_proj.pool.iqk,
            self.up_proj.pool.iqk,
            self.down_proj.pool.iqk,
            value,
            source_indices,
            scores,
        )

    def try_resident_weighted_decode(
        self,
        value,
        source_indices,
        scores,
    ) -> mx.array | None:
        """Attempt only the proven full-resident two-stage IQ_K decode."""
        if getattr(self, "_qwen4_closed", False):
            raise RuntimeError("Qwen4 expert executor is closed")
        rows = 1
        for dimension in value.shape[:-1]:
            rows *= int(dimension)
        if (
            self.training
            or rows != 1
            or not self._iqk_two_dispatch_ready()
            or not self._barrier_free_decode_ready()
            or self._iqk_decode_identity_cached is not True
        ):
            return None
        return self._full_resident_weighted_decode(value, source_indices, scores)

    def try_full_resident_weighted_decode(
        self,
        value,
        source_indices,
        scores,
    ) -> mx.array | None:
        """Return the reduced routed row when the exact IQ_K path engages."""
        if getattr(self, "_qwen4_closed", False):
            raise RuntimeError("Qwen4 expert executor is closed")
        rows = 1
        for dimension in value.shape[:-1]:
            rows *= int(dimension)
        if self.training or rows != 1 or not self._iqk_two_dispatch_ready():
            return None

        if not self._barrier_free_decode_ready() or self._iqk_decode_identity_cached is not True:
            return self._try_bounded_weighted_decode(
                value,
                source_indices,
                scores,
            )

        return self._full_resident_weighted_decode(value, source_indices, scores)

    def build_barrier_free_decode(self, value, indices) -> mx.array:
        self.barrier_free_decode_calls += 1
        if not self._all_iqk:
            return super().build_barrier_free_decode(value, indices)
        self._record_iqk_route(int(indices.size))
        if self._iqk_decode_identity_cached:
            gate_indices = up_indices = down_indices = indices
        else:
            gate_indices = self.gate_proj.pool.remap_ondevice(indices)
            up_indices = self.up_proj.pool.remap_ondevice(indices)
            down_indices = self.down_proj.pool.remap_ondevice(indices)
        operand = mx.expand_dims(value, (-2, -3))
        up = self.up_proj.matmul_slots(operand, up_indices)
        gate = self.gate_proj.matmul_slots(operand, gate_indices)
        output = self.down_proj.matmul_slots(
            self.activation(up, gate),
            down_indices,
        )
        return output.squeeze(-2)

    def build_compact_barrier_free_decode(self, value, indices) -> mx.array:
        return self.build_barrier_free_decode(value, indices)


def _qwen4_projection_geometry(
    *,
    index: ExpertIndex,
    layer: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[str, int]:
    expected = {
        "gate_proj": (intermediate_size, hidden_size),
        "up_proj": (intermediate_size, hidden_size),
        "down_proj": (hidden_size, intermediate_size),
    }
    codecs = set()
    down_stored = intermediate_size
    for projection, (out_features, logical_in) in expected.items():
        geometry = index.geometry(layer=layer, projection=projection)
        codecs.add(geometry.codec)
        if geometry.out_features != out_features:
            raise SSDStreamingBuildError(
                f"layer {layer} {projection}: stored output width "
                f"{geometry.out_features} != logical width {out_features}"
            )
        if geometry.codec == KQUANT_CODEC:
            block_bytes = int(geometry.bytes_per_block or 0)
            block_weights = int(geometry.weights_per_block or 0)
            if block_bytes <= 0 or block_weights <= 0 or geometry.packed_cols % block_bytes:
                raise SSDStreamingBuildError(
                    f"layer {layer} {projection}: invalid Q8_0 row geometry"
                )
            stored_in = geometry.packed_cols // block_bytes * block_weights
            if geometry.kquant_codec != "q8_0":
                raise SSDStreamingBuildError(
                    f"layer {layer} {projection}: Qwen4 fallback must be q8_0"
                )
        elif geometry.codec == IQK_CODEC:
            stored_in = int(geometry.in_features or 0)
            if geometry.iqk_codec not in {"iq1_s_r4", "iq2_k", "iq2_ks", "iq3_k"}:
                raise SSDStreamingBuildError(
                    f"layer {layer} {projection}: unsupported IQ_K member {geometry.iqk_codec!r}"
                )
        else:
            raise SSDStreamingBuildError(
                f"layer {layer} {projection}: unsupported Qwen4 expert codec {geometry.codec!r}"
            )
        expected_stored = (
            768
            if projection == "down_proj"
            and geometry.codec == IQK_CODEC
            and geometry.iqk_codec in {"iq2_k", "iq2_ks", "iq3_k"}
            else logical_in
        )
        if stored_in != expected_stored:
            raise SSDStreamingBuildError(
                f"layer {layer} {projection}: stored input width {stored_in} "
                f"!= expected {expected_stored}"
            )
        if projection == "down_proj":
            down_stored = stored_in
    if len(codecs) != 1:
        raise SSDStreamingBuildError(f"layer {layer} mixes expert codec families {sorted(codecs)}")
    return next(iter(codecs)), down_stored


def build_qwen4_pooled_expert_executor(
    *,
    package_dir: str | Path,
    index: ExpertIndex,
    layer: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    capacity: int,
    eviction_policy: str = "lfu",
    spare_slots: int = 0,
) -> PooledSwitchGLU:
    """Build one Qwen4 executor without replacing the owning MoE block."""
    indexed_experts = index.num_experts_for_layer(layer)
    if indexed_experts != num_experts:
        raise SSDStreamingBuildError(
            f"layer {layer} expert count {indexed_experts} does not match "
            f"Qwen4 geometry {num_experts}"
        )
    first_codec = index.geometry(layer=layer, projection="gate_proj").codec
    if first_codec == MXFP4_CODEC:
        return build_pooled_switchglu(
            package_dir=package_dir,
            index=index,
            layer=layer,
            capacity=capacity,
            projection_dims={
                "gate_proj": (hidden_size, intermediate_size),
                "up_proj": (hidden_size, intermediate_size),
                "down_proj": (intermediate_size, hidden_size),
            },
            activation=_Qwen4SwiGLU(),
            eviction_policy=eviction_policy,
            spare_slots=spare_slots,
        )
    codec, down_stored = _qwen4_projection_geometry(
        index=index,
        layer=layer,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    resolved_capacity = min(int(capacity), indexed_experts)
    resolved_spares = min(
        int(spare_slots),
        max(0, indexed_experts - resolved_capacity),
    )
    if resolved_capacity < 1:
        raise SSDStreamingBuildError(f"layer {layer} capacity must be >= 1")
    if resolved_spares < 0:
        raise ValueError("spare_slots must be >= 0")
    root = Path(package_dir)
    if codec == KQUANT_CODEC:
        row_cache = BundleRowCache(
            package_dir=root,
            index=index,
            layer=layer,
            consumers=2,
        )
        gate = PooledCombinedGateUpKQuantLinear(
            package_dir=root,
            index=index,
            layer=layer,
            capacity=resolved_capacity,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=resolved_spares,
        )
        down = PooledKQuantSwitchLinear(
            package_dir=root,
            index=index,
            layer=layer,
            projection="down_proj",
            capacity=resolved_capacity,
            eviction_policy=eviction_policy,
            row_cache=row_cache,
            spare_slots=resolved_spares,
        )
        pooled = Qwen4PooledSwitchGLU(
            gate_proj=gate,
            up_proj=gate.up_alias,
            down_proj=down,
            activation=_Qwen4SwiGLU(),
        )
    else:
        row_cache = BundleRowCache(
            package_dir=root,
            index=index,
            layer=layer,
            consumers=3,
        )
        projections = {
            projection: PooledIqkSwitchLinear(
                package_dir=root,
                index=index,
                layer=layer,
                projection=projection,
                capacity=resolved_capacity,
                eviction_policy=eviction_policy,
                row_cache=row_cache,
                spare_slots=resolved_spares,
            )
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        down_projection = projections["down_proj"]
        if down_stored == intermediate_size:
            down = down_projection
        else:
            down = Qwen4ZeroPaddedDownProjection(
                down_projection,
                logical_in_features=intermediate_size,
            )
            if down.stored_in_features != down_stored or down.zero_padding != 128:
                raise SSDStreamingBuildError(f"layer {layer}: padded down geometry is inconsistent")
        pooled = Qwen4PaddedPooledSwitchGLU(
            gate_proj=projections["gate_proj"],
            up_proj=projections["up_proj"],
            down_proj=down,
            activation=_Qwen4SwiGLU(),
        )
    pooled.resolved_capacity = resolved_capacity
    pooled.resolved_spare_slots = resolved_spares
    return pooled
