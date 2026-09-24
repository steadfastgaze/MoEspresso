"""Explicit cache-conditioned routing over the existing projection pools."""

from contextlib import ExitStack

import mlx.core as mx

from moespresso.runtime.qwen4.cache_routing_kernel import cache_prior_route
from moespresso.runtime.qwen4.cache_routing_config import (
    CACHE_ROUTING_IDENTITY as CACHE_ROUTING_IDENTITY,
    CACHE_ROUTING_POLICIES as CACHE_ROUTING_POLICIES,
    CacheRoutingConfig,
    resolve_cache_routing,
    validate_cache_routing as validate_cache_routing,
)


class CacheRoutingProvider:
    """Supply immutable selection hints, never physical consumer addresses.

    The loader still resolves and ensures every selected expert. Old hint arrays
    remain owned by their MLX graph when pool membership changes. Busy storage
    transactions reuse the last published hint arrays without waiting for IO.
    """

    def __init__(self, switch, config: CacheRoutingConfig | None = None):
        self.config = config if config is not None else resolve_cache_routing("prefer-resident")
        self.bonus = self.config.bonus
        self.pools = tuple(switch._projection_pools_lockstep())
        self.session = switch._moespresso_pooled_decode_session
        if len(self.pools) != 3 or len({id(p) for p in self.pools}) != 3:
            raise ValueError("cache routing requires three distinct projection pools")
        for pool in self.pools:
            verification_scratch = bool(
                getattr(pool, "_verification_scratch_only", False)
                and pool.spare_slots == min(10, pool.num_experts - pool.capacity)
                and all(value is None for value in pool._expert_at[pool.capacity:])
                and all(slot < pool.capacity for slot in pool._slot_of.values())
            )
            if (pool.num_experts != 512 or pool._slot_sentinel != 512
                    or not 10 <= pool.capacity <= 512 or (pool.spare_slots and not verification_scratch)):
                raise ValueError("cache routing requires full512 pools without spare slots")
        self.calls = 0
        self.snapshot_rebuilds = 0
        self.busy_fallbacks = 0
        self.stale_snapshot_reuses = 0
        self.full_resident_calls = 0
        self._last_snapshot: tuple[mx.array, ...] | None = None

    def _published_full(self):
        return all(p.capacity == p.num_experts and len(p._slot_of) == p.num_experts
                   and not (p._growth_pending or p._loads_inflight or p._prefetch_inflight)
                   for p in self.pools)

    def is_fully_resident(self):
        """Whether compiled original routing can safely omit cache selection."""
        with ExitStack() as stack:
            for pool in self.pools:
                stack.enter_context(pool._bk_lock)
            return self._published_full()

    def snapshot(self):
        self.session.require_current_request()
        with ExitStack() as stack:
            for pool in self.pools:
                stack.enter_context(pool._bk_lock)
            if any(p._growth_pending or p._loads_inflight or p._prefetch_inflight
                   for p in self.pools):
                self.busy_fallbacks += 1
                if self._last_snapshot is not None:
                    self.stale_snapshot_reuses += 1
                    return self._last_snapshot
                return None
            if self._published_full():
                self.full_resident_calls += 1
                return None
            self.snapshot_rebuilds += sum(p._slot_table is None or p._slot_table_dirty for p in self.pools)
            snapshot = tuple(p._ensure_slot_table() for p in self.pools)
            self._last_snapshot = snapshot
            return snapshot

    def select(self, logits, maps):
        indices, scores, _changed = cache_prior_route(
            logits, maps, bonus=self.bonus, protected_routes=self.config.protected_routes,
        )
        self.calls += 1
        return indices, scores


def configure_cache_routing(
    model, policy: str, *, factor: float | None = None, protected_routes: int | None = None,
) -> None:
    """Configure once before creating request state, preserving parameter paths."""
    config = resolve_cache_routing(policy, factor=factor, protected_routes=protected_routes)
    if getattr(model, "_cache_routing_enabled", False):
        raise ValueError("cache routing is immutable after configuration")
    if policy == "off":
        return
    from moespresso.runtime.qwen4.moe import Qwen4SparseMoEBlock, Qwen4TopKRouter

    if model._coordinators or model._plain_serial_lanes:
        raise ValueError("cache routing must be configured before request state is created")
    if len(model.layers) != 48:
        raise ValueError("cache routing requires the full48 Qwen architecture")
    providers = []
    for layer in model.layers[2:]:
        block = layer.mlp
        router = getattr(block, "gate", None)
        if (type(block) is not Qwen4SparseMoEBlock or type(router) is not Qwen4TopKRouter
                or router.num_experts != 512 or router.top_k != 10
                or not router.normalize_topk or router.retained_source_ids is not None
                or router.weight.dtype != mx.bfloat16):
            raise ValueError("cache routing requires unpruned BF16 full512 top10 routers")
        providers.append((router, CacheRoutingProvider(block.experts, config)))
    # Validate every layer before publishing any configuration.
    for router, provider in providers:
        object.__setattr__(router, "_cache_routing_provider", provider)
    model.cache_identity = f"{model.cache_identity}|{config.identity}"
    object.__setattr__(model, "_cache_routing_config", config)
    object.__setattr__(model, "_cache_routing_enabled", True)


def cache_routing_stats(model):
    enabled = getattr(model, "_cache_routing_enabled", False)
    config = getattr(model, "_cache_routing_config", CacheRoutingConfig())
    totals = {"calls": 0, "snapshot_rebuilds": 0, "busy_fallbacks": 0,
              "stale_snapshot_reuses": 0,
              "full_resident_calls": 0}
    if enabled:
        for layer in model.layers[2:]:
            provider = layer.mlp.gate._cache_routing_provider
            for key in totals:
                totals[key] += getattr(provider, key)
    return {"policy": config.policy,
            "identity": config.identity,
            "cache_factor": config.cache_factor if enabled else 1,
            "protected_routes": config.protected_routes if enabled else 10,
            **totals}
