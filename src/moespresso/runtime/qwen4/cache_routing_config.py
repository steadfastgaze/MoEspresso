"""Load-time configuration for cache-conditioned Qwen routing."""

from dataclasses import dataclass
import math


CACHE_ROUTING_POLICIES = ("auto", "off", "prefer-resident")
DEFAULT_CACHE_FACTOR = 2.0
MIN_CACHE_FACTOR = 1.0
MAX_CACHE_FACTOR = 8.0
DEFAULT_PROTECTED_ROUTES = 2
MIN_PROTECTED_ROUTES = 0
MAX_PROTECTED_ROUTES = 3
DEFAULT_CACHE_BONUS = math.log(DEFAULT_CACHE_FACTOR)


def validate_cache_routing(policy: str) -> str:
    if not isinstance(policy, str) or policy not in CACHE_ROUTING_POLICIES:
        raise ValueError(f"cache_routing must be one of {CACHE_ROUTING_POLICIES}")
    return policy


def validate_cache_factor(value: float) -> float:
    """Require a supported finite routing multiplier."""
    valid = (
        type(value) in (float, int)
        and MIN_CACHE_FACTOR <= value <= MAX_CACHE_FACTOR
        and math.isfinite(value)
    )
    if not valid:
        raise ValueError(
            f"cache_routing_factor must be a finite multiplier from "
            f"{MIN_CACHE_FACTOR:g} to {MAX_CACHE_FACTOR:g}"
        )
    return value


def validate_protected_routes(value: int) -> int:
    if type(value) is not int or not MIN_PROTECTED_ROUTES <= value <= MAX_PROTECTED_ROUTES:
        raise ValueError(
            f"cache_routing_protected_routes must be an integer from "
            f"{MIN_PROTECTED_ROUTES} to {MAX_PROTECTED_ROUTES}"
        )
    return value


@dataclass(frozen=True)
class CacheRoutingConfig:
    policy: str = "off"
    cache_factor: float = DEFAULT_CACHE_FACTOR
    protected_routes: int = DEFAULT_PROTECTED_ROUTES

    def __post_init__(self) -> None:
        validate_cache_routing(self.policy)
        validate_cache_factor(self.cache_factor)
        validate_protected_routes(self.protected_routes)

    @property
    def enabled(self) -> bool:
        return self.policy == "prefer-resident"

    @property
    def bonus(self) -> float:
        return math.log(self.cache_factor)

    @property
    def identity(self) -> str:
        if not self.enabled:
            return "exact"
        return (
            f"cache-prior-v2:factor{self.cache_factor:.17g}:"
            f"protect{self.protected_routes}:stale-busy-published:"
            "exact-prefill:exact-layers0-1"
        )

    def load_options(self, *, include_off: bool = False) -> dict:
        """Keep default builder calls compatible with policy-only adapters."""
        if self.policy == "off":
            return {"cache_routing": "off"} if include_off else {}
        options = {"cache_routing": self.policy}
        if self.policy == "auto":
            return options
        if self.cache_factor != DEFAULT_CACHE_FACTOR:
            options["cache_routing_factor"] = self.cache_factor
        if self.protected_routes != DEFAULT_PROTECTED_ROUTES:
            options["cache_routing_protected_routes"] = self.protected_routes
        return options


def resolve_cache_routing(
    policy: str = "auto", *, factor: float | None = None, protected_routes: int | None = None,
) -> CacheRoutingConfig:
    validate_cache_routing(policy)
    if policy != "prefer-resident" and (factor is not None or protected_routes is not None):
        raise ValueError(
            "--cache-routing-factor and --cache-routing-protected-routes require "
            "--cache-routing prefer-resident"
        )
    if policy == "auto":
        return CacheRoutingConfig(policy="auto")
    return CacheRoutingConfig(
        policy=policy,
        cache_factor=DEFAULT_CACHE_FACTOR if factor is None else factor,
        protected_routes=DEFAULT_PROTECTED_ROUTES if protected_routes is None else protected_routes,
    )


def resolve_auto_cache_routing(*, bounded: bool) -> CacheRoutingConfig:
    """Resolve the Qwen4 default from the already-built expert pools."""
    if not bounded:
        return CacheRoutingConfig(policy="off")
    return CacheRoutingConfig(
        policy="prefer-resident",
        cache_factor=DEFAULT_CACHE_FACTOR,
        protected_routes=DEFAULT_PROTECTED_ROUTES,
    )


CACHE_ROUTING_IDENTITY = CacheRoutingConfig(policy="prefer-resident").identity
