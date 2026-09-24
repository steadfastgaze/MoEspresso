from contextlib import ExitStack
import argparse
from dataclasses import FrozenInstanceError
import math
import threading
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten
import pytest

from moespresso.runtime.expert_slot_pool import ExpertSlotPool
from moespresso.runtime.pooled_decode_session import PooledDecodeSession
from moespresso.runtime.qwen4.cache_routing import (
    CACHE_ROUTING_IDENTITY,
    CacheRoutingProvider,
    cache_routing_stats,
    configure_cache_routing,
    validate_cache_routing,
)
from moespresso.runtime.qwen4.moe import Qwen4SparseMoEBlock
from moespresso.runtime.qwen4.cache_routing_config import (
    resolve_auto_cache_routing,
    resolve_cache_routing,
)
from moespresso.runtime.serve import (
    _manifest_driven_backend,
    _runtime_truth_line,
    add_cache_routing_argument,
    validate_cache_routing_option,
)


def _pool(capacity=223):
    pool = object.__new__(ExpertSlotPool)
    pool.num_experts = pool._slot_sentinel = 512
    pool.capacity = capacity
    pool.spare_slots = 0
    pool._slot_of = {i: i for i in range(capacity)}
    pool._slot_table = None
    pool._slot_table_dirty = True
    pool._slot_table_identity = False
    pool.slot_table_rebuilds = 0
    pool._bk_lock = threading.Lock()
    pool._growth_pending = False
    pool._loads_inflight = pool._prefetch_inflight = 0
    return pool


def _switch(capacity=223):
    pools = tuple(_pool(capacity) for _ in range(3))
    return SimpleNamespace(_projection_pools_lockstep=lambda: pools,
                           _moespresso_pooled_decode_session=PooledDecodeSession())


def _model():
    layers = []
    for _ in range(48):
        block = Qwen4SparseMoEBlock(8, 4, 4, 512, 10)
        block.gate.weight = mx.zeros((512, 8), dtype=mx.bfloat16)
        block.experts = _switch()
        block.eval()
        layers.append(SimpleNamespace(mlp=block))
    return SimpleNamespace(layers=layers, _coordinators=set(), _plain_serial_lanes=set(),
                           cache_identity="original", _cache_routing_enabled=False)


def test_configuration_preserves_modules_weights_and_parameter_paths():
    model = _model()
    before = [tree_flatten(layer.mlp.parameters()) for layer in model.layers]
    gates = [layer.mlp.gate for layer in model.layers]
    configure_cache_routing(model, "prefer-resident")
    assert model.cache_identity == "original|" + CACHE_ROUTING_IDENTITY
    for index, (layer, old) in enumerate(zip(model.layers, before, strict=True)):
        assert layer.mlp.gate is gates[index]
        assert hasattr(layer.mlp.gate, "_cache_routing_provider") == (index >= 2)
        current = tree_flatten(layer.mlp.parameters())
        assert [k for k, _v in current] == [k for k, _v in old]
        assert all(a is b for (_k, a), (_l, b) in zip(old, current, strict=True))
    assert cache_routing_stats(model)["policy"] == "prefer-resident"
    with pytest.raises(ValueError, match="immutable"):
        configure_cache_routing(model, "off")


def test_off_configuration_preserves_original_identity_and_stats():
    model = _model()
    configure_cache_routing(model, "off", factor=None, protected_routes=None)
    assert model.cache_identity == "original" and not model._cache_routing_enabled
    assert not any(hasattr(layer.mlp.gate, "_cache_routing_provider") for layer in model.layers)
    assert cache_routing_stats(model) == {
        "policy": "off", "identity": "exact", "cache_factor": 1, "protected_routes": 10,
        "calls": 0, "snapshot_rebuilds": 0, "busy_fallbacks": 0,
        "stale_snapshot_reuses": 0, "full_resident_calls": 0,
    }


def test_configuration_fails_before_partial_publication():
    model = _model()
    model.layers[-1].mlp.gate.weight = mx.zeros((512, 8), dtype=mx.float32)
    with pytest.raises(ValueError, match="BF16"):
        configure_cache_routing(model, "prefer-resident")
    assert not model._cache_routing_enabled and model.cache_identity == "original"
    assert not any(hasattr(layer.mlp.gate, "_cache_routing_provider") for layer in model.layers)


@pytest.mark.parametrize("registry", ["_coordinators", "_plain_serial_lanes"])
def test_existing_state_owner_prevents_policy_change(registry):
    model = _model()
    getattr(model, registry).add(object())
    with pytest.raises(ValueError, match="before request"):
        configure_cache_routing(model, "prefer-resident")


@pytest.mark.parametrize("value", [None, True, 2, ""])
def test_invalid_policy_fails_closed(value):
    with pytest.raises(ValueError):
        validate_cache_routing(value)


def test_cli_option_defaults_to_bounded_qwen4_auto_resolution():
    parser = argparse.ArgumentParser()
    add_cache_routing_argument(parser)
    assert parser.parse_args([]).cache_routing == "auto"
    assert parser.parse_args([]).cache_routing_factor is None
    assert parser.parse_args([]).cache_routing_protected_routes is None
    assert parser.parse_args(["--cache-routing", "prefer-resident"]).cache_routing == "prefer-resident"
    help_text = " ".join(parser.format_help().split())
    assert "multiplier from 1 to 8 (default: 2)" in help_text
    assert "selected ten (default: 2)" in help_text
    assert parser.parse_args(["--cache-routing", "auto"]).cache_routing == "auto"
    args = parser.parse_args([
        "--cache-routing", "prefer-resident", "--cache-routing-factor", "4",
        "--cache-routing-protected-routes", "0",
    ])
    assert args.cache_routing_factor == 4 and args.cache_routing_protected_routes == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["--cache-routing-protected-routes", "4"])


@pytest.mark.parametrize("factor", [True, "2", float("nan"), float("inf"), float("-inf"),
                                   0, 0.99, 8.01, 16, 1e39, 10**1000])
def test_invalid_factor_fails_closed(factor):
    with pytest.raises(ValueError, match="factor"):
        resolve_cache_routing("prefer-resident", factor=factor)


@pytest.mark.parametrize("protected_routes", [True, -1, 4, 2.0, "2"])
def test_invalid_protection_fails_closed(protected_routes):
    with pytest.raises(ValueError, match="protected_routes"):
        resolve_cache_routing("prefer-resident", protected_routes=protected_routes)


@pytest.mark.parametrize("options", [{"factor": 2}, {"protected_routes": 2}, {"factor": 1},
                                     {"factor": 3}, {"protected_routes": 0}])
def test_controls_require_enabled_policy(options):
    with pytest.raises(ValueError, match="require.*prefer-resident"):
        resolve_cache_routing("off", **options)


def test_configuration_defaults_and_identity_are_stable():
    config = resolve_cache_routing("prefer-resident")
    assert config.cache_factor == 2 and config.protected_routes == 2
    assert config.identity == (
        "cache-prior-v2:factor2:protect2:stale-busy-published:exact-prefill:exact-layers0-1"
    )
    assert config.load_options() == {"cache_routing": "prefer-resident"}
    assert resolve_cache_routing().load_options() == {"cache_routing": "auto"}
    assert resolve_cache_routing("prefer-resident", factor=1).bonus == 0
    assert len({resolve_cache_routing("prefer-resident", factor=f).identity
                for f in (2, 2.000000000001, 2.000000000002)}) == 3
    with pytest.raises(FrozenInstanceError):
        config.cache_factor = 4


def test_explicit_factor_two_protect_two_preserves_identity_and_load_options():
    config = resolve_cache_routing("prefer-resident", factor=2, protected_routes=2)
    assert config.identity == (
        "cache-prior-v2:factor2:protect2:stale-busy-published:exact-prefill:exact-layers0-1"
    )
    assert config.load_options() == {"cache_routing": "prefer-resident"}


def test_auto_bounded_resolution_is_exactly_explicit_two_protect_two():
    auto = resolve_auto_cache_routing(bounded=True)
    explicit = resolve_cache_routing(
        "prefer-resident", factor=2, protected_routes=2,
    )
    assert auto == explicit
    assert auto.identity == explicit.identity
    assert auto.load_options() == explicit.load_options()
    assert resolve_auto_cache_routing(bounded=False).policy == "off"


def test_startup_truth_line_reports_auto_resolution():
    manifest = {"artifact_id": "pkg:abcdef0123456789"}
    bounded = SimpleNamespace(
        _moespresso_ssd_streaming_capacity=37,
        _moespresso_ssd_hotlist={},
        _moespresso_cache_routing_resolution="auto-bounded",
        _cache_routing_config=resolve_auto_cache_routing(bounded=True),
    )
    full = SimpleNamespace(
        _moespresso_ssd_streaming_capacity=512,
        _moespresso_ssd_hotlist={},
        _moespresso_cache_routing_resolution="auto-full-resident",
    )
    assert "cache_routing=auto->prefer-resident factor=2 protected_routes=2" in (
        _runtime_truth_line(bounded, manifest)
    )
    assert "cache_routing=auto->off(full-resident)" in _runtime_truth_line(full, manifest)


@pytest.mark.parametrize("factor,protected_routes", [(1, 0), (1.5, 1), (2, 2), (3, 0), (4, 3), (8, 0)])
def test_resolved_configuration_is_shared_by_providers_and_stats(factor, protected_routes):
    model = _model()
    configure_cache_routing(model, "prefer-resident", factor=factor, protected_routes=protected_routes)
    config = model._cache_routing_config
    assert model.cache_identity == "original|" + config.identity
    assert all(layer.mlp.gate._cache_routing_provider.config is config for layer in model.layers[2:])
    stats = cache_routing_stats(model)
    assert stats["cache_factor"] == factor and stats["protected_routes"] == protected_routes
    assert stats["identity"] == config.identity


@pytest.mark.parametrize("other_options", [{"factor": 1}, {"factor": 4}, {"protected_routes": 0}])
def test_composite_state_cannot_cross_routing_configurations(other_options):
    from moespresso.runtime.qwen4.model import Qwen4TextModelShell

    incumbent, other = _model(), _model()
    configure_cache_routing(incumbent, "prefer-resident")
    configure_cache_routing(other, "prefer-resident", **other_options)
    incumbent._require_open = lambda: None
    for layer in incumbent.layers:
        layer.mixer_kind = "gdn"
    state = Qwen4TextModelShell.new_state(incumbent, 1)
    assert state.cache_identity == incumbent.cache_identity
    with pytest.raises(ValueError, match="cache identity"):
        Qwen4TextModelShell._validate_state(other, state, trusted=False)


def test_provider_passes_resolved_configuration_to_kernel(monkeypatch):
    import moespresso.runtime.qwen4.cache_routing as routing

    provider = CacheRoutingProvider(_switch(), resolve_cache_routing(
        "prefer-resident", factor=4, protected_routes=0,
    ))
    calls = []

    def selector(logits, maps, **kwargs):
        calls.append((logits, maps, kwargs))
        return "ids", "scores", "changes"

    monkeypatch.setattr(routing, "cache_prior_route", selector)
    assert provider.select("logits", "maps") == ("ids", "scores")
    assert calls == [("logits", "maps", {"bonus": math.log(4), "protected_routes": 0})]
    assert provider.calls == 1


def test_manifest_backend_forwards_policy_to_runtime_builder(monkeypatch, tmp_path):
    import moespresso.runtime.build as build
    import moespresso.runtime.serve as serve

    calls = []
    manifest = {"architecture": {"family": "qwen4_exp"}}
    monkeypatch.setattr(serve, "_uses_ssd_streaming_runtime", lambda _m: False)

    def builder(actual_manifest, package, **kwargs):
        calls.append((actual_manifest, package, kwargs))
        return "model", "tokenizer"

    monkeypatch.setattr(build, "build_model", builder)
    assert _manifest_driven_backend(manifest, tmp_path, cache_routing="prefer-resident") == (
        "model", "tokenizer"
    )
    assert calls == [(manifest, tmp_path, {
        "context_limit": None, "context_limit_explicit": False, "cache_routing": "prefer-resident"
    })]
    _manifest_driven_backend(manifest, tmp_path)
    assert calls[-1][2]["cache_routing"] == "auto"
    _manifest_driven_backend(manifest, tmp_path, cache_routing="off")
    assert calls[-1][2]["cache_routing"] == "off"
    _manifest_driven_backend(
        manifest, tmp_path, cache_routing="prefer-resident", cache_routing_factor=2,
        cache_routing_protected_routes=2,
    )
    assert calls[-1][2] == {
        "context_limit": None, "context_limit_explicit": False,
        "cache_routing": "prefer-resident",
    }


@pytest.mark.parametrize("family", ["deepseek_v4", "qwen3_5_moe", "ornith"])
def test_unrelated_models_keep_off_and_reject_explicit_bias(family):
    manifest = {"architecture": {"family": family}}
    validate_cache_routing_option(manifest, "off")
    assert validate_cache_routing_option(manifest, "auto").policy == "off"
    with pytest.raises(ValueError, match="Qwen4"):
        validate_cache_routing_option(manifest, "prefer-resident")


@pytest.mark.parametrize("entrypoint", ["generate", "serve"])
@pytest.mark.parametrize("factor", [3, 4.1, 8])
def test_cli_controls_reach_loader(entrypoint, factor, monkeypatch, tmp_path):
    import moespresso.runtime.http as http
    import moespresso.runtime.serve as serve

    calls = []

    def loader(package_dir, **kwargs):
        calls.append((package_dir, kwargs))
        raise ValueError("stop after routing propagation")

    monkeypatch.setattr(serve, "load_served_model", loader)
    args = [str(tmp_path), "--cache-routing", "prefer-resident", "--cache-routing-factor", str(factor),
            "--cache-routing-protected-routes", "0"]
    assert (serve.main(args) if entrypoint == "generate" else http.main(args)) == 2
    assert len(calls) == 1
    assert calls[0][1]["cache_routing"] == "prefer-resident"
    assert calls[0][1]["cache_routing_factor"] == factor
    assert calls[0][1]["cache_routing_protected_routes"] == 0


@pytest.mark.parametrize("entrypoint", ["generate", "serve"])
@pytest.mark.parametrize("args", [
    ["--cache-routing-factor", "2"],
    ["--cache-routing-protected-routes", "0"],
    ["--cache-routing", "prefer-resident", "--cache-routing-factor", "nan"],
    ["--cache-routing", "prefer-resident", "--cache-routing-factor", "inf"],
    ["--cache-routing", "prefer-resident", "--cache-routing-factor", "1e39"],
    ["--cache-routing", "prefer-resident", "--cache-routing-factor", "0.99"],
    ["--cache-routing", "prefer-resident", "--cache-routing-factor", "8.01"],
])
def test_cli_invalid_controls_fail_before_load(entrypoint, args, monkeypatch, tmp_path):
    import moespresso.runtime.http as http
    import moespresso.runtime.serve as serve

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid routing controls reached the loader")

    monkeypatch.setattr(serve, "load_served_model", forbidden)
    main = serve.main if entrypoint == "generate" else http.main
    assert main([str(tmp_path), *args]) == 2


def test_load_served_model_preserves_routing_controls(tmp_path):
    import moespresso.runtime.serve as serve

    calls = []

    def builder(manifest, package, **kwargs):
        calls.append(kwargs)
        raise ValueError("stop after build propagation")

    with pytest.raises(ValueError, match="stop after build"):
        serve.load_served_model(
            tmp_path, manifest={"architecture": {"family": "qwen4_exp"}}, build_fn=builder,
            cache_routing="prefer-resident", cache_routing_factor=3,
            cache_routing_protected_routes=0,
        )
    assert calls == [{"cache_routing": "prefer-resident", "cache_routing_factor": 3,
                      "cache_routing_protected_routes": 0}]


@pytest.mark.parametrize("entrypoint", ["generate", "serve"])
def test_cli_rejects_controls_for_other_adapters_before_load(entrypoint, monkeypatch, tmp_path):
    import moespresso.runtime.http as http
    import moespresso.runtime.serve as serve

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsupported adapter reached the loader")

    manifest = {"architecture": {"family": "qwen3_5_moe"}}
    monkeypatch.setattr(serve, "load_served_model", forbidden)
    for module in (serve, http):
        monkeypatch.setattr(module, "_preflight_manifest_for_cli", lambda _package: manifest)
    main = serve.main if entrypoint == "generate" else http.main
    assert main([str(tmp_path), "--cache-routing", "prefer-resident",
                 "--cache-routing-factor", "4", "--cache-routing-protected-routes", "0"]) == 2


@pytest.mark.parametrize("options", [
    {"cache_routing_factor": 2},
    {"cache_routing": "prefer-resident", "cache_routing_factor": float("nan")},
    {"cache_routing": "prefer-resident", "cache_routing_protected_routes": 4},
])
def test_package_loader_rejects_invalid_controls_before_reading_index(tmp_path, options):
    from moespresso.runtime.qwen4.load import load_qwen4_iqk_package_model

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid routing controls reached package IO")

    with pytest.raises(ValueError, match="routing"):
        load_qwen4_iqk_package_model({}, tmp_path, build_index_fn=forbidden, **options)


def test_snapshot_reuses_old_arrays_without_explicit_eval(monkeypatch):
    provider = CacheRoutingProvider(_switch())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unexpected eval or synchronization")

    with provider.session.request(provider, synchronize=lambda _root: None):
        with monkeypatch.context() as patch:
            for name in ("eval", "async_eval", "synchronize"):
                patch.setattr(mx, name, forbidden)
            old = provider.snapshot()
            again = provider.snapshot()
            target = provider.pools[1]
            with target._bk_lock:
                del target._slot_of[17]
                target._slot_of[499] = 17
                target._slot_table_dirty = True
            new = provider.snapshot()
    assert all(a is b for a, b in zip(old, again, strict=True))
    assert new[0] is old[0] and new[1] is not old[1]
    assert int(old[1][499].item()) == 512 and int(new[1][499].item()) == 17
    assert provider.snapshot_rebuilds == 4


@pytest.mark.parametrize("flag", ["_growth_pending", "_loads_inflight", "_prefetch_inflight"])
def test_busy_transaction_reuses_published_snapshot_without_waiting_for_io(flag):
    provider = CacheRoutingProvider(_switch())
    with provider.session.request(provider, synchronize=lambda _root: None):
        published = provider.snapshot()
    assert published is not None
    entered, finish = threading.Event(), threading.Event()

    def writer():
        with ExitStack() as stack:
            for pool in provider.pools:
                stack.enter_context(pool._bk_lock)
                setattr(pool, flag, 1)
        entered.set()
        if not finish.wait(5):
            return
        for pool in provider.pools:
            with pool._bk_lock:
                setattr(pool, flag, 0)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert entered.wait(1)
        with provider.session.request(provider, synchronize=lambda _root: None):
            assert provider.snapshot() is published
        assert provider.busy_fallbacks == 1 and provider.stale_snapshot_reuses == 1
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive()


def test_full_residency_requires_published_membership_not_capacity_alone():
    provider = CacheRoutingProvider(_switch(512))
    assert provider.is_fully_resident()
    with provider.session.request(provider, synchronize=lambda _root: None):
        assert provider.snapshot() is None
        with provider.pools[1]._bk_lock:
            del provider.pools[1]._slot_of[3]
        assert not provider.is_fully_resident()
        assert provider.snapshot() is not None
    assert provider.full_resident_calls == 1


@pytest.mark.parametrize("full_capacity", [True, False])
def test_compiled_gdn_cannot_bypass_cold_cache_routing(full_capacity, monkeypatch):
    from moespresso.runtime.qwen4.model import Qwen4TextModelShell

    model = _model()
    for layer in model.layers:
        layer.mlp.experts = _switch(512 if full_capacity else 216)
        if full_capacity:
            del layer.mlp.experts._projection_pools_lockstep()[0]._slot_of[3]
    configure_cache_routing(model, "prefer-resident")
    model.hidden_size, model.branch_count, model.expanded_size = 2560, 4, 10240
    model.training = False
    model._moespresso_ssd_streaming_resolved_capacities = {
        i: 512 if full_capacity else 216 for i in range(48)
    }
    if not full_capacity:
        def forbidden():
            raise AssertionError("bounded residency should exit before the full-pool guard")
        for layer in model.layers[2:]:
            monkeypatch.setattr(layer.mlp.gate._cache_routing_provider, "is_fully_resident", forbidden)
    assert not Qwen4TextModelShell._compiled_gdn_run_eligible(model, SimpleNamespace(layers=[None] * 48))


def test_snapshot_failure_releases_locks_and_keeps_owner_cleanup(monkeypatch):
    provider = CacheRoutingProvider(_switch())

    def fail():
        raise MemoryError("allocation failed")

    monkeypatch.setattr(provider.pools[1], "_ensure_slot_table", fail)
    with pytest.raises(MemoryError):
        with provider.session.request(provider, synchronize=lambda _root: None):
            provider.snapshot()
    assert not provider.session.active
    for pool in provider.pools:
        assert pool._bk_lock.acquire(blocking=False)
        pool._bk_lock.release()
