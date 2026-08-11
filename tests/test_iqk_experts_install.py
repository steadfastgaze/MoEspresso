"""IQ_K routed-expert install path: synthetic bundle -> index -> kernels.

Builds a relayout bundle shard with IQ_K experts, indexes it, installs the
switch modules on a stub graph, and checks both served routes against a
plain-ops reference built from the same decoded weights. The layer under
test mixes members inside one layer, which is what the flagship allocation
does.
"""

from __future__ import annotations

import json
import struct
import threading
import time

import mlx.core as mx
import numpy as np
import pytest

from moespresso.core.artifact import make_artifact, read_artifact, write_artifact
from moespresso.package import iqk_relayout as rl
from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IK_WIRE,
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_LEGACY_RELAYOUT,
    iqk_geometry,
)
from moespresso.package.manifest import (
    PACKAGE_FORMAT,
    PACKAGE_FORMAT_VERSION,
    file_identity,
)
from moespresso.runtime.deepseek_v4.iqk_experts import (
    IqkInstallError,
    install_deepseek_v4_iqk_experts,
    install_iqk_decode_flush,
    iqk_decode_flush_layers,
    iqk_engagement,
    iqk_switch_modules,
    sorted_prefill_min_pairs,
    sorted_prefill_nsplit,
)
from moespresso.runtime.deepseek_v4.model import (
    _install_deepseek_v4_pooled_bundles,
)
from moespresso.runtime.expert_index import build_expert_index
from moespresso.runtime.expert_slot_pool import BundleRowCache
from moespresso.runtime.pooled_switchglu import (
    PooledDeepseekV4MoEBlock,
    PooledIqkSwitchLinear,
    PooledSwitchGLU,
    install_compact_iqk_dual_gemv,
)
from moespresso.runtime.ssd_streaming_build import (
    seed_expert_residency,
    ssd_streaming_stats,
)
from moespresso.runtime.verify import verify_package

E, D, H = 2, 2048, 2048
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MEMBERS = {"gate_proj": "iq2_ks", "up_proj": "iq2_k",
           "down_proj": "iq1_s_r4"}
SWIGLU_LIMIT = 10.0


def _wire(codec: str, rows: int, in_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    row_bytes = iqk_geometry(codec).bytes_per_row(in_features)
    out = rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
    nblocks = in_features // 256
    if codec == "iq1_s_r4":
        # The scales sit as a 4 x fp16 prefix on each four-row wire group.
        assert rows % 4 == 0
        groups = out.reshape(rows // 4, 4 * row_bytes)
        scales = (rng.standard_normal(rows).astype(np.float32)
                  * 0.004).astype(np.float16)
        groups[:, :8] = scales.view(np.uint8).reshape(rows // 4, 8)
        return out
    slots = [0] if codec == "iq2_ks" else [b * 76 for b in range(nblocks)]
    # Random codes over 2048 inputs sum to a projection output of about
    # sqrt(2048) times the weight magnitude, and the clamped SwiGLU then
    # squares that into the down projection. This scale puts the gate and up
    # outputs across the clamp, so the activation contract is exercised,
    # while keeping the down output well inside fp16.
    scales = (rng.standard_normal(len(slots) * rows).astype(np.float32)
              * 0.004).astype(np.float16)
    raw = scales.view(np.uint8).reshape(rows, len(slots), 2)
    for i, slot in enumerate(slots):
        out[:, slot:slot + 2] = raw[:, i]
    return out


def _package(
    tmp_path,
    layout=IQK_LAYOUT_IQK_RELAYOUT,
    layer=0,
    *,
    members=None,
    model_width=D,
    expert_width=H,
):
    """One relayout bundle shard plus the reference weights it carries."""
    members = dict(MEMBERS if members is None else members)
    pkg = tmp_path / "pkg"
    pkg.mkdir(exist_ok=True)
    comps, reference = {}, {}
    for i, projection in enumerate(PROJECTIONS):
        codec = members[projection]
        in_features = model_width if projection != "down_proj" else expert_width
        out_features = expert_width if projection != "down_proj" else model_width
        wire = _wire(codec, E * out_features, in_features, seed=11 + 7 * i)
        rows = (rl.pack_rows(codec, wire, in_features)
                if layout == IQK_LAYOUT_IQK_RELAYOUT else wire)
        comps[(projection, "blocks")] = rows.reshape(E, out_features, -1)
        reference[projection] = rl.decode_rows(
            codec, rl.pack_rows(codec, wire, in_features), in_features
        ).reshape(E, out_features, in_features).astype(np.float32)
    bundle, geometry = assemble_layer_bundle(
        comps,
        bits={p: iqk_geometry(members[p]).bits for p in PROJECTIONS},
        codecs={p: "iqk" for p in PROJECTIONS},
        iqk_codecs=members,
        iqk_layout=layout,
    )
    key = f"layers.{layer}.ffn.experts.tq_bundle"
    header = {
        "__metadata__": {
            "format": PACKAGE_FORMAT,
            "expert_bundles": encode_bundle_metadata({layer: geometry}),
        },
        key: {"dtype": "U8", "shape": list(bundle.shape),
              "data_offsets": [0, bundle.nbytes]},
    }
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with open(pkg / "model-00001-of-00001.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(bundle.tobytes())
    return pkg, reference


def _activation():
    from jang_tools.dsv4.mlx_model import _DSV4SwiGLU

    return _DSV4SwiGLU(SWIGLU_LIMIT)


class _StubSwitch:
    def __init__(self, activation):
        self.activation = activation


class _StubMLP:
    def __init__(self, activation):
        self.switch_mlp = _StubSwitch(activation)

    def __call__(self, x, input_ids=None):
        del input_ids
        return x * 1.0


class _StubLayer:
    def __init__(self, activation):
        self.mlp = _StubMLP(activation)


class _StubModel:
    def __init__(self, n_layers, activation):
        self.layers = [_StubLayer(activation) for _ in range(n_layers)]

    def eval(self):
        return self


def _reference_forward(reference, x, indices):
    """The DS4 clamped SwiGLU expert forward, in float64 on plain numpy."""
    tokens, top_k = indices.shape
    out = np.zeros((tokens, top_k, D), dtype=np.float64)
    for t in range(tokens):
        row = x[t].astype(np.float64)
        for k in range(top_k):
            e = int(indices[t, k])
            gate = reference["gate_proj"][e].astype(np.float64) @ row
            up = reference["up_proj"][e].astype(np.float64) @ row
            gate = np.minimum(gate, SWIGLU_LIMIT)
            up = np.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
            hidden = (gate / (1.0 + np.exp(-gate))) * up
            out[t, k] = reference["down_proj"][e].astype(np.float64) @ hidden
    return out


def _install(
    tmp_path,
    layout=IQK_LAYOUT_IQK_RELAYOUT,
    *,
    members=None,
    model_width=D,
    expert_width=H,
):
    pkg, reference = _package(
        tmp_path,
        layout=layout,
        members=members,
        model_width=model_width,
        expert_width=expert_width,
    )
    model = _StubModel(1, _activation())
    installed = install_deepseek_v4_iqk_experts(
        model, pkg, build_expert_index(pkg))
    return model, reference, installed


def _pooled_switch(pkg, resident, *, capacity, shared_row_cache=False):
    index = build_expert_index(pkg)
    row_cache = (
        BundleRowCache(
            package_dir=pkg,
            index=index,
            layer=0,
            consumers=3,
        )
        if shared_row_cache
        else None
    )
    projections = {
        projection: PooledIqkSwitchLinear(
            package_dir=pkg,
            index=index,
            layer=0,
            projection=projection,
            capacity=capacity,
            row_cache=row_cache,
        )
        for projection in PROJECTIONS
    }
    switch = PooledSwitchGLU(
        gate_proj=projections["gate_proj"],
        up_proj=projections["up_proj"],
        down_proj=projections["down_proj"],
        activation=resident.activation,
    )
    switch.eval()
    return switch


def test_install_swaps_one_switch_per_indexed_layer(tmp_path):
    activation = _activation()
    pkg, _reference = _package(tmp_path)
    model = _StubModel(1, activation)
    assert install_deepseek_v4_iqk_experts(model, pkg, build_expert_index(pkg)) == 1

    switch = model.layers[0].mlp.switch_mlp
    assert switch.__class__.__name__ == "IqkDeepseekV4SwitchGLU"
    # The clamped SwiGLU contract lives on the graph's own activation module
    # and travels into the new switch rather than being rebuilt here.
    assert switch.activation is activation
    assert switch.members == MEMBERS
    assert not switch.training
    assert not switch.gate_proj.training
    info = model._moespresso_dsv4_iqk_install
    assert info["layers"] == [0]
    assert info["layout"] == IQK_LAYOUT_IQK_RELAYOUT
    assert info["member_counts"] == {"iq2_ks": 1, "iq2_k": 1,
                                     "iq1_s_r4": 1}


def test_decode_route_matches_the_reference_forward(tmp_path):
    model, reference, _ = _install(tmp_path)
    switch = model.layers[0].mlp.switch_mlp
    rng = np.random.default_rng(23)
    tokens, top_k = 2, 4
    x = (rng.standard_normal((tokens, D)) * 0.5).astype(np.float16)
    indices = rng.integers(0, E, size=(tokens, top_k)).astype(np.uint32)

    got = np.asarray(switch(mx.array(x), mx.array(indices)), dtype=np.float64)
    assert got.shape == (tokens, top_k, D)
    assert np.isfinite(got).all()
    want = _reference_forward(reference, x, indices)
    assert np.max(np.abs(want)) < 65504.0    # the served route stores fp16
    rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
    assert rel < 5e-3, rel
    assert switch.gemv_calls == 1 and switch.sorted_prefill_calls == 0


def test_pooled_projection_forced_miss_preserves_iqk_streams_and_output(tmp_path):
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp.gate_proj
    pkg = tmp_path / "pkg"
    pooled = PooledIqkSwitchLinear(
        package_dir=pkg,
        index=build_expert_index(pkg),
        layer=0,
        projection="gate_proj",
        capacity=1,
    )
    rng = np.random.default_rng(101)
    x = mx.array((rng.standard_normal((1, 1, 1, D)) * 0.5).astype(np.float16))

    for expert in (0, 1):
        indices = mx.array([[expert]], dtype=mx.uint32)
        got = pooled(x, indices)
        want = resident.gemv(x, indices)
        mx.eval(got, want)
        assert np.array_equal(np.asarray(got), np.asarray(want))
        slot = pooled.pool.slot_of(expert)
        for name in resident.stream_names():
            assert np.array_equal(
                np.asarray(getattr(pooled.pool.iqk, name)[slot]),
                np.asarray(getattr(resident, name)[expert]),
            )

    assert pooled.pool.resident_ids() == {1}
    assert pooled.pool.total_loads == 2
    assert pooled.pool.total_evictions == 1


def test_iqk_pool_failed_load_never_publishes_a_slot(tmp_path, monkeypatch):
    pkg, _reference = _package(tmp_path)
    pooled = PooledIqkSwitchLinear(
        package_dir=pkg,
        index=build_expert_index(pkg),
        layer=0,
        projection="gate_proj",
        capacity=1,
    )
    original = pooled.pool._load_iqk_blocks

    def fail_load(_source, *, slot):
        del slot
        raise OSError("synthetic IQ_K landing failure")

    monkeypatch.setattr(pooled.pool, "_load_iqk_blocks", fail_load)
    with pytest.raises(OSError, match="synthetic IQ_K landing failure"):
        pooled.pool.ensure([0])

    assert pooled.pool.resident_ids() == set()
    assert pooled.pool._expert_at == [None]
    monkeypatch.setattr(pooled.pool, "_load_iqk_blocks", original)
    pooled.pool.ensure([0])
    assert pooled.pool.resident_ids() == {0}


def test_iqk_pool_growth_preserves_live_streams_and_adds_capacity(tmp_path):
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp.gate_proj
    pkg = tmp_path / "pkg"
    pooled = PooledIqkSwitchLinear(
        package_dir=pkg,
        index=build_expert_index(pkg),
        layer=0,
        projection="gate_proj",
        capacity=1,
    )
    pooled.pool.ensure([0])
    pooled.pool.grow(2)
    pooled.pool.ensure([1])

    for expert in (0, 1):
        slot = pooled.pool.slot_of(expert)
        for name in resident.stream_names():
            assert np.array_equal(
                np.asarray(getattr(pooled.pool.iqk, name)[slot]),
                np.asarray(getattr(resident, name)[expert]),
            )
    x = mx.ones((1, 1, 1, D), dtype=mx.float16)
    indices = mx.array([[0, 1]], dtype=mx.uint32)
    got = pooled(x, indices)
    want = resident.gemv(x, indices)
    mx.eval(got, want)
    assert np.array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("failure_phase", ["allocate", "copy"])
def test_iqk_switch_growth_failure_leaves_every_projection_untouched(
    tmp_path,
    monkeypatch,
    failure_phase,
):
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=1)
    pools = pooled._projection_pools_lockstep()
    for pool in pools:
        pool.ensure([0])
    pooled._barrier_free_ready_cached = False
    pooled._barrier_free_decode_ready_cached = False
    pooled._iqk_decode_identity_cached = False

    before = [
        {
            "capacity": pool.capacity,
            "iqk": pool.iqk,
            "views": pool._iqk_views,
            "slot_nbytes": pool._iqk_slot_nbytes,
            "expert_at": pool._expert_at,
            "slot_of": pool._slot_of,
            "bytes": {
                name: bytes(view) for name, view in pool._iqk_views.items()
            },
        }
        for pool in pools
    ]
    failed_pool = pools[1]
    method_name = (
        "_allocate_growth_candidate"
        if failure_phase == "allocate"
        else "_prepare_growth_locked"
    )
    original = getattr(failed_pool, method_name)

    def fail_growth(_candidate):
        raise MemoryError(f"synthetic {failure_phase} failure")

    monkeypatch.setattr(failed_pool, method_name, fail_growth)
    with pytest.raises(MemoryError, match=f"synthetic {failure_phase} failure"):
        pooled.grow_capacity(2)

    for pool, old in zip(pools, before, strict=True):
        assert pool.capacity == old["capacity"]
        assert pool.iqk is old["iqk"]
        assert pool._iqk_views is old["views"]
        assert pool._iqk_slot_nbytes is old["slot_nbytes"]
        assert pool._expert_at is old["expert_at"]
        assert pool._slot_of is old["slot_of"]
        assert {
            name: bytes(view) for name, view in pool._iqk_views.items()
        } == old["bytes"]
        assert pool._loads_inflight == 0
        assert pool._growth_pending is False
    assert pooled._barrier_free_ready_cached is False
    assert pooled._barrier_free_decode_ready_cached is False
    assert pooled._iqk_decode_identity_cached is False

    # An aborted transaction must release the growth gate for a later retry.
    monkeypatch.setattr(failed_pool, method_name, original)
    pooled.grow_capacity(2)
    assert all(pool.capacity == 2 for pool in pools)
    assert all(pool._slot_of == pools[0]._slot_of for pool in pools[1:])
    assert pooled._barrier_free_ready_cached is None
    assert pooled._barrier_free_decode_ready_cached is None
    assert pooled._iqk_decode_identity_cached is None


def test_iqk_hot_seed_failure_restores_projection_slot_identity(
    tmp_path,
    monkeypatch,
):
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(
        tmp_path / "pkg",
        resident,
        capacity=1,
        shared_row_cache=True,
    )
    pools = pooled._projection_pools_lockstep()
    row_cache = pools[0].row_cache
    assert row_cache is not None
    assert all(pool.row_cache is row_cache for pool in pools)
    for pool in pools:
        pool.ensure([0])
        pool._freq[1] = 100
    pooled.grow_capacity(2)

    before = [
        {
            "slot_of": dict(pool._slot_of),
            "expert_at": list(pool._expert_at),
            "freq": dict(pool._freq),
            "recency": dict(pool._recency),
            "clock": pool._clock,
            "total_misses": pool.total_misses,
            "total_loads": pool.total_loads,
            "total_load_seconds": pool.total_load_seconds,
        }
        for pool in pools
    ]
    failed_pool = pools[1]
    original_load = failed_pool._load_expert

    def fail_second_projection(*, expert, slot):
        del expert, slot
        raise OSError("synthetic second-projection hot-seed failure")

    monkeypatch.setattr(failed_pool, "_load_expert", fail_second_projection)
    with pytest.raises(
        OSError,
        match="synthetic second-projection hot-seed failure",
    ):
        pooled.seed_hot_free_slots()

    for pool, snapshot in zip(pools, before, strict=True):
        assert pool._slot_of == snapshot["slot_of"] == {0: 0}
        assert pool._expert_at == snapshot["expert_at"] == [0, None]
        assert pool._freq == snapshot["freq"]
        assert pool._recency == snapshot["recency"]
        assert pool._clock == snapshot["clock"]
        assert pool.total_misses == snapshot["total_misses"]
        assert pool.total_loads == snapshot["total_loads"]
        assert pool.total_load_seconds == snapshot["total_load_seconds"]
        assert pool._loads_inflight == 0
        assert pool._growth_pending is False
    assert row_cache._rows == {}
    assert row_cache._inflight == {}

    # The first projection did land bytes before the second failed. They were
    # never published, and a later bounded route must reload all three rows and
    # preserve the resident implementation's exact output.
    monkeypatch.setattr(failed_pool, "_load_expert", original_load)
    rng = np.random.default_rng(115)
    x = mx.array((rng.standard_normal((1, D)) * 0.5).astype(np.float16))
    indices = mx.array([[1]], dtype=mx.uint32)
    pooled.publish_slots(indices)
    got = pooled.build_pipelined(x, indices)
    want = resident(x, indices)
    mx.eval(got, want)

    assert np.array_equal(np.asarray(got), np.asarray(want))
    assert all(pool._slot_of == pools[0]._slot_of for pool in pools[1:])


@pytest.mark.parametrize("writer", ["ensure", "prefetch"])
def test_iqk_switch_growth_waits_for_inflight_storage_writer(
    tmp_path,
    monkeypatch,
    writer,
):
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=1)
    pools = pooled._projection_pools_lockstep()
    pool = pools[0]
    load_started = threading.Event()
    release_load = threading.Event()
    load_finished = threading.Event()
    original_load = pool._load_expert

    def blocked_load(*, expert, slot):
        load_started.set()
        if not release_load.wait(5):
            raise TimeoutError("test did not release the blocked IQ_K load")
        original_load(expert=expert, slot=slot)
        load_finished.set()

    monkeypatch.setattr(pool, "_load_expert", blocked_load)
    errors = []

    def write_pool():
        try:
            if writer == "ensure":
                pool.ensure([0])
            else:
                assert pool.prefetch([0], reserve_floor=0) == 1
        except BaseException as exc:  # surfaced on the test thread below
            errors.append(exc)

    writer_thread = threading.Thread(target=write_pool)
    writer_thread.start()
    assert load_started.wait(5)
    assert pool._loads_inflight == 1

    original_allocate = pool._allocate_growth_candidate

    def checked_allocate(capacity):
        assert load_finished.is_set()
        assert pool._slot_of == {0: 0}
        return original_allocate(capacity)

    monkeypatch.setattr(pool, "_allocate_growth_candidate", checked_allocate)

    def release_after_growth_closes_gate():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            states = []
            for candidate_pool in pools:
                with candidate_pool._bk_lock:
                    states.append(candidate_pool._growth_pending)
            if all(states):
                release_load.set()
                return
            time.sleep(0.001)
        errors.append(TimeoutError("growth did not close every pool gate"))
        release_load.set()

    release_thread = threading.Thread(target=release_after_growth_closes_gate)
    release_thread.start()
    pooled.grow_capacity(2)
    writer_thread.join(5)
    release_thread.join(5)

    assert not writer_thread.is_alive()
    assert not release_thread.is_alive()
    assert errors == []
    assert all(pool.capacity == 2 for pool in pools)
    assert all(not pool._growth_pending for pool in pools)
    assert all(pool._loads_inflight == 0 for pool in pools)
    assert pools[0].resident_ids() == {0}
    slot = pools[0].slot_of(0)
    for name in resident.gate_proj.stream_names():
        assert np.array_equal(
            np.asarray(getattr(pools[0].iqk, name)[slot]),
            np.asarray(getattr(resident.gate_proj, name)[0]),
        )


def test_pooled_switch_full_resident_decode_is_bit_identical_and_sync_free(tmp_path):
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    for pool in pooled._projection_pools_lockstep():
        pool.ensure(range(E))
    rng = np.random.default_rng(103)
    x = mx.array((rng.standard_normal((2, D)) * 0.5).astype(np.float16))
    indices = mx.array([[0, 1], [1, 0]], dtype=mx.uint32)

    got = pooled(x, indices)
    want = resident(x, indices)
    mx.eval(got, want)

    assert np.array_equal(np.asarray(got), np.asarray(want))
    assert pooled._all_iqk
    assert pooled.barrier_free_prefill_calls == 1
    assert pooled.index_sync_calls == 0
    assert pooled.index_resync_calls == 0
    assert pooled.gemv_calls == 1
    assert all(pool.total_loads == E for pool in pooled._projection_pools_lockstep())


def test_pooled_switch_full_resident_nonidentity_slots_remap_on_device(tmp_path):
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    for pool in pooled._projection_pools_lockstep():
        pool.ensure([1])
        pool.ensure([0])
    assert pooled._barrier_free_decode_ready()
    assert pooled._iqk_decode_identity_cached is False
    x = mx.ones((1, D), dtype=mx.float16)
    indices = mx.array([[0, 1]], dtype=mx.uint32)

    got = pooled.build_barrier_free_decode(x, indices)
    want = resident(x, indices)
    mx.eval(got, want)

    assert np.array_equal(np.asarray(got), np.asarray(want))
    assert pooled.index_sync_calls == 0
    assert pooled.index_resync_calls == 0


@pytest.mark.parametrize(
    ("gate_member", "up_member"),
    [
        ("iq1_s_r4", "iq1_s_r4"),
        ("iq1_s_r4", "iq2_ks"),
        ("iq2_ks", "iq2_k"),
        ("iq2_ks", "iq2_ks"),
    ],
)
def test_iqk_dual_gemv_matches_separate_projection_words(
    tmp_path,
    gate_member,
    up_member,
):
    from moespresso.runtime.deepseek_v4.iqk_decode_kernel import dual_gemv

    members = {
        "gate_proj": gate_member,
        "up_proj": up_member,
        "down_proj": "iq2_k",
    }
    model, _reference, _ = _install(tmp_path, members=members)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    for pool in pooled._projection_pools_lockstep():
        pool.ensure(range(E))

    rng = np.random.default_rng(171)
    x = mx.array((rng.standard_normal((1, D)) * 0.5).astype(np.float16))
    indices = mx.array([[0, 1]], dtype=mx.uint32)
    x4 = mx.expand_dims(x, (-2, -3))
    want_gate = pooled.gate_proj.pool.iqk.gemv(x4, indices)
    want_up = pooled.up_proj.pool.iqk.gemv(x4, indices)
    got_gate, got_up = dual_gemv(
        pooled.gate_proj.pool.iqk,
        pooled.up_proj.pool.iqk,
        x4,
        indices,
    )
    mx.eval(want_gate, want_up, got_gate, got_up)

    assert np.array_equal(np.asarray(got_gate), np.asarray(want_gate))
    assert np.array_equal(np.asarray(got_up), np.asarray(want_up))


def test_iqk_dual_gemv_matches_words_at_ds4_gate_up_geometry(tmp_path):
    from moespresso.runtime.deepseek_v4.iqk_decode_kernel import dual_gemv

    model, _reference, _ = _install(
        tmp_path,
        members={
            "gate_proj": "iq1_s_r4",
            "up_proj": "iq2_ks",
            "down_proj": "iq2_k",
        },
        model_width=4096,
        expert_width=2048,
    )
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    for pool in pooled._projection_pools_lockstep():
        pool.ensure(range(E))

    rng = np.random.default_rng(172)
    x = mx.array((rng.standard_normal((1, 4096)) * 0.5).astype(np.float16))
    indices = mx.array([[0, 1]], dtype=mx.uint32)
    x4 = mx.expand_dims(x, (-2, -3))
    want_gate = pooled.gate_proj.pool.iqk.gemv(x4, indices)
    want_up = pooled.up_proj.pool.iqk.gemv(x4, indices)
    got_gate, got_up = dual_gemv(
        pooled.gate_proj.pool.iqk,
        pooled.up_proj.pool.iqk,
        x4,
        indices,
    )
    mx.eval(want_gate, want_up, got_gate, got_up)

    assert np.array_equal(np.asarray(got_gate), np.asarray(want_gate))
    assert np.array_equal(np.asarray(got_up), np.asarray(want_up))


def test_compact_iqk_dual_gemv_is_bit_identical_and_fails_closed(
    tmp_path,
    monkeypatch,
):
    import moespresso.runtime.pooled_switchglu as psg
    from moespresso.runtime.deepseek_v4.speed_stats import _COUNT_KEYS

    monkeypatch.setattr(psg, "_IQK_DUAL_GEMV", True)
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    for pool in pooled._projection_pools_lockstep():
        pool.ensure(range(E))
    assert pooled._barrier_free_decode_ready()

    rng = np.random.default_rng(173)
    x = mx.array((rng.standard_normal((1, D)) * 0.5).astype(np.float16))
    indices = mx.array([[0, 1]], dtype=mx.uint32)
    compact_source_ids = mx.array([3, 11], dtype=mx.uint32)

    incumbent = pooled.build_barrier_free_decode(x, indices)
    assert getattr(pooled, "iqk_dual_gemv_calls", 0) == 0
    assert not install_compact_iqk_dual_gemv(
        pooled,
        mx.array([3], dtype=mx.uint32),
    )
    mismatch = pooled.build_barrier_free_decode(x, indices)
    assert getattr(pooled, "iqk_dual_gemv_calls", 0) == 0
    monkeypatch.setattr(psg, "_IQK_DUAL_GEMV", False)
    assert not install_compact_iqk_dual_gemv(pooled, compact_source_ids)
    assert (
        pooled.build_barrier_free_decode.__func__
        is PooledSwitchGLU.build_barrier_free_decode
    )
    monkeypatch.setattr(psg, "_IQK_DUAL_GEMV", True)
    assert install_compact_iqk_dual_gemv(pooled, compact_source_ids)
    assert (
        pooled.build_barrier_free_decode.__func__
        is PooledSwitchGLU.build_compact_barrier_free_decode
    )
    compact = pooled.build_barrier_free_decode(x, indices)
    mx.eval(incumbent, mismatch, compact)

    assert np.array_equal(np.asarray(mismatch), np.asarray(incumbent))
    assert np.array_equal(np.asarray(compact), np.asarray(incumbent))
    assert pooled.iqk_dual_gemv_calls == 1
    assert pooled.iqk_dual_gemv_pairs == 2

    monkeypatch.setattr(psg, "_IQK_DUAL_GEMV", False)
    killed = pooled.build_barrier_free_decode(x, indices)
    mx.eval(killed)
    assert np.array_equal(np.asarray(killed), np.asarray(incumbent))
    assert pooled.iqk_dual_gemv_calls == 1

    model.layers[0].mlp.switch_mlp = pooled
    engagement = iqk_engagement(model)
    census = ssd_streaming_stats(model)
    assert engagement["iqk_dual_gemv_calls"] == 1
    assert engagement["iqk_dual_gemv_pairs"] == 2
    assert census["iqk_dual_gemv_calls"] == 1
    assert census["iqk_dual_gemv_pairs"] == 2
    assert census["built_iqk_dual_gemv_kernel_count"] == len(
        engagement["built_iqk_dual_gemv_kernels"]
    )
    assert "iqk_dual_gemv_calls" in _COUNT_KEYS
    assert "iqk_dual_gemv_pairs" in _COUNT_KEYS
    assert "built_iqk_dual_gemv_kernel_count" in _COUNT_KEYS


def test_pooled_switch_partial_residency_matches_resident_across_eviction(tmp_path):
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=1)
    rng = np.random.default_rng(107)
    x = mx.array((rng.standard_normal((2, D)) * 0.5).astype(np.float16))
    indices = mx.array([[0], [1]], dtype=mx.uint32)

    got = pooled(x, indices)
    want = resident(x, indices)
    mx.eval(got, want)

    assert np.array_equal(np.asarray(got), np.asarray(want))
    assert pooled.row_chunked_calls == 1
    assert pooled.total_chunks == 2
    pools = pooled._projection_pools_lockstep()
    assert all(pool.total_evictions == 1 for pool in pools)
    assert all(pool._slot_of == pools[0]._slot_of for pool in pools[1:])


def test_pooled_iqk_pipelined_decode_records_the_live_gemv_route(tmp_path):
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    rng = np.random.default_rng(111)
    x = mx.array((rng.standard_normal((1, D)) * 0.5).astype(np.float16))
    indices = mx.array([[0, 1]], dtype=mx.uint32)

    pooled.publish_slots(indices)
    got = pooled.build_pipelined(x, indices)
    want = resident(x, indices)
    mx.eval(got, want)

    assert np.array_equal(np.asarray(got), np.asarray(want))
    assert pooled.pipelined_layers == 1
    assert pooled.gemv_calls == 1
    assert pooled.gemv_pairs == 2


def test_pooled_iqk_cold_full_pool_earns_decode_certificate_after_refill(
    tmp_path,
):
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    for pool in pooled._projection_pools_lockstep():
        pool.ensure([0])

    assert pooled._barrier_free_decode_ready() is False
    assert pooled._barrier_free_decode_ready_cached is False

    rng = np.random.default_rng(116)
    x = mx.array((rng.standard_normal((1, D)) * 0.5).astype(np.float16))
    refill_indices = mx.array([[1]], dtype=mx.uint32)
    pooled.publish_slots(refill_indices)
    refill = pooled.build_pipelined(x, refill_indices)
    refill_reference = resident(x, refill_indices)
    mx.eval(refill, refill_reference)
    assert np.array_equal(np.asarray(refill), np.asarray(refill_reference))

    assert pooled._barrier_free_decode_ready() is True
    assert pooled._barrier_free_decode_ready_cached is True
    assert pooled._iqk_decode_identity_cached is True

    next_indices = mx.array([[0, 1]], dtype=mx.uint32)
    got = pooled.build_barrier_free_decode(x, next_indices)
    want = resident(x, next_indices)
    mx.eval(got, want)
    assert np.array_equal(np.asarray(got), np.asarray(want))


def test_pooled_switch_partial_sorted_route_matches_resident_across_eviction(
    tmp_path,
    monkeypatch,
):
    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_PREFILL_PREFETCH", True)
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_PAIRS", "1")
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", "2")
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=1)
    rng = np.random.default_rng(108)
    x = mx.array((rng.standard_normal((2, D)) * 0.5).astype(np.float16))
    indices = mx.array([[0], [1]], dtype=mx.uint32)

    got = pooled(x, indices)
    want = resident(x, indices)
    mx.eval(got, want)

    assert np.array_equal(np.asarray(got), np.asarray(want))
    assert pooled.sorted_chunked_calls == 1
    assert pooled.sorted_prefill_calls == 1
    assert pooled.total_chunks == 2
    assert pooled.prefetch_ticket_submitted == 0
    assert pooled._prefetch_ticket is None
    pools = pooled._projection_pools_lockstep()
    assert all(pool._slot_of == pools[0]._slot_of for pool in pools[1:])


def test_pooled_switch_full_resident_sorted_route_matches_incumbent(tmp_path, monkeypatch):
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_PAIRS", "1")
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", "2")
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    for pool in pooled._projection_pools_lockstep():
        pool.ensure(range(E))
    rng = np.random.default_rng(109)
    x = mx.array((rng.standard_normal((3, D)) * 0.5).astype(np.float16))
    indices = mx.array([[0, 1], [1, 0], [1, 1]], dtype=mx.uint32)

    got = pooled(x, indices)
    want = resident(x, indices)
    mx.eval(got, want)

    assert np.array_equal(np.asarray(got), np.asarray(want))
    assert pooled.barrier_free_prefill_calls == 1
    assert pooled.sorted_prefill_calls == 1
    assert pooled.sorted_nsplit_calls == 1
    assert pooled.sorted_nsplit_parts == 2
    assert pooled.index_resync_calls == 0


def test_pooled_switch_full_resident_sorted_bf16_matches_incumbent(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_PAIRS", "1")
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", "2")
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    pooled = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    for pool in pooled._projection_pools_lockstep():
        pool.ensure(range(E))
    rng = np.random.default_rng(110)
    x = mx.array(
        (rng.standard_normal((3, D)) * 0.5).astype(np.float32),
    ).astype(mx.bfloat16)
    indices = mx.array([[0, 1], [1, 0], [1, 1]], dtype=mx.uint32)

    got = pooled(x, indices)
    want = resident(x, indices)
    mx.eval(got, want)

    assert got.dtype == want.dtype
    assert bool(mx.array_equal(got, want).item())
    assert pooled.barrier_free_prefill_calls == 1
    assert pooled.index_sync_calls == 0
    assert pooled.index_resync_calls == 0


def test_deepseek_iqk_pooled_installer_and_full_prewarm_share_the_pool_path(
    tmp_path,
):
    model, _reference, _ = _install(tmp_path)
    pkg = tmp_path / "pkg"
    index = build_expert_index(pkg)

    assert _install_deepseek_v4_pooled_bundles(
        model,
        pkg,
        index,
        seed=42,
        capacity_per_layer=E,
    ) == 1
    seeded = seed_expert_residency(model, pkg)

    switch = model.layers[0].mlp.switch_mlp
    assert isinstance(switch, PooledSwitchGLU)
    assert switch._all_iqk
    assert model._moespresso_ssd_streaming_capacity == E
    assert model._moespresso_dsv4_iqk_install["pooled"] is True
    assert seeded["source"] == "all-default"
    assert all(
        pool.resident_ids() == set(range(E))
        for pool in switch._projection_pools_lockstep()
    )
    row_cache = switch.gate_proj.pool.row_cache
    assert row_cache is switch.up_proj.pool.row_cache
    assert row_cache is switch.down_proj.pool.row_cache
    assert row_cache.total_preads == E
    assert row_cache.total_cached_takes == 2 * E


def test_pooled_iqk_block_preserves_decode_and_verify_commit_cadence(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_DECODE_FLUSH_LAYERS", "2")
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    switch = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    switch.iqk_ordinal = 1
    for pool in switch._projection_pools_lockstep():
        pool.ensure(range(E))

    class _Gate:
        def __call__(self, x, input_ids=None):
            del input_ids
            shape = (*x.shape[:-1], 1)
            return (
                mx.zeros(shape, dtype=mx.uint32),
                mx.ones(shape, dtype=mx.float32),
            )

    class _Shared:
        def __call__(self, x):
            return mx.zeros_like(x)

    class _Original:
        gate = _Gate()
        shared_experts = _Shared()
        sharding_group = None

        def __init__(self, switch_mlp):
            self.switch_mlp = switch_mlp

    block = PooledDeepseekV4MoEBlock(_Original(switch))
    block.eval()

    decode = block(mx.ones((1, D), dtype=mx.float16))
    verify = block(mx.ones((1, 6, D), dtype=mx.float16))
    wider = block(mx.ones((1, 9, D), dtype=mx.float16))
    mx.eval(decode, verify, wider)

    assert switch.iqk_decode_flush_calls == 1
    assert switch.iqk_verify_flush_calls == 1
    assert switch.barrier_free_decode_flush_calls == 1
    assert getattr(switch, "iqk_dual_gemv_calls", 0) == 0
    model.layers[0].mlp = block
    engagement = iqk_engagement(model)
    census = ssd_streaming_stats(model)
    assert engagement["switch_modules"] == 1
    assert engagement["iqk_decode_flush_calls"] == 1
    assert engagement["iqk_verify_flush_calls"] == 1
    assert census["iqk_decode_flush_calls"] == 1
    assert census["iqk_verify_flush_calls"] == 1
    assert census["iqk_gemv_calls"] == switch.gemv_calls


def test_compact_install_engages_the_iqk_dual_gemv(tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_IQK_DUAL_GEMV", True)
    model, _reference, _ = _install(tmp_path)
    resident = model.layers[0].mlp.switch_mlp
    switch = _pooled_switch(tmp_path / "pkg", resident, capacity=E)
    for pool in switch._projection_pools_lockstep():
        pool.ensure(range(E))

    class _CompactGate:
        def __call__(self, x, input_ids=None):
            del input_ids
            shape = (*x.shape[:-1], 1)
            return (
                mx.zeros(shape, dtype=mx.uint32),
                mx.ones(shape, dtype=mx.float32),
            )

    class _Shared:
        def __call__(self, x):
            return mx.zeros_like(x)

    class _Original:
        gate = _CompactGate()
        shared_experts = _Shared()
        sharding_group = None

        def __init__(self, switch_mlp):
            self.switch_mlp = switch_mlp

    block = PooledDeepseekV4MoEBlock(_Original(switch))
    block.eval()
    assert install_compact_iqk_dual_gemv(
        switch,
        mx.array([2, 9], dtype=mx.uint32),
    )
    output = block(mx.ones((1, D), dtype=mx.float16))
    mx.eval(output)

    assert switch.iqk_dual_gemv_calls == 1
    assert switch.iqk_dual_gemv_pairs == 1


def test_sorted_route_matches_the_reference_forward(tmp_path, monkeypatch):
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_PAIRS", "1")
    model, reference, _ = _install(tmp_path)
    switch = model.layers[0].mlp.switch_mlp
    rng = np.random.default_rng(29)
    tokens, top_k = 5, 3
    x = (rng.standard_normal((tokens, D)) * 0.5).astype(np.float16)
    indices = rng.integers(0, E, size=(tokens, top_k)).astype(np.uint32)

    got = np.asarray(switch(mx.array(x), mx.array(indices)), dtype=np.float64)
    assert got.shape == (tokens, top_k, D)
    assert np.isfinite(got).all()
    want = _reference_forward(reference, x, indices)
    assert np.max(np.abs(want)) < 65504.0    # the served route stores fp16
    rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
    assert rel < 5e-3, rel
    assert switch.sorted_prefill_calls == 1 and switch.gemv_calls == 0


def test_the_two_routes_agree_on_the_same_dispatch(tmp_path, monkeypatch):
    """Different accumulation orders, so agreement is fp16-class, not exact."""
    model, _reference, _ = _install(tmp_path)
    switch = model.layers[0].mlp.switch_mlp
    rng = np.random.default_rng(31)
    x = mx.array((rng.standard_normal((4, D)) * 0.5).astype(np.float16))
    indices = mx.array(rng.integers(0, E, size=(4, 3)).astype(np.uint32))

    decode = np.asarray(switch(x, indices), dtype=np.float64)
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_PAIRS", "1")
    prefill = np.asarray(switch(x, indices), dtype=np.float64)
    rel = np.max(np.abs(decode - prefill)) / np.max(np.abs(prefill))
    assert rel < 5e-3, rel


def test_engagement_reports_the_kernels_this_process_built(tmp_path):
    model, _reference, _ = _install(tmp_path)
    switch = model.layers[0].mlp.switch_mlp
    rng = np.random.default_rng(37)
    x = mx.array((rng.standard_normal((1, D)) * 0.5).astype(np.float16))
    mx.eval(switch(x, mx.array(rng.integers(0, E, size=(1, 4)).astype(np.uint32))))

    report = iqk_engagement(model)
    assert report["switch_modules"] == 1
    assert report["layers"] == [0]
    assert report["gemv_calls"] >= 1
    assert report["layers_without_a_gemv_call"] == []
    built = {tuple(k[:3]) for k in report["built_gemv_kernels"]}
    assert ("iq2_ks", D, H) in built
    assert ("iq2_k", D, H) in built
    assert ("iq1_s_r4", D, H) in built


def test_install_serves_a_package_written_before_the_name_was_corrected(tmp_path):
    """A package carrying the pre-rename layout spelling still installs.

    The layout is a value inside shard metadata, so packages written before
    the name was corrected hold the old spelling and rewriting them would mean
    shipping their bytes again. The rewrite below reproduces exactly what such
    a package looks like: the substitution is length-preserving, so the header
    length and every tensor offset are untouched and only the twelve-byte word
    differs.
    """
    pkg, _reference = _package(tmp_path)
    shard = pkg / "model-00001-of-00001.safetensors"
    current = shard.read_bytes()
    header_size = struct.unpack("<Q", current[:8])[0]
    header_end = 8 + header_size
    legacy_header = current[8:header_end].replace(
        IQK_LAYOUT_IQK_RELAYOUT.encode(),
        IQK_LAYOUT_LEGACY_RELAYOUT.encode(),
    )
    legacy = current[:8] + legacy_header + current[header_end:]
    assert legacy != current
    assert len(legacy) == len(current)
    assert legacy[header_end:] == current[header_end:]
    shard.write_bytes(legacy)

    manifest = make_artifact(
        "package_manifest",
        {"source_root": "toy"},
        {"tool": "test", "version": "0"},
        status="valid",
        package_format=PACKAGE_FORMAT,
        package_format_version=PACKAGE_FORMAT_VERSION,
        architecture={"family": "deepseek_v4_flash"},
        tensors=[{
            "source_name": f"layers.0.ffn.experts.{projection.removesuffix('_proj')}",
            "kind": "expert",
            "format": "iqk",
            "format_params": {
                "iqk_codec": codec,
                "layout": IQK_LAYOUT_LEGACY_RELAYOUT,
            },
            "shard": shard.name,
            "key_prefix": "layers.0.ffn.experts",
        } for projection, codec in MEMBERS.items()],
        required_ops=["iqk_dequant"],
        files=[file_identity(shard)],
        tokenizer={"files": []},
    )
    manifest_path = pkg / "package_manifest.json"
    write_artifact(manifest_path, manifest)
    sealed = read_artifact(manifest_path)
    assert not any(v.blocking for v in verify_package(sealed, pkg))

    index = build_expert_index(pkg)
    assert index.geometry(
        layer=0, projection="gate_proj").layout == IQK_LAYOUT_IQK_RELAYOUT

    model = _StubModel(1, _activation())
    install_deepseek_v4_iqk_experts(model, pkg, index)
    assert iqk_engagement(model)["switch_modules"] == 1


def test_install_refuses_a_package_still_on_the_quantizers_wire(tmp_path):
    pkg, _reference = _package(tmp_path, layout=IQK_LAYOUT_IK_WIRE)
    model = _StubModel(1, _activation())
    with pytest.raises(IqkInstallError, match=IQK_LAYOUT_IQK_RELAYOUT):
        install_deepseek_v4_iqk_experts(model, pkg, build_expert_index(pkg))


def test_install_refuses_a_seam_with_no_activation(tmp_path):
    pkg, _reference = _package(tmp_path)
    model = _StubModel(1, None)
    with pytest.raises(IqkInstallError, match="activation"):
        install_deepseek_v4_iqk_experts(model, pkg, build_expert_index(pkg))


def test_the_sort_threshold_env_fails_closed(monkeypatch):
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_PAIRS", "many")
    with pytest.raises(IqkInstallError):
        sorted_prefill_min_pairs()
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_PAIRS", "0")
    with pytest.raises(IqkInstallError):
        sorted_prefill_min_pairs()
    monkeypatch.delenv("MOESPRESSO_DSV4_IQK_SORT_PAIRS")
    assert sorted_prefill_min_pairs() == 4096


def test_the_nsplit_env_fails_closed(monkeypatch):
    for bad in ("many", "0", "3", "-2", "6"):
        monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", bad)
        with pytest.raises(IqkInstallError):
            sorted_prefill_nsplit()
    monkeypatch.delenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT")
    assert sorted_prefill_nsplit() == 16
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", "4")
    assert sorted_prefill_nsplit() == 4
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", "1")
    assert sorted_prefill_nsplit() == 1


def test_the_nsplit_sorted_route_matches_the_unsplit_route(tmp_path, monkeypatch):
    """The split route is the same dot products: raw fp16 bit identity.

    Each output element is one dot product over the full input width in
    both forms; the split only caps the size of the dequantized fp16
    temporaries in flight.
    """
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_PAIRS", "1")
    model, _reference, _ = _install(tmp_path)
    switch = model.layers[0].mlp.switch_mlp
    rng = np.random.default_rng(41)
    tokens, top_k = 5, 3
    x = mx.array((rng.standard_normal((tokens, D)) * 0.5).astype(np.float16))
    indices = mx.array(rng.integers(0, E, size=(tokens, top_k)).astype(np.uint32))

    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", "1")
    unsplit = np.asarray(switch(x, indices).view(mx.uint16))
    assert switch.sorted_nsplit_calls == 0

    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", "2")
    split = np.asarray(switch(x, indices).view(mx.uint16))
    assert np.array_equal(split, unsplit)
    assert switch.sorted_nsplit_calls == 1
    assert switch.sorted_nsplit_parts == 2

    report = iqk_engagement(model)
    assert report["sorted_nsplit_calls"] == 1
    assert report["sorted_nsplit_parts"] == 2
    range_built = {tuple(k) for k in report["built_dequant_range_kernels"]}
    assert ("iq2_ks", D, H, H // 2) in range_built
    assert ("iq2_k", D, H, H // 2) in range_built
    assert ("iq1_s_r4", D, H, H // 2) in range_built


def test_routed_seam_counters_reach_all_three_census_surfaces(
        tmp_path, monkeypatch):
    """A counter on one surface defeats a census-gated instrument.

    The arm instruments read `iqk_engagement`, `ssd_streaming_stats`, and the
    speed-stats count keys interchangeably, so a route counter that exists on
    only one of them reports no arm difference on the other two.
    """
    from moespresso.runtime.deepseek_v4.speed_stats import _COUNT_KEYS
    from moespresso.runtime.ssd_streaming_build import ssd_streaming_stats

    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_PAIRS", "1")
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", "2")
    model, _reference, _ = _install(tmp_path)
    switch = model.layers[0].mlp.switch_mlp
    rng = np.random.default_rng(43)
    tokens, top_k = 4, 3
    x = mx.array((rng.standard_normal((tokens, D)) * 0.5).astype(np.float16))
    indices = mx.array(rng.integers(0, E, size=(tokens, top_k)).astype(np.uint32))
    switch(x, indices)

    report = iqk_engagement(model)
    census = ssd_streaming_stats(model)
    for engagement_key, census_key in (
        ("gemv_calls", "iqk_gemv_calls"),
        ("gemv_pairs", "iqk_gemv_pairs"),
        ("sorted_prefill_calls", "iqk_sorted_prefill_calls"),
        ("sorted_prefill_pairs", "iqk_sorted_prefill_pairs"),
        ("sorted_nsplit_calls", "iqk_sorted_nsplit_calls"),
        ("sorted_nsplit_parts", "iqk_sorted_nsplit_parts"),
    ):
        assert census[census_key] == report[engagement_key], census_key
        assert census_key in _COUNT_KEYS, census_key
    # The arm ran the sorted route, so the counters are not all zero.
    assert census["iqk_sorted_prefill_calls"] == 1
    assert census["iqk_sorted_nsplit_parts"] == 2

    # The built dequant-range registry is a kernel-key list on the
    # engagement surface and its size on the count surfaces, which is the
    # form the phase splitter can subtract. The split route above built the
    # three members' range kernels, so this pin is not comparing zeros.
    assert census["built_dequant_range_kernel_count"] == len(
        report["built_dequant_range_kernels"])
    assert report["built_dequant_range_kernel_count"] == len(
        report["built_dequant_range_kernels"])
    assert "built_dequant_range_kernel_count" in _COUNT_KEYS
    assert census["built_dequant_range_kernel_count"] >= 3


def test_the_verify_flush_width_covers_the_drafter_verify_shape():
    """Three constants describe one geometry; nothing tied them together.

    A DSpark chain drafts `block_size` tokens, so a verify forward carries
    the anchor plus those drafts. The flush wrapper's row cap must cover that
    width or the verify commit silently stops firing, and the q8 tiny-M route
    is sized for the same shape. All three moving independently is how a
    scheduling win disappears without a failing test.
    """
    from moespresso.runtime.deepseek_v4.dspark_model import DSparkArgs
    from moespresso.runtime.deepseek_v4.iqk_experts import _VERIFY_FLUSH_MAX_ROWS
    from moespresso.runtime.deepseek_v4.model import _dsv4_q8_tiny_m_rows_max

    verify_rows = DSparkArgs.block_size + 1
    assert _VERIFY_FLUSH_MAX_ROWS >= verify_rows, (
        f"the flush cap {_VERIFY_FLUSH_MAX_ROWS} is below the verify width "
        f"{verify_rows}; the verify-shape commit would stop firing")
    assert _VERIFY_FLUSH_MAX_ROWS == _dsv4_q8_tiny_m_rows_max(), (
        "the flush cap and the q8 tiny-M cap size the same verify shape")


def test_the_decode_flush_env_fails_closed(monkeypatch):
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_DECODE_FLUSH_LAYERS", "four")
    with pytest.raises(IqkInstallError):
        iqk_decode_flush_layers()
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_DECODE_FLUSH_LAYERS", "-1")
    with pytest.raises(IqkInstallError):
        iqk_decode_flush_layers()
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_DECODE_FLUSH_LAYERS", "0")
    assert iqk_decode_flush_layers() == 0
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_DECODE_FLUSH_LAYERS", "8")
    assert iqk_decode_flush_layers() == 8
    monkeypatch.delenv("MOESPRESSO_DSV4_IQK_DECODE_FLUSH_LAYERS")
    assert iqk_decode_flush_layers() == 4


def test_decode_flush_wraps_on_cadence_and_kicks_decode_and_verify_shapes(
        tmp_path, monkeypatch):
    model, _reference, _ = _install(tmp_path)
    switch = model.layers[0].mlp.switch_mlp
    for _ in range(3):
        layer = _StubLayer(_activation())
        layer.mlp.switch_mlp = switch
        model.layers.append(layer)

    # Cadence 0 is the kill switch: nothing installs, nothing is recorded.
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_DECODE_FLUSH_LAYERS", "0")
    assert install_iqk_decode_flush(model) == 0
    assert getattr(model, "_moespresso_dsv4_iqk_decode_flush", None) is None

    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_DECODE_FLUSH_LAYERS", "2")
    assert install_iqk_decode_flush(model) == 2
    kinds = [type(layer.mlp).__name__ for layer in model.layers]
    assert kinds == ["_StubMLP", "IqkDecodeFlushMoEBlock",
                     "_StubMLP", "IqkDecodeFlushMoEBlock"]
    # A second install is a no-op, and the switch seam stays readable
    # through the wrapper.
    assert install_iqk_decode_flush(model) == 2
    assert len(iqk_switch_modules(model)) == 4

    block = model.layers[1].mlp
    assert not block.training
    decode_x = mx.ones((1, 1, D), dtype=mx.float16)
    y = block(decode_x)
    assert block.iqk_decode_flush_calls == 1
    assert np.array_equal(np.asarray(y), np.asarray(block.inner(decode_x)))
    # Verify-shaped multi-token calls commit through their own counter;
    # prefill-shaped calls pass through without a commit.
    verify_x = mx.ones((1, 6, D), dtype=mx.float16)
    yv = block(verify_x)
    assert block.iqk_verify_flush_calls == 1
    assert block.iqk_decode_flush_calls == 1
    assert np.array_equal(np.asarray(yv), np.asarray(block.inner(verify_x)))
    block(mx.ones((1, 9, D), dtype=mx.float16))
    assert block.iqk_verify_flush_calls == 1
    assert block.iqk_decode_flush_calls == 1

    report = iqk_engagement(model)
    assert report["decode_flush"] == {"cadence": 2, "wrapped_blocks": 2}
    assert report["iqk_decode_flush_calls"] == 1
    assert report["iqk_verify_flush_calls"] == 1

    # Both commit counters travel through the shared census surface, so a
    # census-gated instrument can prove verify-flush arms differ.
    from moespresso.runtime.ssd_streaming_build import ssd_streaming_stats

    census = ssd_streaming_stats(model)
    assert census["iqk_decode_flush_calls"] == 1
    assert census["iqk_verify_flush_calls"] == 1

    from moespresso.runtime.deepseek_v4.speed_stats import _COUNT_KEYS

    assert "iqk_decode_flush_calls" in _COUNT_KEYS
    assert "iqk_verify_flush_calls" in _COUNT_KEYS


def test_the_dsv4_adapter_accepts_the_iqk_op():
    from moespresso.runtime.build import (
        UnsupportedRuntimeAdapter,
        _runtime_adapter_kind,
    )

    manifest = {
        "architecture": {"family": "deepseek_v4_flash"},
        "required_ops": ["affine_dequant", "fp16_passthrough", "iqk_dequant",
                         "kquant_dequant", "mxfp8_dequant",
                         "raw_dtype_passthrough"],
    }
    assert _runtime_adapter_kind(manifest) == "mjtq_dsv4"
    with pytest.raises(UnsupportedRuntimeAdapter):
        _runtime_adapter_kind({
            "architecture": {"family": "deepseek_v4_flash"},
            "required_ops": ["iqk_dequant", "nls_dequant"],
        })
