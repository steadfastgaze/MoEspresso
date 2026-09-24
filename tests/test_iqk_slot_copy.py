"""Exact stream-copy storage and shared-pool lifetime contracts."""

import gc
import subprocess
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from moespresso.package import iqk_relayout as rl
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1,
)
from moespresso.runtime import expert_slot_pool as slots
from moespresso.runtime.expert_index import build_expert_index
from moespresso.runtime.qwen4.expert_provider import Qwen4PooledSwitchGLU
import test_iqk_experts_install as fixtures


def _fresh_python(source, *args):
    return subprocess.run(
        [sys.executable, "-c", source, *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _storage(member, total_slots=3):
    geometry = SimpleNamespace(
        iqk_codec=member,
        in_features=256,
        out_features=8,
        packed_cols=rl.relayout_row_bytes(member, 256),
    )
    spans = rl.relayout_stream_spans(member, 256)
    sizes = {span.name: geometry.out_features * span.nbytes for span in spans}
    backing = {name: bytearray([0x5A]) * (32 + total_slots * size) for name, size in sizes.items()}
    views = {name: memoryview(value)[16:-16] for name, value in backing.items()}
    return geometry, spans, sizes, backing, views


@pytest.mark.parametrize("member", rl.RELAYOUT_MEMBERS)
def test_direct_plan_matches_split_streams_for_every_member_and_slot(member):
    geometry, spans, sizes, backing, views = _storage(member)
    plan = slots._IqkSlotCopyPlan(geometry, 3, views, sizes)
    rng = np.random.default_rng(4)
    for slot in range(3):
        source = rng.integers(0, 256, (geometry.out_features, geometry.packed_cols), np.uint8)
        before = {name: bytes(view) for name, view in views.items()}
        expected = rl.split_streams(member, source, geometry.in_features)
        plan.copy(memoryview(source), slot=slot)
        for span in spans:
            size = sizes[span.name]
            actual = bytes(views[span.name])
            assert actual[slot * size : (slot + 1) * size] == expected[span.name].tobytes()
            assert actual[: slot * size] == before[span.name][: slot * size]
            assert actual[(slot + 1) * size :] == before[span.name][(slot + 1) * size :]
            assert backing[span.name][:16] == bytes([0x5A]) * 16
            assert backing[span.name][-16:] == bytes([0x5A]) * 16


@pytest.mark.parametrize("slot", [-1, 3, True, np.int64(1)])
def test_invalid_slot_refuses_before_copy(slot):
    geometry, _, sizes, backing, views = _storage("iq2_k")
    plan = slots._IqkSlotCopyPlan(geometry, 3, views, sizes)
    before = {key: bytes(value) for key, value in backing.items()}
    with pytest.raises(ValueError, match="slot"):
        plan.copy(bytes(geometry.out_features * geometry.packed_cols), slot=slot)
    assert {key: bytes(value) for key, value in backing.items()} == before


@pytest.mark.parametrize("change", ["readonly", "size", "order", "noncontiguous", "geometry"])
def test_invalid_storage_refuses_at_plan_construction(change):
    geometry, _, sizes, _, views = _storage("iq2_k")
    name = next(iter(views))
    if change == "readonly":
        views[name] = views[name].toreadonly()
    elif change == "size":
        sizes[name] += 1
    elif change == "order":
        views = dict(reversed(list(views.items())))
    elif change == "noncontiguous":
        views[name] = views[name][::2]
    else:
        geometry.packed_cols += 1
    with pytest.raises(ValueError):
        slots._IqkSlotCopyPlan(geometry, 3, views, sizes)


@pytest.mark.parametrize("change", ["short", "long", "noncontiguous"])
def test_invalid_source_refuses_before_writing_any_stream(change):
    geometry, _, sizes, backing, views = _storage("iq2_k")
    plan = slots._IqkSlotCopyPlan(geometry, 3, views, sizes)
    size = geometry.out_features * geometry.packed_cols
    source = memoryview(bytes(size + (1 if change == "long" else -1)))
    if change == "noncontiguous":
        source = memoryview(bytes(size * 2))[::2]
    before = {key: bytes(value) for key, value in backing.items()}
    with pytest.raises(ValueError, match="source geometry"):
        plan.copy(source, slot=0)
    assert {key: bytes(value) for key, value in backing.items()} == before


def _pools(tmp_path, monkeypatch, *, spare_slots=0, layout=IQK_LAYOUT_IQK_RELAYOUT):
    monkeypatch.setattr(fixtures, "E", 4)
    package, _ = fixtures._package(tmp_path, layout=layout, model_width=2048, expert_width=2048)
    index = build_expert_index(package)
    return [
        slots.ExpertSlotPool(
            package_dir=package,
            index=index,
            layer=0,
            projection=name,
            capacity=1,
            spare_slots=spare_slots,
        )
        for name in fixtures.PROJECTIONS
    ]




@pytest.mark.parametrize("member", rl.RELAYOUT_MEMBERS)
def test_real_pool_constructs_plan_and_copies(
    tmp_path,
    monkeypatch,
    member,
):
    monkeypatch.setattr(fixtures, "E", 4)
    members = dict(fixtures.MEMBERS)
    members["gate_proj"] = member
    with np.errstate(invalid="ignore"):
        package, _ = fixtures._package(
            tmp_path,
            members=members,
            model_width=2560 if member == "iq3_k" else 2048,
            expert_width=2048,
        )
    result = _fresh_python(
        (
            "import sys; "
            "from pathlib import Path; "
            "from moespresso.runtime.expert_index import build_expert_index; "
            "from moespresso.runtime.expert_slot_pool import "
            "ExpertSlotPool; "
            "package = Path(sys.argv[1]); "
            "pool = ExpertSlotPool(package_dir=package, "
            "index=build_expert_index(package), layer=0, "
            "projection='gate_proj', capacity=1); "
            "pool.ensure([0]); "
            "print(int(pool._iqk_copy_plan is not None), "
            "pool.total_iqk_direct_copies, pool.total_loads)"
        ),
        package,
    )
    assert result.stdout.strip() == "1 1 1"


def test_plan_handles_spares_and_growth_without_reusing_old_storage(tmp_path, monkeypatch):
    pools = _pools(tmp_path, monkeypatch, spare_slots=1)
    for pool in pools:
        pool.ensure([0])
        assert pool._iqk_copy_plan.total_slots == 2
        pool._load_expert(expert=1, slot=pool.capacity)
    plans = [pool._iqk_copy_plan for pool in pools]
    old_bytes = [[field[2].tobytes() for field in plan.fields] for plan in plans]
    old_refs = [weakref.ref(field[2]) for plan in plans for field in plan.fields]
    slots.grow_expert_slot_pools(pools, 2)
    for pool, old_plan in zip(pools, plans, strict=True):
        assert pool._iqk_copy_plan is not old_plan
        assert pool._iqk_copy_plan.total_slots == 3
        assert 1 not in pool._slot_of
        assert [field[2][2].tobytes() for field in pool._iqk_copy_plan.fields] == [
            field[2][1].tobytes() for field in old_plan.fields
        ]
        pool.ensure([2])
        assert pool.total_iqk_direct_copies == pool.total_loads + 1
        assert pool._loads_inflight == 0
    assert [[field[2].tobytes() for field in plan.fields] for plan in plans] == old_bytes
    del old_plan, plans
    gc.collect()
    assert all(ref() is None for ref in old_refs)


@pytest.mark.parametrize("phase", ["plan", "prepare"])
def test_failed_growth_preserves_plans_and_streams(tmp_path, monkeypatch, phase):
    pools = _pools(tmp_path, monkeypatch)
    for pool in pools:
        pool.ensure([0])
    plans = [pool._iqk_copy_plan for pool in pools]
    streams = [pool._iqk_views for pool in pools]
    target = pools[1]
    name = "_new_iqk_copy_plan" if phase == "plan" else "_prepare_growth_locked"
    original = getattr(target, name)

    def fail(*_args, **_kwargs):
        raise MemoryError("synthetic copy-plan growth failure")

    monkeypatch.setattr(target, name, fail)
    with pytest.raises(MemoryError, match="copy-plan growth"):
        slots.grow_expert_slot_pools(pools, 2)
    for pool, plan, views in zip(pools, plans, streams, strict=True):
        assert pool.capacity == 1 and pool._iqk_copy_plan is plan and pool._iqk_views is views
        assert pool._loads_inflight == 0 and not pool._growth_pending
    monkeypatch.setattr(target, name, original)
    slots.grow_expert_slot_pools(pools, 2)
    assert all(pool.capacity == 2 for pool in pools)


def test_partial_copy_remains_unpublished_and_retry_succeeds(tmp_path, monkeypatch):
    pool = _pools(tmp_path, monkeypatch)[0]
    plan = pool._iqk_copy_plan
    original = plan.copy

    def partial(source, *, slot):
        rows = np.frombuffer(source, np.uint8).reshape(plan.rows, plan.row_bytes)
        start, end, target = plan.fields[0]
        np.copyto(target[slot], rows[:, start:end])
        raise OSError("synthetic partial copy")

    monkeypatch.setattr(plan, "copy", partial)
    with pytest.raises(OSError, match="partial copy"):
        pool.ensure([0])
    assert pool._expert_at == [None] and not pool._slot_of
    assert pool._loads_inflight == 0 and pool.total_iqk_direct_copies == 0
    monkeypatch.setattr(plan, "copy", original)
    pool.ensure([0])
    assert pool.resident_ids() == {0} and pool.total_iqk_direct_copies == 1


def _executor(pools):
    executor = Qwen4PooledSwitchGLU.__new__(Qwen4PooledSwitchGLU)
    object.__setattr__(executor, "_unique_projection_pools", lambda **_: pools)
    object.__setattr__(executor, "_prefetch_ticket", None)
    for name in ("_island_cache", "_mxfp4_kernel_cache", "_pipe_island_cache", "_pipe_buf_cache"):
        object.__setattr__(executor, name, {})
    return executor


@pytest.mark.parametrize("busy", [None, "_loads_inflight", "_prefetch_inflight", "_growth_pending"])
def test_close_releases_plan_views_only_after_writers_stop(tmp_path, monkeypatch, busy):
    pools = _pools(tmp_path, monkeypatch)
    references = [weakref.ref(field[2]) for pool in pools for field in pool._iqk_copy_plan.fields]
    executor = _executor(pools)
    if busy:
        setattr(pools[0], busy, 1)
        with pytest.raises(RuntimeError, match="not quiescent"):
            executor.close()
        assert all(pool._iqk_copy_plan is not None for pool in pools)
        setattr(pools[0], busy, 0)
    executor.close()
    executor.close()
    gc.collect()
    assert all(pool._iqk_copy_plan is None for pool in pools)
    assert all(ref() is None for ref in references)


def test_stream_major_keeps_its_existing_copy_path(tmp_path, monkeypatch):
    pools = _pools(tmp_path, monkeypatch, layout=IQK_LAYOUT_QWEN4_STREAM_MAJOR_V1)
    for pool in pools:
        assert pool._iqk_copy_plan is None
        component = pool.index.locate(
            layer=0, expert=0, projection=pool.projection, component="blocks"
        )
        with (pool.package_dir / component.shard).open("rb") as source:
            source.seek(component.offset)
            packed = np.frombuffer(source.read(component.nbytes), np.uint8).reshape(
                pool.geometry.out_features, pool.geometry.packed_cols
            )
        rows = rl.unpack_stream_major(pool.geometry.iqk_codec, packed, pool.geometry.in_features)
        expected = rl.split_streams(pool.geometry.iqk_codec, rows, pool.geometry.in_features)
        pool.ensure([0])
        assert pool.total_iqk_direct_copies == 0
        assert {name: bytes(view) for name, view in pool._iqk_views.items()} == {
            name: value.tobytes() for name, value in expected.items()
        }
