"""Contention contracts for IQ_K direct-copy storage replacement."""

from __future__ import annotations

import threading
import time

import pytest

from moespresso.runtime import expert_slot_pool as slots
from moespresso.runtime.expert_index import build_expert_index
import test_iqk_experts_install as fixtures


def _pools(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "E", 4)
    package, _reference = fixtures._package(
        tmp_path,
        model_width=2048,
        expert_width=2048,
    )
    index = build_expert_index(package)
    return tuple(
        slots.ExpertSlotPool(
            package_dir=package,
            index=index,
            layer=0,
            projection=projection,
            capacity=1,
            spare_slots=1,
        )
        for projection in fixtures.PROJECTIONS
    )


def _wait_for_growth_gate(pools):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        states = []
        for pool in pools:
            with pool._bk_lock:
                states.append(pool._growth_pending)
        if all(states):
            return
        time.sleep(0.001)
    raise TimeoutError("growth did not close every pool mutation gate")


def _start(call, errors, *, name):
    def run():
        try:
            call()
        except BaseException as exc:  # surfaced on the owner thread below
            errors.append(exc)

    thread = threading.Thread(target=run, name=name)
    thread.start()
    return thread


@pytest.mark.parametrize("writer", ["ensure", "prefetch"])
def test_growth_rebinds_only_after_active_direct_copy_finishes(
    tmp_path,
    monkeypatch,
    writer,
):
    pools = _pools(tmp_path, monkeypatch)
    old_plans = tuple(pool._iqk_copy_plan for pool in pools)
    old_modules = tuple(pool.iqk for pool in pools)
    old_views = tuple(pool._iqk_views for pool in pools)
    assert all(plan is not None and plan.total_slots == 2 for plan in old_plans)

    copy_started = threading.Event()
    release_copy = threading.Event()
    copied = {}
    for ordinal, plan in enumerate(old_plans):
        original = plan.copy

        def observed(source, *, slot, _ordinal=ordinal, _plan=plan, _original=original):
            _original(source, slot=slot)
            copied[_ordinal] = (
                slot,
                tuple(field[2][slot].tobytes() for field in _plan.fields),
            )
            if _ordinal == 0:
                copy_started.set()
                if not release_copy.wait(5):
                    raise TimeoutError("test did not release the direct-copy writer")

        monkeypatch.setattr(plan, "copy", observed)

    if writer == "ensure":

        def write():
            pools[0].ensure([0])

    else:

        def write():
            assert pools[0].prefetch([0], reserve_floor=0) == 1

    expert = 0
    errors = []
    writer_thread = _start(write, errors, name=f"direct-copy-{writer}")
    assert copy_started.wait(5)
    assert pools[0]._loads_inflight == 1
    assert pools[0]._prefetch_inflight == int(writer == "prefetch")
    assert pools[0]._expert_at[0] == expert
    assert expert not in pools[0]._slot_of

    growth_thread = _start(
        lambda: slots.grow_expert_slot_pools(pools, 2),
        errors,
        name=f"growth-behind-{writer}",
    )
    _wait_for_growth_gate(pools)
    assert growth_thread.is_alive()
    assert tuple(pool._iqk_copy_plan for pool in pools) == old_plans
    assert tuple(pool.iqk for pool in pools) == old_modules
    assert tuple(pool._iqk_views for pool in pools) == old_views

    release_copy.set()
    writer_thread.join(5)
    growth_thread.join(5)
    assert not writer_thread.is_alive()
    assert not growth_thread.is_alive()
    assert errors == []

    assert set(copied) == {0}
    source_slot, source_bytes = copied[0]
    assert source_slot == 0
    assert pools[0].slot_of(expert) == 0
    assert (
        tuple(field[2][0].tobytes() for field in pools[0]._iqk_copy_plan.fields)
        == source_bytes
    )
    assert tuple(field[2][0].tobytes() for field in old_plans[0].fields) == source_bytes

    for pool, old_plan, old_module, views in zip(
        pools, old_plans, old_modules, old_views, strict=True
    ):
        assert pool.capacity == 2
        assert pool._loads_inflight == 0
        assert pool._prefetch_inflight == 0
        assert pool._growth_pending is False
        assert pool._iqk_copy_plan is not old_plan
        assert pool._iqk_copy_plan.total_slots == 3
        assert pool.iqk is not old_module
        assert pool._iqk_views is not views

    assert pools[0]._slot_of == {expert: 0}
    assert all(not pool._slot_of for pool in pools[1:])
