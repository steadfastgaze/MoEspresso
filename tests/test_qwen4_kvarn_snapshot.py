from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from moespresso.runtime.disk_kv import load_prompt_cache_payload, save_prompt_cache_payload
from moespresso.runtime.qwen4.kvarn_mutable import Qwen4MutableKVarNStorage
from moespresso.runtime.qwen4.kvarn_snapshot import restore_kvarn_state, snapshot_kvarn_state
from moespresso.runtime.qwen4.primitives import Qwen4RMSNorm
from moespresso.runtime.qwen4.qsa_kvarn import Qwen4MutableKVarNQSAState


@pytest.fixture(scope="module")
def history():
    count = 17_024
    rng = np.random.default_rng(922)
    keys = mx.array(rng.normal(size=(1, 2, count, 256)).astype(np.float32)).astype(mx.bfloat16)
    values = mx.array(rng.normal(size=(1, 2, count, 256)).astype(np.float32)).astype(mx.bfloat16)
    index = mx.array(rng.normal(size=(1, count, 128)).astype(np.float32)).astype(mx.bfloat16)
    positions = mx.stack([mx.arange(count, dtype=mx.int64) * n for n in (1, 3, 5)])[:, None]
    mx.eval(keys, values, index, positions)
    return keys, values, index, positions


def append(storage, history, stop):
    keys, values, index, positions = history
    norm = Qwen4RMSNorm(128)
    while storage.frontier < stop:
        start = storage.frontier
        end = min(stop, start + 1024)
        _, update = storage.prepare_index(
            index[:, start:end], positions[..., start:end], norm,
            rotary_dim=64, rope_base=10_000_000.0, mrope_section=(11, 11, 10),
        )
        view = storage.append(
            keys[..., start:end, :], values[..., start:end, :],
            mx.ones((1, end - start), dtype=mx.bool_), index_update=update,
        )
        mx.eval(*storage.state_arrays())
        storage.commit(view)
    return Qwen4MutableKVarNQSAState(storage, storage.checkpoint())


def same_arrays(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        a, b = mx.contiguous(a), mx.contiguous(b)
        mx.eval(a, b)
        if a.size:
            assert bytes(memoryview(a).cast("B")) == bytes(memoryview(b).cast("B"))


@pytest.mark.parametrize("frontier", [0, 127, 128, 256, 8447, 8448, 8449, 16643])
def test_snapshot_roundtrip_preserves_packed_rows_and_future_appends(tmp_path, history, frontier):
    source = Qwen4MutableKVarNStorage(max_context_tokens=17_024, position_dtype=mx.int64)
    state = append(source, history, frontier)
    arrays, meta = snapshot_kvarn_state(state)
    assert sum(a.nbytes for a in arrays) == source.logical_nbytes
    assert sum(a.nbytes for a in arrays) < source.allocated_nbytes
    path, _ = save_prompt_cache_payload(
        tmp_path, "snapshot", cache_state_trees=[arrays], meta_state_trees=[meta],
        safety_metadata={},
    )
    trees, metas, _ = load_prompt_cache_payload(tmp_path, path)
    restored = restore_kvarn_state(trees[0], metas[0], max_context_tokens=17_024)
    assert restored.storage is not source
    assert restored.view._storage_identity is not state.view._storage_identity
    assert restored.view.mutation_cursor == 0
    same_arrays(arrays, snapshot_kvarn_state(restored)[0])
    indices = mx.array([[[0, 127, 128, 255, frontier - 1, frontier, -1]]], dtype=mx.int32)
    if frontier:
        same_arrays(source.gather_selected_rows(indices), restored.storage.gather_selected_rows(indices))
    next_source = append(source, history, frontier + 129)
    next_restored = append(restored.storage, history, frontier + 129)
    same_arrays(snapshot_kvarn_state(next_source)[0], snapshot_kvarn_state(next_restored)[0])
    # Advancing either live owner must not change the captured checkpoint.
    same_arrays(arrays, trees[0])
    again = restore_kvarn_state(arrays, meta, max_context_tokens=17_024)
    same_arrays(arrays, snapshot_kvarn_state(again)[0])


def test_snapshot_rejects_uncommitted_or_stale_views(history):
    storage = Qwen4MutableKVarNStorage(max_context_tokens=1024, position_dtype=mx.int64)
    old = append(storage, history, 256)
    keys, values, index, positions = history
    _, update = storage.prepare_index(
        index[:, 256:257], positions[..., 256:257], Qwen4RMSNorm(128),
        rotary_dim=64, rope_base=10_000_000.0, mrope_section=(11, 11, 10),
    )
    pending = storage.append(
        keys[..., 256:257, :], values[..., 256:257, :],
        mx.ones((1, 1), dtype=mx.bool_), index_update=update,
    )
    with pytest.raises(ValueError, match="committed"):
        snapshot_kvarn_state(Qwen4MutableKVarNQSAState(storage, pending))
    with pytest.raises(ValueError):
        snapshot_kvarn_state(old)
    committed = storage.commit(pending)
    snapshot_kvarn_state(Qwen4MutableKVarNQSAState(storage, committed))


@pytest.mark.parametrize("bad", [
    {"schema": "future"}, {"layout": "other"}, {"frontier": True},
    {"frontier": -1}, {"frontier": 2048}, {"position_dtype": "mlx.core.float32"},
    {"extra": 1},
])
def test_snapshot_rejects_invalid_metadata_before_allocation(history, monkeypatch, bad):
    storage = Qwen4MutableKVarNStorage(max_context_tokens=512, position_dtype=mx.int64)
    arrays, meta = snapshot_kvarn_state(append(storage, history, 256))

    def refuse(**_kwargs):
        raise AssertionError("invalid payload must not allocate storage")

    monkeypatch.setattr("moespresso.runtime.qwen4.kvarn_snapshot.Qwen4MutableKVarNStorage", refuse)
    with pytest.raises(ValueError):
        restore_kvarn_state(arrays, {**meta, **bad}, max_context_tokens=512)


def test_snapshot_rejects_malformed_tensors_and_nonfinite_values(history):
    storage = Qwen4MutableKVarNStorage(max_context_tokens=512, position_dtype=mx.int64)
    arrays, meta = snapshot_kvarn_state(append(storage, history, 256))
    variants = [arrays[:-1], (arrays[0].astype(mx.int32), *arrays[1:])]
    variants.append((arrays[0], mx.full(arrays[1].shape, float("nan"), dtype=arrays[1].dtype), *arrays[2:]))
    for bad in variants:
        with pytest.raises(ValueError):
            restore_kvarn_state(bad, meta, max_context_tokens=512)


def test_snapshot_restores_into_different_safe_capacity(history):
    storage = Qwen4MutableKVarNStorage(max_context_tokens=17_024, position_dtype=mx.int64)
    arrays, meta = snapshot_kvarn_state(append(storage, history, 256))
    restored = restore_kvarn_state(arrays, meta, max_context_tokens=512)
    assert restored.storage.max_context_tokens == 512
    same_arrays(arrays, snapshot_kvarn_state(restored)[0])
