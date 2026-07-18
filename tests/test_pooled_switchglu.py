"""Pooled SwitchGLU correctness.

These tests rebuild the useful part of the old SSD work on the new direct-buffer
primitive: fixed MLX pools filled by `pread_into`, then JANG's gather kernel over
the pool. No Python weight stacking is used.
"""

from __future__ import annotations

import json
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant.tq_kernel")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from moespresso.runtime.expert_index import build_expert_index  # noqa: E402
from moespresso.runtime.expert_slot_pool import (  # noqa: E402
    ExpertCapacityExceeded,
    ExpertSlotPool,
)
from moespresso.runtime.pooled_switchglu import (  # noqa: E402
    PooledMxfp4SwitchLinear,
    PooledKQuantSwitchLinear,
    PooledSparseMoeBlock,
    PooledSwitchGLU,
    PooledTurboQuantSwitchLinear,
    _should_sort_routed_indices,
)
from moespresso.runtime.owned_switchglu import OwnedSwitchGLU  # noqa: E402
from moespresso.probe.deepseek_v4.codec import dequant_fp4_e2m1_ue8m0  # noqa: E402


def _write_safetensors(path, tensors, metadata=None):
    header, blob, off = {}, bytearray(), 0
    if metadata:
        header["__metadata__"] = dict(metadata)
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [off, off + len(raw)],
        }
        blob += raw
        off += len(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def test_q6_down_dedicated_qmv_is_shape_and_codec_guarded(monkeypatch):
    from moespresso.runtime import pooled_switchglu as psg

    calls = {"qmv": 0, "qmm": 0}

    def gather_qmv_kq(x, weight, codec, indices):
        del weight
        calls["qmv"] += 1
        assert codec == "q6_k"
        assert tuple(x.shape) == (1, 8, 512)
        assert tuple(indices.shape) == (1, 8)
        return mx.zeros((1, 8, 2048), dtype=mx.bfloat16)

    def gather_qmm(x, weight, scales, codec, **kwargs):
        del weight, scales, codec, kwargs
        calls["qmm"] += 1
        return mx.zeros((*x.shape[:-1], 2048), dtype=x.dtype)

    monkeypatch.setitem(
        sys.modules,
        "mlx_kquant",
        SimpleNamespace(
            gather_qmv_kq=gather_qmv_kq,
            gather_qmm=gather_qmm,
        ),
    )
    monkeypatch.setattr(psg, "_QWEN_DOWN_Q6_QMV", True)
    projection = SimpleNamespace(
        pool=SimpleNamespace(
            projection="down_proj",
            weight=mx.zeros((4, 2048, 420), dtype=mx.uint8),
            scales=mx.zeros((1,), dtype=mx.uint8),
        ),
        kquant_type="q6_k",
        in_features=512,
        out_features=2048,
        matmul_slot_calls=0,
        matmul_slot_elements=0,
        decode_q6_qmv_calls=0,
    )
    x = mx.zeros((1, 1, 8, 1, 512), dtype=mx.bfloat16)
    indices = mx.zeros((1, 1, 8), dtype=mx.uint32)

    out = PooledKQuantSwitchLinear.matmul_slots(projection, x, indices)
    assert tuple(out.shape) == (1, 1, 8, 1, 2048)
    assert projection.decode_q6_qmv_calls == 1
    assert calls == {"qmv": 1, "qmm": 0}

    projection.kquant_type = "q4_k"
    PooledKQuantSwitchLinear.matmul_slots(projection, x, indices)
    assert projection.decode_q6_qmv_calls == 1
    assert calls == {"qmv": 1, "qmm": 1}


def _resident_projection(n_experts, in_features, out_features, *, bits=2, seed=42):
    from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear

    mod = TurboQuantSwitchLinear(
        in_features, out_features, n_experts, bits=bits, seed=seed)
    vals_per_u32 = 32 // bits
    cols = (in_features + vals_per_u32 - 1) // vals_per_u32
    mod.packed = mx.random.randint(
        0, 2**31, (n_experts, out_features, cols)).astype(mx.uint32)
    mod.norms = (mx.random.normal((n_experts, out_features)) * 0.1).astype(mx.float16)
    mx.eval(mod.packed, mod.norms)
    return mod


def _resident_switch(*, n_experts=32, in_features=64, hidden_features=32,
                     gate_bits=2, up_bits=4, down_bits=2):
    from mlx_lm.models.switch_layers import SwitchGLU

    mx.random.seed(17)
    shape = SwitchGLU(in_features, hidden_features, n_experts)
    gate = _resident_projection(
        n_experts, in_features, hidden_features, bits=gate_bits)
    up = _resident_projection(
        n_experts, in_features, hidden_features, bits=up_bits)
    down = _resident_projection(
        n_experts, hidden_features, in_features, bits=down_bits)
    return OwnedSwitchGLU(
        gate_proj=gate,
        up_proj=up,
        down_proj=down,
        activation=shape.activation,
    )


def _package_from_resident(tmp_path, resident):
    """A bundle-format package carrying the resident module's exact bytes, so
    pooled-vs-resident equivalence tests compare bit-identical expert data."""
    from moespresso.package.bundle import (
        assemble_layer_bundle,
        encode_bundle_metadata,
    )

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    comps, bits = {}, {}
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(resident, proj_name)
        comps[(proj_name, "packed")] = np.array(proj.packed)
        comps[(proj_name, "norms")] = np.array(proj.norms)
        bits[proj_name] = int(proj.bits)
    bundle, geo = assemble_layer_bundle(comps, bits)
    base = "language_model.model.layers.0.mlp.switch_mlp"
    _write_safetensors(pkg / "model-00001-of-00001.safetensors", {
        f"{base}.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes()),
    }, metadata={"expert_bundles": encode_bundle_metadata({0: geo})})
    return pkg


def _pack_mxfp4_source(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed_i8 = (
        codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)
    ).astype(np.uint8).view(np.int8)
    n_exp, rows, packed_byte_cols = packed_i8.shape
    cols = packed_byte_cols * 2
    packed = (
        np.ascontiguousarray(packed_i8)
        .view(np.uint8)
        .reshape(n_exp, rows, cols // 8, 4)
        .copy()
        .view(np.uint32)
        .reshape(n_exp, rows, cols // 8)
    )
    return packed, packed_i8


def _mxfp4_projection(
    rng: np.random.Generator,
    *,
    n_experts: int,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = rng.integers(0, 16, (n_experts, rows, cols), dtype=np.uint8)
    packed, packed_i8 = _pack_mxfp4_source(codes)
    scales = rng.integers(126, 129, (n_experts, rows, cols // 32), dtype=np.uint8)
    dense = np.stack(
        [
            dequant_fp4_e2m1_ue8m0(
                packed_i8[e],
                scales[e],
                out_dtype=np.float32,
            )
            for e in range(n_experts)
        ],
        axis=0,
    )
    return packed, scales, dense


class _SwigluActivation:
    swiglu_limit = 10.0

    def __call__(self, x_up, x_gate):
        return nn.silu(x_gate) * x_up


def _package_from_mxfp4_source(
    tmp_path,
    *,
    n_experts=8,
    in_features=256,
    hidden_features=64,
    seed=321,
):
    from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata

    rng = np.random.default_rng(seed)
    pkg = tmp_path / "pkg_mxfp4"
    pkg.mkdir()
    comps, dense = {}, {}
    for proj, rows, cols in (
        ("gate_proj", hidden_features, in_features),
        ("up_proj", hidden_features, in_features),
        ("down_proj", in_features, hidden_features),
    ):
        packed, scales, weights = _mxfp4_projection(
            rng,
            n_experts=n_experts,
            rows=rows,
            cols=cols,
        )
        comps[(proj, "packed")] = packed
        comps[(proj, "scales")] = scales
        dense[proj] = weights
    bits = {proj: 4 for proj in ("gate_proj", "up_proj", "down_proj")}
    codecs = {proj: "mxfp4" for proj in ("gate_proj", "up_proj", "down_proj")}
    bundle, geo = assemble_layer_bundle(comps, bits, codecs=codecs)
    base = "language_model.model.layers.0.mlp.switch_mlp"
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors",
        {f"{base}.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes())},
        metadata={"expert_bundles": encode_bundle_metadata({0: geo})},
    )
    return pkg, dense


def _pooled_mxfp4_projection(pkg, index, projection, *, capacity):
    return PooledMxfp4SwitchLinear(
        package_dir=pkg,
        index=index,
        layer=0,
        projection=projection,
        capacity=capacity,
    )


def _pooled_mxfp4_switch(pkg, index, *, capacity):
    return PooledSwitchGLU(
        gate_proj=_pooled_mxfp4_projection(
            pkg, index, "gate_proj", capacity=capacity),
        up_proj=_pooled_mxfp4_projection(
            pkg, index, "up_proj", capacity=capacity),
        down_proj=_pooled_mxfp4_projection(
            pkg, index, "down_proj", capacity=capacity),
        activation=_SwigluActivation(),
    )


def _mxfp4_switch_reference(x: np.ndarray, indices: np.ndarray, dense: dict):
    expected = []
    for expert in indices.reshape(-1).tolist():
        expert = int(expert)
        gate = x @ dense["gate_proj"][expert].T
        up = x @ dense["up_proj"][expert].T
        gate = np.minimum(gate, 10.0)
        up = np.clip(up, -10.0, 10.0)
        act = (gate / (1.0 + np.exp(-gate)) * up).astype(np.float16).astype(np.float32)
        expected.append((act @ dense["down_proj"][expert].T)[0])
    return np.asarray(expected, dtype=np.float32).reshape(
        *indices.shape,
        dense["down_proj"].shape[1],
    )


def _pooled_projection(pkg, index, resident, projection, *, capacity,
                       spare_slots=0):
    proj = getattr(resident, projection)
    return PooledTurboQuantSwitchLinear(
        package_dir=pkg,
        index=index,
        layer=0,
        projection=projection,
        capacity=capacity,
        codebook=proj.codebook,
        signs=proj.signs,
        spare_slots=spare_slots,
    )


def _pooled_switch(pkg, index, resident, *, capacity, spare_slots=0):
    return PooledSwitchGLU(
        gate_proj=_pooled_projection(pkg, index, resident, "gate_proj",
                                     capacity=capacity,
                                     spare_slots=spare_slots),
        up_proj=_pooled_projection(pkg, index, resident, "up_proj",
                                   capacity=capacity,
                                   spare_slots=spare_slots),
        down_proj=_pooled_projection(pkg, index, resident, "down_proj",
                                     capacity=capacity,
                                     spare_slots=spare_slots),
        activation=resident.activation,
    )


class _TinySharedMLP(nn.Module):
    def __init__(self, hidden=64, intermediate=32):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinySparseMoeBlock(nn.Module):
    def __init__(self, *, gate, switch_mlp, shared_expert, shared_expert_gate,
                 top_k=4, norm_topk_prob=True, num_experts=16):
        super().__init__()
        self.gate = gate
        self.switch_mlp = switch_mlp
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.num_experts = num_experts
        self.sharding_group = None

    def __call__(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y


def _assert_same(a, b):
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


def test_slot_pool_forced_cold_miss_loads_expert_into_slot(tmp_path):
    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg,
        index=index,
        layer=0,
        projection="gate_proj",
        capacity=4,
    )
    pool.ensure([0, 2, 4])
    assert pool.resident_ids() == {0, 2, 4}

    remapped = pool.remap(mx.array([[0, 2, 4, 6]], dtype=mx.uint32))

    assert pool.resident_ids() == {0, 2, 4, 6}
    assert pool.total_misses == 4
    assert pool.total_hits == 3
    assert int(np.array(remapped)[0, 3]) == pool.slot_of(6)


def test_slot_pool_failed_load_is_fail_closed(tmp_path, monkeypatch):
    """A pread failure mid-batch must not leave experts published as resident
    over stale slot bytes on the two-phase ensure: only fully-loaded experts
    stay resident, failed reservations are released, and a retry recovers
    cleanly."""
    import moespresso.runtime.expert_slot_pool as esp

    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg,
        index=index,
        layer=0,
        projection="gate_proj",
        capacity=4,
    )

    real_pread = esp.pread_view_cached
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:  # second component of the second expert in batch
            raise OSError("injected pread failure")
        return real_pread(*args, **kwargs)

    monkeypatch.setattr(esp, "pread_view_cached", flaky)
    with pytest.raises(OSError, match="injected"):
        pool.ensure([0, 2, 4])

    # expert 0 loaded fully before the failure -> resident; 2 and 4 must not be
    assert pool.resident_ids() == {0}
    # the failed/unloaded reservations were released (slots free again)
    assert pool.free_slots() == 3

    # retry with the intact reader recovers everything
    monkeypatch.setattr(esp, "pread_view_cached", real_pread)
    pool.ensure([0, 2, 4])
    assert pool.resident_ids() == {0, 2, 4}
    # and the recovered pool serves a remap without touching absent slots
    remapped = pool.remap_ondevice(mx.array([[0, 2, 4]], dtype=mx.uint32))
    assert int(np.array(remapped).max()) < pool.capacity


def test_slot_pool_protect_pins_residents_against_eviction(tmp_path):
    """Chunk-ahead: ensure(protect=...) must never evict protected experts
    even when they are not in the active set being loaded."""
    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg,
        index=index,
        layer=0,
        projection="gate_proj",
        capacity=4,
    )
    pool.ensure([0, 1])           # chunk i
    pool.ensure([2, 3], protect={0, 1})  # fills the pool
    # loading two new experts must evict only the unprotected {2, 3}
    pool.ensure([4, 5], protect={0, 1})
    assert {0, 1} <= pool.resident_ids()
    assert pool.resident_ids() == {0, 1, 4, 5}


def test_slot_pool_prefetch_protect_pins_residents_against_eviction(tmp_path):
    """Cross-chunk ticket safety: prefetch(protect=...) must never evict the
    protected experts (the submitting call's final capacity-chunk, whose
    readers may still be in flight at submit time), on top of the
    _demand_protect set the last ensure recorded."""
    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)

    def _pool():
        return ExpertSlotPool(
            package_dir=pkg,
            index=index,
            layer=0,
            projection="gate_proj",
            capacity=4,
        )

    pool = _pool()
    pool.ensure([0, 1])
    pool.ensure([2, 3], protect={0, 1})  # fills; _demand_protect is {2, 3}
    loaded = pool.prefetch([4, 5, 6], protect={0, 1}, reserve_floor=0)
    # Every resident is protected (explicit {0, 1} plus _demand_protect
    # {2, 3}), so the prefetch places nothing and evicts nothing.
    assert loaded == 0
    assert pool.resident_ids() == {0, 1, 2, 3}
    assert pool.total_prefetch_skips >= 1

    # The explicit protect is load-bearing: without it the same prefetch
    # evicts from {0, 1} (the residents outside _demand_protect).
    unprotected = _pool()
    unprotected.ensure([0, 1])
    unprotected.ensure([2, 3], protect={0, 1})
    loaded = unprotected.prefetch([4, 5, 6], reserve_floor=0)
    assert loaded > 0
    assert not {0, 1} <= unprotected.resident_ids()
    assert {2, 3} <= unprotected.resident_ids()


def test_prefill_chunked_paths_match_resident_reference(tmp_path, monkeypatch):
    """Sorted chunked prefill is bit-exact vs the resident reference both in
    half-capacity chunk-ahead overlap mode (capacity >= 2) and in the one-slot
    eval-per-chunk mode (capacity 1, where two chunks cannot coexist)."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", False)

    resident = _resident_switch(n_experts=16, gate_bits=2, up_bits=4)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)

    # 20 tokens x 4 experts, indices.size >= 64 -> sorted chunked
    mx.random.seed(3)
    xs = mx.random.normal((20, 64)).astype(mx.float16)
    idxs = mx.random.randint(0, 16, (20, 4)).astype(mx.uint32)
    mx.eval(xs, idxs)

    ref = resident(xs, idxs)
    mx.eval(ref)
    for capacity in (1, 6):
        pooled = _pooled_switch(pkg, index, resident, capacity=capacity)
        y = pooled(xs, idxs)
        mx.eval(y)
        assert pooled.sorted_chunked_calls == 1
        assert np.array_equal(np.array(ref), np.array(y))


def test_slot_pool_rejects_active_set_larger_than_capacity(tmp_path):
    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg,
        index=index,
        layer=0,
        projection="gate_proj",
        capacity=2,
    )

    with pytest.raises(ExpertCapacityExceeded, match="capacity 2"):
        pool.remap(mx.array([[0, 1, 2]], dtype=mx.uint32))


def test_slot_pool_remap_loaded_does_not_reload_or_recount_hits(tmp_path):
    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg,
        index=index,
        layer=0,
        projection="gate_proj",
        capacity=4,
    )
    pool.ensure([1, 3])
    hits = pool.total_hits
    loads = pool.total_loads

    remapped = pool.remap_loaded(np.array([1, 3], dtype=np.uint32), (1, 2))

    assert pool.total_hits == hits
    assert pool.total_loads == loads
    assert np.array(remapped).tolist() == [[pool.slot_of(1), pool.slot_of(3)]]


def test_lfu_eviction_preserves_historical_frequency(tmp_path):
    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg,
        index=index,
        layer=0,
        projection="gate_proj",
        capacity=2,
        eviction_policy="lfu",
    )

    pool.ensure([0])
    pool.ensure([1])
    pool.ensure([0])
    pool.ensure([2])
    assert 1 not in pool.resident_ids()
    assert pool._freq[1] == 1

    pool.ensure([1])

    assert pool._freq[1] == 2


def test_slot_pool_seed_hot_fills_new_slots_from_frequency_history(tmp_path):
    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg,
        index=index,
        layer=0,
        projection="gate_proj",
        capacity=2,
        eviction_policy="lfu",
    )
    pool.ensure([0])
    pool.ensure([1])
    pool.ensure([0])
    pool.ensure([2])
    assert pool.resident_ids() == {0, 2}

    pool.grow(4)
    seeded = pool.seed_hot()

    assert 1 in seeded
    assert {0, 1, 2}.issubset(pool.resident_ids())


def test_pooled_projection_matches_resident_decode_with_forced_miss(tmp_path):
    resident = _resident_switch(n_experts=16)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_projection(pkg, index, resident, "gate_proj", capacity=4)
    pooled.pool.ensure([3, 5, 7])
    x = mx.random.normal((1, 1, 1, 64)).astype(mx.float16)
    indices = mx.array([[3, 5, 7, 11]], dtype=mx.uint32)

    _assert_same(
        resident.gate_proj(x, indices),
        pooled(x, indices),
    )
    assert 11 in pooled.pool.resident_ids()


def test_pooled_switchglu_matches_resident_unsorted_decode(tmp_path):
    resident = _resident_switch(n_experts=16)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_switch(pkg, index, resident, capacity=4)
    x = mx.random.normal((1, 64)).astype(mx.float16)
    indices = mx.array([[3, 7, 1, 5]], dtype=mx.uint32)

    _assert_same(resident(x, indices), pooled(x, indices))


def test_pooled_switchglu_all_resident_decode_skips_projection_wait(tmp_path):
    resident = _resident_switch(n_experts=16)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_switch(pkg, index, resident, capacity=4)
    x = mx.random.normal((1, 64)).astype(mx.float16)
    indices = mx.array([[3, 7, 1, 5]], dtype=mx.uint32)

    _assert_same(resident(x, indices), pooled(x, indices))
    assert pooled.projection_load_wait_calls == 1
    assert pooled.projection_no_miss_calls == 0

    _assert_same(resident(x, indices), pooled(x, indices))
    assert pooled.projection_load_wait_calls == 1
    assert pooled.projection_no_miss_calls == 1
    assert pooled.gate_proj.pool.total_hits >= 4
    assert pooled.up_proj.pool.total_hits >= 4
    assert pooled.down_proj.pool.total_hits >= 4


def test_mxfp4_slot_pool_loads_packed_and_scales(tmp_path):
    pkg, _dense = _package_from_mxfp4_source(tmp_path)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg,
        index=index,
        layer=0,
        projection="gate_proj",
        capacity=4,
    )

    pool.ensure([0, 2, 4])

    assert pool.codec == "mxfp4"
    assert pool.norms is None
    assert pool.scales is not None
    assert pool.components == ("packed", "scales")
    assert pool.resident_ids() == {0, 2, 4}
    assert pool.scales.shape == (4, 64, 8)


def test_pooled_mxfp4_switchglu_matches_decoded_fp4_reference(tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_COMPILED_ISLAND", True)
    monkeypatch.setattr(psg, "_ONDEVICE_REMAP", True)
    pkg, dense = _package_from_mxfp4_source(tmp_path)
    index = build_expert_index(pkg)
    pooled = _pooled_mxfp4_switch(pkg, index, capacity=4)
    pooled.eval()
    x_np = (np.random.default_rng(55).standard_normal((1, 256)) * 0.03).astype(np.float32)
    x = mx.array(x_np)
    indices_np = np.array([[3, 7, 1, 5]], dtype=np.uint32)
    indices = mx.array(indices_np)

    got = pooled(x, indices)
    mx.eval(got)
    expected = _mxfp4_switch_reference(x_np, indices_np, dense)

    assert pooled._all_mxfp4
    assert pooled.compiled_island_calls == 1
    assert pooled.fused_gate_up_calls == 1
    rel = np.linalg.norm(np.array(got) - expected) / max(np.linalg.norm(expected), 1e-12)
    assert rel < 1e-5


def test_pooled_mxfp4_switchglu_fallback_prefill_matches_fp4_reference(
    tmp_path,
    monkeypatch,
):
    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_COMPILED_ISLAND", False)
    pkg, dense = _package_from_mxfp4_source(tmp_path)
    index = build_expert_index(pkg)
    pooled = _pooled_mxfp4_switch(pkg, index, capacity=8)
    pooled.eval()
    rng = np.random.default_rng(57)
    x_np = (rng.standard_normal((16, 256)) * 0.03).astype(np.float32)
    indices_np = np.asarray(
        [[(row + offset) % 8 for offset in range(4)] for row in range(16)],
        dtype=np.uint32,
    )
    x = mx.array(x_np)
    indices = mx.array(indices_np)

    got = pooled(x, indices)
    mx.eval(got)
    expected = np.concatenate(
        [
            _mxfp4_switch_reference(x_np[row:row + 1], indices_np[row:row + 1], dense)
            for row in range(indices_np.shape[0])
        ],
        axis=0,
    )

    assert pooled.compiled_island_calls == 0
    assert pooled.sorted_chunked_calls == 0
    rel = np.linalg.norm(np.array(got) - expected) / max(np.linalg.norm(expected), 1e-12)
    assert rel < 3e-3


def test_pooled_mxfp4_pipelined_builder_matches_direct_decode(tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_ONDEVICE_REMAP", True)
    pkg, _dense = _package_from_mxfp4_source(tmp_path)
    index = build_expert_index(pkg)
    direct = _pooled_mxfp4_switch(pkg, index, capacity=4)
    piped = _pooled_mxfp4_switch(pkg, index, capacity=4)
    direct.eval()
    piped.eval()
    x = mx.array((np.random.default_rng(56).standard_normal((1, 256)) * 0.03).astype(np.float32))
    indices = mx.array(np.array([[3, 7, 1, 5]], dtype=np.uint32))

    expected = direct(x, indices)
    piped.publish_slots(indices)
    got = piped.build_pipelined(x, indices)
    mx.eval(expected, got)

    rel = np.linalg.norm(np.array(got) - np.array(expected)) / max(
        np.linalg.norm(np.array(expected)),
        1e-12,
    )
    assert rel < 1e-6
    assert piped.compiled_island_calls == 1
    assert piped.fused_gate_up_calls == 1


@pytest.mark.parametrize(
    ("mxfp4_projection", "tq_bits", "expect_fused_gate_up"),
    [
        ("gate_proj", {"gate_proj": 4, "up_proj": 2, "down_proj": 1}, False),
        ("down_proj", {"gate_proj": 1, "up_proj": 2, "down_proj": 4}, False),
        ("down_proj", {"gate_proj": 2, "up_proj": 2, "down_proj": 4}, True),
    ],
)
def test_mixed_mxfp4_tq_switchglu_uses_legal_projection_path(
    tmp_path,
    monkeypatch,
    mxfp4_projection,
    tq_bits,
    expect_fused_gate_up,
):
    import moespresso.runtime.pooled_switchglu as psg
    from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata

    monkeypatch.setattr(psg, "_FUSED_GATE_UP", True)
    monkeypatch.setattr(psg, "_COMPILED_ISLAND", True)
    monkeypatch.setattr(psg, "_ONDEVICE_REMAP", True)
    resident = _resident_switch(
        n_experts=8,
        in_features=256,
        hidden_features=64,
        gate_bits=tq_bits["gate_proj"],
        up_bits=tq_bits["up_proj"],
        down_bits=tq_bits["down_proj"],
    )
    rng = np.random.default_rng(58)
    comps = {
        ("gate_proj", "packed"): np.array(resident.gate_proj.packed),
        ("gate_proj", "norms"): np.array(resident.gate_proj.norms),
        ("up_proj", "packed"): np.array(resident.up_proj.packed),
        ("up_proj", "norms"): np.array(resident.up_proj.norms),
        ("down_proj", "packed"): np.array(resident.down_proj.packed),
        ("down_proj", "norms"): np.array(resident.down_proj.norms),
    }
    rows, cols = (64, 256) if mxfp4_projection == "gate_proj" else (256, 64)
    packed, scales, _dense = _mxfp4_projection(
        rng,
        n_experts=8,
        rows=rows,
        cols=cols,
    )
    comps.pop((mxfp4_projection, "norms"))
    comps[(mxfp4_projection, "packed")] = packed
    comps[(mxfp4_projection, "scales")] = scales
    codecs = {p: "tq" for p in ("gate_proj", "up_proj", "down_proj")}
    codecs[mxfp4_projection] = "mxfp4"
    bundle, geo = assemble_layer_bundle(
        comps,
        bits=tq_bits,
        codecs=codecs,
    )
    pkg = tmp_path / "pkg_mixed_mxfp4_tq"
    pkg.mkdir()
    base = "language_model.model.layers.0.mlp.switch_mlp"
    _write_safetensors(
        pkg / "model-00001-of-00001.safetensors",
        {f"{base}.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes())},
        metadata={"expert_bundles": encode_bundle_metadata({0: geo})},
    )
    index = build_expert_index(pkg)

    def projection(name):
        if name == mxfp4_projection:
            return _pooled_mxfp4_projection(pkg, index, name, capacity=4)
        return _pooled_projection(pkg, index, resident, name, capacity=4)

    pooled = PooledSwitchGLU(
        gate_proj=projection("gate_proj"),
        up_proj=projection("up_proj"),
        down_proj=projection("down_proj"),
        activation=_SwigluActivation(),
    )
    pooled.eval()
    x = mx.array((rng.standard_normal((1, 256)) * 0.03).astype(np.float32))
    indices = mx.array(np.array([[3, 7, 1, 5]], dtype=np.uint32))

    got = pooled(x, indices)
    x4 = mx.expand_dims(x, (-2, -3))
    gate_idx = pooled.gate_proj.pool.remap(indices)
    up_idx = pooled.up_proj.pool.remap(indices)
    down_idx = pooled.down_proj.pool.remap(indices)
    expected = pooled.down_proj.matmul_slots(
        pooled.activation(
            pooled.up_proj.matmul_slots(x4, up_idx, sorted_indices=False),
            pooled.gate_proj.matmul_slots(x4, gate_idx, sorted_indices=False),
        ),
        down_idx,
        sorted_indices=False,
    ).squeeze(-2)
    mx.eval(got, expected)

    assert not pooled._all_mxfp4
    assert pooled._fused_gate_up is expect_fused_gate_up
    if expect_fused_gate_up:
        assert pooled._fused_gate_up
        assert pooled.compiled_island_calls == 0
        assert pooled.fused_gate_up_calls == 1
    else:
        assert pooled.fused_gate_up_calls == 0
    np.testing.assert_allclose(np.array(got), np.array(expected), rtol=3e-2, atol=3e-3)


def test_pooled_switchglu_matches_resident_sorted_prefill(tmp_path):
    resident = _resident_switch(n_experts=32)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_switch(pkg, index, resident, capacity=16)
    x = mx.random.normal((20, 64)).astype(mx.float16)
    selected = [1, 3, 5, 7, 9, 11, 13, 15]
    indices = mx.array(
        [[selected[(t + j) % len(selected)] for j in range(4)] for t in range(20)],
        dtype=mx.uint32,
    )

    assert indices.size >= 64
    assert len(set(np.array(indices).reshape(-1).tolist())) < resident.gate_proj.num_experts
    _assert_same(resident(x, indices), pooled(x, indices))


def test_pooled_switchglu_prefill_sort_predicate_is_unconditional():
    small = mx.zeros((1, 6), dtype=mx.uint32)
    large = mx.zeros((20, 4), dtype=mx.uint32)

    assert not _should_sort_routed_indices(small)
    assert _should_sort_routed_indices(large)


def test_pooled_switchglu_chunks_prefill_when_active_set_exceeds_capacity(tmp_path):
    resident = _resident_switch(n_experts=16)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_switch(pkg, index, resident, capacity=4)
    x = mx.random.normal((20, 64)).astype(mx.float16)
    selected = [1, 3, 5, 7, 9, 11, 13, 15]
    indices = mx.array(
        [[selected[(t + j) % len(selected)] for j in range(4)] for t in range(20)],
        dtype=mx.uint32,
    )

    assert len(set(np.array(indices).reshape(-1).tolist())) > 4
    _assert_same(resident(x, indices), pooled(x, indices))


def test_remap_ondevice_matches_remap_loaded(tmp_path):
    """The on-device slot-table gather is numerically identical to the host remap."""
    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg, index=index, layer=0, projection="gate_proj", capacity=4,
    )
    pool.ensure([1, 3, 5, 7])

    for idx in (mx.array([[1, 3, 5, 7]], dtype=mx.uint32),
                mx.array([[7, 3, 3, 5]], dtype=mx.uint32)):
        host = pool.remap_loaded(np.asarray(idx).reshape(-1), idx.shape)
        dev = pool.remap_ondevice(idx)
        assert np.array_equal(np.array(dev), np.array(host))
        assert np.array(dev).dtype == np.uint32


def test_slot_table_coherent_through_eviction(tmp_path):
    """The on-device table tracks evictions: an evicted expert maps to the sentinel
    and the survivors remap correctly vs the host path."""
    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg, index=index, layer=0, projection="gate_proj", capacity=2,
    )
    pool.ensure([0])
    pool.ensure([1])     # pool full: {0,1}
    pool.ensure([0])     # touch 0 so 1 is the LRU/LFU eviction victim
    pool.ensure([2])     # evicts 1 -> {0,2}

    assert pool.resident_ids() == {0, 2}
    idx = mx.array([[0, 2]], dtype=mx.uint32)
    host = pool.remap_loaded(np.asarray(idx).reshape(-1), idx.shape)
    assert np.array_equal(np.array(pool.remap_ondevice(idx)), np.array(host))
    # expert 1 is absent -> sentinel (== num_experts) in the on-device table
    table = pool._ensure_slot_table()
    assert int(np.array(table)[1]) == pool.num_experts


def test_pooled_switchglu_ondevice_matches_resident_decode(tmp_path, monkeypatch):
    """With the on-device remap active (default), pooled decode output is identical
    to the resident OwnedSwitchGLU reference, and the on-device path is exercised."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_ONDEVICE_REMAP", True)
    resident = _resident_switch(n_experts=16)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_switch(pkg, index, resident, capacity=4)
    x = mx.random.normal((1, 64)).astype(mx.float16)
    indices = mx.array([[3, 7, 1, 5]], dtype=mx.uint32)

    _assert_same(resident(x, indices), pooled(x, indices))
    assert pooled.remap_ondevice_calls > 0


def test_ondevice_remap_kill_switch_falls_back(tmp_path, monkeypatch):
    """With the kill switch off the host remap_loaded path runs and the output is
    still identical to resident (proves the fallback is exact too)."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_ONDEVICE_REMAP", False)
    resident = _resident_switch(n_experts=16)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_switch(pkg, index, resident, capacity=4)
    x = mx.random.normal((1, 64)).astype(mx.float16)
    indices = mx.array([[3, 7, 1, 5]], dtype=mx.uint32)

    _assert_same(resident(x, indices), pooled(x, indices))
    assert pooled.remap_ondevice_calls == 0


def test_compiled_island_matches_eager_fused_decode(tmp_path, monkeypatch):
    """The compiled-island decode path (mx.compile of remap x2 + rotate + fused
    + rotate + gather) must match the eager fused path bit-exactly: same kernels,
    same order, the only difference is compilation. All-hit and mixed-miss decode
    shapes."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", True)
    monkeypatch.setattr(psg, "_ONDEVICE_REMAP", True)
    resident = _resident_switch(n_experts=16, gate_bits=4, up_bits=4, down_bits=2)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)

    x = mx.random.normal((1, 64)).astype(mx.float16)
    cold = mx.array([[3, 7, 1, 5]], dtype=mx.uint32)     # all-miss install
    warm = mx.array([[5, 1, 7, 3]], dtype=mx.uint32)     # all-hit, permuted
    mixed = mx.array([[3, 9, 1, 11]], dtype=mx.uint32)   # 2 hits + 2 misses

    outs = {}
    for flag in (False, True):
        monkeypatch.setattr(psg, "_COMPILED_ISLAND", flag)
        pooled = _pooled_switch(pkg, index, resident, capacity=4)
        pooled.eval()  # production runtime calls model.eval(); island is inference-only
        outs[flag] = [pooled(x, idx) for idx in (cold, warm, mixed)]
        mx.eval(*outs[flag])
        assert pooled.compiled_island_calls == (3 if flag else 0)

    for eager, island in zip(outs[False], outs[True]):
        assert np.array_equal(np.array(eager), np.array(island))


def test_compiled_island_skipped_for_sorted_and_separate_paths(tmp_path, monkeypatch):
    """The island only covers the unsorted single-row decode shape with fused
    preconditions; sorted routing and non-matching codebooks stay eager."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", True)
    monkeypatch.setattr(psg, "_COMPILED_ISLAND", True)
    # different gate/up bits -> fused (and so the island) disabled, exact path
    resident = _resident_switch(n_experts=16, gate_bits=2, up_bits=4)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_switch(pkg, index, resident, capacity=4)
    x = mx.random.normal((1, 64)).astype(mx.float16)
    _assert_same(resident(x, mx.array([[3, 7, 1, 5]], dtype=mx.uint32)),
                 pooled(x, mx.array([[3, 7, 1, 5]], dtype=mx.uint32)))
    assert pooled.compiled_island_calls == 0

    # sorted/prefill shape (indices.size >= 64) -> island skipped
    resident4 = _resident_switch(n_experts=16, gate_bits=4, up_bits=4)
    p4 = tmp_path / "p4"
    p4.mkdir()
    pkg4 = _package_from_resident(p4, resident4)
    index4 = build_expert_index(pkg4)
    pooled4 = _pooled_switch(pkg4, index4, resident4, capacity=16)
    pooled4.eval()
    xs = mx.random.normal((20, 64)).astype(mx.float16)
    idxs = mx.random.randint(0, 16, (20, 4)).astype(mx.uint32)
    mx.eval(pooled4(xs, idxs))
    assert pooled4.compiled_island_calls == 0


def test_fused_gate_up_matches_resident_when_codebooks_match(tmp_path, monkeypatch):
    """When gate/up share a codebook (the real mjtq condition: same in_features/bits/
    seed), the fused gate+up kernel path is active and matches the resident reference
    within fp16 tolerance (fusion reorders fp, so not bit-exact)."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", True)
    # gate_bits == up_bits -> identical codebooks/signs -> fused path enabled
    resident = _resident_switch(n_experts=16, gate_bits=4, up_bits=4, down_bits=2)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_switch(pkg, index, resident, capacity=4)
    assert pooled._fused_gate_up, "fused path should be active when codebooks match"

    x = mx.random.normal((1, 64)).astype(mx.float16)
    indices = mx.array([[3, 7, 1, 5]], dtype=mx.uint32)
    ref = resident(x, indices)
    got = pooled(x, indices)
    mx.eval(ref, got)
    assert pooled.fused_gate_up_calls > 0
    a, b = np.array(ref), np.array(got)
    # fp16 tolerance: fusion changes rounding order vs separate gate/up matmuls
    assert np.allclose(a, b, rtol=2e-2, atol=2e-3), \
        f"fused vs resident max diff {np.abs(a - b).max()} too large"


def test_fused_gate_up_falls_back_when_codebooks_differ(tmp_path, monkeypatch):
    """When gate/up have different codebooks (different bits), the fused path is
    disabled (its single-codebook assumption would be wrong) and output is bit-exact
    via the separate path."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", True)
    resident = _resident_switch(n_experts=16, gate_bits=2, up_bits=4)  # differ
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_switch(pkg, index, resident, capacity=4)
    assert not pooled._fused_gate_up, "fused must be disabled when codebooks differ"

    x = mx.random.normal((1, 64)).astype(mx.float16)
    indices = mx.array([[3, 7, 1, 5]], dtype=mx.uint32)
    _assert_same(resident(x, indices), pooled(x, indices))  # exact via separate path


def test_gate_up_pools_assign_identical_slots(tmp_path):
    """Invariant the fused kernel relies on: gate and up pools assign the same slot to
    the same expert (they are loaded together, so slot N holds the same expert)."""
    resident = _resident_switch(n_experts=16, gate_bits=4, up_bits=4)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pooled = _pooled_switch(pkg, index, resident, capacity=4)
    # drive several different active sets through eviction
    for sel in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 5, 2, 7], [8, 9, 10, 11]):
        pooled(mx.random.normal((1, 64)).astype(mx.float16),
               mx.array([sel], dtype=mx.uint32))
    g = pooled.gate_proj.pool._slot_of
    u = pooled.up_proj.pool._slot_of
    assert set(g) == set(u)
    assert all(g[e] == u[e] for e in g), "gate/up slot maps diverged"


def test_pooled_sparse_moe_overlaps_decode_load_with_shared_eval(
    tmp_path,
    monkeypatch,
):
    """Overlap seam: routed misses start, shared expert is forced while reads are
    in flight, then the pooled switch consumes the ticket and matches resident."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", False)
    monkeypatch.setattr(psg, "_RING_DECODE", False)  # legacy ticket path test

    resident_switch = _resident_switch(n_experts=16, gate_bits=2, up_bits=4)
    pkg = _package_from_resident(tmp_path, resident_switch)
    index = build_expert_index(pkg)
    pooled_switch = _pooled_switch(pkg, index, resident_switch, capacity=4)

    gate = nn.Linear(64, 16, bias=False)
    shared = _TinySharedMLP(64, 32)
    shared_gate = nn.Linear(64, 1, bias=False)
    reference = _TinySparseMoeBlock(
        gate=gate,
        switch_mlp=resident_switch,
        shared_expert=shared,
        shared_expert_gate=shared_gate,
    )
    pooled = PooledSparseMoeBlock(_TinySparseMoeBlock(
        gate=gate,
        switch_mlp=pooled_switch,
        shared_expert=shared,
        shared_expert_gate=shared_gate,
    ))

    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    ref = reference(x)
    got = pooled(x)
    mx.eval(ref, got)

    assert np.array_equal(np.array(ref), np.array(got))
    assert pooled_switch.overlap_load_started_calls == 1
    assert pooled_switch.overlap_load_wait_calls == 1
    assert pooled_switch.overlap_shared_eval_calls == 1
    assert pooled_switch.overlap_prefill_no_eval_calls == 0
    assert pooled_switch.projection_load_wait_calls == 1
    # phase counters accrue through the block wrapper on the decode path
    assert pooled_switch.index_sync_calls == 1
    assert pooled_switch.index_sync_seconds > 0.0
    assert pooled_switch.decode_moe_block_calls == 1
    assert pooled_switch.decode_moe_block_seconds > 0.0
    assert pooled_switch.routed_build_seconds > 0.0
    assert pooled_switch.routed_weighted_sum_calls == 1
    assert pooled_switch.routed_weighted_sum_slot_elements == 4
    assert pooled_switch.routed_weighted_sum_output_elements == 64
    # block-exit kick: the kick fires on decode (output already asserted
    # identical to the resident reference above, so the kick is value-neutral)
    assert pooled_switch.block_exit_kick_calls == 1


def test_ring_decode_matches_normal_two_layer_chain(tmp_path, monkeypatch):
    """Ring decode: GPU exports router ids+seq to a ring the worker polls
    with zero MLX calls; commits stay on main. Two chained MoE blocks across
    cold/hit/eviction decode steps must match the normal path bit-exactly."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", True)
    monkeypatch.setattr(psg, "_ONDEVICE_REMAP", True)

    res0 = _resident_switch(n_experts=16, gate_bits=4, up_bits=4)
    res1 = _resident_switch(n_experts=16, gate_bits=4, up_bits=4)
    p0 = tmp_path / "r0"
    p1 = tmp_path / "r1"
    p0.mkdir()
    p1.mkdir()
    pkg0 = _package_from_resident(p0, res0)
    pkg1 = _package_from_resident(p1, res1)

    gates = [nn.Linear(64, 16, bias=False) for _ in range(2)]
    shareds = [_TinySharedMLP(64, 32) for _ in range(2)]
    sgates = [nn.Linear(64, 1, bias=False) for _ in range(2)]

    xs = [mx.random.normal((1, 1, 64)).astype(mx.float16) for _ in range(4)]
    outs = {}
    for flag in (False, True):
        monkeypatch.setattr(psg, "_RING_DECODE", flag)
        blocks = []
        for i, (pkg, res) in enumerate(((pkg0, res0), (pkg1, res1))):
            index = build_expert_index(pkg)
            block = PooledSparseMoeBlock(_TinySparseMoeBlock(
                gate=gates[i],
                switch_mlp=_pooled_switch(pkg, index, res, capacity=4),
                shared_expert=shareds[i],
                shared_expert_gate=sgates[i],
            ))
            block.eval()
            blocks.append(block)
        blocks[-1].pipeline_is_last = True
        got = []
        for x in xs:
            y = blocks[1](blocks[0](x))
            mx.eval(y)
            got.append(np.array(y))
        outs[flag] = got
        if flag:
            assert blocks[0].switch_mlp.pipelined_layers == len(xs)
            assert blocks[0].switch_mlp.pipeline_read_seconds > 0.0

    for normal, ringed in zip(outs[False], outs[True]):
        assert np.array_equal(normal, ringed)


def _gate_available():
    """Skip-or-fail gate for native tests: in normal CI
    a missing native build skips; with MOESPRESSO_REQUIRE_NATIVE_GATE=1 (the
    native/run_native_tests.sh target) a missing/broken build fails, so green
    unambiguously means native-gate green."""
    import os

    from moespresso.runtime.native_gate import load_gate
    available = load_gate() is not None
    if not available and os.environ.get("MOESPRESSO_REQUIRE_NATIVE_GATE") == "1":
        pytest.fail(
            "MOESPRESSO_REQUIRE_NATIVE_GATE=1 but the native gate is missing "
            "or failed its self-test (build with native/build.sh)")
    return available


def test_gate_decode_matches_normal_two_layer_chain(tmp_path, monkeypatch):
    """Native-gate decode: islands behind in-stream event waits, worker
    signals after publish, main commits immediately. Two chained MoE blocks
    across cold/hit/eviction steps must match the normal path bit-exactly."""
    if not _gate_available():
        pytest.skip("native gate not built in this environment")
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", True)
    monkeypatch.setattr(psg, "_ONDEVICE_REMAP", True)

    res0 = _resident_switch(n_experts=16, gate_bits=4, up_bits=4)
    res1 = _resident_switch(n_experts=16, gate_bits=4, up_bits=4)
    g0 = tmp_path / "g0"
    g1 = tmp_path / "g1"
    g0.mkdir()
    g1.mkdir()
    pkg0 = _package_from_resident(g0, res0)
    pkg1 = _package_from_resident(g1, res1)

    gates = [nn.Linear(64, 16, bias=False) for _ in range(2)]
    shareds = [_TinySharedMLP(64, 32) for _ in range(2)]
    sgates = [nn.Linear(64, 1, bias=False) for _ in range(2)]

    xs = [mx.random.normal((1, 1, 64)).astype(mx.float16) for _ in range(4)]
    outs = {}
    for use_gate in (False, True):
        # arm False = plain normal path (ring and gate off)
        monkeypatch.setattr(psg, "_RING_DECODE", use_gate)
        blocks = []
        for i, (pkg, res) in enumerate(((pkg0, res0), (pkg1, res1))):
            index = build_expert_index(pkg)
            block = PooledSparseMoeBlock(_TinySparseMoeBlock(
                gate=gates[i],
                switch_mlp=_pooled_switch(pkg, index, res, capacity=4),
                shared_expert=shareds[i],
                shared_expert_gate=sgates[i],
            ))
            block.eval()
            blocks.append(block)
        blocks[-1].pipeline_is_last = True
        got = []
        for x in xs:
            y = blocks[1](blocks[0](x))
            mx.eval(y)
            got.append(np.array(y))
        outs[use_gate] = got
        if use_gate:
            assert blocks[0].switch_mlp.pipelined_layers == len(xs)

    for normal, gated in zip(outs[False], outs[True]):
        assert np.array_equal(normal, gated)


def test_gate_decode_worker_error_signals_and_raises(tmp_path, monkeypatch):
    """Native-gate poison path: a worker failure must still signal the gate (no
    GPU hang) and surface as an exception at the once-per-token drain."""
    if not _gate_available():
        pytest.skip("native gate not built in this environment")
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", True)
    monkeypatch.setattr(psg, "_RING_DECODE", True)

    resident = _resident_switch(n_experts=16, gate_bits=4, up_bits=4)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    sw = _pooled_switch(pkg, index, resident, capacity=4)
    block = PooledSparseMoeBlock(_TinySparseMoeBlock(
        gate=nn.Linear(64, 16, bias=False),
        switch_mlp=sw,
        shared_expert=_TinySharedMLP(64, 32),
        shared_expert_gate=nn.Linear(64, 1, bias=False),
    ))
    block.pipeline_is_last = True
    block.eval()

    def boom(active, load_ticket=None):
        raise OSError("injected worker failure")

    monkeypatch.setattr(sw, "_ensure_projection_pools", boom)
    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    with pytest.raises(OSError, match="injected"):
        block(x)  # the last-layer drain re-raises the worker error
    # the gate was still signaled (poison): evaluating does not hang
    from moespresso.runtime.native_gate import load_gate
    assert load_gate().signaled_value() >= 1


def test_ring_self_test_failure_falls_back_to_legacy(tmp_path, monkeypatch):
    """If the once-per-process ring visibility self-test fails, decode
    routes through the proven legacy path instead of per-layer timeouts."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_RING_DECODE", True)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [False])  # simulated failure

    resident = _resident_switch(n_experts=16, gate_bits=4, up_bits=4)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    sw = _pooled_switch(pkg, index, resident, capacity=4)
    block = PooledSparseMoeBlock(_TinySparseMoeBlock(
        gate=nn.Linear(64, 16, bias=False),
        switch_mlp=sw,
        shared_expert=_TinySharedMLP(64, 32),
        shared_expert_gate=nn.Linear(64, 1, bias=False),
    ))
    block.eval()
    mx.eval(block(mx.random.normal((1, 1, 64)).astype(mx.float16)))
    assert sw.pipelined_layers == 0          # ring path not taken
    assert sw.index_resync_seconds > 0.0     # legacy path ran


def test_ring_self_test_passes_for_current_mlx_build():
    """The real visibility self-test must pass for the default-on ring path to be
    used; if it fails, the current MLX build cannot use the ring."""
    import moespresso.runtime.pooled_switchglu as psg
    psg._RING_SELF_TEST[0] = None  # force re-run
    assert psg._ring_visibility_ok() is True


def test_pooled_sparse_moe_does_not_force_prefill_shared_eval(tmp_path, monkeypatch):
    """Prefill can have a large shared-expert intermediate, so the overlap wrapper
    starts loads but does not force shared_y materialization for multi-token input."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", False)

    resident_switch = _resident_switch(n_experts=16, gate_bits=2, up_bits=4)
    pkg = _package_from_resident(tmp_path, resident_switch)
    index = build_expert_index(pkg)
    pooled_switch = _pooled_switch(pkg, index, resident_switch, capacity=16)
    pooled = PooledSparseMoeBlock(_TinySparseMoeBlock(
        gate=nn.Linear(64, 16, bias=False),
        switch_mlp=pooled_switch,
        shared_expert=_TinySharedMLP(64, 32),
        shared_expert_gate=nn.Linear(64, 1, bias=False),
    ))

    mx.eval(pooled(mx.random.normal((1, 3, 64)).astype(mx.float16)))

    assert pooled_switch.overlap_load_started_calls == 1
    assert pooled_switch.overlap_shared_eval_calls == 0
    assert pooled_switch.overlap_prefill_no_eval_calls == 1


def test_bundle_row_cache_one_pread_serves_three_pools(tmp_path):
    """The IO contract: a missed expert = one row pread, shared by the
    layer's three pools (even when they ensure concurrently), with the loaded
    bytes identical to the per-component fallback path."""
    import concurrent.futures

    from moespresso.runtime.expert_slot_pool import BundleRowCache

    resident = _resident_switch(n_experts=16, gate_bits=2, up_bits=4)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)

    def pools(row_cache):
        return {
            proj: ExpertSlotPool(package_dir=pkg, index=index, layer=0,
                                 projection=proj, capacity=8,
                                 row_cache=row_cache)
            for proj in ("gate_proj", "up_proj", "down_proj")
        }

    cache = BundleRowCache(package_dir=pkg, index=index, layer=0)
    cached = pools(cache)
    plain = pools(None)
    active = [1, 5, 9]

    # racing ensures across the three pools, like the projection-load executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(p.ensure, active) for p in cached.values()]
        for f in futures:
            f.result()
    for p in plain.values():
        p.ensure(active)

    # one pread per missed expert row, the other two pools were cache takes
    assert cache.total_preads == len(active)
    assert cache.total_cached_takes == 2 * len(active)
    assert not cache._rows and not cache._inflight  # steady state: empty

    # byte-identical slots vs the per-component fallback
    for proj in cached:
        a, b = cached[proj], plain[proj]
        for e in active:
            import numpy as np
            sa, sb = a.slot_of(e), b.slot_of(e)
            assert np.array_equal(np.array(a.packed[sa]), np.array(b.packed[sb]))
            assert np.array_equal(np.array(a.norms[sa]), np.array(b.norms[sb]))


def test_bundle_row_cache_failed_pread_stays_fail_closed(tmp_path):
    """A failing row read publishes nothing: no residency, no stuck in-flight
    marker, and a retry surfaces the same error (fail-closed, like the
    per-component path)."""
    import pytest as _pytest

    from moespresso.runtime import expert_slot_pool as esp

    resident = _resident_switch(n_experts=16)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    cache = esp.BundleRowCache(package_dir=pkg, index=index, layer=0)
    pool = ExpertSlotPool(package_dir=pkg, index=index, layer=0,
                          projection="gate_proj", capacity=4, row_cache=cache)

    real_pread = esp.pread_view_cached
    calls = {"n": 0}

    def failing_pread(*args, **kwargs):
        calls["n"] += 1
        raise OSError("injected pread failure")

    esp.pread_view_cached = failing_pread
    try:
        with _pytest.raises(OSError, match="injected"):
            pool.ensure([3])
    finally:
        esp.pread_view_cached = real_pread

    assert calls["n"] == 1
    assert 3 not in pool.resident_ids()
    assert not cache._rows and not cache._inflight
    pool.ensure([3])  # recovery works once IO is healthy again
    assert 3 in pool.resident_ids()


def test_route_trace_captures_prefill_and_decode(tmp_path):
    """Route-trace oracle study: opt-in route tracing records position-intact
    prefill routes and per-step decode routes; OFF by default with zero residue."""
    import moespresso.runtime.pooled_switchglu as psg

    resident = _resident_switch(n_experts=16)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    switch = _pooled_switch(pkg, index, resident, capacity=16)

    assert psg._ROUTE_TRACE is None  # off by default
    psg.route_trace_start()
    x_pre = mx.random.normal((1, 3, 64)).astype(mx.float16)
    inds_pre = mx.array([[[1, 2], [3, 4], [5, 1]]], dtype=mx.uint32)
    mx.eval(switch(x_pre, inds_pre))
    x_dec = mx.random.normal((1, 1, 64)).astype(mx.float16)
    mx.eval(switch(x_dec, mx.array([[[7, 2]]], dtype=mx.uint32)))
    trace = psg.route_trace_stop()
    assert psg._ROUTE_TRACE is None  # stopped clean

    tags = [t[0] for t in trace]
    assert "prefill" in tags and "decode_direct" in tags
    pre = next(t for t in trace if t[0] == "prefill")
    assert pre[1] == 0  # layer
    assert pre[2] == [[1, 2], [3, 4], [5, 1]]  # position-intact
    dec = next(t for t in trace if t[0] == "decode_direct")
    assert dec[2] == [[7, 2]]


def test_lookahead_prefetch_keeps_outputs_identical(tmp_path, monkeypatch):
    """Cross-layer lookahead prefetch is a residency-only effect:
    a two-layer gate-decode chain with lookahead Delta=1 wired must produce
    bit-identical outputs to the same chain without it, and the prediction
    machinery must actually run (exports + prefetch loads observed)."""
    if not _gate_available():
        pytest.skip("native gate not built in this environment")
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_FUSED_GATE_UP", True)
    monkeypatch.setattr(psg, "_ONDEVICE_REMAP", True)
    monkeypatch.setattr(psg, "_RING_DECODE", True)

    res0 = _resident_switch(n_experts=32, gate_bits=4, up_bits=4)
    res1 = _resident_switch(n_experts=32, gate_bits=4, up_bits=4)
    g0, g1 = tmp_path / "g0", tmp_path / "g1"
    g0.mkdir(), g1.mkdir()
    pkg0 = _package_from_resident(g0, res0)
    pkg1 = _package_from_resident(g1, res1)

    gates = [nn.Linear(64, 16, bias=False) for _ in range(2)]
    shareds = [_TinySharedMLP(64, 32) for _ in range(2)]
    sgates = [nn.Linear(64, 1, bias=False) for _ in range(2)]
    xs = [mx.random.normal((1, 1, 64)).astype(mx.float16) for _ in range(6)]

    outs = {}
    counters = {}
    for use_lookahead in (False, True):
        blocks = []
        for i, (pkg, res) in enumerate(((pkg0, res0), (pkg1, res1))):
            index = build_expert_index(pkg)
            block = PooledSparseMoeBlock(_TinySparseMoeBlock(
                gate=gates[i],
                switch_mlp=_pooled_switch(pkg, index, res, capacity=12, spare_slots=4),
                shared_expert=shareds[i],
                shared_expert_gate=sgates[i],
            ))
            block.eval()
            blocks.append(block)
        blocks[-1].pipeline_is_last = True
        if use_lookahead:
            sw0, sw1 = blocks[0].switch_mlp, blocks[1].switch_mlp
            sw0.lookahead_w = gates[1].weight.astype(mx.float16)
            mx.eval(sw0.lookahead_w)
            sw0.lookahead_target = sw1
        got = []
        for x in xs:
            y = blocks[1](blocks[0](x))
            mx.eval(y)
            got.append(np.array(y))
        # drain the lookahead executor before reading counters
        if use_lookahead:
            psg._lookahead_executor().submit(lambda: None).result()
            counters["exports"] = blocks[0].switch_mlp.lookahead_exports
            counters["loads"] = blocks[1].switch_mlp.gate_proj.pool \
                .total_prefetch_loads + blocks[0].switch_mlp \
                .lookahead_prefetch_loads
            counters["errors"] = blocks[0].switch_mlp.lookahead_errors
            counters["ring_misses"] = blocks[0].switch_mlp.lookahead_ring_misses
        outs[use_lookahead] = got

    for a, b in zip(outs[False], outs[True], strict=True):
        assert np.array_equal(a, b)
    assert counters["exports"] == len(xs)
    assert counters["errors"] == 0
    # at least one speculative load must have happened across the steps
    assert counters["loads"] > 0, counters


def test_place_spare_trio_refuses_in_flight_demand_placement(tmp_path):
    """A demand ensure's phase-1 placement reserves occupancy only; the trio
    must refuse the expert on that occupancy alone, or the expert splits
    across a demand and a spare slot and the loser publish strands a
    ghost (an occupied-but-unpublished slot that permanently shrinks the
    pool and surfaces as spurious ExpertCapacityExceeded)."""
    import threading

    from moespresso.runtime import expert_slot_pool as esp
    from moespresso.runtime.expert_slot_pool import place_spare_trio

    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pools = tuple(
        ExpertSlotPool(package_dir=pkg, index=index, layer=0,
                       projection=proj, capacity=4, spare_slots=2)
        for proj in ("gate_proj", "up_proj", "down_proj"))

    gate = pools[0]
    hold = threading.Event()
    entered = threading.Event()
    real_load = esp.ExpertSlotPool._load_expert

    def slow_load(self, *, expert, slot):
        entered.set()
        hold.wait(5.0)
        return real_load(self, expert=expert, slot=slot)

    esp.ExpertSlotPool._load_expert = slow_load
    try:
        worker = threading.Thread(target=lambda: gate.ensure([3]))
        worker.start()
        entered.wait(5.0)
        # Expert 3's demand placement is in flight (occupied, unpublished):
        # the trio must refuse it.
        assert place_spare_trio(pools, 3, 0) is False
        hold.set()
        worker.join(5.0)
    finally:
        esp.ExpertSlotPool._load_expert = real_load
        hold.set()

    assert gate.slot_of(3) < gate.capacity  # demand-resident, single slot
    # No ghost occupancies anywhere: every occupied slot is published.
    for pool in pools:
        for sl, occ in enumerate(pool._expert_at):
            if occ is not None:
                assert pool._slot_of.get(occ) == sl
    # With the load complete, the same speculative placement now lands.
    assert place_spare_trio(pools, 5, 0) is True
    assert gate.slot_of(5) == gate.capacity


def test_place_spare_trio_eviction_pops_only_its_own_mapping(tmp_path):
    """The spare-occupant eviction severs only a mapping that points at the
    spare slot being reclaimed; a stale occupancy whose published
    residency lives elsewhere is left resident."""
    from moespresso.runtime.expert_slot_pool import place_spare_trio

    resident = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pools = tuple(
        ExpertSlotPool(package_dir=pkg, index=index, layer=0,
                       projection=proj, capacity=4, spare_slots=2)
        for proj in ("gate_proj", "up_proj", "down_proj"))
    gate = pools[0]
    # Simulate stale spare occupancy metadata for a demand-resident expert;
    # a later ensure moves _demand_protect off it so the occupant check is
    # not what protects it.
    for pool in pools:
        pool.ensure([2])
        pool.ensure([0])
        pool._expert_at[pool.capacity + 0] = 2
    demand_slot = gate.slot_of(2)

    assert place_spare_trio(pools, 6, 0) is True
    # Expert 2's demand residency survived the spare reclamation.
    assert gate.slot_of(2) == demand_slot
    assert gate.slot_of(6) == gate.capacity


def test_ensure_feasibility_waits_out_reservations_not_spare_residents(
        tmp_path):
    """The feasibility gate counts evictable demand slots only: a
    spare-resident expert must not let the batch proceed into a deficit
    while a reservation is in flight; the batch waits for the publish and
    then evicts the published, unprotected expert."""
    import threading
    import time as _time

    resident = _resident_switch(n_experts=16)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg, index=index, layer=0, projection="gate_proj",
        capacity=2, spare_slots=1)

    pool.ensure([0])
    # Spare-resident unprotected expert 9 (would be miscounted as
    # evictable by a demand-only-blind tally).
    pool._expert_at[2] = 9
    pool._slot_of[9] = 2
    # Demand slot 1 is reserved by an in-flight prefetch of expert 5.
    pool._expert_at[1] = 5
    pool._prefetch_reserved.add(5)
    pool._prefetch_inflight = 1

    done = []

    def _demand():
        pool.ensure([3], protect={0})
        done.append(True)

    worker = threading.Thread(target=_demand)
    worker.start()
    _time.sleep(0.05)
    # The ensure remains in its retry loop; nothing has been published.
    assert not done
    assert 3 not in pool._slot_of
    # Publish the in-flight prefetch; the ensure resumes and evicts it.
    with pool._bk_lock:
        pool._slot_of[5] = 1
        pool._prefetch_reserved.discard(5)
        pool._prefetch_inflight = 0
    worker.join(5.0)
    assert done
    assert 3 in pool._slot_of and 0 in pool._slot_of
    assert 5 not in pool._slot_of  # the published prefetch was the victim


def test_ensure_mid_batch_capacity_raise_releases_reservations(tmp_path):
    """A capacity raise partway through a placement batch must release the
    batch's occupancy reservations: leaked occupied-but-unpublished slots
    are neither free nor evictable and permanently shrink the pool."""
    import pytest as _pytest

    resident = _resident_switch(n_experts=16)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pool = ExpertSlotPool(
        package_dir=pkg, index=index, layer=0, projection="gate_proj",
        capacity=2)
    pool.ensure([0, 1])
    # Both residents protected, two misses: the first placement cannot
    # find a slot, and whatever the batch reserved must be released.
    with _pytest.raises(ExpertCapacityExceeded):
        pool.ensure([2, 3], protect={0, 1})
    assert pool.resident_ids() == {0, 1}
    occupied = [sl for sl in range(pool.capacity)
                if pool._expert_at[sl] is not None]
    published = [sl for _e, sl in pool._slot_of.items() if sl < pool.capacity]
    assert sorted(occupied) == sorted(published)
    # The pool still serves once the caller shrinks its demand.
    pool.ensure([2], protect={0})
    assert 2 in pool.resident_ids()


def test_grow_preserves_spare_slots_and_remaps(tmp_path):
    """grow() must carry the spare region (bytes + slot map)
    to its new offset. It used to drop spares while _slot_of still pointed
    at them (OOB on the next trio placement under lookahead+growth)."""
    from moespresso.runtime.expert_slot_pool import place_spare_trio

    resident = _resident_switch(n_experts=32)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    pools = tuple(
        ExpertSlotPool(package_dir=pkg, index=index, layer=0,
                       projection=proj, capacity=4, spare_slots=2)
        for proj in ("gate_proj", "up_proj", "down_proj"))

    pools[0].ensure([1, 2])
    assert place_spare_trio(pools, 9, 0)
    gate = pools[0]
    spare_bytes_before = bytes(
        gate._packed_view[4 * gate._comp_packed["nbytes"] // 1:]
    )[:64]
    assert gate.slot_of(9) == 4  # first spare slot

    gate.grow(8)
    assert gate.capacity == 8 and gate.spare_slots == 2
    assert gate.packed.shape[0] == 10  # 8 demand + 2 spares
    assert gate.slot_of(9) == 8  # spare remapped to the new offset
    assert gate.slot_of(1) == gate._slot_of[1] < 8  # demand untouched
    # spare bytes moved with the map
    pn = gate._comp_packed["nbytes"]
    assert bytes(gate._packed_view[8 * pn:8 * pn + 64]) == spare_bytes_before
    # and the spare ring still works at the new offset
    assert gate._expert_at[8] == 9


# ---- in-session hotness decay + evict-DONTNEED page-cache hygiene ----

def _gate_pool(tmp_path, *, capacity, n_experts=8, projection="gate_proj"):
    resident = _resident_switch(n_experts=n_experts)
    pkg = _package_from_resident(tmp_path, resident)
    index = build_expert_index(pkg)
    return ExpertSlotPool(
        package_dir=pkg,
        index=index,
        layer=0,
        projection=projection,
        capacity=capacity,
        eviction_policy="lfu",
    )


def test_hotness_decay_bounds_counts_and_follows_topic_shift(tmp_path, monkeypatch):
    from moespresso.runtime import expert_slot_pool as esp

    monkeypatch.setattr(esp, "_DECAY_EVERY_TOUCHES", 4)
    pool = _gate_pool(tmp_path, capacity=2)
    for _ in range(20):
        pool.ensure([0])  # early-topic expert: 20 raw touches
    assert pool._freq[0] <= 8  # decay bounds the count (raw would be 20)
    # topic shift: experts 1/2 take over; decayed counts let them win fast
    for _ in range(8):
        pool.ensure([1])
        pool.ensure([2])
    assert 0 not in pool.resident_ids()
    assert pool.resident_ids() == {1, 2}


def test_hotness_decay_disabled_lets_early_experts_squat(tmp_path, monkeypatch):
    from moespresso.runtime import expert_slot_pool as esp

    monkeypatch.setattr(esp, "_DECAY_EVERY_TOUCHES", 0)
    pool = _gate_pool(tmp_path, capacity=2)
    for _ in range(20):
        pool.ensure([0])
    assert pool._freq[0] == 20  # raw counts, no decay
    for _ in range(8):
        pool.ensure([1])
        pool.ensure([2])
    # the documented pre-decay behavior: the inflated early count squats
    assert 0 in pool.resident_ids()


def test_evict_dontneed_advises_evicted_gate_rows(tmp_path, monkeypatch):
    from moespresso.runtime import expert_slot_pool as esp

    monkeypatch.setattr(esp, "_EVICT_DONTNEED", True)
    pool = _gate_pool(tmp_path, capacity=2)
    pool.ensure([0])
    pool.ensure([1])
    assert pool.total_dontneed == 0  # no eviction yet
    pool.ensure([2])  # evicts one resident -> one advise, drained in ensure
    assert pool.total_dontneed == 1
    assert pool.total_dontneed_errors == 0
    assert 2 in pool.resident_ids()  # load path unaffected


def test_evict_dontneed_is_gate_pool_only(tmp_path, monkeypatch):
    from moespresso.runtime import expert_slot_pool as esp

    monkeypatch.setattr(esp, "_EVICT_DONTNEED", True)
    pool = _gate_pool(tmp_path, capacity=2, projection="up_proj")
    pool.ensure([0])
    pool.ensure([1])
    pool.ensure([2])  # eviction in a non-gate pool: no advise recorded
    assert pool.total_dontneed == 0
    assert pool._pending_advise == []


def test_evict_dontneed_failure_is_counted_not_raised(tmp_path, monkeypatch):
    from moespresso.runtime import expert_slot_pool as esp

    monkeypatch.setattr(esp, "_EVICT_DONTNEED", True)
    pool = _gate_pool(tmp_path, capacity=2)
    pool.ensure([0])
    pool.ensure([1])

    def _boom(**kwargs):
        raise RuntimeError("locate_row failed")

    monkeypatch.setattr(pool.index, "locate_row", _boom)
    pool.ensure([2])  # advise fails silently; demand flow unaffected
    assert pool.total_dontneed == 0
    assert pool.total_dontneed_errors == 1
    assert 2 in pool.resident_ids()
