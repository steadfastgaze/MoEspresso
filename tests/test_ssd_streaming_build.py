"""SSD-streaming builder wiring.

These tests stay synthetic: they prove the builder installs the direct-buffer
pooled SwitchGLU seam and exposes stats, without loading a real model.
"""

from __future__ import annotations


import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant.gather_tq_kernel")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from moespresso.package.bundle import assemble_layer_bundle, encode_bundle_metadata  # noqa: E402
from moespresso.package.kquant_format import KQUANT_GEOMETRY  # noqa: E402
from moespresso.runtime.expert_index import build_expert_index  # noqa: E402
from moespresso.runtime.pooled_switchglu import (  # noqa: E402
    PooledDeepseekV4MoEBlock,
    PooledCombinedGateUpKQuantLinear,
    PooledSparseMoeBlock,
    PooledSwitchGLU,
)
from moespresso.runtime.ssd_streaming_build import (  # noqa: E402
    SSDStreamingBuildError,
    _budget_payload,
    _is_routed_expert_key,
    grow_ssd_streaming_capacity,
    install_pooled_switchglus,
    maybe_adapt_ssd_streaming_capacity,
    ssd_streaming_layer_stats,
    ssd_streaming_stats,
    suggest_capacity_overrides_from_layer_stats,
)
from moespresso.runtime.streaming_capacity import CapacityBudget  # noqa: E402


def _package(tmp_path, *, n_experts=8, hidden=64, intermediate=32, layers=(0,)):
    from conftest import write_bundle_package

    pkg = tmp_path / "pkg"
    pkg.mkdir()

    def packed_cols(in_features, bits):
        return (in_features + (32 // bits) - 1) // (32 // bits)

    write_bundle_package(pkg, layers=layers, n_exp=n_experts, specs={
        "gate_proj": (intermediate, packed_cols(hidden, 2), 2),
        "up_proj": (intermediate, packed_cols(hidden, 4), 4),
        "down_proj": (hidden, packed_cols(intermediate, 2), 2),
    })
    return pkg


def _kquant_package(tmp_path, *, n_experts=4, hidden=256, intermediate=256):
    from conftest import write_safetensors_raw

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    rng = np.random.default_rng(3)
    specs = {
        "gate_proj": ("iq2_xxs", intermediate),
        "up_proj": ("iq2_xxs", intermediate),
        "down_proj": ("q2_k", hidden),
    }
    components = {}
    bits = {}
    codecs = {}
    kquant_codecs = {}
    expected = {}
    for projection, (codec, out_features) in specs.items():
        geometry = KQUANT_GEOMETRY[codec]
        weight = rng.integers(
            0,
            255,
            (n_experts, out_features, geometry.bytes_per_block),
            dtype=np.uint8,
        )
        scales = np.zeros((n_experts, 1), dtype=np.uint8)
        components[(projection, "weight")] = weight
        components[(projection, "scales")] = scales
        bits[projection] = geometry.bits
        codecs[projection] = "kquant"
        kquant_codecs[projection] = codec
        expected[projection] = weight
    bundle, geo = assemble_layer_bundle(
        components,
        bits,
        codecs=codecs,
        kquant_codecs=kquant_codecs,
    )
    base = "language_model.model.layers.0.mlp.switch_mlp"
    write_safetensors_raw(
        pkg / "model-00001-of-00001.safetensors",
        {f"{base}.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes())},
        metadata={
            "format": "mjtq",
            "expert_bundles": encode_bundle_metadata({0: geo}),
        },
    )
    return pkg, expected


def _resident_projection(n_experts, in_features, out_features, *, bits, seed=42):
    from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear

    mod = TurboQuantSwitchLinear(
        in_features,
        out_features,
        n_experts,
        bits=bits,
        seed=seed,
    )
    vals_per_u32 = 32 // bits
    cols = (in_features + vals_per_u32 - 1) // vals_per_u32
    mod.packed = mx.random.randint(
        0,
        2**31,
        (n_experts, out_features, cols),
    ).astype(mx.uint32)
    mod.norms = (mx.random.normal((n_experts, out_features)) * 0.1).astype(
        mx.float16
    )
    mx.eval(mod.packed, mod.norms)
    return mod


def _resident_switch(
    *,
    n_experts=8,
    hidden=64,
    intermediate=32,
    gate_bits=2,
    up_bits=4,
    down_bits=2,
):
    from mlx_lm.models.switch_layers import SwitchGLU
    from moespresso.runtime.owned_switchglu import OwnedSwitchGLU

    mx.random.seed(17)
    shape = SwitchGLU(hidden, intermediate, n_experts)
    return OwnedSwitchGLU(
        gate_proj=_resident_projection(
            n_experts, hidden, intermediate, bits=gate_bits),
        up_proj=_resident_projection(
            n_experts, hidden, intermediate, bits=up_bits),
        down_proj=_resident_projection(
            n_experts, intermediate, hidden, bits=down_bits),
        activation=shape.activation,
    )


def _package_from_resident(tmp_path, resident):
    from conftest import write_safetensors_raw

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    components = {}
    bits = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(resident, projection)
        components[(projection, "packed")] = np.array(proj.packed)
        components[(projection, "norms")] = np.array(proj.norms)
        bits[projection] = int(proj.bits)
    bundle, geo = assemble_layer_bundle(components, bits)
    base = "language_model.model.layers.0.mlp.switch_mlp"
    write_safetensors_raw(
        pkg / "model-00001-of-00001.safetensors",
        {f"{base}.experts.tq_bundle": ("U8", bundle.shape, bundle.tobytes())},
        metadata={"expert_bundles": encode_bundle_metadata({0: geo})},
    )
    return pkg


class _Model(nn.Module):
    def __init__(self, *, hidden=64, intermediate=32, n_experts=8, n_layers=1):
        super().__init__()
        from mlx_lm.models.switch_layers import SwitchGLU

        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = []
        for _ in range(n_layers):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.switch_mlp = SwitchGLU(hidden, intermediate, n_experts)
            self.language_model.model.layers.append(layer)


class _TinySharedMLP(nn.Module):
    def __init__(self, hidden=64, intermediate=32):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _DeepseekV4HashGate(nn.Module):
    def __init__(self, *, n_experts=8, top_k=4):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.calls = []

    def __call__(self, x, input_ids=None):
        if input_ids is None:
            raise AssertionError("DS4 hash gate requires input_ids")
        self.calls.append(input_ids)
        base = input_ids.astype(mx.uint32)
        inds = mx.concatenate(
            [((base + offset) % self.n_experts)[..., None] for offset in range(self.top_k)],
            axis=-1,
        )
        scores = mx.ones(inds.shape, dtype=x.dtype) / self.top_k
        return inds, scores


class _DeepseekV4Moe(nn.Module):
    def __init__(self, *, gate, switch_mlp, shared_experts):
        super().__init__()
        self.gate = gate
        self.switch_mlp = switch_mlp
        self.shared_experts = shared_experts

    def __call__(self, x, input_ids=None):
        inds, scores = self.gate(x, input_ids=input_ids)
        inds = inds.astype(mx.uint32)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype).reshape(x.shape)
        return y + self.shared_experts(x)


class _SparseModel(nn.Module):
    def __init__(self, *, hidden=64, intermediate=32, n_experts=8):
        super().__init__()
        from mlx_lm.models.switch_layers import SwitchGLU

        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.gate = nn.Linear(hidden, n_experts, bias=False)
        layer.mlp.switch_mlp = SwitchGLU(hidden, intermediate, n_experts)
        layer.mlp.shared_expert = _TinySharedMLP(hidden, intermediate)
        layer.mlp.shared_expert_gate = nn.Linear(hidden, 1, bias=False)
        layer.mlp.norm_topk_prob = True
        layer.mlp.num_experts = n_experts
        layer.mlp.top_k = min(4, n_experts)
        layer.mlp.sharding_group = None
        self.language_model.model.layers = [layer]


class _DeepseekV4SparseModel(nn.Module):
    def __init__(self, *, gate, switch_mlp, shared_experts):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.gate = gate
        layer.mlp.switch_mlp = switch_mlp
        layer.mlp.shared_experts = shared_experts
        self.language_model.model.layers = [layer]


def test_routed_key_filter_is_exact_to_switch_tq_payloads():
    assert _is_routed_expert_key(
        "language_model.model.layers.0.mlp.switch_mlp.experts.tq_bundle")

    assert not _is_routed_expert_key(
        "language_model.model.layers.0.mlp.gate_proj.weight")
    assert not _is_routed_expert_key(
        "language_model.model.layers.0.mlp.shared_expert.up_proj.weight")
    # legacy stacked keys never reach the resident-load filter: the expert
    # index refuses stacked packages before any weights are loaded
    assert not _is_routed_expert_key(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.tq_packed")


def test_install_pooled_switchglus_replaces_indexed_layers(tmp_path):
    pkg = _package(tmp_path, layers=(0,))
    model = _Model(n_layers=2)
    index = build_expert_index(pkg)

    installed = install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=4,
        seed=42,
    )

    assert installed == 1
    layer0 = model.language_model.model.layers[0]
    layer1 = model.language_model.model.layers[1]
    assert isinstance(layer0.mlp.switch_mlp, PooledSwitchGLU)
    assert not isinstance(layer1.mlp.switch_mlp, PooledSwitchGLU)
    assert layer0.mlp.switch_mlp.gate_proj.bits == 2
    assert layer0.mlp.switch_mlp.up_proj.bits == 4
    assert layer0.mlp.switch_mlp.gate_proj.pool.capacity == 4


def test_install_pooled_switchglus_kquant_combines_gate_up_by_default(tmp_path):
    pkg, expected = _kquant_package(tmp_path)
    model = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    index = build_expert_index(pkg)

    installed = install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=2,
        seed=42,
    )

    assert installed == 1
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    assert isinstance(switch.gate_proj, PooledCombinedGateUpKQuantLinear)
    assert switch.gate_proj.kquant_type == "iq2_xxs"
    assert switch.down_proj.kquant_type == "q2_k"
    assert switch.up_proj.pool is switch.gate_proj.pool
    assert switch.gate_proj.pool.row_cache.consumers == 2
    model.eval()
    switch.gate_proj.pool.ensure([1])
    slot = switch.gate_proj.pool.slot_of(1)
    np.testing.assert_array_equal(
        np.array(switch.gate_proj.pool.weight[slot]),
        np.concatenate(
            [expected["gate_proj"][1], expected["up_proj"][1]],
            axis=0,
        ),
    )


def test_install_pooled_switchglus_can_share_kquant_gate_up_pool(
    tmp_path,
):
    pkg, expected = _kquant_package(tmp_path)
    model = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    index = build_expert_index(pkg)

    installed = install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=2,
        seed=42,
    )

    assert installed == 1
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    assert isinstance(switch.gate_proj, PooledCombinedGateUpKQuantLinear)
    assert switch.up_proj.pool is switch.gate_proj.pool
    assert switch.down_proj.pool is not switch.gate_proj.pool
    assert switch.gate_proj.pool.row_cache.consumers == 2
    model.eval()
    assert switch.gate_proj.pool.slot_nbytes() == (
        expected["gate_proj"][1].nbytes + expected["up_proj"][1].nbytes
    )

    switch.gate_proj.pool.ensure([1])
    slot = switch.gate_proj.pool.slot_of(1)
    np.testing.assert_array_equal(
        np.array(switch.gate_proj.pool.weight[slot]),
        np.concatenate(
            [expected["gate_proj"][1], expected["up_proj"][1]],
            axis=0,
        ),
    )


def test_pooled_kquant_projection_dispatches_to_mlx_kquant(tmp_path, monkeypatch):
    import sys
    import types

    pkg, _expected = _kquant_package(tmp_path)
    model = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=2,
        seed=42,
    )
    calls = []

    def fake_gather_qmm(x, weight, scales, codec, **kwargs):
        calls.append((x, weight, scales, codec, kwargs))
        return mx.zeros((*x.shape[:-1], kwargs["rhs_indices"].shape[-1], 1, 256))

    monkeypatch.setitem(
        sys.modules,
        "mlx_kquant",
        types.SimpleNamespace(gather_qmm=fake_gather_qmm),
    )
    x = mx.zeros((1, 256), dtype=mx.float16)
    indices = mx.array([[1]], dtype=mx.uint32)
    proj = model.language_model.model.layers[0].mlp.switch_mlp.down_proj

    y = proj(x, indices)
    mx.eval(y)

    assert y.shape == (1, 1, 1, 256)
    assert calls[0][3] == "q2_k"
    assert calls[0][4]["transpose"] is True
    assert calls[0][4]["sorted_indices"] is False


def test_combined_kquant_gate_up_dispatches_two_routed_gathers(
    tmp_path,
    monkeypatch,
):
    import sys
    import types

    pkg, _expected = _kquant_package(tmp_path)
    model = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=4,
        seed=42,
    )
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    calls = []

    def fake_gather_qmm(x, weight, scales, codec, **kwargs):
        calls.append((weight.shape, codec, kwargs))
        rhs = kwargs["rhs_indices"]
        return mx.zeros((*x.shape[:-1], rhs.shape[-1], 1, weight.shape[1]))

    monkeypatch.setitem(
        sys.modules,
        "mlx_kquant",
        types.SimpleNamespace(gather_qmm=fake_gather_qmm),
    )

    x = mx.zeros((1, 256), dtype=mx.float16)
    indices = mx.array([[1, 2, 3, 0]], dtype=mx.uint32)
    mx.eval(switch(x, indices))

    assert [call[1] for call in calls] == ["iq2_xxs", "q2_k"]
    assert calls[0][0][1] == 512  # gate rows + up rows in one K-quant gather
    assert switch.gate_proj.matmul_slot_calls == 1
    assert switch.up_proj.matmul_slot_calls == 0
    assert switch.down_proj.matmul_slot_calls == 1

    stats = ssd_streaming_stats(model)
    assert stats["resident_slots"] == 8  # 4 combined gate/up + 4 down
    assert stats["expert_misses"] == 8
    assert stats["expert_loads"] == 8
    assert stats["bundle_row_preads"] == 4
    assert stats["bundle_cached_takes"] == 4
    assert stats["routed_matmul_calls"] == 2
    assert stats["routed_gate_matmul_calls"] == 1
    assert stats["routed_up_matmul_calls"] == 0
    assert stats["routed_down_matmul_calls"] == 1
    assert stats["slot_table_rebuilds"] == 2


def test_combined_kquant_gate_up_pipelined_builder_uses_combined_gather(
    tmp_path,
    monkeypatch,
):
    import sys
    import types

    pkg, _expected = _kquant_package(tmp_path)
    model = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=4,
        seed=42,
    )
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    calls = []

    def fake_gather_qmm(x, weight, scales, codec, **kwargs):
        calls.append(codec)
        rhs = kwargs["rhs_indices"]
        return mx.zeros((*x.shape[:-1], rhs.shape[-1], 1, weight.shape[1]))

    monkeypatch.setitem(
        sys.modules,
        "mlx_kquant",
        types.SimpleNamespace(gather_qmm=fake_gather_qmm),
    )

    x = mx.zeros((1, 256), dtype=mx.float16)
    indices = mx.array([[1, 2, 3, 0]], dtype=mx.uint32)
    switch.publish_slots(indices)
    y = switch.build_pipelined(x, indices)
    mx.eval(y)

    assert calls == ["iq2_xxs", "q2_k"]
    assert switch.gate_proj.matmul_slot_calls == 1
    assert switch.up_proj.matmul_slot_calls == 0
    assert switch.down_proj.matmul_slot_calls == 1


def test_install_pooled_switchglus_wraps_real_sparse_moe_shape(tmp_path):
    """Real Qwen3Next sparse blocks have shared_expert beside switch_mlp. The
    product install path wraps that whole block so routed loads can overlap the
    resident shared expert during decode."""
    pkg = _package(tmp_path, layers=(0,))
    model = _SparseModel()
    index = build_expert_index(pkg)

    installed = install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=4,
        seed=42,
    )

    assert installed == 1
    mlp = model.language_model.model.layers[0].mlp
    assert isinstance(mlp, PooledSparseMoeBlock)
    assert isinstance(mlp.switch_mlp, PooledSwitchGLU)


def test_install_pooled_switchglus_leaves_deepseek_v4_moe_unwrapped_by_default(
    tmp_path,
):
    resident_switch = _resident_switch(n_experts=8)
    pkg = _package_from_resident(tmp_path, resident_switch)
    model = _DeepseekV4SparseModel(
        gate=_DeepseekV4HashGate(n_experts=8),
        switch_mlp=resident_switch,
        shared_experts=_TinySharedMLP(),
    )
    index = build_expert_index(pkg)

    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=4,
        seed=42,
    )

    mlp = model.language_model.model.layers[0].mlp
    assert not isinstance(mlp, PooledDeepseekV4MoEBlock)
    assert isinstance(mlp.switch_mlp, PooledSwitchGLU)


def test_pooled_deepseek_v4_moe_block_matches_direct_hash_route(tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_FUSED_GATE_UP", False)
    monkeypatch.setattr(psg, "_RING_DECODE", False)

    resident_switch = _resident_switch(n_experts=8, gate_bits=2, up_bits=4)
    pkg = _package_from_resident(tmp_path, resident_switch)
    gate = _DeepseekV4HashGate(n_experts=8, top_k=4)
    shared = _TinySharedMLP()
    reference = _DeepseekV4Moe(
        gate=gate,
        switch_mlp=resident_switch,
        shared_experts=shared,
    )
    model = _DeepseekV4SparseModel(
        gate=gate,
        switch_mlp=resident_switch,
        shared_experts=shared,
    )
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=4,
        seed=42,
        wrap_deepseek_v4_moe=True,
    )
    pooled = model.language_model.model.layers[0].mlp

    assert isinstance(pooled, PooledDeepseekV4MoEBlock)
    x_decode = mx.random.normal((1, 1, 64)).astype(mx.float16)
    ids_decode = mx.array([[3]], dtype=mx.int32)
    x_prefill = mx.random.normal((1, 3, 64)).astype(mx.float16)
    ids_prefill = mx.array([[1, 2, 3]], dtype=mx.int32)

    ref_decode = reference(x_decode, input_ids=ids_decode)
    got_decode = pooled(x_decode, input_ids=ids_decode)
    ref_prefill = reference(x_prefill, input_ids=ids_prefill)
    got_prefill = pooled(x_prefill, input_ids=ids_prefill)
    mx.eval(ref_decode, got_decode, ref_prefill, got_prefill)

    np.testing.assert_array_equal(np.array(ref_decode), np.array(got_decode))
    np.testing.assert_array_equal(np.array(ref_prefill), np.array(got_prefill))
    assert pooled.switch_mlp.decode_moe_block_calls == 1
    assert pooled.switch_mlp.routed_weighted_sum_calls == 2
    assert pooled.switch_mlp.routed_weighted_sum_slot_elements == 16
    assert pooled.switch_mlp.routed_weighted_sum_output_elements == 256
    assert pooled.switch_mlp.index_sync_calls == 2
    stats = ssd_streaming_stats(model)
    assert stats["routed_weighted_sum_calls"] == 2
    assert stats["routed_weighted_sum_slot_elements"] == 16
    assert stats["routed_weighted_sum_output_elements"] == 256
    assert gate.calls


def test_pooled_deepseek_v4_ring_decode_matches_legacy(tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_FUSED_GATE_UP", True)
    monkeypatch.setattr(psg, "_ONDEVICE_REMAP", True)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    monkeypatch.setattr(psg, "_GATE_MOD", [False])
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])

    resident_switch = _resident_switch(
        n_experts=8,
        gate_bits=4,
        up_bits=4,
        down_bits=2,
    )
    shared = _TinySharedMLP()
    xs = [mx.random.normal((1, 1, 64)).astype(mx.float16) for _ in range(4)]
    ids = [
        mx.array([[0]], dtype=mx.int32),
        mx.array([[1]], dtype=mx.int32),
        mx.array([[4]], dtype=mx.int32),
        mx.array([[0]], dtype=mx.int32),
    ]
    outs = {}
    stats = {}

    for use_ring in (False, True):
        monkeypatch.setattr(psg, "_RING_DECODE", use_ring)
        root = tmp_path / ("ring" if use_ring else "legacy")
        root.mkdir()
        pkg = _package_from_resident(root, resident_switch)
        model = _DeepseekV4SparseModel(
            gate=_DeepseekV4HashGate(n_experts=8, top_k=4),
            switch_mlp=resident_switch,
            shared_experts=shared,
        )
        install_pooled_switchglus(
            model,
            package_dir=pkg,
            index=build_expert_index(pkg),
            capacity_per_layer=4,
            seed=42,
            wrap_deepseek_v4_moe=True,
        )
        block = model.language_model.model.layers[0].mlp
        block.eval()
        got = []
        for x, input_ids in zip(xs, ids):
            y = block(x, input_ids=input_ids)
            mx.eval(y)
            got.append(np.array(y))
        outs[use_ring] = got
        stats[use_ring] = ssd_streaming_stats(model)

    for legacy, ringed in zip(outs[False], outs[True]):
        np.testing.assert_array_equal(legacy, ringed)
    assert stats[True]["pipelined_layers"] == len(xs)
    assert stats[True]["decode_moe_block_calls"] == len(xs)
    assert stats[True]["index_sync_calls"] == 0
    assert stats[True]["index_resync_calls"] == 0
    assert stats[False]["index_sync_calls"] == len(xs)
    assert stats[False]["index_resync_calls"] == len(xs)


def test_install_pooled_switchglus_fails_on_geometry_mismatch(tmp_path):
    pkg = _package(tmp_path, layers=(0,), hidden=64)
    model = _Model(hidden=128, n_layers=1)
    index = build_expert_index(pkg)

    with pytest.raises(SSDStreamingBuildError, match="skeleton input dim"):
        install_pooled_switchglus(
            model,
            package_dir=pkg,
            index=index,
            capacity_per_layer=4,
        )


def test_ssd_streaming_stats_reports_pool_activity(tmp_path):
    pkg = _package(tmp_path, layers=(0,))
    model = _Model(n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=4,
    )
    switch = model.language_model.model.layers[0].mlp.switch_mlp

    x = mx.random.normal((1, 64)).astype(mx.float16)
    indices = mx.array([[1, 2, 3, 4]], dtype=mx.uint32)
    mx.eval(switch(x, indices))

    stats = ssd_streaming_stats(model)
    assert stats["enabled"] is True
    assert stats["switch_modules"] == 1
    assert stats["resident_slots"] == 12  # 4 experts in each of gate/up/down pools
    assert switch.gate_proj.pool.eviction_policy == "lfu"
    assert stats["expert_misses"] == 12
    assert stats["expert_loads"] == 12
    assert stats["expert_load_seconds"] >= 0.0
    assert stats["projection_load_wait_calls"] == 1
    assert stats["projection_no_miss_calls"] == 0
    assert stats["projection_load_parallel_calls"] == 1
    assert stats["projection_load_wait_seconds"] >= 0.0
    assert stats["switch_calls"] == 1
    assert stats["decode_calls"] == 1
    assert stats["prefill_calls"] == 0
    assert stats["direct_calls"] == 1
    assert stats["token_layers"] == 1
    assert stats["unique_active_experts"] == 4
    assert stats["seen_experts"] == 4
    assert stats["decode_seen_experts"] == 4
    assert stats["prefill_seen_experts"] == 0
    assert stats["max_unique_active_experts"] == 4
    # phase counters: __call__'s host read is timed (index_resync) and the
    # routed graph build is timed once misses are resident. index_sync and
    # decode_moe_block accrue only through PooledSparseMoeBlock.
    assert stats["index_resync_seconds"] > 0.0
    assert stats["index_resync_calls"] == 1
    assert stats["routed_build_seconds"] > 0.0
    assert stats["index_sync_calls"] == 0
    assert stats["decode_moe_block_calls"] == 0
    assert stats["routed_matmul_calls"] == 3
    assert stats["routed_gate_matmul_calls"] == 1
    assert stats["routed_up_matmul_calls"] == 1
    assert stats["routed_down_matmul_calls"] == 1
    assert stats["routed_matmul_slot_elements"] == 12
    # remap_ondevice built each pool's slot table exactly once (no eviction)
    assert stats["slot_table_rebuilds"] == 3

    per_layer = ssd_streaming_layer_stats(model)
    assert len(per_layer) == 1
    assert per_layer[0]["layer"] == 0
    assert per_layer[0]["capacity"] == 4
    assert per_layer[0]["num_experts"] == 8
    assert per_layer[0]["projection_pool_count"] == 3
    assert per_layer[0]["slot_bytes"] == 2304
    assert per_layer[0]["expert_misses"] == 12
    assert per_layer[0]["projection_load_wait_calls"] == 1
    assert per_layer[0]["projection_no_miss_calls"] == 0
    assert per_layer[0]["index_resync_calls"] == 1
    assert per_layer[0]["routed_matmul_calls"] == 3
    assert per_layer[0]["routed_matmul_slot_elements"] == 12
    assert per_layer[0]["token_layers"] == 1
    assert per_layer[0]["decode_seen_experts"] == 4


def test_expert_hotlist_round_trip_warm_starts_residency(tmp_path):
    """Demand saved from one session warm-starts the next session's pools
    (the hottest experts become resident at load time, before any request)."""
    from moespresso.runtime.ssd_streaming_build import (
        load_expert_hotlist,
        save_expert_hotlist,
    )

    pkg = _package(tmp_path, layers=(0,))
    index = build_expert_index(pkg)

    model_a = _Model(n_layers=1)
    install_pooled_switchglus(
        model_a, package_dir=pkg, index=index, capacity_per_layer=3)
    switch_a = model_a.language_model.model.layers[0].mlp.switch_mlp
    x = mx.random.normal((1, 64)).astype(mx.float16)
    # expert 5 is demanded twice, 1 and 2 once -> 5 must rank hottest
    mx.eval(switch_a(x, mx.array([[5, 1]], dtype=mx.uint32)))
    mx.eval(switch_a(x, mx.array([[5, 2]], dtype=mx.uint32)))
    hot_path = tmp_path / "hotlist.json"
    assert save_expert_hotlist(model_a, hot_path) == 1

    model_b = _Model(n_layers=1)
    install_pooled_switchglus(
        model_b, package_dir=pkg, index=index, capacity_per_layer=2)
    seeded = load_expert_hotlist(model_b, hot_path)
    switch_b = model_b.language_model.model.layers[0].mlp.switch_mlp
    assert seeded == 6  # 2 free slots x 3 pools
    # capacity 2: the two hottest experts of the saved demand are resident
    assert switch_b.gate_proj.pool.resident_ids() == {5, 1} or \
        switch_b.gate_proj.pool.resident_ids() == {5, 2}
    assert 5 in switch_b.down_proj.pool.resident_ids()

    # A missing optional hotlist is a no-op.
    assert load_expert_hotlist(model_b, tmp_path / "absent.json") == 0


def test_suggest_capacity_overrides_spends_budget_on_churniest_layers():
    rows = [
        {"layer": 0, "capacity": 4, "expert_loads": 10, "expert_misses": 10,
         "seen_experts": 8, "decode_seen_experts": 6, "max_unique_active_experts": 4},
        {"layer": 1, "capacity": 4, "expert_loads": 30, "expert_misses": 30,
         "seen_experts": 7, "decode_seen_experts": 5, "max_unique_active_experts": 4},
        {"layer": 2, "capacity": 4, "expert_loads": 1, "expert_misses": 1,
         "seen_experts": 9, "decode_seen_experts": 8, "max_unique_active_experts": 4},
    ]

    assert suggest_capacity_overrides_from_layer_stats(
        rows,
        extra_slot_budget=5,
    ) == {1: 7, 0: 6}


def test_suggest_capacity_overrides_can_spend_a_byte_budget():
    rows = [
        {"layer": 0, "capacity": 4, "slot_bytes": 100,
         "expert_loads": 40, "expert_misses": 40,
         "seen_experts": 8, "decode_seen_experts": 6,
         "max_unique_active_experts": 4},
        {"layer": 1, "capacity": 4, "slot_bytes": 30,
         "expert_loads": 30, "expert_misses": 30,
         "seen_experts": 7, "decode_seen_experts": 5,
         "max_unique_active_experts": 4},
        {"layer": 2, "capacity": 4, "slot_bytes": 30,
         "expert_loads": 10, "expert_misses": 10,
         "seen_experts": 9, "decode_seen_experts": 8,
         "max_unique_active_experts": 4},
    ]

    assert suggest_capacity_overrides_from_layer_stats(
        rows,
        extra_byte_budget=90,
    ) == {1: 7}


def test_suggest_capacity_overrides_can_target_decode_seen_experts():
    rows = [
        {"layer": 0, "capacity": 4, "expert_loads": 40, "expert_misses": 40,
         "seen_experts": 10, "decode_seen_experts": 5,
         "max_unique_active_experts": 10},
        {"layer": 1, "capacity": 4, "expert_loads": 30, "expert_misses": 30,
         "seen_experts": 6, "decode_seen_experts": 6,
         "max_unique_active_experts": 6},
    ]

    assert suggest_capacity_overrides_from_layer_stats(
        rows,
        extra_slot_budget=4,
        target="decode",
    ) == {0: 5, 1: 6}


def test_suggest_capacity_overrides_requires_one_budget_kind():
    rows = [{"layer": 0, "capacity": 4}]

    with pytest.raises(ValueError, match="exactly one"):
        suggest_capacity_overrides_from_layer_stats(rows)

    with pytest.raises(ValueError, match="exactly one"):
        suggest_capacity_overrides_from_layer_stats(
            rows,
            extra_slot_budget=1,
            extra_byte_budget=1,
        )

    with pytest.raises(ValueError, match="target"):
        suggest_capacity_overrides_from_layer_stats(
            rows,
            extra_slot_budget=1,
            target="prefill",
        )


def test_grow_ssd_streaming_capacity_applies_layer_overrides(tmp_path):
    pkg = _package(tmp_path, layers=(0,))
    model = _Model(n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=4,
    )
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    switch.gate_proj.pool.ensure([1, 2])

    applied = grow_ssd_streaming_capacity(model, {0: 6})

    assert applied == {0: 6}
    assert switch.gate_proj.pool.capacity == 6
    assert switch.gate_proj.pool.resident_ids() == {1, 2}
    assert ssd_streaming_layer_stats(model)[0]["capacity"] == 6
    assert ssd_streaming_stats(model)["capacity_overrides"] == {0: 6}


def test_maybe_adapt_ssd_streaming_capacity_uses_byte_budget(tmp_path):
    pkg = _package(tmp_path, layers=(0, 1))
    model = _Model(n_layers=2)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=4,
    )
    object.__setattr__(model, "_moespresso_ssd_streaming_capacity", 4)
    object.__setattr__(model, "_moespresso_ssd_streaming_capacity_overrides", {})
    layer0 = model.language_model.model.layers[0].mlp.switch_mlp
    layer1 = model.language_model.model.layers[1].mlp.switch_mlp
    layer0.seen_experts.update({0, 1, 2, 3, 4, 5, 6})
    layer0.decode_seen_experts.update({0, 1, 2, 3, 4, 5, 6})
    layer0.max_unique_active_experts = 7
    layer1.seen_experts.update({0, 1, 2, 3, 4, 5})
    layer1.decode_seen_experts.update({0, 1, 2, 3, 4, 5})
    layer1.max_unique_active_experts = 6
    for projection in ("gate_proj", "up_proj", "down_proj"):
        getattr(layer0, projection).pool.total_loads = 50
        getattr(layer0, projection).pool.total_misses = 50
        getattr(layer1, projection).pool.total_loads = 10
        getattr(layer1, projection).pool.total_misses = 10

    result = maybe_adapt_ssd_streaming_capacity(
        model,
        available_bytes=10_000,
        min_available_bytes=0,
        max_extra_bytes=2 * 2304,
        seed_hot=False,
    )

    assert result["applied"] == {0: 6}
    assert result["used_extra_bytes"] == 2 * 2304
    assert result["seed_hot"] is False
    assert result["seeded_slots"] == 0
    assert result["elapsed_seconds"] >= 0.0
    assert layer0.gate_proj.pool.capacity == 6
    assert layer1.gate_proj.pool.capacity == 4
    assert ssd_streaming_stats(model)["adaptive_growth"] == result

    second = maybe_adapt_ssd_streaming_capacity(
        model,
        available_bytes=10_000,
        min_available_bytes=0,
        max_extra_bytes=2 * 2304,
        seed_hot=False,
    )
    assert second["applied"] == {}
    assert second["extra_byte_budget"] == 0
    assert second["seeded_slots"] == 0
    assert second["elapsed_seconds"] >= 0.0


def test_maybe_adapt_ssd_streaming_capacity_respects_memory_floor(tmp_path):
    pkg = _package(tmp_path, layers=(0,))
    model = _Model(n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=4,
    )
    object.__setattr__(model, "_moespresso_ssd_streaming_capacity", 4)
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    switch.seen_experts.update({0, 1, 2, 3, 4, 5, 6})
    switch.decode_seen_experts.update({0, 1, 2, 3, 4, 5, 6})
    switch.max_unique_active_experts = 7
    for projection in ("gate_proj", "up_proj", "down_proj"):
        getattr(switch, projection).pool.total_loads = 50
        getattr(switch, projection).pool.total_misses = 50

    result = maybe_adapt_ssd_streaming_capacity(
        model,
        available_bytes=4_000,
        min_available_bytes=4_000,
        max_extra_bytes=2304,
    )

    assert result["applied"] == {}
    assert result["extra_byte_budget"] == 0
    assert result["seeded_slots"] == 0
    assert result["elapsed_seconds"] >= 0.0
    assert switch.gate_proj.pool.capacity == 4


def test_maybe_adapt_ssd_streaming_capacity_noops_without_layers():
    assert maybe_adapt_ssd_streaming_capacity("not-a-model") == {
        "enabled": False,
        "applied": {},
    }


def test_install_pooled_switchglus_accepts_per_layer_capacity_overrides(tmp_path):
    pkg = _package(tmp_path, layers=(0, 1))
    model = _Model(n_layers=2)
    index = build_expert_index(pkg)

    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=2,
        capacity_overrides={1: 4},
    )

    layer0 = model.language_model.model.layers[0].mlp.switch_mlp
    layer1 = model.language_model.model.layers[1].mlp.switch_mlp
    assert layer0.gate_proj.pool.capacity == 2
    assert layer1.gate_proj.pool.capacity == 4


def test_install_pooled_switchglus_threads_eviction_policy(tmp_path):
    pkg = _package(tmp_path, layers=(0,))
    model = _Model(n_layers=1)
    index = build_expert_index(pkg)

    install_pooled_switchglus(
        model,
        package_dir=pkg,
        index=index,
        capacity_per_layer=2,
        eviction_policy="lfu",
    )

    switch = model.language_model.model.layers[0].mlp.switch_mlp
    assert switch.gate_proj.pool.eviction_policy == "lfu"


def test_budget_payload_is_reported_verbatim():
    budget = CapacityBudget(
        available_bytes=10,
        resident_base_bytes=2,
        kv_activation_allowance_bytes=3,
        safety_margin_bytes=1,
        bytes_per_capacity_unit=2,
        min_capacity=2,
        max_capacity=8,
    )

    assert _budget_payload(budget) == {
        "available_bytes": 10,
        "resident_base_bytes": 2,
        "runtime_resident_bytes": 0,
        "kv_activation_allowance_bytes": 3,
        "safety_margin_bytes": 1,
        "bytes_per_capacity_unit": 2,
        "usable_bytes": 4,
        "min_capacity": 2,
        "max_capacity": 8,
    }


def test_package_hotlist_seeds_precisely_but_prior_is_capped(tmp_path):
    """An imatrix-scale hotlist (counts in the hundreds of thousands)
    must seed the exact raw-ranked top experts, yet the installed prior must
    be capped so live traffic can overtake a stale prior within a few
    touches (raw install would make seeded experts un-evictable in LFU)."""
    import json as _json

    from moespresso.runtime.ssd_streaming_build import load_expert_hotlist

    pkg = _package(tmp_path, layers=(0,))
    index = build_expert_index(pkg)
    model = _Model(n_layers=1)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=3)

    hot = tmp_path / "expert_hotlist.json"
    hot.write_text(_json.dumps({
        "version": 1, "kind": "expert_hotlist",
        "source": {"kind": "gguf_imatrix_counts"},
        "layers": {"0": {"6": 356499, "2": 153652, "7": 65513, "1": 9423}},
    }))
    seeded = load_expert_hotlist(model, hot)
    switch = model.language_model.model.layers[0].mlp.switch_mlp

    assert seeded == 9  # 3 free slots x 3 pools
    # precise raw ranking decided which experts became resident
    assert switch.gate_proj.pool.resident_ids() == {6, 2, 7}
    # but the installed prior is capped (default 8): top stays 8, ranks kept
    freq = switch.gate_proj.pool._freq
    assert max(freq.values()) == 8
    assert freq[6] == 8 and freq[6] >= freq[2] >= freq[7] >= freq[1] >= 1

    # a genuinely hot new expert overtakes the stale prior within ~cap touches
    x = mx.random.normal((1, 64)).astype(mx.float16)
    for _ in range(9):
        mx.eval(switch(x, mx.array([[3, 3]], dtype=mx.uint32)))
    assert freq if False else switch.gate_proj.pool._freq[3] > 8
    assert 3 in switch.gate_proj.pool.resident_ids()


def _hotlist_payload(layers):
    import json as _json
    return _json.dumps({"version": 1, "kind": "expert_hotlist", "layers": layers})


def test_seed_expert_residency_precedence_and_kill_switch(tmp_path, monkeypatch):
    """Layered seeding: saved demand (0.60 tier) > package imatrix
    hotlist (0.40 tier) > nothing; MOESPRESSO_SSD_HOTLIST=0 disables."""
    from moespresso.package.hotlist import HOTLIST_NAME
    from moespresso.runtime.ssd_streaming_build import (
        default_saved_hotlist_path,
        seed_expert_residency,
    )

    monkeypatch.setenv("MOESPRESSO_HOTLIST_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MOESPRESSO_SSD_HOTLIST", raising=False)
    pkg = _package(tmp_path, layers=(0,))
    index = build_expert_index(pkg)

    def fresh_model():
        m = _Model(n_layers=1)
        install_pooled_switchglus(
            m, package_dir=pkg, index=index, capacity_per_layer=2)
        return m

    # package hotlist alone -> package tier, top experts resident
    (pkg / HOTLIST_NAME).write_text(_hotlist_payload({"0": {"4": 100, "6": 50}}))
    m = fresh_model()
    info = seed_expert_residency(m, pkg)
    assert info["source"] == "package" and info["seeded"] == 6
    assert m.language_model.model.layers[0].mlp.switch_mlp.gate_proj.pool \
        .resident_ids() == {4, 6}

    # saved demand wins over the package hotlist
    saved = default_saved_hotlist_path(pkg)
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(_hotlist_payload({"0": {"1": 3, "2": 2}}))
    m2 = fresh_model()
    info2 = seed_expert_residency(m2, pkg)
    assert info2["source"] == "saved"
    assert m2.language_model.model.layers[0].mlp.switch_mlp.gate_proj.pool \
        .resident_ids() == {1, 2}

    # kill switch: nothing seeded
    monkeypatch.setenv("MOESPRESSO_SSD_HOTLIST", "0")
    m3 = fresh_model()
    info3 = seed_expert_residency(m3, pkg)
    assert info3["source"] == "disabled" and info3["seeded"] == 0
    assert m3.language_model.model.layers[0].mlp.switch_mlp.gate_proj.pool \
        .resident_ids() == set()


def test_seed_expert_residency_can_prewarm_all_full_capacity(
        tmp_path, monkeypatch):
    """Full-resident packages can move cold routed reads into build time.

    The prewarm must walk projection pools in bundle-row-cache-sized windows.
    Four experts therefore cost four bundle-row preads instead of twelve
    projection-component reads.
    """
    from moespresso.runtime.ssd_streaming_build import seed_expert_residency

    monkeypatch.setenv("MOESPRESSO_SSD_PREWARM_EXPERTS", "all")
    monkeypatch.delenv("MOESPRESSO_SSD_HOTLIST", raising=False)
    pkg = _package(tmp_path, n_experts=4, layers=(0,))
    index = build_expert_index(pkg)
    model = _Model(n_layers=1, n_experts=4)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=4)

    info = seed_expert_residency(model, pkg)
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    pools = [
        switch.gate_proj.pool,
        switch.up_proj.pool,
        switch.down_proj.pool,
    ]

    assert info["source"] == "all"
    assert info["seeded"] == 12
    assert all(pool.resident_ids() == {0, 1, 2, 3} for pool in pools)
    assert switch.gate_proj.pool.row_cache.total_preads == 4
    assert switch.gate_proj.pool.row_cache.total_cached_takes == 8


def test_seed_expert_residency_default_prewarms_all_at_full_capacity(
        tmp_path, monkeypatch):
    """With no explicit prewarm request and every pool at full capacity, the
    default prewarms every expert (pool residency selects the routed prefill
    kernel, so serving must start on the fully resident numerics). The
    saved-demand tier does not preempt it: hotlist seeding covers only the
    recorded demand and would leave the cold segmented path live."""
    from moespresso.runtime.ssd_streaming_build import (
        default_saved_hotlist_path,
        seed_expert_residency,
    )

    monkeypatch.setenv("MOESPRESSO_HOTLIST_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MOESPRESSO_SSD_PREWARM_EXPERTS", raising=False)
    monkeypatch.delenv("MOESPRESSO_SSD_PREWARM_DEFAULT", raising=False)
    monkeypatch.delenv("MOESPRESSO_SSD_HOTLIST", raising=False)
    pkg = _package(tmp_path, n_experts=4, layers=(0,))
    index = build_expert_index(pkg)

    saved = default_saved_hotlist_path(pkg)
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(_hotlist_payload({"0": {"1": 3, "2": 2}}))

    model = _Model(n_layers=1, n_experts=4)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=4)
    info = seed_expert_residency(model, pkg)
    switch = model.language_model.model.layers[0].mlp.switch_mlp

    assert info["source"] == "all-default"
    assert info["seeded"] == 12
    assert switch.gate_proj.pool.resident_ids() == {0, 1, 2, 3}
    assert switch.down_proj.pool.resident_ids() == {0, 1, 2, 3}


def test_seed_expert_residency_default_prewarm_needs_full_capacity(
        tmp_path, monkeypatch):
    """Below full capacity the default prewarm cannot engage (a partial "all"
    preload would silently leave the first request on the demand-miss path);
    the hotlist tiers keep their behavior."""
    from moespresso.runtime.ssd_streaming_build import (
        default_saved_hotlist_path,
        seed_expert_residency,
    )

    monkeypatch.setenv("MOESPRESSO_HOTLIST_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MOESPRESSO_SSD_PREWARM_EXPERTS", raising=False)
    monkeypatch.delenv("MOESPRESSO_SSD_PREWARM_DEFAULT", raising=False)
    monkeypatch.delenv("MOESPRESSO_SSD_HOTLIST", raising=False)
    pkg = _package(tmp_path, n_experts=4, layers=(0,))
    index = build_expert_index(pkg)

    saved = default_saved_hotlist_path(pkg)
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(_hotlist_payload({"0": {"1": 3, "2": 2}}))

    model = _Model(n_layers=1, n_experts=4)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=3)
    info = seed_expert_residency(model, pkg)

    assert info["source"] == "saved"
    assert model.language_model.model.layers[0].mlp.switch_mlp.gate_proj.pool \
        .resident_ids() == {1, 2}


def test_seed_expert_residency_default_prewarm_kill_switch(
        tmp_path, monkeypatch):
    """MOESPRESSO_SSD_PREWARM_DEFAULT=0 restores lazy hotlist seeding at full
    capacity; an explicit MOESPRESSO_SSD_PREWARM_EXPERTS=all still wins over
    the kill switch because the switch only controls the default."""
    from moespresso.runtime.ssd_streaming_build import (
        default_saved_hotlist_path,
        seed_expert_residency,
    )

    monkeypatch.setenv("MOESPRESSO_HOTLIST_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MOESPRESSO_SSD_PREWARM_DEFAULT", "0")
    monkeypatch.delenv("MOESPRESSO_SSD_PREWARM_EXPERTS", raising=False)
    monkeypatch.delenv("MOESPRESSO_SSD_HOTLIST", raising=False)
    pkg = _package(tmp_path, n_experts=4, layers=(0,))
    index = build_expert_index(pkg)

    saved = default_saved_hotlist_path(pkg)
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(_hotlist_payload({"0": {"1": 3, "2": 2}}))

    def fresh_model():
        m = _Model(n_layers=1, n_experts=4)
        install_pooled_switchglus(
            m, package_dir=pkg, index=index, capacity_per_layer=4)
        return m

    m = fresh_model()
    info = seed_expert_residency(m, pkg)
    assert info["source"] == "saved"
    assert m.language_model.model.layers[0].mlp.switch_mlp.gate_proj.pool \
        .resident_ids() == {1, 2}

    # with the hotlist tiers also disabled the pool state is fully cold
    monkeypatch.setenv("MOESPRESSO_SSD_HOTLIST", "0")
    m2 = fresh_model()
    info2 = seed_expert_residency(m2, pkg)
    assert info2["source"] == "disabled" and info2["seeded"] == 0

    # explicit env request overrides the kill switch
    monkeypatch.setenv("MOESPRESSO_SSD_PREWARM_EXPERTS", "all")
    m3 = fresh_model()
    info3 = seed_expert_residency(m3, pkg)
    assert info3["source"] == "all" and info3["seeded"] == 12


def test_seed_expert_residency_prewarm_all_fails_without_full_capacity(
        tmp_path, monkeypatch):
    from moespresso.runtime.ssd_streaming_build import seed_expert_residency

    monkeypatch.setenv("MOESPRESSO_SSD_PREWARM_EXPERTS", "all")
    pkg = _package(tmp_path, n_experts=4, layers=(0,))
    index = build_expert_index(pkg)
    model = _Model(n_layers=1, n_experts=4)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=3)

    with pytest.raises(SSDStreamingBuildError, match="full expert prewarm"):
        seed_expert_residency(model, pkg)


def test_seed_expert_residency_rejects_unknown_prewarm_mode(
        tmp_path, monkeypatch):
    from moespresso.runtime.ssd_streaming_build import seed_expert_residency

    monkeypatch.setenv("MOESPRESSO_SSD_PREWARM_EXPERTS", "some")
    pkg = _package(tmp_path, n_experts=4, layers=(0,))
    index = build_expert_index(pkg)
    model = _Model(n_layers=1, n_experts=4)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=4)

    with pytest.raises(SSDStreamingBuildError, match="PREWARM_EXPERTS"):
        seed_expert_residency(model, pkg)


def test_seed_expert_residency_prewarm_none_skips_prewarm_and_default(
        tmp_path, monkeypatch):
    """MOESPRESSO_SSD_PREWARM_EXPERTS=none is the explicit no-prewarm
    override: it skips both the explicit prewarm and the full-capacity
    default and falls through to the hotlist tiers, so callers that pin
    the prewarm (the quality gates) can run a bounded capacity budget."""
    from moespresso.runtime.ssd_streaming_build import (
        default_saved_hotlist_path,
        seed_expert_residency,
    )

    monkeypatch.setenv("MOESPRESSO_HOTLIST_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MOESPRESSO_SSD_PREWARM_EXPERTS", "none")
    monkeypatch.delenv("MOESPRESSO_SSD_PREWARM_DEFAULT", raising=False)
    monkeypatch.delenv("MOESPRESSO_SSD_HOTLIST", raising=False)
    pkg = _package(tmp_path, n_experts=4, layers=(0,))
    index = build_expert_index(pkg)

    saved = default_saved_hotlist_path(pkg)
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(_hotlist_payload({"0": {"1": 3, "2": 2}}))

    # Full-capacity pools: the all-default prewarm would normally engage,
    # so 'none' skipping it (and seeding from the hotlist tier instead) is
    # the override evidence.
    model = _Model(n_layers=1, n_experts=4)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=4)
    info = seed_expert_residency(model, pkg)
    assert info["source"] == "saved"
    assert model.language_model.model.layers[0].mlp.switch_mlp.gate_proj.pool \
        .resident_ids() == {1, 2}

    # Bounded capacity: 'none' runs where 'all' raises.
    bounded = _Model(n_layers=1, n_experts=4)
    install_pooled_switchglus(
        bounded, package_dir=pkg, index=index, capacity_per_layer=3)
    info2 = seed_expert_residency(bounded, pkg)
    assert info2["source"] == "saved"


def test_serve_persists_demand_after_generation(tmp_path, monkeypatch):
    """A served request saves its expert demand (default-ON) so the
    next session warm-starts from the saved-demand tier; the kill switch
    suppresses the save."""
    import json as _json

    from moespresso.runtime.serve import generate_with_metadata
    from moespresso.runtime.ssd_streaming_build import (
        default_saved_hotlist_path,
        seed_expert_residency,
    )

    monkeypatch.setenv("MOESPRESSO_HOTLIST_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MOESPRESSO_SSD_HOTLIST", raising=False)
    pkg = _package(tmp_path, layers=(0,))
    index = build_expert_index(pkg)
    model = _Model(n_layers=1)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=3)
    object.__setattr__(model, "_moespresso_ssd_hotlist",
                       seed_expert_residency(model, pkg))

    switch = model.language_model.model.layers[0].mlp.switch_mlp
    x = mx.random.normal((1, 64)).astype(mx.float16)
    mx.eval(switch(x, mx.array([[5, 1]], dtype=mx.uint32)))

    def fake_stream(model, tokenizer, prompt, **kwargs):
        return iter(())

    generate_with_metadata(model, tokenizer=None, prompt=[1],
                           stream_generate_fn=fake_stream,
                           sampler_factory=lambda **kw: None)
    saved = default_saved_hotlist_path(pkg)
    assert saved.exists()
    assert "5" in _json.loads(saved.read_text())["layers"]["0"]

    # Startup warmup exercises the live model but must not make its synthetic
    # routing demand durable across sessions.
    saved.unlink()
    generate_with_metadata(
        model,
        tokenizer=None,
        prompt=[1],
        persist_expert_demand=False,
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kw: None,
    )
    assert not saved.exists()

    # kill switch suppresses persistence
    monkeypatch.setenv("MOESPRESSO_SSD_HOTLIST", "0")
    generate_with_metadata(model, tokenizer=None, prompt=[1],
                           stream_generate_fn=fake_stream,
                           sampler_factory=lambda **kw: None)
    assert not saved.exists()


def test_growth_budget_default_is_env_tunable(monkeypatch):
    """The growth cap default is 2 GiB (the live memory floor is the
    safety contract; the old 512 MiB cap bound first on the shippable) and
    MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB overrides it."""
    from moespresso.runtime.ssd_streaming_build import (
        _growth_max_extra_bytes_default,
    )

    monkeypatch.delenv("MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB", raising=False)
    assert _growth_max_extra_bytes_default() == 2 << 30
    monkeypatch.setenv("MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB", "0.5")
    assert _growth_max_extra_bytes_default() == 512 << 20
    monkeypatch.setenv("MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB", "0")
    assert _growth_max_extra_bytes_default() == 0


def test_deterministic_available_bytes_quiet_vs_busy(monkeypatch):
    """Capacity budgets from min(total - OS reserve, available now):
    deterministic when memory is idle, clamped by live availability under
    pressure (never budget memory someone else is using)."""
    from moespresso.runtime import ssd_streaming_build as ssb

    class _VM:
        def __init__(self, total, available):
            self.total, self.available = total, available

    monkeypatch.setenv("MOESPRESSO_SSD_OS_RESERVE_GB", "5")
    import psutil as _psutil
    # idle memory: available within 25% of the deterministic budget -> the
    # gap is reclaimable cache, the deterministic number wins (reproducible)
    monkeypatch.setattr(_psutil, "virtual_memory",
                        lambda: _VM(16 << 30, 12 << 30))
    assert ssb._deterministic_available_bytes() == 11 << 30
    monkeypatch.setattr(_psutil, "virtual_memory",
                        lambda: _VM(16 << 30, int(9 << 30)))  # 9 >= 0.75*11
    assert ssb._deterministic_available_bytes() == 11 << 30
    # genuinely under memory pressure: live availability clamps
    monkeypatch.setattr(_psutil, "virtual_memory",
                        lambda: _VM(16 << 30, 6 << 30))
    assert ssb._deterministic_available_bytes() == 6 << 30


def test_lookahead_decision_is_single_sourced(tmp_path, monkeypatch):
    """When the gate path is not live, a requested lookahead
    must be fully disabled: no spares carved and no predictors wired (the
    env must not be re-read after the gating decision)."""
    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setenv("MOESPRESSO_SSD_LOOKAHEAD", "4")
    monkeypatch.setattr(psg, "_RING_DECODE", False)  # gate path not live

    pkg = _package(tmp_path, layers=(0, 1))

    # mimic the build's gating + wiring decision flow without a real model
    lookahead_env = 4
    gate_live = psg._RING_DECODE
    if not gate_live:
        lookahead_env = 0
    spare_slots = 16 if lookahead_env > 0 else 0
    assert lookahead_env == 0 and spare_slots == 0

    # and a directly-wired model only happens through install_lookahead,
    # which the single-sourced decision never calls when gated off
    model = _Model(n_layers=2)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=4,
        spare_slots=spare_slots)
    sw = model.language_model.model.layers[0].mlp.switch_mlp
    assert sw.lookahead_w is None and sw.lookahead_target is None
    assert sw.gate_proj.pool.spare_slots == 0


def test_max_memory_cap_bounds_the_budget(monkeypatch):
    """MOESPRESSO_SSD_MAX_MEMORY_GB / --max-memory-gb caps the capacity
    budget: the UX knob that doubles as a simulator for a memory-constrained
    host's pool."""
    import psutil as _psutil

    from moespresso.runtime import ssd_streaming_build as ssb

    class _VM:
        def __init__(self, total, available):
            self.total, self.available = total, available

    monkeypatch.setenv("MOESPRESSO_SSD_OS_RESERVE_GB", "5")
    monkeypatch.setattr(_psutil, "virtual_memory",
                        lambda: _VM(16 << 30, 12 << 30))
    monkeypatch.setenv("MOESPRESSO_SSD_MAX_MEMORY_GB", "3.5")
    assert ssb._deterministic_available_bytes() == int(3.5 * (1 << 30))
    monkeypatch.delenv("MOESPRESSO_SSD_MAX_MEMORY_GB")
    assert ssb._deterministic_available_bytes() == 11 << 30


def _fake_kquant_module(
        gather_calls, qmm_calls, sorted_calls=None, fused_swiglu_calls=None):
    import types

    def fake_gather_qmm(x, weight, scales, codec, **kwargs):
        gather_calls.append(codec)
        if x.ndim >= 4:
            return mx.zeros((*x.shape[:-1], weight.shape[1]), dtype=x.dtype)
        rhs = kwargs["rhs_indices"]
        return mx.zeros(
            (*x.shape[:-1], rhs.shape[-1], 1, weight.shape[1]), dtype=x.dtype)

    def fake_dequantize(w2, s2, codec, dtype=None):
        qmm_calls.append((codec, int(w2.shape[0]), dtype))
        # One block per row in the test package: 256 weights per row.
        return mx.zeros((w2.shape[0], 256), dtype=dtype)

    module = types.SimpleNamespace(
        gather_qmm=fake_gather_qmm,
        dequantize=fake_dequantize,
    )
    if sorted_calls is not None:
        # Barrier-free eligibility probes for this attribute, so only fakes
        # that opt in expose it.
        def fake_gather_qmm_sorted(x, weight, scales, codec, sorted_ids, **kw):
            sorted_calls.append((codec, sorted_ids))
            return mx.zeros((x.shape[0], weight.shape[1]), dtype=x.dtype)

        module.gather_qmm_sorted = fake_gather_qmm_sorted
    if fused_swiglu_calls is not None:
        # The fused route probes for this attribute per call, so fakes
        # without it exercise the unfused gather_qmm_sorted + activation pair.
        def fake_gather_qmm_sorted_swiglu(
                x, weight, scales, codec, sorted_ids, gate_out,
                swiglu_limit, **kw):
            fused_swiglu_calls.append(
                (codec, sorted_ids, int(gate_out), float(swiglu_limit)))
            return mx.zeros((x.shape[0], int(gate_out)), dtype=x.dtype)

        module.gather_qmm_sorted_swiglu = fake_gather_qmm_sorted_swiglu
    return module


def test_bulk_sorted_prefill_uses_segmented_kquant_matmul(tmp_path, monkeypatch):
    """At bulk sorted-prefill row counts, the combined gate/up K-quant path
    runs one f32 dequantize + GEMM per active-expert segment (weights read
    once per expert), never the per-pair gather kernel."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    pkg, _expected = _kquant_package(tmp_path)
    model = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=4, seed=42)
    switch = model.language_model.model.layers[0].mlp.switch_mlp

    monkeypatch.setattr(psg, "_SEGMENTED_PREFILL_MIN_ROWS", 8)
    gather_calls = []
    qmm_calls = []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _fake_kquant_module(gather_calls, qmm_calls))

    tokens, top_k = 32, 4
    x = mx.zeros((tokens, 256), dtype=mx.float16)
    indices = mx.array(
        np.random.default_rng(0).integers(0, 4, (tokens, top_k)).astype(np.uint32))
    y = switch(x, indices)
    mx.eval(y)

    assert y.shape[-1] == 256 and y.shape[0] == tokens
    assert y.dtype == mx.float16  # segments compute in f32, output back to f16
    assert not gather_calls
    assert switch.segmented_prefill_calls >= 1
    assert switch.gate_proj.segmented_matmul_calls == switch.segmented_prefill_calls
    assert switch.down_proj.segmented_matmul_calls == switch.segmented_prefill_calls
    # Each active expert dequantizes once per projection per segmented call,
    # always in f32: 4 experts x (combined gate/up rows 512, down rows 256).
    assert all(d is mx.float32 for _c, _n, d in qmm_calls)
    gate_deq = [n for c, n, _d in qmm_calls if c == "iq2_xxs"]
    down_deq = [n for c, n, _d in qmm_calls if c == "q2_k"]
    assert len(gate_deq) == 4 * switch.segmented_prefill_calls
    assert len(down_deq) == 4 * switch.segmented_prefill_calls
    assert set(gate_deq) == {512}
    assert set(down_deq) == {256}


def test_small_sorted_prefill_keeps_gather_path(tmp_path, monkeypatch):
    """Below the segmented-row threshold the sorted path stays on the gather
    kernel, where the per-pair vector matmul wins."""
    import sys

    pkg, _expected = _kquant_package(tmp_path)
    model = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=4, seed=42)
    switch = model.language_model.model.layers[0].mlp.switch_mlp

    gather_calls = []
    qmm_calls = []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _fake_kquant_module(gather_calls, qmm_calls))

    tokens, top_k = 32, 4  # 128 sorted rows, below the 4096-row threshold
    x = mx.zeros((tokens, 256), dtype=mx.float16)
    indices = mx.array(
        np.random.default_rng(0).integers(0, 4, (tokens, top_k)).astype(np.uint32))
    y = switch(x, indices)
    mx.eval(y)

    assert gather_calls == ["iq2_xxs", "q2_k"]
    assert not qmm_calls
    assert switch.segmented_prefill_calls == 0


def test_sorted_expert_segments_walks_runs():
    import moespresso.runtime.pooled_switchglu as psg

    segments = psg._sorted_expert_segments(
        np.array([2, 2, 2, 5, 7, 7], dtype=np.uint32))
    assert segments == [(2, 0, 3), (5, 3, 4), (7, 4, 6)]
    assert psg._sorted_expert_segments(np.array([], dtype=np.uint32)) == []


def _full_resident_kquant_switch(tmp_path, *, capacity=4):
    """Install a K-quant pooled switch and prewarm every expert resident."""
    from moespresso.runtime.ssd_streaming_build import seed_all_expert_residency

    pkg, _expected = _kquant_package(tmp_path)
    model = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=capacity,
        seed=42)
    if capacity >= 4:
        seed_all_expert_residency(model)
    return model.language_model.model.layers[0].mlp.switch_mlp


def test_bulk_prefill_barrier_free_when_full_resident(tmp_path, monkeypatch):
    """At full residency (capacity == num_experts, all experts prewarmed) the
    bulk prefill routes to gather_qmm_sorted with device-resident slot ids and
    zero host index reads: index_sync/index_resync stay at 0 and numpy never
    materializes an MLX array inside the call."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    switch = _full_resident_kquant_switch(tmp_path, capacity=4)

    # The route ships gated off (measured served-neutral); force it on so the
    # test exercises the eligibility predicates and the device-only path.
    monkeypatch.setattr(psg, "_BARRIER_FREE_PREFILL", True)
    monkeypatch.setattr(psg, "_SEGMENTED_PREFILL_MIN_ROWS", 8)
    gather_calls, qmm_calls, sorted_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_module(gather_calls, qmm_calls, sorted_calls))

    # np.asarray spy: the barrier-free route must never pull an MLX array to
    # the host (that read is the per-layer graph drain the route removes).
    real_asarray = np.asarray
    mlx_asarray_hits = []

    def _spy_asarray(obj, *args, **kwargs):
        if isinstance(obj, mx.array):
            mlx_asarray_hits.append(obj.shape)
        return real_asarray(obj, *args, **kwargs)

    monkeypatch.setattr(psg.np, "asarray", _spy_asarray)

    tokens, top_k = 32, 4
    x = mx.zeros((tokens, 256), dtype=mx.float16)
    indices = mx.array(
        np.random.default_rng(0).integers(0, 4, (tokens, top_k)).astype(np.uint32))

    # The overlap seam also declines without a host read at this shape.
    assert switch.begin_projection_load(indices) is None

    y = switch(x, indices)
    mx.eval(y)
    monkeypatch.setattr(psg.np, "asarray", real_asarray)

    assert y.shape == (tokens, top_k, 256)
    assert y.dtype == mx.float16
    assert mlx_asarray_hits == []
    assert switch.index_sync_calls == 0
    assert switch.index_resync_calls == 0
    assert switch.barrier_free_prefill_calls == 1
    assert switch.segmented_prefill_calls == 0
    # Full prewarm seeds ascending experts into ascending slots, so both slot
    # tables are the identity map and the route shares one sorted id array
    # between the gate/up and down GEMMs (no per-pool remap, no inter-GEMM
    # re-permutation).
    assert switch.barrier_free_identity_calls == 1
    assert sorted_calls[0][1] is sorted_calls[1][1]
    assert not gather_calls and not qmm_calls
    # One sorted-kernel call per projection (combined gate/up, then down),
    # each with a device uint32 [S] id array, ascending slot ids.
    assert [codec for codec, _ids in sorted_calls] == ["iq2_xxs", "q2_k"]
    for _codec, ids in sorted_calls:
        assert isinstance(ids, mx.array)
        assert ids.dtype == mx.uint32
        assert ids.shape == (tokens * top_k,)
        ids_np = np.array(ids)
        assert np.all(np.diff(ids_np.astype(np.int64)) >= 0)
        assert ids_np.max() < 4  # every id resolves to a valid pool slot
    assert switch.gate_proj.matmul_slot_calls == 1
    assert switch.down_proj.matmul_slot_calls == 1
    assert switch.total_calls == 1 and switch.prefill_calls == 1


def test_barrier_free_prefill_requires_full_capacity(tmp_path, monkeypatch):
    """capacity < num_experts fails the eligibility check closed: the bulk
    prefill keeps the existing (segmented) path with its host index reads."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    switch = _full_resident_kquant_switch(tmp_path, capacity=3)

    monkeypatch.setattr(psg, "_BARRIER_FREE_PREFILL", True)
    monkeypatch.setattr(psg, "_SEGMENTED_PREFILL_MIN_ROWS", 8)
    gather_calls, qmm_calls, sorted_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_module(gather_calls, qmm_calls, sorted_calls))

    tokens, top_k = 32, 4
    x = mx.zeros((tokens, 256), dtype=mx.float16)
    # Three active experts fit the capacity-3 pool and take the direct sorted path.
    indices = mx.array(
        np.random.default_rng(0).integers(0, 3, (tokens, top_k)).astype(np.uint32))
    y = switch(x, indices)
    mx.eval(y)

    assert y.shape == (tokens, top_k, 256)
    assert not sorted_calls
    assert switch.barrier_free_prefill_calls == 0
    assert switch.segmented_prefill_calls >= 1
    assert switch.index_resync_calls >= 1


def test_barrier_free_non_identity_slots_keep_per_pool_remap(
        tmp_path, monkeypatch):
    """A fully resident pool whose slots are not the identity map (demand
    fill order differs from seeding order) must keep the general per-pool
    remap: separate slot-id arrays per projection and no identity-route
    engagement."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    switch = _full_resident_kquant_switch(tmp_path, capacity=4)

    # Swap two experts' slots in the gate pool bookkeeping. The fake kernel
    # module records structure only, so the rows need not move; the point is
    # that a non-identity table must fail the identity check and produce
    # per-pool remapped ids.
    pool = switch.gate_proj.pool
    pool._slot_of[0], pool._slot_of[1] = pool._slot_of[1], pool._slot_of[0]
    pool._slot_table_dirty = True

    monkeypatch.setattr(psg, "_BARRIER_FREE_PREFILL", True)
    monkeypatch.setattr(psg, "_SEGMENTED_PREFILL_MIN_ROWS", 8)
    gather_calls, qmm_calls, sorted_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_module(gather_calls, qmm_calls, sorted_calls))

    tokens, top_k = 32, 4
    x = mx.zeros((tokens, 256), dtype=mx.float16)
    indices = mx.array(
        np.random.default_rng(0).integers(0, 4, (tokens, top_k)).astype(np.uint32))
    y = switch(x, indices)
    mx.eval(y)

    assert y.shape == (tokens, top_k, 256)
    assert switch.barrier_free_prefill_calls == 1
    assert switch.barrier_free_identity_calls == 0
    assert not pool.slot_table_is_identity()
    assert switch.down_proj.pool.slot_table_is_identity()
    assert len(sorted_calls) == 2
    assert sorted_calls[0][1] is not sorted_calls[1][1]
    gate_ids = np.array(sorted_calls[0][1])
    down_ids = np.array(sorted_calls[1][1])
    assert not np.array_equal(gate_ids, down_ids)


def test_barrier_free_fused_swiglu_engages_at_identity_slots(
        tmp_path, monkeypatch):
    """When mlx_kquant ships gather_qmm_sorted_swiglu, the barrier-free
    identity route fuses the combined gate/up GEMM with the SwiGLU: one
    fused call plus one plain sorted call for down, replacing the two-call
    gather_qmm_sorted shape. The fused op receives the gate width and the
    activation's swiglu_limit (0.0 here: the test switch's mlx_lm SwiGLU has
    no swiglu_limit attribute)."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    switch = _full_resident_kquant_switch(tmp_path, capacity=4)

    monkeypatch.setattr(psg, "_BARRIER_FREE_PREFILL", True)
    monkeypatch.setattr(psg, "_SEGMENTED_PREFILL_MIN_ROWS", 8)
    monkeypatch.setattr(psg, "_FUSED_SORTED_SWIGLU", True)
    gather_calls, qmm_calls, sorted_calls, fused_calls = [], [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_module(gather_calls, qmm_calls, sorted_calls, fused_calls))

    tokens, top_k = 32, 4
    x = mx.zeros((tokens, 256), dtype=mx.float16)
    indices = mx.array(
        np.random.default_rng(0).integers(0, 4, (tokens, top_k)).astype(np.uint32))
    y = switch(x, indices)
    mx.eval(y)

    assert y.shape == (tokens, top_k, 256)
    assert y.dtype == mx.float16
    assert switch.barrier_free_prefill_calls == 1
    assert switch.barrier_free_identity_calls == 1
    assert switch.barrier_free_fused_swiglu_calls == 1
    assert not gather_calls and not qmm_calls
    # One fused gate/up+SwiGLU call, then one plain sorted call for down.
    assert [codec for codec, *_ in fused_calls] == ["iq2_xxs"]
    assert [codec for codec, _ids in sorted_calls] == ["q2_k"]
    codec, ids, gate_out, swiglu_limit = fused_calls[0]
    assert gate_out == 256
    assert swiglu_limit == 0.0
    assert isinstance(ids, mx.array)
    assert ids.dtype == mx.uint32
    assert ids.shape == (tokens * top_k,)
    # The identity route still shares one sorted id array between the fused
    # gate/up GEMM and the down GEMM.
    assert ids is sorted_calls[0][1]
    assert switch.gate_proj.matmul_slot_calls == 1
    assert switch.down_proj.matmul_slot_calls == 1


def test_barrier_free_fused_swiglu_kill_switch_falls_back(
        tmp_path, monkeypatch):
    """MOESPRESSO_SSD_FUSED_SORTED_SWIGLU=0 (the module constant) keeps the
    unfused gather_qmm_sorted + activation pair even when the fused kernel
    is available."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    switch = _full_resident_kquant_switch(tmp_path, capacity=4)

    monkeypatch.setattr(psg, "_BARRIER_FREE_PREFILL", True)
    monkeypatch.setattr(psg, "_SEGMENTED_PREFILL_MIN_ROWS", 8)
    monkeypatch.setattr(psg, "_FUSED_SORTED_SWIGLU", False)
    gather_calls, qmm_calls, sorted_calls, fused_calls = [], [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_module(gather_calls, qmm_calls, sorted_calls, fused_calls))

    tokens, top_k = 32, 4
    x = mx.zeros((tokens, 256), dtype=mx.float16)
    indices = mx.array(
        np.random.default_rng(0).integers(0, 4, (tokens, top_k)).astype(np.uint32))
    y = switch(x, indices)
    mx.eval(y)

    assert y.shape == (tokens, top_k, 256)
    assert switch.barrier_free_prefill_calls == 1
    assert switch.barrier_free_identity_calls == 1
    assert switch.barrier_free_fused_swiglu_calls == 0
    assert not fused_calls
    assert [codec for codec, _ids in sorted_calls] == ["iq2_xxs", "q2_k"]


def test_barrier_free_fused_swiglu_stays_off_non_identity(
        tmp_path, monkeypatch):
    """Non-identity slot tables keep the unfused pair even with the fused
    kernel available and enabled: the fused route is scoped to the identity
    slot layout the barrier-free route certifies."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    switch = _full_resident_kquant_switch(tmp_path, capacity=4)

    pool = switch.gate_proj.pool
    pool._slot_of[0], pool._slot_of[1] = pool._slot_of[1], pool._slot_of[0]
    pool._slot_table_dirty = True

    monkeypatch.setattr(psg, "_BARRIER_FREE_PREFILL", True)
    monkeypatch.setattr(psg, "_SEGMENTED_PREFILL_MIN_ROWS", 8)
    monkeypatch.setattr(psg, "_FUSED_SORTED_SWIGLU", True)
    gather_calls, qmm_calls, sorted_calls, fused_calls = [], [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_module(gather_calls, qmm_calls, sorted_calls, fused_calls))

    tokens, top_k = 32, 4
    x = mx.zeros((tokens, 256), dtype=mx.float16)
    indices = mx.array(
        np.random.default_rng(0).integers(0, 4, (tokens, top_k)).astype(np.uint32))
    y = switch(x, indices)
    mx.eval(y)

    assert y.shape == (tokens, top_k, 256)
    assert switch.barrier_free_prefill_calls == 1
    assert switch.barrier_free_identity_calls == 0
    assert switch.barrier_free_fused_swiglu_calls == 0
    assert not fused_calls
    assert [codec for codec, _ids in sorted_calls] == ["iq2_xxs", "q2_k"]


def test_sub_threshold_bulk_prefill_keeps_gather_when_full_resident(
        tmp_path, monkeypatch):
    """Below _SEGMENTED_PREFILL_MIN_ROWS a fully resident pool still takes the
    gather path, where the per-pair vector kernel wins."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    switch = _full_resident_kquant_switch(tmp_path, capacity=4)

    monkeypatch.setattr(psg, "_BARRIER_FREE_PREFILL", True)
    gather_calls, qmm_calls, sorted_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_module(gather_calls, qmm_calls, sorted_calls))

    tokens, top_k = 32, 4  # 128 sorted rows, below the 4096-row threshold
    x = mx.zeros((tokens, 256), dtype=mx.float16)
    indices = mx.array(
        np.random.default_rng(0).integers(0, 4, (tokens, top_k)).astype(np.uint32))
    y = switch(x, indices)
    mx.eval(y)

    assert gather_calls == ["iq2_xxs", "q2_k"]
    assert not sorted_calls
    assert switch.barrier_free_prefill_calls == 0


def _full_resident_kquant_ds4_block(
        tmp_path, *, capacity=4, top_k=4, shared_experts=None):
    """Install a K-quant pooled DS4 MoE block; prewarm at full capacity."""
    from mlx_lm.models.switch_layers import SwitchGLU

    from moespresso.runtime.ssd_streaming_build import seed_all_expert_residency

    pkg, _expected = _kquant_package(tmp_path)
    model = _DeepseekV4SparseModel(
        gate=_DeepseekV4HashGate(n_experts=4, top_k=top_k),
        switch_mlp=SwitchGLU(256, 256, 4),
        shared_experts=(
            shared_experts
            if shared_experts is not None
            else _TinySharedMLP(hidden=256, intermediate=32)
        ),
    )
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=capacity,
        seed=42, wrap_deepseek_v4_moe=True)
    if capacity >= 4:
        seed_all_expert_residency(model)
    block = model.language_model.model.layers[0].mlp
    block.eval()
    return model, block


def _fake_kquant_decode_module(gather_calls):
    """Fake mlx_kquant for decode-shape structural tests.

    Records (codec, rhs_indices, sorted_indices) per gather_qmm call and
    returns the real decode output shape ([..., K, 1, out_features])."""
    import types

    def fake_gather_qmm(x, weight, scales, codec, rhs_indices=None, **kw):
        gather_calls.append((codec, rhs_indices, kw.get("sorted_indices")))
        return mx.zeros(
            (*rhs_indices.shape, 1, weight.shape[1]), dtype=x.dtype)

    return types.SimpleNamespace(gather_qmm=fake_gather_qmm)


def _value_kquant_decode_module():
    """Fake mlx_kquant whose gather_qmm output depends on the selected slots.

    Each slot row gets a signature from its wire bytes, scaled by the input
    sum, so two paths produce equal outputs only when they select the same
    pool slots for the same routed ids. This is the index-plumbing
    equivalence probe for the barrier-free decode route against the ring
    path (the real kernel is shared between the two, so equal plumbing
    means equal math)."""
    import types

    def value_gather_qmm(x, weight, scales, codec, rhs_indices=None, **kw):
        sig = weight.astype(mx.float32).sum(axis=-1)  # [slots, rows]
        sel = sig[rhs_indices.reshape(-1)]
        out = (x.astype(mx.float32).sum() * sel).reshape(
            *rhs_indices.shape, 1, sig.shape[-1])
        return out.astype(x.dtype)

    return types.SimpleNamespace(gather_qmm=value_gather_qmm)


def test_deepseek_v4_decode_barrier_free_when_full_resident(
        tmp_path, monkeypatch):
    """At full residency the DS4 decode block consumes router ids on device:
    no ring export, no worker submit, no per-layer kick, no host index read,
    no LFU touch; the flush knob is the only in-block commit site."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_DECODE_FLUSH_LAYERS", 1)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    monkeypatch.setattr(psg, "_GATE_MOD", [False])
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])

    model, block = _full_resident_kquant_ds4_block(tmp_path, capacity=4)
    switch = block.switch_mlp
    gather_calls = []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _fake_kquant_decode_module(gather_calls))
    hits_before = switch.gate_proj.pool.total_hits

    # np.asarray spy: the barrier-free decode route must never pull an MLX
    # array to the host (routing stays on device end to end).
    real_asarray = np.asarray
    mlx_asarray_hits = []

    def _spy_asarray(obj, *args, **kwargs):
        if isinstance(obj, mx.array):
            mlx_asarray_hits.append(obj.shape)
        return real_asarray(obj, *args, **kwargs)

    monkeypatch.setattr(psg.np, "asarray", _spy_asarray)

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    ids = mx.array([[1]], dtype=mx.int32)
    y = block(x, input_ids=ids)
    mx.eval(y)
    monkeypatch.setattr(psg.np, "asarray", real_asarray)

    assert y.shape == (1, 1, 256)
    assert mlx_asarray_hits == []
    assert switch.barrier_free_decode_calls == 1
    assert switch.barrier_free_decode_flush_calls == 1  # depth 1, layer 0
    # The ring/native-gate machinery never ran: these are the off-arm
    # counters the served A/B reads as the engagement proof.
    assert switch.pipelined_layers == 0
    assert switch.block_exit_kick_calls == 0
    assert switch.router_export_seconds == 0.0
    assert switch.pipeline_read_seconds == 0.0
    assert switch.pipeline_join_seconds == 0.0
    assert switch.index_sync_calls == 0
    assert switch.index_resync_calls == 0
    assert switch.decode_moe_block_calls == 1
    # Full prewarm seeds identity slot tables, so the router ids feed the
    # gathers directly as device uint32 indices, unsorted.
    assert [codec for codec, _rhs, _s in gather_calls] == ["iq2_xxs", "q2_k"]
    for _codec, rhs, sorted_indices in gather_calls:
        assert sorted_indices is False
        assert isinstance(rhs, mx.array)
        assert rhs.dtype == mx.uint32
        np.testing.assert_array_equal(
            np.array(rhs).reshape(-1), np.array([1, 2, 3, 0]))
    # LFU touch accounting is skipped: full-resident pools never evict.
    assert switch.gate_proj.pool.total_hits == hits_before

    # Flush cadence: at depth 4 the layer-0 block queues without a kick.
    monkeypatch.setattr(psg, "_DECODE_FLUSH_LAYERS", 4)
    y = block(x, input_ids=ids)
    mx.eval(y)
    assert switch.barrier_free_decode_calls == 2
    assert switch.barrier_free_decode_flush_calls == 1

    stats = ssd_streaming_stats(model)
    assert stats["barrier_free_decode_calls"] == 2
    assert stats["barrier_free_decode_flush_calls"] == 1
    layer_stats = ssd_streaming_layer_stats(model)
    assert layer_stats[0]["barrier_free_decode_calls"] == 2
    assert layer_stats[0]["barrier_free_decode_flush_calls"] == 1


def test_deepseek_v4_decode_barrier_free_matches_ring_and_kill_switch(
        tmp_path, monkeypatch):
    """The barrier-free decode route selects exactly the slots the ring path
    publishes (value-bearing fake kernel), and the kill switch
    (MOESPRESSO_SSD_BARRIER_FREE_DECODE=0, the module constant) keeps the
    ring path with its counters."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_RING_DECODE", True)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    monkeypatch.setattr(psg, "_GATE_MOD", [False])
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _value_kquant_decode_module())

    mx.random.seed(11)
    xs = [mx.random.normal((1, 1, 256)).astype(mx.float16) for _ in range(4)]
    mx.eval(*xs)
    ids = [mx.array([[step]], dtype=mx.int32) for step in (0, 1, 3, 0)]
    shared = _TinySharedMLP(hidden=256, intermediate=32)
    outs = {}
    switches = {}

    for barrier_free in (False, True):
        monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", barrier_free)
        root = tmp_path / ("bf" if barrier_free else "ring")
        root.mkdir()
        _model, block = _full_resident_kquant_ds4_block(
            root, capacity=4, shared_experts=shared)
        got = []
        for x, input_ids in zip(xs, ids):
            y = block(x, input_ids=input_ids)
            mx.eval(y)
            got.append(np.array(y))
        outs[barrier_free] = got
        switches[barrier_free] = block.switch_mlp

    for ring_y, bf_y in zip(outs[False], outs[True]):
        np.testing.assert_array_equal(ring_y, bf_y)
    assert switches[True].barrier_free_decode_calls == len(xs)
    assert switches[True].pipelined_layers == 0
    assert switches[True].block_exit_kick_calls == 0
    assert switches[False].barrier_free_decode_calls == 0
    assert switches[False]._barrier_free_decode_ready_cached is False
    assert switches[False].pipelined_layers == len(xs)
    assert switches[False].block_exit_kick_calls == len(xs)


def test_deepseek_v4_decode_barrier_free_fails_closed_on_partial_residency(
        tmp_path, monkeypatch):
    """capacity < num_experts fails the decode certificate closed: decode
    keeps the ring path (the product path for partial residency)."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_RING_DECODE", True)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    monkeypatch.setattr(psg, "_GATE_MOD", [False])
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _value_kquant_decode_module())

    _model, block = _full_resident_kquant_ds4_block(
        tmp_path, capacity=3, top_k=2)
    switch = block.switch_mlp

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert switch.barrier_free_decode_calls == 0
    assert switch._barrier_free_decode_ready_cached is False
    assert switch.pipelined_layers == 1
    assert switch.block_exit_kick_calls == 1


def test_deepseek_v4_decode_barrier_free_non_identity_uses_ondevice_remap(
        tmp_path, monkeypatch):
    """A fully resident pool whose slots are not the identity map still
    engages, through one on-device slot-table gather per pool."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])

    _model, block = _full_resident_kquant_ds4_block(tmp_path, capacity=4)
    switch = block.switch_mlp
    pool = switch.gate_proj.pool
    pool._slot_of[0], pool._slot_of[1] = pool._slot_of[1], pool._slot_of[0]
    pool._slot_table_dirty = True

    gather_calls = []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _fake_kquant_decode_module(gather_calls))

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert switch.barrier_free_decode_calls == 1
    assert not pool.slot_table_is_identity()
    assert switch.down_proj.pool.slot_table_is_identity()
    # Router ids [1, 2, 3, 0]: the gate pool's swapped table remaps 1 -> 0
    # and 0 -> 1; the down pool stays identity.
    gate_rhs = np.array(gather_calls[0][1]).reshape(-1)
    down_rhs = np.array(gather_calls[1][1]).reshape(-1)
    np.testing.assert_array_equal(gate_rhs, np.array([0, 2, 3, 1]))
    np.testing.assert_array_equal(down_rhs, np.array([1, 2, 3, 0]))


def test_deepseek_v4_decode_barrier_free_drains_pending_ring_futures(
        tmp_path, monkeypatch):
    """A mixed-path session (some layers on the ring path, some barrier-free)
    still drains the ring workers once per token at the last MoE layer, so
    publish ordering holds and worker errors surface."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])

    _model, block = _full_resident_kquant_ds4_block(tmp_path, capacity=4)
    switch = block.switch_mlp
    gather_calls = []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _fake_kquant_decode_module(gather_calls))

    assert block.pipeline_is_last
    psg._PIPE_PREV.append(psg._PIPELINE_EXECUTOR.submit(lambda: None))
    psg._GATE_PENDING.append(psg._PIPELINE_EXECUTOR.submit(lambda: None))

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert switch.barrier_free_decode_calls == 1
    assert psg._PIPE_PREV == []
    assert psg._GATE_PENDING == []
    assert switch.pipeline_join_seconds > 0.0


def _fake_kquant_decode_fused_module(gather_calls, pair_calls, sum_calls):
    """Fake mlx_kquant exposing the fused decode routed matvec surface.

    Records per-call arguments and returns the real output shapes
    (pair -> [B, gate_out], expert sum -> [1, N])."""
    import types

    def fake_gather_qmm(x, weight, scales, codec, rhs_indices=None, **kw):
        gather_calls.append((codec, rhs_indices, kw.get("sorted_indices")))
        return mx.zeros(
            (*rhs_indices.shape, 1, weight.shape[1]), dtype=x.dtype)

    def fake_pair_swiglu(x, weight, scales, codec, ids, route_weights,
                         gate_out, swiglu_limit, **kw):
        pair_calls.append(
            (codec, ids, route_weights, int(gate_out), float(swiglu_limit)))
        return mx.zeros((ids.shape[0], int(gate_out)), dtype=x.dtype)

    def fake_expert_sum(x, weight, scales, codec, ids, **kw):
        sum_calls.append((codec, ids))
        return mx.zeros((1, weight.shape[1]), dtype=x.dtype)

    return types.SimpleNamespace(
        gather_qmm=fake_gather_qmm,
        gather_qmv_pair_swiglu=fake_pair_swiglu,
        gather_qmv_expert_sum=fake_expert_sum,
    )


def test_deepseek_v4_decode_routed_fused_engages_and_kills_weighted_sum(
        tmp_path, monkeypatch):
    """With the flag on, identity slots, and matching codecs, the decode
    routed block runs as the fused matvec pair: baked route weights, the
    expert sum inside the down kernel, no route-weighted-sum reduction, and
    no unfused decode gathers."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_DECODE_ROUTED_FUSED", True)
    monkeypatch.setattr(psg, "_DECODE_FLUSH_LAYERS", 1)
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])

    model, block = _full_resident_kquant_ds4_block(tmp_path, capacity=4)
    switch = block.switch_mlp
    gather_calls, pair_calls, sum_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_decode_fused_module(gather_calls, pair_calls, sum_calls))

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert y.shape == (1, 1, 256)
    assert switch.decode_routed_fused_calls == 1
    assert switch.barrier_free_decode_calls == 1
    assert switch.routed_weighted_sum_calls == 0
    assert gather_calls == []
    assert len(pair_calls) == 1
    codec, ids, route_weights, gate_out, swiglu_limit = pair_calls[0]
    assert codec == "iq2_xxs"
    assert ids.dtype == mx.uint32
    np.testing.assert_array_equal(
        np.array(ids).reshape(-1), np.array([1, 2, 3, 0]))
    assert route_weights.dtype == mx.float32
    assert route_weights.shape == (4,)
    assert gate_out == 256
    assert swiglu_limit == 0.0
    assert len(sum_calls) == 1
    down_codec, down_ids = sum_calls[0]
    assert down_codec == "q2_k"
    np.testing.assert_array_equal(
        np.array(down_ids).reshape(-1), np.array([1, 2, 3, 0]))

    stats = ssd_streaming_stats(model)
    assert stats["decode_routed_fused_calls"] == 1
    assert stats["routed_weighted_sum_calls"] == 0
    layer_stats = ssd_streaming_layer_stats(model)
    assert layer_stats[0]["decode_routed_fused_calls"] == 1


def test_deepseek_v4_decode_routed_fused_kill_switch(tmp_path, monkeypatch):
    """With the kill switch set (MOESPRESSO_DSV4_DECODE_ROUTED_FUSED=0, the
    module constant), the barrier-free decode route keeps the unfused
    gathers and the route-weighted sum even when the fused kernels are
    available."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_DECODE_ROUTED_FUSED", False)
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])

    _model, block = _full_resident_kquant_ds4_block(tmp_path, capacity=4)
    switch = block.switch_mlp
    gather_calls, pair_calls, sum_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_decode_fused_module(gather_calls, pair_calls, sum_calls))

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert switch.decode_routed_fused_calls == 0
    assert switch._decode_routed_fused_ready_cached is False
    assert switch.barrier_free_decode_calls == 1
    assert switch.routed_weighted_sum_calls == 1
    assert pair_calls == []
    assert sum_calls == []
    assert [codec for codec, _rhs, _s in gather_calls] == ["iq2_xxs", "q2_k"]


def test_deepseek_v4_decode_routed_fused_stays_off_non_identity(
        tmp_path, monkeypatch):
    """Non-identity slot tables keep the unfused barrier-free route (with
    its on-device remap) even with the fused kernels enabled: engagement is
    scoped to the identity slot layout, per call."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_DECODE_ROUTED_FUSED", True)
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])

    _model, block = _full_resident_kquant_ds4_block(tmp_path, capacity=4)
    switch = block.switch_mlp
    pool = switch.gate_proj.pool
    pool._slot_of[0], pool._slot_of[1] = pool._slot_of[1], pool._slot_of[0]
    pool._slot_table_dirty = True

    gather_calls, pair_calls, sum_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_decode_fused_module(gather_calls, pair_calls, sum_calls))

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert switch.decode_routed_fused_calls == 0
    # The static eligibility verdict holds; only the per-call identity
    # condition failed.
    assert switch._decode_routed_fused_ready_cached is True
    assert switch.barrier_free_decode_calls == 1
    assert switch.routed_weighted_sum_calls == 1
    assert pair_calls == []
    assert sum_calls == []
    assert [codec for codec, _rhs, _s in gather_calls] == ["iq2_xxs", "q2_k"]


def test_deepseek_v4_decode_routed_fused_fails_closed_on_codec_mismatch(
        tmp_path, monkeypatch):
    """A pool codec outside the instantiated kernel pair fails the one-shot
    eligibility closed; the unfused barrier-free route keeps serving."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_DECODE_ROUTED_FUSED", True)
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])

    _model, block = _full_resident_kquant_ds4_block(tmp_path, capacity=4)
    switch = block.switch_mlp
    monkeypatch.setattr(switch.down_proj, "kquant_type", "q4_k")

    gather_calls, pair_calls, sum_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_decode_fused_module(gather_calls, pair_calls, sum_calls))

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert switch.decode_routed_fused_calls == 0
    assert switch._decode_routed_fused_ready_cached is False
    assert switch.barrier_free_decode_calls == 1
    assert pair_calls == []
    assert sum_calls == []


def test_deepseek_v4_decode_routed_fused_requires_kernel_surface(
        tmp_path, monkeypatch):
    """An installed mlx_kquant without the decode matvec ops fails the
    one-shot eligibility closed."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_DECODE_ROUTED_FUSED", True)
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])

    _model, block = _full_resident_kquant_ds4_block(tmp_path, capacity=4)
    switch = block.switch_mlp
    gather_calls = []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _fake_kquant_decode_module(gather_calls))

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert switch.decode_routed_fused_calls == 0
    assert switch._decode_routed_fused_ready_cached is False
    assert switch.barrier_free_decode_calls == 1
    assert [codec for codec, _rhs, _s in gather_calls] == ["iq2_xxs", "q2_k"]


def _ring_v3_partial_ds4_block(tmp_path, monkeypatch):
    """A partial-residency DS4 block forced onto the v3 ring decode branch.

    Capacity 3 of 4 keeps the barrier-free certificate closed. The ring
    export and worker are replaced by fakes that publish the hash gate's
    known routing ([1, 2] for input id 1 at top_k 2) through the real
    publish_slots path, so the pipelined build consumes real worker-written
    slot-id buffers without a live GPU ring."""
    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_RING_DECODE", True)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    monkeypatch.setattr(psg, "_GATE_MOD", [False])  # no native gate: v3
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])
    monkeypatch.setattr(psg, "_RING_SEQ", [0])

    model, block = _full_resident_kquant_ds4_block(
        tmp_path, capacity=3, top_k=2)
    switch = block.switch_mlp

    monkeypatch.setattr(
        switch, "export_inds",
        lambda inds, seq: mx.zeros((1,), dtype=mx.uint32))

    def _fake_ring_install(seq, K, gate_mod=None):
        switch.publish_slots(mx.array([1, 2], dtype=mx.uint32))

    monkeypatch.setattr(switch, "ring_install", _fake_ring_install)
    return model, block


def test_deepseek_v4_pipelined_decode_fused_engages_at_partial_residency(
        tmp_path, monkeypatch):
    """At partial residency the DS4 ring decode branch runs the fused matvec
    pair over the worker-published slot ids: baked route weights, the expert
    sum inside the down kernel, no route-weighted-sum reduction, no unfused
    gathers, and slot values (not expert ids) reaching the kernels."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_DECODE_ROUTED_FUSED", True)
    monkeypatch.setattr(psg, "_DECODE_RING_FUSED", True)

    model, block = _ring_v3_partial_ds4_block(tmp_path, monkeypatch)
    switch = block.switch_mlp
    gather_calls, pair_calls, sum_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_decode_fused_module(gather_calls, pair_calls, sum_calls))

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert y.shape == (1, 1, 256)
    assert switch.pipelined_decode_fused_calls == 1
    assert switch.pipelined_layers == 1
    assert switch.decode_routed_fused_calls == 0
    assert switch.barrier_free_decode_calls == 0
    assert switch.routed_weighted_sum_calls == 0
    assert gather_calls == []
    assert len(pair_calls) == 1
    codec, ids, route_weights, gate_out, swiglu_limit = pair_calls[0]
    assert codec == "iq2_xxs"
    assert ids.dtype == mx.uint32
    # The worker published slot ids for experts [1, 2], loaded into the
    # empty capacity-3 pool at slots [0, 1]: the kernels see slot values,
    # not expert ids, in router order.
    gate_pool = switch.gate_proj.pool
    np.testing.assert_array_equal(
        np.array(ids).reshape(-1),
        np.array([gate_pool._slot_of[1], gate_pool._slot_of[2]]))
    np.testing.assert_array_equal(
        np.array(ids).reshape(-1), np.array([0, 1]))
    assert route_weights.dtype == mx.float32
    assert route_weights.shape == (2,)
    assert gate_out == 256
    assert swiglu_limit == 0.0
    assert len(sum_calls) == 1
    down_codec, down_ids = sum_calls[0]
    assert down_codec == "q2_k"
    down_pool = switch.down_proj.pool
    np.testing.assert_array_equal(
        np.array(down_ids).reshape(-1),
        np.array([down_pool._slot_of[1], down_pool._slot_of[2]]))

    stats = ssd_streaming_stats(model)
    assert stats["pipelined_decode_fused_calls"] == 1
    assert stats["decode_routed_fused_calls"] == 0
    assert stats["routed_weighted_sum_calls"] == 0
    layer_stats = ssd_streaming_layer_stats(model)
    assert layer_stats[0]["pipelined_decode_fused_calls"] == 1


def test_deepseek_v4_pipelined_decode_fused_ring_kill_switch(
        tmp_path, monkeypatch):
    """MOESPRESSO_DSV4_DECODE_RING_FUSED=0 (the module constant) restores the
    unfused ring composition at partial residency: separate gathers plus the
    route-weighted sum, with the fused kernels untouched even though they
    are available."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_DECODE_ROUTED_FUSED", True)
    monkeypatch.setattr(psg, "_DECODE_RING_FUSED", False)

    _model, block = _ring_v3_partial_ds4_block(tmp_path, monkeypatch)
    switch = block.switch_mlp
    gather_calls, pair_calls, sum_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_decode_fused_module(gather_calls, pair_calls, sum_calls))

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert y.shape == (1, 1, 256)
    assert switch.pipelined_decode_fused_calls == 0
    assert switch.pipelined_layers == 1
    assert switch.routed_weighted_sum_calls == 1
    assert pair_calls == []
    assert sum_calls == []
    assert [codec for codec, _rhs, _s in gather_calls] == ["iq2_xxs", "q2_k"]


def test_deepseek_v4_pipelined_decode_fused_family_kill_switch(
        tmp_path, monkeypatch):
    """The family switch (MOESPRESSO_DSV4_DECODE_ROUTED_FUSED=0) keeps the
    fused kernels off the ring path too: the ring-scoped flag alone cannot
    engage them."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_DECODE_ROUTED_FUSED", False)
    monkeypatch.setattr(psg, "_DECODE_RING_FUSED", True)

    _model, block = _ring_v3_partial_ds4_block(tmp_path, monkeypatch)
    switch = block.switch_mlp
    gather_calls, pair_calls, sum_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_decode_fused_module(gather_calls, pair_calls, sum_calls))

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert switch.pipelined_decode_fused_calls == 0
    assert switch._decode_routed_fused_ready_cached is False
    assert switch.routed_weighted_sum_calls == 1
    assert pair_calls == []
    assert [codec for codec, _rhs, _s in gather_calls] == ["iq2_xxs", "q2_k"]


# --- DS4 decode lookahead port ------------------------------------------------


class _FakeDs4ScoreGateModule(nn.Module):
    """DS4 score-gate shape for lookahead wiring tests: fp16 router weight
    plus a per-expert selection bias, hash off."""

    def __init__(self, *, n_experts, hidden):
        super().__init__()
        self.hash = False
        self.weight = mx.random.normal((n_experts, hidden)).astype(mx.float16)
        self.bias = mx.array(
            np.linspace(-2.0, 2.0, n_experts).astype(np.float32))


class _FakeDs4HashGateModule(nn.Module):
    """DS4 hash-gate shape: routing comes from a token-id table, so the
    lookahead installer must skip it as a target."""

    def __init__(self, *, n_experts, hidden):
        super().__init__()
        self.hash = True
        self.weight = mx.random.normal((n_experts, hidden)).astype(mx.float16)


def _lookahead_chain_model(tmp_path, gates):
    """A model whose layers carry pooled switches and the given gates."""
    layers = []
    for i, gate in enumerate(gates):
        sub = tmp_path / f"la{i}"
        sub.mkdir()
        switch = _kquant_switch(sub, capacity=3, n_experts=4)
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.switch_mlp = switch
        layer.mlp.gate = gate
        layers.append(layer)
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = layers
    return model


def test_install_lookahead_skips_hash_targets_and_wires_bias(tmp_path):
    from moespresso.runtime.ssd_streaming_build import install_lookahead

    gates = [
        _FakeDs4HashGateModule(n_experts=4, hidden=256),
        _FakeDs4HashGateModule(n_experts=4, hidden=256),
        _FakeDs4ScoreGateModule(n_experts=4, hidden=256),
    ]
    model = _lookahead_chain_model(tmp_path, gates)
    wired = install_lookahead(model, 1)
    layers = model.model.layers
    sw0 = layers[0].mlp.switch_mlp
    sw1 = layers[1].mlp.switch_mlp
    sw2 = layers[2].mlp.switch_mlp
    # layer 0 targets layer 1, a hash gate: skipped, nothing stored.
    assert wired == 1
    assert sw0.lookahead_w is None and sw0.lookahead_b is None
    # layer 1 targets layer 2, score-routed: weight and bias stored.
    assert sw1.lookahead_w is not None
    assert sw1.lookahead_w.dtype == mx.float16
    assert sw1.lookahead_b is not None
    assert sw1.lookahead_b.dtype == mx.float32
    np.testing.assert_array_equal(
        np.array(sw1.lookahead_b), np.array(gates[2].bias))
    assert sw1.lookahead_target is sw2
    # the last layer has no target.
    assert sw2.lookahead_w is None


def test_install_lookahead_reads_wrapped_router_weight(tmp_path):
    from moespresso.runtime.qwen.router_gemv import BF16F32RouterLinear
    from moespresso.runtime.ssd_streaming_build import install_lookahead

    source = _FakeDs4ScoreGateModule(n_experts=4, hidden=256)
    inner = nn.Linear(2048, 256, bias=False)
    inner.weight = mx.zeros((256, 2048), dtype=mx.float32)
    inner.eval()
    wrapped = BF16F32RouterLinear(inner, inner.weight.astype(mx.bfloat16))
    wrapped.eval()
    model = _lookahead_chain_model(tmp_path, [source, wrapped])

    assert install_lookahead(model, 1) == 1
    layers = model.model.layers
    switch = layers[0].mlp.switch_mlp
    mx.eval(switch.lookahead_w)
    assert switch.lookahead_w.dtype == mx.float16
    assert bool(mx.array_equal(switch.lookahead_w, inner.weight.astype(mx.float16)))
    assert switch.lookahead_b is None
    assert switch.lookahead_target is layers[1].mlp.switch_mlp


def test_deepseek_v4_lookahead_export_uses_bias_aware_scoring(
        tmp_path, monkeypatch):
    """The DS4 block's prediction export ranks candidates with the target
    gate's scoring form: the monotone softplus transform plus the
    per-expert bias. A bias that promotes low-logit
    experts must land them in the exported top-16."""
    import sys
    import types

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_DECODE_ROUTED_FUSED", True)
    monkeypatch.setattr(psg, "_DECODE_RING_FUSED", True)
    monkeypatch.setattr(psg, "_RING_DECODE", True)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])
    monkeypatch.setattr(psg, "_RING_SEQ", [0])
    fake_gate_mod = types.SimpleNamespace(
        gate=lambda x, token, seq: x,
        signal_event=lambda seq: None,
        signaled_value=lambda: 0,
    )
    monkeypatch.setattr(psg, "_GATE_MOD", [fake_gate_mod])

    model, block = _full_resident_kquant_ds4_block(
        tmp_path, capacity=3, top_k=2)
    switch = block.switch_mlp
    monkeypatch.setattr(
        switch, "export_inds",
        lambda inds, seq: mx.zeros((1,), dtype=mx.uint32))

    def _fake_ring_install(seq, K, gate_mod=None):
        switch.publish_slots(mx.array([1, 2], dtype=mx.uint32))

    monkeypatch.setattr(switch, "ring_install", _fake_ring_install)
    gather_calls, pair_calls, sum_calls = [], [], []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        _fake_kquant_decode_fused_module(gather_calls, pair_calls, sum_calls))

    # Target router: logits strictly increase with the expert id (raw
    # ranking selects 48..63) and stay small enough that softplus never
    # saturates; the bias promotes experts 0..7 far past any logit gap, so
    # the bias-aware top-16 is {0..7, 56..63}.
    n_target = 64
    lookahead_w = (
        mx.arange(n_target)[:, None].astype(mx.float16) * 1e-4
        * mx.ones((1, 256), dtype=mx.float16))
    bias = np.zeros(n_target, dtype=np.float32)
    bias[:8] = 100.0
    switch.lookahead_w = lookahead_w
    switch.lookahead_b = mx.array(bias)
    mx.eval(switch.lookahead_w, switch.lookahead_b)

    exported = []

    def _spy_export_pred(ids, seq):
        exported.append(np.array(ids).reshape(-1).tolist())
        return mx.zeros((1,), dtype=mx.uint32)

    monkeypatch.setattr(switch, "export_pred", _spy_export_pred)

    x = mx.ones((1, 1, 256)).astype(mx.float16)
    y = block(x, input_ids=mx.array([[1]], dtype=mx.int32))
    mx.eval(y)

    assert len(exported) == 1
    got = {int(e) for e in exported[0]}
    assert got == set(range(8)) | set(range(56, 64))
    # The routed block still served the fused ring path.
    assert switch.pipelined_decode_fused_calls == 1


def test_lookahead_prefetch_pools_dedup_combined_and_forced_miss(
        tmp_path, monkeypatch):
    """On a combined K-quant gate/up switch the spare placement runs over
    the two unique physical pools (one combined-row pread per placement
    instead of two), and a predicted expert placed in a spare slot becomes
    a demand hit: the demand ensure takes no pread and counts no miss."""
    from moespresso.runtime import expert_slot_pool as esp

    pkg, _expected = _kquant_package(tmp_path, n_experts=8)
    model = _Model(hidden=256, intermediate=256, n_experts=8, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=4,
        seed=42, spare_slots=2)
    switch = model.language_model.model.layers[0].mlp.switch_mlp

    real_trio = esp.place_spare_trio
    seen_pool_counts = []

    def _spy_trio(pools, expert, spare_index):
        seen_pool_counts.append(len(tuple(pools)))
        return real_trio(pools, expert, spare_index)

    monkeypatch.setattr(esp, "place_spare_trio", _spy_trio)

    switch._prefetch_pools([5])
    assert switch.lookahead_errors == 0
    assert seen_pool_counts == [2]  # combined gate/up + down, deduplicated
    assert switch.lookahead_prefetch_loads == 1
    gate_pool = switch.gate_proj.pool
    down_pool = switch.down_proj.pool
    assert gate_pool.slot_of(5) == gate_pool.capacity  # first spare slot
    assert down_pool.slot_of(5) == down_pool.capacity

    stats = ssd_streaming_stats(model)
    assert stats["lookahead_prefetch_loads"] == 1
    assert stats["expert_spec_prefetch_loads"] == 2
    rows = ssd_streaming_layer_stats(model)
    assert rows[0]["lookahead_prefetch_loads"] == 1

    # Forced miss becomes a hit: no pread, no miss count on the demand path.
    preads = {"n": 0}
    real_pread = esp.pread_view_cached

    def _count_pread(*args, **kwargs):
        preads["n"] += 1
        return real_pread(*args, **kwargs)

    monkeypatch.setattr(esp, "pread_view_cached", _count_pread)
    misses_before = gate_pool.total_misses
    hits_before = gate_pool.total_hits
    gate_pool.ensure([5])
    assert gate_pool.total_misses == misses_before
    assert gate_pool.total_hits == hits_before + 1
    assert preads["n"] == 0


def test_lookahead_spare_contention_with_demand_ensures(tmp_path):
    """Concurrent demand ensures and spare placements on the same combined
    pools stay consistent: bookkeeping reconciles slot-for-slot, no double
    residency, reservations drain, and the speculative path records no
    errors."""
    import concurrent.futures

    pkg, _expected = _kquant_package(tmp_path, n_experts=12)
    model = _Model(hidden=256, intermediate=256, n_experts=12, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=6,
        seed=42, spare_slots=2)
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    rng = np.random.default_rng(7)
    demand_sets = [
        {int(e) for e in rng.integers(0, 12, 4)} for _ in range(150)
    ]
    predictions = [
        [int(e) for e in rng.integers(0, 12, 3)] for _ in range(150)
    ]

    def _demand():
        pools = switch._projection_pools()
        for active in demand_sets:
            for pool in pools:
                pool.ensure(active)

    def _speculate():
        for pred in predictions:
            switch._prefetch_pools(pred)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(_demand), ex.submit(_speculate)]
        for future in futures:
            future.result()

    assert switch.lookahead_errors == 0
    for pool in switch._projection_pools():
        for expert, slot in pool._slot_of.items():
            assert pool._expert_at[slot] == expert
        resident = pool.resident_ids()
        assert len(resident) <= pool.capacity + pool.spare_slots
        assert pool._prefetch_reserved == set()
        assert pool._prefetch_inflight == 0


def test_lookahead_submissions_shed_load_when_executor_is_busy(
        tmp_path, monkeypatch):
    """Speculation never queues behind itself: while the lookahead executor
    holds the in-flight cap, further submissions are dropped and counted,
    and the pending count drains back to zero when the tasks finish."""
    import threading

    import moespresso.runtime.pooled_switchglu as psg

    switch = _kquant_switch(tmp_path, capacity=3, n_experts=4)
    switch.lookahead_w = mx.zeros((4, 256), dtype=mx.float16)
    switch.lookahead_target = switch

    release = threading.Event()
    started = threading.Event()

    def _slow_task(seq):
        started.set()
        release.wait(5.0)
        with psg._LOOKAHEAD_PENDING_LOCK:
            psg._LOOKAHEAD_PENDING[0] -= 1

    monkeypatch.setattr(switch, "_lookahead_task", _slow_task)
    assert psg._LOOKAHEAD_PENDING[0] == 0
    for seq in range(6):
        switch._maybe_lookahead(seq)
    started.wait(5.0)
    # The cap admits _LOOKAHEAD_MAX_PENDING tasks; the rest dropped.
    assert psg._LOOKAHEAD_PENDING[0] == psg._LOOKAHEAD_MAX_PENDING
    assert switch.lookahead_dropped == 6 - psg._LOOKAHEAD_MAX_PENDING
    release.set()
    psg._lookahead_executor().submit(lambda: None).result()
    assert psg._LOOKAHEAD_PENDING[0] == 0
    # With the executor idle again, a fresh submission is admitted.
    switch._maybe_lookahead(99)
    psg._lookahead_executor().submit(lambda: None).result()
    assert psg._LOOKAHEAD_PENDING[0] == 0
    assert switch.lookahead_dropped == 6 - psg._LOOKAHEAD_MAX_PENDING


def test_deepseek_v4_lookahead_install_gating(tmp_path, monkeypatch):
    """The DS4 build wires the lookahead only when explicitly requested,
    the native-gate path is live, and residency is bounded; spares come
    out of the same capacity budget."""
    import types

    from mlx_lm.models.switch_layers import SwitchGLU

    import moespresso.runtime.pooled_switchglu as psg
    from moespresso.runtime.deepseek_v4.model import (
        _install_deepseek_v4_pooled_bundles,
    )

    monkeypatch.setattr(psg, "_RING_DECODE", True)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    fake_gate_mod = types.SimpleNamespace(signaled_value=lambda: 0)
    monkeypatch.setattr(psg, "_GATE_MOD", [fake_gate_mod])
    monkeypatch.setenv("MOESPRESSO_SSD_LOOKAHEAD", "1")

    def _build_model(n_experts):
        return _DeepseekV4SparseModel(
            gate=_DeepseekV4HashGate(n_experts=n_experts, top_k=2),
            switch_mlp=SwitchGLU(256, 256, n_experts),
            shared_experts=_TinySharedMLP(hidden=256, intermediate=32),
        )

    # Full residency: refused, no spares carved, capacity untouched.
    d1 = tmp_path / "full"
    d1.mkdir()
    pkg, _ = _kquant_package(d1, n_experts=4)
    model = _build_model(4)
    _install_deepseek_v4_pooled_bundles(
        model, pkg, build_expert_index(pkg), seed=42, capacity_per_layer=4)
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    assert switch.gate_proj.pool.capacity == 4
    assert switch.gate_proj.pool.spare_slots == 0
    assert getattr(model, "_moespresso_ssd_lookahead", None) is None

    # Bounded residency: spares carved out of the same budget.
    d2 = tmp_path / "bounded"
    d2.mkdir()
    pkg2, _ = _kquant_package(d2, n_experts=32)
    model2 = _build_model(32)
    _install_deepseek_v4_pooled_bundles(
        model2, pkg2, build_expert_index(pkg2), seed=42,
        capacity_per_layer=28)
    switch2 = model2.language_model.model.layers[0].mlp.switch_mlp
    assert switch2.gate_proj.pool.capacity == 24
    assert switch2.gate_proj.pool.spare_slots == 4
    assert model2._moespresso_ssd_lookahead == {
        "delta": 1, "wired": 0, "spare_slots": 4}

    # Native gate not live: refused, nothing carved.
    monkeypatch.setattr(psg, "_GATE_MOD", [False])
    d3 = tmp_path / "nogate"
    d3.mkdir()
    pkg3, _ = _kquant_package(d3, n_experts=32)
    model3 = _build_model(32)
    _install_deepseek_v4_pooled_bundles(
        model3, pkg3, build_expert_index(pkg3), seed=42,
        capacity_per_layer=28)
    switch3 = model3.language_model.model.layers[0].mlp.switch_mlp
    assert switch3.gate_proj.pool.capacity == 28
    assert switch3.gate_proj.pool.spare_slots == 0
    assert getattr(model3, "_moespresso_ssd_lookahead", None) is None


# --- Qwen sparse MoE block barrier-free decode -------------------------------


def _full_resident_kquant_qwen_block(
        tmp_path, *, capacity=4, top_k=4, non_pool=None):
    """Install a K-quant pooled Qwen sparse MoE block; prewarm at full capacity.

    Mirrors the DS4 helper but wraps the Qwen-style block (softmax router,
    sigmoid-gated shared expert) that PooledSparseMoeBlock owns. `non_pool`
    optionally supplies the resident router gate, shared expert, and shared
    expert gate so two arms can share identical non-pool weights (only the
    pooled routed math then differs between them)."""
    from moespresso.runtime.ssd_streaming_build import seed_all_expert_residency

    pkg, _expected = _kquant_package(tmp_path)
    model = _SparseModel(hidden=256, intermediate=256, n_experts=4)
    layer = model.language_model.model.layers[0]
    layer.mlp.top_k = top_k
    if non_pool is not None:
        layer.mlp.gate = non_pool["gate"]
        layer.mlp.shared_expert = non_pool["shared_expert"]
        layer.mlp.shared_expert_gate = non_pool["shared_expert_gate"]
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=capacity,
        seed=42)
    if capacity >= 4:
        seed_all_expert_residency(model)
    block = model.language_model.model.layers[0].mlp
    block.eval()
    return model, block


def test_qwen_decode_barrier_free_when_full_resident(tmp_path, monkeypatch):
    """At full residency the Qwen decode block consumes router ids on device:
    no ring export, no worker submit, no per-layer kick, no host index read; the
    flush knob is the only in-block commit site."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_QWEN_DECODE_SCHED", True)
    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_DECODE_FLUSH_LAYERS", 1)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    monkeypatch.setattr(psg, "_GATE_MOD", [False])
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])

    _model, block = _full_resident_kquant_qwen_block(tmp_path, capacity=4)
    switch = block.switch_mlp
    gather_calls = []
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _fake_kquant_decode_module(gather_calls))
    hits_before = switch.gate_proj.pool.total_hits

    real_asarray = np.asarray
    mlx_asarray_hits = []

    def _spy_asarray(obj, *args, **kwargs):
        if isinstance(obj, mx.array):
            mlx_asarray_hits.append(obj.shape)
        return real_asarray(obj, *args, **kwargs)

    monkeypatch.setattr(psg.np, "asarray", _spy_asarray)

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x)
    mx.eval(y)
    monkeypatch.setattr(psg.np, "asarray", real_asarray)

    assert y.shape == (1, 1, 256)
    assert mlx_asarray_hits == []
    assert switch.barrier_free_decode_calls == 1
    assert switch.barrier_free_decode_flush_calls == 1  # depth 1, layer 0
    # The ring/native-gate machinery never ran: the off-arm counters.
    assert switch.pipelined_layers == 0
    assert switch.block_exit_kick_calls == 0
    assert switch.router_export_seconds == 0.0
    assert switch.decode_moe_block_calls == 1
    assert [codec for codec, _rhs, _s in gather_calls] == ["iq2_xxs", "q2_k"]
    for _codec, rhs, sorted_indices in gather_calls:
        assert sorted_indices is False
        assert isinstance(rhs, mx.array)
        assert rhs.dtype == mx.uint32
    # LFU touch accounting is skipped: full-resident pools never evict.
    assert switch.gate_proj.pool.total_hits == hits_before

    # Flush cadence: at depth 4 the layer-0 block queues without a kick.
    monkeypatch.setattr(psg, "_DECODE_FLUSH_LAYERS", 4)
    y = block(x)
    mx.eval(y)
    assert switch.barrier_free_decode_calls == 2
    assert switch.barrier_free_decode_flush_calls == 1


def test_qwen_decode_barrier_free_matches_ring_and_kill_switch(
        tmp_path, monkeypatch):
    """The Qwen barrier-free decode route selects exactly the slots the ring
    path publishes (value-bearing fake kernel), and the kill switch
    (MOESPRESSO_SSD_DECODE_SCHED=0, the module constant) keeps the ring path
    with its counters."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_RING_DECODE", True)
    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    monkeypatch.setattr(psg, "_GATE_MOD", [False])
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _value_kquant_decode_module())

    mx.random.seed(11)
    xs = [mx.random.normal((1, 1, 256)).astype(mx.float16) for _ in range(4)]
    mx.eval(*xs)
    # Shared non-pool weights so both arms route identically and add the same
    # shared expert; only the pooled routed dispatch (barrier-free vs ring)
    # differs between them.
    non_pool = {
        "gate": nn.Linear(256, 4, bias=False),
        "shared_expert": _TinySharedMLP(hidden=256, intermediate=32),
        "shared_expert_gate": nn.Linear(256, 1, bias=False),
    }
    mx.eval(non_pool["gate"].parameters(),
            non_pool["shared_expert"].parameters(),
            non_pool["shared_expert_gate"].parameters())
    outs = {}
    switches = {}

    for sched_on in (False, True):
        monkeypatch.setattr(psg, "_QWEN_DECODE_SCHED", sched_on)
        root = tmp_path / ("bf" if sched_on else "ring")
        root.mkdir()
        _model, block = _full_resident_kquant_qwen_block(
            root, capacity=4, non_pool=non_pool)
        got = []
        for x in xs:
            y = block(x)
            mx.eval(y)
            got.append(np.array(y))
        outs[sched_on] = got
        switches[sched_on] = block.switch_mlp

    for ring_y, bf_y in zip(outs[False], outs[True]):
        np.testing.assert_array_equal(ring_y, bf_y)
    assert switches[True].barrier_free_decode_calls == len(xs)
    assert switches[True].pipelined_layers == 0
    assert switches[True].block_exit_kick_calls == 0
    assert switches[False].barrier_free_decode_calls == 0
    # The kill switch short-circuits the branch before the certificate check,
    # so the OFF arm never evaluates the certificate (stays None) and takes the
    # ring path for every token.
    assert switches[False]._barrier_free_decode_ready_cached is None
    assert switches[False].pipelined_layers == len(xs)
    assert switches[False].block_exit_kick_calls == len(xs)


def test_qwen_decode_barrier_free_fails_closed_on_partial_residency(
        tmp_path, monkeypatch):
    """capacity < num_experts fails the decode certificate closed: the Qwen
    decode keeps the ring path (the product path for partial residency)."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_QWEN_DECODE_SCHED", True)
    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_RING_DECODE", True)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    monkeypatch.setattr(psg, "_GATE_MOD", [False])
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _value_kquant_decode_module())

    _model, block = _full_resident_kquant_qwen_block(
        tmp_path, capacity=3, top_k=2)
    switch = block.switch_mlp

    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    y = block(x)
    mx.eval(y)

    assert switch.barrier_free_decode_calls == 0
    assert switch._barrier_free_decode_ready_cached is False
    assert switch.pipelined_layers == 1
    assert switch.block_exit_kick_calls == 1


def test_qwen_decode_barrier_free_forced_miss_serves_rail_identical(
        tmp_path, monkeypatch):
    """Deliberately evicting an expert drops the pool below full residency, so
    the certificate fails closed and the token routes through the ring path,
    which reloads the evicted expert and produces the same routed output the
    full-resident barrier-free decode produced over the same token (the
    value-fake kernel is shared and the reloaded bytes are the same expert, so
    equal slot selection means equal math)."""
    import sys

    import moespresso.runtime.pooled_switchglu as psg

    monkeypatch.setattr(psg, "_QWEN_DECODE_SCHED", True)
    monkeypatch.setattr(psg, "_RING_DECODE", True)
    monkeypatch.setattr(psg, "_BARRIER_FREE_DECODE", True)
    monkeypatch.setattr(psg, "_RING_SELF_TEST", [True])
    monkeypatch.setattr(psg, "_GATE_MOD", [False])
    monkeypatch.setattr(psg, "_PIPE_PREV", [])
    monkeypatch.setattr(psg, "_GATE_PENDING", [])
    monkeypatch.setitem(
        sys.modules, "mlx_kquant", _value_kquant_decode_module())

    mx.random.seed(7)
    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    mx.eval(x)

    _model, block = _full_resident_kquant_qwen_block(tmp_path, capacity=4)
    switch = block.switch_mlp

    # Full-resident barrier-free reference over the token.
    ref = np.array(block(x))
    mx.eval(mx.array(ref))
    assert switch.barrier_free_decode_calls == 1
    assert switch.pipelined_layers == 0

    # Deliberately evict one expert from every projection pool, invalidate the
    # cached certificate, and re-decode. The certificate re-checks and fails
    # closed (a slot is now free), so the token routes through the ring path,
    # which reloads the evicted expert on the miss before the routed matmul.
    loads_before = int(switch.gate_proj.pool.total_loads)
    seen_pools = set()
    for proj in ("gate_proj", "up_proj", "down_proj"):
        pool = getattr(switch, proj).pool
        if id(pool) in seen_pools:  # gate/up share one pool
            continue
        seen_pools.add(id(pool))
        evicted = pool._expert_at[3]
        del pool._slot_of[evicted]
        pool._expert_at[3] = None
        pool._slot_table_dirty = True
    switch._barrier_free_decode_ready_cached = None

    y = np.array(block(x))
    mx.eval(mx.array(y))

    assert switch.barrier_free_decode_calls == 1  # no new barrier-free call
    assert switch.pipelined_layers == 1  # this token took the ring path
    assert int(switch.gate_proj.pool.total_loads) > loads_before  # miss served
    # Rail identity: the reloaded expert lands back in its slot, so the ring
    # path selects the same slots the barrier-free reference did and the shared
    # value-fake kernel returns the same routed output.
    np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)


# --- K-quant dense install on the streaming build path -----------------------


def _dense_kquant_manifest(*, codec="q4_k", kind="affine"):
    """A manifest with one K-quant dense tensor plus expert/affine controls."""
    return {
        "tensors": [
            {
                "source_name": "model.language_model.embed_tokens.weight",
                "format": "kquant",
                "kind": kind,
                "format_params": {"kquant_codec": codec},
                "module_weight_key": "language_model.model.embed_tokens.weight",
            },
            {
                # a routed expert bundle is skipped by the dense codec map
                "source_name": "layers.0.mlp.experts",
                "format": "tq",
                "kind": "expert",
            },
        ]
    }


def _affine_dense_manifest():
    return {
        "tensors": [
            {
                "source_name": "model.language_model.embed_tokens.weight",
                "format": "affine",
                "kind": "affine",
            },
            {
                "source_name": "layers.0.mlp.experts",
                "format": "tq",
                "kind": "expert",
            },
        ]
    }


def test_maybe_install_kquant_dense_fires_for_kquant_dense_manifest():
    from moespresso.runtime import ssd_streaming_build as ssb

    calls = []

    def fake_installer(model, codec_map):
        calls.append(codec_map)
        return len(codec_map)

    model = _Model(n_layers=1)
    manifest = _dense_kquant_manifest(codec="q4_k")
    # inject the mlx-kquant installer so the test needs no built backend
    import moespresso.runtime.kquant_install as ki
    original = ki.install_manifest_kquant_modules

    def patched(model_arg, manifest_arg):
        return original(model_arg, manifest_arg, installer=fake_installer)

    ssb_install = ssb._maybe_install_kquant_dense
    ki_saved = ki.install_manifest_kquant_modules
    ki.install_manifest_kquant_modules = patched
    try:
        installed = ssb_install(model, manifest)
    finally:
        ki.install_manifest_kquant_modules = ki_saved

    assert installed == 1
    assert calls == [
        {"language_model.model.embed_tokens.weight": "q4_k"}]


def test_maybe_install_kquant_dense_no_op_for_affine_dense():
    from moespresso.runtime import ssd_streaming_build as ssb

    # An affine-dense manifest must not import mlx_kquant or install anything.
    import builtins

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "mlx_kquant" or name.startswith("mlx_kquant."):
            raise AssertionError("affine-dense must not import mlx_kquant")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guard
    try:
        installed = ssb._maybe_install_kquant_dense(
            _Model(n_layers=1), _affine_dense_manifest())
    finally:
        builtins.__import__ = real_import

    assert installed == 0


def test_maybe_install_kquant_dense_none_manifest_is_no_op():
    from moespresso.runtime import ssd_streaming_build as ssb

    assert ssb._maybe_install_kquant_dense(_Model(n_layers=1), None) == 0


def test_maybe_install_kquant_dense_fails_closed_on_unknown_codec():
    from moespresso.runtime import ssd_streaming_build as ssb
    from moespresso.runtime.kquant_install import KQuantInstallError

    manifest = _dense_kquant_manifest(codec="q2_not_real")
    with pytest.raises(KQuantInstallError, match="unknown K-quant codec"):
        ssb._maybe_install_kquant_dense(_Model(n_layers=1), manifest)


def test_maybe_install_kquant_dense_kill_switch_refuses(monkeypatch):
    from moespresso.runtime import ssd_streaming_build as ssb

    monkeypatch.setenv("MOESPRESSO_QWEN_STREAMING_KQUANT_DENSE", "0")
    # With the kill switch off the install is skipped even though the manifest
    # carries K-quant dense: the diagnostic that restores the refuse/crash path.
    import builtins

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "mlx_kquant" or name.startswith("mlx_kquant."):
            raise AssertionError("kill switch off must not import mlx_kquant")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guard
    try:
        installed = ssb._maybe_install_kquant_dense(
            _Model(n_layers=1), _dense_kquant_manifest(codec="q4_k"))
    finally:
        builtins.__import__ = real_import

    assert installed == 0


def test_read_manifest_absent_and_malformed(tmp_path):
    from moespresso.runtime.ssd_streaming_build import _read_manifest

    assert _read_manifest(tmp_path) is None
    (tmp_path / "package_manifest.json").write_text("{not json")
    assert _read_manifest(tmp_path) is None


def test_ssd_streaming_stats_reports_kquant_dense_install_count(tmp_path):
    pkg = _package(tmp_path, layers=(0,))
    model = _Model(n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=4)

    # No install recorded yet -> 0.
    assert ssd_streaming_stats(model)["kquant_dense_modules_installed"] == 0

    object.__setattr__(model, "_moespresso_ssd_kquant_dense_installed", 5)
    assert ssd_streaming_stats(model)["kquant_dense_modules_installed"] == 5


def test_kquant_embedding_output_matches_dequantize_reference():
    """A tiny synthetic K-quant embedding through KQuantEmbedding must match a
    kq.dequantize of the same gathered wire-byte rows. The streaming embedding
    path decodes only those rows."""
    kq = pytest.importorskip("mlx_kquant")
    if not kq.metallib_loads():
        pytest.skip("mlx-kquant metal library not loaded")
    from moespresso.package.kquant_backend import encode_kquant_weight
    from moespresso.package.qwen.recipe import QwenKQuantDenseTarget
    from mlx_kquant.nn import KQuantEmbedding

    num_embeddings, dims, codec = 32, 256, "q8_0"
    rng = np.random.default_rng(11)
    table = rng.standard_normal((num_embeddings, dims), dtype=np.float32)
    target = QwenKQuantDenseTarget(
        source_name="model.language_model.embed_tokens.weight",
        role="embed_tokens",
        layer_index=None,
        codec=codec,
        gguf_tensor="token_embd.weight",
        imatrix_key="token_embd.weight",
        module_path="language_model.model.embed_tokens",
        requires_imatrix=False,
        module_weight_key="language_model.model.embed_tokens.weight",
    )
    encoded = encode_kquant_weight(table, target, {}, stream="cpu")

    # kq.dequantize emits float32, so request float32 output for an exact
    # comparison; the serving path uses the module's own default dtype.
    emb = KQuantEmbedding(num_embeddings, dims, codec, out_dtype=mx.float32)
    emb.weight = mx.array(encoded.weight)
    emb.scales = mx.array(encoded.scales)
    mx.eval(emb.weight, emb.scales)

    ids = mx.array([[3, 7, 30], [0, 1, 2]], dtype=mx.uint32)
    got = emb(ids)
    mx.eval(got)

    rows = mx.array(encoded.weight)[ids].reshape(-1, encoded.weight.shape[-1])
    ref = kq.dequantize(rows, mx.array(encoded.scales), codec).reshape(
        2, 3, dims)
    mx.eval(ref)

    np.testing.assert_array_equal(np.array(got), np.array(ref).astype(
        np.array(got).dtype))


# ---- unified sorted prefill across residency modes --------------------------
#
# The full-resident barrier-free prefill and the partial-residency chunked
# prefill must compute the routed MoE with the same fused sorted kernels, so
# the served tokens are identical at any capacity. These tests use the combined
# K-quant fixture (iq2_xxs gate/up, q2_k down), the shape the real package
# serves. Prefill carries no cross-row reduction, so a full-capacity run (one
# barrier-free-shaped call over the whole sorted array) and a reduced-capacity
# run (many capacity-chunks) over the same prompt must agree bit for bit.


def _kquant_switch(tmp_path, *, capacity, n_experts=4, hidden=256,
                   intermediate=256):
    pkg, _expected = _kquant_package(
        tmp_path, n_experts=n_experts, hidden=hidden, intermediate=intermediate)
    model = _Model(hidden=hidden, intermediate=intermediate,
                   n_experts=n_experts, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index,
        capacity_per_layer=capacity, seed=42)
    return model.language_model.model.layers[0].mlp.switch_mlp


def _bulk_prefill_indices(n_experts, rows=32, top_k=4):
    # A sorted-prefill-shaped call (rows * top_k >= 64) whose active set spans
    # every expert, so a capacity below n_experts forces the chunked path.
    return mx.array(
        [[(t + j) % n_experts for j in range(top_k)] for t in range(rows)],
        dtype=mx.uint32,
    )


def test_unified_sorted_prefill_engages_at_partial_capacity(tmp_path):
    switch = _kquant_switch(tmp_path, n_experts=4, capacity=2)
    x = mx.random.normal((32, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(4, rows=32, top_k=4)
    assert indices.size >= 64
    assert len(set(np.array(indices).reshape(-1).tolist())) > 2  # over capacity

    mx.eval(switch(x, indices))
    # The unified fused sorted compute ran; the pre-unification segmented path
    # did not.
    assert switch.unified_sorted_prefill_calls > 0
    assert switch.segmented_prefill_calls == 0
    # The chunked driver still segments by capacity.
    assert switch.sorted_chunked_calls > 0
    assert switch.over_capacity_calls > 0


def test_unified_prefill_matches_full_capacity_bitexact(tmp_path):
    """A reduced-capacity prefill and a full-capacity prefill over the same
    prompt produce bit-identical routed output. Residency changes the byte source
    while preserving the mathematical grouping. The full-capacity arm is
    prewarmed so it takes the barrier-free fused sorted route the served
    product runs, which is the rail the partial arm must reproduce."""
    from moespresso.runtime.ssd_streaming_build import seed_all_expert_residency

    # >= 4096 routed pairs so both arms take the fused sorted route (the
    # barrier-free shape gate is _SEGMENTED_PREFILL_MIN_ROWS); this is the
    # regime the served 4K/37K prefill runs in. Below it both arms take the
    # unfused direct gather and the comparison would not exercise the unified
    # fused path against the barrier-free fused path.
    rows = 1100
    x = mx.random.normal((rows, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(4, rows=rows, top_k=4)
    assert indices.size >= 4096

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()

    pkg_full, _ = _kquant_package(full_dir, n_experts=4, hidden=256,
                                  intermediate=256)
    model_full = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    install_pooled_switchglus(
        model_full, package_dir=pkg_full, index=build_expert_index(pkg_full),
        capacity_per_layer=4, seed=42)
    seed_all_expert_residency(model_full)  # prewarm-all, as the product serves
    full = model_full.language_model.model.layers[0].mlp.switch_mlp
    out_full = full(x, indices)
    mx.eval(out_full)
    # Full capacity, prewarmed: the barrier-free fused sorted route runs.
    assert full.barrier_free_prefill_calls > 0

    partial = _kquant_switch(partial_dir, n_experts=4, capacity=2)
    out_partial = partial(x, indices)
    mx.eval(out_partial)
    assert partial.unified_sorted_prefill_calls > 0

    np.testing.assert_array_equal(np.array(out_full), np.array(out_partial))


def test_unified_prefill_kill_switch_falls_back_to_segmented(tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_UNIFIED_SORTED_PREFILL", False)
    switch = _kquant_switch(tmp_path, n_experts=4, capacity=2)
    x = mx.random.normal((32, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(4, rows=32, top_k=4)

    mx.eval(switch(x, indices))
    # With the kill switch off the unified compute never runs; the chunks fall
    # to the pre-unification segmented (or general) compute.
    assert switch.unified_sorted_prefill_calls == 0
    assert switch.sorted_chunked_calls > 0


def test_unified_prefill_eligibility_fails_closed_off_contract(tmp_path):
    switch = _kquant_switch(tmp_path, n_experts=4, capacity=2)
    assert switch._unified_sorted_ready() is True
    # Off-contract down codec: the fused sorted route requires a K-quant down
    # pool, so a non-K-quant down fails the check closed.
    switch.down_proj.codec = "mxfp4"
    assert switch._unified_sorted_ready() is False


def test_unified_sorted_prefill_calls_exported_in_stats(tmp_path):
    pkg, _expected = _kquant_package(tmp_path, n_experts=4)
    model = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=2, seed=42)
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    x = mx.random.normal((32, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(4, rows=32, top_k=4)
    mx.eval(switch(x, indices))

    stats = ssd_streaming_stats(model)
    assert "unified_sorted_prefill_calls" in stats
    assert stats["unified_sorted_prefill_calls"] == switch.unified_sorted_prefill_calls
    assert stats["unified_sorted_prefill_calls"] > 0
    rows = ssd_streaming_layer_stats(model)
    assert "unified_sorted_prefill_calls" in rows[0]


# --- Cross-chunk predictive expert prefetch (MOESPRESSO_SSD_PREFETCH) ---------
#
# These exercise the ticket lifecycle on the over-capacity sorted-chunked path.
# A capacity below n_experts and a bulk-prefill-shaped call whose active set
# spans every expert forces that path, so consecutive calls with the same
# routing submit and then consume a matching ticket.


def _prefetch_switch(tmp_path, *, n_experts=8, capacity=4, sub="pf"):
    d = tmp_path / sub
    d.mkdir()
    return _kquant_switch(d, n_experts=n_experts, capacity=capacity)


def test_prefetch_ticket_submits_then_a_matching_next_call_consumes_it(
        tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_PREFILL_PREFETCH", True)

    switch = _prefetch_switch(tmp_path, n_experts=8, capacity=4)
    x = mx.random.normal((32, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(8, rows=32, top_k=4)
    assert len(set(np.array(indices).reshape(-1).tolist())) > 4  # over capacity

    mx.eval(switch(x, indices))
    # First over-capacity call submits a ticket, has none to consume yet.
    assert switch.sorted_chunked_calls == 1
    assert switch.prefetch_ticket_submitted == 1
    assert switch.prefetch_ticket_consumed == 0
    assert switch.prefetch_ticket_experts == len(
        set(np.array(indices).reshape(-1).tolist()))

    mx.eval(switch(x, indices))
    # Same routing: the second call consumes the ticket and it matches exactly,
    # so no mismatch and no stale drain, and it submits its own next ticket.
    assert switch.prefetch_ticket_submitted == 2
    assert switch.prefetch_ticket_consumed == 1
    assert switch.prefetch_ticket_mismatched == 0
    assert switch.prefetch_ticket_stale == 0
    # The prefetch warmed at least one slot the second call would have missed.
    assert switch.prefetch_ticket_loaded > 0


def test_prefetch_ticket_submit_protects_final_chunk(tmp_path, monkeypatch):
    """The over-capacity overlap path submits its ticket with the final
    capacity-chunk's set as the explicit protect: that chunk's output is
    never waited before the submit, so its slots must not be prefetch
    victims. The loop drains every earlier chunk's readers, so the
    prefetch may only reclaim drained chunks."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_PREFILL_PREFETCH", True)

    switch = _prefetch_switch(tmp_path, n_experts=8, capacity=4)
    protects = []
    for pool in switch._projection_pools():
        real_prefetch = pool.prefetch

        def _spy(expert_ids, *, protect=None, reserve_floor=16,
                 _real=real_prefetch):
            protects.append(None if protect is None else set(protect))
            return _real(
                expert_ids, protect=protect, reserve_floor=reserve_floor)

        monkeypatch.setattr(pool, "prefetch", _spy)

    x = mx.random.normal((32, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(8, rows=32, top_k=4)
    assert len(set(np.array(indices).reshape(-1).tolist())) > 4
    mx.eval(switch(x, indices))
    switch._drain_stale_prefetch_ticket()  # quiesce the spied prefetches

    assert switch.prefetch_ticket_submitted == 1
    assert len(protects) == len(switch._projection_pools())
    # The final chunk's set is what the last ensure-ahead recorded as
    # _demand_protect; the submit must pass exactly that set explicitly.
    final_chunk = set(switch.gate_proj.pool._demand_protect)
    assert final_chunk
    for protect in protects:
        assert protect == final_chunk


def test_prefetch_ticket_mismatch_is_counted_but_still_serves(
        tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_PREFILL_PREFETCH", True)

    switch = _prefetch_switch(tmp_path, n_experts=8, capacity=4)
    x = mx.random.normal((32, 256)).astype(mx.float16)
    # Two over-capacity prompt chunks with different expert sets: the first
    # call routes to experts {0..5}, the second to {2..7}, so both exceed the
    # capacity-4 pool and the first call's predicted set does not equal the
    # second call's actual demand.
    lo = mx.array([[(t + j) % 6 for j in range(4)] for t in range(32)],
                  dtype=mx.uint32)
    hi = mx.array([[2 + (t + j) % 6 for j in range(4)] for t in range(32)],
                  dtype=mx.uint32)
    assert len(set(np.array(lo).reshape(-1).tolist())) > 4  # over capacity
    assert len(set(np.array(hi).reshape(-1).tolist())) > 4  # over capacity
    assert (set(np.array(lo).reshape(-1).tolist())
            != set(np.array(hi).reshape(-1).tolist()))

    y0 = switch(x, lo)
    mx.eval(y0)
    y1 = switch(x, hi)
    mx.eval(y1)

    assert switch.prefetch_ticket_consumed == 1
    assert switch.prefetch_ticket_mismatched == 1
    # A mismatch does not corrupt the output: the shape/token invariants hold
    # and the normal path serviced the difference.
    assert y1.shape == (32, 4, 256)


def test_prefetch_ticket_goes_stale_when_next_call_is_not_over_capacity(
        tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_PREFILL_PREFETCH", True)

    switch = _prefetch_switch(tmp_path, n_experts=8, capacity=4)
    x = mx.random.normal((32, 256)).astype(mx.float16)
    over = _bulk_prefill_indices(8, rows=32, top_k=4)  # over capacity, submits
    fits = mx.array([[j % 4 for j in range(t, t + 4)] for t in range(32)],
                    dtype=mx.uint32)  # <= 4 active, takes the direct path

    mx.eval(switch(x, over))
    assert switch.prefetch_ticket_submitted == 1

    mx.eval(switch(x, fits))
    # The next call did not reach the sorted-chunked consume point, so the
    # lingering ticket is drained as stale before the demand ensure runs.
    assert switch.prefetch_ticket_stale == 1
    assert switch.prefetch_ticket_consumed == 0
    assert switch._prefetch_ticket is None


def test_prefetch_kill_switch_off_never_submits_or_consumes(
        tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_PREFILL_PREFETCH", False)

    switch = _prefetch_switch(tmp_path, n_experts=8, capacity=4)
    x = mx.random.normal((32, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(8, rows=32, top_k=4)

    mx.eval(switch(x, indices))
    mx.eval(switch(x, indices))

    assert switch.sorted_chunked_calls == 2  # the over-capacity path still ran
    assert switch.prefetch_ticket_submitted == 0
    assert switch.prefetch_ticket_consumed == 0
    assert switch.prefetch_ticket_stale == 0
    assert switch._prefetch_ticket is None


def test_prefetch_never_engages_on_full_capacity_certificate_path(
        tmp_path, monkeypatch):
    """The full-capacity prewarmed build never dispatches over-capacity, so no
    ticket is ever submitted or consumed even with the switch on. The
    barrier-free certificate path must stay untouched."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_PREFILL_PREFETCH", True)
    # Force the barrier-free certificate route on (it ships gated off, measured
    # served-neutral) so the test exercises the real full-resident bulk path the
    # certificate serves in addition to a full-capacity direct call.
    monkeypatch.setattr(psg, "_BARRIER_FREE_PREFILL", True)
    monkeypatch.setattr(psg, "_SEGMENTED_PREFILL_MIN_ROWS", 8)

    switch = _full_resident_kquant_switch(tmp_path, capacity=4)  # n_experts==4
    x = mx.random.normal((32, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(4, rows=32, top_k=4)

    mx.eval(switch(x, indices))
    mx.eval(switch(x, indices))

    # The barrier-free certificate path ran, and it left before any over-capacity
    # dispatch, so the prefetch never engaged on this path.
    assert switch.barrier_free_prefill_calls > 0
    assert switch.sorted_chunked_calls == 0
    assert switch.over_capacity_calls == 0
    assert switch.prefetch_ticket_submitted == 0
    assert switch.prefetch_ticket_consumed == 0
    assert switch._prefetch_ticket is None


def test_prefetch_on_off_are_bit_identical(tmp_path):
    """The prefetch is a pure pre-fill of slots: the routed output over two
    prompt chunks is bit-identical with the switch on and off. This is the
    unit-level one-rail check; the served identity run covers the real model."""
    import moespresso.runtime.pooled_switchglu as psg

    rows = 1100  # >= 4096 routed pairs so the fused sorted route runs
    x = mx.random.normal((rows, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(8, rows=rows, top_k=4)
    assert indices.size >= 4096

    def _run(prefetch_on):
        import pytest as _pytest
        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(psg, "_PREFILL_PREFETCH", prefetch_on)
            sub = "on" if prefetch_on else "off"
            switch = _prefetch_switch(tmp_path, n_experts=8, capacity=4, sub=sub)
            out0 = switch(x, indices)
            mx.eval(out0)
            out1 = switch(x, indices)
            mx.eval(out1)
            return np.array(out0), np.array(out1), switch

    on0, on1, on_switch = _run(True)
    off0, off1, off_switch = _run(False)

    np.testing.assert_array_equal(on0, off0)
    np.testing.assert_array_equal(on1, off1)
    assert on_switch.prefetch_ticket_consumed == 1
    assert off_switch.prefetch_ticket_consumed == 0


def test_prefetch_partial_capacity_matches_full_capacity_bitexact(
        tmp_path, monkeypatch):
    """One-rail identity holds with the prefetch engaged: a cap-below-full
    prefill with prefetch on reproduces the full-capacity prewarmed rail
    bit-for-bit, mirroring test_unified_prefill_matches_full_capacity_bitexact
    with the prefetch switch forced on."""
    from moespresso.runtime.ssd_streaming_build import seed_all_expert_residency

    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_PREFILL_PREFETCH", True)

    rows = 1100
    x = mx.random.normal((rows, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(4, rows=rows, top_k=4)
    assert indices.size >= 4096

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    pkg_full, _ = _kquant_package(full_dir, n_experts=4, hidden=256,
                                  intermediate=256)
    model_full = _Model(hidden=256, intermediate=256, n_experts=4, n_layers=1)
    install_pooled_switchglus(
        model_full, package_dir=pkg_full, index=build_expert_index(pkg_full),
        capacity_per_layer=4, seed=42)
    seed_all_expert_residency(model_full)
    full = model_full.language_model.model.layers[0].mlp.switch_mlp
    out_full = full(x, indices)
    mx.eval(out_full)
    assert full.barrier_free_prefill_calls > 0

    partial = _prefetch_switch(tmp_path, n_experts=4, capacity=2, sub="partial")
    out0 = partial(x, indices)
    mx.eval(out0)
    out1 = partial(x, indices)  # second call consumes the ticket
    mx.eval(out1)
    assert partial.sorted_chunked_calls == 2
    assert partial.prefetch_ticket_consumed == 1

    np.testing.assert_array_equal(np.array(out_full), np.array(out0))
    np.testing.assert_array_equal(np.array(out_full), np.array(out1))


def test_prefetch_survives_contention_over_many_iterations(
        tmp_path, monkeypatch):
    """Shake the shared-pool race: alternate the routed set every call so the
    in-flight prefetch and the next call's demand ensure hit the same pool
    with drifting predictions, over enough iterations to surface a race. The
    output shape stays invariant and the ticket lifecycle stays consistent
    (never negative, consumed + stale <= submitted, one ticket at a time)."""
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_PREFILL_PREFETCH", True)

    switch = _prefetch_switch(tmp_path, n_experts=12, capacity=4)
    rng = np.random.default_rng(1234)
    x = mx.random.normal((32, 256)).astype(mx.float16)

    for it in range(120):
        # Every call routes to a different, mostly-over-capacity expert subset,
        # so the previous call's prediction is usually wrong and the demand
        # path must service the difference while the prefetch is still landing.
        base = rng.integers(0, 12, (32, 4)).astype(np.uint32)
        indices = mx.array(base)
        y = switch(x, indices)
        mx.eval(y)
        assert y.shape == (32, 4, 256)
        # Never more than one ticket in flight, and the counters stay coherent.
        consumed_or_stale = (
            switch.prefetch_ticket_consumed + switch.prefetch_ticket_stale)
        assert consumed_or_stale <= switch.prefetch_ticket_submitted
        assert switch.prefetch_ticket_mismatched <= switch.prefetch_ticket_consumed

    # Drain the trailing ticket so the pool is quiesced, then the counters
    # reconcile: every submitted ticket was eventually consumed or drained.
    fits = mx.array([[j % 4 for j in range(t, t + 4)] for t in range(32)],
                    dtype=mx.uint32)
    mx.eval(switch(x, fits))
    assert switch._prefetch_ticket is None
    assert (switch.prefetch_ticket_consumed + switch.prefetch_ticket_stale
            == switch.prefetch_ticket_submitted)


def test_prefetch_counters_exported_in_stats(tmp_path, monkeypatch):
    import moespresso.runtime.pooled_switchglu as psg
    monkeypatch.setattr(psg, "_PREFILL_PREFETCH", True)

    pkg, _expected = _kquant_package(tmp_path, n_experts=8)
    model = _Model(hidden=256, intermediate=256, n_experts=8, n_layers=1)
    index = build_expert_index(pkg)
    install_pooled_switchglus(
        model, package_dir=pkg, index=index, capacity_per_layer=4, seed=42)
    switch = model.language_model.model.layers[0].mlp.switch_mlp
    x = mx.random.normal((32, 256)).astype(mx.float16)
    indices = _bulk_prefill_indices(8, rows=32, top_k=4)
    mx.eval(switch(x, indices))
    mx.eval(switch(x, indices))

    stats = ssd_streaming_stats(model)
    for key in (
        "prefetch_ticket_submitted",
        "prefetch_ticket_consumed",
        "prefetch_ticket_mismatched",
        "prefetch_ticket_stale",
        "prefetch_ticket_experts",
        "prefetch_ticket_loaded",
        "prefetch_ticket_wait_seconds",
    ):
        assert key in stats
    assert stats["prefetch_ticket_submitted"] == switch.prefetch_ticket_submitted
    assert stats["prefetch_ticket_consumed"] == 1
    rows = ssd_streaming_layer_stats(model)
    assert "prefetch_ticket_consumed" in rows[0]
