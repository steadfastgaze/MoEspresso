"""Runtime side of dense-tensor IQ_K serving.

Every route is driven through an injected backend builder that mirrors the
kernel module's operand contract (including the pairs-per-activation-row
mapping), so the suite passes with the dense kernel members absent and the
seam moves to the real kernels without changing a test. Width classes are
the production graph's own reduction widths (1024, 2048, 4096, 8192); out
widths are scaled where the real tensor would be gigabytes, which changes
no operand mapping.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402

import moespresso.runtime.deepseek_v4.iqk_dense as iqk_dense  # noqa: E402
from moespresso.package.iqk_format import (  # noqa: E402
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUT_LEGACY_RELAYOUT,
)
from moespresso.runtime.deepseek_v4.iqk_dense import (  # noqa: E402
    IqkDenseInstallError,
    install_deepseek_v4_iqk_dense_modules,
    install_deepseek_v4_iqk_dense_seams,
    iqk_dense_matmul_call_counts,
    iqk_dense_weight_map_from_manifest,
    manifest_requires_iqk_dense,
    patch_deepseek_v4_iqk_dense_lm_head,
)

# --------------------------------------------------------------------------
# The stub backend: the kernel module's operand contract, reference math


def _ref_weights(stacks: int, out_features: int, in_features: int) -> np.ndarray:
    """Deterministic reference weights derived from the geometry alone.

    The builder and the test reference must agree on the decoded weights
    without a real codec, so both derive them from the same seeded shape.
    """
    rng = np.random.default_rng(stacks * 1_000_003 + out_features * 101
                                + in_features)
    return rng.standard_normal(
        (stacks, out_features, in_features)).astype(np.float32) * 0.05


class _StubBackend:
    """Mirrors `IqkSwitchLinear`'s served surface for the dense seams.

    `gemv` reproduces the kernel's pairs-per-activation-row mapping: with
    `mats` pairs over `rows` activation rows, pair `p` reads row
    `p // (mats // rows)`, and a pair count that does not tile the rows
    raises. This is the operand-shape contract the pinned down-seam defect
    was about, so the stub enforces it rather than accepting anything.
    """

    def __init__(self, weights: np.ndarray):
        self.weights = mx.array(weights)
        self.num_experts = int(weights.shape[0])
        self.out_features = int(weights.shape[1])
        self.in_features = int(weights.shape[2])

    def gemv(self, x, indices):
        xt = x.reshape(-1, self.in_features).astype(mx.float16)
        sel = [int(v) for v in np.asarray(indices).reshape(-1)]
        mats = len(sel)
        rows = int(xt.shape[0])
        if rows <= 0 or mats % rows:
            raise RuntimeError(
                f"{mats} pairs do not tile {rows} activation row(s)")
        per = mats // rows
        outs = []
        for pair, slot in enumerate(sel):
            row = xt[pair // per].astype(mx.float32)
            outs.append(mx.matmul(row, self.weights[slot].T))
        out = mx.stack(outs).astype(mx.float16)
        out = out.reshape(list(indices.shape) + [self.out_features])
        return mx.expand_dims(out, -2)

    def dequantized(self):
        return self.weights.astype(mx.float16)


def _install_stub_builder(monkeypatch):
    built = []

    def builder(member, wire_stack, out_features, in_features):
        stacks = int(wire_stack.shape[0])
        built.append({
            "member": member,
            "stacks": stacks,
            "out": out_features,
            "in": in_features,
        })
        return _StubBackend(_ref_weights(stacks, out_features, in_features))

    monkeypatch.setattr(iqk_dense, "_backend_builder", builder)
    return built


def _dense_module(member="iq6_k", out_features=1024, in_features=4096):
    cls = iqk_dense._dense_cls()
    return cls(member, "iqk_relayout", out_features, in_features)


def _delta(before, key):
    return iqk_dense_matmul_call_counts()[key] - before[key]


# --------------------------------------------------------------------------
# Manifest map


def _manifest_entry(**overrides):
    entry = {
        "source_name": "layers.0.attn.wq_a.weight",
        "kind": "affine",
        "format": "iqk",
        "format_params": {"iqk_codec": "iq6_k", "layout": "iqk_relayout"},
        "module_weight_key": "model.layers.0.self_attn.wq_a.weight",
    }
    entry.update(overrides)
    return entry


def test_weight_map_reads_dense_iqk_entries_and_skips_experts():
    manifest = {"tensors": [
        _manifest_entry(),
        {"source_name": "bundle", "kind": "expert", "format": "iqk",
         "format_params": {"iqk_codec": "iq2_ks", "layout": "iqk_relayout"}},
        {"source_name": "other", "kind": "affine", "format": "kquant",
         "format_params": {"kquant_codec": "q8_0"}},
    ]}
    assert manifest_requires_iqk_dense(manifest)
    weight_map = iqk_dense_weight_map_from_manifest(manifest)
    assert list(weight_map) == ["model.layers.0.self_attn.wq_a.weight"]
    assert weight_map[list(weight_map)[0]]["member"] == "iq6_k"


def test_weight_map_normalizes_the_pre_rename_layout():
    manifest = {"tensors": [_manifest_entry(format_params={
        "iqk_codec": "iq6_k", "layout": IQK_LAYOUT_LEGACY_RELAYOUT,
    })]}

    facts = next(iter(iqk_dense_weight_map_from_manifest(manifest).values()))

    assert facts["layout"] == IQK_LAYOUT_IQK_RELAYOUT


def test_weight_map_fails_closed_on_a_routed_member():
    manifest = {"tensors": [_manifest_entry(
        format_params={"iqk_codec": "iq2_ks", "layout": "iqk_relayout"})]}
    with pytest.raises(IqkDenseInstallError, match="not a dense member"):
        iqk_dense_weight_map_from_manifest(manifest)


def test_weight_map_fails_closed_on_an_unknown_layout():
    manifest = {"tensors": [_manifest_entry(
        format_params={"iqk_codec": "iq6_k", "layout": "row_major"})]}
    with pytest.raises(IqkDenseInstallError, match="wire layout"):
        iqk_dense_weight_map_from_manifest(manifest)


def test_weight_map_fails_closed_on_a_malformed_module_key():
    manifest = {"tensors": [_manifest_entry(module_weight_key="wq_a")]}
    with pytest.raises(IqkDenseInstallError, match="module_weight_key"):
        iqk_dense_weight_map_from_manifest(manifest)


# --------------------------------------------------------------------------
# The dense module's routes


def test_dense_module_refuses_the_quantizer_wire_layout():
    cls = iqk_dense._dense_cls()
    with pytest.raises(IqkDenseInstallError, match="does not serve"):
        cls("iq6_k", "ik_wire", 1024, 4096)


def test_dense_module_refuses_a_width_off_the_block_grid():
    cls = iqk_dense._dense_cls()
    with pytest.raises(ValueError, match="multiple of 256"):
        cls("iq5_k", "iqk_relayout", 1024, 4032)


# The production dense families by reduction width class; out widths that
# would be gigabytes in fp32 reference form are scaled (the operand
# mapping is width-independent). wkv keeps its real 512-row out width and
# wq_a its real 1024.
_WIDTH_CLASSES = [
    ("wq_a", 1024, 4096),
    ("wkv", 512, 4096),
    ("wq_b", 2048, 1024),       # real out 32768, scaled
    ("w1", 2048, 4096),
    ("w2", 4096, 2048),
    ("wo_b", 4096, 8192),
    ("lm_head", 1616, 4096),    # real out 129280, scaled; stays % 8 == 0
]


@pytest.mark.parametrize("family,out_features,in_features", _WIDTH_CLASSES)
def test_decode_row_takes_the_gemv_and_matches_the_reference(
        monkeypatch, family, out_features, in_features):
    _install_stub_builder(monkeypatch)
    module = _dense_module(out_features=out_features,
                           in_features=in_features)
    before = iqk_dense_matmul_call_counts()

    x = mx.array(np.random.default_rng(3).standard_normal(
        (1, 1, in_features)).astype(np.float32))
    y = module(x)

    reference = _ref_weights(1, out_features, in_features)[0]
    want = np.asarray(x).reshape(-1).astype(np.float16).astype(
        np.float32) @ reference.T
    assert y.shape == (1, 1, out_features)
    assert y.dtype == mx.float32
    np.testing.assert_allclose(
        np.asarray(y), want.reshape(1, 1, -1), rtol=2e-3, atol=2e-3)
    assert _delta(before, "decode_gemv") == 1
    assert _delta(before, "prefill_dequant") == 0


@pytest.mark.parametrize("family,out_features,in_features", _WIDTH_CLASSES)
def test_bulk_rows_take_the_dequant_route(
        monkeypatch, family, out_features, in_features):
    _install_stub_builder(monkeypatch)
    module = _dense_module(out_features=out_features,
                           in_features=in_features)
    before = iqk_dense_matmul_call_counts()

    x = mx.array(np.random.default_rng(4).standard_normal(
        (1, 5, in_features)).astype(np.float16))
    y = module(x)

    reference = _ref_weights(1, out_features, in_features)[0]
    want = np.asarray(x, dtype=np.float32).reshape(5, -1) @ (
        reference.astype(np.float16).astype(np.float32).T)
    assert y.shape == (1, 5, out_features)
    assert y.dtype == mx.float16
    np.testing.assert_allclose(
        np.asarray(y, dtype=np.float32), want.reshape(1, 5, -1),
        rtol=2e-2, atol=2e-2)
    assert _delta(before, "prefill_dequant") == 1
    assert _delta(before, "decode_gemv") == 0


def test_the_kill_switch_routes_through_the_counted_bridge(monkeypatch):
    _install_stub_builder(monkeypatch)
    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_DENSE_QMV", "0")
    module = _dense_module(out_features=512, in_features=1024)
    before = iqk_dense_matmul_call_counts()

    x = mx.array(np.ones((1, 1, 1024), dtype=np.float32))
    y = module(x)

    assert y.shape == (1, 1, 512)
    assert _delta(before, "delegated") == 1
    assert _delta(before, "decode_gemv") == 0


def test_a_wrong_activation_width_refuses_loudly(monkeypatch):
    _install_stub_builder(monkeypatch)
    module = _dense_module(out_features=512, in_features=1024)
    with pytest.raises(IqkDenseInstallError, match="activation width"):
        module(mx.zeros((1, 1, 512)))


def test_the_default_builder_fails_closed_on_an_unserved_member():
    pytest.importorskip("mlx_iqk")
    module = _dense_module(member="iq6_k", out_features=512,
                           in_features=1024)
    # The kernel repository serves iq2_ks/iq2_k only; the dense members
    # must refuse by name rather than decode garbage.
    with pytest.raises(IqkDenseInstallError, match="no serving kernels"):
        module(mx.zeros((1, 1, 1024)))


# --------------------------------------------------------------------------
# Module install on a graph


def _fake_graph(vocab=1616):
    def linear(out_features, in_features):
        return SimpleNamespace(weight=mx.zeros((out_features, in_features)))

    attn = SimpleNamespace(
        wo_a=linear(8 * 64, 256),
        wo_b=linear(512, 8 * 64),
        o_groups=8,
        o_lora_rank=64,
        n_heads=8,
        head_dim=256,
    )
    shared = SimpleNamespace(
        w1=linear(256, 512), down_proj=linear(512, 256))
    layer = SimpleNamespace(
        self_attn=attn, mlp=SimpleNamespace(shared_experts=shared))
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[layer]),
        lm_head=linear(vocab, 512),
    )
    return model


def _graph_manifest():
    def entry(source, key, member="iq4_k"):
        return {
            "source_name": source,
            "kind": "affine",
            "format": "iqk",
            "format_params": {"iqk_codec": member, "layout": "iqk_relayout"},
            "module_weight_key": key,
        }
    return {"tensors": [
        entry("layers.0.attn.wo_a.weight",
              "model.layers.0.self_attn.wo_a.weight"),
        entry("layers.0.attn.wo_b.weight",
              "model.layers.0.self_attn.wo_b.weight"),
        entry("layers.0.ffn.shared_experts.w2.weight",
              "model.layers.0.mlp.shared_experts.down_proj.weight"),
        entry("head.weight", "lm_head.weight"),
    ]}


def test_module_install_swaps_the_declared_modules(monkeypatch):
    _install_stub_builder(monkeypatch)
    model = _fake_graph()
    installed = install_deepseek_v4_iqk_dense_modules(
        model, _graph_manifest())

    assert installed == 4
    wo_a = model.model.layers[0].self_attn.wo_a
    assert wo_a.mode == "iqk_dense"
    assert (wo_a.out_features, wo_a.in_features) == (512, 256)
    assert model.lm_head.mode == "iqk_dense"
    facts = model._moespresso_dsv4_iqk_dense_install
    assert facts["modules"] == 4
    assert facts["member_counts"] == {"iq4_k": 4}


def test_seam_install_wraps_wo_b_with_the_fp32_contract(monkeypatch):
    _install_stub_builder(monkeypatch)
    model = _fake_graph()
    install_deepseek_v4_iqk_dense_modules(model, _graph_manifest())
    patched = install_deepseek_v4_iqk_dense_seams(model)

    assert patched == 1
    wo_b = model.model.layers[0].self_attn.wo_b
    assert wo_b._moespresso_dsv4_iqk_fp32_dense
    y = wo_b(mx.zeros((1, 1, 512), dtype=mx.float16))
    assert y.dtype == mx.float32
    down = model.model.layers[0].mlp.shared_experts.down_proj
    assert down.counter_site == "w2"
    assert model.lm_head.counter_site == "lm_head"


# --------------------------------------------------------------------------
# The grouped wo_a projection: the operand-mapping lesson


def test_wo_a_gather_uses_per_group_rows_not_a_broadcast_token(monkeypatch):
    _install_stub_builder(monkeypatch)
    model = _fake_graph()
    install_deepseek_v4_iqk_dense_modules(model, _graph_manifest())
    install_deepseek_v4_iqk_dense_seams(model)
    attn = model.model.layers[0].self_attn
    groups, rank, group_feat = 8, 64, 256
    before = iqk_dense_matmul_call_counts()

    x = np.random.default_rng(9).standard_normal(
        (1, 1, groups * group_feat)).astype(np.float32)
    y = attn._grouped_output_projection(mx.array(x))

    weights = _ref_weights(groups, rank, group_feat)
    grouped = x.reshape(groups, group_feat).astype(np.float16).astype(
        np.float32)
    right = np.concatenate(
        [grouped[g] @ weights[g].T for g in range(groups)])
    # The wrong mapping broadcasts group 0's activation row over every
    # group; asserting it differs keeps this pin non-vacuous.
    wrong = np.concatenate(
        [grouped[0] @ weights[g].T for g in range(groups)])
    assert float(np.abs(right - wrong).max()) > 1e-3

    assert y.shape == (1, 1, groups * rank)
    assert y.dtype == mx.float32
    np.testing.assert_allclose(
        np.asarray(y).reshape(-1), right, rtol=2e-3, atol=2e-3)
    assert _delta(before, "wo_a_gather") == 1
    assert _delta(before, "wo_a_bulk") == 0


def test_a_refusing_backend_counts_no_grouped_route(monkeypatch):
    """A route counted before its backend builds reports a phantom forward.

    Both grouped `wo_a` routes build the backend first, so a member the
    kernel repository does not serve leaves the census untouched rather than
    recording engagement no forward ever completed.
    """
    def refusing(member, wire_stack, out_features, in_features):
        raise IqkDenseInstallError("no serving kernels")

    monkeypatch.setattr(iqk_dense, "_backend_builder", refusing)
    model = _fake_graph()
    install_deepseek_v4_iqk_dense_modules(model, _graph_manifest())
    install_deepseek_v4_iqk_dense_seams(model)
    attn = model.model.layers[0].self_attn
    groups, group_feat = 8, 256

    for rows in (1, 3):
        before = iqk_dense_matmul_call_counts()
        x = np.zeros((1, rows, groups * group_feat), dtype=np.float32)
        with pytest.raises(IqkDenseInstallError, match="no serving kernels"):
            attn._grouped_output_projection(mx.array(x))
        assert _delta(before, "wo_a_gather") == 0
        assert _delta(before, "wo_a_bulk") == 0


def test_dense_counters_reach_all_three_census_surfaces(monkeypatch):
    """A counter on one surface defeats a census-gated instrument.

    The arm instruments read the module's own report, `ssd_streaming_stats`,
    and the speed-stats count keys interchangeably, so a route counter that
    exists on only one of them reports no arm difference on the other two.
    """
    from moespresso.runtime.deepseek_v4.speed_stats import _COUNT_KEYS
    from moespresso.runtime.ssd_streaming_build import ssd_streaming_stats

    _install_stub_builder(monkeypatch)
    model = _fake_graph()
    install_deepseek_v4_iqk_dense_modules(model, _graph_manifest())
    install_deepseek_v4_iqk_dense_seams(model)
    attn = model.model.layers[0].self_attn
    groups, group_feat = 8, 256
    attn._grouped_output_projection(
        mx.array(np.zeros((1, 1, groups * group_feat), dtype=np.float32)))

    counts = iqk_dense_matmul_call_counts()
    census = ssd_streaming_stats(model)
    for count_key, census_key in (
        ("decode_gemv", "iqk_dense_decode_gemv_calls"),
        ("decode_gemv_wo_b", "iqk_dense_decode_gemv_wo_b_calls"),
        ("decode_gemv_lm_head", "iqk_dense_decode_gemv_lm_head_calls"),
        ("decode_gemv_w2", "iqk_dense_decode_gemv_w2_calls"),
        ("prefill_dequant", "iqk_dense_prefill_dequant_calls"),
        ("wo_a_gather", "iqk_dense_wo_a_gather_calls"),
        ("wo_a_bulk", "iqk_dense_wo_a_bulk_calls"),
        ("delegated", "iqk_dense_delegated_calls"),
    ):
        assert census[census_key] == counts[count_key], census_key
        assert census_key in _COUNT_KEYS, census_key
    # The arm ran the gather route, so the pin cannot pass on all zeros.
    assert census["iqk_dense_wo_a_gather_calls"] >= 1


def test_wo_a_bulk_rows_take_the_grouped_dequant_route(monkeypatch):
    _install_stub_builder(monkeypatch)
    model = _fake_graph()
    install_deepseek_v4_iqk_dense_modules(model, _graph_manifest())
    install_deepseek_v4_iqk_dense_seams(model)
    attn = model.model.layers[0].self_attn
    groups, rank, group_feat = 8, 64, 256
    before = iqk_dense_matmul_call_counts()

    x = np.random.default_rng(11).standard_normal(
        (1, 3, groups * group_feat)).astype(np.float32)
    y = attn._grouped_output_projection(mx.array(x))

    weights = _ref_weights(groups, rank, group_feat).astype(
        np.float16).astype(np.float32)
    grouped = x.reshape(3, groups, group_feat)
    want = np.concatenate(
        [grouped[:, g, :] @ weights[g].T for g in range(groups)], axis=-1)
    assert y.shape == (1, 3, groups * rank)
    np.testing.assert_allclose(
        np.asarray(y).reshape(3, -1), want, rtol=2e-3, atol=2e-3)
    assert _delta(before, "wo_a_bulk") == 1
    assert _delta(before, "wo_a_gather") == 0


# --------------------------------------------------------------------------
# The lm_head patch


def test_lm_head_patch_slices_prefill_and_serves_fp32_logits(monkeypatch):
    _install_stub_builder(monkeypatch)
    vocab, hidden = 1616, 512
    dense_cls = iqk_dense._dense_cls()
    head = dense_cls("iq4_k", "iqk_relayout", vocab, hidden)
    hidden_rows = mx.array(np.random.default_rng(13).standard_normal(
        (1, 7, hidden)).astype(np.float32))

    class _FakeModel:
        def __init__(self):
            self.lm_head = head
            self.model = lambda input_ids, cache=None, mask=None: hidden_rows

        def __call__(self, input_ids, cache=None, mask=None):
            raise AssertionError("stock call must be replaced")

    model = _FakeModel()
    assert patch_deepseek_v4_iqk_dense_lm_head(model)
    before = iqk_dense_matmul_call_counts()

    logits = model(mx.zeros((1, 7)), cache=object())
    assert logits.shape == (1, 1, vocab)
    assert logits.dtype == mx.float32
    assert _delta(before, "decode_gemv") == 1
    assert _delta(before, "decode_gemv_lm_head") == 1

    # Scorer path: no cache keeps every row on the bridge.
    scored = model(mx.zeros((1, 7)))
    assert scored.shape == (1, 7, vocab)
    assert _delta(before, "prefill_dequant") == 1


def test_lm_head_patch_declines_a_non_iqk_head():
    model = SimpleNamespace(lm_head=SimpleNamespace(mode="kquant"))
    assert patch_deepseek_v4_iqk_dense_lm_head(model) is False
