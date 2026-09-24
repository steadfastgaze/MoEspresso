from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import moespresso.runtime.deepseek_v4.model as dsv4_runtime
import moespresso.runtime.verify as runtime_verify
from moespresso.core.artifact import make_artifact, write_artifact
from moespresso.runtime.deepseek_v4.expert_layout import (
    PER_LAYER_EXPERTS_FEATURE,
    DeepseekV4ExpertLayoutError,
    parse_deepseek_v4_expert_layout,
    validate_expert_index_counts,
)
from moespresso.runtime.deepseek_v4.model import (
    DeepseekV4RuntimeLoadError,
    _DeepseekV4CompactRouterProxy,
    _DeepseekV4RouterGateContract,
    _deepseek_v4_compact_router_reserve_bytes,
    _resize_deepseek_v4_score_router_gates,
    _restore_deepseek_v4_source_width_routers,
    _validate_deepseek_v4_compact_router_shapes,
    load_deepseek_v4_package_model,
)


_SOURCE_PACKAGE_ID = "pkg:" + "a" * 64
_SELECTION_ID = "select:" + "b" * 64


def _layers(score_ids=(2, 7, 18, 39, 90, 155, 203, 250)):
    full = list(range(256))
    return {
        "0": {"num_experts": 256, "source_expert_ids": full},
        "1": {"num_experts": 256, "source_expert_ids": full},
        "2": {"num_experts": 256, "source_expert_ids": full},
        "3": {
            "num_experts": len(score_ids),
            "source_expert_ids": list(score_ids),
        },
    }


def _manifest(*, selection_id=_SELECTION_ID, layers=None):
    router_tensors = [
        {
            "source_name": f"layers.{layer}.ffn.gate.weight",
            "role": "moe.router_gate",
            "kind": "passthrough",
            "layer_index": layer,
            "shard": "model-00001-of-00001.safetensors",
            "key_prefix": f"layers.{layer}.ffn.gate.weight",
            "format": "fp16",
            "format_params": {},
        }
        for layer in range(4)
    ]
    router_tensors.append({
        "source_name": "layers.3.ffn.gate.bias",
        "role": "moe.router_bias",
        "kind": "passthrough",
        "layer_index": 3,
        "shard": "model-00001-of-00001.safetensors",
        "key_prefix": "layers.3.ffn.gate.bias",
        "format": "raw_dtype_passthrough",
        "format_params": {},
    })
    return {
        "architecture": {
            "family": "deepseek_v4_flash",
            "config": {"n_routed_experts": 256},
        },
        "required_features": [PER_LAYER_EXPERTS_FEATURE],
        "inputs": [selection_id],
        "files": [
            {"path": "expert_selection.json"},
            {"path": "model-00001-of-00001.safetensors"},
        ],
        "required_ops": ["iqk_dequant"],
        "tensors": [
            *router_tensors,
            {"format": "iqk", "kind": "expert"},
        ],
        "expert_layout": {
            "bundled": True,
            "per_layer_experts": {
                "source_selection_artifact_id": selection_id,
                "source_package_manifest_id": _SOURCE_PACKAGE_ID,
                "source_num_experts": 256,
                "top_k": 6,
                "layers": layers or _layers(),
            },
        },
    }


class _Index:
    def __init__(self, counts):
        self.counts = dict(counts)

    def layers_indexed(self):
        return tuple(sorted(self.counts))

    def num_experts_for_layer(self, layer):
        return self.counts[layer]

    def has_projection(self, *, layer, projection):
        del layer, projection
        return False


class _Tensor:
    def __init__(self, shape, dtype="float16"):
        self.shape = tuple(shape)
        self.dtype = dtype


class _Gate:
    def __init__(self, layer, *, hash_router):
        self.layer_id = layer
        self.hash = hash_router
        self.args = SimpleNamespace(n_routed_experts=256, vocab_size=1024)
        self.weight = _Tensor((256, 12))
        if hash_router:
            self.tid2eid = _Tensor((1024, 6), "int32")
        else:
            self.bias = _Tensor((256,))


def _router_model():
    gates = [_Gate(layer, hash_router=layer < 3) for layer in range(4)]
    layers = [SimpleNamespace(mlp=SimpleNamespace(gate=gate)) for gate in gates]
    return SimpleNamespace(model=SimpleNamespace(layers=layers)), gates


def test_parse_compact_layout_and_validate_index_counts():
    layout = parse_deepseek_v4_expert_layout(_manifest())

    assert layout is not None
    assert layout.source_num_experts == 256
    assert layout.top_k == 6
    assert layout.layers[3].source_expert_ids == (2, 7, 18, 39, 90, 155, 203, 250)
    validate_expert_index_counts(
        layout,
        _Index({0: 256, 1: 256, 2: 256, 3: 8}),
    )

    with pytest.raises(DeepseekV4ExpertLayoutError, match="mismatch at layer 3"):
        validate_expert_index_counts(
            layout,
            _Index({0: 256, 1: 256, 2: 256, 3: 7}),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest["required_features"].clear(),
            "required_features",
        ),
        (
            lambda manifest: manifest["inputs"].clear(),
            "inputs must contain exactly one",
        ),
        (
            lambda manifest: manifest["expert_layout"]["per_layer_experts"][
                "layers"
            ]["3"].update(source_expert_ids=[2, 7, 18, 39, 90, 155, 155, 250]),
            "strictly increasing",
        ),
        (
            lambda manifest: manifest["expert_layout"]["per_layer_experts"][
                "layers"
            ]["0"].update(
                num_experts=255,
                source_expert_ids=list(range(255)),
            ),
            "hash layer",
        ),
        (
            lambda manifest: manifest["architecture"]["config"].update(
                n_routed_experts=8
            ),
            "must remain 256",
        ),
    ],
)
def test_compact_layout_rejects_partial_or_inconsistent_contract(mutate, message):
    manifest = copy.deepcopy(_manifest())
    mutate(manifest)

    with pytest.raises(DeepseekV4ExpertLayoutError, match=message):
        parse_deepseek_v4_expert_layout(manifest)


def test_legacy_manifest_has_no_compact_layout():
    assert parse_deepseek_v4_expert_layout({
        "architecture": {"family": "deepseek_v4_flash"},
        "expert_layout": {"bundled": True},
    }) is None


def test_resize_changes_only_score_router_weight_and_bias():
    layout = parse_deepseek_v4_expert_layout(_manifest())
    assert layout is not None
    model, gates = _router_model()
    hash_state = [(gate.weight, gate.tid2eid) for gate in gates[:3]]
    allocations = []

    def zeros(shape, *, dtype):
        allocations.append((tuple(shape), dtype))
        return _Tensor(shape, dtype)

    resized = _resize_deepseek_v4_score_router_gates(
        model,
        layout,
        zeros_fn=zeros,
    )

    assert resized == 1
    assert [(gate.weight, gate.tid2eid) for gate in gates[:3]] == hash_state
    assert gates[3].weight.shape == (8, 12)
    assert gates[3].bias.shape == (8,)
    assert gates[3].args.n_routed_experts == 256
    assert allocations == [((8, 12), "float16"), ((8,), "float16")]
    assert model._moespresso_dsv4_compact_router_layers == {
        0: 256,
        1: 256,
        2: 256,
        3: 8,
    }


def test_resize_rejects_partly_compact_score_router():
    layout = parse_deepseek_v4_expert_layout(_manifest())
    assert layout is not None
    model, gates = _router_model()
    gates[3].weight = _Tensor((8, 12))

    with pytest.raises(DeepseekV4RuntimeLoadError, match="mixed compact/source"):
        _resize_deepseek_v4_score_router_gates(
            model,
            layout,
            zeros_fn=lambda shape, *, dtype: _Tensor(shape, dtype),
        )


def test_post_load_router_validation_rejects_half_repacked_score_gate():
    layout = parse_deepseek_v4_expert_layout(
        _manifest(layers=_layers(tuple(range(192))))
    )
    assert layout is not None
    model, gates = _router_model()
    _resize_deepseek_v4_score_router_gates(
        model,
        layout,
        zeros_fn=lambda shape, *, dtype: _Tensor(shape, dtype),
    )
    assert gates[3].weight.shape[0] == 192
    gates[3].weight = _Tensor((256, 12))
    gates[3].bias = _Tensor((256,))

    with pytest.raises(DeepseekV4RuntimeLoadError, match="weight width 256 != 192"):
        _validate_deepseek_v4_compact_router_shapes(model, layout)


def test_post_load_router_validation_requires_intact_hash_tid2eid():
    layout = parse_deepseek_v4_expert_layout(_manifest())
    assert layout is not None
    model, gates = _router_model()
    _resize_deepseek_v4_score_router_gates(
        model,
        layout,
        zeros_fn=lambda shape, *, dtype: _Tensor(shape, dtype),
    )

    assert _validate_deepseek_v4_compact_router_shapes(model, layout) == 4
    gates[1].tid2eid = _Tensor((1024, 5), "int32")
    with pytest.raises(DeepseekV4RuntimeLoadError, match="tid2eid was not preserved"):
        _validate_deepseek_v4_compact_router_shapes(model, layout)


def test_compact_score_router_matches_source_router_with_dropped_rows_masked():
    rng = np.random.default_rng(917)
    selected = np.asarray(_layers()["3"]["source_expert_ids"], dtype=np.int64)
    weight = rng.normal(size=(256, 9))
    bias = rng.normal(scale=0.2, size=(256,))
    hidden = rng.normal(size=(5, 9))

    full_logits = hidden @ weight.T
    full_scores = np.sqrt(np.logaddexp(0.0, full_logits))
    masked_selection = full_scores + bias
    dropped = np.ones(256, dtype=bool)
    dropped[selected] = False
    masked_selection[:, dropped] = -np.inf
    full_top = np.argsort(-masked_selection, axis=-1)[:, :6]

    compact_logits = hidden @ weight[selected].T
    compact_scores = np.sqrt(np.logaddexp(0.0, compact_logits))
    compact_top = np.argsort(-(compact_scores + bias[selected]), axis=-1)[:, :6]
    mapped_compact_top = selected[compact_top]

    np.testing.assert_array_equal(mapped_compact_top, full_top)
    full_weights = np.take_along_axis(full_scores, full_top, axis=-1)
    compact_weights = np.take_along_axis(compact_scores, compact_top, axis=-1)
    full_weights /= full_weights.sum(axis=-1, keepdims=True)
    compact_weights /= compact_weights.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(compact_weights, full_weights, rtol=0.0, atol=1e-14)


def test_source_width_router_restores_geometry_and_returns_compact_ids(monkeypatch):
    import mlx.core as mx
    import moespresso.runtime.pooled_switchglu as psg

    layout = parse_deepseek_v4_expert_layout(_manifest())
    assert layout is not None
    score_ids = layout.layers[3].source_expert_ids

    class Gate:
        def __init__(self, layer, *, hash_router, weight, bias=None):
            self.layer_id = layer
            self.hash = hash_router
            self.weight = weight
            self.bias = bias
            self.args = SimpleNamespace(
                n_routed_experts=256,
                num_experts_per_tok=6,
                norm_topk_prob=True,
                routed_scaling_factor=1.0,
                vocab_size=1024,
            )
            if hash_router:
                self.tid2eid = mx.zeros((1024, 6), dtype=mx.int32)

        def __call__(self, x, input_ids=None):
            del input_ids
            gates = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
            return select(gates, self.bias, 6, 1.0, True)

    def select(gates, bias, k, scaling, norm):
        scores = mx.sqrt(mx.log1p(mx.exp(gates)))
        ids = mx.argpartition(-(scores + bias), kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, ids, axis=-1)
        if norm:
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        return ids, weights * scaling

    rng = np.random.default_rng(29)
    compact_weight_np = rng.normal(size=(len(score_ids), 12)).astype(np.float16)
    compact_bias_np = rng.normal(size=len(score_ids)).astype(np.float32)
    layers = []
    bases = []
    for layer in range(4):
        if layer < 3:
            weight = mx.zeros((256, 12), dtype=mx.float16)
            bias = None
        else:
            weight = mx.array(compact_weight_np)
            bias = mx.array(compact_bias_np)
        base = Gate(layer, hash_router=layer < 3, weight=weight, bias=bias)
        contract = _DeepseekV4RouterGateContract(
            base,
            mx=mx,
            dsv4_model=SimpleNamespace(sqrtsoftplus_select=select),
        )
        count = 256 if layer < 3 else len(score_ids)
        switch = SimpleNamespace(**{
            name: SimpleNamespace(pool=SimpleNamespace(num_experts=count))
            for name in ("gate_proj", "up_proj", "down_proj")
        })
        layers.append(SimpleNamespace(mlp=SimpleNamespace(
            gate=contract,
            switch_mlp=switch,
        )))
        bases.append(base)
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    model._moespresso_dsv4_compact_router_layers = {
        layer: layout.layers[layer].num_experts for layer in layout.layers
    }
    dual_installs = []

    def install_dual(switch, source_ids):
        dual_installs.append((switch, np.asarray(source_ids)))
        return True

    monkeypatch.setattr(psg, "install_compact_iqk_dual_gemv", install_dual)

    assert _deepseek_v4_compact_router_reserve_bytes(model) == 248 * (12 * 2 + 4)
    assert _restore_deepseek_v4_source_width_routers(model, layout) == 1
    assert len(dual_installs) == 1
    assert dual_installs[0][0] is layers[3].mlp.switch_mlp
    np.testing.assert_array_equal(dual_installs[0][1], np.asarray(score_ids))
    assert model._moespresso_dsv4_compact_iqk_dual_gemv_layers == 1
    proxy = layers[3].mlp.gate
    assert isinstance(proxy, _DeepseekV4CompactRouterProxy)
    assert bases[3].weight.shape == (256, 12)
    assert bases[3].bias.shape == (256,)
    np.testing.assert_array_equal(
        np.asarray(bases[3].weight)[np.asarray(score_ids)],
        compact_weight_np,
    )
    np.testing.assert_array_equal(
        np.asarray(bases[3].bias)[np.asarray(score_ids)],
        compact_bias_np,
    )
    removed = np.ones(256, dtype=np.bool_)
    removed[np.asarray(score_ids)] = False
    assert np.count_nonzero(np.asarray(bases[3].weight)[removed]) == 0
    assert np.all(np.isneginf(np.asarray(bases[3].bias)[removed]))
    compact_ids, scores = proxy(mx.array(rng.normal(size=(1, 2, 12))))
    mx.eval(compact_ids, scores)
    compact_ids_np = np.asarray(compact_ids)
    assert compact_ids_np.min() >= 0
    assert compact_ids_np.max() < len(score_ids)
    assert model._moespresso_dsv4_source_width_router_layers == 1
    assert (
        model._moespresso_dsv4_source_width_router_selection_id
        == layout.source_selection_artifact_id
    )

    bases[3].weight = mx.array(compact_weight_np)
    bases[3].bias = mx.array(compact_bias_np)
    object.__setattr__(layers[3].mlp, "gate", proxy._contract)
    layers[3].mlp.switch_mlp.gate_proj.pool.num_experts = len(score_ids) - 1
    with pytest.raises(DeepseekV4RuntimeLoadError, match="disagrees with its pools"):
        _restore_deepseek_v4_source_width_routers(model, layout)


def test_source_width_router_requires_runtime_gate_contract():
    layout = parse_deepseek_v4_expert_layout(_manifest())
    assert layout is not None
    model, _gates = _router_model()
    for layer, decoder in enumerate(model.model.layers):
        count = layout.layers[layer].num_experts
        decoder.mlp.switch_mlp = SimpleNamespace(**{
            name: SimpleNamespace(pool=SimpleNamespace(num_experts=count))
            for name in ("gate_proj", "up_proj", "down_proj")
        })

    with pytest.raises(DeepseekV4RuntimeLoadError, match="no runtime contract"):
        _restore_deepseek_v4_source_width_routers(model, layout)


def _selection_artifact(*, layers=None):
    return make_artifact(
        "deepseek_v4_expert_selection",
        {"model_family": "deepseek_v4_flash"},
        {"tool": "test", "version": "1"},
        required_features=[PER_LAYER_EXPERTS_FEATURE],
        status="valid",
        source_package_manifest_id=_SOURCE_PACKAGE_ID,
        source_num_experts=256,
        top_k=6,
        layers=layers or _layers(),
    )


def _router_headers(*, score_width):
    headers = {
        f"layers.{layer}.ffn.gate.weight": {
            "dtype": "F16",
            "shape": [256 if layer < 3 else score_width, 12],
            "data_offsets": [0, 0],
        }
        for layer in range(4)
    }
    headers["layers.3.ffn.gate.bias"] = {
        "dtype": "F32",
        "shape": [score_width],
        "data_offsets": [0, 0],
    }
    return headers


def test_verifier_binds_selection_artifact_and_bundle_counts(tmp_path, monkeypatch):
    selection = _selection_artifact()
    write_artifact(tmp_path / "expert_selection.json", selection)
    manifest = _manifest(selection_id=selection["artifact_id"])
    monkeypatch.setattr(
        runtime_verify,
        "build_expert_index",
        lambda _path: _Index({0: 256, 1: 256, 2: 256, 3: 8}),
    )
    (tmp_path / "model-00001-of-00001.safetensors").touch()
    monkeypatch.setattr(
        runtime_verify,
        "read_header",
        lambda _path: _router_headers(score_width=8),
    )

    assert runtime_verify._verify_deepseek_v4_expert_layout(
        manifest,
        tmp_path,
    ) == []

    manifest["expert_layout"]["per_layer_experts"]["layers"]["3"] = {
        "num_experts": 8,
        "source_expert_ids": [1, 7, 18, 39, 90, 155, 203, 250],
    }
    issues = runtime_verify._verify_deepseek_v4_expert_layout(manifest, tmp_path)
    assert any(issue.code == "runtime.expert_selection_payload_mismatch" for issue in issues)


def test_verifier_rejects_256_router_headers_with_192_expert_bundle(
    tmp_path,
    monkeypatch,
):
    layers = _layers(tuple(range(192)))
    selection = _selection_artifact(layers=layers)
    write_artifact(tmp_path / "expert_selection.json", selection)
    (tmp_path / "model-00001-of-00001.safetensors").touch()
    manifest = _manifest(selection_id=selection["artifact_id"], layers=layers)
    monkeypatch.setattr(
        runtime_verify,
        "build_expert_index",
        lambda _path: _Index({0: 256, 1: 256, 2: 256, 3: 192}),
    )
    monkeypatch.setattr(
        runtime_verify,
        "read_header",
        lambda _path: _router_headers(score_width=256),
    )

    issues = runtime_verify._verify_deepseek_v4_expert_layout(manifest, tmp_path)

    mismatches = [
        issue for issue in issues
        if issue.code == "runtime.router_header_width_mismatch"
    ]
    assert len(mismatches) == 2
    assert {issue.actual for issue in mismatches} == {256}
    assert {issue.expected for issue in mismatches} == {192}


def test_loader_validates_and_resizes_before_regular_weight_hydration(
    tmp_path: Path,
    monkeypatch,
):
    events = []
    model = SimpleNamespace(model=SimpleNamespace(layers=[]))
    index = _Index({0: 256, 1: 256, 2: 256, 3: 8})

    def expert_index(_path):
        events.append("index")
        return index

    def skeleton(_path, **kwargs):
        events.append("skeleton")
        return model, kwargs["model_config"]

    def resize(model_arg, layout):
        assert model_arg is model
        assert layout.layers[3].num_experts == 8
        events.append("resize")
        return 1

    def load_regular(model_arg, package_dir, **kwargs):
        del package_dir
        assert model_arg is model
        assert kwargs["shard_names"] == ["model-00001-of-00001.safetensors"]
        events.append("regular")
        return 1, 1

    def validate_post(model_arg, layout):
        assert model_arg is model
        assert layout.layers[3].num_experts == 8
        events.append("post")
        return 4

    def restore_source_width(model_arg, layout):
        assert model_arg is model
        assert layout.layers[3].num_experts == 8
        events.append("source_width")
        return 1

    def install_bundles(*_args, **_kwargs):
        events.append("bundles")
        return 1

    def wrap_switchglus(*_args, **_kwargs):
        events.append("wrap")
        return 0

    monkeypatch.setattr(
        dsv4_runtime,
        "_resize_deepseek_v4_score_router_gates",
        resize,
    )
    monkeypatch.setattr(
        dsv4_runtime,
        "_load_deepseek_v4_regular_weights",
        load_regular,
    )
    monkeypatch.setattr(
        dsv4_runtime,
        "_validate_deepseek_v4_compact_router_shapes",
        validate_post,
    )
    monkeypatch.setattr(
        dsv4_runtime,
        "_restore_deepseek_v4_source_width_routers",
        restore_source_width,
    )
    monkeypatch.setattr(dsv4_runtime, "_prewarm_wired_limit", lambda _model: None)

    result = load_deepseek_v4_package_model(
        _manifest(),
        tmp_path,
        load_config_fn=lambda _path: {
            "model_type": "deepseek_v4",
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "num_hash_layers": 3,
            "num_hidden_layers": 4,
        },
        load_skeleton_fn=skeleton,
        load_tokenizer_fn=lambda _path: "TOK",
        expert_index_fn=expert_index,
        install_bundles_fn=install_bundles,
        wrap_switchglus_fn=wrap_switchglus,
        apply_tensor_map_fn=lambda *_args, **_kwargs: None,
        read_jang_config_fn=lambda _path: {},
    )

    assert result == (model, "TOK")
    assert events == [
        "index",
        "skeleton",
        "resize",
        "regular",
        "post",
        "bundles",
        "wrap",
        "source_width",
    ]
    assert events.count("index") == 1


def test_loader_rejects_half_repacked_router_after_weight_hydration(
    tmp_path: Path,
    monkeypatch,
):
    layers = _layers(tuple(range(192)))
    manifest = _manifest(layers=layers)
    model, gates = _router_model()
    index = _Index({0: 256, 1: 256, 2: 256, 3: 192})

    def resize(model_arg, layout):
        return _resize_deepseek_v4_score_router_gates(
            model_arg,
            layout,
            zeros_fn=lambda shape, *, dtype: _Tensor(shape, dtype),
        )

    def load_regular(_model, _package_dir, **_kwargs):
        assert gates[3].weight.shape == (192, 12)
        gates[3].weight = _Tensor((256, 12))
        gates[3].bias = _Tensor((256,))
        return 1, 1

    monkeypatch.setattr(
        dsv4_runtime,
        "_resize_deepseek_v4_score_router_gates",
        resize,
    )
    monkeypatch.setattr(
        dsv4_runtime,
        "_load_deepseek_v4_regular_weights",
        load_regular,
    )

    with pytest.raises(DeepseekV4RuntimeLoadError, match="weight width 256 != 192"):
        load_deepseek_v4_package_model(
            manifest,
            tmp_path,
            load_config_fn=lambda _path: {
                "model_type": "deepseek_v4",
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
                "num_hash_layers": 3,
                "num_hidden_layers": 4,
            },
            load_skeleton_fn=lambda _path, **kwargs: (
                model,
                kwargs["model_config"],
            ),
            load_tokenizer_fn=lambda _path: "TOK",
            expert_index_fn=lambda _path: index,
            install_bundles_fn=lambda *_args, **_kwargs: 1,
            wrap_switchglus_fn=lambda *_args, **_kwargs: 0,
            apply_tensor_map_fn=lambda *_args, **_kwargs: None,
            read_jang_config_fn=lambda _path: {},
        )
