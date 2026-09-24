from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import mlx.core as mx
import pytest

from moespresso.runtime.disk_kv import load_prompt_cache_payload, save_prompt_cache_payload
from moespresso.runtime.qwen4.cache_snapshot import snapshot_composite_state, restore_composite_state
from moespresso.runtime.qwen4.model import Qwen4DecoderLayer, Qwen4StateCoordinator
from moespresso.runtime.qwen4.qsa import Qwen4QSAAdapter, Qwen4SparseAttention
from moespresso.runtime.qwen4.qsa_kvarn import Qwen4MutableKVarNQSAStateBackend
from test_qwen4_kvarn_snapshot import same_arrays
from test_qwen4_model_shell import _model_with_real_gdn, _MLP, _PLE, _Residual


def model():
    result = _model_with_real_gdn()
    result.layers[0].mixer.module.set_dtype(mx.bfloat16)
    result.embedding = lambda tokens: mx.stack([tokens, tokens + 1], axis=-1).astype(mx.bfloat16)
    qsa = Qwen4SparseAttention(hidden_size=2)
    qsa.set_dtype(mx.bfloat16)
    result.layers.append(Qwen4DecoderLayer(
        mixer_kind="qsa", attention_residual=_Residual([], "attention", 0.0),
        mixer=Qwen4QSAAdapter(
            qsa, state_backend=Qwen4MutableKVarNQSAStateBackend(qsa, max_context_tokens=1024),
        ),
        mlp_residual=_Residual([], "mlp", 0.0), mlp=_MLP([], "mlp", 0.2, 0.0),
        ple=_PLE([], 0.01),
    ))
    return result


def leaves(tree):
    if tree is None:
        return []
    if isinstance(tree, mx.array):
        return [tree]
    return [array for node in tree for array in leaves(node)]


@pytest.fixture
def live():
    mx.random.seed(914)
    target = model()
    coordinator = target.new_coordinator(1)
    coordinator.forward_chunk(mx.array([[1, 2, 3, 1] * 64], dtype=mx.int64))
    try:
        yield target, coordinator
    finally:
        coordinator.close()
        target.close()


def test_composite_disk_roundtrip_preserves_real_mixer_continuation(live, tmp_path):
    target, source = live
    trees, meta = snapshot_composite_state(target, source.state)
    path, _ = save_prompt_cache_payload(
        tmp_path, "composite", cache_state_trees=[trees], meta_state_trees=[meta], safety_metadata={},
    )
    disk_trees, disk_meta, _ = load_prompt_cache_payload(tmp_path, path)
    restored = restore_composite_state(target, disk_trees[0], disk_meta[0], expected_frontier=256)
    same_arrays(leaves(trees), leaves(snapshot_composite_state(target, restored)[0]))
    other = Qwen4StateCoordinator(target, restored)
    try:
        for token in (2, 1, 3):
            inputs = mx.array([[token]], dtype=mx.int64)
            a = source.propose(inputs)
            b = other.propose(inputs)
            source.commit(a, 1)
            other.commit(b, 1)
            same_arrays([a.logits], [b.logits])
            same_arrays(
                leaves(snapshot_composite_state(target, source.state)[0]),
                leaves(snapshot_composite_state(target, other.state)[0]),
            )
        # Both owners advanced; the original disk frontier remains reusable.
        again = restore_composite_state(target, disk_trees[0], disk_meta[0], expected_frontier=256)
        same_arrays(leaves(trees), leaves(snapshot_composite_state(target, again)[0]))
    finally:
        other.close()


@pytest.mark.parametrize("change", ["identity", "schema", "frontier", "revision", "kind", "ple", "gdn_schema", "qsa_frontier"])
def test_composite_rejects_incompatible_state_without_changing_live_owner(live, change):
    target, coordinator = live
    trees, meta = snapshot_composite_state(target, coordinator.state)
    altered = deepcopy(meta)
    if change == "identity":
        altered["cache_identity"] = "other-package"
    elif change == "schema":
        altered["schema"] = "future"
    elif change == "frontier":
        altered["frontier"] = 257
    elif change == "revision":
        altered["revision"] = True
    elif change == "kind":
        altered["layers"][0]["kind"] = "qsa"
    elif change == "ple":
        altered["layers"][1]["ple"] = False
    elif change == "gdn_schema":
        altered["layers"][0]["mixer"]["schema"] = "future"
    else:
        altered["layers"][1]["mixer"]["frontier"] = 128
    original = coordinator.state
    with pytest.raises(ValueError):
        restore_composite_state(target, trees, altered, expected_frontier=256)
    assert coordinator.state is original
    same_arrays(leaves(trees), leaves(snapshot_composite_state(target, original)[0]))


def test_composite_refuses_missing_layer_and_position_state(live):
    target, coordinator = live
    trees, meta = snapshot_composite_state(target, coordinator.state)
    with pytest.raises(ValueError):
        restore_composite_state(target, trees[:-1], meta, expected_frontier=256)
    malformed = ((trees[0][0], trees[0][1][..., :-1]), *trees[1:])
    with pytest.raises(ValueError, match="position"):
        restore_composite_state(target, malformed, meta, expected_frontier=256)
    invalid = replace(coordinator.state, batch_size=2)
    with pytest.raises(ValueError):
        snapshot_composite_state(target, invalid)
