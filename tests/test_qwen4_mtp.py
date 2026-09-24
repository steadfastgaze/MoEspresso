"""Shifted MTP fusion and functional attention continuation contracts."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from moespresso.runtime.qwen4.model import Qwen4DecoderLayer
from moespresso.runtime.qwen4.moe import Qwen4SparseMoEBlock
from moespresso.runtime.qwen4.mtp import MTP_GRAPH_CONTRACT, Qwen4MTPHead
from moespresso.runtime.qwen4.primitives import Qwen4GatedResidual
from moespresso.runtime.qwen4.qsa import Qwen4QSAAdapter, Qwen4SparseAttention


def _head():
    mx.random.seed(730)
    hidden, branches = 8, 4
    attention = Qwen4SparseAttention(
        hidden_size=hidden, num_query_heads=2, num_kv_heads=1, head_dim=4,
        index_query_heads=1, index_head_dim=4, token_budget=4,
        compress_ratio=2, rotary_dim=4, mrope_section=(1, 1, 0),
    )
    layer = Qwen4DecoderLayer(
        mixer_kind="qsa",
        attention_residual=Qwen4GatedResidual(hidden, branches, 4),
        mixer=Qwen4QSAAdapter(attention),
        mlp_residual=Qwen4GatedResidual(hidden, branches, 4),
        mlp=Qwen4SparseMoEBlock(hidden, 8, 8, 4, 2),
    )
    layer.mlp.gate.weight = mx.random.normal((4, hidden)) * 0.2
    layer.mlp.experts.gate_up_proj = mx.random.normal((4, 16, hidden)) * 0.2
    layer.mlp.experts.down_proj = mx.random.normal((4, hidden, 8)) * 0.2
    return Qwen4MTPHead(
        hidden, branches, layer=layer,
        fc_embedding=nn.Linear(hidden, hidden, bias=False),
        fc_hidden=nn.Linear(hidden, hidden, bias=False),
        final_residual=Qwen4GatedResidual(hidden, branches, 4, combine=False),
        lm_head=nn.Linear(hidden, 13, bias=False),
    )


def _inputs(rows=5):
    rng = np.random.default_rng(360)
    return (
        mx.array(rng.standard_normal((1, rows, 32)).astype(np.float32)),
        mx.array(rng.standard_normal((1, rows, 8)).astype(np.float32)),
    )


def test_mtp_fusion_matches_full_width_numpy_reference():
    head = _head()
    hidden, embeddings = _inputs()
    head.pre_fc_norm_hidden.weight = mx.linspace(-0.3, 0.2, 32)
    head.pre_fc_norm_embedding.weight = mx.linspace(-0.2, 0.3, 8)

    def norm(values, weights):
        values = np.asarray(values)
        return values / np.sqrt(np.mean(values ** 2, axis=-1, keepdims=True) + 1e-6) * (
            1 + np.asarray(weights)
        )

    h = norm(hidden, head.pre_fc_norm_hidden.weight).reshape(1, 5, 4, 8)
    e = norm(embeddings, head.pre_fc_norm_embedding.weight)
    expected = h @ np.asarray(head.fc_hidden.weight).T + (
        e @ np.asarray(head.fc_embedding.weight).T
    )[:, :, None, :]
    np.testing.assert_allclose(np.asarray(head.fuse(hidden, embeddings)), expected.reshape(1, 5, 32),
                               rtol=1e-5, atol=1e-6)
    assert head.pre_fc_norm_hidden.group_size is None
    assert head.graph_contract == MTP_GRAPH_CONTRACT


def test_mtp_chunk_and_row_continuations_match_logits_hidden_and_attention_state():
    head = _head()
    hidden, embeddings = _inputs()
    whole = head(hidden, embeddings)
    state = None
    parts = []
    for row in range(hidden.shape[1]):
        part = head(hidden[:, row:row + 1], embeddings[:, row:row + 1], state=state)
        parts.append(part)
        state = part.state
    row_logits = mx.concatenate([part.logits for part in parts], axis=1)
    row_hidden = mx.concatenate([part.widened for part in parts], axis=1)
    np.testing.assert_allclose(np.asarray(row_logits), np.asarray(whole.logits), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(row_hidden), np.asarray(whole.widened), rtol=2e-5, atol=2e-6)
    for name in ("keys", "values", "raw_index_keys", "position_ids"):
        np.testing.assert_allclose(np.asarray(getattr(state.attention, name)),
                                   np.asarray(getattr(whole.state.attention, name)), rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(np.asarray(state.attention.position_ids),
                                  np.broadcast_to(np.arange(1, 6)[None, None, :], (3, 1, 5)))
    assert state.frontier == 5


def test_mtp_rejected_branch_and_failed_append_preserve_prior_attention():
    head = _head()
    hidden, embeddings = _inputs(4)
    first = head(hidden[:, :2], embeddings[:, :2])
    saved = {name: np.asarray(getattr(first.state.attention, name)).copy()
             for name in ("keys", "values", "raw_index_keys", "position_ids")}
    rejected = head(hidden[:, 2:3] * 7, embeddings[:, 2:3], state=first.state)
    mx.eval(rejected.logits)
    correct = head(hidden[:, 2:], embeddings[:, 2:], state=first.state)
    uninterrupted = head(hidden, embeddings)
    np.testing.assert_allclose(np.asarray(correct.logits), np.asarray(uninterrupted.logits[:, 2:]),
                               rtol=2e-5, atol=2e-6)

    def fail(_values):
        raise RuntimeError("injected MTP MLP failure")

    head.layers[0].mlp = fail
    with pytest.raises(RuntimeError, match="injected"):
        head(hidden[:, 2:3], embeddings[:, 2:3], state=first.state)
    for name, values in saved.items():
        np.testing.assert_array_equal(np.asarray(getattr(first.state.attention, name)), values)


def test_mtp_refuses_foreign_state_and_misaligned_hidden_rows():
    head = _head()
    hidden, embeddings = _inputs(1)
    foreign = _head()(hidden, embeddings).state
    with pytest.raises(ValueError, match="another head"):
        head(hidden, embeddings, state=foreign)
    with pytest.raises(ValueError, match="aligned"):
        head(hidden[..., :8], embeddings)
    with pytest.raises(ValueError, match="aligned"):
        head(hidden, embeddings.astype(mx.float16))


def test_mtp_iq2_complete_tiny_head_matches_a_same_weight_decode_reference():
    from mlx.utils import tree_flatten, tree_unflatten

    from moespresso.package.qwen4.mtp_format import mtp_iq2_projection_plan
    from moespresso.package.qwen4.mtp_sidecar import encode_mtp_iq2_projection
    from moespresso.runtime.qwen4.moe import Qwen4ExpertStack, Qwen4TopKRouter
    from moespresso.runtime.qwen4.mtp_iq2 import (
        Qwen4MTPIQ2Experts,
        Qwen4MTPIQ2Linear,
        Qwen4MTPIQ2Router,
    )

    from test_qwen4_mtp_iq2 import _decoded, _load

    head, reference = _head(), _head()
    head.apply(lambda value: value.astype(mx.bfloat16))
    reference.apply(lambda value: value.astype(mx.bfloat16))
    reference.lm_head = head.lm_head
    def leaves(module):
        return tree_flatten(module.leaf_modules(), is_leaf=lambda value: isinstance(value, nn.Module))

    reference_leaves = dict(leaves(reference))
    replacements = []
    kinds = []

    def convert(values, destination):
        weights = np.asarray(values.astype(mx.float32))
        plan = mtp_iq2_projection_plan(weights.shape)
        packed = encode_mtp_iq2_projection(weights, plan, np.ones(weights.shape[-1], np.float32))
        _load(destination, packed)
        decoded = np.concatenate([
            _decoded(streams, part.stored_width)[:, :plan.out_features, :part.end - part.begin]
            for part, streams in zip(plan.slices, packed, strict=True)
        ], axis=-1)
        return mx.array(decoded.reshape(weights.shape)).astype(mx.bfloat16)

    for path, leaf in leaves(head):
        ref = reference_leaves[path]
        if path == "lm_head":
            continue
        if isinstance(leaf, nn.Linear):
            replacement = Qwen4MTPIQ2Linear(*leaf.weight.shape)
            ref.weight = convert(leaf.weight, replacement)
        elif isinstance(leaf, Qwen4TopKRouter):
            replacement = Qwen4MTPIQ2Router(leaf.hidden_size, leaf.num_experts, leaf.top_k)
            ref.weight = convert(leaf.weight, replacement)
        elif isinstance(leaf, Qwen4ExpertStack):
            replacement = Qwen4MTPIQ2Experts(leaf.hidden_size, leaf.intermediate_size, leaf.num_experts)
            ref.gate_up_proj = convert(leaf.gate_up_proj, replacement.gate_up_proj)
            ref.down_proj = convert(leaf.down_proj, replacement.down_proj)
        else:
            continue
        replacements.append((path, replacement))
        kinds.append(type(replacement))
    head.update_modules(tree_unflatten(replacements))
    assert Qwen4MTPIQ2Router in kinds and Qwen4MTPIQ2Experts in kinds
    assert kinds.count(Qwen4MTPIQ2Linear) >= 10
    assert head.lm_head is reference.lm_head
    hidden, embeddings = (value.astype(mx.bfloat16) for value in _inputs(3))
    got, expected = head(hidden, embeddings), reference(hidden, embeddings)
    got_values = np.asarray(got.logits.astype(mx.float32))
    expected_values = np.asarray(expected.logits.astype(mx.float32))
    scale = float(np.max(np.abs(expected_values)))
    assert float(np.max(np.abs(got_values - expected_values))) / scale < 0.04
    np.testing.assert_array_equal(np.argmax(got_values, axis=-1), np.argmax(expected_values, axis=-1))
    continuation = head(hidden[:, :1], embeddings[:, :1], state=got.state)
    assert continuation.state.frontier == 4
    assert continuation.logits.dtype == mx.bfloat16
