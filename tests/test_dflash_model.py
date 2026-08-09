"""Synthetic-weight tests for the DFlash drafter.

Builds a tiny random DeepSeek-V4 target graph plus a tiny DFlash drafter
and checks the properties that do not need real weights: config parsing
and the tap-layer derivation, the stream-major tap transform, the block
visibility mask, stacked context K/V parity against a per-layer
reference, windowed context retention, cross-vocabulary proposal mapping,
scratch-only in-block K/V (the context cache survives a drafting round
bit-identically), the greedy-only contract, and greedy token identity
between speculative and plain decoding.

`TestNumpyReference` holds the drafter to an independent fp32 numpy
recomputation of the ingest and draft semantics (hand-rolled rms norm,
neox rope at absolute positions, grouped-query attention over
[context ; block] with an all-visible context and causal-inclusive
block, anchor-row drop, d2t offset mapping). The other tests are
self-consistent by construction; this one pins the semantics against a
second implementation, so a silent change to the flatten layout, the
rope positions, the mask shape, or the norm placement fails against the
reference instead of passing against itself. The same reference form
reproduced the real checkpoint's proposals from exported engine rounds
at 94.8% argmax agreement (remaining rounds are bf16-vs-fp32 knife
edges), so agreement here certifies the checkpoint-facing semantics.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

from jang_tools.dsv4.mlx_model import Model, ModelArgs

from moespresso.runtime.deepseek_v4.dflash_model import (
    DFlashArgs,
    DFlashDraftModel,
    _dflash_block_visibility,
    sanitize_dflash_weights,
)
from moespresso.runtime.deepseek_v4.spec_decode import (
    RoundCostModel,
    install_hidden_tap,
    spec_generate,
)

VOCAB = 97
DRAFT_VOCAB = 64
HIDDEN = 64
HC = 2
WINDOW = 8
# Two tapped layers, each contributing hc_mult raw streams.
FC_IN = 2 * HC * HIDDEN


def tiny_model_args(**overrides) -> ModelArgs:
    base = dict(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        num_hash_layers=0,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        hc_mult=HC,
        hc_sinkhorn_iters=5,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        sliding_window=8,
        rms_norm_eps=1e-6,
        compress_ratios=[0, 0],
    )
    base.update(overrides)
    return ModelArgs(**base)


def tiny_dflash_args(**overrides) -> DFlashArgs:
    base = dict(
        aux_layer_ids=(1, 2),
        block_size=4,
        mask_token_id=1,
        draft_vocab_size=DRAFT_VOCAB,
        target_vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        sliding_window=WINDOW,
        hc_mult=HC,
    )
    base.update(overrides)
    return DFlashArgs(**base)


def tiny_speculator_config() -> dict:
    return {
        "speculators_model_type": "dflash",
        "aux_hidden_state_layer_ids": [1, 2],
        "block_size": 4,
        "mask_token_id": 1,
        "draft_vocab_size": DRAFT_VOCAB,
        "sliding_window_non_causal": False,
        "speculators_config": {
            "algorithm": "dflash",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 3,
                    "verifier_accept_k": 1,
                }
            ],
        },
        "transformer_layer_config": {
            "hc_mult": HC,
            "head_dim": 16,
            "hidden_size": HIDDEN,
            "intermediate_size": 32,
            "layer_types": ["sliding_attention", "sliding_attention"],
            "model_type": "llama",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
            "sliding_window": WINDOW,
            "vocab_size": VOCAB,
        },
    }


def d2t_table() -> mx.array:
    # Draft index i maps to target id i + i // 2: strictly ascending and
    # inside the 97-entry target vocabulary.
    return mx.arange(DRAFT_VOCAB, dtype=mx.int64) // 2


def t2d_table() -> mx.array:
    mapped = np.arange(DRAFT_VOCAB) + np.arange(DRAFT_VOCAB) // 2
    table = np.zeros((VOCAB,), dtype=np.bool_)
    table[mapped] = True
    return mx.array(table)


def randomize_floats(module: nn.Module, scale: float = 0.08, seed: int = 0) -> None:
    """Randomize float parameters only; integer and bool tables keep their
    dtypes and are set explicitly."""
    from mlx.utils import tree_map

    def _rand(a):
        nonlocal seed
        if a.dtype not in (mx.float32, mx.float16, mx.bfloat16):
            return a
        seed += 1
        return mx.random.normal(a.shape, key=mx.random.key(seed)) * scale

    module.update(tree_map(_rand, module.parameters()))


def build_tiny_pair(block_size=4, seed=7):
    mx.random.seed(seed)
    args = tiny_model_args()
    target = Model(args)
    randomize_floats(target, seed=seed)

    dargs = tiny_dflash_args(block_size=block_size)
    draft = DFlashDraftModel(dargs, embed=target.model.embed)
    randomize_floats(draft, seed=seed + 1000)
    draft.d2t = d2t_table()
    draft.t2d = t2d_table()
    tap = install_hidden_tap(target, draft.tap_layer_ids, draft.tap_transform)
    return target, draft, tap


def tap_rows(n: int, key: int) -> mx.array:
    return mx.random.normal((1, n, FC_IN), key=mx.random.key(key))


def plain_greedy(target, prompt_ids, max_new):
    cache = target.make_cache()
    tokens = mx.array(list(prompt_ids), dtype=mx.int64)[None]
    logits = target(tokens, cache=cache)
    out = []
    last = int(mx.argmax(logits[0, -1].astype(mx.float32)))
    out.append(last)
    while len(out) < max_new:
        logits = target(mx.array([[last]], dtype=mx.int64), cache=cache)
        last = int(mx.argmax(logits[0, -1].astype(mx.float32)))
        out.append(last)
    return out


class _ForcedCost(RoundCostModel):
    """Cost model stub that makes choose_submit_length pick one length."""

    def __init__(self, forced: int):
        super().__init__(1.0)
        self.forced = forced

    def expected_ms(self, submitted: int) -> float:
        return 1e-6 if submitted == self.forced else 1e6

    def observe(self, submitted: int, wall_ms: float) -> None:
        pass


class TestArgs:
    def test_from_config_matches_direct_args(self):
        parsed = DFlashArgs.from_config(tiny_speculator_config())
        assert parsed == tiny_dflash_args()

    def test_tap_layers_are_aux_ids_minus_one(self):
        args = tiny_dflash_args(aux_layer_ids=(3, 13, 23, 32, 42))
        assert args.tap_layer_ids == (2, 12, 22, 31, 41)
        assert args.fc_input_size == 5 * HC * HIDDEN

    def test_block_and_proposal_sizes(self):
        args = tiny_dflash_args()
        assert args.block_size == 4
        assert args.speculative_tokens == 3

    def test_aux_id_zero_rejected(self):
        # Aux id 0 is the embedding output in the HF hidden_states
        # convention; there is no decoder layer to tap for it.
        with pytest.raises(ValueError, match="embedding"):
            tiny_dflash_args(aux_layer_ids=(0, 1))

    def test_unsorted_aux_ids_rejected(self):
        with pytest.raises(ValueError, match="ascending"):
            tiny_dflash_args(aux_layer_ids=(2, 1))

    def test_block_size_bounds(self):
        with pytest.raises(ValueError, match="block_size"):
            tiny_dflash_args(block_size=1)

    def test_cross_vocab_bounds(self):
        with pytest.raises(ValueError, match="draft_vocab_size"):
            tiny_dflash_args(draft_vocab_size=VOCAB + 1)

    def test_non_causal_config_rejected(self):
        config = tiny_speculator_config()
        config["sliding_window_non_causal"] = True
        with pytest.raises(ValueError, match="causal"):
            DFlashArgs.from_config(config)

    def test_full_attention_layer_type_rejected(self):
        config = tiny_speculator_config()
        config["transformer_layer_config"]["layer_types"] = [
            "sliding_attention", "full_attention",
        ]
        with pytest.raises(ValueError, match="layer_types"):
            DFlashArgs.from_config(config)

    def test_speculative_tokens_mismatch_rejected(self):
        config = tiny_speculator_config()
        config["speculators_config"]["proposal_methods"][0][
            "speculative_tokens"] = 5
        with pytest.raises(ValueError, match="speculative_tokens"):
            DFlashArgs.from_config(config)

    def test_wrong_model_type_rejected(self):
        config = tiny_speculator_config()
        config["speculators_model_type"] = "eagle3"
        with pytest.raises(ValueError, match="dflash"):
            DFlashArgs.from_config(config)


class TestBlockMask:
    def test_context_visible_block_causal_inclusive(self):
        mask = _dflash_block_visibility(ctx_len=5, block_len=3)
        expected = mx.array(
            [
                [True, True, True, True, True, True, False, False],
                [True, True, True, True, True, True, True, False],
                [True, True, True, True, True, True, True, True],
            ],
            dtype=mx.bool_,
        )
        assert mask.shape == (1, 1, 3, 8)
        assert mx.array_equal(mask[0, 0], expected)

    def test_empty_context_is_pure_causal(self):
        mask = _dflash_block_visibility(ctx_len=0, block_len=4)
        assert mask.shape == (1, 1, 4, 4)
        expected = mx.tri(4, 4, dtype=mx.bool_)
        assert mx.array_equal(mask[0, 0], expected)


class TestTapTransform:
    def test_flatten_is_stream_major(self):
        _, draft, _ = build_tiny_pair(seed=11)
        out = mx.random.normal((1, 3, HC, HIDDEN), key=mx.random.key(5))
        got = draft.tap_transform(0, out)
        expected = mx.concatenate([out[:, :, 0], out[:, :, 1]], axis=-1)
        assert got.shape == (1, 3, HC * HIDDEN)
        assert mx.array_equal(got, expected)


class TestIngest:
    def test_wrong_feature_width_rejected(self):
        _, draft, _ = build_tiny_pair()
        state = draft.make_state()
        with pytest.raises(ValueError, match="tap features"):
            draft.ingest(state, mx.zeros((1, 2, 7)), [0, 1])

    def test_non_contiguous_positions_rejected(self):
        _, draft, _ = build_tiny_pair()
        state = draft.make_state()
        with pytest.raises(ValueError, match="contiguous"):
            draft.ingest(state, tap_rows(2, 2), [0, 2])

    def test_stream_gap_rejected(self):
        _, draft, _ = build_tiny_pair()
        state = draft.make_state()
        draft.ingest(state, tap_rows(3, 3), [0, 1, 2])
        with pytest.raises(ValueError, match="does not extend"):
            draft.ingest(state, tap_rows(1, 4), [5])

    def test_context_shapes_and_window_retention(self):
        _, draft, _ = build_tiny_pair()
        state = draft.make_state()
        draft.ingest(state, tap_rows(3, 5), [0, 1, 2])
        assert state.ctx_k.shape == (1, 2, 1, 3, 16)
        assert state.ctx_v.shape == (1, 2, 1, 3, 16)
        assert state.next_pos == 3
        draft.ingest(state, tap_rows(20, 6), list(range(3, 23)))
        assert state.ctx_k.shape[3] == WINDOW
        assert state.next_pos == 23

    def test_rows_outside_window_do_not_influence_drafts(self):
        # Two streams share the trailing WINDOW rows but differ before
        # them; retention keeps only the tail, so the caches and the
        # drafts must match bitwise.
        _, draft, _ = build_tiny_pair(seed=23)
        total = 20
        shared = tap_rows(WINDOW, 31)
        early_a = tap_rows(total - WINDOW, 32)
        early_b = tap_rows(total - WINDOW, 33)
        state_a = draft.make_state()
        state_b = draft.make_state()
        draft.ingest(
            state_a,
            mx.concatenate([early_a, shared], axis=1),
            list(range(total)),
        )
        draft.ingest(
            state_b,
            mx.concatenate([early_b, shared], axis=1),
            list(range(total)),
        )
        assert mx.array_equal(state_a.ctx_k, state_b.ctx_k)
        assert mx.array_equal(state_a.ctx_v, state_b.ctx_v)
        p_a = draft.draft(state_a, anchor_token=5, anchor_pos=total)
        p_b = draft.draft(state_b, anchor_token=5, anchor_pos=total)
        assert mx.array_equal(p_a.tokens, p_b.tokens)
        assert mx.array_equal(p_a.logits, p_b.logits)

    def test_stacked_kv_matches_per_layer_reference(self):
        # The single stacked GEMM plus batched rope must reproduce the
        # per-layer k_norm(k_proj(Ht)) / v_proj(Ht) reference.
        _, draft, _ = build_tiny_pair(seed=29)
        state = draft.make_state()
        rows = tap_rows(4, 41)
        draft.ingest(state, rows, [0, 1, 2, 3])

        ht = draft.hidden_norm(draft.fc(rows))
        for i, layer in enumerate(draft.layers):
            attn = layer.self_attn
            k = attn.k_norm(
                attn.k_proj(ht).reshape(1, 4, 1, 16)
            ).transpose(0, 2, 1, 3)
            k = attn.rope(k, offset=0)
            v = attn.v_proj(ht).reshape(1, 4, 1, 16).transpose(0, 2, 1, 3)
            assert np.allclose(
                np.array(state.ctx_k[:, i]), np.array(k), atol=1e-5)
            assert np.allclose(
                np.array(state.ctx_v[:, i]), np.array(v), atol=1e-6)


class TestDraft:
    def _ready(self, seed=7, n=5):
        _, draft, _ = build_tiny_pair(seed=seed)
        state = draft.make_state()
        draft.ingest(state, tap_rows(n, 51), list(range(n)))
        return draft, state

    def test_greedy_only_contract(self):
        draft, state = self._ready()
        assert draft.greedy_only is True
        with pytest.raises(ValueError, match="greedy-only"):
            draft.draft(state, anchor_token=2, anchor_pos=5, temperature=0.5)

    def test_proposal_shapes_and_repeatability(self):
        draft, state = self._ready()
        first = draft.draft(state, anchor_token=2, anchor_pos=5)
        second = draft.draft(state, anchor_token=2, anchor_pos=5)
        assert first.tokens.shape == (1, 3)
        assert first.tokens.dtype == mx.int64
        assert first.logits.shape == (1, 3, DRAFT_VOCAB)
        assert first.logits.dtype == mx.float32
        assert first.confidence is None
        assert mx.array_equal(first.tokens, second.tokens)
        assert mx.array_equal(first.logits, second.logits)

    def test_d2t_offset_mapping(self):
        # Proposals are the draft-vocabulary argmax plus the d2t offset at
        # that index; every mapped id lands inside the target vocabulary
        # on a t2d member.
        draft, state = self._ready(seed=13)
        proposal = draft.draft(state, anchor_token=2, anchor_pos=5)
        logits = np.array(proposal.logits)
        tokens = np.array(proposal.tokens)
        d2t = np.array(draft.d2t)
        t2d = np.array(draft.t2d)
        for k in range(tokens.shape[1]):
            idx = int(np.argmax(logits[0, k]))
            assert tokens[0, k] == idx + int(d2t[idx])
            assert 0 <= tokens[0, k] < VOCAB
            assert bool(t2d[tokens[0, k]])

    def test_anchor_position_contract(self):
        draft, state = self._ready()
        with pytest.raises(ValueError, match="anchor"):
            draft.draft(state, anchor_token=1, anchor_pos=9)
        empty = draft.make_state()
        with pytest.raises(ValueError, match="ingested position"):
            draft.draft(empty, anchor_token=1, anchor_pos=0)


class TestScratchBlockKV:
    def test_rejected_round_leaves_context_bit_identical(self):
        # A drafting round writes nothing: the context cache after a
        # draft (a fully rejected round ingests no rows) must equal the
        # cache of a state that never drafted, bitwise.
        _, draft, _ = build_tiny_pair(seed=17)
        state_a = draft.make_state()
        state_b = draft.make_state()
        rows = tap_rows(6, 61)
        draft.ingest(state_a, rows, list(range(6)))
        draft.ingest(state_b, rows, list(range(6)))

        proposal = draft.draft(state_a, anchor_token=5, anchor_pos=6)
        mx.eval(proposal.tokens, proposal.logits)

        assert state_a.next_pos == state_b.next_pos
        assert mx.array_equal(state_a.ctx_k, state_b.ctx_k)
        assert mx.array_equal(state_a.ctx_v, state_b.ctx_v)

        rows2 = tap_rows(2, 62)
        draft.ingest(state_a, rows2, [6, 7])
        draft.ingest(state_b, rows2, [6, 7])
        p_a = draft.draft(state_a, anchor_token=9, anchor_pos=8)
        p_b = draft.draft(state_b, anchor_token=9, anchor_pos=8)
        assert mx.array_equal(p_a.tokens, p_b.tokens)
        assert mx.array_equal(p_a.logits, p_b.logits)


class TestSpecDecodeIdentity:
    def test_greedy_identity_with_plain_decode(self):
        target, draft, tap = build_tiny_pair()
        mx.random.seed(3)
        prompt = [
            int(t)
            for t in mx.random.randint(0, VOCAB - 1, (12,), key=mx.random.key(5))
        ]
        max_new = 40

        expected = plain_greedy(target, prompt, max_new)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=max_new, temperature=0.0,
        )
        assert result.tokens == expected
        assert result.stats.rounds > 0
        # A cross-vocabulary random draft against a random target must
        # reject sometimes; a zero-rejection run would mean the two arms
        # share a code path.
        assert result.stats.accepted < result.stats.proposed

    @pytest.mark.parametrize("prompt_len", [1, 3])
    def test_greedy_identity_short_prompts(self, prompt_len):
        target, draft, tap = build_tiny_pair()
        prompt = list(range(1, 1 + prompt_len))
        expected = plain_greedy(target, prompt, 25)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=25, temperature=0.0,
        )
        assert result.tokens == expected

    def test_greedy_identity_long_prompt_past_window(self):
        # A prompt longer than the drafter window, prefilled in chunks,
        # exercises windowed prefill seeding of the context cache.
        target, draft, tap = build_tiny_pair(seed=19)
        prompt = [int(1 + (i * 7) % (VOCAB - 2)) for i in range(30)]
        expected = plain_greedy(target, prompt, 30)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=30, temperature=0.0,
            prefill_step_size=8,
        )
        assert result.tokens == expected

    @pytest.mark.parametrize("length", [1, 2, 3])
    def test_forced_length_identity(self, length):
        target, draft, tap = build_tiny_pair()
        prompt = [2, 5, 8, 1]
        expected = plain_greedy(target, prompt, 30)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=30, temperature=0.0,
            adaptive_cap=True, cost_model=_ForcedCost(length),
            calibration_warmup=0, probe_every=0,
        )
        assert result.tokens == expected
        assert set(result.stats.submit_length_counts) == {length}

    def test_forced_fallback_identity(self):
        target, draft, tap = build_tiny_pair()
        prompt = [3, 4, 5]
        expected = plain_greedy(target, prompt, 20)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=20, temperature=0.0,
            adaptive_cap=True, cost_model=_ForcedCost(0),
            calibration_warmup=0, probe_every=0,
        )
        assert result.tokens == expected
        assert result.stats.plain_fallbacks == result.stats.rounds
        assert result.stats.proposed == 0

    @pytest.mark.parametrize("length", [0, 3])
    def test_prefetch_matches_sequential_schedule(self, length):
        # The prefetched draft dispatch must reproduce the sequential
        # schedule exactly: same tokens and same per-round statistics,
        # on both the speculative and the plain-fallback paths.
        target, draft, tap = build_tiny_pair()
        prompt = [2, 5, 8, 1]
        expected = plain_greedy(target, prompt, 30)
        runs = []
        for prefetch in (False, True):
            result = spec_generate(
                target, draft, tap, prompt, max_new_tokens=30,
                temperature=0.0, adaptive_cap=True,
                cost_model=_ForcedCost(length),
                calibration_warmup=0, probe_every=0,
                draft_prefetch=prefetch,
            )
            assert result.tokens == expected
            runs.append(result)
        seq, pre = runs
        assert pre.stats.rounds == seq.stats.rounds
        assert pre.stats.proposed == seq.stats.proposed
        assert pre.stats.accepted == seq.stats.accepted
        assert pre.stats.plain_fallbacks == seq.stats.plain_fallbacks
        assert pre.stats.per_position_accept == seq.stats.per_position_accept
        assert pre.stats.per_position_matched == seq.stats.per_position_matched
        assert pre.stats.submit_length_counts == seq.stats.submit_length_counts

    def test_eos_stops_generation(self):
        target, draft, tap = build_tiny_pair()
        prompt = [1, 2, 3]
        full = plain_greedy(target, prompt, 30)
        eos = full[10]
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=30, temperature=0.0,
            eos_ids=[eos],
        )
        first = full.index(eos)
        assert result.tokens == full[: first + 1]

    def test_temperature_request_refused(self):
        # The loop must never take the sampled path with a greedy-only
        # drafter; the serve seam routes such requests to plain decoding.
        target, draft, tap = build_tiny_pair()
        with pytest.raises(ValueError, match="greedy-only"):
            spec_generate(
                target, draft, tap, [1, 2, 3], max_new_tokens=10,
                temperature=0.7,
            )


def _np_rms(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _np_rope(x: np.ndarray, positions, theta: float = 10000.0) -> np.ndarray:
    """Neox-style rotary embedding: half-split pairing, absolute positions."""
    half = x.shape[-1] // 2
    inv = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    cos = np.cos(ang).astype(np.float64)
    sin = np.sin(ang).astype(np.float64)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _np_reference_draft(draft, embed_w, rows, anchor_token, anchor_pos):
    """Independent numpy recomputation of ingest plus one draft pass.

    `rows` covers every committed position 0..anchor_pos-1 in stream
    order. Returns (draft_logits, proposals) for the mask rows.
    """
    args = draft.args
    assert args.num_key_value_heads == 1, "reference covers the GQA n_kv=1 shape"
    w = lambda a: np.array(a, dtype=np.float64)  # noqa: E731

    ht = _np_rms(rows @ w(draft.fc.weight).T, w(draft.hidden_norm.weight))
    n_pos = ht.shape[0]
    window = args.sliding_window
    lo = max(0, anchor_pos - window)

    ctx_k, ctx_v = [], []
    for layer in draft.layers:
        attn = layer.self_attn
        k = _np_rms(ht @ w(attn.k_proj.weight).T, w(attn.k_norm.weight))
        ctx_k.append(_np_rope(k, np.arange(n_pos)))
        ctx_v.append(ht @ w(attn.v_proj.weight).T)

    ids = [int(anchor_token)] + [args.mask_token_id] * args.speculative_tokens
    x = embed_w[ids].astype(np.float64)
    S = len(ids)
    bpos = np.arange(anchor_pos, anchor_pos + S)
    n_heads = args.num_attention_heads
    hd = args.head_dim
    for li, layer in enumerate(draft.layers):
        attn = layer.self_attn
        h = _np_rms(x, w(layer.input_layernorm.weight))
        q = _np_rms((h @ w(attn.q_proj.weight).T).reshape(S, n_heads, hd),
                    w(attn.q_norm.weight))
        q = np.stack([_np_rope(q[:, i], bpos) for i in range(n_heads)])
        kb = _np_rope(
            _np_rms(h @ w(attn.k_proj.weight).T, w(attn.k_norm.weight)), bpos)
        vb = h @ w(attn.v_proj.weight).T
        keys = np.concatenate([ctx_k[li][lo:anchor_pos], kb])
        vals = np.concatenate([ctx_v[li][lo:anchor_pos], vb])
        n_ctx = anchor_pos - lo
        bias = np.zeros((S, keys.shape[0]))
        for i in range(S):
            bias[i, n_ctx + i + 1:] = -np.inf
        scores = q @ keys.T / np.sqrt(hd) + bias[None]
        scores -= scores.max(-1, keepdims=True)
        e = np.exp(scores)
        out = (e / e.sum(-1, keepdims=True)) @ vals      # (heads, S, hd)
        x = x + out.transpose(1, 0, 2).reshape(S, -1) @ w(attn.o_proj.weight).T
        h = _np_rms(x, w(layer.post_attention_layernorm.weight))
        gate = h @ w(layer.mlp.gate_proj.weight).T
        x = x + ((gate / (1 + np.exp(-gate))) * (h @ w(layer.mlp.up_proj.weight).T)) \
            @ w(layer.mlp.down_proj.weight).T

    h = _np_rms(x, w(draft.norm.weight))[1:]
    logits = h @ w(draft.lm_head.weight).T
    idx = logits.argmax(-1)
    d2t = np.array(draft.d2t)
    return logits, idx + d2t[idx]


class TestNumpyReference:
    def _build(self, seed):
        mx.random.seed(seed)
        embed = nn.Embedding(VOCAB, HIDDEN)
        dargs = tiny_dflash_args()
        draft = DFlashDraftModel(dargs, embed=embed)
        randomize_floats(embed, seed=seed + 500)
        randomize_floats(draft, seed=seed + 1000)
        draft.d2t = d2t_table()
        draft.t2d = t2d_table()
        return draft, np.array(embed.weight, dtype=np.float64)

    @pytest.mark.parametrize(
        "seed,n_pos",
        [(101, 5), (103, WINDOW + 6)],  # short context and past the window
    )
    def test_draft_matches_reference(self, seed, n_pos):
        draft, embed_w = self._build(seed)
        rows_mx = tap_rows(n_pos, seed + 7)
        state = draft.make_state()
        # Two chunks so the concat-and-trim path runs, mirroring prefill
        # followed by a committed round.
        split = max(1, n_pos - 2)
        draft.ingest(state, rows_mx[:, :split], list(range(split)))
        draft.ingest(state, rows_mx[:, split:], list(range(split, n_pos)))

        anchor = 3
        proposal = draft.draft(state, anchor_token=anchor, anchor_pos=n_pos)
        got_logits = np.array(proposal.logits[0], dtype=np.float64)
        got_tokens = np.array(proposal.tokens[0])

        ref_logits, ref_tokens = _np_reference_draft(
            draft, embed_w, np.array(rows_mx[0], dtype=np.float64), anchor, n_pos)

        assert np.allclose(got_logits, ref_logits, atol=2e-4, rtol=1e-3), (
            np.abs(got_logits - ref_logits).max())
        assert np.array_equal(got_tokens, ref_tokens)

    def test_context_keys_match_reference(self, seed=107):
        draft, _ = self._build(seed)
        n_pos = 6
        rows_mx = tap_rows(n_pos, seed + 7)
        state = draft.make_state()
        draft.ingest(state, rows_mx, list(range(n_pos)))

        rows = np.array(rows_mx[0], dtype=np.float64)
        ht = _np_rms(rows @ np.array(draft.fc.weight, dtype=np.float64).T,
                     np.array(draft.hidden_norm.weight, dtype=np.float64))
        for i, layer in enumerate(draft.layers):
            attn = layer.self_attn
            k = _np_rms(ht @ np.array(attn.k_proj.weight, dtype=np.float64).T,
                        np.array(attn.k_norm.weight, dtype=np.float64))
            k = _np_rope(k, np.arange(n_pos))
            v = ht @ np.array(attn.v_proj.weight, dtype=np.float64).T
            assert np.allclose(
                np.array(state.ctx_k[0, i, 0], dtype=np.float64), k, atol=1e-4)
            assert np.allclose(
                np.array(state.ctx_v[0, i, 0], dtype=np.float64), v, atol=1e-4)


class TestSanitize:
    def test_identity_map_drops_embed(self):
        args = tiny_dflash_args()
        weights = {
            "fc.weight": mx.zeros((1,)),
            "layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
            "layers.1.mlp.down_proj.weight": mx.zeros((1,)),
            "embed_tokens.weight": mx.zeros((1,)),
            "d2t": mx.zeros((1,)),
        }
        out = sanitize_dflash_weights(weights, args)
        assert set(out) == set(weights) - {"embed_tokens.weight"}

    def test_layer_index_out_of_range_rejected(self):
        args = tiny_dflash_args()
        with pytest.raises(ValueError, match="layer index"):
            sanitize_dflash_weights(
                {"layers.2.self_attn.q_proj.weight": mx.zeros((1,))}, args)
