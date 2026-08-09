"""Synthetic-weight tests for the MTP drafter.

Builds a tiny random DeepSeek-V4 target graph plus a tiny MTP drafter and
checks the properties that do not need real weights: fusion parity against
a numpy oracle, greedy token identity between speculative and plain
decoding at chained depths including forced depths, the ingest stream
contract, bit-exact drafter cache rewind after a drafting round, and the
sanitize mapping.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from jang_tools.dsv4.mlx_model import Model, ModelArgs

from moespresso.runtime.deepseek_v4.spec_decode import (
    RoundCostModel,
    install_hidden_tap,
    spec_generate,
)
from moespresso.runtime.deepseek_v4.mtp_model import (
    MTPArgs,
    MTPDraftModel,
    sanitize_mtp_weights,
)

VOCAB = 97
HIDDEN = 64
HC = 2


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


def randomize(module: nn.Module, scale: float = 0.08, seed: int = 0) -> None:
    def _rand(a):
        nonlocal seed
        seed += 1
        return mx.random.normal(a.shape, key=mx.random.key(seed)) * scale

    from mlx.utils import tree_map

    module.update(tree_map(_rand, module.parameters()))


def build_tiny_pair(block_size=3, seed=7):
    mx.random.seed(seed)
    args = tiny_model_args()
    target = Model(args)
    randomize(target, seed=seed)

    dargs = MTPArgs(model_args=args, block_size=block_size)
    draft = MTPDraftModel(dargs, embed=target.model.embed, lm_head=target.lm_head)
    randomize(draft, seed=seed + 1000)
    tap = install_hidden_tap(target, draft.tap_layer_ids, draft.tap_transform)
    return target, draft, tap


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


class TestFusionOracle:
    def test_fusion_matches_numpy_oracle(self):
        _, draft, _ = build_tiny_pair(seed=13)
        eps = 1e-6
        m = 3
        hidden = mx.random.normal((1, m, HC, HIDDEN), key=mx.random.key(99))
        toks = mx.array([[1, 5, 9]], dtype=mx.int64)
        got = np.array(draft.fuse(toks, hidden).astype(mx.float32))

        def rms_norm(x, w):
            denom = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
            return (x / denom) * w

        embed_w = np.array(draft.embed.weight.astype(mx.float32))
        enorm_w = np.array(draft.enorm.weight.astype(mx.float32))
        hnorm_w = np.array(draft.hnorm.weight.astype(mx.float32))
        e_proj_w = np.array(draft.e_proj.weight.astype(mx.float32))
        h_proj_w = np.array(draft.h_proj.weight.astype(mx.float32))
        h_np = np.array(hidden.astype(mx.float32))

        e = rms_norm(embed_w[np.array([[1, 5, 9]])], enorm_w) @ e_proj_w.T
        h = rms_norm(h_np, hnorm_w) @ h_proj_w.T
        expected = e[:, :, None, :] + h
        assert got.shape == expected.shape == (1, m, HC, HIDDEN)
        assert np.allclose(got, expected, atol=2e-5, rtol=1e-5)


class TestSpecDecodeIdentity:
    @pytest.mark.parametrize("block_size", [2, 3])
    def test_greedy_identity_with_plain_decode(self, block_size):
        target, draft, tap = build_tiny_pair(block_size=block_size)
        mx.random.seed(3)
        prompt = [
            int(t)
            for t in mx.random.randint(0, VOCAB - 1, (12,), key=mx.random.key(5))
        ]
        max_new = 40

        expected = plain_greedy(target, prompt, max_new)
        result = spec_generate(
            target,
            draft,
            tap,
            prompt,
            max_new_tokens=max_new,
            temperature=0.0,
        )
        assert result.tokens == expected
        assert result.stats.rounds > 0
        # A random draft against a random target must reject sometimes;
        # a zero-rejection run would mean the two arms share a code path.
        assert result.stats.accepted < result.stats.proposed

    def test_greedy_identity_depth1_cap(self):
        # The calibrator's half-block probe has a floor of two submitted
        # tokens, above a depth cap of one, so probing stays disabled for
        # a block_size-1 drafter.
        target, draft, tap = build_tiny_pair(block_size=1)
        prompt = [4, 8, 15, 16]
        expected = plain_greedy(target, prompt, 30)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=30, temperature=0.0,
            probe_every=0,
        )
        assert result.tokens == expected

    @pytest.mark.parametrize("prompt_len", [1, 3])
    def test_greedy_identity_short_prompts(self, prompt_len):
        target, draft, tap = build_tiny_pair()
        prompt = list(range(1, 1 + prompt_len))
        expected = plain_greedy(target, prompt, 25)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=25, temperature=0.0,
        )
        assert result.tokens == expected

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_forced_depth_identity(self, depth):
        # Every round submits exactly `depth` of the three chained draft
        # tokens, so depth 3 exercises the full chain each round.
        target, draft, tap = build_tiny_pair(block_size=3)
        prompt = [2, 5, 8, 1]
        expected = plain_greedy(target, prompt, 30)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=30, temperature=0.0,
            adaptive_cap=True, cost_model=_ForcedCost(depth),
            calibration_warmup=0, probe_every=0,
        )
        assert result.tokens == expected
        assert set(result.stats.submit_length_counts) == {depth}

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
            target,
            draft,
            tap,
            prompt,
            max_new_tokens=30,
            temperature=0.0,
            eos_ids=[eos],
        )
        first = full.index(eos)
        assert result.tokens == full[: first + 1]


class TestIngestContract:
    def _rows(self, n, key):
        return mx.random.normal((1, n, HC, HIDDEN), key=mx.random.key(key))

    def test_token_ids_required(self):
        _, draft, _ = build_tiny_pair()
        state = draft.make_state()
        with pytest.raises(ValueError, match="token ids"):
            draft.ingest(state, self._rows(2, 1), [0, 1], None)

    def test_non_contiguous_positions_rejected(self):
        _, draft, _ = build_tiny_pair()
        state = draft.make_state()
        with pytest.raises(ValueError, match="contiguous"):
            draft.ingest(state, self._rows(2, 2), [0, 2], [1, 2])

    def test_stream_gap_rejected(self):
        _, draft, _ = build_tiny_pair()
        state = draft.make_state()
        draft.ingest(state, self._rows(3, 3), [0, 1, 2], [1, 2, 3])
        with pytest.raises(ValueError, match="does not extend"):
            draft.ingest(state, self._rows(1, 4), [5], [4])

    def test_draft_requires_matching_anchor(self):
        _, draft, _ = build_tiny_pair()
        state = draft.make_state()
        with pytest.raises(ValueError, match="ingested position"):
            draft.draft(state, anchor_token=1, anchor_pos=1)
        draft.ingest(state, self._rows(3, 5), [0, 1, 2], [1, 2, 3])
        with pytest.raises(ValueError, match="anchor"):
            draft.draft(state, anchor_token=1, anchor_pos=5)

    def test_committed_stream_appends_one_row_per_known_next_token(self):
        _, draft, _ = build_tiny_pair()
        state = draft.make_state()
        # The newest hidden row stays pending until the following token
        # arrives, so n committed positions yield n - 1 fused KV rows.
        draft.ingest(state, self._rows(4, 6), [0, 1, 2, 3], [1, 2, 3, 4])
        assert state.cache[0].offset == 3
        assert state.next_pos == 4
        draft.ingest(state, self._rows(1, 7), [4], [5])
        assert state.cache[0].offset == 4
        assert state.next_pos == 5


class TestDrafterRewind:
    def _visible(self, cache):
        if cache.keys is None:
            return None, None
        return (
            cache.keys[..., : cache.offset, :],
            cache.values[..., : cache.offset, :],
        )

    def test_rejected_round_leaves_cache_bit_identical(self):
        _, draft, _ = build_tiny_pair(seed=17)
        state_a = draft.make_state()
        state_b = draft.make_state()
        rows = mx.random.normal((1, 6, HC, HIDDEN), key=mx.random.key(31))
        toks = [3, 7, 11, 2, 9, 4]
        draft.ingest(state_a, rows, list(range(6)), toks)
        draft.ingest(state_b, rows, list(range(6)), toks)

        # A drafting round on state A only; the rewind must leave its
        # committed cache bit-identical to state B, which never drafted.
        proposal = draft.draft(state_a, anchor_token=5, anchor_pos=6)
        mx.eval(proposal.tokens, proposal.logits, proposal.confidence)

        cache_a, cache_b = state_a.cache[0], state_b.cache[0]
        assert cache_a.offset == cache_b.offset
        keys_a, values_a = self._visible(cache_a)
        keys_b, values_b = self._visible(cache_b)
        assert (keys_a is None) == (keys_b is None)
        if keys_a is not None:
            assert mx.array_equal(keys_a, keys_b)
            assert mx.array_equal(values_a, values_b)
        assert mx.array_equal(state_a.pending_hidden, state_b.pending_hidden)
        assert state_a.next_pos == state_b.next_pos

        # Both streams continue identically and must draft identically.
        rows2 = mx.random.normal((1, 2, HC, HIDDEN), key=mx.random.key(32))
        draft.ingest(state_a, rows2, [6, 7], [5, 8])
        draft.ingest(state_b, rows2, [6, 7], [5, 8])
        p_a = draft.draft(state_a, anchor_token=9, anchor_pos=8)
        p_b = draft.draft(state_b, anchor_token=9, anchor_pos=8)
        assert mx.array_equal(p_a.tokens, p_b.tokens)
        assert mx.array_equal(p_a.logits, p_b.logits)

    def test_draft_shapes_and_repeatability(self):
        _, draft, _ = build_tiny_pair(block_size=3, seed=19)
        state = draft.make_state()
        rows = mx.random.normal((1, 4, HC, HIDDEN), key=mx.random.key(41))
        draft.ingest(state, rows, [0, 1, 2, 3], [1, 2, 3, 4])
        first = draft.draft(state, anchor_token=2, anchor_pos=4)
        second = draft.draft(state, anchor_token=2, anchor_pos=4)
        assert first.tokens.shape == (1, 3)
        assert first.logits.shape == (1, 3, VOCAB)
        assert first.logits.dtype == mx.float32
        assert mx.array_equal(first.tokens, second.tokens)
        assert mx.array_equal(first.logits, second.logits)

    def test_chained_logits_are_finite(self):
        _, draft, _ = build_tiny_pair(block_size=3, seed=23)
        state = draft.make_state()
        rows = mx.random.normal((1, 5, HC, HIDDEN), key=mx.random.key(43))
        draft.ingest(state, rows, list(range(5)), [1, 2, 3, 4, 5])
        proposal = draft.draft(state, anchor_token=6, anchor_pos=5)
        logits = np.array(proposal.logits)
        assert np.all(np.isfinite(logits))
        assert not math.isclose(float(np.abs(logits).sum()), 0.0)


class TestConfidenceProxy:
    def _drafted(self, block_size=3, seed=29):
        _, draft, _ = build_tiny_pair(block_size=block_size, seed=seed)
        state = draft.make_state()
        rows = mx.random.normal((1, 5, HC, HIDDEN), key=mx.random.key(47))
        draft.ingest(state, rows, list(range(5)), [1, 2, 3, 4, 5])
        return draft, state

    def test_default_is_confidence_free(self):
        draft, state = self._drafted()
        proposal = draft.draft(state, anchor_token=6, anchor_pos=5)
        assert proposal.confidence is None
        assert proposal.tokens.shape == (1, 3)

    def test_margin_matches_proposal_logits(self, monkeypatch):
        # Greedy drafting picks the argmax per step, so each confidence
        # entry must equal the top-1/top-2 gap of that step's logits row.
        monkeypatch.setenv("MOESPRESSO_DS4_MTP_CONFIDENCE", "1")
        draft, state = self._drafted()
        proposal = draft.draft(state, anchor_token=6, anchor_pos=5)
        repeat = draft.draft(state, anchor_token=6, anchor_pos=5)
        conf = np.array(proposal.confidence)
        logits = np.array(proposal.logits)
        tokens = np.array(proposal.tokens)
        assert conf.shape == (1, 3)
        assert proposal.confidence.dtype == mx.float32
        assert mx.array_equal(proposal.confidence, repeat.confidence)
        for k in range(3):
            row = logits[0, k]
            assert tokens[0, k] == int(np.argmax(row))
            top2 = np.sort(row)[-2:]
            expected = float(top2[1] - top2[0])
            assert conf[0, k] >= 0.0
            assert math.isclose(float(conf[0, k]), expected, abs_tol=2e-5)

    def test_proxy_toggle_leaves_tokens_identical(self, monkeypatch):
        monkeypatch.setenv("MOESPRESSO_DS4_MTP_CONFIDENCE", "1")
        draft, state = self._drafted(seed=31)
        with_conf = draft.draft(state, anchor_token=6, anchor_pos=5)
        monkeypatch.setenv("MOESPRESSO_DS4_MTP_CONFIDENCE", "0")
        without = draft.draft(state, anchor_token=6, anchor_pos=5)
        assert with_conf.confidence is not None
        assert without.confidence is None
        assert mx.array_equal(with_conf.tokens, without.tokens)
        assert mx.array_equal(with_conf.logits, without.logits)

    def test_greedy_identity_with_proxy_on(self, monkeypatch):
        # The proxy feeds the calibrator's scheduling only; token identity
        # with plain decoding must hold with it enabled.
        monkeypatch.setenv("MOESPRESSO_DS4_MTP_CONFIDENCE", "1")
        target, draft, tap = build_tiny_pair(block_size=3, seed=37)
        prompt = [3, 1, 4, 1, 5]
        expected = plain_greedy(target, prompt, 30)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=30, temperature=0.0,
        )
        assert result.tokens == expected


class TestSanitize:
    def _args(self):
        return MTPArgs(model_args=tiny_model_args(), block_size=2)

    def test_module_tree_mapping(self):
        args = self._args()
        n_exp = args.model_args.n_routed_experts
        weights = {
            "mtp.0.attn.wkv.weight": mx.zeros((32, HIDDEN)),
            "mtp.0.attn_norm.weight": mx.ones((HIDDEN,)),
            "mtp.0.ffn_norm.weight": mx.ones((HIDDEN,)),
            "mtp.0.e_proj.weight": mx.zeros((HIDDEN, HIDDEN)),
            "mtp.0.h_proj.weight": mx.zeros((HIDDEN, HIDDEN)),
            "mtp.0.enorm.weight": mx.ones((HIDDEN,)),
            "mtp.0.hnorm.weight": mx.ones((HIDDEN,)),
            "mtp.0.norm.weight": mx.ones((HIDDEN,)),
            "mtp.0.hc_head_fn": mx.zeros((HC, HC * HIDDEN)),
            "mtp.0.hc_attn_fn": mx.zeros(((2 + HC) * HC, HC * HIDDEN)),
            "mtp.0.ffn.gate.weight": mx.zeros((n_exp, HIDDEN)),
            "mtp.0.ffn.shared_experts.w1.weight": mx.zeros((32, HIDDEN)),
            "mtp.0.embed.weight": mx.zeros((VOCAB, HIDDEN)),
            "mtp.0.head.weight": mx.zeros((VOCAB, HIDDEN)),
        }
        for e in range(n_exp):
            for w in ("w1", "w2", "w3"):
                weights[f"mtp.0.ffn.experts.{e}.{w}.weight"] = mx.zeros((2, 2))
        out = sanitize_mtp_weights(weights, args)
        assert "block.self_attn.wkv.weight" in out
        assert "block.input_layernorm.weight" in out
        assert "block.post_attention_layernorm.weight" in out
        assert "e_proj.weight" in out
        assert "h_proj.weight" in out
        assert "enorm.weight" in out and "hnorm.weight" in out
        assert "norm.weight" in out
        assert "hc_head_fn" in out
        assert "block.hc_attn_fn" in out
        assert "block.mlp.gate.weight" in out
        assert "block.mlp.shared_experts.gate_proj.weight" in out
        assert out["block.mlp.switch_mlp.gate_proj.weight"].shape == (n_exp, 2, 2)
        assert not any("embed" in k for k in out)
        assert "head.weight" not in out

    def test_incomplete_experts_fail_closed(self):
        args = self._args()
        weights = {"mtp.0.ffn.experts.0.w1.weight": mx.zeros((2, 2))}
        with pytest.raises((ValueError, KeyError)):
            sanitize_mtp_weights(weights, args)

    def test_second_module_index_rejected(self):
        args = self._args()
        weights = {"mtp.1.attn_norm.weight": mx.ones((HIDDEN,))}
        with pytest.raises(ValueError, match="module index"):
            sanitize_mtp_weights(weights, args)


class TestArgs:
    def test_compress_ratio_padding_and_tap(self):
        args = MTPArgs(model_args=tiny_model_args(), block_size=3)
        assert args.model_args.compress_ratios == [0, 0, 0]
        assert args.tap_layer_id == 1

    def test_missing_target_ratios_rejected(self):
        with pytest.raises(ValueError, match="compress_ratios"):
            MTPArgs(model_args=tiny_model_args(compress_ratios=[0]), block_size=3)

    def test_nonzero_draft_ratio_rejected(self):
        with pytest.raises(ValueError, match="compress_ratio"):
            MTPArgs(
                model_args=tiny_model_args(compress_ratios=[0, 0, 4]), block_size=3,
            )

    def test_block_size_bounds(self):
        with pytest.raises(ValueError, match="block_size"):
            MTPArgs(model_args=tiny_model_args(), block_size=0)
