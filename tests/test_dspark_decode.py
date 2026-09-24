"""Synthetic-weight tests for the speculative decoding loop.

Builds a tiny random DeepSeek-V4 target graph plus a tiny DSpark drafter
and checks the properties that do not need real weights: greedy token
identity between speculative and plain decoding, the drafter protocol
seam, the window ring buffer, capsule bit identity, nonzero-frontier resume,
and the statistical losslessness of the sampled acceptance rule.
"""

import copy
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import pytest

from jang_tools.dsv4.mlx_model import Model, ModelArgs

from moespresso.runtime.deepseek_v4.spec_decode import (
    DrafterStateCapsule,
    DraftProposal,
    RoundCostModel,
    SpecStats,
    choose_submit_length,
    greedy_accept,
    install_hidden_tap,
    sampled_accept,
    spec_generate,
)
from moespresso.runtime.deepseek_v4.dspark_model import (
    DSparkArgs,
    DSparkBlock,
    DSparkDraftModel,
    DSparkWindowCache,
    sanitize_dspark_weights,
)

VOCAB = 97
NOISE = VOCAB - 1


def tiny_model_args(**overrides) -> ModelArgs:
    base = dict(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=64,
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
        hc_mult=2,
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

    dargs = DSparkArgs(
        model_args=args,
        n_mtp_layers=2,
        block_size=block_size,
        noise_token_id=NOISE,
        target_layer_ids=(0, 1),
        markov_rank=8,
    )
    draft = DSparkDraftModel(dargs, embed=target.model.embed, lm_head=target.lm_head)
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


class TestWindowCache:
    def test_ring_layout_and_mask(self):
        win = DSparkWindowCache(window=4, head_dim=2)
        rows = mx.arange(6, dtype=mx.float32).reshape(1, 3, 2)
        win.write(rows, [0, 1, 2])
        mask = win.valid_mask()
        assert mask is not None
        assert [bool(v) for v in mask] == [True, True, True, False]

        more = mx.arange(10, 14, dtype=mx.float32).reshape(1, 2, 2)
        win.write(more, [3, 4])
        assert win.valid_mask() is None
        # Position 4 wraps onto slot 0.
        assert mx.allclose(win.kv[0, 0, 0], mx.array([12.0, 13.0]))
        assert mx.allclose(win.kv[0, 0, 3], mx.array([10.0, 11.0]))

    def test_non_contiguous_write_rejected(self):
        win = DSparkWindowCache(window=4, head_dim=2)
        with pytest.raises(ValueError):
            win.write(mx.zeros((1, 1, 2)), [3])

    def test_fresh_full_window_write_allowed(self):
        win = DSparkWindowCache(window=2, head_dim=2)
        win.write(mx.zeros((1, 2, 2)), [10, 11])
        assert win.last_pos == 11
        assert win.valid_mask() is None


def _capsule_drafter() -> DSparkDraftModel:
    args = tiny_model_args()
    return DSparkDraftModel(
        DSparkArgs(
            model_args=args,
            n_mtp_layers=2,
            block_size=3,
            noise_token_id=NOISE,
            target_layer_ids=(0, 1),
            markov_rank=8,
        )
    )


def _write_window_chunks(state, chunks, *, salt=0):
    head_dim = int(state[0].kv.shape[-1])
    for positions in chunks:
        positions = list(positions)
        for stage, window in enumerate(state):
            rows = mx.stack(
                [
                    mx.arange(head_dim, dtype=mx.float32)
                    + int(salt + stage * 10_000 + position * head_dim)
                    for position in positions
                ],
                axis=0,
            )[None]
            window.write(rows, positions)
    mx.eval(*(window.kv for window in state))


def _assert_float32_bits_equal(left, right):
    assert left.dtype == right.dtype == mx.float32
    assert bool(mx.array_equal(left.view(mx.uint32), right.view(mx.uint32)).item())


@pytest.mark.parametrize(
    "chunks,frontier",
    [
        (((0, 1, 2),), 3),
        ((tuple(range(0, 6)), tuple(range(6, 11))), 11),
        ((tuple(range(20, 28)),), 28),
    ],
    ids=("pre_wrap", "wrapped", "nonzero_absolute_frontier"),
)
def test_dspark_state_capsule_round_trip_is_bit_exact_and_isolated(
    chunks, frontier
):
    drafter = _capsule_drafter()
    assert drafter.state_capsule_kind == "deepseek_v4_dspark_window_state"
    assert drafter.state_capsule_schema_major == 1
    assert drafter.state_capsule_schema_minor == 0
    state = drafter.make_state()
    _write_window_chunks(state, chunks)
    # Arithmetic copies such as ``state + 0`` erase this sign bit while
    # remaining numerically equal. The capsule contract is bit identity.
    state[0].kv[0, 0, 0, 0] = mx.array(0x80000000, dtype=mx.uint32).view(mx.float32)
    mx.eval(state[0].kv)

    capsule = drafter.export_state(state)
    assert isinstance(capsule, DrafterStateCapsule)
    assert capsule.frontier == frontier
    assert drafter.state_frontier(state) == frontier
    assert capsule.nbytes == drafter.state_nbytes(state) == 2048
    assert isinstance(capsule.metadata, tuple)
    metadata_copy = capsule.metadata_dict()
    metadata_copy["window"] = -1
    assert capsule.metadata_dict()["window"] == 8

    exported = tuple(
        mx.array(tensor.tolist(), dtype=tensor.dtype) for tensor in capsule.tensors
    )
    mx.eval(*exported)
    _write_window_chunks(state, ((frontier,),), salt=1_000_000)
    for expected, tensor in zip(exported, capsule.tensors):
        _assert_float32_bits_equal(expected, tensor)

    restored_a = drafter.import_state(capsule)
    restored_b = drafter.import_state(capsule)
    for original, restored in zip(capsule.tensors, restored_a):
        assert restored.last_pos == frontier - 1
        _assert_float32_bits_equal(original, restored.kv)
    for left, right in zip(restored_a, restored_b):
        mask_left = left.valid_mask()
        mask_right = right.valid_mask()
        if mask_left is None:
            assert mask_right is None
        else:
            assert bool(mx.array_equal(mask_left, mask_right).item())

    continuation = ((frontier,),)
    _write_window_chunks(restored_a, continuation, salt=2_000_000)
    for expected, capsule_tensor, untouched in zip(
        exported, capsule.tensors, restored_b
    ):
        _assert_float32_bits_equal(expected, capsule_tensor)
        _assert_float32_bits_equal(expected, untouched.kv)
    _write_window_chunks(restored_b, continuation, salt=2_000_000)
    for left, right in zip(restored_a, restored_b):
        assert left.last_pos == right.last_pos == frontier
        _assert_float32_bits_equal(left.kv, right.kv)

    capsule.tensors[0][0, 0, 0, 0] = 123.0
    mx.eval(capsule.tensors[0])
    assert not bool(
        mx.array_equal(
            exported[0].view(mx.uint32), capsule.tensors[0].view(mx.uint32)
        ).item()
    )
    _assert_float32_bits_equal(restored_b[0].kv, restored_a[0].kv)


def test_dspark_state_capsule_materializes_round_sized_lazy_ingest():
    _target, drafter, _tap = build_tiny_pair(block_size=3)
    hidden = drafter.args.model_args.hidden_size
    n_targets = len(drafter.args.target_layer_ids)
    rows = mx.random.normal(
        (1, 4, hidden * n_targets), key=mx.random.key(113)
    )
    positions = [0, 1, 2, 3]

    lazy_state = drafter.make_state()
    drafter.ingest(lazy_state, rows, positions, positions)
    capsule = drafter.export_state(lazy_state)

    reference_state = drafter.make_state()
    drafter.ingest(reference_state, rows, positions, positions)
    mx.eval(*(window.kv for window in reference_state))

    assert capsule.frontier == 4
    for tensor, reference in zip(capsule.tensors, reference_state):
        _assert_float32_bits_equal(tensor, reference.kv)


def test_dspark_state_capsule_fails_closed_on_state_or_payload_drift():
    drafter = _capsule_drafter()
    state = drafter.make_state()
    _write_window_chunks(state, ((0, 1, 2),))
    capsule = drafter.export_state(state)

    state[0].last_pos -= 1
    with pytest.raises(ValueError, match="frontier"):
        drafter.export_state(state)

    with pytest.raises(ValueError, match="kind"):
        drafter.import_state(replace(capsule, kind="unknown"))
    with pytest.raises(ValueError, match="schema"):
        drafter.import_state(replace(capsule, schema_major=2))
    with pytest.raises(ValueError, match="shape"):
        drafter.import_state(
            replace(capsule, tensors=(capsule.tensors[0][:, :, :-1],) + capsule.tensors[1:])
        )
    with pytest.raises(ValueError, match="dtype"):
        drafter.import_state(
            replace(
                capsule,
                tensors=(capsule.tensors[0].astype(mx.float16),) + capsule.tensors[1:],
            )
        )


class TestPrefillIngestDispatch:
    def test_prefill_shaped_ingest_matches_chunked_ingests(self):
        """The prefill-shaped ingest dispatch is scheduling only.

        One ingest above the dispatch threshold must leave every window
        bit-identical to the same rows fed as round-sized ingests, which
        stay on the lazy path.
        """
        _target, draft, _tap = build_tiny_pair(block_size=3)
        hidden = draft.args.model_args.hidden_size
        n_targets = len(draft.args.target_layer_ids)
        rows = mx.random.normal(
            (1, 8, hidden * n_targets), key=mx.random.key(31))

        assert 8 > draft.args.block_size + 1
        big = draft.make_state()
        draft.ingest(big, rows, list(range(8)), list(range(8)))

        small = draft.make_state()
        draft.ingest(small, rows[:, :4], [0, 1, 2, 3], [0, 1, 2, 3])
        draft.ingest(small, rows[:, 4:], [4, 5, 6, 7], [4, 5, 6, 7])

        for wa, wb in zip(big, small):
            mx.eval(wa.kv, wb.kv)
            assert wa.last_pos == wb.last_pos
            assert bool(mx.array_equal(wa.kv, wb.kv).item())


class TestAcceptRules:
    def test_greedy_accept_prefix_and_correction(self):
        logits = mx.array(
            [
                [[0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
            ]
        )
        # Target argmax rows: 1, 2, 0, 1
        res = greedy_accept([1, 2, 2], logits)
        assert res.accepted == 2
        assert res.next_token == 0
        assert res.emitted == [1, 2, 0]

    def test_greedy_accept_bonus(self):
        logits = mx.array([[[0.0, 5.0], [5.0, 0.0], [0.0, 5.0]]])
        res = greedy_accept([1, 0], logits)
        assert res.accepted == 2
        assert res.emitted == [1, 0, 1]

    def test_greedy_accept_matches_past_first_rejection(self):
        # Position matches keep recording after the prefix breaks: the
        # teacher-forced accuracy view is independent of acceptance.
        logits = mx.array(
            [
                [[0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
            ]
        )
        # Target argmax rows: 1, 2, 0, 1
        res = greedy_accept([1, 0, 0], logits)
        assert res.accepted == 1
        assert res.position_matches == [True, False, True]

    def test_stats_record_matched_and_accepted_diverge(self):
        stats = SpecStats()
        stats.record(3, 1, [True, False, True])
        stats.record(3, 3, [True, True, True])
        stats.record(2, 0, [False, True])
        assert stats.per_position_offered == [3, 3, 2]
        assert stats.per_position_accept == [2, 1, 1]
        assert stats.per_position_matched == [2, 2, 2]

    def test_stats_record_without_matches(self):
        # Sampled acceptance records no argmax matches; the matched
        # counters stay aligned with the offered counters at zero.
        stats = SpecStats()
        stats.record(2, 1)
        assert stats.per_position_offered == [1, 1]
        assert stats.per_position_matched == [0, 0]

    def test_sampled_accept_recovers_target_distribution(self):
        # Draft and target disagree; the marginal of the first emitted token
        # must match the target distribution regardless.
        mx.random.seed(11)
        vocab = 5
        target_logits = mx.array([0.9, -0.3, 0.4, -1.2, 0.1])
        draft_logits = mx.array([-0.8, 1.1, -0.2, 0.7, -0.5])
        p_t = mx.softmax(target_logits)
        trials = 20000
        counts = [0] * vocab
        d_probs = mx.softmax(draft_logits)
        draws = mx.random.categorical(
            mx.log(d_probs)[None, :].astype(mx.float32), num_samples=trials
        )
        mx.eval(draws)
        for i in range(trials):
            token = int(draws[0, i])
            res = sampled_accept(
                [token],
                draft_logits[None, None, :],
                mx.stack([target_logits, target_logits])[None],
                temperature=1.0,
            )
            counts[res.emitted[0]] += 1
        mx.eval(p_t)
        for v in range(vocab):
            observed = counts[v] / trials
            expected = float(p_t[v])
            assert abs(observed - expected) < 0.02, (v, observed, expected)


class TestSpecDecodeIdentity:
    @pytest.mark.parametrize("prompt_len", [3, 12])
    def test_greedy_identity_with_plain_decode(self, prompt_len):
        target, draft, tap = build_tiny_pair()
        mx.random.seed(3)
        prompt = [
            int(t)
            for t in mx.random.randint(0, VOCAB - 1, (prompt_len,), key=mx.random.key(5))
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

    def test_confidence_truncation_preserves_identity(self):
        target, draft, tap = build_tiny_pair()
        prompt = [1, 2, 3, 4]
        expected = plain_greedy(target, prompt, 25)
        result = spec_generate(
            target,
            draft,
            tap,
            prompt,
            max_new_tokens=25,
            temperature=0.0,
            confidence_threshold=0.5,
        )
        assert result.tokens == expected

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


class _ForcedCost(RoundCostModel):
    """Cost model stub that makes choose_submit_length pick one length."""

    def __init__(self, forced: int):
        super().__init__(1.0)
        self.forced = forced

    def expected_ms(self, submitted: int) -> float:
        return 1e-6 if submitted == self.forced else 1e6

    def observe(self, submitted: int, wall_ms: float) -> None:
        pass


class TestAdaptiveCap:
    def test_choose_prefers_plain_on_hopeless_drafts(self):
        cost = RoundCostModel(50.0)
        assert choose_submit_length([0.01, 0.001, 0.0001], cost) == 0

    def test_choose_takes_full_block_on_confident_drafts(self):
        cost = RoundCostModel(50.0)
        assert choose_submit_length([0.98, 0.95, 0.9, 0.85, 0.8], cost) == 5

    def test_choose_truncates_decaying_tail(self):
        # Strong head, dead tail: the marginal cost of the tail tokens is
        # not covered by their survival probability.
        cost = RoundCostModel(50.0)
        cost._ema = {0: 65.0, 1: 130.0, 2: 137.0, 3: 145.0, 4: 152.0, 5: 160.0}
        j = choose_submit_length([0.9, 0.8, 0.02, 0.001, 0.0001], cost)
        assert j in (2, 3)

    def test_observe_updates_moving_average(self):
        cost = RoundCostModel(50.0)
        cost.observe(3, 100.0)
        assert cost.expected_ms(3) == 100.0
        cost.observe(3, 200.0)
        assert 100.0 < cost.expected_ms(3) < 200.0

    def test_calibrator_rescales_pessimistic_head(self):
        from moespresso.runtime.deepseek_v4.spec_decode import (
            OnlineCalibrator,
        )
        cal = OnlineCalibrator(block_size=3, warmup=2, probe_every=0)
        assert cal.force_full_block()
        # The head predicts 0.4 while position 0 is always accepted and
        # position 1 always rejected; calibration must lift position 0
        # toward 1 and keep position 1 low.
        for _ in range(20):
            cal.record([0.4, 0.4, 0.4], submitted=3, accepted=1)
        survival = cal.calibrated_survival([0.4, 0.4, 0.4])
        assert survival[0] > 0.8
        assert survival[1] < 0.2
        assert not cal.force_full_block()

    def test_probe_length_never_exceeds_block(self):
        from moespresso.runtime.deepseek_v4.spec_decode import (
            OnlineCalibrator,
        )
        for block in (1, 2, 3, 5):
            cal = OnlineCalibrator(block_size=block, warmup=0, probe_every=1)
            for _ in range(6):
                forced = cal.forced_length()
                assert forced is None or forced <= block
                cal.record([0.5] * block, submitted=block, accepted=0)

    def test_calibrator_probe_cadence(self):
        from moespresso.runtime.deepseek_v4.spec_decode import (
            OnlineCalibrator,
        )
        cal = OnlineCalibrator(block_size=2, warmup=1, probe_every=4)
        forced = []
        for _ in range(9):
            forced.append(cal.force_full_block())
            cal.record([0.5, 0.5], submitted=2, accepted=2)
        assert forced[0] is True
        assert any(forced[1:])
        assert not all(forced[1:])

    def test_adaptive_identity_with_plain_decode(self):
        target, draft, tap = build_tiny_pair()
        prompt = [2, 5, 8, 1]
        expected = plain_greedy(target, prompt, 30)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=30, temperature=0.0,
            adaptive_cap=True,
        )
        assert result.tokens == expected

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

    def test_forced_partial_block_identity(self):
        target, draft, tap = build_tiny_pair()
        prompt = [3, 4, 5]
        expected = plain_greedy(target, prompt, 24)
        result = spec_generate(
            target, draft, tap, prompt, max_new_tokens=24, temperature=0.0,
            adaptive_cap=True, cost_model=_ForcedCost(2),
            calibration_warmup=0, probe_every=0,
        )
        assert result.tokens == expected
        assert set(result.stats.submit_length_counts) == {2}


class TestDraftPrefetch:
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


class _FixedDrafter:
    """Minimal `Drafter` implementation proposing a fixed token block.

    Uses an identity tap transform (raw hyper-connection copies) and no
    confidence head, so the loop's neutral-confidence calibration path is
    exercised. Records every ingested position and token id so the
    contiguous committed-stream contract can be asserted.
    """

    def __init__(self, tokens, vocab):
        self._tokens = list(tokens)
        self._vocab = vocab
        self.positions = []
        self.token_stream = []

    @property
    def block_size(self):
        return len(self._tokens)

    @property
    def tap_layer_ids(self):
        return (0, 1)

    def tap_transform(self, layer_id, out):
        return out

    def make_state(self):
        return object()

    def ingest(self, state, rows, positions, token_ids):
        assert rows.shape[1] == len(positions)
        assert len(token_ids) == len(positions)
        self.positions.extend(positions)
        self.token_stream.extend(int(t) for t in token_ids)

    def draft(self, state, anchor_token, anchor_pos, temperature):
        k = len(self._tokens)
        tokens = mx.array([self._tokens], dtype=mx.int64)
        logits = mx.zeros((1, k, self._vocab))
        for i, t in enumerate(self._tokens):
            logits[0, i, t] = 8.0
        return DraftProposal(tokens=tokens, logits=logits, confidence=None)


class TestDrafterProtocol:
    def test_fake_drafter_identity_and_neutral_confidence(self):
        mx.random.seed(9)
        target = Model(tiny_model_args())
        randomize(target, seed=21)
        drafter = _FixedDrafter([5, 9], VOCAB)
        tap = install_hidden_tap(target, drafter.tap_layer_ids, drafter.tap_transform)

        prompt = [4, 8, 15, 16]
        max_new = 24
        expected = plain_greedy(target, prompt, max_new)
        result = spec_generate(
            target,
            drafter,
            tap,
            prompt,
            max_new_tokens=max_new,
            temperature=0.0,
            adaptive_cap=True,
        )
        assert result.tokens == expected
        assert result.stats.rounds > 0
        # The scheduler ran on neutral confidence: warmup rounds submitted
        # the full block, so speculative rounds were proposed.
        assert result.stats.proposed > 0
        # The loop fed the drafter a contiguous position stream carrying
        # the committed tokens: the prompt followed by the emitted prefix.
        assert drafter.positions == list(range(len(drafter.positions)))
        committed = prompt + result.tokens
        assert drafter.token_stream == committed[: len(drafter.token_stream)]
        assert len(drafter.token_stream) > len(prompt)

    @pytest.mark.parametrize("stop", ["length", "eos"])
    def test_terminal_accepted_prefix_sets_exact_public_frontier(self, stop):
        target, _draft, _tap = build_tiny_pair()
        prompt = [1, 2, 3]
        plain = plain_greedy(target, prompt, 5)
        drafter = _FixedDrafter(plain[1:4], VOCAB)
        tap = install_hidden_tap(
            target, drafter.tap_layer_ids, drafter.tap_transform
        )
        cache = target.make_cache()

        if stop == "length":
            max_new_tokens = 2
            eos_ids = None
            expected = plain[:2]
        else:
            # The stop token is the second accepted draft token, not the
            # anchor or the first accepted token.
            assert plain[2] not in plain[:2]
            max_new_tokens = 5
            eos_ids = [plain[2]]
            expected = plain[:3]

        result = spec_generate(
            target,
            drafter,
            tap,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            eos_ids=eos_ids,
            adaptive_cap=False,
            target_cache=cache,
        )

        # All three proposals matched. Termination cut through that accepted
        # prefix, so only the tokens visible to the caller may remain in the
        # target and drafter states.
        assert result.stats.accepted == 3
        assert result.tokens == expected
        expected_frontier = len(prompt) + len(expected)
        assert {int(layer.offset) for layer in cache} == {expected_frontier}
        assert drafter.positions == list(range(expected_frontier))
        assert drafter.token_stream == prompt + expected

    def test_nonzero_live_and_capsule_resumes_match(self):
        target, drafter, tap = build_tiny_pair()
        prompt = [4, 8, 15, 16]
        first = spec_generate(
            target,
            drafter,
            tap,
            prompt,
            max_new_tokens=7,
            temperature=0.0,
            adaptive_cap=False,
            draft_prefetch=False,
        )
        assert first.frontier > 0
        assert drafter.state_frontier(first.drafter_state) == first.frontier

        imported_cache = copy.deepcopy(first.target_cache)
        capsule = drafter.export_state(first.drafter_state)
        imported_state = drafter.import_state(capsule)
        timeline = prompt + first.tokens
        suffix = timeline[first.frontier :] + [23, 42]
        assert suffix

        live = spec_generate(
            target,
            drafter,
            tap,
            suffix,
            max_new_tokens=12,
            temperature=0.0,
            adaptive_cap=False,
            target_cache=first.target_cache,
            drafter_state=first.drafter_state,
            prompt_offset=first.frontier,
            state_owner=drafter,
            draft_prefetch=False,
        )
        restored = spec_generate(
            target,
            drafter,
            tap,
            suffix,
            max_new_tokens=12,
            temperature=0.0,
            adaptive_cap=False,
            target_cache=imported_cache,
            drafter_state=imported_state,
            prompt_offset=first.frontier,
            state_owner=drafter,
            draft_prefetch=False,
        )

        assert restored.tokens == live.tokens
        assert restored.frontier == live.frontier
        assert restored.stats.proposed == live.stats.proposed
        assert restored.stats.accepted == live.stats.accepted
        assert restored.stats.per_position_accept == live.stats.per_position_accept
        for live_window, restored_window in zip(live.drafter_state, restored.drafter_state):
            assert live_window.last_pos == restored_window.last_pos
            _assert_float32_bits_equal(live_window.kv, restored_window.kv)
        assert {int(layer.offset) for layer in live.target_cache} == {live.frontier}
        assert {int(layer.offset) for layer in restored.target_cache} == {restored.frontier}

    def test_nonzero_resume_rejects_target_or_drafter_frontier_mismatch(self):
        target, drafter, tap = build_tiny_pair()
        first = spec_generate(
            target,
            drafter,
            tap,
            [1, 2, 3],
            max_new_tokens=4,
            temperature=0.0,
            adaptive_cap=False,
            draft_prefetch=False,
        )

        with pytest.raises(ValueError, match="target cache positional offsets"):
            spec_generate(
                target,
                drafter,
                tap,
                [5],
                max_new_tokens=1,
                target_cache=first.target_cache,
                drafter_state=first.drafter_state,
                prompt_offset=first.frontier + 1,
                state_owner=drafter,
            )

        fresh_state = drafter.make_state()
        with pytest.raises(ValueError, match="drafter state frontier"):
            spec_generate(
                target,
                drafter,
                tap,
                [5],
                max_new_tokens=1,
                target_cache=first.target_cache,
                drafter_state=fresh_state,
                prompt_offset=first.frontier,
                state_owner=drafter,
            )


class TestReferenceGraphContract:
    """The two draft-graph points the reference states and the vendored
    decoder layer does not: the recombine orientation and the E4M3FN
    round trip on the non-RoPE prefix of every draft KV row."""

    def _served_geometry_args(self):
        return tiny_model_args(
            hidden_size=64,
            num_attention_heads=1,
            head_dim=512,
            qk_rope_head_dim=64,
            q_lora_rank=32,
            o_lora_rank=16,
            o_groups=1,
            sliding_window=4,
        )

    def test_recombine_contracts_comb_over_its_first_hc_axis(self):
        # `inference/model.py` Block.hc_post:
        #   y[b,s,j,d] = post[b,s,j] * x[b,s,d]
        #              + sum_i comb[b,s,i,j] * residual[b,s,i,d]
        x = mx.array([[[5.0, 7.0]]], dtype=mx.float32)
        post = mx.array([[[1.0, 0.0]]], dtype=mx.float32)
        residual = mx.array([[[[10.0, 100.0], [1.0, 2.0]]]], dtype=mx.float32)
        comb = mx.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=mx.float32)

        out = DSparkBlock._hc_post(None, x, residual, post, comb)
        mx.eval(out)

        # Stream 0: 1*[10,100] + 3*[1,2] + 1*[5,7]  = [18, 113]
        # Stream 1: 2*[10,100] + 4*[1,2] + 0*[5,7]  = [24, 208]
        expected = mx.array([[[[18.0, 113.0], [24.0, 208.0]]]], dtype=mx.float32)
        assert bool(mx.allclose(out, expected).item())

        # The two orientations are not the same arithmetic on this case.
        plain = post[..., None] * x[..., None, :] + mx.matmul(comb, residual)
        mx.eval(plain)
        assert not bool(mx.allclose(out, plain).item())

    def test_draft_stage_uses_the_reference_recombine(self):
        _, draft, _ = build_tiny_pair()
        hc = draft.args.model_args.hc_mult
        hidden = draft.args.model_args.hidden_size
        key = mx.random.key(11)
        x = mx.random.normal((1, 2, hidden), key=key)
        residual = mx.random.normal((1, 2, hc, hidden), key=mx.random.key(12))
        post = mx.random.normal((1, 2, hc), key=mx.random.key(13))
        comb = mx.random.normal((1, 2, hc, hc), key=mx.random.key(14))

        got = draft.blocks[0]._hc_post(x, residual, post, comb)
        want = post[..., None] * x[..., None, :].astype(mx.float32) + mx.matmul(
            mx.swapaxes(comb, -1, -2).astype(mx.float32),
            residual.astype(mx.float32),
        )
        mx.eval(got, want)
        assert bool(mx.allclose(got, want.astype(got.dtype)).item())

    def test_fp8_prefix_round_trip_is_the_reference_grid(self):
        from moespresso.runtime.deepseek_v4.dspark_model import _fp8_kv_roundtrip

        row = mx.concatenate(
            [
                mx.full((1, 1, 448), 257.0, dtype=mx.float32),
                mx.full((1, 1, 64), 257.0, dtype=mx.float32),
            ],
            axis=-1,
        )
        out = _fp8_kv_roundtrip(row)
        mx.eval(out)

        # Block amax 257 rounds the scale up to 2**0, and E4M3FN spaces the
        # [256, 512) binade by 32, so every non-RoPE entry lands on 256.
        assert float(mx.max(mx.abs(out[..., :448] - 256.0)).item()) == 0.0
        # The RoPE tail keeps its positional precision.
        assert float(mx.max(mx.abs(out[..., 448:] - 257.0)).item()) == 0.0

    def test_main_stream_rows_enter_the_window_rounded(self):
        from jang_tools.dsv4.mlx_model import _apply_partial_rope
        from moespresso.runtime.deepseek_v4.dspark_model import (
            DSparkDraftAttention,
            _fp8_kv_roundtrip,
        )

        args = self._served_geometry_args()
        attn = DSparkDraftAttention(args, layer_id=0)
        randomize(attn, scale=1.0, seed=21)
        window = DSparkWindowCache(window=args.sliding_window, head_dim=args.head_dim)

        main_x = mx.random.normal((1, 3, args.hidden_size), key=mx.random.key(22))
        attn.ingest_main_rows(main_x, [0, 1, 2], window)

        raw = _apply_partial_rope(
            attn.kv_norm(attn.wkv(main_x)),
            attn.rope,
            positions=mx.array([0, 1, 2], dtype=mx.int32),
        )
        rounded = _fp8_kv_roundtrip(raw)
        mx.eval(raw, rounded)
        # The two arms differ, so the assertion below is not vacuous.
        assert not bool(mx.allclose(raw, rounded).item())

        stored = window.kv[0, 0, :3]
        assert float(mx.max(mx.abs(stored - rounded[0])).item()) == 0.0

    def test_block_rows_are_rounded_before_attention(self, monkeypatch):
        from jang_tools.dsv4.mlx_model import _apply_partial_rope
        from moespresso.runtime.deepseek_v4 import dspark_model
        from moespresso.runtime.deepseek_v4.dspark_model import (
            DSparkDraftAttention,
            _fp8_kv_roundtrip,
        )

        args = self._served_geometry_args()
        attn = DSparkDraftAttention(args, layer_id=0)
        randomize(attn, scale=1.0, seed=23)
        window = DSparkWindowCache(window=args.sliding_window, head_dim=args.head_dim)
        window.write(
            mx.zeros((1, args.sliding_window, args.head_dim)),
            list(range(args.sliding_window)),
        )

        seen = {}

        def _capture(q, keys, values, **kwargs):
            seen["keys"] = keys
            return mx.zeros(
                (q.shape[0], q.shape[1], q.shape[2], args.head_dim), dtype=q.dtype
            )

        monkeypatch.setattr(
            dspark_model, "scaled_dot_product_attention", _capture)

        positions = [4, 5]
        x = mx.random.normal((1, 2, args.hidden_size), key=mx.random.key(24))
        attn(x, positions=positions, window=window)

        raw = _apply_partial_rope(
            attn.kv_norm(attn.wkv(x)).reshape(1, 2, 1, args.head_dim).transpose(
                0, 2, 1, 3),
            attn.rope,
            positions=mx.array(positions, dtype=mx.int32),
        )
        rounded = _fp8_kv_roundtrip(raw)
        mx.eval(raw, rounded)
        assert not bool(mx.allclose(raw, rounded).item())

        block_rows = seen["keys"][:, :, args.sliding_window:, :]
        assert float(mx.max(mx.abs(block_rows - rounded)).item()) == 0.0


class TestSanitize:
    def test_expert_stacking_and_renames(self):
        args = DSparkArgs(
            model_args=tiny_model_args(),
            n_mtp_layers=1,
            block_size=2,
            noise_token_id=NOISE,
            target_layer_ids=(0, 1),
            markov_rank=8,
        )
        n_exp = args.model_args.n_routed_experts
        weights = {"mtp.0.attn.wkv.weight": mx.zeros((32, 64))}
        weights["mtp.0.attn_norm.weight"] = mx.ones((64,))
        weights["mtp.0.embed.weight"] = mx.zeros((VOCAB, 64))
        for e in range(n_exp):
            for w in ("w1", "w2", "w3"):
                weights[f"mtp.0.ffn.experts.{e}.{w}.weight"] = mx.zeros((2, 2))
        out = sanitize_dspark_weights(weights, args)
        assert "blocks.0.self_attn.wkv.weight" in out
        assert "blocks.0.input_layernorm.weight" in out
        assert "blocks.0.mlp.switch_mlp.gate_proj.weight" in out
        assert out["blocks.0.mlp.switch_mlp.gate_proj.weight"].shape == (n_exp, 2, 2)
        assert not any(k.startswith("blocks.0.embed") for k in out)

    def test_incomplete_experts_fail_closed(self):
        args = DSparkArgs(
            model_args=tiny_model_args(),
            n_mtp_layers=1,
            block_size=2,
            noise_token_id=NOISE,
            target_layer_ids=(0,),
            markov_rank=8,
        )
        weights = {"mtp.0.ffn.experts.0.w1.weight": mx.zeros((2, 2))}
        with pytest.raises((ValueError, KeyError)):
            sanitize_dspark_weights(weights, args)


class TestCommitStream:
    def test_on_commit_streams_every_token_in_order(self):
        target, draft, tap = build_tiny_pair()
        prompt = [1, 2, 3, 4]
        commits = []
        result = spec_generate(
            target,
            draft,
            tap,
            prompt,
            max_new_tokens=20,
            temperature=0.0,
            on_commit=lambda tokens: commits.append(list(tokens)),
        )
        assert [t for run in commits for t in run] == result.tokens
        assert all(commits)  # every commit carries at least one token


class TestServeSeamIntegration:
    """The serve seam end to end on the tiny pair: an installed drafter takes
    a greedy request through the real spec loop and returns the plain path's
    result surface plus the speculative stats block."""

    def _served_target(self):
        from pathlib import Path

        from moespresso.runtime.deepseek_v4.spec_serve import ServedDrafter

        target, draft, tap = build_tiny_pair()
        target._moespresso_ds4_drafter = ServedDrafter(
            family="dspark", sidecar_dir=Path("unused"), drafter=draft, tap=tap)
        return target

    class _Tokenizer:
        bos_token = None
        eos_token_id = None

        def decode(self, token_ids):
            return " ".join(str(int(t)) for t in token_ids)

    def test_greedy_request_serves_via_spec_with_token_identity(self):
        from moespresso.runtime.serve import generate_with_metadata

        target = self._served_target()
        prompt = [1, 2, 3, 4]
        max_new = 24
        expected = plain_greedy(target, prompt, max_new)

        streamed = []
        result = generate_with_metadata(
            target,
            self._Tokenizer(),
            prompt,
            max_tokens=max_new,
            temperature=0.0,
            response_callback=lambda step, resp: streamed.append(
                (step, int(resp.token))),
        )

        assert list(result.generated_token_ids) == expected
        assert result.text == " ".join(str(t) for t in expected)
        assert result.finish_reason == "length"
        assert result.prompt_tokens == len(prompt)
        assert result.completion_tokens == max_new
        assert result.prompt_cache is None
        assert result.first_token_seconds is not None
        assert result.generation_seconds is not None
        stats = result.speculative
        assert stats["drafter"] == "dspark"
        assert stats["rounds"] > 0
        assert stats["mean_accepted_length"] > 0
        assert sum(stats["submit_length_counts"].values()) == stats["rounds"]
        # Streaming callbacks covered every generated token once, in order.
        assert [token for _, token in streamed] == expected

    def test_eos_request_finishes_with_stop(self):
        from moespresso.runtime.serve import generate_with_metadata

        target = self._served_target()
        prompt = [1, 2, 3]
        full = plain_greedy(target, prompt, 30)
        eos = full[10]
        first = full.index(eos)

        tokenizer = self._Tokenizer()
        tokenizer.eos_token_id = eos
        result = generate_with_metadata(
            target,
            tokenizer,
            prompt,
            max_tokens=30,
            temperature=0.0,
        )

        assert list(result.generated_token_ids) == full[: first + 1]
        assert result.finish_reason == "stop"
        # The stop token never surfaces in the text.
        assert result.text == " ".join(str(t) for t in full[:first])
