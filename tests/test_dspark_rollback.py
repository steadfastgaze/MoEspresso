"""Synthetic tests for bit-exact DSpark verify rollback.

Covers the compressed-layer cache stack the runtime serves with: a tiny
DeepSeek-V4 graph with one sliding-window layer and one ratio-4 compressed
layer, caches built by the runtime cache factory (plain KVCache plus
DeepseekV4Cache wrapped with the fp8 KV round trip, the aux trim clear,
and the fixed decode state), and the attention-side compressor wrapper
that maintains the derived pool caches and the indexer advance.

Block forwards and single-token forwards are different numeric lattices
even at these sizes (matmul and attention kernels differ per shape), so a
bitwise comparison across forward shapes is not meaningful. The tested
bitwise properties are the ones the rollback contract states directly:

- restoring zero kept tokens returns every mutated location to its
  pre-verify content;
- the restored state and all subsequent compute are independent of the
  rejected suffix tokens;
- at the raw cache level, where identical row values can be fed to both
  arms, restore reproduces the stepwise-fed reference state exactly,
  including full-block keeps.

Token-level losslessness of the whole loop is covered end to end against
plain greedy decoding.
"""

import copy

import mlx.core as mx
import mlx.nn as nn
import pytest

from jang_tools.dsv4.mlx_model import DeepseekV4Cache, Model, ModelArgs
from mlx_lm.models.cache import KVCache, RotatingKVCache

from moespresso.runtime.deepseek_v4 import fixed_decode_state
from moespresso.runtime.deepseek_v4 import spec_decode as spec_decode_module
from moespresso.runtime.deepseek_v4.spec_decode import (
    install_hidden_tap,
    spec_generate,
)
from moespresso.runtime.deepseek_v4.dspark_model import DSparkArgs, DSparkDraftModel
from moespresso.runtime.deepseek_v4.dspark_rollback import (
    capture_verify_state,
    restore_verify_state,
)
from moespresso.runtime.deepseek_v4.model import (
    _cache_with_compressed_pool_aux_trim_clear,
    _cache_with_fp8_kv_roundtrip,
    _patch_deepseek_v4_attention_compressor_fp8_kv,
    _patch_deepseek_v4_required_attention_cache,
)

VOCAB = 97
NOISE = VOCAB - 1
WINDOW = 8


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
        sliding_window=WINDOW,
        rms_norm_eps=1e-6,
        compress_ratios=[0, 4],
        index_n_heads=2,
        index_head_dim=16,
        index_topk=1,
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


def build_tiny_model(seed: int = 3) -> Model:
    args = tiny_model_args()
    model = Model(args)
    randomize(model, seed=seed)
    _patch_deepseek_v4_required_attention_cache(model)
    _patch_deepseek_v4_attention_compressor_fp8_kv(model)
    return model


@pytest.fixture(scope="module")
def tiny_model() -> Model:
    return build_tiny_model()


def tokens(seed: int, n: int):
    return [
        int(t) for t in mx.random.randint(0, VOCAB - 1, (n,), key=mx.random.key(seed))
    ]


def _copy(x):
    return None if x is None else x[...]


def full_state_copy(cache):
    layers = []
    arrays = []
    for c in cache:
        if isinstance(c, DeepseekV4Cache):
            entry = {
                "kind": "composite",
                "offset": int(c.local.offset),
                "idx": int(c.local._idx),
                "keys": _copy(c.local.keys),
                "values": _copy(c.local.values),
            }
            for state_key in ("compressor_state", "indexer_state"):
                state = getattr(c, state_key)
                entry[state_key] = {
                    key: (_copy(value) if isinstance(value, mx.array) else value)
                    for key, value in state.items()
                }
            layers.append(entry)
        else:
            offset = int(c.offset)
            layers.append(
                {
                    "kind": "plain",
                    "offset": offset,
                    "keys": None if c.keys is None else _copy(c.keys[..., :offset, :]),
                    "values": None
                    if c.values is None
                    else _copy(c.values[..., :offset, :]),
                }
            )
    for entry in layers:
        arrays += [v for v in entry.values() if isinstance(v, mx.array)]
        for state_key in ("compressor_state", "indexer_state"):
            if state_key in entry:
                arrays += [
                    v for v in entry[state_key].values() if isinstance(v, mx.array)
                ]
    mx.eval(*arrays)
    return layers


def assert_arrays_equal(a, b, label):
    a_empty = a is None or 0 in a.shape
    b_empty = b is None or 0 in b.shape
    if a_empty or b_empty:
        assert a_empty and b_empty, (label, a, b)
        return
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert bool(mx.array_equal(a, b)), (
        label,
        float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max()),
    )


def assert_states_equal(lhs, rhs):
    assert len(lhs) == len(rhs)
    for i, (a, b) in enumerate(zip(lhs, rhs)):
        assert a["kind"] == b["kind"], i
        assert a["offset"] == b["offset"], (i, a["offset"], b["offset"])
        if a["kind"] == "composite":
            assert a["idx"] == b["idx"], (i, a["idx"], b["idx"])
        assert_arrays_equal(a["keys"], b["keys"], (i, "keys"))
        assert_arrays_equal(a["values"], b["values"], (i, "values"))
        if a["kind"] != "composite":
            continue
        for state_key in ("compressor_state", "indexer_state"):
            sa, sb = a[state_key], b[state_key]
            for key in sorted(set(sa) | set(sb)):
                va, vb = sa.get(key), sb.get(key)
                if isinstance(va, mx.array) or isinstance(vb, mx.array):
                    assert_arrays_equal(va, vb, (i, state_key, key))
                else:
                    assert va == vb, (i, state_key, key, va, vb)


def assert_float32_bits_equal(lhs, rhs, label):
    assert lhs.dtype == rhs.dtype == mx.float32, (label, lhs.dtype, rhs.dtype)
    assert lhs.shape == rhs.shape, (label, lhs.shape, rhs.shape)
    assert bool(mx.array_equal(lhs.view(mx.uint32), rhs.view(mx.uint32))), label


def assert_dspark_capsules_equal(lhs, rhs):
    assert lhs.kind == rhs.kind
    assert lhs.schema_major == rhs.schema_major
    assert lhs.schema_minor == rhs.schema_minor
    assert lhs.frontier == rhs.frontier
    assert lhs.metadata == rhs.metadata
    assert lhs.nbytes == rhs.nbytes
    assert len(lhs.tensors) == len(rhs.tensors)
    for stage, (left, right) in enumerate(zip(lhs.tensors, rhs.tensors)):
        assert_float32_bits_equal(left, right, ("drafter", stage))


class _TracingDrafter:
    """Record full DSpark proposals while delegating the draft protocol."""

    def __init__(self, inner):
        self.inner = inner
        self.greedy_only = getattr(inner, "greedy_only", False)
        self.proposals = []

    @property
    def block_size(self):
        return self.inner.block_size

    @property
    def tap_layer_ids(self):
        return self.inner.tap_layer_ids

    def tap_transform(self, layer_id, out):
        return self.inner.tap_transform(layer_id, out)

    def make_state(self):
        return self.inner.make_state()

    def ingest(self, state, rows, positions, token_ids):
        return self.inner.ingest(state, rows, positions, token_ids)

    def draft(self, state, anchor_token, anchor_pos, temperature):
        proposal = self.inner.draft(state, anchor_token, anchor_pos, temperature)
        arrays = [proposal.tokens, proposal.logits]
        if proposal.confidence is not None:
            arrays.append(proposal.confidence)
        mx.eval(*arrays)
        self.proposals.append(
            {
                "anchor_token": int(anchor_token),
                "anchor_pos": int(anchor_pos),
                "tokens": tuple(int(token) for token in proposal.tokens[0]),
                "logits": _copy(proposal.logits),
                "confidence": _copy(proposal.confidence),
            }
        )
        mx.eval(*(value for value in self.proposals[-1].values() if isinstance(value, mx.array)))
        return proposal


def assert_proposal_traces_equal(lhs, rhs):
    assert len(lhs) == len(rhs)
    for round_id, (left, right) in enumerate(zip(lhs, rhs)):
        for key in ("anchor_token", "anchor_pos", "tokens"):
            assert left[key] == right[key], (round_id, key)
        assert_float32_bits_equal(left["logits"], right["logits"], (round_id, "draft_logits"))
        assert_float32_bits_equal(
            left["confidence"],
            right["confidence"],
            (round_id, "draft_confidence"),
        )


def prefill(model, cache, prompt):
    logits = model(mx.array(prompt, dtype=mx.int64)[None], cache=cache)
    mx.eval(logits, *(c.state for c in cache))
    return logits


def step(model, cache, token):
    logits = model(mx.array([[token]], dtype=mx.int64), cache=cache)
    mx.eval(logits, *(c.state for c in cache))
    return logits


def block(model, cache, ids):
    logits = model(mx.array(ids, dtype=mx.int64)[None], cache=cache)
    mx.eval(logits, *(c.state for c in cache))
    return logits


# Prefill lengths cover the pre-wrap regime, the post-wrap regime
# (offset past the sliding window of 8), and verify blocks that straddle
# a compress-ratio-4 window boundary at both offsets.
REGIMES = [5, 11, 22]


class TestRestoreExactness:
    @pytest.mark.parametrize("prefill_len", REGIMES)
    def test_restore_zero_returns_pre_verify_state(self, tiny_model, prefill_len):
        cache = tiny_model.make_cache()
        prefill(tiny_model, cache, tokens(31, prefill_len))
        reference = full_state_copy(cache)

        snapshot = capture_verify_state(cache, 5)
        block(tiny_model, cache, tokens(32, 5))
        restore_verify_state(cache, snapshot, 0)

        assert_states_equal(reference, full_state_copy(cache))

    @pytest.mark.parametrize("prefill_len", REGIMES)
    @pytest.mark.parametrize("keep", [1, 3, 4])
    def test_restore_is_independent_of_rejected_suffix(
        self, tiny_model, prefill_len, keep
    ):
        n = 5
        prompt = tokens(41, prefill_len)
        kept = tokens(42, keep)
        suffix_a = tokens(43, n - keep)
        suffix_b = [(t + 1) % (VOCAB - 1) for t in suffix_a]
        assert suffix_a != suffix_b
        continuation = tokens(45, 4)

        states = []
        logit_runs = []
        for suffix in (suffix_a, suffix_b):
            cache = tiny_model.make_cache()
            prefill(tiny_model, cache, prompt)
            snapshot = capture_verify_state(cache, n)
            block(tiny_model, cache, kept + suffix)
            restore_verify_state(cache, snapshot, keep)
            states.append(full_state_copy(cache))
            rows = [step(tiny_model, cache, t)[0, -1] for t in continuation]
            logit_runs.append(rows)
            states.append(full_state_copy(cache))

        assert_states_equal(states[0], states[2])
        assert_states_equal(states[1], states[3])
        for row_a, row_b in zip(*logit_runs):
            diff = mx.abs(row_a - row_b).max()
            assert float(diff) == 0.0


def make_runtime_composite(ratio: int) -> DeepseekV4Cache:
    """Build one compressed-layer cache the way the runtime factory does."""
    cache = _cache_with_compressed_pool_aux_trim_clear(
        _cache_with_fp8_kv_roundtrip(DeepseekV4Cache(WINDOW, compress_ratio=ratio))
    )
    return fixed_decode_state.install_fixed_decode_state(cache)


class _CacheDriver:
    """Feed a composite cache the way attention does, with chosen values.

    Every row is a deterministic function of its position, so a stepwise
    reference arm and a block-fed rollback arm receive bitwise-identical
    inputs and every comparison below is exact.
    """

    def __init__(self, ratio: int, overlap: bool, head_dim: int = 4, local_dim: int = 4):
        self.ratio = ratio
        self.overlap = overlap
        self.head_dim = head_dim
        self.local_dim = local_dim
        self.out_dim = 2 * head_dim if overlap else head_dim

    def _local_rows(self, start, length):
        rows = mx.arange(self.local_dim, dtype=mx.float32)[None, None, None]
        pos = mx.arange(start, start + length, dtype=mx.float32)[None, None, :, None]
        return rows + 10.0 * pos + 1.0

    def _branch_rows(self, start, length, salt):
        rows = mx.arange(self.out_dim, dtype=mx.float32)[None, None]
        pos = mx.arange(start, start + length, dtype=mx.float32)[None, :, None]
        return rows + 100.0 * pos + salt

    def feed(self, cache, start, length):
        k = self._local_rows(start, length)
        cache.update_and_fetch(k, k)
        for salt, state_key in ((3.0, "compressor_state"), (7.0, "indexer_state")):
            kv = self._branch_rows(start, length, salt)
            gate = self._branch_rows(start, length, salt + 0.5)
            if self.overlap:
                rows_kv, _, _ = cache.accumulate_overlap_windows(
                    kv, gate, state_key, self.ratio, start, self.head_dim
                )
                pooled = rows_kv[:, :, 0, :]
            else:
                flat_kv, _, _ = cache.accumulate_windows(
                    kv, gate, state_key, self.ratio, start
                )
                windows = flat_kv.shape[1] // self.ratio
                pooled = flat_kv.reshape(
                    1, windows, self.ratio, self.out_dim
                )[:, :, 0, : self.head_dim]
            cache.update_pool(pooled, state_key)
        mx.eval(*(v for v in cache.state if v is not None))


def composite_state_copy(cache):
    return full_state_copy([cache])


class TestCacheLevelStepwiseEquivalence:
    @pytest.mark.parametrize("ratio,overlap", [(4, True), (3, False)])
    @pytest.mark.parametrize("prefed", [5, 21])
    @pytest.mark.parametrize("keep", [0, 2, 5])
    def test_restore_matches_stepwise_reference(self, ratio, overlap, prefed, keep):
        n = 5
        driver = _CacheDriver(ratio, overlap)

        reference = make_runtime_composite(ratio)
        for pos in range(prefed + keep):
            driver.feed(reference, pos, 1)

        cache = make_runtime_composite(ratio)
        for pos in range(prefed):
            driver.feed(cache, pos, 1)
        # Simulate the derived caches the attention wrappers maintain, so
        # the restore path that truncates them is exercised.
        for state_key, (aux_key, rows_key) in (
            ("compressor_state", ("pooled_fp8", "pooled_fp8_rows")),
            ("indexer_state", ("pooled_qat", "pooled_qat_rows")),
        ):
            state = getattr(cache, state_key)
            pooled = state.get("pooled")
            if pooled is not None:
                state[aux_key] = pooled * 2.0
                state[rows_key] = int(pooled.shape[1])
                mx.eval(state[aux_key])

        pre = composite_state_copy(cache)
        snapshot = capture_verify_state([cache], n)
        driver.feed(cache, prefed, n)
        restore_verify_state([cache], snapshot, keep)

        post = composite_state_copy(cache)
        expected = composite_state_copy(reference)

        # The reference arm carries no derived caches; check those against
        # the pre-verify entries truncated to the surviving pool rows, then
        # compare everything else.
        for entry, state_key, aux_key, rows_key in (
            (post[0], "compressor_state", "pooled_fp8", "pooled_fp8_rows"),
            (post[0], "indexer_state", "pooled_qat", "pooled_qat_rows"),
        ):
            state = entry[state_key]
            pre_state = pre[0][state_key]
            pre_rows = int(pre_state.get(rows_key, 0) or 0)
            pooled = state.get("pooled")
            target_rows = 0 if pooled is None else int(pooled.shape[1])
            survive = min(pre_rows, target_rows)
            if survive > 0:
                assert state.get(rows_key) == survive
                assert_arrays_equal(
                    state.get(aux_key),
                    pre_state[aux_key][:, :survive],
                    (state_key, aux_key),
                )
            else:
                assert aux_key not in state and rows_key not in state
            state.pop(aux_key, None)
            state.pop(rows_key, None)
        assert_states_equal(post, expected)

    def test_recorders_are_removed_and_snapshot_is_single_use(self):
        driver = _CacheDriver(4, True)
        cache = make_runtime_composite(4)
        for pos in range(6):
            driver.feed(cache, pos, 1)
        fixed_accumulate = cache.__dict__.get("accumulate_windows")

        snapshot = capture_verify_state([cache], 2)
        with pytest.raises(RuntimeError):
            capture_verify_state([cache], 2)
        driver.feed(cache, 6, 2)
        restore_verify_state([cache], snapshot, 1)
        assert cache.__dict__.get("accumulate_windows") is fixed_accumulate

        with pytest.raises(RuntimeError):
            restore_verify_state([cache], snapshot, 1)

    def test_fail_closed_on_bounds_and_foreign_caches(self):
        driver = _CacheDriver(4, True)
        cache = make_runtime_composite(4)
        for pos in range(6):
            driver.feed(cache, pos, 1)

        with pytest.raises(ValueError):
            capture_verify_state([cache], 0)
        with pytest.raises(TypeError):
            capture_verify_state([RotatingKVCache(max_size=4)], 2)

        snapshot = capture_verify_state([cache], 2)
        driver.feed(cache, 6, 2)
        with pytest.raises(ValueError):
            restore_verify_state([cache], snapshot, 3)
        other = make_runtime_composite(4)
        with pytest.raises(ValueError):
            restore_verify_state([other], snapshot, 1)
        restore_verify_state([cache], snapshot, 2)

    def test_plain_kvcache_rows_and_offset_restored(self):
        cache = _cache_with_fp8_kv_roundtrip(KVCache())
        rows = mx.arange(24, dtype=mx.float32).reshape(1, 1, 6, 4)
        cache.update_and_fetch(rows, rows)
        stale = mx.full((1, 1, 3, 4), -1.0)
        cache.keys[..., 6:9, :] = stale
        cache.values[..., 6:9, :] = stale
        mx.eval(cache.keys, cache.values)

        snapshot = capture_verify_state([cache], 3)
        fresh = mx.full((1, 1, 3, 4), 9.0)
        cache.update_and_fetch(fresh, fresh)
        restore_verify_state([cache], snapshot, 1)

        assert int(cache.offset) == 7
        assert_arrays_equal(cache.keys[..., :6, :], rows, "prefix")
        assert_arrays_equal(cache.keys[..., 6:7, :], fresh[..., :1, :], "kept")
        assert_arrays_equal(cache.keys[..., 7:9, :], stale[..., 1:, :], "restored")


def plain_greedy(target, prompt_ids, max_new):
    cache = target.make_cache()
    logits = prefill(target, cache, list(prompt_ids))
    out = [int(mx.argmax(logits[0, -1].astype(mx.float32)))]
    while len(out) < max_new:
        logits = step(target, cache, out[-1])
        out.append(int(mx.argmax(logits[0, -1].astype(mx.float32))))
    return out


class TestSpecDecodeIdentityRatio4:
    def test_spec_matches_plain_greedy_over_compressed_layers(self):
        target = build_tiny_model(seed=3)
        args = target.args
        dargs = DSparkArgs(
            model_args=args,
            n_mtp_layers=2,
            block_size=3,
            noise_token_id=NOISE,
            target_layer_ids=(0, 1),
            markov_rank=8,
        )
        draft = DSparkDraftModel(dargs, embed=target.model.embed, lm_head=target.lm_head)
        randomize(draft, seed=1003)
        tap = install_hidden_tap(target, draft.tap_layer_ids, draft.tap_transform)

        prompt = tokens(51, 12)
        max_new = 64
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
        # A random draft against a random target must reject sometimes; a
        # zero-rejection run would mean the two arms share a code path.
        assert result.stats.accepted < result.stats.proposed

    def test_live_and_exported_resume_match_complete_state_and_trace(self, monkeypatch):
        target = build_tiny_model(seed=3)
        args = target.args
        dargs = DSparkArgs(
            model_args=args,
            n_mtp_layers=2,
            block_size=3,
            noise_token_id=NOISE,
            target_layer_ids=(0, 1),
            markov_rank=8,
        )
        drafter = DSparkDraftModel(dargs, embed=target.model.embed, lm_head=target.lm_head)
        randomize(drafter, seed=1003)
        tap = install_hidden_tap(target, drafter.tap_layer_ids, drafter.tap_transform)

        # Stop one token before the rotating-cache wrap. The pending target
        # token plus two new suffix tokens then complete a ratio-4 group and
        # cross the local window boundary before the first resumed proposal.
        prompt = tokens(61, WINDOW - 1)
        first = spec_generate(
            target,
            drafter,
            tap,
            prompt,
            max_new_tokens=1,
            temperature=0.0,
            adaptive_cap=False,
            draft_prefetch=False,
        )
        assert first.frontier == WINDOW - 1
        assert drafter.state_frontier(first.drafter_state) == first.frontier
        composite = [cache for cache in first.target_cache if isinstance(cache, DeepseekV4Cache)]
        assert len(composite) == 1
        assert composite[0].compress_ratio == 4

        timeline = prompt + first.tokens
        pending = timeline[first.frontier :]
        assert pending == first.tokens
        suffix = pending + tokens(62, 2)
        assert suffix
        assert first.frontier % 4 == 3
        assert first.frontier + len(suffix) > WINDOW

        boundary_target = full_state_copy(first.target_cache)
        imported_cache = copy.deepcopy(first.target_cache)
        imported_target = full_state_copy(imported_cache)
        assert imported_cache is not first.target_cache
        assert all(
            live_layer is not imported_layer
            for live_layer, imported_layer in zip(first.target_cache, imported_cache)
        )
        assert_states_equal(boundary_target, imported_target)

        boundary_capsule = drafter.export_state(first.drafter_state)
        imported_state = drafter.import_state(boundary_capsule)
        imported_capsule = drafter.export_state(imported_state)
        assert imported_state is not first.drafter_state
        assert all(
            live_window is not imported_window
            for live_window, imported_window in zip(first.drafter_state, imported_state)
        )
        assert_dspark_capsules_equal(boundary_capsule, imported_capsule)
        assert boundary_capsule.frontier == first.frontier
        assert boundary_capsule.metadata_dict() == {
            "batch": 1,
            "dtype": str(mx.float32),
            "head_dim": args.head_dim,
            "stages": dargs.n_mtp_layers,
            "window": WINDOW,
        }

        real_greedy_accept = spec_decode_module.greedy_accept
        real_sample_from_logits = spec_decode_module.sample_from_logits
        active_accept_trace = None
        active_seed_trace = None

        def traced_greedy_accept(draft_tokens, target_logits):
            result = real_greedy_accept(draft_tokens, target_logits)
            logits_copy = _copy(target_logits)
            top_tokens = mx.argmax(logits_copy[0], axis=-1)
            mx.eval(logits_copy, top_tokens)
            assert active_accept_trace is not None
            active_accept_trace.append(
                {
                    "submitted": tuple(int(token) for token in draft_tokens[0]),
                    "accepted": result.accepted,
                    "emitted": tuple(result.emitted),
                    "matches": tuple(result.position_matches or ()),
                    "target_top_tokens": tuple(int(token) for token in top_tokens),
                    "target_logits": logits_copy,
                }
            )
            return result

        def traced_sample_from_logits(logits, temperature):
            result = real_sample_from_logits(logits, temperature)
            logits_copy = _copy(logits)
            mx.eval(logits_copy, result)
            assert active_seed_trace is not None
            active_seed_trace.append(
                {
                    "top_tokens": tuple(int(token) for token in result),
                    "target_logits": logits_copy,
                }
            )
            return result

        monkeypatch.setattr(spec_decode_module, "greedy_accept", traced_greedy_accept)
        monkeypatch.setattr(spec_decode_module, "sample_from_logits", traced_sample_from_logits)

        live_drafter = _TracingDrafter(drafter)
        live_accept_trace = []
        live_seed_trace = []
        active_accept_trace = live_accept_trace
        active_seed_trace = live_seed_trace
        live = spec_generate(
            target,
            live_drafter,
            tap,
            suffix,
            max_new_tokens=24,
            temperature=0.0,
            adaptive_cap=False,
            target_cache=first.target_cache,
            drafter_state=first.drafter_state,
            prompt_offset=first.frontier,
            state_owner=drafter,
            draft_prefetch=False,
        )

        # Running the live arm must not mutate either restored arm object.
        assert_states_equal(imported_target, full_state_copy(imported_cache))
        assert_dspark_capsules_equal(imported_capsule, drafter.export_state(imported_state))

        restored_drafter = _TracingDrafter(drafter)
        restored_accept_trace = []
        restored_seed_trace = []
        active_accept_trace = restored_accept_trace
        active_seed_trace = restored_seed_trace
        restored = spec_generate(
            target,
            restored_drafter,
            tap,
            suffix,
            max_new_tokens=24,
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
        assert restored.stats == live.stats
        assert_states_equal(
            full_state_copy(live.target_cache),
            full_state_copy(restored.target_cache),
        )
        assert_dspark_capsules_equal(
            drafter.export_state(live.drafter_state),
            drafter.export_state(restored.drafter_state),
        )
        assert [window.last_pos for window in live.drafter_state] == [
            window.last_pos for window in restored.drafter_state
        ]

        assert_proposal_traces_equal(live_drafter.proposals, restored_drafter.proposals)
        assert len(live_seed_trace) == len(restored_seed_trace) == 1
        assert live_seed_trace[0]["top_tokens"] == restored_seed_trace[0]["top_tokens"]
        assert_float32_bits_equal(
            live_seed_trace[0]["target_logits"],
            restored_seed_trace[0]["target_logits"],
            "suffix_seed_logits",
        )
        assert len(live_accept_trace) == len(restored_accept_trace)
        for round_id, (left, right) in enumerate(zip(live_accept_trace, restored_accept_trace)):
            for key in (
                "submitted",
                "accepted",
                "emitted",
                "matches",
                "target_top_tokens",
            ):
                assert left[key] == right[key], (round_id, key)
            assert_float32_bits_equal(
                left["target_logits"],
                right["target_logits"],
                (round_id, "target_logits"),
            )

        # Fixed scheduling makes every full proposal the submitted block,
        # and these independent counters prove that verification ran.
        assert live.stats.rounds == len(live_accept_trace)
        assert live.stats.rounds == len(live_drafter.proposals)
        assert live.stats.rounds > 0
        for proposal, acceptance in zip(live_drafter.proposals, live_accept_trace):
            assert acceptance["submitted"] == proposal["tokens"]
            assert len(acceptance["target_top_tokens"]) == (len(acceptance["submitted"]) + 1)
        assert live.stats.proposed == sum(
            len(round_trace["submitted"]) for round_trace in live_accept_trace
        )
        assert live.stats.accepted == sum(
            round_trace["accepted"] for round_trace in live_accept_trace
        )
        assert live.stats.submit_length_counts == {drafter.block_size: live.stats.rounds}
        assert live.stats.proposed > 0
        assert live.stats.accepted < live.stats.proposed
        assert live_drafter.proposals[0]["anchor_pos"] == (first.frontier + len(suffix))
        assert live.frontier > WINDOW

        live_composite = next(
            entry for entry in full_state_copy(live.target_cache) if entry["kind"] == "composite"
        )
        for state_key in ("compressor_state", "indexer_state"):
            pooled = live_composite[state_key]["pooled"]
            assert pooled is not None
            assert pooled.shape[1] > 0
