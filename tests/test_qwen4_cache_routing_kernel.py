"""Independent CPU selection and MLX scoring references for the fused router."""

import math

import mlx.core as mx
import numpy as np
import pytest

from moespresso.runtime.qwen4.cache_routing_kernel import (
    cache_prior_route as route,
)


DEFAULT_BONUS = math.log(2)
DEFAULT_PROTECTED_ROUTES = 2
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def original(logits):
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    candidates = mx.argpartition(probabilities, kth=-10, axis=-1)[..., -10:]
    candidate_scores = mx.take_along_axis(probabilities, candidates, axis=-1)
    rank = mx.argsort(-candidate_scores, axis=-1)
    ids = mx.take_along_axis(candidates, rank, axis=-1).astype(mx.uint32)
    scores = mx.take_along_axis(candidate_scores, rank, axis=-1)
    return ids, (scores / mx.sum(scores, axis=-1, keepdims=True)).astype(mx.bfloat16)


def reference(logits, maps, bonus, protected_routes=DEFAULT_PROTECTED_ROUTES):
    base, _scores = original(logits)
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    raw = np.asarray(logits.astype(mx.float32)).reshape(-1, 512)
    probs = np.asarray(probabilities).reshape(-1, 512)
    old = np.asarray(base).reshape(-1, 10)
    resident = np.logical_and.reduce([np.asarray(m) < 512 for m in maps])
    output, changes = [], []
    for values, probability, ids in zip(raw, probs, old, strict=True):
        if bonus == 0 or np.all(resident[ids[protected_routes:]]) or not np.isfinite(values).all():
            selected = ids.tolist()
        else:
            biased = probability.copy()
            biased[resident] *= np.float32(math.exp(bonus))
            biased[ids[:protected_routes]] = np.inf
            chosen = sorted(range(512), key=lambda i: (float(biased[i]), i), reverse=True)[:10]
            selected = sorted(chosen, key=lambda i: (-float(probability[i]), i))
        output.append(selected)
        changes.append(len(set(selected) - set(ids.tolist())))
    ids = mx.array(output, dtype=mx.uint32).reshape(base.shape)
    scores = mx.take_along_axis(probabilities, ids, axis=-1)
    scores = (scores / mx.sum(scores, axis=-1, keepdims=True)).astype(mx.bfloat16)
    return ids, scores, mx.array(changes, dtype=mx.uint32)


def assert_exact(actual, expected):
    mx.eval(*actual, *expected)
    for got, wanted in zip(actual, expected, strict=True):
        assert got.shape == wanted.shape and got.dtype == wanted.dtype
        if got.dtype == mx.bfloat16:
            got, wanted = got.view(mx.uint16), wanted.view(mx.uint16)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(wanted))


def sample(seed=71):
    rng = np.random.default_rng(seed)
    logits = mx.array(rng.standard_normal((23, 512)).astype(np.float32)).astype(mx.bfloat16)
    maps = []
    for _ in range(3):
        ids = rng.choice(512, 223, replace=False)
        slots = np.full(512, 512, np.uint32)
        slots[ids] = np.arange(223, dtype=np.uint32)
        maps.append(mx.array(slots))
    return logits, tuple(maps)


@pytest.mark.parametrize("seed", [71, 72, 73, 74, 75])
@pytest.mark.parametrize("protected_routes", range(4))
def test_zero_is_exact_original_routing(seed, protected_routes):
    logits, maps = sample(seed)
    got = route(logits, maps, bonus=0, protected_routes=protected_routes)
    assert_exact(got[:2], original(logits))
    assert not np.asarray(got[2]).any()


@pytest.mark.parametrize("seed", [71, 72, 73, 74, 75])
def test_default_bias_matches_independent_oracle(seed):
    logits, maps = sample(seed)
    got = route(logits, maps)
    assert_exact(got, reference(logits, maps, DEFAULT_BONUS))
    for selected in np.asarray(got[0]):
        assert len(set(selected)) == 10


@pytest.mark.parametrize("factor", [1, 1.5, 2, 3, 4, 4.01, 5, 8])
@pytest.mark.parametrize("protected_routes", range(4))
def test_configured_bias_matches_independent_oracle(factor, protected_routes):
    logits, maps = sample()
    bonus = math.log(factor)
    got = route(logits, maps, bonus=bonus, protected_routes=protected_routes)
    assert_exact(got, reference(logits, maps, bonus, protected_routes))
    for selected, baseline in zip(np.asarray(got[0]), np.asarray(original(logits)[0]), strict=True):
        assert set(baseline[:protected_routes]) <= set(selected)
        assert len(set(selected)) == 10


def test_explicit_factor_three_protect_zero_is_bit_exact():
    logits, maps = sample()
    assert_exact(route(logits, maps, bonus=math.log(3), protected_routes=0),
                 reference(logits, maps, math.log(3), 0))


def test_default_is_explicit_factor_two_protect_two():
    logits, maps = sample()
    assert_exact(route(logits, maps), route(logits, maps, bonus=math.log(2), protected_routes=2))


@pytest.mark.parametrize("resident", [True, False])
def test_uniform_residency_does_not_change_original_routes(resident):
    logits, _ = sample()
    maps = (mx.full((512,), 0 if resident else 512, dtype=mx.uint32),) * 3
    got = route(logits, maps)
    assert_exact(got[:2], original(logits))
    assert not np.asarray(got[2]).any()


@pytest.mark.parametrize("protected_routes", range(4))
def test_ties_protect_original_routes_and_need_all_three_projections(protected_routes):
    logits = mx.zeros((1, 1, 512), dtype=mx.bfloat16)
    hot = mx.concatenate((mx.arange(223, dtype=mx.uint32), mx.full((289,), 512, dtype=mx.uint32)))
    missing = mx.full((512,), 512, dtype=mx.uint32)
    all_hot = (hot,) * 3
    got = route(logits, all_hot, protected_routes=protected_routes)
    assert_exact(got, reference(logits, all_hot, DEFAULT_BONUS, protected_routes))
    assert int(got[2].item()) == 10 - protected_routes
    protected = set(range(502, 502 + protected_routes))
    assert protected <= set(np.asarray(got[0]).reshape(-1).tolist())
    partial = (hot, hot, missing)
    assert_exact(route(logits, partial), reference(logits, partial, DEFAULT_BONUS))
    assert not np.asarray(route(logits, partial)[2]).any()


@pytest.mark.parametrize("protected_routes", range(4))
def test_probability_rounding_and_underflow_boundaries(protected_routes):
    values = np.zeros((4, 512), np.float32)
    values[0] = np.linspace(-80, 0, 512)
    values[1] = np.repeat(np.linspace(-4, 4, 64), 8)
    values[2, 0] = 100
    values[2, 1:] = np.linspace(-100, -80, 511)
    values[3, [7, 11, 19, 31, 47, 71, 101, 131, 173, 223, 281]] = 2
    logits = mx.array(values).astype(mx.bfloat16)
    _, maps = sample()
    assert_exact(route(logits, maps, bonus=0, protected_routes=protected_routes)[:2], original(logits))
    assert_exact(route(logits, maps, protected_routes=protected_routes),
                 reference(logits, maps, DEFAULT_BONUS, protected_routes))
    absent = (mx.full((512,), 512, dtype=mx.uint32),) * 3
    assert_exact(route(logits, absent)[:2], original(logits))


def test_lazy_strided_logits():
    logits, maps = sample()
    strided = mx.stack((logits, mx.zeros_like(logits)), axis=-1)[..., 0]
    assert_exact(route(strided, maps), reference(logits, maps, DEFAULT_BONUS))


@pytest.mark.parametrize("bonus", [True, float("nan"), float("inf"), float("-inf"), -1,
                                   math.log(8.01), math.log(1e39), 1000, 10**1000])
def test_invalid_bonus_fails_before_dispatch(bonus):
    logits, maps = sample()
    with pytest.raises(ValueError, match="bonus"):
        route(logits, maps, bonus=bonus)


@pytest.mark.parametrize("protected_routes", [True, None, -1, 4, 2.0, "2"])
def test_invalid_protection_fails_before_dispatch(protected_routes):
    logits, maps = sample()
    with pytest.raises(ValueError, match="protected_routes"):
        route(logits, maps, protected_routes=protected_routes)


def test_invalid_shapes_and_dtypes():
    logits, maps = sample()
    for bad in (logits.astype(mx.float32), logits[..., :511], logits[0], logits[:0]):
        with pytest.raises(ValueError, match="BF16"):
            route(bad, maps)
    with pytest.raises(ValueError, match="BF16"):
        route(logits, (maps[0].astype(mx.int32), *maps[1:]))
