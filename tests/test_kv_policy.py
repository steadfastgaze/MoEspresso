"""KV policy: parse config names, fail closed, expose only proven live formats.

These tests are pure. They pin the runtime contract: MoEspresso owns the policy, symmetric
q8 is the default live format, raw is an explicit fallback, and q6/TurboQuant are refused.
"""

from __future__ import annotations

import pytest

from moespresso.runtime.kv_policy import (
    KVPolicyError,
    parse_kv_policy,
    stream_generate_kv_kwargs,
    suffix_token_slice,
    validate_runtime_policy,
)


def test_kv_policy_defaults_to_symmetric_q8_from_the_first_token():
    p = parse_kv_policy({})
    assert p.live_kv_format == "mlx_affine_q8"
    assert p.kv_group_size == 64
    assert p.quantized_kv_start == 0
    assert p.prompt_cache_size == 10
    assert p.prompt_cache_bytes is None


def test_kv_policy_parses_declared_q8_config():
    p = parse_kv_policy({
        "live_kv_format": "mlx_affine_q8",
        "kv_group_size": 64,
        "quantized_kv_start": 5000,
        "prompt_cache_size": 4,
        "prompt_cache_bytes": 1024,
    })
    assert p.live_kv_format == "mlx_affine_q8"
    assert p.kv_group_size == 64
    assert p.quantized_kv_start == 5000
    assert p.prompt_cache_size == 4
    assert p.prompt_cache_bytes == 1024


def test_kv_policy_parses_raw_as_explicit_fallback():
    p = parse_kv_policy({"live_kv_format": "raw"})
    assert p.live_kv_format == "raw"
    assert stream_generate_kv_kwargs(p) == {}


@pytest.mark.parametrize("bad", ["q6", "mlx_affine_q6", "turboquant", "tq", "vmlx_tq"])
def test_kv_policy_rejects_unsupported_live_formats(bad):
    with pytest.raises(KVPolicyError, match="live_kv_format"):
        parse_kv_policy({"live_kv_format": bad})


def test_kv_policy_rejects_invalid_numeric_values():
    with pytest.raises(KVPolicyError, match="kv_group_size"):
        parse_kv_policy({"kv_group_size": 0})
    with pytest.raises(KVPolicyError, match="quantized_kv_start"):
        parse_kv_policy({"quantized_kv_start": -1})
    with pytest.raises(KVPolicyError, match="prompt_cache_size"):
        parse_kv_policy({"prompt_cache_size": 0})


def test_runtime_policy_allows_raw_and_q8():
    validate_runtime_policy(parse_kv_policy({"live_kv_format": "raw"}))
    validate_runtime_policy(parse_kv_policy({"live_kv_format": "mlx_affine_q8"}))


def test_q8_policy_uses_mlx_stock_stream_generate_knobs():
    policy = parse_kv_policy({
        "live_kv_format": "mlx_affine_q8",
        "kv_group_size": 128,
        "quantized_kv_start": 12,
    })
    assert stream_generate_kv_kwargs(policy) == {
        "kv_bits": 8,
        "kv_group_size": 128,
        "quantized_kv_start": 12,
    }
    assert stream_generate_kv_kwargs(parse_kv_policy({})) == {
        "kv_bits": 8,
        "kv_group_size": 64,
        "quantized_kv_start": 0,
    }


def test_q8_policy_rejects_unproven_group_sizes_at_runtime():
    policy = parse_kv_policy({
        "live_kv_format": "mlx_affine_q8",
        "kv_group_size": 7,
    })
    with pytest.raises(KVPolicyError, match="kv_group_size"):
        validate_runtime_policy(policy)


def test_suffix_token_slice_uses_token_offsets_not_strings():
    full_tokens = [10, 20, 30, 40]
    assert suffix_token_slice(full_tokens, 2) == [30, 40]
    assert suffix_token_slice(full_tokens, 4) == []


def test_suffix_token_slice_rejects_impossible_prefix_lengths():
    with pytest.raises(KVPolicyError, match="prefix length"):
        suffix_token_slice([1, 2], -1)
    with pytest.raises(KVPolicyError, match="prefix length"):
        suffix_token_slice([1, 2], 3)
