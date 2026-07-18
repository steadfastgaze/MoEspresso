"""KV cache policy for the serve path.

Pure and import-light: this module only parses and validates the policy MoEspresso owns.
Actual cache objects and MLX calls live at the runtime edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LIVE_KV_RAW = "raw"
LIVE_KV_Q8 = "mlx_affine_q8"

SUPPORTED_LIVE_KV_FORMATS = frozenset({LIVE_KV_RAW, LIVE_KV_Q8})
SUPPORTED_Q8_GROUP_SIZES = frozenset({32, 64, 128})


class KVPolicyError(ValueError):
    """Invalid or unsupported KV policy."""


@dataclass(frozen=True)
class KVPolicy:
    live_kv_format: str = LIVE_KV_Q8
    kv_group_size: int = 64
    quantized_kv_start: int = 0
    prompt_cache_size: int = 10
    prompt_cache_bytes: int | None = None


def _read_int(source: dict[str, Any], name: str, default: int) -> int:
    value = source.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise KVPolicyError(f"{name} must be an integer")
    if value <= 0:
        raise KVPolicyError(f"{name} must be positive")
    return value


def _read_non_negative_int(source: dict[str, Any], name: str, default: int) -> int:
    value = source.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise KVPolicyError(f"{name} must be an integer")
    if value < 0:
        raise KVPolicyError(f"{name} must be non-negative")
    return value


def _read_optional_positive_int(source: dict[str, Any], name: str) -> int | None:
    value = source.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise KVPolicyError(f"{name} must be an integer")
    if value <= 0:
        raise KVPolicyError(f"{name} must be positive")
    return value


def parse_kv_policy(source: dict[str, Any] | None) -> KVPolicy:
    """Parse known KV config names.

    Symmetric q8 is the default live format. Raw remains an explicit fallback. q6 and
    TurboQuant/vMLX KV are refused.
    """
    source = source or {}
    live = source.get("live_kv_format", LIVE_KV_Q8)
    if live not in SUPPORTED_LIVE_KV_FORMATS:
        raise KVPolicyError(
            f"live_kv_format must be one of {sorted(SUPPORTED_LIVE_KV_FORMATS)} "
            f"(got {live!r})")

    return KVPolicy(
        live_kv_format=live,
        kv_group_size=_read_int(source, "kv_group_size", 64),
        quantized_kv_start=_read_non_negative_int(source, "quantized_kv_start", 0),
        prompt_cache_size=_read_int(source, "prompt_cache_size", 10),
        prompt_cache_bytes=_read_optional_positive_int(source, "prompt_cache_bytes"),
    )


def validate_runtime_policy(policy: KVPolicy) -> None:
    """Fail closed for unsupported runtime policies."""
    if policy.live_kv_format == LIVE_KV_Q8 and policy.kv_group_size not in SUPPORTED_Q8_GROUP_SIZES:
        raise KVPolicyError(
            f"kv_group_size for mlx_affine_q8 must be one of "
            f"{sorted(SUPPORTED_Q8_GROUP_SIZES)} (got {policy.kv_group_size})")


def stream_generate_kv_kwargs(policy: KVPolicy) -> dict:
    """MLX stream_generate kwargs for MoEspresso-owned live KV policy."""
    if policy.live_kv_format == LIVE_KV_RAW:
        return {}
    if policy.live_kv_format == LIVE_KV_Q8:
        validate_runtime_policy(policy)
        return {
            "kv_bits": 8,
            "kv_group_size": policy.kv_group_size,
            "quantized_kv_start": policy.quantized_kv_start,
        }
    raise KVPolicyError(f"unsupported live_kv_format {policy.live_kv_format!r}")


def suffix_token_slice(full_tokens: list[int], prefix_len: int) -> list[int]:
    """Return the suffix by token offset, never by string slicing/re-tokenizing."""
    if prefix_len < 0 or prefix_len > len(full_tokens):
        raise KVPolicyError(
            f"prefix length {prefix_len} outside token sequence length {len(full_tokens)}")
    return list(full_tokens[prefix_len:])
