"""Pure manifest contract for package-owned Qwen4 PLE rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import math


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007
_QWEN4_FAMILIES = frozenset({"qwen4_exp", "qwen4_exp_text"})
_COMPONENT_FIELDS = frozenset(
    {
        "schema",
        "layer_index",
        "dtype",
        "row_width",
        "row_bytes",
        "logical_rows",
        "padded_rows",
        "rows_per_shard",
        "ngram_size",
        "heads_per_ngram",
        "multipliers",
        "table_sizes",
        "table_offsets",
        "shards",
    }
)


class Qwen4PLEProviderError(RuntimeError):
    """Raised when PLE storage metadata or a selected-row read is invalid."""


@dataclass(frozen=True)
class Qwen4PLEProviderContract:
    """Model-owned facts that a PLE provider component must reproduce exactly."""

    layer_index: int
    dtype: str
    row_width: int
    row_bytes: int
    logical_rows: int
    padded_rows: int
    rows_per_shard: int
    shard_count: int
    ngram_size: int
    heads_per_ngram: int
    multipliers: tuple[int, ...]
    table_sizes: tuple[int, ...]
    table_offsets: tuple[int, ...]


def _fail(message: str) -> Qwen4PLEProviderError:
    return Qwen4PLEProviderError(message)


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{field} must be an integer >= {minimum}")
    return value


def _integer_tuple(
    value: object,
    *,
    field: str,
    length: int,
    minimum: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(f"{field} must be an array")
    if len(value) != length:
        raise _fail(f"{field} must contain {length} entries")
    return tuple(
        _integer(item, field=f"{field}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _architecture_config(architecture: Mapping[str, object]) -> Mapping[str, object]:
    family = architecture.get("family")
    text_type = architecture.get("text_model_type")
    config = architecture.get("config")
    if not isinstance(config, Mapping):
        raise _fail("architecture.config must be an object")
    config_type = config.get("model_type")
    if not any(value in _QWEN4_FAMILIES for value in (family, text_type, config_type)):
        raise _fail("architecture does not declare the Qwen4-Exp text family")
    return config


def derive_qwen4_ple_provider_contract(
    architecture: Mapping[str, object],
) -> Qwen4PLEProviderContract | None:
    """Derive PLE storage facts from validated package architecture fields."""
    if not isinstance(architecture, Mapping):
        raise _fail("architecture must be an object")
    config = _architecture_config(architecture)
    layer_ids = config.get("ple_layer_ids")
    if layer_ids in (None, []):
        return None
    if (
        isinstance(layer_ids, (str, bytes))
        or not isinstance(layer_ids, Sequence)
        or len(layer_ids) != 1
    ):
        raise _fail("the PLE provider supports exactly one declared PLE layer")
    one_based_layer = _integer(layer_ids[0], field="ple_layer_ids[0]", minimum=1)
    layer_count = _integer(config.get("num_hidden_layers"), field="num_hidden_layers", minimum=1)
    if one_based_layer > layer_count:
        raise _fail("ple_layer_ids[0] exceeds num_hidden_layers")
    layer_types = config.get("layer_types")
    if (
        isinstance(layer_types, (str, bytes))
        or not isinstance(layer_types, Sequence)
        or len(layer_types) != layer_count
    ):
        raise _fail("layer_types must cover every decoder layer")
    if layer_types[one_based_layer - 1] != "linear_attention":
        raise _fail("PLE must be attached to a linear_attention layer")
    if config.get("dtype") != "bfloat16":
        raise _fail("Qwen4 PLE source dtype must be bfloat16")

    ngram_size = _integer(config.get("ngram_size"), field="ngram_size", minimum=2)
    heads_per_ngram = _integer(
        config.get("heads_per_ngram"), field="heads_per_ngram", minimum=1
    )
    head_count = (ngram_size - 1) * heads_per_ngram
    embedding_width = _integer(config.get("ple_embed_dim"), field="ple_embed_dim", minimum=1)
    if embedding_width % head_count:
        raise _fail("ple_embed_dim must be divisible by the PLE head count")
    hidden_size = _integer(config.get("hidden_size"), field="hidden_size", minimum=1)
    if embedding_width != hidden_size:
        raise _fail("ple_embed_dim must match hidden_size")

    vocabulary = _integer(config.get("vocab_size"), field="vocab_size", minimum=1)
    base = _integer(
        config.get("ngram_vocab_size_base"),
        field="ngram_vocab_size_base",
        minimum=2,
    )
    divisor = _integer(
        config.get("make_ngram_vocab_size_divisible_by"),
        field="make_ngram_vocab_size_divisible_by",
        minimum=1,
    )
    shard_count = _integer(
        config.get("split_ngram_parts"), field="split_ngram_parts", minimum=1
    )
    seed = _integer(config.get("seed"), field="seed")

    table_sizes = tuple(
        _nth_prime_after(base - 1, head_index + 1)
        for head_index in range(head_count)
    )
    offsets = []
    logical_rows = 0
    for size in table_sizes:
        offsets.append(logical_rows)
        logical_rows += size
    padded_rows = math.ceil(logical_rows / divisor) * divisor
    if padded_rows % shard_count:
        raise _fail("padded PLE rows must divide evenly across split_ngram_parts")

    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // vocabulary) // 2)
    multipliers = []
    ple_layer_index = 0
    base_seed = seed + _PLE_LAYER_PRIME * ple_layer_index
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)

    row_width = embedding_width // head_count
    return Qwen4PLEProviderContract(
        layer_index=one_based_layer - 1,
        dtype="BF16",
        row_width=row_width,
        row_bytes=row_width * 2,
        logical_rows=logical_rows,
        padded_rows=padded_rows,
        rows_per_shard=padded_rows // shard_count,
        shard_count=shard_count,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        multipliers=tuple(multipliers),
        table_sizes=table_sizes,
        table_offsets=tuple(offsets),
    )


def parse_qwen4_ple_component_contract(
    component: Mapping[str, object],
) -> Qwen4PLEProviderContract:
    """Validate a provider component's repeated semantic and row geometry."""
    if not isinstance(component, Mapping):
        raise _fail("ple_provider must be an object")
    unknown = set(component) - _COMPONENT_FIELDS
    missing = _COMPONENT_FIELDS - set(component)
    if unknown:
        raise _fail(f"ple_provider has unknown fields: {sorted(unknown)}")
    if missing:
        raise _fail(f"ple_provider is missing fields: {sorted(missing)}")
    if component.get("schema") != "qwen4_ple_provider_v1":
        raise _fail("ple_provider schema must be qwen4_ple_provider_v1")

    layer_index = _integer(component.get("layer_index"), field="layer_index")
    dtype = component.get("dtype")
    if dtype != "BF16":
        raise _fail("dtype must be BF16")
    row_width = _integer(component.get("row_width"), field="row_width", minimum=1)
    row_bytes = _integer(component.get("row_bytes"), field="row_bytes", minimum=1)
    if row_bytes != row_width * 2:
        raise _fail("row_bytes must equal two bytes per BF16 element")
    logical_rows = _integer(component.get("logical_rows"), field="logical_rows", minimum=1)
    padded_rows = _integer(component.get("padded_rows"), field="padded_rows", minimum=1)
    rows_per_shard = _integer(
        component.get("rows_per_shard"), field="rows_per_shard", minimum=1
    )
    if logical_rows > padded_rows:
        raise _fail("logical_rows must not exceed padded_rows")
    if padded_rows - logical_rows >= rows_per_shard:
        raise _fail("padding must be confined to the final shard")

    ngram_size = _integer(component.get("ngram_size"), field="ngram_size", minimum=2)
    heads_per_ngram = _integer(
        component.get("heads_per_ngram"), field="heads_per_ngram", minimum=1
    )
    head_count = (ngram_size - 1) * heads_per_ngram
    multipliers = _integer_tuple(
        component.get("multipliers"),
        field="multipliers",
        length=ngram_size,
        minimum=0,
    )
    table_sizes = _integer_tuple(
        component.get("table_sizes"),
        field="table_sizes",
        length=head_count,
        minimum=1,
    )
    table_offsets = _integer_tuple(
        component.get("table_offsets"),
        field="table_offsets",
        length=head_count,
        minimum=0,
    )
    expected_offsets = []
    offset = 0
    for size in table_sizes:
        expected_offsets.append(offset)
        offset += size
    if tuple(expected_offsets) != table_offsets:
        raise _fail("table_offsets must concatenate table_sizes")
    if offset != logical_rows:
        raise _fail("table_sizes must sum to logical_rows")

    shards = component.get("shards")
    if not isinstance(shards, list) or not shards:
        raise _fail("shards must be a non-empty array")
    shard_count = len(shards)
    if padded_rows != rows_per_shard * shard_count:
        raise _fail("padded_rows must equal rows_per_shard times shard count")
    return Qwen4PLEProviderContract(
        layer_index=layer_index,
        dtype=dtype,
        row_width=row_width,
        row_bytes=row_bytes,
        logical_rows=logical_rows,
        padded_rows=padded_rows,
        rows_per_shard=rows_per_shard,
        shard_count=shard_count,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        multipliers=multipliers,
        table_sizes=table_sizes,
        table_offsets=table_offsets,
    )


def qwen4_ple_contract_mismatches(
    actual: Qwen4PLEProviderContract,
    expected: Qwen4PLEProviderContract,
) -> tuple[str, ...]:
    """Name every provider field that disagrees with architecture authority."""
    return tuple(
        item.name
        for item in fields(expected)
        if getattr(actual, item.name) != getattr(expected, item.name)
    )
