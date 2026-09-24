"""Two-row input projections for one-draft MTP QSA verification."""

from dataclasses import dataclass
from functools import cache
from typing import Any

import mlx.core as mx


@dataclass(frozen=True, eq=False)
class Qwen4MTPQSAProjectionRow:
    """Cooked QSA projections bound to one exact hidden-state row."""

    mixer: Any
    hidden_states: mx.array
    index_queries: mx.array
    raw_index_keys: mx.array
    queries: mx.array
    gate: mx.array
    keys: mx.array
    values: mx.array

    def validate(self, mixer: Any, hidden_states: mx.array) -> None:
        """Reject reuse with another mixer or hidden-state row."""

        if mixer is not self.mixer or hidden_states is not self.hidden_states:
            raise ValueError("MTP QSA projections do not belong to this hidden-state row")


@dataclass(frozen=True, eq=False)
class Qwen4MTPQSAProjectionRows:
    """Two cooked row views backed by one shared two-row preparation."""

    mixer: Any
    rows: tuple[Qwen4MTPQSAProjectionRow, Qwen4MTPQSAProjectionRow]


def _rms_norm(values, weight, eps):
    source = values.astype(mx.float32)
    normalized = source * mx.rsqrt(mx.mean(mx.square(source), axis=-1, keepdims=True) + eps)
    return (normalized * (1 + weight.astype(mx.float32))).astype(values.dtype)


def _partial_rope(values, cosine, sine, rotary_dim):
    rotary = values[..., :rotary_dim]
    half = rotary_dim // 2
    rotated_half = mx.concatenate([-rotary[..., half:], rotary[..., :half]], axis=-1)
    rotated = rotary * cosine + rotated_half * sine
    return mx.concatenate([rotated, values[..., rotary_dim:]], axis=-1)


@cache
def _compiled_qsa_cook(geometry, eps):
    (
        index_query_heads,
        index_kv_heads,
        index_head_dim,
        query_heads,
        kv_heads,
        head_dim,
        rotary_dim,
    ) = geometry
    index_query_width = index_query_heads * index_head_dim

    def cook(
        index_projection,
        query_projection,
        key_projection,
        value_projection,
        index_norm_weight,
        query_norm_weight,
        key_norm_weight,
        cosine,
        sine,
    ):
        index_queries, raw_index_keys = mx.split(
            index_projection,
            [index_query_width],
            axis=-1,
        )
        index_queries = index_queries.reshape(1, 2, index_query_heads, index_head_dim)
        raw_index_keys = raw_index_keys.reshape(1, 2, index_kv_heads, index_head_dim)[:, :, 0]
        index_queries = _partial_rope(
            _rms_norm(index_queries, index_norm_weight, eps[0]),
            cosine,
            sine,
            rotary_dim,
        )

        query_projection = query_projection.reshape(1, 2, query_heads, head_dim * 2)
        queries, gate = mx.split(query_projection, [head_dim], axis=-1)
        gate = gate.reshape(1, 2, -1)
        queries = _partial_rope(
            _rms_norm(queries, query_norm_weight, eps[1]),
            cosine,
            sine,
            rotary_dim,
        )

        keys = key_projection.reshape(1, 2, kv_heads, head_dim)
        keys = _partial_rope(
            _rms_norm(keys, key_norm_weight, eps[2]),
            cosine,
            sine,
            rotary_dim,
        ).transpose(0, 2, 1, 3)
        values = value_projection.reshape(1, 2, kv_heads, head_dim).transpose(0, 2, 1, 3)
        return index_queries, raw_index_keys, queries, gate, keys, values

    return mx.compile(cook)


def prepare_mtp_qsa_projection_rows(
    mixer: Any,
    hidden_states: mx.array,
    position_ids: mx.array,
    shared_rope_factors: tuple[Any, Any] = (None, None),
) -> Qwen4MTPQSAProjectionRows:
    """Project and prepare both verification rows without advancing QSA state."""

    module = mixer.module
    expected_shape = (1, 2, module.hidden_size)
    projection_dtype = mixer._projection_dtype()
    if tuple(hidden_states.shape) != expected_shape or hidden_states.dtype != projection_dtype:
        raise ValueError("MTP QSA projection input must contain two decode rows")
    if tuple(position_ids.shape) != (3, 1, 2) or position_ids.dtype not in (
        mx.int32,
        mx.int64,
        mx.uint32,
        mx.uint64,
    ):
        raise ValueError("MTP QSA projection positions must contain two decode rows")

    index_projection = module.indexer.index_qk_proj(hidden_states)
    query_gate = module.q_proj(hidden_states)
    keys = module.k_proj(hidden_states)
    values = module.v_proj(hidden_states)
    expected_outputs = (
        (
            index_projection,
            (1, 2, (module.index_query_heads + module.index_kv_heads) * module.index_head_dim),
        ),
        (query_gate, (1, 2, module.num_query_heads * module.head_dim * 2)),
        (keys, (1, 2, module.num_kv_heads * module.head_dim)),
        (values, (1, 2, module.num_kv_heads * module.head_dim)),
    )
    if any(
        tuple(value.shape) != shape or value.dtype != projection_dtype
        for value, shape in expected_outputs
    ):
        raise ValueError("MTP QSA projections produced incompatible outputs")

    if all(factor is not None for factor in shared_rope_factors):
        cosine = mx.concatenate([factor.cosine for factor in shared_rope_factors], axis=1)
        sine = mx.concatenate([factor.sine for factor in shared_rope_factors], axis=1)
    elif any(factor is not None for factor in shared_rope_factors):
        raise ValueError("MTP QSA shared RoPE factors are incomplete")
    else:
        prepare_factors = getattr(mixer, "_prepare_mtp_rope_factors", None)
        if not callable(prepare_factors):
            raise TypeError("MTP QSA mixer does not prepare partial RoPE factors")
        factors = prepare_factors(position_ids)
        cosine, sine = factors.cosine, factors.sine
    expected_factors = (1, 2, 1, module.rotary_dim)
    if (
        tuple(cosine.shape) != expected_factors
        or tuple(sine.shape) != expected_factors
        or cosine.dtype != projection_dtype
        or sine.dtype != projection_dtype
    ):
        raise ValueError("MTP QSA shared RoPE factors have incompatible geometry")

    norm_eps = tuple(
        float(norm.eps) for norm in (module.indexer.q_layernorm, module.q_norm, module.k_norm)
    )
    geometry = (
        module.index_query_heads,
        module.index_kv_heads,
        module.index_head_dim,
        module.num_query_heads,
        module.num_kv_heads,
        module.head_dim,
        module.rotary_dim,
    )
    cooked = _compiled_qsa_cook(geometry, norm_eps)(
        index_projection,
        query_gate,
        keys,
        values,
        module.indexer.q_layernorm.weight,
        module.q_norm.weight,
        module.k_norm.weight,
        cosine,
        sine,
    )
    expected_cooked = (
        (cooked[0], (1, 2, module.index_query_heads, module.index_head_dim)),
        (cooked[1], (1, 2, module.index_head_dim)),
        (cooked[2], (1, 2, module.num_query_heads, module.head_dim)),
        (cooked[3], (1, 2, module.num_query_heads * module.head_dim)),
        (cooked[4], (1, module.num_kv_heads, 2, module.head_dim)),
        (cooked[5], (1, module.num_kv_heads, 2, module.head_dim)),
    )
    if any(
        tuple(value.shape) != shape or value.dtype != projection_dtype
        for value, shape in expected_cooked
    ):
        raise ValueError("MTP QSA preparation produced incompatible outputs")

    row_views = tuple(
        Qwen4MTPQSAProjectionRow(
            mixer=mixer,
            hidden_states=hidden_states[:, row : row + 1],
            index_queries=cooked[0][:, row : row + 1],
            raw_index_keys=cooked[1][:, row : row + 1],
            queries=cooked[2][:, row : row + 1],
            gate=cooked[3][:, row : row + 1],
            keys=cooked[4][:, :, row : row + 1],
            values=cooked[5][:, :, row : row + 1],
        )
        for row in range(2)
    )
    return Qwen4MTPQSAProjectionRows(mixer=mixer, rows=row_views)
