"""Small NumPy references for released Qwen4-Exp architecture primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _sigmoid(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    output = np.empty_like(source)
    positive = source >= 0
    output[positive] = np.float32(1) / (np.float32(1) + np.exp(-source[positive]))
    exponential = np.exp(source[~positive])
    output[~positive] = exponential / (np.float32(1) + exponential)
    return output


def _silu(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    return source * _sigmoid(source)


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def ngram_layer_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    *,
    ple_layer_index: int,
    seed: int,
) -> np.ndarray:
    """Build the deterministic odd multipliers stored by the release."""
    vocabulary = _positive_integer("unigram_vocab_size", unigram_vocab_size)
    order = _positive_integer("ngram_size", ngram_size)
    if isinstance(ple_layer_index, bool) or not isinstance(ple_layer_index, int):
        raise TypeError("ple_layer_index must be an int")
    if ple_layer_index < 0:
        raise ValueError("ple_layer_index must be nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // vocabulary) // 2)
    base_seed = seed + _PLE_LAYER_PRIME * ple_layer_index
    multipliers = []
    for index in range(order):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return np.asarray(multipliers, dtype=np.int64)


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


@dataclass(frozen=True)
class NGramTableGeometry:
    """Per-head prime table sizes and concatenated row offsets."""

    sizes: np.ndarray
    offsets: np.ndarray
    total_rows: int
    padded_rows: int


def ngram_table_geometry(
    *,
    ngram_size: int,
    heads_per_ngram: int,
    vocab_size_base: int,
    ple_layer_index: int,
    make_divisible_by: int = 128,
) -> NGramTableGeometry:
    """Construct the released distinct-prime table layout."""
    order = _positive_integer("ngram_size", ngram_size)
    heads = _positive_integer("heads_per_ngram", heads_per_ngram)
    base = _positive_integer("vocab_size_base", vocab_size_base)
    divisor = _positive_integer("make_divisible_by", make_divisible_by)
    if order < 2:
        raise ValueError("ngram_size must be at least two")
    if isinstance(ple_layer_index, bool) or not isinstance(ple_layer_index, int):
        raise TypeError("ple_layer_index must be an int")
    if ple_layer_index < 0:
        raise ValueError("ple_layer_index must be nonnegative")
    total_heads = (order - 1) * heads
    sizes = []
    for head_idx in range(total_heads):
        global_head = ple_layer_index * total_heads + head_idx
        sizes.append(_nth_prime_after(base - 1, global_head + 1))
    size_array = np.asarray(sizes, dtype=np.int64)
    offsets = np.zeros_like(size_array)
    if offsets.size > 1:
        offsets[1:] = np.cumsum(size_array[:-1], dtype=np.int64)
    total_rows = int(size_array.sum(dtype=np.int64))
    padded_rows = math.ceil(total_rows / divisor) * divisor
    return NGramTableGeometry(
        sizes=size_array,
        offsets=offsets,
        total_rows=total_rows,
        padded_rows=padded_rows,
    )


def _shift_right_ignore_eos(
    token_ids: np.ndarray,
    *,
    shift: int,
    eos_token_id: int,
) -> np.ndarray:
    tokens = np.asarray(token_ids, dtype=np.int64)
    if tokens.ndim != 2:
        raise ValueError("token_ids must have shape [batch, tokens]")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise TypeError("shift must be an int")
    if shift < 0:
        raise ValueError("shift must be nonnegative")
    if shift == 0:
        return tokens.copy()
    output = np.full_like(tokens, int(eos_token_id))
    for batch in range(tokens.shape[0]):
        previous_eos = -1
        for position in range(tokens.shape[1]):
            source = position - shift
            if source >= 0 and position - (previous_eos + 1) >= shift:
                output[batch, position] = tokens[batch, source]
            if tokens[batch, position] == eos_token_id:
                previous_eos = position
    return output


@dataclass(frozen=True)
class NGramHashResult:
    """Physical embedding row ids and the next short token history."""

    row_ids: np.ndarray
    next_context: np.ndarray


def ngram_embedding_row_ids(
    input_ids: np.ndarray,
    *,
    eos_token_id: int,
    ngram_size: int,
    heads_per_ngram: int,
    multipliers: np.ndarray,
    table_geometry: NGramTableGeometry,
    previous_context: np.ndarray | None = None,
    valid_tokens: np.ndarray | None = None,
) -> NGramHashResult:
    """Map current tokens to released bigram and higher-order table rows."""
    order = _positive_integer("ngram_size", ngram_size)
    heads = _positive_integer("heads_per_ngram", heads_per_ngram)
    if order < 2:
        raise ValueError("ngram_size must be at least two")
    current = np.asarray(input_ids, dtype=np.int64)
    if current.ndim != 2 or not all(current.shape):
        raise ValueError("input_ids must have shape [batch, tokens]")
    if valid_tokens is not None:
        mask = np.asarray(valid_tokens, dtype=bool)
        if mask.shape != current.shape:
            raise ValueError("valid_tokens must match input_ids")
        current = np.where(mask, current, np.int64(eos_token_id))
    factors = np.asarray(multipliers, dtype=np.int64)
    if factors.shape != (order,):
        raise ValueError("multipliers must have one value per n-gram position")
    total_heads = (order - 1) * heads
    if table_geometry.sizes.shape != (total_heads,) or table_geometry.offsets.shape != (
        total_heads,
    ):
        raise ValueError("table geometry does not match n-gram heads")
    context_len = order - 1
    if previous_context is None:
        previous = np.full((current.shape[0], context_len), int(eos_token_id), dtype=np.int64)
    else:
        previous = np.asarray(previous_context, dtype=np.int64)
        if previous.shape != (current.shape[0], context_len):
            raise ValueError("previous_context has an invalid shape")
    history = np.concatenate((previous, current), axis=-1)
    shifted = [
        _shift_right_ignore_eos(history, shift=shift, eos_token_id=eos_token_id)
        for shift in range(order)
    ]
    blocks = []
    for ngram in range(2, order + 1):
        start = (ngram - 2) * heads
        end = start + heads
        mixed = shifted[0] * factors[0]
        for position in range(1, ngram):
            mixed = np.bitwise_xor(mixed, shifted[position] * factors[position])
        block = np.remainder(mixed[..., None], table_geometry.sizes[start:end])
        blocks.append(block + table_geometry.offsets[start:end])
    all_rows = np.concatenate(blocks, axis=-1)
    return NGramHashResult(
        row_ids=all_rows[:, -current.shape[1] :],
        next_context=history[:, -context_len:],
    )


def grouped_zero_centered_rms_norm(
    values: np.ndarray,
    weight: np.ndarray,
    *,
    group_size: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize independent residual branches and apply ``1 + weight``."""
    width = _positive_integer("group_size", group_size)
    source = np.asarray(values)
    scale = np.asarray(weight, dtype=np.float32)
    if source.ndim < 1 or not source.shape[-1] or source.shape[-1] % width:
        raise ValueError("values width must be a positive multiple of group_size")
    if scale.shape != (source.shape[-1],):
        raise ValueError("weight must match the values width")
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eps must be finite and positive")
    grouped = source.astype(np.float32).reshape(*source.shape[:-1], -1, width)
    variance = np.mean(grouped * grouped, axis=-1, keepdims=True, dtype=np.float32)
    normalized = grouped / np.sqrt(variance + np.float32(epsilon))
    normalized = normalized.reshape(source.shape)
    return (normalized * (np.float32(1) + scale)).astype(source.dtype)


@dataclass(frozen=True)
class GatedResidualRead:
    """Observable values from one gated-residual read."""

    normalized_input: np.ndarray
    input_mix_weight: np.ndarray
    mixed_input: np.ndarray
    injection_weights: np.ndarray | None


def gated_residual_read(
    hyper_input: np.ndarray,
    norm_weight: np.ndarray,
    down_weight: np.ndarray,
    up_weight: np.ndarray,
    *,
    branch_count: int,
    hidden_size: int,
    injection_weight: np.ndarray | None = None,
    eps: float = 1e-6,
) -> GatedResidualRead:
    """Read the released four-branch residual through its low-rank gate."""
    branches = _positive_integer("branch_count", branch_count)
    hidden = _positive_integer("hidden_size", hidden_size)
    source = np.asarray(hyper_input)
    expanded = branches * hidden
    if source.ndim < 1 or source.shape[-1] != expanded:
        raise ValueError("hyper_input does not match branch geometry")
    down = np.asarray(down_weight, dtype=np.float32)
    up = np.asarray(up_weight, dtype=np.float32)
    if down.ndim != 2 or down.shape[1] != expanded:
        raise ValueError("down_weight must have shape [lowrank, expanded_width]")
    if up.shape != (expanded, down.shape[0]):
        raise ValueError("up_weight must have shape [expanded_width, lowrank]")

    normalized = grouped_zero_centered_rms_norm(
        source,
        norm_weight,
        group_size=hidden,
        eps=eps,
    )
    lowrank = _silu(normalized.astype(np.float32) @ down.T / np.float32(branches))
    mix = _sigmoid(lowrank @ up.T).reshape(*source.shape[:-1], branches, hidden)
    streams = normalized.reshape(*source.shape[:-1], branches, hidden)
    mixed = np.mean(mix * streams, axis=-2, dtype=np.float32).astype(source.dtype)

    injection = None
    if injection_weight is not None:
        inject = np.asarray(injection_weight, dtype=np.float32)
        if inject.shape != (branches, expanded):
            raise ValueError("injection_weight must have shape [branch_count, expanded_width]")
        injection = np.float32(2) * _sigmoid(
            normalized.astype(np.float32) @ inject.T / np.float32(branches)
        )
        injection = injection.astype(source.dtype)
    return GatedResidualRead(
        normalized_input=normalized,
        input_mix_weight=mix.astype(source.dtype),
        mixed_input=mixed,
        injection_weights=injection,
    )


def gated_residual_write(
    hyper_input: np.ndarray,
    block_output: np.ndarray,
    injection_weights: np.ndarray,
) -> np.ndarray:
    """Inject one sublayer output into every expanded residual branch."""
    residual = np.asarray(hyper_input)
    output = np.asarray(block_output)
    injection = np.asarray(injection_weights)
    if output.ndim < 1 or residual.shape[:-1] != output.shape[:-1]:
        raise ValueError("block_output batch dimensions must match hyper_input")
    if not output.shape[-1] or residual.shape[-1] % output.shape[-1]:
        raise ValueError("block_output width must divide the expanded residual width")
    branches = residual.shape[-1] // output.shape[-1]
    if injection.shape != (*output.shape[:-1], branches):
        raise ValueError("injection_weights do not match branch geometry")
    update = output[..., None, :] * injection[..., :, None]
    return residual + update.reshape(residual.shape)


def ple_gated_value(
    expanded_hidden: np.ndarray,
    projected_key: np.ndarray,
    projected_value: np.ndarray,
    query_norm_weight: np.ndarray,
    key_norm_weight: np.ndarray,
    *,
    branch_count: int,
    hidden_size: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply the signed-square-root PLE gate as an FP32 mathematical reference."""
    branches = _positive_integer("branch_count", branch_count)
    hidden = _positive_integer("hidden_size", hidden_size)
    expanded = branches * hidden
    query = np.asarray(expanded_hidden)
    key = np.asarray(projected_key)
    value = np.asarray(projected_value)
    if query.shape != key.shape or query.ndim < 1 or query.shape[-1] != expanded:
        raise ValueError("expanded_hidden and projected_key must match branch geometry")
    if value.shape != (*query.shape[:-1], hidden):
        raise ValueError("projected_value must have one hidden row per token")
    query_normed = grouped_zero_centered_rms_norm(
        query,
        query_norm_weight,
        group_size=hidden,
        eps=eps,
    ).reshape(*query.shape[:-1], branches, hidden)
    key_normed = grouped_zero_centered_rms_norm(
        key,
        key_norm_weight,
        group_size=hidden,
        eps=eps,
    ).reshape(*key.shape[:-1], branches, hidden)
    score = np.sum(
        query_normed.astype(np.float32) * key_normed.astype(np.float32),
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    ) / np.float32(math.sqrt(hidden))
    transformed = np.sign(score) * np.sqrt(np.maximum(np.abs(score), np.float32(1e-6)))
    gate = _sigmoid(transformed)
    return (gate * value.astype(np.float32)[..., None, :]).astype(value.dtype)


@dataclass(frozen=True)
class PleConvolution:
    """Output and committed history from the PLE depthwise convolution."""

    output: np.ndarray
    next_state: np.ndarray


def ple_depthwise_convolution(
    values: np.ndarray,
    weight: np.ndarray,
    *,
    dilation: int,
    previous_state: np.ndarray | None = None,
) -> PleConvolution:
    """Apply the released causal, dilated, depthwise SiLU convolution."""
    source = np.asarray(values)
    kernel = np.asarray(weight)
    spacing = _positive_integer("dilation", dilation)
    if source.ndim != 3 or not all(source.shape):
        raise ValueError("values must have shape [batch, tokens, channels]")
    if kernel.ndim == 3 and kernel.shape[1] == 1:
        kernel = kernel[:, 0, :]
    if kernel.ndim != 2 or kernel.shape[0] != source.shape[-1] or not kernel.shape[-1]:
        raise ValueError("weight must have shape [channels, kernel_size]")
    state_len = (kernel.shape[-1] - 1) * spacing
    if previous_state is None:
        previous = np.zeros((source.shape[0], state_len, source.shape[-1]), dtype=source.dtype)
    else:
        state = np.asarray(previous_state)
        if state.shape != (source.shape[0], source.shape[-1], state_len):
            raise ValueError("previous_state does not match convolution geometry")
        previous = np.swapaxes(state, 1, 2)
    history = np.concatenate((previous, source), axis=1)
    output = np.zeros_like(source, dtype=np.float32)
    for tap in range(kernel.shape[-1]):
        start = tap * spacing
        output += history[:, start : start + source.shape[1], :].astype(np.float32) * kernel[
            None, None, :, tap
        ].astype(np.float32)
    next_state = history[:, -state_len:] if state_len else history[:, :0]
    return PleConvolution(
        output=_silu(output).astype(source.dtype),
        next_state=np.swapaxes(next_state, 1, 2).copy(),
    )


@dataclass(frozen=True)
class PleOutput:
    """Output and convolution state from the post-gate PLE path."""

    output: np.ndarray
    next_conv_state: np.ndarray


def ple_finish(
    gated_value: np.ndarray,
    conv_norm_weight: np.ndarray,
    conv_weight: np.ndarray,
    *,
    branch_count: int,
    hidden_size: int,
    dilation: int,
    previous_conv_state: np.ndarray | None = None,
    valid_tokens: np.ndarray | None = None,
    eps: float = 1e-6,
) -> PleOutput:
    """Normalize, mask and convolve an already gated PLE value."""
    branches = _positive_integer("branch_count", branch_count)
    hidden = _positive_integer("hidden_size", hidden_size)
    gated = np.asarray(gated_value)
    if gated.ndim != 4 or gated.shape[-2:] != (branches, hidden):
        raise ValueError("gated_value must have shape [batch, tokens, branches, hidden_size]")
    flattened = gated.reshape(*gated.shape[:-2], branches * hidden)
    normalized = grouped_zero_centered_rms_norm(
        flattened,
        conv_norm_weight,
        group_size=hidden,
        eps=eps,
    )
    if valid_tokens is not None:
        mask = np.asarray(valid_tokens, dtype=bool)
        if mask.shape != gated.shape[:2]:
            raise ValueError("valid_tokens must have shape [batch, tokens]")
        flattened = np.where(mask[..., None], flattened, np.zeros((), dtype=flattened.dtype))
        normalized = np.where(mask[..., None], normalized, np.zeros((), dtype=normalized.dtype))
    convolved = ple_depthwise_convolution(
        normalized,
        conv_weight,
        dilation=dilation,
        previous_state=previous_conv_state,
    )
    return PleOutput(
        output=flattened + convolved.output,
        next_conv_state=convolved.next_state,
    )


@dataclass(frozen=True)
class GatedDeltaStep:
    """Output and committed state from one gated-delta token."""

    output: np.ndarray
    state: np.ndarray


def gated_delta_step(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    state: np.ndarray,
    *,
    eps: float = 1e-6,
) -> GatedDeltaStep:
    """Apply the released FP32 recurrent gated-delta update for one token."""
    q = np.asarray(query, dtype=np.float32)
    k = np.asarray(key, dtype=np.float32)
    v = np.asarray(value, dtype=np.float32)
    g = np.asarray(log_decay, dtype=np.float32)
    write = np.asarray(beta, dtype=np.float32)
    previous = np.asarray(state, dtype=np.float32)
    if q.ndim != 2 or k.shape != q.shape:
        raise ValueError("query and key must have matching [heads, key_dim] shapes")
    if v.ndim != 2 or v.shape[0] != q.shape[0]:
        raise ValueError("value must have shape [heads, value_dim]")
    if previous.shape != (q.shape[0], q.shape[1], v.shape[1]):
        raise ValueError("state must have shape [heads, key_dim, value_dim]")
    if g.shape != (q.shape[0],) or write.shape != (q.shape[0],):
        raise ValueError("log_decay and beta must have one value per head")
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eps must be finite and positive")

    q_norm = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + np.float32(epsilon))
    k_norm = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + np.float32(epsilon))
    q_norm /= np.float32(math.sqrt(q.shape[-1]))
    committed = previous * np.exp(g)[..., None, None]
    prediction = np.einsum("hkv,hk->hv", committed, k_norm, optimize=True)
    residual = (v - prediction) * write[..., None]
    committed = committed + k_norm[..., :, None] * residual[..., None, :]
    output = np.einsum("hkv,hk->hv", committed, q_norm, optimize=True)
    return GatedDeltaStep(output=output, state=committed)


@dataclass(frozen=True)
class Qwen4RouterSelection:
    """Observable values from the released FP32-softmax top-k router."""

    logits: np.ndarray
    scores: np.ndarray
    indices: np.ndarray


def qwen4_topk_router(
    hidden_states: np.ndarray,
    weight: np.ndarray,
    *,
    top_k: int,
    normalize_topk: bool,
) -> Qwen4RouterSelection:
    """Select Qwen4 experts after a full FP32 softmax."""
    selected = _positive_integer("top_k", top_k)
    source = np.asarray(hidden_states)
    router_weight = np.asarray(weight)
    if source.ndim < 1 or not source.shape[-1]:
        raise ValueError("hidden_states must have a positive feature width")
    if router_weight.ndim != 2 or router_weight.shape[1] != source.shape[-1]:
        raise ValueError("weight must have shape [experts, hidden_size]")
    if selected > router_weight.shape[0]:
        raise ValueError("top_k cannot exceed the expert count")

    logits = (source.astype(np.float32) @ router_weight.astype(np.float32).T).astype(source.dtype)
    logits32 = logits.astype(np.float32)
    shifted = logits32 - np.max(logits32, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True, dtype=np.float32)
    order = np.argsort(-probabilities, axis=-1, kind="stable")[..., :selected]
    scores = np.take_along_axis(probabilities, order, axis=-1)
    if normalize_topk:
        scores /= np.sum(scores, axis=-1, keepdims=True, dtype=np.float32)
    return Qwen4RouterSelection(
        logits=logits,
        scores=scores.astype(logits.dtype),
        indices=order.astype(np.int64),
    )


@dataclass(frozen=True)
class Qwen4MoEOutput:
    """Output and router observables from the released sparse MoE block."""

    output: np.ndarray
    router: Qwen4RouterSelection


def qwen4_sparse_moe(
    hidden_states: np.ndarray,
    router_weight: np.ndarray,
    expert_gate_up_weight: np.ndarray,
    expert_down_weight: np.ndarray,
    shared_gate_weight: np.ndarray,
    shared_up_weight: np.ndarray,
    shared_down_weight: np.ndarray,
    shared_router_weight: np.ndarray,
    *,
    top_k: int,
    normalize_topk: bool,
) -> Qwen4MoEOutput:
    """Evaluate a small FP32 Qwen4 MoE in expert-major accumulation order."""
    source = np.asarray(hidden_states)
    if source.ndim < 2 or not source.shape[-1]:
        raise ValueError("hidden_states must include tokens and features")
    hidden_size = source.shape[-1]
    flat = source.reshape(-1, hidden_size)
    gate_up = np.asarray(expert_gate_up_weight)
    down = np.asarray(expert_down_weight)
    if gate_up.ndim != 3 or gate_up.shape[2] != hidden_size or gate_up.shape[1] % 2:
        raise ValueError("expert_gate_up_weight has invalid geometry")
    intermediate_size = gate_up.shape[1] // 2
    if down.shape != (gate_up.shape[0], hidden_size, intermediate_size):
        raise ValueError("expert_down_weight has invalid geometry")

    router = qwen4_topk_router(
        flat,
        router_weight,
        top_k=top_k,
        normalize_topk=normalize_topk,
    )
    routed = np.zeros_like(flat)
    for expert in range(gate_up.shape[0]):
        token_positions, topk_positions = np.where(router.indices == expert)
        if token_positions.size == 0:
            continue
        combined = flat[token_positions].astype(np.float32) @ gate_up[expert].astype(np.float32).T
        gate, up = np.split(combined, 2, axis=-1)
        activated = _silu(gate) * up
        expert_output = activated @ down[expert].astype(np.float32).T
        expert_output *= router.scores[token_positions, topk_positions, None].astype(np.float32)
        for row, token in zip(expert_output, token_positions, strict=True):
            routed[token] = (routed[token].astype(np.float32) + row).astype(routed.dtype)

    shared_gate = np.asarray(shared_gate_weight)
    shared_up = np.asarray(shared_up_weight)
    shared_down = np.asarray(shared_down_weight)
    shared_router = np.asarray(shared_router_weight)
    if shared_gate.ndim != 2 or shared_gate.shape[1] != hidden_size:
        raise ValueError("shared_gate_weight has invalid geometry")
    if shared_up.shape != shared_gate.shape:
        raise ValueError("shared_up_weight must match shared_gate_weight")
    if shared_down.shape != (hidden_size, shared_gate.shape[0]):
        raise ValueError("shared_down_weight has invalid geometry")
    if shared_router.shape != (1, hidden_size):
        raise ValueError("shared_router_weight must have shape [1, hidden_size]")
    shared = _silu(flat.astype(np.float32) @ shared_gate.astype(np.float32).T)
    shared *= flat.astype(np.float32) @ shared_up.astype(np.float32).T
    shared = shared @ shared_down.astype(np.float32).T
    shared *= _sigmoid(flat.astype(np.float32) @ shared_router.astype(np.float32).T)
    output = (routed.astype(np.float32) + shared).astype(source.dtype)
    return Qwen4MoEOutput(output=output.reshape(source.shape), router=router)
