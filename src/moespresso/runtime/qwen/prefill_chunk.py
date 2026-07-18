"""Served prefill chunk size for the qwen3_5_moe sorted routed-MoE path.

Default-on for long prompts: the shipped serving raises the prompt chunk to 4096
once the prompt is long enough to be multi-chunk at 4096, and keeps the mlx_lm
chunk for short prompts. `MOESPRESSO_QWEN_PREFILL_CHUNK=<n>` overrides the value;
the env wins over the default.

mlx_lm chunks a long prompt at `prefill_step_size` (mlx_lm default 2048): each
chunk is one forward pass, and the sorted routed-MoE route
(`sorted_switch_glu.SortedKQuantSwitchGLU`) engages per chunk. Every chunk reads
each active expert's quantized weights once for its whole segment, so a prompt
split into more chunks re-reads the expert weights once per chunk. At a 37K
prompt the mlx_lm 2048 step is 18 full sorted chunks plus a tail, so 18 of 19
chunks re-read every active expert. A larger prefill chunk coalesces those
re-reads: the same sorted kernels run on the same math with the expert weights
read fewer times.

Why 4096 is the long-prompt default. The sorted-call counters confirm the
mechanism works (sorted calls per layer drop 18 to 9 to 5 to 3 as the chunk
doubles from 2048 to 16384). Under the composed head-dimension-256 prefill
attention the coalescing did not convert to served throughput at long context:
that attention path materializes a quadratic score tensor per chunk, and a
larger chunk grew the score tensor faster than it saved routed-MoE weight
bandwidth, so peak memory climbed (chunk 4096 exceeded the product budget at
37K) and the rate was flat at 4096 and worse beyond it. The flash prefill route
removes the score-tensor materialization, and the re-pricing reverses the
verdict: at 37K under the flash route chunk 4096 runs about 10 t/s faster than
chunk 2048 and cuts about 0.6 s off the time to first token, at a 26.43 GiB peak
that fits the 32 GB product budget. Chunk 8192 stays out (over budget and
slower). So 4096 is the served step for long prompts, and the env override
remains for experiments and for re-pricing.

Short prompts keep the mlx_lm chunk. The chunk is applied only when the prompt
is longer than the 4096 step, so a prompt of 4096 tokens or fewer is a single
chunk either way and its serving is byte-for-byte the stock 2048 path (this
keeps the 3969-token 4K anchor and any shorter prompt on the recorded token
rail). The small sub-bar win at 4K and 8K in earlier composed sweeps is not the
reason for the default; the long-context re-pricing is.

This module sets the served prefill chunk size on the model so `serve` passes it
to mlx_lm. It reuses the existing `_moespresso_prefill_step_size` seam that
`serve.generate_with_metadata` already consumes, plus the companion
`_moespresso_prefill_step_size_min_prompt_tokens` gate so the chunk applies only
to long prompts; there is no new prefill loop and no second knob.

Both the resident K-quant build and the SSD-streaming build install it. On the
streaming path the pool holds a bounded expert residency per layer, so a larger
prefill chunk routes more token-expert pairs per forward pass and can raise the
streaming path's expert cache misses. The long-prompt gate applies the chunk to
the same long prompts on both builds; the streaming path prices the chunk
against its own residency budget in the campaign measurements.

Larger chunks are math-affecting on the served path. The sorted kernels are
chunk-invariant per token-expert pair, but the product q8 KV cache
(`quantized_kv_start=0`) quantizes each chunk's keys and values after the chunk,
so the first chunk's queries attend over a dense cache and later chunks attend
over a quantized cache. The dense-to-quantized boundary is the first-chunk
boundary, so a different chunk size moves that boundary and can fork the greedy
stream deep in the generation. Chunk 4096 forks from chunk 2048 at generated
token 48 on the 37K prompt (deterministic across re-runs), and chunk 8192 forks
earlier. The campaign quality ladder therefore judges the chunk's numerical
effect: the recommended-profile gate stays 9/9 clean-pass on the changed
default. Short prompts that fit one chunk are exempt (the long-prompt gate keeps
them on the stock granularity), so every recorded short-prompt token rail is
byte-for-byte preserved.

Override and default: the shipped default resolves to a 4096 chunk applied to
long prompts. `MOESPRESSO_QWEN_PREFILL_CHUNK=<n>` overrides the value; a value
equal to the mlx_lm default, `0`, or a malformed value falls back to the mlx_lm
chunk across the whole prompt-length range (fail-closed to the stock
granularity). Peak memory grows with the chunk, so an override above 4096 must
be priced against the product memory budget.
"""

from __future__ import annotations

import os

# The mlx_lm default prompt chunk. A value equal to this is the kill-switch
# no-op: setting the attribute to this is behaviorally identical to not setting
# it, and the code below leaves it unset in that case so serving is
# byte-for-byte the stock path.
_MLX_DEFAULT_PREFILL_STEP = 2048

# The shipped default served prefill chunk. Larger than the mlx_lm chunk, so the
# resolver below sets the override; the long-prompt gate keeps short prompts on
# the mlx_lm chunk. `MOESPRESSO_QWEN_PREFILL_CHUNK=<n>` overrides the value.
_QWEN_PREFILL_CHUNK_DEFAULT = 4096

# Long-prompt gate: the chunk override applies only when the prompt is longer
# than the chunk, so a prompt that fits in one chunk is served byte-for-byte on
# the stock path. Serving reads this as
# `_moespresso_prefill_step_size_min_prompt_tokens`.
_LONG_PROMPT_MIN_TOKENS = _QWEN_PREFILL_CHUNK_DEFAULT + 1


def _configured_chunk() -> int | None:
    """The served prefill chunk, or None to leave the mlx_lm default in force.

    Reads `MOESPRESSO_QWEN_PREFILL_CHUNK`; falls back to the shipped default when
    the env is unset. Returns None (no override) when the resolved value is the
    mlx_lm default or when the env is present but not a positive integer, so a
    kill-switch value and a malformed override both fail closed to the stock
    chunk granularity.
    """
    raw = os.environ.get("MOESPRESSO_QWEN_PREFILL_CHUNK")
    if raw is None:
        chunk = _QWEN_PREFILL_CHUNK_DEFAULT
    else:
        try:
            chunk = int(raw)
        except (TypeError, ValueError):
            return None
        if chunk <= 0:
            return None
    if chunk == _MLX_DEFAULT_PREFILL_STEP:
        return None
    return chunk


def install_prefill_chunk(model) -> int | None:
    """Set the served prefill chunk on the model for the sorted routed-MoE path.

    Writes `model._moespresso_prefill_step_size` so `serve.generate_with_metadata`
    passes the chunk to mlx_lm's prompt loop, and
    `model._moespresso_prefill_step_size_min_prompt_tokens` so the chunk applies
    only to prompts longer than the chunk (short prompts keep the mlx_lm chunk).
    Returns the chunk that was set, or None when there is no override (a value
    equal to the mlx_lm default or a malformed value), in which case both model
    attributes are left untouched and serving uses the mlx_lm chunk. Idempotent:
    re-running with the same env resolves the same value.
    """
    chunk = _configured_chunk()
    if chunk is None:
        return None
    model._moespresso_prefill_step_size = int(chunk)
    model._moespresso_prefill_step_size_min_prompt_tokens = int(chunk) + 1
    return int(chunk)
