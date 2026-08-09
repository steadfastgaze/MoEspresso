"""MTP draft model for DeepSeek-V4-Flash speculative decoding.

Retained capability for future checkpoints, not a supported path on the
current weights. Two known draft-tree graph defects live in this module and
are documented rather than fixed:

1. the mHC recombine inherits the stock vendored ``matmul(comb, residual)``
   orientation, where the reference contracts over the first
   hyper-connection axis;
2. the compressed-KV non-RoPE rows are stored unrounded, where the
   reference rounds them through E4M3FN.

The same pair was fixed in the DSpark draft tree; the trunk serving graph
was never affected, and greedy token identity is not at risk either way
because speculative decoding accepts by verification against the target.
Any acceptance or decode number measured through this drafter is invalid
until the two are fixed. The module also targets preview-era ``mtp.0``
semantics (an ``e_proj``/``h_proj`` head) that the 0731 checkpoint does not
carry, and the MTP sidecar builder refuses that checkpoint, so no valid
sidecar for the current weights exists. This release does not expose MTP
through serving, replay, or battery selectors.

Implements the checkpoint's own multi-token-prediction module as a drafter
for the speculative decoding loop. The reference implementation is the
`inference/model.py` file inside the DeepSeek-V4-Flash checkpoint (class
MTPBlock): the fused input for a position is the sum of two projections,
`e_proj(enorm(embed(next_token)))` broadcast over the hyper-connection
copies plus `h_proj(hnorm(h))` on the raw hidden, where `h` is the
(B, L, hc_mult, hidden) hyper-connection state produced by the last trunk
layer before the trunk's hc-head reduce. The fused state runs through one
full vendored decoder block at layer id `num_hidden_layers` (score-routed
MoE, compress_ratio 0 attention with the base rope), then through the
module-owned hc-head reduce, the final norm, and the shared trunk
language-model head.

Stream layout: the fused row at position p combines the trunk hidden at
position p with the embedding of the committed token at position p + 1
and predicts the token at position p + 2. The drafter keeps one plain KV
cache over the fused committed stream; `ingest` appends rows by running
the fused inputs through the block (lazily, only the KV projection of
that forward is ever evaluated), and the newest hidden row stays pending
until the token after it is known. Attention over the committed stream is
causal with the model's sliding window, enforced with an explicit
visibility mask at every call so single-row draft steps see exactly the
window that sliding-window attention sees.

Chained drafting reuses the single module at depths 1..block_size: step
d + 1 fuses the embedding of the token sampled at step d with the block's
own output hidden from step d, one position further right. The reference
trunk chains layers on the hyper-connection state and treats the hc-head
reduce as a readout branch, so the chained hidden is the block output
before the reduce. Draft steps append temporary KV rows past the
committed frontier; every `draft` call rewinds the cache offset before
returning. Reads never pass the offset, so the temporary rows become
unreachable and the next append overwrites them in place; buffer growth
inside the chain preserves the committed prefix bitwise, so the offset
reset is an exact rewind with no row copies or extra evaluation
barriers.

The module has no confidence head; by default proposals carry
`confidence=None` and the loop schedules verify lengths from observed
acceptance. `MOESPRESSO_DS4_MTP_CONFIDENCE=1` enables a
weight-independent confidence proxy instead: the raw logit margin
between each drafted token and its best alternative (the top-1/top-2
margin under greedy drafting), passed through unscaled; consumers apply
the sigmoid and the online calibrator levels the resulting range
against observed acceptance. The proxy measured battery-neutral on the
evaluation prompt set (mean speedup ratio 1.262 against a 1.238-1.257
confidence-free spread, with the submit-length distribution and the
mean accepted length unchanged): sigmoid saturation flattens large
margins, so the per-round length discrimination the proxy is meant to
add does not engage, and the default stays confidence-free. Rescaling
the margin to unsaturate the sigmoid would be a constant tuned to the
current weights; revisit at the next weight drop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

from jang_tools.dsv4.mlx_model import (
    ModelArgs,
    DeepseekV4DecoderLayer,
    _dsv4_window_visibility,
)
from mlx_lm.models.cache import KVCache

from .dspark_model import _hc_sigmoid_reduce
from .spec_decode import DraftProposal, make_fp32_logits_fn, sample_from_logits

_CONFIDENCE_ENV = "MOESPRESSO_DS4_MTP_CONFIDENCE"


def _margin_confidence_enabled() -> bool:
    """Whether proposals carry the drafted-token margin confidence proxy.

    Default off: the proxy measured battery-neutral, so proposals stay
    confidence-free. `MOESPRESSO_DS4_MTP_CONFIDENCE=1` enables it.
    """
    return os.environ.get(_CONFIDENCE_ENV, "0") == "1"


def _drafted_token_margin(row: mx.array, token: mx.array) -> mx.array:
    """Raw confidence logit for one drafted token: the margin between the
    drafted token's logit and the best alternative logit.

    Greedy drafting picks the argmax, so the margin is the top-1/top-2
    logit gap, nonnegative; a sampled token below the argmax yields a
    negative margin. The value is structural with no tuned constants:
    consumers apply the sigmoid, and the online calibrator's
    observed-to-predicted ratio does the leveling.
    """
    chosen = mx.take_along_axis(row, token[:, None], axis=-1)[:, 0]
    best_other = mx.max(
        mx.put_along_axis(
            row, token[:, None], mx.array(-mx.inf, row.dtype), axis=-1
        ),
        axis=-1,
    )
    return chosen - best_other


@dataclass
class MTPArgs:
    """Draft-model arguments layered over the target architecture args.

    `model_args` carries the shared DeepSeek-V4 dimensions. Its
    `compress_ratios` list must cover the MTP block (the entry at layer id
    num_hidden_layers, zero); `pad_compress_ratios` extends a target-only
    list. `block_size` is the chained draft depth cap; the adaptive
    scheduler chooses the per-round depth below it.
    """

    model_args: ModelArgs
    block_size: int = 3

    def __post_init__(self):
        if self.block_size < 1:
            raise ValueError(f"block_size must be at least 1, got {self.block_size}")
        self.pad_compress_ratios()

    @property
    def tap_layer_id(self) -> int:
        """The tapped trunk layer: the last one, whose raw hyper-connection
        state feeds the fusion."""
        return self.model_args.num_hidden_layers - 1

    def pad_compress_ratios(self) -> None:
        args = self.model_args
        n = args.num_hidden_layers
        ratios = list(args.compress_ratios or [])
        if len(ratios) < n:
            raise ValueError(
                "MTP requires explicit compress_ratios for the target layers"
            )
        if len(ratios) < n + 1:
            ratios.append(0)
        if ratios[n] != 0:
            raise ValueError(
                f"MTP block resolved compress_ratio {ratios[n]}; "
                "the MTP attention requires 0"
            )
        args.compress_ratios = ratios


@dataclass
class MTPDraftState:
    """Per-generation draft state.

    `cache` holds the single-layer KV cache over the fused committed
    stream as a one-element list, mirroring the per-layer cache list
    convention. `pending_hidden` is the newest tapped trunk hidden
    row, at committed position `next_pos - 1`; its fused row cannot be
    appended until the following token is committed. `next_pos` is the
    number of committed positions ingested so far.
    """

    cache: List[KVCache] = field(default_factory=lambda: [KVCache()])
    pending_hidden: Optional[mx.array] = None
    next_pos: int = 0


class MTPDraftModel(nn.Module):
    """Chained single-module MTP drafter with injected target embed and
    lm head.

    Implements the `spec_decode.Drafter` protocol: `make_state` returns
    the per-generation committed-stream cache, `ingest` appends fused
    committed rows to it, and `draft` chains the module to `block_size`
    depth with an exact per-round cache rewind.
    """

    def __init__(self, args: MTPArgs, embed=None, lm_head=None):
        super().__init__()
        self.args = args
        margs = args.model_args
        self.block = DeepseekV4DecoderLayer(margs, margs.num_hidden_layers)
        if self.block.self_attn.compress_ratio != 0:
            raise ValueError("MTP attention must resolve compress_ratio 0")
        d = margs.hidden_size
        self.e_proj = nn.Linear(d, d, bias=False)
        self.h_proj = nn.Linear(d, d, bias=False)
        self.enorm = nn.RMSNorm(d, eps=margs.rms_norm_eps)
        self.hnorm = nn.RMSNorm(d, eps=margs.rms_norm_eps)
        self.norm = nn.RMSNorm(d, eps=margs.rms_norm_eps)
        self.hc_head_fn = mx.zeros((margs.hc_mult, margs.hc_mult * d))
        self.hc_head_base = mx.zeros((margs.hc_mult,))
        self.hc_head_scale = mx.zeros((1,))
        # Target-owned frozen modules, stored outside the module registry so
        # they never appear in the draft parameter tree.
        self.set_shared(embed, lm_head)

    @property
    def block_size(self) -> int:
        return self.args.block_size

    @property
    def tap_layer_ids(self) -> Sequence[int]:
        return (self.args.tap_layer_id,)

    def tap_transform(self, layer_id: int, out: mx.array) -> mx.array:
        """Record the raw hyper-connection state unchanged."""
        return out

    @property
    def embed(self):
        return self._shared_modules[0]

    @property
    def lm_head(self):
        return self._shared_modules[1]

    @property
    def logits_fn(self):
        return self._shared_modules[2]

    def set_shared(self, embed, lm_head) -> None:
        logits_fn = make_fp32_logits_fn(lm_head) if lm_head is not None else None
        object.__setattr__(self, "_shared_modules", (embed, lm_head, logits_fn))

    def make_state(self) -> MTPDraftState:
        """Per-generation draft state: the fused committed-stream cache."""
        return MTPDraftState()

    def fuse(self, token_ids: mx.array, hidden: mx.array) -> mx.array:
        """Fused module input for token/hidden pairs.

        token_ids: (B, m) committed or drafted tokens; hidden: the
        (B, m, hc_mult, hidden) hyper-connection state one position to
        the left of each token. Returns (B, m, hc_mult, hidden).
        """
        e = self.e_proj(self.enorm(self.embed(token_ids)))
        return e[:, :, None, :] + self.h_proj(self.hnorm(hidden))

    def _window_mask(self, batch: int, length: int, offset: int) -> mx.array:
        """Causal sliding-window visibility over the plain KV cache."""
        window = self.args.model_args.sliding_window
        return _dsv4_window_visibility(batch, length, offset, window, offset + length)

    def _append_rows(self, cache: KVCache, z: mx.array) -> mx.array:
        """One block forward over fused rows, appending their KV rows."""
        mask = self._window_mask(z.shape[0], z.shape[1], cache.offset)
        return self.block(z, mask=mask, cache=cache)

    def ingest(
        self,
        state: MTPDraftState,
        rows: mx.array,
        positions: Sequence[int],
        token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Append fused committed rows for newly forwarded target rows.

        `rows` is the raw tapped hyper-connection state (B, n, hc_mult,
        hidden) at `positions`; `token_ids` are the committed tokens at
        those positions. Each token fuses with the hidden one position to
        its left, so the newest hidden row stays pending until the next
        call supplies the token after it.
        """
        positions = [int(p) for p in positions]
        n = len(positions)
        if n == 0:
            return
        if token_ids is None:
            raise ValueError("the MTP drafter requires committed token ids")
        token_ids = [int(t) for t in token_ids]
        if len(token_ids) != n:
            raise ValueError(
                f"ingest got {len(token_ids)} token ids for {n} positions"
            )
        if rows.ndim != 4 or rows.shape[1] != n:
            raise ValueError(
                f"ingest rows shape {tuple(rows.shape)} does not carry "
                f"{n} raw hyper-connection rows"
            )
        if positions != list(range(positions[0], positions[0] + n)):
            raise ValueError(f"ingest positions {positions} are not contiguous")
        if positions[0] != state.next_pos:
            raise ValueError(
                f"ingest at position {positions[0]} does not extend the "
                f"committed stream at {state.next_pos}"
            )

        if state.pending_hidden is None:
            # First ingest: token 0 has no hidden to its left, so the
            # fused stream starts at position 0 with (h_0, t_1).
            hidden = rows[:, : n - 1]
            fuse_tokens = token_ids[1:]
        else:
            if n > 1:
                hidden = mx.concatenate(
                    [state.pending_hidden, rows[:, : n - 1]], axis=1
                )
            else:
                hidden = state.pending_hidden
            fuse_tokens = token_ids
        if fuse_tokens:
            tok = mx.array(fuse_tokens, dtype=mx.int64)[None]
            self._append_rows(state.cache[0], self.fuse(tok, hidden))
        state.pending_hidden = rows[:, n - 1 :]
        state.next_pos = positions[-1] + 1

    def draft(
        self,
        state: MTPDraftState,
        anchor_token: int,
        anchor_pos: int,
        temperature: float = 0.0,
    ) -> DraftProposal:
        """Propose block_size tokens following the anchor.

        The committed stream in `state` must end exactly at
        `anchor_pos - 1`. The chained steps append temporary KV rows;
        restoring the pre-round cache offset rewinds them exactly, since
        every read slices the buffers at the offset and the committed
        prefix survives buffer growth bitwise. The proposal stays lazy:
        in-place cache writes rebind array handles instead of mutating
        referenced nodes, so the proposal graph keeps reading the rows it
        attended to after later appends overwrite the buffer region.
        """
        if state.pending_hidden is None:
            raise ValueError("draft requires at least one ingested position")
        if anchor_pos != state.next_pos:
            raise ValueError(
                f"draft anchor at position {anchor_pos}; the committed "
                f"stream ends at position {state.next_pos - 1}"
            )
        margs = self.args.model_args
        depth = self.args.block_size
        cache = state.cache[0]
        rewind_offset = cache.offset
        want_conf = _margin_confidence_enabled()

        hidden = state.pending_hidden
        tok = mx.array([[int(anchor_token)]], dtype=mx.int64)
        tokens: List[mx.array] = []
        logits_rows: List[mx.array] = []
        conf_rows: List[mx.array] = []
        for _ in range(depth):
            z = self.fuse(tok, hidden)
            out = self._append_rows(cache, z)
            reduced = _hc_sigmoid_reduce(
                out,
                self.hc_head_fn,
                self.hc_head_scale,
                self.hc_head_base,
                margs.rms_norm_eps,
                margs.hc_eps,
            )
            row = self.logits_fn(self.norm(reduced)).astype(mx.float32)[:, 0]
            next_tok = sample_from_logits(row, temperature)
            if want_conf:
                conf_rows.append(_drafted_token_margin(row, next_tok))
            logits_rows.append(row)
            tokens.append(next_tok)
            tok = next_tok[:, None]
            hidden = out

        cache.offset = rewind_offset
        return DraftProposal(
            tokens=mx.stack(tokens, axis=1),
            logits=mx.stack(logits_rows, axis=1),
            confidence=mx.stack(conf_rows, axis=1) if want_conf else None,
        )


def sanitize_mtp_weights(weights: dict, args: MTPArgs) -> dict:
    """Map checkpoint mtp.0.* keys onto the draft module tree.

    Mirrors the vendored target sanitize: attn -> self_attn, attn_norm ->
    input_layernorm, ffn_norm -> post_attention_layernorm, ffn -> mlp,
    w1/w2/w3 -> gate/down/up_proj, routed experts stacked into switch_mlp.
    Module-level fusion tensors (e_proj/h_proj/enorm/hnorm/norm) and the
    hc-head parameters keep their names; the shared embed and lm head are
    dropped, they come from the target. Any MTP module index other than 0
    is rejected: the drafter chains the single shipped module.
    """
    import re

    w1w2w3 = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    module_names = ("enorm.weight", "hnorm.weight", "norm.weight")
    out: dict = {}
    pending: dict = {}
    for key, value in weights.items():
        m = re.match(r"mtp\.(\d+)\.(.+)", key)
        if not m:
            continue
        index, rest = int(m.group(1)), m.group(2)
        if index != 0:
            raise ValueError(
                f"unsupported MTP module index in {key}; the drafter "
                "chains module 0"
            )
        if rest.startswith("embed.") or rest.startswith("head."):
            continue
        if (
            rest in module_names
            or rest.startswith(("e_proj.", "h_proj."))
            or rest.startswith("hc_head_")
        ):
            out[rest] = value
        elif rest == "attn_norm.weight":
            out["block.input_layernorm.weight"] = value
        elif rest == "ffn_norm.weight":
            out["block.post_attention_layernorm.weight"] = value
        elif rest.startswith("attn."):
            out[f"block.self_attn.{rest[len('attn.'):]}"] = value
        elif rest.startswith("ffn."):
            inner = rest[len("ffn."):]
            m2 = re.match(
                r"shared_experts\.(w[123])\.(weight|scales|biases|scale)$", inner
            )
            m3 = re.match(
                r"experts\.(\d+)\.(w[123])\.(weight|scales|biases|scale)$", inner
            )
            if m2:
                out[
                    f"block.mlp.shared_experts.{w1w2w3[m2.group(1)]}.{m2.group(2)}"
                ] = value
            elif m3:
                pending[(w1w2w3[m3.group(2)], m3.group(3), int(m3.group(1)))] = value
            else:
                out[f"block.mlp.{inner}"] = value
        else:
            out[f"block.{rest}"] = value

    n_experts = args.model_args.n_routed_experts
    for proj in ("gate_proj", "down_proj", "up_proj"):
        kinds = sorted({k for (p, k, _) in pending if p == proj})
        for kind in kinds:
            rows = [pending.pop((proj, kind, e)) for e in range(n_experts)]
            out[f"block.mlp.switch_mlp.{proj}.{kind}"] = mx.stack(rows)
    if pending:
        example = next(iter(pending))
        raise ValueError(f"unstacked draft expert tensors remain, e.g. {example}")
    return out
