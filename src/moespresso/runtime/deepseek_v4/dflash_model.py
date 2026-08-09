"""DFlash draft model for DeepSeek-V4-Flash speculative decoding.

Implements the block-diffusion drafter shipped in the
RedHatAI/DeepSeek-V4-Flash-speculator.dflash checkpoint as a drafter for
the speculative decoding loop: dense llama-type decoder layers conditioned
on target hidden features through per-layer context K/V injection,
proposing a whole block of tokens in one non-autoregressive pass.

Conditioning. The drafter taps the raw hyper-connection output
(B, L, hc_mult, hidden) of the target layers named by the checkpoint's
`aux_hidden_state_layer_ids`. Those ids index the HF hidden_states
convention where entry 0 is the embedding output, so the tapped 0-based
decoder layers are the aux ids minus one. Each tapped layer output is
flattened stream-major to (B, L, hc_mult * hidden) and the tapped layers
concatenate in ascending order. A committed token's fused feature row is
`Ht = hidden_norm(fc(concat))`, computed once per row. Per draft layer,
the context key is `rope(k_norm(k_proj(Ht)))` at the token's absolute
position and the context value is `v_proj(Ht)`; no input_layernorm
applies to Ht (that norm belongs to the in-block stream). The per-layer
context K/V projections run as one stacked GEMM per ingest batch with one
batched rope call, and the cache keeps only the trailing `sliding_window`
entries.

Draft pass. A round embeds `[anchor, mask_token x speculative_tokens]`
through the target embedding at absolute positions
`anchor_pos .. anchor_pos + block - 1` and runs the layers once. Per
layer: input_layernorm, q/k/v projections with per-head q_norm/k_norm
before rope, grouped-query attention over `[context ; block]`, o_proj
residual, post_attention_layernorm, SwiGLU residual. The logits rows for
the mask positions go through the drafter's own pruned language-model
head; the anchor row proposes nothing.

Attention mask. The checkpoint trains every layer as causal
sliding-window attention (`sliding_window_non_causal` false), so in-block
visibility is causal inclusive: block row j sees block rows 0..j. The
context bound is the anchor-anchored window: the cache retains the
trailing `sliding_window` committed entries measured back from the
anchor, and every retained entry is visible to every block row. Training
measured the window from the anchor position, and a single retention
bound keeps the mask shape-stable (an all-visible context segment beside
a lower-triangular block). A per-query causal window would differ only in
whether the oldest few boundary entries of a full window stay visible to
the deepest block rows; the anchor-anchored form is the documented pick.

Cross-vocabulary proposals. Draft logits cover the pruned draft
vocabulary; proposals are the in-graph argmax mapped to target ids with
the d2t offset table (`target_id = draft_index + d2t[draft_index]`).
There is no draft distribution over the target vocabulary, so the
sampled acceptance rule cannot apply: the drafter advertises
`greedy_only` and both the loop and the serve seam keep temperature > 0
requests on the plain path. Proposals carry the draft-vocabulary logits
for diagnostics and `confidence=None`. The t2d membership table is a
training-time artifact carried in the sidecar for documentation and
unused at inference.

Rollback. Rejected draft rows never enter the context cache: the loop
ingests only the anchor and the accepted rows after verification, and
the draft pass keeps the in-block K/V in scratch, so the rewind to the
committed frontier is inherent and needs no cache surgery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import scaled_dot_product_attention

from .spec_decode import DraftProposal


def _require(config: dict, key: str, where: str) -> object:
    if key not in config:
        raise ValueError(f"missing {key!r} in {where}")
    return config[key]


@dataclass
class DFlashArgs:
    """Drafter dimensions parsed from the speculator checkpoint config.

    `block_size` follows the checkpoint convention: the attention block
    width, one anchor row plus `block_size - 1` mask rows. The drafter
    proposes `speculative_tokens = block_size - 1` tokens per round.
    `aux_layer_ids` keeps the checkpoint's HF hidden_states indices;
    `tap_layer_ids` derives the 0-based decoder layer ids from them.
    """

    aux_layer_ids: Tuple[int, ...]
    block_size: int
    mask_token_id: int
    draft_vocab_size: int
    target_vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    sliding_window: int
    hc_mult: int

    def __post_init__(self):
        self.aux_layer_ids = tuple(int(i) for i in self.aux_layer_ids)
        if not self.aux_layer_ids:
            raise ValueError("aux_layer_ids must name at least one target layer")
        if any(i < 1 for i in self.aux_layer_ids):
            raise ValueError(
                f"aux_layer_ids {self.aux_layer_ids} must all be >= 1: they "
                "index HF hidden_states, where entry 0 is the embedding output")
        if list(self.aux_layer_ids) != sorted(set(self.aux_layer_ids)):
            raise ValueError(
                f"aux_layer_ids {self.aux_layer_ids} must be strictly ascending")
        if self.block_size < 2:
            raise ValueError(
                f"block_size must be at least 2 (anchor plus one mask row), "
                f"got {self.block_size}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} is not a "
                f"multiple of num_key_value_heads {self.num_key_value_heads}")
        if not (0 < self.draft_vocab_size <= self.target_vocab_size):
            raise ValueError(
                f"draft_vocab_size {self.draft_vocab_size} must be positive "
                f"and no larger than target_vocab_size {self.target_vocab_size}")
        if not (0 <= self.mask_token_id < self.target_vocab_size):
            raise ValueError(
                f"mask_token_id {self.mask_token_id} is outside the target "
                f"vocabulary of {self.target_vocab_size}")
        if self.sliding_window < 1:
            raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")
        if self.num_hidden_layers < 1:
            raise ValueError(
                f"num_hidden_layers must be positive, got {self.num_hidden_layers}")

    @property
    def speculative_tokens(self) -> int:
        return self.block_size - 1

    @property
    def tap_layer_ids(self) -> Tuple[int, ...]:
        """0-based target decoder layers to tap: the aux ids minus one."""
        return tuple(i - 1 for i in self.aux_layer_ids)

    @property
    def fc_input_size(self) -> int:
        return len(self.aux_layer_ids) * self.hc_mult * self.hidden_size

    @classmethod
    def from_config(cls, config: dict) -> "DFlashArgs":
        """Parse the speculators dflash config.json, failing closed on any
        shape this implementation does not cover."""
        model_type = config.get("speculators_model_type")
        if model_type != "dflash":
            raise ValueError(
                f"speculators_model_type {model_type!r} is not 'dflash'")
        if config.get("sliding_window_non_causal"):
            raise ValueError(
                "sliding_window_non_causal is set; this checkpoint family is "
                "trained causal-in-block and non-causal blocks are unsupported")
        t = _require(config, "transformer_layer_config", "the speculator config")
        layer_types = t.get("layer_types")
        n_layers = int(_require(t, "num_hidden_layers", "transformer_layer_config"))
        if layer_types is not None:
            if len(layer_types) != n_layers or any(
                lt != "sliding_attention" for lt in layer_types
            ):
                raise ValueError(
                    f"layer_types {layer_types!r} must be 'sliding_attention' "
                    f"for all {n_layers} layers")
        rope = t.get("rope_parameters") or {}
        rope_type = rope.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(f"unsupported rope_type {rope_type!r}")
        rope_theta = rope.get("rope_theta", t.get("rope_theta"))
        if rope_theta is None:
            raise ValueError("the speculator config declares no rope theta")

        block_size = int(_require(config, "block_size", "the speculator config"))
        methods = (config.get("speculators_config") or {}).get("proposal_methods") or []
        for method in methods:
            declared = method.get("speculative_tokens")
            if declared is not None and int(declared) != block_size - 1:
                raise ValueError(
                    f"proposal method declares speculative_tokens {declared}; "
                    f"block_size {block_size} implies {block_size - 1}")

        return cls(
            aux_layer_ids=tuple(
                _require(config, "aux_hidden_state_layer_ids", "the speculator config")
            ),
            block_size=block_size,
            mask_token_id=int(_require(config, "mask_token_id", "the speculator config")),
            draft_vocab_size=int(
                _require(config, "draft_vocab_size", "the speculator config")),
            target_vocab_size=int(_require(t, "vocab_size", "transformer_layer_config")),
            hidden_size=int(_require(t, "hidden_size", "transformer_layer_config")),
            intermediate_size=int(
                _require(t, "intermediate_size", "transformer_layer_config")),
            num_hidden_layers=n_layers,
            num_attention_heads=int(
                _require(t, "num_attention_heads", "transformer_layer_config")),
            num_key_value_heads=int(
                _require(t, "num_key_value_heads", "transformer_layer_config")),
            head_dim=int(_require(t, "head_dim", "transformer_layer_config")),
            rope_theta=float(rope_theta),
            rms_norm_eps=float(_require(t, "rms_norm_eps", "transformer_layer_config")),
            sliding_window=int(_require(t, "sliding_window", "transformer_layer_config")),
            hc_mult=int(_require(t, "hc_mult", "transformer_layer_config")),
        )


@dataclass
class DFlashDraftState:
    """Per-generation draft state: the windowed per-layer context K/V.

    `ctx_k` and `ctx_v` hold (B, n_layers, n_kv_heads, len, head_dim)
    arrays over the trailing committed positions; keys are stored roped at
    their absolute positions, so window truncation never re-encodes rows.
    `next_pos` is the number of committed positions ingested so far.
    """

    ctx_k: Optional[mx.array] = None
    ctx_v: Optional[mx.array] = None
    next_pos: int = 0


def _dflash_block_visibility(ctx_len: int, block_len: int) -> mx.array:
    """Boolean visibility for one draft block over [context ; block] keys.

    Shape (1, 1, block_len, ctx_len + block_len) so it broadcasts onto the
    SDPA scores. Every retained context entry is visible to every block
    row (the anchor-anchored window bound is enforced by cache retention),
    and in-block visibility is causal inclusive.
    """
    block = mx.tri(block_len, block_len, dtype=mx.bool_)
    if ctx_len > 0:
        ctx = mx.ones((block_len, ctx_len), dtype=mx.bool_)
        block = mx.concatenate([ctx, block], axis=1)
    return block[None, None]


class DFlashAttention(nn.Module):
    """Grouped-query block attention over [context ; block].

    Per-head q_norm/k_norm apply before rope. Context K/V arrive already
    projected, normed, and roped from the ingest path; the in-block K/V
    computed here stay in scratch and are never cached.
    """

    def __init__(self, args: DFlashArgs):
        super().__init__()
        d = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        ctx_k: mx.array,
        ctx_v: mx.array,
        mask: mx.array,
        offset: int,
    ) -> mx.array:
        B, S, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim)
        q = self.rope(q.transpose(0, 2, 1, 3), offset=offset)
        k = self.rope(k.transpose(0, 2, 1, 3), offset=offset)
        v = v.transpose(0, 2, 1, 3)
        keys = mx.concatenate([ctx_k.astype(k.dtype), k], axis=2)
        values = mx.concatenate([ctx_v.astype(v.dtype), v], axis=2)
        out = scaled_dot_product_attention(
            q, keys, values, cache=None, scale=self.scale, mask=mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class DFlashMLP(nn.Module):
    """SwiGLU feed-forward."""

    def __init__(self, args: DFlashArgs):
        super().__init__()
        d, m = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(d, m, bias=False)
        self.up_proj = nn.Linear(d, m, bias=False)
        self.down_proj = nn.Linear(m, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    """One dense draft layer: pre-norm attention plus pre-norm SwiGLU."""

    def __init__(self, args: DFlashArgs):
        super().__init__()
        self.self_attn = DFlashAttention(args)
        self.mlp = DFlashMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        ctx_k: mx.array,
        ctx_v: mx.array,
        mask: mx.array,
        offset: int,
    ) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), ctx_k, ctx_v, mask, offset)
        return x + self.mlp(self.post_attention_layernorm(x))


class DFlashDraftModel(nn.Module):
    """Single-pass block drafter with an injected target embedding.

    Implements the `spec_decode.Drafter` protocol: `make_state` returns
    the per-generation windowed context cache, `ingest` appends context
    K/V rows for committed positions, and `draft` runs one block pass.
    The drafter owns its pruned language-model head and the d2t/t2d
    vocabulary tables; the embedding is the target model's frozen module.

    `greedy_only` is True: proposals are argmax-only over the draft
    vocabulary and carry no draft distribution over the target
    vocabulary, so sampled acceptance cannot apply.
    """

    greedy_only = True

    def __init__(self, args: DFlashArgs, embed=None):
        super().__init__()
        self.args = args
        d = args.hidden_size
        self.fc = nn.Linear(args.fc_input_size, d, bias=False)
        self.hidden_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(d, args.draft_vocab_size, bias=False)
        self.layers = [
            DFlashDecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.d2t = mx.zeros((args.draft_vocab_size,), dtype=mx.int64)
        self.t2d = mx.zeros((args.target_vocab_size,), dtype=mx.bool_)
        # The target-owned frozen embedding, stored outside the module
        # registry so it never appears in the draft parameter tree.
        self.set_shared(embed)

    @property
    def block_size(self) -> int:
        """Tokens proposed per round (the mask rows of the block)."""
        return self.args.speculative_tokens

    @property
    def tap_layer_ids(self) -> Sequence[int]:
        return self.args.tap_layer_ids

    def tap_transform(self, layer_id: int, out: mx.array) -> mx.array:
        """Flatten one raw (B, L, hc_mult, hidden) tap output stream-major."""
        return mx.flatten(out, start_axis=2)

    @property
    def embed(self):
        return self._shared_modules[0]

    def set_shared(self, embed) -> None:
        object.__setattr__(self, "_shared_modules", (embed,))

    def make_state(self) -> DFlashDraftState:
        return DFlashDraftState()

    def _context_kv(self):
        """Stacked context K/V projection over all layers, built once.

        Concatenates every layer's k_proj and v_proj rows into one weight
        so an ingest batch runs a single GEMM, plus the stacked per-layer
        k_norm weights. Built lazily on first use; the drafter weights are
        frozen after load, so the packed copies never go stale.
        """
        packed = getattr(self, "_ctx_kv_packed", None)
        if packed is not None:
            return packed
        mods = []
        for layer in self.layers:
            mods.extend([layer.self_attn.k_proj, layer.self_attn.v_proj])
        quantized = [isinstance(m, nn.QuantizedLinear) for m in mods]
        if all(quantized):
            params = {
                (m.group_size, m.bits, getattr(m, "mode", "affine")) for m in mods
            }
            if len(params) != 1:
                raise ValueError(
                    "context K/V projections carry mixed quantization "
                    f"parameters: {sorted(params)}")
            group_size, bits, mode = params.pop()
            w = mx.concatenate([m.weight for m in mods], axis=0)
            s = mx.concatenate([m.scales for m in mods], axis=0)
            b = (
                mx.concatenate([m.biases for m in mods], axis=0)
                if hasattr(mods[0], "biases") else None
            )

            def project(x: mx.array) -> mx.array:
                return mx.quantized_matmul(
                    x, w, scales=s, biases=b, transpose=True,
                    group_size=group_size, bits=bits, mode=mode,
                )
        elif not any(quantized):
            w = mx.concatenate([m.weight for m in mods], axis=0)

            def project(x: mx.array) -> mx.array:
                return x @ w.T
        else:
            raise ValueError(
                "context K/V projections mix quantized and float modules")
        k_norm_w = mx.stack(
            [layer.self_attn.k_norm.weight for layer in self.layers]
        )
        packed = (project, k_norm_w)
        object.__setattr__(self, "_ctx_kv_packed", packed)
        return packed

    def ingest(
        self,
        state: DFlashDraftState,
        rows: mx.array,
        positions: Sequence[int],
        token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Append context K/V rows for newly committed target rows.

        `rows` is the concatenated stream-major tap output
        (B, n, len(aux) * hc_mult * hidden) at `positions`. `token_ids`
        is accepted for protocol conformance and unused: the conditioning
        needs only the tapped hidden rows. Only committed rows ever reach
        this method, so the window cache never holds a rejected position.
        """
        positions = [int(p) for p in positions]
        n = len(positions)
        if n == 0:
            return
        args = self.args
        if rows.ndim != 3 or rows.shape[1] != n or rows.shape[2] != args.fc_input_size:
            raise ValueError(
                f"ingest rows shape {tuple(rows.shape)} does not carry {n} "
                f"rows of {args.fc_input_size} tap features")
        if positions != list(range(positions[0], positions[0] + n)):
            raise ValueError(f"ingest positions {positions} are not contiguous")
        if positions[0] != state.next_pos:
            raise ValueError(
                f"ingest at position {positions[0]} does not extend the "
                f"committed stream at {state.next_pos}")

        ht = self.hidden_norm(self.fc(rows))
        project, k_norm_w = self._context_kv()
        n_layers = args.num_hidden_layers
        n_kv = args.num_key_value_heads
        hd = args.head_dim
        B = ht.shape[0]
        kv = project(ht).reshape(B, n, n_layers, 2, n_kv, hd)
        k, v = kv[:, :, :, 0], kv[:, :, :, 1]
        # Per-layer k_norm over the stacked layout, mean in float32 as the
        # module norm computes it.
        kf = k.astype(mx.float32)
        k = (
            kf * mx.rsqrt(kf.square().mean(axis=-1, keepdims=True)
                          + args.rms_norm_eps)
            * k_norm_w.astype(mx.float32)[None, None, :, None, :]
        ).astype(k.dtype)
        # One batched rope call at the absolute positions, layers folded
        # into the head axis.
        k = k.transpose(0, 2, 3, 1, 4).reshape(B, n_layers * n_kv, n, hd)
        k = self.layers[0].self_attn.rope(k, offset=positions[0])
        k = k.reshape(B, n_layers, n_kv, n, hd)
        v = v.transpose(0, 2, 3, 1, 4)

        if state.ctx_k is None:
            state.ctx_k, state.ctx_v = k, v
        else:
            state.ctx_k = mx.concatenate([state.ctx_k, k], axis=3)
            state.ctx_v = mx.concatenate([state.ctx_v, v], axis=3)
        window = args.sliding_window
        if state.ctx_k.shape[3] > window:
            state.ctx_k = state.ctx_k[:, :, :, -window:]
            state.ctx_v = state.ctx_v[:, :, :, -window:]
        state.next_pos = positions[-1] + 1

    def draft(
        self,
        state: DFlashDraftState,
        anchor_token: int,
        anchor_pos: int,
        temperature: float = 0.0,
    ) -> DraftProposal:
        """Propose `block_size` tokens following the anchor in one pass.

        The committed stream in `state` must end exactly at
        `anchor_pos - 1`. The pass reads the context cache and writes
        nothing: in-block K/V live in scratch, so a rejected round leaves
        the state untouched. Greedy only; a positive temperature is a
        caller error because no draft distribution over the target
        vocabulary exists.
        """
        if temperature > 0:
            raise ValueError(
                "the DFlash drafter is greedy-only; temperature > 0 "
                "requests take the plain path")
        if state.ctx_k is None:
            raise ValueError("draft requires at least one ingested position")
        if anchor_pos != state.next_pos:
            raise ValueError(
                f"draft anchor at position {anchor_pos}; the committed "
                f"stream ends at position {state.next_pos - 1}")
        args = self.args
        block = args.block_size
        ids = mx.array(
            [[int(anchor_token)] + [args.mask_token_id] * args.speculative_tokens],
            dtype=mx.int64,
        )
        x = self.embed(ids)
        ctx_len = int(state.ctx_k.shape[3])
        mask = _dflash_block_visibility(ctx_len, block)
        for i, layer in enumerate(self.layers):
            x = layer(x, state.ctx_k[:, i], state.ctx_v[:, i], mask, anchor_pos)
        h = self.norm(x)[:, 1:]
        draft_logits = self.lm_head(h).astype(mx.float32)
        draft_idx = mx.argmax(draft_logits, axis=-1)
        tokens = draft_idx.astype(mx.int64) + mx.take(self.d2t, draft_idx)
        return DraftProposal(tokens=tokens, logits=draft_logits, confidence=None)


_LAYER_RE = re.compile(r"layers\.(\d+)\.")

_DROPPED_PREFIXES = ("embed_tokens.",)


def sanitize_dflash_weights(weights: dict, args: DFlashArgs) -> dict:
    """Map raw checkpoint names onto the draft module tree.

    The checkpoint stores final module-path names already, so the map is
    the identity apart from dropping the verifier embedding copy (the
    target model owns the embedding). Layer indices beyond the declared
    layer count are rejected; completeness against the parameter tree is
    the builder's key-set check.
    """
    out: dict = {}
    for key, value in weights.items():
        if key.startswith(_DROPPED_PREFIXES):
            continue
        m = _LAYER_RE.match(key)
        if m and int(m.group(1)) >= args.num_hidden_layers:
            raise ValueError(
                f"{key}: layer index outside the declared "
                f"{args.num_hidden_layers} draft layers")
        out[key] = value
    return out


__all__: List[str] = [
    "DFlashArgs",
    "DFlashDraftModel",
    "DFlashDraftState",
    "sanitize_dflash_weights",
]
