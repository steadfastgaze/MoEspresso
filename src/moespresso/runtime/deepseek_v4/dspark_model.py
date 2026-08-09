"""DSpark draft model for DeepSeek-V4-Flash speculative decoding.

Implements the semi-autoregressive drafter from the DSpark paper
(arXiv:2607.05147) as shipped in the DeepSeek-V4-Flash-DSpark checkpoint:
three MoE decoder blocks over a sliding-window view of the target token
stream, a low-rank Markov head that biases each block position on the
previously sampled token, and a confidence head that scores per-position
acceptance. The reference implementation is the `inference/model.py` file
inside the checkpoint (classes DSparkAttention, DSparkBlock,
DSparkMarkovHead, DSparkConfidenceHead, Transformer.forward_spec).

The draft blocks reuse the vendored DeepSeek-V4 MLX building blocks
(decoder layer with manifold-constrained hyper-connections, MoE with
sqrtsoftplus routing, MLA weight layout, partial rope) so draft numerics
match the target graph conventions. The embedding and language-model head
are the target model's own frozen modules, injected at construction; the
draft checkpoint does not carry them.

Positions and roles per drafting round, with P the position of the last
token the target model has forwarded:
- The window caches hold main-stream KV rows for positions <= P.
- The anchor token (sampled from position P logits) sits at position P+1
  and is the first block input; the remaining block inputs are the noise
  token. Block slot k is roped at position P+1+k.
- Block slot k logits (base logits plus Markov bias on the previously
  sampled block token) propose the token at position P+2+k.
- The confidence head reads the post-hc-head, pre-norm hidden state and
  returns a raw logit; consumers apply the sigmoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

from jang_tools.dsv4.mlx_model import (
    ModelArgs,
    DeepseekV4Attention,
    DeepseekV4DecoderLayer,
    _apply_partial_rope,
    _get_q_norm_ones,
)
from mlx_lm.models.base import scaled_dot_product_attention

from .spec_decode import (
    DrafterStateCapsule,
    DraftProposal,
    make_fp32_logits_fn,
    sample_from_logits,
)


_STATE_CAPSULE_KIND = "deepseek_v4_dspark_window_state"
_STATE_CAPSULE_SCHEMA_MAJOR = 1
_STATE_CAPSULE_SCHEMA_MINOR = 0
_STATE_CAPSULE_METADATA_KEYS = frozenset(
    {"batch", "dtype", "head_dim", "stages", "window"}
)


@dataclass
class DSparkArgs:
    """Draft-model arguments layered over the target architecture args.

    `model_args` carries the shared DeepSeek-V4 dimensions. Its
    `compress_ratios` list must cover the draft stages (entries for
    layer ids num_hidden_layers .. num_hidden_layers + n_mtp_layers - 1,
    all zero); `pad_compress_ratios` extends a target-only list.
    """

    model_args: ModelArgs
    n_mtp_layers: int = 3
    block_size: int = 5
    noise_token_id: int = 128799
    target_layer_ids: Sequence[int] = (40, 41, 42)
    markov_rank: int = 256

    def __post_init__(self):
        self.pad_compress_ratios()

    def pad_compress_ratios(self) -> None:
        args = self.model_args
        n = args.num_hidden_layers
        ratios = list(args.compress_ratios or [])
        if len(ratios) < n:
            raise ValueError(
                "DSpark requires explicit compress_ratios for the target layers"
            )
        while len(ratios) < n + self.n_mtp_layers:
            ratios.append(0)
        for stage in range(self.n_mtp_layers):
            if ratios[n + stage] != 0:
                raise ValueError(
                    f"draft stage {stage} resolved compress_ratio "
                    f"{ratios[n + stage]}; DSpark attention requires 0"
                )
        args.compress_ratios = ratios


class DSparkWindowCache:
    """Ring buffer of main-stream KV rows for one draft block.

    Rows are stored roped at their absolute positions, at slot
    `position % window`, matching the reference layout. Attention over the
    ring is permutation-invariant given the validity mask, so slots are
    never rotated back into positional order.
    """

    def __init__(self, window: int, head_dim: int, batch: int = 1):
        self.window = window
        self.kv = mx.zeros((batch, 1, window, head_dim))
        self.last_pos = -1

    def write(self, rows: mx.array, positions: Sequence[int]) -> None:
        # rows: (B, n, head_dim) roped main-stream KV rows at `positions`,
        # which must be contiguous and extend the stream.
        if len(positions) == 0:
            return
        fresh_full = self.last_pos == -1 and len(positions) >= self.window
        if positions[0] != self.last_pos + 1 and not fresh_full:
            raise ValueError(
                f"window write at position {positions[0]} does not extend "
                f"last position {self.last_pos}"
            )
        n = rows.shape[1]
        if n > self.window:
            rows = rows[:, -self.window :]
            positions = positions[-self.window :]
            n = self.window
        slots = mx.array([p % self.window for p in positions], dtype=mx.int64)
        self.kv[:, :, slots, :] = rows[:, None, :, :].astype(self.kv.dtype)
        self.last_pos = positions[-1]

    def valid_mask(self) -> Optional[mx.array]:
        # True for slots holding a live row; None when the whole ring is live.
        if self.last_pos + 1 >= self.window:
            return None
        return mx.arange(self.window) <= self.last_pos


class DSparkDraftAttention(DeepseekV4Attention):
    """Draft-block attention over [main-stream window ; draft block].

    Inherits the MLA weight layout, per-layer rope (base theta, no YaRN at
    compress_ratio 0), grouped output projection, and attention sink from
    the target attention. Draft queries attend bidirectionally to every
    block position and to all live window rows.
    """

    def ingest_main_rows(
        self, main_x: mx.array, positions: Sequence[int], window: DSparkWindowCache
    ) -> None:
        # main_x: (B, n, hidden) projected target hidden rows at `positions`.
        rows = self.kv_norm(self.wkv(main_x))
        pos = mx.array(list(positions), dtype=mx.int32)
        rows = _apply_partial_rope(rows, self.rope, positions=pos)
        rows = _fp8_kv_roundtrip(rows)
        window.write(rows, list(positions))

    def __call__(self, x, positions=None, window=None, mask=None, cache=None):
        if positions is None or window is None:
            raise ValueError("draft attention requires positions and window")
        B, L, _ = x.shape
        pos = mx.array(list(positions), dtype=mx.int32)

        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(
            q,
            weight=_get_q_norm_ones(self.head_dim, q.dtype),
            eps=self.args.rms_norm_eps,
        )
        q = q.transpose(0, 2, 1, 3)
        q = _apply_partial_rope(q, self.rope, positions=pos)

        kv = self.kv_norm(self.wkv(x)).reshape(B, L, 1, self.head_dim)
        kv = kv.transpose(0, 2, 1, 3)
        kv = _apply_partial_rope(kv, self.rope, positions=pos)
        kv = _fp8_kv_roundtrip(kv)

        win_kv = window.kv.astype(kv.dtype)
        full_kv = mx.concatenate([win_kv, kv], axis=2)

        valid = window.valid_mask()
        if valid is None:
            attn_mask = None
        else:
            keys_ok = mx.concatenate(
                [valid, mx.ones((L,), dtype=mx.bool_)], axis=0
            )
            attn_mask = mx.broadcast_to(
                keys_ok[None, None, None, :], (B, 1, L, window.window + L)
            )

        out = scaled_dot_product_attention(
            q,
            full_kv,
            full_kv,
            cache=None,
            scale=self.softmax_scale,
            mask=attn_mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = _apply_partial_rope(out, self.rope, positions=pos, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)


class DSparkMarkovHead(nn.Module):
    """Low-rank first-order transition bias over the vocabulary."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array):
        embed = self.markov_w1(token_ids)
        bias = self.markov_w2(embed)
        return bias, embed


class DSparkConfidenceHead(nn.Module):
    """Per-position acceptance predictor. Returns a raw logit; the caller
    applies the sigmoid. Computation is float32 as in the reference."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def __call__(self, hidden: mx.array, markov_embed: mx.array) -> mx.array:
        feat = mx.concatenate(
            [hidden.astype(mx.float32), markov_embed.astype(mx.float32)], axis=-1
        )
        w = self.proj.weight.astype(mx.float32)
        return (feat @ w.T).squeeze(-1)


def _fp8_kv_roundtrip(rows):
    """Round a draft KV row's non-RoPE prefix through E4M3FN.

    The reference draft attention quantizes the non-RoPE dims of both the
    main-stream rows and the block's own rows before attention consumes
    them (`inference/model.py` DSparkAttention.forward, the two
    `act_quant(..., 64, scale_fmt, scale_dtype, True)` calls), the same
    contract the target attention applies to its window rows. The shared
    helper carries the served geometry, so a row width other than the
    model's head dim passes through untouched.
    """
    from moespresso.runtime.deepseek_v4.model import _deepseek_v4_fp8_kv_roundtrip

    return _deepseek_v4_fp8_kv_roundtrip(rows)


def _hc_post_reference(x, residual, post, comb):
    """Recombine a block output with its hyper-connection residual streams.

    The combination matrix contracts over its first hyper-connection axis
    (`inference/model.py` Block.hc_post:
    ``sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)``), which is
    the transpose of a plain ``comb @ residual``. The vendored decoder
    layer recombines with the plain orientation, so the draft stages state
    the contract here; the served target graph states the same contract in
    ``_patch_deepseek_v4_hc_post_float32``. The result is cast back to the
    activation dtype as the reference does (``y.type_as(x)``).
    """
    y = post[..., None] * x[..., None, :].astype(mx.float32) + mx.matmul(
        mx.swapaxes(comb, -1, -2).astype(mx.float32),
        residual.astype(mx.float32),
    )
    return y.astype(x.dtype)


def _hc_sigmoid_reduce(x, fn, scale, base, eps_rms, eps_hc):
    # x: (B, L, hc, D) -> (B, L, D), sigmoid-gated sum over the hc copies.
    # Stage-owned analogue of the model-level hc head reduce.
    shape = x.shape
    x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
    rsqrt = mx.rsqrt(mx.mean(x_flat.square(), axis=-1, keepdims=True) + eps_rms)
    mixes = (x_flat @ fn.astype(mx.float32).T) * rsqrt
    pre = mx.sigmoid(mixes * scale.astype(mx.float32) + base.astype(mx.float32))
    pre = pre + eps_hc
    y = mx.sum(pre[..., None] * mx.reshape(x_flat, shape), axis=2)
    return y.astype(x.dtype)


class DSparkBlock(DeepseekV4DecoderLayer):
    """One draft decoder stage.

    Layer id is num_hidden_layers + stage so the MoE gate resolves to the
    score-routed mode (hash routing covers only the first target layers)
    and the attention resolves compress_ratio 0. Stage 0 additionally owns
    the projection of concatenated target hidden states; the last stage
    owns the hc head reduce, final norm, Markov head, and confidence head.
    """

    def __init__(self, dspark_args: DSparkArgs, stage: int):
        args = dspark_args.model_args
        layer_id = args.num_hidden_layers + stage
        super().__init__(args, layer_id)
        self.stage = stage
        self.self_attn = DSparkDraftAttention(args, layer_id=layer_id)
        if self.self_attn.compress_ratio != 0:
            raise ValueError("draft attention must resolve compress_ratio 0")

        n_targets = len(dspark_args.target_layer_ids)
        if stage == 0:
            self.main_proj = nn.Linear(
                args.hidden_size * n_targets, args.hidden_size, bias=False
            )
            self.main_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if stage == dspark_args.n_mtp_layers - 1:
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(
                args.vocab_size, dspark_args.markov_rank
            )
            self.confidence_head = DSparkConfidenceHead(
                args.hidden_size + dspark_args.markov_rank
            )
            self.hc_head_fn = mx.zeros(
                (args.hc_mult, args.hc_mult * args.hidden_size)
            )
            self.hc_head_base = mx.zeros((args.hc_mult,))
            self.hc_head_scale = mx.zeros((1,))

    def _hc_post(self, x, residual, post, comb):
        # The vendored decoder layer contracts the combination matrix over
        # its last hyper-connection axis; the reference draft stage
        # inherits Block.hc_post, which contracts over the first.
        return _hc_post_reference(x, residual, post, comb)

    def __call__(self, x, positions=None, window=None, mask=None, cache=None,
                 input_ids=None):
        residual = x
        x, post, comb = self._hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions=positions, window=window)
        x = self._hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, input_ids=None)
        x = self._hc_post(x, residual, post, comb)
        return x


class DSparkDraftModel(nn.Module):
    """Three-stage DSpark drafter with injected target embed and lm head.

    Implements the `spec_decode.Drafter` protocol: `make_state` returns
    the per-generation window caches, `ingest` feeds tapped target rows
    into them, and `draft` proposes a Markov-biased block with raw
    confidence logits.
    """

    state_capsule_kind = _STATE_CAPSULE_KIND
    state_capsule_schema_major = _STATE_CAPSULE_SCHEMA_MAJOR
    state_capsule_schema_minor = _STATE_CAPSULE_SCHEMA_MINOR

    def __init__(self, args: DSparkArgs, embed=None, lm_head=None):
        super().__init__()
        self.args = args
        self.blocks = [DSparkBlock(args, s) for s in range(args.n_mtp_layers)]
        # Target-owned frozen modules, stored outside the module registry so
        # they never appear in the draft parameter tree.
        self.set_shared(embed, lm_head)

    @property
    def block_size(self) -> int:
        return self.args.block_size

    @property
    def tap_layer_ids(self) -> Sequence[int]:
        return tuple(self.args.target_layer_ids)

    def tap_transform(self, layer_id: int, out: mx.array) -> mx.array:
        """Record the mean over the hyper-connection copies."""
        return out.mean(axis=2)

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

    def make_window_caches(self, batch: int = 1) -> List[DSparkWindowCache]:
        args = self.args.model_args
        return [
            DSparkWindowCache(args.sliding_window, args.head_dim, batch)
            for _ in range(self.args.n_mtp_layers)
        ]

    def make_state(self) -> List[DSparkWindowCache]:
        """Per-generation draft state: one window cache per block."""
        return self.make_window_caches()

    def _validate_state(
        self, state: object
    ) -> tuple[List[DSparkWindowCache], int, int, str]:
        if not isinstance(state, (list, tuple)):
            raise TypeError("DSpark state must be a sequence of window caches")
        if len(state) != self.args.n_mtp_layers:
            raise ValueError(
                "DSpark state stage count does not match the draft architecture"
            )
        if not state:
            raise ValueError("DSpark state must contain at least one stage")

        expected_window = int(self.args.model_args.sliding_window)
        expected_head_dim = int(self.args.model_args.head_dim)
        expected_dtype = str(mx.float32)
        batch = None
        last_pos = None
        windows = []
        for stage, window in enumerate(state):
            if not isinstance(window, DSparkWindowCache):
                raise TypeError(f"DSpark state stage {stage} is not a window cache")
            if int(window.window) != expected_window:
                raise ValueError(f"DSpark state stage {stage} has the wrong window size")
            if not isinstance(window.kv, mx.array):
                raise TypeError(f"DSpark state stage {stage} payload is not an MLX array")
            shape = tuple(int(dim) for dim in window.kv.shape)
            if len(shape) != 4:
                raise ValueError(f"DSpark state stage {stage} has invalid payload rank")
            if batch is None:
                batch = shape[0]
                if batch < 1:
                    raise ValueError("DSpark state batch size must be positive")
            expected_shape = (batch, 1, expected_window, expected_head_dim)
            if shape != expected_shape:
                raise ValueError(
                    f"DSpark state stage {stage} has payload shape {shape}, "
                    f"expected {expected_shape}"
                )
            dtype = str(window.kv.dtype)
            if dtype != expected_dtype:
                raise ValueError(
                    f"DSpark state stage {stage} has payload dtype {dtype}, "
                    f"expected {expected_dtype}"
                )
            if isinstance(window.last_pos, bool) or not isinstance(window.last_pos, int):
                raise TypeError(
                    f"DSpark state stage {stage} frontier must be a Python int"
                )
            if window.last_pos < -1:
                raise ValueError(f"DSpark state stage {stage} has an invalid frontier")
            if last_pos is None:
                last_pos = window.last_pos
            elif window.last_pos != last_pos:
                raise ValueError("DSpark state stages disagree on the frontier")
            windows.append(window)

        if batch is None or last_pos is None:
            raise ValueError("DSpark state geometry could not be resolved")
        return windows, last_pos + 1, batch, expected_dtype

    def state_frontier(self, state: object) -> int:
        """Return the exclusive frontier shared by every DSpark stage."""
        _, frontier, _, _ = self._validate_state(state)
        return frontier

    def state_nbytes(self, state: object) -> int:
        """Return the bytes occupied by the live DSpark window tensors."""
        windows, _, _, _ = self._validate_state(state)
        return sum(int(window.kv.nbytes) for window in windows)

    def export_state(self, state: object) -> DrafterStateCapsule:
        """Copy the DSpark ring buffers into a validated state capsule."""
        windows, frontier, batch, dtype = self._validate_state(state)
        tensors = tuple(mx.array(window.kv) for window in windows)
        mx.eval(*tensors)
        metadata = (
            ("batch", batch),
            ("dtype", dtype),
            ("head_dim", int(self.args.model_args.head_dim)),
            ("stages", int(self.args.n_mtp_layers)),
            ("window", int(self.args.model_args.sliding_window)),
        )
        return DrafterStateCapsule(
            kind=_STATE_CAPSULE_KIND,
            schema_major=_STATE_CAPSULE_SCHEMA_MAJOR,
            schema_minor=_STATE_CAPSULE_SCHEMA_MINOR,
            frontier=frontier,
            metadata=metadata,
            tensors=tensors,
        )

    def import_state(
        self, capsule: DrafterStateCapsule
    ) -> List[DSparkWindowCache]:
        """Validate a DSpark capsule and copy it into fresh ring buffers."""
        if not isinstance(capsule, DrafterStateCapsule):
            raise TypeError("DSpark state restore requires a drafter state capsule")
        if capsule.kind != _STATE_CAPSULE_KIND:
            raise ValueError(f"unsupported DSpark state capsule kind: {capsule.kind!r}")
        if capsule.schema_major != _STATE_CAPSULE_SCHEMA_MAJOR:
            raise ValueError(
                "unsupported DSpark state capsule schema major: "
                f"{capsule.schema_major}"
            )
        if capsule.schema_minor != _STATE_CAPSULE_SCHEMA_MINOR:
            raise ValueError(
                "unsupported DSpark state capsule schema minor: "
                f"{capsule.schema_minor}"
            )

        metadata = capsule.metadata_dict()
        if set(metadata) != _STATE_CAPSULE_METADATA_KEYS:
            raise ValueError("DSpark state capsule metadata fields do not match schema")

        def require_int(key: str) -> int:
            value = metadata[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"DSpark state capsule {key} must be an int")
            return value

        batch = require_int("batch")
        stages = require_int("stages")
        window = require_int("window")
        head_dim = require_int("head_dim")
        dtype = metadata["dtype"]
        if batch < 1:
            raise ValueError("DSpark state capsule batch size must be positive")
        if stages != self.args.n_mtp_layers:
            raise ValueError("DSpark state capsule stage count does not match drafter")
        if window != self.args.model_args.sliding_window:
            raise ValueError("DSpark state capsule window size does not match drafter")
        if head_dim != self.args.model_args.head_dim:
            raise ValueError("DSpark state capsule head dimension does not match drafter")
        if not isinstance(dtype, str) or dtype != str(mx.float32):
            raise ValueError("DSpark state capsule dtype does not match drafter")
        if len(capsule.tensors) != stages:
            raise ValueError("DSpark state capsule tensor count does not match stages")

        state = self.make_window_caches(batch=batch)
        expected_shape = (batch, 1, window, head_dim)
        for stage, (source, destination) in enumerate(zip(capsule.tensors, state)):
            shape = tuple(int(dim) for dim in source.shape)
            if shape != expected_shape:
                raise ValueError(
                    f"DSpark state capsule tensor {stage} has shape {shape}, "
                    f"expected {expected_shape}"
                )
            if str(source.dtype) != dtype or source.dtype != destination.kv.dtype:
                raise ValueError(
                    f"DSpark state capsule tensor {stage} has an incompatible dtype"
                )
            destination.kv = mx.array(source)
            destination.last_pos = capsule.frontier - 1

        mx.eval(*(window_cache.kv for window_cache in state))
        self._validate_state(state)
        return state

    def project_main(self, main_hidden: mx.array) -> mx.array:
        """Project concatenated target hidden rows into the draft stream.

        main_hidden: (B, n, hidden * len(target_layer_ids)), rows built from
        the hc-mean of the target layer outputs at the tap layers.
        """
        stage0 = self.blocks[0]
        return stage0.main_norm(stage0.main_proj(main_hidden))

    def ingest(
        self,
        state: List[DSparkWindowCache],
        rows: mx.array,
        positions: Sequence[int],
        token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Feed newly forwarded target rows into every block's window.

        `token_ids` is accepted for protocol conformance and unused: the
        window conditioning needs only the tapped hidden rows.

        A prefill-shaped ingest (more rows than a verify block carries)
        dispatches its device graph at once: the spec loop's prefill
        evals only the target caches, so a lazy prompt ingest otherwise
        executes inside the first draft eval, off the prefill clock and
        on the decode clock. At a 3,844-token prompt that deferral
        measured 430-460 ms on the first round of every request (524-551
        ms against 92-112 ms steady rounds) in a process with warm
        kernels. The dispatch is scheduling only: the written values are
        exactly the lazy graph's. Round-sized ingests stay lazy so the
        round tail keeps overlapping them with the next draft.
        """
        main_x = self.project_main(rows)
        for block, window in zip(self.blocks, state):
            block.self_attn.ingest_main_rows(main_x, positions, window)
        if len(positions) > self.args.block_size + 1:
            mx.async_eval(*(window.kv for window in state))

    def draft(
        self,
        state: List[DSparkWindowCache],
        anchor_token: int,
        anchor_pos: int,
        temperature: float = 0.0,
        batch: int = 1,
    ) -> DraftProposal:
        """Propose block_size tokens following the anchor.

        The `state` window caches must already cover positions
        <= anchor_pos - 1.
        """
        windows = state
        args = self.args
        block_size = args.block_size
        draft_ids = mx.full((batch, block_size), args.noise_token_id, dtype=mx.int64)
        draft_ids[:, 0] = anchor_token
        x = self.embed(draft_ids)
        hc_mult = args.model_args.hc_mult
        x = mx.tile(x[..., None, :], (1, 1, hc_mult, 1))

        positions = list(range(anchor_pos, anchor_pos + block_size))
        for block, window in zip(self.blocks, windows):
            x = block(x, positions=positions, window=window)

        last = self.blocks[-1]
        x = _hc_sigmoid_reduce(
            x,
            last.hc_head_fn,
            last.hc_head_scale,
            last.hc_head_base,
            args.model_args.rms_norm_eps,
            args.model_args.hc_eps,
        )
        base_logits = self.logits_fn(last.norm(x))

        tokens = []
        logits_rows = []
        embeds = []
        prev = mx.full((batch,), anchor_token, dtype=mx.int64)
        for k in range(block_size):
            bias, embed = last.markov_head(prev)
            row = base_logits[:, k] + bias.astype(mx.float32)
            logits_rows.append(row)
            embeds.append(embed)
            prev = sample_from_logits(row, temperature)
            tokens.append(prev)

        logits = mx.stack(logits_rows, axis=1)
        markov_embed = mx.stack(embeds, axis=1)
        confidence = last.confidence_head(x, markov_embed)
        return DraftProposal(
            tokens=mx.stack(tokens, axis=1),
            logits=logits,
            confidence=confidence,
        )


def sanitize_dspark_weights(weights: dict, args: DSparkArgs) -> dict:
    """Map checkpoint mtp.* keys onto the draft module tree.

    Mirrors the vendored target sanitize: attn -> self_attn, attn_norm ->
    input_layernorm, ffn_norm -> post_attention_layernorm, ffn -> mlp,
    w1/w2/w3 -> gate/down/up_proj, routed experts stacked into switch_mlp.
    The shared embed and lm head are dropped; they come from the target.
    """
    import re

    w1w2w3 = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    out: dict = {}
    pending: dict = {}
    for key, value in weights.items():
        m = re.match(r"mtp\.(\d+)\.(.+)", key)
        if not m:
            continue
        stage, rest = int(m.group(1)), m.group(2)
        pfx = f"blocks.{stage}"
        if rest.startswith("embed.") or rest.startswith("head."):
            continue
        if rest == "attn_norm.weight":
            out[f"{pfx}.input_layernorm.weight"] = value
        elif rest == "ffn_norm.weight":
            out[f"{pfx}.post_attention_layernorm.weight"] = value
        elif rest.startswith("attn."):
            out[f"{pfx}.self_attn.{rest[len('attn.'):]}"] = value
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
                    f"{pfx}.mlp.shared_experts.{w1w2w3[m2.group(1)]}.{m2.group(2)}"
                ] = value
            elif m3:
                pending[
                    (stage, w1w2w3[m3.group(2)], m3.group(3), int(m3.group(1)))
                ] = value
            else:
                out[f"{pfx}.mlp.{inner}"] = value
        else:
            out[f"{pfx}.{rest}"] = value

    n_experts = args.model_args.n_routed_experts
    stages = sorted({s for (s, _, _, _) in pending})
    for stage in stages:
        for proj in ("gate_proj", "down_proj", "up_proj"):
            kinds = sorted({k for (s, p, k, _) in pending if s == stage and p == proj})
            for kind in kinds:
                rows = [pending.pop((stage, proj, kind, e)) for e in range(n_experts)]
                out[f"blocks.{stage}.mlp.switch_mlp.{proj}.{kind}"] = mx.stack(rows)
    if pending:
        example = next(iter(pending))
        raise ValueError(f"unstacked draft expert tensors remain, e.g. {example}")
    return out
