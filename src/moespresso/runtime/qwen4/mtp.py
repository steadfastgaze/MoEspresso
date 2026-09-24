"""Qwen MTP head composed from the existing residual, QSA and MoE modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from moespresso.package.qwen4.mtp_format import MTP_GRAPH_CONTRACT
from moespresso.runtime.qwen4.model import Qwen4DecoderLayer
from moespresso.runtime.qwen4.primitives import Qwen4RMSNorm, gated_residual_write
from moespresso.runtime.qwen4.qsa import Qwen4BF16QSAStateBackend, Qwen4QSAAdapter, Qwen4QSAState


@dataclass(frozen=True)
class Qwen4MTPState:
    """Functional attention state for shifted hidden/token pairs.

    A row is authoritative only when its input hidden state came from target
    verification. Accepting a chained draft token does not certify its input
    hidden state. The request owner retains or rebuilds rows on that basis.
    """

    owner: object
    attention: Qwen4QSAState

    @property
    def frontier(self) -> int:
        return self.attention.offset


@dataclass(frozen=True)
class Qwen4MTPOutput:
    logits: mx.array
    widened: mx.array
    state: Qwen4MTPState


class Qwen4MTPHead(nn.Module):
    """One text-only MTP block with full-width pre-fusion normalization.

    The graph is a reconstruction from the checkpoint tensors with an explicit
    contract identifier. Input hidden rows are the target's widened stream before
    its final residual mixer. Token embeddings belong to the following positions.
    The target embedding and language head are shared, not copied into the sidecar.
    """

    def __init__(
        self,
        hidden_size: int,
        branch_count: int,
        *,
        fc_embedding: Any,
        fc_hidden: Any,
        layer: Qwen4DecoderLayer,
        final_residual: Any,
        lm_head: Any,
        eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_size <= 0 or branch_count <= 0:
            raise ValueError("MTP hidden and branch dimensions must be positive")
        if layer.mixer_kind != "qsa" or layer.ple is not None:
            raise ValueError("MTP requires one QSA layer without PLE")
        if (
            not isinstance(layer.mixer, Qwen4QSAAdapter)
            or type(layer.mixer.state_backend) is not Qwen4BF16QSAStateBackend
        ):
            raise ValueError("MTP head requires the functional QSA state backend")
        self.hidden_size = hidden_size
        self.branch_count = branch_count
        self.pre_fc_norm_embedding = Qwen4RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = Qwen4RMSNorm(hidden_size * branch_count, eps=eps)
        self.fc_embedding = fc_embedding
        self.fc_hidden = fc_hidden
        self.layers = [layer]
        self.final_residual = final_residual
        self.lm_head = lm_head
        self.graph_contract = MTP_GRAPH_CONTRACT
        self._state_owner = object()

    def fuse(self, widened: mx.array, token_embeddings: mx.array) -> mx.array:
        """Fuse aligned previous hidden rows and following token embeddings."""
        if (
            widened.ndim != 3
            or widened.shape[0] != 1
            or widened.shape[1] == 0
            or widened.shape[-1] != self.hidden_size * self.branch_count
            or token_embeddings.shape != (*widened.shape[:2], self.hidden_size)
            or widened.dtype not in (mx.bfloat16, mx.float16, mx.float32)
            or token_embeddings.dtype != widened.dtype
        ):
            raise ValueError("MTP fusion requires aligned single-request hidden and embedding rows")
        hidden = self.pre_fc_norm_hidden(widened).reshape(
            *widened.shape[:2], self.branch_count, self.hidden_size,
        )
        embedding = self.fc_embedding(self.pre_fc_norm_embedding(token_embeddings))
        return (self.fc_hidden(hidden) + embedding[..., None, :]).reshape(widened.shape)

    def __call__(
        self,
        widened: mx.array,
        token_embeddings: mx.array,
        *,
        state: Qwen4MTPState | None = None,
    ) -> Qwen4MTPOutput:
        """Append shifted text positions and return an uncommitted draft state.

        The functional backend never overwrites prior arrays. Rejection and
        failure can retain the earlier state without copying the full cache.
        This method does not publish target or drafter request frontiers.
        """
        attention, residual, injection = self._append_attention(widened, token_embeddings, state)
        layer = self.layers[0]
        hidden = gated_residual_write(residual, attention.output, injection)
        mixed, residual, injection = layer.mlp_residual(hidden)
        hidden = gated_residual_write(residual, layer.mlp(mixed), injection)
        collapsed = self.final_residual(hidden)
        if collapsed.shape != token_embeddings.shape:
            raise ValueError("MTP final residual returned an invalid shape")
        logits = self.lm_head(collapsed)
        if logits.ndim != 3 or logits.shape[:2] != widened.shape[:2]:
            raise ValueError("MTP language head returned an invalid shape")
        return Qwen4MTPOutput(
            logits=logits,
            widened=hidden,
            state=Qwen4MTPState(self._state_owner, attention.state),
        )

    def append_context(
        self, widened: mx.array, token_embeddings: mx.array, *, state: Qwen4MTPState | None = None,
    ) -> Qwen4MTPState:
        """Append target-authoritative pairs without running unused draft MoE/head.

        The draft has one attention layer. Its cache depends on the fusion and
        attention input, not the subsequent MoE, residual output or vocabulary
        projection. The returned state equals a full forward's attention state.
        """
        attention, _residual, _injection = self._append_attention(widened, token_embeddings, state)
        return Qwen4MTPState(self._state_owner, attention.state)

    def _append_attention(self, widened, token_embeddings, state):
        if state is not None and (
            not isinstance(state, Qwen4MTPState) or state.owner is not self._state_owner
        ):
            raise ValueError("MTP state belongs to another head")
        fused = self.fuse(widened, token_embeddings)
        offset = 0 if state is None else state.frontier
        count = widened.shape[1]
        positions = mx.broadcast_to(
            mx.arange(offset + 1, offset + count + 1, dtype=mx.int32)[None, None, :],
            (3, 1, count),
        )
        layer = self.layers[0]
        mixed, residual, injection = layer.attention_residual(fused)
        attention = layer.mixer.step_trusted(
            mixed,
            valid_tokens=mx.ones((1, count), dtype=mx.bool_),
            visible_history=mx.ones((1, offset + count), dtype=mx.bool_),
            position_ids=positions,
            state=None if state is None else state.attention,
        )
        if attention.frontier != offset + count or attention.output.shape != mixed.shape:
            raise ValueError("MTP attention returned an invalid frontier or output shape")
        return attention, residual, injection
