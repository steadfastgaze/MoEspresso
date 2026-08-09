"""Speculative decoding loop for the DeepSeek-V4 runtime.

Drives the target model and a drafter through the draft/verify cycle. Any
object satisfying the `Drafter` protocol can drive the loop;
`dspark_model.DSparkDraftModel` is one concrete implementation. The loop
is lossless: with temperature 0 a draft token is accepted only when it
equals the target argmax at that position, and with temperature > 0 the
standard speculative sampling rule (accept with probability
min(1, p_target/p_draft), resample the first rejection from the residual
distribution) recovers the target distribution exactly.

Round layout, with P the last target-forwarded position:
- the drafter proposes tokens for positions P+2 .. P+1+k from the anchor
  at P+1
- one target forward over [anchor, d_1 .. d_k] advances the cache to
  position P+1+k and yields per-position logits and tap rows
- the longest matching prefix is kept, the caches are rolled back to the
  accepted frontier from a pre-verify snapshot, and the corrected or
  bonus token becomes the next anchor

Draft prefetch: once a round's acceptance is decided and the accepted
rows are ingested, the next anchor and anchor position are fully
determined, and the next round's draft graph reads only the drafter
state, never the target caches. The loop therefore builds that graph at
the end of the round and dispatches it with `mx.async_eval`, so the
ingest and draft kernels execute while the host performs the round's
bookkeeping, commit callbacks, and the next round's cache snapshot. The
graph construction order over the drafter state is unchanged (ingest
then draft, with no drafter-state operation between them), so the
proposals are identical to the sequential schedule.

Rollback restores the pre-verify cache state bit-exactly for the rejected
positions (see `dspark_rollback`). Trimming cannot do that on the hybrid
caches: it clears partial-window pool buffers whose source tokens are
never re-fed, and the rotating local window stops being trimmable once it
wraps, at which point `trim_prompt_cache` silently trims nothing.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import mlx.core as mx

from moespresso.runtime.disk_kv import caches_all_at_offset

from .dspark_rollback import capture_verify_state, restore_verify_state


@dataclass
class DraftProposal:
    """One drafting round's proposal.

    `confidence` carries raw per-position confidence logits when the
    drafter has a confidence head and None otherwise; consumers apply
    the sigmoid.
    """

    tokens: mx.array          # (B, k) proposed tokens
    logits: mx.array          # (B, k, vocab) fp32 draft logits
    confidence: Optional[mx.array] = None  # (B, k) raw confidence logits


CapsuleScalar = str | int | float | bool | None


@dataclass(frozen=True)
class DrafterStateCapsule:
    """Portable state owned by a resumable drafter.

    The generic cache layers treat the tensor payload as opaque. The drafter
    records enough immutable scalar metadata to validate that payload before
    reconstructing its private state.
    """

    kind: str
    schema_major: int
    schema_minor: int
    frontier: int
    metadata: tuple[tuple[str, CapsuleScalar], ...]
    tensors: tuple[mx.array, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("drafter state capsule kind must be a non-empty string")
        for name, value in (
            ("schema_major", self.schema_major),
            ("schema_minor", self.schema_minor),
            ("frontier", self.frontier),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"drafter state capsule {name} must be a non-negative int")

        metadata_items = []
        keys = set()
        for item in self.metadata:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("drafter state capsule metadata entries must be key/value pairs")
            key, value = item
            if not isinstance(key, str) or not key:
                raise ValueError("drafter state capsule metadata keys must be non-empty strings")
            if key in keys:
                raise ValueError(f"duplicate drafter state capsule metadata key: {key!r}")
            keys.add(key)
            if value is not None and type(value) not in (str, int, float, bool):
                raise TypeError(
                    "drafter state capsule metadata values must be JSON scalar types"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("drafter state capsule float metadata must be finite")
            metadata_items.append((key, value))

        metadata = tuple(metadata_items)
        tensors = tuple(self.tensors)
        if any(not isinstance(tensor, mx.array) for tensor in tensors):
            raise TypeError("drafter state capsule tensors must be MLX arrays")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "tensors", tensors)

    @property
    def nbytes(self) -> int:
        """Return the byte size of the tensor payload."""
        return sum(int(tensor.nbytes) for tensor in self.tensors)

    def metadata_dict(self) -> dict[str, CapsuleScalar]:
        """Return a mutable copy of the immutable scalar metadata."""
        return dict(self.metadata)


class Drafter(Protocol):
    """Contract between `spec_generate` and a draft model.

    A drafter owns its per-generation state, consumes the target's tapped
    hidden rows and the committed token stream in order, and proposes a
    block of tokens after an anchor. `tap_layer_ids` names the target
    layers to tap and `tap_transform` shapes each tapped layer output at
    record time; the caller installs the hidden tap with exactly those.

    `greedy_only` is an optional attribute, read with a default of False.
    A drafter that proposes argmax-only tokens without a draft
    distribution over the target vocabulary sets it True; the loop then
    refuses temperature > 0 (sampled acceptance needs the proposing
    distribution) and the serve seam keeps such requests on the plain
    path.
    """

    greedy_only: bool = False

    @property
    def block_size(self) -> int:
        """Maximum number of tokens proposed per round."""
        ...

    @property
    def tap_layer_ids(self) -> Sequence[int]:
        """Target layer ids whose tapped outputs feed `ingest`."""
        ...

    def tap_transform(self, layer_id: int, out: mx.array) -> mx.array:
        """Map one tapped layer output (B, L, hc, D) to the recorded rows.

        Applied at record time inside the target forward. A drafter that
        conditions on the hyper-connection mean reduces here; a drafter
        that consumes the raw copies returns `out` unchanged.
        """
        ...

    def make_state(self) -> object:
        """Opaque per-generation draft state, for example window caches."""
        ...

    def ingest(
        self,
        state: object,
        rows: mx.array,
        positions: Sequence[int],
        token_ids: Sequence[int],
    ) -> None:
        """Feed tapped target rows at `positions` into `state`.

        `rows` carries the transformed tap output with `len(positions)`
        entries along axis 1 (multiple tapped layers are concatenated
        along the last axis). `token_ids` are host ints, one per
        position: the committed token at that position. Calls arrive in
        stream order with contiguous positions; the drafter decides what
        to retain.
        """
        ...

    def draft(
        self, state: object, anchor_token: int, anchor_pos: int, temperature: float
    ) -> DraftProposal:
        """Propose up to `block_size` tokens following the anchor.

        `state` must already cover positions up to `anchor_pos - 1`.
        """
        ...


class ResumableDrafter(Protocol):
    """Optional capability for exporting and restoring drafter state."""

    def state_frontier(self, state: object) -> int:
        """Return the exclusive token frontier represented by `state`."""
        ...

    def state_nbytes(self, state: object) -> int:
        """Return the live byte size of `state`'s tensor payload."""
        ...

    def export_state(self, state: object) -> DrafterStateCapsule:
        """Copy `state` into a detached capsule."""
        ...

    def import_state(self, capsule: DrafterStateCapsule) -> object:
        """Validate and copy a capsule into fresh drafter state."""
        ...


@dataclass(frozen=True)
class SpecPrefillProgress:
    """One materialized target and drafter frontier during prompt prefill.

    ``processed`` and ``total`` use the prompt-progress convention: both are
    relative to ``prompt_ids``, and ``total`` includes the final prompt token
    retained for the anchor-logits forward. ``frontier`` is the absolute
    exclusive cache position, including any restored ``prompt_offset``.

    The cache and drafter fields are live request-owned objects. A synchronous
    consumer may inspect or detach them before returning. This event does not
    export or serialize drafter state.
    """

    processed: int
    total: int
    frontier: int
    target_cache: object
    drafter_state: object
    state_owner: ResumableDrafter


class FixedSubmitDrafter:
    """Truncate every proposal to a fixed submit length.

    The inner chain still computes its full native block (the draft cost is
    unchanged); only the submitted length changes, which sets the
    verify-side L. With ``adaptive_cap=False`` and no confidence threshold
    this is the fixed-L schedule: the served schedule sweep on the DSpark
    chain measured fixed:3 at 28.666 tok/s against 28.263 (fixed:2), 26.343
    (fixed:4), and 27.621 (adaptive) at the long-prompt anchor, and the
    drafter-on certification ran fixed:3.
    """

    def __init__(self, inner: Drafter, submit_length: int):
        if int(submit_length) < 1:
            raise ValueError(
                f"fixed submit length must be >= 1, got {submit_length!r}")
        self._inner = inner
        self._submit_length = int(submit_length)
        self.greedy_only = getattr(inner, "greedy_only", False)

    @property
    def block_size(self) -> int:
        return min(self._submit_length, self._inner.block_size)

    @property
    def tap_layer_ids(self) -> Sequence[int]:
        return self._inner.tap_layer_ids

    def tap_transform(self, layer_id: int, out: mx.array) -> mx.array:
        return self._inner.tap_transform(layer_id, out)

    def make_state(self) -> object:
        return self._inner.make_state()

    def ingest(self, state, rows, positions, token_ids) -> None:
        return self._inner.ingest(state, rows, positions, token_ids)

    def draft(
        self, state: object, anchor_token: int, anchor_pos: int, temperature: float
    ) -> DraftProposal:
        proposal = self._inner.draft(state, anchor_token, anchor_pos, temperature)
        keep = self.block_size
        confidence = (None if proposal.confidence is None
                      else proposal.confidence[:, :keep])
        return DraftProposal(
            tokens=proposal.tokens[:, :keep],
            logits=proposal.logits[:, :keep],
            confidence=confidence,
        )


def strip_draft_skeleton(model) -> int:
    """Replace every constructed draft parameter with a broadcast placeholder.

    Module constructors fill the draft tree with lazy random-init arrays; at
    production dims the three-block DSpark skeleton is about 72 GiB of
    float32. The sidecar loaders replace every leaf through a strict
    `load_weights` before the hydration eval, so the skeleton normally stays
    unevaluated, but any evaluation that reaches the tree between
    construction and the strict load allocates the whole skeleton at once.
    Swapping each leaf for a zero-stride broadcast of a scalar drops the
    random-init graph unevaluated while keeping the names, shapes, and
    dtypes the strict load validates against; evaluating a placeholder
    allocates only its scalar donor.

    A placeholder cannot reach serving: the strict load fails on any
    parameter the sidecar does not cover, and a successful load replaces
    every leaf. Returns the number of replaced arrays.
    """
    count = 0

    def _placeholder(a: mx.array) -> mx.array:
        nonlocal count
        count += 1
        return mx.broadcast_to(mx.zeros((), dtype=a.dtype), a.shape)

    model.apply(_placeholder)
    return count


class HiddenTap:
    """Records per-layer target hidden rows for the draft conditioning.

    Installed as a class-level wrapper around the decoder layer call with a
    per-instance opt-in attribute, so untapped layers pay nothing. The
    drafter-supplied transform maps each tapped layer output (B, L, hc, D)
    to the recorded rows at record time.
    """

    def __init__(self, layer_ids: Sequence[int], transform):
        self.layer_ids = list(layer_ids)
        self.transform = transform
        self.active = False
        self.rows: Dict[int, mx.array] = {}

    def take_rows(self) -> mx.array:
        # Transformed rows in tap order, concatenated along the last axis
        # when several layers are tapped; clears the collected rows.
        parts = [self.rows[i] for i in self.layer_ids]
        self.rows = {}
        if len(parts) == 1:
            return parts[0]
        return mx.concatenate(parts, axis=-1)


def install_hidden_tap(model, layer_ids: Sequence[int], transform) -> HiddenTap:
    """Attach a tap to the target model's layers at `layer_ids`.

    `transform(layer_id, out)` shapes each tapped layer output at record
    time; drafters expose it as `Drafter.tap_transform`.
    """
    tap = HiddenTap(layer_ids, transform)
    layers = model.model.layers
    classes = set()
    for i in layer_ids:
        layers[i]._moespresso_hidden_tap = tap
        classes.add(type(layers[i]))
    for cls in classes:
        if getattr(cls, "_moespresso_hidden_tap_wrapped", False):
            continue
        orig = cls.__call__

        def wrapped(self, x, *a, _tap_orig=orig, **k):
            out = _tap_orig(self, x, *a, **k)
            t = getattr(self, "_moespresso_hidden_tap", None)
            if t is not None and t.active:
                t.rows[self.layer_id] = t.transform(self.layer_id, out)
            return out

        cls.__call__ = wrapped
        cls._moespresso_hidden_tap_wrapped = True
    return tap


@dataclass
class AcceptResult:
    accepted: int          # number of accepted draft tokens
    next_token: int        # corrected token (on rejection) or bonus token
    emitted: List[int]     # accepted draft tokens followed by next_token
    # Per offered position, whether the draft token equals the target
    # argmax at that position regardless of earlier positions. Greedy
    # acceptance fills this; sampled acceptance has no argmax notion and
    # leaves it None. Distinct from `accepted`: acceptance stops at the
    # first mismatch, while a later position can still match on its own.
    position_matches: Optional[List[bool]] = None


def _host_tokens(draft_tokens) -> List[int]:
    if isinstance(draft_tokens, mx.array):
        row = draft_tokens[0] if draft_tokens.ndim == 2 else draft_tokens
        return [int(t) for t in row]
    return list(draft_tokens)


def greedy_accept(draft_tokens, target_logits: mx.array) -> AcceptResult:
    """Deterministic acceptance: row k of target_logits predicts the token
    after verify input k, so draft token k must equal argmax(row k).
    Accepts draft tokens as host ints or as a device array; the device
    form is evaluated together with the argmax in one synchronization."""
    argmax = mx.argmax(target_logits[0], axis=-1)
    if isinstance(draft_tokens, mx.array):
        mx.eval(argmax, draft_tokens)
    else:
        mx.eval(argmax)
    draft_tokens = _host_tokens(draft_tokens)
    argmax_list = [int(t) for t in argmax]
    matches = [argmax_list[k] == token for k, token in enumerate(draft_tokens)]
    accepted = 0
    for hit in matches:
        if not hit:
            break
        accepted += 1
    next_token = argmax_list[accepted]
    emitted = list(draft_tokens[:accepted]) + [next_token]
    return AcceptResult(
        accepted=accepted,
        next_token=next_token,
        emitted=emitted,
        position_matches=matches,
    )


def sampled_accept(
    draft_tokens,
    draft_logits: mx.array,
    target_logits: mx.array,
    temperature: float,
) -> AcceptResult:
    """Speculative sampling acceptance preserving the target distribution.

    draft_logits: (1, k, vocab) fp32 draft logits.
    target_logits: (1, k+1, vocab) fp32 target logits for the verify block.
    """
    inv_t = 1.0 / max(temperature, 1e-5)
    p_t = mx.softmax(target_logits[0].astype(mx.float32) * inv_t, axis=-1)
    p_d = mx.softmax(draft_logits[0].astype(mx.float32) * inv_t, axis=-1)
    if isinstance(draft_tokens, mx.array):
        k = int(draft_tokens.shape[-1])
        uniforms = mx.random.uniform(shape=(k,))
        mx.eval(p_t, p_d, uniforms, draft_tokens)
    else:
        k = len(draft_tokens)
        uniforms = mx.random.uniform(shape=(k,))
        mx.eval(p_t, p_d, uniforms)
    draft_tokens = _host_tokens(draft_tokens)

    accepted = 0
    for i, token in enumerate(draft_tokens):
        ratio = float(p_t[i, token]) / max(float(p_d[i, token]), 1e-30)
        if float(uniforms[i]) < min(1.0, ratio):
            accepted += 1
        else:
            residual = mx.maximum(p_t[i] - p_d[i], 0.0)
            total = residual.sum()
            if float(total) <= 0.0:
                next_token = int(mx.argmax(p_t[i]))
            else:
                logp = mx.log(mx.maximum(residual / total, 1e-30))
                next_token = int(mx.random.categorical(logp))
            emitted = list(draft_tokens[:accepted]) + [next_token]
            return AcceptResult(accepted, next_token, emitted)

    bonus = int(sample_from_logits(target_logits[0, k][None], temperature)[0])
    emitted = list(draft_tokens) + [bonus]
    return AcceptResult(accepted, bonus, emitted)


def sample_from_logits(logits: mx.array, temperature: float) -> mx.array:
    """Sample one token per row from fp32 logits."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits * (1.0 / temperature))


def lm_head_logits_fp32(lm_head, h: mx.array) -> mx.array:
    """Float32 language-model head matmul, mirroring the target graph."""
    if hasattr(lm_head, "scales"):
        w = mx.dequantize(
            lm_head.weight,
            lm_head.scales,
            getattr(lm_head, "biases", None),
            group_size=lm_head.group_size,
            bits=lm_head.bits,
            mode=getattr(lm_head, "mode", "affine"),
        ).astype(mx.float32)
    else:
        w = lm_head.weight.astype(mx.float32)
    return h.astype(mx.float32) @ w.T


def make_fp32_logits_fn(lm_head):
    """Build a callable mapping hidden rows to float32 logits for every
    position, matching the served target's logits seam exactly.

    A K-quant head goes through the runtime's fp32 matmul one row at a
    time, taking the decode-shaped wire route per row; the multi-row
    scorer bridge is avoided because it dequantizes the full head per
    call. Other heads use the dequantized fp32 matmul.
    """
    if getattr(lm_head, "mode", None) == "kquant":
        import mlx_kquant as kq

        from .model import (
            _deepseek_v4_q8_affine_views,
            _kquant_matmul_ds4_fp32,
        )

        affine = None
        if lm_head.kquant_type == "q8_0":
            affine = lambda: _deepseek_v4_q8_affine_views(  # noqa: E731
                lm_head, mx=mx)

        def _kquant_call(h: mx.array) -> mx.array:
            return _kquant_matmul_ds4_fp32(
                h.astype(mx.float32),
                lm_head["weight"],
                lm_head["scales"],
                lm_head.kquant_type,
                mx=mx,
                kq=kq,
                affine=affine,
                wire_decode_site="lm_head",
                tiny_m_site="lm_head",
            )

        def _kquant_rows(h: mx.array) -> mx.array:
            # Single rows take the decode wire route; 2..8 rows take the
            # batched tiny multi-row affine route at the declared site.
            # Wider blocks fall back to per-row wire calls so the bridge
            # never materializes the full head for a verify shape.
            if h.shape[1] <= 8:
                return _kquant_call(h)
            rows = [_kquant_call(h[:, i : i + 1]) for i in range(h.shape[1])]
            return mx.concatenate(rows, axis=1)

        return _kquant_rows

    def _dense(h: mx.array) -> mx.array:
        return lm_head_logits_fp32(lm_head, h)

    return _dense


@dataclass
class SpecStats:
    """Per-generation speculative decoding statistics.

    Two per-position views coexist and answer different questions.
    `per_position_accept[i]` counts rounds where position i was accepted,
    which requires every earlier position accepted too: the cumulative
    prefix-survival curve that the engine's emitted-token count follows.
    `per_position_matched[i]` counts rounds where the draft token at
    position i equals the target argmax at that position regardless of
    earlier positions: the teacher-forced accuracy curve, the one
    comparable to a checkpoint's per-position validation accuracy. The
    cumulative curve is the product of conditional rates and decays
    steeply at depth even when the matched curve stays flat; comparing a
    cumulative curve against a validation accuracy curve misreads healthy
    deep positions as dead. Matched counts come from greedy rounds;
    sampled acceptance has no argmax notion and records no matches.
    """

    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0
    plain_fallbacks: int = 0
    per_position_accept: List[int] = field(default_factory=list)
    per_position_offered: List[int] = field(default_factory=list)
    per_position_matched: List[int] = field(default_factory=list)
    submit_length_counts: Dict[int, int] = field(default_factory=dict)

    def record(
        self,
        offered: int,
        accepted: int,
        matches: Optional[Sequence[bool]] = None,
    ) -> None:
        self.rounds += 1
        self.proposed += offered
        self.accepted += accepted
        while len(self.per_position_offered) < offered:
            self.per_position_offered.append(0)
            self.per_position_accept.append(0)
            self.per_position_matched.append(0)
        for i in range(offered):
            self.per_position_offered[i] += 1
            if i < accepted:
                self.per_position_accept[i] += 1
            if matches is not None and i < len(matches) and matches[i]:
                self.per_position_matched[i] += 1

    @property
    def mean_accepted_length(self) -> float:
        # Accepted draft tokens plus the bonus/corrected token per round.
        if not self.rounds:
            return 0.0
        return (self.accepted + self.rounds) / self.rounds


@dataclass
class SpecGeneration:
    tokens: List[int]
    stats: SpecStats
    target_cache: object
    drafter_state: object
    frontier: int


class RoundCostModel:
    """Online model of per-round wall cost by submitted draft length.

    Keeps an exponential moving average per length observed in this
    generation, with a prior anchored on the measured plain decode step so
    the model adapts to the host and package without baked constants. The
    prior shape reflects the multi-token forward's structure: a fixed cost
    for entering the L>1 path plus a small marginal cost per draft token;
    both are replaced by observations after the first round at a length.
    Length 0 is a plain decode step taken instead of a speculative round
    (the draft cost is still paid, so its prior sits above one step).
    """

    _PRIOR_FIXED = 2.0
    _PRIOR_MARGINAL = 0.2
    _PRIOR_FALLBACK = 1.3
    _ALPHA = 0.25

    def __init__(self, plain_step_ms: float):
        self.plain_step_ms = max(plain_step_ms, 1e-3)
        self._ema: Dict[int, float] = {}

    def _marginal_ms(self) -> float:
        # Least-squares slope over observed speculative lengths once two
        # or more are seen; the hardware-ratio prior otherwise. The slope
        # is clamped positive so extrapolation stays monotone.
        pts = [(j, v) for j, v in self._ema.items() if j >= 1]
        if len(pts) >= 2:
            n = len(pts)
            mean_j = sum(j for j, _ in pts) / n
            mean_v = sum(v for _, v in pts) / n
            denom = sum((j - mean_j) ** 2 for j, _ in pts)
            if denom > 0:
                slope = (
                    sum((j - mean_j) * (v - mean_v) for j, v in pts) / denom
                )
                return max(slope, 0.01 * self.plain_step_ms)
        return self._PRIOR_MARGINAL * self.plain_step_ms

    def expected_ms(self, submitted: int) -> float:
        if submitted in self._ema:
            return self._ema[submitted]
        if submitted == 0:
            return self.plain_step_ms * self._PRIOR_FALLBACK
        marginal = self._marginal_ms()
        pts = [(j, v) for j, v in self._ema.items() if j >= 1]
        if pts:
            fixed = sum(v - marginal * j for j, v in pts) / len(pts)
        else:
            fixed = self._PRIOR_FIXED * self.plain_step_ms
        return max(fixed, 0.0) + marginal * submitted

    def observe(self, submitted: int, wall_ms: float) -> None:
        prev = self._ema.get(submitted)
        if prev is None:
            self._ema[submitted] = wall_ms
        else:
            self._ema[submitted] = prev + self._ALPHA * (wall_ms - prev)


class OnlineCalibrator:
    """Online calibration of the drafter's confidence against observed
    acceptance, with exploration so estimates never freeze.

    A confidence head's raw sigmoid levels are unreliable as absolute
    probabilities (measured pessimistic by roughly two to one on content
    with high real acceptance), while its per-round ranking is useful.
    This keeps a per-position exponential moving average of the observed
    conditional acceptance and of the drafter's prediction, and rescales
    each round's predictions by the observed-to-predicted ratio. A
    drafter without a confidence head reports a neutral prediction of
    1.0 per position, so the calibrated value reduces to the observed
    acceptance ratio. The first `warmup` rounds and every
    `probe_every`-th round submit the full block, so the acceptance
    statistics keep flowing even when the scheduler would otherwise
    shorten or skip rounds.
    """

    _ALPHA = 0.15

    def __init__(self, block_size: int, warmup: int = 8, probe_every: int = 8):
        self.block_size = block_size
        self.warmup = warmup
        self.probe_every = probe_every
        self.rounds = 0
        self._observed: List[Optional[float]] = [None] * block_size
        self._predicted: List[Optional[float]] = [None] * block_size

    def forced_length(self) -> Optional[int]:
        """Length to force this round, or None to let the chooser decide.

        Warmup rounds submit the full block. Periodic probes alternate
        between the full block, which refreshes the tail acceptance
        statistics, and the half block, which keeps the cost model
        supplied with a second observed length for its slope fit.
        """
        if self.rounds < self.warmup:
            return self.block_size
        if self.probe_every > 0 and (self.rounds % self.probe_every == 0):
            phase = self.rounds // self.probe_every
            if phase % 2 == 0:
                return self.block_size
            return min(self.block_size, max(2, self.block_size // 2))
        return None

    def force_full_block(self) -> bool:
        return self.forced_length() == self.block_size

    def calibrated_survival(self, conf: Sequence[float]) -> List[float]:
        survival: List[float] = []
        running = 1.0
        for k, c in enumerate(conf):
            ratio = 1.0
            if self._observed[k] is not None and self._predicted[k]:
                ratio = self._observed[k] / max(self._predicted[k], 1e-3)
            running *= min(max(c * ratio, 0.0), 1.0)
            survival.append(running)
        return survival

    def record(self, conf: Sequence[float], submitted: int, accepted: int) -> None:
        self.rounds += 1
        for k in range(submitted):
            reached = k <= accepted
            if not reached:
                break
            hit = 1.0 if k < accepted else 0.0
            prev_o = self._observed[k]
            self._observed[k] = (
                hit if prev_o is None else prev_o + self._ALPHA * (hit - prev_o)
            )
            prev_p = self._predicted[k]
            self._predicted[k] = (
                conf[k]
                if prev_p is None
                else prev_p + self._ALPHA * (conf[k] - prev_p)
            )


def choose_submit_length(
    survival: Sequence[float], cost: RoundCostModel
) -> int:
    """Pick the verify length maximizing expected emitted tokens per
    millisecond, against the plain-step fallback at length 0.

    `survival[j]` is the probability that draft token j+1 is accepted
    (cumulative product of the calibrated per-position confidences).
    Every speculative round also emits the corrected or bonus token.
    """
    best_j = 0
    best_rate = 1.0 / cost.expected_ms(0)
    expected = 1.0
    for j, prob in enumerate(survival, start=1):
        expected += prob
        rate = expected / cost.expected_ms(j)
        if rate > best_rate:
            best_j, best_rate = j, rate
    return best_j


def _dispatch_proposal(proposal: DraftProposal) -> None:
    """Dispatch a proposal's device graph without blocking the host.

    Covers every array the loop later evaluates from the proposal, so the
    ingest and draft kernels behind them start executing while the host
    performs round bookkeeping.
    """
    arrays = [proposal.tokens, proposal.logits]
    if proposal.confidence is not None:
        arrays.append(proposal.confidence)
    mx.async_eval(*arrays)


def _drafter_state_arrays(state: object) -> list[mx.array]:
    """Return live arrays that must finish before a paired prefill callback.

    Resumable DSpark state is a sequence of window objects whose ``kv`` fields
    hold the complete tensor state. The small recursive cases also keep the
    progress seam usable by protocol fakes without coupling it to a concrete
    DSpark class or invoking the capsule export path.
    """
    if isinstance(state, mx.array):
        return [state]
    if isinstance(state, dict):
        arrays: list[mx.array] = []
        for value in state.values():
            arrays.extend(_drafter_state_arrays(value))
        return arrays
    if isinstance(state, (list, tuple)):
        arrays = []
        for value in state:
            arrays.extend(_drafter_state_arrays(value))
        return arrays
    kv = getattr(state, "kv", None)
    if isinstance(kv, mx.array):
        return [kv]
    return []


def _confidence_keep(confidence_row: mx.array, threshold: float) -> int:
    """Length of the leading prefix with sigmoid(confidence) >= threshold."""
    if threshold <= 0.0:
        return confidence_row.shape[0]
    probs = mx.sigmoid(confidence_row.astype(mx.float32))
    mx.eval(probs)
    keep = 0
    for i in range(probs.shape[0]):
        if float(probs[i]) < threshold:
            break
        keep += 1
    return keep


def spec_generate(
    model,
    drafter: Drafter,
    tap: HiddenTap,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    eos_ids: Optional[Sequence[int]] = None,
    confidence_threshold: float = 0.0,
    prefill_step_size: int = 2048,
    prefill_plan: Optional[Sequence[int]] = None,
    target_cache: Optional[list] = None,
    adaptive_cap: bool = True,
    cost_model: Optional[RoundCostModel] = None,
    calibration_warmup: int = 8,
    probe_every: int = 8,
    on_commit: Optional[Callable[[Sequence[int]], None]] = None,
    prefill_progress_callback: Optional[
        Callable[[SpecPrefillProgress], None]
    ] = None,
    prefill_progress_frontiers: Optional[Sequence[int]] = None,
    draft_prefetch: bool = True,
    drafter_state: object | None = None,
    prompt_offset: int = 0,
    state_owner: ResumableDrafter | None = None,
) -> SpecGeneration:
    """Generate with speculative decoding.

    `model` is the loaded DeepSeek-V4 target graph, `drafter` a `Drafter`
    implementation over the target's token space, `tap` the hidden tap
    installed with the drafter's tap layers and transform.

    `on_commit` observes every committed token run in stream order: the
    anchor produced by prefill, then each round's accepted and corrected
    tokens (a terminal stop token included). The serve seam drives its
    incremental detokenization and streaming callbacks from it.

    With `adaptive_cap` the verify length is chosen per round from the
    drafter's survival probabilities and the online round-cost model,
    including a plain decode step as the floor when speculation is not
    expected to pay. A proposal without confidence contributes a neutral
    per-position confidence of 1.0, so the calibrated survival reduces to
    the observed acceptance ratios. `confidence_threshold` applies only
    to the fixed schedule and only when the proposal carries confidence.
    `cost_model` overrides the online model (tests).

    With `draft_prefetch` the loop builds and asynchronously dispatches
    the next round's draft at the end of each round, overlapping the
    ingest and draft kernels with the host-side round tail (see the
    module docstring); the proposals are identical to the sequential
    schedule, which `draft_prefetch=False` restores.

    `prompt_offset` is the absolute exclusive frontier already represented
    by `target_cache` and `drafter_state`; `prompt_ids` is the non-empty
    suffix beginning there. A resumable `state_owner` may differ from the
    schedule wrapper that drives `drafter`. Its frontier is checked before
    any model call and again before return. This lets a fixed-submit wrapper
    control proposals without claiming ownership of DSpark's portable state.

    ``prefill_plan`` contains variable chunk sizes for a leading part of the
    prompt suffix. The uniform ``prefill_step_size`` covers the remaining
    prefill span, and the final prompt token is always retained for the anchor
    forward. When ``prefill_progress_callback`` is present, each completed
    prefill chunk materializes and validates the target and raw drafter state
    at the same absolute frontier before publishing one
    :class:`SpecPrefillProgress` event. Verify and decode forwards never publish
    these events. ``prefill_progress_frontiers`` optionally limits publication
    and drafter-state materialization to an explicit set of absolute chunk-end
    frontiers.
    """
    if temperature > 0 and getattr(drafter, "greedy_only", False):
        raise ValueError(
            "the drafter is greedy-only: sampled acceptance needs the draft "
            "distribution and temperature > 0 requests must take the plain "
            "path")
    eos = set(eos_ids or [])
    stats = SpecStats()

    if isinstance(prompt_offset, bool) or not isinstance(prompt_offset, int) or prompt_offset < 0:
        raise ValueError("prompt_offset must be a non-negative int")
    if target_cache is None:
        if prompt_offset != 0:
            raise ValueError("a nonzero prompt_offset requires a target cache")
        cache = model.make_cache()
    else:
        cache = target_cache
        if not caches_all_at_offset(cache, prompt_offset):
            raise ValueError("target cache positional offsets do not match prompt_offset")

    if drafter_state is None:
        if prompt_offset != 0:
            raise ValueError("a nonzero prompt_offset requires drafter state")
        state = drafter.make_state()
    else:
        state = drafter_state

    frontier_owner = state_owner
    if frontier_owner is None and callable(getattr(drafter, "state_frontier", None)):
        frontier_owner = drafter  # type: ignore[assignment]
    if prefill_progress_callback is not None and frontier_owner is None:
        raise ValueError(
            "a speculative prefill progress callback requires a resumable "
            "drafter state owner"
        )
    if frontier_owner is not None:
        state_frontier = int(frontier_owner.state_frontier(state))
        if state_frontier != prompt_offset:
            raise ValueError(
                f"drafter state frontier {state_frontier} does not match "
                f"prompt_offset {prompt_offset}"
            )
    # The verify pass reads hidden states from the inner graph and applies
    # the fp32 head itself: the served kquant head patch slices multi-token
    # forwards to the newest position, and verification needs every row.
    verify_logits_fn = make_fp32_logits_fn(model.lm_head)

    prompt_list = [int(t) for t in prompt_ids]
    prompt = mx.array(prompt_list, dtype=mx.int64)[None]
    n_prompt = prompt.shape[1]
    if n_prompt < 1:
        raise ValueError("empty prompt")
    if isinstance(prefill_step_size, bool) or int(prefill_step_size) < 1:
        raise ValueError("prefill_step_size must be a positive int")
    prefill_step_size = int(prefill_step_size)

    planned_chunks: list[int] = []
    if prefill_plan is not None:
        for raw_size in prefill_plan:
            if isinstance(raw_size, bool) or not isinstance(raw_size, int):
                raise ValueError("prefill plan chunk sizes must be positive ints")
            if raw_size < 1:
                raise ValueError("prefill plan chunk sizes must be positive ints")
            planned_chunks.append(raw_size)
    prefill_tokens = n_prompt - 1
    planned_tokens = sum(planned_chunks)
    if planned_tokens > prefill_tokens:
        raise ValueError(
            f"prefill plan covers {planned_tokens} tokens; at most "
            f"{prefill_tokens} of the {n_prompt}-token prompt may be prefilled"
        )
    position = planned_tokens
    while position < prefill_tokens:
        size = min(prefill_step_size, prefill_tokens - position)
        planned_chunks.append(size)
        position += size

    progress_frontiers = None
    if prefill_progress_frontiers is not None:
        if prefill_progress_callback is None:
            raise ValueError(
                "prefill progress frontiers require a progress callback"
            )
        normalized_frontiers: list[int] = []
        for raw_frontier in prefill_progress_frontiers:
            if isinstance(raw_frontier, bool) or not isinstance(raw_frontier, int):
                raise ValueError("prefill progress frontiers must be ints")
            normalized_frontiers.append(raw_frontier)
        if not normalized_frontiers:
            raise ValueError("prefill progress frontiers must not be empty")
        if normalized_frontiers != sorted(set(normalized_frontiers)):
            raise ValueError(
                "prefill progress frontiers must be unique and ascending"
            )
        chunk_frontiers = {
            prompt_offset + stop
            for stop in itertools.accumulate(planned_chunks)
        }
        invalid = [
            frontier
            for frontier in normalized_frontiers
            if frontier not in chunk_frontiers
        ]
        if invalid:
            raise ValueError(
                "prefill progress frontiers must name prefill chunk ends: "
                f"{invalid}"
            )
        progress_frontiers = frozenset(normalized_frontiers)

    # Prefill all but the final prompt token in chunks, then run the final
    # token alone so only its logits row is ever materialized. Each chunk's
    # tap rows go straight to the drafter, whose state decides what to
    # retain (a sliding-window drafter keeps only the trailing rows).
    tap.active = True
    start = 0
    for size in planned_chunks:
        stop = start + size
        model(prompt[:, start:stop], cache=cache)
        drafter.ingest(
            state,
            tap.take_rows(),
            list(range(prompt_offset + start, prompt_offset + stop)),
            prompt_list[start:stop],
        )
        target_states = [c.state for c in cache if c is not None]
        frontier = prompt_offset + stop
        publish_progress = (
            prefill_progress_callback is not None
            and (
                progress_frontiers is None
                or frontier in progress_frontiers
            )
        )
        if not publish_progress:
            mx.eval(*target_states)
        else:
            assert frontier_owner is not None
            mx.eval(*target_states, *_drafter_state_arrays(state))
            if not caches_all_at_offset(cache, frontier):
                raise RuntimeError(
                    "target cache positional offsets disagree at speculative "
                    f"prefill frontier {frontier}"
                )
            state_frontier = int(frontier_owner.state_frontier(state))
            if state_frontier != frontier:
                raise RuntimeError(
                    f"drafter state frontier {state_frontier} does not match "
                    f"target prefill frontier {frontier}"
                )
            prefill_progress_callback(
                SpecPrefillProgress(
                    processed=stop,
                    total=n_prompt,
                    frontier=frontier,
                    target_cache=cache,
                    drafter_state=state,
                    state_owner=frontier_owner,
                )
            )
        start = stop
    # The final prompt token runs alone and is timed as the plain-step seed
    # for the round-cost model.
    t0 = time.perf_counter()
    logits = model(prompt[:, -1:], cache=cache)
    anchor_logits = logits[0, -1][None].astype(mx.float32)
    anchor_tok = sample_from_logits(anchor_logits, temperature)
    mx.eval(anchor_tok)
    plain_seed_ms = (time.perf_counter() - t0) * 1000.0
    drafter.ingest(
        state,
        tap.take_rows(),
        [prompt_offset + n_prompt - 1],
        [prompt_list[-1]],
    )

    if cost_model is None:
        cost_model = RoundCostModel(plain_seed_ms)
    block_size = drafter.block_size
    calibrator = OnlineCalibrator(
        block_size, warmup=calibration_warmup, probe_every=probe_every
    )
    anchor = int(anchor_tok[0])
    emitted: List[int] = [anchor]
    if on_commit is not None:
        on_commit([anchor])
    last_pos = prompt_offset + n_prompt - 1
    # Prefetched (proposal, anchor, anchor_pos) from the previous round's
    # tail; consumed only when it targets exactly this round's anchor.
    prefetched: Optional[tuple] = None

    def prefetch_next() -> Optional[tuple]:
        """Build and dispatch the next round's draft when the loop will
        continue from the just-committed frontier."""
        if not draft_prefetch:
            return None
        if len(emitted) >= max_new_tokens or emitted[-1] in eos:
            return None
        nxt = drafter.draft(state, emitted[-1], last_pos + 1, temperature)
        _dispatch_proposal(nxt)
        return (nxt, emitted[-1], last_pos + 1)

    while len(emitted) < max_new_tokens and anchor not in eos:
        t_round = time.perf_counter()
        if prefetched is not None and prefetched[1:] == (anchor, last_pos + 1):
            proposal = prefetched[0]
        else:
            proposal = drafter.draft(state, anchor, last_pos + 1, temperature)
        prefetched = None

        conf_list: List[float] = []
        if adaptive_cap:
            if proposal.confidence is None:
                conf_list = [1.0] * int(proposal.tokens.shape[1])
            else:
                conf = mx.sigmoid(proposal.confidence[0].astype(mx.float32))
                mx.eval(conf, proposal.tokens)
                conf_list = [float(conf[i]) for i in range(conf.shape[0])]
            forced = calibrator.forced_length()
            if forced is not None:
                offered = forced
            else:
                offered = choose_submit_length(
                    calibrator.calibrated_survival(conf_list), cost_model
                )
        elif confidence_threshold > 0.0 and proposal.confidence is not None:
            offered = max(
                _confidence_keep(proposal.confidence[0], confidence_threshold), 1
            )
        else:
            offered = int(proposal.tokens.shape[1])

        if offered == 0:
            # Speculation is not expected to pay this round: one plain
            # decode step through the production path. The tap still
            # records the anchor's hidden row for the draft state.
            step_logits = model(
                mx.array([[anchor]], dtype=mx.int64), cache=cache
            )
            next_tok = sample_from_logits(
                step_logits[0, -1][None].astype(mx.float32), temperature
            )
            mx.eval(next_tok)
            drafter.ingest(state, tap.take_rows(), [last_pos + 1], [anchor])
            last_pos += 1
            anchor = int(next_tok[0])
            emitted.append(anchor)
            prefetched = prefetch_next()
            if on_commit is not None:
                on_commit([anchor])
            stats.rounds += 1
            stats.plain_fallbacks += 1
            stats.submit_length_counts[0] = (
                stats.submit_length_counts.get(0, 0) + 1
            )
            stats.emitted = len(emitted)
            cost_model.observe(
                0, (time.perf_counter() - t_round) * 1000.0
            )
            calibrator.record(conf_list, 0, 0)
            continue

        tokens_offered = proposal.tokens[:, :offered]
        anchor_arr = mx.array([[anchor]], dtype=proposal.tokens.dtype)
        verify_ids = mx.concatenate([anchor_arr, tokens_offered], axis=1)
        snapshot = capture_verify_state(cache, 1 + offered)
        verify_hidden = model.model(verify_ids, cache=cache)
        target_logits = verify_logits_fn(verify_hidden).astype(mx.float32)
        verify_rows = tap.take_rows()

        if temperature <= 0:
            result = greedy_accept(tokens_offered, target_logits)
        else:
            result = sampled_accept(
                tokens_offered,
                proposal.logits[:, :offered],
                target_logits,
                temperature,
            )
        stats.record(offered, result.accepted, result.position_matches)
        stats.submit_length_counts[offered] = (
            stats.submit_length_counts.get(offered, 0) + 1
        )

        # An EOS token or the request budget can stop inside the accepted
        # draft prefix. Only accepted tokens that become public may remain
        # in the target and drafter states.
        committed = []
        remaining = max_new_tokens - len(emitted)
        for token in result.emitted[:remaining]:
            committed.append(token)
            if token in eos:
                break
        accepted_committed = min(result.accepted, len(committed))

        # The kept verify rows carry the committed tokens at their
        # positions: the anchor followed by the public accepted tokens.
        keep = 1 + accepted_committed
        restore_verify_state(cache, snapshot, keep)
        drafter.ingest(
            state,
            verify_rows[:, :keep],
            list(range(last_pos + 1, last_pos + 1 + keep)),
            [anchor] + committed[:accepted_committed],
        )
        last_pos += keep

        committed_from = len(emitted)
        emitted.extend(committed)
        # Ingest is enqueued and the committed frontier is final, so the
        # next round's draft can dispatch here; the host-side tail below
        # then overlaps its device work.
        prefetched = prefetch_next()
        cost_model.observe(offered, (time.perf_counter() - t_round) * 1000.0)
        if conf_list:
            calibrator.record(conf_list, offered, result.accepted)
        if on_commit is not None:
            on_commit(emitted[committed_from:])
        anchor = emitted[-1]
        if emitted[-1] in eos:
            break
        stats.emitted = len(emitted)

    tap.active = False
    stats.emitted = len(emitted)
    frontier = last_pos + 1
    if not caches_all_at_offset(cache, frontier):
        raise RuntimeError("target cache positional offsets disagree at speculative return")
    if frontier_owner is not None:
        state_frontier = int(frontier_owner.state_frontier(state))
        if state_frontier != frontier:
            raise RuntimeError(
                f"drafter state frontier {state_frontier} does not match "
                f"target frontier {frontier} at speculative return"
            )
    return SpecGeneration(
        tokens=emitted,
        stats=stats,
        target_cache=cache,
        drafter_state=state,
        frontier=frontier,
    )
